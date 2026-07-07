"""Tests for the RevIN σ blend weight (`revin_blend_weight`).

Linearly blends the per-window σ_cond_w toward the per-stock long-run σ
(= 1.0 after the upstream per-stock standardization the SP500 pipeline
applies):

    σ_w_blended = w · σ_cond_w + (1 − w) · 1.0

The blend is applied BEFORE `revin_sigma_correction`, so the two compose
multiplicatively as σ_w_final = (w·σ + (1−w))·α.

Tests cover:
1. Default w=1.0 → no change (pure 20-day estimator, backward-compat).
2. w=0.0 → σ_w is the constant 1.0 everywhere (RevIN scale is a no-op).
3. w=0.5 → σ_w is exactly the midpoint between the raw σ and 1.0.
4. Round-trip identity ``denorm(norm(x)) == x`` holds for any (w, α)
   because the same σ_w is used on both sides.
5. Blend and σ correction COMPOSE: σ_w_final = (w·σ + (1−w))·α.
6. Buffer is non-persistent (mirrors `revin_sigma_correction`).
7. DDIM inheritance (DDIM picks up the new buffer via super().__init__).
"""
from __future__ import annotations

import pytest
import torch

from tests.diffusion.test_revin import _make_ddpm, _make_ddim, _make_batch


def _raw_sigma(ddpm, batch):
    """Compute σ_cond_w from raw cond data, exactly as the model would
    without any blend or correction. Used as a ground-truth reference."""
    B, N = batch.num_graphs, batch.y.size(0) // batch.num_graphs
    T_obs, F_obs = batch.x.size(1), batch.x.size(2)
    ut = batch.x.view(B, N, T_obs, F_obs).swapaxes(1, 2)
    target_idx = ddpm._resolve_revin_target_idx(batch)
    cond_target = ut[:, :, :, target_idx]
    sigma_raw = cond_target.std(dim=1, unbiased=False, keepdim=True).unsqueeze(-1)
    return sigma_raw.clamp(min=ddpm.revin_eps), ut, target_idx


def test_default_blend_weight_is_one_no_change():
    """Without the kwarg, w defaults to 1.0 → σ_w is unchanged from the
    raw per-window estimator."""
    ddpm = _make_ddpm(revin=True)
    assert float(ddpm.revin_blend_weight) == 1.0

    batch = _make_batch(B=4)
    raw, ut, target_idx = _raw_sigma(ddpm, batch)
    _, sigma_w = ddpm._compute_revin_stats(ut, target_idx)

    torch.testing.assert_close(sigma_w, raw, atol=1e-6, rtol=1e-6)


def test_blend_weight_zero_gives_constant_one():
    """w=0 → σ_w_blended = 1.0 everywhere, regardless of input variability."""
    ddpm = _make_ddpm(revin=True, revin_blend_weight=0.0)
    batch = _make_batch(B=4)
    _, ut, target_idx = _raw_sigma(ddpm, batch)
    _, sigma_w = ddpm._compute_revin_stats(ut, target_idx)
    expected = torch.ones_like(sigma_w)
    torch.testing.assert_close(sigma_w, expected, atol=1e-6, rtol=1e-6)


def test_blend_weight_half_is_midpoint():
    """w=0.5 → σ_w_blended = 0.5 · σ_raw + 0.5 · 1.0 exactly."""
    ddpm = _make_ddpm(revin=True, revin_blend_weight=0.5)
    batch = _make_batch(B=4)
    raw, ut, target_idx = _raw_sigma(ddpm, batch)
    _, sigma_w = ddpm._compute_revin_stats(ut, target_idx)
    expected = 0.5 * raw + 0.5
    torch.testing.assert_close(sigma_w, expected, atol=1e-6, rtol=1e-6)


def test_roundtrip_identity_holds_for_any_blend_weight():
    """For ANY (w, α), denorm(norm(x)) == x — the same σ_w is used on
    both sides, so blend+correction cancels in the round trip.
    """
    for w in (0.0, 0.3, 0.5, 0.7, 1.0):
        for alpha in (1.0, 1.22):
            ddpm = _make_ddpm(
                revin=True,
                revin_blend_weight=w,
                revin_sigma_correction=alpha,
            )
            x = torch.randn(4, 5, 8, 1)
            mu = torch.randn(4, 1, 8, 1)
            sigma = torch.rand(4, 1, 8, 1).clamp(min=1e-5)
            x_norm = ddpm._revin_normalize(x, mu, sigma)
            x_recovered = ddpm._revin_denormalize(x_norm, mu, sigma)
            torch.testing.assert_close(
                x_recovered, x, atol=1e-5, rtol=1e-5,
                msg=f"Round-trip failed at w={w}, α={alpha}",
            )


def test_blend_and_correction_compose_multiplicatively():
    """σ_w_final = (w·σ_raw + (1−w)) · α — verify the composition order.

    This is the algebraic identity that makes the residual α deterministic
    given w: pick w, then set α = 1 / E[w·σ_raw + (1−w)].
    """
    W = 0.5
    ALPHA = 1.10
    ddpm = _make_ddpm(
        revin=True,
        revin_blend_weight=W,
        revin_sigma_correction=ALPHA,
    )
    batch = _make_batch(B=4)
    raw, ut, target_idx = _raw_sigma(ddpm, batch)
    _, sigma_w = ddpm._compute_revin_stats(ut, target_idx)

    expected = (W * raw + (1.0 - W)) * ALPHA
    torch.testing.assert_close(sigma_w, expected, atol=1e-5, rtol=1e-5)


def test_blend_weight_is_non_persistent_buffer():
    """Mirrors revin_sigma_correction — config-controlled, not in state_dict."""
    ddpm = _make_ddpm(revin=True, revin_blend_weight=0.7)
    sd = ddpm.state_dict()
    assert "revin_blend_weight" not in sd, (
        "revin_blend_weight must NOT appear in state_dict — it's a "
        "non-persistent buffer driven by config."
    )
    # Attribute still present and applies
    assert float(ddpm.revin_blend_weight) == pytest.approx(0.7, rel=1e-6)


def test_old_checkpoint_loads_strict_with_new_blend_weight():
    """A state_dict from a pre-blend model must load strict=True into a
    model built with a non-default w."""
    ddpm_old = _make_ddpm(revin=True)  # default w=1.0
    sd_old = ddpm_old.state_dict()
    assert "revin_blend_weight" not in sd_old

    ddpm_new = _make_ddpm(revin=True, revin_blend_weight=0.5)
    missing, unexpected = ddpm_new.load_state_dict(sd_old, strict=True)
    assert missing == [] and unexpected == []
    assert float(ddpm_new.revin_blend_weight) == pytest.approx(0.5, rel=1e-6)


def test_ddim_inherits_blend_weight():
    """DDIM picks up the new buffer through super().__init__."""
    ddim = _make_ddim(revin=True, revin_blend_weight=0.3)
    assert float(ddim.revin_blend_weight) == pytest.approx(0.3, rel=1e-6)
    batch = _make_batch(B=4)
    raw, ut, target_idx = _raw_sigma(ddim, batch)
    _, sigma_w = ddim._compute_revin_stats(ut, target_idx)
    expected = 0.3 * raw + 0.7
    torch.testing.assert_close(sigma_w, expected, atol=1e-6, rtol=1e-6)
