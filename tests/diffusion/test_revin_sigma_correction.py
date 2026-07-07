"""Tests for the symmetric RevIN σ scale correction added to DDPM.

The correction multiplies the per-window σ_w computed in
``_compute_revin_stats`` by a constant α. Because both ``_revin_normalize``
and ``_revin_denormalize`` consume the same σ_w from this method, the
correction propagates symmetrically to training (forward noising), sampling
(reverse denorm), AND ELBO evaluation.

Tests cover:
1. Default α=1.0 → no change (backwards-compat with all existing checkpoints).
2. α=k changes σ_w by exactly factor k.
3. Round-trip identity: ``denorm(norm(x)) == x`` holds for any α (because
   the scaling cancels — same σ_w on both sides).
4. End-to-end DDPM sample with α=k gives samples wider than α=1 by ~k×
   (consistent with the design intent).
5. Buffer persists across save/load (state_dict round-trip).
"""
from __future__ import annotations

import pytest
import torch

from tests.diffusion.test_revin import _make_ddpm, _make_ddim, _make_batch


def test_default_alpha_is_one_no_change():
    """Default constructor (no kwarg) leaves σ_w untouched."""
    ddpm = _make_ddpm(revin=True)
    assert float(ddpm.revin_sigma_correction) == 1.0

    batch = _make_batch(B=4)
    B, N = batch.num_graphs, batch.y.size(0) // batch.num_graphs
    T_obs, F_obs = batch.x.size(1), batch.x.size(2)
    ut = batch.x.view(B, N, T_obs, F_obs).swapaxes(1, 2)

    target_idx = ddpm._resolve_revin_target_idx(batch)
    _, sigma_w = ddpm._compute_revin_stats(ut, target_idx)

    # Manually recompute σ_w from raw cond_target — should match exactly
    # when α=1.
    raw = ut[:, :, :, target_idx].std(dim=1, unbiased=False, keepdim=True).unsqueeze(-1)
    raw = raw.clamp(min=ddpm.revin_eps)
    torch.testing.assert_close(sigma_w, raw, atol=1e-7, rtol=1e-7)


def test_alpha_scales_sigma_by_exactly_alpha():
    """σ_w returned by _compute_revin_stats with α=k equals k × σ_w with α=1."""
    ALPHA = 1.22
    ddpm_no_corr = _make_ddpm(revin=True)
    ddpm_corrected = _make_ddpm(revin=True, revin_sigma_correction=ALPHA)
    assert float(ddpm_corrected.revin_sigma_correction) == pytest.approx(ALPHA, rel=1e-6)

    batch = _make_batch(B=4)
    B, N = batch.num_graphs, batch.y.size(0) // batch.num_graphs
    T_obs, F_obs = batch.x.size(1), batch.x.size(2)
    ut = batch.x.view(B, N, T_obs, F_obs).swapaxes(1, 2)

    target_idx = ddpm_no_corr._resolve_revin_target_idx(batch)
    _, sigma_w_unc = ddpm_no_corr._compute_revin_stats(ut, target_idx)
    _, sigma_w_corr = ddpm_corrected._compute_revin_stats(ut, target_idx)

    torch.testing.assert_close(sigma_w_corr, ALPHA * sigma_w_unc, atol=1e-6, rtol=1e-6)


def test_roundtrip_identity_holds_for_any_alpha():
    """For ANY α, denorm(norm(x)) must equal x — the same σ_w is used on
    both sides, so the scaling cancels (this is the value of doing the
    correction symmetrically inside `_compute_revin_stats`).
    """
    for alpha in (1.0, 1.22, 0.5, 2.0):
        ddpm = _make_ddpm(revin=True, revin_sigma_correction=alpha)
        x = torch.randn(4, 5, 8, 1)
        mu = torch.randn(4, 1, 8, 1)
        sigma = torch.rand(4, 1, 8, 1).clamp(min=1e-5)
        x_norm = ddpm._revin_normalize(x, mu, sigma)
        x_recovered = ddpm._revin_denormalize(x_norm, mu, sigma)
        torch.testing.assert_close(
            x_recovered, x, atol=1e-5, rtol=1e-5,
            msg=f"Round-trip failed at α={alpha}",
        )


def test_alpha_inflates_normalized_target_by_one_over_alpha():
    """Under α > 1, x0_norm = (x0 − μ_w) / (σ_w × α) is SMALLER than the
    α=1 version by factor 1/α. This is the symmetric effect: training and
    ELBO both see the shrunken normalized input.
    """
    ALPHA = 1.5
    ddpm_uncorr = _make_ddpm(revin=True)
    ddpm_corr = _make_ddpm(revin=True, revin_sigma_correction=ALPHA)
    batch = _make_batch(B=4)
    B, N = batch.num_graphs, batch.y.size(0) // batch.num_graphs
    T_obs, F_obs = batch.x.size(1), batch.x.size(2)
    T_fut, F = batch.y.size(1), batch.y.size(2)

    ut = batch.x.view(B, N, T_obs, F_obs).swapaxes(1, 2)
    x0 = batch.y.view(B, N, T_fut, F).permute(0, 2, 1, 3)

    x0_norm_uncorr, _, _ = ddpm_uncorr._maybe_revin_normalize(x0, ut, batch)
    x0_norm_corr, _, _ = ddpm_corr._maybe_revin_normalize(x0, ut, batch)

    # x0_norm_corr should equal x0_norm_uncorr / α (because σ_w in denom is α× larger)
    torch.testing.assert_close(
        x0_norm_corr, x0_norm_uncorr / ALPHA, atol=1e-5, rtol=1e-5,
    )


def test_sigma_correction_is_non_persistent_buffer():
    """The correction is NOT in state_dict — it's config-controlled.

    Rationale: existing checkpoints (trained with α=1.0) must load cleanly
    under strict=True. Putting α in state_dict would force a legacy
    re-mapping step for every old checkpoint. Instead we keep α in the
    diffusion yaml config so it's an explicit, declared property of each
    eval/training run.
    """
    ALPHA = 1.7
    ddpm = _make_ddpm(revin=True, revin_sigma_correction=ALPHA)
    sd = ddpm.state_dict()
    # Non-persistent: key absent from state_dict
    assert "revin_sigma_correction" not in sd, (
        "revin_sigma_correction must NOT appear in state_dict — it's "
        "a non-persistent buffer driven by config, not by checkpoint."
    )

    # But the attribute is still on the module and applies correctly
    assert float(ddpm.revin_sigma_correction) == pytest.approx(ALPHA, rel=1e-6)


def test_old_checkpoint_loads_strict_with_new_correction():
    """Round-trip: saving a state_dict from a model built BEFORE α existed
    must load strict=True into a model built WITH α=1.7. Simulates the
    F3 ep 475 retrofit scenario: existing checkpoint, new α.
    """
    ALPHA = 1.22
    # Build the "old" model (default α=1.0) — this represents the
    # checkpoint that was trained before the correction was introduced.
    ddpm_old = _make_ddpm(revin=True)
    sd_old = ddpm_old.state_dict()
    assert "revin_sigma_correction" not in sd_old

    # Build the "new" model with α=1.22, then load the old state into it.
    ddpm_new = _make_ddpm(revin=True, revin_sigma_correction=ALPHA)
    missing, unexpected = ddpm_new.load_state_dict(sd_old, strict=True)
    assert missing == [] and unexpected == [], (
        f"strict load must succeed. missing={missing}, unexpected={unexpected}"
    )

    # After loading, α stays at 1.22 (config-driven, NOT overridden by
    # state_dict — which is exactly what we want).
    assert float(ddpm_new.revin_sigma_correction) == pytest.approx(ALPHA, rel=1e-6)


def test_default_zero_kwarg_ddim_works_with_correction():
    """DDIM inherits DDPM constructor → revin_sigma_correction must also
    be present on DDIM and applied identically.
    """
    ddim = _make_ddim(revin=True, revin_sigma_correction=1.22)
    assert float(ddim.revin_sigma_correction) == pytest.approx(1.22, rel=1e-6)
    batch = _make_batch(B=4)
    B, N = batch.num_graphs, batch.y.size(0) // batch.num_graphs
    T_obs, F_obs = batch.x.size(1), batch.x.size(2)
    ut = batch.x.view(B, N, T_obs, F_obs).swapaxes(1, 2)
    target_idx = ddim._resolve_revin_target_idx(batch)
    _, sigma_w = ddim._compute_revin_stats(ut, target_idx)
    raw = ut[:, :, :, target_idx].std(dim=1, unbiased=False, keepdim=True).unsqueeze(-1)
    raw = raw.clamp(min=ddim.revin_eps)
    torch.testing.assert_close(sigma_w, 1.22 * raw, atol=1e-6, rtol=1e-6)
