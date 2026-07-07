"""Regression tests for noise/pred mean+std diagnostics added to
``DDPM.training_loss`` and ``DDPM.training_loss_stratified_t``.

The diagnostics are used to detect noise-scale miscalibration (e.g.,
``pred_std`` systematically deviating from ``noise_std`` under
eps-parameterization, which would indicate a noise-add/subtract scaling
bug in the U-GNN or related modules). Tests cover:

1. Backward-compat: default call returns the same scalar/tuple shape as
   before (no behavioral change for existing callers).
2. ``return_pred_stats=True`` returns the expected dict keys, all finite,
   under both eps and v parameterizations.
3. The injected noise's marginal stats match the construction (mean ≈ 0,
   std ≈ 1) — sanity check that there's no hidden rescaling in
   ``add_noise``.
4. Per-bin keys are present in the stratified variant for all 3 bins.
"""
from __future__ import annotations

import math

import pytest
import torch

from tests.diffusion.test_revin import _make_ddpm, _make_batch


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

def test_training_loss_default_returns_scalar_unchanged():
    """Without return_pred_stats, training_loss must return a single scalar
    tensor (no tuple), preserving the existing call sites.
    """
    ddpm = _make_ddpm(revin=True)
    batch = _make_batch(B=4)
    out = ddpm.training_loss(batch)
    assert isinstance(out, torch.Tensor)
    assert out.ndim == 0


def test_stratified_loss_default_returns_two_tuple_unchanged():
    """Without return_pred_stats, the stratified variant must return
    exactly (loss, bin_losses) — not the 3-tuple version.
    """
    ddpm = _make_ddpm(revin=True)
    batch = _make_batch(B=6)
    out = ddpm.training_loss_stratified_t(batch, num_bins=3)
    assert isinstance(out, tuple)
    assert len(out) == 2
    loss, bin_losses = out
    assert isinstance(loss, torch.Tensor) and loss.ndim == 0
    assert isinstance(bin_losses, dict)


# ---------------------------------------------------------------------------
# return_pred_stats=True: stats dict shape & finiteness
# ---------------------------------------------------------------------------

_MARGINAL_KEYS = {
    "noise_mean", "noise_std",
    "pred_mean", "pred_std",
    "target_mean", "target_std",
    "residual_std",
}


@pytest.mark.parametrize("parameterization", ["eps", "v"])
def test_training_loss_with_pred_stats_returns_expected_keys(parameterization):
    ddpm = _make_ddpm(revin=True, parameterization=parameterization)
    batch = _make_batch(B=4)
    loss, stats = ddpm.training_loss(batch, return_pred_stats=True)

    assert isinstance(loss, torch.Tensor) and loss.ndim == 0
    assert isinstance(stats, dict)
    assert set(stats.keys()) == _MARGINAL_KEYS, (
        f"Unexpected keys for parameterization={parameterization}: "
        f"got {set(stats.keys())}, expected {_MARGINAL_KEYS}"
    )
    for k, v in stats.items():
        assert isinstance(v, float)
        assert math.isfinite(v), f"{k}={v} is not finite under {parameterization=}"


def test_stratified_loss_with_pred_stats_returns_three_tuple_with_per_bin_keys():
    """The stratified variant with return_pred_stats=True must return
    (loss, bin_losses, pred_stats) where pred_stats has both marginal
    keys (e.g. 'pred_std') AND per-bin keys (e.g. 'pred_std_t_low_noise').
    """
    ddpm = _make_ddpm(revin=True)
    batch = _make_batch(B=6)  # divides 3 bins evenly
    out = ddpm.training_loss_stratified_t(batch, num_bins=3, return_pred_stats=True)
    assert isinstance(out, tuple)
    assert len(out) == 3
    loss, bin_losses, stats = out
    assert isinstance(loss, torch.Tensor) and loss.ndim == 0
    assert isinstance(bin_losses, dict) and len(bin_losses) > 0
    assert isinstance(stats, dict)

    # Marginal keys present
    assert _MARGINAL_KEYS.issubset(stats.keys())

    # Per-bin keys for the 3 standard bin labels
    expected_per_bin = {
        f"{stat}_{label}"
        for stat in ("noise_mean", "noise_std", "pred_mean", "pred_std", "residual_std")
        for label in ("t_low_noise", "t_mid", "t_high_noise")
    }
    assert expected_per_bin.issubset(stats.keys()), (
        f"Missing per-bin keys: {expected_per_bin - set(stats.keys())}"
    )

    # All values finite
    for k, v in stats.items():
        assert isinstance(v, float)
        assert math.isfinite(v), f"non-finite {k}={v}"


# ---------------------------------------------------------------------------
# Sanity check: injected noise has the expected marginal stats
# ---------------------------------------------------------------------------

def test_injected_noise_has_unit_variance_zero_mean():
    """The forward process adds ε ~ N(0, I). With a large enough batch the
    measured noise_mean should be ≈ 0 and noise_std ≈ 1 — verifies there's
    no hidden rescaling in add_noise / the RevIN path that would bias the
    noise scale before the prediction.
    """
    # Make the batch big enough that empirical mean/std are tight.
    ddpm = _make_ddpm(revin=True)
    batch = _make_batch(B=64, T_fut=5, N=8, F=1)
    torch.manual_seed(0)
    _, stats = ddpm.training_loss(batch, return_pred_stats=True)
    # B=64 × T=5 × N=8 × F=1 = 2560 samples — std of sample mean ~ 1/sqrt(2560) ≈ 0.02
    assert abs(stats["noise_mean"]) < 0.1, (
        f"noise_mean={stats['noise_mean']:.4f} is too far from 0; expected ≈ 0"
    )
    # Population std should be very close to 1 at this sample size.
    assert 0.9 < stats["noise_std"] < 1.1, (
        f"noise_std={stats['noise_std']:.4f} is far from 1 — possible rescaling bug"
    )


def test_eps_param_target_stats_equal_noise_stats():
    """Under eps-parameterization, target == noise. The diagnostic must
    report identical marginal stats for both."""
    ddpm = _make_ddpm(revin=True, parameterization="eps")
    batch = _make_batch(B=8)
    _, stats = ddpm.training_loss(batch, return_pred_stats=True)
    assert stats["noise_mean"] == pytest.approx(stats["target_mean"], abs=1e-6)
    assert stats["noise_std"] == pytest.approx(stats["target_std"], abs=1e-6)
