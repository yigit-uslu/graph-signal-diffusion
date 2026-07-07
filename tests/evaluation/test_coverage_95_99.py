"""Tests for the cov_95 / cov_99 extension to `compute_probabilistic_metrics`.

Adds tail-coverage diagnostics for rare-event probabilistic forecasting.
Verifies:
  1. Default alphas now emit both new keys (coverage_99/95 + mis_99/95).
  2. Numerical correctness at coverage_95 and coverage_99 for a known
     Gaussian ensemble — the empirical coverage of the [2.5%, 97.5%] and
     [0.5%, 99.5%] intervals should match the target as ensemble size grows.
  3. Monotonicity: at the same predictions/target, coverage_99 ≥ coverage_95
     ≥ coverage_90 ≥ coverage_80 (wider intervals always cover more).
  4. Backwards-compat: legacy `coverage_90` / `coverage_80` / `mis_90` /
     `mis_80` keys are still present.
"""
from __future__ import annotations

import math

import pytest
import torch

from graph_signal_diffusion.evaluation.probabilistic_metrics import (
    compute_probabilistic_metrics,
)


def test_default_alphas_include_99_95():
    torch.manual_seed(0)
    preds = torch.randn(8, 200, 5, 3, 1)
    target = torch.randn(8, 5, 3, 1)

    m = compute_probabilistic_metrics(preds, target, ensemble_dim=1)

    for k in ("coverage_99", "coverage_95", "coverage_90", "coverage_80",
              "mis_99", "mis_95", "mis_90", "mis_80",
              "crps_mean", "crps_std",
              "ensemble_spread_mean", "ensemble_spread_std"):
        assert k in m, f"missing metric key: {k}"


def test_coverage_monotonic_in_nominal_level():
    torch.manual_seed(0)
    preds = torch.randn(4, 500, 4, 3, 1) * 0.7 + torch.randn(4, 1, 4, 3, 1) * 1.5
    target = torch.randn(4, 4, 3, 1) * 1.5

    m = compute_probabilistic_metrics(preds, target, ensemble_dim=1)

    # Wider nominal interval ⇒ no fewer empirical hits.
    assert m["coverage_99"] >= m["coverage_95"] >= m["coverage_90"] >= m["coverage_80"], (
        f"coverage not monotonic: 99={m['coverage_99']:.4f}, 95={m['coverage_95']:.4f}, "
        f"90={m['coverage_90']:.4f}, 80={m['coverage_80']:.4f}"
    )


def test_gaussian_coverage_matches_nominal_with_large_ensemble():
    """For y ~ N(μ, 1), preds ~ N(μ, 1), the [2.5%, 97.5%] and [0.5%, 99.5%]
    quantiles of the ensemble should cover the target at the nominal rate.

    Note: the calibration is measured *across many independent (pred, y)
    pairs*, so set B large to drive Monte-Carlo error well below the test
    tolerance."""
    torch.manual_seed(42)
    B, M = 1000, 2000
    preds = torch.randn(B, M, 1, 1, 1)   # N(0, 1) per element
    target = torch.randn(B, 1, 1, 1)     # N(0, 1) — same marginal as preds

    m = compute_probabilistic_metrics(preds, target, ensemble_dim=1)

    # Tolerances:
    #   cov_99: SE ~ √(0.99·0.01/B) ~ 0.0031, allow 0.012 (≈ 4σ)
    #   cov_95: SE ~ √(0.95·0.05/B) ~ 0.0069, allow 0.020
    assert abs(m["coverage_99"] - 0.99) < 0.012, f"cov_99 = {m['coverage_99']:.4f}"
    assert abs(m["coverage_95"] - 0.95) < 0.020, f"cov_95 = {m['coverage_95']:.4f}"


def test_mis_99_present_and_positive():
    """MIS scores are non-negative and finite for a non-trivial ensemble."""
    torch.manual_seed(7)
    preds = torch.randn(4, 100, 3, 2, 1)
    target = torch.randn(4, 3, 2, 1) * 2  # mismatched scale → non-zero MIS
    m = compute_probabilistic_metrics(preds, target, ensemble_dim=1)
    for k in ("mis_99", "mis_95"):
        assert math.isfinite(m[k]), f"{k} not finite: {m[k]}"
        assert m[k] >= 0, f"{k} negative: {m[k]}"


def test_caller_explicit_alphas_still_supported():
    """Existing callers passing alphas explicitly still get the right keys."""
    preds = torch.randn(2, 50, 3, 2, 1)
    target = torch.randn(2, 3, 2, 1)
    m = compute_probabilistic_metrics(preds, target, ensemble_dim=1,
                                       alphas=[0.05])
    assert "coverage_95" in m and "mis_95" in m
    # Other coverage keys should be absent when not requested.
    assert "coverage_90" not in m
    assert "coverage_99" not in m
