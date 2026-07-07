"""Tests for the absolute (threshold=0) direction-accuracy metric and the
shared helper ``_compute_direction_accuracy_pair``.

Two thresholds reported by the SP500 evaluator:
  * vs per-stock long-run mean μ_s (existing — drift-removed)
  * vs 0                          (new — absolute sign test)

These tests cover the helper directly and verify the absolute-vs-μ_s
divergence patterns we'd expect from a positively-drifting series.
"""
from __future__ import annotations

import torch
import pytest

from graph_signal_diffusion.tasks.stock_price_forecasting.evaluator import (
    _compute_direction_accuracy_pair,
)


# ---------------------------------------------------------------------------
# Helper-level tests
# ---------------------------------------------------------------------------

def test_perfect_match_with_zero_threshold():
    """Predictions matching target signs => acc=1.0 against threshold=0."""
    # 1 batch, 2 timesteps, 2 nodes, 1 feature
    target = torch.tensor([[[[1.0], [-1.0]], [[-2.0], [3.0]]]])   # [1, 2, 2, 1]
    # Ensemble of 3 — all members agree on sign with target
    gen = target.unsqueeze(1).expand(-1, 3, -1, -1, -1).clone()    # [1, 3, 2, 2, 1]
    pred_mean = target.clone()

    per_sample, majority = _compute_direction_accuracy_pair(
        target=target, gen_ensemble=gen, pred_mean=pred_mean,
        threshold=0.0, ensemble_dim=1,
    )
    assert per_sample == 1.0
    assert majority == 1.0


def test_zero_threshold_vs_positive_mu_divergence():
    """When μ_s > 0 and target sits between 0 and μ_s, absolute says 'up' but
    μ_s-relative says 'below trend'. The two metrics should disagree."""
    # 1 batch, 1 timestep, 1 node, 1 feature
    target = torch.tensor([[[[0.5]]]])               # [1, 1, 1, 1]; > 0, < μ_s=1.0
    pred_mean = torch.tensor([[[[0.3]]]])            # [1, 1, 1, 1]; > 0, < μ_s
    mu_s = torch.tensor([[[[1.0]]]])                 # [1, 1, 1, 1]

    # vs μ_s: target_anomaly = (0.5 > 1.0) = False;  pred_anomaly = (0.3 > 1.0) = False
    #          → both "down" → per_sample=1.0
    acc_mu, _ = _compute_direction_accuracy_pair(
        target=target, gen_ensemble=None, pred_mean=pred_mean,
        threshold=mu_s, ensemble_dim=1,
    )
    assert acc_mu == 1.0

    # vs 0: target_anomaly = (0.5 > 0) = True; pred_anomaly = (0.3 > 0) = True
    #        → both "up" → per_sample=1.0 (also)
    acc_zero, _ = _compute_direction_accuracy_pair(
        target=target, gen_ensemble=None, pred_mean=pred_mean,
        threshold=0.0, ensemble_dim=1,
    )
    assert acc_zero == 1.0

    # Now flip pred to a value that's between 0 and μ_s — but on the OTHER side of 0 vs μ_s
    pred_misalign = torch.tensor([[[[-0.2]]]])  # < 0, < μ_s
    # vs μ_s: pred_anomaly = (-0.2 > 1.0) = False; target_anomaly = False → MATCH
    acc_mu_misalign, _ = _compute_direction_accuracy_pair(
        target=target, gen_ensemble=None, pred_mean=pred_misalign,
        threshold=mu_s, ensemble_dim=1,
    )
    assert acc_mu_misalign == 1.0
    # vs 0:   pred_anomaly = (-0.2 > 0) = False; target_anomaly = True → MISS
    acc_zero_misalign, _ = _compute_direction_accuracy_pair(
        target=target, gen_ensemble=None, pred_mean=pred_misalign,
        threshold=0.0, ensemble_dim=1,
    )
    assert acc_zero_misalign == 0.0


def test_per_stock_mean_broadcasts_correctly():
    """Threshold [1, 1, N, 1] broadcasts with target [B, T, N, F] and
    ensemble [B, n, T, N, F]."""
    B, T, N, F = 2, 3, 4, 1
    n = 5
    target = torch.randn(B, T, N, F)
    gen = torch.randn(B, n, T, N, F)
    pred_mean = gen.mean(dim=1)  # ensemble mean

    # Per-stock means: nonzero, varied
    mu_s = torch.tensor([0.1, -0.2, 0.05, 0.3]).view(1, 1, N, 1)

    per_sample, majority = _compute_direction_accuracy_pair(
        target=target, gen_ensemble=gen, pred_mean=pred_mean,
        threshold=mu_s, ensemble_dim=1,
    )
    # Just verify it runs and returns sensible values.
    assert 0.0 <= per_sample <= 1.0
    assert majority is not None
    assert 0.0 <= majority <= 1.0


def test_single_sample_fallback_returns_none_majority():
    """When no ensemble is provided, majority vote is None."""
    target = torch.tensor([[[[1.0], [-1.0]]]])     # [1, 1, 2, 1]
    pred_mean = torch.tensor([[[[0.5], [-0.5]]]])  # both signs match

    per_sample, majority = _compute_direction_accuracy_pair(
        target=target, gen_ensemble=None, pred_mean=pred_mean,
        threshold=0.0, ensemble_dim=1,
    )
    assert per_sample == 1.0
    assert majority is None


def test_ensemble_size_one_treated_as_single_sample():
    """ensemble of size 1 along ensemble_dim hits the single-sample branch
    (per spec: >1 ensemble members trigger ensemble path)."""
    target = torch.tensor([[[[1.0], [-1.0]]]])     # [1, 1, 2, 1]
    gen = torch.tensor([[[[[0.5], [-0.5]]]]])      # [1, 1, 1, 2, 1] — ensemble_dim=1, n=1
    pred_mean = torch.tensor([[[[0.5], [-0.5]]]])

    per_sample, majority = _compute_direction_accuracy_pair(
        target=target, gen_ensemble=gen, pred_mean=pred_mean,
        threshold=0.0, ensemble_dim=1,
    )
    assert majority is None  # single-sample fallback path


def test_majority_vote_breaks_tie_to_down():
    """When exactly half the ensemble says 'up', vote_frac == 0.5; the rule
    `vote_frac > 0.5` evaluates False — majority predicts 'down'."""
    # 1 batch, 1 timestep, 1 node, 1 feature; ensemble of 2: one up, one down
    target = torch.tensor([[[[1.0]]]])              # "up"
    gen = torch.tensor([[[[[ 1.0]]],
                         [[[-1.0]]]]])              # [1, 2, 1, 1, 1]
    pred_mean = gen.mean(dim=1)                     # 0.0

    per_sample, majority = _compute_direction_accuracy_pair(
        target=target, gen_ensemble=gen, pred_mean=pred_mean,
        threshold=0.0, ensemble_dim=1,
    )
    # per_sample: member 0 correct (up==up), member 1 wrong (down!=up) → 0.5
    assert per_sample == 0.5
    # majority: vote_frac=0.5 → > 0.5 is False → "down" → mismatch with target "up" → 0.0
    assert majority == 0.0


def test_all_below_threshold_returns_one():
    """If every point in target and gen_ensemble is below the threshold,
    both anomalies are False everywhere → trivially perfect match."""
    target = torch.full((1, 1, 3, 1), -1.0)
    gen = torch.full((1, 4, 1, 3, 1), -1.0)
    pred_mean = gen.mean(dim=1)

    per_sample, majority = _compute_direction_accuracy_pair(
        target=target, gen_ensemble=gen, pred_mean=pred_mean,
        threshold=0.0, ensemble_dim=1,
    )
    assert per_sample == 1.0
    assert majority == 1.0


# ---------------------------------------------------------------------------
# Random-data baseline: absolute and μ_s metrics should fall in [0,1]
# ---------------------------------------------------------------------------

def test_random_data_stable_in_unit_interval():
    """For ε~N(0,1) target and generation, both threshold variants stay in
    [0, 1] and aren't 0 or 1 (i.e., not degenerate)."""
    torch.manual_seed(0)
    B, T, N, F = 8, 5, 6, 1
    n = 10
    target = torch.randn(B, T, N, F)
    gen = torch.randn(B, n, T, N, F)
    pred_mean = gen.mean(dim=1)
    mu_s = torch.randn(1, 1, N, 1) * 0.1  # small per-stock means

    for thresh in [0.0, mu_s]:
        per_sample, majority = _compute_direction_accuracy_pair(
            target=target, gen_ensemble=gen, pred_mean=pred_mean,
            threshold=thresh, ensemble_dim=1,
        )
        assert 0.0 < per_sample < 1.0, f"per_sample={per_sample} for threshold={thresh}"
        assert 0.0 < majority < 1.0,  f"majority={majority} for threshold={thresh}"


# ---------------------------------------------------------------------------
# Equivalence: threshold=0 with zero-μ_s tensor matches scalar threshold
# ---------------------------------------------------------------------------

def test_zero_tensor_threshold_equivalent_to_scalar_zero():
    """Threshold=torch.zeros([1, 1, N, 1]) gives identical results to
    threshold=0.0."""
    torch.manual_seed(1)
    B, T, N, F = 3, 4, 5, 1
    n = 7
    target = torch.randn(B, T, N, F)
    gen = torch.randn(B, n, T, N, F)
    pred_mean = gen.mean(dim=1)
    zero_tensor = torch.zeros(1, 1, N, 1)

    per_sample_scalar, majority_scalar = _compute_direction_accuracy_pair(
        target=target, gen_ensemble=gen, pred_mean=pred_mean,
        threshold=0.0, ensemble_dim=1,
    )
    per_sample_tensor, majority_tensor = _compute_direction_accuracy_pair(
        target=target, gen_ensemble=gen, pred_mean=pred_mean,
        threshold=zero_tensor, ensemble_dim=1,
    )
    assert per_sample_scalar == pytest.approx(per_sample_tensor)
    assert majority_scalar == pytest.approx(majority_tensor)
