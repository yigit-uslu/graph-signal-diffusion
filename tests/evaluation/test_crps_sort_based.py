"""Regression tests for the sort-based CRPS rewrite in
`graph_signal_diffusion.evaluation.probabilistic_metrics.continuous_ranked_probability_score`.

The rewrite replaces the O(M²) pairwise-distance tensor with the O(M log M)
Hersbach identity. These tests verify numerical equivalence against a
brute-force reference and sanity-check edge cases (M=1, non-default `dim`,
degenerate-ensemble, analytic Gaussian limit).
"""

import math

import pytest
import torch

from graph_signal_diffusion.evaluation.probabilistic_metrics import (
    continuous_ranked_probability_score,
)


def _brute_force_crps(predictions: torch.Tensor, target: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Reference O(M²) implementation matching the pre-rewrite formula."""
    predictions = predictions.float()
    target = target.float()
    if target.dim() < predictions.dim():
        target = target.unsqueeze(dim)
    target = target.expand_as(predictions)
    mae_term = torch.mean(torch.abs(predictions - target), dim=dim)
    pred_i = predictions.unsqueeze(dim)
    pred_j = predictions.unsqueeze(dim + 1)
    pairwise = torch.abs(pred_i - pred_j)
    diversity = torch.mean(pairwise, dim=[dim, dim + 1])
    return mae_term - 0.5 * diversity


@pytest.mark.parametrize("shape,dim", [
    ((4, 10, 5, 3, 1), 1),   # canonical (B, n, T, N, F)
    ((2, 8, 6), 1),           # 3D
    ((3, 4, 7, 2), 2),        # non-default dim
    ((1, 20, 1, 1, 1), 1),    # large-n single batch
])
def test_matches_brute_force(shape, dim):
    torch.manual_seed(42)
    predictions = torch.randn(*shape)
    target_shape = list(shape)
    target_shape[dim] = 1
    target = torch.randn(*target_shape)

    crps_new = continuous_ranked_probability_score(predictions, target, dim=dim)
    crps_ref = _brute_force_crps(predictions, target, dim=dim)

    assert crps_new.shape == crps_ref.shape
    assert torch.allclose(crps_new, crps_ref, atol=1e-5, rtol=1e-4), (
        f"Sort-based CRPS diverges from brute-force reference: "
        f"max abs diff = {(crps_new - crps_ref).abs().max().item():.2e}"
    )


def test_broadcasts_target_missing_ensemble_dim():
    torch.manual_seed(0)
    predictions = torch.randn(3, 6, 4)
    target = torch.randn(3, 4)
    crps_new = continuous_ranked_probability_score(predictions, target, dim=1)
    crps_ref = _brute_force_crps(predictions, target, dim=1)
    assert torch.allclose(crps_new, crps_ref, atol=1e-5)


def test_single_member_ensemble_equals_mae():
    predictions = torch.tensor([[[1.2]], [[-3.4]]])
    target = torch.tensor([[[0.2]], [[-2.0]]])
    crps = continuous_ranked_probability_score(predictions, target, dim=1)
    expected_mae = torch.abs(predictions.squeeze(1) - target.squeeze(1))
    assert torch.allclose(crps, expected_mae, atol=1e-6)


def test_degenerate_ensemble_all_equal():
    predictions = torch.full((2, 5, 3), 2.0)
    target = torch.tensor([[0.0, 1.0, 2.0], [3.0, -1.0, 4.0]])
    crps = continuous_ranked_probability_score(predictions, target, dim=1)
    expected = torch.abs(predictions[:, 0, :] - target)
    assert torch.allclose(crps, expected, atol=1e-6)


def test_gaussian_analytic_limit():
    """For X ~ N(μ, σ²) and y fixed, CRPS → σ·[2φ(z) + z(2Φ(z) - 1) - 1/√π]
    where z = (y-μ)/σ. With a large ensemble, the empirical CRPS converges."""
    torch.manual_seed(7)
    mu, sigma, y = 0.0, 1.0, 0.5
    M = 20000
    predictions = torch.randn(1, M, 1) * sigma + mu
    target = torch.tensor([[y]])
    crps = continuous_ranked_probability_score(predictions, target, dim=1).item()

    z = (y - mu) / sigma
    phi = math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    Phi = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    crps_analytic = sigma * (2 * phi + z * (2 * Phi - 1) - 1 / math.sqrt(math.pi))

    assert abs(crps - crps_analytic) < 0.02, (
        f"Empirical CRPS {crps:.4f} deviates from analytic {crps_analytic:.4f}"
    )


def test_large_m_does_not_oom_and_stays_fast():
    """Memory regression guard: M=200 would previously materialize a
    (B, M, M, T, N) tensor. Sort-based path should handle it trivially."""
    B, M, T, N = 4, 200, 10, 50
    predictions = torch.randn(B, M, T, N)
    target = torch.randn(B, 1, T, N)
    crps = continuous_ranked_probability_score(predictions, target, dim=1)
    assert crps.shape == (B, T, N)
    assert torch.isfinite(crps).all()
