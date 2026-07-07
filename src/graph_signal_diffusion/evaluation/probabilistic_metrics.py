"""
Probabilistic forecasting metrics for evaluating ensemble predictions.

These metrics assess the quality of probabilistic forecasts by comparing
the full distribution of predictions against ground truth.
"""

import torch
import numpy as np
from typing import Optional


def continuous_ranked_probability_score(
    predictions: torch.Tensor,
    target: torch.Tensor,
    dim: int = 1
) -> torch.Tensor:
    """
    Compute Continuous Ranked Probability Score (CRPS).
    
    CRPS measures the distance between the predictive CDF and the empirical CDF
    of the observation. Lower is better.
    
    For ensemble forecasts, we use the empirical CRPS formula:
    CRPS = (1/M) * Σ|y_pred - y_true| - (1/2M²) * Σ Σ |y_pred_i - y_pred_j|

    The pairwise-distance term is evaluated via the sort-based O(M log M) identity
    (Hersbach 2000) rather than materializing the O(M²) pairwise matrix, which
    keeps memory at O(B·M·T·N) instead of O(B·M²·T·N).
    
    Args:
        predictions: Ensemble predictions of shape (..., n_samples, ...)
                    e.g., (B, n, T, N, F) where n is the ensemble dimension
        target: Ground truth of shape (..., 1, ...) or (..., ...)
               e.g., (B, 1, T, N, F) or (B, T, N, F)
        dim: Dimension along which ensemble members are stored (default: 1)
    
    Returns:
        CRPS values with ensemble dimension removed
        
    Reference:
        Gneiting & Raftery (2007). Strictly Proper Scoring Rules, Prediction, and Estimation.
    """
    # Ensure predictions and target are float
    predictions = predictions.float()
    target = target.float()

    if target.dim() < predictions.dim():
        # Add dimension if target is missing the ensemble dimension
        target = target.unsqueeze(dim)

    assert target.shape[dim] == 1, "Target must have a singleton dimension at the ensemble dimension."
    target = target.expand_as(predictions)
    assert predictions.shape == target.shape, "Predictions and target must have the same shape after expansion."

    M = predictions.shape[dim]  # Number of ensemble members

    # First term: mean absolute error between predictions and target
    mae_term = torch.mean(torch.abs(predictions - target), dim=dim)

    # Second term: mean pairwise distance between ensemble members, computed via
    # the O(M log M) sort-based identity (avoids the O(M²) pairwise tensor that
    # otherwise OOMs for large B·M²·T·N on the full SP500 test split):
    #   (1/M²) Σᵢ Σⱼ |xᵢ - xⱼ| = (2/M²) Σₖ (2k - M - 1) · x₍ₖ₎   (k = 1..M, sorted asc.)
    sorted_pred, _ = torch.sort(predictions, dim=dim)
    weights = torch.arange(M, device=predictions.device, dtype=predictions.dtype)
    weights = 2.0 * weights - (M - 1)  # k=0..M-1 → coefficients [-(M-1), ..., (M-1)]
    weights_shape = [1] * predictions.dim()
    weights_shape[dim] = M
    weights = weights.view(weights_shape)
    diversity_term = (weights * sorted_pred).sum(dim=dim) * (2.0 / (M * M))

    # CRPS formula
    crps = mae_term - 0.5 * diversity_term

    return crps


def mean_interval_score(
    predictions: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.1,
    dim: int = 1
) -> torch.Tensor:
    """
    Compute Mean Interval Score (MIS).
    
    The interval score evaluates prediction intervals by penalizing:
    1. Wide intervals (uncertainty penalty)
    2. Observations outside the interval (miscalibration penalty)
    
    IS(α) = (U - L) + (2/α) * (L - y) * I{y < L} + (2/α) * (y - U) * I{y > U}
    
    where L and U are the lower and upper bounds of the (1-α) prediction interval.
    
    Args:
        predictions: Ensemble predictions (B, n, T, N, F)
        target: Ground truth (B, T, N, F) or (B, 1, T, N, F)
        alpha: Significance level (default: 0.1 for 90% prediction interval)
        dim: Dimension along which ensemble members are stored
        
    Returns:
        Interval scores
        
    Reference:
        Gneiting & Raftery (2007). Strictly Proper Scoring Rules, Prediction, and Estimation.
    """
    predictions = predictions.float()
    target = target.float()
    
    
    # Compute quantiles for the prediction interval
    lower_quantile = alpha / 2
    upper_quantile = 1 - alpha / 2
    
    # Get lower and upper bounds
    lower = torch.quantile(predictions, lower_quantile, dim=dim)
    upper = torch.quantile(predictions, upper_quantile, dim=dim)
    
    # Squeeze target if it has a singleton dimension
    if target.dim() == predictions.dim() and target.shape[dim] == 1:
        target = target.squeeze(dim)
    
    # Interval width
    width = upper - lower
    
    # Penalties for observations outside the interval
    lower_penalty = (2.0 / alpha) * torch.clamp(lower - target, min=0)
    upper_penalty = (2.0 / alpha) * torch.clamp(target - upper, min=0)
    
    # Total interval score
    interval_score = width + lower_penalty + upper_penalty
    
    return interval_score


def prediction_interval_coverage(
    predictions: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.1,
    dim: int = 1
) -> torch.Tensor:
    """
    Compute empirical coverage of prediction intervals.
    
    Returns the fraction of observations that fall within the (1-α) prediction interval.
    Ideally, this should be close to (1-α).
    
    Args:
        predictions: Ensemble predictions (B, n, T, N, F)
        target: Ground truth (B, T, N, F) or (B, 1, T, N, F)
        alpha: Significance level
        dim: Ensemble dimension
        
    Returns:
        Coverage probability (0 to 1)
    """
    predictions = predictions.float()
    target = target.float()

    
    # Compute quantiles
    lower_quantile = alpha / 2
    upper_quantile = 1 - alpha / 2
    
    lower = torch.quantile(predictions, lower_quantile, dim=dim)
    upper = torch.quantile(predictions, upper_quantile, dim=dim)
    
    # Squeeze target if needed
    if target.dim() == predictions.dim() and target.shape[dim] == 1:
        target = target.squeeze(dim)
    
    # Check if target falls within interval
    within_interval = (target >= lower) & (target <= upper)
    
    # Compute coverage
    coverage = within_interval.float().mean()
    
    return coverage


def ensemble_spread(
    predictions: torch.Tensor,
    dim: int = 1
) -> torch.Tensor:
    """
    Compute ensemble spread (standard deviation across ensemble members).
    
    High spread indicates high uncertainty, low spread indicates confidence.
    
    Args:
        predictions: Ensemble predictions (B, n, T, N, F)
        dim: Ensemble dimension
        
    Returns:
        Standard deviation across ensemble dimension
    """
    return predictions.std(dim=dim)


def compute_probabilistic_metrics(
    predictions: torch.Tensor,
    target: torch.Tensor,
    ensemble_dim: int = 1,
    alphas: Optional[list] = None
) -> dict:
    """
    Compute all probabilistic metrics.
    
    Args:
        predictions: Ensemble predictions (B, n, T, N, F)
        target: Ground truth (B, T, N, F)
        ensemble_dim: Dimension of ensemble members
        alphas: List of significance levels for interval scores
        
    Returns:
        Dictionary of probabilistic metrics
    """
    if alphas is None:
        # 99%, 95%, 90%, 80% prediction intervals.
        # Tail-coverage (99/95) requires enough ensemble samples for stable
        # quantile estimation — keep n_samples ≥ 50 for cov_95, ≥ 200 for
        # cov_99 to read it meaningfully.
        alphas = [0.01, 0.05, 0.1, 0.2]
    
    metrics = {}
    
    # CRPS
    crps = continuous_ranked_probability_score(predictions, target, dim=ensemble_dim)
    metrics['crps_mean'] = crps.mean().item()
    metrics['crps_std'] = crps.std().item()
    
    # Interval scores for different confidence levels
    for alpha in alphas:
        mis = mean_interval_score(predictions, target, alpha=alpha, dim=ensemble_dim)
        coverage = prediction_interval_coverage(predictions, target, alpha=alpha, dim=ensemble_dim)
        
        conf_level = int((1 - alpha) * 100)
        metrics[f'mis_{conf_level}'] = mis.mean().item()
        metrics[f'coverage_{conf_level}'] = coverage.item()
    
    # Ensemble spread
    spread = ensemble_spread(predictions, dim=ensemble_dim)
    metrics['ensemble_spread_mean'] = spread.mean().item()
    metrics['ensemble_spread_std'] = spread.std().item()
    
    return metrics


# Example usage
if __name__ == "__main__":
    print("Testing Probabilistic Metrics")
    print("="*70)
    
    # Create example data
    B, n, T, N, F = 4, 10, 5, 100, 1
    
    # Generate predictions with some spread
    predictions = torch.randn(B, n, T, N, F) * 0.5 + torch.randn(B, 1, T, N, F) * 2
    
    # Generate target
    target = torch.randn(B, T, N, F) * 2
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Target shape: {target.shape}")
    
    # Compute metrics
    metrics = compute_probabilistic_metrics(predictions, target, ensemble_dim=1)
    
    print("\nProbabilistic Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nInterpretation:")
    print(f"  - CRPS measures forecast quality (lower is better)")
    print(f"  - MIS measures interval quality (lower is better)")
    print(f"  - Coverage should be close to the nominal level (e.g., 90% for 90% interval)")
    print(f"  - Ensemble spread quantifies prediction uncertainty")
