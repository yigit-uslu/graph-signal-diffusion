"""
Common evaluation metrics for time series forecasting and graph signal generation.
"""
from typing import Dict, Optional
import torch
import torch.nn.functional as F
import numpy as np


def compute_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean Squared Error."""
    return F.mse_loss(pred, target).item()


def compute_mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean Absolute Error."""
    return F.l1_loss(pred, target).item()


def compute_rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Root Mean Squared Error."""
    return torch.sqrt(F.mse_loss(pred, target)).item()


def compute_mape(pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-8) -> float:
    """Mean Absolute Percentage Error."""
    return (torch.abs((target - pred) / (target + epsilon))).mean().item() * 100


def compute_smape(pred: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-8) -> float:
    """Symmetric Mean Absolute Percentage Error."""
    numerator = torch.abs(pred - target)
    denominator = (torch.abs(pred) + torch.abs(target)) / 2 + epsilon
    return (numerator / denominator).mean().item() * 100


def compute_direction_accuracy(pred: torch.Tensor, target: torch.Tensor, 
                               prev_values: Optional[torch.Tensor] = None) -> float:
    """
    Directional accuracy: fraction of correct up/down predictions.
    
    Args:
        pred: Predictions [B, T, N, F]
        target: Ground truth [B, T, N, F]
        prev_values: Previous values for computing returns [B, N, F]
    """
    if prev_values is None:
        # Use first timestep as reference
        pred_direction = (pred[:, 1:] > pred[:, :-1]).float()
        target_direction = (target[:, 1:] > target[:, :-1]).float()
    else:
        # Use provided previous values
        prev_expanded = prev_values.unsqueeze(1)  # [B, 1, N, F]
        pred_direction = (pred > prev_expanded).float()
        target_direction = (target > prev_expanded).float()
    
    return (pred_direction == target_direction).float().mean().item()


def compute_correlation(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Pearson correlation coefficient."""
    pred_flat = pred.reshape(-1)
    target_flat = target.reshape(-1)
    
    pred_mean = pred_flat.mean()
    target_mean = target_flat.mean()
    
    numerator = ((pred_flat - pred_mean) * (target_flat - target_mean)).sum()
    denominator = torch.sqrt(((pred_flat - pred_mean) ** 2).sum() * 
                            ((target_flat - target_mean) ** 2).sum())
    
    return (numerator / (denominator + 1e-8)).item()


def compute_r2_score(pred: torch.Tensor, target: torch.Tensor) -> float:
    """R² (coefficient of determination)."""
    ss_res = ((target - pred) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    return (1 - ss_res / (ss_tot + 1e-8)).item()


def compute_quantile_loss(pred: torch.Tensor, target: torch.Tensor, quantile: float = 0.5) -> float:
    """Quantile loss (pinball loss)."""
    errors = target - pred
    loss = torch.max((quantile - 1) * errors, quantile * errors)
    return loss.mean().item()


def compute_all_metrics(pred: torch.Tensor, target: torch.Tensor, 
                       prev_values: Optional[torch.Tensor] = None,
                       prefix: str = "") -> Dict[str, float]:
    """
    Compute all standard metrics.
    
    Args:
        pred: Predictions [B, T, N, F]
        target: Ground truth [B, T, N, F]
        prev_values: Previous values for directional accuracy [B, N, F]
        prefix: Prefix for metric names (e.g., "test_")
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        f"{prefix}mse": compute_mse(pred, target),
        f"{prefix}mae": compute_mae(pred, target),
        f"{prefix}rmse": compute_rmse(pred, target),
        f"{prefix}mape": compute_mape(pred, target),
        f"{prefix}smape": compute_smape(pred, target),
        f"{prefix}correlation": compute_correlation(pred, target),
        f"{prefix}r2": compute_r2_score(pred, target),
    }
    
    if prev_values is not None:
        metrics[f"{prefix}direction_accuracy"] = compute_direction_accuracy(
            pred, target, prev_values
        )
    
    return metrics


# Alias for backward compatibility
compute_metrics = compute_all_metrics