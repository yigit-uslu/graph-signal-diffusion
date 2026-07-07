"""
Visualization utilities for evaluation results.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional
from graph_signal_diffusion.baselines.plot_utils import format_baseline_display_name


def _as_dict(value) -> Dict:
    return value if isinstance(value, dict) else {}


def plot_predictions_vs_actual(
    predictions: torch.Tensor,
    actuals: torch.Tensor,
    timestamps: Optional[np.ndarray] = None,
    stock_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    num_samples: int = 5,
    num_stocks: int = 3
):
    """
    Plot predicted vs actual values for multiple stocks.
    
    Args:
        predictions: [B, T, N, F]
        actuals: [B, T, N, F]
        timestamps: Time indices
        stock_names: Names of stocks
        save_path: Where to save the plot
        num_samples: Number of batch samples to plot
        num_stocks: Number of stocks per sample
    """
    B, T, N, F = predictions.shape
    num_samples = min(num_samples, B)
    num_stocks = min(num_stocks, N)
    
    fig, axes = plt.subplots(num_samples, num_stocks, 
                            figsize=(5*num_stocks, 4*num_samples))
    if num_samples == 1 and num_stocks == 1:
        axes = np.array([[axes]])
    elif num_samples == 1:
        axes = axes.reshape(1, -1)
    elif num_stocks == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(num_samples):
        for j in range(num_stocks):
            ax = axes[i, j]
            
            pred_series = predictions[i, :, j, 0].cpu().numpy()
            actual_series = actuals[i, :, j, 0].cpu().numpy()
            
            x = timestamps if timestamps is not None else np.arange(T)
            
            ax.plot(x, actual_series, label='Actual', marker='o', linewidth=2)
            ax.plot(x, pred_series, label='Predicted', marker='x', linewidth=2, linestyle='--')
            
            stock_name = stock_names[j] if stock_names else f"Stock {j}"
            ax.set_title(f"Sample {i} - {stock_name}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_error_distribution(
    predictions: torch.Tensor,
    actuals: torch.Tensor,
    save_path: Optional[str] = None
):
    """Plot distribution of prediction errors."""
    errors = (predictions - actuals).cpu().numpy().flatten()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram
    axes[0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[0].set_xlabel("Prediction Error")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Error Distribution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Box plot
    axes[1].boxplot(errors, vert=True)
    axes[1].set_ylabel("Prediction Error")
    axes[1].set_title("Error Box Plot")
    axes[1].grid(True, alpha=0.3)
    
    # plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_baseline_comparison(
    results_df,
    metric: str = 'mse',
    save_path: Optional[str] = None,
    style_cfg: Optional[Dict] = None,
):
    """
    Plot comparison of different baselines.
    
    Args:
        results_df: DataFrame with baselines as rows and metrics as columns
        metric: Which metric to plot
        save_path: Where to save
    """
    if metric not in results_df.columns:
        raise ValueError(f"Metric {metric} not found in results")
    
    style = _as_dict(style_cfg)

    figsize = tuple(style.get("figure_size", (10, 6)))
    palette = style.get("palette", "husl")
    bar_edgecolor = style.get("bar_edgecolor", "black")
    value_label_fmt = str(style.get("value_label_format", "{:.4f}"))
    value_label_fontsize = int(style.get("value_label_fontsize", 10))
    y_label_fontsize = int(style.get("y_label_fontsize", 12))
    title_fontsize = int(style.get("title_fontsize", 14))
    title_fontweight = style.get("title_fontweight", "bold")
    x_tick_rotation = float(style.get("x_tick_rotation", 45))
    x_tick_ha = style.get("x_tick_ha", "right")
    grid_axis = style.get("grid_axis", "y")
    grid_alpha = float(style.get("grid_alpha", 0.3))
    dpi = int(style.get("dpi", 150))

    fig, ax = plt.subplots(figsize=figsize)
    
    baselines = results_df.index.tolist()
    baseline_labels = [format_baseline_display_name(name) for name in baselines]
    values = results_df[metric].values
    
    colors = sns.color_palette(palette, len(baselines))
    bars = ax.bar(baseline_labels, values, color=colors, edgecolor=bar_edgecolor)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                value_label_fmt.format(val),
                ha='center', va='bottom', fontsize=value_label_fontsize)
    
    ax.set_ylabel(metric.upper(), fontsize=y_label_fontsize)
    ax.set_title(
        f"Baseline Comparison - {metric.upper()}",
        fontsize=title_fontsize,
        fontweight=title_fontweight,
    )
    ax.grid(True, axis=grid_axis, alpha=grid_alpha)
    plt.xticks(rotation=x_tick_rotation, ha=x_tick_ha)
    # plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_all_metrics_comparison(
    results_df,
    save_path: Optional[str] = None,
    style_cfg: Optional[Dict] = None,
):
    """Plot radar chart comparing all metrics across baselines."""
    import pandas as pd
    from math import pi
    
    # Normalize metrics to [0, 1] for fair comparison
    df_norm = results_df.copy()
    for col in df_norm.columns:
        min_val = df_norm[col].min()
        max_val = df_norm[col].max()
        if max_val > min_val:
            df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
    
    categories = list(df_norm.columns)
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    style = _as_dict(style_cfg)

    figsize = tuple(style.get("figure_size", (10, 10)))
    line_style = str(style.get("line_style", "o-"))
    line_width = float(style.get("line_width", 2.0))
    fill_alpha = float(style.get("fill_alpha", 0.15))
    legend_loc = style.get("legend_loc", "upper right")
    legend_bbox_to_anchor = tuple(style.get("legend_bbox_to_anchor", (1.3, 1.1)))
    title = str(style.get("title", "Multi-Metric Comparison"))
    title_fontsize = int(style.get("title_fontsize", 16))
    title_fontweight = style.get("title_fontweight", "bold")
    title_pad = float(style.get("title_pad", 20))
    show_grid = bool(style.get("grid", True))
    dpi = int(style.get("dpi", 150))

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'),
                           layout=None)
    
    for idx, baseline in enumerate(df_norm.index):
        values = df_norm.loc[baseline].tolist()
        values += values[:1]

        ax.plot(
            angles,
            values,
            line_style,
            linewidth=line_width,
            label=format_baseline_display_name(baseline),
        )
        ax.fill(angles, values, alpha=fill_alpha)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.legend(loc=legend_loc, bbox_to_anchor=legend_bbox_to_anchor)
    ax.set_title(
        title,
        fontsize=title_fontsize,
        fontweight=title_fontweight,
        pad=title_pad,
    )
    ax.grid(show_grid)
    
    # plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_temporal_error_evolution(
    predictions: torch.Tensor,
    actuals: torch.Tensor,
    save_path: Optional[str] = None
):
    """
    Plot how prediction error evolves over time horizon.
    
    Args:
        predictions: [B, T, N, F]
        actuals: [B, T, N, F]
    """
    T = predictions.size(1)
    
    # Compute error at each timestep
    errors = torch.abs(predictions - actuals).mean(dim=(0, 2, 3)).cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(range(1, T+1), errors, marker='o', linewidth=2, markersize=8)
    ax.set_xlabel("Time Horizon (steps ahead)", fontsize=12)
    ax.set_ylabel("Mean Absolute Error", fontsize=12)
    ax.set_title("Prediction Error vs Time Horizon", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
