"""
Visualization helpers for Primal-Dual training progress.

Extracted from ``primal_dual_trainer.py`` to keep the trainer module focused
on the optimisation loop.  All public symbols that were previously importable
from the trainer are re-exported there for backward compatibility.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import uniform_filter1d

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Small utility helpers
# ---------------------------------------------------------------------------


def smooth_curve(data, window=50):
    """Apply moving average smoothing."""
    if len(data) < window:
        return data
    return uniform_filter1d(data, size=window, mode='nearest')


def _load_training_cfg_from_hydra(output_dir: str) -> dict:
    """Best-effort loader for Hydra training config from an output directory."""
    config_path = Path(output_dir) / ".hydra" / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        import yaml
    except Exception:
        return {}
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        return {}
    training_cfg = cfg.get("training", {})
    return training_cfg if isinstance(training_cfg, dict) else {}


def _parse_cfg_float(value, default: float) -> float:
    """Parse float-like config values, including string infinities and Hydra placeholders."""
    if value is None:
        return float(default)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        raw = value.strip()
        low = raw.lower()
        if "${" in raw:
            return float(default)
        if low in {"inf", "+inf", "infinity", "+infinity"}:
            return float("inf")
        if low in {"-inf", "-infinity"}:
            return float("-inf")
        try:
            return float(raw)
        except ValueError:
            return float(default)
    return float(default)


def _parse_cfg_int(value, default: int) -> int:
    """Parse int-like config values while tolerating unresolved Hydra placeholders."""
    if value is None:
        return int(default)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and np.isfinite(value):
        return int(value)
    if isinstance(value, str):
        raw = value.strip()
        if "${" in raw:
            return int(default)
        try:
            return int(float(raw))
        except ValueError:
            return int(default)
    return int(default)


# ---------------------------------------------------------------------------
# Convergence-criteria helpers
# ---------------------------------------------------------------------------


def _build_criterion_status_history(
    summaries: list[dict], training_cfg: dict
) -> tuple[list[str], dict[str, list[float]], dict[str, list[float]], dict[str, float]]:
    """Build status/value traces and thresholds for convergence criteria."""
    # Preferred path: use explicitly persisted statuses from epoch summaries.
    has_persisted_status = any("convergence_criteria_status" in s for s in summaries)

    # Backward-compatible fallback: reconstruct from scalar summary metrics.
    convergence_window = _parse_cfg_int(training_cfg.get("convergence_window"), 50)
    warmup_cfg = training_cfg.get("convergence_warmup_epochs", convergence_window)
    convergence_warmup_epochs = (
        convergence_window
        if warmup_cfg is None
        else _parse_cfg_int(warmup_cfg, convergence_window)
    )

    thresholds = {
        "grad_norm": _parse_cfg_float(training_cfg.get("gradient_norm_threshold"), 1e-4),
        "dual_variance": _parse_cfg_float(training_cfg.get("dual_variance_threshold"), 0.01),
        "dual_stationarity": _parse_cfg_float(training_cfg.get("dual_stationarity_threshold"), 0.05),
        "violation_fraction": _parse_cfg_float(training_cfg.get("violation_fraction_threshold"), 0.05),
        "violation_fraction_on_model_avg_rates": _parse_cfg_float(
            training_cfg.get("violation_fraction_on_model_avg_rates_threshold"), 0.05
        ),
        "mean_violation_slack_on_model_avg_rates": _parse_cfg_float(
            training_cfg.get("mean_violation_slack_on_model_avg_rates_threshold"), float("inf")
        ),
    }

    has_violation_fraction = any("violation_fraction" in s for s in summaries)
    has_violation_fraction_model_avg = any(
        "violation_fraction_on_model_avg_rates" in s for s in summaries
    )
    has_mean_violation_slack_model_avg = any(
        "model_avg_mean_violation_slack" in s for s in summaries
    )

    criterion_names = ["grad_norm", "dual_variance", "dual_stationarity"]
    if has_violation_fraction:
        criterion_names.append("violation_fraction")
    if has_violation_fraction_model_avg:
        criterion_names.append("violation_fraction_on_model_avg_rates")
    if has_mean_violation_slack_model_avg:
        criterion_names.append("mean_violation_slack_on_model_avg_rates")

    fallback_status_history = {name: [] for name in criterion_names}
    criterion_value_history = {name: [] for name in criterion_names}
    for idx, summary in enumerate(summaries):
        epoch_val = int(summary.get("epoch", idx + 1))
        start = max(0, idx - convergence_window + 1)
        window_summaries = summaries[start : idx + 1]

        if epoch_val < convergence_warmup_epochs or len(window_summaries) < convergence_window:
            for name in criterion_names:
                fallback_status_history[name].append(np.nan)
                criterion_value_history[name].append(np.nan)
            continue

        grad_values = [s.get("gradient_norm") for s in window_summaries if s.get("gradient_norm") is not None]
        if len(grad_values) >= convergence_window:
            mean_grad = float(np.mean(grad_values))
            criterion_value_history["grad_norm"].append(mean_grad)
            grad_converged = mean_grad < thresholds["grad_norm"]
            fallback_status_history["grad_norm"].append(1.0 if grad_converged else 0.0)
        else:
            criterion_value_history["grad_norm"].append(np.nan)
            fallback_status_history["grad_norm"].append(np.nan)

        std_values = [s.get("std_lambda") for s in window_summaries if s.get("std_lambda") is not None]
        if len(std_values) >= convergence_window:
            first_half = std_values[:convergence_window // 2]
            second_half = std_values[convergence_window // 2 :]
            std_change_rate = abs(np.mean(second_half) - np.mean(first_half)) / (np.mean(first_half) + 1e-6)
            criterion_value_history["dual_variance"].append(float(std_change_rate))
            dual_var_converged = std_change_rate < thresholds["dual_variance"]
            fallback_status_history["dual_variance"].append(1.0 if dual_var_converged else 0.0)
        else:
            criterion_value_history["dual_variance"].append(np.nan)
            fallback_status_history["dual_variance"].append(np.nan)

        comp_abs_values = [
            s.get("ergodic_complementary_slackness_abs")
            for s in window_summaries
            if s.get("ergodic_complementary_slackness_abs") is not None
        ]
        if len(comp_abs_values) >= convergence_window:
            mean_abs_comp_slack = float(np.mean(comp_abs_values))
            criterion_value_history["dual_stationarity"].append(mean_abs_comp_slack)
            dual_stationarity_converged = mean_abs_comp_slack < thresholds["dual_stationarity"]
            fallback_status_history["dual_stationarity"].append(1.0 if dual_stationarity_converged else 0.0)
        else:
            # Backward-compatible fallback for historical summaries that only
            # persisted per-epoch projected residual diagnostics.
            projected_values = [
                s.get("projected_dual_residual")
                for s in window_summaries
                if s.get("projected_dual_residual") is not None
            ]
            if len(projected_values) >= convergence_window:
                mean_projected_dual_residual = float(np.mean(projected_values))
                criterion_value_history["dual_stationarity"].append(mean_projected_dual_residual)
                dual_stationarity_converged = mean_projected_dual_residual < thresholds["dual_stationarity"]
                fallback_status_history["dual_stationarity"].append(1.0 if dual_stationarity_converged else 0.0)
            else:
                criterion_value_history["dual_stationarity"].append(np.nan)
                fallback_status_history["dual_stationarity"].append(np.nan)

        if has_violation_fraction:
            vf_values = [s.get("violation_fraction") for s in window_summaries if s.get("violation_fraction") is not None]
            if len(vf_values) >= convergence_window:
                mean_violation_fraction = float(np.mean(vf_values))
                criterion_value_history["violation_fraction"].append(mean_violation_fraction)
                vf_converged = mean_violation_fraction < thresholds["violation_fraction"]
                fallback_status_history["violation_fraction"].append(1.0 if vf_converged else 0.0)
            else:
                criterion_value_history["violation_fraction"].append(np.nan)
                fallback_status_history["violation_fraction"].append(np.nan)

        if has_violation_fraction_model_avg:
            vf_model_values = [
                s.get("violation_fraction_on_model_avg_rates")
                for s in window_summaries
                if s.get("violation_fraction_on_model_avg_rates") is not None
            ]
            if len(vf_model_values) >= convergence_window:
                mean_violation_fraction_model_avg = float(np.mean(vf_model_values))
                criterion_value_history["violation_fraction_on_model_avg_rates"].append(
                    mean_violation_fraction_model_avg
                )
                vf_model_converged = (
                    mean_violation_fraction_model_avg < thresholds["violation_fraction_on_model_avg_rates"]
                )
                fallback_status_history["violation_fraction_on_model_avg_rates"].append(
                    1.0 if vf_model_converged else 0.0
                )
            else:
                # Matches trainer behavior: criterion is considered not converged
                # until enough non-None model-averaged values are available.
                criterion_value_history["violation_fraction_on_model_avg_rates"].append(np.nan)
                fallback_status_history["violation_fraction_on_model_avg_rates"].append(0.0)

        if has_mean_violation_slack_model_avg:
            slack_model_values = [
                s.get("model_avg_mean_violation_slack")
                for s in window_summaries
                if s.get("model_avg_mean_violation_slack") is not None
            ]
            if len(slack_model_values) >= convergence_window:
                mean_violation_slack_model_avg = float(np.mean(slack_model_values))
                criterion_value_history["mean_violation_slack_on_model_avg_rates"].append(
                    mean_violation_slack_model_avg
                )
                slack_model_converged = (
                    mean_violation_slack_model_avg
                    < thresholds["mean_violation_slack_on_model_avg_rates"]
                )
                fallback_status_history["mean_violation_slack_on_model_avg_rates"].append(
                    1.0 if slack_model_converged else 0.0
                )
            else:
                criterion_value_history["mean_violation_slack_on_model_avg_rates"].append(np.nan)
                fallback_status_history["mean_violation_slack_on_model_avg_rates"].append(0.0)

    if has_persisted_status:
        criterion_names = []
        for summary in summaries:
            for criterion_name in summary.get("convergence_criteria_status", {}):
                if criterion_name not in criterion_names:
                    criterion_names.append(criterion_name)
        criterion_status_history = {name: [] for name in criterion_names}
        for summary in summaries:
            criterion_statuses = summary.get("convergence_criteria_status", {})
            for name in criterion_names:
                if name in criterion_statuses:
                    criterion_status_history[name].append(1.0 if criterion_statuses[name] else 0.0)
                else:
                    criterion_status_history[name].append(np.nan)
    else:
        criterion_status_history = fallback_status_history

    criterion_thresholds = {name: thresholds.get(name, np.nan) for name in criterion_names}
    for name in criterion_names:
        if name not in criterion_value_history:
            criterion_value_history[name] = [np.nan] * len(summaries)
    if 'dual_stationarity' in criterion_value_history:
        # Prefer persisted convergence-time ergodic complementary-slackness
        # metric when available.
        if any('ergodic_complementary_slackness_abs' in s for s in summaries):
            criterion_value_history['dual_stationarity'] = [
                float(s['ergodic_complementary_slackness_abs'])
                if s.get('ergodic_complementary_slackness_abs') is not None
                else np.nan
                for s in summaries
            ]
        elif any('ergodic_complementary_slackness' in s for s in summaries):
            criterion_value_history['dual_stationarity'] = [
                abs(float(s['ergodic_complementary_slackness']))
                if s.get('ergodic_complementary_slackness') is not None
                else np.nan
                for s in summaries
            ]
        elif any('ergodic_projected_dual_residual' in s for s in summaries):
            # Backward compatibility for older runs.
            criterion_value_history['dual_stationarity'] = [
                float(s['ergodic_projected_dual_residual'])
                if s.get('ergodic_projected_dual_residual') is not None
                else np.nan
                for s in summaries
            ]

    return criterion_names, criterion_status_history, criterion_value_history, criterion_thresholds


# ---------------------------------------------------------------------------
# r_min helpers
# ---------------------------------------------------------------------------


def _extract_scalar_r_min_from_summary(summary: dict) -> Optional[float]:
    """Best-effort extraction of scalar r_min from an epoch summary."""
    direct = summary.get("r_min")
    if direct is not None:
        return float(direct)

    r_min_min = summary.get("r_min_min")
    r_min_max = summary.get("r_min_max")
    if r_min_min is None or r_min_max is None:
        return None

    try:
        min_v = float(r_min_min)
        max_v = float(r_min_max)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(min_v) or not np.isfinite(max_v):
        return None

    if bool(summary.get("r_min_is_scalar", False)) or np.isclose(
        min_v, max_v, rtol=1e-6, atol=1e-8
    ):
        return float(0.5 * (min_v + max_v))
    return None


def _infer_scalar_r_min_from_summaries(summaries: list[dict]) -> Optional[float]:
    """Return scalar r_min if summaries indicate homogeneous thresholds."""
    for summary in reversed(summaries):
        scalar = _extract_scalar_r_min_from_summary(summary)
        if scalar is not None:
            return scalar
    return None


def _draw_r_min_reference_line(ax, r_min_scalar: Optional[float]) -> None:
    """Overlay scalar r_min reference line on a rate plot axis."""
    if r_min_scalar is None:
        return
    ax.axhline(
        y=float(r_min_scalar),
        color='#2f4f4f',
        linestyle='--',
        linewidth=1.2,
        alpha=0.8,
        label=f'r_min={float(r_min_scalar):.4f}',
    )


# ---------------------------------------------------------------------------
# Constraint-profile series builder
# ---------------------------------------------------------------------------


def _collect_constraint_profile_series(
    summaries: list[dict],
) -> tuple[list[int], dict[int, Optional[str]], dict[int, dict[str, np.ndarray]]]:
    """
    Build dense per-profile metric arrays from epoch summaries.

    Returns
    -------
    profile_ids : list[int]
        Sorted profile IDs observed in summaries.
    profile_names : dict[int, str | None]
        Optional profile labels keyed by profile ID.
    series : dict[int, dict[str, np.ndarray]]
        Per-profile metric arrays aligned with summary order.
    """
    profile_ids_set = set()
    profile_names: dict[int, Optional[str]] = {}

    for summary in summaries:
        for entry in summary.get('constraint_profile_metrics', []) or []:
            try:
                profile_id = int(entry.get('constraint_profile_id'))
            except (TypeError, ValueError):
                continue
            profile_ids_set.add(profile_id)
            if profile_id not in profile_names and entry.get('constraint_profile_name') is not None:
                profile_names[profile_id] = str(entry.get('constraint_profile_name'))

    profile_ids = sorted(profile_ids_set)
    if not profile_ids:
        return [], {}, {}

    metric_keys = (
        'mean_rate',
        'avg_per_network_min_rate',
        'global_min_rate',
        'global_5th_percentile_rate',
        'violation_fraction',
        'mean_violation_slack',
        'r_min',
        'r_min_min',
        'r_min_max',
    )

    series: dict[int, dict[str, np.ndarray]] = {}
    n_epochs = len(summaries)
    for profile_id in profile_ids:
        profile_series: dict[str, np.ndarray] = {
            key: np.full(n_epochs, np.nan, dtype=np.float64) for key in metric_keys
        }
        series[profile_id] = profile_series

    for epoch_idx, summary in enumerate(summaries):
        entries = summary.get('constraint_profile_metrics', []) or []
        entries_by_profile: dict[int, dict] = {}
        for entry in entries:
            try:
                profile_id = int(entry.get('constraint_profile_id'))
            except (TypeError, ValueError):
                continue
            entries_by_profile[profile_id] = entry

        for profile_id in profile_ids:
            entry = entries_by_profile.get(profile_id)
            if entry is None:
                continue
            for key in metric_keys:
                value = entry.get(key)
                if value is None:
                    continue
                try:
                    series[profile_id][key][epoch_idx] = float(value)
                except (TypeError, ValueError):
                    continue

    return profile_ids, profile_names, series


# ---------------------------------------------------------------------------
# Main visualisation entry-points
# ---------------------------------------------------------------------------


def visualize_training_progress(epoch, output_dir="output", moving_avg_window=10, convergence_patience=5000, summaries=None):
    """Create visualization plots from epoch summaries.

    Parameters
    ----------
    summaries : list[dict] or None
        Pre-loaded list of epoch summary dicts.  When provided the JSONL file
        is not read, eliminating the O(epoch) disk I/O on every call.  When
        None (default) the file is read for backward compatibility.
    """
    if summaries is None:
        summaries_file = os.path.join(output_dir, "epoch_summaries.jsonl")
        if not os.path.exists(summaries_file):
            return
        with open(summaries_file, 'r') as f:
            summaries = [json.loads(line) for line in f]

    if not summaries:
        return

    training_cfg = _load_training_cfg_from_hydra(output_dir)

    # Calculate window sizes
    window_5M = 5 * moving_avg_window
    window_25M = 25 * moving_avg_window

    # Unpack summaries into per-metric arrays
    epochs = []
    losses = []
    grad_norms = []
    dual_means = []
    dual_stds = []
    power_percentiles = (5, 25, 50, 75, 90, 95, 99)
    normalized_power_percentiles = {pctl: [] for pctl in power_percentiles}
    rate_means = []
    per_network_rate_mins = []
    global_rate_mins = []
    rate_5th = []
    violation_fractions = []
    mean_violation_slacks = []
    model_avg_rate_means = []
    model_avg_per_network_mins = []
    model_avg_global_mins = []
    model_avg_rate_5th = []
    violation_fractions_on_model_avg_rates = []
    model_avg_mean_violation_slacks = []
    model_avg_rate_means_5M = []
    model_avg_per_network_mins_5M = []
    model_avg_global_mins_5M = []
    model_avg_rate_5th_5M = []
    violation_fractions_on_model_avg_rates_5M = []
    model_avg_mean_violation_slacks_5M = []
    model_avg_rate_means_25M = []
    model_avg_per_network_mins_25M = []
    model_avg_global_mins_25M = []
    model_avg_rate_5th_25M = []
    violation_fractions_on_model_avg_rates_25M = []
    model_avg_mean_violation_slacks_25M = []
    convergence_met_counts = []

    for summary in summaries:
        epochs.append(summary['epoch'])
        losses.append(summary.get('loss', np.nan))
        grad_norms.append(summary['gradient_norm'])
        dual_means.append(summary['mean_lambda'])
        dual_stds.append(summary['std_lambda'])
        for pctl in power_percentiles:
            value = summary.get(f'normalized_power_p{pctl}')
            normalized_power_percentiles[pctl].append(
                np.nan if value is None else float(value)
            )
        rate_means.append(summary['mean_rate'])
        per_network_rate_mins.append(summary['avg_per_network_min_rate'])
        global_rate_mins.append(summary['global_min_rate'])
        rate_5th.append(summary['global_5th_percentile_rate'])
        violation_fractions.append(summary['violation_fraction'])
        mean_violation_slacks.append(summary['mean_violation_slack'])
        convergence_met_counts.append(summary.get('convergence_met_count', 0))
        # Model-averaged metrics (if available)
        if 'model_avg_mean_rate' in summary:
            model_avg_rate_means.append(summary['model_avg_mean_rate'])
            model_avg_per_network_mins.append(summary['model_avg_avg_per_network_min_rate'])
            model_avg_global_mins.append(summary['model_avg_global_min_rate'])
            model_avg_rate_5th.append(summary['model_avg_global_5th_percentile_rate'])
            violation_fractions_on_model_avg_rates.append(summary['violation_fraction_on_model_avg_rates'])
            model_avg_mean_violation_slacks.append(summary.get('model_avg_mean_violation_slack'))
        if 'model_avg_mean_rate_5M' in summary:
            model_avg_rate_means_5M.append(summary['model_avg_mean_rate_5M'])
            model_avg_per_network_mins_5M.append(summary['model_avg_avg_per_network_min_rate_5M'])
            model_avg_global_mins_5M.append(summary['model_avg_global_min_rate_5M'])
            model_avg_rate_5th_5M.append(summary['model_avg_global_5th_percentile_rate_5M'])
            violation_fractions_on_model_avg_rates_5M.append(summary['violation_fraction_on_model_avg_rates_5M'])
            model_avg_mean_violation_slacks_5M.append(summary.get('model_avg_mean_violation_slack_5M'))
        if 'model_avg_mean_rate_25M' in summary:
            model_avg_rate_means_25M.append(summary['model_avg_mean_rate_25M'])
            model_avg_per_network_mins_25M.append(summary['model_avg_avg_per_network_min_rate_25M'])
            model_avg_global_mins_25M.append(summary['model_avg_global_min_rate_25M'])
            model_avg_rate_5th_25M.append(summary['model_avg_global_5th_percentile_rate_25M'])
            violation_fractions_on_model_avg_rates_25M.append(summary['violation_fraction_on_model_avg_rates_25M'])
            model_avg_mean_violation_slacks_25M.append(summary.get('model_avg_mean_violation_slack_25M'))

    # Build per-criterion convergence status traces (0=failed, 1=satisfied, NaN=not available).
    (
        criterion_names,
        criterion_status_history,
        criterion_value_history,
        criterion_thresholds,
    ) = _build_criterion_status_history(summaries, training_cfg)
    
    # Smooth curves
    smooth_window = min(50, len(epochs) // 10) if len(epochs) > 10 else 1
    losses_smooth = smooth_curve(losses, smooth_window)
    grad_norms_smooth = smooth_curve(grad_norms, smooth_window)
    rate_means_smooth = smooth_curve(rate_means, smooth_window)
    per_network_rate_mins_smooth = smooth_curve(per_network_rate_mins, smooth_window)
    global_rate_mins_smooth = smooth_curve(global_rate_mins, smooth_window)
    rate_5th_smooth = smooth_curve(rate_5th, smooth_window)
    violation_fractions_smooth = smooth_curve(violation_fractions, smooth_window)
    mean_violation_slacks_smooth = smooth_curve(mean_violation_slacks, smooth_window)
    scalar_r_min = _infer_scalar_r_min_from_summaries(summaries)
    
    # Create figure with 14 subplots (7x2 layout)
    fig, axes = plt.subplots(7, 2, figsize=(14, 36))
    fig.suptitle(f'Training Progress - Epoch {epoch}', fontsize=14, fontweight='bold')
    
    # 1) Gradient norms (log-scale) with smoothing
    ax = axes[0, 0]
    ax.plot(epochs, grad_norms, '-', linewidth=1, color='#1f77b4', alpha=0.3, label='Per-epoch')
    ax.plot(epochs, grad_norms_smooth, '-', linewidth=2, color='#1f77b4', label='Smoothed')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gradient Norm')
    ax.set_title('Gradient Norm Evolution')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2) Dual multiplier mean and std
    ax = axes[0, 1]
    ax.plot(epochs, dual_means, '-', linewidth=2, label='Mean', color='#ff7f0e')
    ax.plot(epochs, dual_stds, '-', linewidth=2, label='Std', color='#2ca02c')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Dual Multiplier')
    ax.set_title('Dual Multiplier Statistics')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3) Lagrangian loss evolution
    ax = axes[1, 0]
    finite_losses = np.asarray(losses, dtype=np.float64)
    if np.isfinite(finite_losses).any():
        ax.plot(epochs, losses, '-', linewidth=1, color='#bc5090', alpha=0.3, label='Per-epoch')
        if np.isfinite(finite_losses).all():
            ax.plot(epochs, losses_smooth, '-', linewidth=2, color='#bc5090', label='Smoothed')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Lagrangian Loss')
        ax.set_title('Lagrangian Loss Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Lagrangian loss unavailable', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Lagrangian Loss Evolution')
        ax.axis('off')

    # 4) Normalized transmit-power percentiles
    ax = axes[1, 1]
    plotted_percentile_count = 0
    percentile_colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(power_percentiles)))
    for color, pctl in zip(percentile_colors, power_percentiles):
        series = np.asarray(normalized_power_percentiles[pctl], dtype=np.float64)
        if not np.isfinite(series).any():
            continue
        ax.plot(epochs, series, '-', linewidth=1.8, color=color, label=f'P{pctl}')
        plotted_percentile_count += 1
    if plotted_percentile_count > 0:
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Normalized Transmit Power (p / P_max)')
        ax.set_title('Normalized Transmit Power Percentiles')
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(
            0.5,
            0.5,
            'Normalized transmit-power percentiles unavailable',
            ha='center',
            va='center',
            transform=ax.transAxes,
        )
        ax.set_title('Normalized Transmit Power Percentiles')
        ax.axis('off')

    # 5) Ergodic rates with smoothing
    ax = axes[2, 0]
    ax.plot(epochs, rate_means, '-', linewidth=1, color='#d62728', alpha=0.3)
    ax.plot(epochs, per_network_rate_mins, '-', linewidth=1, color='#9467bd', alpha=0.3)
    ax.plot(epochs, global_rate_mins, '-', linewidth=1, color='#7f3c8d', alpha=0.3)
    ax.plot(epochs, rate_5th, '-', linewidth=1, color='#8c564b', alpha=0.3)
    ax.plot(epochs, rate_means_smooth, '-', linewidth=2, label='Mean', color='#d62728')
    ax.plot(epochs, per_network_rate_mins_smooth, '-', linewidth=2, label='Per-Net Min (avg)', color='#9467bd')
    ax.plot(epochs, global_rate_mins_smooth, '-', linewidth=2, label='Global Min (worst)', color='#7f3c8d')
    ax.plot(epochs, rate_5th_smooth, '-', linewidth=2, label='5th Pctl', color='#8c564b')
    _draw_r_min_reference_line(ax, scalar_r_min)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Rate (bps/Hz)')
    ax.set_title('Ergodic Rate Statistics')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 6) Ergodic violation fractions with smoothing and mean violation slack
    ax = axes[2, 1]
    ax.plot(epochs, violation_fractions, '-', linewidth=1, color='#e377c2', alpha=0.3)
    ax.plot(epochs, violation_fractions_smooth, '-', linewidth=2, color='#e377c2', label='Violation Fraction (smoothed)')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Violation Fraction', color='#e377c2')
    ax.set_title('Constraint Violation Fraction & Mean Violation Slack')
    ax.tick_params(axis='y', labelcolor='#e377c2')
    ax.grid(True, alpha=0.3)

    # Add secondary y-axis for mean violation slack
    ax2 = ax.twinx()
    ax2.plot(epochs, mean_violation_slacks, '-', linewidth=1, color='#17becf', alpha=0.3)
    ax2.plot(epochs, mean_violation_slacks_smooth, '-', linewidth=2, color='#17becf', label='Mean Violation Slack (smoothed)')
    ax2.set_ylabel('Mean Violation Slack (bits/s/Hz)', color='#17becf')
    ax2.tick_params(axis='y', labelcolor='#17becf')
    ax2.grid(True, alpha=0.3)

    # Combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    # 7) Model-averaged rates (M=moving_avg_window)
    ax = axes[3, 0]
    if model_avg_rate_means:
        # Use actual epoch numbers starting from moving_avg_window
        available_epochs = epochs[-len(model_avg_rate_means):]
        # Apply smoothing
        ax.plot(available_epochs, model_avg_rate_means, '-', linewidth=1, color='#d62728', alpha=0.3)
        ax.plot(available_epochs, model_avg_per_network_mins, '-', linewidth=1, color='#9467bd', alpha=0.3)
        ax.plot(available_epochs, model_avg_global_mins, '-', linewidth=1, color='#7f3c8d', alpha=0.3)
        ax.plot(available_epochs, model_avg_rate_5th, '-', linewidth=1, color='#8c564b', alpha=0.3)
        ax.plot(available_epochs, smooth_curve(model_avg_rate_means, smooth_window), '-', linewidth=2, label='Mean', color='#d62728')
        ax.plot(available_epochs, smooth_curve(model_avg_per_network_mins, smooth_window), '-', linewidth=2, label='Per-Net Min (avg)', color='#9467bd')
        ax.plot(available_epochs, smooth_curve(model_avg_global_mins, smooth_window), '-', linewidth=2, label='Global Min (worst)', color='#7f3c8d')
        ax.plot(available_epochs, smooth_curve(model_avg_rate_5th, smooth_window), '-', linewidth=2, label='5th Pctl', color='#8c564b')
        _draw_r_min_reference_line(ax, scalar_r_min)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Rate (bps/Hz)')
        ax.set_title(f'Model-Averaged Ergodic Rate Statistics (M={moving_avg_window})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, f'Not available (need ≥{moving_avg_window} epochs)', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'Model-Averaged Ergodic Rate Statistics (M={moving_avg_window})')
    
    # 8) Model-averaged violation fractions (M=moving_avg_window)
    ax = axes[3, 1]
    if violation_fractions_on_model_avg_rates:
        # Use actual epoch numbers starting from moving_avg_window
        available_epochs = epochs[-len(violation_fractions_on_model_avg_rates):]
        ax.plot(available_epochs, violation_fractions_on_model_avg_rates, '-', linewidth=1, color='#e377c2', alpha=0.3)
        ax.plot(
            available_epochs,
            smooth_curve(violation_fractions_on_model_avg_rates, smooth_window),
            '-',
            linewidth=2,
            color='#e377c2',
            label='Violation Fraction on Model-Avg Rates (smoothed)'
        )
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Violation Fraction', color='#e377c2')
        ax.set_title(f'Violation Fraction on Model-Avg Rates & Mean Violation Slack (M={moving_avg_window})')
        ax.tick_params(axis='y', labelcolor='#e377c2')
        ax.grid(True, alpha=0.3)
        
        # Add secondary y-axis for model-averaged mean violation slack when available.
        if model_avg_mean_violation_slacks and all(v is not None for v in model_avg_mean_violation_slacks):
            ax2 = ax.twinx()
            ax2.plot(available_epochs, model_avg_mean_violation_slacks, '-', linewidth=1, color='#17becf', alpha=0.3)
            ax2.plot(
                available_epochs,
                smooth_curve(model_avg_mean_violation_slacks, smooth_window),
                '-',
                linewidth=2,
                color='#17becf',
                label='Mean Violation Slack (smoothed)'
            )
            ax2.set_ylabel('Mean Violation Slack (bits/s/Hz)', color='#17becf')
            ax2.tick_params(axis='y', labelcolor='#17becf')
            ax2.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, f'Not available (need ≥{moving_avg_window} epochs)', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'Violation Fraction on Model-Avg Rates (M={moving_avg_window})')
    
    # 9) Model-averaged ergodic rates (5*M)
    ax = axes[4, 0]
    if model_avg_rate_means_5M:
        # Use actual epoch numbers starting from window_5M
        available_epochs_5M = epochs[-len(model_avg_rate_means_5M):]
        # Apply smoothing
        ax.plot(available_epochs_5M, model_avg_rate_means_5M, '-', linewidth=1, color='#d62728', alpha=0.3)
        ax.plot(available_epochs_5M, model_avg_per_network_mins_5M, '-', linewidth=1, color='#9467bd', alpha=0.3)
        ax.plot(available_epochs_5M, model_avg_global_mins_5M, '-', linewidth=1, color='#7f3c8d', alpha=0.3)
        ax.plot(available_epochs_5M, model_avg_rate_5th_5M, '-', linewidth=1, color='#8c564b', alpha=0.3)
        ax.plot(available_epochs_5M, smooth_curve(model_avg_rate_means_5M, smooth_window), '-', linewidth=2, label='Mean', color='#d62728')
        ax.plot(available_epochs_5M, smooth_curve(model_avg_per_network_mins_5M, smooth_window), '-', linewidth=2, label='Per-Net Min (avg)', color='#9467bd')
        ax.plot(available_epochs_5M, smooth_curve(model_avg_global_mins_5M, smooth_window), '-', linewidth=2, label='Global Min (worst)', color='#7f3c8d')
        ax.plot(available_epochs_5M, smooth_curve(model_avg_rate_5th_5M, smooth_window), '-', linewidth=2, label='5th Pctl', color='#8c564b')
        _draw_r_min_reference_line(ax, scalar_r_min)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Rate (bps/Hz)')
        ax.set_title(f'Model-Averaged Ergodic Rate Statistics (5M={window_5M})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, f'Not available (need ≥{window_5M} epochs)', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'Model-Averaged Ergodic Rate Statistics (5M={window_5M})')
    
    # 10) Model-averaged violation fractions (5*M)
    ax = axes[4, 1]
    if violation_fractions_on_model_avg_rates_5M:
        # Use actual epoch numbers starting from window_5M
        available_epochs_5M = epochs[-len(violation_fractions_on_model_avg_rates_5M):]
        ax.plot(available_epochs_5M, violation_fractions_on_model_avg_rates_5M, '-', linewidth=1, color='#e377c2', alpha=0.3)
        ax.plot(
            available_epochs_5M,
            smooth_curve(violation_fractions_on_model_avg_rates_5M, smooth_window),
            '-',
            linewidth=2,
            color='#e377c2',
            label='Violation Fraction on Model-Avg Rates (smoothed)'
        )
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Violation Fraction', color='#e377c2')
        ax.set_title(f'Violation Fraction on Model-Avg Rates & Mean Violation Slack (5M={window_5M})')
        ax.tick_params(axis='y', labelcolor='#e377c2')
        ax.grid(True, alpha=0.3)
        
        # Add secondary y-axis for model-averaged mean violation slack when available.
        if model_avg_mean_violation_slacks_5M and all(v is not None for v in model_avg_mean_violation_slacks_5M):
            ax2 = ax.twinx()
            ax2.plot(available_epochs_5M, model_avg_mean_violation_slacks_5M, '-', linewidth=1, color='#17becf', alpha=0.3)
            ax2.plot(
                available_epochs_5M,
                smooth_curve(model_avg_mean_violation_slacks_5M, smooth_window),
                '-',
                linewidth=2,
                color='#17becf',
                label='Mean Violation Slack (smoothed)'
            )
            ax2.set_ylabel('Mean Violation Slack (bits/s/Hz)', color='#17becf')
            ax2.tick_params(axis='y', labelcolor='#17becf')
            ax2.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, f'Not available (need ≥{window_5M} epochs)', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'Violation Fraction on Model-Avg Rates (5M={window_5M})')
    
    # 11) Model-averaged ergodic rates (25*M)
    ax = axes[5, 0]
    if model_avg_rate_means_25M:
        # Use actual epoch numbers starting from window_25M
        available_epochs_25M = epochs[-len(model_avg_rate_means_25M):]
        # Apply smoothing
        ax.plot(available_epochs_25M, model_avg_rate_means_25M, '-', linewidth=1, color='#d62728', alpha=0.3)
        ax.plot(available_epochs_25M, model_avg_per_network_mins_25M, '-', linewidth=1, color='#9467bd', alpha=0.3)
        ax.plot(available_epochs_25M, model_avg_global_mins_25M, '-', linewidth=1, color='#7f3c8d', alpha=0.3)
        ax.plot(available_epochs_25M, model_avg_rate_5th_25M, '-', linewidth=1, color='#8c564b', alpha=0.3)
        ax.plot(available_epochs_25M, smooth_curve(model_avg_rate_means_25M, smooth_window), '-', linewidth=2, label='Mean', color='#d62728')
        ax.plot(available_epochs_25M, smooth_curve(model_avg_per_network_mins_25M, smooth_window), '-', linewidth=2, label='Per-Net Min (avg)', color='#9467bd')
        ax.plot(available_epochs_25M, smooth_curve(model_avg_global_mins_25M, smooth_window), '-', linewidth=2, label='Global Min (worst)', color='#7f3c8d')
        ax.plot(available_epochs_25M, smooth_curve(model_avg_rate_5th_25M, smooth_window), '-', linewidth=2, label='5th Pctl', color='#8c564b')
        _draw_r_min_reference_line(ax, scalar_r_min)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Ergodic Rate (bps/Hz)')
        ax.set_title(f'Model-Averaged Ergodic Rate Statistics (25M={window_25M})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, f'Not available (need ≥{window_25M} epochs)', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'Model-Averaged Ergodic Rate Statistics (25M={window_25M})')
    
    # 12) Model-averaged violation fractions (25*M)
    ax = axes[5, 1]
    if violation_fractions_on_model_avg_rates_25M:
        # Use actual epoch numbers starting from window_25M
        available_epochs_25M = epochs[-len(violation_fractions_on_model_avg_rates_25M):]
        ax.plot(available_epochs_25M, violation_fractions_on_model_avg_rates_25M, '-', linewidth=1, color='#e377c2', alpha=0.3)
        ax.plot(
            available_epochs_25M,
            smooth_curve(violation_fractions_on_model_avg_rates_25M, smooth_window),
            '-',
            linewidth=2,
            color='#e377c2',
            label='Violation Fraction on Model-Avg Rates (smoothed)'
        )
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Violation Fraction', color='#e377c2')
        ax.set_title(f'Violation Fraction on Model-Avg Rates & Mean Violation Slack (25M={window_25M})')
        ax.tick_params(axis='y', labelcolor='#e377c2')
        ax.grid(True, alpha=0.3)
        
        # Add secondary y-axis for model-averaged mean violation slack when available.
        if model_avg_mean_violation_slacks_25M and all(v is not None for v in model_avg_mean_violation_slacks_25M):
            ax2 = ax.twinx()
            ax2.plot(available_epochs_25M, model_avg_mean_violation_slacks_25M, '-', linewidth=1, color='#17becf', alpha=0.3)
            ax2.plot(
                available_epochs_25M,
                smooth_curve(model_avg_mean_violation_slacks_25M, smooth_window),
                '-',
                linewidth=2,
                color='#17becf',
                label='Mean Violation Slack (smoothed)'
            )
            ax2.set_ylabel('Mean Violation Slack (bits/s/Hz)', color='#17becf')
            ax2.tick_params(axis='y', labelcolor='#17becf')
            ax2.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, f'Not available (need ≥{window_25M} epochs)', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'Violation Fraction on Model-Avg Rates (25M={window_25M})')
    
    # 13) Convergence tracking
    ax = axes[6, 0]
    ax.plot(epochs, convergence_met_counts, '-', linewidth=2, color='#2ca02c', label='Consecutive Convergence Count')
    ax.axhline(y=convergence_patience, color='red', linestyle='--', linewidth=2, label=f'Patience Threshold ({convergence_patience})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Consecutive Epochs')
    ax.set_title('Convergence Tracking')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, max(max(convergence_met_counts) if convergence_met_counts else 1, convergence_patience) * 1.1])
    
    # 14) Individual convergence criteria status summary
    ax = axes[6, 1]
    if criterion_names:
        ax.text(
            0.5,
            0.5,
            (
                "Convergence criteria status is shown on page 2\n"
                "(stacked subplots, shared x-axis,\n"
                "and value/threshold overlays)"
            ),
            ha='center',
            va='center',
            transform=ax.transAxes,
            fontsize=11,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
        )
        ax.set_title('Convergence Criteria Status')
        ax.axis('off')
    else:
        ax.text(0.5, 0.5, 'Criterion status unavailable', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Convergence Criteria Status by Epoch')
        ax.axis('off')

    status_fig = None
    if criterion_names:
        status_fig_height = max(8, 2.8 * len(criterion_names))
        status_fig, status_axes = plt.subplots(
            len(criterion_names),
            1,
            figsize=(14, status_fig_height),
            sharex=True,
        )
        if len(criterion_names) == 1:
            status_axes = [status_axes]

        status_fig.suptitle(
            'Convergence Criteria Status by Epoch (0=Failed, 1=Satisfied)',
            fontsize=13,
            fontweight='bold',
        )
        colors = plt.cm.Set2(np.linspace(0.15, 0.95, len(criterion_names)))
        for idx, criterion_name in enumerate(criterion_names):
            status_ax = status_axes[idx]
            status_line = status_ax.plot(
                epochs,
                criterion_status_history[criterion_name],
                '-',
                drawstyle='steps-post',
                linewidth=1.6,
                color=colors[idx],
                label='Status',
            )
            status_ax.set_yticks([0, 1])
            status_ax.set_yticklabels(['Fail', 'OK'])
            status_ax.set_ylim([-0.1, 1.1])
            status_ax.set_ylabel(f'{criterion_name}\nstatus', fontsize=9)
            status_ax.grid(True, alpha=0.3)

            value_ax = status_ax.twinx()
            value_trace = np.asarray(criterion_value_history[criterion_name], dtype=float)
            finite_value_trace = np.where(np.isfinite(value_trace), value_trace, np.nan)
            value_line = value_ax.plot(
                epochs,
                finite_value_trace,
                '-',
                linewidth=1.2,
                color='#1f77b4',
                alpha=0.9,
                label='Value',
            )
            value_ax.set_ylabel('Value', color='#1f77b4', fontsize=8)
            value_ax.tick_params(axis='y', labelcolor='#1f77b4')

            threshold = criterion_thresholds.get(criterion_name, np.nan)
            threshold_line = None
            threshold_label = "n/a"
            if np.isfinite(threshold):
                threshold_line = value_ax.axhline(
                    y=float(threshold),
                    color='#d62728',
                    linestyle='--',
                    linewidth=1.2,
                    label='Threshold',
                )
                threshold_label = f"{threshold:.4g}"
            elif np.isinf(threshold):
                threshold_label = "inf"
                value_ax.text(
                    0.99,
                    0.90,
                    "threshold=inf (no finite line)",
                    transform=value_ax.transAxes,
                    ha='right',
                    va='top',
                    fontsize=7.5,
                    color='#d62728',
                )

            status_ax.set_title(f"{criterion_name} (threshold={threshold_label})", loc='left', fontsize=10)

            legend_handles = [status_line[0], value_line[0]]
            legend_labels = ['Status', 'Value']
            if threshold_line is not None:
                legend_handles.append(threshold_line)
                legend_labels.append('Threshold')
            status_ax.legend(legend_handles, legend_labels, loc='upper right', fontsize=7, framealpha=0.85)

        status_axes[-1].set_xlabel('Epoch')

    # Save figure
    plot_path = os.path.join(output_dir, 'training_progress.pdf')
    with PdfPages(plot_path) as pdf:
        fig.tight_layout(rect=[0, 0, 1, 0.985])
        pdf.savefig(fig, dpi=150, bbox_inches='tight')
        if status_fig is not None:
            status_fig.tight_layout(rect=[0, 0, 1, 0.975])
            pdf.savefig(status_fig, dpi=150, bbox_inches='tight')
    plt.close(fig)
    if status_fig is not None:
        plt.close(status_fig)
    
    logger.info(f"Saved training progress plot to {plot_path}")


def visualize_training_progress_by_profile(
    epoch,
    output_dir="output",
    moving_avg_window=10,
    summaries=None,
):
    """
    Save profile-specific training progress views with hybrid global/profile panels.

    Profile-independent traces (gradient norm, dual stats, convergence count) are
    reused globally, while rate/violation panels are computed per profile.
    """
    if summaries is None:
        summaries_file = os.path.join(output_dir, "epoch_summaries.jsonl")
        if not os.path.exists(summaries_file):
            return
        with open(summaries_file, 'r') as f:
            summaries = [json.loads(line) for line in f]

    if not summaries:
        return

    profile_ids, profile_names, profile_series = _collect_constraint_profile_series(summaries)
    if not profile_ids:
        return

    epochs = np.asarray([s.get('epoch', idx) for idx, s in enumerate(summaries)], dtype=np.int64)
    grad_norms = np.asarray([s.get('gradient_norm', np.nan) for s in summaries], dtype=np.float64)
    dual_means = np.asarray([s.get('mean_lambda', np.nan) for s in summaries], dtype=np.float64)
    dual_stds = np.asarray([s.get('std_lambda', np.nan) for s in summaries], dtype=np.float64)
    convergence_counts = np.asarray(
        [s.get('convergence_met_count', np.nan) for s in summaries], dtype=np.float64
    )
    smooth_window = min(50, len(epochs) // 10) if len(epochs) > 10 else 1

    plot_path = os.path.join(output_dir, 'training_progress_by_profile.pdf')
    with PdfPages(plot_path) as pdf:
        for profile_id in profile_ids:
            profile_name = profile_names.get(profile_id)
            profile_label = (
                f"{profile_id} ({profile_name})"
                if profile_name is not None
                else str(profile_id)
            )
            metrics = profile_series[profile_id]

            fig, axes = plt.subplots(3, 2, figsize=(14, 16))
            fig.suptitle(
                f'Training Progress by Constraint Profile {profile_label} - Epoch {epoch}',
                fontsize=13,
                fontweight='bold',
            )

            # 1) Global gradient norm (profile-independent).
            ax = axes[0, 0]
            ax.plot(epochs, grad_norms, '-', linewidth=1.0, color='#1f77b4', alpha=0.35, label='Per-epoch')
            ax.plot(
                epochs,
                smooth_curve(grad_norms, smooth_window),
                '-',
                linewidth=2.0,
                color='#1f77b4',
                label='Smoothed',
            )
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Gradient Norm')
            ax.set_title('Global Gradient Norm (shared across profiles)')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

            # 2) Global dual stats (profile-independent).
            ax = axes[0, 1]
            ax.plot(epochs, dual_means, '-', linewidth=2, color='#ff7f0e', label='Mean λ')
            ax.plot(epochs, dual_stds, '-', linewidth=2, color='#2ca02c', label='Std λ')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Dual Multiplier')
            ax.set_title('Global Dual Statistics (shared across profiles)')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

            # 3) Profile-specific rate panel.
            ax = axes[1, 0]
            ax.plot(epochs, metrics['mean_rate'], '-', linewidth=1.0, color='#d62728', alpha=0.35)
            ax.plot(epochs, metrics['avg_per_network_min_rate'], '-', linewidth=1.0, color='#9467bd', alpha=0.35)
            ax.plot(epochs, metrics['global_min_rate'], '-', linewidth=1.0, color='#7f3c8d', alpha=0.35)
            ax.plot(epochs, metrics['global_5th_percentile_rate'], '-', linewidth=1.0, color='#8c564b', alpha=0.35)
            ax.plot(
                epochs,
                smooth_curve(metrics['mean_rate'], smooth_window),
                '-',
                linewidth=2.0,
                color='#d62728',
                label='Mean',
            )
            ax.plot(
                epochs,
                smooth_curve(metrics['avg_per_network_min_rate'], smooth_window),
                '-',
                linewidth=2.0,
                color='#9467bd',
                label='Per-Net Min (avg)',
            )
            ax.plot(
                epochs,
                smooth_curve(metrics['global_min_rate'], smooth_window),
                '-',
                linewidth=2.0,
                color='#7f3c8d',
                label='Global Min (worst)',
            )
            ax.plot(
                epochs,
                smooth_curve(metrics['global_5th_percentile_rate'], smooth_window),
                '-',
                linewidth=2.0,
                color='#8c564b',
                label='5th Pctl',
            )
            profile_scalar_r_min = None
            for raw_r_min in metrics['r_min'][::-1]:
                if np.isfinite(raw_r_min):
                    profile_scalar_r_min = float(raw_r_min)
                    break
            if profile_scalar_r_min is None:
                r_min_min = metrics['r_min_min']
                r_min_max = metrics['r_min_max']
                finite = np.isfinite(r_min_min) & np.isfinite(r_min_max)
                if np.any(finite):
                    idx = np.where(finite)[0][-1]
                    if np.isclose(r_min_min[idx], r_min_max[idx], rtol=1e-6, atol=1e-8):
                        profile_scalar_r_min = float(0.5 * (r_min_min[idx] + r_min_max[idx]))
            _draw_r_min_reference_line(ax, profile_scalar_r_min)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Rate (bps/Hz)')
            ax.set_title(f'Profile-{profile_label} Rate Statistics')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

            # 4) Profile-specific violation panel.
            ax = axes[1, 1]
            ax.plot(
                epochs,
                metrics['violation_fraction'],
                '-',
                linewidth=1.0,
                color='#e377c2',
                alpha=0.35,
            )
            ax.plot(
                epochs,
                smooth_curve(metrics['violation_fraction'], smooth_window),
                '-',
                linewidth=2.0,
                color='#e377c2',
                label='Violation Fraction (smoothed)',
            )
            ax.axhline(y=0.0, color='black', linestyle='--', linewidth=1.0)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Violation Fraction', color='#e377c2')
            ax.tick_params(axis='y', labelcolor='#e377c2')
            ax.grid(True, alpha=0.3)
            ax.set_title(f'Profile-{profile_label} Violation Fraction & Mean Slack')
            ax2 = ax.twinx()
            ax2.plot(
                epochs,
                metrics['mean_violation_slack'],
                '-',
                linewidth=1.0,
                color='#17becf',
                alpha=0.35,
            )
            ax2.plot(
                epochs,
                smooth_curve(metrics['mean_violation_slack'], smooth_window),
                '-',
                linewidth=2.0,
                color='#17becf',
                label='Mean Violation Slack (smoothed)',
            )
            ax2.set_ylabel('Mean Violation Slack (bits/s/Hz)', color='#17becf')
            ax2.tick_params(axis='y', labelcolor='#17becf')
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

            # 5) Global convergence count (profile-independent).
            ax = axes[2, 0]
            ax.plot(
                epochs,
                convergence_counts,
                '-',
                linewidth=2.0,
                color='#2ca02c',
                label='Consecutive Convergence Count',
            )
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Consecutive Epochs')
            ax.set_title('Global Convergence Count (shared across profiles)')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

            # 6) Hybrid note and r_min range diagnostics.
            ax = axes[2, 1]
            finite_min = metrics['r_min_min'][np.isfinite(metrics['r_min_min'])]
            finite_max = metrics['r_min_max'][np.isfinite(metrics['r_min_max'])]
            r_min_range_text = "n/a"
            if finite_min.size and finite_max.size:
                r_min_range_text = f"[{finite_min.min():.4f}, {finite_max.max():.4f}]"
            ax.text(
                0.03,
                0.85,
                (
                    "Hybrid view:\n"
                    "- Top row and convergence panel are global (profile-independent).\n"
                    "- Middle row panels are computed from this profile only.\n"
                    f"- r_min range for this profile: {r_min_range_text}\n"
                    f"- Moving-average window: {moving_avg_window}"
                ),
                transform=ax.transAxes,
                va='top',
                ha='left',
                fontsize=10,
            )
            ax.set_axis_off()

            fig.tight_layout(rect=[0, 0, 1, 0.97])
            pdf.savefig(fig, dpi=150, bbox_inches='tight')
            plt.close(fig)

    logger.info("Saved profile-specific training progress plot to %s", plot_path)


def visualize_dual_multipliers(
    output_dir='output',
    top_k=10,
    network_id=0,
    all_entries=None,
    network_label=None,
):
    """Visualize evolution of dual multipliers for top-k and bottom-k receivers."""
    if all_entries is not None:
        entries = [e for e in all_entries if e.get('network_id', 0) == network_id]
    else:
        dual_history_file = os.path.join(output_dir, "dual_history.jsonl")
        if not os.path.exists(dual_history_file):
            return
        entries = []
        with open(dual_history_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                if data.get('network_id', 0) == network_id:
                    entries.append(data)

    if not entries:
        return

    epochs = [e['epoch'] for e in entries]
    all_duals = [np.array(e['dual_multipliers']) for e in entries]
    
    # Find top-k and bottom-k receivers based on final dual values
    final_duals = all_duals[-1]
    top_k_indices = np.argsort(final_duals)[-top_k:][::-1]  # Largest duals (most constrained)
    bottom_k_indices = np.argsort(final_duals)[:top_k]  # Smallest duals (least constrained)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    title_suffix = f"{network_label}" if network_label is not None else f"Network {network_id}"
    
    # Plot top-k (most constrained)
    colors_top = plt.cm.Reds(np.linspace(0.4, 0.9, top_k))
    for i, idx in enumerate(top_k_indices):
        dual_evolution = [duals[idx] for duals in all_duals]
        ax1.plot(epochs, dual_evolution, '-', linewidth=1.5, color=colors_top[i],
                label=f'Receiver {idx} (final λ={final_duals[idx]:.2f})')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Dual Multiplier (λ)')
    ax1.set_title(f'Dual Multiplier Evolution: Top-{top_k} Most Constrained Receivers ({title_suffix})')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot bottom-k (least constrained)
    colors_bottom = plt.cm.Blues(np.linspace(0.4, 0.9, top_k))
    for i, idx in enumerate(bottom_k_indices):
        dual_evolution = [duals[idx] for duals in all_duals]
        ax2.plot(epochs, dual_evolution, '-', linewidth=1.5, color=colors_bottom[i],
                label=f'Receiver {idx} (final λ={final_duals[idx]:.2f})')
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dual Multiplier (λ)')
    ax2.set_title(f'Dual Multiplier Evolution: Bottom-{top_k} Least Constrained Receivers ({title_suffix})')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure with network_id in filename
    plot_path = os.path.join(output_dir, f'dual_multipliers_network_{network_id}.pdf')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved dual multiplier plot to {plot_path}")


def visualize_power_allocations(
    output_dir='output',
    P_max=1.0,
    r_min=None,
    top_k=5,
    network_id=0,
    all_entries=None,
    metadata_entries=None,
    network_label=None,
):
    """Visualize evolution of power allocations for top-k and bottom-k transmitters.

    Top/bottom transmitters are determined by their corresponding receivers' constraint violations.

    Parameters
    ----------
    output_dir : str
        Directory containing primal_history.jsonl (used only when all_entries is None)
    P_max : float
        Maximum power for normalization
    r_min : float or array-like
        Minimum rate constraint used to compute per-receiver violations for
        ranking transmitters. Scalars are broadcast to all
        receivers; vectors must have length num_receivers.
    top_k : int
        Number of top (most constrained) and bottom (least constrained) transmitters to plot
    network_id : int
        Network ID to visualize (default: 0)
    all_entries : list or None
        In-memory epoch entries (dicts with epoch/network_id/power_allocations/rates).
        When provided, skips reading primal_history.jsonl from disk.
    metadata_entries : dict or None
        In-memory metadata: {network_id: associations_as_list}.
        Used together with all_entries to avoid disk reads.
    network_label : str or None
        Optional human-readable label shown in subplot titles.
    """
    if all_entries is not None:
        net_entries = [e for e in all_entries if e.get('network_id') == network_id]
        associations = (
            np.array(metadata_entries[network_id])
            if metadata_entries and network_id in metadata_entries
            else None
        )
    else:
        primal_history_file = os.path.join(output_dir, "primal_history.jsonl")
        if not os.path.exists(primal_history_file):
            return
        net_entries = []
        associations = None
        with open(primal_history_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                if data.get('type') == 'metadata':
                    if data.get('network_id') == network_id:
                        associations = np.array(data['associations'])
                    continue
                if data['network_id'] == network_id:
                    net_entries.append(data)

    if not net_entries:
        return

    if associations is None:
        logger.warning(
            "visualize_power_allocations: associations metadata not found for "
            "network %d – skipping power-allocation plot.",
            network_id,
        )
        return

    associations = np.asarray(associations)
    if associations.ndim != 2:
        logger.warning(
            "visualize_power_allocations: expected 2D associations matrix for "
            "network %d, got shape %s – skipping power-allocation plot.",
            network_id,
            associations.shape,
        )
        return
    if associations.shape[0] == 0 or associations.shape[1] == 0:
        logger.warning(
            "visualize_power_allocations: empty associations matrix for "
            "network %d (shape %s) – skipping power-allocation plot.",
            network_id,
            associations.shape,
        )
        return

    epochs = [e['epoch'] for e in net_entries]
    all_powers = [np.array(e['power_allocations']) for e in net_entries]
    all_rates = [np.array(e['rates']) for e in net_entries]

    if r_min is None:
        raise ValueError(
            "r_min must be provided to visualize_power_allocations. "
            "Pass the actual training threshold(s) for this network."
        )

    # Compute constraint violations per receiver (based on final rates)
    final_rates = all_rates[-1]
    if np.isscalar(r_min):
        r_min_vec = np.full_like(final_rates, float(r_min), dtype=np.float64)
    else:
        r_min_vec = np.asarray(r_min, dtype=np.float64).reshape(-1)
        if r_min_vec.shape[0] != final_rates.shape[0]:
            raise ValueError(
                f"r_min length {r_min_vec.shape[0]} does not match "
                f"num_receivers {final_rates.shape[0]} for network {network_id}."
            )
    violations = np.maximum(0.0, r_min_vec - final_rates)  # (n,)
    if associations.shape[1] != final_rates.shape[0]:
        logger.warning(
            "visualize_power_allocations: associations receiver dimension (%d) "
            "does not match rate dimension (%d) for network %d – skipping "
            "power-allocation plot.",
            associations.shape[1],
            final_rates.shape[0],
            network_id,
        )
        return
    
    # For each transmitter, find its corresponding receiver(s) and sum their violations
    m, _ = associations.shape
    transmitter_violations = np.zeros(m)
    
    for tx in range(m):
        receivers_served = np.where(associations[tx, :] == 1)[0]
        if len(receivers_served) > 0:
            transmitter_violations[tx] = violations[receivers_served].sum()
    
    # Find top-k (most constrained) and bottom-k (least constrained) transmitters
    top_k_indices = np.argsort(transmitter_violations)[-top_k:][::-1]  # Highest violations
    bottom_k_indices = np.argsort(transmitter_violations)[:top_k]  # Lowest violations
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    title_suffix = f"{network_label}" if network_label is not None else f"Network {network_id}"
    
    # Normalize powers by P_max
    all_powers_normalized = [powers / P_max for powers in all_powers]
    
    # Plot top-k (most constrained)
    colors_top = plt.cm.Reds(np.linspace(0.4, 0.9, top_k))
    for i, tx_idx in enumerate(top_k_indices):
        power_evolution = [powers[tx_idx] for powers in all_powers_normalized]
        receivers_served = np.where(associations[tx_idx, :] == 1)[0]
        rx_str = ','.join(map(str, receivers_served))
        ax1.plot(epochs, power_evolution, '-', linewidth=1.5, color=colors_top[i],
                label=f'TX {tx_idx} → RX [{rx_str}] (violation={transmitter_violations[tx_idx]:.3f})')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Normalized Power (p/P_max)')
    ax1.set_title(f'Power Allocation Evolution: Top-{top_k} Most Constrained Transmitters ({title_suffix})')
    ax1.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.3, label='P_max')
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.1])
    
    # Plot bottom-k (least constrained)
    colors_bottom = plt.cm.Blues(np.linspace(0.4, 0.9, top_k))
    for i, tx_idx in enumerate(bottom_k_indices):
        power_evolution = [powers[tx_idx] for powers in all_powers_normalized]
        receivers_served = np.where(associations[tx_idx, :] == 1)[0]
        rx_str = ','.join(map(str, receivers_served))
        ax2.plot(epochs, power_evolution, '-', linewidth=1.5, color=colors_bottom[i],
                label=f'TX {tx_idx} → RX [{rx_str}] (violation={transmitter_violations[tx_idx]:.3f})')
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Normalized Power (p/P_max)')
    ax2.set_title(f'Power Allocation Evolution: Bottom-{top_k} Least Constrained Transmitters ({title_suffix})')
    ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=1, alpha=0.3, label='P_max')
    ax2.legend(loc='best', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1.1])
    
    plt.tight_layout()
    
    # Save figure with network_id in filename
    plot_path = os.path.join(output_dir, f'power_allocations_network_{network_id}.pdf')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved power allocation plot to {plot_path}")


# ---------------------------------------------------------------------------
# Tracked-network trace visualization (Phase-2)
# ---------------------------------------------------------------------------


def parse_trace_jsonl(trace_path: str | Path) -> dict[int, dict]:
    """Parse a ``tracked_network_trace.jsonl`` file into structured per-network data.

    Returns
    -------
    parsed : dict[int, dict]
        Mapping ``network_id`` → ``{"metadata": {...}, "epochs": [sorted list of epoch dicts]}``.
        Each epoch dict retains the full JSON payload from the JSONL line.
    """
    trace_path = Path(trace_path)
    if not trace_path.exists():
        logger.warning("Trace file not found: %s", trace_path)
        return {}

    networks: dict[int, dict] = {}
    with open(trace_path, "r") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON at line %d of %s", line_no, trace_path)
                continue

            net_id = int(entry.get("network_id", -1))
            if net_id < 0:
                continue

            if net_id not in networks:
                networks[net_id] = {"metadata": None, "epochs": []}

            if entry.get("type") == "metadata":
                networks[net_id]["metadata"] = entry
            elif entry.get("type") == "epoch_trace":
                networks[net_id]["epochs"].append(entry)

    # Sort epoch entries by epoch number.
    for net_data in networks.values():
        net_data["epochs"].sort(key=lambda e: int(e.get("epoch", 0)))

    return networks


def select_interesting_receivers(
    trace_data: dict,
    network_id: int,
    k: int = 2,
    *,
    strategy: str = "top_dual_var",
    neg_corr_pair_cfg: dict | None = None,
    collection_window: int | None = None,
) -> list[int]:
    """Select receivers whose training dynamics are most interesting to visualize.

    Parameters
    ----------
    trace_data : dict
        Output of :func:`parse_trace_jsonl`.
    network_id : int
        Network to analyze.
    k : int
        Number of receivers to return.
    strategy : str
        Selection strategy.  One of:

        - ``"top_dual_var"`` — rank by dual multiplier variance, pick top-k.
        - ``"top_primal_var"`` — rank by receiver power variance, pick top-k.
        - ``"top_combined_var"`` — rank by sum of dual and primal variance ranks.
        - ``"neg_corr_pair"`` — (k=2 only) variance + interference filtered, then
          pick the most negatively correlated dual pair.  Falls back to
          ``"top_dual_var"`` when k != 2 or data is insufficient.
    neg_corr_pair_cfg : dict or None
        Settings for ``"neg_corr_pair"`` strategy:

        - ``variance_quantile`` (float, default 0.5): keep receivers in the
          top fraction of either dual or primal variance.
        - ``interference_weight`` (float, default 1.0): weight of mutual
          interference from ``cross_channel_gains`` in pair scoring.

    Returns
    -------
    receiver_indices : list[int]
        Sorted list of up to *k* receiver indices.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    valid_strategies = ("top_dual_var", "top_primal_var", "top_combined_var", "neg_corr_pair")
    if strategy not in valid_strategies:
        raise ValueError(
            f"Unknown auto_strategy {strategy!r}; expected one of {valid_strategies}"
        )

    net = trace_data.get(int(network_id))
    if net is None:
        return []

    epochs = net["epochs"]
    metadata = net.get("metadata") or {}

    # ---- extract full-vector trajectories ----------------------------------
    dual_vectors = [
        e.get("dual_multipliers") for e in epochs
        if e.get("dual_multipliers") is not None
    ]
    dual_arr = (
        np.array(dual_vectors) if dual_vectors and len(dual_vectors) >= 2
        else None
    )  # (T, n) or None

    power_vectors = [
        e.get("receiver_power_allocations") for e in epochs
        if e.get("receiver_power_allocations") is not None
    ]
    power_arr = (
        np.array(power_vectors) if power_vectors and len(power_vectors) >= 2
        else None
    )  # (T, n) or None

    # Restrict to the collection window (tail of the training run).
    if collection_window is not None and collection_window > 0:
        if dual_arr is not None and dual_arr.shape[0] > collection_window:
            dual_arr = dual_arr[-collection_window:]
        if power_arr is not None and power_arr.shape[0] > collection_window:
            power_arr = power_arr[-collection_window:]

    # ---- metadata fallback -------------------------------------------------
    def _fallback() -> list[int]:
        fb = metadata.get("tracked_receiver_indices", [])
        return [int(i) for i in fb[:k]]

    # ---- top_dual_var ------------------------------------------------------
    def _top_dual_var() -> list[int]:
        if dual_arr is None:
            return _fallback()
        variances = np.var(dual_arr, axis=0)
        top_k = np.argsort(variances)[-k:][::-1]
        return sorted(int(idx) for idx in top_k)

    # ---- top_primal_var ----------------------------------------------------
    def _top_primal_var() -> list[int]:
        if power_arr is None:
            logger.debug("No full power vectors; falling back to top_dual_var.")
            return _top_dual_var()
        variances = np.var(power_arr, axis=0)
        top_k = np.argsort(variances)[-k:][::-1]
        return sorted(int(idx) for idx in top_k)

    # ---- top_combined_var --------------------------------------------------
    def _top_combined_var() -> list[int]:
        if dual_arr is not None and power_arr is not None:
            dual_var = np.var(dual_arr, axis=0)
            power_var = np.var(power_arr, axis=0)
            n = len(dual_var)
            # Rank: 0 = highest variance (best).
            dual_ranks = (n - 1) - np.argsort(np.argsort(dual_var))
            power_ranks = (n - 1) - np.argsort(np.argsort(power_var))
            combined = dual_ranks + power_ranks  # lower = more interesting
            top_k = np.argsort(combined)[:k]
            return sorted(int(idx) for idx in top_k)
        if dual_arr is not None:
            logger.debug("No full power vectors; falling back to top_dual_var.")
            return _top_dual_var()
        if power_arr is not None:
            logger.debug("No full dual vectors; falling back to top_primal_var.")
            return _top_primal_var()
        return _fallback()

    # ---- neg_corr_pair (k=2 only) ------------------------------------------
    def _neg_corr_pair() -> list[int]:
        if k != 2:
            logger.debug("neg_corr_pair requires k=2 (got k=%d); using top_dual_var.", k)
            return _top_dual_var()
        if dual_arr is None:
            return _fallback()

        cfg = neg_corr_pair_cfg or {}
        variance_quantile = float(cfg.get("variance_quantile", 0.5))
        interference_weight = float(cfg.get("interference_weight", 1.0))
        quantile_source = list(cfg.get("quantile_source", ["dual_var", "primal_var"]))

        n = dual_arr.shape[1]
        if n < 2:
            return _fallback()

        # Filter candidates based on configured sources.
        candidate_mask = np.zeros(n, dtype=bool)
        if "dual_var" in quantile_source:
            dv = np.var(dual_arr, axis=0)
            candidate_mask |= dv >= np.quantile(dv, 1.0 - variance_quantile)
        if "primal_var" in quantile_source and power_arr is not None:
            pv = np.var(power_arr, axis=0)
            candidate_mask |= pv >= np.quantile(pv, 1.0 - variance_quantile)
        if "dual_mean" in quantile_source:
            dm = np.mean(dual_arr, axis=0)
            candidate_mask |= dm >= np.quantile(dm, 1.0 - variance_quantile)
        if "primal_mean" in quantile_source and power_arr is not None:
            pm = np.mean(power_arr, axis=0)
            candidate_mask |= pm >= np.quantile(pm, 1.0 - variance_quantile)
        if "bottom_rate_mean" in quantile_source:
            rv = [e.get("ergodic_rates") for e in epochs if e.get("ergodic_rates") is not None]
            r_arr = np.array(rv) if rv and len(rv) >= 2 else None
            if r_arr is not None:
                if collection_window is not None and collection_window > 0 and r_arr.shape[0] > collection_window:
                    r_arr = r_arr[-collection_window:]
                rm = np.mean(r_arr, axis=0)
                # Lowest rate → candidate (bottom quantile of rates).
                candidate_mask |= rm <= np.quantile(rm, variance_quantile)
        if not np.any(candidate_mask):
            dv = np.var(dual_arr, axis=0)
            candidate_mask = dv >= np.quantile(dv, 1.0 - variance_quantile)
        candidates = list(np.where(candidate_mask)[0])

        if len(candidates) < 2:
            # Not enough candidates after filtering; take top-2 by dual variance.
            top_2 = np.argsort(np.var(dual_arr, axis=0))[-2:][::-1]
            return sorted(int(idx) for idx in top_2)

        # Dual correlation matrix.
        corr_matrix = np.corrcoef(dual_arr.T)  # (n, n)

        # Mutual interference from cross_channel_gains (if available).
        ccg = metadata.get("cross_channel_gains")
        mutual_norm: np.ndarray | None = None
        if ccg is not None and interference_weight > 0:
            ccg_arr = np.array(ccg, dtype=float)
            mutual = ccg_arr + ccg_arr.T
            np.fill_diagonal(mutual, 0.0)
            mutual_max = mutual.max()
            if mutual_max > 0:
                mutual_norm = mutual / mutual_max

        # Score every candidate pair.
        best_score = -np.inf
        best_pair = (candidates[0], candidates[1])
        for ii in range(len(candidates)):
            for jj in range(ii + 1, len(candidates)):
                ci, cj = candidates[ii], candidates[jj]
                neg_corr = -corr_matrix[ci, cj]
                if np.isnan(neg_corr):
                    neg_corr = 0.0  # constant signals → undefined correlation
                score = neg_corr
                if mutual_norm is not None:
                    score += interference_weight * mutual_norm[ci, cj]
                if score > best_score:
                    best_score = score
                    best_pair = (ci, cj)

        return sorted(best_pair)

    # ---- dispatch ----------------------------------------------------------
    if strategy == "top_dual_var":
        return _top_dual_var()
    elif strategy == "top_primal_var":
        return _top_primal_var()
    elif strategy == "top_combined_var":
        return _top_combined_var()
    else:  # neg_corr_pair
        return _neg_corr_pair()


def select_receivers_by_groups(
    trace_data: dict,
    network_id: int,
    groups: list[dict],
    *,
    strategy: str = "top_dual_var",
    neg_corr_pair_cfg: dict | None = None,
    collection_window: int | None = None,
    seed: int | None = None,
) -> list[list[int]]:
    """Select receivers by quantile groups of the variance ranking.

    Parameters
    ----------
    groups : list[dict]
        Each dict has keys ``quantile`` (``[lo, hi]``, 1.0 = highest variance),
        ``count`` (int), and optionally ``palette``, ``palette_range``,
        ``scatter_pairs``.
    strategy : str
        Ranking strategy (same as :func:`select_interesting_receivers`).
    collection_window : int or None
        If set, compute statistics over the last *collection_window* epochs only.

    Returns
    -------
    grouped : list[list[int]]
        One list per group containing the selected receiver indices.
    """
    net = trace_data.get(int(network_id))
    if net is None:
        return [[] for _ in groups]

    epochs = net["epochs"]

    # ---- extract full-vector trajectories ----------------------------------
    dual_vectors = [
        e.get("dual_multipliers") for e in epochs
        if e.get("dual_multipliers") is not None
    ]
    dual_arr = (
        np.array(dual_vectors) if dual_vectors and len(dual_vectors) >= 2
        else None
    )
    power_vectors = [
        e.get("receiver_power_allocations") for e in epochs
        if e.get("receiver_power_allocations") is not None
    ]
    power_arr = (
        np.array(power_vectors) if power_vectors and len(power_vectors) >= 2
        else None
    )
    rate_vectors = [
        e.get("ergodic_rates") for e in epochs
        if e.get("ergodic_rates") is not None
    ]
    rate_arr = (
        np.array(rate_vectors) if rate_vectors and len(rate_vectors) >= 2
        else None
    )

    # Restrict to collection window.
    if collection_window is not None and collection_window > 0:
        if dual_arr is not None and dual_arr.shape[0] > collection_window:
            dual_arr = dual_arr[-collection_window:]
        if power_arr is not None and power_arr.shape[0] > collection_window:
            power_arr = power_arr[-collection_window:]
        if rate_arr is not None and rate_arr.shape[0] > collection_window:
            rate_arr = rate_arr[-collection_window:]

    # ---- compute ranking metric --------------------------------------------
    n = 0
    if dual_arr is not None:
        n = dual_arr.shape[1]
    elif power_arr is not None:
        n = power_arr.shape[1]
    if n < 2:
        return [[] for _ in groups]

    if strategy == "top_dual_var":
        if dual_arr is not None:
            scores = np.var(dual_arr, axis=0)
        elif power_arr is not None:
            scores = np.var(power_arr, axis=0)
        else:
            return [[] for _ in groups]
    elif strategy == "neg_corr_pair":
        # Correlation-aware composite scoring: variance rank + max-neg-corr rank.
        cfg = neg_corr_pair_cfg or {}
        variance_quantile = float(cfg.get("variance_quantile", 0.5))
        quantile_source = list(cfg.get("quantile_source", ["dual_var", "primal_var"]))

        # Score components from configured sources (normalize to [0,1], take max).
        var_components: list[np.ndarray] = []
        if "dual_var" in quantile_source and dual_arr is not None:
            var_components.append(np.var(dual_arr, axis=0))
        if "primal_var" in quantile_source and power_arr is not None:
            var_components.append(np.var(power_arr, axis=0))
        if "dual_mean" in quantile_source and dual_arr is not None:
            var_components.append(np.mean(dual_arr, axis=0))
        if "primal_mean" in quantile_source and power_arr is not None:
            var_components.append(np.mean(power_arr, axis=0))
        if "bottom_rate_mean" in quantile_source and rate_arr is not None:
            # Negate so lowest rate → highest score.
            var_components.append(-np.mean(rate_arr, axis=0))
        if not var_components:
            # Fallback: use whatever is available.
            if dual_arr is not None:
                var_components.append(np.var(dual_arr, axis=0))
            elif power_arr is not None:
                var_components.append(np.var(power_arr, axis=0))
            else:
                return [[] for _ in groups]

        norm_vars: list[np.ndarray] = []
        for v in var_components:
            vmin, vmax = float(v.min()), float(v.max())
            norm_vars.append((v - vmin) / (vmax - vmin) if vmax > vmin else np.zeros_like(v))
        var_scores = np.maximum.reduce(norm_vars) if len(norm_vars) > 1 else norm_vars[0]

        # Pre-filter: top variance_quantile fraction of receivers.
        var_thresh = float(np.quantile(var_scores, 1.0 - variance_quantile))
        candidate_idx = np.where(var_scores >= var_thresh)[0]

        # Per-receiver max negative correlation with any other candidate.
        max_neg_corr = np.zeros(n)
        if dual_arr is not None and len(candidate_idx) >= 2:
            corr_matrix = np.corrcoef(dual_arr.T)  # (n, n)
            for i in candidate_idx:
                neg_corrs = -corr_matrix[i, candidate_idx]
                neg_corrs[candidate_idx == i] = -np.inf  # exclude self
                valid = neg_corrs[np.isfinite(neg_corrs) & ~np.isnan(neg_corrs)]
                if len(valid) > 0:
                    max_neg_corr[i] = float(np.max(valid))

        # Composite: sum of variance rank and correlation rank.
        var_ranks = np.argsort(np.argsort(var_scores)).astype(float)
        corr_ranks = np.argsort(np.argsort(max_neg_corr)).astype(float)
        scores = var_ranks + corr_ranks
    elif strategy == "top_primal_var":
        if power_arr is not None:
            scores = np.var(power_arr, axis=0)
        elif dual_arr is not None:
            scores = np.var(dual_arr, axis=0)
        else:
            return [[] for _ in groups]
    elif strategy == "top_combined_var":
        if dual_arr is not None and power_arr is not None:
            dual_var = np.var(dual_arr, axis=0)
            power_var = np.var(power_arr, axis=0)
            # Combined score: sum of normalized ranks (higher = more interesting).
            dual_ranks = np.argsort(np.argsort(dual_var)).astype(float)
            power_ranks = np.argsort(np.argsort(power_var)).astype(float)
            scores = dual_ranks + power_ranks
        elif dual_arr is not None:
            scores = np.var(dual_arr, axis=0)
        elif power_arr is not None:
            scores = np.var(power_arr, axis=0)
        else:
            return [[] for _ in groups]
    elif strategy == "top_dual_mean":
        if dual_arr is not None:
            scores = np.mean(dual_arr, axis=0)
        elif power_arr is not None:
            scores = np.mean(power_arr, axis=0)
        else:
            return [[] for _ in groups]
    elif strategy == "top_primal_mean":
        if power_arr is not None:
            scores = np.mean(power_arr, axis=0)
        elif dual_arr is not None:
            scores = np.mean(dual_arr, axis=0)
        else:
            return [[] for _ in groups]
    elif strategy == "bottom_rate_mean":
        # Receivers with the worst (lowest) mean ergodic rate rank highest.
        rate_vectors = [
            e.get("ergodic_rates") for e in net["epochs"]
            if e.get("ergodic_rates") is not None
        ]
        rate_arr = (
            np.array(rate_vectors) if rate_vectors and len(rate_vectors) >= 2
            else None
        )
        if rate_arr is not None:
            if collection_window is not None and collection_window > 0 and rate_arr.shape[0] > collection_window:
                rate_arr = rate_arr[-collection_window:]
            # Negate so lowest rate → highest score → top quantile group.
            scores = -np.mean(rate_arr, axis=0)
        elif dual_arr is not None:
            scores = np.mean(dual_arr, axis=0)
        else:
            return [[] for _ in groups]
    else:
        return [[] for _ in groups]

    # Rank receivers from lowest (0) to highest (n-1) score.
    ranked = np.argsort(np.argsort(scores))  # rank[i] = position of receiver i
    # Normalized rank in [0, 1]: 0 = lowest score, 1 = highest.
    norm_rank = ranked / max(n - 1, 1)

    # ---- sample from each quantile group -----------------------------------
    rng = np.random.default_rng(seed)
    selected_set: set[int] = set()
    result: list[list[int]] = []

    for g in groups:
        q_lo, q_hi = float(g["quantile"][0]), float(g["quantile"][1])
        count = int(g.get("count", 2))

        # Find receivers in this quantile band (excluding already selected).
        candidates = [
            i for i in range(n)
            if q_lo <= norm_rank[i] <= q_hi and i not in selected_set
        ]

        if len(candidates) <= count:
            chosen = candidates
        else:
            chosen = list(rng.choice(candidates, size=count, replace=False))

        chosen = sorted(chosen)
        selected_set.update(chosen)
        result.append(chosen)

    return result


def select_bottom_receivers(
    trace_data: dict,
    network_id: int,
    k: int = 2,
    *,
    strategy: str = "top_dual_var",
    collection_window: int | None = None,
) -> list[int]:
    """Select the *k* least interesting receivers (bottom-k by variance).

    Mirrors :func:`select_interesting_receivers` but picks from the bottom of
    the ranking.  Only variance-based strategies are supported; ``neg_corr_pair``
    falls back to ``top_dual_var`` ordering (bottom-k of that ranking).
    """
    if k < 1:
        return []

    net = trace_data.get(int(network_id))
    if net is None:
        return []

    epochs = net["epochs"]

    dual_vectors = [
        e.get("dual_multipliers") for e in epochs
        if e.get("dual_multipliers") is not None
    ]
    dual_arr = (
        np.array(dual_vectors) if dual_vectors and len(dual_vectors) >= 2
        else None
    )
    power_vectors = [
        e.get("receiver_power_allocations") for e in epochs
        if e.get("receiver_power_allocations") is not None
    ]
    power_arr = (
        np.array(power_vectors) if power_vectors and len(power_vectors) >= 2
        else None
    )

    # Restrict to the collection window (tail of the training run).
    if collection_window is not None and collection_window > 0:
        if dual_arr is not None and dual_arr.shape[0] > collection_window:
            dual_arr = dual_arr[-collection_window:]
        if power_arr is not None and power_arr.shape[0] > collection_window:
            power_arr = power_arr[-collection_window:]

    if strategy in ("top_dual_var", "neg_corr_pair"):
        if dual_arr is None:
            return []
        variances = np.var(dual_arr, axis=0)
    elif strategy == "top_primal_var":
        if power_arr is None and dual_arr is None:
            return []
        variances = np.var(power_arr, axis=0) if power_arr is not None else np.var(dual_arr, axis=0)
    elif strategy == "top_combined_var":
        if dual_arr is not None and power_arr is not None:
            n = dual_arr.shape[1]
            dual_ranks = (n - 1) - np.argsort(np.argsort(np.var(dual_arr, axis=0)))
            power_ranks = (n - 1) - np.argsort(np.argsort(np.var(power_arr, axis=0)))
            combined = dual_ranks + power_ranks
            # Bottom-k = highest combined rank (least interesting).
            bottom_k = np.argsort(combined)[-(k):][::-1]
            return sorted(int(idx) for idx in bottom_k)
        if dual_arr is not None:
            variances = np.var(dual_arr, axis=0)
        elif power_arr is not None:
            variances = np.var(power_arr, axis=0)
        else:
            return []
    else:
        return []

    # Bottom-k = lowest variance.
    bottom_k_idx = np.argsort(variances)[:k]
    return sorted(int(idx) for idx in bottom_k_idx)


def select_interesting_pairs(
    trace_data: dict,
    network_id: int,
    n_pairs: int = 4,
    *,
    strategy: str = "neg_corr_pair",
    neg_corr_pair_cfg: dict | None = None,
    rank: str = "top",
    collection_window: int | None = None,
) -> list[tuple[int, int]]:
    """Select up to *n_pairs* distinct receiver pairs for scatter plots.

    Uses the receiver selection strategy to rank pairs.  For strategies that
    return single receivers (``top_dual_var``, etc.), the top-``2*n_pairs``
    receivers are selected and paired sequentially.  For ``neg_corr_pair``,
    the top-*n_pairs* scored pairs are returned directly.

    Parameters
    ----------
    rank : str
        ``"top"`` — most interesting pairs (default).
        ``"bottom"`` — least interesting pairs (lowest variance / lowest score).

    Returns
    -------
    pairs : list[tuple[int, int]]
        Up to *n_pairs* pairs of receiver indices, each sorted (i < j).
        Pairs are ordered from most to least interesting (or vice versa for bottom).
    """
    net = trace_data.get(int(network_id))
    if net is None:
        return []

    epochs = net["epochs"]
    metadata = net.get("metadata") or {}

    # ---- extract full dual / power vectors ---------------------------------
    dual_vectors = [
        e.get("dual_multipliers") for e in epochs
        if e.get("dual_multipliers") is not None
    ]
    dual_arr = (
        np.array(dual_vectors) if dual_vectors and len(dual_vectors) >= 2
        else None
    )
    power_vectors = [
        e.get("receiver_power_allocations") for e in epochs
        if e.get("receiver_power_allocations") is not None
    ]
    power_arr = (
        np.array(power_vectors) if power_vectors and len(power_vectors) >= 2
        else None
    )

    # Restrict to the collection window (tail of the training run).
    if collection_window is not None and collection_window > 0:
        if dual_arr is not None and dual_arr.shape[0] > collection_window:
            dual_arr = dual_arr[-collection_window:]
        if power_arr is not None and power_arr.shape[0] > collection_window:
            power_arr = power_arr[-collection_window:]

    n = 0
    if dual_arr is not None:
        n = dual_arr.shape[1]
    elif power_arr is not None:
        n = power_arr.shape[1]
    if n < 2:
        return []

    # ---- neg_corr_pair: score all candidate pairs, return top-n_pairs ------
    if strategy == "neg_corr_pair" and dual_arr is not None:
        cfg = neg_corr_pair_cfg or {}
        variance_quantile = float(cfg.get("variance_quantile", 0.5))
        interference_weight = float(cfg.get("interference_weight", 1.0))
        quantile_source = list(cfg.get("quantile_source", ["dual_var", "primal_var"]))

        # Filter candidates based on configured sources.
        candidate_mask = np.zeros(n, dtype=bool)
        if "dual_var" in quantile_source:
            dv = np.var(dual_arr, axis=0)
            candidate_mask |= dv >= np.quantile(dv, 1.0 - variance_quantile)
        if "primal_var" in quantile_source and power_arr is not None:
            pv = np.var(power_arr, axis=0)
            candidate_mask |= pv >= np.quantile(pv, 1.0 - variance_quantile)
        if "dual_mean" in quantile_source:
            dm = np.mean(dual_arr, axis=0)
            candidate_mask |= dm >= np.quantile(dm, 1.0 - variance_quantile)
        if "primal_mean" in quantile_source and power_arr is not None:
            pm = np.mean(power_arr, axis=0)
            candidate_mask |= pm >= np.quantile(pm, 1.0 - variance_quantile)
        if "bottom_rate_mean" in quantile_source:
            rv = [e.get("ergodic_rates") for e in epochs if e.get("ergodic_rates") is not None]
            r_arr = np.array(rv) if rv and len(rv) >= 2 else None
            if r_arr is not None:
                if collection_window is not None and collection_window > 0 and r_arr.shape[0] > collection_window:
                    r_arr = r_arr[-collection_window:]
                rm = np.mean(r_arr, axis=0)
                candidate_mask |= rm <= np.quantile(rm, variance_quantile)
        if not np.any(candidate_mask):
            dv = np.var(dual_arr, axis=0)
            candidate_mask = dv >= np.quantile(dv, 1.0 - variance_quantile)
        candidates = list(np.where(candidate_mask)[0])
        if len(candidates) < 2:
            candidates = list(np.argsort(np.var(dual_arr, axis=0))[-max(4, 2 * n_pairs):])

        corr_matrix = np.corrcoef(dual_arr.T)

        ccg = metadata.get("cross_channel_gains")
        mutual_norm: np.ndarray | None = None
        if ccg is not None and interference_weight > 0:
            ccg_arr = np.array(ccg, dtype=float)
            mutual = ccg_arr + ccg_arr.T
            np.fill_diagonal(mutual, 0.0)
            mutual_max = mutual.max()
            if mutual_max > 0:
                mutual_norm = mutual / mutual_max

        # Score every candidate pair.
        scored: list[tuple[float, int, int]] = []
        for ii in range(len(candidates)):
            for jj in range(ii + 1, len(candidates)):
                ci, cj = candidates[ii], candidates[jj]
                neg_corr = -corr_matrix[ci, cj]
                if np.isnan(neg_corr):
                    neg_corr = 0.0
                score = neg_corr
                if mutual_norm is not None:
                    score += interference_weight * mutual_norm[ci, cj]
                scored.append((score, min(ci, cj), max(ci, cj)))

        # Sort: top = descending by score, bottom = ascending.
        scored.sort(key=lambda x: x[0], reverse=(rank == "top"))

        # Greedily pick pairs, ensuring no pair is duplicated.
        seen_pairs: set[tuple[int, int]] = set()
        result: list[tuple[int, int]] = []
        for _score, ci, cj in scored:
            pair = (ci, cj)
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                result.append(pair)
                if len(result) >= n_pairs:
                    break
        return result

    # ---- score-based: pick receivers, form sequential pairs -----------------
    if strategy == "top_dual_mean":
        var_arr = np.mean(dual_arr, axis=0) if dual_arr is not None else (
            np.mean(power_arr, axis=0) if power_arr is not None else None)
    elif strategy == "top_primal_mean":
        var_arr = np.mean(power_arr, axis=0) if power_arr is not None else (
            np.mean(dual_arr, axis=0) if dual_arr is not None else None)
    elif strategy == "bottom_rate_mean":
        rate_vectors = [
            e.get("ergodic_rates") for e in net["epochs"]
            if e.get("ergodic_rates") is not None
        ]
        rate_arr = (
            np.array(rate_vectors) if rate_vectors and len(rate_vectors) >= 2
            else None
        )
        if rate_arr is not None:
            if collection_window is not None and collection_window > 0 and rate_arr.shape[0] > collection_window:
                rate_arr = rate_arr[-collection_window:]
            var_arr = -np.mean(rate_arr, axis=0)
        elif dual_arr is not None:
            var_arr = np.mean(dual_arr, axis=0)
        else:
            var_arr = None
    elif dual_arr is not None:
        var_arr = np.var(dual_arr, axis=0)
    elif power_arr is not None:
        var_arr = np.var(power_arr, axis=0)
    else:
        var_arr = None
    if var_arr is None:
        return []

    if strategy == "top_combined_var" and dual_arr is not None and power_arr is not None:
        dual_var = np.var(dual_arr, axis=0)
        power_var = np.var(power_arr, axis=0)
        dual_ranks = (n - 1) - np.argsort(np.argsort(dual_var))
        power_ranks = (n - 1) - np.argsort(np.argsort(power_var))
        combined = dual_ranks + power_ranks
        if rank == "bottom":
            selected_indices = list(np.argsort(combined)[-(2 * n_pairs):][::-1])
        else:
            selected_indices = list(np.argsort(combined)[:2 * n_pairs])
    else:
        if rank == "bottom":
            selected_indices = list(np.argsort(var_arr)[:2 * n_pairs])
        else:
            selected_indices = list(np.argsort(var_arr)[-(2 * n_pairs):][::-1])

    # Form pairs from selected receivers.
    pairs: list[tuple[int, int]] = []
    for i in range(0, len(selected_indices) - 1, 2):
        a, b = selected_indices[i], selected_indices[i + 1]
        pairs.append((min(a, b), max(a, b)))
        if len(pairs) >= n_pairs:
            break
    return pairs


def visualize_trace_dynamics(
    trace_data: dict,
    network_id: int,
    output_dir: str | Path = "output",
    *,
    grouped_receivers: list[list[int]] | None = None,
    group_cfgs: list[dict] | None = None,
    auto_strategy: str = "top_dual_var",
    neg_corr_pair_cfg: dict | None = None,
    collection_window: int | None = None,
    overlay_samples: dict[int, np.ndarray] | None = None,
    overlay_cfg: dict | None = None,
    plot_cfg: dict | None = None,
    seed: int | None = None,
) -> Path | None:
    """Produce a multi-panel training-dynamics figure for one tracked network.

    Parameters
    ----------
    trace_data : dict
        Output of :func:`parse_trace_jsonl`.
    network_id : int
        Network to visualize.
    output_dir : str or Path
        Directory where the PDF is saved.
    grouped_receivers : list[list[int]] or None
        Receivers to plot, organized by quantile groups.  Each inner list
        is one group.  ``None`` triggers auto-selection.
    group_cfgs : list[dict] or None
        Per-group config (palette, palette_range, scatter_pairs, etc.).
    plot_cfg : dict or None
        Unified plot-style config (from ``plot_style`` + ``modes`` + ``panels``).

    Returns
    -------
    plot_path : Path or None
        Path to the saved figure, or ``None`` if there is insufficient data.
    """
    net = trace_data.get(int(network_id))
    if net is None or not net["epochs"]:
        logger.warning("No trace data for network %d; skipping visualization.", network_id)
        return None

    metadata = net.get("metadata") or {}
    epochs_list = net["epochs"]

    # ---- resolve plot style ------------------------------------------------
    pcfg = plot_cfg or {}
    full_mode = pcfg.get("modes", {}).get("full", {})
    figsize = tuple(full_mode.get("figsize", [12, 16]))
    dpi = int(pcfg.get("dpi", 150))
    lw = float(pcfg.get("linewidth", 1.2))
    alpha_raw = float(pcfg.get("alpha_raw", 0.35))
    grid_alpha = float(pcfg.get("grid_alpha", 0.3))
    legend_fs = int(pcfg.get("legend_fontsize", 8))
    title_fs = int(pcfg.get("title_fontsize", 11))
    label_fs = int(pcfg.get("label_fontsize", 10))
    raw_lw = float(pcfg.get("raw_linewidth", 1.0))
    smoothing_window = int(pcfg.get("smoothing_window", 50))
    fmt = str(pcfg.get("format", "pdf"))
    ts_grid_ls = str(pcfg.get("grid_linestyle", "-"))

    tick_fs_raw = pcfg.get("tick_fontsize")
    tick_fs: int | None = int(tick_fs_raw) if tick_fs_raw is not None else None

    suptitle_mode = str(pcfg.get("suptitle", "auto")).strip().lower()
    suptitle_text_cfg = pcfg.get("suptitle_text")
    suptitle_fs = int(pcfg.get("suptitle_fontsize", title_fs + 2))

    # Per-panel configuration.
    panels = pcfg.get("panels") or {}
    power_panel = panels.get("power") or {}
    rates_panel = panels.get("rates") or {}
    dual_panel = panels.get("dual") or {}
    slack_panel = panels.get("slack") or {}
    scatter_panel = panels.get("scatter") or {}

    # ---- resolve receiver groups -------------------------------------------
    if grouped_receivers is None:
        grouped_receivers = []
    if group_cfgs is None:
        group_cfgs = []
    if not grouped_receivers or all(len(g) == 0 for g in grouped_receivers):
        grouped_receivers = [select_interesting_receivers(
            trace_data, network_id, k=2,
            strategy=auto_strategy,
            neg_corr_pair_cfg=neg_corr_pair_cfg,
            collection_window=collection_window,
        )]
        group_cfgs = [{"palette": "tab10", "palette_range": [0.0, 1.0], "scatter_pairs": 1}]

    # Flatten all receivers (deduplicated, preserving group order).
    receiver_indices: list[int] = []
    _seen: set[int] = set()
    for grx in grouped_receivers:
        for ri in grx:
            if ri not in _seen:
                receiver_indices.append(ri)
                _seen.add(ri)

    # Build subset for dual panel (respects per-group dual_count).
    dual_rx: list[int] = []
    _seen_dual: set[int] = set()
    for gi, grx in enumerate(grouped_receivers):
        gcfg = group_cfgs[gi] if gi < len(group_cfgs) else {}
        dc = int(gcfg.get("dual_count", len(grx)))
        for ri in grx[:dc]:
            if ri not in _seen_dual:
                dual_rx.append(ri)
                _seen_dual.add(ri)

    if not receiver_indices:
        logger.warning("No receiver indices resolved for network %d; skipping.", network_id)
        return None

    # ---- validate receiver indices against actual data dimension -----------
    n_receivers_data: int | None = None
    for entry in epochs_list:
        full_duals = entry.get("dual_multipliers")
        if full_duals is not None:
            n_receivers_data = len(full_duals)
            break
        full_rates = entry.get("ergodic_rates")
        if full_rates is not None:
            n_receivers_data = len(full_rates)
            break
    if n_receivers_data is None:
        assoc = metadata.get("associations")
        if assoc and isinstance(assoc, list) and len(assoc) > 0:
            n_receivers_data = len(assoc[0]) if isinstance(assoc[0], list) else len(assoc)

    if n_receivers_data is not None:
        receiver_indices = [ri for ri in receiver_indices if ri < n_receivers_data]
        if not receiver_indices:
            logger.warning("No valid receiver indices for network %d; skipping.", network_id)
            return None

    # Keep the dual-panel subset consistent with the validated receivers so
    # out-of-range indices cannot leak into duals_series and raise KeyError.
    _valid_rx = set(receiver_indices)
    _dropped_dual = [ri for ri in dual_rx if ri not in _valid_rx]
    if _dropped_dual:
        logger.warning(
            "Dropping out-of-range dual-panel receiver indices for network %d: %s",
            network_id, _dropped_dual,
        )
    dual_rx = [ri for ri in dual_rx if ri in _valid_rx]

    # ---- extract time-series from epoch entries ----------------------------
    epoch_nums: list[int] = []
    powers_series: dict[int, list[float]] = {i: [] for i in receiver_indices}
    rates_series: dict[int, list[float]] = {i: [] for i in receiver_indices}
    duals_series: dict[int, list[float]] = {i: [] for i in receiver_indices}
    slacks_series: dict[int, list[float]] = {i: [] for i in receiver_indices}
    # Windowed averages: {window_key: {receiver_idx: [values]}}.
    windowed_rates: dict[str, dict[int, list[float]]] = {}

    r_min_per_receiver = metadata.get("r_min_per_receiver", [])
    window_sizes = metadata.get("window_sizes", [])

    for entry in epochs_list:
        epoch_nums.append(int(entry["epoch"]))
        sel = entry.get("selected_receiver_trace", {})
        entry_rx_indices = sel.get("receiver_indices", [])

        # Build a fast look-up from the entry's receiver list to position.
        rx_pos = {int(ri): pos for pos, ri in enumerate(entry_rx_indices)}

        for ri in receiver_indices:
            pos = rx_pos.get(ri)
            if pos is not None:
                powers_series[ri].append(
                    float(sel.get("receiver_power_allocations", [0.0])[pos])
                )
                rates_series[ri].append(
                    float(sel.get("ergodic_rates", [0.0])[pos])
                )
                duals_series[ri].append(
                    float(sel.get("dual_multipliers", [0.0])[pos])
                )
                slacks_series[ri].append(
                    float(sel.get("constraint_slacks", [0.0])[pos])
                )
            else:
                # Fall back to full vectors if available.
                full_rates = entry.get("ergodic_rates")
                full_duals = entry.get("dual_multipliers")
                full_slacks = entry.get("constraint_slacks")
                full_powers = entry.get("receiver_power_allocations")
                powers_series[ri].append(
                    float(full_powers[ri]) if full_powers and ri < len(full_powers) else 0.0
                )
                rates_series[ri].append(
                    float(full_rates[ri]) if full_rates and ri < len(full_rates) else 0.0
                )
                duals_series[ri].append(
                    float(full_duals[ri]) if full_duals and ri < len(full_duals) else 0.0
                )
                slacks_series[ri].append(
                    float(full_slacks[ri]) if full_slacks and ri < len(full_slacks) else 0.0
                )

        # Windowed average rates — handle selected and full separately to
        # avoid position-indexed sel values overwriting index-addressed full
        # values for the same window key.
        sel_windowed = sel.get("windowed_avg_rates", {})
        full_windowed = entry.get("windowed_avg_rates", {})
        all_wkeys = set(sel_windowed) | set(full_windowed)

        for wkey in all_wkeys:
            if wkey not in windowed_rates:
                windowed_rates[wkey] = {i: [] for i in receiver_indices}
            sel_wvals = sel_windowed.get(wkey)
            full_wvals = full_windowed.get(wkey)
            for ri in receiver_indices:
                pos = rx_pos.get(ri)
                if pos is not None and sel_wvals is not None and pos < len(sel_wvals):
                    # Receiver is in the selected subset — use position index.
                    windowed_rates[wkey][ri].append(float(sel_wvals[pos]))
                elif full_wvals is not None and isinstance(full_wvals, list) and ri < len(full_wvals):
                    # Receiver not in selected subset — use global index into full vector.
                    windowed_rates[wkey][ri].append(float(full_wvals[ri]))

    if not epoch_nums:
        return None

    epoch_arr = np.array(epoch_nums)

    # ---- collection window boundary ----------------------------------------
    cw_epoch: int | None = None  # epoch number of the collection-window start
    if collection_window is not None and collection_window > 0:
        epoch_max = int(epoch_arr[-1])
        cw_epoch = epoch_max - collection_window
        if cw_epoch < int(epoch_arr[0]):
            # Window covers the entire trace — no boundary to draw.
            cw_epoch = None

    # ---- colour palette (one colour per receiver, shared across panels) ----
    n_rx = len(receiver_indices)
    cmap_rx = plt.cm.tab10 if n_rx <= 10 else plt.cm.tab20
    colors = {ri: cmap_rx(i / max(n_rx - 1, 1)) for i, ri in enumerate(receiver_indices)}

    # ---- decide whether to include a scatter panel -------------------------
    show_scatter = (n_rx == 2)

    # ---- figure layout -----------------------------------------------------
    import matplotlib.gridspec as gridspec

    if show_scatter:
        # Row 0: [power time-series | 2D scatter], rows 1-3: full-width.
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(
            4, 2, figure=fig,
            height_ratios=[1, 1, 1, 1],
            width_ratios=[3, 2],
        )
        ax_power = fig.add_subplot(gs[0, 0])
        ax_scatter = fig.add_subplot(gs[0, 1])
        ax_rate = fig.add_subplot(gs[1, :], sharex=ax_power)
        ax_dual = fig.add_subplot(gs[2, :], sharex=ax_power)
        ax_slack = fig.add_subplot(gs[3, :], sharex=ax_power)
        ts_axes = [ax_power, ax_rate, ax_dual, ax_slack]
    else:
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
        ax_power, ax_rate, ax_dual, ax_slack = axes
        ax_scatter = None
        ts_axes = list(axes)

    net_label = metadata.get("constraint_profile_name")
    if net_label:
        auto_suptitle = f"Training Dynamics \u2014 Network {network_id} ({net_label})"
    else:
        auto_suptitle = f"Training Dynamics \u2014 Network {network_id}"
    suptitle = _resolve_text(suptitle_mode, auto_suptitle, suptitle_text_cfg)
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=suptitle_fs, fontweight="bold", y=0.995)

    # ---- collection-window boundary styling (from slack panel config) -------
    cw_line_color = str(slack_panel.get("cw_line_color", "darkorange"))
    cw_line_alpha = float(slack_panel.get("cw_line_alpha", 0.8))
    cw_shade_color = str(slack_panel.get("cw_shade_color", "gold"))
    cw_shade_alpha = float(slack_panel.get("cw_shade_alpha", 0.12))

    def _draw_collection_boundary(ax):
        if cw_epoch is None:
            return
        ax.axvline(
            x=cw_epoch, color=cw_line_color, linestyle="--",
            linewidth=1.0, alpha=cw_line_alpha,
        )
        ax.axvspan(
            cw_epoch, epoch_arr[-1],
            color=cw_shade_color, alpha=cw_shade_alpha,
        )

    # -- Panel 1: Receiver Power Allocations ---------------------------------
    for ri in receiver_indices:
        vals = np.array(powers_series[ri])
        ax_power.plot(
            epoch_arr, vals, color=colors[ri], alpha=alpha_raw, linewidth=raw_lw,
        )
        if len(vals) >= smoothing_window:
            ax_power.plot(
                epoch_arr, smooth_curve(vals, smoothing_window),
                color=colors[ri], linewidth=lw, label=f"RX {ri}",
            )
        else:
            ax_power.plot(
                epoch_arr, vals, color=colors[ri], linewidth=lw, label=f"RX {ri}",
            )
    _draw_collection_boundary(ax_power)

    _t = _resolve_text(power_panel.get("title", "auto"), "Power Allocations", power_panel.get("title_text"))
    if _t is not None:
        ax_power.set_title(_t, fontsize=title_fs)
    _t = _resolve_text(power_panel.get("ylabel", "auto"), "Receiver Power", power_panel.get("ylabel_text"))
    if _t is not None:
        ax_power.set_ylabel(_t, fontsize=label_fs)
    if not _is_off(power_panel.get("legend", "auto")):
        ax_power.legend(loc="best", fontsize=legend_fs)
    ax_power.grid(True, alpha=grid_alpha, linestyle=ts_grid_ls)
    if tick_fs is not None:
        ax_power.tick_params(labelsize=tick_fs)

    # -- Panel 1b: 2D Power Scatter (k=2 only) ------------------------------
    if show_scatter and ax_scatter is not None:
        from matplotlib.ticker import FormatStrFormatter

        scfg = scatter_panel  # scatter config now lives in panels.scatter
        transient_display = str(scfg.get("transient_display", "both")).strip().lower()
        coll_ms = float(scfg.get("collection_marker_size", 50))
        trans_ms = float(scfg.get("transient_marker_size", 20))
        trans_alpha = float(scfg.get("transient_alpha", 0.15))
        coll_alpha = float(scfg.get("collection_alpha", 0.6))
        scatter_cmap_name = str(full_mode.get("scatter_colormap", "viridis"))
        axis_range = str(scfg.get("axis_range", "auto")).strip().lower()

        # Edge styling.
        coll_edgecolor = str(scfg.get("collection_edgecolor", "navy"))
        trans_edgecolor = str(scfg.get("transient_edgecolor", "none"))
        coll_lw = float(scfg.get("collection_linewidth", 0.5))
        trans_lw = float(scfg.get("transient_linewidth", 0.0))

        # Axis padding, bounds, grid, tick format.
        axis_pad = float(scfg.get("axis_padding", 0.05))
        show_bounds = bool(scfg.get("feasibility_bounds", True))
        bounds_color = str(scfg.get("bounds_color", "gray"))
        bounds_ls = str(scfg.get("bounds_linestyle", ":"))
        bounds_alpha = float(scfg.get("bounds_alpha", 0.5))
        bounds_lw = float(scfg.get("bounds_linewidth", 1.5))
        sc_grid_ls = str(scfg.get("grid_linestyle", "--"))
        tick_fmt_str = scfg.get("tick_format")
        sc_legend_loc = str(scfg.get("legend_loc", "upper right"))

        ri_x, ri_y = receiver_indices[0], receiver_indices[1]
        px = np.array(powers_series[ri_x])
        py = np.array(powers_series[ri_y])

        # Normalize by P_max when unit axis range is requested.
        p_max = scfg.get("p_max")
        if axis_range == "unit" and p_max is not None and float(p_max) > 0:
            p_max = float(p_max)
            px = px / p_max
            py = py / p_max

        # Split into transient / collection epochs.
        transient_scope = str(scfg.get("transient_scope", "all")).strip().lower()
        if cw_epoch is not None:
            coll_mask = epoch_arr >= cw_epoch
        else:
            coll_mask = np.ones(len(epoch_arr), dtype=bool)
        trans_mask = ~coll_mask

        # Optionally restrict transient to the early window only.
        if transient_scope == "early" and collection_window is not None and collection_window > 0:
            early_end = int(epoch_arr[0]) + collection_window
            trans_mask = trans_mask & (epoch_arr < early_end)

        scatter_cmap = plt.get_cmap(scatter_cmap_name)

        if transient_display == "both" and np.any(trans_mask):
            ax_scatter.scatter(
                px[trans_mask], py[trans_mask],
                c=epoch_arr[trans_mask], cmap=scatter_cmap,
                s=trans_ms, alpha=trans_alpha,
                edgecolors=trans_edgecolor, linewidth=trans_lw,
            )

        if np.any(coll_mask):
            sc = ax_scatter.scatter(
                px[coll_mask], py[coll_mask],
                c=epoch_arr[coll_mask], cmap=scatter_cmap,
                s=coll_ms, alpha=coll_alpha,
                edgecolors=coll_edgecolor, linewidth=coll_lw,
            )
            cb = fig.colorbar(sc, ax=ax_scatter, pad=0.02)
            cb.set_label("Epoch", fontsize=label_fs - 1)
            cb.ax.tick_params(labelsize=legend_fs)

        if axis_range == "unit" and p_max is not None and p_max > 0:
            auto_sc_xlabel = f"Power at RX {ri_x} / $P_{{\\mathrm{{max}}}}$"
            auto_sc_ylabel = f"Power at RX {ri_y} / $P_{{\\mathrm{{max}}}}$"
        else:
            auto_sc_xlabel = f"Power RX {ri_x}"
            auto_sc_ylabel = f"Power RX {ri_y}"
        _t = _resolve_text(scatter_panel.get("xlabel", "auto"), auto_sc_xlabel, scatter_panel.get("xlabel_text"))
        if _t is not None:
            ax_scatter.set_xlabel(_t, fontsize=label_fs)
        _t = _resolve_text(scatter_panel.get("ylabel", "auto"), auto_sc_ylabel, scatter_panel.get("ylabel_text"))
        if _t is not None:
            ax_scatter.set_ylabel(_t, fontsize=label_fs)

        auto_sc_title = "Power Scatter"
        if cw_epoch is not None:
            if transient_display == "collection_only":
                auto_sc_title += " (collection only)"
            else:
                auto_sc_title += " (transient faint)"
        _t = _resolve_text(scatter_panel.get("title", "auto"), auto_sc_title, scatter_panel.get("title_text"))
        if _t is not None:
            ax_scatter.set_title(_t, fontsize=title_fs)

        # Axis range with padding.
        if axis_range == "unit":
            ax_scatter.set_xlim(-axis_pad, 1.0 + axis_pad)
            ax_scatter.set_ylim(-axis_pad, 1.0 + axis_pad)

        # Feasibility bounds at 0 and 1.
        if show_bounds and axis_range == "unit":
            for bval in (0.0, 1.0):
                ax_scatter.axhline(y=bval, color=bounds_color, linestyle=bounds_ls,
                                   alpha=bounds_alpha, linewidth=bounds_lw)
                ax_scatter.axvline(x=bval, color=bounds_color, linestyle=bounds_ls,
                                   alpha=bounds_alpha, linewidth=bounds_lw)

        show_scatter_legend = not _is_off(scatter_panel.get("legend", "auto"))

        # -- Overlay: generated samples on the scatter panel -----------------
        net_overlay = (overlay_samples or {}).get(int(network_id))
        if net_overlay is not None:
            ov_arr = np.asarray(net_overlay, dtype=float)
            if ov_arr.ndim == 2 and ov_arr.shape[1] > max(ri_x, ri_y):
                if axis_range == "unit" and p_max is not None and p_max > 0:
                    ov_arr = ov_arr / p_max
                ocfg = overlay_cfg or {}
                ax_scatter.scatter(
                    ov_arr[:, ri_x], ov_arr[:, ri_y],
                    marker=str(ocfg.get("marker", "x")),
                    c=str(ocfg.get("color", "crimson")),
                    s=float(ocfg.get("marker_size", 30)),
                    alpha=float(ocfg.get("alpha", 0.5)),
                    label=str(ocfg.get("label", "Generated")),
                    edgecolors="none", zorder=5,
                )
                if show_scatter_legend:
                    ax_scatter.legend(loc=sc_legend_loc, fontsize=legend_fs)

        ax_scatter.grid(True, alpha=grid_alpha, linestyle=sc_grid_ls)
        if tick_fs is not None:
            ax_scatter.tick_params(labelsize=tick_fs)
        if tick_fmt_str:
            ax_scatter.xaxis.set_major_formatter(FormatStrFormatter(tick_fmt_str))
            ax_scatter.yaxis.set_major_formatter(FormatStrFormatter(tick_fmt_str))

    # -- Panel 2: Ergodic Rates + Windowed Averages + r_min ------------------
    # Sort window keys numerically for consistent legend ordering.
    sorted_wkeys = sorted(windowed_rates.keys(), key=lambda k: int(k))
    dash_styles = ["-", "--", ":"]

    for ri in receiver_indices:
        raw_vals = np.array(rates_series[ri])
        ax_rate.plot(
            epoch_arr, raw_vals,
            color=colors[ri], alpha=alpha_raw, linewidth=raw_lw,
        )
        # Windowed averages.
        for wi, wkey in enumerate(sorted_wkeys):
            wvals_ri = windowed_rates[wkey].get(ri, [])
            if not wvals_ri:
                continue
            w_arr = np.array(wvals_ri)
            # Windowed values may start later; align to end of epoch_arr.
            w_epochs = epoch_arr[-len(w_arr):]
            dash = dash_styles[wi % len(dash_styles)]
            label = f"RX {ri} (W={wkey})" if ri == receiver_indices[0] or wi == 0 else None
            # Only label first receiver per window to avoid legend clutter.
            ax_rate.plot(
                w_epochs, w_arr, color=colors[ri], linestyle=dash,
                linewidth=lw, label=f"RX {ri} W={wkey}",
            )

    # r_min reference lines.
    for ri in receiver_indices:
        if ri < len(r_min_per_receiver):
            r_min_val = float(r_min_per_receiver[ri])
            ax_rate.axhline(
                y=r_min_val, color=colors[ri], linestyle="--",
                linewidth=0.8, alpha=0.6,
            )
            ax_rate.annotate(
                f"r_min[{ri}]={r_min_val:.3f}",
                xy=(epoch_arr[0], r_min_val),
                fontsize=legend_fs - 1, color=colors[ri], alpha=0.8,
                xytext=(5, 3), textcoords="offset points",
            )

    _draw_collection_boundary(ax_rate)

    _t = _resolve_text(rates_panel.get("title", "auto"), "Ergodic Rates & Windowed Averages", rates_panel.get("title_text"))
    if _t is not None:
        ax_rate.set_title(_t, fontsize=title_fs)
    _t = _resolve_text(rates_panel.get("ylabel", "auto"), "Ergodic Rate (bits/s/Hz)", rates_panel.get("ylabel_text"))
    if _t is not None:
        ax_rate.set_ylabel(_t, fontsize=label_fs)
    if not _is_off(rates_panel.get("legend", "auto")):
        ax_rate.legend(loc="best", fontsize=legend_fs, ncol=2)
    ax_rate.grid(True, alpha=grid_alpha, linestyle=ts_grid_ls)
    if tick_fs is not None:
        ax_rate.tick_params(labelsize=tick_fs)

    # -- Panel 3: Dual Multipliers -------------------------------------------
    legend_values_mode = str(dual_panel.get("legend_values", "final")).strip().lower()
    dual_yscale = str(dual_panel.get("yscale", "linear")).strip().lower()

    # Stationary-regime mask for computing mean dual values.
    if cw_epoch is not None:
        stationary_mask = epoch_arr >= cw_epoch
    else:
        stationary_mask = np.ones(len(epoch_arr), dtype=bool)

    for ri in dual_rx:
        vals = np.array(duals_series[ri])
        ax_dual.plot(
            epoch_arr, vals, color=colors[ri], alpha=alpha_raw, linewidth=raw_lw,
        )
        if legend_values_mode == "final":
            final_val = float(vals[-1]) if len(vals) > 0 else 0.0
            dual_label = f"RX {ri} (\u03bb={final_val:.2f})"
        elif legend_values_mode == "mean":
            stat_vals = vals[stationary_mask] if np.any(stationary_mask) else vals
            mean_val = float(np.mean(stat_vals)) if len(stat_vals) > 0 else 0.0
            dual_label = f"RX {ri} ($\\bar{{\\lambda}}$={mean_val:.2f})"
        else:
            dual_label = f"RX {ri}"
        if len(vals) >= smoothing_window:
            ax_dual.plot(
                epoch_arr, smooth_curve(vals, smoothing_window),
                color=colors[ri], linewidth=lw, label=dual_label,
            )
        else:
            ax_dual.plot(
                epoch_arr, vals, color=colors[ri], linewidth=lw, label=dual_label,
            )
    if dual_yscale == "log":
        ax_dual.set_yscale("log")
    elif dual_yscale == "symlog":
        linthresh = float(dual_panel.get("symlog_linthresh", 1.0))
        ax_dual.set_yscale("symlog", linthresh=linthresh)
    _dual_ymin = dual_panel.get("ymin")
    _dual_ymax = dual_panel.get("ymax")
    if _dual_ymin is not None or _dual_ymax is not None:
        ax_dual.set_ylim(
            bottom=float(_dual_ymin) if _dual_ymin is not None else None,
            top=float(_dual_ymax) if _dual_ymax is not None else None,
        )
    dual_xscale = str(dual_panel.get("xscale", "linear")).strip().lower()
    if dual_xscale == "log":
        ax_dual.set_xscale("log")
    elif dual_xscale == "compressed":
        if cw_epoch is not None:
            _comp = float(dual_panel.get("transient_compression", 5.0))
            ax_dual.set_xscale("function", functions=_make_compressed_transient(cw_epoch, _comp))
            _apply_compressed_ticks(ax_dual, epoch_arr, cw_epoch, _comp)
    _draw_collection_boundary(ax_dual)

    _t = _resolve_text(dual_panel.get("title", "auto"), "Dual Multiplier Evolution", dual_panel.get("title_text"))
    if _t is not None:
        ax_dual.set_title(_t, fontsize=title_fs)
    _t = _resolve_text(dual_panel.get("ylabel", "auto"), "Dual Multiplier (\u03bb)", dual_panel.get("ylabel_text"))
    if _t is not None:
        ax_dual.set_ylabel(_t, fontsize=label_fs)
    if not _is_off(dual_panel.get("legend", "auto")):
        ax_dual.legend(loc="best", fontsize=legend_fs, ncol=2)
    ax_dual.grid(True, alpha=grid_alpha, linestyle=ts_grid_ls)
    if tick_fs is not None:
        ax_dual.tick_params(labelsize=tick_fs)

    # -- Panel 4: Constraint Slacks ------------------------------------------
    zl_style = str(slack_panel.get("zero_line_style", "--"))
    zl_color = str(slack_panel.get("zero_line_color", "black"))
    zl_width = float(slack_panel.get("zero_line_width", 1.0))
    zl_alpha = float(slack_panel.get("zero_line_alpha", 1.0))

    # Convention: "max" flips sign so negative = violated.
    slack_conv = str(slack_panel.get("convention", "min")).strip().lower()
    slack_sign = -1.0 if slack_conv == "max" else 1.0

    # Feasibility region colours.
    feasible_color = str(slack_panel.get("feasible_color", "green"))
    feasible_alpha = float(slack_panel.get("feasible_alpha", 0.04))
    infeasible_color = str(slack_panel.get("infeasible_color", "red"))
    infeasible_alpha = float(slack_panel.get("infeasible_alpha", 0.04))

    for ri in receiver_indices:
        vals = slack_sign * np.array(slacks_series[ri])
        # Raw per-epoch values: solid transparent line.
        ax_slack.plot(
            epoch_arr, vals,
            color=colors[ri], alpha=alpha_raw, linewidth=raw_lw,
        )
        if len(vals) >= smoothing_window:
            ax_slack.plot(
                epoch_arr, smooth_curve(vals, smoothing_window),
                color=colors[ri], linewidth=lw, label=f"RX {ri}",
            )
        else:
            ax_slack.plot(
                epoch_arr, vals, color=colors[ri], linewidth=lw, label=f"RX {ri}",
            )
    _draw_collection_boundary(ax_slack)
    ax_slack.axhline(y=0.0, color=zl_color, linestyle=zl_style, linewidth=zl_width, alpha=zl_alpha)
    # Green/red feasibility region shading.
    ylims = ax_slack.get_ylim()
    if slack_conv == "max":
        if ylims[1] > 0:
            ax_slack.axhspan(0.0, ylims[1], color=feasible_color, alpha=feasible_alpha)
        if ylims[0] < 0:
            ax_slack.axhspan(ylims[0], 0.0, color=infeasible_color, alpha=infeasible_alpha)
    else:
        if ylims[0] < 0:
            ax_slack.axhspan(ylims[0], 0.0, color=feasible_color, alpha=feasible_alpha)
        if ylims[1] > 0:
            ax_slack.axhspan(0.0, ylims[1], color=infeasible_color, alpha=infeasible_alpha)

    if slack_conv == "max":
        auto_slack_title = "Constraint Slacks (negative = violated)"
        auto_slack_ylabel = "Constraint Slack (rate \u2212 r_min)"
    else:
        auto_slack_title = "Constraint Slacks (positive = violated)"
        auto_slack_ylabel = "Constraint Slack (r_min \u2212 rate)"

    _t = _resolve_text(slack_panel.get("title", "auto"), auto_slack_title, slack_panel.get("title_text"))
    if _t is not None:
        ax_slack.set_title(_t, fontsize=title_fs)
    _t = _resolve_text(slack_panel.get("ylabel", "auto"), auto_slack_ylabel, slack_panel.get("ylabel_text"))
    if _t is not None:
        ax_slack.set_ylabel(_t, fontsize=label_fs)
    _t = _resolve_text(slack_panel.get("xlabel", "auto"), "Epoch", slack_panel.get("xlabel_text"))
    if _t is not None:
        ax_slack.set_xlabel(_t, fontsize=label_fs)
    if not _is_off(slack_panel.get("legend", "auto")):
        ax_slack.legend(loc="best", fontsize=legend_fs)
    slack_xscale = str(slack_panel.get("xscale", "linear")).strip().lower()
    if slack_xscale == "log":
        ax_slack.set_xscale("log")
    elif slack_xscale == "compressed":
        if cw_epoch is not None:
            _comp = float(slack_panel.get("transient_compression", 5.0))
            ax_slack.set_xscale("function", functions=_make_compressed_transient(cw_epoch, _comp))
            _nticks_slack = int(slack_panel.get("target_nticks", 10))
            _apply_compressed_ticks(ax_slack, epoch_arr, cw_epoch, _comp, target_nticks=_nticks_slack)
    ax_slack.grid(True, alpha=grid_alpha, linestyle=ts_grid_ls)
    if tick_fs is not None:
        ax_slack.tick_params(labelsize=tick_fs)

    # ---- collection-window annotation on bottom axis -----------------------
    if cw_epoch is not None:
        ax_slack.annotate(
            f"collection window ({collection_window} epochs)",
            xy=(cw_epoch, ax_slack.get_ylim()[0]),
            fontsize=legend_fs, color="gray", alpha=0.8,
            xytext=(8, 8), textcoords="offset points",
        )

    plt.tight_layout()

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"trace_dynamics_network_{network_id}_{timestamp}.{fmt}"
    plot_path = output_dir / filename
    fig.savefig(plot_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved trace dynamics plot to %s", plot_path)
    return plot_path


# ---------------------------------------------------------------------------
# Row visualization helpers
# ---------------------------------------------------------------------------


def _resolve_palette(spec, fallback_name: str = "tab10"):
    """Build a matplotlib colormap from a palette specification.

    *spec* can be:
    - A string → treated as a named matplotlib colormap.
    - A list of colour strings (e.g. ``["#dc143c", "#8b5cf6"]``) →
      builds a ``LinearSegmentedColormap`` interpolating between them.
    """
    from matplotlib.colors import LinearSegmentedColormap

    if isinstance(spec, (list, tuple)) and len(spec) >= 2:
        return LinearSegmentedColormap.from_list("custom", list(spec), N=256)
    name = str(spec) if spec else fallback_name
    return plt.get_cmap(name)


def _subsample_indices(n: int, cap: int, method: str = "stride", seed: int | None = None) -> np.ndarray:
    """Return indices to subsample *n* points down to *cap*.

    Parameters
    ----------
    n : int
        Total number of available points.
    cap : int
        Maximum number of points to keep.
    method : str
        ``"stride"`` — uniform spacing (deterministic).
        ``"random"`` — random subset.
    seed : int | None
        RNG seed for reproducibility (``"random"`` method only).
    """
    if n <= cap:
        return np.arange(n)
    if method == "random":
        return np.sort(np.random.default_rng(seed).choice(n, size=cap, replace=False))
    # stride: evenly spaced, always includes first and last.
    return np.linspace(0, n - 1, cap, dtype=int)


def _is_off(val) -> bool:
    """Check if a config value means 'off', handling YAML 1.1 boolean coercion."""
    if val is None or val is False:
        return True
    return str(val).strip().lower() in ("off", "false", "no", "0")


def _make_compressed_transient(
    cw_epoch: float,
    compression: float = 5.0,
) -> tuple:
    """Piecewise-linear x-axis: transient compressed, collection window 1:1.

    Epochs before *cw_epoch* are scaled by ``1 / compression``; epochs at or
    after *cw_epoch* keep their natural linear spacing.  The two segments are
    joined continuously at *cw_epoch*.

    Returns ``(forward, inverse)`` callables suitable for
    ``ax.set_xscale('function', functions=...)``.
    """
    cw_t = cw_epoch / compression  # visual position of the boundary

    def forward(x):
        x = np.asarray(x, dtype=float)
        return np.where(x < cw_epoch, x / compression, cw_t + (x - cw_epoch))

    def inverse(y):
        y = np.asarray(y, dtype=float)
        return np.where(y < cw_t, y * compression, cw_epoch + (y - cw_t))

    return forward, inverse


def _apply_compressed_ticks(
    ax, epoch_arr: np.ndarray, cw_epoch: float, compression: float,
    target_nticks: int = 10,
) -> None:
    """Place major ticks approximately uniform in visual (post-compression) space.

    Allocates ticks to the transient and collection regions proportionally
    to their visual widths, so the collection window gets denser ticks to
    compensate for being uncompressed.
    """
    from matplotlib.ticker import FixedLocator, MaxNLocator

    e_min, e_max = float(epoch_arr[0]), float(epoch_arr[-1])
    tv = cw_epoch / compression       # transient visual width
    cv = e_max - cw_epoch              # collection visual width
    total = tv + cv
    if total <= 0:
        return

    n_t = max(2, round(target_nticks * tv / total))
    n_c = max(2, round(target_nticks * cv / total))

    t_ticks = MaxNLocator(nbins=n_t, integer=True).tick_values(e_min, cw_epoch)
    t_ticks = t_ticks[(t_ticks >= e_min) & (t_ticks < cw_epoch)]

    c_ticks = MaxNLocator(nbins=n_c, integer=True).tick_values(cw_epoch, e_max)
    c_ticks = c_ticks[(c_ticks >= cw_epoch) & (c_ticks <= e_max)]

    ax.xaxis.set_major_locator(
        FixedLocator(np.unique(np.concatenate([t_ticks, c_ticks])))
    )


def _resolve_text(
    mode: str | bool, auto_text: str, explicit_text: str | None = None,
) -> str | None:
    """Return label/title text based on *mode*.

    * ``"off"`` / ``False`` → ``None`` (caller should skip setting the label).
    * ``"explicit"``        → *explicit_text* (falls back to *auto_text* if empty).
    * ``"auto"`` / ``True`` → *auto_text*.

    Handles YAML 1.1 boolean coercion where bare ``off`` → ``False``
    and bare ``on`` → ``True``.
    """
    if mode is None or mode is False:
        return None
    if mode is True:
        return auto_text
    mode = str(mode).strip().lower()
    if mode in ("off", "false", "no", "0"):
        return None
    if mode == "explicit" and explicit_text:
        return str(explicit_text)
    return auto_text


# ---------------------------------------------------------------------------
# Row visualization: two-column layout (dual+slack | 2×2 scatter)
# ---------------------------------------------------------------------------


def visualize_trace_row(
    trace_data: dict,
    network_id: int,
    output_dir: str | Path = "output",
    *,
    grouped_receivers: list[list[int]] | None = None,
    group_cfgs: list[dict] | None = None,
    auto_strategy: str = "top_dual_var",
    neg_corr_pair_cfg: dict | None = None,
    collection_window: int | None = None,
    overlay_samples: dict[int, np.ndarray] | None = None,
    overlay_cfg: dict | None = None,
    plot_cfg: dict | None = None,
    seed: int | None = None,
) -> Path | None:
    """Produce a two-column figure for one tracked network.

    Layout:

    * **Left column** — vertically stacked with shared x-axis:

      - *Top*: Dual multiplier evolution for selected receivers (raw + smoothed).
      - *Bottom*: Worst constraint slack — per-epoch (raw + smoothed) and,
        when *collection_window* is set, worst of the running
        ``collection_window``-average slacks.

    * **Right column** — 2×2 grid of 2-D power scatter plots for the first
      two selected receivers (epoch-coloured, with transient/collection split
      and optional overlay).  Currently all four sub-plots are identical.

    The function signature mirrors :func:`visualize_trace_dynamics` so the
    CLI can dispatch to either visualisation with the same keyword arguments.
    """
    if grouped_receivers is None:
        grouped_receivers = []
    if group_cfgs is None:
        group_cfgs = []

    net = trace_data.get(int(network_id))
    if net is None or not net["epochs"]:
        logger.warning("No trace data for network %d; skipping.", network_id)
        return None

    metadata = net.get("metadata") or {}
    epochs_list = net["epochs"]

    # ---- resolve plot style ------------------------------------------------
    pcfg = plot_cfg or {}
    row_mode = pcfg.get("modes", {}).get("row", {})
    figsize = tuple(row_mode.get("figsize", [20, 8]))
    dpi = int(pcfg.get("dpi", 150))
    lw = float(pcfg.get("linewidth", 1.2))
    alpha_raw = float(pcfg.get("alpha_raw", 0.35))
    grid_alpha = float(pcfg.get("grid_alpha", 0.3))
    legend_fs = int(pcfg.get("legend_fontsize", 8))
    title_fs = int(pcfg.get("title_fontsize", 11))
    label_fs = int(pcfg.get("label_fontsize", 10))
    raw_lw = float(pcfg.get("raw_linewidth", 1.0))
    smoothing_window = int(pcfg.get("smoothing_window", 50))
    fmt = str(pcfg.get("format", "pdf"))
    ts_grid_ls = str(pcfg.get("grid_linestyle", "-"))

    tick_fs_raw = pcfg.get("tick_fontsize")
    tick_fs: int | None = int(tick_fs_raw) if tick_fs_raw is not None else None

    # Suptitle settings.
    suptitle_mode = str(pcfg.get("suptitle", "auto")).strip().lower()
    suptitle_text_cfg = pcfg.get("suptitle_text")
    suptitle_fs = int(pcfg.get("suptitle_fontsize", title_fs + 2))

    slack_raw_color = str(row_mode.get("slack_raw_color", "tab:blue"))
    slack_avg_color = str(row_mode.get("slack_avg_color", "tab:orange"))
    overlay_color = str(row_mode.get("overlay_color", "darkorange"))

    # Per-panel configuration (shared with full mode).
    panels = pcfg.get("panels") or {}
    dual_panel = panels.get("dual") or {}
    slack_panel = panels.get("slack") or {}
    scatter_panel = panels.get("scatter") or {}

    # ---- resolve receiver groups -------------------------------------------
    if not grouped_receivers or all(len(g) == 0 for g in grouped_receivers):
        # Fallback: select top-2 receivers.
        from graph_signal_diffusion.trainers.pd_visualization import select_interesting_receivers
        grouped_receivers = [select_interesting_receivers(
            trace_data, network_id, k=2,
            strategy=auto_strategy,
            neg_corr_pair_cfg=neg_corr_pair_cfg,
            collection_window=collection_window,
        )]
        group_cfgs = [{"palette": "PuRd", "palette_range": [0.35, 0.85], "scatter_pairs": 1}]

    # Flatten all receivers (deduplicated, preserving group order).
    all_rx: list[int] = []
    seen_rx: set[int] = set()
    for grx in grouped_receivers:
        for ri in grx:
            if ri not in seen_rx:
                all_rx.append(ri)
                seen_rx.add(ri)

    # Build subset for dual panel (respects per-group dual_count).
    dual_rx: list[int] = []
    _seen_dual: set[int] = set()
    for gi, grx in enumerate(grouped_receivers):
        gcfg = group_cfgs[gi] if gi < len(group_cfgs) else {}
        dc = int(gcfg.get("dual_count", len(grx)))
        for ri in grx[:dc]:
            if ri not in _seen_dual:
                dual_rx.append(ri)
                _seen_dual.add(ri)

    if not all_rx:
        logger.warning("No receiver indices for network %d; skipping.", network_id)
        return None

    # ---- validate receiver indices against data dimension ------------------
    n_receivers_data: int | None = None
    for entry in epochs_list:
        full_duals = entry.get("dual_multipliers")
        if full_duals is not None:
            n_receivers_data = len(full_duals)
            break
        full_rates = entry.get("ergodic_rates")
        if full_rates is not None:
            n_receivers_data = len(full_rates)
            break
    if n_receivers_data is None:
        assoc = metadata.get("associations")
        if assoc and isinstance(assoc, list) and len(assoc) > 0:
            n_receivers_data = len(assoc[0]) if isinstance(assoc[0], list) else len(assoc)
    if n_receivers_data is not None:
        all_rx = [ri for ri in all_rx if ri < n_receivers_data]
        # Update grouped_receivers to match.
        grouped_receivers = [
            [ri for ri in grx if ri < n_receivers_data]
            for grx in grouped_receivers
        ]
        if not all_rx:
            logger.warning("No valid receiver indices for network %d; skipping.", network_id)
            return None

    # Keep the dual-panel subset consistent with the validated receivers so
    # out-of-range indices cannot leak into duals_series and raise KeyError.
    _valid_rx = set(all_rx)
    _dropped_dual = [ri for ri in dual_rx if ri not in _valid_rx]
    if _dropped_dual:
        logger.warning(
            "Dropping out-of-range dual-panel receiver indices for network %d: %s",
            network_id, _dropped_dual,
        )
    dual_rx = [ri for ri in dual_rx if ri in _valid_rx]

    # ---- extract per-receiver time-series ----------------------------------
    epoch_nums: list[int] = []
    powers_series: dict[int, list[float]] = {i: [] for i in all_rx}
    duals_series: dict[int, list[float]] = {i: [] for i in all_rx}
    slacks_series: dict[int, list[float]] = {i: [] for i in all_rx}
    # Network-wide slack vectors (all receivers) for network-scope worst.
    network_slacks: list[list[float]] = []

    for entry in epochs_list:
        epoch_nums.append(int(entry["epoch"]))
        sel = entry.get("selected_receiver_trace", {})
        entry_rx_indices = sel.get("receiver_indices", [])
        rx_pos = {int(ri): pos for pos, ri in enumerate(entry_rx_indices)}

        for ri in all_rx:
            pos = rx_pos.get(ri)
            if pos is not None:
                powers_series[ri].append(
                    float(sel.get("receiver_power_allocations", [0.0])[pos])
                )
                duals_series[ri].append(
                    float(sel.get("dual_multipliers", [0.0])[pos])
                )
                slacks_series[ri].append(
                    float(sel.get("constraint_slacks", [0.0])[pos])
                )
            else:
                full_powers = entry.get("receiver_power_allocations")
                full_duals = entry.get("dual_multipliers")
                full_slacks = entry.get("constraint_slacks")
                powers_series[ri].append(
                    float(full_powers[ri]) if full_powers and ri < len(full_powers) else 0.0
                )
                duals_series[ri].append(
                    float(full_duals[ri]) if full_duals and ri < len(full_duals) else 0.0
                )
                slacks_series[ri].append(
                    float(full_slacks[ri]) if full_slacks and ri < len(full_slacks) else 0.0
                )

        # Collect full network slack vector for network-wide worst.
        full_slacks = entry.get("constraint_slacks")
        if full_slacks is not None:
            network_slacks.append([float(s) for s in full_slacks])
        else:
            network_slacks.append([])

    if not epoch_nums:
        return None

    epoch_arr = np.array(epoch_nums)

    # ---- collection-window boundary ----------------------------------------
    cw_epoch: int | None = None
    if collection_window is not None and collection_window > 0:
        epoch_max = int(epoch_arr[-1])
        cw_epoch = epoch_max - collection_window
        if cw_epoch < int(epoch_arr[0]):
            cw_epoch = None

    # ---- colour palette (per-group, per-pair base colour) -------------------
    # Each group has its own palette.  Within each group, receivers are paired
    # sequentially: (rx[0],rx[1]) share base colour, (rx[2],rx[3]) another.
    # Even-index = solid, odd-index = dashed.
    colors: dict[int, tuple] = {}
    linestyles: dict[int, str] = {}
    pair_colors: dict[int, tuple] = {}  # scatter subplot_idx → base colour
    scatter_subplot_idx = 0

    for gi, grx in enumerate(grouped_receivers):
        gcfg = group_cfgs[gi] if gi < len(group_cfgs) else {}
        pal_spec = gcfg.get("palette", "tab10")
        pal_range = gcfg.get("palette_range", [0.0, 1.0])
        cmap_g = _resolve_palette(pal_spec)
        is_custom = isinstance(pal_spec, (list, tuple))
        g_lo = 0.0 if is_custom else float(pal_range[0])
        g_hi = 1.0 if is_custom else float(pal_range[1])
        n_g_pairs = max((len(grx) + 1) // 2, 1)

        for i, ri in enumerate(grx):
            if ri in colors:
                continue
            pair_idx = i // 2
            t = g_lo + (g_hi - g_lo) * pair_idx / max(n_g_pairs - 1, 1)
            base_color = cmap_g(t)
            colors[ri] = base_color
            linestyles[ri] = "-" if i % 2 == 0 else "--"

        # Assign scatter pair colours for this group.
        n_scatter = int(gcfg.get("scatter_pairs", 0))
        for p in range(min(n_scatter, n_g_pairs)):
            t = g_lo + (g_hi - g_lo) * p / max(n_g_pairs - 1, 1)
            pair_colors[scatter_subplot_idx] = cmap_g(t)
            scatter_subplot_idx += 1

    # ---- figure layout: left (dual+slack stacked) | right (2×2 scatter) ----
    import matplotlib.gridspec as gridspec

    show_scatter = len(all_rx) >= 2

    fig = plt.figure(figsize=figsize)

    if show_scatter:
        col_wspace = float(scatter_panel.get("column_wspace", 0.15))
        outer = gridspec.GridSpec(
            1, 2, figure=fig, width_ratios=[1, 1.2], wspace=col_wspace,
        )
        # Left column: dual (top) + slack (bottom), shared x-axis.
        gs_left = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[0, 0], hspace=0.12,
        )
        ax_dual = fig.add_subplot(gs_left[0])
        ax_slack = fig.add_subplot(gs_left[1], sharex=ax_dual)

        # Right column: 2×2 scatter grid.
        sc_hspace = float(scatter_panel.get("hspace", 0.15))
        sc_wspace = float(scatter_panel.get("wspace", 0.15))
        gs_right = gridspec.GridSpecFromSubplotSpec(
            2, 2, subplot_spec=outer[0, 1], hspace=sc_hspace, wspace=sc_wspace,
        )
        ax_scatter_grid = [
            [fig.add_subplot(gs_right[r, c]) for c in range(2)]
            for r in range(2)
        ]
    else:
        fig, (ax_dual, ax_slack) = plt.subplots(
            2, 1, figsize=(figsize[0] / 2, figsize[1]), sharex=True,
        )
        ax_scatter_grid = None

    net_label = metadata.get("constraint_profile_name")
    if net_label:
        auto_suptitle = f"Training Dynamics \u2014 Network {network_id} ({net_label})"
    else:
        auto_suptitle = f"Training Dynamics \u2014 Network {network_id}"
    suptitle = _resolve_text(suptitle_mode, auto_suptitle, suptitle_text_cfg)
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=suptitle_fs, fontweight="bold", y=1.02)

    # ---- collection-window boundary styling (from slack panel config) -------
    cw_line_color = str(slack_panel.get("cw_line_color", "darkorange"))
    cw_line_alpha = float(slack_panel.get("cw_line_alpha", 0.8))
    cw_shade_color = str(slack_panel.get("cw_shade_color", "gold"))
    cw_shade_alpha = float(slack_panel.get("cw_shade_alpha", 0.12))

    def _draw_cw(ax):
        if cw_epoch is None:
            return
        ax.axvline(x=cw_epoch, color=cw_line_color, linestyle="--",
                    linewidth=1.0, alpha=cw_line_alpha)
        ax.axvspan(cw_epoch, epoch_arr[-1], color=cw_shade_color, alpha=cw_shade_alpha)

    # Stationary-regime mask for computing mean dual values.
    if cw_epoch is not None:
        stationary_mask = epoch_arr >= cw_epoch
    else:
        stationary_mask = np.ones(len(epoch_arr), dtype=bool)

    # == Left-top: Dual Multiplier Evolution =================================
    legend_values_mode = str(dual_panel.get("legend_values", "final")).strip().lower()
    dual_yscale = str(dual_panel.get("yscale", "linear")).strip().lower()

    for ri in dual_rx:
        vals = np.array(duals_series[ri])
        ls = linestyles.get(ri, "-")
        ax_dual.plot(epoch_arr, vals, color=colors[ri], alpha=alpha_raw,
                     linewidth=raw_lw, linestyle=ls)
        sm = smooth_curve(vals, smoothing_window) if len(vals) >= smoothing_window else vals

        # Legend label with optional value annotation.
        if legend_values_mode == "final":
            final_val = float(vals[-1]) if len(vals) > 0 else 0.0
            dual_label = f"RX {ri} (\u03bb={final_val:.2f})"
        elif legend_values_mode == "mean":
            stat_vals = vals[stationary_mask] if np.any(stationary_mask) else vals
            mean_val = float(np.mean(stat_vals)) if len(stat_vals) > 0 else 0.0
            dual_label = f"RX {ri} ($\\bar{{\\lambda}}$={mean_val:.2f})"
        else:
            dual_label = f"RX {ri}"

        ax_dual.plot(epoch_arr, sm, color=colors[ri], linewidth=lw,
                     linestyle=ls, label=dual_label)

    if dual_yscale == "log":
        ax_dual.set_yscale("log")
    elif dual_yscale == "symlog":
        linthresh = float(dual_panel.get("symlog_linthresh", 1.0))
        ax_dual.set_yscale("symlog", linthresh=linthresh)
    _dual_ymin = dual_panel.get("ymin")
    _dual_ymax = dual_panel.get("ymax")
    if _dual_ymin is not None or _dual_ymax is not None:
        ax_dual.set_ylim(
            bottom=float(_dual_ymin) if _dual_ymin is not None else None,
            top=float(_dual_ymax) if _dual_ymax is not None else None,
        )
    dual_xscale = str(dual_panel.get("xscale", "linear")).strip().lower()
    if dual_xscale == "log":
        ax_dual.set_xscale("log")
    elif dual_xscale == "compressed":
        if cw_epoch is not None:
            _comp = float(dual_panel.get("transient_compression", 5.0))
            _nticks = int(dual_panel.get("target_nticks", 10))
            ax_dual.set_xscale("function", functions=_make_compressed_transient(cw_epoch, _comp))
            _apply_compressed_ticks(ax_dual, epoch_arr, cw_epoch, _comp, target_nticks=_nticks)
    _draw_cw(ax_dual)

    _ylabel = _resolve_text(
        dual_panel.get("ylabel", "auto"),
        "Dual Multiplier (\u03bb)",
        dual_panel.get("ylabel_text"),
    )
    if _ylabel is not None:
        ax_dual.set_ylabel(_ylabel, fontsize=label_fs)

    _title = _resolve_text(
        dual_panel.get("title", "auto"),
        "Dual Multiplier Evolution",
        dual_panel.get("title_text"),
    )
    if _title is not None:
        ax_dual.set_title(_title, fontsize=title_fs)

    if not _is_off(dual_panel.get("legend", "auto")):
        ax_dual.legend(loc="best", fontsize=legend_fs, ncol=2)

    ax_dual.grid(True, alpha=grid_alpha, linestyle=ts_grid_ls)
    if tick_fs is not None:
        ax_dual.tick_params(labelsize=tick_fs)
    # Hide x-tick labels on top panel (shared axis).
    plt.setp(ax_dual.get_xticklabels(), visible=False)

    # == Left-bottom: Worst Constraint Slack =================================
    zl_style = str(slack_panel.get("zero_line_style", "--"))
    zl_color = str(slack_panel.get("zero_line_color", "black"))
    zl_width = float(slack_panel.get("zero_line_width", 1.0))
    zl_alpha = float(slack_panel.get("zero_line_alpha", 1.0))

    # Convention: "max" flips sign so negative = violated.
    slack_conv = str(slack_panel.get("convention", "min")).strip().lower()
    slack_sign = -1.0 if slack_conv == "max" else 1.0
    worst_fn = np.min if slack_conv == "max" else np.max

    # Feasibility region colours.
    feasible_color = str(slack_panel.get("feasible_color", "green"))
    feasible_alpha = float(slack_panel.get("feasible_alpha", 0.04))
    infeasible_color = str(slack_panel.get("infeasible_color", "red"))
    infeasible_alpha = float(slack_panel.get("infeasible_alpha", 0.04))

    worst_scope = str(slack_panel.get("worst_scope", "selected")).strip().lower()
    worst_pctl_raw = slack_panel.get("worst_percentile")
    worst_pctl: float | None = float(worst_pctl_raw) if worst_pctl_raw is not None else None

    if worst_scope == "network" and network_slacks and len(network_slacks[0]) > 0:
        # Network-wide: use full slack vectors across all receivers.
        full_slack_arr = slack_sign * np.array(network_slacks)  # (T, N)
        slack_source = full_slack_arr.T  # (N, T) for windowed average
    else:
        # Selected receivers only (top-k + bottom-k).
        slack_source = slack_sign * np.array(
            [np.array(slacks_series[ri]) for ri in all_rx]
        )  # shape (k_total, T)

    # Windowed-average worst slack (left y-axis): causal trailing average.
    if collection_window is not None and collection_window > 1:
        causal_kernel = np.ones(collection_window) / collection_window
        T = slack_source.shape[1]
        trailing_avg = np.array([
            np.convolve(slack_source[i], causal_kernel, mode="full")[-T:]
            for i in range(slack_source.shape[0])
        ])  # (k_or_N, T)
        if worst_pctl is not None and worst_pctl > 0 and worst_scope == "network":
            if slack_conv == "max":
                worst_avg = np.percentile(trailing_avg, worst_pctl, axis=0)
            else:
                worst_avg = np.percentile(trailing_avg, 100.0 - worst_pctl, axis=0)
        else:
            worst_avg = worst_fn(trailing_avg, axis=0)
        _worst_avg_label = str(slack_panel.get(
            "worst_avg_label",
            f"Worst avg (W={collection_window})",
        ))
        ax_slack.plot(
            epoch_arr, worst_avg,
            color=slack_avg_color, linewidth=lw, linestyle="-",
            label=_worst_avg_label,
        )

    # Infeasibility percentage (right y-axis): fraction of receivers violating
    # their rate constraint at each epoch.
    infeas_color = str(slack_panel.get("infeasibility_line_color", "firebrick"))
    infeas_alpha = float(slack_panel.get("infeasibility_line_alpha", 0.8))
    infeas_lw = float(slack_panel.get("infeasibility_line_width", 1.2))
    ax_infeas = None
    # Compute infeasibility percentage over the same scope as worst slack.
    if worst_scope == "network" and network_slacks and len(network_slacks[0]) > 0:
        # Raw slacks = r_min - rate; infeasible when positive (r_min > rate).
        infeas_mask = np.array(network_slacks) > 0  # (T, N)
        infeas_pct = 100.0 * np.mean(infeas_mask, axis=1)  # (T,)
    elif worst_scope != "network" and all_rx:
        raw_selected = np.array(
            [np.array(slacks_series[ri]) for ri in all_rx]
        )  # (k_total, T), raw sign
        infeas_mask = raw_selected > 0
        infeas_pct = 100.0 * np.mean(infeas_mask, axis=0)  # (T,)
    else:
        infeas_pct = None
    if infeas_pct is not None:

        ax_infeas = ax_slack.twinx()
        ax_infeas.plot(
            epoch_arr, infeas_pct,
            color=infeas_color, alpha=infeas_alpha, linewidth=infeas_lw,
            linestyle="-", label=str(slack_panel.get("infeasibility_label", "Infeasible %")),
        )
        ax_infeas.set_ylim(0, 105)
        _infeas_ylabel = _resolve_text(
            slack_panel.get("infeasibility_ylabel", "auto"),
            "Infeasible receivers (%)",
            slack_panel.get("infeasibility_ylabel_text"),
        )
        if _infeas_ylabel is not None:
            ax_infeas.set_ylabel(_infeas_ylabel, fontsize=label_fs, color=infeas_color)
        ax_infeas.tick_params(axis="y", labelcolor=infeas_color)
        if tick_fs is not None:
            ax_infeas.tick_params(labelsize=tick_fs)

    # Feasibility reference line + green/red region shading.
    ax_slack.axhline(y=0.0, color=zl_color, linestyle=zl_style, linewidth=zl_width, alpha=zl_alpha)
    ylims = ax_slack.get_ylim()
    if slack_conv == "max":
        # max convention: positive = feasible, negative = violated.
        if ylims[1] > 0:
            ax_slack.axhspan(0.0, ylims[1], color=feasible_color, alpha=feasible_alpha)
        if ylims[0] < 0:
            ax_slack.axhspan(ylims[0], 0.0, color=infeasible_color, alpha=infeasible_alpha)
    else:
        # min convention: negative = feasible, positive = violated.
        if ylims[0] < 0:
            ax_slack.axhspan(ylims[0], 0.0, color=feasible_color, alpha=feasible_alpha)
        if ylims[1] > 0:
            ax_slack.axhspan(0.0, ylims[1], color=infeasible_color, alpha=infeasible_alpha)
    _draw_cw(ax_slack)

    if slack_conv == "max":
        auto_slack_title = "Worst Constraint Slack (negative = violated)"
        auto_slack_ylabel = "Constraint Slack (rate \u2212 r_min)"
    else:
        auto_slack_title = "Worst Constraint Slack"
        auto_slack_ylabel = "Constraint Slack (r_min \u2212 rate)"

    _title = _resolve_text(
        slack_panel.get("title", "off"),
        auto_slack_title,
        slack_panel.get("title_text"),
    )
    if _title is not None:
        ax_slack.set_title(_title, fontsize=title_fs)

    _ylabel = _resolve_text(
        slack_panel.get("ylabel", "auto"),
        auto_slack_ylabel,
        slack_panel.get("ylabel_text"),
    )
    _slack_yaxis_color = str(slack_panel.get("ylabel_color", slack_avg_color))
    if _ylabel is not None:
        ax_slack.set_ylabel(_ylabel, fontsize=label_fs, color=_slack_yaxis_color)
    ax_slack.tick_params(axis="y", labelcolor=_slack_yaxis_color)

    _xlabel = _resolve_text(
        slack_panel.get("xlabel", "auto"),
        "Epoch",
        slack_panel.get("xlabel_text"),
    )
    if _xlabel is not None:
        ax_slack.set_xlabel(_xlabel, fontsize=label_fs)

    if not _is_off(slack_panel.get("legend", "auto")):
        # Merge legends from left (slack) and right (infeasibility %) axes.
        handles, labels = ax_slack.get_legend_handles_labels()
        if ax_infeas is not None:
            h2, l2 = ax_infeas.get_legend_handles_labels()
            handles += h2
            labels += l2
        ax_slack.legend(handles, labels, loc="best", fontsize=legend_fs)

    slack_xscale = str(slack_panel.get("xscale", "linear")).strip().lower()
    if slack_xscale == "log":
        ax_slack.set_xscale("log")
    elif slack_xscale == "compressed":
        if cw_epoch is not None:
            _comp = float(slack_panel.get("transient_compression", 5.0))
            ax_slack.set_xscale("function", functions=_make_compressed_transient(cw_epoch, _comp))
            _nticks_slack = int(slack_panel.get("target_nticks", 10))
            _apply_compressed_ticks(ax_slack, epoch_arr, cw_epoch, _comp, target_nticks=_nticks_slack)
    ax_slack.grid(True, alpha=grid_alpha, linestyle=ts_grid_ls)
    if tick_fs is not None:
        ax_slack.tick_params(labelsize=tick_fs)

    # == Right: 2×2 Power Scatter (distinct pairs) ============================
    if show_scatter and ax_scatter_grid is not None:
        from matplotlib.ticker import FormatStrFormatter

        scfg = scatter_panel  # scatter config now lives in panels.scatter
        transient_display = str(scfg.get("transient_display", "both")).strip().lower()
        coll_ms = float(scfg.get("collection_marker_size", 50))
        trans_ms = float(scfg.get("transient_marker_size", 20))
        trans_alpha = float(scfg.get("transient_alpha", 0.15))
        coll_alpha = float(scfg.get("collection_alpha", 0.6))
        axis_range = str(scfg.get("axis_range", "auto")).strip().lower()

        # Edge styling.
        coll_edgecolor = str(scfg.get("collection_edgecolor", "navy"))
        trans_edgecolor = str(scfg.get("transient_edgecolor", "none"))
        coll_lw = float(scfg.get("collection_linewidth", 0.5))
        trans_lw = float(scfg.get("transient_linewidth", 0.0))

        # Axis padding, bounds, grid, tick format.
        axis_pad = float(scfg.get("axis_padding", 0.05))
        show_bounds = bool(scfg.get("feasibility_bounds", True))
        bounds_color = str(scfg.get("bounds_color", "gray"))
        bounds_ls = str(scfg.get("bounds_linestyle", ":"))
        bounds_alpha = float(scfg.get("bounds_alpha", 0.5))
        bounds_lw = float(scfg.get("bounds_linewidth", 1.5))
        sc_grid_ls = str(scfg.get("grid_linestyle", "--"))
        tick_fmt_str = scfg.get("tick_format")

        # Legend placement.
        sc_legend_loc = str(scfg.get("legend_loc", "upper right"))
        legend_first_only = bool(row_mode.get("legend_first_subplot_only", True))

        # Subsampling.
        max_coll_raw = scfg.get("max_collection_points")
        max_trans_raw = scfg.get("max_transient_points")
        max_coll = int(max_coll_raw) if max_coll_raw is not None else None
        max_trans = int(max_trans_raw) if max_trans_raw is not None else None
        subsample_method = str(scfg.get("subsample_method", "stride")).strip().lower()

        # P_max for normalization.
        p_max = scfg.get("p_max")
        if axis_range == "unit" and p_max is not None and float(p_max) > 0:
            p_max = float(p_max)
        else:
            p_max = None

        # Select distinct pairs for the 2×2 grid from quantile groups.
        n_sc_pairs = int(scfg.get("n_pairs", 4))

        def _pairs_from_pool(pool: list[int], n: int) -> list[tuple[int, int]]:
            """Form up to *n* pairs from a receiver pool by sequential pairing."""
            pairs: list[tuple[int, int]] = []
            for i in range(0, len(pool) - 1, 2):
                pairs.append((min(pool[i], pool[i + 1]), max(pool[i], pool[i + 1])))
                if len(pairs) >= n:
                    break
            return pairs

        scatter_pairs: list[tuple[int, int]] = []
        seen_pairs: set[tuple[int, int]] = set()
        for gi, grx in enumerate(grouped_receivers):
            gcfg_i = group_cfgs[gi] if gi < len(group_cfgs) else {}
            n_sp = int(gcfg_i.get("scatter_pairs", 0))
            g_pairs = _pairs_from_pool(grx, n_sp)
            for p in g_pairs:
                if p not in seen_pairs and len(scatter_pairs) < n_sc_pairs:
                    scatter_pairs.append(p)
                    seen_pairs.add(p)

        if not scatter_pairs and len(all_rx) >= 2:
            scatter_pairs = [(min(all_rx[0], all_rx[1]),
                              max(all_rx[0], all_rx[1]))]

        # Extract power data for all receivers involved in scatter pairs.
        all_scatter_rx = set()
        for pa, pb in scatter_pairs:
            all_scatter_rx.add(pa)
            all_scatter_rx.add(pb)
        # Receivers already extracted in powers_series.
        extra_rx = all_scatter_rx - set(powers_series.keys())
        extra_powers: dict[int, list[float]] = {i: [] for i in extra_rx}
        if extra_rx:
            for entry in epochs_list:
                full_powers = entry.get("receiver_power_allocations")
                sel = entry.get("selected_receiver_trace", {})
                entry_rx_indices = sel.get("receiver_indices", [])
                rx_pos = {int(ri): pos for pos, ri in enumerate(entry_rx_indices)}
                for ri in extra_rx:
                    pos = rx_pos.get(ri)
                    if pos is not None:
                        extra_powers[ri].append(
                            float(sel.get("receiver_power_allocations", [0.0])[pos])
                        )
                    elif full_powers and ri < len(full_powers):
                        extra_powers[ri].append(float(full_powers[ri]))
                    else:
                        extra_powers[ri].append(0.0)
        all_powers = {**powers_series, **extra_powers}

        # Split into transient / collection epochs.
        transient_scope = str(scfg.get("transient_scope", "all")).strip().lower()
        if cw_epoch is not None:
            coll_mask = epoch_arr >= cw_epoch
        else:
            coll_mask = np.ones(len(epoch_arr), dtype=bool)
        trans_mask = ~coll_mask

        # Optionally restrict transient to the early window only.
        if transient_scope == "early" and collection_window is not None and collection_window > 0:
            early_end = int(epoch_arr[0]) + collection_window
            trans_mask = trans_mask & (epoch_arr < early_end)

        # Subsample indices for collection and transient.
        coll_indices = np.where(coll_mask)[0]
        trans_indices = np.where(trans_mask)[0]
        if max_coll is not None and len(coll_indices) > max_coll:
            coll_indices = coll_indices[_subsample_indices(len(coll_indices), max_coll, subsample_method, seed=seed)]
        if max_trans is not None and len(trans_indices) > max_trans:
            trans_indices = trans_indices[_subsample_indices(len(trans_indices), max_trans, subsample_method, seed=seed)]

        # Overlay data.
        net_overlay = (overlay_samples or {}).get(int(network_id))
        ov_arr_raw: np.ndarray | None = None
        if net_overlay is not None:
            ov_arr_raw = np.asarray(net_overlay, dtype=float)
            if ov_arr_raw.ndim != 2:
                ov_arr_raw = None

        show_overlay_legend = not _is_off(scatter_panel.get("legend", "auto"))

        for subplot_idx, (ri_x, ri_y) in enumerate(scatter_pairs):
            r, c = divmod(subplot_idx, 2)
            if r >= 2 or c >= 2:
                break
            ax = ax_scatter_grid[r][c]

            # Per-pair base colour from the dual panel palette.
            sc_color = pair_colors.get(subplot_idx, colors.get(ri_x, "tab:blue"))

            px_all = np.array(all_powers[ri_x])
            py_all = np.array(all_powers[ri_y])
            if p_max is not None:
                px_all = px_all / p_max
                py_all = py_all / p_max

            # Transient: fainter, smaller, no outline.
            if transient_display == "both" and len(trans_indices) > 0:
                ax.scatter(
                    px_all[trans_indices], py_all[trans_indices],
                    c=[sc_color],
                    s=trans_ms, alpha=trans_alpha,
                    edgecolors=trans_edgecolor, linewidth=trans_lw,
                )

            # Stationary / collection window.
            if len(coll_indices) > 0:
                ax.scatter(
                    px_all[coll_indices], py_all[coll_indices],
                    c=[sc_color],
                    s=coll_ms, alpha=coll_alpha,
                    edgecolors=coll_edgecolor, linewidth=coll_lw,
                )

            # Axis range with padding.
            if axis_range == "unit":
                ax.set_xlim(-axis_pad, 1.0 + axis_pad)
                ax.set_ylim(-axis_pad, 1.0 + axis_pad)

            # Feasibility bounds at 0 and 1.
            if show_bounds and axis_range == "unit":
                for bval in (0.0, 1.0):
                    ax.axhline(y=bval, color=bounds_color, linestyle=bounds_ls,
                               alpha=bounds_alpha, linewidth=bounds_lw)
                    ax.axvline(x=bval, color=bounds_color, linestyle=bounds_ls,
                               alpha=bounds_alpha, linewidth=bounds_lw)

            # Force square aspect ratio.
            ax.set_aspect("equal", adjustable="box")

            # Overlay: always uses the fixed overlay colour.
            if ov_arr_raw is not None and ov_arr_raw.shape[1] > max(ri_x, ri_y):
                ov_x = ov_arr_raw[:, ri_x]
                ov_y = ov_arr_raw[:, ri_y]
                if p_max is not None:
                    ov_x = ov_x / p_max
                    ov_y = ov_y / p_max
                ocfg = overlay_cfg or {}
                ax.scatter(
                    ov_x, ov_y,
                    marker=str(ocfg.get("marker", "x")),
                    c=overlay_color,
                    s=float(ocfg.get("marker_size", 30)),
                    alpha=float(ocfg.get("alpha", 0.5)),
                    label=str(ocfg.get("label", "Generated")),
                    edgecolors="none", zorder=5,
                )

            # Legend: respect first_subplot_only.
            if show_overlay_legend and ov_arr_raw is not None:
                if not legend_first_only or subplot_idx == 0:
                    ax.legend(loc=sc_legend_loc, fontsize=legend_fs)

            # Grid with dashed linestyle.
            ax.grid(True, alpha=grid_alpha, linestyle=sc_grid_ls)

            # Tick formatting.
            if tick_fs is not None:
                ax.tick_params(labelsize=tick_fs)
            if tick_fmt_str:
                ax.xaxis.set_major_formatter(FormatStrFormatter(tick_fmt_str))
                ax.yaxis.set_major_formatter(FormatStrFormatter(tick_fmt_str))

            # Per-pair axis labels.
            if axis_range == "unit" and p_max is not None:
                pair_xlabel = f"RX {ri_x} / $P_{{\\mathrm{{max}}}}$"
                pair_ylabel = f"RX {ri_y} / $P_{{\\mathrm{{max}}}}$"
            else:
                pair_xlabel = f"Power RX {ri_x}"
                pair_ylabel = f"Power RX {ri_y}"

            if r == 1:
                ax.set_xlabel(pair_xlabel, fontsize=label_fs - 1)
            else:
                plt.setp(ax.get_xticklabels(), visible=False)
            if c == 0:
                ax.set_ylabel(pair_ylabel, fontsize=label_fs - 1)
            else:
                plt.setp(ax.get_yticklabels(), visible=False)

            # Title: show pair on each subplot.
            pair_title = f"RX ({ri_x}, {ri_y})"
            ax.set_title(pair_title, fontsize=title_fs - 1)

        # Hide unused subplots if fewer than 4 pairs.
        for idx in range(len(scatter_pairs), 4):
            r, c = divmod(idx, 2)
            ax_scatter_grid[r][c].axis("off")

    # ---- save --------------------------------------------------------------
    # plt.tight_layout()

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"trace_row_network_{network_id}_{timestamp}.{fmt}"
    plot_path = output_dir / filename
    fig.savefig(plot_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved trace row plot to %s", plot_path)
    return plot_path
