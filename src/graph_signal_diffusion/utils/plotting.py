"""Plotting helpers for training summaries.

Provides `plot_epoch_summaries_jsonl` which reads a JSONL-formatted
`epoch_summaries.jsonl` file and writes PDF plots for training loss vs val loss,
train grad norm, train learning-rate evolution, and task metrics such as mse/mae/rmse.

This module is robust to missing keys and tolerates JSONL being appended to
while reading (uses a short retry loop on JSON errors).
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Plot size presets (larger canvases improve label/tick/legend readability).
EPOCH_FIGSIZE = (8.0, 5.0)
EPOCH_SUBPLOT_WIDTH = 8.0
EPOCH_SUBPLOT_ROW_HEIGHT = 3.8
EPOCH_SUBPLOT_ROW_HEIGHT_LARGE = 4.2
SELECTOR_SERIES_FIGSIZE = (8.0, 5.0)
SELECTOR_NETWORK_WIDTH = 12.0
SELECTOR_NETWORK_ROW_HEIGHT = 3.2
SELECTOR_FREQ_WIDTH = 11.0
SELECTOR_FREQ_ROW_HEIGHT = 2.8
SELECTOR_SAMPLING_WIDTH = 12.0
SELECTOR_SAMPLING_ROW_HEIGHT = 3.2
SELECTOR_SAMPLING_MAX_NODES = 20


def _smooth(values: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    """Apply causal exponential moving average (EMA) smoothing.

    Uses only past values so the final epoch value is never distorted by
    future look-ahead. NaN values are skipped and the EMA is carried forward,
    which keeps sparse series (e.g. val_loss on non-eval epochs) continuous.

    alpha=0.3 gives roughly comparable smoothing to a 5-point moving average
    while being strictly one-sided (no edge artefacts at either end).
    """
    if len(values) == 0:
        return values
    smoothed = np.empty_like(values, dtype=float)
    last = None
    for i, v in enumerate(values):
        if np.isnan(v):
            smoothed[i] = last if last is not None else np.nan
        elif last is None:
            last = float(v)
            smoothed[i] = last
        else:
            last = alpha * float(v) + (1.0 - alpha) * last
            smoothed[i] = last
    return smoothed


def _get_marker_frequency(n_points: int) -> int:
    """Determine marker frequency based on number of data points to avoid clutter."""
    if n_points <= 20:
        return 1  # Show all markers
    elif n_points <= 50:
        return 2  # Every 2nd point
    elif n_points <= 100:
        return 5  # Every 5th point
    elif n_points <= 200:
        return 10  # Every 10th point
    else:
        return max(1, n_points // 20)  # ~20 markers total


def _compute_robust_ylim_upper(
    *arrays: np.ndarray,
    ignore_first_frac: float = 0.1,
    percentile: float = 95.0,
    margin_factor: float = 1.1
) -> Optional[float]:
    """Compute robust upper y-axis limit ignoring early-epoch outliers.
    
    Parameters
    ----------
    *arrays : np.ndarray
        One or more arrays of values (e.g., train_loss, val_loss)
    ignore_first_frac : float
        Fraction of initial points to ignore when computing limit (default: 0.1 = 10%)
    percentile : float
        Percentile to use for upper limit computation (default: 95.0)
    margin_factor : float
        Multiply percentile value by this factor for visual margin (default: 1.1)
    
    Returns
    -------
    float or None
        Robust upper limit, or None if no valid data
    """
    # Collect all non-NaN values from all arrays, excluding initial fraction
    all_values = []
    for arr in arrays:
        if arr.size == 0:
            continue
        # Skip first N% of epochs
        skip_n = max(0, int(len(arr) * ignore_first_frac))
        valid_values = arr[skip_n:]
        valid_values = valid_values[~np.isnan(valid_values)]
        if valid_values.size > 0:
            all_values.append(valid_values)
    
    if not all_values:
        return None
    
    # Concatenate all values
    combined = np.concatenate(all_values)
    if combined.size == 0:
        return None
    
    # Compute percentile-based upper limit with margin
    upper = np.percentile(combined, percentile) * margin_factor
    return float(upper)


def _compute_zoomed_probability_ylim(
    values: np.ndarray,
    min_lower: float = 0.0,
    max_upper: float = 1.0,
    relative_margin: float = 0.12,
    min_margin: float = 0.01,
    min_span: float = 0.06,
) -> Tuple[float, float]:
    """
    Compute a zoomed y-range for probability-like values while respecting bounds.

    If values are constant/nearly-constant, expands a minimum visible span.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float(min_lower), float(max_upper)

    y_min = float(np.min(finite))
    y_max = float(np.max(finite))
    y_min = max(min_lower, min(max_upper, y_min))
    y_max = max(min_lower, min(max_upper, y_max))

    span = max(y_max - y_min, 0.0)
    margin = max(min_margin, span * relative_margin)
    lower = max(min_lower, y_min - margin)
    upper = min(max_upper, y_max + margin)

    if upper - lower < min_span:
        center = 0.5 * (lower + upper)
        half = 0.5 * min_span
        lower = center - half
        upper = center + half
        if lower < min_lower:
            upper = min(max_upper, upper + (min_lower - lower))
            lower = min_lower
        if upper > max_upper:
            lower = max(min_lower, lower - (upper - max_upper))
            upper = max_upper

    if upper <= lower:
        lower, upper = float(min_lower), float(max_upper)
    return float(lower), float(upper)


def _select_sampling_heatmap_node_indices(
    activity_signal: np.ndarray,
    max_nodes: int = SELECTOR_SAMPLING_MAX_NODES,
) -> np.ndarray:
    """
    Select a node subset for sampling heatmap readability.

    Strategy:
      - Keep all nodes if N <= max_nodes
      - Else pick least-active and most-active halves (10 + 10 when max_nodes=20)
    """
    activity = np.asarray(activity_signal, dtype=float).reshape(-1)
    num_nodes = int(activity.size)
    if num_nodes <= max_nodes:
        return np.arange(num_nodes, dtype=int)
    if max_nodes <= 0:
        return np.array([], dtype=int)

    act = activity.copy()
    act[~np.isfinite(act)] = 0.0

    k_low = max_nodes // 2
    k_high = max_nodes - k_low
    sorted_idx = np.argsort(act, kind="stable")
    low_idx = sorted_idx[:k_low].tolist()
    high_idx = sorted_idx[-k_high:].tolist()

    picked: List[int] = []
    seen = set()
    for idx in low_idx + high_idx:
        idx_int = int(idx)
        if idx_int not in seen:
            picked.append(idx_int)
            seen.add(idx_int)

    if len(picked) < max_nodes:
        for idx in sorted_idx.tolist():
            idx_int = int(idx)
            if idx_int not in seen:
                picked.append(idx_int)
                seen.add(idx_int)
            if len(picked) >= max_nodes:
                break

    picked = sorted(picked[:max_nodes])
    return np.asarray(picked, dtype=int)


def _resolve_sampling_level_cumulative_factors(
    selector_sampling_diagnostics: Dict[str, Any],
    level_keys: List[Tuple[int, str]],
) -> np.ndarray:
    """
    Resolve cumulative downsampling factors used to weight per-level probabilities.

    Priority:
      1) `cumulative_downsampling_factor_by_level`
      2) `gamma_by_level` -> cumulative product by level index
      3) inferred from `available_count_by_level` + `probe_graph_count`
      4) fallback geometric progression [1, 2, 4, ...]
    """
    num_levels = len(level_keys)
    if num_levels == 0:
        return np.zeros(0, dtype=float)

    # 1) Explicit cumulative factors.
    cum_factor_raw = selector_sampling_diagnostics.get(
        "cumulative_downsampling_factor_by_level", {}
    )
    if isinstance(cum_factor_raw, dict):
        factors: List[float] = []
        ok = True
        for _, level_key in level_keys:
            v = cum_factor_raw.get(level_key, None)
            try:
                factors.append(float(v))
            except (TypeError, ValueError):
                ok = False
                break
        if ok and len(factors) == num_levels:
            arr = np.asarray(factors, dtype=float)
            arr[~np.isfinite(arr)] = 1.0
            arr = np.clip(arr, 1.0, None)
            return arr

    # 2) Per-level gamma config (if attached).
    gamma_raw = selector_sampling_diagnostics.get("gamma_by_level", None)
    if isinstance(gamma_raw, dict):
        gammas: List[float] = []
        ok = True
        for level_idx, level_key in level_keys:
            g = gamma_raw.get(level_key, gamma_raw.get(str(level_idx), None))
            try:
                gammas.append(float(g))
            except (TypeError, ValueError):
                ok = False
                break
        if ok and len(gammas) == num_levels:
            gammas_arr = np.asarray(gammas, dtype=float)
            gammas_arr[~np.isfinite(gammas_arr)] = 1.0
            gammas_arr = np.clip(gammas_arr, 1.0, None)
            return np.cumprod(gammas_arr)
    if isinstance(gamma_raw, (list, tuple)):
        gamma_list = list(gamma_raw)
        gammas = []
        ok = True
        for level_idx, _ in level_keys:
            if level_idx < 0 or level_idx >= len(gamma_list):
                ok = False
                break
            try:
                gammas.append(float(gamma_list[level_idx]))
            except (TypeError, ValueError):
                ok = False
                break
        if ok and len(gammas) == num_levels:
            gammas_arr = np.asarray(gammas, dtype=float)
            gammas_arr[~np.isfinite(gammas_arr)] = 1.0
            gammas_arr = np.clip(gammas_arr, 1.0, None)
            return np.cumprod(gammas_arr)

    # 3) Infer cumulative factors from expected active-node counts.
    available_by_level = selector_sampling_diagnostics.get("available_count_by_level", {})
    probe_graph_count = np.asarray(
        selector_sampling_diagnostics.get("probe_graph_count", []), dtype=float
    )
    if isinstance(available_by_level, dict) and probe_graph_count.ndim == 1 and probe_graph_count.size > 0:
        active_node_means: List[float] = []
        for _, level_key in level_keys:
            avail_raw = available_by_level.get(level_key, None)
            avail = np.asarray(avail_raw, dtype=float) if avail_raw is not None else np.array([])
            if avail.ndim != 2 or avail.shape[0] <= 0 or avail.shape[1] <= 0:
                active_node_means.append(float("nan"))
                continue

            steps = min(avail.shape[0], probe_graph_count.size)
            avail = avail[:steps, :]
            graphs = np.clip(probe_graph_count[:steps], 1.0, None)
            active_nodes_per_graph = np.nansum(avail, axis=1) / graphs
            if active_nodes_per_graph.size == 0:
                active_node_means.append(float("nan"))
            else:
                active_node_means.append(float(np.nanmean(active_nodes_per_graph)))

        active_arr = np.asarray(active_node_means, dtype=float)
        finite = np.isfinite(active_arr) & (active_arr > 0.0)
        if np.any(finite):
            base = float(np.nanmax(active_arr[finite]))
            factors = np.ones_like(active_arr, dtype=float)
            factors[finite] = base / active_arr[finite]
            factors[~np.isfinite(factors)] = 1.0
            factors = np.clip(factors, 1.0, None)
            return factors

    # 4) Final fallback.
    return np.asarray([2.0**i for i in range(num_levels)], dtype=float)


def _select_sampling_heatmap_nodes_from_reward(
    selector_sampling_diagnostics: Dict[str, Any],
    level_keys: List[Tuple[int, str]],
    matrix_by_level: Dict[str, Any],
    max_nodes: int = SELECTOR_SAMPLING_MAX_NODES,
) -> np.ndarray:
    """
    Select one shared node subset across all selector levels.

    Reward for node i:
        reward(i) = sum_l [ p_l(i) * c_l ]
    where p_l(i) is mean selection probability at level l and c_l is the
    cumulative downsampling factor for that level.
    """
    if len(level_keys) == 0:
        return np.array([], dtype=int)

    num_nodes: Optional[int] = None
    for _, level_key in level_keys:
        raw = matrix_by_level.get(level_key, None)
        mat = np.asarray(raw, dtype=float) if raw is not None else np.array([])
        if mat.ndim == 2 and mat.shape[1] > 0:
            num_nodes = int(mat.shape[1])
            break
    if num_nodes is None or num_nodes <= 0:
        return np.array([], dtype=int)

    cumulative_factors = _resolve_sampling_level_cumulative_factors(
        selector_sampling_diagnostics=selector_sampling_diagnostics,
        level_keys=level_keys,
    )
    reward = np.zeros(num_nodes, dtype=float)
    for factor, (_, level_key) in zip(cumulative_factors.tolist(), level_keys):
        raw = matrix_by_level.get(level_key, None)
        mat = np.asarray(raw, dtype=float) if raw is not None else np.array([])
        if mat.ndim != 2 or mat.shape[1] != num_nodes:
            continue
        probs = np.nanmean(mat, axis=0)
        probs = np.where(np.isfinite(probs), probs, 0.0)
        reward += float(factor) * probs

    return _select_sampling_heatmap_node_indices(reward, max_nodes=max_nodes)


def _read_jsonl(path: Path, tries: int = 3, wait: float = 0.2) -> List[dict]:
    for i in range(tries):
        try:
            items = []
            with path.open("r") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    items.append(json.loads(line))
            return items
        except (json.JSONDecodeError, OSError):
            if i + 1 == tries:
                raise
            time.sleep(wait)
    return []


def _extract_series(rows: List[dict], key_variants: Iterable[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Return epochs and values array for first matching key variant present.
    If no key is present, returns empty arrays.
    """
    if not rows:
        return np.array([], dtype=int), np.array([], dtype=float)

    epochs = []
    vals = []
    for i, r in enumerate(rows):
        e = r.get("epoch", None)
        if e is None:
            e = i + 1
        epochs.append(int(e))

        v = None
        for kv in key_variants:
            # direct key
            if kv in r:
                v = r[kv]
                break
            # try prefixed variants (val_*, train_*, task_*) matching
            for k in r.keys():
                if k.lower().endswith(kv.lower()):
                    v = r[k]
                    break
            if v is not None:
                break

        vals.append(np.nan if v is None else float(v))

    return np.array(epochs, dtype=int), np.array(vals, dtype=float)


def _extract_ema_metrics(rows: List[dict]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Extract all best_model_ema_metrics as individual series.
    
    Returns:
        Dict mapping metric name to (epochs, values) tuple.
    """
    if not rows:
        return {}
    
    # Collect all unique metric names from ema_metrics dicts
    all_metric_names = set()
    for r in rows:
        ema_dict = r.get("best_model_ema_metrics", {})
        if isinstance(ema_dict, dict):
            all_metric_names.update(ema_dict.keys())
    
    # Extract series for each metric
    result = {}
    for metric_name in all_metric_names:
        epochs = []
        vals = []
        for i, r in enumerate(rows):
            e = r.get("epoch", None)
            if e is None:
                e = i + 1
            epochs.append(int(e))
            
            ema_dict = r.get("best_model_ema_metrics", {})
            if isinstance(ema_dict, dict) and metric_name in ema_dict:
                v = ema_dict[metric_name]
                vals.append(np.nan if v is None else float(v))
            else:
                vals.append(np.nan)
        
        result[metric_name] = (np.array(epochs, dtype=int), np.array(vals, dtype=float))
    
    return result


def plot_epoch_summaries_jsonl(
    epoch_jsonl_path: str | Path,
    out_dir: str | Path,
    overwrite: bool = True,
    grad_logscale: bool = True,
    save_pdf: bool = True,
):
    """Plot training evolution from an `epoch_summaries.jsonl` file.

    Parameters
    ----------
    epoch_jsonl_path:
        Path to the JSONL file where each line is a JSON object for an epoch summary.
    out_dir:
        Directory where plots will be saved.
    overwrite:
        If True, previous plots with the same filename will be overwritten.
    grad_logscale:
        Use log scale for gradient norm plot.
    save_pdf:
        Save output files as PDF (if False, PNG is used).
    """
    p = Path(epoch_jsonl_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(p)
    if not rows:
        return

    # Epochs are extracted with first available epoch information
    # Extract series for train/val loss, grad, lr, and task metrics
    epochs, train_loss = _extract_series(rows, ["train_loss"])
    _, val_loss = _extract_series(rows, ["val_loss"])
    _, train_val_loss = _extract_series(rows, ["train-val_loss"]) # optional
    _, grad_norm = _extract_series(rows, ["train_grad_norm", "grad_norm", "gradnorm"])  # possible variants
    _, train_lr = _extract_series(rows, ["train_lr", "train_learning_rate", "learning_rate", "lr"])

    # Task metrics computed for the validation dataset.
    # New convention: val_{metric} (no task_ infix).
    # Old convention kept in fallback chains for backward-compat with historical JSONL logs.
    # Three-tier metric system: primary keys are return_* (Tier 1, log returns), legacy aliases
    # (mae, rmse, crps_mean, etc.) also point to return-space metrics.
    _, task_mse = _extract_series(rows, ["val_price_mse", "val_mse", "val_task_price_mse", "val_task_mse", "task_mse", "mse"])
    _, task_mae = _extract_series(rows, ["val_price_mae", "val_mae", "val_task_price_mae", "val_task_mae", "task_mae", "mae"])
    _, task_rmse = _extract_series(rows, ["val_price_rmse", "val_rmse", "val_task_price_rmse", "val_task_rmse", "task_rmse", "rmse"])
    _, task_crps_mean = _extract_series(rows, ["val_crps_mean", "val_task_crps_mean", "task_crps_mean", "crps_mean"])
    _, task_crps_std = _extract_series(rows, ["val_crps_std", "val_task_crps_std", "task_crps_std", "crps_std"])
    _, task_mis_90 = _extract_series(rows, ["val_mis_90", "val_task_mis_90", "task_mis_90", "mis_90"])

    # Tier 1 — primary metrics on daily log returns (direction accuracy, correlation)
    _, task_direction_acc = _extract_series(rows, ["val_direction_accuracy", "val_task_direction_accuracy", "task_direction_accuracy"])
    _, task_return_corr = _extract_series(rows, ["val_volclustering_gen", "val_return_correlation", "val_task_return_correlation", "task_return_correlation"])
    _, task_volclustering_real = _extract_series(rows, ["val_volclustering_real"])
    _, task_return_mae = _extract_series(rows, ["val_return_mae", "val_task_return_mae", "task_return_mae"])
    _, task_return_rmse = _extract_series(rows, ["val_return_rmse", "val_task_return_rmse", "task_return_rmse"])
    _, task_return_crps = _extract_series(rows, ["val_return_crps", "val_task_return_crps", "task_return_crps"])

    # Tier 2 — secondary metrics on closing prices
    _, task_price_mae = _extract_series(rows, ["val_price_mae", "val_task_price_mae", "task_price_mae"])
    _, task_price_rmse = _extract_series(rows, ["val_price_rmse", "val_task_price_rmse", "task_price_rmse"])
    _, task_price_crps = _extract_series(rows, ["val_price_crps", "val_task_price_crps", "task_price_crps"])

    # Tier 3 — cumulative return metrics (final horizon)
    _, task_cum_T_mae = _extract_series(rows, ["val_cum_T_mae", "val_task_cum_T_mae", "task_cum_T_mae"])
    _, task_cum_T_rmse = _extract_series(rows, ["val_cum_T_rmse", "val_task_cum_T_rmse", "task_cum_T_rmse"])
    _, task_cum_T_dir_acc = _extract_series(rows, ["val_cum_T_direction_accuracy", "val_task_cum_T_direction_accuracy", "task_cum_T_direction_accuracy"])
    _, task_cum_T_crps = _extract_series(rows, ["val_cum_T_crps", "val_task_cum_T_crps", "task_cum_T_crps"])

    # Optional: Task metrics computed also for a subset (train-val) dataset
    _, train_val_task_mse = _extract_series(rows, ["train-val_price_mse", "train-val_mse", "train-val_task_price_mse", "train-val_task_mse"])
    _, train_val_task_mae = _extract_series(rows, ["train-val_price_mae", "train-val_mae", "train-val_task_price_mae", "train-val_task_mae"])
    _, train_val_task_rmse = _extract_series(rows, ["train-val_price_rmse", "train-val_rmse", "train-val_task_price_rmse", "train-val_task_rmse"])
    _, train_val_task_crps_mean = _extract_series(rows, ["train-val_crps_mean", "train-val_task_crps_mean"])
    _, train_val_task_crps_std = _extract_series(rows, ["train-val_crps_std", "train-val_task_crps_std"])
    _, train_val_task_mis_90 = _extract_series(rows, ["train-val_mis_90", "train-val_task_mis_90"])

    # Train-val three-tier extras
    _, train_val_direction_acc = _extract_series(rows, ["train-val_direction_accuracy", "train-val_task_direction_accuracy"])
    _, train_val_return_corr = _extract_series(rows, ["train-val_volclustering_gen", "train-val_return_correlation", "train-val_task_return_correlation"])
    _, train_val_return_mae = _extract_series(rows, ["train-val_return_mae", "train-val_task_return_mae"])
    _, train_val_return_rmse = _extract_series(rows, ["train-val_return_rmse", "train-val_task_return_rmse"])
    _, train_val_cum_T_mae = _extract_series(rows, ["train-val_cum_T_mae", "train-val_task_cum_T_mae"])
    _, train_val_cum_T_dir_acc = _extract_series(rows, ["train-val_cum_T_direction_accuracy", "train-val_task_cum_T_direction_accuracy"])
    
    # Wireless Resource Allocation metrics (validation)
    _, wra_sum_rate_gen = _extract_series(rows, ["val_sum_rate_generated", "val_task_sum_rate_generated", "task_sum_rate_generated"])
    _, wra_sum_rate_real = _extract_series(rows, ["val_sum_rate_real", "val_task_sum_rate_real", "task_sum_rate_real"])
    _, wra_min_rate_gen = _extract_series(rows, ["val_min_rate_generated", "val_task_min_rate_generated", "task_min_rate_generated"])
    _, wra_min_rate_real = _extract_series(rows, ["val_min_rate_real", "val_task_min_rate_real", "task_min_rate_real"])
    _, wra_fairness_gen = _extract_series(rows, ["val_fairness_generated", "val_task_fairness_generated", "task_fairness_generated"])
    _, wra_fairness_real = _extract_series(rows, ["val_fairness_real", "val_task_fairness_real", "task_fairness_real"])
    _, wra_power_violation_pct = _extract_series(
        rows,
        ["val_power_violation_percentage_generated", "val_task_power_violation_percentage_generated", "task_power_violation_percentage_generated"],
    )
    _, wra_rate_violation_pct = _extract_series(
        rows,
        ["val_rate_violation_percentage_generated", "val_task_rate_violation_percentage_generated", "task_rate_violation_percentage_generated"],
    )
    # Backward-compatible fallback: legacy violation "rate" keys are fractions [0, 1].
    _, wra_power_violation_rate = _extract_series(
        rows,
        ["val_power_violation_rate_generated", "val_task_power_violation_rate_generated", "task_power_violation_rate_generated"],
    )
    _, wra_rate_violation_rate = _extract_series(
        rows,
        ["val_rate_violation_rate_generated", "val_task_rate_violation_rate_generated", "task_rate_violation_rate_generated"],
    )
    _, wra_rate_mean_violation_gap_pct_gen = _extract_series(
        rows,
        ["val_rate_mean_violation_gap_pct_generated", "val_task_rate_mean_violation_gap_pct_generated", "task_rate_mean_violation_gap_pct_generated",
         "val_rate_mean_violation_gap_generated", "val_task_rate_mean_violation_gap_generated", "task_rate_mean_violation_gap_generated"],
    )
    _, wra_rate_mean_violation_gap_pct_real = _extract_series(
        rows,
        ["val_rate_mean_violation_gap_pct_real", "val_task_rate_mean_violation_gap_pct_real", "task_rate_mean_violation_gap_pct_real",
         "val_rate_mean_violation_gap_real", "val_task_rate_mean_violation_gap_real", "task_rate_mean_violation_gap_real"],
    )
    _, wra_sum_rate_gap = _extract_series(rows, ["val_sum_rate_gap_pct", "val_task_sum_rate_gap_pct", "task_sum_rate_gap_pct"])
    _, wra_min_rate_gap = _extract_series(rows, ["val_min_rate_gap_pct", "val_task_min_rate_gap_pct", "task_min_rate_gap_pct"])
    _, wra_rate_1pct_gap = _extract_series(rows, ["val_rate_1pct_gap_pct", "val_task_rate_1pct_gap_pct", "task_rate_1pct_gap_pct"])
    _, wra_rate_5pct_gap = _extract_series(rows, ["val_rate_5pct_gap_pct", "val_task_rate_5pct_gap_pct", "task_rate_5pct_gap_pct"])
    
    # Wireless Resource Allocation metrics (train-val)
    _, train_val_wra_sum_rate_gen = _extract_series(rows, ["train-val_sum_rate_generated", "train-val_task_sum_rate_generated"])
    _, train_val_wra_sum_rate_real = _extract_series(rows, ["train-val_sum_rate_real", "train-val_task_sum_rate_real"])
    _, train_val_wra_fairness_gen = _extract_series(rows, ["train-val_fairness_generated", "train-val_task_fairness_generated"])
    _, train_val_wra_fairness_real = _extract_series(rows, ["train-val_fairness_real", "train-val_task_fairness_real"])
    _, train_val_wra_power_violation_pct = _extract_series(
        rows,
        ["train-val_power_violation_percentage_generated", "train-val_task_power_violation_percentage_generated"],
    )
    _, train_val_wra_rate_violation_pct = _extract_series(
        rows,
        ["train-val_rate_violation_percentage_generated", "train-val_task_rate_violation_percentage_generated"],
    )
    _, train_val_wra_power_violation_rate = _extract_series(
        rows,
        ["train-val_power_violation_rate_generated", "train-val_task_power_violation_rate_generated"],
    )
    _, train_val_wra_rate_violation_rate = _extract_series(
        rows,
        ["train-val_rate_violation_rate_generated", "train-val_task_rate_violation_rate_generated"],
    )
    _, train_val_wra_rate_mean_violation_gap_pct_gen = _extract_series(
        rows,
        ["train-val_rate_mean_violation_gap_pct_generated", "train-val_task_rate_mean_violation_gap_pct_generated",
         "train-val_rate_mean_violation_gap_generated", "train-val_task_rate_mean_violation_gap_generated"],
    )
    _, train_val_wra_rate_mean_violation_gap_pct_real = _extract_series(
        rows,
        ["train-val_rate_mean_violation_gap_pct_real", "train-val_task_rate_mean_violation_gap_pct_real",
         "train-val_rate_mean_violation_gap_real", "train-val_task_rate_mean_violation_gap_real"],
    )
    _, train_val_wra_rate_1pct_gap = _extract_series(rows, ["train-val_rate_1pct_gap_pct", "train-val_task_rate_1pct_gap_pct"])
    _, train_val_wra_rate_5pct_gap = _extract_series(rows, ["train-val_rate_5pct_gap_pct", "train-val_task_rate_5pct_gap_pct"])

    # Best model tracker metrics (composite score and EMA-smoothed metrics)
    _, best_model_composite_score = _extract_series(rows, ["best_model_composite_score"])
    best_model_ema_metrics = _extract_ema_metrics(rows)

    def _prefer_percentage_series(percentage_series: np.ndarray, fractional_series: np.ndarray) -> np.ndarray:
        has_percentage = percentage_series.size and np.any(~np.isnan(percentage_series))
        if has_percentage:
            return percentage_series
        return fractional_series * 100.0

    wra_power_violation = _prefer_percentage_series(wra_power_violation_pct, wra_power_violation_rate)
    wra_rate_violation = _prefer_percentage_series(wra_rate_violation_pct, wra_rate_violation_rate)
    train_val_wra_power_violation = _prefer_percentage_series(
        train_val_wra_power_violation_pct, train_val_wra_power_violation_rate
    )
    train_val_wra_rate_violation = _prefer_percentage_series(
        train_val_wra_rate_violation_pct, train_val_wra_rate_violation_rate
    )
    
    fmt = "pdf" if save_pdf else "png"

    # 1) train vs val loss
    if train_loss.size or val_loss.size or train_val_loss.size:
        plt.figure(figsize=EPOCH_FIGSIZE)
        marker_freq = _get_marker_frequency(len(epochs))
        
        if train_loss.size:
            # Raw values as transparent background
            plt.plot(epochs, train_loss, color='C0', alpha=0.2, linewidth=1, label='_nolegend_')
            # Smoothed values
            smoothed = _smooth(train_loss)
            # Create marker indices
            marker_indices = np.arange(0, len(epochs), marker_freq)
            plt.plot(epochs, smoothed, label="Train", color='C0', linewidth=2)
            plt.plot(epochs[marker_indices], smoothed[marker_indices], 'o', color='C0', markersize=4)
            
        if val_loss.size and np.any(~np.isnan(val_loss)):
            plt.plot(epochs, val_loss, color='C1', alpha=0.2, linewidth=1, label='_nolegend_')
            val_mask = ~np.isnan(val_loss)
            val_epochs_plot = epochs[val_mask]
            smoothed = _smooth(val_loss[val_mask])
            marker_freq_val = _get_marker_frequency(len(val_epochs_plot))
            marker_indices = np.arange(0, len(val_epochs_plot), marker_freq_val)
            plt.plot(val_epochs_plot, smoothed, label="Val", color='C1', linewidth=2)
            plt.plot(val_epochs_plot[marker_indices], smoothed[marker_indices], 'o', color='C1', markersize=4)
            
        if train_val_loss.size and np.any(~np.isnan(train_val_loss)):
            plt.plot(epochs, train_val_loss, color='C2', alpha=0.2, linewidth=1, label='_nolegend_')
            tv_mask = ~np.isnan(train_val_loss)
            tv_epochs_plot = epochs[tv_mask]
            smoothed = _smooth(train_val_loss[tv_mask])
            marker_freq_tv = _get_marker_frequency(len(tv_epochs_plot))
            marker_indices = np.arange(0, len(tv_epochs_plot), marker_freq_tv)
            plt.plot(tv_epochs_plot, smoothed, label="Train-val", color='C2', linewidth=2)
            plt.plot(tv_epochs_plot[marker_indices], smoothed[marker_indices], 'o', color='C2', markersize=4)
            
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        
        # Compute robust y-axis upper limit ignoring early-epoch outliers
        robust_upper = _compute_robust_ylim_upper(
            train_loss, val_loss, train_val_loss,
            ignore_first_frac=0.1,  # Ignore first 10% of epochs
            percentile=95.0,  # Use 95th percentile
            margin_factor=1.1  # Add 10% visual margin
        )
        if robust_upper is not None:
            plt.ylim(top=robust_upper)
        
        plt.legend()
        plt.grid(alpha=0.1)
        # plt.tight_layout()
        f1 = out / f"training_loss.{fmt}"
        plt.savefig(f1, dpi=150)
        plt.close()


    # 2) grad norm
    if grad_norm.size:
        plt.figure(figsize=EPOCH_FIGSIZE)
        marker_freq = _get_marker_frequency(len(epochs))
        
        # Raw values as transparent background
        plt.plot(epochs, grad_norm, color='C0', alpha=0.2, linewidth=1, label='_nolegend_')
        # Smoothed values
        smoothed = _smooth(grad_norm)
        marker_indices = np.arange(0, len(epochs), marker_freq)
        plt.plot(epochs, smoothed, color='C0', linewidth=2)
        plt.plot(epochs[marker_indices], smoothed[marker_indices], 'o', color='C0', markersize=4)
        
        plt.xlabel("Epoch")
        plt.ylabel("Gradient norm")
        plt.ylim(bottom = max(0.01, plt.gca().get_ylim()[0]),
                 top = min(100, plt.gca().get_ylim()[1])
        )  # set upper limit to 10 or current upper limit, whichever is smaller

        if grad_logscale:
            plt.yscale("log")
        plt.grid(alpha=0.1)
        # plt.tight_layout()
        f2 = out / f"grad_norm.{fmt}"
        plt.savefig(f2, dpi=150)
        plt.close()

    # 3) learning rate
    if train_lr.size and np.any(~np.isnan(train_lr)):
        lr_mask = ~np.isnan(train_lr)
        lr_epochs = epochs[lr_mask]
        lr_values = train_lr[lr_mask]

        plt.figure(figsize=EPOCH_FIGSIZE)
        marker_freq = _get_marker_frequency(len(lr_epochs))

        plt.plot(lr_epochs, lr_values, color='C0', alpha=0.2, linewidth=1, label='_nolegend_')
        smoothed = _smooth(lr_values)
        marker_indices = np.arange(0, len(lr_epochs), marker_freq)
        plt.plot(lr_epochs, smoothed, color='C0', linewidth=2)
        plt.plot(lr_epochs[marker_indices], smoothed[marker_indices], 'o', color='C0', markersize=4)

        plt.xlabel("Epoch")
        plt.ylabel("Learning rate")
        plt.grid(alpha=0.1)
        f_lr = out / f"learning_rate.{fmt}"
        plt.savefig(f_lr, dpi=150)
        plt.close()

    # 4) Deterministic metrics (MAE) - twin axes if 2 datasets, subplots if more
    has_val_mae = task_mae.size and np.any(~np.isnan(task_mae))
    has_train_val_mae = train_val_task_mae.size and np.any(~np.isnan(train_val_task_mae))
    n_mae_datasets = sum([has_val_mae, has_train_val_mae])
    
    if n_mae_datasets > 0:
        if n_mae_datasets == 2:
            # Use twin axes for 2 datasets
            fig, ax1 = plt.subplots(figsize=EPOCH_FIGSIZE)
            
            # Val MAE on left axis
            color1 = 'tab:blue'
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Val MAE', color=color1)
            ax1.tick_params(axis='y', labelcolor=color1)
            ax1.grid(alpha=0.1)
            
            mae_mask = ~np.isnan(task_mae)
            mae_epochs = epochs[mae_mask]
            mae_values = task_mae[mae_mask]
            marker_freq = _get_marker_frequency(len(mae_epochs))
            marker_indices = np.arange(0, len(mae_epochs), marker_freq)
            ax1.plot(mae_epochs, mae_values, color=color1, linewidth=2)
            ax1.plot(mae_epochs[marker_indices], mae_values[marker_indices], 'o', color=color1, markersize=4)
            
            # Train-val MAE on right axis
            ax2 = ax1.twinx()
            color2 = 'tab:orange'
            ax2.set_ylabel('Train-val MAE', color=color2)
            ax2.tick_params(axis='y', labelcolor=color2)
            
            train_val_mae_mask = ~np.isnan(train_val_task_mae)
            train_val_mae_epochs = epochs[train_val_mae_mask]
            train_val_mae_values = train_val_task_mae[train_val_mae_mask]
            marker_freq = _get_marker_frequency(len(train_val_mae_epochs))
            marker_indices = np.arange(0, len(train_val_mae_epochs), marker_freq)
            ax2.plot(train_val_mae_epochs, train_val_mae_values, color=color2, linewidth=2)
            ax2.plot(train_val_mae_epochs[marker_indices], train_val_mae_values[marker_indices], 's', color=color2, markersize=4)
            
            f3 = out / f"mae.{fmt}"
            plt.savefig(f3, dpi=150)
            plt.close()
        elif n_mae_datasets > 2:
            # Use subplots for >2 datasets
            fig, axes = plt.subplots(
                n_mae_datasets,
                1,
                figsize=(EPOCH_SUBPLOT_WIDTH, EPOCH_SUBPLOT_ROW_HEIGHT * n_mae_datasets),
                sharex=True,
            )
            if n_mae_datasets == 1:
                axes = [axes]
            
            plot_idx = 0
            if has_val_mae:
                ax = axes[plot_idx]
                mae_mask = ~np.isnan(task_mae)
                mae_epochs = epochs[mae_mask]
                mae_values = task_mae[mae_mask]
                marker_freq = _get_marker_frequency(len(mae_epochs))
                marker_indices = np.arange(0, len(mae_epochs), marker_freq)
                ax.plot(mae_epochs, mae_values, color='tab:blue', linewidth=2)
                ax.plot(mae_epochs[marker_indices], mae_values[marker_indices], 'o', color='tab:blue', markersize=4)
                ax.set_ylabel('Val MAE')
                ax.grid(alpha=0.1)
                plot_idx += 1
            
            if has_train_val_mae:
                ax = axes[plot_idx]
                train_val_mae_mask = ~np.isnan(train_val_task_mae)
                train_val_mae_epochs = epochs[train_val_mae_mask]
                train_val_mae_values = train_val_task_mae[train_val_mae_mask]
                marker_freq = _get_marker_frequency(len(train_val_mae_epochs))
                marker_indices = np.arange(0, len(train_val_mae_epochs), marker_freq)
                ax.plot(train_val_mae_epochs, train_val_mae_values, color='tab:orange', linewidth=2)
                ax.plot(train_val_mae_epochs[marker_indices], train_val_mae_values[marker_indices], 's', color='tab:orange', markersize=4)
                ax.set_ylabel('Train-val MAE')
                ax.grid(alpha=0.1)
                plot_idx += 1
            
            axes[-1].set_xlabel('Epoch')
            # plt.tight_layout()
            f3 = out / f"mae.{fmt}"
            plt.savefig(f3, dpi=150)
            plt.close()
        else:
            # Single dataset - simple plot
            plt.figure(figsize=EPOCH_FIGSIZE)
            if has_val_mae:
                mae_mask = ~np.isnan(task_mae)
                mae_epochs = epochs[mae_mask]
                mae_values = task_mae[mae_mask]
                marker_freq = _get_marker_frequency(len(mae_epochs))
                marker_indices = np.arange(0, len(mae_epochs), marker_freq)
                plt.plot(mae_epochs, mae_values, label="Val", color='tab:blue', linewidth=2)
                plt.plot(mae_epochs[marker_indices], mae_values[marker_indices], 'o', color='tab:blue', markersize=4)
            elif has_train_val_mae:
                train_val_mae_mask = ~np.isnan(train_val_task_mae)
                train_val_mae_epochs = epochs[train_val_mae_mask]
                train_val_mae_values = train_val_task_mae[train_val_mae_mask]
                marker_freq = _get_marker_frequency(len(train_val_mae_epochs))
                marker_indices = np.arange(0, len(train_val_mae_epochs), marker_freq)
                plt.plot(train_val_mae_epochs, train_val_mae_values, label="Train-val", color='tab:orange', linewidth=2)
                plt.plot(train_val_mae_epochs[marker_indices], train_val_mae_values[marker_indices], 's', color='tab:orange', markersize=4)
            
            plt.xlabel("Epoch")
            plt.ylabel("MAE")
            plt.legend(loc='upper right')
            plt.grid(alpha=0.1)
            f3 = out / f"mae.{fmt}"
            plt.savefig(f3, dpi=150)
            plt.close()


    # 4) Probabilistic metrics (CRPS) - subplots for each dataset with twin y-axes for mean and std
    has_val_crps_mean = task_crps_mean.size and np.any(~np.isnan(task_crps_mean))
    has_val_crps_std = task_crps_std.size and np.any(~np.isnan(task_crps_std))
    has_train_val_crps_mean = train_val_task_crps_mean.size and np.any(~np.isnan(train_val_task_crps_mean))
    has_train_val_crps_std = train_val_task_crps_std.size and np.any(~np.isnan(train_val_task_crps_std))
    
    has_val_crps = has_val_crps_mean and has_val_crps_std
    has_train_val_crps = has_train_val_crps_mean and has_train_val_crps_std
    n_crps_datasets = sum([has_val_crps, has_train_val_crps])
    
    if n_crps_datasets > 0:
        fig, axes = plt.subplots(
            n_crps_datasets,
            1,
            figsize=(EPOCH_SUBPLOT_WIDTH, EPOCH_SUBPLOT_ROW_HEIGHT_LARGE * n_crps_datasets),
            sharex=True,
        )
        if n_crps_datasets == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Val CRPS subplot
        if has_val_crps:
            ax1 = axes[plot_idx]
            
            # CRPS Mean on left y-axis
            color1 = 'tab:orange'
            ax1.set_ylabel('CRPS Mean', color=color1)
            ax1.tick_params(axis='y', labelcolor=color1)
            ax1.grid(alpha=0.1)
            
            crps_mask = ~np.isnan(task_crps_mean)
            crps_epochs = epochs[crps_mask]
            crps_values = task_crps_mean[crps_mask]
            marker_freq = _get_marker_frequency(len(crps_epochs))
            marker_indices = np.arange(0, len(crps_epochs), marker_freq)
            line1 = ax1.plot(crps_epochs, crps_values, label="Mean", color=color1, linewidth=2)
            ax1.plot(crps_epochs[marker_indices], crps_values[marker_indices], 'o', color=color1, markersize=4)
            
            # CRPS Std on right y-axis
            ax2 = ax1.twinx()
            color2 = 'tab:purple'
            ax2.set_ylabel('CRPS Std', color=color2)
            ax2.tick_params(axis='y', labelcolor=color2)
            
            crps_std_mask = ~np.isnan(task_crps_std)
            crps_std_epochs = epochs[crps_std_mask]
            crps_std_values = task_crps_std[crps_std_mask]
            marker_freq = _get_marker_frequency(len(crps_std_epochs))
            marker_indices = np.arange(0, len(crps_std_epochs), marker_freq)
            line2 = ax2.plot(crps_std_epochs, crps_std_values, label="Std", color=color2, linewidth=2)
            ax2.plot(crps_std_epochs[marker_indices], crps_std_values[marker_indices], 's', color=color2, markersize=4)
            
            # Legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper right', title='Val CRPS')
            
            plot_idx += 1
        
        # Train-val CRPS subplot
        if has_train_val_crps:
            ax1 = axes[plot_idx]
            
            # CRPS Mean on left y-axis
            color1 = 'tab:orange'
            ax1.set_ylabel('CRPS Mean', color=color1)
            ax1.tick_params(axis='y', labelcolor=color1)
            ax1.grid(alpha=0.1)
            
            train_val_crps_mask = ~np.isnan(train_val_task_crps_mean)
            train_val_crps_epochs = epochs[train_val_crps_mask]
            train_val_crps_values = train_val_task_crps_mean[train_val_crps_mask]
            marker_freq = _get_marker_frequency(len(train_val_crps_epochs))
            marker_indices = np.arange(0, len(train_val_crps_epochs), marker_freq)
            line1 = ax1.plot(train_val_crps_epochs, train_val_crps_values, label="Mean", color=color1, linewidth=2)
            ax1.plot(train_val_crps_epochs[marker_indices], train_val_crps_values[marker_indices], 'o', color=color1, markersize=4)
            
            # CRPS Std on right y-axis
            ax2 = ax1.twinx()
            color2 = 'tab:purple'
            ax2.set_ylabel('CRPS Std', color=color2)
            ax2.tick_params(axis='y', labelcolor=color2)
            
            train_val_crps_std_mask = ~np.isnan(train_val_task_crps_std)
            train_val_crps_std_epochs = epochs[train_val_crps_std_mask]
            train_val_crps_std_values = train_val_task_crps_std[train_val_crps_std_mask]
            marker_freq = _get_marker_frequency(len(train_val_crps_std_epochs))
            marker_indices = np.arange(0, len(train_val_crps_std_epochs), marker_freq)
            line2 = ax2.plot(train_val_crps_std_epochs, train_val_crps_std_values, label="Std", color=color2, linewidth=2)
            ax2.plot(train_val_crps_std_epochs[marker_indices], train_val_crps_std_values[marker_indices], 's', color=color2, markersize=4)
            
            # Legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper right', title='Train-val CRPS')
            
            plot_idx += 1
        
        axes[-1].set_xlabel('Epoch')
        # plt.tight_layout()
        f4 = out / f"crps.{fmt}"
        plt.savefig(f4, dpi=150)
        plt.close()


    # 5) Point forecast vs distributional quality (MAE vs CRPS) - subplots for each dataset with twin y-axes
    has_val_mae_for_comparison = task_mae.size and np.any(~np.isnan(task_mae))
    has_val_crps_for_comparison = task_crps_mean.size and np.any(~np.isnan(task_crps_mean))
    has_train_val_mae_for_comparison = train_val_task_mae.size and np.any(~np.isnan(train_val_task_mae))
    has_train_val_crps_for_comparison = train_val_task_crps_mean.size and np.any(~np.isnan(train_val_task_crps_mean))
    
    has_val_mae_crps = has_val_mae_for_comparison and has_val_crps_for_comparison
    has_train_val_mae_crps = has_train_val_mae_for_comparison and has_train_val_crps_for_comparison
    n_mae_crps_datasets = sum([has_val_mae_crps, has_train_val_mae_crps])
    
    if n_mae_crps_datasets > 0:
        fig, axes = plt.subplots(
            n_mae_crps_datasets,
            1,
            figsize=(EPOCH_SUBPLOT_WIDTH, EPOCH_SUBPLOT_ROW_HEIGHT_LARGE * n_mae_crps_datasets),
            sharex=True,
        )
        if n_mae_crps_datasets == 1:
            axes = [axes]
        
        plot_idx = 0
        
        # Val MAE vs CRPS subplot
        if has_val_mae_crps:
            ax1 = axes[plot_idx]
            
            # Plot MAE on left y-axis
            color1 = 'tab:blue'
            ax1.set_ylabel('MAE', color=color1)
            ax1.tick_params(axis='y', labelcolor=color1)
            ax1.grid(alpha=0.1)
            
            mae_mask = ~np.isnan(task_mae)
            mae_epochs = epochs[mae_mask]
            mae_values = task_mae[mae_mask]
            marker_freq = _get_marker_frequency(len(mae_epochs))
            marker_indices = np.arange(0, len(mae_epochs), marker_freq)
            line1 = ax1.plot(mae_epochs, mae_values, label="MAE", color=color1, linewidth=2)
            ax1.plot(mae_epochs[marker_indices], mae_values[marker_indices], 'o', color=color1, markersize=4)
            
            # Plot CRPS on right y-axis
            ax2 = ax1.twinx()
            color2 = 'tab:orange'
            ax2.set_ylabel('CRPS Mean', color=color2)
            ax2.tick_params(axis='y', labelcolor=color2)
            
            crps_mask = ~np.isnan(task_crps_mean)
            crps_epochs = epochs[crps_mask]
            crps_values = task_crps_mean[crps_mask]
            marker_freq = _get_marker_frequency(len(crps_epochs))
            marker_indices = np.arange(0, len(crps_epochs), marker_freq)
            line2 = ax2.plot(crps_epochs, crps_values, label="CRPS", color=color2, linewidth=2)
            ax2.plot(crps_epochs[marker_indices], crps_values[marker_indices], 's', color=color2, markersize=4)
            
            # Legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper right', title='Val')
            
            plot_idx += 1
        
        # Train-val MAE vs CRPS subplot
        if has_train_val_mae_crps:
            ax1 = axes[plot_idx]
            
            # Plot MAE on left y-axis
            color1 = 'tab:blue'
            ax1.set_ylabel('MAE', color=color1)
            ax1.tick_params(axis='y', labelcolor=color1)
            ax1.grid(alpha=0.1)
            
            train_val_mae_mask = ~np.isnan(train_val_task_mae)
            train_val_mae_epochs = epochs[train_val_mae_mask]
            train_val_mae_values = train_val_task_mae[train_val_mae_mask]
            marker_freq = _get_marker_frequency(len(train_val_mae_epochs))
            marker_indices = np.arange(0, len(train_val_mae_epochs), marker_freq)
            line1 = ax1.plot(train_val_mae_epochs, train_val_mae_values, label="MAE", color=color1, linewidth=2)
            ax1.plot(train_val_mae_epochs[marker_indices], train_val_mae_values[marker_indices], 'o', color=color1, markersize=4)
            
            # Plot CRPS on right y-axis
            ax2 = ax1.twinx()
            color2 = 'tab:orange'
            ax2.set_ylabel('CRPS Mean', color=color2)
            ax2.tick_params(axis='y', labelcolor=color2)
            
            train_val_crps_mask = ~np.isnan(train_val_task_crps_mean)
            train_val_crps_epochs = epochs[train_val_crps_mask]
            train_val_crps_values = train_val_task_crps_mean[train_val_crps_mask]
            marker_freq = _get_marker_frequency(len(train_val_crps_epochs))
            marker_indices = np.arange(0, len(train_val_crps_epochs), marker_freq)
            line2 = ax2.plot(train_val_crps_epochs, train_val_crps_values, label="CRPS", color=color2, linewidth=2)
            ax2.plot(train_val_crps_epochs[marker_indices], train_val_crps_values[marker_indices], 's', color=color2, markersize=4)
            
            # Legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper right', title='Train-val')
            
            plot_idx += 1
        
        axes[-1].set_xlabel('Epoch')
        # plt.tight_layout()
        f5 = out / f"mae_vs_crps.{fmt}"
        plt.savefig(f5, dpi=150)
        plt.close()


    # === Stock Price Forecasting Three-Tier Metric Plots ===

    # 5b) Direction Accuracy (Tier 1 daily + Tier 3 cumulative) — val & train-val
    has_val_dir_acc = task_direction_acc.size and np.any(~np.isnan(task_direction_acc))
    has_val_cum_dir_acc = task_cum_T_dir_acc.size and np.any(~np.isnan(task_cum_T_dir_acc))
    has_train_val_dir_acc = train_val_direction_acc.size and np.any(~np.isnan(train_val_direction_acc))
    has_train_val_cum_dir_acc = train_val_cum_T_dir_acc.size and np.any(~np.isnan(train_val_cum_T_dir_acc))

    if has_val_dir_acc or has_val_cum_dir_acc:
        plt.figure(figsize=EPOCH_FIGSIZE)

        if has_val_dir_acc:
            mask = ~np.isnan(task_direction_acc)
            ep = epochs[mask]
            vals = task_direction_acc[mask]
            marker_freq = _get_marker_frequency(len(ep))
            marker_indices = np.arange(0, len(ep), marker_freq)
            plt.plot(ep, vals, label="Val daily", color='tab:blue', linewidth=2)
            plt.plot(ep[marker_indices], vals[marker_indices], 'o', color='tab:blue', markersize=4)

        if has_val_cum_dir_acc:
            mask = ~np.isnan(task_cum_T_dir_acc)
            ep = epochs[mask]
            vals = task_cum_T_dir_acc[mask]
            marker_freq = _get_marker_frequency(len(ep))
            marker_indices = np.arange(0, len(ep), marker_freq)
            plt.plot(ep, vals, label="Val cumulative T", color='tab:cyan', linewidth=2, linestyle='--')
            plt.plot(ep[marker_indices], vals[marker_indices], 's', color='tab:cyan', markersize=4)

        if has_train_val_dir_acc:
            mask = ~np.isnan(train_val_direction_acc)
            ep = epochs[mask]
            vals = train_val_direction_acc[mask]
            marker_freq = _get_marker_frequency(len(ep))
            marker_indices = np.arange(0, len(ep), marker_freq)
            plt.plot(ep, vals, label="Train-val daily", color='tab:orange', linewidth=2)
            plt.plot(ep[marker_indices], vals[marker_indices], '^', color='tab:orange', markersize=4)

        if has_train_val_cum_dir_acc:
            mask = ~np.isnan(train_val_cum_T_dir_acc)
            ep = epochs[mask]
            vals = train_val_cum_T_dir_acc[mask]
            marker_freq = _get_marker_frequency(len(ep))
            marker_indices = np.arange(0, len(ep), marker_freq)
            plt.plot(ep, vals, label="Train-val cumulative T", color='tab:red', linewidth=2, linestyle='--')
            plt.plot(ep[marker_indices], vals[marker_indices], 'v', color='tab:red', markersize=4)

        plt.axhline(y=0.5, color='grey', linestyle=':', alpha=0.5, label='Random (0.5)')
        plt.xlabel("Epoch")
        plt.ylabel("Direction Accuracy")
        plt.ylim(0.3, 1.0)
        plt.legend(fontsize=8)
        plt.grid(alpha=0.1)
        f_dirac = out / f"direction_accuracy.{fmt}"
        plt.savefig(f_dirac, dpi=150)
        plt.close()

    # 5c) Volatility Clustering / Return Correlation (Tier 1) — val & train-val
    has_val_corr = task_return_corr.size and np.any(~np.isnan(task_return_corr))
    has_train_val_corr = train_val_return_corr.size and np.any(~np.isnan(train_val_return_corr))
    has_volclustering_real = task_volclustering_real.size and np.any(~np.isnan(task_volclustering_real))
    # Detect whether new-style keys were found (volclustering) vs legacy (Pearson r)
    is_legacy_corr = not (task_volclustering_real.size and np.any(~np.isnan(task_volclustering_real)))

    if has_val_corr or has_train_val_corr:
        plt.figure(figsize=EPOCH_FIGSIZE)

        if has_val_corr:
            mask = ~np.isnan(task_return_corr)
            ep = epochs[mask]
            vals = task_return_corr[mask]
            marker_freq = _get_marker_frequency(len(ep))
            marker_indices = np.arange(0, len(ep), marker_freq)
            plt.plot(ep, vals, label="Val", color='tab:blue', linewidth=2)
            plt.plot(ep[marker_indices], vals[marker_indices], 'o', color='tab:blue', markersize=4)

        if has_train_val_corr:
            mask = ~np.isnan(train_val_return_corr)
            ep = epochs[mask]
            vals = train_val_return_corr[mask]
            marker_freq = _get_marker_frequency(len(ep))
            marker_indices = np.arange(0, len(ep), marker_freq)
            plt.plot(ep, vals, label="Train-val", color='tab:orange', linewidth=2)
            plt.plot(ep[marker_indices], vals[marker_indices], 's', color='tab:orange', markersize=4)

        # Reference line for real-data volclustering (constant across epochs)
        if has_volclustering_real:
            real_val = task_volclustering_real[~np.isnan(task_volclustering_real)][0]
            plt.axhline(y=real_val, color='tab:green', linestyle='--', linewidth=1.5,
                       label=f"Real ACF(r²)={real_val:.3f}", alpha=0.8)

        plt.xlabel("Epoch")
        if is_legacy_corr:
            plt.ylabel("Return Correlation (legacy Pearson r)")
            plt.ylim(-0.2, 1.0)
            f_corr = out / f"return_correlation.{fmt}"
        else:
            plt.ylabel("Volatility Clustering (lag-1 ACF of r²)")
            plt.ylim(-0.1, 0.6)
            f_corr = out / f"volclustering.{fmt}"
        plt.legend()
        plt.grid(alpha=0.1)
        plt.savefig(f_corr, dpi=150)
        plt.close()

    # 5d) Price MAE vs Return MAE (Tier 2 vs Tier 1) — val only, twin y-axes
    has_val_price_mae = task_price_mae.size and np.any(~np.isnan(task_price_mae))
    has_val_return_mae = task_return_mae.size and np.any(~np.isnan(task_return_mae))

    if has_val_price_mae and has_val_return_mae:
        fig, ax1 = plt.subplots(figsize=EPOCH_FIGSIZE)

        # Price MAE (left y-axis)
        color1 = 'tab:green'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Price MAE ($)', color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(alpha=0.1)

        mask = ~np.isnan(task_price_mae)
        ep = epochs[mask]
        vals = task_price_mae[mask]
        marker_freq = _get_marker_frequency(len(ep))
        marker_indices = np.arange(0, len(ep), marker_freq)
        line1 = ax1.plot(ep, vals, label="Price MAE", color=color1, linewidth=2)
        ax1.plot(ep[marker_indices], vals[marker_indices], 'o', color=color1, markersize=4)

        # Return MAE (right y-axis)
        ax2 = ax1.twinx()
        color2 = 'tab:blue'
        ax2.set_ylabel('Return MAE', color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)

        mask = ~np.isnan(task_return_mae)
        ep = epochs[mask]
        vals = task_return_mae[mask]
        marker_freq = _get_marker_frequency(len(ep))
        marker_indices = np.arange(0, len(ep), marker_freq)
        line2 = ax2.plot(ep, vals, label="Return MAE", color=color2, linewidth=2)
        ax2.plot(ep[marker_indices], vals[marker_indices], 's', color=color2, markersize=4)

        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right', title='Val (Tier 2 vs Tier 1)')
        f_pr = out / f"price_vs_return_mae.{fmt}"
        plt.savefig(f_pr, dpi=150)
        plt.close()

    # 5e) Cumulative return metrics at final horizon (Tier 3) — MAE & CRPS, val only
    has_val_cum_mae = task_cum_T_mae.size and np.any(~np.isnan(task_cum_T_mae))
    has_val_cum_crps = task_cum_T_crps.size and np.any(~np.isnan(task_cum_T_crps))

    if has_val_cum_mae or has_val_cum_crps:
        fig, ax1 = plt.subplots(figsize=EPOCH_FIGSIZE)

        if has_val_cum_mae:
            color1 = 'tab:blue'
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Cumulative Return MAE', color=color1)
            ax1.tick_params(axis='y', labelcolor=color1)
            ax1.grid(alpha=0.1)

            mask = ~np.isnan(task_cum_T_mae)
            ep = epochs[mask]
            vals = task_cum_T_mae[mask]
            marker_freq = _get_marker_frequency(len(ep))
            marker_indices = np.arange(0, len(ep), marker_freq)
            line1 = ax1.plot(ep, vals, label="MAE", color=color1, linewidth=2)
            ax1.plot(ep[marker_indices], vals[marker_indices], 'o', color=color1, markersize=4)

            if has_val_cum_crps:
                ax2 = ax1.twinx()
                color2 = 'tab:orange'
                ax2.set_ylabel('Cumulative Return CRPS', color=color2)
                ax2.tick_params(axis='y', labelcolor=color2)

                mask = ~np.isnan(task_cum_T_crps)
                ep = epochs[mask]
                vals = task_cum_T_crps[mask]
                marker_freq = _get_marker_frequency(len(ep))
                marker_indices = np.arange(0, len(ep), marker_freq)
                line2 = ax2.plot(ep, vals, label="CRPS", color=color2, linewidth=2)
                ax2.plot(ep[marker_indices], vals[marker_indices], 's', color=color2, markersize=4)

                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax1.legend(lines, labels, loc='upper right', title='Val Tier 3')
            else:
                ax1.legend()
        elif has_val_cum_crps:
            mask = ~np.isnan(task_cum_T_crps)
            ep = epochs[mask]
            vals = task_cum_T_crps[mask]
            marker_freq = _get_marker_frequency(len(ep))
            marker_indices = np.arange(0, len(ep), marker_freq)
            ax1.plot(ep, vals, label="CRPS", color='tab:orange', linewidth=2)
            ax1.plot(ep[marker_indices], vals[marker_indices], 's', color='tab:orange', markersize=4)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Cumulative Return CRPS')
            ax1.grid(alpha=0.1)
            ax1.legend()

        f_cum = out / f"cumulative_return_metrics.{fmt}"
        plt.savefig(f_cum, dpi=150)
        plt.close()


    # === Wireless Resource Allocation Plots ===
    
    # 6) Sum Rate: Generated vs Real (twin y-axes for generated and real)
    has_wra_sum_gen = (wra_sum_rate_gen.size and np.any(~np.isnan(wra_sum_rate_gen))) or \
                      (train_val_wra_sum_rate_gen.size and np.any(~np.isnan(train_val_wra_sum_rate_gen)))
    has_wra_sum_real = (wra_sum_rate_real.size and np.any(~np.isnan(wra_sum_rate_real))) or \
                       (train_val_wra_sum_rate_real.size and np.any(~np.isnan(train_val_wra_sum_rate_real)))
    if has_wra_sum_gen and has_wra_sum_real:
        fig, ax1 = plt.subplots(figsize=EPOCH_FIGSIZE)
        lines_all = []
        
        # Plot Generated Sum Rate on left y-axis
        color1 = 'tab:blue'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Sum Rate Generated', color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(alpha=0.1)
        
        if wra_sum_rate_gen.size:
            mask = ~np.isnan(wra_sum_rate_gen)
            ep = epochs[mask]
            vals = wra_sum_rate_gen[mask]
            marker_freq = _get_marker_frequency(len(ep))
            marker_indices = np.arange(0, len(ep), marker_freq)
            line1 = ax1.plot(ep, vals, label="Val Generated", color=color1, linewidth=2)
            ax1.plot(ep[marker_indices], vals[marker_indices], 'o', color=color1, markersize=4)
            lines_all.extend(line1)
        
        if train_val_wra_sum_rate_gen.size:
            mask = ~np.isnan(train_val_wra_sum_rate_gen)
            ep = epochs[mask]
            vals = train_val_wra_sum_rate_gen[mask]
            marker_freq = _get_marker_frequency(len(ep))
            marker_indices = np.arange(0, len(ep), marker_freq)
            line1b = ax1.plot(ep, vals, label="Train-val Generated", 
                             linestyle="--", color=color1, alpha=0.7, linewidth=2)
            ax1.plot(ep[marker_indices], vals[marker_indices], '^', color=color1, markersize=4, alpha=0.7)
            lines_all.extend(line1b)
        
        # Plot Real Sum Rate on right y-axis
        ax2 = ax1.twinx()
        color2 = 'tab:green'
        ax2.set_ylabel('Sum Rate Real (Primal-Dual)', color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)
        
        if wra_sum_rate_real.size:
            mask = ~np.isnan(wra_sum_rate_real)
            ep = epochs[mask]
            vals = wra_sum_rate_real[mask]
            marker_freq = _get_marker_frequency(len(ep))
            marker_indices = np.arange(0, len(ep), marker_freq)
            line2 = ax2.plot(ep, vals, label="Val Real", color=color2, linewidth=2)
            ax2.plot(ep[marker_indices], vals[marker_indices], 's', color=color2, markersize=4)
            lines_all.extend(line2)
        
        if train_val_wra_sum_rate_real.size:
            mask = ~np.isnan(train_val_wra_sum_rate_real)
            ep = epochs[mask]
            vals = train_val_wra_sum_rate_real[mask]
            marker_freq = _get_marker_frequency(len(ep))
            marker_indices = np.arange(0, len(ep), marker_freq)
            line2b = ax2.plot(ep, vals, label="Train-val Real", 
                             linestyle="--", color=color2, alpha=0.7, linewidth=2)
            ax2.plot(ep[marker_indices], vals[marker_indices], 'D', color=color2, markersize=4, alpha=0.7)
            lines_all.extend(line2b)
        
        if lines_all:
            labels = [l.get_label() for l in lines_all]
            ax1.legend(lines_all, labels, loc='upper left')
        
        f6 = out / f"wra_sum_rate.{fmt}"
        plt.savefig(f6, dpi=150)
        plt.close()


    # 7) Fairness: Generated vs Real
    has_wra_fair_gen = (wra_fairness_gen.size and np.any(~np.isnan(wra_fairness_gen))) or \
                       (train_val_wra_fairness_gen.size and np.any(~np.isnan(train_val_wra_fairness_gen)))
    has_wra_fair_real = (wra_fairness_real.size and np.any(~np.isnan(wra_fairness_real))) or \
                        (train_val_wra_fairness_real.size and np.any(~np.isnan(train_val_wra_fairness_real)))
    if has_wra_fair_gen and has_wra_fair_real:
        fig, ax1 = plt.subplots(figsize=EPOCH_FIGSIZE)
        lines_all = []
        
        color1 = 'tab:purple'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Fairness Generated', color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(alpha=0.1)
        ax1.set_ylim([0, 1.05])  # Fairness is between 0 and 1
        
        if wra_fairness_gen.size:
            mask = ~np.isnan(wra_fairness_gen)
            ep = epochs[mask]
            vals = wra_fairness_gen[mask]
            marker_freq = _get_marker_frequency(len(ep))
            marker_indices = np.arange(0, len(ep), marker_freq)
            line1 = ax1.plot(ep, vals, label="Val Generated", color=color1, linewidth=2)
            ax1.plot(ep[marker_indices], vals[marker_indices], 'o', color=color1, markersize=4)
            lines_all.extend(line1)
        
        if train_val_wra_fairness_gen.size:
            mask = ~np.isnan(train_val_wra_fairness_gen)
            ep = epochs[mask]
            vals = train_val_wra_fairness_gen[mask]
            marker_freq = _get_marker_frequency(len(ep))
            marker_indices = np.arange(0, len(ep), marker_freq)
            line1b = ax1.plot(ep, vals, label="Train-val Generated", 
                             linestyle="--", color=color1, alpha=0.7, linewidth=2)
            ax1.plot(ep[marker_indices], vals[marker_indices], '^', color=color1, markersize=4, alpha=0.7)
            lines_all.extend(line1b)
        
        ax2 = ax1.twinx()
        color2 = 'tab:olive'
        ax2.set_ylabel('Fairness Real', color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.set_ylim([0, 1.05])
        
        if wra_fairness_real.size:
            mask = ~np.isnan(wra_fairness_real)
            ep = epochs[mask]
            vals = wra_fairness_real[mask]
            marker_freq = _get_marker_frequency(len(ep))
            marker_indices = np.arange(0, len(ep), marker_freq)
            line2 = ax2.plot(ep, vals, label="Val Real", color=color2, linewidth=2)
            ax2.plot(ep[marker_indices], vals[marker_indices], 's', color=color2, markersize=4)
            lines_all.extend(line2)
        
        if train_val_wra_fairness_real.size:
            mask = ~np.isnan(train_val_wra_fairness_real)
            ep = epochs[mask]
            vals = train_val_wra_fairness_real[mask]
            marker_freq = _get_marker_frequency(len(ep))
            marker_indices = np.arange(0, len(ep), marker_freq)
            line2b = ax2.plot(ep, vals, label="Train-val Real", 
                             linestyle="--", color=color2, alpha=0.7, linewidth=2)
            ax2.plot(ep[marker_indices], vals[marker_indices], 'D', color=color2, markersize=4, alpha=0.7)
            lines_all.extend(line2b)
        
        if lines_all:
            labels = [l.get_label() for l in lines_all]
            ax1.legend(lines_all, labels, loc='lower right')
        
        f7 = out / f"wra_fairness.{fmt}"
        plt.savefig(f7, dpi=150)
        plt.close()


    # 8) Violations: percentages (top) and mean violation gap (bottom)
    has_wra_violations = (
        (wra_power_violation.size and np.any(~np.isnan(wra_power_violation)))
        or (wra_rate_violation.size and np.any(~np.isnan(wra_rate_violation)))
        or (train_val_wra_power_violation.size and np.any(~np.isnan(train_val_wra_power_violation)))
        or (train_val_wra_rate_violation.size and np.any(~np.isnan(train_val_wra_rate_violation)))
        or (wra_rate_mean_violation_gap_pct_gen.size and np.any(~np.isnan(wra_rate_mean_violation_gap_pct_gen)))
        or (train_val_wra_rate_mean_violation_gap_pct_gen.size and np.any(~np.isnan(train_val_wra_rate_mean_violation_gap_pct_gen)))
        or (wra_rate_mean_violation_gap_pct_real.size and np.any(~np.isnan(wra_rate_mean_violation_gap_pct_real)))
        or (train_val_wra_rate_mean_violation_gap_pct_real.size and np.any(~np.isnan(train_val_wra_rate_mean_violation_gap_pct_real)))
    )
    if has_wra_violations:
        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=(EPOCH_SUBPLOT_WIDTH, EPOCH_SUBPLOT_ROW_HEIGHT_LARGE * 2),
            sharex=True,
        )

        # Top subplot: violation percentages.
        if wra_power_violation.size:
            mask = ~np.isnan(wra_power_violation)
            ep = epochs[mask]
            vals = wra_power_violation[mask]
            marker_freq = _get_marker_frequency(len(ep))
            marker_indices = np.arange(0, len(ep), marker_freq)
            ax1.plot(ep, vals, label="Val Power Violation", color='tab:red', linewidth=2)
            ax1.plot(ep[marker_indices], vals[marker_indices], 'o', color='tab:red', markersize=4)

        if train_val_wra_power_violation.size:
            mask = ~np.isnan(train_val_wra_power_violation)
            ep = epochs[mask]
            vals = train_val_wra_power_violation[mask]
            marker_freq = _get_marker_frequency(len(ep))
            marker_indices = np.arange(0, len(ep), marker_freq)
            ax1.plot(
                ep,
                vals,
                label="Train-val Power Violation",
                linestyle="--",
                color='tab:red',
                alpha=0.7,
                linewidth=2,
            )
            ax1.plot(ep[marker_indices], vals[marker_indices], '^', color='tab:red', markersize=4, alpha=0.7)

        if wra_rate_violation.size:
            mask = ~np.isnan(wra_rate_violation)
            ep = epochs[mask]
            vals = wra_rate_violation[mask]
            marker_freq = _get_marker_frequency(len(ep))
            marker_indices = np.arange(0, len(ep), marker_freq)
            ax1.plot(ep, vals, label="Val Rate Violation", color='tab:orange', linewidth=2)
            ax1.plot(ep[marker_indices], vals[marker_indices], 's', color='tab:orange', markersize=4)

        if train_val_wra_rate_violation.size:
            mask = ~np.isnan(train_val_wra_rate_violation)
            ep = epochs[mask]
            vals = train_val_wra_rate_violation[mask]
            marker_freq = _get_marker_frequency(len(ep))
            marker_indices = np.arange(0, len(ep), marker_freq)
            ax1.plot(
                ep,
                vals,
                label="Train-val Rate Violation",
                linestyle="--",
                color='tab:orange',
                alpha=0.7,
                linewidth=2,
            )
            ax1.plot(ep[marker_indices], vals[marker_indices], 'D', color='tab:orange', markersize=4, alpha=0.7)

        ax1.set_ylabel("Violation Percentage (%)")
        ax1.set_ylim([0, min(100, ax1.get_ylim()[1])])  # Cap at 100%.
        handles1, _ = ax1.get_legend_handles_labels()
        if handles1:
            ax1.legend(loc='upper right')
        ax1.grid(alpha=0.1)

        # Bottom subplot: mean rate-violation gaps.
        if wra_rate_mean_violation_gap_pct_gen.size:
            mask = ~np.isnan(wra_rate_mean_violation_gap_pct_gen)
            ep = epochs[mask]
            vals = wra_rate_mean_violation_gap_pct_gen[mask]
            marker_freq = _get_marker_frequency(len(ep))
            marker_indices = np.arange(0, len(ep), marker_freq)
            ax2.plot(ep, vals, label="Val Mean Gap (Generated)", color='tab:blue', linewidth=2)
            ax2.plot(ep[marker_indices], vals[marker_indices], 'o', color='tab:blue', markersize=4)

        if train_val_wra_rate_mean_violation_gap_pct_gen.size:
            mask = ~np.isnan(train_val_wra_rate_mean_violation_gap_pct_gen)
            ep = epochs[mask]
            vals = train_val_wra_rate_mean_violation_gap_pct_gen[mask]
            marker_freq = _get_marker_frequency(len(ep))
            marker_indices = np.arange(0, len(ep), marker_freq)
            ax2.plot(
                ep,
                vals,
                label="Train-val Mean Gap (Generated)",
                linestyle="--",
                color='tab:blue',
                alpha=0.7,
                linewidth=2,
            )
            ax2.plot(ep[marker_indices], vals[marker_indices], '^', color='tab:blue', markersize=4, alpha=0.7)

        if wra_rate_mean_violation_gap_pct_real.size:
            mask = ~np.isnan(wra_rate_mean_violation_gap_pct_real)
            ep = epochs[mask]
            vals = wra_rate_mean_violation_gap_pct_real[mask]
            marker_freq = _get_marker_frequency(len(ep))
            marker_indices = np.arange(0, len(ep), marker_freq)
            ax2.plot(ep, vals, label="Val Mean Gap (Real)", color='tab:green', linewidth=2)
            ax2.plot(ep[marker_indices], vals[marker_indices], 's', color='tab:green', markersize=4)

        if train_val_wra_rate_mean_violation_gap_pct_real.size:
            mask = ~np.isnan(train_val_wra_rate_mean_violation_gap_pct_real)
            ep = epochs[mask]
            vals = train_val_wra_rate_mean_violation_gap_pct_real[mask]
            marker_freq = _get_marker_frequency(len(ep))
            marker_indices = np.arange(0, len(ep), marker_freq)
            ax2.plot(
                ep,
                vals,
                label="Train-val Mean Gap (Real)",
                linestyle="--",
                color='tab:green',
                alpha=0.7,
                linewidth=2,
            )
            ax2.plot(ep[marker_indices], vals[marker_indices], 'D', color='tab:green', markersize=4, alpha=0.7)

        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Mean Violation Gap (%)")
        ax2.set_ylim(bottom=0.0)
        handles2, _ = ax2.get_legend_handles_labels()
        if handles2:
            ax2.legend(loc='upper right')
        ax2.grid(alpha=0.1)

        # plt.tight_layout()
        f8 = out / f"wra_violations.{fmt}"
        plt.savefig(f8, dpi=150)
        plt.close()


    # 9) Performance Gaps: Sum/Min/1st-pct/5th-pct rate gaps (all on same plot)
    has_wra_gaps = (wra_sum_rate_gap.size and np.any(~np.isnan(wra_sum_rate_gap))) or \
                   (wra_min_rate_gap.size and np.any(~np.isnan(wra_min_rate_gap))) or \
                   (wra_rate_1pct_gap.size and np.any(~np.isnan(wra_rate_1pct_gap))) or \
                   (wra_rate_5pct_gap.size and np.any(~np.isnan(wra_rate_5pct_gap))) or \
                   (train_val_wra_rate_1pct_gap.size and np.any(~np.isnan(train_val_wra_rate_1pct_gap))) or \
                   (train_val_wra_rate_5pct_gap.size and np.any(~np.isnan(train_val_wra_rate_5pct_gap)))
    if has_wra_gaps:
        plt.figure(figsize=EPOCH_FIGSIZE)
        
        if wra_sum_rate_gap.size:
            mask = ~np.isnan(wra_sum_rate_gap)
            ep = epochs[mask]
            vals = wra_sum_rate_gap[mask]
            marker_freq = _get_marker_frequency(len(ep))
            marker_indices = np.arange(0, len(ep), marker_freq)
            plt.plot(ep, vals, label="Sum Rate Gap", color='tab:cyan', linewidth=2)
            plt.plot(ep[marker_indices], vals[marker_indices], 'o', color='tab:cyan', markersize=4)
        
        if wra_min_rate_gap.size:
            mask = ~np.isnan(wra_min_rate_gap)
            ep = epochs[mask]
            vals = wra_min_rate_gap[mask]
            marker_freq = _get_marker_frequency(len(ep))
            marker_indices = np.arange(0, len(ep), marker_freq)
            plt.plot(ep, vals, label="Min Rate Gap", color='tab:pink', linewidth=2)
            plt.plot(ep[marker_indices], vals[marker_indices], 's', color='tab:pink', markersize=4)

        if wra_rate_1pct_gap.size:
            mask = ~np.isnan(wra_rate_1pct_gap)
            ep = epochs[mask]
            vals = wra_rate_1pct_gap[mask]
            marker_freq = _get_marker_frequency(len(ep))
            marker_indices = np.arange(0, len(ep), marker_freq)
            plt.plot(ep, vals, label="1st-pct Rate Gap", color='tab:green', linewidth=2)
            plt.plot(ep[marker_indices], vals[marker_indices], '^', color='tab:green', markersize=4)

        if wra_rate_5pct_gap.size:
            mask = ~np.isnan(wra_rate_5pct_gap)
            ep = epochs[mask]
            vals = wra_rate_5pct_gap[mask]
            marker_freq = _get_marker_frequency(len(ep))
            marker_indices = np.arange(0, len(ep), marker_freq)
            plt.plot(ep, vals, label="5th-pct Rate Gap", color='tab:purple', linewidth=2)
            plt.plot(ep[marker_indices], vals[marker_indices], 'D', color='tab:purple', markersize=4)

        if train_val_wra_rate_1pct_gap.size:
            mask = ~np.isnan(train_val_wra_rate_1pct_gap)
            ep = epochs[mask]
            vals = train_val_wra_rate_1pct_gap[mask]
            marker_freq = _get_marker_frequency(len(ep))
            marker_indices = np.arange(0, len(ep), marker_freq)
            plt.plot(
                ep,
                vals,
                label="Train-val 1st-pct Gap",
                linestyle="--",
                color='tab:green',
                alpha=0.75,
                linewidth=2,
            )
            plt.plot(ep[marker_indices], vals[marker_indices], 'v', color='tab:green', markersize=4, alpha=0.75)

        if train_val_wra_rate_5pct_gap.size:
            mask = ~np.isnan(train_val_wra_rate_5pct_gap)
            ep = epochs[mask]
            vals = train_val_wra_rate_5pct_gap[mask]
            marker_freq = _get_marker_frequency(len(ep))
            marker_indices = np.arange(0, len(ep), marker_freq)
            plt.plot(
                ep,
                vals,
                label="Train-val 5th-pct Gap",
                linestyle="--",
                color='tab:purple',
                alpha=0.75,
                linewidth=2,
            )
            plt.plot(ep[marker_indices], vals[marker_indices], 'P', color='tab:purple', markersize=4, alpha=0.75)
        
        plt.xlabel("Epoch")
        plt.ylabel("Performance Gap (%)")
        plt.axhline(y=0, color='black', linestyle=':', alpha=0.5)  # Reference line at 0%
        plt.legend(loc='upper right')
        plt.grid(alpha=0.1)
        
        f9 = out / f"wra_performance_gaps.{fmt}"
        plt.savefig(f9, dpi=150)
        plt.close()

    # Best model tracker: composite score + EMA-smoothed component metrics
    has_composite = best_model_composite_score.size and np.any(~np.isnan(best_model_composite_score))
    has_ema_metrics = bool(best_model_ema_metrics)
    
    if has_composite or has_ema_metrics:
        # Filter EMA metrics to those with valid data
        valid_ema_metrics = {}
        for metric_name, (ema_epochs, ema_vals) in best_model_ema_metrics.items():
            if ema_vals.size and np.any(~np.isnan(ema_vals)):
                valid_ema_metrics[metric_name] = (ema_epochs, ema_vals)
        
        n_subplots = int(has_composite) + len(valid_ema_metrics)
        
        if n_subplots > 0:
            fig, axes = plt.subplots(
                n_subplots,
                1,
                figsize=(EPOCH_SUBPLOT_WIDTH, EPOCH_SUBPLOT_ROW_HEIGHT * n_subplots),
                sharex=True,
            )
            if n_subplots == 1:
                axes = [axes]
            
            plot_idx = 0
            
            # Plot composite score
            if has_composite:
                ax = axes[plot_idx]
                mask = ~np.isnan(best_model_composite_score)
                ep = epochs[mask]
                vals = best_model_composite_score[mask]
                marker_freq = _get_marker_frequency(len(ep))
                marker_indices = np.arange(0, len(ep), marker_freq)
                
                ax.plot(ep, vals, color='tab:red', linewidth=2, label='Composite Score')
                ax.plot(ep[marker_indices], vals[marker_indices], 'o', color='tab:red', markersize=4)
                ax.set_ylabel("Composite Score")
                ax.set_title("Best Model Tracker (Lower is Better)")
                ax.grid(alpha=0.1)
                ax.legend(loc='upper right')
                plot_idx += 1
            
            # Plot individual EMA metrics
            colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:cyan', 'tab:pink']
            for color_idx, (metric_name, (ema_epochs, ema_vals)) in enumerate(sorted(valid_ema_metrics.items())):
                ax = axes[plot_idx]
                mask = ~np.isnan(ema_vals)
                ep = ema_epochs[mask]
                vals = ema_vals[mask]
                marker_freq = _get_marker_frequency(len(ep))
                marker_indices = np.arange(0, len(ep), marker_freq)
                
                color = colors[color_idx % len(colors)]
                # Shorten metric name for display (remove common prefixes)
                display_name = metric_name.replace('val_task_', '').replace('train-val_task_', '').replace('val_', '').replace('train-val_', '').replace('task_', '')
                
                ax.plot(ep, vals, color=color, linewidth=2, label=display_name)
                ax.plot(ep[marker_indices], vals[marker_indices], 'o', color=color, markersize=4)
                ax.set_ylabel(display_name)
                ax.grid(alpha=0.1)
                ax.legend(loc='upper right')
                plot_idx += 1
            
            axes[-1].set_xlabel('Epoch')
            # plt.tight_layout()
            
            f_best = out / f"best_model_composite_score.{fmt}"
            plt.savefig(f_best, dpi=150)
            plt.close()


def plot_selector_diagnostics_jsonl(
    epoch_jsonl_path: str | Path,
    out_dir: str | Path,
    overwrite: bool = True,
    save_pdf: bool = True,
) -> None:
    """Plot selector diagnostics (temperature/aux/entropy) from epoch summaries JSONL."""
    del overwrite  # Kept for API compatibility.
    p = Path(epoch_jsonl_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(p)
    if not rows:
        return

    # Support both:
    # 1) epoch_summaries.jsonl (train_selector_* keys)
    # 2) selector_diagnostic_summaries.jsonl (scope='global', plain keys)
    has_scope = any(isinstance(r, dict) and "scope" in r for r in rows)
    if has_scope:
        rows = [r for r in rows if isinstance(r, dict) and r.get("scope") == "global"]
        if not rows:
            return

    epochs = np.array([int(r.get("epoch", i + 1)) for i, r in enumerate(rows)], dtype=int)
    fmt = "pdf" if save_pdf else "png"
    def _extract_numeric_series(key: str, legacy_key: Optional[str] = None) -> np.ndarray:
        candidate_keys = [key]
        if legacy_key is not None:
            candidate_keys.append(legacy_key)
        return np.array(
            [
                (
                    float(next((r[k] for k in candidate_keys if k in r and isinstance(r[k], (float, int))), float("nan")))
                )
                for r in rows
            ],
            dtype=float,
        )

    def _plot_series(
        values: np.ndarray,
        *,
        filename: str,
        ylabel: str,
        title: str,
        color: str,
    ) -> None:
        mask = ~np.isnan(values)
        if not np.any(mask):
            return

        x = epochs[mask]
        y = values[mask]
        plt.figure(figsize=SELECTOR_SERIES_FIGSIZE)
        marker_freq = _get_marker_frequency(len(x))
        marker_indices = np.arange(0, len(x), marker_freq)
        plt.plot(x, y, color=color, alpha=0.2, linewidth=1, label="_nolegend_")
        y_smooth = _smooth(y)
        plt.plot(x, y_smooth, color=color, linewidth=2)
        plt.plot(x[marker_indices], y_smooth[marker_indices], "o", color=color, markersize=4)
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(alpha=0.1)
        plt.savefig(out / f"{filename}.{fmt}", dpi=150)
        plt.close()

    # 1) Temperature evolution.
    _plot_series(
        _extract_numeric_series("temperature", legacy_key="train_selector_temperature"),
        filename="selector_temperature",
        ylabel="Selector temperature",
        title="Selector Temperature",
        color="C0",
    )

    # 2) Auxiliary selector loss evolution.
    aux_values = _extract_numeric_series("selector_aux_total", legacy_key="train_selector_selector_aux_total")
    if np.isnan(aux_values).all():
        aux_values = _extract_numeric_series("aux_loss", legacy_key="train_selector_aux_loss")
    _plot_series(
        aux_values,
        filename="selector_aux_loss",
        ylabel="Auxiliary selector loss",
        title="Selector Auxiliary Loss",
        color="C1",
    )

    # 3) Entropy evolution (prefer normalized entropy, fallback from entropy_reg).
    entropy_values = _extract_numeric_series("entropy_mean_norm", legacy_key="train_selector_entropy_mean_norm")
    entropy_ylabel = "Entropy (normalized)"
    entropy_title = "Selector Entropy"
    if np.isnan(entropy_values).all():
        entropy_reg_values = _extract_numeric_series("entropy_reg", legacy_key="train_selector_entropy_reg")
        if not np.isnan(entropy_reg_values).all():
            entropy_values = 1.0 - entropy_reg_values
    _plot_series(
        entropy_values,
        filename="selector_entropy",
        ylabel=entropy_ylabel,
        title=entropy_title,
        color="C2",
    )


def _parse_selector_network_key(network_key: str) -> Tuple[Optional[str], str]:
    """
    Parse trainer graph key into (dataset_name, network_tag).

    Examples:
        "setA::network_7" -> ("setA", "network_7")
        "network_7" -> (None, "network_7")
        "graph_0" -> (None, "graph_0")
    """
    if "::" in network_key:
        dataset_name, network_tag = network_key.split("::", 1)
        return dataset_name, network_tag
    return None, network_key


def _sanitize_tag(value: Optional[str]) -> str:
    """Convert arbitrary text to filesystem-safe token."""
    if value is None or value == "":
        return "unknown"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def plot_selector_network_node_statistics(
    selector_network_node_stats: Dict[str, Dict[int, Dict[str, Any]]],
    epoch: int,
    out_dir: str | Path,
    save_pdf: bool = True,
) -> None:
    """
    Plot per-network selector node diagnostics with one row per selector level.

    Each level subplot overlays:
    - score mean ± std per node
    - soft-mask mean ± std per node (secondary y-axis)
    """
    if not selector_network_node_stats:
        return

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fmt = "pdf" if save_pdf else "png"

    for network_key, levels_dict in selector_network_node_stats.items():
        if not levels_dict:
            continue

        levels = sorted(levels_dict.keys())
        n_levels = len(levels)
        fig, axes = plt.subplots(
            n_levels,
            1,
            figsize=(SELECTOR_NETWORK_WIDTH, max(4.0, SELECTOR_NETWORK_ROW_HEIGHT * n_levels)),
            sharex=True,
        )
        if n_levels == 1:
            axes = [axes]

        dataset_name, network_tag = _parse_selector_network_key(network_key)

        for ax, level_idx in zip(axes, levels):
            stats = levels_dict[level_idx]
            score_mean = stats.get("score_mean", None)
            score_std = stats.get("score_std", None)
            soft_mean = stats.get("soft_mask_mean", None)
            soft_std = stats.get("soft_mask_std", None)

            # Determine node axis size from available tensors.
            num_nodes = None
            for arr in (score_mean, soft_mean, score_std, soft_std):
                if arr is not None:
                    num_nodes = int(np.asarray(arr).shape[0])
                    break
            if num_nodes is None or num_nodes <= 0:
                ax.text(
                    0.5,
                    0.5,
                    f"Level {level_idx}: no data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_ylabel(f"L{level_idx}")
                ax.grid(alpha=0.15)
                continue

            nodes = np.arange(num_nodes)
            legend_handles = []
            legend_labels = []

            if score_mean is not None:
                score_mean_np = np.asarray(score_mean, dtype=float)
                score_std_np = np.asarray(score_std, dtype=float) if score_std is not None else np.zeros_like(score_mean_np)
                # Inactive nodes are represented as NaNs in score stats; drop them from
                # score plotting. Use marker+errorbar style (no line connection) so
                # sparse validity patterns are clear per-node.
                score_valid = np.isfinite(score_mean_np)
                if np.any(score_valid):
                    nodes_valid = nodes[score_valid]
                    score_mean_valid = score_mean_np[score_valid]
                    score_std_valid = score_std_np[score_valid]
                    score_std_valid = np.where(np.isfinite(score_std_valid), score_std_valid, 0.0)
                    score_err = ax.errorbar(
                        nodes_valid,
                        score_mean_valid,
                        yerr=score_std_valid,
                        fmt="o",
                        color="C0",
                        ecolor="C0",
                        elinewidth=1.0,
                        capsize=2.5,
                        markersize=3.2,
                        alpha=0.95,
                        linestyle="None",
                        label="score mean±std",
                    )
                    legend_handles.append(score_err)
                    legend_labels.append("score mean±std")
                ax.set_ylabel(f"L{level_idx} score", color="C0")
                ax.tick_params(axis="y", colors="C0")

            ax_soft = ax.twinx()
            if soft_mean is not None:
                soft_mean_np = np.asarray(soft_mean, dtype=float)
                soft_std_np = np.asarray(soft_std, dtype=float) if soft_std is not None else np.zeros_like(soft_mean_np)
                soft_low = np.clip(soft_mean_np - soft_std_np, 0.0, 1.0)
                soft_high = np.clip(soft_mean_np + soft_std_np, 0.0, 1.0)
                soft_line, = ax_soft.plot(nodes, soft_mean_np, color="C1", linewidth=1.3, label="soft-mask mean")
                ax_soft.fill_between(nodes, soft_low, soft_high, color="C1", alpha=0.18, linewidth=0)
                legend_handles.append(soft_line)
                legend_labels.append("soft-mask mean±std")
            ax_soft.set_ylim(0.0, 1.0)
            ax_soft.set_ylabel(f"L{level_idx} soft-mask", color="C1")
            ax_soft.tick_params(axis="y", colors="C1")

            if legend_handles:
                ax.legend(legend_handles, legend_labels, loc="upper right", fontsize=8, framealpha=0.9)
            ax.grid(alpha=0.15)

        axes[-1].set_xlabel("Node index")

        title_parts = [f"Selector Node Diagnostics | Epoch {epoch + 1}"]
        if dataset_name is not None:
            title_parts.append(f"Dataset={dataset_name}")
        title_parts.append(f"Network={network_tag}")
        fig.suptitle(" | ".join(title_parts), fontsize=11)
        # fig.tight_layout(rect=(0, 0, 1, 0.95))

        dataset_tag = _sanitize_tag(dataset_name) if dataset_name is not None else None
        network_id_tag = _sanitize_tag(network_tag)
        if dataset_tag is not None:
            filename = f"selector_node_stats_d{dataset_tag}_n{network_id_tag}_epoch_{epoch + 1:04d}.{fmt}"
        else:
            filename = f"selector_node_stats_n{network_id_tag}_epoch_{epoch + 1:04d}.{fmt}"
        plt.savefig(out / filename, dpi=150)
        plt.close()


def plot_selector_network_hard_mask_statistics(
    selector_network_node_stats: Dict[str, Dict[int, Dict[str, Any]]],
    epoch: int,
    out_dir: str | Path,
    save_pdf: bool = True,
) -> None:
    """
    Plot per-network hard-mask node diagnostics with one row per selector level.

    Each level subplot shows hard-mask mean ± std over node index.
    """
    if not selector_network_node_stats:
        return

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fmt = "pdf" if save_pdf else "png"

    for network_key, levels_dict in selector_network_node_stats.items():
        if not levels_dict:
            continue

        levels = sorted(levels_dict.keys())
        n_levels = len(levels)
        fig, axes = plt.subplots(
            n_levels,
            1,
            figsize=(SELECTOR_NETWORK_WIDTH, max(3.8, SELECTOR_NETWORK_ROW_HEIGHT * n_levels)),
            sharex=True,
        )
        if n_levels == 1:
            axes = [axes]

        dataset_name, network_tag = _parse_selector_network_key(network_key)

        for ax, level_idx in zip(axes, levels):
            stats = levels_dict[level_idx]
            hard_mean = stats.get("hard_mask_mean", None)
            hard_std = stats.get("hard_mask_std", None)

            if hard_mean is None:
                ax.text(
                    0.5,
                    0.5,
                    f"Level {level_idx}: no hard-mask data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_ylabel(f"L{level_idx}")
                ax.set_ylim(0.0, 1.0)
                ax.grid(alpha=0.15)
                continue

            hard_mean_np = np.asarray(hard_mean, dtype=float)
            hard_std_np = np.asarray(hard_std, dtype=float) if hard_std is not None else np.zeros_like(hard_mean_np)
            nodes = np.arange(hard_mean_np.shape[0])
            hard_low = np.clip(hard_mean_np - hard_std_np, 0.0, 1.0)
            hard_high = np.clip(hard_mean_np + hard_std_np, 0.0, 1.0)

            ax.plot(nodes, hard_mean_np, color="C3", linewidth=1.4, label="hard-mask mean")
            ax.fill_between(nodes, hard_low, hard_high, color="C3", alpha=0.18, linewidth=0)
            ax.set_ylabel(f"L{level_idx}")
            ax.set_ylim(0.0, 1.0)
            ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
            ax.grid(alpha=0.15)

        axes[-1].set_xlabel("Node index")
        title_parts = [f"Hard-Mask Node Diagnostics | Epoch {epoch + 1}"]
        if dataset_name is not None:
            title_parts.append(f"Dataset={dataset_name}")
        title_parts.append(f"Network={network_tag}")
        fig.suptitle(" | ".join(title_parts), fontsize=11)
        # fig.tight_layout(rect=(0, 0, 1, 0.95))

        dataset_tag = _sanitize_tag(dataset_name) if dataset_name is not None else None
        network_id_tag = _sanitize_tag(network_tag)
        if dataset_tag is not None:
            filename = f"selector_hard_mask_d{dataset_tag}_n{network_id_tag}_epoch_{epoch + 1:04d}.{fmt}"
        else:
            filename = f"selector_hard_mask_n{network_id_tag}_epoch_{epoch + 1:04d}.{fmt}"
        plt.savefig(out / filename, dpi=150)
        plt.close()


def plot_selector_network_selection_frequency_evolution(
    diagnostic_jsonl_path: str | Path,
    out_dir: str | Path,
    save_pdf: bool = True,
    num_nodes_to_track: int = 10,
    random_seed: int = 0,
    max_network_plots: int = -1,
) -> None:
    """
    Plot per-network, per-level selection-frequency evolution across epochs.

    Reads selector_diagnostic_summaries.jsonl network rows and renders one figure per network
    with one subplot per selector level.

    Preferred mode (new summaries):
        Uses tracked per-node EMA fields and plots trajectories for tracked nodes.
        - selection_freq_tracked_node_indices_by_level
        - selection_freq_tracked_node_ma_by_level

    Backward compatibility:
        If tracked per-node EMA is unavailable, falls back to the old full per-node field:
        - selection_freq_ma_per_node_by_level

    Legacy fallback:
        If per-node EMA is unavailable, plots level-wise mean ± std curves from:
        - selection_freq_mean_by_level
        - selection_freq_std_by_level
    """
    p = Path(diagnostic_jsonl_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(p)
    if not rows:
        return

    network_rows = [r for r in rows if isinstance(r, dict) and r.get("scope") == "network"]
    if not network_rows:
        return

    grouped: Dict[str, List[dict]] = {}
    for row in network_rows:
        key = str(row.get("network_key", row.get("network_id", "unknown")))
        grouped.setdefault(key, []).append(row)

    rng = np.random.default_rng(int(random_seed))
    fmt = "pdf" if save_pdf else "png"
    network_keys = sorted(grouped.keys())
    if int(max_network_plots) == 0:
        return
    if int(max_network_plots) > 0:
        network_keys = network_keys[: int(max_network_plots)]

    for network_key in network_keys:
        entries = grouped[network_key]
        entries = sorted(entries, key=lambda r: int(r.get("epoch", 0)))

        level_keys_ma = set()
        level_keys_legacy_ma = set()
        for row in entries:
            level_keys_ma.update(
                str(k) for k in row.get("selection_freq_tracked_node_ma_by_level", {}).keys()
            )
            level_keys_legacy_ma.update(
                str(k) for k in row.get("selection_freq_ma_per_node_by_level", {}).keys()
            )

        use_per_node_ema = len(level_keys_ma) > 0 or len(level_keys_legacy_ma) > 0
        if use_per_node_ema:
            levels = sorted(int(k) for k in (level_keys_ma | level_keys_legacy_ma))
        else:
            level_keys = set()
            for row in entries:
                level_keys.update(str(k) for k in row.get("selection_freq_mean_by_level", {}).keys())
            if not level_keys:
                continue
            levels = sorted(int(k) for k in level_keys)

        n_levels = len(levels)
        fig, axes = plt.subplots(
            n_levels,
            1,
            figsize=(SELECTOR_FREQ_WIDTH, max(3.6, SELECTOR_FREQ_ROW_HEIGHT * n_levels)),
            sharex=True,
        )
        if n_levels == 1:
            axes = [axes]

        for ax, level_idx in zip(axes, levels):
            level_key = str(level_idx)
            if use_per_node_ema:
                tracked_indices = None
                for row in entries:
                    idx_map = row.get("selection_freq_tracked_node_indices_by_level", {})
                    idx_values = idx_map.get(level_key, None)
                    if isinstance(idx_values, list) and len(idx_values) > 0:
                        tracked_indices = [int(i) for i in idx_values]
                        break

                if tracked_indices is None:
                    for row in entries:
                        tracked_ma_map = row.get("selection_freq_tracked_node_ma_by_level", {})
                        tracked_ma_values = tracked_ma_map.get(level_key, None)
                        if isinstance(tracked_ma_values, list) and len(tracked_ma_values) > 0:
                            tracked_indices = list(range(len(tracked_ma_values)))
                            break

                if tracked_indices is None:
                    for row in entries:
                        ma_map = row.get("selection_freq_ma_per_node_by_level", {})
                        ma_values = ma_map.get(level_key, None)
                        if isinstance(ma_values, list) and len(ma_values) > 0:
                            num_nodes = len(ma_values)
                            k = min(max(int(num_nodes_to_track), 1), num_nodes)
                            if k == num_nodes:
                                tracked_indices = list(range(num_nodes))
                            else:
                                tracked_indices = sorted(rng.choice(num_nodes, size=k, replace=False).tolist())
                            break

                if tracked_indices is None or len(tracked_indices) == 0:
                    ax.text(0.5, 0.5, f"Level {level_idx}: no per-node EMA data", ha="center", va="center", transform=ax.transAxes)
                    ax.set_ylabel(f"L{level_idx}")
                    ax.set_ylim(0.0, 1.0)
                    ax.grid(alpha=0.15)
                    continue

                epochs_for_level: List[int] = []
                per_node_series: Dict[int, List[float]] = {idx: [] for idx in tracked_indices}
                for row in entries:
                    epochs_for_level.append(int(row.get("epoch", 0)) + 1)

                    tracked_ma_map = row.get("selection_freq_tracked_node_ma_by_level", {})
                    tracked_ma_values = tracked_ma_map.get(level_key, None)
                    tracked_idx_map = row.get("selection_freq_tracked_node_indices_by_level", {})
                    tracked_idx_values = tracked_idx_map.get(level_key, None)

                    if (
                        isinstance(tracked_ma_values, list)
                        and isinstance(tracked_idx_values, list)
                        and len(tracked_ma_values) == len(tracked_idx_values)
                    ):
                        row_lookup = {
                            int(idx): float(val)
                            for idx, val in zip(tracked_idx_values, tracked_ma_values)
                        }
                        for idx in tracked_indices:
                            per_node_series[idx].append(float(row_lookup.get(idx, float("nan"))))
                        continue

                    if isinstance(tracked_ma_values, list) and len(tracked_ma_values) == len(tracked_indices):
                        for idx, val in zip(tracked_indices, tracked_ma_values):
                            per_node_series[idx].append(float(val))
                        continue

                    ma_map = row.get("selection_freq_ma_per_node_by_level", {})
                    ma_values = ma_map.get(level_key, None)
                    if isinstance(ma_values, list):
                        for idx in tracked_indices:
                            if 0 <= idx < len(ma_values):
                                per_node_series[idx].append(float(ma_values[idx]))
                            else:
                                per_node_series[idx].append(float("nan"))
                    else:
                        for idx in tracked_indices:
                            per_node_series[idx].append(float("nan"))

                if len(epochs_for_level) == 0:
                    ax.text(0.5, 0.5, f"Level {level_idx}: no per-node EMA data", ha="center", va="center", transform=ax.transAxes)
                    ax.set_ylabel(f"L{level_idx}")
                    ax.set_ylim(0.0, 1.0)
                    ax.grid(alpha=0.15)
                    continue

                x_np = np.asarray(epochs_for_level, dtype=int)
                colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(len(tracked_indices), 1)))
                plotted_values = []
                for color, node_idx in zip(colors, tracked_indices):
                    y_np = np.asarray(per_node_series[node_idx], dtype=float)
                    valid = np.isfinite(y_np)
                    if not np.any(valid):
                        continue
                    plotted_values.append(y_np[valid])
                    ax.plot(
                        x_np[valid],
                        y_np[valid],
                        marker="o",
                        markersize=2.8,
                        linewidth=1.2,
                        color=color,
                        label=f"node {node_idx}",
                    )

                ax.set_ylabel(f"L{level_idx}")
                if plotted_values:
                    y_all = np.concatenate(plotted_values, axis=0)
                    lo, hi = _compute_zoomed_probability_ylim(y_all)
                    ax.set_ylim(lo, hi)
                else:
                    ax.set_ylim(0.0, 1.0)
                ax.grid(alpha=0.15)
                ax.legend(loc="upper right", fontsize=7, framealpha=0.9, ncol=2)
            else:
                x_epochs = []
                y_mean = []
                y_std = []
                for row in entries:
                    mean_dict = row.get("selection_freq_mean_by_level", {})
                    if level_key not in mean_dict:
                        continue
                    x_epochs.append(int(row.get("epoch", 0)) + 1)
                    y_mean.append(float(mean_dict[level_key]))
                    std_dict = row.get("selection_freq_std_by_level", {})
                    y_std.append(float(std_dict.get(level_key, 0.0)))

                if len(x_epochs) == 0:
                    ax.text(0.5, 0.5, f"Level {level_idx}: no data", ha="center", va="center", transform=ax.transAxes)
                    ax.set_ylabel(f"L{level_idx}")
                    ax.set_ylim(0.0, 1.0)
                    ax.grid(alpha=0.15)
                    continue

                x_np = np.asarray(x_epochs, dtype=int)
                mean_np = np.asarray(y_mean, dtype=float)
                std_np = np.asarray(y_std, dtype=float)
                lo = np.clip(mean_np - std_np, 0.0, 1.0)
                hi = np.clip(mean_np + std_np, 0.0, 1.0)

                ax.plot(x_np, mean_np, color="C4", linewidth=1.7, label="mean selection frequency")
                ax.fill_between(x_np, lo, hi, color="C4", alpha=0.2, linewidth=0)
                ax.set_ylabel(f"L{level_idx}")
                y_for_limits = np.concatenate([lo[np.isfinite(lo)], hi[np.isfinite(hi)]], axis=0)
                if y_for_limits.size > 0:
                    y_lo, y_hi = _compute_zoomed_probability_ylim(y_for_limits)
                    ax.set_ylim(y_lo, y_hi)
                else:
                    ax.set_ylim(0.0, 1.0)
                ax.grid(alpha=0.15)
                ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

        axes[-1].set_xlabel("Epoch")
        dataset_name, network_tag = _parse_selector_network_key(network_key)
        title_parts = ["Selection Frequency Evolution"]
        if use_per_node_ema:
            title_parts.append(f"EMA nodes={num_nodes_to_track}")
        if dataset_name is not None:
            title_parts.append(f"Dataset={dataset_name}")
        title_parts.append(f"Network={network_tag}")
        fig.suptitle(" | ".join(title_parts), fontsize=11)
        # fig.tight_layout(rect=(0, 0, 1, 0.95))

        dataset_tag = _sanitize_tag(dataset_name) if dataset_name is not None else None
        network_id_tag = _sanitize_tag(network_tag)
        if dataset_tag is not None:
            filename = f"selector_freq_evolution_d{dataset_tag}_n{network_id_tag}.{fmt}"
        else:
            filename = f"selector_freq_evolution_n{network_id_tag}.{fmt}"
        plt.savefig(out / filename, dpi=150)
        plt.close()


def aggregate_selector_sampling_diagnostics(
    diags_list: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Pool a list of per-batch selector-sampling diagnostics into one dict.

    Selection frequencies are ratios of sums, so pooling across batches is just
    summing numerators and denominators: per (level, probe-timestep, node) we add
    the ``selected_count`` and ``available_count``, and per probe-timestep we add
    the ``probe_graph_count``.  Frequencies are then recomputed from the pooled
    counts:

        selection_freq            = selected / graph_count
        selection_given_available = selected / available   (NaN where available==0)

    Rows are aligned by **probe-timestep value** (not list position) using the
    first non-empty diagnostics dict as the reference grid; timesteps absent from
    the reference are ignored.  Both the top-level (all-networks) bucket and each
    ``by_network`` key are pooled independently, preserving the same schema as a
    single batch's diagnostics.  Returns ``None`` if nothing poolable is found.
    """
    diags = [
        d
        for d in (diags_list or [])
        if isinstance(d, dict) and d.get("probe_timesteps")
    ]
    if not diags:
        return None

    ref_timesteps = [int(t) for t in diags[0]["probe_timesteps"]]
    num_probe = len(ref_timesteps)
    pos_by_t = {t: i for i, t in enumerate(ref_timesteps)}

    def _new_accum() -> Dict[str, Any]:
        return {
            "selected": {},  # level_key -> (num_probe, N)
            "available": {},  # level_key -> (num_probe, N)
            "graph_count": np.zeros(num_probe, dtype=np.float64),
        }

    def _row_map(diag: Dict[str, Any]) -> List[Tuple[int, int]]:
        d_ts = [int(t) for t in diag.get("probe_timesteps", [])]
        return [(j, pos_by_t[t]) for j, t in enumerate(d_ts) if t in pos_by_t]

    def _accumulate(accum: Dict[str, Any], diag: Dict[str, Any]) -> None:
        rmap = _row_map(diag)
        gc = np.asarray(diag.get("probe_graph_count", []), dtype=float)
        for j, i in rmap:
            if j < gc.size:
                accum["graph_count"][i] += float(gc[j])
        for metric_key, store_key in (
            ("selected_count_by_level", "selected"),
            ("available_count_by_level", "available"),
        ):
            by_level = diag.get(metric_key, {})
            if not isinstance(by_level, dict):
                continue
            store = accum[store_key]
            for level_key, mat_raw in by_level.items():
                mat = np.asarray(mat_raw, dtype=float)
                if mat.ndim != 2 or mat.shape[1] <= 0:
                    continue
                n_nodes = int(mat.shape[1])
                if level_key not in store:
                    store[level_key] = np.zeros((num_probe, n_nodes), dtype=np.float64)
                elif store[level_key].shape[1] != n_nodes:
                    # Node-count mismatch across batches — skip this contribution.
                    continue
                for j, i in rmap:
                    if j < mat.shape[0]:
                        store[level_key][i] += mat[j]

    def _finalize(accum: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "probe_timesteps": list(ref_timesteps),
            "probe_graph_count": accum["graph_count"].tolist(),
            "selected_count_by_level": {},
            "available_count_by_level": {},
            "selection_freq_by_level": {},
            "selection_given_available_by_level": {},
        }
        graphs = np.clip(accum["graph_count"], 1.0, None)
        level_keys = sorted(
            set(accum["selected"].keys()) | set(accum["available"].keys()),
            key=lambda k: (int(k) if str(k).lstrip("-").isdigit() else 0, str(k)),
        )
        for level_key in level_keys:
            selected = accum["selected"].get(level_key)
            available = accum["available"].get(level_key)
            if selected is None and available is None:
                continue
            if selected is None:
                selected = np.zeros_like(available)
            if available is None:
                available = np.zeros_like(selected)
            freq = selected / graphs[:, None]
            cond = np.full_like(selected, np.nan)
            valid = available > 0
            cond[valid] = selected[valid] / available[valid]
            out["selected_count_by_level"][level_key] = selected.tolist()
            out["available_count_by_level"][level_key] = available.tolist()
            out["selection_freq_by_level"][level_key] = freq.tolist()
            out["selection_given_available_by_level"][level_key] = cond.tolist()
        if extra:
            out.update(extra)
        return out

    top_accum = _new_accum()
    network_accums: Dict[str, Dict[str, Any]] = {}
    network_meta: Dict[str, Dict[str, Any]] = {}
    for diag in diags:
        _accumulate(top_accum, diag)
        by_network = diag.get("by_network", {})
        if not isinstance(by_network, dict):
            continue
        for graph_key, network_diag in by_network.items():
            if not isinstance(network_diag, dict):
                continue
            if graph_key not in network_accums:
                network_accums[graph_key] = _new_accum()
                network_meta[graph_key] = {
                    "network_key": graph_key,
                    "dataset_name": network_diag.get("dataset_name"),
                    "network_id": network_diag.get("network_id"),
                }
            _accumulate(network_accums[graph_key], network_diag)

    result = _finalize(top_accum)
    result["by_network"] = {
        graph_key: _finalize(network_accums[graph_key], extra=network_meta[graph_key])
        for graph_key in sorted(network_accums.keys())
    }
    return result


def plot_selector_sampling_phase_heatmaps(
    selector_sampling_diagnostics: Optional[Dict[str, Any]],
    out_dir: str | Path,
    epoch: int,
    split: str = "eval",
    dataset_name: Optional[str] = None,
    network_id: Optional[str] = None,
    save_pdf: bool = True,
    batch_idx: Optional[int] = None,
    aggregate: bool = False,
) -> None:
    """
    Plot per-level heatmaps of selector behavior across DDPM sampling phases.

    ``aggregate=True`` marks the figure (filename ``_aggregate`` token + title
    note) as a pooled-over-all-batches heatmap and takes precedence over
    ``batch_idx`` for the filename token.

    The preferred signal is P(selected | active), falling back to raw per-graph
    selection frequency when availability counts are absent.

    For readability on large graphs, plots at most `SELECTOR_SAMPLING_MAX_NODES`
    nodes using one shared subset across all levels:
      - top 10 by expected reward
      - bottom 10 by expected reward
    where expected reward is level-selection probability weighted by cumulative
    downsampling factor of that level.
    """
    if not isinstance(selector_sampling_diagnostics, dict):
        return

    probe_timesteps_raw = selector_sampling_diagnostics.get("probe_timesteps", [])
    probe_timesteps: List[int] = []
    for t in probe_timesteps_raw:
        try:
            probe_timesteps.append(int(t))
        except (TypeError, ValueError):
            continue
    if len(probe_timesteps) == 0:
        return

    cond_by_level = selector_sampling_diagnostics.get("selection_given_available_by_level", {})
    freq_by_level = selector_sampling_diagnostics.get("selection_freq_by_level", {})
    matrix_by_level = cond_by_level if isinstance(cond_by_level, dict) and len(cond_by_level) > 0 else freq_by_level
    if not isinstance(matrix_by_level, dict) or len(matrix_by_level) == 0:
        return

    level_keys: List[Tuple[int, str]] = []
    for level_key in matrix_by_level.keys():
        try:
            level_keys.append((int(level_key), str(level_key)))
        except (TypeError, ValueError):
            continue
    if len(level_keys) == 0:
        return
    level_keys = sorted(level_keys, key=lambda x: x[0])

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fmt = "pdf" if save_pdf else "png"

    fig, axes = plt.subplots(
        len(level_keys),
        1,
        figsize=(SELECTOR_SAMPLING_WIDTH, max(4.0, SELECTOR_SAMPLING_ROW_HEIGHT * len(level_keys))),
        sharex=True,
    )
    if len(level_keys) == 1:
        axes = [axes]

    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#d9d9d9", alpha=0.7)
    metric_name = "P(selected | active)" if matrix_by_level is cond_by_level else "Selection frequency"
    shared_node_indices = _select_sampling_heatmap_nodes_from_reward(
        selector_sampling_diagnostics=selector_sampling_diagnostics,
        level_keys=level_keys,
        matrix_by_level=matrix_by_level,
        max_nodes=SELECTOR_SAMPLING_MAX_NODES,
    )

    for ax, (level_idx, level_key) in zip(axes, level_keys):
        raw = matrix_by_level.get(level_key, None)
        mat = np.asarray(raw, dtype=float) if raw is not None else np.array([])
        if mat.ndim != 2 or mat.shape[0] <= 0 or mat.shape[1] <= 0:
            ax.text(
                0.5,
                0.5,
                f"Level {level_idx}: no probe data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_ylabel(f"L{level_idx}\nNode")
            ax.grid(alpha=0.1)
            continue

        if mat.shape[0] != len(probe_timesteps):
            # Prefer consistent x-axis length; truncate to the shared prefix when needed.
            steps = min(mat.shape[0], len(probe_timesteps))
            mat = mat[:steps, :]
            timesteps = probe_timesteps[:steps]
        else:
            timesteps = probe_timesteps

        node_indices = shared_node_indices
        if node_indices.size > 0:
            mat = mat[:, node_indices]

        data = np.ma.masked_invalid(mat.T)  # (nodes, probe_steps)
        im = ax.imshow(
            data,
            aspect="auto",
            interpolation="nearest",
            origin="lower",
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
        )
        cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
        cbar.ax.set_ylabel(metric_name, rotation=90)

        ax.set_ylabel(f"L{level_idx}\nNode")
        if node_indices.size > 0:
            yticks = np.arange(int(node_indices.size))
            ylabels = [str(int(i)) for i in node_indices.tolist()]
            ax.set_yticks(yticks)
            ax.set_yticklabels(ylabels, fontsize=7)
        ax.grid(False)

        if len(timesteps) <= 12:
            xticks = np.arange(len(timesteps))
            xlabels = [str(int(t)) for t in timesteps]
        else:
            step = max(1, len(timesteps) // 10)
            xticks = np.arange(0, len(timesteps), step)
            if xticks[-1] != len(timesteps) - 1:
                xticks = np.append(xticks, len(timesteps) - 1)
            xlabels = [str(int(timesteps[i])) for i in xticks]
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, rotation=0)

    axes[-1].set_xlabel("DDPM sampling timestep t (high noise -> low noise)")
    title_parts = [f"Selector Sampling Phase Heatmap | Epoch {epoch + 1} | Split={split}"]
    if dataset_name is not None:
        title_parts.append(f"Dataset={dataset_name}")
    if network_id is not None:
        title_parts.append(f"Network={network_id}")
    if aggregate:
        title_parts.append("AGGREGATE (all batches)")
    fig.suptitle(" | ".join(title_parts), fontsize=11)

    split_tag = _sanitize_tag(split)
    dataset_tag = _sanitize_tag(dataset_name) if dataset_name is not None else None
    network_tag = _sanitize_tag(network_id) if network_id is not None else "unknown"
    # Optional filename token. Without it, multiple eval batches sharing the same
    # (split, dataset, network, epoch) — e.g. SP500 where every window has
    # network_id=0 — overwrite a single file. ``aggregate`` (pooled over all
    # batches) takes precedence over the per-batch index.
    if aggregate:
        batch_tag = "_aggregate"
    elif batch_idx is not None:
        batch_tag = f"_b{int(batch_idx):03d}"
    else:
        batch_tag = ""
    if dataset_tag is not None:
        filename = (
            f"selector_sampling_heatmap_{split_tag}_d{dataset_tag}_n{network_tag}"
            f"{batch_tag}_epoch_{epoch + 1:04d}.{fmt}"
        )
    else:
        filename = (
            f"selector_sampling_heatmap_{split_tag}_n{network_tag}"
            f"{batch_tag}_epoch_{epoch + 1:04d}.{fmt}"
        )
    plt.savefig(out / filename, dpi=150)
    plt.close()
