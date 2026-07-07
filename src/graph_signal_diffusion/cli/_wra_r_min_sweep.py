"""Helpers for r_min sweep evaluation in the compare_baselines pipeline.

Creates temporary symlinked datasets with modified r_min values, extracts
dataset_info for the evaluator, generates summary tables and sweep plots.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import shutil
import os.path as osp
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Guard checks
# ---------------------------------------------------------------------------


def check_npz_no_r_min_per_receiver(reference_raw_dir: str) -> None:
    """Raise if the reference NPZ contains per-network r_min_per_receiver.

    When r_min_per_receiver is baked into the NPZ, it takes priority over
    network_info.json and our r_min override would be silently ignored.
    """
    npz_path = osp.join(reference_raw_dir, "collected_samples.npz")
    if not osp.exists(npz_path):
        raise FileNotFoundError(f"Reference NPZ not found: {npz_path}")

    with np.load(npz_path, allow_pickle=True) as npz:
        for key in npz.files:
            if not key.startswith("network_"):
                continue
            # Skip legacy/flat keys like network_0_H_instantaneous that
            # are not dict-valued per-network entries.
            try:
                net_data = npz[key].item()
            except (ValueError, AttributeError):
                continue
            if not isinstance(net_data, dict):
                continue
            if "r_min_per_receiver" in net_data:
                raise ValueError(
                    f"Reference NPZ contains r_min_per_receiver in '{key}'. "
                    "The sweep cannot override r_min via network_info.json "
                    "because NPZ-level r_min_per_receiver takes priority in "
                    "WRADataset.process(). Remove r_min_per_receiver from "
                    "the NPZ or use a dataset without it."
                )


def check_model_cond_channels(
    cfg_model_cond_channels: int,
    checkpoint_path: Optional[str] = None,
) -> None:
    """Raise if model_cond_channels < 3 or cfg/checkpoint disagree."""
    if cfg_model_cond_channels < 3:
        raise ValueError(
            f"r_min sweep requires model_cond_channels >= 3, "
            f"but cfg.dataset.model_cond_channels={cfg_model_cond_channels}."
        )

    if checkpoint_path is not None:
        from graph_signal_diffusion.cli._wra_transferability import (
            load_checkpoint_hydra_config,
        )

        ckpt_cfg = load_checkpoint_hydra_config(Path(checkpoint_path))
        ckpt_mcc = ckpt_cfg.get("dataset", {}).get("model_cond_channels", None)
        if ckpt_mcc is not None and int(ckpt_mcc) != cfg_model_cond_channels:
            raise ValueError(
                f"model_cond_channels mismatch: cfg={cfg_model_cond_channels}, "
                f"checkpoint={ckpt_mcc}. These must agree for r_min sweep."
            )


# ---------------------------------------------------------------------------
# Sweep dataset directory creation
# ---------------------------------------------------------------------------


def sweep_dataset_name(r_min: float) -> str:
    """Canonical sweep dataset directory name for a given r_min."""
    return f"sweep_rmin_{r_min:.2f}"


def create_sweep_dataset_dirs(
    *,
    reference_raw_dir: str,
    output_root: str,
    r_min_values: Sequence[float],
) -> list[str]:
    """Create symlinked dataset directories for each swept r_min value.

    Returns list of dataset names (relative to *output_root*).
    """
    reference_raw_dir = osp.realpath(reference_raw_dir)

    # Load reference network_info.json
    ref_info_path = osp.join(reference_raw_dir, "network_info.json")
    with open(ref_info_path) as f:
        ref_network_info = json.load(f)

    dataset_names: list[str] = []
    for r_min in r_min_values:
        ds_name = sweep_dataset_name(r_min)
        raw_dir = osp.join(output_root, ds_name, "raw")
        os.makedirs(raw_dir, exist_ok=True)

        # Symlink raw artifacts (not network_info.json)
        for fname in ("h_instantaneous", "collected_samples.npz", "config.yaml"):
            src = osp.join(reference_raw_dir, fname)
            dst = osp.join(raw_dir, fname)
            if osp.isdir(dst) and not osp.islink(dst):
                shutil.rmtree(dst)
            elif osp.exists(dst) or osp.islink(dst):
                os.remove(dst)
            if osp.exists(src) or osp.isdir(src):
                os.symlink(src, dst)

        # Write modified network_info.json with r_min overridden at all levels
        modified_info = _override_r_min_in_network_info(ref_network_info, r_min)
        info_path = osp.join(raw_dir, "network_info.json")
        with open(info_path, "w") as f:
            json.dump(modified_info, f, indent=2)

        dataset_names.append(ds_name)

    return dataset_names


def _override_r_min_in_network_info(
    network_info: dict,
    r_min: float,
) -> dict:
    """Deep-copy network_info and override r_min at global and per-network levels."""
    info = copy.deepcopy(network_info)

    # Global system_params.r_min
    if "system_params" not in info:
        info["system_params"] = {}
    info["system_params"]["r_min"] = float(r_min)

    # Per-network system_params.r_min
    for key in list(info.keys()):
        if key.startswith("network_") and isinstance(info[key], dict):
            if "system_params" not in info[key]:
                info[key]["system_params"] = {}
            info[key]["system_params"]["r_min"] = float(r_min)

    return info


# ---------------------------------------------------------------------------
# Dataset info extraction (self-contained, no WRABuilder dependency)
# ---------------------------------------------------------------------------


def build_sweep_dataset_info(
    *,
    dataset_root: str,
    dataset_name: str,
    r_min: float,
    system_params: dict[str, Any],
    channel_params: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Extract evaluator-consumable dataset_info from processed sweep dataset.

    Reads graph.pt and network_info.json directly instead of depending on
    WRABuilder._extract_dataset_info().
    """
    processed_dir = osp.join(dataset_root, dataset_name, "processed")
    raw_dir = osp.join(dataset_root, dataset_name, "raw")

    # Load network_info for channel metadata
    info_path = osp.join(raw_dir, "network_info.json")
    with open(info_path) as f:
        network_info = json.load(f)

    associations_map: dict[tuple[str, int], np.ndarray] = {}
    h_ls_gains_map: dict[tuple[str, int], np.ndarray] = {}
    network_seeds_map: dict[tuple[str, int], int] = {}
    channel_versions_map: dict[tuple[str, int], str] = {}
    h_timeslot_dirs_map: dict[tuple[str, int], str] = {}
    h_num_timesteps_map: dict[tuple[str, int], int] = {}

    # Propagate channel_cache_info from network_info.json so the evaluator
    # can fall back to on-demand H generation if precomputed H is missing.
    channel_cache_info_map: dict[str, dict[str, Any]] = {}
    cci = network_info.get("channel_cache_info")
    if isinstance(cci, dict) and cci.get("channel_cache_file"):
        channel_cache_info_map[dataset_name] = dict(cci)

    # Scan processed directory for network_* subdirs
    if not osp.isdir(processed_dir):
        raise FileNotFoundError(f"Processed directory not found: {processed_dir}")

    for entry in sorted(os.listdir(processed_dir)):
        if not entry.startswith("network_"):
            continue
        network_dir = osp.join(processed_dir, entry)
        if not osp.isdir(network_dir):
            continue

        graph_path = osp.join(network_dir, "graph.pt")
        if not osp.exists(graph_path):
            continue

        graph_data = torch.load(graph_path, map_location="cpu", weights_only=False)

        # Parse network_id from directory name
        net_id_str = entry  # e.g. "network_0"
        net_id = int(net_id_str.split("_", 1)[1])
        key = (dataset_name, net_id)

        # Associations
        assoc = graph_data.get("associations")
        if assoc is not None:
            if isinstance(assoc, torch.Tensor):
                assoc = assoc.numpy()
            associations_map[key] = np.asarray(assoc, dtype=np.float32)

        # H_ls gains
        h_ls = graph_data.get("H_ls")
        if h_ls is not None:
            if isinstance(h_ls, torch.Tensor):
                h_ls = h_ls.numpy()
            h_ls_gains_map[key] = np.asarray(h_ls, dtype=np.float32)

        # Network seed
        seed = graph_data.get("network_seed")
        if seed is not None:
            network_seeds_map[key] = int(seed)

        # Channel version
        cv = graph_data.get("channel_version")
        if cv is not None:
            channel_versions_map[key] = str(cv)
        else:
            # Fallback to network_info
            per_net = network_info.get(net_id_str, {})
            channel_versions_map[key] = str(
                per_net.get("channel_version",
                            network_info.get("channel_version", "v2"))
            )

        # H_instantaneous timeslot directory
        h_dir = osp.join(network_dir, "H_instantaneous")
        if osp.isdir(h_dir) or osp.islink(h_dir):
            # Resolve symlink to actual directory
            real_h_dir = osp.realpath(h_dir)
            if osp.isdir(real_h_dir):
                h_timeslot_dirs_map[key] = real_h_dir
                # Count timestep files
                n_ts = len([
                    f for f in os.listdir(real_h_dir)
                    if f.startswith("timestep_") and f.endswith(".pt")
                ])
                h_num_timesteps_map[key] = n_ts

    # Inject swept r_min into system_params
    swept_system_params = dict(system_params)
    swept_system_params["r_min"] = float(r_min)

    dataset_info: dict[str, Any] = {
        "system_params": swept_system_params,
        "channel_params": dict(channel_params) if channel_params else {},
        "associations": associations_map,
        "h_ls_gains": h_ls_gains_map,
        "network_seeds": network_seeds_map,
        "channel_versions": channel_versions_map,
        "h_timeslot_dirs": h_timeslot_dirs_map,
        "h_num_timesteps": h_num_timesteps_map,
        "channel_cache_info": channel_cache_info_map,
        "r_min_per_dataset": {dataset_name: float(r_min)},
    }
    return dataset_info


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

_SUMMARY_METRICS = [
    "mean_rate_generated",
    "sum_rate_generated",
    "rate_0.1pct_generated",
    "rate_0.5pct_generated",
    "rate_1pct_generated",
    "rate_5pct_generated",
    "rate_10pct_generated",
    "rate_violation_percentage_generated",
    "rate_mean_violation_gap_pct_generated",
    "feasibility_rate",
]


def save_sweep_summary(
    *,
    sweep_results: list[dict[str, Any]],
    native_r_min_values: Sequence[float],
    output_dir: str,
) -> None:
    """Save r_min sweep summary as CSV and markdown table.

    Parameters
    ----------
    sweep_results : list of dicts
        Each dict has ``"r_min"`` and metric keys from ``_SUMMARY_METRICS``.
    native_r_min_values : sequence of float
        r_min values that were in the training set.
    output_dir : str
        Directory to save outputs.
    """
    import pandas as pd

    native_set = {round(float(v), 6) for v in native_r_min_values}

    rows = []
    for entry in sorted(sweep_results, key=lambda e: e["r_min"]):
        r = float(entry["r_min"])
        row: dict[str, Any] = {
            "r_min": r,
            "is_native": round(r, 6) in native_set,
        }
        for metric in _SUMMARY_METRICS:
            row[metric] = entry.get(metric)
            # Include per-network std if available (for error bars)
            std_key = metric.replace("_generated", "_generated_std")
            if std_key in entry:
                row[std_key] = entry[std_key]
        rows.append(row)

    df = pd.DataFrame(rows)

    csv_path = osp.join(output_dir, "r_min_sweep_summary.csv")
    df.to_csv(csv_path, index=False)

    # Markdown table
    md_lines = ["# r_min Sweep Summary\n"]
    header_cols = ["r_min", "native"] + [m.replace("_generated", "") for m in _SUMMARY_METRICS]
    md_lines.append("| " + " | ".join(header_cols) + " |")
    md_lines.append("|" + "|".join(["---"] * len(header_cols)) + "|")
    for _, row in df.iterrows():
        cells = [f"{row['r_min']:.2f}", "yes" if row["is_native"] else ""]
        for metric in _SUMMARY_METRICS:
            v = row.get(metric)
            if v is not None and not (isinstance(v, float) and math.isnan(v)):
                cells.append(f"{v:.4f}")
            else:
                cells.append("—")
        md_lines.append("| " + " | ".join(cells) + " |")

    md_path = osp.join(output_dir, "r_min_sweep_summary.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"  r_min sweep summary saved to {csv_path}")
    print(f"  r_min sweep summary (md) saved to {md_path}")


# ---------------------------------------------------------------------------
# r_min-vs-percentile plots
# ---------------------------------------------------------------------------

_SWEEP_PLOT_PERCENTILES = [
    ("rate_0.1pct_generated", "0.1st percentile"),
    ("rate_0.5pct_generated", "0.5th percentile"),
    ("rate_1pct_generated", "1st percentile"),
    ("rate_5pct_generated", "5th percentile"),
    ("rate_10pct_generated", "10th percentile"),
    ("mean_rate_generated", "Mean rate"),
]


def _style_get(style_cfg: dict, path: str, default: Any) -> Any:
    current: Any = style_cfg
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def plot_r_min_sweep(
    *,
    sweep_results: list[dict[str, Any]],
    native_r_min_values: Sequence[float],
    output_dir: str,
    plot_style: Optional[dict[str, Any]] = None,
) -> None:
    """Generate r_min-vs-percentile plots.

    Produces individual per-percentile PDFs and a combined figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8-darkgrid")

    style = plot_style or {}
    sweep_style = _style_get(style, "plots.r_min_sweep", {})
    if not isinstance(sweep_style, dict):
        sweep_style = {}

    dpi = int(_style_get(sweep_style, "dpi", 300))
    fig_size = _style_get(sweep_style, "figure_size", [7.5, 5.0])
    combined_fig_size = _style_get(sweep_style, "combined_figure_size", [24.0, 12.0])
    line_color = str(_style_get(sweep_style, "line.color", "#ff7f0e"))
    line_width = float(_style_get(sweep_style, "line.linewidth", 2.0))
    marker_native = str(_style_get(sweep_style, "line.marker_native", "D"))
    marker_transfer = str(_style_get(sweep_style, "line.marker_transfer", "o"))
    marker_size = float(_style_get(sweep_style, "line.markersize", 8.0))
    qos_color = str(_style_get(sweep_style, "qos_line.color", "red"))
    qos_ls = str(_style_get(sweep_style, "qos_line.linestyle", "--"))
    qos_lw = float(_style_get(sweep_style, "qos_line.linewidth", 1.6))
    qos_alpha = float(_style_get(sweep_style, "qos_line.alpha", 0.8))
    qos_label_symbol = str(
        _style_get(sweep_style, "qos_line.label_symbol", r"f_{\min}") or r"f_{\min}"
    )
    font_axis = int(_style_get(sweep_style, "fonts.axis_label", 16))
    font_title = int(_style_get(sweep_style, "fonts.title", 16))
    font_legend = int(_style_get(sweep_style, "fonts.legend", 14))
    font_tick = int(_style_get(sweep_style, "fonts.tick", 16))
    grid_alpha = float(_style_get(sweep_style, "grid.alpha", 0.3))
    grid_ls = str(_style_get(sweep_style, "grid.linestyle", "--"))
    eb_capsize = float(_style_get(sweep_style, "error_bar.capsize", 3.0))
    eb_capthick = float(_style_get(sweep_style, "error_bar.capthick", 1.2))
    eb_elinewidth = float(_style_get(sweep_style, "error_bar.elinewidth", 1.0))
    eb_alpha = float(_style_get(sweep_style, "error_bar.alpha", 0.6))

    native_set = {round(float(v), 6) for v in native_r_min_values}

    # Sort results by r_min
    sorted_results = sorted(sweep_results, key=lambda e: e["r_min"])
    r_min_vals = [e["r_min"] for e in sorted_results]
    is_native = [round(r, 6) in native_set for r in r_min_vals]

    sweep_dir = osp.join(output_dir, "r_min_sweep_plots")
    os.makedirs(sweep_dir, exist_ok=True)

    def _std_key(mk: str) -> str:
        return mk.replace("_generated", "_generated_std")

    # Individual per-percentile plots
    for metric_key, metric_label in _SWEEP_PLOT_PERCENTILES:
        values = [e.get(metric_key) for e in sorted_results]
        yerr = [e.get(_std_key(metric_key)) for e in sorted_results]
        fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
        _draw_sweep_panel(
            ax, r_min_vals, values, is_native,
            metric_label=metric_label,
            line_color=line_color, line_width=line_width,
            marker_native=marker_native, marker_transfer=marker_transfer,
            marker_size=marker_size,
            qos_color=qos_color, qos_ls=qos_ls, qos_lw=qos_lw, qos_alpha=qos_alpha,
            qos_label_symbol=qos_label_symbol,
            font_axis=font_axis, font_title=font_title, font_legend=font_legend,
            font_tick=font_tick,
            grid_alpha=grid_alpha, grid_ls=grid_ls,
            yerr_values=yerr,
            eb_capsize=eb_capsize, eb_capthick=eb_capthick,
            eb_elinewidth=eb_elinewidth, eb_alpha=eb_alpha,
        )
        safe_name = metric_key.replace(".", "_")
        fig.tight_layout()
        fig.savefig(osp.join(sweep_dir, f"r_min_sweep_{safe_name}.pdf"), dpi=dpi)
        plt.close(fig)

    # Combined figure
    n_panels = len(_SWEEP_PLOT_PERCENTILES)
    n_cols = 3
    n_rows = math.ceil(n_panels / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=combined_fig_size, dpi=dpi)
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for idx, (metric_key, metric_label) in enumerate(_SWEEP_PLOT_PERCENTILES):
        values = [e.get(metric_key) for e in sorted_results]
        yerr = [e.get(_std_key(metric_key)) for e in sorted_results]
        _draw_sweep_panel(
            axes_flat[idx], r_min_vals, values, is_native,
            metric_label=metric_label,
            line_color=line_color, line_width=line_width,
            marker_native=marker_native, marker_transfer=marker_transfer,
            marker_size=marker_size,
            qos_color=qos_color, qos_ls=qos_ls, qos_lw=qos_lw, qos_alpha=qos_alpha,
            qos_label_symbol=qos_label_symbol,
            font_axis=font_axis, font_title=font_title - 2, font_legend=font_legend - 1,
            font_tick=font_tick - 2,
            grid_alpha=grid_alpha, grid_ls=grid_ls,
            yerr_values=yerr,
            eb_capsize=eb_capsize, eb_capthick=eb_capthick,
            eb_elinewidth=eb_elinewidth, eb_alpha=eb_alpha,
        )
    # Hide unused axes
    for idx in range(n_panels, len(axes_flat)):
        axes_flat[idx].set_visible(False)
    fig.tight_layout()
    fig.savefig(osp.join(sweep_dir, "r_min_sweep_all_percentiles.pdf"), dpi=dpi)
    plt.close(fig)

    # --- Overlay plot: configurable percentiles on one figure ---
    _DEFAULT_OVERLAY = [
        {"metric": "rate_1pct_generated", "label": "p1"},
        {"metric": "rate_5pct_generated", "label": "p5"},
        {"metric": "rate_10pct_generated", "label": "p10"},
    ]
    _overlay_cfg_raw = _style_get(sweep_style, "overlay_percentiles", _DEFAULT_OVERLAY)
    if not isinstance(_overlay_cfg_raw, list) or not _overlay_cfg_raw:
        _overlay_cfg_raw = _DEFAULT_OVERLAY
    _OVERLAY_METRICS = [
        (entry.get("metric", ""), entry.get("label", ""))
        for entry in _overlay_cfg_raw
        if isinstance(entry, dict) and entry.get("metric")
    ]
    if not _OVERLAY_METRICS:
        _OVERLAY_METRICS = [(e["metric"], e["label"]) for e in _DEFAULT_OVERLAY]

    _overlay_colors = _style_get(sweep_style, "overlay_colors",
                                  ["#1f77b4", "#ff7f0e", "#2ca02c"])
    if not isinstance(_overlay_colors, list) or len(_overlay_colors) < len(_OVERLAY_METRICS):
        _overlay_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig_ov, ax_ov = plt.subplots(figsize=fig_size, dpi=dpi)

    for ci, (metric_key, metric_label) in enumerate(_OVERLAY_METRICS):
        values = [e.get(metric_key) for e in sorted_results]
        errs = [e.get(_std_key(metric_key)) for e in sorted_results]
        color = _overlay_colors[ci]

        def _ok(v):
            return v is not None and not (isinstance(v, float) and math.isnan(v))

        # Line through all valid points (with error band if available)
        r_valid = [r for r, v in zip(r_min_vals, values) if _ok(v)]
        v_valid = [v for v in values if _ok(v)]
        e_valid = [float(ev) if ev and ev > 0 else 0.0
                   for ev, v in zip(errs, values) if _ok(v)]
        has_errs = any(e > 0 for e in e_valid)
        if r_valid:
            if has_errs:
                ax_ov.errorbar(r_valid, v_valid, yerr=e_valid, color=color,
                               linewidth=line_width, zorder=1, fmt="none",
                               capsize=eb_capsize, capthick=eb_capthick,
                               elinewidth=eb_elinewidth, alpha=eb_alpha)
            ax_ov.plot(r_valid, v_valid, color=color, linewidth=line_width,
                       zorder=1, label=metric_label)

        # Native markers (filled) — no label (shared legend entry added below)
        r_nat = [r for r, v, n in zip(r_min_vals, values, is_native) if n and _ok(v)]
        v_nat = [v for v, n in zip(values, is_native) if n and _ok(v)]
        if r_nat:
            ax_ov.scatter(r_nat, v_nat, marker=marker_native, s=marker_size ** 2,
                          color=color, zorder=2)

        # Transfer markers (open) — no label
        r_tr = [r for r, v, n in zip(r_min_vals, values, is_native) if not n and _ok(v)]
        v_tr = [v for v, n in zip(values, is_native) if not n and _ok(v)]
        if r_tr:
            ax_ov.scatter(r_tr, v_tr, marker=marker_transfer, s=marker_size ** 2,
                          facecolors="white", edgecolors=color, linewidths=1.5,
                          zorder=2)

    # Shared marker legend entries (gray, one per marker type)
    ax_ov.scatter([], [], marker=marker_native, s=marker_size ** 2,
                  color="gray", label=rf"Native ${qos_label_symbol}$")
    ax_ov.scatter([], [], marker=marker_transfer, s=marker_size ** 2,
                  facecolors="white", edgecolors="gray", linewidths=1.5,
                  label=rf"Transferred ${qos_label_symbol}$")

    # QoS reference line
    if r_min_vals:
        r_range = [min(r_min_vals), max(r_min_vals)]
        ax_ov.plot(r_range, r_range, color=qos_color, linestyle=qos_ls,
                   linewidth=qos_lw, alpha=qos_alpha,
                   label=rf"QoS threshold ($y = {qos_label_symbol}$)")

    ax_ov.set_xlabel(rf"${qos_label_symbol}$ (bits/s/Hz)", fontsize=font_axis)
    ax_ov.set_ylabel("Ergodic rate (bits/s/Hz)", fontsize=font_axis)
    ax_ov.tick_params(axis="both", labelsize=font_tick)
    leg = ax_ov.legend(fontsize=font_legend, frameon=True, framealpha=0.5,
                       edgecolor="gray", fancybox=True, facecolor="white")
    leg.get_frame().set_linewidth(1.0)
    ax_ov.grid(alpha=grid_alpha, linestyle=grid_ls)
    fig_ov.tight_layout()
    fig_ov.savefig(osp.join(sweep_dir, "r_min_sweep_tail_overlay.pdf"), dpi=dpi)
    plt.close(fig_ov)

    print(f"  r_min sweep plots saved to {sweep_dir}/")


def _draw_sweep_panel(
    ax,
    r_min_vals: list[float],
    values: list[Optional[float]],
    is_native: list[bool],
    *,
    metric_label: str,
    line_color: str,
    line_width: float,
    marker_native: str,
    marker_transfer: str,
    marker_size: float,
    qos_color: str,
    qos_ls: str,
    qos_lw: float,
    qos_alpha: float,
    qos_label_symbol: str = r"f_{\min}",
    font_axis: int,
    font_title: int,
    font_legend: int,
    font_tick: int = 12,
    grid_alpha: float,
    grid_ls: str,
    yerr_values: Optional[list[Optional[float]]] = None,
    eb_capsize: float = 3.0,
    eb_capthick: float = 1.2,
    eb_elinewidth: float = 1.0,
    eb_alpha: float = 0.6,
) -> None:
    """Draw a single r_min sweep panel on the given axes."""
    _yerr = yerr_values or [None] * len(r_min_vals)
    # Split into native and transferred points
    r_native, v_native, e_native = [], [], []
    r_transfer, v_transfer, e_transfer = [], [], []
    r_all, v_all, e_all = [], [], []
    for r, v, native, ev in zip(r_min_vals, values, is_native, _yerr):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        _e = float(ev) if ev is not None and ev > 0 else 0.0
        r_all.append(r)
        v_all.append(v)
        e_all.append(_e)
        if native:
            r_native.append(r)
            v_native.append(v)
            e_native.append(_e)
        else:
            r_transfer.append(r)
            v_transfer.append(v)
            e_transfer.append(_e)

    has_errs = any(e > 0 for e in e_all)

    # Line through all points (with error bars if available)
    if r_all:
        if has_errs:
            ax.errorbar(r_all, v_all, yerr=e_all, color=line_color,
                        linewidth=line_width, zorder=1, fmt="none",
                        capsize=eb_capsize, capthick=eb_capthick,
                        elinewidth=eb_elinewidth, alpha=eb_alpha)
            ax.plot(r_all, v_all, color=line_color, linewidth=line_width, zorder=1)
        else:
            ax.plot(r_all, v_all, color=line_color, linewidth=line_width, zorder=1)

    # Native markers (filled)
    if r_native:
        ax.scatter(
            r_native, v_native,
            marker=marker_native, s=marker_size ** 2, color=line_color,
            zorder=2, label=rf"Native ${qos_label_symbol}$",
        )
    # Transfer markers (open)
    if r_transfer:
        ax.scatter(
            r_transfer, v_transfer,
            marker=marker_transfer, s=marker_size ** 2,
            facecolors="white", edgecolors=line_color, linewidths=1.5,
            zorder=2, label=rf"Transferred ${qos_label_symbol}$",
        )

    # QoS reference line (y = f_min)
    if r_all:
        r_range = [min(r_all), max(r_all)]
        ax.plot(
            r_range, r_range,
            color=qos_color, linestyle=qos_ls, linewidth=qos_lw,
            alpha=qos_alpha, label=rf"QoS threshold ($y = {qos_label_symbol}$)",
        )

    ax.set_xlabel(rf"${qos_label_symbol}$ (bits/s/Hz)", fontsize=font_axis)
    ax.set_ylabel("Ergodic rate (bits/s/Hz)", fontsize=font_axis)
    ax.set_title(metric_label, fontsize=font_title)
    ax.tick_params(axis="both", labelsize=font_tick)
    # Only show legend when at least one artist has a label.
    _, labels = ax.get_legend_handles_labels()
    if labels:
        _leg = ax.legend(fontsize=font_legend, frameon=True, framealpha=0.9,
                         edgecolor="gray", fancybox=True, facecolor="white")
        _leg.get_frame().set_linewidth(1.0)
    ax.grid(alpha=grid_alpha, linestyle=grid_ls)
