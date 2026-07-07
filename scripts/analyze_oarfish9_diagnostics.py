#!/usr/bin/env python3
"""Extract per-subdataset gap percentile metrics from all-density-all-rmin
experiments and produce CSVs + evolution plots.

Usage:
    python scripts/analyze_oarfish9_diagnostics.py [EXP_NAME ...]
    # defaults to both: sophisticated-oarfish-9 loutish-wolf-454
"""

import json
import csv
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

BASE_DIR = (
    "outputs/wireless_resource_allocation-wra/"
    "ugnn_wra_v3_ds4-ddim_wra-gdm_wra_medium-large_outdoor_all_density_all_rmin"
)
DEFAULT_EXPERIMENTS = ["sophisticated-oarfish-9", "loutish-wolf-454"]

# ── Hash → (density_label, r_min) mapping from dataset config ──
HASH_TO_LABEL = {
    # ultra-low density
    "h51d52d48c355": ("ultra-low", 0.4),
    "h54416c2fba47": ("ultra-low", 0.5),
    "h0dd7afd393f9": ("ultra-low", 0.6),
    "h56b70909ae82": ("ultra-low", 0.7),
    "he1da74a5635c": ("ultra-low", 0.8),
    # low density
    "hd8d168fe5810": ("low", 0.4),
    "h26dde690eca5": ("low", 0.5),
    "h43d4a26a4203": ("low", 0.6),
    "h0f9e518a51ae": ("low", 0.7),
    "h391ad7cf511b": ("low", 0.8),
    # mid density
    "haa4d4fdc8221": ("mid", 0.4),
    "h3fddacd0eadc": ("mid", 0.5),
    "hc1f8f7a25432": ("mid", 0.6),
    "h4ab03934d654": ("mid", 0.7),
    "h943a82aae11f": ("mid", 0.8),
    # high density
    "h9c4284c3b7ea": ("high", 0.4),
    "h5b791d7f2908": ("high", 0.5),
    "ha6c7c432ee13": ("high", 0.6),
    "h26614d2f6640": ("high", 0.7),
    "h9681f9e137e7": ("high", 0.8),
}

# Reverse: hash suffix from full subdataset name
def _extract_hash(subdataset_name: str) -> str:
    """e.g. 'wrpc_v1_primal_history_k200_h26dde690eca5' -> 'h26dde690eca5'"""
    return subdataset_name.rsplit("_", 1)[-1]


def load_epoch_summaries(jsonl_path):
    records = []
    with open(jsonl_path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def extract_metrics(records):
    """Return dict: {epoch -> {(density, r_min) -> {metric_name: value}}}"""
    data = {}
    # Determine which split prefix to use (train-val or val)
    # by checking what's available in the first eval epoch
    split_prefix = None
    for rec in records:
        for k in rec:
            if "subdataset/" in k and "rate_1pct_gap_pct" in k:
                split_prefix = k.split("subdataset/")[0] + "subdataset/"
                break
        if split_prefix:
            break

    if not split_prefix:
        raise RuntimeError("No subdataset metrics found in epoch summaries")

    print(f"Using split prefix: '{split_prefix}'")

    METRICS = [
        "min_rate_gap_pct",  # 0.1pct gap (min-rate gap)
        "rate_1pct_gap_pct", "rate_5pct_gap_pct",
        "rate_mean_violation_gap_pct_generated",
        "rate_1pct_generated", "rate_1pct_real",
        "rate_5pct_generated", "rate_5pct_real",
        "rate_0.1pct_generated", "rate_0.1pct_real",
        "rate_0.2pct_generated", "rate_0.2pct_real",
        "min_rate_generated", "min_rate_real",
        "sum_rate_gap_pct",
    ]

    for rec in records:
        epoch = rec["epoch"]
        # Check if this epoch has subdataset metrics
        has_subdataset = any(split_prefix in k for k in rec)
        if not has_subdataset:
            continue

        epoch_data = {}
        for subdataset_name, (density, r_min) in HASH_TO_LABEL.items():
            full_hash_name = None
            # Find the matching full subdataset key
            for k in rec:
                if split_prefix in k and subdataset_name in k:
                    full_hash_name = k.split("/")[1]
                    break
            if full_hash_name is None:
                continue

            metrics = {}
            for m in METRICS:
                key = f"{split_prefix}{full_hash_name}/{m}"
                if key in rec:
                    metrics[m] = rec[key]

            # Compute composite score: mean of 0.1p, 1p, 5p gaps (matching tracker logic)
            g01 = metrics.get("min_rate_gap_pct")
            g1 = metrics.get("rate_1pct_gap_pct")
            g5 = metrics.get("rate_5pct_gap_pct")
            if g01 is not None and g1 is not None and g5 is not None:
                metrics["composite_score"] = (g01 + g1 + g5) / 3.0

            if metrics:
                epoch_data[(density, r_min)] = metrics

        if epoch_data:
            data[epoch] = epoch_data

    return data


def write_csv(data, out_path):
    """Write CSV with columns: epoch, then per-(density,r_min) gap metrics."""
    densities = ["ultra-low", "low", "mid", "high"]
    r_mins = [0.4, 0.5, 0.6, 0.7, 0.8]

    header = ["epoch"]
    for density in densities:
        for r_min in r_mins:
            tag = f"{density}_r{r_min}"
            header.extend([
                f"{tag}_0.1p_gap", f"{tag}_1p_gap", f"{tag}_5p_gap",
                f"{tag}_mean_gap", f"{tag}_composite",
            ])

    rows = []
    for epoch in sorted(data.keys()):
        row = [epoch]
        for density in densities:
            for r_min in r_mins:
                m = data[epoch].get((density, r_min), {})
                row.extend([
                    round(m.get("min_rate_gap_pct", float("nan")), 2),
                    round(m.get("rate_1pct_gap_pct", float("nan")), 2),
                    round(m.get("rate_5pct_gap_pct", float("nan")), 2),
                    round(m.get("rate_mean_violation_gap_pct_generated", float("nan")), 2),
                    round(m.get("composite_score", float("nan")), 2),
                ])
        rows.append(row)

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"Wrote CSV: {out_path} ({len(rows)} epochs)")


DENSITIES = ["ultra-low", "low", "mid", "high"]
R_MINS = [0.4, 0.5, 0.6, 0.7, 0.8]
DENSITY_COLORS = {
    "ultra-low": "tab:cyan",
    "low": "tab:blue",
    "mid": "tab:green",
    "high": "tab:red",
}
R_MIN_COLORS = {
    0.4: "tab:blue",
    0.5: "tab:orange",
    0.6: "tab:green",
    0.7: "tab:red",
    0.8: "tab:purple",
}
R_MIN_LINESTYLES = {
    0.4: "-",
    0.5: "--",
    0.6: "-.",
    0.7: ":",
    0.8: (0, (3, 1, 1, 1, 1, 1)),
}
DENSITY_LINESTYLES = {
    "ultra-low": "-",
    "low": "--",
    "mid": "-.",
    "high": ":",
}
PANEL_CONFIGS = [
    ("Mean Violation Gap %", "mean_gap", "Gap %"),
    ("Composite Score", "composite", "Score"),
    ("Min-rate (0.1pct) Gap %", "min_gap", "Gap %"),
    ("1-percentile Gap %", "1p_gap", "Gap %"),
    ("5-percentile Gap %", "5p_gap", "Gap %"),
    ("Min Rate (bps/Hz)", "min_rate", "Rate (bps/Hz)"),
    ("0.1-percentile Rate (bps/Hz)", "01p_rate", "Rate (bps/Hz)"),
    ("0.2-percentile Rate (bps/Hz)", "02p_rate", "Rate (bps/Hz)"),
    ("1-percentile Rate (bps/Hz)", "1p_rate", "Rate (bps/Hz)"),
    ("5-percentile Rate (bps/Hz)", "5p_rate", "Rate (bps/Hz)"),
]


def _build_series(data):
    """Build time series dict keyed by (density, r_min)."""
    epochs = sorted(data.keys())
    series = {}
    for density in DENSITIES:
        for r_min in R_MINS:
            vals = {
                "mean_gap": [], "composite": [], "min_gap": [],
                "1p_gap": [], "5p_gap": [],
                "1p_gen": [], "1p_real": [],
                "5p_gen": [], "5p_real": [],
                "01p_gen": [], "01p_real": [],
                "02p_gen": [], "02p_real": [],
                "min_gen": [], "min_real": [],
                "epochs": [],
            }
            for ep in epochs:
                m = data[ep].get((density, r_min))
                if m is None:
                    continue
                vals["epochs"].append(ep)
                vals["mean_gap"].append(m.get("rate_mean_violation_gap_pct_generated", float("nan")))
                vals["composite"].append(m.get("composite_score", float("nan")))
                vals["min_gap"].append(m.get("min_rate_gap_pct", float("nan")))
                vals["1p_gap"].append(m.get("rate_1pct_gap_pct", float("nan")))
                vals["5p_gap"].append(m.get("rate_5pct_gap_pct", float("nan")))
                vals["1p_gen"].append(m.get("rate_1pct_generated", float("nan")))
                vals["1p_real"].append(m.get("rate_1pct_real", float("nan")))
                vals["5p_gen"].append(m.get("rate_5pct_generated", float("nan")))
                vals["5p_real"].append(m.get("rate_5pct_real", float("nan")))
                vals["01p_gen"].append(m.get("rate_0.1pct_generated", float("nan")))
                vals["01p_real"].append(m.get("rate_0.1pct_real", float("nan")))
                vals["02p_gen"].append(m.get("rate_0.2pct_generated", float("nan")))
                vals["02p_real"].append(m.get("rate_0.2pct_real", float("nan")))
                vals["min_gen"].append(m.get("min_rate_generated", float("nan")))
                vals["min_real"].append(m.get("min_rate_real", float("nan")))
            series[(density, r_min)] = vals
    return series


def _plot_metric_on_ax(ax, series, metric_key, density, r_min, color, ls, label):
    """Plot a single metric series on an axis."""
    v = series[(density, r_min)]
    ep = v["epochs"]
    if not ep:
        return
    # Rate metrics: plot generated (solid) + real (faded)
    rate_key_map = {
        "1p_rate": ("1p_gen", "1p_real"),
        "5p_rate": ("5p_gen", "5p_real"),
        "01p_rate": ("01p_gen", "01p_real"),
        "02p_rate": ("02p_gen", "02p_real"),
        "min_rate": ("min_gen", "min_real"),
    }
    if metric_key in rate_key_map:
        gen_key, real_key = rate_key_map[metric_key]
        ax.plot(ep, v[gen_key], color=color, linestyle=ls,
                alpha=0.8, linewidth=1.5, label=label)
        ax.plot(ep, v[real_key], color=color, linestyle=ls,
                alpha=0.25, linewidth=1.0)
    else:
        ax.plot(ep, v[metric_key], color=color, linestyle=ls,
                alpha=0.8, linewidth=1.5, label=label)


def plot_evolution(data, out_path, exp_name=""):
    """6-panel overview: all 20 subdatasets overlaid, saved as PDF."""
    from matplotlib.lines import Line2D
    series = _build_series(data)

    nrows = (len(PANEL_CONFIGS) + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(18, 5 * nrows), sharex=True)
    fig.suptitle(f"Per-subdataset Metrics Evolution ({exp_name})",
                 fontsize=14, y=0.98)

    for idx, (title, metric_key, ylabel) in enumerate(PANEL_CONFIGS):
        ax = axes[idx // 2, idx % 2]
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        for density in DENSITIES:
            for r_min in R_MINS:
                _plot_metric_on_ax(
                    ax, series, metric_key, density, r_min,
                    color=DENSITY_COLORS[density],
                    ls=R_MIN_LINESTYLES[r_min],
                    label=f"{density} r={r_min}",
                )

    # Hide unused axes if odd number of panels
    if len(PANEL_CONFIGS) % 2 == 1:
        axes[-1, -1].set_visible(False)

    for col in range(2):
        axes[-1, col].set_xlabel("Epoch")

    legend_elements = []
    for density in DENSITIES:
        legend_elements.append(
            Line2D([0], [0], color=DENSITY_COLORS[density], linewidth=2,
                   label=density))
    for r_min in R_MINS:
        legend_elements.append(
            Line2D([0], [0], color="gray", linestyle=R_MIN_LINESTYLES[r_min],
                   linewidth=1.5, label=f"r_min={r_min}"))
    fig.legend(handles=legend_elements, loc="upper center",
               bbox_to_anchor=(0.5, 0.96), ncol=9, fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote plot: {out_path}")


def plot_per_density(data, out_path, exp_name=""):
    """For each of 6 metrics, create 4 sub-panels (one per density).
    Each sub-panel shows 5 r_min curves in distinct colors.
    Layout: 6 rows x 4 columns."""
    from matplotlib.lines import Line2D
    series = _build_series(data)

    nrows = len(PANEL_CONFIGS)
    fig, axes = plt.subplots(nrows, 4, figsize=(24, 4.5 * nrows), sharex=True)
    fig.suptitle(f"Per-density Metrics Evolution ({exp_name})",
                 fontsize=15, y=0.99)

    for row, (metric_title, metric_key, ylabel) in enumerate(PANEL_CONFIGS):
        for col, density in enumerate(DENSITIES):
            ax = axes[row, col]
            if row == 0:
                ax.set_title(density, fontsize=12, fontweight="bold")
            ax.set_ylabel(ylabel, fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)

            for r_min in R_MINS:
                _plot_metric_on_ax(
                    ax, series, metric_key, density, r_min,
                    color=R_MIN_COLORS[r_min],
                    ls="-",
                    label=f"r_min={r_min}",
                )

            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="upper right")

            # Row label on leftmost column
            if col == 0:
                ax.annotate(
                    metric_title, xy=(-0.35, 0.5),
                    xycoords="axes fraction", fontsize=9,
                    ha="center", va="center", rotation=90,
                    fontweight="bold",
                )

    for col in range(4):
        axes[-1, col].set_xlabel("Epoch")

    plt.tight_layout(rect=[0.03, 0, 1, 0.97])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote plot: {out_path}")


def plot_per_rmin(data, out_path, exp_name=""):
    """For each of 6 metrics, create 5 sub-panels (one per r_min).
    Each sub-panel shows 4 density curves in distinct colors.
    Layout: 6 rows x 5 columns."""
    from matplotlib.lines import Line2D
    series = _build_series(data)

    nrows = len(PANEL_CONFIGS)
    fig, axes = plt.subplots(nrows, 5, figsize=(28, 4.5 * nrows), sharex=True)
    fig.suptitle(f"Per-r_min Metrics Evolution ({exp_name})",
                 fontsize=15, y=0.99)

    for row, (metric_title, metric_key, ylabel) in enumerate(PANEL_CONFIGS):
        for col, r_min in enumerate(R_MINS):
            ax = axes[row, col]
            if row == 0:
                ax.set_title(f"r_min={r_min}", fontsize=12, fontweight="bold")
            ax.set_ylabel(ylabel, fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=8)

            for density in DENSITIES:
                _plot_metric_on_ax(
                    ax, series, metric_key, density, r_min,
                    color=DENSITY_COLORS[density],
                    ls="-",
                    label=density,
                )

            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="upper right")

            if col == 0:
                ax.annotate(
                    metric_title, xy=(-0.4, 0.5),
                    xycoords="axes fraction", fontsize=9,
                    ha="center", va="center", rotation=90,
                    fontweight="bold",
                )

    for col in range(5):
        axes[-1, col].set_xlabel("Epoch")

    plt.tight_layout(rect=[0.03, 0, 1, 0.97])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote plot: {out_path}")


def run_for_experiment(exp_name):
    exp_dir = os.path.join(BASE_DIR, exp_name)
    jsonl_path = os.path.join(exp_dir, "epoch_summaries.jsonl")

    if not os.path.exists(jsonl_path):
        print(f"ERROR: {jsonl_path} not found, skipping {exp_name}")
        return

    print(f"\n{'='*60}")
    print(f"  Analyzing: {exp_name}")
    print(f"{'='*60}")

    records = load_epoch_summaries(jsonl_path)
    print(f"Loaded {len(records)} epoch records")

    data = extract_metrics(records)
    eval_epochs = sorted(data.keys())
    print(f"Found {len(data)} eval epochs with subdataset metrics")
    if eval_epochs:
        print(f"Eval epoch range: {eval_epochs[0]} .. {eval_epochs[-1]}")

    csv_path = os.path.join(exp_dir, "per_subdataset_gap_percentiles.csv")
    write_csv(data, csv_path)

    plot_evolution(data, os.path.join(exp_dir, "per_subdataset_gap_evolution.pdf"), exp_name)
    plot_per_density(data, os.path.join(exp_dir, "per_density_gap_evolution.pdf"), exp_name)
    plot_per_rmin(data, os.path.join(exp_dir, "per_rmin_gap_evolution.pdf"), exp_name)


def main():
    experiments = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_EXPERIMENTS
    for exp_name in experiments:
        run_for_experiment(exp_name)


if __name__ == "__main__":
    main()
