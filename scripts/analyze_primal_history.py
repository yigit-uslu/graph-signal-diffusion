#!/usr/bin/env python3
"""Analyze primal_history.jsonl files: report statistics and plot histograms of power allocations."""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


FILES = {
    "low_density (r0.8)": (
        "outputs/wra_medium_outdoor_low_density/"
        "wrpd_v1_wrach_v1_s42_D128_N200_R5000_v3_hec5460acfc89_r0.8_a0.1_hdad67fbcb85f/"
        "2026-03-09/14-30-38/primal_history.jsonl"
    ),
    "mid_density (r0.6)": (
        "outputs/wra_medium_outdoor_mid_density/"
        "wrpd_v1_wrach_v1_s42_D128_N200_R4000_v3_h141b045276c2_r0.6_a0.1_h9b90ec129fac/"
        "2026-03-09/14-31-21/primal_history.jsonl"
    ),
}

P_MAX = 0.01  # Maximum power budget for normalization


def load_primal_history(filepath):
    """Load primal_history.jsonl, returning metadata and data records separately."""
    metadata = []
    data = []  # list of (epoch, network_id, power_allocations, rates)
    with open(filepath) as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("type") == "metadata":
                metadata.append(rec)
            else:
                data.append(rec)
    return metadata, data


def compute_statistics(data):
    """Compute per-epoch and global statistics for power allocations."""
    # Group by epoch
    epoch_data = defaultdict(list)
    all_powers = []
    all_rates = []

    for rec in data:
        epoch = rec["epoch"]
        powers = [p / P_MAX for p in rec["power_allocations"]]  # normalize by P_max
        rates = rec["rates"]
        epoch_data[epoch].extend(powers)
        all_powers.extend(powers)
        all_rates.extend(rates)

    all_powers = np.array(all_powers)
    all_rates = np.array(all_rates)

    # Per-epoch statistics
    epochs_sorted = sorted(epoch_data.keys())
    epoch_means = []
    epoch_medians = []
    epoch_stds = []
    epoch_maxs = []
    epoch_mins = []
    epoch_p90s = []
    epoch_p99s = []

    for ep in epochs_sorted:
        arr = np.array(epoch_data[ep])
        epoch_means.append(arr.mean())
        epoch_medians.append(np.median(arr))
        epoch_stds.append(arr.std())
        epoch_maxs.append(arr.max())
        epoch_mins.append(arr.min())
        epoch_p90s.append(np.percentile(arr, 90))
        epoch_p99s.append(np.percentile(arr, 99))

    stats = {
        "n_records": len(data),
        "n_epochs": len(epochs_sorted),
        "epoch_range": (epochs_sorted[0], epochs_sorted[-1]) if epochs_sorted else (None, None),
        "n_total_power_values": len(all_powers),
        "global_mean": all_powers.mean(),
        "global_median": np.median(all_powers),
        "global_std": all_powers.std(),
        "global_min": all_powers.min(),
        "global_max": all_powers.max(),
        "global_p5": np.percentile(all_powers, 5),
        "global_p25": np.percentile(all_powers, 25),
        "global_p75": np.percentile(all_powers, 75),
        "global_p90": np.percentile(all_powers, 90),
        "global_p95": np.percentile(all_powers, 95),
        "global_p99": np.percentile(all_powers, 99),
        "frac_zero": (all_powers == 0).sum() / len(all_powers),
        "frac_near_zero": (all_powers < 1e-6).sum() / len(all_powers),
        # Rate stats
        "rate_mean": all_rates.mean(),
        "rate_median": np.median(all_rates),
        "rate_std": all_rates.std(),
        "rate_min": all_rates.min(),
        "rate_max": all_rates.max(),
        # Per-epoch arrays for plotting
        "epochs": epochs_sorted,
        "epoch_means": epoch_means,
        "epoch_medians": epoch_medians,
        "epoch_stds": epoch_stds,
        "epoch_maxs": epoch_maxs,
        "epoch_p90s": epoch_p90s,
        "epoch_p99s": epoch_p99s,
    }
    return stats, all_powers, all_rates


def print_statistics(label, stats):
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")
    print(f"  Records (network snapshots): {stats['n_records']:,}")
    print(f"  Epochs: {stats['n_epochs']:,}  (range: {stats['epoch_range'][0]} → {stats['epoch_range'][1]})")
    print(f"  Total power values: {stats['n_total_power_values']:,}")
    print()
    print(f"  ── Power Allocation Statistics (normalized by P_max={P_MAX}) ──")
    print(f"  Mean:    {stats['global_mean']:.6f}")
    print(f"  Median:  {stats['global_median']:.6f}")
    print(f"  Std:     {stats['global_std']:.6f}")
    print(f"  Min:     {stats['global_min']:.6f}")
    print(f"  Max:     {stats['global_max']:.6f}")
    print(f"  P5:      {stats['global_p5']:.6f}")
    print(f"  P25:     {stats['global_p25']:.6f}")
    print(f"  P75:     {stats['global_p75']:.6f}")
    print(f"  P90:     {stats['global_p90']:.6f}")
    print(f"  P95:     {stats['global_p95']:.6f}")
    print(f"  P99:     {stats['global_p99']:.6f}")
    print(f"  Frac exactly 0:    {stats['frac_zero']:.4%}")
    print(f"  Frac < 1e-6/P_max: {stats['frac_near_zero']:.4%}")
    print()
    print("  ── Rate Statistics ──")
    print(f"  Mean:    {stats['rate_mean']:.4f}")
    print(f"  Median:  {stats['rate_median']:.4f}")
    print(f"  Std:     {stats['rate_std']:.4f}")
    print(f"  Min:     {stats['rate_min']:.4f}")
    print(f"  Max:     {stats['rate_max']:.4f}")
    print()


def plot_histograms(results, output_path):
    """Plot combined figure with histograms and epoch progression for all files."""
    n_files = len(results)
    fig, axes = plt.subplots(n_files, 3, figsize=(18, 5 * n_files))
    if n_files == 1:
        axes = axes.reshape(1, -1)

    for idx, (label, (stats, all_powers, all_rates)) in enumerate(results.items()):
        # ── Panel 1: Power allocation histogram (normalized p/P_max) ──
        ax = axes[idx, 0]
        # Clip for better visualization; show the bulk of the distribution
        p99 = np.percentile(all_powers, 99.5)
        clipped = all_powers[all_powers <= p99]
        ax.hist(clipped, bins=150, color="steelblue", alpha=0.8, edgecolor="none", density=True)
        ax.axvline(stats["global_mean"], color="red", linestyle="--", linewidth=1.5, label=f'Mean={stats["global_mean"]:.4f}')
        ax.axvline(stats["global_median"], color="orange", linestyle=":", linewidth=1.5, label=f'Median={stats["global_median"]:.4f}')
        ax.set_xlabel(r"$p / P_{\max}$")
        ax.set_ylabel("Density")
        ax.set_title(f"{label}\nNormalized Power Histogram (clipped at p99.5)")
        ax.legend(fontsize=9)

        # ── Panel 2: Log-scale histogram ──
        ax = axes[idx, 1]
        positive = all_powers[all_powers > 0]
        log_powers = np.log10(positive)
        ax.hist(log_powers, bins=150, color="darkorange", alpha=0.8, edgecolor="none", density=True)
        ax.axvline(np.log10(stats["global_mean"]), color="red", linestyle="--", linewidth=1.5,
                    label=f'log10(Mean)={np.log10(stats["global_mean"]):.2f}')
        ax.set_xlabel(r"$\log_{10}(p / P_{\max})$")
        ax.set_ylabel("Density")
        ax.set_title(f"{label}\n" + r"$\log_{10}$ Normalized Power (positive values only)")
        ax.legend(fontsize=9)

        # ── Panel 3: Epoch progression ──
        ax = axes[idx, 2]
        epochs = stats["epochs"]
        ax.plot(epochs, stats["epoch_means"], label="Mean", color="steelblue", linewidth=1.5)
        ax.fill_between(epochs,
                         np.array(stats["epoch_means"]) - np.array(stats["epoch_stds"]),
                         np.array(stats["epoch_means"]) + np.array(stats["epoch_stds"]),
                         alpha=0.2, color="steelblue", label="±1 Std")
        ax.plot(epochs, stats["epoch_medians"], label="Median", color="orange", linewidth=1.5, linestyle="--")
        ax.plot(epochs, stats["epoch_p90s"], label="P90", color="red", linewidth=1.0, linestyle=":")
        ax.plot(epochs, stats["epoch_p99s"], label="P99", color="darkred", linewidth=1.0, linestyle=":")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(r"$p / P_{\max}$")
        ax.set_title(f"{label}\nNormalized Power vs Epoch")
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"  Histogram saved to: {output_path}")


def main():
    root = Path(__file__).resolve().parent.parent
    results = {}

    for label, relpath in FILES.items():
        filepath = root / relpath
        print(f"\nLoading {label} from {filepath.name} ...")
        metadata, data = load_primal_history(filepath)
        print(f"  Loaded {len(metadata)} metadata records, {len(data)} data records.")
        stats, all_powers, all_rates = compute_statistics(data)
        print_statistics(label, stats)
        results[label] = (stats, all_powers, all_rates)

    # Plot
    out_dir = root / "figures"
    out_dir.mkdir(exist_ok=True)
    output_path = out_dir / "primal_history_power_allocation_analysis.png"
    plot_histograms(results, output_path)

    # Also save a combined summary
    print(f"\n{'=' * 70}")
    print("  COMBINED COMPARISON")
    print(f"{'=' * 70}")
    print(f"  (All power values normalized by P_max = {P_MAX})")
    for label, (stats, _, _) in results.items():
        print(f"  {label:30s}  mean={stats['global_mean']:.6f}  median={stats['global_median']:.6f}  "
              f"std={stats['global_std']:.6f}  p99={stats['global_p99']:.6f}  "
              f"frac<1e-6/P_max={stats['frac_near_zero']:.2%}")


if __name__ == "__main__":
    main()
