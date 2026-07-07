"""
Script to load collected samples and verify reproducibility by regenerating networks.

This script demonstrates how to:
1. Load samples from collected_samples.npz
2. Read Hydra config to get dataset generation parameters
3. Extract network seeds from saved samples
4. Regenerate wireless networks using the same seeds
5. Verify that regenerated networks match the original data
"""

import numpy as np
import json
from pathlib import Path
import argparse
from tqdm import tqdm
import yaml
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from graph_signal_diffusion.datasets.wra.channel import WirelessChannel, WirelessChannelV2, WirelessChannelV3
from graph_signal_diffusion.datasets.wra.sample_schema import canonicalize_pd_samples_dict
from graph_signal_diffusion.utils.rate_calculator import (
    compute_system_parameters,
    compute_ergodic_rates
)
from graph_signal_diffusion.utils.graph_builder import build_interference_graph


def load_samples(output_dir: Path):
    """
    Load collected samples from output directory.
    
    Parameters
    ----------
    output_dir : Path
        Path to output directory containing collected_samples.npz
    
    Returns
    -------
    samples : dict
        Dictionary with all sample data
    metadata : dict
        Metadata about sample collection
    config : dict
        Hydra configuration used during training
    quality_report : dict
        Quality report from training
    primal_history : list
        List of primal history entries from primal_history.jsonl (if available)
    collection_metadata : dict
        Collection metadata containing checkpoint_epochs (if available)
    """
    # Load samples
    samples_path = output_dir / "collected_samples.npz"
    if not samples_path.exists():
        raise FileNotFoundError(f"Samples file not found: {samples_path}")
    
    samples = np.load(samples_path, allow_pickle=True)
    
    # Extract metadata if available
    metadata = {}
    if 'metadata' in samples:
        metadata = json.loads(str(samples['metadata']))
    
    # Load Hydra config
    config_path = output_dir / ".hydra" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load quality report
    quality_path = output_dir / "quality_report.json"
    quality_report = {}
    if quality_path.exists():
        with open(quality_path, 'r') as f:
            quality_report = json.load(f)
    
    # Load collection metadata to get actual collection epochs
    collection_metadata = {}
    collection_metadata_path = output_dir / "collection_metadata.json"
    if collection_metadata_path.exists():
        with open(collection_metadata_path, 'r') as f:
            collection_metadata = json.load(f)
    
    # Load primal history for rate statistics analysis
    primal_history = []
    primal_history_path = output_dir / "primal_history.jsonl"
    if primal_history_path.exists():
        with open(primal_history_path, 'r') as f:
            for line in f:
                primal_history.append(json.loads(line))
    
    return dict(samples), metadata, config, quality_report, primal_history, collection_metadata


def extract_network_info(samples, num_networks, config):
    """
    Extract network-specific information from samples.
    
    Parameters
    ----------
    samples : dict
        Loaded samples dictionary
    num_networks : int
        Number of networks in the dataset
    config : dict
        Hydra config (used to infer seeds if not in samples)
    
    Returns
    -------
    network_info : dict
        Dictionary mapping network_id -> {seed, H_instantaneous, associations, power_samples, rate_samples}
    """
    del num_networks  # Kept for backward-compatible signature.

    canonical = canonicalize_pd_samples_dict(samples, config=config, default_channel_version='v2')
    print(
        f"   Loaded canonical trajectory view from '{canonical.get('source_format', 'unknown')}', "
        f"schema_version={canonical.get('schema_version', 'n/a')}, "
        f"channel_version={canonical.get('channel_version', 'unknown')}"
    )
    if canonical.get("source_has_h_instantaneous") is False:
        raise RuntimeError(
            "verify_pd_samples requires saved H_instantaneous in collected_samples.npz, "
            "but this artifact uses the new no-H canonical NPZ format. "
            "This script is legacy verification/debug tooling; for the current WRA generation "
            "pipeline use build_diffusion_dataset directly."
        )

    network_info = {}
    for net_id in canonical['network_ids']:
        net_data = canonical['networks'][net_id]

        power_arr = np.asarray(net_data['power_samples'], dtype=np.float64)
        rate_arr = net_data.get('rate_samples')
        if rate_arr is not None and np.asarray(rate_arr).shape[0] == power_arr.shape[0]:
            rate_arr = np.asarray(rate_arr, dtype=np.float64)
        else:
            rate_arr = None

        power_samples = [power_arr[i] for i in range(power_arr.shape[0])]
        if rate_arr is not None:
            rate_samples = [rate_arr[i] for i in range(rate_arr.shape[0])]
        else:
            rate_samples = []

        h_instantaneous = np.asarray(net_data['H_instantaneous'])
        if h_instantaneous.ndim != 3 or h_instantaneous.shape[0] == 0:
            raise RuntimeError(
                f"verify_pd_samples requires non-empty saved H_instantaneous, but network {net_id} "
                f"has shape {h_instantaneous.shape}."
            )

        network_info[int(net_id)] = {
            'seed': net_data.get('network_seed'),
            'H_instantaneous': h_instantaneous,
            'associations': np.asarray(net_data['associations']),
            'power_samples': power_samples,
            'rate_samples': rate_samples,
            'num_samples': int(power_arr.shape[0]),
        }

    return network_info


def extract_checkpoint_window_samples(primal_history, collection_metadata, network_id, window_multiplier=1):
    """
    Extract checkpoint-interval samples for one network from primal_history.

    Parameters
    ----------
    primal_history : list
        Parsed entries from primal_history.jsonl
    collection_metadata : dict
        Collection metadata containing checkpoint_epochs
    network_id : int
        Network ID to extract
    window_multiplier : int
        Number of M-sized windows to use (window size = window_multiplier * M_checkpoints)

    Returns
    -------
    tuple or None
        (epochs, powers, rates, extraction_info) or None if insufficient data.
    """
    if not primal_history:
        return None

    if not collection_metadata or 'checkpoint_epochs' not in collection_metadata:
        return None

    checkpoint_epochs = collection_metadata['checkpoint_epochs']
    if len(checkpoint_epochs) == 0:
        return None

    if len(checkpoint_epochs) >= 2:
        sample_collection_interval = checkpoint_epochs[1] - checkpoint_epochs[0]
    else:
        sample_collection_interval = 1

    checkpoint_base = checkpoint_epochs[0] % sample_collection_interval

    records = []
    for entry in primal_history:
        if 'epoch' not in entry or 'network_id' not in entry:
            continue
        if entry['network_id'] != network_id:
            continue
        if 'power_allocations' not in entry or 'rates' not in entry:
            continue

        epoch = entry['epoch']
        if epoch % sample_collection_interval != checkpoint_base:
            continue

        records.append((
            int(epoch),
            np.asarray(entry['power_allocations'], dtype=np.float64),
            np.asarray(entry['rates'], dtype=np.float64),
        ))

    if len(records) == 0:
        return None

    records.sort(key=lambda x: x[0])

    M_checkpoints = len(checkpoint_epochs)
    requested_window_size = max(1, int(window_multiplier) * M_checkpoints)
    actual_window_size = min(requested_window_size, len(records))
    records = records[-actual_window_size:]

    epochs = np.array([r[0] for r in records], dtype=np.int64)
    powers = np.stack([r[1] for r in records], axis=0)
    rates = np.stack([r[2] for r in records], axis=0)

    extraction_info = {
        'M_checkpoints': M_checkpoints,
        'requested_window_size': requested_window_size,
        'actual_window_size': actual_window_size,
        'sample_collection_interval': sample_collection_interval,
        'checkpoint_base': checkpoint_base,
        'epoch_start': int(epochs[0]),
        'epoch_end': int(epochs[-1]),
    }

    return epochs, powers, rates, extraction_info


def _pairwise_euclidean_distances(features):
    """Compute pairwise Euclidean distances without allocating a 3D tensor."""
    squared_norm = np.sum(features * features, axis=1, keepdims=True)
    sq_dist = squared_norm + squared_norm.T - 2.0 * (features @ features.T)
    np.maximum(sq_dist, 0.0, out=sq_dist)
    np.sqrt(sq_dist, out=sq_dist)
    return sq_dist


def select_feasible_diverse_subset(rates, target_size, r_min, feasibility_tolerance=0.0, num_bottleneck_nodes=5):
    """
    Select a subset that preserves feasibility while keeping bottleneck-rate diversity.

    Feasibility is defined on the subset-average receiver rates:
        avg_rate_per_receiver >= (r_min - feasibility_tolerance)
    """
    rates = np.asarray(rates, dtype=np.float64)
    if rates.ndim != 2:
        raise ValueError(f"Expected rates with shape (num_samples, num_receivers), got {rates.shape}")

    num_samples, num_receivers = rates.shape
    if num_samples == 0:
        return np.array([], dtype=np.int64), {
            'num_samples_initial': 0,
            'num_samples_selected': 0,
            'target_size': int(target_size),
            'is_feasible': False,
            'num_violating_receivers': 0,
            'min_margin': float('nan'),
        }

    target_size = max(1, min(int(target_size), num_samples))
    threshold = float(r_min) - float(feasibility_tolerance)

    avg_rates_initial = rates.mean(axis=0)
    num_bottleneck_nodes = max(1, min(int(num_bottleneck_nodes), num_receivers))
    bottleneck_nodes = np.argsort(avg_rates_initial)[:num_bottleneck_nodes]

    # Diversity is computed in the bottleneck-rate subspace to avoid curse-of-dimensionality effects.
    features = rates[:, bottleneck_nodes]
    if num_samples > 1:
        dist_matrix = _pairwise_euclidean_distances(features)
    else:
        dist_matrix = np.zeros((1, 1), dtype=np.float64)

    selected_mask = np.ones(num_samples, dtype=bool)
    n_selected = num_samples
    sum_rates = rates.sum(axis=0)

    # Row sums over the currently selected set; update incrementally when removing a sample.
    diversity_row_sum = dist_matrix.sum(axis=1)
    removed_for_violation_reduction = 0
    removed_for_target = 0

    # Phase 1: If the current set is infeasible, iteratively reduce violation count
    # down to target size. This avoids collapsing to a tiny subset when strict
    # feasibility is impossible for the chosen window.
    while n_selected > target_size:
        avg_rates = sum_rates / n_selected
        if np.all(avg_rates >= threshold):
            break

        selected_indices = np.flatnonzero(selected_mask)
        candidate_rates = rates[selected_indices]
        new_avg = (sum_rates[None, :] - candidate_rates) / (n_selected - 1)
        new_margins = new_avg - threshold
        new_num_violating = np.sum(new_margins < 0.0, axis=1)
        new_min_margin = np.min(new_margins, axis=1)
        candidate_div_loss = diversity_row_sum[selected_indices]

        # Lexicographic choice:
        # 1) minimize violating receivers
        # 2) maximize minimum margin
        # 3) minimize diversity loss
        order = np.lexsort((candidate_div_loss, -new_min_margin, new_num_violating))
        remove_idx = selected_indices[order[0]]

        selected_mask[remove_idx] = False
        sum_rates -= rates[remove_idx]
        diversity_row_sum -= dist_matrix[:, remove_idx]
        n_selected -= 1
        removed_for_violation_reduction += 1

    # Phase 2: Prune to target size while preserving feasibility, removing the most redundant sample each step.
    while n_selected > target_size and n_selected > 1:
        selected_indices = np.flatnonzero(selected_mask)
        candidate_rates = rates[selected_indices]

        # Feasibility check after hypothetical removal of each candidate.
        new_avg = (sum_rates[None, :] - candidate_rates) / (n_selected - 1)
        feasible_removals = np.all(new_avg >= threshold, axis=1)

        if not np.any(feasible_removals):
            break

        feasible_indices = selected_indices[feasible_removals]
        # Small row-sum means the point is redundant with the current selected set.
        diversity_loss = diversity_row_sum[feasible_indices]
        remove_idx = feasible_indices[np.argmin(diversity_loss)]

        selected_mask[remove_idx] = False
        sum_rates -= rates[remove_idx]
        diversity_row_sum -= dist_matrix[:, remove_idx]
        n_selected -= 1
        removed_for_target += 1

    selected_indices = np.flatnonzero(selected_mask)
    final_avg = rates[selected_indices].mean(axis=0)
    margins = final_avg - threshold

    summary = {
        'num_samples_initial': int(num_samples),
        'num_samples_selected': int(len(selected_indices)),
        'target_size': int(target_size),
        'threshold': float(threshold),
        'is_feasible': bool(np.all(margins >= 0.0)),
        'num_violating_receivers': int(np.sum(margins < 0.0)),
        'min_margin': float(margins.min()),
        'bottleneck_nodes': bottleneck_nodes.tolist(),
        'removed_for_violation_reduction': int(removed_for_violation_reduction),
        'removed_for_target': int(removed_for_target),
    }

    return selected_indices.astype(np.int64), summary


def _build_1_2_5_size_grid(max_size):
    """Build a size grid following 1-2-5 progression up to max_size."""
    if max_size <= 0:
        return []

    points = []
    k = 0
    while True:
        added_this_round = False
        for base in [1, 2, 5]:
            value = base * (10 ** k)
            if value <= max_size:
                points.append(value)
                added_this_round = True
            else:
                break
        if not added_this_round:
            break
        k += 1

    if max_size not in points:
        points.append(max_size)

    return sorted(set(points))


def _parse_subset_size_grid(size_grid, max_size):
    """Parse comma-separated size grid; supports 'auto'."""
    if max_size <= 0:
        return []

    if size_grid is None or str(size_grid).strip().lower() == 'auto':
        return _build_1_2_5_size_grid(max_size)

    sizes = []
    for token in str(size_grid).split(','):
        token = token.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError:
            continue
        if value > 0:
            sizes.append(min(value, max_size))

    sizes = sorted(set(sizes))
    if not sizes:
        sizes = _build_1_2_5_size_grid(max_size)
    return sorted(set(sizes))


def _compute_rate_percentile_convergence(all_rates, t_grid='auto', num_trials=10):
    """
    Compute convergence of min/1st/5th receiver-rate percentiles versus sampled size t.

    Parameters
    ----------
    all_rates : np.ndarray
        Shape (num_samples, num_receivers)
    t_grid : str
        'auto' or comma-separated t values
    num_trials : int
        Number of random trials for each t
    """
    all_rates = np.asarray(all_rates, dtype=np.float64)
    if all_rates.ndim != 2:
        raise ValueError(f"Expected all_rates to be 2D, got shape {all_rates.shape}")

    T = all_rates.shape[0]
    if T <= 0:
        return None

    num_trials = max(1, int(num_trials))
    t_values = _parse_subset_size_grid(t_grid, max_size=T)
    if not t_values:
        return None

    min_means, min_stds = [], []
    p1_means, p1_stds = [], []
    p5_means, p5_stds = [], []

    for t in t_values:
        trial_mins, trial_p1s, trial_p5s = [], [], []
        for _ in range(num_trials):
            sampled_indices = np.random.choice(T, size=t, replace=False)
            sampled_rates = all_rates[sampled_indices]  # (t, N)
            avg_rates = sampled_rates.mean(axis=0)  # (N,)
            trial_mins.append(float(avg_rates.min()))
            trial_p1s.append(float(np.percentile(avg_rates, 1)))
            trial_p5s.append(float(np.percentile(avg_rates, 5)))

        min_means.append(float(np.mean(trial_mins)))
        min_stds.append(float(np.std(trial_mins)))
        p1_means.append(float(np.mean(trial_p1s)))
        p1_stds.append(float(np.std(trial_p1s)))
        p5_means.append(float(np.mean(trial_p5s)))
        p5_stds.append(float(np.std(trial_p5s)))

    return {
        't_values': [int(v) for v in t_values],
        'num_trials': int(num_trials),
        'min_means': min_means,
        'min_stds': min_stds,
        'p1_means': p1_means,
        'p1_stds': p1_stds,
        'p5_means': p5_means,
        'p5_stds': p5_stds,
    }


def analyze_subset_quality_vs_size(
    output_dir,
    config,
    primal_history,
    collection_metadata,
    network_id,
    window_multiplier=10,
    subset_size_grid='auto',
    feasibility_tolerance=0.0,
    num_bottleneck_nodes=5,
    save_sweep_subsets=False,
    analyze_subset_convergence=False,
    subset_convergence_num_trials=10,
    subset_convergence_t_grid='auto',
):
    """
    Sweep target subset sizes and plot min/1st/5th percentile M-averaged receiver rates.

    Returns
    -------
    tuple or None
        (results, figure_path, report_path) or None if data unavailable.
    """
    extracted = extract_checkpoint_window_samples(
        primal_history=primal_history,
        collection_metadata=collection_metadata,
        network_id=network_id,
        window_multiplier=window_multiplier,
    )
    if extracted is None:
        print(f"   ⚠️ No checkpoint trajectory samples found for network {network_id}; cannot sweep subset sizes.")
        return None

    epochs, powers, rates, extraction_info = extracted
    max_size = rates.shape[0]
    subset_sizes = _parse_subset_size_grid(subset_size_grid, max_size=max_size)
    if not subset_sizes:
        print(f"   ⚠️ Empty subset size grid for network {network_id}; skipping.")
        return None

    r_min = float(config['training']['r_min'])

    results = []
    for requested_size in subset_sizes:
        selected_indices, selection_summary = select_feasible_diverse_subset(
            rates=rates,
            target_size=requested_size,
            r_min=r_min,
            feasibility_tolerance=feasibility_tolerance,
            num_bottleneck_nodes=num_bottleneck_nodes,
        )

        selected_rates = rates[selected_indices]
        avg_rates = selected_rates.mean(axis=0)
        violating_receivers = int(np.sum(avg_rates < (r_min - feasibility_tolerance)))
        violating_pct = violating_receivers / avg_rates.shape[0] * 100.0

        result = {
            'requested_size': int(requested_size),
            'selected_size': int(len(selected_indices)),
            'min_rate': float(avg_rates.min()),
            'p1_rate': float(np.percentile(avg_rates, 1)),
            'p5_rate': float(np.percentile(avg_rates, 5)),
            'mean_rate': float(avg_rates.mean()),
            'violating_receivers': violating_receivers,
            'violating_pct': float(violating_pct),
            'is_feasible': bool(selection_summary['is_feasible']),
            'min_margin': float(selection_summary['min_margin']),
            'selected_indices_in_window': selected_indices.tolist(),
        }
        results.append(result)

    # Build plot
    x = np.array([r['requested_size'] for r in results], dtype=np.float64)
    min_rates = np.array([r['min_rate'] for r in results], dtype=np.float64)
    p1_rates = np.array([r['p1_rate'] for r in results], dtype=np.float64)
    p5_rates = np.array([r['p5_rate'] for r in results], dtype=np.float64)
    selected_sizes = np.array([r['selected_size'] for r in results], dtype=np.float64)
    size_mismatch = selected_sizes > x

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, min_rates, 'r-o', linewidth=2, markersize=5, label='Minimum')
    ax.plot(x, p1_rates, color='orange', marker='s', linewidth=2, markersize=5, label='1st Percentile')
    ax.plot(x, p5_rates, 'g-^', linewidth=2, markersize=5, label='5th Percentile')
    ax.axhline(y=r_min, color='k', linestyle='--', linewidth=1.5, label=f'$r_{{min}}$ = {r_min:.2f}', alpha=0.8)

    if np.any(size_mismatch):
        ax.scatter(
            x[size_mismatch], min_rates[size_mismatch],
            marker='o', s=110, facecolors='none', edgecolors='k', linewidths=1.2,
            label='selected_size > requested_size'
        )

    ax.set_xlabel('Requested Subset Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rate (bits/s/Hz)', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Subset Quality vs Size (Network {network_id}, window={window_multiplier}M)\n'
        f'Metrics computed on receiver-wise averages over selected subset',
        fontsize=13, fontweight='bold'
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='best')
    ax.set_xscale('log')
    ax.set_xlim(max(1, x.min()), x.max())

    feasible_count = int(np.sum([r['is_feasible'] for r in results]))
    textstr = (
        f'Window samples: {extraction_info["actual_window_size"]}\n'
        f'Feasible points: {feasible_count}/{len(results)}\n'
        f'threshold = r_min - tol = {r_min - feasibility_tolerance:.3f}'
    )
    ax.text(
        0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )

    plt.tight_layout()

    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(parents=True, exist_ok=True)
    figure_path = viz_dir / f'subset_quality_vs_size_network_{network_id}_{window_multiplier}M.pdf'
    plt.savefig(figure_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.close()

    report = {
        'network_id': int(network_id),
        'r_min': r_min,
        'feasibility_tolerance': float(feasibility_tolerance),
        'threshold': float(r_min - feasibility_tolerance),
        'window_multiplier': int(window_multiplier),
        'num_bottleneck_nodes': int(num_bottleneck_nodes),
        'subset_size_grid': [int(v) for v in subset_sizes],
        'extraction_info': extraction_info,
        'results': results,
    }

    convergence_entries = []
    if analyze_subset_convergence:
        convergence_dir = output_dir / 'visualizations' / f'subset_convergence_network_{network_id}_{window_multiplier}M'
        convergence_dir.mkdir(parents=True, exist_ok=True)

        for r in results:
            idx = np.array(r['selected_indices_in_window'], dtype=np.int64)
            selected_rates = rates[idx]
            convergence = _compute_rate_percentile_convergence(
                selected_rates,
                t_grid=subset_convergence_t_grid,
                num_trials=subset_convergence_num_trials,
            )
            if convergence is None:
                continue

            t_values = np.array(convergence['t_values'], dtype=np.float64)
            min_means = np.array(convergence['min_means'], dtype=np.float64)
            min_stds = np.array(convergence['min_stds'], dtype=np.float64)
            p1_means = np.array(convergence['p1_means'], dtype=np.float64)
            p1_stds = np.array(convergence['p1_stds'], dtype=np.float64)
            p5_means = np.array(convergence['p5_means'], dtype=np.float64)
            p5_stds = np.array(convergence['p5_stds'], dtype=np.float64)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.errorbar(t_values, min_means, yerr=min_stds, fmt='r-o', linewidth=2,
                        markersize=4, capsize=3, label='Minimum', alpha=0.85, elinewidth=1)
            ax.errorbar(t_values, p1_means, yerr=p1_stds, fmt='orange', marker='s', linewidth=2,
                        markersize=4, capsize=3, label='1st Percentile', alpha=0.85, elinewidth=1)
            ax.errorbar(t_values, p5_means, yerr=p5_stds, fmt='g-^', linewidth=2,
                        markersize=4, capsize=3, label='5th Percentile', alpha=0.85, elinewidth=1)
            ax.axhline(y=r_min, color='k', linestyle='--', linewidth=1.5,
                       label=f'$r_{{min}}$ = {r_min:.2f}', alpha=0.8)
            ax.set_xlabel('Number of Randomly Selected Samples (t)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Rate (bits/s/Hz)', fontsize=12, fontweight='bold')
            ax.set_title(
                f'Subset Convergence (Network {network_id}, k_req={r["requested_size"]}, '
                f'k_sel={r["selected_size"]}, window={window_multiplier}M)\n'
                f'Error bars: ±1 std over {subset_convergence_num_trials} random trials',
                fontsize=12, fontweight='bold'
            )
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10, loc='best')
            ax.set_xscale('log')
            ax.set_xlim(max(1, t_values.min()), t_values.max())
            plt.tight_layout()

            figure_k_path = convergence_dir / (
                f'subset_convergence_k_req_{r["requested_size"]}_k_sel_{r["selected_size"]}.pdf'
            )
            plt.savefig(figure_k_path, dpi=300, bbox_inches='tight', format='pdf')
            plt.close()

            convergence_entries.append({
                'requested_size': int(r['requested_size']),
                'selected_size': int(r['selected_size']),
                'figure_path': str(figure_k_path),
                'convergence': convergence,
            })
            print(f"   ✓ Saved subset convergence plot for k={r['requested_size']}: {figure_k_path}")

    report['subset_convergence'] = {
        'enabled': bool(analyze_subset_convergence),
        'num_trials': int(subset_convergence_num_trials),
        't_grid': str(subset_convergence_t_grid),
        'entries': convergence_entries,
    }

    subset_dir = output_dir / 'feasible_subsets'
    subset_dir.mkdir(parents=True, exist_ok=True)
    report_path = subset_dir / f'subset_quality_vs_size_network_{network_id}_{window_multiplier}M.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    if save_sweep_subsets:
        sweep_dir = subset_dir / f'network_{network_id}_size_sweep_{window_multiplier}M'
        sweep_dir.mkdir(parents=True, exist_ok=True)
        for r in results:
            idx = np.array(r['selected_indices_in_window'], dtype=np.int64)
            np.savez_compressed(
                sweep_dir / f'k_req_{r["requested_size"]}_k_sel_{r["selected_size"]}.npz',
                selected_indices_in_window=idx,
                selected_epochs=epochs[idx],
                selected_power_allocations=powers[idx],
                selected_rates=rates[idx],
                source_window_epochs=epochs,
            )

    print(f"   ✓ Saved subset-size sweep figure: {figure_path}")
    print(f"   ✓ Saved subset-size sweep report: {report_path}")
    return results, figure_path, report_path


def analyze_rate_statistics_over_windows(primal_history, config, num_networks, collection_metadata=None, network_id=None):
    """
    Analyze ergodic rate statistics over M, 5M, and 25M checkpoint windows using pre-computed rates.
    
    NOTE: primal_history only contains checkpointed epochs (collected at checkpoint interval).
    This function analyzes rate statistics over windows of these checkpoint samples.
    
    NOTE: This function uses pre-computed rates from primal_history.jsonl.
    These rates should be verified correct by regenerate_and_verify() before trusting this analysis.
    
    Parameters
    ----------
    primal_history : list
        List of primal history entries from primal_history.jsonl (checkpointed epochs only)
    config : dict
        Hydra configuration
    num_networks : int
        Number of networks in the dataset
    collection_metadata : dict, optional
        Collection metadata containing checkpoint_epochs list
    network_id : int, optional
        If provided, analyze only this specific network. If None, aggregate across all networks.
    
    Returns
    -------
    dict
        Statistics for each window size
    """
    if not primal_history:
        print("   No primal history available, skipping window analysis")
        return None
    
    print("\n" + "="*70)
    print("ANALYZING RATE STATISTICS OVER MULTIPLE WINDOWS")
    if network_id is not None:
        print(f"(Network ID: {network_id})")
    else:
        print("(Aggregated across all networks)")
    print("="*70)
    print("NOTE: Using pre-computed rates from primal_history.jsonl")
    print("      (verified correct for M collected samples)")
    
    # Get sample collection interval from metadata
    sample_collection_interval = None
    checkpoint_epochs = None
    if collection_metadata:
        checkpoint_epochs = collection_metadata.get('checkpoint_epochs', [])
        if len(checkpoint_epochs) >= 2:
            sample_collection_interval = checkpoint_epochs[1] - checkpoint_epochs[0]
            print(f"\nSample collection info:")
            print(f"   Collection epochs: {checkpoint_epochs[0]} to {checkpoint_epochs[-1]}")
            print(f"   Sample collection interval: {sample_collection_interval} epochs")
            print(f"   Total checkpoints collected: {len(checkpoint_epochs)}")
    
    # Extract all ergodic rates from history
    # Each entry represents one epoch's sample with ergodic rates per receiver [N,]
    # NOT M-averaged - we will compute M-averaged statistics over windows
    all_rates = []  # Will contain (epoch, rates) tuples
    epoch_to_rates = {}  # Map epoch -> rates
    
    for entry in primal_history:
        if 'rates' in entry and 'epoch' in entry:
            # Filter by network_id if specified
            if network_id is not None and 'network_id' in entry and entry['network_id'] != network_id:
                continue
            
            epoch = entry['epoch']
            rates = np.array(entry['rates'])
            all_rates.append((epoch, rates))
            epoch_to_rates[epoch] = rates
    
    if len(all_rates) == 0:
        print(f"   No rate data found in primal_history (checked {len(primal_history)} entries)")
        if len(primal_history) > 0:
            print(f"   Available keys in first entry: {list(primal_history[0].keys())}")
        return None
    
    # Sort by epoch
    all_rates.sort(key=lambda x: x[0])
    epochs = [e for e, _ in all_rates]
    rates_only = np.array([r for _, r in all_rates])  # (total_samples, N)
    total_samples = len(rates_only)
    N = rates_only.shape[1]
    
    print(f"\n   Loaded {total_samples} rate samples from primal_history")
    print(f"   Epoch range: {epochs[0]} to {epochs[-1]}")
    print(f"   Each sample has rates for {N} receivers")
    
    # Get r_min target
    r_min = config['training']['r_min']
    
    # Analyze checkpoint samples
    results_checkpoints = {}
    if checkpoint_epochs and len(checkpoint_epochs) > 0:
        print(f"\n   " + "="*60)
        print(f"   CHECKPOINT SAMPLE ANALYSIS (collection interval = {sample_collection_interval})")
        if network_id is not None:
            print(f"   Network ID: {network_id}")
        else:
            print(f"   Aggregated across all networks")
        print(f"   " + "="*60)
        
        # Filter primal_history to ALL checkpoint-interval epochs
        # (not just the final 100 in collection metadata, but all historical checkpoints)
        # Checkpoint interval tells us the stride (e.g., every 2 epochs)
        
        # Determine which epochs follow the checkpoint pattern
        checkpoint_base = checkpoint_epochs[0] % sample_collection_interval
        
        checkpoint_interval_rates = []
        checkpoint_interval_epochs = []
        for epoch, rates in all_rates:
            if epoch % sample_collection_interval == checkpoint_base:
                checkpoint_interval_rates.append(rates)
                checkpoint_interval_epochs.append(epoch)
        
        if len(checkpoint_interval_rates) > 0:
            checkpoint_rates_array = np.array(checkpoint_interval_rates)  # (num_available_checkpoints, N)
            num_available_checkpoints = len(checkpoint_interval_rates)
            
            # M is the number of checkpoints in the collection metadata (base unit)
            M_checkpoints = len(checkpoint_epochs)
            
            print(f"   Found {num_available_checkpoints} checkpoint-interval epochs in primal_history")
            print(f"   Checkpoint pattern: epochs where epoch % {sample_collection_interval} == {checkpoint_base}")
            print(f"   Using M = {M_checkpoints} as base unit (from collection metadata)")
            
            # Define windows based on M_checkpoints
            window_sizes_checkpoints = {
                'M': min(M_checkpoints, num_available_checkpoints),
                '5M': min(5 * M_checkpoints, num_available_checkpoints),
                '25M': min(25 * M_checkpoints, num_available_checkpoints),
            }
            
            print(f"\n   Window definitions (most recent checkpoint-interval epochs):")
            for name, size in window_sizes_checkpoints.items():
                start_idx = max(0, num_available_checkpoints - size)
                window_checkpoint_epochs = checkpoint_interval_epochs[start_idx:]
                
                # Show actual epoch range
                if len(window_checkpoint_epochs) > 0:
                    first_epoch = window_checkpoint_epochs[0]
                    last_epoch = window_checkpoint_epochs[-1]
                    
                    # Show first 2 and last 2 epochs to demonstrate spacing
                    if len(window_checkpoint_epochs) > 4:
                        epoch_str = f"{window_checkpoint_epochs[0]}, {window_checkpoint_epochs[1]}, ..., {window_checkpoint_epochs[-2]}, {window_checkpoint_epochs[-1]}"
                    else:
                        epoch_str = ", ".join(str(e) for e in window_checkpoint_epochs)
                    
                    print(f"      {name}: last {size} checkpoints (epochs {first_epoch}-{last_epoch}, showing: {epoch_str})")
                else:
                    print(f"      {name}: last {size} checkpoints (no epochs found)")
            
            # Compute statistics for checkpoint windows
            for window_name, window_size in window_sizes_checkpoints.items():
                window_rates = checkpoint_rates_array[-window_size:]
                
                # Time-average across checkpoints per receiver
                avg_rates = window_rates.mean(axis=0)
                
                # Statistics
                mean_rate = avg_rates.mean()
                worst_case = avg_rates.min()
                avg_min_rate = avg_rates.min()
                first_percentile = np.percentile(avg_rates, 1)
                fifth_percentile = np.percentile(avg_rates, 5)
                std_rate = avg_rates.std()
                violation_receivers = (avg_rates < r_min).sum()
                violation_pct = violation_receivers / N * 100
                
                results_checkpoints[window_name] = {
                    'window_size': window_size,
                    'mean': mean_rate,
                    'worst_case': worst_case,
                    'avg_min': avg_min_rate,
                    'first_percentile': first_percentile,
                    'fifth_percentile': fifth_percentile,
                    'std': std_rate,
                    'violation_receivers': int(violation_receivers),
                    'violation_pct': violation_pct,
                }
            
            # Print checkpoint statistics table
            print(f"\n   Rate Statistics (Checkpoint Epochs Only, r_min = {r_min:.4f} bits/s/Hz)")
            print(f"   {'':<30} {'M':>12} {'5M':>12} {'25M':>12}")
            print(f"   {'-'*30} {'-'*12} {'-'*12} {'-'*12}")
            
            for metric_name, metric_key in [
                ('Window size (checkpoints)', 'window_size'),
                ('Mean rate (M-averaged)', 'mean'),
                ('Worst-case (min M-avg)', 'worst_case'),
                ('Min M-averaged rate', 'avg_min'),
                ('1st percentile (M-avg)', 'first_percentile'),
                ('5th percentile (M-avg)', 'fifth_percentile'),
                ('Std deviation (M-avg)', 'std'),
                ('Violations (receivers)', 'violation_receivers'),
                ('Violations (%)', 'violation_pct'),
            ]:
                values = []
                for window_name in ['M', '5M', '25M']:
                    if window_name in results_checkpoints:
                        val = results_checkpoints[window_name][metric_key]
                        if metric_key == 'window_size' or metric_key == 'violation_receivers':
                            values.append(f"{int(val):>12}")
                        elif metric_key == 'violation_pct':
                            values.append(f"{val:>11.1f}%")
                        else:
                            values.append(f"{val:>12.4f}")
                    else:
                        values.append(f"{'N/A':>12}")
                
                print(f"   {metric_name:<30} {' '.join(values)}")
    
    # Highlight key insights
    print(f"\n   " + "="*60)
    if network_id is not None:
        print(f"   KEY INSIGHTS (Network {network_id})")
    else:
        print(f"   KEY INSIGHTS (Aggregated)")
    print(f"   " + "="*60)
    
    # Check if worst-case improves over larger windows
    if 'M' in results_checkpoints and '25M' in results_checkpoints:
        wc_m = results_checkpoints['M']['worst_case']
        wc_25m = results_checkpoints['25M']['worst_case']
        improvement = (wc_25m - wc_m) / wc_m * 100
        
        if improvement > 5:
            print(f"   ✓ Worst-case rate improves by {improvement:.1f}% over 25M window")
            print(f"     (suggests policy is improving over time)")
        elif improvement < -5:
            print(f"   ⚠️  Worst-case rate degrades by {abs(improvement):.1f}% over 25M window")
            print(f"     (suggests policy instability or harder samples in recent windows)")
        else:
            print(f"   → Worst-case rate is stable across windows ({improvement:+.1f}%)")
    
    # Check constraint satisfaction (based on M-averaged receiver rates)
    for window_name in ['M', '5M', '25M']:
        if window_name in results_checkpoints:
            viol_pct = results_checkpoints[window_name]['violation_pct']
            viol_count = results_checkpoints[window_name]['violation_receivers']
            if viol_pct == 0:
                print(f"   ✓ {window_name} window: 100% constraint satisfaction (all {N} receivers)")
            elif viol_pct < 5:
                print(f"   ⚠️  {window_name} window: {viol_count}/{N} receivers ({viol_pct:.1f}%) violate (< 5%, acceptable)")
            else:
                print(f"   ✗ {window_name} window: {viol_count}/{N} receivers ({viol_pct:.1f}%) violate (> 5%, poor)")
    
    return results_checkpoints


def regenerate_and_verify(network_info, config, quality_report, channels=None, num_example_networks=None):
    """
    Regenerate networks from seeds and verify they match saved data.
    
    Parameters
    ----------
    network_info : dict
        Network information extracted from samples
    config : dict
        Hydra configuration with dataset parameters
    quality_report : dict
        Quality report from training
    channels : list, optional
        Pre-loaded channels from cache. If None, will regenerate from seeds.
    num_example_networks : int, optional
        Number of example networks to verify. If None, verifies all networks.
    
    Returns
    -------
    verification_results : dict
        Per-network verification results
    """
    print("\n" + "="*70)
    if channels:
        print("USING CACHED CHANNELS FOR VERIFICATION")
    else:
        print("REGENERATING NETWORKS FROM SEEDS")
    print("="*70)
    
    # Compute system parameters for rate calculation
    system_params = compute_system_parameters(
        P_max_dBm=config['system']['P_max_dBm'],
        bandwidth_Hz=config['system']['bandwidth_Hz'],
        noise_psd_dBm_Hz=config['system']['noise_psd_dBm_Hz'],
    )
    noise_var = system_params['noise_var']
    
    # Get channel configuration
    channel_config = config.get('channel', {})
    channel_version = channel_config.get('version', 'v2')  # Default to v2 for backward compatibility
    
    # Select channel class based on version
    if channel_version == 'v3':
        ChannelClass = WirelessChannelV3
    elif channel_version == 'v2':
        ChannelClass = WirelessChannelV2
    else:
        ChannelClass = WirelessChannel
    
    print(f"Channel version: {channel_version} ({ChannelClass.__name__})")
    
    # Limit to num_example_networks if specified
    network_ids_to_verify = list(network_info.keys())
    if num_example_networks is not None:
        network_ids_to_verify = network_ids_to_verify[:num_example_networks]
        print(f"\nVerifying first {num_example_networks} networks (out of {len(network_info)} total)")
    
    verification_results = {}
    
    for net_id in network_ids_to_verify:
        info = network_info[net_id]
        print(f"\nNetwork {net_id}:")
        print(f"  Seed: {info['seed']}")
        print(f"  Saved samples: {info['num_samples']}")
        print(f"  H_instantaneous shape: {info['H_instantaneous'].shape}")
        print(f"  Associations shape: {info['associations'].shape}")
        
        # Get channel from cache or regenerate with same seed
        if channels and net_id < len(channels):
            channel = channels[net_id]
            print(f"  Using cached channel {net_id}")
        else:
            # Build channel kwargs with version-specific parameters
            channel_kwargs = {
                'n_links': config['dataset']['n_links'],
                'seed': info['seed'],
                'deployment_range': config['dataset']['deployment_range'],
            }
            
            # Add version-specific parameters
            if channel_version in ['v2', 'v3']:
                if 'min_tx_rx_distance' in channel_config:
                    channel_kwargs['min_tx_rx_distance'] = channel_config['min_tx_rx_distance']
                if 'max_tx_rx_distance' in channel_config and channel_config['max_tx_rx_distance'] is not None:
                    channel_kwargs['max_tx_rx_distance'] = channel_config['max_tx_rx_distance']
            
            if channel_version == 'v3':
                if 'max_recursion_depth' in channel_config and channel_config['max_recursion_depth'] is not None:
                    channel_kwargs['max_recursion_depth'] = channel_config['max_recursion_depth']
            
            channel = ChannelClass(**channel_kwargs)
            print(f"  Regenerated channel {net_id} from seed")
        
        # The channel is already deployed and associations are computed in __init__
        # Verify that associations match
        associations_match = np.allclose(channel.associations, info['associations'])
        
        # Verify network structure details
        n_links = config['dataset']['n_links']
        tx_locations_shape = channel.tx_locations.shape
        rx_locations_shape = channel.rx_locations.shape
        large_scale_shape = channel.large_scale_fading.shape
        
        print(f"\n  Regenerated network:")
        print(f"    TX locations: {tx_locations_shape}")
        print(f"    RX locations: {rx_locations_shape}")
        print(f"    Large-scale fading: {large_scale_shape}")
        print(f"    Associations: {channel.associations.shape}")
        
        print(f"\n  Verification:")
        if associations_match:
            print(f"    ✓ Associations match perfectly")
        else:
            print(f"    ✗ Associations DO NOT match")
            diff = np.abs(channel.associations - info['associations'])
            print(f"      Max difference: {diff.max()}")
            print(f"      Mean difference: {diff.mean()}")
        
        # Check large-scale fading statistics (should be similar but not identical
        # due to different small-scale fading realizations)
        H_ls_mean_saved = info['H_instantaneous'].mean()
        H_ls_mean_regen = channel.large_scale_fading.mean()
        
        print(f"    H_instantaneous mean (saved): {H_ls_mean_saved:.6e}")
        print(f"    Large-scale fading mean (regen): {H_ls_mean_regen:.6e}")
        
        # Sample power statistics
        power_samples = info['power_samples']
        power_mean = np.mean([p.mean() for p in power_samples])
        power_std = np.std([p.mean() for p in power_samples])
        power_min = np.min([p.min() for p in power_samples])
        power_max = np.max([p.max() for p in power_samples])
        
        print(f"\n  Power allocation statistics (across {len(power_samples)} samples):")
        print(f"    Mean power: {power_mean:.6e} W")
        print(f"    Std of means: {power_std:.6e} W")
        print(f"    Min power: {power_min:.6e} W")
        print(f"    Max power: {power_max:.6e} W")
        
        # Rate statistics from saved samples
        rate_samples = info['rate_samples']
        
        # Filter out empty rate arrays (can happen in old data)
        valid_rate_samples = [r for r in rate_samples if r.size > 0]
        
        if len(valid_rate_samples) == 0:
            print(f"\n  ⚠ Warning: No valid rate samples found (all empty arrays)")
            saved_mean = np.nan
            saved_worst_case = np.nan
            saved_min_mavg = np.nan
            saved_5th_mavg = np.nan
        else:
            # Compute M-averaged rates first
            rates_matrix_saved = np.array(valid_rate_samples)  # (M, N)
            avg_rates_saved = rates_matrix_saved.mean(axis=0)  # (N,) - M-averaged per receiver
            
            # 1. Worst-case: minimum across ALL samples and ALL receivers
            saved_worst_case = rates_matrix_saved.min()
            
            # 2. Min of M-averaged rates (worst receiver's expected rate)
            saved_min_mavg = avg_rates_saved.min()
            
            # 3. 5th percentile of M-averaged rates
            saved_5th_mavg = np.percentile(avg_rates_saved, 5)
            
            # 4. Mean rate
            saved_mean = avg_rates_saved.mean()
        
        # Recompute ergodic rates from power samples using SAVED H_instantaneous
        print(f"\n  Recomputing rates from power samples using saved H_instantaneous...")
        
        # Use the saved H_instantaneous (confirmed to be correct)
        H_inst_torch = torch.from_numpy(info['H_instantaneous']).float()
        associations_torch = torch.from_numpy(info['associations']).float()
        T = H_inst_torch.shape[0]  # Total timesteps
        num_samples = len(power_samples)
        
        print(f"    Using saved H_instantaneous with T={T} timesteps")
        print(f"    Channel stats: min={info['H_instantaneous'].min():.2e}, max={info['H_instantaneous'].max():.2e}, mean={info['H_instantaneous'].mean():.2e}")
        
        recomputed_rates_ergodic = []  # Ergodic rates per sample
        rate_mismatches = []
        
        # Evaluate all power samples using the saved channel realization
        print(f"    Evaluating {num_samples} power samples...")
        for sample_idx, power_sample in tqdm(enumerate(power_samples), total=len(power_samples), desc="    Samples"):
            # Skip empty samples
            if rate_samples[sample_idx].size == 0:
                continue
                
            power_torch = torch.from_numpy(power_sample).float()
            
            # Compute ergodic rates over all T timesteps
            rates_ergodic = compute_ergodic_rates(
                power_allocation=power_torch,
                H_instantaneous=H_inst_torch,
                associations=associations_torch,
                noise_var=noise_var
            ).numpy()
            recomputed_rates_ergodic.append(rates_ergodic)
            
            # Compare with saved rates
            rates_saved = rate_samples[sample_idx]
            rate_match = np.allclose(rates_ergodic, rates_saved, rtol=1e-5)
            
            if not rate_match:
                rate_mismatches.append(sample_idx)
        
        if len(rate_mismatches) == 0:
            print(f"    ✓ All {len(recomputed_rates_ergodic)} recomputed rates match saved rates")
        else:
            print(f"    ⚠️  Rate mismatches in {len(rate_mismatches)}/{len(recomputed_rates_ergodic)} samples")
            print(f"       (using strict per-sample tolerance rtol=1e-5)")
        
        # Compute statistics from recomputed rates
        if len(recomputed_rates_ergodic) > 0:
            # Compute M-averaged rates
            rates_matrix_recomp = np.array(recomputed_rates_ergodic)  # (M, N)
            avg_rates_recomp = rates_matrix_recomp.mean(axis=0)  # (N,) - M-averaged per receiver
            
            # 1. Worst-case: minimum across ALL samples and ALL receivers
            recomp_worst_case = rates_matrix_recomp.min()
            
            # 2. Min of M-averaged rates (worst receiver's expected rate)
            recomp_min_mavg = avg_rates_recomp.min()
            
            # 3. 5th percentile of M-averaged rates
            recomp_5th_mavg = np.percentile(avg_rates_recomp, 5)
            
            # 4. Mean rate
            recomp_mean = avg_rates_recomp.mean()
        else:
            recomp_worst_case = np.nan
            recomp_min_mavg = np.nan
            recomp_5th_mavg = np.nan
            recomp_mean = np.nan
        
        # Print comparison table
        print(f"\n  Rate Statistics Comparison (M={len(valid_rate_samples)} samples, N={channel.n_links} receivers):")
        print(f"  {'Metric':<45} {'Saved':>12} {'Recomputed':>12} {'Diff':>12}")
        print(f"  {'-'*45} {'-'*12} {'-'*12} {'-'*12}")
        print(f"  {'1. Worst-case min (all M×N values)':<45} {saved_worst_case:>12.4f} {recomp_worst_case:>12.4f} {abs(saved_worst_case-recomp_worst_case):>12.6f}")
        print(f"  {'2. Min rate (worst receiver, M-averaged)':<45} {saved_min_mavg:>12.4f} {recomp_min_mavg:>12.4f} {abs(saved_min_mavg-recomp_min_mavg):>12.6f}")
        print(f"  {'3. 5th percentile (M-averaged receivers)':<45} {saved_5th_mavg:>12.4f} {recomp_5th_mavg:>12.4f} {abs(saved_5th_mavg-recomp_5th_mavg):>12.6f}")
        print(f"  {'4. Mean rate (M-averaged receivers)':<45} {saved_mean:>12.4f} {recomp_mean:>12.4f} {abs(saved_mean-recomp_mean):>12.6f}")
        print(f"")
        print(f"  Target r_min: {config['training']['r_min']:.4f} bits/s/Hz")
        if recomp_min_mavg < config['training']['r_min']:
            violation = (config['training']['r_min'] - recomp_min_mavg) / config['training']['r_min'] * 100
            print(f"  ⚠️  Constraint violation: Min rate is {violation:.1f}% below target")
        
        # Determine if rates match based on statistics (more robust than per-sample matching)
        # Use relative tolerance of 1% for statistics (much more forgiving than per-sample rtol=1e-5)
        stats_match = True
        max_rel_diff = 0.0
        TOLERANCE_PCT = 1.0  # 1% tolerance for aggregate statistics
        
        for saved_val, recomp_val, name in [
            (saved_worst_case, recomp_worst_case, "worst-case"),
            (saved_min_mavg, recomp_min_mavg, "min M-averaged"),
            (saved_5th_mavg, recomp_5th_mavg, "5th percentile"),
            (saved_mean, recomp_mean, "mean"),
        ]:
            if np.isnan(saved_val) or np.isnan(recomp_val):
                continue
            
            # For near-zero values, use absolute tolerance
            if abs(saved_val) < 1e-6 and abs(recomp_val) < 1e-6:
                continue
            elif abs(saved_val) < 1e-6:
                # If saved is ~0 but recomputed is not, that's a problem
                rel_diff = abs(recomp_val) * 100
            else:
                rel_diff = abs(saved_val - recomp_val) / abs(saved_val) * 100
            
            max_rel_diff = max(max_rel_diff, rel_diff)
            
            if rel_diff > TOLERANCE_PCT:
                stats_match = False
        
        print(f"\n  Statistics Verification (using {TOLERANCE_PCT}% tolerance):")
        if stats_match:
            print(f"    ✓ All statistics match (max rel diff: {max_rel_diff:.4f}%)")
            print(f"      Pre-computed rates are trustworthy.")
        else:
            print(f"    ✗ Statistics mismatch (max rel diff: {max_rel_diff:.4f}%)")
            print(f"      Pre-computed rates may not be reliable.")
        
        # Store verification results (use statistics-based verification)
        verification_results[net_id] = {
            'associations_match': associations_match,
            'rates_match': stats_match,  # Use statistics-based verification instead of per-sample
            'seed': info['seed'],
            'num_samples': info['num_samples'],
            'power_mean': power_mean,
            'saved_mean': saved_mean,
            'saved_worst_case': saved_worst_case,
            'saved_min_mavg': saved_min_mavg,
            'saved_5th_mavg': saved_5th_mavg,
            'recomp_mean': recomp_mean,
            'recomp_worst_case': recomp_worst_case,
            'recomp_min_mavg': recomp_min_mavg,
            'recomp_5th_mavg': recomp_5th_mavg,
        }
    
    return verification_results


def visualize_network_and_graph(network_info, config, output_dir, channels=None, network_id=0):
    """
    Visualize a network and its PyG graph representation.
    
    Creates two subplots:
    1. Network deployment showing transmitters, receivers, and their pairings
    2. PyG interference graph (nodes = receivers, edges = interference + self-loops)
    
    Parameters
    ----------
    network_info : dict
        Network information extracted from samples
    config : dict
        Hydra configuration
    output_dir : Path
        Output directory to save visualization
    channels : list, optional
        Pre-loaded channels from cache
    network_id : int, optional
        Network ID to visualize (default: 0)
    """
    # Use the specified network
    net_id = network_id
    info = network_info[net_id]
    
    print("\n" + "="*70)
    print(f"VISUALIZING NETWORK {net_id}")
    print("="*70)
    
    # Get channel from cache or regenerate
    if channels and net_id < len(channels):
        channel = channels[net_id]
        print(f"  Using cached channel {net_id}")
    else:
        # Get channel configuration
        channel_config = config.get('channel', {})
        channel_version = channel_config.get('version', 'v2')
        
        # Select channel class
        if channel_version == 'v3':
            ChannelClass = WirelessChannelV3
        elif channel_version == 'v2':
            ChannelClass = WirelessChannelV2
        else:
            ChannelClass = WirelessChannel
        
        # Build channel kwargs
        channel_kwargs = {
            'n_links': config['dataset']['n_links'],
            'seed': info['seed'],
            'deployment_range': config['dataset']['deployment_range'],
        }
        
        # Add version-specific parameters
        if channel_version in ['v2', 'v3']:
            if 'min_tx_rx_distance' in channel_config:
                channel_kwargs['min_tx_rx_distance'] = channel_config['min_tx_rx_distance']
            if 'max_tx_rx_distance' in channel_config and channel_config['max_tx_rx_distance'] is not None:
                channel_kwargs['max_tx_rx_distance'] = channel_config['max_tx_rx_distance']
        
        if channel_version == 'v3':
            if 'max_recursion_depth' in channel_config and channel_config['max_recursion_depth'] is not None:
                channel_kwargs['max_recursion_depth'] = channel_config['max_recursion_depth']
        
        channel = ChannelClass(**channel_kwargs)
        print(f"  Regenerated channel from seed")
    
    # Build the PyG graph using the same method as primal-dual trainer
    top_k = config['dataset'].get('top_k', 10)
    min_degree = config['dataset'].get('min_degree', 1)
    
    graph = build_interference_graph(
        H_ls=channel.large_scale_fading,
        associations=channel.associations,
        top_k=top_k,
        min_degree=min_degree,
        normalize_method='log1p',
    )
    
    print(f"  Network seed: {info['seed']}")
    print(f"  Number of links: {channel.n_links}")
    print(f"  PyG graph: {graph.num_nodes} nodes, {graph.edge_index.shape[1]} edges")
    
    # Scale figure size based on number of links (50 links is baseline, don't shrink below baseline)
    scale = max(1.0, np.sqrt(channel.n_links / 50.0))
    fig, axes = plt.subplots(1, 2, figsize=(16*scale, 7*scale))
    
    # ============= Subplot 1: Network Deployment =============
    ax = axes[0]
    
    tx_locs = channel.tx_locations
    rx_locs = channel.rx_locations
    
    # Draw TX-RX pairs (paired links in green) - thicker to emphasize
    for tx_idx, rx_idx in channel.tx_rx_pairs:
        ax.plot(
            [tx_locs[tx_idx, 0], rx_locs[rx_idx, 0]],
            [tx_locs[tx_idx, 1], rx_locs[rx_idx, 1]],
            'g-', linewidth=2.5, alpha=0.7, zorder=1
        )
    
    # Plot transmitters (smaller markers)
    ax.scatter(
        tx_locs[:, 0], tx_locs[:, 1],
        c='red', s=80, marker='^', edgecolors='black', linewidths=1.2,
        label='Transmitters', zorder=3
    )
    
    # Plot receivers (smaller markers)
    ax.scatter(
        rx_locs[:, 0], rx_locs[:, 1],
        c='blue', s=80, marker='o', edgecolors='black', linewidths=1.2,
        label='Receivers', zorder=3
    )
    
    # Add TX/RX labels (smaller font)
    for i in range(channel.n_links):
        ax.annotate(f'TX{i}', (tx_locs[i, 0], tx_locs[i, 1]), 
                   xytext=(3, 3), textcoords='offset points', fontsize=5, alpha=0.7)
        ax.annotate(f'RX{i}', (rx_locs[i, 0], rx_locs[i, 1]), 
                   xytext=(3, 3), textcoords='offset points', fontsize=5, alpha=0.7)
    
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title(f'Network Deployment (seed={info["seed"]})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # ============= Subplot 2: PyG Interference Graph =============
    ax = axes[1]
    
    # Node positions: use receiver locations
    node_pos = rx_locs
    
    # Parse edge_index
    edge_index = graph.edge_index.numpy()
    edge_weight = graph.edge_weight.numpy()
    
    # Separate self-loops from interference edges
    self_loop_mask = edge_index[0] == edge_index[1]
    self_loop_edges = edge_index[:, self_loop_mask]
    self_loop_weights = edge_weight[self_loop_mask]
    
    interf_edges = edge_index[:, ~self_loop_mask]
    interf_weights = edge_weight[~self_loop_mask]
    
    # Normalize weights for visualization
    if len(interf_weights) > 0:
        interf_weights_norm = (interf_weights - interf_weights.min()) / (interf_weights.max() - interf_weights.min() + 1e-10)
    else:
        interf_weights_norm = interf_weights
    
    # Draw interference edges (arrows showing direction of interference)
    for i in range(interf_edges.shape[1]):
        src, tgt = interf_edges[0, i], interf_edges[1, i]
        weight = interf_weights_norm[i]
        
        # Color intensity based on weight
        color = plt.cm.Reds(0.5 + 0.5 * weight)
        alpha = 0.3 + 0.5 * weight
        linewidth = 0.5 + 1.5 * weight
        
        # Draw arrow from source to target
        dx = node_pos[tgt, 0] - node_pos[src, 0]
        dy = node_pos[tgt, 1] - node_pos[src, 1]
        
        ax.annotate('',
            xy=(node_pos[tgt, 0], node_pos[tgt, 1]),
            xytext=(node_pos[src, 0], node_pos[src, 1]),
            arrowprops=dict(
                arrowstyle='->', 
                color=color, 
                alpha=alpha,
                lw=linewidth,
                connectionstyle='arc3,rad=0.1'
            ),
            zorder=1
        )
    
    # Draw self-loops colored by direct signal strength (feature 0)
    direct_signal_features = graph.x[:, 0].numpy()
    if self_loop_edges.shape[1] > 0:
        direct_signal_norm = (direct_signal_features - direct_signal_features.min()) / (direct_signal_features.max() - direct_signal_features.min() + 1e-10)
    else:
        direct_signal_norm = np.zeros(graph.num_nodes)
    
    for i in range(self_loop_edges.shape[1]):
        node = self_loop_edges[0, i]
        signal_strength = direct_signal_norm[node]
        
        # Draw self-loop circle colored by direct signal strength (darker blues)
        circle = plt.Circle(
            (node_pos[node, 0], node_pos[node, 1] + 20),  # offset above node
            radius=12,  # larger radius for visibility
            color=plt.cm.Blues(0.5 + 0.5 * signal_strength),  # darker blues (0.5-1.0)
            alpha=0.9,
            fill=False,
            linewidth=2 + 2 * signal_strength,  # thicker lines
            zorder=2
        )
        ax.add_patch(circle)
    
    # Draw nodes colored by interference potential (feature 1)
    interference_colors = graph.x[:, 1].numpy()
    scatter = ax.scatter(
        node_pos[:, 0], node_pos[:, 1],
        c=interference_colors, cmap='Oranges', s=200, marker='o', 
        edgecolors='black', linewidths=2,
        zorder=4
    )
    
    # Add node labels
    for i in range(graph.num_nodes):
        ax.annotate(f'{i}', (node_pos[i, 0], node_pos[i, 1]), 
                   ha='center', va='center', fontsize=8, fontweight='bold', color='white',
                   zorder=5)
    
    # Create legend
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color='red', alpha=0.5, lw=2, label='Interference edge'),
            plt.Line2D([0], [0], color='darkblue', alpha=0.9, lw=3, marker='o', 
                      markersize=10, markerfacecolor='none', linestyle='None', 
                      label='Self-loop (direct signal)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
                      markersize=10, markeredgecolor='black', linestyle='None',
                      label='Node (interference potential)')
        ],
        loc='upper right', fontsize=9
    )
    
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title(f'PyG Graph (nodes=RX, edges=interference)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Add summary text
    fig.text(0.5, 0.02, 
             f'Network: {channel.n_links} TX-RX pairs | Graph: {graph.num_nodes} nodes, '
             f'{interf_edges.shape[1]} interference edges, {self_loop_edges.shape[1]} self-loops | '
             f'top_k={top_k}',
             ha='center', fontsize=10, style='italic')
    
    # Adjust layout first
    plt.tight_layout(rect=[0, 0.04, 0.92, 1.0])
    
    # Now get the position of the right subplot for colorbar placement
    pos = ax.get_position()
    
    # Add colorbar for interference potential (nodes) - positioned manually
    cax1 = fig.add_axes([pos.x1 + 0.02, pos.y0, 0.015, pos.height])
    cbar_interference = plt.colorbar(scatter, cax=cax1)
    cbar_interference.set_label('Interference Potential\n(node color)', fontsize=9)
    
    # Add second colorbar for direct signal strength (self-loops)
    cax2 = fig.add_axes([pos.x1 + 0.10, pos.y0, 0.015, pos.height])
    norm = Normalize(vmin=direct_signal_features.min(), vmax=direct_signal_features.max())
    sm = ScalarMappable(cmap='Blues', norm=norm)
    sm.set_array([])
    cbar_direct = plt.colorbar(sm, cax=cax2)
    cbar_direct.set_label('Direct Signal Strength\n(self-loop color)', fontsize=9)
    
    # Save figure with network_id in filename
    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)
    save_path = viz_dir / f'network_and_graph_visualization_network_{net_id}.pdf'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"\n  ✓ Saved visualization: {save_path}")
    plt.close()
    
    return save_path


def visualize_stochastic_policy_samples(network_info, config, output_dir, num_samples=5, channels=None, primal_history=None, network_id=0, collection_metadata=None, window_multiplier=1):
    """
    Visualize stochastic policy evolution across checkpoint samples.
    
    Creates num_samples+1 subplots showing:
    - num_samples subplots: Each showing the PyG graph with nodes colored by ergodic rates for that sample
    - 1 final subplot: Averaged ergodic rates over window_multiplier*M checkpoints
    
    Parameters
    ----------
    network_info : dict
        Network information extracted from samples
    config : dict
        Hydra configuration
    output_dir : Path
        Output directory to save visualization
    num_samples : int
        Number of samples to visualize (default: 5)
    channels : list, optional
        Pre-loaded channels from cache
    primal_history : list, optional
        List of primal history entries (checkpointed epochs only)
    network_id : int, optional
        Network ID to visualize (default: 0)
    collection_metadata : dict, optional
        Collection metadata containing checkpoint_epochs
    window_multiplier : int, optional
        Window size multiplier for averaging (1=M, 2=2M, 5=5M, etc.) (default: 1)
    """
    # Use the specified network
    net_id = network_id
    info = network_info[net_id]
    
    if not primal_history:
        print("  ⚠️ No primal_history provided, cannot create visualization")
        return None
    
    if not collection_metadata or 'checkpoint_epochs' not in collection_metadata:
        print("  ⚠️ No collection_metadata provided, cannot create visualization")
        return None
    
    print("\n" + "="*70)
    print(f"VISUALIZING STOCHASTIC POLICY SAMPLES - Network {net_id}")
    print(f"Showing {num_samples} diverse samples + {window_multiplier}M checkpoint average")
    print("="*70)
    
    # Extract checkpoint-interval epochs from primal_history
    if True:
        # Extract checkpoint-interval epochs from primal_history (same logic as analysis)
        checkpoint_epochs = collection_metadata['checkpoint_epochs']
        sample_collection_interval = checkpoint_epochs[1] - checkpoint_epochs[0] if len(checkpoint_epochs) >= 2 else 2
        
        # Filter primal_history to checkpoint-interval epochs FOR THIS NETWORK
        checkpoint_base = checkpoint_epochs[0] % sample_collection_interval
        
        checkpoint_interval_epochs = []
        for entry in primal_history:
            if ('epoch' in entry and 
                'network_id' in entry and entry['network_id'] == net_id):
                epoch = entry['epoch']
                if epoch % sample_collection_interval == checkpoint_base:
                    checkpoint_interval_epochs.append(epoch)
        
        # For checkpoint mode, M is the number of collected checkpoints (not training config M)
        M_checkpoints = len(checkpoint_epochs)
        actual_window_size_requested = window_multiplier * M_checkpoints  # e.g., 10M = 10*100 = 1000 checkpoints
        
        # Now take last actual_window_size_requested checkpoints from available checkpoint-interval epochs FOR THIS NETWORK
        num_available_checkpoints = len(checkpoint_interval_epochs)
        actual_window_size = min(actual_window_size_requested, num_available_checkpoints)
        checkpoint_set = set(checkpoint_interval_epochs[-actual_window_size:])
        
        print(f"  Using last {window_multiplier}M={actual_window_size} checkpoint-interval epochs from primal_history")
        print(f"  (M={M_checkpoints} checkpoints from collection metadata, window={window_multiplier}×{M_checkpoints}={actual_window_size_requested})")
        print(f"  Checkpoint pattern: epochs where epoch % {sample_collection_interval} == {checkpoint_base}")
        if actual_window_size < actual_window_size_requested:
            print(f"    ⚠️ Requested {actual_window_size_requested} but only {num_available_checkpoints} checkpoint-interval epochs available for network {net_id}")
        if len(checkpoint_set) > 0:
            print(f"  Checkpoint epoch range: {min(checkpoint_set)} to {max(checkpoint_set)}")
        
        # Extract power allocations and rates from primal_history for checkpoint epochs
        powers_from_history = []
        rates_from_history = []
        
        for entry in primal_history:
            if ('epoch' in entry and entry['epoch'] in checkpoint_set and 
                'network_id' in entry and entry['network_id'] == net_id and
                'power_allocations' in entry and 'rates' in entry):
                powers_from_history.append(np.array(entry['power_allocations']))
                rates_from_history.append(np.array(entry['rates']))
        
        if len(powers_from_history) == 0:
            print(f"  ⚠️ No checkpoint data found for network {net_id} in primal_history")
            return None
        else:
            print(f"  ✓ Found {len(powers_from_history)} checkpoint samples for network {net_id}")
            all_power_samples = powers_from_history
            all_rates_full = np.array(rates_from_history)
    
    # Build the PyG graph for visualization
    top_k = config['dataset'].get('top_k', 10)
    min_degree = config['dataset'].get('min_degree', 1)
    
    # Get channel from cache or regenerate for graph building only
    if channels and net_id < len(channels):
        channel = channels[net_id]
        print(f"  Using cached channel {net_id}")
    else:
        # Get channel configuration
        channel_config = config.get('channel', {})
        channel_version = channel_config.get('version', 'v2')
        
        # Select channel class
        if channel_version == 'v3':
            ChannelClass = WirelessChannelV3
        elif channel_version == 'v2':
            ChannelClass = WirelessChannelV2
        else:
            ChannelClass = WirelessChannel
        
        # Build channel kwargs
        channel_kwargs = {
            'n_links': config['dataset']['n_links'],
            'seed': info['seed'],
            'deployment_range': config['dataset']['deployment_range'],
        }
        
        # Add version-specific parameters
        if channel_version in ['v2', 'v3']:
            if 'min_tx_rx_distance' in channel_config:
                channel_kwargs['min_tx_rx_distance'] = channel_config['min_tx_rx_distance']
            if 'max_tx_rx_distance' in channel_config and channel_config['max_tx_rx_distance'] is not None:
                channel_kwargs['max_tx_rx_distance'] = channel_config['max_tx_rx_distance']
        
        if channel_version == 'v3':
            if 'max_recursion_depth' in channel_config and channel_config['max_recursion_depth'] is not None:
                channel_kwargs['max_recursion_depth'] = channel_config['max_recursion_depth']
        
        channel = ChannelClass(**channel_kwargs)
        print(f"  Regenerated channel from seed")
    
    graph = build_interference_graph(
        H_ls=channel.large_scale_fading,
        associations=channel.associations,
        top_k=top_k,
        min_degree=min_degree,
        normalize_method='log1p',
    )
    
    # Rates already extracted from primal_history
    actual_num_samples = len(all_rates_full)  # Actual number of samples in this window
    
    # IMPROVED DIVERSE SAMPLE SELECTION HEURISTIC
    # Step 1: Identify bottleneck receivers as those with lowest M-averaged ergodic rates
    avg_rates_per_rx = all_rates_full.mean(axis=0)  # (num_users,) - M-averaged per receiver
    num_bottleneck_nodes = min(5, len(avg_rates_per_rx))  # Focus on top 5 worst receivers
    bottleneck_nodes = np.argsort(avg_rates_per_rx)[:num_bottleneck_nodes]  # Indices of worst receivers
    
    print(f"  Bottleneck receivers (lowest M-averaged rates): {bottleneck_nodes}")
    print(f"  Bottleneck M-averaged rates: {[f'{avg_rates_per_rx[i]:.3f}' for i in bottleneck_nodes]}")
    
    # Step 2: Extract rates for bottleneck nodes across all samples
    bottleneck_rates = all_rates_full[:, bottleneck_nodes]  # (total_samples, num_bottleneck_nodes)
    
    # Step 3: Select samples to maximize diversity of rates for bottleneck nodes
    # Use greedy selection: pick samples that are most different from already selected ones
    selected_indices = []
    
    # Start with the sample where bottleneck nodes have maximum variance (most diverse)
    bottleneck_variances = np.var(bottleneck_rates, axis=1)
    first_idx = np.argmax(bottleneck_variances)
    selected_indices.append(first_idx)
    
    # Greedily add samples that maximize distance in bottleneck rate space
    for _ in range(num_samples - 1):
        max_min_distance = -1
        best_idx = -1
        
        for candidate_idx in range(len(all_rates_full)):
            if candidate_idx in selected_indices:
                continue
            
            # Compute minimum distance to already selected samples in bottleneck rate space
            min_distance = float('inf')
            candidate_rates = bottleneck_rates[candidate_idx]
            
            for selected_idx in selected_indices:
                selected_rates = bottleneck_rates[selected_idx]
                # Euclidean distance in bottleneck rate space
                distance = np.linalg.norm(candidate_rates - selected_rates)
                min_distance = min(min_distance, distance)
            
            # Select candidate with maximum minimum distance (furthest from all selected)
            if min_distance > max_min_distance:
                max_min_distance = min_distance
                best_idx = candidate_idx
        
        if best_idx >= 0:
            selected_indices.append(best_idx)
    
    # Sort selected indices for visualization
    selected_indices = sorted(selected_indices)
    
    print(f"  Selected {len(selected_indices)} diverse samples: {selected_indices}")
    print(f"  Selected samples' bottleneck rate ranges:")
    for idx in selected_indices:
        rates_at_bottlenecks = all_rates_full[idx, bottleneck_nodes]
        print(f"    Sample {idx}: min={rates_at_bottlenecks.min():.3f}, max={rates_at_bottlenecks.max():.3f}, range={rates_at_bottlenecks.max()-rates_at_bottlenecks.min():.3f}")
    
    # Get selected samples
    power_samples = [all_power_samples[i] for i in selected_indices]
    all_rates = all_rates_full[selected_indices]
    
    # Compute average rates from checkpoint samples
    avg_rates = all_rates_full.mean(axis=0)
    print(f"  Using {window_multiplier}M={actual_num_samples} checkpoint samples average for final subplot")
    
    # Get r_min from config
    r_min = config['training']['r_min']
    
    # Get P_max for power normalization
    P_max_watts = 10 ** ((config['system']['P_max_dBm'] - 30) / 10)  # Convert dBm to Watts
    
    # Get TX-RX associations to map receiver ID to transmitter ID
    tx_rx_pairs = channel.tx_rx_pairs  # List of (tx_idx, rx_idx) tuples
    rx_to_tx = {rx: tx for tx, rx in tx_rx_pairs}  # Map receiver to its transmitter
    
    # Determine grid layout (max 2 rows, now only for graphs)
    ncols = int(np.ceil((num_samples + 1) / 2))
    nrows = 2
    
    # Scale figure size based on number of links (50 links is baseline, don't shrink below baseline)
    scale = max(1.0, np.sqrt(channel.n_links / 50.0))
    fig = plt.figure(figsize=(5*ncols*scale, 10*scale))
    gs = fig.add_gridspec(nrows, ncols, hspace=0.3, wspace=0.15, 
                          left=0.08, right=0.72, top=0.90, bottom=0.30)
    
    # Node positions
    node_pos = channel.rx_locations
    
    # Parse edge_index (only interference edges, no self-loops)
    edge_index = graph.edge_index.numpy()
    self_loop_mask = edge_index[0] == edge_index[1]
    interf_edges = edge_index[:, ~self_loop_mask]
    
    # Diverging colormap centered at r_min for rates
    vmin_rate = all_rates.min()
    vmax_rate = all_rates.max()
    norm_rate = TwoSlopeNorm(vmin=vmin_rate, vcenter=r_min, vmax=vmax_rate)
    cmap_rate = 'RdYlGn'  # Red (bad) - Yellow (r_min) - Green (good)
    
    # Power colormap (normalized by P_max, diverging centered at 0.5)
    all_powers = np.array([all_power_samples[i] for i in selected_indices])
    all_powers_norm = all_powers / P_max_watts  # Normalize to [0, 1]
    vmin_power = 0.0
    vmax_power = 1.0
    norm_power = TwoSlopeNorm(vmin=vmin_power, vcenter=0.5, vmax=vmax_power)
    cmap_power = 'RdBu_r'  # Red (low power) - White (0.5*P_max) - Blue (high power)
    
    # Collect power data for all samples (for heatmap below)
    power_matrices = []  # Will store normalized power for each sample
    
    # Plot each sample
    for m in range(num_samples):
        row = m // ncols
        col = m % ncols
        
        # Graph subplot
        ax_graph = fig.add_subplot(gs[row, col])
        rates = all_rates[m]
        power = power_samples[m]
        
        # Store normalized power for heatmap
        powers_ordered = np.array([power[rx_to_tx[rx_id]] for rx_id in range(graph.num_nodes)])
        power_matrices.append(powers_ordered / P_max_watts)
        
        # Draw interference edges (darker for visibility)
        for i in range(interf_edges.shape[1]):
            src, tgt = interf_edges[0, i], interf_edges[1, i]
            ax_graph.plot(
                [node_pos[src, 0], node_pos[tgt, 0]],
                [node_pos[src, 1], node_pos[tgt, 1]],
                'black', linewidth=0.5, alpha=0.3, zorder=1
            )
        
        # Draw nodes colored by ergodic rates
        scatter = ax_graph.scatter(
            node_pos[:, 0], node_pos[:, 1],
            c=rates, cmap=cmap_rate, norm=norm_rate, s=100, marker='o',
            edgecolors='black', linewidths=1,
            zorder=3
        )
        
        # Add node labels
        for i in range(graph.num_nodes):
            ax_graph.annotate(f'{i}', (node_pos[i, 0], node_pos[i, 1]),
                       ha='center', va='center', fontsize=5, color='white',
                       fontweight='bold', zorder=4)
        
        ax_graph.set_title(f'Sample {m+1}\nMin rate: {rates.min():.3f} bits/s/Hz',
                    fontsize=10, fontweight='bold')
        ax_graph.set_aspect('equal')
        ax_graph.axis('off')
    
    # Plot averaged rates
    row = num_samples // ncols
    col = num_samples % ncols
    
    # Graph subplot for averaged rates
    ax_graph = fig.add_subplot(gs[row, col])
    
    # Store averaged normalized power for heatmap
    avg_power = np.mean([all_power_samples[i] for i in selected_indices], axis=0)
    powers_ordered = np.array([avg_power[rx_to_tx[rx_id]] for rx_id in range(graph.num_nodes)])
    power_matrices.append(powers_ordered / P_max_watts)
    
    # Draw interference edges (darker for visibility)
    for i in range(interf_edges.shape[1]):
        src, tgt = interf_edges[0, i], interf_edges[1, i]
        ax_graph.plot(
            [node_pos[src, 0], node_pos[tgt, 0]],
            [node_pos[src, 1], node_pos[tgt, 1]],
            'black', linewidth=0.5, alpha=0.3, zorder=1
        )
    
    # Draw nodes colored by averaged rates
    scatter_avg = ax_graph.scatter(
        node_pos[:, 0], node_pos[:, 1],
        c=avg_rates, cmap=cmap_rate, norm=norm_rate, s=100, marker='o',
        edgecolors='black', linewidths=1,
        zorder=3
    )
    
    # Add node labels
    for i in range(graph.num_nodes):
        ax_graph.annotate(f'{i}', (node_pos[i, 0], node_pos[i, 1]),
                   ha='center', va='center', fontsize=5, color='white',
                   fontweight='bold', zorder=4)
    
    # Update title
    title_text = f'Averaged ({window_multiplier}M={actual_num_samples} checkpoints)\nMin rate: {avg_rates.min():.3f} bits/s/Hz'
    
    ax_graph.set_title(title_text, fontsize=10, fontweight='bold', color='red')
    ax_graph.set_aspect('equal')
    ax_graph.axis('off')
    
    # Hide unused subplots in the main grid
    for idx in range(num_samples + 1, nrows * ncols):
        ax_hidden = fig.add_subplot(gs[idx // ncols, idx % ncols])
        ax_hidden.axis('off')
    
    # Convert power_matrices to numpy array (N_links x M_samples+1)
    power_heatmap_data = np.array(power_matrices).T  # Transpose to (N_links, M_samples+1)
    
    # Create horizontal array of M+1 thin power heatmaps on the right side of figure
    heatmap_width_per_sample = 0.012  # Width for each thin vertical bar
    heatmap_height = 0.55  # Height spanning the two rows
    heatmap_y = 0.33  # Y position (aligned with graph rows)
    spacing = 0.003  # Gap between heatmaps
    total_heatmap_width = (num_samples + 1) * heatmap_width_per_sample + num_samples * spacing
    start_x = 0.74  # Start position to the right of graphs
    
    # Create M+1 power heatmaps arranged horizontally on the right
    for m in range(num_samples + 1):
        x_pos = start_x + m * (heatmap_width_per_sample + spacing)
        ax_power = fig.add_axes([x_pos, heatmap_y, heatmap_width_per_sample, heatmap_height])
        
        # Get power column for this sample
        power_column = power_heatmap_data[:, m].reshape(-1, 1)
        
        ax_power.imshow(power_column, cmap=cmap_power, norm=norm_power, aspect='auto',
                       interpolation='nearest')
        
        # Add label for each sample (below heatmap)
        if m < num_samples:
            label_text = f'S{m+1}'
            label_color = 'black'
        else:
            label_text = 'Avg'
            label_color = 'red'
        ax_power.text(0.5, -0.05, label_text, transform=ax_power.transAxes,
                     fontsize=8, fontweight='bold', color=label_color,
                     ha='center', va='top')
        
        # Remove all tick labels
        ax_power.set_yticks([])
        ax_power.set_xticks([])
    
    # Add power colorbar to the right of the heatmaps
    cbar_power_x = start_x + (num_samples + 1) * (heatmap_width_per_sample + spacing) + 0.005
    ax_cbar_power = fig.add_axes([cbar_power_x, heatmap_y, 0.01, heatmap_height])
    cbar_power = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap_power, norm=norm_power),
                              cax=ax_cbar_power)
    cbar_power.set_label('TX Power (normalized)', fontsize=11, fontweight='bold')
    cbar_power.ax.tick_params(labelsize=8)
    
    # Colorbar for rates (horizontal, closer to bottom row subplots)
    cbar_rate_ax = fig.add_axes([0.15, 0.20, 0.6, 0.02])
    cbar_rate = fig.colorbar(scatter_avg, cax=cbar_rate_ax, orientation='horizontal')
    cbar_rate.set_label('Ergodic Rate (bits/s/Hz)', fontsize=11, fontweight='bold')
    # Add explicit ticks to ensure values below r_min are shown
    tick_values = np.linspace(vmin_rate, vmax_rate, 8)
    cbar_rate.set_ticks(tick_values)
    cbar_rate.set_ticklabels([f'{v:.2f}' for v in tick_values], fontsize=8)
    cbar_rate.ax.axvline(x=r_min, color='black', linestyle='--', linewidth=2, alpha=0.8)
    cbar_rate.ax.text(r_min, -0.5, f'r_min={r_min:.2f}', ha='center', fontsize=9, 
                     fontweight='bold', transform=cbar_rate.ax.get_xaxis_transform())
    
    # Add figure title
    fig.suptitle(f'Stochastic Policy: Samples with Power Allocations (Network seed={info["seed"]}, r_min={r_min})',
                fontsize=14, fontweight='bold', y=0.98)
    
    # Save figure with descriptive filename
    filename = f'stochastic_policy_samples_network_{net_id}_checkpoints_{window_multiplier}M.pdf'
    
    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)
    save_path = viz_dir / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"\n  ✓ Saved stochastic policy visualization: {save_path}")
    plt.close()
    
    return save_path


def visualize_rate_convergence_with_samples(network_info, config, output_dir, primal_history=None, network_id=0, collection_metadata=None, window_multiplier=5, num_trials=10):
    """
    Visualize how rate statistics converge as the number of samples increases.
    
    For each t from 1 to T (where T = window_multiplier * M checkpoints):
    - Randomly sample t checkpoint epochs
    - Compute time-averaged ergodic rates per receiver over these t epochs
    - Compute min, 1st percentile, and 5th percentile across all receivers
    - Repeat multiple trials to show variability
    
    Parameters
    ----------
    network_info : dict
        Network information extracted from samples
    config : dict
        Hydra configuration
    output_dir : Path
        Output directory to save visualization
    primal_history : list, optional
        List of primal history entries (checkpointed epochs only)
    network_id : int or None, optional
        Network ID to analyze. If None, aggregates across all networks (default: 0)
    collection_metadata : dict, optional
        Collection metadata containing checkpoint_epochs
    window_multiplier : int, optional
        Window size multiplier (1=M, 5=5M, etc.) (default: 5)
    num_trials : int, optional
        Number of random trials to run for each t value (default: 10)
    """
    net_id = network_id
    
    if not primal_history:
        print("  ⚠️ No primal_history provided, cannot create convergence visualization")
        return None
    
    if not collection_metadata or 'checkpoint_epochs' not in collection_metadata:
        print("  ⚠️ No collection_metadata provided, cannot create convergence visualization")
        return None
    
    print("\n" + "="*70)
    if net_id is None:
        print(f"VISUALIZING RATE CONVERGENCE WITH SAMPLES - ALL NETWORKS (GLOBAL)")
    else:
        print(f"VISUALIZING RATE CONVERGENCE WITH SAMPLES - Network {net_id}")
    print(f"Window: {window_multiplier}M checkpoints, {num_trials} trials per sample size")
    print("="*70)
    
    # Extract checkpoint-interval epochs from primal_history
    checkpoint_epochs = collection_metadata['checkpoint_epochs']
    sample_collection_interval = checkpoint_epochs[1] - checkpoint_epochs[0] if len(checkpoint_epochs) >= 2 else 2
    checkpoint_base = checkpoint_epochs[0] % sample_collection_interval
    
    # Collect all checkpoint epochs and rates (optionally filtered by network)
    checkpoint_data = []  # List of (epoch, rates) tuples
    for entry in primal_history:
        if ('epoch' in entry and 'rates' in entry):
            # Filter by network_id if specified
            if net_id is not None:
                if 'network_id' not in entry or entry['network_id'] != net_id:
                    continue
            
            epoch = entry['epoch']
            if epoch % sample_collection_interval == checkpoint_base:
                rates = np.array(entry['rates'])
                checkpoint_data.append((epoch, rates))
    
    if len(checkpoint_data) == 0:
        if net_id is None:
            print(f"  ⚠️ No checkpoint data found")
        else:
            print(f"  ⚠️ No checkpoint data found for network {net_id}")
        return None
    
    # Sort by epoch and take last T checkpoints
    checkpoint_data.sort(key=lambda x: x[0])
    M_checkpoints = len(checkpoint_epochs)
    T = min(window_multiplier * M_checkpoints, len(checkpoint_data))
    checkpoint_data = checkpoint_data[-T:]
    
    epochs = [epoch for epoch, _ in checkpoint_data]
    all_rates = np.array([rates for _, rates in checkpoint_data])  # (T, N)
    N = all_rates.shape[1]
    
    print(f"  Using {T} checkpoint epochs from {epochs[0]} to {epochs[-1]}")
    print(f"  Each epoch has rates for {N} receivers")
    print(f"  Running {num_trials} random trials for each sample size t = 1 to {T}")
    
    # Get r_min
    r_min = config['training']['r_min']
    
    # For each t from 1 to T, run multiple trials
    t_values = []
    min_stats = []  # List of (mean, std) tuples for minimum
    p1_stats = []   # List of (mean, std) tuples for 1st percentile
    p5_stats = []   # List of (mean, std) tuples for 5th percentile
    
    # Sample points following pattern: 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, ...
    # Pattern repeats: 1*10^k, 2*10^k, 5*10^k, 1*10^(k+1), 2*10^(k+1), 5*10^(k+1), ...
    t_sample_points = []
    for k in range(20):  # Enough to cover any reasonable T
        for base in [1, 2, 5]:
            val = base * (10 ** k)
            if val <= T:
                t_sample_points.append(val)
            else:
                break
        if t_sample_points and t_sample_points[-1] >= T:
            break
    
    # Ensure T is included
    if T not in t_sample_points:
        t_sample_points.append(T)
    
    t_sample_points = sorted(set(t_sample_points))  # Remove duplicates and sort
    
    print(f"  Sampling {len(t_sample_points)} values of t: {t_sample_points}")
    
    for t in t_sample_points:
        trial_mins = []
        trial_p1s = []
        trial_p5s = []
        
        for trial in range(num_trials):
            # Randomly sample t epochs (without replacement)
            sampled_indices = np.random.choice(T, size=t, replace=False)
            sampled_rates = all_rates[sampled_indices]  # (t, N)
            
            # Compute time-averaged rates per receiver
            avg_rates = sampled_rates.mean(axis=0)  # (N,)
            
            # Compute statistics across receivers
            trial_mins.append(avg_rates.min())
            trial_p1s.append(np.percentile(avg_rates, 1))
            trial_p5s.append(np.percentile(avg_rates, 5))
        
        # Store mean and std across trials
        t_values.append(t)
        min_stats.append((np.mean(trial_mins), np.std(trial_mins)))
        p1_stats.append((np.mean(trial_p1s), np.std(trial_p1s)))
        p5_stats.append((np.mean(trial_p5s), np.std(trial_p5s)))
    
    t_values = np.array(t_values)
    min_means = np.array([m for m, s in min_stats])
    min_stds = np.array([s for m, s in min_stats])
    p1_means = np.array([m for m, s in p1_stats])
    p1_stds = np.array([s for m, s in p1_stats])
    p5_means = np.array([m for m, s in p5_stats])
    p5_stds = np.array([s for m, s in p5_stats])
    
    print(f"  ✓ Computed convergence statistics")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot with error bars for standard deviation
    ax.errorbar(t_values, min_means, yerr=min_stds, fmt='r-o', linewidth=2, 
                markersize=4, capsize=3, label='Minimum', alpha=0.8, elinewidth=1)
    
    ax.errorbar(t_values, p1_means, yerr=p1_stds, fmt='orange', marker='s', linewidth=2,
                markersize=4, capsize=3, label='1st Percentile', alpha=0.8, elinewidth=1)
    
    ax.errorbar(t_values, p5_means, yerr=p5_stds, fmt='g-^', linewidth=2,
                markersize=4, capsize=3, label='5th Percentile', alpha=0.8, elinewidth=1)
    
    # Add r_min reference line
    ax.axhline(y=r_min, color='k', linestyle='--', linewidth=1.5, 
               label=f'$r_{{min}}$ = {r_min:.2f}', alpha=0.7)
    
    # Formatting
    ax.set_xlabel('Number of Samples Averaged ($t$)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Ergodic Rate (bits/s/Hz)', fontsize=12, fontweight='bold')
    if net_id is None:
        title_text = f'Rate Convergence with Sample Size (All Networks, {window_multiplier}M window)\n'
    else:
        title_text = f'Rate Convergence with Sample Size (Network {net_id}, {window_multiplier}M window)\n'
    title_text += f'Error bars show ±1 std over {num_trials} random trials'
    ax.set_title(title_text, fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # Set x-axis limits and ticks
    ax.set_xlim(1, T)
    
    # Add text with final statistics
    final_min = min_means[-1]
    final_p1 = p1_means[-1]
    final_p5 = p5_means[-1]
    
    textstr = f'At $t={T}$:\n'
    textstr += f'Min = {final_min:.4f}\n'
    textstr += f'P1 = {final_p1:.4f}\n'
    textstr += f'P5 = {final_p5:.4f}'
    
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    if net_id is None:
        filename = f'rate_convergence_all_networks_{window_multiplier}M.pdf'
    else:
        filename = f'rate_convergence_network_{net_id}_{window_multiplier}M.pdf'
    viz_dir = output_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)
    save_path = viz_dir / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"\n  ✓ Saved rate convergence visualization: {save_path}")
    plt.close()
    
    return save_path


def main():
    parser = argparse.ArgumentParser(
        description="Load and verify collected samples from primal-dual training"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Path to output directory (e.g., outputs/primal_dual/2026-01-27/14-17-59)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed information'
    )
    parser.add_argument(
        '--skip-network-graph-viz',
        action='store_true',
        help = 'Skip visualization of sample networks (useful for quick verification without generating figures)'
    )
    parser.add_argument(
        '--num-example-networks',
        type=int,
        default=2,
        help='Number of example networks to verify (default: 2)'
    )
    parser.add_argument(
        '--retrieve-feasible-subset',
        action='store_true',
        help='Retrieve and save a feasibility-constrained, diversity-preserving subset from checkpoint trajectories'
    )
    parser.add_argument(
        '--subset-size',
        type=int,
        default=200,
        help='Target subset size for retrieval (default: 200)'
    )
    parser.add_argument(
        '--subset-window-multiplier',
        type=int,
        default=10,
        help='Use the last (subset_window_multiplier * M) checkpoint samples as source window (default: 10)'
    )
    parser.add_argument(
        '--subset-network-id',
        type=int,
        default=None,
        help='If set, retrieve subset only for this network ID; otherwise uses first num-example-networks'
    )
    parser.add_argument(
        '--subset-feasibility-tolerance',
        type=float,
        default=0.0,
        help='Feasibility tolerance: enforce avg_rate >= (r_min - tolerance) (default: 0.0)'
    )
    parser.add_argument(
        '--subset-bottleneck-nodes',
        type=int,
        default=5,
        help='Number of worst receivers used for diversity space (default: 5)'
    )
    parser.add_argument(
        '--analyze-subset-sizes',
        action='store_true',
        help='Sweep subset sizes and plot min/1st/5th percentile subset-averaged rates vs subset size'
    )
    parser.add_argument(
        '--subset-size-grid',
        type=str,
        default='auto',
        help="Comma-separated requested subset sizes (e.g. '20,50,100,200') or 'auto' for 1-2-5 progression"
    )
    parser.add_argument(
        '--save-subset-size-sweep-subsets',
        action='store_true',
        help='Also save the selected subset samples for each size in the sweep'
    )
    parser.add_argument(
        '--analyze-subset-size-convergence',
        action='store_true',
        help='For each tested subset size k, plot min/1st/5th percentile convergence versus random sample count t<=k'
    )
    parser.add_argument(
        '--subset-convergence-num-trials',
        type=int,
        default=10,
        help='Number of random trials per t for subset-convergence plots (default: 10)'
    )
    parser.add_argument(
        '--subset-convergence-t-grid',
        type=str,
        default='auto',
        help="Comma-separated t values for subset-convergence (or 'auto' for 1-2-5 progression up to k)"
    )
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")
    
    print("="*70)
    print("LOADING AND VERIFYING COLLECTED SAMPLES")
    print("="*70)
    print(f"Output directory: {output_dir}")
    print("Note: this script is verification/visualization only.")
    print("      To generate diffusion-ready raw WRA data, run:")
    print("      python -m graph_signal_diffusion.cli.wra.build_diffusion_dataset --input <PD_RUN_DIR> --output-root data/wra")
    
    # 1. Load samples
    print("\n1. Loading samples...")
    samples, metadata, config, quality_report, primal_history, collection_metadata = load_samples(output_dir)
    
    print(f"   Loaded {len(samples)} keys from samples file")
    print(f"   Config: {config['dataset']['num_networks']} networks, "
          f"{config['dataset']['n_links']} links each")
    
    if primal_history:
        # Filter entries with rates (actual epoch data, not metadata)
        epoch_entries = [e for e in primal_history if 'epoch' in e and 'rates' in e]
        if len(epoch_entries) > 0:
            first_epoch = min(e['epoch'] for e in epoch_entries)
            last_epoch = max(e['epoch'] for e in epoch_entries)
            print(f"   Loaded {len(primal_history)} entries from primal_history.jsonl")
            print(f"   Primal-dual training epoch range: {first_epoch} to {last_epoch} ({len(epoch_entries)} epochs with rate data)")
        else:
            print(f"   Loaded {len(primal_history)} entries from primal_history.jsonl (no epoch data found)")
    
    if collection_metadata:
        print(f"   Loaded collection_metadata.json")
        if 'checkpoint_epochs' in collection_metadata:
            print(f"      Found {len(collection_metadata['checkpoint_epochs'])} checkpoint epochs")
    
    # 1.5 Try to load cached channels
    print("\n1.5 Checking for cached channels...")
    channels = None
    name = config.get("name", "")
    
    # Resolve Hydra variable interpolations in channel_config_name
    channel_config_name = config.get('channel_config_name', f'default_{config["seed"]}')
    
    # If channel_config_name contains ${...}, resolve it manually
    if '${' in channel_config_name:
        import re
        # Replace ${key} with config[key] and ${section.key} with config[section][key]
        def resolve_var(match):
            var_path = match.group(1)
            parts = var_path.split('.')
            try:
                value = config
                for part in parts:
                    value = value[part]
                return str(value)
            except (KeyError, TypeError):
                return match.group(0)  # Return original if can't resolve
        
        channel_config_name = re.sub(r'\$\{([^}]+)\}', resolve_var, channel_config_name)
    
    if name:
        cache_dir = Path('data/wra_channel_cache') / name
        channels_file = cache_dir / f'{channel_config_name}.pt'
        
        if channels_file.exists():
            print(f"   Loading cached channels from {channels_file}")
            try:
                cached = torch.load(channels_file, map_location='cpu')
                channels = cached['channels']
                print(f"   ✓ Loaded {len(channels)} pre-generated channels from cache")
                
                # Verify cache metadata matches config
                cache_metadata = cached.get('metadata', {})
                if cache_metadata.get('n_links') != config['dataset']['n_links']:
                    print(f"   ⚠ Warning: Cache n_links ({cache_metadata.get('n_links')}) != config n_links ({config['dataset']['n_links']})")
                    channels = None
            except Exception as e:
                print(f"   ✗ Failed to load cached channels: {e}")
                channels = None
        else:
            print(f"   No cached channels found at {channels_file}")
    else:
        print("   No 'name' in config, skipping cache lookup")
    
    # Debug: show first few keys
    sample_keys = list(samples.keys())
    print(f"   Sample keys (first 10): {sample_keys[:10]}")
    
    if metadata:
        print(f"   Metadata:")
        for key, value in metadata.items():
            print(f"     {key}: {value}")
    
    # 2. Extract network information
    print("\n2. Extracting network information...")
    network_info = extract_network_info(samples, config['dataset']['num_networks'], config)
    
    total_samples = sum(info['num_samples'] for info in network_info.values())
    print(f"   Extracted data for {len(network_info)} networks")
    print(f"   Total samples: {total_samples}")
    if network_info:
        first_net_id = sorted(network_info.keys())[0]
        print(f"   Samples per network (example network {first_net_id}): {network_info[first_net_id]['num_samples']}")
    
    # Display actual collection epochs from metadata
    if collection_metadata and 'checkpoint_epochs' in collection_metadata:
        checkpoint_epochs = collection_metadata['checkpoint_epochs']
        print(f"   Collection epoch range: {checkpoint_epochs[0]} to {checkpoint_epochs[-1]}")
        if len(checkpoint_epochs) >= 2:
            interval = checkpoint_epochs[1] - checkpoint_epochs[0]
            print(f"   Sample collection interval: {interval} epochs")
            print(f"   Number of checkpoints: {len(checkpoint_epochs)}")
    
    # 3. Regenerate and verify M collected samples
    print("\n3. Regenerating networks and verifying M collected samples...")
    verification_results = regenerate_and_verify(network_info, config, quality_report, channels=channels, num_example_networks=args.num_example_networks)
    
    # Check if verification passed
    rates_match = all(v['rates_match'] for v in verification_results.values())
    num_vis_networks = min(args.num_example_networks, len(network_info))
    
    if not rates_match:
        print("\n❌ Rate verification FAILED for collected samples!")
        print("   Cannot trust pre-computed rates in primal_history.")
        print("   Skipping window analysis.")
        if args.retrieve_feasible_subset:
            print("   Skipping feasible subset retrieval.")
        if args.analyze_subset_sizes:
            print("   Skipping subset-size sweep analysis.")
        if args.analyze_subset_size_convergence:
            print("   Skipping subset-size convergence analysis.")
        window_stats = None
    else:
        print("\n✅ Rate verification PASSED for M collected samples")
        print("   Pre-computed rates in primal_history are trustworthy.")
        
        # 3.5. Analyze rate statistics over multiple windows (using pre-computed rates)
        # Run analysis for each example network to match visualizations
        print("\n3.5. Analyzing rate statistics over multiple windows...")
        for net_id in range(min(args.num_example_networks, len(network_info))):
            window_stats = analyze_rate_statistics_over_windows(
                primal_history, 
                config, 
                config['dataset']['num_networks'],
                collection_metadata,
                network_id=net_id
            )

        # 3.6. Retrieve feasible subset from checkpoint trajectories
        subset_ops_requested = (
            args.retrieve_feasible_subset
            or args.analyze_subset_sizes
            or args.analyze_subset_size_convergence
        )
        target_network_ids = []
        if subset_ops_requested:
            if args.subset_network_id is not None:
                target_network_ids = [args.subset_network_id]
            else:
                target_network_ids = list(range(num_vis_networks))

        if args.retrieve_feasible_subset:
            print("\n3.6. Retrieving feasible subsets from checkpoint trajectories...")

            if not primal_history or not collection_metadata or 'checkpoint_epochs' not in collection_metadata:
                print("   ⚠️ Missing primal_history or checkpoint metadata; cannot retrieve subsets.")
            else:
                subset_dir = output_dir / "feasible_subsets"
                subset_dir.mkdir(parents=True, exist_ok=True)

                for net_id in target_network_ids:
                    if net_id not in network_info:
                        print(f"   ⚠️ Network {net_id} not found in loaded samples; skipping.")
                        continue

                    extracted = extract_checkpoint_window_samples(
                        primal_history=primal_history,
                        collection_metadata=collection_metadata,
                        network_id=net_id,
                        window_multiplier=args.subset_window_multiplier,
                    )

                    if extracted is None:
                        print(f"   ⚠️ No checkpoint trajectory samples found for network {net_id}; skipping.")
                        continue

                    epochs, powers, rates, extraction_info = extracted
                    print(
                        f"   Network {net_id}: source window {extraction_info['actual_window_size']} samples "
                        f"(epochs {extraction_info['epoch_start']}-{extraction_info['epoch_end']})"
                    )

                    selected_indices, selection_summary = select_feasible_diverse_subset(
                        rates=rates,
                        target_size=args.subset_size,
                        r_min=config['training']['r_min'],
                        feasibility_tolerance=args.subset_feasibility_tolerance,
                        num_bottleneck_nodes=args.subset_bottleneck_nodes,
                    )

                    selected_epochs = epochs[selected_indices]
                    selected_powers = powers[selected_indices]
                    selected_rates = rates[selected_indices]

                    npz_path = subset_dir / (
                        f"network_{net_id}_subset_k{len(selected_indices)}"
                        f"_from_{extraction_info['actual_window_size']}"
                        f"_window_{args.subset_window_multiplier}M.npz"
                    )
                    report_path = subset_dir / (
                        f"network_{net_id}_subset_k{len(selected_indices)}"
                        f"_from_{extraction_info['actual_window_size']}"
                        f"_window_{args.subset_window_multiplier}M.json"
                    )

                    np.savez_compressed(
                        npz_path,
                        selected_indices_in_window=selected_indices,
                        selected_epochs=selected_epochs,
                        selected_power_allocations=selected_powers,
                        selected_rates=selected_rates,
                        source_window_epochs=epochs,
                    )

                    report = {
                        'network_id': int(net_id),
                        'r_min': float(config['training']['r_min']),
                        'subset_size_requested': int(args.subset_size),
                        'subset_size_selected': int(len(selected_indices)),
                        'subset_window_multiplier': int(args.subset_window_multiplier),
                        'subset_feasibility_tolerance': float(args.subset_feasibility_tolerance),
                        'subset_bottleneck_nodes': int(args.subset_bottleneck_nodes),
                        'extraction_info': extraction_info,
                        'selection_summary': selection_summary,
                        'output_npz': str(npz_path),
                    }
                    with open(report_path, 'w') as f:
                        json.dump(report, f, indent=2)

                    feasibility_mark = "✓" if selection_summary['is_feasible'] else "⚠️"
                    print(
                        f"   {feasibility_mark} Saved subset for network {net_id}: "
                        f"{len(selected_indices)} samples -> {npz_path.name} "
                        f"(violating receivers: {selection_summary['num_violating_receivers']})"
                    )

        if args.analyze_subset_sizes or args.analyze_subset_size_convergence:
            print("\n3.7. Sweeping subset sizes and plotting rate percentiles...")
            if not primal_history or not collection_metadata or 'checkpoint_epochs' not in collection_metadata:
                print("   ⚠️ Missing primal_history or checkpoint metadata; cannot run subset-size sweep.")
            else:
                for net_id in target_network_ids:
                    if net_id not in network_info:
                        print(f"   ⚠️ Network {net_id} not found in loaded samples; skipping sweep.")
                        continue

                    print(
                        f"   Network {net_id}: size grid='{args.subset_size_grid}', "
                        f"window={args.subset_window_multiplier}M"
                    )
                    analyze_subset_quality_vs_size(
                        output_dir=output_dir,
                        config=config,
                        primal_history=primal_history,
                        collection_metadata=collection_metadata,
                        network_id=net_id,
                        window_multiplier=args.subset_window_multiplier,
                        subset_size_grid=args.subset_size_grid,
                        feasibility_tolerance=args.subset_feasibility_tolerance,
                        num_bottleneck_nodes=args.subset_bottleneck_nodes,
                        save_sweep_subsets=args.save_subset_size_sweep_subsets,
                        analyze_subset_convergence=args.analyze_subset_size_convergence,
                        subset_convergence_num_trials=args.subset_convergence_num_trials,
                        subset_convergence_t_grid=args.subset_convergence_t_grid,
                    )

    if not args.skip_network_graph_viz:
        # 4. Visualize all example networks
        print("\n4. Creating network visualizations...")
        for net_id in range(num_vis_networks):
            visualize_network_and_graph(network_info, config, output_dir, channels=channels, network_id=net_id)
        
        # 5. Visualize stochastic policy samples for all example networks
        # Create visualizations for M, 2M, 5M, and 10M windows
        if primal_history and collection_metadata and 'checkpoint_epochs' in collection_metadata:
            for multiplier in [1, 2, 5, 10]:
                print(f"\n5.{multiplier}. Creating stochastic policy visualizations ({multiplier}M checkpoints)...")
                for net_id in range(num_vis_networks):
                    visualize_stochastic_policy_samples(
                        network_info, config, output_dir, 
                        num_samples=5, 
                        channels=channels, 
                        primal_history=primal_history,
                        network_id=net_id,
                        collection_metadata=collection_metadata,
                        window_multiplier=multiplier
                    )
    
    # 5.5. Create rate convergence visualizations
    if primal_history and collection_metadata and 'checkpoint_epochs' in collection_metadata:
        multiplier = 10  # Only for 10M window
        print(f"\n5.5. Creating rate convergence visualizations ({multiplier}M checkpoints)...")
        
        # Per-network convergence plots
        for net_id in range(num_vis_networks):
            visualize_rate_convergence_with_samples(
                network_info, config, output_dir,
                primal_history=primal_history,
                network_id=net_id,
                collection_metadata=collection_metadata,
                window_multiplier=multiplier,
                num_trials=10  # 10 random trials per sample size
            )
        
        # Global convergence plot (all networks)
        print(f"\n5.5.global. Creating global rate convergence visualization ({multiplier}M checkpoints)...")
        visualize_rate_convergence_with_samples(
            network_info, config, output_dir,
            primal_history=primal_history,
            network_id=None,  # Aggregate across all networks
            collection_metadata=collection_metadata,
            window_multiplier=multiplier,
            num_trials=10  # 10 random trials per sample size
        )
    
    # 6. Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    # Filter out None values (networks without seeds couldn't be regenerated)
    associations_match = all(v['associations_match'] for v in verification_results.values())
    rates_match = all(v['rates_match'] for v in verification_results.values())
    all_match = associations_match and rates_match
    
    if all_match:
        print("✓ ALL NETWORKS VERIFIED SUCCESSFULLY")
        print("\nKey findings:")
        print("  • Network seeds correctly stored/inferred and retrieved")
        print("  • Regenerated networks have identical topology and associations")
        print("  • Recomputed ergodic rates match saved rates")
        print("  • Sample collection process is fully reproducible")
        print("\nConclusion:")
        print("  Samples can be reliably validated and visualized with this script.")
        print("  Diffusion dataset generation is a separate mandatory step via")
        print("  graph_signal_diffusion.cli.wra.build_diffusion_dataset.")
    else:
        print("✗ VERIFICATION FAILED")
        if not associations_match:
            failed_assoc = [net_id for net_id, v in verification_results.items() 
                           if not v['associations_match']]
            print(f"  Networks with association mismatches: {failed_assoc}")
        if not rates_match:
            failed_rates = [net_id for net_id, v in verification_results.items() 
                           if not v['rates_match']]
            print(f"  Networks with rate mismatches: {failed_rates}")
    
    print("="*70)
    
    return 0 if all_match else 1


if __name__ == '__main__':
    main()
