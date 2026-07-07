#!/usr/bin/env python3
"""
Configure and analyze WRA dataset parameters.

Generates channel networks, computes full-power ergodic rates, and recommends
r_min targets for primal-dual training.  Channels are saved to a content-
addressed cache for fast reuse by the training script.

Usage:
    python scripts/configure_wra_dataset.py --config-name=wra_small_channel
    python scripts/configure_wra_dataset.py --config-name=wra_large_low_density_channel
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from graph_signal_diffusion.datasets.wra.channel import WirelessChannel
from graph_signal_diffusion.datasets.wra.channel_factory import (
    build_wra_channel_cache_metadata,
    canonicalize_channel_cache_metadata,
    find_channel_cache_metadata_mismatches,
    get_channel_class,
    normalize_channel_version,
    register_wra_hydra_resolvers,
    sanitize_channel_params,
)
from graph_signal_diffusion.utils.graph_builder import build_interference_graph

register_wra_hydra_resolvers()

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging / output setup
# ---------------------------------------------------------------------------

def _suppress_console_logging_for(module_prefix: str) -> None:
    """Filter out logs from *module_prefix* on console handlers only.

    Hydra's FileHandler still receives all messages; only the StreamHandler
    (terminal output) is filtered so noisy deployment/pairing logs stay in the
    log file but don't flood the console.
    """
    class _ModuleBlockFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return not record.name.startswith(module_prefix)

    root = logging.getLogger()
    for handler in root.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler, logging.FileHandler
        ):
            handler.addFilter(_ModuleBlockFilter(module_prefix))


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def visualize_network_deployment(
    channel: WirelessChannel,
    network_id: int,
    save_path: Path,
    config: dict,
):
    """Visualize a single network deployment with its PyG interference graph.

    Creates two subplots:
    1. TX/RX spatial deployment with paired links in green.
    2. PyG interference graph (nodes = receivers, edges = interference).
    """
    top_k = config.get('dataset', {}).get('top_k', 10)
    min_degree = config.get('dataset', {}).get('min_degree', 1)
    graph = build_interference_graph(
        H_ls=channel.large_scale_fading,
        associations=channel.associations,
        top_k=top_k,
        min_degree=min_degree,
        normalize_method='log1p',
    )

    scale = max(1.0, np.sqrt(channel.n_links / 50.0))
    fig, axes = plt.subplots(1, 2, figsize=(16 * scale, 7 * scale))
    tx_locs = channel.tx_locations
    rx_locs = channel.rx_locations

    # -- Subplot 1: spatial deployment -----------------------------------------
    ax = axes[0]
    for tx_idx, rx_idx in channel.tx_rx_pairs:
        ax.plot(
            [tx_locs[tx_idx, 0], rx_locs[rx_idx, 0]],
            [tx_locs[tx_idx, 1], rx_locs[rx_idx, 1]],
            'g-', linewidth=2.5, alpha=0.7, zorder=1,
        )
    ax.scatter(tx_locs[:, 0], tx_locs[:, 1], c='red', s=80, marker='^',
               edgecolors='black', linewidths=1.2, label='Transmitters', zorder=3)
    ax.scatter(rx_locs[:, 0], rx_locs[:, 1], c='blue', s=80, marker='o',
               edgecolors='black', linewidths=1.2, label='Receivers', zorder=3)
    for i in range(channel.n_links):
        ax.annotate(f'TX{i}', (tx_locs[i, 0], tx_locs[i, 1]),
                    xytext=(3, 3), textcoords='offset points', fontsize=5, alpha=0.7)
        ax.annotate(f'RX{i}', (rx_locs[i, 0], rx_locs[i, 1]),
                    xytext=(3, 3), textcoords='offset points', fontsize=5, alpha=0.7)
    ax.set(xlabel='X Position (m)', ylabel='Y Position (m)')
    ax.set_title(f'Network {network_id}: Spatial Deployment', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # -- Subplot 2: PyG interference graph -------------------------------------
    ax = axes[1]
    node_pos = rx_locs
    edge_index = graph.edge_index.numpy()
    edge_weight = graph.edge_weight.numpy()

    self_loop_mask = edge_index[0] == edge_index[1]
    self_loop_edges = edge_index[:, self_loop_mask]
    interf_edges = edge_index[:, ~self_loop_mask]
    interf_weights = edge_weight[~self_loop_mask]

    if len(interf_weights) > 0:
        interf_weights_norm = (interf_weights - interf_weights.min()) / (
            interf_weights.max() - interf_weights.min() + 1e-10
        )
    else:
        interf_weights_norm = interf_weights

    for i in range(interf_edges.shape[1]):
        src, tgt = interf_edges[0, i], interf_edges[1, i]
        w = interf_weights_norm[i]
        ax.annotate(
            '', xy=(node_pos[tgt, 0], node_pos[tgt, 1]),
            xytext=(node_pos[src, 0], node_pos[src, 1]),
            arrowprops=dict(arrowstyle='->', color=plt.cm.Reds(0.5 + 0.5 * w),
                            alpha=0.3 + 0.5 * w, lw=0.5 + 1.5 * w,
                            connectionstyle='arc3,rad=0.1'),
            zorder=1,
        )

    direct_signal = graph.x[:, 0].numpy()
    if self_loop_edges.shape[1] > 0:
        ds_norm = (direct_signal - direct_signal.min()) / (
            direct_signal.max() - direct_signal.min() + 1e-10
        )
    else:
        ds_norm = np.zeros(graph.num_nodes)
    for i in range(self_loop_edges.shape[1]):
        node = self_loop_edges[0, i]
        s = ds_norm[node]
        circle = plt.Circle(
            (node_pos[node, 0], node_pos[node, 1] + 20), radius=12,
            color=plt.cm.Blues(0.5 + 0.5 * s), alpha=0.9, fill=False,
            linewidth=2 + 2 * s, zorder=2,
        )
        ax.add_patch(circle)

    interference_colors = graph.x[:, 1].numpy()
    ax.scatter(node_pos[:, 0], node_pos[:, 1], c=interference_colors,
               cmap='Oranges', s=200, marker='o', edgecolors='black',
               linewidths=2, zorder=4)
    for i in range(graph.num_nodes):
        ax.annotate(f'{i}', (node_pos[i, 0], node_pos[i, 1]),
                    ha='center', va='center', fontsize=8, fontweight='bold',
                    color='white', zorder=5)

    ax.legend(
        handles=[
            plt.Line2D([0], [0], color='red', alpha=0.5, lw=2,
                       label='Interference edge'),
            plt.Line2D([0], [0], color='darkblue', alpha=0.9, lw=3, marker='o',
                       markersize=10, markerfacecolor='none', linestyle='None',
                       label='Self-loop (direct signal)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
                       markersize=10, markeredgecolor='black', linestyle='None',
                       label='Node (interference potential)'),
        ],
        loc='upper right', fontsize=9,
    )
    ax.set(xlabel='X Position (m)', ylabel='Y Position (m)')
    ax.set_title(f'PyG Graph (nodes=RX, edges=interference, top_k={top_k})',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    fig.text(0.5, 0.02,
             f'{channel.n_links} TX-RX pairs | {graph.num_nodes} nodes, '
             f'{interf_edges.shape[1]} interference edges, '
             f'{self_loop_edges.shape[1]} self-loops',
             ha='center', fontsize=10, style='italic')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    filename = save_path / f'network_{network_id}_deployment.pdf'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved deployment visualization: {filename}")


def visualize_rate_results(
    channel: WirelessChannel,
    results: dict,
    network_id: int,
    save_path: Path,
):
    """Visualize ergodic-rate analysis for a single network."""
    rates_over_time = results['rates_over_time']
    ergodic_rates = results['ergodic_rates']
    T, n = rates_over_time.shape

    scale = max(1.0, np.sqrt(channel.n_links / 50.0))
    fig, axes = plt.subplots(2, 2, figsize=(16 * scale, 12 * scale))

    # Rates over time
    ax = axes[0, 0]
    for i in range(n):
        ax.plot(rates_over_time[:, i], alpha=0.7, linewidth=0.8,
                label=f'User {i+1}' if n <= 10 else '')
    ax.set(xlabel='Time Step', ylabel='Instantaneous Rate (bits/s/Hz)')
    ax.set_title(f'Network {network_id}: Rates Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if n <= 10:
        ax.legend(loc='best', fontsize=8)

    # Ergodic rates per user
    ax = axes[0, 1]
    ax.bar(np.arange(1, n + 1), ergodic_rates, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axhline(results['mean'], color='red', linestyle='--', lw=2,
               label=f"Mean: {results['mean']:.3f}")
    ax.axhline(results['min'], color='orange', linestyle='--', lw=2,
               label=f"Min: {results['min']:.3f}")
    ax.set(xlabel='User Index', ylabel='Ergodic Rate (bits/s/Hz)')
    ax.set_title('Ergodic Rates per User', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Rate distribution
    ax = axes[1, 0]
    ax.hist(rates_over_time.flatten(), bins=50, color='skyblue', alpha=0.7, edgecolor='black')
    ax.axvline(results['mean'], color='red', linestyle='--', lw=2, label='Mean Ergodic')
    ax.set(xlabel='Instantaneous Rate (bits/s/Hz)', ylabel='Frequency')
    ax.set_title('Distribution of Instantaneous Rates', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # CDF of ergodic rates
    ax = axes[1, 1]
    sorted_rates = np.sort(ergodic_rates)
    cdf = np.arange(1, n + 1) / n
    ax.plot(sorted_rates, cdf, marker='o', lw=2, markersize=5, color='darkblue')
    ax.axvline(results['mean'], color='red', linestyle='--', lw=2, label='Mean')
    ax.axvline(results['percentile_5'], color='orange', linestyle='--', lw=2,
               label=f"5th %ile: {results['percentile_5']:.3f}")
    ax.set(xlabel='Ergodic Rate (bits/s/Hz)', ylabel='CDF')
    ax.set_title('CDF of Ergodic Rates', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filename = save_path / f'network_{network_id}_rates.pdf'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved rate visualization: {filename}")


def visualize_global_statistics(all_results: list, save_path: Path,
                                config: dict, n_links: int):
    """Create six-panel overview of rate statistics across all networks."""
    n_networks = len(all_results)
    mean_rates = [r['mean'] for r in all_results]
    min_rates = [r['min'] for r in all_results]
    percentile_5 = [r['percentile_5'] for r in all_results]
    percentile_10 = [r['percentile_10'] for r in all_results]
    fairness = [r['fairness'] for r in all_results]

    scale = max(1.0, np.sqrt(n_links / 50.0))
    fig, axes = plt.subplots(2, 3, figsize=(18 * scale, 12 * scale))

    # Mean rates
    ax = axes[0, 0]
    ax.bar(range(n_networks), mean_rates, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axhline(np.mean(mean_rates), color='red', linestyle='--', lw=2,
               label=f'Global Mean: {np.mean(mean_rates):.3f}')
    ax.set(xlabel='Network ID', ylabel='Mean Ergodic Rate (bits/s/Hz)')
    ax.set_title('Mean Rate per Network', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis='y')

    # Min rates
    ax = axes[0, 1]
    ax.bar(range(n_networks), min_rates, color='coral', alpha=0.7, edgecolor='black')
    ax.axhline(np.mean(min_rates), color='red', linestyle='--', lw=2,
               label=f'Global Mean: {np.mean(min_rates):.3f}')
    ax.set(xlabel='Network ID', ylabel='Min Ergodic Rate (bits/s/Hz)')
    ax.set_title('Min Rate per Network', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis='y')

    # Fairness
    ax = axes[0, 2]
    ax.bar(range(n_networks), fairness, color='lightgreen', alpha=0.7, edgecolor='black')
    ax.axhline(np.mean(fairness), color='red', linestyle='--', lw=2,
               label=f'Global Mean: {np.mean(fairness):.3f}')
    ax.set(xlabel='Network ID', ylabel="Jain's Fairness Index")
    ax.set_title('Fairness per Network', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis='y')

    # Distribution of mean rates
    ax = axes[1, 0]
    ax.hist(mean_rates, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(mean_rates), color='red', linestyle='--', lw=2, label='Mean')
    ax.set(xlabel='Mean Ergodic Rate (bits/s/Hz)', ylabel='Number of Networks')
    ax.set_title('Distribution of Mean Rates', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis='y')

    # Percentile box plots
    ax = axes[1, 1]
    bp = ax.boxplot(
        [min_rates, percentile_5, percentile_10, mean_rates],
        positions=[1, 2, 3, 4],
        labels=['Min', '5th %ile', '10th %ile', 'Mean'],
        # tick_labels=['Min', '5th %ile', '10th %ile', 'Mean'], # deprecated in newer Matplotlib versions
        patch_artist=True,
    )
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    ax.set_ylabel('Ergodic Rate (bits/s/Hz)')
    ax.set_title('Rate Percentiles Across Networks', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Recommended r_min
    ax = axes[1, 2]
    all_ergodic = np.concatenate([r['ergodic_rates'] for r in all_results])
    g5  = float(np.percentile(all_ergodic, 5))
    g10 = float(np.percentile(all_ergodic, 10))
    g50 = float(np.percentile(all_ergodic, 50))
    recommendations = {
        'Easy\n(+10% above 5th)': g5 * 1.1,
        'Moderate\n(halfway to median)': (g10 + g50) / 2,
        'Challenging\n(at 25th %ile)': float(np.percentile(all_ergodic, 25)),
        'Very Hard\n(at median)': g50,
    }
    bars = ax.bar(range(len(recommendations)), list(recommendations.values()),
                  color=['lightgreen', 'yellow', 'orange', 'red'],
                  alpha=0.7, edgecolor='black')
    ax.set_xticks(range(len(recommendations)))
    ax.set_xticklabels(list(recommendations.keys()), fontsize=10)
    ax.set_ylabel('Suggested r_min (bits/s/Hz)')
    ax.set_title('Recommended r_min Values', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(g5, color='blue', ls=':', lw=1.5, alpha=0.5, label=f'5th %ile: {g5:.3f}')
    ax.axhline(g50, color='purple', ls=':', lw=1.5, alpha=0.5, label=f'Median: {g50:.3f}')
    ax.legend(fontsize=8, loc='upper left')
    for bar, val in zip(bars, recommendations.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    filename = save_path / 'global_statistics.pdf'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved global statistics: {filename}")


# ---------------------------------------------------------------------------
# Rate computation (local full-power baseline, distinct from rate_calculator)
# ---------------------------------------------------------------------------

def compute_ergodic_rates(
    channel: WirelessChannel,
    num_timesteps: int,
    P_max_dBm: float,
    bandwidth_Hz: float = 10e6,
    noise_psd_dBm_Hz: float = -174.0,
) -> dict:
    """Compute ergodic rates under constant full-power allocation.

    Returns a dict with per-user ergodic rates, percentile statistics,
    and Jain's fairness index.
    """
    n = channel.n_links

    # Convert to linear units
    P_max_watts = 10 ** ((P_max_dBm - 30) / 10)
    noise_power_dBm = noise_psd_dBm_Hz + 10 * np.log10(bandwidth_Hz)
    noise_var = 10 ** ((noise_power_dBm - 30) / 10)

    realization = channel.sample_realization(num_timesteps=num_timesteps)
    H = realization['H']  # (m, n, T)

    # Vectorized SINR/rate computation across all receivers and timesteps.
    # associations is one-hot per RX after pairing; paired_txs[rx] gives serving TX index.
    paired_txs = np.argmax(channel.associations, axis=0)  # (n,)
    rx_indices = np.arange(n)

    # Per-RX/per-time channel gains from paired TXs and all TXs.
    signal_gains = H[paired_txs, rx_indices, :]       # (n, T)
    total_gains = np.sum(H, axis=0)                   # (n, T)
    interference_gains = total_gains - signal_gains   # (n, T)

    signal_power = P_max_watts * signal_gains
    interference_power = P_max_watts * interference_gains

    rates_over_time = np.log2(
        1.0 + signal_power / (interference_power + noise_var)
    ).T  # (T, n)

    ergodic_rates = np.mean(rates_over_time, axis=0)

    return {
        'rates_over_time': rates_over_time,
        'ergodic_rates': ergodic_rates,
        'mean': float(np.mean(ergodic_rates)),
        'min': float(np.min(ergodic_rates)),
        'max': float(np.max(ergodic_rates)),
        'std': float(np.std(ergodic_rates)),
        'percentile_5': float(np.percentile(ergodic_rates, 5)),
        'percentile_10': float(np.percentile(ergodic_rates, 10)),
        'percentile_25': float(np.percentile(ergodic_rates, 25)),
        'percentile_50': float(np.percentile(ergodic_rates, 50)),
        'fairness': float(np.sum(ergodic_rates) ** 2 / (n * np.sum(ergodic_rates ** 2))),
        'P_max_watts': P_max_watts,
        'noise_var': noise_var,
    }


# ---------------------------------------------------------------------------
# Channel generation, caching, and statistics
# ---------------------------------------------------------------------------

def _build_channel_constructor_kwargs(cfg):
    """Return ``(channel_version, ChannelClass, param_overrides)`` from config."""
    channel_config = cfg.get('channel', {})
    channel_version = normalize_channel_version(
        channel_config.get('version'), default='v2',
    )
    ChannelClass = get_channel_class(channel_version, default='v2')

    overrides = sanitize_channel_params(channel_config)
    overrides.pop('deployment_range', None)        # dataset.deployment_range is authoritative
    if channel_version != 'v3':
        overrides.pop('max_recursion_depth', None)  # V1/V2 constructors don't accept it

    return channel_version, ChannelClass, overrides


def _generate_channels_and_rates(cfg, ChannelClass, param_overrides):
    """Generate all channel networks and compute full-power ergodic rates.

    Returns ``(channels, rate_results)`` — parallel lists of channel objects
    and per-network rate statistic dicts.
    """
    channels, rate_results = [], []
    for net_id in tqdm(range(cfg.dataset.num_networks), desc="Generating networks"):
        channel = ChannelClass(
            n_links=cfg.dataset.n_links,
            deployment_range=cfg.dataset.deployment_range,
            seed=cfg.seed + net_id,
            **param_overrides,
        )
        channels.append(channel)

        rates = compute_ergodic_rates(
            channel=channel,
            num_timesteps=cfg.dataset.get('num_timesteps', 200),
            P_max_dBm=cfg.system.P_max_dBm,
            bandwidth_Hz=cfg.system.bandwidth_Hz,
            noise_psd_dBm_Hz=cfg.system.noise_psd_dBm_Hz,
        )
        rate_results.append(rates)

    return channels, rate_results


def _save_channel_cache(channels, cfg, channel_version, param_overrides):
    """Save channels to a content-addressed cache file (validate-or-skip).

    If a cache file already exists with matching metadata the save is skipped.
    If metadata differs a ``RuntimeError`` is raised so stale files are never
    silently overwritten.
    """
    name = cfg.get('name', '')
    channel_config_name = cfg.get('channel_config_name', f'default_{cfg.seed}')

    cache_metadata = build_wra_channel_cache_metadata(
        seed=cfg.seed,
        num_networks=cfg.dataset.num_networks,
        n_links=cfg.dataset.n_links,
        deployment_range=cfg.dataset.deployment_range,
        channel_version=channel_version,
        channel_params=param_overrides,
        channel_config_name=channel_config_name,
    )
    cache_metadata['timestamp'] = datetime.now().isoformat()
    cache_key = cache_metadata['channel_cache_key']

    cache_dir = Path('data/wra_channel_cache') / name
    cache_dir.mkdir(parents=True, exist_ok=True)
    channels_file = cache_dir / f'{cache_key}.pt'

    if channels_file.exists():
        log.info("Cache file already exists: %s — validating …", channels_file)
        try:
            existing = torch.load(channels_file, map_location='cpu')
            existing_meta = canonicalize_channel_cache_metadata(
                existing.get('metadata') if isinstance(existing, dict) else None,
                default_channel_version='v2',
            )
            expected_meta = canonicalize_channel_cache_metadata(
                cache_metadata, default_channel_version='v2',
            )
            mismatches = find_channel_cache_metadata_mismatches(expected_meta, existing_meta)
            if not mismatches:
                log.info("✓ Existing cache matches current config — skipping save.")
                return cache_key
            # Metadata mismatch — refuse to silently overwrite.
            log.warning("!" * 80)
            log.warning("CACHE FILE EXISTS WITH DIFFERENT METADATA!")
            log.warning("!" * 80)
            for key, (exp_v, act_v) in mismatches.items():
                log.warning("  %s: expected=%r  actual=%r", key, exp_v, act_v)
            log.warning("Delete the stale file and re-run, or use a different name.")
            log.warning("  File: %s", channels_file)
            raise RuntimeError(
                f"Cache file {channels_file} exists with mismatched metadata. "
                "Delete it manually and re-run."
            )
        except RuntimeError:
            raise
        except Exception as exc:
            log.warning("Could not validate existing cache (%s); overwriting.", exc)

    torch.save({'metadata': cache_metadata, 'channels': channels}, channels_file)
    log.info("✓ Saved %d channels → %s", len(channels), channels_file)
    return cache_key


def _compute_global_statistics(rate_results, cfg):
    """Aggregate per-network rate results into a summary statistics dict."""
    all_ergodic = np.concatenate([r['ergodic_rates'] for r in rate_results])
    mean_rates = [r['mean'] for r in rate_results]
    min_rates = [r['min'] for r in rate_results]
    p10_list = [r['percentile_10'] for r in rate_results]
    p25_list = [r['percentile_25'] for r in rate_results]

    pct = lambda q: float(np.percentile(all_ergodic, q))  # noqa: E731

    return {
        'num_networks': int(cfg.dataset.num_networks),
        'n_links': int(cfg.dataset.n_links),
        'deployment_range': float(cfg.dataset.deployment_range),
        'P_max_dBm': float(cfg.system.P_max_dBm),
        'bandwidth_Hz': float(cfg.system.bandwidth_Hz),
        'noise_psd_dBm_Hz': float(cfg.system.noise_psd_dBm_Hz),
        'num_timesteps': int(cfg.dataset.get('num_timesteps', 200)),
        'global_statistics': {
            'mean_of_means': float(np.mean(mean_rates)),
            'std_of_means': float(np.std(mean_rates)),
            'mean_of_mins': float(np.mean(min_rates)),
            'std_of_mins': float(np.std(min_rates)),
            'global_min': float(min(min_rates)),
            'global_max_mean': float(max(mean_rates)),
            'global_5th_percentile': pct(5),
            'global_10th_percentile': pct(10),
            'global_25th_percentile': pct(25),
            'global_50th_percentile': pct(50),
            'global_95th_percentile': pct(95),
            'mean_of_10th_percentiles': float(np.mean(p10_list)),
            'mean_of_25th_percentiles': float(np.mean(p25_list)),
        },
        'recommendations': {
            'r_min_easy': pct(5) * 1.1,
            'r_min_moderate': (pct(10) + pct(50)) / 2,
            'r_min_challenging': pct(25),
            'r_min_very_hard': pct(50),
        },
        'per_network_statistics': [
            {
                'network_id': i,
                'mean': float(r['mean']),
                'min': float(r['min']),
                'max': float(r['max']),
                'std': float(r['std']),
                'percentile_5': float(r['percentile_5']),
                'percentile_10': float(r['percentile_10']),
                'percentile_25': float(r['percentile_25']),
                'fairness': float(r['fairness']),
            }
            for i, r in enumerate(rate_results)
        ],
    }


def _log_statistics(stats):
    """Print a concise summary of the global statistics to the log."""
    gs = stats['global_statistics']
    rec = stats['recommendations']

    log.info("Global statistics:")
    log.info("  Mean of means: %.4f ± %.4f  |  Mean of mins: %.4f ± %.4f",
             gs['mean_of_means'], gs['std_of_means'],
             gs['mean_of_mins'], gs['std_of_mins'])
    log.info("  Global min: %.4f  |  5th: %.4f  |  10th: %.4f  |  25th: %.4f",
             gs['global_min'], gs['global_5th_percentile'],
             gs['global_10th_percentile'], gs['global_25th_percentile'])
    log.info("Recommended r_min targets:")
    log.info("  Easy (+10%% above 5th):  %.4f", rec['r_min_easy'])
    log.info("  Moderate (→ median):    %.4f", rec['r_min_moderate'])
    log.info("  Challenging (25th %%):   %.4f", rec['r_min_challenging'])
    log.info("  Very hard (median):     %.4f", rec['r_min_very_hard'])


def _write_summary_files(output_dir, stats, cfg):
    """Write statistics.json and summary.txt to *output_dir*."""
    # JSON
    stats_file = output_dir / 'statistics.json'
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    log.info("✓ Saved statistics → %s", stats_file)

    # Human-readable text
    gs = stats['global_statistics']
    rec = stats['recommendations']
    summary_file = output_dir / 'summary.txt'
    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("WRA DATASET CONFIGURATION ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Configuration: {cfg.get('name', '?')}\n")
        f.write(f"Generated:     {datetime.now():%Y-%m-%d %H:%M:%S}\n\n")

        f.write("Dataset parameters:\n")
        f.write(f"  Networks:         {stats['num_networks']}\n")
        f.write(f"  Links / network:  {stats['n_links']}\n")
        f.write(f"  Deployment range: {stats['deployment_range']} m\n")
        f.write(f"  P_max:            {stats['P_max_dBm']} dBm\n")
        f.write(f"  Bandwidth:        {stats['bandwidth_Hz']} Hz\n")
        f.write(f"  Noise PSD:        {stats['noise_psd_dBm_Hz']} dBm/Hz\n")
        f.write(f"  Timesteps:        {stats['num_timesteps']}\n\n")

        f.write("Global statistics (full-power allocation):\n")
        f.write(f"  Mean of means: {gs['mean_of_means']:.4f} ± {gs['std_of_means']:.4f}\n")
        f.write(f"  Mean of mins:  {gs['mean_of_mins']:.4f} ± {gs['std_of_mins']:.4f}\n")
        f.write(f"  Global min:    {gs['global_min']:.4f}\n")
        f.write(f"  5th  %%ile:     {gs['global_5th_percentile']:.4f}\n")
        f.write(f"  10th %%ile:     {gs['global_10th_percentile']:.4f}\n")
        f.write(f"  Median:        {gs['global_50th_percentile']:.4f}\n\n")

        f.write("Recommended r_min values:\n")
        f.write(f"  Easy   (+10%% above 5th):  {rec['r_min_easy']:.4f}\n")
        f.write(f"  Moderate (→ median):      {rec['r_min_moderate']:.4f}\n")
        f.write(f"  Challenging (25th %%ile):  {rec['r_min_challenging']:.4f}\n")
        f.write(f"  Very hard (median):       {rec['r_min_very_hard']:.4f}\n")
        f.write("=" * 80 + "\n")

    log.info("✓ Saved summary → %s", summary_file)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../../conf/wra_generation", config_name="channel_analysis/wra_small")
def main(cfg: DictConfig):
    """Analyze WRA dataset configuration and cache channels for training."""
    output_dir = Path(HydraConfig.get().runtime.output_dir)
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    _suppress_console_logging_for("graph_signal_diffusion.datasets.wra.channel")

    # Plain-dict copy for visualization helpers that use nested .get()
    config = OmegaConf.to_container(cfg, resolve=True)

    log.info("=" * 80)
    log.info("WRA DATASET CONFIGURATION ANALYZER")
    log.info("=" * 80)
    log.info("Networks: %d × %d links  |  range: %s m  |  seed: %s",
             cfg.dataset.num_networks, cfg.dataset.n_links,
             cfg.dataset.deployment_range, cfg.seed)
    log.info("System: P_max=%s dBm  BW=%s Hz  noise_PSD=%s dBm/Hz",
             cfg.system.P_max_dBm, cfg.system.bandwidth_Hz, cfg.system.noise_psd_dBm_Hz)

    # --- Generate channels and compute full-power rates -----------------------
    channel_version, ChannelClass, param_overrides = _build_channel_constructor_kwargs(cfg)
    log.info("Channel: %s (%s)  overrides=%s",
             channel_version, ChannelClass.__name__, param_overrides or '(none)')

    channels, rate_results = _generate_channels_and_rates(cfg, ChannelClass, param_overrides)

    # --- Visualize example networks -------------------------------------------
    num_examples = min(cfg.get('num_examples', 5), len(channels))
    for i in range(num_examples):
        log.info("Visualizing example network %d …", i)
        visualize_network_deployment(channels[i], i, figures_dir, config)
        visualize_rate_results(channels[i], rate_results[i], i, figures_dir)

    # --- Save channels to cache -----------------------------------------------
    cache_key = _save_channel_cache(channels, cfg, channel_version, param_overrides)
    log.info("Cache key: %s  |  version: %s  |  config_name: %s",
             cache_key, channel_version, cfg.get('channel_config_name', '?'))

    # --- Compute and report statistics ----------------------------------------
    stats = _compute_global_statistics(rate_results, cfg)
    _log_statistics(stats)
    _write_summary_files(output_dir, stats, cfg)

    # --- Global visualizations ------------------------------------------------
    visualize_global_statistics(rate_results, figures_dir, config, cfg.dataset.n_links)

    log.info("=" * 80)
    log.info("ANALYSIS COMPLETE — all results in %s", output_dir)
    log.info("=" * 80)


if __name__ == "__main__":
    main()
