"""
Test script for evaluating full-power strategy across a batch of wireless networks.

This script:
1. Generates B=16 wireless networks with identical configuration
2. Each network has N=50 links
3. Evaluates constant power allocation (P_max) strategy
4. Computes ergodic rates and statistics across the entire batch
5. Creates comprehensive visualizations
"""

import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

from graph_signal_diffusion.datasets.wra.channel import WirelessChannel


def calc_rates(p: torch.Tensor, gamma: torch.Tensor, h: torch.Tensor, noise_var: float) -> torch.Tensor:
    """
    Calculate achievable rates for wireless users.
    
    IMPORTANT: The association matrix can be non-diagonal! TX i may serve RX j where i ≠ j.
    
    Parameters
    ----------
    p : torch.Tensor
        Transmit power levels, shape (batch, m, 1) or (m, 1)
    gamma : torch.Tensor
        User scheduling decisions, shape (batch, n, 1) or (n, 1)
    h : torch.Tensor
        Weighted adjacency matrix containing channel gains, shape (batch, m+n, m+n) or (m+n, m+n)
    noise_var : float
        Noise variance (linear scale)
    
    Returns
    -------
    rates : torch.Tensor
        Achievable rates in bits/s/Hz, shape (batch, n, 1) or (n, 1)
    """
    # Handle both batched and unbatched inputs
    if p.dim() == 2:
        p = p.unsqueeze(0)
    if gamma.dim() == 2:
        gamma = gamma.unsqueeze(0)
    if h.dim() == 2:
        h = h.unsqueeze(0)
    
    b = h.shape[0]  # batch size
    p = p.view(b, -1, 1)
    gamma = gamma.view(b, -1, 1)
    m = p.shape[1]  # number of transmitters
    
    # Compute power allocation: outer product of p (m,1) and gamma^T (1,n)
    combined_p_gamma = torch.bmm(p, torch.transpose(gamma, 1, 2))
    
    # Signal power
    signal = torch.sum(combined_p_gamma * h[:, :m, m:], dim=1)
    
    # Interference power
    interference = torch.sum(combined_p_gamma * torch.transpose(h[:, m:, :m], 1, 2), dim=1)
    
    # Shannon capacity
    rates = torch.log2(1 + signal / (noise_var + interference)).view(-1, 1)
    
    return rates


def generate_batch_of_channels(
    batch_size: int,
    n_links: int,
    deployment_range: float = 1000.0,
    seed_start: int = 42,
    **channel_kwargs
) -> list:
    """
    Generate a batch of wireless channels with identical configuration.
    
    Parameters
    ----------
    batch_size : int
        Number of channels to generate (B)
    n_links : int
        Number of links per channel (N)
    deployment_range : float
        Deployment area side length in meters
    seed_start : int
        Starting seed (each channel gets seed_start + b)
    **channel_kwargs
        Additional arguments for WirelessChannel
    
    Returns
    -------
    channels : list
        List of B WirelessChannel instances
    """
    print(f"Generating batch of {batch_size} channels with {n_links} links each...")
    print(f"Deployment range: {deployment_range}m × {deployment_range}m")
    
    channels = []
    for b in tqdm(range(batch_size), desc="Creating channels"):
        channel = WirelessChannel(
            n_links=n_links,
            deployment_range=deployment_range,
            seed=seed_start + b,
            **channel_kwargs
        )
        channels.append(channel)
    
    print(f"✓ Generated {batch_size} channels\n")
    return channels


def evaluate_full_power_batch(
    channels: list,
    num_timesteps: int,
    P_max_dBm: float = 10.0,
    bandwidth_Hz: float = 10e6,
    noise_psd_dBm_Hz: float = -174.0
) -> dict:
    """
    Evaluate full-power strategy across a batch of channels.
    
    Parameters
    ----------
    channels : list
        List of WirelessChannel instances
    num_timesteps : int
        Number of time steps per channel
    P_max_dBm : float
        Maximum transmit power in dBm
    bandwidth_Hz : float
        Bandwidth in Hz
    noise_psd_dBm_Hz : float
        Noise PSD in dBm/Hz
    
    Returns
    -------
    results : dict
        Comprehensive results dictionary
    """
    batch_size = len(channels)
    n_links = channels[0].n_links
    
    # Convert power parameters
    P_max_watts = 10 ** ((P_max_dBm - 30) / 10)
    noise_power_dBm = noise_psd_dBm_Hz + 10 * np.log10(bandwidth_Hz)
    noise_var = 10 ** ((noise_power_dBm - 30) / 10)
    snr_dB = P_max_dBm - noise_power_dBm
    
    print(f"{'='*70}")
    print(f"EVALUATING FULL-POWER STRATEGY")
    print(f"{'='*70}")
    print(f"Batch size (B):     {batch_size}")
    print(f"Links per net (N):  {n_links}")
    print(f"Timesteps (T):      {num_timesteps}")
    print(f"P_max:              {P_max_dBm} dBm ({P_max_watts*1000:.3f} mW)")
    print(f"Bandwidth:          {bandwidth_Hz/1e6:.1f} MHz")
    print(f"Noise power:        {noise_power_dBm:.1f} dBm")
    print(f"SNR:                {snr_dB:.1f} dB")
    print(f"{'='*70}\n")
    
    # Storage for results
    all_rates_over_time = []  # List of (T, N) arrays
    all_ergodic_rates = []    # List of (N,) arrays
    network_statistics = []   # Per-network stats
    
    # Constant power allocation
    p = P_max_watts * torch.ones(n_links, 1)
    gamma = torch.ones(n_links, 1)
    
    # Evaluate each channel
    print("Evaluating each network...")
    for b, channel in enumerate(tqdm(channels, desc="Networks")):
        # Sample time-varying realization
        realization = channel.sample_realization(num_timesteps=num_timesteps)
        H = realization['H']  # Shape: (m, n, T)
        
        # Storage for this channel
        rates_over_time = np.zeros((num_timesteps, n_links))
        
        # Compute rates at each timestep
        for t in range(num_timesteps):
            h_t = H[:, :, t]  # Shape: (m, n)
            
            # Build weighted adjacency matrix
            h_adj = np.zeros((n_links + n_links, n_links + n_links))
            
            # TX→RX: Signal (only paired links)
            h_adj[:n_links, n_links:] = channel.associations * h_t
            
            # RX←TX: Interference (non-paired links)
            # CRITICAL: Transpose AFTER element-wise multiplication
            h_adj[n_links:, :n_links] = ((1 - channel.associations) * h_t).T
            
            h_adj = torch.from_numpy(h_adj).float()
            
            # Compute rates
            rates = calc_rates(p, gamma, h_adj, noise_var)
            rates_over_time[t, :] = rates.squeeze().numpy()
        
        # Compute ergodic rates for this network
        ergodic_rates = np.mean(rates_over_time, axis=0)
        
        # Network-level statistics
        network_stats = {
            'network_id': b,
            'mean_ergodic_rate': np.mean(ergodic_rates),
            'min_ergodic_rate': np.min(ergodic_rates),
            'max_ergodic_rate': np.max(ergodic_rates),
            'percentile_5_rate': np.percentile(ergodic_rates, 5),
            'std_ergodic_rate': np.std(ergodic_rates),
            'fairness_index': (np.sum(ergodic_rates) ** 2) / (n_links * np.sum(ergodic_rates ** 2)),
            'rate_range': np.max(ergodic_rates) - np.min(ergodic_rates),
        }
        
        all_rates_over_time.append(rates_over_time)
        all_ergodic_rates.append(ergodic_rates)
        network_statistics.append(network_stats)
    
    # Aggregate statistics across batch
    mean_rates_per_network = [s['mean_ergodic_rate'] for s in network_statistics]
    min_rates_per_network = [s['min_ergodic_rate'] for s in network_statistics]
    percentile_5_rates_per_network = [s['percentile_5_rate'] for s in network_statistics]
    fairness_per_network = [s['fairness_index'] for s in network_statistics]
    
    # Overall statistics
    all_ergodic_rates_flat = np.concatenate(all_ergodic_rates)
    
    batch_statistics = {
        # Per-network aggregates
        'mean_rate_across_networks': np.mean(mean_rates_per_network),
        'std_mean_rate_across_networks': np.std(mean_rates_per_network),
        'min_mean_rate': np.min(mean_rates_per_network),
        'max_mean_rate': np.max(mean_rates_per_network),
        
        'mean_min_rate_across_networks': np.mean(min_rates_per_network),
        'worst_min_rate': np.min(min_rates_per_network),
        'best_min_rate': np.max(min_rates_per_network),
        
        'mean_5th_percentile_across_networks': np.mean(percentile_5_rates_per_network),
        'worst_5th_percentile': np.min(percentile_5_rates_per_network),
        'best_5th_percentile': np.max(percentile_5_rates_per_network),
        
        'mean_fairness_across_networks': np.mean(fairness_per_network),
        'std_fairness_across_networks': np.std(fairness_per_network),
        
        # Overall across all users in all networks
        'overall_mean_rate': np.mean(all_ergodic_rates_flat),
        'overall_std_rate': np.std(all_ergodic_rates_flat),
        'overall_min_rate': np.min(all_ergodic_rates_flat),
        'overall_max_rate': np.max(all_ergodic_rates_flat),
    }
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"BATCH RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"\nPer-Network Statistics (averaged over {batch_size} networks):")
    print(f"  Mean rate:        {batch_statistics['mean_rate_across_networks']:.4f} ± {batch_statistics['std_mean_rate_across_networks']:.4f} bits/s/Hz")
    print(f"    Range:          [{batch_statistics['min_mean_rate']:.4f}, {batch_statistics['max_mean_rate']:.4f}]")
    print(f"  Min rate:         {batch_statistics['mean_min_rate_across_networks']:.4f} bits/s/Hz")
    print(f"    Worst case:     {batch_statistics['worst_min_rate']:.4f} bits/s/Hz")
    print(f"  5th percentile:   {batch_statistics['mean_5th_percentile_across_networks']:.4f} bits/s/Hz")
    print(f"    Worst case:     {batch_statistics['worst_5th_percentile']:.4f} bits/s/Hz")
    print(f"  Fairness index:   {batch_statistics['mean_fairness_across_networks']:.4f} ± {batch_statistics['std_fairness_across_networks']:.4f}")
    
    print(f"\nOverall Statistics (all {batch_size * n_links} users):")
    print(f"  Mean rate:        {batch_statistics['overall_mean_rate']:.4f} bits/s/Hz")
    print(f"  Std dev:          {batch_statistics['overall_std_rate']:.4f} bits/s/Hz")
    print(f"  Min rate:         {batch_statistics['overall_min_rate']:.4f} bits/s/Hz")
    print(f"  Max rate:         {batch_statistics['overall_max_rate']:.4f} bits/s/Hz")
    print(f"{'='*70}\n")
    
    return {
        'all_rates_over_time': all_rates_over_time,
        'all_ergodic_rates': all_ergodic_rates,
        'network_statistics': network_statistics,
        'batch_statistics': batch_statistics,
        'power_params': {
            'P_max_dBm': P_max_dBm,
            'P_max_watts': P_max_watts,
            'noise_var': noise_var,
            'snr_dB': snr_dB,
        }
    }


def visualize_batch_results(channels: list, results: dict, save_dir: str = "tests/figs/wra_channel"):
    """
    Create comprehensive visualizations of batch results.
    
    Parameters
    ----------
    channels : list
        List of channel instances
    results : dict
        Results from evaluate_full_power_batch
    save_dir : str
        Directory to save figures
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    batch_size = len(channels)
    n_links = channels[0].n_links
    
    network_stats = results['network_statistics']
    batch_stats = results['batch_statistics']
    
    # Create comprehensive figure with 6 subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # ========== Plot 1: Mean rates per network ==========
    ax1 = fig.add_subplot(gs[0, 0])
    mean_rates = [s['mean_ergodic_rate'] for s in network_stats]
    network_ids = np.arange(1, batch_size + 1)
    
    bars = ax1.bar(network_ids, mean_rates, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axhline(batch_stats['mean_rate_across_networks'], 
                color='red', linestyle='--', linewidth=2, label='Batch mean')
    ax1.set_xlabel('Network ID', fontsize=11)
    ax1.set_ylabel('Mean Ergodic Rate (bits/s/Hz)', fontsize=11)
    ax1.set_title(f'Mean Rate per Network (B={batch_size})', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # ========== Plot 2: Min rates per network ==========
    ax2 = fig.add_subplot(gs[0, 1])
    min_rates = [s['min_ergodic_rate'] for s in network_stats]
    
    bars = ax2.bar(network_ids, min_rates, color='coral', alpha=0.7, edgecolor='black')
    ax2.axhline(batch_stats['mean_min_rate_across_networks'],
                color='red', linestyle='--', linewidth=2, label='Batch mean')
    ax2.axhline(batch_stats['worst_min_rate'],
                color='orange', linestyle=':', linewidth=2, label='Worst case')
    ax2.set_xlabel('Network ID', fontsize=11)
    ax2.set_ylabel('Min Ergodic Rate (bits/s/Hz)', fontsize=11)
    ax2.set_title('Min Rate per Network', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # ========== Plot 3: 5th percentile rates per network ==========
    ax3 = fig.add_subplot(gs[0, 2])
    percentile_5_rates = [s['percentile_5_rate'] for s in network_stats]
    
    bars = ax3.bar(network_ids, percentile_5_rates, color='orchid', alpha=0.7, edgecolor='black')
    ax3.axhline(batch_stats['mean_5th_percentile_across_networks'],
                color='red', linestyle='--', linewidth=2, label='Batch mean')
    ax3.axhline(batch_stats['worst_5th_percentile'],
                color='orange', linestyle=':', linewidth=2, label='Worst case')
    ax3.set_xlabel('Network ID', fontsize=11)
    ax3.set_ylabel('5th Percentile Rate (bits/s/Hz)', fontsize=11)
    ax3.set_title('5th Percentile Rate per Network', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # ========== Plot 4: Fairness index per network ==========
    ax4 = fig.add_subplot(gs[1, 0])
    fairness = [s['fairness_index'] for s in network_stats]
    
    bars = ax4.bar(network_ids, fairness, color='mediumseagreen', alpha=0.7, edgecolor='black')
    ax4.axhline(batch_stats['mean_fairness_across_networks'],
                color='red', linestyle='--', linewidth=2, label='Batch mean')
    ax4.axhline(1.0, color='gray', linestyle=':', linewidth=1, alpha=0.5, label='Perfect fairness')
    ax4.set_xlabel('Network ID', fontsize=11)
    ax4.set_ylabel('Fairness Index', fontsize=11)
    ax4.set_title('Fairness Index per Network', fontsize=12, fontweight='bold')
    ax4.set_ylim([0, 1.05])
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # ========== Plot 5: Overall rate distribution (all users) ==========
    ax5 = fig.add_subplot(gs[1, 1])
    all_ergodic_rates_flat = np.concatenate(results['all_ergodic_rates'])
    
    ax5.hist(all_ergodic_rates_flat, bins=50, color='skyblue', alpha=0.7, edgecolor='black')
    ax5.axvline(batch_stats['overall_mean_rate'],
                color='red', linestyle='--', linewidth=2, label='Mean')
    ax5.axvline(batch_stats['overall_min_rate'],
                color='orange', linestyle=':', linewidth=2, label='Min')
    ax5.set_xlabel('Ergodic Rate (bits/s/Hz)', fontsize=11)
    ax5.set_ylabel('Frequency', fontsize=11)
    ax5.set_title(f'Overall Rate Distribution ({batch_size * n_links} users)', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # ========== Plot 6: CDF of overall rates ==========
    ax6 = fig.add_subplot(gs[1, 2])
    sorted_rates = np.sort(all_ergodic_rates_flat)
    cdf = np.arange(1, len(sorted_rates) + 1) / len(sorted_rates)
    
    ax6.plot(sorted_rates, cdf, linewidth=2, color='darkblue')
    ax6.axvline(batch_stats['overall_mean_rate'],
                color='red', linestyle='--', linewidth=2, label='Mean')
    ax6.axhline(0.05, color='gray', linestyle=':', alpha=0.5)
    ax6.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax6.set_xlabel('Ergodic Rate (bits/s/Hz)', fontsize=11)
    ax6.set_ylabel('CDF', fontsize=11)
    ax6.set_title('CDF of Ergodic Rates', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    # ========== Plot 7: Rates over time for one sample network ==========
    ax7 = fig.add_subplot(gs[2, :2])
    sample_network_idx = 0
    rates_over_time = results['all_rates_over_time'][sample_network_idx]
    T = rates_over_time.shape[0]
    
    # Plot subset of users for clarity
    users_to_plot = min(20, n_links)
    for i in range(users_to_plot):
        ax7.plot(rates_over_time[:, i], alpha=0.6, linewidth=0.8)
    
    ax7.set_xlabel('Time Step', fontsize=11)
    ax7.set_ylabel('Instantaneous Rate (bits/s/Hz)', fontsize=11)
    ax7.set_title(f'Rates Over Time (Network {sample_network_idx+1}, {users_to_plot}/{n_links} users shown)',
                  fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3)
    
    # ========== Plot 8: Box plot of rates per network ==========
    ax8 = fig.add_subplot(gs[2, 2])
    
    # Sample a subset of networks for clarity
    networks_to_plot = min(8, batch_size)
    sampled_rates = [results['all_ergodic_rates'][i] for i in range(0, batch_size, batch_size // networks_to_plot)][:networks_to_plot]
    sampled_ids = list(range(1, batch_size + 1, batch_size // networks_to_plot))[:networks_to_plot]
    
    bp = ax8.boxplot(sampled_rates, positions=sampled_ids, widths=batch_size/(networks_to_plot*2),
                     patch_artist=True, showmeans=True)
    
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    ax8.set_xlabel('Network ID (sample)', fontsize=11)
    ax8.set_ylabel('Ergodic Rate (bits/s/Hz)', fontsize=11)
    ax8.set_title(f'Rate Distribution per Network (sample)', fontsize=12, fontweight='bold')
    ax8.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle(f'Full-Power Strategy: Batch Evaluation (B={batch_size}, N={n_links}, P_max={results["power_params"]["P_max_dBm"]} dBm)',
                 fontsize=14, fontweight='bold', y=0.995)
    
    # Save figure
    save_file = save_path / f"batch_full_power_B{batch_size}_N{n_links}.png"
    plt.savefig(save_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved batch visualization to {save_file}")
    plt.close()


def main():
    """Main test function for batch evaluation."""
    
    # Configuration
    batch_size = 16
    n_links = 50
    num_timesteps = 200
    P_max_dBm = 10.0
    deployment_range = 700.0  # Reduced from 1000m for higher interference
    seed_start = 42
    
    print("="*70)
    print("BATCH EVALUATION: FULL-POWER STRATEGY")
    print("="*70)
    print(f"Configuration:")
    print(f"  Batch size (B):      {batch_size}")
    print(f"  Links per net (N):   {n_links}")
    print(f"  Timesteps (T):       {num_timesteps}")
    print(f"  Deployment range:    {deployment_range}m")
    print(f"  Starting seed:       {seed_start}")
    print("="*70 + "\n")
    
    # Generate batch of channels
    channels = generate_batch_of_channels(
        batch_size=batch_size,
        n_links=n_links,
        deployment_range=deployment_range,
        seed_start=seed_start
    )
    
    # Evaluate full-power strategy
    results = evaluate_full_power_batch(
        channels=channels,
        num_timesteps=num_timesteps,
        P_max_dBm=P_max_dBm
    )
    
    # Create visualizations
    print("Creating visualizations...")
    visualize_batch_results(channels, results)
    
    print("\n" + "="*70)
    print("✓ BATCH EVALUATION COMPLETE!")
    print("="*70)
    
    return channels, results


if __name__ == "__main__":
    channels, results = main()
