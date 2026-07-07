"""
Test script for calc_rates implementation with constant power allocation.

This script:
1. Generates a wireless channel using the new WirelessChannel class
2. Implements a calc_rates function based on the legacy code
3. Tests constant power allocation (P_max = 10 dBm)
4. Computes ergodic rates over time
"""

import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt

from graph_signal_diffusion.datasets.wra.channel import WirelessChannel


def calc_rates(p: torch.Tensor, gamma: torch.Tensor, h: torch.Tensor, noise_var: float) -> torch.Tensor:
    """
    Calculate achievable rates for wireless users.
    
    This is based on the legacy calc_rates function from channel_utils.py.
    The key insight is that rates are computed using Shannon capacity:
        R = log2(1 + SINR)
    where SINR = Signal / (Noise + Interference)
    
    IMPORTANT: The association matrix can be non-diagonal! TX i may serve RX j where i ≠ j.
    
    Parameters
    ----------
    p : torch.Tensor
        Transmit power levels, shape (batch, m, 1) or (m, 1)
        where m is number of transmitters
    gamma : torch.Tensor
        User scheduling decisions, shape (batch, n, 1) or (n, 1)
        where n is number of receivers (gamma[i]=1 means RX i is scheduled)
    h : torch.Tensor
        Weighted adjacency matrix containing channel gains, shape (batch, m+n, m+n) or (m+n, m+n)
        - h[:m, m:] contains TX→RX channel gains (desired signals)
        - h[m:, :m] contains RX←TX channel gains (interference)
    noise_var : float
        Noise variance (linear scale)
    
    Returns
    -------
    rates : torch.Tensor
        Achievable rates in bits/s/Hz, shape (batch, n, 1) or (n, 1)
    
    Notes
    -----
    The weighted adjacency matrix h is structured as:
        h = [TX-TX   TX-RX]
            [RX-TX   RX-RX]
    
    For rate calculation:
    - Signal power at RX j from its paired TX: p_i * h[i, m+j] where TX i serves RX j
    - Interference at RX j: sum over all other TXs k≠i: p_k * h[m+j, k]
    """
    # Handle both batched and unbatched inputs
    if p.dim() == 2:
        p = p.unsqueeze(0)  # Add batch dimension
    if gamma.dim() == 2:
        gamma = gamma.unsqueeze(0)
    if h.dim() == 2:
        h = h.unsqueeze(0)
    
    b = h.shape[0]  # batch size
    p = p.view(b, -1, 1)
    gamma = gamma.view(b, -1, 1)
    m = p.shape[1]  # number of transmitters
    
    # Compute power allocation: outer product of p (m,1) and gamma^T (1,n)
    # Result: (b, m, n) matrix where entry [i,j] = p_i * gamma_j
    combined_p_gamma = torch.bmm(p, torch.transpose(gamma, 1, 2))
    
    # Signal power: For each RX, sum the power from all TXs weighted by h[:, :m, m:]
    # h[:, :m, m:] has shape (b, m, n) containing TX→RX channel gains
    # signal[j] = sum_i (p_i * gamma_j * h[i, m+j])
    signal = torch.sum(combined_p_gamma * h[:, :m, m:], dim=1)
    
    # Interference power: For each RX, sum the interference from all TXs
    # h[:, m:, :m] has shape (b, n, m) containing RX←TX channel gains (interference)
    # We transpose to get (b, m, n) for compatibility
    # interference[j] = sum_i (p_i * gamma_j * h[m+j, i]) for i ≠ serving TX
    interference = torch.sum(combined_p_gamma * torch.transpose(h[:, m:, :m], 1, 2), dim=1)
    
    # Shannon capacity: R = log2(1 + S/(N+I))
    rates = torch.log2(1 + signal / (noise_var + interference)).view(-1, 1)
    
    return rates


def compute_ergodic_rates(channel: WirelessChannel,
                         num_timesteps: int,
                         P_max_dBm: float = 10.0,
                         bandwidth_Hz: float = 10e6,
                         noise_psd_dBm_Hz: float = -174.0) -> dict:
    """
    Compute ergodic rates using constant power allocation strategy.
    
    Parameters
    ----------
    channel : WirelessChannel
        Wireless channel instance
    num_timesteps : int
        Number of time steps to simulate
    P_max_dBm : float
        Maximum transmit power in dBm (default: 10 dBm)
    bandwidth_Hz : float
        Bandwidth in Hz (default: 10 MHz)
    noise_psd_dBm_Hz : float
        Noise power spectral density in dBm/Hz (default: -174 dBm/Hz)
    
    Returns
    -------
    results : dict
        Dictionary containing:
        - 'rates_over_time': (T, n) array of rates at each timestep
        - 'ergodic_rates': (n,) array of time-averaged rates per user
        - 'mean_ergodic_rate': scalar average over all users
        - 'min_ergodic_rate': scalar minimum rate over all users
        - 'fairness_index': Jain's fairness index
        - 'P_max_watts': Maximum power in watts
        - 'noise_var': Noise variance
        - 'snr_dB': SNR in dB
    """
    n = channel.n_links
    m = n  # Assuming one-to-one pairing
    
    # Convert power parameters
    P_max_watts = 10 ** ((P_max_dBm - 30) / 10)  # Convert dBm to Watts
    noise_power_dBm = noise_psd_dBm_Hz + 10 * np.log10(bandwidth_Hz)  # Total noise power
    noise_var = 10 ** ((noise_power_dBm - 30) / 10)  # Convert dBm to Watts
    snr_dB = P_max_dBm - noise_power_dBm
    
    print(f"\nPower Parameters:")
    print(f"  P_max: {P_max_dBm} dBm = {P_max_watts*1000:.3f} mW")
    print(f"  Noise PSD: {noise_psd_dBm_Hz} dBm/Hz")
    print(f"  Bandwidth: {bandwidth_Hz/1e6:.1f} MHz")
    print(f"  Noise Power: {noise_power_dBm:.1f} dBm = {noise_var*1000:.6f} mW")
    print(f"  SNR: {snr_dB:.1f} dB")
    
    # Sample time-varying channel realization
    print(f"\nSampling channel over {num_timesteps} timesteps...")
    realization = channel.sample_realization(num_timesteps=num_timesteps)
    H = realization['H']  # Shape: (m, n, T) - channel gains (squared magnitude)
    
    # Constant power allocation: all transmitters use P_max
    p = P_max_watts * torch.ones(m, 1)
    
    # Storage for rates over time
    rates_over_time = np.zeros((num_timesteps, n))
    
    # Compute rates at each timestep
    print(f"Computing rates over time...")
    for t in range(num_timesteps):
        # Get channel gains at time t
        h_t = H[:, :, t]  # Shape: (m, n)
        
        # Build weighted adjacency matrix for calc_rates
        # Format: [TX-TX   TX-RX]
        #         [RX-TX   RX-RX]
        h_adj = np.zeros((m + n, m + n))
        
        # TX→RX: Use association matrix (only paired links have signal)
        # h_adj[i, m+j] = associations[i,j] * h_t[i,j]
        h_adj[:m, m:] = channel.associations * h_t
        
        # RX←TX: Interference from all TXs (use complement of associations)
        # h_adj[m+j, i] = (1 - associations[i,j]) * h_t[i,j]
        # Must transpose AFTER element-wise multiplication to get correct ordering
        h_adj[m:, :m] = ((1 - channel.associations) * h_t).T
        
        h_adj = torch.from_numpy(h_adj).float()
        
        # Schedule all receivers (gamma = 1 for all)
        gamma = torch.ones(n, 1)
        
        # Compute rates
        rates = calc_rates(p, gamma, h_adj, noise_var)
        rates_over_time[t, :] = rates.squeeze().numpy()
    
    # Compute ergodic rates (time average)
    ergodic_rates = np.mean(rates_over_time, axis=0)
    
    # Statistics
    mean_ergodic_rate = np.mean(ergodic_rates)
    min_ergodic_rate = np.min(ergodic_rates)
    max_ergodic_rate = np.max(ergodic_rates)
    
    # Jain's fairness index: (sum x_i)^2 / (n * sum x_i^2)
    fairness_index = (np.sum(ergodic_rates) ** 2) / (n * np.sum(ergodic_rates ** 2))
    
    print(f"\n{'='*60}")
    print(f"ERGODIC RATE RESULTS (Constant Power P_max={P_max_dBm} dBm)")
    print(f"{'='*60}")
    print(f"Mean ergodic rate:  {mean_ergodic_rate:.4f} bits/s/Hz")
    print(f"Min ergodic rate:   {min_ergodic_rate:.4f} bits/s/Hz")
    print(f"Max ergodic rate:   {max_ergodic_rate:.4f} bits/s/Hz")
    print(f"Std dev:            {np.std(ergodic_rates):.4f} bits/s/Hz")
    print(f"Fairness index:     {fairness_index:.4f}")
    print(f"{'='*60}\n")
    
    return {
        'rates_over_time': rates_over_time,
        'ergodic_rates': ergodic_rates,
        'mean_ergodic_rate': mean_ergodic_rate,
        'min_ergodic_rate': min_ergodic_rate,
        'max_ergodic_rate': max_ergodic_rate,
        'fairness_index': fairness_index,
        'P_max_watts': P_max_watts,
        'noise_var': noise_var,
        'snr_dB': snr_dB,
    }


def visualize_results(channel: WirelessChannel, results: dict, save_dir: str = "tests/figs/wra_channel"):
    """Create visualizations of rate results."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    rates_over_time = results['rates_over_time']
    ergodic_rates = results['ergodic_rates']
    T, n = rates_over_time.shape
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Rates over time for all users
    ax = axes[0, 0]
    for i in range(n):
        ax.plot(rates_over_time[:, i], alpha=0.7, linewidth=0.8, label=f'User {i+1}' if n <= 10 else '')
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Instantaneous Rate (bits/s/Hz)', fontsize=12)
    ax.set_title(f'Rates Over Time for All {n} Users', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    if n <= 10:
        ax.legend(loc='best', fontsize=8)
    
    # Plot 2: Ergodic rates per user
    ax = axes[0, 1]
    user_indices = np.arange(1, n + 1)
    bars = ax.bar(user_indices, ergodic_rates, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axhline(results['mean_ergodic_rate'], color='red', linestyle='--', linewidth=2, label='Mean')
    ax.axhline(results['min_ergodic_rate'], color='orange', linestyle='--', linewidth=2, label='Min')
    ax.set_xlabel('User Index', fontsize=12)
    ax.set_ylabel('Ergodic Rate (bits/s/Hz)', fontsize=12)
    ax.set_title('Ergodic Rates per User', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Rate distribution histogram
    ax = axes[1, 0]
    ax.hist(rates_over_time.flatten(), bins=50, color='skyblue', alpha=0.7, edgecolor='black')
    ax.axvline(results['mean_ergodic_rate'], color='red', linestyle='--', linewidth=2, label='Mean Ergodic')
    ax.set_xlabel('Instantaneous Rate (bits/s/Hz)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Instantaneous Rates', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Cumulative distribution of ergodic rates
    ax = axes[1, 1]
    sorted_rates = np.sort(ergodic_rates)
    cdf = np.arange(1, n + 1) / n
    ax.plot(sorted_rates, cdf, marker='o', linewidth=2, markersize=5, color='darkblue')
    ax.axvline(results['mean_ergodic_rate'], color='red', linestyle='--', linewidth=2, label='Mean')
    ax.axvline(results['min_ergodic_rate'], color='orange', linestyle='--', linewidth=2, label='Min')
    ax.set_xlabel('Ergodic Rate (bits/s/Hz)', fontsize=12)
    ax.set_ylabel('CDF', fontsize=12)
    ax.set_title('CDF of Ergodic Rates', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    save_file = save_path / f"constant_power_rates_n{n}.png"
    plt.savefig(save_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to {save_file}")
    plt.close()


def main():
    """Main test function."""
    print("="*60)
    print("TESTING calc_rates WITH CONSTANT POWER ALLOCATION")
    print("="*60)
    
    # Test configuration
    n_links = 20
    num_timesteps = 200
    P_max_dBm = 10.0
    seed = 42
    
    # Create wireless channel
    print(f"\nCreating wireless channel with {n_links} links...")
    channel = WirelessChannel(
        n_links=n_links,
        deployment_range=600.0,
        seed=seed
    )
    
    # Compute ergodic rates
    results = compute_ergodic_rates(
        channel=channel,
        num_timesteps=num_timesteps,
        P_max_dBm=P_max_dBm
    )
    
    # Visualize results
    print("Creating visualizations...")
    visualize_results(channel, results)
    
    print("\n✓ Test complete!")
    
    # Return results for further analysis
    return channel, results


if __name__ == "__main__":
    channel, results = main()
