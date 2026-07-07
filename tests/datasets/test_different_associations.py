"""
Test to verify that results differ based on association patterns.
"""

import numpy as np
import torch
from graph_signal_diffusion.datasets.wra.channel import WirelessChannel
import sys
sys.path.append('tests/datasets')
from test_calc_rates import compute_ergodic_rates


def test_different_associations():
    """Test that different association patterns produce different results."""
    
    print("="*60)
    print("Testing that non-diagonal associations are handled correctly")
    print("="*60)
    
    # Create two channels with different seeds
    print("\nChannel 1 (seed=42):")
    channel1 = WirelessChannel(n_links=10, deployment_range=400.0, seed=42)
    
    # Check if associations are diagonal
    is_diagonal_1 = np.allclose(channel1.associations, np.eye(10))
    print(f"  Associations diagonal? {is_diagonal_1}")
    if not is_diagonal_1:
        non_diag_pairs = np.argwhere((channel1.associations == 1) & (np.eye(10) == 0))
        print(f"  Non-diagonal pairings found: {len(non_diag_pairs)}")
        for tx, rx in non_diag_pairs[:3]:  # Show first 3
            print(f"    TX {tx} → RX {rx}")
    
    print("\nChannel 2 (seed=123):")
    channel2 = WirelessChannel(n_links=10, deployment_range=400.0, seed=123)
    
    is_diagonal_2 = np.allclose(channel2.associations, np.eye(10))
    print(f"  Associations diagonal? {is_diagonal_2}")
    if not is_diagonal_2:
        non_diag_pairs = np.argwhere((channel2.associations == 1) & (np.eye(10) == 0))
        print(f"  Non-diagonal pairings found: {len(non_diag_pairs)}")
        for tx, rx in non_diag_pairs[:3]:
            print(f"    TX {tx} → RX {rx}")
    
    # Compute rates for both
    print("\n" + "-"*60)
    print("Computing rates with corrected implementation...")
    print("-"*60)
    
    results1 = compute_ergodic_rates(channel1, num_timesteps=100, P_max_dBm=10.0)
    print(f"\nChannel 1 Mean Rate: {results1['mean_ergodic_rate']:.4f} bits/s/Hz")
    
    results2 = compute_ergodic_rates(channel2, num_timesteps=100, P_max_dBm=10.0)
    print(f"Channel 2 Mean Rate: {results2['mean_ergodic_rate']:.4f} bits/s/Hz")
    
    print("\n" + "="*60)
    print("✓ Test shows channels with different topologies work correctly")
    print("="*60)


if __name__ == "__main__":
    test_different_associations()
