"""
Test to verify WirelessChannelV3 correctness:
1. Associations and optimal pairing are correct
2. Seed reproducibility works properly
"""

import sys
import os
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(project_root, 'src'))

from graph_signal_diffusion.datasets.wra.channel import (
    WirelessChannel, WirelessChannelV2, WirelessChannelV3
)


def test_associations_correctness():
    """Test that associations are correctly computed after subdivision."""
    print("="*70)
    print("TEST 1: Associations and Pairing Correctness")
    print("="*70)
    
    channel = WirelessChannelV3(
        n_links=40,
        deployment_range=150.0,
        min_tx_tx_distance=12.0,
        max_tx_rx_distance=45.0,
        seed=999
    )
    
    # Check associations matrix properties
    print("\n1. Checking associations matrix properties:")
    print(f"   Shape: {channel.associations.shape}")
    print(f"   Expected: ({channel.n_links}, {channel.n_links})")
    
    # Each TX should be paired with exactly one RX
    txs_per_tx = np.sum(channel.associations, axis=1)
    print(f"\n2. TX pairing: Each TX paired with exactly 1 RX?")
    print(f"   Min: {np.min(txs_per_tx)}, Max: {np.max(txs_per_tx)}")
    assert np.all(txs_per_tx == 1), "Each TX must pair with exactly one RX"
    print(f"   ✓ PASS: All TXs paired with exactly 1 RX")
    
    # Each RX should be paired with exactly one TX
    rxs_per_rx = np.sum(channel.associations, axis=0)
    print(f"\n3. RX pairing: Each RX paired with exactly 1 TX?")
    print(f"   Min: {np.min(rxs_per_rx)}, Max: {np.max(rxs_per_rx)}")
    assert np.all(rxs_per_rx == 1), "Each RX must pair with exactly one TX"
    print(f"   ✓ PASS: All RXs paired with exactly 1 TX")
    
    # Check tx_rx_pairs consistency
    print(f"\n4. Checking tx_rx_pairs consistency:")
    print(f"   Number of pairs: {len(channel.tx_rx_pairs)}")
    assert len(channel.tx_rx_pairs) == channel.n_links
    print(f"   ✓ PASS: Correct number of pairs")
    
    # Verify associations matrix matches tx_rx_pairs
    associations_from_pairs = np.zeros_like(channel.associations)
    for tx_idx, rx_idx in channel.tx_rx_pairs:
        associations_from_pairs[tx_idx, rx_idx] = True
    
    assert np.all(channel.associations == associations_from_pairs)
    print(f"   ✓ PASS: Associations matrix matches tx_rx_pairs")
    
    # Check distance constraints
    print(f"\n5. Checking distance constraints:")
    paired_distances = []
    for tx_idx, rx_idx in channel.tx_rx_pairs:
        dist = np.linalg.norm(channel.tx_locations[tx_idx] - channel.rx_locations[rx_idx])
        paired_distances.append(dist)
    
    paired_distances = np.array(paired_distances)
    violations = np.sum(paired_distances > channel.max_tx_rx_distance)
    print(f"   Mean distance: {np.mean(paired_distances):.2f}m")
    print(f"   Max distance: {np.max(paired_distances):.2f}m")
    print(f"   Violations: {violations}/{channel.n_links}")
    
    if violations == 0:
        print(f"   ✓ PASS: All pairs satisfy max_tx_rx_distance constraint")
    else:
        print(f"   ⚠ WARNING: {violations} pairs violate constraint")
    
    # Check large-scale fading matrix
    print(f"\n6. Checking large-scale fading:")
    print(f"   Shape: {channel.large_scale_fading.shape}")
    print(f"   Expected: ({channel.n_links}, {channel.n_links})")
    assert channel.large_scale_fading.shape == (channel.n_links, channel.n_links)
    print(f"   ✓ PASS: Large-scale fading has correct shape")
    
    # Check for NaN or invalid values
    assert not np.any(np.isnan(channel.large_scale_fading))
    assert not np.any(np.isinf(channel.large_scale_fading))
    print(f"   ✓ PASS: No NaN or Inf values in large-scale fading")
    
    # Check shadowing matrix
    print(f"\n7. Checking shadowing matrix:")
    print(f"   Shape: {channel.shadowing_db.shape}")
    if channel.shadowing_db_deployment is not None:
        # Check if it's block diagonal (indicates subdivision was used)
        off_diagonal = channel.shadowing_db[~np.eye(channel.n_links, dtype=bool)]
        num_zeros = np.sum(off_diagonal == 0)
        total_off_diag = len(off_diagonal)
        print(f"   Off-diagonal zeros: {num_zeros}/{total_off_diag} ({100*num_zeros/total_off_diag:.1f}%)")
        
        if num_zeros > total_off_diag * 0.5:
            print(f"   ⚠ WARNING: Shadowing appears block-diagonal (subdivision artifact)")
        else:
            print(f"   ✓ Shadowing has cross-subnetwork values")
    
    print("\n" + "="*70)
    print("✓ TEST 1 PASSED: Associations and pairing are correct")
    print("="*70)
    
    return channel


def test_seed_reproducibility():
    """Test that same seed produces identical results."""
    print("\n" + "="*70)
    print("TEST 2: Seed Reproducibility")
    print("="*70)
    
    seed = 12345
    n_links = 40
    deployment_range = 150.0
    
    print(f"\nCreating two V3 channels with same seed={seed}...")
    
    # Create first channel
    channel1 = WirelessChannelV3(
        n_links=n_links,
        deployment_range=deployment_range,
        min_tx_tx_distance=12.0,
        max_tx_rx_distance=45.0,
        seed=seed
    )
    
    # Create second channel with same parameters
    channel2 = WirelessChannelV3(
        n_links=n_links,
        deployment_range=deployment_range,
        min_tx_tx_distance=12.0,
        max_tx_rx_distance=45.0,
        seed=seed
    )
    
    # Compare TX locations
    print("\n1. Comparing TX locations:")
    tx_diff = np.abs(channel1.tx_locations - channel2.tx_locations)
    print(f"   Max difference: {np.max(tx_diff):.10f}m")
    if np.allclose(channel1.tx_locations, channel2.tx_locations):
        print(f"   ✓ PASS: TX locations are identical")
    else:
        print(f"   ✗ FAIL: TX locations differ!")
    
    # Compare RX locations
    print("\n2. Comparing RX locations:")
    rx_diff = np.abs(channel1.rx_locations - channel2.rx_locations)
    print(f"   Max difference: {np.max(rx_diff):.10f}m")
    if np.allclose(channel1.rx_locations, channel2.rx_locations):
        print(f"   ✓ PASS: RX locations are identical")
    else:
        print(f"   ✗ FAIL: RX locations differ!")
    
    # Compare associations
    print("\n3. Comparing associations:")
    if np.all(channel1.associations == channel2.associations):
        print(f"   ✓ PASS: Associations are identical")
    else:
        diff_count = np.sum(channel1.associations != channel2.associations)
        print(f"   ✗ FAIL: {diff_count} association differences!")
    
    # Compare tx_rx_pairs
    print("\n4. Comparing tx_rx_pairs:")
    if np.all(channel1.tx_rx_pairs == channel2.tx_rx_pairs):
        print(f"   ✓ PASS: TX-RX pairs are identical")
    else:
        print(f"   ✗ FAIL: TX-RX pairs differ!")
    
    # Compare large-scale fading
    print("\n5. Comparing large-scale fading:")
    lsf_diff = np.abs(channel1.large_scale_fading - channel2.large_scale_fading)
    print(f"   Max difference: {np.max(lsf_diff):.10e}")
    if np.allclose(channel1.large_scale_fading, channel2.large_scale_fading):
        print(f"   ✓ PASS: Large-scale fading is identical")
    else:
        print(f"   ✗ FAIL: Large-scale fading differs!")
    
    # Compare subdivision logs
    print("\n6. Comparing subdivision behavior:")
    summary1 = channel1.get_subdivision_summary()
    summary2 = channel2.get_subdivision_summary()
    print(f"   Channel 1 - Depth: {summary1['max_depth_reached']}, Subnetworks: {summary1['successful_subnetworks']}")
    print(f"   Channel 2 - Depth: {summary2['max_depth_reached']}, Subnetworks: {summary2['successful_subnetworks']}")
    
    if summary1 == summary2:
        print(f"   ✓ PASS: Subdivision behavior is identical")
    else:
        print(f"   ⚠ WARNING: Subdivision summaries differ (may be OK if both succeed)")
    
    # Overall verdict
    all_match = (
        np.allclose(channel1.tx_locations, channel2.tx_locations) and
        np.allclose(channel1.rx_locations, channel2.rx_locations) and
        np.all(channel1.associations == channel2.associations) and
        np.allclose(channel1.large_scale_fading, channel2.large_scale_fading)
    )
    
    print("\n" + "="*70)
    if all_match:
        print("✓ TEST 2 PASSED: Seed reproducibility works correctly")
    else:
        print("✗ TEST 2 FAILED: Seed does not produce identical results")
    print("="*70)
    
    return all_match


def test_shadowing_cross_subnetwork():
    """Test shadowing values between subnetworks."""
    print("\n" + "="*70)
    print("TEST 3: Cross-Subnetwork Shadowing")
    print("="*70)
    
    print("\nThis test checks if shadowing is properly computed for")
    print("TX-RX pairs across different subnetworks after subdivision.")
    
    channel = WirelessChannelV3(
        n_links=40,
        deployment_range=150.0,
        min_tx_tx_distance=12.0,
        max_tx_rx_distance=45.0,
        seed=777
    )
    
    # Check subdivision summary
    summary = channel.get_subdivision_summary()
    print(f"\nSubdivision occurred: depth={summary['max_depth_reached']}")
    
    if summary['max_depth_reached'] == 0:
        print("No subdivision occurred - test not applicable")
        return
    
    # Analyze shadowing matrix structure
    print("\nAnalyzing shadowing matrix structure:")
    
    # For a block diagonal matrix, off-diagonal blocks should be zero
    # We'll check what percentage of off-diagonal elements are zero
    shadowing = channel.shadowing_db
    n = shadowing.shape[0]
    
    # Count zeros in off-diagonal elements
    mask = ~np.eye(n, dtype=bool)
    off_diag = shadowing[mask]
    num_zeros = np.sum(np.abs(off_diag) < 1e-10)
    num_nonzeros = len(off_diag) - num_zeros
    
    print(f"  Total off-diagonal elements: {len(off_diag)}")
    print(f"  Zero elements: {num_zeros} ({100*num_zeros/len(off_diag):.1f}%)")
    print(f"  Non-zero elements: {num_nonzeros} ({100*num_nonzeros/len(off_diag):.1f}%)")
    
    # After V2's surgical redeployment and recomputation, cross-subnetwork
    # shadowing should be filled in
    if num_zeros > len(off_diag) * 0.3:
        print(f"\n  ⚠ WARNING: Shadowing is sparse (>30% zeros)")
        print(f"  This suggests block-diagonal structure from subdivision")
        print(f"  Cross-subnetwork shadowing may not be properly recomputed")
    else:
        print(f"\n  ✓ GOOD: Shadowing appears fully populated")
        print(f"  Cross-subnetwork values are present")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("WIRELESSCHANNELV3 CORRECTNESS TESTS")
    print("="*70)
    
    # Run tests
    channel = test_associations_correctness()
    reproducible = test_seed_reproducibility()
    test_shadowing_cross_subnetwork()
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print("Test 1 - Associations & Pairing: ✓ PASSED")
    print(f"Test 2 - Seed Reproducibility: {'✓ PASSED' if reproducible else '✗ FAILED'}")
    print("Test 3 - Cross-Subnetwork Shadowing: See output above")
    print("="*70)
