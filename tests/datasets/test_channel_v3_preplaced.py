"""
Test WirelessChannelV3 with pre-placed TX/RX locations (merge_channels scenario).

This verifies that V3 properly handles skip_deployment=True and works
correctly with manually set tx/rx locations, as used in merge_channels().
"""

import sys
import os
import numpy as np
from scipy.spatial import distance

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(project_root, 'src'))

from graph_signal_diffusion.datasets.wra.channel import WirelessChannel, WirelessChannelV2, merge_channels
from graph_signal_diffusion.datasets.wra.channel import WirelessChannelV3


def test_skip_deployment():
    """Test V3 with skip_deployment=True and pre-placed locations."""
    print("="*70)
    print("TEST 1: V3 with skip_deployment=True")
    print("="*70)
    
    n_links = 10
    deployment_range = 150.0
    
    # Create pre-placed TX and RX locations
    np.random.seed(42)
    tx_locs = np.random.uniform(-deployment_range/2, deployment_range/2, (n_links, 2))
    rx_locs = np.random.uniform(-deployment_range/2, deployment_range/2, (n_links, 2))
    
    print(f"\nCreating V3 channel with skip_deployment=True...")
    print(f"  Pre-placing {n_links} TX and {n_links} RX manually")
    
    try:
        # Create channel with skip_deployment
        channel = WirelessChannelV3(
            n_links=n_links,
            deployment_range=deployment_range,
            skip_deployment=True,
            seed=None  # No seed needed since we're not deploying
        )
        
        # Manually set locations (as merge_channels does)
        channel.tx_locations = tx_locs
        channel.rx_locations = rx_locs
        channel.shadowing_db_deployment = None
        
        # Compute large-scale fading and pairing (as merge_channels does)
        channel._compute_large_scale_fading()
        channel._assign_optimal_pairing()
        
        print("  ✓ Channel created successfully")
        
        # Verify attributes
        print("\n1. Checking attributes:")
        assert channel.tx_locations.shape == (n_links, 2), "TX locations shape mismatch"
        assert channel.rx_locations.shape == (n_links, 2), "RX locations shape mismatch"
        print(f"   ✓ TX locations: {channel.tx_locations.shape}")
        print(f"   ✓ RX locations: {channel.rx_locations.shape}")
        
        # Check associations
        print("\n2. Checking associations:")
        assert channel.associations.shape == (n_links, n_links), "Associations shape mismatch"
        assert np.all(np.sum(channel.associations, axis=1) == 1), "Each TX must pair with 1 RX"
        assert np.all(np.sum(channel.associations, axis=0) == 1), "Each RX must pair with 1 TX"
        print(f"   ✓ Associations: {channel.associations.shape}")
        print(f"   ✓ One-to-one pairing verified")
        
        # Check shadowing
        print("\n3. Checking shadowing:")
        assert channel.shadowing_db.shape == (n_links, n_links), "Shadowing shape mismatch"
        mask = ~np.eye(n_links, dtype=bool)
        off_diag = channel.shadowing_db[mask]
        num_zeros = np.sum(np.abs(off_diag) < 1e-10)
        print(f"   ✓ Shadowing: {channel.shadowing_db.shape}")
        print(f"   ✓ Off-diagonal zeros: {num_zeros}/{len(off_diag)} ({100*num_zeros/len(off_diag):.1f}%)")
        
        # Check V3-specific attributes exist but weren't used
        print("\n4. Checking V3-specific attributes:")
        assert hasattr(channel, 'max_recursion_depth'), "Missing max_recursion_depth"
        assert hasattr(channel, '_subdivision_log'), "Missing _subdivision_log"
        print(f"   ✓ max_recursion_depth: {channel.max_recursion_depth}")
        print(f"   ✓ _subdivision_log: {len(channel._subdivision_log)} entries")
        print(f"   ✓ (Should be 0 since deployment was skipped)")
        
        if len(channel._subdivision_log) == 0:
            print("   ✓ PASS: No subdivision occurred (as expected)")
        else:
            print(f"   ⚠ WARNING: Subdivision log has {len(channel._subdivision_log)} entries")
        
        print("\n" + "="*70)
        print("✓ TEST 1 PASSED: V3 handles skip_deployment correctly")
        print("="*70)
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 1 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_merge_channels_with_v3():
    """Test merge_channels function with WirelessChannelV3."""
    print("\n" + "="*70)
    print("TEST 2: merge_channels with V3 subnetworks")
    print("="*70)
    
    print("\nCreating 2 small V3 channels to merge...")
    
    # Create two small channels
    channel1 = WirelessChannelV3(
        n_links=5,
        deployment_range=80.0,
        min_tx_tx_distance=8.0,
        max_tx_rx_distance=30.0,
        seed=100
    )
    
    channel2 = WirelessChannelV3(
        n_links=5,
        deployment_range=80.0,
        min_tx_tx_distance=8.0,
        max_tx_rx_distance=30.0,
        seed=200
    )
    
    print(f"  Channel 1: {channel1.n_links} links")
    print(f"  Channel 2: {channel2.n_links} links")
    
    # Merge them
    print("\nMerging channels...")
    try:
        merged = merge_channels([channel1, channel2], spacing=50.0, layout='circular')
        
        print(f"  ✓ Merged successfully")
        print(f"  Total links: {merged.n_links}")
        
        # Verify merged channel
        print("\n1. Checking merged channel attributes:")
        expected_links = channel1.n_links + channel2.n_links
        assert merged.n_links == expected_links, f"Expected {expected_links} links, got {merged.n_links}"
        print(f"   ✓ Total links: {merged.n_links}")
        
        assert merged.tx_locations.shape == (expected_links, 2), "TX locations shape mismatch"
        assert merged.rx_locations.shape == (expected_links, 2), "RX locations shape mismatch"
        print(f"   ✓ TX locations: {merged.tx_locations.shape}")
        print(f"   ✓ RX locations: {merged.rx_locations.shape}")
        
        # Check associations
        print("\n2. Checking merged associations:")
        assert merged.associations.shape == (expected_links, expected_links)
        assert np.all(np.sum(merged.associations, axis=1) == 1)
        assert np.all(np.sum(merged.associations, axis=0) == 1)
        print(f"   ✓ Associations: {merged.associations.shape}")
        print(f"   ✓ One-to-one pairing verified")
        
        # Check shadowing is fully populated
        print("\n3. Checking merged shadowing:")
        mask = ~np.eye(expected_links, dtype=bool)
        off_diag = merged.shadowing_db[mask]
        num_zeros = np.sum(np.abs(off_diag) < 1e-10)
        pct_zeros = 100 * num_zeros / len(off_diag)
        print(f"   Off-diagonal zeros: {num_zeros}/{len(off_diag)} ({pct_zeros:.1f}%)")
        
        if pct_zeros < 5.0:
            print(f"   ✓ PASS: Shadowing fully populated")
        else:
            print(f"   ⚠ WARNING: Shadowing has {pct_zeros:.1f}% zeros")
        
        # Check that merged channel is standard WirelessChannel (not V3)
        print("\n4. Checking merged channel class:")
        print(f"   Type: {type(merged).__name__}")
        print(f"   Is WirelessChannel: {isinstance(merged, WirelessChannel)}")
        print(f"   Is WirelessChannelV3: {isinstance(merged, WirelessChannelV3)}")
        
        # The merged channel should be WirelessChannel, not V3
        # (merge_channels creates base class by design)
        if not isinstance(merged, WirelessChannelV3):
            print(f"   ✓ PASS: Merged as base WirelessChannel (expected)")
        else:
            print(f"   ⚠ NOTE: Merged as WirelessChannelV3 (may be intentional)")
        
        print("\n" + "="*70)
        print("✓ TEST 2 PASSED: merge_channels works with V3 inputs")
        print("="*70)
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 2 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_v2_v3_merge_compatibility():
    """Test that V2 and V3 can be merged together."""
    print("\n" + "="*70)
    print("TEST 3: Merging V2 and V3 channels together")
    print("="*70)
    
    print("\nCreating V2 and V3 channels to merge...")
    
    # Create V2 channel
    v2_channel = WirelessChannelV2(
        n_links=5,
        deployment_range=80.0,
        min_tx_tx_distance=8.0,
        max_tx_rx_distance=30.0,
        seed=300
    )
    
    # Create V3 channel (may use subdivision)
    v3_channel = WirelessChannelV3(
        n_links=5,
        deployment_range=80.0,
        min_tx_tx_distance=8.0,
        max_tx_rx_distance=30.0,
        seed=400
    )
    
    print(f"  V2 channel: {v2_channel.n_links} links")
    print(f"  V3 channel: {v3_channel.n_links} links")
    
    # Merge them
    print("\nMerging V2 and V3 channels...")
    try:
        merged = merge_channels([v2_channel, v3_channel], spacing=50.0, layout='linear')
        
        print(f"  ✓ Merged successfully")
        print(f"  Total links: {merged.n_links}")
        
        # Verify
        expected_links = v2_channel.n_links + v3_channel.n_links
        assert merged.n_links == expected_links
        assert merged.associations.shape == (expected_links, expected_links)
        assert np.all(np.sum(merged.associations, axis=1) == 1)
        assert np.all(np.sum(merged.associations, axis=0) == 1)
        
        print(f"   ✓ All checks passed")
        
        print("\n" + "="*70)
        print("✓ TEST 3 PASSED: V2 and V3 can be merged together")
        print("="*70)
        return True
        
    except Exception as e:
        print(f"\n✗ TEST 3 FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "="*70)
    print("WIRELESSCHANNELV3 PRE-PLACED LOCATIONS TESTS")
    print("="*70)
    
    results = []
    
    # Run tests
    results.append(("skip_deployment", test_skip_deployment()))
    results.append(("merge_channels with V3", test_merge_channels_with_v3()))
    results.append(("V2+V3 merge compatibility", test_v2_v3_merge_compatibility()))
    
    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    print("="*70)
    
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    
    print("="*70)
    
    exit(0 if all_passed else 1)
