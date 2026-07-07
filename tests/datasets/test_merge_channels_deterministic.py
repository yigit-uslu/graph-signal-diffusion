"""
Test that merge_channels produces deterministic results with local seeds.
"""

import sys
import os
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(project_root, 'src'))

from graph_signal_diffusion.datasets.wra.channel import merge_channels
from graph_signal_diffusion.datasets.wra.channel import WirelessChannelV3


def test_merge_channels_deterministic():
    """Test that merge_channels produces deterministic results."""
    print("="*70)
    print("TEST: merge_channels Deterministic Behavior")
    print("="*70)
    
    # Create two small channels
    print("\nCreating two channels with fixed seeds...")
    ch1 = WirelessChannelV3(n_links=5, deployment_range=80.0, seed=100)
    ch2 = WirelessChannelV3(n_links=5, deployment_range=80.0, seed=200)
    
    print(f"  Channel 1: {ch1.n_links} links (seed=100)")
    print(f"  Channel 2: {ch2.n_links} links (seed=200)")
    
    # Merge them multiple times
    print("\nMerging channels 3 times...")
    merged_channels = []
    for i in range(3):
        # Reset RNG to ensure independence
        np.random.seed(None)
        
        # Create fresh channel instances for merging
        ch1_fresh = WirelessChannelV3(n_links=5, deployment_range=80.0, seed=100)
        ch2_fresh = WirelessChannelV3(n_links=5, deployment_range=80.0, seed=200)
        
        merged = merge_channels([ch1_fresh, ch2_fresh], spacing=50.0, layout='linear')
        merged_channels.append(merged)
        print(f"  ✓ Merge {i+1} complete: {merged.n_links} links")
    
    # Compare merged channels
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    
    all_identical = True
    
    print("\n1. Comparing TX locations:")
    for i in range(len(merged_channels) - 1):
        tx_match = np.allclose(
            merged_channels[i].tx_locations,
            merged_channels[i+1].tx_locations
        )
        print(f"   Merge {i+1} vs {i+2}: {tx_match}")
        if not tx_match:
            all_identical = False
    
    print("\n2. Comparing RX locations:")
    for i in range(len(merged_channels) - 1):
        rx_match = np.allclose(
            merged_channels[i].rx_locations,
            merged_channels[i+1].rx_locations
        )
        print(f"   Merge {i+1} vs {i+2}: {rx_match}")
        if not rx_match:
            all_identical = False
    
    print("\n3. Comparing shadowing (should be identical with local seed):")
    for i in range(len(merged_channels) - 1):
        shadow_match = np.allclose(
            merged_channels[i].shadowing_db,
            merged_channels[i+1].shadowing_db
        )
        max_diff = np.max(np.abs(merged_channels[i].shadowing_db - merged_channels[i+1].shadowing_db))
        print(f"   Merge {i+1} vs {i+2}: {shadow_match} (max diff: {max_diff:.10f} dB)")
        if not shadow_match:
            all_identical = False
    
    print("\n4. Comparing associations:")
    for i in range(len(merged_channels) - 1):
        assoc_match = np.array_equal(
            merged_channels[i].associations,
            merged_channels[i+1].associations
        )
        print(f"   Merge {i+1} vs {i+2}: {assoc_match}")
        if not assoc_match:
            all_identical = False
    
    # Summary
    print("\n" + "="*70)
    
    if all_identical:
        print("✓ TEST PASSED: All merges are identical")
        print("\nKey insights:")
        print("  • Subnetwork locations are deterministic (from seeds 100 & 200)")
        print("  • Combined locations are deterministic (fixed spatial offset)")
        print("  • Local seed derived from combined locations")
        print("  • Shadowing is deterministic from local seed")
        print("  • Associations are deterministic from shadowing")
        print("  • merge_channels is fully reproducible!")
    else:
        print("✗ TEST FAILED: Merges differ")
    
    print("="*70)
    
    return all_identical


def test_merge_channels_reproducibility():
    """Test that we can reproduce a merged channel from saved locations."""
    print("\n" + "="*70)
    print("TEST: Reproduce Merged Channel from Locations")
    print("="*70)
    
    # Create and merge channels
    print("\nCreating and merging two channels...")
    ch1 = WirelessChannelV3(n_links=5, deployment_range=80.0, seed=300)
    ch2 = WirelessChannelV3(n_links=5, deployment_range=80.0, seed=400)
    
    merged = merge_channels([ch1, ch2], spacing=50.0, layout='circular')
    print(f"  ✓ Merged: {merged.n_links} links")
    
    # Save locations
    print("\nSaving merged channel locations...")
    saved_tx = merged.tx_locations.copy()
    saved_rx = merged.rx_locations.copy()
    
    # Reproduce from locations
    print("\nReproducing merged channel from locations...")
    from graph_signal_diffusion.datasets.wra.channel import WirelessChannel
    
    reproduced = WirelessChannel(
        n_links=merged.n_links,
        deployment_range=merged.deployment_range,
        skip_deployment=True,
        tx_locations=saved_tx,
        rx_locations=saved_rx
    )
    
    # Compare
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    
    tx_match = np.allclose(merged.tx_locations, reproduced.tx_locations)
    rx_match = np.allclose(merged.rx_locations, reproduced.rx_locations)
    shadow_match = np.allclose(merged.shadowing_db, reproduced.shadowing_db)
    assoc_match = np.array_equal(merged.associations, reproduced.associations)
    
    print(f"  TX locations: {tx_match}")
    print(f"  RX locations: {rx_match}")
    print(f"  Shadowing: {shadow_match}")
    print(f"  Associations: {assoc_match}")
    
    all_match = tx_match and rx_match and shadow_match and assoc_match
    
    print("\n" + "="*70)
    
    if all_match:
        print("✓ TEST PASSED: Reproduced channel is identical")
        print("\nUsage pattern:")
        print("  1. Merge channels → get combined locations")
        print("  2. Save locations (not shadowing/associations)")
        print("  3. Restore from locations → identical shadowing/associations")
    else:
        print("✗ TEST FAILED: Reproduced channel differs")
    
    print("="*70)
    
    return all_match


if __name__ == "__main__":
    print("\n" + "="*70)
    print("MERGE_CHANNELS DETERMINISTIC TESTS")
    print("="*70)
    
    results = []
    
    # Run tests
    results.append(("merge_channels deterministic", test_merge_channels_deterministic()))
    results.append(("merge_channels reproducibility", test_merge_channels_reproducibility()))
    
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
        print("\nConclusion:")
        print("  • merge_channels is now fully deterministic")
        print("  • Same subnetwork seeds → same merged network")
        print("  • Can save/restore merged networks from locations alone")
        print("  • No need to save shadowing or associations matrices")
    else:
        print("✗ SOME TESTS FAILED")
    
    print("="*70)
    
    exit(0 if all_passed else 1)
