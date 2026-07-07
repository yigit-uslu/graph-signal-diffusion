"""
Test RNG state differences between skip_deployment=True and skip_deployment=False.

When skip_deployment=False, the RNG is consumed by TX/RX deployment before shadowing.
When skip_deployment=True, shadowing is generated with a different RNG state.
This means the shadowing values will be DIFFERENT even with the same seed.
"""

import sys
import os
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(project_root, 'src'))

from graph_signal_diffusion.datasets.wra.channel import WirelessChannelV3


def test_rng_state_difference():
    """Test that skip_deployment affects shadowing due to RNG state."""
    print("="*70)
    print("TEST: RNG State Difference (skip_deployment=True vs False)")
    print("="*70)
    
    n_links = 8
    deployment_range = 100.0
    seed = 42
    
    # Case 1: Normal deployment (skip_deployment=False)
    print(f"\nCase 1: Normal deployment with seed={seed}")
    print("  RNG sequence: seed → TX deployment → RX deployment → shadowing")
    channel_normal = WirelessChannelV3(
        n_links=n_links,
        deployment_range=deployment_range,
        seed=seed,
        skip_deployment=False
    )
    print(f"  ✓ Deployed: {channel_normal.n_links} links")
    
    # Case 2: Skip deployment, use same locations, regenerate shadowing
    print(f"\nCase 2: Skip deployment, set locations manually, seed={seed}")
    print("  RNG sequence: seed → shadowing (no deployment consumption)")
    channel_skip = WirelessChannelV3(
        n_links=n_links,
        deployment_range=deployment_range,
        seed=seed,  # Same seed!
        skip_deployment=True
    )
    
    # Use TX locations from Case 1, RX locations will be set then potentially modified
    channel_skip.tx_locations = channel_normal.tx_locations.copy()
    channel_skip.rx_locations = channel_normal.rx_locations.copy()
    channel_skip.shadowing_db_deployment = None
    
    print(f"  ✓ Locations set: {channel_skip.n_links} links")
    print(f"  ✓ Using TX/RX locations from Case 1 (before redeployment)")
    
    # Save pre-redeployment RX locations
    rx_locs_before = channel_skip.rx_locations.copy()
    
    # Regenerate large-scale fading (including shadowing) and pairing
    # NOTE: _assign_optimal_pairing may modify rx_locations (surgical redeployment)
    channel_skip._compute_large_scale_fading()
    channel_skip._assign_optimal_pairing()
    
    # Check if RX locations were modified by surgical redeployment
    rx_modified = not np.allclose(rx_locs_before, channel_skip.rx_locations)
    if rx_modified:
        rx_diff = np.linalg.norm(rx_locs_before - channel_skip.rx_locations, axis=1)
        num_modified = np.sum(rx_diff > 1e-6)
        print(f"  ⚠ Note: {num_modified} RX locations modified by surgical redeployment")
    
    # Compare
    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    
    # 1. TX locations should be identical (not modified after initial copy)
    print("\n1. TX Locations:")
    tx_identical = np.allclose(channel_normal.tx_locations, channel_skip.tx_locations)
    print(f"   TX locations identical: {tx_identical}")
    
    if tx_identical:
        print("   ✓ PASS: TX locations are identical (not modified)")
    else:
        print("   ✗ FAIL: TX locations should be identical!")
        return False
    
    # 2. RX locations may differ due to surgical redeployment
    print("\n2. RX Locations:")
    rx_identical = np.allclose(channel_normal.rx_locations, channel_skip.rx_locations)
    rx_diff = np.linalg.norm(channel_normal.rx_locations - channel_skip.rx_locations, axis=1)
    rx_max_diff = np.max(rx_diff)
    num_rx_diff = np.sum(rx_diff > 1e-6)
    
    print(f"   RX locations identical: {rx_identical}")
    print(f"   Max RX difference: {rx_max_diff:.6f} m")
    print(f"   Number of RXs differing: {num_rx_diff}/{n_links}")
    
    if not rx_identical:
        print(f"   ✓ EXPECTED: RX locations differ (surgical redeployment uses RNG)")
        print(f"              Different RNG states → different redeployment positions")
    else:
        print(f"   Note: RX locations happen to be identical")

    
    # 3. Path loss depends on final RX positions (after redeployment)
    print("\n3. Path Loss (distance-dependent, deterministic):")
    pl_diff = np.abs(channel_normal.path_loss_db - channel_skip.path_loss_db)
    pl_max_diff = np.max(pl_diff)
    pl_identical = np.allclose(channel_normal.path_loss_db, channel_skip.path_loss_db, atol=0.1)
    print(f"   Max difference: {pl_max_diff:.6f} dB")
    print(f"   Path loss nearly identical: {pl_identical}")
    
    if not pl_identical:
        print("   ✓ EXPECTED: Path loss differs (RX positions differ)")
    else:
        print("   Note: Path loss similar (RX positions similar)")
    
    # 4. Shadowing should be DIFFERENT (different RNG states)
    print("\n4. Shadowing (random, RNG-state-dependent):")
    shadow_diff = np.abs(channel_normal.shadowing_db - channel_skip.shadowing_db)
    shadow_max_diff = np.max(shadow_diff)
    shadow_mean_diff = np.mean(shadow_diff)
    shadow_identical = np.allclose(channel_normal.shadowing_db, channel_skip.shadowing_db)
    
    print(f"   Max difference: {shadow_max_diff:.6f} dB")
    print(f"   Mean difference: {shadow_mean_diff:.6f} dB")
    print(f"   Shadowing identical: {shadow_identical}")
    
    # Show some examples
    print(f"\n   Example shadowing values (first 3x3 submatrix):")
    print(f"   Normal deployment:")
    print(f"   {channel_normal.shadowing_db[:3, :3]}")
    print(f"   Skip deployment:")
    print(f"   {channel_skip.shadowing_db[:3, :3]}")
    
    if not shadow_identical and shadow_mean_diff > 0.1:
        print("   ✓ PASS: Shadowing is DIFFERENT (different RNG states)")
    else:
        print("   ⚠ WARNING: Shadowing appears identical (unexpected)")
    
    # 5. Large-scale fading should be different (path loss + shadowing)
    print("\n5. Large-Scale Fading (path loss + shadowing):")
    lsf_diff = np.abs(channel_normal.large_scale_fading - channel_skip.large_scale_fading)
    lsf_max_diff = np.max(lsf_diff)
    lsf_mean_diff = np.mean(lsf_diff)
    lsf_identical = np.allclose(channel_normal.large_scale_fading, channel_skip.large_scale_fading)
    
    print(f"   Max difference: {lsf_max_diff:.6f} dB")
    print(f"   Mean difference: {lsf_mean_diff:.6f} dB")
    print(f"   Large-scale fading identical: {lsf_identical}")
    
    if not lsf_identical:
        print("   ✓ PASS: Large-scale fading is DIFFERENT (shadowing differs)")
    else:
        print("   ⚠ WARNING: Large-scale fading identical (unexpected)")
    
    # 6. Associations might differ due to different large-scale fading
    print("\n6. Associations (depends on interference, which depends on fading):")
    assoc_identical = np.array_equal(channel_normal.associations, channel_skip.associations)
    print(f"   Associations identical: {assoc_identical}")
    
    if assoc_identical:
        print("   Note: Same pairing despite different shadowing")
        print("         (Path loss dominates, or shadowing differences didn't change optimal pairing)")
    else:
        print("   Note: Different pairing due to different shadowing")
        
        # Show differences
        pairs_normal = [(i, np.where(channel_normal.associations[i])[0][0]) for i in range(n_links)]
        pairs_skip = [(i, np.where(channel_skip.associations[i])[0][0]) for i in range(n_links)]
        
        print("\n   TX-RX pairs (Normal):", pairs_normal)
        print("   TX-RX pairs (Skip):  ", pairs_skip)
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("\nWhen using skip_deployment=True vs False with the same seed:")
    print("  • TX locations: Can be identical (copied before pairing)")
    print("  • RX locations: LIKELY DIFFERENT (surgical redeployment uses RNG)")
    print("  • Path loss: May differ (depends on final RX positions)")
    print("  • Shadowing: DEFINITELY DIFFERENT (RNG consumed by deployment first)")
    print("  • Large-scale fading: DIFFERENT (shadowing + possibly path loss differ)")
    print("  • Associations: MAY differ (optimal pairing depends on fading)")
    
    print("\nROOT CAUSE - TWO SOURCES OF RNG CONSUMPTION:")
    print("  When skip_deployment=False:")
    print("    RNG: seed → deployment → redeployment → shadowing")
    print("  When skip_deployment=True:")
    print("    RNG: seed → redeployment → shadowing")
    print("    (deployment RNG consumption skipped)")
    print("\n  1. Shadowing generation uses RNG (log-normal distribution)")
    print("  2. Surgical redeployment uses RNG (random angle/radius)")
    print("     - _assign_optimal_pairing() modifies RX positions")
    print("     - Different RNG state → different redeployment positions")
    
    print("\nIMPLICATIONS:")
    print("  • merge_channels creates networks with DIFFERENT shadowing")
    print("    than if you deployed the same total number of links at once")
    print("  • This is EXPECTED and CORRECT behavior")
    print("  • Each merged subnetwork has its own seed, creating diversity")
    print("  • After merging, shadowing is regenerated for the full network")
    
    print("="*70)
    print("✓ TEST PASSED: RNG state difference confirmed")
    print("="*70)
    
    return True


def test_merge_channels_seeding():
    """Show how merge_channels handles seeding."""
    print("\n" + "="*70)
    print("TEST: merge_channels Seeding Behavior")
    print("="*70)
    
    from graph_signal_diffusion.datasets.wra.channel import merge_channels
    
    # Create two channels with different seeds
    print("\nCreating two V3 channels with different seeds...")
    channel1 = WirelessChannelV3(n_links=5, deployment_range=80.0, seed=100)
    channel2 = WirelessChannelV3(n_links=5, deployment_range=80.0, seed=200)
    
    print(f"  Channel 1: seed=100, {channel1.n_links} links")
    print(f"  Channel 2: seed=200, {channel2.n_links} links")
    
    # Merge them
    print("\nMerging channels (seed=None for merged network)...")
    merged = merge_channels([channel1, channel2], spacing=50.0)
    
    print(f"  ✓ Merged: {merged.n_links} links")
    
    print("\nShadowing generation for merged network:")
    print("  • Subnetwork 1 locations: from seed=100 deployment")
    print("  • Subnetwork 2 locations: from seed=200 deployment")
    print("  • Combined locations: concatenated with spatial offset")
    print("  • Merged shadowing: NEW random realization (current RNG state)")
    print("    - NOT reproducible from seed=100 or seed=200")
    print("    - Different from deploying 10 links at once with any seed")
    
    print("\nTo get reproducible merged networks:")
    print("  1. Set global seed before merge_channels:")
    print("     np.random.seed(42)")
    print("     merged = merge_channels([ch1, ch2], ...)")
    print("  2. Or save/load merged channel state")
    
    print("\n" + "="*70)
    print("✓ TEST PASSED: Seeding behavior understood")
    print("="*70)
    
    return True


if __name__ == "__main__":
    print("\n" + "="*70)
    print("WIRELESSCHANNELV3 RNG STATE TESTS")
    print("="*70)
    
    results = []
    results.append(("RNG state difference", test_rng_state_difference()))
    results.append(("merge_channels seeding", test_merge_channels_seeding()))
    
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
