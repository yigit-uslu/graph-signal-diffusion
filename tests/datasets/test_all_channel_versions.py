"""
Test all three channel versions (V1, V2, V3) for consistency with pre-placed deployments.
"""

import sys
import os
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(project_root, 'src'))

from graph_signal_diffusion.datasets.wra.channel import WirelessChannel, WirelessChannelV2
from graph_signal_diffusion.datasets.wra.channel import WirelessChannelV3


def test_version_consistency(version_name, channel_class):
    """Test a specific channel version for pre-placed deployment consistency."""
    print("="*70)
    print(f"TEST: {version_name} Pre-Placed Deployment Consistency")
    print("="*70)
    
    n_links = 5  # Use fewer links for V1 compatibility
    deployment_range = 150.0  # Larger range
    seed = 200
    
    print(f"\nStep 1: Deploy original {version_name} with seed={seed}")
    original = channel_class(
        n_links=n_links,
        deployment_range=deployment_range,
        seed=seed,
        skip_deployment=False
    )
    print(f"  ✓ Original deployed: {original.n_links} links")
    
    # Save locations
    saved_tx = original.tx_locations.copy()
    saved_rx = original.rx_locations.copy()
    
    print(f"\nStep 2: Restore {version_name} from locations (skip_deployment=True)")
    restored = channel_class(
        n_links=n_links,
        deployment_range=deployment_range,
        skip_deployment=True,
        tx_locations=saved_tx,
        rx_locations=saved_rx
    )
    print(f"  ✓ Restored: {restored.n_links} links")
    
    print(f"\nStep 3: Verify consistency")
    
    checks = []
    
    # TX locations should be identical
    tx_match = np.array_equal(original.tx_locations, restored.tx_locations)
    checks.append(("TX locations", tx_match))
    print(f"  TX locations identical: {tx_match}")
    
    # RX locations should be identical (no surgical redeployment for pre-placed)
    rx_match = np.array_equal(original.rx_locations, restored.rx_locations)
    checks.append(("RX locations", rx_match))
    print(f"  RX locations identical: {rx_match}")
    
    if not rx_match:
        rx_diff = np.linalg.norm(original.rx_locations - restored.rx_locations, axis=1)
        num_diff = np.sum(rx_diff > 1e-6)
        max_diff = np.max(rx_diff)
        print(f"    ⚠ {num_diff}/{n_links} RX locations differ (max: {max_diff:.6f} m)")
    
    # Path loss should be identical (deterministic from locations)
    pl_match = np.allclose(original.path_loss_db, restored.path_loss_db)
    checks.append(("Path loss", pl_match))
    print(f"  Path loss identical: {pl_match}")
    
    # Check that shadowing is internally consistent
    # (Original and restored will differ due to different RNG, but each should be self-consistent)
    for ch, name in [(original, "Original"), (restored, "Restored")]:
        expected_lsf_db = ch.path_loss_db + ch.shadowing_db
        expected_lsf_linear = np.sqrt(10 ** (-expected_lsf_db / 10))
        actual_lsf = ch.large_scale_fading
        
        consistent = np.allclose(expected_lsf_linear, actual_lsf)
        checks.append((f"{name} LSF consistent", consistent))
        print(f"  {name} LSF = sqrt(10^(-(PL+S)/10)): {consistent}")
        
        if not consistent:
            max_diff = np.max(np.abs(expected_lsf_linear - actual_lsf))
            print(f"    ✗ Max difference: {max_diff:.10f}")
    
    # Multiple restorations should be identical
    print(f"\nStep 4: Test multiple restorations are identical")
    restored2 = channel_class(
        n_links=n_links,
        deployment_range=deployment_range,
        skip_deployment=True,
        tx_locations=saved_tx.copy(),
        rx_locations=saved_rx.copy()
    )
    
    shadow_match = np.allclose(restored.shadowing_db, restored2.shadowing_db)
    assoc_match = np.array_equal(restored.associations, restored2.associations)
    checks.append(("Multiple restorations identical", shadow_match and assoc_match))
    print(f"  Restoration 1 vs 2: shadowing={shadow_match}, associations={assoc_match}")
    
    # Summary
    print("\n" + "="*70)
    all_passed = all(passed for _, passed in checks)
    
    if all_passed:
        print(f"✓ {version_name} PASSED: All consistency checks passed")
    else:
        print(f"✗ {version_name} FAILED: Some checks failed")
        for name, passed in checks:
            if not passed:
                print(f"  ✗ {name}")
    
    print("="*70)
    
    return all_passed


def test_v1_no_redeployment():
    """Verify that V1 doesn't do surgical redeployment."""
    print("\n" + "="*70)
    print("TEST: V1 Has No Surgical Redeployment")
    print("="*70)
    
    # Check that V1's _assign_optimal_pairing doesn't modify RX locations
    print("\nVerifying V1 implementation...")
    import inspect
    source = inspect.getsource(WirelessChannel._assign_optimal_pairing)
    
    has_redeployment = "Redeploying" in source or "new_rx_pos" in source
    
    if not has_redeployment:
        print("  ✓ V1 _assign_optimal_pairing has no RX redeployment code")
        print("  ✓ V1 doesn't need shadowing regeneration fix")
        return True
    else:
        print("  ✗ V1 appears to have redeployment code")
        return False


def test_v2_has_redeployment():
    """Verify that V2 has surgical redeployment and the fix."""
    print("\n" + "="*70)
    print("TEST: V2 Has Surgical Redeployment With Fix")
    print("="*70)
    
    print("\nVerifying V2 implementation...")
    import inspect
    source = inspect.getsource(WirelessChannelV2._assign_optimal_pairing)
    
    has_redeployment = "Redeploying" in source
    has_fix = "shadowing_db_deployment = None" in source
    
    print(f"  Has surgical redeployment: {has_redeployment}")
    print(f"  Has shadowing regeneration fix: {has_fix}")
    
    if has_redeployment and has_fix:
        print("  ✓ V2 has redeployment with proper shadowing regeneration")
        return True
    elif has_redeployment and not has_fix:
        print("  ✗ V2 has redeployment but missing fix!")
        return False
    else:
        print("  ⚠ V2 implementation unexpected")
        return False


if __name__ == "__main__":
    print("\n" + "="*70)
    print("CHANNEL VERSION CONSISTENCY TESTS")
    print("="*70)
    
    results = []
    
    # Test implementation details
    results.append(("V1 no redeployment", test_v1_no_redeployment()))
    results.append(("V2 has redeployment+fix", test_v2_has_redeployment()))
    
    # Test each version
    results.append(("V1 consistency", test_version_consistency("V1 (WirelessChannel)", WirelessChannel)))
    results.append(("V2 consistency", test_version_consistency("V2 (WirelessChannelV2)", WirelessChannelV2)))
    results.append(("V3 consistency", test_version_consistency("V3 (WirelessChannelV3)", WirelessChannelV3)))
    
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
        print("  • V1: No surgical redeployment → no fix needed")
        print("  • V2: Has surgical redeployment → fix applied")
        print("  • V3: Inherits from V2 → gets fix automatically")
        print("  • All versions support local seed for pre-placed deployments")
        print("  • All versions are internally consistent")
    else:
        print("✗ SOME TESTS FAILED")
    
    print("="*70)
    
    exit(0 if all_passed else 1)
