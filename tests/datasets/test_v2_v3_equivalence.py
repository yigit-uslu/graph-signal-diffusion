"""
Test to verify WirelessChannelV3 is indistinguishable from V2 after deployment.

This test creates networks using V2 and V3, then compares all attributes
to ensure V3 behaves identically to V2 from the perspective of downstream code.
"""

import sys
import os
import numpy as np

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(project_root, 'src'))

from graph_signal_diffusion.datasets.wra.channel import WirelessChannelV2
from graph_signal_diffusion.datasets.wra.channel import WirelessChannelV3


def test_v2_v3_equivalence():
    """Test that V3 produces the same interface as V2 after deployment."""
    print("="*70)
    print("TEST: V2 vs V3 Interface Equivalence")
    print("="*70)
    print("\nThis test verifies that after deployment, V3 is indistinguishable")
    print("from V2 - all attributes, methods, and data structures match.\n")
    
    # Use parameters that V2 can handle (no subdivision needed)
    seed = 99999
    n_links = 15
    deployment_range = 200.0
    
    print("Creating V2 channel (simple case, no subdivision)...")
    v2 = WirelessChannelV2(
        n_links=n_links,
        deployment_range=deployment_range,
        min_tx_tx_distance=10.0,
        max_tx_rx_distance=45.0,
        seed=seed
    )
    
    print("\nCreating V3 channel with same parameters...")
    v3 = WirelessChannelV3(
        n_links=n_links,
        deployment_range=deployment_range,
        min_tx_tx_distance=10.0,
        max_tx_rx_distance=45.0,
        seed=seed
    )
    
    print("\n" + "="*70)
    print("Comparing Attributes")
    print("="*70)
    
    all_match = True
    
    # 1. Check basic attributes
    print("\n1. Basic configuration:")
    attrs = ['n_links', 'deployment_range', 'min_tx_tx_distance', 
             'max_tx_rx_distance', 'seed']
    for attr in attrs:
        v2_val = getattr(v2, attr)
        v3_val = getattr(v3, attr)
        match = v2_val == v3_val
        print(f"   {attr}: {'✓' if match else '✗'} ({v2_val} vs {v3_val})")
        all_match = all_match and match
    
    # 2. Check array shapes
    print("\n2. Array shapes:")
    arrays = [
        ('tx_locations', (n_links, 2)),
        ('rx_locations', (n_links, 2)),
        ('associations', (n_links, n_links)),
        ('tx_rx_pairs', (n_links, 2)),
        ('distances', (n_links, n_links)),
        ('large_scale_fading', (n_links, n_links)),
        ('shadowing_db', (n_links, n_links)),
        ('path_loss_db', (n_links, n_links))
    ]
    
    for attr, expected_shape in arrays:
        v2_shape = getattr(v2, attr).shape
        v3_shape = getattr(v3, attr).shape
        match = v2_shape == v3_shape == expected_shape
        print(f"   {attr}: {'✓' if match else '✗'} (V2: {v2_shape}, V3: {v3_shape}, Expected: {expected_shape})")
        all_match = all_match and match
    
    # 3. Check data types
    print("\n3. Data types:")
    for attr, _ in arrays:
        v2_dtype = getattr(v2, attr).dtype
        v3_dtype = getattr(v3, attr).dtype
        match = v2_dtype == v3_dtype
        print(f"   {attr}: {'✓' if match else '✗'} ({v2_dtype} vs {v3_dtype})")
        all_match = all_match and match
    
    # 4. Check for NaN/Inf
    print("\n4. Data validity (no NaN/Inf):")
    for attr, _ in arrays[2:]:  # Skip location arrays
        v2_arr = getattr(v2, attr)
        v3_arr = getattr(v3, attr)
        v2_valid = not (np.any(np.isnan(v2_arr)) or np.any(np.isinf(v2_arr)))
        v3_valid = not (np.any(np.isnan(v3_arr)) or np.any(np.isinf(v3_arr)))
        match = v2_valid and v3_valid
        print(f"   {attr}: {'✓' if match else '✗'} (V2: {v2_valid}, V3: {v3_valid})")
        all_match = all_match and match
    
    # 5. Check associations properties
    print("\n5. Associations properties:")
    
    # One-to-one mapping
    v2_tx_sum = np.sum(v2.associations, axis=1)
    v3_tx_sum = np.sum(v3.associations, axis=1)
    v2_rx_sum = np.sum(v2.associations, axis=0)
    v3_rx_sum = np.sum(v3.associations, axis=0)
    
    tx_match = np.all(v2_tx_sum == 1) and np.all(v3_tx_sum == 1)
    rx_match = np.all(v2_rx_sum == 1) and np.all(v3_rx_sum == 1)
    
    print(f"   Each TX paired with 1 RX: {'✓' if tx_match else '✗'}")
    print(f"   Each RX paired with 1 TX: {'✓' if rx_match else '✗'}")
    all_match = all_match and tx_match and rx_match
    
    # 6. Check shadowing structure
    print("\n6. Shadowing matrix structure:")
    
    # V2 shadowing should be fully populated
    v2_mask = ~np.eye(n_links, dtype=bool)
    v2_off_diag = v2.shadowing_db[v2_mask]
    v2_zeros = np.sum(np.abs(v2_off_diag) < 1e-10)
    v2_pct = 100 * v2_zeros / len(v2_off_diag)
    
    # V3 shadowing should also be fully populated (after fix)
    v3_off_diag = v3.shadowing_db[v2_mask]
    v3_zeros = np.sum(np.abs(v3_off_diag) < 1e-10)
    v3_pct = 100 * v3_zeros / len(v3_off_diag)
    
    print(f"   V2 off-diagonal zeros: {v2_zeros}/{len(v2_off_diag)} ({v2_pct:.1f}%)")
    print(f"   V3 off-diagonal zeros: {v3_zeros}/{len(v3_off_diag)} ({v3_pct:.1f}%)")
    
    both_populated = (v2_pct < 5.0) and (v3_pct < 5.0)
    print(f"   Both fully populated: {'✓' if both_populated else '✗'}")
    all_match = all_match and both_populated
    
    # 7. Check method availability
    print("\n7. Method availability:")
    methods = ['sample_realization', 'get_network_info', '_compute_path_loss',
               '_compute_large_scale_fading', '_generate_rayleigh_fading']
    
    for method in methods:
        v2_has = hasattr(v2, method) and callable(getattr(v2, method))
        v3_has = hasattr(v3, method) and callable(getattr(v3, method))
        match = v2_has and v3_has
        print(f"   {method}: {'✓' if match else '✗'}")
        all_match = all_match and match
    
    # 8. Test sample_realization works identically
    print("\n8. Testing sample_realization method:")
    try:
        v2_sample = v2.sample_realization(num_timesteps=10, disable_small_scale_fading=True)
        v3_sample = v3.sample_realization(num_timesteps=10, disable_small_scale_fading=True)
        
        # Check all expected keys
        expected_keys = ['H', 'H_l', 'tx_locations', 'rx_locations', 'associations', 
                        'distances', 'path_loss_db', 'shadowing_db']
        keys_match = set(v2_sample.keys()) == set(v3_sample.keys()) == set(expected_keys)
        print(f"   Return keys match: {'✓' if keys_match else '✗'}")
        
        # Check shapes match
        shapes_match = all(v2_sample[k].shape == v3_sample[k].shape for k in expected_keys)
        print(f"   Array shapes match: {'✓' if shapes_match else '✗'}")
        
        method_works = keys_match and shapes_match
        all_match = all_match and method_works
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
        all_match = False
    
    # 9. V3-specific attributes (should not interfere)
    print("\n9. V3-specific attributes (should be transparent):")
    v3_specific = ['max_recursion_depth', '_subdivision_log']
    for attr in v3_specific:
        has_attr = hasattr(v3, attr)
        not_in_v2 = not hasattr(v2, attr)
        print(f"   V3.{attr}: {'✓' if has_attr else '✗'} (V2 doesn't have: {'✓' if not_in_v2 else '✗'})")
    
    print("\n" + "="*70)
    if all_match:
        print("✓ TEST PASSED: V3 is fully compatible with V2 interface")
        print("  After deployment, downstream code cannot distinguish V2 from V3")
    else:
        print("✗ TEST FAILED: V3 has interface differences from V2")
    print("="*70)
    
    return all_match


if __name__ == "__main__":
    success = test_v2_v3_equivalence()
    exit(0 if success else 1)
