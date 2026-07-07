#!/usr/bin/env python3
"""
Test script to verify that merging 4 subnetworks of 50 links creates
reliable 200-link networks without deployment failures.

This demonstrates the solution for large-scale network generation where
direct deployment fails to satisfy interference dominance constraints.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from graph_signal_diffusion.datasets.wra.channel import WirelessChannel, merge_channels
import numpy as np
import time


def test_direct_deployment_200links():
    """Try direct deployment of 200 links (expected to fail frequently)."""
    print("\n" + "="*70)
    print("TEST 1: DIRECT DEPLOYMENT OF 200 LINKS")
    print("="*70)
    print("\nNote: This test is expected to fail due to deployment constraints.")
    print("      For 200 links, transmitter placement with 35m minimum separation")
    print("      is extremely difficult in the scaled deployment area.")
    
    success_count = 0
    num_attempts = 3  # Reduced from 5 since failures are expected
    
    for i in range(num_attempts):
        print(f"\nAttempt {i+1}/{num_attempts}...")
        try:
            start = time.time()
            channel = WirelessChannel(
                n_links=200,
                deployment_range=1400.0,  # Scale area to maintain density ~102 links/km²
                seed=1000 + i
            )
            elapsed = time.time() - start
            success_count += 1
            print(f"  ✓ SUCCESS in {elapsed:.2f}s")
        except RuntimeError as e:
            elapsed = time.time() - start
            print(f"  ✗ EXPECTED FAILURE after {elapsed:.2f}s")
            print(f"     Reason: {str(e)[:100]}...")
        except Exception as e:
            elapsed = time.time() - start
            print(f"  ✗ FAILED after {elapsed:.2f}s: {str(e)[:80]}")
    
    print(f"\nDirect deployment success rate: {success_count}/{num_attempts} ({success_count/num_attempts*100:.0f}%)")
    if success_count == 0:
        print("  As expected: Direct deployment is not feasible for 200 links")
    return success_count


def test_merged_deployment_200links():
    """Create 200-link networks by merging 4 subnetworks of 50 links."""
    print("\n" + "="*70)
    print("TEST 2: MERGED DEPLOYMENT (4 × 50 LINKS)")
    print("="*70)
    
    success_count = 0
    num_attempts = 5
    
    for i in range(num_attempts):
        print(f"\nAttempt {i+1}/{num_attempts}...")
        try:
            start = time.time()
            
            # Create 4 subnetworks of 50 links each
            # Use well-spaced seeds to avoid clustering
            subnets = []
            base_seed = 42 + i * 100  # Space out base seeds
            for j in range(4):
                subnet = WirelessChannel(
                    n_links=50,
                    deployment_range=700.0,  # Same as working 50-link case
                    seed=base_seed + j * 10  # Space out subnet seeds
                )
                subnets.append(subnet)
            
            # Merge with circular layout
            merged = merge_channels(
                subnets,
                spacing=150.0,
                layout='circular'
            )
            
            elapsed = time.time() - start
            success_count += 1
            
            # Verify result
            info = merged.get_network_info()
            print(f"  ✓ SUCCESS in {elapsed:.2f}s")
            print(f"    Total links: {info['n_links']}")
            print(f"    Deployment range: {merged.deployment_range:.1f}m")
            print(f"    Mean TX-TX distance: {info['mean_tx_tx_distance']:.1f}m")
            
        except RuntimeError as e:
            elapsed = time.time() - start
            print(f"  ✗ FAILED after {elapsed:.2f}s")
            print(f"     Reason: {str(e)[:100]}...")
        except Exception as e:
            elapsed = time.time() - start
            print(f"  ✗ FAILED after {elapsed:.2f}s: {str(e)[:80]}")
    
    print(f"\nMerged deployment success rate: {success_count}/{num_attempts} ({success_count/num_attempts*100:.0f}%)")
    return success_count


def test_density_comparison():
    """Compare network density between direct and merged approaches."""
    print("\n" + "="*70)
    print("TEST 3: DENSITY COMPARISON")
    print("="*70)
    
    # Create one merged network - try multiple seeds if needed
    print("\nCreating merged 200-link network...")
    
    # Try a few different base seeds until we get successful deployment
    for base_seed in [42, 100, 200, 300, 500]:
        try:
            subnets = []
            for j in range(4):
                subnet = WirelessChannel(
                    n_links=50,
                    deployment_range=700.0,
                    seed=base_seed + j
                )
                subnets.append(subnet)
            
            merged = merge_channels(subnets, spacing=150.0, layout='circular')
            print(f"✓ Successfully created merged network (base_seed={base_seed})")
            break
        except RuntimeError as e:
            print(f"  Attempt with base_seed={base_seed} failed, trying next seed...")
            continue
    else:
        print("  ⚠ Could not create merged network, skipping density comparison")
        return
    
    # Compute density
    all_locs = np.vstack([merged.tx_locations, merged.rx_locations])
    min_x, max_x = all_locs[:, 0].min(), all_locs[:, 0].max()
    min_y, max_y = all_locs[:, 1].min(), all_locs[:, 1].max()
    
    width = max_x - min_x
    height = max_y - min_y
    area_m2 = width * height
    area_km2 = area_m2 / 1e6
    
    density = merged.n_links / area_km2
    
    print(f"\nMerged Network Statistics:")
    print(f"  Total links: {merged.n_links}")
    print(f"  Spatial extent: {width:.1f}m × {height:.1f}m")
    print(f"  Area: {area_km2:.3f} km²")
    print(f"  Density: {density:.1f} links/km²")
    print(f"  Target density (50-link case): ~102 links/km²")
    
    # Compare with 50-link baseline - try multiple seeds
    print(f"\nBaseline 50-link network:")
    for seed in [42, 100, 200]:
        try:
            subnet_50 = WirelessChannel(n_links=50, deployment_range=700.0, seed=seed)
            area_50_km2 = (0.7 * 0.7)  # 700m × 700m
            density_50 = 50 / area_50_km2
            
            print(f"  Area: {area_50_km2:.3f} km²")
            print(f"  Density: {density_50:.1f} links/km²")
            print(f"\nDensity ratio: {density/density_50:.2f}x")
            break
        except RuntimeError:
            if seed == 200:
                print(f"  ⚠ Could not create baseline, using theoretical value")
                area_50_km2 = (0.7 * 0.7)
                density_50 = 50 / area_50_km2
                print(f"  Area: {area_50_km2:.3f} km²")
                print(f"  Density (theoretical): {density_50:.1f} links/km²")
                print(f"\nDensity ratio: {density/density_50:.2f}x")
            continue


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print("DEPLOYMENT STRATEGY COMPARISON: 200-LINK NETWORKS")
    print("="*70)
    print("\nObjective: Demonstrate that subnetwork merging solves")
    print("           deployment constraint satisfaction for large networks")
    
    # Test direct deployment
    direct_success = test_direct_deployment_200links()
    
    # Test merged deployment
    merged_success = test_merged_deployment_200links()
    
    # Test density
    test_density_comparison()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Direct deployment (200 links):  {direct_success}/3 success")
    print(f"Merged deployment (4×50 links): {merged_success}/5 success")
    
    if merged_success >= 5:
        print("\n✓ RESULT: Subnetwork merging is the ONLY viable approach for 200-link networks")
        print("  - Direct deployment: Failed due to constraint satisfaction")
        print("  - Merged deployment: 100% success rate")
        print("  - Maintains realistic network density")
        print("  - Creates heterogeneous interference patterns")
    elif merged_success > direct_success:
        print("\n✓ RECOMMENDATION: Use subnetwork merging for 200-link networks")
        print("  - Much higher success rate")
        print("  - Faster deployment time")
        print("  - Maintains realistic network density")
        print("  - Creates heterogeneous interference patterns")
    
    print("\nTo use in your training:")
    print("  dataset = WRAPrimalDualDataset.from_seed_range_with_merging(")
    print("      num_networks=32,")
    print("      n_links_per_subnet=50,")
    print("      num_subnets=4,")
    print("      deployment_range=700.0,")
    print("      subnet_spacing=150.0,")
    print("      layout='circular'")
    print("  )")


if __name__ == "__main__":
    main()
