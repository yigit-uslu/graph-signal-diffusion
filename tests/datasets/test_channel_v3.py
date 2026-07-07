"""
Test script for WirelessChannelV3 with recursive subdivision.

This test demonstrates:
1. V1: Simple fallback pairing (may violate max_tx_rx_distance)
2. V2: Surgical redeployment (enforces max_tx_rx_distance)
3. V3: Recursive subdivision (handles difficult large networks)

Includes network visualization showing TX-RX deployment and pairing.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(project_root, 'src'))

# Import directly from modules to avoid registry conflicts
from graph_signal_diffusion.datasets.wra.channel import WirelessChannel, WirelessChannelV2
from graph_signal_diffusion.datasets.wra.channel import WirelessChannelV3


def visualize_network(channel, title: str, save_path: str = None):
    """
    Visualize network deployment with TX-RX locations and pairing.
    
    Parameters
    ----------
    channel : WirelessChannel
        Channel instance to visualize
    title : str
        Plot title
    save_path : str, optional
        Path to save figure (if None, displays interactively)
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot transmitters
    ax.scatter(channel.tx_locations[:, 0], channel.tx_locations[:, 1], 
              c='red', s=100, marker='^', label='Transmitters', zorder=3, edgecolors='darkred', linewidths=1.5)
    
    # Plot receivers
    ax.scatter(channel.rx_locations[:, 0], channel.rx_locations[:, 1],
              c='blue', s=100, marker='o', label='Receivers', zorder=3, edgecolors='darkblue', linewidths=1.5)
    
    # Plot TX-RX pairings
    for tx_idx, rx_idx in channel.tx_rx_pairs:
        tx_pos = channel.tx_locations[tx_idx]
        rx_pos = channel.rx_locations[rx_idx]
        dist = np.linalg.norm(tx_pos - rx_pos)
        
        # Color based on distance constraint violation
        if hasattr(channel, 'max_tx_rx_distance') and channel.max_tx_rx_distance is not None:
            if dist > channel.max_tx_rx_distance:
                color = 'red'
                alpha = 0.8
                linewidth = 2
            else:
                color = 'green'
                alpha = 0.3
                linewidth = 1
        else:
            color = 'gray'
            alpha = 0.3
            linewidth = 1
        
        ax.plot([tx_pos[0], rx_pos[0]], [tx_pos[1], rx_pos[1]], 
               color=color, alpha=alpha, linewidth=linewidth, zorder=1)
    
    # Add labels for first few nodes
    for i in range(min(5, len(channel.tx_locations))):
        ax.text(channel.tx_locations[i, 0], channel.tx_locations[i, 1] + 5, 
               f'TX{i}', fontsize=8, ha='center', va='bottom')
    for i in range(min(5, len(channel.rx_locations))):
        ax.text(channel.rx_locations[i, 0], channel.rx_locations[i, 1] - 5,
               f'RX{i}', fontsize=8, ha='center', va='top')
    
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Add statistics text box
    paired_distances = []
    for tx_idx, rx_idx in channel.tx_rx_pairs:
        dist = np.linalg.norm(channel.tx_locations[tx_idx] - channel.rx_locations[rx_idx])
        paired_distances.append(dist)
    
    paired_distances = np.array(paired_distances)
    violations = 0
    if hasattr(channel, 'max_tx_rx_distance') and channel.max_tx_rx_distance is not None:
        violations = np.sum(paired_distances > channel.max_tx_rx_distance)
    
    stats_text = f"Links: {len(channel.tx_rx_pairs)}\n"
    stats_text += f"Mean distance: {np.mean(paired_distances):.1f}m\n"
    stats_text += f"Max distance: {np.max(paired_distances):.1f}m\n"
    if hasattr(channel, 'max_tx_rx_distance') and channel.max_tx_rx_distance is not None:
        stats_text += f"Violations: {violations}/{len(channel.tx_rx_pairs)}"
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved visualization: {save_path}")
        plt.close()
    else:
        plt.show()


def test_channel_version(
    ChannelClass,
    version_name: str,
    n_links: int,
    deployment_range: float,
    min_tx_tx_distance: float = 10.0,
    max_tx_rx_distance: float = 50.0,
    seed: int = 42,
    visualize: bool = False,
    save_dir: str = None
):
    """Test a channel version and report statistics."""
    print(f"\n{'='*70}")
    print(f"Testing {version_name}")
    print(f"{'='*70}")
    print(f"Configuration:")
    print(f"  n_links: {n_links}")
    print(f"  deployment_range: {deployment_range}m")
    print(f"  min_tx_tx_distance: {min_tx_tx_distance}m")
    print(f"  max_tx_rx_distance: {max_tx_rx_distance}m")
    print(f"  seed: {seed}")
    print()
    
    try:
        channel = ChannelClass(
            n_links=n_links,
            deployment_range=deployment_range,
            min_tx_tx_distance=min_tx_tx_distance,
            max_tx_rx_distance=max_tx_rx_distance,
            seed=seed
        )
        
        # Compute TX-RX distances for paired links
        all_distances = distance.cdist(channel.tx_locations, channel.rx_locations, 'euclidean')
        
        paired_distances = []
        for tx_idx in range(n_links):
            rx_idx = np.where(channel.associations[tx_idx, :])[0]
            if len(rx_idx) > 0:
                paired_distances.append(all_distances[tx_idx, rx_idx[0]])
        
        paired_distances = np.array(paired_distances)
        
        # Check for violations
        violations = np.sum(paired_distances > max_tx_rx_distance)
        
        print(f"\nResults:")
        print(f"  ✓ Deployment successful!")
        print(f"  Paired TX-RX distances:")
        print(f"    Mean: {np.mean(paired_distances):.2f}m")
        print(f"    Min: {np.min(paired_distances):.2f}m")
        print(f"    Max: {np.max(paired_distances):.2f}m")
        print(f"    Std: {np.std(paired_distances):.2f}m")
        print(f"  Constraint violations: {violations}/{n_links} pairs exceed {max_tx_rx_distance}m")
        
        # V3-specific information
        if hasattr(channel, 'get_subdivision_summary'):
            summary = channel.get_subdivision_summary()
            print(f"\n  Subdivision Statistics:")
            print(f"    Max depth reached: {summary['max_depth_reached']}")
            print(f"    Total subdivisions: {summary['total_subdivisions']}")
            print(f"    Successful subnetworks: {summary['successful_subnetworks']}")
            print(f"    Failed attempts: {summary['failed_subnetworks']}")
        
        # Visualize if requested
        if visualize and save_dir:
            os.makedirs(save_dir, exist_ok=True)
            version_short = version_name.split()[0].lower().replace('wireless', '').replace('channel', '')
            save_path = os.path.join(save_dir, f'{version_short}_{n_links}links_seed{seed}.png')
            visualize_network(channel, version_name, save_path)
        
        return True, channel
        
    except Exception as e:
        print(f"\n✗ Deployment failed!")
        print(f"  Error: {str(e)}")
        return False, None


def main():
    """Run comparison tests with visualization."""
    print("="*70)
    print("WirelessChannel Version Comparison Test")
    print("="*70)
    
    # Create output directory for visualizations
    script_dir = os.path.dirname(os.path.abspath(__file__))
    viz_dir = os.path.join(script_dir, '../figs/channel_v3_test')
    
    # Test 1: Small network (all versions should succeed)
    print("\n" + "="*70)
    print("TEST 1: Small Network (20 links, relaxed constraints)")
    print("="*70)
    
    n_links_small = 20
    deployment_range_small = 200.0
    min_tx_tx_small = 10.0
    max_distance = 50.0
    seed = 42
    
    success_v1, channel_v1 = test_channel_version(
        WirelessChannel, "WirelessChannel V1 (Simple Fallback)",
        n_links_small, deployment_range_small, min_tx_tx_small, max_distance, seed,
        visualize=True, save_dir=viz_dir
    )
    
    success_v2, channel_v2 = test_channel_version(
        WirelessChannelV2, "WirelessChannelV2 (Surgical Redeployment)",
        n_links_small, deployment_range_small, min_tx_tx_small, max_distance, seed,
        visualize=True, save_dir=viz_dir
    )
    
    success_v3, channel_v3 = test_channel_version(
        WirelessChannelV3, "WirelessChannelV3 (Recursive Subdivision)",
        n_links_small, deployment_range_small, min_tx_tx_small, max_distance, seed,
        visualize=True, save_dir=viz_dir
    )
    
    # Test 2: Challenging network (V1/V2 may fail, V3 should succeed via subdivision)
    print("\n" + "="*70)
    print("TEST 2: Challenging Network (80 links, tight spatial constraints)")
    print("="*70)
    print("Note: This configuration stresses deployment with tight packing.")
    print("V1/V2 may struggle, but V3 should succeed through recursive subdivision.")
    
    n_links_large = 80
    deployment_range_large = 200.0
    min_tx_tx_tight = 15.0  # Tighter spacing
    max_distance_tight = 40.0
    seed_challenging = 123
    
    print("\n--- V1: Simple Fallback ---")
    success_v1_large, _ = test_channel_version(
        WirelessChannel, "WirelessChannel V1",
        n_links_large, deployment_range_large, min_tx_tx_tight, max_distance_tight, seed_challenging,
        visualize=False, save_dir=viz_dir
    )
    
    print("\n--- V2: Surgical Redeployment ---")
    success_v2_large, _ = test_channel_version(
        WirelessChannelV2, "WirelessChannelV2",
        n_links_large, deployment_range_large, min_tx_tx_tight, max_distance_tight, seed_challenging,
        visualize=False, save_dir=viz_dir
    )
    
    print("\n--- V3: Recursive Subdivision ---")
    success_v3_large, channel_v3_large = test_channel_version(
        WirelessChannelV3, "WirelessChannelV3",
        n_links_large, deployment_range_large, min_tx_tx_tight, max_distance_tight, seed_challenging,
        visualize=True, save_dir=viz_dir
    )
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nSmall Network (20 links, relaxed constraints):")
    print(f"  V1: {'✓ Success' if success_v1 else '✗ Failed'}")
    print(f"  V2: {'✓ Success' if success_v2 else '✗ Failed'}")
    print(f"  V3: {'✓ Success' if success_v3 else '✗ Failed'}")
    
    print(f"\nChallenging Network (80 links, tight constraints):")
    print(f"  V1: {'✓ Success' if success_v1_large else '✗ Failed'}")
    print(f"  V2: {'✓ Success' if success_v2_large else '✗ Failed'}")
    print(f"  V3: {'✓ Success' if success_v3_large else '✗ Failed'}")
    
    print("\nKey Takeaways:")
    print("• V1: Fast but may violate distance constraints")
    print("• V2: Enforces constraints via surgical redeployment")
    print("• V3: Most robust - handles difficult cases via recursive subdivision")
    print("• V3 is recommended for large networks with tight constraints")
    
    print(f"\n✓ Network visualizations saved to: {viz_dir}")


if __name__ == "__main__":
    main()
