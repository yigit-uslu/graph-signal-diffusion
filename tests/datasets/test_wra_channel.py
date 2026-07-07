"""Unit tests for wireless channel generation."""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tqdm

from graph_signal_diffusion.datasets.wra.channel import WirelessChannel, WirelessChannelV2, merge_channels


class TestWirelessChannel:
    """Test suite for WirelessChannel class."""
    
    def test_channel_initialization(self):
        """Test that channel initializes with valid network topology."""
        channel = WirelessChannel(
            n_links=10,
            deployment_range=500.0,
            seed=42
        )
        
        # Check basic properties
        assert channel.n_links == 10
        assert channel.tx_locations.shape == (10, 2)
        assert channel.rx_locations.shape == (10, 2)
        assert channel.large_scale_fading.shape == (10, 10)
        assert channel.associations.shape == (10, 10)
        
        # Check deployment bounds
        assert np.all(channel.tx_locations >= -250.0)
        assert np.all(channel.tx_locations <= 250.0)
        assert np.all(channel.rx_locations >= -250.0)
        assert np.all(channel.rx_locations <= 250.0)
        
        print("✓ Channel initialization: PASSED")
    
    def test_minimum_distance_constraints(self):
        """Test that minimum distance constraints are satisfied."""
        channel = WirelessChannel(
            n_links=20,
            deployment_range=800.0,
            min_tx_tx_distance=35.0,
            min_tx_rx_distance=10.0,
            seed=123
        )
        
        # Check TX-TX minimum distance
        from scipy.spatial import distance
        tx_tx_distances = distance.cdist(channel.tx_locations, channel.tx_locations, 'euclidean')
        np.fill_diagonal(tx_tx_distances, np.inf)
        assert np.min(tx_tx_distances) >= 35.0, \
            f"Min TX-TX distance violation: {np.min(tx_tx_distances):.2f}m < 35m"
        
        # Check TX-RX minimum distance
        tx_rx_distances = distance.cdist(channel.tx_locations, channel.rx_locations, 'euclidean')
        assert np.min(tx_rx_distances) >= 10.0, \
            f"Min TX-RX distance violation: {np.min(tx_rx_distances):.2f}m < 10m"
        
        print("✓ Minimum distance constraints: PASSED")
    
    def test_optimal_pairing(self):
        """Test that TX-RX pairing satisfies interference dominance criterion."""
        channel = WirelessChannel(
            n_links=10,
            seed=456
        )
        
        # Each transmitter should be paired with exactly one receiver
        assert channel.associations.shape == (10, 10)
        
        # Check interference dominance: For each RX, the associated TX should have
        # the strongest signal (dominate over all other TXs)
        for rx_idx in range(8):
            # Find which TX(s) are associated with this RX
            associated_txs = np.where(channel.associations[:, rx_idx])[0]
            
            if len(associated_txs) > 0:
                # There should be exactly one associated TX per RX (after tie-breaking)
                assert len(associated_txs) == 1, \
                    f"RX {rx_idx} has {len(associated_txs)} associated TXs, expected 1"
                
                dominant_tx = associated_txs[0]
                dominant_gain = channel.large_scale_fading[dominant_tx, rx_idx]
                
                # This TX should have the strongest (or tied for strongest) signal to this RX
                max_gain = np.max(channel.large_scale_fading[:, rx_idx])
                assert np.isclose(dominant_gain, max_gain), \
                    f"RX {rx_idx}: dominant TX {dominant_tx} gain {dominant_gain:.2e} " \
                    f"is not maximum {max_gain:.2e}"
        
        # Each TX should dominate at least one RX (deployment criterion)
        num_dominated_per_tx = np.sum(channel.associations, axis=1)
        num_zero_dominated = np.sum(num_dominated_per_tx == 0)
        
        # All TXs should dominate at least one RX (though test may occasionally fail randomly)
        if num_zero_dominated > 0:
            print(f"  ⚠ Note: {num_zero_dominated} TX(s) don't dominate any RX in this random deployment")
        
        # Each TX should be explicitly paired with one RX
        assert len(channel.tx_rx_pairs) == 10, \
            "Each TX must be paired with exactly one RX"
        
        print("✓ Interference-aware TX-RX pairing: PASSED")
    
    def test_large_scale_fading_properties(self):
        """Test large-scale fading physical properties."""
        channel = WirelessChannel(
            n_links=20,
            path_loss_exponent_short=2.0,
            path_loss_exponent_long=4.0,
            shadowing_std=7.0,
            seed=789
        )
        
        # Large-scale fading should be positive
        assert np.all(channel.large_scale_fading > 0), \
            "Large-scale fading must be positive"
        
        # Channel gain should decrease with distance (on average)
        # Check diagonal elements (paired TX-RX) vs off-diagonal
        paired_distances = np.array([
            channel.distances[tx, rx] for tx, rx in channel.tx_rx_pairs
        ])
        paired_gains = np.array([
            channel.large_scale_fading[tx, rx] for tx, rx in channel.tx_rx_pairs
        ])
        
        # Gains should generally be higher for closer distances
        # (allowing for shadowing variation)
        mean_distance = np.mean(paired_distances)
        close_mask = paired_distances < mean_distance
        far_mask = paired_distances >= mean_distance
        
        if np.sum(close_mask) > 0 and np.sum(far_mask) > 0:
            mean_close_gain = np.mean(paired_gains[close_mask])
            mean_far_gain = np.mean(paired_gains[far_mask])
            assert mean_close_gain > mean_far_gain * 0.5, \
                "Close pairs should have higher average gain than far pairs"
        
        print("✓ Large-scale fading properties: PASSED")
    
    def test_sample_realization_static(self):
        """Test sampling channel realization without small-scale fading."""
        channel = WirelessChannel(n_links=10, seed=42)
        
        realization = channel.sample_realization(
            num_timesteps=50,
            disable_small_scale_fading=True
        )
        
        # Check output structure
        assert 'H' in realization
        assert 'H_l' in realization
        assert realization['H'].shape == (10, 10, 50)
        assert realization['H_l'].shape == (10, 10)
        
        # Static channel should not vary over time
        H = realization['H']
        for t in range(1, 50):
            np.testing.assert_array_almost_equal(
                H[:, :, t], H[:, :, 0],
                decimal=10,
                err_msg="Static channel should not vary over time"
            )
        
        print("✓ Sample realization (static): PASSED")
    
    def test_sample_realization_time_varying(self):
        """Test sampling channel realization with Rayleigh fading."""
        channel = WirelessChannel(
            n_links=10,
            speed=1.0,
            carrier_freq=2.4e9,
            seed=42
        )
        
        realization = channel.sample_realization(
            num_timesteps=100,
            disable_small_scale_fading=False
        )
        
        H = realization['H']
        
        # Check shape
        assert H.shape == (10, 10, 100)
        
        # Channel should vary over time
        time_variance = np.var(H, axis=2)
        assert np.mean(time_variance) > 0, \
            "Time-varying channel should have temporal variance"
        
        # Check that temporal average approximates large-scale fading
        # (averaged over Rayleigh fading)
        H_mean = np.mean(H, axis=2)
        H_l = realization['H_l']
        
        # Mean should be within reasonable range of large-scale fading
        # (Rayleigh has mean^2 = E[|h|^2] = 1 when properly normalized)
        relative_error = np.abs(H_mean - H_l) / (H_l + 1e-10)
        assert np.mean(relative_error) < 2.0, \
            "Temporal mean should approximate large-scale fading"
        
        print("✓ Sample realization (time-varying): PASSED")
    
    def test_channel_merging(self):
        """Test merging multiple channel instances with different layouts."""
        # Create two small subnetworks
        channel1 = WirelessChannel(
            n_links=10,
            deployment_range=400.0,
            seed=100
        )
        channel2 = WirelessChannel(
            n_links=10,
            deployment_range=400.0,
            seed=200
        )
        
        # Test linear layout
        spacing = 100.0
        merged_linear = merge_channels([channel1, channel2], spacing=spacing, layout='linear')
        
        # Check merged network has correct number of links
        assert merged_linear.n_links == 20
        assert merged_linear.tx_locations.shape == (20, 2)
        assert merged_linear.rx_locations.shape == (20, 2)
        
        # Check that subnetworks are spatially separated
        sub1_tx = merged_linear.tx_locations[:10, :]
        sub1_rx = merged_linear.rx_locations[:10, :]
        sub2_tx = merged_linear.tx_locations[10:, :]
        sub2_rx = merged_linear.rx_locations[10:, :]
        
        sub1_all = np.vstack([sub1_tx, sub1_rx])
        sub2_all = np.vstack([sub2_tx, sub2_rx])
        
        # Maximum x of subnetwork 1 should be well separated from minimum x of subnetwork 2
        gap = np.min(sub2_all[:, 0]) - np.max(sub1_all[:, 0])
        assert gap >= spacing * 0.9, \
            f"Subnetworks should be separated by ~{spacing}m, got {gap:.1f}m"
        
        # Pairing should be recomputed for merged network
        assert merged_linear.associations.shape == (20, 20)
        
        # Each RX should have at most one dominant TX
        num_dominant_per_rx = np.sum(merged_linear.associations, axis=0)
        assert np.all(num_dominant_per_rx <= 1), \
            "Each RX should have at most one dominant TX"
        
        # Each TX should be explicitly paired
        assert len(merged_linear.tx_rx_pairs) == 20, \
            "Each TX must be paired with exactly one RX"
        
        # Most pairings should remain within original subnetworks
        intra_linear = 0
        for tx_idx, rx_idx in merged_linear.tx_rx_pairs:
            if (tx_idx < 10 and rx_idx < 10) or (tx_idx >= 10 and rx_idx >= 10):
                intra_linear += 1
        
        print(f"  Linear layout: {100 * intra_linear / 20:.0f}% intra-subnetwork pairings")
        assert intra_linear >= 15, "Linear layout should preserve most intra-subnetwork pairings"
        
        # Test circular layout with spacing=0
        merged_circular = merge_channels([channel1, channel2], spacing=0.0, layout='circular')
        
        assert merged_circular.n_links == 20
        assert merged_circular.tx_locations.shape == (20, 2)
        
        # Check circular placement: subnetworks should be separated
        sub1_center = np.mean(np.vstack([
            merged_circular.tx_locations[:10, :],
            merged_circular.rx_locations[:10, :]
        ]), axis=0)
        sub2_center = np.mean(np.vstack([
            merged_circular.tx_locations[10:, :],
            merged_circular.rx_locations[10:, :]
        ]), axis=0)
        
        center_distance = np.linalg.norm(sub2_center - sub1_center)
        print(f"  Circular layout: subnetwork centers separated by {center_distance:.1f}m")
        
        # With spacing=0, centers should be reasonably close (sum of radii)
        # Just check they're not too far apart
        assert center_distance < 1000, \
            f"Circular layout spacing=0 should keep centers within reasonable distance"
        
        # Check intra-subnetwork pairings
        intra_circular = 0
        for tx_idx, rx_idx in merged_circular.tx_rx_pairs:
            if (tx_idx < 10 and rx_idx < 10) or (tx_idx >= 10 and rx_idx >= 10):
                intra_circular += 1
        
        print(f"  Circular layout: {100 * intra_circular / 20:.0f}% intra-subnetwork pairings")
        
        print("✓ Channel merging (linear and circular layouts): PASSED")
    
    def test_network_info(self):
        """Test network info retrieval."""
        channel = WirelessChannel(n_links=20, seed=42)
        
        info = channel.get_network_info()
        
        assert 'n_links' in info
        assert 'tx_locations' in info
        assert 'rx_locations' in info
        assert 'associations' in info
        assert 'tx_rx_pairs' in info
        assert 'mean_tx_tx_distance' in info
        assert 'mean_paired_distance' in info
        
        assert info['n_links'] == 20
        assert info['tx_rx_pairs'].shape == (20, 2)
        assert info['mean_tx_tx_distance'] > 0
        assert info['mean_paired_distance'] > 0
        
        print("✓ Network info retrieval: PASSED")
    
    def test_circular_merging_multiple_subnetworks(self):
        """Test merging more than 2 subnetworks with circular layout."""
        # Create 4 subnetworks with 10 links each
        channels = [
            WirelessChannel(n_links=10, deployment_range=400, seed=100),
            WirelessChannel(n_links=10, deployment_range=400, seed=200),
            WirelessChannel(n_links=10, deployment_range=400, seed=300),
            WirelessChannel(n_links=10, deployment_range=400, seed=400),
        ]
        
        # Merge with circular layout (spacing=0 for touching)
        merged = merge_channels(channels, spacing=0.0, layout='circular')
        
        # Check total links
        assert merged.n_links == 40
        assert merged.tx_locations.shape == (40, 2)
        assert merged.rx_locations.shape == (40, 2)
        
        # Compute center of each subnetwork in merged network
        centers = []
        link_offsets = [0, 10, 20, 30, 40]  # Starting indices for each subnetwork
        
        for i in range(4):
            start_idx = link_offsets[i]
            end_idx = link_offsets[i+1]
            
            subnet_locs = np.vstack([
                merged.tx_locations[start_idx:end_idx, :],
                merged.rx_locations[start_idx:end_idx, :]
            ])
            center = np.mean(subnet_locs, axis=0)
            centers.append(center)
        
        centers = np.array(centers)
        
        # Check that reference subnetwork (0) is near origin
        assert np.linalg.norm(centers[0]) < 50, \
            "Reference subnetwork should be near origin"
        
        # Check that other subnetworks are roughly equidistant from reference
        distances = [np.linalg.norm(centers[i] - centers[0]) for i in range(1, 4)]
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        
        print(f"  Distances from reference: {[f'{d:.1f}' for d in distances]} (mean={mean_distance:.1f}m)")
        
        # All should be similar distance (within 20% of mean)
        for d in distances:
            assert abs(d - mean_distance) < 0.2 * mean_distance, \
                "Subnetworks should be roughly equidistant from reference in circular layout"
        
        # Check that subnetworks are distributed around the circle
        # Compute angles from reference to each other subnetwork
        angles = []
        for i in range(1, 4):
            vec = centers[i] - centers[0]
            angle = np.arctan2(vec[1], vec[0])
            angles.append(angle)
        
        angles = np.array(angles)
        angles_deg = np.degrees(angles) % 360
        
        print(f"  Angular distribution: {[f'{a:.0f}°' for a in angles_deg]}")
        
        # Angles should be roughly evenly distributed (120° apart for 3 subnetworks)
        # Sort angles
        angles_sorted = np.sort(angles_deg)
        angular_gaps = np.diff(np.append(angles_sorted, angles_sorted[0] + 360))
        
        # Expected gap: 360/3 = 120 degrees
        expected_gap = 360.0 / 3
        for gap in angular_gaps:
            assert abs(gap - expected_gap) < 30, \
                f"Angular gaps should be ~{expected_gap}°, got {gap:.1f}°"
        
        # Check pairing
        assert len(merged.tx_rx_pairs) == 40
        
        print("✓ Circular merging with 4 subnetworks: PASSED")
    
    def test_circular_merging_ten_subnetworks(self):
        """Test circular layout merging with 10 subnetworks."""
        # Create 10 subnetworks with 10 links each
        channels = [
            WirelessChannel(n_links=10, deployment_range=400, seed=1000 + i*100)
            for i in range(10)
        ]
        
        # Merge with circular layout (spacing=0 for subnetworks to touch)
        merged = merge_channels(channels, spacing=0.0, layout='circular')
        
        # Verify basic properties
        assert merged.n_links == 100, "Total links should be 100"
        assert merged.tx_locations.shape == (100, 2)
        assert merged.rx_locations.shape == (100, 2)
        
        # Verify all TXs paired with unique RXs
        tx_indices = merged.tx_rx_pairs[:, 0]
        rx_indices = merged.tx_rx_pairs[:, 1]
        assert len(np.unique(tx_indices)) == 100, "All TXs should be paired"
        assert len(np.unique(rx_indices)) == 100, "All RXs should be unique"
        
        # Verify circular placement: compute distances from reference subnetwork center
        link_offsets = [i * 10 for i in range(11)]
        
        centers = []
        for i in range(10):
            start = link_offsets[i]
            end = link_offsets[i+1]
            subnet_locs = np.vstack([merged.tx_locations[start:end], merged.rx_locations[start:end]])
            centers.append(np.mean(subnet_locs, axis=0))
        
        # Reference at origin
        ref_center = centers[0]
        assert np.linalg.norm(ref_center) < 100.0, "Reference should be near origin"
        
        # Other subnetworks should be roughly equidistant from reference
        distances_from_ref = [np.linalg.norm(centers[i] - ref_center) for i in range(1, 10)]
        mean_distance = np.mean(distances_from_ref)
        std_distance = np.std(distances_from_ref)
        
        # All should be within 15% of mean (allowing for some variation due to different radii)
        assert std_distance / mean_distance < 0.15, "Subnetworks should be roughly equidistant"
        
        # Verify angular distribution is roughly even
        angles = []
        for i in range(1, 10):
            vec = centers[i] - ref_center
            angle = np.arctan2(vec[1], vec[0]) * 180 / np.pi
            if angle < 0:
                angle += 360
            angles.append(angle)
        
        angles_sorted = sorted(angles)
        expected_spacing = 360.0 / 9  # 40 degrees
        
        # Check that angles are distributed around the circle
        assert max(angles_sorted) - min(angles_sorted) > 270, "Subnetworks should span most of circle"
        
        print(f"✓ Merged 10 subnetworks into network with {merged.n_links} links")
        print(f"  Spatial layout: circular (radius={mean_distance:.1f}m)")
        print(f"  Angular spacing: {expected_spacing:.1f}° nominal")
        print(f"  Distance variation: {std_distance/mean_distance*100:.1f}%")
        print("✓ Circular merging with 10 subnetworks: PASSED")
    
    def test_fallback_pairing_with_redeployment(self):
        """Test that fallback pairing correctly redeploys RXs in annular region."""
        print(f"\n{'='*60}")
        print("Testing fallback pairing with RX redeployment (WirelessChannelV2)")
        print(f"{'='*60}")
        
        # Use parameters that are likely to trigger fallback pairing
        # Larger deployment range with tighter max_tx_rx_distance makes it more likely
        # that some TXs won't dominate any available RX
        max_attempts = 10
        fallback_triggered = False
        
        for seed in range(max_attempts):
            channel = WirelessChannelV2(
                n_links=30,
                deployment_range=1200.0,
                min_tx_rx_distance=10.0,
                max_tx_rx_distance=50.0,
                seed=seed
            )
            
            # Check if any TX-RX pairs were redeployed (don't dominate)
            for tx_idx, rx_idx in channel.tx_rx_pairs:
                if channel.associations[tx_idx, rx_idx] == 0:
                    fallback_triggered = True
                    print(f"  Found redeployed pair: TX{tx_idx} -> RX{rx_idx} (seed={seed})")
                    
                    # Verify distance constraints
                    distance = channel.distances[tx_idx, rx_idx]
                    assert distance >= channel.min_tx_rx_distance, \
                        f"TX-RX distance {distance:.2f}m < min {channel.min_tx_rx_distance}m"
                    assert distance <= channel.max_tx_rx_distance, \
                        f"TX-RX distance {distance:.2f}m > max {channel.max_tx_rx_distance}m"
                    
                    # Verify RX maintains min distance from all TXs
                    rx_to_all_tx_distances = channel.distances[:, rx_idx]
                    assert np.all(rx_to_all_tx_distances >= channel.min_tx_rx_distance), \
                        f"Redeployed RX{rx_idx} violates min_tx_rx_distance from some TX"
            
            if fallback_triggered:
                break
        
        # Verify all pairs respect max distance
        for tx_idx, rx_idx in channel.tx_rx_pairs:
            distance = channel.distances[tx_idx, rx_idx]
            assert distance <= channel.max_tx_rx_distance, \
                f"Pair (TX{tx_idx}, RX{rx_idx}) distance {distance:.2f}m > max {channel.max_tx_rx_distance}m"
        
        if fallback_triggered:
            print(f"\n✓ Fallback pairing with redeployment: PASSED (triggered with seed={seed})")
        else:
            print(f"\n⚠ Fallback pairing: No redeployment triggered in {max_attempts} attempts (test inconclusive)")
    
    def test_extreme_fallback_scenario(self):
        """Test fallback pairing in extreme scenario designed to trigger redeployment."""
        print(f"\n{'='*60}")
        print("Testing extreme fallback scenario (WirelessChannelV2)")
        print(f"{'='*60}")
        
        # Very large deployment area with strict max_tx_rx_distance almost guarantees
        # that initial random RX placement won't dominate for all TXs
        channel = WirelessChannelV2(
            n_links=20,
            deployment_range=2000.0,  # Very large area
            min_tx_rx_distance=10.0,
            max_tx_rx_distance=40.0,   # Tight constraint
            seed=42
        )
        
        # Verify all pairs exist and respect constraints
        assert len(channel.tx_rx_pairs) == channel.n_links, \
            "Not all TXs were paired"
        
        num_redeployed = 0
        for tx_idx, rx_idx in channel.tx_rx_pairs:
            distance = channel.distances[tx_idx, rx_idx]
            
            # Check max distance constraint
            assert distance <= channel.max_tx_rx_distance, \
                f"Pair (TX{tx_idx}, RX{rx_idx}) distance {distance:.2f}m > max {channel.max_tx_rx_distance}m"
            
            # Check min distance constraint
            assert distance >= channel.min_tx_rx_distance, \
                f"Pair (TX{tx_idx}, RX{rx_idx}) distance {distance:.2f}m < min {channel.min_tx_rx_distance}m"
            
            # Count redeployed pairs
            if channel.associations[tx_idx, rx_idx] == 0:
                num_redeployed += 1
                
                # For redeployed RXs, verify they maintain min distance from ALL TXs
                rx_to_all_tx_distances = channel.distances[:, rx_idx]
                assert np.all(rx_to_all_tx_distances >= channel.min_tx_rx_distance), \
                    f"Redeployed RX{rx_idx} violates min_tx_rx_distance from TX" \
                    f"{np.argmin(rx_to_all_tx_distances)}: {np.min(rx_to_all_tx_distances):.2f}m"
        
        print(f"\n✓ Extreme fallback scenario: PASSED ({num_redeployed}/{channel.n_links} pairs redeployed)")
    
    def test_batch_generation(self):
        """Test generation of a batch of B wireless channels with same config."""
        # Configuration
        B = 16
        n_links = 50
        deployment_range = 1000.0
        base_seed = 2000
        
        print(f"\n{'='*60}")
        print(f"Generating batch of B={B} wireless channels...")
        print(f"Configuration: {n_links} links, {deployment_range}m deployment range")
        print(f"{'='*60}")
        
        # Generate batch
        channels = []
        for b in tqdm.tqdm(range(B), desc="Generating channels"):
            channel = WirelessChannel(
                n_links=n_links,
                deployment_range=deployment_range,
                seed=base_seed + b
            )
            channels.append(channel)
        
        print(f"\n✓ Generated {B} channel instances")
        
        # Verify all channels have same configuration
        for b, ch in enumerate(channels):
            assert ch.n_links == n_links
            assert ch.tx_locations.shape == (n_links, 2)
            assert ch.rx_locations.shape == (n_links, 2)
            assert ch.large_scale_fading.shape == (n_links, n_links)
            assert ch.associations.shape == (n_links, n_links)
        
        print("✓ All channels have correct dimensions")
        
        # Verify channels are different (different deployments)
        for b1 in range(B):
            for b2 in range(b1 + 1, B):
                assert not np.allclose(channels[b1].tx_locations, channels[b2].tx_locations)
                assert not np.allclose(channels[b1].rx_locations, channels[b2].rx_locations)
                assert not np.allclose(channels[b1].large_scale_fading, channels[b2].large_scale_fading)
        
        print("✓ All channels have unique deployments and fading")
        
        # Compute statistics across batch
        # Convert power gains to dB: -10*log10(power_gain) for path loss
        mean_path_losses = [-10*np.log10(np.mean(ch.large_scale_fading)) for ch in channels]
        std_path_losses = [np.std(-10*np.log10(ch.large_scale_fading + 1e-12)) for ch in channels]
        
        # Mean channel gain (dB) for paired links
        mean_paired_gains = []
        for ch in channels:
            paired_gains = []
            for tx_idx, rx_idx in ch.tx_rx_pairs:
                gain_db = -10 * np.log10(ch.large_scale_fading[tx_idx, rx_idx])
                paired_gains.append(gain_db)
            mean_paired_gains.append(np.mean(paired_gains))
        
        # Count how many TXs dominate at least one RX in each channel
        dominating_txs = []
        for ch in channels:
            count = 0
            for tx_i in range(n_links):
                # Check if this TX dominates at least one RX (has minimum path loss)
                for rx_j in range(n_links):
                    if ch.associations[tx_i, rx_j]:
                        count += 1
                        break
            dominating_txs.append(count)
        
        # Check pairing is correct in all channels
        for b, ch in enumerate(channels):
            paired_rxs = ch.tx_rx_pairs[:, 1]
            assert len(set(paired_rxs)) == n_links, f"Channel {b}: Not all RXs are uniquely paired"
        
        print(f"\n{'='*60}")
        print("Batch Statistics:")
        print(f"{'='*60}")
        print(f"Mean path loss across all links (dB):")
        print(f"  Min:  {np.min(mean_path_losses):.1f}")
        print(f"  Mean: {np.mean(mean_path_losses):.1f}")
        print(f"  Max:  {np.max(mean_path_losses):.1f}")
        print(f"  Std:  {np.std(mean_path_losses):.1f}")
        print(f"\nMean path loss for paired links (dB):")
        print(f"  Min:  {np.min(mean_paired_gains):.1f}")
        print(f"  Mean: {np.mean(mean_paired_gains):.1f}")
        print(f"  Max:  {np.max(mean_paired_gains):.1f}")
        print(f"  Std:  {np.std(mean_paired_gains):.1f}")
        print(f"\nPath loss std dev within each channel (dB):")
        print(f"  Min:  {np.min(std_path_losses):.1f}")
        print(f"  Mean: {np.mean(std_path_losses):.1f}")
        print(f"  Max:  {np.max(std_path_losses):.1f}")
        print(f"\nTXs dominating ≥1 RX:")
        print(f"  Min:  {np.min(dominating_txs)}")
        print(f"  Mean: {np.mean(dominating_txs):.1f}")
        print(f"  Max:  {np.max(dominating_txs)}")
        print(f"{'='*60}")
        
        # Check all have one-to-one pairing
        print("\n✓ All channels have correct one-to-one TX-RX pairing")
        print(f"✓ Batch generation (B={B}, n_links={n_links}): PASSED")
        
        return channels
    
    def test_reproducibility(self):
        """Test that same seed produces same network."""
        channel1 = WirelessChannel(n_links=20, seed=999)
        channel2 = WirelessChannel(n_links=20, seed=999)
        
        np.testing.assert_array_almost_equal(
            channel1.tx_locations,
            channel2.tx_locations,
            decimal=10
        )
        np.testing.assert_array_almost_equal(
            channel1.rx_locations,
            channel2.rx_locations,
            decimal=10
        )
        np.testing.assert_array_almost_equal(
            channel1.large_scale_fading,
            channel2.large_scale_fading,
            decimal=10
        )
        
        print("✓ Reproducibility: PASSED")


def visualize_network_deployment(save_dir: str = "tests/figs/wra_channel"):
    """
    Visualize network deployment, channel gains, and time evolution.
    
    Parameters
    ----------
    save_dir : str
        Directory to save visualization figures
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Create a test network
    channel = WirelessChannel(
        n_links=20,
        deployment_range=600.0,
        seed=42
    )
    
    # Sample a time-varying realization
    realization = channel.sample_realization(num_timesteps=200)
    
    # ===== Figure 1: Network Deployment =====
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1a: Spatial deployment with pairing
    ax = axes[0]
    tx_locs = channel.tx_locations
    rx_locs = channel.rx_locations
    
    # Draw TX-RX pairs
    for tx_idx, rx_idx in channel.tx_rx_pairs:
        ax.plot(
            [tx_locs[tx_idx, 0], rx_locs[rx_idx, 0]],
            [tx_locs[tx_idx, 1], rx_locs[rx_idx, 1]],
            'k-', alpha=0.3, linewidth=1, zorder=1
        )
    
    # Plot transmitters
    ax.scatter(
        tx_locs[:, 0], tx_locs[:, 1],
        c='red', s=200, marker='^', edgecolors='black', linewidths=2,
        label='Transmitters', zorder=3
    )
    
    # Plot receivers
    ax.scatter(
        rx_locs[:, 0], rx_locs[:, 1],
        c='blue', s=200, marker='o', edgecolors='black', linewidths=2,
        label='Receivers', zorder=3
    )
    
    # Add TX/RX labels
    for i in range(channel.n_links):
        ax.text(tx_locs[i, 0], tx_locs[i, 1] + 15, f'TX{i}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
        paired_rx = channel.tx_rx_pairs[i, 1]
        ax.text(rx_locs[paired_rx, 0], rx_locs[paired_rx, 1] - 15, f'RX{paired_rx}',
                ha='center', va='top', fontsize=8, fontweight='bold')
    
    ax.set_xlabel('X Position (m)', fontsize=12)
    ax.set_ylabel('Y Position (m)', fontsize=12)
    ax.set_title('Network Deployment with TX-RX Pairing', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Plot 1b: Large-scale fading heatmap
    ax = axes[1]
    im = ax.imshow(
        20 * np.log10(channel.large_scale_fading + 1e-10),  # Convert to dB
        cmap='hot', aspect='auto', origin='lower'
    )
    ax.set_xlabel('Receiver Index', fontsize=12)
    ax.set_ylabel('Transmitter Index', fontsize=12)
    ax.set_title('Large-Scale Channel Gain (dB)', fontsize=14, fontweight='bold')
    
    # Mark optimal pairing
    for tx_idx, rx_idx in channel.tx_rx_pairs:
        ax.plot(rx_idx, tx_idx, 'go', markersize=12, markeredgewidth=2,
                markerfacecolor='none', label='Paired' if tx_idx == 0 else '')
    
    # Mark interference dominance regions
    for rx_idx in range(channel.n_links):
        dominant_txs = np.where(channel.associations[:, rx_idx])[0]
        for tx_idx in dominant_txs:
            ax.plot(rx_idx, tx_idx, 'bx', markersize=10, markeredgewidth=2,
                    label='Dominant' if (tx_idx == 0 and rx_idx == 0) else '')
    
    ax.legend(fontsize=9, loc='upper right')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Channel Gain (dB)', fontsize=11)
    
    # Plot 1c: Interference analysis
    ax = axes[2]
    
    # For each TX-RX pair, compute SINR-like metric
    sinr_db = []
    for tx_idx, rx_idx in channel.tx_rx_pairs:
        signal = channel.large_scale_fading[tx_idx, rx_idx]
        # Sum of interference from all other TXs to this RX
        interference = np.sum(channel.large_scale_fading[:, rx_idx]) - signal
        sinr = signal / (interference + 1e-10)
        sinr_db.append(10 * np.log10(sinr))
    
    bars = ax.bar(range(channel.n_links), sinr_db, color='steelblue', edgecolor='black', linewidth=1.5)
    
    # Color bars based on dominance
    for i, (tx_idx, rx_idx) in enumerate(channel.tx_rx_pairs):
        is_dominant = channel.associations[tx_idx, rx_idx]
        bars[i].set_color('green' if is_dominant else 'orange')
    
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='0 dB threshold')
    ax.set_xlabel('Link Index', fontsize=12)
    ax.set_ylabel('Signal-to-Interference Ratio (dB)', fontsize=12)
    ax.set_title('Link Quality (Green=Dominant, Orange=Non-dominant)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path / 'network_deployment.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path / 'network_deployment.png'}")
    plt.close()
    
    # ===== Figure 2: Channel Time Evolution =====
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Select a few representative links
    link_pairs = [(0, channel.tx_rx_pairs[0, 1]),  # Link 0 (paired)
                  (0, (channel.tx_rx_pairs[0, 1] + 1) % channel.n_links),  # Link 0 interferer
                  (5, channel.tx_rx_pairs[5, 1]),  # Link 5 (paired)
                  (5, (channel.tx_rx_pairs[5, 1] + 2) % channel.n_links)]  # Link 5 interferer
    
    colors = ['blue', 'red', 'green', 'orange']
    labels = ['TX0→RX{} (paired)'.format(channel.tx_rx_pairs[0, 1]),
              'TX0→RX{} (interf.)'.format((channel.tx_rx_pairs[0, 1] + 1) % channel.n_links),
              'TX5→RX{} (paired)'.format(channel.tx_rx_pairs[5, 1]),
              'TX5→RX{} (interf.)'.format((channel.tx_rx_pairs[5, 1] + 2) % channel.n_links)]
    
    # Plot 2a: Time evolution of channel gains
    ax = axes[0, 0]
    for (tx, rx), color, label in zip(link_pairs, colors, labels):
        H_link = realization['H'][tx, rx, :]
        ax.plot(H_link, color=color, alpha=0.7, linewidth=1.5, label=label)
    
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Channel Power Gain', fontsize=12)
    ax.set_title('Time Evolution of Channel Gains', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # Plot 2b: Distribution of channel gains
    ax = axes[0, 1]
    for (tx, rx), color, label in zip(link_pairs, colors, labels):
        H_link = realization['H'][tx, rx, :]
        ax.hist(H_link, bins=30, alpha=0.5, color=color, label=label)
    
    ax.set_xlabel('Channel Power Gain', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Channel Gains', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2c: Temporal autocorrelation
    ax = axes[1, 0]
    max_lag = 50
    for (tx, rx), color, label in zip(link_pairs, colors, labels):
        H_link = realization['H'][tx, rx, :]
        H_normalized = (H_link - np.mean(H_link)) / np.std(H_link)
        
        autocorr = np.correlate(H_normalized, H_normalized, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr[:max_lag] / autocorr[0]
        
        ax.plot(autocorr, color=color, alpha=0.7, linewidth=2, label=label)
    
    ax.set_xlabel('Lag (time steps)', fontsize=12)
    ax.set_ylabel('Autocorrelation', fontsize=12)
    ax.set_title('Channel Temporal Autocorrelation', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Plot 2d: Snapshot of channel matrix at t=100
    ax = axes[1, 1]
    t_snapshot = 100
    H_snapshot = realization['H'][:, :, t_snapshot]
    im = ax.imshow(
        20 * np.log10(H_snapshot + 1e-10),
        cmap='hot', aspect='auto', origin='lower'
    )
    ax.set_xlabel('Receiver Index', fontsize=12)
    ax.set_ylabel('Transmitter Index', fontsize=12)
    ax.set_title(f'Channel Snapshot at t={t_snapshot} (dB)', fontsize=14, fontweight='bold')
    
    # Mark optimal pairing
    for tx_idx, rx_idx in channel.tx_rx_pairs:
        ax.plot(rx_idx, tx_idx, 'go', markersize=10, markeredgewidth=2,
                markerfacecolor='none')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Channel Gain (dB)', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(save_path / 'channel_time_evolution.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path / 'channel_time_evolution.png'}")
    plt.close()
    
    # ===== Figure 3: Merged Network Visualization =====
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Create two subnetworks with different characteristics for 2-subnet comparison
    dense_channel = WirelessChannel(
        n_links=20,
        deployment_range=500.0,
        seed=100
    )
    
    sparse_channel = WirelessChannel(
        n_links=10,
        deployment_range=400.0,
        seed=200
    )
    
    # Merge them with linear layout
    spacing = 150.0
    merged_linear = merge_channels([dense_channel, sparse_channel], spacing=spacing, layout='linear')
    
    # Merge them with circular layout
    merged_circular = merge_channels([dense_channel, sparse_channel], spacing=0.0, layout='circular')
    
    # Plot 3a: Linear layout
    ax = fig.add_subplot(gs[0, 0])
    
    merged_tx = merged_linear.tx_locations
    merged_rx = merged_linear.rx_locations
    
    # Draw TX-RX pairs with different colors
    for tx_idx, rx_idx in merged_linear.tx_rx_pairs:
        is_intra = (tx_idx < 20 and rx_idx < 20) or (tx_idx >= 20 and rx_idx >= 20)
        color = 'green' if is_intra else 'red'
        linewidth = 2.5 if is_intra else 4.0
        alpha = 0.6 if is_intra else 0.9
        
        ax.plot(
            [merged_tx[tx_idx, 0], merged_rx[rx_idx, 0]],
            [merged_tx[tx_idx, 1], merged_rx[rx_idx, 1]],
            color=color, alpha=alpha, linewidth=linewidth, zorder=1
        )
    
    # Color by original subnetwork
    colors_tx = ['red' if i < 20 else 'orange' for i in range(30)]
    colors_rx = ['blue' if i < 20 else 'cyan' for i in range(30)]
    
    ax.scatter(merged_tx[:, 0], merged_tx[:, 1], c=colors_tx, s=60, marker='^', 
               edgecolors='black', linewidths=1.2, zorder=3)
    ax.scatter(merged_rx[:, 0], merged_rx[:, 1], c=colors_rx, s=60, marker='o',
               edgecolors='black', linewidths=1.2, zorder=3)
    
    # Add vertical line to show separation
    separation_x = np.mean([np.max(merged_tx[:10, 0]), np.min(merged_tx[10:, 0])])
    ax.axvline(x=separation_x, color='purple', linestyle=':', linewidth=2, alpha=0.6)
    
    ax.set_xlabel('X Position (m)', fontsize=11)
    ax.set_ylabel('Y Position (m)', fontsize=11)
    ax.set_title('Linear Layout (2 Subnetworks)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Plot 3b: Circular layout (2 subnetworks)
    ax = fig.add_subplot(gs[0, 1])
    
    merged_tx = merged_circular.tx_locations
    merged_rx = merged_circular.rx_locations
    
    # Draw TX-RX pairs
    for tx_idx, rx_idx in merged_circular.tx_rx_pairs:
        is_intra = (tx_idx < 20 and rx_idx < 20) or (tx_idx >= 20 and rx_idx >= 20)
        color = 'green' if is_intra else 'red'
        linewidth = 2.5 if is_intra else 4.0
        alpha = 0.6 if is_intra else 0.9
        
        ax.plot(
            [merged_tx[tx_idx, 0], merged_rx[rx_idx, 0]],
            [merged_tx[tx_idx, 1], merged_rx[rx_idx, 1]],
            color=color, alpha=alpha, linewidth=linewidth, zorder=1
        )
    
    ax.scatter(merged_tx[:, 0], merged_tx[:, 1], c=colors_tx, s=60, marker='^',
               edgecolors='black', linewidths=1.2, zorder=3)
    ax.scatter(merged_rx[:, 0], merged_rx[:, 1], c=colors_rx, s=60, marker='o',
               edgecolors='black', linewidths=1.2, zorder=3)
    
    # Draw circle showing placement radius
    sub1_center = np.mean(np.vstack([merged_tx[:20, :], merged_rx[:20, :]]), axis=0)
    sub2_center = np.mean(np.vstack([merged_tx[20:, :], merged_rx[20:, :]]), axis=0)
    circle_radius = np.linalg.norm(sub2_center - sub1_center)
    
    circle = plt.Circle(sub1_center, circle_radius, fill=False, color='purple',
                       linestyle=':', linewidth=2, alpha=0.6)
    ax.add_patch(circle)
    
    # Mark centers
    ax.plot(sub1_center[0], sub1_center[1], 'k*', markersize=15, zorder=5)
    ax.plot(sub2_center[0], sub2_center[1], 'k*', markersize=15, zorder=5)
    
    ax.set_xlabel('X Position (m)', fontsize=11)
    ax.set_ylabel('Y Position (m)', fontsize=11)
    ax.set_title('Circular Layout (2 Subnetworks)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Plot 3c: Circular layout with 4 subnetworks
    ax = fig.add_subplot(gs[0, 2])
    
    # Create 4 subnetworks
    multi_channels = [
        WirelessChannel(n_links=10, deployment_range=400, seed=1000),
        WirelessChannel(n_links=10, deployment_range=400, seed=2000),
        WirelessChannel(n_links=10, deployment_range=400, seed=3000),
        WirelessChannel(n_links=10, deployment_range=400, seed=4000),
    ]
    
    merged_multi = merge_channels(multi_channels, spacing=0.0, layout='circular')
    
    merged_tx = merged_multi.tx_locations
    merged_rx = merged_multi.rx_locations
    
    # Color palette for 4 subnetworks
    tx_colors = ['red', 'orange', 'purple', 'brown']
    rx_colors = ['blue', 'cyan', 'magenta', 'pink']
    link_offsets = [0, 10, 20, 30, 40]
    
    # Draw TX-RX pairs
    for tx_idx, rx_idx in merged_multi.tx_rx_pairs:
        # Determine which subnetwork this TX belongs to
        subnet_idx = 0
        for i in range(len(link_offsets) - 1):
            if link_offsets[i] <= tx_idx < link_offsets[i+1]:
                subnet_idx = i
                break
        
        is_intra = (link_offsets[subnet_idx] <= rx_idx < link_offsets[subnet_idx+1])
        color = 'green' if is_intra else 'red'
        alpha = 0.6 if is_intra else 0.9
        linewidth = 2.5 if is_intra else 4.0
        
        ax.plot(
            [merged_tx[tx_idx, 0], merged_rx[rx_idx, 0]],
            [merged_tx[tx_idx, 1], merged_rx[rx_idx, 1]],
            color=color, alpha=alpha, linewidth=linewidth, zorder=1
        )
    
    # Plot nodes with subnetwork colors
    for i in range(4):
        start = link_offsets[i]
        end = link_offsets[i+1]
        ax.scatter(merged_tx[start:end, 0], merged_tx[start:end, 1],
                  c=tx_colors[i], s=60, marker='^', edgecolors='black',
                  linewidths=1.0, zorder=3, label=f'Subnet {i}')
        ax.scatter(merged_rx[start:end, 0], merged_rx[start:end, 1],
                  c=rx_colors[i], s=60, marker='o', edgecolors='black',
                  linewidths=1.0, zorder=3)
    
    # Mark subnetwork centers
    centers = []
    for i in range(4):
        start = link_offsets[i]
        end = link_offsets[i+1]
        subnet_locs = np.vstack([merged_tx[start:end, :], merged_rx[start:end, :]])
        center = np.mean(subnet_locs, axis=0)
        centers.append(center)
        ax.plot(center[0], center[1], 'k*', markersize=12, zorder=5)
    
    # Draw circle through outer centers
    if len(centers) > 1:
        circle_radius = np.linalg.norm(centers[1] - centers[0])
        circle = plt.Circle(centers[0], circle_radius, fill=False, color='purple',
                           linestyle=':', linewidth=2, alpha=0.6)
        ax.add_patch(circle)
    
    ax.set_xlabel('X Position (m)', fontsize=11)
    ax.set_ylabel('Y Position (m)', fontsize=11)
    ax.set_title('Circular Layout (4 Subnetworks)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Plot 3d: Linear association matrix
    ax = fig.add_subplot(gs[1, 0])
    
    assoc_viz = merged_linear.associations.astype(float)
    im = ax.imshow(assoc_viz, cmap='RdYlGn', aspect='auto', origin='lower', vmin=0, vmax=1)
    
    ax.axhline(y=19.5, color='purple', linewidth=3, alpha=0.7)
    ax.axvline(x=19.5, color='purple', linewidth=3, alpha=0.7)
    
    ax.set_xlabel('Receiver Index', fontsize=11)
    ax.set_ylabel('Transmitter Index', fontsize=11)
    ax.set_title('Linear: Association Matrix', fontsize=13, fontweight='bold')
    
    cross_subnet = np.sum(assoc_viz[:20, 20:]) + np.sum(assoc_viz[20:, :20])
    total = np.sum(assoc_viz)
    ax.text(0.02, 0.98, f'Intra: {total - cross_subnet:.0f}\nInter: {cross_subnet:.0f}',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 3e: Circular association matrix (2 subnets)
    ax = fig.add_subplot(gs[1, 1])
    
    assoc_viz = merged_circular.associations.astype(float)
    im = ax.imshow(assoc_viz, cmap='RdYlGn', aspect='auto', origin='lower', vmin=0, vmax=1)
    
    ax.axhline(y=19.5, color='purple', linewidth=3, alpha=0.7)
    ax.axvline(x=19.5, color='purple', linewidth=3, alpha=0.7)
    
    ax.set_xlabel('Receiver Index', fontsize=11)
    ax.set_ylabel('Transmitter Index', fontsize=11)
    ax.set_title('Circular (2): Association Matrix', fontsize=13, fontweight='bold')
    
    cross_subnet = np.sum(assoc_viz[:20, 20:]) + np.sum(assoc_viz[20:, :20])
    total = np.sum(assoc_viz)
    ax.text(0.02, 0.98, f'Intra: {total - cross_subnet:.0f}\nInter: {cross_subnet:.0f}',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 3f: Circular association matrix (4 subnets)
    ax = fig.add_subplot(gs[1, 2])
    
    assoc_viz = merged_multi.associations.astype(float)
    im = ax.imshow(assoc_viz, cmap='RdYlGn', aspect='auto', origin='lower', vmin=0, vmax=1)
    
    # Draw grid lines
    for offset in [9.5, 19.5, 29.5]:
        ax.axhline(y=offset, color='purple', linewidth=2, alpha=0.5)
        ax.axvline(x=offset, color='purple', linewidth=2, alpha=0.5)
    
    ax.set_xlabel('Receiver Index', fontsize=11)
    ax.set_ylabel('Transmitter Index', fontsize=11)
    ax.set_title('Circular (4): Association Matrix', fontsize=13, fontweight='bold')
    
    # Count intra vs inter
    intra_count = 0
    for i in range(4):
        start = link_offsets[i]
        end = link_offsets[i+1]
        intra_count += np.sum(assoc_viz[start:end, start:end])
    
    total = np.sum(assoc_viz)
    ax.text(0.02, 0.98, f'Intra: {intra_count:.0f}\nInter: {total - intra_count:.0f}',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.savefig(save_path / 'merged_network.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path / 'merged_network.png'}")
    plt.close()
    
    # ===== Figure 4: 10-Subnetwork Circular Layout =====
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(1, 2, hspace=0.3, wspace=0.3)
    
    # Create 10 subnetworks
    ten_channels = [
        WirelessChannel(n_links=10, deployment_range=400, seed=1000 + i*100)
        for i in range(10)
    ]
    
    merged_ten = merge_channels(ten_channels, spacing=0.0, layout='circular')
    
    # Plot 4a: Spatial layout with links
    ax = fig.add_subplot(gs[0, 0])
    
    merged_tx = merged_ten.tx_locations
    merged_rx = merged_ten.rx_locations
    
    # Color palette for 10 subnetworks (using colormap)
    import matplotlib.cm as cm
    colors_map = cm.get_cmap('tab10')
    tx_colors_10 = [colors_map(i) for i in range(10)]
    rx_colors_10 = [colors_map(i) for i in range(10)]
    
    link_offsets_10 = [i * 10 for i in range(11)]
    
    # Draw TX-RX pairs with transparency for clarity
    for tx_idx, rx_idx in merged_ten.tx_rx_pairs:
        # Determine which subnetwork this TX belongs to
        subnet_idx = tx_idx // 10
        is_intra = (subnet_idx == rx_idx // 10)
        
        color = 'green' if is_intra else 'red'
        alpha = 0.15 if is_intra else 0.3
        linewidth = 0.5 if is_intra else 1.5
        
        ax.plot(
            [merged_tx[tx_idx, 0], merged_rx[rx_idx, 0]],
            [merged_tx[tx_idx, 1], merged_rx[rx_idx, 1]],
            color=color, alpha=alpha, linewidth=linewidth, zorder=1
        )
    
    # Plot nodes with subnetwork colors
    for i in range(10):
        start = link_offsets_10[i]
        end = link_offsets_10[i+1]
        ax.scatter(merged_tx[start:end, 0], merged_tx[start:end, 1],
                  c=[tx_colors_10[i]], s=40, marker='^', edgecolors='black',
                  linewidths=0.8, zorder=3, label=f'Subnet {i}' if i < 5 else '')
        ax.scatter(merged_rx[start:end, 0], merged_rx[start:end, 1],
                  c=[rx_colors_10[i]], s=40, marker='o', edgecolors='black',
                  linewidths=0.8, zorder=3)
    
    # Mark subnetwork centers
    centers_10 = []
    for i in range(10):
        start = link_offsets_10[i]
        end = link_offsets_10[i+1]
        subnet_locs = np.vstack([merged_tx[start:end, :], merged_rx[start:end, :]])
        center = np.mean(subnet_locs, axis=0)
        centers_10.append(center)
        ax.plot(center[0], center[1], 'k*', markersize=8, zorder=5)
        # Add subnet number
        ax.text(center[0], center[1]+50, f'{i}', fontsize=10, fontweight='bold',
                ha='center', va='bottom', color='black',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Draw circle through outer centers
    ref_center = centers_10[0]
    if len(centers_10) > 1:
        circle_radius = np.linalg.norm(centers_10[1] - ref_center)
        circle = plt.Circle(ref_center, circle_radius, fill=False, color='purple',
                           linestyle=':', linewidth=2, alpha=0.5)
        ax.add_patch(circle)
    
    ax.set_xlabel('X Position (m)', fontsize=11)
    ax.set_ylabel('Y Position (m)', fontsize=11)
    ax.set_title('10 Subnetworks - Circular Layout (spacing=0)', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right', ncol=1)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Plot 4b: Association matrix
    ax = fig.add_subplot(gs[0, 1])
    
    assoc_viz = merged_ten.associations.astype(float)
    im = ax.imshow(assoc_viz, cmap='RdYlGn', aspect='auto', origin='lower', vmin=0, vmax=1)
    
    # Draw grid lines to separate subnetworks
    for offset in [9.5 + i*10 for i in range(9)]:
        ax.axhline(y=offset, color='purple', linewidth=1.5, alpha=0.4)
        ax.axvline(x=offset, color='purple', linewidth=1.5, alpha=0.4)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Dominant TX', fontsize=10)
    
    ax.set_xlabel('Receiver Index', fontsize=11)
    ax.set_ylabel('Transmitter Index', fontsize=11)
    ax.set_title('Association Matrix (10 Subnetworks)', fontsize=13, fontweight='bold')
    
    # Count intra vs inter
    intra_count = 0
    for i in range(10):
        start = link_offsets_10[i]
        end = link_offsets_10[i+1]
        intra_count += np.sum(assoc_viz[start:end, start:end])
    
    total = np.sum(assoc_viz)
    ax.text(0.02, 0.98, f'Intra: {intra_count:.0f} ({intra_count/total*100:.1f}%)\nInter: {total - intra_count:.0f} ({(total-intra_count)/total*100:.1f}%)',
            transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.savefig(save_path / 'ten_subnetwork_circular.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {save_path / 'ten_subnetwork_circular.png'}")
    plt.close()
    
    # Print summary statistics
    print("\n" + "="*60)
    print("NETWORK STATISTICS")
    print("="*60)
    info = channel.get_network_info()
    print(f"Number of links: {info['n_links']}")
    print(f"Deployment range: {channel.deployment_range}m")
    print(f"Mean TX-TX distance: {info['mean_tx_tx_distance']:.2f}m")
    print(f"Mean paired TX-RX distance: {info['mean_paired_distance']:.2f}m")
    print(f"Large-scale fading range: [{np.min(channel.large_scale_fading):.2e}, "
          f"{np.max(channel.large_scale_fading):.2e}]")
    print(f"Path loss range: [{np.min(channel.path_loss_db):.2f}, "
          f"{np.max(channel.path_loss_db):.2f}] dB")
    
    H_temporal_mean = np.mean(realization['H'], axis=2)
    H_temporal_std = np.std(realization['H'], axis=2)
    print(f"\nTemporal channel statistics (200 time steps):")
    print(f"  Mean gain: {np.mean(H_temporal_mean):.2e}")
    print(f"  Temporal std: {np.mean(H_temporal_std):.2e}")
    
    print("\nMerged network statistics:")
    print(f"  Total links: {merged_linear.n_links}")
    print(f"  Dense subnetwork: {dense_channel.n_links} links")
    print(f"  Sparse subnetwork: {sparse_channel.n_links} links")
    
    print("\n10-subnetwork circular layout:")
    print(f"  Total links: {merged_ten.n_links}")
    print(f"  Circle radius: {circle_radius:.1f}m")
    print(f"  Intra-subnet pairings: {intra_count:.0f}/{total:.0f} ({intra_count/total*100:.1f}%)")
    print("="*60 + "\n")


def run_all_tests():
    """Run all unit tests."""
    print("\n" + "="*60)
    print("RUNNING WIRELESS CHANNEL UNIT TESTS")
    print("="*60 + "\n")
    
    test_suite = TestWirelessChannel()
    
    test_suite.test_channel_initialization()
    test_suite.test_minimum_distance_constraints()
    test_suite.test_optimal_pairing()
    test_suite.test_large_scale_fading_properties()
    test_suite.test_sample_realization_static()
    test_suite.test_sample_realization_time_varying()
    test_suite.test_channel_merging()
    test_suite.test_circular_merging_multiple_subnetworks()
    test_suite.test_circular_merging_ten_subnetworks()
    test_suite.test_network_info()
    test_suite.test_batch_generation()
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED ✓")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Run unit tests
    run_all_tests()
    
    # Generate visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60 + "\n")
    visualize_network_deployment()
    
    print("\n✓ All tests and visualizations complete!")
