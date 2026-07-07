"""
Demonstration of WirelessChannel vs WirelessChannelV2.

This script shows the difference between the original WirelessChannel
and the enhanced WirelessChannelV2 with surgical RX redeployment.
"""

import numpy as np
from graph_signal_diffusion.datasets.wra.channel import WirelessChannel, WirelessChannelV2

print("=" * 80)
print("Comparison: WirelessChannel vs WirelessChannelV2")
print("=" * 80)

# Test parameters designed to trigger max_tx_rx_distance violations
n_links = 20
deployment_range = 1200.0
min_tx_rx_distance = 10.0
max_tx_rx_distance = 50.0
seed = 5

print("\nTest Configuration:")
print(f"  n_links: {n_links}")
print(f"  deployment_range: {deployment_range}m")
print(f"  min_tx_rx_distance: {min_tx_rx_distance}m")
print(f"  max_tx_rx_distance: {max_tx_rx_distance}m")
print(f"  seed: {seed}")

# Test original WirelessChannel
print("\n" + "-" * 80)
print("1. WirelessChannel (original, simple fallback)")
print("-" * 80)
channel_v1 = WirelessChannel(
    n_links=n_links,
    deployment_range=deployment_range,
    min_tx_rx_distance=min_tx_rx_distance,
    max_tx_rx_distance=max_tx_rx_distance,
    seed=seed
)

# Check distance violations in V1
v1_violations = 0
v1_max_distance = 0
for tx_idx, rx_idx in channel_v1.tx_rx_pairs:
    distance = channel_v1.distances[tx_idx, rx_idx]
    if distance > max_tx_rx_distance:
        v1_violations += 1
    v1_max_distance = max(v1_max_distance, distance)

print(f"\nResults:")
print(f"  Max TX-RX distance: {v1_max_distance:.2f}m")
print(f"  Pairs violating max_tx_rx_distance: {v1_violations}/{n_links}")
if v1_violations > 0:
    print(f"  ⚠ Warning: {v1_violations} pairs exceed {max_tx_rx_distance}m constraint!")

# Test enhanced WirelessChannelV2
print("\n" + "-" * 80)
print("2. WirelessChannelV2 (enhanced with surgical RX redeployment)")
print("-" * 80)
channel_v2 = WirelessChannelV2(
    n_links=n_links,
    deployment_range=deployment_range,
    min_tx_rx_distance=min_tx_rx_distance,
    max_tx_rx_distance=max_tx_rx_distance,
    seed=seed
)

# Check distance violations in V2
v2_violations = 0
v2_max_distance = 0
for tx_idx, rx_idx in channel_v2.tx_rx_pairs:
    distance = channel_v2.distances[tx_idx, rx_idx]
    if distance > max_tx_rx_distance:
        v2_violations += 1
    v2_max_distance = max(v2_max_distance, distance)

print(f"\nResults:")
print(f"  Max TX-RX distance: {v2_max_distance:.2f}m")
print(f"  Pairs violating max_tx_rx_distance: {v2_violations}/{n_links}")
if v2_violations == 0:
    print(f"  ✓ All pairs satisfy {max_tx_rx_distance}m constraint!")

# Summary
print("\n" + "=" * 80)
print("Summary")
print("=" * 80)
print(f"\nWirelessChannel (V1):")
print(f"  - Uses simple fallback: assigns closest unassigned RX")
print(f"  - May violate max_tx_rx_distance constraint")
print(f"  - Violations in this test: {v1_violations}/{n_links}")

print(f"\nWirelessChannelV2 (V2):")
print(f"  - Uses surgical RX redeployment in annular region")
print(f"  - Enforces max_tx_rx_distance for all pairs")
print(f"  - Preserves well-paired TX-RX associations")
print(f"  - Violations in this test: {v2_violations}/{n_links}")

print(f"\n{'✓ V2 successfully enforces distance constraints!' if v2_violations == 0 else '⚠ Issue detected'}")
print("=" * 80)
