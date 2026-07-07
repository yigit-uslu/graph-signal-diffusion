# Channel V3 Test Visualizations

This directory contains network visualizations from testing WirelessChannelV3 with recursive subdivision.

## Visualization Legend

- **Red Triangles (▲)**: Transmitters (TX)
- **Blue Circles (●)**: Receivers (RX)
- **Green Lines**: Valid TX-RX pairings (within max_tx_rx_distance)
- **Red Lines**: Constraint violations (exceeds max_tx_rx_distance)

## Files

### Small Network Test (20 links, relaxed constraints)

- **`_20links_seed42.png`**: WirelessChannel V1 (Simple Fallback)
  - Shows baseline pairing without distance constraint enforcement
  - May have red lines indicating constraint violations
  
- **`v2_20links_seed42.png`**: WirelessChannelV2 (Surgical Redeployment)
  - Demonstrates surgical redeployment of problematic RXs
  - All lines should be green (no violations)
  
- **`v3_20links_seed42.png`**: WirelessChannelV3 (Recursive Subdivision)
  - Shows network deployed via 1-level subdivision into 4 subnetworks
  - All lines should be green (no violations)

### Challenging Network Test (80 links, tight constraints)

- **`v3_80links_seed123.png`**: WirelessChannelV3 (Large Network)
  - V1 and V2 failed to deploy (15m min TX-TX distance, 200m area)
  - V3 succeeded via 2-level recursive subdivision (16 subnetworks)
  - Demonstrates V3's ability to handle difficult deployment scenarios
  - All lines should be green (40m max distance enforced)

## Statistics Box

Each visualization includes a statistics box showing:
- **Links**: Number of TX-RX pairs
- **Mean distance**: Average distance between paired TX-RX
- **Max distance**: Maximum distance among all pairs
- **Violations**: Number of pairs exceeding max_tx_rx_distance constraint

## Running the Test

```bash
cd /path/to/graph-signal-diffusion
conda activate torch_env
python tests/datasets/test_channel_v3.py
```

The test will regenerate all visualizations and display comprehensive deployment statistics.
