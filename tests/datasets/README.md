# WRA Channel Tests

This directory contains unit tests and visualizations for the wireless resource allocation (WRA) channel generation module.

## Test File

- `test_wra_channel.py`: Comprehensive unit tests for the `WirelessChannel` class

## Running Tests

```bash
# From project root
PYTHONPATH=src:$PYTHONPATH python tests/datasets/test_wra_channel.py
```

## Test Coverage

### Unit Tests
1. **Channel Initialization**: Validates network topology and dimensions
2. **Minimum Distance Constraints**: Ensures TX-TX and TX-RX separation constraints
3. **Optimal TX-RX Pairing**: Verifies each TX paired with best RX by channel gain
4. **Large-Scale Fading Properties**: Validates physical channel properties
5. **Static Channel Sampling**: Tests channel realizations without small-scale fading
6. **Time-Varying Channel Sampling**: Tests Rayleigh fading over time
7. **Channel Merging**: Validates subnetwork merging and pairing recomputation
8. **Network Info Retrieval**: Tests metadata extraction
9. **Reproducibility**: Ensures deterministic results with fixed seed

### Visualizations
Generated in `tests/figs/wra_channel/`:

1. **network_deployment.png**
   - Spatial layout of transmitters and receivers
   - TX-RX pairing visualization
   - Large-scale channel gain heatmap

2. **channel_time_evolution.png**
   - Time series of channel gains
   - Distribution of channel gains
   - Temporal autocorrelation
   - Channel matrix snapshot

3. **merged_network.png**
   - Subnetworks before merging
   - Merged network with recomputed pairing

## Key Test Results

✅ All 9 unit tests passed
✅ Network deployment satisfies distance constraints
✅ TX-RX pairing is optimal based on channel gain
✅ Time-varying channels show proper Rayleigh fading characteristics
✅ Channel merging correctly recomputes associations
✅ Results are reproducible with fixed seed

## Network Statistics (Example)

From a 15-link network:
- Deployment range: 500m × 500m
- Mean TX-TX distance: 273.44m
- Mean paired TX-RX distance: 46.47m
- Path loss range: [59.41, 109.96] dB
- Large-scale fading ensures realistic wireless propagation
