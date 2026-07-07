# WirelessChannelV3 Test Summary

This document summarizes the test results for WirelessChannelV3 with recursive subdivision.

## Test Files

1. **test_channel_v3.py** - Visual comparison of V1, V2, and V3 with network graphs
2. **test_channel_v3_correctness.py** - Associations, pairing, seed reproducibility, and shadowing tests
3. **test_v2_v3_equivalence.py** - Interface compatibility test between V2 and V3

## Test Results

### ✅ Test 1: Associations and Pairing Correctness

**Status: PASSED**

- ✓ Each TX paired with exactly 1 RX
- ✓ Each RX paired with exactly 1 TX  
- ✓ All pairs satisfy max_tx_rx_distance constraint
- ✓ Large-scale fading computed correctly (no NaN/Inf values)
- ✓ Associations matrix matches tx_rx_pairs

### ✅ Test 2: Seed Reproducibility

**Status: PASSED**

- ✓ TX locations identical across reinitializations
- ✓ RX locations identical across reinitializations
- ✓ Associations identical across reinitializations
- ✓ TX-RX pairs identical across reinitializations
- ✓ Large-scale fading identical across reinitializations
- ✓ Subdivision behavior identical across reinitializations

**Conclusion**: Randomness is fully controlled by the seed parameter.

### ✅ Test 3: Cross-Subnetwork Shadowing

**Status: FIXED**

**Before Fix**:
- ⚠️ 95.9% of off-diagonal shadowing values were zero
- Block-diagonal structure indicated subdivision artifacts
- Cross-subnetwork interference not properly modeled

**After Fix**:
- ✓ 0% of off-diagonal shadowing values are zero
- ✓ Shadowing fully populated across all TX-RX pairs
- ✓ Cross-subnetwork interference properly captured

**Fix Applied**: Clear block-diagonal `shadowing_db_deployment` after subdivision and regenerate for full combined network.

### ✅ Test 4: V2-V3 Interface Equivalence

**Status: PASSED**

All attributes match:
- ✓ Basic configuration (n_links, deployment_range, etc.)
- ✓ Array shapes (tx_locations, rx_locations, associations, etc.)
- ✓ Data types (float64, bool, int64)
- ✓ Data validity (no NaN/Inf)
- ✓ Associations properties (one-to-one pairing)
- ✓ Shadowing matrix structure (fully populated)
- ✓ Method availability (sample_realization, get_network_info, etc.)
- ✓ Method functionality (sample_realization returns correct data)

**Conclusion**: After deployment, V3 is indistinguishable from V2. Downstream code cannot tell which version was used.

### ✅ Test 5: Visual Network Comparison

**Status: PASSED**

Visualizations created for:
- **Small Network (20 links)**: All versions succeed
  - V1: 5/20 constraint violations (red lines)
  - V2: 0/20 violations (surgical redeployment)
  - V3: 0/20 violations (subdivision + surgical redeployment)

- **Challenging Network (80 links, tight constraints)**:
  - V1: Failed (cannot deploy with 15m TX-TX spacing)
  - V2: Failed (cannot deploy with 15m TX-TX spacing)
  - V3: **Success** (2-level subdivision into 16 subnetworks)

## Key Features Verified

1. **Recursive Subdivision**: Successfully handles networks that V1/V2 cannot deploy
2. **Distance Constraint Enforcement**: Inherits V2's surgical redeployment
3. **Seed Control**: Full reproducibility across reinitializations
4. **Cross-Interference**: All TX-RX pairs have proper shadowing values
5. **Interface Compatibility**: Drop-in replacement for V2
6. **Transparency**: V3-specific attributes don't interfere with standard usage

## Usage Recommendation

Use WirelessChannelV3 when:
- Deploying large networks (>50 links)
- Working with tight spatial constraints (small deployment area, large min TX-TX distance)
- Standard deployment repeatedly fails
- You need guaranteed successful deployment

WirelessChannelV3 will automatically:
1. Try standard deployment first
2. Fall back to recursive subdivision if needed
3. Regenerate full shadowing matrix for all interference terms
4. Produce identical interface to V2

## Files Generated

- `tests/figs/channel_v3_test/` - Network visualizations
  - `_20links_seed42.png` - V1 visualization
  - `v2_20links_seed42.png` - V2 visualization
  - `v3_20links_seed42.png` - V3 visualization (small network)
  - `v3_80links_seed123.png` - V3 visualization (challenging network)

## Conclusion

WirelessChannelV3 is production-ready:
- ✅ Correct associations and pairing
- ✅ Fully reproducible with seed control
- ✅ Proper cross-interference modeling
- ✅ Full V2 compatibility
- ✅ Handles challenging deployments that V1/V2 cannot
