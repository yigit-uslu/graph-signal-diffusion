# WirelessChannelV3 RNG State Analysis

## Question
When V3 regenerates shadowing, does `skip_deployment=True` and `skip_deployment=False` lead to exactly the same shadowing realizations?

## Answer: NO - They produce DIFFERENT results

Even with the same seed and identical initial TX/RX locations, the two paths produce **different shadowing values and RX positions** due to different RNG consumption patterns.

## Root Causes

### 1. Shadowing Generation (Primary Factor)
- **skip_deployment=False**: RNG sequence is `seed → deployment → shadowing`
  - Deployment consumes many random numbers (TX positions, RX positions, deployment attempts)
  - Shadowing is generated from RNG state after deployment
  
- **skip_deployment=True**: RNG sequence is `seed → shadowing`
  - No deployment consumption
  - Shadowing is generated from earlier RNG state
  
- **Result**: Different RNG states → completely different shadowing values (log-normal distribution)

### 2. Surgical Redeployment (Secondary Factor)
- `_assign_optimal_pairing()` modifies RX locations when constraints are violated
- Uses RNG to generate random angle and radius for redeployment
- Different RNG states → different redeployment positions
- **Result**: RX locations differ between the two paths

## Test Results (8 links, seed=42)

| Attribute | Normal Deployment | Skip Deployment | Identical? |
|-----------|-------------------|-----------------|------------|
| TX locations | - | Copied from normal | ✓ Yes |
| RX locations | After redeployment | After redeployment | ✗ No (2/8 differ by up to 38.9m) |
| Path loss | Distance-based | Distance-based | ✗ No (max diff: 10.2 dB) |
| Shadowing | Random | Random | ✗ No (max diff: 24.4 dB, mean: 8.6 dB) |
| Large-scale fading | PL + shadowing | PL + shadowing | ✗ No (max diff: 21.9 dB) |
| Associations | Optimal pairing | Optimal pairing | ✗ No (different pairings) |

### Example Shadowing Values (3×3 submatrix)

**Normal deployment:**
```
[[  3.75  -3.01  -1.16]
 [  6.49   1.40   1.88]
 [-10.95   1.14  -8.22]]
```

**Skip deployment:**
```
[[  2.53  10.77  -0.25]
 [  0.64 -13.91  -1.54]
 [  6.41   2.30  -3.71]]
```

Completely different values!

## Implications for merge_channels

### Expected Behavior
`merge_channels` creates networks with **DIFFERENT shadowing** than deploying the same total number of links at once:

1. Each subnetwork is deployed with its own seed
2. Subnetwork locations are combined with spatial offsets
3. Merged channel is created with `skip_deployment=True`
4. Shadowing is regenerated for the full combined network using **current RNG state**

**Result**: The merged network's shadowing is:
- NOT reproducible from subnetwork seeds
- NOT the same as deploying all links at once
- A fresh random realization based on current RNG state

### How to Get Reproducible Merged Networks

**Option 1: Set global seed before merging**
```python
np.random.seed(42)
merged = merge_channels([channel1, channel2], ...)
```

**Option 2: Save/load merged channel state**
```python
# After merging
np.save('merged_tx.npy', merged.tx_locations)
np.save('merged_rx.npy', merged.rx_locations)
np.save('merged_shadowing.npy', merged.shadowing_db)

# Later, recreate
merged = WirelessChannel(skip_deployment=True, ...)
merged.tx_locations = np.load('merged_tx.npy')
merged.rx_locations = np.load('merged_rx.npy')
merged.shadowing_db_deployment = np.load('merged_shadowing.npy')
merged._compute_large_scale_fading()
merged._assign_optimal_pairing()
```

## Conclusion

✅ This behavior is **EXPECTED and CORRECT**

The different RNG states are intentional:
- `skip_deployment=True` is designed for merge_channels where you want fresh randomness
- Each merged subnetwork contributes its unique deployment pattern
- The merged network gets new shadowing independent of subnetwork seeds
- This creates realistic heterogeneous network diversity

❌ If you want identical shadowing:
- Don't use `skip_deployment=True` with the same seed
- Deploy once with the desired seed instead of merging

## Test Files
- `tests/datasets/test_channel_v3_rng_state.py`: Comprehensive RNG state analysis
- `tests/datasets/test_channel_v3_preplaced.py`: Skip deployment functionality tests
