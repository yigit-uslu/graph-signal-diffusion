#!/usr/bin/env python3
"""
Test script to verify SP500 target destandardization.

Verifies that destandardized DailyLogReturn matches the actual
closing price ratios: DailyLogReturn = log(Close_t / Close_{t-1}) * 100
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omegaconf import DictConfig
from graph_signal_diffusion.datasets.sp500.datamodule import SP500Builder


def test_destandardization():
    """Test that destandardization correctly inverts standardization."""
    print("=" * 80)
    print("Testing SP500 Target Destandardization")
    print("=" * 80)
    
    # Initialize builder
    print("\n[1] Initializing SP500Builder...")
    builder = SP500Builder()
    
    # Create config for cleaned dataset
    cfg = DictConfig({
        'root': 'data/sp500/cleaned_drop_incomplete_min_coverage_0.95_corr_0.55_sector_bonus_0.1',
        'past_window': 25,
        'future_window': 5,  # Use 5-day future window for better testing
        'target_column_name': 'DailyLogReturn',
        'pool_ratio': 0.5,
    })
    
    # Build dataset
    print("\n[2] Building dataset...")
    datasets = builder.build_datasets(cfg)
    dataset = datasets['full']
    
    print(f"    Dataset length: {len(dataset)}")
    print(f"    Feature names: {dataset.feature_names}")
    print(f"    Target column: {dataset.target_column_name}")
    print(f"    Scale factor: {dataset.sp500_scale_factor}")
    
    # Get a sample from the middle of the dataset
    print("\n[3] Loading sample data...")
    idx = len(dataset) // 2
    sample = dataset.get(idx)
    
    print(f"    Sample index: {idx}")
    print(f"    x shape: {sample.x.shape}")  # [N, T_past, F]
    print(f"    y shape (standardized target): {sample.y.shape}")  # [N, T_future, 1]
    print(f"    close_price shape: {sample.close_price.shape}")  # [N, T_past, 1]
    print(f"    close_price_y shape: {sample.close_price_y.shape}")  # [N, T_future, 1]
    print(f"    Number of stocks: {sample.x.shape[0]}")
    
    # Extract data
    y_standardized = sample.y.squeeze(-1)  # [N, T_future]
    close_past = sample.close_price.squeeze(-1)  # [N, T_past]
    close_future = sample.close_price_y.squeeze(-1)  # [N, T_future]
    
    print(f"\n    Standardized target (y) statistics:")
    print(f"      Mean: {y_standardized.mean():.4f}")
    print(f"      Std:  {y_standardized.std():.4f}")
    print(f"      Min:  {y_standardized.min():.4f}")
    print(f"      Max:  {y_standardized.max():.4f}")
    
    # Get per-stock stats for destandardization
    print("\n[4] Destandardizing target...")
    target_feat_idx = dataset.feature_names.index(dataset.target_column_name)
    stats = dataset.per_stock_stats[dataset.target_column_name]
    means = torch.from_numpy(stats['mean']).float()  # [N,]
    stds = torch.from_numpy(stats['std']).float()    # [N,]
    
    # Reshape for broadcasting
    means = means.view(-1, 1)  # [N, 1]
    stds = stds.view(-1, 1)    # [N, 1]
    
    # Destandardize: x = z * std + mean
    y_destandardized = y_standardized * (stds + 1e-8) + means
    
    # Multiply back by scale factor (since we divided by 100 during standardization)
    y_destandardized = y_destandardized * dataset.sp500_scale_factor
    
    print(f"    Destandardized target statistics:")
    print(f"      Mean: {y_destandardized.mean():.4f}")
    print(f"      Std:  {y_destandardized.std():.4f}")
    print(f"      Min:  {y_destandardized.min():.4f}")
    print(f"      Max:  {y_destandardized.max():.4f}")
    
    # Compute actual log returns from closing prices
    print("\n[5] Computing log returns from closing prices...")
    # For the first future timestep, we compare with last past closing price
    # For subsequent future timesteps, we compare with previous future closing price
    
    # Get the last closing price from past window
    close_last_past = close_past[:, -1:]  # [N, 1]
    
    # Concatenate to get full sequence: [last_past, future_0, future_1, ..., future_T-1]
    close_sequence = torch.cat([close_last_past, close_future], dim=1)  # [N, T_future+1]
    
    # Compute log returns: log(Close_t / Close_{t-1}) * 100
    price_ratios = close_sequence[:, 1:] / close_sequence[:, :-1]  # [N, T_future]
    actual_log_returns = torch.log(price_ratios) * dataset.sp500_scale_factor  # [N, T_future]
    
    print(f"    Actual log returns from prices:")
    print(f"      Mean: {actual_log_returns.mean():.4f}")
    print(f"      Std:  {actual_log_returns.std():.4f}")
    print(f"      Min:  {actual_log_returns.min():.4f}")
    print(f"      Max:  {actual_log_returns.max():.4f}")
    
    # Compare destandardized y with actual log returns
    print("\n[6] Comparing destandardized target with actual log returns...")
    
    difference = torch.abs(y_destandardized - actual_log_returns)
    
    print(f"    Absolute difference statistics:")
    print(f"      Mean: {difference.mean():.6f}")
    print(f"      Std:  {difference.std():.6f}")
    print(f"      Max:  {difference.max():.6f}")
    print(f"      Median: {difference.median():.6f}")
    
    # Check a few specific stocks in detail
    print("\n[7] Detailed comparison for first 5 stocks, first future timestep:")
    print(f"    {'Stock':<6} {'Destd Y':<12} {'Actual LogRet':<15} {'Diff':<12} {'Price Ratio':<12}")
    print("    " + "-" * 70)
    
    for i in range(min(5, y_destandardized.shape[0])):
        destd = y_destandardized[i, 0].item()
        actual = actual_log_returns[i, 0].item()
        diff = difference[i, 0].item()
        ratio = price_ratios[i, 0].item()
        
        print(f"    {i:<6} {destd:<12.4f} {actual:<15.4f} {diff:<12.6f} {ratio:<12.6f}")
    
    # Verify they match within tolerance
    print("\n[8] Verification:")
    
    max_diff = difference.max().item()
    mean_diff = difference.mean().item()
    
    # The difference should be very small (numerical precision errors)
    tolerance = 1e-4
    
    if max_diff < tolerance:
        print(f"    ✓ PASS: Max difference {max_diff:.6f} < {tolerance}")
        print(f"    ✓ Destandardization correctly inverts standardization!")
    else:
        print(f"    ✗ FAIL: Max difference {max_diff:.6f} >= {tolerance}")
        print(f"    ✗ Destandardization may be incorrect!")
        
        # Show worst mismatches
        print("\n    Worst 5 mismatches:")
        flat_diff = difference.flatten()
        flat_indices = torch.argsort(flat_diff, descending=True)[:5]
        
        for rank, flat_idx in enumerate(flat_indices):
            stock_idx = flat_idx // y_destandardized.shape[1]
            time_idx = flat_idx % y_destandardized.shape[1]
            
            destd = y_destandardized[stock_idx, time_idx].item()
            actual = actual_log_returns[stock_idx, time_idx].item()
            diff = flat_diff[flat_idx].item()
            
            print(f"      {rank+1}. Stock {stock_idx}, Time {time_idx}: "
                  f"Destd={destd:.4f}, Actual={actual:.4f}, Diff={diff:.6f}")
    
    print(f"\n    Mean absolute difference: {mean_diff:.6f}")
    
    # Additional check: verify close_prices are NOT standardized
    print("\n[9] Verifying close_prices are NOT standardized...")
    print(f"    close_price range: [{close_past.min():.2f}, {close_past.max():.2f}]")
    print(f"    close_price_y range: [{close_future.min():.2f}, {close_future.max():.2f}]")
    
    # Typical stock prices should be in range [~1, ~10000]
    if close_past.min() > 0 and close_past.max() < 100000:
        print(f"    ✓ close_prices appear to be in original scale (not standardized)")
    else:
        print(f"    ⚠  close_prices may be standardized or contain unusual values")
    
    print("\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)
    
    return max_diff < tolerance


if __name__ == "__main__":
    try:
        success = test_destandardization()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
