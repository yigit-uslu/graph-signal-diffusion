#!/usr/bin/env python3
"""
Simple test to verify SP500 target destandardization.

Directly loads timestep_0.pt and verifies that destandardized DailyLogReturn
matches the actual closing price ratios.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_destandardization_simple():
    """Test destandardization by directly loading processed data."""
    print("=" * 80)
    print("Testing SP500 Target Destandardization (Simple)")
    print("=" * 80)
    
    root_dir = Path('data/sp500/cleaned_drop_incomplete_min_coverage_0.95_corr_0.55_sector_bonus_0.1')
    processed_dir = root_dir / 'processed'
    
    # Load static graph data
    print("\n[1] Loading static graph data...")
    static_data = torch.load(processed_dir / 'graph_static.pt', weights_only=True)
    info = static_data['info']
    
    print(f"    Info keys: {list(info.keys())}")
    print(f"    Features: {info['Features']}")
    print(f"    Target: {info['Target']}")
    print(f"    Number of nodes: {info['Num_nodes']}")
    
    # Load values.csv to compute per-stock stats manually
    print("\n[2] Loading values.csv to compute per-stock stats...")
    values = pd.read_csv(root_dir / 'raw' / 'values.csv').set_index(['Symbol', 'Date'])
    
    # Get feature names (after dropping Close)
    feature_names = values.drop(columns=['Close']).columns.tolist()
    print(f"    Feature names: {feature_names}")
    
    # Compute per-stock stats for DailyLogReturn
    target_col = info['Target']
    scale_factor = 100.0
    
    symbols = values.index.get_level_values('Symbol').unique()
    print(f"    Number of stocks: {len(symbols)}")
    
    # Compute mean and std for each stock
    means = []
    stds = []
    for symbol in symbols:
        stock_data = values.loc[symbol, target_col].values
        # Divide by 100 first (as per our refactoring)
        stock_data_scaled = stock_data / scale_factor
        means.append(stock_data_scaled.mean())
        stds.append(stock_data_scaled.std())
    
    means = torch.tensor(means).float().view(-1, 1, 1)  # [N, 1, 1]
    stds = torch.tensor(stds).float().view(-1, 1, 1)    # [N, 1, 1]
    
    print(f"    Per-stock stats computed")
    print(f"    Mean range: [{means.min():.6f}, {means.max():.6f}]")
    print(f"    Std range: [{stds.min():.6f}, {stds.max():.6f}]")
    
    # Load timestep_0.pt
    print("\n[3] Loading timestep_0.pt...")
    timestep_data = torch.load(processed_dir / 'timestep_0.pt', weights_only=True)
    
    print(f"    Keys: {list(timestep_data.keys())}")
    print(f"    x shape: {timestep_data['x'].shape}")  # [N, F, T_past]
    print(f"    y shape: {timestep_data['y'].shape}")  # [N, T_future]
    print(f"    close_price shape: {timestep_data['close_price'].shape}")  # [N, T_past]
    print(f"    close_price_y shape: {timestep_data['close_price_y'].shape}")  # [N, T_future]
    
    # Extract data
    y = timestep_data['y']  # [N, T_future] - this is the RAW target (unstandardized in new refactoring)
    close_past = timestep_data['close_price']  # [N, T_past]
    close_future = timestep_data['close_price_y']  # [N, T_future]
    
    print(f"\n    Target (y) statistics (should be RAW in new refactoring):")
    print(f"      Mean: {y.mean():.4f}")
    print(f"      Std:  {y.std():.4f}")
    print(f"      Min:  {y.min():.4f}")
    print(f"      Max:  {y.max():.4f}")
    
    # Check if y looks standardized or raw
    if abs(y.mean()) < 0.2 and abs(y.std() - 1.0) < 0.5:
        print(f"    ⚠️  WARNING: y appears STANDARDIZED (old processing)")
        y_is_standardized = True
    else:
        print(f"    ✓ y appears RAW/unstandardized (new processing)")
        y_is_standardized = False
    
    # Compute actual log returns from closing prices
    print("\n[4] Computing log returns from closing prices...")
    
    # Get the last closing price from past window
    close_last_past = close_past[:, -1:]  # [N, 1]
    
    # Concatenate to get full sequence: [last_past, future_0, future_1, ..., future_T-1]
    close_sequence = torch.cat([close_last_past, close_future], dim=1)  # [N, T_future+1]
    
    # Compute log returns: log(Close_t / Close_{t-1}) * 100
    price_ratios = close_sequence[:, 1:] / close_sequence[:, :-1]  # [N, T_future]
    actual_log_returns = torch.log(price_ratios) * scale_factor  # [N, T_future]
    
    print(f"    Actual log returns from prices:")
    print(f"      Mean: {actual_log_returns.mean():.4f}")
    print(f"      Std:  {actual_log_returns.std():.4f}")
    print(f"      Min:  {actual_log_returns.min():.4f}")
    print(f"      Max:  {actual_log_returns.max():.4f}")
    
    # If y is standardized (old processing), destandardize it
    if y_is_standardized:
        print("\n[5] Destandardizing y (old processing detected)...")
        # Destandardize: x = z * std + mean, then multiply by scale_factor
        y_destandardized = y.unsqueeze(-1) * (stds + 1e-8) + means
        y_destandardized = y_destandardized.squeeze(-1) * scale_factor
        
        print(f"    Destandardized y statistics:")
        print(f"      Mean: {y_destandardized.mean():.4f}")
        print(f"      Std:  {y_destandardized.std():.4f}")
    else:
        print("\n[5] y is already raw/unstandardized (new processing)")
        y_destandardized = y
    
    # Compare with actual log returns
    print("\n[6] Comparing y with actual log returns from close_price_y...")
    
    difference = torch.abs(y_destandardized - actual_log_returns)
    
    print(f"    Absolute difference statistics:")
    print(f"      Mean: {difference.mean():.6f}")
    print(f"      Std:  {difference.std():.6f}")
    print(f"      Max:  {difference.max():.6f}")
    print(f"      Median: {difference.median():.6f}")
    
    # Check a few specific stocks in detail
    print("\n[7] Detailed comparison for first 5 stocks, first future timestep:")
    print(f"    {'Stock':<6} {'Y Value':<12} {'LogRet from Price':<18} {'Diff':<12} {'Price Ratio':<12}")
    print("    " + "-" * 75)
    
    for i in range(min(5, y_destandardized.shape[0])):
        y_val = y_destandardized[i, 0].item()
        actual = actual_log_returns[i, 0].item()
        diff = difference[i, 0].item()
        ratio = price_ratios[i, 0].item()
        
        print(f"    {i:<6} {y_val:<12.4f} {actual:<18.4f} {diff:<12.6f} {ratio:<12.6f}")
    
    # Verify they match within tolerance
    print("\n[8] Verification:")
    
    max_diff = difference.max().item()
    mean_diff = difference.mean().item()
    
    # The difference should be very small (numerical precision errors)
    tolerance = 1e-4
    
    if max_diff < tolerance:
        print(f"    ✓ PASS: Max difference {max_diff:.6f} < {tolerance}")
        print(f"    ✓ Target y correctly matches log returns from close_price_y!")
    else:
        print(f"    ✗ FAIL: Max difference {max_diff:.6f} >= {tolerance}")
        print(f"    ✗ Target y does not match close_price_y!")
        
        # Show worst mismatches
        print("\n    Worst 5 mismatches:")
        flat_diff = difference.flatten()
        flat_indices = torch.argsort(flat_diff, descending=True)[:5]
        
        for rank, flat_idx in enumerate(flat_indices):
            stock_idx = flat_idx // y_destandardized.shape[1]
            time_idx = flat_idx % y_destandardized.shape[1]
            
            y_val = y_destandardized[stock_idx, time_idx].item()
            actual = actual_log_returns[stock_idx, time_idx].item()
            diff = flat_diff[flat_idx].item()
            
            print(f"      {rank+1}. Stock {stock_idx}, Time {time_idx}: "
                  f"Y={y_val:.4f}, LogRet={actual:.4f}, Diff={diff:.6f}")
    
    print(f"\n    Mean absolute difference: {mean_diff:.6f}")
    
    # Verify close_prices are NOT standardized
    print("\n[9] Verifying close_prices are NOT standardized...")
    print(f"    close_price range: [{close_past.min():.2f}, {close_past.max():.2f}]")
    print(f"    close_price_y range: [{close_future.min():.2f}, {close_future.max():.2f}]")
    
    # Typical stock prices should be in range [~1, ~10000]
    if close_past.min() > 0 and close_past.max() < 100000:
        print(f"    ✓ close_prices are in original scale (NOT standardized)")
    else:
        print(f"    ⚠  close_prices may contain unusual values")
    
    # NEW: Test full round-trip via get() method
    print("\n[10] Testing full round-trip: get() → standardize → destandardize...")
    from graph_signal_diffusion.datasets.sp500.dataset import SP500Stocks
    from graph_signal_diffusion.datasets.sp500.datamodule import SP500Builder
    
    # Initialize builder with necessary attributes
    builder = SP500Builder()
    builder.feature_names = feature_names
    builder.standardized_features = [f for f in feature_names if f not in {'NormClose', 'RSI'}]
    builder.log_transform_features = ['Volume']
    builder.sp500_scale_factor = scale_factor
    
    # Compute stats
    values_df = pd.read_csv(root_dir / 'raw' / 'values.csv').set_index(['Symbol', 'Date'])
    per_stock_stats = builder._compute_per_stock_stats(values_df)
    
    # Create dataset with standardization enabled
    dataset = SP500Stocks(
        root=str(root_dir),
        values_file_name="values.csv",
        adj_file_name="adj.npy",
        past_window=25,
        future_window=5,
        target_column_name='DailyLogReturn',
        pool_ratio=0.5,
        per_stock_stats=per_stock_stats,
        feature_names=feature_names,
        standardized_features=[f for f in feature_names if f not in {'NormClose', 'RSI'}],
        log_transform_features=['Volume'],
        sp500_scale_factor=100.0,
    )
    
    print(f"    Dataset initialized with standardization enabled")
    
    # Get target stats using new API
    target_stats = dataset.get_target_standardization_stats()
    print(f"    Target stats: {target_stats['target_name']}")
    print(f"      mean shape: {target_stats['mean'].shape}")
    print(f"      std shape: {target_stats['std'].shape}")
    print(f"      scale_factor: {target_stats['scale_factor']}")
    
    # Get sample via get() method (applies standardization)
    sample = dataset.get(0)
    y_standardized = sample.y  # [N, T_future, 1]
    
    print(f"    y from get() (standardized):")
    print(f"      Shape: {y_standardized.shape}")
    print(f"      Mean: {y_standardized.mean():.4f}")
    print(f"      Std:  {y_standardized.std():.4f}")
    
    # Destandardize using new static method
    y_destandardized_roundtrip = SP500Stocks.destandardize_target(y_standardized, target_stats)
    
    print(f"    y after destandardization:")
    print(f"      Shape: {y_destandardized_roundtrip.shape}")
    print(f"      Mean: {y_destandardized_roundtrip.mean():.4f}")
    print(f"      Std:  {y_destandardized_roundtrip.std():.4f}")
    
    # Compare with original raw y (need to unsqueeze for comparison)
    y_raw_comparison = y_destandardized.unsqueeze(-1)  # [N, T] -> [N, T, 1]
    roundtrip_difference = torch.abs(y_destandardized_roundtrip - y_raw_comparison)
    
    print(f"    Round-trip error (destandardized vs original raw y):")
    print(f"      Mean: {roundtrip_difference.mean():.6f}")
    print(f"      Max:  {roundtrip_difference.max():.6f}")
    
    if roundtrip_difference.max() < tolerance:
        print(f"    ✓ PASS: Round-trip successful!")
        print(f"    ✓ get() → destandardize correctly recovers original y values")
        roundtrip_success = True
    else:
        print(f"    ✗ FAIL: Round-trip error too large")
        roundtrip_success = False
    
    print("\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)
    
    return max_diff < tolerance and roundtrip_success


if __name__ == "__main__":
    try:
        success = test_destandardization_simple()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
