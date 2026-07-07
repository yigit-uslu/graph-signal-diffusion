"""
Test script to verify SP500 standardization/de-standardization pipeline.

This script tests:
1. SP500Builder computes per_stock_stats correctly
2. SP500Stocks dataset applies standardization
3. StockPriceForecastingTaskV2 de-standardizes correctly
4. End-to-end pipeline produces raw log returns (std ~0.02)
"""

import torch
import numpy as np
import sys
from pathlib import Path
from omegaconf import OmegaConf

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from graph_signal_diffusion.datasets.sp500.datamodule import SP500Builder
from graph_signal_diffusion.tasks.stock_price_forecasting.evaluator import StockPriceForecastingTaskV2

print("=" * 80)
print("SP500 Standardization Pipeline Test")
print("=" * 80)

# Test 1: Load SP500 dataset and check per_stock_stats
print("\n[Test 1] Loading SP500 dataset and checking per_stock_stats...")

# Create config manually
cfg = OmegaConf.create({
    "name": "sp500",
    "root": str(project_root / "data" / "sp500"),
    "past_window": 20,
    "future_window": 5,
    "target_column_name": "DailyLogReturn",
    "corr_threshold": None,
    "pool_ratio": 0.5,
    "dataset_split_strategy": "chronological",
    "train_dataset_fraction": 0.8,
    "normalize": False
})

print(f"Using config: {OmegaConf.to_yaml(cfg)}")
print("Building SP500 dataset...")

try:
    builder = SP500Builder()
    datasets = builder.build_datasets(cfg)
    
    print(f"✅ Built datasets: {list(datasets.keys())}")
    
    # Check per_stock_stats
    if hasattr(builder, 'per_stock_stats'):
        print(f"✅ Found per_stock_stats for {len(builder.per_stock_stats)} features")
        
        # Verify DailyLogReturn stats
        if 'DailyLogReturn' in builder.per_stock_stats:
            dlr_stats = builder.per_stock_stats['DailyLogReturn']
            dlr_mean = dlr_stats['mean']
            dlr_std = dlr_stats['std']
            
            print(f"\nDailyLogReturn statistics:")
            print(f"  Mean across stocks: {dlr_mean.mean():.6f} (expected ~0.0)")
            print(f"  Std across stocks: {dlr_std.mean():.6f} (expected ~0.02)")
            print(f"  Min std: {dlr_std.min():.6f}")
            print(f"  Max std: {dlr_std.max():.6f}")
            
            # Verify std is in raw scale (not percentage scale)
            if dlr_std.mean() < 0.1:
                print("  ✅ Std is in raw scale (~0.02), as expected")
            else:
                print(f"  ⚠️  Std is too large ({dlr_std.mean():.6f}), expected ~0.02")
        else:
            print("  ❌ DailyLogReturn not found in per_stock_stats")
    else:
        print("❌ per_stock_stats not found in builder")
    
    # Test 2: Load a sample and verify standardization
    print("\n[Test 2] Loading sample and verifying standardization...")
    
    train_dataset = datasets['train']
    sample = train_dataset[0]
    
    print(f"Sample keys: {sample.keys}")
    print(f"  y shape: {sample.y.shape}")  # [N, T, 1]
    print(f"  x shape: {sample.x.shape}")  # [N, T_hist, F]
    
    # Check if y is standardized (should have std ~1.0 after standardization)
    y_std = sample.y.std().item()
    print(f"\nTarget y statistics:")
    print(f"  Mean: {sample.y.mean().item():.6f}")
    print(f"  Std: {y_std:.6f}")
    
    if 0.5 < y_std < 2.0:
        print(f"  ✅ Std is in standardized range (~1.0), as expected")
    else:
        print(f"  ⚠️  Std is outside standardized range: {y_std:.6f}")
    
    # Test 3: Create TaskV2 and verify de-standardization
    print("\n[Test 3] Testing StockPriceForecastingTaskV2 de-standardization...")
    
    task = StockPriceForecastingTaskV2(
        forecast_horizon=5,
        n_samples_per_input=1
    )
    
    # Inject dataset_info
    dataset_info = builder.get_dataset_info()
    task.set_dataset_info(dataset_info)
    
    print(f"✅ Dataset info injected into task")
    
    # Create a mock batch
    B, N, T, F = 2, sample.y.shape[0], sample.y.shape[1], 1
    
    # Simulate standardized targets (std ~1.0)
    mock_standardized = torch.randn(B, T, N, F) * 1.0  # Standardized scale
    
    print(f"\nMock standardized targets:")
    print(f"  Shape: {mock_standardized.shape}")
    print(f"  Mean: {mock_standardized.mean().item():.6f}")
    print(f"  Std: {mock_standardized.std().item():.6f}")
    
    # De-standardize using task method
    destandardized = task._destandardize_samples(
        standardized_samples=mock_standardized,
        per_stock_stats=dataset_info['per_stock_stats'],
        stock_symbols=dataset_info['stock_symbols'],
        feature_idx=0,  # DailyLogReturn
        stock_dim=-2  # [B, T, N, F]
    )
    
    print(f"\nDe-standardized targets:")
    print(f"  Mean: {destandardized.mean().item():.6f}")
    print(f"  Std: {destandardized.std().item():.6f}")
    
    # Verify std is back to raw scale (~0.02)
    destd_std = destandardized.std().item()
    if 0.01 < destd_std < 0.05:
        print(f"  ✅ De-standardization successful (std ~0.02)")
    else:
        print(f"  ⚠️  De-standardized std outside expected range: {destd_std:.6f}")
    
    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)

except Exception as e:
    print(f"\n❌ Test failed with error:")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
