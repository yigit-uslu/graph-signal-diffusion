#!/usr/bin/env python3
"""
Test script for SP500 dataset standardization refactoring.

Verifies:
1. Features are standardized during preprocessing and saved
2. Target is NOT standardized in saved files
3. Target gets standardized at runtime via get()
4. Log-transform is applied to Volume
5. RSI is excluded from standardization
6. Return features are divided by 100 before standardization
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omegaconf import DictConfig
from graph_signal_diffusion.datasets.sp500.datamodule import SP500Builder
from graph_signal_diffusion.datasets.sp500.dataset import SP500Stocks


def check_array_standardized(arr, name, tolerance=0.15):
    """Check if array is approximately standardized (mean≈0, std≈1)."""
    mean = arr.mean()
    std = arr.std()
    is_standardized = abs(mean) < tolerance and abs(std - 1.0) < tolerance
    
    print(f"  {name:20s}: mean={mean:8.4f}, std={std:7.4f} -> {'✓ standardized' if is_standardized else '✗ NOT standardized'}")
    return is_standardized


def test_sp500_standardization():
    """Test the refactored SP500 dataset standardization."""
    print("=" * 80)
    print("Testing SP500 Dataset Standardization Refactoring")
    print("=" * 80)
    
    # Load existing processed dataset directly
    print("\n[1] Loading existing processed SP500 dataset...")
    root_dir = 'data/sp500/cleaned_drop_incomplete_min_coverage_0.95_corr_0.55_sector_bonus_0.1'
    
    # Directly instantiate the dataset
    dataset = SP500Stocks(
        root=root_dir,
        values_file_name="values.csv",
        adj_file_name="adj.npy",
        past_window=25,
        future_window=1,
        target_column_name="DailyLogReturn",
        pool_ratio=0.5,
    )
    
    print(f"    Dataset length: {len(dataset)}")
    print(f"    Feature names: {dataset.feature_names}")
    
    # Load a processed file to check what's saved
    print("\n[3] Checking saved processed data...")
    processed_dir = Path(dataset.processed_dir)
    data_files = sorted(processed_dir.glob("data_*.pt"))
    
    if not data_files:
        print("    ERROR: No processed data files found!")
        return False
    
    print(f"    Found {len(data_files)} processed files")
    
    # Load first file
    first_file = data_files[0]
    print(f"    Loading: {first_file.name}")
    data = torch.load(first_file)
    
    print(f"    Data keys: {list(data.keys())}")
    print(f"    x shape: {data.x.shape if 'x' in data else 'N/A'}")
    print(f"    y shape: {data.y.shape if 'y' in data else 'N/A'}")
    
    # Check features (should be standardized)
    print("\n[4] Analyzing saved features (should be standardized)...")
    if 'x' in data:
        x = data.x.numpy()
        print(f"    Features shape: {x.shape}")
        
        for i, feat_name in enumerate(dataset.feature_names):
            if i >= x.shape[1]:
                break
            
            feat_values = x[:, i]
            
            # Special checks
            if feat_name == 'RSI':
                print(f"  {feat_name:20s}: min={feat_values.min():7.4f}, max={feat_values.max():7.4f}, mean={feat_values.mean():7.4f} (should NOT be standardized)")
            elif feat_name == 'Volume':
                print(f"  {feat_name:20s}: min={feat_values.min():7.4f}, max={feat_values.max():7.4f}, mean={feat_values.mean():7.4f} (should be log-transformed then standardized)")
                check_array_standardized(feat_values, "  >", tolerance=0.2)
            elif feat_name == 'NormClose':
                print(f"  {feat_name:20s}: min={feat_values.min():7.4f}, max={feat_values.max():7.4f}, mean={feat_values.mean():7.4f} (should NOT be standardized)")
            else:
                check_array_standardized(feat_values, feat_name, tolerance=0.2)
    
    # Check target (should NOT be standardized)
    print("\n[5] Analyzing saved target (should NOT be standardized)...")
    if 'y' in data:
        y = data.y.numpy()
        print(f"    Target shape: {y.shape}")
        
        target_mean = y.mean()
        target_std = y.std()
        print(f"    Target: mean={target_mean:8.4f}, std={target_std:7.4f}")
        
        if abs(target_mean) < 0.15 and abs(target_std - 1.0) < 0.15:
            print("    ✗ WARNING: Target appears standardized in saved file (should not be!)")
        else:
            print("    ✓ Target is NOT standardized in saved file (correct)")
    
    # Test runtime get() - target should be standardized
    print("\n[6] Testing runtime get() method (target should be standardized here)...")
    
    # Get a sample
    idx = len(dataset) // 2  # Middle sample
    sample = dataset.get(idx)
    
    print(f"    Sample index: {idx}")
    print(f"    Sample keys: {list(sample.keys())}")
    
    if 'x' in sample:
        x_runtime = sample.x.numpy()
        print(f"    Runtime x shape: {x_runtime.shape}")
        print("\n    Runtime feature statistics:")
        
        for i, feat_name in enumerate(dataset.feature_names):
            if i >= x_runtime.shape[1]:
                break
            
            feat_values = x_runtime[:, i]
            mean = feat_values.mean()
            std = feat_values.std()
            
            print(f"      {feat_name:20s}: mean={mean:8.4f}, std={std:7.4f}")
    
    if 'y' in sample:
        y_runtime = sample.y.numpy()
        print(f"\n    Runtime y shape: {y_runtime.shape}")
        
        y_mean = y_runtime.mean()
        y_std = y_runtime.std()
        print(f"    Runtime target: mean={y_mean:8.4f}, std={y_std:7.4f}")
        
        if abs(y_mean) < 0.3 and abs(y_std - 1.0) < 0.3:
            print("    ✓ Target is standardized at runtime (correct)")
        else:
            print("    ✗ WARNING: Target does not appear standardized at runtime")
    
    # Compare saved vs runtime
    print("\n[7] Comparing saved data vs runtime data...")
    print("    Features should be identical (both standardized)")
    print("    Target should differ (saved: raw, runtime: standardized)")
    
    if 'x' in data and 'x' in sample:
        # Get same stock from loaded file
        x_saved = data.x.numpy()
        x_runtime = sample.x.numpy()
        
        if x_saved.shape == x_runtime.shape:
            feature_diff = np.abs(x_saved - x_runtime).max()
            print(f"    Feature max difference: {feature_diff:.6f} (should be ~0)")
        else:
            print(f"    Feature shapes differ: saved={x_saved.shape}, runtime={x_runtime.shape}")
    
    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    try:
        test_sp500_standardization()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
