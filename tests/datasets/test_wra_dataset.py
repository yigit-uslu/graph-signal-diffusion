#!/usr/bin/env python
"""
Test script to verify WRA dataset loading and processing.

This script:
1. Loads the WRADataset from raw data
2. Verifies processing (raw → processed)
3. Tests dataset indexing and data shapes
4. Creates a DataLoader and tests batching
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
from torch_geometric.loader import DataLoader
from graph_signal_diffusion.datasets.wra.dataset import WRADataset
import numpy as np


def test_dataset_loading():
    """Test basic dataset loading and processing."""
    print("="*60)
    print("Testing WRA Dataset Loading")
    print("="*60)
    
    # Configuration
    root = Path("data/wra")
    dataset_names = ["N_50_density_102.0_seed_42_Pmax_dBm_10.0_rmin_0.5"]
    
    print(f"\nDataset: {dataset_names[0]}")
    print(f"Root: {root}")
    
    # Check if raw data exists
    raw_path = root / "raw" / dataset_names[0]
    if not raw_path.exists():
        print(f"❌ Raw data not found at: {raw_path}")
        return False
    
    print(f"✓ Raw data found at: {raw_path}")
    
    # Load dataset (this will trigger processing if needed)
    print("\nLoading dataset (will process if needed)...")
    try:
        dataset = WRADataset(
            root=str(root),
            dataset_names=dataset_names,
            split='train',
            train_fraction=0.75,
            val_fraction=0.125,
            test_fraction=0.125,
            normalize=True,
            normalize_method='rescale',
            cache_graphs=True,
        )
        print(f"✓ Dataset loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check dataset statistics
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Split: {dataset.split}")
    
    # Test indexing
    print("\nTesting dataset indexing...")
    try:
        sample = dataset[0]
        print(f"✓ Successfully indexed sample 0")
        print(f"\n  Sample attributes:")
        print(f"    - x shape: {sample.x.shape}")
        print(f"    - y shape: {sample.y.shape}")
        print(f"    - edge_index shape: {sample.edge_index.shape}")
        print(f"    - edge_weight shape: {sample.edge_weight.shape}")
        print(f"    - num_nodes: {sample.num_nodes}")
        print(f"    - num_edges: {sample.num_edges}")
        
        # Check data types
        print(f"\n  Data types:")
        print(f"    - x dtype: {sample.x.dtype}")
        print(f"    - y dtype: {sample.y.dtype}")
        print(f"    - edge_weight dtype: {sample.edge_weight.dtype}")
        
        # Check value ranges
        print(f"\n  Value ranges:")
        print(f"    - x: [{sample.x.min():.4f}, {sample.x.max():.4f}]")
        print(f"    - y: [{sample.y.min():.4f}, {sample.y.max():.4f}]")
        print(f"    - edge_weight: [{sample.edge_weight.min():.4f}, {sample.edge_weight.max():.4f}]")
        
        # Check attributes
        if hasattr(sample, 'network_id'):
            print(f"\n  Metadata:")
            print(f"    - network_id: {sample.network_id}")
        
    except Exception as e:
        print(f"❌ Failed to index sample: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_dataloader():
    """Test DataLoader batching."""
    print("\n" + "="*60)
    print("Testing DataLoader")
    print("="*60)
    
    root = Path("data/wra")
    dataset_names = ["N_50_density_102.0_seed_42_Pmax_dBm_10.0_rmin_0.5"]
    
    # Load dataset
    print("\nCreating dataset...")
    dataset = WRADataset(
        root=str(root),
        dataset_names=dataset_names,
        split='train',
        train_fraction=0.75,
        val_fraction=0.125,
        test_fraction=0.125,
        normalize=True,
        normalize_method='rescale',
        cache_graphs=False,  # Disable caching for this test
    )
    
    # Create DataLoader
    print("Creating DataLoader...")
    try:
        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,
            follow_batch=['y'],  # Important for variable-size graphs
        )
        print(f"✓ DataLoader created with batch_size=4")
    except Exception as e:
        print(f"❌ Failed to create DataLoader: {e}")
        return False
    
    # Test batching
    print("\nTesting batch iteration...")
    try:
        batch = next(iter(loader))
        print(f"✓ Successfully loaded batch")
        print(f"\n  Batch attributes:")
        print(f"    - x shape: {batch.x.shape}")
        print(f"    - y shape: {batch.y.shape}")
        print(f"    - edge_index shape: {batch.edge_index.shape}")
        print(f"    - edge_weight shape: {batch.edge_weight.shape}")
        print(f"    - num_graphs: {batch.num_graphs}")
        print(f"    - batch vector shape: {batch.batch.shape}")
        print(f"    - y_batch vector shape: {batch.y_batch.shape}")
        print(f"    - ptr: {batch.ptr}")
        
        # Verify batching is correct
        print(f"\n  Batch validation:")
        print(f"    - Total nodes in batch: {batch.num_nodes}")
        print(f"    - Expected nodes (if all same size): ~{batch.num_graphs * (batch.num_nodes // batch.num_graphs)}")
        print(f"    - Node assignment correct: {torch.all(batch.batch < batch.num_graphs)}")
        
    except Exception as e:
        print(f"❌ Failed to iterate batch: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_all_splits():
    """Test loading all splits."""
    print("\n" + "="*60)
    print("Testing All Splits")
    print("="*60)
    
    root = Path("data/wra")
    dataset_names = ["N_50_density_102.0_seed_42_Pmax_dBm_10.0_rmin_0.5"]
    
    for split in ['train', 'val', 'test']:
        print(f"\n{split.upper()} split:")
        try:
            dataset = WRADataset(
                root=str(root),
                dataset_names=dataset_names,
                split=split,
                train_fraction=0.75,
                val_fraction=0.125,
                test_fraction=0.125,
                normalize=True,
                normalize_method='rescale',
                cache_graphs=True,
            )
            print(f"  ✓ Loaded {len(dataset)} samples")
        except Exception as e:
            print(f"  ❌ Failed to load {split}: {e}")
            return False
    
    return True


def test_graph_caching():
    """Test graph caching functionality."""
    print("\n" + "="*60)
    print("Testing Graph Caching")
    print("="*60)
    
    root = Path("data/wra")
    dataset_names = ["N_50_density_102.0_seed_42_Pmax_dBm_10.0_rmin_0.5"]
    
    # Load with caching enabled
    print("\nLoading with cache_graphs=True...")
    dataset_cached = WRADataset(
        root=str(root),
        dataset_names=dataset_names,
        split='train',
        train_fraction=0.75,
        val_fraction=0.125,
        test_fraction=0.125,
        normalize=True,
        cache_graphs=True,
    )
    
    # Load same sample twice to trigger cache
    print("Loading same sample twice...")
    import time
    
    start = time.time()
    sample1 = dataset_cached[0]
    time1 = time.time() - start
    
    start = time.time()
    sample2 = dataset_cached[0]
    time2 = time.time() - start
    
    print(f"  First load: {time1*1000:.2f}ms")
    print(f"  Second load (cached): {time2*1000:.2f}ms")
    print(f"  Speedup: {time1/time2:.1f}x")
    
    # Verify cache was used
    if time2 < time1 * 0.5:  # At least 2x faster
        print(f"  ✓ Caching appears to be working")
    else:
        print(f"  ⚠ Caching may not be working (similar times)")
    
    print(f"\n  Cache statistics:")
    print(f"    Cached graphs: {len(dataset_cached.graph_cache)}")
    
    return True


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print(" WRA Dataset Test Suite")
    print("="*70)
    
    tests = [
        ("Dataset Loading", test_dataset_loading),
        ("DataLoader", test_dataloader),
        ("All Splits", test_all_splits),
        ("Graph Caching", test_graph_caching),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            success = test_func()
            results[name] = success
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False
    
    # Summary
    print("\n" + "="*70)
    print(" Test Summary")
    print("="*70)
    for name, success in results.items():
        status = "✓ PASS" if success else "❌ FAIL"
        print(f"  {status}: {name}")
    
    all_passed = all(results.values())
    print("\n" + "="*70)
    if all_passed:
        print(" ✓ All tests passed!")
    else:
        print(" ❌ Some tests failed")
    print("="*70 + "\n")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())
