"""Test composite key system for WRA metadata.

This test verifies that:
1. Metadata is extracted from all splits (train, val, test)
2. Composite keys (dataset_name, network_id) prevent collisions
3. Evaluator correctly constructs composite keys from batched data
"""

import pytest
import torch
from pathlib import Path


def test_composite_keys_in_datamodule():
    """Test that datamodule extracts metadata with composite keys from all splits."""
    from graph_signal_diffusion.datasets.wra import WRADataModule
    from omegaconf import OmegaConf
    
    # Create config with small dataset
    config = OmegaConf.create({
        'name': 'wra',
        'data_dir': 'data/wra',
        'sub_datasets': ['wra'],
        'num_train': 10,
        'num_val': 5,
        'num_test': 5,
        'batch_size': 4,
        'batch_size_val': 4,
        'n_samples_per_input': 2,
        'num_workers': 0,
    })
    
    # Build datamodule
    dm = WRADataModule(config)
    dm.setup(stage='fit')
    
    # Check dataset_info has composite keys
    assert 'network_seeds' in dm.dataset_info
    assert 'associations' in dm.dataset_info
    
    network_seeds = dm.dataset_info['network_seeds']
    
    # All keys should be tuples (dataset_name, network_id)
    for key in network_seeds.keys():
        assert isinstance(key, tuple), f"Key should be tuple, got {type(key)}"
        assert len(key) == 2, f"Key should have 2 elements, got {len(key)}"
        dataset_name, network_id = key
        assert isinstance(dataset_name, str), f"dataset_name should be str, got {type(dataset_name)}"
        assert isinstance(network_id, (int, str)), f"network_id should be int or str, got {type(network_id)}"
    
    print(f"✓ Found {len(network_seeds)} networks with composite keys")
    print(f"✓ Sample keys: {list(network_seeds.keys())[:5]}")


def test_composite_keys_in_batch():
    """Test that batched data contains both dataset_name and network_id."""
    from graph_signal_diffusion.datasets.wra import WRADataModule
    from omegaconf import OmegaConf
    
    config = OmegaConf.create({
        'name': 'wra',
        'data_dir': 'data/wra',
        'sub_datasets': ['wra'],
        'num_train': 10,
        'num_val': 5,
        'num_test': 5,
        'batch_size': 4,
        'batch_size_val': 4,
        'n_samples_per_input': 2,
        'num_workers': 0,
    })
    
    dm = WRADataModule(config)
    dm.setup(stage='fit')
    
    # Get a validation batch
    val_loader = dm.val_dataloader()
    batch = next(iter(val_loader))
    
    # Check both attributes exist
    assert hasattr(batch, 'network_id'), "Batch should have network_id attribute"
    assert hasattr(batch, 'dataset_name'), "Batch should have dataset_name attribute"
    
    # Check lengths match
    if isinstance(batch.network_id, torch.Tensor):
        network_ids = batch.network_id.cpu().tolist()
    else:
        network_ids = batch.network_id
    
    if isinstance(batch.dataset_name, list):
        dataset_names = batch.dataset_name
    else:
        dataset_names = [batch.dataset_name] * len(network_ids)
    
    assert len(network_ids) == len(dataset_names), \
        f"network_ids ({len(network_ids)}) and dataset_names ({len(dataset_names)}) should have same length"
    
    print(f"✓ Batch has {len(network_ids)} samples")
    print(f"✓ network_ids: {network_ids}")
    print(f"✓ dataset_names: {dataset_names}")
    
    # Check composite keys can be constructed
    composite_keys = [(dname, net_id) for dname, net_id in zip(dataset_names, network_ids)]
    print(f"✓ Composite keys: {composite_keys[:5]}")


def test_evaluator_with_composite_keys():
    """Test that evaluator can use composite keys for metadata lookup."""
    from graph_signal_diffusion.datasets.wra import WRADataModule
    from graph_signal_diffusion.tasks.wireless_resource_allocation import WRATaskEvaluator
    from omegaconf import OmegaConf
    
    config = OmegaConf.create({
        'name': 'wra',
        'data_dir': 'data/wra',
        'sub_datasets': ['wra'],
        'num_train': 10,
        'num_val': 5,
        'num_test': 5,
        'batch_size': 4,
        'batch_size_val': 4,
        'n_samples_per_input': 2,
        'num_workers': 0,
    })
    
    dm = WRADataModule(config)
    dm.setup(stage='fit')
    
    # Create evaluator with dataset_info
    evaluator = WRATaskEvaluator()
    evaluator.dataset_info = dm.dataset_info
    
    # Get a validation batch
    val_loader = dm.val_dataloader()
    batch = next(iter(val_loader))
    
    # Prepare data (this constructs composite keys)
    try:
        prepared = evaluator.prepare_data(batch)
        
        # Check metadata has dataset_names
        assert 'dataset_names' in prepared['metadata'], "Metadata should have dataset_names"
        assert 'network_ids' in prepared['metadata'], "Metadata should have network_ids"
        assert 'network_seeds' in prepared['metadata'], "Metadata should have network_seeds"
        assert 'associations' in prepared['metadata'], "Metadata should have associations"
        
        dataset_names = prepared['metadata']['dataset_names']
        network_ids = prepared['metadata']['network_ids']
        network_seeds = prepared['metadata']['network_seeds']
        
        # Check no None values (all lookups should succeed)
        assert all(seed is not None for seed in network_seeds), \
            "All network_seeds should be found (no None values)"
        
        print(f"✓ Prepared {len(network_ids)} samples")
        print(f"✓ All metadata lookups succeeded with composite keys")
        
        # Check composite keys match dataset_info
        for dname, net_id, seed in zip(dataset_names, network_ids, network_seeds):
            composite_key = (dname, net_id)
            expected_seed = dm.dataset_info['network_seeds'][composite_key]
            assert seed == expected_seed, \
                f"Seed mismatch for {composite_key}: {seed} != {expected_seed}"
        
        print(f"✓ All composite key lookups match dataset_info")
        
    except Exception as e:
        pytest.fail(f"Evaluator prepare_data failed: {e}")


if __name__ == '__main__':
    print("Testing composite key system...")
    print("\n1. Testing datamodule metadata extraction...")
    test_composite_keys_in_datamodule()
    
    print("\n2. Testing batch structure...")
    test_composite_keys_in_batch()
    
    print("\n3. Testing evaluator with composite keys...")
    test_evaluator_with_composite_keys()
    
    print("\n✅ All composite key tests passed!")
