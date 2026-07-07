"""
Unit tests for ReplicatedDataset wrapper.

Tests the functionality of sample replication for evaluation scenarios
where multiple predictions per input are needed.
"""

import pytest
import torch
from torch.utils.data import Dataset, DataLoader

import sys
sys.path.insert(0, 'src')

from graph_signal_diffusion.datasets import ReplicatedDataset


class SimpleDataset(Dataset):
    """Simple dataset for testing."""
    def __init__(self, size=10):
        self.size = size
        self.data = [f"sample_{i}" for i in range(size)]
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            'data': self.data[idx],
            'value': idx * 10,
            'tensor': torch.tensor([idx, idx + 1, idx + 2])
        }


class TestReplicatedDataset:
    """Test suite for ReplicatedDataset."""
    
    def test_basic_properties(self):
        """Test basic length and properties."""
        dataset = SimpleDataset(size=10)
        replicated = ReplicatedDataset(dataset, n_replicas=5)
        
        assert len(replicated) == 50  # 10 * 5
        assert replicated.n_replicas == 5
        assert replicated._original_length == 10
    
    def test_no_replication(self):
        """Test that n_replicas=1 behaves like original dataset."""
        dataset = SimpleDataset(size=10)
        replicated = ReplicatedDataset(dataset, n_replicas=1)
        
        assert len(replicated) == 10
        
        # Should get same results as original
        for i in range(10):
            original_sample = dataset[i]
            replicated_sample = replicated[i]
            assert original_sample['data'] == replicated_sample['data']
            assert original_sample['value'] == replicated_sample['value']
            assert torch.equal(original_sample['tensor'], replicated_sample['tensor'])
    
    def test_index_mapping(self):
        """Test that replicated indices map correctly to original indices."""
        dataset = SimpleDataset(size=5)
        replicated = ReplicatedDataset(dataset, n_replicas=3)
        
        # 5 samples × 3 replicas = 15 total
        assert len(replicated) == 15
        
        # Indices 0-2 should map to original sample 0
        for i in range(3):
            sample = replicated[i]
            assert sample['data'] == 'sample_0'
            assert sample['value'] == 0
        
        # Indices 3-5 should map to original sample 1
        for i in range(3, 6):
            sample = replicated[i]
            assert sample['data'] == 'sample_1'
            assert sample['value'] == 10
        
        # Indices 6-8 should map to original sample 2
        for i in range(6, 9):
            sample = replicated[i]
            assert sample['data'] == 'sample_2'
            assert sample['value'] == 20
        
        # Indices 9-11 should map to original sample 3
        for i in range(9, 12):
            sample = replicated[i]
            assert sample['data'] == 'sample_3'
            assert sample['value'] == 30
        
        # Indices 12-14 should map to original sample 4
        for i in range(12, 15):
            sample = replicated[i]
            assert sample['data'] == 'sample_4'
            assert sample['value'] == 40
    
    def test_index_formula(self):
        """Test the index mapping formula: original_idx = idx // n_replicas."""
        dataset = SimpleDataset(size=10)
        replicated = ReplicatedDataset(dataset, n_replicas=7)
        
        for i in range(len(replicated)):
            expected_original_idx = i // 7
            sample = replicated[i]
            assert sample['value'] == expected_original_idx * 10
    
    def test_with_dataloader(self):
        """Test integration with PyTorch DataLoader."""
        dataset = SimpleDataset(size=10)
        replicated = ReplicatedDataset(dataset, n_replicas=5)
        
        loader = DataLoader(replicated, batch_size=10, shuffle=False)
        
        batches = list(loader)
        assert len(batches) == 5  # 50 samples / batch_size=10
        
        # First batch should contain first 10 samples (indices 0-9)
        # These are 2 complete replicas of samples 0 and 1
        first_batch = batches[0]
        assert len(first_batch['data']) == 10
        
        # Indices 0-4 → sample 0, indices 5-9 → sample 1
        expected_values = [0, 0, 0, 0, 0, 10, 10, 10, 10, 10]
        assert first_batch['value'].tolist() == expected_values
    
    def test_replica_grouping(self):
        """Test that replicas can be grouped correctly."""
        dataset = SimpleDataset(size=4)
        replicated = ReplicatedDataset(dataset, n_replicas=3)
        
        # Collect all samples
        all_samples = [replicated[i] for i in range(len(replicated))]
        
        # Group by original index
        grouped = {}
        for i, sample in enumerate(all_samples):
            original_idx = i // 3
            if original_idx not in grouped:
                grouped[original_idx] = []
            grouped[original_idx].append(sample)
        
        # Should have 4 groups
        assert len(grouped) == 4
        
        # Each group should have 3 replicas
        for original_idx, group in grouped.items():
            assert len(group) == 3
            # All replicas should be identical
            for replica in group:
                assert replica['data'] == f'sample_{original_idx}'
                assert replica['value'] == original_idx * 10
    
    def test_large_replication(self):
        """Test with large replication factor."""
        dataset = SimpleDataset(size=3)
        replicated = ReplicatedDataset(dataset, n_replicas=100)
        
        assert len(replicated) == 300
        
        # First 100 should all be sample 0
        for i in range(100):
            assert replicated[i]['data'] == 'sample_0'
        
        # Next 100 should all be sample 1
        for i in range(100, 200):
            assert replicated[i]['data'] == 'sample_1'
        
        # Last 100 should all be sample 2
        for i in range(200, 300):
            assert replicated[i]['data'] == 'sample_2'
    
    def test_tensor_data_integrity(self):
        """Test that tensor data is correctly replicated."""
        dataset = SimpleDataset(size=5)
        replicated = ReplicatedDataset(dataset, n_replicas=4)
        
        # Get replicas of sample 2 (indices 8-11)
        replicas = [replicated[i] for i in range(8, 12)]
        
        # All should be identical
        reference = replicas[0]
        for replica in replicas[1:]:
            assert torch.equal(replica['tensor'], reference['tensor'])
            assert torch.equal(replica['tensor'], torch.tensor([2, 3, 4]))
    
    def test_repr(self):
        """Test string representation."""
        dataset = SimpleDataset(size=10)
        replicated = ReplicatedDataset(dataset, n_replicas=5)
        
        repr_str = repr(replicated)
        assert "ReplicatedDataset" in repr_str
        assert "SimpleDataset" in repr_str
        assert "original_length=10" in repr_str
        assert "n_replicas=5" in repr_str
        assert "total_length=50" in repr_str
    
    def test_edge_case_single_sample(self):
        """Test with single-sample dataset."""
        dataset = SimpleDataset(size=1)
        replicated = ReplicatedDataset(dataset, n_replicas=10)
        
        assert len(replicated) == 10
        
        # All should be the same sample
        for i in range(10):
            sample = replicated[i]
            assert sample['data'] == 'sample_0'
            assert sample['value'] == 0
    
    def test_with_subset(self):
        """Test compatibility with torch.utils.data.Subset."""
        from torch.utils.data import Subset
        
        dataset = SimpleDataset(size=20)
        subset = Subset(dataset, indices=[2, 5, 8, 11, 14])  # 5 samples
        replicated = ReplicatedDataset(subset, n_replicas=3)
        
        assert len(replicated) == 15  # 5 * 3
        
        # First 3 should be sample 2
        for i in range(3):
            assert replicated[i]['value'] == 20  # sample 2 has value 20
        
        # Next 3 should be sample 5
        for i in range(3, 6):
            assert replicated[i]['value'] == 50  # sample 5 has value 50


class TestReplicatedDatasetBatching:
    """Test batching behavior with ReplicatedDataset."""
    
    def test_consecutive_replicas_in_batch(self):
        """Test that consecutive replicas can be batched together."""
        dataset = SimpleDataset(size=5)
        replicated = ReplicatedDataset(dataset, n_replicas=10)
        
        # With batch_size=10 and shuffle=False, first batch should be all replicas of sample 0
        loader = DataLoader(replicated, batch_size=10, shuffle=False)
        first_batch = next(iter(loader))
        
        # All values should be 0 (sample 0)
        assert all(v == 0 for v in first_batch['value'].tolist())
        assert all(d == 'sample_0' for d in first_batch['data'])
    
    def test_mixed_samples_in_batch(self):
        """Test batches with replicas from multiple original samples."""
        dataset = SimpleDataset(size=10)
        replicated = ReplicatedDataset(dataset, n_replicas=3)
        
        # batch_size=15 should span 5 original samples (each with 3 replicas)
        loader = DataLoader(replicated, batch_size=15, shuffle=False)
        first_batch = next(iter(loader))
        
        # Should have samples 0-4, each replicated 3 times
        expected_values = []
        for i in range(5):
            expected_values.extend([i * 10] * 3)
        
        assert first_batch['value'].tolist() == expected_values
    
    def test_drop_last_behavior(self):
        """Test drop_last behavior with replicated dataset."""
        dataset = SimpleDataset(size=5)
        replicated = ReplicatedDataset(dataset, n_replicas=7)  # 35 total
        
        loader_drop = DataLoader(replicated, batch_size=10, shuffle=False, drop_last=True)
        loader_keep = DataLoader(replicated, batch_size=10, shuffle=False, drop_last=False)
        
        batches_drop = list(loader_drop)
        batches_keep = list(loader_keep)
        
        assert len(batches_drop) == 3  # 30 samples (dropped 5)
        assert len(batches_keep) == 4  # 35 samples (last batch has 5)
        assert len(batches_keep[-1]['value']) == 5


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
