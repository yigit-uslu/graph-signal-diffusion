"""Tests for ReplicatedGroupedBatchSampler."""

import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader

from graph_signal_diffusion.datasets.replicated_dataset import ReplicatedDataset
from graph_signal_diffusion.datasets.replicated_sampler import ReplicatedGroupedBatchSampler


class TestReplicatedGroupedBatchSampler:
    """Test suite for ReplicatedGroupedBatchSampler."""
    
    def test_basic_grouping(self):
        """Test that replicas are grouped together in batches."""
        # Original dataset with 5 samples
        original = TensorDataset(torch.arange(5).unsqueeze(1))
        replicated = ReplicatedDataset(original, n_replicas=3)
        
        sampler = ReplicatedGroupedBatchSampler(replicated, shuffle=False)
        
        batches = list(sampler)
        assert len(batches) == 5  # One batch per original sample
        
        # Check first batch: should be indices [0, 1, 2] (3 replicas of sample 0)
        assert batches[0] == [0, 1, 2]
        
        # Check second batch: should be indices [3, 4, 5] (3 replicas of sample 1)
        assert batches[1] == [3, 4, 5]
        
        # Check last batch: should be indices [12, 13, 14] (3 replicas of sample 4)
        assert batches[4] == [12, 13, 14]
    
    def test_with_dataloader(self):
        """Test integration with DataLoader."""
        # Original dataset with 10 samples
        original_data = torch.arange(10).unsqueeze(1).float()
        original = TensorDataset(original_data)
        replicated = ReplicatedDataset(original, n_replicas=5)
        
        sampler = ReplicatedGroupedBatchSampler(replicated, shuffle=False, seed=42)
        loader = DataLoader(replicated, batch_sampler=sampler)
        
        batches = list(loader)
        
        # Should have 10 batches (one per original sample)
        assert len(batches) == 10
        
        # Each batch should have 5 samples
        for batch in batches:
            assert len(batch) == 1  # DataLoader returns list of tensors
            assert batch[0].shape[0] == 5  # Each tensor has 5 replicas
        
        # First batch should have 5 copies of value 0
        first_batch_values = batches[0][0].squeeze()
        assert torch.all(first_batch_values == 0.0)
        
        # Second batch should have 5 copies of value 1
        second_batch_values = batches[1][0].squeeze()
        assert torch.all(second_batch_values == 1.0)
    
    def test_shuffle(self):
        """Test that shuffle randomizes order of original samples."""
        original = TensorDataset(torch.arange(10).unsqueeze(1))
        replicated = ReplicatedDataset(original, n_replicas=4)
        
        sampler_no_shuffle = ReplicatedGroupedBatchSampler(replicated, shuffle=False, seed=42)
        sampler_shuffle = ReplicatedGroupedBatchSampler(replicated, shuffle=True, seed=42)
        
        batches_no_shuffle = list(sampler_no_shuffle)
        batches_shuffle = list(sampler_shuffle)
        
        # Both should have same length
        assert len(batches_no_shuffle) == len(batches_shuffle) == 10
        
        # Shuffled order should be different from sequential
        assert batches_no_shuffle != batches_shuffle
        
        # But each batch in shuffled version should still be a valid replica group
        for batch_indices in batches_shuffle:
            assert len(batch_indices) == 4
            # All indices in a batch should map to same original sample
            original_idx = batch_indices[0] // 4
            for idx in batch_indices:
                assert idx // 4 == original_idx
    
    def test_length(self):
        """Test __len__ returns correct number of batches."""
        original = TensorDataset(torch.arange(20).unsqueeze(1))
        replicated = ReplicatedDataset(original, n_replicas=7)
        
        sampler = ReplicatedGroupedBatchSampler(replicated)
        
        # Should have 20 batches (one per original sample)
        assert len(sampler) == 20
        assert len(list(sampler)) == 20
    
    def test_type_checking(self):
        """Test that sampler requires ReplicatedDataset."""
        regular_dataset = TensorDataset(torch.arange(10).unsqueeze(1))
        
        with pytest.raises(TypeError, match="Expected ReplicatedDataset"):
            ReplicatedGroupedBatchSampler(regular_dataset)
    
    def test_single_replica(self):
        """Test edge case with n_replicas=1."""
        original = TensorDataset(torch.arange(5).unsqueeze(1))
        replicated = ReplicatedDataset(original, n_replicas=1)
        
        sampler = ReplicatedGroupedBatchSampler(replicated, shuffle=False)
        batches = list(sampler)
        
        # Should have 5 batches, each with 1 sample
        assert len(batches) == 5
        assert all(len(batch) == 1 for batch in batches)
        assert batches == [[0], [1], [2], [3], [4]]
    
    def test_repr(self):
        """Test string representation."""
        original = TensorDataset(torch.arange(10).unsqueeze(1))
        replicated = ReplicatedDataset(original, n_replicas=3)
        sampler = ReplicatedGroupedBatchSampler(replicated, shuffle=True)
        
        repr_str = repr(sampler)
        assert "ReplicatedGroupedBatchSampler" in repr_str
        assert "original_samples=10" in repr_str
        assert "n_replicas=3" in repr_str
        assert "batches=10" in repr_str
        assert "batch_size=3" in repr_str
        assert "shuffle=True" in repr_str
    
    def test_deterministic_with_seed(self):
        """Test that same seed produces same order."""
        original = TensorDataset(torch.arange(20).unsqueeze(1))
        replicated = ReplicatedDataset(original, n_replicas=5)
        
        sampler1 = ReplicatedGroupedBatchSampler(replicated, shuffle=True, seed=123)
        sampler2 = ReplicatedGroupedBatchSampler(replicated, shuffle=True, seed=123)
        
        batches1 = list(sampler1)
        batches2 = list(sampler2)
        
        assert batches1 == batches2
    
    def test_different_seeds_different_order(self):
        """Test that different seeds produce different orders."""
        original = TensorDataset(torch.arange(20).unsqueeze(1))
        replicated = ReplicatedDataset(original, n_replicas=5)
        
        sampler1 = ReplicatedGroupedBatchSampler(replicated, shuffle=True, seed=1)
        sampler2 = ReplicatedGroupedBatchSampler(replicated, shuffle=True, seed=2)
        
        batches1 = list(sampler1)
        batches2 = list(sampler2)
        
        # Orders should be different (with very high probability for 20 samples)
        assert batches1 != batches2
    
    def test_originals_per_batch_basic(self):
        """Test batching multiple original samples together."""
        # 10 original samples, 3 replicas each
        original = TensorDataset(torch.arange(10).unsqueeze(1))
        replicated = ReplicatedDataset(original, n_replicas=3)
        
        # Batch 2 originals at a time
        sampler = ReplicatedGroupedBatchSampler(replicated, originals_per_batch=2, shuffle=False)
        batches = list(sampler)
        
        # Should have 5 batches (10 originals / 2 per batch)
        assert len(batches) == 5
        
        # Each batch should have 6 samples (2 originals × 3 replicas)
        for batch in batches:
            assert len(batch) == 6
        
        # First batch: originals 0 and 1, with their replicas
        assert batches[0] == [0, 1, 2, 3, 4, 5]
        
        # Second batch: originals 2 and 3
        assert batches[1] == [6, 7, 8, 9, 10, 11]
    
    def test_originals_per_batch_with_dataloader(self):
        """Test originals_per_batch with DataLoader."""
        original_data = torch.arange(12).unsqueeze(1).float()
        original = TensorDataset(original_data)
        replicated = ReplicatedDataset(original, n_replicas=4)
        
        # Batch 3 originals at a time (batch_size = 3 × 4 = 12)
        sampler = ReplicatedGroupedBatchSampler(replicated, originals_per_batch=3, shuffle=False)
        loader = DataLoader(replicated, batch_sampler=sampler)
        
        batches = list(loader)
        
        # Should have 4 batches (12 originals / 3 per batch)
        assert len(batches) == 4
        
        # Each batch should have 12 samples (3 originals × 4 replicas)
        for batch in batches:
            assert batch[0].shape[0] == 12
        
        # First batch: originals 0, 1, 2 with their replicas
        first_batch_values = batches[0][0].squeeze()
        expected = torch.tensor([0., 0., 0., 0., 1., 1., 1., 1., 2., 2., 2., 2.])
        assert torch.all(first_batch_values == expected)
    
    def test_originals_per_batch_not_divisible(self):
        """Test when original_length is not divisible by originals_per_batch."""
        # 10 originals, batch 3 at a time
        original = TensorDataset(torch.arange(10).unsqueeze(1))
        replicated = ReplicatedDataset(original, n_replicas=2)
        
        sampler = ReplicatedGroupedBatchSampler(replicated, originals_per_batch=3, shuffle=False)
        batches = list(sampler)
        
        # Should have 4 batches (ceil(10/3) = 4)
        assert len(batches) == 4
        
        # First 3 batches have 6 samples (3 originals × 2 replicas)
        assert len(batches[0]) == 6
        assert len(batches[1]) == 6
        assert len(batches[2]) == 6
        
        # Last batch has 2 samples (1 original × 2 replicas)
        assert len(batches[3]) == 2
    
    def test_originals_per_batch_equals_total(self):
        """Test when originals_per_batch equals total originals."""
        original = TensorDataset(torch.arange(5).unsqueeze(1))
        replicated = ReplicatedDataset(original, n_replicas=3)
        
        # All originals in one batch
        sampler = ReplicatedGroupedBatchSampler(replicated, originals_per_batch=5, shuffle=False)
        batches = list(sampler)
        
        # Should have 1 batch
        assert len(batches) == 1
        
        # Should have all 15 samples (5 originals × 3 replicas)
        assert len(batches[0]) == 15
    
    def test_originals_per_batch_invalid(self):
        """Test validation of originals_per_batch."""
        original = TensorDataset(torch.arange(5).unsqueeze(1))
        replicated = ReplicatedDataset(original, n_replicas=2)
        
        with pytest.raises(ValueError, match="originals_per_batch must be >= 1"):
            ReplicatedGroupedBatchSampler(replicated, originals_per_batch=0)
        
        with pytest.raises(ValueError, match="originals_per_batch must be >= 1"):
            ReplicatedGroupedBatchSampler(replicated, originals_per_batch=-1)
