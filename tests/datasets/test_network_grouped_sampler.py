"""Test NetworkGroupedBatchSampler for WRA dataset batching."""

import random
import numpy as np
from collections import defaultdict
from torch.utils.data import Sampler


class NetworkGroupedBatchSampler(Sampler):
    """Batch sampler that groups samples by network.
    
    Each batch contains samples_per_network samples from each of 
    batch_size/samples_per_network different networks.
    
    Parameters
    ----------
    dataset : Dataset
        Dataset with samples attribute containing (dataset_name, network_id, sample_idx) tuples
    batch_size : int
        Total number of samples per batch
    samples_per_network : int
        Number of samples to draw from each network (K)
    shuffle : bool
        Whether to shuffle network order each epoch
    seed : int, optional
        Random seed for reproducibility
    """
    
    def __init__(self, dataset, batch_size, samples_per_network, shuffle=True, seed=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.samples_per_network = samples_per_network
        self.shuffle = shuffle
        
        if batch_size % samples_per_network != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be divisible by "
                f"samples_per_network ({samples_per_network})"
            )
        
        self.networks_per_batch = batch_size // samples_per_network
        
        # Group samples by network
        self.network_to_indices = defaultdict(list)
        for idx, (dataset_name, network_id, sample_idx) in enumerate(dataset.samples):
            self.network_to_indices[network_id].append(idx)
        
        self.networks = list(self.network_to_indices.keys())
        self.num_networks = len(self.networks)
        
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        print(f"NetworkGroupedBatchSampler initialized:")
        print(f"  Total networks: {self.num_networks}")
        print(f"  Samples per network (K): {self.samples_per_network}")
        print(f"  Networks per batch (B/K): {self.networks_per_batch}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Total batches per epoch: {len(self)}")
    
    def __iter__(self):
        # Shuffle network order at start of each epoch
        networks = self.networks.copy()
        if self.shuffle:
            random.shuffle(networks)
        
        # Generate batches
        for i in range(0, len(networks), self.networks_per_batch):
            batch_networks = networks[i:i + self.networks_per_batch]
            batch_indices = []
            
            for net_id in batch_networks:
                available_indices = self.network_to_indices[net_id]
                
                # Sample K indices from this network
                if len(available_indices) >= self.samples_per_network:
                    # Have enough samples - random sample without replacement
                    selected = random.sample(available_indices, self.samples_per_network)
                else:
                    # Not enough samples - sample with replacement
                    selected = random.choices(available_indices, k=self.samples_per_network)
                
                batch_indices.extend(selected)
            
            yield batch_indices
    
    def __len__(self):
        """Number of batches per epoch."""
        return (self.num_networks + self.networks_per_batch - 1) // self.networks_per_batch


class MockWRADataset:
    """Mock dataset for testing."""
    
    def __init__(self, num_networks=10, samples_per_network=100):
        self.samples = []
        for net_id in range(num_networks):
            for sample_idx in range(samples_per_network):
                self.samples.append(('wra', net_id, sample_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def test_basic_batching():
    """Test basic batch structure."""
    print("\n" + "="*60)
    print("TEST 1: Basic Batching")
    print("="*60)
    
    # Create mock dataset: 10 networks, 100 samples each
    dataset = MockWRADataset(num_networks=10, samples_per_network=100)
    
    # Batch size 32, K=4 samples per network -> 8 networks per batch
    sampler = NetworkGroupedBatchSampler(
        dataset=dataset,
        batch_size=32,
        samples_per_network=4,
        shuffle=False,
        seed=42
    )
    
    # Test first batch
    batches = list(sampler)
    first_batch = batches[0]
    
    print(f"\nFirst batch indices (first 16): {first_batch[:16]}")
    print(f"Total indices in batch: {len(first_batch)}")
    
    # Extract network IDs from batch
    batch_network_ids = [dataset.samples[idx][1] for idx in first_batch]
    unique_networks = set(batch_network_ids)
    
    print(f"Unique networks in batch: {sorted(unique_networks)}")
    print(f"Number of unique networks: {len(unique_networks)}")
    
    # Count samples per network
    from collections import Counter
    network_counts = Counter(batch_network_ids)
    print(f"Samples per network: {dict(network_counts)}")
    
    # Assertions
    assert len(first_batch) == 32, f"Expected 32 samples, got {len(first_batch)}"
    assert len(unique_networks) == 8, f"Expected 8 networks, got {len(unique_networks)}"
    assert all(count == 4 for count in network_counts.values()), \
        "Not all networks have exactly 4 samples"
    
    print("✅ PASSED: Batch has correct structure (32 samples, 8 networks, 4 samples each)")


def test_network_grouping():
    """Test that samples from same network are consecutive."""
    print("\n" + "="*60)
    print("TEST 2: Network Grouping")
    print("="*60)
    
    dataset = MockWRADataset(num_networks=6, samples_per_network=50)
    
    sampler = NetworkGroupedBatchSampler(
        dataset=dataset,
        batch_size=12,  # 3 networks × 4 samples
        samples_per_network=4,
        shuffle=False,
        seed=42
    )
    
    first_batch = list(sampler)[0]
    batch_network_ids = [dataset.samples[idx][1] for idx in first_batch]
    
    print(f"Network IDs in batch order: {batch_network_ids}")
    
    # Check consecutive grouping
    for i in range(0, len(batch_network_ids), 4):
        group = batch_network_ids[i:i+4]
        assert len(set(group)) == 1, f"Samples {i}-{i+3} are not from same network: {group}"
        print(f"Samples {i:2d}-{i+3:2d}: All from network {group[0]} ✓")
    
    print("✅ PASSED: Samples are grouped by network")


def test_multiple_epochs():
    """Test shuffling across epochs."""
    print("\n" + "="*60)
    print("TEST 3: Multiple Epochs with Shuffling")
    print("="*60)
    
    dataset = MockWRADataset(num_networks=8, samples_per_network=50)
    
    sampler = NetworkGroupedBatchSampler(
        dataset=dataset,
        batch_size=16,
        samples_per_network=4,
        shuffle=True,
        seed=42
    )
    
    # Get first batch from 3 epochs
    epoch_batches = []
    for epoch in range(3):
        batches = list(sampler)
        first_batch = batches[0]
        batch_network_ids = [dataset.samples[idx][1] for idx in first_batch]
        unique_nets = sorted(set(batch_network_ids))
        epoch_batches.append(unique_nets)
        print(f"Epoch {epoch} - First batch networks: {unique_nets}")
    
    # Check that network order changes (with shuffling)
    assert epoch_batches[0] != epoch_batches[1] or epoch_batches[0] != epoch_batches[2], \
        "Networks should be shuffled between epochs"
    
    print("✅ PASSED: Network order shuffles between epochs")


def test_insufficient_samples():
    """Test handling when network has fewer samples than requested."""
    print("\n" + "="*60)
    print("TEST 4: Insufficient Samples (Sample with Replacement)")
    print("="*60)
    
    # Create dataset where some networks have fewer samples
    dataset = MockWRADataset(num_networks=4, samples_per_network=2)
    
    sampler = NetworkGroupedBatchSampler(
        dataset=dataset,
        batch_size=16,  # Want 4 samples per network, but only 2 available
        samples_per_network=4,
        shuffle=False,
        seed=42
    )
    
    first_batch = list(sampler)[0]
    print(f"Batch size: {len(first_batch)} (requested 16)")
    
    batch_network_ids = [dataset.samples[idx][1] for idx in first_batch]
    from collections import Counter
    network_counts = Counter(batch_network_ids)
    print(f"Samples per network: {dict(network_counts)}")
    
    assert len(first_batch) == 16, "Should still generate full batch size"
    assert all(count == 4 for count in network_counts.values()), \
        "Should use sampling with replacement"
    
    print("✅ PASSED: Handles insufficient samples via replacement")


def test_batch_count():
    """Test total number of batches."""
    print("\n" + "="*60)
    print("TEST 5: Batch Count")
    print("="*60)
    
    dataset = MockWRADataset(num_networks=25, samples_per_network=100)
    
    sampler = NetworkGroupedBatchSampler(
        dataset=dataset,
        batch_size=20,  # 5 networks per batch
        samples_per_network=4,
        shuffle=False
    )
    
    expected_batches = (25 + 5 - 1) // 5  # ceil(25/5) = 5
    actual_batches = len(list(sampler))
    
    print(f"Total networks: 25")
    print(f"Networks per batch: 5")
    print(f"Expected batches: {expected_batches}")
    print(f"Actual batches: {actual_batches}")
    
    assert actual_batches == expected_batches, \
        f"Expected {expected_batches} batches, got {actual_batches}"
    
    print("✅ PASSED: Correct number of batches")


if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# NetworkGroupedBatchSampler Unit Tests")
    print("#"*60)
    
    test_basic_batching()
    test_network_grouping()
    test_multiple_epochs()
    test_insufficient_samples()
    test_batch_count()
    
    print("\n" + "#"*60)
    print("# All Tests Passed! ✅")
    print("#"*60 + "\n")
