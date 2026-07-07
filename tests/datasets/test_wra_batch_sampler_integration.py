"""Integration test for NetworkGroupedBatchSampler with real WRA dataset."""

import sys
import random
from pathlib import Path
import numpy as np
import torch
from collections import defaultdict, Counter
from torch.utils.data import Sampler
from torch_geometric.loader import DataLoader

# Resolve the repo root from this file (tests/datasets/<file> -> repo root) so the
# test is portable instead of hardcoding an absolute home path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from graph_signal_diffusion.datasets.wra.dataset import WRADataset


class NetworkGroupedBatchSampler(Sampler):
    """Batch sampler that groups samples by network."""
    
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
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        print(f"NetworkGroupedBatchSampler initialized:")
        print(f"  Total networks: {self.num_networks}")
        print(f"  Samples per network (K): {self.samples_per_network}")
        print(f"  Networks per batch (B/K): {self.networks_per_batch}")
        print(f"  Batch size: {self.batch_size}")
    
    def __iter__(self):
        networks = self.networks.copy()
        if self.shuffle:
            random.shuffle(networks)
        
        for i in range(0, len(networks), self.networks_per_batch):
            batch_networks = networks[i:i + self.networks_per_batch]
            batch_indices = []
            
            for net_id in batch_networks:
                available_indices = self.network_to_indices[net_id]
                
                if len(available_indices) >= self.samples_per_network:
                    selected = random.sample(available_indices, self.samples_per_network)
                else:
                    selected = random.choices(available_indices, k=self.samples_per_network)
                
                batch_indices.extend(selected)
            
            yield batch_indices
    
    def __len__(self):
        return (self.num_networks + self.networks_per_batch - 1) // self.networks_per_batch


def test_wra_batch_attributes():
    """Test PyG batching with NetworkGroupedBatchSampler on real WRA data."""
    print("\n" + "="*70)
    print("TEST: WRA Dataset with NetworkGroupedBatchSampler - Batch Attributes")
    print("="*70)
    
    # Load real WRA dataset
    print("\nLoading WRA dataset...")
    dataset = WRADataset(
        root=str(_REPO_ROOT / "data" / "wra"),
        dataset_names=['N_50_density_102.0_seed_42_Pmax_dBm_10.0_rmin_0.5'],
        split='train',
        train_fraction=0.7,
        val_fraction=0.15,
        test_fraction=0.15,
        normalize=True,
        cache_graphs=True,
        max_networks_per_split=10,  # Use only 10 networks for testing
        max_samples_per_network=20,  # 20 samples per network
    )
    
    print(f"Dataset loaded: {len(dataset)} total samples")
    print(f"Sample 0 structure: {dataset[0]}")
    
    # Create sampler: batch_size=16, K=4 samples per network
    sampler = NetworkGroupedBatchSampler(
        dataset=dataset,
        batch_size=16,
        samples_per_network=4,
        shuffle=False,
        seed=42
    )
    
    # Create DataLoader
    print("\nCreating DataLoader with PyG batching...")
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        follow_batch=['y'],
        exclude_keys=['info'],
        num_workers=0,
    )
    
    # Get first batch
    print("\nFetching first batch...")
    batch = next(iter(loader))
    
    print("\n" + "-"*70)
    print("Batch Contents:")
    print("-"*70)
    print(f"x shape: {batch.x.shape}")
    print(f"edge_index shape: {batch.edge_index.shape}")
    print(f"edge_weight shape: {batch.edge_weight.shape}")
    print(f"y shape: {batch.y.shape}")
    print(f"batch vector shape: {batch.batch.shape}")
    print(f"y_batch vector shape: {batch.y_batch.shape if hasattr(batch, 'y_batch') else 'N/A'}")
    print(f"network_id: {batch.network_id if hasattr(batch, 'network_id') else 'N/A'}")
    print(f"ptr: {batch.ptr if hasattr(batch, 'ptr') else 'N/A'}")
    
    # Analyze network grouping
    print("\n" + "-"*70)
    print("Network Grouping Analysis:")
    print("-"*70)
    
    if hasattr(batch, 'network_id'):
        network_ids = batch.network_id.tolist() if torch.is_tensor(batch.network_id) else batch.network_id
        print(f"Network IDs in batch: {network_ids}")
        
        # Count unique networks
        unique_networks = set(network_ids)
        print(f"Unique networks: {len(unique_networks)}")
        print(f"Unique network IDs: {sorted(unique_networks)}")
        
        # Count samples per network
        network_counts = Counter(network_ids)
        print(f"Samples per network: {dict(network_counts)}")
        
        # Check consecutive grouping
        print("\nChecking consecutive grouping:")
        for i in range(0, len(network_ids), 4):
            group = network_ids[i:min(i+4, len(network_ids))]
            is_same = len(set(group)) == 1
            status = "✓" if is_same else "✗"
            print(f"  Indices {i:2d}-{i+3:2d}: {group} {status}")
        
        # Verify structure
        assert len(network_ids) == 16, f"Expected 16 graphs, got {len(network_ids)}"
        assert len(unique_networks) == 4, f"Expected 4 unique networks, got {len(unique_networks)}"
        assert all(count == 4 for count in network_counts.values()), \
            f"Expected 4 samples per network, got {network_counts}"
    
    # Analyze batch vector
    print("\n" + "-"*70)
    print("Batch Vector Analysis:")
    print("-"*70)
    
    batch_vector = batch.batch.cpu().numpy()
    unique_batch_ids, counts = np.unique(batch_vector, return_counts=True)
    
    print(f"Unique batch IDs: {len(unique_batch_ids)}")
    print(f"Nodes per graph: {counts[:5]}... (showing first 5)")
    print(f"All graphs have same node count: {len(set(counts)) == 1}")
    
    if len(set(counts)) == 1:
        nodes_per_graph = counts[0]
        print(f"Nodes per graph: {nodes_per_graph}")
        
        # Verify y shape
        expected_y_shape = (16 * nodes_per_graph, 1, 1)
        assert batch.y.shape == expected_y_shape, \
            f"Expected y shape {expected_y_shape}, got {batch.y.shape}"
        print(f"✓ y shape correct: {batch.y.shape}")
    
    # Analyze edge batching
    print("\n" + "-"*70)
    print("Edge Batching Analysis:")
    print("-"*70)
    
    num_nodes_total = batch.x.shape[0]
    num_edges_total = batch.edge_index.shape[1]
    
    print(f"Total nodes: {num_nodes_total}")
    print(f"Total edges: {num_edges_total}")
    print(f"Nodes per graph: {num_nodes_total // 16}")
    print(f"Edges per graph (avg): {num_edges_total / 16:.1f}")
    
    # Check edge_index offsets
    max_node_in_edges = batch.edge_index.max().item()
    print(f"Max node index in edge_index: {max_node_in_edges}")
    print(f"✓ Edge offsets correct: {max_node_in_edges < num_nodes_total}")
    
    print("\n" + "="*70)
    print("✅ ALL CHECKS PASSED - PyG Batching Works Correctly!")
    print("="*70)


def test_multiple_batches():
    """Test that multiple batches maintain correct structure."""
    print("\n" + "="*70)
    print("TEST: Multiple Batches Structure")
    print("="*70)
    
    dataset = WRADataset(
        root=str(_REPO_ROOT / "data" / "wra"),
        dataset_names=['N_50_density_102.0_seed_42_Pmax_dBm_10.0_rmin_0.5'],
        split='train',
        train_fraction=0.7,
        val_fraction=0.15,
        test_fraction=0.15,
        normalize=True,
        cache_graphs=True,
        max_networks_per_split=12,
        max_samples_per_network=20,
    )
    
    sampler = NetworkGroupedBatchSampler(
        dataset=dataset,
        batch_size=16,
        samples_per_network=4,
        shuffle=False,
        seed=42
    )
    
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        follow_batch=['y'],
        exclude_keys=['info'],
        num_workers=0,
    )
    
    print(f"\nIterating through {len(loader)} batches...")
    
    for batch_idx, batch in enumerate(loader):
        network_ids = batch.network_id.tolist() if torch.is_tensor(batch.network_id) else batch.network_id
        unique_nets = len(set(network_ids))
        
        print(f"Batch {batch_idx}: {len(network_ids)} samples, "
              f"{unique_nets} unique networks, "
              f"y shape: {batch.y.shape}")
        
        # Verify grouping
        network_counts = Counter(network_ids)
        if len(network_ids) == 16:
            assert all(count == 4 for count in network_counts.values()), \
                f"Batch {batch_idx}: Incorrect grouping {network_counts}"
    
    print("\n✅ All batches have correct structure!")


if __name__ == "__main__":
    print("\n" + "#"*70)
    print("# WRA NetworkGroupedBatchSampler Integration Tests")
    print("#"*70)
    
    try:
        test_wra_batch_attributes()
        test_multiple_batches()
        
        print("\n" + "#"*70)
        print("# All Integration Tests Passed! ✅")
        print("#"*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
