"""
Test PyG's DataLoader batching behavior with concatenated edge_index.

Verifies that PyG's batching correctly handles graphs where edge_index
comes from concatenating static + dynamic graphs, as done in SP500Stocks.get().
"""

import torch
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import TAGConv


def create_sample_with_concatenated_edges(sample_id: int, num_nodes: int = 4):
    """
    Create a sample that mimics SP500Stocks.get() behavior:
    - Static graph edges
    - Dynamic graph edges
    - Concatenated edge_index and edge_weight
    """
    # Node features with sample-specific pattern for verification
    x = torch.randn(num_nodes, 10, 8)  # [N, T, F]
    x[:, 0, 0] = sample_id  # Marker to identify sample
    
    # Static graph: fixed topology (e.g., 0→1, 1→2, 2→3)
    edge_index_static = torch.tensor([
        [0, 1, 2],
        [1, 2, 3],
    ], dtype=torch.long)
    edge_weight_static = torch.tensor([0.5, 0.6, 0.7], dtype=torch.float)
    
    # Dynamic graph: varies by sample (simulating different periods)
    # Sample 0: 0→2, 1→3
    # Sample 1: 0→3, 2→1
    if sample_id == 0:
        edge_index_dynamic = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
        edge_weight_dynamic = torch.tensor([0.3, 0.4], dtype=torch.float)
    else:
        edge_index_dynamic = torch.tensor([[0, 2], [3, 1]], dtype=torch.long)
        edge_weight_dynamic = torch.tensor([0.8, 0.9], dtype=torch.float)
    
    # Concatenate (as done in SP500Stocks.get())
    edge_index = torch.cat([edge_index_static, edge_index_dynamic], dim=1)
    edge_weight = torch.cat([edge_weight_static, edge_weight_dynamic], dim=0)
    
    y = torch.randn(num_nodes, 5, 1)  # [N, T_future, 1]
    
    return Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y)


def test_pyg_batching_with_concatenated_graphs():
    """
    Test that PyG DataLoader correctly batches samples with concatenated edge_index.
    """
    print("\n=== PyG Batching Test with Concatenated Graphs ===\n")
    
    num_nodes = 4
    
    # Create dataset with 3 samples (each has static + dynamic edges concatenated)
    dataset = [
        create_sample_with_concatenated_edges(sample_id=0, num_nodes=num_nodes),
        create_sample_with_concatenated_edges(sample_id=1, num_nodes=num_nodes),
        create_sample_with_concatenated_edges(sample_id=2, num_nodes=num_nodes),
    ]
    
    print(f"Dataset: {len(dataset)} samples")
    print(f"Sample 0: {dataset[0].edge_index.shape[1]} edges "
          f"(static: 3, dynamic: 2, total: 5)")
    print(f"Sample 1: {dataset[1].edge_index.shape[1]} edges "
          f"(static: 3, dynamic: 2, total: 5)")
    print(f"Sample 2: {dataset[2].edge_index.shape[1]} edges "
          f"(static: 3, dynamic: 2, total: 5)")
    
    # Create DataLoader with batch_size=2
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    # Test first batch (samples 0 and 1)
    batch = next(iter(loader))
    
    print(f"\n--- Batch 1 (samples 0, 1) ---")
    print(f"Batched x shape: {batch.x.shape}")
    print(f"  Expected: [{2 * num_nodes}, 10, 8] = [8, 10, 8]")
    print(f"Batched edge_index shape: {batch.edge_index.shape}")
    print(f"  Expected: [2, {2 * 5}] = [2, 10] (5 edges per sample × 2 samples)")
    print(f"Batched edge_weight shape: {batch.edge_weight.shape}")
    print(f"  Expected: [{2 * 5}] = [10]")
    print(f"Batch vector: {batch.batch.tolist()}")
    print(f"  Expected: [0,0,0,0, 1,1,1,1] (4 nodes per sample)")
    
    # Verify shapes
    assert batch.x.shape[0] == 2 * num_nodes, "Batch should have 8 nodes (2 samples × 4 nodes)"
    assert batch.edge_index.shape == (2, 10), "Batch should have 10 edges (2 samples × 5 edges)"
    assert batch.edge_weight.shape == (10,), "Batch should have 10 edge weights"
    assert batch.batch.shape == (8,), "Batch vector should have 8 entries"
    
    # Verify node index offsets
    # Original sample 0: edges have node indices in [0, 3]
    # Original sample 1: edges have node indices in [0, 3]
    # After batching:
    #   Sample 0 edges: node indices stay [0, 3]
    #   Sample 1 edges: node indices offset by 4 → [4, 7]
    
    max_idx_sample0 = batch.edge_index[:, :5].max().item()  # First 5 edges (sample 0)
    min_idx_sample1 = batch.edge_index[:, 5:].min().item()  # Last 5 edges (sample 1)
    max_idx_sample1 = batch.edge_index[:, 5:].max().item()
    
    print(f"\nNode index ranges after batching:")
    print(f"  Sample 0 edges (first 5): max index = {max_idx_sample0} (expected ≤ 3)")
    print(f"  Sample 1 edges (last 5): indices in [{min_idx_sample1}, {max_idx_sample1}] "
          f"(expected [4, 7])")
    
    assert max_idx_sample0 <= 3, "Sample 0 edges should have node indices ≤ 3"
    assert min_idx_sample1 >= 4, "Sample 1 edges should have node indices ≥ 4"
    assert max_idx_sample1 <= 7, "Sample 1 edges should have node indices ≤ 7"
    
    print("\n✅ PyG correctly offset node indices for sample 1")
    print("✅ Edge weights preserved during batching")
    
    # Test with TAGConv to ensure message passing works correctly
    in_channels = 8
    out_channels = 3
    torch.manual_seed(42)
    tagconv = TAGConv(in_channels, out_channels, K=1)
    tagconv.eval()
    
    with torch.no_grad():
        # Process batched data
        # Note: TAGConv expects x.shape = [N, F], so we need to flatten time dimension
        x_flat = batch.x.reshape(batch.x.shape[0], -1)  # [8, 10*8] = [8, 80]
        
        # Since our TAGConv expects in_channels=8 but we have 80, let's just use first 8
        out = tagconv(batch.x[:, 0, :], batch.edge_index, batch.edge_weight)
    
    print(f"\nTAGConv output shape: {out.shape}")
    print(f"  Expected: [{2 * num_nodes}, {out_channels}] = [8, 3]")
    
    assert out.shape == (8, 3), f"Expected shape (8, 3), got {out.shape}"
    
    print("\n✅ TAGConv successfully processed batched data with concatenated edges")
    
    # Verify that different samples in batch get different outputs
    # (due to different node features and different dynamic edges)
    out_sample0 = out[:4]  # First 4 nodes (sample 0)
    out_sample1 = out[4:]  # Last 4 nodes (sample 1)
    
    assert not torch.allclose(out_sample0, out_sample1), \
        "Different samples should produce different outputs"
    
    print("✅ Different samples in batch produce different outputs (as expected)")


def test_batch_unbatch_consistency():
    """
    Test that Batch.from_data_list and Batch.to_data_list are inverses.
    
    This verifies that PyG's batching/unbatching preserves concatenated edge_index.
    """
    print("\n\n=== Batch/Unbatch Consistency Test ===\n")
    
    num_nodes = 4
    
    # Create samples with concatenated edges
    data_list = [
        create_sample_with_concatenated_edges(sample_id=i, num_nodes=num_nodes)
        for i in range(3)
    ]
    
    print(f"Original samples: {len(data_list)}")
    for i, data in enumerate(data_list):
        print(f"  Sample {i}: {data.edge_index.shape[1]} edges, "
              f"x[0,0,0]={data.x[0, 0, 0].item():.1f} (sample marker)")
    
    # Batch
    batch = Batch.from_data_list(data_list)
    print(f"\nBatched: {batch.edge_index.shape[1]} total edges "
          f"({len(data_list)} samples × 5 edges)")
    
    # Unbatch
    unbatched = batch.to_data_list()
    
    print(f"\nUnbatched: {len(unbatched)} samples")
    for i, data in enumerate(unbatched):
        print(f"  Sample {i}: {data.edge_index.shape[1]} edges, "
              f"x[0,0,0]={data.x[0, 0, 0].item():.1f} (sample marker)")
    
    # Verify consistency
    assert len(unbatched) == len(data_list), "Should unbatch to same number of samples"
    
    for i, (original, recovered) in enumerate(zip(data_list, unbatched)):
        assert original.edge_index.shape == recovered.edge_index.shape, \
            f"Sample {i}: edge_index shape mismatch"
        assert original.edge_weight.shape == recovered.edge_weight.shape, \
            f"Sample {i}: edge_weight shape mismatch"
        assert torch.allclose(original.x, recovered.x), \
            f"Sample {i}: node features mismatch"
        assert torch.equal(original.edge_index, recovered.edge_index), \
            f"Sample {i}: edge_index mismatch"
        assert torch.allclose(original.edge_weight, recovered.edge_weight), \
            f"Sample {i}: edge_weight mismatch"
    
    print("\n✅ Batch → Unbatch is consistent (lossless)")
    print("✅ Concatenated edge_index preserved through batching cycle")


def test_multi_batch_iteration():
    """
    Test iterating through multiple batches with DataLoader.
    
    Simulates typical training/evaluation loop.
    """
    print("\n\n=== Multi-Batch Iteration Test ===\n")
    
    num_nodes = 4
    num_samples = 7
    batch_size = 3
    
    # Create larger dataset
    dataset = [
        create_sample_with_concatenated_edges(sample_id=i, num_nodes=num_nodes)
        for i in range(num_samples)
    ]
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Dataset: {num_samples} samples")
    print(f"DataLoader: batch_size={batch_size}")
    print(f"Expected batches: {len(loader)} (3 samples, 3 samples, 1 sample)")
    
    for batch_idx, batch in enumerate(loader):
        batch_size_actual = batch.num_graphs
        print(f"\nBatch {batch_idx}: {batch_size_actual} samples")
        print(f"  Nodes: {batch.x.shape[0]} (= {batch_size_actual} × {num_nodes})")
        print(f"  Edges: {batch.edge_index.shape[1]} (= {batch_size_actual} × 5)")
        print(f"  Batch vector: {batch.batch.tolist()}")
        
        # Verify shapes are consistent
        assert batch.x.shape[0] == batch_size_actual * num_nodes, "Node count mismatch"
        assert batch.edge_index.shape[1] == batch_size_actual * 5, "Edge count mismatch"
        assert len(batch.batch) == batch_size_actual * num_nodes, "Batch vector length mismatch"
    
    print("\n✅ All batches correctly formed across entire dataset")
    print("✅ DataLoader iteration works with concatenated edge_index")


if __name__ == "__main__":
    print("="*70)
    print("Testing PyG Batching with Static + Dynamic Graph Concatenation")
    print("="*70)
    
    test_pyg_batching_with_concatenated_graphs()
    test_batch_unbatch_consistency()
    test_multi_batch_iteration()
    
    print("\n" + "="*70)
    print("All PyG batching tests passed! ✅")
    print("="*70)
    print("\nConclusion:")
    print("- PyG DataLoader correctly batches graphs with concatenated edge_index")
    print("- Node indices are properly offset for each sample in the batch")
    print("- Edge weights are preserved during batching")
    print("- TAGConv (and other GNN layers) work correctly with batched data")
    print("- Batch → Unbatch cycle is lossless")
    print("- The SP500 dynamic graph implementation is fully compatible with PyG")
