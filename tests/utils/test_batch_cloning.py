"""
Test batch cloning utilities for multi-sample generation.
"""

import pytest
import torch
from torch_geometric.data import Data, Batch

from graph_signal_diffusion.utils.batch_cloning import (
    clone_batch_graphs,
    reshape_generated_samples,
    repeat_real_samples,
)


def test_clone_single_graph():
    """Test cloning a single graph."""
    # Create a simple graph
    N = 100
    data = Data(
        x=torch.randn(N, 32),
        edge_index=torch.randint(0, N, (2, 500)),
        edge_weight=torch.rand(500),
    )
    
    # Clone 3 times
    cloned = clone_batch_graphs(data, n_clones=3)
    
    assert cloned.num_graphs == 3
    assert cloned.num_nodes == N * 3
    assert cloned.num_edges == 500 * 3
    
    # Check edge_index offsets
    assert cloned.edge_index.min() >= 0
    assert cloned.edge_index.max() < N * 3
    
    print("✓ Clone single graph: PASSED")


def test_clone_batched_graphs():
    """Test cloning a batch of graphs."""
    # Create batch of 4 graphs
    N = 100
    graphs = []
    for i in range(4):
        data = Data(
            x=torch.randn(N, 32),
            edge_index=torch.randint(0, N, (2, 500)),
            edge_weight=torch.rand(500),
        )
        graphs.append(data)
    
    batch = Batch.from_data_list(graphs)
    
    # Clone each graph 5 times
    cloned = clone_batch_graphs(batch, n_clones=5)
    
    assert cloned.num_graphs == 4 * 5  # 20 graphs total
    assert cloned.num_nodes == N * 4 * 5
    assert cloned.num_edges == 500 * 4 * 5
    
    # Check edge_index validity
    assert cloned.edge_index.min() >= 0
    assert cloned.edge_index.max() < N * 4 * 5
    
    print("✓ Clone batched graphs: PASSED")


def test_reshape_generated_samples():
    """Test reshaping generated samples."""
    B, n, T, N, F = 4, 5, 10, 100, 1
    
    # Simulate generated samples
    samples = torch.randn(B * n, T, N, F)
    
    # Reshape
    reshaped = reshape_generated_samples(samples, n_samples_per_input=n)
    
    assert reshaped.shape == (B, n, T, N, F)
    
    # Check data integrity
    for i in range(B):
        for j in range(n):
            idx = i * n + j
            assert torch.allclose(samples[idx], reshaped[i, j])
    
    print("✓ Reshape generated samples: PASSED")


def test_repeat_real_samples():
    """Test repeating real samples."""
    B, T, N, F = 4, 10, 100, 1
    n = 5
    
    # Create real samples
    real = torch.randn(B, T, N, F)
    
    # Repeat
    repeated = repeat_real_samples(real, n_samples_per_input=n)
    
    assert repeated.shape == (B * n, T, N, F)
    
    # Check that samples are correctly repeated
    for i in range(B):
        for j in range(n):
            idx = i * n + j
            assert torch.allclose(repeated[idx], real[i])
    
    print("✓ Repeat real samples: PASSED")


def test_edge_index_offsets():
    """Test that edge indices are correctly offset after cloning."""
    N = 10
    
    # Simple chain graph: 0→1→2→...→9
    edge_index = torch.tensor([[i, i+1] for i in range(N-1)]).t()
    
    data = Data(
        x=torch.randn(N, 8),
        edge_index=edge_index,
    )
    
    # Clone 3 times
    cloned = clone_batch_graphs(data, n_clones=3)
    
    # Check that edge indices are in correct ranges
    # Graph 0: edges should reference nodes 0-9
    # Graph 1: edges should reference nodes 10-19
    # Graph 2: edges should reference nodes 20-29
    
    graph_list = cloned.to_data_list()
    assert len(graph_list) == 3
    
    for i, graph in enumerate(graph_list):
        expected_min = i * N
        expected_max = (i + 1) * N - 1
        
        assert graph.edge_index.min() >= expected_min
        assert graph.edge_index.max() <= expected_max
        
        # Check that it's still a chain
        assert graph.edge_index.shape[1] == N - 1
    
    print("✓ Edge index offsets: PASSED")


def test_with_edge_weights():
    """Test that edge weights are preserved after cloning."""
    N = 100
    edge_weight = torch.rand(500)
    
    data = Data(
        x=torch.randn(N, 32),
        edge_index=torch.randint(0, N, (2, 500)),
        edge_weight=edge_weight,
    )
    
    # Clone
    cloned = clone_batch_graphs(data, n_clones=3)
    
    # Extract graphs
    graphs = cloned.to_data_list()
    
    # Check that edge weights are preserved in each clone
    for graph in graphs:
        assert torch.allclose(graph.edge_weight, edge_weight)
    
    print("✓ Edge weights preserved: PASSED")


def test_integration_with_diffusion_shape():
    """Test integration with typical diffusion model shapes."""
    # Setup
    B, T, N, F = 4, 10, 100, 1
    n_samples_per_input = 5
    
    # Create batched graphs
    graphs = []
    for i in range(B):
        data = Data(
            x=torch.randn(N, 32),
            edge_index=torch.randint(0, N, (2, 500)),
            edge_weight=torch.rand(500),
        )
        graphs.append(data)
    
    batch = Batch.from_data_list(graphs)
    
    # Clone for multi-sample generation
    cloned_batch = clone_batch_graphs(batch, n_clones=n_samples_per_input)
    
    # Check batch size matches expected
    assert cloned_batch.num_graphs == B * n_samples_per_input
    
    # Simulate generated samples
    generated = torch.randn(B * n_samples_per_input, T, N, F)
    
    # Reshape for evaluation
    reshaped = reshape_generated_samples(generated, n_samples_per_input)
    assert reshaped.shape == (B, n_samples_per_input, T, N, F)
    
    # Compute mean across samples
    mean_samples = reshaped.mean(dim=1)
    assert mean_samples.shape == (B, T, N, F)
    
    print("✓ Integration with diffusion: PASSED")


if __name__ == "__main__":
    print("="*60)
    print("Testing batch cloning utilities")
    print("="*60)
    
    test_clone_single_graph()
    test_clone_batched_graphs()
    test_reshape_generated_samples()
    test_repeat_real_samples()
    test_edge_index_offsets()
    test_with_edge_weights()
    test_integration_with_diffusion_shape()
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)
