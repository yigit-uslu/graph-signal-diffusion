"""
Test to verify the correct order of batch cloning and reshaping.
"""

import torch
from torch_geometric.data import Data, Batch


def test_batch_cloning_order():
    """Test that batch cloning produces the expected order."""
    
    print("="*70)
    print("Testing Batch Cloning Order")
    print("="*70)
    
    # Create 4 graphs with unique identifiers
    B = 4
    n_samples_per_input = 3
    
    graphs = []
    for i in range(B):
        # Use node features as identifiers: all nodes in graph i have value i
        data = Data(
            x=torch.full((10, 1), fill_value=float(i)),  # All nodes have value i
            edge_index=torch.randint(0, 10, (2, 20)),
        )
        graphs.append(data)
    
    print(f"\nOriginal graphs:")
    for i, g in enumerate(graphs):
        print(f"  Graph {i}: x[0] = {g.x[0, 0].item()}")
    
    # First, create original batch to check its batch indices
    print(f"\nOriginal batch (before cloning):")
    original_batch = Batch.from_data_list(graphs)
    print(f"  num_graphs: {original_batch.num_graphs}")
    print(f"  num_nodes: {original_batch.num_nodes}")
    print(f"  batch tensor shape: {original_batch.batch.shape}")
    print(f"  batch tensor: {original_batch.batch.tolist()}")
    
    # Clone using the same method as in trainer
    data_cloned = Batch.from_data_list(
        [g for g in graphs for _ in range(n_samples_per_input)]
    )
    
    print(f"\nCloned batch: {data_cloned.num_graphs} graphs total")
    print(f"  num_nodes: {data_cloned.num_nodes}")
    print(f"  batch tensor shape: {data_cloned.batch.shape}")
    
    # Print batch indices (first 10 nodes per graph for readability)
    print(f"\nBatch indices (showing pattern):")
    nodes_per_graph = 10
    for graph_idx in range(data_cloned.num_graphs):
        start_node = graph_idx * nodes_per_graph
        end_node = start_node + 3  # Show first 3 nodes
        batch_vals = data_cloned.batch[start_node:end_node].tolist()
        node_features = data_cloned.x[start_node, 0].item()
        print(f"  Graph {graph_idx:2d}: batch[{start_node}:{end_node}] = {batch_vals}, "
              f"x[{start_node}] = {node_features:.0f}")
    
    # Unbatch to see the order
    cloned_list = data_cloned.to_data_list()
    
    print(f"\nOrder after cloning (by unbatching):")
    for idx, g in enumerate(cloned_list):
        input_id = int(g.x[0, 0].item())
        print(f"  Index {idx:2d}: Graph from input_{input_id}")
    
    # Verify the pattern
    expected_pattern = []
    for i in range(B):
        for _ in range(n_samples_per_input):
            expected_pattern.append(i)
    
    actual_pattern = [int(g.x[0, 0].item()) for g in cloned_list]
    
    print(f"\nExpected pattern: {expected_pattern}")
    print(f"Actual pattern:   {actual_pattern}")
    
    assert actual_pattern == expected_pattern, "Order mismatch!"
    
    # Verify batch indices match expected order
    print(f"\nVerifying batch indices align with cloning order:")
    expected_batch_indices = []
    for graph_idx in range(data_cloned.num_graphs):
        for _ in range(nodes_per_graph):
            expected_batch_indices.append(graph_idx)
    
    actual_batch_indices = data_cloned.batch.tolist()
    assert actual_batch_indices == expected_batch_indices, "Batch indices don't match!"
    
    # Map batch indices back to original input IDs
    print(f"\nMapping batch index to original input:")
    for graph_idx in range(data_cloned.num_graphs):
        original_input = actual_pattern[graph_idx]
        sample_num = graph_idx % n_samples_per_input if graph_idx >= n_samples_per_input else graph_idx
        # Actually compute which sample this is
        sample_count = 0
        original_input_count = -1
        for i in range(graph_idx + 1):
            if i % n_samples_per_input == 0:
                original_input_count += 1
            if i == graph_idx:
                sample_count = i - (original_input_count * n_samples_per_input)
        
        print(f"  batch_idx={graph_idx:2d} → input_{original_input}_sample_{sample_count}")
    
    print("\n✓ Cloning order is: [i0s0, i0s1, i0s2, i1s0, i1s1, i1s2, ...]")
    print("✓ Batch indices correctly increment: [0,0,...,0,1,1,...,1,2,2,...,2,...]")
    
    # Now test reshaping
    print("\n" + "="*70)
    print("Testing Reshape")
    print("="*70)
    
    # Simulate generated samples with identifiable values
    # Each "sample" will have value = input_id * 10 + sample_num
    generated = torch.zeros(B * n_samples_per_input, 5, 10, 1)  # (12, T=5, N=10, F=1)
    
    idx = 0
    for i in range(B):
        for s in range(n_samples_per_input):
            value = i * 10 + s  # e.g., input_0_sample_0 = 0, input_1_sample_2 = 12
            generated[idx] = value
            idx += 1
    
    print(f"\nGenerated samples (showing first element of each):")
    for idx in range(B * n_samples_per_input):
        print(f"  Index {idx:2d}: value = {generated[idx, 0, 0, 0].item():.0f}")
    
    # Reshape using current implementation
    B_calc = B
    reshaped = generated.view(B_calc, n_samples_per_input, 5, 10, 1)
    
    print(f"\nAfter reshape to ({B}, {n_samples_per_input}, T, N, F):")
    for i in range(B):
        for s in range(n_samples_per_input):
            value = reshaped[i, s, 0, 0, 0].item()
            expected = i * 10 + s
            print(f"  reshaped[{i}, {s}] = {value:.0f} (expected {expected})")
            assert value == expected, f"Mismatch at [{i}, {s}]!"
    
    print("\n✓ Reshaping is CORRECT!")
    print("  reshaped[i, s] correctly gives input_i_sample_s")
    
    return True


def test_alternative_cloning_order():
    """Test what would happen with a different cloning order."""
    
    print("\n" + "="*70)
    print("Testing Alternative Cloning Order (what you were worried about)")
    print("="*70)
    
    B = 4
    n = 3
    
    graphs = []
    for i in range(B):
        data = Data(
            x=torch.full((10, 1), fill_value=float(i)),
            edge_index=torch.randint(0, 10, (2, 20)),
        )
        graphs.append(data)
    
    # Alternative order: [i1s1, i2s1, i3s1, i4s1, i1s2, i2s2, ...]
    alternative_list = []
    for s in range(n):
        for i in range(B):
            alternative_list.append(graphs[i])
    
    print(f"\nAlternative order (samples-first):")
    for idx, g in enumerate(alternative_list):
        input_id = int(g.x[0, 0].item())
        print(f"  Index {idx:2d}: Graph from input_{input_id}")
    
    # Simulate generated samples with this order
    generated_alt = torch.zeros(B * n, 5, 10, 1)
    idx = 0
    for s in range(n):
        for i in range(B):
            value = i * 10 + s
            generated_alt[idx] = value
            idx += 1
    
    print(f"\nGenerated samples (showing first element of each):")
    for idx in range(B * n):
        print(f"  Index {idx:2d}: value = {generated_alt[idx, 0, 0, 0].item():.0f}")
    
    # Try naive reshape
    print(f"\nIf we naively reshape to ({B}, {n}, T, N, F):")
    reshaped_wrong = generated_alt.view(B, n, 5, 10, 1)
    for i in range(B):
        for s in range(n):
            value = reshaped_wrong[i, s, 0, 0, 0].item()
            expected = i * 10 + s
            status = "✓" if value == expected else f"✗ (expected {expected})"
            print(f"  reshaped[{i}, {s}] = {value:.0f} {status}")
    
    # Correct reshape for this order
    print(f"\nCorrect reshape for this order: ({n}, {B}, T, N, F) then transpose:")
    reshaped_correct = generated_alt.view(n, B, 5, 10, 1).permute(1, 0, 2, 3, 4)
    for i in range(B):
        for s in range(n):
            value = reshaped_correct[i, s, 0, 0, 0].item()
            expected = i * 10 + s
            print(f"  reshaped[{i}, {s}] = {value:.0f} ✓")
    
    print("\n⚠️ But this is NOT our cloning order!")


if __name__ == "__main__":
    test_batch_cloning_order()
    test_alternative_cloning_order()
    
    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("""
Our cloning code:
    [g for g in data_list for _ in range(n)]
    
Produces order:
    [i1s1, i1s2, i1s3, i2s1, i2s2, i2s3, ...]
    
Current reshape:
    view(B, n, T, N, F)  ✓ CORRECT
    
No transpose needed!
    """)
