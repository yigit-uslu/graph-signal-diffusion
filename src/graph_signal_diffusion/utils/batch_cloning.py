"""
Utility functions for cloning batched graphs for multi-sample generation.

This module provides utilities to handle generating multiple synthetic samples
per real sample when working with graph-structured data in PyTorch Geometric.
"""

import torch
from torch_geometric.data import Data, Batch
from typing import Union


def clone_batch_graphs(
    data: Union[Data, Batch],
    n_clones: int
) -> Batch:
    """
    Clone each graph in a batch n_clones times.
    
    This is useful for generating multiple samples per input graph during
    diffusion model sampling. The function properly handles edge_index offsets
    and batch indices using PyTorch Geometric's batching utilities.
    
    Args:
        data: Input graph or batch of graphs (Data or Batch object)
        n_clones: Number of clones per graph
        
    Returns:
        Batch object with each graph cloned n_clones times
        
    Example:
        >>> # Single graph with 100 nodes
        >>> data = Data(
        ...     x=torch.randn(100, 32),
        ...     edge_index=torch.randint(0, 100, (2, 500)),
        ...     edge_weight=torch.rand(500)
        ... )
        >>> 
        >>> # Generate 5 clones
        >>> cloned_data = clone_batch_graphs(data, n_clones=5)
        >>> print(cloned_data.num_graphs)  # 5
        >>> print(cloned_data.num_nodes)   # 500 (100 * 5)
        
        >>> # Batched graphs (e.g., batch_size=4)
        >>> from torch_geometric.data import Batch
        >>> batch = Batch.from_data_list([data] * 4)
        >>> print(batch.num_graphs)  # 4
        >>> 
        >>> # Clone each graph 3 times
        >>> cloned_batch = clone_batch_graphs(batch, n_clones=3)
        >>> print(cloned_batch.num_graphs)  # 12 (4 * 3)
    """
    # Convert single Data to list
    if isinstance(data, Data) and not isinstance(data, Batch):
        data_list = [data] * n_clones
    else:
        # Unbatch, repeat each graph, and collect
        data_list = data.to_data_list()
        # For each graph, create n_clones copies: [g1, g1, g1, g2, g2, g2, ...]
        data_list = [g for g in data_list for _ in range(n_clones)]
    
    # Rebatch into a single Batch object
    # PyTorch Geometric will handle edge_index offsets and batch indices
    return Batch.from_data_list(data_list)


def reshape_generated_samples(
    samples: torch.Tensor,
    n_samples_per_input: int
) -> torch.Tensor:
    """
    Reshape generated samples from (B*n, T, N, F) to (B, n, T, N, F).
    
    This is useful when you want to evaluate multiple samples per input
    or compute per-input statistics (mean, variance, etc.).
    
    Args:
        samples: Generated samples of shape (B*n_samples_per_input, T, N, F)
        n_samples_per_input: Number of samples generated per input
        
    Returns:
        Reshaped tensor of shape (B, n_samples_per_input, T, N, F)
        
    Example:
        >>> # Generated 5 samples per input for batch_size=4
        >>> samples = torch.randn(20, 10, 100, 1)  # (B*n, T, N, F)
        >>> reshaped = reshape_generated_samples(samples, n_samples_per_input=5)
        >>> print(reshaped.shape)  # torch.Size([4, 5, 10, 100, 1])
        >>> 
        >>> # Compute mean and std across samples per input
        >>> mean_samples = reshaped.mean(dim=1)  # (B, T, N, F)
        >>> std_samples = reshaped.std(dim=1)    # (B, T, N, F)
    """
    B_times_n, T, N, F = samples.shape
    B = B_times_n // n_samples_per_input
    
    # Reshape to (B, n_samples_per_input, T, N, F)
    return samples.view(B, n_samples_per_input, T, N, F)


def repeat_real_samples(
    real_samples: torch.Tensor,
    n_samples_per_input: int
) -> torch.Tensor:
    """
    Repeat real samples to match the shape of generated samples.
    
    This is useful for computing per-sample metrics when you have
    multiple generated samples per real sample.
    
    Args:
        real_samples: Real samples of shape (B, T, N, F)
        n_samples_per_input: Number of generated samples per input
        
    Returns:
        Repeated tensor of shape (B*n_samples_per_input, T, N, F)
        
    Example:
        >>> real = torch.randn(4, 10, 100, 1)  # (B, T, N, F)
        >>> repeated = repeat_real_samples(real, n_samples_per_input=5)
        >>> print(repeated.shape)  # torch.Size([20, 10, 100, 1])
        >>> 
        >>> # Now can compute MSE per sample
        >>> generated = torch.randn(20, 10, 100, 1)
        >>> mse_per_sample = ((generated - repeated) ** 2).mean(dim=(1, 2, 3))
        >>> print(mse_per_sample.shape)  # torch.Size([20])
    """
    B, T, N, F = real_samples.shape
    
    # Repeat along batch dimension: [b0, b0, b0, b1, b1, b1, ...]
    return real_samples.repeat_interleave(n_samples_per_input, dim=0)


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("Example: Cloning batched graphs for multi-sample generation")
    print("=" * 60)
    
    # Create a simple graph
    data = Data(
        x=torch.randn(100, 32),
        edge_index=torch.randint(0, 100, (2, 500)),
        edge_weight=torch.rand(500)
    )
    
    print(f"\nOriginal graph:")
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.num_edges}")
    
    # Clone 3 times
    cloned = clone_batch_graphs(data, n_clones=3)
    print(f"\nCloned batch:")
    print(f"  Num graphs: {cloned.num_graphs}")
    print(f"  Total nodes: {cloned.num_nodes}")
    print(f"  Total edges: {cloned.num_edges}")
    
    # Batch multiple graphs
    batch = Batch.from_data_list([data] * 4)
    print(f"\nOriginal batch:")
    print(f"  Num graphs: {batch.num_graphs}")
    print(f"  Total nodes: {batch.num_nodes}")
    
    # Clone each graph 5 times
    cloned_batch = clone_batch_graphs(batch, n_clones=5)
    print(f"\nCloned batch:")
    print(f"  Num graphs: {cloned_batch.num_graphs}")
    print(f"  Total nodes: {cloned_batch.num_nodes}")
    
    print("\n" + "=" * 60)
    print("Example: Reshaping generated samples")
    print("=" * 60)
    
    # Simulate generated samples
    B, n, T, N, F = 4, 5, 10, 100, 1
    samples = torch.randn(B * n, T, N, F)
    
    print(f"\nGenerated samples shape: {samples.shape}")
    
    reshaped = reshape_generated_samples(samples, n_samples_per_input=n)
    print(f"Reshaped samples shape: {reshaped.shape}")
    
    # Compute statistics
    mean_samples = reshaped.mean(dim=1)
    std_samples = reshaped.std(dim=1)
    
    print(f"\nMean across samples per input: {mean_samples.shape}")
    print(f"Std across samples per input: {std_samples.shape}")
