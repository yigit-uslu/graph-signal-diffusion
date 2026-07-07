import torch
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from torch_geometric.utils import dense_to_sparse, to_dense_adj
import networkx as nx

#
# NB: ``utils_graph`` historically mutated matplotlib rcParams at import
# time.  We lazy-import ``draw_stocks_graph_by_correlation`` inside the
# one function that uses it so that merely importing this module does
# *not* pollute global rcParams.  See Phase 2 of the plot-style cleanup.


def compute_covariance_matrix(target: torch.Tensor, method: str = "sample") -> torch.Tensor:
    """
    Compute the covariance matrix of closing prices.
    
    Args:
        target: Tensor of shape (num_stocks, num_timestamps)
        method: "sample" for sample covariance or "population" for population covariance
        
    Returns:
        Covariance matrix of shape (num_stocks, num_stocks)
    """
    if method == "sample":
        # Sample covariance (divide by n-1)
        cov_matrix = torch.cov(target)

    elif method == "population":
        # Population covariance (divide by n)
        mean_prices = target.mean(dim=1, keepdim=True)
        centered_prices = target - mean_prices
        cov_matrix = torch.mm(centered_prices, centered_prices.t()) / target.shape[1]

    else:
        raise ValueError("method must be 'sample' or 'population'")
    
    return cov_matrix


def compute_correlation_matrix(target: torch.Tensor) -> torch.Tensor:
    """
    Compute the correlation matrix of closing prices.
    
    Args:
        target: Tensor of shape (num_stocks, num_timestamps)

    Returns:
        Correlation matrix of shape (num_stocks, num_stocks)
    """
    return torch.corrcoef(target)


def covariance_to_edge_index_weight(
    covariance_matrix: torch.Tensor,
    threshold: float = 0.0,
    use_absolute: bool = True,
    remove_self_loops: bool = True,
    normalize: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a covariance matrix to edge_index and edge_weight format for PyTorch Geometric.
    
    Args:
        covariance_matrix: Square covariance matrix of shape (num_stocks, num_stocks)
        threshold: Minimum absolute value for an edge to be included
        use_absolute: Whether to use absolute values of covariance
        remove_self_loops: Whether to remove self-loops (diagonal entries)
        normalize: Whether to normalize edge weights to [0, 1]
        
    Returns:
        edge_index: Tensor of shape (2, num_edges) containing source and target node indices
        edge_weight: Tensor of shape (num_edges,) containing edge weights
    """
    # Make a copy to avoid modifying the original
    adj_matrix = covariance_matrix.clone()
    
    # Remove self-loops if requested
    if remove_self_loops:
        adj_matrix.fill_diagonal_(0.0)

    # Fill in NaN values with 0
    adj_matrix = torch.nan_to_num(adj_matrix, nan=0.0)
    
    # Use absolute values if requested
    if use_absolute:
        adj_matrix = torch.abs(adj_matrix)
    
    # Apply threshold
    adj_matrix = adj_matrix * (adj_matrix > threshold)
    
    # Normalize if requested
    if normalize and adj_matrix.max() > 0:
        adj_matrix = adj_matrix / adj_matrix.max()
    
    # Convert to sparse format
    edge_index, edge_weight = dense_to_sparse(adj_matrix)
    
    return edge_index, edge_weight


def target_to_graph(
    target: torch.Tensor,
    method: str = "covariance",
    cov_method: str = "sample",
    threshold: float = 0.0,
    use_absolute: bool = True,
    remove_self_loops: bool = True,
    normalize: bool = True,
    save_path: Optional[str] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create edge_index and edge_weight from target using covariance or correlation.

    Args:
        target: Tensor of shape (num_stocks, num_timestamps)
        method: "covariance" or "correlation"
        cov_method: "sample" or "population" (only used when method="covariance")
        threshold: Minimum absolute value for an edge to be included
        use_absolute: Whether to use absolute values
        remove_self_loops: Whether to remove self-loops
        normalize: Whether to normalize edge weights to [0, 1]
        save_path: Optional path to save the graph visualization
    Returns:
        edge_index: Tensor of shape (2, num_edges)
        edge_weight: Tensor of shape (num_edges,)
    """
    if method == "covariance":
        matrix = compute_covariance_matrix(target, method=cov_method)
    elif method == "correlation":
        matrix = compute_correlation_matrix(target)
        # print("Computed correlation matrix: ", matrix)
    else:
        raise ValueError("method must be 'covariance' or 'correlation'")
    
    edge_index, edge_weight = covariance_to_edge_index_weight(
        matrix, 
        threshold=threshold,
        use_absolute=use_absolute,
        remove_self_loops=remove_self_loops,
        normalize=normalize
    )

    if save_path is not None:
        from .utils_graph import draw_stocks_graph_by_correlation  # lazy
        adj = to_dense_adj(edge_index, edge_attr=edge_weight, max_num_nodes=matrix.shape[0]).squeeze(0)
        G = nx.from_numpy_array(adj.numpy())
        draw_stocks_graph_by_correlation(G, save_path=save_path)

    return edge_index, edge_weight


def pandas_closing_prices_to_graph(
    values_path: str,
    method: str = "covariance",
    cov_method: str = "sample", 
    threshold: float = 0.0,
    use_absolute: bool = True,
    remove_self_loops: bool = True,
    normalize: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create edge_index and edge_weight from a CSV file containing stock data.
    
    Args:
        values_path: Path to CSV file with stock data (should have 'Symbol', 'Date', 'Close' columns)
        method: "covariance" or "correlation"
        cov_method: "sample" or "population" (only used when method="covariance")
        threshold: Minimum absolute value for an edge to be included
        use_absolute: Whether to use absolute values
        remove_self_loops: Whether to remove self-loops
        normalize: Whether to normalize edge weights to [0, 1]
        
    Returns:
        edge_index: Tensor of shape (2, num_edges)
        edge_weight: Tensor of shape (num_edges,)
    """
    # Load and process the data
    values = pd.read_csv(values_path).set_index(['Symbol', 'Date'])
    
    # Extract closing prices and reshape to (num_stocks, num_timestamps)
    num_stocks = len(values.index.get_level_values('Symbol').unique())
    closing_prices = torch.tensor(
        values[["Close"]].to_numpy().reshape((num_stocks, -1)), 
        dtype=torch.float32
    )
    
    return target_to_graph(
        closing_prices,
        method=method,
        cov_method=cov_method,
        threshold=threshold,
        use_absolute=use_absolute,
        remove_self_loops=remove_self_loops,
        normalize=normalize
    )


# Example usage and demonstration functions
def example_usage():
    """
    Example of how to use the covariance graph functions.
    """
    # Example with synthetic data
    num_stocks, num_timestamps = 5, 100
    
    # Create some correlated closing prices
    torch.manual_seed(42)
    base_prices = torch.randn(num_stocks, num_timestamps).cumsum(dim=1) + 100
    
    # Add some correlation structure
    correlation_factor = torch.tensor([[1.0, 0.8, 0.3, 0.1, 0.2],
                                      [0.8, 1.0, 0.4, 0.2, 0.1],
                                      [0.3, 0.4, 1.0, 0.6, 0.3],
                                      [0.1, 0.2, 0.6, 1.0, 0.5],
                                      [0.2, 0.1, 0.3, 0.5, 1.0]])
    
    # Apply correlation structure (simplified)
    closing_prices = torch.mm(correlation_factor, base_prices)
    
    print("Closing prices shape:", closing_prices.shape)
    
    # Method 1: Using covariance
    edge_index_cov, edge_weight_cov = target_to_graph(
        closing_prices, 
        method="covariance",
        threshold=0.1,
        normalize=True
    )
    
    print("\nCovariance-based graph:")
    print("Edge index shape:", edge_index_cov.shape)
    print("Edge weight shape:", edge_weight_cov.shape)
    print("Number of edges:", edge_index_cov.shape[1])
    
    # Method 2: Using correlation
    edge_index_corr, edge_weight_corr = target_to_graph(
        closing_prices,
        method="correlation", 
        threshold=0.3,
        normalize=True
    )
    
    print("\nCorrelation-based graph:")
    print("Edge index shape:", edge_index_corr.shape)
    print("Edge weight shape:", edge_weight_corr.shape)
    print("Number of edges:", edge_index_corr.shape[1])
    
    return edge_index_cov, edge_weight_cov, edge_index_corr, edge_weight_corr


if __name__ == "__main__":
    example_usage()
