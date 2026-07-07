"""
Graph construction utilities for wireless resource allocation.

This module builds PyTorch Geometric graphs from wireless channel data for
use with GNN-based power allocation policies.
"""

import numpy as np
import torch
from torch_geometric.data import Data
from typing import Dict, Tuple, Optional


def _spectral_normalize_weights(
    edge_index: np.ndarray,
    edge_weights: np.ndarray,
    num_nodes: int,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Scale edge weights so the weighted adjacency spectral radius is 1.

    Parameters
    ----------
    edge_index : np.ndarray
        Edge list of shape (E, 2) with (source, target) node indices.
    edge_weights : np.ndarray
        Edge weights of shape (E,).
    num_nodes : int
        Number of nodes in graph.
    eps : float, optional
        Numerical tolerance to avoid division by zero.

    Returns
    -------
    np.ndarray
        Spectrally normalized edge weights. If the graph has zero spectral
        radius, returns input weights unchanged.
    """
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    src = edge_index[:, 0].astype(np.int64)
    dst = edge_index[:, 1].astype(np.int64)
    np.add.at(adjacency, (src, dst), edge_weights.astype(np.float64))

    eigenvalues = np.linalg.eigvals(adjacency)
    spectral_radius = float(np.max(np.abs(eigenvalues))) if eigenvalues.size > 0 else 0.0
    if spectral_radius <= eps:
        return edge_weights

    return edge_weights / spectral_radius


def build_interference_graph(
    H_ls: np.ndarray,
    associations: np.ndarray,
    top_k: int = 10,
    min_degree: int = 1,
    normalize_method: str = "log",
    self_loop_weight_scale: float = 1.0,
    spectral_normalize_edge_weights: bool = False,
) -> Data:
    """
    Build PyTorch Geometric graph from wireless channel data.
    
    Graph structure:
    - Nodes: Receivers (one node per receiver)
    - Self-loops: Weighted by direct channel gain from paired transmitter
    - Interference edges: Weighted by channel gain from other transmitters
    - Node features: [direct_signal_strength, total_interference_potential]
    
    Parameters
    ----------
    H_ls : np.ndarray
        Large-scale channel power gains, shape (m, n) where m = # transmitters, n = # receivers
        H_ls[i, j] = channel gain from TX_i to RX_j
    associations : np.ndarray
        Binary association matrix, shape (m, n)
        associations[i, j] = 1 if TX_i is paired with RX_j
    top_k : int, optional
        Keep top-K strongest edges per node (default: 10)
    min_degree : int, optional
        Ensure at least this many edges per node (default: 1)
    normalize_method : str, optional
        Method to normalize edge weights: 'log', 'log1p', or 'none' (default: 'log')
    self_loop_weight_scale : float, optional
        Scaling factor for self-loop weights (default: 1.0)
    spectral_normalize_edge_weights : bool, optional
        If True, scale (possibly already normalized) edge weights so the
        weighted adjacency has spectral radius 1 (default: False)
    
    Returns
    -------
    graph : Data
        PyTorch Geometric graph with:
        - x: Node features (n, 2) - [direct_signal, total_interference]
        - edge_index: Edge connectivity (2, E)
        - edge_weight: Normalized edge weights (E,)
        - num_nodes: Number of nodes (n)
    
    Examples
    --------
    >>> H_ls = np.random.rand(10, 10)  # 10 transmitters, 10 receivers
    >>> associations = np.eye(10)  # One-to-one pairing
    >>> graph = build_interference_graph(H_ls, associations, top_k=5)
    >>> print(graph.x.shape, graph.edge_index.shape)
    (10, 2), (2, E)
    """
    m, n = H_ls.shape
    assert associations.shape == (m, n), f"Associations shape {associations.shape} != H_ls shape {(m, n)}"
    
    # Find which transmitter is paired with each receiver
    # tx_assoc[j] = index of transmitter paired with receiver j
    tx_assoc = np.argmax(associations, axis=0)  # (n,)
    
    # Verify one-to-one pairing
    if not np.all(np.sum(associations, axis=0) == 1):
        raise ValueError("Each receiver must be paired with exactly one transmitter")
    
    # ============= Build Edge List =============
    
    edge_list = []  # (source_rx, target_rx, weight)
    
    # 1. Add self-loops: receiver i receives from its paired transmitter
    for rx_idx in range(n):
        tx_idx = tx_assoc[rx_idx]
        direct_gain = H_ls[tx_idx, rx_idx]
        
        # Self-loop: rx_idx -> rx_idx with weight = direct channel gain
        edge_list.append((rx_idx, rx_idx, direct_gain * self_loop_weight_scale))
    
    # 2. Add interference edges: receiver j receives interference from transmitter paired with receiver i
    for rx_i in range(n):
        for rx_j in range(n):
            if rx_i == rx_j:
                continue  # Skip self (already handled by self-loops)
            
            # Transmitter paired with rx_i interferes with rx_j
            tx_i = tx_assoc[rx_i]
            interference_gain = H_ls[tx_i, rx_j]
            
            # Edge from rx_i to rx_j: "TX paired with rx_i interferes at rx_j"
            edge_list.append((rx_i, rx_j, interference_gain))
    
    edge_list = np.array(edge_list)  # (E, 3)
    
    # ============= Sparsification =============
    
    # Apply top-K sparsification per target node
    sparse_edge_list = []
    
    for target_rx in range(n):
        # Get all edges pointing to this target receiver
        incoming_edges = edge_list[edge_list[:, 1] == target_rx]
        
        # Separate self-loop from other edges
        self_loop_mask = incoming_edges[:, 0] == target_rx
        self_loops = incoming_edges[self_loop_mask]
        other_edges = incoming_edges[~self_loop_mask]
        
        # Always keep self-loop
        sparse_edge_list.extend(self_loops.tolist())
        
        # Keep top-K strongest interference edges (or at least min_degree-1 edges)
        k = max(top_k, min_degree - 1)  # -1 because self-loop already added
        
        if len(other_edges) > k:
            # Sort by weight (descending) and keep top-K
            sorted_indices = np.argsort(other_edges[:, 2])[::-1]
            top_k_edges = other_edges[sorted_indices[:k]]
            sparse_edge_list.extend(top_k_edges.tolist())
        else:
            # Keep all edges if fewer than K
            sparse_edge_list.extend(other_edges.tolist())
    
    sparse_edge_list = np.array(sparse_edge_list)  # (E_sparse, 3)
    
    # ============= Normalize Edge Weights =============
    
    edge_weights = sparse_edge_list[:, 2]
    
    # Compute scaling factor based on maximum channel gain (L∞ normalization)
    # This brings the strongest channel to a reference value for numerical stability
    if normalize_method == "log1p":
        # Scale so max channel gain = 1.0
        # This makes log10(1 + max_gain * scale_factor) = log10(1 + 1) = log10(2) ≈ 0.301
        max_gain = np.max(H_ls)
        scale_factor = 1.0 / (max_gain + 1e-12)  # Avoid division by zero
        # print(f"  Channel scaling: max_gain={max_gain:.6e}, scale_factor={scale_factor:.6e}")
    else:
        scale_factor = 1.0
    
    if normalize_method == "log":
        # Log-normalization: log10(gain + eps)
        edge_weights_normalized = np.log10(edge_weights + 1e-12)
    elif normalize_method == "log1p":
        # Log1p-normalization: log10(1 + scaled_gain)
        edge_weights_normalized = np.log10(1 + edge_weights * scale_factor)
    elif normalize_method == "none":
        edge_weights_normalized = edge_weights
    else:
        raise ValueError(f"Unknown normalize_method: {normalize_method}")

    if spectral_normalize_edge_weights:
        edge_weights_normalized = _spectral_normalize_weights(
            edge_index=sparse_edge_list[:, :2],
            edge_weights=edge_weights_normalized,
            num_nodes=n,
        )
    
    # ============= Compute Node Features =============
    
    node_features = np.zeros((n, 2))
    
    for rx_idx in range(n):
        tx_idx = tx_assoc[rx_idx]
        
        # Feature 1: Direct signal strength (linear scale)
        direct_signal = H_ls[tx_idx, rx_idx]
        
        # Feature 2: Total interference potential (sum of all other TXs' gains to this RX)
        interference_mask = np.ones(m, dtype=bool)
        interference_mask[tx_idx] = False
        total_interference = np.sum(H_ls[interference_mask, rx_idx])
        
        node_features[rx_idx, 0] = direct_signal
        node_features[rx_idx, 1] = total_interference
    
    # Normalize node features (use log1p if edge weights use log1p, otherwise log)
    if normalize_method == "log1p":
        # Apply same scaling as edge weights for consistency
        node_features_normalized = np.log10(1 + node_features * scale_factor)
    else:
        # Original log normalization for backwards compatibility
        node_features_normalized = np.log10(node_features + 1e-12)
    
    # ============= Build PyTorch Geometric Data Object =============
    
    edge_index = torch.tensor(sparse_edge_list[:, :2].T, dtype=torch.long)  # (2, E)
    edge_weight = torch.tensor(edge_weights_normalized, dtype=torch.float32)  # (E,)
    x = torch.tensor(node_features_normalized, dtype=torch.float32)  # (n, 2)
    
    graph = Data(
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
        num_nodes=n,
    )
    
    return graph


def extract_large_scale_gains(
    channel_realization: Dict[str, np.ndarray]
) -> np.ndarray:
    """
    Extract large-scale channel gains from channel realization.
    
    Parameters
    ----------
    channel_realization : dict
        Dictionary from WirelessChannel.sample_realization() containing:
        - 'H': Full time-varying channel gains (m, n, T)
        - 'H_l': Large-scale gains (m, n)
        - ...
    
    Returns
    -------
    H_ls : np.ndarray
        Large-scale channel power gains (m, n)
    """
    if 'H_l' in channel_realization:
        return channel_realization['H_l']
    else:
        raise KeyError("Channel realization must contain 'H_l' (large-scale gains)")


def build_graph_from_channel(
    channel_realization: Dict[str, np.ndarray],
    top_k: int = 10,
    min_degree: int = 1,
    normalize_method: str = "log",
    spectral_normalize_edge_weights: bool = False,
) -> Data:
    """
    Convenience function to build graph directly from channel realization.
    
    Parameters
    ----------
    channel_realization : dict
        Output from WirelessChannel.sample_realization()
    top_k : int, optional
        Keep top-K strongest edges per node (default: 10)
    min_degree : int, optional
        Ensure at least this many edges per node (default: 1)
    normalize_method : str, optional
        Method to normalize edge weights (default: 'log')
    spectral_normalize_edge_weights : bool, optional
        If True, additionally spectral-normalize edge weights so adjacency
        spectral radius is 1 (default: False)
    
    Returns
    -------
    graph : Data
        PyTorch Geometric graph
    """
    H_ls = extract_large_scale_gains(channel_realization)
    associations = channel_realization['associations']
    
    return build_interference_graph(
        H_ls=H_ls,
        associations=associations,
        top_k=top_k,
        min_degree=min_degree,
        normalize_method=normalize_method,
        spectral_normalize_edge_weights=spectral_normalize_edge_weights,
    )


def get_transmitter_association_map(
    associations: np.ndarray
) -> np.ndarray:
    """
    Get mapping from receiver index to transmitter index.
    
    Parameters
    ----------
    associations : np.ndarray
        Binary association matrix, shape (m, n)
    
    Returns
    -------
    tx_assoc : np.ndarray
        Transmitter indices for each receiver, shape (n,)
        tx_assoc[j] = index of transmitter paired with receiver j
    """
    return np.argmax(associations, axis=0)
