"""Visualization tests for graph pooling components on 2D lattice graphs."""

import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

# Import pooling modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from graph_signal_diffusion.models.components.pooling import StridedGraphMaxPool, LearnableGraphPool

# Create output directory for visualizations
OUTPUT_DIR = Path(__file__).parent / "plot_pooling_visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)


def save_plot_with_archive(fig: plt.Figure, output_path: Path, dpi: int = 300):
    """
    Save a matplotlib figure, archiving any existing file with the same name.
    
    If a file with the same name exists, it will be moved to an 'archives/'
    subfolder with a timestamp appended to avoid overwriting.
    
    Args:
        fig: Matplotlib figure to save
        output_path: Path where the figure should be saved
        dpi: DPI resolution for saving (default: 300)
    """
    output_path = Path(output_path)
    
    # Check if file already exists
    if output_path.exists():
        # Create archives subdirectory
        archives_dir = output_path.parent / "archives"
        archives_dir.mkdir(exist_ok=True)
        
        # Generate timestamp for unique identifier
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create archived filename with timestamp
        stem = output_path.stem  # filename without extension
        suffix = output_path.suffix  # extension (e.g., .pdf)
        archived_name = f"{stem}_{timestamp}{suffix}"
        archived_path = archives_dir / archived_name
        
        # Move existing file to archives
        shutil.move(str(output_path), str(archived_path))
        print(f"    Archived existing file to: {archived_path.relative_to(output_path.parent.parent)}")
    
    # Save the new figure
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')


def create_2d_lattice_graph(N: int = 32, B: int = 1, T: int = 1, F: int = 1) -> Tuple[Data, nx.Graph]:
    """
    Create a 2D lattice graph (like an image grid) with node indices as features.
    
    Args:
        N: Grid size (N x N lattice)
        B: Batch size (default: 1)
        T: Time dimension (default: 1)
        F: Feature dimension (default: 1)
    
    Returns:
        data: PyG Data object with edge_index and x (B, T, N*N, F)
        G: NetworkX graph for visualization
    
    Example:
        >>> data, G = create_2d_lattice_graph(N=32)
        >>> # data.x: (1, 1, 1024, 1) - node features with indices
        >>> # data.edge_index: (2, E) - grid connectivity
    """
    num_nodes = N * N
    
    # Create edge list for 2D lattice (4-connected grid)
    edges = []
    
    # Helper to convert 2D coordinates to node index
    def coord_to_idx(i, j):
        return i * N + j
    
    # Add edges (4-connectivity: up, down, left, right)
    for i in range(N):
        for j in range(N):
            node_idx = coord_to_idx(i, j)
            
            # Right neighbor
            if j < N - 1:
                edges.append([node_idx, coord_to_idx(i, j + 1)])
            
            # Down neighbor
            if i < N - 1:
                edges.append([node_idx, coord_to_idx(i + 1, j)])
    
    # Convert to edge_index format (2, E)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    # Make edges bidirectional
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    # Create node features: shape (B, T, N*N, F)
    # Each node's feature is its index (for visualization)
    node_indices = torch.arange(num_nodes, dtype=torch.float32).reshape(1, 1, num_nodes, 1)
    x = node_indices.expand(B, T, num_nodes, F)
    
    # Create PyG Data object
    # Note: PyG expects 2D node features, but we'll keep our 4D format for consistency
    data = Data(edge_index=edge_index)
    data.x = x
    data.num_nodes = num_nodes
    data.grid_size = N
    
    # Create NetworkX graph for visualization
    G = to_networkx(Data(edge_index=edge_index, num_nodes=num_nodes), to_undirected=True)
    
    return data, G


def create_strided_mask(N: int, stride: int, B: int = 1) -> torch.Tensor:
    """
    Create an active mask that samples every stride-th node in both directions.
    
    This creates a regular grid pattern that downsamples by a factor of stride².
    
    Args:
        N: Grid size (N x N lattice)
        stride: Stride/downsampling factor (sample every stride-th node)
        B: Batch size (default: 1)
    
    Returns:
        active_mask: Binary mask (B, N*N) with nodes at positions (i, j)
                    where i % stride == 0 and j % stride == 0 set to True
    
    Example:
        >>> mask = create_strided_mask(N=8, stride=2)
        >>> # Selects nodes at (0,0), (0,2), (0,4), ... (2,0), (2,2), ...
        >>> # Total: (8/2)² = 16 nodes out of 64
    """
    num_nodes = N * N
    active_mask = torch.zeros(B, num_nodes, dtype=torch.bool)
    
    for i in range(N):
        for j in range(N):
            if i % stride == 0 and j % stride == 0:
                node_idx = i * N + j
                active_mask[:, node_idx] = True
    
    return active_mask


def visualize_lattice_graph(
    G: nx.Graph,
    N: int,
    node_values: Optional[torch.Tensor] = None,
    active_mask: Optional[torch.Tensor] = None,
    title: str = "2D Lattice Graph",
    figsize: Tuple[int, int] = (10, 10),
    cmap: str = 'viridis',
    node_size: int = 100,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Visualize a 2D lattice graph with node values and active mask.
    
    Args:
        G: NetworkX graph
        N: Grid size (N x N)
        node_values: Node values to display (N*N,) or None for indices
        active_mask: Binary mask indicating active nodes (N*N,)
        title: Plot title
        figsize: Figure size (only used if ax is None)
        cmap: Colormap for node values
        node_size: Size of nodes
        ax: Optional matplotlib axis to plot on (if None, creates new figure)
    
    Returns:
        fig: Matplotlib figure
    """
    num_nodes = N * N
    
    # Create grid layout for nodes
    pos = {}
    for i in range(N):
        for j in range(N):
            node_idx = i * N + j
            pos[node_idx] = (j, N - 1 - i)  # (x, y) with origin at top-left
    
    # Prepare node values
    if node_values is None:
        node_values = torch.arange(num_nodes, dtype=torch.float32)
    else:
        if node_values.dim() > 1:
            node_values = node_values.flatten()
        node_values = node_values.cpu()
    
    # Prepare active mask
    if active_mask is not None:
        if active_mask.dim() > 1:
            active_mask = active_mask.flatten()
        active_mask = active_mask.cpu().bool()
    else:
        active_mask = torch.ones(num_nodes, dtype=torch.bool)
    
    # Create figure if ax not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.2, width=0.5)
    
    # Separate active and inactive nodes
    active_nodes = [i for i in range(num_nodes) if active_mask[i]]
    inactive_nodes = [i for i in range(num_nodes) if not active_mask[i]]
    
    # Draw active nodes with colors based on values
    if active_nodes:
        active_values = node_values[active_nodes]
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=active_nodes,
            node_color=active_values.numpy(),
            node_size=node_size,
            cmap=cmap,
            ax=ax,
            vmin=node_values.min().item(),
            vmax=node_values.max().item(),
        )
    
    # Draw inactive nodes in gray
    if inactive_nodes:
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=inactive_nodes,
            node_color='lightgray',
            node_size=node_size * 0.5,
            ax=ax,
        )
    
    # Add node labels (only for small grids)
    if N <= 32:
        # Show actual node values (or indices if values are just indices)
        labels = {i: f"{node_values[i]:.0f}" for i in range(num_nodes)}
        nx.draw_networkx_labels(G, pos, labels, font_size=int(128/N), ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    ax.set_aspect('equal')
    
    # Add colorbar
    if active_nodes:
        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=plt.Normalize(vmin=node_values.min().item(), vmax=node_values.max().item())
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Node Value', rotation=270, labelpad=20)
    
    plt.tight_layout()
    
    return fig


def test_create_lattice_graph(N: int = 16):
    """Test creation of 2D lattice graph."""
    data, G = create_2d_lattice_graph(N=N)
    
    # Verify dimensions
    assert data.x.shape == (1, 1, N*N, 1), \
        f"Expected shape (1, 1, {N*N}, 1), got {data.x.shape}"
    
    # Verify edge connectivity (4-connected grid should have ~2*N*(N-1)*2 edges)
    # Each internal node has 4 neighbors, edge nodes have 2-3
    expected_edges = 2 * (2 * N * (N - 1))  # Bidirectional
    assert data.edge_index.shape[1] == expected_edges, \
        f"Expected {expected_edges} edges, got {data.edge_index.shape[1]}"
    
    # Verify NetworkX graph
    assert G.number_of_nodes() == N * N
    assert G.number_of_edges() == expected_edges // 2  # Undirected
    
    print(f"✓ Created {N}x{N} lattice graph:")
    print(f"  - Nodes: {G.number_of_nodes()}")
    print(f"  - Edges: {G.number_of_edges()}")
    print(f"  - Feature shape: {data.x.shape}")
    print(f"  - Node values range: [{data.x.min().item():.0f}, {data.x.max().item():.0f}]")


def test_visualize_lattice_graph(N: int = 16):
    """Test visualization of 2D lattice graph."""
    # Create a small lattice for visualization
    data, G = create_2d_lattice_graph(N=N)
    
    # Extract node values (indices)
    node_values = data.x[0, 0, :, 0]  # (N*N,)
    
    # Visualize full graph
    fig = visualize_lattice_graph(
        G, N,
        node_values=node_values,
        title=f"{N}x{N} Lattice Graph - All Nodes Active",
        figsize=(8, 8),
    )
    
    output_path = OUTPUT_DIR / f'test_{N}_lattice_full.pdf'
    save_plot_with_archive(fig, output_path)
    plt.close()
    print(f"✓ Saved visualization: {output_path}")
    
    # Test with active mask (checkerboard pattern)
    active_mask = torch.zeros(N * N, dtype=torch.bool)
    for i in range(N):
        for j in range(N):
            if (i + j) % 2 == 0:
                active_mask[i * N + j] = True
    
    fig = visualize_lattice_graph(
        G, N,
        node_values=node_values,
        active_mask=active_mask,
        title=f"{N}x{N} Lattice Graph - Checkerboard Active Mask",
        figsize=(8, 8),
    )
    
    output_path = OUTPUT_DIR / f'test_{N}_lattice_checkerboard.pdf'
    save_plot_with_archive(fig, output_path)
    plt.close()
    print(f"✓ Saved visualization: {output_path}")
    
    # Test with strided mask (sample every 2nd node in each direction)
    stride = 2
    strided_mask = create_strided_mask(N, stride)[0]  # (N*N,)
    num_active = strided_mask.sum().item()
    expected_active = (N // stride) ** 2
    
    print(f"  Strided mask (d={stride}): {num_active}/{N*N} nodes active (expected ~{expected_active})")
    
    fig = visualize_lattice_graph(
        G, N,
        node_values=node_values,
        active_mask=strided_mask,
        title=f"{N}x{N} Lattice Graph - Strided Mask (d={stride}, downsample by {stride}²={stride**2}x)",
        figsize=(8, 8),
    )
    
    output_path = OUTPUT_DIR / f'test_{N}_lattice_strided_d{stride}.pdf'
    save_plot_with_archive(fig, output_path)
    plt.close()
    print(f"✓ Saved visualization: {output_path}")
    
    # Test with larger stride
    stride = 3
    strided_mask = create_strided_mask(N, stride)[0]
    num_active = strided_mask.sum().item()
    expected_active = (N // stride) ** 2
    
    print(f"  Strided mask (d={stride}): {num_active}/{N*N} nodes active (expected ~{expected_active})")
    
    fig = visualize_lattice_graph(
        G, N,
        node_values=node_values,
        active_mask=strided_mask,
        title=f"{N}x{N} Lattice Graph - Strided Mask (d={stride}, downsample by {stride}²={stride**2}x)",
        figsize=(8, 8),
    )
    
    output_path = OUTPUT_DIR / f'test_{N}_lattice_strided_d{stride}.pdf'
    save_plot_with_archive(fig, output_path)
    plt.close()
    print(f"✓ Saved visualization: {output_path}")


    # Test with larger stride
    stride = 4
    strided_mask = create_strided_mask(N, stride)[0]
    num_active = strided_mask.sum().item()
    expected_active = (N // stride) ** 2
    
    print(f"  Strided mask (d={stride}): {num_active}/{N*N} nodes active (expected ~{expected_active})")
    
    fig = visualize_lattice_graph(
        G, N,
        node_values=node_values,
        active_mask=strided_mask,
        title=f"{N}x{N} Lattice Graph - Strided Mask (d={stride}, downsample by {stride}²={stride**2}x)",
        figsize=(8, 8),
    )
    
    output_path = OUTPUT_DIR / f'test_{N}_lattice_strided_d{stride}.pdf'
    save_plot_with_archive(fig, output_path)
    plt.close()
    print(f"✓ Saved visualization: {output_path}")


def test_strided_mask_properties(N: int = 16):
    """Test properties of strided mask."""
    
    print("Testing strided mask properties:")
    for stride in [1, 2, 3, 4, 8]:
        mask = create_strided_mask(N, stride)[0]
        num_active = mask.sum().item()
        # expected = (N // stride) ** 2
        expected = np.ceil((N / stride)) ** 2  # Adjust for non-divisible cases
        downsampling_factor = (N * N) / num_active if num_active > 0 else float('inf')
        
        print(f"  stride={stride}: {num_active}/{N*N} active "
              f"(expected {expected}, downsampling factor: {downsampling_factor:.1f}x)")
        
        assert num_active == expected, \
            f"Expected {expected} active nodes, got {num_active}"
    
    print("✓ Strided mask properties verified")


def test_strided_graph_max_pool(N: int = 16):
    """Test StridedGraphMaxPool on 2D lattice graph."""
    print(f"\nTesting StridedGraphMaxPool on {N}x{N} lattice:")
    
    # Create lattice graph
    data, G = create_2d_lattice_graph(N=N, B=1, T=1, F=1)
    edge_index = data.edge_index
    
    # Initial node features (indices)
    x = data.x  # (1, 1, N*N, 1)
    node_values = x[0, 0, :, 0]  # (N*N,)
    
    # Create initial strided mask (d=2, downsamples by 4x)
    d = 2
    initial_mask = create_strided_mask(N, stride=d, B=1)  # (1, N*N)
    num_initial_active = initial_mask.sum().item()
    
    print(f"  Initial active mask (d={d}): {num_initial_active}/{N*N} nodes")
    
    # Visualize initial state
    fig = visualize_lattice_graph(
        G, N,
        node_values=node_values,
        active_mask=initial_mask[0],
        title=f"{N}x{N} Lattice - Initial Active Mask (d={d}, {num_initial_active} nodes)",
        figsize=(10, 10),
    )
    output_path = OUTPUT_DIR / f'strided_maxpool_N{N}_initial_d{d}.pdf'
    save_plot_with_archive(fig, output_path)
    plt.close()
    print(f"  ✓ Saved initial state: {output_path}")
    
    # Apply StridedGraphMaxPool
    gamma = 2  # Downsampling factor (reduces active nodes by γ)
    K = 2  # Number of neighborhood scales
    # With stride_input=d=2 (sparse input), pooling aggregates {0, 2, 4}-hop
    
    pool = StridedGraphMaxPool(gamma=gamma, K=K, stride_input=d, selection_method='stride')
    
    print(f"  Applying StridedGraphMaxPool (γ={gamma}, K={K})...")
    x_pooled, new_active_mask, selected_indices, _ = pool(
        x, edge_index, active_mask=initial_mask
    )
    
    num_pooled_active = new_active_mask.sum().item()
    pooling_factor = num_initial_active / num_pooled_active if num_pooled_active > 0 else float('inf')
    
    print(f"  After pooling: {num_pooled_active}/{N*N} nodes active")
    print(f"  Pooling factor: {pooling_factor:.2f}x (from {num_initial_active} to {num_pooled_active})")
    
    # Extract pooled node values
    pooled_values = x_pooled[0, 0, :, 0]  # (N*N,)
    
    # Debug: Check some specific node values
    print(f"\n  Debug: Checking specific nodes after pooling:")
    for node_idx in [0, 2, 48, 96]:
        i, j = node_idx // N, node_idx % N
        original_val = node_values[node_idx].item()
        pooled_val = pooled_values[node_idx].item()
        is_active = new_active_mask[0, node_idx].item()
        print(f"    Node {node_idx} at ({i},{j}): "
              f"original={original_val:.0f}, pooled={pooled_val:.1f}, active={is_active}")
    
    # Visualize pooled result
    fig = visualize_lattice_graph(
        G, N,
        node_values=pooled_values,
        active_mask=new_active_mask[0],
        title=f"{N}x{N} Lattice - After StridedMaxPool (γ={gamma}, K={K})\n"
              f"Active: {num_pooled_active} nodes ({pooling_factor:.1f}x downsample from initial)",
        figsize=(10, 10),
    )
    output_path = OUTPUT_DIR / f'strided_maxpool_N{N}_pooled_g{gamma}_K{K}.pdf'
    save_plot_with_archive(fig, output_path)
    plt.close()
    print(f"  ✓ Saved pooled state: {output_path}")
    
    # Create side-by-side comparison
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Left: Initial state
    visualize_lattice_graph(
        G, N,
        node_values=node_values,
        active_mask=initial_mask[0],
        title=f"Initial: {num_initial_active} active nodes",
        ax=axes[0],
    )
    
    # Right: Pooled state
    visualize_lattice_graph(
        G, N,
        node_values=pooled_values,
        active_mask=new_active_mask[0],
        title=f"After Pooling: {num_pooled_active} active nodes",
        ax=axes[1],
    )
    
    plt.suptitle(
        f"StridedGraphMaxPool on {N}x{N} Lattice (γ={gamma}, K={K})",
        fontsize=16,
        fontweight='bold',
        y=0.98
    )
    
    output_path = OUTPUT_DIR / f'strided_maxpool_N{N}_comparison_g{gamma}_K{K}.pdf'
    save_plot_with_archive(fig, output_path)
    plt.close()
    print(f"  ✓ Saved comparison: {output_path}")
    
    print(f"✓ StridedGraphMaxPool test completed")


def test_learnable_graph_pool(
        N: int = 24,
        pooling_ratio: float = 0.5 # downsample by 2x
    ):
    """Test LearnableGraphPool on 2D lattice graph."""
    print(f"\nTesting LearnableGraphPool on {N}x{N} lattice:")
    
    # Create lattice graph
    data, G = create_2d_lattice_graph(N=N, B=1, T=1, F=1)
    edge_index = data.edge_index
    
    # Initial node features (indices)
    x = data.x  # (1, 1, N*N, 1)
    node_values = x[0, 0, :, 0]  # (N*N,)
    
    # Start with all nodes active
    initial_mask = torch.ones(1, N * N, dtype=torch.bool)
    num_initial_active = initial_mask.sum().item()
    
    print(f"  Initial state: {num_initial_active}/{N*N} nodes active (all nodes)")
    
    # Visualize initial state
    fig = visualize_lattice_graph(
        G, N,
        node_values=node_values,
        active_mask=initial_mask[0],
        title=f"{N}x{N} Lattice - Initial State (All Nodes Active)",
        figsize=(10, 10),
    )
    output_path = OUTPUT_DIR / f'learnable_pool_N{N}_initial.pdf'
    save_plot_with_archive(fig, output_path)
    plt.close()
    print(f"  ✓ Saved initial state: {output_path}")
    
    # Create LearnableGraphPool
    in_channels = 1  # Single feature channel (node indices)
    
    pool = LearnableGraphPool(
        in_channels=in_channels,
        pooling_ratio=pooling_ratio,
        hidden_channels=32,
        num_gnn_layers=1, # 2
        K=2,
        selection_mode='soft',  # Use soft selection for training
        temperature=1.0,
        cond_dim=None,  # No conditioning
    )
    
    # Set to eval mode for deterministic behavior
    pool.eval()
    
    print(f"  Applying LearnableGraphPool (ratio={pooling_ratio}, mode=soft)...")
    
    # Apply pooling
    with torch.no_grad():
        x_pooled, new_active_mask, selected_indices, scores = pool(
            x, edge_index, active_mask=initial_mask
        )
    
    num_pooled_active = new_active_mask.sum().item()
    pooling_factor = num_initial_active / num_pooled_active if num_pooled_active > 0 else float('inf')
    
    print(f"  After pooling: {num_pooled_active}/{N*N} nodes active")
    print(f"  Pooling factor: {pooling_factor:.2f}x (from {num_initial_active} to {num_pooled_active})")
    print(f"  Target: {pooling_ratio * num_initial_active:.0f} nodes")
    
    # Extract pooled node values and scores
    pooled_values = x_pooled[0, 0, :, 0]  # (N*N,)
    score_values = scores[0]  # (N*N,)
    
    # Debug: Check some specific node values and scores
    print(f"\n  Debug: Checking specific nodes after pooling:")
    for node_idx in [0, N//2, N*N//2, N*N-1]:
        i, j = node_idx // N, node_idx % N
        original_val = node_values[node_idx].item()
        pooled_val = pooled_values[node_idx].item()
        score_val = score_values[node_idx].item()
        is_active = new_active_mask[0, node_idx].item()
        print(f"    Node {node_idx} at ({i},{j}): "
              f"original={original_val:.0f}, pooled={pooled_val:.2f}, "
              f"score={score_val:.3f}, active={is_active}")
    
    # Visualize selection scores
    fig = visualize_lattice_graph(
        G, N,
        node_values=score_values,
        active_mask=torch.ones(N*N, dtype=torch.bool),  # Show all nodes for scores
        title=f"{N}x{N} Lattice - Selection Scores from Pooling GNN\n"
              f"Higher scores = more likely to be selected",
        figsize=(10, 10),
        cmap='RdYlGn',  # Red (low) to Green (high) colormap
    )
    output_path = OUTPUT_DIR / f'learnable_pool_N{N}_scores_ratio{pooling_ratio}.pdf'
    save_plot_with_archive(fig, output_path)
    plt.close()
    print(f"  ✓ Saved selection scores: {output_path}")
    
    # Visualize pooled result
    fig = visualize_lattice_graph(
        G, N,
        node_values=pooled_values,
        active_mask=new_active_mask[0],
        title=f"{N}x{N} Lattice - After LearnableGraphPool\n"
              f"Active: {num_pooled_active} nodes ({pooling_factor:.1f}x downsample)",
        figsize=(10, 10),
    )
    output_path = OUTPUT_DIR / f'learnable_pool_N{N}_pooled_ratio{pooling_ratio}.pdf'
    save_plot_with_archive(fig, output_path)
    plt.close()
    print(f"  ✓ Saved pooled state: {output_path}")
    
    # Create three-panel comparison
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    
    # Left: Initial state
    visualize_lattice_graph(
        G, N,
        node_values=node_values,
        active_mask=initial_mask[0],
        title=f"Initial: {num_initial_active} active nodes",
        ax=axes[0],
    )
    
    # Middle: Selection scores
    visualize_lattice_graph(
        G, N,
        node_values=score_values,
        active_mask=torch.ones(N*N, dtype=torch.bool),
        title=f"Selection Scores\n(higher = more important)",
        cmap='RdYlGn',
        ax=axes[1],
    )
    
    # Right: Pooled state
    visualize_lattice_graph(
        G, N,
        node_values=pooled_values,
        active_mask=new_active_mask[0],
        title=f"After Pooling: {num_pooled_active} active nodes",
        ax=axes[2],
    )
    
    plt.suptitle(
        f"LearnableGraphPool on {N}x{N} Lattice (ratio={pooling_ratio})",
        fontsize=16,
        fontweight='bold',
        y=0.98
    )
    
    output_path = OUTPUT_DIR / f'learnable_pool_N{N}_comparison_ratio{pooling_ratio}.pdf'
    save_plot_with_archive(fig, output_path)
    plt.close()
    print(f"  ✓ Saved comparison: {output_path}")
    
    print(f"✓ LearnableGraphPool test completed")


def test_strided_graph_max_pool_learned(
        N: int = 24,
        gamma: int = 2,  # Downsampling factor
        K: int = 2,  # Number of scales in pooling (0, stride_input, 2×stride_input-hop)
        shuffle_features: bool = False,  # Whether to shuffle node features
        random_seed: int = 42,  # Random seed for reproducibility
    ):
    """Test StridedGraphMaxPool with learned selection on 2D lattice graph.
    
    Args:
        N: Grid size (N x N lattice)
        gamma: Downsampling factor (reduces active nodes by γ)
        K: Number of neighborhood scales
        shuffle_features: If True, randomly shuffle node features instead of using sequential indices
        random_seed: Random seed for shuffling (if shuffle_features=True)
    """
    setup_str = "shuffled" if shuffle_features else "sequential"
    print(f"\nTesting StridedGraphMaxPool (learned) on {N}x{N} lattice ({setup_str} features):")
    
    # Create lattice graph
    data, G = create_2d_lattice_graph(N=N, B=1, T=1, F=1)
    edge_index = data.edge_index
    
    # Initial node features (indices)
    x = data.x  # (1, 1, N*N, 1)
    node_values = x[0, 0, :, 0]  # (N*N,)
    
    # Optionally shuffle the node features
    if shuffle_features:
        torch.manual_seed(random_seed)
        shuffled_indices = torch.randperm(N * N)
        x = torch.arange(N * N, dtype=torch.float32)[shuffled_indices].reshape(1, 1, N * N, 1)
        node_values = x[0, 0, :, 0]  # (N*N,)
        print(f"  Features shuffled with seed={random_seed}")
    
    # Start with all nodes active
    initial_mask = torch.ones(1, N * N, dtype=torch.bool)
    num_initial_active = initial_mask.sum().item()
    
    print(f"  Initial state: {num_initial_active}/{N*N} nodes active (all nodes)")
    
    # Visualize initial state
    fig = visualize_lattice_graph(
        G, N,
        node_values=node_values,
        active_mask=initial_mask[0],
        title=f"{N}x{N} Lattice - Initial State ({setup_str.capitalize()} Features)",
        figsize=(10, 10),
    )
    output_path = OUTPUT_DIR / f'strided_maxpool_learned_N{N}_initial_{setup_str}.pdf'
    save_plot_with_archive(fig, output_path)
    plt.close()
    print(f"  ✓ Saved initial state: {output_path}")
    
    # Create StridedGraphMaxPool with learned selection
    in_channels = 1  # Single feature channel (node indices)
    stride_input = 1  # Input is dense (all nodes active), so stride=1
    
    pool = StridedGraphMaxPool(
        gamma=gamma,
        K=K,
        stride_input=stride_input,  # Important: pooling windows based on input stride
        selection_method='learned',
        in_channels=in_channels,
        pooling_ratio=None,  # Will default to 1/gamma = 0.5
        selector_kwargs={
            'hidden_channels': 32,
            'num_gnn_layers': 1, # 2
            'K': 2,  # For TAGConv in selection GNN
            'selection_mode': 'soft',
            'temperature': 1.0,
            'cond_dim': None,
        },
    )
    
    # Set to eval mode for deterministic behavior
    pool.eval()
    
    print(f"  Applying StridedGraphMaxPool (γ={gamma}, K={K}, learned selection)...")
    
    # Apply pooling
    with torch.no_grad():
        x_pooled, new_active_mask, selected_indices, _ = pool(
            x, edge_index, active_mask=initial_mask
        )
    
    num_pooled_active = new_active_mask.sum().item()
    pooling_factor = num_initial_active / num_pooled_active if num_pooled_active > 0 else float('inf')
    
    print(f"  After pooling: {num_pooled_active}/{N*N} nodes active")
    print(f"  Pooling factor: {pooling_factor:.2f}x (from {num_initial_active} to {num_pooled_active})")
    
    # Extract pooled node values
    pooled_values = x_pooled[0, 0, :, 0]  # (N*N,)
    
    # Get selection scores from the learned selector
    # We need to access the selector's pooling GNN
    if hasattr(pool, 'selector') and pool.selector is not None:
        with torch.no_grad():
            scores = pool.selector.pooling_gnn(
                x, edge_index, active_mask=initial_mask
            )
        score_values = scores[0]  # (N*N,)
        
        print(f"\n  Debug: Checking specific nodes after pooling:")
        for node_idx in [0, N//2, N*N//2, N*N-1]:
            i, j = node_idx // N, node_idx % N
            original_val = node_values[node_idx].item()
            pooled_val = pooled_values[node_idx].item()
            score_val = score_values[node_idx].item()
            is_active = new_active_mask[0, node_idx].item()
            print(f"    Node {node_idx} at ({i},{j}): "
                  f"original={original_val:.0f}, pooled={pooled_val:.2f}, "
                  f"score={score_val:.3f}, active={is_active}")
        
        # Visualize selection scores
        fig = visualize_lattice_graph(
            G, N,
            node_values=score_values,
            active_mask=torch.ones(N*N, dtype=torch.bool),
            title=f"{N}x{N} Lattice - Selection Scores (Strided Max-Pool, {setup_str})\n"
                  f"Pooling windows: 0-hop, {stride_input}-hop, {2*stride_input}-hop (stride_input={stride_input})",
            figsize=(10, 10),
            cmap='RdYlGn',
        )
        output_path = OUTPUT_DIR / f'strided_maxpool_learned_N{N}_scores_g{gamma}_K{K}_{setup_str}.pdf'
        save_plot_with_archive(fig, output_path)
        plt.close()
        print(f"  ✓ Saved selection scores: {output_path}")
    else:
        score_values = None
    
    # Visualize pooled result
    fig = visualize_lattice_graph(
        G, N,
        node_values=pooled_values,
        active_mask=new_active_mask[0],
        title=f"{N}x{N} Lattice - After StridedGraphMaxPool ({setup_str})\n"
              f"γ={gamma}, K={K}, Active: {num_pooled_active} nodes ({pooling_factor:.1f}x downsample)",
        figsize=(10, 10),
    )
    output_path = OUTPUT_DIR / f'strided_maxpool_learned_N{N}_pooled_g{gamma}_K{K}_{setup_str}.pdf'
    save_plot_with_archive(fig, output_path)
    plt.close()
    print(f"  ✓ Saved pooled state: {output_path}")
    
    # Create three-panel comparison
    if score_values is not None:
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        
        # Left: Initial state
        visualize_lattice_graph(
            G, N,
            node_values=node_values,
            active_mask=initial_mask[0],
            title=f"Initial: {num_initial_active} active nodes",
            ax=axes[0],
        )
        
        # Middle: Selection scores
        visualize_lattice_graph(
            G, N,
            node_values=score_values,
            active_mask=torch.ones(N*N, dtype=torch.bool),
            title=f"Selection Scores\n(windows: 0, {stride_input}, {2*stride_input}-hop)",
            cmap='RdYlGn',
            ax=axes[1],
        )
        
        # Right: Pooled state
        visualize_lattice_graph(
            G, N,
            node_values=pooled_values,
            active_mask=new_active_mask[0],
            title=f"After Pooling: {num_pooled_active} active nodes",
            ax=axes[2],
        )
        
        plt.suptitle(
            f"StridedGraphMaxPool (learned, {setup_str}) on {N}x{N} Lattice (γ={gamma}, K={K})",
            fontsize=16,
            fontweight='bold',
            y=0.98
        )
        
        output_path = OUTPUT_DIR / f'strided_maxpool_learned_N{N}_comparison_g{gamma}_K{K}_{setup_str}.pdf'
        save_plot_with_archive(fig, output_path)
        plt.close()
        print(f"  ✓ Saved comparison: {output_path}")
    
    print(f"✓ StridedGraphMaxPool (learned) test completed")


def test_strided_hop_adjacency_equivalence(N: int = 16):
    """Test that Boolean semiring and float implementations give identical results."""
    print(f"\nTesting equivalence of strided hop adjacency implementations (N={N}):")
    
    # Create lattice graph
    data, G = create_2d_lattice_graph(N=N, B=1, T=1, F=1)
    edge_index = data.edge_index
    num_nodes = N * N
    
    # Test various configurations
    test_configs = [
        (1, 1),  # γ=1, K=1 (baseline)
        (1, 3),  # γ=1, K=3 (no stride, multiple scales)
        (2, 1),  # γ=2, K=1 (2-hop only)
        (2, 2),  # γ=2, K=2 (0, 2, 4-hop)
        (3, 2),  # γ=3, K=2 (0, 3, 6-hop)
        (4, 1),  # γ=4, K=1 (4-hop only)
    ]
    
    all_passed = True
    num_warmup = 2  # Warmup iterations
    num_trials = 10  # Number of timing trials
    
    print(f"\n{'Config':<12} {'Edges':<10} {'Equal':<8} {'Boolean (ms)':<15} {'Float (ms)':<15} {'Speedup':<10}")
    print("-" * 80)
    
    for gamma, K in test_configs:
        # Create pooling objects with different implementations
        pool_bool = StridedGraphMaxPool(
            gamma=gamma, K=K, 
            selection_method='stride',
        )
        
        pool_float = StridedGraphMaxPool(
            gamma=gamma, K=K,
            selection_method='stride', 
        )
        
        # Warmup runs
        for _ in range(num_warmup):
            _ = pool_bool._compute_strided_hop_adjacency_boolean(
                edge_index, num_nodes, gamma, K
            )
            _ = pool_float._compute_strided_hop_adjacency(
                edge_index, num_nodes, gamma, K
            )
        
        # Time Boolean implementation
        start_time = time.perf_counter()
        for _ in range(num_trials):
            adj_bool = pool_bool._compute_strided_hop_adjacency_boolean(
                edge_index, num_nodes, gamma, K
            )
        bool_time = (time.perf_counter() - start_time) / num_trials * 1000  # ms
        
        # Time float implementation
        start_time = time.perf_counter()
        for _ in range(num_trials):
            adj_float = pool_float._compute_strided_hop_adjacency(
                edge_index, num_nodes, gamma, K
            )
        float_time = (time.perf_counter() - start_time) / num_trials * 1000  # ms
        
        # Convert to dense for comparison
        from torch_geometric.utils import to_dense_adj
        adj_bool_dense = to_dense_adj(adj_bool, max_num_nodes=num_nodes)[0]
        adj_float_dense = to_dense_adj(adj_float, max_num_nodes=num_nodes)[0]
        
        # Check equality
        is_equal = torch.allclose(adj_bool_dense, adj_float_dense)
        
        # Count edges
        num_edges = (adj_bool_dense > 0).sum().item()
        
        # Compute speedup
        speedup = float_time / bool_time if bool_time > 0 else float('inf')
        
        status = "✓" if is_equal else "✗"
        config_str = f"γ={gamma},K={K}"
        print(f"{config_str:<12} {num_edges:<10} {status:<8} {bool_time:>12.3f}   {float_time:>12.3f}   {speedup:>8.2f}x")
        
        if not is_equal:
            all_passed = False
            # Debug: show differences
            diff = (adj_bool_dense != adj_float_dense).sum().item()
            print(f"      WARNING: {diff} elements differ!")
    
    print("-" * 80)
    if all_passed:
        print("✓ All configurations produce identical results!")
    else:
        print("✗ Some configurations have mismatches!")
        raise AssertionError("Boolean and float implementations are not equivalent")
    
    return all_passed


# Run tests
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing 2D Lattice Graph Creation and Visualization")
    print("="*60 + "\n")

    N = 24 # Grid size for tests
    
    # test_create_lattice_graph(N)
    # print()
    # test_visualize_lattice_graph(N)
    # print()
    # test_strided_mask_properties(N)
    # print()
    # test_strided_graph_max_pool(N)
    # print()
    # test_learnable_graph_pool(N)
    # print()
    
    # Test equivalence of implementations
    test_strided_hop_adjacency_equivalence(N)
    print()
    
    # # K = 2
    K = 1
    # Test with sequential features (original)
    test_strided_graph_max_pool_learned(N, K = K, shuffle_features=False)
    print()
    
    # Test with shuffled features (stress test)
    test_strided_graph_max_pool_learned(N, K = K, shuffle_features=True, random_seed=42)
    
    print("\n" + "="*60)
    print("ALL VISUALIZATION TESTS PASSED! ✓")
    print("="*60)
