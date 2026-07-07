"""Visualization tests for nested graph pooling (multi-level downsampling)."""

import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, List
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, from_networkx

# Import pooling modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from graph_signal_diffusion.models.components.pooling import StridedGraphMaxPool

# Create output directory for visualizations
OUTPUT_DIR = Path(__file__).parent / "plot_nested_pooling_visualizations"
OUTPUT_DIR.mkdir(exist_ok=True)


def save_plot_with_archive(fig, output_path: Path):
    """Save plot, archiving existing file with timestamp if it exists."""
    if output_path.exists():
        # Create archives directory if needed
        archives_dir = output_path.parent / "archives"
        archives_dir.mkdir(exist_ok=True)
        
        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archived_name = f"{output_path.stem}_{timestamp}{output_path.suffix}"
        archived_path = archives_dir / archived_name
        
        # Move existing file to archives
        shutil.move(str(output_path), str(archived_path))
        print(f"  Archived existing file to: {archived_path}")
    
    # Save new plot
    fig.savefig(output_path, dpi=150, bbox_inches='tight')


def create_2d_lattice_graph(
    N: int = 16,
    B: int = 1,
    T: int = 1,
    F: int = 1,
) -> Tuple[Data, nx.Graph]:
    """
    Create a 2D lattice graph (grid) with 4-connectivity.
    
    Args:
        N: Grid size (N×N nodes)
        B: Batch size
        T: Number of time steps
        F: Feature dimension
    
    Returns:
        data: PyG Data object with (B, T, N*N, F) features
        G: NetworkX graph for visualization
    """
    # Create N×N grid graph with 4-connectivity
    G = nx.grid_2d_graph(N, N)
    
    # Convert node labels from (i,j) tuples to integers
    mapping = {(i, j): i * N + j for i in range(N) for j in range(N)}
    G = nx.relabel_nodes(G, mapping)
    
    # Convert to PyG format
    edge_index = torch.tensor(list(G.edges())).t().contiguous()
    # Make undirected (add reverse edges)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    num_nodes = N * N
    
    # Initialize features with SHUFFLED node indices (helps stress-test pooling)
    shuffled_indices = torch.randperm(num_nodes)
    x = shuffled_indices.float().view(1, 1, num_nodes, 1)
    x = x.expand(B, T, num_nodes, F)
    
    # Create PyG Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        num_nodes=num_nodes,
    )
    
    return data, G


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
            pos[node_idx] = (j, i)  # (x, y) = (col, row)
    
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
            node_color=active_values.detach().numpy(),
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
        # Format large negative values (masked scores) as -∞
        labels = {}
        for i in range(num_nodes):
            val = node_values[i].item()
            if val < -1e8:  # Masked-out score (typically -1e9)
                labels[i] = r"$-\infty$"
            else:
                labels[i] = f"{val:.0f}"
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


def test_nested_strided_pooling(N: int = 32, gamma: int = 2, K: int = 2, num_levels: int = 3):
    """
    Test nested/hierarchical strided max pooling.
    
    Applies multiple levels of pooling sequentially to progressively
    reduce the number of active nodes. Each level uses accumulated gamma
    to account for the overall downsampling factor.
    
    Args:
        N: Grid size (N×N lattice)
        gamma: Stride factor for each pooling level
        K: Number of hop scales
        num_levels: Number of pooling levels to apply
    """
    print(f"\nTesting nested strided pooling (N={N}, γ={gamma}, K={K}, levels={num_levels}):")
    
    # Create lattice graph with shuffled features for learned selection
    data, G = create_2d_lattice_graph(N=N, B=1, T=1, F=1)
    x = data.x
    edge_index = data.edge_index
    
    B, T, num_nodes, F = x.shape
    print(f"  Initial: {num_nodes} nodes")
    
    # Track states through pooling hierarchy
    states_before = []  # State before pooling at each level
    states_after = []   # State after pooling at each level
    scores_list = []    # Selection scores at each level
    active_masks_before = []  # Active mask before pooling
    active_masks_after = []   # Active mask after pooling
    gamma_values = []
    
    # Initial state (use mean across features for visualization)
    initial_state = x[0, 0, :, :].mean(dim=-1).clone()
    initial_mask = torch.ones(num_nodes, dtype=torch.bool)
    
    # Apply pooling levels sequentially
    current_x = x.clone()
    current_active_mask = initial_mask.unsqueeze(0)  # (B, N)
    gamma_accumulated = 1
    
    for level in range(num_levels):
        # Store state BEFORE pooling
        before_state = current_x[0, 0, :, :].mean(dim=-1).clone()
        states_before.append(before_state)
        active_masks_before.append(current_active_mask[0].clone())
        
        # Accumulate gamma for this level
        gamma_accumulated *= gamma
        
        # Create pooling layer with accumulated gamma and LEARNED selection
        pool = StridedGraphMaxPool(
            gamma=gamma_accumulated,
            K=K,
            selection_method='learned',
            in_channels=F,
            pooling_ratio=1.0 / gamma,  # Downsample by gamma factor
            selector_kwargs={
                'hidden_channels': 32,
                'num_gnn_layers': 1,
                'K': 2
            },
        )
        pool.eval()  # Use inference mode for deterministic selection
        
        # Get selection scores from the selector
        # We need to access the selector's forward pass
        selector = pool.selector
        selector_scores = selector.pooling_gnn(
            current_x,
            edge_index,
            active_mask=current_active_mask,
        )
        scores_list.append(selector_scores[0].clone())
        
        # Apply pooling
        pooled_x, new_active_mask, _, _ = pool(
            current_x,
            edge_index,
            active_mask=current_active_mask,
        )
        
        # Store state AFTER pooling
        after_state = pooled_x[0, 0, :, :].mean(dim=-1).clone()
        states_after.append(after_state)
        active_masks_after.append(new_active_mask[0].clone())
        gamma_values.append(gamma_accumulated)
        
        num_active = new_active_mask.sum().item()
        print(f"  Level {level+1} (γ_acc={gamma_accumulated}): {num_active} active nodes")
        
        # Update for next iteration
        current_x = pooled_x
        current_active_mask = new_active_mask
    
    # Visualize all levels with detailed comparison plots
    # Each row = one pooling level with 3 columns: [Before, Scores, After]
    fig, axes = plt.subplots(num_levels, 3, figsize=(24, 8 * num_levels))
    if num_levels == 1:
        axes = axes.reshape(1, -1)
    
    for level in range(num_levels):
        # Left: State before pooling
        before_mask = active_masks_before[level]
        num_before = before_mask.sum().item()
        visualize_lattice_graph(
            G, N,
            node_values=states_before[level],
            active_mask=before_mask,
            title=f"Level {level+1} - Before\n{num_before} active nodes",
            ax=axes[level, 0],
        )
        
        # Middle: Selection scores
        visualize_lattice_graph(
            G, N,
            node_values=scores_list[level],
            active_mask=before_mask,  # Show scores only for active nodes
            title=f"Level {level+1} - Scores\nγ_acc={gamma_values[level]}",
            cmap='RdYlGn',
            ax=axes[level, 1],
        )
        
        # Right: State after pooling
        after_mask = active_masks_after[level]
        num_after = after_mask.sum().item()
        visualize_lattice_graph(
            G, N,
            node_values=states_after[level],
            active_mask=after_mask,
            title=f"Level {level+1} - After\n{num_after} active nodes",
            ax=axes[level, 2],
        )
    
    plt.suptitle(
        f"Nested Strided Pooling (Learned) on {N}×{N} Lattice\n"
        f"γ={gamma}, K={K} (each row shows one pooling level)",
        fontsize=14,
        fontweight='bold',
        y=0.995
    )
    
    output_path = OUTPUT_DIR / f'nested_strided_N{N}_g{gamma}_K{K}_L{num_levels}.pdf'
    save_plot_with_archive(fig, output_path)
    plt.close()
    print(f"  ✓ Saved visualization: {output_path}")
    
    print(f"✓ Nested strided pooling test completed")


def visualize_arbitrary_graph(
    G: nx.Graph,
    pos: dict,
    node_values: Optional[torch.Tensor] = None,
    active_mask: Optional[torch.Tensor] = None,
    title: str = "Graph",
    ax: Optional[plt.Axes] = None,
    cmap: str = 'viridis',
    node_size: int = 50,
    show_colorbar: bool = True,
) -> plt.Figure:
    """
    Visualize an arbitrary graph with node values and active mask.
    
    Args:
        G: NetworkX graph
        pos: Node positions (dict mapping node_id to (x, y))
        node_values: Node values to display or None
        active_mask: Binary mask indicating active nodes
        title: Plot title
        ax: Optional matplotlib axis to plot on
        cmap: Colormap for node values
        node_size: Size of nodes
        show_colorbar: Whether to show colorbar
    
    Returns:
        fig: Matplotlib figure
    """
    num_nodes = len(G.nodes())
    
    # Create figure if ax not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.get_figure()
    
    # Prepare node values
    if node_values is None:
        node_values = torch.arange(num_nodes, dtype=torch.float32)
    else:
        if node_values.dim() > 1:
            node_values = node_values.flatten()
        node_values = node_values.cpu().detach()
    
    # Prepare active mask
    if active_mask is not None:
        if active_mask.dim() > 1:
            active_mask = active_mask.flatten()
        active_mask = active_mask.cpu().bool()
    else:
        active_mask = torch.ones(num_nodes, dtype=torch.bool)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.2, width=0.5)
    
    # Separate active and inactive nodes
    active_nodes = [i for i in range(num_nodes) if i < len(active_mask) and active_mask[i]]
    inactive_nodes = [i for i in range(num_nodes) if i >= len(active_mask) or not active_mask[i]]
    
    # Draw active nodes with colors based on values
    if active_nodes:
        active_values = node_values[active_nodes]
        # Handle -inf values in scores
        active_values_display = active_values.clone()
        active_values_display[active_values < -1e8] = active_values[active_values >= -1e8].min() if (active_values >= -1e8).any() else 0
        
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=active_nodes,
            node_color=active_values_display.numpy(),
            node_size=node_size,
            cmap=cmap,
            ax=ax,
            vmin=active_values_display.min().item(),
            vmax=active_values_display.max().item(),
        )
    
    # Draw inactive nodes in gray
    if inactive_nodes:
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=inactive_nodes,
            node_color='lightgray',
            node_size=node_size * 0.3,
            ax=ax,
        )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    ax.set_aspect('equal')
    
    # Add colorbar
    if show_colorbar and active_nodes:
        vmin = active_values_display.min().item()
        vmax = active_values_display.max().item()
        if vmax > vmin:  # Only show colorbar if there's variation
            sm = plt.cm.ScalarMappable(
                cmap=cmap,
                norm=plt.Normalize(vmin=vmin, vmax=vmax)
            )
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Value', rotation=270, labelpad=15)
    
    return fig


def test_nested_pooling_arbitrary_graph(
    graph_type: str = 'barabasi_albert',
    n_nodes: int = 500,
    gamma: int = 2,
    K: int = 1,
    num_levels: int = 3,
    seed: Optional[int] = 42
):
    """
    Test nested pooling on arbitrary graphs (not regular lattices).
    
    Args:
        graph_type: Type of graph to generate ('barabasi_albert', 'erdos_renyi', 
                   'watts_strogatz', 'cora', 'citeseer')
        n_nodes: Number of nodes (for random graphs)
        gamma: Downsampling factor at each level
        K: Number of hop neighborhoods to consider
        num_levels: Number of pooling levels
        seed: Random seed for reproducibility
    """
    print(f"\nTesting nested pooling on {graph_type} graph")
    print(f"  Parameters: n_nodes={n_nodes}, γ={gamma}, K={K}, levels={num_levels}")
    
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Generate or load graph
    if graph_type == 'barabasi_albert':
        # Barabási-Albert: scale-free network
        G = nx.barabasi_albert_graph(n_nodes, m=3, seed=seed)
        print(f"  Generated Barabási-Albert graph (m=3)")
    elif graph_type == 'erdos_renyi':
        # Erdős-Rényi: random graph with fixed edge probability
        p = 6 / n_nodes  # Expected degree ~6
        G = nx.erdos_renyi_graph(n_nodes, p, seed=seed)
        print(f"  Generated Erdős-Rényi graph (p={p:.4f})")
    elif graph_type == 'watts_strogatz':
        # Watts-Strogatz: small-world network
        G = nx.watts_strogatz_graph(n_nodes, k=6, p=0.1, seed=seed)
        print(f"  Generated Watts-Strogatz graph (k=6, p=0.1)")
    elif graph_type == 'powerlaw_cluster':
        # Holme-Kim: power-law with clustering
        G = nx.powerlaw_cluster_graph(n_nodes, m=3, p=0.1, seed=seed)
        print(f"  Generated Powerlaw Cluster graph (m=3, p=0.1)")
    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")
    
    # Convert to PyG Data object
    data = from_networkx(G)
    edge_index = data.edge_index
    num_nodes = G.number_of_nodes()
    
    # Add random node features with proper shape (B, T, N, F)
    B = 1  # Batch size
    T = 1  # Time steps
    F = 1  # Single feature channel
    x = torch.randn(B, T, num_nodes, F)
    
    # Print graph statistics
    print(f"\n  Graph Statistics:")
    print(f"    Nodes: {num_nodes}")
    print(f"    Edges: {G.number_of_edges()}")
    print(f"    Avg degree: {2*G.number_of_edges()/num_nodes:.2f}")
    print(f"    Clustering: {nx.average_clustering(G):.4f}")
    if nx.is_connected(G):
        print(f"    Diameter: {nx.diameter(G)}")
        print(f"    Avg path length: {nx.average_shortest_path_length(G):.2f}")
    else:
        print(f"    Connected components: {nx.number_connected_components(G)}")
    
    # Initialize active mask (all nodes active) with shape (B, N)
    active_mask = torch.ones(B, num_nodes, dtype=torch.bool)
    
    # Track statistics at each level
    stats_per_level = []
    
    # Store states at each level for detailed visualization
    states_before = []  # State before pooling at each level
    states_after = []   # State after pooling at each level
    scores_list = []    # Selection scores at each level
    active_masks_before = []  # Active mask before pooling
    active_masks_after = []   # Active mask after pooling
    gamma_values = []
    
    # Compute statistics for initial graph
    stats = {
        'num_nodes': num_nodes,
        'num_edges': G.number_of_edges(),
        'avg_degree': 2*G.number_of_edges()/num_nodes,
        'clustering': nx.average_clustering(G),
    }
    stats_per_level.append(stats)
    
    # Apply nested pooling
    gamma_accumulated = 1
    current_x = x
    current_edge_index = edge_index
    current_active_mask = active_mask
    
    for level in range(num_levels):
        # Store state BEFORE pooling
        before_state = current_x[0, 0, :, :].mean(dim=-1).clone()
        states_before.append(before_state)
        active_masks_before.append(current_active_mask[0].clone())
        
        gamma_accumulated *= gamma
        
        print(f"\n  Level {level+1}: γ_acc = {gamma_accumulated}")
        
        # Initialize pooling layer
        pool = StridedGraphMaxPool(
            in_channels=F,
            gamma=gamma_accumulated,
            K=K,
            selection_method='learned',
            pooling_ratio=1.0 / gamma,
            selector_kwargs={
                'hidden_channels': 32,
                'num_gnn_layers': 1,
                'K': 2
            },
        )
        pool.eval()  # Use inference mode for deterministic selection
        
        # Get selection scores from the selector
        selector = pool.selector
        selector_scores = selector.pooling_gnn(
            current_x,
            current_edge_index,
            active_mask=current_active_mask,
        )
        scores_list.append(selector_scores[0].clone())
        
        # Apply pooling
        pooled_x, new_active_mask, _, _ = pool(
            current_x,
            current_edge_index,
            active_mask=current_active_mask,
        )
        
        # Store state AFTER pooling
        after_state = pooled_x[0, 0, :, :].mean(dim=-1).clone()
        states_after.append(after_state)
        active_masks_after.append(new_active_mask[0].clone())
        gamma_values.append(gamma_accumulated)
        
        # Extract subgraph for active nodes
        active_nodes = torch.where(new_active_mask[0])[0].numpy()
        G_pooled = G.subgraph(active_nodes).copy()
        
        # Compute statistics
        num_active = new_active_mask.sum().item()
        stats = {
            'num_nodes': num_active,
            'num_edges': G_pooled.number_of_edges(),
            'avg_degree': 2*G_pooled.number_of_edges()/num_active if num_active > 0 else 0,
            'clustering': nx.average_clustering(G_pooled) if num_active > 0 else 0,
        }
        stats_per_level.append(stats)
        
        print(f"    Nodes: {num_active} ({100*num_active/num_nodes:.1f}%)")
        print(f"    Edges: {stats['num_edges']}")
        print(f"    Avg degree: {stats['avg_degree']:.2f}")
        
        # Update for next level
        current_x = pooled_x
        current_active_mask = new_active_mask
    
    # Create detailed level-by-level visualization (similar to lattice version)
    print(f"\n  Creating detailed level-by-level visualization...")
    
    # Compute graph layout once for consistency
    pos = nx.spring_layout(G, seed=seed, k=1/np.sqrt(num_nodes), iterations=50)
    
    # Create figure with 3 columns per level: [Before, Scores, After]
    fig_detailed = plt.figure(figsize=(24, 8 * num_levels))
    
    for level in range(num_levels):
        # Left: State before pooling
        ax_before = plt.subplot(num_levels, 3, level * 3 + 1)
        before_mask = active_masks_before[level]
        num_before = before_mask.sum().item()
        visualize_arbitrary_graph(
            G, pos,
            node_values=states_before[level],
            active_mask=before_mask,
            title=f"Level {level+1} - Before\n{num_before} active nodes",
            ax=ax_before,
            node_size=100 if num_nodes < 200 else 50,
        )
        
        # Middle: Selection scores
        ax_scores = plt.subplot(num_levels, 3, level * 3 + 2)
        visualize_arbitrary_graph(
            G, pos,
            node_values=scores_list[level],
            active_mask=before_mask,  # Show scores only for active nodes
            title=f"Level {level+1} - Scores\nγ_acc={gamma_values[level]}",
            ax=ax_scores,
            cmap='RdYlGn',
            node_size=100 if num_nodes < 200 else 50,
        )
        
        # Right: State after pooling
        ax_after = plt.subplot(num_levels, 3, level * 3 + 3)
        after_mask = active_masks_after[level]
        num_after = after_mask.sum().item()
        visualize_arbitrary_graph(
            G, pos,
            node_values=states_after[level],
            active_mask=after_mask,
            title=f"Level {level+1} - After\n{num_after} active nodes",
            ax=ax_after,
            node_size=100 if num_nodes < 200 else 50,
        )
    
    plt.suptitle(
        f"Nested Pooling on {graph_type.replace('_', ' ').title()} Graph (Detailed)\n"
        f"n={n_nodes}, γ={gamma}, K={K}, levels={num_levels}",
        fontsize=16,
        fontweight='bold',
        y=0.995
    )
    
    output_path_detailed = OUTPUT_DIR / f'nested_arbitrary_{graph_type}_n{n_nodes}_g{gamma}_K{K}_L{num_levels}_detailed.pdf'
    save_plot_with_archive(fig_detailed, output_path_detailed)
    plt.close(fig_detailed)
    print(f"  ✓ Saved detailed visualization: {output_path_detailed}")
    
    # Visualization: Statistics summary
    print(f"\n  Creating statistics summary visualization...")
    
    # Create figure with statistics plots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Number of nodes over levels
    ax1 = fig.add_subplot(gs[0, 0])
    levels = list(range(len(stats_per_level)))
    num_nodes_list = [s['num_nodes'] for s in stats_per_level]
    ax1.plot(levels, num_nodes_list, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Level', fontsize=12)
    ax1.set_ylabel('Number of Nodes', fontsize=12)
    ax1.set_title('Node Count', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Number of edges over levels
    ax2 = fig.add_subplot(gs[0, 1])
    num_edges_list = [s['num_edges'] for s in stats_per_level]
    ax2.plot(levels, num_edges_list, 'o-', linewidth=2, markersize=8, color='orange')
    ax2.set_xlabel('Level', fontsize=12)
    ax2.set_ylabel('Number of Edges', fontsize=12)
    ax2.set_title('Edge Count', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Average degree over levels
    ax3 = fig.add_subplot(gs[0, 2])
    avg_degree_list = [s['avg_degree'] for s in stats_per_level]
    ax3.plot(levels, avg_degree_list, 'o-', linewidth=2, markersize=8, color='green')
    ax3.set_xlabel('Level', fontsize=12)
    ax3.set_ylabel('Average Degree', fontsize=12)
    ax3.set_title('Average Degree', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Clustering coefficient over levels
    ax4 = fig.add_subplot(gs[1, 0])
    clustering_list = [s['clustering'] for s in stats_per_level]
    ax4.plot(levels, clustering_list, 'o-', linewidth=2, markersize=8, color='red')
    ax4.set_xlabel('Level', fontsize=12)
    ax4.set_ylabel('Clustering Coefficient', fontsize=12)
    ax4.set_title('Clustering Coefficient', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Degree distribution at level 0
    ax5 = fig.add_subplot(gs[1, 1])
    degrees = [d for n, d in G.degree()]
    ax5.hist(degrees, bins=30, alpha=0.7, edgecolor='black')
    ax5.set_xlabel('Degree', fontsize=12)
    ax5.set_ylabel('Count', fontsize=12)
    ax5.set_title(f'Degree Distribution (Level 0)', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Degree distribution at final level
    ax6 = fig.add_subplot(gs[1, 2])
    # Get final active nodes
    final_active_nodes = torch.where(active_masks_after[-1])[0].numpy()
    if len(final_active_nodes) > 0:
        G_final = G.subgraph(final_active_nodes).copy()
        degrees = [d for n, d in G_final.degree()]
        ax6.hist(degrees, bins=30, alpha=0.7, edgecolor='black', color='purple')
    ax6.set_xlabel('Degree', fontsize=12)
    ax6.set_ylabel('Count', fontsize=12)
    ax6.set_title(f'Degree Distribution (Level {num_levels})', fontsize=14, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # Plot 7-9: Graph visualizations (initial, middle, final)
    # For small graphs (< 200 nodes), show network layout
    if num_nodes < 200:
        # Initial graph
        ax7 = fig.add_subplot(gs[2, 0])
        nx.draw_networkx(G, pos, ax=ax7, node_size=20, 
                        node_color='lightblue', with_labels=False, width=0.5)
        ax7.set_title(f'Level 0 ({num_nodes_list[0]} nodes)', fontsize=14, fontweight='bold')
        ax7.axis('off')
        
        # Middle level
        if num_levels > 1:
            ax8 = fig.add_subplot(gs[2, 1])
            mid_level = num_levels // 2
            mid_active_nodes = torch.where(active_masks_after[mid_level-1])[0].numpy()
            if len(mid_active_nodes) > 0:
                G_mid = G.subgraph(mid_active_nodes).copy()
                pos_mid = {n: pos[n] for n in G_mid.nodes()}
                nx.draw_networkx(G_mid, pos_mid, ax=ax8, node_size=20, 
                               node_color='lightgreen', with_labels=False, width=0.5)
            ax8.set_title(f'Level {mid_level} ({num_nodes_list[mid_level]} nodes)', 
                         fontsize=14, fontweight='bold')
            ax8.axis('off')
        
        # Final graph
        ax9 = fig.add_subplot(gs[2, 2])
        if len(final_active_nodes) > 0:
            pos_final = {n: pos[n] for n in G_final.nodes()}
            nx.draw_networkx(G_final, pos_final, ax=ax9, node_size=30, 
                           node_color='lightcoral', with_labels=False, width=0.5)
        ax9.set_title(f'Level {num_levels} ({num_nodes_list[-1]} nodes)', 
                     fontsize=14, fontweight='bold')
        ax9.axis('off')
    else:
        # For large graphs, show text summary
        ax7 = fig.add_subplot(gs[2, :])
        summary_text = "Graph visualizations omitted for large graphs (>200 nodes)\n\n"
        summary_text += "Summary:\n"
        for i, stats in enumerate(stats_per_level):
            summary_text += f"  Level {i}: {stats['num_nodes']} nodes, {stats['num_edges']} edges\n"
        ax7.text(0.5, 0.5, summary_text, ha='center', va='center', 
                fontsize=12, family='monospace')
        ax7.axis('off')
    
    plt.suptitle(
        f"Nested Pooling on {graph_type.replace('_', ' ').title()} Graph (Statistics)\n"
        f"n={n_nodes}, γ={gamma}, K={K}, levels={num_levels}",
        fontsize=16,
        fontweight='bold',
        y=0.995
    )
    
    output_path = OUTPUT_DIR / f'nested_arbitrary_{graph_type}_n{n_nodes}_g{gamma}_K{K}_L{num_levels}_stats.pdf'
    save_plot_with_archive(fig, output_path)
    plt.close()
    print(f"  ✓ Saved statistics visualization: {output_path}")
    
    print(f"✓ Nested pooling test on arbitrary graph completed")
    
    return stats_per_level


def visualize_node_neighborhoods(
    G: nx.Graph,
    pos: dict,
    target_node: int,
    neighborhood_edges: torch.Tensor,
    node_values: torch.Tensor,
    active_mask: torch.Tensor,
    title: str = "Node Neighborhoods",
    ax: Optional[plt.Axes] = None,
    node_size: int = 50,
) -> plt.Figure:
    """
    Visualize a specific node and its pooling neighborhoods with value-based coloring.
    
    Args:
        G: NetworkX graph
        pos: Node positions
        target_node: The node whose neighborhoods to visualize
        neighborhood_edges: Edge index (2, E) of the neighborhood structure
        node_values: Node feature values (N,) for coloring
        active_mask: Binary mask (N,) indicating active nodes
        title: Plot title
        ax: Optional matplotlib axis
        node_size: Size of nodes
    
    Returns:
        fig: Matplotlib figure
    """
    from torch_geometric.utils import to_dense_adj
    
    # Create figure if ax not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.get_figure()
    
    num_nodes = len(G.nodes())
    
    # Prepare node values
    if node_values.dim() > 1:
        node_values = node_values.flatten()
    node_values = node_values.cpu().detach().numpy()
    
    # Prepare active mask
    if active_mask.dim() > 1:
        active_mask = active_mask.flatten()
    active_mask = active_mask.cpu().bool().numpy()
    
    # Get adjacency matrix
    adj = to_dense_adj(neighborhood_edges, max_num_nodes=num_nodes)[0].cpu().numpy()
    
    # Find nodes in neighborhood (connected to target node)
    neighbors = np.where(adj[target_node, :] > 0)[0]
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.1, width=0.3)
    
    # Highlight edges in the pooling neighborhood
    neighborhood_edge_list = []
    for i in range(neighborhood_edges.shape[1]):
        src, dst = neighborhood_edges[0, i].item(), neighborhood_edges[1, i].item()
        if G.has_edge(src, dst):
            neighborhood_edge_list.append((src, dst))
    
    if neighborhood_edge_list:
        nx.draw_networkx_edges(G, pos, edgelist=neighborhood_edge_list, 
                             ax=ax, edge_color='red', alpha=0.3, width=1.5)
    
    # Separate nodes by type
    target_nodes = [target_node]
    
    # Separate neighborhood nodes into active and inactive
    active_neighborhood_nodes = [n for n in neighbors if n != target_node and active_mask[n]]
    inactive_neighborhood_nodes = [n for n in neighbors if n != target_node and not active_mask[n]]
    other_nodes = [n for n in range(num_nodes) if n not in neighbors and n != target_node]
    
    # Draw other nodes (gray)
    if other_nodes:
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=other_nodes,
            node_color='lightgray',
            node_size=node_size * 0.3,
            ax=ax,
            alpha=0.3,
        )
    
    # Draw INACTIVE neighborhood nodes with hatching/border (value-based color but distinguishable)
    if inactive_neighborhood_nodes:
        inactive_values = [node_values[n] for n in inactive_neighborhood_nodes]
        
        # Get value range for colormap (from all neighborhood nodes)
        all_neighborhood_values = ([node_values[target_node]] + 
                                   [node_values[n] for n in active_neighborhood_nodes] +
                                   inactive_values)
        vmin, vmax = min(all_neighborhood_values), max(all_neighborhood_values)
        
        # Draw inactive nodes with value-based color but with thick red border
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=inactive_neighborhood_nodes,
            node_color=inactive_values,
            node_size=node_size,
            cmap='viridis',
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            alpha=0.5,  # More transparent
            edgecolors='red',  # Red border to indicate inactive
            linewidths=2.5,  # Thick border
        )
    
    # Draw ACTIVE neighborhood nodes with value-based coloring
    if active_neighborhood_nodes:
        active_values = [node_values[n] for n in active_neighborhood_nodes]
        
        # Get value range for colormap
        all_neighborhood_values = ([node_values[target_node]] + 
                                   active_values +
                                   ([node_values[n] for n in inactive_neighborhood_nodes] if inactive_neighborhood_nodes else []))
        vmin, vmax = min(all_neighborhood_values), max(all_neighborhood_values)
        
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=active_neighborhood_nodes,
            node_color=active_values,
            node_size=node_size,
            cmap='viridis',
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            edgecolors='black',  # Black border for active nodes
            linewidths=1.0,
        )
    
    # Draw target node with its value (larger, star shape)
    # Set vmin/vmax based on what we have
    if active_neighborhood_nodes or inactive_neighborhood_nodes:
        all_neighborhood_values = ([node_values[target_node]] + 
                                   [node_values[n] for n in active_neighborhood_nodes] +
                                   [node_values[n] for n in inactive_neighborhood_nodes])
        vmin, vmax = min(all_neighborhood_values), max(all_neighborhood_values)
    else:
        vmin = vmax = node_values[target_node]
    
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=target_nodes,
        node_color=[node_values[target_node]],
        node_size=node_size * 3,
        ax=ax,
        node_shape='*',
        cmap='viridis',
        vmin=vmin,
        vmax=vmax,
        edgecolors='black',
        linewidths=2.0,
    )
    
    # Add colorbar for node values
    if active_neighborhood_nodes or inactive_neighborhood_nodes or target_nodes:
        sm = plt.cm.ScalarMappable(
            cmap='viridis',
            norm=plt.Normalize(vmin=vmin, vmax=vmax)
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Node Value', rotation=270, labelpad=15)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')
    ax.set_aspect('equal')
    
    # Add text annotation with active/inactive counts and max values
    num_neighbors = len(neighbors)
    num_active = len(active_neighborhood_nodes)
    num_inactive = len(inactive_neighborhood_nodes)
    target_value = node_values[target_node]
    
    # Compute max values over neighborhoods
    max_active = max([node_values[n] for n in active_neighborhood_nodes]) if active_neighborhood_nodes else float('-inf')
    max_inactive = max([node_values[n] for n in inactive_neighborhood_nodes]) if inactive_neighborhood_nodes else float('-inf')
    
    annotation_text = (f'Target node: {target_node}\n'
                      f'Value: {target_value:.2f}\n'
                      f'Neighborhood: {num_neighbors} nodes\n'
                      f'  Active: {num_active}')
    if num_active > 0:
        annotation_text += f' (max: {max_active:.2f})'
    annotation_text += f'\n  Inactive: {num_inactive}'
    if num_inactive > 0:
        annotation_text += f' (max: {max_inactive:.2f})'
    
    ax.text(0.02, 0.98, annotation_text,
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Add legend for node types
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gray', 
               markersize=15, label='Target Node', markeredgecolor='black', markeredgewidth=1.5),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=10, label='Active Neighbor', markeredgecolor='black', markeredgewidth=1),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
               markersize=10, label='Inactive Neighbor', markeredgecolor='red', 
               markeredgewidth=2.5, alpha=0.5),
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    return fig


def test_neighborhood_visualization_arbitrary_graph(
    graph_type: str = 'barabasi_albert',
    n_nodes: int = 150,
    gamma: int = 2,
    K: int = 1,
    num_levels: int = 3,
    seed: Optional[int] = 42
):
    """
    Visualize how pooling neighborhoods evolve for a specific node across levels.
    
    Tracks a single node that remains active throughout all pooling levels and
    visualizes its γ_accumulated-strided neighborhoods at each level.
    """
    print(f"\nVisualizing neighborhoods on {graph_type} graph")
    print(f"  Parameters: n_nodes={n_nodes}, γ={gamma}, K={K}, levels={num_levels}")
    
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Generate graph
    if graph_type == 'barabasi_albert':
        G = nx.barabasi_albert_graph(n_nodes, m=3, seed=seed)
    elif graph_type == 'erdos_renyi':
        p = 6 / n_nodes
        G = nx.erdos_renyi_graph(n_nodes, p, seed=seed)
    elif graph_type == 'watts_strogatz':
        G = nx.watts_strogatz_graph(n_nodes, k=6, p=0.1, seed=seed)
    elif graph_type == 'powerlaw_cluster':
        G = nx.powerlaw_cluster_graph(n_nodes, m=3, p=0.1, seed=seed)
    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")
    
    print(f"  Generated {graph_type} graph with {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Convert to PyG
    data = from_networkx(G)
    edge_index = data.edge_index
    num_nodes = G.number_of_nodes()
    
    # Create features
    B, T, F = 1, 1, 1
    x = torch.randn(B, T, num_nodes, F)
    active_mask = torch.ones(B, num_nodes, dtype=torch.bool)
    
    # Compute graph layout once
    pos = nx.spring_layout(G, seed=seed, k=1/np.sqrt(num_nodes), iterations=50)
    
    # Track neighborhoods and active nodes for each level
    neighborhoods_per_level = []
    active_masks_per_level = []
    node_values_per_level = []  # Track node values at each level
    gamma_values = []
    stride_input_values = []  # Track stride_input for each level
    
    gamma_accumulated = 1
    stride_input = 1  # Start with dense signal
    current_x = x
    current_active_mask = active_mask
    
    # Apply pooling and collect neighborhoods
    for level in range(num_levels):
        # Store node values and active mask BEFORE pooling
        node_vals = current_x[0, 0, :, :].mean(dim=-1).clone()
        node_values_per_level.append(node_vals)
        
        stride_input_values.append(stride_input)  # Track stride_input BEFORE pooling
        gamma_accumulated *= gamma
        
        print(f"  Level {level+1}: stride_input = {stride_input}, γ_acc = {gamma_accumulated}")
        
        pool = StridedGraphMaxPool(
            in_channels=F,
            gamma=gamma,  # Local downsampling ratio
            K=K,
            stride_input=stride_input,  # Input signal spacing
            selection_method='learned',
            pooling_ratio=1.0 / gamma,
            selector_kwargs={
                'hidden_channels': 32,
                'num_gnn_layers': 1,
                'K': 2
            },
        )
        pool.eval()
        
        # Apply pooling WITH neighborhood return
        pooled_x, new_active_mask, _, neighborhoods = pool(
            current_x,
            edge_index,
            active_mask=current_active_mask,
            return_neighborhoods=True,
        )
        
        neighborhoods_per_level.append(neighborhoods)
        active_masks_per_level.append(current_active_mask[0].clone())  # Store BEFORE pooling mask
        gamma_values.append(gamma_accumulated)
        
        num_active = new_active_mask.sum().item()
        print(f"    Active nodes: {num_active}")
        
        # Update stride_input for next level (output stride becomes next input stride)
        stride_input *= gamma
        
        current_x = pooled_x
        current_active_mask = new_active_mask
    
    # Find a node that remains active throughout all levels
    # Start with all nodes active
    always_active = torch.ones(num_nodes, dtype=torch.bool)
    for mask in active_masks_per_level:
        always_active &= mask
    
    always_active_nodes = torch.where(always_active)[0].numpy()
    
    if len(always_active_nodes) == 0:
        print("  ⚠ No node remained active through all levels. Using node active in most levels.")
        # Find node active in most levels
        active_counts = torch.zeros(num_nodes)
        for mask in active_masks_per_level:
            active_counts += mask.float()
        target_node = active_counts.argmax().item()
    else:
        # Pick a node with high degree (more interesting neighborhoods)
        degrees = dict(G.degree())
        target_node = max(always_active_nodes, key=lambda n: degrees.get(n, 0))
    
    print(f"  Selected target node: {target_node} (degree: {G.degree(target_node)})")
    
    # Create visualization
    fig = plt.figure(figsize=(24, 8 * num_levels))
    
    for level in range(num_levels):
        ax = plt.subplot(num_levels, 1, level + 1)
        
        visualize_node_neighborhoods(
            G, pos,
            target_node=target_node,
            neighborhood_edges=neighborhoods_per_level[level],
            node_values=node_values_per_level[level],
            active_mask=active_masks_per_level[level],
            title=f"Level {level+1}: $\\gamma_{{{level}}}$={stride_input_values[level]}, $\\gamma_{{{level+1}}}$={gamma_values[level]}, K={K}\n"
                  f"Pooling neighborhoods: {', '.join([str(k*stride_input_values[level]) for k in range(K+1)])}-hops",
            ax=ax,
            node_size=100 if num_nodes < 200 else 50,
        )
    
    plt.suptitle(
        f"Strided Pooling Neighborhoods for Node {target_node}\n"
        f"{graph_type.replace('_', ' ').title()} Graph (n={n_nodes}, γ={gamma}, K={K})",
        fontsize=16,
        fontweight='bold',
        y=0.995
    )
    
    output_path = OUTPUT_DIR / f'neighborhoods_{graph_type}_n{n_nodes}_g{gamma}_K{K}_L{num_levels}_node{target_node}.pdf'
    save_plot_with_archive(fig, output_path)
    plt.close()
    print(f"  ✓ Saved neighborhood visualization: {output_path}")


def test_neighborhood_visualization_lattice_graph(
    N: int = 24,
    gamma: int = 2,
    K: int = 1,
    num_levels: int = 3,
    seed: Optional[int] = 42
):
    """
    Visualize how pooling neighborhoods evolve for a specific node on a 2D lattice graph.
    
    Similar to test_neighborhood_visualization_arbitrary_graph but for regular lattice grids.
    Tracks a single node that remains active throughout all pooling levels and
    visualizes its γ_accumulated-strided neighborhoods at each level.
    
    Args:
        N: Grid size (N×N lattice)
        gamma: Downsampling factor at each level
        K: Number of hop neighborhoods to consider
        num_levels: Number of pooling levels
        seed: Random seed for reproducibility
    """
    print(f"\nVisualizing neighborhoods on {N}×{N} lattice graph")
    print(f"  Parameters: N={N}, γ={gamma}, K={K}, levels={num_levels}")
    
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    # Create lattice graph
    data, G = create_2d_lattice_graph(N=N, B=1, T=1, F=1)
    edge_index = data.edge_index
    num_nodes = N * N
    x = data.x
    
    print(f"  Generated {N}×{N} lattice graph with {num_nodes} nodes, {G.number_of_edges()} edges")
    
    # Create grid layout for nodes
    pos = {}
    for i in range(N):
        for j in range(N):
            node_idx = i * N + j
            pos[node_idx] = (j, i)  # (x, y) = (col, row)
    
    # Initialize
    B, T, F = 1, 1, 1
    active_mask = torch.ones(B, num_nodes, dtype=torch.bool)
    
    # Track neighborhoods and active nodes for each level
    neighborhoods_per_level = []
    active_masks_per_level = []
    node_values_per_level = []  # Track node values at each level
    gamma_values = []
    stride_input_values = []  # Track stride_input for each level
    
    gamma_accumulated = 1
    stride_input = 1  # Start with dense signal
    current_x = x
    current_active_mask = active_mask
    
    # Apply pooling and collect neighborhoods
    for level in range(num_levels):
        # Store node values and active mask BEFORE pooling
        node_vals = current_x[0, 0, :, :].mean(dim=-1).clone()
        node_values_per_level.append(node_vals)
        
        stride_input_values.append(stride_input)  # Track stride_input BEFORE pooling
        gamma_accumulated *= gamma
        
        print(f"  Level {level+1}: stride_input = {stride_input}, γ_acc = {gamma_accumulated}")
        
        pool = StridedGraphMaxPool(
            in_channels=F,
            gamma=gamma,  # Local downsampling ratio
            K=K,
            stride_input=stride_input,  # Input signal spacing
            selection_method='learned',
            pooling_ratio=1.0 / gamma,
            selector_kwargs={
                'hidden_channels': 32,
                'num_gnn_layers': 1,
                'K': 2
            },
        )
        pool.eval()
        
        # Apply pooling WITH neighborhood return
        pooled_x, new_active_mask, _, neighborhoods = pool(
            current_x,
            edge_index,
            active_mask=current_active_mask,
            return_neighborhoods=True,
        )
        
        neighborhoods_per_level.append(neighborhoods)
        active_masks_per_level.append(current_active_mask[0].clone())  # Store BEFORE pooling mask
        gamma_values.append(gamma_accumulated)
        
        num_active = new_active_mask.sum().item()
        print(f"    Active nodes: {num_active}")
        
        # Update stride_input for next level (output stride becomes next input stride)
        stride_input *= gamma
        
        current_x = pooled_x
        current_active_mask = new_active_mask
    
    # Find a node that remains active throughout all levels
    # Start with all nodes active
    always_active = torch.ones(num_nodes, dtype=torch.bool)
    for mask in active_masks_per_level:
        always_active &= mask
    
    always_active_nodes = torch.where(always_active)[0].numpy()
    
    if len(always_active_nodes) == 0:
        print("  ⚠ No node remained active through all levels. Using node active in most levels.")
        # Find node active in most levels
        active_counts = torch.zeros(num_nodes)
        for mask in active_masks_per_level:
            active_counts += mask.float()
        target_node = active_counts.argmax().item()
    else:
        # For lattice, pick a central node (more interesting neighborhoods)
        # Central nodes are around (N//2, N//2)
        center_node = (N//2) * N + (N//2)
        if center_node in always_active_nodes:
            target_node = center_node
        else:
            # Pick the node closest to center
            center_row, center_col = N//2, N//2
            min_dist = float('inf')
            target_node = always_active_nodes[0]
            for node in always_active_nodes:
                row, col = node // N, node % N
                dist = abs(row - center_row) + abs(col - center_col)
                if dist < min_dist:
                    min_dist = dist
                    target_node = node
    
    target_row, target_col = target_node // N, target_node % N
    print(f"  Selected target node: {target_node} (position: [{target_row}, {target_col}], degree: {G.degree(target_node)})")
    
    # Create visualization
    fig = plt.figure(figsize=(24, 8 * num_levels))
    
    for level in range(num_levels):
        ax = plt.subplot(num_levels, 1, level + 1)
        
        visualize_node_neighborhoods(
            G, pos,
            target_node=target_node,
            neighborhood_edges=neighborhoods_per_level[level],
            node_values=node_values_per_level[level],
            active_mask=active_masks_per_level[level],
            title=f"Level {level+1}: $\\gamma_{{{level}}}$={stride_input_values[level]}, $\\gamma_{{{level+1}}}$={gamma_values[level]}, K={K}\n"
                  f"Pooling neighborhoods: {', '.join([str(k*stride_input_values[level]) for k in range(K+1)])}-hops",
            ax=ax,
            node_size=300 if N <= 16 else (200 if N <= 24 else 100),
        )
    
    plt.suptitle(
        f"Strided Pooling Neighborhoods for Node {target_node} [{target_row}, {target_col}]\n"
        f"{N}×{N} Lattice Graph (γ={gamma}, K={K})",
        fontsize=16,
        fontweight='bold',
        y=0.995
    )
    
    output_path = OUTPUT_DIR / f'neighborhoods_lattice_N{N}_g{gamma}_K{K}_L{num_levels}_node{target_node}.pdf'
    save_plot_with_archive(fig, output_path)
    plt.close()
    print(f"  ✓ Saved neighborhood visualization: {output_path}")


# Run tests
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing Nested Strided Pooling with Learned Selection")
    print("="*60 + "\n")

    N = 24  # Grid size for lattice tests

    # Test 1: Lattice graph (original test)
    print("Test 1: Uniform downsampling on 2D lattice (γ=2 at each level)")
    test_nested_strided_pooling(N=N, gamma=2, K=1, num_levels=3)
    
    print("\n" + "-"*60 + "\n")
    
    # Test 2: Arbitrary graphs
    print("Test 2: Nested pooling on arbitrary graphs")
    
    # Test different graph types
    graph_types = [
        'barabasi_albert',
        'erdos_renyi',
        'watts_strogatz',
        'powerlaw_cluster'
    ]
    
    for graph_type in graph_types:
        print(f"\n  Testing {graph_type}...")
        test_nested_pooling_arbitrary_graph(
            graph_type=graph_type,
            n_nodes=512,
            gamma=2,
            K=1,
            num_levels=3,
            seed=42
        )
    
    print("\n" + "-"*60 + "\n")
    
    # Test 3: Neighborhood visualization for arbitrary graphs
    print("Test 3: Visualizing strided pooling neighborhoods (arbitrary graphs)")
    
    for graph_type in graph_types:
        print(f"\n  Testing {graph_type}...")
        test_neighborhood_visualization_arbitrary_graph(
            graph_type=graph_type,
            n_nodes=512,
            gamma=2,
            K=1,
            num_levels=3,
            seed=42
        )
    
    print("\n" + "-"*60 + "\n")
    
    # Test 4: Neighborhood visualization for lattice graph
    print("Test 4: Visualizing strided pooling neighborhoods (2D lattice)")
    test_neighborhood_visualization_lattice_graph(
        N=N,
        gamma=2,
        K=1,
        num_levels=3,
        seed=42
    )
    
    print("\n" + "="*60)
    print("All nested pooling tests completed!")
    print("="*60 + "\n")
