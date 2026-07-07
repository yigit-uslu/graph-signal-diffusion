"""Visualization tests for UGNN encoding path on various graphs."""
import sys
import os
import shutil
import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional, List
from torch_geometric.utils import to_networkx, from_networkx

# Import UGNN modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from graph_signal_diffusion.models.ugnn import (
    UGNN,
    UGNNEncoder,
    GNNConfig,
    PoolingConfig,
    EmbeddingConfig,
    UGNNConfig,
)

# Create output directory for visualizations
OUTPUT_DIR = Path(__file__).parent / "plot_ugnn_encoding_visualizations"
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


def create_watts_strogatz_graph(
    n_nodes: int = 100,
    k: int = 6,
    p: float = 0.3,
    B: int = 1,
    T: int = 10,
    F: int = 1,
    seed: Optional[int] = 42
) -> Tuple[torch.Tensor, torch.Tensor, nx.Graph, dict]:
    """
    Create a Watts-Strogatz small-world graph.
    
    Args:
        n_nodes: Number of nodes
        k: Each node connected to k nearest neighbors in ring topology
        p: Probability of rewiring each edge
        B: Batch size
        T: Number of time steps
        F: Feature dimension
        seed: Random seed
    
    Returns:
        x: Node features (B, T, N, F)
        edge_index: Edge connectivity (2, E)
        G: NetworkX graph
        pos: Node positions for visualization
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    # Create Watts-Strogatz graph
    G = nx.watts_strogatz_graph(n_nodes, k, p, seed=seed)
    
    # Convert to PyG format
    edge_index = torch.tensor(list(G.edges())).t().contiguous()
    # Make undirected (add reverse edges)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    # Create random initial features
    x = torch.randn(B, T, n_nodes, F)
    
    # Compute layout for visualization
    pos = nx.spring_layout(G, seed=seed, k=1/np.sqrt(n_nodes), iterations=50)
    
    return x, edge_index, G, pos


def visualize_graph_with_features(
    G: nx.Graph,
    pos: dict,
    node_features: torch.Tensor,
    active_mask: Optional[torch.Tensor] = None,
    title: str = "Graph",
    ax: Optional[plt.Axes] = None,
    cmap: str = 'viridis',
    node_size: int = 100,
    show_colorbar: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> plt.Figure:
    """
    Visualize graph with node features as colors.

    Args:
        G: NetworkX graph
        pos: Node positions (dict)
        node_features: Node feature values (N,) or (N, F) - will use mean if multi-dim
        active_mask: Binary mask (N,) indicating active nodes
        title: Plot title
        ax: Optional matplotlib axis
        cmap: Colormap
        node_size: Size of nodes
        show_colorbar: Whether to show colorbar
        vmin/vmax: Optional color range overrides

    Returns:
        fig: Matplotlib figure
    """
    num_nodes = len(G.nodes())

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.get_figure()

    # Ensure node_features is a torch tensor
    if isinstance(node_features, np.ndarray):
        node_features = torch.from_numpy(node_features)
    if not torch.is_tensor(node_features):
        node_features = torch.tensor(node_features)

    # Process node features
    if node_features.dim() > 1:
        # Take mean across feature dimensions
        node_values = node_features.mean(dim=-1).cpu().detach().numpy()
    else:
        node_values = node_features.cpu().detach().numpy()

    # Ensure node_values matches graph size
    if len(node_values) != num_nodes:
        raise ValueError(f"node_features size ({len(node_values)}) doesn't match graph size ({num_nodes})")

    # Process active mask
    if active_mask is not None:
        if isinstance(active_mask, np.ndarray):
            active_mask = torch.from_numpy(active_mask)
        if active_mask.dim() > 1:
            active_mask_proc = active_mask[0]  # Take first batch
        else:
            active_mask_proc = active_mask
        active_mask_proc = active_mask_proc.cpu().bool().numpy()

        # Ensure mask matches graph size
        if len(active_mask_proc) != num_nodes:
            raise ValueError(f"active_mask size ({len(active_mask_proc)}) doesn't match graph size ({num_nodes})")
        active_mask = active_mask_proc
    else:
        active_mask = np.ones(num_nodes, dtype=bool)

    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.15, width=0.5)

    # Separate active and inactive nodes
    active_nodes = [i for i in range(num_nodes) if active_mask[i]]
    inactive_nodes = [i for i in range(num_nodes) if not active_mask[i]]

    # Draw active nodes with feature-based colors
    if active_nodes:
        active_values = node_values[active_nodes]
        # Use provided vmin/vmax if available, otherwise compute from active nodes
        vmin_use, vmax_use = (vmin, vmax) if (vmin is not None and vmax is not None) else (active_values.min(), active_values.max())
        if vmin_use == vmax_use:
            vmin_use, vmax_use = vmin_use - 0.1, vmax_use + 0.1

        nx.draw_networkx_nodes(
            G, pos,
            nodelist=active_nodes,
            node_color=active_values,
            node_size=node_size,
            cmap=cmap,
            vmin=vmin_use,
            vmax=vmax_use,
            ax=ax,
            edgecolors='black',
            linewidths=0.5,
        )

    # Draw inactive nodes in gray
    if inactive_nodes:
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=inactive_nodes,
            node_color='lightgray',
            node_size=node_size * 0.3,
            ax=ax,
            alpha=0.4,
        )

    ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
    ax.axis('off')
    ax.set_aspect('equal')

    # Add colorbar
    if show_colorbar and active_nodes:
        vmin_use, vmax_use = (vmin, vmax) if (vmin is not None and vmax is not None) else (active_values.min(), active_values.max())
        if vmin_use == vmax_use:
            vmin_use, vmax_use = vmin_use - 0.1, vmax_use + 0.1
        sm = plt.cm.ScalarMappable(
            cmap=cmap,
            norm=plt.Normalize(vmin=vmin_use, vmax=vmax_use)
        )
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Feature Value', rotation=270, labelpad=20)

    # Add statistics text
    num_active = len(active_nodes)
    num_inactive = len(inactive_nodes)
    if active_nodes:
        mean_val = node_values[active_nodes].mean()
        std_val = node_values[active_nodes].std()
        stats_text = f'Active: {num_active}/{num_nodes}\nMean: {mean_val:.3f}\nStd: {std_val:.3f}'
    else:
        stats_text = f'Active: 0/{num_nodes}'

    ax.text(0.02, 0.98, stats_text,
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    return fig


def test_ugnn_encoding_watts_strogatz(
    n_nodes: int = 100,
    k: int = 6,
    p: float = 0.3,
    base_channels: int = 16,
    channel_multipliers: List[int] = [1, 2, 4],
    gamma: int = 2,
    pool_K: int = 1,
    gnn_K: int = 1,
    selection_method: str = 'learned',
    seed: Optional[int] = 42
):
    """
    Test and visualize UGNN encoding path on Watts-Strogatz graph.
    
    Args:
        n_nodes: Number of nodes
        k: Watts-Strogatz k parameter (nearest neighbors)
        p: Watts-Strogatz p parameter (rewiring probability)
        base_channels: Base number of channels in UGNN
        channel_multipliers: Channel multipliers at each level
        gamma: Downsampling factor
        pool_K: Number of pooling neighborhood scales
        gnn_K: Number of GNN hop scales
        selection_method: Pooling selection method ('stride', 'learned', 'random')
        seed: Random seed
    """
    print(f"\nTesting UGNN encoding on Watts-Strogatz graph")
    print(f"  Graph: n={n_nodes}, k={k}, p={p}")
    print(f"  UGNN: base_channels={base_channels}, multipliers={channel_multipliers}")
    print(f"  Pooling: γ={gamma}, pool_K={pool_K}, selection={selection_method}")
    print(f"  GNN: K={gnn_K}")
    
    # Create graph
    B, T, F = 1, 10, 1
    x, edge_index, G, pos = create_watts_strogatz_graph(
        n_nodes=n_nodes, k=k, p=p, B=B, T=T, F=F, seed=seed
    )
    
    print(f"\n  Graph statistics:")
    print(f"    Nodes: {G.number_of_nodes()}")
    print(f"    Edges: {G.number_of_edges()}")
    print(f"    Avg degree: {2*G.number_of_edges()/G.number_of_nodes():.2f}")
    print(f"    Clustering: {nx.average_clustering(G):.4f}")
    if nx.is_connected(G):
        print(f"    Diameter: {nx.diameter(G)}")
    
    # Create UGNN config
    config = UGNNConfig(
        in_channels=F,
        out_channels=F,
        base_channels=base_channels,
        channel_multipliers=channel_multipliers,
        gnn_config=GNNConfig(
            K=gnn_K,
            num_layers=1,
            norm_type='layer',
            dropout=0.0,
            activation='silu',
            use_strided_conv=True,
            use_pre_activation=False,
        ),
        pooling_config=PoolingConfig(
            gamma=gamma,
            pool_K=pool_K,
            selection_method=selection_method,
        ),
        embedding_config=EmbeddingConfig(
            time_embed_dim=64,
            num_timesteps=1000,
        ),
    )
    
    # Create UGNN model
    ugnn = UGNN(config=config)
    ugnn.eval()
    
    print(f"\n  UGNN architecture:")
    print(f"    Encoder levels: {config.num_levels}")
    print(f"    Stride progression: {ugnn.encoder.stride_pre} (pre) -> {ugnn.encoder.stride_post} (post)")
    
    # Run encoding
    timesteps = torch.zeros(B, dtype=torch.long)  # t=0
    
    with torch.no_grad():
        # Get time embedding
        time_emb = ugnn.time_embed(timesteps)
        
        # Run encoder to get skip features and masks
        res = ugnn.encoder(
            x=x,
            timesteps=timesteps,
            edge_index=edge_index,
            time_emb=time_emb,
        )
        # Support both forms: (x_enc, skip_features, active_masks, intermediates) or direct tuple
        if isinstance(res, tuple) and len(res) >= 3:
            skip_features, active_masks = res[1], res[2]
        else:
            # Fallback (shouldn't happen) — assume legacy return ordering
            skip_features, active_masks = res
    
    print(f"\n  Encoding results:")
    for i, (feats, mask) in enumerate(zip(skip_features, active_masks)):
        num_active = mask.sum().item()
        # Properly expand mask to match feature dimensions (B, T, N, C)
        mask_expanded = mask.unsqueeze(1).unsqueeze(-1).expand(-1, feats.shape[1], -1, feats.shape[-1])
        active_feats = feats[mask_expanded]
        mean_val = active_feats.mean().item()
        std_val = active_feats.std().item()
        print(f"    Level {i}: {feats.shape[-1]} channels, {num_active} active nodes, "
              f"mean={mean_val:.3f}, std={std_val:.3f}")
    
    # Create visualization
    num_levels = len(skip_features)
    fig = plt.figure(figsize=(24, 8 * num_levels))
    
    for level in range(num_levels):
        # Extract features at this level (B, T, N, C) -> (N,) by taking mean over B, T, C
        level_features = skip_features[level][0].mean(dim=(0, -1))  # (N,)
        level_mask = active_masks[level][0]  # (N,)
        
        # Visualize at multiple time steps
        time_indices = [0, T//2, T-1] if T > 2 else [0]
        
        for col_idx, t_idx in enumerate(time_indices):
            ax = fig.add_subplot(num_levels, len(time_indices), level * len(time_indices) + col_idx + 1)
            
            # Get features at this time step
            time_features = skip_features[level][0, t_idx, :, :].mean(dim=-1)  # (N,)
            
            visualize_graph_with_features(
                G=G,
                pos=pos,
                node_features=time_features,
                active_mask=level_mask,
                title=f'Level {level+1} (t={t_idx})\n$\\gamma_{{{level}}}$={ugnn.encoder.stride_pre[level]}, $\\gamma_{{{level+1}}}$={ugnn.encoder.stride_post[level]}',
                ax=ax,
                cmap='RdYlBu_r',
                node_size=max(20, 200 // np.sqrt(n_nodes)),
                show_colorbar=(col_idx == len(time_indices) - 1),
            )
    
    plt.suptitle(
        f"UGNN Encoding Path on Watts-Strogatz Graph\n"
        f"n={n_nodes}, k={k}, p={p}, γ={gamma}, base_ch={base_channels}, "
        f"selection={selection_method}",
        fontsize=16,
        fontweight='bold',
        y=0.995
    )
    
    output_path = OUTPUT_DIR / f'ugnn_encoding_watts_strogatz_n{n_nodes}_k{k}_p{int(p*100)}_g{gamma}_pK{pool_K}_gK{gnn_K}_{selection_method}.pdf'
    save_plot_with_archive(fig, output_path)
    plt.close()
    print(f"\n  ✓ Saved visualization: {output_path}")
    
    # Create a detailed feature evolution plot
    print(f"\n  Creating feature statistics plot...")
    fig2, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Number of active nodes per level
    ax1 = axes[0, 0]
    num_active_list = [mask.sum().item() for mask in active_masks]
    levels = list(range(num_levels))
    ax1.bar(levels, num_active_list, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Encoder Level', fontsize=12)
    ax1.set_ylabel('Number of Active Nodes', fontsize=12)
    ax1.set_title('Active Nodes per Level', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(num_active_list):
        ax1.text(i, v + max(num_active_list)*0.02, str(v), ha='center', fontsize=10)
    
    # Plot 2: Feature statistics (mean absolute value) per level
    ax2 = axes[0, 1]
    mean_abs_vals = []
    for level, (feats, mask) in enumerate(zip(skip_features, active_masks)):
        mask_expanded = mask.unsqueeze(1).unsqueeze(-1).expand(-1, feats.shape[1], -1, feats.shape[-1])
        active_feats = feats[mask_expanded]
        mean_abs_vals.append(active_feats.abs().mean().item())
    ax2.plot(levels, mean_abs_vals, 'o-', linewidth=2, markersize=8, color='coral')
    ax2.set_xlabel('Encoder Level', fontsize=12)
    ax2.set_ylabel('Mean Absolute Feature Value', fontsize=12)
    ax2.set_title('Feature Magnitude per Level', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Feature variance per level
    ax3 = axes[1, 0]
    std_vals = []
    for level, (feats, mask) in enumerate(zip(skip_features, active_masks)):
        mask_expanded = mask.unsqueeze(1).unsqueeze(-1).expand(-1, feats.shape[1], -1, feats.shape[-1])
        active_feats = feats[mask_expanded]
        std_vals.append(active_feats.std().item())
    ax3.plot(levels, std_vals, 'o-', linewidth=2, markersize=8, color='mediumseagreen')
    ax3.set_xlabel('Encoder Level', fontsize=12)
    ax3.set_ylabel('Feature Std Dev', fontsize=12)
    ax3.set_title('Feature Variability per Level', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Channel dimensions per level
    ax4 = axes[1, 1]
    channel_dims = [feats.shape[-1] for feats in skip_features]
    ax4.bar(levels, channel_dims, color='mediumpurple', alpha=0.7, edgecolor='black')
    ax4.set_xlabel('Encoder Level', fontsize=12)
    ax4.set_ylabel('Number of Channels', fontsize=12)
    ax4.set_title('Channel Dimensions per Level', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(channel_dims):
        ax4.text(i, v + max(channel_dims)*0.02, str(v), ha='center', fontsize=10)
    
    plt.suptitle(
        f"UGNN Encoding Statistics\n"
        f"Watts-Strogatz (n={n_nodes}, k={k}, p={p})",
        fontsize=16,
        fontweight='bold',
    )
    plt.tight_layout()
    
    output_path2 = OUTPUT_DIR / f'ugnn_encoding_stats_watts_strogatz_n{n_nodes}_k{k}_p{int(p*100)}_g{gamma}_pK{pool_K}_gK{gnn_K}_{selection_method}.pdf'
    save_plot_with_archive(fig2, output_path2)
    plt.close()
    print(f"  ✓ Saved statistics plot: {output_path2}")
    
    print(f"\n✓ UGNN encoding test completed")


def test_ugnn_encoding_comparison(
    n_nodes: int = 100,
    k: int = 6,
    p: float = 0.3,
    pool_K: int = 1,
    gnn_K: int = 1,
    gamma: int = 2,
    seed: Optional[int] = 42
):
    """
    Compare UGNN encoding with different pooling selection methods.
    """
    print(f"\nComparing UGNN encoding with different selection methods")
    print(f"  Graph: Watts-Strogatz n={n_nodes}, k={k}, p={p}")
    
    selection_methods = ['stride', 'learned', 'random']
    
    # Create graph once
    B, T, F = 1, 10, 1
    x, edge_index, G, pos = create_watts_strogatz_graph(
        n_nodes=n_nodes, k=k, p=p, B=B, T=T, F=F, seed=seed
    )
    
    results = {}
    
    for selection_method in selection_methods:
        print(f"\n  Testing with selection_method='{selection_method}'...")
        
        # Create UGNN
        config = UGNNConfig(
            in_channels=F,
            out_channels=F,
            base_channels=16,
            channel_multipliers=[1, 2, 4],
            gnn_config=GNNConfig(K=gnn_K, num_layers=1, use_strided_conv=True),
            pooling_config=PoolingConfig(
                gamma=gamma,
                pool_K=pool_K,
                selection_method=selection_method,
            ),
            embedding_config=EmbeddingConfig(time_embed_dim=64),
        )
        
        ugnn = UGNN(config=config)
        ugnn.eval()
        
        # Run encoding
        timesteps = torch.zeros(B, dtype=torch.long)
        
        with torch.no_grad():
            time_emb = ugnn.time_embed(timesteps)
            _, skip_features, active_masks, _ = ugnn.encoder(
                x=x, timesteps=timesteps, edge_index=edge_index, time_emb=time_emb
            )
        
        results[selection_method] = {
            'skip_features': skip_features,
            'active_masks': active_masks,
            'stride_pre': ugnn.encoder.stride_pre,
            'stride_post': ugnn.encoder.stride_post,
        }
        
        for i, (feats, mask) in enumerate(zip(skip_features, active_masks)):
            num_active = mask.sum().item()
            print(f"    Level {i}: {num_active} active nodes")
    
    # Create comparison visualization
    print(f"\n  Creating comparison visualization...")
    num_levels = len(results['stride']['skip_features'])
    fig = plt.figure(figsize=(8 * len(selection_methods), 6 * num_levels))
    
    for level in range(num_levels):
        for col_idx, selection_method in enumerate(selection_methods):
            ax = fig.add_subplot(num_levels, len(selection_methods), 
                                level * len(selection_methods) + col_idx + 1)
            
            feats = results[selection_method]['skip_features'][level]
            mask = results[selection_method]['active_masks'][level]
            stride_pre = results[selection_method]['stride_pre'][level]
            stride_post = results[selection_method]['stride_post'][level]
            
            # Take time average and channel average
            time_features = feats[0].mean(dim=(0, -1))  # (N,)
            
            visualize_graph_with_features(
                G=G,
                pos=pos,
                node_features=time_features,
                active_mask=mask[0],
                title=f'Level {level+1} - {selection_method}\n$\\gamma_{{{level}}}$={stride_pre}, $\\gamma_{{{level+1}}}$={stride_post}',
                ax=ax,
                cmap='RdYlBu_r',
                node_size=max(20, 200 // np.sqrt(n_nodes)),
                show_colorbar=(col_idx == len(selection_methods) - 1),
            )
    
    plt.suptitle(
        f"UGNN Encoding Comparison: Different Selection Methods\n"
        f"Watts-Strogatz (n={n_nodes}, k={k}, p={p})",
        fontsize=16,
        fontweight='bold',
        y=0.995
    )
    
    output_path = OUTPUT_DIR / f'ugnn_encoding_comparison_n{n_nodes}_k{k}_p{int(p*100)}_g{gamma}_pK{pool_K}_gK{gnn_K}.pdf'
    save_plot_with_archive(fig, output_path)
    plt.close()
    print(f"  ✓ Saved comparison: {output_path}")
    
    print(f"\n✓ Comparison test completed")


def _compute_panel_vmin_vmax(levels_data: List[dict], panel_field: str):
    """Compute vmin/vmax across all levels for a given panel field.

    Handles scalar vectors (N,) and 2D/3D tensors by extracting the first batch/time/channel.
    For 'selection_scores' we ignore extreme sentinel values (e.g., -1e9) used for inactive nodes.
    Returns (vmin, vmax) or (None, None) if no data found.
    """
    all_vals = []
    for ld in levels_data:
        val = ld.get(panel_field, None)
        if val is None:
            continue
        try:
            if hasattr(val, 'ndim') and val.ndim == 2 and val.shape[0] >= 1:
                arr = val[0].cpu().numpy() if torch.is_tensor(val) else val[0]
            else:
                arr = val[0, 0, :, 0].cpu().numpy() if torch.is_tensor(val) else val[0, 0, :, 0]
            all_vals.append(arr)
        except Exception:
            continue
    if not all_vals:
        return None, None
    concat = np.concatenate([a.ravel() for a in all_vals])
    # Special handling for selection_scores: ignore extreme low sentinel values
    if panel_field == 'selection_scores':
        concat = concat[np.isfinite(concat) & (concat > -1e6)]
        if concat.size == 0:
            return None, None
    else:
        concat = concat[np.isfinite(concat)]
        if concat.size == 0:
            return None, None
    return float(concat.min()), float(concat.max())


def save_intermediates_multipage(
    intermediates: dict,
    G: nx.Graph,
    pos: dict,
    ugnn: UGNN,
    file_path: Path,
    mode: str = 'encoder',
    dpi: int = 300,
):
    """Save detailed intermediates into a multi-page PDF.

    Pages group panels as: (a,b), (c,d), (e,f), (g,h).
    """
    from matplotlib.backends.backend_pdf import PdfPages

    if mode == 'decoder':
        panel_seq = [
            ('a) Input', 'input'),
            ('b) Time Emb', 'time_emb'),
            ('c) Cond Emb', 'cond_emb'),
            ('d) Fused', 'fused'),
            ('e) GNN Output', 'gnn_output'),
            ('f) Max-Pooled (Skip)', 'before_pool'),
            ('g) Selection Scores', 'selection_scores'),
            ('h) Downsampled', 'after_pool'),
        ]
    else:  # encoder
        panel_seq = [
            ('a) Input', 'input'),
            ('b) Time Emb', 'time_emb'),
            ('c) Cond Emb', 'cond_emb'),
            ('d) Fused', 'fused'),
            ('e) GNN Output', 'gnn_output'),
            ('f) Max-Pooled (Skip)', 'before_pool'),
            ('g) Selection Scores', 'selection_scores'),
            ('h) Downsampled', 'after_pool'),
        ]

    num_levels = ugnn.config.num_levels
    level_keys = [f'{mode}_level_{i}' for i in range(num_levels)]

    levels_data = []
    for k_ in level_keys:
        levels_data.append(intermediates.get(k_, {}))

    vmin_vmax = {}
    for _, fld in panel_seq:
        vmin, vmax = _compute_panel_vmin_vmax(levels_data, fld)
        vmin_vmax[fld] = (vmin, vmax)

    page_groups = [(0,1), (2,3), (4,5), (6,7)]

    with PdfPages(file_path) as pdf:
        for page_idx, (i1, i2) in enumerate(page_groups):
            labels = [panel_seq[i1][0], panel_seq[i2][0]]
            fields = [panel_seq[i1][1], panel_seq[i2][1]]

            fig, axes = plt.subplots(nrows=num_levels, ncols=2, figsize=(8.5, max(7, num_levels * 3.5)))
            # Increase spacing for print layout (portrait-oriented)
            fig.subplots_adjust(left=0.06, right=0.98, top=0.96, bottom=0.04, hspace=0.45, wspace=0.60)
            if num_levels == 1:
                axes = np.expand_dims(axes, 0)
            for lvl_idx, ld in enumerate(levels_data):
                for col_idx, fld in enumerate(fields):
                    ax = axes[lvl_idx, col_idx]
                    val = ld.get(fld, None)
                    mask = ld.get('active_mask_in', None) if fld in ('input', 'time_emb', 'cond_emb', 'fused', 'gnn_output') else ld.get('active_mask_out', None)
                    if val is None:
                        n_nodes = G.number_of_nodes()
                        feat = torch.zeros(n_nodes)
                        m = np.zeros(n_nodes, dtype=bool)
                    else:
                        if val.ndim == 2:
                            feat = val[0]
                        else:
                            feat = val[0, 0, :, 0]
                        m = mask[0] if mask is not None else None
                    try:
                        # determine number of channels for annotation
                        if val is None:
                            num_channels = 1
                        elif val.ndim == 2:
                            num_channels = 1
                        else:
                            num_channels = val.shape[-1]

                        vmin, vmax = vmin_vmax.get(fld, (None, None))
                        visualize_graph_with_features(
                            G=G,
                            pos=pos,
                            node_features=feat,
                            active_mask=m,
                            title=f'Level {lvl_idx+1} - {labels[col_idx]}',
                            ax=ax,
                            cmap='RdYlBu_r',
                            node_size=max(15, 150 // np.sqrt(G.number_of_nodes())),
                            show_colorbar=True,
                            vmin=vmin,
                            vmax=vmax,
                        )

                        # Add channel annotation
                        channel_text = f'Ch 1/{num_channels}' if num_channels > 1 else 'Ch 1/1'
                        ax.text(0.98, 0.02, channel_text, transform=ax.transAxes,
                                fontsize=8, verticalalignment='bottom', horizontalalignment='right',
                                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

                    except Exception as e:
                        ax.text(0.5, 0.5, f'Error plotting: {e}', ha='center')
            # Remove page-level suptitle to keep layout clean; per-plot titles are shown
            plt.tight_layout()
            pdf.savefig(fig, dpi=dpi)
            plt.close(fig)
def test_ugnn_encoding_detailed_intermediates(
    n_nodes: int = 100,
    k: int = 6,
    p: float = 0.3,
    base_channels: int = 16,
    channel_multipliers: List[int] = [1, 2, 4],
    gamma: int = 2,
    pool_K: int = 1,
    gnn_K: int = 1,
    selection_method: str = 'learned',
    seed: Optional[int] = 42
):
    """
    Test UGNN encoding with detailed visualization of all intermediate steps.
    
    For each encoder level, visualizes:
    a) Encoder input (with active_mask applied)
    b) Time embeddings
    c) Conditional embeddings (if applicable)
    d) Fused signal (input to main GNN)
    e) Output of main GNN
    f) Selection scores from pooling GNN (learned method only)
    g) Output of downsampling (skip connection signal)
    h) Final output of encoder block (after pooling)
    """
    print(f"\nTesting UGNN encoding with detailed intermediates")
    print(f"  Graph: Watts-Strogatz n={n_nodes}, k={k}, p={p}")
    print(f"  UGNN: base_channels={base_channels}, multipliers={channel_multipliers}")
    print(f"  Pooling: γ={gamma}, pool_K={pool_K}, selection={selection_method}")
    print(f"  GNN: K={gnn_K}")
    
    # Create graph
    B, T, F = 1, 10, 1
    x, edge_index, G, pos = create_watts_strogatz_graph(
        n_nodes=n_nodes, k=k, p=p, B=B, T=T, F=F, seed=seed
    )
    
    # Create UGNN config
    config = UGNNConfig(
        in_channels=F,
        out_channels=F,
        base_channels=base_channels,
        channel_multipliers=channel_multipliers,
        gnn_config=GNNConfig(
            K=gnn_K,
            num_layers=1,
            norm_type='layer',
            dropout=0.0,
            activation='silu',
            use_strided_conv=True,
            use_pre_activation=False,
        ),
        pooling_config=PoolingConfig(
            gamma=gamma,
            pool_K=pool_K,
            selection_method=selection_method,
        ),
        embedding_config=EmbeddingConfig(
            time_embed_dim=64,
            num_timesteps=1000,
        ),
    )
    
    # Create UGNN model
    ugnn = UGNN(config=config)
    ugnn.eval()
    
    print(f"\n  UGNN architecture:")
    print(f"    Encoder levels: {config.num_levels}")
    print(f"    Selection method: {selection_method}")
    
    # Run encoding with intermediates
    timesteps = torch.zeros(B, dtype=torch.long)  # t=0
    
    with torch.no_grad():
        res = ugnn(
            x=x,
            timesteps=timesteps,
            edge_index=edge_index,
            return_intermediates=True,
        )
        if isinstance(res, tuple):
            output, intermediates = res
        else:
            output = res
            intermediates = None

    
    # Check what keys are available
    all_keys = list(intermediates.keys())
    encoder_keys = [k for k in all_keys if 'encoder' in k.lower()]

    if not encoder_keys:
        print(f"\n  ⚠️  ERROR: No encoder intermediates found!")
        print(f"      All intermediate keys: {all_keys}")
        print(f"      Expected keys: 'encoder_level_0', 'encoder_level_1', 'encoder_level_2'")
        return

    print(f"\n  Captured encoder intermediates: {encoder_keys}")


    # Determine number of panels per level
    has_selection_scores = selection_method == 'learned'
    num_panels = 8 if has_selection_scores else 7

    # Panel labels
    if has_selection_scores:
        panel_labels = ['a) Input', 'b) Time Emb', 'c) Cond Emb', 'd) Fused', 
                       'e) GNN Output', 'f) Max-Pooled (Skip)', 'g) Selection Scores', 'h) Downsampled']
    else:
        panel_labels = ['a) Input', 'b) Time Emb', 'c) Cond Emb', 'd) Fused',
                       'e) GNN Output', 'f) Max-Pooled (Skip)', 'g) Downsampled']

    # num_levels = len(encoder_keys)
    num_levels = config.num_levels

    # Create comprehensive visualization
    fig = plt.figure(figsize=(5.5 * num_panels, 8 * num_levels))

    # for level_idx, level_key in enumerate(encoder_keys):
    for level_idx in range(num_levels):
        # level_idx = plot_row_idx
        level_key = f'encoder_level_{level_idx}'
        
        if level_key not in intermediates:
            print(f"  Warning: {level_key} not found in intermediates")
            continue
            
        level_data = intermediates[level_key]

        # level_data = intermediates[level_key]
        
        # Prepare data for each panel
        # Always use first batch, first timestep, first channel
        t_idx = 0  # First timestep
        
        panels_data = []
        
        # Panel a: Input
        if level_data['input'] is not None:
            feat = level_data['input'][0, t_idx, :, 0]  # (N,) first channel
            mask = level_data['active_mask_in']
            num_channels = level_data['input'].shape[-1]
            panels_data.append(('a) Input', feat, mask, num_channels))
        
        # Panel b: Time embeddings
        if level_data['time_emb'] is not None:
            feat = level_data['time_emb'][0, t_idx, :, 0]
            mask = level_data['active_mask_in']
            num_channels = level_data['time_emb'].shape[-1]
            panels_data.append(('b) Time Emb', feat, mask, num_channels))
        
        # Panel c: Conditional embeddings
        if level_data['cond_emb'] is not None:
            feat = level_data['cond_emb'][0, t_idx, :, 0]
            mask = level_data['active_mask_in']
            num_channels = level_data['cond_emb'].shape[-1]
            panels_data.append(('c) Cond Emb', feat, mask, num_channels))
        else:
            # Draw panel with all nodes as inactive (gray) when no conditional
            n_graph_nodes = level_data['input'].shape[2]  # Get number of nodes
            feat = torch.zeros(n_graph_nodes)  # Dummy features
            mask = torch.zeros(1, n_graph_nodes, dtype=torch.bool)  # All inactive
            panels_data.append(('c) Cond Emb', feat, mask, 1))
        
        # Panel d: Fused signal
        if level_data['fused'] is not None:
            feat = level_data['fused'][0, t_idx, :, 0]
            mask = level_data['active_mask_in']
            num_channels = level_data['fused'].shape[-1]
            panels_data.append(('d) Fused', feat, mask, num_channels))
        
        # Panel e: GNN output
        if level_data['gnn_output'] is not None:
            feat = level_data['gnn_output'][0, t_idx, :, 0]
            mask = level_data['active_mask_in']
            num_channels = level_data['gnn_output'].shape[-1]
            panels_data.append(('e) GNN Output', feat, mask, num_channels))
        
        # Panel f: Max-pooled features (output of StridedGraphMaxPool before selection)
        if level_data['before_pool'] is not None:
            feat = level_data['before_pool'][0, t_idx, :, 0]
            mask = level_data['active_mask_in']  # Show all nodes before selection
            num_channels = level_data['before_pool'].shape[-1]
            panels_data.append(('f) Max-Pooled (Skip)', feat, mask, num_channels))
        
        # Panel g: Selection scores (only if learned)
        if has_selection_scores and level_data['selection_scores'] is not None:
            # Selection scores are (B, N) - no time or channel dimension
            feat = level_data['selection_scores'][0, :]
            mask = level_data['active_mask_in']
            num_channels = 1
            panels_data.append(('g) Selection Scores', feat, mask, num_channels))
        
        # Panel h: Downsampled signal (final output after applying selection mask)
        if level_data['after_pool'] is not None:
            feat = level_data['after_pool'][0, t_idx, :, 0]
            mask = level_data['active_mask_out']  # Use output mask to show only selected nodes
            num_channels = level_data['after_pool'].shape[-1]
            panels_data.append(('h) Downsampled', feat, mask, num_channels))
        
        # Plot all panels for this level
        for panel_idx, (label, feat, mask, num_channels) in enumerate(panels_data):
            ax = fig.add_subplot(num_levels, num_panels, level_idx * num_panels + panel_idx + 1)
            
            # Compute channel annotation
            if num_channels > 1:
                channel_text = f'Ch 1/{num_channels}'
            else:
                channel_text = f'Ch 1/1'
            
            visualize_graph_with_features(
                G=G,
                pos=pos,
                node_features=feat,
                active_mask=mask[0] if mask is not None else None,
                title=f'Level {level_idx+1} - {label}\n$\\gamma_{{{level_idx}}}$={ugnn.encoder.stride_pre[level_idx]}, $\\gamma_{{{level_idx+1}}}$={ugnn.encoder.stride_post[level_idx]}',
                ax=ax,
                cmap='RdYlBu_r',
                node_size=max(15, 150 // np.sqrt(n_nodes)),
                show_colorbar=True,
            )
            
            # Add channel annotation
            ax.text(0.98, 0.02, channel_text,
                   transform=ax.transAxes, fontsize=8, 
                   verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.suptitle(
        f"UGNN Encoding: Detailed Intermediate Visualization\n"
        f"Watts-Strogatz (n={n_nodes}, k={k}, p={p}), γ={gamma}, "
        f"selection={selection_method}, timestep t=0",
        fontsize=16,
        fontweight='bold',
        y=0.995
    )
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / f'ugnn_encoding_detailed_n{n_nodes}_k{k}_p{int(p*100)}_g{gamma}_pK{pool_K}_gK{gnn_K}_{selection_method}.pdf'
    save_plot_with_archive(fig, output_path)
    plt.close()
    print(f"\n  ✓ Saved detailed visualization: {output_path}")
    
    # Also save multi-page, printer-friendly PDF (pages: a-b, c-d, e-f, g-h)
    try:
        multipage_path = OUTPUT_DIR / f'ugnn_encoding_detailed_multipage_n{n_nodes}_k{k}_p{int(p*100)}_g{gamma}_pK{pool_K}_gK{gnn_K}_{selection_method}.pdf'
        save_intermediates_multipage(
            intermediates=intermediates,
            G=G,
            pos=pos,
            ugnn=ugnn,
            file_path=multipage_path,
            mode='encoder',
            dpi=300,
        )
        print(f"\n  ✓ Saved multi-page detailed visualization: {multipage_path}")
    except Exception as e:
        print(f"\n  ⚠️ Failed to save multi-page PDF: {e}")


if __name__ == "__main__":

    print("\n" + "="*60)
    print("Testing UGNN Encoding Path Visualizations")
    print("="*60 + "\n")

    n_nodes = 512
    k = 6
    p = 0.1
    gamma = 2 # 2
    pool_K = 1
    gnn_K = 2
    
    # # Test 1: Basic UGNN encoding with learned selection
    # print("Test 1: UGNN encoding with learned selection")
    # test_ugnn_encoding_watts_strogatz(
    #     n_nodes=n_nodes,
    #     k=k,
    #     p=p,
    #     base_channels=16,
    #     channel_multipliers=[1, 2, 4],
    #     gamma=gamma,
    #     pool_K=pool_K,
    #     gnn_K=gnn_K,
    #     selection_method='learned',
    #     seed=42
    # )
    
    # print("\n" + "-"*60 + "\n")
    
    # # Test 2: UGNN encoding with stride selection
    # print("Test 2: UGNN encoding with stride selection")
    # test_ugnn_encoding_watts_strogatz(
    #     n_nodes=n_nodes,
    #     k=k,
    #     p=p,
    #     base_channels=16,
    #     channel_multipliers=[1, 2, 4],
    #     gamma=gamma,
    #     pool_K=pool_K,
    #     gnn_K=gnn_K,
    #     selection_method='stride',
    #     seed=42
    # )
    
    # print("\n" + "-"*60 + "\n")
    
    # # Test 3: Compare different selection methods
    # print("Test 3: Compare selection methods")
    # test_ugnn_encoding_comparison(
    #     n_nodes=n_nodes,
    #     k=k,
    #     p=p,
    #     pool_K=pool_K,
    #     gnn_K=gnn_K,
    #     gamma=gamma,
    #     seed=42,
    # )
    
    # print("\n" + "-"*60 + "\n")
    
    # Test 4: Detailed intermediate visualization with learned selection
    print("Test 4: Detailed intermediate visualization")
    test_ugnn_encoding_detailed_intermediates(
        n_nodes=n_nodes,
        k=k,
        p=p,
        base_channels=16,
        channel_multipliers=[1, 2, 4],
        gamma=gamma,
        pool_K=pool_K,
        gnn_K=gnn_K,
        selection_method='learned',
        seed=42
    )
    
    print("\n" + "="*60 + "\n")
    print("All UGNN encoding visualization tests completed!")
    print("="*60 + "\n")
