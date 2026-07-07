"""Visualization tests for UGNN decoding path on various graphs."""
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
    UGNNDecoder,
    GNNConfig,
    PoolingConfig,
    EmbeddingConfig,
    UGNNConfig,
)

# Create output directory for visualizations
OUTPUT_DIR = Path(__file__).parent / "plot_ugnn_decoding_visualizations"
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
            # selection_scores are (B, N)
            if hasattr(val, 'ndim') and val.ndim == 2 and val.shape[0] >= 1:
                arr = val[0].cpu().numpy() if torch.is_tensor(val) else val[0]
            else:
                # (B, T, N, C) -> take first batch, timestep, channel
                arr = val[0, 0, :, 0].cpu().numpy() if torch.is_tensor(val) else val[0, 0, :, 0]
            all_vals.append(arr)
        except Exception:
            continue
    if not all_vals:
        return None, None
    concat = np.concatenate([a.ravel() for a in all_vals])
    # Special handling for selection_scores: ignore extreme low sentinel values
    if panel_field == 'selection_scores':
        # Filter out very low sentinel values (e.g., -1e9) used to mark inactive nodes
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
    mode: str = 'decoder',
    dpi: int = 300,
):
    """Save detailed intermediates into a multi-page PDF.

    Pages group panels as: (a,b), (c,d), (e,f), (g,h).
    """
    from matplotlib.backends.backend_pdf import PdfPages

    if mode == 'decoder':
        panel_seq = [
            ('a) Input', 'input'),
            ('b) Upsampled', 'after_upsample'),
            ('c) Skip Conn', 'skip_connection'),
            ('d) Upsampled+Skip', 'fused'),
            ('e) Time Emb', 'time_emb'),
            ('f) Cond Emb', 'cond_emb'),
            ('g) Fused', 'gnn_input'),
            ('h) GNN/Decoder Output', 'gnn_output'),
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

    # Levels in order top (0) to bottom (num_levels-1)
    num_levels = ugnn.config.num_levels
    level_keys = [f'{mode}_level_{i}' for i in range(num_levels)]

    # Prepare level_data list for available levels
    levels_data = []
    for k_ in level_keys:
        levels_data.append(intermediates.get(k_, {}))

    # Compute vmin/vmax per panel field
    field_names = [fld for (_, fld) in panel_seq]
    vmin_vmax = {}
    for _, fld in panel_seq:
        vmin, vmax = _compute_panel_vmin_vmax(levels_data, fld)
        vmin_vmax[fld] = (vmin, vmax)

    # Page groups (pairs of indices into panel_seq)
    page_groups = [(0,1), (2,3), (4,5), (6,7)]

    with PdfPages(file_path) as pdf:
        for page_idx, (i1, i2) in enumerate(page_groups):
            labels = [panel_seq[i1][0], panel_seq[i2][0]]
            fields = [panel_seq[i1][1], panel_seq[i2][1]]

            # Portrait-oriented page: narrower width, taller height
            fig, axes = plt.subplots(nrows=num_levels, ncols=2, figsize=(8.5, max(7, num_levels * 3.5)))
            # Increase spacing for print-friendly layout
            # Increase horizontal (wspace) for clearer separation between left/right panels
            fig.subplots_adjust(left=0.06, right=0.98, top=0.96, bottom=0.04, hspace=0.45, wspace=0.60)
            if num_levels == 1:
                axes = np.expand_dims(axes, 0)
            for lvl_idx, ld in enumerate(levels_data):
                for col_idx, fld in enumerate(fields):
                    ax = axes[lvl_idx, col_idx]
                    # Prepare feature and mask (ensure torch tensors)
                    val = ld.get(fld, None)
                    mask = ld.get('active_mask_out', None) if fld != 'input' else ld.get('active_mask_in', None)
                    if val is None:
                        # Dummy
                        n_nodes = G.number_of_nodes()
                        feat = torch.zeros(n_nodes)
                        m = torch.zeros(n_nodes, dtype=torch.bool)
                    else:
                        # Convert numpy arrays to torch tensors if needed
                        if isinstance(val, np.ndarray):
                            if val.ndim == 2:
                                feat = torch.from_numpy(val[0]).float()
                            else:
                                feat = torch.from_numpy(val[0, 0, :, 0]).float()
                        elif torch.is_tensor(val):
                            if val.dim() == 2:
                                feat = val[0]
                            else:
                                feat = val[0, 0, :, 0]
                        else:
                            # Fallback
                            feat = torch.tensor(val).float()
                        # Mask processing
                        if mask is None:
                            m = None
                        else:
                            if isinstance(mask, np.ndarray):
                                m = torch.from_numpy(mask)
                            elif torch.is_tensor(mask):
                                m = mask
                            else:
                                m = torch.tensor(mask)
                            # Take first batch if 2D
                            if m.dim() > 1:
                                m = m[0]
                            m = m.bool()
                    vmin, vmax = vmin_vmax.get(fld, (None, None))
                    try:
                        # determine number of channels for annotation
                        if val is None:
                            num_channels = 1
                        elif isinstance(val, np.ndarray):
                            if val.ndim == 2:
                                num_channels = 1
                            else:
                                num_channels = val.shape[-1]
                        elif torch.is_tensor(val):
                            if val.dim() == 2:
                                num_channels = 1
                            else:
                                num_channels = val.shape[-1]
                        else:
                            num_channels = 1

                        visualize_graph_with_features(
                            G=G,
                            pos=pos,
                            node_features=feat,
                            active_mask=m.cpu() if (m is not None and torch.is_tensor(m)) else None,
                            title=f'Level {lvl_idx+1} - {labels[col_idx]}',
                            ax=ax,
                            cmap='RdYlBu_r',
                            node_size=max(15, 150 // np.sqrt(G.number_of_nodes())),
                            show_colorbar=True,
                            vmin=vmin,
                            vmax=vmax,
                        )

                        # Add channel annotation (bottom-right)
                        channel_text = f'Ch 1/{num_channels}' if num_channels > 1 else 'Ch 1/1'
                        ax.text(0.98, 0.02, channel_text,
                                transform=ax.transAxes, fontsize=8,
                                verticalalignment='bottom', horizontalalignment='right',
                                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

                    except Exception as e:
                        ax.text(0.5, 0.5, f'Error plotting: {e}', ha='center')
            # Remove page-level suptitle to avoid overlapping with per-plot titles
            plt.tight_layout()
            pdf.savefig(fig, dpi=dpi)
            plt.close(fig)

def test_ugnn_decoding_detailed_intermediates(
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
    Test UGNN decoding with detailed visualization of all intermediate steps.
    
    For each decoder level, visualizes:
    a) Decoder input (from previous level or bottleneck)
    b) Skip connection (from encoder)
    c) Time embeddings
    d) Conditional embeddings (if applicable)
    e) Fused signal (after combining input + skip)
    f) After upsampling
    g) GNN output
    h) Final decoder block output
    """
    print(f"\nTesting UGNN decoding with detailed intermediates")
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
    print(f"    Decoder levels: {config.num_levels}")
    print(f"    Selection method: {selection_method}")
    
    # Run full forward pass with intermediates
    timesteps = torch.zeros(B, dtype=torch.long)  # t=0
    
    with torch.no_grad():
        res = ugnn(
            x=x,
            timesteps=timesteps,
            edge_index=edge_index,
            return_intermediates=True,
        )
        # Support models that return either `output` or `(output, intermediates)`
        if isinstance(res, tuple):
            output, intermediates = res
        else:
            output = res
            intermediates = None
    
    # Check what keys are available
    all_keys = list(intermediates.keys())
    decoder_keys = [k for k in all_keys if 'decoder' in k.lower()]
    
    if not decoder_keys:
        print(f"\n  ⚠️  ERROR: No decoder intermediates found!")
        print(f"      All intermediate keys: {all_keys}")
        print(f"      Expected keys: 'decoder_level_0', 'decoder_level_1', 'decoder_level_2'")
        return
    
    print(f"\n  Captured decoder intermediates: {decoder_keys}")
    
    # Panel labels for decoder
    num_panels = 8
    panel_labels = ['a) Input', 'b) Skip Conn', 'c) Time Emb', 'd) Cond Emb',
                   'e) Fused', 'f) Upsampled', 'g) GNN Output', 'h) Output']
    
    num_levels = config.num_levels
    
    # Create comprehensive visualization
    # Decoder levels are drawn from top (shallowest) to bottom (deepest)
    fig = plt.figure(figsize=(5.5 * num_panels, 8 * num_levels))
    
    # Process decoder levels: Level 0 at top, Level 2 at bottom
    for plot_row_idx in range(num_levels):
        # Decoder level index goes from 0 (top row) to num_levels-1 (bottom row)
        level_idx = plot_row_idx
        level_key = f'decoder_level_{level_idx}'
        
        if level_key not in intermediates:
            print(f"  Warning: {level_key} not found in intermediates")
            continue
            
        level_data = intermediates[level_key]
        
        # Prepare data for each panel
        # Always use first batch, first timestep, first channel
        t_idx = 0  # First timestep
        
        panels_data = []
        
        # Panel a: Decoder input (before upsampling)
        if level_data['input'] is not None:
            feat = level_data['input'][0, t_idx, :, 0]  # (N,) first channel
            mask = level_data['active_mask_in']
            num_channels = level_data['input'].shape[-1]
            panels_data.append(('a) Input', feat, mask, num_channels))
        
        # Panel b: After upsampling (newly active nodes should be zero)
        if level_data['after_upsample'] is not None:
            feat = level_data['after_upsample'][0, t_idx, :, 0]
            mask = level_data['active_mask_out']  # Show all upsampled nodes (more than input)
            num_channels = level_data['after_upsample'].shape[-1]
            panels_data.append(('b) Upsampled', feat, mask, num_channels))
        
        # Panel c: Skip connection from encoder
        if level_data['skip_connection'] is not None:
            feat = level_data['skip_connection'][0, t_idx, :, 0]
            mask = level_data['active_mask_out']  # Skip has same resolution as upsampled
            num_channels = level_data['skip_connection'].shape[-1]
            panels_data.append(('c) Skip Conn', feat, mask, num_channels))
        
        # Panel d: Fused signal (after combining upsampled + skip)
        if level_data['fused'] is not None:
            feat = level_data['fused'][0, t_idx, :, 0]
            mask = level_data['active_mask_out']
            num_channels = level_data['fused'].shape[-1]
            panels_data.append(('d) Upsampled+Skip', feat, mask, num_channels))
        
        # Panel e: Time embeddings
        if level_data['time_emb'] is not None:
            feat = level_data['time_emb'][0, t_idx, :, 0]
            mask = level_data['active_mask_out']
            num_channels = level_data['time_emb'].shape[-1]
            panels_data.append(('e) Time Emb', feat, mask, num_channels))
        
        # Panel f: Conditional embeddings
        if level_data['cond_emb'] is not None:
            feat = level_data['cond_emb'][0, t_idx, :, 0]
            mask = level_data['active_mask_out']
            num_channels = level_data['cond_emb'].shape[-1]
            panels_data.append(('f) Cond Emb', feat, mask, num_channels))
        else:
            # Draw panel with all nodes as inactive (gray) when no conditional
            n_graph_nodes = level_data['input'].shape[2]  # Get number of nodes
            feat = torch.zeros(n_graph_nodes)  # Dummy features
            mask = torch.zeros(1, n_graph_nodes, dtype=torch.bool)  # All inactive
            panels_data.append(('f) Cond Emb', feat, mask, 1))
        
        # Panel g: GNN input (fused with time+cond embeddings)
        if level_data['gnn_input'] is not None:
            feat = level_data['gnn_input'][0, t_idx, :, 0]
            mask = level_data['active_mask_out']
            num_channels = level_data['gnn_input'].shape[-1]
            panels_data.append(('g) Fused', feat, mask, num_channels))
        
        # Panel h: GNN/Decoder output
        if level_data['gnn_output'] is not None:
            feat = level_data['gnn_output'][0, t_idx, :, 0]
            mask = level_data['active_mask_out']
            num_channels = level_data['gnn_output'].shape[-1]
            panels_data.append(('h) GNN/Decoder Output', feat, mask, num_channels))
        
        # Plot all panels for this level
        for panel_idx, (label, feat, mask, num_channels) in enumerate(panels_data):
            ax = fig.add_subplot(num_levels, num_panels, plot_row_idx * num_panels + panel_idx + 1)
            
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
                title=f'Decoder Level {level_idx+1} - {label}\n$\\gamma_{{{level_idx+1}}}$={ugnn.decoder.stride_post[level_idx]}, $\\gamma_{{{level_idx}}}$={ugnn.decoder.stride_pre[level_idx]}',
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
        f"UGNN Decoding: Detailed Intermediate Visualization\n"
        f"Watts-Strogatz (n={n_nodes}, k={k}, p={p}), γ={gamma}, "
        f"selection={selection_method}, timestep t=0",
        fontsize=16,
        fontweight='bold',
        y=0.995
    )
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / f'ugnn_decoding_detailed_n{n_nodes}_k{k}_p{int(p*100)}_g{gamma}_pK{pool_K}_gK{gnn_K}_{selection_method}.pdf'
    save_plot_with_archive(fig, output_path)
    plt.close()
    print(f"\n  ✓ Saved detailed visualization: {output_path}")

    # Also save multi-page, printer-friendly PDF (pages: a-b, c-d, e-f, g-h)
    try:
        multipage_path = OUTPUT_DIR / f'ugnn_decoding_detailed_multipage_n{n_nodes}_k{k}_p{int(p*100)}_g{gamma}_pK{pool_K}_gK{gnn_K}_{selection_method}.pdf'
        save_intermediates_multipage(
            intermediates=intermediates,
            G=G,
            pos=pos,
            ugnn=ugnn,
            file_path=multipage_path,
            mode='decoder',
            dpi=300,
        )
        print(f"\n  ✓ Saved multi-page detailed visualization: {multipage_path}")
    except Exception as e:
        print(f"\n  ⚠️ Failed to save multi-page PDF: {e}")

    print(f"\n✓ Detailed decoder intermediates test completed")


# Run tests
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing UGNN Decoding Path Visualizations")
    print("="*60 + "\n")

    n_nodes = 512
    k = 6
    p = 0.1
    gamma = 2
    pool_K = 1
    gnn_K = 2
    
    # Test 1: Detailed intermediate visualization with learned selection
    print("Test 1: Detailed decoder intermediate visualization")
    test_ugnn_decoding_detailed_intermediates(
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
    print("All UGNN decoding visualization tests completed!")
    print("="*60 + "\n")
