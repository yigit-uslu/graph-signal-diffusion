"""
Tests for NeighborhoodFeaturePooling and its composition inside StridedGraphMaxPool.

Standalone tests (1-10): CPU, fixed seed, bit-exact.
Integration tests (11-18): Verify StridedGraphMaxPool refactored pipeline.
"""

from pathlib import Path

import pytest
import torch
from torch_geometric.utils import to_dense_adj

from graph_signal_diffusion.models.components.pooling import (
    NeighborhoodFeaturePooling,
    StridedGraphMaxPool,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
DEFAULT_VIS_OUTPUT_DIR = (
    Path(__file__).resolve().parents[1] / "figs" / "NeighborhoodFeaturePooling"
)


def _line_graph_edge_index(N: int) -> torch.Tensor:
    """0-1-2-..-(N-1) undirected line graph."""
    src = list(range(N - 1)) + list(range(1, N))
    dst = list(range(1, N)) + list(range(N - 1))
    return torch.tensor([src, dst], dtype=torch.long)


def _lattice_graph_edge_index(side: int) -> torch.Tensor:
    """2D lattice (4-neighborhood), undirected edges."""
    src = []
    dst = []

    def node_id(r: int, c: int) -> int:
        return r * side + c

    for r in range(side):
        for c in range(side):
            u = node_id(r, c)
            if c + 1 < side:
                v = node_id(r, c + 1)
                src.extend([u, v])
                dst.extend([v, u])
            if r + 1 < side:
                v = node_id(r + 1, c)
                src.extend([u, v])
                dst.extend([v, u])

    return torch.tensor([src, dst], dtype=torch.long)


def _checkerboard_active_mask(side: int) -> torch.Tensor:
    """Predefined active mask for lattice toy tests."""
    rows = torch.arange(side).unsqueeze(1).expand(side, side)
    cols = torch.arange(side).unsqueeze(0).expand(side, side)
    mask = ((rows + cols) % 2 == 0).reshape(1, side * side)
    return mask.bool()


def _watts_strogatz_active_mask(num_nodes: int, keep_every: int = 3) -> torch.Tensor:
    """Predefined sparse active mask for Watts-Strogatz toy tests."""
    mask = torch.zeros(1, num_nodes, dtype=torch.bool)
    mask[:, ::keep_every] = True
    return mask


def _triangle_edge_index() -> torch.Tensor:
    """3-node triangle: 0-1, 1-2, 0-2."""
    return torch.tensor([[0, 1, 1, 2, 0, 2], [1, 0, 2, 1, 2, 0]], dtype=torch.long)


def _star_graph_edge_index(N: int) -> torch.Tensor:
    """Star graph: node 0 connected to all others."""
    src = [0] * (N - 1) + list(range(1, N))
    dst = list(range(1, N)) + [0] * (N - 1)
    return torch.tensor([src, dst], dtype=torch.long)


def _dense_within_hop_max_reference(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    num_hops: int,
    include_self: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense reference for sparse max path: exactly num_hops on A(+I)."""
    N = x.size(2)
    adj = to_dense_adj(edge_index, max_num_nodes=N)[0].bool()
    eye = torch.eye(N, device=x.device, dtype=torch.bool)
    adj_base = adj | eye if include_self else adj

    if num_hops <= 0:
        adj_power = eye if include_self else torch.zeros_like(adj)
    else:
        adj_power = adj_base.clone()
        for _ in range(num_hops - 1):
            adj_power = torch.matmul(adj_power.float(), adj_base.float()) > 0

    x_source = x.unsqueeze(2)  # (B, T, 1, N, F)
    adj_mask = adj_power.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # (1, 1, N, N, 1)
    x_masked = torch.where(adj_mask, x_source, torch.full_like(x_source, float('-inf')))
    x_pooled = torch.max(x_masked, dim=3).values
    x_pooled = torch.where(torch.isinf(x_pooled), torch.zeros_like(x_pooled), x_pooled)
    return x_pooled, adj_power


def _compute_lattice_toy_pool(
    side: int = 5,
    active_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute toy lattice max-pooling outputs for visualization tests."""
    N = side * side
    edge_index = _lattice_graph_edge_index(side)

    rows = torch.arange(side).repeat_interleave(side)
    cols = torch.arange(side).repeat(side)
    base = (0.6 * cols.float()) - (0.4 * rows.float())
    base = base / base.abs().max() * 2.0
    # Add local peaks away from the center so the chosen center node is
    # visibly affected by max-pooling (its pooled value comes from neighbors).
    base[2 * side + 0] = 6.0  # 2-hop from center (2,2)
    base[0 * side + 2] = 5.2  # 2-hop from center (2,2)
    base[4 * side + 2] = 4.3  # 2-hop from center (2,2)
    x = base.view(1, 1, N, 1)

    nfp = NeighborhoodFeaturePooling(K=1, stride=2, aggregation='max', include_self=True)
    x_pooled, edge_index_multi = nfp(
        x, edge_index, active_mask=active_mask, return_adj=True
    )
    return x, x_pooled, edge_index, edge_index_multi


def _compute_watts_strogatz_toy_pool(
    num_nodes: int = 64,
    k: int = 6,
    p: float = 0.18,
    active_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Compute toy max-pooling outputs on a Watts-Strogatz graph."""
    nx = pytest.importorskip("networkx")
    np = pytest.importorskip("numpy")
    G = nx.watts_strogatz_graph(num_nodes, k, p, seed=11)

    # Undirected edge_index (both directions)
    src = []
    dst = []
    for u, v in G.edges():
        src.extend([u, v])
        dst.extend([v, u])
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # Deterministic node layout for visualization.
    pos_dict = nx.spring_layout(G, seed=23, k=1.4 / (num_nodes ** 0.5), iterations=250)
    pos_np = np.asarray([pos_dict[i] for i in range(num_nodes)], dtype=float)
    pos = torch.from_numpy(pos_np).to(dtype=torch.float32)  # (N, 2)

    # Degree-driven base feature field.
    deg = torch.tensor([G.degree(i) for i in range(num_nodes)], dtype=torch.float32)
    base = (deg - deg.mean()) / (deg.std(unbiased=False) + 1e-6)
    base = 0.7 * base
    x = base.view(1, 1, num_nodes, 1).clone()

    nfp = NeighborhoodFeaturePooling(K=1, stride=2, aggregation='max', include_self=True)
    # Use neighborhood structure to choose a node and create a clear pooling effect.
    _, edge_index_multi = nfp(torch.zeros_like(x), edge_index, return_adj=True)
    adj_multi = to_dense_adj(edge_index_multi, max_num_nodes=num_nodes)[0].bool()

    # Pick a node with the largest pooled neighborhood.
    example_node = int(torch.argmax(adj_multi.sum(dim=1)).item())
    neighbors = torch.where(adj_multi[example_node])[0]
    donor_candidates = neighbors[neighbors != example_node]
    if donor_candidates.numel() > 0:
        donor_node = int(donor_candidates[0].item())
    else:
        donor_node = (example_node + 1) % num_nodes

    # Force example node to be below donor so max-pooling visibly changes it.
    x[0, 0, example_node, 0] = -0.8
    x[0, 0, donor_node, 0] = 6.5

    x_pooled, edge_index_multi = nfp(
        x, edge_index, active_mask=active_mask, return_adj=True
    )
    return x, x_pooled, edge_index, edge_index_multi, pos, example_node


def _save_max_toy_visualization(
    output_dir: Path | None = None,
    active_mask: torch.Tensor | None = None,
    filename: str = "nfp_lattice_max_pool_all_active.png",
) -> Path:
    """Render and save a lattice toy max-pooling visualization."""
    plt = pytest.importorskip("matplotlib.pyplot")
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if output_dir is None:
        output_dir = DEFAULT_VIS_OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    side = 5
    N = side * side
    x, x_pooled, edge_index, edge_index_multi = _compute_lattice_toy_pool(
        side=side, active_mask=active_mask
    )

    # Safety re-mask for visualization: inactive targets are forced to zero.
    if active_mask is not None:
        x_pooled_plot = x_pooled * active_mask.unsqueeze(1).unsqueeze(-1).to(dtype=x_pooled.dtype)
    else:
        x_pooled_plot = x_pooled

    x_in = x[0, 0, :, 0].cpu().numpy()
    x_out = x_pooled_plot[0, 0, :, 0].cpu().numpy()
    if active_mask is None:
        active = torch.ones(N, dtype=torch.bool)
    else:
        active = active_mask[0].cpu().bool()
    inactive = ~active

    xs = torch.arange(side).repeat(side).cpu().numpy()
    ys = (-torch.arange(side).repeat_interleave(side)).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=False)
    pooled_scatter = None

    for ax, vals, title in [
        (axes[0], x_in, "Input Features (5x5 Lattice)"),
        (axes[1], x_out, "Pooled Features (K=1, stride=2)"),
    ]:
        for s, d in zip(edge_index[0].tolist(), edge_index[1].tolist()):
            ax.plot([xs[s], xs[d]], [ys[s], ys[d]], color='lightgray', linewidth=0.8, alpha=0.5, zorder=1)
        sc = ax.scatter(xs, ys, c=vals, cmap='viridis', s=140, edgecolors='black', linewidths=0.7, zorder=2)
        if ax is axes[1]:
            pooled_scatter = sc
        if inactive.any():
            idx = inactive.nonzero(as_tuple=False).view(-1).cpu().numpy()
            ax.scatter(
                xs[idx], ys[idx],
                s=230,
                facecolors='none',
                edgecolors='red',
                linewidths=1.3,
                zorder=3,
            )
        ax.set_title(title)
        ax.set_xticks(range(side))
        ax.set_yticks([-i for i in range(side)])
        ax.set_xlim(-0.8, side - 0.2)
        ax.set_ylim(-(side - 0.2), 0.8)
        ax.set_aspect('equal')

    # Annotate one example node's pooling neighborhood in the pooled panel.
    center_node = (side // 2) * side + (side // 2)
    if active.any():
        example_node = center_node if active[center_node] else int(torch.where(active)[0][0].item())
    else:
        example_node = center_node
    adj_multi = to_dense_adj(edge_index_multi, max_num_nodes=N)[0].bool().cpu()
    neighborhood = torch.where(adj_multi[example_node])[0]
    neighborhood_active = neighborhood[active[neighborhood]]
    neighborhood_inactive = neighborhood[~active[neighborhood]]

    ax_pool = axes[1]
    # Ring all neighborhood nodes used for this target node's aggregation.
    if neighborhood.numel() > 0:
        idx = neighborhood.cpu().numpy()
        ax_pool.scatter(
            xs[idx], ys[idx],
            s=260, facecolors='none', edgecolors='orange', linewidths=1.8, zorder=4,
        )
    # Inactive neighbors are crossed to show they are excluded as sources.
    if neighborhood_inactive.numel() > 0:
        idx_inactive = neighborhood_inactive.cpu().numpy()
        ax_pool.scatter(
            xs[idx_inactive], ys[idx_inactive],
            s=70, marker='x', c='orange', linewidths=1.4, zorder=5,
        )
    # Mark the example target node.
    ax_pool.scatter(
        [xs[example_node]], [ys[example_node]],
        s=360, marker='*', c='red', edgecolors='black', linewidths=0.8, zorder=6,
    )
    for nb in neighborhood.tolist():
        if nb == example_node:
            continue
        ax_pool.plot(
            [xs[example_node], xs[nb]],
            [ys[example_node], ys[nb]],
            linestyle='--', color='orange', linewidth=0.8, alpha=0.45, zorder=3,
        )

    pre_val = float(x_in[example_node])
    pooled_val = float(x_out[example_node])
    delta_val = pooled_val - pre_val
    if neighborhood_active.numel() > 0:
        src_max = float(x_in[neighborhood_active.cpu().numpy()].max())
        summary = (
            f"Example node {example_node}\n"
            f"|N|={neighborhood.numel()} active={neighborhood_active.numel()} inactive={neighborhood_inactive.numel()}\n"
            f"input={pre_val:.2f}, pooled={pooled_val:.2f}, Δ={delta_val:+.2f}\n"
            f"max(active N)={src_max:.2f}"
        )
    else:
        summary = (
            f"Example node {example_node}\n"
            f"|N|={neighborhood.numel()} active=0 inactive={neighborhood_inactive.numel()}\n"
            f"input={pre_val:.2f}, pooled={pooled_val:.2f}, Δ={delta_val:+.2f}\n"
            f"(no active sources)"
        )
    ax_pool.text(
        0.02, 0.98, summary,
        transform=ax_pool.transAxes, va='top', ha='left', fontsize=8,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray'),
    )

    # Keep colorbar anchored to the pooled (right) panel.
    divider = make_axes_locatable(ax_pool)
    cax = divider.append_axes("right", size="4.5%", pad=0.08)
    fig.colorbar(pooled_scatter, cax=cax)
    if inactive.any():
        fig.suptitle("Red circles: inactive source nodes in active_mask", fontsize=10, y=0.995)

    # Keep generous top margin so title/annotations are never clipped on save.
    if inactive.any():
        fig.subplots_adjust(top=0.88, wspace=0.20, right=0.94)
    else:
        fig.subplots_adjust(top=0.92, wspace=0.20, right=0.94)

    out_path = output_dir / filename
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def _save_watts_strogatz_max_toy_visualization(
    output_dir: Path | None = None,
    active_mask: torch.Tensor | None = None,
    filename: str = "nfp_watts_strogatz_max_pool_all_active.png",
) -> Path:
    """Render and save a Watts-Strogatz toy max-pooling visualization."""
    plt = pytest.importorskip("matplotlib.pyplot")
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    if output_dir is None:
        output_dir = DEFAULT_VIS_OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x, x_pooled, edge_index, edge_index_multi, pos, example_node = _compute_watts_strogatz_toy_pool(
        active_mask=active_mask
    )

    N = x.size(2)
    if active_mask is not None:
        x_pooled_plot = x_pooled * active_mask.unsqueeze(1).unsqueeze(-1).to(dtype=x_pooled.dtype)
    else:
        x_pooled_plot = x_pooled

    x_in = x[0, 0, :, 0].cpu().numpy()
    x_out = x_pooled_plot[0, 0, :, 0].cpu().numpy()
    if active_mask is None:
        active = torch.ones(N, dtype=torch.bool)
    else:
        active = active_mask[0].cpu().bool()
    inactive = ~active

    xs = pos[:, 0].cpu().numpy()
    ys = pos[:, 1].cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=False)
    pooled_scatter = None

    # Draw each undirected edge once.
    undirected_edges = set()
    for s, d in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        a, b = (s, d) if s <= d else (d, s)
        if a != b:
            undirected_edges.add((a, b))

    for ax, vals, title in [
        (axes[0], x_in, "Input Features (Watts-Strogatz)"),
        (axes[1], x_out, "Pooled Features (K=1, stride=2)"),
    ]:
        for s, d in undirected_edges:
            ax.plot([xs[s], xs[d]], [ys[s], ys[d]], color='lightgray', linewidth=0.5, alpha=0.35, zorder=1)
        sc = ax.scatter(xs, ys, c=vals, cmap='viridis', s=55, edgecolors='black', linewidths=0.35, zorder=2)
        if ax is axes[1]:
            pooled_scatter = sc
        if inactive.any():
            idx = inactive.nonzero(as_tuple=False).view(-1).cpu().numpy()
            ax.scatter(
                xs[idx], ys[idx],
                s=85, facecolors='none', edgecolors='red', linewidths=0.8, zorder=3,
            )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')

    adj_multi = to_dense_adj(edge_index_multi, max_num_nodes=N)[0].bool().cpu()
    neighborhood = torch.where(adj_multi[example_node])[0]
    neighborhood_active = neighborhood[active[neighborhood]]
    neighborhood_inactive = neighborhood[~active[neighborhood]]

    ax_pool = axes[1]
    if neighborhood.numel() > 0:
        idx = neighborhood.cpu().numpy()
        ax_pool.scatter(
            xs[idx], ys[idx],
            s=115, facecolors='none', edgecolors='orange', linewidths=1.1, zorder=4,
        )
    if neighborhood_inactive.numel() > 0:
        idx_inactive = neighborhood_inactive.cpu().numpy()
        ax_pool.scatter(
            xs[idx_inactive], ys[idx_inactive],
            s=50, marker='x', c='orange', linewidths=1.0, zorder=5,
        )
    ax_pool.scatter(
        [xs[example_node]], [ys[example_node]],
        s=250, marker='*', c='red', edgecolors='black', linewidths=0.7, zorder=6,
    )

    # Keep connector clutter bounded for larger neighborhoods.
    max_connectors = 24
    for nb in neighborhood.tolist()[:max_connectors]:
        if nb == example_node:
            continue
        ax_pool.plot(
            [xs[example_node], xs[nb]],
            [ys[example_node], ys[nb]],
            linestyle='--', color='orange', linewidth=0.5, alpha=0.35, zorder=3,
        )

    pre_val = float(x_in[example_node])
    pooled_val = float(x_out[example_node])
    delta_val = pooled_val - pre_val
    if neighborhood_active.numel() > 0:
        src_max = float(x_in[neighborhood_active.cpu().numpy()].max())
        summary = (
            f"Example node {example_node}\n"
            f"|N|={neighborhood.numel()} active={neighborhood_active.numel()} inactive={neighborhood_inactive.numel()}\n"
            f"input={pre_val:.2f}, pooled={pooled_val:.2f}, Δ={delta_val:+.2f}\n"
            f"max(active N)={src_max:.2f}"
        )
    else:
        summary = (
            f"Example node {example_node}\n"
            f"|N|={neighborhood.numel()} active=0 inactive={neighborhood_inactive.numel()}\n"
            f"input={pre_val:.2f}, pooled={pooled_val:.2f}, Δ={delta_val:+.2f}\n"
            f"(no active sources)"
        )
    ax_pool.text(
        0.02, 0.98, summary,
        transform=ax_pool.transAxes, va='top', ha='left', fontsize=8,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.88, edgecolor='gray'),
    )

    divider = make_axes_locatable(ax_pool)
    cax = divider.append_axes("right", size="4.5%", pad=0.08)
    fig.colorbar(pooled_scatter, cax=cax)
    if inactive.any():
        fig.suptitle("Red circles: inactive source nodes in active_mask", fontsize=10, y=0.995)

    if inactive.any():
        fig.subplots_adjust(top=0.90, wspace=0.12, right=0.95)
    else:
        fig.subplots_adjust(top=0.94, wspace=0.12, right=0.95)

    out_path = output_dir / filename
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


# ===========================================================================
# Standalone tests
# ===========================================================================

class TestNeighborhoodFeaturePoolingStandalone:
    """Tests 1-10: standalone NeighborhoodFeaturePooling on CPU."""

    # Test 1: Max pooling correctness
    def test_max_pooling_correctness(self):
        """Max pooling over triangle graph with K=1, stride=1."""
        torch.manual_seed(42)
        N = 3
        edge_index = _triangle_edge_index()
        nfp = NeighborhoodFeaturePooling(K=1, stride=1, aggregation='max')

        # (1, 1, 3, 2) features
        x = torch.tensor([[[[1.0, 2.0], [-1.0, 3.0], [0.5, -0.5]]]])
        x_pooled, adj = nfp(x, edge_index)

        assert x_pooled.shape == x.shape
        # K=1, stride=1 with include_self=True: every node sees all 3 nodes
        # (triangle => A|I is all-ones for K=1 with stride=1 and (A|I)^1)
        # Node 0: max(1, -1, 0.5)=1.0, max(2, 3, -0.5)=3.0
        assert torch.allclose(x_pooled[0, 0, 0], torch.tensor([1.0, 3.0]))
        # Node 1: max(1, -1, 0.5)=1.0, max(2, 3, -0.5)=3.0
        assert torch.allclose(x_pooled[0, 0, 1], torch.tensor([1.0, 3.0]))
        # Node 2: same
        assert torch.allclose(x_pooled[0, 0, 2], torch.tensor([1.0, 3.0]))

    # Test 2: Avg pooling correctness
    def test_avg_pooling_correctness(self):
        """Avg pooling over triangle graph with K=1, stride=1."""
        N = 3
        edge_index = _triangle_edge_index()
        nfp = NeighborhoodFeaturePooling(K=1, stride=1, aggregation='avg')

        x = torch.tensor([[[[3.0, 6.0], [0.0, 3.0], [6.0, 0.0]]]])
        x_pooled, _ = nfp(x, edge_index)

        # Triangle, K=1, stride=1 => all 3 nodes are neighbors of each other
        # Each node's avg = mean of all 3 = [3.0, 3.0]
        expected = torch.tensor([3.0, 3.0])
        for n in range(3):
            assert torch.allclose(x_pooled[0, 0, n], expected), (
                f"Node {n}: {x_pooled[0, 0, n]} != {expected}"
            )

    # Test 3: Active mask with max pooling
    def test_active_mask_max(self):
        """Max pooling skips inactive nodes."""
        N = 4
        edge_index = _line_graph_edge_index(N)
        nfp = NeighborhoodFeaturePooling(K=1, stride=1, aggregation='max')

        # Features: node 0=10, node 1=-5, node 2=20, node 3=-1
        x = torch.tensor([[[[10.0], [-5.0], [20.0], [-1.0]]]])
        # Only nodes 0 and 2 active
        active_mask = torch.tensor([[True, False, True, False]])

        x_pooled, _ = nfp(x, edge_index, active_mask=active_mask)

        # Node 0 neighbors (with K=1, stride=1): {0, 1}. Active: {0}. Max = 10.
        assert x_pooled[0, 0, 0, 0] == 10.0
        # Node 1 neighbors: {0, 1, 2}. Active: {0, 2}. Max = 20. But node 1 was
        # inactive, so after -inf cleanup its features become 0.
        # Actually: -inf masking sets inactive features to -inf before pooling,
        # then after pooling -inf -> 0. So node 1 (inactive) gets max of neighbors
        # but any node that was ALL -inf will be zeroed.
        # Node 1 neighbors include active 0 and 2, so max(10, -inf, 20, -inf)=20
        # But node 1 is inactive, so after pooling it should still show 20 because
        # the -inf cleanup only converts remaining -inf to 0.
        assert x_pooled[0, 0, 1, 0] == 20.0

    # Test 4: Active mask with avg pooling
    def test_active_mask_avg(self):
        """Avg pooling only counts active neighbors."""
        N = 3
        edge_index = _triangle_edge_index()
        nfp = NeighborhoodFeaturePooling(K=1, stride=1, aggregation='avg')

        x = torch.tensor([[[[6.0], [999.0], [12.0]]]])
        active_mask = torch.tensor([[True, False, True]])

        x_pooled, _ = nfp(x, edge_index, active_mask=active_mask)

        # Node 0 neighbors (triangle, K=1): {0,1,2}. Active: {0,2}. Avg = (6+12)/2 = 9
        assert torch.allclose(x_pooled[0, 0, 0], torch.tensor([9.0]))

    # Test 5: K=0 identity (include_self=True)
    def test_k_zero_identity(self):
        """K=0 returns input unchanged."""
        torch.manual_seed(0)
        N = 5
        edge_index = _line_graph_edge_index(N)
        nfp = NeighborhoodFeaturePooling(K=0, stride=1, aggregation='max')

        x = torch.randn(2, 3, N, 4)
        x_pooled, adj = nfp(x, edge_index)

        assert torch.equal(x_pooled, x)
        assert adj is None

    # Test 6: Strided neighborhoods (stride>1)
    def test_strided_neighborhoods(self):
        """stride=2, K=1 max path uses 2-step propagation on A+I."""
        N = 5
        edge_index = _line_graph_edge_index(N)
        nfp = NeighborhoodFeaturePooling(K=1, stride=2, aggregation='max')

        x = torch.zeros(1, 1, N, 1)
        x[0, 0, :, 0] = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])

        x_pooled, adj = nfp(x, edge_index, return_adj=True)

        # Verify adjacency via dense within-hop reference
        x_ref, adj_ref = _dense_within_hop_max_reference(
            x, edge_index, num_hops=2, include_self=True
        )
        assert torch.allclose(x_pooled, x_ref)

        adj_dense = to_dense_adj(adj, max_num_nodes=N)[0]
        assert torch.equal(adj_dense.bool(), adj_ref)

    # Test 7: include_self=False
    def test_include_self_false_stride1(self):
        """include_self=False (relaxed) with one hop has no diagonal."""
        N = 4
        edge_index = _line_graph_edge_index(N)
        nfp = NeighborhoodFeaturePooling(
            K=1, stride=1, aggregation='max', include_self=False
        )

        # Use return_adj to check adjacency directly
        x = torch.randn(1, 1, N, 2)
        _, adj = nfp(x, edge_index, return_adj=True)
        adj_dense = to_dense_adj(adj, max_num_nodes=N)[0]

        # One-step without self-loops => no diagonal entries
        for i in range(N):
            assert adj_dense[i, i] == 0, f"Self-loop at node {i}"

    def test_include_self_false_stride_gt1(self):
        """include_self=False relaxed mode may reintroduce diagonal via cycles."""
        N = 5
        edge_index = _line_graph_edge_index(N)
        nfp = NeighborhoodFeaturePooling(
            K=1, stride=2, aggregation='max', include_self=False
        )

        x = torch.randn(1, 1, N, 2)
        _, adj = nfp(x, edge_index, return_adj=True)
        adj_dense = to_dense_adj(adj, max_num_nodes=N)[0]

        # Relaxed mode uses A^2 (no explicit self-loops), so diagonal can appear via 2-cycles.
        assert adj_dense[0, 0] == 1
        # Node 0 still reaches node 2 in two steps.
        assert adj_dense[0, 2] == 1

    # Test 8: Avg with zero active neighbors => 0, not NaN
    def test_avg_zero_active_neighbors(self):
        """Avg pooling with all neighbors inactive produces 0."""
        N = 3
        # Disconnected node 2: only 0-1 edge
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        nfp = NeighborhoodFeaturePooling(
            K=1, stride=1, aggregation='avg', include_self=False
        )

        x = torch.tensor([[[[1.0], [2.0], [3.0]]]])
        # Only node 0 active, node 2 has no active neighbors (and no self-loop)
        active_mask = torch.tensor([[True, False, False]])

        x_pooled, _ = nfp(x, edge_index, active_mask=active_mask)
        # Node 2 (include_self=False, no edge to node 0) has empty neighborhood
        # Should be 0, not NaN
        assert not torch.isnan(x_pooled).any()

    # Test 9: Adjacency cache hit and invalidation
    def test_adjacency_cache(self):
        """Avg path caches exact strided adjacency by graph content."""
        N = 4
        edge_index = _line_graph_edge_index(N)
        nfp = NeighborhoodFeaturePooling(K=1, stride=1, aggregation='avg')

        x = torch.randn(1, 1, N, 2)

        # First call: compute adjacency
        nfp(x, edge_index)
        assert len(nfp._adj_cache) == 1

        # Same tensor again: should hit cache
        nfp(x, edge_index)
        assert len(nfp._adj_cache) == 1

        # Clone with same content: should also hit (content hash matches)
        edge_index_clone = edge_index.clone()
        nfp(x, edge_index_clone)
        assert len(nfp._adj_cache) == 1

        # Different content: should miss
        edge_index_diff = _triangle_edge_index()
        x_diff = torch.randn(1, 1, 3, 2)
        nfp(x_diff, edge_index_diff)
        assert len(nfp._adj_cache) == 2

    # Test 10: K=0 + include_self=False -> identity (empty neighborhood = zeros for aggregation)
    def test_k_zero_include_self_false(self):
        """K=0 is always identity regardless of include_self."""
        N = 4
        edge_index = _line_graph_edge_index(N)
        nfp_max = NeighborhoodFeaturePooling(
            K=0, stride=1, aggregation='max', include_self=False
        )
        nfp_avg = NeighborhoodFeaturePooling(
            K=0, stride=1, aggregation='avg', include_self=False
        )

        x = torch.randn(1, 1, N, 2)
        x_out_max, _ = nfp_max(x, edge_index)
        x_out_avg, _ = nfp_avg(x, edge_index)

        # K=0 is a no-op identity
        assert torch.equal(x_out_max, x)
        assert torch.equal(x_out_avg, x)

    def test_max_matches_dense_reference(self):
        """Sparse max path matches dense A(+I)^h reference on toy graph."""
        torch.manual_seed(11)
        N = 7
        edge_index = _line_graph_edge_index(N)
        x = torch.randn(2, 3, N, 4)
        nfp = NeighborhoodFeaturePooling(K=2, stride=2, aggregation='max', include_self=True)

        x_pooled, adj = nfp(x, edge_index, return_adj=True)
        x_ref, adj_ref = _dense_within_hop_max_reference(
            x, edge_index, num_hops=4, include_self=True
        )

        assert torch.allclose(x_pooled, x_ref)
        assert torch.equal(to_dense_adj(adj, max_num_nodes=N)[0].bool(), adj_ref)

    def test_max_fallback_without_torch_scatter(self, monkeypatch):
        """Fallback scatter_reduce_ path works when torch_scatter is unavailable."""
        import graph_signal_diffusion.models.components.pooling as pooling_mod

        monkeypatch.setattr(pooling_mod, "torch_scatter_scatter", None)

        N = 6
        edge_index = _line_graph_edge_index(N)
        x = torch.randn(1, 2, N, 3)
        nfp = NeighborhoodFeaturePooling(K=1, stride=2, aggregation='max')
        x_pooled, _ = nfp(x, edge_index)
        assert x_pooled.shape == x.shape

    def test_max_toy_visualization_lattice_all_active(self):
        """Generate and save lattice toy visualization with all nodes active."""
        out_path = _save_max_toy_visualization(
            filename="nfp_lattice_max_pool_all_active.png"
        )
        assert out_path.exists()
        assert out_path.stat().st_size > 0
        assert out_path.parent == DEFAULT_VIS_OUTPUT_DIR

    def test_max_toy_visualization_lattice_with_active_mask(self):
        """Generate lattice toy visualization with predefined active_mask."""
        side = 5
        active_mask = _checkerboard_active_mask(side)
        out_path = _save_max_toy_visualization(
            active_mask=active_mask,
            filename="nfp_lattice_max_pool_with_active_mask.png",
        )
        assert out_path.exists()
        assert out_path.stat().st_size > 0
        assert out_path.parent == DEFAULT_VIS_OUTPUT_DIR

        # Masking should change pooled outputs compared to all-active case.
        _, x_pooled_all, _, _ = _compute_lattice_toy_pool(side=side, active_mask=None)
        _, x_pooled_masked, _, _ = _compute_lattice_toy_pool(
            side=side, active_mask=active_mask
        )
        assert not torch.allclose(x_pooled_all, x_pooled_masked)

    def test_max_toy_visualization_watts_strogatz(self):
        """Generate and save Watts-Strogatz toy visualization."""
        out_path = _save_watts_strogatz_max_toy_visualization(
            filename="nfp_watts_strogatz_max_pool_all_active.png"
        )
        assert out_path.exists()
        assert out_path.stat().st_size > 0
        assert out_path.parent == DEFAULT_VIS_OUTPUT_DIR

        # Ensure selected example node is visibly affected by pooling.
        x, x_pooled, _, _, _, example_node = _compute_watts_strogatz_toy_pool()
        assert x_pooled[0, 0, example_node, 0] > x[0, 0, example_node, 0]

    def test_max_toy_visualization_watts_strogatz_with_active_mask(self):
        """Generate Watts-Strogatz toy visualization with predefined active_mask."""
        num_nodes = 64
        active_mask = _watts_strogatz_active_mask(num_nodes=num_nodes, keep_every=3)
        out_path = _save_watts_strogatz_max_toy_visualization(
            active_mask=active_mask,
            filename="nfp_watts_strogatz_max_pool_with_active_mask.png",
        )
        assert out_path.exists()
        assert out_path.stat().st_size > 0
        assert out_path.parent == DEFAULT_VIS_OUTPUT_DIR

        # Masking should change pooled outputs compared to all-active case.
        _, x_pooled_all, _, _, _, _ = _compute_watts_strogatz_toy_pool(
            num_nodes=num_nodes, active_mask=None
        )
        _, x_pooled_masked, _, _, _, _ = _compute_watts_strogatz_toy_pool(
            num_nodes=num_nodes, active_mask=active_mask
        )
        assert not torch.allclose(x_pooled_all, x_pooled_masked)


# ===========================================================================
# Integration tests (with StridedGraphMaxPool)
# ===========================================================================

class TestStridedGraphMaxPoolIntegration:
    """Tests 11-18: StridedGraphMaxPool with composed NeighborhoodFeaturePooling."""

    # Test 11: K=0 path unchanged
    def test_k_zero_stride_unchanged(self):
        """K=0 stride selection produces same results as before refactor."""
        torch.manual_seed(7)
        N = 8
        pool = StridedGraphMaxPool(gamma=2, K=0, selection_method='stride')

        x = torch.randn(2, 3, N, 4)
        edge_index = _line_graph_edge_index(N)
        active_mask = torch.ones(2, N, dtype=torch.bool)
        active_mask[1, 6:] = False

        x_pooled, new_mask, sel_idx, fourth = pool(x, edge_index, active_mask=active_mask)

        assert x_pooled.shape == (2, 3, N, 4)
        assert new_mask.shape == (2, N)
        assert fourth is None  # stride selection, K=0

    # Test 12: K>0 max parity
    def test_k_gt0_max_parity(self):
        """K>0 max pooling with new pipeline matches expected behavior."""
        torch.manual_seed(99)
        N = 6
        edge_index = _line_graph_edge_index(N)

        pool = StridedGraphMaxPool(
            gamma=2, K=2, selection_method='stride',
            stride_input=1,
        )

        x = torch.randn(1, 2, N, 3)
        active_mask = torch.ones(1, N, dtype=torch.bool)

        x_pooled, new_mask, sel_idx, neighborhoods = pool(
            x, edge_index, active_mask=active_mask
        )

        assert x_pooled.shape == (1, 2, N, 3)
        # Stride selection: every 2nd node => nodes 0, 2, 4
        assert new_mask.sum().item() == 3

    # Test 13: K>0 avg produces valid output
    def test_k_gt0_avg_valid(self):
        """K>0 avg pooling produces no NaN and correct shape."""
        torch.manual_seed(42)
        N = 6
        edge_index = _line_graph_edge_index(N)

        pool = StridedGraphMaxPool(
            gamma=2, K=1, selection_method='stride',
            stride_input=1, neighborhood_pooling='avg',
        )

        x = torch.randn(1, 2, N, 4)
        active_mask = torch.ones(1, N, dtype=torch.bool)

        x_pooled, new_mask, _, _ = pool(x, edge_index, active_mask=active_mask)

        assert x_pooled.shape == (1, 2, N, 4)
        assert not torch.isnan(x_pooled).any()

    # Test 14: return_neighborhoods for stride/random
    def test_return_neighborhoods_stride(self):
        """return_neighborhoods=True returns edge_index_multi for stride selection."""
        N = 6
        edge_index = _line_graph_edge_index(N)

        pool = StridedGraphMaxPool(
            gamma=2, K=2, selection_method='stride', stride_input=1,
        )

        x = torch.randn(1, 1, N, 2)
        _, _, _, neighborhoods = pool(
            x, edge_index, return_neighborhoods=True
        )

        assert neighborhoods is not None
        assert neighborhoods.shape[0] == 2  # edge_index format (2, E)

    # Test 15: return_neighborhoods for learned = scores
    def test_return_neighborhoods_learned_returns_scores(self):
        """Learned selection returns scores in 4th element, not neighborhoods."""
        N = 6
        edge_index = _line_graph_edge_index(N)

        pool = StridedGraphMaxPool(
            gamma=2, K=1, selection_method='learned',
            in_channels=4, stride_input=1,
        )

        x = torch.randn(1, 2, N, 4)
        _, _, _, fourth = pool(
            x, edge_index, return_neighborhoods=True
        )

        # Should be scores (B, N) tensor, not neighborhoods edge_index
        assert fourth is not None
        assert fourth.dim() == 2  # (B, N) scores

    # Test 16: Compatibility shim monkeypatch (existing test pattern)
    def test_k_zero_fast_path_skips_adjacency_and_maxpool(self):
        """K=0 fast path bypasses adjacency/max-pooling helpers (monkeypatch test)."""
        pool = StridedGraphMaxPool(gamma=2, K=0, selection_method='stride')

        def _should_not_be_called(*args, **kwargs):
            raise AssertionError("Adjacency/max-pool helpers must not be called when K=0")

        pool._compute_strided_hop_adjacency_boolean = _should_not_be_called
        pool._compute_strided_hop_adjacency = _should_not_be_called
        pool._max_pool_neighborhoods = _should_not_be_called

        B, T, N, F = 2, 3, 8, 4
        x = torch.randn(B, T, N, F)
        edge_index = torch.randint(0, N, (2, 24))
        active_mask = torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 0, 0],
                [0, 1, 1, 0, 1, 0, 0, 1],
            ],
            dtype=torch.bool,
        )

        x_pooled, new_mask, _, neighborhoods = pool(
            x, edge_index,
            active_mask=active_mask,
            return_neighborhoods=True,
        )

        assert neighborhoods is None
        # Selected nodes for stride=2:
        # batch 0: active=[0,1,2,3,4,5], select every 2nd -> [0,2,4]
        # batch 1: active=[1,2,4,7], select every 2nd -> [1,4]
        expected_mask = torch.zeros_like(active_mask)
        expected_mask[0, [0, 2, 4]] = True
        expected_mask[1, [1, 4]] = True

        expected_x = x * expected_mask.unsqueeze(1).unsqueeze(-1).float()

        assert torch.equal(new_mask, expected_mask)
        assert torch.allclose(x_pooled, expected_x)

    # Test 17: Direct shim calls at K=0 are callable
    def test_shim_calls_at_k_zero(self):
        """Compatibility shims work even when K=0 (no self.neighborhood_agg)."""
        N = 4
        edge_index = _line_graph_edge_index(N)
        pool = StridedGraphMaxPool(gamma=2, K=0, selection_method='stride')

        # These should not raise — they create a fresh helper internally
        adj_bool = pool._compute_strided_hop_adjacency_boolean(
            edge_index, N, 1, 2
        )
        assert adj_bool.shape[0] == 2

        adj_float = pool._compute_strided_hop_adjacency(
            edge_index, N, 1, 2
        )
        assert adj_float.shape[0] == 2

        x = torch.randn(1, 1, N, 2)
        x_pooled = pool._max_pool_neighborhoods(x, adj_bool, N)
        assert x_pooled.shape == x.shape

    # Test 18: Memory guard two-sided
    def test_memory_guard(self):
        """Avg path honors dense memory guard while max path bypasses it."""
        N = 4
        edge_index = _line_graph_edge_index(N)

        # Very generous budget on avg: should not raise
        nfp_ok = NeighborhoodFeaturePooling(
            K=1, stride=1, aggregation='avg', max_dense_gb=100.0
        )
        x = torch.randn(1, 1, N, 2)
        nfp_ok(x, edge_index)  # Should not raise

        # Tiny budget on avg: should raise
        nfp_bad = NeighborhoodFeaturePooling(
            K=1, stride=1, aggregation='avg', max_dense_gb=1e-12
        )
        with pytest.raises(RuntimeError, match="NeighborhoodFeaturePooling"):
            nfp_bad(x, edge_index)

        # Max path should ignore dense memory guard.
        nfp_max = NeighborhoodFeaturePooling(
            K=2, stride=2, aggregation='max', max_dense_gb=1e-12
        )
        x_max, _ = nfp_max(x, edge_index)
        assert x_max.shape == x.shape

    # Test: neighborhood_agg attribute exists at K>0 and is None at K=0
    def test_neighborhood_agg_attribute(self):
        """Verify neighborhood_agg is properly composed."""
        pool_k0 = StridedGraphMaxPool(gamma=2, K=0, selection_method='stride')
        pool_k2 = StridedGraphMaxPool(gamma=2, K=2, selection_method='stride')

        assert pool_k0.neighborhood_agg is None
        assert isinstance(pool_k2.neighborhood_agg, NeighborhoodFeaturePooling)

    # Test: selector attribute still accessible for learned selection
    def test_selector_attribute_accessible(self):
        """block.pool.selector access pattern still works."""
        pool = StridedGraphMaxPool(
            gamma=2, K=1, selection_method='learned',
            in_channels=16, stride_input=1,
        )
        assert pool.selector is not None

    # Test: return_adj defaults to None unless explicitly requested
    def test_return_adj_default_none(self):
        N = 6
        edge_index = _line_graph_edge_index(N)
        x = torch.randn(1, 1, N, 2)

        nfp_max = NeighborhoodFeaturePooling(K=1, stride=2, aggregation='max')
        nfp_avg = NeighborhoodFeaturePooling(K=1, stride=1, aggregation='avg')

        _, adj_max = nfp_max(x, edge_index, return_adj=False)
        _, adj_avg = nfp_avg(x, edge_index, return_adj=False)

        assert adj_max is None
        assert adj_avg is None
