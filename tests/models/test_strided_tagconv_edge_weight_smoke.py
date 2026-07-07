"""Smoke tests for StridedTAGConv edge-weight support and gamma=1 parity."""

from pathlib import Path

import matplotlib
import torch

from graph_signal_diffusion.models.components.graph_conv import TAGConvLayer

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _build_batched_tiny_graph(num_graphs: int, nodes_per_graph: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a tiny directed cycle graph with self-loops, batched by index offsets."""
    edge_index_single = torch.tensor(
        [
            [0, 1, 2, 0, 1, 2],  # source
            [1, 2, 0, 0, 1, 2],  # target
        ],
        dtype=torch.long,
    )
    edge_weight_single = torch.tensor([0.7, 1.3, 0.5, 1.0, 1.0, 1.0], dtype=torch.float32)

    edge_index_parts = []
    edge_weight_parts = []
    for b in range(num_graphs):
        offset = b * nodes_per_graph
        edge_index_parts.append(edge_index_single + offset)
        # Slightly vary the second graph weights so batching isn't degenerate.
        edge_weight_parts.append(edge_weight_single + 0.1 * b)

    edge_index = torch.cat(edge_index_parts, dim=1)
    edge_weight = torch.cat(edge_weight_parts, dim=0)
    return edge_index, edge_weight


def test_strided_tagconv_edge_weights_and_temporal_replication_smoke():
    """
    Verify strided conv:
    1) consumes edge weights,
    2) supports T=5 temporal inputs, and
    3) has consistent 4D vs 3D temporal replication behavior.
    """
    torch.manual_seed(7)

    B, T, N, F = 2, 5, 3, 1
    edge_index, edge_weight = _build_batched_tiny_graph(num_graphs=B, nodes_per_graph=N)
    x = torch.randn(B, T, N, F)

    layer = TAGConvLayer(
        in_channels=F,
        out_channels=1,
        K=2,
        normalize=False,
        use_strided=True,
        gamma=2,
    )

    out_weighted = layer(x, edge_index, edge_weight=edge_weight)
    out_unweighted = layer(x, edge_index, edge_weight=None)

    assert out_weighted.shape == (B, T, N, 1)
    assert out_unweighted.shape == (B, T, N, 1)
    assert not torch.allclose(out_weighted, out_unweighted, atol=1e-8, rtol=0.0)

    # Compare 4D path against 3D path: they should match exactly.
    x_3d = x.permute(0, 2, 1, 3).reshape(B * N, T, F)
    out_3d = layer(x_3d, edge_index, edge_weight=edge_weight)
    out_3d_as_4d = out_3d.reshape(B, N, T, 1).permute(0, 2, 1, 3)

    torch.testing.assert_close(out_weighted, out_3d_as_4d, atol=1e-7, rtol=0.0)


def test_strided_tagconv_gamma1_matches_regular_tagconv_smoke():
    """
    Verify StridedTAGConv (gamma=1) coincides with regular TAGConv when
    parameters are matched and edge weights are provided.
    """
    torch.manual_seed(11)

    B, T, N, F = 1, 5, 4, 1
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 0, 2],
            [1, 2, 3, 0, 2, 1],
        ],
        dtype=torch.long,
    )
    edge_weight = torch.tensor([0.5, 1.2, 0.7, 1.1, 0.9, 0.3], dtype=torch.float32)
    x = torch.randn(B, T, N, F)

    # Explicit normalize=False since this repo sets TAGConvLayer default to False.
    regular = TAGConvLayer(
        in_channels=F,
        out_channels=1,
        K=2,
        normalize=False,
        use_strided=False,
    )
    strided = TAGConvLayer(
        in_channels=F,
        out_channels=1,
        K=2,
        normalize=False,
        use_strided=True,
        gamma=1,
    )

    # Align parameters exactly.
    with torch.no_grad():
        for k in range(3):
            strided.conv.lins[k].weight.copy_(regular.conv.lins[k].weight)
        if regular.conv.bias is not None:
            strided.conv.bias.copy_(regular.conv.bias)

    y_regular = regular(x, edge_index, edge_weight=edge_weight)
    y_strided = strided(x, edge_index, edge_weight=edge_weight)

    torch.testing.assert_close(y_regular, y_strided, atol=1e-7, rtol=0.0)


def test_tagconv_gamma1_visual_compare_all_ones_filters():
    """
    Visualize and compare regular TAGConv vs StridedTAGConv(gamma=1)
    with all filter weights set to 1 (normalize=False).
    """
    B, T, N, F = 1, 5, 4, 1

    # Small directed cycle + shortcut + self-loops.
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 0, 2, 0, 1, 2, 3],
            [1, 2, 3, 0, 2, 1, 0, 1, 2, 3],
        ],
        dtype=torch.long,
    )
    edge_weight = torch.tensor([1.0, 0.6, 1.2, 0.8, 0.7, 0.5, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32)

    # Deterministic temporal signal: (B=1, T=5, N=4, F=1)
    x_vals = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0],
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
            [4.0, 5.0, 6.0, 7.0],
        ],
        dtype=torch.float32,
    )
    x = x_vals.view(B, T, N, F)

    regular = TAGConvLayer(
        in_channels=F,
        out_channels=1,
        K=2,
        normalize=False,
        use_strided=False,
    )
    strided = TAGConvLayer(
        in_channels=F,
        out_channels=1,
        K=2,
        normalize=False,
        use_strided=True,
        gamma=1,
    )

    # Force identical "all ones" TAG filters and zero bias.
    with torch.no_grad():
        for k in range(3):
            regular.conv.lins[k].weight.fill_(1.0)
            strided.conv.lins[k].weight.fill_(1.0)
        if regular.conv.bias is not None:
            regular.conv.bias.zero_()
        if strided.conv.bias is not None:
            strided.conv.bias.zero_()

    y_regular = regular(x, edge_index, edge_weight=edge_weight)  # (1,5,4,1)
    y_strided = strided(x, edge_index, edge_weight=edge_weight)  # (1,5,4,1)

    # Numerical equivalence check.
    torch.testing.assert_close(y_regular, y_strided, atol=1e-7, rtol=0.0)

    # Save a visual comparison (time x nodes).
    x_img = x[0, :, :, 0].detach().cpu().numpy()
    reg_img = y_regular[0, :, :, 0].detach().cpu().numpy()
    str_img = y_strided[0, :, :, 0].detach().cpu().numpy()
    diff_img = (y_regular - y_strided)[0, :, :, 0].detach().cpu().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), constrained_layout=True)
    images = [
        (x_img, "Input (T x N)"),
        (reg_img, "Regular TAGConv"),
        (str_img, "Strided TAGConv (gamma=1)"),
        (diff_img, "Difference"),
    ]
    for ax, (img, title) in zip(axes, images):
        im = ax.imshow(img, aspect="auto", origin="lower")
        ax.set_title(title)
        ax.set_xlabel("Node")
        ax.set_ylabel("Time")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    output_dir = Path("tests/figs/strided_graph_conv_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "tagconv_gamma1_all_ones_compare.png", dpi=150)
    plt.close(fig)
