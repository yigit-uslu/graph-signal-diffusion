"""Smoke tests for per-layer active masking in GNN."""

import torch

from graph_signal_diffusion.models.components.graph_conv import GNN


def _set_propagation_only_weights(gnn: GNN) -> None:
    """Configure each TAGConv as pure 1-hop propagation with zero bias."""
    with torch.no_grad():
        for block in gnn.layers:
            tagconv = block.conv.conv  # TAGConv inside TAGConvLayer
            # K=1 => lins[0] for 0-hop, lins[1] for 1-hop.
            tagconv.lins[0].weight.zero_()
            tagconv.lins[1].weight.fill_(1.0)
            if tagconv.bias is not None:
                tagconv.bias.zero_()


def test_gnn_active_mask_prevents_inactive_node_relay_smoke():
    """
    Verify masking after each residual block prevents inactive nodes from
    relaying messages across stacked layers.
    """
    B, T, N, F = 1, 5, 3, 1

    # Directed chain 0 -> 1 -> 2.
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

    # Only node 0 and node 2 are active; node 1 is inactive.
    active_mask = torch.tensor([[True, False, True]], dtype=torch.bool)

    # Signal starts only at node 0.
    x = torch.zeros(B, T, N, F, dtype=torch.float32)
    x[:, :, 0, 0] = 1.0

    gnn = GNN(
        in_channels=F,
        hidden_channels=F,
        out_channels=F,
        num_layers=2,
        K=1,
        norm_type="none",
        dropout=0.0,
        activation="relu",
        use_input_proj=False,
        use_output_proj=False,
        normalize=False,
    )
    _set_propagation_only_weights(gnn)

    y_no_mask = gnn(x, edge_index)
    y_masked = gnn(x, edge_index, active_mask=active_mask)

    # Without per-layer masking, inactive node 1 can relay to active node 2
    # by the second block, so node 2 becomes non-zero for at least one timestep.
    assert torch.max(y_no_mask[0, :, 2, 0]) > 0.0

    # With per-layer masking, node 1 is zeroed after each block and cannot relay.
    torch.testing.assert_close(
        y_masked[0, :, 1, 0],
        torch.zeros(T, dtype=y_masked.dtype),
        atol=1e-7,
        rtol=0.0,
    )
    torch.testing.assert_close(
        y_masked[0, :, 2, 0],
        torch.zeros(T, dtype=y_masked.dtype),
        atol=1e-7,
        rtol=0.0,
    )
