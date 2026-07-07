"""Focused tests for NodeSelector ``cond_fusion_mode='add'`` behavior."""

import torch
import pytest

from graph_signal_diffusion.models.components.node_selector import NodeSelector
from graph_signal_diffusion.models.components.pooling import StridedGraphMaxPool


def _chain_edge_index(num_nodes: int) -> torch.Tensor:
    edges = [(i, i + 1) for i in range(num_nodes - 1)]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return torch.cat([edge_index, edge_index.flip(0)], dim=1)


def test_add_mode_zero_init_is_cond_identity():
    torch.manual_seed(0)
    B, T, N, C = 2, 4, 9, 6
    x = torch.randn(B, T, N, C)
    edge_index = _chain_edge_index(N)
    active_mask = torch.ones(B, N, dtype=torch.bool)
    cond_a = torch.randn(B, T, N, C)
    cond_b = torch.randn(B, T, N, C)

    selector = NodeSelector(
        in_channels=C,
        pooling_ratio=0.5,
        selection_mode='hard',
        cond_fusion_mode='add',
    )
    selector.eval()

    scores_a = selector.compute_scores(
        x=x,
        edge_index=edge_index,
        active_mask=active_mask,
        cond=cond_a,
    )
    scores_b = selector.compute_scores(
        x=x,
        edge_index=edge_index,
        active_mask=active_mask,
        cond=cond_b,
    )
    scores_none = selector.compute_scores(
        x=x,
        edge_index=edge_index,
        active_mask=active_mask,
        cond=None,
    )

    assert torch.allclose(scores_a, scores_b)
    assert torch.allclose(scores_a, scores_none)


def test_add_mode_nonzero_bias_changes_scores():
    torch.manual_seed(0)
    B, T, N, C = 1, 3, 10, 8
    x = torch.randn(B, T, N, C)
    edge_index = _chain_edge_index(N)
    active_mask = torch.ones(B, N, dtype=torch.bool)
    cond_zero = torch.zeros(B, T, N, C)
    cond_profile = torch.linspace(0.0, 1.0, N).view(1, 1, N, 1).expand(B, T, N, C)

    selector = NodeSelector(
        in_channels=C,
        pooling_ratio=0.5,
        selection_mode='hard',
        cond_fusion_mode='add',
    )
    selector.eval()

    assert selector.cond_add_bias is not None
    with torch.no_grad():
        # Make selector score projection deterministic so additive cond bias
        # reliably changes final logits after score projection.
        selector.score_proj.weight.fill_(1.0)
        selector.score_proj.bias.zero_()
        selector.cond_add_bias.weight.fill_(0.25)
        selector.cond_add_bias.bias.zero_()

    scores_zero = selector.compute_scores(
        x=x,
        edge_index=edge_index,
        active_mask=active_mask,
        cond=cond_zero,
    )
    scores_profile = selector.compute_scores(
        x=x,
        edge_index=edge_index,
        active_mask=active_mask,
        cond=cond_profile,
    )

    assert not torch.allclose(scores_zero, scores_profile)


@pytest.mark.parametrize("cond_shape", ["btnc", "bt1c", "bnc", "btc", "bc"])
def test_add_mode_supports_common_cond_shapes_with_projection(cond_shape: str):
    torch.manual_seed(0)
    B, T, N = 2, 3, 8
    in_c, cond_c = 6, 4
    x = torch.randn(B, T, N, in_c)
    edge_index = _chain_edge_index(N)
    active_mask = torch.ones(B, N, dtype=torch.bool)

    if cond_shape == "btnc":
        cond = torch.randn(B, T, N, cond_c)
    elif cond_shape == "bt1c":
        cond = torch.randn(B, T, 1, cond_c)
    elif cond_shape == "bnc":
        cond = torch.randn(B, N, cond_c)
    elif cond_shape == "btc":
        cond = torch.randn(B, T, cond_c)
    elif cond_shape == "bc":
        cond = torch.randn(B, cond_c)
    else:
        raise AssertionError(f"Unhandled cond_shape={cond_shape}")

    selector = NodeSelector(
        in_channels=in_c,
        pooling_ratio=0.5,
        selection_mode='hard',
        cond_fusion_mode='add',
        cond_dim=cond_c,
    )
    selector.eval()

    scores = selector.compute_scores(
        x=x,
        edge_index=edge_index,
        active_mask=active_mask,
        cond=cond,
    )

    assert scores.shape == (B, N)


def test_v3_pool_accepts_add_cond_fusion_mode():
    torch.manual_seed(0)
    B, T, N, F = 2, 4, 12, 8
    x = torch.randn(B, T, N, F)
    edge_index = _chain_edge_index(N)
    active_mask = torch.ones(B, N, dtype=torch.bool)
    cond = torch.randn(B, T, N, F)

    pool = StridedGraphMaxPool(
        gamma=2,
        K=0,
        selection_method='learned',
        in_channels=F,
        selector_kwargs={
            'selector_version': 'v3',
            'selection_mode': 'hard',
            'pooling_ratio': 0.5,
            'cond_fusion_mode': 'add',
        },
        stride_input=1,
    )
    pool.eval()

    assert pool.selector is not None
    assert isinstance(pool.selector, NodeSelector)
    assert pool.selector.cond_fusion_mode == 'add'

    _, new_mask, selected_indices, scores = pool(
        x=x,
        edge_index=edge_index,
        active_mask=active_mask,
        cond=cond,
    )
    assert new_mask.shape == (B, N)
    assert selected_indices is not None
    assert selected_indices.shape[0] == B
    assert scores is not None
    assert scores.shape == (B, N)


def test_v3_pool_add_mode_projects_mismatched_cond_width():
    torch.manual_seed(0)
    B, T, N = 2, 3, 10
    selector_c = 16
    cond_c = 8
    x = torch.randn(B, T, N, selector_c)
    cond = torch.randn(B, T, N, cond_c)
    edge_index = _chain_edge_index(N)
    active_mask = torch.ones(B, N, dtype=torch.bool)

    pool = StridedGraphMaxPool(
        gamma=2,
        K=0,
        selection_method='learned',
        in_channels=selector_c,
        selector_kwargs={
            'selector_version': 'v3',
            'selection_mode': 'hard',
            'pooling_ratio': 0.5,
            'cond_fusion_mode': 'add',
            'cond_dim': cond_c,
        },
        stride_input=1,
    )
    pool.eval()

    assert pool.selector is not None
    assert pool.selector.cond_projection is not None
    assert hasattr(pool.selector.cond_projection, "weight")
    assert pool.selector.cond_projection.weight.shape == (selector_c, cond_c)

    _, new_mask, selected_indices, scores = pool(
        x=x,
        edge_index=edge_index,
        active_mask=active_mask,
        cond=cond,
    )
    assert new_mask.shape == (B, N)
    assert selected_indices is not None
    assert scores is not None
    assert scores.shape == (B, N)
