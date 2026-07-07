"""Unit tests for v3-aware NodeSelector-based learned selection."""

import torch
import pytest

from graph_signal_diffusion.models.components.pooling import StridedGraphMaxPool
from graph_signal_diffusion.models.components.node_selector import NodeSelector


def _chain_edge_index(num_nodes: int) -> torch.Tensor:
    edges = [(i, i + 1) for i in range(num_nodes - 1)]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return torch.cat([edge_index, edge_index.flip(0)], dim=1)


def test_v3_node_selector_topk_respects_min_retained_nodes():
    torch.manual_seed(0)
    B, T, N, F = 2, 4, 16, 8
    x = torch.randn(B, T, N, F)
    edge_index = _chain_edge_index(N)
    active_mask = torch.zeros(B, N, dtype=torch.bool)
    active_mask[0, :9] = True  # 9 active -> floor(9*0.4)=3, floor to 4
    active_mask[1, :4] = True  # 4 active -> floor(4*0.4)=1, floor to 4 then clamp to 4

    pool = StridedGraphMaxPool(
        gamma=2,
        K=0,
        selection_method='learned',
        in_channels=F,
        selector_kwargs={
            'selector_version': 'v3',
            'selection_mode': 'hard',
            'pooling_ratio': 0.4,
            'min_retained_nodes': 4,
            'cond_fusion': False,
        },
        stride_input=1,
    )
    pool.eval()

    _, new_mask, _, scores = pool(x, edge_index, active_mask=active_mask)
    assert new_mask.shape == (B, N)
    assert scores is not None
    assert scores.shape == (B, N)
    assert int(new_mask[0].sum().item()) == 4
    assert int(new_mask[1].sum().item()) == 4


def test_v3_node_selector_zero_active_graph_keeps_no_selection():
    torch.manual_seed(0)
    B, T, N, F = 3, 4, 12, 8
    x = torch.randn(B, T, N, F)
    edge_index = _chain_edge_index(N)
    active_mask = torch.zeros(B, N, dtype=torch.bool)

    pool = StridedGraphMaxPool(
        gamma=2,
        K=0,
        selection_method='learned',
        in_channels=F,
        selector_kwargs={
            'selector_version': 'v3',
            'selection_mode': 'hard',
            'pooling_ratio': 0.4,
            'min_retained_nodes': 1,
            'cond_fusion': False,
        },
        stride_input=1,
    )
    pool.eval()

    _, new_mask, selected_indices, scores = pool(x, edge_index, active_mask=active_mask)
    assert new_mask.shape == (B, N)
    assert new_mask.sum() == 0
    assert scores is not None
    assert scores.shape == (B, N)
    assert selected_indices.shape[1] == 0


def test_v3_node_selector_batched_topk_matches_per_sample_topk():
    """Batched helper must preserve exact per-sample top-k when k varies by row."""
    selector = NodeSelector(
        in_channels=8,
        pooling_ratio=0.5,
        selection_mode='hard',
        cond_fusion=False,
    )

    # Fixed seed reproduces a case where unsorted batched top-k can mis-route
    # the first k[b] positions for rows with smaller k.
    torch.manual_seed(0)
    scores = torch.randn(3, 10)
    k = torch.tensor([2, 5, 7], dtype=torch.long)

    top_k_indices = selector._batched_topk_indices(scores, k)
    assert top_k_indices.shape == (3, int(k.max().item()))

    for b in range(scores.size(0)):
        k_b = int(k[b].item())
        got = set(top_k_indices[b, :k_b].tolist())
        expected = set(
            torch.topk(
                scores[b],
                k=k_b,
                dim=0,
                largest=True,
                sorted=True,
            ).indices.tolist()
        )
        assert got == expected


def test_v3_node_selector_conditioning_changes_scores():
    torch.manual_seed(0)
    B, T, N, F = 1, 3, 10, 6
    x = torch.randn(B, T, N, F)
    edge_index = _chain_edge_index(N)
    active_mask = torch.ones(B, N, dtype=torch.bool)

    pool = StridedGraphMaxPool(
        gamma=2,
        K=0,
        selection_method='learned',
        in_channels=F,
        selector_kwargs={
            'selector_version': 'v3',
            'selection_mode': 'hard',
            'cond_fusion': True,
            'cond_fusion_heads': 2,
        },
        stride_input=1,
    )
    pool.eval()

    # Build condition tensor in the same shape as the selector conditioning interface.
    cond_zero = torch.zeros(B, T, N, F)
    cond_node_profile = torch.zeros(B, T, N, F)
    node_weights = torch.linspace(0.0, 1.0, N, dtype=cond_node_profile.dtype).view(1, 1, N, 1)
    cond_node_profile = cond_node_profile + node_weights

    selector = pool.selector
    scores_no_cond = selector.compute_scores(
        x=x,
        edge_index=edge_index,
        active_mask=active_mask,
        cond=cond_zero,
    )
    scores_with_cond = selector.compute_scores(
        x=x,
        edge_index=edge_index,
        active_mask=active_mask,
        cond=cond_node_profile,
    )

    assert scores_no_cond.shape == (B, N)
    assert scores_with_cond.shape == (B, N)
    assert not torch.allclose(scores_no_cond, scores_with_cond)


def test_v3_encoder_block_intermediates_capture_selector_scores_once():
    torch.manual_seed(0)
    pytest.importorskip("beartype")

    from graph_signal_diffusion.models.ugnn import (
        EmbeddingConfig,
        EncoderBlock,
        GNNConfig,
        PoolingConfig,
    )

    block = EncoderBlock(
        in_channels=8,
        out_channels=8,
        stride_pre=1,
        stride_post=2,
        gnn_config=GNNConfig(
            K=1,
            num_layers=1,
            norm_type='none',
            dropout=0.0,
            activation='relu',
        ),
        pooling_config=PoolingConfig(
            gamma=2,
            pool_K=0,
            selection_method='learned',
            selector_version='v3',
            selector_kwargs={
                'selection_mode': 'hard',
                'pooling_ratio': 0.5,
                'cond_fusion': False,
            },
        ),
        embedding_config=EmbeddingConfig(cond_channels=None, time_embed_dim=16),
    )

    B, T, N = 2, 4, 12
    x = torch.randn(B, T, N, 8)
    timesteps = torch.randint(0, 50, (B,), dtype=torch.long)
    time_emb = torch.randn(B, 16)
    edge_index = _chain_edge_index(N)
    active_mask = torch.ones(B, N, dtype=torch.bool)

    # Spy on compute_scores call count to ensure we don't run it a second time in encoder.
    selector = block.pool.selector
    calls = {"count": 0}
    original_compute_scores = selector.compute_scores

    def _spy_compute(*args, **kwargs):
        calls["count"] += 1
        return original_compute_scores(*args, **kwargs)

    selector.compute_scores = _spy_compute  # type: ignore[assignment]

    try:
        _, _, _, intermediates = block(
            x=x,
            timesteps=timesteps,
            edge_index=edge_index,
            time_emb=time_emb,
            active_mask=active_mask,
            return_intermediates=True,
        )
    finally:
        selector.compute_scores = original_compute_scores

    assert intermediates is not None
    assert intermediates['selection_scores'] is not None
    assert intermediates['selection_scores'].shape == (B, N)
    assert calls["count"] == 1
