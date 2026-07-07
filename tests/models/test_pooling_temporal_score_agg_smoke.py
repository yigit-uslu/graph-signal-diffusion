"""Smoke tests for temporal score aggregation modes in learned pooling."""

import torch

from graph_signal_diffusion.models.components.pooling import LearnableGraphPool
from graph_signal_diffusion.models.ugnn import EmbeddingConfig, PoolingConfig, UGNNConfig, UGNNEncoder


def _tiny_edge_index(num_nodes: int) -> torch.Tensor:
    """Build an undirected chain edge_index with 2 x E shape."""
    edges = [(i, i + 1) for i in range(num_nodes - 1)]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return torch.cat([edge_index, edge_index.flip(0)], dim=1)


def test_temporal_score_agg_modes_smoke():
    B, T, N, F = 2, 5, 8, 4
    x = torch.randn(B, T, N, F)
    edge_index = _tiny_edge_index(N)
    active_mask = torch.ones(B, N, dtype=torch.bool)

    for mode in ("mean", "last", "ema", "attention"):
        pool = LearnableGraphPool(
            in_channels=F,
            pooling_ratio=0.5,
            gamma=1,
            temporal_score_agg=mode,
            ema_alpha_init=0.7,
        )
        x_pooled, new_mask, top_k_indices, scores = pool(
            x=x,
            edge_index=edge_index,
            active_mask=active_mask,
        )
        assert x_pooled.shape == (B, T, N, F)
        assert new_mask.shape == (B, N)
        assert top_k_indices.shape[0] == B
        assert scores.shape == (B, N)


def test_ema_alpha_is_learnable_smoke():
    B, T, N, F = 2, 5, 8, 4
    x = torch.randn(B, T, N, F)
    edge_index = _tiny_edge_index(N)
    active_mask = torch.ones(B, N, dtype=torch.bool)

    pool = LearnableGraphPool(
        in_channels=F,
        pooling_ratio=0.5,
        gamma=1,
        temporal_score_agg="ema",
        ema_alpha_init=0.6,
    )
    assert pool.pooling_gnn.ema_alpha_logit is not None
    assert pool.pooling_gnn.ema_alpha_logit.requires_grad

    x_pooled, _, _, _ = pool(
        x=x,
        edge_index=edge_index,
        active_mask=active_mask,
    )
    loss = x_pooled.sum()
    loss.backward()
    assert pool.pooling_gnn.ema_alpha_logit.grad is not None
    assert pool.pooling_gnn.ema_alpha_logit.grad.abs().sum() > 0


def test_ugnn_pooling_config_forwards_temporal_score_agg_smoke():
    config = UGNNConfig(
        in_channels=1,
        out_channels=1,
        base_channels=8,
        channel_multipliers=[1, 2],
        embedding_config=EmbeddingConfig(time_embed_dim=16),
        pooling_config=PoolingConfig(
            gamma=2,
            pool_K=1,
            selection_method="learned",
            temporal_score_agg="ema",
            ema_alpha_init=0.65,
        ),
    )
    encoder = UGNNEncoder(in_channels=1, config=config)
    first_block = encoder.encoder_blocks[0]
    assert first_block.pool.selection_method == "learned"
    assert first_block.pool.selector is not None
    assert first_block.pool.selector.pooling_gnn.temporal_score_agg == "ema"
    assert first_block.pool.selector.pooling_gnn.ema_alpha_logit is not None
