import pytest
import torch

from graph_signal_diffusion.models.ugnn import (
    EmbeddingConfig,
    EncoderBlock,
    GNNConfig,
    PoolingConfig,
    UGNNConfig,
    UGNNEncoder,
)


def _chain_edge_index(num_nodes: int) -> torch.Tensor:
    """Return undirected chain edges (2, E)."""
    src = torch.arange(num_nodes - 1, dtype=torch.long)
    dst = src + 1
    return torch.stack(
        [torch.cat([src, dst], dim=0), torch.cat([dst, src], dim=0)],
        dim=0,
    )


def test_score_gain_per_level_requires_exact_active_selector_count():
    cfg = UGNNConfig(
        in_channels=1,
        out_channels=1,
        base_channels=8,
        channel_multipliers=[1, 1, 1, 1],
        gnn_config=GNNConfig(),
        pooling_config=PoolingConfig(
            gamma=[1, 2, 2, 2],
            pool_K=0,
            selection_method='learned',
            selector_kwargs={'use_pooling_gnn': False, 'hidden_channels': 8},
            score_gain_per_level=[1.0, 10.0],  # should be length 3
        ),
        embedding_config=EmbeddingConfig(cond_channels=None),
    )

    with pytest.raises(ValueError, match="score_gain_per_level"):
        UGNNEncoder(in_channels=1, config=cfg)


def test_encoder_intermediate_selection_scores_are_post_gain_logits():
    torch.manual_seed(0)

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
            selector_kwargs={
                'use_pooling_gnn': False,
                'hidden_channels': 8,
                'selection_mode': 'ste',
                'temperature': 1.0,
                'temperature_schedule': 'constant',
                'score_gain_init': 7.0,
            },
        ),
        embedding_config=EmbeddingConfig(time_embed_dim=16, cond_channels=None),
    )
    block.eval()

    B, T, N, C = 2, 4, 10, 8
    x = torch.randn(B, T, N, C)
    timesteps = torch.randint(0, 50, (B,), dtype=torch.long)
    time_emb = torch.randn(B, 16)
    edge_index = _chain_edge_index(N)
    active_mask = torch.ones(B, N, dtype=torch.bool)
    active_mask[:, -2:] = False

    _, _, _, intermediates = block(
        x=x,
        timesteps=timesteps,
        edge_index=edge_index,
        time_emb=time_emb,
        active_mask=active_mask,
        return_intermediates=True,
    )

    assert intermediates is not None
    selection_scores = intermediates['selection_scores']
    assert selection_scores is not None

    selector = block.pool.selector
    with torch.no_grad():
        raw_scores = selector.compute_scores(
            x=intermediates['gnn_output'],
            edge_index=edge_index,
            edge_weight=None,
            active_mask=active_mask,
        )
        expected_scores = raw_scores * torch.exp(selector.log_score_gain).to(
            dtype=raw_scores.dtype,
            device=raw_scores.device,
        )

    torch.testing.assert_close(selection_scores, expected_scores, atol=1e-6, rtol=1e-5)
