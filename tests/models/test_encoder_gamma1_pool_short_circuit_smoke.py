"""Smoke tests for EncoderBlock gamma=1 pooling behavior."""

import pytest
import torch
import torch.nn as nn

from graph_signal_diffusion.models.ugnn import (
    EmbeddingConfig,
    EncoderBlock,
    GNNConfig,
    PoolingConfig,
)


class _FailIfCalledPool(nn.Module):
    """Test double that fails if pooling path is executed."""

    selection_method = "learned"

    def forward(self, *args, **kwargs):
        raise RuntimeError("pool should not be called")


class _FailIfCalledSelector(nn.Module):
    """Test double that fails if selector stage is executed."""

    def forward(self, *args, **kwargs):
        raise RuntimeError("selector should not be called")


class _TrackPool(nn.Module):
    """Wrap a pool module and record whether forward was called."""

    def __init__(self, pool: nn.Module):
        super().__init__()
        self.pool = pool
        self.called = False
        self.selection_method = pool.selection_method
        self.K = pool.K

    def forward(self, *args, **kwargs):
        self.called = True
        return self.pool(*args, **kwargs)


def _build_encoder_block(
    gamma: int,
    pool_K: int = 1,
    selection_method: str = "learned",
    stride_pre: int = 1,
) -> EncoderBlock:
    gnn_config = GNNConfig(K=1, num_layers=1, norm_type="none", dropout=0.0, activation="relu")
    pooling_config = PoolingConfig(gamma=gamma, pool_K=pool_K, selection_method=selection_method)
    embedding_config = EmbeddingConfig(time_embed_dim=16, cond_channels=None)
    return EncoderBlock(
        in_channels=8,
        out_channels=8,
        stride_pre=stride_pre,
        stride_post=stride_pre * gamma,
        gnn_config=gnn_config,
        pooling_config=pooling_config,
        embedding_config=embedding_config,
    )


def test_encoder_gamma1_learned_poolk1_calls_pool_and_keeps_mask_smoke():
    torch.manual_seed(0)
    B, T, N, C = 2, 5, 6, 8

    block = _build_encoder_block(gamma=1, pool_K=1, selection_method="learned")
    tracking_pool = _TrackPool(block.pool)
    block.pool = tracking_pool

    x = torch.randn(B, T, N, C)
    timesteps = torch.randint(0, 1000, (B,), dtype=torch.long)
    time_emb = torch.randn(B, 16)
    edge_index = torch.tensor([[0, 1, 2, 6, 7, 8], [1, 2, 0, 7, 8, 6]], dtype=torch.long)
    active_mask = torch.tensor(
        [
            [True, True, False, True, False, True],
            [True, False, True, True, True, False],
        ],
        dtype=torch.bool,
    )

    x_pooled, new_mask, x_skip, _ = block(
        x=x,
        timesteps=timesteps,
        edge_index=edge_index,
        time_emb=time_emb,
        active_mask=active_mask,
    )

    assert tracking_pool.called is True
    torch.testing.assert_close(new_mask, active_mask)
    assert x_pooled.shape == x_skip.shape

    # Also validate default mask behavior when active_mask is omitted.
    x_pooled_all, new_mask_all, x_skip_all, _ = block(
        x=x,
        timesteps=timesteps,
        edge_index=edge_index,
        time_emb=time_emb,
        active_mask=None,
    )
    assert x_pooled_all.shape == x_skip_all.shape
    assert new_mask_all.dtype == torch.bool
    assert torch.all(new_mask_all)


def test_encoder_gamma1_stride_poolk0_strict_identity_smoke():
    torch.manual_seed(0)
    B, T, N, C = 1, 3, 4, 8

    block = _build_encoder_block(gamma=1, pool_K=0, selection_method="stride")
    tracking_pool = _TrackPool(block.pool)
    block.pool = tracking_pool

    x = torch.randn(B, T, N, C)
    timesteps = torch.randint(0, 1000, (B,), dtype=torch.long)
    time_emb = torch.randn(B, 16)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    active_mask = torch.tensor([[True, False, True, True]], dtype=torch.bool)

    x_pooled, new_mask, x_skip, intermediates = block(
        x=x,
        timesteps=timesteps,
        edge_index=edge_index,
        time_emb=time_emb,
        active_mask=active_mask,
        return_intermediates=True,
    )

    assert tracking_pool.called is False
    assert intermediates["pool_skipped"] is True
    torch.testing.assert_close(x_pooled, x_skip, atol=1e-7, rtol=0.0)
    torch.testing.assert_close(new_mask, active_mask)


def test_encoder_gamma1_stride_poolk1_calls_pool_without_downsampling_smoke():
    torch.manual_seed(0)
    B, T, N, C = 1, 3, 4, 8

    block = _build_encoder_block(gamma=1, pool_K=1, selection_method="stride")
    tracking_pool = _TrackPool(block.pool)
    block.pool = tracking_pool

    x = torch.randn(B, T, N, C)
    timesteps = torch.randint(0, 1000, (B,), dtype=torch.long)
    time_emb = torch.randn(B, 16)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    active_mask = torch.tensor([[True, False, True, True]], dtype=torch.bool)

    x_pooled, new_mask, x_skip, intermediates = block(
        x=x,
        timesteps=timesteps,
        edge_index=edge_index,
        time_emb=time_emb,
        active_mask=active_mask,
        return_intermediates=True,
    )

    assert tracking_pool.called is True
    assert intermediates["pool_skipped"] is False
    # gamma=1 + stride selection keeps active node set unchanged.
    torch.testing.assert_close(new_mask, active_mask)
    assert x_pooled.shape == x_skip.shape


def test_encoder_gamma_gt1_still_uses_pooling_smoke():
    torch.manual_seed(0)
    B, T, N, C = 1, 3, 4, 8

    block = _build_encoder_block(gamma=2, pool_K=1, selection_method="learned")
    block.pool = _FailIfCalledPool()

    x = torch.randn(B, T, N, C)
    timesteps = torch.randint(0, 1000, (B,), dtype=torch.long)
    time_emb = torch.randn(B, 16)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

    with pytest.raises(RuntimeError, match="pool should not be called"):
        block(
            x=x,
            timesteps=timesteps,
            edge_index=edge_index,
            time_emb=time_emb,
            active_mask=None,
        )


def test_encoder_gamma1_deeper_stride_runs_neighborhood_pooling_and_skips_selector_smoke():
    """gamma=1 at deeper levels should run stage-1 pooling but bypass node selection."""
    torch.manual_seed(0)
    B, T, N, C = 1, 3, 8, 8

    block = _build_encoder_block(gamma=1, pool_K=1, selection_method="learned", stride_pre=4)
    block.pool.selector = _FailIfCalledSelector()
    tracking_pool = _TrackPool(block.pool)
    block.pool = tracking_pool

    x = torch.randn(B, T, N, C)
    timesteps = torch.randint(0, 1000, (B,), dtype=torch.long)
    time_emb = torch.randn(B, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    active_mask = torch.tensor([[True, False, True, False, True, False, True, False]], dtype=torch.bool)

    x_pooled, new_mask, x_skip, intermediates = block(
        x=x,
        timesteps=timesteps,
        edge_index=edge_index,
        time_emb=time_emb,
        active_mask=active_mask,
        return_intermediates=True,
    )

    assert tracking_pool.called is True
    torch.testing.assert_close(new_mask, active_mask)
    assert x_pooled.shape == x_skip.shape
    assert intermediates["pool_skipped"] is False
