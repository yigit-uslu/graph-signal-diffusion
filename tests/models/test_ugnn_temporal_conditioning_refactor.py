"""Tests for UGNN temporal-conditioning refactor behavior."""

from __future__ import annotations

import copy

import pytest
import torch
import time

from graph_signal_diffusion.cli.train import _make_optimizer as _make_train_optimizer
import graph_signal_diffusion.models.ugnn.ugnn as ugnn_module
from graph_signal_diffusion.models.components.embeddings import ConditionalTemporalMixerEmbedding
from graph_signal_diffusion.models.components.graph_conv import TemporalDepthwiseMixer
from graph_signal_diffusion.models.ugnn import (
    DecoderBlock,
    EmbeddingConfig,
    GNNConfig,
    PoolingConfig,
    UGNN,
    UGNNConfig,
    UGNNDecoder,
    UpsamplingConfig,
)


def _build_batched_ring_graph(batch_size: int, nodes_per_graph: int) -> tuple[torch.Tensor, torch.Tensor]:
    src_single = []
    dst_single = []
    for i in range(nodes_per_graph):
        src_single.extend([i, i])
        dst_single.extend([(i + 1) % nodes_per_graph, i])

    edge_index_single = torch.tensor([src_single, dst_single], dtype=torch.long)
    edge_weight_single = torch.ones(edge_index_single.size(1), dtype=torch.float32)

    edge_index_parts = []
    edge_weight_parts = []
    for b in range(batch_size):
        offset = b * nodes_per_graph
        edge_index_parts.append(edge_index_single + offset)
        edge_weight_parts.append(edge_weight_single + 0.05 * b)

    edge_index = torch.cat(edge_index_parts, dim=1)
    edge_weight = torch.cat(edge_weight_parts, dim=0)
    return edge_index, edge_weight


def _build_static_config(dropout: float = 0.05) -> UGNNConfig:
    return UGNNConfig(
        in_channels=1,
        out_channels=1,
        base_channels=8,
        channel_multipliers=[1, 1],
        gnn_config=GNNConfig(
            K=1,
            num_layers=1,
            norm_type="layer",
            dropout=dropout,
            activation="relu",
            normalize=False,
            use_strided_conv=False,
            use_pre_activation=False,
            use_temporal_mixer=True,
            temporal_kernel_size=3,
            temporal_causal=True,
            temporal_use_pointwise=True,
        ),
        pooling_config=PoolingConfig(
            gamma=[1, 1],
            pool_K=0,
            selection_method="stride",
        ),
        embedding_config=EmbeddingConfig(
            time_embed_dim=16,
            num_timesteps=32,
            cond_embed_dim=8,
            cond_channels=2,
            cond_temporal_hidden_channels=[4, 4],
            cond_temporal_num_layers=2,
            cond_temporal_kernel_size=5,
            cond_temporal_pooling="ema",
            cond_temporal_causal=True,
            cond_temporal_use_pointwise=True,
            cond_temporal_ema_alpha_init=0.8,
            cond_temporal_output_mode="static",
        ),
        num_bottleneck_layers=0,
    )


def _build_time_varying_config() -> UGNNConfig:
    cfg = _build_static_config(dropout=0.0)
    cfg.embedding_config.cond_temporal_output_mode = "time_varying"
    cfg.embedding_config.cond_temporal_time_varying_method = "learned_projection"
    cfg.embedding_config.cond_temporal_projection_source_timesteps = 20
    cfg.embedding_config.cond_temporal_projection_target_timesteps = 5
    cfg.embedding_config.cond_temporal_dilations = [1, 4]
    cfg.embedding_config.cond_temporal_hidden_channels = [8, 8]
    return cfg


def _build_time_varying_cross_attention_config() -> UGNNConfig:
    cfg = _build_time_varying_config()
    cfg.embedding_config.cond_fusion_mode = "cross_attention"
    cfg.embedding_config.cond_cross_attn_heads = 2
    cfg.embedding_config.cond_cross_attn_dropout = 0.0
    cfg.embedding_config.cond_cross_attn_bias = True
    cfg.embedding_config.cond_cross_attn_causal = True
    return cfg


def test_cond_encoder_called_once_per_ugnn_forward():
    cfg = _build_static_config(dropout=0.05)
    model = UGNN(config=cfg).train()

    calls = {"encoder": 0, "decoder": 0}
    encoder_forward = model.encoder.cond_encoder.forward
    decoder_forward = model.decoder.cond_encoder.forward

    def _encoder_wrapped(*args, **kwargs):
        calls["encoder"] += 1
        return encoder_forward(*args, **kwargs)

    def _decoder_wrapped(*args, **kwargs):
        calls["decoder"] += 1
        return decoder_forward(*args, **kwargs)

    model.encoder.cond_encoder.forward = _encoder_wrapped
    model.decoder.cond_encoder.forward = _decoder_wrapped

    B, T, N = 2, 5, 7
    x = torch.randn(B, T, N, 1)
    cond = torch.randn(B, 20, N, 2)
    timesteps = torch.randint(0, cfg.embedding_config.num_timesteps, (B,))
    edge_index, edge_weight = _build_batched_ring_graph(batch_size=B, nodes_per_graph=N)

    out, _ = model(
        x=x,
        timesteps=timesteps,
        edge_index=edge_index,
        edge_weight=edge_weight,
        cond=cond,
        return_intermediates=False,
    )
    assert out.shape == (B, T, N, 1)
    assert calls["encoder"] == 1
    assert calls["decoder"] == 0


def test_decoder_cond_encoder_not_called_by_ugnn_forward():
    cfg = _build_static_config(dropout=0.0)
    model = UGNN(config=cfg).eval()

    B, T, N = 1, 5, 6
    x = torch.randn(B, T, N, 1)
    cond = torch.randn(B, 1, N, 2)  # Exercises short-circuit path.
    timesteps = torch.randint(0, cfg.embedding_config.num_timesteps, (B,))
    edge_index, edge_weight = _build_batched_ring_graph(batch_size=B, nodes_per_graph=N)

    with torch.no_grad():
        out, _ = model(
            x=x,
            timesteps=timesteps,
            edge_index=edge_index,
            edge_weight=edge_weight,
            cond=cond,
            return_intermediates=False,
        )

    assert out.shape == (B, T, N, 1)
    assert model.encoder.cond_encoder.last_forward_short_circuited is True
    assert model.decoder.cond_encoder.last_forward_short_circuited is False


def test_decoder_requires_precomputed_cond_emb_when_conditioning_enabled():
    cfg = _build_static_config(dropout=0.0)
    decoder = UGNNDecoder(out_channels=1, config=cfg)

    B, T, N = 1, 5, 8
    x = torch.randn(B, T, N, 8)
    skip_features = [torch.randn(B, T, N, 8), torch.randn(B, T, N, 8)]
    active_masks = [torch.ones(B, N, dtype=torch.bool), torch.ones(B, N, dtype=torch.bool)]
    timesteps = torch.randint(0, cfg.embedding_config.num_timesteps, (B,))
    edge_index, edge_weight = _build_batched_ring_graph(batch_size=B, nodes_per_graph=N)
    time_emb = torch.randn(B, cfg.embedding_config.time_embed_dim)
    cond = torch.randn(B, 20, N, 2)

    with pytest.raises(ValueError, match="cond_emb_precomputed"):
        decoder(
            x=x,
            skip_features=skip_features,
            active_masks=active_masks,
            timesteps=timesteps,
            edge_index=edge_index,
            edge_weight=edge_weight,
            cond=cond,
            time_emb=time_emb,
            return_intermediates=False,
        )


def test_decoder_cond_encoder_frozen():
    cfg = _build_static_config(dropout=0.0)
    model = UGNN(config=cfg).train()
    assert all(not p.requires_grad for p in model.decoder.cond_encoder.parameters())


def test_decoder_cond_encoder_no_grad_after_backward():
    cfg = _build_static_config(dropout=0.0)
    model = UGNN(config=cfg).train()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)

    B, T, N = 2, 5, 7
    x = torch.randn(B, T, N, 1)
    cond = torch.randn(B, 20, N, 2)
    target = torch.randn(B, T, N, 1)
    timesteps = torch.randint(0, cfg.embedding_config.num_timesteps, (B,))
    edge_index, edge_weight = _build_batched_ring_graph(batch_size=B, nodes_per_graph=N)

    pred, _ = model(
        x=x,
        timesteps=timesteps,
        edge_index=edge_index,
        edge_weight=edge_weight,
        cond=cond,
    )
    loss = torch.nn.functional.mse_loss(pred, target)
    optim.zero_grad()
    loss.backward()
    optim.step()

    for p in model.decoder.cond_encoder.parameters():
        assert p.grad is None


def test_decoder_cond_encoder_params_unchanged_after_optimizer_step():
    cfg = _build_static_config(dropout=0.0)
    model = UGNN(config=cfg).train()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)
    before = [p.detach().clone() for p in model.decoder.cond_encoder.parameters()]

    B, T, N = 2, 5, 7
    x = torch.randn(B, T, N, 1)
    cond = torch.randn(B, 20, N, 2)
    target = torch.randn(B, T, N, 1)
    timesteps = torch.randint(0, cfg.embedding_config.num_timesteps, (B,))
    edge_index, edge_weight = _build_batched_ring_graph(batch_size=B, nodes_per_graph=N)

    pred, _ = model(
        x=x,
        timesteps=timesteps,
        edge_index=edge_index,
        edge_weight=edge_weight,
        cond=cond,
    )
    loss = torch.nn.functional.mse_loss(pred, target)
    optim.zero_grad()
    loss.backward()
    optim.step()

    after = list(model.decoder.cond_encoder.parameters())
    for b, a in zip(before, after):
        torch.testing.assert_close(a.detach(), b)


def test_time_varying_projection_exact_shape_param():
    encoder = ConditionalTemporalMixerEmbedding(
        in_channels=2,
        embed_dim=8,
        hidden_channels=[4, 4],
        num_layers=2,
        kernel_size=5,
        pooling="ema",
        output_mode="time_varying",
        time_varying_method="learned_projection",
        projection_source_timesteps=20,
        projection_target_timesteps=5,
        dilations=[1, 4],
    )
    assert encoder.temporal_projection_logits is not None
    assert tuple(encoder.temporal_projection_logits.shape) == (5, 20)


def test_time_varying_projection_runtime_length_validation():
    encoder = ConditionalTemporalMixerEmbedding(
        in_channels=2,
        embed_dim=8,
        hidden_channels=[4, 4],
        num_layers=2,
        kernel_size=5,
        pooling="ema",
        output_mode="time_varying",
        time_varying_method="learned_projection",
        projection_source_timesteps=20,
        projection_target_timesteps=5,
        dilations=[1, 4],
    )
    x_bad_source = torch.randn(1, 19, 4, 2)
    with pytest.raises(ValueError, match="T_cond mismatch"):
        encoder(x_bad_source, target_timesteps=5)

    x = torch.randn(1, 20, 4, 2)
    with pytest.raises(ValueError, match="target_timesteps mismatch"):
        encoder(x, target_timesteps=4)


def test_target_timesteps_plumbed_from_x_shape():
    cfg = _build_time_varying_config()
    model = UGNN(config=cfg).eval()

    B, N = 1, 6
    x = torch.randn(B, 5, N, 1)
    cond = torch.randn(B, 20, N, 2)
    timesteps = torch.randint(0, cfg.embedding_config.num_timesteps, (B,))
    edge_index, edge_weight = _build_batched_ring_graph(batch_size=B, nodes_per_graph=N)

    with torch.no_grad():
        out, _ = model(
            x=x,
            timesteps=timesteps,
            edge_index=edge_index,
            edge_weight=edge_weight,
            cond=cond,
        )
    assert out.shape == (B, 5, N, 1)

    x_bad_t = torch.randn(B, 4, N, 1)
    with pytest.raises(ValueError, match="target_timesteps mismatch"):
        model(
            x=x_bad_t,
            timesteps=timesteps,
            edge_index=edge_index,
            edge_weight=edge_weight,
            cond=cond,
        )


def test_time_varying_output_proj_flatten_unflatten():
    encoder = ConditionalTemporalMixerEmbedding(
        in_channels=2,
        embed_dim=8,
        hidden_channels=[4, 4],
        num_layers=2,
        kernel_size=5,
        pooling="ema",
        output_mode="time_varying",
        time_varying_method="learned_projection",
        projection_source_timesteps=20,
        projection_target_timesteps=5,
        dilations=[1, 4],
    )
    x = torch.randn(2, 20, 7, 2)
    y = encoder(x, target_timesteps=5)
    assert y.shape == (2, 5, 7, 8)

    x_bn = torch.randn(14, 20, 2)
    y_bn = encoder(x_bn, target_timesteps=5)
    assert y_bn.shape == (14, 5, 8)


def test_cond_temporal_dilation_per_layer_applied():
    encoder = ConditionalTemporalMixerEmbedding(
        in_channels=2,
        embed_dim=8,
        hidden_channels=[4, 4],
        num_layers=2,
        kernel_size=5,
        pooling="ema",
        output_mode="time_varying",
        time_varying_method="learned_projection",
        projection_source_timesteps=20,
        projection_target_timesteps=5,
        dilations=[1, 4],
    )
    assert encoder.mixers[0].depthwise.dilation == (1,)
    assert encoder.mixers[1].depthwise.dilation == (4,)


def test_causal_no_future_leak_with_dilation():
    mixer = TemporalDepthwiseMixer(
        channels=1,
        kernel_size=3,
        dilation=2,
        causal=True,
        dropout=0.0,
        activation="relu",
        use_pointwise=False,
        norm_type="none",
    )
    with torch.no_grad():
        mixer.depthwise.weight.fill_(1.0)

    x_base = torch.zeros(1, 7, 1, 1)
    x_with_future_impulse = x_base.clone()
    x_with_future_impulse[:, 6, 0, 0] = 10.0

    y_base = mixer(x_base)
    y_impulse = mixer(x_with_future_impulse)

    torch.testing.assert_close(
        y_impulse[:, :6, :, :],
        y_base[:, :6, :, :],
        atol=1e-7,
        rtol=0.0,
    )


def test_strict_load_pre_shared_static_checkpoint_layout():
    cfg = _build_static_config(dropout=0.0)
    model = UGNN(config=cfg)
    state = model.state_dict()
    assert not any(k.startswith("shared_cond_encoder.") for k in state.keys())

    reloaded = UGNN(config=copy.deepcopy(cfg))
    reloaded.load_state_dict(state, strict=True)


def test_time_varying_checkpoint_expectation_documented():
    static_cfg = _build_static_config(dropout=0.0)
    static_model = UGNN(config=static_cfg)
    static_state = static_model.state_dict()

    time_varying_cfg = _build_time_varying_config()
    time_varying_model = UGNN(config=time_varying_cfg)

    with pytest.raises(RuntimeError, match="Missing key\\(s\\)"):
        time_varying_model.load_state_dict(static_state, strict=True)


def test_optimizer_param_group_excludes_frozen_decoder_cond_encoder():
    cfg = _build_static_config(dropout=0.0)
    model = UGNN(config=cfg)
    trainer_cfg = {"optimizer": "adam", "learning_rate": 1e-3, "weight_decay": 1e-2}
    optimizer = _make_train_optimizer(trainer_cfg, model.parameters())

    optimizer_param_ids = {
        id(p)
        for group in optimizer.param_groups
        for p in group["params"]
    }
    frozen_decoder_param_ids = {id(p) for p in model.decoder.cond_encoder.parameters()}

    assert frozen_decoder_param_ids
    assert optimizer_param_ids.isdisjoint(frozen_decoder_param_ids)


def test_static_checkpoint_roundtrip_inference_regression():
    cfg = _build_static_config(dropout=0.0)
    model_before = UGNN(config=cfg).eval()
    checkpoint = copy.deepcopy(model_before.state_dict())

    model_after = UGNN(config=copy.deepcopy(cfg)).eval()
    model_after.load_state_dict(checkpoint, strict=True)

    B, T, N = 2, 5, 7
    torch.manual_seed(1234)
    x = torch.randn(B, T, N, 1)
    cond = torch.randn(B, 20, N, 2)
    timesteps = torch.randint(0, cfg.embedding_config.num_timesteps, (B,))
    edge_index, edge_weight = _build_batched_ring_graph(batch_size=B, nodes_per_graph=N)

    with torch.no_grad():
        out_before, _ = model_before(
            x=x,
            timesteps=timesteps,
            edge_index=edge_index,
            edge_weight=edge_weight,
            cond=cond,
        )
        out_after, _ = model_after(
            x=x,
            timesteps=timesteps,
            edge_index=edge_index,
            edge_weight=edge_weight,
            cond=cond,
        )

    assert out_before.shape == out_after.shape == (B, T, N, 1)
    assert torch.isfinite(out_before).all()
    assert torch.isfinite(out_after).all()
    torch.testing.assert_close(out_after, out_before)


def test_decoder_block_accepts_time_varying_cond_emb():
    gnn_config = GNNConfig()
    upsampling_config = UpsamplingConfig(method="zero")
    embedding_config = EmbeddingConfig(
        time_embed_dim=32,
        cond_channels=2,
        cond_embed_dim=16,
    )
    block = DecoderBlock(
        in_channels=16,
        skip_channels=8,
        out_channels=8,
        stride_pre=1,
        gnn_config=gnn_config,
        upsampling_config=upsampling_config,
        embedding_config=embedding_config,
        skip_connection_mode="concat",
    )

    B, T, N = 2, 5, 10
    x = torch.randn(B, T, N, 16)
    skip_features = torch.randn(B, T, N, 8)
    cond_emb = torch.randn(B, T, N, 16)
    timesteps = torch.randint(0, 1000, (B,))
    time_emb = torch.randn(B, 32)
    edge_index = torch.randint(0, B * N, (2, 80))
    active_mask_target = torch.ones(B, N, dtype=torch.bool)
    active_mask_input = torch.ones(B, N, dtype=torch.bool)

    out, _ = block(
        x=x,
        skip_features=skip_features,
        timesteps=timesteps,
        edge_index=edge_index,
        active_mask_target=active_mask_target,
        active_mask_input=active_mask_input,
        cond_emb=cond_emb,
        time_emb=time_emb,
    )
    assert out.shape == (B, T, N, 8)


def test_cross_attention_active_nodes_only(monkeypatch):
    cfg = _build_time_varying_cross_attention_config()
    gnn_cfg = cfg.gnn_config
    upsampling_cfg = cfg.upsampling_config
    emb_cfg = cfg.embedding_config
    block = DecoderBlock(
        in_channels=8,
        skip_channels=8,
        out_channels=8,
        stride_pre=1,
        gnn_config=gnn_cfg,
        upsampling_config=upsampling_cfg,
        embedding_config=emb_cfg,
        skip_connection_mode="concat",
    )
    block.eval()

    B, T, N = 2, 5, 6
    x = torch.randn(B, T, N, 8)
    skip_features = torch.randn(B, T, N, 8)
    cond_emb = torch.randn(B, T, N, 8)
    timesteps = torch.randint(0, 1000, (B,))
    time_emb = torch.randn(B, emb_cfg.time_embed_dim)
    edge_index, edge_weight = _build_batched_ring_graph(batch_size=B, nodes_per_graph=N)
    active_mask_target = torch.tensor(
        [[True, False, True, False, False, True], [True, True, False, False, True, False]],
        dtype=torch.bool,
    )
    active_mask_input = active_mask_target.clone()

    call_batch_sizes: list[int] = []
    real_sdpa = ugnn_module.F.scaled_dot_product_attention

    def _wrapped_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False):
        call_batch_sizes.append(int(q.shape[0]))
        return real_sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)

    monkeypatch.setattr(ugnn_module.F, "scaled_dot_product_attention", _wrapped_sdpa)

    out, _ = block(
        x=x,
        skip_features=skip_features,
        timesteps=timesteps,
        edge_index=edge_index,
        edge_weight=edge_weight,
        active_mask_target=active_mask_target,
        active_mask_input=active_mask_input,
        cond_emb=cond_emb,
        time_emb=time_emb,
    )
    assert out.shape == (B, T, N, 8)
    # Vectorized cross-attention gathers all active nodes across the batch
    # into a single SDPA call: 3 active nodes/sample * 2 samples = 6 total.
    assert call_batch_sizes == [6]


def _flatten_head_temporal_cross_attention(
    module: ugnn_module.TemporalNodeCrossAttention,
    query: torch.Tensor,
    key_value: torch.Tensor,
    active_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference implementation that flattens heads into batch before SDPA.

    This mirrors the same projection + masking behavior as
    TemporalNodeCrossAttention but avoids the fast-path head tensor shape.
    """
    if query.shape != key_value.shape:
        raise ValueError(
            "TemporalNodeCrossAttention expects query and key_value with identical "
            f"shape, got {tuple(query.shape)} vs {tuple(key_value.shape)}"
        )
    if query.dim() != 4:
        raise ValueError(
            f"TemporalNodeCrossAttention expects rank-4 tensors (B,T,N,C), got {query.dim()}D"
        )

    bsz, timesteps, num_nodes, channels = query.shape
    if channels != module.channels:
        raise ValueError(
            f"Channel mismatch: got C={channels}, expected {module.channels} for cross-attention."
        )

    if active_mask is None:
        active_mask = torch.ones(
            bsz,
            num_nodes,
            dtype=torch.bool,
            device=query.device,
        )
    elif active_mask.shape != (bsz, num_nodes):
        raise ValueError(
            f"active_mask shape must be (B, N)=({bsz}, {num_nodes}), got {tuple(active_mask.shape)}"
        )

    q_bn = query.permute(0, 2, 1, 3).reshape(bsz * num_nodes, timesteps, channels)
    kv_bn = key_value.permute(0, 2, 1, 3).reshape(bsz * num_nodes, timesteps, channels)
    active_bn = active_mask.reshape(-1)
    num_active = int(active_bn.sum().item())

    if num_active > 0:
        q_active = q_bn[active_bn]
        kv_active = kv_bn[active_bn]

        q = module.q_proj(q_active).reshape(
            q_active.shape[0],
            timesteps,
            module.num_heads,
            module.head_dim,
        ).permute(0, 2, 1, 3).contiguous()
        k = module.k_proj(kv_active).reshape(
            kv_active.shape[0],
            timesteps,
            module.num_heads,
            module.head_dim,
        ).permute(0, 2, 1, 3).contiguous()
        v = module.v_proj(kv_active).reshape(
            kv_active.shape[0],
            timesteps,
            module.num_heads,
            module.head_dim,
        ).permute(0, 2, 1, 3).contiguous()

        # Flatten head into batch.
        qf = q.reshape(q.shape[0] * module.num_heads, timesteps, module.head_dim)
        kf = k.reshape(k.shape[0] * module.num_heads, timesteps, module.head_dim)
        vf = v.reshape(v.shape[0] * module.num_heads, timesteps, module.head_dim)

        attn = ugnn_module.F.scaled_dot_product_attention(
            qf,
            kf,
            vf,
            attn_mask=None,
            dropout_p=module.dropout if module.training else 0.0,
            is_causal=module.causal,
        )
        attn = attn.reshape(
            q.shape[0],
            module.num_heads,
            timesteps,
            module.head_dim,
        ).permute(0, 2, 1, 3).contiguous()

        out_active = module.out_proj(attn.reshape(q_active.shape[0], timesteps, channels))
        out_bn = torch.zeros_like(q_bn)
        out_bn[active_bn] = out_active
    else:
        out_bn = torch.zeros_like(q_bn)

    out = out_bn.view(bsz, num_nodes, timesteps, channels).permute(0, 2, 1, 3).contiguous()
    out = out * active_mask.unsqueeze(1).unsqueeze(-1).to(dtype=out.dtype)
    return out


def _fallback_attention(
    module: ugnn_module.TemporalNodeCrossAttention,
    query: torch.Tensor,
    key_value: torch.Tensor,
    active_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if query.shape != key_value.shape:
        raise ValueError(
            "TemporalNodeCrossAttention expects query and key_value with identical "
            f"shape, got {tuple(query.shape)} vs {tuple(key_value.shape)}"
        )
    if query.dim() != 4:
        raise ValueError(
            f"TemporalNodeCrossAttention expects rank-4 tensors (B,T,N,C), got {query.dim()}D"
        )

    bsz, timesteps, num_nodes, channels = query.shape
    if channels != module.channels:
        raise ValueError(
            f"Channel mismatch: got C={channels}, expected {module.channels} for cross-attention."
        )

    if active_mask is None:
        active_mask = torch.ones(
            bsz,
            num_nodes,
            dtype=torch.bool,
            device=query.device,
        )
    else:
        if active_mask.shape != (bsz, num_nodes):
            raise ValueError(
                f"active_mask shape must be (B, N)=({bsz}, {num_nodes}), got {tuple(active_mask.shape)}"
            )
        if active_mask.dtype != torch.bool:
            raise ValueError(
                f"active_mask dtype must be torch.bool, got {active_mask.dtype}"
            )

    q_bn = query.permute(0, 2, 1, 3).reshape(bsz * num_nodes, timesteps, channels)
    kv_bn = key_value.permute(0, 2, 1, 3).reshape(bsz * num_nodes, timesteps, channels)

    active_bn = active_mask.reshape(-1)
    num_active = int(active_bn.sum().item())
    if num_active > 0:
        q_active = q_bn[active_bn]
        kv_active = kv_bn[active_bn]

        qh = module._to_heads(module.q_proj(q_active))
        kh = module._to_heads(module.k_proj(kv_active))
        vh = module._to_heads(module.v_proj(kv_active))

        scale = 1.0 / (module.head_dim ** 0.5)
        attn_scores = torch.matmul(qh, kh.transpose(-2, -1)) * scale
        if module.causal:
            causal_mask = torch.triu(
                torch.ones(
                    (timesteps, timesteps),
                    device=qh.device,
                    dtype=torch.bool,
                ),
                diagonal=1,
            )
            attn_scores = attn_scores.masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(0),
                float("-inf"),
            )
        attn = torch.softmax(attn_scores, dim=-1)
        attn = torch.matmul(attn, vh)

        attn = module.out_proj(module._from_heads(attn))
        out_bn = torch.zeros_like(q_bn)
        out_bn[active_bn] = attn
    else:
        out_bn = torch.zeros_like(q_bn)

    out = out_bn.view(bsz, num_nodes, timesteps, channels).permute(0, 2, 1, 3).contiguous()
    out = out * active_mask.unsqueeze(1).unsqueeze(-1).to(dtype=out.dtype)
    return out


def _benchmark_cross_attention_approach(
    fn,
    repeats: int = 20,
    warmup: int = 5,
    device: torch.device = torch.device("cpu"),
) -> float:
    # Warm up
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / float(repeats)
    return elapsed_ms


def _active_mask_from_ratio(
    batch_size: int,
    num_nodes: int,
    active_ratio: float,
    device: torch.device,
) -> torch.Tensor:
    if not (0.0 < active_ratio <= 1.0):
        raise ValueError(f"active_ratio must be in (0.0, 1.0], got {active_ratio}")

    total_nodes = batch_size * num_nodes
    num_active = max(1, int(round(total_nodes * active_ratio)))
    if num_active >= total_nodes:
        return torch.ones((batch_size, num_nodes), dtype=torch.bool, device=device)

    active_mask = torch.zeros(total_nodes, dtype=torch.bool, device=device)
    # Randomly select active nodes while preserving the requested ratio.
    active_mask[torch.randperm(total_nodes, device=device)[:num_active]] = True
    return active_mask.view(batch_size, num_nodes)


def _suggest_repeats(batch_size: int, timesteps: int, num_nodes: int, channels: int) -> int:
    """Adaptive repeat count for stable timing across different tensor scales."""
    complexity = batch_size * num_nodes * timesteps * timesteps * channels
    if complexity <= 2e4:
        return 120
    if complexity <= 1e5:
        return 60
    if complexity <= 5e5:
        return 30
    return 15


def test_cross_attention_sdpa_vs_fallback_equivalent(monkeypatch):
    attn = ugnn_module.TemporalNodeCrossAttention(
        channels=12,
        num_heads=3,
        dropout=0.0,
        bias=True,
        causal=True,
    ).eval()
    torch.manual_seed(0)

    query = torch.randn(2, 7, 5, 12)
    key_value = torch.randn(2, 7, 5, 12)
    active_mask = torch.tensor(
        [[True, False, True, True, False], [False, True, True, False, False]],
        dtype=torch.bool,
    )

    # Baseline: direct SDPA path.
    with torch.no_grad():
        expected = attn(query=query, key_value=key_value, active_mask=active_mask)

    def _fail_sdpa(*args, **kwargs):
        raise RuntimeError("forced fallback")

    # Fallback path: patch SDPA to trigger except-path.
    monkeypatch.setattr(ugnn_module.F, "scaled_dot_product_attention", _fail_sdpa)
    try:
        with torch.no_grad():
            fallback = attn(query=query, key_value=key_value, active_mask=active_mask)
    finally:
        monkeypatch.undo()

    torch.testing.assert_close(expected, fallback, rtol=1e-6, atol=1e-7)


def test_cross_attention_head_flatten_vs_current_path():
    attn = ugnn_module.TemporalNodeCrossAttention(
        channels=8,
        num_heads=2,
        dropout=0.0,
        bias=False,
        causal=False,
    ).eval()
    torch.manual_seed(0)

    query = torch.randn(2, 4, 6, 8)
    key_value = torch.randn(2, 4, 6, 8)
    active_mask = torch.tensor(
        [[True, True, False, True, False, False], [True, False, True, False, True, True]],
        dtype=torch.bool,
    )

    with torch.no_grad():
        current = attn(query=query, key_value=key_value, active_mask=active_mask)
        flattened = _flatten_head_temporal_cross_attention(
            module=attn,
            query=query,
            key_value=key_value,
            active_mask=active_mask,
        )

    torch.testing.assert_close(current, flattened, rtol=1e-6, atol=1e-7)


def test_cross_attention_requires_bool_active_mask():
    attn = ugnn_module.TemporalNodeCrossAttention(
        channels=8,
        num_heads=2,
        dropout=0.0,
        bias=True,
        causal=False,
    ).eval()

    query = torch.randn(2, 5, 6, 8)
    key_value = torch.randn(2, 5, 6, 8)
    active_mask = torch.tensor(
        [[1, 0, 1, 0, 0, 1], [1, 1, 0, 0, 1, 0]],
        dtype=torch.long,
    )

    with pytest.raises(ValueError, match="active_mask dtype must be torch.bool"):
        _ = attn(query=query, key_value=key_value, active_mask=active_mask)


def test_cross_attention_t1_wra_like_smoke():
    B, N = 2, 7
    cond = torch.randn(B, 20, N, 2)
    timesteps = torch.randint(0, 32, (B,))
    edge_index, edge_weight = _build_batched_ring_graph(batch_size=B, nodes_per_graph=N)

    # T_out=1 is the true WRA-like edge case for decoder/encoder temporal streams.
    x_t1 = torch.randn(B, 1, N, 1)
    cfg_t1 = _build_time_varying_cross_attention_config()
    cfg_t1.embedding_config.cond_temporal_projection_target_timesteps = 1
    cfg_t1.embedding_config.num_timesteps = 32
    model_t1 = UGNN(config=cfg_t1).eval()
    with torch.no_grad():
        out_t1, _ = model_t1(
            x=x_t1,
            timesteps=timesteps,
            edge_index=edge_index,
            edge_weight=edge_weight,
            cond=cond,
        )
    assert out_t1.shape == (B, 1, N, 1)
    assert torch.isfinite(out_t1).all()


def test_ugnn_cross_attention_time_varying_forward():
    cfg = _build_time_varying_cross_attention_config()
    model = UGNN(config=cfg).eval()

    B, N = 1, 6
    x = torch.randn(B, 5, N, 1)
    cond = torch.randn(B, 20, N, 2)
    timesteps = torch.randint(0, cfg.embedding_config.num_timesteps, (B,))
    edge_index, edge_weight = _build_batched_ring_graph(batch_size=B, nodes_per_graph=N)

    with torch.no_grad():
        out, _ = model(
            x=x,
            timesteps=timesteps,
            edge_index=edge_index,
            edge_weight=edge_weight,
            cond=cond,
        )
    assert out.shape == (B, 5, N, 1)
    assert torch.isfinite(out).all()


# ─── ARM D-v2: cross-attention with mismatched T + method=none ──────────────


def _build_method_none_cross_attention_config() -> UGNNConfig:
    """ARM D-v2 style: method=none (full T=20 cond) + cross_attention fusion."""
    cfg = _build_time_varying_cross_attention_config()
    cfg.embedding_config.cond_temporal_time_varying_method = "none"
    # projection params become irrelevant with method=none
    return cfg


def test_cross_attention_mismatched_t_smoke():
    """T_q=5, T_kv=20 non-causal cross-attention: shape + finiteness + backward."""
    torch.manual_seed(42)
    attn = ugnn_module.TemporalNodeCrossAttention(
        channels=8, num_heads=2, dropout=0.0, causal=False,
    )
    query = torch.randn(2, 5, 16, 8, requires_grad=True)
    key_value = torch.randn(2, 20, 16, 8, requires_grad=True)
    out = attn(query, key_value)
    assert out.shape == (2, 5, 16, 8)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert query.grad is not None
    assert key_value.grad is not None


def test_cross_attention_mismatched_t_active_mask():
    """T_q != T_kv with 50% active nodes: inactive nodes produce zeros."""
    torch.manual_seed(42)
    attn = ugnn_module.TemporalNodeCrossAttention(
        channels=8, num_heads=2, dropout=0.0, causal=False,
    )
    B, T_q, T_kv, N, C = 2, 5, 20, 16, 8
    query = torch.randn(B, T_q, N, C)
    key_value = torch.randn(B, T_kv, N, C)
    active_mask = torch.ones(B, N, dtype=torch.bool)
    active_mask[:, N // 2:] = False  # 50% active

    out = attn(query, key_value, active_mask=active_mask)
    assert out.shape == (B, T_q, N, C)
    assert torch.isfinite(out).all()
    # Inactive nodes must be exactly zero
    inactive_out = out[:, :, N // 2:, :]
    assert (inactive_out == 0).all(), "Inactive nodes should produce zero output"


def test_cross_attention_mismatched_t_causal(monkeypatch):
    """T_q=5, T_kv=20 causal: explicit mask path + fallback path both work."""
    torch.manual_seed(42)
    attn = ugnn_module.TemporalNodeCrossAttention(
        channels=8, num_heads=2, dropout=0.0, causal=True,
    )
    query = torch.randn(2, 5, 8, 8)
    key_value = torch.randn(2, 20, 8, 8)

    # SDPA path (explicit causal mask for mismatched T)
    out_sdpa = attn(query, key_value)
    assert out_sdpa.shape == (2, 5, 8, 8)
    assert torch.isfinite(out_sdpa).all()

    # Force fallback path by monkeypatching SDPA to raise
    import torch.nn.functional as _F
    real_sdpa = _F.scaled_dot_product_attention

    def _raise_sdpa(*args, **kwargs):
        raise RuntimeError("forced fallback")

    monkeypatch.setattr(_F, "scaled_dot_product_attention", _raise_sdpa)
    out_fallback = attn(query, key_value)
    assert out_fallback.shape == (2, 5, 8, 8)
    assert torch.isfinite(out_fallback).all()

    # Both paths should produce the same result
    monkeypatch.setattr(_F, "scaled_dot_product_attention", real_sdpa)
    torch.testing.assert_close(out_sdpa, out_fallback, atol=1e-5, rtol=1e-5)


def test_cross_attention_equal_t_regression():
    """T_q == T_kv still works after the T-mismatch support."""
    torch.manual_seed(42)
    attn = ugnn_module.TemporalNodeCrossAttention(
        channels=8, num_heads=2, dropout=0.0, causal=False,
    )
    query = torch.randn(2, 10, 8, 8)
    key_value = torch.randn(2, 10, 8, 8)
    out = attn(query, key_value)
    assert out.shape == (2, 10, 8, 8)
    assert torch.isfinite(out).all()


def test_cond_encoder_method_none_output_shape():
    """method=none outputs (B, T_cond, N, embed_dim) with no projection params."""
    encoder = ConditionalTemporalMixerEmbedding(
        in_channels=12, embed_dim=8,
        hidden_channels=[16, 16], num_layers=2,
        kernel_size=3, causal=False, use_pointwise=True,
        output_mode="time_varying", time_varying_method="none",
    )
    # No temporal_projection_logits parameter
    assert encoder.temporal_projection_logits is None
    param_names = [n for n, _ in encoder.named_parameters()]
    assert not any("projection_logits" in n for n in param_names)

    x = torch.randn(2, 20, 7, 12)  # (B, T_cond=20, N=7, F=12)
    out = encoder(x)
    assert out.shape == (2, 20, 7, 8)  # (B, T_cond, N, embed_dim)
    assert torch.isfinite(out).all()


def test_cond_encoder_method_none_backward():
    """Gradients flow through method=none output_proj and temporal conv layers."""
    encoder = ConditionalTemporalMixerEmbedding(
        in_channels=12, embed_dim=8,
        hidden_channels=[16, 16], num_layers=2,
        kernel_size=3, causal=False, use_pointwise=True,
        output_mode="time_varying", time_varying_method="none",
    )
    x = torch.randn(2, 20, 7, 12)
    out = encoder(x)
    out.sum().backward()
    # Check gradients on output_proj and mixer conv layers
    assert encoder.output_proj.mlp[0].weight.grad is not None
    assert encoder.mixers[0].depthwise.weight.grad is not None


def test_embedding_config_accepts_method_none():
    """EmbeddingConfig with method=none parses without error."""
    cfg = EmbeddingConfig(
        cond={
            "channels": 2,
            "embed_dim": 8,
            "shared_encoder": {
                "temporal": {
                    "hidden_channels": [8, 8],
                    "num_layers": 2,
                    "output": {
                        "mode": "time_varying",
                        "time_varying": {
                            "method": "none",
                        },
                    },
                },
            },
        },
    )
    assert cfg.cond_temporal_output_mode == "time_varying"
    assert cfg.cond_temporal_time_varying_method == "none"
    # projection timesteps are not required
    assert cfg.cond_temporal_projection_source_timesteps is None
    assert cfg.cond_temporal_projection_target_timesteps is None


def test_concat_fusion_rejects_t_mismatch():
    """concat fusion with T_cond != T_denoiser raises ValueError."""
    cfg = _build_time_varying_config()
    cfg.embedding_config.cond_fusion_mode = "concat"
    # method=none outputs T=20, but denoiser has T=5 — concat can't handle this
    cfg.embedding_config.cond_temporal_time_varying_method = "none"
    model = UGNN(config=cfg).eval()

    B, N = 1, 6
    x = torch.randn(B, 5, N, 1)
    cond = torch.randn(B, 20, N, 2)
    timesteps = torch.randint(0, cfg.embedding_config.num_timesteps, (B,))
    edge_index, edge_weight = _build_batched_ring_graph(batch_size=B, nodes_per_graph=N)

    with pytest.raises(ValueError, match="must match T="):
        with torch.no_grad():
            model(x=x, timesteps=timesteps, edge_index=edge_index,
                  edge_weight=edge_weight, cond=cond)


def test_ugnn_cross_attention_method_none_e2e():
    """ARM D-v2 config: method=none + cross_attention, T_cond=20, T_future=5."""
    torch.manual_seed(42)
    cfg = _build_method_none_cross_attention_config()
    model = UGNN(config=cfg)

    B, N = 1, 6
    x = torch.randn(B, 5, N, 1)
    cond = torch.randn(B, 20, N, 2)
    timesteps = torch.randint(0, cfg.embedding_config.num_timesteps, (B,))
    edge_index, edge_weight = _build_batched_ring_graph(batch_size=B, nodes_per_graph=N)

    out, _ = model(x=x, timesteps=timesteps, edge_index=edge_index,
                   edge_weight=edge_weight, cond=cond)
    assert out.shape == (B, 5, N, 1)
    assert torch.isfinite(out).all()

    # Backward
    out.sum().backward()
    cond_encoder_grads = [
        p.grad for p in model.encoder.cond_encoder.parameters() if p.grad is not None
    ]
    assert len(cond_encoder_grads) > 0, "Cond encoder should receive gradients"


def test_ugnn_cross_attention_learned_projection_backward_compat():
    """method=learned_projection + cross_attention still works (T_q == T_kv)."""
    torch.manual_seed(42)
    cfg = _build_time_varying_cross_attention_config()
    assert cfg.embedding_config.cond_temporal_time_varying_method == "learned_projection"
    model = UGNN(config=cfg).eval()

    B, N = 1, 6
    x = torch.randn(B, 5, N, 1)
    cond = torch.randn(B, 20, N, 2)
    timesteps = torch.randint(0, cfg.embedding_config.num_timesteps, (B,))
    edge_index, edge_weight = _build_batched_ring_graph(batch_size=B, nodes_per_graph=N)

    with torch.no_grad():
        out, _ = model(x=x, timesteps=timesteps, edge_index=edge_index,
                       edge_weight=edge_weight, cond=cond)
    assert out.shape == (B, 5, N, 1)
    assert torch.isfinite(out).all()



    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    torch.manual_seed(0)

    # (batch, time, nodes, channels, active_ratio)
    benchmark_cases = [
        (1, 1, 16, 16, 1.0),
        (1, 4, 32, 16, 1.0),
        (2, 4, 64, 16, 0.75),
        (2, 8, 64, 16, 0.5),
        (4, 8, 128, 16, 1.0),
        (4, 16, 128, 32, 0.5),
        (8, 8, 64, 32, 0.25),
    ]

    def _fail_sdpa(*args, **kwargs):  # pragma: no cover - only for benchmark path
        raise RuntimeError("forced fallback path")

    rows = []
    for case in benchmark_cases:
        bsz, timesteps, num_nodes, channels, active_ratio = case
        num_heads = 4
        if channels % num_heads != 0:
            raise AssertionError(f"Case has non-divisible channels: {channels} % {num_heads} != 0")

        repeats = _suggest_repeats(bsz, timesteps, num_nodes, channels)

        attn = ugnn_module.TemporalNodeCrossAttention(
            channels=channels,
            num_heads=num_heads,
            dropout=0.0,
            bias=True,
            causal=True,
        ).to(device).eval()
        active_mask = _active_mask_from_ratio(
            batch_size=bsz,
            num_nodes=num_nodes,
            active_ratio=active_ratio,
            device=device,
        )
        query = torch.randn(bsz, timesteps, num_nodes, channels, device=device)
        key_value = torch.randn(bsz, timesteps, num_nodes, channels, device=device)

        with torch.no_grad():
            ms_regular = _benchmark_cross_attention_approach(
                lambda: attn(query=query, key_value=key_value, active_mask=active_mask),
                repeats=repeats,
                device=device,
            )

            sdpa_orig = ugnn_module.F.scaled_dot_product_attention
            ugnn_module.F.scaled_dot_product_attention = _fail_sdpa
            try:
                ms_fallback = _benchmark_cross_attention_approach(
                    lambda: attn(query=query, key_value=key_value, active_mask=active_mask),
                    repeats=repeats,
                    device=device,
                )
            finally:
                ugnn_module.F.scaled_dot_product_attention = sdpa_orig

            ms_flatten = _benchmark_cross_attention_approach(
                lambda: _flatten_head_temporal_cross_attention(
                    module=attn,
                    query=query,
                    key_value=key_value,
                    active_mask=active_mask,
                ),
                repeats=repeats,
                device=device,
            )

        rows.append(
            {
                "shape": f"{bsz}x{timesteps}x{num_nodes}x{channels}",
                "active_ratio": active_ratio,
                "active_nodes": int(active_mask.sum().item()),
                "repeats": repeats,
                "sdpa_ms": ms_regular,
                "fallback_ms": ms_fallback,
                "flatten_ms": ms_flatten,
            }
        )

    # Present timing results in a compact, readable table.
    header = (
        "\nCross-attention benchmark (ms / call, averaged):\n"
        "| shape (B,T,N,C) | active_ratio | active_nodes | repeats | sdpa(ms) | "
        "fallback(ms) | flatten-head(ms) |\n"
        "|---|---:|---:|---:|---:|---:|---:|"
    )
    lines = [header]
    for row in rows:
        lines.append(
            f"| {row['shape']} | {row['active_ratio']:.2f} | {row['active_nodes']:>11d} | "
            f"{row['repeats']:>7d} | {row['sdpa_ms']:>8.4f} | {row['fallback_ms']:>11.4f} | "
            f"{row['flatten_ms']:>15.4f} |"
        )
    print("\n".join(lines))

    assert len(rows) == len(benchmark_cases)
    for row in rows:
        assert row["sdpa_ms"] > 0
        assert row["fallback_ms"] > 0
        assert row["flatten_ms"] > 0
