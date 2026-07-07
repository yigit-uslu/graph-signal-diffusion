"""Smoke tests for optional temporal depthwise mixing in GNN/UGNN."""

import pytest
import torch

from graph_signal_diffusion.models.components.graph_conv import GNN, TemporalDepthwiseMixer, _TemporalSelfAttention
from graph_signal_diffusion.models.components.embeddings import ConditionalTemporalMixerEmbedding
from graph_signal_diffusion.models.ugnn import (
    EmbeddingConfig,
    GNNConfig,
    PoolingConfig,
    UGNN,
    UGNNConfig,
)


def _build_batched_ring_graph(batch_size: int, nodes_per_graph: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Create a batched directed ring graph with self-loops."""
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


def test_gnn_temporal_mixer_t1_smoke():
    """Temporal mixer path should handle T=1 without shape/runtime issues."""
    torch.manual_seed(0)

    B, T, N, C = 2, 1, 4, 2
    x = torch.randn(B, T, N, C)
    edge_index, edge_weight = _build_batched_ring_graph(batch_size=B, nodes_per_graph=N)
    active_mask = torch.tensor(
        [[True, True, False, True], [True, False, True, True]],
        dtype=torch.bool,
    )

    gnn = GNN(
        in_channels=C,
        hidden_channels=C,
        out_channels=C,
        num_layers=2,
        K=2,
        norm_type="layer",
        dropout=0.0,
        activation="silu",
        use_input_proj=False,
        use_output_proj=False,
        use_temporal_mixer=True,
        temporal_kernel_size=3,
        temporal_causal=True,
        temporal_use_pointwise=True,
        normalize=False,
    )

    out = gnn(x, edge_index, edge_weight=edge_weight, active_mask=active_mask)
    assert out.shape == (B, T, N, C)
    assert torch.isfinite(out).all()


def test_gnn_temporal_mixer_schedule_per_block_calls_once(monkeypatch):
    B, T, N, C = 1, 5, 4, 2
    x = torch.randn(B, T, N, C)
    edge_index, edge_weight = _build_batched_ring_graph(batch_size=B, nodes_per_graph=N)

    gnn = GNN(
        in_channels=C,
        hidden_channels=C,
        out_channels=C,
        num_layers=3,
        K=1,
        norm_type="layer",
        dropout=0.0,
        activation="silu",
        use_input_proj=False,
        use_output_proj=False,
        use_temporal_mixer=True,
        temporal_mixer_schedule="per_block",
        temporal_kernel_size=3,
        temporal_causal=True,
        temporal_use_pointwise=True,
        temporal_dilations=[2],
        normalize=False,
    )
    assert gnn.temporal_mixers is None
    assert gnn.block_temporal_mixer is not None

    calls = {"n": 0}
    real_forward = gnn.block_temporal_mixer.forward

    def _wrapped_forward(inp):
        calls["n"] += 1
        return real_forward(inp)

    monkeypatch.setattr(gnn.block_temporal_mixer, "forward", _wrapped_forward)

    out = gnn(x, edge_index, edge_weight=edge_weight, active_mask=None)
    assert out.shape == (B, T, N, C)
    assert calls["n"] == 1


def test_gnn_temporal_mixer_gated_packed_path_smoke():
    """Gated temporal mixer should run in packed per-layer fast path."""
    B, T, N, C = 2, 4, 6, 3
    x = torch.randn(B, T, N, C)
    edge_index, edge_weight = _build_batched_ring_graph(batch_size=B, nodes_per_graph=N)
    active_mask = torch.tensor(
        [
            [True, False, False, True, False, False],
            [False, True, False, False, False, True],
        ],
        dtype=torch.bool,
    )

    gnn = GNN(
        in_channels=C,
        hidden_channels=C,
        out_channels=C,
        num_layers=2,
        K=1,
        norm_type="layer",
        dropout=0.0,
        activation="silu",
        use_input_proj=False,
        use_output_proj=False,
        use_temporal_mixer=True,
        temporal_mixer_schedule="per_layer",
        temporal_kernel_size=3,
        temporal_causal=True,
        temporal_use_pointwise=False,
        temporal_gated=True,
        pointwise_sparse_mode="on",
        normalize=False,
    )

    out = gnn(x, edge_index, edge_weight=edge_weight, active_mask=active_mask)
    assert out.shape == (B, T, N, C)
    assert torch.isfinite(out).all()
    assert gnn._last_packed_layer_mixer_path_used is True


def test_gnn_temporal_dilations_applied_by_schedule():
    per_layer = GNN(
        in_channels=2,
        hidden_channels=2,
        out_channels=2,
        num_layers=2,
        K=1,
        norm_type="none",
        use_input_proj=False,
        use_output_proj=False,
        use_temporal_mixer=True,
        temporal_mixer_schedule="per_layer",
        temporal_dilations=[1, 3],
    )
    assert per_layer.temporal_mixers is not None
    assert [m.dilation for m in per_layer.temporal_mixers] == [1, 3]

    per_block = GNN(
        in_channels=2,
        hidden_channels=2,
        out_channels=2,
        num_layers=2,
        K=1,
        norm_type="none",
        use_input_proj=False,
        use_output_proj=False,
        use_temporal_mixer=True,
        temporal_mixer_schedule="per_block",
        temporal_dilations=[4],
    )
    assert per_block.temporal_mixers is None
    assert per_block.block_temporal_mixer is not None
    assert per_block.block_temporal_mixer.dilation == 4


def test_grouped_config_mapping_for_temporal_and_cond():
    gnn_cfg = GNNConfig(
        temporal_mixer={
            "enabled": True,
            "schedule": "per_block",
            "kernel_size": 5,
            "causal": True,
            "use_pointwise": False,
            "gated": True,
            "dilations": [2],
        }
    )
    assert gnn_cfg.use_temporal_mixer is True
    assert gnn_cfg.temporal_mixer_schedule == "per_block"
    assert gnn_cfg.temporal_kernel_size == 5
    assert gnn_cfg.temporal_causal is True
    assert gnn_cfg.temporal_use_pointwise is False
    assert gnn_cfg.temporal_gated is True
    assert gnn_cfg.temporal_dilations == [2]

    emb_cfg = EmbeddingConfig(
        cond={
            "channels": 2,
            "embed_dim": 8,
            "shared_encoder": {
                "temporal": {
                    "hidden_channels": [8, 8],
                    "num_layers": 2,
                    "mixer": {
                        "kernel_size": 5,
                        "causal": True,
                        "use_pointwise": True,
                        "dilations": [1, 4],
                    },
                    "output": {
                        "mode": "time_varying",
                        "time_varying": {
                            "method": "learned_projection",
                            "learned_projection": {
                                "source_timesteps": 20,
                                "target_timesteps": 5,
                            },
                        },
                    },
                },
            },
            "block_fusion": {
                "mode": "cross_attention",
                "projection": {
                    "type": "linear",
                    "per_level": True,
                },
                "cross_attention": {
                    "heads": 2,
                    "dropout": 0.1,
                    "bias": True,
                    "causal": True,
                },
            },
        }
    )
    assert emb_cfg.cond_channels == 2
    assert emb_cfg.cond_embed_dim == 8
    assert emb_cfg.cond_temporal_hidden_channels == [8, 8]
    assert emb_cfg.cond_temporal_num_layers == 2
    assert emb_cfg.cond_temporal_kernel_size == 5
    assert emb_cfg.cond_temporal_causal is True
    assert emb_cfg.cond_temporal_use_pointwise is True
    assert emb_cfg.cond_temporal_dilations == [1, 4]
    assert emb_cfg.cond_temporal_output_mode == "time_varying"
    assert emb_cfg.cond_temporal_time_varying_method == "learned_projection"
    assert emb_cfg.cond_temporal_projection_source_timesteps == 20
    assert emb_cfg.cond_temporal_projection_target_timesteps == 5
    assert emb_cfg.cond_fusion_mode == "cross_attention"
    assert emb_cfg.cond_projection_type == "linear"
    assert emb_cfg.cond_projection_per_level is True
    assert emb_cfg.cond_cross_attn_heads == 2
    assert emb_cfg.cond_cross_attn_dropout == 0.1
    assert emb_cfg.cond_cross_attn_bias is True
    assert emb_cfg.cond_cross_attn_causal is True


def test_cond_projection_unsupported_config_fails_loud():
    cfg = UGNNConfig(
        in_channels=1,
        out_channels=1,
        base_channels=8,
        channel_multipliers=[1, 1],
        gnn_config=GNNConfig(
            num_layers=1,
            K=1,
            norm_type="layer",
            dropout=0.0,
            activation="silu",
        ),
        pooling_config=PoolingConfig(
            gamma=[1, 1],
            pool_K=0,
        ),
        embedding_config=EmbeddingConfig(
            cond_channels=2,
            cond_embed_dim=8,
            cond_projection_per_level=False,
        ),
        num_bottleneck_layers=0,
    )

    with pytest.raises(ValueError, match="projection.per_level=false"):
        UGNN(cfg)


def test_temporal_depthwise_mixer_causal_no_future_leak_smoke():
    """Causal mixer should not let the last timestep affect earlier outputs."""
    mixer = TemporalDepthwiseMixer(
        channels=1,
        kernel_size=3,
        causal=True,
        dropout=0.0,
        activation="relu",
        use_pointwise=False,
        norm_type="none",
    )

    with torch.no_grad():
        mixer.depthwise.weight.fill_(1.0)

    x_base = torch.zeros(1, 5, 1, 1)
    x_with_future_impulse = x_base.clone()
    x_with_future_impulse[:, 4, 0, 0] = 10.0

    y_base = mixer(x_base)
    y_impulse = mixer(x_with_future_impulse)

    torch.testing.assert_close(
        y_impulse[:, :4, :, :],
        y_base[:, :4, :, :],
        atol=1e-7,
        rtol=0.0,
    )


def test_temporal_depthwise_mixer_gated_pairs_filter_and_gate_per_channel():
    """Gated split must pair filter/gate outputs within each depthwise group."""
    mixer = TemporalDepthwiseMixer(
        channels=2,
        kernel_size=1,
        causal=False,
        dropout=0.0,
        activation="relu",
        use_pointwise=False,
        norm_type="none",
        gated=True,
    )

    with torch.no_grad():
        mixer.depthwise.weight.zero_()
        # groups=channels=2, out_channels=4
        # order: [g0_o0, g0_o1, g1_o0, g1_o1]
        mixer.depthwise.weight[0, 0, 0] = 1.0
        mixer.depthwise.weight[1, 0, 0] = 10.0
        mixer.depthwise.weight[2, 0, 0] = 2.0
        mixer.depthwise.weight[3, 0, 0] = 20.0
        mixer.gate_proj.weight.zero_()
        mixer.gate_proj.weight[0, 0, 0] = 1.0
        mixer.gate_proj.weight[1, 1, 0] = 1.0

    x_seq = torch.tensor([[[1.0], [1.0]]])  # (B*=1, C=2, T=1)

    y = mixer._mix_sequence(x_seq)
    pre = mixer.depthwise(x_seq)
    expected_filter = torch.stack([pre[:, 0, :], pre[:, 2, :]], dim=1)
    expected_gate = torch.stack([pre[:, 1, :], pre[:, 3, :]], dim=1)
    expected = torch.tanh(expected_filter) * torch.sigmoid(expected_gate)

    torch.testing.assert_close(y, expected, atol=1e-7, rtol=0.0)


def test_temporal_depthwise_mixer_nongated_keeps_activation_after_pointwise():
    """Non-gated path should apply activation after pointwise projection."""
    mixer = TemporalDepthwiseMixer(
        channels=1,
        kernel_size=1,
        causal=False,
        dropout=0.0,
        activation="relu",
        use_pointwise=True,
        norm_type="none",
        gated=False,
    )

    with torch.no_grad():
        mixer.depthwise.weight.fill_(1.0)
        mixer.pointwise.weight.fill_(-1.0)

    x_seq = torch.tensor([[[2.0]]])  # (B*=1, C=1, T=1)
    y = mixer._mix_sequence(x_seq)

    # Expected with correct ordering:
    # depthwise: 2.0 -> pointwise: -2.0 -> relu: 0.0
    expected = torch.zeros_like(y)
    torch.testing.assert_close(y, expected, atol=1e-7, rtol=0.0)


def test_ugnn_temporal_mixer_forward_t1_smoke():
    """UGNN should run end-to-end with temporal mixer enabled and T=1."""
    torch.manual_seed(1)

    config = UGNNConfig(
        in_channels=1,
        out_channels=1,
        base_channels=8,
        channel_multipliers=[1, 1],
        gnn_config=GNNConfig(
            K=1,
            num_layers=2,
            norm_type="layer",
            dropout=0.0,
            activation="silu",
            normalize=False,
            use_strided_conv=True,
            use_pre_activation=False,
            use_temporal_mixer=True,
            temporal_kernel_size=3,
            temporal_causal=True,
            temporal_use_pointwise=True,
        ),
        pooling_config=PoolingConfig(
            gamma=[1, 1],
            pool_K=1,
            selection_method="learned",
            selector_kwargs={"temperature": 1.0},
            temporal_score_agg="mean",
        ),
        embedding_config=EmbeddingConfig(
            time_embed_dim=16,
            num_timesteps=10,
            cond_embed_dim=8,
            cond_channels=None,
        ),
        num_bottleneck_layers=1,
    )
    model = UGNN(config=config).eval()

    B, T, N, F = 1, 1, 5, 1
    x = torch.randn(B, T, N, F)
    timesteps = torch.tensor([0], dtype=torch.long)
    edge_index, edge_weight = _build_batched_ring_graph(batch_size=B, nodes_per_graph=N)

    with torch.no_grad():
        y, _ = model(
            x=x,
            timesteps=timesteps,
            edge_index=edge_index,
            edge_weight=edge_weight,
            return_intermediates=False,
        )

    assert y.shape == (B, T, N, 1)
    assert torch.isfinite(y).all()


def test_conditional_temporal_mixer_embedding_t1_short_circuit_smoke():
    """Conditional temporal mixer should short-circuit temporal mixing at T_cond=1."""
    torch.manual_seed(2)

    encoder = ConditionalTemporalMixerEmbedding(
        in_channels=3,
        embed_dim=4,
        hidden_channels=[6, 6],
        num_layers=2,
        kernel_size=3,
        pooling="mean",
        activation="silu",
        dropout=0.0,
        norm_type="layer",
        use_pointwise=True,
        causal=True,
    )

    x_t1 = torch.randn(2, 1, 5, 3)  # (B, T_cond=1, N, F_cond)
    y_t1 = encoder(x_t1)
    assert y_t1.shape == (2, 5, 4)
    assert encoder.last_forward_short_circuited is True
    assert torch.isfinite(y_t1).all()

    x_t5 = torch.randn(2, 5, 5, 3)
    y_t5 = encoder(x_t5)
    assert y_t5.shape == (2, 5, 4)
    assert encoder.last_forward_short_circuited is False
    assert torch.isfinite(y_t5).all()


def test_ugnn_precomputed_conditioning_smoke():
    """UGNN should precompute conditioning via encoder module and keep decoder module unused."""
    config = UGNNConfig(
        in_channels=1,
        out_channels=1,
        base_channels=8,
        channel_multipliers=[1, 1],
        gnn_config=GNNConfig(
            K=1,
            num_layers=1,
            norm_type="layer",
            dropout=0.0,
            activation="silu",
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
            pool_K=1,
            selection_method="learned",
            selector_kwargs={"temperature": 1.0},
            temporal_score_agg="mean",
        ),
        embedding_config=EmbeddingConfig(
            time_embed_dim=16,
            num_timesteps=16,
            cond_embed_dim=8,
            cond_channels=2,
            cond_temporal_num_layers=2,
            cond_temporal_kernel_size=5,
            cond_temporal_pooling="last",
            cond_temporal_causal=False,
            cond_temporal_use_pointwise=False,
            cond_temporal_ema_alpha_init=0.7,
        ),
        num_bottleneck_layers=1,
    )
    model = UGNN(config=config).eval()

    assert model.encoder.cond_encoder is not None
    assert model.decoder.cond_encoder is not None
    assert model.encoder.cond_encoder is not model.decoder.cond_encoder
    assert model.encoder.cond_encoder.num_layers == 2
    assert model.encoder.cond_encoder.kernel_size == 5
    assert model.encoder.cond_encoder.pooling == "last"
    assert model.encoder.cond_encoder.causal is False
    assert model.encoder.cond_encoder.use_pointwise is False
    assert model.encoder.cond_encoder.mixers[0].use_pointwise is False
    assert all(not p.requires_grad for p in model.decoder.cond_encoder.parameters())

    B, T, N = 1, 3, 5
    x = torch.randn(B, T, N, 1)
    cond = torch.randn(B, 1, N, 2)  # T_cond=1 exercises short-circuit path
    timesteps = torch.tensor([3], dtype=torch.long)
    edge_index, edge_weight = _build_batched_ring_graph(batch_size=B, nodes_per_graph=N)

    with torch.no_grad():
        y, _ = model(
            x=x,
            timesteps=timesteps,
            edge_index=edge_index,
            edge_weight=edge_weight,
            cond=cond,
            return_intermediates=False,
        )

    assert y.shape == (B, T, N, 1)
    assert model.encoder.cond_encoder.last_forward_short_circuited is True
    assert model.decoder.cond_encoder.last_forward_short_circuited is False


# ─── Temporal self-attention tests ───────────────────────────────────────────


def test_temporal_mixer_attention_basic_smoke():
    """Forward pass with attention_enabled=True, T=5."""
    torch.manual_seed(42)
    mixer = TemporalDepthwiseMixer(
        channels=8, kernel_size=3, dilation=1,
        attention_enabled=True, attention_num_heads=2,
    )
    x = torch.randn(2, 5, 10, 8)  # (B, T, N, C)
    y = mixer(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_temporal_mixer_attention_t1_skipped_verified(monkeypatch):
    """T=1: attention.forward must NOT be called (skipped by _mix_sequence guard)."""
    torch.manual_seed(42)
    mixer = TemporalDepthwiseMixer(
        channels=8, kernel_size=1, dilation=1,
        attention_enabled=True, attention_num_heads=2,
    )
    call_count = 0
    original_forward = mixer.self_attention.forward

    def counting_forward(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(mixer.self_attention, "forward", counting_forward)

    x = torch.randn(2, 1, 10, 8)  # T=1
    y = mixer(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert call_count == 0, f"Attention forward was called {call_count} times for T=1"

    # Also verify numerical parity with attention-disabled mixer
    torch.manual_seed(42)
    mixer_no_attn = TemporalDepthwiseMixer(
        channels=8, kernel_size=1, dilation=1,
        attention_enabled=False,
    )
    # Copy conv weights
    mixer_no_attn.load_state_dict(
        {k: v for k, v in mixer.state_dict().items() if not k.startswith("self_attention.")},
        strict=False,
    )
    y_no_attn = mixer_no_attn(x)
    torch.testing.assert_close(y, y_no_attn)


def test_gnn_attention_packed_fast_path():
    """per_layer + attention + sparse active_mask → packed path used."""
    torch.manual_seed(42)
    B, T, N, C = 2, 5, 16, 8
    gnn = GNN(
        in_channels=C, hidden_channels=C, out_channels=C,
        num_layers=2, K=2,
        use_temporal_mixer=True,
        temporal_mixer_schedule='per_layer',
        temporal_dilations=[1, 1],
        temporal_attention_enabled=True,
        temporal_attention_num_heads=2,
        pointwise_sparse_mode='on',
    )
    edge_index, edge_weight = _build_batched_ring_graph(B, N)
    active_mask = torch.ones(B, N, dtype=torch.bool)
    active_mask[:, N // 2:] = False  # 50% active

    x = torch.randn(B, T, N, C)
    y = gnn(x, edge_index, edge_weight=edge_weight, active_mask=active_mask)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert gnn._last_packed_layer_mixer_path_used is True


def test_gnn_attention_per_block_smoke():
    """per_block + attention, forward pass shape + finiteness."""
    torch.manual_seed(42)
    B, T, N, C = 2, 5, 16, 8
    gnn = GNN(
        in_channels=C, hidden_channels=C, out_channels=C,
        num_layers=2, K=2,
        use_temporal_mixer=True,
        temporal_mixer_schedule='per_block',
        temporal_dilations=[1],
        temporal_attention_enabled=True,
        temporal_attention_num_heads=2,
    )
    edge_index, edge_weight = _build_batched_ring_graph(B, N)
    x = torch.randn(B, T, N, C)
    y = gnn(x, edge_index, edge_weight=edge_weight)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_grouped_config_mapping_attention():
    """temporal_mixer.attention.* maps to flat GNNConfig fields."""
    cfg = GNNConfig(
        temporal_mixer={
            'enabled': True,
            'attention': {
                'enabled': True,
                'num_heads': 8,
                'dropout': 0.1,
            },
        },
    )
    assert cfg.temporal_attention_enabled is True
    assert cfg.temporal_attention_num_heads == 8
    assert cfg.temporal_attention_dropout == 0.1


def test_attention_disabled_by_default():
    """Default GNNConfig has temporal_attention_enabled=False."""
    cfg = GNNConfig()
    assert cfg.temporal_attention_enabled is False
    assert cfg.temporal_attention_num_heads == 4
    assert cfg.temporal_attention_dropout == 0.0


def test_attention_channels_not_divisible_by_heads_raises():
    """channels % num_heads != 0 raises; num_heads=0 raises before divisibility."""
    with pytest.raises(ValueError, match="num_heads must be >= 1"):
        _TemporalSelfAttention(channels=8, num_heads=0)

    with pytest.raises(ValueError, match="divisible by num_heads"):
        _TemporalSelfAttention(channels=7, num_heads=4)


def test_attention_invalid_dropout_raises():
    """dropout outside [0, 1] raises ValueError."""
    with pytest.raises(ValueError, match="dropout must be in"):
        _TemporalSelfAttention(channels=8, num_heads=4, dropout=-0.1)

    with pytest.raises(ValueError, match="dropout must be in"):
        _TemporalSelfAttention(channels=8, num_heads=4, dropout=1.5)


def test_temporal_mixer_attention_3d_input():
    """3D input (B*N, T, C) with attention enabled."""
    torch.manual_seed(42)
    mixer = TemporalDepthwiseMixer(
        channels=8, kernel_size=3, dilation=1,
        attention_enabled=True, attention_num_heads=2,
    )
    x = torch.randn(20, 5, 8)  # (B*N, T, C)
    y = mixer(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


# ─── Positional embedding tests ──────────────────────────────────────────────


def test_attention_pos_emb_smoke():
    """max_timesteps=5, T=5 input — forward + backward with pos_emb."""
    torch.manual_seed(42)
    mixer = TemporalDepthwiseMixer(
        channels=8, kernel_size=3, dilation=1,
        attention_enabled=True, attention_num_heads=2,
        attention_max_timesteps=5,
    )
    assert mixer.self_attention.pos_emb is not None
    assert mixer.self_attention.pos_emb.shape == (1, 5, 8)

    x = torch.randn(2, 5, 10, 8)  # (B, T, N, C)
    y = mixer(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    y.sum().backward()
    assert mixer.self_attention.pos_emb.grad is not None


def test_attention_pos_emb_none_by_default():
    """max_timesteps=None means no positional embedding."""
    mixer = TemporalDepthwiseMixer(
        channels=8, kernel_size=3, dilation=1,
        attention_enabled=True, attention_num_heads=2,
        attention_max_timesteps=None,
    )
    assert mixer.self_attention.pos_emb is None


def test_attention_pos_emb_t_exceeds_max_raises():
    """T=10 input with max_timesteps=5 raises ValueError."""
    mixer = TemporalDepthwiseMixer(
        channels=8, kernel_size=3, dilation=1,
        attention_enabled=True, attention_num_heads=2,
        attention_max_timesteps=5,
    )
    x = torch.randn(2, 10, 4, 8)  # T=10 > max_timesteps=5
    with pytest.raises(ValueError, match="exceeds max_timesteps"):
        mixer(x)


def test_attention_pos_emb_config_mapping():
    """temporal_mixer.attention.max_timesteps maps to flat GNNConfig field."""
    cfg = GNNConfig(
        temporal_mixer={
            'enabled': True,
            'attention': {
                'enabled': True,
                'max_timesteps': 10,
            },
        },
    )
    assert cfg.temporal_attention_max_timesteps == 10

    # Default is None
    cfg_default = GNNConfig()
    assert cfg_default.temporal_attention_max_timesteps is None


def test_attention_pos_emb_max_timesteps_zero_raises():
    """max_timesteps=0 raises ValueError at module level."""
    with pytest.raises(ValueError, match="max_timesteps must be >= 1"):
        _TemporalSelfAttention(channels=8, num_heads=2, max_timesteps=0)


def test_attention_pos_emb_max_timesteps_negative_raises():
    """max_timesteps=-1 raises at both module and config levels."""
    with pytest.raises(ValueError, match="max_timesteps must be >= 1"):
        _TemporalSelfAttention(channels=8, num_heads=2, max_timesteps=-1)

    with pytest.raises(ValueError, match="temporal_attention_max_timesteps must be >= 1"):
        GNNConfig(temporal_attention_max_timesteps=-1)


# ─── Per-block positional embedding tests ────────────────────────────────────


def test_gnn_attention_per_block_pos_emb_smoke():
    """per_block + attention + max_timesteps: forward + backward + pos_emb present."""
    torch.manual_seed(42)
    B, T, N, C = 2, 5, 16, 8
    gnn = GNN(
        in_channels=C, hidden_channels=C, out_channels=C,
        num_layers=2, K=2,
        use_temporal_mixer=True,
        temporal_mixer_schedule='per_block',
        temporal_dilations=[1],
        temporal_attention_enabled=True,
        temporal_attention_num_heads=2,
        temporal_attention_max_timesteps=5,
    )
    assert gnn.block_temporal_mixer.self_attention.pos_emb is not None
    assert gnn.block_temporal_mixer.self_attention.pos_emb.shape == (1, 5, C)

    edge_index, edge_weight = _build_batched_ring_graph(B, N)
    x = torch.randn(B, T, N, C)
    y = gnn(x, edge_index, edge_weight=edge_weight)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    y.sum().backward()
    assert gnn.block_temporal_mixer.self_attention.pos_emb.grad is not None


def test_gnn_attention_per_block_pos_emb_t_exceeds_max_raises():
    """per_block + attention + T > max_timesteps raises ValueError."""
    torch.manual_seed(42)
    B, T, N, C = 2, 10, 16, 8
    gnn = GNN(
        in_channels=C, hidden_channels=C, out_channels=C,
        num_layers=2, K=2,
        use_temporal_mixer=True,
        temporal_mixer_schedule='per_block',
        temporal_dilations=[1],
        temporal_attention_enabled=True,
        temporal_attention_num_heads=2,
        temporal_attention_max_timesteps=5,
    )
    edge_index, edge_weight = _build_batched_ring_graph(B, N)
    x = torch.randn(B, T, N, C)
    with pytest.raises(ValueError, match="exceeds max_timesteps"):
        gnn(x, edge_index, edge_weight=edge_weight)


def test_gnn_attention_per_block_no_pos_emb_when_none():
    """per_block + attention + max_timesteps=None: no pos_emb parameter."""
    gnn = GNN(
        in_channels=8, hidden_channels=8, out_channels=8,
        num_layers=2, K=2,
        use_temporal_mixer=True,
        temporal_mixer_schedule='per_block',
        temporal_dilations=[1],
        temporal_attention_enabled=True,
        temporal_attention_num_heads=2,
        temporal_attention_max_timesteps=None,
    )
    assert gnn.block_temporal_mixer.self_attention.pos_emb is None


# ─── TemporalNodeCrossAttention mismatched T tests ────────────────────────────

from graph_signal_diffusion.models.ugnn.ugnn import TemporalNodeCrossAttention
from graph_signal_diffusion.models.components.embeddings import ConditionalTemporalMixerEmbedding
import torch.nn.functional as F


def test_cross_attention_mismatched_t_smoke():
    """T_q=5, T_kv=20: forward + backward with non-causal cross-attention."""
    torch.manual_seed(42)
    B, T_q, T_kv, N, C = 2, 5, 20, 16, 8
    ca = TemporalNodeCrossAttention(channels=C, num_heads=2, causal=False)
    q = torch.randn(B, T_q, N, C, requires_grad=True)
    kv = torch.randn(B, T_kv, N, C)
    out = ca(q, kv)
    assert out.shape == (B, T_q, N, C)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert q.grad is not None


def test_cross_attention_mismatched_t_active_mask():
    """T_q != T_kv with 50% active nodes: inactive nodes produce zeros."""
    torch.manual_seed(42)
    B, T_q, T_kv, N, C = 2, 5, 20, 16, 8
    ca = TemporalNodeCrossAttention(channels=C, num_heads=2, causal=False)
    q = torch.randn(B, T_q, N, C)
    kv = torch.randn(B, T_kv, N, C)
    active_mask = torch.ones(B, N, dtype=torch.bool)
    active_mask[:, N // 2:] = False
    out = ca(q, kv, active_mask=active_mask)
    assert out.shape == (B, T_q, N, C)
    assert (out[:, :, N // 2:, :] == 0).all()
    assert torch.isfinite(out).all()


def test_cross_attention_mismatched_t_causal_and_fallback(monkeypatch):
    """T_q=5, T_kv=20, causal=True: test both SDPA and fallback paths."""
    torch.manual_seed(42)
    B, T_q, T_kv, N, C = 2, 5, 20, 16, 8
    ca = TemporalNodeCrossAttention(channels=C, num_heads=2, causal=True)
    q = torch.randn(B, T_q, N, C)
    kv = torch.randn(B, T_kv, N, C)

    # SDPA path (explicit causal mask for mismatched T)
    out_sdpa = ca(q, kv)
    assert out_sdpa.shape == (B, T_q, N, C)
    assert torch.isfinite(out_sdpa).all()

    # Force fallback path by monkeypatching SDPA to raise
    original_sdpa = F.scaled_dot_product_attention

    def _failing_sdpa(*args, **kwargs):
        raise RuntimeError("forced fallback")

    monkeypatch.setattr(F, "scaled_dot_product_attention", _failing_sdpa)
    out_fallback = ca(q, kv)
    assert out_fallback.shape == (B, T_q, N, C)
    assert torch.isfinite(out_fallback).all()

    # Both paths should produce the same result (deterministic, no dropout)
    torch.testing.assert_close(out_sdpa, out_fallback, atol=1e-5, rtol=1e-5)


def test_cross_attention_equal_t_still_works():
    """T_q == T_kv: regression guard that equal-T case is not broken."""
    torch.manual_seed(42)
    B, T, N, C = 2, 10, 16, 8
    ca = TemporalNodeCrossAttention(channels=C, num_heads=2, causal=False)
    q = torch.randn(B, T, N, C)
    kv = torch.randn(B, T, N, C)
    out = ca(q, kv)
    assert out.shape == (B, T, N, C)
    assert torch.isfinite(out).all()


# ─── Cond encoder method=none tests ──────────────────────────────────────────


def test_cond_encoder_method_none_output_shape():
    """method='none' outputs (B, T_cond, N, embed_dim) with no projection params."""
    enc = ConditionalTemporalMixerEmbedding(
        in_channels=12, embed_dim=32, hidden_channels=[16],
        num_layers=1, kernel_size=3, output_mode='time_varying',
        time_varying_method='none',
    )
    assert enc.temporal_projection_logits is None
    assert 'temporal_projection_logits' not in dict(enc.named_parameters())

    x = torch.randn(2, 20, 8, 12)  # (B, T_cond=20, N=8, F=12)
    out = enc(x)
    assert out.shape == (2, 20, 8, 32)  # (B, T_cond, N, embed_dim)
    assert torch.isfinite(out).all()


def test_cond_encoder_method_none_backward():
    """Gradients flow through output_proj and temporal conv layers."""
    enc = ConditionalTemporalMixerEmbedding(
        in_channels=12, embed_dim=32, hidden_channels=[16],
        num_layers=1, kernel_size=3, output_mode='time_varying',
        time_varying_method='none',
    )
    x = torch.randn(2, 20, 8, 12)
    out = enc(x)
    out.sum().backward()
    # Check gradients on output_proj and mixer conv weights
    assert enc.output_proj.mlp[0].weight.grad is not None
    assert enc.mixers[0].depthwise.weight.grad is not None


def test_cond_encoder_method_none_config_mapping():
    """EmbeddingConfig accepts method='none' via nested config dict."""
    cfg = EmbeddingConfig(
        cond={
            'channels': 12,
            'embed_dim': 32,
            'shared_encoder': {
                'temporal': {
                    'hidden_channels': [16],
                    'num_layers': 1,
                    'output': {
                        'mode': 'time_varying',
                        'time_varying': {
                            'method': 'none',
                        },
                    },
                },
            },
        },
    )
    assert cfg.cond_temporal_output_mode == 'time_varying'
    assert cfg.cond_temporal_time_varying_method == 'none'


# ─── Negative regression: concat/film still reject T mismatch ────────────────


def _build_ugnn_with_fusion(fusion_mode):
    """Helper: build a UGNN for T-mismatch tests."""
    return UGNN(UGNNConfig(
        in_channels=1, out_channels=1, base_channels=8,
        channel_multipliers=[1, 1],
        gnn_config=GNNConfig(
            K=1, num_layers=1, norm_type='layer', dropout=0.0,
            activation='silu', normalize=False,
        ),
        pooling_config=PoolingConfig(
            gamma=[1, 1], pool_K=1,
            selection_method='learned',
            selector_kwargs={'temperature': 1.0},
            temporal_score_agg='mean',
        ),
        embedding_config=EmbeddingConfig(
            time_embed_dim=16, num_timesteps=10,
            cond_channels=2, cond_embed_dim=8,
            cond_fusion_mode=fusion_mode,
            cond_projection_per_level=True,
            cond_cross_attn_heads=2,
        ),
        num_bottleneck_layers=1,
    )).eval()


def test_encoder_block_concat_rejects_t_mismatch():
    """concat fusion with T_cond != T_denoiser raises ValueError."""
    model = _build_ugnn_with_fusion('concat')
    B, T, N = 1, 5, 5
    x = torch.randn(B, T, N, 1)
    timesteps = torch.tensor([0], dtype=torch.long)
    edge_index, edge_weight = _build_batched_ring_graph(B, N)
    # Inject mismatched cond_emb via cond_emb_precomputed on encoder
    cond_emb_bad = torch.randn(B, 20, N, 8)  # T_cond=20 != T=5
    with pytest.raises(ValueError, match="must match T="):
        model.encoder(
            x=x, timesteps=timesteps, edge_index=edge_index,
            edge_weight=edge_weight, cond_emb_precomputed=cond_emb_bad,
            time_emb=model.time_embed(timesteps),
        )


def test_encoder_block_film_rejects_t_mismatch():
    """film fusion with T_cond != T_denoiser raises ValueError."""
    model = _build_ugnn_with_fusion('film')
    B, T, N = 1, 5, 5
    x = torch.randn(B, T, N, 1)
    timesteps = torch.tensor([0], dtype=torch.long)
    edge_index, edge_weight = _build_batched_ring_graph(B, N)
    cond_emb_bad = torch.randn(B, 20, N, 8)
    with pytest.raises(ValueError, match="must match T="):
        model.encoder(
            x=x, timesteps=timesteps, edge_index=edge_index,
            edge_weight=edge_weight, cond_emb_precomputed=cond_emb_bad,
            time_emb=model.time_embed(timesteps),
        )


# ─── UGNN end-to-end: cross_attention + method=none ──────────────────────────


def test_ugnn_cross_attention_method_none_e2e():
    """UGNN with cross_attention fusion + method=none: T_cond=20, T_future=5."""
    torch.manual_seed(42)
    config = UGNNConfig(
        in_channels=1, out_channels=1, base_channels=8,
        channel_multipliers=[1, 1],
        gnn_config=GNNConfig(
            K=1, num_layers=1, norm_type='layer', dropout=0.0,
            activation='silu', normalize=False,
            use_temporal_mixer=True, temporal_kernel_size=3,
            temporal_causal=False, temporal_use_pointwise=True,
        ),
        pooling_config=PoolingConfig(
            gamma=[1, 1], pool_K=1,
            selection_method='learned',
            selector_kwargs={'temperature': 1.0},
            temporal_score_agg='mean',
        ),
        embedding_config=EmbeddingConfig(
            time_embed_dim=16, num_timesteps=10,
            cond_channels=2, cond_embed_dim=8,
            cond_temporal_num_layers=1,
            cond_temporal_kernel_size=3,
            cond_temporal_output_mode='time_varying',
            cond_temporal_time_varying_method='none',
            cond_fusion_mode='cross_attention',
            cond_projection_per_level=True,
            cond_cross_attn_heads=2,
            cond_cross_attn_dropout=0.0,
            cond_cross_attn_bias=True,
            cond_cross_attn_causal=False,
        ),
        num_bottleneck_layers=1,
    )
    model = UGNN(config)

    B, T_future, T_cond, N = 1, 5, 20, 5
    x = torch.randn(B, T_future, N, 1)
    cond = torch.randn(B, T_cond, N, 2)
    timesteps = torch.tensor([3], dtype=torch.long)
    edge_index, edge_weight = _build_batched_ring_graph(B, N)

    y, _ = model(
        x=x, timesteps=timesteps,
        edge_index=edge_index, edge_weight=edge_weight,
        cond=cond, return_intermediates=False,
    )
    assert y.shape == (B, T_future, N, 1)
    assert torch.isfinite(y).all()
    # Backward
    y.sum().backward()
    # Cond encoder grads should flow
    assert model.encoder.cond_encoder.output_proj.mlp[0].weight.grad is not None


def test_ugnn_cross_attention_learned_projection_backward_compat():
    """method=learned_projection + cross_attention with T_q == T_kv still works."""
    torch.manual_seed(42)
    config = UGNNConfig(
        in_channels=1, out_channels=1, base_channels=8,
        channel_multipliers=[1, 1],
        gnn_config=GNNConfig(
            K=1, num_layers=1, norm_type='layer', dropout=0.0,
            activation='silu', normalize=False,
        ),
        pooling_config=PoolingConfig(
            gamma=[1, 1], pool_K=1,
            selection_method='learned',
            selector_kwargs={'temperature': 1.0},
            temporal_score_agg='mean',
        ),
        embedding_config=EmbeddingConfig(
            time_embed_dim=16, num_timesteps=10,
            cond_channels=2, cond_embed_dim=8,
            cond_temporal_num_layers=1,
            cond_temporal_kernel_size=3,
            cond_temporal_output_mode='time_varying',
            cond_temporal_time_varying_method='learned_projection',
            cond_temporal_projection_source_timesteps=20,
            cond_temporal_projection_target_timesteps=5,
            cond_fusion_mode='cross_attention',
            cond_projection_per_level=True,
            cond_cross_attn_heads=2,
            cond_cross_attn_dropout=0.0,
            cond_cross_attn_bias=True,
            cond_cross_attn_causal=False,
        ),
        num_bottleneck_layers=1,
    )
    model = UGNN(config).eval()

    B, T_future, T_cond, N = 1, 5, 20, 5
    x = torch.randn(B, T_future, N, 1)
    cond = torch.randn(B, T_cond, N, 2)
    timesteps = torch.tensor([3], dtype=torch.long)
    edge_index, edge_weight = _build_batched_ring_graph(B, N)

    with torch.no_grad():
        y, _ = model(
            x=x, timesteps=timesteps,
            edge_index=edge_index, edge_weight=edge_weight,
            cond=cond, return_intermediates=False,
        )
    assert y.shape == (B, T_future, N, 1)
    assert torch.isfinite(y).all()
