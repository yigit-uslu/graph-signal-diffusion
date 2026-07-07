"""Tests for the gated + temporal self-attention plumbing in
``ConditionalTemporalMixerEmbedding`` and ``EmbeddingConfig``.

These cover three layers of plumbing:
  1. Constructor kwargs reach every internal ``TemporalDepthwiseMixer``.
  2. ``EmbeddingConfig`` flat fields can be set both directly and via the
     nested ``cond.shared_encoder.temporal.mixer.{gated,attention.*}`` YAML.
  3. The ``_build_conditional_encoder`` helper plumbs the flat fields into
     the actual cond-encoder modules built inside a full ``UGNN``.

The wiring follows the same pattern as the pre-existing ``kernel_size``,
``use_pointwise``, ``dilations`` plumbing (see
test_grouped_config_mapping_for_temporal_and_cond in
test_temporal_mixer_smoke.py).
"""
from __future__ import annotations

import pytest
import torch

from graph_signal_diffusion.models.components.embeddings import (
    ConditionalTemporalMixerEmbedding,
)
from graph_signal_diffusion.models.components.graph_conv import (
    TemporalDepthwiseMixer,
    _TemporalSelfAttention,
)
from graph_signal_diffusion.models.ugnn import EmbeddingConfig


# ---------------------------------------------------------------------------
# (1) Constructor kwargs reach every internal mixer
# ---------------------------------------------------------------------------

def test_cond_encoder_gated_default_is_false():
    """Backward compat: gated kwarg defaults to False; all mixers remain non-gated."""
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
    assert len(encoder.mixers) == 2
    for m in encoder.mixers:
        assert isinstance(m, TemporalDepthwiseMixer)
        assert m.gated is False
        assert m.self_attention is None
        # depthwise out_channels = C in non-gated mode
        assert m.depthwise.out_channels == m.channels


def test_cond_encoder_gated_true_flows_to_all_mixers():
    """gated=True on the embedding constructor ⇒ every mixer has gated=True,
    depthwise out_channels=2*C, and an extra gate_proj 1x1 conv."""
    encoder = ConditionalTemporalMixerEmbedding(
        in_channels=3,
        embed_dim=4,
        hidden_channels=[6, 6, 6],
        num_layers=3,
        kernel_size=3,
        pooling="mean",
        activation="silu",
        dropout=0.0,
        norm_type="layer",
        use_pointwise=True,
        causal=False,
        gated=True,
    )
    assert len(encoder.mixers) == 3
    for m in encoder.mixers:
        assert m.gated is True
        # gated depthwise emits 2*C output channels paired filter/gate per group
        assert m.depthwise.out_channels == 2 * m.channels
        assert hasattr(m, "gate_proj")
        assert m.gate_proj.in_channels == m.channels
        assert m.gate_proj.out_channels == m.channels
        assert m.gate_proj.kernel_size == (1,)


def test_cond_encoder_attention_true_flows_to_all_mixers():
    """attention_enabled=True ⇒ every mixer has a _TemporalSelfAttention with
    matching num_heads / dropout / max_timesteps."""
    encoder = ConditionalTemporalMixerEmbedding(
        in_channels=3,
        embed_dim=4,
        hidden_channels=[8, 8],
        num_layers=2,
        kernel_size=3,
        pooling="mean",
        activation="silu",
        dropout=0.0,
        norm_type="layer",
        use_pointwise=True,
        causal=False,
        attention_enabled=True,
        attention_num_heads=2,
        attention_dropout=0.1,
        attention_max_timesteps=20,
    )
    for m in encoder.mixers:
        assert m.self_attention is not None
        assert isinstance(m.self_attention, _TemporalSelfAttention)
        assert m.self_attention.num_heads == 2
        assert m.self_attention.dropout == pytest.approx(0.1)
        # max_timesteps controls the learnable positional embedding size
        assert m.self_attention.pos_emb is not None
        assert m.self_attention.pos_emb.shape == (1, 20, m.channels)


def test_cond_encoder_gated_and_attention_combined():
    """Both knobs may be enabled together — the per-channel gated activation
    is layered before the temporal self-attention."""
    encoder = ConditionalTemporalMixerEmbedding(
        in_channels=3,
        embed_dim=4,
        hidden_channels=[8, 8],
        num_layers=2,
        kernel_size=3,
        pooling="mean",
        activation="silu",
        dropout=0.0,
        norm_type="layer",
        use_pointwise=True,
        causal=False,
        gated=True,
        attention_enabled=True,
        attention_num_heads=4,
        attention_dropout=0.0,
        attention_max_timesteps=10,
    )
    for m in encoder.mixers:
        assert m.gated is True
        assert hasattr(m, "gate_proj")
        assert m.self_attention is not None
        assert m.self_attention.num_heads == 4


# ---------------------------------------------------------------------------
# (2) Forward smoke — gated + attention must preserve output shape
# ---------------------------------------------------------------------------

def test_cond_encoder_gated_attention_forward_static_smoke():
    torch.manual_seed(0)
    B, T_cond, N, F_cond = 2, 8, 4, 3
    encoder = ConditionalTemporalMixerEmbedding(
        in_channels=F_cond,
        embed_dim=8,
        hidden_channels=[8, 8],
        num_layers=2,
        kernel_size=3,
        pooling="mean",
        activation="silu",
        dropout=0.0,
        norm_type="layer",
        use_pointwise=True,
        causal=False,
        gated=True,
        attention_enabled=True,
        attention_num_heads=2,
        attention_dropout=0.0,
        attention_max_timesteps=T_cond,
    )
    x = torch.randn(B, T_cond, N, F_cond)
    y = encoder(x)
    assert y.shape == (B, N, 8)
    assert torch.isfinite(y).all()


def test_cond_encoder_gated_attention_forward_time_varying_smoke():
    torch.manual_seed(0)
    B, T_cond, T_out, N, F_cond = 2, 20, 5, 4, 3
    encoder = ConditionalTemporalMixerEmbedding(
        in_channels=F_cond,
        embed_dim=8,
        hidden_channels=[8, 8],
        num_layers=2,
        kernel_size=5,
        pooling="mean",
        activation="silu",
        dropout=0.0,
        norm_type="layer",
        use_pointwise=True,
        causal=False,
        output_mode="time_varying",
        time_varying_method="learned_projection",
        projection_source_timesteps=T_cond,
        projection_target_timesteps=T_out,
        dilations=[1, 2],
        gated=True,
        attention_enabled=True,
        attention_num_heads=2,
        attention_dropout=0.0,
        attention_max_timesteps=T_cond,
    )
    x = torch.randn(B, T_cond, N, F_cond)
    y = encoder(x, target_timesteps=T_out)
    assert y.shape == (B, T_out, N, 8)
    assert torch.isfinite(y).all()


# ---------------------------------------------------------------------------
# (3) Param-count delta sanity check
# ---------------------------------------------------------------------------

def _total_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def test_cond_encoder_gated_param_count_increases():
    """Switching gated on increases parameter count by the expected amount per
    layer: extra depthwise channels (k * C) + new gate_proj (C * C)."""
    kwargs = dict(
        in_channels=3,
        embed_dim=4,
        hidden_channels=[8, 8],
        num_layers=2,
        kernel_size=3,
        pooling="mean",
        activation="silu",
        dropout=0.0,
        norm_type="layer",
        use_pointwise=True,
        causal=False,
    )
    encoder_off = ConditionalTemporalMixerEmbedding(**kwargs, gated=False)
    encoder_on = ConditionalTemporalMixerEmbedding(**kwargs, gated=True)
    delta = _total_params(encoder_on) - _total_params(encoder_off)
    # Per layer at C=8, kernel=3:
    #   depthwise: 8*3=24 → 16*3=48      (+24)
    #   gate_proj: 0     → 8*8=64        (+64)
    # → +88 per layer × 2 layers = +176 total
    assert delta == 2 * (24 + 64)


def test_cond_encoder_attention_param_count_increases():
    """Switching attention on adds q/k/v/out projections + LayerNorm + pos_emb."""
    kwargs = dict(
        in_channels=3,
        embed_dim=4,
        hidden_channels=[8, 8],
        num_layers=2,
        kernel_size=3,
        pooling="mean",
        activation="silu",
        dropout=0.0,
        norm_type="layer",
        use_pointwise=True,
        causal=False,
    )
    encoder_off = ConditionalTemporalMixerEmbedding(**kwargs, attention_enabled=False)
    encoder_on = ConditionalTemporalMixerEmbedding(
        **kwargs,
        attention_enabled=True,
        attention_num_heads=2,
        attention_dropout=0.0,
        attention_max_timesteps=10,
    )
    # At C=8, max_timesteps=10:
    #   LayerNorm: 8*2 = 16
    #   q,k,v,out_proj: each Linear(8,8) → 8*8 + 8 = 72; ×4 = 288
    #   pos_emb: 1*10*8 = 80
    # → 16 + 288 + 80 = 384 per layer × 2 = 768
    delta = _total_params(encoder_on) - _total_params(encoder_off)
    assert delta == 2 * (16 + 4 * (8 * 8 + 8) + 10 * 8)


# ---------------------------------------------------------------------------
# (4) EmbeddingConfig YAML-path: nested config flows to flat fields
# ---------------------------------------------------------------------------

def test_embedding_config_defaults_for_new_fields():
    cfg = EmbeddingConfig()
    assert cfg.cond_temporal_gated is False
    assert cfg.cond_temporal_attention_enabled is False
    assert cfg.cond_temporal_attention_num_heads == 4
    assert cfg.cond_temporal_attention_dropout == pytest.approx(0.0)
    assert cfg.cond_temporal_attention_max_timesteps is None


def test_embedding_config_nested_gated_maps_to_flat_field():
    cfg = EmbeddingConfig(
        cond={
            "channels": 2,
            "embed_dim": 8,
            "shared_encoder": {
                "temporal": {
                    "hidden_channels": [8, 8],
                    "num_layers": 2,
                    "mixer": {
                        "kernel_size": 3,
                        "causal": False,
                        "use_pointwise": True,
                        "dilations": [1, 2],
                        "gated": True,
                    },
                },
            },
        }
    )
    assert cfg.cond_temporal_gated is True
    # Other flat fields still get extracted as before.
    assert cfg.cond_temporal_kernel_size == 3
    assert cfg.cond_temporal_dilations == [1, 2]
    # Attention block absent → defaults preserved.
    assert cfg.cond_temporal_attention_enabled is False


def test_embedding_config_nested_attention_block_maps_to_flat_fields():
    cfg = EmbeddingConfig(
        cond={
            "channels": 2,
            "embed_dim": 8,
            "shared_encoder": {
                "temporal": {
                    "hidden_channels": [8],
                    "num_layers": 1,
                    "mixer": {
                        "kernel_size": 3,
                        "attention": {
                            "enabled": True,
                            "num_heads": 8,
                            "dropout": 0.25,
                            "max_timesteps": 17,
                        },
                    },
                },
            },
        }
    )
    assert cfg.cond_temporal_attention_enabled is True
    assert cfg.cond_temporal_attention_num_heads == 8
    assert cfg.cond_temporal_attention_dropout == pytest.approx(0.25)
    assert cfg.cond_temporal_attention_max_timesteps == 17
    # Gated unset → default preserved.
    assert cfg.cond_temporal_gated is False


def test_embedding_config_nested_gated_and_attention_combined_maps_correctly():
    cfg = EmbeddingConfig(
        cond={
            "channels": 2,
            "embed_dim": 16,
            "shared_encoder": {
                "temporal": {
                    "hidden_channels": [16, 16],
                    "num_layers": 2,
                    "mixer": {
                        "kernel_size": 5,
                        "causal": False,
                        "use_pointwise": True,
                        "dilations": [1, 2],
                        "gated": True,
                        "attention": {
                            "enabled": True,
                            "num_heads": 4,
                            "dropout": 0.1,
                            "max_timesteps": 20,
                        },
                    },
                },
            },
        }
    )
    assert cfg.cond_temporal_gated is True
    assert cfg.cond_temporal_attention_enabled is True
    assert cfg.cond_temporal_attention_num_heads == 4
    assert cfg.cond_temporal_attention_dropout == pytest.approx(0.1)
    assert cfg.cond_temporal_attention_max_timesteps == 20


# ---------------------------------------------------------------------------
# (5) End-to-end: EmbeddingConfig → _build_conditional_encoder → cond mixers
# ---------------------------------------------------------------------------

def test_embedding_config_to_cond_encoder_propagates_gated_and_attention():
    """Constructing the cond encoder directly from a fully-populated
    EmbeddingConfig must produce mixers with gated=True and self-attention
    matching the config."""
    from graph_signal_diffusion.models.ugnn.ugnn import _build_conditional_encoder
    from graph_signal_diffusion.models.ugnn import UGNNConfig, GNNConfig

    emb_cfg = EmbeddingConfig(
        cond_channels=3,
        cond_embed_dim=8,
        cond_temporal_hidden_channels=[8, 8],
        cond_temporal_num_layers=2,
        cond_temporal_kernel_size=3,
        cond_temporal_use_pointwise=True,
        cond_temporal_causal=False,
        cond_temporal_gated=True,
        cond_temporal_attention_enabled=True,
        cond_temporal_attention_num_heads=2,
        cond_temporal_attention_dropout=0.0,
        cond_temporal_attention_max_timesteps=10,
    )
    # _build_conditional_encoder uses config.gnn_config for activation/dropout/norm_type
    # only; a minimal GNNConfig is enough.
    parent_cfg = UGNNConfig(
        in_channels=1,
        out_channels=1,
        base_channels=8,
        embedding_config=emb_cfg,
        gnn_config=GNNConfig(),
    )
    encoder = _build_conditional_encoder(parent_cfg)
    assert isinstance(encoder, ConditionalTemporalMixerEmbedding)
    assert len(encoder.mixers) == 2
    for m in encoder.mixers:
        assert m.gated is True
        assert m.depthwise.out_channels == 2 * m.channels
        assert hasattr(m, "gate_proj")
        assert m.self_attention is not None
        assert m.self_attention.num_heads == 2
        assert m.self_attention.pos_emb is not None
        assert m.self_attention.pos_emb.shape == (1, 10, m.channels)


def test_embedding_config_defaults_keep_cond_encoder_backwards_compat():
    """When the new fields are not set, the cond encoder behaves exactly as
    before (gated=False, no self-attention)."""
    from graph_signal_diffusion.models.ugnn.ugnn import _build_conditional_encoder
    from graph_signal_diffusion.models.ugnn import UGNNConfig, GNNConfig

    emb_cfg = EmbeddingConfig(
        cond_channels=3,
        cond_embed_dim=8,
        cond_temporal_hidden_channels=[8, 8],
        cond_temporal_num_layers=2,
        cond_temporal_kernel_size=3,
        cond_temporal_use_pointwise=True,
        cond_temporal_causal=False,
        # gated + attention deliberately unset → defaults
    )
    parent_cfg = UGNNConfig(
        in_channels=1,
        out_channels=1,
        base_channels=8,
        embedding_config=emb_cfg,
        gnn_config=GNNConfig(),
    )
    encoder = _build_conditional_encoder(parent_cfg)
    for m in encoder.mixers:
        assert m.gated is False
        assert m.depthwise.out_channels == m.channels
        assert m.self_attention is None
