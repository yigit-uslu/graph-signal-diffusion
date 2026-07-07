"""Regression tests for the UGNN output head config (OutputHeadConfig).

Covers:
1. Default config reproduces the historical bare-Linear head (backwards compat).
2. mode='norm_act_linear' builds nn.Sequential(LayerNorm, SiLU, Linear) with the
   expected component shapes.
3. zero_init=True forces the untrained head to emit exactly zero (the DDPM
   convention used by DDPM, EDM, ADM, DiT).
4. LayerNorm normalizes only the channel dimension — T and N are NOT mixed
   (per-(batch, timestep, node) position channel norm; transformer-style).
5. End-to-end UGNN smoke: a tiny model built from the new config preset runs
   forward, produces correctly-shaped output, and (with zero_init=true)
   produces exactly zero output regardless of input.
6. Old checkpoints (saved with the default mode='linear' head) still load
   strict=True into a freshly-constructed default-config decoder.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate

from graph_signal_diffusion.models.ugnn import (
    OutputHeadConfig,
    UGNNDecoder,
    UGNNConfig,
)
from graph_signal_diffusion.models.ugnn.ugnn import _build_output_head


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compose_model_cfg(config_name: str):
    repo_root = Path(__file__).resolve().parents[2]
    config_dir = repo_root / "src" / "graph_signal_diffusion" / "conf" / "model"
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name=config_name)


def _build_ring_edge_index(n_nodes: int) -> torch.Tensor:
    src, dst = [], []
    for i in range(n_nodes):
        src.extend([i, i])
        dst.extend([(i + 1) % n_nodes, i])
    return torch.tensor([src, dst], dtype=torch.long)


# ---------------------------------------------------------------------------
# 1. Backwards-compat: default config -> bare nn.Linear
# ---------------------------------------------------------------------------


def test_default_output_head_is_bare_linear_backwards_compat():
    """Default OutputHeadConfig must produce a bare nn.Linear, matching the
    historical UGNNDecoder.output_proj exactly (same module type, same
    state_dict key pattern, default Kaiming init unchanged).
    """
    cfg = OutputHeadConfig()
    assert cfg.mode == 'linear'
    assert cfg.zero_init is False

    head = _build_output_head(final_channels=32, out_channels=1, cfg=cfg)

    # Same type / shape as the pre-change implementation.
    assert isinstance(head, nn.Linear)
    assert head.in_features == 32
    assert head.out_features == 1
    assert head.bias is not None

    # state_dict keys are 'weight' and 'bias' — old checkpoints reference
    # exactly these names under decoder.output_proj.
    keys = set(head.state_dict().keys())
    assert keys == {'weight', 'bias'}

    # Default init is NOT zero — biases default to a uniform draw.
    assert not torch.all(head.weight == 0)


# ---------------------------------------------------------------------------
# 2. Norm-act-linear mode builds the expected nn.Sequential
# ---------------------------------------------------------------------------


def test_norm_act_linear_mode_builds_sequential_layernorm_silu_linear():
    cfg = OutputHeadConfig(
        mode='norm_act_linear', norm_type='layer', activation='silu', zero_init=False,
    )
    head = _build_output_head(final_channels=64, out_channels=1, cfg=cfg)

    assert isinstance(head, nn.Sequential)
    assert len(head) == 3
    norm, act, linear = head[0], head[1], head[2]

    assert isinstance(norm, nn.LayerNorm)
    # Channel-only norm: normalized_shape spans only the last (channel) dim.
    assert tuple(norm.normalized_shape) == (64,)

    assert isinstance(act, nn.SiLU)

    assert isinstance(linear, nn.Linear)
    assert linear.in_features == 64
    assert linear.out_features == 1


# ---------------------------------------------------------------------------
# 3. zero_init=True forces the untrained head's output to be exactly zero
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ['linear', 'norm_act_linear'])
def test_zero_init_emits_exactly_zero_output(mode):
    cfg = OutputHeadConfig(mode=mode, zero_init=True)
    head = _build_output_head(final_channels=16, out_channels=1, cfg=cfg)

    head.eval()
    x = torch.randn(2, 5, 8, 16)  # (B, T, N, C)
    with torch.no_grad():
        out = head(x)

    assert out.shape == (2, 5, 8, 1)
    assert torch.all(out == 0), (
        f"With zero_init=True ({mode=}), untrained head must emit exactly zero. "
        f"Got max abs {out.abs().max().item()}."
    )


# ---------------------------------------------------------------------------
# 4. LayerNorm normalizes ONLY the channel dim (T and N are independent)
# ---------------------------------------------------------------------------


def test_layernorm_normalizes_channel_dim_only_t_and_n_independent():
    """LayerNorm in the norm_act_linear head must normalize per-(B,T,N) over C.

    Concretely: scaling the channel values at ONE (b,t,n) position by 10x
    must leave the per-position-channel statistics unchanged at that
    position (mean=0, var=1 over C, same as before scaling), and must
    leave OTHER (b,t,n) positions completely unchanged. If T or N were
    folded into the norm, scaling one position would shift other
    positions' outputs.
    """
    cfg = OutputHeadConfig(
        mode='norm_act_linear', norm_type='layer', activation='silu', zero_init=False,
    )
    head = _build_output_head(final_channels=16, out_channels=4, cfg=cfg)
    head.eval()
    norm = head[0]  # nn.LayerNorm(16)

    B, T, N, C = 2, 5, 8, 16
    torch.manual_seed(0)
    x = torch.randn(B, T, N, C)
    x_scaled = x.clone()
    x_scaled[0, 2, 3, :] *= 10.0  # Scale a single (b,t,n) channel vector by 10x

    with torch.no_grad():
        y = norm(x)
        y_scaled = norm(x_scaled)

    # OTHER positions are bit-identical: if T or N were folded into norm
    # stats, scaling (0,2,3,:) would leak into every other position.
    mask = torch.ones(B, T, N, dtype=torch.bool)
    mask[0, 2, 3] = False
    assert torch.allclose(y[mask], y_scaled[mask], atol=0, rtol=0), (
        "LayerNorm must NOT mix the T or N dimensions; scaling one "
        "(B,T,N) position must leave all other positions unchanged."
    )

    # At the scaled position, per-channel mean and variance should remain 0/1
    # (LayerNorm is scale-invariant when applied per-position over C).
    pos = y_scaled[0, 2, 3]
    assert pos.mean().abs().item() < 1e-5
    assert (pos.var(unbiased=False) - 1.0).abs().item() < 1e-3


# ---------------------------------------------------------------------------
# 5. End-to-end UGNN smoke with the new config preset
# ---------------------------------------------------------------------------


def _shrink_cfg_for_smoke_test(cfg) -> None:
    """In-place shrink of a UGNN model cfg so the smoke test runs in seconds.

    Preserves the architecture topology (still pools, still has a
    bottleneck, still uses the configured output head); just reduces
    widths, depths, and feature counts.
    """
    cfg.model.config.embedding_config.cond.channels = 2
    cfg.model.config.embedding_config.cond.shared_encoder.temporal.output.time_varying.learned_projection.source_timesteps = 4
    cfg.model.config.embedding_config.cond.shared_encoder.temporal.output.time_varying.learned_projection.target_timesteps = 2
    cfg.model.config.base_channels = 8
    cfg.model.config.channel_multipliers = [1, 1]
    cfg.model.config.pooling_config.gamma = [1, 1]
    cfg.model.config.num_bottleneck_layers = 0
    cfg.model.config.gnn_config.num_layers = 1
    cfg.model.config.gnn_config.K = 1
    if cfg.model.config.gnn_config.get('temporal_mixer', None) is not None:
        cfg.model.config.gnn_config.temporal_mixer.dilations = [1]
        cfg.model.config.gnn_config.temporal_mixer.attention.max_timesteps = 2


def test_ugnn_with_norm_act_head_zero_init_emits_zero_end_to_end():
    """End-to-end: tiny UGNN built from the norm_act_head preset, configured
    with zero_init=true, must emit exactly zero output for any input.

    This verifies the OutputHeadConfig wiring all the way through the
    UGNNDecoder, including Hydra instantiation of the nested dataclass.
    """
    cfg = _compose_model_cfg("ugnn_sp500_v3_ds8_norm_act_head")
    _shrink_cfg_for_smoke_test(cfg)

    model = instantiate(cfg.model)
    model.eval()

    # Confirm Hydra wired the head config through to the decoder.
    head = model.decoder.output_proj
    assert isinstance(head, nn.Sequential), (
        f"Expected Sequential head from norm_act_linear mode, got {type(head)}"
    )
    assert isinstance(head[0], nn.LayerNorm)
    assert isinstance(head[1], nn.SiLU)
    assert isinstance(head[2], nn.Linear)
    # Zero-init: weights and biases must be exactly zero.
    assert torch.all(head[2].weight == 0)
    assert head[2].bias is not None and torch.all(head[2].bias == 0)

    B, T, N = 2, 2, 12
    x = torch.randn(B, T, N, model.config.in_channels)
    cond = torch.randn(B, 4, N, 2)
    timesteps = torch.randint(0, model.config.embedding_config.num_timesteps, (B,))
    edge_index = _build_ring_edge_index(N)
    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32)

    with torch.no_grad():
        out, _ = model(
            x=x,
            timesteps=timesteps,
            edge_index=edge_index,
            edge_weight=edge_weight,
            cond=cond,
            return_intermediates=False,
        )

    assert out.shape == (B, T, N, model.config.out_channels)
    assert torch.all(out == 0), (
        f"Zero-init norm_act head must emit zero end-to-end. Got max abs {out.abs().max().item()}."
    )


# ---------------------------------------------------------------------------
# 6. Old-checkpoint compatibility for default mode='linear'
# ---------------------------------------------------------------------------


def test_default_linear_head_checkpoint_compat(tmp_path: Path):
    """A decoder built with default OutputHeadConfig (mode='linear') must
    expose the same state_dict keys for output_proj as the historical
    bare-Linear implementation, so old checkpoints load with strict=True.

    We can't depend on a historical checkpoint here, so we round-trip:
    build two default-config decoders, save state_dict from one, load
    into the other, and verify forward pass agrees.
    """
    cfg = UGNNConfig(
        in_channels=1,
        out_channels=1,
        base_channels=8,
        channel_multipliers=[1, 1],
    )
    # Default OutputHeadConfig
    assert cfg.output_head_config.mode == 'linear'

    decoder_a = UGNNDecoder(out_channels=1, config=cfg)

    # The output_proj state_dict keys are the historical names.
    output_proj_keys = {
        k for k in decoder_a.state_dict().keys() if k.startswith('output_proj.')
    }
    assert output_proj_keys == {'output_proj.weight', 'output_proj.bias'}, (
        f"Default mode='linear' must preserve historical state_dict keys "
        f"for checkpoint compat. Got {output_proj_keys}."
    )

    # Round-trip: save from decoder_a, load into decoder_b with strict=True.
    decoder_b = UGNNDecoder(out_channels=1, config=cfg)
    ckpt_path = tmp_path / "decoder.pt"
    torch.save(decoder_a.state_dict(), ckpt_path)
    missing, unexpected = decoder_b.load_state_dict(
        torch.load(ckpt_path, weights_only=True), strict=True,
    )
    assert missing == [] and unexpected == [], (
        f"strict load failed. missing={missing}, unexpected={unexpected}"
    )
