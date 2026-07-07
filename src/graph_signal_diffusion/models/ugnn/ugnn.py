"""
U-Net style Graph Neural Network (UGNN) for diffusion models on graph signals.

This module implements a hierarchical GNN architecture with:
- Encoder blocks: Process features → downsample/pool via StridedGraphMaxPool
- Decoder blocks: Upsample → fuse with skip connections → process features
- Multi-scale processing: Each level operates at different graph resolutions

Key Design:
- Same graph structure (edge_index, edge_weight) across all levels
- Pooling = selecting active nodes via StridedGraphMaxPool
- Skip connections between encoder and decoder at matching resolutions
- Time and conditional embeddings fused at each level

Architecture:
    Input (N nodes) 
    ↓
    [Encoder Block 1] → N active nodes
    ↓ pool (γ=2)
    [Encoder Block 2] → N/2 active nodes
    ↓ pool (γ=2)
    [Encoder Block 3] → N/4 active nodes
    ↓ pool (γ=2)
    [Bottleneck] → N/8 active nodes
    ↑ unpool
    [Decoder Block 3] ← skip connection from Encoder 3
    ↑ unpool
    [Decoder Block 2] ← skip connection from Encoder 2
    ↑ unpool
    [Decoder Block 1] ← skip connection from Encoder 1
    ↓
    Output (N nodes)

References:
    - U-Net: Ronneberger et al. "U-Net: Convolutional Networks for 
      Biomedical Image Segmentation" (2015)
    - Graph U-Net: Gao & Ji "Graph U-Nets" (2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Literal, Union, Dict, Any
from collections.abc import Mapping
from beartype import beartype
from jaxtyping import jaxtyped, Float, Int, Bool

from graph_signal_diffusion.models.components.graph_conv import (
    ResidualGNNBlock,
    GNN,
    _pack_active_nodes_4d_from_indices,
    _unpack_active_nodes_4d,
)
from graph_signal_diffusion.models.components.pooling import StridedGraphMaxPool
from graph_signal_diffusion.models.components.embeddings import (
    TimeEmbedding,
    ConditionalTemporalMixerEmbedding,
)


def _cfg_map_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Safely read a key from mapping-like config nodes (dict / DictConfig)."""
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return default


def _apply_fusion_proj_without_concat(
    *,
    x: torch.Tensor,
    time_token: torch.Tensor,
    fusion_proj: nn.Linear,
    cond_feats: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply fusion projection using split weights without materializing concat buffers."""
    B, T, N, C = x.shape
    if time_token.shape != (B, C):
        raise ValueError(
            f"time_token must have shape (B, C)=({B}, {C}), got {tuple(time_token.shape)}"
        )

    weight = fusion_proj.weight
    bias = fusion_proj.bias

    if cond_feats is None:
        expected_in = 2 * C
        if int(weight.shape[1]) != expected_in:
            raise ValueError(
                f"fusion_proj in_features mismatch: expected {expected_in}, got {int(weight.shape[1])}"
            )
        # Split order must match original concatenation: [x, time].
        w_x, w_t = torch.split(weight, [C, C], dim=1)
        out = F.linear(x, w_x, bias)
        out = out + F.linear(time_token, w_t, None).unsqueeze(1).unsqueeze(2)
        return out

    if cond_feats.shape[:3] != x.shape[:3] or cond_feats.shape[-1] != C:
        raise ValueError(
            "cond_feats must match x shape [B,T,N,C] for concat fusion path, "
            f"got {tuple(cond_feats.shape)} vs {tuple(x.shape)}"
        )
    expected_in = 3 * C
    if int(weight.shape[1]) != expected_in:
        raise ValueError(
            f"fusion_proj in_features mismatch: expected {expected_in}, got {int(weight.shape[1])}"
        )

    # Split order must match original concatenation: [x, time, cond].
    w_x, w_t, w_c = torch.split(weight, [C, C, C], dim=1)
    out = F.linear(x, w_x, bias)
    out = out + F.linear(time_token, w_t, None).unsqueeze(1).unsqueeze(2)
    out = out + F.linear(cond_feats, w_c, None)
    return out


def _pack_skip_tensor(x: torch.Tensor, active_mask: torch.Tensor) -> Dict[str, Any]:
    """Pack skip tensor from (B,T,N,C) to (A,T,C) using active mask."""
    B, T, N, C = x.shape
    if active_mask.shape != (B, N):
        raise ValueError(
            f"active_mask must match skip tensor (B, N)=({B}, {N}), got {tuple(active_mask.shape)}"
        )
    mask_flat = active_mask.to(dtype=torch.bool).reshape(B * N)
    active_indices = torch.nonzero(mask_flat, as_tuple=False).squeeze(-1)
    x_bn = x.permute(0, 2, 1, 3).reshape(B * N, T, C)
    x_packed = x_bn.index_select(0, active_indices)
    return {
        "kind": "packed_skip_v1",
        "packed": x_packed,
        "active_indices": active_indices,
        "shape": (B, T, N, C),
    }


def _unpack_skip_tensor(skip_repr: Any) -> torch.Tensor:
    """Return dense skip tensor (B,T,N,C) from packed representation or passthrough tensor."""
    if isinstance(skip_repr, torch.Tensor):
        return skip_repr
    if not isinstance(skip_repr, dict) or skip_repr.get("kind") != "packed_skip_v1":
        raise TypeError(
            f"Unsupported skip representation type: {type(skip_repr)} with keys {list(skip_repr.keys()) if isinstance(skip_repr, dict) else None}"
        )

    packed = skip_repr["packed"]
    active_indices = skip_repr["active_indices"]
    B, T, N, C = skip_repr["shape"]
    x_bn = packed.new_zeros((B * N, T, C))
    x_bn.index_copy_(0, active_indices, packed)
    return x_bn.reshape(B, N, T, C).permute(0, 2, 1, 3).contiguous()


def _active_indices_from_mask(active_mask: torch.Tensor) -> torch.Tensor:
    """Flattened active-node indices for a (B, N) boolean mask."""
    if active_mask.dim() != 2:
        raise ValueError(
            f"active_mask must have shape (B, N), got {tuple(active_mask.shape)}"
        )
    return torch.nonzero(active_mask.to(dtype=torch.bool).reshape(-1), as_tuple=False).squeeze(-1)


def _active_batch_indices(
    active_indices: torch.Tensor,
    *,
    num_nodes: int,
) -> torch.Tensor:
    """Map flattened active node indices (B*N) to batch indices (B)."""
    return torch.div(active_indices, num_nodes, rounding_mode='floor')


def _apply_fusion_proj_packed_without_concat(
    *,
    x_packed: torch.Tensor,
    active_indices: torch.Tensor,
    num_nodes: int,
    time_token: torch.Tensor,
    fusion_proj: nn.Linear,
    cond_packed: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Packed counterpart of _apply_fusion_proj_without_concat."""
    if x_packed.dim() != 3:
        raise ValueError(f"x_packed must be rank-3 (A,T,C), got {x_packed.dim()}D")
    A, T, C = x_packed.shape
    if time_token.dim() != 2 or time_token.shape[1] != C:
        raise ValueError(
            f"time_token must have shape (B, C= {C}), got {tuple(time_token.shape)}"
        )

    weight = fusion_proj.weight
    bias = fusion_proj.bias
    batch_indices = _active_batch_indices(active_indices, num_nodes=num_nodes)

    if cond_packed is None:
        expected_in = 2 * C
        if int(weight.shape[1]) != expected_in:
            raise ValueError(
                f"fusion_proj in_features mismatch: expected {expected_in}, got {int(weight.shape[1])}"
            )
        w_x, w_t = torch.split(weight, [C, C], dim=1)
        out = F.linear(x_packed, w_x, bias)
        t_proj = F.linear(time_token, w_t, None).index_select(0, batch_indices).unsqueeze(1)
        out = out + t_proj
        return out

    if cond_packed.shape != x_packed.shape:
        raise ValueError(
            f"cond_packed must match x_packed shape {tuple(x_packed.shape)}, got {tuple(cond_packed.shape)}"
        )
    expected_in = 3 * C
    if int(weight.shape[1]) != expected_in:
        raise ValueError(
            f"fusion_proj in_features mismatch: expected {expected_in}, got {int(weight.shape[1])}"
        )

    w_x, w_t, w_c = torch.split(weight, [C, C, C], dim=1)
    out = F.linear(x_packed, w_x, bias)
    t_proj = F.linear(time_token, w_t, None).index_select(0, batch_indices).unsqueeze(1)
    out = out + t_proj
    out = out + F.linear(cond_packed, w_c, None)
    return out


def _apply_linear_concat_packed(
    *,
    lhs_packed: torch.Tensor,
    rhs_packed: torch.Tensor,
    proj: nn.Linear,
) -> torch.Tensor:
    """Apply linear([lhs, rhs]) on packed tensors without materializing concatenation."""
    if lhs_packed.dim() != 3 or rhs_packed.dim() != 3:
        raise ValueError("lhs_packed and rhs_packed must be rank-3 tensors (A,T,C)")
    if lhs_packed.shape[:2] != rhs_packed.shape[:2]:
        raise ValueError(
            f"Packed lhs/rhs leading dims must match, got {tuple(lhs_packed.shape)} vs {tuple(rhs_packed.shape)}"
        )

    c_lhs = lhs_packed.shape[-1]
    c_rhs = rhs_packed.shape[-1]
    expected_in = c_lhs + c_rhs
    if int(proj.weight.shape[1]) != expected_in:
        raise ValueError(
            f"Linear in_features mismatch: expected {expected_in}, got {int(proj.weight.shape[1])}"
        )
    w_lhs, w_rhs = torch.split(proj.weight, [c_lhs, c_rhs], dim=1)
    out = F.linear(lhs_packed, w_lhs, proj.bias)
    out = out + F.linear(rhs_packed, w_rhs, None)
    return out


@dataclass
class GNNConfig:
    """Configuration for GNN layers."""
    K: int = 3
    num_layers: int = 2
    norm_type: Literal['layer', 'group', 'instance', 'none'] = 'layer'
    dropout: float = 0.0
    activation: Literal['relu', 'silu', 'gelu', 'leaky_relu'] = 'silu'
    normalize: bool = False  # Whether to apply symmetric normalization in TAGConv
    use_strided_conv: bool = False
    # For StridedTAGConv power mode (ignored for regular TAGConv and boolean semiring mode):
    # - 'a_plus_i': A_base=A+I for gamma>1 (historical default)
    # - 'exact_walk': A_base=A (exact walk-length powers)
    strided_power_mode: Literal['a_plus_i', 'exact_walk'] = 'a_plus_i'
    max_gnn_stride: Optional[int] = None  # Cap on effective GNN gamma. None = no cap.
    use_pre_activation: bool = False
    use_temporal_mixer: bool = False
    temporal_mixer_schedule: Literal['per_layer', 'per_block', 'off'] = 'per_layer'
    temporal_kernel_size: int = 3
    temporal_causal: bool = False
    temporal_use_pointwise: bool = True
    temporal_gated: bool = False
    temporal_attention_enabled: bool = False
    temporal_attention_num_heads: int = 4
    temporal_attention_dropout: float = 0.0
    temporal_attention_max_timesteps: Optional[int] = None
    temporal_dilations: Optional[List[int]] = None
    pointwise_sparse_mode: Literal['off', 'auto', 'on'] = 'auto'
    pointwise_sparse_threshold: float = 0.75
    # Optional grouped override; when provided, values here are mapped into the
    # flat backward-compatible fields above.
    temporal_mixer: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        temporal_mixer_cfg = self.temporal_mixer
        if temporal_mixer_cfg is not None:
            enabled = _cfg_map_get(temporal_mixer_cfg, 'enabled', None)
            schedule = _cfg_map_get(temporal_mixer_cfg, 'schedule', None)
            kernel_size = _cfg_map_get(temporal_mixer_cfg, 'kernel_size', None)
            causal = _cfg_map_get(temporal_mixer_cfg, 'causal', None)
            use_pointwise = _cfg_map_get(temporal_mixer_cfg, 'use_pointwise', None)
            dilations = _cfg_map_get(temporal_mixer_cfg, 'dilations', None)

            if schedule is None and enabled is not None:
                schedule = 'per_layer' if bool(enabled) else 'off'

            if enabled is not None:
                self.use_temporal_mixer = bool(enabled)
                if not self.use_temporal_mixer:
                    self.temporal_mixer_schedule = 'off'
                else:
                    if schedule == 'off':
                        raise ValueError(
                            "In gnn_config.temporal_mixer, schedule='off' conflicts with enabled=true."
                        )
                    self.temporal_mixer_schedule = schedule or 'per_layer'
            elif schedule is not None:
                self.temporal_mixer_schedule = schedule
                self.use_temporal_mixer = schedule != 'off'

            if kernel_size is not None:
                self.temporal_kernel_size = int(kernel_size)
            if causal is not None:
                self.temporal_causal = bool(causal)
            if use_pointwise is not None:
                self.temporal_use_pointwise = bool(use_pointwise)
            gated = _cfg_map_get(temporal_mixer_cfg, 'gated', None)
            if gated is not None:
                self.temporal_gated = bool(gated)
            attention = _cfg_map_get(temporal_mixer_cfg, 'attention', None)
            if attention is not None:
                enabled = _cfg_map_get(attention, 'enabled', None)
                if enabled is not None:
                    self.temporal_attention_enabled = bool(enabled)
                num_heads = _cfg_map_get(attention, 'num_heads', None)
                if num_heads is not None:
                    self.temporal_attention_num_heads = int(num_heads)
                dropout_val = _cfg_map_get(attention, 'dropout', None)
                if dropout_val is not None:
                    self.temporal_attention_dropout = float(dropout_val)
                max_ts = _cfg_map_get(attention, 'max_timesteps', None)
                if max_ts is not None:
                    self.temporal_attention_max_timesteps = int(max_ts)
            if dilations is not None:
                self.temporal_dilations = list(dilations)

        if self.temporal_attention_num_heads < 1:
            raise ValueError(
                f"temporal_attention_num_heads must be >= 1, got {self.temporal_attention_num_heads}."
            )
        if not (0.0 <= self.temporal_attention_dropout <= 1.0):
            raise ValueError(
                f"temporal_attention_dropout must be in [0, 1], got {self.temporal_attention_dropout}."
            )
        if (self.temporal_attention_max_timesteps is not None
                and self.temporal_attention_max_timesteps < 1):
            raise ValueError(
                f"temporal_attention_max_timesteps must be >= 1, got {self.temporal_attention_max_timesteps}."
            )
        if self.pointwise_sparse_mode not in {'off', 'auto', 'on'}:
            raise ValueError(
                f"Unknown pointwise_sparse_mode: {self.pointwise_sparse_mode}. "
                "Expected one of: 'off', 'auto', 'on'."
            )
        if not (0.0 <= float(self.pointwise_sparse_threshold) <= 1.0):
            raise ValueError(
                f"pointwise_sparse_threshold must be in [0, 1], got {self.pointwise_sparse_threshold}."
            )
        if self.max_gnn_stride is not None and self.max_gnn_stride < 1:
            raise ValueError(
                f"max_gnn_stride must be >= 1 when set, got {self.max_gnn_stride}."
            )


@dataclass
class PoolingConfig:
    """Configuration for graph pooling.

    selector_kwargs: Optional[dict] can contain kwargs forwarded to the
    LearnableGraphPool when `selection_method='learned'` (e.g., selection_mode,
    temperature schedule). Score gain and entropy regularization parameters
    should be specified at PoolingConfig level, not in selector_kwargs.

    selector_hidden_norm_type controls hidden-state normalization inside the
    non-GNN selector head path (`use_pooling_gnn=false`).

    Depth-aware selector controls:
        entropy_reg_weight: Base entropy regularization weight (default 0.001).
            Applied to all selectors, then scaled per-level by entropy_reg_depth_scale.
        entropy_reg_depth_scale: Geometric multiplier for entropy_reg_weight at
            each successive selector level.  Level i receives
            ``entropy_reg_weight * entropy_reg_depth_scale ** i``.
            Default 1.0 (uniform across levels).
        score_gain_per_level: Optional explicit score_gain_init values per level.
            If provided, sets the per-level gain initialization.
            Example: [1.0, 3.0, 10.0] for 3 active selectors.
        learnable_score_gain: If True, the per-level score gain becomes a learnable
            parameter rather than a fixed buffer. Default False.
    """
    gamma: Union[int, List[int]] = 2
    pool_K: int = 1
    selection_method: Literal['stride', 'learned', 'random'] = 'stride'
    selector_kwargs: Optional[dict] = None
    temporal_score_agg: Literal['mean', 'last', 'ema', 'attention'] = 'mean'
    ema_alpha_init: float = 0.8
    selector_hidden_norm_type: Literal['layer', 'none'] = 'layer'
    entropy_reg_weight: float = 0.001
    entropy_reg_depth_scale: float = 1.0
    score_gain_per_level: Optional[List[float]] = None
    learnable_score_gain: bool = False
    selector_version: Literal['legacy', 'v3'] = 'legacy'
    min_retained_nodes: int = 1
    selector_temporal_reduce: Literal['mean', 'last', 'attn'] = 'mean'
    selector_use_cond_fusion: bool = True
    selector_cond_fusion_mode: Optional[Literal['cross_attention', 'film', 'add', 'none']] = None
    selector_cond_fusion_heads: int = 4
    selector_cond_fusion_dropout: float = 0.0
    selector_time_emb_dim: Optional[int] = None
    selector_exploration_noise: float = 0.0
    selector_exploration_noise_min: float = 0.0
    selector_exploration_noise_schedule: Literal['constant', 'linear', 'cosine'] = 'constant'
    selector_exploration_noise_anneal_steps: int = 0
    selector_exploration_noise_warmup_steps: int = 0
    selector_collect_heavy_diagnostics: bool = False
    selector_packed_score_mode: Literal['off', 'auto'] = 'off'
    selector_packed_score_threshold: float = 0.5
    # Optional when pool_K=0 (no neighborhood aggregation stage).
    # Accepts YAML null/None, plus explicit string 'null' for config consistency.
    neighborhood_pooling: Optional[Literal['max', 'avg', 'null']] = 'max'
    neighborhood_include_self: bool = True
    neighborhood_max_dense_gb: float = 8.0
    selector_skip_gamma1_levels: bool = False
    max_pool_stride: Optional[int] = None  # Cap on effective pool stride_input. None = no cap.

    def __post_init__(self) -> None:
        if self.max_pool_stride is not None and self.max_pool_stride < 1:
            raise ValueError(
                f"max_pool_stride must be >= 1 when set, got {self.max_pool_stride}."
            )


@dataclass
class UpsamplingConfig:
    """Configuration for decoder upsampling."""
    method: Literal['zero'] = 'zero'  # Will be extended to ['zero', 'nearest', 'graph_avg', 'graph_weighted', 'learned']


@dataclass
class EmbeddingConfig:
    """Configuration for embeddings."""
    time_embed_dim: int = 128
    num_timesteps: int = 1000
    cond_embed_dim: int = 64
    cond_channels: Optional[int] = None
    cond_temporal_hidden_channels: Optional[List[int]] = None
    cond_temporal_num_layers: int = 3
    cond_temporal_kernel_size: int = 3
    cond_temporal_pooling: Literal['max', 'mean', 'last', 'ema', 'attention'] = 'mean'
    cond_temporal_causal: bool = True
    cond_temporal_use_pointwise: bool = True
    cond_temporal_ema_alpha_init: float = 0.8
    cond_temporal_output_mode: Literal['static', 'time_varying'] = 'static'
    cond_temporal_time_varying_method: Literal['learned_projection', 'none'] = 'learned_projection'
    cond_temporal_projection_source_timesteps: Optional[int] = None
    cond_temporal_projection_target_timesteps: Optional[int] = None
    cond_temporal_dilations: Optional[List[int]] = None
    cond_temporal_gated: bool = False
    cond_temporal_attention_enabled: bool = False
    cond_temporal_attention_num_heads: int = 4
    cond_temporal_attention_dropout: float = 0.0
    cond_temporal_attention_max_timesteps: Optional[int] = None
    cond_fusion_mode: Literal['concat', 'cross_attention', 'film'] = 'concat'
    cond_projection_type: Literal['linear'] = 'linear'
    cond_projection_per_level: bool = True
    cond_cross_attn_heads: int = 4
    cond_cross_attn_dropout: float = 0.0
    cond_cross_attn_bias: bool = True
    cond_cross_attn_causal: bool = False
    # Optional grouped override; when provided, values here are mapped into the
    # flat backward-compatible fields above.
    cond: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        cond_cfg = self.cond
        if cond_cfg is None:
            return

        channels = _cfg_map_get(cond_cfg, 'channels', None)
        embed_dim = _cfg_map_get(cond_cfg, 'embed_dim', None)
        if channels is not None:
            self.cond_channels = channels
        if embed_dim is not None:
            self.cond_embed_dim = int(embed_dim)

        shared_encoder = _cfg_map_get(cond_cfg, 'shared_encoder', None)
        temporal = _cfg_map_get(shared_encoder, 'temporal', None)
        if temporal is not None:
            hidden_channels = _cfg_map_get(temporal, 'hidden_channels', None)
            num_layers = _cfg_map_get(temporal, 'num_layers', None)
            if hidden_channels is not None:
                self.cond_temporal_hidden_channels = list(hidden_channels)
            if num_layers is not None:
                self.cond_temporal_num_layers = int(num_layers)

            mixer = _cfg_map_get(temporal, 'mixer', None)
            if mixer is not None:
                kernel_size = _cfg_map_get(mixer, 'kernel_size', None)
                causal = _cfg_map_get(mixer, 'causal', None)
                use_pointwise = _cfg_map_get(mixer, 'use_pointwise', None)
                dilations = _cfg_map_get(mixer, 'dilations', None)
                gated = _cfg_map_get(mixer, 'gated', None)
                if kernel_size is not None:
                    self.cond_temporal_kernel_size = int(kernel_size)
                if causal is not None:
                    self.cond_temporal_causal = bool(causal)
                if use_pointwise is not None:
                    self.cond_temporal_use_pointwise = bool(use_pointwise)
                if dilations is not None:
                    self.cond_temporal_dilations = list(dilations)
                if gated is not None:
                    self.cond_temporal_gated = bool(gated)

                attention = _cfg_map_get(mixer, 'attention', None)
                if attention is not None:
                    attn_enabled = _cfg_map_get(attention, 'enabled', None)
                    attn_num_heads = _cfg_map_get(attention, 'num_heads', None)
                    attn_dropout = _cfg_map_get(attention, 'dropout', None)
                    attn_max_timesteps = _cfg_map_get(attention, 'max_timesteps', None)
                    if attn_enabled is not None:
                        self.cond_temporal_attention_enabled = bool(attn_enabled)
                    if attn_num_heads is not None:
                        self.cond_temporal_attention_num_heads = int(attn_num_heads)
                    if attn_dropout is not None:
                        self.cond_temporal_attention_dropout = float(attn_dropout)
                    if attn_max_timesteps is not None:
                        self.cond_temporal_attention_max_timesteps = int(attn_max_timesteps)

            output = _cfg_map_get(temporal, 'output', None)
            if output is not None:
                output_mode = _cfg_map_get(output, 'mode', None)
                if output_mode is not None:
                    self.cond_temporal_output_mode = output_mode

                static_cfg = _cfg_map_get(output, 'static', None)
                pooling_cfg = _cfg_map_get(static_cfg, 'pooling', None)
                if pooling_cfg is not None:
                    method = _cfg_map_get(pooling_cfg, 'method', None)
                    if method is not None:
                        self.cond_temporal_pooling = method
                    ema_cfg = _cfg_map_get(pooling_cfg, 'ema', None)
                    alpha_init = _cfg_map_get(ema_cfg, 'alpha_init', None)
                    if alpha_init is not None:
                        self.cond_temporal_ema_alpha_init = float(alpha_init)

                time_varying_cfg = _cfg_map_get(output, 'time_varying', None)
                if time_varying_cfg is not None:
                    method = _cfg_map_get(time_varying_cfg, 'method', None)
                    if method is not None:
                        self.cond_temporal_time_varying_method = method
                    learned_projection = _cfg_map_get(time_varying_cfg, 'learned_projection', None)
                    if learned_projection is not None:
                        source = _cfg_map_get(learned_projection, 'source_timesteps', None)
                        target = _cfg_map_get(learned_projection, 'target_timesteps', None)
                        if source is not None:
                            self.cond_temporal_projection_source_timesteps = int(source)
                        if target is not None:
                            self.cond_temporal_projection_target_timesteps = int(target)

        block_fusion = _cfg_map_get(cond_cfg, 'block_fusion', None)
        if block_fusion is not None:
            mode = _cfg_map_get(block_fusion, 'mode', None)
            if mode is not None:
                self.cond_fusion_mode = mode

            projection = _cfg_map_get(block_fusion, 'projection', None)
            if projection is not None:
                projection_type = _cfg_map_get(projection, 'type', None)
                projection_per_level = _cfg_map_get(projection, 'per_level', None)
                if projection_type is not None:
                    self.cond_projection_type = projection_type
                if projection_per_level is not None:
                    self.cond_projection_per_level = bool(projection_per_level)

            cross_attention = _cfg_map_get(block_fusion, 'cross_attention', None)
            if cross_attention is not None:
                heads = _cfg_map_get(cross_attention, 'heads', None)
                dropout = _cfg_map_get(cross_attention, 'dropout', None)
                bias = _cfg_map_get(cross_attention, 'bias', None)
                causal = _cfg_map_get(cross_attention, 'causal', None)
                if heads is not None:
                    self.cond_cross_attn_heads = int(heads)
                if dropout is not None:
                    self.cond_cross_attn_dropout = float(dropout)
                if bias is not None:
                    self.cond_cross_attn_bias = bool(bias)
                if causal is not None:
                    self.cond_cross_attn_causal = bool(causal)


@dataclass
class OutputHeadConfig:
    """Configuration for the UGNNDecoder's final output head.

    The default (mode='linear', zero_init=False) reproduces the original
    head — a single nn.Linear(final_channels, out_channels) with default
    Kaiming initialization — so existing configs and checkpoints are
    unaffected.

    mode='norm_act_linear' wraps the final linear with a DDPM/EDM-style
    LayerNorm + activation, applied over the channel dimension only
    (per-(batch, timestep, node) position; T and N are not mixed by the
    norm). zero_init=True additionally zero-initializes the final linear's
    weight and bias so the untrained head emits exactly zero — under the
    eps parameterization this makes the initial DDIM step approximately
    the identity (predicts no noise), which is the canonical convergence-
    friendly starting point used across DDPM, EDM, ADM, and DiT.
    """
    mode: Literal['linear', 'norm_act_linear'] = 'linear'
    norm_type: Literal['layer', 'none'] = 'layer'
    activation: Literal['silu', 'gelu'] = 'silu'
    zero_init: bool = False


def _build_output_head(
    final_channels: int,
    out_channels: int,
    cfg: OutputHeadConfig,
) -> nn.Module:
    """Construct the UGNNDecoder's final output head from an OutputHeadConfig.

    Returns either a bare nn.Linear (mode='linear', backwards-compatible)
    or nn.Sequential(LayerNorm, activation, Linear) (mode='norm_act_linear').
    """
    if cfg.activation == 'silu':
        act_module: nn.Module = nn.SiLU()
    elif cfg.activation == 'gelu':
        act_module = nn.GELU()
    else:
        raise ValueError(
            f"OutputHeadConfig.activation must be one of {{'silu', 'gelu'}}, "
            f"got {cfg.activation!r}."
        )

    if cfg.mode == 'linear':
        head: nn.Module = nn.Linear(final_channels, out_channels)
        final_linear = head
    elif cfg.mode == 'norm_act_linear':
        if cfg.norm_type == 'layer':
            norm_module: nn.Module = nn.LayerNorm(final_channels)
        elif cfg.norm_type == 'none':
            norm_module = nn.Identity()
        else:
            raise ValueError(
                f"OutputHeadConfig.norm_type must be one of {{'layer', 'none'}}, "
                f"got {cfg.norm_type!r}."
            )
        final_linear = nn.Linear(final_channels, out_channels)
        head = nn.Sequential(norm_module, act_module, final_linear)
    else:
        raise ValueError(
            f"OutputHeadConfig.mode must be one of {{'linear', 'norm_act_linear'}}, "
            f"got {cfg.mode!r}."
        )

    if cfg.zero_init:
        nn.init.zeros_(final_linear.weight)
        if final_linear.bias is not None:
            nn.init.zeros_(final_linear.bias)

    return head


@dataclass
class UGNNConfig:
    """Top-level UGNN architecture configuration."""
    in_channels: int
    out_channels: int
    base_channels: int = 64
    channel_multipliers: List[int] = field(default_factory=lambda: [1, 2, 4, 8])

    # Sub-configurations
    gnn_config: GNNConfig = field(default_factory=GNNConfig)
    pooling_config: PoolingConfig = field(default_factory=PoolingConfig)
    upsampling_config: UpsamplingConfig = field(default_factory=UpsamplingConfig)
    embedding_config: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    output_head_config: OutputHeadConfig = field(default_factory=OutputHeadConfig)

    # Decoder settings
    num_bottleneck_layers: int = 1  # GNN layers at coarsest resolution (0 to disable)
    skip_connection_mode: Literal['concat', 'add'] = 'concat'  # How to fuse skip connections
    compact_skip_cache: bool = False
    
    @property
    def num_levels(self) -> int:
        return len(self.channel_multipliers)
    
    @property
    def has_conditioning(self) -> bool:
        return self.embedding_config.cond_channels is not None


def _build_conditional_encoder(config: "UGNNConfig") -> ConditionalTemporalMixerEmbedding:
    """Build the conditional temporal encoder from EmbeddingConfig + GNN defaults."""
    emb_cfg = config.embedding_config
    if emb_cfg.cond_channels is None:
        raise ValueError("cond_channels must be set when building conditional encoder")
    ema_alpha_init = emb_cfg.cond_temporal_ema_alpha_init
    

    return ConditionalTemporalMixerEmbedding(
        in_channels=emb_cfg.cond_channels,
        embed_dim=emb_cfg.cond_embed_dim,
        hidden_channels=emb_cfg.cond_temporal_hidden_channels,
        kernel_size=emb_cfg.cond_temporal_kernel_size,
        num_layers=emb_cfg.cond_temporal_num_layers,
        pooling=emb_cfg.cond_temporal_pooling,
        activation=config.gnn_config.activation,
        dropout=config.gnn_config.dropout,
        norm_type=config.gnn_config.norm_type,
        use_pointwise=emb_cfg.cond_temporal_use_pointwise,
        causal=emb_cfg.cond_temporal_causal,
        ema_alpha_init=float(ema_alpha_init),
        output_mode=emb_cfg.cond_temporal_output_mode,
        time_varying_method=emb_cfg.cond_temporal_time_varying_method,
        projection_source_timesteps=emb_cfg.cond_temporal_projection_source_timesteps,
        projection_target_timesteps=emb_cfg.cond_temporal_projection_target_timesteps,
        dilations=emb_cfg.cond_temporal_dilations,
        gated=emb_cfg.cond_temporal_gated,
        attention_enabled=emb_cfg.cond_temporal_attention_enabled,
        attention_num_heads=emb_cfg.cond_temporal_attention_num_heads,
        attention_dropout=emb_cfg.cond_temporal_attention_dropout,
        attention_max_timesteps=emb_cfg.cond_temporal_attention_max_timesteps,
    )


class TemporalNodeCrossAttention(nn.Module):
    """
    Cross-attention over temporal tokens, independently per node.

    Inputs are shape (B, T, N, C). Attention is applied over T for each (B, N)
    stream and restricted to active nodes when a mask is provided.

    Notes:
    - Query and key/value may have different temporal lengths (T_q != T_kv).
      When causal=True with mismatched T, an explicit (T_q, T_kv) causal mask
      is built (position i in query attends to positions 0..i in key_value).
    - Active-node masking is handled via vectorized boolean gather/scatter
      across the full batch, avoiding per-sample Python loops.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        bias: bool = True,
        causal: bool = False,
    ):
        super().__init__()
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}")
        if channels % num_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by num_heads ({num_heads}) "
                "for temporal cross-attention."
            )

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.dropout = float(dropout)
        self.causal = bool(causal)

        self.q_proj = nn.Linear(channels, channels, bias=bias)
        self.k_proj = nn.Linear(channels, channels, bias=bias)
        self.v_proj = nn.Linear(channels, channels, bias=bias)
        self.out_proj = nn.Linear(channels, channels, bias=bias)

    def _to_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: (A, T, C) -> (A, H, T, D)
        a, t, c = x.shape
        return x.view(a, t, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()

    def _from_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: (A, H, T, D) -> (A, T, C)
        a, h, t, d = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(a, t, h * d)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (B, T_q, N, C)
            key_value: (B, T_kv, N, C) — T_kv may differ from T_q
            active_mask: Optional (B, N) boolean mask. Attention is evaluated only
                on active nodes, then output is masked again for safety.
        """
        if query.dim() != 4:
            raise ValueError(
                f"TemporalNodeCrossAttention expects rank-4 tensors (B,T,N,C), got {query.dim()}D"
            )
        if key_value.dim() != 4:
            raise ValueError(
                f"TemporalNodeCrossAttention expects rank-4 key_value (B,T,N,C), got {key_value.dim()}D"
            )
        if query.shape[0] != key_value.shape[0] or query.shape[2:] != key_value.shape[2:]:
            raise ValueError(
                "TemporalNodeCrossAttention expects query and key_value with matching "
                f"(B, N, C), got {tuple(query.shape)} vs {tuple(key_value.shape)}"
            )

        bsz, t_q, num_nodes, channels = query.shape
        t_kv = key_value.shape[1]
        if channels != self.channels:
            raise ValueError(
                f"Channel mismatch: got C={channels}, expected {self.channels} for cross-attention."
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

        attn_dropout = self.dropout if self.training else 0.0

        # Vectorized active-node processing: flatten (B, T, N, C) -> (B*N, T, C),
        # boolean-gather active node streams, run a single SDPA, scatter back.
        q_bn = query.permute(0, 2, 1, 3).reshape(bsz * num_nodes, t_q, channels)
        kv_bn = key_value.permute(0, 2, 1, 3).reshape(bsz * num_nodes, t_kv, channels)

        # (B, N) -> (B*N,)
        active_bn = active_mask.reshape(-1)
        num_active = int(active_bn.sum().item())

        if num_active > 0:
            # Gather only active node streams across the whole batch.
            q_active = q_bn[active_bn]    # (A, T_q, C)
            kv_active = kv_bn[active_bn]  # (A, T_kv, C)

            qh = self._to_heads(self.q_proj(q_active))   # (A, H, T_q, D)
            kh = self._to_heads(self.k_proj(kv_active))   # (A, H, T_kv, D)
            vh = self._to_heads(self.v_proj(kv_active))   # (A, H, T_kv, D)

            # SDPA's is_causal flag only works for square (T_q == T_kv) masks.
            # For mismatched T, build an explicit causal mask.
            use_sdpa_causal = self.causal and (t_q == t_kv)
            need_explicit_causal = self.causal and (t_q != t_kv)

            try:
                if need_explicit_causal:
                    # Causal: position i in query can attend to positions 0..i in key_value.
                    causal_mask = torch.ones(
                        (t_q, t_kv), device=qh.device, dtype=torch.bool,
                    ).triu(diagonal=1)  # True = masked
                    attn_mask = causal_mask.logical_not()  # True = attend
                    attn = F.scaled_dot_product_attention(
                        qh, kh, vh,
                        attn_mask=attn_mask.unsqueeze(0).unsqueeze(0),
                        dropout_p=attn_dropout,
                    )
                else:
                    attn = F.scaled_dot_product_attention(
                        qh, kh, vh,
                        attn_mask=None,
                        dropout_p=attn_dropout,
                        is_causal=use_sdpa_causal,
                    )
            except RuntimeError:
                # Fallback for kernels that fail on edge-case input layouts/size
                # combinations (e.g., some FlashAttention backends on older GPU stacks).
                scale = 1.0 / (self.head_dim ** 0.5)
                attn_scores = torch.matmul(qh, kh.transpose(-2, -1)) * scale
                if self.causal:
                    causal_mask = torch.ones(
                        (t_q, t_kv), device=qh.device, dtype=torch.bool,
                    ).triu(diagonal=1)
                    attn_scores = attn_scores.masked_fill(
                        causal_mask.unsqueeze(0).unsqueeze(0),
                        float("-inf"),
                    )
                attn = torch.softmax(attn_scores, dim=-1)
                if attn_dropout > 0.0:
                    attn = F.dropout(attn, p=attn_dropout, training=self.training)
                attn = torch.matmul(attn, vh)
            attn = self.out_proj(self._from_heads(attn))  # (A, T_q, C)

            # Scatter back to full (B*N, T_q, C); inactive rows remain zero.
            out_bn = torch.zeros_like(q_bn)
            out_bn[active_bn] = attn
        else:
            out_bn = torch.zeros_like(q_bn)

        # (B*N, T_q, C) -> (B, T_q, N, C)
        out = out_bn.view(bsz, num_nodes, t_q, channels).permute(0, 2, 1, 3).contiguous()

        # Final defensive mask to preserve inactive-node zeros in residual paths.
        out = out * active_mask.unsqueeze(1).unsqueeze(-1).to(dtype=out.dtype)
        return out


class TemporalNodeFiLM(nn.Module):
    """
    Lightweight FiLM generator for per-node time-wise conditioning.

    Produces (gamma, beta) from conditioning-aware features for each (B, T, N, C).
    Initialized so the first application starts as an identity: gamma=0, beta=0.
    """

    def __init__(
        self,
        channels: int,
        input_channels: Optional[int] = None,
        hidden_channels: Optional[int] = None,
    ) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError(f"channels must be > 0, got {channels}")
        if input_channels is None:
            input_channels = channels * 2
        if hidden_channels is None:
            hidden_channels = channels

        self.channels = channels
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(input_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, channels * 2),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 4:
            raise ValueError(f"TemporalNodeFiLM expects rank-4 input, got {x.dim()}D")
        if x.shape[-1] != self.input_channels:
            raise ValueError(
                f"TemporalNodeFiLM expects feature dimension {self.input_channels}, got {x.shape[-1]}"
            )
        out = self.mlp(x)
        gamma, beta = out.chunk(2, dim=-1)
        return gamma, beta


class EncoderBlock(nn.Module):
    """
    Encoder block for UGNN contracting path.
    
    Each encoder block:
    1. Embeds timesteps and conditional signals
    2. Fuses embeddings with input features
    3. Processes through GNN layers
    4. Applies StridedGraphMaxPool for downsampling
    
    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension (after GNN processing)
        stride_pre: Spacing between active (non-zero) nodes entering this block (default: 1)
            Determines GNN neighborhood scales to match signal sparsity pattern
            Level 0: stride_pre=1 (dense), Level 1: stride_pre=γ₀, Level 2: stride_pre=γ₀·γ₁
            GNN processes k*stride_pre hop neighborhoods for k=0,1,...,K
        stride_post: Spacing between active nodes leaving this block (default: None)
            If None, computed as stride_post = stride_pre * gamma
            Determines pooling aggregation neighborhoods to match output signal scale
            Pooling aggregates from k*stride_post hop neighborhoods for k=0,1,...,pool_K
        gnn_config: Configuration for GNN layers
        pooling_config: Configuration for pooling
        embedding_config: Configuration for embeddings
    
    Shape:
        - Input x: (B, T, N, F_in)
        - Timesteps: (B,)
        - Conditional: (B, T, N, F_cond) or None
        - Edge index: (2, E)
        - Active mask: Binary boolean mask (B, N) or None (dtype=torch.bool)
        - Output features: (B, T, N, F_out) with ~N/γ active nodes
        - Output active mask: (B, N)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride_pre: int,
        stride_post: Optional[int],
        gnn_config: GNNConfig,
        pooling_config: PoolingConfig,
        embedding_config: EmbeddingConfig,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gnn_config = gnn_config
        self.pooling_config = pooling_config
        self.embedding_config = embedding_config
        
        # Extract gamma from pooling config
        self.gamma = pooling_config.gamma if isinstance(pooling_config.gamma, int) else pooling_config.gamma[0]
        self.stride_pre = stride_pre
        self.stride_post = stride_post if stride_post is not None else stride_pre * self.gamma
        self.has_conditioning = embedding_config.cond_channels is not None
        self.cond_fusion_mode = embedding_config.cond_fusion_mode
        self.cond_projection_type = embedding_config.cond_projection_type
        self.cond_projection_per_level = embedding_config.cond_projection_per_level
        if self.cond_fusion_mode not in {'concat', 'cross_attention', 'film'}:
            raise ValueError(
                f"Unknown cond_fusion_mode: {self.cond_fusion_mode}. "
                "Expected one of: 'concat', 'cross_attention', 'film'."
            )
        
        # Time embedding projection
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding_config.time_embed_dim, in_channels),
        )
        
        # Conditional projection (if provided)
        # Projects pre-computed shared conditional embeddings to this block's feature space
        if self.has_conditioning:
            if self.cond_projection_type != 'linear':
                raise ValueError(
                    f"Unsupported conditional projection type: {self.cond_projection_type}. "
                    "Only 'linear' is currently implemented."
                )
            if not self.cond_projection_per_level:
                raise ValueError(
                    "Unsupported conditional projection config: projection.per_level=false. "
                    "Current implementation supports only per-level projection."
                )
            # Simple projection from shared embedding to block's feature space
            # cond_embed_dim -> in_channels
            self.cond_proj = nn.Linear(embedding_config.cond_embed_dim, in_channels)
            if self.cond_fusion_mode == 'cross_attention':
                self.cond_cross_attention = TemporalNodeCrossAttention(
                    channels=in_channels,
                    num_heads=embedding_config.cond_cross_attn_heads,
                    dropout=embedding_config.cond_cross_attn_dropout,
                    bias=embedding_config.cond_cross_attn_bias,
                    causal=embedding_config.cond_cross_attn_causal,
                )
            elif self.cond_fusion_mode == 'film':
                self.film_generator = TemporalNodeFiLM(
                    in_channels, input_channels=2 * in_channels,
                )
        
        # Feature fusion:
        # - concat mode: [input, time_emb, cond_emb] -> projection
        # - cross_attention mode: [input, time_emb] -> projection, then + cross-attn delta
        # - film mode: [input, time_emb] -> projection, then FiLM modulation
        fusion_channels = in_channels
        if self.has_conditioning and self.cond_fusion_mode == 'concat':
            fusion_channels += in_channels  # Add conditional features
        fusion_channels += in_channels  # Add time features
        
        self.fusion_proj = nn.Linear(fusion_channels, in_channels)
        
        # GNN for processing zero-padded features on fixed N-node graph
        # stride_pre determines neighborhood scales to match active node spacing
        # Example: stride_pre=2 means active nodes are ~2 hops apart,
        #          so we use 0,2,4,6,... hop neighborhoods to capture structure
        self.gnn = GNN(
            in_channels=in_channels,
            hidden_channels=out_channels,
            out_channels=out_channels,
            num_layers=gnn_config.num_layers,
            K=gnn_config.K,
            norm_type=gnn_config.norm_type,
            dropout=gnn_config.dropout,
            activation=gnn_config.activation,
            normalize=gnn_config.normalize,
            use_strided=gnn_config.use_strided_conv,
            gamma=(
                min(self.stride_pre, gnn_config.max_gnn_stride)
                if gnn_config.use_strided_conv and gnn_config.max_gnn_stride is not None
                else (self.stride_pre if gnn_config.use_strided_conv else 2)
            ),
            strided_power_mode=gnn_config.strided_power_mode,
            use_pre_activation=gnn_config.use_pre_activation,
            use_temporal_mixer=gnn_config.use_temporal_mixer,
            temporal_mixer_schedule=gnn_config.temporal_mixer_schedule,
            temporal_kernel_size=gnn_config.temporal_kernel_size,
            temporal_causal=gnn_config.temporal_causal,
            temporal_use_pointwise=gnn_config.temporal_use_pointwise,
            temporal_gated=gnn_config.temporal_gated,
            temporal_attention_enabled=gnn_config.temporal_attention_enabled,
            temporal_attention_num_heads=gnn_config.temporal_attention_num_heads,
            temporal_attention_dropout=gnn_config.temporal_attention_dropout,
            temporal_attention_max_timesteps=gnn_config.temporal_attention_max_timesteps,
            temporal_dilations=gnn_config.temporal_dilations,
            pointwise_sparse_mode=gnn_config.pointwise_sparse_mode,
            pointwise_sparse_threshold=gnn_config.pointwise_sparse_threshold,
        )
        self._last_packed_pre_gnn_fusion_used: bool = False

        # Strided max pooling for downsampling (selecting active node subset)
        # Operates on zero-padded signal on fixed N-node graph
        # 
        # Uses local gamma for BOTH node selection and multi-scale aggregation:
        #   1. Selects every gamma-th ACTIVE node from current active set (local operation)
        #   2. Aggregates from 0, gamma, 2*gamma, ..., K*gamma hop neighborhoods
        # 
        # Example at Level 1 (stride_pre=2, gamma=2, stride_post=4, pool_K=2):
        #   - Input: signal with stride=2 spacing (~N/2 active nodes)
        #   - Select: every 2nd active node → ~N/4 output active nodes
        #   - Aggregate: 0, 2, 4-hop on original graph (gamma * k)
        #   - Output: signal with stride=4 spacing
        # 
        # Note: Selection operates on active mask (local), so aggregation uses local gamma.
        # This is correct for stride-based selection which is relative to the active set.
        # 
        # If selection_method='learned', the internal StridedPoolingGNN will
        # use stride_pre for its convolutions, properly respecting input signal sparsity.
        selector_kwargs = dict(pooling_config.selector_kwargs or {})
        selector_kwargs.setdefault('temporal_score_agg', pooling_config.temporal_score_agg)
        selector_kwargs.setdefault('ema_alpha_init', pooling_config.ema_alpha_init)
        selector_kwargs.setdefault('hidden_norm_type', pooling_config.selector_hidden_norm_type)
        selector_kwargs.setdefault('selector_version', pooling_config.selector_version)
        selector_kwargs.setdefault('min_retained_nodes', pooling_config.min_retained_nodes)
        selector_kwargs.setdefault('temporal_reduce', pooling_config.selector_temporal_reduce)
        selector_kwargs.setdefault('cond_fusion', pooling_config.selector_use_cond_fusion)
        if pooling_config.selector_cond_fusion_mode is not None:
            selector_kwargs.setdefault('cond_fusion_mode', pooling_config.selector_cond_fusion_mode)
        selector_kwargs.setdefault('cond_fusion_heads', pooling_config.selector_cond_fusion_heads)
        selector_kwargs.setdefault('cond_fusion_dropout', pooling_config.selector_cond_fusion_dropout)
        if self.has_conditioning:
            # Encoder forwards cond tokens projected to this block's in_channels.
            # Learned selector may run at out_channels, so pass the explicit
            # incoming cond width to allow NodeSelector to project in->out.
            selector_kwargs.setdefault('cond_dim', in_channels)
        if pooling_config.selector_time_emb_dim is not None:
            selector_kwargs.setdefault('time_emb_dim', pooling_config.selector_time_emb_dim)
        selector_kwargs.setdefault('exploration_noise', pooling_config.selector_exploration_noise)
        selector_kwargs.setdefault('exploration_noise_min', pooling_config.selector_exploration_noise_min)
        selector_kwargs.setdefault('exploration_noise_schedule', pooling_config.selector_exploration_noise_schedule)
        selector_kwargs.setdefault('exploration_noise_anneal_steps', pooling_config.selector_exploration_noise_anneal_steps)
        selector_kwargs.setdefault('exploration_noise_warmup_steps', pooling_config.selector_exploration_noise_warmup_steps)
        selector_kwargs.setdefault('collect_heavy_diagnostics', pooling_config.selector_collect_heavy_diagnostics)
        selector_kwargs.setdefault('packed_score_mode', pooling_config.selector_packed_score_mode)
        selector_kwargs.setdefault('packed_score_threshold', pooling_config.selector_packed_score_threshold)

        # Skip selector creation for gamma=1 levels when the opt-in knob is
        # enabled.  _select_nodes() already short-circuits at gamma=1 (the
        # selector is never called), so this only saves memory / optimizer state.
        skip_selector = (
            pooling_config.selector_skip_gamma1_levels
            and pooling_config.selector_version == 'v3'
            and self.gamma == 1
        )

        self.pool = StridedGraphMaxPool(
            gamma=self.gamma,            # Local downsampling ratio
            K=pooling_config.pool_K,
            selection_method=pooling_config.selection_method,
            in_channels=out_channels if (
                pooling_config.selection_method == 'learned' and not skip_selector
            ) else None,
            stride_input=(
                min(self.stride_pre, pooling_config.max_pool_stride)
                if pooling_config.max_pool_stride is not None
                else self.stride_pre
            ),  # Input signal spacing (for learned selector)
            selector_kwargs=selector_kwargs,
            skip_selector=skip_selector,
            neighborhood_pooling=pooling_config.neighborhood_pooling,
            neighborhood_include_self=pooling_config.neighborhood_include_self,
            max_dense_gb=pooling_config.neighborhood_max_dense_gb,
        )
    
    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[torch.Tensor, "batch timesteps nodes features"],
        timesteps: Int[torch.Tensor, "batch"],
        edge_index: Int[torch.Tensor, "2 edges"],
        edge_weight: Optional[Float[torch.Tensor, "edges"]] = None,
        active_mask: Optional[Bool[torch.Tensor, "batch nodes"]] = None,
        cond_emb: Optional[torch.Tensor] = None,
        time_emb: Optional[Float[torch.Tensor, "batch time_embed_dim"]] = None,
        return_intermediates: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[dict | None]]:
        """
        Forward pass through encoder block.
        
        Args:
            x: Node features (B, T, N, F_in)
            timesteps: Diffusion timesteps (B,)
            edge_index: Edge indices (2, E)
            edge_weight: Optional edge weights (E,)
            active_mask: Binary mask of active nodes (B, N)
            cond_emb: Pre-computed conditional embeddings. Supported shapes:
                - (B, N, cond_embed_dim) static conditioning
                - (B, T, N, cond_embed_dim) time-varying conditioning
            time_emb: Precomputed time embeddings (B, time_embed_dim) or None
            return_intermediates: If True, return dict of intermediate tensors
        
        Returns:
            x_pooled: Processed and pooled features (B, T, N, F_out)
            new_active_mask: Updated active mask after pooling (B, N)
            x_before_pool: Features before pooling for skip connection (B, T, N, F_out)
            intermediates: Dict of intermediate tensors (if return_intermediates=True)
                            None (if return_intermediates=False)  
        """
        B, T, N, F = x.shape
        
        # Initialize intermediates dict
        intermediates = {} if return_intermediates else None
        if return_intermediates:
            intermediates['input'] = x.clone()
            intermediates['active_mask_in'] = active_mask.clone() if active_mask is not None else None
        
        # Use precomputed time embedding if provided
        if time_emb is None:
            raise ValueError("time_emb must be provided")
        
        # Project time embedding to feature space; keep compact (B, C) form.
        time_token = self.time_proj(time_emb)  # (B, in_channels)
        time_feats_expanded = None
        if return_intermediates or (self.has_conditioning and self.cond_fusion_mode == 'film'):
            time_feats_expanded = time_token.unsqueeze(1).unsqueeze(2).expand(B, T, N, -1)
        if return_intermediates and time_feats_expanded is not None:
            intermediates['time_emb'] = time_feats_expanded.clone()

        packed_pre_gnn_eligible = (
            (not return_intermediates)
            and active_mask is not None
            and self.gnn.should_use_packed_pointwise(x, active_mask)
            and (not self.has_conditioning or self.cond_fusion_mode == 'concat')
        )
        active_indices = (
            _active_indices_from_mask(active_mask)
            if packed_pre_gnn_eligible and active_mask is not None
            else None
        )
        self._last_packed_pre_gnn_fusion_used = False

        x_fused = None
        cond_feats = None

        # Process conditional embeddings if provided
        if self.has_conditioning:
            if cond_emb is None:
                raise ValueError("Conditional embeddings must be provided when has_conditioning=True")

            # Project conditional embeddings to this block's feature space.
            # This keeps conditioning width aligned with this block's channels
            # before pooling/selection. The current v3 NodeSelector path relies
            # on this matched-width contract (no explicit cond_dim forwarded).
            cond_feats = self.cond_proj(cond_emb)
            if cond_feats.dim() == 3:
                # Static conditioning: (B, N, in_channels) -> broadcast to (B, T, N, in_channels)
                if active_mask is not None:
                    mask_2d = active_mask.unsqueeze(-1).float()  # (B, N, 1)
                    cond_feats = cond_feats * mask_2d
                cond_feats = cond_feats.unsqueeze(1).expand(B, T, N, -1)
            elif cond_feats.dim() == 4:
                # Time-varying conditioning: T must match the main signal horizon
                # for concat/film fusion (they operate element-wise along T).
                # Cross-attention handles T_cond != T natively.
                if self.cond_fusion_mode != 'cross_attention' and cond_feats.size(1) != T:
                    raise ValueError(
                        f"Time-varying conditional embeddings must match T={T} for "
                        f"{self.cond_fusion_mode} fusion, got {cond_feats.size(1)}"
                    )
                if active_mask is not None:
                    mask_4d = active_mask.unsqueeze(1).unsqueeze(-1).float()
                    cond_feats = cond_feats * mask_4d
            else:
                raise ValueError(
                    "Conditional embeddings must be rank-3 (B, N, C) or rank-4 (B, T, N, C), "
                    f"got shape {tuple(cond_feats.shape)}"
                )

            if return_intermediates:
                intermediates['cond_emb'] = cond_feats.clone()
        else:
            if return_intermediates:
                intermediates['cond_emb'] = None

        if packed_pre_gnn_eligible and active_indices is not None:
            x_packed = _pack_active_nodes_4d_from_indices(x, active_indices)
            if self.has_conditioning:
                cond_packed = _pack_active_nodes_4d_from_indices(cond_feats, active_indices)
                x_fused_packed = _apply_fusion_proj_packed_without_concat(
                    x_packed=x_packed,
                    active_indices=active_indices,
                    num_nodes=N,
                    time_token=time_token,
                    fusion_proj=self.fusion_proj,
                    cond_packed=cond_packed,
                )
            else:
                x_fused_packed = _apply_fusion_proj_packed_without_concat(
                    x_packed=x_packed,
                    active_indices=active_indices,
                    num_nodes=N,
                    time_token=time_token,
                    fusion_proj=self.fusion_proj,
                )
            x_fused = _unpack_active_nodes_4d(
                x_fused_packed,
                active_indices,
                batch_size=B,
                timesteps=T,
                num_nodes=N,
                channels=self.in_channels,
            )
            self._last_packed_pre_gnn_fusion_used = True
        else:
            if self.has_conditioning:
                if self.cond_fusion_mode == 'cross_attention' and cond_feats.dim() == 4:
                    x_base = _apply_fusion_proj_without_concat(
                        x=x,
                        time_token=time_token,
                        fusion_proj=self.fusion_proj,
                    )
                    cond_delta = self.cond_cross_attention(
                        query=x_base,
                        key_value=cond_feats,
                        active_mask=active_mask,
                    )
                    x_fused = x_base + cond_delta
                elif self.cond_fusion_mode == 'film':
                    x_base = _apply_fusion_proj_without_concat(
                        x=x,
                        time_token=time_token,
                        fusion_proj=self.fusion_proj,
                    )
                    if time_feats_expanded is None:
                        time_feats_expanded = time_token.unsqueeze(1).unsqueeze(2).expand(B, T, N, -1)
                    film_input = torch.cat([time_feats_expanded, cond_feats], dim=-1)  # (B, T, N, 2C)
                    gamma, beta = self.film_generator(film_input)  # (B, T, N, C) each
                    x_fused = x_base * (1.0 + gamma) + beta
                else:
                    # Fallback path: static conditioning (rank-3) in cross-attention mode,
                    # or explicit concat mode.
                    if self.cond_fusion_mode == 'concat':
                        x_fused = _apply_fusion_proj_without_concat(
                            x=x,
                            time_token=time_token,
                            cond_feats=cond_feats,
                            fusion_proj=self.fusion_proj,
                        )
                    else:
                        x_fused = _apply_fusion_proj_without_concat(
                            x=x,
                            time_token=time_token,
                            fusion_proj=self.fusion_proj,
                        ) + cond_feats
            else:
                x_fused = _apply_fusion_proj_without_concat(
                    x=x,
                    time_token=time_token,
                    fusion_proj=self.fusion_proj,
                )
        
        # Mask inactive nodes before GNN
        if active_mask is not None:
            mask_4d = active_mask.unsqueeze(1).unsqueeze(-1).float()
            x_fused = x_fused * mask_4d
        
        if return_intermediates:
            intermediates['fused'] = x_fused.clone()
        
        # Process through GNN
        x_processed = self.gnn(
            x_fused,
            edge_index,
            edge_weight,
            active_mask=active_mask,
        )  # (B, T, N, out_channels)

        # GNN blocks do not enforce active-node masking internally.
        # Re-apply mask so inactive nodes remain zero in skip features and pooling input.
        if active_mask is not None:
            mask_4d = active_mask.unsqueeze(1).unsqueeze(-1).float()
            x_processed = x_processed * mask_4d

        if return_intermediates:
            intermediates['gnn_output'] = x_processed.clone()
        
        # Store before pooling for skip connection
        x_before_pool = x_processed
        
        # Apply pooling.
        # gamma=1 with pool_K=0 is a strict no-op (no neighborhood aggregation and
        # no downsampling), so we can bypass the pooling module entirely.
        # For gamma=1 with pool_K>0, still run StridedGraphMaxPool so neighborhood
        # feature aggregation can happen at the current cumulative stride.
        strict_identity_no_pool = (
            self.gamma == 1
            and self.pool.K == 0
        )
        pool_skipped = strict_identity_no_pool

        if pool_skipped:
            x_pooled = x_processed
            if active_mask is None:
                new_active_mask = torch.ones(B, N, dtype=torch.bool, device=x.device)
            else:
                new_active_mask = active_mask
            selected_indices = None
            selection_scores = None
        elif self.pool.selection_method == 'learned' and return_intermediates:
            # For learned selection, we want to capture the selection scores
            x_pooled, new_active_mask, selected_indices, selection_scores = self.pool(
                x_processed,
                edge_index,
                edge_weight,
                active_mask,
                cond=cond_feats if self.has_conditioning else None,
                time_emb=time_emb,
            )
        else:
            # For all other cases: learned without intermediates, stride, random
            # Return value varies: selected_indices for learned, same for stride/random
            x_pooled, new_active_mask, selected_indices, selection_scores = self.pool(
                x_processed,
                edge_index,
                edge_weight,
                active_mask,
                cond=cond_feats if self.has_conditioning else None,
                time_emb=time_emb,
            )
        
        if return_intermediates:
            intermediates['selected_indices'] = selected_indices
            intermediates['selection_scores'] = selection_scores  # Will be (B, N) for learned, None otherwise
            intermediates['before_pool'] = x_before_pool.clone()
            intermediates['after_pool'] = x_pooled.clone()
            intermediates['active_mask_out'] = new_active_mask.clone() if new_active_mask is not None else None
            intermediates['pool_skipped'] = pool_skipped
        
        # NOTE: Masking inactive nodes after pooling is redundant because:
        # StridedGraphMaxPool already ensures inactive nodes have zero features:
        # - Sets inactive inputs to -inf before max-pooling
        # - Converts -inf back to zero after max-pooling
        # - All selection methods explicitly zero non-selected nodes
        # Keeping this code commented for defensive programming reference:
        # if new_active_mask is not None:
        #     mask_4d_pooled = new_active_mask.unsqueeze(1).unsqueeze(-1).float()  # (B, 1, N, 1)
        #     x_pooled = x_pooled * mask_4d_pooled
        
        return x_pooled, new_active_mask, x_before_pool, intermediates
    

    def extra_repr(self) -> str:
        return (f"in_channels={self.in_channels}, "
                f"out_channels={self.out_channels}, "
                f"gamma={self.gamma}, "
                f"has_cond={self.has_conditioning}, "
                f"cond_fusion_mode={self.cond_fusion_mode}")


class DecoderBlock(nn.Module):
    """
    Decoder block for UGNN expanding path.
    
    Each decoder block:
    1. Upsamples input by restoring previously active nodes
    2. Fuses with skip connection from encoder
    3. Embeds timesteps and conditional signals
    4. Processes through GNN layers
    
    Args:
        in_channels: Input feature dimension (from previous decoder level)
        skip_channels: Skip connection feature dimension (from encoder)
        out_channels: Output feature dimension (after GNN processing)
        stride_pre: Spacing between active nodes entering this block (after unpooling)
        gnn_config: Configuration for GNN layers
        upsampling_config: Configuration for upsampling
        embedding_config: Configuration for embeddings
        skip_connection_mode: How to fuse skip connections ('concat' or 'add')
    
    Shape:
        - Input x: (B, T, N, F_in) with fewer active nodes
        - Skip features: (B, T, N, F_skip) with more active nodes
        - Active mask target: (B, N) - which nodes should be active after unpooling
        - Output: (B, T, N, F_out) with more active nodes
    """
    
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        stride_pre: int,
        gnn_config: GNNConfig,
        upsampling_config: UpsamplingConfig,
        embedding_config: EmbeddingConfig,
        skip_connection_mode: Literal['concat', 'add'] = 'concat',
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.out_channels = out_channels
        self.stride_pre = stride_pre
        self.gnn_config = gnn_config
        self.upsampling_config = upsampling_config
        self.embedding_config = embedding_config
        self.skip_connection_mode = skip_connection_mode
        self.has_conditioning = embedding_config.cond_channels is not None
        self.cond_fusion_mode = embedding_config.cond_fusion_mode
        self.cond_projection_type = embedding_config.cond_projection_type
        self.cond_projection_per_level = embedding_config.cond_projection_per_level
        if self.cond_fusion_mode not in {'concat', 'cross_attention', 'film'}:
            raise ValueError(
                f"Unknown cond_fusion_mode: {self.cond_fusion_mode}. "
                "Expected one of: 'concat', 'cross_attention', 'film'."
            )
        
        # Skip connection fusion
        if skip_connection_mode == 'concat':
            fused_channels = in_channels + skip_channels
            self.skip_proj = nn.Linear(fused_channels, in_channels)
        elif skip_connection_mode == 'add':
            # Require matching dimensions for addition
            if in_channels != skip_channels:
                self.skip_proj = nn.Linear(skip_channels, in_channels)
            else:
                self.skip_proj = nn.Identity()
        
        # Time embedding projection
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding_config.time_embed_dim, in_channels),
        )
        
        # Conditional projection (if provided)
        if self.has_conditioning:
            if self.cond_projection_type != 'linear':
                raise ValueError(
                    f"Unsupported conditional projection type: {self.cond_projection_type}. "
                    "Only 'linear' is currently implemented."
                )
            if not self.cond_projection_per_level:
                raise ValueError(
                    "Unsupported conditional projection config: projection.per_level=false. "
                    "Current implementation supports only per-level projection."
                )
            self.cond_proj = nn.Linear(embedding_config.cond_embed_dim, in_channels)
            if self.cond_fusion_mode == 'cross_attention':
                self.cond_cross_attention = TemporalNodeCrossAttention(
                    channels=in_channels,
                    num_heads=embedding_config.cond_cross_attn_heads,
                    dropout=embedding_config.cond_cross_attn_dropout,
                    bias=embedding_config.cond_cross_attn_bias,
                    causal=embedding_config.cond_cross_attn_causal,
                )
            elif self.cond_fusion_mode == 'film':
                self.film_generator = TemporalNodeFiLM(
                    in_channels, input_channels=2 * in_channels,
                )
        
        # Feature fusion:
        # - concat mode: [input, time_emb, cond_emb] -> projection
        # - cross_attention mode: [input, time_emb] -> projection, then + cross-attn delta
        # - film mode: [input, time_emb] -> projection, then FiLM modulation
        fusion_channels = in_channels + in_channels  # input + time
        if self.has_conditioning and self.cond_fusion_mode == 'concat':
            fusion_channels += in_channels  # + conditional
        
        self.fusion_proj = nn.Linear(fusion_channels, in_channels)
        
        # GNN for processing
        self.gnn = GNN(
            in_channels=in_channels,
            hidden_channels=out_channels,
            out_channels=out_channels,
            num_layers=gnn_config.num_layers,
            K=gnn_config.K,
            norm_type=gnn_config.norm_type,
            dropout=gnn_config.dropout,
            activation=gnn_config.activation,
            normalize=gnn_config.normalize,
            use_strided=gnn_config.use_strided_conv,
            gamma=(
                min(self.stride_pre, gnn_config.max_gnn_stride)
                if gnn_config.use_strided_conv and gnn_config.max_gnn_stride is not None
                else (self.stride_pre if gnn_config.use_strided_conv else 2)
            ),
            strided_power_mode=gnn_config.strided_power_mode,
            use_pre_activation=gnn_config.use_pre_activation,
            use_temporal_mixer=gnn_config.use_temporal_mixer,
            temporal_mixer_schedule=gnn_config.temporal_mixer_schedule,
            temporal_kernel_size=gnn_config.temporal_kernel_size,
            temporal_causal=gnn_config.temporal_causal,
            temporal_use_pointwise=gnn_config.temporal_use_pointwise,
            temporal_gated=gnn_config.temporal_gated,
            temporal_attention_enabled=gnn_config.temporal_attention_enabled,
            temporal_attention_num_heads=gnn_config.temporal_attention_num_heads,
            temporal_attention_dropout=gnn_config.temporal_attention_dropout,
            temporal_attention_max_timesteps=gnn_config.temporal_attention_max_timesteps,
            temporal_dilations=gnn_config.temporal_dilations,
            pointwise_sparse_mode=gnn_config.pointwise_sparse_mode,
            pointwise_sparse_threshold=gnn_config.pointwise_sparse_threshold,
        )
        self._last_packed_pre_gnn_fusion_used: bool = False
        self._last_packed_skip_fusion_used: bool = False
    
    def _upsample_zero(self, x: torch.Tensor, active_mask_target: torch.Tensor, active_mask_input: torch.Tensor) -> torch.Tensor:
        """
        Zero-filling upsampling: newly active nodes start with zeros.
        
        Args:
            x: Input features (B, T, N, C) - sparse with current active nodes
            active_mask_target: (B, N) - target active nodes after upsampling
            active_mask_input: (B, N) - current active nodes before upsampling
        
        Returns:
            Upsampled features (B, T, N, C) with newly active nodes set to zero
        """
        # Newly active nodes are those in target but not in input
        # newly_active = target & ~input
        newly_active = active_mask_target & (~active_mask_input)  # (B, N)
        
        # Zero out the newly active positions (they'll get values from skip connection)
        if newly_active.any():
            mask_4d = newly_active.unsqueeze(1).unsqueeze(-1).float()  # (B, 1, N, 1)
            x = x * (1 - mask_4d)  # Zero out newly active positions
        
        return x
    
    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[torch.Tensor, "batch timesteps nodes channels"],
        skip_features: Float[torch.Tensor, "batch timesteps nodes skip_channels"],
        timesteps: Int[torch.Tensor, "batch"],
        edge_index: Int[torch.Tensor, "2 edges"],
        edge_weight: Optional[Float[torch.Tensor, "edges"]] = None,
        active_mask_target: Optional[Bool[torch.Tensor, "batch nodes"]] = None,
        active_mask_input: Optional[Bool[torch.Tensor, "batch nodes"]] = None,
        cond_emb: Optional[torch.Tensor] = None,
        time_emb: Optional[Float[torch.Tensor, "batch time_embed_dim"]] = None,
        return_intermediates: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict | None]]:
        """
        Forward pass through decoder block.
        
        Args:
            x: Input features from previous decoder level (B, T, N, F_in)
            skip_features: Skip connection from encoder (B, T, N, F_skip)
            timesteps: Diffusion timesteps (B,)
            edge_index: Edge indices (2, E)
            edge_weight: Optional edge weights (E,)
            active_mask_target: Target active mask after unpooling (B, N) - Binary boolean mask (dtype=torch.bool)
            active_mask_input: Input active mask before unpooling (B, N) - Binary boolean mask (dtype=torch.bool)
            cond_emb: Pre-computed conditional embeddings. Supported shapes:
                - (B, N, cond_embed_dim) static conditioning
                - (B, T, N, cond_embed_dim) time-varying conditioning
            time_emb: Precomputed time embeddings (B, time_embed_dim) or None
            return_intermediates: If True, return dict of intermediate tensors
        
        Returns:
            Processed features (B, T, N, F_out)
            intermediates: Dict of intermediate tensors (if return_intermediates=True)
                            None if return_intermediates=False
        """
        B, T, N, F_in = x.shape
        
        # Initialize intermediates dict
        intermediates = {} if return_intermediates else None
        if return_intermediates:
            intermediates['input'] = x
            # Use the provided input mask (tracks actual active nodes before upsampling)
            intermediates['active_mask_in'] = active_mask_input if active_mask_input is not None else active_mask_target
            intermediates['skip_connection'] = skip_features
        
        # Upsample: restore previously active nodes
        if self.upsampling_config.method == 'zero':
            x_upsampled = self._upsample_zero(x, active_mask_target, active_mask_input)
        else:
            raise NotImplementedError(f"Upsampling method '{self.upsampling_config.method}' not implemented")
        
        if return_intermediates:
            intermediates['after_upsample'] = x_upsampled

        packed_pre_gnn_eligible = (
            (not return_intermediates)
            and active_mask_target is not None
            and self.gnn.should_use_packed_pointwise(x_upsampled, active_mask_target)
            and (not self.has_conditioning or self.cond_fusion_mode == 'concat')
        )
        active_indices = (
            _active_indices_from_mask(active_mask_target)
            if packed_pre_gnn_eligible and active_mask_target is not None
            else None
        )
        self._last_packed_pre_gnn_fusion_used = False
        self._last_packed_skip_fusion_used = False
        x_fused = None
        x_fused_packed = None

        # Fuse with skip connection
        if packed_pre_gnn_eligible and active_indices is not None:
            x_upsampled_packed = _pack_active_nodes_4d_from_indices(x_upsampled, active_indices)
            skip_packed = _pack_active_nodes_4d_from_indices(skip_features, active_indices)
            if self.skip_connection_mode == 'concat':
                x_fused_packed = _apply_linear_concat_packed(
                    lhs_packed=x_upsampled_packed,
                    rhs_packed=skip_packed,
                    proj=self.skip_proj,
                )
            elif self.skip_connection_mode == 'add':
                skip_proj = self.skip_proj(skip_packed)
                x_fused_packed = x_upsampled_packed + skip_proj
            else:
                raise ValueError(f"Unknown skip_connection_mode: {self.skip_connection_mode}")
            self._last_packed_skip_fusion_used = True
        else:
            if self.skip_connection_mode == 'concat':
                x_fused = torch.cat([x_upsampled, skip_features], dim=-1)  # (B, T, N, in+skip)
                x_fused = self.skip_proj(x_fused)  # (B, T, N, in_channels)
            elif self.skip_connection_mode == 'add':
                skip_proj = self.skip_proj(skip_features)  # (B, T, N, in_channels)
                x_fused = x_upsampled + skip_proj
            else:
                raise ValueError(f"Unknown skip_connection_mode: {self.skip_connection_mode}")
            
            # Mask inactive nodes after skip fusion
            if active_mask_target is not None:
                mask_4d = active_mask_target.unsqueeze(1).unsqueeze(-1).float()
                x_fused = x_fused * mask_4d
        
        # Use precomputed time embedding
        if time_emb is None:
            raise ValueError("time_emb must be provided")
        
        # Project time embedding to feature space; keep compact (B, C) form.
        time_token = self.time_proj(time_emb)  # (B, in_channels)
        time_feats_expanded = None
        if return_intermediates or (self.has_conditioning and self.cond_fusion_mode == 'film'):
            time_feats_expanded = time_token.unsqueeze(1).unsqueeze(2).expand(B, T, N, -1)
        if return_intermediates and time_feats_expanded is not None:
            intermediates['time_emb'] = time_feats_expanded

        # Process conditional embeddings if provided
        if self.has_conditioning:
            if cond_emb is None:
                raise ValueError("Conditional embeddings must be provided when has_conditioning=True")

            # Project conditional embeddings.
            cond_feats = self.cond_proj(cond_emb)
            if cond_feats.dim() == 3:
                if active_mask_target is not None:
                    mask_2d = active_mask_target.unsqueeze(-1).float()
                    cond_feats = cond_feats * mask_2d
                cond_feats = cond_feats.unsqueeze(1).expand(B, T, N, -1)
            elif cond_feats.dim() == 4:
                if self.cond_fusion_mode != 'cross_attention' and cond_feats.size(1) != T:
                    raise ValueError(
                        f"Time-varying conditional embeddings must match T={T} for "
                        f"{self.cond_fusion_mode} fusion, got {cond_feats.size(1)}"
                    )
                if active_mask_target is not None:
                    mask_4d = active_mask_target.unsqueeze(1).unsqueeze(-1).float()
                    cond_feats = cond_feats * mask_4d
            else:
                raise ValueError(
                    "Conditional embeddings must be rank-3 (B, N, C) or rank-4 (B, T, N, C), "
                    f"got shape {tuple(cond_feats.shape)}"
                )
        else:
            cond_feats = None
        
        if packed_pre_gnn_eligible and active_indices is not None and x_fused_packed is not None:
            if self.has_conditioning:
                cond_packed = _pack_active_nodes_4d_from_indices(cond_feats, active_indices)
                x_final_packed = _apply_fusion_proj_packed_without_concat(
                    x_packed=x_fused_packed,
                    active_indices=active_indices,
                    num_nodes=N,
                    time_token=time_token,
                    fusion_proj=self.fusion_proj,
                    cond_packed=cond_packed,
                )
            else:
                x_final_packed = _apply_fusion_proj_packed_without_concat(
                    x_packed=x_fused_packed,
                    active_indices=active_indices,
                    num_nodes=N,
                    time_token=time_token,
                    fusion_proj=self.fusion_proj,
                )
            x_final = _unpack_active_nodes_4d(
                x_final_packed,
                active_indices,
                batch_size=B,
                timesteps=T,
                num_nodes=N,
                channels=self.in_channels,
            )
            self._last_packed_pre_gnn_fusion_used = True
        else:
            if self.has_conditioning:
                if self.cond_fusion_mode == 'cross_attention' and cond_feats.dim() == 4:
                    x_base = _apply_fusion_proj_without_concat(
                        x=x_fused,
                        time_token=time_token,
                        fusion_proj=self.fusion_proj,
                    )
                    cond_delta = self.cond_cross_attention(
                        query=x_base,
                        key_value=cond_feats,
                        active_mask=active_mask_target,
                    )
                    x_final = x_base + cond_delta
                elif self.cond_fusion_mode == 'film':
                    x_base = _apply_fusion_proj_without_concat(
                        x=x_fused,
                        time_token=time_token,
                        fusion_proj=self.fusion_proj,
                    )
                    if time_feats_expanded is None:
                        time_feats_expanded = time_token.unsqueeze(1).unsqueeze(2).expand(B, T, N, -1)
                    film_input = torch.cat([time_feats_expanded, cond_feats], dim=-1)  # (B, T, N, 2C)
                    gamma, beta = self.film_generator(film_input)
                    x_final = x_base * (1.0 + gamma) + beta
                else:
                    if self.cond_fusion_mode == 'concat':
                        x_final = _apply_fusion_proj_without_concat(
                            x=x_fused,
                            time_token=time_token,
                            cond_feats=cond_feats,
                            fusion_proj=self.fusion_proj,
                        )
                    else:
                        x_final = _apply_fusion_proj_without_concat(
                            x=x_fused,
                            time_token=time_token,
                            fusion_proj=self.fusion_proj,
                        ) + cond_feats
            else:
                x_final = _apply_fusion_proj_without_concat(
                    x=x_fused,
                    time_token=time_token,
                    fusion_proj=self.fusion_proj,
                )

        if return_intermediates:
            intermediates['cond_emb'] = cond_feats
        
        if return_intermediates:
            intermediates['fused'] = x_final
        
        # Mask inactive nodes before GNN
        if active_mask_target is not None:
            mask_4d = active_mask_target.unsqueeze(1).unsqueeze(-1).float()
            x_final = x_final * mask_4d
        
        if return_intermediates:
            intermediates['gnn_input'] = x_final
        
        # Process through GNN
        x_out = self.gnn(
            x_final,
            edge_index,
            edge_weight,
            active_mask=active_mask_target,
        )

        # Preserve sparsity semantics after message passing.
        if active_mask_target is not None:
            mask_4d = active_mask_target.unsqueeze(1).unsqueeze(-1).float()
            x_out = x_out * mask_4d
        
        if return_intermediates:
            intermediates['gnn_output'] = x_out
            intermediates['output'] = x_out
            intermediates['active_mask_out'] = active_mask_target
        
        # if return_intermediates:
        return x_out, intermediates
        # return x_out
    
    def extra_repr(self) -> str:
        return (f"in_channels={self.in_channels}, "
                f"skip_channels={self.skip_channels}, "
                f"out_channels={self.out_channels}, "
                f"skip_mode={self.skip_connection_mode}, "
                f"cond_fusion_mode={self.cond_fusion_mode}")


class UGNNEncoder(nn.Module):
    """
    Contracting path (encoder) of UGNN.
    
    Stacks multiple encoder blocks with progressive downsampling.
    Each level processes features at different graph resolutions.
    
    Args:
        in_channels: Input feature dimension
        config: UGNN configuration containing all architecture parameters
    
    Example:
        >>> config = UGNNConfig(
        ...     in_channels=32,
        ...     out_channels=32,
        ...     base_channels=64,
        ...     channel_multipliers=[1, 2, 4],
        ... )
        >>> encoder = UGNNEncoder(in_channels=32, config=config)
        >>> x = torch.randn(4, 10, 100, 32)
        >>> timesteps = torch.randint(0, 1000, (4,))
        >>> edge_index = torch.randint(0, 400, (2, 500))
        >>> time_emb = torch.randn(4, 128)
        >>> 
        >>> features, masks = encoder(x, timesteps, edge_index, time_emb=time_emb)
        >>> # features: list of (B, T, N, F) at each level
        >>> # masks: list of (B, N) active masks at each level
    """
    
    def __init__(
        self,
        in_channels: int,
        config: UGNNConfig,
        cond_encoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.config = config
        self.base_channels = config.base_channels
        self.num_levels = config.num_levels
        self.channel_multipliers = config.channel_multipliers
        self.has_conditioning = config.has_conditioning
        self.compact_skip_cache = bool(config.compact_skip_cache)
        
        # Handle gamma: convert to list if single int
        gamma = config.pooling_config.gamma
        if isinstance(gamma, int):
            self.gammas = [gamma] * self.num_levels
        else:
            self.gammas = gamma
            if len(self.gammas) != self.num_levels:
                raise ValueError(
                    f"Length of gamma list ({len(self.gammas)}) must match "
                    f"number of levels ({self.num_levels})"
                )

        # score_gain_per_level is indexed only by active learned selectors
        # (levels where gamma > 1). Enforce exact length to avoid silent fallbacks.
        if config.pooling_config.score_gain_per_level is not None:
            if config.pooling_config.selection_method != 'learned':
                raise ValueError(
                    "score_gain_per_level can only be used with selection_method='learned', "
                    f"but got selection_method='{config.pooling_config.selection_method}'. "
                    "For stride/random selection, score gain is not applicable."
                )
            if config.pooling_config.selector_version != 'legacy':
                raise ValueError(
                    "score_gain_per_level is only supported for legacy selector. "
                    f"Got selector_version='{config.pooling_config.selector_version}'. "
                    "Use learned selector-specific arguments for v3 selectors."
                )
            active_selector_count = sum(1 for g in self.gammas if g > 1)
            provided = len(config.pooling_config.score_gain_per_level)
            if provided != active_selector_count:
                raise ValueError(
                    "score_gain_per_level length must match the number of active "
                    "learned selector levels (gamma > 1). "
                    f"Got {provided} values, expected {active_selector_count} "
                    f"for gamma={self.gammas}. "
                    f"Levels with gamma > 1 require score_gain_init values: "
                    f"indices {[i for i, g in enumerate(self.gammas) if g > 1]}."
                )
        
        # Compute stride at each level for GNN and pooling neighborhoods
        # stride_pre[i] = spacing between active nodes BEFORE pooling at level i
        # stride_post[i] = spacing between active nodes AFTER pooling at level i
        # 
        # NOTE: Graph structure (N nodes, edge_index) NEVER changes!
        # "Downsampling" = masking inactive nodes to zero (zero-padding)
        # Stride tracks the spacing between non-zero signal values
        self.stride_pre = [1]  # Level 0: dense signal (all nodes active)
        self.stride_post = []  # Will store accumulated stride after each level
        
        accum = 1
        for i, g in enumerate(self.gammas):
            accum *= g
            self.stride_post.append(accum)
            if i < self.num_levels - 1:
                self.stride_pre.append(accum)
        
        # Example: gammas=[2,2,3], N=100 nodes
        # stride_pre  = [1, 2, 4]    (GNN input signal spacing)
        # stride_post = [2, 4, 12]   (pooling output signal spacing)
        # 
        # Level 0: GNN sees dense signal, pooling uses gamma=2
        #   - GNN: 0,1,2,3-hop neighborhoods (stride_pre=1, K=3)
        #   - Pool: selects every 2nd node, aggregates 0,2,4-hop (gamma=2, pool_K=2)
        #   - Output: ~N/2 active nodes with stride=2 spacing
        # 
        # Level 1: GNN sees sparse signal (stride=2), pooling uses gamma=2
        #   - GNN: 0,2,4,6-hop neighborhoods (stride_pre=2, K=3)
        #   - Pool: selects every 2nd active node, aggregates 0,2,4-hop (gamma=2, pool_K=2)
        #   - Output: ~N/4 active nodes with stride=4 spacing
        # 
        # Level 2: GNN sees sparser signal (stride=4), pooling uses gamma=3
        #   - GNN: 0,4,8,12-hop neighborhoods (stride_pre=4, K=3)
        #   - Pool: selects every 3rd active node, aggregates 0,3,6-hop (gamma=3, pool_K=2)
        #   - Output: ~N/12 active nodes with stride=12 spacing
        
        # Conditional encoder (shared module can be injected by top-level UGNN)
        if self.has_conditioning:
            if cond_encoder is None:
                cond_encoder = _build_conditional_encoder(config)
            self.cond_encoder = cond_encoder
        
        # Input projection to base channels
        self.input_proj = nn.Linear(in_channels, self.base_channels * self.channel_multipliers[0])
        
        # Build encoder blocks
        self.encoder_blocks = nn.ModuleList()

        # Track selector level index (only levels with gamma > 1 have active selectors).
        selector_level_idx = 0
        
        for i, mult in enumerate(self.channel_multipliers):
            if i == 0:
                in_ch = self.base_channels * mult
                out_ch = self.base_channels * mult
            else:
                in_ch = self.base_channels * self.channel_multipliers[i-1]
                out_ch = self.base_channels * mult

            # Build per-level selector_kwargs with depth-aware overrides.
            level_selector_kwargs = dict(config.pooling_config.selector_kwargs or {})
            legacy_selector = (
                config.pooling_config.selection_method == 'learned'
                and config.pooling_config.selector_version == 'legacy'
            )

            has_active_selector = (
                self.gammas[i] > 1
                and config.pooling_config.selection_method == 'learned'
            )

            if has_active_selector:
                # — Per-level entropy_reg_weight scaling —
                base_erw = config.pooling_config.entropy_reg_weight
                depth_scale = config.pooling_config.entropy_reg_depth_scale
                if depth_scale != 1.0 and base_erw > 0:
                    level_selector_kwargs['entropy_reg_weight'] = (
                        base_erw * (depth_scale ** selector_level_idx)
                    )
                else:
                    # Even with depth_scale=1.0, forward the base value
                    level_selector_kwargs['entropy_reg_weight'] = base_erw

                if legacy_selector:
                    # — Per-level score_gain_init (legacy path only) —
                    sgl = config.pooling_config.score_gain_per_level
                    if sgl is not None and selector_level_idx < len(sgl):
                        level_selector_kwargs['score_gain_init'] = sgl[selector_level_idx]

                    # — learnable_score_gain —
                    level_selector_kwargs.setdefault(
                        'learnable_score_gain',
                        config.pooling_config.learnable_score_gain,
                    )

                selector_level_idx += 1
            
            # Create per-level pooling config with the right gamma
            level_pooling_config = PoolingConfig(
                gamma=self.gammas[i],
                pool_K=config.pooling_config.pool_K,
                selection_method=config.pooling_config.selection_method,
                selector_kwargs=level_selector_kwargs,
                temporal_score_agg=config.pooling_config.temporal_score_agg,
                ema_alpha_init=config.pooling_config.ema_alpha_init,
                selector_hidden_norm_type=config.pooling_config.selector_hidden_norm_type,
                selector_version=config.pooling_config.selector_version,
                min_retained_nodes=config.pooling_config.min_retained_nodes,
                selector_temporal_reduce=config.pooling_config.selector_temporal_reduce,
                selector_use_cond_fusion=config.pooling_config.selector_use_cond_fusion,
                selector_cond_fusion_mode=config.pooling_config.selector_cond_fusion_mode,
                selector_cond_fusion_heads=config.pooling_config.selector_cond_fusion_heads,
                selector_cond_fusion_dropout=config.pooling_config.selector_cond_fusion_dropout,
                selector_time_emb_dim=config.pooling_config.selector_time_emb_dim,
                selector_exploration_noise=config.pooling_config.selector_exploration_noise,
                selector_exploration_noise_min=config.pooling_config.selector_exploration_noise_min,
                selector_exploration_noise_schedule=config.pooling_config.selector_exploration_noise_schedule,
                selector_exploration_noise_anneal_steps=config.pooling_config.selector_exploration_noise_anneal_steps,
                selector_exploration_noise_warmup_steps=config.pooling_config.selector_exploration_noise_warmup_steps,
                selector_collect_heavy_diagnostics=config.pooling_config.selector_collect_heavy_diagnostics,
                selector_packed_score_mode=config.pooling_config.selector_packed_score_mode,
                selector_packed_score_threshold=config.pooling_config.selector_packed_score_threshold,
                neighborhood_pooling=config.pooling_config.neighborhood_pooling,
                neighborhood_include_self=config.pooling_config.neighborhood_include_self,
                neighborhood_max_dense_gb=config.pooling_config.neighborhood_max_dense_gb,
                selector_skip_gamma1_levels=config.pooling_config.selector_skip_gamma1_levels,
                max_pool_stride=config.pooling_config.max_pool_stride,
            )
            
            block = EncoderBlock(
                in_channels=in_ch,
                out_channels=out_ch,
                stride_pre=self.stride_pre[i],   # Input signal spacing (for GNN)
                stride_post=self.stride_post[i], # Output signal spacing (for pooling)
                gnn_config=config.gnn_config,
                pooling_config=level_pooling_config,
                embedding_config=config.embedding_config,
            )
            self.encoder_blocks.append(block)
    
    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[torch.Tensor, "batch timesteps nodes features"],
        timesteps: Int[torch.Tensor, "batch"],
        edge_index: Int[torch.Tensor, "2 edges"],
        edge_weight: Optional[Float[torch.Tensor, "edges"]] = None,
        cond: Optional[Float[torch.Tensor, "batch timesteps_cond nodes cond_features"]] = None,
        cond_emb_precomputed: Optional[torch.Tensor] = None,
        time_emb: Optional[Float[torch.Tensor, "batch time_embed_dim"]] = None,
        return_intermediates: bool = False,
    ) -> Tuple[torch.Tensor, List[Any], List[torch.Tensor], Optional[dict | None]]:
        """
        Forward pass through encoder (contracting path).

        Args:
            x: Input features (B, T, N, F_in)
            timesteps: Diffusion timesteps (B,)
            edge_index: Edge indices (2, E)
            edge_weight: Optional edge weights (E,)
            cond: Optional conditional signals (B, T_cond, N, F_cond)
            cond_emb_precomputed: Optional precomputed conditioning embeddings.
                Supported shapes:
                - (B, N, cond_embed_dim) static conditioning
                - (B, T, N, cond_embed_dim) time-varying conditioning
            time_emb: Precomputed time embeddings (B, time_embed_dim)
            return_intermediates: If True, return dict of intermediate tensors from each level

        Returns:
            x: Post-pooling features from the deepest encoder level (B, T, N, F_deep)
            skip_features: List of features at each level for skip connections
            active_masks: List of active masks at each level
            intermediates: Dict mapping level index to intermediate tensors (if return_intermediates=True)
                            None (if return_intermediates=False)
        """
        B, T, N, F = x.shape
        
        # Use precomputed conditional embeddings when available; otherwise compute
        # locally for standalone encoder usage.
        cond_emb = None
        if self.has_conditioning:
            if cond_emb_precomputed is not None:
                cond_emb = cond_emb_precomputed
            elif cond is not None:
                cond_emb = self.cond_encoder(cond, target_timesteps=T)
            else:
                raise ValueError(
                    "Conditioning is enabled but neither cond_emb_precomputed nor cond was provided."
                )
        
        # Input projection: (B, T, N, F) -> (B, T, N, base_channels)
        x = self.input_proj(x)  # (B, T, N, base_channels * channel_multipliers[0])
        
        # Store features and masks at each level
        skip_features = []
        active_masks = []
        active_mask = None  # Start with all nodes active
        all_intermediates = {} if return_intermediates else None
        
        # Process through encoder blocks
        for i, block in enumerate(self.encoder_blocks):
            active_mask_in = active_mask
            x, active_mask, x_skip, block_intermediates = block(
                x=x,
                timesteps=timesteps,
                edge_index=edge_index,
                edge_weight=edge_weight,
                active_mask=active_mask,
                cond_emb=cond_emb,
                time_emb=time_emb,
                return_intermediates=return_intermediates,
            )
            
            # Store for skip connections (use features BEFORE pooling)
            if (
                self.compact_skip_cache
                and not return_intermediates
                and i > 0
                and active_mask_in is not None
                and not bool(active_mask_in.all())
            ):
                skip_features.append(_pack_skip_tensor(x_skip, active_mask_in))
            else:
                skip_features.append(x_skip)
            active_masks.append(active_mask)
            
            # Store intermediates for this level
            if return_intermediates:
                all_intermediates[f'encoder_level_{i}'] = block_intermediates
        
        return x, skip_features, active_masks, all_intermediates

    def extra_repr(self) -> str:
        return (f"in_channels={self.in_channels}, "
                f"base_channels={self.base_channels}, "
                f"num_levels={self.num_levels}, "
                f"multipliers={self.channel_multipliers}")


class UGNNDecoder(nn.Module):
    """
    Expanding path (decoder) of UGNN.
    
    Stacks multiple decoder blocks with progressive upsampling.
    Each level processes features at different graph resolutions,
    fusing with skip connections from encoder.
    
    Args:
        out_channels: Output feature dimension (final output)
        config: UGNN configuration containing all architecture parameters
    
    Example:
        >>> config = UGNNConfig(
        ...     in_channels=32,
        ...     out_channels=32,
        ...     base_channels=64,
        ...     channel_multipliers=[1, 2, 4],
        ... )
        >>> decoder = UGNNDecoder(out_channels=32, config=config)
        >>> bottleneck = torch.randn(4, 10, 100, 256)  # From encoder
        >>> skip_features = [...]  # List of encoder skip features
        >>> active_masks = [...]  # List of encoder active masks
        >>> timesteps = torch.randint(0, 1000, (4,))
        >>> edge_index = torch.randint(0, 400, (2, 500))
        >>> time_emb = torch.randn(4, 128)
        >>> 
        >>> output = decoder(bottleneck, skip_features, active_masks,
        ...                 timesteps, edge_index, time_emb=time_emb)
    """
    
    def __init__(
        self,
        out_channels: int,
        config: UGNNConfig,
        cond_encoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        
        self.out_channels = out_channels
        self.config = config
        self.base_channels = config.base_channels
        self.num_levels = config.num_levels
        self.channel_multipliers = config.channel_multipliers
        self.has_conditioning = config.has_conditioning
        self.skip_connection_mode = config.skip_connection_mode
        self.compact_skip_cache = bool(config.compact_skip_cache)
        
        # Handle gamma: convert to list if single int
        gamma = config.pooling_config.gamma
        if isinstance(gamma, int):
            self.gammas = [gamma] * self.num_levels
        else:
            self.gammas = gamma
        
        # Compute stride at each level (same as encoder)
        self.stride_pre = [1]
        self.stride_post = []
        
        accum = 1
        for i, g in enumerate(self.gammas):
            accum *= g
            self.stride_post.append(accum)
            if i < self.num_levels - 1:
                self.stride_pre.append(accum)
        
        # Conditional encoder (shared module can be injected by top-level UGNN)
        if self.has_conditioning:
            if cond_encoder is None:
                cond_encoder = _build_conditional_encoder(config)
            self.cond_encoder = cond_encoder
        
        # Build decoder blocks (reverse order of encoder)
        self.decoder_blocks = nn.ModuleList()
        
        # Decoder processes from coarse to fine (reverse of encoder)
        for i in reversed(range(self.num_levels)):
            # Decoder level mirrors encoder level i
            if i == self.num_levels - 1:
                # First decoder block: input from bottleneck
                in_ch = self.base_channels * self.channel_multipliers[i]
            else:
                # Subsequent blocks: input from previous decoder
                in_ch = self.base_channels * self.channel_multipliers[i + 1]
            
            # Skip connection from encoder level i
            skip_ch = self.base_channels * self.channel_multipliers[i]
            
            # Output channels at this level
            out_ch = self.base_channels * self.channel_multipliers[i]
            
            block = DecoderBlock(
                in_channels=in_ch,
                skip_channels=skip_ch,
                out_channels=out_ch,
                stride_pre=self.stride_pre[i],  # After unpooling, matches encoder input stride
                gnn_config=config.gnn_config,
                upsampling_config=config.upsampling_config,
                embedding_config=config.embedding_config,
                skip_connection_mode=config.skip_connection_mode,
            )
            self.decoder_blocks.append(block)
        
        # Final output projection.
        # Default (output_head_config.mode='linear') reproduces the historical
        # bare nn.Linear head exactly (same module type, same parameter names),
        # so prior checkpoints load with strict=True. Set
        # output_head_config.mode='norm_act_linear' (+ zero_init=true) to opt
        # into the DDPM-style Norm→Activation→Linear head.
        final_channels = self.base_channels * self.channel_multipliers[0]
        self.output_proj = _build_output_head(
            final_channels=final_channels,
            out_channels=out_channels,
            cfg=config.output_head_config,
        )
    
    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[torch.Tensor, "batch timesteps nodes features"],
        skip_features: List[Any],
        active_masks: List[Bool[torch.Tensor, "batch nodes"]],
        timesteps: Int[torch.Tensor, "batch"],
        edge_index: Int[torch.Tensor, "2 edges"],
        edge_weight: Optional[Float[torch.Tensor, "edges"]] = None,
        cond: Optional[Float[torch.Tensor, "batch cond_timesteps nodes cond_features"]] = None,
        cond_emb_precomputed: Optional[torch.Tensor] = None,
        time_emb: Optional[Float[torch.Tensor, "batch time_embed_dim"]] = None,
        return_intermediates: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict | None]]:
        """
        Forward pass through decoder (expanding path).
        
        Args:
            x: Bottleneck features (B, T, N, F_bottleneck)
            skip_features: List of encoder skip features (finest to coarsest)
            active_masks: List of encoder active masks (finest to coarsest). Each mask is a binary boolean tensor (B, N, dtype=torch.bool)
            timesteps: Diffusion timesteps (B,)
            edge_index: Edge indices (2, E)
            edge_weight: Optional edge weights (E,) always scalarized by UGNN
            cond: Optional conditional signals (B, T_cond, N, F_cond).
                Kept for API symmetry/backward signature stability and ignored
                by the conditioning-enabled decoder path.
            cond_emb_precomputed: Precomputed conditional embeddings shared with encoder.
                Required when has_conditioning=True.
            time_emb: Precomputed time embeddings (B, time_embed_dim)
            return_intermediates: If True, return dict of intermediate tensors from each level
        
        Returns:
            Output features (B, T, N, F_out)
            intermediates: Dict mapping decoder level to intermediate tensors (if return_intermediates=True)
                            None (if return_intermediates=False)
        """
        # Decoder never performs local conditional encoding to avoid stochastic
        # inconsistency with encoder-side conditioning.
        cond_emb = None
        if self.has_conditioning:
            if cond_emb_precomputed is None:
                raise ValueError(
                    "Decoder requires cond_emb_precomputed when has_conditioning=True. "
                    "Pass precomputed conditioning from UGNN.forward()."
                )
            cond_emb = cond_emb_precomputed
        
        # Store intermediates from all levels
        all_intermediates = {} if return_intermediates else None
        
        # Track current active mask as we progress through decoder
        # Start with the bottleneck mask (most coarse, fewest active nodes)
        current_active_mask = active_masks[-1]  # Mask from deepest encoder level
        
        # Process through decoder blocks (coarse to fine)
        for i, block in enumerate(self.decoder_blocks):
            # Map decoder block i to encoder level
            encoder_level = self.num_levels - 1 - i
            
            # Get skip features and target active mask from encoder
            # NOTE(v2 deferred): when compact_skip_cache=True and sparse packed
            # skip fusion is enabled, this unpack could be skipped by passing the
            # packed representation through the matched decoder level directly.
            skip = _unpack_skip_tensor(skip_features[encoder_level])
            
            # Target mask after upsampling should match the INPUT to that encoder level
            # For encoder level 0: all nodes active (initial input)
            # For encoder level i>0: active_masks[i-1] (output of previous encoder level)
            if encoder_level == 0:
                # Upsample to full resolution (all nodes active)
                N = skip.shape[2]
                target_mask = torch.ones(skip.shape[0], N, dtype=torch.bool, device=skip.device)
            else:
                # Upsample to the mask from previous encoder level
                target_mask = active_masks[encoder_level - 1]
            
            # Process through decoder block
            x, level_intermediates = block(
                x=x,
                skip_features=skip,
                timesteps=timesteps,
                edge_index=edge_index,
                edge_weight=edge_weight,
                active_mask_target=target_mask,
                active_mask_input=current_active_mask,  # Pass current mask
                cond_emb=cond_emb,
                time_emb=time_emb,
                return_intermediates=return_intermediates,
            )
            if return_intermediates:
                all_intermediates[f'decoder_level_{encoder_level}'] = level_intermediates
            
            # Update current mask for next iteration
            current_active_mask = target_mask
        
        # Final output projection
        out = self.output_proj(x)
        
        return out, all_intermediates
    
    def extra_repr(self) -> str:
        return (f"out_channels={self.out_channels}, "
                f"base_channels={self.base_channels}, "
                f"num_levels={self.num_levels}, "
                f"skip_mode={self.skip_connection_mode}")


from graph_signal_diffusion.models import MODEL_REGISTRY

@MODEL_REGISTRY.register("ugnn")
class UGNN(nn.Module):
    """
    U-Net style Graph Neural Network for diffusion models.
    
    Complete U-Net architecture with encoder, bottleneck, and decoder.
    
    Architecture:
        Input → Encoder (contracting path)
                ↓
              Bottleneck (coarsest resolution processing)
                ↓
              Decoder (expanding path with skip connections)
                ↓
              Output
    
    Args:
        config: UGNN configuration containing all architecture parameters
    
    Example:
        >>> config = UGNNConfig(
        ...     in_channels=1,
        ...     out_channels=1,
        ...     base_channels=64,
        ...     channel_multipliers=[1, 2, 4, 8],
        ...     num_bottleneck_layers=1,
        ... )
        >>> ugnn = UGNN(config=config)
        >>> x = torch.randn(4, 10, 100, 1)  # (B, T, N, 1)
        >>> timesteps = torch.randint(0, 1000, (4,))
        >>> edge_index = torch.randint(0, 400, (2, 500))
        >>> out = ugnn(x, timesteps, edge_index)  # (B, T, N, 1)
    """
    
    def __init__(
        self,
        config: UGNNConfig,
        **kwargs, # absorb non-init config fields like `name`
    ):
        super().__init__()
        
        self.config = config
        self.in_channels = config.in_channels
        self.out_channels = config.out_channels
        self.base_channels = config.base_channels
        self.num_levels = config.num_levels
        self.num_bottleneck_layers = config.num_bottleneck_layers
        
        # Time embedding
        self.time_embed = TimeEmbedding(
            embed_dim=config.embedding_config.time_embed_dim,
            num_timesteps=config.embedding_config.num_timesteps,
        )

        # Encoder (contracting path)
        self.encoder = UGNNEncoder(
            in_channels=config.in_channels,
            config=config,
        )
        
        # Bottleneck (optional processing at coarsest resolution)
        bottleneck_channels = config.base_channels * config.channel_multipliers[-1]
        if self.num_bottleneck_layers > 0:
            # Compute stride at coarsest level
            gamma = config.pooling_config.gamma
            if isinstance(gamma, int):
                gammas = [gamma] * self.num_levels
            else:
                gammas = gamma
            
            # Accumulate stride through all encoder levels
            bottleneck_stride = 1
            for g in gammas:
                bottleneck_stride *= g
            
            self.bottleneck = GNN(
                in_channels=bottleneck_channels,
                hidden_channels=bottleneck_channels,
                out_channels=bottleneck_channels,
                num_layers=self.num_bottleneck_layers,
                K=config.gnn_config.K,
                norm_type=config.gnn_config.norm_type,
                dropout=config.gnn_config.dropout,
                activation=config.gnn_config.activation,
                normalize=config.gnn_config.normalize,
                use_strided=config.gnn_config.use_strided_conv,
                gamma=(
                    min(bottleneck_stride, config.gnn_config.max_gnn_stride)
                    if config.gnn_config.use_strided_conv and config.gnn_config.max_gnn_stride is not None
                    else (bottleneck_stride if config.gnn_config.use_strided_conv else 2)
                ),
                strided_power_mode=config.gnn_config.strided_power_mode,
                use_pre_activation=config.gnn_config.use_pre_activation,
                use_temporal_mixer=config.gnn_config.use_temporal_mixer,
                temporal_mixer_schedule=config.gnn_config.temporal_mixer_schedule,
                temporal_kernel_size=config.gnn_config.temporal_kernel_size,
                temporal_causal=config.gnn_config.temporal_causal,
                temporal_use_pointwise=config.gnn_config.temporal_use_pointwise,
                temporal_gated=config.gnn_config.temporal_gated,
                temporal_attention_enabled=config.gnn_config.temporal_attention_enabled,
                temporal_attention_num_heads=config.gnn_config.temporal_attention_num_heads,
                temporal_attention_dropout=config.gnn_config.temporal_attention_dropout,
                temporal_attention_max_timesteps=config.gnn_config.temporal_attention_max_timesteps,
                # When per_layer schedule is used, the dilations list must match
                # the layer count.  The bottleneck may have a different number of
                # layers than encoder/decoder blocks, so pass None to let the GNN
                # constructor build its own default ([1]*num_layers) when they differ.
                temporal_dilations=(
                    config.gnn_config.temporal_dilations
                    if (config.gnn_config.temporal_dilations is None
                        or config.gnn_config.temporal_mixer_schedule != 'per_layer'
                        or len(config.gnn_config.temporal_dilations) == self.num_bottleneck_layers)
                    else None
                ),
                pointwise_sparse_mode=config.gnn_config.pointwise_sparse_mode,
                pointwise_sparse_threshold=config.gnn_config.pointwise_sparse_threshold,
            )
        else:
            self.bottleneck = nn.Identity()
        
        # Decoder (expanding path)
        self.decoder = UGNNDecoder(
            out_channels=config.out_channels,
            config=config,
        )

        # Keep decoder conditional encoder registered for checkpoint compatibility,
        # but freeze it because UGNN.forward uses encoder-side precomputed conditioning.
        if config.has_conditioning:
            self.decoder.cond_encoder.requires_grad_(False)
    
    @jaxtyped(typechecker=beartype)
    def forward(
        self,
        x: Float[torch.Tensor, "batch timesteps nodes features"],
        timesteps: Int[torch.Tensor, "batch"],
        edge_index: Int[torch.Tensor, "2 edges"],
        # edge_weight: Optional[Float[torch.Tensor, "edges"]] = None,
        edge_weight: Optional[Union[Float[torch.Tensor, "edges"], Float[torch.Tensor, "edges channels"]]] = None,
        cond: Optional[Float[torch.Tensor, "batch cond_timesteps nodes cond_features"]] = None,
        return_intermediates: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, dict | None]]:
        """
        Forward pass through UGNN.
        
        Args:
            x: Input features (B, T, N, F_in)
            timesteps: Diffusion timesteps (B,)
            edge_index: Edge indices (2, E)
            edge_weight: Optional edge weights (E,) or (E, channels) for multi-channel edges
            cond: Optional conditional signals (B, T_cond, N, F_cond)
            return_intermediates: If True, return (output, intermediates_dict) from encoder, decoder and bottleneck
        
        Returns:
            Output features (B, T, N, F_out)
            If return_intermediates=True: (output, dict of encoder, decoder and bottleneck intermediate tensors)
                else: (output, None)
        """

        # Scalarize the edge weights if multi-channel
        if edge_weight is not None and edge_weight.dim() == 2:
            edge_weight = edge_weight.mean(dim=-1)  # (E,)

        # Embed timesteps
        time_emb = self.time_embed(timesteps)  # (B, time_embed_dim)

        # Compute conditional embeddings exactly once from encoder-side module.
        target_timesteps = x.shape[1]
        cond_emb_once = None
        if self.config.has_conditioning:
            if cond is None:
                raise ValueError(
                    "Conditioning is enabled but no `cond` tensor was provided to UGNN.forward()."
                )
            cond_emb_once = self.encoder.cond_encoder(
                cond,
                target_timesteps=target_timesteps,
            )
        
        # Encoder (contracting path)
        x_enc, skip_features, active_masks, encoder_intermediates = self.encoder(
            x=x,
            timesteps=timesteps,
            edge_index=edge_index,
            edge_weight=edge_weight,
            cond=cond,
            cond_emb_precomputed=cond_emb_once,
            time_emb=time_emb,
            return_intermediates=return_intermediates,
        )

        # Bottleneck processing at coarsest resolution.
        # Use x_enc (post-pooling from deepest encoder level) so that the
        # deepest selector's STE selection_mask stays in the computation graph
        # and receives gradients from the diffusion loss.
        x_bottleneck = x_enc  # (B, T, N, bottleneck_channels)
        # Always handle bottleneck intermediates for interface consistency
        bottleneck_input = x_bottleneck.clone()
        if self.num_bottleneck_layers > 0:
            # Get active mask at coarsest level
            bottleneck_mask = active_masks[-1]

            # Mask inactive nodes before bottleneck
            if bottleneck_mask is not None:
                mask_4d = bottleneck_mask.unsqueeze(1).unsqueeze(-1).float()
                x_bottleneck = x_bottleneck * mask_4d

            # Process through bottleneck GNN
            x_bottleneck = self.bottleneck(
                x_bottleneck,
                edge_index,
                edge_weight,
                active_mask=bottleneck_mask,
            )
            if bottleneck_mask is not None:
                mask_4d = bottleneck_mask.unsqueeze(1).unsqueeze(-1).float()
                x_bottleneck = x_bottleneck * mask_4d

        bottleneck_output = x_bottleneck.clone()

        # Decoder (expanding path)
        decoder_out, decoder_intermediates = self.decoder(
            x=x_bottleneck,
            skip_features=skip_features,
            active_masks=active_masks,
            timesteps=timesteps,
            edge_index=edge_index,
            edge_weight=edge_weight,
            cond=cond,
            cond_emb_precomputed=cond_emb_once,
            time_emb=time_emb,
            return_intermediates=return_intermediates,
        ) 
        
        # if return_intermediates else (self.decoder(
        #     x=x_bottleneck,
        #     skip_features=skip_features,
        #     active_masks=active_masks,
        #     timesteps=timesteps,
        #     edge_index=edge_index,
        #     edge_weight=edge_weight,
        #     cond=cond,
        #     time_emb=time_emb,
        #     return_intermediates=False,
        # ), None)

        if return_intermediates:
            # Merge all intermediates
            intermediates = encoder_intermediates if encoder_intermediates is not None else {}
            if decoder_intermediates is not None:
                intermediates.update(decoder_intermediates)
            intermediates['bottleneck_input'] = bottleneck_input
            intermediates['bottleneck_output'] = bottleneck_output
            # return decoder_out, intermediates
        else:
            intermediates = None

        return decoder_out, intermediates
    
    def get_architecture_stats(self) -> Dict[str, Any]:
        """Return JSON-serializable architecture/size stats for logging."""
        def _num_params(module: Optional[nn.Module], trainable_only: bool = False) -> int:
            if module is None:
                return 0
            if trainable_only:
                return int(sum(p.numel() for p in module.parameters() if p.requires_grad))
            return int(sum(p.numel() for p in module.parameters()))

        def _num_bytes(module: Optional[nn.Module]) -> int:
            if module is None:
                return 0
            param_bytes = sum(p.numel() * p.element_size() for p in module.parameters())
            buffer_bytes = sum(b.numel() * b.element_size() for b in module.buffers())
            return int(param_bytes + buffer_bytes)

        pooling_cfg = self.config.pooling_config
        gamma_cfg = pooling_cfg.gamma
        gamma_levels = (
            [int(gamma_cfg)] * self.num_levels
            if isinstance(gamma_cfg, int)
            else [int(g) for g in gamma_cfg]
        )

        total_params = _num_params(self)
        trainable_params = _num_params(self, trainable_only=True)
        frozen_params = int(total_params - trainable_params)
        total_size_bytes = _num_bytes(self)

        stats: Dict[str, Any] = {
            "model_name": self.__class__.__name__,
            "in_channels": int(self.in_channels),
            "out_channels": int(self.out_channels),
            "base_channels": int(self.base_channels),
            "num_levels": int(self.num_levels),
            "channel_multipliers": [int(c) for c in self.config.channel_multipliers],
            "num_bottleneck_layers": int(self.num_bottleneck_layers),
            "use_conditioning": bool(self.config.has_conditioning),
            "pooling": {
                "selection_method": str(pooling_cfg.selection_method),
                "pool_K": int(pooling_cfg.pool_K),
                "gamma_by_level": gamma_levels,
            },
            "gnn": {
                "num_layers": int(self.config.gnn_config.num_layers),
                "K": int(self.config.gnn_config.K),
                "use_strided_conv": bool(self.config.gnn_config.use_strided_conv),
                "strided_power_mode": str(self.config.gnn_config.strided_power_mode),
            },
            "params_total": total_params,
            "params_trainable": trainable_params,
            "params_frozen": frozen_params,
            "size_total_mb": float(total_size_bytes / (1024.0 ** 2)),
            "params_by_component": {
                "time_embed": _num_params(self.time_embed),
                "encoder_cond_encoder": _num_params(getattr(self.encoder, "cond_encoder", None)),
                "decoder_cond_encoder": _num_params(getattr(self.decoder, "cond_encoder", None)),
                "encoder": _num_params(self.encoder),
                "bottleneck": _num_params(self.bottleneck),
                "decoder": _num_params(self.decoder),
            },
            "trainable_params_by_component": {
                "time_embed": _num_params(self.time_embed, trainable_only=True),
                "encoder_cond_encoder": _num_params(
                    getattr(self.encoder, "cond_encoder", None), trainable_only=True
                ),
                "decoder_cond_encoder": _num_params(
                    getattr(self.decoder, "cond_encoder", None), trainable_only=True
                ),
                "encoder": _num_params(self.encoder, trainable_only=True),
                "bottleneck": _num_params(self.bottleneck, trainable_only=True),
                "decoder": _num_params(self.decoder, trainable_only=True),
            },
        }
        return stats
    
    
    def extra_repr(self) -> str:
        return (f"in_channels={self.in_channels}, "
                f"out_channels={self.out_channels}, "
                f"base_channels={self.base_channels}, "
                f"num_levels={self.num_levels}, "
                f"has_conditioning={self.config.has_conditioning}")
