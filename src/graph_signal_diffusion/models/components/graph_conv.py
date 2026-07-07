"""
Graph convolutional layers for graph signal diffusion models.

This module provides:
- TAGConvLayer: Wrapper around PyG's TAGConv for temporal graph signals
- ResidualGNNBlock: Residual block with TAGConv + normalization + activation
- GNN: Complete GNN architecture with multiple TAGConv layers

TAGConv (Topology Adaptive Graph Convolutional Networks):
    Aggregates information from K-hop neighborhoods using learnable weights.
    More flexible than GCN - can capture multi-scale graph structure.
    
    Formula: h_i^(k) = sum_{j in N^k(i)} alpha^(k) * W^(k) * x_j
    where N^k(i) is the k-hop neighborhood of node i.

References:
    - TAGConv: Du et al. "Topology Adaptive Graph Convolutional Networks" (2017)
      https://arxiv.org/abs/1710.10370
    - PyTorch Geometric: https://pytorch-geometric.readthedocs.io/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TAGConv
from typing import Optional, Literal, Union, List

from .normalization import (
    get_normalization,
    AdaptiveLayerNorm,
    AdaptiveGroupNorm,
)



# Fallback for torch_scatter when CUDA extensions fail to load
def _scatter_add_fallback(src, index, dim=0, dim_size=None):
    """
    Native PyTorch implementation of scatter_add for when torch_scatter is unavailable.
    
    This performs: out[index[i]] += src[i] along the specified dimension.
    Equivalent to torch_scatter.scatter_add but using native PyTorch operations.
    """
    if dim_size is None:
        dim_size = int(index.max()) + 1
    
    # Create output tensor
    out_shape = list(src.shape)
    out_shape[dim] = dim_size
    out = torch.zeros(out_shape, dtype=src.dtype, device=src.device)
    
    # Use native scatter_add_
    out.scatter_add_(dim, index.unsqueeze(-1).expand_as(src), src)
    return out


# Try to import torch_scatter, fall back to native implementation if unavailable
try:
    from torch_scatter import scatter_add
except (ImportError, OSError):
    scatter_add = _scatter_add_fallback


def _pack_active_nodes_4d(
    x: torch.Tensor,
    active_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack active nodes from (B, T, N, C) into (A, T, C), where A=sum(active)."""
    if x.dim() != 4:
        raise ValueError(f"_pack_active_nodes_4d expects rank-4 x, got {x.dim()}D")
    if active_mask.dim() != 2:
        raise ValueError(
            f"_pack_active_nodes_4d expects active_mask shape (B, N), got {tuple(active_mask.shape)}"
        )

    B, T, N, C = x.shape
    if active_mask.size(0) != B or active_mask.size(1) != N:
        raise ValueError(
            f"active_mask must match (B, N)=({B}, {N}), got {tuple(active_mask.shape)}"
        )

    active_flat = active_mask.to(dtype=torch.bool).reshape(B * N)
    active_indices = torch.nonzero(active_flat, as_tuple=False).squeeze(-1)

    # (B, T, N, C) -> (B*N, T, C)
    x_bn = x.permute(0, 2, 1, 3).reshape(B * N, T, C)
    packed = x_bn.index_select(0, active_indices)
    return packed, active_indices


def _pack_active_nodes_4d_from_indices(
    x: torch.Tensor,
    active_indices: torch.Tensor,
) -> torch.Tensor:
    """Pack active nodes from (B,T,N,C) using precomputed flattened active indices."""
    if x.dim() != 4:
        raise ValueError(
            f"_pack_active_nodes_4d_from_indices expects rank-4 x, got {x.dim()}D"
        )
    B, T, N, C = x.shape
    x_bn = x.permute(0, 2, 1, 3).reshape(B * N, T, C)
    return x_bn.index_select(0, active_indices)


def _unpack_active_nodes_4d(
    packed: torch.Tensor,
    active_indices: torch.Tensor,
    *,
    batch_size: int,
    timesteps: int,
    num_nodes: int,
    channels: int,
) -> torch.Tensor:
    """Unpack (A, T, C) back to dense (B, T, N, C) with zeros at inactive nodes."""
    dense_bn = packed.new_zeros((batch_size * num_nodes, timesteps, channels))
    dense_bn.index_copy_(0, active_indices, packed)
    return dense_bn.reshape(batch_size, num_nodes, timesteps, channels).permute(0, 2, 1, 3).contiguous()


class TAGConvLayer(nn.Module):
    """
    TAGConv layer wrapper for temporal graph signals.
    
    Handles the shape transformations needed to apply PyG's TAGConv
    to temporal graph signals with shape (B, T, N, F).
    
    **Important:** PyG already batches B graphs by offsetting node indices.
    We only need to replicate for T timesteps.
    
    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension
        K: Number of hops (default: 3)
        bias: Whether to use bias (default: True)
        normalize: Whether to apply symmetric normalization (default: False)
        use_strided: Whether to use StridedTAGConv instead of regular TAGConv (default: False)
        gamma: Stride parameter for StridedTAGConv (default: 2). Only used if use_strided=True.
        use_boolean_semiring: For StridedTAGConv, use exact γ-hop (True) or all paths ≤ γ (False)
        strided_power_mode: Power-base behavior for StridedTAGConv when use_boolean_semiring=False.
            - 'a_plus_i' (default): use A_base = A + I for gamma > 1 (historical behavior)
            - 'exact_walk': use A_base = A (exact walk-length powers)
    
    Shape:
        - Input: (B, T, N, F_in) or (B*N, T, F_in)
        - Edge index: (2, E) - already batched for B graphs by PyG
        - Output: (B, T, N, F_out) or (B*N, T, F_out)
    
    Examples:
        >>> # Basic usage with regular TAGConv
        >>> layer = TAGConvLayer(in_channels=64, out_channels=64, K=3)
        >>> x = torch.randn(4, 10, 100, 64)  # (B=4, T=10, N=100, F=64)
        >>> edge_index = torch.randint(0, 400, (2, 2000))
        >>> out = layer(x, edge_index)  # (4, 10, 100, 64)
        
        >>> # With StridedTAGConv for multi-scale features
        >>> layer = TAGConvLayer(in_channels=64, out_channels=64, K=2, 
        ...                      use_strided=True, gamma=2)
        >>> out = layer(x, edge_index)  # Aggregates 0, 2, 4-hop neighbors
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int = 3,
        bias: bool = True,
        normalize: bool = False,
        use_strided: bool = False,
        gamma: int = 2,
        use_boolean_semiring: bool = False,
        strided_power_mode: Literal['a_plus_i', 'exact_walk'] = 'a_plus_i',
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K
        self.use_strided = use_strided
        self.gamma = gamma
        
        # Choose between regular TAGConv and StridedTAGConv
        if use_strided:
            self.conv = StridedTAGConv(
                in_channels=in_channels,
                out_channels=out_channels,
                gamma=gamma,
                K=K,
                bias=bias,
                normalize=normalize,
                use_boolean_semiring=use_boolean_semiring,
                power_mode=strided_power_mode,
            )
        else:
            # PyG's TAGConv layer
            self.conv = TAGConv(
                in_channels=in_channels,
                out_channels=out_channels,
                K=K,
                bias=bias,
                normalize=normalize,
            )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through TAGConv or StridedTAGConv.
        
        Args:
            x: Node features of shape (B, T, N, F) or (B*N, T, F)
            edge_index: Edge indices of shape (2, E) - already batched by PyG
            edge_weight: Optional edge weights of shape (E,)
        
        Returns:
            Output features with same shape as input
        """
        # StridedTAGConv handles temporal replication internally
        if self.use_strided:
            return self.conv(x, edge_index, edge_weight)
        
        # Regular TAGConv: need to handle temporal dimension manually
        is_4d = x.dim() == 4
        
        if is_4d:
            # (B, T, N, F) -> process each timestep independently
            B, T, N, F = x.shape
            num_nodes_per_timestep = B * N
            
            # Permute to (T, B, N, F) then reshape to (T*B*N, F)
            # This keeps nodes from the same batch together
            x = x.permute(1, 0, 2, 3).reshape(T * B * N, F)
            
            # Replicate edge_index for T timesteps
            # PyG already batched for B graphs, we only replicate for T
            edge_index_t = edge_index.unsqueeze(0).expand(T, -1, -1)  # (T, 2, E)
            
            # Offset node indices for each timestep
            # Timestep 0: nodes 0 to (B*N-1)
            # Timestep 1: nodes (B*N) to (2*B*N-1)
            # Timestep t: nodes (t*B*N) to ((t+1)*B*N-1)
            offsets = torch.arange(T, device=edge_index.device) * num_nodes_per_timestep
            edge_index_t = edge_index_t + offsets.view(T, 1, 1)
            edge_index_t = edge_index_t.reshape(2, T * edge_index.size(1))
            
            # Replicate edge weights if provided
            edge_weight_t = edge_weight.repeat(T) if edge_weight is not None else None
            
            # Apply TAGConv
            x = self.conv(x, edge_index_t, edge_weight_t)
            
            # Reshape back: (T*B*N, F_out) -> (T, B, N, F_out) -> (B, T, N, F_out)
            x = x.reshape(T, B, N, self.out_channels).permute(1, 0, 2, 3)
            
        elif x.dim() == 3:
            # (B*N, T, F) -> process each timestep independently
            BN, T, F = x.shape
            
            # Permute to (T, B*N, F) then reshape to (T*B*N, F)
            x = x.permute(1, 0, 2).reshape(T * BN, F)
            
            # Replicate edge_index for T timesteps
            edge_index_t = edge_index.unsqueeze(0).expand(T, -1, -1)
            offsets = torch.arange(T, device=edge_index.device) * BN
            edge_index_t = edge_index_t + offsets.view(T, 1, 1)
            edge_index_t = edge_index_t.reshape(2, T * edge_index.size(1))
            
            # Replicate edge weights if provided
            edge_weight_t = edge_weight.repeat(T) if edge_weight is not None else None
            
            # Apply TAGConv
            x = self.conv(x, edge_index_t, edge_weight_t)
            
            # Reshape back: (T*B*N, F_out) -> (T, B*N, F_out) -> (B*N, T, F_out)
            x = x.reshape(T, BN, self.out_channels).permute(1, 0, 2)
        
        else:
            # Already 2D (B*N*T, F) - apply directly
            # Assume edge_index is already properly formatted
            x = self.conv(x, edge_index, edge_weight)
        
        return x
    
    def extra_repr(self) -> str:
        if self.use_strided:
            return (f"in_channels={self.in_channels}, "
                    f"out_channels={self.out_channels}, K={self.K}, "
                    f"strided=True, gamma={self.gamma}")
        else:
            return (f"in_channels={self.in_channels}, "
                    f"out_channels={self.out_channels}, K={self.K}")


class ResidualGNNBlock(nn.Module):
    """
    Residual GNN block with TAGConv + normalization + activation.
    
    Architecture:
        If use_pre_activation=True:
            x -> Norm -> Activation -> TAGConv -> Dropout -> + (residual)
            |___________________________________________________________|
        Else (default):
            x -> TAGConv -> Norm -> Activation -> Dropout -> + (residual)
            |___________________________________________________________|
    
    Optionally supports adaptive normalization (time-conditioned).
    This block does not apply node activity masking internally; callers should
    mask inactive nodes before/after the block when hierarchical masking is used.
    
    Args:
        hidden_channels: Number of hidden channels
        K: Number of TAGConv hops (default: 3)
        norm_type: Type of normalization ('layer', 'group', 'instance', 'none')
        num_groups: Number of groups for GroupNorm (default: 8)
        dropout: Dropout rate (default: 0.0)
        activation: Activation function ('relu', 'silu', 'gelu', 'leaky_relu')
        use_adaptive_norm: Whether to use adaptive normalization (default: False)
        cond_dim: Conditioning dimension for adaptive norm (required if use_adaptive_norm=True)
        use_pre_activation: Whether to use pre-activation ordering (norm→act→conv) or
            post-activation ordering (conv→norm→act). Default: False (post-activation)
        use_strided: Whether to use StridedTAGConv (default: False)
        gamma: Stride parameter for StridedTAGConv (default: 2)
        use_boolean_semiring: For StridedTAGConv, use exact γ-hop (default: False)
        strided_power_mode: Power-base behavior for StridedTAGConv when use_boolean_semiring=False.
            'a_plus_i' (default) or 'exact_walk'.
    
    Shape:
        - Input x: (B, T, N, F) or (B*N, T, F)
        - Edge index: (2, E) - batched by PyG
        - Conditioning (optional): (B, cond_dim) or (B, N, cond_dim)
        - Output: Same shape as input
    
    Examples:
        >>> # Basic residual block with regular TAGConv
        >>> block = ResidualGNNBlock(hidden_channels=64, K=3, norm_type='group')
        >>> x = torch.randn(4, 10, 100, 64)
        >>> edge_index = torch.randint(0, 400, (2, 2000))
        >>> out = block(x, edge_index)  # (4, 10, 100, 64)
        
        >>> # With StridedTAGConv for multi-scale features
        >>> block = ResidualGNNBlock(hidden_channels=64, K=2, 
        ...                          use_strided=True, gamma=2)
        >>> out = block(x, edge_index)  # Aggregates 0, 2, 4-hop neighbors
    """
    
    def __init__(
        self,
        hidden_channels: int,
        K: int = 3,
        norm_type: Literal['layer', 'group', 'instance', 'none'] = 'group',
        num_groups: int = 8,
        dropout: float = 0.0,
        activation: Literal['relu', 'silu', 'gelu', 'leaky_relu'] = 'silu',
        use_adaptive_norm: bool = False,
        cond_dim: Optional[int] = None,
        use_pre_activation: bool = False,
        normalize: bool = False,  # Whether to apply symmetric normalization in TAGConv
        use_strided: bool = False,
        gamma: int = 2,
        use_boolean_semiring: bool = False,
        strided_power_mode: Literal['a_plus_i', 'exact_walk'] = 'a_plus_i',
    ):
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.K = K
        self.use_adaptive_norm = use_adaptive_norm
        self.use_pre_activation = use_pre_activation
        self.use_strided = use_strided
        self.normalize = normalize
        self.norm_type = norm_type

        if norm_type == 'graph':
            raise ValueError(
                "norm_type='graph' is not supported in ResidualGNNBlock. "
                "Use one of: 'layer', 'group', 'instance', 'none'."
            )
        
        # Normalization
        if use_adaptive_norm:
            if cond_dim is None:
                raise ValueError("cond_dim must be provided when use_adaptive_norm=True")
            
            if norm_type == 'group':
                self.norm = AdaptiveGroupNorm(
                    num_features=hidden_channels,
                    num_groups=num_groups,
                    cond_dim=cond_dim,
                )
            elif norm_type == 'layer':
                self.norm = AdaptiveLayerNorm(
                    num_features=hidden_channels,
                    cond_dim=cond_dim,
                )
            else:
                raise ValueError(
                    f"Adaptive normalization only supports 'layer' and 'group', got '{norm_type}'"
                )
        else:
            self.norm = get_normalization(
                norm_type=norm_type,
                num_features=hidden_channels,
                num_groups=num_groups,
            )
        
        # Activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # TAGConv layer (regular or strided)
        self.conv = TAGConvLayer(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            K=K,
            normalize=normalize,
            use_strided=use_strided,
            gamma=gamma,
            use_boolean_semiring=use_boolean_semiring,
            strided_power_mode=strided_power_mode,
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        active_mask: Optional[torch.Tensor] = None,
        use_packed_pointwise: bool = False,
        precomputed_active_indices: Optional[torch.Tensor] = None,
        return_packed_post: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through residual GNN block.
        
        Args:
            x: Node features of shape (B, T, N, F) or (B*N, T, F)
            edge_index: Edge indices of shape (2, E) - batched by PyG
            edge_weight: Optional edge weights of shape (E,)
            cond: Optional conditioning of shape (B, cond_dim) or (B, N, cond_dim)
        
        Returns:
            Output features with same shape as input
        """
        can_pack_pointwise = (
            bool(use_packed_pointwise)
            and active_mask is not None
            and x.dim() == 4
            and active_mask.dim() == 2
            and not self.use_adaptive_norm
        )

        # Store input for residual connection
        residual = x
        
        if self.use_pre_activation:
            # Pre-activation: Norm → Activation → Conv → Dropout
            # Better gradient flow through identity residual path

            if can_pack_pointwise:
                B, T, N, _ = x.shape
                if precomputed_active_indices is not None:
                    active_indices = precomputed_active_indices
                    x_packed = _pack_active_nodes_4d_from_indices(x, active_indices)
                else:
                    x_packed, active_indices = _pack_active_nodes_4d(x, active_mask)

                x_packed = self.norm(x_packed)
                x_packed = self.activation(x_packed)

                x = _unpack_active_nodes_4d(
                    x_packed,
                    active_indices,
                    batch_size=B,
                    timesteps=T,
                    num_nodes=N,
                    channels=self.hidden_channels,
                )
                x = self.conv(x, edge_index, edge_weight)

                x_packed = _pack_active_nodes_4d_from_indices(x, active_indices)
                residual_packed = _pack_active_nodes_4d_from_indices(residual, active_indices)

                x_packed = self.dropout(x_packed)
                x_packed = x_packed + residual_packed

                x = _unpack_active_nodes_4d(
                    x_packed,
                    active_indices,
                    batch_size=B,
                    timesteps=T,
                    num_nodes=N,
                    channels=self.hidden_channels,
                )
            else:
                # Normalization
                if self.use_adaptive_norm:
                    if cond is None:
                        raise ValueError("cond must be provided when use_adaptive_norm=True")
                    x = self.norm(x, cond)
                else:
                    x = self.norm(x)

                # Activation
                x = self.activation(x)

                # Graph convolution
                x = self.conv(x, edge_index, edge_weight)

                # Dropout
                x = self.dropout(x)
        else:
            # Post-activation: Conv → Norm → Activation → Dropout
            # More intuitive: conv produces logits, then normalize and activate

            # Graph convolution
            x = self.conv(x, edge_index, edge_weight)

            if can_pack_pointwise:
                B, T, N, _ = x.shape
                if precomputed_active_indices is not None:
                    active_indices = precomputed_active_indices
                    x_packed = _pack_active_nodes_4d_from_indices(x, active_indices)
                else:
                    x_packed, active_indices = _pack_active_nodes_4d(x, active_mask)
                residual_packed = _pack_active_nodes_4d_from_indices(residual, active_indices)

                x_packed = self.norm(x_packed)
                x_packed = self.activation(x_packed)
                x_packed = self.dropout(x_packed)
                x_packed = x_packed + residual_packed

                if return_packed_post:
                    # Packed epilogue for per-layer temporal mixer fast path.
                    return x_packed, active_indices

                x = _unpack_active_nodes_4d(
                    x_packed,
                    active_indices,
                    batch_size=B,
                    timesteps=T,
                    num_nodes=N,
                    channels=self.hidden_channels,
                )
            else:
                # Normalization
                if self.use_adaptive_norm:
                    if cond is None:
                        raise ValueError("cond must be provided when use_adaptive_norm=True")
                    x = self.norm(x, cond)
                else:
                    x = self.norm(x)

                # Activation
                x = self.activation(x)

                # Dropout
                x = self.dropout(x)

                # Residual connection
                x = x + residual

        if self.use_pre_activation and not can_pack_pointwise:
            # Residual connection for dense pre-activation path.
            x = x + residual
        
        return x
    
    def extra_repr(self) -> str:
        return (f"hidden_channels={self.hidden_channels}, K={self.K}, "
                f"adaptive={self.use_adaptive_norm}, strided={self.use_strided}")


class _TemporalSelfAttention(nn.Module):
    """
    Multi-head self-attention over the temporal dimension.

    Operates on (B*, T, C) tensors — each "stream" (node in the graph) attends
    independently over its T timesteps.  Always **bidirectional** (non-causal):
    the diffusion denoiser receives the full noisy future window at once, so
    every timestep should attend to every other timestep.

    Uses pre-norm (LayerNorm before attention) for training stability, and has
    its own residual connection: ``x + out_proj(attention(norm(x)))``.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        max_timesteps: Optional[int] = None,
    ):
        super().__init__()
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}")
        if channels % num_heads != 0:
            raise ValueError(
                f"channels ({channels}) must be divisible by num_heads ({num_heads}) "
                "for temporal self-attention."
            )
        if not (0.0 <= dropout <= 1.0):
            raise ValueError(f"dropout must be in [0, 1], got {dropout}")
        if max_timesteps is not None and max_timesteps < 1:
            raise ValueError(f"max_timesteps must be >= 1, got {max_timesteps}")

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.dropout = float(dropout)

        self.norm = nn.LayerNorm(channels)
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels)

        if max_timesteps is not None:
            self.pos_emb = nn.Parameter(torch.zeros(1, max_timesteps, channels))
            nn.init.trunc_normal_(self.pos_emb, std=0.02)
        else:
            self.pos_emb = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B*, T, C)
        Returns:
            (B*, T, C) with residual: x + attention(norm(x))
        """
        residual = x
        x = self.norm(x)  # (B*, T, C)

        if self.pos_emb is not None:
            T = x.size(-2)
            if T > self.pos_emb.size(1):
                raise ValueError(
                    f"Input T={T} exceeds max_timesteps={self.pos_emb.size(1)}. "
                    "Set attention.max_timesteps >= dataset.future_window."
                )
            x = x + self.pos_emb[:, :T, :]

        # Q/K/V projections → multi-head: (B*, T, C) → (B*, num_heads, T, head_dim)
        def _to_heads(z: torch.Tensor) -> torch.Tensor:
            return z.unflatten(-1, (self.num_heads, self.head_dim)).transpose(-3, -2)

        q = _to_heads(self.q_proj(x))
        k = _to_heads(self.k_proj(x))
        v = _to_heads(self.v_proj(x))

        attn_dropout = self.dropout if self.training else 0.0
        try:
            attn = F.scaled_dot_product_attention(
                q, k, v, dropout_p=attn_dropout, is_causal=False,
            )
        except RuntimeError:
            # Fallback for GPU backends that fail on edge-case layouts/sizes.
            scale = 1.0 / (self.head_dim ** 0.5)
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn = torch.softmax(scores, dim=-1)
            if attn_dropout > 0.0:
                attn = F.dropout(attn, p=attn_dropout, training=self.training)
            attn = torch.matmul(attn, v)

        # Merge heads: (B*, num_heads, T, head_dim) → (B*, T, C)
        attn = attn.transpose(-3, -2).flatten(-2)
        attn = self.out_proj(attn)

        return residual + attn


class TemporalDepthwiseMixer(nn.Module):
    """
    Temporal mixing block with depthwise Conv1d + norm + activation + dropout + residual.

    This block mixes information across timesteps while keeping node identities separate.
    It supports both:
    - 4D inputs: (B, T, N, C)
    - 3D inputs: (B*N, T, C)
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        causal: bool = False,
        dropout: float = 0.0,
        activation: Literal['relu', 'silu', 'gelu', 'leaky_relu'] = 'silu',
        use_pointwise: bool = True,
        norm_type: Literal['layer', 'group', 'instance', 'none'] = 'layer',
        num_groups: int = 8,
        gated: bool = False,
        attention_enabled: bool = False,
        attention_num_heads: int = 4,
        attention_dropout: float = 0.0,
        attention_max_timesteps: Optional[int] = None,
    ):
        super().__init__()
        if kernel_size < 1:
            raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
        if dilation < 1:
            raise ValueError(f"dilation must be >= 1, got {dilation}")

        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.causal = causal
        self.use_pointwise = use_pointwise
        self.norm_type = norm_type
        self.gated = gated

        # When gated, the depthwise conv outputs 2*channels (filter + gate),
        # then applies tanh(filter) * sigmoid(gate) — the WaveNet/DiffWave
        # gated activation pattern.  A 1x1 conv projects back to channels.
        out_channels = 2 * channels if gated else channels
        padding = 0 if causal else dilation * (kernel_size // 2)
        self.depthwise = nn.Conv1d(
            in_channels=channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=channels,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        if gated:
            # Project 2*channels back to channels after gated activation.
            self.gate_proj = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=False) if use_pointwise else nn.Identity()

        if norm_type == 'layer':
            # Layer-style normalization across channels for each timestep.
            self.norm = nn.GroupNorm(num_groups=1, num_channels=channels)
        elif norm_type == 'group':
            groups = min(num_groups, channels)
            while channels % groups != 0 and groups > 1:
                groups -= 1
            self.norm = nn.GroupNorm(num_groups=groups, num_channels=channels)
        elif norm_type == 'instance':
            self.norm = nn.InstanceNorm1d(channels, affine=True)
        elif norm_type == 'none':
            self.norm = nn.Identity()
        else:
            raise ValueError(
                f"Unknown temporal mixer norm_type: {norm_type}. "
                "Expected one of: 'layer', 'group', 'instance', 'none'."
            )

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if attention_enabled:
            self.self_attention = _TemporalSelfAttention(
                channels=channels,
                num_heads=attention_num_heads,
                dropout=attention_dropout,
                max_timesteps=attention_max_timesteps,
            )
            if causal:
                import warnings
                warnings.warn(
                    "TemporalDepthwiseMixer: causal=True applies to conv only; "
                    "self-attention is always bidirectional (correct for diffusion denoising).",
                    stacklevel=2,
                )
        else:
            self.self_attention = None

    @staticmethod
    def _match_temporal_length(x: torch.Tensor, target_t: int) -> torch.Tensor:
        """Crop/pad temporal axis to exactly target_t."""
        current_t = x.size(-1)
        if current_t == target_t:
            return x
        if current_t > target_t:
            return x[..., :target_t]
        return torch.nn.functional.pad(x, (0, target_t - current_t))

    def _mix_sequence(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Mix a 1D temporal sequence tensor.

        Args:
            x_seq: (B*, C, T)
        Returns:
            (B*, C, T)
        """
        target_t = x_seq.size(-1)
        if self.causal and self.kernel_size > 1:
            # Left-pad only: output at t depends on <= t.
            x_seq = torch.nn.functional.pad(
                x_seq,
                ((self.kernel_size - 1) * self.dilation, 0),
            )
        x_seq = self.depthwise(x_seq)
        x_seq = self._match_temporal_length(x_seq, target_t)
        if self.gated:
            # Depthwise grouped Conv1d emits channels ordered by group:
            # [g0_o0, g0_o1, g1_o0, g1_o1, ...]. Pair filter/gate within
            # each group instead of splitting contiguous halves.
            x_seq = x_seq.reshape(x_seq.size(0), self.channels, 2, x_seq.size(-1))
            x_filter = x_seq[:, :, 0, :]
            x_gate = x_seq[:, :, 1, :]
            x_seq = torch.tanh(x_filter) * torch.sigmoid(x_gate)
            x_seq = self.gate_proj(x_seq)
        x_seq = self.pointwise(x_seq)
        x_seq = self.norm(x_seq)
        if not self.gated:
            # Preserve legacy non-gated ordering: pointwise -> norm -> activation.
            x_seq = self.activation(x_seq)
        x_seq = self.dropout(x_seq)

        # Optional self-attention over temporal dimension.
        if self.self_attention is not None and x_seq.size(-1) > 1:
            # (B*, C, T) -> (B*, T, C) for attention, then back.
            x_seq = self.self_attention(x_seq.transpose(-2, -1)).transpose(-2, -1)

        return x_seq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        if x.dim() == 4:
            # (B, T, N, C) -> (B*N, C, T)
            B, T, N, C = x.shape
            x_seq = x.permute(0, 2, 3, 1).reshape(B * N, C, T)
            x_seq = self._mix_sequence(x_seq)
            x = x_seq.reshape(B, N, C, T).permute(0, 3, 1, 2)
        elif x.dim() == 3:
            # (B*N, T, C) -> (B*N, C, T)
            BN, T, C = x.shape
            x_seq = x.permute(0, 2, 1)
            x_seq = self._mix_sequence(x_seq)
            x = x_seq.permute(0, 2, 1).reshape(BN, T, C)
        else:
            raise ValueError(f"TemporalDepthwiseMixer expects 3D/4D input, got {x.dim()}D")

        return x + residual


class GNN(nn.Module):
    """
    Complete GNN architecture with multiple TAGConv layers.
    
    Stacks multiple ResidualGNNBlocks to create a deep GNN.
    Supports both standard and adaptive (time-conditioned) normalization.
    Can use either regular TAGConv or StridedTAGConv for multi-scale features.
    If `active_mask` is provided in forward, it is re-applied after each
    residual block to prevent inactive-node states from propagating across depth.
    
    Architecture:
        Input -> [ResidualGNNBlock -> (optional TemporalDepthwiseMixer)] x num_layers -> Output
    
    Args:
        in_channels: Input feature dimension
        hidden_channels: Hidden feature dimension
        out_channels: Output feature dimension
        num_layers: Number of GNN layers (default: 4)
        K: Number of TAGConv hops per layer (default: 3)
        norm_type: Type of normalization (default: 'group')
        num_groups: Number of groups for GroupNorm (default: 8)
        dropout: Dropout rate (default: 0.0)
        activation: Activation function (default: 'silu')
        use_adaptive_norm: Whether to use adaptive normalization (default: False)
        cond_dim: Conditioning dimension for adaptive norm (default: None)
        use_pre_activation: Whether to use pre-activation ordering in residual blocks (default: False)
        use_input_proj: Whether to project input to hidden_channels (default: True)
        use_output_proj: Whether to project output to out_channels (default: True)
        use_temporal_mixer: Whether to add temporal depthwise mixing after each GNN block (default: False)
        temporal_kernel_size: Kernel size for temporal depthwise Conv1d (default: 3)
        temporal_causal: Whether temporal convolution is causal (default: False)
        temporal_use_pointwise: Whether to add 1x1 pointwise conv after depthwise temporal conv (default: True)
        use_strided: Whether to use StridedTAGConv (default: False)
        gamma: Stride parameter for StridedTAGConv (default: 2)
        use_boolean_semiring: For StridedTAGConv, use exact γ-hop (default: False)
        strided_power_mode: Power-base behavior for StridedTAGConv when use_boolean_semiring=False.
            'a_plus_i' (default) or 'exact_walk'.
    
    Shape:
        - Input: (B, T, N, in_channels) or (B*N, T, in_channels)
        - Edge index: (2, E) - batched by PyG for B graphs
        - Conditioning (optional): (B, cond_dim) or (B, N, cond_dim)
        - Active mask (optional): (B, N) for 4D input, (B*N,) for 3D input
        - Output: (B, T, N, out_channels) or (B*N, T, out_channels)
    
    Examples:
        >>> # Basic GNN with regular TAGConv
        >>> gnn = GNN(
        ...     in_channels=32,
        ...     hidden_channels=64,
        ...     out_channels=32,
        ...     num_layers=4,
        ...     K=3,
        ... )
        >>> x = torch.randn(4, 10, 100, 32)
        >>> edge_index = torch.randint(0, 400, (2, 2000))
        >>> out = gnn(x, edge_index)  # (4, 10, 100, 32)
        
        >>> # GNN with StridedTAGConv for multi-scale features
        >>> gnn = GNN(
        ...     in_channels=32,
        ...     hidden_channels=64,
        ...     out_channels=32,
        ...     num_layers=4,
        ...     K=2,
        ...     use_strided=True,
        ...     gamma=2,
        ... )
        >>> out = gnn(x, edge_index)  # Aggregates 0, 2, 4-hop at each layer
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 4,
        K: int = 3,
        norm_type: Literal['layer', 'group', 'instance', 'none'] = 'group',
        num_groups: int = 8,
        dropout: float = 0.0,
        activation: Literal['relu', 'silu', 'gelu', 'leaky_relu'] = 'silu',
        use_adaptive_norm: bool = False,
        cond_dim: Optional[int] = None,
        use_pre_activation: bool = False,
        use_input_proj: bool = True,
        use_output_proj: bool = True,
        use_temporal_mixer: bool = False,
        temporal_mixer_schedule: Literal['per_layer', 'per_block', 'off'] = 'per_layer',
        temporal_kernel_size: int = 3,
        temporal_causal: bool = False,
        temporal_use_pointwise: bool = True,
        temporal_dilations: Optional[List[int]] = None,
        temporal_gated: bool = False,
        temporal_attention_enabled: bool = False,
        temporal_attention_num_heads: int = 4,
        temporal_attention_dropout: float = 0.0,
        temporal_attention_max_timesteps: Optional[int] = None,
        normalize: bool = False,  # Whether to apply symmetric normalization in TAGConv
        use_strided: bool = False,
        gamma: int = 2,
        use_boolean_semiring: bool = False,
        strided_power_mode: Literal['a_plus_i', 'exact_walk'] = 'a_plus_i',
        pointwise_sparse_mode: Literal['off', 'auto', 'on'] = 'auto',
        pointwise_sparse_threshold: float = 0.75,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.use_adaptive_norm = use_adaptive_norm
        self.normalize = normalize
        self.use_temporal_mixer = use_temporal_mixer
        self.temporal_mixer_schedule = temporal_mixer_schedule
        self.temporal_kernel_size = temporal_kernel_size
        self.temporal_causal = temporal_causal
        self.use_strided = use_strided
        self.norm_type = norm_type
        self.pointwise_sparse_mode = pointwise_sparse_mode
        self.pointwise_sparse_threshold = float(pointwise_sparse_threshold)
        self._last_pointwise_sparse_used: bool = False
        self._last_pointwise_sparse_active_ratio: float = 1.0
        self._last_packed_layer_mixer_path_used: bool = False

        if self.temporal_mixer_schedule not in {'per_layer', 'per_block', 'off'}:
            raise ValueError(
                f"Unknown temporal_mixer_schedule: {self.temporal_mixer_schedule}. "
                "Expected one of: 'per_layer', 'per_block', 'off'."
            )
        if self.pointwise_sparse_mode not in {'off', 'auto', 'on'}:
            raise ValueError(
                f"Unknown pointwise_sparse_mode: {self.pointwise_sparse_mode}. "
                "Expected one of: 'off', 'auto', 'on'."
            )
        if not (0.0 <= self.pointwise_sparse_threshold <= 1.0):
            raise ValueError(
                f"pointwise_sparse_threshold must be in [0, 1], got {self.pointwise_sparse_threshold}."
            )

        # Backward compatibility:
        # - use_temporal_mixer=False always disables temporal mixing.
        # - use_temporal_mixer=True keeps schedule-driven behavior (default per_layer).
        if not self.use_temporal_mixer:
            self.temporal_mixer_schedule = 'off'

        if norm_type == 'graph':
            raise ValueError(
                "norm_type='graph' is not supported in GNN. "
                "Use one of: 'layer', 'group', 'instance', 'none'."
            )
        
        # Input projection
        if use_input_proj:
            self.input_proj = nn.Linear(in_channels, hidden_channels)
        else:
            if in_channels != hidden_channels:
                raise ValueError(
                    f"in_channels ({in_channels}) must equal hidden_channels "
                    f"({hidden_channels}) when use_input_proj=False"
                )
            self.input_proj = nn.Identity()
        
        # GNN layers
        self.layers = nn.ModuleList([
            ResidualGNNBlock(
                hidden_channels=hidden_channels,
                K=K,
                norm_type=norm_type,
                num_groups=num_groups,
                dropout=dropout,
                activation=activation,
                use_adaptive_norm=use_adaptive_norm,
                cond_dim=cond_dim,
                use_pre_activation=use_pre_activation,
                normalize=normalize,
                use_strided=use_strided,
                gamma=gamma,
                use_boolean_semiring=use_boolean_semiring,
                strided_power_mode=strided_power_mode,
            )
            for _ in range(num_layers)
        ])

        if self.temporal_mixer_schedule == 'off':
            self.temporal_mixers = None
            self.block_temporal_mixer = None
        else:
            if temporal_dilations is None:
                if self.temporal_mixer_schedule == 'per_layer':
                    temporal_dilations = [1] * num_layers
                else:
                    temporal_dilations = [1]

            if self.temporal_mixer_schedule == 'per_layer':
                if len(temporal_dilations) != num_layers:
                    raise ValueError(
                        f"temporal_dilations length ({len(temporal_dilations)}) must match "
                        f"num_layers ({num_layers}) when temporal_mixer_schedule='per_layer'."
                    )
                if any(d < 1 for d in temporal_dilations):
                    raise ValueError("All temporal_dilations must be >= 1")

                self.temporal_mixers = nn.ModuleList([
                    TemporalDepthwiseMixer(
                        channels=hidden_channels,
                        kernel_size=temporal_kernel_size,
                        dilation=dilation,
                        causal=temporal_causal,
                        dropout=dropout,
                        activation=activation,
                        use_pointwise=temporal_use_pointwise,
                        norm_type=norm_type,
                        num_groups=num_groups,
                        gated=temporal_gated,
                        attention_enabled=temporal_attention_enabled,
                        attention_num_heads=temporal_attention_num_heads,
                        attention_dropout=temporal_attention_dropout,
                        attention_max_timesteps=temporal_attention_max_timesteps,
                    )
                    for dilation in temporal_dilations
                ])
                self.block_temporal_mixer = None
            else:
                if len(temporal_dilations) != 1:
                    raise ValueError(
                        f"temporal_dilations length ({len(temporal_dilations)}) must be 1 "
                        "when temporal_mixer_schedule='per_block'."
                    )
                dilation = temporal_dilations[0]
                if dilation < 1:
                    raise ValueError("All temporal_dilations must be >= 1")

                self.temporal_mixers = None
                self.block_temporal_mixer = TemporalDepthwiseMixer(
                    channels=hidden_channels,
                    kernel_size=temporal_kernel_size,
                    dilation=dilation,
                    causal=temporal_causal,
                    dropout=dropout,
                    activation=activation,
                    use_pointwise=temporal_use_pointwise,
                    norm_type=norm_type,
                    num_groups=num_groups,
                    gated=temporal_gated,
                    attention_enabled=temporal_attention_enabled,
                    attention_num_heads=temporal_attention_num_heads,
                    attention_dropout=temporal_attention_dropout,
                    attention_max_timesteps=temporal_attention_max_timesteps,
                )

        # Output projection
        if use_output_proj:
            self.output_proj = nn.Linear(hidden_channels, out_channels)
        else:
            if hidden_channels != out_channels:
                raise ValueError(
                    f"hidden_channels ({hidden_channels}) must equal out_channels "
                    f"({out_channels}) when use_output_proj=False"
                )
            self.output_proj = nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        active_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through GNN.
        
        Args:
            x: Node features of shape (B, T, N, F_in) or (B*N, T, F_in)
            edge_index: Edge indices of shape (2, E) - batched by PyG
            edge_weight: Optional edge weights of shape (E,)
            cond: Optional conditioning of shape (B, cond_dim) or (B, N, cond_dim)
            active_mask: Optional active-node mask.
                - For 4D inputs: shape (B, N)
                - For 3D inputs: shape (B*N,) or (B*N, 1)
        
        Returns:
            Output features of shape (B, T, N, F_out) or (B*N, T, F_out)
        """
        use_packed_pointwise = self._should_use_packed_pointwise(x, active_mask)
        active_indices = self._resolve_active_indices(
            x=x,
            active_mask=active_mask,
            use_packed_pointwise=use_packed_pointwise,
        )
        self._last_packed_layer_mixer_path_used = False

        # Input projection
        x = self._apply_linear_pointwise(
            x,
            self.input_proj,
            active_mask=active_mask,
            use_packed_pointwise=use_packed_pointwise,
            precomputed_active_indices=active_indices,
        )
        
        # Apply GNN layers
        for layer_idx, layer in enumerate(self.layers):
            packed_layer_mixer_fast_path = (
                use_packed_pointwise
                and active_indices is not None
                and self.temporal_mixer_schedule == 'per_layer'
                and self.temporal_mixers is not None
                and not layer.use_pre_activation
            )

            layer_out = layer(
                x,
                edge_index,
                edge_weight,
                cond,
                active_mask=active_mask,
                use_packed_pointwise=use_packed_pointwise,
                precomputed_active_indices=active_indices,
                return_packed_post=packed_layer_mixer_fast_path,
            )

            if packed_layer_mixer_fast_path and isinstance(layer_out, tuple):
                x_packed, packed_indices = layer_out
                mixed_packed = self.temporal_mixers[layer_idx](x_packed)
                B, T, N, _ = x.shape
                x = _unpack_active_nodes_4d(
                    mixed_packed,
                    packed_indices,
                    batch_size=B,
                    timesteps=T,
                    num_nodes=N,
                    channels=self.hidden_channels,
                )
                x = self._apply_active_mask(x, active_mask)
                self._last_packed_layer_mixer_path_used = True
                continue

            x = layer_out if isinstance(layer_out, torch.Tensor) else layer_out[0]
            x = self._apply_active_mask(x, active_mask)

            if self.temporal_mixer_schedule == 'per_layer' and self.temporal_mixers is not None:
                x = self._apply_temporal_mixer_pointwise(
                    x,
                    self.temporal_mixers[layer_idx],
                    active_mask=active_mask,
                    use_packed_pointwise=use_packed_pointwise,
                    precomputed_active_indices=active_indices,
                )
                x = self._apply_active_mask(x, active_mask)

        if self.temporal_mixer_schedule == 'per_block' and self.block_temporal_mixer is not None:
            x = self._apply_temporal_mixer_pointwise(
                x,
                self.block_temporal_mixer,
                active_mask=active_mask,
                use_packed_pointwise=use_packed_pointwise,
                precomputed_active_indices=active_indices,
            )
            x = self._apply_active_mask(x, active_mask)
        
        # Output projection
        x = self._apply_linear_pointwise(
            x,
            self.output_proj,
            active_mask=active_mask,
            use_packed_pointwise=use_packed_pointwise,
            precomputed_active_indices=active_indices,
        )
        
        return x

    def should_use_packed_pointwise(
        self,
        x: torch.Tensor,
        active_mask: Optional[torch.Tensor],
    ) -> bool:
        """
        Public routing helper reused by UGNN encoder/decoder pre-GNN fusion.
        Keeps sparse routing logic centralized in GNN.
        """
        return self._should_use_packed_pointwise(x, active_mask)

    @staticmethod
    def _resolve_active_indices(
        *,
        x: torch.Tensor,
        active_mask: Optional[torch.Tensor],
        use_packed_pointwise: bool,
    ) -> Optional[torch.Tensor]:
        if (
            not use_packed_pointwise
            or active_mask is None
            or x.dim() != 4
            or active_mask.dim() != 2
        ):
            return None
        active_flat = active_mask.to(dtype=torch.bool).reshape(-1)
        return torch.nonzero(active_flat, as_tuple=False).squeeze(-1)

    def _should_use_packed_pointwise(
        self,
        x: torch.Tensor,
        active_mask: Optional[torch.Tensor],
    ) -> bool:
        """Resolve off/auto/on sparse-pointwise policy per GNN invocation."""
        if self.pointwise_sparse_mode == 'off':
            self._last_pointwise_sparse_used = False
            self._last_pointwise_sparse_active_ratio = 1.0
            return False
        if active_mask is None or x.dim() != 4 or active_mask.dim() != 2:
            self._last_pointwise_sparse_used = False
            self._last_pointwise_sparse_active_ratio = 1.0
            return False
        if self.use_adaptive_norm:
            # Conservative Phase-1 behavior: packed adaptive norm is deferred.
            # Co-packing conditioning can relax this in a future iteration.
            self._last_pointwise_sparse_used = False
            self._last_pointwise_sparse_active_ratio = float(active_mask.float().mean().item())
            return False
        if self.norm_type == 'instance':
            self._last_pointwise_sparse_used = False
            self._last_pointwise_sparse_active_ratio = float(active_mask.float().mean().item())
            return False
        # Note: norm_type='graph' is already rejected in __init__.

        active_ratio = float(active_mask.float().mean().item())
        self._last_pointwise_sparse_active_ratio = active_ratio
        if self.pointwise_sparse_mode == 'on':
            self._last_pointwise_sparse_used = True
            return True

        use_sparse = active_ratio < self.pointwise_sparse_threshold
        self._last_pointwise_sparse_used = bool(use_sparse)
        return bool(use_sparse)

    @staticmethod
    def _apply_linear_pointwise(
        x: torch.Tensor,
        proj: nn.Module,
        *,
        active_mask: Optional[torch.Tensor],
        use_packed_pointwise: bool,
        precomputed_active_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply a linear pointwise projection with optional active-node packing."""
        if not use_packed_pointwise or active_mask is None or x.dim() != 4 or active_mask.dim() != 2:
            return proj(x)
        if not isinstance(proj, nn.Linear):
            return proj(x)

        B, T, N, _ = x.shape
        if precomputed_active_indices is not None:
            active_indices = precomputed_active_indices
            packed = _pack_active_nodes_4d_from_indices(x, active_indices)
        else:
            packed, active_indices = _pack_active_nodes_4d(x, active_mask)
        packed_out = proj(packed)
        out = _unpack_active_nodes_4d(
            packed_out,
            active_indices,
            batch_size=B,
            timesteps=T,
            num_nodes=N,
            channels=proj.out_features,
        )

        # Dense baseline on zeroed inactive rows yields bias; preserve that.
        if proj.bias is not None:
            inactive = ~active_mask.to(dtype=torch.bool, device=out.device)
            if bool(inactive.any()):
                bias = proj.bias.to(dtype=out.dtype, device=out.device).view(1, 1, 1, -1)
                out = out + inactive.unsqueeze(1).unsqueeze(-1).to(dtype=out.dtype) * bias
        return out

    @staticmethod
    def _apply_temporal_mixer_pointwise(
        x: torch.Tensor,
        mixer: nn.Module,
        *,
        active_mask: Optional[torch.Tensor],
        use_packed_pointwise: bool,
        precomputed_active_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply temporal mixer with optional active-node packing."""
        if not use_packed_pointwise or active_mask is None or x.dim() != 4 or active_mask.dim() != 2:
            return mixer(x)

        B, T, N, C = x.shape
        if precomputed_active_indices is not None:
            active_indices = precomputed_active_indices
            packed = _pack_active_nodes_4d_from_indices(x, active_indices)
        else:
            packed, active_indices = _pack_active_nodes_4d(x, active_mask)
        mixed_packed = mixer(packed)
        return _unpack_active_nodes_4d(
            mixed_packed,
            active_indices,
            batch_size=B,
            timesteps=T,
            num_nodes=N,
            channels=C,
        )

    @staticmethod
    def _apply_active_mask(x: torch.Tensor, active_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Apply active-node mask to rank-4 (B,T,N,C) or rank-3 (B*N,T,C) tensors."""
        if active_mask is None:
            return x

        if x.dim() == 4:
            if active_mask.dim() != 2:
                raise ValueError(
                    f"For 4D input, active_mask must have shape (B, N), got {tuple(active_mask.shape)}"
                )
            if active_mask.size(0) != x.size(0) or active_mask.size(1) != x.size(2):
                raise ValueError(
                    "For 4D input, active_mask shape must match (B, N) = "
                    f"({x.size(0)}, {x.size(2)}), got {tuple(active_mask.shape)}"
                )
            mask = active_mask.unsqueeze(1).unsqueeze(-1).to(dtype=x.dtype, device=x.device)
            return x * mask

        if x.dim() == 3:
            if active_mask.dim() == 2 and active_mask.size(1) == 1:
                mask_bn = active_mask.squeeze(-1)
            elif active_mask.dim() == 1:
                mask_bn = active_mask
            else:
                raise ValueError(
                    f"For 3D input, active_mask must have shape (B*N,) or (B*N, 1), got {tuple(active_mask.shape)}"
                )
            if mask_bn.size(0) != x.size(0):
                raise ValueError(
                    f"active_mask length ({mask_bn.size(0)}) must match x.size(0) ({x.size(0)}) for 3D input"
                )
            mask = mask_bn.view(-1, 1, 1).to(dtype=x.dtype, device=x.device)
            return x * mask

        raise ValueError(f"Unsupported input rank for masking: x.dim()={x.dim()}")
    
    def extra_repr(self) -> str:
        return (f"in_channels={self.in_channels}, "
                f"hidden_channels={self.hidden_channels}, "
                f"out_channels={self.out_channels}, "
                f"num_layers={self.num_layers}, "
                f"adaptive={self.use_adaptive_norm}, "
                f"strided={self.use_strided}, "
                f"temporal_mixer={self.use_temporal_mixer}, "
                f"temporal_schedule={self.temporal_mixer_schedule}, "
                f"pointwise_sparse_mode={self.pointwise_sparse_mode}, "
                f"pointwise_sparse_threshold={self.pointwise_sparse_threshold}")



class StridedTAGConv(nn.Module):
    """
    Strided Topology Adaptive Graph Convolution (StridedTAGConv).
    
    Instead of using the standard adjacency matrix A for message passing,
    this variant uses A^γ (the γ-hop adjacency) as the base matrix, then
    applies K iterations of message passing with A^γ.
    
    Mathematical formulation:
        Standard TAGConv:  h = Σ_{k=0}^K W_k · A^k · x
        StridedTAGConv:    h = Σ_{k=0}^K W_k · (A^γ)^k · x
    
    Where:
        - A is the (normalized) adjacency matrix
        - A^γ is the γ-hop adjacency (computed once)
        - K is the number of message passing iterations
        - W_k are learnable weight matrices for each scale
    
    Implementation:
        1. Compute A^γ (γ-hop adjacency) once
        2. Use A^γ as base adjacency for iterative message passing
        3. At iteration k: aggregate from (A^γ)^k neighborhood
        4. Combine: h = Σ_{k=0}^K W_k · h_k
    
    This is exactly how regular TAGConv works, except using A^γ instead of A.
    
    Key differences from TAGConv:
        - Standard TAGConv with K=3: aggregates via A^0, A^1, A^2, A^3
        - StridedTAGConv with γ=2, K=3: aggregates via (A^2)^0, (A^2)^1, (A^2)^2, (A^2)^3
          = A^0, A^2, A^4, A^6
    
    Benefits:
        - Multi-resolution graph analysis (coarse-to-fine features)
        - Computational efficiency (fewer intermediate hops)
        - Hierarchical graph processing (matching strided pooling layers)
        - Capturing long-range dependencies with fewer layers
    
    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension
        gamma: Stride parameter (default: 2). Hop distance for base adjacency A^γ.
        K: Number of message passing iterations (default: 2)
        bias: Whether to use bias in linear transformations (default: True)
        normalize: Whether to apply symmetric normalization to adjacency (default: True)
        use_boolean_semiring: Whether to use exact γ-hop (True) or all paths
            up to γ hops (False) for computing A^γ (default: False).
            - False (default): A^γ via matrix power, includes all paths ≤ γ hops
            - True: Exact γ-hop via Boolean BFS, only nodes exactly γ hops away
        power_mode: Base adjacency mode for non-boolean power path (default: 'a_plus_i')
            - 'a_plus_i': A_base = A + I for gamma > 1 (historical behavior)
            - 'exact_walk': A_base = A (exact walk-length powers)
    
    Shape:
        - Input: (B, T, N, F_in) or (B*N, T, F_in)
        - Edge index: (2, E) - already batched for B graphs by PyG
        - Output: (B, T, N, F_out) or (B*N, T, F_out)
    
    Examples:
        >>> # Standard usage with γ=2, K=2
        >>> layer = StridedTAGConv(in_channels=64, out_channels=64, gamma=2, K=2)
        >>> x = torch.randn(4, 10, 100, 64)
        >>> edge_index = torch.randint(0, 400, (2, 2000))
        >>> out = layer(x, edge_index)  # Aggregates via A^0, A^2, A^4
        
        >>> # Larger stride for long-range dependencies
        >>> layer = StridedTAGConv(in_channels=64, out_channels=128, gamma=3, K=3)
        >>> # Aggregates via A^0, A^3, A^6, A^9
        
        >>> # γ=1 is equivalent to standard TAGConv
        >>> layer = StridedTAGConv(in_channels=64, out_channels=64, gamma=1, K=3)
        >>> # Aggregates via A^0, A^1, A^2, A^3 (same as TAGConv)
    
    References:
        - TAGConv: Du et al. "Topology Adaptive Graph Convolutional Networks" (2017)
        - Strided convolutions: He et al. "Deep Residual Learning" (2016)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        gamma: int = 2,
        K: int = 2,
        bias: bool = True,
        normalize: bool = True,
        use_boolean_semiring: bool = False,
        power_mode: Literal['a_plus_i', 'exact_walk'] = 'a_plus_i',
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gamma = gamma
        self.K = K
        self.normalize = normalize
        self.use_boolean_semiring = use_boolean_semiring
        if power_mode not in {'a_plus_i', 'exact_walk'}:
            raise ValueError(
                f"Unknown power_mode='{power_mode}'. Expected 'a_plus_i' or 'exact_walk'."
            )
        self.power_mode = power_mode
        
        # Learnable weight for each power k=0,1,...,K
        # Each corresponds to (A^γ)^k
        self.lins = nn.ModuleList([
            nn.Linear(in_channels, out_channels, bias=False)
            for _ in range(K + 1)
        ])
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        for lin in self.lins:
            nn.init.xavier_uniform_(lin.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def _compute_gamma_hop_adjacency_power(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        edge_weight: Optional[torch.Tensor] = None,
        return_edge_weight: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        Compute A^γ via matrix power (includes all paths up to γ hops).
        
        This is matrix multiplication A × A × ... × A (γ times).
        Includes all nodes reachable within γ hops (not just exactly γ).
        
        Args:
            edge_index: Edge indices (2, E)
            num_nodes: Number of nodes
            edge_weight: Optional scalar edge weights (E,)
        
        Returns:
            edge_index_gamma: Edge indices for A^γ (2, E_γ)
            edge_weight_gamma: Edge weights for A^γ (E_γ,) or None (if return_edge_weight=True)
        """
        if self.gamma == 1:
            # γ=1: return original adjacency and optionally weights
            if return_edge_weight:
                return edge_index, edge_weight
            return edge_index
        
        from torch_geometric.utils import to_dense_adj, dense_to_sparse
        
        device = edge_index.device
        
        # Convert to dense weighted adjacency.
        # If edge_weight is None, to_dense_adj builds a binary adjacency.
        adj = to_dense_adj(
            edge_index,
            edge_attr=edge_weight,
            max_num_nodes=num_nodes,
        )[0].float()

        # Historical behavior ('a_plus_i') uses A+I as base for gamma>1.
        # Exact-walk mode uses A as base for all gamma.
        if self.gamma > 1 and self.power_mode == 'a_plus_i':
            adj_base = adj + torch.eye(num_nodes, device=device, dtype=adj.dtype)
        else:
            adj_base = adj
        
        # Compute A^γ by repeated multiplication.
        adj_gamma = adj_base.clone()
        for _ in range(self.gamma - 1):
            adj_gamma = torch.matmul(adj_gamma, adj_base)
        
        # Convert back to sparse (indices + weights).
        edge_index_gamma, edge_weight_gamma = dense_to_sparse(adj_gamma)
        
        if return_edge_weight:
            return edge_index_gamma, edge_weight_gamma
        return edge_index_gamma
    
    def _compute_gamma_hop_adjacency_boolean(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        edge_weight: Optional[torch.Tensor] = None,
        return_edge_weight: bool = False,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        Compute exact γ-hop adjacency via Boolean semiring BFS.
        
        Only includes nodes that are **exactly** γ hops away (shortest path),
        not closer. This is more selective than matrix power.
        
        Args:
            edge_index: Edge indices (2, E)
            num_nodes: Number of nodes
            edge_weight: Optional scalar edge weights (E,)
        
        Returns:
            edge_index_gamma: Edge indices for exact γ-hop neighbors (2, E_γ)
            edge_weight_gamma: Edge weights for exact γ-hop neighbors (E_γ,) or None
                (if return_edge_weight=True)
        """
        if self.gamma == 1:
            if return_edge_weight:
                return edge_index, edge_weight
            return edge_index
        
        from torch_geometric.utils import to_dense_adj, dense_to_sparse
        
        device = edge_index.device
        
        # Weighted adjacency for path-strength accumulation.
        # If edge_weight is None, fall back to binary adjacency.
        adj_weighted = to_dense_adj(
            edge_index,
            edge_attr=edge_weight,
            max_num_nodes=num_nodes,
        )[0].float()

        # Boolean adjacency for exact-hop BFS.
        adj = adj_weighted != 0
        
        # Boolean BFS for exact γ-hop distances
        current_layer = torch.eye(num_nodes, device=device, dtype=torch.bool)
        visited = current_layer.clone()
        
        for hop in range(self.gamma):
            # Matrix multiplication in Boolean semiring
            next_layer = torch.matmul(current_layer.float(), adj.float()) > 0
            # Remove visited nodes (shortest path only)
            next_layer = next_layer & ~visited
            visited = visited | next_layer
            current_layer = next_layer
        
        # current_layer contains exactly γ-hop neighbors
        adj_gamma_mask = current_layer

        # Compute weighted γ-hop path strengths, then keep exact γ-hop pairs only.
        adj_paths = adj_weighted.clone()
        for _ in range(self.gamma - 1):
            adj_paths = torch.matmul(adj_paths, adj_weighted)
        adj_gamma = adj_paths * adj_gamma_mask.float()
        
        # Convert back to sparse (indices + weights)
        edge_index_gamma, edge_weight_gamma = dense_to_sparse(adj_gamma)
        
        if return_edge_weight:
            return edge_index_gamma, edge_weight_gamma
        return edge_index_gamma

    def _build_power_base_adjacency(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Build sparse base adjacency used in power-mode propagation without densification.

        - power_mode='a_plus_i' (default): for gamma>1, uses A+I as base.
        - power_mode='exact_walk': always uses A as base.
        """
        if self.gamma == 1 or self.power_mode == 'exact_walk':
            return edge_index, edge_weight

        from torch_geometric.utils import add_self_loops

        if edge_weight is None:
            edge_index_base, _ = add_self_loops(edge_index, num_nodes=num_nodes)
            return edge_index_base, None

        edge_index_base, edge_weight_base = add_self_loops(
            edge_index,
            edge_weight,
            fill_value=1.0,
            num_nodes=num_nodes,
        )
        return edge_index_base, edge_weight_base

    @staticmethod
    def _propagate_once_sparse(
        x_state: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Single sparse propagation: x_next = A * x_state."""
        row, col = edge_index
        x_neighbors = x_state[row]
        if edge_weight is not None:
            x_neighbors = x_neighbors * edge_weight.view(-1, 1)
        return scatter_add(x_neighbors, col, dim=0, dim_size=x_state.size(0))

    def _forward_power_sparse_no_dense(
        self,
        x_seq: torch.Tensor,
        edge_index_base: torch.Tensor,
        edge_weight_base: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Memory-efficient power-mode forward without explicit A^gamma construction.

        x_seq has shape (T, num_nodes, F_in). This computes:
            out_t = sum_{k=0..K} W_k * ((A_base^gamma)^k * x_t)
        where A_base is chosen by power_mode:
            - 'a_plus_i': A for gamma=1, A+I for gamma>1
            - 'exact_walk': A for all gamma
        """
        T, num_nodes, _ = x_seq.shape
        out_seq = torch.zeros(
            T,
            num_nodes,
            self.out_channels,
            device=x_seq.device,
            dtype=x_seq.dtype,
        )

        for t in range(T):
            x_k = x_seq[t]
            out_t = torch.zeros(
                num_nodes,
                self.out_channels,
                device=x_seq.device,
                dtype=x_seq.dtype,
            )
            for k in range(self.K + 1):
                out_t = out_t + self.lins[k](x_k)
                if k < self.K:
                    for _ in range(self.gamma):
                        x_k = self._propagate_once_sparse(
                            x_state=x_k,
                            edge_index=edge_index_base,
                            edge_weight=edge_weight_base,
                        )
            if self.bias is not None:
                out_t = out_t + self.bias
            out_seq[t] = out_t

        return out_seq
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through StridedTAGConv.
        
        Computes: h = Σ_{k=0}^K W_k · (A^γ)^k · x
        
        Args:
            x: Node features of shape (B, T, N, F) or (B*N, T, F)
            edge_index: Edge indices of shape (2, E) - batched by PyG for B graphs
            edge_weight: Optional edge weights (E,) used to build A^γ and in aggregation
        
        Returns:
            Output features with same shape as input
        """
        is_4d = x.dim() == 4

        # Fast path: avoid explicit dense A^gamma materialization in common power-mode.
        # This significantly reduces peak memory for large batched graphs.
        use_sparse_power_path = (not self.use_boolean_semiring) and (not self.normalize)
        
        if is_4d:
            B, T, N, F = x.shape
            BN = B * N
            x_seq = x.permute(1, 0, 2, 3).reshape(T, BN, F)

            if use_sparse_power_path:
                edge_index_base, edge_weight_base = self._build_power_base_adjacency(
                    edge_index=edge_index,
                    num_nodes=BN,
                    edge_weight=edge_weight,
                )
                out_seq = self._forward_power_sparse_no_dense(
                    x_seq=x_seq,
                    edge_index_base=edge_index_base,
                    edge_weight_base=edge_weight_base,
                )
                return out_seq.reshape(T, B, N, self.out_channels).permute(1, 0, 2, 3)

            num_nodes_per_timestep = BN

            # Reshape to (T*B*N, F) for compatibility with legacy dense path.
            x_flat = x_seq.reshape(T * BN, F)
            
            # Step 1: Compute A^γ (γ-hop adjacency) once
            if self.use_boolean_semiring:
                edge_index_gamma, edge_weight_gamma = self._compute_gamma_hop_adjacency_boolean(
                    edge_index, BN, edge_weight, return_edge_weight=True
                )
            else:
                edge_index_gamma, edge_weight_gamma = self._compute_gamma_hop_adjacency_power(
                    edge_index, BN, edge_weight, return_edge_weight=True
                )
            
            # Step 2: Replicate A^γ for T timesteps
            edge_index_gamma_t = edge_index_gamma.unsqueeze(0).expand(T, -1, -1)
            offsets = torch.arange(T, device=edge_index.device) * num_nodes_per_timestep
            edge_index_gamma_t = edge_index_gamma_t + offsets.view(T, 1, 1)
            edge_index_gamma_t = edge_index_gamma_t.reshape(2, T * edge_index_gamma.size(1))
            edge_weight_gamma_t = edge_weight_gamma.repeat(T) if edge_weight_gamma is not None else None
            
            # Step 3: Normalize A^γ if requested
            if self.normalize:
                from torch_geometric.nn.conv.gcn_conv import gcn_norm
                edge_index_gamma_t, edge_weight_gamma = gcn_norm(
                    edge_index_gamma_t,
                    edge_weight_gamma_t,
                    num_nodes=T * BN,
                    add_self_loops=False,
                )
            else:
                edge_weight_gamma = edge_weight_gamma_t
            
            # Step 4: Iteratively apply message passing with A^γ
            # This gives us: x, (A^γ)x, (A^γ)^2 x, ..., (A^γ)^K x
            # NOTE: scatter_add imported at top of file with native fallback
            # from torch_scatter import scatter_add
            
            out = torch.zeros(T * BN, self.out_channels, device=x.device)
            
            x_k = x_flat  # Start with x^(0) = x
            
            for k in range(self.K + 1):
                # Apply learnable weight to current power
                out = out + self.lins[k](x_k)
                
                # Message passing: x^(k+1) = A^γ · x^(k)
                if k < self.K:  # Don't need to compute for last iteration
                    row, col = edge_index_gamma_t
                    # PyG convention: edge_index[0]=source (j), edge_index[1]=target (i)
                    # Aggregate messages from source features onto target nodes.
                    x_neighbors = x_k[row]  # (E, F)
                    
                    if edge_weight_gamma is not None:
                        x_neighbors = x_neighbors * edge_weight_gamma.view(-1, 1)
                    
                    # Aggregate: sum_{j ∈ N_γ(i)} x_j^(k)
                    x_k = scatter_add(x_neighbors, col, dim=0, dim_size=T * BN)
            
            # Add bias
            if self.bias is not None:
                out = out + self.bias
            
            # Reshape back to (B, T, N, F_out)
            out = out.reshape(T, B, N, self.out_channels).permute(1, 0, 2, 3)
            
        elif x.dim() == 3:
            # (B*N, T, F) -> process similarly
            BN, T, F = x.shape
            x_seq = x.permute(1, 0, 2).reshape(T, BN, F)

            if use_sparse_power_path:
                edge_index_base, edge_weight_base = self._build_power_base_adjacency(
                    edge_index=edge_index,
                    num_nodes=BN,
                    edge_weight=edge_weight,
                )
                out_seq = self._forward_power_sparse_no_dense(
                    x_seq=x_seq,
                    edge_index_base=edge_index_base,
                    edge_weight_base=edge_weight_base,
                )
                return out_seq.permute(1, 0, 2)
            
            # Reshape to (T*B*N, F)
            x_flat = x_seq.reshape(T * BN, F)
            
            # Compute A^γ
            if self.use_boolean_semiring:
                edge_index_gamma, edge_weight_gamma = self._compute_gamma_hop_adjacency_boolean(
                    edge_index, BN, edge_weight, return_edge_weight=True
                )
            else:
                edge_index_gamma, edge_weight_gamma = self._compute_gamma_hop_adjacency_power(
                    edge_index, BN, edge_weight, return_edge_weight=True
                )
            
            # Replicate for T timesteps
            edge_index_gamma_t = edge_index_gamma.unsqueeze(0).expand(T, -1, -1)
            offsets = torch.arange(T, device=edge_index.device) * BN
            edge_index_gamma_t = edge_index_gamma_t + offsets.view(T, 1, 1)
            edge_index_gamma_t = edge_index_gamma_t.reshape(2, T * edge_index_gamma.size(1))
            edge_weight_gamma_t = edge_weight_gamma.repeat(T) if edge_weight_gamma is not None else None
            
            # Normalize if requested
            if self.normalize:
                from torch_geometric.nn.conv.gcn_conv import gcn_norm
                edge_index_gamma_t, edge_weight_gamma = gcn_norm(
                    edge_index_gamma_t,
                    edge_weight_gamma_t,
                    num_nodes=T * BN,
                    add_self_loops=False,
                )
            else:
                edge_weight_gamma = edge_weight_gamma_t
            
            # Iterative message passing with A^γ
            out = torch.zeros(T * BN, self.out_channels, device=x.device)
            x_k = x_flat
            
            for k in range(self.K + 1):
                out = out + self.lins[k](x_k)
                
                if k < self.K:
                    row, col = edge_index_gamma_t
                    x_neighbors = x_k[row]
                    
                    if edge_weight_gamma is not None:
                        x_neighbors = x_neighbors * edge_weight_gamma.view(-1, 1)
                    
                    x_k = scatter_add(x_neighbors, col, dim=0, dim_size=T * BN)
            
            # Reshape back to (B*N, T, F_out)
            out = out.reshape(T, BN, self.out_channels).permute(1, 0, 2)
        
        else:
            raise ValueError(f"Expected 3D or 4D input, got {x.dim()}D")
        
        return out
    
    def extra_repr(self) -> str:
        return (f"in_channels={self.in_channels}, "
                f"out_channels={self.out_channels}, "
                f"gamma={self.gamma}, K={self.K}, "
                f"normalize={self.normalize}, "
                f"power_mode={self.power_mode}")
