"""
Embedding layers for graph diffusion models.

This module provides reusable embedding components:
- MLPEmbedding: Generic MLP acting on feature dimension
- SinusoidalEmbedding: Positional encoding for timesteps
- TimeEmbedding: Sinusoidal + MLP for diffusion timesteps
- ConditionalMLPEmbedding: Flatten spatio-temporal + MLP for conditioning

Shape Convention:
    - Main signals: (B, T, N, F) - batch, time, nodes, features
    - Flattened signals: (B*N, T, F) - batch×nodes, time, features
    - Time embeddings: (B, embed_dim) - one per batch
    - Conditional embeddings: (B, N, embed_dim) - one per node

Usage:
    # Generic MLP
    mlp = MLPEmbedding(in_dim=64, out_dim=128, hidden_dims=[128, 128])
    
    # Time embedding
    time_embed = TimeEmbedding(embed_dim=128)
    emb = time_embed(timesteps)  # (B,) -> (B, 128)
    
    # Conditional MLP embedding
    cond_embed = ConditionalMLPEmbedding(in_channels=8, embed_dim=64)
    emb = cond_embed(observations)  # (B, T, N, F) -> (B, N, 64)
"""

import torch
import torch.nn as nn
import math
from typing import Optional, List, Literal

from .graph_conv import TemporalDepthwiseMixer


class MLPEmbedding(nn.Module):
    """
    Generic Multi-Layer Perceptron for embedding.
    
    Acts on the last dimension (feature/channel dimension) of input tensor.
    Supports arbitrary input shapes: (B, ..., in_dim) -> (B, ..., out_dim)
    
    Args:
        in_dim: Input feature dimension
        out_dim: Output embedding dimension
        hidden_dims: List of hidden layer dimensions (if None, uses [out_dim * 2])
        activation: Activation function ('relu', 'silu', 'gelu', 'tanh')
        dropout: Dropout rate (applied after each hidden layer)
        use_layer_norm: Whether to use LayerNorm after each layer
        bias: Whether to use bias in linear layers
    
    Example:
        >>> mlp = MLPEmbedding(in_dim=16, out_dim=64, hidden_dims=[32, 48])
        >>> x = torch.randn(4, 10, 16)  # (batch, nodes, features)
        >>> out = mlp(x)  # (4, 10, 64)
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Optional[List[int]] = None,
        activation: Literal['relu', 'silu', 'gelu', 'tanh'] = 'silu',
        dropout: float = 0.0,
        use_layer_norm: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Default hidden dims: single layer with out_dim * 2
        if hidden_dims is None:
            hidden_dims = [out_dim * 2]
        
        # Build layers
        layers = []
        dims = [in_dim] + hidden_dims + [out_dim]
        
        for i in range(len(dims) - 1):
            # Linear layer
            layers.append(nn.Linear(dims[i], dims[i + 1], bias=bias))
            
            # Layer norm (optional, before activation)
            if use_layer_norm and i < len(dims) - 2:
                layers.append(nn.LayerNorm(dims[i + 1]))
            
            # Activation (not on last layer)
            if i < len(dims) - 2:
                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'silu':
                    layers.append(nn.SiLU())
                elif activation == 'gelu':
                    layers.append(nn.GELU())
                elif activation == 'tanh':
                    layers.append(nn.Tanh())
                else:
                    raise ValueError(f"Unknown activation: {activation}")
                
                # Dropout (after activation)
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (..., in_dim)
        
        Returns:
            Output tensor of shape (..., out_dim)
        """
        return self.mlp(x)


class SinusoidalEmbedding(nn.Module):
    """
    Sinusoidal positional embedding (Vaswani et al., 2017).
    
    Encodes scalar values (e.g., timesteps, positions) into high-dimensional
    embeddings using sine and cosine functions of different frequencies.
    
    Args:
        embed_dim: Embedding dimension (must be even)
        max_period: Maximum period for sinusoidal encoding (default: 10000)
        scale: Scaling factor for input before encoding (default: 1.0)
    
    Example:
        >>> sin_embed = SinusoidalEmbedding(embed_dim=128)
        >>> t = torch.tensor([0, 10, 100, 999])
        >>> emb = sin_embed(t)  # (4, 128)
    """
    
    def __init__(
        self,
        embed_dim: int,
        max_period: int = 10000,
        scale: float = 1.0,
    ):
        super().__init__()
        
        assert embed_dim % 2 == 0, f"embed_dim must be even, got {embed_dim}"
        
        self.embed_dim = embed_dim
        self.max_period = max_period
        self.scale = scale
        
        # Precompute frequency bands
        half_dim = embed_dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half_dim, dtype=torch.float32) / (half_dim - 1)
        )
        self.register_buffer('freqs', freqs)  # (embed_dim // 2,)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode scalar values into sinusoidal embeddings.
        
        Args:
            x: Input tensor of shape (...,) containing scalar values
        
        Returns:
            Embeddings of shape (..., embed_dim)
        """
        # Scale input
        x = x * self.scale
        
        # Add dimension for broadcasting: (...,) -> (..., 1)
        x = x.unsqueeze(-1)
        
        # Compute angles: (..., 1) * (embed_dim // 2,) -> (..., embed_dim // 2)
        angles = x * self.freqs
        
        # Concatenate sin and cos
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        
        return emb



class AdaptiveSinusoidalEmbedding(nn.Module):
    """
    Sinusoidal embedding with adaptive frequency based on noise schedule.
    
    Args:
        embed_dim: Embedding dimension (must be even)
        num_timesteps: Total number of diffusion timesteps
        min_period: Minimum period (default: 2, for adjacent timesteps)
        max_period: Maximum period (if None, set to 2*num_timesteps)
        scale_method: How to scale ('linear', 'log', 'auto')
    
    Example:
        >>> # Adaptive to your specific timestep range
        >>> adaptive_embed = AdaptiveSinusoidalEmbedding(
        ...     embed_dim=128,
        ...     num_timesteps=1000,
        ...     max_period=None  # Will be set to 2000 automatically
        ... )
        >>> t = torch.tensor([0, 100, 500, 999])
        >>> emb = adaptive_embed(t)  # (4, 128)
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_timesteps: int,
        min_period: float = 2.0,
        max_period: Optional[float] = None,
        scale_method: Literal['linear', 'log', 'auto'] = 'auto',
    ):
        super().__init__()
        
        assert embed_dim % 2 == 0, f"embed_dim must be even, got {embed_dim}"
        
        self.embed_dim = embed_dim
        self.num_timesteps = num_timesteps
        self.min_period = min_period
        
        # Automatically set max_period if not provided
        if max_period is None:
            if scale_method == 'auto':
                # Heuristic: 2 * num_timesteps ensures good coverage
                max_period = 2.0 * num_timesteps
            elif scale_method == 'log':
                # For log-scale noise schedules (common in DDPM)
                max_period = num_timesteps ** 1.5
            else:
                max_period = 2.0 * num_timesteps
        
        self.max_period = max_period
        
        # Precompute frequency bands
        half_dim = embed_dim // 2
        
        # Frequencies range from 1/max_period to 1/min_period
        # In log space: log(1/max_period) to log(1/min_period)
        freqs = torch.exp(
            torch.linspace(
                math.log(1.0 / max_period),
                math.log(1.0 / min_period),
                half_dim,
                dtype=torch.float32
            )
        )
        
        self.register_buffer('freqs', freqs)  # (embed_dim // 2,)
        
        # Also store normalization factor
        # Normalize timesteps to [0, 1] range
        self.register_buffer(
            'timestep_scale',
            torch.tensor(1.0 / num_timesteps, dtype=torch.float32)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode timesteps into sinusoidal embeddings.
        
        Args:
            x: Input tensor of shape (...,) containing timestep indices
        
        Returns:
            Embeddings of shape (..., embed_dim)
        """
        # Normalize timesteps to [0, 1] range
        x_normalized = x.float() * self.timestep_scale
        
        # Add dimension for broadcasting: (...,) -> (..., 1)
        x_normalized = x_normalized.unsqueeze(-1)
        
        # Compute angles: (..., 1) * (embed_dim // 2,) -> (..., embed_dim // 2)
        angles = x_normalized * 2 * math.pi / self.freqs
        
        # Concatenate sin and cos
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        
        return emb
    


class TimeEmbedding(nn.Module):
    """
    Time embedding for diffusion models with principled frequency scaling.
    
    Combines sinusoidal positional encoding with an MLP projection.
    
    Args:
        embed_dim: Output embedding dimension
        num_timesteps: Total number of diffusion timesteps (for adaptive scaling)
        max_period: Maximum period (if None, set to 2*num_timesteps)
        hidden_multiplier: Multiplier for hidden layer size (default: 4)
        activation: Activation function for MLP
        dropout: Dropout rate for MLP
    
    Example:
        >>> # Adaptive to your timestep range
        >>> time_embed = TimeEmbedding(
        ...     embed_dim=128,
        ...     num_timesteps=1000,  # Now required!
        ...     max_period=None      # Will be set to 2000
        ... )
        >>> timesteps = torch.randint(0, 1000, (32,))
        >>> emb = time_embed(timesteps)  # (32, 128)
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_timesteps: int,
        max_period: Optional[int] = None,
        hidden_multiplier: int = 4,
        activation: str = 'silu',
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_timesteps = num_timesteps

        # Principled choice of max period
        if max_period is None:
            max_period = 2 * num_timesteps
            print(f"[TimeEmbedding] Setting max_period={max_period} "
                  f"(2 × num_timesteps={num_timesteps})")
        
        # # Sinusoidal encoding
        # self.sinusoidal = SinusoidalEmbedding(
        #     embed_dim=embed_dim,
        #     max_period=max_period
        # )
        # Sinusoidal encoding
        self.sinusoidal = AdaptiveSinusoidalEmbedding(
            embed_dim=embed_dim,
            num_timesteps=num_timesteps,
            max_period=max_period
        )
        
        # MLP projection
        self.mlp = MLPEmbedding(
            in_dim=embed_dim,
            out_dim=embed_dim,
            hidden_dims=[embed_dim * hidden_multiplier],
            activation=activation,
            dropout=dropout,
        )
    
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Encode timesteps into embeddings.
        
        Args:
            timesteps: Timestep indices of shape (B,) or (B, ...)
        
        Returns:
            Time embeddings of shape (B, embed_dim) or (B, ..., embed_dim)
        """
        # Sinusoidal encoding
        emb = self.sinusoidal(timesteps)
        
        # MLP projection
        emb = self.mlp(emb)
        
        return emb


class ConditionalMLPEmbedding(nn.Module):
    """
    Embedding for conditional information (e.g., past observations).
    
    Flattens temporal-spatial dimensions and applies MLP to create embeddings.
    Supports both 3D (B*N, T, F) and 4D (B, T, N, F) inputs.
    
    Args:
        in_channels: Input feature dimension (F)
        embed_dim: Output embedding dimension
        hidden_multiplier: Multiplier for hidden layer size
        activation: Activation function
        dropout: Dropout rate
        flatten_time: Whether to flatten time dimension (default: True)
    
    Example:
        >>> cond_embed = ConditionalMLPEmbedding(in_channels=8, embed_dim=64)
        >>> x = torch.randn(2, 25, 100, 8)  # (B, T, N, F)
        >>> emb = cond_embed(x)  # (2, 100, 64) - one embedding per node
    """
    
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        hidden_multiplier: int = 2,
        activation: str = 'silu',
        dropout: float = 0.0,
        flatten_time: bool = True,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.flatten_time = flatten_time
        
        # Use nn.LazyLinear to handle variable time dimensions
        self.mlp = nn.Sequential(
            nn.LazyLinear(embed_dim * hidden_multiplier),
            nn.SiLU() if activation == 'silu' else nn.ReLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(embed_dim * hidden_multiplier, embed_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed conditional information.
        
        Args:
            x: Input tensor of shape (B, T, N, F) or (B*N, T, F)
        
        Returns:
            Embeddings of shape (B, N, embed_dim) or (B*N, embed_dim)
        """
        if x.dim() == 4:
            # Input: (B, T, N, F) -> Flatten to (B, N, T*F)
            B, T, N, F = x.shape
            # Permute to (B, N, T, F) then reshape to (B, N, T*F)
            x = x.permute(0, 2, 1, 3)  # (B, T, N, F) -> (B, N, T, F)
            x = x.reshape(B, N, T * F)  # (B, N, T, F) -> (B, N, T*F)
        elif x.dim() == 3:
            # Input: (B*N, T, F) -> Flatten to (B*N, T*F)
            B_N, T, F = x.shape
            x = x.reshape(B_N, T * F)  # (B*N, T, F) -> (B*N, T*F)
        else:
            raise ValueError(f"Expected 3D or 4D input, got shape {x.shape}")
        
        # Apply MLP: (B, N, T*F) -> (B, N, embed_dim) or (B*N, T*F) -> (B*N, embed_dim)
        emb = self.mlp(x)
        
        return emb
    


# Add this after ConditionalMLPEmbedding
class ConditionalTemporalConvEmbedding(nn.Module):
    """
    Conditional embedding via causal temporal convolution.
    
    Uses 1D causal convolutions along the temporal dimension to extract
    temporal patterns from past observations. Unlike ConditionalMLPEmbedding
    which flattens time, this preserves temporal structure.
    
    Architecture:
        Input (B, T, N, F) -> Per-node: (T, F)
        -> CausalConv1d layers (with dilation for multi-scale)
        -> Global pooling (max/mean) over time
        -> MLP projection -> (embed_dim,)
        Output: (B, N, embed_dim)
    
    Args:
        in_channels: Input feature dimension (F)
        embed_dim: Output embedding dimension
        hidden_channels: Hidden channel dimensions for conv layers
        kernel_size: Convolution kernel size (default: 3)
        num_layers: Number of conv layers (default: 3)
        dilation_growth: Dilation growth factor (default: 2)
        pooling: Temporal pooling method ('max', 'mean', 'last', 'attention')
        activation: Activation function
        dropout: Dropout rate
        use_residual: Whether to use residual connections
    
    Example:
        >>> conv_embed = ConditionalTemporalConvEmbedding(
        ...     in_channels=8,
        ...     embed_dim=64,
        ...     hidden_channels=[16, 32],
        ...     kernel_size=3,
        ...     num_layers=3
        ... )
        >>> x = torch.randn(2, 25, 100, 8)  # (B, T, N, F)
        >>> emb = conv_embed(x)  # (2, 100, 64) - one embedding per node
    """
    
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        hidden_channels: Optional[List[int]] = None,
        kernel_size: int = 3,
        num_layers: int = 3,
        dilation_growth: int = 2,
        pooling: Literal['max', 'mean', 'last', 'attention'] = 'mean',
        activation: str = 'silu',
        dropout: float = 0.0,
        use_residual: bool = True,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.pooling = pooling
        self.use_residual = use_residual
        
        # Default hidden channels: gradually increase
        if hidden_channels is None:
            hidden_channels = [in_channels * (2 ** i) for i in range(1, num_layers + 1)]
        
        assert len(hidden_channels) == num_layers, \
            f"hidden_channels length ({len(hidden_channels)}) must match num_layers ({num_layers})"
        
        # Build causal conv layers
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.residual_projections = nn.ModuleList()
        
        channels = [in_channels] + hidden_channels
        
        for i in range(num_layers):
            in_ch = channels[i]
            out_ch = channels[i + 1]
            dilation = dilation_growth ** i
            
            # Causal conv: padding on left side only
            # For kernel_size K and dilation D: padding = (K-1) * D
            padding = (kernel_size - 1) * dilation
            
            conv = nn.Conv1d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
                bias=False  # Use bias in norm layer instead
            )
            self.conv_layers.append(conv)
            
            # Normalization
            self.norm_layers.append(nn.BatchNorm1d(out_ch))
            
            # Residual projection (if dimensions don't match)
            if use_residual and in_ch != out_ch:
                self.residual_projections.append(nn.Conv1d(in_ch, out_ch, 1))
            else:
                self.residual_projections.append(None)
        
        # Activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.Identity()
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Temporal pooling
        if pooling == 'attention':
            # Learnable attention pooling
            self.attention_pool = nn.Sequential(
                nn.Linear(hidden_channels[-1], hidden_channels[-1] // 2),
                nn.Tanh(),
                nn.Linear(hidden_channels[-1] // 2, 1)
            )
        
        # Final projection to embed_dim
        self.output_proj = MLPEmbedding(
            in_dim=hidden_channels[-1],
            out_dim=embed_dim,
            hidden_dims=[embed_dim],
            activation=activation,
            dropout=dropout
        )
    
    def _causal_crop(self, x: torch.Tensor, target_length: int) -> torch.Tensor:
        """
        Crop the output to remove future-looking padding.
        
        Args:
            x: Conv output of shape (B*N, C, T')
            target_length: Target temporal length T
        
        Returns:
            Cropped tensor of shape (B*N, C, T)
        """
        if x.size(2) > target_length:
            # Remove rightmost padding (future information)
            x = x[:, :, :target_length]
        return x
    
    def _pool_temporal(
        self,
        x: torch.Tensor,
        method: str = 'mean'
    ) -> torch.Tensor:
        """
        Pool over temporal dimension.
        
        Args:
            x: Input of shape (B*N, C, T)
            method: Pooling method
        
        Returns:
            Pooled tensor of shape (B*N, C)
        """
        if method == 'max':
            return x.max(dim=2)[0]
        elif method == 'mean':
            return x.mean(dim=2)
        elif method == 'last':
            return x[:, :, -1]
        elif method == 'attention':
            # x: (B*N, C, T) -> (B*N, T, C)
            x_transposed = x.transpose(1, 2)
            # Compute attention scores: (B*N, T, 1)
            attn_scores = self.attention_pool(x_transposed)
            attn_weights = torch.softmax(attn_scores, dim=1)
            # Weighted sum: (B*N, T, C) * (B*N, T, 1) -> (B*N, C)
            pooled = (x_transposed * attn_weights).sum(dim=1)
            return pooled
        else:
            raise ValueError(f"Unknown pooling method: {method}")
    
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed conditional information via causal temporal convolution.
        
        Args:
            x: Input tensor of shape (B, T, N, F) or (B*N, T, F)
        
        Returns:
            Embeddings of shape (B, N, embed_dim) or (B*N, embed_dim)
        """
        original_shape = x.shape
        
        if x.dim() == 4:
            # Input: (B, T, N, F)
            B, T, N, F = x.shape
            # Reshape for per-node processing: (B*N, T, F)
            x = x.permute(0, 2, 1, 3).reshape(B * N, T, F)
            reshaped = True
        elif x.dim() == 3:
            # Input: (B*N, T, F)
            B_N, T, F = x.shape
            reshaped = False
        else:
            raise ValueError(
                f"Expected 3D (B*N, T, F) or 4D (B, T, N, F) input, "
                f"got shape {x.shape}"
            )
        
        # Conv1d expects (B, C, T) format
        # Current: (B*N, T, F) -> (B*N, F, T)
        x = x.transpose(1, 2)
        
        # Apply causal conv layers
        for i, (conv, norm, residual_proj) in enumerate(
            zip(self.conv_layers, self.norm_layers, self.residual_projections)
        ):
            identity = x
            
            # Causal convolution
            x = conv(x)
            
            # Crop to maintain causality (remove future padding)
            x = self._causal_crop(x, T)
            
            # Normalization
            x = norm(x)
            
            # Activation
            x = self.activation(x)
            
            # Dropout
            x = self.dropout(x)
            
            # Residual connection
            if self.use_residual:
                if residual_proj is not None:
                    identity = residual_proj(identity)
                    identity = self._causal_crop(identity, T)
                x = x + identity
        
        # Temporal pooling: (B*N, C, T) -> (B*N, C)
        x = self._pool_temporal(x, method=self.pooling)
        
        # Project to embed_dim: (B*N, C) -> (B*N, embed_dim)
        x = self.output_proj(x)
        
        # Reshape back if needed
        if reshaped:
            x = x.reshape(B, N, self.embed_dim)
        
        return x


class ConditionalTemporalMixerEmbedding(nn.Module):
    """
    Conditional embedding via stacked TemporalDepthwiseMixer blocks.

    This mirrors the temporal mixing logic used in the main GNN path while
    keeping the conditioning path as a separate encoder.

    Args:
        in_channels: Input conditional feature dimension
        embed_dim: Output embedding dimension
        hidden_channels: Hidden channel dimensions for mixer blocks
        num_layers: Number of temporal mixer blocks
        kernel_size: Temporal convolution kernel size used by mixers
        pooling: Temporal aggregation method ('max', 'mean', 'last', 'ema', 'attention')
        activation: Activation used inside mixers
        dropout: Dropout used inside mixers
        norm_type: Normalization inside mixers ('layer', 'group', 'instance', 'none')
        use_pointwise: Whether each mixer uses pointwise 1x1 Conv1d
        causal: Whether mixer temporal convolution is causal
        ema_alpha_init: Initial EMA coefficient if pooling='ema'
        output_mode: Conditioning output format: 'static' or 'time_varying'
        time_varying_method: Temporal mapping method for 'time_varying' mode
        projection_source_timesteps: Required T_cond for learned projection
        projection_target_timesteps: Required T_out for learned projection
        dilations: Optional per-layer dilation list for temporal mixers

    Shape:
        Input: (B, T_cond, N, F_cond) or (B*N, T_cond, F_cond)
        Output (static): (B, N, embed_dim) or (B*N, embed_dim)
        Output (time_varying): (B, T_out, N, embed_dim) or (B*N, T_out, embed_dim)
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        hidden_channels: Optional[List[int]] = None,
        num_layers: int = 3,
        kernel_size: int = 3,
        pooling: Literal['max', 'mean', 'last', 'ema', 'attention'] = 'mean',
        activation: Literal['relu', 'silu', 'gelu', 'leaky_relu'] = 'silu',
        dropout: float = 0.0,
        norm_type: Literal['layer', 'group', 'instance', 'none'] = 'layer',
        use_pointwise: bool = True,
        causal: bool = True,
        ema_alpha_init: float = 0.8,
        output_mode: Literal['static', 'time_varying'] = 'static',
        time_varying_method: Literal['learned_projection'] = 'learned_projection',
        projection_source_timesteps: Optional[int] = None,
        projection_target_timesteps: Optional[int] = None,
        dilations: Optional[List[int]] = None,
        gated: bool = False,
        attention_enabled: bool = False,
        attention_num_heads: int = 4,
        attention_dropout: float = 0.0,
        attention_max_timesteps: Optional[int] = None,
    ):
        super().__init__()

        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if output_mode not in {'static', 'time_varying'}:
            raise ValueError(f"Unknown output_mode: {output_mode}")
        if output_mode == 'time_varying' and time_varying_method not in {'learned_projection', 'none'}:
            raise ValueError(
                f"Unknown time_varying_method: {time_varying_method}. "
                "Expected 'learned_projection' or 'none'."
            )
        if output_mode == 'time_varying' and time_varying_method == 'learned_projection':
            if projection_source_timesteps is None or projection_target_timesteps is None:
                raise ValueError(
                    "projection_source_timesteps and projection_target_timesteps "
                    "must be set when time_varying_method='learned_projection'."
                )
            if projection_source_timesteps < 1 or projection_target_timesteps < 1:
                raise ValueError(
                    "projection_source_timesteps and projection_target_timesteps must be >= 1."
                )

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.pooling = pooling
        self.kernel_size = kernel_size
        self.causal = causal
        self.use_pointwise = use_pointwise
        self.output_mode = output_mode
        self.time_varying_method = time_varying_method
        self.projection_source_timesteps = projection_source_timesteps
        self.projection_target_timesteps = projection_target_timesteps
        self.last_forward_short_circuited = False

        if hidden_channels is None:
            hidden_channels = [in_channels * (2 ** i) for i in range(1, num_layers + 1)]
        if len(hidden_channels) != num_layers:
            raise ValueError(
                f"hidden_channels length ({len(hidden_channels)}) must match num_layers ({num_layers})"
            )
        self.hidden_channels = hidden_channels

        if dilations is None:
            dilations = [1] * num_layers
        if len(dilations) != num_layers:
            raise ValueError(
                f"dilations length ({len(dilations)}) must match num_layers ({num_layers})"
            )
        if any(d < 1 for d in dilations):
            raise ValueError("All dilations must be >= 1")
        self.dilations = dilations

        self.input_projs = nn.ModuleList()
        self.mixers = nn.ModuleList()

        prev_channels = in_channels
        for ch, dilation in zip(hidden_channels, self.dilations):
            if prev_channels == ch:
                self.input_projs.append(nn.Identity())
            else:
                self.input_projs.append(nn.Linear(prev_channels, ch))

            self.mixers.append(
                TemporalDepthwiseMixer(
                    channels=ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    causal=causal,
                    dropout=dropout,
                    activation=activation,
                    use_pointwise=use_pointwise,
                    norm_type=norm_type,
                    gated=gated,
                    attention_enabled=attention_enabled,
                    attention_num_heads=attention_num_heads,
                    attention_dropout=attention_dropout,
                    attention_max_timesteps=attention_max_timesteps,
                )
            )
            prev_channels = ch

        if pooling == 'attention':
            hidden = max(1, hidden_channels[-1] // 2)
            self.attention_pool = nn.Sequential(
                nn.Linear(hidden_channels[-1], hidden),
                nn.Tanh(),
                nn.Linear(hidden, 1),
            )

        if pooling == 'ema':
            alpha = min(max(float(ema_alpha_init), 1e-4), 1 - 1e-4)
            self.ema_alpha_logit = nn.Parameter(torch.logit(torch.tensor(alpha)))
        else:
            self.ema_alpha_logit = None

        self.output_proj = MLPEmbedding(
            in_dim=hidden_channels[-1],
            out_dim=embed_dim,
            hidden_dims=[embed_dim],
            activation=activation,
            dropout=dropout,
        )

        if self.output_mode == 'time_varying' and self.time_varying_method == 'learned_projection':
            self.temporal_projection_logits = nn.Parameter(
                torch.zeros(
                    self.projection_target_timesteps,
                    self.projection_source_timesteps,
                )
            )
        else:
            self.temporal_projection_logits = None

    def _pool_temporal(
        self,
        x: torch.Tensor,
        method: str,
    ) -> torch.Tensor:
        """
        Pool over temporal axis.

        Args:
            x: Tensor of shape (B*N, C, T)
            method: Pooling method
        Returns:
            Tensor of shape (B*N, C)
        """
        if method == 'max':
            return x.max(dim=2)[0]
        if method == 'mean':
            return x.mean(dim=2)
        if method == 'last':
            return x[:, :, -1]
        if method == 'attention':
            x_transposed = x.transpose(1, 2)  # (B*N, T, C)
            attn_scores = self.attention_pool(x_transposed)
            attn_weights = torch.softmax(attn_scores, dim=1)
            return (x_transposed * attn_weights).sum(dim=1)
        if method == 'ema':
            alpha = torch.sigmoid(self.ema_alpha_logit).to(dtype=x.dtype, device=x.device)
            ema = x[:, :, 0]
            for t in range(1, x.size(2)):
                ema = alpha * ema + (1.0 - alpha) * x[:, :, t]
            return ema
        raise ValueError(f"Unknown pooling method: {method}")

    def forward(
        self,
        x: torch.Tensor,
        target_timesteps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Embed conditional information via temporal mixer stack.

        Args:
            x: Tensor of shape (B, T_cond, N, F_cond) or (B*N, T_cond, F_cond)
            target_timesteps: Required output horizon when output_mode='time_varying'
        Returns:
            Embeddings of shape (B, N, embed_dim) / (B*N, embed_dim) for static mode
            or (B, T_out, N, embed_dim) / (B*N, T_out, embed_dim) for time_varying mode
        """
        if x.dim() == 4:
            B, T_cond, N, F = x.shape
            x = x.permute(0, 2, 1, 3).reshape(B * N, T_cond, F)
            reshaped = True
        elif x.dim() == 3:
            B_N, T_cond, _ = x.shape
            reshaped = False
        else:
            raise ValueError(
                f"Expected 3D (B*N, T_cond, F) or 4D (B, T_cond, N, F) input, got shape {x.shape}"
            )

        self.last_forward_short_circuited = T_cond == 1

        for input_proj, mixer in zip(self.input_projs, self.mixers):
            x = input_proj(x)  # (B*N, T_cond, C_i)
            if not self.last_forward_short_circuited:
                x = mixer(x)  # (B*N, T_cond, C_i)

        if self.output_mode == 'static':
            if self.last_forward_short_circuited:
                pooled = x[:, 0, :]
            else:
                pooled = self._pool_temporal(x.transpose(1, 2), method=self.pooling)

            emb = self.output_proj(pooled)

            if reshaped:
                emb = emb.reshape(B, N, self.embed_dim)
            return emb

        if self.time_varying_method == 'none':
            # No temporal projection — pass full T_cond through output_proj per-timestep.
            x_flat = x.reshape(x.size(0) * x.size(1), x.size(2))  # (B*N*T_cond, C_hidden)
            emb_flat = self.output_proj(x_flat)
            emb = emb_flat.reshape(x.size(0), T_cond, self.embed_dim)  # (B*N, T_cond, embed_dim)

            if reshaped:
                emb = emb.reshape(B, N, T_cond, self.embed_dim).permute(0, 2, 1, 3)
            return emb

        # method == 'learned_projection'
        if target_timesteps is None:
            raise ValueError(
                "target_timesteps must be provided when time_varying_method='learned_projection'."
            )
        if T_cond != self.projection_source_timesteps:
            raise ValueError(
                f"T_cond mismatch for time_varying conditioning: got {T_cond}, "
                f"expected {self.projection_source_timesteps}."
            )
        if target_timesteps != self.projection_target_timesteps:
            raise ValueError(
                f"target_timesteps mismatch for time_varying conditioning: got {target_timesteps}, "
                f"expected {self.projection_target_timesteps}."
            )

        projection_weights = torch.softmax(self.temporal_projection_logits, dim=1)  # (T_out, T_cond)
        projected = torch.einsum("btc,ot->boc", x, projection_weights)  # (B*N, T_out, C_hidden)
        projected_flat = projected.reshape(projected.size(0) * projected.size(1), projected.size(2))
        emb_flat = self.output_proj(projected_flat)
        emb = emb_flat.reshape(projected.size(0), projected.size(1), self.embed_dim)

        if reshaped:
            emb = emb.reshape(B, N, target_timesteps, self.embed_dim).permute(0, 2, 1, 3)
        return emb


class HybridConditionalEmbedding(nn.Module):
    """
    Hybrid conditional embedding combining temporal conv + MLP.
    
    Uses both temporal convolution (for local patterns) and MLP 
    (for global patterns) and fuses them.
    
    Args:
        in_channels: Input feature dimension
        embed_dim: Output embedding dimension
        conv_hidden_channels: Hidden channels for conv branch
        mlp_hidden_multiplier: Hidden multiplier for MLP branch
        kernel_size: Conv kernel size
        num_conv_layers: Number of conv layers
        pooling: Temporal pooling method
        fusion_method: How to fuse conv + MLP ('concat', 'add', 'gate')
        activation: Activation function
        dropout: Dropout rate
    
    Example:
        >>> hybrid_embed = HybridConditionalEmbedding(
        ...     in_channels=8,
        ...     embed_dim=64,
        ...     fusion_method='concat'
        ... )
        >>> x = torch.randn(2, 25, 100, 8)  # (B, T, N, F)
        >>> emb = hybrid_embed(x)  # (2, 100, 64)
    """
    
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        conv_hidden_channels: Optional[List[int]] = None,
        mlp_hidden_multiplier: int = 2,
        kernel_size: int = 3,
        num_conv_layers: int = 3,
        pooling: str = 'mean',
        fusion_method: Literal['concat', 'add', 'gate'] = 'concat',
        activation: str = 'silu',
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.fusion_method = fusion_method
        
        # Temporal conv branch
        self.conv_branch = ConditionalTemporalConvEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim if fusion_method != 'concat' else embed_dim // 2,
            hidden_channels=conv_hidden_channels,
            kernel_size=kernel_size,
            num_layers=num_conv_layers,
            pooling=pooling,
            activation=activation,
            dropout=dropout,
        )
        
        # MLP branch (flattens time)
        self.mlp_branch = ConditionalMLPEmbedding(
            in_channels=in_channels,
            embed_dim=embed_dim if fusion_method != 'concat' else embed_dim // 2,
            hidden_multiplier=mlp_hidden_multiplier,
            activation=activation,
            dropout=dropout,
        )
        
        # Fusion
        if fusion_method == 'concat':
            # Already handled by splitting embed_dim
            pass
        elif fusion_method == 'add':
            # Direct addition
            pass
        elif fusion_method == 'gate':
            # Gated fusion (learnable weights)
            self.gate = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Sigmoid()
            )
        else:
            raise ValueError(f"Unknown fusion_method: {fusion_method}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Embed via hybrid conv + MLP approach.
        
        Args:
            x: Input tensor of shape (B, T, N, F) or (B*N, T, F)
        
        Returns:
            Embeddings of shape (B, N, embed_dim) or (B*N, embed_dim)
        """
        # Get embeddings from both branches
        conv_emb = self.conv_branch(x)
        mlp_emb = self.mlp_branch(x)
        
        # Fuse
        if self.fusion_method == 'concat':
            return torch.cat([conv_emb, mlp_emb], dim=-1)
        elif self.fusion_method == 'add':
            return conv_emb + mlp_emb
        elif self.fusion_method == 'gate':
            # Learnable gating
            combined = torch.cat([conv_emb, mlp_emb], dim=-1)
            gate = self.gate(combined)
            return gate * conv_emb + (1 - gate) * mlp_emb


    

class CombinedEmbedding(nn.Module):
    """
    Fuse multiple embeddings (e.g., time + conditional).
    
    Takes two embeddings and fuses them via concatenation + MLP.
    Handles broadcasting of time embeddings across spatial dimension.
    
    Args:
        time_dim: Time embedding dimension
        cond_dim: Conditional embedding dimension
        out_dim: Output dimension
        hidden_multiplier: Multiplier for hidden layer size
        activation: Activation function
        dropout: Dropout rate
        fusion_method: How to combine embeddings ('concat', 'add', 'film')
    
    Example:
        >>> combine = CombinedEmbedding(time_dim=128, cond_dim=64, out_dim=128)
        >>> time_emb = torch.randn(4, 128)  # (B, time_dim)
        >>> cond_emb = torch.randn(4, 100, 64)  # (B, N, cond_dim) - one per node
        >>> fused = combine(time_emb, cond_emb)  # (4, 100, 128)
    """
    
    def __init__(
        self,
        time_dim: int,
        cond_dim: int,
        out_dim: int,
        hidden_multiplier: int = 2,
        activation: str = 'silu',
        dropout: float = 0.0,
        fusion_method: Literal['concat', 'add', 'film'] = 'concat',
    ):
        super().__init__()
        
        self.time_dim = time_dim
        self.cond_dim = cond_dim
        self.out_dim = out_dim
        self.fusion_method = fusion_method
        
        if fusion_method == 'concat':
            # Concatenate time + cond
            self.fusion = MLPEmbedding(
                in_dim=time_dim + cond_dim,
                out_dim=out_dim,
                hidden_dims=[out_dim * hidden_multiplier],
                activation=activation,
                dropout=dropout,
            )
        elif fusion_method == 'add':
            # Project both to same dimension and add
            self.time_proj = nn.Linear(time_dim, out_dim)
            self.cond_proj = nn.Linear(cond_dim, out_dim)
            self.fusion = MLPEmbedding(
                in_dim=out_dim,
                out_dim=out_dim,
                hidden_dims=[out_dim * hidden_multiplier],
                activation=activation,
                dropout=dropout,
            )
        elif fusion_method == 'film':
            # Feature-wise Linear Modulation (FiLM)
            self.time_to_scale = nn.Linear(time_dim, cond_dim)
            self.time_to_shift = nn.Linear(time_dim, cond_dim)
            self.fusion = MLPEmbedding(
                in_dim=cond_dim,
                out_dim=out_dim,
                hidden_dims=[out_dim * hidden_multiplier],
                activation=activation,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown fusion_method: {fusion_method}")
    
    def forward(
        self,
        time_emb: torch.Tensor,
        cond_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse time and conditional embeddings.
        
        Args:
            time_emb: Time embedding of shape (B, time_dim)
            cond_emb: Conditional embedding of shape (B, N, cond_dim) or (B*N, cond_dim)
        
        Returns:
            Fused embedding of shape (B, N, out_dim) or (B*N, out_dim)
        """
        if self.fusion_method == 'concat':
            # Broadcast time embedding to match spatial dimension
            if cond_emb.dim() == 3:
                # (B, time_dim) -> (B, N, time_dim)
                B, N, _ = cond_emb.shape
                time_emb = time_emb.unsqueeze(1).expand(B, N, -1)
            elif cond_emb.dim() == 2 and time_emb.size(0) != cond_emb.size(0):
                # Assume cond_emb is (B*N, cond_dim) and time_emb is (B, time_dim)
                # This case needs batch information to broadcast correctly
                # For now, assume they're already aligned
                pass
            
            # Concatenate
            fused = torch.cat([time_emb, cond_emb], dim=-1)
            
            # Apply MLP
            return self.fusion(fused)
        
        elif self.fusion_method == 'add':
            # Project and add
            time_proj = self.time_proj(time_emb)
            cond_proj = self.cond_proj(cond_emb)
            
            # Broadcast time if needed
            if cond_proj.dim() == 3 and time_proj.dim() == 2:
                time_proj = time_proj.unsqueeze(1)
            
            # Add and fuse
            fused = time_proj + cond_proj
            return self.fusion(fused)
        
        elif self.fusion_method == 'film':
            # FiLM: scale and shift
            scale = self.time_to_scale(time_emb)
            shift = self.time_to_shift(time_emb)
            
            # Broadcast if needed
            if cond_emb.dim() == 3 and scale.dim() == 2:
                scale = scale.unsqueeze(1)
                shift = shift.unsqueeze(1)
            
            # Apply FiLM
            modulated = scale * cond_emb + shift
            
            return self.fusion(modulated)
        
