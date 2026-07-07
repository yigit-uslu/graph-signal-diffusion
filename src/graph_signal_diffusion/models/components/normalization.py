"""
Normalization layers for graph diffusion models.

This module provides flexible normalization layers that handle:
- Variable graph sizes (different N across batches)
- Temporal sequences (T dimension)
- Multiple input formats: (B, T, N, F) or (B*N, T, F)

We implement these ourselves (not using PyG) because:
1. PyG normalizations expect (B*N, F) format, we use (B, T, N, F)
2. PyG doesn't support temporal sequences natively
3. PyG doesn't have adaptive/conditional normalization (needed for diffusion)
4. Our implementation is faster for our use case (no reshape overhead)

Normalization Types:
    - LayerNorm: Normalize over feature dimension (per sample)
    - GroupNorm: Normalize over feature groups (independent of batch/graph size)
    - InstanceNorm: Normalize per sample (per node or per graph)
    - GraphNorm: Normalize per graph (aggregates over nodes)
    - AdaptiveLayerNorm: Conditional normalization (FiLM-style modulation)
    - AdaptiveGroupNorm: Conditional group normalization

Shape Convention:
    - Main format: (B, T, N, F) - batch, time, nodes, features
    - Flattened: (B*N, T, F) - for per-node processing
    - Conditioning: (B, emb_dim) or (B, N, emb_dim)

Usage Examples:
    >>> # Standard layer normalization
    >>> norm = LayerNorm(num_features=64)
    >>> x = torch.randn(4, 10, 100, 64)  # (B, T, N, F)
    >>> out = norm(x)  # (4, 10, 100, 64)
    
    >>> # Group normalization (better for variable batch sizes)
    >>> norm = GroupNorm(num_features=64, num_groups=8)
    >>> out = norm(x)
    
    >>> # Adaptive normalization with time conditioning
    >>> from graph_signal_diffusion.models.components import TimeEmbedding
    >>> time_embed = TimeEmbedding(embed_dim=128, num_timesteps=1000)
    >>> norm = AdaptiveLayerNorm(num_features=64, cond_dim=128)
    >>> 
    >>> timesteps = torch.randint(0, 1000, (4,))
    >>> time_emb = time_embed(timesteps)  # (4, 128)
    >>> out = norm(x, time_emb)  # (4, 10, 100, 64) - modulated by time
    
    >>> # Factory function for easy switching
    >>> norm = get_normalization('group', num_features=64, num_groups=8)
"""

import torch
import torch.nn as nn
from typing import Optional, Literal


class LayerNorm(nn.Module):
    """
    Layer normalization for graph signals.
    
    Normalizes over the feature dimension, independently for each
    sample (batch item, timestep, node).
    
    Works with both 3D (B*N, T, F) and 4D (B, T, N, F) inputs.
    
    Args:
        num_features: Number of features (F)
        eps: Small constant for numerical stability
        elementwise_affine: Whether to learn affine parameters (scale, shift)
    
    Shape:
        - Input: (B, T, N, F) or (B*N, T, F) or (B*N, F)
        - Output: Same shape as input
    
    Examples:
        >>> norm = LayerNorm(num_features=64)
        >>> x = torch.randn(4, 10, 100, 64)  # (B, T, N, F)
        >>> out = norm(x)  # (4, 10, 100, 64)
        
        >>> # Also works with 3D input
        >>> x = torch.randn(400, 10, 64)  # (B*N, T, F)
        >>> out = norm(x)  # (400, 10, 64)
    """
    
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        
        self.num_features = num_features
        self.eps = eps
        
        # Learnable affine parameters
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize over feature dimension.
        
        Args:
            x: Input of shape (B, T, N, F) or (B*N, T, F) or (B*N, F)
        
        Returns:
            Normalized tensor with same shape as input
        """
        # Normalize over last dimension (features)
        normalized_shape = (self.num_features,)
        
        # Use PyTorch's layer_norm for efficiency
        x = torch.nn.functional.layer_norm(
            x, normalized_shape, self.weight, self.bias, self.eps
        )
        
        return x
    
    def extra_repr(self) -> str:
        return f"num_features={self.num_features}, eps={self.eps}"


class GroupNorm(nn.Module):
    """
    Group normalization for graph signals.
    
    Divides features into groups and normalizes within each group.
    Independent of batch size and graph size - ideal for variable graphs.
    
    Args:
        num_features: Number of features (F)
        num_groups: Number of groups (F must be divisible by num_groups)
        eps: Small constant for numerical stability
        affine: Whether to learn affine parameters
    
    Shape:
        - Input: (B, T, N, F) or (B*N, T, F) or (B*N, F)
        - Output: Same shape as input
    
    Examples:
        >>> norm = GroupNorm(num_features=64, num_groups=8)
        >>> x = torch.randn(4, 10, 100, 64)  # (B, T, N, F)
        >>> out = norm(x)  # (4, 10, 100, 64)
        
        >>> # Works regardless of batch size
        >>> x_small = torch.randn(1, 10, 100, 64)
        >>> out_small = norm(x_small)  # Still works!
    """
    
    def __init__(
        self,
        num_features: int,
        num_groups: int = 32,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        super().__init__()
        
        assert num_features % num_groups == 0, \
            f"num_features ({num_features}) must be divisible by num_groups ({num_groups})"
        
        self.num_features = num_features
        self.num_groups = num_groups
        self.eps = eps
        
        # Learnable affine parameters
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
            
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize within feature groups.
        
        Args:
            x: Input of shape (B, T, N, F) or (B*N, T, F) or (B*N, F)
        
        Returns:
            Normalized tensor with same shape as input
        """
        original_shape = x.shape
        
        # Reshape to (N_samples, F) for group norm
        if x.dim() == 4:
            # (B, T, N, F) -> (B*T*N, F)
            x = x.reshape(-1, self.num_features)
            reshape_needed = True
        elif x.dim() == 3:
            # (B*N, T, F) -> (B*N*T, F)
            x = x.reshape(-1, self.num_features)
            reshape_needed = True
        else:
            reshape_needed = False
        
        # Add dummy dimensions for group_norm: (N_samples, F) -> (N_samples, F, 1)
        x = x.unsqueeze(-1)
        
        # Apply group norm
        x = torch.nn.functional.group_norm(
            x, self.num_groups, self.weight, self.bias, self.eps
        )
        
        # Remove dummy dimension
        x = x.squeeze(-1)
        
        # Reshape back
        if reshape_needed:
            x = x.reshape(original_shape)
        
        return x
    
    def extra_repr(self) -> str:
        return f"num_features={self.num_features}, num_groups={self.num_groups}, eps={self.eps}"


class InstanceNorm(nn.Module):
    """
    Instance normalization for graph signals.
    
    Normalizes each sample independently across spatial-temporal dimensions.
    Useful when each graph/sample has different statistics.
    
    Args:
        num_features: Number of features (F)
        eps: Small constant for numerical stability
        affine: Whether to learn affine parameters
        track_running_stats: Whether to track running statistics
    
    Shape:
        - Input: (B, T, N, F) or (B*N, T, F)
        - Output: Same shape as input
    
    Examples:
        >>> norm = InstanceNorm(num_features=64)
        >>> x = torch.randn(4, 10, 100, 64)  # (B, T, N, F)
        >>> out = norm(x)  # (4, 10, 100, 64)
        
        >>> # Each batch item is normalized independently
        >>> # Useful when graphs have very different characteristics
    """
    
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = False,
        track_running_stats: bool = False,
    ):
        super().__init__()
        
        self.num_features = num_features
        self.eps = eps
        self.track_running_stats = track_running_stats
        
        # Instance norm typically doesn't use affine parameters
        # But we allow it for flexibility
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
        # Running statistics (optional)
        if track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)
            self.register_buffer('num_batches_tracked', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize per instance.
        
        Args:
            x: Input of shape (B, T, N, F) or (B*N, T, F)
        
        Returns:
            Normalized tensor with same shape as input
        """
        if x.dim() == 4:
            # (B, T, N, F) -> (B, F, T*N) for instance_norm
            B, T, N, F = x.shape
            x_reshaped = x.permute(0, 3, 1, 2).reshape(B, F, T * N)
        elif x.dim() == 3:
            # (B*N, T, F) -> (B*N, F, T) for instance_norm
            x_reshaped = x.transpose(1, 2)
        else:
            raise ValueError(f"Expected 3D or 4D input, got shape {x.shape}")
        
        # Apply instance norm
        x_normalized = torch.nn.functional.instance_norm(
            x_reshaped,
            running_mean=self.running_mean if not self.training else None,
            running_var=self.running_var if not self.training else None,
            weight=self.weight,
            bias=self.bias,
            use_input_stats=True,
            momentum=0.1,
            eps=self.eps,
        )
        
        # Reshape back
        if x.dim() == 4:
            x_normalized = x_normalized.reshape(B, F, T, N).permute(0, 2, 3, 1)
        elif x.dim() == 3:
            x_normalized = x_normalized.transpose(1, 2)
        
        return x_normalized
    
    def extra_repr(self) -> str:
        return f"num_features={self.num_features}, eps={self.eps}, affine={self.weight is not None}"


class GraphNorm(nn.Module):
    """
    Graph-level normalization.
    
    Normalizes statistics computed over all nodes in each graph.
    Useful for graph-level tasks and when graph sizes vary.
    
    Args:
        num_features: Number of features (F)
        eps: Small constant for numerical stability
        affine: Whether to learn affine parameters
    
    Shape:
        - Input: (B, T, N, F) or (B*N, T, F)
        - Output: Same shape as input
    
    Examples:
        >>> norm = GraphNorm(num_features=64)
        >>> x = torch.randn(4, 10, 100, 64)  # (B, T, N, F)
        >>> out = norm(x)  # (4, 10, 100, 64)
        
        >>> # Statistics computed over all nodes in each graph
        >>> # Useful when node features should be compared within each graph
    """
    
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        super().__init__()
        
        self.num_features = num_features
        self.eps = eps
        
        # Learnable affine parameters
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize at graph level.
        
        Args:
            x: Input of shape (B, T, N, F) or (B*N, T, F)
        
        Returns:
            Normalized tensor with same shape as input
        """
        if x.dim() == 4:
            # (B, T, N, F) - normalize over N dimension
            # Compute mean and var over nodes (dim=2)
            mean = x.mean(dim=2, keepdim=True)  # (B, T, 1, F)
            var = x.var(dim=2, keepdim=True, unbiased=False)  # (B, T, 1, F)
            
        elif x.dim() == 3:
            # (B*N, T, F) - can't distinguish graphs, treat as batch norm
            # This is ambiguous - ideally need batch info
            # Fall back to normalizing over batch dimension
            mean = x.mean(dim=0, keepdim=True)  # (1, T, F)
            var = x.var(dim=0, keepdim=True, unbiased=False)  # (1, T, F)
        else:
            raise ValueError(f"Expected 3D or 4D input, got shape {x.shape}")
        
        # Normalize
        x_normalized = (x - mean) / torch.sqrt(var + self.eps)
        
        # Apply affine transformation
        if self.weight is not None:
            x_normalized = x_normalized * self.weight
        if self.bias is not None:
            x_normalized = x_normalized + self.bias
        
        return x_normalized
    
    def extra_repr(self) -> str:
        return f"num_features={self.num_features}, eps={self.eps}"


class AdaptiveLayerNorm(nn.Module):
    """
    Adaptive layer normalization with conditional modulation (FiLM).
    
    Applies layer norm followed by feature-wise affine transformation
    conditioned on external input (e.g., timestep embedding).
    
    Scale and shift are predicted from conditioning input:
        out = (1 + scale) * layer_norm(x) + shift
    
    The residual formulation (1 + scale) ensures the transformation is
    centered around the identity when scale=0, which helps training stability.
    
    Args:
        num_features: Number of features (F)
        cond_dim: Conditioning embedding dimension
        eps: Small constant for numerical stability
    
    Shape:
        - Input x: (B, T, N, F) or (B*N, T, F)
        - Conditioning cond: (B, cond_dim) or (B, N, cond_dim) or (B*N, cond_dim)
        - Output: Same shape as x
    
    Examples:
        >>> # Time-conditioned normalization (most common)
        >>> norm = AdaptiveLayerNorm(num_features=64, cond_dim=128)
        >>> x = torch.randn(4, 10, 100, 64)  # (B, T, N, F)
        >>> time_emb = torch.randn(4, 128)  # (B, cond_dim)
        >>> out = norm(x, time_emb)  # (4, 10, 100, 64)
        
        >>> # Per-node conditioning
        >>> x = torch.randn(4, 10, 100, 64)  # (B, T, N, F)
        >>> node_cond = torch.randn(4, 100, 128)  # (B, N, cond_dim)
        >>> out = norm(x, node_cond)  # (4, 10, 100, 64)
    
    References:
        - FiLM: Visual Reasoning with Feature-wise Linear Modulation
          (Perez et al., 2018): https://arxiv.org/abs/1709.07871
        - Denoising Diffusion Probabilistic Models
          (Ho et al., 2020): https://arxiv.org/abs/2006.11239
    """
    
    def __init__(
        self,
        num_features: int,
        cond_dim: int,
        eps: float = 1e-5,
    ):
        super().__init__()
        
        self.num_features = num_features
        self.cond_dim = cond_dim
        self.eps = eps
        
        # Base layer norm (no affine parameters)
        self.norm = LayerNorm(num_features, eps=eps, elementwise_affine=False)
        
        # Conditioning network: predicts scale and shift
        self.cond_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, num_features * 2),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Adaptive normalization with conditioning.
        
        Args:
            x: Input of shape (B, T, N, F) or (B*N, T, F)
            cond: Conditioning of shape (B, cond_dim) or (B, N, cond_dim) or (B*N, cond_dim)
        
        Returns:
            Normalized and modulated tensor with same shape as x
        """
        # Apply base layer norm
        x_normalized = self.norm(x)
        
        # Predict scale and shift from conditioning
        cond_params = self.cond_layers(cond)  # (..., 2*F)
        scale, shift = torch.chunk(cond_params, 2, dim=-1)  # Each: (..., F)
        
        # Broadcast conditioning to match input shape
        if x.dim() == 4 and cond.dim() == 2:
            # x: (B, T, N, F), cond: (B, cond_dim)
            # scale/shift: (B, F) -> (B, 1, 1, F)
            scale = scale.unsqueeze(1).unsqueeze(2)
            shift = shift.unsqueeze(1).unsqueeze(2)
        elif x.dim() == 4 and cond.dim() == 3:
            # x: (B, T, N, F), cond: (B, N, cond_dim)
            # scale/shift: (B, N, F) -> (B, 1, N, F)
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        elif x.dim() == 3 and cond.dim() == 2:
            # x: (B*N, T, F), cond: (B*N, cond_dim)
            # scale/shift: (B*N, F) -> (B*N, 1, F)
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        
        # Apply affine transformation (FiLM)
        # Add 1 to scale for residual connection (identity at initialization)
        x_modulated = (1 + scale) * x_normalized + shift
        
        return x_modulated
    
    def extra_repr(self) -> str:
        return f"num_features={self.num_features}, cond_dim={self.cond_dim}, eps={self.eps}"


class AdaptiveGroupNorm(nn.Module):
    """
    Adaptive group normalization with conditional modulation.
    
    Combines group normalization with FiLM-style conditioning.
    More stable than AdaptiveLayerNorm for small batches and variable graph sizes.
    
    Args:
        num_features: Number of features (F)
        num_groups: Number of groups
        cond_dim: Conditioning embedding dimension
        eps: Small constant for numerical stability
    
    Shape:
        - Input x: (B, T, N, F) or (B*N, T, F)
        - Conditioning cond: (B, cond_dim) or (B, N, cond_dim) or (B*N, cond_dim)
        - Output: Same shape as x
    
    Examples:
        >>> # Time-conditioned group normalization (recommended for diffusion)
        >>> norm = AdaptiveGroupNorm(num_features=64, num_groups=8, cond_dim=128)
        >>> x = torch.randn(4, 10, 100, 64)  # (B, T, N, F)
        >>> time_emb = torch.randn(4, 128)  # (B, cond_dim)
        >>> out = norm(x, time_emb)  # (4, 10, 100, 64)
        
        >>> # Works well even with batch_size=1
        >>> x_small = torch.randn(1, 10, 100, 64)
        >>> time_emb_small = torch.randn(1, 128)
        >>> out_small = norm(x_small, time_emb_small)  # Still stable!
    
    References:
        - Group Normalization (Wu & He, 2018): https://arxiv.org/abs/1803.08494
        - Denoising Diffusion Probabilistic Models (Ho et al., 2020)
    """
    
    def __init__(
        self,
        num_features: int,
        num_groups: int = 32,
        cond_dim: int = 128,
        eps: float = 1e-5,
    ):
        super().__init__()
        
        self.num_features = num_features
        self.num_groups = num_groups
        self.cond_dim = cond_dim
        self.eps = eps
        
        # Base group norm (no affine parameters)
        self.norm = GroupNorm(
            num_features=num_features,
            num_groups=num_groups,
            eps=eps,
            affine=False,
        )
        
        # Conditioning network
        self.cond_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, num_features * 2),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Adaptive group normalization with conditioning.
        
        Args:
            x: Input of shape (B, T, N, F) or (B*N, T, F)
            cond: Conditioning of shape (B, cond_dim) or (B, N, cond_dim) or (B*N, cond_dim)
        
        Returns:
            Normalized and modulated tensor with same shape as x
        """
        # Apply base group norm
        x_normalized = self.norm(x)
        
        # Predict scale and shift
        cond_params = self.cond_layers(cond)
        scale, shift = torch.chunk(cond_params, 2, dim=-1)
        
        # Broadcast conditioning
        if x.dim() == 4 and cond.dim() == 2:
            scale = scale.unsqueeze(1).unsqueeze(2)
            shift = shift.unsqueeze(1).unsqueeze(2)
        elif x.dim() == 4 and cond.dim() == 3:
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        elif x.dim() == 3 and cond.dim() == 2:
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        
        # Apply affine transformation
        x_modulated = (1 + scale) * x_normalized + shift
        
        return x_modulated
    
    def extra_repr(self) -> str:
        return (f"num_features={self.num_features}, num_groups={self.num_groups}, "
                f"cond_dim={self.cond_dim}, eps={self.eps}")


def get_normalization(
    norm_type: Literal['layer', 'group', 'instance', 'graph', 'none'],
    num_features: int,
    num_groups: int = 32,
    **kwargs,
) -> nn.Module:
    """
    Factory function to create normalization layers.
    
    Provides a convenient way to switch between normalization types
    during experimentation or hyperparameter tuning.
    
    Args:
        norm_type: Type of normalization
            - 'layer': LayerNorm (default, most common)
            - 'group': GroupNorm (better for variable batch sizes)
            - 'instance': InstanceNorm (per-sample normalization)
            - 'graph': GraphNorm (per-graph normalization)
            - 'none': Identity (no normalization)
        num_features: Number of features
        num_groups: Number of groups (for GroupNorm only)
        **kwargs: Additional arguments passed to normalization layer
    
    Returns:
        Normalization module
    
    Examples:
        >>> # Try different normalizations easily
        >>> for norm_type in ['layer', 'group', 'instance']:
        ...     norm = get_normalization(norm_type, num_features=64)
        ...     x = torch.randn(4, 10, 100, 64)
        ...     out = norm(x)
        
        >>> # Use in config-driven code
        >>> config = {'norm_type': 'group', 'num_groups': 8}
        >>> norm = get_normalization(
        ...     config['norm_type'],
        ...     num_features=64,
        ...     num_groups=config.get('num_groups', 32)
        ... )
    """
    if norm_type == 'layer':
        return LayerNorm(num_features, **kwargs)
    elif norm_type == 'group':
        return GroupNorm(num_features, num_groups=num_groups, **kwargs)
    elif norm_type == 'instance':
        return InstanceNorm(num_features, **kwargs)
    elif norm_type == 'graph':
        return GraphNorm(num_features, **kwargs)
    elif norm_type == 'none':
        return nn.Identity()
    else:
        raise ValueError(
            f"Unknown norm_type: {norm_type}. "
            f"Choose from: 'layer', 'group', 'instance', 'graph', 'none'"
        )