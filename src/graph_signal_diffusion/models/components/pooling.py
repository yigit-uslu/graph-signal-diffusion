"""
Learnable graph pooling/unpooling for hierarchical GNN architectures.

Key Design Principles:
- Same graph structure (edge_index, edge_weight) used across all UNet levels
- Pooling = selecting active nodes + masking their features
- Unpooling = concatenating with skip connections (inactive nodes stay zero)
- Nested selection: Level k+1 selects from Level k's active nodes
- Explicit masking after each GNN layer to prevent message passing to/from inactive nodes

This module provides:
- PoolingGNN: Lightweight GNN for computing node selection scores
- LearnableGraphPool: Learn to select important nodes for downsampling
- GraphUnpool: Upsample by concatenating with skip connections

References:
    - Graph U-Nets: Gao & Ji "Graph U-Nets" (2019)
    - DiffPool: Ying et al. "Hierarchical Graph Representation Learning" (2018)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Literal

from .graph_conv import TAGConvLayer, StridedTAGConv
from .node_selector import NodeSelector

try:
    from torch_scatter import scatter as torch_scatter_scatter
except (ImportError, OSError):
    torch_scatter_scatter = None


class PoolingGNN(nn.Module):
    """
    Lightweight GNN for computing node selection scores.
    
    Uses a shallow GNN to compute importance scores for each node
    based on its features, graph structure, and optional conditioning.
    
    **Important:** Explicitly masks inactive nodes after each layer to prevent
    message passing from "leaking" information to/from inactive nodes.
    
    Args:
        in_channels: Input feature dimension
        hidden_channels: Hidden dimension (default: 32)
        num_layers: Number of GNN layers (default: 2)
        K: TAGConv hop count (default: 2)
        use_global_context: Whether to include global graph features (default: True)
        cond_dim: Optional conditioning dimension (default: None)
    
    Shape:
        - Input x: (B, T, N, F)
        - Edge index: (2, E) - same graph for all levels
        - Active mask: (B, N) - binary mask of active nodes
        - Conditioning: (B, cond_dim) optional
        - Output scores: (B, N) - selection probability per node
    
    Example:
        >>> pooling_gnn = PoolingGNN(in_channels=64, hidden_channels=32)
        >>> x = torch.randn(4, 10, 100, 64)  # (B, T, N, F)
        >>> edge_index = torch.randint(0, 400, (2, 500))
        >>> active_mask = torch.ones(4, 100, dtype=torch.bool)
        >>> scores = pooling_gnn(x, edge_index, active_mask=active_mask)
        >>> # scores: (4, 100) - importance score per node
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        num_layers: int = 2,
        K: int = 2,
        use_global_context: bool = True,
        cond_dim: Optional[int] = None,
        temporal_score_agg: Literal['mean', 'last', 'ema', 'attention'] = 'mean',
        ema_alpha_init: float = 0.8,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.use_global_context = use_global_context
        self.temporal_score_agg = temporal_score_agg
        self.ema_alpha_init = ema_alpha_init

        if temporal_score_agg not in {'mean', 'last', 'ema', 'attention'}:
            raise ValueError(
                f"Unknown temporal_score_agg='{temporal_score_agg}'. "
                "Expected one of: 'mean', 'last', 'ema', 'attention'."
            )
        if temporal_score_agg == 'ema':
            if not (0.0 < ema_alpha_init < 1.0):
                raise ValueError(
                    f"ema_alpha_init must be in (0, 1), got {ema_alpha_init}."
                )
            init_logit = torch.logit(torch.tensor(ema_alpha_init, dtype=torch.float32))
            self.ema_alpha_logit = nn.Parameter(init_logit)
        else:
            self.register_parameter('ema_alpha_logit', None)
        if temporal_score_agg == 'attention':
            self.temporal_attn_proj = nn.Linear(in_channels, 1)
        else:
            self.temporal_attn_proj = None
        
        # Input projection
        input_dim = in_channels
        if cond_dim is not None:
            input_dim += cond_dim
        if use_global_context:
            input_dim += in_channels
        
        self.input_proj = nn.Linear(input_dim, hidden_channels)
        
        # Lightweight GNN layers
        self.conv_layers = nn.ModuleList([
            TAGConvLayer(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                K=K,
            )
            for _ in range(num_layers)
        ])
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels)
            for _ in range(num_layers)
        ])
        
        # Output projection to selection scores
        self.score_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(hidden_channels // 2, 1),
        )

    def _aggregate_temporal(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate (B, T, N, F) over time to (B, N, F)."""
        if self.temporal_score_agg == 'mean':
            return x.mean(dim=1)
        if self.temporal_score_agg == 'last':
            return x[:, -1]
        if self.temporal_score_agg == 'ema':
            alpha = torch.sigmoid(self.ema_alpha_logit).to(dtype=x.dtype, device=x.device)
            x_agg = x[:, 0]
            for t in range(1, x.size(1)):
                x_agg = alpha * x_agg + (1.0 - alpha) * x[:, t]
            return x_agg
        # attention
        logits = self.temporal_attn_proj(x).squeeze(-1)  # (B, T, N)
        attn = torch.softmax(logits, dim=1).unsqueeze(-1)  # (B, T, N, 1)
        return (attn * x).sum(dim=1)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        active_mask: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute selection scores for each node.
        
        Args:
            x: Node features (B, T, N, F)
            edge_index: Edge indices (2, E) - same for all batches
            edge_weight: Optional edge weights (E,)
            active_mask: Binary mask (B, N) - which nodes are currently active
            cond: Optional conditioning (B, cond_dim)
        
        Returns:
            Selection scores (B, N) - importance score per node
        """
        B, T, N, F = x.shape
        
        # Aggregate features across time for scoring.
        x_agg = self._aggregate_temporal(x)  # (B, N, F)
        
        # CRITICAL: Zero out inactive nodes BEFORE GNN processing
        if active_mask is not None:
            x_agg = x_agg * active_mask.unsqueeze(-1).float()
        
        # Compute global context (only from active nodes)
        if self.use_global_context:
            if active_mask is not None:
                num_active = active_mask.sum(dim=1, keepdim=True).clamp(min=1)
                global_feats = x_agg.sum(dim=1) / num_active  # (B, F)
            else:
                global_feats = x_agg.mean(dim=1)  # (B, F)
            
            # Broadcast to all nodes
            global_feats = global_feats.unsqueeze(1).expand(-1, N, -1)  # (B, N, F)
            x_agg = torch.cat([x_agg, global_feats], dim=-1)
        
        # Add conditioning if provided
        if cond is not None:
            cond_expanded = cond.unsqueeze(1).expand(-1, N, -1)  # (B, N, cond_dim)
            x_agg = torch.cat([x_agg, cond_expanded], dim=-1)
        
        # Input projection
        h = self.input_proj(x_agg)  # (B, N, hidden)
        
        # CRITICAL: Zero out inactive nodes after projection
        if active_mask is not None:
            h = h * active_mask.unsqueeze(-1).float()
        
        # Apply GNN layers with residuals and explicit masking
        h = h.unsqueeze(1)  # (B, 1, N, hidden) - add dummy time dimension
        
        for conv, norm in zip(self.conv_layers, self.norms):
            h_new = conv(h, edge_index, edge_weight)
            
            # CRITICAL: Mask outputs AFTER each GNN layer
            # This prevents message passing from affecting inactive nodes
            if active_mask is not None:
                mask_4d = active_mask.unsqueeze(1).unsqueeze(-1).float()  # (B, 1, N, 1)
                h_new = h_new * mask_4d
            
            h_new = norm(h_new)
            h = torch.nn.functional.silu(h_new) + h
            
            # Mask residual output too
            if active_mask is not None:
                h = h * mask_4d
        
        h = h.squeeze(1)  # (B, N, hidden)
        
        # Compute selection scores
        scores = self.score_proj(h).squeeze(-1)  # (B, N)
        
        # Apply mask using large negative value for inactive nodes
        # This ensures they can't be selected
        # Use dtype-aware min value to support FP16 (AMP)
        if active_mask is not None:
            mask_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~active_mask.bool(), mask_value)
        
        return scores


class TemporalMLPSelectorHead(nn.Module):
    """
    Per-node scoring head without graph convolutions.

    Computes selector logits from x_processed using only:
    1) temporal aggregation over T
    2) optional global-context concatenation
    3) per-node MLP

    Optional hidden normalization can be applied before the final score MLP.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        use_global_context: bool = True,
        cond_dim: Optional[int] = None,
        temporal_score_agg: Literal['mean', 'last', 'ema', 'attention'] = 'mean',
        ema_alpha_init: float = 0.8,
        hidden_norm_type: Literal['layer', 'none'] = 'layer',
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.use_global_context = use_global_context
        self.temporal_score_agg = temporal_score_agg
        self.ema_alpha_init = ema_alpha_init
        self.hidden_norm_type = hidden_norm_type

        if temporal_score_agg not in {'mean', 'last', 'ema', 'attention'}:
            raise ValueError(
                f"Unknown temporal_score_agg='{temporal_score_agg}'. "
                "Expected one of: 'mean', 'last', 'ema', 'attention'."
            )
        if temporal_score_agg == 'ema':
            if not (0.0 < ema_alpha_init < 1.0):
                raise ValueError(
                    f"ema_alpha_init must be in (0, 1), got {ema_alpha_init}."
                )
            init_logit = torch.logit(torch.tensor(ema_alpha_init, dtype=torch.float32))
            self.ema_alpha_logit = nn.Parameter(init_logit)
        else:
            self.register_parameter('ema_alpha_logit', None)
        if temporal_score_agg == 'attention':
            self.temporal_attn_proj = nn.Linear(in_channels, 1)
        else:
            self.temporal_attn_proj = None
        if hidden_norm_type not in {'layer', 'none'}:
            raise ValueError(
                f"Unknown hidden_norm_type='{hidden_norm_type}'. "
                "Expected one of: 'layer', 'none'."
            )

        input_dim = in_channels
        if cond_dim is not None:
            input_dim += cond_dim
        if use_global_context:
            input_dim += in_channels

        self.input_proj = nn.Linear(input_dim, hidden_channels)
        self.hidden_norm: nn.Module
        if hidden_norm_type == 'layer':
            self.hidden_norm = nn.LayerNorm(hidden_channels)
        else:
            self.hidden_norm = nn.Identity()
        self.score_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(hidden_channels // 2, 1),
        )

    def _aggregate_temporal(self, x: torch.Tensor) -> torch.Tensor:
        if self.temporal_score_agg == 'mean':
            return x.mean(dim=1)
        if self.temporal_score_agg == 'last':
            return x[:, -1]
        if self.temporal_score_agg == 'ema':
            alpha = torch.sigmoid(self.ema_alpha_logit).to(dtype=x.dtype, device=x.device)
            x_agg = x[:, 0]
            for t in range(1, x.size(1)):
                x_agg = alpha * x_agg + (1.0 - alpha) * x[:, t]
            return x_agg
        logits = self.temporal_attn_proj(x).squeeze(-1)  # (B, T, N)
        attn = torch.softmax(logits, dim=1).unsqueeze(-1)  # (B, T, N, 1)
        return (attn * x).sum(dim=1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        active_mask: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del edge_index, edge_weight  # Not used by this head.
        B, _, N, _ = x.shape

        x_agg = self._aggregate_temporal(x)  # (B, N, F)
        if active_mask is not None:
            x_agg = x_agg * active_mask.unsqueeze(-1).float()

        if self.use_global_context:
            if active_mask is not None:
                num_active = active_mask.sum(dim=1, keepdim=True).clamp(min=1)
                global_feats = x_agg.sum(dim=1) / num_active
            else:
                global_feats = x_agg.mean(dim=1)
            global_feats = global_feats.unsqueeze(1).expand(-1, N, -1)
            x_agg = torch.cat([x_agg, global_feats], dim=-1)

        if cond is not None:
            cond_expanded = cond.unsqueeze(1).expand(-1, N, -1)
            x_agg = torch.cat([x_agg, cond_expanded], dim=-1)

        h = self.input_proj(x_agg)
        if active_mask is not None:
            h = h * active_mask.unsqueeze(-1).float()
        h = self.hidden_norm(h)
        if active_mask is not None:
            # LayerNorm can reintroduce non-zero values on inactive nodes; re-mask.
            h = h * active_mask.unsqueeze(-1).float()

        scores = self.score_proj(h).squeeze(-1)
        if active_mask is not None:
            mask_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~active_mask.bool(), mask_value)
        return scores


class StridedPoolingGNN(nn.Module):
    """
    Strided lightweight GNN for computing node selection scores on sparse signals.
    
    Similar to PoolingGNN but uses StridedTAGConv to respect signal sparsity.
    Processes neighborhoods at γ×k hops where γ is the stride (spacing between
    active nodes) and k=0,1,...,K.
    
    Args:
        in_channels: Input feature dimension
        hidden_channels: Hidden dimension (default: 32)
        num_layers: Number of GNN layers (default: 2)
        K: TAGConv hop count (default: 2)
        gamma: Stride factor matching signal sparsity (default: 2)
        use_global_context: Whether to include global graph features (default: True)
        cond_dim: Optional conditioning dimension (default: None)
    
    Shape:
        - Input x: (B, T, N, F) - zero-padded signal with ~N/γ active nodes
        - Edge index: (2, E) - same graph for all levels
        - Active mask: (B, N) - binary mask of active nodes
        - Conditioning: (B, cond_dim) optional
        - Output scores: (B, N) - selection probability per node
    
    Example:
        >>> # For level 1 with stride=2 signal
        >>> pooling_gnn = StridedPoolingGNN(
        ...     in_channels=64,
        ...     hidden_channels=32,
        ...     gamma=2,  # Match signal sparsity
        ... )
        >>> x = torch.randn(4, 10, 100, 64)  # (B, T, N, F)
        >>> edge_index = torch.randint(0, 400, (2, 500))
        >>> active_mask = torch.zeros(4, 100, dtype=torch.bool)
        >>> active_mask[:, ::2] = True  # Every 2nd node active
        >>> scores = pooling_gnn(x, edge_index, active_mask=active_mask)
        >>> # scores: (4, 100) - importance score per node
        >>> # GNN processes 0, 2, 4-hop neighborhoods matching signal sparsity
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        num_layers: int = 2,
        K: int = 2,
        gamma: int = 2,
        use_global_context: bool = True,
        cond_dim: Optional[int] = None,
        temporal_score_agg: Literal['mean', 'last', 'ema', 'attention'] = 'mean',
        ema_alpha_init: float = 0.8,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.gamma = gamma
        self.use_global_context = use_global_context
        self.temporal_score_agg = temporal_score_agg
        self.ema_alpha_init = ema_alpha_init

        if temporal_score_agg not in {'mean', 'last', 'ema', 'attention'}:
            raise ValueError(
                f"Unknown temporal_score_agg='{temporal_score_agg}'. "
                "Expected one of: 'mean', 'last', 'ema', 'attention'."
            )
        if temporal_score_agg == 'ema':
            if not (0.0 < ema_alpha_init < 1.0):
                raise ValueError(
                    f"ema_alpha_init must be in (0, 1), got {ema_alpha_init}."
                )
            init_logit = torch.logit(torch.tensor(ema_alpha_init, dtype=torch.float32))
            self.ema_alpha_logit = nn.Parameter(init_logit)
        else:
            self.register_parameter('ema_alpha_logit', None)
        if temporal_score_agg == 'attention':
            self.temporal_attn_proj = nn.Linear(in_channels, 1)
        else:
            self.temporal_attn_proj = None
        
        # Input projection
        input_dim = in_channels
        if cond_dim is not None:
            input_dim += cond_dim
        if use_global_context:
            input_dim += in_channels
        
        self.input_proj = nn.Linear(input_dim, hidden_channels)
        
        # Strided GNN layers respecting signal sparsity
        self.conv_layers = nn.ModuleList([
            StridedTAGConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                K=K,
                gamma=gamma,
            )
            for _ in range(num_layers)
        ])
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels)
            for _ in range(num_layers)
        ])
        
        # Output projection to selection scores
        self.score_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(hidden_channels // 2, 1),
        )

    def _aggregate_temporal(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate (B, T, N, F) over time to (B, N, F)."""
        if self.temporal_score_agg == 'mean':
            return x.mean(dim=1)
        if self.temporal_score_agg == 'last':
            return x[:, -1]
        if self.temporal_score_agg == 'ema':
            alpha = torch.sigmoid(self.ema_alpha_logit).to(dtype=x.dtype, device=x.device)
            x_agg = x[:, 0]
            for t in range(1, x.size(1)):
                x_agg = alpha * x_agg + (1.0 - alpha) * x[:, t]
            return x_agg
        # attention
        logits = self.temporal_attn_proj(x).squeeze(-1)  # (B, T, N)
        attn = torch.softmax(logits, dim=1).unsqueeze(-1)  # (B, T, N, 1)
        return (attn * x).sum(dim=1)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        active_mask: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute selection scores for each node using strided convolutions.
        
        Args:
            x: Node features (B, T, N, F) - zero-padded signal
            edge_index: Edge indices (2, E) - same for all batches
            edge_weight: Optional edge weights (E,)
            active_mask: Binary mask (B, N) - which nodes are currently active
            cond: Optional conditioning (B, cond_dim)
        
        Returns:
            Selection scores (B, N) - importance score per node
        """
        B, T, N, F = x.shape
        
        # Aggregate features across time for scoring.
        x_agg = self._aggregate_temporal(x)  # (B, N, F)
        
        # CRITICAL: Zero out inactive nodes BEFORE GNN processing
        if active_mask is not None:
            x_agg = x_agg * active_mask.unsqueeze(-1).float()
        
        # Compute global context (only from active nodes)
        if self.use_global_context:
            if active_mask is not None:
                num_active = active_mask.sum(dim=1, keepdim=True).clamp(min=1)
                global_feats = x_agg.sum(dim=1) / num_active  # (B, F)
            else:
                global_feats = x_agg.mean(dim=1)  # (B, F)
            
            # Broadcast to all nodes
            global_feats = global_feats.unsqueeze(1).expand(-1, N, -1)  # (B, N, F)
            x_agg = torch.cat([x_agg, global_feats], dim=-1)
        
        # Add conditioning if provided
        if cond is not None:
            cond_expanded = cond.unsqueeze(1).expand(-1, N, -1)  # (B, N, cond_dim)
            x_agg = torch.cat([x_agg, cond_expanded], dim=-1)
        
        # Input projection
        h = self.input_proj(x_agg)  # (B, N, hidden)
        
        # CRITICAL: Zero out inactive nodes after projection
        if active_mask is not None:
            h = h * active_mask.unsqueeze(-1).float()
        
        # Apply strided GNN layers with residuals and explicit masking
        h = h.unsqueeze(1)  # (B, 1, N, hidden) - add dummy time dimension
        
        for conv, norm in zip(self.conv_layers, self.norms):
            h_new = conv(h, edge_index, edge_weight)
            
            # CRITICAL: Mask outputs AFTER each GNN layer
            # This prevents message passing from affecting inactive nodes
            if active_mask is not None:
                mask_4d = active_mask.unsqueeze(1).unsqueeze(-1).float()  # (B, 1, N, 1)
                h_new = h_new * mask_4d
            
            h_new = norm(h_new)
            h = torch.nn.functional.silu(h_new) + h
            
            # Mask residual output too
            if active_mask is not None:
                h = h * mask_4d
        
        h = h.squeeze(1)  # (B, N, hidden)
        
        # Compute selection scores
        scores = self.score_proj(h).squeeze(-1)  # (B, N)
        
        # Apply mask using large negative value for inactive nodes
        # This ensures they can't be selected
        # Use dtype-aware min value to support FP16 (AMP)
        if active_mask is not None:
            mask_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~active_mask.bool(), mask_value)
        
        return scores


class LearnableGraphPool(nn.Module):
    """
    Learnable graph pooling via node selection.
    
    Uses a lightweight GNN to learn which nodes to keep based on their
    importance. Supports soft selection (training) and hard selection (inference).
    
    **Key Design:**
    - Uses the SAME graph structure (edge_index) across all levels
    - Only selects which nodes are "active" at this level
    - Nested selection: only selects from currently active nodes
    - Inactive nodes have zero features
    
    Args:
        in_channels: Input feature dimension
        pooling_ratio: Fraction of active nodes to keep (default: 0.5)
        hidden_channels: Hidden dimension for pooling GNN (default: 32)
        num_gnn_layers: Number of GNN layers (default: 2)
        K: TAGConv hop count (default: 2)
        gamma: Stride of input signal - spacing between active nodes (default: 1)
            If gamma > 1, uses StridedPoolingGNN with strided convolutions
            If gamma = 1, uses standard PoolingGNN (dense signal)
        use_pooling_gnn: If False, disables selector GNN entirely and uses
            temporal aggregation + per-node MLP for scoring (default: True).
        use_strided_pooling_gnn: If True and gamma>1, use StridedPoolingGNN; if False,
            always use standard PoolingGNN for selector scoring (default: True).
        selection_mode: 'soft', 'hard', or 'ste' (default: 'soft')
            - 'soft': Soft mask in training, hard in eval
            - 'hard': Hard top-k mask in both training and eval
            - 'ste': Straight-through estimator (hard forward, soft backward)
        temperature: Initial temperature for soft/STE masks (default: 1.0)
        temperature_schedule: Temperature schedule over training steps (default: 'constant')
        temperature_min: Minimum temperature reached after annealing (default: 0.1)
        temperature_anneal_steps: Number of steps to anneal temperature (default: 0 = disabled)
        temperature_warmup_steps: Number of steps to keep initial temperature before annealing
            (default: 0)
        temperature_anneal_epochs: Deprecated alias for temperature_anneal_steps.
        temperature_warmup_epochs: Deprecated alias for temperature_warmup_steps.
        entropy_reg_weight: Weight for entropy regularization term to avoid selector collapse
            (default: 0.0). This is exposed via get_selector_aux_loss().
        cond_dim: Optional conditioning dimension (default: None)
        hidden_norm_type: Hidden normalization for MLP selector path when
            use_pooling_gnn=False. One of {'layer', 'none'} (default: 'layer').
        score_gain_init: Initial value for the multiplicative score gain applied
            before sigmoid(score / temperature). Addresses the ~10× per-level
            score attenuation in deep selectors by amplifying the effective logit
            dynamic range. Default 1.0 (identity).
        learnable_score_gain: If True, score gain is a learnable parameter
            (exp(log_score_gain)); if False, it is a fixed buffer. Set True to
            let each selector level self-calibrate its score range. Default False.
    
    Shape:
        - Input x: (B, T, N, F)
        - Edge index: (2, E) - stays the same!
        - Active mask: (B, N) - which nodes are currently active
        - Output x_pooled: (B, T, N, F) - same shape, but fewer active nodes
        - Output new_active_mask: (B, N) - updated active mask
        - Output selection_indices: (B, max_k) - which nodes were selected
    
    Example:
        >>> pool = LearnableGraphPool(
        ...     in_channels=64,
        ...     pooling_ratio=0.5,
        ...     selection_mode='soft',
        ... )
        >>> x = torch.randn(4, 10, 100, 64)  # (B, T, N, F)
        >>> edge_index = torch.randint(0, 400, (2, 500))
        >>> x_pooled, new_mask, indices = pool(x, edge_index)
        >>> # x_pooled: (4, 10, 100, 64) - ~50 nodes active, rest zero
        >>> # new_mask: (4, 100) - binary mask of selected nodes
        >>> # indices: (4, 50) - indices of selected nodes
    """
    
    def __init__(
        self,
        in_channels: int,
        pooling_ratio: float = 0.5,
        hidden_channels: int = 32,
        num_gnn_layers: int = 2,
        K: int = 2,
        gamma: int = 1,
        use_pooling_gnn: bool = True,
        use_strided_pooling_gnn: bool = True,
        selection_mode: Literal['soft', 'hard', 'ste'] = 'soft',
        temperature: float = 1.0,
        temperature_schedule: Literal['constant', 'linear', 'cosine'] = 'constant',
        temperature_min: float = 0.1,
        temperature_anneal_steps: int = 0,
        temperature_warmup_steps: int = 0,
        temperature_anneal_epochs: Optional[int] = None,
        temperature_warmup_epochs: Optional[int] = None,
        entropy_reg_weight: float = 0.0,
        cond_dim: Optional[int] = None,
        temporal_score_agg: Literal['mean', 'last', 'ema', 'attention'] = 'mean',
        ema_alpha_init: float = 0.8,
        hidden_norm_type: Literal['layer', 'none'] = 'layer',
        score_gain_init: float = 1.0,
        learnable_score_gain: bool = False,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.pooling_ratio = pooling_ratio
        self.selection_mode = selection_mode
        self.temperature = temperature
        self.temperature_schedule = temperature_schedule
        self.temperature_min = temperature_min

        # Backward compatibility for configs still using epoch-based field names.
        if temperature_anneal_epochs is not None:
            if temperature_anneal_steps not in {0, int(temperature_anneal_epochs)}:
                raise ValueError(
                    "Specify only one of temperature_anneal_steps or "
                    "temperature_anneal_epochs (deprecated alias)."
                )
            temperature_anneal_steps = int(temperature_anneal_epochs)
        if temperature_warmup_epochs is not None:
            if temperature_warmup_steps not in {0, int(temperature_warmup_epochs)}:
                raise ValueError(
                    "Specify only one of temperature_warmup_steps or "
                    "temperature_warmup_epochs (deprecated alias)."
                )
            temperature_warmup_steps = int(temperature_warmup_epochs)

        self.temperature_anneal_steps = temperature_anneal_steps
        self.temperature_warmup_steps = temperature_warmup_steps
        # Expose deprecated aliases as read-only attributes for legacy callers.
        self.temperature_anneal_epochs = self.temperature_anneal_steps
        self.temperature_warmup_epochs = self.temperature_warmup_steps
        self.entropy_reg_weight = entropy_reg_weight
        self.gamma = gamma
        self.use_pooling_gnn = use_pooling_gnn
        self.use_strided_pooling_gnn = use_strided_pooling_gnn
        self.temporal_score_agg = temporal_score_agg
        self.ema_alpha_init = ema_alpha_init
        self.hidden_norm_type = hidden_norm_type
        self.current_step = 0
        self.collect_heavy_diagnostics = True
        self.collect_sampling_probe = False

        # Per-level learnable logit gain: effective_score = score * exp(log_score_gain).
        # This lets each selector level self-calibrate its score dynamic range,
        # addressing the ~10× per-level score attenuation observed in deep selectors.
        if score_gain_init <= 0:
            raise ValueError(f"score_gain_init must be > 0, got {score_gain_init}")
        _init_log_gain = math.log(score_gain_init)
        if learnable_score_gain:
            self.log_score_gain = nn.Parameter(
                torch.tensor(_init_log_gain, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                'log_score_gain',
                torch.tensor(_init_log_gain, dtype=torch.float32),
            )

        if selection_mode not in {'soft', 'hard', 'ste'}:
            raise ValueError(
                f"Unknown selection_mode='{selection_mode}'. "
                "Expected one of: 'soft', 'hard', 'ste'."
            )
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if temperature_min <= 0:
            raise ValueError(f"temperature_min must be > 0, got {temperature_min}")
        if temperature_schedule not in {'constant', 'linear', 'cosine'}:
            raise ValueError(
                f"Unknown temperature_schedule='{temperature_schedule}'. "
                "Expected one of: 'constant', 'linear', 'cosine'."
            )
        if temperature_anneal_steps < 0:
            raise ValueError(
                f"temperature_anneal_steps must be >= 0, got {temperature_anneal_steps}"
            )
        if temperature_warmup_steps < 0:
            raise ValueError(
                f"temperature_warmup_steps must be >= 0, got {temperature_warmup_steps}"
            )
        if entropy_reg_weight < 0:
            raise ValueError(f"entropy_reg_weight must be >= 0, got {entropy_reg_weight}")

        # Diagnostics and auxiliary losses from the most recent forward pass.
        self.reset_selector_state()
        
        if use_pooling_gnn:
            # Use strided pooling GNN only when explicitly requested; for large batched
            # graphs this can be memory-heavy due dense gamma-hop adjacency powers.
            if gamma > 1 and use_strided_pooling_gnn:
                self.pooling_gnn = StridedPoolingGNN(
                    in_channels=in_channels,
                    hidden_channels=hidden_channels,
                    num_layers=num_gnn_layers,
                    K=K,
                    gamma=gamma,
                    use_global_context=True,
                    cond_dim=cond_dim,
                    temporal_score_agg=temporal_score_agg,
                    ema_alpha_init=ema_alpha_init,
                )
            else:
                self.pooling_gnn = PoolingGNN(
                    in_channels=in_channels,
                    hidden_channels=hidden_channels,
                    num_layers=num_gnn_layers,
                    K=K,
                    use_global_context=True,
                    cond_dim=cond_dim,
                    temporal_score_agg=temporal_score_agg,
                    ema_alpha_init=ema_alpha_init,
                )
            self.selector_head = None
        else:
            self.pooling_gnn = None
            self.selector_head = TemporalMLPSelectorHead(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                use_global_context=True,
                cond_dim=cond_dim,
                temporal_score_agg=temporal_score_agg,
                ema_alpha_init=ema_alpha_init,
                hidden_norm_type=hidden_norm_type,
            )
    
    def reset_selector_state(self) -> None:
        """Reset cached auxiliary losses/diagnostics for the next forward pass."""
        self._selector_called_in_last_forward = False
        self._last_aux_loss: Optional[torch.Tensor] = None
        self._last_diagnostics: dict = {}

    def set_training_step(self, step: int) -> None:
        """Set the current global training step used by step-based schedules."""
        self.current_step = max(0, int(step))

    def set_training_epoch(self, epoch: int) -> None:
        """Backward-compatible alias; maps epoch input to current training step."""
        self.set_training_step(epoch)

    def _current_temperature(self) -> float:
        """Resolve current temperature according to schedule and current_step."""
        if self.temperature_schedule == 'constant' or self.temperature_anneal_steps <= 0:
            return float(self.temperature)

        if self.current_step < self.temperature_warmup_steps:
            return float(self.temperature)

        progress = (self.current_step - self.temperature_warmup_steps) / max(
            self.temperature_anneal_steps, 1
        )
        progress = min(max(progress, 0.0), 1.0)

        if self.temperature_schedule == 'linear':
            temp = self.temperature + progress * (self.temperature_min - self.temperature)
        elif self.temperature_schedule == 'cosine':
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            temp = self.temperature_min + (self.temperature - self.temperature_min) * cosine
        else:
            temp = self.temperature

        # Numerical guard.
        return float(max(temp, 1e-6))

    def get_current_temperature(self) -> float:
        """Public getter for the current scheduled temperature."""
        return self._current_temperature()

    def _batched_topk_indices(self, scores: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """
        Per-sample top-k indices.

        Unlike torch.topk(..., k=max_k) for all rows, this respects each sample's own k[b].
        Output is padded to (B, max_k) for compatibility.
        """
        B, _ = scores.shape
        max_k = int(k.max().item())
        top_k_indices = torch.zeros(B, max_k, device=scores.device, dtype=torch.long)

        for b in range(B):
            k_b = int(k[b].item())
            _, idx_b = torch.topk(scores[b], k_b, dim=0)
            top_k_indices[b, :k_b] = idx_b

        return top_k_indices

    def _indices_to_hard_mask(self, top_k_indices: torch.Tensor, k: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Convert per-sample top-k indices to a dense hard mask (B, N)."""
        B = top_k_indices.size(0)
        hard_mask = torch.zeros(B, num_nodes, device=top_k_indices.device, dtype=torch.float32)
        for b in range(B):
            k_b = int(k[b].item())
            hard_mask[b, top_k_indices[b, :k_b]] = 1.0
        return hard_mask

    @staticmethod
    def _entropy_regularizer(soft_mask: torch.Tensor, active_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute normalized entropy and anti-collapse regularizer.

        Returns:
            entropy_reg: scalar in [0, 1], where lower is more entropic / less collapsed
            entropy_mean_norm: mean normalized entropy in [0, 1]
        """
        eps = 1e-8
        active_probs = soft_mask[active_mask.bool()]
        if active_probs.numel() == 0:
            zero = soft_mask.new_zeros(())
            return zero, zero

        p = active_probs.clamp(min=eps, max=1.0 - eps)
        entropy = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
        max_entropy = math.log(2.0)
        entropy_mean_norm = (entropy / max_entropy).mean()
        entropy_reg = 1.0 - entropy_mean_norm
        return entropy_reg, entropy_mean_norm

    def get_selector_aux_loss(self) -> Optional[torch.Tensor]:
        """Return selector auxiliary loss from the most recent forward pass (if any)."""
        if not self._selector_called_in_last_forward:
            return None
        return self._last_aux_loss

    def get_selector_diagnostics(self) -> dict:
        """Return selector diagnostics from the most recent forward pass."""
        if not self._selector_called_in_last_forward:
            return {}
        return dict(self._last_diagnostics)

    def set_collect_heavy_diagnostics(self, enabled: bool) -> None:
        """
        Enable/disable heavy selector diagnostics.

        Heavy diagnostics include per-graph/per-node tensors used for detailed
        trainer-side aggregation and visualizations.
        """
        self.collect_heavy_diagnostics = bool(enabled)

    def set_collect_sampling_probe(self, enabled: bool) -> None:
        """
        Enable/disable lightweight sampling diagnostics for DDPM phase probing.

        This captures only active masks before/after selection to keep overhead low.
        """
        self.collect_sampling_probe = bool(enabled)

    def compute_scores(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        active_mask: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute selector logits using configured scoring backbone."""
        if self.pooling_gnn is not None:
            return self.pooling_gnn(
                x=x,
                edge_index=edge_index,
                edge_weight=edge_weight,
                active_mask=active_mask,
                cond=cond,
            )
        return self.selector_head(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            active_mask=active_mask,
            cond=cond,
        )

    def apply_score_gain(self, scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply multiplicative score gain and return both scaled scores and gain.

        Args:
            scores: Raw selector logits (B, N)

        Returns:
            scaled_scores: Gain-scaled logits (B, N)
            score_gain: Scalar gain tensor on the same dtype/device as scores
        """
        score_gain = torch.exp(self.log_score_gain.clamp(-5.0, 5.0)).to(
            dtype=scores.dtype,
            device=scores.device,
        )
        return scores * score_gain, score_gain
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        active_mask: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pool graph features by selecting important nodes.
        
        Args:
            x: Node features (B, T, N, F)
            edge_index: Edge indices (2, E) - same graph
            edge_weight: Optional edge weights (E,)
            active_mask: Binary mask (B, N) indicating currently active nodes
            cond: Optional conditioning (B, cond_dim)
        
        Returns:
            x_pooled: (B, T, N, F) - features with inactive nodes zeroed
            new_active_mask: (B, N) - updated binary mask of active nodes
            top_k_indices: (B, max_k) - indices of selected nodes
            scores: (B, N) - selection scores from pooling GNN
        """
        B, T, N, F = x.shape
        self.reset_selector_state()
        self._selector_called_in_last_forward = True
        
        # If no active mask provided, all nodes are active
        if active_mask is None:
            active_mask = torch.ones(B, N, device=x.device, dtype=torch.bool)
        
        # ENSURE inactive nodes have zero features in input
        mask_4d = active_mask.unsqueeze(1).unsqueeze(-1).float()  # (B, 1, N, 1)
        x = x * mask_4d
        
        # Count active nodes per batch
        num_active = active_mask.sum(dim=1)  # (B,)
        
        # Compute number of nodes to keep (per batch)
        k = torch.clamp((num_active.float() * self.pooling_ratio).long(), min=1)
        
        # Compute selection scores from configured selector scorer.
        scores = self.compute_scores(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            active_mask=active_mask,
            cond=cond,
        )  # (B, N)

        # Apply learnable per-level score gain before routing and soft masks.
        # effective_score = raw_score * exp(log_score_gain)
        # This amplifies the score dynamic range for deeper levels where the
        # selector network produces very small logit spreads (~10× attenuation
        # per level), preventing gradient desert in sigmoid(score / τ).
        scores, score_gain = self.apply_score_gain(scores)

        # Use the same logits for hard top-k routing and soft surrogate masks in STE.
        # Keep optional Gumbel exploration for legacy 'soft' mode during training.
        ranking_scores = scores
        if self.training and self.selection_mode == 'soft':
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-10) + 1e-10)
            ranking_scores = scores + gumbel_noise

        top_k_indices = self._batched_topk_indices(ranking_scores, k)
        hard_mask = self._indices_to_hard_mask(top_k_indices, k, N)
        hard_mask = hard_mask * active_mask.float()

        current_temperature = self._current_temperature()
        soft_probs = torch.sigmoid(scores / current_temperature)
        soft_mask = soft_probs * active_mask.float()

        if self.training:
            if self.selection_mode == 'soft':
                selection_mask = soft_mask
            elif self.selection_mode == 'ste':
                # Hard routing in forward; soft surrogate gradients in backward.
                selection_mask = hard_mask + soft_mask - soft_mask.detach()
            else:  # hard
                selection_mask = hard_mask
        else:
            # Always hard in eval for deterministic topology.
            selection_mask = hard_mask
        
        # Apply mask to features
        mask_expanded = selection_mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, N, 1)
        x_pooled = x * mask_expanded  # (B, T, N, F)
        
        # Create new active mask (hard, for next level)
        new_active_mask = hard_mask.bool()
        
        # Ensure nested selection (only from previously active nodes)
        new_active_mask = new_active_mask & active_mask

        # Entropy regularization: minimize entropy → decisive selection.
        entropy_reg, entropy_mean_norm = self._entropy_regularizer(soft_mask, active_mask)
        if self.training and self.entropy_reg_weight > 0:
            # Minimize entropy_mean_norm → push soft_mask toward 0 or 1
            # (decisive selection).  The original formulation minimized
            # (1 - entropy_mean_norm), which rewarded uniform / random-dropout
            # masks and actively prevented the selector from learning.
            self._last_aux_loss = entropy_mean_norm * self.entropy_reg_weight
        else:
            self._last_aux_loss = None

        # Diagnostics for logging.
        active_scores = scores[active_mask]
        active_soft_mask = soft_mask[active_mask]
        selected_counts = new_active_mask.sum(dim=1).float()
        active_counts = active_mask.sum(dim=1).float().clamp(min=1.0)
        selected_ratio = selected_counts / active_counts
        soft_mask_mean_per_graph = soft_mask.sum(dim=1) / active_counts
        # Per-node counts across the batch for level-wise histogram diagnostics.
        selected_node_counts_per_graph = new_active_mask.float()
        selected_node_counts = selected_node_counts_per_graph.sum(dim=0)
        diag = {
            "selection_mode": self.selection_mode,
            "temperature": float(current_temperature),
            "entropy_mean_norm": float(entropy_mean_norm.detach().item()),
            "entropy_reg": float(entropy_reg.detach().item()),
            "entropy_reg_weight": float(self.entropy_reg_weight),
            "num_active_mean": float(active_counts.mean().detach().item()),
            "num_selected_mean": float(selected_counts.mean().detach().item()),
            "selected_ratio_mean": float(selected_ratio.mean().detach().item()),
            "score_mean_active": float(active_scores.mean().detach().item()) if active_scores.numel() > 0 else 0.0,
            "score_std_active": float(active_scores.std(unbiased=False).detach().item()) if active_scores.numel() > 1 else 0.0,
            "soft_mask_mean_active": float(active_soft_mask.mean().detach().item()) if active_soft_mask.numel() > 0 else 0.0,
            "soft_mask_std_active": float(active_soft_mask.std(unbiased=False).detach().item()) if active_soft_mask.numel() > 1 else 0.0,
            "score_gain": float(score_gain.detach().item()),
            "num_graphs": float(B),
        }
        if self.collect_heavy_diagnostics:
            # Tensor diagnostics consumed by trainer-level visualization utilities.
            diag.update({
                "selected_node_counts": selected_node_counts.detach(),
                "selected_node_counts_per_graph": selected_node_counts_per_graph.detach(),
                "selected_ratio_per_graph": selected_ratio.detach(),
                "scores_per_graph": scores.detach(),
                "active_mask_per_graph": active_mask.detach(),
                "soft_mask_per_graph": soft_mask.detach(),
                "soft_mask_mean_per_graph": soft_mask_mean_per_graph.detach(),
            })
        if self.collect_sampling_probe:
            # Lightweight probe tensors for DDPM sampling-phase diagnostics.
            diag.update({
                "probe_active_mask_in": active_mask.detach(),
                "probe_active_mask_out": new_active_mask.detach(),
            })
        self._last_diagnostics = diag
        
        return x_pooled, new_active_mask, top_k_indices, scores


class GraphUnpool(nn.Module):
    """
    Graph unpooling via concatenation with skip connections.
    
    Simply concatenates pooled features with skip connection features.
    Since both have the same shape (B, T, N, F) and inactive nodes are zero,
    no explicit "unpooling" operation is needed.
    
    Shape:
        - Input x_pooled: (B, T, N, F_pooled) - pooled features (sparse)
        - Skip connection x_skip: (B, T, N, F_skip) - from encoder
        - Output: (B, T, N, F_pooled + F_skip)
    
    Example:
        >>> unpool = GraphUnpool()
        >>> x_pooled = torch.randn(4, 10, 100, 64)  # From decoder
        >>> x_skip = torch.randn(4, 10, 100, 64)    # From encoder
        >>> x = unpool(x_pooled, x_skip)
        >>> # x: (4, 10, 100, 128) - concatenated features
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        x_pooled: torch.Tensor,
        x_skip: torch.Tensor,
    ) -> torch.Tensor:
        """
        Unpool features and combine with skip connection.
        
        Args:
            x_pooled: Pooled features (B, T, N, F_pooled)
            x_skip: Skip connection features (B, T, N, F_skip)
        
        Returns:
            x: Concatenated features (B, T, N, F_pooled + F_skip)
        """
        # Simply concatenate - both have same shape
        # x_pooled already has correct node positions (inactive nodes are zero)
        x = torch.cat([x_pooled, x_skip], dim=-1)
        
        return x


class GraphMaxPool(nn.Module):
    """
    Graph max-pooling over γ-hop neighborhoods.
    
    Analogous to max-pooling in CNNs, but operates on graph neighborhoods
    instead of spatial grids. For each node, computes the max over its
    γ-hop neighborhood, then downsamples by selecting nodes.
    
    **Operation:**
    1. Compute γ-hop neighborhood for each node via graph powers
    2. Apply max-pooling over neighborhoods (per feature channel)
    3. Downsample by selecting nodes (every γ-th node or learned)
    
    Args:
        gamma: Hop count for neighborhood (downsampling factor)
        selection_method: How to select nodes after pooling
            - 'stride': Select every γ-th node (deterministic)
            - 'learned': Use learnable selection (default)
            - 'random': Random sampling (for data augmentation)
        in_channels: Input feature dimension (required for learned selection)
        pooling_ratio: Override pooling ratio (default: 1/γ)
        selector_kwargs: Additional kwargs for LearnableGraphPool (only for learned selection)
            e.g., {'hidden_channels': 32, 'num_gnn_layers': 2, 'K': 2}
    
    Shape:
        - Input x: (B, T, N, F)
        - Edge index: (2, E)
        - Output x_pooled: (B, T, N', F) where N' ≈ N/γ
        - Output active_mask: (B, N') - which nodes are active
    
    Example:
        >>> pool = GraphMaxPool(gamma=2, selection_method='learned')
        >>> x = torch.randn(4, 10, 100, 64)
        >>> edge_index = torch.randint(0, 400, (2, 500))
        >>> x_pooled, mask = pool(x, edge_index)
        >>> # x_pooled: (4, 10, ~50, 64) - max-pooled then downsampled
    """
    
    def __init__(
        self,
        gamma: int = 2,
        selection_method: Literal['stride', 'learned', 'random'] = 'learned',
        in_channels: Optional[int] = None,
        pooling_ratio: Optional[float] = None,
        selector_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        
        self.gamma = gamma
        self.selection_method = selection_method
        
        # If using learned selection, create a LearnableGraphPool
        if selection_method == 'learned':
            assert in_channels is not None, \
                "in_channels required for learned selection"
            
            pooling_ratio = pooling_ratio or (1.0 / gamma)
            selector_kwargs = selector_kwargs or {}
            
            self.selector = LearnableGraphPool(
                in_channels=in_channels,
                pooling_ratio=pooling_ratio,
                **selector_kwargs
            )
        else:
            self.selector = None
    
    def _compute_gamma_hop_adjacency(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        gamma: int,
    ) -> torch.Tensor:
        """
        Compute γ-hop adjacency matrix via matrix powers.
        
        A^γ gives γ-hop connectivity (with path counting).
        We use this to determine which nodes are in each node's
        γ-hop neighborhood.
        
        Args:
            edge_index: Edge indices (2, E)
            num_nodes: Number of nodes
            gamma: Number of hops
        
        Returns:
            gamma_hop_adj: Sparse adjacency matrix for γ-hop neighbors
        """
        from torch_geometric.utils import to_dense_adj, dense_to_sparse
        
        # Convert to dense adjacency (includes batch dimension)
        # edge_index should be for single graph, so we need to extract it
        adj_dense = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]  # (N, N)
        
        # Add self-loops (node is in its own 0-hop neighborhood)
        adj_dense = adj_dense + torch.eye(num_nodes, device=adj_dense.device)
        
        # Compute A^γ (γ-hop adjacency)
        adj_gamma = adj_dense.clone()
        for _ in range(gamma - 1):
            adj_gamma = torch.matmul(adj_gamma, adj_dense)
        
        # Binarize (we only care about connectivity, not path count)
        adj_gamma = (adj_gamma > 0).float()
        
        # Convert back to sparse edge_index format
        edge_index_gamma, _ = dense_to_sparse(adj_gamma)
        
        return edge_index_gamma


    def _max_pool_neighborhoods(
        self,
        x: torch.Tensor,
        edge_index_gamma: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """
        Apply max-pooling over γ-hop neighborhoods for each node.
        
        For each node i, compute:
            x_pooled[i, f] = max_{j ∈ N_γ(i)} x[j, f]
        
        where N_γ(i) is the γ-hop neighborhood of node i.
        
        Args:
            x: Node features (B, T, N, F)
            edge_index_gamma: γ-hop edge indices (2, E_gamma)
            num_nodes: Number of nodes
        
        Returns:
            x_pooled: Max-pooled features (B, T, N, F)
        """
        B, T, N, F = x.shape
        device = x.device
        
        # Convert edge_index_gamma to dense adjacency for easier max-pooling
        from torch_geometric.utils import to_dense_adj
        
        # Get adjacency matrix (N, N)
        adj = to_dense_adj(edge_index_gamma, max_num_nodes=N)[0]  # (N, N)
        
        # For each target node, find which source nodes are in its neighborhood
        # adj[i, j] = 1 means node j is in node i's neighborhood
        
        # Reshape x for broadcasting: (B, T, N, F) -> (B, T, N, 1, F)
        x_expanded = x.unsqueeze(3)  # (B, T, N, 1, F)
        
        # For each target node i, we want max over all j where adj[i, j] = 1
        # Broadcast x to (B, T, 1, N, F) and adj to (1, 1, N, N, 1)
        x_source = x.unsqueeze(2)  # (B, T, 1, N, F) - source nodes
        adj_mask = adj.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # (1, 1, N, N, 1)
        
        # Mask out non-neighbors with -inf (so they don't affect max)
        x_masked = torch.where(
            adj_mask.bool(),
            x_source,
            torch.tensor(float('-inf'), device=device),
        )  # (B, T, N, N, F) - target x source x features
        
        # Take max over source dimension (dim=3)
        x_pooled, _ = torch.max(x_masked, dim=3)  # (B, T, N, F)
        
        # Handle case where node has no neighbors (max would be -inf)
        # Use original value for isolated nodes
        x_pooled = torch.where(
            torch.isinf(x_pooled),
            x,
            x_pooled,
        )
        
        return x_pooled
    
    def _stride_selection(
        self,
        x: torch.Tensor,
        gamma: int,
        current_active_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select every γ-th node (stride-based downsampling).
        
        Args:
            x: Max-pooled features (B, T, N, F)
            gamma: Stride factor
            current_active_mask: Current active nodes (B, N) to select from
        
        Returns:
            x_selected: Selected features (B, T, N', F)
            active_mask: Binary mask (B, N) of selected nodes
        """
        B, T, N, F = x.shape
        
        # Create active mask
        active_mask = torch.zeros(B, N, device=x.device, dtype=torch.bool)
        if current_active_mask is None:
            # Select every γ-th node from all nodes
            selected_indices = torch.arange(0, N, gamma, device=x.device)
            active_mask[:, selected_indices] = True
        else:
            # Select every γ-th node from currently active set.
            for b in range(B):
                active_indices = torch.where(current_active_mask[b])[0]
                selected_from_active = active_indices[::gamma]
                active_mask[b, selected_from_active] = True
        
        # Zero out non-selected nodes
        mask_4d = active_mask.unsqueeze(1).unsqueeze(-1).float()
        x_selected = x * mask_4d
        
        return x_selected, active_mask
    
    def _random_selection(
        self,
        x: torch.Tensor,
        gamma: int,
        active_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly select N/γ nodes.
        
        Args:
            x: Max-pooled features (B, T, N, F)
            gamma: Downsampling factor
            active_mask: Current active nodes (B, N)
        
        Returns:
            x_selected: Selected features (B, T, N, F)
            new_active_mask: Binary mask (B, N) of selected nodes
        """
        B, T, N, F = x.shape
        
        if active_mask is None:
            active_mask = torch.ones(B, N, device=x.device, dtype=torch.bool)
        
        # Number of nodes to keep
        num_active = active_mask.sum(dim=1)
        k = torch.clamp((num_active.float() / gamma).long(), min=1)
        
        max_k = k.max().item()
        
        # Random selection from active nodes
        new_active_mask = torch.zeros(B, N, device=x.device, dtype=torch.bool)
        
        for b in range(B):
            active_indices = torch.where(active_mask[b])[0]
            if len(active_indices) > 0:
                # Randomly select k[b] nodes
                perm = torch.randperm(len(active_indices), device=x.device)[:k[b]]
                selected = active_indices[perm]
                new_active_mask[b, selected] = True
        
        # Zero out non-selected nodes
        mask_4d = new_active_mask.unsqueeze(1).unsqueeze(-1).float()
        x_selected = x * mask_4d
        
        return x_selected, new_active_mask
    
    @staticmethod
    def _selected_indices_from_mask(active_mask: torch.Tensor) -> torch.Tensor:
        B, _ = active_mask.shape
        max_selected = int(active_mask.sum(dim=1).max().item())
        if max_selected <= 0:
            return torch.empty((B, 0), dtype=torch.long, device=active_mask.device)

        selected_indices = torch.full(
            (B, max_selected),
            -1,
            dtype=torch.long,
            device=active_mask.device,
        )
        for b in range(B):
            idx = torch.where(active_mask[b])[0]
            selected_indices[b, :idx.numel()] = idx
        return selected_indices
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        active_mask: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Apply graph max-pooling over γ-hop neighborhoods, then downsample.
        
        Args:
            x: Node features (B, T, N, F)
            edge_index: Edge indices (2, E)
            edge_weight: Optional edge weights (not used for max-pooling)
            active_mask: Binary mask (B, N) of currently active nodes
            cond: Optional conditioning for learned selection
        
        Returns:
            x_pooled: Pooled and downsampled features (B, T, N, F)
            new_active_mask: Binary mask (B, N) of selected nodes
            selected_indices: Indices of selected nodes (B, k) or None
            neighborhoods: Always None (for compatibility with StridedGraphMaxPool)
        """
        B, T, N, F = x.shape
        
        if active_mask is None:
            active_mask = torch.ones(B, N, device=x.device, dtype=torch.bool)
        
        # CRITICAL: Zero out inactive nodes before pooling
        mask_4d = active_mask.unsqueeze(1).unsqueeze(-1).float()
        x = x * mask_4d
        
        # Step 1: Compute γ-hop adjacency
        edge_index_gamma = self._compute_gamma_hop_adjacency(
            edge_index, N, self.gamma
        )
        
        # Step 2: Apply max-pooling over γ-hop neighborhoods
        x_maxpooled = self._max_pool_neighborhoods(x, edge_index_gamma, N)
        
        # CRITICAL: Re-apply mask after max-pooling
        # (max-pooling might have propagated values to inactive nodes)
        x_maxpooled = x_maxpooled * mask_4d
        
        # Step 3: Downsample by selecting nodes
        if self.selection_method == 'learned':
            # Learnable selector returns (x_pooled, new_active_mask, selected_indices, scores)
            x_pooled, new_active_mask, selected_indices, scores = self.selector(
                x_maxpooled,
                edge_index,
                edge_weight=edge_weight,
                active_mask=active_mask,
                cond=cond,
            )
        elif self.selection_method == 'stride':
            x_pooled, new_active_mask = self._stride_selection(
                x_maxpooled, self.gamma, current_active_mask=active_mask
            )
            selected_indices = self._selected_indices_from_mask(new_active_mask)
            scores = None
        elif self.selection_method == 'random':
            x_pooled, new_active_mask = self._random_selection(
                x_maxpooled, self.gamma, active_mask
            )
            selected_indices = self._selected_indices_from_mask(new_active_mask)
            scores = None
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")
        
        # Return 4-tuple for consistency with StridedGraphMaxPool
        # (when learned selection is used, the 4th element contains selection scores)
        return x_pooled, new_active_mask, selected_indices, scores


class NeighborhoodFeaturePooling(nn.Module):
    """
    Standalone neighborhood feature aggregation via max or average pooling.

    No learnable parameters — this module is pure aggregation logic.
    The two aggregations use different implementations:
      - ``max``: sparse iterative propagation over ``K * stride`` hops.
      - ``avg``: legacy exact strided-frontier adjacency + dense broadcast.

    **Two-stage pipeline inside StridedGraphMaxPool:**
        Stage 1 (this module): neighborhood aggregation
        Stage 2 (handled by caller): node selection / downsampling

    Args:
        K: Number of neighborhood scales (k = 0, 1, …, K).  K = 0 is a
            no-op identity pass-through.
        stride: Hop stride multiplier. Neighborhoods are at distances
            0, stride, 2·stride, …, K·stride.
        aggregation: ``'max'`` or ``'avg'``.
        include_self: If True (default), the 0-hop (self-loop) layer is
            included in max-propagation (preferred path). If False,
            max-propagation runs without self-loops (relaxed semantics).
        max_dense_gb: Runtime memory guard for the avg path.  If the projected dense
            ``(B, T, N, N, F)`` intermediate exceeds this budget the
            forward call raises ``RuntimeError``.

    Shape:
        - Input x: (B, T, N, F)
        - Input edge_index: (2, E)
        - Output x_pooled: (B, T, N, F)  — same shape, aggregated features
    """

    def __init__(
        self,
        K: int = 2,
        stride: int = 1,
        aggregation: Literal['max', 'avg'] = 'max',
        include_self: bool = True,
        max_dense_gb: float = 8.0,
    ):
        super().__init__()
        self.K = K
        self.stride = stride
        self.aggregation = aggregation
        self.include_self = include_self
        self.max_dense_gb = max_dense_gb

        # Adjacency cache: graph_signature -> edge_index_multi
        self._adj_cache: dict = {}
        # Identity shortcut: (id, data_ptr) -> content_hash
        self._identity_hash_cache: dict = {}

    # ------------------------------------------------------------------
    # Adjacency caching helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _content_hash(edge_index: torch.Tensor) -> bytes:
        """Deterministic hash of tensor values (blake2b over contiguous CPU bytes)."""
        import hashlib
        t = edge_index.detach().contiguous().cpu()
        return hashlib.blake2b(t.numpy().tobytes(), digest_size=16).digest()

    def _get_content_hash(self, edge_index: torch.Tensor) -> bytes:
        """Content hash with identity shortcut to avoid redundant CPU copies."""
        ident_key = (id(edge_index), edge_index.data_ptr())
        h = self._identity_hash_cache.get(ident_key)
        if h is None:
            h = self._content_hash(edge_index)
            self._identity_hash_cache[ident_key] = h
        return h

    def _graph_signature(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> tuple:
        """Build a robust cache key for the adjacency computation."""
        content_hash = self._get_content_hash(edge_index)
        return (
            num_nodes,
            edge_index.device,
            edge_index.dtype,
            tuple(edge_index.shape),
            self.include_self,
            self.stride,
            self.K,
            content_hash,
        )

    def _get_cached_adjacency(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """Return cached multi-scale adjacency, computing on miss."""
        sig = self._graph_signature(edge_index, num_nodes)
        cached = self._adj_cache.get(sig)
        if cached is not None:
            return cached

        edge_index_multi = self._compute_strided_hop_adjacency(
            edge_index, num_nodes, self.stride, self.K,
        )
        self._adj_cache[sig] = edge_index_multi
        return edge_index_multi

    # ------------------------------------------------------------------
    # Adjacency computation (moved from StridedGraphMaxPool)
    # ------------------------------------------------------------------
    def _compute_strided_hop_adjacency_boolean(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        stride: int,
        K: int,
    ) -> torch.Tensor:
        """
        Backward-compatibility shim for legacy callers.

        Delegates to ``_compute_strided_hop_adjacency``.
        """
        return self._compute_strided_hop_adjacency(
            edge_index=edge_index,
            num_nodes=num_nodes,
            stride=stride,
            K=K,
        )

    def _compute_strided_hop_adjacency(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        stride: int,
        K: int,
    ) -> torch.Tensor:
        """
        Compute union of exactly stride×k-hop adjacencies for k=0,1,...,K.

        Uses algebraic BFS (level-synchronous) in the Boolean semiring.
        Optimization: When stride=1, uses faster matrix powers.

        Args:
            edge_index: Edge indices (2, E)
            num_nodes: Number of nodes
            stride: Stride factor (hop distance multiplier)
            K: Number of scales

        Returns:
            edge_index_multi: Edge indices for multi-scale neighborhood
        """
        from torch_geometric.utils import to_dense_adj, dense_to_sparse

        device = edge_index.device
        valid = (
            (edge_index[0] >= 0) & (edge_index[0] < num_nodes)
            & (edge_index[1] >= 0) & (edge_index[1] < num_nodes)
        )
        edge_index = edge_index[:, valid]

        # Convert to dense adjacency (Boolean matrix)
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]  # (N, N)
        adj = (adj > 0)  # Ensure Boolean

        # Special case: stride=1, use faster matrix powers approach
        if stride == 1:
            # Add self-loops for matrix multiplication
            adj_with_self = adj | torch.eye(num_nodes, device=device, dtype=torch.bool)

            # Start with 0-hop (self-loops only)
            adj_multi = torch.eye(num_nodes, device=device, dtype=torch.bool)

            # Iteratively add k-hop connectivity for k=1,...,K using powers
            adj_power = adj_with_self.clone()
            for k in range(1, K + 1):
                # A^k in Boolean semiring (matrix multiplication with OR/AND)
                if k > 1:
                    adj_power = torch.matmul(adj_power.float(), adj_with_self.float()) > 0

                # Union (⊕ in Boolean semiring)
                adj_multi = adj_multi | adj_power

            # include_self=False: remove diagonal after power accumulation.
            if not self.include_self:
                adj_multi = adj_multi & ~torch.eye(num_nodes, device=device, dtype=torch.bool)

            # Convert back to sparse edge_index format
            edge_index_multi, _ = dense_to_sparse(adj_multi.float())
            return edge_index_multi

        # General case: stride>1, use BFS for exact k×stride-hop distances
        # Initialize multi-scale adjacency
        if self.include_self:
            adj_multi = torch.eye(num_nodes, device=device, dtype=torch.bool)
        else:
            adj_multi = torch.zeros(num_nodes, num_nodes, device=device, dtype=torch.bool)

        # For each target hop distance k×stride where k=1,...,K
        for k in range(1, K + 1):
            target_distance = k * stride

            # BFS to distance target_distance (seed is always I)
            current_layer = torch.eye(num_nodes, device=device, dtype=torch.bool)
            visited = current_layer.clone()

            for hop in range(target_distance):
                next_layer = torch.matmul(current_layer.float(), adj.float()) > 0
                next_layer = next_layer & ~visited
                visited = visited | next_layer
                current_layer = next_layer

            adj_multi = adj_multi | current_layer

        edge_index_multi, _ = dense_to_sparse(adj_multi.float())
        return edge_index_multi

    # ------------------------------------------------------------------
    # Neighborhood pooling
    # ------------------------------------------------------------------
    def _compute_within_hop_adjacency(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        num_hops: int,
    ) -> torch.Tensor:
        """
        Compute adjacency used by iterative max propagation (for return_adj=True).

        This computes reachability after exactly ``num_hops`` walks on:
          - A + I if include_self=True
          - A     if include_self=False (relaxed mode)
        """
        from torch_geometric.utils import to_dense_adj, dense_to_sparse

        device = edge_index.device
        valid = (
            (edge_index[0] >= 0) & (edge_index[0] < num_nodes)
            & (edge_index[1] >= 0) & (edge_index[1] < num_nodes)
        )
        edge_index = edge_index[:, valid]
        adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0].bool()
        eye = torch.eye(num_nodes, device=device, dtype=torch.bool)

        if num_hops <= 0:
            adj_out = eye if self.include_self else torch.zeros_like(adj)
            edge_index_multi, _ = dense_to_sparse(adj_out.float())
            return edge_index_multi

        adj_base = adj | eye if self.include_self else adj
        adj_power = adj_base.clone()
        for _ in range(num_hops - 1):
            adj_power = torch.matmul(adj_power.float(), adj_base.float()) > 0

        edge_index_multi, _ = dense_to_sparse(adj_power.float())
        return edge_index_multi

    @staticmethod
    def _scatter_max_neighbors(
        x_src: torch.Tensor,
        dst: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """Reduce gathered edge features to destination nodes via max."""
        if x_src.size(2) == 0:
            B, T, _, F = x_src.shape
            return torch.full(
                (B, T, num_nodes, F),
                float('-inf'),
                device=x_src.device,
                dtype=x_src.dtype,
            )

        if torch_scatter_scatter is not None:
            return torch_scatter_scatter(
                x_src,
                dst,
                dim=2,
                dim_size=num_nodes,
                reduce='max',
            )

        # Fallback for environments without torch_scatter.
        B, T, E, F = x_src.shape
        out = torch.full(
            (B, T, num_nodes, F),
            float('-inf'),
            device=x_src.device,
            dtype=x_src.dtype,
        )
        dst_idx = dst.view(1, 1, E, 1).expand_as(x_src)
        out.scatter_reduce_(
            2,
            dst_idx,
            x_src,
            reduce='amax',
            include_self=True,
        )
        return out

    def _max_pool_neighborhoods_sparse(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
        num_hops: int,
    ) -> torch.Tensor:
        """
        Sparse iterative max over K*stride hops without dense NxN materialization.

        include_self=True: uses A+I and yields within-hop neighborhoods.
        include_self=False: relaxed mode (A only); cycle-based self-return is allowed.
        """
        from torch_geometric.utils import add_self_loops

        if num_hops <= 0:
            return x

        if self.include_self:
            edge_index_prop, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        else:
            edge_index_prop = edge_index

        valid = (
            (edge_index_prop[0] >= 0) & (edge_index_prop[0] < num_nodes)
            & (edge_index_prop[1] >= 0) & (edge_index_prop[1] < num_nodes)
        )
        edge_index_prop = edge_index_prop[:, valid]

        # Match the legacy dense pooling orientation (target, source) used in this module.
        src = edge_index_prop[1]
        dst = edge_index_prop[0]

        x_state = x
        for _ in range(num_hops):
            x_src = x_state[:, :, src]  # (B, T, E_prop, F)
            x_state = self._scatter_max_neighbors(x_src, dst, num_nodes)

        return x_state

    def _max_pool_neighborhoods(
        self,
        x: torch.Tensor,
        edge_index_multi: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """
        Apply max-pooling over multi-scale neighborhoods for each node.

        For each node i, compute:
            x_pooled[i, f] = max_{j ∈ N_multi(i)} x[j, f]

        Args:
            x: Node features (B, T, N, F)
            edge_index_multi: Multi-scale edge indices (2, E_multi)
            num_nodes: Number of nodes

        Returns:
            x_pooled: Max-pooled features (B, T, N, F)
        """
        B, T, N, F = x.shape
        device = x.device

        from torch_geometric.utils import to_dense_adj

        adj = to_dense_adj(edge_index_multi, max_num_nodes=N)[0]  # (N, N)

        x_source = x.unsqueeze(2)  # (B, T, 1, N, F)
        adj_mask = adj.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # (1, 1, N, N, 1)

        x_masked = torch.where(
            adj_mask.bool(),
            x_source,
            torch.tensor(float('-inf'), device=device),
        )  # (B, T, N, N, F)

        x_pooled, _ = torch.max(x_masked, dim=3)  # (B, T, N, F)
        return x_pooled

    def _avg_pool_neighborhoods(
        self,
        x: torch.Tensor,
        edge_index_multi: torch.Tensor,
        num_nodes: int,
        active_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply average-pooling over multi-scale neighborhoods for each node.

        Active-mask-aware: only active neighbors contribute to the average.
        Nodes with zero active neighbors produce output 0 (not NaN).

        Args:
            x: Node features (B, T, N, F)
            edge_index_multi: Multi-scale edge indices (2, E_multi)
            num_nodes: Number of nodes
            active_mask: (B, N) boolean mask of active nodes

        Returns:
            x_pooled: Avg-pooled features (B, T, N, F)
        """
        B, T, N, F = x.shape
        device = x.device

        from torch_geometric.utils import to_dense_adj

        adj = to_dense_adj(edge_index_multi, max_num_nodes=N)[0]  # (N, N)

        # Build combined mask: adjacency AND active source nodes
        adj_mask = adj.unsqueeze(0).unsqueeze(0).unsqueeze(-1).bool()  # (1, 1, N, N, 1)

        if active_mask is not None:
            # active_mask (B, N) -> (B, 1, 1, N, 1) masks source nodes
            active_src = active_mask.unsqueeze(1).unsqueeze(1).unsqueeze(-1)  # (B, 1, 1, N, 1)
            combined_mask = adj_mask & active_src  # (B, 1, N, N, 1)
        else:
            combined_mask = adj_mask  # (1, 1, N, N, 1)

        # Count active neighbors per target node, clamped to min=1 for safety
        neighbor_count = combined_mask.float().sum(dim=3)  # (B|1, 1|T, N, 1)
        neighbor_count = neighbor_count.clamp(min=1.0)

        x_source = x.unsqueeze(2)  # (B, T, 1, N, F)
        x_masked = torch.where(combined_mask, x_source, torch.zeros_like(x_source))  # (B, T, N, N, F)

        x_summed = x_masked.sum(dim=3)  # (B, T, N, F)
        x_pooled = x_summed / neighbor_count  # (B, T, N, F)

        return x_pooled

    # ------------------------------------------------------------------
    # Memory guard
    # ------------------------------------------------------------------
    def _check_dense_memory(self, B: int, T: int, N: int, F: int, element_size: int) -> None:
        """Raise RuntimeError if projected dense intermediate exceeds budget."""
        dense_bytes = B * T * N * N * F * element_size
        dense_gb = dense_bytes / (1024 ** 3)
        if dense_gb > self.max_dense_gb:
            raise RuntimeError(
                f"NeighborhoodFeaturePooling: dense (B,T,N,N,F) intermediate "
                f"would require {dense_gb:.1f} GB (limit {self.max_dense_gb} GB). "
                "Reduce batch size, timesteps, or num_nodes, or use pool_K=0."
            )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None,
        return_adj: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Aggregate features over multi-scale neighborhoods.

        Two-stage protocol for active_mask:
        1. Inactive nodes are set to -inf (max) or 0 (avg) before aggregation
        2. Remaining -inf values are zeroed after aggregation (max only)

        Args:
            x: Node features (B, T, N, F)
            edge_index: Edge indices (2, E) of the base graph
            active_mask: (B, N) boolean mask of active nodes. None = all active.
            return_adj: If True, also return the multi-scale edge_index.

        Returns:
            x_pooled: Aggregated features (B, T, N, F)
            edge_index_multi: Multi-scale adjacency (2, E') if return_adj else None
        """
        B, T, N, F = x.shape

        # K=0: identity pass-through (no aggregation)
        if self.K == 0:
            return x, (None if not return_adj else None)

        if self.aggregation == 'max':
            num_hops = self.K * self.stride

            # Apply -inf masking for inactive nodes before max-pooling
            if active_mask is not None:
                mask_4d = active_mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, N, 1)
                x = torch.where(mask_4d, x, torch.full_like(x, float('-inf')))

            x_pooled = self._max_pool_neighborhoods_sparse(
                x=x,
                edge_index=edge_index,
                num_nodes=N,
                num_hops=num_hops,
            )

            # Convert remaining -inf to zero (nodes whose entire neighborhood was inactive)
            x_pooled = torch.where(
                torch.isinf(x_pooled),
                torch.zeros_like(x_pooled),
                x_pooled,
            )
            edge_index_multi = (
                self._compute_within_hop_adjacency(edge_index, N, num_hops)
                if return_adj
                else None
            )
        elif self.aggregation == 'avg':
            # Avg path remains legacy dense implementation.
            self._check_dense_memory(B, T, N, F, x.element_size())
            edge_index_multi = self._get_cached_adjacency(edge_index, N)

            # For avg pooling, zero out inactive nodes (not -inf) and pass active_mask
            # so normalization only counts active neighbors
            if active_mask is not None:
                mask_4d = active_mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, N, 1)
                x = x * mask_4d.float()

            x_pooled = self._avg_pool_neighborhoods(x, edge_index_multi, N, active_mask)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        adj_out = edge_index_multi if return_adj else None
        return x_pooled, adj_out


class StridedGraphMaxPool(nn.Module):
    """
    Strided graph pooling with neighborhood aggregation + node selection.
    
    Stage 1 uses NeighborhoodFeaturePooling:
      - ``neighborhood_pooling='max'``: sparse iterative max over ``K * stride_input`` hops
      - ``neighborhood_pooling='avg'``: legacy exact strided-frontier avg
      - ``neighborhood_pooling=None`` / ``'null'``: disabled (valid when ``K=0``)

    **Operation:**
    1. Aggregate over graph neighborhoods (max or avg)
    2. Downsample by selecting nodes (every γ-th active node or learned)
    
    **Key Design:** 
    - Max-pooling window size is based on **input stride** (signal spacing)
    - Downsampling ratio is determined by **gamma** (stride_post/stride_pre)
    - These are decoupled for flexibility
    
    **Example:** stride_input=1, γ=2, K=2 on a 2D lattice
    - Each node's pooling window includes:
      * 0-hop: itself (1 node)
      * (max path) sparse propagation over up to K·stride_input hops
      * (avg path) exact strided-frontier adjacency
    - Then select every 2nd active node (downsample by γ=2)
    
    Args:
        gamma: Stride factor (downsampling ratio, stride_post/stride_pre)
        K: Number of neighborhood scales to consider (k=0,1,...,K)
        selection_method: How to select nodes after pooling
            - 'stride': Select every γ-th active node (deterministic, default)
            - 'learned': Use learnable selection with StridedPoolingGNN
            - 'random': Random sampling (for data augmentation)
        in_channels: Input feature dimension (required for learned selection)
        pooling_ratio: Override pooling ratio (default: 1/γ)
        selector_kwargs: Additional kwargs for LearnableGraphPool (only for learned selection)
            e.g., {'hidden_channels': 32, 'num_gnn_layers': 2, 'K': 2}
        stride_input: Spacing between active nodes in input signal (default: 1)
            Used for computing max-pooling neighborhoods AND by learned selector
            For level i with stride_pre, set stride_input=stride_pre
    
    Shape:
        - Input x: (B, T, N, F)
        - Edge index: (2, E)
        - Output x_pooled: (B, T, N, F) with ~N/γ active nodes
        - Output active_mask: (B, N) - which nodes are active
    
    Example:
        >>> # stride_input=1 (dense), γ=2 downsampling, consider 0,1,2-hop neighborhoods
        >>> pool = StridedGraphMaxPool(gamma=2, K=2, stride_input=1, selection_method='stride')
        >>> x = torch.randn(4, 10, 100, 64)
        >>> edge_index = torch.randint(0, 400, (2, 500))
        >>> x_pooled, mask, _ = pool(x, edge_index)
        >>> # x_pooled: (4, 10, 100, 64) - max-pooled with ~50 active nodes
    """
    
    def __init__(
        self,
        gamma: int = 2,
        K: int = 2,
        selection_method: Literal['stride', 'learned', 'random'] = 'stride',
        in_channels: Optional[int] = None,
        pooling_ratio: Optional[float] = None,
        selector_kwargs: Optional[dict] = None,
        stride_input: int = 1,
        skip_selector: bool = False,
        neighborhood_pooling: Optional[Literal['max', 'avg', 'null']] = 'max',
        neighborhood_include_self: bool = True,
        max_dense_gb: float = 8.0,
    ):
        super().__init__()
        
        self.gamma = gamma
        self.K = K
        self.selection_method = selection_method
        self.stride_input = stride_input
        self.selector_version = 'legacy'
        self.neighborhood_include_self = neighborhood_include_self
        self.max_dense_gb = max_dense_gb

        # Normalize explicit string 'null' to Python None for config compatibility.
        normalized_neighborhood_pooling: Optional[str] = (
            None if neighborhood_pooling == 'null' else neighborhood_pooling
        )
        self.neighborhood_pooling = normalized_neighborhood_pooling

        # Stage 1: Compose NeighborhoodFeaturePooling when K > 0
        if K > 0:
            if normalized_neighborhood_pooling not in {'max', 'avg'}:
                raise ValueError(
                    "neighborhood_pooling must be 'max' or 'avg' when K > 0; "
                    f"got {neighborhood_pooling!r}. "
                    "Use pool_K=0 with neighborhood_pooling=null to disable it."
                )
            self.neighborhood_agg = NeighborhoodFeaturePooling(
                K=K,
                stride=stride_input,
                aggregation=normalized_neighborhood_pooling,
                include_self=neighborhood_include_self,
                max_dense_gb=max_dense_gb,
            )
        else:
            self.neighborhood_agg = None

        # Validate skip_selector usage.
        # skip_selector is only meaningful at gamma=1 (where _select_nodes()
        # short-circuits before calling the selector). Using it with gamma>1
        # would produce self.selector=None while _select_nodes() unconditionally
        # calls self.selector(...) for the 'learned' path, causing a NoneType error.
        if skip_selector and selection_method == 'learned' and gamma > 1:
            raise ValueError(
                "skip_selector=True is invalid with selection_method='learned' and gamma>1. "
                "The selector would be skipped at construction but called at runtime. "
                "skip_selector is only safe when gamma==1 (identity selection, no-op)."
            )

        # If using learned selection, create a LearnableGraphPool.
        # skip_selector allows callers to suppress selector creation (e.g. for
        # gamma=1 levels where _select_nodes() short-circuits anyway).
        if selection_method == 'learned' and not skip_selector:
            assert in_channels is not None, \
                "in_channels required for learned selection"
            
            # Normalize optional override, but allow explicit zero-like values
            # to fail fast in downstream constructors.
            pooling_ratio = 1.0 / gamma if pooling_ratio is None else pooling_ratio
            selector_kwargs = selector_kwargs or {}
            selector_version = selector_kwargs.get('selector_version', 'legacy')
            if selector_version not in {'legacy', 'v3'}:
                raise ValueError(
                    f"Unknown selector_version='{selector_version}'. "
                    "Expected one of: 'legacy', 'v3'."
                )
            self.selector_version = selector_version

            # Prevent duplicate kwargs from nested configs (legacy configs still pass
            # pooling_kwargs to keep compatibility with older paths).
            selector_kwargs = dict(selector_kwargs)
            explicit_pooling_ratio = selector_kwargs.pop('pooling_ratio', None)
            if explicit_pooling_ratio is not None:
                pooling_ratio = explicit_pooling_ratio
            
            # Legacy mode preserves historical behavior for compatibility.
            if selector_version == 'legacy':
                selector_kwargs.pop('selector_version', None)
                selector_kwargs.pop('min_retained_nodes', None)
                selector_kwargs.pop('temporal_reduce', None)
                selector_kwargs.pop('cond_fusion', None)
                selector_kwargs.pop('cond_fusion_mode', None)
                selector_kwargs.pop('cond_fusion_heads', None)
                selector_kwargs.pop('cond_fusion_dropout', None)
                selector_kwargs.pop('exploration_noise', None)
                selector_kwargs.pop('exploration_noise_min', None)
                selector_kwargs.pop('exploration_noise_schedule', None)
                selector_kwargs.pop('exploration_noise_anneal_steps', None)
                selector_kwargs.pop('exploration_noise_warmup_steps', None)
                selector_kwargs.pop('collect_heavy_diagnostics', None)
                selector_kwargs.pop('collect_sampling_probe', None)
                selector_kwargs.pop('collect_sampling_probe', None)
                selector_kwargs.pop('time_emb_dim', None)
                selector_kwargs.pop('packed_score_mode', None)
                selector_kwargs.pop('packed_score_threshold', None)
                # Pass stride_input to selector so internal GNN respects signal sparsity
                if 'gamma' not in selector_kwargs:
                    selector_kwargs['gamma'] = stride_input
                self.selector = LearnableGraphPool(
                    in_channels=in_channels,
                    pooling_ratio=pooling_ratio,
                    **selector_kwargs
                )
            else:
                # New v3-aware path uses a dedicated node selector implementation.
                # Ignore legacy-only knobs when using the new selector.
                #
                # v3 selector can receive explicit cond_dim to support call sites
                # where cond width differs from selector in_channels (for example,
                # encoder cond projected at block in_channels while selector runs
                # on out_channels).
                selector_kwargs.pop('selector_version', None)
                selector_kwargs.pop('use_pooling_gnn', None)
                selector_kwargs.pop('use_strided_pooling_gnn', None)
                selector_kwargs.pop('hidden_channels', None)
                selector_kwargs.pop('num_gnn_layers', None)
                selector_kwargs.pop('temporal_score_agg', None)
                selector_kwargs.pop('ema_alpha_init', None)
                selector_kwargs.pop('hidden_norm_type', None)
                selector_kwargs.pop('score_gain_init', None)
                selector_kwargs.pop('learnable_score_gain', None)
                selector_kwargs.pop('selector_kwargs', None)
                self.selector = NodeSelector(
                    in_channels=in_channels,
                    pooling_ratio=pooling_ratio,
                    **selector_kwargs
                )
        else:
            self.selector = None

    # ------------------------------------------------------------------
    # Compatibility shims — delegate adjacency / pooling methods to
    # NeighborhoodFeaturePooling so that legacy callers (tests,
    # monkeypatches, visualization benchmarks) continue to work.
    # ------------------------------------------------------------------
    def _get_neighborhood_helper(
        self,
        K_override: Optional[int] = None,
        stride_override: Optional[int] = None,
    ) -> 'NeighborhoodFeaturePooling':
        """Return self.neighborhood_agg if params match, else create a fresh helper."""
        if (
            self.neighborhood_agg is not None
            and K_override in (None, self.K)
            and stride_override in (None, self.stride_input)
        ):
            return self.neighborhood_agg
        return NeighborhoodFeaturePooling(
            K=self.K if K_override is None else K_override,
            stride=self.stride_input if stride_override is None else stride_override,
            aggregation='max',
            include_self=self.neighborhood_include_self,
            max_dense_gb=self.max_dense_gb,
        )

    @staticmethod
    def _extract_stride_k_from_call(args, kwargs):
        """Extract stride and K from positional / keyword arguments."""
        stride = kwargs.get('stride')
        K = kwargs.get('K')
        if stride is None and len(args) >= 3:
            stride = args[2]
        if K is None and len(args) >= 4:
            K = args[3]
        return stride, K

    def _compute_strided_hop_adjacency_boolean(self, *args, **kwargs):
        """Compatibility shim — delegates to NeighborhoodFeaturePooling."""
        stride, K = self._extract_stride_k_from_call(args, kwargs)
        helper = self._get_neighborhood_helper(K_override=K, stride_override=stride)
        return helper._compute_strided_hop_adjacency_boolean(*args, **kwargs)

    def _compute_strided_hop_adjacency(self, *args, **kwargs):
        """Compatibility shim — delegates to NeighborhoodFeaturePooling."""
        stride, K = self._extract_stride_k_from_call(args, kwargs)
        helper = self._get_neighborhood_helper(K_override=K, stride_override=stride)
        return helper._compute_strided_hop_adjacency(*args, **kwargs)

    def _max_pool_neighborhoods(self, *args, **kwargs):
        """Compatibility shim — delegates to NeighborhoodFeaturePooling."""
        helper = self._get_neighborhood_helper()
        return helper._max_pool_neighborhoods(*args, **kwargs)

    # ------------------------------------------------------------------
    # Node selection (unified dispatch for K=0 and K>0)
    # ------------------------------------------------------------------
    def _stride_selection(
        self,
        x: torch.Tensor,
        gamma: int,
        current_active_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select every γ-th node from currently active nodes (stride-based downsampling).
        
        Args:
            x: Max-pooled features (B, T, N, F)
            gamma: Stride factor
            current_active_mask: Current active nodes (B, N) to select from
        
        Returns:
            x_selected: Selected features (B, T, N, F)
            active_mask: Binary mask (B, N) of selected nodes
        """
        B, T, N, F = x.shape
        
        if current_active_mask is None:
            # No active mask provided, select every γ-th node from all nodes
            selected_indices = torch.arange(0, N, gamma, device=x.device)
            active_mask = torch.zeros(B, N, device=x.device, dtype=torch.bool)
            active_mask[:, selected_indices] = True
        else:
            # Select every γ-th node from the currently active nodes
            active_mask = torch.zeros(B, N, device=x.device, dtype=torch.bool)
            
            for b in range(B):
                # Get indices of currently active nodes
                active_indices = torch.where(current_active_mask[b])[0]
                
                # Select every γ-th node from active indices
                selected_from_active = active_indices[::gamma]
                
                # Mark selected nodes as active
                active_mask[b, selected_from_active] = True
        
        # Zero out non-selected nodes
        mask_4d = active_mask.unsqueeze(1).unsqueeze(-1).float()
        x_selected = x * mask_4d
        
        return x_selected, active_mask
    
    def _random_selection(
        self,
        x: torch.Tensor,
        gamma: int,
        active_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly select N/γ nodes.
        
        Args:
            x: Max-pooled features (B, T, N, F)
            gamma: Downsampling factor
            active_mask: Current active nodes (B, N)
        
        Returns:
            x_selected: Selected features (B, T, N, F)
            new_active_mask: Binary mask (B, N) of selected nodes
        """
        B, T, N, F = x.shape
        
        if active_mask is None:
            active_mask = torch.ones(B, N, device=x.device, dtype=torch.bool)
        
        # Number of nodes to keep
        num_active = active_mask.sum(dim=1)
        k = torch.clamp((num_active.float() / gamma).long(), min=1)
        
        max_k = k.max().item()
        
        # Random selection from active nodes
        new_active_mask = torch.zeros(B, N, device=x.device, dtype=torch.bool)
        
        for b in range(B):
            active_indices = torch.where(active_mask[b])[0]
            if len(active_indices) > 0:
                # Randomly select k[b] nodes
                perm = torch.randperm(len(active_indices), device=x.device)[:k[b]]
                selected = active_indices[perm]
                new_active_mask[b, selected] = True
        
        # Zero out non-selected nodes
        mask_4d = new_active_mask.unsqueeze(1).unsqueeze(-1).float()
        x_selected = x * mask_4d
        
        return x_selected, new_active_mask

    @staticmethod
    def _selected_indices_from_mask(active_mask: torch.Tensor) -> torch.Tensor:
        B, _ = active_mask.shape
        max_selected = int(active_mask.sum(dim=1).max().item())
        if max_selected <= 0:
            return torch.empty((B, 0), dtype=torch.long, device=active_mask.device)

        selected_indices = torch.full(
            (B, max_selected),
            -1,
            dtype=torch.long,
            device=active_mask.device,
        )
        for b in range(B):
            idx = torch.where(active_mask[b])[0]
            selected_indices[b, :idx.numel()] = idx
        return selected_indices

    def _select_nodes(
        self,
        x: torch.Tensor,
        active_mask: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        _selector_extra: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Unified node selection dispatch for all selection methods.

        Args:
            x: Features to select from (B, T, N, F)
            active_mask: Current active mask (B, N)
            edge_index: Edge indices (needed for learned selection)
            edge_weight: Edge weights (needed for learned selection)
            cond: Conditioning tensor (needed for learned selection)
            _selector_extra: Extra kwargs for v3 selector (e.g. time_emb)

        Returns:
            x_pooled: Selected features (B, T, N, F)
            new_active_mask: Updated mask (B, N)
            selected_indices: (B, k) or None
            scores: Selection scores for learned, None otherwise
        """
        scores: Optional[torch.Tensor] = None
        # Local gamma=1 means no downsampling at this level. Keep stage-1
        # neighborhood aggregation output, but skip selector computation.
        if self.gamma == 1:
            new_active_mask = active_mask
            mask_4d = new_active_mask.unsqueeze(1).unsqueeze(-1).to(dtype=x.dtype)
            x_pooled = x * mask_4d
            selected_indices = self._selected_indices_from_mask(new_active_mask)
            return x_pooled, new_active_mask, selected_indices, scores

        if self.selection_method == 'learned':
            x_pooled, new_active_mask, selected_indices, scores = self.selector(
                x,
                edge_index,
                edge_weight=edge_weight,
                active_mask=active_mask,
                cond=cond,
                **((_selector_extra or {})),
            )
        elif self.selection_method == 'stride':
            x_pooled, new_active_mask = self._stride_selection(
                x, self.gamma, current_active_mask=active_mask
            )
            selected_indices = self._selected_indices_from_mask(new_active_mask)
        elif self.selection_method == 'random':
            x_pooled, new_active_mask = self._random_selection(
                x, self.gamma, active_mask
            )
            selected_indices = self._selected_indices_from_mask(new_active_mask)
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")

        return x_pooled, new_active_mask, selected_indices, scores

    # ------------------------------------------------------------------
    # Forward — two-stage pipeline: neighborhood aggregation + node selection
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        active_mask: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        time_emb: Optional[torch.Tensor] = None,
        return_neighborhoods: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Apply neighborhood aggregation then node selection (downsampling).

        Two-stage pipeline:
            Stage 1 — Neighborhood aggregation (K > 0 only):
                Delegates to composed NeighborhoodFeaturePooling which handles
                -inf masking, adjacency computation, and max/avg pooling.
            Stage 2 — Node selection:
                Unified dispatch to learned / stride / random.

        Args:
            x: Node features (B, T, N, F)
            edge_index: Edge indices (2, E)
            edge_weight: Optional edge weights (not used for max-pooling)
            active_mask: Binary mask (B, N) of currently active nodes
            cond: Optional conditioning for learned selection
            time_emb: Optional raw time embedding (B, D) for learned selector
            return_neighborhoods: If True, return neighborhood adjacency matrix

        Returns:
            x_pooled: Pooled and downsampled features (B, T, N, F)
            new_active_mask: Binary mask (B, N) of selected nodes
            selected_indices: Indices of selected nodes (B, k) or None
            fourth: (B, N) scores for learned selection, or neighborhood adjacency
                (2, E') if return_neighborhoods=True and selection_method='stride'/'random',
                else None.
        """
        B, T, N, F = x.shape

        if active_mask is None:
            active_mask = torch.ones(B, N, device=x.device, dtype=torch.bool)

        # Prepare selector kwargs shared between K==0 and K>0 paths.
        _selector_extra = {}
        if self.selector_version == 'v3' and time_emb is not None:
            _selector_extra['time_emb'] = time_emb

        # Stage 1: Neighborhood aggregation (skip when K=0)
        edge_index_multi: Optional[torch.Tensor] = None
        if self.neighborhood_agg is not None:
            x, edge_index_multi = self.neighborhood_agg(
                x, edge_index, active_mask=active_mask,
                return_adj=return_neighborhoods,
            )

        # Stage 2: Node selection (unified dispatch)
        x_pooled, new_active_mask, selected_indices, scores = self._select_nodes(
            x, active_mask,
            edge_index=edge_index,
            edge_weight=edge_weight,
            cond=cond,
            _selector_extra=_selector_extra,
        )

        # 4th return element: scores for learned, neighborhoods otherwise
        fourth = scores if self.selection_method == 'learned' else edge_index_multi

        return x_pooled, new_active_mask, selected_indices, fourth


class GraphMaxUnpool(nn.Module):
    """
    Unpooling for GraphMaxPool - upsamples with nearest-neighbor interpolation.
    
    Since max-pooling is not invertible, we use nearest-neighbor interpolation
    to upsample: each node takes the value of the nearest selected node in
    the graph (via shortest path distance or γ-hop neighborhood).
    
    Args:
        gamma: Hop count (should match the pooling gamma)
        interpolation: Interpolation method
            - 'nearest': Use nearest selected neighbor (default)
            - 'zero': Zero-padding (like standard GraphUnpool)
    
    Shape:
        - Input x_pooled: (B, T, N, F) - pooled features (sparse)
        - Input x_skip: (B, T, N, F_skip) - skip connection
        - Selection mask: (B, N) - which nodes were selected
        - Output: (B, T, N, F + F_skip)
    
    Example:
        >>> unpool = GraphMaxUnpool(gamma=2)
        >>> x_pooled = torch.randn(4, 10, 100, 64)  # Sparse (some nodes zero)
        >>> x_skip = torch.randn(4, 10, 100, 64)
        >>> selection_mask = torch.rand(4, 100) > 0.5  # Which nodes selected
        >>> x = unpool(x_pooled, x_skip, selection_mask, edge_index)
    """
    
    def __init__(
        self,
        gamma: int = 2,
        interpolation: Literal['nearest', 'zero'] = 'nearest',
    ):
        super().__init__()
        
        self.gamma = gamma
        self.interpolation = interpolation
    
    def _nearest_neighbor_interpolation(
        self,
        x_pooled: torch.Tensor,
        selection_mask: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """
        Upsample using nearest-neighbor interpolation on the graph.
        
        Each inactive node takes the value of its nearest active neighbor.
        
        Args:
            x_pooled: Pooled features (B, T, N, F) - sparse
            selection_mask: Binary mask (B, N) of selected nodes
            edge_index: Edge indices (2, E)
            num_nodes: Number of nodes
        
        Returns:
            x_upsampled: Upsampled features (B, T, N, F)
        """
        B, T, N, F = x_pooled.shape
        
        # For each inactive node, find nearest active neighbor and copy value
        # Use BFS-style propagation from active nodes
        
        x_upsampled = x_pooled.clone()
        
        for b in range(B):
            # Get selected node indices
            selected_nodes = torch.where(selection_mask[b])[0]
            
            if len(selected_nodes) == 0:
                continue
            
            # Simple approach: Propagate from selected nodes via edges
            # Inactive nodes take value from nearest selected neighbor
            
            # Use iterative propagation (like label propagation)
            active = selection_mask[b].clone()
            
            for hop in range(self.gamma):
                # Get edges where source is active
                src, tgt = edge_index
                
                # Find edges where source is active and target is not
                mask = active[src] & ~active[tgt]
                
                if not mask.any():
                    break
                
                # For each such edge, propagate value from source to target
                for s, t in zip(src[mask], tgt[mask]):
                    if not active[t]:
                        x_upsampled[b, :, t, :] = x_upsampled[b, :, s, :]
                        active[t] = True
        
        return x_upsampled
    
    def forward(
        self,
        x_pooled: torch.Tensor,
        x_skip: torch.Tensor,
        selection_mask: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Unpool and combine with skip connection.
        
        Args:
            x_pooled: Pooled features (B, T, N, F_pooled)
            x_skip: Skip connection (B, T, N, F_skip)
            selection_mask: Which nodes were selected (B, N)
            edge_index: Edge indices (2, E) - required for nearest interpolation
        
        Returns:
            x: Upsampled features (B, T, N, F_pooled + F_skip)
        """
        B, T, N, F = x_pooled.shape
        
        if self.interpolation == 'nearest':
            assert edge_index is not None, \
                "edge_index required for nearest-neighbor interpolation"
            
            x_upsampled = self._nearest_neighbor_interpolation(
                x_pooled, selection_mask, edge_index, N
            )
        else:  # 'zero'
            x_upsampled = x_pooled  # Already has zeros for inactive nodes
        
        # Concatenate with skip connection
        x = torch.cat([x_upsampled, x_skip], dim=-1)
        
        return x
