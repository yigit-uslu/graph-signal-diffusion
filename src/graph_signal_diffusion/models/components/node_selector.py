"""Selection modules used by v3-aware learned pooling paths."""

import math
from typing import Optional, Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class NodeSelector(nn.Module):
    """Lightweight learned node selector for v3-style selector path.

    The selector consumes processed encoder node embeddings and produces
    per-node logits that are routed into top-k downsampling.

    Conditioning-channel assumption in current UGNN wiring:
    - UGNN encoder blocks project conditioning features with a per-block
      projection (`cond_proj`) before passing `cond` to pooling/selection.
    - As a result, selector conditioning features already match each block's
      selector width (`in_channels`).
    - Therefore, the default `cond_dim=None` path (identity cond projection)
      is the expected path for current UGNN runs.
    - If `cond_dim` were set to a different width, NodeSelector would create
      an internal projection `Linear(cond_dim -> in_channels)`. However, the
      current StridedGraphMaxPool v3 wiring intentionally drops `cond_dim`
      from `selector_kwargs` and relies on the matched-width assumption above.
    """

    def __init__(
        self,
        in_channels: int,
        pooling_ratio: float = 0.5,
        selection_mode: Literal['soft', 'hard', 'ste'] = 'ste',
        temperature: float = 1.0,
        temperature_schedule: Literal['constant', 'linear', 'cosine'] = 'constant',
        temperature_min: float = 0.1,
        temperature_anneal_steps: int = 0,
        temperature_warmup_steps: int = 0,
        temperature_anneal_epochs: Optional[int] = None,
        temperature_warmup_epochs: Optional[int] = None,
        entropy_reg_weight: float = 0.0,
        min_retained_nodes: int = 1,
        temporal_reduce: Literal['mean', 'last', 'attn'] = 'mean',
        # Conditioning fusion ---
        # ``cond_fusion_mode`` supersedes the legacy boolean ``cond_fusion``.
        # Accepted values: 'cross_attention', 'film', 'add', 'none'.
        # For backward-compat the old ``cond_fusion: bool`` is still accepted:
        #   True  -> 'cross_attention'  (default)
        #   False -> 'none'
        cond_fusion: bool = True,
        cond_fusion_mode: Optional[Literal['cross_attention', 'film', 'add', 'none']] = None,
        cond_fusion_heads: int = 4,
        cond_fusion_dropout: float = 0.0,
        # Optional explicit conditioning width. When None, selector assumes
        # cond features already have width == in_channels and uses identity.
        #
        # NOTE: In current UGNN v3 wiring this arg is intentionally left unset:
        # cond is projected per encoder block before selector invocation, so
        # the widths already match by construction.
        cond_dim: Optional[int] = None,
        # Time-embedding fusion ---
        time_emb_dim: Optional[int] = None,
        exploration_noise: float = 0.0,
        exploration_noise_min: float = 0.0,
        exploration_noise_schedule: Literal['constant', 'linear', 'cosine'] = 'constant',
        exploration_noise_anneal_steps: int = 0,
        exploration_noise_warmup_steps: int = 0,
        collect_heavy_diagnostics: bool = False,
        collect_sampling_probe: bool = False,
        # Packed score computation ---
        packed_score_mode: Literal['off', 'auto'] = 'off',
        packed_score_threshold: float = 0.5,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.pooling_ratio = float(pooling_ratio)
        self.selection_mode = selection_mode
        self.temperature = float(temperature)
        self.temperature_schedule = temperature_schedule
        self.temperature_min = float(temperature_min)
        self.temperature_anneal_steps = int(temperature_anneal_steps)
        self.temperature_warmup_steps = int(temperature_warmup_steps)
        self.entropy_reg_weight = float(entropy_reg_weight)
        self.min_retained_nodes = int(min_retained_nodes)
        self.temporal_reduce = temporal_reduce
        self.exploration_noise = float(exploration_noise)
        self.exploration_noise_min = float(exploration_noise_min)
        self.exploration_noise_schedule = exploration_noise_schedule
        self.exploration_noise_anneal_steps = int(exploration_noise_anneal_steps)
        self.exploration_noise_warmup_steps = int(exploration_noise_warmup_steps)
        self.current_step = 0
        self.collect_heavy_diagnostics = bool(collect_heavy_diagnostics)
        self.collect_sampling_probe = bool(collect_sampling_probe)
        self.packed_score_mode = packed_score_mode
        self.packed_score_threshold = float(packed_score_threshold)

        # --- Resolve cond_fusion_mode from (maybe legacy) arguments --------
        if cond_fusion_mode is not None:
            if cond_fusion_mode not in {'cross_attention', 'film', 'add', 'none'}:
                raise ValueError(
                    f"Unknown cond_fusion_mode='{cond_fusion_mode}'. "
                    "Expected one of: 'cross_attention', 'film', 'add', 'none'."
                )
            self.cond_fusion_mode = cond_fusion_mode
        else:
            # Legacy: bool cond_fusion → mode string
            self.cond_fusion_mode = 'cross_attention' if cond_fusion else 'none'
        # Keep a convenience flag for quick "any conditioning?" checks
        self.cond_fusion = self.cond_fusion_mode != 'none'
        self.cond_fusion_heads = int(cond_fusion_heads)
        self.cond_fusion_dropout = float(cond_fusion_dropout)

        # --- Time-embedding FiLM -------------------------------------------
        self.time_emb_dim = int(time_emb_dim) if time_emb_dim else 0
        if self.time_emb_dim > 0:
            self.time_film = nn.Sequential(
                nn.SiLU(),
                nn.Linear(self.time_emb_dim, in_channels * 2),
            )
            # Zero-init → identity at start
            nn.init.zeros_(self.time_film[-1].weight)
            nn.init.zeros_(self.time_film[-1].bias)
        else:
            self.time_film = None

        if temperature_anneal_epochs is not None:
            if self.temperature_anneal_steps not in {0, int(temperature_anneal_epochs)}:
                raise ValueError(
                    "Specify only one of temperature_anneal_steps or "
                    "temperature_anneal_epochs (deprecated alias)."
                )
            self.temperature_anneal_steps = int(temperature_anneal_epochs)
        if temperature_warmup_epochs is not None:
            if self.temperature_warmup_steps not in {0, int(temperature_warmup_epochs)}:
                raise ValueError(
                    "Specify only one of temperature_warmup_steps or "
                    "temperature_warmup_epochs (deprecated alias)."
                )
            self.temperature_warmup_steps = int(temperature_warmup_epochs)

        self.temperature_anneal_epochs = self.temperature_anneal_steps
        self.temperature_warmup_epochs = self.temperature_warmup_steps

        if self.pooling_ratio <= 0.0 or self.pooling_ratio > 1.0:
            raise ValueError(
                f"pooling_ratio must be in (0, 1], got {self.pooling_ratio}"
            )
        if self.selection_mode not in {'soft', 'hard', 'ste'}:
            raise ValueError(
                f"Unknown selection_mode='{self.selection_mode}'. "
                "Expected one of: 'soft', 'hard', 'ste'."
            )
        if self.min_retained_nodes < 1:
            raise ValueError(
                f"min_retained_nodes must be >= 1, got {self.min_retained_nodes}"
            )
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
        if self.temperature_min <= 0:
            raise ValueError(f"temperature_min must be > 0, got {self.temperature_min}")
        if self.temperature_schedule not in {'constant', 'linear', 'cosine'}:
            raise ValueError(
                f"Unknown temperature_schedule='{self.temperature_schedule}'. "
                "Expected one of: 'constant', 'linear', 'cosine'."
            )
        if self.temperature_anneal_steps < 0:
            raise ValueError(
                f"temperature_anneal_steps must be >= 0, got {self.temperature_anneal_steps}"
            )
        if self.temperature_warmup_steps < 0:
            raise ValueError(
                f"temperature_warmup_steps must be >= 0, got {self.temperature_warmup_steps}"
            )
        if self.exploration_noise < 0:
            raise ValueError(
                f"exploration_noise must be >= 0, got {self.exploration_noise}"
            )
        if self.exploration_noise_min < 0:
            raise ValueError(
                f"exploration_noise_min must be >= 0, got {self.exploration_noise_min}"
            )
        if self.exploration_noise_schedule not in {'constant', 'linear', 'cosine'}:
            raise ValueError(
                f"Unknown exploration_noise_schedule='{self.exploration_noise_schedule}'. "
                "Expected one of: 'constant', 'linear', 'cosine'."
            )
        if self.exploration_noise_anneal_steps < 0:
            raise ValueError(
                f"exploration_noise_anneal_steps must be >= 0, got {self.exploration_noise_anneal_steps}"
            )
        if self.exploration_noise_warmup_steps < 0:
            raise ValueError(
                f"exploration_noise_warmup_steps must be >= 0, got {self.exploration_noise_warmup_steps}"
            )
        if self.entropy_reg_weight < 0:
            raise ValueError(
                f"entropy_reg_weight must be >= 0, got {self.entropy_reg_weight}"
            )
        if self.temporal_reduce not in {'mean', 'last', 'attn'}:
            raise ValueError(
                f"Unknown temporal_reduce='{self.temporal_reduce}'. "
                "Expected one of: 'mean', 'last', 'attn'."
            )
        if self.packed_score_mode not in {'off', 'auto'}:
            raise ValueError(
                f"Unknown packed_score_mode='{self.packed_score_mode}'. "
                "Expected one of: 'off', 'auto'."
            )
        if self.packed_score_threshold <= 0.0 or self.packed_score_threshold > 1.0:
            raise ValueError(
                f"packed_score_threshold must be in (0, 1], got {self.packed_score_threshold}"
            )

        # --- Build conditioning sub-modules --------------------------------
        if self.cond_fusion_mode == 'cross_attention':
            if self.cond_fusion_heads < 1:
                raise ValueError(
                    f"cond_fusion_heads must be >= 1, got {self.cond_fusion_heads}"
                )
            if in_channels % self.cond_fusion_heads != 0:
                raise ValueError(
                    "cond_fusion_heads must divide in_channels for MultiheadAttention."
                )
            self.cond_projection = (
                nn.Identity()
                if cond_dim is None or cond_dim == in_channels
                else nn.Linear(cond_dim, in_channels)
            )
            self.cond_attention = nn.MultiheadAttention(
                embed_dim=in_channels,
                num_heads=self.cond_fusion_heads,
                dropout=self.cond_fusion_dropout,
                batch_first=True,
            )
            self.cond_film = None
            self.cond_add_bias = None
        elif self.cond_fusion_mode == 'film':
            cond_c = cond_dim if cond_dim is not None else in_channels
            self.cond_projection = (
                nn.Identity()
                if cond_c == in_channels
                else nn.Linear(cond_c, in_channels)
            )
            self.cond_film = nn.Sequential(
                nn.SiLU(),
                nn.Linear(in_channels, in_channels),
                nn.SiLU(),
                nn.Linear(in_channels, in_channels * 2),
            )
            # Zero-init → identity at start
            nn.init.zeros_(self.cond_film[-1].weight)
            nn.init.zeros_(self.cond_film[-1].bias)
            self.cond_attention = None
            self.cond_add_bias = None
        elif self.cond_fusion_mode == 'add':
            cond_c = cond_dim if cond_dim is not None else in_channels
            self.cond_projection = (
                nn.Identity()
                if cond_c == in_channels
                else nn.Linear(cond_c, in_channels)
            )
            # Scalar bias per node: Linear(C → 1), zero-init for identity at start.
            self.cond_add_bias = nn.Linear(in_channels, 1)
            nn.init.zeros_(self.cond_add_bias.weight)
            nn.init.zeros_(self.cond_add_bias.bias)
            self.cond_attention = None
            self.cond_film = None
        else:
            # 'none'
            self.cond_projection = None
            self.cond_attention = None
            self.cond_film = None
            self.cond_add_bias = None

        if self.temporal_reduce == 'attn':
            self.temporal_attn_proj = nn.Linear(in_channels, 1)
        else:
            self.temporal_attn_proj = None

        self.score_proj = nn.Linear(in_channels, 1)

        self.reset_selector_state()

    def reset_selector_state(self) -> None:
        """Reset cached auxiliary losses and diagnostics for the next forward pass."""
        self._selector_called_in_last_forward = False
        self._last_aux_loss: Optional[torch.Tensor] = None
        self._last_diagnostics: dict = {}

    def set_training_step(self, step: int) -> None:
        self.current_step = max(0, int(step))

    def set_training_epoch(self, epoch: int) -> None:
        self.set_training_step(epoch)

    def _current_temperature(self) -> float:
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

        return float(max(temp, self.temperature_min))

    def get_current_temperature(self) -> float:
        return self._current_temperature()

    def _current_exploration_noise(self) -> float:
        if (
            self.exploration_noise_schedule == 'constant'
            or self.exploration_noise_anneal_steps <= 0
        ):
            return float(self.exploration_noise)

        if self.current_step < self.exploration_noise_warmup_steps:
            return float(self.exploration_noise)

        progress = (self.current_step - self.exploration_noise_warmup_steps) / max(
            self.exploration_noise_anneal_steps, 1
        )
        progress = min(max(progress, 0.0), 1.0)

        if self.exploration_noise_schedule == 'linear':
            noise = self.exploration_noise + progress * (
                self.exploration_noise_min - self.exploration_noise
            )
        elif self.exploration_noise_schedule == 'cosine':
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            noise = self.exploration_noise_min + (
                self.exploration_noise - self.exploration_noise_min
            ) * cosine
        else:
            noise = self.exploration_noise

        return float(max(noise, self.exploration_noise_min))

    def get_current_exploration_noise(self) -> float:
        return self._current_exploration_noise()

    def set_collect_heavy_diagnostics(self, enabled: bool) -> None:
        self.collect_heavy_diagnostics = bool(enabled)

    def set_collect_sampling_probe(self, enabled: bool) -> None:
        self.collect_sampling_probe = bool(enabled)

    def get_selector_aux_loss(self) -> Optional[torch.Tensor]:
        if not self._selector_called_in_last_forward:
            return None
        return self._last_aux_loss

    def get_selector_diagnostics(self) -> dict:
        if not self._selector_called_in_last_forward:
            return {}
        return dict(self._last_diagnostics)

    def _aggregate_temporal(self, x: torch.Tensor) -> torch.Tensor:
        """Aggregate (B, T, N, C) node embeddings to (B, N, C)."""
        if self.temporal_reduce == 'mean':
            return x.mean(dim=1)
        if self.temporal_reduce == 'last':
            return x[:, -1]

        # Temporal attention weighted pooling.
        attn_scores = self.temporal_attn_proj(x).squeeze(-1)
        attn_w = F.softmax(attn_scores, dim=1).unsqueeze(-1)
        return (x * attn_w).sum(dim=1)

    @staticmethod
    def _entropy_regularizer(
        soft_mask: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        eps = 1e-6  # Must exceed float32 ULP near 1.0 (~1.19e-7)
        active_probs = soft_mask[active_mask.bool()]
        if active_probs.numel() == 0:
            zero = soft_mask.new_zeros(())
            return zero, zero

        p = active_probs.clamp(min=eps, max=1.0 - eps)
        entropy = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
        max_entropy = math.log(2.0)
        entropy_mean_norm = (entropy / max_entropy).mean()
        return (1.0 - entropy_mean_norm), entropy_mean_norm

    def _prepare_cond_tokens(
        self,
        cond: torch.Tensor,
        num_nodes: int,
        num_timesteps: int,
    ) -> Optional[torch.Tensor]:
        """Return cond tokens shaped (B, T, N, C) or None."""
        B = cond.size(0)
        if cond.dim() == 4:
            if cond.size(2) == num_nodes:
                if cond.size(1) != num_timesteps:
                    # Node-specific cond with different timestep count; allow broadcast at batch level.
                    if cond.size(1) == 1:
                        cond = cond.expand(B, num_timesteps, num_nodes, cond.size(-1))
                    else:
                        raise ValueError(
                            "cond temporal dimension must match x temporal dimension "
                            "or be 1 when cond is node-specific."
                        )
                return cond

            if cond.size(2) == 1:
                return cond.expand(B, cond.size(1), num_nodes, cond.size(3))

            raise ValueError(
                "Expected cond to be (B, T, N, C), (B, T, 1, C), "
                "or a global-temporal cond that can be broadcast."
            )

        if cond.dim() == 3:
            if cond.size(1) == num_nodes:
                return cond.unsqueeze(1).expand(B, num_timesteps, num_nodes, cond.size(-1))

            if cond.size(1) == num_timesteps:
                return cond.unsqueeze(2).expand(B, num_timesteps, num_nodes, cond.size(-1))

            raise ValueError(
                "Unexpected cond shape (B, ?, C). "
                "For per-node condition, pass (B, N, C). "
                "For global-temporal, pass (B, T, C)."
            )

        if cond.dim() == 2:
            return cond.unsqueeze(1).unsqueeze(2).expand(B, num_timesteps, num_nodes, cond.size(-1))

        return None

    def _fuse_time(
        self,
        node_repr: torch.Tensor,
        time_emb: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply global FiLM modulation from diffusion time embedding.

        Args:
            node_repr: (B, N, C) node representations.
            time_emb:  (B, D) raw time embedding, or None.

        Returns:
            Modulated (B, N, C) node representations.
        """
        if self.time_film is None or time_emb is None:
            return node_repr
        # time_film: D → 2C, zero-init
        film_params = self.time_film(time_emb)  # (B, 2C)
        gamma, beta = film_params.chunk(2, dim=-1)  # each (B, C)
        # Broadcast over nodes: gamma/beta are global per sample
        return node_repr * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

    def _fuse_cond(
        self,
        node_repr: torch.Tensor,
        cond: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Fuse conditional tokens with node features.

        Routes to cross-attention, FiLM, or additive-bias fusion depending on
        ``self.cond_fusion_mode``.
        """
        if not self.cond_fusion or cond is None:
            return node_repr

        B, N, C = node_repr.shape
        num_timesteps = self._last_timesteps

        # --- Cross-attention path (original) ---------------------------------
        if self.cond_fusion_mode == 'cross_attention':
            if self.cond_attention is None or self.cond_projection is None:
                return node_repr

            cond_tokens = self._prepare_cond_tokens(cond, num_nodes=N, num_timesteps=num_timesteps)
            if cond_tokens is None:
                return node_repr

            cond_tokens = self.cond_projection(cond_tokens)
            q = node_repr.view(B * N, 1, C)
            k = cond_tokens.permute(0, 2, 1, 3).reshape(B * N, -1, C)
            attn_out, _ = self.cond_attention(q, k, k, need_weights=False)
            fused = node_repr + attn_out.view(B, N, C)
            return fused

        # --- FiLM path -------------------------------------------------------
        if self.cond_fusion_mode == 'film':
            if self.cond_film is None or self.cond_projection is None:
                return node_repr

            cond_tokens = self._prepare_cond_tokens(cond, num_nodes=N, num_timesteps=num_timesteps)
            if cond_tokens is None:
                return node_repr

            cond_tokens = self.cond_projection(cond_tokens)  # (B, T, N, C)
            # Collapse temporal dim to get per-node conditioning
            cond_repr = cond_tokens.mean(dim=1)  # (B, N, C)
            film_params = self.cond_film(cond_repr)  # (B, N, 2C)
            gamma, beta = film_params.chunk(2, dim=-1)  # each (B, N, C)
            return node_repr * (1.0 + gamma) + beta

        # --- Additive bias path (lightweight) --------------------------------
        if self.cond_fusion_mode == 'add':
            if self.cond_add_bias is None or self.cond_projection is None:
                return node_repr

            cond_tokens = self._prepare_cond_tokens(cond, num_nodes=N, num_timesteps=num_timesteps)
            if cond_tokens is None:
                return node_repr

            cond_tokens = self.cond_projection(cond_tokens)  # (B, T, N, C)
            cond_repr = cond_tokens.mean(dim=1)  # (B, N, C)
            bias = self.cond_add_bias(cond_repr)  # (B, N, 1)
            return node_repr + bias

        return node_repr

    def _fuse_cond_packed(
        self,
        node_repr: torch.Tensor,
        cond: torch.Tensor,
        active_indices: torch.Tensor,
        B: int,
        T: int,
        N: int,
    ) -> torch.Tensor:
        """Packed equivalent of _fuse_cond.  Operates on (A, C) node representations."""
        C = node_repr.size(-1)
        batch_idx = torch.div(active_indices, N, rounding_mode='floor')

        # --- Additive bias path (lightweight): direct packed gather ----------
        # Avoid materializing dense (B, T, N, C) cond tokens in the packed add
        # route; only the active-node representations are needed.
        if self.cond_fusion_mode == 'add':
            if self.cond_add_bias is None or self.cond_projection is None:
                return node_repr

            cond_repr_raw = self._prepare_cond_repr_packed_add(
                cond=cond,
                active_indices=active_indices,
                batch_idx=batch_idx,
                B=B,
                T=T,
                N=N,
            )
            if cond_repr_raw is None:
                return node_repr
            cond_repr = self.cond_projection(cond_repr_raw)  # (A, C)
            bias = self.cond_add_bias(cond_repr)             # (A, 1)
            return node_repr + bias

        # Prepare dense cond tokens, then pack to (A, T, C_cond).
        cond_tokens_dense = self._prepare_cond_tokens(cond, num_nodes=N, num_timesteps=T)
        if cond_tokens_dense is None:
            return node_repr
        C_cond = cond_tokens_dense.size(-1)
        cond_bn = cond_tokens_dense.permute(0, 2, 1, 3).reshape(B * N, T, C_cond)
        cond_packed = cond_bn.index_select(0, active_indices)  # (A, T, C_cond)

        if self.cond_fusion_mode == 'cross_attention':
            if self.cond_attention is None or self.cond_projection is None:
                return node_repr
            cond_packed = self.cond_projection(cond_packed)  # (A, T, C)
            q = node_repr.unsqueeze(1)                       # (A, 1, C)
            attn_out, _ = self.cond_attention(q, cond_packed, cond_packed, need_weights=False)
            return node_repr + attn_out.squeeze(1)

        if self.cond_fusion_mode == 'film':
            if self.cond_film is None or self.cond_projection is None:
                return node_repr
            cond_packed = self.cond_projection(cond_packed)  # (A, T, C)
            cond_repr = cond_packed.mean(dim=1)              # (A, C)
            film_params = self.cond_film(cond_repr)          # (A, 2C)
            gamma, beta = film_params.chunk(2, dim=-1)
            return node_repr * (1.0 + gamma) + beta

        return node_repr

    @staticmethod
    def _prepare_cond_repr_packed_add(
        *,
        cond: torch.Tensor,
        active_indices: torch.Tensor,
        batch_idx: torch.Tensor,
        B: int,
        T: int,
        N: int,
    ) -> Optional[torch.Tensor]:
        """Return packed per-node cond repr (A, C_cond_raw) for add-mode fusion.

        This mirrors _prepare_cond_tokens() semantics followed by temporal mean
        in the dense add path, but avoids constructing dense (B, T, N, C).
        """
        if cond.dim() == 4:
            # Node-specific time-varying/static cond: (B, T_cond, N, C)
            if cond.size(2) == N:
                if cond.size(1) == T:
                    C_cond = cond.size(-1)
                    cond_bn = cond.permute(0, 2, 1, 3).reshape(B * N, T, C_cond)
                    return cond_bn.index_select(0, active_indices).mean(dim=1)
                if cond.size(1) == 1:
                    cond_bn = cond[:, 0, :, :].reshape(B * N, cond.size(-1))
                    return cond_bn.index_select(0, active_indices)
                raise ValueError(
                    "cond temporal dimension must match x temporal dimension "
                    "or be 1 when cond is node-specific."
                )

            # Global-per-graph cond across nodes: (B, T_cond, 1, C)
            if cond.size(2) == 1:
                # Dense path expands over N then averages over T_cond.
                # Equivalent packed path: average over T_cond at batch level,
                # then gather per active node by batch index.
                cond_batch_mean = cond.mean(dim=1).squeeze(1)  # (B, C)
                return cond_batch_mean.index_select(0, batch_idx)

            raise ValueError(
                "Expected cond to be (B, T, N, C), (B, T, 1, C), "
                "or a global-temporal cond that can be broadcast."
            )

        if cond.dim() == 3:
            # Per-node static cond: (B, N, C)
            if cond.size(1) == N:
                cond_bn = cond.reshape(B * N, cond.size(-1))
                return cond_bn.index_select(0, active_indices)

            # Global temporal cond: (B, T_cond, C)
            if cond.size(1) == T:
                cond_batch_mean = cond.mean(dim=1)  # (B, C)
                return cond_batch_mean.index_select(0, batch_idx)

            raise ValueError(
                "Unexpected cond shape (B, ?, C). "
                "For per-node condition, pass (B, N, C). "
                "For global-temporal, pass (B, T, C)."
            )

        if cond.dim() == 2:
            # Global static cond: (B, C)
            return cond.index_select(0, batch_idx)

        return None

    def _compute_scores_packed(
        self,
        x: torch.Tensor,
        active_mask: torch.Tensor,
        cond: Optional[torch.Tensor],
        time_emb: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute per-node scores using packed active-node computation.

        All pointwise operations (temporal aggregation, time FiLM, cond fusion,
        score projection) run on the compact (A, ...) representation where
        A = number of active nodes << B*N.

        Returns:
            (B, N) raw scores with zeros at inactive positions.
        """
        B, T, N, C = x.shape

        active_indices = torch.nonzero(
            active_mask.to(dtype=torch.bool).reshape(-1), as_tuple=False,
        ).squeeze(-1)

        if active_indices.numel() == 0:
            return x.new_zeros(B, N)

        # Pack x: (B, T, N, C) → (A, T, C)
        x_bn = x.permute(0, 2, 1, 3).reshape(B * N, T, C)
        x_packed = x_bn.index_select(0, active_indices)

        # Temporal aggregation: (A, T, C) → (A, C)
        if self.temporal_reduce == 'mean':
            node_repr = x_packed.mean(dim=1)
        elif self.temporal_reduce == 'last':
            node_repr = x_packed[:, -1]
        else:  # 'attn'
            attn_scores = self.temporal_attn_proj(x_packed).squeeze(-1)  # (A, T)
            attn_w = F.softmax(attn_scores, dim=1).unsqueeze(-1)        # (A, T, 1)
            node_repr = (x_packed * attn_w).sum(dim=1)                  # (A, C)

        # Time FiLM — gather per-batch time_emb via batch indices
        if self.time_film is not None and time_emb is not None:
            batch_idx = torch.div(active_indices, N, rounding_mode='floor')
            film_params = self.time_film(time_emb)                   # (B, 2C)
            film_per_node = film_params.index_select(0, batch_idx)   # (A, 2C)
            gamma, beta = film_per_node.chunk(2, dim=-1)
            node_repr = node_repr * (1.0 + gamma) + beta

        # Cond fusion
        self._last_timesteps = T
        if self.cond_fusion and cond is not None:
            node_repr = self._fuse_cond_packed(node_repr, cond, active_indices, B, T, N)

        # Score projection: (A, C) → (A,)
        scores_packed = self.score_proj(node_repr).squeeze(-1)

        # Scatter back to dense (B, N)
        scores = scores_packed.new_zeros(B * N)
        scores.index_copy_(0, active_indices, scores_packed)
        return scores.reshape(B, N)

    def _normalize_scores(
        self,
        scores: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Normalize per-graph over active nodes only.
        active_f = active_mask.float()
        num_active = active_f.sum(dim=1).clamp(min=1.0).unsqueeze(-1)
        active_sum = (scores * active_f).sum(dim=1, keepdim=True)
        active_sq = (scores * scores * active_f).sum(dim=1, keepdim=True)
        mean = active_sum / num_active
        var = (active_sq / num_active) - (mean * mean)
        std = torch.sqrt(torch.clamp(var, min=1e-8))
        std = torch.where(std > 0, std, torch.ones_like(std))
        normed = (scores - mean) / std
        return normed

    def _batched_topk_indices(self, scores: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        B, _ = scores.shape
        max_k = int(k.max().item())
        if max_k <= 0:
            return torch.zeros(B, 0, dtype=torch.long, device=scores.device)
        # Keep rows sorted so truncating to per-sample k[b] stays a true top-k[b].
        # Using sorted=False here can return arbitrary ordering inside top-max_k,
        # which may include lower-ranked nodes in the first k[b] positions.
        topk = torch.topk(scores, k=max_k, dim=1, largest=True, sorted=True).indices
        return topk

    def _indices_to_hard_mask(
        self,
        top_k_indices: torch.Tensor,
        k: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        B = top_k_indices.size(0)
        max_k = top_k_indices.size(1)
        hard_mask = torch.zeros(B, num_nodes, device=top_k_indices.device, dtype=torch.float32)
        if max_k == 0:
            return hard_mask

        row_ids = torch.arange(B, device=top_k_indices.device).unsqueeze(1).expand(B, max_k)
        rank = torch.arange(max_k, device=top_k_indices.device).unsqueeze(0).expand(B, max_k)
        valid = (rank < k.unsqueeze(1)).reshape(-1)
        hard_mask[row_ids.reshape(-1)[valid], top_k_indices.reshape(-1)[valid]] = 1.0
        return hard_mask

    def compute_scores(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        active_mask: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        time_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del edge_index
        del edge_weight
        B, T, N, C = x.shape
        if active_mask is None:
            active_mask = torch.ones(B, N, device=x.device, dtype=torch.bool)

        # --- Packed path: all pointwise ops on compact (A, ...) tensors ---
        use_packed = (
            self.packed_score_mode == 'auto'
            and active_mask.float().mean().item() < self.packed_score_threshold
        )
        if use_packed:
            scores = self._compute_scores_packed(x, active_mask, cond, time_emb)
        else:
            node_repr = self._aggregate_temporal(x)
            self._last_timesteps = T
            node_repr = self._fuse_time(node_repr, time_emb)
            node_repr = self._fuse_cond(node_repr, cond)
            scores = self.score_proj(node_repr).squeeze(-1)

        if active_mask is not None:
            scores = self._normalize_scores(scores, active_mask)
            scores = scores.masked_fill(~active_mask, torch.finfo(scores.dtype).min / 4)
        return scores

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        active_mask: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        time_emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, N, C = x.shape
        self.reset_selector_state()
        self._selector_called_in_last_forward = True

        if active_mask is None:
            active_mask = torch.ones(B, N, device=x.device, dtype=torch.bool)
        else:
            active_mask = active_mask.to(dtype=torch.bool)

        mask_4d = active_mask.unsqueeze(1).unsqueeze(-1).float()
        x = x * mask_4d

        num_active = active_mask.sum(dim=1)
        num_to_keep = torch.floor(num_active.float() * self.pooling_ratio).long()
        # If a graph has no active nodes, do not select any nodes.
        has_active = num_active > 0
        num_to_keep = torch.where(
            has_active,
            torch.maximum(num_to_keep, torch.full_like(num_to_keep, self.min_retained_nodes)),
            torch.zeros_like(num_to_keep),
        )
        num_to_keep = torch.minimum(num_to_keep, num_active)

        scores = self.compute_scores(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            active_mask=active_mask,
            cond=cond,
            time_emb=time_emb,
        )

        max_k = int(num_to_keep.max().item())
        ranking_scores = scores
        current_exploration_noise = self._current_exploration_noise()
        if (
            self.training
            and self.selection_mode in {'soft', 'ste'}
            and current_exploration_noise > 0
        ):
            gumbel = -torch.log(-torch.log(torch.rand_like(scores) + 1e-10) + 1e-10)
            ranking_scores = scores + current_exploration_noise * gumbel

        top_k_indices = self._batched_topk_indices(ranking_scores, num_to_keep)
        hard_mask = self._indices_to_hard_mask(top_k_indices, num_to_keep, N)
        hard_mask = hard_mask * active_mask.float()

        current_temperature = self._current_temperature()
        _LOGIT_CLIP = 20.0
        scaled_scores = torch.clamp(
            scores / current_temperature, -_LOGIT_CLIP, _LOGIT_CLIP
        )
        soft_mask = torch.sigmoid(scaled_scores)
        soft_mask = soft_mask * active_mask.float()

        if self.training:
            if self.selection_mode == 'soft':
                selection_mask = soft_mask
            elif self.selection_mode == 'ste':
                selection_mask = hard_mask + soft_mask - soft_mask.detach()
            else:
                selection_mask = hard_mask
        else:
            selection_mask = hard_mask

        x_pooled = x * selection_mask.unsqueeze(1).unsqueeze(-1)
        new_active_mask = (hard_mask > 0.0)
        new_active_mask = new_active_mask & active_mask

        entropy_reg, entropy_mean_norm = self._entropy_regularizer(soft_mask, active_mask)
        if self.training and self.entropy_reg_weight > 0:
            # Minimize entropy_mean_norm → push soft_mask toward 0 or 1
            # (decisive selection).  The original formulation minimized
            # (1 - entropy_mean_norm), which rewarded uniform / random-dropout
            # masks and actively prevented the selector from learning.
            self._last_aux_loss = entropy_mean_norm * self.entropy_reg_weight
        else:
            self._last_aux_loss = None

        active_counts = num_active.float().clamp(min=1.0)
        selected_counts = new_active_mask.sum(dim=1).float()
        selected_ratio = selected_counts / active_counts
        soft_mask_mean_per_graph = soft_mask.sum(dim=1) / active_counts
        selected_node_counts_per_graph = new_active_mask.float()
        selected_node_counts = selected_node_counts_per_graph.sum(dim=0)
        active_scores = scores[active_mask.bool()]
        active_soft_mask = soft_mask[active_mask.bool()]

        diag = {
            "selection_mode": self.selection_mode,
            "temperature": float(current_temperature),
            "exploration_noise": float(current_exploration_noise),
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
            "num_graphs": float(B),
        }

        if self.collect_heavy_diagnostics:
            diag.update(
                {
                    "selected_node_counts": selected_node_counts.detach(),
                    "selected_node_counts_per_graph": selected_node_counts_per_graph.detach(),
                    "selected_ratio_per_graph": selected_ratio.detach(),
                    "scores_per_graph": scores.detach(),
                    "active_mask_per_graph": active_mask.detach(),
                    "soft_mask_per_graph": soft_mask.detach(),
                    "soft_mask_mean_per_graph": soft_mask_mean_per_graph.detach(),
                }
            )
        if self.collect_sampling_probe:
            diag.update(
                {
                    "probe_active_mask_in": active_mask.detach(),
                    "probe_active_mask_out": new_active_mask.detach(),
                }
            )

        self._last_diagnostics = diag
        return x_pooled, new_active_mask, top_k_indices, scores
