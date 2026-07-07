import logging
import math
from contextlib import contextmanager, nullcontext
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import tqdm

log = logging.getLogger(__name__)

import torch
import torch.nn as nn
from torch_geometric.data import Data

from jaxtyping import Float, Int, jaxtyped
from beartype import beartype

from graph_signal_diffusion.diffusion.base import BaseDiffusion


# ---------------------------------------------------------------------------
# Probe state container — encapsulates all mutable state for selector
# sampling diagnostics so that DDPM.sample / DDIM.sample stay short.
# ---------------------------------------------------------------------------

class _SamplingProbeState:
    """Accumulates per-timestep selector probe statistics during sampling."""

    def __init__(
        self,
        probe_timesteps: List[int],
        num_nodes: int,
        device: torch.device,
        graph_keys: List[str],
    ):
        self.probe_timesteps = probe_timesteps
        self.num_nodes = num_nodes
        self.device = device
        self.step_to_idx: Dict[int, int] = {
            int(t): i for i, t in enumerate(probe_timesteps)
        }
        self.graph_count: Optional[torch.Tensor] = None
        self.selected_sum_by_level: Dict[int, torch.Tensor] = {}
        self.available_sum_by_level: Dict[int, torch.Tensor] = {}

        # Per-network accumulators.
        self.row_indices_by_network: Dict[str, torch.Tensor] = {}
        self.graph_count_by_network: Dict[str, torch.Tensor] = {}
        self.selected_sum_by_level_by_network: Dict[str, Dict[int, torch.Tensor]] = {}
        self.available_sum_by_level_by_network: Dict[str, Dict[int, torch.Tensor]] = {}

        num_probe = len(probe_timesteps)
        for graph_key in sorted(set(graph_keys)):
            row_ids = [i for i, g in enumerate(graph_keys) if g == graph_key]
            if not row_ids:
                continue
            self.row_indices_by_network[graph_key] = torch.tensor(
                row_ids, device=device, dtype=torch.long,
            )
            self.graph_count_by_network[graph_key] = torch.zeros(
                num_probe, device=device, dtype=torch.float32,
            )

    # -- per-timestep accumulation ------------------------------------------

    def accumulate(
        self,
        t_int: int,
        model: nn.Module,
        batch_size: int,
    ) -> None:
        """Accumulate probe diagnostics for a single reverse-process timestep."""
        if t_int not in self.step_to_idx:
            return
        probe_idx = self.step_to_idx[t_int]
        N = self.num_nodes
        num_probe = len(self.probe_timesteps)

        selector_diags = _collect_selector_diags_from_model(model)
        if not selector_diags:
            return

        level_ids = DDPM._selector_probe_level_ids(selector_diags)
        if self.graph_count is None:
            self.graph_count = torch.zeros(
                num_probe, device=self.device, dtype=torch.float32,
            )
        self.graph_count[probe_idx] += float(batch_size)
        for graph_key, row_idx in self.row_indices_by_network.items():
            self.graph_count_by_network[graph_key][probe_idx] += float(row_idx.numel())

        for level_id, diag in zip(level_ids, selector_diags):
            active_in = diag.get("probe_active_mask_in")
            active_out = diag.get("probe_active_mask_out")
            if not isinstance(active_in, torch.Tensor) or not isinstance(active_out, torch.Tensor):
                continue
            if active_in.dim() != 2 or active_out.dim() != 2:
                continue
            if active_in.shape != active_out.shape or active_in.shape[1] != N:
                continue

            active_in_f = active_in.float()
            active_out_f = active_out.float()
            if level_id not in self.selected_sum_by_level:
                self.selected_sum_by_level[level_id] = torch.zeros(
                    (num_probe, N), device=self.device, dtype=torch.float32,
                )
                self.available_sum_by_level[level_id] = torch.zeros(
                    (num_probe, N), device=self.device, dtype=torch.float32,
                )
            self.selected_sum_by_level[level_id][probe_idx] += active_out_f.sum(dim=0)
            self.available_sum_by_level[level_id][probe_idx] += active_in_f.sum(dim=0)

            for graph_key, row_idx in self.row_indices_by_network.items():
                sel_levels = self.selected_sum_by_level_by_network.setdefault(graph_key, {})
                avl_levels = self.available_sum_by_level_by_network.setdefault(graph_key, {})
                if level_id not in sel_levels:
                    sel_levels[level_id] = torch.zeros(
                        (num_probe, N), device=self.device, dtype=torch.float32,
                    )
                    avl_levels[level_id] = torch.zeros(
                        (num_probe, N), device=self.device, dtype=torch.float32,
                    )
                sel_levels[level_id][probe_idx] += active_out_f.index_select(0, row_idx).sum(dim=0)
                avl_levels[level_id][probe_idx] += active_in_f.index_select(0, row_idx).sum(dim=0)

    # -- finalization -------------------------------------------------------

    def finalize(self) -> Dict[str, Any]:
        """Build the final selector sampling diagnostics dictionary."""
        result = self._build_probe_bucket(
            self.selected_sum_by_level,
            self.available_sum_by_level,
            self.graph_count,
            self.probe_timesteps,
        )
        result["by_network"] = {}
        for graph_key in sorted(self.selected_sum_by_level_by_network.keys()):
            network_probe = self._build_probe_bucket(
                self.selected_sum_by_level_by_network[graph_key],
                self.available_sum_by_level_by_network.get(graph_key, {}),
                self.graph_count_by_network.get(graph_key, None),
                self.probe_timesteps,
            )
            if "::" in graph_key:
                dataset_name, network_id = graph_key.split("::", 1)
            else:
                dataset_name, network_id = None, graph_key
            network_probe["network_key"] = graph_key
            network_probe["dataset_name"] = dataset_name
            network_probe["network_id"] = network_id
            result["by_network"][graph_key] = network_probe
        return result

    @staticmethod
    def _build_probe_bucket(
        selected_by_level: Dict[int, torch.Tensor],
        available_by_level: Dict[int, torch.Tensor],
        graph_count: Optional[torch.Tensor],
        probe_timesteps: List[int],
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "probe_timesteps": [int(t) for t in probe_timesteps],
            "probe_graph_count": (
                graph_count.detach().cpu().tolist()
                if isinstance(graph_count, torch.Tensor)
                else [0.0] * len(probe_timesteps)
            ),
            "selected_count_by_level": {},
            "available_count_by_level": {},
            "selection_freq_by_level": {},
            "selection_given_available_by_level": {},
        }
        for level_id in sorted(selected_by_level.keys()):
            selected = selected_by_level[level_id].detach().cpu()
            available = available_by_level[level_id].detach().cpu()
            graphs = (
                graph_count.detach().cpu().clamp(min=1.0)
                if isinstance(graph_count, torch.Tensor)
                else torch.ones(selected.shape[0], dtype=selected.dtype)
            )
            selection_freq = selected / graphs.unsqueeze(-1)
            selection_given_available = torch.full_like(selected, float("nan"))
            valid = available > 0
            selection_given_available[valid] = selected[valid] / available[valid]

            level_key = str(level_id)
            out["selected_count_by_level"][level_key] = selected.tolist()
            out["available_count_by_level"][level_key] = available.tolist()
            out["selection_freq_by_level"][level_key] = selection_freq.tolist()
            out["selection_given_available_by_level"][level_key] = selection_given_available.tolist()
        return out


def _collect_selector_diags_from_model(model: nn.Module) -> List[dict]:
    """Walk *model* modules and return non-empty selector diagnostics dicts."""
    diags: List[dict] = []
    for module in model.modules():
        get_diag = getattr(module, "get_selector_diagnostics", None)
        if callable(get_diag):
            diag = get_diag()
            if isinstance(diag, dict) and len(diag) > 0:
                diags.append(diag)
    return diags


class DDPM(BaseDiffusion):
    """
    Denoising Diffusion Probabilistic Model (Ho et al., 2020).

    Expects the model to have signature: model(x_t, t, data) -> pred
    where pred is parameterized as 'eps' (noise), 'x0' (clean), or 'v' (velocity).
    """
    def __init__(
        self,
        model: nn.Module,
        num_timesteps: int = 1000,
        beta_schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        loss_type: str = "l2",
        parameterization: str = "eps",
        clip_denoised: bool = True,
        cosine_s: float = 0.008,
        compile_model: bool = False,
        **kwargs,
    ):
        super().__init__(model=model, **kwargs)
        
        # Store compile flag - actual compilation should happen after loading weights
        self._compile_requested = compile_model
        self._is_compiled = False
        self.last_selector_diagnostics: dict = {}
        self.last_selector_level_diagnostics: list = []
        self.last_selector_stats_by_level: dict = {}
        self.collect_selector_heavy_diagnostics = True
        
        self.num_timesteps = int(num_timesteps)
        self.loss_type = loss_type
        self.parameterization = parameterization
        self.clip_denoised = clip_denoised

        # Initialize diffusion schedule buffers
        betas = self._make_betas(
            schedule=beta_schedule,
            T=self.num_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            cosine_s=cosine_s,
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones_like(alphas_cumprod[:1]), alphas_cumprod[:-1]], dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped", torch.log(torch.clamp(posterior_variance, min=1e-20)))

        coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_mean_coef1", coef1)
        self.register_buffer("posterior_mean_coef2", coef2)

        # RevIN (Reversible Instance Normalization) — opt-in
        self.revin_enabled = bool(kwargs.get('revin', False))
        self.revin_eps = float(kwargs.get('revin_eps', 1e-5))
        self.revin_cond_feature_idx = int(kwargs.get('revin_cond_feature_idx', 0))
        # RevIN σ blend weight. Linearly blends the per-window σ_cond_w
        # toward the per-stock long-run σ (= 1.0 after the upstream
        # per-stock standardization that the SP500 pipeline applies):
        #
        #     σ_w_blended = w · σ_cond_w + (1 − w) · 1.0
        #
        # Trade-off: w controls how much per-window regime adaptation σ
        # carries. w=1.0 (default) is the current pure-20-day estimator;
        # w=0.0 collapses RevIN's σ to a constant and the model gets all
        # regime info from the conditioning channels (ALR1W/2W/1M/2M,
        # MACD, etc.). Intermediate values reduce the heteroscedasticity
        # bias while retaining most of the regime-adaptive component.
        # See docs/agent_summaries/revin_blend_weight_*.md for the
        # rationale and how `revin_blend_weight` and
        # `revin_sigma_correction` compose (α = 1 / E[σ_w_blended]).
        self.register_buffer(
            "revin_blend_weight",
            torch.tensor(
                float(kwargs.get('revin_blend_weight', 1.0)),
                dtype=torch.float32,
            ),
            persistent=False,
        )
        # RevIN σ scale correction. Multiplies the (possibly blended)
        # per-window σ_w by a constant, so that BOTH normalize
        # (training, ELBO scoring) and denormalize (sampling) use the
        # corrected scale. This compensates for the small-sample +
        # heteroscedasticity bias that makes the past-window σ_cond_w
        # underestimate the per-stock long-run σ — see
        # docs/agent_summaries/sample_trajectory_F3_ep475_2026-05-26.md.
        # Default 1.0 is a no-op (backwards-compat with all existing
        # checkpoints). Registered as a non-persistent buffer so it is NOT
        # saved/loaded in state_dict: the value comes from the diffusion
        # config yaml (or hydra overrides), keeping it under config control
        # rather than embedded in checkpoint weights.
        self.register_buffer(
            "revin_sigma_correction",
            torch.tensor(
                float(kwargs.get('revin_sigma_correction', 1.0)),
                dtype=torch.float32,
            ),
            persistent=False,
        )

    def compile(self) -> bool:
        """
        Apply torch.compile() to the model for faster inference.
        
        This should be called:
        - After loading checkpoint weights (in test.py)
        - Before training starts for fresh models (in train.py)
        
        Returns:
            True if compilation was successful, False otherwise
        """
        if self._is_compiled:
            print("⚠️  Model already compiled, skipping")
            return True
        
        if hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)
            self._is_compiled = True
            print("🚀 Model compiled with torch.compile()")
            return True
        else:
            print("⚠️  torch.compile() not available (requires PyTorch 2.0+)")
            return False

    def set_collect_selector_heavy_diagnostics(self, enabled: bool) -> None:
        """
        Enable/disable heavy selector diagnostics for training/evaluation passes.

        When disabled, selector tensor diagnostics (per-node/per-graph arrays) are
        not propagated to trainer-side state, which avoids frequent GPU->CPU copies.
        """
        enabled = bool(enabled)
        self.collect_selector_heavy_diagnostics = enabled
        for module in self.model.modules():
            set_fn = getattr(module, "set_collect_heavy_diagnostics", None)
            if callable(set_fn):
                set_fn(enabled)

    def set_collect_selector_sampling_probe(self, enabled: bool) -> None:
        """Enable/disable lightweight selector sampling probes in selector modules."""
        enabled = bool(enabled)
        for module in self.model.modules():
            set_fn = getattr(module, "set_collect_sampling_probe", None)
            if callable(set_fn):
                set_fn(enabled)

    @contextmanager
    def _sampling_probe_context(self) -> Iterator[None]:
        """Context manager: enable sampling probes on enter, restore previous state on exit."""
        prev_states: List[Tuple[nn.Module, bool]] = []
        try:
            for module in self.model.modules():
                if hasattr(module, "collect_sampling_probe"):
                    prev_states.append(
                        (module, bool(getattr(module, "collect_sampling_probe")))
                    )
                set_fn = getattr(module, "set_collect_sampling_probe", None)
                if callable(set_fn):
                    set_fn(True)
            yield
        finally:
            for module, prev in prev_states:
                set_fn = getattr(module, "set_collect_sampling_probe", None)
                if callable(set_fn):
                    set_fn(prev)

    def _resolve_probe_timesteps(
        self, selector_probe_timesteps: Optional[List[int]],
    ) -> List[int]:
        """Resolve and deduplicate probe timesteps (descending)."""
        if selector_probe_timesteps is None:
            return self._default_selector_probe_timesteps()
        clamped = [
            int(max(0, min(self.num_timesteps - 1, int(t))))
            for t in selector_probe_timesteps
        ]
        return sorted(set(clamped), reverse=True)

    def _collect_selector_losses_and_diagnostics(self) -> Optional[torch.Tensor]:
        """Gather selector aux losses and diagnostics from model modules.

        Updates ``self.last_selector_diagnostics``,
        ``self.last_selector_level_diagnostics``, and
        ``self.last_selector_stats_by_level``.

        Returns:
            Total auxiliary loss tensor (for backprop), or *None* if no
            selector modules produced losses.
        """
        aux_losses: List[torch.Tensor] = []
        selector_diags: List[dict] = []
        for module in self.model.modules():
            get_aux = getattr(module, "get_selector_aux_loss", None)
            get_diag = getattr(module, "get_selector_diagnostics", None)
            if callable(get_aux):
                loss = get_aux()
                if loss is not None:
                    if not torch.isfinite(loss):
                        log.warning(
                            "Non-finite selector aux loss detected (value=%s), "
                            "sanitized to 0.",
                            loss.item(),
                        )
                        loss = torch.zeros_like(loss)
                    aux_losses.append(loss)
            if callable(get_diag):
                diag = get_diag()
                if diag:
                    selector_diags.append(diag)

        aux_total_tensor = torch.stack(aux_losses).sum() if aux_losses else None
        aux_total_scalar = float(aux_total_tensor.detach().item()) if aux_total_tensor is not None else 0.0

        if selector_diags:
            self.last_selector_level_diagnostics = []
            for idx, d in enumerate(selector_diags):
                level_diag: Dict[str, Any] = {"selector_level": int(idx)}
                for k, v in d.items():
                    if isinstance(v, torch.Tensor):
                        if self.collect_selector_heavy_diagnostics:
                            level_diag[k] = v.detach().cpu()
                    elif isinstance(v, (float, int, str, bool)):
                        level_diag[k] = v
                self.last_selector_level_diagnostics.append(level_diag)

            numeric_keys = [
                "temperature", "exploration_noise", "entropy_mean_norm", "entropy_reg",
                "entropy_reg_weight", "num_active_mean", "num_selected_mean",
                "selected_ratio_mean", "score_mean_active", "score_std_active",
                "soft_mask_mean_active", "soft_mask_std_active", "score_gain",
            ]
            stats_by_level: Dict[str, Dict[int, float]] = {k: {} for k in numeric_keys}
            agg: Dict[str, Any] = {k: 0.0 for k in numeric_keys}
            for idx, d in enumerate(selector_diags):
                for k in numeric_keys:
                    v = float(d.get(k, 0.0))
                    if not math.isfinite(v):
                        v = 0.0
                    agg[k] += v
                    stats_by_level[k][int(idx)] = v
            n = float(len(selector_diags))
            for k in numeric_keys:
                agg[k] /= n
            agg["num_selector_levels"] = int(len(selector_diags))
            agg["selector_aux_total"] = aux_total_scalar
            agg["stats_by_level"] = stats_by_level
            self.last_selector_stats_by_level = stats_by_level
            self.last_selector_diagnostics = agg
        else:
            self.last_selector_level_diagnostics = []
            self.last_selector_stats_by_level = {}
            self.last_selector_diagnostics = {
                "num_selector_levels": 0,
                "selector_aux_total": aux_total_scalar,
                "stats_by_level": {},
            }

        return aux_total_tensor

    def _default_selector_probe_timesteps(self) -> List[int]:
        """
        Return default probe timesteps spanning early/mid/late denoising phases.

        Timesteps are in descending DDPM order (high noise -> low noise).
        """
        T = int(self.num_timesteps)
        if T <= 1:
            return [0]
        candidates = [
            T - 1,
            int(0.9 * (T - 1)),
            int(0.8 * (T - 1)),
            int(0.7 * (T - 1)),
            int(0.6 * (T - 1)),
            int(0.5 * (T - 1)),
            int(0.4 * (T - 1)),
            int(0.3 * (T - 1)),
            int(0.2 * (T - 1)),
            int(0.1 * (T - 1)),
            int(0.05 * (T - 1)),
            0,
        ]
        # Clamp, unique, and keep descending order.
        out = []
        seen = set()
        for t in candidates:
            t_int = int(max(0, min(T - 1, t)))
            if t_int not in seen:
                seen.add(t_int)
                out.append(t_int)
        out.sort(reverse=True)
        return out

    @staticmethod
    def _selector_probe_level_ids(selector_diags: List[dict]) -> List[int]:
        """Resolve stable selector level IDs from current-step selector diagnostics."""
        level_ids: List[int] = []
        for idx, diag in enumerate(selector_diags):
            level_id = diag.get("selector_level", idx)
            try:
                level_ids.append(int(level_id))
            except (TypeError, ValueError):
                level_ids.append(int(idx))
        return level_ids

    @staticmethod
    def _to_python_list(value: Any) -> List[Any]:
        """Convert batched PyG attributes to a flat Python list."""
        if value is None:
            return []
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return []
            return value.detach().cpu().reshape(-1).tolist()
        if isinstance(value, (list, tuple)):
            out: List[Any] = []
            for v in value:
                if isinstance(v, torch.Tensor):
                    out.extend(v.detach().cpu().reshape(-1).tolist())
                else:
                    out.append(v)
            return out
        return [value]

    @classmethod
    def _resolve_probe_graph_keys(cls, data: Optional[Data], batch_size: int) -> List[str]:
        """
        Build per-sample graph keys for sampling diagnostics aggregation.

        Priority:
            1) dataset_name + network_id
            2) network_id
            3) dataset_name + default network_0
            4) fallback network_0 (single static graph)
        """
        if batch_size <= 0:
            return []

        if data is None:
            return ["network_0"] * batch_size

        network_ids = cls._to_python_list(getattr(data, "network_id", None))
        dataset_names = cls._to_python_list(getattr(data, "dataset_name", None))

        if len(network_ids) == 1 and batch_size > 1:
            network_ids = network_ids * batch_size
        if len(dataset_names) == 1 and batch_size > 1:
            dataset_names = dataset_names * batch_size

        if len(network_ids) == 0:
            if len(dataset_names) == batch_size:
                return [f"{str(dataset_names[i])}::network_0" for i in range(batch_size)]
            if len(dataset_names) >= 1:
                return [f"{str(dataset_names[0])}::network_0"] * batch_size
            return ["network_0"] * batch_size

        keys: List[str] = []
        for i in range(batch_size):
            net = network_ids[i] if i < len(network_ids) else network_ids[-1]
            dname = dataset_names[i] if i < len(dataset_names) else (dataset_names[-1] if len(dataset_names) > 0 else None)
            net_tag = str(net)
            if not net_tag.startswith("network_"):
                net_tag = f"network_{net_tag}"
            if dname is not None:
                keys.append(f"{str(dname)}::{net_tag}")
            else:
                keys.append(net_tag)
        return keys

    @staticmethod
    def _make_betas(
        schedule: str,
        T: int,
        beta_start: float,
        beta_end: float,
        cosine_s: float = 0.008,
    ) -> torch.Tensor:
        """Create noise schedule."""
        # ...existing code...
        if schedule == "linear":
            return torch.linspace(beta_start, beta_end, T, dtype=torch.float32)
        elif schedule == "cosine":
            steps = T + 1
            x = torch.linspace(0, T, steps, dtype=torch.float32)
            f = lambda t: math.cos((t / T + cosine_s) / (1 + cosine_s) * math.pi / 2) ** 2
            alphas_bar = torch.tensor([f(t) for t in x], dtype=torch.float32)
            alphas_bar = alphas_bar / alphas_bar[0]
            betas = torch.zeros(T, dtype=torch.float32)
            for t in range(T):
                betas[t] = 1 - alphas_bar[t + 1] / alphas_bar[t]
            return torch.clamp(betas, 1e-8, 0.999)
        else:
            raise ValueError(f"Unknown beta_schedule: {schedule}")
        

    @jaxtyped(typechecker=beartype)
    def add_noise(
        self,
        x0: Float[torch.Tensor, "batch channels nodes timesteps"],
        t: Int[torch.Tensor, "batch"],
        noise: Optional[Float[torch.Tensor, "batch channels nodes timesteps"]] = None
    ) -> Float[torch.Tensor, "batch channels nodes timesteps"]:
        """Add noise to clean samples according to DDPM forward process."""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self._gather(self.sqrt_alphas_cumprod, t).view(-1, 1, 1, 1)
        sqrt_omb = self._gather(self.sqrt_one_minus_alphas_cumprod, t).view(-1, 1, 1, 1)
        return sqrt_ab * x0 + sqrt_omb * noise

    # ------------------------------------------------------------------
    # RevIN helpers
    # ------------------------------------------------------------------

    def _resolve_revin_target_idx(self, data) -> int:
        """Resolve target feature index from data metadata, with config fallback.

        Priority:
          1. data.target_column_idx  (top-level attr, survives PyG batching)
          2. data.info.target_feature_idx  (canonical key, requires reprocessing)
          3. data.info.Target_column_idx   (legacy key)
          4. self.revin_cond_feature_idx   (config fallback)

        For batched PyG data, validates all values in the batch are identical.
        """
        # 1. Top-level attribute (preferred — always survives PyG batching)
        top_level = getattr(data, 'target_column_idx', None)
        if top_level is not None:
            return self._extract_scalar_idx(top_level, 'target_column_idx')

        # 2-3. Info dict keys (may be excluded by dataloader exclude_keys)
        info = getattr(data, 'info', None)
        if info is not None:
            for key in ('target_feature_idx', 'Target_column_idx'):
                idx = info.get(key) if isinstance(info, dict) else getattr(info, key, None)
                if idx is not None:
                    return self._extract_scalar_idx(idx, key)

        # 4. Config fallback
        return self.revin_cond_feature_idx

    @staticmethod
    def _extract_scalar_idx(idx, key: str) -> int:
        """Extract a scalar int from a possibly-batched metadata value."""
        if isinstance(idx, torch.Tensor):
            if idx.numel() == 0:
                raise ValueError(f"Empty tensor for target index metadata '{key}'.")
            idx_flat = idx.reshape(-1)
            if idx_flat.numel() > 1 and not (idx_flat == idx_flat[0]).all():
                raise ValueError(
                    f"Mixed target feature indices in batch for '{key}': "
                    f"{idx_flat.unique().tolist()}. "
                    f"All samples in a batch must share the same target feature index."
                )
            return int(idx_flat[0].item())
        if isinstance(idx, (list, tuple)):
            if len(idx) == 0:
                raise ValueError(f"Empty list/tuple for target index metadata '{key}'.")
            if len(set(idx)) > 1:
                raise ValueError(
                    f"Mixed target feature indices in batch for '{key}': "
                    f"{sorted(set(idx))}."
                )
            return int(idx[0])
        return int(idx)

    def _compute_revin_stats(self, cond, target_idx):
        """Compute per-window, per-stock mean and std from conditioning target channel.

        Args:
            cond: [B, T_obs, N, F_obs] conditioning tensor (standardized space)
            target_idx: index of target feature in F_obs dimension

        Returns:
            mu_w: [B, 1, N, 1]
            sigma_w: [B, 1, N, 1]
        """
        F_obs = cond.shape[-1]
        if target_idx < 0 or target_idx >= F_obs:
            raise IndexError(
                f"RevIN target_idx={target_idx} out of bounds for conditioning "
                f"with {F_obs} features (shape {cond.shape}). "
                f"Check dataset metadata or revin_cond_feature_idx config."
            )
        cond_target = cond[:, :, :, target_idx]             # [B, T_obs, N]
        mu_w = cond_target.mean(dim=1, keepdim=True)         # [B, 1, N]
        sigma_w = cond_target.std(dim=1, unbiased=False, keepdim=True)  # [B, 1, N]
        sigma_w = torch.nan_to_num(sigma_w, nan=0.0)
        sigma_w = sigma_w.clamp(min=self.revin_eps)
        # Shrink the per-window σ_w toward the per-stock long-run σ (=1.0
        # after upstream per-stock standardization). w=1.0 → pure 20-day
        # (current behavior); w=0.0 → constant 1.0 (RevIN becomes a no-op
        # on scale). See revin_blend_weight docstring in __init__.
        w = self.revin_blend_weight
        sigma_w = w * sigma_w + (1.0 - w) * 1.0
        # Scale-correct σ_w so that both normalize and denormalize use σ_w × α.
        # Applied AFTER the blend so α compensates the residual bias of the
        # blended estimator: α = 1 / E[σ_w_blended] = 1 / (1 - w·(1 - E[σ_w_20d])).
        # See revin_sigma_correction docstring in __init__.
        sigma_w = sigma_w * self.revin_sigma_correction
        return mu_w.unsqueeze(-1), sigma_w.unsqueeze(-1)     # [B, 1, N, 1]

    def _revin_normalize(self, x0, mu_w, sigma_w):
        """Normalize x0 into RevIN space."""
        return (x0 - mu_w) / sigma_w

    def _revin_denormalize(self, x, mu_w, sigma_w):
        """Denormalize x from RevIN space back to global-standardized space."""
        return x * sigma_w + mu_w

    def _maybe_revin_normalize(self, x0, ut, data):
        """Apply RevIN normalization to x0 if enabled.

        Must be called AFTER x0 and ut have been reshaped to
        [B, T, N, F] and [B, T_obs, N, F_obs] respectively.

        Returns:
            x0: normalized x0 (or unchanged if RevIN disabled)
            mu_w: [B, 1, N, 1] or None
            sigma_w: [B, 1, N, 1] or None
        """
        if not self.revin_enabled:
            return x0, None, None
        if ut is None:
            raise ValueError("RevIN requires conditioning tensor (data.x)")
        target_idx = self._resolve_revin_target_idx(data)
        mu_w, sigma_w = self._compute_revin_stats(ut, target_idx)
        x0 = self._revin_normalize(x0, mu_w, sigma_w)
        return x0, mu_w, sigma_w

    def training_loss(
        self,
        data: Data,
        return_pred_stats: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        """Compute DDPM training loss on graph data.

        Args:
            data: PyG batch.
            return_pred_stats: When True, also return marginal mean/std
                diagnostics for the injected noise, the model prediction,
                the regression target, and the residual (pred - target).
                Used to detect noise-scale miscalibration (e.g., pred_std
                systematically != 1 under eps-parameterization).

        Returns:
            Either ``loss`` (default), or ``(loss, pred_stats)`` when
            ``return_pred_stats`` is True.
        """
        # ...existing implementation...
        # y0 = data.y.reshape(data.num_graphs, 1, -1, data.y.size(1))

        B = data.num_graphs
        N = data.y.size(0) // B  # Assuming uniform nodes per graph
        T = data.y.size(1)
        F = data.y.size(2)

        # Reshape from [B*N, T, F] to [B, N, T, F] then permute to [B, T, N, F]
        x0 = data.y.view(B, N, T, F).permute(0, 2, 1, 3)  # [B, T, N, F]

        self.debug_print(f"x0 shape: {x0.shape} = [B={B}, T={T}, N={N}, F={F}]")

        device = x0.device
        t = torch.randint(0, self.num_timesteps, (B,), device=device, dtype=torch.long)

        # Reshape conditioning BEFORE RevIN so stats can be computed
        ut = data.x.view(B, N, *data.x.shape[1:]).swapaxes(1, 2) if data.x is not None else None  # [B, T_obs, N, F_obs]

        # RevIN: normalize x0 BEFORE noise addition so xt and target share the same space
        x0, _, _ = self._maybe_revin_normalize(x0, ut, data)

        noise = torch.randn_like(x0)
        xt = self.add_noise(x0, t, noise)

        # Reset cached selector state so diagnostics/aux losses are from this forward only.
        for module in self.model.modules():
            reset_fn = getattr(module, "reset_selector_state", None)
            if callable(reset_fn):
                reset_fn()

        pred, _ = self.model(
            x = xt,
            timesteps = t,
            edge_index = data.edge_index,
            edge_weight = data.edge_weight if hasattr(data, "edge_weight") else None,
            cond = ut,
            return_intermediates = False,
        )

        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x0
        elif self.parameterization == "v":
            a_bar = self._gather(self.alphas_cumprod, t).view(-1, 1, 1, 1)
            sqrt_ab = torch.sqrt(a_bar)
            sqrt_omb = torch.sqrt(1.0 - a_bar)
            target = sqrt_ab * noise - sqrt_omb * x0
        else:
            raise ValueError(f"Unknown parameterization: {self.parameterization}")

        if self.loss_type == "l2":
            loss = torch.nn.functional.mse_loss(pred, target)
        elif self.loss_type == "l1":
            loss = torch.nn.functional.l1_loss(pred, target)
        elif self.loss_type == "huber":
            loss = torch.nn.functional.smooth_l1_loss(pred, target)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        # Selector auxiliary losses & diagnostics.
        aux_total = self._collect_selector_losses_and_diagnostics()
        if aux_total is not None:
            loss = loss + aux_total

        if return_pred_stats:
            pred_stats = self._compute_pred_stats_marginal(noise, pred, target)
            return loss, pred_stats
        return loss

    @staticmethod
    def _compute_pred_stats_marginal(
        noise: torch.Tensor,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, float]:
        """Marginal mean/std of noise, pred, target, and the residual.

        All stats are computed in float32 (AMP-safe). For eps-parameterization
        the expected values are:
            noise_mean ≈ 0, noise_std ≈ 1  (by construction; sanity check)
            target_mean ≈ 0, target_std ≈ 1  (target = noise)
            pred_mean ≈ 0, pred_std ≈ 1  (if model is well-calibrated)
            residual_std ≈ sqrt(MSE_loss)  (unexplained noise)
        A systematic pred_std < 1 indicates the noise predictor is
        under-shooting the noise scale.
        """
        nf = noise.detach().float()
        pf = pred.detach().float()
        tf = target.detach().float()
        return {
            "noise_mean": float(nf.mean().item()),
            "noise_std": float(nf.std().item()),
            "pred_mean": float(pf.mean().item()),
            "pred_std": float(pf.std().item()),
            "target_mean": float(tf.mean().item()),
            "target_std": float(tf.std().item()),
            "residual_std": float((pf - tf).std().item()),
        }

    def training_loss_stratified_t(
        self,
        data: Data,
        num_bins: int = 3,
        return_pred_stats: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, Dict[str, List[float]]],
        Tuple[torch.Tensor, Dict[str, List[float]], Dict[str, float]],
    ]:
        """Compute DDPM training loss with stratified timestep sampling across bins.

        Partitions the B-graph batch into ``num_bins`` equal segments and forces
        each segment's timesteps to fall in a distinct band of [0, num_timesteps).
        This lets callers compute per-bin mean/std loss statistics at zero extra
        forward-pass cost.

        Bin labels (low t → high t):
            * ``"t_low_noise"``  — bin 0, t ∈ [0,         T/num_bins)
            * ``"t_mid"``        — bin 1, t ∈ [T/num_bins, 2T/num_bins)   (only if num_bins >= 2)
            * ``"t_high_noise"`` — bin N-1, t ∈ [(N-1)T/N, T)

        The returned scalar loss is identical in expectation to ``training_loss``
        (same model, same reduction — mean over B×T_seq×N×F), so it can be used
        as a drop-in replacement in evaluation loops without changing ``val_loss``.

        Args:
            data: PyG batch.
            num_bins: Number of equally-sized timestep bins (default 3).
            return_pred_stats: When True, also return marginal-and-per-bin
                stats for noise / pred / target / residual. See
                ``_compute_pred_stats_marginal`` for the stat definitions.

        Returns:
            ``(scalar_loss, bin_losses)`` by default, or
            ``(scalar_loss, bin_losses, pred_stats)`` when ``return_pred_stats``
            is True. ``pred_stats`` includes marginal keys (e.g. ``pred_std``)
            and per-bin keys (e.g. ``pred_std_t_low_noise``).
        """
        B = data.num_graphs
        N = data.y.size(0) // B
        T_seq = data.y.size(1)
        F = data.y.size(2)

        x0 = data.y.view(B, N, T_seq, F).permute(0, 2, 1, 3)  # [B, T_seq, N, F]
        device = x0.device

        # Build stratified timestep vector: graph i → bin b = (i * num_bins) // B
        bin_size = self.num_timesteps // num_bins
        t_list = []
        for i in range(B):
            b = (i * num_bins) // B
            t_lo = b * bin_size
            t_hi = (b + 1) * bin_size if b < num_bins - 1 else self.num_timesteps
            t_list.append(torch.randint(t_lo, t_hi, (1,), device=device, dtype=torch.long))
        t = torch.cat(t_list, dim=0)  # [B]

        # Reshape conditioning BEFORE RevIN so stats can be computed
        ut = data.x.view(B, N, *data.x.shape[1:]).swapaxes(1, 2) if data.x is not None else None

        # RevIN: normalize x0 BEFORE noise addition so xt and target share the same space
        x0, _, _ = self._maybe_revin_normalize(x0, ut, data)

        noise = torch.randn_like(x0)
        xt = self.add_noise(x0, t, noise)

        # Reset cached selector state so diagnostics/aux losses are from this forward only.
        for module in self.model.modules():
            reset_fn = getattr(module, "reset_selector_state", None)
            if callable(reset_fn):
                reset_fn()

        pred, _ = self.model(
            x=xt,
            timesteps=t,
            edge_index=data.edge_index,
            edge_weight=data.edge_weight if hasattr(data, "edge_weight") else None,
            cond=ut,
            return_intermediates=False,
        )

        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x0
        elif self.parameterization == "v":
            a_bar = self._gather(self.alphas_cumprod, t).view(-1, 1, 1, 1)
            sqrt_ab = torch.sqrt(a_bar)
            sqrt_omb = torch.sqrt(1.0 - a_bar)
            target = sqrt_ab * noise - sqrt_omb * x0
        else:
            raise ValueError(f"Unknown parameterization: {self.parameterization}")

        # Per-graph reconstruction loss — shape [B] (no aux loss here).
        if self.loss_type == "l2":
            per_graph_loss = torch.nn.functional.mse_loss(pred, target, reduction="none").mean(dim=[1, 2, 3])
        elif self.loss_type == "l1":
            per_graph_loss = torch.nn.functional.l1_loss(pred, target, reduction="none").mean(dim=[1, 2, 3])
        elif self.loss_type == "huber":
            per_graph_loss = torch.nn.functional.smooth_l1_loss(pred, target, reduction="none").mean(dim=[1, 2, 3])
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        scalar_loss = per_graph_loss.mean()

        # Selector auxiliary losses & diagnostics (added to scalar only).
        aux_total = self._collect_selector_losses_and_diagnostics()
        if aux_total is not None:
            scalar_loss = scalar_loss + aux_total

        # Build bin label map (consistent regardless of num_bins).
        _BIN_LABELS = ["t_low_noise", "t_mid", "t_high_noise"]

        def _bin_label(b: int, total: int) -> str:
            if total == 1:
                return "t_mid"
            if total == 2:
                return "t_low_noise" if b == 0 else "t_high_noise"
            # For 3+ bins: first → low_noise, last → high_noise, rest → mid (suffixed)
            if b == 0:
                return "t_low_noise"
            if b == total - 1:
                return "t_high_noise"
            return "t_mid" if total == 3 else f"t_mid_{b}"

        bin_losses: Dict[str, List[float]] = {}
        per_graph_loss_cpu = per_graph_loss.detach().cpu().tolist()
        for i in range(B):
            b = (i * num_bins) // B
            label = _bin_label(b, num_bins)
            bin_losses.setdefault(label, []).append(per_graph_loss_cpu[i])

        if return_pred_stats:
            pred_stats = self._compute_pred_stats_marginal(noise, pred, target)
            # Per-bin stats: partition the batch dim by the same rule the
            # timestep was assigned, then take element-pooled mean/std within
            # each bin. Cheap — a few index_select reductions per call.
            nf = noise.detach().float()
            pf = pred.detach().float()
            tf = target.detach().float()
            rf = pf - tf
            for b in range(num_bins):
                bin_indices = [i for i in range(B) if (i * num_bins) // B == b]
                if not bin_indices:
                    continue
                label = _bin_label(b, num_bins)
                idx = torch.tensor(bin_indices, device=nf.device, dtype=torch.long)
                pred_stats[f"noise_mean_{label}"] = float(nf.index_select(0, idx).mean().item())
                pred_stats[f"noise_std_{label}"] = float(nf.index_select(0, idx).std().item())
                pred_stats[f"pred_mean_{label}"] = float(pf.index_select(0, idx).mean().item())
                pred_stats[f"pred_std_{label}"] = float(pf.index_select(0, idx).std().item())
                pred_stats[f"residual_std_{label}"] = float(rf.index_select(0, idx).std().item())
            return scalar_loss, bin_losses, pred_stats

        return scalar_loss, bin_losses

    @torch.inference_mode()
    def compute_elbo_per_trajectory(
        self, data: Data, n_mc_samples: int = 16,
    ) -> torch.Tensor:
        """Monte-Carlo NLL proxy: per-trajectory reconstruction loss.

        Returns a ``[B]`` tensor of per-graph scores (lower = more likely
        under the learned denoiser).  This is a simplified ELBO surrogate,
        **not** a strict variational bound — labelled ``diffusion_nll_proxy``
        to avoid overclaiming.

        Uses **stratified t-sampling**: the diffusion timestep range
        ``[0, num_timesteps)`` is partitioned into ``K = n_mc_samples``
        roughly-equal bins and at iteration ``k`` each trajectory draws
        ``t`` uniformly from bin ``k`` (independently per trajectory). The
        estimator is unbiased (each bin carries the correct 1/K probability
        mass) and has lower variance than independent uniform draws — the
        dominant source of MC noise is across-t variance (per-step loss
        varies ~20× across noise levels on SP500), which stratification
        collapses to intra-bin variance.

        If ``n_mc_samples > num_timesteps`` the function falls back to
        independent uniform sampling (stratification would require empty
        bins).

        Only called from ``DiffusionBaseline`` during ``compare_baselines``;
        never invoked during training.
        """
        B = data.num_graphs
        N = data.y.size(0) // B
        T = data.y.size(1)
        F = data.y.size(2)

        x0 = data.y.view(B, N, T, F).permute(0, 2, 1, 3)  # [B, T, N, F]
        device = x0.device

        ut = (
            data.x.view(B, N, *data.x.shape[1:]).swapaxes(1, 2)
            if data.x is not None else None
        )

        # RevIN: normalize x0 using per-window conditioning stats
        x0, _, _ = self._maybe_revin_normalize(x0, ut, data)

        T_diff = int(self.num_timesteps)
        K = int(n_mc_samples)
        stratified = K <= T_diff
        if stratified:
            # Integer-exact bin edges so every bin has width >= 1.
            bin_edges = [k * T_diff // K for k in range(K + 1)]
        else:
            bin_edges = None  # marker — fall back to uniform

        proxy_accum = torch.zeros(B, device=device)
        for k in tqdm.tqdm(range(K), desc="ELBO estimator MC samples", unit="sample"):
            if stratified:
                lo, hi = bin_edges[k], bin_edges[k + 1]
                t = torch.randint(lo, hi, (B,), device=device, dtype=torch.long)
            else:
                t = torch.randint(0, T_diff, (B,), device=device, dtype=torch.long)
            noise = torch.randn_like(x0)
            xt = self.add_noise(x0, t, noise)

            for module in self.model.modules():
                reset_fn = getattr(module, "reset_selector_state", None)
                if callable(reset_fn):
                    reset_fn()

            pred, _ = self.model(
                x=xt,
                timesteps=t,
                edge_index=data.edge_index,
                edge_weight=data.edge_weight if hasattr(data, "edge_weight") else None,
                cond=ut,
                return_intermediates=False,
            )

            if self.parameterization == "eps":
                target = noise
            elif self.parameterization == "x0":
                target = x0
            elif self.parameterization == "v":
                a_bar = self._gather(self.alphas_cumprod, t).view(-1, 1, 1, 1)
                sqrt_ab = torch.sqrt(a_bar)
                sqrt_omb = torch.sqrt(1.0 - a_bar)
                target = sqrt_ab * noise - sqrt_omb * x0
            else:
                raise ValueError(f"Unknown parameterization: {self.parameterization}")

            if self.loss_type == "l2":
                per_graph = torch.nn.functional.mse_loss(pred, target, reduction="none").mean(dim=[1, 2, 3])
            elif self.loss_type == "l1":
                per_graph = torch.nn.functional.l1_loss(pred, target, reduction="none").mean(dim=[1, 2, 3])
            elif self.loss_type == "huber":
                per_graph = torch.nn.functional.smooth_l1_loss(pred, target, reduction="none").mean(dim=[1, 2, 3])
            else:
                raise ValueError(f"Unknown loss_type: {self.loss_type}")

            proxy_accum += per_graph

        return (proxy_accum / n_mc_samples) * self.num_timesteps

    @torch.inference_mode()
    def sample(
        self,
        shape: Tuple[int, int, int, int],
        device: torch.device,
        data: Optional[Data] = None,
        use_amp: bool = False,
        return_selector_sampling_diagnostics: bool = False,
        selector_probe_timesteps: Optional[List[int]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[Dict[str, Any]]]]:
        """Generate samples using DDPM reverse process.

        Args:
            shape: (batch, timesteps, nodes, features)
            device: Device to run on
            data: Graph data for conditional generation
            use_amp: Use automatic mixed precision (FP16) for faster inference
            return_selector_sampling_diagnostics: If True, also return a compact
                selector diagnostics dictionary aggregated across probe timesteps.
            selector_probe_timesteps: Optional explicit probe timesteps in DDPM index
                space [0, T-1]. If None, a default phase-spanning set is used.

        Returns:
            Generated samples, and optionally sampling diagnostics.
        """
        autocast_ctx = torch.amp.autocast(device_type=device.type, dtype=torch.float16) if (use_amp and device.type == "cuda") else nullcontext()
        probe_ctx = self._sampling_probe_context() if return_selector_sampling_diagnostics else nullcontext()
        probe_state: Optional[_SamplingProbeState] = None

        B, N = shape[0], shape[2]

        with probe_ctx:
            if return_selector_sampling_diagnostics:
                probe_timesteps = self._resolve_probe_timesteps(selector_probe_timesteps)
                graph_keys = self._resolve_probe_graph_keys(data, B)
                probe_state = _SamplingProbeState(
                    probe_timesteps=probe_timesteps,
                    num_nodes=N,
                    device=device,
                    graph_keys=graph_keys,
                )

            with autocast_ctx:
                x_t = torch.randn(shape, device=device)

                # Conditioning and graph structure are static across reverse steps.
                edge_index = data.edge_index if data is not None else None
                edge_weight = data.edge_weight if data is not None and hasattr(data, "edge_weight") else None
                u_0 = data.x.view(B, N, *data.x.shape[1:]).swapaxes(1, 2) if data is not None and hasattr(data, "x") and data.x is not None else None  # [B, T_obs, N, F_obs]

                # RevIN: compute stats from conditioning for denormalization after sampling
                revin_mu, revin_sigma = None, None
                if self.revin_enabled:
                    if u_0 is None:
                        raise ValueError("RevIN requires conditioning tensor (data.x)")
                    target_idx = self._resolve_revin_target_idx(data)
                    revin_mu, revin_sigma = self._compute_revin_stats(u_0, target_idx)

                for t_int in tqdm.tqdm(reversed(range(self.num_timesteps)), desc="DDPM Sampling", unit="step"):
                    t = torch.full((B,), t_int, device=device, dtype=torch.long)
                    pred, _ = self.model(
                        x=x_t, timesteps=t, edge_index=edge_index,
                        edge_weight=edge_weight, cond=u_0,
                        return_intermediates=False,
                    )
                    x0_pred, _ = self._pred_to_x0_eps(x_t, t, pred)
                    mean = self._posterior_mean(x0_pred, x_t, t)

                    if t_int > 0:
                        var = self._gather(self.posterior_variance, t).view(-1, 1, 1, 1)
                        noise = torch.randn_like(x_t)
                        x_t = mean + torch.sqrt(var) * noise
                    else:
                        x_t = mean

                    if probe_state is not None:
                        probe_state.accumulate(t_int, self.model, B)

        # RevIN: denormalize from RevIN space back to global-standardized space
        if self.revin_enabled:
            x_t = self._revin_denormalize(x_t, revin_mu, revin_sigma)

        if probe_state is not None:
            return x_t, probe_state.finalize()
        return x_t


    @jaxtyped(typechecker=beartype)
    def _pred_to_x0_eps(
        self,
        x_t: Float[torch.Tensor, "batch timesteps nodes features"],
        t: Int[torch.Tensor, "batch"],
        pred: Float[torch.Tensor, "batch timesteps nodes features"]
    ) -> Tuple[Float[torch.Tensor, "batch timesteps nodes features"], Float[torch.Tensor, "batch timesteps nodes features"]]:
        
        """Convert model prediction to x0 and epsilon predictions."""
        a_bar = self._gather(self.alphas_cumprod, t).view(-1, 1, 1, 1)
        sqrt_ab = torch.sqrt(a_bar)
        sqrt_omb = torch.sqrt(1.0 - a_bar)

        if self.parameterization == "eps":
            eps_pred = pred
            x0_pred = (x_t - sqrt_omb * eps_pred) / sqrt_ab
        elif self.parameterization == "x0":
            x0_pred = pred
            sqrt_omb_safe = sqrt_omb.clamp_min(1e-12)
            eps_pred = (x_t - sqrt_ab * x0_pred) / sqrt_omb_safe
        elif self.parameterization == "v":
            v = pred
            x0_pred = sqrt_ab * x_t - sqrt_omb * v
            eps_pred = sqrt_ab * v + sqrt_omb * x_t
        else:
            raise ValueError(f"Unknown parameterization: {self.parameterization}")

        if self.clip_denoised:
            x0_pred = x0_pred.clamp(-0.5, 0.5)
        return x0_pred, eps_pred

    def _posterior_mean(self, x0_pred: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute posterior mean q(x_{t-1} | x_t, x_0)."""
        coef1 = self._gather(self.posterior_mean_coef1, t).view(-1, 1, 1, 1)
        coef2 = self._gather(self.posterior_mean_coef2, t).view(-1, 1, 1, 1)
        return coef1 * x0_pred + coef2 * x_t

    @staticmethod
    def _gather(buffer: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Gather values from buffer at indices t."""
        return buffer[t]
