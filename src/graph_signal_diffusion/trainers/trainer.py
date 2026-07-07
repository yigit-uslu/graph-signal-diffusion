import inspect
import os
import random
import math
import time
from collections import defaultdict
from datetime import datetime
import torch
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset
from tqdm import tqdm

from graph_signal_diffusion.diffusion.base import BaseDiffusion
from graph_signal_diffusion.tasks.base import BaseTask
from graph_signal_diffusion.trainers.base_trainer import BaseTrainer
from graph_signal_diffusion.trainers.best_model_tracker import BestModelTracker
from graph_signal_diffusion.trainers.eval_schedule import (
    EvalSchedule,
    CosineEvalSchedule,
    MultiPhaseEvalSchedule,
    UniformEvalSchedule,
)

import logging 
import json
from hydra.core.hydra_config import HydraConfig

log = logging.getLogger(__name__)


@dataclass
class _SelectorEpochDiagnostics:
    """Finalized selector diagnostics for one training epoch."""
    avg_selector_diags: Dict[str, float]
    avg_selector_stats_by_level: Dict[str, Dict[str, float]]
    selector_network_stats: Dict[str, Dict[str, dict]]
    selector_level_freqs: Dict[int, torch.Tensor]
    selector_network_node_stats: Dict[str, Dict[int, Dict[str, torch.Tensor]]]
    summary_path: Optional[str] = None


class _SelectorDiagnosticsManager:
    """
    Encapsulate selector-diagnostics state and epoch lifecycle.

    Keeps `fit()` focused on training/evaluation flow while centralizing
    diagnostics accumulation, EMA updates, and summary persistence.
    """

    def __init__(self, trainer: "DiffusionTrainer"):
        self.trainer = trainer
        self.selector_freq_ma_state: Dict[str, Dict[int, torch.Tensor]] = {}
        self.selector_freq_tracked_nodes: Dict[str, Dict[int, List[int]]] = {}
        self.selector_freq_rng = random.Random(trainer.selector_freq_track_seed)
        self._epoch_state: Optional[Dict[str, Any]] = None

    def begin_epoch(self) -> None:
        self._epoch_state = self.trainer._init_selector_diag_state()

    def record_forward_pass(self, diffusion: BaseDiffusion, batch_graph_keys: List[str]) -> Optional[dict]:
        if self._epoch_state is None:
            raise RuntimeError("Selector diagnostics epoch state was not initialized.")
        selector_diag = getattr(diffusion, "last_selector_diagnostics", None)
        self.trainer._accumulate_selector_diag(self._epoch_state, selector_diag)
        tracked_network_keys = self.trainer._resolve_selector_diag_tracked_network_keys(
            batch_graph_keys
        )
        selector_level_diags = getattr(diffusion, "last_selector_level_diagnostics", None)
        self.trainer._accumulate_selector_level_diags(
            self._epoch_state,
            selector_level_diags,
            batch_graph_keys,
            tracked_network_keys=tracked_network_keys,
        )
        return selector_diag if isinstance(selector_diag, dict) else None

    def finalize_epoch(self, epoch: int, *, write_network_rows: bool) -> _SelectorEpochDiagnostics:
        if self._epoch_state is None:
            raise RuntimeError("Selector diagnostics epoch state was not initialized.")

        (
            avg_selector_diags,
            avg_selector_stats_by_level,
            selector_network_stats,
            selector_level_freqs,
            selector_network_node_stats,
        ) = self.trainer._finalize_selector_diag_state(self._epoch_state)

        self.trainer._update_selection_frequency_moving_average(
            selector_freq_ma_state=self.selector_freq_ma_state,
            selector_network_node_stats=selector_network_node_stats,
            ema_alpha=self.trainer.selector_freq_ema_alpha,
        )
        self.trainer._update_selection_frequency_tracked_nodes(
            selector_freq_tracked_nodes=self.selector_freq_tracked_nodes,
            selector_freq_ma_state=self.selector_freq_ma_state,
            num_nodes_to_track=self.trainer.selector_freq_track_num_nodes,
            rng=self.selector_freq_rng,
        )

        summary_path = self.trainer._append_selector_diagnostic_summaries_jsonl(
            epoch=epoch,
            avg_selector_diags=avg_selector_diags,
            avg_selector_stats_by_level=avg_selector_stats_by_level,
            selector_level_freqs=selector_level_freqs,
            selector_network_stats=selector_network_stats,
            selector_freq_ma_state=self.selector_freq_ma_state,
            selector_freq_tracked_nodes=self.selector_freq_tracked_nodes,
            write_network_rows=write_network_rows,
        )

        self._epoch_state = None
        return _SelectorEpochDiagnostics(
            avg_selector_diags=avg_selector_diags,
            avg_selector_stats_by_level=avg_selector_stats_by_level,
            selector_network_stats=selector_network_stats,
            selector_level_freqs=selector_level_freqs,
            selector_network_node_stats=selector_network_node_stats,
            summary_path=summary_path,
        )


def _inject_val_loss_gap(epoch_metrics: Dict[str, Any]) -> None:
    """Add ``val_loss_gap = |val_loss − train-val_loss|`` in place.

    No-op when either ``val_loss`` or ``train-val_loss`` is absent or
    non-finite. The metric is intended for use in best-model trackers
    that reward checkpoints with small generalization gaps.
    """
    v = epoch_metrics.get("val_loss")
    tv = epoch_metrics.get("train-val_loss")
    if v is None or tv is None:
        return
    try:
        gap = abs(float(v) - float(tv))
    except (TypeError, ValueError):
        return
    if not math.isfinite(gap):
        return
    epoch_metrics["val_loss_gap"] = gap


class DiffusionTrainer(BaseTrainer):
    def __init__(
        self,
        diffusion: BaseDiffusion,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        accelerator=None,
        log_every_n_steps: int = 50,
        save_dir: Optional[str] = None,
        task: Optional[BaseTask] = None,
        use_amp: bool = False,
        wandb_run=None,
        **kwargs
    ):
        super().__init__(device=device, save_dir=save_dir)
        
        self.diffusion = diffusion
        self.optimizer = optimizer
        self.lr_scheduler = kwargs.get("lr_scheduler", None)
        self.accelerator = accelerator
        self.log_every_n_steps = log_every_n_steps
        self.task = task
        self.epoch = 0  # Will be updated from checkpoint if loading
        # Total planned epochs for the active fit() run.
        # Populated by fit(); None means "no active fit context yet".
        self.max_epochs: Optional[int] = None

        # wandb integration (optional — None when disabled)
        self.wandb_run = wandb_run
        
        # Mixed precision support
        # self.use_amp preserves the user-requested flag (used for checkpointing).
        # self._amp_enabled is the runtime predicate: True only when CUDA is available.
        self.use_amp = use_amp
        self._amp_enabled = bool(use_amp and device.type == "cuda")
        self.scaler = torch.amp.GradScaler('cuda') if self._amp_enabled else None
        if use_amp and not self._amp_enabled:
            log.warning(
                "AMP requested but device is '%s' — AMP requires CUDA. Disabling.",
                device.type,
            )
        if self._amp_enabled:
            log.info("Using Automatic Mixed Precision (AMP) for training")

        self.max_grad_norm = kwargs.get("max_grad_norm", 1.0)  # Default to 1.0 if not provided
        print(f"Gradient clipping max norm set to: {self.max_grad_norm}")

        # Max batches per evaluation call. None/-1 = all batches.
        self.max_num_val_batches = int(kwargs.get("max_num_val_batches", 1))
        self.max_num_train_val_batches = int(kwargs.get("max_num_train_val_batches", 1))
        self.max_num_test_batches = int(kwargs.get("max_num_test_batches", -1))
        self.run_final_test = bool(kwargs.get("run_final_test", True))

        # Optional forward-pass timing during training (for smoke benchmarking).
        # Times `self.diffusion.training_loss(data)` and reports epoch averages.
        self.measure_forward_speed = bool(kwargs.get("measure_forward_speed", False))
        self.forward_speed_warmup_steps = int(kwargs.get("forward_speed_warmup_steps", 5))
        self.forward_speed_max_steps = int(kwargs.get("forward_speed_max_steps", -1))
        if self.forward_speed_warmup_steps < 0:
            raise ValueError(
                f"forward_speed_warmup_steps must be >= 0, got {self.forward_speed_warmup_steps}"
            )
        if self.forward_speed_max_steps < -1:
            raise ValueError(
                f"forward_speed_max_steps must be >= -1, got {self.forward_speed_max_steps}"
            )
        if self.measure_forward_speed and self.device.type != "cuda":
            log.warning(
                "measure_forward_speed=true on non-CUDA device '%s'; timing will use CPU wall-clock.",
                self.device.type,
            )
        if self.measure_forward_speed:
            log.info(
                "Forward speed timing enabled: warmup_steps=%d, max_steps=%d",
                self.forward_speed_warmup_steps,
                self.forward_speed_max_steps,
            )
        if not self.run_final_test:
            log.info("Final test evaluation disabled (run_final_test=false).")

        diagnose_model_cfg = kwargs.get("diagnose_model", None)

        def _cfg_get(cfg_obj: Any, key: str, default: Any) -> Any:
            if isinstance(cfg_obj, dict):
                return cfg_obj.get(key, default)
            get_fn = getattr(cfg_obj, "get", None)
            if callable(get_fn):
                try:
                    return get_fn(key, default)
                except Exception:
                    return default
            return default

        # Selection-frequency EMA diagnostics:
        # alpha=2/(window+1) with window~10 => ~0.1818.
        self.diagnose_model_enabled = bool(
            _cfg_get(
                diagnose_model_cfg,
                "enabled",
                kwargs.get("diagnose_model_enabled", True),
            )
        )
        self.selector_freq_ema_alpha = float(kwargs.get("selector_freq_ema_alpha", 2.0 / 11.0))
        self.selector_freq_track_num_nodes = int(kwargs.get("selector_freq_track_num_nodes", 10))
        self.selector_freq_track_seed = int(kwargs.get("selector_freq_track_seed", 0))
        self.diagnose_selector_every_n_epochs = int(
            _cfg_get(
                diagnose_model_cfg,
                "diagnose_selector_every_n_epochs",
                kwargs.get("diagnose_selector_every_n_epochs", 1),
            )
        )
        self.selector_sampling_max_network_plots = int(
            _cfg_get(
                diagnose_model_cfg,
                "selector_sampling_max_network_plots",
                kwargs.get("selector_sampling_max_network_plots", 4),
            )
        )
        # Cap for per-network selector diagnostic summary rows written to JSONL.
        # -1 means unlimited; 0 disables per-network summary rows.
        self.selector_summary_max_network_rows = int(
            _cfg_get(
                diagnose_model_cfg,
                "selector_summary_max_network_rows",
                kwargs.get("selector_summary_max_network_rows", -1),
            )
        )
        # Nested selector-plot controls under diagnose_model.plot_*.
        def _resolve_selector_plot_cfg(
            key: str,
            *,
            default_enabled: bool,
            default_plot_every_n_epochs: Optional[int],
            default_max_network_plots: Optional[int],
        ) -> Dict[str, Any]:
            plot_cfg_raw = _cfg_get(diagnose_model_cfg, key, None)
            plot_cfg = plot_cfg_raw if plot_cfg_raw is not None else {}
            plot_enabled = bool(_cfg_get(plot_cfg, "enabled", default_enabled))
            every_raw = _cfg_get(plot_cfg, "plot_every_n_epochs", default_plot_every_n_epochs)
            max_raw = _cfg_get(plot_cfg, "max_network_plots", default_max_network_plots)
            return {
                "enabled": plot_enabled,
                "plot_every_n_epochs": (
                    None if every_raw is None else int(every_raw)
                ),
                "max_network_plots": (
                    None if max_raw is None else int(max_raw)
                ),
            }

        selector_snapshot_default_every_n_epochs = (
            int(self.diagnose_selector_every_n_epochs)
            if int(self.diagnose_selector_every_n_epochs) > 0
            else 5
        )
        self.plot_node_stats_cfg = _resolve_selector_plot_cfg(
            "plot_node_stats",
            default_enabled=True,
            default_plot_every_n_epochs=selector_snapshot_default_every_n_epochs,
            default_max_network_plots=-1,
        )
        self.plot_hard_mask_cfg = _resolve_selector_plot_cfg(
            "plot_hard_mask",
            default_enabled=True,
            default_plot_every_n_epochs=selector_snapshot_default_every_n_epochs,
            default_max_network_plots=-1,
        )
        self.plot_freq_evolution_cfg = _resolve_selector_plot_cfg(
            "plot_freq_evolution",
            default_enabled=True,
            default_plot_every_n_epochs=5,
            default_max_network_plots=-1,
        )
        self.plot_scalar_series_cfg = _resolve_selector_plot_cfg(
            "plot_scalar_series",
            default_enabled=True,
            default_plot_every_n_epochs=5,
            default_max_network_plots=None,
        )
        self.plot_sampling_heatmap_cfg = _resolve_selector_plot_cfg(
            "plot_sampling_heatmap",
            default_enabled=True,
            default_plot_every_n_epochs=None,
            default_max_network_plots=self.selector_sampling_max_network_plots,
        )
        # Keep legacy attribute as the effective cap for backward compatibility.
        sampling_cap = self.plot_sampling_heatmap_cfg.get("max_network_plots", None)
        if sampling_cap is None:
            sampling_cap = self.selector_sampling_max_network_plots
        self.selector_sampling_max_network_plots = int(sampling_cap)
        if not (0.0 < self.selector_freq_ema_alpha <= 1.0):
            raise ValueError(
                f"selector_freq_ema_alpha must be in (0, 1], got {self.selector_freq_ema_alpha}"
            )
        if self.selector_freq_track_num_nodes <= 0:
            raise ValueError(
                f"selector_freq_track_num_nodes must be > 0, got {self.selector_freq_track_num_nodes}"
            )
        if self.diagnose_selector_every_n_epochs < 0:
            raise ValueError(
                "diagnose_selector_every_n_epochs must be >= 0, "
                f"got {self.diagnose_selector_every_n_epochs}"
            )
        if self.selector_summary_max_network_rows < -1:
            raise ValueError(
                "selector_summary_max_network_rows must be >= -1, "
                f"got {self.selector_summary_max_network_rows}"
            )
        for plot_name, plot_cfg in (
            ("plot_node_stats", self.plot_node_stats_cfg),
            ("plot_hard_mask", self.plot_hard_mask_cfg),
            ("plot_freq_evolution", self.plot_freq_evolution_cfg),
            ("plot_scalar_series", self.plot_scalar_series_cfg),
            ("plot_sampling_heatmap", self.plot_sampling_heatmap_cfg),
        ):
            every_n = plot_cfg.get("plot_every_n_epochs", None)
            max_networks = plot_cfg.get("max_network_plots", None)
            if every_n is not None and int(every_n) < 0:
                raise ValueError(
                    f"{plot_name}.plot_every_n_epochs must be >= 0 or null, got {every_n}"
                )
            if max_networks is not None and int(max_networks) < -1:
                raise ValueError(
                    f"{plot_name}.max_network_plots must be >= -1 or null, got {max_networks}"
                )
        self._selector_diag_tracked_network_key_order: List[str] = []
        self._selector_diag_tracked_network_key_set: Set[str] = set()
        log.info(
            "Model diagnostics: enabled=%s, diagnose_selector_every_n_epochs=%d, "
            "selector_sampling_max_network_plots=%d, selector_summary_max_network_rows=%d",
            self.diagnose_model_enabled,
            self.diagnose_selector_every_n_epochs,
            self.selector_sampling_max_network_plots,
            self.selector_summary_max_network_rows,
        )
        
        self.DEBUG_FLAG = True  # Set to True to enable debug prints

        # Evaluation scheduling (trainer is sole owner; tasks no longer track this)
        self.eval_every_n_epochs = int(kwargs.get("eval_every_n_epochs", 10))

        # Generalised eval schedule — defaults to UniformEvalSchedule(eval_every_n_epochs)
        # when no eval_schedule config key is present (backward-compatible).
        eval_schedule_cfg = kwargs.get("eval_schedule", None)
        self.eval_schedule: EvalSchedule = EvalSchedule.from_config(
            eval_schedule_cfg,
            fallback_period=self.eval_every_n_epochs,
        )
        # Policy for epoch-0 evaluation lives under eval_schedule.
        eval_on_first_raw = _cfg_get(eval_schedule_cfg, "eval_on_first_epoch", None)
        self.eval_on_first_epoch = bool(
            eval_on_first_raw if eval_on_first_raw is not None else False
        )
        log.info("Eval schedule: %s", self.eval_schedule)
        log.info("Evaluate on epoch 0: %s", self.eval_on_first_epoch)

        # Best-model checkpoint tracker (optional, configured via YAML)
        best_model_cfg = kwargs.get("best_model", None)
        self.best_model_tracker: Optional[BestModelTracker] = None
        if best_model_cfg is not None and _cfg_get(best_model_cfg, "enabled", False):
            try:
                self.best_model_tracker = BestModelTracker.from_config(best_model_cfg)
                log.info(
                    "Best-model tracker enabled: top_k=%d, ema_alpha=%.3f, "
                    "min_warmup=%d, metrics=%s",
                    self.best_model_tracker.top_k,
                    self.best_model_tracker.ema_alpha,
                    self.best_model_tracker.min_warmup_evals,
                    [m.name for m in self.best_model_tracker.metrics],
                )
            except Exception as e:
                log.warning("Failed to initialize best-model tracker: %s", e)
                self.best_model_tracker = None

        # Inject debug_print into diffusion and diffusion.model for internal debugging
        # self.diffusion.debug_print = self.debug_print
        # if hasattr(self.diffusion, "model"):
        #     self.diffusion.model.debug_print = self.debug_print

        # if self.task is not None:
        #     self.task.debug_print = self.debug_print

    @staticmethod
    def _to_python_list(value) -> List:
        """Convert batched PyG attributes to a flat Python list."""
        if value is None:
            return []
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return []
            return value.detach().cpu().reshape(-1).tolist()
        if isinstance(value, (list, tuple)):
            out = []
            for v in value:
                if isinstance(v, torch.Tensor):
                    out.extend(v.detach().cpu().reshape(-1).tolist())
                else:
                    out.append(v)
            return out
        return [value]

    @staticmethod
    def _as_float_scalar(value: Union[torch.Tensor, float, int]) -> float:
        """Convert a scalar tensor or numeric value to a Python float."""
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(f"Expected scalar tensor, got shape {tuple(value.shape)}")
            return float(value.detach().item())
        return float(value)

    @classmethod
    def _is_finite_scalar(cls, value: Union[torch.Tensor, float, int]) -> bool:
        """Return whether a scalar tensor or numeric value is finite."""
        try:
            return math.isfinite(cls._as_float_scalar(value))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _compute_selector_grad_norms(
        model: torch.nn.Module,
    ) -> Dict[str, float]:
        """Compute gradient L2-norms for *active* selector parameters, per level and total.

        Only counts encoder blocks where ``gamma > 1`` (i.e. blocks that
        actually execute their selector during forward).  This matches the
        active-selector level indexing used by the existing selector
        diagnostics (``selector_level0`` = first block with gamma > 1).

        Accumulates squared norms on-device (lazy-init, no CPU tensor
        allocation) to avoid per-parameter host/device syncs.

        Returns a dict such as::

            {"selector_grad_norm_total": 1.23,
             "selector_grad_norm_level0": 0.81,
             "selector_grad_norm_level1": 0.42}

        Returns an empty dict when no active selector gradients are found
        (e.g. stride-based baselines or gamma=1 everywhere).
        """
        encoder = getattr(model, "encoder", None)
        if encoder is None:
            return {}
        blocks = getattr(encoder, "encoder_blocks", None)
        if blocks is None:
            return {}

        result: Dict[str, float] = {}
        total_sq = 0.0
        level_idx = 0
        found_any_grad = False
        for block in blocks:
            # Match active-selector semantics: only gamma > 1 blocks.
            gamma = getattr(block, "gamma", 1)
            if gamma <= 1:
                continue
            pool = getattr(block, "pool", None)
            if pool is None:
                continue
            selector = getattr(pool, "selector", None)
            if selector is None:
                continue

            # Accumulate on-device to avoid per-parameter .item() syncs.
            level_sq_t: Optional[torch.Tensor] = None
            for p in selector.parameters():
                if p.grad is not None:
                    g = p.grad.detach()
                    g_sq = g.norm(2) ** 2
                    level_sq_t = g_sq if level_sq_t is None else (level_sq_t + g_sq)

            if level_sq_t is None:
                level_sq = 0.0
            else:
                level_sq = float(level_sq_t.item())
                found_any_grad = True
            result[f"selector_grad_norm_level{level_idx}"] = level_sq ** 0.5
            total_sq += level_sq
            level_idx += 1

        if not found_any_grad:
            return {}
        if level_idx > 0:
            result["selector_grad_norm_total"] = total_sq ** 0.5
        return result

    @staticmethod
    def _compute_stratified_t_bin_stats(
        bin_loss_accumulator: Dict[str, List[float]],
    ) -> Dict[str, float]:
        """Compute mean and population std for each timestep bin.

        Args:
            bin_loss_accumulator: dict mapping bin label (e.g. ``"t_low_noise"``)
                to all per-graph reconstruction float values accumulated over the
                evaluation loop.

        Returns:
            Flat dict with keys ``"loss_<bin>_mean"`` and ``"loss_<bin>_std"``
            for each bin present in the accumulator.
        """
        result: Dict[str, float] = {}
        for label, values in bin_loss_accumulator.items():
            if not values:
                continue
            n = len(values)
            mean_val = sum(values) / n
            if n > 1:
                variance = sum((v - mean_val) ** 2 for v in values) / n  # population std (ddof=0)
                std_val = math.sqrt(variance)
            else:
                std_val = 0.0
            result[f"loss_{label}_mean"] = mean_val
            result[f"loss_{label}_std"] = std_val
        return result

    def _should_collect_heavy_selector_diagnostics(
        self, epoch: int, *, is_last_epoch: bool = False,
    ) -> bool:
        """
        Return whether heavy selector diagnostics should be collected this epoch.

        Global gate:
            - if `diagnose_model_enabled` is False, always returns False.

        `diagnose_selector_every_n_epochs` semantics:
            - 0: disabled
            - n>0: collect on eval epochs satisfying ``epoch % n == 0``
              (0-indexed, matching ``EvalSchedule.should_eval``). Epoch 0
              is skipped — no diagnostic history yet. When ``n`` divides
              the eval period (the common case, e.g. n=25 with eval period
              50/100), every natural eval epoch fires heavy collection.

        Heavy diagnostics are always collected on the last epoch so that the
        final evaluation has ``selector_network_node_stats`` available, even
        if the eval schedule / cadence would otherwise skip it.
        """
        if not self.diagnose_model_enabled:
            return False
        if self.diagnose_selector_every_n_epochs <= 0:
            return False
        if is_last_epoch:
            return True
        # Align with the eval schedule: only collect heavy diagnostics on
        # epochs that will actually run evaluation, so the collected stats
        # are consumed by the visualization code in the same epoch.
        if self.max_epochs is not None:
            if not self.eval_schedule.should_eval(epoch, self.max_epochs):
                return False
        # 0-indexed cadence matching ``EvalSchedule.should_eval`` semantics
        # (``epoch % period == 0``). Skip epoch 0 (no diagnostic history yet
        # to collect). The forced last-epoch path above (``is_last_epoch``)
        # guarantees a final-epoch collection even when ``max_epochs - 1``
        # isn't on the cadence grid.
        n = self.diagnose_selector_every_n_epochs
        return epoch > 0 and epoch % n == 0

    def _set_selector_heavy_diagnostics_enabled(self, enabled: bool) -> None:
        """
        Propagate selector-heavy-diagnostics mode to diffusion/model modules.

        Diagnostics are globally disabled when `diagnose_model_enabled` is False.
        """
        enabled = bool(enabled) and self.diagnose_model_enabled
        set_fn = getattr(self.diffusion, "set_collect_selector_heavy_diagnostics", None)
        if callable(set_fn):
            set_fn(enabled)

    @staticmethod
    def _empty_selector_epoch_diagnostics() -> _SelectorEpochDiagnostics:
        """Return an empty selector diagnostics bundle."""
        return _SelectorEpochDiagnostics(
            avg_selector_diags={},
            avg_selector_stats_by_level={},
            selector_network_stats={},
            selector_level_freqs={},
            selector_network_node_stats={},
            summary_path=None,
        )

    @staticmethod
    def _normalize_violation_metric_names(metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Canonicalize legacy WRA violation-rate metric keys to percentage keys.

        Legacy keys are interpreted as fractions in [0, 1] and converted to [%].
        Legacy keys are removed from the returned dict.
        """
        if not isinstance(metrics, dict):
            return {}

        normalized = dict(metrics)
        key_pairs = [
            ("power_violation_rate_generated", "power_violation_percentage_generated"),
            ("rate_violation_rate_generated", "rate_violation_percentage_generated"),
            ("rate_violation_rate_real", "rate_violation_percentage_real"),
        ]
        for old_key, new_key in key_pairs:
            if old_key in normalized:
                if new_key not in normalized:
                    old_val = normalized.get(old_key, None)
                    if isinstance(old_val, (int, float)):
                        normalized[new_key] = float(old_val) * 100.0
                normalized.pop(old_key, None)
        return normalized

    @staticmethod
    def _resolve_run_output_dir(save_dir: Optional[str]) -> Optional[str]:
        """Resolve best output directory for per-run artifacts."""
        if save_dir:
            return save_dir
        try:
            hydra_cfg = HydraConfig.get()
            runtime = getattr(hydra_cfg, "runtime", None)
            out_dir = getattr(runtime, "output_dir", None)
            if isinstance(out_dir, str) and out_dir:
                return out_dir
        except Exception:
            return None
        return None

    def _collect_model_architecture_stats(self) -> Dict[str, Any]:
        """
        Collect model architecture/size stats from the underlying model.

        If model exposes `get_architecture_stats()`, use it; otherwise fall back
        to generic parameter counting.
        """
        model = getattr(self.diffusion, "model", None)
        target = model if isinstance(model, torch.nn.Module) else self.diffusion

        if hasattr(target, "get_architecture_stats") and callable(getattr(target, "get_architecture_stats")):
            stats = target.get_architecture_stats()
            if not isinstance(stats, dict):
                stats = {"raw_stats_repr": str(stats)}
        else:
            params_total = int(sum(p.numel() for p in target.parameters()))
            params_trainable = int(sum(p.numel() for p in target.parameters() if p.requires_grad))
            params_frozen = int(params_total - params_trainable)
            size_total_bytes = int(
                sum(p.numel() * p.element_size() for p in target.parameters())
                + sum(b.numel() * b.element_size() for b in target.buffers())
            )
            stats = {
                "model_name": target.__class__.__name__,
                "params_total": params_total,
                "params_trainable": params_trainable,
                "params_frozen": params_frozen,
                "size_total_mb": float(size_total_bytes / (1024.0 ** 2)),
            }

        stats["diffusion_wrapper"] = self.diffusion.__class__.__name__
        return stats

    def _log_and_save_model_architecture_stats(self) -> Optional[str]:
        """Log and persist architecture stats once per training run."""
        def _to_jsonable(value):
            if value is None or isinstance(value, (str, int, float, bool)):
                return value
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    return float(value.detach().cpu().item())
                return value.detach().cpu().tolist()
            if isinstance(value, dict):
                return {str(k): _to_jsonable(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_to_jsonable(v) for v in value]
            try:
                return float(value)
            except Exception:
                return str(value)

        stats = _to_jsonable(self._collect_model_architecture_stats())
        log.info("Model Architecture Stats: %s", json.dumps(stats, sort_keys=True))

        out_dir = self._resolve_run_output_dir(self.save_dir)
        if not out_dir:
            return None

        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "model_architecture_stats.json")
        with open(path, "w") as f:
            json.dump(stats, f, indent=2, sort_keys=True)
        log.info("Saved model architecture stats to %s", path)
        return path

    @classmethod
    def _extract_graph_keys(cls, data: Data) -> List[str]:
        """
        Build stable per-graph identifiers from batch metadata when available.

        Priority:
            1) dataset_name + network_id (WRA)
            2) network_id
            3) fallback graph_{idx}
        """
        num_graphs = int(getattr(data, "num_graphs", 0) or 0)
        if num_graphs <= 0:
            return []

        network_ids = cls._to_python_list(getattr(data, "network_id", None))
        dataset_names = cls._to_python_list(getattr(data, "dataset_name", None))

        if len(network_ids) == 1 and num_graphs > 1:
            network_ids = network_ids * num_graphs
        if len(dataset_names) == 1 and num_graphs > 1:
            dataset_names = dataset_names * num_graphs

        keys = []
        for i in range(num_graphs):
            net = network_ids[i] if i < len(network_ids) else None
            dname = dataset_names[i] if i < len(dataset_names) else None
            if dname is not None and net is not None:
                keys.append(f"{dname}::network_{net}")
            elif net is not None:
                keys.append(f"network_{net}")
            else:
                keys.append(f"graph_{i}")
        return keys

    def _resolve_selector_diag_tracked_network_keys(
        self,
        batch_graph_keys: List[str],
    ) -> Optional[Set[str]]:
        """
        Resolve/capture the subset of network keys used for per-network diagnostics.

        Returns:
            - None when no cap is applied (track all networks).
            - A possibly-empty set of tracked keys when cap is active.
        """
        cap = int(self.selector_summary_max_network_rows)
        if cap < 0:
            return None
        if cap == 0:
            return set()

        for key in batch_graph_keys:
            if key in self._selector_diag_tracked_network_key_set:
                continue
            if len(self._selector_diag_tracked_network_key_order) >= cap:
                continue
            self._selector_diag_tracked_network_key_order.append(key)
            self._selector_diag_tracked_network_key_set.add(key)
        return set(self._selector_diag_tracked_network_key_set)

    @staticmethod
    def _init_selector_diag_state() -> Dict[str, Any]:
        """Create mutable state containers for selector diagnostics within one epoch."""
        return {
            "selector_diag_sums": {},
            "selector_diag_steps": 0,
            "selector_level_node_counts": {},
            "selector_level_graph_counts": {},
            "selector_level_stat_sums": defaultdict(dict),
            "selector_level_stat_counts": defaultdict(dict),
            # Per-level scalar stats that are always available from compact
            # selector diagnostics (independent of heavy tensor diagnostics).
            "selector_level_compact_stat_sums": defaultdict(dict),
            "selector_level_compact_stat_counts": defaultdict(dict),
            "selector_network_level_node_counts": defaultdict(dict),
            "selector_network_level_graph_counts": defaultdict(dict),
            "selector_network_level_score_sum": defaultdict(dict),
            "selector_network_level_score_sq_sum": defaultdict(dict),
            "selector_network_level_score_count": defaultdict(dict),
            "selector_network_level_soft_sum": defaultdict(dict),
            "selector_network_level_soft_sq_sum": defaultdict(dict),
            "selector_network_level_soft_count": defaultdict(dict),
        }

    @staticmethod
    def _accumulate_selector_diag(
        selector_diag_state: Dict[str, Any],
        selector_diag: Optional[dict],
    ) -> None:
        """Accumulate scalar selector diagnostics from one forward pass."""
        if not isinstance(selector_diag, dict) or len(selector_diag) == 0:
            return

        selector_diag_state["selector_diag_steps"] += 1
        selector_diag_sums: Dict[str, float] = selector_diag_state["selector_diag_sums"]
        for k, v in selector_diag.items():
            if isinstance(v, (int, float)):
                selector_diag_sums[k] = selector_diag_sums.get(k, 0.0) + float(v)

        # Compact per-level scalar stats (available even when heavy diagnostics
        # are disabled). Keep only metrics expected to differ by level.
        stats_by_level = selector_diag.get("stats_by_level", None)
        if isinstance(stats_by_level, dict):
            compact_sums: Dict[str, Dict[int, float]] = selector_diag_state[
                "selector_level_compact_stat_sums"
            ]
            compact_counts: Dict[str, Dict[int, float]] = selector_diag_state[
                "selector_level_compact_stat_counts"
            ]
            for stat_name, per_level in stats_by_level.items():
                if stat_name == "temperature":
                    # Temperature is a global schedule parameter.
                    continue
                if not isinstance(per_level, dict):
                    continue
                for level_idx_raw, stat_val in per_level.items():
                    if not isinstance(stat_val, (int, float)):
                        continue
                    try:
                        level_idx = int(level_idx_raw)
                    except (TypeError, ValueError):
                        continue
                    stat_val = float(stat_val)
                    if not math.isfinite(stat_val):
                        continue
                    prev_sum = compact_sums[stat_name].get(level_idx, 0.0)
                    prev_cnt = compact_counts[stat_name].get(level_idx, 0.0)
                    compact_sums[stat_name][level_idx] = prev_sum + stat_val
                    compact_counts[stat_name][level_idx] = prev_cnt + 1.0

    @staticmethod
    def _accumulate_selector_level_diags(
        selector_diag_state: Dict[str, Any],
        selector_level_diags: Optional[list],
        batch_graph_keys: List[str],
        tracked_network_keys: Optional[Set[str]] = None,
    ) -> None:
        """Accumulate per-level/per-network selector diagnostics from one forward pass."""
        if not isinstance(selector_level_diags, list):
            return

        selector_level_node_counts: Dict[int, torch.Tensor] = selector_diag_state["selector_level_node_counts"]
        selector_level_graph_counts: Dict[int, float] = selector_diag_state["selector_level_graph_counts"]
        selector_level_stat_sums: Dict[str, Dict[int, float]] = selector_diag_state["selector_level_stat_sums"]
        selector_level_stat_counts: Dict[str, Dict[int, float]] = selector_diag_state["selector_level_stat_counts"]
        selector_level_compact_stat_sums: Dict[str, Dict[int, float]] = selector_diag_state[
            "selector_level_compact_stat_sums"
        ]
        selector_level_compact_stat_counts: Dict[str, Dict[int, float]] = selector_diag_state[
            "selector_level_compact_stat_counts"
        ]
        selector_network_level_node_counts: Dict[str, Dict[int, torch.Tensor]] = selector_diag_state[
            "selector_network_level_node_counts"
        ]
        selector_network_level_graph_counts: Dict[str, Dict[int, float]] = selector_diag_state[
            "selector_network_level_graph_counts"
        ]
        selector_network_level_score_sum: Dict[str, Dict[int, torch.Tensor]] = selector_diag_state[
            "selector_network_level_score_sum"
        ]
        selector_network_level_score_sq_sum: Dict[str, Dict[int, torch.Tensor]] = selector_diag_state[
            "selector_network_level_score_sq_sum"
        ]
        selector_network_level_score_count: Dict[str, Dict[int, torch.Tensor]] = selector_diag_state[
            "selector_network_level_score_count"
        ]
        selector_network_level_soft_sum: Dict[str, Dict[int, torch.Tensor]] = selector_diag_state[
            "selector_network_level_soft_sum"
        ]
        selector_network_level_soft_sq_sum: Dict[str, Dict[int, torch.Tensor]] = selector_diag_state[
            "selector_network_level_soft_sq_sum"
        ]
        selector_network_level_soft_count: Dict[str, Dict[int, torch.Tensor]] = selector_diag_state[
            "selector_network_level_soft_count"
        ]

        for level_diag in selector_level_diags:
            if not isinstance(level_diag, dict):
                continue
            level_idx = int(level_diag.get("selector_level", -1))
            node_counts = level_diag.get("selected_node_counts", None)
            node_counts_per_graph = level_diag.get("selected_node_counts_per_graph", None)
            scores_per_graph = level_diag.get("scores_per_graph", None)
            soft_mask_per_graph = level_diag.get("soft_mask_per_graph", None)
            active_mask_per_graph = level_diag.get("active_mask_per_graph", None)
            num_graphs = float(level_diag.get("num_graphs", 0.0))
            if level_idx < 0 or not isinstance(node_counts, torch.Tensor):
                continue

            node_counts = node_counts.detach().float().cpu()
            prev = selector_level_node_counts.get(level_idx)
            if prev is None:
                selector_level_node_counts[level_idx] = node_counts.clone()
            elif prev.shape == node_counts.shape:
                selector_level_node_counts[level_idx] = prev + node_counts
            else:
                log.warning(
                    "Selector level %d node count shape changed within epoch (%s -> %s); skipping this batch.",
                    level_idx, tuple(prev.shape), tuple(node_counts.shape)
                )
                continue
            selector_level_graph_counts[level_idx] = selector_level_graph_counts.get(level_idx, 0.0) + max(num_graphs, 0.0)

            # Per-level scalar diagnostics (organized as stat -> level -> value).
            for stat_name, stat_val in level_diag.items():
                if stat_name in {
                    "selector_level",
                    "selected_node_counts",
                    "selected_node_counts_per_graph",
                    # Temperature is global/shared; do not duplicate as per-level metric.
                    "temperature",
                }:
                    continue
                if not isinstance(stat_val, (float, int)):
                    continue
                prev_sum = selector_level_stat_sums[stat_name].get(level_idx, 0.0)
                prev_cnt = selector_level_stat_counts[stat_name].get(level_idx, 0.0)
                selector_level_stat_sums[stat_name][level_idx] = prev_sum + float(stat_val)
                selector_level_stat_counts[stat_name][level_idx] = prev_cnt + 1.0

            # Per-network/node selection counts (for split diagnostics by unique graph/network).
            if (
                isinstance(node_counts_per_graph, torch.Tensor)
                and node_counts_per_graph.dim() == 2
                and len(batch_graph_keys) == int(node_counts_per_graph.size(0))
            ):
                node_counts_pg = node_counts_per_graph.detach().float().cpu()
                score_pg = (
                    scores_per_graph.detach().float().cpu()
                    if isinstance(scores_per_graph, torch.Tensor) and scores_per_graph.dim() == 2
                    else None
                )
                soft_pg = (
                    soft_mask_per_graph.detach().float().cpu()
                    if isinstance(soft_mask_per_graph, torch.Tensor) and soft_mask_per_graph.dim() == 2
                    else None
                )
                active_pg = (
                    active_mask_per_graph.detach().bool().cpu()
                    if isinstance(active_mask_per_graph, torch.Tensor) and active_mask_per_graph.dim() == 2
                    else None
                )
                for graph_row, graph_key in enumerate(batch_graph_keys):
                    if tracked_network_keys is not None and graph_key not in tracked_network_keys:
                        continue
                    graph_counts_for_levels = selector_network_level_node_counts[graph_key]
                    prev_counts = graph_counts_for_levels.get(level_idx)
                    row_counts = node_counts_pg[graph_row]
                    if prev_counts is None:
                        graph_counts_for_levels[level_idx] = row_counts.clone()
                    elif prev_counts.shape == row_counts.shape:
                        graph_counts_for_levels[level_idx] = prev_counts + row_counts
                    else:
                        log.warning(
                            "Selector level %d node count shape changed for %s (%s -> %s); skipping this sample.",
                            level_idx,
                            graph_key,
                            tuple(prev_counts.shape),
                            tuple(row_counts.shape),
                        )
                        continue
                    prev_graph_cnt = selector_network_level_graph_counts[graph_key].get(level_idx, 0.0)
                    selector_network_level_graph_counts[graph_key][level_idx] = prev_graph_cnt + 1.0

                    # Per-node soft-mask moments (always defined for all nodes).
                    if soft_pg is not None and soft_pg.shape[1] == row_counts.shape[0]:
                        row_soft = soft_pg[graph_row]
                        row_soft_sq = row_soft * row_soft
                        soft_sum_levels = selector_network_level_soft_sum[graph_key]
                        soft_sq_levels = selector_network_level_soft_sq_sum[graph_key]
                        soft_cnt_levels = selector_network_level_soft_count[graph_key]
                        if level_idx not in soft_sum_levels:
                            soft_sum_levels[level_idx] = row_soft.clone()
                            soft_sq_levels[level_idx] = row_soft_sq.clone()
                            soft_cnt_levels[level_idx] = torch.ones_like(row_soft)
                        else:
                            soft_sum_levels[level_idx] += row_soft
                            soft_sq_levels[level_idx] += row_soft_sq
                            soft_cnt_levels[level_idx] += torch.ones_like(row_soft)

                    # Per-node score moments (active nodes only).
                    if (
                        score_pg is not None
                        and active_pg is not None
                        and score_pg.shape[1] == row_counts.shape[0]
                        and active_pg.shape[1] == row_counts.shape[0]
                    ):
                        row_scores = score_pg[graph_row]
                        row_active = active_pg[graph_row].float()
                        score_sum_row = row_scores * row_active
                        score_sq_sum_row = (row_scores * row_scores) * row_active
                        score_sum_levels = selector_network_level_score_sum[graph_key]
                        score_sq_levels = selector_network_level_score_sq_sum[graph_key]
                        score_cnt_levels = selector_network_level_score_count[graph_key]
                        if level_idx not in score_sum_levels:
                            score_sum_levels[level_idx] = score_sum_row.clone()
                            score_sq_levels[level_idx] = score_sq_sum_row.clone()
                            score_cnt_levels[level_idx] = row_active.clone()
                        else:
                            score_sum_levels[level_idx] += score_sum_row
                            score_sq_levels[level_idx] += score_sq_sum_row
                            score_cnt_levels[level_idx] += row_active

    @staticmethod
    def _finalize_selector_diag_state(
        selector_diag_state: Dict[str, Any]
    ) -> Tuple[
        Dict[str, float],
        Dict[str, Dict[str, float]],
        Dict[str, Dict[str, dict]],
        Dict[int, torch.Tensor],
        Dict[str, Dict[int, Dict[str, torch.Tensor]]],
    ]:
        """Finalize epoch-level selector diagnostics from accumulated state."""
        selector_diag_sums: Dict[str, float] = selector_diag_state["selector_diag_sums"]
        selector_diag_steps: int = selector_diag_state["selector_diag_steps"]
        selector_level_node_counts: Dict[int, torch.Tensor] = selector_diag_state["selector_level_node_counts"]
        selector_level_graph_counts: Dict[int, float] = selector_diag_state["selector_level_graph_counts"]
        selector_level_stat_sums: Dict[str, Dict[int, float]] = selector_diag_state["selector_level_stat_sums"]
        selector_level_stat_counts: Dict[str, Dict[int, float]] = selector_diag_state["selector_level_stat_counts"]
        selector_level_compact_stat_sums: Dict[str, Dict[int, float]] = selector_diag_state[
            "selector_level_compact_stat_sums"
        ]
        selector_level_compact_stat_counts: Dict[str, Dict[int, float]] = selector_diag_state[
            "selector_level_compact_stat_counts"
        ]
        selector_network_level_node_counts: Dict[str, Dict[int, torch.Tensor]] = selector_diag_state[
            "selector_network_level_node_counts"
        ]
        selector_network_level_graph_counts: Dict[str, Dict[int, float]] = selector_diag_state[
            "selector_network_level_graph_counts"
        ]
        selector_network_level_score_sum: Dict[str, Dict[int, torch.Tensor]] = selector_diag_state[
            "selector_network_level_score_sum"
        ]
        selector_network_level_score_sq_sum: Dict[str, Dict[int, torch.Tensor]] = selector_diag_state[
            "selector_network_level_score_sq_sum"
        ]
        selector_network_level_score_count: Dict[str, Dict[int, torch.Tensor]] = selector_diag_state[
            "selector_network_level_score_count"
        ]
        selector_network_level_soft_sum: Dict[str, Dict[int, torch.Tensor]] = selector_diag_state[
            "selector_network_level_soft_sum"
        ]
        selector_network_level_soft_sq_sum: Dict[str, Dict[int, torch.Tensor]] = selector_diag_state[
            "selector_network_level_soft_sq_sum"
        ]
        selector_network_level_soft_count: Dict[str, Dict[int, torch.Tensor]] = selector_diag_state[
            "selector_network_level_soft_count"
        ]

        avg_selector_diags = {
            k: (v / max(selector_diag_steps, 1))
            for k, v in selector_diag_sums.items()
        } if selector_diag_steps > 0 else {}

        avg_selector_stats_by_level = {}
        for stat_name, level_sums in selector_level_stat_sums.items():
            per_level = {}
            for level_idx, sum_val in level_sums.items():
                denom = max(selector_level_stat_counts.get(stat_name, {}).get(level_idx, 0.0), 1.0)
                per_level[str(level_idx)] = float(sum_val / denom)
            if per_level:
                avg_selector_stats_by_level[stat_name] = per_level

        # Fallback from compact per-level diagnostics so scalar-by-level stats
        # are still available when heavy diagnostics are disabled.
        for stat_name, level_sums in selector_level_compact_stat_sums.items():
            if stat_name in avg_selector_stats_by_level and avg_selector_stats_by_level[stat_name]:
                continue
            per_level = {}
            for level_idx, sum_val in level_sums.items():
                denom = max(selector_level_compact_stat_counts.get(stat_name, {}).get(level_idx, 0.0), 1.0)
                per_level[str(level_idx)] = float(sum_val / denom)
            if per_level:
                avg_selector_stats_by_level[stat_name] = per_level

        selector_network_stats = {}
        for graph_key, level_counts in selector_network_level_node_counts.items():
            graph_level_stats = {}
            for level_idx, counts in level_counts.items():
                denom = max(
                    selector_network_level_graph_counts.get(graph_key, {}).get(level_idx, 0.0),
                    1.0,
                )
                freqs = (counts / denom).clamp(min=0.0, max=1.0)
                graph_level_stats[str(level_idx)] = {
                    "freq_mean": float(freqs.mean().item()),
                    "freq_std": float(freqs.std(unbiased=False).item()),
                    "num_graphs": float(denom),
                    "num_nodes": int(freqs.numel()),
                }
            if graph_level_stats:
                selector_network_stats[graph_key] = graph_level_stats

        selector_level_freqs = {}
        for level_idx, counts in selector_level_node_counts.items():
            denom = max(selector_level_graph_counts.get(level_idx, 0.0), 1.0)
            selector_level_freqs[level_idx] = (counts / denom).clamp(min=0.0, max=1.0)

        selector_network_node_stats: Dict[str, Dict[int, Dict[str, torch.Tensor]]] = {}
        network_keys = (
            set(selector_network_level_node_counts.keys())
            | set(selector_network_level_soft_sum.keys())
            | set(selector_network_level_score_sum.keys())
        )
        for graph_key in network_keys:
            levels: Dict[int, Dict[str, torch.Tensor]] = {}
            level_keys = (
                set(selector_network_level_node_counts.get(graph_key, {}).keys())
                | set(selector_network_level_soft_sum.get(graph_key, {}).keys())
                | set(selector_network_level_score_sum.get(graph_key, {}).keys())
            )
            for level_idx in level_keys:
                level_stats: Dict[str, torch.Tensor] = {}

                hard_counts = selector_network_level_node_counts.get(graph_key, {}).get(level_idx, None)
                if hard_counts is not None:
                    denom = max(
                        selector_network_level_graph_counts.get(graph_key, {}).get(level_idx, 0.0),
                        1.0,
                    )
                    hard_mean = (hard_counts / denom).clamp(min=0.0, max=1.0)
                    hard_std = torch.sqrt(torch.clamp(hard_mean * (1.0 - hard_mean), min=0.0))
                    level_stats["hard_mask_mean"] = hard_mean
                    level_stats["hard_mask_std"] = hard_std

                soft_sum = selector_network_level_soft_sum.get(graph_key, {}).get(level_idx, None)
                soft_sq_sum = selector_network_level_soft_sq_sum.get(graph_key, {}).get(level_idx, None)
                soft_count = selector_network_level_soft_count.get(graph_key, {}).get(level_idx, None)
                if soft_sum is not None and soft_sq_sum is not None and soft_count is not None:
                    safe_soft_count = soft_count.clamp(min=1.0)
                    soft_mean = soft_sum / safe_soft_count
                    soft_var = (soft_sq_sum / safe_soft_count) - (soft_mean * soft_mean)
                    soft_std = torch.sqrt(torch.clamp(soft_var, min=0.0))
                    level_stats["soft_mask_mean"] = soft_mean
                    level_stats["soft_mask_std"] = soft_std
                    level_stats["soft_mask_count"] = soft_count

                score_sum = selector_network_level_score_sum.get(graph_key, {}).get(level_idx, None)
                score_sq_sum = selector_network_level_score_sq_sum.get(graph_key, {}).get(level_idx, None)
                score_count = selector_network_level_score_count.get(graph_key, {}).get(level_idx, None)
                if score_sum is not None and score_sq_sum is not None and score_count is not None:
                    safe_score_count = score_count.clamp(min=1.0)
                    score_mean_raw = score_sum / safe_score_count
                    score_var_raw = (score_sq_sum / safe_score_count) - (score_mean_raw * score_mean_raw)
                    score_std_raw = torch.sqrt(torch.clamp(score_var_raw, min=0.0))
                    nan_template = torch.full_like(score_mean_raw, float("nan"))
                    has_obs = score_count > 0
                    score_mean = torch.where(has_obs, score_mean_raw, nan_template)
                    score_std = torch.where(has_obs, score_std_raw, nan_template)
                    level_stats["score_mean"] = score_mean
                    level_stats["score_std"] = score_std
                    level_stats["score_count"] = score_count

                if level_stats:
                    levels[level_idx] = level_stats
            if levels:
                selector_network_node_stats[graph_key] = levels

        return (
            avg_selector_diags,
            avg_selector_stats_by_level,
            selector_network_stats,
            selector_level_freqs,
            selector_network_node_stats,
        )

    @staticmethod
    def _parse_graph_key(graph_key: str) -> Tuple[Optional[str], str]:
        """Parse trainer graph key into (dataset_name, network_tag)."""
        if "::" in graph_key:
            dataset_name, network_tag = graph_key.split("::", 1)
            return dataset_name, network_tag
        return None, graph_key

    @classmethod
    def _resolve_sampling_diag_identity(cls, data: Optional[Data]) -> Tuple[Optional[str], Optional[str]]:
        """
        Resolve a compact (dataset_name, network_id) tag for sampling diagnostics.

        If a batch mixes multiple network keys, returns network_id='mixed_batch'.
        """
        if data is None:
            return None, None
        graph_keys = cls._extract_graph_keys(data)
        if len(graph_keys) == 0:
            return None, None
        unique_keys = sorted(set(graph_keys))
        if len(unique_keys) == 1:
            dataset_name, network_tag = cls._parse_graph_key(unique_keys[0])
            return dataset_name, network_tag
        return None, "mixed_batch"

    @staticmethod
    def _should_plot_on_epoch(
        *,
        epoch: int,
        plot_every_n_epochs: Optional[int],
    ) -> bool:
        """Return whether a periodic plot is due on this epoch.

        Uses 0-indexed ``epoch % period == 0`` semantics to match
        ``EvalSchedule.should_eval``. ``_plot_epoch_artifacts`` is only
        invoked from the eval branch, so plot epochs are
        ``eval_epochs ∩ {n : n % period == 0}``. Skips epoch 0 (no JSONL
        history yet to plot).
        """
        if plot_every_n_epochs is None:
            return False
        period = int(plot_every_n_epochs)
        if period <= 0:
            return False
        epoch_int = int(epoch)
        return epoch_int > 0 and epoch_int % period == 0

    @staticmethod
    def _cap_network_level_stats(
        selector_network_node_stats: Dict[str, Dict[int, Dict[str, torch.Tensor]]],
        max_network_plots: Optional[int],
    ) -> Dict[str, Dict[int, Dict[str, torch.Tensor]]]:
        """Return selector network stats capped to a deterministic key subset."""
        if not selector_network_node_stats:
            return {}
        if max_network_plots is None:
            return selector_network_node_stats
        cap = int(max_network_plots)
        if cap < 0:
            return selector_network_node_stats
        if cap == 0:
            return {}
        network_keys = sorted(selector_network_node_stats.keys())
        selected = network_keys[:cap]
        return {k: selector_network_node_stats[k] for k in selected}

    def _plot_selector_sampling_diagnostics(
        self,
        selector_sampling_diagnostics: Optional[Dict[str, Any]],
        *,
        data: Optional[Data],
        viz_save_dir: Optional[str],
        eval_split: str,
        batch_idx: Optional[int] = None,
        aggregate: bool = False,
        ignore_network_cap: bool = False,
    ) -> None:
        """Render selector sampling-phase heatmaps for one evaluation call.

        ``batch_idx`` (when provided) is woven into each output filename so that
        successive eval batches do not overwrite one another — required for
        datasets like SP500 where every window shares ``network_id=0`` and would
        otherwise collide on a single per-network file.

        ``aggregate=True`` marks the figures as pooled-over-all-batches (filename
        ``_aggregate`` token + title note) instead of per-batch.

        ``ignore_network_cap=True`` plots every network regardless of the
        ``max_network_plots`` config (used by the aggregate path, which is a
        single end-of-eval figure that should always be complete).
        """
        if not self.diagnose_model_enabled:
            return
        if not bool(self.plot_sampling_heatmap_cfg.get("enabled", True)):
            return
        heatmap_every_n = self.plot_sampling_heatmap_cfg.get("plot_every_n_epochs", None)
        if heatmap_every_n is not None:
            heatmap_every_n = int(heatmap_every_n)
            # Treat 0 as "plot on every eval event" for convenience.
            if heatmap_every_n > 0 and not self._should_plot_on_epoch(
                epoch=self.epoch,
                plot_every_n_epochs=heatmap_every_n,
            ):
                return
        if not isinstance(selector_sampling_diagnostics, dict) or len(selector_sampling_diagnostics) == 0:
            return
        if not viz_save_dir:
            return
        try:
            from graph_signal_diffusion.utils.plotting import plot_selector_sampling_phase_heatmaps
        except Exception as e:
            log.warning("Failed to import selector sampling diagnostics plotting utility: %s", e)
            return

        selector_plot_dir = os.path.join(viz_save_dir, "selector_diagnostics")
        os.makedirs(selector_plot_dir, exist_ok=True)

        # Persist the pooled aggregate diagnostics (per-node selection
        # frequencies) as JSON for offline analysis — e.g. the leverage-score
        # comparison. Runs only for the aggregate, after all plotting gates pass.
        if aggregate:
            self._dump_aggregate_selector_sampling_json(
                selector_sampling_diagnostics, selector_plot_dir, eval_split
            )

        by_network = selector_sampling_diagnostics.get("by_network", None)
        if isinstance(by_network, dict) and len(by_network) > 0:
            network_keys = sorted(by_network.keys())
            sampling_cap = self.plot_sampling_heatmap_cfg.get("max_network_plots", None)
            if ignore_network_cap or sampling_cap is None:
                max_networks = len(network_keys)
            else:
                cap = int(sampling_cap)
                if cap == 0:
                    return
                if cap < 0:
                    max_networks = len(network_keys)
                else:
                    max_networks = min(cap, len(network_keys))
            for graph_key in network_keys[:max_networks]:
                network_diag = by_network.get(graph_key, None)
                if not isinstance(network_diag, dict):
                    continue
                dataset_name = network_diag.get("dataset_name", None)
                network_id = network_diag.get("network_id", None)
                if network_id is None:
                    _, network_id = self._parse_graph_key(str(graph_key))
                try:
                    plot_selector_sampling_phase_heatmaps(
                        selector_sampling_diagnostics=network_diag,
                        out_dir=selector_plot_dir,
                        epoch=self.epoch,
                        split=eval_split,
                        dataset_name=dataset_name,
                        network_id=network_id,
                        save_pdf=True,
                        batch_idx=batch_idx,
                        aggregate=aggregate,
                    )
                except Exception as e:
                    log.warning(
                        "Failed to plot selector sampling diagnostics (%s, %s): %s",
                        eval_split,
                        graph_key,
                        e,
                    )
            if len(network_keys) > max_networks:
                log.info(
                    "Selector sampling diagnostics: plotted %d/%d networks (cap=%d) for split %s",
                    max_networks,
                    len(network_keys),
                    (
                        int(sampling_cap)
                        if sampling_cap is not None
                        else len(network_keys)
                    ),
                    eval_split,
                )
            return

        dataset_name, network_id = self._resolve_sampling_diag_identity(data)
        try:
            plot_selector_sampling_phase_heatmaps(
                selector_sampling_diagnostics=selector_sampling_diagnostics,
                out_dir=selector_plot_dir,
                epoch=self.epoch,
                split=eval_split,
                dataset_name=dataset_name,
                network_id=network_id,
                save_pdf=True,
                batch_idx=batch_idx,
                aggregate=aggregate,
            )
            log.info(
                "Saved selector sampling diagnostics heatmap (%s) to %s",
                eval_split,
                selector_plot_dir,
            )
        except Exception as e:
            log.warning("Failed to plot selector sampling diagnostics (%s): %s", eval_split, e)

    def _plot_aggregate_selector_sampling_diagnostics(
        self,
        per_batch_diags: List[Dict[str, Any]],
        *,
        viz_save_dir: Optional[str],
        eval_split: str,
    ) -> None:
        """Pool per-batch selector diagnostics and plot one aggregate heatmap.

        Counts (selected/available/graph_count) are summed across all eval
        batches and the frequencies recomputed, so the representative-node set
        and the plotted selection probabilities are determined over the entire
        evaluated set rather than a single batch. Emitted alongside (not instead
        of) the per-batch heatmaps.
        """
        if not per_batch_diags:
            return
        try:
            from graph_signal_diffusion.utils.plotting import (
                aggregate_selector_sampling_diagnostics,
            )
        except Exception as e:
            log.warning("Failed to import selector sampling aggregation utility: %s", e)
            return
        aggregated = aggregate_selector_sampling_diagnostics(per_batch_diags)
        if not isinstance(aggregated, dict) or len(aggregated) == 0:
            return
        # Reuse the per-network dispatch + cadence/enabled gating; data=None is
        # fine because the aggregated dict carries its own by_network metadata.
        self._plot_selector_sampling_diagnostics(
            selector_sampling_diagnostics=aggregated,
            data=None,
            viz_save_dir=viz_save_dir,
            eval_split=eval_split,
            aggregate=True,
            ignore_network_cap=True,
        )

    def _dump_aggregate_selector_sampling_json(
        self,
        aggregated: Dict[str, Any],
        selector_plot_dir: str,
        eval_split: str,
    ) -> None:
        """Write the pooled aggregate diagnostics to JSON for offline analysis.

        The file carries the per-node ``selection_freq_by_level`` /
        ``selection_given_available_by_level`` arrays (top-level + per-network),
        which the standalone leverage-score comparison reads without re-sampling.
        ``NaN`` (never-available) cells are preserved via the default
        ``allow_nan`` and round-trip through ``json.load``.
        """
        try:
            import json as _json

            split_tag = (
                "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(eval_split)).strip("_")
                or "eval"
            )
            json_path = os.path.join(
                selector_plot_dir,
                f"selector_sampling_aggregate_{split_tag}_epoch_{self.epoch + 1:04d}.json",
            )
            with open(json_path, "w") as f:
                _json.dump(aggregated, f)
            log.info("Saved aggregate selector sampling diagnostics JSON to %s", json_path)
        except Exception as e:
            log.warning("Failed to dump aggregate selector sampling diagnostics JSON: %s", e)

    def _sample_with_selector_sampling_diagnostics(
        self,
        *,
        shape: Tuple[int, int, int, int],
        data: Optional[Data],
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """
        Sample from diffusion and request selector sampling diagnostics when supported.

        Sampling-phase selector diagnostics are requested only when
        `diagnose_model_enabled` is True.

        Falls back to the legacy sample API for older diffusion wrappers/tests.
        """
        want_selector_sampling_diagnostics = bool(self.diagnose_model_enabled)
        try:
            sample_output = self.diffusion.sample(
                shape=shape,
                device=self.device,
                data=data,
                use_amp=self._amp_enabled,
                return_selector_sampling_diagnostics=want_selector_sampling_diagnostics,
            )
        except TypeError:
            sample_output = self.diffusion.sample(
                shape=shape,
                device=self.device,
                data=data,
                use_amp=self._amp_enabled,
            )

        if not want_selector_sampling_diagnostics:
            if isinstance(sample_output, tuple) and len(sample_output) >= 1:
                return sample_output[0], None
            return sample_output, None

        if isinstance(sample_output, tuple) and len(sample_output) == 2:
            return sample_output[0], sample_output[1]
        return sample_output, None

    @staticmethod
    def _update_selection_frequency_moving_average(
        selector_freq_ma_state: Dict[str, Dict[int, torch.Tensor]],
        selector_network_node_stats: Dict[str, Dict[int, Dict[str, torch.Tensor]]],
        ema_alpha: float,
    ) -> None:
        """
        Update per-network/per-level per-node EMA of selection frequency.

        EMA update:
            ma_t = (1 - alpha) * ma_{t-1} + alpha * x_t
        where x_t is the current epoch's hard-mask frequency per node.
        """
        one_minus_alpha = 1.0 - float(ema_alpha)
        for graph_key, level_stats in selector_network_node_stats.items():
            ma_levels = selector_freq_ma_state.setdefault(graph_key, {})
            for level_idx, stats in level_stats.items():
                current = stats.get("hard_mask_mean", None)
                if not isinstance(current, torch.Tensor):
                    continue
                current = current.detach().float().cpu().clamp(min=0.0, max=1.0)
                prev = ma_levels.get(level_idx, None)
                if prev is None or not isinstance(prev, torch.Tensor) or prev.shape != current.shape:
                    ma_levels[level_idx] = current.clone()
                else:
                    ma_levels[level_idx] = one_minus_alpha * prev + float(ema_alpha) * current

    @staticmethod
    def _update_selection_frequency_tracked_nodes(
        selector_freq_tracked_nodes: Dict[str, Dict[int, List[int]]],
        selector_freq_ma_state: Dict[str, Dict[int, torch.Tensor]],
        num_nodes_to_track: int,
        rng: random.Random,
    ) -> None:
        """Pick and persist one random subset of node indices per network/level."""
        for graph_key, level_ma in selector_freq_ma_state.items():
            tracked_levels = selector_freq_tracked_nodes.setdefault(graph_key, {})
            for level_idx, ma_vec in level_ma.items():
                if not isinstance(ma_vec, torch.Tensor) or ma_vec.numel() <= 0:
                    continue
                num_nodes = int(ma_vec.numel())
                target_k = min(int(num_nodes_to_track), num_nodes)
                prev_indices = tracked_levels.get(level_idx, None)
                if (
                    isinstance(prev_indices, list)
                    and len(prev_indices) == target_k
                    and all(isinstance(i, int) and 0 <= i < num_nodes for i in prev_indices)
                ):
                    continue
                if target_k == num_nodes:
                    tracked_levels[level_idx] = list(range(num_nodes))
                else:
                    tracked_levels[level_idx] = sorted(rng.sample(range(num_nodes), target_k))


    def _append_selector_diagnostic_summaries_jsonl(
        self,
        epoch: int,
        avg_selector_diags: Dict[str, float],
        avg_selector_stats_by_level: Dict[str, Dict[str, float]],
        selector_level_freqs: Dict[int, torch.Tensor],
        selector_network_stats: Dict[str, Dict[str, dict]],
        selector_freq_ma_state: Optional[Dict[str, Dict[int, torch.Tensor]]] = None,
        selector_freq_tracked_nodes: Optional[Dict[str, Dict[int, List[int]]]] = None,
        write_network_rows: bool = True,
    ) -> Optional[str]:
        """
        Append selector diagnostic summaries for history-based plotting.

        Writes to:
            <save_dir>/selector_diagnostics/selector_diagnostic_summaries.jsonl
        with rows scoped as:
            - global: overall selector diagnostics per epoch
            - network: per-network/per-level selection-frequency summaries
        """
        if not self.save_dir:
            return None

        selector_plot_dir = os.path.join(self.save_dir, "selector_diagnostics")
        os.makedirs(selector_plot_dir, exist_ok=True)
        summary_path = os.path.join(selector_plot_dir, "selector_diagnostic_summaries.jsonl")

        global_row = {
            "scope": "global",
            "epoch": int(epoch),
            # Unprefixed copy of all selector scalar summaries that were previously
            # mirrored under train_selector_* in epoch_summaries.jsonl.
            **{
                str(k): float(v) if isinstance(v, (int, float)) else float("nan")
                for k, v in avg_selector_diags.items()
            },
        }
        global_row["stats_by_level"] = avg_selector_stats_by_level
        global_row["selection_freq_mean_by_level"] = {
            str(level_idx): float(freqs.mean().item())
            for level_idx, freqs in selector_level_freqs.items()
        }
        global_row["selection_freq_std_by_level"] = {
            str(level_idx): float(freqs.std(unbiased=False).item())
            for level_idx, freqs in selector_level_freqs.items()
        }

        selector_freq_ma_state = selector_freq_ma_state or {}
        selector_freq_tracked_nodes = selector_freq_tracked_nodes or {}

        with open(summary_path, "a") as f:
            f.write(json.dumps(global_row) + "\n")

            if not write_network_rows:
                return summary_path

            network_key_set = set(selector_network_stats.keys()) | set(selector_freq_ma_state.keys())
            if self._selector_diag_tracked_network_key_order:
                network_keys = [
                    key for key in self._selector_diag_tracked_network_key_order if key in network_key_set
                ]
            else:
                network_keys = sorted(network_key_set)

            if self.selector_summary_max_network_rows >= 0:
                network_keys = network_keys[: self.selector_summary_max_network_rows]

            for graph_key in network_keys:
                level_stats_raw = selector_network_stats.get(graph_key, {})
                level_ma_raw = selector_freq_ma_state.get(graph_key, {})
                tracked_levels_raw = selector_freq_tracked_nodes.get(graph_key, {})

                def _coerce_level_keys(level_dict: Dict[Any, Any]) -> Dict[int, Any]:
                    out: Dict[int, Any] = {}
                    for k, v in level_dict.items():
                        try:
                            level_int = int(k)
                        except (TypeError, ValueError):
                            continue
                        out[level_int] = v
                    return out

                level_stats = _coerce_level_keys(level_stats_raw)
                level_ma = _coerce_level_keys(level_ma_raw)
                tracked_levels = _coerce_level_keys(tracked_levels_raw)

                level_ids = sorted(set(level_stats.keys()) | set(level_ma.keys()))

                selection_freq_mean_by_level: Dict[str, float] = {}
                selection_freq_std_by_level: Dict[str, float] = {}
                num_nodes_by_level: Dict[str, int] = {}
                num_graphs_by_level: Dict[str, float] = {}
                selection_freq_tracked_node_indices_by_level: Dict[str, List[int]] = {}
                selection_freq_tracked_node_ma_by_level: Dict[str, List[float]] = {}

                for level_idx in level_ids:
                    level_key = str(level_idx)
                    ma_vec = level_ma.get(level_idx, None)
                    if isinstance(ma_vec, torch.Tensor):
                        ma_vec = ma_vec.detach().float().cpu().clamp(min=0.0, max=1.0)
                        selection_freq_mean_by_level[level_key] = float(ma_vec.mean().item())
                        selection_freq_std_by_level[level_key] = float(ma_vec.std(unbiased=False).item())
                        num_nodes_by_level[level_key] = int(ma_vec.numel())
                    else:
                        level_data = level_stats.get(level_idx, {})
                        selection_freq_mean_by_level[level_key] = float(level_data.get("freq_mean", float("nan")))
                        selection_freq_std_by_level[level_key] = float(level_data.get("freq_std", float("nan")))
                        num_nodes_by_level[level_key] = int(level_data.get("num_nodes", 0))

                    level_data = level_stats.get(level_idx, {})
                    num_graphs_by_level[level_key] = float(level_data.get("num_graphs", float("nan")))

                    tracked_idx = tracked_levels.get(level_idx, None)
                    if isinstance(tracked_idx, list) and len(tracked_idx) > 0:
                        tracked_idx_valid = [int(i) for i in tracked_idx]
                        selection_freq_tracked_node_indices_by_level[level_key] = tracked_idx_valid
                        if isinstance(ma_vec, torch.Tensor):
                            tracked_ma = []
                            for idx in tracked_idx_valid:
                                if 0 <= idx < int(ma_vec.numel()):
                                    tracked_ma.append(float(ma_vec[idx].item()))
                                else:
                                    tracked_ma.append(float("nan"))
                            selection_freq_tracked_node_ma_by_level[level_key] = tracked_ma

                dataset_name, network_tag = self._parse_graph_key(graph_key)
                row = {
                    "scope": "network",
                    "epoch": int(epoch),
                    "network_key": graph_key,
                    "dataset_name": dataset_name,
                    "network_id": network_tag,
                    "selection_freq_mean_by_level": selection_freq_mean_by_level,
                    "selection_freq_std_by_level": selection_freq_std_by_level,
                    "num_nodes_by_level": num_nodes_by_level,
                    "num_graphs_by_level": num_graphs_by_level,
                    "selection_freq_tracked_node_indices_by_level": selection_freq_tracked_node_indices_by_level,
                    "selection_freq_tracked_node_ma_by_level": selection_freq_tracked_node_ma_by_level,
                }
                f.write(json.dumps(row) + "\n")

        return summary_path

    @staticmethod
    def _is_task_v2(task: Optional[BaseTask]) -> bool:
        """Return whether current task uses evaluate_v2 pathway."""
        return task is not None and task.__class__.__name__ == "StockPriceForecastingTaskV2"

    def _resolve_epoch_summary_path(self) -> Optional[str]:
        """Resolve persistent epoch-summary JSONL path."""
        outdir = None
        try:
            hydra_cfg = HydraConfig.get()
            outdir = getattr(getattr(hydra_cfg, "runtime", None), "output_dir", None)
        except Exception:
            outdir = None
        if not outdir:
            outdir = self._resolve_run_output_dir(self.save_dir)
        if not outdir:
            return None
        os.makedirs(outdir, exist_ok=True)
        return os.path.join(outdir, "epoch_summaries.jsonl")

    def _append_epoch_summary_jsonl(self, epoch_metrics: Dict[str, Any]) -> Optional[str]:
        """Append one epoch summary row to `epoch_summaries.jsonl`."""
        summary_path = self._resolve_epoch_summary_path()
        if summary_path is None:
            log.warning("Skipping epoch summary JSONL write: no output directory resolved.")
            return None
        with open(summary_path, "a") as f:
            f.write(json.dumps(epoch_metrics) + "\n")
        return summary_path

    def _save_test_summary(self, test_metrics: Dict[str, Any], best_ckpt_path: Optional[str] = None) -> Optional[str]:
        """Save final test evaluation results to a separate JSON file.
        
        Args:
            test_metrics: Dictionary of test metrics
            best_ckpt_path: Path to the best checkpoint used for evaluation
            
        Returns:
            Path to the test summary file if successfully saved, None otherwise
        """
        outdir = None
        try:
            hydra_cfg = HydraConfig.get()
            outdir = getattr(getattr(hydra_cfg, "runtime", None), "output_dir", None)
        except Exception:
            outdir = None
        if not outdir:
            outdir = self._resolve_run_output_dir(self.save_dir)
        if not outdir:
            log.warning("Skipping test summary write: no output directory resolved.")
            return None
        
        os.makedirs(outdir, exist_ok=True)
        test_summary_path = os.path.join(outdir, "test_summary.json")
        
        test_summary = {
            "timestamp": datetime.now().isoformat(),
            "best_checkpoint": best_ckpt_path,
            "metrics": test_metrics,
        }
        
        with open(test_summary_path, "w") as f:
            json.dump(test_summary, f, indent=2)
        
        log.info(f"Saved test summary to {test_summary_path}")
        return test_summary_path

    def _save_checkpoint_if_scheduled(self, epoch: int, save_checkpoint_every_n_epochs: int) -> None:
        """Save periodic checkpoint if enabled."""
        if not self.save_dir:
            return
        if (epoch + 1) % int(save_checkpoint_every_n_epochs) != 0:
            return
        os.makedirs(self.save_dir, exist_ok=True)
        self.save_checkpoint(f"{self.save_dir}/{self.diffusion.__class__.__name__}_epoch_{epoch+1}.pt")
        log.info("Saved checkpoint at epoch %d to %s", epoch + 1, self.save_dir)

    def _plot_epoch_artifacts(
        self,
        *,
        epoch: int,
        epoch_summary_path: Optional[str],
        selector_diag_summary_path: Optional[str],
        selector_network_node_stats: Dict[str, Dict[int, Dict[str, torch.Tensor]]],
    ) -> None:
        """Generate periodic epoch/selector summary plots."""
        plot_epoch_summaries_every_n_epochs = 5
        # Mirror EvalSchedule.eval_on_last_epoch: force a final render so the
        # PDF on disk reflects the full training history (the forced last-epoch
        # eval is unlikely to coincide with the natural cadence multiple).
        is_last_epoch = (
            self.max_epochs is not None and int(epoch) == int(self.max_epochs) - 1
        )
        should_plot_epoch_summaries = is_last_epoch or self._should_plot_on_epoch(
            epoch=epoch,
            plot_every_n_epochs=plot_epoch_summaries_every_n_epochs,
        )

        should_plot_scalar_series = False
        should_plot_freq_evolution = False
        should_plot_node_stats = False
        should_plot_hard_mask = False

        if self.diagnose_model_enabled:
            selector_snapshot_every_n_epochs = int(self.diagnose_selector_every_n_epochs)
            if selector_snapshot_every_n_epochs <= 0:
                selector_snapshot_every_n_epochs = plot_epoch_summaries_every_n_epochs

            scalar_every_raw = self.plot_scalar_series_cfg.get("plot_every_n_epochs", None)
            if scalar_every_raw is None or int(scalar_every_raw) <= 0:
                scalar_every = plot_epoch_summaries_every_n_epochs
            else:
                scalar_every = int(scalar_every_raw)
            should_plot_scalar_series = bool(
                self.plot_scalar_series_cfg.get("enabled", True)
            ) and selector_diag_summary_path is not None and (
                is_last_epoch or self._should_plot_on_epoch(
                    epoch=epoch,
                    plot_every_n_epochs=scalar_every,
                )
            )

            freq_every_raw = self.plot_freq_evolution_cfg.get("plot_every_n_epochs", None)
            if freq_every_raw is None or int(freq_every_raw) <= 0:
                freq_every = plot_epoch_summaries_every_n_epochs
            else:
                freq_every = int(freq_every_raw)
            should_plot_freq_evolution = bool(
                self.plot_freq_evolution_cfg.get("enabled", True)
            ) and selector_diag_summary_path is not None and (
                is_last_epoch or self._should_plot_on_epoch(
                    epoch=epoch,
                    plot_every_n_epochs=freq_every,
                )
            )

            node_every_raw = self.plot_node_stats_cfg.get("plot_every_n_epochs", None)
            if node_every_raw is None or int(node_every_raw) <= 0:
                node_every = selector_snapshot_every_n_epochs
            else:
                node_every = int(node_every_raw)

            hard_every_raw = self.plot_hard_mask_cfg.get("plot_every_n_epochs", None)
            if hard_every_raw is None or int(hard_every_raw) <= 0:
                hard_every = selector_snapshot_every_n_epochs
            else:
                hard_every = int(hard_every_raw)

            if selector_network_node_stats:
                capped_node_stats = self._cap_network_level_stats(
                    selector_network_node_stats,
                    self.plot_node_stats_cfg.get("max_network_plots", -1),
                )
                should_plot_node_stats = bool(
                    self.plot_node_stats_cfg.get("enabled", True)
                ) and len(capped_node_stats) > 0 and (
                    is_last_epoch or self._should_plot_on_epoch(
                        epoch=epoch,
                        plot_every_n_epochs=node_every,
                    )
                )

                capped_hard_stats = self._cap_network_level_stats(
                    selector_network_node_stats,
                    self.plot_hard_mask_cfg.get("max_network_plots", -1),
                )
                should_plot_hard_mask = bool(
                    self.plot_hard_mask_cfg.get("enabled", True)
                ) and len(capped_hard_stats) > 0 and (
                    is_last_epoch or self._should_plot_on_epoch(
                        epoch=epoch,
                        plot_every_n_epochs=hard_every,
                    )
                )
        if not self.save_dir or not (
            should_plot_epoch_summaries
            or should_plot_scalar_series
            or should_plot_freq_evolution
            or should_plot_node_stats
            or should_plot_hard_mask
        ):
            return

        os.makedirs(self.save_dir, exist_ok=True)
        try:
            from graph_signal_diffusion.utils.plotting import plot_epoch_summaries_jsonl
            if should_plot_epoch_summaries and epoch_summary_path is not None:
                plot_epoch_summaries_jsonl(
                    epoch_summary_path,
                    self.save_dir,
                    overwrite=True,
                    grad_logscale=True,
                    save_pdf=True,
                )
            if self.diagnose_model_enabled and (
                should_plot_scalar_series
                or should_plot_freq_evolution
                or should_plot_node_stats
                or should_plot_hard_mask
            ):
                from graph_signal_diffusion.utils.plotting import (
                    plot_selector_diagnostics_jsonl,
                    plot_selector_network_hard_mask_statistics,
                    plot_selector_network_node_statistics,
                    plot_selector_network_selection_frequency_evolution,
                )

                selector_plot_dir = os.path.join(self.save_dir, "selector_diagnostics")
                if should_plot_scalar_series and selector_diag_summary_path is not None:
                    plot_selector_diagnostics_jsonl(
                        selector_diag_summary_path,
                        selector_plot_dir,
                        overwrite=True,
                        save_pdf=True,
                    )
                if should_plot_freq_evolution and selector_diag_summary_path is not None:
                    plot_selector_network_selection_frequency_evolution(
                        diagnostic_jsonl_path=selector_diag_summary_path,
                        out_dir=selector_plot_dir,
                        save_pdf=True,
                        max_network_plots=self.plot_freq_evolution_cfg.get(
                            "max_network_plots",
                            -1,
                        ),
                    )
                if should_plot_node_stats:
                    plot_selector_network_node_statistics(
                        selector_network_node_stats=self._cap_network_level_stats(
                            selector_network_node_stats,
                            self.plot_node_stats_cfg.get("max_network_plots", -1),
                        ),
                        epoch=epoch,
                        out_dir=selector_plot_dir,
                        save_pdf=True,
                    )
                if should_plot_hard_mask:
                    plot_selector_network_hard_mask_statistics(
                        selector_network_node_stats=self._cap_network_level_stats(
                            selector_network_node_stats,
                            self.plot_hard_mask_cfg.get("max_network_plots", -1),
                        ),
                        epoch=epoch,
                        out_dir=selector_plot_dir,
                        save_pdf=True,
                    )
            log.info("Updated training plots at %s", self.save_dir)
        except Exception as e:
            log.exception("Failed to update training plots: %s", e)


    def track_best_model(
        self,
        epoch: int,
        epoch_metrics: Dict[str, Any],
        *,
        has_val: bool,
    ) -> None:
        """Update best-model tracker and save checkpoint if a new best is found.

        Mutates *epoch_metrics* in-place to inject composite score / EMA info.
        """
        if self.best_model_tracker is None or not has_val:
            return

        raw_composite_score = self.best_model_tracker.raw_composite_score(epoch_metrics)
        composite_score = self.best_model_tracker.update(epoch, epoch_metrics)
        ema_snapshot = self.best_model_tracker.get_ema_metrics()
        epoch_metrics["best_model_raw_composite_score"] = raw_composite_score
        epoch_metrics["best_model_composite_score"] = composite_score
        epoch_metrics["best_model_ema_metrics"] = {
            k: v for k, v in ema_snapshot.items() if v is not None
        }

        if not self.best_model_tracker.should_save(composite_score):
            return

        best_dir = os.path.join(self.save_dir, "best_models") if self.save_dir else None
        if not best_dir:
            return

        os.makedirs(best_dir, exist_ok=True)
        ckpt_name = f"best_model_epoch_{epoch}.pt"
        ckpt_path = os.path.join(best_dir, ckpt_name)
        self.save_checkpoint(ckpt_path)

        evicted = self.best_model_tracker.register_checkpoint(
            epoch=epoch,
            score=composite_score,
            path=ckpt_path,
            raw_metrics={
                k: v for k, v in epoch_metrics.items()
                if isinstance(v, (int, float))
            },
        )
        self.best_model_tracker.update_symlink(best_dir)
        self.best_model_tracker.save_manifest(best_dir)
        log.info(
            "Saved best-model checkpoint: %s (score=%.4f, rank=%d/%d)",
            ckpt_name,
            composite_score,
            self.best_model_tracker.leaderboard_size,
            self.best_model_tracker.top_k,
        )
        if evicted:
            log.info("Evicted checkpoint: %s", evicted)



    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
            test_loader: Optional[DataLoader] = None, train_val_loader: Optional[DataLoader] = None,
            max_epochs: int = 1,
            save_checkpoint_every_n_epochs: int = 100,
            max_num_test_batches: Optional[int] = None,
    ):
        self.max_epochs = int(max_epochs)
        if self.max_epochs <= 0:
            raise ValueError(f"max_epochs must be > 0, got {self.max_epochs}")

        self._log_and_save_model_architecture_stats()

        epoch_schedule_fns = []
        step_schedule_fns = []
        for module in self.diffusion.modules():
            set_epoch_fn = getattr(module, "set_training_epoch", None)
            if callable(set_epoch_fn):
                epoch_schedule_fns.append(set_epoch_fn)
            set_step_fn = getattr(module, "set_training_step", None)
            if callable(set_step_fn):
                step_schedule_fns.append(set_step_fn)

        step = 0
        optim_step = 0
        selector_diag_manager = _SelectorDiagnosticsManager(self) if self.diagnose_model_enabled else None
        task_uses_v2 = self._is_task_v2(self.task)
        for epoch in range(self.max_epochs):
            self.epoch = epoch # for potential use in debug_print and evaluation criterions
            is_last_epoch = (epoch == self.max_epochs - 1)
            collect_heavy_selector_diags = self._should_collect_heavy_selector_diagnostics(
                epoch, is_last_epoch=is_last_epoch,
            )
            self._set_selector_heavy_diagnostics_enabled(collect_heavy_selector_diags)
            # Keep epoch propagation for any legacy modules still expecting it.
            for set_epoch_fn in epoch_schedule_fns:
                set_epoch_fn(epoch)

            epoch_losses = epoch_grad_norms = epoch_lr_sum = 0.0
            epoch_selector_grad_norms: Dict[str, float] = {}  # accumulated selector grad norms
            epoch_pred_stats_sum: Dict[str, float] = {}  # accumulated marginal pred/noise stats (eps-param diagnostic)
            epoch_pred_stats_count: int = 0
            epoch_forward_ms_sum = 0.0
            epoch_forward_steps = 0
            num_successful_steps = 0
            num_seen_steps = 0
            if selector_diag_manager is not None:
                selector_diag_manager.begin_epoch()
            self.diffusion.train()
            for data in train_loader:
                num_seen_steps += 1

                log.debug(f"Batch idx: {num_seen_steps - 1}\t Data: {data}")
                log.debug(f"Data.num_graphs: {data.num_graphs}")
                log.debug(f"Data.num_nodes: {data.num_nodes}") 

                # Propagate successful optimizer-step count to step-scheduled modules.
                for set_step_fn in step_schedule_fns:
                    set_step_fn(optim_step)

                self.optimizer.zero_grad(set_to_none=True)
                data = data.to(self.device)
                batch_graph_keys = self._extract_graph_keys(data)

                # Optional timing of the training forward path.
                timed_forward_ms: Optional[float] = None
                should_time_forward = (
                    self.measure_forward_speed
                    and (num_seen_steps - 1) >= self.forward_speed_warmup_steps
                    and (
                        self.forward_speed_max_steps < 0
                        or epoch_forward_steps < self.forward_speed_max_steps
                    )
                )
                fwd_t0: Optional[float] = None
                fwd_start_event: Optional[torch.cuda.Event] = None
                fwd_end_event: Optional[torch.cuda.Event] = None
                if should_time_forward and self.device.type == "cuda":
                    fwd_start_event = torch.cuda.Event(enable_timing=True)
                    fwd_end_event = torch.cuda.Event(enable_timing=True)
                    fwd_start_event.record()
                elif should_time_forward:
                    fwd_t0 = time.perf_counter()

                # Forward pass with optional AMP.
                # ``return_pred_stats=True`` adds marginal mean/std diagnostics for
                # the injected noise, the model prediction, the regression target,
                # and the residual. Used to detect noise-scale miscalibration
                # (e.g., pred_std systematically deviating from noise_std under
                # eps-parameterization).
                if self._amp_enabled:
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        loss, step_pred_stats = self.diffusion.training_loss(
                            data, return_pred_stats=True,
                        )
                else:
                    loss, step_pred_stats = self.diffusion.training_loss(
                        data, return_pred_stats=True,
                    )
                loss_value = self._as_float_scalar(loss)
                log.debug(f"Loss: {loss_value}")
                if should_time_forward:
                    if self.device.type == "cuda":
                        assert fwd_start_event is not None and fwd_end_event is not None
                        fwd_end_event.record()
                        fwd_end_event.synchronize()
                        timed_forward_ms = float(fwd_start_event.elapsed_time(fwd_end_event))
                    else:
                        assert fwd_t0 is not None
                        timed_forward_ms = float((time.perf_counter() - fwd_t0) * 1000.0)
                    epoch_forward_ms_sum += timed_forward_ms
                    epoch_forward_steps += 1

                # Selector diagnostics from this forward pass (if available).
                selector_diag = None
                if selector_diag_manager is not None:
                    selector_diag = selector_diag_manager.record_forward_pass(
                        self.diffusion,
                        batch_graph_keys,
                    )

                # LR used for this optimizer step (before optional scheduler.step()).
                step_lr = float(self.optimizer.param_groups[0]["lr"])

                # Backward pass with optional AMP
                if self._amp_enabled:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.diffusion.parameters(),
                        max_norm=self.max_grad_norm
                    )
                    
                    # Check if gradients are finite before stepping
                    scale_before = self.scaler.get_scale()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    scale_after = self.scaler.get_scale()
                    
                    # If scale decreased, step was skipped due to inf/nan gradients
                    if scale_after < scale_before:
                        log.warning(f"Epoch {epoch} Step {step} | Optimizer step SKIPPED due to inf/nan gradients. "
                                   f"Scale: {scale_before:.1f} -> {scale_after:.1f}")
                        step += 1  # Still increment step counter
                        # Don't accumulate this step's loss/grad in epoch stats
                        continue
                    grad_norm_value = self._as_float_scalar(grad_norm)
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                        
                else:
                    # Non-AMP path: skip optimizer step if loss is already non-finite.
                    if not self._is_finite_scalar(loss):
                        log.warning(
                            "Epoch %d Step %d | Optimizer step SKIPPED due to non-finite loss (non-AMP). "
                            "loss=%s",
                            epoch,
                            step,
                            loss_value,
                        )
                        step += 1
                        continue

                    if self.accelerator:
                        self.accelerator.backward(loss)
                    else:
                        loss.backward()

                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.diffusion.parameters(),
                        max_norm=self.max_grad_norm
                    )

                    # Non-AMP path: skip optimizer step if gradients are non-finite.
                    if not self._is_finite_scalar(grad_norm):
                        grad_norm_value = self._as_float_scalar(grad_norm)
                        log.warning(
                            "Epoch %d Step %d | Optimizer step SKIPPED due to non-finite gradients (non-AMP). "
                            "loss=%s grad_norm=%s",
                            epoch,
                            step,
                            loss_value,
                            grad_norm_value,
                        )
                        step += 1
                        continue

                    grad_norm_value = self._as_float_scalar(grad_norm)
                    self.optimizer.step()
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

                optim_step += 1
                epoch_grad_norms += grad_norm_value
                epoch_losses += loss_value
                epoch_lr_sum += step_lr
                num_successful_steps += 1

                # Accumulate noise/pred marginal stats for this step.
                if isinstance(step_pred_stats, dict):
                    for k, v in step_pred_stats.items():
                        if isinstance(v, (int, float)) and math.isfinite(v):
                            epoch_pred_stats_sum[k] = epoch_pred_stats_sum.get(k, 0.0) + float(v)
                    epoch_pred_stats_count += 1

                # Selector-parameter gradient norms (post-clip, pre-zero_grad).
                step_sel_gnorms = self._compute_selector_grad_norms(
                    self.diffusion.model
                )
                for k, v in step_sel_gnorms.items():
                    epoch_selector_grad_norms[k] = (
                        epoch_selector_grad_norms.get(k, 0.0) + v
                    )

                if step % self.log_every_n_steps == 0:
                    # msg = f"Epoch {epoch} Step {step} | loss={loss.item():.4f}"
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    msg = (
                        f"Epoch: {epoch} | Step: {step} | Loss: {loss_value:.4f} "
                        f"| Grad Norm: {grad_norm_value:.4f} | LR: {current_lr:.6e}"
                    )
                    if timed_forward_ms is not None:
                        msg += f" | FwdMs: {timed_forward_ms:.2f}"
                    if isinstance(selector_diag, dict) and selector_diag.get("num_selector_levels", 0) > 0:
                        msg += (
                            f" | SelTemp: {selector_diag.get('temperature', 0.0):.4f}"
                            f" | SelRatio: {selector_diag.get('selected_ratio_mean', 0.0):.4f}"
                            f" | SelEntReg: {selector_diag.get('entropy_reg', 0.0):.4f}"
                            f" | SelAux: {selector_diag.get('selector_aux_total', 0.0):.6f}"
                        )
                    log.info(msg)

                    # Confirm any GPU memory increase is due to caching of dynamic graphs.
                    log.info(f"\nGPU allocated: {torch.cuda.memory_allocated()/1e6:.0f}MB, "
                        f"reserved: {torch.cuda.memory_reserved()/1e6:.0f}MB")

                step += 1

            ### Epoch summary ###
            avg_epoch_loss = epoch_losses / max(num_successful_steps, 1)
            avg_epoch_grad_norm = epoch_grad_norms / max(num_successful_steps, 1)
            if num_successful_steps > 0:
                avg_epoch_lr = epoch_lr_sum / num_successful_steps
            else:
                avg_epoch_lr = float(self.optimizer.param_groups[0]["lr"])
            avg_epoch_forward_ms: Optional[float] = None
            avg_epoch_forward_steps_per_sec: Optional[float] = None
            if self.measure_forward_speed:
                if epoch_forward_steps > 0:
                    avg_epoch_forward_ms = epoch_forward_ms_sum / float(epoch_forward_steps)
                    if avg_epoch_forward_ms > 0.0:
                        avg_epoch_forward_steps_per_sec = 1000.0 / avg_epoch_forward_ms
                        log.info(
                            "Avg Forward (training_loss): %.3f ms/step | %.2f steps/s "
                            "(timed_steps=%d, warmup_steps=%d)",
                            avg_epoch_forward_ms,
                            avg_epoch_forward_steps_per_sec,
                            epoch_forward_steps,
                            self.forward_speed_warmup_steps,
                        )
                    else:
                        log.info(
                            "Avg Forward (training_loss): %.3f ms/step (timed_steps=%d)",
                            avg_epoch_forward_ms,
                            epoch_forward_steps,
                        )
                else:
                    log.info(
                        "Forward timing enabled but no steps were timed "
                        "(warmup_steps=%d, max_steps=%d).",
                        self.forward_speed_warmup_steps,
                        self.forward_speed_max_steps,
                    )
            if selector_diag_manager is not None:
                selector_epoch_diags = selector_diag_manager.finalize_epoch(
                    epoch,
                    write_network_rows=collect_heavy_selector_diags,
                )
            else:
                selector_epoch_diags = self._empty_selector_epoch_diagnostics()
            avg_selector_diags = selector_epoch_diags.avg_selector_diags

            log.info(f"=== Epoch {epoch} Summary ===")
            log.info(
                "Avg Loss: %.4f | Avg Grad Norm: %.4f | Avg LR: %.6e",
                avg_epoch_loss,
                avg_epoch_grad_norm,
                avg_epoch_lr,
            )
            if self.diagnose_model_enabled:
                log.info(
                    "Heavy selector diagnostics this epoch: %s (every %d epochs)",
                    "ON" if collect_heavy_selector_diags else "OFF",
                    self.diagnose_selector_every_n_epochs,
                )
            else:
                log.info("Model diagnostics are disabled (trainer.diagnose_model.enabled=false)")
            if num_successful_steps < num_seen_steps:
                log.warning(f"Skipped {num_seen_steps - num_successful_steps} steps due to inf/nan gradients")
            if avg_selector_diags:
                log.info(
                    "Avg Selector Diagnostics: temp=%.4f expl=%.4f ratio=%.4f ent_reg=%.4f aux=%.6f levels=%.2f",
                    avg_selector_diags.get("temperature", 0.0),
                    avg_selector_diags.get("exploration_noise", 0.0),
                    avg_selector_diags.get("selected_ratio_mean", 0.0),
                    avg_selector_diags.get("entropy_reg", 0.0),
                    avg_selector_diags.get("selector_aux_total", 0.0),
                    avg_selector_diags.get("num_selector_levels", 0.0),
                )
            if epoch_selector_grad_norms and num_successful_steps > 0:
                avg_sel_gn = {
                    k: v / num_successful_steps
                    for k, v in epoch_selector_grad_norms.items()
                }
                parts = " | ".join(
                    f"{k}={v:.4f}" for k, v in sorted(avg_sel_gn.items())
                )
                log.info("Avg Selector Grad Norms: %s", parts)

            # Epoch-averaged noise/pred stats. Diagnostic for noise-scale
            # miscalibration; under eps-parameterization, pred_std and
            # noise_std should both be ≈ 1.
            avg_epoch_pred_stats: Dict[str, float] = {}
            if epoch_pred_stats_count > 0:
                avg_epoch_pred_stats = {
                    k: v / epoch_pred_stats_count
                    for k, v in epoch_pred_stats_sum.items()
                }
                _key_order = [
                    "noise_mean", "noise_std",
                    "pred_mean", "pred_std",
                    "target_mean", "target_std",
                    "residual_std",
                ]
                parts = " | ".join(
                    f"{k}={avg_epoch_pred_stats[k]:.4f}"
                    for k in _key_order if k in avg_epoch_pred_stats
                )
                if parts:
                    log.info("Avg Noise/Pred Stats: %s", parts)

            # End of epoch evaluation
            should_eval = self.eval_schedule.should_eval(epoch, self.max_epochs)
            if epoch == 0 and not self.eval_on_first_epoch:
                should_eval = False
            val_metrics = None
            if val_loader is not None:
                if should_eval:
                    if task_uses_v2:
                        val_metrics = self.evaluate_v2(
                            val_loader,
                            collect_heavy_selector_diagnostics=collect_heavy_selector_diags,
                            eval_split="val",
                            selector_network_node_stats=selector_epoch_diags.selector_network_node_stats,
                            num_max_eval_batches=self.max_num_val_batches,
                        )
                    else:
                        val_metrics = self.evaluate(
                            val_loader,
                            collect_heavy_selector_diagnostics=collect_heavy_selector_diags,
                            eval_split="val",
                            selector_network_node_stats=selector_epoch_diags.selector_network_node_stats,
                            num_max_eval_batches=self.max_num_val_batches,
                        )
                else:
                    val_metrics = {}
                metrics_str = " | ".join([f"{k}={v:.4f}" for k, v in val_metrics.items()])
                log.info(f"[val] Epoch {epoch} | {metrics_str}")

            train_val_metrics = None
            if train_val_loader is not None:
                if should_eval:
                    if task_uses_v2:
                        train_val_metrics = self.evaluate_v2(
                            train_val_loader,
                            collect_heavy_selector_diagnostics=collect_heavy_selector_diags,
                            eval_split="train-val",
                            selector_network_node_stats=selector_epoch_diags.selector_network_node_stats,
                            num_max_eval_batches=self.max_num_train_val_batches,
                        )
                    else:
                        train_val_metrics = self.evaluate(
                            train_val_loader,
                            collect_heavy_selector_diagnostics=collect_heavy_selector_diags,
                            eval_split="train-val",
                            selector_network_node_stats=selector_epoch_diags.selector_network_node_stats,
                            num_max_eval_batches=self.max_num_train_val_batches,
                        )
                else:
                    train_val_metrics = {}
                metrics_str = " | ".join([f"{k}={v:.4f}" for k, v in train_val_metrics.items()])
                log.info(f"[train-val] Epoch {epoch} | {metrics_str}")

            epoch_metrics = {
                "epoch": epoch,
                "train_loss": avg_epoch_loss,      # scalar (float)
                "train_grad_norm": avg_epoch_grad_norm,  # scalar (float)
                "train_lr": avg_epoch_lr,  # scalar (float)
                "num_successful_steps": num_successful_steps,
                "num_skipped_steps": num_seen_steps - num_successful_steps,
                # "val_loss": metrics["loss"] if metrics is not None and "loss" in metrics else None,  # scalar (float) or None
                # "lr": current_lr,
            }
            if avg_epoch_forward_ms is not None:
                epoch_metrics["train_forward_ms_avg"] = avg_epoch_forward_ms
            if avg_epoch_forward_steps_per_sec is not None:
                epoch_metrics["train_forward_steps_per_sec"] = avg_epoch_forward_steps_per_sec
            if self.measure_forward_speed:
                epoch_metrics["train_forward_steps_timed"] = int(epoch_forward_steps)
                epoch_metrics["train_forward_warmup_steps"] = int(self.forward_speed_warmup_steps)

            for stat_name, stat_val in avg_epoch_pred_stats.items():
                epoch_metrics[f"train_{stat_name}"] = stat_val

            epoch_metrics.update({f"val_{k}": v for k, v in val_metrics.items()} if val_metrics is not None else {})
            epoch_metrics.update({f"train-val_{k}": v for k, v in train_val_metrics.items()} if train_val_metrics is not None else {})

            # Derived generalization-gap metric. Used by best-model trackers
            # that prefer epochs where val and train-val losses are close
            # (small gap ⇒ no overfitting). Emit only when both halves were
            # evaluated this epoch.
            _inject_val_loss_gap(epoch_metrics)

            # ── Inject selector diagnostics into epoch_metrics ────────
            # Only emit selector metrics when real selector levels are active.
            # For stride/no-downsampling ablations (num_selector_levels == 0 and
            # empty per-level stats), keep selector_* metrics out of W&B.
            level_stats = selector_epoch_diags.avg_selector_stats_by_level
            num_selector_levels = avg_selector_diags.get("num_selector_levels", None)
            has_real_selector_levels = (
                isinstance(num_selector_levels, (int, float))
                and float(num_selector_levels) > 0.0
            )
            if not has_real_selector_levels and isinstance(level_stats, dict):
                has_real_selector_levels = any(
                    isinstance(per_level, dict) and len(per_level) > 0
                    for per_level in level_stats.values()
                )

            if has_real_selector_levels:
                if avg_selector_diags:
                    for stat_name, stat_val in avg_selector_diags.items():
                        if not isinstance(stat_val, (int, float)):
                            continue
                        stat_val = float(stat_val)
                        if not math.isfinite(stat_val):
                            continue
                        # Avoid duplicated prefixes such as selector_selector_aux_total.
                        metric_name = str(stat_name)
                        if metric_name.startswith("selector_"):
                            metric_name = metric_name[len("selector_"):]
                        epoch_metrics[f"selector_{metric_name}"] = stat_val

                # Add per-level selector scalars as flattened keys
                # (e.g., selector_level0_entropy_reg).
                if isinstance(level_stats, dict):
                    for stat_name, per_level in level_stats.items():
                        if not isinstance(per_level, dict):
                            continue
                        stat_key = str(stat_name)
                        if stat_key.startswith("selector_"):
                            stat_key = stat_key[len("selector_"):]
                        for level_idx, stat_val in per_level.items():
                            if not isinstance(stat_val, (int, float)):
                                continue
                            stat_val = float(stat_val)
                            if not math.isfinite(stat_val):
                                continue
                            level_str = str(level_idx)
                            epoch_metrics[f"selector_level{level_str}_{stat_key}"] = stat_val

            # ── Selector gradient norms (epoch-averaged) ─────────────
            # Emit whenever selector gradients were observed this epoch.
            if epoch_selector_grad_norms and num_successful_steps > 0:
                for gn_key, gn_sum in epoch_selector_grad_norms.items():
                    epoch_metrics[gn_key] = gn_sum / num_successful_steps

            # Only update best-model tracker on epochs where task evaluation
            # actually ran (task metrics are the best-model criteria; on
            # non-eval epochs they are absent and _num_updates should not tick).
            has_task_metrics = val_metrics is not None and any(
                not k.startswith(("loss", "loss_t_")) for k in val_metrics
            )
            if has_task_metrics:
                self.track_best_model(epoch, epoch_metrics, has_val=True)

            # ── wandb epoch logging ───────────────────────────────────
            if self.wandb_run is not None:
                # Log all scalar epoch metrics; track_best_model may have
                # injected best_model_composite_score and best_model_ema_metrics.
                wandb_metrics = {
                    k: v for k, v in epoch_metrics.items()
                    if isinstance(v, (int, float))
                    # Skip per-subdataset _std keys to reduce W&B overhead
                    # (they are still written to epoch_summaries.jsonl).
                    and not ("subdataset/" in k and k.endswith("_std"))
                }
                
                # Flatten best_model_ema_metrics dict to individual wandb keys
                if "best_model_ema_metrics" in epoch_metrics:
                    ema_metrics = epoch_metrics["best_model_ema_metrics"]
                    if isinstance(ema_metrics, dict):
                        for metric_name, metric_value in ema_metrics.items():
                            if isinstance(metric_value, (int, float)):
                                wandb_metrics[f"best_model_ema_{metric_name}"] = metric_value
                
                self.wandb_run.log(wandb_metrics, step=epoch)

                # Update wandb summary with best composite score for sweep ranking
                if "best_model_composite_score" in epoch_metrics:
                    self.wandb_run.summary["best_composite_score"] = epoch_metrics["best_model_composite_score"]

            # Human-readable — goes to hydra.log
            log.info("Epoch summary: %s", json.dumps(epoch_metrics))

            summary_path = self._append_epoch_summary_jsonl(epoch_metrics)
            self._save_checkpoint_if_scheduled(epoch, save_checkpoint_every_n_epochs)
            self._plot_epoch_artifacts(
                epoch=epoch,
                epoch_summary_path=summary_path,
                selector_diag_summary_path=selector_epoch_diags.summary_path,
                selector_network_node_stats=selector_epoch_diags.selector_network_node_stats,
            )


        #### Final test evaluation on the whole test dataset after training loop ####
        if not self.run_final_test:
            log.info("Skipping final test evaluation (trainer.run_final_test=false).")
        elif test_loader is not None:
            test_batch_limit = (
                self.max_num_test_batches
                if max_num_test_batches is None
                else int(max_num_test_batches)
            )
            test_metrics = self.predict(
                test_loader,
                num_max_eval_batches=test_batch_limit,
            )

        


    def predict(
        self,
        test_loader,
        num_max_eval_batches: Optional[int] = None,
    ) -> Dict:
        if num_max_eval_batches is not None:
            num_max_eval_batches = int(num_max_eval_batches)

        log.info("=" * 80)
        if num_max_eval_batches is None or num_max_eval_batches < 0:
            log.info("Starting final test evaluation on complete test dataset")
        else:
            log.info(
                "Starting final test evaluation on up to %d batch(es)",
                num_max_eval_batches,
            )
        log.info("=" * 80)

        # Load the best model checkpoint before final evaluation if available
        best_ckpt_path = None
        if self.best_model_tracker is not None and self.best_model_tracker.best_path is not None:
            best_ckpt_path = self.best_model_tracker.best_path
            if best_ckpt_path and os.path.isfile(best_ckpt_path):
                self.load_checkpoint(best_ckpt_path)
                log.info(f"Loaded best model checkpoint from {best_ckpt_path} for final evaluation.")
            else:
                log.warning("Best model checkpoint path is invalid: %s", best_ckpt_path)
                best_ckpt_path = None
        else:
            log.info("Using final epoch model for test evaluation (no best model tracked).")

        task_uses_v2 = self._is_task_v2(self.task)
        # Evaluate on the final test split (optionally limited by num_max_eval_batches).
        if task_uses_v2:
            test_metrics = self.evaluate_v2(
                test_loader,
                collect_heavy_selector_diagnostics=False,
                eval_split="test",
                force_task_eval=True,  # Always evaluate task metrics on final test set
                num_max_eval_batches=num_max_eval_batches,
            )
        else:
            test_metrics = self.evaluate(
                test_loader,
                collect_heavy_selector_diagnostics=False,
                eval_split="test",
                force_task_eval=True,  # Always evaluate task metrics on final test set
                num_max_eval_batches=num_max_eval_batches,
            )
        
        # Log test metrics
        metrics_str = " | ".join([f"{k}={v:.4f}" for k, v in test_metrics.items()])
        log.info(f"[test] Final evaluation | {metrics_str}")
        
        # Save test summary to separate file
        test_summary_path = self._save_test_summary(test_metrics, best_ckpt_path)
        if test_summary_path:
            log.info(f"Test results saved to: {test_summary_path}")
        
        # Log test metrics to wandb if enabled
        if self.wandb_run is not None:
            wandb_test_metrics = {
                f"test_{k}": v for k, v in test_metrics.items()
                if isinstance(v, (int, float))
            }
            self.wandb_run.log(wandb_test_metrics)
            # Update wandb summary with test metrics for easy access
            for k, v in wandb_test_metrics.items():
                self.wandb_run.summary[k] = v
            log.info("Test metrics logged to wandb")
        
        log.info("=" * 80)
        log.info("Final test evaluation completed")
        log.info("=" * 80)

        return test_metrics

                


    def collect_selector_network_node_stats(
        self,
        loader: Union[DataLoader, Data],
        *,
        eval_split: str = "eval",
        num_max_batches: Optional[int] = None,
    ) -> Dict[str, Dict[int, Dict[str, torch.Tensor]]]:
        """Forward-only selector probe over ``loader`` → per-network node stats.

        Mirrors the training-loop selector probe (``_SelectorDiagnosticsManager``):
        for each batch it runs a forward pass with heavy selector diagnostics
        enabled and records the per-network, per-level masks, then finalizes into
        the same ``selector_network_node_stats`` dict (keyed
        ``dataset::network_<id>`` → level → ``{hard_mask_mean, ...}``) consumed by
        the WRA selector-rate-correlation visualization.

        Unlike the in-epoch probe, this is a standalone pass usable for offline
        regeneration from a checkpoint. It temporarily forces
        ``diagnose_model_enabled`` and heavy diagnostics on, restoring both
        afterwards. Returns ``{}`` when the model exposes no selector diagnostics.
        """
        if isinstance(loader, Data):
            loader = [loader]
        max_batches = (
            len(loader) if (num_max_batches is None or num_max_batches < 0) else num_max_batches
        )
        manager = _SelectorDiagnosticsManager(self)
        manager.begin_epoch()
        prev_diagnose = self.diagnose_model_enabled
        prev_heavy = bool(getattr(self.diffusion, "collect_selector_heavy_diagnostics", True))
        self.diagnose_model_enabled = True
        self.diffusion.eval()
        self._set_selector_heavy_diagnostics_enabled(True)
        try:
            with torch.no_grad():
                for batch_idx, data in enumerate(
                    tqdm(loader, desc=f"Selector probe on {eval_split} set", unit="batch", total=max_batches)
                ):
                    data = data.to(self.device)
                    batch_graph_keys = self._extract_graph_keys(data)
                    # Forward pass populates diffusion.last_selector_{,level_}diagnostics.
                    self.diffusion.training_loss(data)
                    manager.record_forward_pass(self.diffusion, batch_graph_keys)
                    if (batch_idx + 1) >= max_batches:
                        break
        finally:
            self._set_selector_heavy_diagnostics_enabled(prev_heavy)
            self.diagnose_model_enabled = prev_diagnose
        epoch_diags = manager.finalize_epoch(self.epoch, write_network_rows=True)
        stats = epoch_diags.selector_network_node_stats
        # Offload to CPU and release GPU memory before the (sampling-heavy) eval
        # pass — the viz reads these as numpy anyway. Avoids the probe's resident
        # tensors stacking with eval activations and causing OOM.
        for levels in stats.values():
            for level_stats in levels.values():
                for key, value in list(level_stats.items()):
                    if isinstance(value, torch.Tensor):
                        level_stats[key] = value.detach().to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return stats

    @torch.inference_mode()
    def evaluate(
        self,
        loader: Union[DataLoader, Data],
        collect_heavy_selector_diagnostics: bool = True,
        eval_split: str = "eval",
        selector_network_node_stats: Optional[Dict[str, Dict[int, Dict[str, torch.Tensor]]]] = None,
        force_task_eval: bool = False,
        num_max_eval_batches: Optional[int] = None,
        batch_hook: Optional[Callable] = None,
        record_sink: Optional[list] = None,
    ) -> dict:
        """Evaluation loop (v1 path, used by WRA tasks).

        Args:
            batch_hook: Optional callback invoked with
                ``(real_samples, generated_samples_normalized, metadata)``
                immediately after sampling each batch.  All tensors keep
                their original dtype / device and are **not** denormalized.
                Callers (e.g.
                ``DiffusionBaseline``) use this to accumulate tensors for
                structural-metric computation without a second inference pass.
                An optional ``data=`` keyword argument is passed when the
                hook's signature accepts it.
        """
        # Check once whether the hook accepts a ``data`` keyword.
        _hook_accepts_data = False
        if batch_hook is not None:
            _sig = inspect.signature(batch_hook)
            _hook_accepts_data = (
                "data" in _sig.parameters
                or any(
                    p.kind == inspect.Parameter.VAR_KEYWORD
                    for p in _sig.parameters.values()
                )
            )

        if isinstance(loader, Data):
            loader = [loader]

        # Resolve effective batch cap: None or -1 → process all batches
        max_batches = len(loader) if (num_max_eval_batches is None or num_max_eval_batches < 0) else num_max_eval_batches

        # Task-specific metrics accumulator
        task_metrics: Dict[str, float] = {}
        task_metric_counts: Dict[str, int] = {}
        total_loss, count = 0.0, 0

        # Per-timestep-bin loss accumulator (populated when diffusion supports stratified sampling)
        bin_loss_accumulator: Dict[str, List[float]] = {}
        # Per-batch noise/pred stats accumulator (marginal + per-bin keys).
        pred_stats_accumulator: Dict[str, List[float]] = {}
        # Per-batch selector sampling diagnostics, pooled after the loop into a
        # single test-set-wide aggregate heatmap (emitted alongside per-batch ones).
        agg_sampling_diags: List[Dict[str, Any]] = []
        agg_sampling_viz_dir: Optional[str] = None

        # Setup autocast context for AMP evaluation (loss-only; sampling runs outside).
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16) if self._amp_enabled else nullcontext()

        prev_heavy_diag = bool(
            getattr(self.diffusion, "collect_selector_heavy_diagnostics", True)
        )
        self.diffusion.eval()
        self._set_selector_heavy_diagnostics_enabled(collect_heavy_selector_diagnostics)
        try:
            for batch_idx, data in enumerate(tqdm(loader, desc=f"Evaluating on {eval_split} set", unit="batch", total=max_batches)):
                    data = data.to(self.device)

                    # Stratified-t loss: same scalar, but also fills per-bin stats
                    # AND marginal/per-bin noise-prediction diagnostics.
                    with autocast_ctx:
                        if hasattr(self.diffusion, "training_loss_stratified_t"):
                            loss, bin_losses, step_pred_stats = self.diffusion.training_loss_stratified_t(
                                data, return_pred_stats=True,
                            )
                            for k, vs in bin_losses.items():
                                bin_loss_accumulator.setdefault(k, []).extend(vs)
                            for k, v in step_pred_stats.items():
                                if isinstance(v, (int, float)) and math.isfinite(v):
                                    pred_stats_accumulator.setdefault(k, []).append(float(v))
                        else:
                            loss = self.diffusion.training_loss(data)
                    total_loss += loss.item()
                    count += 1

                    if self.task is not None:
                        # Prepare data and get ground truth (real) samples (DENORMALIZED by task.prepare_data)
                        data_dict = self.task.prepare_data(data)
                        real_samples = data_dict["samples"]
                        metadata = data_dict["metadata"]
                        if isinstance(selector_network_node_stats, dict) and len(selector_network_node_stats) > 0:
                            metadata["selector_network_node_stats"] = selector_network_node_stats
                        # Per-pass batch position — lets per-batch task viz accumulate
                        # across batches (e.g. the selector-rate aggregate) and flush
                        # on the final batch.
                        metadata["eval_batch_idx"] = int(batch_idx)
                        metadata["eval_num_batches"] = int(max_batches)

                        # Generate (predicted) samples (NORMALIZED)
                        # Get n_samples_per_input from task config (default to 1 if not specified)
                        n_samples_per_input = getattr(self.task, 'n_samples_per_input', 1)
                        
                        # Clone batch if generating multiple samples per input
                        if n_samples_per_input > 1:
                            from torch_geometric.data import Batch
                            
                            # Clone the data object (graphs) n_samples_per_input times
                            if isinstance(data, Data) and not isinstance(data, Batch):
                                data_cloned = Batch.from_data_list([data] * n_samples_per_input)
                            else:
                                # Unbatch, repeat each graph, rebatch
                                data_list = data.to_data_list()
                                data_cloned = Batch.from_data_list(
                                    [g for g in data_list for _ in range(n_samples_per_input)]
                                )
                            
                            # Adjust shape for cloned batch
                            B_original = real_samples.shape[0]
                            sample_shape = (B_original * n_samples_per_input, *real_samples.shape[1:])
                        else:
                            data_cloned = data
                            sample_shape = real_samples.shape

                        generated_samples_normalized, selector_sampling_diagnostics = self._sample_with_selector_sampling_diagnostics(
                            shape=sample_shape,
                            data=data_cloned,
                        )
                        if isinstance(selector_sampling_diagnostics, dict) and len(selector_sampling_diagnostics) > 0:
                            metadata["selector_sampling_diagnostics"] = selector_sampling_diagnostics

                        if batch_hook is not None:
                            if _hook_accepts_data:
                                batch_hook(real_samples, generated_samples_normalized, metadata, data=data_cloned)
                            else:
                                batch_hook(real_samples, generated_samples_normalized, metadata)

                        # Update metadata after sampling if multiple samples were generated
                        if n_samples_per_input > 1:
                            metadata['n_samples_per_input'] = n_samples_per_input
                            metadata['batch_size_cloned'] = B_original * n_samples_per_input
                            log.info(
                                f"Generated {n_samples_per_input} samples per input. "
                                f"Generated shape: {generated_samples_normalized.shape}, Real shape: {real_samples.shape}"
                            )

                        # NEW: Use TASK'S Normalizer to DENORMALIZE generated samples
                        # Note: Normalizer operates on batch dimension independently, so (B*n, T, N, F) is fine
                        if hasattr(self.task, "normalizer") and self.task.normalizer is not None and getattr(self.task.normalizer, 'fitted', False):
                            log.info("Denormalizing generated samples for task evaluation...")
                            generated_samples = self.task.normalizer.denormalize(generated_samples_normalized)
                            log.info(f"After denorm - generated_samples Mean: {generated_samples.mean():.4f}")
                        else:
                            generated_samples = generated_samples_normalized
                        
                        # Determine visualization save directory
                        # Priority: 1) self.save_dir (set by test.py), 2) Hydra output_dir (training), 3) None
                        if self.save_dir:
                            viz_save_dir = os.path.join(self.save_dir, f"eval_viz/epoch_{self.epoch}")
                        elif HydraConfig.get().runtime.output_dir:
                            viz_save_dir = os.path.join(HydraConfig.get().runtime.output_dir, f"eval_viz/epoch_{self.epoch}")
                        else:
                            viz_save_dir = None

                        self._plot_selector_sampling_diagnostics(
                            selector_sampling_diagnostics=selector_sampling_diagnostics,
                            data=data_cloned,
                            viz_save_dir=viz_save_dir,
                            eval_split=eval_split,
                            batch_idx=batch_idx,
                        )
                        if isinstance(selector_sampling_diagnostics, dict) and len(selector_sampling_diagnostics) > 0:
                            agg_sampling_diags.append(selector_sampling_diagnostics)
                            agg_sampling_viz_dir = viz_save_dir
                        
                        metrics = self.task.evaluate_samples(
                            generated_samples,
                            real_samples,
                            metadata,
                            viz_save_dir
                        )
                        metrics = self._normalize_violation_metric_names(metrics)

                        if record_sink is not None and hasattr(self.task, "last_evaluation_records"):
                            record_sink.extend(self.task.last_evaluation_records)

                        # Accumulate metrics with per-key counts so that
                        # keys present in only a subset of batches (e.g.
                        # subdataset/* metrics) are averaged correctly.
                        for k, v in metrics.items():
                            if k not in task_metrics:
                                task_metrics[k] = 0.0
                                task_metric_counts[k] = 0
                            task_metrics[k] += v
                            task_metric_counts[k] += 1

                    if count >= max_batches:
                        break

            self._plot_aggregate_selector_sampling_diagnostics(
                agg_sampling_diags,
                viz_save_dir=agg_sampling_viz_dir,
                eval_split=eval_split,
            )
        finally:
            self._set_selector_heavy_diagnostics_enabled(prev_heavy_diag)

        # Average metrics
        result = {"loss": total_loss / max(count, 1)}
        if self.task is not None:
            for k in task_metrics:
                result[k] = task_metrics[k] / max(task_metric_counts[k], 1)

        # Per-timestep-bin stats (mean + std per bin; absent if no graphs were evaluated)
        result.update(self._compute_stratified_t_bin_stats(bin_loss_accumulator))

        # Noise/pred marginal + per-bin diagnostics, averaged over batches.
        for stat_name, values in pred_stats_accumulator.items():
            if values:
                result[stat_name] = sum(values) / len(values)

        result = self._normalize_violation_metric_names(result)
        return result

    @torch.inference_mode()
    def evaluate_v2(
        self,
        loader: Union[DataLoader, Data],
        collect_heavy_selector_diagnostics: bool = True,
        eval_split: str = "eval",
        selector_network_node_stats: Optional[Dict[str, Dict[int, Dict[str, torch.Tensor]]]] = None,
        force_task_eval: bool = False,
        num_max_eval_batches: Optional[int] = None,
        batch_hook: Optional[Callable] = None,
    ) -> dict:
        """
        Evaluation method for StockPriceForecastingTaskV2.

        Simplified version that relies on task.dataset_info being pre-injected by CLI.
        No need to pass dataset_info around since task already has it.

        Args:
            num_max_eval_batches: Maximum number of batches to process. None or -1 means
                process all batches in the loader.
            batch_hook: Optional callback invoked with
                ``(real_samples, generated_samples, metadata)`` immediately
                after sampling each batch.  All tensors are **standardized**
                ``[B, T, N, F]`` on the trainer's device.  Callers (e.g.
                ``DiffusionBaseline``) use this to accumulate tensors for
                structural-metric computation without a second inference pass.
                An optional ``data=`` keyword argument is passed when the
                hook's signature accepts it.
        """
        # Check once whether the hook accepts a ``data`` keyword.
        _hook_accepts_data = False
        if batch_hook is not None:
            _sig = inspect.signature(batch_hook)
            _hook_accepts_data = (
                "data" in _sig.parameters
                or any(
                    p.kind == inspect.Parameter.VAR_KEYWORD
                    for p in _sig.parameters.values()
                )
            )

        if isinstance(loader, Data):
            loader = [loader]

        # Resolve effective batch cap: None or -1 → process all batches
        max_batches = len(loader) if (num_max_eval_batches is None or num_max_eval_batches < 0) else num_max_eval_batches

        # Task-specific metrics accumulator
        task_metrics: Dict[str, float] = {}
        task_metric_counts: Dict[str, int] = {}
        total_loss, count = 0.0, 0

        # Per-timestep-bin loss accumulator (populated when diffusion supports stratified sampling)
        bin_loss_accumulator: Dict[str, List[float]] = {}
        # Per-batch noise/pred stats accumulator (marginal + per-bin keys).
        pred_stats_accumulator: Dict[str, List[float]] = {}
        # Per-batch selector sampling diagnostics, pooled after the loop into a
        # single test-set-wide aggregate heatmap (emitted alongside per-batch ones).
        agg_sampling_diags: List[Dict[str, Any]] = []
        agg_sampling_viz_dir: Optional[str] = None

        # Setup autocast context for AMP evaluation (loss-only; sampling runs outside).
        autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.float16) if self._amp_enabled else nullcontext()

        prev_heavy_diag = bool(
            getattr(self.diffusion, "collect_selector_heavy_diagnostics", True)
        )
        self.diffusion.eval()
        self._set_selector_heavy_diagnostics_enabled(collect_heavy_selector_diagnostics)
        try:
            for batch_idx, data in enumerate(tqdm(loader, desc=f"Evaluating on {eval_split} set", unit="batch", total=max_batches)):
                    data = data.to(self.device)

                    # Stratified-t loss: same scalar, but also fills per-bin stats
                    # AND marginal/per-bin noise-prediction diagnostics.
                    with autocast_ctx:
                        if hasattr(self.diffusion, "training_loss_stratified_t"):
                            loss, bin_losses, step_pred_stats = self.diffusion.training_loss_stratified_t(
                                data, return_pred_stats=True,
                            )
                            for k, vs in bin_losses.items():
                                bin_loss_accumulator.setdefault(k, []).extend(vs)
                            for k, v in step_pred_stats.items():
                                if isinstance(v, (int, float)) and math.isfinite(v):
                                    pred_stats_accumulator.setdefault(k, []).append(float(v))
                        else:
                            loss = self.diffusion.training_loss(data)
                    total_loss += loss.item()
                    count += 1

                    if self.task is not None:
                        # Prepare data and get ground truth (real) samples (STANDARDIZED)
                        # Task accesses dataset_info via self.task.dataset_info (injected by CLI)
                        data_dict = self.task.prepare_data(data)
                        real_samples = data_dict["samples"]
                        metadata = data_dict["metadata"]
                        if isinstance(selector_network_node_stats, dict) and len(selector_network_node_stats) > 0:
                            metadata["selector_network_node_stats"] = selector_network_node_stats

                        # NOTE: No need to reshape data.x here - diffusion.sample() already handles
                        # the reshape from [B*N, T, F] to [B, T, N, F] internally (same as training_loss)
                        generated_samples, selector_sampling_diagnostics = self._sample_with_selector_sampling_diagnostics(
                            shape=real_samples.shape,
                            data=data,
                        )
                        if isinstance(selector_sampling_diagnostics, dict) and len(selector_sampling_diagnostics) > 0:
                            metadata["selector_sampling_diagnostics"] = selector_sampling_diagnostics

                        # Optional variance correction in standardized space.
                        # Runs BEFORE batch_hook so both task-level metrics
                        # and structural diagnostics see the corrected
                        # samples. Gated on (a) task has the method
                        # (V2-only), (b) task.variance_correction is set
                        # AND mode != off. See
                        # docs/agent_summaries/variance_correction_plan_2026-05-20.md.
                        _vc = getattr(self.task, "variance_correction", None)
                        if (
                            _vc is not None
                            and getattr(_vc, "mode", "off") != "off"
                            and hasattr(self.task, "apply_variance_correction")
                        ):
                            generated_samples = self.task.apply_variance_correction(
                                generated_samples, metadata,
                            )

                        if batch_hook is not None:
                            if _hook_accepts_data:
                                batch_hook(real_samples, generated_samples, metadata, data=data)
                            else:
                                batch_hook(real_samples, generated_samples, metadata)

                        # Determine visualization save directory
                        if self.save_dir:
                            viz_save_dir = os.path.join(self.save_dir, f"eval_viz/epoch_{self.epoch}")
                        elif HydraConfig.get().runtime.output_dir:
                            viz_save_dir = os.path.join(HydraConfig.get().runtime.output_dir, f"eval_viz/epoch_{self.epoch}")
                        else:
                            viz_save_dir = None

                        self._plot_selector_sampling_diagnostics(
                            selector_sampling_diagnostics=selector_sampling_diagnostics,
                            data=data,
                            viz_save_dir=viz_save_dir,
                            eval_split=eval_split,
                            batch_idx=batch_idx,
                        )
                        if isinstance(selector_sampling_diagnostics, dict) and len(selector_sampling_diagnostics) > 0:
                            agg_sampling_diags.append(selector_sampling_diagnostics)
                            agg_sampling_viz_dir = viz_save_dir

                        # Evaluate samples using task (handles destandardization internally)
                        metrics = self.task.evaluate_samples(
                            generated_samples=generated_samples,
                            real_samples=real_samples,
                            metadata=metadata,
                            viz_save_dir=viz_save_dir
                        )
                        metrics = self._normalize_violation_metric_names(metrics)

                        # Accumulate metrics with per-key counts so that
                        # keys present in only a subset of batches (e.g.
                        # subdataset/* metrics) are averaged correctly.
                        for k, v in metrics.items():
                            if k not in task_metrics:
                                task_metrics[k] = 0.0
                                task_metric_counts[k] = 0
                            task_metrics[k] += v
                            task_metric_counts[k] += 1

                    if count >= max_batches:
                        break

            self._plot_aggregate_selector_sampling_diagnostics(
                agg_sampling_diags,
                viz_save_dir=agg_sampling_viz_dir,
                eval_split=eval_split,
            )
        finally:
            self._set_selector_heavy_diagnostics_enabled(prev_heavy_diag)

        # Average metrics
        result = {"loss": total_loss / max(count, 1)}
        if self.task is not None:
            for k in task_metrics:
                result[k] = task_metrics[k] / max(task_metric_counts[k], 1)

        # Per-timestep-bin stats (mean + std per bin; absent if no graphs were evaluated)
        result.update(self._compute_stratified_t_bin_stats(bin_loss_accumulator))

        # Noise/pred marginal + per-bin diagnostics, averaged over batches.
        for stat_name, values in pred_stats_accumulator.items():
            if values:
                result[stat_name] = sum(values) / len(values)

        result = self._normalize_violation_metric_names(result)
        return result


    def extract_dataset_info(self, dataset: Union[Dataset, Subset]) -> dict:
        """Extract dataset info for task evaluation.
        
        For tasks with injected dataset info (e.g., WRA), use task.dataset_info.
        For tasks needing per-sample info (e.g., stock forecasting), extract from samples.
        """
        # If task has dataset_info attribute (injected via set_dataset_info), use it
        if self.task is not None and hasattr(self.task, 'dataset_info') and self.task.dataset_info is not None:
            log.info(f"Using dataset info from task.dataset_info")
            return self.task.dataset_info
        
        # Unwrap ReplicatedDataset if present
        from graph_signal_diffusion.datasets.replicated_dataset import ReplicatedDataset
        if isinstance(dataset, ReplicatedDataset):
            log.info(f"Unwrapping ReplicatedDataset to access base dataset")
            dataset = dataset.dataset
        
        # Otherwise extract from individual samples (for stock forecasting)
        if isinstance(dataset, Subset):
            info = dataset.dataset[0].get("info", {})
        else:
            info = dataset[0].get("info", {})
        
        log.info(f"Extracted dataset info from samples")
        return info
        

    def save_checkpoint(self, path: str):
        # Save diffusion + underlying model weights
        # Handle torch.compile wrapping: temporarily unwrap to get clean state dict without _orig_mod prefix        
        # Save reference to compiled models
        diffusion_was_compiled = hasattr(self.diffusion, "_orig_mod")
        model_was_compiled = hasattr(self.diffusion, "model") and hasattr(self.diffusion.model, "_orig_mod")
        
        # Temporarily swap in unwrapped versions for state_dict extraction
        if model_was_compiled:
            compiled_model = self.diffusion.model
            self.diffusion.model = compiled_model._orig_mod
        
        if diffusion_was_compiled:
            compiled_diffusion = self.diffusion
            # Can't unwrap diffusion itself easily, so we'll save the _orig_mod
            diffusion_to_save = compiled_diffusion._orig_mod
        else:
            diffusion_to_save = self.diffusion
        
        state = {
            "epoch": self.epoch,
            "diffusion": diffusion_to_save.state_dict(),
            "model": self.diffusion.model.state_dict() if hasattr(self.diffusion, "model") else None,
            "optimizer": self.optimizer.state_dict(),
            "use_amp": self.use_amp,  # Metadata for reference
        }
        
        # Restore compiled versions
        if model_was_compiled:
            self.diffusion.model = compiled_model
        
        # Save GradScaler state for AMP training resumption
        if self.scaler is not None:
            state["scaler"] = self.scaler.state_dict()
        
        if self.accelerator:
            self.accelerator.save(state, path)
        else:
            torch.save(state, path)

    def load_checkpoint(self, path: str) -> int:
        """Load model state from checkpoint.
        
        Robust to torch.compile and AMP:
        - Automatically unwraps compiled models before loading
        - Restores GradScaler state for AMP training resumption
        - Works whether checkpoint was saved with/without compile
        
        Args:
            path: Path to checkpoint file
        
        Returns:
            Epoch number of loaded checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)

        def _adapt_legacy_model_state_dict(raw_state: dict[str, Any]) -> dict[str, Any]:
            if not isinstance(raw_state, dict):
                return raw_state

            adapted: dict[str, Any] = {}
            for key, value in raw_state.items():
                if not isinstance(key, str):
                    continue

                remapped = key
                if remapped.startswith("module."):
                    remapped = remapped[len("module."):]
                if remapped.startswith("shared_cond_encoder."):
                    remapped = f"encoder.cond_encoder.{remapped.split('.', 1)[1]}"
                elif remapped.startswith("model.shared_cond_encoder."):
                    remapped = f"encoder.cond_encoder.{remapped.split('.', 2)[2]}"

                if remapped in adapted:
                    log.debug(
                        "Legacy checkpoint remap produced duplicate key %s (from %s); "
                        "keeping the last-seen value.",
                        remapped, key,
                    )
                adapted[remapped] = value

            return adapted

        def _adapt_legacy_diffusion_state_dict(raw_state: dict[str, Any]) -> dict[str, Any]:
            if not isinstance(raw_state, dict):
                return raw_state

            adapted: dict[str, Any] = {}
            for key, value in raw_state.items():
                if not isinstance(key, str):
                    continue

                remapped = key
                if remapped.startswith("module."):
                    remapped = remapped[len("module."):]
                if remapped.startswith("shared_cond_encoder."):
                    remapped = f"model.encoder.cond_encoder.{remapped.split('.', 1)[1]}"
                elif remapped.startswith("model.shared_cond_encoder."):
                    remapped = f"model.encoder.cond_encoder.{remapped.split('.', 2)[2]}"

                if remapped in adapted:
                    log.debug(
                        "Legacy diffusion state remap produced duplicate key %s (from %s); "
                        "keeping the last-seen value.",
                        remapped, key,
                    )
                adapted[remapped] = value

            return adapted

        def _load_state_dict_with_compat(
            module: torch.nn.Module,
            raw_state: dict[str, Any],
            target_name: str,
            remap_fn,
        ) -> None:
            if raw_state is None:
                return
            if not isinstance(raw_state, dict):
                raise TypeError(
                    f"Expected dict checkpoint state for {target_name}, got "
                    f"{type(raw_state)!r}"
                )

            try:
                module.load_state_dict(raw_state)
                return
            except RuntimeError as primary_error:
                adapted_state = remap_fn(raw_state)
                if adapted_state == raw_state:
                    raise primary_error
                try:
                    module.load_state_dict(adapted_state)
                    log.warning(
                        "Applied legacy UGNN checkpoint key remap for %s. "
                        "This indicates the checkpoint used legacy key layout "
                        "(e.g., shared_cond_encoder / wrapped prefixes).",
                        target_name,
                    )
                except RuntimeError as compat_error:
                    raise RuntimeError(
                        f"Failed to load {target_name} state dict from checkpoint "
                        f"even after legacy key remapping."
                    ) from compat_error

        # Handle torch.compile: temporarily unwrap for loading, then restore
        diffusion_was_compiled = hasattr(self.diffusion, "_orig_mod")
        model_was_compiled = hasattr(self.diffusion, "model") and hasattr(self.diffusion.model, "_orig_mod")
        
        # Save references to compiled versions
        if model_was_compiled:
            compiled_model = self.diffusion.model
            self.diffusion.model = compiled_model._orig_mod
        
        if diffusion_was_compiled:
            compiled_diffusion = self.diffusion
            diffusion_to_load = compiled_diffusion._orig_mod
        else:
            diffusion_to_load = self.diffusion
        
        # Load diffusion state (checkpoint saved without _orig_mod prefix)
        if "diffusion" in checkpoint:
            _load_state_dict_with_compat(
                diffusion_to_load,
                checkpoint["diffusion"],
                "diffusion",
                _adapt_legacy_diffusion_state_dict,
            )
        
        # Load underlying model state if it exists
        if "model" in checkpoint and checkpoint["model"] is not None:
            if hasattr(self.diffusion, "model") and self.diffusion.model is not None:
                _load_state_dict_with_compat(
                    self.diffusion.model,
                    checkpoint["model"],
                    "model",
                    _adapt_legacy_model_state_dict,
                )
        
        # Restore compiled versions
        if model_was_compiled:
            self.diffusion.model = compiled_model
        if diffusion_was_compiled:
            self.diffusion = compiled_diffusion
        
        # Load optimizer state (useful for resuming training)
        if "optimizer" in checkpoint and hasattr(self, "optimizer"):
            try:
                self.optimizer.load_state_dict(checkpoint["optimizer"])
            except Exception as e:
                log.warning(f"Could not load optimizer state: {e}")
        
        # Load GradScaler state for AMP training resumption
        if "scaler" in checkpoint and self.scaler is not None:
            try:
                self.scaler.load_state_dict(checkpoint["scaler"])
                log.info("Restored GradScaler state for AMP training")
            except Exception as e:
                log.warning(f"Could not load GradScaler state: {e}")
        
        epoch = checkpoint.get("epoch", -1)
        
        # Log metadata if available
        if "use_amp" in checkpoint:
            saved_amp = checkpoint["use_amp"]
            if saved_amp != self.use_amp:
                log.warning(
                    f"Checkpoint was saved with use_amp={saved_amp}, but current trainer has use_amp={self.use_amp}"
                )
        
        log.info(f"Loaded checkpoint from epoch {epoch}")
        return epoch


    def debug_print(self, msg: str):
        # Simple debug print function; can be enhanced to use logging
        if self.DEBUG_FLAG:
            log.debug(f"[DEBUG] {msg}")
