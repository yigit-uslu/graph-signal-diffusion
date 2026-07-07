"""Diffusion model wrapped as a ``BaseBaseline`` for ``UnifiedEvaluator``,
``evaluate.py``, and ``compare_baselines.py``.

Primary workflow (checkpoint-based evaluation)::

    python -m graph_signal_diffusion.cli.evaluate \\
        baseline=diffusion \\
        checkpoint_path=/path/to/experiment/trainer_chkpts/DDPM_epoch_1500.pt \\
        dataset=sp500_cleaned  task=stock_price_forecasting_v2

The user **never** needs to specify model / diffusion architecture: the
baseline auto-discovers the experiment's ``.hydra/config.yaml``, reconstructs
the full pipeline, and loads the saved weights.
"""
from __future__ import annotations

import logging
import os
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from graph_signal_diffusion.baselines import BASELINE_REGISTRY
from graph_signal_diffusion.baselines.base import BaseBaseline
from graph_signal_diffusion.evaluation import structural_metrics as sm
from graph_signal_diffusion.trainers.trainer import DiffusionTrainer
from graph_signal_diffusion.utils.selector_temperature_restore import (
    restore_selector_temperature_from_records,
)

log = logging.getLogger(__name__)


@dataclass
class VarianceCorrectionConfig:
    """Resolved variance-correction settings for a single evaluate() call.

    Built in ``DiffusionBaseline.__init__`` / ``_build_from_experiment`` from
    (in precedence order): CLI override → checkpoint-embedded ``alpha_ema``
    → default ``mode='off'``. Set transiently on the task instance during
    ``evaluate()`` so non-V2 tasks see no field at all.

    Attribute contract matches the duck-typed access in
    ``StockPriceForecastingTaskV2.apply_variance_correction``.

    ``pivot`` controls what point the rescale is centered around:
      - ``ensemble_mean`` (default, preserves per-window per-stock per-horizon
        ensemble mean → MAE strictly invariant per horizon; structural metrics
        only approximately scale-stable because the t-varying intercept
        breaks γ₀-normalization).
      - ``zero`` (multiplies samples by α around 0 in standardized space →
        global rescale; in destandardized space this is equivalent to
        pivoting around the per-stock long-run mean; structural metrics
        strictly invariant; per-horizon point forecast moves by (α-1)·μ̂
        which is small for daily log returns).
    """
    mode: str = "off"                                # "off" | "scalar" | "per_horizon"
    alpha: Optional[float] = None                    # required when mode == "scalar"
    alpha_per_horizon: Optional[List[float]] = None  # required when mode == "per_horizon"
    pivot: str = "ensemble_mean"                     # "ensemble_mean" | "zero"
    source: str = "default"                          # "default" | "cli" | "checkpoint_alpha_ema"


# ── Helpers ─────────────────────────────────────────────────────────────


def _find_experiment_dir(checkpoint_path: str) -> Path:
    """Walk up from *checkpoint_path* until ``.hydra/config.yaml`` is found.

    Handles common layouts::

        <experiment>/.hydra/config.yaml
        <experiment>/trainer_chkpts/<ckpt>.pt   (2 levels up)
        <experiment>/model_chkpts/<ckpt>.pt     (2 levels up)
        <experiment>/<ckpt>.pt                   (1 level up)
    """
    current = Path(checkpoint_path).resolve().parent
    for _ in range(5):
        if (current / ".hydra" / "config.yaml").is_file():
            return current
        current = current.parent
    raise FileNotFoundError(
        f"Could not locate .hydra/config.yaml in any parent of "
        f"{checkpoint_path!r}. Make sure the checkpoint lives inside (or "
        f"under) its Hydra experiment directory."
    )


def _load_experiment_config(experiment_dir: Path):
    """Load the saved experiment config and register custom OmegaConf resolvers.

    Mirrors the resolver-registration logic from ``test.py`` so that
    interpolations like ``${hydra:runtime.cwd}`` resolve properly outside
    the Hydra runtime.
    """
    config_path = experiment_dir / ".hydra" / "config.yaml"
    cfg = OmegaConf.load(str(config_path))

    # Register the ``hydra`` resolver (needed for ${hydra:runtime.cwd}).
    if not OmegaConf.has_resolver("hydra"):
        hydra_yaml = experiment_dir / ".hydra" / "hydra.yaml"
        if hydra_yaml.exists():
            hcfg = OmegaConf.load(str(hydra_yaml))
            project_root = str(hcfg.hydra.runtime.cwd)
        else:
            project_root = os.getcwd()
        OmegaConf.register_new_resolver(
            "hydra",
            lambda path, _root=project_root: _root if path == "runtime.cwd" else None,
            replace=False,
        )
        log.info("Registered hydra resolver (project root: %s)", project_root)

    return cfg


def _make_optimizer(trainer_cfg, params) -> torch.optim.Optimizer:
    """Build an optimizer from the saved ``trainer`` config block.

    Mirrors the logic in ``test.py`` / ``train.py``.  Only used for the
    ``DiffusionTrainer`` constructor (the optimizer is **not** stepped during
    evaluation).
    """
    def _trainable_params(param_iter):
        filtered = [p for p in param_iter if getattr(p, "requires_grad", False)]
        if not filtered:
            raise ValueError("No trainable parameters found for optimizer construction.")
        return filtered

    params = _trainable_params(params)

    def _cfg_get(cfg_obj, key, default):
        get_fn = getattr(cfg_obj, "get", None)
        if callable(get_fn):
            try:
                return get_fn(key, default)
            except Exception:
                return default
        if isinstance(cfg_obj, dict):
            return cfg_obj.get(key, default)
        return default

    def _has_key(cfg_obj, key):
        try:
            return key in cfg_obj
        except Exception:
            return isinstance(cfg_obj, dict) and key in cfg_obj

    opt_cfg = _cfg_get(trainer_cfg, "optimizer", "adam")
    if isinstance(opt_cfg, str) or opt_cfg is None:
        name = str(opt_cfg or "adam").lower()
        lr = float(_cfg_get(trainer_cfg, "learning_rate", 1e-3))
        raw_betas = _cfg_get(trainer_cfg, "betas", (0.9, 0.999))
        weight_decay = float(_cfg_get(trainer_cfg, "weight_decay", 0.0))
        momentum = float(_cfg_get(trainer_cfg, "momentum", 0.9))
    else:
        name = str(_cfg_get(opt_cfg, "name", "adam")).lower()
        lr = float(
            _cfg_get(
                trainer_cfg, "learning_rate",
                _cfg_get(opt_cfg, "learning_rate", 1e-3),
            ) if _has_key(trainer_cfg, "learning_rate")
            else _cfg_get(opt_cfg, "learning_rate", 1e-3)
        )
        raw_betas = (
            _cfg_get(trainer_cfg, "betas", _cfg_get(opt_cfg, "betas", (0.9, 0.999)))
            if _has_key(trainer_cfg, "betas")
            else _cfg_get(opt_cfg, "betas", (0.9, 0.999))
        )
        weight_decay = float(
            _cfg_get(
                trainer_cfg, "weight_decay",
                _cfg_get(opt_cfg, "weight_decay", 0.0),
            ) if _has_key(trainer_cfg, "weight_decay")
            else _cfg_get(opt_cfg, "weight_decay", 0.0)
        )
        momentum = float(
            _cfg_get(
                trainer_cfg, "momentum",
                _cfg_get(opt_cfg, "momentum", 0.9),
            ) if _has_key(trainer_cfg, "momentum")
            else _cfg_get(opt_cfg, "momentum", 0.9)
        )

    if weight_decay < 0.0:
        raise ValueError(f"optimizer weight_decay must be >= 0, got {weight_decay}")
    if momentum < 0.0:
        raise ValueError(f"optimizer momentum must be >= 0, got {momentum}")

    def _parse_betas(value):
        if value is None:
            value = (0.9, 0.999)
        if isinstance(value, str):
            raise ValueError(f"optimizer betas must be a 2-element sequence, got {value!r}")
        try:
            out = tuple(float(x) for x in value)
        except TypeError as e:
            raise ValueError("optimizer betas must be a 2-element sequence") from e
        if len(out) != 2:
            raise ValueError(f"optimizer betas must contain exactly 2 values, got {len(out)}")
        if not (0.0 <= out[0] < 1.0 and 0.0 <= out[1] < 1.0):
            raise ValueError(f"optimizer betas values must be in [0, 1), got {out}")
        return out

    if name == "adamw":
        return torch.optim.AdamW(
            params, lr=lr, betas=_parse_betas(raw_betas), weight_decay=weight_decay
        )
    if name == "adam":
        return torch.optim.Adam(
            params, lr=lr, betas=_parse_betas(raw_betas), weight_decay=weight_decay
        )
    if name == "sgd":
        return torch.optim.SGD(
            params, lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    raise ValueError(f"Unknown optimizer in saved config: {name!r}")


# ── Baseline ────────────────────────────────────────────────────────────


@BASELINE_REGISTRY.register("diffusion")
class DiffusionBaseline(BaseBaseline):
    """Wraps a trained diffusion model as a ``BaseBaseline``.

    Designed for **checkpoint-based evaluation**: all you need is the path to
    a ``.pt`` file produced by ``DiffusionTrainer``.  Model architecture,
    diffusion schedule, and trainer hyper-parameters are automatically
    recovered from the experiment's ``.hydra/config.yaml``.

    Two initialisation paths
    ~~~~~~~~~~~~~~~~~~~~~~~~

    1. **Lazy** (``evaluate.py``)::

           baseline = instantiate(cfg.baseline, device=device)   # shell
           baseline.load_checkpoint(cfg.checkpoint_path)          # auto-build

    2. **Eager** (``compare_baselines.py`` per-baseline config)::

           # baseline YAML includes  checkpoint_path: /path/...
           baseline = instantiate(baseline_cfg, device=device)    # build+load

    ``has_ensemble_evaluate = True`` tells ``UnifiedEvaluator`` to call
    ``self.evaluate()`` instead of per-batch ``self.predict()``.
    """

    has_ensemble_evaluate: bool = True
    needs_replicated_loader: bool = True       # uses ReplicatedDataset

    def __init__(
        self,
        device: Any,
        checkpoint_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(device=device, **kwargs)

        self.trainer: Optional[DiffusionTrainer] = None
        self._init_kwargs = kwargs            # stashed for deferred build
        self._experiment_cfg = None           # populated by _build_from_experiment

        # n_samples=None means no replication override (dataset.n_samples_per_input
        # keeps its default of 1 when there is no trainer section).
        # evaluate.py propagates this to dataset.n_samples_per_input before the
        # datamodule is built, so DiffusionBaseline itself never needs to act on it.
        self.n_samples: Optional[int] = kwargs.get("n_samples", None)

        # Optional dict of field-level overrides merged into the experiment's saved
        # diffusion config before instantiation.  Covers sampler class swap
        # (_target_), eta, sampling_timesteps, or any other diffusion field.
        self.diffusion_overrides: dict = kwargs.get("diffusion_overrides", {}) or {}

        # Evaluation settings (from nested YAML block or defaults)
        eval_cfg = dict(kwargs.get("evaluation", {}) or {})
        self.structural_metrics: bool = eval_cfg.get("structural_metrics", False)
        self.structural_max_lag: int = eval_cfg.get("structural_max_lag", 2)
        self.structural_n_eigenvalues: int = eval_cfg.get(
            "structural_n_eigenvalues", 5,
        )
        self.log_every_n_batches: int = eval_cfg.get("log_every_n_batches", 10)
        self.plot_dpi: int = eval_cfg.get("plot_dpi", 150)
        self.elbo_n_mc_samples: int = int(eval_cfg.get("elbo_n_mc_samples", 16))
        # When True, save raw plot-data sidecars (.npz) next to every
        # polished PDF so that figures can be regenerated by
        # ``python -m graph_signal_diffusion.cli.replot_baselines`` without
        # re-running evaluation.
        self.dump_plot_data: bool = bool(eval_cfg.get("dump_plot_data", True))

        # Variance correction (parsed eagerly; checkpoint-embedded alpha_ema
        # is layered on later inside _build_from_experiment).
        self.variance_correction: VarianceCorrectionConfig = (
            self._parse_variance_correction_config(
                kwargs.get("variance_correction", None)
            )
        )

        if checkpoint_path is not None:
            self._build_from_experiment(str(checkpoint_path))

    # ------------------------------------------------------------------
    # Variance-correction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_variance_correction_config(
        raw: Any,
    ) -> "VarianceCorrectionConfig":
        """Validate and materialize the variance_correction block.

        ``raw`` may be ``None`` (legacy / no CLI), a dict, or an OmegaConf
        DictConfig. ``alpha`` / ``alpha_per_horizon`` content is validated
        only when ``mode != off`` so callers without correction never trip
        validation errors.
        """
        if raw is None:
            return VarianceCorrectionConfig()
        if isinstance(raw, VarianceCorrectionConfig):
            return raw
        if hasattr(raw, "_content") or "DictConfig" in type(raw).__name__:
            # OmegaConf DictConfig
            raw = OmegaConf.to_container(raw, resolve=True)
        if not isinstance(raw, dict):
            raise ValueError(
                "baseline.variance_correction must be a mapping; got "
                f"{type(raw).__name__}."
            )
        mode_raw = raw.get("mode", "off")
        # YAML 1.1 maps unquoted off/on/yes/no to bool — coerce defensively.
        if isinstance(mode_raw, bool):
            mode = "off" if mode_raw is False else "on"
        else:
            mode = str(mode_raw or "off").lower()
        if mode not in ("off", "scalar", "per_horizon"):
            raise ValueError(
                "variance_correction.mode must be one of 'off', 'scalar', "
                f"'per_horizon'; got {mode!r}."
            )
        alpha = raw.get("alpha", None)
        alpha_per_horizon = raw.get("alpha_per_horizon", None)
        pivot_raw = raw.get("pivot", "ensemble_mean")
        pivot = str(pivot_raw or "ensemble_mean").lower()
        if pivot not in ("ensemble_mean", "zero"):
            raise ValueError(
                "variance_correction.pivot must be one of "
                f"'ensemble_mean', 'zero'; got {pivot!r}."
            )

        if mode == "scalar":
            if alpha is None:
                raise ValueError(
                    "variance_correction.mode='scalar' requires "
                    "variance_correction.alpha (float)."
                )
            alpha = float(alpha)
        elif mode == "per_horizon":
            if alpha_per_horizon is None:
                raise ValueError(
                    "variance_correction.mode='per_horizon' requires "
                    "variance_correction.alpha_per_horizon (list[float])."
                )
            try:
                alpha_per_horizon = [float(x) for x in alpha_per_horizon]
            except (TypeError, ValueError) as e:
                raise ValueError(
                    "variance_correction.alpha_per_horizon must be a "
                    "sequence of floats."
                ) from e
            if len(alpha_per_horizon) == 0:
                raise ValueError(
                    "variance_correction.alpha_per_horizon must be non-empty."
                )
        else:
            # mode == "off": ignore alpha / alpha_per_horizon
            alpha = None
            alpha_per_horizon = None

        return VarianceCorrectionConfig(
            mode=mode,
            alpha=alpha,
            alpha_per_horizon=alpha_per_horizon,
            pivot=pivot,
            source="cli" if mode != "off" else "default",
        )

    def _maybe_promote_alpha_ema(
        self,
        checkpoint_path: str,
    ) -> None:
        """Promote a checkpoint-embedded ``alpha_ema`` tensor to per_horizon mode.

        Only fires when ``self.variance_correction.mode == 'off'`` (CLI
        override always wins). The promoted tensor is treated as the
        ``alpha_per_horizon`` list. Missing key → no-op (pretrained
        checkpoints stay backward-compatible).
        """
        vc = self.variance_correction
        if vc.mode != "off":
            log.info(
                "Variance correction: CLI override active (mode=%s, "
                "source=cli); ignoring any checkpoint-embedded alpha_ema.",
                vc.mode,
            )
            return
        try:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
        except Exception as e:
            log.warning(
                "Variance correction: could not re-open checkpoint at %s "
                "to inspect alpha_ema (%s). Leaving mode=off.",
                checkpoint_path, e,
            )
            return
        if not isinstance(ckpt, dict):
            return
        alpha_ema = ckpt.get("alpha_ema", None)
        if alpha_ema is None:
            log.info(
                "Variance correction: no alpha_ema in checkpoint; "
                "mode remains 'off'."
            )
            return
        if isinstance(alpha_ema, torch.Tensor):
            values = alpha_ema.detach().flatten().tolist()
        else:
            try:
                values = [float(x) for x in alpha_ema]
            except (TypeError, ValueError):
                log.warning(
                    "Variance correction: checkpoint alpha_ema has "
                    "unexpected type %s; ignoring.",
                    type(alpha_ema).__name__,
                )
                return
        if not values:
            log.warning(
                "Variance correction: checkpoint alpha_ema is empty; "
                "ignoring."
            )
            return
        self.variance_correction = VarianceCorrectionConfig(
            mode="per_horizon",
            alpha=None,
            alpha_per_horizon=[float(x) for x in values],
            pivot=vc.pivot,   # preserve whatever pivot was configured (or default)
            source="checkpoint_alpha_ema",
        )
        log.info(
            "Variance correction: auto-promoted from checkpoint alpha_ema "
            "→ mode=per_horizon, alpha_per_horizon=%s.",
            ["%.4f" % v for v in values],
        )

    # ------------------------------------------------------------------
    # Internal: build the full pipeline from the experiment directory
    # ------------------------------------------------------------------

    def _build_from_experiment(self, checkpoint_path: str) -> None:
        """Discover experiment dir, rebuild model/diffusion, load weights."""
        experiment_dir = _find_experiment_dir(checkpoint_path)
        exp_cfg = _load_experiment_config(experiment_dir)
        self._experiment_cfg = exp_cfg

        print(f"Auto-discovered experiment: {experiment_dir}")
        print(f"  Model  : {exp_cfg.model.get('name', '?')}")
        diffusion_target = exp_cfg.diffusion.get("_target_", "?")
        print(f"  Diffusion: {diffusion_target.rsplit('.', 1)[-1]}")

        # ── Model ─────────────────────────────────────────────────────
        model = instantiate(exp_cfg.model)
        model.to(self.device)

        # ── Diffusion (inject model) ──────────────────────────────────
        diffusion_cfg = exp_cfg.diffusion
        if self.diffusion_overrides:
            from omegaconf import OmegaConf
            diffusion_cfg = OmegaConf.merge(diffusion_cfg, OmegaConf.create(self.diffusion_overrides))
            print(f"  Diffusion overrides applied: {dict(self.diffusion_overrides)}")
        diffusion = instantiate(diffusion_cfg, model=model)
        diffusion.to(self.device)

        # ── Optimizer (required by DiffusionTrainer; not stepped) ─────
        optimizer = _make_optimizer(exp_cfg.trainer, model.parameters())

        # ── DiffusionTrainer ──────────────────────────────────────────
        tcfg = exp_cfg.trainer
        self.trainer = DiffusionTrainer(
            diffusion=diffusion,
            optimizer=optimizer,
            device=self.device,
            save_dir=self._init_kwargs.get("save_dir"),
            log_every_n_steps=int(tcfg.get("log_every_n_steps", 50)),
            use_amp=self._init_kwargs.get(
                "use_amp", bool(tcfg.get("use_amp", False)),
            ),
            max_num_val_batches=int(tcfg.get("max_num_val_batches", 1)),
            max_num_train_val_batches=int(tcfg.get("max_num_train_val_batches", 1)),
            max_grad_norm=float(tcfg.get("max_grad_norm", 1.0)),
        )

        # ── Load weights ──────────────────────────────────────────────
        epoch = self.trainer.load_checkpoint(checkpoint_path)
        self.trainer.epoch = epoch
        print(f"  Loaded weights from epoch {epoch}")

        # ── Variance-correction: read-only promotion from checkpoint ──
        # If the checkpoint contains an `alpha_ema` key AND no CLI override
        # is active, promote to mode=per_horizon. Old checkpoints lack the
        # key → no-op; CLI override always wins.
        self._maybe_promote_alpha_ema(checkpoint_path)

        restored_temp = restore_selector_temperature_from_records(
            root_module=self.trainer.diffusion,
            checkpoint_path=checkpoint_path,
            checkpoint_epoch=epoch,
            experiment_dir=experiment_dir,
        )
        if restored_temp is not None and restored_temp.modules_updated > 0:
            print(
                "  Restored selector temperature "
                f"{restored_temp.temperature:.6f} from {restored_temp.source} "
                f"(modules={restored_temp.modules_updated})"
            )
        elif restored_temp is None:
            print("  Selector temperature records not found; using config default")

    def _require_trainer(self) -> DiffusionTrainer:
        if self.trainer is None:
            raise RuntimeError(
                "DiffusionBaseline: trainer not initialised. "
                "Call load_checkpoint(path) first, or pass checkpoint_path "
                "in the baseline config."
            )
        return self.trainer

    # ------------------------------------------------------------------
    # BaseBaseline interface
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        **kwargs,
    ):
        trainer = self._require_trainer()
        max_epochs = kwargs.get("max_epochs", 100)
        trainer.fit(train_loader, val_loader, max_epochs=max_epochs)

    def predict(self, data) -> torch.Tensor:
        """Single-sample prediction (fallback path).

        Returns tensor shaped ``[B*N, T, F]``.
        """
        trainer = self._require_trainer()
        trainer.diffusion.eval()
        with torch.inference_mode():
            real_samples = data.y  # [B*N, T, F]
            B = data.num_graphs
            N = real_samples.size(0) // B
            T = real_samples.size(1)
            F = real_samples.size(2)
            shape = (B, T, N, F)
            samples, _ = trainer._sample_with_selector_sampling_diagnostics(
                shape=shape, data=data,
            )
        return samples.view(B * N, T, F)

    def evaluate(
        self,
        loader: DataLoader,
        task,
        max_batches: Optional[int] = None,
        viz_save_dir: Optional[str] = None,
        eval_split_name: str = "eval",
    ) -> Dict[str, float]:
        """Ensemble-aware evaluation via ``DiffusionTrainer``.

        Called by ``UnifiedEvaluator.evaluate_baseline_on_split()`` when
        ``has_ensemble_evaluate`` is ``True``.

        The *task* passed in is the fully-configured instance from
        ``evaluate.py`` / ``compare_baselines.py``'s ``_setup_task()``
        (with SP500 destandardisation already injected).

        When ``self.structural_metrics`` is enabled, a ``batch_hook`` is
        registered that captures ``(real, gen, metadata)`` triples from
        each batch.  The hook extracts ``window_start_indices`` from
        ``metadata['timestamp']`` so the phased spectral estimator can
        group windows by ``start_index % T`` rather than positional order,
        making the computation correct on any DataLoader ordering (shuffled
        or strided test splits alike).
        """
        trainer = self._require_trainer()

        # Reset scoring state from any prior evaluate() call.
        self.last_nll_tensors = None
        self.last_structural_tensors = None
        self._cached_eval_data_batches: List[Data] = []

        # Structural diagnostics are for standardized return trajectories.
        # WRA/power-allocation tasks should not run these diagnostics.
        is_power_task = bool(getattr(task, "is_power_allocation_task", False))
        run_structural_metrics = bool(self.structural_metrics) and not is_power_task
        if self.structural_metrics and is_power_task:
            log.info(
                "[Diffusion] Skipping structural metrics: task '%s' is power-allocation.",
                getattr(task, "name", task.__class__.__name__),
            )

        # Replication is handled entirely by the datamodule (ReplicatedDataset).
        # evaluate.py propagates baseline.n_samples → dataset.n_samples_per_input
        # before the loaders are built, so nothing to do here.

        # Inject the caller's task so trainer can compute task metrics.
        trainer.task = task

        # Transient variance correction: set on the shared task instance
        # for the duration of this evaluate() only, then restore. GRW (or
        # any other baseline) never sets task.variance_correction, so its
        # samples flow through unchanged (the trainer-side gate stays
        # false). See variance_correction_plan_2026-05-20.md.
        vc = self.variance_correction
        _vc_active = vc is not None and vc.mode != "off"
        if _vc_active:
            _is_v2 = task.__class__.__name__ == "StockPriceForecastingTaskV2"
            if not _is_v2 or not hasattr(task, "apply_variance_correction"):
                log.warning(
                    "Variance correction configured (mode=%s, source=%s) "
                    "but task '%s' does not support correction (V2-only). "
                    "No-op for this run.",
                    vc.mode, vc.source, task.__class__.__name__,
                )
            else:
                log.info(
                    "Variance correction ACTIVE: mode=%s, pivot=%s, "
                    "source=%s, alpha=%s, alpha_per_horizon=%s.",
                    vc.mode, vc.pivot, vc.source, vc.alpha, vc.alpha_per_horizon,
                )
        _prev_vc = getattr(task, "variance_correction", None)
        if hasattr(task, "variance_correction") or _vc_active:
            task.variance_correction = vc if _vc_active else None

        # Temporarily redirect viz output to the per-split directory
        # chosen by UnifiedEvaluator.
        original_save_dir = trainer.save_dir
        if viz_save_dir is not None:
            trainer.save_dir = viz_save_dir

        # ------------------------------------------------------------------
        # Batch hook — always-on for power-mean accumulation;
        # structural-metrics path piggybacks when self.structural_metrics=True
        # ------------------------------------------------------------------
        all_real: List[torch.Tensor] = []
        all_gen: List[torch.Tensor] = []
        # Pooled full ensemble for stylized-fact computation:
        # [B*n_rep, T, N] per batch — every generated sample for every window.
        all_gen_full: List[torch.Tensor] = []
        all_start_indices: List[torch.Tensor] = []
        all_data_batches: List[Data] = []
        all_power_means_real: List[torch.Tensor] = []
        all_power_means_gen: List[torch.Tensor] = []

        # n_rep is needed for deduplication in both paths.
        n_rep = self._detect_n_rep(loader)

        def _hook(
            real: torch.Tensor,
            gen: torch.Tensor,
            metadata: Optional[Dict[str, Any]] = None,
            data: Optional[Data] = None,
        ) -> None:
            # Dedup replicated samples once — used by both paths below.
            r_dedup = real[::n_rep].detach().cpu()
            g_dedup = gen[::n_rep].detach().cpu()

            # ── Power-mean accumulation (WRA tasks only) ──────────────
            # Gated on task.is_power_allocation_task so that stock-
            # forecasting runs never accumulate or expose last_power_tensors.
            # real/gen: [B*n_rep, T, N, F]; dedup → [B, ...], mean over all
            # non-batch dims → scalar per sample.
            if getattr(task, "is_power_allocation_task", False):
                mean_dims = tuple(range(1, r_dedup.dim()))
                all_power_means_real.append(r_dedup.mean(dim=mean_dims))  # [B]
                all_power_means_gen.append(g_dedup.mean(dim=mean_dims))   # [B]

            # ── Structural metrics accumulation (conditional) ─────────
            if not run_structural_metrics:
                return

            r = r_dedup.squeeze(-1)   # drop F dim → [B, T, N]
            g = g_dedup.squeeze(-1)
            all_real.append(r)
            all_gen.append(g)
            # Full pooled ensemble (pre-dedup) for stylized-fact recompute.
            # gen is [B*n_rep, T, N, F] → squeeze F → [B*n_rep, T, N]
            all_gen_full.append(gen.squeeze(-1).detach().cpu())
            # Extract per-window start indices from timestamp
            if metadata is not None:
                ts = metadata.get("timestamp", None)
                if ts is not None:
                    N_nodes = r.shape[2]  # number of stocks / nodes
                    # timestamp is [B*n_rep*N]; take every (n_rep*N)-th
                    idx = ts[::n_rep * N_nodes].detach().cpu().long()
                    all_start_indices.append(idx)
            # Cache deduped data for ELBO scoring. Always wrap in Batch (even
            # for single-graph dedup): downstream compute_elbo_per_trajectory /
            # compute_nll access .num_graphs, a Batch-only attribute. Raw Data
            # here triggered AttributeError under n_rep == batch_size_val on a
            # ReplicatedDataset (e.g. n_samples=50, batch_size_val=50 → exactly
            # one unique window per batch after dedup).
            if data is not None:
                data_list = data.to_data_list() if hasattr(data, "to_data_list") else [data]
                dedup_list = data_list[::max(int(n_rep), 1)]
                if dedup_list:
                    all_data_batches.append(
                        Batch.from_data_list(dedup_list).detach().cpu()
                    )

        batch_hook = _hook

        # Record sink for WRA tasks (v1 path only) — collects per-batch
        # evaluation records across the entire evaluate() loop so they
        # survive across multiple task.evaluate_samples() calls.
        wra_records: list = []
        is_wra = getattr(task, "is_power_allocation_task", False)

        try:
            num_max = -1 if max_batches is None else max_batches

            if trainer._is_task_v2(task):
                metrics = trainer.evaluate_v2(
                    loader,
                    eval_split=eval_split_name,
                    force_task_eval=True,
                    collect_heavy_selector_diagnostics=False,
                    num_max_eval_batches=num_max,
                    batch_hook=batch_hook,
                )
            else:
                metrics = trainer.evaluate(
                    loader,
                    eval_split=eval_split_name,
                    force_task_eval=True,
                    collect_heavy_selector_diagnostics=False,
                    num_max_eval_batches=num_max,
                    batch_hook=batch_hook,
                    record_sink=wra_records if is_wra else None,
                )
        finally:
            trainer.save_dir = original_save_dir
            # Restore previous variance_correction on the (shared) task so
            # subsequent baselines (GRW, etc.) are not affected.
            if hasattr(task, "variance_correction"):
                task.variance_correction = _prev_vc

        if run_structural_metrics and all_real:
            cat_indices = (
                torch.cat(all_start_indices, dim=0)
                if all_start_indices else None
            )
            metrics.update(
                self._compute_structural_metrics_from_tensors(
                    all_real, all_gen, viz_save_dir,
                    window_start_indices=cat_indices,
                    all_gen_full=all_gen_full,
                )
            )

            # Store cached data batches for cross-scoring
            self._cached_eval_data_batches = all_data_batches

            # Self-scoring: compute diffusion NLL proxy for real/gen
            if all_data_batches:
                try:
                    self._compute_elbo_nll_tensors(all_real, all_gen, all_data_batches)
                except Exception:
                    log.warning(
                        "Diffusion ELBO self-scoring failed; skipping NLL proxy.\n%s",
                        traceback.format_exc(),
                    )
                    self.last_nll_tensors = None

                # Surface NLL summaries to the metrics row so they appear in
                # baseline_comparison.csv alongside GRW's same-named columns.
                # nll_mean_real here is: NLL of real test data under the
                # diffusion ELBO proxy (lower ⇒ diffusion fits reality better).
                # nll_w1: Wasserstein-1 between the per-trajectory NLL
                # distributions for real vs generated samples (lower ⇒ samples
                # are more reality-like under the diffusion's own likelihood).
                if isinstance(self.last_nll_tensors, dict):
                    try:
                        from scipy.stats import wasserstein_distance
                        import numpy as np
                        nll_real_np = self.last_nll_tensors.get("nll_real")
                        nll_gen_np = self.last_nll_tensors.get("nll_gen")
                        if (
                            nll_real_np is not None
                            and nll_gen_np is not None
                            and len(nll_real_np) > 0
                            and len(nll_gen_np) > 0
                        ):
                            metrics["nll_mean_real"] = float(np.mean(nll_real_np))
                            metrics["nll_mean_gen"] = float(np.mean(nll_gen_np))
                            metrics["nll_std_real"] = float(np.std(nll_real_np))
                            metrics["nll_std_gen"] = float(np.std(nll_gen_np))
                            metrics["nll_w1"] = float(
                                wasserstein_distance(nll_real_np, nll_gen_np)
                            )
                    except Exception:
                        log.warning(
                            "Failed to compute NLL summary metrics from "
                            "last_nll_tensors:\n%s",
                            traceback.format_exc(),
                        )

        # ── Power-mean tensors (for UnifiedEvaluator power histogram) ─
        if all_power_means_real:
            self.last_power_tensors: dict = {
                "real": torch.cat(all_power_means_real).numpy(),
                "gen":  torch.cat(all_power_means_gen).numpy(),
            }

        # ── WRA evaluation records (for cross-baseline percentile plot) ─
        if wra_records:
            self.last_wra_evaluation_records = wra_records

        return metrics

    # ------------------------------------------------------------------
    # Structural metrics helpers
    # ------------------------------------------------------------------

    def _detect_n_rep(self, loader: DataLoader) -> int:
        """Return the replication stride from a ``ReplicatedDataset``, or 1."""
        raw_ds = getattr(loader, "dataset", None)
        try:
            from graph_signal_diffusion.datasets.replicated_dataset import (
                ReplicatedDataset,
            )
            if isinstance(raw_ds, ReplicatedDataset):
                return int(getattr(raw_ds, "n_replicas", 1))
        except ImportError:
            pass
        return 1

    def _compute_structural_metrics_from_tensors(
        self,
        all_real: List[torch.Tensor],
        all_gen: List[torch.Tensor],
        viz_save_dir: Optional[str] = None,
        window_start_indices: Optional[torch.Tensor] = None,
        all_gen_full: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, float]:
        """Compute structural metrics and optional plots from accumulated tensors.

        Args:
            all_real: List of ``[B_eff, T, N]`` real-return tensors (CPU).
            all_gen:  Corresponding list of generated-return tensors,
                deduped to one sample per window (used by the structural
                metrics module which expects 1-to-1 alignment with real).
            viz_save_dir: Directory to write PDF plots into, or ``None``.
            window_start_indices: ``[B_total]`` integer tensor of window
                start positions, or ``None``.
            all_gen_full: Optional list of ``[B_eff*n_rep, T, N]`` generated
                tensors with the full ensemble pooled.  When provided, the
                non-linear stylized facts (kurtosis / vol-clustering /
                momentum / top-eigval) are recomputed on this pooled tensor
                rather than the deduped ``all_gen``, giving a much
                lower-variance estimate of the model's generated
                distribution.

        Returns:
            Flat dict of structural metric scalars.
        """
        import numpy as np

        cat_real = torch.cat(all_real, dim=0)  # [B_total, T, N]
        cat_gen = torch.cat(all_gen, dim=0)
        T_steps = cat_real.shape[1]

        # ---- Scalar metrics ------------------------------------------------
        struct = sm.compute_structural_metrics(
            cat_real, cat_gen,
            max_lag=self.structural_max_lag,
            n_eigenvalues=self.structural_n_eigenvalues,
            window_stride=T_steps,
            window_start_indices=window_start_indices,
        )

        # ---- Recompute non-linear stylized facts on full tensors -----------
        # The trainer.evaluate_v2 loop averages per-batch kurtosis_real /
        # volclustering_real / momentum_real, which is biased for 4th-moment
        # and ratio-style statistics.  Computing them once over the full
        # accumulated tensors gives the unbiased dataset-level value and
        # makes cross-baseline comparison meaningful.
        #
        # cat_real has one trajectory per window; gen uses the pooled full
        # ensemble (all_gen_full) when available for a much lower-variance
        # estimate of the model's distribution, falling back to the deduped
        # cat_gen otherwise.  These keys overwrite the per-batch averages
        # already in `struct`.
        try:
            from graph_signal_diffusion.tasks.stock_price_forecasting.evaluator import (
                StockPriceForecastingTaskV2,
            )
            cat_gen_for_stats = (
                torch.cat(all_gen_full, dim=0) if all_gen_full else cat_gen
            )
            struct.update(
                StockPriceForecastingTaskV2.compute_stylized_facts_from_tensors(
                    cat_real, cat_gen_for_stats,
                )
            )
        except Exception:
            log.warning(
                "[Diffusion] Failed to recompute stylized facts on full "
                "tensors; keeping per-batch averages.\n%s",
                traceback.format_exc(),
            )

        # Expose tensors for cross-baseline comparative plots
        self.last_structural_tensors = {
            "real": cat_real,
            "gen": cat_gen,
            "window_start_indices": window_start_indices,
            "window_stride": T_steps,
        }

        print(
            f"[Diffusion] Structural metrics computed "
            f"({cat_real.shape[0]} trajectories, "
            f"T={cat_real.shape[1]}, N={cat_real.shape[2]}, "
            f"window_stride={T_steps})"
        )

        # ---- Visualisation --------------------------------------------------
        if viz_save_dir is not None:
            sm.plot_structural_metrics(
                cat_real, cat_gen,
                save_path=os.path.join(
                    viz_save_dir, "structural_metrics_diffusion.pdf",
                ),
                max_lag=self.structural_max_lag,
                n_eigenvalues=self.structural_n_eigenvalues,
                dpi=self.plot_dpi,
                model_name="Diffusion",
                window_stride=T_steps,
                window_start_indices=window_start_indices,
            )
            sm.plot_autocorr_per_stock(
                cat_real, cat_gen,
                save_path=os.path.join(
                    viz_save_dir, "autocorr_per_stock_diffusion.pdf",
                ),
                max_lag=self.structural_max_lag,
                dpi=self.plot_dpi,
                model_name="Diffusion",
                window_stride=T_steps,
                window_start_indices=window_start_indices,
            )
            sm.plot_autocorr_violin(
                cat_real, cat_gen,
                save_path=os.path.join(
                    viz_save_dir, "autocorr_violin_diffusion.pdf",
                ),
                max_lag=self.structural_max_lag,
                dpi=self.plot_dpi,
                model_name="Diffusion",
                window_stride=T_steps,
                window_start_indices=window_start_indices,
            )
            sm.plot_autocorr_heatmap(
                cat_gen,
                save_path=os.path.join(
                    viz_save_dir, "autocorr_heatmap_diffusion.pdf",
                ),
                max_lag=self.structural_max_lag,
                dpi=self.plot_dpi,
                model_name="Diffusion",
                window_stride=T_steps,
                window_start_indices=window_start_indices,
                real_returns=cat_real,
            )
            # Cumulative-return std vs horizon, with GRW theory overlay
            T_steps = cat_real.shape[1]
            # sigma_bar from real single-step returns (model-agnostic)
            sigma_bar = float(
                cat_real.std(dim=(0, 1)).mean().item()
            )
            theoretical = sigma_bar * np.sqrt(np.arange(1, T_steps + 1))
            _cum_std_real = sm.compute_cumulative_return_std(cat_real).numpy()
            _cum_std_gen = sm.compute_cumulative_return_std(cat_gen).numpy()
            _cum_std_real_tn = sm.compute_cumulative_return_std(
                cat_real, per_stock=True,
            ).numpy()
            _cum_std_gen_tn = sm.compute_cumulative_return_std(
                cat_gen, per_stock=True,
            ).numpy()
            sm.plot_cumulative_return_std(
                _cum_std_real, _cum_std_gen,
                save_path=os.path.join(
                    viz_save_dir, "cumulative_return_std_diffusion.pdf",
                ),
                theoretical=theoretical,
                theoretical_label=r"i.i.d. theory  $\bar{\sigma}\sqrt{t}$",
                std_real_tn=_cum_std_real_tn,
                std_gen_tn=_cum_std_gen_tn,
                dpi=self.plot_dpi,
                model_name="Diffusion",
            )

            # Dump consolidated plot-data sidecar for replot_baselines.
            if self.dump_plot_data:
                try:
                    from graph_signal_diffusion.evaluation.plot_data_io import (
                        dump_per_baseline_structural,
                    )
                    dump_per_baseline_structural(
                        os.path.join(viz_save_dir, "structural_data.npz"),
                        cat_real=cat_real,
                        cat_gen=cat_gen,
                        model_name="Diffusion",
                        max_lag=self.structural_max_lag,
                        n_eigenvalues=self.structural_n_eigenvalues,
                        window_stride=T_steps,
                        window_start_indices=window_start_indices,
                        cum_std_real=_cum_std_real,
                        cum_std_gen=_cum_std_gen,
                        cum_std_real_tn=_cum_std_real_tn,
                        cum_std_gen_tn=_cum_std_gen_tn,
                        theoretical=theoretical,
                        theoretical_label=(
                            r"i.i.d. theory  $\bar{\sigma}\sqrt{t}$"
                        ),
                        dpi=self.plot_dpi,
                    )
                except Exception:
                    log.warning(
                        "[Diffusion] Failed to dump structural plot data sidecar:\n%s",
                        traceback.format_exc(),
                    )

        return struct

    # ------------------------------------------------------------------
    # Diffusion NLL proxy scoring
    # ------------------------------------------------------------------

    def _compute_elbo_nll_tensors(
        self,
        all_real: List[torch.Tensor],
        all_gen: List[torch.Tensor],
        all_data_batches: List[Data],
    ) -> None:
        """Score real/gen trajectories under the diffusion NLL proxy.

        Populates ``self.last_nll_tensors`` with numpy arrays on success.
        """
        import numpy as np

        trainer = self._require_trainer()
        proxy_real_parts: List[torch.Tensor] = []
        proxy_gen_parts: List[torch.Tensor] = []

        for real_batch, gen_batch, data_batch in zip(
            all_real, all_gen, all_data_batches,
        ):
            B_eff = real_batch.shape[0]
            T = real_batch.shape[1]
            N = real_batch.shape[2]

            for returns, accum in [
                (real_batch, proxy_real_parts),
                (gen_batch, proxy_gen_parts),
            ]:
                data_scored = data_batch.clone()
                # Reshape [B_eff, T, N] → [B_eff*N, T, 1] to match DDPM
                # data.y convention (B*N nodes, T timesteps, F=1 feature).
                data_scored.y = (
                    returns.unsqueeze(-1)             # [B, T, N, 1]
                    .permute(0, 2, 1, 3)              # [B, N, T, 1]
                    .reshape(B_eff * N, T, 1)
                )
                data_scored = data_scored.to(self.device)
                scores = trainer.diffusion.compute_elbo_per_trajectory(
                    data_scored, n_mc_samples=self.elbo_n_mc_samples,
                )
                accum.append(scores.detach().cpu())

        proxy_real = torch.cat(proxy_real_parts, dim=0).numpy()
        proxy_gen = torch.cat(proxy_gen_parts, dim=0).numpy()

        self.last_nll_tensors = {
            "nll_real": proxy_real,
            "nll_gen": proxy_gen,
        }
        log.info(
            "[Diffusion] ELBO NLL proxy computed: %d real, %d gen trajectories",
            len(proxy_real), len(proxy_gen),
        )

    def compute_nll(self, returns: torch.Tensor) -> torch.Tensor:
        """Score arbitrary ``[B, T, N]`` returns under the diffusion NLL proxy.

        Iterates over cached data batches in order so that the conditioning
        context (``data.x``) and graph structure (``data.edge_index``) for
        each chunk come from the correspondingly-positioned cached batch,
        matching the same time windows evaluated during
        ``DiffusionBaseline.evaluate()``. This is correct when the
        cross-scored baseline (e.g. GRW) evaluates over the same unshuffled
        test loader in the same window order.

        Cached batches may have varying sizes — the final (tail) batch is
        commonly smaller when the dataset length is not divisible by
        ``batch_size_val``. We walk each cached batch by its own
        ``num_graphs`` rather than assuming a uniform template size.

        Returns a CPU tensor ``[B]``.
        """
        if not getattr(self, "_cached_eval_data_batches", None):
            raise RuntimeError(
                "compute_nll requires cached evaluation data. "
                "Run evaluate() with structural_metrics=true first."
            )

        if returns.ndim == 4:
            returns = returns.squeeze(-1)  # [B, T, N, 1] → [B, T, N]
        B = returns.shape[0]
        T = returns.shape[1]
        N = returns.shape[2]

        cached_batches = self._cached_eval_data_batches
        cached_total = sum(int(b.num_graphs) for b in cached_batches)
        if cached_total != B:
            log.warning(
                "compute_nll: cached-batch total (%d) differs from returns B=%d; "
                "aligning what we can by truncating to the smaller of the two.",
                cached_total, B,
            )

        try:
            trainer = self._require_trainer()
            parts: List[torch.Tensor] = []
            pos = 0
            for cached_batch in cached_batches:
                if pos >= B:
                    break
                b_cached = int(cached_batch.num_graphs)
                take = min(b_cached, B - pos)
                data_scored = cached_batch.clone()
                if take < b_cached:
                    data_list = data_scored.to_data_list()[:take]
                    data_scored = (
                        Batch.from_data_list(data_list)
                        if take > 1 else data_list[0]
                    )
                chunk = returns[pos : pos + take]
                data_scored.y = (
                    chunk.unsqueeze(-1)            # [take, T, N, 1]
                    .permute(0, 2, 1, 3)           # [take, N, T, 1]
                    .reshape(take * N, T, 1)
                )
                data_scored = data_scored.to(self.device)
                scores = trainer.diffusion.compute_elbo_per_trajectory(
                    data_scored, n_mc_samples=self.elbo_n_mc_samples,
                )
                parts.append(scores.detach().cpu())
                pos += take

            if pos < B:
                log.warning(
                    "compute_nll: only scored %d of %d trajectories (cached "
                    "batches exhausted); padding remainder with NaN.",
                    pos, B,
                )
                parts.append(torch.full((B - pos,), float("nan")))

            return torch.cat(parts, dim=0)
        except Exception:
            log.warning(
                "Diffusion ELBO cross-scoring failed.\n%s",
                traceback.format_exc(),
            )
            # Return NaN tensor so caller can detect failure
            return torch.full((B,), float("nan"))

    def save_checkpoint(self, path: str):
        if self.trainer is not None:
            self.trainer.save_checkpoint(path)

    def load_checkpoint(self, path: str):
        """Load a checkpoint, auto-building the pipeline if necessary.

        If the trainer has not been initialised yet (lazy path), this
        discovers the experiment directory, reads the saved Hydra config,
        constructs model / diffusion / optimizer, and loads the weights.

        If the trainer already exists (eager path / repeated call), this
        simply reloads weights from *path*.
        """
        if self.trainer is None:
            self._build_from_experiment(path)
        else:
            epoch = self.trainer.load_checkpoint(path)
            self.trainer.epoch = epoch
            restored_temp = restore_selector_temperature_from_records(
                root_module=self.trainer.diffusion,
                checkpoint_path=path,
                checkpoint_epoch=epoch,
            )
            if restored_temp is not None and restored_temp.modules_updated > 0:
                log.info(
                    "Restored selector temperature %.6f from %s (modules=%d)",
                    restored_temp.temperature,
                    restored_temp.source,
                    restored_temp.modules_updated,
                )
