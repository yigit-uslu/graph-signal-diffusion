# src/graph_signal_diffusion/cli/train.py
from __future__ import annotations

import contextlib
import math
import logging
import os
import random
import signal
from typing import Any, Optional, Tuple

try:
    import coolname
    _HAS_COOLNAME = True
except ModuleNotFoundError:
    _HAS_COOLNAME = False
    
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
import jaxtyping
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

from graph_signal_diffusion.datasets import (
    discover_datasets,
    get_dataset_builder,
    DATASET_REGISTRY,
)

from graph_signal_diffusion.tasks import (
    discover_tasks,
    TASK_REGISTRY,
)

from graph_signal_diffusion.models import (
    discover_models,
    MODEL_REGISTRY,
)

from graph_signal_diffusion.trainers import DiffusionTrainer


# ── Custom OmegaConf resolver for run naming ──────────────────────────────
_CACHED_RUN_NAME: str | None = None


def _gen_run_name_resolver(run_name: Any) -> str:
    """Return *run_name* when explicitly set, or generate a coolname slug.

    The result is cached so every ``${gen_run_name:...}`` reference inside the
    same process resolves to the **same** value (Hydra may resolve the
    interpolation more than once).
    """
    global _CACHED_RUN_NAME

    # OmegaConf passes the MISSING sentinel or None when run_name is null
    if run_name is not None and str(run_name) not in ("", "null", "None"):
        return str(run_name)

    if _CACHED_RUN_NAME is not None:
        return _CACHED_RUN_NAME

    if _HAS_COOLNAME:
        slug = coolname.generate_slug(2)
        num = random.randint(1, 999)
        _CACHED_RUN_NAME = f"{slug}-{num}"
    else:
        import uuid
        _CACHED_RUN_NAME = uuid.uuid4().hex[:8]

    return _CACHED_RUN_NAME


OmegaConf.register_new_resolver("gen_run_name", _gen_run_name_resolver, replace=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Optional determinism (can hurt performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _pretty_known(registry: dict) -> str:
    """Format registry keys for display (e.g. 'cifar10 / sp500 / wra')."""
    try:
        return " / ".join(registry.keys())
    except Exception:
        return "<unavailable>"
    

def _make_optimizer(trainer_cfg, params):
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

    opt_cfg = trainer_cfg.get("optimizer", "adam")
    if isinstance(opt_cfg, str) or opt_cfg is None:
        # Legacy flat schema:
        #   trainer.optimizer: adam
        #   trainer.learning_rate: 1e-3
        #   trainer.betas: [0.9, 0.999]
        #   trainer.weight_decay: 0.0
        #   trainer.momentum: 0.9
        name = str(opt_cfg or "adam").lower()
        lr = float(_cfg_get(trainer_cfg, "learning_rate", 1e-3))
        raw_betas = _cfg_get(trainer_cfg, "betas", (0.9, 0.999))
        weight_decay = float(_cfg_get(trainer_cfg, "weight_decay", 0.0))
        momentum = float(_cfg_get(trainer_cfg, "momentum", 0.9))
    else:
        # Nested schema:
        #   trainer.optimizer:
        #     name: adam
        #     learning_rate: 1e-3
        #     betas: [0.9, 0.999]
        #     weight_decay: 0.0
        #     momentum: 0.9
        name = str(_cfg_get(opt_cfg, "name", "adam")).lower()
        # Compatibility rule: if legacy flat keys are explicitly provided,
        # they override nested values (so existing scripts keep working).
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
        raise ValueError(
            "optimizer weight_decay must be >= 0 "
            f"(trainer.optimizer.weight_decay or legacy trainer.weight_decay), got {weight_decay}"
        )
    if momentum < 0.0:
        raise ValueError(
            "optimizer momentum must be >= 0 "
            f"(trainer.optimizer.momentum or legacy trainer.momentum), got {momentum}"
        )

    def _parse_betas(value):
        if value is None:
            value = (0.9, 0.999)
        if isinstance(value, str):
            raise ValueError(
                "optimizer betas must be a 2-element list/tuple of floats "
                "(trainer.optimizer.betas or legacy trainer.betas), "
                f"got string: {value!r}"
            )
        try:
            out = tuple(float(x) for x in value)
        except TypeError as e:
            raise ValueError(
                "optimizer betas must be a 2-element list/tuple of floats "
                "(trainer.optimizer.betas or legacy trainer.betas)"
            ) from e
        if len(out) != 2:
            raise ValueError(
                "optimizer betas must contain exactly 2 values "
                f"(trainer.optimizer.betas or legacy trainer.betas), got {len(out)}"
            )
        if not (0.0 <= out[0] < 1.0 and 0.0 <= out[1] < 1.0):
            raise ValueError(
                "optimizer betas values must be in [0, 1) "
                f"(trainer.optimizer.betas or legacy trainer.betas), got {out}"
            )
        return out

    if name == "adam":
        betas = _parse_betas(raw_betas)
        return torch.optim.Adam(
            params, lr=lr, betas=betas, weight_decay=weight_decay
        )
    elif name == "adamw":
        betas = _parse_betas(raw_betas)
        return torch.optim.AdamW(
            params, lr=lr, betas=betas, weight_decay=weight_decay
        )
    elif name == "sgd":
        return torch.optim.SGD(
            params, lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def _make_lr_scheduler(
    trainer_cfg,
    optimizer: torch.optim.Optimizer,
    total_steps: int,
) -> Tuple[Optional[torch.optim.lr_scheduler.LambdaLR], Optional[dict]]:
    scheduler_cfg = trainer_cfg.get("lr_scheduler", {})
    enabled = bool(getattr(scheduler_cfg, "get", lambda k, d=None: d)("enabled", False))
    if not enabled:
        return None, None

    _SUPPORTED = ("cosine", "linear", "step")
    name = str(getattr(scheduler_cfg, "get", lambda k, d=None: d)("name", "cosine")).lower()
    if name not in _SUPPORTED:
        raise ValueError(
            f"Unknown lr scheduler '{name}'. Supported scheduler names: {' / '.join(_SUPPORTED)}"
        )

    if total_steps <= 0:
        raise ValueError(
            f"total_steps must be > 0 when lr_scheduler is enabled, got {total_steps}"
        )

    # ── Shared params ─────────────────────────────────────────────────────
    def _sget(k, d=None):
        return getattr(scheduler_cfg, "get", lambda k, d=None: d)(k, d)

    warmup_ratio = float(_sget("warmup_ratio", 0.01))
    if not (0.0 <= warmup_ratio < 1.0):
        raise ValueError(
            f"trainer.lr_scheduler.warmup_ratio must be in [0, 1), got {warmup_ratio}"
        )

    min_lr_ratio = float(_sget("min_lr_ratio", 0.05))
    if not (0.0 <= min_lr_ratio <= 1.0):
        raise ValueError(
            f"trainer.lr_scheduler.min_lr_ratio must be in [0, 1], got {min_lr_ratio}"
        )

    warmup_steps = max(1, int(total_steps * warmup_ratio))
    if warmup_steps >= total_steps:
        warmup_steps = max(1, total_steps - 1)

    # ── Scheduler-specific lr_lambda ──────────────────────────────────────
    schedule_meta: dict = {
        "name": name,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "warmup_ratio": warmup_ratio,
        "min_lr_ratio": min_lr_ratio,
    }

    if name == "cosine":
        # Linear warmup then cosine decay from 1.0 → min_lr_ratio, completing at
        # `decay_fraction` of total_steps.  Steps beyond that hold at min_lr_ratio.
        # decay_fraction=1.0 (default) gives the original behaviour.
        decay_fraction = float(_sget("decay_fraction", 1.0))
        if not (0.0 < decay_fraction <= 1.0):
            raise ValueError(
                f"trainer.lr_scheduler.decay_fraction must be in (0, 1], got {decay_fraction}"
            )
        # Effective end step for the cosine curve (absolute step index)
        decay_end_step = max(warmup_steps + 1, int(total_steps * decay_fraction))
        decay_steps = max(decay_end_step - warmup_steps, 1)
        schedule_meta["decay_fraction"] = decay_fraction
        schedule_meta["decay_end_step"] = decay_end_step

        def lr_lambda(step_idx: int) -> float:
            if step_idx < warmup_steps:
                return float(step_idx + 1) / float(warmup_steps)
            progress = min(1.0, float(step_idx - warmup_steps) / float(decay_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return float(min_lr_ratio + (1.0 - min_lr_ratio) * cosine)

    elif name == "linear":
        # Linear warmup then linear decay from 1.0 → min_lr_ratio, completing at
        # `decay_fraction` of total_steps.  Steps beyond that hold at min_lr_ratio.
        # decay_fraction=1.0 (default) gives the original behaviour.
        decay_fraction = float(_sget("decay_fraction", 1.0))
        if not (0.0 < decay_fraction <= 1.0):
            raise ValueError(
                f"trainer.lr_scheduler.decay_fraction must be in (0, 1], got {decay_fraction}"
            )
        decay_end_step = max(warmup_steps + 1, int(total_steps * decay_fraction))
        decay_steps = max(decay_end_step - warmup_steps, 1)
        schedule_meta["decay_fraction"] = decay_fraction
        schedule_meta["decay_end_step"] = decay_end_step

        def lr_lambda(step_idx: int) -> float:
            if step_idx < warmup_steps:
                return float(step_idx + 1) / float(warmup_steps)
            progress = min(1.0, float(step_idx - warmup_steps) / float(decay_steps))
            return float(min_lr_ratio + (1.0 - min_lr_ratio) * (1.0 - progress))

    elif name == "step":
        # Linear warmup then multiplicative drops every `step_size` optimizer steps.
        # min_lr_ratio acts as a hard floor (default 0.0 for step_decay).
        step_size_ratio = float(_sget("step_size_ratio", 0.1))
        if not (0.0 < step_size_ratio <= 1.0):
            raise ValueError(
                f"trainer.lr_scheduler.step_size_ratio must be in (0, 1], got {step_size_ratio}"
            )
        gamma = float(_sget("gamma", 0.5))
        if not (0.0 < gamma <= 1.0):
            raise ValueError(
                f"trainer.lr_scheduler.gamma must be in (0, 1], got {gamma}"
            )

        _decay_steps = max(total_steps - warmup_steps, 1)
        step_size = max(1, int(_decay_steps * step_size_ratio))
        num_drops = _decay_steps // step_size  # informational

        def lr_lambda(step_idx: int) -> float:
            if step_idx < warmup_steps:
                return float(step_idx + 1) / float(warmup_steps)
            steps_since_warmup = step_idx - warmup_steps
            drop_count = steps_since_warmup // step_size
            scale = gamma ** drop_count
            return float(max(min_lr_ratio, scale))

        schedule_meta.update({
            "step_size_ratio": step_size_ratio,
            "step_size": step_size,
            "gamma": gamma,
            "num_drops": num_drops,
        })

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Ensure the first optimizer step uses the warmup-start LR.
    initial_scale = lr_lambda(0)
    for param_group, base_lr in zip(optimizer.param_groups, scheduler.base_lrs):
        param_group["lr"] = float(base_lr) * initial_scale

    schedule_meta["initial_lr_scale"] = initial_scale
    return scheduler, schedule_meta


def _resolve_selector_temperature_schedule(
    cfg: DictConfig,
    total_steps: int,
    logger: Optional[logging.Logger] = None,
) -> Optional[dict]:
    """
    Resolve selector temperature warmup/anneal steps from trainer schedule config.

    Writes resolved integers into:
      model.config.pooling_config.selector_kwargs.temperature_warmup_steps
      model.config.pooling_config.selector_kwargs.temperature_anneal_steps

    Logs the outcome when *logger* is provided.
    """
    schedule_cfg = cfg.trainer.get("selector_temperature_schedule", {})
    enabled = bool(getattr(schedule_cfg, "get", lambda k, d=None: d)("enabled", False))
    if not enabled:
        return None

    if total_steps <= 0:
        raise ValueError(
            f"total_steps must be > 0 when selector_temperature_schedule is enabled, got {total_steps}"
        )

    selection_method = str(
        OmegaConf.select(cfg, "model.config.pooling_config.selection_method", default="stride")
    ).lower()
    if selection_method != "learned":
        reason = f"selection_method='{selection_method}' (requires 'learned')"
        if logger is not None:
            logger.info("Selector temperature schedule enabled but not applied: %s", reason)
        return {
            "enabled": True,
            "applied": False,
            "reason": reason,
            "total_steps": total_steps,
        }

    warmup_steps_override = getattr(schedule_cfg, "get", lambda k, d=None: d)("warmup_steps", None)
    anneal_steps_override = getattr(schedule_cfg, "get", lambda k, d=None: d)("anneal_steps", None)

    warmup_ratio = float(getattr(schedule_cfg, "get", lambda k, d=None: d)("warmup_ratio", 0.01))
    anneal_ratio = float(getattr(schedule_cfg, "get", lambda k, d=None: d)("anneal_ratio", 0.40))
    if not (0.0 <= warmup_ratio <= 1.0):
        raise ValueError(
            f"trainer.selector_temperature_schedule.warmup_ratio must be in [0, 1], got {warmup_ratio}"
        )
    if not (0.0 <= anneal_ratio <= 1.0):
        raise ValueError(
            f"trainer.selector_temperature_schedule.anneal_ratio must be in [0, 1], got {anneal_ratio}"
        )

    def _resolve_steps(
        *,
        name: str,
        override: Optional[Any],
        ratio: float,
        total_steps: int,
    ) -> Tuple[int, str]:
        if override is not None:
            resolved = int(override)
            if resolved < 0:
                raise ValueError(
                    f"trainer.selector_temperature_schedule.{name} must be >= 0, got {resolved}"
                )
            return resolved, "explicit"
        return max(0, int(total_steps * ratio)), "ratio"

    warmup_steps, warmup_source = _resolve_steps(
        name="warmup_steps",
        override=warmup_steps_override,
        ratio=warmup_ratio,
        total_steps=total_steps,
    )
    anneal_steps, anneal_source = _resolve_steps(
        name="anneal_steps",
        override=anneal_steps_override,
        ratio=anneal_ratio,
        total_steps=total_steps,
    )

    selector_kwargs_path = "model.config.pooling_config.selector_kwargs"
    selector_kwargs = OmegaConf.to_container(
        OmegaConf.select(cfg, selector_kwargs_path),
        resolve=False,
        throw_on_missing=False,
    )
    if selector_kwargs is None:
        selector_kwargs = {}
    if not isinstance(selector_kwargs, dict):
        raise TypeError(
            "model.config.pooling_config.selector_kwargs must resolve to a mapping "
            f"when selector temperature scheduling is enabled, got {type(selector_kwargs)!r}"
        )

    selector_kwargs["temperature_warmup_steps"] = int(warmup_steps)
    selector_kwargs["temperature_anneal_steps"] = int(anneal_steps)
    OmegaConf.update(cfg, selector_kwargs_path, selector_kwargs, merge=False)

    if logger is not None:
        logger.info(
            "Selector temperature schedule resolved: total_steps=%d, "
            "warmup_steps=%d (%s), anneal_steps=%d (%s), "
            "warmup_ratio=%.4f, anneal_ratio=%.4f",
            total_steps, warmup_steps, warmup_source,
            anneal_steps, anneal_source, warmup_ratio, anneal_ratio,
        )

    return {
        "enabled": True,
        "applied": True,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "anneal_steps": anneal_steps,
        "warmup_ratio": warmup_ratio,
        "anneal_ratio": anneal_ratio,
        "warmup_source": warmup_source,
        "anneal_source": anneal_source,
    }


def _resolve_selector_exploration_schedule(
    cfg: DictConfig,
    total_steps: int,
    logger: Optional[logging.Logger] = None,
) -> Optional[dict]:
    """
    Resolve selector Gumbel exploration_noise warmup/anneal steps from trainer config.

    Mirrors ``_resolve_selector_temperature_schedule``: reads
    ``trainer.selector_exploration_schedule.{enabled, warmup_ratio, anneal_ratio,
    warmup_steps, anneal_steps}`` and writes resolved integers into:
      model.config.pooling_config.selector_kwargs.exploration_noise_warmup_steps
      model.config.pooling_config.selector_kwargs.exploration_noise_anneal_steps
    """
    schedule_cfg = cfg.trainer.get("selector_exploration_schedule", {})
    enabled = bool(getattr(schedule_cfg, "get", lambda k, d=None: d)("enabled", False))
    if not enabled:
        return None

    if total_steps <= 0:
        raise ValueError(
            f"total_steps must be > 0 when selector_exploration_schedule is enabled, got {total_steps}"
        )

    selection_method = str(
        OmegaConf.select(cfg, "model.config.pooling_config.selection_method", default="stride")
    ).lower()
    if selection_method != "learned":
        reason = f"selection_method='{selection_method}' (requires 'learned')"
        if logger is not None:
            logger.info("Selector exploration schedule enabled but not applied: %s", reason)
        return {
            "enabled": True,
            "applied": False,
            "reason": reason,
            "total_steps": total_steps,
        }

    warmup_steps_override = getattr(schedule_cfg, "get", lambda k, d=None: d)("warmup_steps", None)
    anneal_steps_override = getattr(schedule_cfg, "get", lambda k, d=None: d)("anneal_steps", None)

    warmup_ratio = float(getattr(schedule_cfg, "get", lambda k, d=None: d)("warmup_ratio", 0.0))
    anneal_ratio = float(getattr(schedule_cfg, "get", lambda k, d=None: d)("anneal_ratio", 0.90))
    if not (0.0 <= warmup_ratio <= 1.0):
        raise ValueError(
            f"trainer.selector_exploration_schedule.warmup_ratio must be in [0, 1], got {warmup_ratio}"
        )
    if not (0.0 <= anneal_ratio <= 1.0):
        raise ValueError(
            f"trainer.selector_exploration_schedule.anneal_ratio must be in [0, 1], got {anneal_ratio}"
        )

    def _resolve_steps(
        *,
        name: str,
        override: Optional[Any],
        ratio: float,
        total_steps: int,
    ) -> Tuple[int, str]:
        if override is not None:
            resolved = int(override)
            if resolved < 0:
                raise ValueError(
                    f"trainer.selector_exploration_schedule.{name} must be >= 0, got {resolved}"
                )
            return resolved, "explicit"
        return max(0, int(total_steps * ratio)), "ratio"

    warmup_steps, warmup_source = _resolve_steps(
        name="warmup_steps",
        override=warmup_steps_override,
        ratio=warmup_ratio,
        total_steps=total_steps,
    )
    anneal_steps, anneal_source = _resolve_steps(
        name="anneal_steps",
        override=anneal_steps_override,
        ratio=anneal_ratio,
        total_steps=total_steps,
    )

    selector_kwargs_path = "model.config.pooling_config.selector_kwargs"
    selector_kwargs = OmegaConf.to_container(
        OmegaConf.select(cfg, selector_kwargs_path),
        resolve=False,
        throw_on_missing=False,
    )
    if selector_kwargs is None:
        selector_kwargs = {}
    if not isinstance(selector_kwargs, dict):
        raise TypeError(
            "model.config.pooling_config.selector_kwargs must resolve to a mapping "
            f"when selector exploration scheduling is enabled, got {type(selector_kwargs)!r}"
        )

    selector_kwargs["exploration_noise_warmup_steps"] = int(warmup_steps)
    selector_kwargs["exploration_noise_anneal_steps"] = int(anneal_steps)
    OmegaConf.update(cfg, selector_kwargs_path, selector_kwargs, merge=False)

    if logger is not None:
        logger.info(
            "Selector exploration schedule resolved: total_steps=%d, "
            "warmup_steps=%d (%s), anneal_steps=%d (%s), "
            "warmup_ratio=%.4f, anneal_ratio=%.4f",
            total_steps, warmup_steps, warmup_source,
            anneal_steps, anneal_source, warmup_ratio, anneal_ratio,
        )

    return {
        "enabled": True,
        "applied": True,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "anneal_steps": anneal_steps,
        "warmup_ratio": warmup_ratio,
        "anneal_ratio": anneal_ratio,
        "warmup_source": warmup_source,
        "anneal_source": anneal_source,
    }


def _setup_logging(cfg: DictConfig, hydra_cfg) -> tuple[logging.Logger, str]:
    """
    Configure logging with file and console handlers.
    
    Args:
        cfg: Hydra config containing trainer.log_level
        hydra_cfg: Hydra runtime config for output directory
        
    Returns:
        Tuple of (logger, log_file_path)
    """
    log_file = f"{hydra_cfg.runtime.output_dir}/train.log"
    
    # Determine logging level (configurable via cfg.trainer.log_level)
    level_name = str(cfg.trainer.get("log_level", "INFO")).upper() if "trainer" in cfg else "INFO"

    # Validate log level; fall back to INFO if invalid
    if level_name not in logging._nameToLevel:
        print(f"Warning: Invalid log level '{level_name}'; defaulting to INFO")
        level_name = "INFO"

    level = getattr(logging, level_name)

    # Configure root logger to write all logs to file
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove any existing handlers to avoid duplicates
    root_logger.handlers.clear()
    
    # File handler for all logs
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s')
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Console handler for important logs
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter('[%(levelname)s] %(message)s')
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # matplotlib.mathtext logs an INFO message for every Greek-letter glyph
    # substitution when rendering math text (e.g. \rho in scatter plot
    # annotations).  These messages are never actionable, so suppress them
    # unconditionally regardless of the active log level.
    logging.getLogger("matplotlib.mathtext").setLevel(logging.WARNING)
    
    # matplotlib.mathtext logs an INFO message for every Greek-letter glyph
    # substitution when rendering math text (e.g. \rho in scatter plot
    # annotations).  These messages are never actionable, so suppress them
    # unconditionally regardless of the active log level.
    logging.getLogger("matplotlib.mathtext").setLevel(logging.WARNING)

    # Reduce verbosity of third-party libraries when using DEBUG level
    if level_name == "DEBUG":
        # Matplotlib is extremely verbose at DEBUG - set to INFO
        logging.getLogger("matplotlib").setLevel(logging.INFO)
        logging.getLogger("PIL").setLevel(logging.INFO)
        # You can add other noisy libraries here as needed
        # logging.getLogger("torch").setLevel(logging.INFO)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to: {log_file} (level={level_name})")
    
    return logger, log_file


def _log_device_info(device: torch.device, logger: logging.Logger) -> None:
    """Log process ID and GPU/device information to console and log file."""
    separator = "="*60
    print(f"\n{separator}")
    logger.info(separator)
    
    pid_msg = f"Process ID (PID): {os.getpid()}"
    print(pid_msg)
    logger.info(pid_msg)
    
    if torch.cuda.is_available():
        gpu_id = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(gpu_id)
        gpu_msg = f"Using GPU {gpu_id}: {gpu_name}"
        device_msg = f"CUDA Device: {device}"
        print(gpu_msg)
        print(device_msg)
        logger.info(gpu_msg)
        logger.info(device_msg)
    else:
        device_msg = f"Using device: {device}"
        print(device_msg)
        logger.info(device_msg)
    
    print(f"{separator}\n")
    logger.info(separator)


# ── Pre-training setup helpers ────────────────────────────────────────────


def _sync_resolved_run_name(cfg: DictConfig) -> None:
    """Ensure ``cfg.run_name`` reflects the value chosen by the resolver.

    By the time ``main()`` runs, Hydra has already resolved
    ``${gen_run_name:${run_name}}`` to create the output directory.  This
    helper writes that resolved value back into the live config so that
    downstream code (e.g. ``_init_wandb``) can read ``cfg.run_name``.
    """
    hydra_cfg = HydraConfig.get()
    # The last path component of the output dir is the resolved run name
    resolved = os.path.basename(hydra_cfg.runtime.output_dir)
    OmegaConf.update(cfg, "run_name", resolved, merge=False)


def _validate_task_dataset(cfg: DictConfig) -> None:
    """Raise early if the task-dataset combination is invalid."""
    if "task" not in cfg or cfg.task is None:
        return
    valid_datasets = cfg.task.get("valid_datasets", [])
    if valid_datasets and cfg.dataset.name not in valid_datasets:
        available = ", ".join(valid_datasets)
        raise ValueError(
            f"\n{'='*60}\n"
            f"ERROR: Invalid task-dataset combination!\n"
            f"  Task: {cfg.task.name}\n"
            f"  Dataset: {cfg.dataset.name}\n"
            f"  Valid datasets for this task: {available}\n"
            f"{'='*60}\n"
        )


def _validate_revin_flags(cfg: DictConfig, logger: logging.Logger) -> None:
    """Ensure RevIN diffusion flag and dataset space-alignment flag are consistent.

    The cross-check only applies when the dataset config defines
    ``standardize_target_in_x_for_revin`` (currently SP500).  Datasets that
    don't define the key (e.g. WRA) are unconstrained — they handle
    conditioning/target spaces differently and don't need the flag.
    """
    revin_on = bool(cfg.diffusion.get("revin", False))
    has_flag = "standardize_target_in_x_for_revin" in cfg.dataset
    space_align = bool(cfg.dataset.get("standardize_target_in_x_for_revin", False))
    if revin_on and has_flag and not space_align:
        raise ValueError(
            "diffusion.revin=true requires dataset.standardize_target_in_x_for_revin=true "
            "so that conditioning and target are in the same standardized space. "
            "Add  dataset.standardize_target_in_x_for_revin=true  to your launch command."
        )
    if space_align and not revin_on:
        logger.warning(
            "dataset.standardize_target_in_x_for_revin=true has no effect "
            "without diffusion.revin=true. The target channel in x will be "
            "standardized but RevIN normalization is disabled."
        )


def _build_data(
    cfg: DictConfig,
    logger: logging.Logger,
) -> Tuple[Any, dict, dict]:
    """
    Build datasets and data-loaders from the registry.

    Returns:
        (builder, datasets_dict, loaders_dict)
    """
    if "dataset" not in cfg or "name" not in cfg.dataset:
        raise ValueError(
            "Config must define `dataset.name`. "
            f"Known datasets: {_pretty_known(DATASET_REGISTRY)}"
        )

    dataset_name = str(cfg.dataset.name)
    try:
        builder_cls = get_dataset_builder(dataset_name)
    except KeyError as e:
        raise KeyError(
            f"Unknown dataset '{dataset_name}'. "
            f"Known datasets: {_pretty_known(DATASET_REGISTRY)}"
        ) from e

    builder = builder_cls()
    datasets = builder.build_datasets(cfg.dataset)
    loaders = builder.build_loaders(cfg.dataset, datasets)

    # If the builder computed σ_cond_w training-set statistics, expose them
    # in the cfg under `dataset_info.sigma_cond_mean_train` so downstream
    # diffusion configs can interpolate via ${revin_alpha:w,b}.
    _inject_dataset_info_into_cfg(cfg, builder, logger)

    # Log summary
    logger.info("Known datasets: %s", _pretty_known(DATASET_REGISTRY))
    logger.info("Using dataset: %s", dataset_name)
    logger.info("Train batches/epoch: %d", len(loaders["train"]))
    for split in ("val", "test", "train-val"):
        if split in loaders and loaders[split] is not None:
            logger.info("%s batches/epoch: %d", split.capitalize(), len(loaders[split]))

    return builder, datasets, loaders


def _inject_dataset_info_into_cfg(
    cfg: DictConfig,
    builder: Any,
    logger: logging.Logger,
) -> None:
    """Surface builder-side statistics into the cfg tree.

    Currently exposes only ``dataset_info.sigma_cond_mean_train`` (used by
    the ``${revin_alpha:w,b}`` resolver to auto-derive the RevIN σ scale
    correction from the blend weight). Silently skips when the builder does
    not have an ``sigma_cond_mean_train`` attribute (e.g., non-SP500 builders).
    """
    from omegaconf import OmegaConf

    sigma_mean = getattr(builder, "_sigma_cond_mean_train", None)
    if sigma_mean is None:
        return
    # Open the struct temporarily so we can add a top-level key.
    was_struct = OmegaConf.is_struct(cfg)
    try:
        OmegaConf.set_struct(cfg, False)
        OmegaConf.update(
            cfg,
            "dataset_info.sigma_cond_mean_train",
            float(sigma_mean),
            merge=True,
        )
        logger.info(
            "Injected dataset_info.sigma_cond_mean_train = %.4f into cfg "
            "(available to ${revin_alpha:w,b} resolver).",
            float(sigma_mean),
        )
    finally:
        OmegaConf.set_struct(cfg, was_struct)


def _persist_resolved_revin_sigma_correction(
    cfg: DictConfig,
    hydra_cfg: Any,
    logger: logging.Logger,
) -> None:
    """Materialize ``cfg.diffusion.revin_sigma_correction`` to a literal
    float and re-save ``.hydra/config.yaml``.

    Without this step the saved config retains the
    ``${revin_alpha:${.revin_blend_weight},${oc.select:dataset_info.sigma_cond_mean_train,0.82}}``
    interpolation. On checkpoint reload (e.g., compare_baselines), the
    ``dataset_info.sigma_cond_mean_train`` injection does not re-run, so the
    resolver falls back to its literal default (0.82). The α used at eval
    time would then differ from the training-time α by a few percent — small,
    but a real mismatch.

    By resolving the interpolation here (AFTER ``_inject_dataset_info_into_cfg``
    has populated ``dataset_info``) and rewriting ``.hydra/config.yaml``, both
    the cfg snapshot in memory and the on-disk artifact carry the same
    literal numeric α. The non-persistent ``revin_sigma_correction`` buffer
    on the DDPM instance ends up matching automatically on reload.

    Silently no-ops when:
      - ``cfg.diffusion`` is missing,
      - ``cfg.diffusion.revin`` is false,
      - ``cfg.diffusion.revin_sigma_correction`` is missing,
      - or the on-disk ``.hydra/config.yaml`` cannot be located.
    """
    from omegaconf import OmegaConf  # local import: identical to module-level

    if "diffusion" not in cfg or cfg.diffusion is None:
        return
    if not bool(OmegaConf.select(cfg, "diffusion.revin", default=False)):
        return
    if "revin_sigma_correction" not in cfg.diffusion:
        return

    try:
        resolved_alpha = float(cfg.diffusion.revin_sigma_correction)
    except Exception as exc:
        logger.warning(
            "Could not resolve diffusion.revin_sigma_correction: %s. "
            ".hydra/config.yaml will retain the unresolved interpolation.",
            exc,
        )
        return

    was_struct = OmegaConf.is_struct(cfg)
    try:
        OmegaConf.set_struct(cfg, False)
        cfg.diffusion.revin_sigma_correction = resolved_alpha
    finally:
        OmegaConf.set_struct(cfg, was_struct)

    output_dir = getattr(getattr(hydra_cfg, "runtime", None), "output_dir", None)
    if output_dir is None:
        logger.warning(
            "Materialized diffusion.revin_sigma_correction = %.6f in cfg, but "
            "HydraConfig.runtime.output_dir is None — literal not persisted to disk.",
            resolved_alpha,
        )
        return

    config_path = os.path.join(str(output_dir), ".hydra", "config.yaml")
    if not os.path.isdir(os.path.dirname(config_path)):
        logger.warning(
            "Materialized diffusion.revin_sigma_correction = %.6f in cfg, but "
            ".hydra dir not found at %s — literal not persisted to disk.",
            resolved_alpha,
            os.path.dirname(config_path),
        )
        return

    try:
        OmegaConf.save(cfg, config_path)
    except Exception as exc:
        logger.warning(
            "Failed to re-save %s after materializing diffusion.revin_sigma_correction: %s",
            config_path,
            exc,
        )
        return

    logger.info(
        "Materialized diffusion.revin_sigma_correction = %.4f and re-saved %s "
        "(eval-time α now matches training-time α regardless of dataset_info injection).",
        resolved_alpha,
        config_path,
    )


def _inject_sp500_destandardization(
    task: Any,
    datasets: dict,
    logger: logging.Logger,
) -> None:
    """Inject target destandardization for SP500 dataset variants."""
    train_dataset = datasets.get("full", datasets.get("train"))
    if train_dataset is None or not hasattr(train_dataset, "get_target_standardization_stats"):
        return
    try:
        target_stats = train_dataset.get_target_standardization_stats()
        if hasattr(task, "set_target_destandardization"):
            from graph_signal_diffusion.datasets.sp500.dataset import SP500Stocks

            task.set_target_destandardization(
                target_stats, SP500Stocks.destandardize_target
            )
            logger.info(
                "Target destandardization injected: %s, scale_factor=%s",
                target_stats["target_name"],
                target_stats["scale_factor"],
            )
    except ValueError as e:
        logger.warning("Could not inject target destandardization: %s", e)


def _setup_task(
    cfg: DictConfig,
    builder: Any,
    datasets: dict,
    dataset_name: str,
    logger: logging.Logger,
) -> Optional[Any]:
    """
    Instantiate the task evaluator and inject dataset-specific metadata
    (normalizer, dataset info, destandardization).
    """
    if "task" not in cfg or cfg.task is None:
        return None

    task = instantiate(cfg.task)
    logger.info("Using task evaluator: %s", cfg.task.name)

    # Normalizer
    if hasattr(builder, "normalizer") and builder.normalizer is not None:
        if hasattr(task, "set_normalizer"):
            task.set_normalizer(builder.normalizer)
            logger.info("Normalizer injected into task evaluator")

    # Dataset info (scale factors, etc.)
    if hasattr(builder, "get_dataset_info"):
        dataset_info = builder.get_dataset_info()
        if dataset_info is not None and hasattr(task, "set_dataset_info"):
            task.set_dataset_info(dataset_info)
            logger.info("Dataset info injected into task evaluator")

    # SP500-specific destandardization
    if dataset_name in ("sp500", "sp500_cleaned"):
        _inject_sp500_destandardization(task, datasets, logger)

    return task


def _apply_training_optimizations(
    cfg: DictConfig,
    diffusion: Any,
    logger: logging.Logger,
) -> bool:
    """
    Apply AMP and/or ``torch.compile()`` if configured.

    Returns:
        use_amp flag for the trainer.
    """
    use_amp = bool(cfg.trainer.get("use_amp", False))

    if bool(cfg.trainer.get("compile", False)):
        # jaxtyping runtime checks are incompatible with torch.compile
        jaxtyping.config.update("jaxtyping_disable", True)
        logger.info("Disabled jaxtyping type checking (incompatible with torch.compile)")

        if hasattr(diffusion, "compile"):
            diffusion.compile()
        elif hasattr(torch, "compile"):
            diffusion.model = torch.compile(diffusion.model)
            logger.info("Model compiled with torch.compile()")

    return use_amp


def _init_wandb(cfg: DictConfig, hydra_cfg):
    """
    Build the wandb context manager.

    Returns a context manager that yields the wandb Run when enabled,
    or ``contextlib.nullcontext()`` (yields None) when disabled.

    Auto-derives ``group``, ``tags``, and ``job_type`` from the Hydra
    config.  Explicit ``wandb.group`` / ``wandb.tags`` values in the
    config take precedence over auto-derived defaults.
    """
    if not cfg.get("wandb", {}).get("enabled", False):
        return contextlib.nullcontext()

    from graph_signal_diffusion.utils.wandb_context import (
        build_wandb_context,
        merge_wandb_context,
    )

    context = build_wandb_context(cfg, stage="train")
    merged = merge_wandb_context(cfg, context)

    wandb_config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=False
    )

    run_name = cfg.get("run_name", None) or cfg.wandb.get("name", None)

    return wandb.init(
        project=cfg.wandb.get("project", "graph-signal-diffusion"),
        entity=cfg.wandb.get("entity", None),
        group=merged["group"],
        job_type=merged["job_type"],
        tags=merged["tags"],
        notes=cfg.wandb.get("notes", None),
        name=run_name,
        mode=cfg.wandb.get("mode", "online"),
        config=wandb_config,
        dir=hydra_cfg.runtime.output_dir,
    )


def _is_graceful_system_exit(exc: SystemExit) -> bool:
    """Return True when SystemExit likely corresponds to an intentional stop."""
    code = exc.code
    if code is None:
        return True
    if isinstance(code, int):
        return code in (0, 130, 143)
    if isinstance(code, str):
        stripped = code.strip()
        if stripped.isdigit():
            return int(stripped) in (0, 130, 143)
    return False


@contextlib.contextmanager
def _map_sigterm_to_keyboard_interrupt():
    """Translate SIGTERM to KeyboardInterrupt for graceful W&B run closure."""
    previous_sigterm_handler = signal.getsignal(signal.SIGTERM)

    def _handle_sigterm(signum, _frame):
        try:
            signal_name = signal.Signals(signum).name
        except Exception:
            signal_name = str(signum)
        raise KeyboardInterrupt(f"Received {signal_name}")

    signal.signal(signal.SIGTERM, _handle_sigterm)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm_handler)


# ── Main entry point ──────────────────────────────────────────────────────


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # ── 1. Run naming & early validation ──────────────────────────────
    _sync_resolved_run_name(cfg)
    _validate_task_dataset(cfg)

    # ── 2. Discover plugins ───────────────────────────────────────────
    discover_datasets()
    discover_tasks()
    discover_models()

    # ── 3. Logging & config dump ──────────────────────────────────────
    hydra_cfg = HydraConfig.get()
    logger, log_file = _setup_logging(cfg, hydra_cfg)
    logger.info("=== Resolved config ===")
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))

    # ── 4. Seed ───────────────────────────────────────────────────────
    set_seed(int(cfg.get("seed", 0)))

    # ── 5. Data ───────────────────────────────────────────────────────
    builder, datasets, loaders = _build_data(cfg, logger)

    # Now that the builder has injected dataset_info.sigma_cond_mean_train,
    # materialize the ${revin_alpha:...} interpolation as a literal in cfg
    # AND in .hydra/config.yaml — so checkpoint reload (compare_baselines,
    # etc.) uses the same α as training without needing a re-injection.
    _persist_resolved_revin_sigma_correction(cfg, hydra_cfg, logger)

    dataset_name = str(cfg.dataset.name)
    train_loader = loaders["train"]
    val_loader = loaders.get("val")
    test_loader = loaders.get("test")
    train_val_loader = loaders.get("train-val")
    max_num_train_val_batches = int(cfg.trainer.get("max_num_train_val_batches", 1))
    if max_num_train_val_batches == 0:
        logger.info(
            "Skipping train-val evaluation: trainer.max_num_train_val_batches=0"
        )
        train_val_loader = None

    # ── 6. Schedule pre-computation ───────────────────────────────────
    max_epochs = int(cfg.trainer.get("max_epochs", 1))
    total_steps = max_epochs * len(train_loader)
    _resolve_selector_temperature_schedule(cfg, total_steps, logger)
    _resolve_selector_exploration_schedule(cfg, total_steps, logger)

    # ── 7. wandb context ──────────────────────────────────────────────
    with _init_wandb(cfg, hydra_cfg) as wandb_run:
        if wandb_run is not None:
            logger.info(
                "wandb run initialized: %s (project=%s)",
                wandb_run.name, wandb_run.project,
            )

        # ── 8. Model & diffusion ──────────────────────────────────────
        model = None
        logger.info("Known models: %s", _pretty_known(MODEL_REGISTRY))
        if "model" in cfg and cfg.model is not None:
            model = instantiate(cfg.model)
            logger.info("Using model: %s", cfg.model.name)

        diffusion = instantiate(cfg.diffusion, model=model)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _log_device_info(device, logger)

        model.to(device)
        diffusion.to(device)

        # ── 8b. RevIN flag cross-check ────────────────────────────────
        _validate_revin_flags(cfg, logger)

        # ── 9. Optimizer & LR scheduler ───────────────────────────────
        optimizer = _make_optimizer(cfg.trainer, model.parameters())
        lr_scheduler, schedule_meta = _make_lr_scheduler(
            cfg.trainer, optimizer=optimizer, total_steps=total_steps,
        )
        if schedule_meta is not None:
            _df = schedule_meta.get("decay_fraction")
            _de = schedule_meta.get("decay_end_step")
            if _df is not None:
                logger.info(
                    "Enabled LR scheduler '%s': total_steps=%d, "
                    "warmup_steps=%d (%.2f%%), min_lr_ratio=%.4f, "
                    "decay_fraction=%.3f (decay_end_step=%d)",
                    schedule_meta["name"],
                    schedule_meta["total_steps"],
                    schedule_meta["warmup_steps"],
                    schedule_meta["warmup_ratio"] * 100.0,
                    schedule_meta["min_lr_ratio"],
                    _df,
                    _de,
                )
            else:
                logger.info(
                    "Enabled LR scheduler '%s': total_steps=%d, "
                    "warmup_steps=%d (%.2f%%), min_lr_ratio=%.4f",
                    schedule_meta["name"],
                    schedule_meta["total_steps"],
                    schedule_meta["warmup_steps"],
                    schedule_meta["warmup_ratio"] * 100.0,
                    schedule_meta["min_lr_ratio"],
                )

        # ── 10. Task evaluator ────────────────────────────────────────
        logger.info("Known tasks: %s", _pretty_known(TASK_REGISTRY))
        task = _setup_task(cfg, builder, datasets, dataset_name, logger)

        # ── 11. Training optimizations (AMP / compile) ────────────────
        use_amp = _apply_training_optimizations(cfg, diffusion, logger)

        # ── 12. Trainer ───────────────────────────────────────────────
        trainer = DiffusionTrainer(
            diffusion=diffusion,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device=device,
            accelerator=None,
            max_grad_norm=float(cfg.trainer.get("max_grad_norm", 1.0)),
            log_every_n_steps=int(cfg.trainer.get("log_every_n_steps", 50)),
            save_dir=f"{hydra_cfg.runtime.output_dir}/trainer_chkpts",
            task=task,
            use_amp=use_amp,
            wandb_run=wandb_run,
            diagnose_model=cfg.trainer.get("diagnose_model", None),
            diagnose_selector_every_n_epochs=int(
                cfg.trainer.get("diagnose_selector_every_n_epochs", 1)
            ),
            selector_sampling_max_network_plots=int(
                cfg.trainer.get("selector_sampling_max_network_plots", 4)
            ),
            best_model=cfg.trainer.get("best_model", None),
            max_num_val_batches=int(cfg.trainer.get("max_num_val_batches", 1)),
            max_num_train_val_batches=max_num_train_val_batches,
            max_num_test_batches=int(cfg.trainer.get("max_num_test_batches", -1)),
            run_final_test=bool(cfg.trainer.get("run_final_test", True)),
            measure_forward_speed=bool(cfg.trainer.get("measure_forward_speed", False)),
            forward_speed_warmup_steps=int(cfg.trainer.get("forward_speed_warmup_steps", 5)),
            forward_speed_max_steps=int(cfg.trainer.get("forward_speed_max_steps", -1)),
            eval_every_n_epochs=int(cfg.trainer.get("eval_every_n_epochs", 10)),
            eval_schedule=cfg.trainer.get("eval_schedule", None),
        )

        try:
            with _map_sigterm_to_keyboard_interrupt():
                trainer.fit(
                    train_loader, val_loader, test_loader, train_val_loader,
                    max_epochs=max_epochs,
                    save_checkpoint_every_n_epochs=int(
                        cfg.trainer.get("save_checkpoint_every_n_epochs", 100)
                    ),
                    max_num_test_batches=int(
                        cfg.trainer.get("max_num_test_batches", -1)
                    ),
                )
        except KeyboardInterrupt as exc:
            if wandb_run is not None:
                with contextlib.suppress(Exception):
                    wandb_run.summary["graceful_stop"] = True
                    wandb_run.summary["stop_reason"] = str(exc) or "KeyboardInterrupt"
            logger.warning(
                "Training interrupted (%s). Exiting gracefully so W&B marks the run as finished.",
                type(exc).__name__,
            )
            return
        except SystemExit as exc:
            if _is_graceful_system_exit(exc):
                if wandb_run is not None:
                    with contextlib.suppress(Exception):
                        wandb_run.summary["graceful_stop"] = True
                        wandb_run.summary["stop_reason"] = f"SystemExit({exc.code})"
                logger.warning(
                    "Training interrupted via SystemExit(%s). Exiting gracefully so W&B marks the run as finished.",
                    exc.code,
                )
                return
            raise


if __name__ == "__main__":
    main()
