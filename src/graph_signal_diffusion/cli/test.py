# src/graph_signal_diffusion/cli/test.py
"""Evaluate a trained diffusion model from a saved checkpoint.

Runs outside the Hydra runtime: loads the saved ``.hydra/config.yaml``,
reconstructs the model/diffusion pipeline, and evaluates on requested
data splits.  Supports diffusion-type overrides (DDPM ↔ DDIM),
sampling-timestep changes, AMP, and ``torch.compile()``.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import jaxtyping
import numpy as np
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate

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
from graph_signal_diffusion.utils.selector_temperature_restore import (
    restore_selector_temperature_from_records,
)


# ── Helpers ────────────────────────────────────────────────────────────


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _pretty_known(registry: dict) -> str:
    """Format registry keys for display (e.g. 'cifar10 / sp500 / wra')."""
    try:
        return " / ".join(registry.keys())
    except Exception:
        return "<unavailable>"
    

def _make_optimizer(trainer_cfg, params):
    """Create optimizer (needed for trainer instantiation even for evaluation)"""
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

    if name == "adam":
        return torch.optim.Adam(
            params, lr=lr, betas=_parse_betas(raw_betas), weight_decay=weight_decay
        )
    if name == "adamw":
        return torch.optim.AdamW(
            params, lr=lr, betas=_parse_betas(raw_betas), weight_decay=weight_decay
        )
    if name == "sgd":
        return torch.optim.SGD(
            params, lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    raise ValueError(f"Unknown optimizer: {name}")


def _inject_sp500_destandardization(
    task: Any,
    datasets: dict,
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
            print(
                f"✅ SP500 target destandardisation injected: "
                f"{target_stats['target_name']}, "
                f"scale_factor={target_stats['scale_factor']}"
            )
    except ValueError as exc:
        print(f"WARNING: Could not inject target destandardisation: {exc}")


def _setup_task(
    cfg,
    builder: Any,
    datasets: dict,
    dataset_name: str,
) -> Optional[Any]:
    """Instantiate the task evaluator and inject dataset-specific metadata.

    Handles normalizer, dataset_info, and SP500 destandardization — matching
    the logic in ``train.py`` and ``evaluate.py``.
    """
    if "task" not in cfg or cfg.task is None:
        return None

    task = instantiate(cfg.task)
    print(f"Using task evaluator: {cfg.task.name}")

    # Normalizer
    if hasattr(builder, "normalizer") and builder.normalizer is not None:
        if hasattr(task, "set_normalizer"):
            task.set_normalizer(builder.normalizer)
            print("✅ Normalizer injected into task evaluator")

    # Dataset info (scale factors, etc.)
    if hasattr(builder, "get_dataset_info"):
        dataset_info = builder.get_dataset_info()
        if dataset_info is not None and hasattr(task, "set_dataset_info"):
            task.set_dataset_info(dataset_info)
            print("✅ Dataset info injected into task evaluator")

    # SP500-specific destandardization
    if dataset_name in ("sp500", "sp500_cleaned"):
        _inject_sp500_destandardization(task, datasets)

    return task


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, diffusion: torch.nn.Module) -> int:
    """Load model and diffusion weights from checkpoint.
    
    Returns:
        epoch: The epoch number from the checkpoint (parsed from filename if not in checkpoint dict)
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model state dict (keys: 'model' or 'model_state_dict')
    if 'model' in checkpoint and checkpoint['model'] is not None:
        model.load_state_dict(checkpoint['model'])
        print("✅ Model weights loaded")
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Model weights loaded")
    else:
        print("⚠️  Warning: 'model' or 'model_state_dict' not found in checkpoint")
    
    # Load diffusion state dict (keys: 'diffusion' or 'diffusion_state_dict')
    # Use strict=False because diffusion buffers (betas, alphas, etc.) are recomputed
    # from config and may differ in size if num_timesteps changed
    if 'diffusion' in checkpoint:
        diffusion.load_state_dict(checkpoint['diffusion'], strict=False)
        print("✅ Diffusion weights loaded (strict=False for buffer compatibility)")
    elif 'diffusion_state_dict' in checkpoint:
        diffusion.load_state_dict(checkpoint['diffusion_state_dict'], strict=False)
        print("✅ Diffusion weights loaded (strict=False for buffer compatibility)")
    
    # Extract epoch info (from checkpoint dict or filename)
    epoch = checkpoint.get('epoch', None)
    
    if epoch is None:
        # Try to parse epoch from filename (e.g., "DDIM_epoch_100.pt" -> 100)
        filename = os.path.basename(checkpoint_path)
        match = re.search(r'epoch[_-](\d+)', filename, re.IGNORECASE)
        if match:
            # Trainer saves with 1-based naming (epoch+1), convert to 0-based
            epoch = int(match.group(1)) - 1
            print(f"   Parsed epoch from filename: {match.group(1)} (1-based) → {epoch} (0-based)")
        else:
            epoch = 0
            print("   Warning: Could not find epoch in checkpoint or filename, defaulting to 0")
    else:
        # Assume checkpoint epoch is already 0-based (consistent with trainer.epoch)
        print(f"   Checkpoint epoch (0-based): {epoch}")
    
    # Print additional checkpoint info
    if 'loss' in checkpoint:
        print(f"   Checkpoint loss: {checkpoint['loss']:.4f}")
    
    return epoch


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained diffusion model using saved checkpoint and Hydra config"
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        required=True,
        help="Path to .hydra directory containing config.yaml (e.g., outputs/.../run/.hydra)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file (e.g., DDIM_epoch_350.pt)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory for evaluation results (default: same as config dir)"
    )
    
    # Diffusion sampling overrides
    parser.add_argument(
        "--diffusion-type",
        type=str,
        default=None,
        choices=["ddpm", "ddim"],
        help="Override diffusion type for sampling (e.g., use DDIM sampler on DDPM-trained model)"
    )
    parser.add_argument(
        "--sampling-timesteps",
        type=int,
        default=None,
        help="Override number of sampling timesteps (for DDIM acceleration, e.g., 50, 100)"
    )
    parser.add_argument(
        "--ddim-eta",
        type=float,
        default=None,
        help="Override DDIM eta parameter (0.0=deterministic, 1.0=stochastic like DDPM)"
    )
    
    # Task evaluation overrides
    parser.add_argument(
        "--n-samples-per-input",
        type=int,
        default=None,
        help="Override number of samples to generate per input (for probabilistic evaluation)"
    )
    parser.add_argument(
        "--eval-on",
        type=str,
        default="test",
        help="Which dataloaders to evaluate on: 'val', 'test', 'train-val' or their combinations like 'val,test' (comma-separated for multiple)"
    )
    parser.add_argument(
        "--single-batch",
        action="store_true",
        help="Evaluate on single batch only (faster). Default: evaluate on full dataloader"
    )
    parser.add_argument(
        "--selector-rate-correlation",
        action="store_true",
        help=(
            "Run a forward-only selector probe and feed the per-network node stats "
            "into evaluation so selector-rate-correlation plots (per-network AND the "
            "per-density / global aggregates) are regenerated. Requires a model with a "
            "learned selector/pooling GNN; no-op otherwise."
        ),
    )
    
    # Performance optimization options
    parser.add_argument(
        "--use-amp",
        action="store_true",
        help="Use automatic mixed precision (FP16) for faster inference"
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile() for model optimization (PyTorch 2.0+)"
    )
    
    args = parser.parse_args()
    
    # Load Hydra config from .hydra directory
    config_path = Path(args.config_dir) / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    print(f"Loading config from: {config_path}")
    cfg = OmegaConf.load(config_path)
    
    # Register custom resolver for hydra interpolations (not available outside Hydra runtime)
    # This allows ${hydra:runtime.cwd} to resolve to the original project root
    if not OmegaConf.has_resolver("hydra"):
        hydra_cfg_path = Path(args.config_dir)
        if hydra_cfg_path.name == ".hydra":
            # Get the original project root from hydra.yaml if available
            hydra_yaml_path = hydra_cfg_path / "hydra.yaml"
            if hydra_yaml_path.exists():
                hydra_cfg = OmegaConf.load(hydra_yaml_path)
                project_root = hydra_cfg.hydra.runtime.cwd
            else:
                # Fallback to current working directory
                project_root = os.getcwd()
        else:
            project_root = os.getcwd()
        
        # Register resolver that returns project_root for runtime.cwd
        OmegaConf.register_new_resolver(
            "hydra",
            lambda path: project_root if path == "runtime.cwd" else None,
            replace=False
        )
        print(f"Registered hydra resolver with project root: {project_root}")
    
    print("=== Loaded config ===")
    try:
        print(OmegaConf.to_yaml(cfg, resolve=True))
    except Exception as e:
        print(f"Warning: Could not fully resolve config: {e}")
        print(OmegaConf.to_yaml(cfg, resolve=False))
    
    # Set output directory
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = str(Path(args.config_dir).parent / "evaluation_results")
    
    # Add diffusion override suffix to output dir for organization
    if args.diffusion_type is not None or args.sampling_timesteps is not None or args.ddim_eta is not None:
        suffix_parts = []
        if args.diffusion_type:
            suffix_parts.append(args.diffusion_type)
        if args.sampling_timesteps:
            suffix_parts.append(f"steps{args.sampling_timesteps}")
        if args.ddim_eta is not None:
            suffix_parts.append(f"eta{args.ddim_eta}")
        suffix = "_".join(suffix_parts)
        output_dir = f"{output_dir}_{suffix}"
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Resolve experiment directory once from --config-dir.
    config_dir_path = Path(args.config_dir)
    if config_dir_path.name == ".hydra":
        experiment_dir = config_dir_path.parent
    else:
        experiment_dir = config_dir_path
    
    # Set seed
    seed = int(cfg.get("seed", 0))
    set_seed(seed)
    
    # Discover plugins
    discover_datasets()
    discover_tasks()
    discover_models()
    
    # Validate task-dataset compatibility
    if "task" in cfg and cfg.task is not None:
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
    
    # Build datasets/loaders
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
            f"Unknown dataset '{dataset_name}'. Known datasets: {_pretty_known(DATASET_REGISTRY)}"
        ) from e
    
    builder = builder_cls()
    datasets = builder.build_datasets(cfg.dataset)
    loaders = builder.build_loaders(cfg.dataset, datasets)
    
    train_loader = loaders["train"]
    val_loader = loaders.get("val")
    test_loader = loaders.get("test")
    train_val_loader = loaders.get("train-val")
    
    print(f"Known datasets: {_pretty_known(DATASET_REGISTRY)}")
    print(f"Using dataset: {dataset_name}")
    print(f"Train batches: {len(train_loader)}")
    if val_loader is not None:
        print(f"Val batches: {len(val_loader)}")
    if test_loader is not None:
        print(f"Test batches: {len(test_loader)}")
    if train_val_loader is not None:
        print(f"Train-Val batches: {len(train_val_loader)}")
    
    # Instantiate model
    model = None
    print(f"Known models: {_pretty_known(MODEL_REGISTRY)}")
    if "model" in cfg and cfg.model is not None:
        model = instantiate(cfg.model)
        print(f"Using model: {cfg.model.name}")
    
    # Apply diffusion overrides if provided
    diffusion_cfg = OmegaConf.to_container(cfg.diffusion, resolve=True)
    original_diffusion_type = diffusion_cfg.get('_target_', '').split('.')[-1].lower()
    
    if args.diffusion_type is not None:
        # Override diffusion type
        if args.diffusion_type == "ddim":
            diffusion_cfg['_target_'] = 'graph_signal_diffusion.diffusion.ddim.DDIM'
            print(f"🔄 Overriding diffusion type: {original_diffusion_type} → DDIM")
        elif args.diffusion_type == "ddpm":
            diffusion_cfg['_target_'] = 'graph_signal_diffusion.diffusion.ddpm.DDPM'
            print(f"🔄 Overriding diffusion type: {original_diffusion_type} → DDPM")
    
    if args.sampling_timesteps is not None:
        diffusion_cfg['sampling_timesteps'] = args.sampling_timesteps
        print(f"🔄 Overriding sampling_timesteps: {diffusion_cfg.get('sampling_timesteps', 'default')} → {args.sampling_timesteps}")
    
    if args.ddim_eta is not None:
        diffusion_cfg['ddim_eta'] = args.ddim_eta
        print(f"🔄 Overriding ddim_eta: {diffusion_cfg.get('ddim_eta', 'default')} → {args.ddim_eta}")
    
    # Apply compile_model if requested
    # Note: compile_model is handled AFTER loading checkpoint to avoid state_dict key mismatch
    # if args.compile:
    #     diffusion_cfg['compile_model'] = True
    #     print(f"🚀 Enabling torch.compile() for model optimization")
    
    # Reconstruct diffusion config and instantiate
    diffusion_cfg_omega = OmegaConf.create(diffusion_cfg)
    diffusion = instantiate(diffusion_cfg_omega, model=model)
    
    print(f"\n✅ Using diffusion: {diffusion_cfg['_target_'].split('.')[-1]}")
    if 'sampling_timesteps' in diffusion_cfg:
        print(f"   Sampling timesteps: {diffusion_cfg['sampling_timesteps']}")
    if 'ddim_eta' in diffusion_cfg:
        print(f"   DDIM eta: {diffusion_cfg['ddim_eta']}")
    if args.use_amp:
        print(f"   AMP (FP16) requested — effective only on CUDA (confirmed at trainer init)")
    if args.compile:
        print(f"   🚀 torch.compile() will be enabled after loading checkpoint")
    print()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    diffusion.to(device)
    
    # Load checkpoint
    checkpoint_epoch = load_checkpoint(args.checkpoint, model, diffusion)

    # Restore selector temperature from run artifacts (epoch summaries / manifest).
    restored_selector_temp = restore_selector_temperature_from_records(
        root_module=diffusion,
        checkpoint_path=args.checkpoint,
        checkpoint_epoch=checkpoint_epoch,
        experiment_dir=experiment_dir,
    )
    if restored_selector_temp is not None and restored_selector_temp.modules_updated > 0:
        print(
            "✅ Restored selector temperature "
            f"{restored_selector_temp.temperature:.6f} "
            f"from {restored_selector_temp.source} "
            f"(applied to {restored_selector_temp.modules_updated} selector modules)"
        )
    elif restored_selector_temp is not None:
        print(
            "ℹ️  Selector temperature record found "
            f"({restored_selector_temp.temperature:.6f}) but no selector modules "
            "were detected in the loaded diffusion model."
        )
    else:
        print(
            "⚠️  No selector temperature record found; keeping selector "
            "temperature from model config."
        )
    
    # Apply torch.compile AFTER loading checkpoint to avoid state_dict key mismatch
    # (torch.compile wraps the model and adds '_orig_mod.' prefix to state dict keys)
    if args.compile:
        # Disable jaxtyping runtime type checking - incompatible with torch.compile()
        jaxtyping.config.update("jaxtyping_disable", True)
        print("⚠️  Disabled jaxtyping type checking (incompatible with torch.compile)")
        
        if hasattr(diffusion, 'compile'):
            diffusion.compile()
        elif hasattr(torch, 'compile'):
            diffusion.model = torch.compile(diffusion.model)
            print("🚀 Model compiled with torch.compile()")
        else:
            print("⚠️  torch.compile() not available (requires PyTorch 2.0+)")
    
    # Create optimizer (needed for trainer but not used during evaluation)
    optimizer = _make_optimizer(cfg.trainer, model.parameters())
    
    # Instantiate task evaluator with full dataset-specific injection
    print(f"Known tasks: {_pretty_known(TASK_REGISTRY)}")
    task = _setup_task(cfg, builder, datasets, dataset_name)
    
    # Override n_samples_per_input if provided
    if task is not None and args.n_samples_per_input is not None:
        original_n = getattr(task, 'n_samples_per_input', 1)
        task.n_samples_per_input = args.n_samples_per_input
        print(f"🔄 Overriding n_samples_per_input: {original_n} → {args.n_samples_per_input}")
    
    # Create trainer
    trainer = DiffusionTrainer(
        diffusion=diffusion,
        optimizer=optimizer,
        device=device,
        max_grad_norm=float(cfg.trainer.get("max_grad_norm", 1.0)),
        log_every_n_steps=int(cfg.trainer.get("log_every_n_steps", 50)),
        save_dir=output_dir,
        task=task,
        use_amp=args.use_amp,
        max_num_val_batches=int(cfg.trainer.get("max_num_val_batches", 1)),
        max_num_train_val_batches=int(cfg.trainer.get("max_num_train_val_batches", 1)),
    )
    
    # Set epoch from checkpoint for correct visualization naming
    trainer.epoch = checkpoint_epoch
    print(f"✅ Trainer epoch set to: {checkpoint_epoch}")
    
    print("\n" + "="*60)
    print("Starting evaluation...")
    print("="*60 + "\n")
    
    # Parse which dataloaders to evaluate on
    eval_on_list = [x.strip() for x in args.eval_on.split(',')]
    
    # Collect available loaders
    available_loaders = {}
    if val_loader is not None:
        available_loaders['val'] = val_loader
    if test_loader is not None:
        available_loaders['test'] = test_loader
    if train_val_loader is not None:
        available_loaders['train-val'] = train_val_loader
    
    # Validate requested loaders
    for eval_name in eval_on_list:
        if eval_name not in available_loaders:
            available = ', '.join(available_loaders.keys()) if available_loaders else 'none'
            raise ValueError(f"Requested evaluation on '{eval_name}' but only {available} available")
    
    # Aggregate results
    # Get n_samples_per_input (from override or task config)
    n_samples = args.n_samples_per_input if args.n_samples_per_input is not None else getattr(task, 'n_samples_per_input', 1)
    
    # Get actual diffusion configuration used
    final_diffusion_type = diffusion_cfg['_target_'].split('.')[-1].lower()
    final_sampling_timesteps = diffusion_cfg.get('sampling_timesteps', diffusion_cfg.get('num_timesteps', 'N/A'))
    final_ddim_eta = diffusion_cfg.get('ddim_eta', 'N/A') if final_diffusion_type == 'ddim' else 'N/A'
    
    all_results = {
        "experiment_dir": str(experiment_dir),
        "epoch": checkpoint_epoch,
        "n_samples_per_input": n_samples,
        "eval_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "diffusion_type": final_diffusion_type,
        "sampling_timesteps": final_sampling_timesteps,
        "ddim_eta": final_ddim_eta,
        "eval_mode": "single_batch" if args.single_batch else "full_dataloader",
        "eval_on": args.eval_on,
        "use_amp": args.use_amp,
        "compile": args.compile,
    }
    
    # Determine evaluation pathway once (mirrors predict())
    task_uses_v2 = trainer._is_task_v2(task)
    print(f"Task evaluation pathway: {'v2 (evaluate_v2)' if task_uses_v2 else 'legacy (evaluate)'}")

    # Run evaluation on each requested dataloader
    for eval_name in eval_on_list:
        loader = available_loaders[eval_name]
        print(f"\nEvaluating on: {eval_name} set")
        print(f"Number of batches: {len(loader)}")

        # Optional selector probe: build per-network node stats so the
        # selector-rate-correlation plots (per-network + aggregates) regenerate.
        selector_stats = None
        if args.selector_rate_correlation:
            print(f"  Running selector probe on {eval_name} (selector-rate correlation)...")
            selector_stats = trainer.collect_selector_network_node_stats(
                loader,
                eval_split=eval_name,
                num_max_batches=(1 if args.single_batch else -1),
            )
            print(f"  Selector probe collected {len(selector_stats)} network key(s)")
            if len(selector_stats) == 0:
                print("  ⚠️  No selector node stats collected — model may lack a learned selector; "
                      "selector-rate-correlation plots will be skipped.")

        # Run evaluation — mirror predict()'s v2 / legacy branch
        if task_uses_v2:
            num_batches = 1 if args.single_batch else -1  # -1 = all batches
            if args.single_batch:
                print("  Mode: Single batch evaluation (evaluate_v2)")
            else:
                print("  Mode: Full dataloader evaluation (evaluate_v2)")
            eval_metrics = trainer.evaluate_v2(
                loader,
                collect_heavy_selector_diagnostics=True,
                eval_split=eval_name,
                selector_network_node_stats=selector_stats,
                force_task_eval=True,
                num_max_eval_batches=num_batches,
            )
        else:
            if args.single_batch:
                print("  Mode: Single batch evaluation (evaluate)")
                eval_data = next(iter(loader))
                eval_metrics = trainer.evaluate(
                    loader=eval_data,
                    eval_split=eval_name,
                    selector_network_node_stats=selector_stats,
                    force_task_eval=True,
                )
            else:
                print("  Mode: Full dataloader evaluation (evaluate)")
                eval_metrics = trainer.evaluate(
                    loader=loader,
                    eval_split=eval_name,
                    selector_network_node_stats=selector_stats,
                    force_task_eval=True,
                )
        
        # Add metrics with prefix (e.g., val_mae, test_loss)
        all_results.update({f"{eval_name}_{k}": v for k, v in eval_metrics.items()})
        
        # Print results for this loader
        print(f"Results for {eval_name} set:")
        for metric_name, metric_value in eval_metrics.items():
            print(f"  {metric_name}: {metric_value:.4f}")
    
    # Print combined results
    print("\n" + "="*60)
    print("Evaluation Results (Combined):")
    print("="*60)
    print(f"  experiment_dir: {all_results['experiment_dir']}")
    print(f"  epoch: {all_results['epoch']}")
    print(f"  n_samples_per_input: {all_results['n_samples_per_input']}")
    print(f"  eval_timestamp: {all_results['eval_timestamp']}")
    print(f"  diffusion_type: {all_results['diffusion_type']}")
    print(f"  sampling_timesteps: {all_results['sampling_timesteps']}")
    print(f"  ddim_eta: {all_results['ddim_eta']}")
    print(f"  eval_mode: {all_results['eval_mode']}")
    print(f"  eval_on: {all_results['eval_on']}")
    print(f"  use_amp: {all_results['use_amp']}")
    print(f"  compile: {all_results['compile']}")
    
    # Define metadata keys to skip in metric printing
    metadata_keys = ['experiment_dir', 'epoch', 'n_samples_per_input', 'eval_timestamp', 'diffusion_type', 
                     'sampling_timesteps', 'ddim_eta', 'eval_mode', 'eval_on', 'use_amp', 'compile']
    
    for metric_name, metric_value in all_results.items():
        if metric_name not in metadata_keys:
            print(f"  {metric_name}: {metric_value:.4f}")
    
    # Save most recent results to JSON file (overwrites)
    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nMost recent results saved to: {results_path}")
    
    # Append to cumulative history file (like epoch_summaries.jsonl)
    history_path = os.path.join(output_dir, "evaluation_history.jsonl")
    with open(history_path, 'a') as f:
        f.write(json.dumps(all_results) + "\n")
    print(f"Results appended to history: {history_path}")
    
    print("\n" + "="*60)
    print("Evaluation complete!")
    print(f"Results saved to: {output_dir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
