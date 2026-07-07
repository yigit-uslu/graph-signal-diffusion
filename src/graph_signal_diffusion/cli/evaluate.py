"""
Evaluation-only script for a single trained baseline.

Mirrors the pipeline in ``compare_baselines.py`` but for one baseline:
  1. Discover plugins
  2. Setup device
  3. Load data (builder → datasets → loaders)
  4. Setup task with dataset-specific injection (setup_task)
  5. Setup UnifiedEvaluator
  6. Load / fit baseline
  7. Evaluate on configurable splits with visualisation
  8. Log to W&B (optional)
  9. Save results
"""
from __future__ import annotations
import contextlib
import os

import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
import torch

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore[assignment]

from graph_signal_diffusion.datasets import discover_datasets, get_dataset_builder
from graph_signal_diffusion.tasks import discover_tasks
from graph_signal_diffusion.baselines import discover_baselines
from graph_signal_diffusion.evaluation import UnifiedEvaluator
from graph_signal_diffusion.cli._baseline_shared import (
    setup_task,
    log_results_to_wandb,
)


# ── Helpers ────────────────────────────────────────────────────────────


def _init_wandb(cfg: DictConfig, hydra_cfg, baseline_name: str):
    """Build the wandb context manager for single-baseline evaluation.

    Returns a context manager that yields the wandb Run when enabled,
    or ``contextlib.nullcontext()`` (yields None) when disabled.

    Auto-derives ``group``, ``tags``, and ``job_type`` from the Hydra
    config.  Explicit ``wandb.group`` / ``wandb.tags`` values in the
    config take precedence over auto-derived defaults.
    """
    if wandb is None:
        return contextlib.nullcontext()
    if not cfg.get("wandb", {}).get("enabled", False):
        return contextlib.nullcontext()

    from graph_signal_diffusion.utils.wandb_context import (
        build_wandb_context,
        merge_wandb_context,
    )

    context = build_wandb_context(cfg, stage="eval", baseline_name=baseline_name)
    merged = merge_wandb_context(cfg, context)

    wandb_config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=False
    )

    dataset_name = cfg.dataset.get("name", "unknown")
    run_name = (
        cfg.wandb.get("name", None)
        or f"baseline-{baseline_name}-{dataset_name}"
    )

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


# ── Main ──────────────────────────────────────────────────────────────


@hydra.main(version_base=None, config_path="../conf", config_name="evaluate")
def main(cfg: DictConfig):
    """Evaluate a single trained baseline."""

    # Discover plugins
    discover_datasets()
    discover_tasks()
    discover_baselines()

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Propagate baseline.n_samples → dataset.n_samples_per_input BEFORE building
    # the datamodule so ReplicatedDataset applies the right replication factor.
    # This mirrors the role of GRW's n_samples knob for the diffusion baseline.
    _baseline_n_samples = cfg.baseline.get("n_samples") if "baseline" in cfg else None
    if _baseline_n_samples is not None:
        from omegaconf import OmegaConf as _OC
        _OC.update(cfg, "dataset.n_samples_per_input", int(_baseline_n_samples))
        print(f"Propagated baseline.n_samples={_baseline_n_samples} → dataset.n_samples_per_input")

    # Load data
    print(f"Loading dataset: {cfg.dataset.name}")
    builder = get_dataset_builder(cfg.dataset.name)()
    datasets = builder.build_datasets(cfg.dataset)
    loaders = builder.build_loaders(cfg.dataset, datasets)

    # Load task with proper dataset-specific initialisation
    task = setup_task(cfg, builder, datasets)
    print(f"Using task: {cfg.task.name if task else 'None'}")

    # Sync task.n_samples_per_input → baseline.n_samples so V2 ensemble
    # reshaping and downstream variance correction operate on the right
    # ensemble axis. Mirrors compare_baselines.py:1705-1708 (the
    # canonical reference). The dataset-side sync at line 112 happens
    # *before* setup_task() is called, so the task-side sync cannot live
    # in the same block.
    if (
        _baseline_n_samples is not None
        and task is not None
        and hasattr(task, "n_samples_per_input")
    ):
        task.n_samples_per_input = int(_baseline_n_samples)
        print(
            f"Propagated baseline.n_samples={_baseline_n_samples} → "
            f"task.n_samples_per_input"
        )

    # Setup evaluator (pass all available loaders)
    hydra_cfg = HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir
    evaluator = UnifiedEvaluator(task, loaders, save_dir=output_dir)

    # Load baseline
    baseline_name = cfg.baseline.get("name", "baseline")
    print(f"\n{'='*60}")
    print(f"Setting up: {baseline_name}")
    print(f"{'='*60}")
    # Convert to plain dict so we can safely extract fields that
    # contain _target_ keys meant as *data* (e.g. diffusion_overrides)
    # without Hydra recursively trying to instantiate them.
    # We also extract checkpoint_path so we can control the init order:
    # set diffusion_overrides *before* the checkpoint triggers model build.
    baseline_dict = OmegaConf.to_container(cfg.baseline, resolve=True)
    _post_init: dict = {}
    for _key in ("diffusion_overrides",):
        val = baseline_dict.pop(_key, None)
        if val:
            _post_init[_key] = val

    # If we have overrides that must be set before checkpoint loading,
    # defer the checkpoint load to after instantiation.
    _deferred_ckpt = None
    if _post_init and baseline_dict.get("checkpoint_path"):
        _deferred_ckpt = baseline_dict.pop("checkpoint_path")

    baseline_cfg_clean = OmegaConf.create(baseline_dict)
    baseline = instantiate(baseline_cfg_clean, device=device)

    # Inject fields that couldn't go through Hydra instantiate, then
    # trigger the deferred checkpoint load.
    for k, v in _post_init.items():
        setattr(baseline, k, v)
    if _deferred_ckpt is not None:
        baseline.load_checkpoint(_deferred_ckpt)

    # Load checkpoint if provided, otherwise fit if needed
    if cfg.get("checkpoint_path") and _deferred_ckpt is None:
        print(f"Loading checkpoint: {cfg.checkpoint_path}")
        baseline.load_checkpoint(cfg.checkpoint_path)
    elif cfg.baseline.get("needs_training", True) and _deferred_ckpt is None:
        print(f"Training {baseline_name}...")
        baseline.fit(loaders["train"], loaders["val"])

        # Save checkpoint
        if baseline.save_dir:
            checkpoint_path = os.path.join(output_dir, f"{baseline_name}_final.pt")
            baseline.save_checkpoint(checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    # Wrap single baseline in a dict for UnifiedEvaluator (same API as compare_baselines)
    baselines = {baseline_name: baseline}

    # Read eval_splits from config (with sensible fallback)
    if "eval_splits" in cfg:
        eval_splits = OmegaConf.to_container(cfg.eval_splits, resolve=True)
    else:
        eval_splits = {"test": None}

    # ── W&B initialisation ────────────────────────────────────────────
    with _init_wandb(cfg, hydra_cfg, baseline_name) as wandb_run:
        print(f"\n{'='*60}")
        print("EVALUATING BASELINE")
        print(f"  Baseline: {baseline_name}")
        print(f"  Splits: { {s: f'max_batches={m}' if m else 'all' for s, m in eval_splits.items()} }")
        if wandb_run is not None:
            print(f"  W&B run: {wandb_run.name} ({wandb_run.url})")
        print(f"{'='*60}")

        results_df = evaluator.compare_baselines(
            baselines, splits=eval_splits, visualize=True,
        )

        # ── Log metrics to W&B ────────────────────────────────────────
        log_results_to_wandb(wandb_run, results_df)

        print(f"\n{'='*60}")
        print("EVALUATION RESULTS")
        print(f"{'='*60}")
        print(results_df.to_string())
        print(f"\nResults saved to {output_dir}/")
        print(f"Visualisations saved to {output_dir}/eval_viz/")


if __name__ == "__main__":
    main()
