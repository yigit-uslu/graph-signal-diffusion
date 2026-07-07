"""Shared helpers for baseline evaluation CLIs (evaluate.py, compare_baselines.py).

Consolidates duplicated logic for task setup, metric-key formatting, and
W&B result logging.
"""
from __future__ import annotations

from hydra.utils import instantiate


# ---------------------------------------------------------------------------
# Task setup
# ---------------------------------------------------------------------------


def setup_task(cfg, builder, datasets):
    """Instantiate the task and inject dataset-specific information.

    For SP500/SP500-cleaned this injects:
      - ``dataset_info`` (legacy path for per-stock stats)
      - ``target_stats`` + ``destandardize_target_fn`` (new API)
    """
    if "task" not in cfg or cfg.task is None:
        return None

    task = instantiate(cfg.task)
    dataset_name = cfg.dataset.name

    # Inject dataset info (legacy path, works for all datasets)
    if hasattr(builder, "get_dataset_info"):
        dataset_info = builder.get_dataset_info()
        if dataset_info is not None and hasattr(task, "set_dataset_info"):
            task.set_dataset_info(dataset_info)
            print("Injected dataset_info into task")

    # SP500-specific: inject target destandardisation (new API)
    if dataset_name in ("sp500", "sp500_cleaned"):
        train_ds = datasets.get("full", datasets.get("train"))
        if train_ds is not None and hasattr(train_ds, "get_target_standardization_stats"):
            try:
                target_stats = train_ds.get_target_standardization_stats()
                if hasattr(task, "set_target_destandardization"):
                    from graph_signal_diffusion.datasets.sp500.dataset import SP500Stocks
                    task.set_target_destandardization(
                        target_stats, SP500Stocks.destandardize_target
                    )
                    print(
                        f"Injected SP500 target destandardisation: "
                        f"{target_stats['target_name']}, "
                        f"scale_factor={target_stats['scale_factor']}"
                    )
            except ValueError as exc:
                print(f"WARNING: Could not inject target destandardisation: {exc}")

    return task


# ---------------------------------------------------------------------------
# Metric-key formatting
# ---------------------------------------------------------------------------


def to_trainer_key(split: str, metric: str) -> str:
    """Convert (split, metric) to the trainer's W&B naming convention.

    The trainer logs task metrics as ``{split}_{metric}``, e.g.:
      - val/crps_mean      → val_crps_mean
      - test/price_rmse    → test_price_rmse
      - train-val/price_mae → train-val_price_mae

    This ensures baseline metrics appear on the same W&B chart axes as
    training curves, enabling automatic horizontal-line overlay when both
    a training run and a baseline run are toggled visible.
    """
    return f"{split}_{metric}"


# ---------------------------------------------------------------------------
# W&B result logging
# ---------------------------------------------------------------------------


def log_results_to_wandb(wandb_run, results_df) -> None:
    """Log the final comparison metrics to the active W&B run.

    Metrics are emitted under trainer-compatible names (``{split}_{metric}``)
    so they appear on the same chart axes as training curves in W&B.  When a
    baseline run and a training run are both selected in the W&B runs table,
    the baseline value shows up as a flat horizontal reference line.

    A secondary namespaced copy (``{baseline}/{split}_{metric}``) is also
    stored in the run summary for unambiguous identification when multiple
    baselines are present in one comparison run.
    """
    if wandb_run is None:
        return

    # Primary: trainer-compatible names for overlay.
    # If multiple baselines are in results_df, last row wins for the primary
    # key — run one W&B run per baseline for clean per-baseline overlay.
    primary_metrics: dict = {}
    summary_extras: dict = {}

    for baseline_name in results_df.index:
        for col in results_df.columns:
            # col is "{split}/{metric}", e.g. "val/crps_mean"
            if "/" not in col:
                continue
            split, metric = col.split("/", 1)
            val = results_df.loc[baseline_name, col]
            if not isinstance(val, (int, float)):
                continue

            trainer_key = to_trainer_key(split, metric)
            primary_metrics[trainer_key] = val

            # Namespaced copy in summary for multi-baseline disambiguation
            ns_key = f"{baseline_name}/{trainer_key}"
            summary_extras[ns_key] = val

    # Single log call — appears as a step-0 point on all training charts
    wandb_run.log(primary_metrics)

    # Populate summary: both primary and namespaced copies
    for k, v in primary_metrics.items():
        wandb_run.summary[k] = v
    for k, v in summary_extras.items():
        wandb_run.summary[k] = v

    print(f"Logged {len(primary_metrics)} metrics to W&B (trainer-compatible names)")
