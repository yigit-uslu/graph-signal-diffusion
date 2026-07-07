#!/usr/bin/env python
"""Benchmark GRW baseline on SP500 with configurable n_split_chunks.

Usage (matching D-v2f: 3-chunk interleaved split):
    conda run -n torch_env python scripts/stock/sp500/benchmark_grw.py \
        --n-split-chunks 3 --past-window 20 --future-window 5

Usage (matching D-v2e: single-chunk, 500-day window):
    conda run -n torch_env python scripts/stock/sp500/benchmark_grw.py \
        --n-split-chunks 1 --past-window 20 --future-window 5 \
        --train-window-days 500
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from omegaconf import OmegaConf

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description="GRW baseline benchmark on SP500")
    parser.add_argument("--n-split-chunks", type=int, default=3)
    parser.add_argument("--past-window", type=int, default=20)
    parser.add_argument("--future-window", type=int, default=5)
    parser.add_argument("--train-window-days", type=int, default=None,
                        help="Truncate training set to last N samples (None = full)")
    parser.add_argument("--n-grw-samples", type=int, default=20,
                        help="Monte-Carlo samples per input for GRW ensemble")
    parser.add_argument("--batch-size-val", type=int, default=200)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # ── Build minimal config matching D-v2f dataset settings ─────────
    data_root = PROJECT_ROOT / "data" / "sp500" / "cleaned_drop_incomplete_min_coverage_0.95_corr_0.6_sector_bonus_0.05"
    assert data_root.exists(), f"Data root not found: {data_root}"

    cfg = OmegaConf.create({
        "root": str(data_root),
        "name": "sp500_cleaned",
        "past_window": args.past_window,
        "future_window": args.future_window,
        "target_column_name": "DailyLogReturn",
        "pool_ratio": 0.5,
        "top_k": None,
        "min_degree": None,
        "edge_weight_norm": "none",
        "spectral_normalize_edge_weights": True,
        "dynamic_graph_period": None,
        "dynamic_graph_top_k": None,
        "dynamic_graph_min_degree": None,
        "dynamic_graph_method": "pearson",
        "dynamic_graph_edge_weight_threshold": None,
        "dynamic_graph_edge_budget_ratio": None,
        "standardize_target_in_x_for_revin": False,
        # Split settings
        "dataset_split_strategy": "chronological",
        "train_dataset_fraction": 0.8,
        "n_split_chunks": args.n_split_chunks,
        "train_window_days": args.train_window_days,
        # Loader settings — GRW handles its own ensemble, so n_samples_per_input=1
        "n_samples_per_input": 1,
        "shuffle_val": False,
        "batch_size": 32,
        "batch_size_val": args.batch_size_val,
        "num_workers": 4,
        "pin_memory": False,
        "persistent_workers": False,
        "drop_last": False,
        "normalize": False,
    })

    # ── Build dataset + loaders ──────────────────────────────────────
    from graph_signal_diffusion.datasets.sp500.datamodule import SP500Builder

    builder = SP500Builder()
    datasets = builder.build_datasets(cfg)
    loaders = builder.build_loaders(cfg, datasets)

    print(f"\nDataset splits (n_split_chunks={args.n_split_chunks}):")
    for name, ds in datasets.items():
        print(f"  {name:10s}: {len(ds)} samples")

    # ── Instantiate task + inject dataset_info ───────────────────────
    from graph_signal_diffusion.tasks.stock_price_forecasting.evaluator import (
        StockPriceForecastingTaskV2,
    )

    task = StockPriceForecastingTaskV2(
        forecast_horizon=args.future_window,
        n_samples_per_input=args.n_grw_samples,
        valid_datasets=["sp500", "sp500_cleaned"],
    )
    dataset_info = builder.get_dataset_info()
    task.set_dataset_info(dataset_info)

    # ── Fit GRW on training set ──────────────────────────────────────
    from graph_signal_diffusion.baselines.stock_price_forecasting.grw import (
        GeometricRandomWalk,
    )

    grw = GeometricRandomWalk(
        device=args.device,
        n_samples=args.n_grw_samples,
        shrinkage_strength=10.0,
        min_samples_per_stock=20,
    )
    grw.fit(loaders["train"])

    # ── Evaluate on val and test ─────────────────────────────────────
    from graph_signal_diffusion.evaluation.evaluator import UnifiedEvaluator

    evaluator = UnifiedEvaluator(task=task, loaders=loaders)

    results = {}
    for split in ["val", "test"]:
        print(f"\n{'='*60}")
        print(f"  Evaluating GRW on [{split}]")
        print(f"{'='*60}")
        metrics = evaluator.evaluate_baseline_on_split(grw, "grw", split=split)
        results[split] = metrics

    # ── Print summary table ──────────────────────────────────────────
    key_metrics = [
        "return_crps", "return_mae", "return_rmse",
        "coverage_80", "coverage_90",
        "direction_accuracy",
        "price_mae", "price_rmse", "price_crps",
        "cum_T_mae", "cum_T_crps",
    ]

    print(f"\n{'='*70}")
    print(f"  GRW Baseline Results  (n_split_chunks={args.n_split_chunks}, "
          f"past={args.past_window}, future={args.future_window}, "
          f"n_samples={args.n_grw_samples})")
    if args.train_window_days:
        print(f"  train_window_days={args.train_window_days}")
    print(f"{'='*70}")
    print(f"{'Metric':<25s} {'Val':>10s} {'Test':>10s}")
    print("-" * 47)
    for m in key_metrics:
        val_v = results.get("val", {}).get(m, float("nan"))
        test_v = results.get("test", {}).get(m, float("nan"))
        print(f"{m:<25s} {val_v:10.4f} {test_v:10.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
