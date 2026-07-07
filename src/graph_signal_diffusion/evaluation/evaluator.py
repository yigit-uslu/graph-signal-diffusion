# from typing import Dict, List
# import torch
# from torch_geometric.loader import DataLoader
# import pandas as pd

# class UnifiedEvaluator:
#     """Unified evaluation pipeline for comparing baselines."""
    
#     def __init__(self, task, test_loader: DataLoader):
#         self.task = task
#         self.test_loader = test_loader
    
#     def evaluate_baseline(self, baseline, baseline_name: str) -> Dict[str, float]:
#         """Evaluate a single baseline."""
#         print(f"\nEvaluating: {baseline_name}")
#         metrics = baseline.evaluate(self.test_loader, self.task)
#         return metrics
    
#     def compare_baselines(self, baselines: Dict) -> pd.DataFrame:
#         """Compare multiple baselines."""
#         results = {}
#         for name, baseline in baselines.items():
#             results[name] = self.evaluate_baseline(baseline, name)
        
#         df = pd.DataFrame(results).T
#         return df


from typing import Any, Dict, Optional
import torch
from torch_geometric.loader import DataLoader
import pandas as pd
from pathlib import Path

from .metrics import compute_all_metrics
from .visualizations import (
    plot_predictions_vs_actual,
    plot_error_distribution,
    plot_baseline_comparison,
    plot_all_metrics_comparison,
    plot_temporal_error_evolution,
)


def _as_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    try:
        from omegaconf import DictConfig
        if isinstance(value, DictConfig):
            from omegaconf import OmegaConf
            return OmegaConf.to_container(value, resolve=True)
    except ImportError:
        pass
    return {}


class UnifiedEvaluator:
    """Unified evaluation pipeline for comparing baselines.

    Supports evaluating baselines on multiple data splits (train-val, val,
    test) and saving results / visualisations under an ``eval_viz/``
    sub-directory, mirroring the structure used by ``DiffusionTrainer``.
    """

    def __init__(
        self,
        task,
        loaders: Dict[str, DataLoader],
        save_dir: Optional[str] = None,
        plot_style: Optional[dict] = None,
    ):
        self.task = task
        self.loaders = loaders
        self.save_dir = save_dir
        self.plot_style = _as_dict(plot_style)

        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Single-split evaluation
    # ------------------------------------------------------------------

    def evaluate_baseline_on_split(
        self,
        baseline,
        baseline_name: str,
        split: str,
        max_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """Evaluate *baseline* on *split* and return scalar metrics.

        For ensemble-capable baselines (``has_ensemble_evaluate``), delegates
        to ``baseline.evaluate()``.

        Args:
            baseline: Fitted baseline instance.
            baseline_name: Human-readable name (used for directory naming).
            split: Loader key, e.g. ``'train-val'``, ``'val'``, ``'test'``.
            max_batches: Cap batches for quick validation (``None`` = all).

        Returns:
            Dict of metric-name → float.
        """
        # Expose the active baseline/split to the task so task-side
        # paper-figure plotters can gate themselves (no-op otherwise).
        try:
            setattr(self.task, "_active_baseline_name", baseline_name)
            setattr(self.task, "_active_split", split)
        except Exception:
            pass

        loader = self.loaders.get(split)
        if loader is None:
            print(f"  ⚠  Loader for split '{split}' not available — skipping.")
            return {}

        # Build per-split viz directory: <save_dir>/eval_viz/<baseline>/<split>/
        viz_save_dir: Optional[str] = None
        if self.save_dir:
            viz_save_dir = str(
                Path(self.save_dir) / "eval_viz" / baseline_name / split
            )
            Path(viz_save_dir).mkdir(parents=True, exist_ok=True)

        print(f"\n— Evaluating {baseline_name} on [{split}]"
              f"{f' (max_batches={max_batches})' if max_batches else ''} —")

        # --- Ensemble-capable baselines handle evaluation themselves ----------
        if getattr(baseline, "has_ensemble_evaluate", False):
            metrics = baseline.evaluate(
                loader, self.task,
                max_batches=max_batches,
                viz_save_dir=viz_save_dir,
                eval_split_name=split,
            )

            # ── Power-distribution histogram (any baseline that exposes
            # last_power_tensors after evaluate, e.g. FP, WMMSE, Diffusion
            # on WRA tasks).  Gated on the task opting in via
            # ``is_power_allocation_task = True`` so that stock-forecasting
            # tasks never see a spurious power-histogram or power_dist_w1
            # metric.
            last_pt = getattr(baseline, "last_power_tensors", None)
            if last_pt is not None and getattr(self.task, "is_power_allocation_task", False):
                import os
                from scipy.stats import wasserstein_distance as _scipy_w1
                from graph_signal_diffusion.baselines.wireless_resource_allocation.viz import (
                    plot_power_allocation_histogram,
                )
                power_real_np = last_pt["real"]
                power_gen_np  = last_pt["gen"]
                if viz_save_dir is not None:
                    plot_style_plots = _as_dict(self.plot_style.get("plots", {}))
                    power_hist_style = _as_dict(
                        plot_style_plots.get("power_allocation_histogram", {})
                    )
                    w1 = plot_power_allocation_histogram(
                        power_real_np, power_gen_np,
                        save_path=os.path.join(
                            viz_save_dir,
                            f"power_allocation_histogram_{baseline_name}_{split}.pdf",
                        ),
                        baseline_name=baseline_name,
                        plot_style=power_hist_style,
                    )
                else:
                    w1 = float(_scipy_w1(power_real_np, power_gen_np))
                metrics["power_dist_w1"] = w1

            return metrics

        # --- Standard single-sample path for other baselines -----------------

        task_metrics: dict = {}
        count = 0
        all_predictions = []
        all_targets = []

        for data in loader:
            data = data.to(baseline.device)

            predictions = baseline.predict(data)
            data_dict = self.task.prepare_data(data)
            targets = data_dict["samples"]

            B = data.num_graphs
            N = targets.size(0) // B
            predictions = predictions.view(B, -1, N, predictions.size(-1))
            targets = targets.view(B, -1, N, targets.size(-1))

            all_predictions.append(predictions)
            all_targets.append(targets)

            # Evaluate per batch so metadata is correct for each batch
            batch_metrics = self.task.evaluate_samples(
                predictions,
                targets,
                data_dict["metadata"],
                viz_save_dir=viz_save_dir,
            )
            for k, v in batch_metrics.items():
                task_metrics[k] = task_metrics.get(k, 0.0) + v
            count += 1

        metrics = {k: v / max(count, 1) for k, v in task_metrics.items()}

        # compute_all_metrics doesn't use metadata so it's safe to run on the
        # full concatenated tensors for dataset-level structural metrics.
        predictions_all = torch.cat(all_predictions, dim=0)
        targets_all = torch.cat(all_targets, dim=0)
        standard_metrics = compute_all_metrics(
            predictions_all,
            targets_all,
            prefix="",
        )
        metrics.update(standard_metrics)

        return metrics

    # ------------------------------------------------------------------
    # Legacy single-split convenience
    # ------------------------------------------------------------------

    def evaluate_baseline(
        self,
        baseline,
        baseline_name: str,
        visualize: bool = True,  # kept for back-compat
    ) -> Dict[str, float]:
        """Evaluate a baseline on **test** (backward-compatible wrapper)."""
        return self.evaluate_baseline_on_split(
            baseline, baseline_name, split="test",
        )

    # ------------------------------------------------------------------
    # Multi-split comparison
    # ------------------------------------------------------------------

    def compare_baselines(
        self,
        baselines: Dict,
        splits: Optional[Dict[str, Optional[int]]] = None,
        visualize: bool = True,
    ) -> pd.DataFrame:
        """Compare baselines across one or more splits.

        Args:
            baselines: ``{name: fitted_baseline}``.
            splits: ``{split_name: max_batches}``.  Defaults to
                ``{'test': None}`` (full test set) for backward compatibility.
            visualize: Whether to create comparison plots.

        Returns:
            DataFrame with rows = baselines, columns = ``<split>/<metric>``.
        """
        if splits is None:
            splits = {"test": None}

        all_results: Dict[str, Dict[str, float]] = {}

        for name, baseline in baselines.items():
            row: Dict[str, float] = {}
            for split, max_batches in splits.items():
                metrics = self.evaluate_baseline_on_split(
                    baseline, name, split=split, max_batches=max_batches,
                )
                for k, v in metrics.items():
                    row[f"{split}/{k}"] = v
            all_results[name] = row

        df = pd.DataFrame(all_results).T

        if self.save_dir:
            save_comparison_results(
                df,
                self.save_dir,
                visualize=visualize,
                plot_style=self.plot_style,
            )

        return df


# ------------------------------------------------------------------
# Standalone comparison results / visualisation
# ------------------------------------------------------------------

def save_comparison_results(
    results_df: pd.DataFrame,
    save_dir: str,
    visualize: bool = True,
    plot_style: Optional[dict] = None,
) -> None:
    """Save comparison artifacts from a pre-computed results DataFrame.

    Produces:
    * ``baseline_comparison.csv``
    * ``metrics_summary.md``
    * Per-metric bar charts and a radar chart under ``eval_viz/comparison/``

    This function is intentionally decoupled from evaluation so that
    callers with custom evaluation loops (e.g. ``compare_baselines.py``,
    which rebuilds loaders per baseline) can still generate the same
    comparison artifacts.

    Args:
        results_df: DataFrame with baselines as rows and
            ``<split>/<metric>`` as columns.
        save_dir: Root output directory.
        visualize: Whether to create comparison plots.
    """
    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(parents=True, exist_ok=True)

    # ── CSV ───────────────────────────────────────────────────────
    csv_path = save_dir_path / "baseline_comparison.csv"
    results_df.to_csv(csv_path)
    print(f"Saved metrics CSV to {csv_path}")

    # ── Markdown summary ──────────────────────────────────────────
    try:
        md_path = save_dir_path / "metrics_summary.md"
        save_metrics_table_markdown(
            str(csv_path),
            str(md_path),
            metrics=[
                # Tier 1 (primary, daily log returns)
                'return_mae', 'return_rmse', 'return_crps', 'direction_accuracy',
                'volclustering_gap', 'momentum_gap', 'kurtosis_gap',
                # Tier 2 (secondary, closing prices)
                'price_mae', 'price_rmse', 'price_crps',
                # Tier 3 (cumulative return at T)
                'cum_T_mae', 'cum_T_crps', 'cum_T_direction_accuracy',
                # Legacy aliases (backward compatibility)
                'rmse', 'crps_mean', 'crps_std', 'mae', 'mse',
            ],
            format='grouped',
        )
        print(f"Saved markdown summary to {md_path}")
    except Exception as e:
        print(f"Could not save markdown summary: {e}")

    # ── Comparison plots ──────────────────────────────────────────
    if visualize:
        plot_style = _as_dict(plot_style)
        plot_style_plots = _as_dict(plot_style.get("plots", {}))
        bar_style = _as_dict(plot_style_plots.get("comparison_bar", {}))
        radar_style = _as_dict(plot_style_plots.get("comparison_radar", {}))

        plots_dir = save_dir_path / "eval_viz" / "comparison"
        plots_dir.mkdir(parents=True, exist_ok=True)

        for metric in results_df.columns:
            try:
                plot_baseline_comparison(
                    results_df,
                    metric=metric,
                    save_path=str(plots_dir / f"comparison_{metric.replace('/', '_')}.png"),
                    style_cfg=bar_style,
                )
            except Exception as e:
                print(f"Could not plot {metric}: {e}")

        try:
            plot_all_metrics_comparison(
                results_df,
                save_path=str(plots_dir / "comparison_radar.png"),
                style_cfg=radar_style,
            )
        except Exception as e:
            print(f"Could not create radar chart: {e}")


# ------------------------------------------------------------------
# Utility function for extracting metrics table
# ------------------------------------------------------------------

def extract_metrics_table(
    csv_path: str,
    metrics: Optional[list] = None,
    splits: Optional[list] = None,
) -> pd.DataFrame:
    """Extract and format key metrics from baseline_comparison.csv.

    Args:
        csv_path: Path to baseline_comparison.csv file.
        metrics: List of metric names to extract (e.g., ['rmse', 'mae', 'mse']).
                 Defaults to ['rmse', 'crps_mean', 'crps_std', 'mae', 'mse'].
        splits: List of data splits to extract (e.g., ['train-val', 'val', 'test']).
                Defaults to ['train-val', 'val', 'test'].

    Returns:
        DataFrame with baselines as rows and (split, metric) as hierarchical columns,
        formatted for easy reading.

    Example:
        >>> df = extract_metrics_table('outputs/.../baseline_comparison.csv')
        >>> print(df)
    """
    if metrics is None:
        metrics = ['rmse', 'crps_mean', 'crps_std', 'mae', 'mse']
    
    if splits is None:
        splits = ['train-val', 'val', 'test']
    
    # Read the full comparison CSV
    df_full = pd.read_csv(csv_path, index_col=0)
    
    # Extract relevant columns
    columns_to_extract = []
    for split in splits:
        for metric in metrics:
            col_name = f"{split}/{metric}"
            if col_name in df_full.columns:
                columns_to_extract.append(col_name)
    
    if not columns_to_extract:
        raise ValueError(
            f"No matching columns found for splits={splits} and metrics={metrics}. "
            f"Available columns: {list(df_full.columns)}"
        )
    
    # Extract and reorganize
    df_extracted = df_full[columns_to_extract].copy()
    
    # Create multi-index columns for better display
    multi_cols = []
    for col in df_extracted.columns:
        parts = col.split('/')
        if len(parts) == 2:
            multi_cols.append(tuple(parts))
        else:
            multi_cols.append((col, ''))
    
    df_extracted.columns = pd.MultiIndex.from_tuples(multi_cols, names=['Split', 'Metric'])
    
    return df_extracted


def display_metrics_table(
    csv_path: str,
    metrics: Optional[list] = None,
    splits: Optional[list] = None,
    decimals: int = 4,
    format: str = "grouped",  # "grouped" or "flat"
) -> pd.DataFrame:
    """Display key metrics from baseline_comparison.csv in readable table format.

    Args:
        csv_path: Path to baseline_comparison.csv file.
        metrics: List of metric names to extract. 
                 Defaults to ['rmse', 'crps_mean', 'crps_std', 'mae', 'mse'].
        splits: List of data splits. Defaults to ['train-val', 'val', 'test'].
        decimals: Number of decimal places to display.
        format: Display format - "grouped" (by split) or "flat" (single table).

    Example:
        >>> display_metrics_table('outputs/.../baseline_comparison.csv')
    """
    df = extract_metrics_table(csv_path, metrics=metrics, splits=splits)
    
    # Round for display
    df_rounded = df.round(decimals)
    
    print("\n" + "="*80)
    print("Baseline Metrics Comparison")
    print("="*80)
    
    if format == "grouped":
        # Display each split separately for better readability
        for split in df_rounded.columns.get_level_values(0).unique():
            print(f"\n{split.upper()} Split:")
            print("-" * 60)
            split_data = df_rounded[split]
            print(split_data.to_string())
    else:
        # Flatten multi-index columns for single table display
        flat_cols = [f"{split}/{metric}" for split, metric in df_rounded.columns]
        df_flat = df_rounded.copy()
        df_flat.columns = flat_cols
        print(df_flat.to_string())
    
    print("="*80 + "\n")
    
    return df_rounded


def save_metrics_table_markdown(
    csv_path: str,
    output_path: str,
    metrics: Optional[list] = None,
    splits: Optional[list] = None,
    decimals: int = 4,
    format: str = "grouped",
) -> None:
    """Save formatted metrics table to a markdown file.

    Args:
        csv_path: Path to baseline_comparison.csv file.
        output_path: Path to save the markdown file.
        metrics: List of metric names to extract. 
                 Defaults to ['rmse', 'crps_mean', 'crps_std', 'mae', 'mse'].
        splits: List of data splits. Defaults to ['train-val', 'val', 'test'].
        decimals: Number of decimal places to display.
        format: Display format - "grouped" (by split) or "flat" (single table).

    Example:
        >>> save_metrics_table_markdown(
        ...     'outputs/.../baseline_comparison.csv',
        ...     'outputs/.../metrics_summary.md'
        ... )
    """
    df = extract_metrics_table(csv_path, metrics=metrics, splits=splits)
    df_rounded = df.round(decimals)
    
    # Create markdown content
    markdown_lines = []
    markdown_lines.append("# Baseline Metrics Comparison\n")
    markdown_lines.append(f"*Generated from: `{csv_path}`*\n\n")
    
    if format == "grouped":
        # Create separate tables for each split
        for split in df_rounded.columns.get_level_values(0).unique():
            markdown_lines.append(f"## {split.upper()} Split\n")
            split_data = df_rounded[split]
            
            # Convert to markdown table
            md_table = split_data.to_markdown()
            markdown_lines.append(md_table)
            markdown_lines.append("\n")
    else:
        # Create a single flat table
        flat_cols = [f"{split}/{metric}" for split, metric in df_rounded.columns]
        df_flat = df_rounded.copy()
        df_flat.columns = flat_cols
        
        markdown_lines.append("## All Metrics\n")
        md_table = df_flat.to_markdown()
        markdown_lines.append(md_table)
        markdown_lines.append("\n")
    
    # Write to file
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.writelines(markdown_lines)
    
    print(f"\n✓ Saved markdown table to: {output_path}\n")
