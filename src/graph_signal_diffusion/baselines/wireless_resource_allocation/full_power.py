from __future__ import annotations
import torch
from typing import Optional, Dict, Any, List
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from graph_signal_diffusion.baselines import BASELINE_REGISTRY
from graph_signal_diffusion.baselines.base import BaseBaseline


@BASELINE_REGISTRY.register("full_power_wra")
class FullPowerBaseline(BaseBaseline):
    """Constant full-power allocation baseline for wireless resource allocation.

    Generates samples that are constant full-power allocation policies: every
    transmitter always uses maximum power P_max.  In the normalized space used
    by the WRA dataset (``y_norm = power/P_max - 0.5``), this corresponds to
    the constant value ``0.5`` for every node and feature dimension.

    This is the WRA analogue of the GRW baseline for StockPriceForecastingTask:
    a parameter-free, non-adaptive reference point that requires no training.

    Attributes:
        has_ensemble_evaluate: Signals to ``UnifiedEvaluator`` that this
            baseline handles its own evaluation via ``evaluate()``.
    """

    has_ensemble_evaluate: bool = True
    # Use loader-level grouped sampling so real-policy references are comparable
    # with diffusion (same n_samples_per_network from dataloader).
    needs_replicated_loader: bool = True

    def __init__(
        self,
        device: str = "cuda",
        n_samples: int = 1,
        log_every_n_batches: int = 10,
        **kwargs,
    ):
        """
        Args:
            device: Device to run computations on.
            n_samples: Samples per network drawn by evaluation dataloaders.
            log_every_n_batches: Print progress every N batches (0 to disable).
        """
        super().__init__(device=torch.device(device), **kwargs)
        self.n_samples = max(1, int(n_samples))
        self.log_every_n_batches = log_every_n_batches

    # ------------------------------------------------------------------
    # Fitting (no-op — full-power requires no training)
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        **kwargs,
    ):
        """No-op: full-power allocation requires no training."""
        print("FP baseline: no training required.")

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, data) -> torch.Tensor:
        """Generate a constant full-power allocation for the batch.

        Returns:
            ``[B, T, N, F]`` tensor of all ``0.5`` (normalised P_max).
        """
        B = data.num_graphs
        N = data.y.size(0) // B
        T = data.y.size(1)
        F = data.y.size(2)
        return torch.full((B, T, N, F), 0.5, device=self.device)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        loader: DataLoader,
        task,
        max_batches: Optional[int] = None,
        viz_save_dir: Optional[str] = None,
        eval_split_name: str = "eval",
    ) -> Dict[str, float]:
        """Evaluate the full-power baseline against ground-truth allocations.

        Follows the same per-batch pattern as ``GeometricRandomWalk.evaluate``:
        evaluate each batch independently via ``task.evaluate_samples()``,
        accumulate the scalar metrics, and average at the end.

        For each batch:
        1. ``task.prepare_data(data)`` → targets ``[B, T, N, F]`` + metadata.
        2. ``predict(data)`` → ``[B, T, N, F]`` (all ``0.5``).
        3. Call ``task.evaluate_samples()`` with predictions and targets.
        4. Accumulate metrics across batches and average.

        Args:
            loader: DataLoader to iterate over.
            task: Task instance with ``prepare_data()`` and ``evaluate_samples()``.
            max_batches: Maximum number of batches to process (``None`` = all).
            viz_save_dir: Directory for per-batch visualisations.
            eval_split_name: Name of the evaluation split (for logging).
        """
        total_batches = (
            len(loader) if max_batches is None else min(max_batches, len(loader))
        )
        batch_limit_str = (
            f" (first {total_batches})" if max_batches is not None else ""
        )

        task_metrics: Dict[str, float] = {}
        task_metric_counts: Dict[str, int] = {}
        count = 0

        # Accumulators for power-distribution diagnostics
        all_real_power: List[torch.Tensor] = []
        all_gen_power:  List[torch.Tensor] = []

        # Accumulator for WRA evaluation records (one list of dicts per batch)
        all_wra_records: List[Any] = []

        print(f"\nEvaluating FP baseline on [{eval_split_name}]{batch_limit_str}...")
        for i, data in tqdm(
            enumerate(loader), desc=f"FP [{eval_split_name}]", total=total_batches
        ):
            data = data.to(self.device)

            data_dict = task.prepare_data(data)
            targets = data_dict["samples"]   # [B, T, N, F]
            metadata = data_dict["metadata"]

            B = targets.size(0)

            predictions = torch.full_like(targets, 0.5)  # [B, T, N, F], all 0.5

            # Per-sample mean normalised power (mean over T, N, F dims)
            all_real_power.append(targets.mean(dim=(1, 2, 3)).detach().cpu())
            all_gen_power.append(predictions.mean(dim=(1, 2, 3)).detach().cpu())

            meta = dict(metadata)
            meta["n_samples_per_input"] = self.n_samples
            meta["batch_size"] = B

            batch_metrics = task.evaluate_samples(
                predictions, targets, meta, viz_save_dir=viz_save_dir
            )
            if hasattr(task, "last_evaluation_records"):
                all_wra_records.extend(task.last_evaluation_records)

            for k, v in batch_metrics.items():
                if k not in task_metrics:
                    task_metrics[k] = 0.0
                    task_metric_counts[k] = 0
                task_metrics[k] += v
                task_metric_counts[k] += 1
            count += 1

            if self.log_every_n_batches and (i + 1) % self.log_every_n_batches == 0:
                print(f"  Processed {i + 1}/{total_batches} batches")

            if max_batches is not None and (i + 1) >= max_batches:
                break

        metrics = {k: v / max(task_metric_counts[k], 1) for k, v in task_metrics.items()}

        # Expose per-sample mean power for UnifiedEvaluator's power histogram
        if all_real_power:
            self.last_power_tensors: dict = {
                "real": torch.cat(all_real_power).numpy(),
                "gen":  torch.cat(all_gen_power).numpy(),
            }

        # Expose WRA evaluation records for cross-baseline percentile-evolution plot
        if all_wra_records:
            self.last_wra_evaluation_records = all_wra_records

        print(f"\nFP [{eval_split_name}]: evaluated {count} batches (averaged metrics over {count} batches)")
        return metrics
