from __future__ import annotations
import logging
from collections import defaultdict

import torch
from typing import Optional, Dict, Any, List, Hashable
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from graph_signal_diffusion.baselines import BASELINE_REGISTRY
from graph_signal_diffusion.baselines.base import BaseBaseline

logger = logging.getLogger(__name__)


@BASELINE_REGISTRY.register("ap")
class AveragePowerBaseline(BaseBaseline):
    """Average-power baseline for WRA.

    This baseline predicts a constant per-receiver power vector for each
    network, where the vector is the average expert power allocation observed
    in the evaluation split for that network.
    """

    has_ensemble_evaluate: bool = True
    # Use loader-level grouped sampling so AP statistics and references are
    # computed on the same replicated sample set as diffusion/FP comparisons.
    needs_replicated_loader: bool = True

    def __init__(
        self,
        device: str = "cuda",
        n_samples: int = 1,
        log_every_n_batches: int = 10,
        **kwargs,
    ):
        super().__init__(device=torch.device(device), **kwargs)
        self.n_samples = max(1, int(n_samples))
        self.log_every_n_batches = log_every_n_batches

    # ------------------------------------------------------------------
    # Fitting (no-op)
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        **kwargs,
    ):
        """No-op: AP statistics are computed directly from evaluation batches."""
        print("AP baseline: no training required.")

    # ------------------------------------------------------------------
    # Prediction (fallback only)
    # ------------------------------------------------------------------

    def predict(self, data) -> torch.Tensor:
        """Fallback prediction for generic baseline API calls.

        The meaningful AP path is through ``evaluate()``, where per-network
        averages are computed from expert targets in the eval split.
        """
        B = data.num_graphs
        N = data.y.size(0) // B
        T = data.y.size(1)
        F = data.y.size(2)
        return torch.zeros((B, T, N, F), device=self.device)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def _canonical_network_key(dataset_name: Any, network_id: Any) -> tuple[str, Hashable]:
        ds_name = str(dataset_name)
        try:
            net_id = int(network_id)
        except (TypeError, ValueError):
            net_id = str(network_id)
        return (ds_name, net_id)

    def evaluate(
        self,
        loader: DataLoader,
        task,
        max_batches: Optional[int] = None,
        viz_save_dir: Optional[str] = None,
        eval_split_name: str = "eval",
    ) -> Dict[str, float]:
        """Evaluate AP baseline against ground-truth allocations.

        For each network key (dataset_name, network_id) in a batch, AP computes
        the per-receiver mean expert power over that key's samples and broadcasts
        this constant allocation across time for all replicas in the batch.
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
        all_gen_power: List[torch.Tensor] = []

        # Accumulator for WRA evaluation records
        all_wra_records: List[Any] = []

        print(f"\nEvaluating AP baseline on [{eval_split_name}]{batch_limit_str}...")
        for i, data in tqdm(
            enumerate(loader), desc=f"AP [{eval_split_name}]", total=total_batches
        ):
            data = data.to(self.device)

            data_dict = task.prepare_data(data)
            targets = data_dict["samples"]   # [B, T, N, F]
            metadata = data_dict["metadata"]

            B, T, N, F = targets.shape
            if F != 1:
                raise ValueError(
                    "AveragePowerBaseline expects targets with feature dimension F==1 "
                    f"(power-only schema), but got F={F}. "
                    "Update AveragePowerBaseline.evaluate() before using multi-feature "
                    "WRA targets."
                )

            predictions = torch.zeros_like(targets)

            dataset_names = metadata.get("dataset_names", ["unknown"] * B)
            network_ids = metadata.get("network_ids", list(range(B)))

            # Group batch items by (dataset_name, network_id)
            network_to_indices: Dict[tuple[str, Hashable], List[int]] = defaultdict(list)
            for idx in range(B):
                key = self._canonical_network_key(dataset_names[idx], network_ids[idx])
                network_to_indices[key].append(idx)

            # Compute per-network, per-receiver AP vectors and broadcast over time.
            for _, indices in network_to_indices.items():
                # [K, T, N] -> [N]
                mean_per_receiver = targets[indices, :, :, 0].mean(dim=(0, 1))
                mean_per_receiver_t = mean_per_receiver.unsqueeze(0).expand(T, -1)
                for idx in indices:
                    predictions[idx, :, :, 0] = mean_per_receiver_t

            # Per-sample mean normalised power (mean over T, N, F dims)
            all_real_power.append(targets.mean(dim=(1, 2, 3)).detach().cpu())
            all_gen_power.append(predictions.mean(dim=(1, 2, 3)).detach().cpu())

            meta = dict(metadata)
            meta["n_samples_per_input"] = self.n_samples
            meta["batch_size"] = B

            batch_metrics = task.evaluate_samples(
                predictions, targets, meta, viz_save_dir=viz_save_dir,
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

        if all_real_power:
            self.last_power_tensors: dict = {
                "real": torch.cat(all_real_power).numpy(),
                "gen": torch.cat(all_gen_power).numpy(),
            }

        if all_wra_records:
            self.last_wra_evaluation_records = all_wra_records

        print(
            f"\nAP [{eval_split_name}]: evaluated {count} batches "
            f"(averaged metrics over {count} batches)"
        )
        return metrics
