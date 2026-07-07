from __future__ import annotations
import logging
from collections import defaultdict

import numpy as np
import torch
from typing import Optional, Dict, Any, List
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from graph_signal_diffusion.baselines import BASELINE_REGISTRY
from graph_signal_diffusion.baselines.base import BaseBaseline

logger = logging.getLogger(__name__)


@BASELINE_REGISTRY.register("wmmse")
class WMMSEBaseline(BaseBaseline):
    """WMMSE (Weighted Minimum Mean Square Error) baseline for WRA.

    Iteratively maximises the weighted sum-rate for each network at evaluation
    time using the large-scale channel gains ``H_ls``.  The algorithm alternates
    updates on MMSE receiver coefficients, MSE weights, and transmit amplitudes
    (Shi et al., 2011).

    This is an online, non-trainable baseline: ``fit()`` is a no-op, and the
    optimisation is applied independently per network during ``evaluate()``.

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
        num_wmmse_iters: int = 100,
        log_every_n_batches: int = 10,
        **kwargs,
    ):
        """
        Args:
            device: Device to run computations on.
            n_samples: Samples per network drawn by evaluation dataloaders.
            num_wmmse_iters: Number of WMMSE iterations per network.
            log_every_n_batches: Print progress every N batches (0 to disable).
        """
        super().__init__(device=torch.device(device), **kwargs)
        self.n_samples = max(1, int(n_samples))
        self.num_wmmse_iters = num_wmmse_iters
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
        """No-op: WMMSE is solved online per network at evaluation time."""
        print("WMMSE baseline: no training required.")

    # ------------------------------------------------------------------
    # Prediction (fallback — real evaluation happens in evaluate())
    # ------------------------------------------------------------------

    def predict(self, data) -> torch.Tensor:
        """Fallback prediction (full-power) when H_ls is unavailable.

        The meaningful WMMSE evaluation path is via ``evaluate()``, which has
        access to per-network channel metadata.
        """
        B = data.num_graphs
        N = data.y.size(0) // B
        T = data.y.size(1)
        F = data.y.size(2)
        return torch.full((B, T, N, F), 0.5, device=self.device)

    # ------------------------------------------------------------------
    # WMMSE solver
    # ------------------------------------------------------------------

    @staticmethod
    def _wmmse_solve(
        H_pow: np.ndarray,
        associations: np.ndarray,
        P_max: float,
        noise_var: float,
        num_iters: int = 100,
    ) -> np.ndarray:
        """WMMSE algorithm for sum-rate maximisation.

        Iteratively updates MMSE receiver coefficients (u), MSE weights (w),
        and transmit amplitudes (v) to converge to a locally-optimal power
        allocation that maximises the weighted sum-rate.

        Args:
            H_pow: Channel power gains ``|h_{ij}|^2`` in WRA convention,
                shape ``(m, n)`` where ``H_pow[i, j]`` is the gain from TX i
                to RX j.
            associations: Binary TX–RX pairing matrix, shape ``(m, n)``.
                ``associations[i, j] = 1`` means TX i serves RX j.
            P_max: Maximum per-transmitter power constraint.
            noise_var: Noise variance.
            num_iters: Number of alternating-update iterations.

        Returns:
            Per-transmitter power allocations in ``[0, P_max]``, shape ``(m,)``.
        """
        m, n = H_pow.shape
        H_amp = np.sqrt(np.maximum(H_pow, 0.0))  # channel amplitudes

        # TX–RX pairing indices (one-to-one)
        paired_tx = associations.argmax(axis=0)  # (n,) TX serving each RX
        paired_rx = associations.argmax(axis=1)  # (m,) RX served by each TX

        # Direct channel amplitude for each RX
        h_d = H_amp[paired_tx, np.arange(n)]  # (n,)

        # Initialise TX amplitudes
        v = np.ones(m) * np.sqrt(P_max) / 2  # (m,)

        # Initial MMSE receivers and weights
        rx_total = H_pow.T @ (v ** 2) + noise_var  # (n,)
        u = h_d * v[paired_tx] / rx_total           # (n,)
        mse = 1.0 - u * h_d * v[paired_tx]
        w = 1.0 / np.maximum(mse, 1e-10)            # (n,)

        for _ in range(num_iters):
            # --- v update (TX amplitudes) ---
            wu2 = w * (u ** 2)                                  # (n,)
            denom = H_pow @ wu2                                 # (m,)
            numer = w[paired_rx] * u[paired_rx] * h_d[paired_rx]  # (m,)
            v = np.where(denom > 1e-12, numer / denom, 0.0)
            v = np.clip(v, 0.0, np.sqrt(P_max))

            # --- u update (MMSE receivers) ---
            rx_total = H_pow.T @ (v ** 2) + noise_var  # (n,)
            u = h_d * v[paired_tx] / rx_total

            # --- w update (MSE weights) ---
            mse = 1.0 - u * h_d * v[paired_tx]
            w = 1.0 / np.maximum(mse, 1e-10)

        return v ** 2  # per-TX power (m,)

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
        """Evaluate the WMMSE baseline against ground-truth allocations.

        For each batch, runs WMMSE on the large-scale channel gains ``H_ls``
        of every network, produces normalised power allocations, and delegates
        metric computation to ``task.evaluate_samples()``.

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

        print(f"\nEvaluating WMMSE baseline on [{eval_split_name}]{batch_limit_str}...")
        for i, data in tqdm(
            enumerate(loader), desc=f"WMMSE [{eval_split_name}]", total=total_batches
        ):
            data = data.to(self.device)

            data_dict = task.prepare_data(data)
            targets = data_dict["samples"]   # [B, T, N, F]
            metadata = data_dict["metadata"]

            B, T, N, F = targets.shape
            if F != 1:
                raise ValueError(
                    "WMMSEBaseline expects targets with feature dimension F==1 "
                    f"(power-only schema), but got F={F}. "
                    "Update WMMSEBaseline.evaluate() before using multi-feature "
                    "WRA targets."
                )
            system_params = metadata["system_params"]
            P_max = system_params.get("P_max", 1.0)
            noise_var = system_params.get("noise_var", 1e-10)

            predictions = torch.zeros_like(targets)  # [B, T, N, F]

            # Group batch items by (dataset_name, network_id)
            network_to_indices: Dict[tuple, list] = defaultdict(list)
            for idx in range(metadata["batch_size"]):
                key = (metadata["dataset_names"][idx], metadata["network_ids"][idx])
                network_to_indices[key].append(idx)

            for (ds_name, net_id), indices in network_to_indices.items():
                rep_idx = indices[0]

                # Get large-scale channel gains
                h_ls = metadata.get("h_ls_gains", [None] * B)[rep_idx]
                if h_ls is None:
                    logger.warning(
                        "H_ls unavailable for (%s, %s); falling back to full-power.",
                        ds_name, net_id,
                    )
                    for idx in indices:
                        predictions[idx] = 0.5
                    continue

                # Convert to numpy
                if isinstance(h_ls, torch.Tensor):
                    h_ls_np = h_ls.cpu().numpy().astype(np.float64)
                else:
                    h_ls_np = np.asarray(h_ls, dtype=np.float64)

                assoc = metadata["associations"][rep_idx]
                if isinstance(assoc, torch.Tensor):
                    assoc_np = assoc.cpu().numpy().astype(np.float64)
                else:
                    assoc_np = np.asarray(assoc, dtype=np.float64)

                # Run WMMSE
                p_tx = self._wmmse_solve(
                    h_ls_np, assoc_np, P_max, noise_var, self.num_wmmse_iters,
                )  # (m,)

                # Convert per-TX power to per-RX power
                p_rx = assoc_np.T @ p_tx  # (n,)

                # Normalise to [-0.5, 0.5]
                p_norm = p_rx / P_max - 0.5
                p_norm_t = torch.from_numpy(p_norm).float().to(self.device)

                # WMMSE produces a static allocation (independent of time),
                # so broadcast the same per-node power across all T steps.
                # Only feature channel 0 carries power; F>1 is not expected
                # for WRA but would remain zero (harmless) if it ever occurs.
                for idx in indices:
                    predictions[idx, :, :, 0] = p_norm_t.unsqueeze(0).expand(T, -1)

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

        # Expose per-sample mean power for UnifiedEvaluator's power histogram
        if all_real_power:
            self.last_power_tensors: dict = {
                "real": torch.cat(all_real_power).numpy(),
                "gen":  torch.cat(all_gen_power).numpy(),
            }

        # Expose WRA evaluation records for cross-baseline percentile-evolution plot
        if all_wra_records:
            self.last_wra_evaluation_records = all_wra_records

        print(f"\nWMMSE [{eval_split_name}]: evaluated {count} batches (averaged metrics over {count} batches)")
        return metrics
