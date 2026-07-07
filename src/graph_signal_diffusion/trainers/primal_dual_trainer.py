"""
Primal-Dual Trainer for Power Allocation GNN.

This module implements the primal-dual training algorithm where:
- Primal variables (power allocations) are learned by a GNN
- Dual variables (Lagrange multipliers) are updated via subgradient method
- Training minimizes the Lagrangian with min-rate constraints
"""

import torch
import torch.optim as optim
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import json
import os
import logging
import abc
from collections import deque

logger = logging.getLogger(__name__)

from ..models.power_allocation_gnn import PowerAllocationGNN
from .dual_optimizer import DualOptimizer
from ..utils.rate_calculator import compute_ergodic_rates
from ..datasets.wra.utils import receiver_to_transmitter_power
from ..datasets.wra.sample_schema import save_pd_samples_npz
from ..datasets.wra.channel_factory import normalize_channel_version
from .pd_visualization import (
    smooth_curve,
    visualize_training_progress,
    visualize_training_progress_by_profile,
    visualize_dual_multipliers,
    visualize_power_allocations,
    _extract_scalar_r_min_from_summary,
    _infer_scalar_r_min_from_summaries,
    _collect_constraint_profile_series,
)


def _to_json_compatible(value):
    """Convert tensors/arrays/scalars recursively into JSON-serializable values."""
    if torch.is_tensor(value):
        if value.ndim == 0:
            return value.item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_json_compatible(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_compatible(v) for v in value]
    return value


class PrimalDualTrainer(abc.ABC):
    """
    Abstract primal-dual trainer for constrained power allocation policies.

    Jointly optimizes:
    1. Primal policy (model parameters) via gradient descent on the Lagrangian
    2. Dual variables (Lagrange multipliers) via projected subgradient ascent

    Generic Lagrangian (minimised over primal, maximised over dual):
        L(x, λ) = -f(x) + λᵀ g(x)
    where f(x) is the per-network objective and g(x) are constraint slacks
    (positive = violation). Subclasses implement the problem-specific physics
    via four abstract methods:

        primal_forward(batch)  →  (primal_vars, forward_ctx)
        compute_constraints(primal_vars, forward_ctx, batch)
                               →  (objective, g, per_user_metrics)
        collect_samples(dataloader)  →  samples
        analyze_sample_quality(samples, dataloader)  →  quality_report

    The base class assembles the Lagrangian, creates λ as a leaf tensor so
    that after loss.backward() lambdas.grad = ∂L/∂λ = g/B (dual subgradient),
    and feeds that gradient to DualOptimizer.update_from_gradients().

    Parameters
    ----------
    model : PowerAllocationGNN
        Policy model for power allocation
    dual_optimizer : DualOptimizer
        Dual variable optimizer
    system_params : dict
        System parameters (P_max, noise_var, etc.)
    learning_rate : float
        Learning rate for the primal optimizer (default: 1e-3)
    max_epochs : int
        Maximum number of training epochs (default: 1000)
    checkpoint_dir : str
        Directory for saving checkpoints (default: 'checkpoints')
    convergence_window : int
        Window size for checking convergence (default: 50)
    convergence_warmup_epochs : int, optional
        Earliest epoch at which convergence checks are enabled. If None, defaults
        to convergence_window.
    convergence_patience : int
        Number of epochs to wait for convergence (default: 10)
    gradient_norm_threshold : float
        Threshold for gradient norm convergence (default: 1e-4)
    dual_variance_threshold : float
        Threshold for dual variance change rate (default: 0.01)
    dual_stationarity_threshold : float
        Threshold on ergodic complementary slackness, i.e.
        mean_i | \bar{λ}_i · \bar{g}_i | over recent epochs, where i indexes
        dual coordinates (default: 0.05).
    violation_fraction_threshold : float
        WRA-specific threshold for constraint violation fraction (default: 0.05)
    violation_fraction_on_model_avg_rates_threshold : float
        WRA-specific threshold for violation fraction on model-averaged metrics
        (default: 0.05)
    mean_violation_slack_on_model_avg_rates_threshold : float
        WRA-specific threshold for mean violation slack on model-averaged metrics
        (default: inf)
    num_samples_per_network : int
        Number of samples to collect per network after convergence (default: 20)
    device : str
        Device for training (default: 'cuda' if available else 'cpu')
    """

    # ------------------------------------------------------------------
    # Abstract interface — subclasses supply the problem-specific physics
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def primal_forward(
        self,
        batch: Batch,
    ) -> Tuple[torch.Tensor, object]:
        """
        Run the policy model and produce decision variables.

        Parameters
        ----------
        batch : Batch
            PyG batch (already on self.device).

        Returns
        -------
        primal_vars : torch.Tensor
            Decision variables, shape (B, d).  For WRA: transmit powers (B, m).
        forward_ctx : object
            Any intermediate tensors needed by compute_constraints to avoid
            recomputation (e.g. the stacked association matrix for WRA).
        """

    @abc.abstractmethod
    def compute_constraints(
        self,
        primal_vars: torch.Tensor,
        forward_ctx: object,
        batch: Batch,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate the objective and constraint slacks.

        Parameters
        ----------
        primal_vars : torch.Tensor
            Decision variables from primal_forward, shape (B, d).
        forward_ctx : object
            Context returned by primal_forward.
        batch : Batch
            PyG batch (already on self.device).

        Returns
        -------
        objective : torch.Tensor
            Per-network objective value to *maximise*, shape (B,).
            For WRA: sum of ergodic rates per network.
        g : torch.Tensor
            Constraint slacks — positive means violated, shape (B, n_con).
            For WRA: r_min − R_i for each receiver i.
        per_user_metrics : torch.Tensor
            Per-user quality measure used for model averaging and logging,
            shape (B, n).  For WRA: ergodic rates (B, n).
        """
    
    def __init__(
        self,
        model: PowerAllocationGNN,
        dual_optimizer: DualOptimizer,
        system_params: Dict,
        learning_rate: float = 1e-3,
        max_epochs: int = 1000,
        checkpoint_dir: str = 'checkpoints',
        convergence_window: int = 50,
        convergence_warmup_epochs: Optional[int] = None,
        convergence_patience: int = 10,
        gradient_norm_threshold: float = 1e-4,
        dual_variance_threshold: float = 0.01,
        dual_stationarity_threshold: float = 0.05,
        violation_fraction_threshold: float = 0.05,
        violation_fraction_on_model_avg_rates_threshold: float = 0.05,
        mean_violation_slack_on_model_avg_rates_threshold: float = float('inf'),
        num_samples_per_network: int = 20,
        moving_avg_window: int = 10,
        dual_update_mode: str = 'step',
        sample_collection_interval: Optional[int] = None,
        trace_logging_enabled: bool = False,
        trace_network_ids: Optional[List[int]] = None,
        trace_receiver_indices: Optional[List[int]] = None,
        trace_include_full_vectors: bool = True,
        trace_write_interval: int = 50,
        trace_output_filename: str = "tracked_network_trace.jsonl",
        channel_version: str = 'v2',
        device: Optional[str] = None,
    ):
        self.model = model
        self.dual_optimizer = dual_optimizer
        self.system_params = system_params
        self.max_epochs = max_epochs
        self.convergence_window = convergence_window
        self.convergence_warmup_epochs = (
            convergence_window if convergence_warmup_epochs is None else int(convergence_warmup_epochs)
        )
        if self.convergence_warmup_epochs < 0:
            raise ValueError("convergence_warmup_epochs must be >= 0")
        self.convergence_patience = convergence_patience
        self.gradient_norm_threshold = gradient_norm_threshold
        self.dual_variance_threshold = dual_variance_threshold
        self.dual_stationarity_threshold = dual_stationarity_threshold
        self.violation_fraction_threshold = violation_fraction_threshold
        self.violation_fraction_on_model_avg_rates_threshold = violation_fraction_on_model_avg_rates_threshold
        self.mean_violation_slack_on_model_avg_rates_threshold = (
            mean_violation_slack_on_model_avg_rates_threshold
        )
        self._warned_nonpositive_alpha_dual = False
        self._warned_nonpositive_p_max = False
        self.num_samples_per_network = num_samples_per_network
        self.moving_avg_window = moving_avg_window
        self.dual_update_mode = dual_update_mode  # 'epoch' or 'step'
        self.channel_version = normalize_channel_version(channel_version, default='v2')
        
        
        # Set device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        self.model.to(self.device)
        
        # Optimizer for GNN parameters (primal) uses trainable parameters only.
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError("No trainable model parameters found for primal optimizer.")
        self.optimizer = optim.Adam(trainable_params, lr=learning_rate)
        
        # Checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Model checkpoint subfolder for Polyak-style sampling
        self.model_checkpoint_dir = self.checkpoint_dir / "model_chkpts"
        self.model_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.epoch = 0
        self.training_history = {
            'loss': [],
            'avg_per_network_min_rate': [],
            'global_min_rate': [],
            'mean_rate': [],
            'global_5th_percentile_rate': [],
            'mean_slack': [],
            'mean_violation_slack': [],
            'violation_fraction': [],
            'mean_dual_subgradient': [],
            'positive_subgradient_fraction': [],
            'projected_dual_residual': [],
            'violation_fraction_on_model_avg_rates': [],
            'mean_violation_slack_on_model_avg_rates': [],
            'normalized_power_p5': [],
            'normalized_power_p25': [],
            'normalized_power_p50': [],
            'normalized_power_p75': [],
            'normalized_power_p90': [],
            'normalized_power_p95': [],
            'normalized_power_p99': [],
            'mean_lambda': [],
            'std_lambda': [],
            'gradient_norm': [],
            # Convergence-time scalar from check_convergence() using ergodic buffers.
            'ergodic_complementary_slackness': [],
            'ergodic_complementary_slackness_abs': [],
        }
        
        # Buffers for model averaging (stores per-receiver rates for last M epochs).
        # deque(maxlen=N) evicts the oldest entry automatically in O(1).
        self.window_5M = 5 * moving_avg_window
        self.window_25M = 25 * moving_avg_window
        self.rate_buffer          = deque(maxlen=moving_avg_window)
        self.network_sizes_buffer = deque(maxlen=moving_avg_window)
        self.slack_buffer         = deque(maxlen=moving_avg_window)
        self.rate_buffer_5M          = deque(maxlen=self.window_5M)
        self.network_sizes_buffer_5M = deque(maxlen=self.window_5M)
        self.slack_buffer_5M         = deque(maxlen=self.window_5M)
        self.rate_buffer_25M          = deque(maxlen=self.window_25M)
        self.network_sizes_buffer_25M = deque(maxlen=self.window_25M)
        self.slack_buffer_25M         = deque(maxlen=self.window_25M)
        # Convergence tracking
        self.convergence_met_count = 0
        self.converged = False
        # Ergodic dual stationarity: rolling buffers of per-receiver duals and
        # constraint slacks for complementary-slackness evaluation on windowed
        # averages (λ̄, ḡ).
        self._ergodic_dual_buffer = deque(maxlen=convergence_window)
        self._ergodic_slack_buffer = deque(maxlen=convergence_window)
        self._ergodic_buffers_updated_this_epoch = False

        # In-memory caches for visualization (avoids re-reading growing JSONL files each epoch).
        # dual: flat list of all history dicts; primal: metadata dict + per-network epoch deques.
        self._dual_history_entries: list = []
        self._primal_metadata_entries: dict = {}       # {network_id: associations as list}
        self._primal_extra_metadata_entries: dict = {}  # {network_id: extra metadata for primal_history.jsonl}
        self._primal_history_window = max(5 * moving_avg_window, num_samples_per_network)
        self._primal_epoch_entries: dict = {}          # {network_id: deque(maxlen=window)}
        # Rewrite primal history less frequently to reduce JSON serialization + disk I/O overhead.
        self.primal_history_write_interval = 50
        
        # Checkpoint buffer for Polyak-style sampling
        # Store last num_samples_per_network checkpoints for diverse sampling
        self.checkpoint_paths = []  # Rolling buffer of checkpoint paths
        self.sample_checkpoint_frequency = max(1, int(moving_avg_window / num_samples_per_network))
        
        # Continuous sample collection interval
        # Default: collect every num_samples_per_network * sample_checkpoint_frequency epochs so that consecutive rewrites sample non-overlapping checkpoints
        if sample_collection_interval is None:
            self.sample_collection_interval = num_samples_per_network * self.sample_checkpoint_frequency
        else:
            self.sample_collection_interval = sample_collection_interval

        # Optional full-trajectory trace logging for selected networks.
        network_ids_raw = trace_network_ids or []
        self.trace_network_ids = sorted({int(net_id) for net_id in network_ids_raw})
        self.trace_network_id_set = set(self.trace_network_ids)
        receiver_indices_raw = trace_receiver_indices or []
        self.trace_receiver_indices = sorted({int(idx) for idx in receiver_indices_raw})
        self.trace_include_full_vectors = bool(trace_include_full_vectors)
        self.trace_write_interval = max(1, int(trace_write_interval))
        trace_name = str(trace_output_filename).strip()
        if not trace_name:
            trace_name = "tracked_network_trace.jsonl"
        self.trace_output_path = self.checkpoint_dir / trace_name
        self.trace_output_path.parent.mkdir(parents=True, exist_ok=True)
        self.trace_logging_enabled = bool(trace_logging_enabled and self.trace_network_id_set)
        if trace_logging_enabled and not self.trace_network_id_set:
            logger.warning(
                "trace_logging_enabled=True but trace_network_ids is empty; trace logging disabled."
            )

        # Rolling window sizes reused for tracked-network rate averages.
        trace_windows = sorted({
            int(self.moving_avg_window),
            int(self.window_5M),
            int(self.window_25M),
        })
        self.trace_window_sizes = [w for w in trace_windows if w > 0] or [1]

        # In-memory state for tracked-network trace buffering.
        self._trace_metadata_written: set[int] = set()
        self._trace_entries_buffer: list[dict] = []
        self._trace_rate_window_state: dict[int, dict[int, dict[str, object]]] = {}
        self._trace_receiver_indices_by_network: dict[int, list[int]] = {}
        self._trace_receiver_index_warning_emitted: set[int] = set()

    def _network_ids_from_batch(self, batch: Batch) -> torch.Tensor:
        """Extract network IDs as a long tensor on self.device."""
        raw_ids = batch.network_id
        if torch.is_tensor(raw_ids):
            return raw_ids.to(self.device, dtype=torch.long).reshape(-1)
        return torch.tensor(
            [nid.item() if torch.is_tensor(nid) else int(nid) for nid in raw_ids],
            dtype=torch.long,
            device=self.device,
        )

    def _r_min_batch(self, network_ids: torch.Tensor, num_receivers: int) -> torch.Tensor:
        """Return per-network/per-receiver thresholds with shape (B, num_receivers)."""
        r_min_batch = self.dual_optimizer.r_min_per_network(network_ids)
        if r_min_batch.shape != (network_ids.shape[0], num_receivers):
            raise ValueError(
                "Dual optimizer returned unexpected r_min batch shape: "
                f"{tuple(r_min_batch.shape)} vs "
                f"({network_ids.shape[0]}, {num_receivers})."
            )
        return r_min_batch.to(self.device)

    def _r_min_for_network(self, network_id: int, num_receivers: int) -> torch.Tensor:
        """Return per-receiver threshold vector for one network, shape (num_receivers,)."""
        net_ids = torch.tensor([int(network_id)], dtype=torch.long, device=self.device)
        return self._r_min_batch(net_ids, num_receivers)[0]

    def _r_min_scalar_for_logging(self) -> Optional[float]:
        """Return scalar threshold for homogeneous runs, else None."""
        return self.dual_optimizer.scalar_r_min_or_none()

    def _r_min_summary(self) -> Dict[str, object]:
        """Return canonical summary statistics for min-rate thresholds."""
        table_tensor = self.dual_optimizer.r_min_table.detach().float()
        scalar = self.dual_optimizer.scalar_r_min_or_none()
        is_scalar = bool(self.dual_optimizer.is_scalar_r_min())
        return {
            'r_min': scalar,
            'r_min_is_scalar': is_scalar,
            'r_min_min': float(table_tensor.min().item()),
            'r_min_max': float(table_tensor.max().item()),
            'r_min_mean': float(table_tensor.mean().item()),
        }

    def _resolve_trace_receiver_indices(
        self,
        network_id: int,
        num_receivers: int,
    ) -> list[int]:
        """Resolve valid receiver indices for tracked-node trace output."""
        cached = self._trace_receiver_indices_by_network.get(int(network_id))
        if cached is not None:
            return list(cached)

        if num_receivers <= 0:
            self._trace_receiver_indices_by_network[int(network_id)] = []
            return []

        requested = (
            list(self.trace_receiver_indices)
            if self.trace_receiver_indices
            else list(range(min(2, num_receivers)))
        )
        valid = [idx for idx in requested if 0 <= idx < num_receivers]
        dropped = [idx for idx in requested if idx not in valid]

        if not valid:
            # Ensure downstream consumers always have at least one valid receiver.
            valid = [0]

        if dropped and int(network_id) not in self._trace_receiver_index_warning_emitted:
            logger.warning(
                "Trace receiver indices %s out of range for network %d (num_receivers=%d); "
                "using %s instead.",
                dropped,
                int(network_id),
                num_receivers,
                valid,
            )
            self._trace_receiver_index_warning_emitted.add(int(network_id))

        self._trace_receiver_indices_by_network[int(network_id)] = list(valid)
        return list(valid)

    def _update_trace_rate_windows(
        self,
        network_id: int,
        rates: torch.Tensor,
    ) -> dict[str, list[float]]:
        """
        Update rolling per-network rate windows and return current averages.

        The returned dictionary is keyed by window size as a string.
        """
        net_id = int(network_id)
        net_state = self._trace_rate_window_state.setdefault(net_id, {})
        averages: dict[str, list[float]] = {}
        rates_cpu = rates.detach().float().cpu()

        for window in self.trace_window_sizes:
            state = net_state.get(window)
            if state is None:
                state = {
                    'values': deque(maxlen=window),
                    'sum': torch.zeros_like(rates_cpu),
                }
                net_state[window] = state

            values: deque = state['values']
            running_sum: torch.Tensor = state['sum']
            if values.maxlen is not None and len(values) == values.maxlen and len(values) > 0:
                running_sum.sub_(values[0])

            values.append(rates_cpu.clone())
            running_sum.add_(rates_cpu)
            averages[str(window)] = (running_sum / float(len(values))).tolist()

        return averages

    def _append_trace_entry(self, entry: dict) -> None:
        """Buffer one tracked-network trace entry and flush periodically."""
        self._trace_entries_buffer.append(entry)
        if len(self._trace_entries_buffer) >= self.trace_write_interval:
            self._flush_trace_history(force=False)

    def _flush_trace_history(self, force: bool = False) -> None:
        """Flush buffered tracked-network trace entries to JSONL."""
        if not self.trace_logging_enabled:
            return
        if not self._trace_entries_buffer:
            return
        if not force and len(self._trace_entries_buffer) < self.trace_write_interval:
            return

        with open(self.trace_output_path, "a") as f:
            for entry in self._trace_entries_buffer:
                f.write(json.dumps(entry) + "\n")
        self._trace_entries_buffer.clear()

    def _record_tracked_network_epoch(
        self,
        epoch: int,
        tracked_epoch_payload: dict[int, dict[str, object]],
    ) -> None:
        """Record one epoch of full trajectory diagnostics for tracked networks."""
        if not self.trace_logging_enabled:
            return
        if not tracked_epoch_payload:
            return

        for net_id in sorted(tracked_epoch_payload):
            payload = tracked_epoch_payload[net_id]
            network_id = int(net_id)
            network_seed = int(payload.get('network_seed', network_id))
            rates = payload['rates'].detach().float().cpu()
            slacks = payload['slacks'].detach().float().cpu()
            powers = payload['power'].detach().float().cpu()
            associations = payload['associations'].detach().float().cpu()
            if associations.ndim == 2 and associations.shape[0] == powers.numel():
                receiver_powers = torch.matmul(associations.transpose(0, 1), powers)
            else:
                receiver_powers = torch.zeros_like(rates)

            receiver_indices = self._resolve_trace_receiver_indices(
                network_id=network_id,
                num_receivers=int(rates.numel()),
            )

            net_id_tensor = torch.tensor([network_id], dtype=torch.long, device=self.device)
            duals = self.dual_optimizer.get_duals(net_id_tensor)[0].detach().float().cpu()
            windowed_avg_rates = self._update_trace_rate_windows(network_id, rates)

            if network_id not in self._trace_metadata_written:
                metadata_entry = {
                    'type': 'metadata',
                    'network_id': network_id,
                    'network_seed': network_seed,
                    'associations': associations.tolist(),
                    'r_min_per_receiver': self._r_min_for_network(
                        network_id=network_id,
                        num_receivers=int(rates.numel()),
                    ).detach().cpu().tolist(),
                    'tracked_receiver_indices': receiver_indices,
                    'window_sizes': list(self.trace_window_sizes),
                    'include_full_vectors': bool(self.trace_include_full_vectors),
                }

                # Cross-channel gains matrix (n x n) for interference analysis.
                # Entry (j1, j2) = total channel gain from TX(s) serving j1 to
                # receiver j2.  Diagonal = direct signal; off-diagonal = interference.
                # Use H_l if available; fall back to time-averaged H_instantaneous.
                H_l = payload.get('H_l')
                if H_l is None:
                    H_inst = payload.get('H_instantaneous')
                    if H_inst is not None:
                        H_l = H_inst.float().mean(dim=0)  # (T, m, n) → (m, n)
                if H_l is not None and associations.ndim == 2:
                    # cross = A^T @ H_l → (n, n)
                    cross = torch.matmul(associations.transpose(0, 1), H_l.float())
                    metadata_entry['cross_channel_gains'] = cross.tolist()

                extra_hook = getattr(self, "_extra_sample_metadata", None)
                if callable(extra_hook):
                    try:
                        extra_meta = extra_hook(network_id)
                    except Exception as exc:
                        logger.debug(
                            "Skipping extra trace metadata for network %d due to error: %s",
                            network_id,
                            exc,
                        )
                        extra_meta = {}
                    if isinstance(extra_meta, dict):
                        metadata_entry.update(extra_meta)
                self._append_trace_entry(_to_json_compatible(metadata_entry))
                self._trace_metadata_written.add(network_id)

            selected_windowed_avg_rates = {
                window_key: [float(values[idx]) for idx in receiver_indices]
                for window_key, values in windowed_avg_rates.items()
            }
            selected_trace = {
                'receiver_indices': receiver_indices,
                'receiver_power_allocations': [
                    float(receiver_powers[idx].item()) for idx in receiver_indices
                ],
                'ergodic_rates': [float(rates[idx].item()) for idx in receiver_indices],
                'dual_multipliers': [float(duals[idx].item()) for idx in receiver_indices],
                'constraint_slacks': [float(slacks[idx].item()) for idx in receiver_indices],
                'windowed_avg_rates': selected_windowed_avg_rates,
            }

            epoch_entry = {
                'type': 'epoch_trace',
                'epoch': int(epoch),
                'network_id': network_id,
                'network_seed': network_seed,
                'selected_receiver_trace': selected_trace,
            }
            if self.trace_include_full_vectors:
                epoch_entry.update({
                    'power_allocations': powers.tolist(),
                    'receiver_power_allocations': receiver_powers.tolist(),
                    'ergodic_rates': rates.tolist(),
                    'dual_multipliers': duals.tolist(),
                    'constraint_slacks': slacks.tolist(),
                    'windowed_avg_rates': windowed_avg_rates,
                })
            self._append_trace_entry(epoch_entry)

    def _constraint_profile_info(self, network_id: int) -> Optional[Dict[str, object]]:
        """Optional hook for subclasses to map expanded IDs to profile metadata."""
        return None

    def _build_constraint_profile_epoch_metrics(
        self,
        epoch_network_ids: List[int],
        epoch_rates: List[torch.Tensor],
        epoch_slacks: List[torch.Tensor],
    ) -> list[dict]:
        """
        Aggregate profile-specific metrics for one epoch.

        Returns empty list for trainers without profile metadata.
        """
        if not epoch_network_ids:
            return []

        profile_buckets: dict[int, dict[str, object]] = {}
        for net_id, rates, slacks in zip(epoch_network_ids, epoch_rates, epoch_slacks):
            profile_info = self._constraint_profile_info(int(net_id))
            if profile_info is None:
                continue

            profile_id = int(profile_info['constraint_profile_id'])
            bucket = profile_buckets.setdefault(
                profile_id,
                {
                    'name': profile_info.get('constraint_profile_name'),
                    'rates': [],
                    'slacks': [],
                    'r_min_rows': [],
                },
            )
            bucket['rates'].append(rates.detach().float().cpu())
            bucket['slacks'].append(slacks.detach().float().cpu())
            bucket['r_min_rows'].append((rates + slacks).detach().float().cpu())

        if not profile_buckets:
            return []

        results: list[dict] = []
        for profile_id in sorted(profile_buckets):
            bucket = profile_buckets[profile_id]
            rates_tensor = torch.stack(bucket['rates'])  # (N, n)
            slacks_tensor = torch.stack(bucket['slacks'])  # (N, n)
            r_min_tensor = torch.stack(bucket['r_min_rows'])  # (N, n)

            flat_rates = rates_tensor.reshape(-1)
            flat_slacks = slacks_tensor.reshape(-1)
            violations = torch.clamp(flat_slacks, min=0.0)

            per_network_mins = rates_tensor.min(dim=1).values

            first_r_min = float(r_min_tensor.reshape(-1)[0].item())
            r_min_is_scalar = bool(
                torch.allclose(
                    r_min_tensor,
                    torch.full_like(r_min_tensor, first_r_min),
                    rtol=1e-6,
                    atol=1e-8,
                )
            )
            scalar_r_min = first_r_min if r_min_is_scalar else None

            results.append(
                {
                    'constraint_profile_id': int(profile_id),
                    'constraint_profile_name': (
                        str(bucket['name']) if bucket['name'] is not None else None
                    ),
                    'mean_rate': float(flat_rates.mean().item()),
                    'avg_per_network_min_rate': float(per_network_mins.mean().item()),
                    'global_min_rate': float(flat_rates.min().item()),
                    'global_5th_percentile_rate': float(torch.quantile(flat_rates, 0.05).item()),
                    'violation_fraction': float((flat_slacks > 0.0).float().mean().item()),
                    'mean_violation_slack': float(violations.mean().item()),
                    'r_min': scalar_r_min,
                    'r_min_is_scalar': r_min_is_scalar,
                    'r_min_min': float(r_min_tensor.min().item()),
                    'r_min_max': float(r_min_tensor.max().item()),
                    'r_min_mean': float(r_min_tensor.mean().item()),
                }
            )

        return results

    def _select_visualization_network_ids(self, max_base_networks: int = 2) -> list[int]:
        """
        Select representative network IDs for visualization.

        Base behavior picks the first ``max_base_networks`` network IDs currently
        available in primal history metadata.
        """
        available_ids = sorted(int(net_id) for net_id in self._primal_metadata_entries.keys())
        return available_ids[:max_base_networks]

    def _visualization_network_label(self, network_id: int) -> str:
        """Human-readable label for visualization titles."""
        return f"Network {int(network_id)}"
    
    def compute_lagrangian_loss(
        self,
        batch: Batch,
        network_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Compute the Lagrangian loss for a batch of networks.

        Uses the abstract primal_forward / compute_constraints methods so the
        training loop is fully problem-agnostic.

        λ is created as a *leaf tensor* with requires_grad=True.  After the
        caller runs loss.backward(), lambdas.grad = ∂L/∂λ = g / B, i.e. the
        batch-mean constraint slack.  Rescaling by B recovers the per-network
        subgradient used by DualOptimizer.update_from_gradients().

        Returns
        -------
        loss : torch.Tensor
            Scalar Lagrangian loss.
        lambdas : torch.Tensor
            Leaf tensor (B, n_con), requires_grad=True.
            After backward: lambdas.grad ≈ g / B.
        g : torch.Tensor
            Detached constraint slacks (B, n_con), positive = violated.
        per_user_metrics : torch.Tensor
            Detached per-user quality measures (B, n), e.g. ergodic rates.
        primal_vars : torch.Tensor
            Detached decision variables (B, d), e.g. transmit powers.
        stats : dict
            Scalar diagnostics for logging.
        """
        batch = batch.to(self.device)
        batch_size = batch.num_graphs
        if network_ids is None:
            network_ids = self._network_ids_from_batch(batch)

        # λ as leaf tensor: after loss.backward(), lambdas.grad = ∂L/∂λ = g / B.
        lambdas = self.dual_optimizer.get_duals(network_ids).detach().requires_grad_(True)

        # Problem-specific forward pass and constraint evaluation.
        primal_vars, forward_ctx = self.primal_forward(batch)
        objective, g, per_user_metrics = self.compute_constraints(
            primal_vars, forward_ctx, batch
        )

        # Generic Lagrangian: L = mean_b [ -f_b + λ_b^T g_b ]
        loss = (-objective + (lambdas * g).sum(dim=1)).mean()

        # Scalar diagnostics (problem-agnostic names; callers may interpret as rates).
        g_det  = g.detach()
        pm_det = per_user_metrics.detach()
        pv_det = primal_vars.detach()
        all_metrics    = pm_det.reshape(-1)
        min_metrics    = pm_det.min(dim=1).values
        slacks_flat    = g_det.reshape(-1)
        violations_flat = torch.clamp(slacks_flat, min=0.0)
        alpha_dual = float(self.dual_optimizer.alpha_dual)
        denom = alpha_dual
        if alpha_dual <= 0.0:
            denom = 1e-12
            if not self._warned_nonpositive_alpha_dual:
                logger.warning(
                    "Dual learning rate alpha_dual=%s is non-positive; "
                    "projected_dual_residual uses epsilon denominator.",
                    alpha_dual,
                )
                self._warned_nonpositive_alpha_dual = True
        lambda_curr = lambdas.detach()
        projected = torch.clamp(lambda_curr + alpha_dual * g_det, min=0.0)
        projected_dual_residual = ((projected - lambda_curr).abs() / denom).mean().item()

        stats = {
            'loss': loss.item(),
            'avg_per_network_min_rate': min_metrics.mean().item(),
            'global_min_rate': min_metrics.min().item(),
            'mean_rate': all_metrics.mean().item(),
            'global_5th_percentile_rate': torch.quantile(all_metrics, 0.05).item(),
            'mean_slack': slacks_flat.mean().item(),
            'mean_violation_slack': violations_flat.mean().item(),
            'violation_fraction': (violations_flat > 0).float().mean().item(),
            'projected_dual_residual': projected_dual_residual,
        }

        return loss, lambdas, g_det, pm_det, pv_det, stats
    
    def train_epoch(
        self,
        dataloader: DataLoader,
    ) -> Dict:
        """
        Train for one epoch.
        
        Parameters
        ----------
        dataloader : DataLoader
            DataLoader providing batches of network data
        
        Returns
        -------
        epoch_stats : dict
            Statistics for this epoch
        all_epoch_rates : list[torch.Tensor]
            Per-network per-user metrics for this epoch, sorted by network_id.
        all_epoch_dual_subgradients : list[torch.Tensor]
            Per-network dual subgradients g for this epoch, sorted by network_id.
        all_epoch_net_ids : list[int]
            Sorted network IDs aligned with all_epoch_rates/all_epoch_dual_subgradients.
        tracked_epoch_payload : dict[int, dict]
            Per-network tensors for full trajectory logging of tracked networks.
        """
        self.model.train()
        
        epoch_stats = {
            'loss': [],
            'avg_per_network_min_rate': [],
            'global_min_rate': [],
            'mean_rate': [],
            'global_5th_percentile_rate': [],
            'mean_slack': [],
            'mean_violation_slack': [],
            'violation_fraction': [],
            'mean_dual_subgradient': [],
            'positive_subgradient_fraction': [],
            'projected_dual_residual': [],
            'gradient_norm': [],
            'normalized_power_p5': [],
            'normalized_power_p25': [],
            'normalized_power_p50': [],
            'normalized_power_p75': [],
            'normalized_power_p90': [],
            'normalized_power_p95': [],
            'normalized_power_p99': [],
        }
        
        epoch_dual_stats = None
        # For epoch-based dual updates, accumulate constraint slacks across batches.
        if self.dual_update_mode == 'epoch':
            epoch_network_ids = []
            epoch_gs = []  # list of (B, n) detached constraint-slack tensors

        # Collect all per-user metrics across epoch for percentile stats and model averaging.
        # Track network IDs alongside so we can sort for consistent cross-epoch ordering.
        all_epoch_net_ids = []
        all_epoch_rates = []
        all_epoch_dual_subgradients = []
        all_epoch_normalized_powers = []
        p_max_normalizer = self.system_params.get('P_max_watts', self.system_params.get('P_max', 1.0))
        try:
            p_max_normalizer = float(p_max_normalizer)
        except (TypeError, ValueError):
            p_max_normalizer = 1.0
        if p_max_normalizer <= 0.0:
            if not self._warned_nonpositive_p_max:
                logger.warning(
                    "Non-positive P_max=%s detected; normalized power diagnostics will use 1.0.",
                    p_max_normalizer,
                )
                self._warned_nonpositive_p_max = True
            p_max_normalizer = 1.0
        # Collect per-graph (net_id, power, metrics, assoc) for visualization.
        # Only on checkpointing epochs to amortise the numpy conversion cost.
        collect_viz = (self.epoch % self.sample_checkpoint_frequency == 0 and self.epoch > 0)
        network_data_for_viz = []
        tracked_epoch_payload: dict[int, dict[str, object]] = {}

        for data_batch in tqdm(dataloader, desc=f"Epoch {self.epoch}", disable=True):
            # Pre-extract network IDs once; pass to compute_lagrangian_loss to
            # avoid a second extraction inside the Lagrangian computation.
            network_ids_tensor = self._network_ids_from_batch(data_batch)

            # Forward pass and compute Lagrangian loss.
            # lambdas is a leaf tensor — after backward, lambdas.grad = g / B.
            loss, lambdas, g, per_user_metrics, primal_vars, stats = (
                self.compute_lagrangian_loss(data_batch, network_ids=network_ids_tensor)
            )

            # Backward pass (populates both model param grads and lambdas.grad).
            self.optimizer.zero_grad()
            loss.backward()

            # Compute gradient norm and clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0
            )
            stats['gradient_norm'] = grad_norm.item()

            # Update primal variables (GNN parameters)
            self.optimizer.step()

            # Dual update — subgradients come from autograd, not handwritten formula.
            # lambdas.grad = g / B  ⟹  lambdas.grad * B = g = r_min_batch - rates.
            B = data_batch.num_graphs
            if self.dual_update_mode == 'step':
                if self.dual_optimizer.step():
                    dual_stats = self.dual_optimizer.update_from_gradients(
                        network_ids=network_ids_tensor,
                        gradients=lambdas.grad * B,
                    )
                    # Keep primal batch metrics authoritative; attach only non-overlapping
                    # dual update diagnostics to avoid semantic collisions.
                    for key, value in dual_stats.items():
                        if key not in stats:
                            stats[key] = value
            else:
                # Epoch-based: accumulate constraint slacks for end-of-epoch update.
                epoch_network_ids.extend(network_ids_tensor.tolist())
                epoch_gs.append(g)  # g is already detached

            # Accumulate statistics
            for key in epoch_stats:
                if key in stats:
                    epoch_stats[key].append(stats[key])

            # Collect all per-user metrics for epoch-level percentile stats and model averaging.
            for idx in range(B):
                all_epoch_net_ids.append(network_ids_tensor[idx].item())
                all_epoch_rates.append(per_user_metrics[idx])  # (n,), already detached
                all_epoch_dual_subgradients.append(g[idx])  # (n,), already detached
            all_epoch_normalized_powers.append(
                (primal_vars.detach().reshape(-1).cpu() / p_max_normalizer)
            )

            # Full trajectory payload for user-selected networks.
            if self.trace_logging_enabled and self.trace_network_id_set:
                raw_seed = getattr(data_batch, "network_seed", None)
                for idx in range(B):
                    net_id = int(network_ids_tensor[idx].item())
                    if net_id not in self.trace_network_id_set:
                        continue
                    if raw_seed is None:
                        network_seed = net_id
                    else:
                        seed_value = raw_seed[idx]
                        network_seed = (
                            int(seed_value.item())
                            if torch.is_tensor(seed_value)
                            else int(seed_value)
                        )
                    payload_entry = {
                        'network_seed': int(network_seed),
                        'power': primal_vars[idx].detach().cpu(),
                        'rates': per_user_metrics[idx].detach().cpu(),
                        'slacks': g[idx].detach().cpu(),
                        'associations': data_batch.associations[idx].detach().cpu(),
                    }
                    # Attach channel data for one-time interference matrix computation.
                    H_l_attr = getattr(data_batch, 'H_l', None)
                    if H_l_attr is not None:
                        payload_entry['H_l'] = H_l_attr[idx].detach().cpu()
                    else:
                        H_inst_attr = getattr(data_batch, 'H_instantaneous', None)
                        if H_inst_attr is not None:
                            payload_entry['H_instantaneous'] = H_inst_attr[idx].detach().cpu()
                    tracked_epoch_payload[net_id] = payload_entry

            # Collect per-graph data for primal/dual history visualization.
            if collect_viz:
                for idx in range(B):
                    raw_seed = getattr(data_batch, "network_seed", None)
                    if raw_seed is None:
                        network_seed = int(network_ids_tensor[idx].item())
                    else:
                        seed_value = raw_seed[idx]
                        network_seed = (
                            int(seed_value.item())
                            if torch.is_tensor(seed_value)
                            else int(seed_value)
                        )
                    network_data_for_viz.append((
                        network_ids_tensor[idx].item(),
                        network_seed,
                        primal_vars[idx].cpu().numpy(),       # (m,), already detached
                        per_user_metrics[idx].cpu().numpy(),  # (n,), already detached
                        data_batch.associations[idx].cpu().numpy(),
                    ))

        # Epoch-based dual update: update all networks with accumulated constraint slacks.
        if self.dual_update_mode == 'epoch' and self.dual_optimizer.step():
            if epoch_gs:
                all_network_ids = torch.tensor(epoch_network_ids, dtype=torch.long, device=self.device)
                all_g = torch.cat(epoch_gs, dim=0)  # (total_networks_seen, n)
                epoch_dual_stats = self.dual_optimizer.update_from_gradients(
                    network_ids=all_network_ids,
                    gradients=all_g,
                )
        
        # Aggregate statistics over epoch
        for key in epoch_stats:
            if len(epoch_stats[key]) > 0:
                if key == 'global_min_rate':
                    # Use minimum for worst-case receiver
                    epoch_stats[key] = np.min(epoch_stats[key])
                elif key == 'global_5th_percentile_rate':
                    # Recompute from all rates for accurate percentile
                    if len(all_epoch_rates) > 0:
                        all_rates_tensor = torch.cat(all_epoch_rates)
                        epoch_stats[key] = torch.quantile(all_rates_tensor, 0.05).item()
                    else:
                        epoch_stats[key] = 0.0
                else:
                    # Average for other metrics
                    epoch_stats[key] = np.mean(epoch_stats[key])
            else:
                epoch_stats[key] = 0.0

        # Epoch-level percentiles for normalized transmit powers (p / P_max).
        if all_epoch_normalized_powers:
            all_powers_tensor = torch.cat(all_epoch_normalized_powers)
            for pctl in (5, 25, 50, 75, 90, 95, 99):
                epoch_stats[f'normalized_power_p{pctl}'] = torch.quantile(
                    all_powers_tensor, float(pctl) / 100.0
                ).item()
        else:
            for pctl in (5, 25, 50, 75, 90, 95, 99):
                epoch_stats[f'normalized_power_p{pctl}'] = 0.0
        
        # Merge epoch-mode dual update diagnostics (mirrors step-mode per-batch behavior).
        # Exclude mean_lambda / std_lambda — those are written authoritatively from
        # dual_summary in train() to avoid double-appending to training_history.
        if epoch_dual_stats is not None:
            for key, value in epoch_dual_stats.items():
                if key in ('mean_lambda', 'std_lambda'):
                    continue
                # In epoch-update mode, these diagnostics are only produced at the end.
                # Ensure they are persisted even though epoch_stats predefines their keys.
                if key in ('mean_dual_subgradient', 'positive_subgradient_fraction'):
                    epoch_stats[key] = value
                    continue
                if key not in epoch_stats:
                    epoch_stats[key] = value

        # Sort by network_id for a consistent receiver ordering across epochs,
        # regardless of dataloader shuffle order
        sort_order = sorted(range(len(all_epoch_net_ids)), key=lambda i: all_epoch_net_ids[i])
        all_epoch_rates = [all_epoch_rates[i] for i in sort_order]
        all_epoch_dual_subgradients = [all_epoch_dual_subgradients[i] for i in sort_order]
        all_epoch_net_ids = [all_epoch_net_ids[i] for i in sort_order]

        return (
            epoch_stats,
            all_epoch_rates,
            all_epoch_dual_subgradients,
            all_epoch_net_ids,
            network_data_for_viz,
            tracked_epoch_payload,
        )

    def _log_problem_specific_convergence_criteria(self) -> None:
        """Log problem-specific convergence criteria (subclasses may override)."""
        return

    def get_problem_specific_convergence_status(self) -> dict:
        """Return problem-specific convergence criteria status for this trainer."""
        return {}

    def check_convergence(self) -> tuple[bool, dict]:
        """
        Check if joint convergence criteria are met.
        
        Returns
        -------
        converged : bool
            True if all convergence criteria satisfied
        status : dict
            Detailed status of each criterion with values and thresholds
        """
        if len(self.training_history['gradient_norm']) < self.convergence_window:
            return False, {}
        if len(self.training_history['std_lambda']) < self.convergence_window:
            return False, {}
        if not self._ergodic_buffers_updated_this_epoch:
            return False, {}
        if len(self._ergodic_dual_buffer) < self.convergence_window:
            return False, {}
        if len(self._ergodic_slack_buffer) < self.convergence_window:
            return False, {}

        # Core trainer-agnostic criteria.
        recent_grad_norms = self.training_history['gradient_norm'][-self.convergence_window:]

        # Criterion 1: Gradient norm below threshold
        mean_grad_norm = np.mean(recent_grad_norms)
        grad_converged = mean_grad_norm < self.gradient_norm_threshold

        # Criterion 2: Dual variance stabilized
        # Check if dual variance change rate is small
        recent_std = self.training_history['std_lambda'][-self.convergence_window:]
        first_half = recent_std[:self.convergence_window//2]
        second_half = recent_std[self.convergence_window//2:]
        std_change_rate = abs(np.mean(second_half) - np.mean(first_half)) / (np.mean(first_half) + 1e-6)
        dual_converged = std_change_rate < self.dual_variance_threshold

        # Criterion 3: ergodic complementary slackness.
        # Evaluate mean_i |λ̄_i · ḡ_i| on windowed averages (λ̄, ḡ). Taking the
        # absolute value before averaging avoids cancellation between positive
        # and negative components.
        lambda_bar = torch.stack(list(self._ergodic_dual_buffer)).mean(dim=0)
        g_bar = torch.stack(list(self._ergodic_slack_buffer)).mean(dim=0)
        comp_products = lambda_bar * g_bar
        mean_complementary_slackness = comp_products.mean().item()
        mean_abs_complementary_slackness = comp_products.abs().mean().item()
        dual_stationarity_converged = (
            mean_abs_complementary_slackness < self.dual_stationarity_threshold
        )
        
        # Build status dictionary
        status = {
            'grad_norm': {'value': mean_grad_norm, 'threshold': self.gradient_norm_threshold, 'converged': grad_converged},
            'dual_variance': {'value': std_change_rate, 'threshold': self.dual_variance_threshold, 'converged': dual_converged},
            'dual_stationarity': {
                'value': mean_abs_complementary_slackness,
                'raw_value': mean_complementary_slackness,
                'threshold': self.dual_stationarity_threshold,
                'converged': dual_stationarity_converged,
            },
        }
        problem_specific_status = self.get_problem_specific_convergence_status()
        for key in problem_specific_status:
            if key in status:
                raise ValueError(f"Duplicate convergence status key '{key}' from subclass.")
        status.update(problem_specific_status)

        # All criteria must be met
        converged = all(info.get('converged', False) for info in status.values())
        
        return converged, status
    
    def train(
        self,
        dataloader: DataLoader,
    ) -> Dict:
        """
        Main training loop.
        
        Parameters
        ----------
        dataloader : DataLoader
            DataLoader for training data
        
        Returns
        -------
        results : dict
            Training results and collected samples
        """
        logger.info(f"Starting primal-dual training for {self.max_epochs} epochs...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Convergence criteria:")
        logger.info(f"  - Gradient norm < {self.gradient_norm_threshold}")
        logger.info(f"  - Dual variance change < {self.dual_variance_threshold}")
        logger.info(
            f"  - Ergodic complementary slackness mean_i |lambda_bar_i * g_bar_i| "
            f"< {self.dual_stationarity_threshold}"
        )
        self._log_problem_specific_convergence_criteria()
        logger.info(f"  - Convergence checks enabled after epoch >= {self.convergence_warmup_epochs}")
        logger.info(f"  - Patience: {self.convergence_patience} epochs")
        logger.info(f"Polyak-style sampling:")
        logger.info(f"  - Saving checkpoints every {self.sample_checkpoint_frequency} epochs")
        logger.info(f"  - Keeping last {self.num_samples_per_network} checkpoints for diverse sampling")
        logger.info(f"  - Rewriting primal_history.jsonl every {self.primal_history_write_interval} epochs")
        if self.sample_collection_interval > 0:
            logger.info(f"  - Collecting samples every {self.sample_collection_interval} epochs")
        else:
            logger.info(f"  - Collecting samples only at convergence")
        if self.trace_logging_enabled:
            logger.info(
                "  - Tracked-network trace logging enabled: networks=%s, receivers=%s, "
                "windows=%s, file=%s, flush_interval=%d",
                self.trace_network_ids,
                self.trace_receiver_indices if self.trace_receiver_indices else "auto-first-two",
                self.trace_window_sizes,
                self.trace_output_path.name,
                self.trace_write_interval,
            )
        else:
            logger.info("  - Tracked-network trace logging disabled")

        epoch_summaries_list: list = []  # in-memory cache; avoids re-reading JSONL each epoch

        for epoch in tqdm(range(self.max_epochs), desc="Training"):
            self.epoch = epoch
            
            # Train for one epoch
            (
                epoch_stats,
                all_epoch_rates,
                all_epoch_dual_subgradients,
                epoch_net_ids,
                network_data_for_viz,
                tracked_epoch_payload,
            ) = self.train_epoch(dataloader)

            # Update training history
            for key, value in epoch_stats.items():
                if key in self.training_history:
                    self.training_history[key].append(value)
            
            # Add dual statistics from DualOptimizer
            dual_summary = self.dual_optimizer.get_summary_statistics()
            self.training_history['mean_lambda'].append(dual_summary['mean_lambda'])
            self.training_history['std_lambda'].append(dual_summary['std_lambda'])
            
            # Update model-averaging rate buffers using rates already computed in train_epoch
            self._ergodic_buffers_updated_this_epoch = False
            if not all_epoch_rates:
                logger.warning(f"Epoch {epoch}: dataloader yielded no batches; skipping rate buffer update.")
            else:
                if len(all_epoch_rates) != len(all_epoch_dual_subgradients):
                    raise RuntimeError(
                        "Length mismatch between per-network metrics and dual subgradients: "
                        f"{len(all_epoch_rates)} vs {len(all_epoch_dual_subgradients)}"
                    )
                epoch_rates = torch.cat(all_epoch_rates)  # (total_receivers,)
                epoch_slacks = torch.cat(all_epoch_dual_subgradients)  # (total_receivers,)
                network_sizes = [len(r) for r in all_epoch_rates]

                epoch_rates_cpu = epoch_rates.cpu()
                epoch_slacks_cpu = epoch_slacks.cpu()

                self.rate_buffer.append(epoch_rates_cpu)
                self.network_sizes_buffer.append(network_sizes)
                self.slack_buffer.append(epoch_slacks_cpu)

                self.rate_buffer_5M.append(epoch_rates_cpu)
                self.network_sizes_buffer_5M.append(network_sizes)
                self.slack_buffer_5M.append(epoch_slacks_cpu)

                self.rate_buffer_25M.append(epoch_rates_cpu)
                self.network_sizes_buffer_25M.append(network_sizes)
                self.slack_buffer_25M.append(epoch_slacks_cpu)

                # Ergodic dual stationarity: snapshot end-of-epoch duals and
                # constraint slacks per unique network.
                unique_nets = list(dict.fromkeys(epoch_net_ids))  # deduplicate, preserve order
                net_ids_t = torch.tensor(unique_nets, dtype=torch.long, device=self.device)
                epoch_duals = self.dual_optimizer.get_duals(net_ids_t).detach().reshape(-1).cpu()
                # Average dual subgradients per unique network (handles repeated IDs).
                _slacks_by_net: dict = {}
                for nid, slack in zip(epoch_net_ids, all_epoch_dual_subgradients):
                    _slacks_by_net.setdefault(nid, []).append(slack)
                avg_slacks = torch.cat([
                    torch.stack(v).mean(dim=0) for v in
                    (_slacks_by_net[n] for n in unique_nets)
                ])
                epoch_slacks = avg_slacks.cpu()
                self._ergodic_dual_buffer.append(epoch_duals)
                self._ergodic_slack_buffer.append(epoch_slacks)
                self._ergodic_buffers_updated_this_epoch = True

            # Write primal/dual history for visualization (only on checkpointing epochs)
            if network_data_for_viz:
                dual_history_file = self.checkpoint_dir / "dual_history.jsonl"
                with open(dual_history_file, 'a') as f:
                    for network_id, network_seed, power, rates, associations in network_data_for_viz:
                        network_id_tensor = torch.tensor([network_id], dtype=torch.long, device=self.device)
                        duals = self.dual_optimizer.get_duals(network_id_tensor)[0].cpu().numpy()
                        entry = {
                            'epoch': epoch,
                            'network_id': int(network_id),
                            'dual_multipliers': duals.tolist(),
                        }
                        f.write(json.dumps(entry) + '\n')
                        self._dual_history_entries.append(entry)

                # Accumulate primal metadata (once per network) and epoch entries (rolling window).
                for network_id, network_seed, power, rates, associations in network_data_for_viz:
                    if network_id not in self._primal_metadata_entries:
                        self._primal_metadata_entries[network_id] = associations.tolist()
                        metadata_payload: dict = {'network_seed': int(network_seed)}
                        if associations.ndim == 2:
                            r_min_vec = self._r_min_for_network(
                                network_id,
                                num_receivers=int(associations.shape[1]),
                            ).detach().cpu().numpy()
                            metadata_payload['r_min_per_receiver'] = r_min_vec
                        extra_hook = getattr(self, "_extra_sample_metadata", None)
                        if callable(extra_hook):
                            try:
                                extra_meta = extra_hook(int(network_id))
                            except Exception as exc:
                                logger.debug(
                                    "Skipping extra primal metadata for network %d due to error: %s",
                                    int(network_id),
                                    exc,
                                )
                                extra_meta = {}
                            if isinstance(extra_meta, dict):
                                metadata_payload.update(extra_meta)
                        self._primal_extra_metadata_entries[network_id] = _to_json_compatible(
                            metadata_payload
                        )
                    if network_id not in self._primal_epoch_entries:
                        self._primal_epoch_entries[network_id] = deque(maxlen=self._primal_history_window)
                    self._primal_epoch_entries[network_id].append({
                        'epoch': epoch,
                        'network_id': int(network_id),
                        'power_allocations': power.tolist(),
                        'rates': rates.tolist(),
                    })

                # Rewrite rolling-window primal history at a lower cadence to reduce I/O load.
                if epoch % self.primal_history_write_interval == 0:
                    self._flush_primal_history()

            # Record full per-epoch traces for selected networks.
            if tracked_epoch_payload:
                self._record_tracked_network_epoch(
                    epoch=epoch,
                    tracked_epoch_payload=tracked_epoch_payload,
                )
            
            # Compute model-averaged metrics for all three window sizes
            model_avg_metrics = {}

            # Define window configurations:
            # (suffix, rate_buffer, slack_buffer, sizes_buffer, window_size)
            window_configs = [
                ('', self.rate_buffer, self.slack_buffer, self.network_sizes_buffer, self.moving_avg_window),
                ('_5M', self.rate_buffer_5M, self.slack_buffer_5M, self.network_sizes_buffer_5M, self.window_5M),
                ('_25M', self.rate_buffer_25M, self.slack_buffer_25M, self.network_sizes_buffer_25M, self.window_25M),
            ]

            for suffix, rate_buffer, slack_buffer, sizes_buffer, window_size in window_configs:
                if len(rate_buffer) >= window_size and len(slack_buffer) >= window_size:
                    # Stack rates/slacks from buffer and compute model averages.
                    stacked_rates = torch.stack(list(rate_buffer))  # (window_size, total_receivers)
                    stacked_slacks = torch.stack(list(slack_buffer))  # (window_size, total_receivers)
                    model_avg_rates = stacked_rates.mean(dim=0)  # (total_receivers,)
                    model_avg_slacks = stacked_slacks.mean(dim=0)  # (total_receivers,)
                    model_avg_violation_magnitudes = torch.clamp(model_avg_slacks, min=0.0)

                    # Compute per-network mins and violation fractions
                    network_sizes = sizes_buffer[0]
                    per_network_mins = []
                    offset = 0
                    for size in network_sizes:
                        network_rates = model_avg_rates[offset:offset+size]
                        per_network_mins.append(network_rates.min().item())
                        offset += size

                    violation_fraction_on_model_avg_rates = (
                        (model_avg_slacks > 0).float().mean().item()
                    )

                    # Add metrics for this window
                    model_avg_metrics.update({
                        f'model_avg_mean_rate{suffix}': model_avg_rates.mean().item(),
                        f'model_avg_avg_per_network_min_rate{suffix}': np.mean(per_network_mins),
                        f'model_avg_global_min_rate{suffix}': np.min(per_network_mins),
                        f'model_avg_global_5th_percentile_rate{suffix}': torch.quantile(model_avg_rates, 0.05).item(),
                        f'model_avg_mean_violation_slack{suffix}': model_avg_violation_magnitudes.mean().item(),
                        f'violation_fraction_on_model_avg_rates{suffix}': violation_fraction_on_model_avg_rates,
                    })

                    # Only M window gets added to training history for convergence check
                    if suffix == '':
                        self.training_history['violation_fraction_on_model_avg_rates'].append(
                            violation_fraction_on_model_avg_rates
                        )
                        self.training_history['mean_violation_slack_on_model_avg_rates'].append(
                            model_avg_violation_magnitudes.mean().item()
                        )
                elif suffix == '':
                    # Only M window needs None when not enough data
                    self.training_history['violation_fraction_on_model_avg_rates'].append(None)
                    self.training_history['mean_violation_slack_on_model_avg_rates'].append(None)
            
            # Compute spaced model-averaged metrics (from checkpoint buffer)
            # These metrics use spaced checkpoints (every 5 epochs) instead of consecutive epochs
            # This matches the collected_samples_spaced.npz methodology
            if len(self.checkpoint_paths) >= min(self.num_samples_per_network, self.moving_avg_window):
                # We have enough spaced checkpoints to compute spaced M-averaged metrics
                # For now, we mark that we have them but don't recompute (expensive)
                # Instead, we add a note that these can be computed from checkpoint_paths
                model_avg_metrics.update({
                    'num_spaced_checkpoints': len(self.checkpoint_paths),
                    'spaced_checkpoint_epochs': [int(cp.stem.split('_')[-1]) for cp in self.checkpoint_paths[-self.moving_avg_window:]],
                })
                # Note: Actual spaced M-averaged rates would require loading each checkpoint
                # and running inference, which is too expensive to do every epoch.
                # These are computed during sample collection and saved separately.
            
            # Save checkpoint for Polyak-style sampling
            # Save every convergence_window / num_samples_per_network epochs
            # Keep only last num_samples_per_network checkpoints
            if epoch % self.sample_checkpoint_frequency == 0 and epoch > 0:
                checkpoint_filename = f"sample_checkpoint_epoch_{epoch}.pt"
                self.save_model_checkpoint(checkpoint_filename)
                self.checkpoint_paths.append(self.model_checkpoint_dir / checkpoint_filename)
                
                # Remove oldest checkpoint if buffer is full
                if len(self.checkpoint_paths) > self.num_samples_per_network:
                    oldest_checkpoint = self.checkpoint_paths.pop(0)
                    if oldest_checkpoint.exists():
                        oldest_checkpoint.unlink()  # Delete old checkpoint file
            
            # Continuous sample collection (if enabled)
            if self.sample_collection_interval > 0 and epoch % self.sample_collection_interval == 0 and epoch > 0:
                if len(self.checkpoint_paths) > 0:
                    logger.info(f"\n{'='*60}")
                    logger.info(f"Collecting samples at epoch {epoch} ({len(self.checkpoint_paths)} checkpoints available)")
                    logger.info(f"{'='*60}")
                    
                    # Collect samples from checkpoints
                    samples = self.collect_samples(dataloader)
                    
                    # Save samples
                    self._save_samples_only(samples)
                    
                    logger.info(f"Samples saved to: {self.checkpoint_dir / 'collected_samples.npz'}")
                    logger.info(f"{'='*60}\n")
            
            stop_after_epoch = False
            status = {}

            # Check convergence
            if epoch >= self.convergence_warmup_epochs:
                converged, status = self.check_convergence()
                
                if converged:
                    self.convergence_met_count += 1
                    logger.info(f"\n✓ Convergence criteria met ({self.convergence_met_count}/{self.convergence_patience})")
                    logger.info("  All criteria satisfied:")
                    for name, info in status.items():
                        if info['threshold'] is not None:
                            logger.info(f"    ✓ {name}: {info['value']:.6f} < {info['threshold']:.6f}")
                        else:
                            logger.info(f"    ✓ {name}: {info['value']:.4f}")
                    
                    if self.convergence_met_count >= self.convergence_patience:
                        self.converged = True
                        logger.info(f"\n{'='*60}")
                        logger.info(f"CONVERGENCE ACHIEVED at epoch {epoch}")
                        logger.info(f"{'='*60}")
                        stop_after_epoch = True
                else:
                    # Check if only one criterion failed (close to convergence)
                    failed_criteria = [name for name, info in status.items() if not info['converged']]
                    if len(failed_criteria) == 1:
                        name = failed_criteria[0]
                        info = status[name]
                        logger.info(f"\n⚠ Close to convergence! Only 1 criterion not met:")
                        if info['threshold'] is not None:
                            logger.info(f"    ✗ {name}: {info['value']:.6f} >= {info['threshold']:.6f}")
                        else:
                            logger.info(f"    ✗ {name}: {info['value']:.4f} (needs improvement)")
                        logger.info("  Other criteria satisfied:")
                        for other_name, other_info in status.items():
                            if other_info['converged']:
                                if other_info['threshold'] is not None:
                                    logger.info(f"    ✓ {other_name}: {other_info['value']:.6f} < {other_info['threshold']:.6f}")
                                else:
                                    logger.info(f"    ✓ {other_name}: {other_info['value']:.4f}")
                    
                    self.convergence_met_count = 0

            ergodic_complementary_slackness_abs = None
            ergodic_complementary_slackness = None
            if 'dual_stationarity' in status:
                ergodic_complementary_slackness_abs = float(
                    status['dual_stationarity'].get('value')
                )
                raw_comp = status['dual_stationarity'].get('raw_value')
                if raw_comp is not None:
                    ergodic_complementary_slackness = float(raw_comp)
            self.training_history['ergodic_complementary_slackness_abs'].append(
                ergodic_complementary_slackness_abs
            )
            self.training_history['ergodic_complementary_slackness'].append(
                ergodic_complementary_slackness
            )

            # Save epoch summary to JSON file (after convergence check so the counter
            # reflects this epoch's post-check value)
            r_min_summary = self._r_min_summary()
            constraint_profile_metrics = self._build_constraint_profile_epoch_metrics(
                epoch_network_ids=epoch_net_ids,
                epoch_rates=all_epoch_rates,
                epoch_slacks=all_epoch_dual_subgradients,
            )
            epoch_summary = {
                'epoch': epoch,
                'loss': epoch_stats['loss'],
                'avg_per_network_min_rate': epoch_stats['avg_per_network_min_rate'],
                'global_min_rate': epoch_stats['global_min_rate'],
                'mean_rate': epoch_stats['mean_rate'],
                'global_5th_percentile_rate': epoch_stats['global_5th_percentile_rate'],
                'mean_slack': epoch_stats['mean_slack'],
                'violation_fraction': epoch_stats['violation_fraction'],
                'mean_violation_slack': epoch_stats['mean_violation_slack'],
                'mean_dual_subgradient': epoch_stats['mean_dual_subgradient'],
                'positive_subgradient_fraction': epoch_stats['positive_subgradient_fraction'],
                'projected_dual_residual': epoch_stats['projected_dual_residual'],
                'normalized_power_p5': epoch_stats['normalized_power_p5'],
                'normalized_power_p25': epoch_stats['normalized_power_p25'],
                'normalized_power_p50': epoch_stats['normalized_power_p50'],
                'normalized_power_p75': epoch_stats['normalized_power_p75'],
                'normalized_power_p90': epoch_stats['normalized_power_p90'],
                'normalized_power_p95': epoch_stats['normalized_power_p95'],
                'normalized_power_p99': epoch_stats['normalized_power_p99'],
                'ergodic_complementary_slackness': ergodic_complementary_slackness,
                'ergodic_complementary_slackness_abs': ergodic_complementary_slackness_abs,
                **r_min_summary,
                'convergence_criteria_status': {
                    name: bool(info.get('converged', False))
                    for name, info in status.items()
                },
                'gradient_norm': epoch_stats['gradient_norm'],
                'mean_lambda': dual_summary['mean_lambda'],
                'std_lambda': dual_summary['std_lambda'],
                'convergence_met_count': self.convergence_met_count,
                **model_avg_metrics
            }
            if constraint_profile_metrics:
                epoch_summary['constraint_profile_metrics'] = constraint_profile_metrics

            summary_file = self.checkpoint_dir / "epoch_summaries.jsonl"
            with open(summary_file, 'a') as f:
                f.write(json.dumps(epoch_summary) + '\n')
            epoch_summaries_list.append(epoch_summary)

            # Visualize every 50 epochs
            if epoch % 50 == 0:
                try:
                    visualize_training_progress(epoch, self.checkpoint_dir, self.moving_avg_window, self.convergence_patience, summaries=epoch_summaries_list)
                    visualize_training_progress_by_profile(
                        epoch,
                        self.checkpoint_dir,
                        self.moving_avg_window,
                        summaries=epoch_summaries_list,
                    )

                    P_max = self.system_params.get('P_max_watts', self.system_params.get('P_max', 1.0))
                    for viz_net_id in self._select_visualization_network_ids(max_base_networks=2):
                        network_label = self._visualization_network_label(viz_net_id)
                        visualize_dual_multipliers(
                            self.checkpoint_dir,
                            top_k=10,
                            network_id=viz_net_id,
                            all_entries=self._dual_history_entries,
                            network_label=network_label,
                        )
                        if viz_net_id not in self._primal_metadata_entries:
                            continue
                        associations = np.asarray(self._primal_metadata_entries[viz_net_id])
                        if associations.ndim != 2:
                            continue
                        r_min_vec = self._r_min_for_network(
                            viz_net_id, num_receivers=int(associations.shape[1])
                        ).detach().cpu().numpy()
                        visualize_power_allocations(
                            self.checkpoint_dir,
                            P_max=P_max,
                            r_min=r_min_vec,
                            top_k=5,
                            network_id=viz_net_id,
                            all_entries=list(self._primal_epoch_entries.get(viz_net_id, [])),
                            metadata_entries=self._primal_metadata_entries,
                            network_label=network_label,
                        )
                except Exception as e:
                    logger.warning(f"Visualization failed at epoch {epoch}: {e}")

            if stop_after_epoch:
                break

        # Generate a final convergence plot with the latest epoch summaries
        if epoch_summaries_list:
            try:
                visualize_training_progress(self.epoch, self.checkpoint_dir, self.moving_avg_window, self.convergence_patience, summaries=epoch_summaries_list)
                visualize_training_progress_by_profile(
                    self.epoch,
                    self.checkpoint_dir,
                    self.moving_avg_window,
                    summaries=epoch_summaries_list,
                )
            except Exception as e:
                logger.warning(f"Final visualization failed: {e}")
        
        # Training complete (converged or max_epochs reached)
        if not self.converged:
            logger.info(f"\n{'='*60}")
            logger.info(f"MAX EPOCHS REACHED ({self.max_epochs}) - collecting samples anyway")
            logger.info(f"{'='*60}")

        # Flush latest rolling-window primal history so downstream tools can read the final trajectory view.
        if self._primal_metadata_entries or self._primal_epoch_entries:
            self._flush_primal_history()
        self._flush_trace_history(force=True)
        
        # Save pre-collection checkpoint
        logger.info("Saving pre-collection checkpoint")
        self.save_checkpoint("pre_collection_checkpoint.pt")
        
        # Collect samples from checkpoints
        logger.info(f"\n{'='*70}")
        logger.info(f"COLLECTING SAMPLES (from {len(self.checkpoint_paths)} checkpoints)")
        logger.info(f"{'='*70}")
        logger.info(f"Collecting {self.num_samples_per_network} samples per network from checkpoints")
        samples = self.collect_samples(dataloader)

        # Analyze sample quality
        logger.info("Analyzing sample quality")
        quality_report = self.analyze_sample_quality(samples, dataloader)
        
        results = {
            'training_history': self.training_history,
            'converged': self.converged,
            'final_epoch': self.epoch,
            'samples': samples,
            'quality_report': quality_report,
            'metadata': {
                'num_samples_per_network': self.num_samples_per_network,
                'checkpoint_epochs': [int(cp.stem.split('_')[-1]) for cp in self.checkpoint_paths] if self.checkpoint_paths else [],
                'checkpoint_frequency': self.sample_checkpoint_frequency,
                'channel_version': self.channel_version,
                'r_min_summary': self._r_min_summary(),
                'trace_logging': {
                    'enabled': bool(self.trace_logging_enabled),
                    'network_ids': list(self.trace_network_ids),
                    'receiver_indices': list(self.trace_receiver_indices),
                    'window_sizes': list(self.trace_window_sizes),
                    'include_full_vectors': bool(self.trace_include_full_vectors),
                    'write_interval': int(self.trace_write_interval),
                    'trace_history_file': (
                        self.trace_output_path.name
                        if self.trace_logging_enabled else None
                    ),
                },
            }
        }
        
        # Save final results
        logger.info(f"Training complete at epoch {self.epoch}. Converged: {self.converged}. Samples: {len(samples)} networks")
        self.save_results(results)
        
        return results
    
    @abc.abstractmethod
    def collect_samples(
        self,
        dataloader: DataLoader,
    ) -> Dict:
        """
        Collect decision-variable samples using Polyak-style checkpoint loading.

        Loads each of the last num_samples_per_network model checkpoints and
        generates one sample per network from each checkpoint, providing diverse
        samples from different points in the training trajectory.

        Parameters
        ----------
        dataloader : DataLoader
            DataLoader for networks.

        Returns
        -------
        samples : dict
            Dictionary mapping network_id -> per-network sample data.
        """

    @abc.abstractmethod
    def analyze_sample_quality(
        self,
        samples: Dict,
        dataloader: DataLoader,
    ) -> Dict:
        """
        Analyze quality of collected samples.

        Parameters
        ----------
        samples : dict
            Collected samples per network (from collect_samples).
        dataloader : DataLoader
            DataLoader to get network data.

        Returns
        -------
        quality_report : dict
            Per-network quality statistics.
        """
    
    def save_checkpoint(self, filename: str) -> None:
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'dual_optimizer_state_dict': self.dual_optimizer.state_dict(),
            'training_history': self.training_history,
            'converged': self.converged,
        }
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def save_model_checkpoint(self, filename: str) -> None:
        """Save model checkpoint to model_chkpts subfolder for Polyak sampling."""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
        }
        
        path = self.model_checkpoint_dir / filename
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, checkpoint_path: Path) -> int:
        """Load model state from checkpoint.
        
        Parameters
        ----------
        checkpoint_path : Path
            Path to checkpoint file
        
        Returns
        -------
        epoch : int
            Epoch number of loaded checkpoint
        """
        # Prefer safe loading mode to avoid pickle warnings on modern PyTorch.
        # Fall back for older PyTorch versions that do not support weights_only.
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=True,
            )
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint['epoch']
    
    def _flush_primal_history(self) -> None:
        """Rewrite primal_history.jsonl from in-memory state (metadata + rolling window)."""
        primal_history_file = self.checkpoint_dir / "primal_history.jsonl"
        with open(primal_history_file, 'w') as f:
            for net_id, assoc_list in self._primal_metadata_entries.items():
                metadata_entry = {
                    'type': 'metadata',
                    'network_id': int(net_id),
                    'associations': assoc_list,
                }
                metadata_entry.update(
                    self._primal_extra_metadata_entries.get(net_id, {})
                )
                f.write(json.dumps(metadata_entry) + '\n')
            for net_id, epoch_deque in self._primal_epoch_entries.items():
                for entry in epoch_deque:
                    f.write(json.dumps(entry) + '\n')

    def _save_samples_only(self, samples: Dict) -> None:
        """
        Save collected samples without quality analysis (for continuous collection).
        
        Parameters
        ----------
        samples : dict
            Samples collected from checkpoints
        """
        samples_path = self._save_samples_artifact(samples, "collected_samples.npz")
        logger.info(f"Saved canonical samples to {samples_path}")

    def _save_samples_artifact(self, samples: Dict, filename: str) -> Path:
        """Save canonical primal-dual samples to NPZ."""
        output_path = self.checkpoint_dir / filename
        save_pd_samples_npz(
            output_path=output_path,
            samples_by_network=samples,
            channel_version=self.channel_version,
        )
        return output_path

    def _save_quality_report(self, quality_report: Dict, filename: str) -> Path:
        """Save quality report JSON with stable key serialization."""
        report_path = self.checkpoint_dir / filename
        serializable = {str(k): v for k, v in quality_report.items()}
        with open(report_path, "w") as f:
            json.dump(serializable, f, indent=2)
        return report_path
    
    def save_results(self, results: Dict) -> None:
        """Save final training results with network seeds and metadata."""
        # Save training history as JSON
        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(results['training_history'], f, indent=2)

        if 'samples' in results:
            samples_path = self._save_samples_artifact(results['samples'], 'collected_samples.npz')
            # Keep both names for compatibility with older verification scripts.
            self._save_quality_report(results.get('quality_report', {}), 'quality_report.json')
            self._save_quality_report(results.get('quality_report', {}), 'collected_samples_quality_report.json')
            logger.info(f"Samples saved to: {samples_path}")
        
        # Save metadata
        metadata = results.get('metadata', {})
        if metadata:
            metadata_path = self.checkpoint_dir / "collection_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        logger.info(f"\nAll results saved to {self.checkpoint_dir}")


# ---------------------------------------------------------------------------
# WRA-specific concrete subclass
# ---------------------------------------------------------------------------

class WRAPrimalDualTrainer(PrimalDualTrainer):
    """
    Primal-dual trainer for the Wireless Resource Allocation (WRA) problem.

    Implements the four abstract methods of PrimalDualTrainer for the WRA
    setting:

    * primal_forward:  GNN → per-receiver powers → per-transmitter powers via
                       the association matrix.
    * compute_constraints: vectorised SINR/Shannon-capacity ergodic-rate
                       computation; objective = Σ R_i, constraints = r_min − R_i.
    * collect_samples: Polyak-style checkpoint sampling with WRA tensors.
    * analyze_sample_quality: stochastic/deterministic policy quality report.

    The generic training loop/checkpointing flow is inherited from
    PrimalDualTrainer; this subclass adds WRA-specific convergence criteria.
    """

    def _log_problem_specific_convergence_criteria(self) -> None:
        """Log WRA-specific convergence criteria in addition to base criteria."""
        logger.info(f"  - Violation fraction < {self.violation_fraction_threshold}")
        logger.info(
            f"  - Violation fraction on model-averaged rates < "
            f"{self.violation_fraction_on_model_avg_rates_threshold}"
        )
        logger.info(
            "  - Mean violation slack on model-averaged rates < "
            f"{self.mean_violation_slack_on_model_avg_rates_threshold}"
        )

    def get_problem_specific_convergence_status(self) -> dict:
        """WRA-specific feasibility convergence criteria based on rate violations."""
        if len(self.training_history['violation_fraction']) < self.convergence_window:
            return {}

        recent_violation_fractions = self.training_history[
            'violation_fraction'
        ][-self.convergence_window:]
        mean_violation_fraction = np.mean(recent_violation_fractions)
        violation_converged = mean_violation_fraction < self.violation_fraction_threshold

        if len(self.training_history['violation_fraction_on_model_avg_rates']) >= self.convergence_window:
            recent_violation_fractions_on_model_avg_rates = self.training_history[
                'violation_fraction_on_model_avg_rates'
            ][-self.convergence_window:]
            # Filter out None values (epochs before enough data for model averaging)
            recent_violation_fractions_on_model_avg_rates = [
                v for v in recent_violation_fractions_on_model_avg_rates if v is not None
            ]
            if len(recent_violation_fractions_on_model_avg_rates) >= self.convergence_window:
                mean_violation_fraction_on_model_avg_rates = np.mean(
                    recent_violation_fractions_on_model_avg_rates
                )
                violation_fraction_on_model_avg_rates_converged = (
                    mean_violation_fraction_on_model_avg_rates
                    < self.violation_fraction_on_model_avg_rates_threshold
                )
            else:
                violation_fraction_on_model_avg_rates_converged = False
                mean_violation_fraction_on_model_avg_rates = float('inf')
        else:
            violation_fraction_on_model_avg_rates_converged = False
            mean_violation_fraction_on_model_avg_rates = float('inf')

        if len(self.training_history['mean_violation_slack_on_model_avg_rates']) >= self.convergence_window:
            recent_mean_violation_slacks_on_model_avg_rates = self.training_history[
                'mean_violation_slack_on_model_avg_rates'
            ][-self.convergence_window:]
            recent_mean_violation_slacks_on_model_avg_rates = [
                v for v in recent_mean_violation_slacks_on_model_avg_rates if v is not None
            ]
            if len(recent_mean_violation_slacks_on_model_avg_rates) >= self.convergence_window:
                mean_violation_slack_on_model_avg_rates = np.mean(
                    recent_mean_violation_slacks_on_model_avg_rates
                )
                mean_violation_slack_on_model_avg_rates_converged = (
                    mean_violation_slack_on_model_avg_rates
                    < self.mean_violation_slack_on_model_avg_rates_threshold
                )
            else:
                mean_violation_slack_on_model_avg_rates_converged = False
                mean_violation_slack_on_model_avg_rates = float('inf')
        else:
            mean_violation_slack_on_model_avg_rates_converged = False
            mean_violation_slack_on_model_avg_rates = float('inf')

        return {
            'violation_fraction': {
                'value': mean_violation_fraction,
                'threshold': self.violation_fraction_threshold,
                'converged': violation_converged,
            },
            'violation_fraction_on_model_avg_rates': {
                'value': mean_violation_fraction_on_model_avg_rates,
                'threshold': self.violation_fraction_on_model_avg_rates_threshold,
                'converged': violation_fraction_on_model_avg_rates_converged,
            },
            'mean_violation_slack_on_model_avg_rates': {
                'value': mean_violation_slack_on_model_avg_rates,
                'threshold': self.mean_violation_slack_on_model_avg_rates_threshold,
                'converged': mean_violation_slack_on_model_avg_rates_converged,
            },
        }

    def primal_forward(
        self,
        batch: Batch,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run the GNN and map per-receiver powers to per-transmitter powers.

        Returns
        -------
        power_batch : torch.Tensor
            Per-transmitter powers, shape (B, m).
        assoc_batch : torch.Tensor
            Stacked association matrices (B, m, n) — passed as forward_ctx to
            compute_constraints so they are only loaded once per step.
        """
        batch_size = batch.num_graphs
        ptr = batch.ptr  # [0, n, 2n, ..., B*n]
        n_receivers = (ptr[1] - ptr[0]).item()

        # GNN forward: (total_nodes,) — one output per receiver node
        power_per_receiver = self.model(
            x=batch.x,
            edge_index=batch.edge_index,
            edge_weight=batch.edge_weight,
            batch=batch.batch,
        )
        power_per_receiver_2d = power_per_receiver.view(batch_size, n_receivers)  # (B, n)

        # Stack association matrices; PyG does not move list attrs to device automatically
        assoc_batch = torch.stack(
            [batch.associations[i].to(self.device, non_blocking=True) for i in range(batch_size)]
        )  # (B, m, n)

        # Receiver → transmitter power via association: (B, m, n) @ (B, n, 1) → (B, m)
        power_batch = torch.bmm(
            assoc_batch, power_per_receiver_2d.unsqueeze(-1)
        ).squeeze(-1)  # (B, m)

        return power_batch, assoc_batch  # primal_vars, forward_ctx

    def compute_constraints(
        self,
        primal_vars: torch.Tensor,
        forward_ctx: torch.Tensor,
        batch: Batch,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute ergodic rates and return objective + constraint slacks.

        Parameters
        ----------
        primal_vars : torch.Tensor
            Per-transmitter powers, shape (B, m).
        forward_ctx : torch.Tensor
            Association matrices (B, m, n) from primal_forward.
        batch : Batch
            PyG batch; batch.H_instantaneous[i] is the channel tensor (T, m, n).

        Returns
        -------
        objective : torch.Tensor
            Sum of ergodic rates per network, shape (B,).
        g : torch.Tensor
            Min-rate constraint slacks r_min − R_i, shape (B, n).
        per_user_metrics : torch.Tensor
            Ergodic rates (B, n) — used for model averaging and logging.
        """
        power_batch = primal_vars   # (B, m)
        assoc_batch = forward_ctx   # (B, m, n)
        batch_size  = batch.num_graphs
        noise_var   = self.system_params['noise_var']
        network_ids = self._network_ids_from_batch(batch)

        H_inst_batch = torch.stack(
            [batch.H_instantaneous[i].to(self.device, non_blocking=True) for i in range(batch_size)]
        )  # (B, T, m, n)

        # Vectorised ergodic-rate computation without materialising received_power
        # (B, T, m, n), which is the dominant memory hotspot for large N/T runs.
        total_received = torch.einsum("btmn,bm->btn", H_inst_batch, power_batch)  # (B, T, n)
        direct_signal = torch.einsum(
            "btmn,bmn,bm->btn", H_inst_batch, assoc_batch, power_batch
        )  # (B, T, n)
        interference = total_received - direct_signal  # (B, T, n)
        ergodic_rates   = torch.log2(
            1.0 + direct_signal / (interference + noise_var)
        ).mean(dim=1)  # (B, n)
        r_min_batch = self._r_min_batch(network_ids, ergodic_rates.shape[1])  # (B, n)

        objective = ergodic_rates.sum(dim=1)   # (B,) — maximise total rate
        g         = r_min_batch - ergodic_rates       # (B, n) — positive = violated

        return objective, g, ergodic_rates

    # ------------------------------------------------------------------
    # Sample collection and quality analysis (WRA-specific)
    # ------------------------------------------------------------------

    def _extra_sample_metadata(self, net_id: int) -> Dict:
        """Hook for subclasses to attach extra per-network metadata to samples."""
        return {}

    def collect_samples(
        self,
        dataloader: DataLoader,
    ) -> Dict:
        """
        Collect WRA power-allocation samples using Polyak-style checkpoint loading.

        Uses primal_forward / compute_constraints so the forward pass is
        consistent with the training loop and no WRA-specific model call
        is duplicated here.
        """
        self.model.eval()
        samples = {}

        checkpoints_to_use = self.checkpoint_paths if self.checkpoint_paths else [None]

        logger.info(f"\nUsing {len(checkpoints_to_use)} checkpoint(s) for sample collection:")
        for cp in checkpoints_to_use:
            if cp is not None:
                logger.info(f"  - {cp.name}")

        with torch.no_grad():
            for checkpoint_idx, checkpoint_path in enumerate(checkpoints_to_use):
                checkpoint_epoch = self.epoch
                if checkpoint_path is not None:
                    checkpoint_epoch = self.load_checkpoint(checkpoint_path)
                    logger.info(
                        f"\nSample {checkpoint_idx + 1}/{len(checkpoints_to_use)}: "
                        f"Using model from epoch {checkpoint_epoch}"
                    )
                else:
                    logger.info(f"\nSample 1/1: Using current model (no checkpoints saved)")

                for batch in dataloader:
                    batch = batch.to(self.device)
                    batch_size = batch.num_graphs

                    # Use abstract interface — consistent with the training loop.
                    power_batch, assoc_batch = self.primal_forward(batch)      # (B, m), (B, m, n)
                    _, _, per_user_metrics = self.compute_constraints(
                        power_batch, assoc_batch, batch
                    )  # per_user_metrics: (B, n)

                    for idx in range(batch_size):
                        net_id = batch.network_id[idx].item()

                        power = power_batch[idx]          # (m,) on device
                        ergodic_rates = per_user_metrics[idx]  # (n,) on device

                        if net_id not in samples:
                            samples[net_id] = {
                                # Static per-network tensors: capture once from first checkpoint pass.
                                'H_instantaneous': batch.H_instantaneous[idx].cpu().numpy(),
                                'associations': assoc_batch[idx].cpu().numpy(),
                                'power_samples': [],
                                'rate_samples': [],
                            }
                            if hasattr(batch, 'network_seed') and batch.network_seed is not None:
                                samples[net_id]['network_seed'] = batch.network_seed[idx].item()
                            samples[net_id].update(self._extra_sample_metadata(net_id))

                        power_np = power.cpu().numpy()
                        rates_np = ergodic_rates.cpu().numpy()
                        sum_rate = ergodic_rates.sum().item()
                        min_rate = ergodic_rates.min().item()
                        sample_data = {
                            'power': power_np,
                            'rates': rates_np,
                            'sum_rate': sum_rate,
                            'min_rate': min_rate,
                            'checkpoint_epoch': int(checkpoint_epoch),
                        }
                        samples[net_id]['power_samples'].append(sample_data)
                        samples[net_id]['rate_samples'].append({
                            'rates': rates_np,
                            'sum_rate': sum_rate,
                            'min_rate': min_rate,
                            'checkpoint_epoch': int(checkpoint_epoch),
                        })

        return samples

    def analyze_sample_quality(
        self,
        samples: Dict,
        dataloader: DataLoader,
    ) -> Dict:
        """
        Analyze quality of collected WRA samples with stochastic and deterministic policies.
        """
        quality_report = {}

        for net_id, network_data in samples.items():
            sample_list = network_data['power_samples']
            H_inst = network_data['H_instantaneous']
            associations = network_data['associations']

            rates_all = np.array([s['rates'] for s in sample_list])   # (num_samples, n)
            powers_all = np.array([s['power'] for s in sample_list])  # (num_samples, m)
            n_receivers = int(rates_all.shape[1]) if rates_all.ndim == 2 else 0
            r_min_vec = self._r_min_for_network(net_id, n_receivers).detach().cpu().numpy()

            # === STOCHASTIC POLICY: Average rates across samples ===
            rates_averaged = rates_all.mean(axis=0)
            stochastic_min = rates_averaged.min()
            stochastic_1st_percentile = np.percentile(rates_averaged, 1)
            stochastic_5th_percentile = np.percentile(rates_averaged, 5)
            stochastic_receiver_violations = rates_averaged < r_min_vec

            # === DETERMINISTIC POLICY: Averaged power allocation ===
            power_averaged = powers_all.mean(axis=0)

            power_tensor = torch.from_numpy(power_averaged).float()
            H_tensor = torch.from_numpy(H_inst).float()
            assoc_tensor = torch.from_numpy(associations).float()

            rates_deterministic = compute_ergodic_rates(
                power_allocation=power_tensor,
                H_instantaneous=H_tensor,
                associations=assoc_tensor,
                noise_var=self.system_params['noise_var'],
            ).numpy()

            deterministic_min = rates_deterministic.min()
            deterministic_1st_percentile = np.percentile(rates_deterministic, 1)
            deterministic_5th_percentile = np.percentile(rates_deterministic, 5)
            deterministic_receiver_violations = rates_deterministic < r_min_vec

            # === LEGACY METRICS ===
            satisfied = rates_all >= r_min_vec[None, :]
            satisfaction_rate = satisfied.mean()
            violations = np.maximum(0.0, r_min_vec[None, :] - rates_all)
            mean_violation_slack = violations.mean()
            mean_rates = rates_all.mean(axis=0)
            min_rates = rates_all.min(axis=0)

            quality_report[net_id] = {
                'r_min_per_receiver': r_min_vec.tolist(),
                'constraint_satisfaction_rate': float(satisfaction_rate),
                'mean_violation_slack': float(mean_violation_slack),
                'mean_rate_per_user': mean_rates.tolist(),
                'min_rate_per_user': min_rates.tolist(),
                'overall_mean_rate': float(rates_all.mean()),
                'overall_min_rate': float(rates_all.min()),
                'stochastic_policy': {
                    'min_rate': float(stochastic_min),
                    '1st_percentile': float(stochastic_1st_percentile),
                    '5th_percentile': float(stochastic_5th_percentile),
                    'rates_per_user': rates_averaged.tolist(),
                    'num_receivers_violated': int(np.sum(stochastic_receiver_violations)),
                    'any_receiver_violated': bool(np.any(stochastic_receiver_violations)),
                },
                'deterministic_policy': {
                    'min_rate': float(deterministic_min),
                    '1st_percentile': float(deterministic_1st_percentile),
                    '5th_percentile': float(deterministic_5th_percentile),
                    'rates_per_user': rates_deterministic.tolist(),
                    'power': power_averaged.tolist(),
                    'num_receivers_violated': int(np.sum(deterministic_receiver_violations)),
                    'any_receiver_violated': bool(np.any(deterministic_receiver_violations)),
                },
            }

        satisfaction_rates = [r['constraint_satisfaction_rate'] for r in quality_report.values()]
        stochastic_mins = [r['stochastic_policy']['min_rate'] for r in quality_report.values()]
        deterministic_mins = [r['deterministic_policy']['min_rate'] for r in quality_report.values()]
        stochastic_any_violations = [
            bool(r['stochastic_policy'].get('any_receiver_violated', False))
            for r in quality_report.values()
        ]
        deterministic_any_violations = [
            bool(r['deterministic_policy'].get('any_receiver_violated', False))
            for r in quality_report.values()
        ]
        r_min_scalar = self._r_min_scalar_for_logging()
        num_samples_per_network = 0
        if samples:
            first_key = next(iter(samples.keys()))
            num_samples_per_network = len(samples[first_key].get('power_samples', []))

        logger.info(f"\n{'='*70}")
        logger.info(f"QUALITY SUMMARY")
        logger.info(f"{'='*70}")
        logger.info(f"\nLegacy Metrics (per-sample):")
        logger.info(f"  Mean satisfaction rate: {np.mean(satisfaction_rates):.2%}")
        logger.info(f"  Min satisfaction rate: {np.min(satisfaction_rates):.2%}")
        logger.info(f"  Networks with > 80% satisfaction: {np.sum(np.array(satisfaction_rates) > 0.8)} / {len(satisfaction_rates)}")
        logger.info(f"  Networks with > 95% satisfaction: {np.sum(np.array(satisfaction_rates) > 0.95)} / {len(satisfaction_rates)}")
        logger.info(f"\nStochastic Policy (time-sharing between {num_samples_per_network} models):")
        logger.info(f"  Min rate across networks: {np.min(stochastic_mins):.4f} bits/s/Hz")
        logger.info(f"  Mean min rate: {np.mean(stochastic_mins):.4f} bits/s/Hz")
        if r_min_scalar is not None:
            logger.info(
                f"  Constraint violations (any receiver, r_min={r_min_scalar}): "
                f"{np.sum(stochastic_any_violations)} / {len(stochastic_any_violations)} networks"
            )
        else:
            logger.info(
                "  Constraint violations (any receiver): "
                f"{np.sum(stochastic_any_violations)} / {len(stochastic_any_violations)} networks"
            )
        logger.info(f"\nDeterministic Policy (averaged power allocation):")
        logger.info(f"  Min rate across networks: {np.min(deterministic_mins):.4f} bits/s/Hz")
        logger.info(f"  Mean min rate: {np.mean(deterministic_mins):.4f} bits/s/Hz")
        if r_min_scalar is not None:
            logger.info(
                f"  Constraint violations (any receiver, r_min={r_min_scalar}): "
                f"{np.sum(deterministic_any_violations)} / {len(deterministic_any_violations)} networks"
            )
        else:
            logger.info(
                "  Constraint violations (any receiver): "
                f"{np.sum(deterministic_any_violations)} / {len(deterministic_any_violations)} networks"
            )
        logger.info(f"\nPolicy Comparison (Stochastic vs Deterministic):")
        logger.info(f"  Min rate improvement: {np.mean(stochastic_mins) - np.mean(deterministic_mins):.4f} bits/s/Hz")
        logger.info(f"  Networks where stochastic > deterministic: {np.sum(np.array(stochastic_mins) > np.array(deterministic_mins))} / {len(stochastic_mins)}")
        logger.info(f"{'='*70}")

        return quality_report


class WRAConditionalPrimalDualTrainer(WRAPrimalDualTrainer):
    """WRA trainer variant that attaches constraint-profile metadata to samples."""

    def __init__(self, *, constraint_profile_dataset, **kwargs):
        super().__init__(**kwargs)
        self.constraint_profile_dataset = constraint_profile_dataset

    def _constraint_profile_info(self, network_id: int) -> Optional[Dict[str, object]]:
        base_id, profile_id = self.constraint_profile_dataset.decode_expanded_id(network_id)
        metadata = {
            'base_network_id': int(base_id),
            'constraint_profile_id': int(profile_id),
        }
        if hasattr(self.constraint_profile_dataset, "get_profile_name"):
            metadata['constraint_profile_name'] = self.constraint_profile_dataset.get_profile_name(profile_id)
        return metadata

    def _select_visualization_network_ids(self, max_base_networks: int = 2) -> list[int]:
        """
        Pick representative base networks and include all profiles for each base.
        """
        available_ids = sorted(int(net_id) for net_id in self._primal_metadata_entries.keys())
        if not available_ids:
            return []

        base_to_profiles: dict[int, list[tuple[int, int]]] = {}
        for net_id in available_ids:
            info = self._constraint_profile_info(net_id)
            if info is None:
                continue
            base_id = int(info['base_network_id'])
            profile_id = int(info['constraint_profile_id'])
            base_to_profiles.setdefault(base_id, []).append((profile_id, net_id))

        selected_base_ids = sorted(base_to_profiles.keys())[:max_base_networks]
        selected_network_ids: list[int] = []
        for base_id in selected_base_ids:
            for _, net_id in sorted(base_to_profiles[base_id], key=lambda pair: pair[0]):
                selected_network_ids.append(int(net_id))
        return selected_network_ids

    def _visualization_network_label(self, network_id: int) -> str:
        info = self._constraint_profile_info(network_id)
        if info is None:
            return super()._visualization_network_label(network_id)
        base_id = int(info['base_network_id'])
        profile_id = int(info['constraint_profile_id'])
        profile_name = info.get('constraint_profile_name')
        if profile_name is None:
            return f"Network {int(network_id)} | base={base_id} profile={profile_id}"
        return (
            f"Network {int(network_id)} | base={base_id} "
            f"profile={profile_id} ({profile_name})"
        )

    def _extra_sample_metadata(self, net_id: int) -> Dict:
        profile_info = self._constraint_profile_info(net_id)
        if profile_info is None:
            return {}
        profile_id = int(profile_info['constraint_profile_id'])
        metadata = {
            'base_network_id': int(profile_info['base_network_id']),
            'constraint_profile_id': profile_id,
            'r_min_per_receiver': np.asarray(
                self.constraint_profile_dataset.get_profile_vector(profile_id),
                dtype=np.float32,
            ),
        }
        if profile_info.get('constraint_profile_name') is not None:
            metadata['constraint_profile_name'] = profile_info['constraint_profile_name']
        return metadata
