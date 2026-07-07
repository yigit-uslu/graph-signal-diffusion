"""
Dual variable optimizer for primal-dual power allocation training.

This module maintains and updates dual variables (Lagrange multipliers) for
min-rate constraints in the wireless resource allocation problem.
"""

import torch
import numpy as np
from typing import Optional, Dict, Tuple, Sequence, TypeAlias
from collections import defaultdict
import abc

RMinArrayLike: TypeAlias = (
    torch.Tensor
    | np.ndarray
    | Sequence[float]
    | Sequence[Sequence[float]]
)
RMinInput: TypeAlias = float | int | RMinArrayLike


class DualOptimizer(abc.ABC):
    """
    Dual variable optimizer using projected subgradient method.
    
    Maintains dual variables λ ∈ R^(N_networks × n) where:
    - N_networks: Number of training networks
    - n: Number of receivers per network (assumes fixed size)
    - λ[net_id, i]: Dual variable for min-rate constraint of receiver i in network net_id
    
    Update rule (projected subgradient ascent):
        λ[net_id, i] ← max(0, λ[net_id, i] + α * (r_min[net_id, i] - R_i))
    
    Parameters
    ----------
    num_networks : int
        Number of training networks
    num_receivers : int
        Number of receivers per network (fixed size)
    r_min : RMinInput
        Minimum rate requirement(s) in bits/s/Hz. Accepted forms:
        1. Scalar: one shared threshold for all networks/receivers.
        2. Vector (num_networks,): per-network scalar thresholds (broadcast to receivers).
        3. Matrix (num_networks, num_receivers): per-network/per-receiver thresholds.
    alpha_dual : float
        Dual learning rate / step size (default: 0.01)
    update_frequency : int
        Update duals every K gradient steps (default: 10)
    momentum : float
        Momentum coefficient for dual updates (default: 0.0, range [0, 1))
        Higher values encourage faster oscillation around optimal values
    device : str or torch.device
        Device for dual variables (default: 'cpu')
    
    Examples
    --------
    >>> dual_opt = DualOptimizer(num_networks=100, num_receivers=10, r_min=0.7)
    >>> rates = torch.rand(1, 10)  # Ergodic rates for one network batch
    >>> net_ids = torch.tensor([0], dtype=torch.long)
    >>> should_update = dual_opt.step()
    >>> if should_update:
    >>>     dual_opt.update(network_ids=net_ids, rates=rates)
    >>> lambdas = dual_opt.get_duals(net_ids)

    >>> # Per-network thresholds (shape: num_networks)
    >>> r_min_vec = torch.linspace(0.4, 0.8, steps=100)
    >>> dual_opt = DualOptimizer(num_networks=100, num_receivers=10, r_min=r_min_vec)

    >>> # Per-network, per-receiver thresholds (shape: num_networks x num_receivers)
    >>> r_min_table = torch.full((100, 10), 0.7)
    >>> r_min_table[:, 0] = 0.9
    >>> dual_opt = DualOptimizer(num_networks=100, num_receivers=10, r_min=r_min_table)
    """
    
    def __init__(
        self,
        num_networks: int,
        num_receivers: int,
        r_min: RMinInput = 0.7,
        alpha_dual: float = 0.01,
        update_frequency: int = 10,
        momentum: float = 0.0,
        device: str = 'cpu',
    ):
        self.num_networks = num_networks
        self.num_receivers = num_receivers
        self.r_min_input = r_min
        self.alpha_dual = alpha_dual
        self.update_frequency = update_frequency
        self.momentum = momentum
        self.device = torch.device(device)
        (
            self.r_min_table,
            self._is_scalar_r_min,
            self._scalar_r_min,
        ) = self._canonicalize_r_min(
            r_min=r_min,
            num_networks=num_networks,
            num_receivers=num_receivers,
            device=self.device,
        )
        # Backward-compatible mirror used by legacy callsites:
        # - float for homogeneous thresholds
        # - canonical (num_networks, num_receivers) table otherwise
        self.r_min = (
            float(self._scalar_r_min)
            if self._scalar_r_min is not None
            else self.r_min_table
        )
        
        # Initialize dual variables: λ[net_id, i] for each receiver i in network net_id
        # Start at 1.0 for better initial penalty on constraint violations
        self.lambdas = torch.ones(
            (num_networks, num_receivers),
            dtype=torch.float32,
            device=self.device
        )
        
        # Initialize momentum buffer (velocity term for momentum updates)
        self.velocity = torch.zeros(
            (num_networks, num_receivers),
            dtype=torch.float32,
            device=self.device
        )
        
        # Track step counter for update frequency
        self.step_counter = 0
        
        # Track dual trajectories for oscillation analysis
        self.dual_history = defaultdict(list)  # {network_id: [λ_t1, λ_t2, ...]}
        
        # Track generic dual subgradients g = dL/dlambda (constraint slacks for standard formulations).
        self.dual_subgradient_history = defaultdict(list)  # {network_id: [g_t1, ...]}
        # Backward-compatible alias for legacy code paths.
        self.violation_history = self.dual_subgradient_history

    @staticmethod
    def _canonicalize_r_min(
        r_min: RMinInput,
        num_networks: int,
        num_receivers: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, bool, Optional[float]]:
        """
        Canonicalize min-rate thresholds to shape (num_networks, num_receivers).

        Broadcasting rules:
        - scalar -> full table
        - 1D (num_networks,) -> expand across receivers
        - 1D (num_receivers,) allowed only when num_networks == 1
        - 2D must match (num_networks, num_receivers)
        """
        if isinstance(r_min, (int, float)):
            table = torch.full(
                (num_networks, num_receivers),
                float(r_min),
                dtype=torch.float32,
                device=device,
            )
        else:
            tensor = torch.as_tensor(r_min, dtype=torch.float32, device=device)
            if tensor.ndim == 0:
                table = torch.full(
                    (num_networks, num_receivers),
                    float(tensor.item()),
                    dtype=torch.float32,
                    device=device,
                )
            elif tensor.ndim == 1:
                if tensor.shape[0] == num_networks:
                    table = tensor.unsqueeze(1).expand(num_networks, num_receivers).clone()
                elif num_networks == 1 and tensor.shape[0] == num_receivers:
                    table = tensor.unsqueeze(0).clone()
                else:
                    raise ValueError(
                        "1D r_min tensor must have length num_networks "
                        f"(={num_networks}) or, for num_networks=1, length num_receivers "
                        f"(={num_receivers}). Got shape {tuple(tensor.shape)}."
                    )
            elif tensor.ndim == 2:
                if tensor.shape != (num_networks, num_receivers):
                    raise ValueError(
                        "2D r_min tensor must have shape "
                        f"({num_networks}, {num_receivers}); got {tuple(tensor.shape)}."
                    )
                table = tensor.clone()
            else:
                raise ValueError(
                    "r_min must be scalar, 1D, or 2D. "
                    f"Got ndim={tensor.ndim} with shape {tuple(tensor.shape)}."
                )

        first = float(table[0, 0].item())
        is_scalar = bool(torch.allclose(table, torch.full_like(table, first)))
        scalar = first if is_scalar else None
        return table, is_scalar, scalar

    def r_min_per_network(self, network_ids: torch.Tensor) -> torch.Tensor:
        """Return per-network/per-receiver min-rate thresholds, shape (B, num_receivers)."""
        return self.r_min_table[network_ids]

    def is_scalar_r_min(self) -> bool:
        """Whether current thresholds are homogeneous across networks/receivers."""
        return bool(self._is_scalar_r_min)

    def scalar_r_min_or_none(self) -> Optional[float]:
        """Return scalar threshold if homogeneous, else None."""
        return self._scalar_r_min
    
    def step(self) -> bool:
        """
        Increment step counter and check if duals should be updated.
        
        Returns
        -------
        should_update : bool
            True if dual update should be performed this step
        """
        self.step_counter += 1
        return (self.step_counter % self.update_frequency) == 0
    
    def _apply_update(
        self,
        network_ids: torch.Tensor,
        dual_subgradients: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Core momentum + projection update, shared by update() and update_from_gradients().

        Parameters
        ----------
        network_ids : torch.Tensor
            Network IDs, shape (batch_size,)
        dual_subgradients : torch.Tensor
            Per-receiver subgradients ∂L/∂λ = g(x) = constraint slacks,
            shape (batch_size, num_receivers). Positive values mean the
            constraint is violated (r_min > R_i).
        """
        # Momentum-based projected subgradient ascent:
        #   v_t = β * v_{t-1} + α * g
        #   λ_t = max(0, λ_{t-1} + v_t)
        for idx, net_id in enumerate(network_ids):
            net_id = net_id.item()
            velocity_new = (
                self.momentum * self.velocity[net_id]
                + self.alpha_dual * dual_subgradients[idx]
            )
            lambda_new = torch.clamp(self.lambdas[net_id] + velocity_new, min=0.0)
            self.lambdas[net_id] = lambda_new
            self.velocity[net_id] = velocity_new
            self.dual_history[net_id].append(lambda_new.detach().cpu().numpy().copy())
            self.dual_subgradient_history[net_id].append(
                dual_subgradients[idx].detach().cpu().numpy().copy()
            )

        updated_lambdas = self.lambdas[network_ids]
        mean_dual_subgradient = dual_subgradients.mean().item()
        positive_subgradient_fraction = (dual_subgradients > 0).float().mean().item()
        return {
            'mean_lambda': updated_lambdas.mean().item(),
            'std_lambda': updated_lambdas.std().item(),
            'max_lambda': updated_lambdas.max().item(),
            # Generic names.
            'mean_dual_subgradient': mean_dual_subgradient,
            'positive_subgradient_fraction': positive_subgradient_fraction,
            # Backward-compatible aliases (for standard g = constraint slack conventions).
            'mean_slack': mean_dual_subgradient,
            'violation_fraction': positive_subgradient_fraction,
        }

    def update(
        self,
        network_ids: torch.Tensor,
        rates: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Update dual variables for a batch of networks using projected subgradient.

        Computes per-network slacks g = r_min_batch - rates and delegates to _apply_update().

        Parameters
        ----------
        network_ids : torch.Tensor
            Network IDs in the batch, shape (batch_size,)
        rates : torch.Tensor
            Ergodic rates for each receiver, shape (batch_size, num_receivers)
        """
        batch_size = network_ids.shape[0]
        assert rates.shape == (batch_size, self.num_receivers), \
            f"Expected rates shape ({batch_size}, {self.num_receivers}), got {rates.shape}"
        r_min_batch = self.r_min_per_network(network_ids)  # (batch_size, num_receivers)
        constraint_slacks = r_min_batch - rates  # (batch_size, num_receivers)
        return self._apply_update(network_ids, constraint_slacks)

    def update_from_gradients(
        self,
        network_ids: torch.Tensor,
        gradients: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Update dual variables from pre-computed subgradients ∂L/∂λ.

        For a standard Lagrangian L = f(x) + λᵀg(x), the dual subgradient is
        ∂L/∂λ = g(x) = constraint slacks (r_min - R for min-rate constraints).
        This method accepts those pre-computed gradients directly, bypassing
        the r_min - rates computation in update().  It enables callers to
        obtain subgradients via torch.autograd rather than by hand.

        Parameters
        ----------
        network_ids : torch.Tensor
            Network IDs in the batch, shape (batch_size,)
        gradients : torch.Tensor
            Pre-computed ∂L/∂λ per network, shape (batch_size, num_receivers).
            Must already be detached from the primal computational graph.
        """
        batch_size = network_ids.shape[0]
        assert gradients.shape == (batch_size, self.num_receivers), \
            f"Expected gradients shape ({batch_size}, {self.num_receivers}), got {gradients.shape}"
        return self._apply_update(network_ids, gradients)
    
    def get_duals(
        self,
        network_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get dual variables for a batch of networks.
        
        Parameters
        ----------
        network_ids : torch.Tensor
            Network IDs, shape (batch_size,)
        
        Returns
        -------
        lambdas : torch.Tensor
            Dual variables, shape (batch_size, num_receivers)
        """
        return self.lambdas[network_ids]
    
    def get_dual_history(
        self,
        network_id: int,
    ) -> np.ndarray:
        """
        Get dual variable history for a specific network.
        
        Parameters
        ----------
        network_id : int
            Network ID
        
        Returns
        -------
        history : np.ndarray
            Dual trajectory, shape (num_updates, num_receivers)
        """
        if network_id not in self.dual_history:
            return np.array([])
        return np.array(self.dual_history[network_id])
    
    def get_violation_history(
        self,
        network_id: int,
    ) -> np.ndarray:
        """
        Get constraint violation/slack history for a specific network.
        
        Parameters
        ----------
        network_id : int
            Network ID
        
        Returns
        -------
        history : np.ndarray
            Violation trajectory, shape (num_updates, num_receivers)
        """
        if network_id not in self.violation_history:
            return np.array([])
        return np.array(self.violation_history[network_id])

    def get_dual_subgradient_history(
        self,
        network_id: int,
    ) -> np.ndarray:
        """
        Get dual-subgradient history g = dL/dlambda for a specific network.

        Parameters
        ----------
        network_id : int
            Network ID

        Returns
        -------
        history : np.ndarray
            Dual-subgradient trajectory, shape (num_updates, num_receivers)
        """
        if network_id not in self.dual_subgradient_history:
            return np.array([])
        return np.array(self.dual_subgradient_history[network_id])
    
    def analyze_oscillations(
        self,
        network_id: int,
        window_size: int = 50,
    ) -> Dict[str, float]:
        """
        Analyze dual oscillation patterns for a specific network.
        
        Parameters
        ----------
        network_id : int
            Network ID
        window_size : int
            Window size for computing rolling statistics (default: 50)
        
        Returns
        -------
        stats : dict
            Oscillation statistics:
            - 'mean': Mean dual value over recent window
            - 'std': Standard deviation over recent window
            - 'oscillation_amplitude': Max - Min over recent window
            - 'converged': Whether oscillations have stabilized
        """
        history = self.get_dual_history(network_id)
        
        if len(history) < window_size:
            return {
                'mean': 0.0,
                'std': 0.0,
                'oscillation_amplitude': 0.0,
                'converged': False,
            }
        
        # Compute statistics over recent window
        recent = history[-window_size:]  # (window_size, num_receivers)
        
        mean_dual = np.mean(recent)
        std_dual = np.std(recent)
        amplitude = np.max(recent) - np.min(recent)
        
        # Heuristic for convergence: std should be relatively stable
        # Compare std of first half vs second half of window
        if len(recent) >= 2 * (window_size // 2):
            std_first_half = np.std(recent[:window_size//2])
            std_second_half = np.std(recent[window_size//2:])
            std_change_rate = abs(std_second_half - std_first_half) / (std_first_half + 1e-6)
            converged = std_change_rate < 0.01  # Less than 1% change
        else:
            converged = False
        
        return {
            'mean': float(mean_dual),
            'std': float(std_dual),
            'oscillation_amplitude': float(amplitude),
            'converged': converged,
        }
    
    def state_dict(self) -> Dict:
        """
        Get state dictionary for checkpointing.
        
        Returns
        -------
        state : dict
            State dictionary containing dual variables, histories, and
            canonical threshold metadata (including r_min_table).
        """
        return {
            'lambdas': self.lambdas.cpu(),
            'velocity': self.velocity.cpu(),
            'step_counter': self.step_counter,
            'dual_history': dict(self.dual_history),
            'dual_subgradient_history': dict(self.dual_subgradient_history),
            # Keep legacy key for old checkpoints/tools.
            'violation_history': dict(self.dual_subgradient_history),
            'num_networks': self.num_networks,
            'num_receivers': self.num_receivers,
            'r_min': self._scalar_r_min,
            'r_min_table': self.r_min_table.cpu(),
            '_is_scalar_r_min': self._is_scalar_r_min,
            '_scalar_r_min': self._scalar_r_min,
            'alpha_dual': self.alpha_dual,
            'update_frequency': self.update_frequency,
            'momentum': self.momentum,
        }
    
    def load_state_dict(self, state: Dict) -> None:
        """
        Load state from checkpoint.
        
        Parameters
        ----------
        state : dict
            State dictionary from state_dict()
        """
        self.lambdas = state['lambdas'].to(self.device)
        self.velocity = state.get('velocity', torch.zeros_like(self.lambdas)).to(self.device)
        self.step_counter = state['step_counter']
        self.dual_history = defaultdict(list, state['dual_history'])
        self.dual_subgradient_history = defaultdict(
            list,
            state.get('dual_subgradient_history', state.get('violation_history', {})),
        )
        # Backward-compatible alias for legacy code paths.
        self.violation_history = self.dual_subgradient_history
        
        # Verify dimensions match
        assert state['num_networks'] == self.num_networks
        assert state['num_receivers'] == self.num_receivers
        assert state['alpha_dual'] == self.alpha_dual
        assert state['update_frequency'] == self.update_frequency
        self.momentum = state.get('momentum', 0.0)

        if 'r_min_table' in state:
            loaded_r_min_table = state['r_min_table'].to(self.device)
            if loaded_r_min_table.shape != (self.num_networks, self.num_receivers):
                raise ValueError(
                    "Loaded r_min_table shape mismatch: "
                    f"expected {(self.num_networks, self.num_receivers)}, "
                    f"got {tuple(loaded_r_min_table.shape)}."
                )
            self.r_min_table = loaded_r_min_table
        else:
            # Backward compatibility: old checkpoints only persisted scalar r_min.
            legacy_r_min = state.get('r_min', self._scalar_r_min if self._scalar_r_min is not None else 0.0)
            (
                self.r_min_table,
                _,
                _,
            ) = self._canonicalize_r_min(
                r_min=legacy_r_min,
                num_networks=self.num_networks,
                num_receivers=self.num_receivers,
                device=self.device,
            )

        # Keep scalar metadata derived from canonical table.
        first = float(self.r_min_table[0, 0].item())
        self._is_scalar_r_min = bool(torch.allclose(self.r_min_table, torch.full_like(self.r_min_table, first)))
        self._scalar_r_min = first if self._is_scalar_r_min else None
        self.r_min = float(self._scalar_r_min) if self._scalar_r_min is not None else self.r_min_table
    
    def reset(self) -> None:
        """Reset all dual variables to zero and clear history."""
        self.lambdas.zero_()
        self.velocity.zero_()
        self.step_counter = 0
        self.dual_history.clear()
        self.dual_subgradient_history.clear()
    
    def get_summary_statistics(self) -> Dict[str, float]:
        """
        Get summary statistics across all networks.
        
        Returns
        -------
        stats : dict
            Summary statistics:
            - 'mean_lambda': Mean dual value across all networks
            - 'std_lambda': Std of dual values
            - 'num_active_constraints': Number of receivers with λ > 0
        """
        return {
            'mean_lambda': self.lambdas.mean().item(),
            'std_lambda': self.lambdas.std().item(),
            'max_lambda': self.lambdas.max().item(),
            'num_active_constraints': (self.lambdas > 1e-6).sum().item(),
            'fraction_active': (self.lambdas > 1e-6).float().mean().item(),
        }
