"""Utility functions for wireless resource allocation computations."""

import torch
import numpy as np
from typing import Union


def receiver_to_transmitter_power(
    power_per_receiver: torch.Tensor,
    associations: Union[torch.Tensor, np.ndarray],
) -> torch.Tensor:
    """Convert per-receiver power allocations to per-transmitter powers.
    
    In one-to-one pairing, each transmitter sends to exactly one receiver.
    The associations matrix (m, n) indicates pairing: associations[i,j]=1 means TX i → RX j.
    Transmitter i gets the power allocated to its paired receiver.
    
    Parameters
    ----------
    power_per_receiver : torch.Tensor
        Power allocations per receiver, shape (n,)
    associations : torch.Tensor or np.ndarray
        TX-RX pairing matrix, shape (m, n), where associations[i,j]=1 means TX i → RX j
    
    Returns
    -------
    torch.Tensor
        Power per transmitter, shape (m,)
    
    Examples
    --------
    >>> power_rx = torch.tensor([0.5, 0.8, 0.6])  # 3 receivers
    >>> assoc = torch.tensor([[1, 0, 0],
    ...                        [0, 1, 0],
    ...                        [0, 0, 1]])  # Identity pairing
    >>> receiver_to_transmitter_power(power_rx, assoc)
    tensor([0.5000, 0.8000, 0.6000])
    """
    if isinstance(associations, np.ndarray):
        associations = torch.from_numpy(associations).to(power_per_receiver.device)
    
    # associations: (m, n), power_per_receiver: (n,)
    # power_per_transmitter[i] = power_per_receiver[j] where associations[i,j]=1
    power_per_transmitter = associations @ power_per_receiver  # (m,)
    
    return power_per_transmitter


def compute_sinr(
    power: torch.Tensor,
    H_inst: torch.Tensor,
    associations: torch.Tensor,
    noise_var: float,
) -> torch.Tensor:
    """Compute SINR for each receiver.
    
    SINR_j = (direct_signal_j) / (interference_j + noise_var)
    where:
        direct_signal_j = power_i * |H[i,j]|^2 for paired TX i
        interference_j = sum_{k != i} power_k * |H[k,j]|^2
    
    Parameters
    ----------
    power : torch.Tensor
        Transmit powers, shape (m,)
    H_inst : torch.Tensor
        Instantaneous channel power gains (already squared), shape (m, n)
    associations : torch.Tensor
        TX-RX pairing matrix, shape (m, n)
    noise_var : float
        Noise variance
    
    Returns
    -------
    torch.Tensor
        SINR per receiver, shape (n,)
    """
    m, n = H_inst.shape
    
    # H_inst is already power gains (|H|^2), no need to square again
    channel_gains = H_inst  # (m, n)
    
    # Received power at each receiver from each transmitter
    # received_power[i,j] = power[i] * |H[i,j]|^2
    received_power = power.unsqueeze(1) * channel_gains  # (m, n)
    
    # Direct signal: power from paired transmitter
    direct_signal = (associations * received_power).sum(dim=0)  # (n,)
    
    # Total received power (signal + interference)
    total_power = received_power.sum(dim=0)  # (n,)
    
    # Interference: total - direct
    interference = total_power - direct_signal  # (n,)
    
    # SINR
    sinr = direct_signal / (interference + noise_var)  # (n,)
    
    return sinr


def compute_ergodic_rates(
    power: torch.Tensor,
    H_samples: torch.Tensor,
    associations: Union[torch.Tensor, np.ndarray],
    noise_var: float,
) -> torch.Tensor:
    """Compute ergodic rates by averaging over channel realizations.

    Fully vectorized over T: no Python loop over timesteps.

    For each receiver j:
        R_j = E[log2(1 + SINR_j)] ≈ (1/T) * sum_t log2(1 + SINR_j^(t))

    Parameters
    ----------
    power : torch.Tensor
        Transmit powers, shape (m,)
    H_samples : torch.Tensor
        Instantaneous channel power gains (already squared), shape (T, m, n)
    associations : torch.Tensor or np.ndarray
        TX-RX pairing matrix, shape (m, n)
    noise_var : float
        Noise variance

    Returns
    -------
    torch.Tensor
        Ergodic rates per receiver, shape (n,)
    """
    if isinstance(associations, np.ndarray):
        associations = torch.from_numpy(associations).to(power.device)

    # Avoid materialising received_power (T, m, n) to keep memory footprint low.
    total_received = torch.einsum("tmn,m->tn", H_samples, power)  # (T, n)
    direct_signal = torch.einsum("tmn,mn,m->tn", H_samples, associations, power)  # (T, n)
    interference = total_received - direct_signal  # (T, n)

    # Shannon capacity per timestep; average gives the ergodic rate.
    sinr = direct_signal / (interference + noise_var)  # (T, n)
    return torch.log2(1.0 + sinr).mean(dim=0)  # (n,)


def compute_ergodic_rates_batched(
    powers: torch.Tensor,
    H_samples: torch.Tensor,
    associations: Union[torch.Tensor, np.ndarray],
    noise_var: float,
) -> torch.Tensor:
    """Compute ergodic rates for multiple power vectors in one pass.

    Parameters
    ----------
    powers : torch.Tensor
        Stacked transmit powers, shape (P, m).
    H_samples : torch.Tensor
        Instantaneous channel power gains (already squared), shape (T, m, n).
    associations : torch.Tensor or np.ndarray
        TX-RX pairing matrix, shape (m, n).
    noise_var : float
        Noise variance.

    Returns
    -------
    torch.Tensor
        Ergodic rates per receiver for each power vector, shape (P, n).
    """
    if isinstance(associations, np.ndarray):
        associations = torch.from_numpy(associations).to(powers.device)

    total_received = torch.einsum("tmn,pm->ptn", H_samples, powers)  # (P, T, n)
    direct_signal = torch.einsum("tmn,mn,pm->ptn", H_samples, associations, powers)  # (P, T, n)
    interference = total_received - direct_signal  # (P, T, n)
    sinr = direct_signal / (interference + noise_var)  # (P, T, n)
    return torch.log2(1.0 + sinr).mean(dim=1)  # (P, n)


def jains_fairness_index(values: torch.Tensor) -> float:
    """Compute Jain's fairness index.
    
    JFI = (sum(x_i))^2 / (n * sum(x_i^2))
    
    Range: [0, 1], where 1 is perfectly fair.
    
    Parameters
    ----------
    values : torch.Tensor
        Values to measure fairness of (e.g., rates), shape (n,)
    
    Returns
    -------
    float
        Jain's fairness index
    """
    n = values.numel()
    if n == 0:
        return 1.0
    
    sum_x = values.sum()
    sum_x_sq = (values ** 2).sum()
    
    if sum_x_sq == 0:
        return 1.0
    
    jfi = (sum_x ** 2) / (n * sum_x_sq)
    return jfi.item()


def clamp_power(power: torch.Tensor, P_max: float) -> torch.Tensor:
    """Clamp power allocations to [0, P_max].
    
    Parameters
    ----------
    power : torch.Tensor
        Power allocations
    P_max : float
        Maximum power constraint
    
    Returns
    -------
    torch.Tensor
        Clamped power allocations
    """
    return torch.clamp(power, min=0.0, max=P_max)


def compute_violation_rate(values: torch.Tensor, threshold: float, lower_bound: bool = True) -> float:
    """Compute rate of constraint violations.
    
    Parameters
    ----------
    values : torch.Tensor
        Values to check (e.g., rates, powers)
    threshold : float
        Constraint threshold
    lower_bound : bool
        If True, check values >= threshold (e.g., r_min)
        If False, check values <= threshold (e.g., P_max)
    
    Returns
    -------
    float
        Violation rate [0, 1]
    """
    if lower_bound:
        violations = values < threshold
    else:
        violations = values > threshold
    
    return violations.float().mean().item()
