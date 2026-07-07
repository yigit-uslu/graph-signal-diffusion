"""
Rate calculation utilities for wireless resource allocation.

This module provides functions to compute achievable rates using Shannon capacity
formula with interference from other transmitters.
"""

import torch
import numpy as np
from typing import Dict, Union, Optional


def calc_rates(
    p: torch.Tensor,
    gamma: torch.Tensor,
    h: torch.Tensor,
    noise_var: float,
    associations: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Calculate achievable rates for wireless users using Shannon capacity.
    
    Computes: R = log2(1 + SINR) where SINR = Signal / (Noise + Interference)
    
    Parameters
    ----------
    p : torch.Tensor
        Transmit power levels, shape (m,) or (m, 1)
        where m is number of transmitters
    gamma : torch.Tensor
        User scheduling decisions, shape (n,) or (n, 1)
        where n is number of receivers (gamma[i]=1 means RX i is scheduled)
    h : torch.Tensor
        Weighted adjacency matrix containing channel gains, shape (m+n, m+n)
        - h[:m, m:] contains TX→RX channel gains
        - h[m:, :m] contains RX←TX channel gains (same as TX→RX for symmetric channels)
    noise_var : float
        Noise variance (linear scale)
    associations : torch.Tensor, optional
        Binary association matrix, shape (m, n), where associations[i,j]=1 means TX i serves RX j
        If not provided, assumes h[:m, m:] already contains only direct channel gains
    
    Returns
    -------
    rates : torch.Tensor
        Achievable rates in bits/s/Hz, shape (n,)
    
    Notes
    -----
    The weighted adjacency matrix h is structured as:
        h = [TX-TX   TX-RX]
            [RX-TX   RX-RX]
    
    For rate calculation:
    - Signal power at RX j from its paired TX: p_i * h[i, m+j] where TX i serves RX j
    - Interference at RX j: sum over all other TXs k≠i: p_k * h[k, m+j]
    """
    # Reshape inputs
    p = p.view(-1)  # (m,)
    gamma = gamma.view(-1)  # (n,)
    m = p.shape[0]  # number of transmitters
    
    # Extract channel gain matrix: TX → RX
    H_tx_rx = h[:m, m:]  # (m, n)
    
    # Compute received power at each receiver from each transmitter
    # received_power[i,j] = p_i * H[i,j]
    received_power = p.unsqueeze(1) * H_tx_rx  # (m, n)
    
    # Compute direct signal and interference
    if associations is not None:
        # Direct signal: power from paired transmitter only
        direct_signal = (associations * received_power).sum(dim=0)  # (n,)
        
        # Total received power
        total_power = received_power.sum(dim=0)  # (n,)
        
        # Interference: total - direct
        interference = total_power - direct_signal  # (n,)
    else:
        # Legacy behavior: assume h already filtered by associations
        # This is incorrect but kept for backward compatibility
        direct_signal = received_power.sum(dim=0)  # (n,)
        interference = torch.zeros_like(direct_signal)
    
    # Apply user scheduling (gamma): only scheduled users contribute to rates
    direct_signal = direct_signal * gamma
    
    # Shannon capacity: R = log2(1 + S/(N+I))
    rates = torch.log2(1 + direct_signal / (noise_var + interference))  # (n,)
    
    return rates


def build_adjacency_matrix(
    H: np.ndarray,
    associations: np.ndarray
) -> np.ndarray:
    """
    Build weighted adjacency matrix from channel gains and associations.
    
    Parameters
    ----------
    H : np.ndarray
        Channel power gains, shape (m, n) where m = # transmitters, n = # receivers
        H[i, j] = power gain from TX_i to RX_j
    associations : np.ndarray
        Binary association matrix, shape (m, n)
        associations[i, j] = 1 if TX_i is paired with RX_j
    
    Returns
    -------
    h_adj : np.ndarray
        Weighted adjacency matrix, shape (m+n, m+n)
        Structured as: [TX-TX   TX-RX]
                      [RX-TX   RX-RX]
    """
    m, n = H.shape
    h_adj = np.zeros((m + n, m + n))
    
    # TX-RX channel gains (upper-right block)
    # Include all channel gains (signal/interference separation handled in calc_rates)
    h_adj[:m, m:] = H
    
    # RX-TX channel gains (lower-left block) - symmetric for wireless channels
    h_adj[m:, :m] = H.T
    
    return h_adj


def build_adjacency_matrix_torch(
    H: torch.Tensor,
    associations: torch.Tensor
) -> torch.Tensor:
    """
    Build weighted adjacency matrix from channel gains and associations (PyTorch version).
    Maintains gradient flow for differentiable rate computation.
    
    Parameters
    ----------
    H : torch.Tensor
        Channel power gains, shape (m, n) where m = # transmitters, n = # receivers
        H[i, j] = power gain from TX_i to RX_j
    associations : torch.Tensor
        Binary association matrix, shape (m, n)
        associations[i, j] = 1 if TX_i is paired with RX_j
    
    Returns
    -------
    h_adj : torch.Tensor
        Weighted adjacency matrix, shape (m+n, m+n)
        Structured as: [TX-TX   TX-RX]
                      [RX-TX   RX-RX]
    """
    m, n = H.shape
    h_adj = torch.zeros((m + n, m + n), dtype=H.dtype, device=H.device)
    
    # TX-RX channel gains (upper-right block)
    # Include all channel gains (signal/interference separation handled in calc_rates)
    h_adj[:m, m:] = H
    
    # RX-TX channel gains (lower-left block) - symmetric for wireless channels
    h_adj[m:, :m] = H.T
    
    return h_adj


def compute_ergodic_rates(
    power_allocation: torch.Tensor,
    H_instantaneous: torch.Tensor,
    associations: torch.Tensor,
    noise_var: float,
    user_scheduling: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute ergodic rates (time-averaged) for a network.

    Fully vectorized over T: no Python loop, no intermediate adjacency matrix.

    Parameters
    ----------
    power_allocation : torch.Tensor
        Power levels for each transmitter, shape (m,)
    H_instantaneous : torch.Tensor
        Instantaneous channel gains over time, shape (T, m, n)
        where T = # timesteps, m = # transmitters, n = # receivers
    associations : torch.Tensor
        Binary association matrix, shape (m, n)
    noise_var : float
        Noise variance (linear scale)
    user_scheduling : torch.Tensor, optional
        User scheduling decisions, shape (n,). Default: all ones (all users scheduled)

    Returns
    -------
    ergodic_rates : torch.Tensor
        Time-averaged rates per receiver, shape (n,)
    """
    # Avoid materialising received_power (T, m, n) to keep memory footprint low
    # on large-N/T regimes.
    total_received = torch.einsum("tmn,m->tn", H_instantaneous, power_allocation)  # (T, n)
    direct_signal = torch.einsum(
        "tmn,mn,m->tn", H_instantaneous, associations, power_allocation
    )  # (T, n)
    interference = total_received - direct_signal  # (T, n)

    # Apply user scheduling (zeros out unscheduled users before log).
    if user_scheduling is not None:
        direct_signal = direct_signal * user_scheduling[None]  # (T, n)

    # Shannon capacity per timestep; average gives the ergodic rate.
    rates = torch.log2(1.0 + direct_signal / (interference + noise_var))  # (T, n)
    return rates.mean(dim=0)  # (n,)


def compute_system_parameters(
    P_max_dBm: float = 10.0,
    bandwidth_Hz: float = 10e6,
    noise_psd_dBm_Hz: float = -174.0
) -> Dict[str, float]:
    """
    Compute system parameters for wireless network.
    
    Parameters
    ----------
    P_max_dBm : float
        Maximum transmit power in dBm (default: 10 dBm = 10 mW)
    bandwidth_Hz : float
        Bandwidth in Hz (default: 10 MHz)
    noise_psd_dBm_Hz : float
        Noise power spectral density in dBm/Hz (default: -174 dBm/Hz)
    
    Returns
    -------
    params : dict
        Dictionary containing:
        - 'P_max_watts': Maximum power in watts
        - 'noise_power_dBm': Noise power in dBm
        - 'noise_var': Noise variance (linear scale)
        - 'snr_dB': Signal-to-noise ratio in dB
    """
    # Convert P_max from dBm to watts
    P_max_watts = 10 ** ((P_max_dBm - 30) / 10)
    
    # Compute noise power in dBm
    noise_power_dBm = noise_psd_dBm_Hz + 10 * np.log10(bandwidth_Hz)
    
    # Convert noise power to watts (variance)
    noise_var = 10 ** ((noise_power_dBm - 30) / 10)
    
    # Compute SNR
    snr_dB = P_max_dBm - noise_power_dBm
    
    return {
        'P_max_watts': P_max_watts,
        'noise_power_dBm': noise_power_dBm,
        'noise_var': noise_var,
        'snr_dB': snr_dB,
    }


def compute_jains_fairness_index(rates: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Compute Jain's fairness index.
    
    Fairness index = (sum x_i)^2 / (n * sum x_i^2)
    
    Parameters
    ----------
    rates : torch.Tensor or np.ndarray
        Rates for each user, shape (n,)
    
    Returns
    -------
    fairness : float
        Fairness index in [0, 1], where 1 = perfect fairness
    """
    if isinstance(rates, torch.Tensor):
        rates = rates.detach().cpu().numpy()
    
    n = len(rates)
    numerator = np.sum(rates) ** 2
    denominator = n * np.sum(rates ** 2)
    
    return float(numerator / (denominator + 1e-12))
