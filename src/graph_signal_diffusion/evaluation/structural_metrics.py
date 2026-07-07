"""Shared structural-metric compute & visualisation functions.

These functions are **model-agnostic**: they operate on ``[B, T, N]``
standardised log-return tensors and produce diagnostic metrics /
figures suitable for comparing any generative model against real data.

Originally implemented inside ``GeometricRandomWalk``; extracted here so
that both the GRW and Diffusion baselines (and any future baseline) can
reuse the same code.

Window-overlap correction
~~~~~~~~~~~~~~~~~~~~~~~~~

Sliding-window datasets (stride 1) produce highly overlapping batches.
For **real** data, overlapping rows are identical calendar-date returns, so
the correlation matrix is computed on the true time series (just with
some dates over-represented).  For **generated** data each batch is an
independent sample; pooling overlapping windows *dilutes* any learned
cross-stock correlation by a factor of roughly ``T``.

Two complementary corrections are provided:

* **Phased Option A** (``window_stride`` + optional ``window_start_indices``)
  — windows are grouped into ``window_stride`` non-overlapping phases and the
  per-phase correlation / autocorrelation estimates are averaged.

  *Grouping strategy*:

  - If ``window_start_indices`` (a ``[B]`` integer tensor of window start
    positions in the original time series) is supplied, each window is
    assigned to phase ``start_index % window_stride``.  This is
    **order-robust**: it produces correct non-overlapping phases regardless
    of how the DataLoader shuffles or strides the batch dimension.
  - If ``window_start_indices`` is ``None``, positional slicing
    (``returns[p::window_stride]``) is used as a fallback.  This is correct
    when the batch dimension is chronologically ordered (e.g. test split
    with ``shuffle=False``) but silently degrades on shuffled splits.

  Callers should always supply ``window_start_indices`` extracted from
  ``metadata['timestamp']`` (the dataset's window-start index, available
  in every Data object as ``data.timestamp[::N]``).

* **Option C** (``*_per_trajectory`` variants) — compute the metric
  *inside each trajectory* separately, then average.  The per-trajectory
  estimate is very noisy (``T`` ≪ ``N``) but the mean over many
  trajectories converges and is entirely order-independent.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import torch
from graph_signal_diffusion.baselines.plot_utils import format_baseline_display_name

# ======================================================================
# Compute functions
# ======================================================================


def _autocorr_single_phase(
    r2: torch.Tensor, max_lag: int, N: int,
) -> torch.Tensor:
    """Normalized ACF helper for one set of (already-selected) windows.

    Uses a single global mean and normalises by the lag-0 autocovariance
    (variance) so that ρ(0) = 1 and ρ(l) ∈ [-1, 1].

    Returns ``[max_lag + 1]`` with index *l* = normalized ACF at lag *l*.
    """
    T = r2.shape[1]
    r2_flat = r2.reshape(-1, N)
    mu = r2_flat.mean(0)                      # [N] global mean
    gamma_0 = ((r2_flat - mu) ** 2).mean(0)   # [N] lag-0 autocovariance

    profile = torch.zeros(max_lag + 1)
    profile[0] = 1.0  # ρ(0) = γ(0)/γ(0) = 1
    for lag in range(1, max_lag + 1):
        r2_t = r2[:, :T - lag, :].reshape(-1, N)
        r2_tl = r2[:, lag:, :].reshape(-1, N)

        gamma_l = ((r2_t - mu) * (r2_tl - mu)).mean(0)  # [N]
        rho_l = gamma_l / (gamma_0 + 1e-8)               # [N]
        profile[lag] = rho_l.mean()
    return profile


def compute_autocorr_profile(
    returns: torch.Tensor,
    max_lag: int = 10,
    window_stride: int = 1,
    window_start_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Lag-0…max_lag normalized ACF of r², averaged over stocks.

    For each lag ``l`` and stock ``s``:

    .. math::

        \\rho_l^{(s)} = \\gamma_l^{(s)} / \\gamma_0^{(s)}

    where :math:`\\gamma_l^{(s)} = \\mathrm{Cov}(r_{b,t,s}^2,\\,
    r_{b,t+l,s}^2)` uses a single global mean and
    :math:`\\gamma_0^{(s)} = \\mathrm{Var}(r_{b,t,s}^2)`.  This
    ensures :math:`\\rho_0 = 1` and :math:`|\\rho_l| \\le 1`.

    When ``window_stride > 1`` the **phased** estimator is used: windows
    are grouped by ``start_index % window_stride`` into non-overlapping
    phases, autocorrelation is computed per phase, and the results are
    averaged.  This uses all data while keeping each phase overlap-free.

    Args:
        returns: ``[B, T, N]`` standardised log-returns.
        max_lag: Maximum lag to compute (inclusive).
        window_stride: Phase stride.  Set to ``T`` when batches come
            from a stride-1 sliding window.  Default ``1`` (no phasing;
            all windows pooled directly).
        window_start_indices: ``[B]`` integer tensor of window start
            positions in the original time series.  When provided,
            phasing groups by ``index % window_stride`` (robust to
            arbitrary batch ordering).  When ``None``, falls back to
            positional slicing ``returns[p::window_stride]`` (assumes
            chronological B order).

    Returns:
        ``[max_lag + 1]`` tensor of autocorrelations at lags 0…max_lag.
    """
    B, T, N = returns.shape
    r2 = returns ** 2

    if window_stride <= 1:
        return _autocorr_single_phase(r2, max_lag, N)

    # Phased averaging: iterate over all phase offsets
    accum = torch.zeros(max_lag + 1)
    valid_phases = 0

    if window_start_indices is not None:
        phases = window_start_indices % window_stride  # [B]
        unique_phases = phases.unique()
        for p in unique_phases:
            mask = phases == p
            r2_phase = r2[mask]
            if r2_phase.shape[0] == 0:
                continue
            accum += _autocorr_single_phase(r2_phase, max_lag, N)
            valid_phases += 1
    else:
        n_phases = min(window_stride, B)
        for phase in range(n_phases):
            r2_phase = r2[phase::window_stride]
            if r2_phase.shape[0] == 0:
                continue
            accum += _autocorr_single_phase(r2_phase, max_lag, N)
            valid_phases += 1

    return accum / max(valid_phases, 1)


def compute_autocorr_profile_per_trajectory(
    returns: torch.Tensor,
    max_lag: int = 10,
) -> torch.Tensor:
    """Per-trajectory normalized ACF of r², averaged across trajectories.

    Option C complement to :func:`compute_autocorr_profile`.  Computes
    the normalized ACF of r² **within each trajectory** (batch element)
    independently using a per-trajectory global mean and lag-0
    autocovariance, then averages over the B dimension.

    With short sequences (small T) individual estimates are noisy, but the
    mean over many trajectories converges.

    Args:
        returns: ``[B, T, N]`` standardised log-returns.
        max_lag: Maximum lag to compute (inclusive).

    Returns:
        ``[max_lag + 1]`` tensor of mean autocorrelations at lags 0…max_lag.
    """
    B, T, N = returns.shape
    r2 = returns ** 2  # [B, T, N]

    # Per-trajectory global mean and variance (across T)
    mu = r2.mean(dim=1, keepdim=True)        # [B, 1, N]
    gamma_0 = ((r2 - mu) ** 2).mean(dim=1)   # [B, N]

    profile = torch.zeros(max_lag + 1)
    profile[0] = 1.0  # ρ(0) = γ(0)/γ(0) = 1
    for lag in range(1, max_lag + 1):
        # Keep batch dim separate: [B, T-lag, N]
        r2_t = r2[:, :T - lag, :]
        r2_tl = r2[:, lag:, :]

        # Per-trajectory, per-stock normalized ACF
        gamma_l = ((r2_t - mu) * (r2_tl - mu)).mean(dim=1)  # [B, N]
        rho_l = gamma_l / (gamma_0 + 1e-8)                   # [B, N]
        # Average over trajectories and stocks
        profile[lag] = rho_l.mean()

    return profile


def _autocorr_per_stock_single_phase(
    r2: torch.Tensor, max_lag: int, T: int, N: int, normalize: bool = True,
) -> torch.Tensor:
    """Per-stock ACF helper for one set of (already-selected) windows.

    When ``normalize`` (default) returns the normalized ACF ``ρ_l =
    γ_l / γ_0`` (row 0 ≡ 1).  When ``normalize=False`` returns the raw
    autocovariance ``γ_l`` (row 0 = the lag-0 autocovariance = variance).

    Returns ``[max_lag + 1, N]`` with row *l* = (normalized) ACF at lag *l*.
    """
    r2_flat = r2.reshape(-1, N)
    mu = r2_flat.mean(0)                      # [N] global mean
    gamma_0 = ((r2_flat - mu) ** 2).mean(0)   # [N] lag-0 autocovariance

    profile = torch.zeros(max_lag + 1, N)
    profile[0] = 1.0 if normalize else gamma_0  # ρ(0)=1, or γ(0)=variance
    for lag in range(1, max_lag + 1):
        r2_t = r2[:, :T - lag, :].reshape(-1, N)
        r2_tl = r2[:, lag:, :].reshape(-1, N)

        gamma_l = ((r2_t - mu) * (r2_tl - mu)).mean(0)  # [N]
        profile[lag] = gamma_l / (gamma_0 + 1e-8) if normalize else gamma_l
    return profile


def compute_autocorr_profile_per_stock(
    returns: torch.Tensor,
    max_lag: int = 10,
    window_stride: int = 1,
    window_start_indices: Optional[torch.Tensor] = None,
    squared: bool = True,
    normalize: bool = True,
) -> torch.Tensor:
    """Per-stock lag-0…max_lag normalized ACF of r² (or r when ``squared=False``).

    When ``normalize=False`` the raw autocovariance ``γ_l`` is returned
    instead of the normalized ACF ``ρ_l = γ_l / γ_0`` (row 0 then holds the
    per-stock variance rather than 1).  Phased averaging is unaffected
    (autocovariances average linearly across phases).

    Identical to :func:`compute_autocorr_profile` but retains the
    per-stock dimension instead of averaging over stocks.

    When ``window_stride > 1`` the same **phased** estimator as
    :func:`compute_autocorr_profile` is used: windows are grouped by
    ``start_index % window_stride`` (or positionally if
    ``window_start_indices`` is ``None``), per-phase per-stock
    Normalized ACF values are computed, and the results are averaged over phases.

    Args:
        returns: ``[B, T, N]`` standardised log-returns.
        max_lag: Maximum lag to compute (inclusive).
        window_stride: Phase stride.  Set to ``T`` for stride-1 sliding
            window datasets.  Default ``1`` (no phasing).
        window_start_indices: ``[B]`` integer tensor of window start
            positions for order-robust phase assignment.  ``None`` falls
            back to positional slicing.
        squared: When ``True`` (default) compute the normalized ACF of
            ``r²`` (volatility clustering).  When ``False`` compute the
            normalized ACF of raw returns ``r`` (linear predictability).

    Returns:
        ``[max_lag + 1, N]`` tensor — row *l* holds the normalized ACF at lag *l*
        for every individual stock.  Row 0 is identically 1.
    """
    B, T, N = returns.shape
    r2 = returns ** 2 if squared else returns

    if window_stride <= 1:
        return _autocorr_per_stock_single_phase(r2, max_lag, T, N, normalize=normalize)

    accum = torch.zeros(max_lag + 1, N)
    valid_phases = 0

    if window_start_indices is not None:
        phases = window_start_indices % window_stride
        for p in phases.unique():
            mask = phases == p
            r2_phase = r2[mask]
            if r2_phase.shape[0] == 0:
                continue
            T_phase = r2_phase.shape[1]
            accum += _autocorr_per_stock_single_phase(
                r2_phase, max_lag, T_phase, N, normalize=normalize,
            )
            valid_phases += 1
    else:
        n_phases = min(window_stride, B)
        for phase in range(n_phases):
            r2_phase = r2[phase::window_stride]
            if r2_phase.shape[0] == 0:
                continue
            T_phase = r2_phase.shape[1]
            accum += _autocorr_per_stock_single_phase(
                r2_phase, max_lag, T_phase, N, normalize=normalize,
            )
            valid_phases += 1

    return accum / max(valid_phases, 1)  # [max_lag, N]


def _corr_matrix_single_phase(returns: torch.Tensor) -> torch.Tensor:
    """Normalised correlation matrix for one set of (already-selected) windows."""
    B, T, N = returns.shape
    r = returns.reshape(-1, N).float()
    r = r - r.mean(0)
    r = r / (r.std(0) + 1e-8)
    return (r.T @ r) / r.shape[0]  # [N, N]


def compute_top_eigenvalues(
    returns: torch.Tensor,
    k: int = 5,
    window_stride: int = 1,
    window_start_indices: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Top-k eigenvalues of the empirical return correlation matrix.

    When ``window_stride > 1`` the **phased** estimator is used:
    windows are grouped by ``start_index % window_stride`` into non-
    overlapping phases, a correlation matrix is computed per phase,
    and the phase matrices are averaged before eigendecomposition.

    When ``window_stride == 1`` (default), all windows are pooled
    directly into a single ``[B*T, N]`` design matrix.

    Args:
        returns: ``[B, T, N]`` standardised log-returns.
        k: Number of top eigenvalues to return.
        window_stride: Phase stride.  Set to ``T`` when batches come
            from a stride-1 sliding window.  Default ``1`` (no phasing;
            all windows pooled directly).
        window_start_indices: ``[B]`` integer tensor of window start
            positions in the original time series.  When provided,
            phasing groups by ``index % window_stride`` (robust to
            arbitrary batch ordering).  When ``None``, falls back to
            positional slicing ``returns[p::window_stride]`` (assumes
            chronological B order).

    Returns:
        Eigenvalues in descending order, shape ``[k]``.
    """
    B, T, N = returns.shape

    if window_stride <= 1:
        C = _corr_matrix_single_phase(returns)
    else:
        C = torch.zeros(N, N)
        valid_phases = 0

        if window_start_indices is not None:
            phases = window_start_indices % window_stride  # [B]
            unique_phases = phases.unique()
            for p in unique_phases:
                mask = phases == p
                r_phase = returns[mask]
                if r_phase.shape[0] == 0:
                    continue
                C += _corr_matrix_single_phase(r_phase)
                valid_phases += 1
        else:
            n_phases = min(window_stride, B)
            for phase in range(n_phases):
                r_phase = returns[phase::window_stride]
                if r_phase.shape[0] == 0:
                    continue
                C += _corr_matrix_single_phase(r_phase)
                valid_phases += 1

        C = C / max(valid_phases, 1)

    vals = torch.linalg.eigvalsh(C)
    return vals.flip(0)[:k]


def compute_top_eigenvalues_per_trajectory(
    returns: torch.Tensor, k: int = 5,
) -> torch.Tensor:
    """Per-trajectory top-k eigenvalues, averaged across trajectories.

    Option C complement to :func:`compute_top_eigenvalues`.  Computes the
    correlation-matrix eigenvalues **within each trajectory** (T
    observations of N stocks) independently, then averages the top-k
    spectrum over the B dimension.

    Because T ≪ N, most per-trajectory eigenvalues are zero; only at
    most ``min(T, N)`` are non-zero.  The average over many trajectories
    gives a meaningful (if noisy) estimate of the spectrum the model
    actually generates.

    Args:
        returns: ``[B, T, N]`` standardised log-returns.
        k: Number of top eigenvalues to return.

    Returns:
        Eigenvalues in descending order, shape ``[k]``.
    """
    B, T, N = returns.shape
    k = min(k, T, N)
    accum = torch.zeros(k)

    for b in range(B):
        r = returns[b].float()  # [T, N]
        r = r - r.mean(0)
        std = r.std(0)
        r = r / (std + 1e-8)
        C = (r.T @ r) / T  # [N, N]
        vals = torch.linalg.eigvalsh(C)
        accum += vals.flip(0)[:k]

    return accum / B


def compute_structural_metrics(
    real_returns: torch.Tensor,
    gen_returns: torch.Tensor,
    max_lag: int = 2,
    n_eigenvalues: int = 5,
    window_stride: int = 1,
    window_start_indices: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Compute cross-stock and temporal structure diagnostics.

    Returns a flat dict with keys:

    - ``autocorr_lag{l}_real``, ``autocorr_lag{l}_gen``,
      ``autocorr_lag{l}_diff`` for each lag *l* in 1…max_lag.
    - ``eigenvalue_{i}_real``, ``eigenvalue_{i}_gen``,
      ``eigenvalue_{i}_ratio`` for each rank *i* in 1…n_eigenvalues.

    When ``window_stride > 1`` (Phased Option A), the above keys are
    computed by averaging correlation / autocorrelation estimates across
    all ``window_stride`` phase offsets, each using non-overlapping
    windows.  This uses all available data while keeping each phase
    estimate overlap-free.

    Additionally emits per-trajectory (Option C) metrics under the prefix
    ``ptraj_``:

    - ``ptraj_autocorr_lag{l}_real``, ``ptraj_autocorr_lag{l}_gen``,
      ``ptraj_autocorr_lag{l}_diff``
    - ``ptraj_eigenvalue_{i}_real``, ``ptraj_eigenvalue_{i}_gen``,
      ``ptraj_eigenvalue_{i}_ratio``

    Args:
        real_returns: ``[B, T, N]`` ground-truth standardised log-returns.
        gen_returns:  ``[B, T, N]`` generated standardised log-returns.
        max_lag: Maximum lag for autocorrelation of r².
        n_eigenvalues: Number of top eigenvalues.
        window_stride: Sub-sample stride for non-overlapping windows.
            Typically set to ``T`` for stride-1 sliding-window datasets.
            Default ``1`` (no sub-sampling).
        window_start_indices: ``[B]`` integer tensor of window start
            positions.  Used by the phased estimator to group windows
            correctly regardless of batch ordering.  ``None`` falls
            back to positional slicing (chronological B assumed).
    """
    metrics: Dict[str, float] = {}

    # ==================================================================
    # Option A — pooled with non-overlapping window deduplication
    # ==================================================================

    # --- Autocorrelation of r² (normalized ACF) ---
    ac_real = compute_autocorr_profile(real_returns, max_lag,
                                       window_stride=window_stride,
                                       window_start_indices=window_start_indices)
    ac_gen = compute_autocorr_profile(gen_returns, max_lag,
                                      window_stride=window_stride,
                                      window_start_indices=window_start_indices)
    for lag in range(max_lag + 1):
        metrics[f"autocorr_lag{lag}_real"] = ac_real[lag].item()
        metrics[f"autocorr_lag{lag}_gen"] = ac_gen[lag].item()
        metrics[f"autocorr_lag{lag}_diff"] = abs(
            ac_real[lag].item() - ac_gen[lag].item()
        )

    # --- Eigenvalue spectrum ---
    ev_real = compute_top_eigenvalues(real_returns, n_eigenvalues,
                                      window_stride=window_stride,
                                      window_start_indices=window_start_indices)
    ev_gen = compute_top_eigenvalues(gen_returns, n_eigenvalues,
                                     window_stride=window_stride,
                                     window_start_indices=window_start_indices)
    for i in range(n_eigenvalues):
        rank = i + 1
        r_val = ev_real[i].item()
        g_val = ev_gen[i].item()
        metrics[f"eigenvalue_{rank}_real"] = r_val
        metrics[f"eigenvalue_{rank}_gen"] = g_val
        metrics[f"eigenvalue_{rank}_ratio"] = (
            g_val / r_val if abs(r_val) > 1e-8 else float("nan")
        )

    # ==================================================================
    # Option C — per-trajectory average (complementary diagnostic)
    # ==================================================================

    ac_real_pt = compute_autocorr_profile_per_trajectory(
        real_returns, max_lag)
    ac_gen_pt = compute_autocorr_profile_per_trajectory(
        gen_returns, max_lag)
    for lag in range(max_lag + 1):
        metrics[f"ptraj_autocorr_lag{lag}_real"] = ac_real_pt[lag].item()
        metrics[f"ptraj_autocorr_lag{lag}_gen"] = ac_gen_pt[lag].item()
        metrics[f"ptraj_autocorr_lag{lag}_diff"] = abs(
            ac_real_pt[lag].item() - ac_gen_pt[lag].item()
        )

    T = gen_returns.shape[1]
    k_pt = min(n_eigenvalues, T, gen_returns.shape[2])
    ev_real_pt = compute_top_eigenvalues_per_trajectory(
        real_returns, k_pt)
    ev_gen_pt = compute_top_eigenvalues_per_trajectory(
        gen_returns, k_pt)
    for i in range(k_pt):
        rank = i + 1
        r_val = ev_real_pt[i].item()
        g_val = ev_gen_pt[i].item()
        metrics[f"ptraj_eigenvalue_{rank}_real"] = r_val
        metrics[f"ptraj_eigenvalue_{rank}_gen"] = g_val
        metrics[f"ptraj_eigenvalue_{rank}_ratio"] = (
            g_val / r_val if abs(r_val) > 1e-8 else float("nan")
        )

    return metrics


def compute_cumulative_return_std(
    returns: torch.Tensor,
    per_stock: bool = False,
) -> torch.Tensor:
    """Empirical std of cumulative log returns at each prediction horizon.

    Computes :math:`R_{b,t,s} = \\sum_{\\tau=1}^{t} r_{b,\\tau,s}` and
    returns the cross-batch std per stock (or averaged over stocks):

    .. math::

        \\hat{\\sigma}_{\\text{cum}}(t, s)
        = \\mathrm{std}_{b}\\left(R_{b,t,s}\\right)

    Under i.i.d. returns this grows as :math:`\\sigma_s\\sqrt{t}`.

    Args:
        returns:   ``[B, T, N]`` or ``[B, T, N, 1]``.
        per_stock: If ``True`` return ``[T, N]`` (one value per horizon
                   and stock).  If ``False`` (default) return ``[T]``
                   by averaging over stocks.

    Returns:
        ``[T, N]`` if *per_stock* else ``[T]``.
    """
    if returns.dim() == 4:
        returns = returns.squeeze(-1)  # [B, T, N]
    cum = returns.cumsum(dim=1)        # [B, T, N]
    std_tn = cum.std(dim=0)            # [T, N]  (std over batch)
    if per_stock:
        return std_tn                  # [T, N]
    return std_tn.mean(dim=1)          # [T]     (mean over stocks)


# ======================================================================
# Visualisation functions
# ======================================================================


def plot_structural_metrics(
    real_returns: torch.Tensor,
    gen_returns: torch.Tensor,
    save_path: str,
    max_lag: int = 2,
    n_eigenvalues: int = 5,
    dpi: int = 150,
    model_name: str = "GRW",
    window_stride: int = 1,
    window_start_indices: Optional[torch.Tensor] = None,
) -> None:
    """Save a three-panel structural diagnostics figure.

    Left panel: normalized ACF of raw returns r at lags 0…max_lag
    (linear predictability / momentum).
    Middle panel: normalized ACF of squared returns r² at lags 0…max_lag
    (volatility clustering / ARCH effect).
    Right panel: top-k eigenvalue spectrum of the return correlation matrix.

    Both panels show the **phased pooled** (Option A) values as solid
    bars and the **per-trajectory** (Option C) values as hatched bars
    for comparison.

    Args:
        real_returns: ``[B, T, N]`` ground-truth standardised log-returns.
        gen_returns:  ``[B, T, N]`` generated standardised log-returns.
        save_path: File path to save the figure.
        max_lag: Maximum lag for the autocorrelation profile.
        n_eigenvalues: Number of top eigenvalues to display.
        dpi: Figure DPI.
        model_name: Display name for the generative model (e.g.
            ``"GRW"`` or ``"Diffusion"``).  Used in print messages.
        window_stride: Sub-sample stride for non-overlapping windows
            (passed to pooled compute functions).
        window_start_indices: ``[B]`` integer tensor of window start
            positions (passed to pooled compute functions).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Option A — pooled deduped
    profile_real = compute_autocorr_profile(
        real_returns, max_lag=max_lag, window_stride=window_stride,
        window_start_indices=window_start_indices,
    ).numpy()
    profile_gen = compute_autocorr_profile(
        gen_returns, max_lag=max_lag, window_stride=window_stride,
        window_start_indices=window_start_indices,
    ).numpy()

    k = min(n_eigenvalues, real_returns.shape[-1])
    eigs_real = compute_top_eigenvalues(
        real_returns, k=k, window_stride=window_stride,
        window_start_indices=window_start_indices,
    ).numpy()
    eigs_gen = compute_top_eigenvalues(
        gen_returns, k=k, window_stride=window_stride,
        window_start_indices=window_start_indices,
    ).numpy()

    # Option C — per-trajectory
    profile_real_pt = compute_autocorr_profile_per_trajectory(
        real_returns, max_lag=max_lag,
    ).numpy()
    profile_gen_pt = compute_autocorr_profile_per_trajectory(
        gen_returns, max_lag=max_lag,
    ).numpy()

    T = gen_returns.shape[1]
    k_pt = min(k, T, gen_returns.shape[2])
    eigs_real_pt = compute_top_eigenvalues_per_trajectory(
        real_returns, k=k_pt,
    ).numpy()
    eigs_gen_pt = compute_top_eigenvalues_per_trajectory(
        gen_returns, k=k_pt,
    ).numpy()

    fig, (ax_r, ax_r2, ax_ev) = plt.subplots(1, 3, figsize=(20, 5))

    lags = np.arange(0, max_lag + 1)
    stride_note = f" (stride={window_stride})" if window_stride > 1 else ""

    def _draw_bars(ax, lags, per_stock, color, label, x_offset=0.0, bar_w=0.35):
        median = np.median(per_stock, axis=1)
        p25, p75 = np.percentile(per_stock, [25, 75], axis=1)
        p10, p90 = np.percentile(per_stock, [10, 90], axis=1)
        x = lags + x_offset
        ax.bar(x, median, width=bar_w, color=color, alpha=0.75, label=label)
        ax.errorbar(x, median,
                    yerr=[median - p25, p75 - median],
                    fmt="none", ecolor=color, elinewidth=2.5,
                    capsize=4, capthick=2.0, alpha=0.9)
        ax.errorbar(x, median,
                    yerr=[median - p10, p90 - median],
                    fmt="none", ecolor=color, elinewidth=1.0,
                    capsize=3, capthick=1.0, alpha=0.6)

    # Left — normalized ACF of raw returns r (linear predictability)
    per_stock_real_r = compute_autocorr_profile_per_stock(
        real_returns, max_lag=max_lag, window_stride=window_stride,
        window_start_indices=window_start_indices, squared=False,
    ).numpy()  # [max_lag+1, N]
    per_stock_gen_r = compute_autocorr_profile_per_stock(
        gen_returns, max_lag=max_lag, window_stride=window_stride,
        window_start_indices=window_start_indices, squared=False,
    ).numpy()  # [max_lag+1, N]
    model_label = format_baseline_display_name(model_name)
    _draw_bars(ax_r, lags, per_stock_real_r, "steelblue", "Real", x_offset=-0.20)
    _draw_bars(ax_r, lags, per_stock_gen_r, "darkorange", model_label, x_offset=+0.20)
    ax_r.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_r.set_xlabel("Lag  $l$", fontsize=11)
    ax_r.set_ylabel(r"Normalized ACF of $r$  $\rho_l = \gamma_l / \gamma_0$", fontsize=11)
    ax_r.set_title(
        r"Normalized ACF of $r$ — median w/ IQR & 10\%–90\% error bars" + stride_note,
        fontsize=12,
    )
    ax_r.set_xticks(lags)
    ax_r.legend(fontsize=7, ncol=2)

    # Middle — normalized ACF of squared returns r² (volatility clustering)
    per_stock_real = compute_autocorr_profile_per_stock(
        real_returns, max_lag=max_lag, window_stride=window_stride,
        window_start_indices=window_start_indices,
    ).numpy()  # [max_lag+1, N]
    per_stock_gen = compute_autocorr_profile_per_stock(
        gen_returns, max_lag=max_lag, window_stride=window_stride,
        window_start_indices=window_start_indices,
    ).numpy()  # [max_lag+1, N]
    _draw_bars(ax_r2, lags, per_stock_real, "steelblue", "Real", x_offset=-0.20)
    _draw_bars(ax_r2, lags, per_stock_gen, "darkorange", model_label, x_offset=+0.20)
    ax_r2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_r2.set_xlabel("Lag  $l$", fontsize=11)
    ax_r2.set_ylabel(r"Normalized ACF of $r^2$  $\rho_l = \gamma_l / \gamma_0$", fontsize=11)
    ax_r2.set_title(
        r"Normalized ACF of $r^2$ — median w/ IQR & 10\%–90\% error bars" + stride_note,
        fontsize=12,
    )
    ax_r2.set_xticks(lags)
    ax_r2.legend(fontsize=7, ncol=2)

    # Right — eigenvalue spectrum
    idx = np.arange(1, k + 1)
    idx_pt = np.arange(1, k_pt + 1)
    w = 0.20
    ax_ev.bar(idx - 1.5 * w, eigs_real, width=w,
              color="steelblue", alpha=0.85, label=r"Real (pooled)")
    ax_ev.bar(idx + 0.5 * w, eigs_gen, width=w,
              color="darkorange", alpha=0.85, label=f"{model_label} (pooled)")
    ax_ev.bar(idx_pt - 0.5 * w, eigs_real_pt, width=w,
              color="steelblue", alpha=0.45, hatch="//",
              label=r"Real (per-traj)")
    ax_ev.bar(idx_pt + 1.5 * w, eigs_gen_pt, width=w,
              color="darkorange", alpha=0.45, hatch="//",
              label=f"{model_label} (per-traj)")
    ax_ev.axhline(1.0, color="black", linewidth=0.8, linestyle="--",
                  label="i.i.d. reference ($\\lambda = 1$)")
    ax_ev.set_xlabel("Eigenvalue rank", fontsize=11)
    ax_ev.set_ylabel(r"Eigenvalue  $\lambda_k$", fontsize=11)
    ax_ev.set_title(fr"Top-{k} eigenvalues of $\hat{{C}}$" + stride_note,
                    fontsize=12)
    ax_ev.set_xticks(idx)
    ax_ev.legend(fontsize=8, ncol=2)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{model_name}] Structural metrics figure saved to: {save_path}")


def plot_autocorr_per_stock(
    real_returns: torch.Tensor,
    gen_returns: torch.Tensor,
    save_path: str,
    max_lag: int = 2,
    dpi: int = 150,
    model_name: str = "GRW",
    window_stride: int = 1,
    window_start_indices: Optional[torch.Tensor] = None,
) -> None:
    """Save a per-stock autocorrelation figure.

    Two panels (real | generated), each showing one curve per lag.
    Stocks are sorted in descending order of their lag-1 real
    autocorrelation.

    When ``window_stride > 1`` the phased estimator is used (same logic
    as :func:`compute_autocorr_profile`) so that overlap-induced dilution
    is avoided for generated samples.

    Args:
        real_returns: ``[B, T, N]`` ground-truth standardised log-returns.
        gen_returns:  ``[B, T, N]`` generated standardised log-returns.
        save_path: File path to save the figure.
        max_lag: Maximum lag for the autocorrelation profile.
        dpi: Figure DPI.
        model_name: Display name for the generative model.
        window_stride: Phase stride passed to
            :func:`compute_autocorr_profile_per_stock`.
        window_start_indices: ``[B]`` integer tensor of window start
            positions for order-robust phase assignment.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np

    # [max_lag, N]
    prof_real = compute_autocorr_profile_per_stock(
        real_returns, max_lag=max_lag,
        window_stride=window_stride,
        window_start_indices=window_start_indices,
    ).numpy()
    prof_gen = compute_autocorr_profile_per_stock(
        gen_returns, max_lag=max_lag,
        window_stride=window_stride,
        window_start_indices=window_start_indices,
    ).numpy()

    N = prof_real.shape[1]

    # Sort stocks by lag-1 real normalized ACF, descending
    sort_idx = np.argsort(prof_real[1])[::-1]
    prof_real = prof_real[:, sort_idx]   # [max_lag+1, N] reordered
    prof_gen = prof_gen[:, sort_idx]

    stock_x = np.arange(N)
    colors = cm.plasma(np.linspace(0.1, 0.85, max_lag + 1))

    fig, (ax_real, ax_gen) = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

    for lag_idx in range(max_lag + 1):
        lag = lag_idx
        c = colors[lag_idx]
        ax_real.plot(stock_x, prof_real[lag_idx], color=c,
                     linewidth=1.2, label=f"lag {lag}")
        ax_gen.plot(stock_x, prof_gen[lag_idx], color=c,
                    linewidth=1.2, label=f"lag {lag}")

    for ax, title in [
        (ax_real, r"Real returns  $r^{\mathrm{real}}$"),
        (ax_gen, f"{model_name}-generated  " + r"$r^{\mathrm{gen}}$"),
    ]:
        ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
        ax.set_xlabel("Stock (sorted by lag-1 normalized ACF)", fontsize=11)
        ax.set_ylabel(r"Normalized ACF of $r^2$  $\rho_l^{(s)}$", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_xlim(0, N - 1)
        ax.legend(fontsize=9, ncol=min(max_lag + 1, 5))

    fig.suptitle(
        fr"Per-stock $r^2$ normalized ACF profiles  (N={N})",
        fontsize=13,
        y=1.02,
    )
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{model_name}] Per-stock autocorr figure saved to: {save_path}")


def plot_autocorr_violin(
    real_returns: torch.Tensor,
    gen_returns: torch.Tensor,
    save_path: str,
    max_lag: int = 2,
    dpi: int = 150,
    model_name: str = "GRW",
    window_stride: int = 1,
    window_start_indices: Optional[torch.Tensor] = None,
) -> None:
    """Violin plot of per-stock normalized ACF distributions at each lag.

    For each lag *l*, shows two side-by-side violins (real vs baseline)
    depicting the distribution of :math:`\\rho_l^{(s)}` across all
    *N* stocks.  This reveals heterogeneity (skew, outliers, bimodality)
    that the stock-averaged profile hides.

    Args:
        real_returns: ``[B, T, N]`` ground-truth standardised log-returns.
        gen_returns:  ``[B, T, N]`` generated standardised log-returns.
        save_path: File path to save the figure.
        max_lag: Maximum lag (inclusive).  Lag 0 is omitted since
            :math:`\\rho_0 = 1` for every stock by definition.
        dpi: Figure DPI.
        model_name: Display name for the generative model.
        window_stride: Phase stride for the phased estimator.
        window_start_indices: ``[B]`` window start positions.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    prof_real = compute_autocorr_profile_per_stock(
        real_returns, max_lag=max_lag,
        window_stride=window_stride,
        window_start_indices=window_start_indices,
    ).numpy()  # [max_lag+1, N]
    prof_gen = compute_autocorr_profile_per_stock(
        gen_returns, max_lag=max_lag,
        window_stride=window_stride,
        window_start_indices=window_start_indices,
    ).numpy()  # [max_lag+1, N]

    # Skip lag 0 (always 1.0 for every stock)
    lags = np.arange(1, max_lag + 1)
    n_lags = len(lags)

    fig, ax = plt.subplots(1, 1, figsize=(max(6, 2.5 * n_lags), 5))

    positions_real = lags - 0.18
    positions_gen = lags + 0.18
    width = 0.32

    vp_real = ax.violinplot(
        [prof_real[l] for l in lags],
        positions=positions_real, widths=width,
        showmeans=True, showmedians=True, showextrema=True,
        bw_method=0.5,
    )
    vp_gen = ax.violinplot(
        [prof_gen[l] for l in lags],
        positions=positions_gen, widths=width,
        showmeans=True, showmedians=True, showextrema=True,
        bw_method=0.5,
    )

    # Style real violins
    for body in vp_real["bodies"]:
        body.set_facecolor("steelblue")
        body.set_edgecolor("steelblue")
        body.set_alpha(0.5)
    vp_real["cmedians"].set_color("steelblue")
    vp_real["cmedians"].set_linewidth(1.5)
    vp_real["cmeans"].set_color("steelblue")
    vp_real["cmeans"].set_linewidth(1.5)
    vp_real["cmeans"].set_linestyle("--")
    vp_real["cbars"].set_color("steelblue")
    vp_real["cbars"].set_linewidth(0.8)
    vp_real["cmaxes"].set_color("steelblue")
    vp_real["cmaxes"].set_linewidth(1.2)
    vp_real["cmins"].set_color("steelblue")
    vp_real["cmins"].set_linewidth(1.2)

    # Style gen violins
    for body in vp_gen["bodies"]:
        body.set_facecolor("darkorange")
        body.set_edgecolor("darkorange")
        body.set_alpha(0.5)
    vp_gen["cmedians"].set_color("darkorange")
    vp_gen["cmedians"].set_linewidth(1.5)
    vp_gen["cmeans"].set_color("darkorange")
    vp_gen["cmeans"].set_linewidth(1.5)
    vp_gen["cmeans"].set_linestyle("--")
    vp_gen["cbars"].set_color("darkorange")
    vp_gen["cbars"].set_linewidth(0.8)
    vp_gen["cmaxes"].set_color("darkorange")
    vp_gen["cmaxes"].set_linewidth(1.2)
    vp_gen["cmins"].set_color("darkorange")
    vp_gen["cmins"].set_linewidth(1.2)

    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_xticks(lags)
    ax.set_xticklabels([str(l) for l in lags])
    ax.set_xlabel("Lag  $l$", fontsize=11)
    ax.set_ylabel(r"Per-stock normalized ACF  $\rho_l^{(s)} = \gamma_l^{(s)} / \gamma_0^{(s)}$",
                  fontsize=11)
    stride_note = f" (stride={window_stride})" if window_stride > 1 else ""
    N = prof_real.shape[1]
    ax.set_title(
        fr"Per-stock $r^2$ normalized ACF distribution  (N={N})"
        + stride_note,
        fontsize=12,
    )
    # Manual legend entries
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    model_label = format_baseline_display_name(model_name)
    ax.legend(
        handles=[
            Patch(facecolor="steelblue", alpha=0.5, label="Real"),
            Patch(facecolor="darkorange", alpha=0.5, label=model_label),
            Line2D([0], [0], color="gray", linewidth=1.5, linestyle="-",
                   label="median"),
            Line2D([0], [0], color="gray", linewidth=1.5, linestyle="--",
                   label="mean"),
        ],
        fontsize=9, loc="upper right",
    )

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{model_name}] Normalized ACF violin figure saved to: {save_path}")


def plot_autocorr_heatmap(
    gen_returns: torch.Tensor,
    save_path: str,
    max_lag: int = 2,
    dpi: int = 150,
    model_name: str = "GRW",
    window_stride: int = 1,
    window_start_indices: Optional[torch.Tensor] = None,
    real_returns: Optional[torch.Tensor] = None,
) -> None:
    """Heatmap of per-stock normalized ACF values (stocks × lags).

    Each cell *(s, l)* shows :math:`\\rho_l^{(s)}`, the normalized ACF of
    :math:`r^2` for stock *s* at lag *l*.  Stocks are sorted by their
    lag-1 real normalized ACF (descending) when *real_returns* is provided,
    otherwise by lag-1 baseline normalized ACF.

    Args:
        gen_returns: ``[B, T, N]`` generated standardised log-returns.
        save_path: File path to save the figure.
        max_lag: Maximum lag (inclusive).  Lag 0 is omitted (always 1).
        dpi: Figure DPI.
        model_name: Display name for the generative model.
        window_stride: Phase stride for the phased estimator.
        window_start_indices: ``[B]`` window start positions.
        real_returns: Optional ``[B, T, N]`` ground-truth returns.  When
            provided, stocks are sorted by lag-1 real normalized ACF for
            consistent ordering across baselines.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np

    prof_gen = compute_autocorr_profile_per_stock(
        gen_returns, max_lag=max_lag,
        window_stride=window_stride,
        window_start_indices=window_start_indices,
    ).numpy()  # [max_lag+1, N]

    # Sort stocks by lag-1 normalized ACF (use real if available)
    if real_returns is not None:
        prof_real = compute_autocorr_profile_per_stock(
            real_returns, max_lag=max_lag,
            window_stride=window_stride,
            window_start_indices=window_start_indices,
        ).numpy()
        sort_idx = np.argsort(prof_real[1])[::-1]  # descending lag-1 real
    else:
        sort_idx = np.argsort(prof_gen[1])[::-1]  # descending lag-1 gen

    # Skip lag 0, sort stocks
    heatmap_data = prof_gen[1:, sort_idx]  # [max_lag, N]
    N = heatmap_data.shape[1]
    n_lags = heatmap_data.shape[0]

    fig_w = max(8, N * 0.04 + 2)
    fig_h = max(3, n_lags * 0.6 + 1.5)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

    # Diverging colormap centred at 0
    vmax = max(abs(heatmap_data.min()), abs(heatmap_data.max()), 0.1)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(
        heatmap_data, aspect="auto", cmap="RdBu_r", norm=norm,
        interpolation="none",
    )

    ax.set_xlabel(
        "Stock (sorted by lag-1 "
        + ("real" if real_returns is not None else model_name)
        + " normalized ACF)",
        fontsize=11,
    )
    ax.set_ylabel("Lag  $l$", fontsize=11)
    ax.set_yticks(np.arange(n_lags))
    ax.set_yticklabels([str(l) for l in range(1, max_lag + 1)])
    # Reduce x-tick density for large N
    if N > 50:
        step = max(N // 10, 1)
        ax.set_xticks(np.arange(0, N, step))
    else:
        ax.set_xticks(np.arange(N))

    stride_note = f" (stride={window_stride})" if window_stride > 1 else ""
    ax.set_title(
        fr"{model_name} per-stock $r^2$ normalized ACF heatmap  (N={N})"
        + stride_note,
        fontsize=12,
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label(
        r"$\rho_l^{(s)}$", fontsize=11,
    )

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{model_name}] Normalized ACF heatmap saved to: {save_path}")


def plot_cumulative_return_std(
    std_real: Any,
    std_gen: Any,
    save_path: str,
    theoretical: Any = None,
    std_real_tn: Any = None,
    std_gen_tn: Any = None,
    dpi: int = 150,
    model_name: str = "GRW",
    theoretical_label: Optional[str] = None,
) -> None:
    """Plot empirical std of cumulative log returns vs prediction horizon.

    **Left panel** shows horizon-vs-std lines averaged over stocks:

    * **Real data** — empirical std of :math:`R_{b,t}` across batch.
    * **Model samples** — same quantity from generated samples.
    * **Theory** (optional) — e.g. :math:`\\bar{\\sigma}\\sqrt{t}`,
      the i.i.d. Gaussian closed-form prediction.

    **Right panel** (shown when *std_real_tn* and *std_gen_tn* are
    provided) plots the per-stock relative difference.

    Args:
        std_real:    ``[T]`` numpy array — real empirical std (avg over stocks).
        std_gen:     ``[T]`` numpy array — model sample empirical std.
        save_path:   File path for the saved figure.
        theoretical: Optional ``[T]`` numpy array — theoretical reference curve.
            Pass ``None`` to omit.
        std_real_tn: Optional ``[T, N]`` numpy array — per-stock real std.
        std_gen_tn:  Optional ``[T, N]`` numpy array — per-stock model std.
        dpi:         Figure DPI.
        model_name: Display name for the generative model.
        theoretical_label: Legend label for the theoretical curve.
            Defaults to ``"GRW theory  σ̄√t"`` when *None*.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import numpy as np

    if theoretical_label is None:
        theoretical_label = r"GRW theory  $\bar{\sigma}\sqrt{t}$"

    model_label = format_baseline_display_name(model_name)
    T = len(std_real)
    t_vals = np.arange(1, T + 1)

    two_panels = (std_real_tn is not None) and (std_gen_tn is not None)
    fig, axes = plt.subplots(
        1, 2 if two_panels else 1,
        figsize=(14 if two_panels else 7, 5),
    )
    ax = axes[0] if two_panels else axes

    # ---- Left panel: avg std vs horizon ----------------------------------
    ax.plot(t_vals, std_real, "o-", color="steelblue", linewidth=2,
            label=r"Real data  $\hat{\sigma}_{\mathrm{cum}}(t)$")
    ax.plot(t_vals, std_gen, "s--", color="darkorange", linewidth=2,
            label=f"{model_label} samples  "
                  + r"$\hat{\sigma}_{\mathrm{cum}}(t)$")
    if theoretical is not None:
        ax.plot(t_vals, theoretical, "k:", linewidth=1.8,
                label=theoretical_label)

    ax.set_xlabel("Prediction horizon $t$", fontsize=12)
    ax.set_ylabel(
        r"Std of cumulative log return  "
        r"$\hat{\sigma}_{\mathrm{cum}}(t)$",
        fontsize=12,
    )
    ax.set_title("Cumulative log-return std vs horizon", fontsize=13)
    ax.set_xticks(t_vals)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # ---- Right panel: per-stock relative difference ----------------------
    if two_panels:
        ax2 = axes[1]
        std_real_tn = np.asarray(std_real_tn)  # [T, N]
        std_gen_tn = np.asarray(std_gen_tn)    # [T, N]

        # Sort stocks by single-step real std (t=1) descending
        order = np.argsort(std_real_tn[0])[::-1]  # [N]
        std_real_sorted = std_real_tn[:, order]    # [T, N]
        std_gen_sorted = std_gen_tn[:, order]      # [T, N]

        # Relative difference: (model - real) / real per stock and horizon
        rel_diff = (std_gen_sorted - std_real_sorted) / (std_real_sorted + 1e-12)

        N = std_real_tn.shape[1]
        stock_idx = np.arange(N)
        colors = cm.plasma(np.linspace(0.05, 0.90, T))

        for ti in range(T):
            ax2.plot(
                stock_idx, rel_diff[ti],
                linewidth=1.2, alpha=0.75, color=colors[ti],
                label=f"$t={ti+1}$",
            )

        ax2.axhline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.5)
        ax2.set_xlabel(
            "Stock index (sorted by single-step real std $\\downarrow$)",
            fontsize=11,
        )
        mn_esc = model_name.replace("_", r"\_")
        ax2.set_ylabel(
            r"$(\hat{\sigma}_{\mathrm{" + mn_esc + r"}}"
            r" - \hat{\sigma}_{\mathrm{real}})"
            r"\;/\;\hat{\sigma}_{\mathrm{real}}$",
            fontsize=11,
        )
        ax2.set_title(
            f"Per-stock relative std difference ({model_name} − real) / real",
            fontsize=12,
        )
        ax2.legend(
            fontsize=7, ncol=max(1, T // 5),
            loc="upper right",
        )
        ax2.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{model_name}] Cumulative-return std figure saved to: {save_path}")


# ======================================================================
# Comparative (multi-baseline) visualisation functions
# ======================================================================

# Colour palette shared by both comparative plots.
# Uses matplotlib's tab10 palette which is perceptually distinct and
# well-calibrated in saturation — avoids the neon/garish look of many
# named CSS colours.  Extend if more than 6 baselines are compared.
_BASELINE_COLORS = [
    "tab:orange",   # first baseline (e.g. GRW)
    "tab:green",    # second baseline (e.g. Diffusion)
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
]


def _resolve_baseline_color_map(
    baseline_names: List[str],
    baseline_colors: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Resolve baseline colors with optional name->color overrides."""
    configured = {
        str(name).strip().lower(): str(color)
        for name, color in (baseline_colors or {}).items()
        if str(name).strip()
    }
    resolved: Dict[str, str] = {}

    for name in baseline_names:
        key = str(name).strip().lower()
        if key in configured:
            resolved[name] = configured[key]

    used = {c.lower() for c in resolved.values()}
    fallback_cycle = [c for c in _BASELINE_COLORS if c.lower() not in used]
    if not fallback_cycle:
        fallback_cycle = list(_BASELINE_COLORS)

    next_idx = 0
    for name in baseline_names:
        if name in resolved:
            continue
        resolved[name] = fallback_cycle[next_idx % len(fallback_cycle)]
        next_idx += 1

    return resolved


def plot_structural_metrics_comparison(
    real_returns: torch.Tensor,
    gen_returns_dict: Dict[str, torch.Tensor],
    save_path: str,
    max_lag: int = 2,
    n_eigenvalues: int = 5,
    dpi: int = 150,
    window_stride: int = 1,
    window_start_indices: Optional[torch.Tensor] = None,
    window_start_indices_gen: Optional[Dict[str, Optional[torch.Tensor]]] = None,
    baseline_colors: Optional[Dict[str, str]] = None,
) -> None:
    """Three-panel structural diagnostics with *all* baselines overlaid.

    Left panel: normalized ACF of r (linear predictability) at lags 0…max_lag.
    Middle panel: normalized ACF of r² (volatility clustering) at lags 0…max_lag.
    Right panel: top-k eigenvalue spectrum of the return correlation matrix.

    Only the **pooled / phased (Option A)** estimator is shown (no
    per-trajectory bars) to keep the figure readable.

    Args:
        real_returns: ``[B, T, N]`` ground-truth standardised log-returns.
        gen_returns_dict: ``{baseline_name: [B, T, N]}`` generated returns
            for each baseline.
        save_path: File path to save the figure.
        max_lag: Maximum lag for autocorrelation.
        n_eigenvalues: Top-k eigenvalues to show.
        dpi: Figure DPI.
        window_stride: Phase stride for the real-data phased estimator.
        window_start_indices: ``[B_real]`` window start positions for
            *real_returns*.  Applied to real only.
        window_start_indices_gen: Optional dict mapping each baseline name
            to its own ``[B_gen]`` window start positions.  When ``None``
            (or a key is missing) the corresponding gen tensor is processed
            with ``window_start_indices=None`` (positional fallback).
            Use this when different baselines have different batch sizes.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    baseline_names = list(gen_returns_dict.keys())
    baseline_display_names = {
        name: format_baseline_display_name(name) for name in baseline_names
    }
    n_baselines = len(baseline_names)

    # --- Compute real-data profiles (once) --------------------------------
    profile_real = compute_autocorr_profile(
        real_returns, max_lag=max_lag, window_stride=window_stride,
        window_start_indices=window_start_indices,
    ).numpy()

    k = min(n_eigenvalues, real_returns.shape[-1])
    eigs_real = compute_top_eigenvalues(
        real_returns, k=k, window_stride=window_stride,
        window_start_indices=window_start_indices,
    ).numpy()

    # --- Compute per-baseline profiles ------------------------------------
    profiles_gen: Dict[str, Any] = {}
    eigs_gen: Dict[str, Any] = {}
    for name, gen_ret in gen_returns_dict.items():
        # Each baseline may have a different number of windows; use its own
        # window_start_indices so the phased estimator mask sizes match.
        wsi_gen = (
            window_start_indices_gen.get(name)
            if window_start_indices_gen is not None
            else None
        )
        profiles_gen[name] = compute_autocorr_profile(
            gen_ret, max_lag=max_lag, window_stride=window_stride,
            window_start_indices=wsi_gen,
        ).numpy()
        eigs_gen[name] = compute_top_eigenvalues(
            gen_ret, k=k, window_stride=window_stride,
            window_start_indices=wsi_gen,
        ).numpy()

    fig, (ax_r, ax_r2, ax_ev) = plt.subplots(1, 3, figsize=(20, 5))

    lags = np.arange(0, max_lag + 1)
    stride_note = f" (stride={window_stride})" if window_stride > 1 else ""
    n_groups = 1 + n_baselines
    total_bar_w = 0.70
    bar_w_ac = total_bar_w / n_groups
    offsets_ac = np.linspace(
        -total_bar_w / 2 + bar_w_ac / 2,
        total_bar_w / 2 - bar_w_ac / 2,
        n_groups,
    )
    gen_colors = _resolve_baseline_color_map(
        baseline_names,
        baseline_colors=baseline_colors,
    )

    def _draw_bars(ax, lags, per_stock, color, label, x_offset=0.0, bar_w=0.35):
        median = np.median(per_stock, axis=1)
        p25, p75 = np.percentile(per_stock, [25, 75], axis=1)
        p10, p90 = np.percentile(per_stock, [10, 90], axis=1)
        x = lags + x_offset
        ax.bar(x, median, width=bar_w, color=color, alpha=0.75, label=label)
        ax.errorbar(x, median,
                    yerr=[median - p25, p75 - median],
                    fmt="none", ecolor=color, elinewidth=2.5,
                    capsize=4, capthick=2.0, alpha=0.9)
        ax.errorbar(x, median,
                    yerr=[median - p10, p90 - median],
                    fmt="none", ecolor=color, elinewidth=1.0,
                    capsize=3, capthick=1.0, alpha=0.6)

    # --- Left: normalized ACF of r (linear predictability) ---------------
    per_stock_real_r = compute_autocorr_profile_per_stock(
        real_returns, max_lag=max_lag, window_stride=window_stride,
        window_start_indices=window_start_indices, squared=False,
    ).numpy()  # [max_lag+1, N]
    _draw_bars(ax_r, lags, per_stock_real_r, "steelblue", "Real",
               x_offset=offsets_ac[0], bar_w=bar_w_ac)
    for bi, (name, gen_ret) in enumerate(gen_returns_dict.items()):
        wsi_gen = (
            window_start_indices_gen.get(name)
            if window_start_indices_gen is not None
            else None
        )
        per_stock_gen_r = compute_autocorr_profile_per_stock(
            gen_ret, max_lag=max_lag, window_stride=window_stride,
            window_start_indices=wsi_gen, squared=False,
        ).numpy()
        _draw_bars(
            ax_r,
            lags,
            per_stock_gen_r,
            gen_colors[name],
            baseline_display_names[name],
            x_offset=offsets_ac[1 + bi],
            bar_w=bar_w_ac,
        )
    ax_r.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_r.set_xlabel("Lag  $l$", fontsize=11)
    ax_r.set_ylabel(r"Normalized ACF of $r$  $\rho_l = \gamma_l / \gamma_0$", fontsize=11)
    ax_r.set_title(
        r"Normalized ACF of $r$ — median w/ IQR & 10\%–90\% error bars" + stride_note,
        fontsize=12,
    )
    ax_r.set_xticks(lags)
    ax_r.legend(fontsize=8, ncol=max(1, 1 + n_baselines))

    # --- Middle: normalized ACF of r² (volatility clustering) ------------
    per_stock_real = compute_autocorr_profile_per_stock(
        real_returns, max_lag=max_lag, window_stride=window_stride,
        window_start_indices=window_start_indices,
    ).numpy()  # [max_lag+1, N]
    _draw_bars(ax_r2, lags, per_stock_real, "steelblue", "Real",
               x_offset=offsets_ac[0], bar_w=bar_w_ac)
    for bi, (name, gen_ret) in enumerate(gen_returns_dict.items()):
        wsi_gen = (
            window_start_indices_gen.get(name)
            if window_start_indices_gen is not None
            else None
        )
        per_stock_gen = compute_autocorr_profile_per_stock(
            gen_ret, max_lag=max_lag, window_stride=window_stride,
            window_start_indices=wsi_gen,
        ).numpy()
        _draw_bars(
            ax_r2,
            lags,
            per_stock_gen,
            gen_colors[name],
            baseline_display_names[name],
            x_offset=offsets_ac[1 + bi],
            bar_w=bar_w_ac,
        )
    ax_r2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_r2.set_xlabel("Lag  $l$", fontsize=11)
    ax_r2.set_ylabel(r"Normalized ACF of $r^2$  $\rho_l = \gamma_l / \gamma_0$",
                     fontsize=11)
    ax_r2.set_title(
        r"Normalized ACF of $r^2$ — median w/ IQR & 10\%–90\% error bars" + stride_note,
        fontsize=12,
    )
    ax_r2.set_xticks(lags)
    ax_r2.legend(fontsize=8, ncol=max(1, 1 + n_baselines))

    # --- Right: eigenvalue spectrum ---------------------------------------
    n_groups = 1 + n_baselines
    total_w = 0.70
    bar_w = total_w / n_groups
    offsets = np.linspace(
        -total_w / 2 + bar_w / 2,
        total_w / 2 - bar_w / 2,
        n_groups,
    )
    idx = np.arange(1, k + 1)
    ax_ev.bar(idx + offsets[0], eigs_real, width=bar_w,
              color="steelblue", alpha=0.85, label="Real")
    for bi, name in enumerate(baseline_names):
        color = gen_colors[name]
        ax_ev.bar(idx + offsets[1 + bi], eigs_gen[name], width=bar_w,
                  color=color, alpha=0.85, label=baseline_display_names[name])

    ax_ev.axhline(1.0, color="black", linewidth=0.8, linestyle="--",
                  label=r"i.i.d. reference ($\lambda = 1$)")
    ax_ev.set_xlabel("Eigenvalue rank", fontsize=11)
    ax_ev.set_ylabel(r"Eigenvalue  $\lambda_k$", fontsize=11)
    ax_ev.set_title(
        fr"Top-{k} eigenvalues of $\hat{{C}}$" + stride_note, fontsize=12,
    )
    ax_ev.set_xticks(idx)
    ax_ev.legend(fontsize=8, ncol=max(1, n_groups))

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Comparison] Structural metrics figure saved to: {save_path}")


def plot_structural_comparison_paper(
    real_returns: torch.Tensor,
    gen_returns_dict: Dict[str, torch.Tensor],
    save_path: str,
    *,
    max_lag: int = 2,
    n_eigenvalues: int = 5,
    dpi: int = 300,
    window_stride: int = 1,
    window_start_indices: Optional[torch.Tensor] = None,
    window_start_indices_gen: Optional[Dict[str, Optional[torch.Tensor]]] = None,
    baseline_colors: Optional[Dict[str, str]] = None,
    normalize: bool = True,
    include_lag0: bool = False,
    show_iqr: bool = True,
    twin_axis: bool = False,
    fig_width: float = 7.16,
    plot_style: Optional[dict] = None,
) -> None:
    """Paper-ready 2-panel structural comparison (SP500 Fig 2).

    **Panel 1 — combined autocorrelation.**  The ACF of ``r`` and of ``r²``
    overlaid on one lag axis, distinguished by shade (``r`` solid, ``r²``
    hatched).  By default (``normalize=True``) the normalized ACF
    ``ρ_l = γ_l/γ_0`` is shown on a single shared y-axis; ``normalize=False``
    shows the raw autocovariance ``γ_l`` (and ``twin_axis=True`` then gives r
    and r² separate left/right axes).

    **Panel 2 — top-k eigenvalue spectrum** of the return correlation matrix
    (Real + each baseline).

    Self-styling: ``plot_style`` (the full ``plot_style`` mapping) supplies the
    global look via ``matplotlib.rc_context`` and per-figure aesthetics via
    ``plot_style.plots.structural_comparison`` — so the figure is identical
    whether drawn in a live run or a sidecar replot.  Bars are per-stock
    medians; ``show_iqr`` adds 25–75% error bars.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors
    import numpy as np
    from graph_signal_diffusion.evaluation.plot_style_apply import resolve_paper_style

    rc, pstyle, colors = resolve_paper_style(
        plot_style, "structural_comparison", baseline_colors,
    )
    # Thicker hatch strokes so the r² stripes survive PDF rasterization/viewers.
    rc.setdefault("hatch.linewidth", 0.7)
    hatch_edge = pstyle.get("hatch_edgecolor", "0.25")  # contrasting stripe color
    _dpi = int(pstyle.get("dpi", dpi))
    _figsize = pstyle.get("figure_size") or [fig_width, max(2.6, fig_width * 0.46)]
    _f = pstyle.get("fonts") or {}
    fs_title = float(_f.get("title", 10)); fs_label = float(_f.get("label", 9))
    fs_legend = float(_f.get("legend", 6.5)); fs_tick = float(_f.get("tick", 8))

    baseline_names = list(gen_returns_dict.keys())
    display = {name: format_baseline_display_name(name) for name in baseline_names}
    gen_colors = _resolve_baseline_color_map(
        baseline_names, baseline_colors=(colors or None),
    )
    real_color = pstyle.get("real_color") or colors.get("real") or "steelblue"

    # Entities in draw order: Real first, then each baseline.
    entities = [("__real__", "Real", real_color)] + [
        (name, display[name], gen_colors[name]) for name in baseline_names
    ]
    n_ent = len(entities)

    def _wsi_gen(name):
        return (
            window_start_indices_gen.get(name)
            if window_start_indices_gen is not None else None
        )

    def _per_stock(returns, wsi, squared):
        return compute_autocorr_profile_per_stock(
            returns, max_lag=max_lag, window_stride=window_stride,
            window_start_indices=wsi, squared=squared, normalize=normalize,
        ).numpy()  # [max_lag+1, N]

    # --- Per-stock ACF profiles of r and r² -----------------------------------
    prof_r = {"__real__": _per_stock(real_returns, window_start_indices, False)}
    prof_r2 = {"__real__": _per_stock(real_returns, window_start_indices, True)}
    for name, gen_ret in gen_returns_dict.items():
        prof_r[name] = _per_stock(gen_ret, _wsi_gen(name), False)
        prof_r2[name] = _per_stock(gen_ret, _wsi_gen(name), True)

    # --- Eigenvalues ----------------------------------------------------------
    k = min(n_eigenvalues, real_returns.shape[-1])
    eigs_real = compute_top_eigenvalues(
        real_returns, k=k, window_stride=window_stride,
        window_start_indices=window_start_indices,
    ).numpy()
    eigs_gen = {
        name: compute_top_eigenvalues(
            gen_ret, k=k, window_stride=window_stride,
            window_start_indices=_wsi_gen(name),
        ).numpy()
        for name, gen_ret in gen_returns_dict.items()
    }

    # --- Lag selection --------------------------------------------------------
    all_lags = np.arange(0, max_lag + 1)
    lag_slice = slice(0, max_lag + 1) if include_lag0 else slice(1, max_lag + 1)
    lags_plot = all_lags[lag_slice]
    x = np.arange(len(lags_plot))

    # Self-styling: the global rcParams (background/fonts) are applied locally so
    # the output matches regardless of whether the caller applied them globally.
    with matplotlib.rc_context(rc):
        fig, (ax_ac, ax_ev) = plt.subplots(1, 2, figsize=_figsize)
        ax_r2_axis = ax_ac.twinx() if twin_axis else ax_ac

        # --- Panel 1: combined autocorrelation (r solid / r² hatched) ---------
        if len(lags_plot) > 0:
            total_w = 0.8
            slot = total_w / (2 * n_ent)
            offsets = np.linspace(
                -total_w / 2 + slot / 2, total_w / 2 - slot / 2, 2 * n_ent,
            )

            def _bars(ax, xpos, rows, color, hatch, alpha):
                median = np.median(rows, axis=1)
                # Bake the fill alpha into the facecolor (RGBA) and keep the
                # edge at full opacity, so a hatched bar shows crisp, contrasting
                # stripes instead of same-color-as-fill (invisible) ones.
                face = mcolors.to_rgba(color, alpha)
                edge = hatch_edge if hatch else face
                ax.bar(xpos, median, width=slot, facecolor=face,
                       hatch=hatch, edgecolor=edge, linewidth=0.0)
                if show_iqr:
                    p25, p75 = np.percentile(rows, [25, 75], axis=1)
                    ax.errorbar(xpos, median, yerr=[median - p25, p75 - median],
                                fmt="none", ecolor=color, elinewidth=1.2,
                                capsize=2.5, capthick=1.0, alpha=0.85)

            for ei, (key, _label, color) in enumerate(entities):
                _bars(ax_ac, x + offsets[2 * ei], prof_r[key][lag_slice],
                      color, None, 0.85)
                _bars(ax_r2_axis, x + offsets[2 * ei + 1], prof_r2[key][lag_slice],
                      color, "////", 0.45)

        ax_ac.axhline(0, color="black", linewidth=0.6, linestyle="--")
        ax_ac.set_xlabel(r"Lag  $l$", fontsize=fs_label)
        ylab_r = (r"ACF of $r$  $\rho_l$" if normalize
                  else r"Autocovariance of $r$  $\gamma_l$")
        ylab_r2 = (r"ACF of $r^2$  $\rho_l$" if normalize
                   else r"Autocovariance of $r^2$  $\gamma_l$")
        if twin_axis:
            ax_ac.set_ylabel(ylab_r, fontsize=fs_label)
            ax_r2_axis.set_ylabel(ylab_r2, fontsize=fs_label)
        else:
            # Shared single y-axis: r and r² are comparable when normalized.
            ax_ac.set_ylabel(
                r"Normalized ACF  $\rho_l$" if normalize
                else r"Autocovariance  $\gamma_l$",
                fontsize=fs_label,
            )
        ax_ac.set_xticks(x)
        ax_ac.set_xticklabels([str(int(l)) for l in lags_plot])
        ax_ac.tick_params(labelsize=fs_tick)
        if twin_axis:
            ax_r2_axis.tick_params(labelsize=fs_tick)
        ax_ac.set_title(r"Autocorrelation of $r$ and $r^2$", fontsize=fs_title)

        # Legend: entity colors + r/r² shade key.
        ent_handles = [
            mpatches.Patch(facecolor=c, edgecolor=c, label=lab)
            for (_k, lab, c) in entities
        ]
        style_handles = [
            mpatches.Patch(facecolor=mcolors.to_rgba("gray", 0.85), label=r"$r$"),
            mpatches.Patch(facecolor=mcolors.to_rgba("gray", 0.45),
                           edgecolor=hatch_edge, hatch="////", label=r"$r^2$"),
        ]
        ax_ac.legend(handles=ent_handles + style_handles, fontsize=fs_legend,
                     ncol=2, framealpha=0.85, loc="best")

        # --- Panel 2: eigenvalue spectrum -------------------------------------
        n_groups = 1 + len(baseline_names)
        total_ev = 0.7
        bw = total_ev / n_groups
        ev_off = np.linspace(-total_ev / 2 + bw / 2, total_ev / 2 - bw / 2, n_groups)
        idx = np.arange(1, k + 1)
        ax_ev.bar(idx + ev_off[0], eigs_real, width=bw, color=real_color,
                  alpha=0.85, label="Real")
        for bi, name in enumerate(baseline_names):
            ax_ev.bar(idx + ev_off[1 + bi], eigs_gen[name], width=bw,
                      color=gen_colors[name], alpha=0.85, label=display[name])
        ax_ev.axhline(1.0, color="black", linewidth=0.6, linestyle="--",
                      label=r"i.i.d. ($\lambda = 1$)")
        ax_ev.set_xlabel("Eigenvalue rank", fontsize=fs_label)
        ax_ev.set_ylabel(r"Eigenvalue  $\lambda_k$", fontsize=fs_label)
        ax_ev.set_title(fr"Top-{k} eigenvalues of $\hat{{C}}$", fontsize=fs_title)
        ax_ev.set_xticks(idx)
        ax_ev.tick_params(labelsize=fs_tick)
        ax_ev.legend(fontsize=fs_legend, ncol=2, framealpha=0.85)

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        try:
            fig.tight_layout()
        except Exception:
            pass
        fig.savefig(save_path, dpi=_dpi, bbox_inches="tight")
        plt.close(fig)
    print(f"  [Paper] Fig 2 structural comparison saved to: {save_path}")


def plot_cumulative_return_std_comparison(
    std_real: Any,
    std_gen_dict: Dict[str, Any],
    save_path: str,
    theoretical: Any = None,
    dpi: int = 150,
    theoretical_label: Optional[str] = None,
    std_real_per_stock: Any = None,
    std_gen_per_stock_dict: Optional[Dict[str, Any]] = None,
    baseline_colors: Optional[Dict[str, str]] = None,
) -> None:
    """Cumulative-return std vs prediction horizon with all baselines.

    A single panel with one line per baseline (plus real data and an
    optional theoretical reference).  When per-stock data is supplied
    the figure also shows individual stock trajectories as very thin
    transparent lines and a shaded 25th–75th percentile band, giving
    a visual sense of cross-sectional dispersion.

    Args:
        std_real: ``[T]`` numpy array — real empirical std (avg over stocks).
        std_gen_dict: ``{baseline_name: [T] numpy array}`` — per-baseline
            model sample empirical std.
        save_path: File path for the saved figure.
        theoretical: Optional ``[T]`` numpy array — theoretical reference.
        dpi: Figure DPI.
        theoretical_label: Legend label for the theoretical curve.
        std_real_per_stock: Optional ``[T, N]`` numpy array — per-stock
            empirical std of real returns.  When provided, each stock's
            trajectory is rendered as a thin line and a 25th–75th
            percentile band is shaded.
        std_gen_per_stock_dict: Optional ``{name: [T, N] numpy array}`` —
            same as above for each baseline's generated returns.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if theoretical_label is None:
        theoretical_label = r"i.i.d. theory  $\bar{\sigma}\sqrt{t}$"

    T = len(std_real)
    t_vals = np.arange(1, T + 1)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    def _draw_per_stock(
        ax, t_vals, per_stock, color, alpha_lines=0.04, alpha_band=0.15,
    ):
        """Draw individual-stock thin lines + IQR band."""
        # per_stock: [T, N]
        N = per_stock.shape[1]
        for s in range(N):
            ax.plot(t_vals, per_stock[:, s], color=color,
                    linewidth=0.35, alpha=alpha_lines)
        q25 = np.percentile(per_stock, 25, axis=1)  # [T]
        q75 = np.percentile(per_stock, 75, axis=1)  # [T]
        ax.fill_between(t_vals, q25, q75, color=color, alpha=alpha_band)

    # ── Real data ───────────────────────────────────────────────────
    if std_real_per_stock is not None:
        _draw_per_stock(ax, t_vals, std_real_per_stock, color="steelblue")
    ax.plot(t_vals, std_real, "o-", color="steelblue", linewidth=2,
            label=r"Real data  $\hat{\sigma}_{\mathrm{cum}}(t)$")

    # ── Baselines ───────────────────────────────────────────────────
    baseline_names = list(std_gen_dict.keys())
    color_map = _resolve_baseline_color_map(
        baseline_names,
        baseline_colors=baseline_colors,
    )
    markers = ["s", "^", "D", "v", "P", "X"]
    for bi, name in enumerate(baseline_names):
        display_name = format_baseline_display_name(name)
        color = color_map[name]
        marker = markers[bi % len(markers)]
        if std_gen_per_stock_dict is not None:
            per_s = std_gen_per_stock_dict.get(name)
            if per_s is not None:
                _draw_per_stock(ax, t_vals, per_s, color=color)
        ax.plot(t_vals, std_gen_dict[name], f"{marker}--", color=color,
                linewidth=2,
                label=f"{display_name}  " + r"$\hat{\sigma}_{\mathrm{cum}}(t)$")

    if theoretical is not None:
        ax.plot(t_vals, theoretical, "k:", linewidth=1.8,
                label=theoretical_label)

    ax.set_xlabel("Prediction horizon $t$", fontsize=12)
    ax.set_ylabel(
        r"Std of cumulative log return  "
        r"$\hat{\sigma}_{\mathrm{cum}}(t)$",
        fontsize=12,
    )
    _subtitle = (
        "\n(thin lines = individual stocks, band = 25th–75th pct)"
        if std_real_per_stock is not None else ""
    )
    ax.set_title("Cumulative log-return std vs horizon" + _subtitle, fontsize=12)
    ax.set_xticks(t_vals)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Comparison] Cumulative-return std figure saved to: {save_path}")


def plot_nll_histograms_comparison(
    nll_real: Any,
    nll_gen_dict: Dict[str, Any],
    save_path: str,
    bins: int = 60,
    dpi: int = 150,
    log_x: bool = True,
    scoring_model_name: Optional[str] = None,
    baseline_colors: Optional[Dict[str, str]] = None,
) -> None:
    """Two-panel NLL histogram + CDF figure with all baselines overlaid.

    Left panel: overlaid density histograms of per-trajectory NLL for
    real returns and each baseline's generated returns, all scored
    under the same model (typically GRW).

    Right panel: empirical CDFs with per-baseline
    :math:`|F_{\\text{real}} - F_{\\text{baseline}}|` areas shaded
    and :math:`W_1` values in the legend.

    Args:
        nll_real: 1-D numpy array — NLL of real returns.
        nll_gen_dict: ``{baseline_name: 1-D numpy array}`` — NLL of
            each baseline's generated returns.
        save_path: File path for the saved figure.
        bins: Number of histogram bins.
        dpi: Figure DPI.
        log_x: If ``True`` (default), use a logarithmic x-axis on both
            panels.  NLL values are always positive and typically
            right-skewed, so log scale reveals the bulk and tail
            simultaneously.  Set to ``False`` for a linear x-axis.
        scoring_model_name: Name of the model used to compute NLL scores,
            shown in plot titles.  Defaults to ``None`` (generic title).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import wasserstein_distance as _w1

    baseline_names = list(nll_gen_dict.keys())
    color_map = _resolve_baseline_color_map(
        baseline_names,
        baseline_colors=baseline_colors,
    )

    # Shared x-grid spanning all distributions
    all_vals = np.concatenate(
        [nll_real] + [nll_gen_dict[n] for n in baseline_names]
    )
    x_min, x_max = float(all_vals.min()), float(all_vals.max())
    if log_x and x_min > 0:
        x_grid = np.logspace(np.log10(x_min), np.log10(x_max), 2000)
        hist_bins = np.logspace(np.log10(x_min), np.log10(x_max), bins + 1)
    else:
        x_grid = np.linspace(x_min, x_max, 2000)
        hist_bins = bins

    def _ecdf(s, x):
        return np.searchsorted(np.sort(s), x, side="right") / len(s)

    cdf_real = _ecdf(nll_real, x_grid)

    fig, (ax_hist, ax_cdf) = plt.subplots(
        1, 2, figsize=(14, 5),
        gridspec_kw={"width_ratios": [1.2, 1]},
    )

    # ---- Left: density histograms ------------------------------------
    ax_hist.hist(
        nll_real, bins=hist_bins, density=True, alpha=0.50, color="steelblue",
        label=(
            r"$\mathrm{NLL}(r^{\mathrm{real}})$"
            f"  n={len(nll_real)}"
            f"  \u03bc={nll_real.mean():.1f}"
            f"  \u03c3={nll_real.std():.1f}"
        ),
    )
    for bi, name in enumerate(baseline_names):
        display_name = format_baseline_display_name(name)
        color = color_map[name]
        arr = nll_gen_dict[name]
        ax_hist.hist(
            arr, bins=hist_bins, density=True, alpha=0.40, color=color,
            label=(
                r"$\mathrm{NLL}(r^{\mathrm{" + display_name + r"}})$"
                f"  n={len(arr)}"
                f"  \u03bc={arr.mean():.1f}"
                f"  \u03c3={arr.std():.1f}"
            ),
        )
    if log_x and x_min > 0:
        ax_hist.set_xscale("log")
    if scoring_model_name and scoring_model_name.lower() == "grw":
        xlabel_nll = (
            r"Per-trajectory NLL  $= \sum_{s,t} -\log\,\mathcal{N}"
            r"(r_{s,t}\mid\tilde{\mu}_s,\tilde{\sigma}_s^2)$"
        )
    else:
        xlabel_nll = "Per-trajectory score (lower = more likely under scorer)"
    ax_hist.set_xlabel(xlabel_nll, fontsize=11)
    ax_hist.set_ylabel("Density", fontsize=11)
    ax_hist.tick_params(axis="both", which="both", labelsize=8)
    _scoring_label = f"scored under {scoring_model_name}" if scoring_model_name else "shared scoring model"
    ax_hist.set_title(
        f"NLL Histograms  (real vs generated, {_scoring_label})",
        fontsize=12,
    )
    ax_hist.legend(fontsize=8)

    # ---- Right: empirical CDFs + W1 areas ----------------------------
    ax_cdf.plot(
        x_grid, cdf_real, color="steelblue", linewidth=1.8,
        label=r"$F_{\mathrm{real}}$",
    )
    fill_colors = ["mediumpurple", "lightcoral", "khaki", "palegreen",
                   "lightsalmon", "lightskyblue"]
    for bi, name in enumerate(baseline_names):
        display_name = format_baseline_display_name(name)
        color = color_map[name]
        fc = fill_colors[bi % len(fill_colors)]
        arr = nll_gen_dict[name]
        cdf_gen = _ecdf(arr, x_grid)
        w1 = float(_w1(nll_real, arr))
        ax_cdf.plot(
            x_grid, cdf_gen, color=color, linewidth=1.8,
            label=f"$F_{{\\mathrm{{{display_name}}}}}$",
        )
        ax_cdf.fill_between(
            x_grid, cdf_real, cdf_gen, alpha=0.20, color=fc,
            label=(
                f"$|F_{{\\mathrm{{real}}}} - F_{{\\mathrm{{{display_name}}}}}|$"
                f"  W$_1$={w1:.2f}"
            ),
        )
    if log_x and x_min > 0:
        ax_cdf.set_xscale("log")
    ax_cdf.set_xlabel(r"Per-trajectory NLL  $\ell$", fontsize=11)
    ax_cdf.set_ylabel("Empirical CDF", fontsize=11)
    ax_cdf.tick_params(axis="both", which="both", labelsize=8)
    ax_cdf.set_title("Empirical CDFs — Wasserstein-1 distances", fontsize=12)
    ax_cdf.legend(fontsize=8)
    ax_cdf.set_ylim(0, 1)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Comparison] NLL histogram figure saved to: {save_path}")


def _nll_use_log(x_min: float, x_max: float, log_x: bool,
                 min_ratio: float = 10.0) -> bool:
    """Decide whether to use a log x-axis for an NLL row.

    Only use log when requested, strictly positive, AND the distribution spans
    at least ``min_ratio`` (≈1 decade) — otherwise a log axis over a narrow
    range just crams overlapping scientific-notation minor ticks.
    """
    return bool(log_x and x_min > 0 and (x_max / x_min) >= min_ratio)


def _robust_density_ylim_top(
    hist_heights: Dict[str, Any], scorer_name: str, headroom: float = 1.25,
) -> Optional[float]:
    """Robust y-upper for an NLL density panel.

    A baseline scored under *itself* (its key equals ``scorer_name``) collapses
    to a tall, narrow peak that squashes every other distribution.  Return the
    tallest bin height among the *non-self* series (real + the other baselines)
    scaled by ``headroom`` so the spike is clipped and the rest stays legible.
    Returns ``None`` when there is no non-self series to clip against.
    """
    import numpy as np

    sname = (scorer_name or "").lower()
    ref = []
    for key, h in hist_heights.items():
        if key == "__real__" or str(key).lower() != sname:
            arr = np.asarray(h, dtype=float)
            if arr.size:
                ref.append(float(arr.max()))
    if not ref:
        return None
    top = max(ref) * float(headroom)
    return top if top > 0 else None


def annotate_grw_per_element_nats(
    scorer: Dict[str, Any], n_elements: Optional[int],
) -> Dict[str, Any]:
    """Tag the GRW NLL scorer with per-(stock, timestep) nats units for Fig 3.

    GRW's ``compute_nll`` *sums* the Gaussian NLL over all ``N`` stocks and ``T``
    timesteps of a trajectory, so dividing by ``n_elements = N·T`` gives the mean
    per-element NLL in nats — directly comparable to the i.i.d. Gaussian floor
    ½·ln(2πe) ≈ 1.42.  Returns a copy of ``scorer`` with ``x_divisor`` /
    ``x_label`` / ``reference_value`` / ``reference_label`` set, which
    :func:`plot_nll_comparison_paper` consumes (rescaling that row, relabelling
    its axes, and drawing the floor line).

    No-op for non-GRW scorers (e.g. the diffusion *MSE proxy*, which is **not** a
    per-element log-density and must not be divided by N·T) or when
    ``n_elements`` is missing/invalid.
    """
    import math

    if (not scorer or not n_elements or int(n_elements) <= 0
            or str(scorer.get("name", "")).lower() != "grw"):
        return scorer
    out = dict(scorer)
    out["x_divisor"] = float(int(n_elements))
    out["x_label"] = r"GRW NLL per (stock, step)  [nats]"
    out["reference_value"] = 0.5 * math.log(2.0 * math.pi * math.e)
    out["reference_label"] = r"i.i.d. Gaussian floor ($\approx$1.42)"
    return out


def annotate_diffusion_per_step_mse(
    scorer: Dict[str, Any], num_timesteps: Optional[int],
) -> Dict[str, Any]:
    """Express the diffusion (U-GNN) proxy row in per-element denoising-MSE units.

    ``compute_elbo_per_trajectory`` returns ``num_timesteps × mean_{T,N,F}‖·‖²``
    (the L_simple denoising objective scaled up by the diffusion-step count), so
    dividing by ``num_timesteps`` recovers the **per-element mean denoising MSE**
    — the same quantity (and scale) as the model's L2 training/val loss.  Returns
    a copy of ``scorer`` with ``x_divisor``/``x_label`` set.  No-op for non-
    diffusion scorers (e.g. GRW, a true summed log-density) or missing
    ``num_timesteps``.
    """
    if (not scorer or not num_timesteps or int(num_timesteps) <= 0
            or str(scorer.get("name", "")).lower() != "diffusion"):
        return scorer
    out = dict(scorer)
    out["x_divisor"] = float(int(num_timesteps))
    out["x_label"] = r"Denoising MSE per element  (ELBO)"
    return out


def plot_nll_comparison_paper(
    scorers: list,
    save_path: str,
    *,
    bins: int = 60,
    dpi: int = 300,
    log_x: bool = True,
    baseline_colors: Optional[Dict[str, str]] = None,
    fig_width: float = 7.16,
    row_height: float = 3.0,
    plot_style: Optional[dict] = None,
) -> None:
    """Paper-ready dual-scorer NLL comparison (SP500 Fig 3).

    Produces an ``n_scorers × 2`` grid: one **row per scoring model** (e.g. GRW,
    U-GNN), columns = {density histogram, empirical CDF}.  Within each row the
    real returns and every baseline's generated returns are overlaid, scored
    under that row's model, with per-baseline :math:`W_1` in the CDF legend.
    Each row uses its **own** x-range (NLL scales differ across scorers).  With
    both the GRW and U-GNN scorers present this is the 4-panel (2×2) figure; if
    only one scorer is available it degrades to a single 1×2 row.

    Self-styling: ``plot_style`` supplies the global look via
    ``matplotlib.rc_context`` and per-figure aesthetics via
    ``plot_style.plots.nll_comparison`` — identical output in a live run or a
    sidecar replot.

    Args:
        scorers: ordered list of dicts, each
            ``{"name": str, "nll_real": 1-D array,
               "nll_gen_dict": {baseline_name: 1-D array}}``.  Entries with an
            empty ``nll_gen_dict`` are skipped; an empty list is a no-op.
        save_path: File path for the saved figure.
        bins: Histogram bins.  dpi: figure DPI.
        log_x: Logarithmic x-axis (per row) when the row spans ≥ 1 decade.
        baseline_colors: Optional ``{name: color}`` overrides.
        fig_width: Figure width (inches).  row_height: per-row height (inches).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter
    import numpy as np
    from scipy.stats import wasserstein_distance as _w1
    from graph_signal_diffusion.evaluation.plot_style_apply import resolve_paper_style

    scorers = [s for s in scorers if s and s.get("nll_gen_dict")]
    n = len(scorers)
    if n == 0:
        return

    rc, pstyle, colors = resolve_paper_style(
        plot_style, "nll_comparison", baseline_colors,
    )
    _dpi = int(pstyle.get("dpi", dpi))
    _fig_width = float(pstyle.get("fig_width", fig_width))
    _row_height = float(pstyle.get("row_height", row_height))
    real_color = pstyle.get("real_color") or colors.get("real") or "steelblue"
    fill_colors = pstyle.get("fill_colors") or [
        "mediumpurple", "lightcoral", "khaki", "palegreen",
        "lightsalmon", "lightskyblue",
    ]
    _f = pstyle.get("fonts") or {}
    fs_title = float(_f.get("title", 9)); fs_label = float(_f.get("label", 8))
    fs_legend = float(_f.get("legend", 6)); fs_tick = float(_f.get("tick", 7))

    def _ecdf(s, x):
        return np.searchsorted(np.sort(s), x, side="right") / len(s)

    _width_ratios = list(pstyle.get("width_ratios", [1.2, 1]))
    with matplotlib.rc_context(rc):
        fig, axes = plt.subplots(
            n, 2, figsize=(_fig_width, max(2.6, n * _row_height)),
            squeeze=False, gridspec_kw={"width_ratios": _width_ratios},
        )

        for ri, sc in enumerate(scorers):
            name = sc["name"]
            # Optional per-row unit change: divide by x_divisor (e.g. GRW summed
            # NLL → per-(stock,step) nats) and relabel; W1 below is then in the
            # same units.  reference_value draws a floor line (e.g. 1.42 nats).
            divisor = float(sc.get("x_divisor", 1.0) or 1.0)
            nll_real = np.asarray(sc["nll_real"]).ravel() / divisor
            nll_gen_dict = {
                k: np.asarray(v).ravel() / divisor
                for k, v in sc["nll_gen_dict"].items()
            }
            baseline_names = list(nll_gen_dict.keys())
            color_map = _resolve_baseline_color_map(
                baseline_names, baseline_colors=(colors or None),
            )
            disp_scorer = format_baseline_display_name(name)
            # Score kind drives the histogram/CDF panel titles ("NLL" vs "ELBO");
            # the diffusion row scores with the per-element denoising MSE (ELBO
            # proxy), every other scorer (incl. GRW) with a true NLL.
            score_kind = "ELBO" if name.lower() == "diffusion" else "NLL"

            all_vals = np.concatenate(
                [nll_real] + [nll_gen_dict[k] for k in baseline_names]
            )
            x_min, x_max = float(all_vals.min()), float(all_vals.max())
            # Only log-x when the row spans ≥1 decade (else linear → clean ticks).
            use_log = _nll_use_log(x_min, x_max, log_x)
            if use_log:
                x_grid = np.logspace(np.log10(x_min), np.log10(x_max), 2000)
                hist_bins = np.logspace(np.log10(x_min), np.log10(x_max), bins + 1)
            else:
                x_grid = np.linspace(x_min, x_max, 2000)
                hist_bins = bins

            ax_hist = axes[ri, 0]
            ax_cdf = axes[ri, 1]

            # ---- Density histograms --------------------------------------
            hist_heights: Dict[str, Any] = {}
            n_r, _, _ = ax_hist.hist(
                nll_real, bins=hist_bins, density=True, alpha=0.50,
                color=real_color, label="Real",
            )
            hist_heights["__real__"] = n_r
            for bn in baseline_names:
                n_b, _, _ = ax_hist.hist(
                    nll_gen_dict[bn], bins=hist_bins, density=True,
                    alpha=0.40, color=color_map[bn],
                    label=format_baseline_display_name(bn),
                )
                hist_heights[bn] = n_b
            if use_log:
                ax_hist.set_xscale("log")
                ax_hist.xaxis.set_minor_formatter(NullFormatter())  # avoid collisions
            # Robust y-top: clip a self-scored spike so the others stay legible.
            if bool(pstyle.get("density_clip_self_scored", True)):
                _top = _robust_density_ylim_top(
                    hist_heights, name,
                    float(pstyle.get("density_ylim_headroom", 1.25)),
                )
                if _top is not None and _top < ax_hist.get_ylim()[1]:
                    ax_hist.set_ylim(top=_top)
            # Optional floor reference line (e.g. the i.i.d. Gaussian NLL floor).
            ref_v = sc.get("reference_value")
            ref_l = sc.get("reference_label")
            if ref_v is not None:
                ax_hist.axvline(float(ref_v), color="0.35", linestyle=":",
                                linewidth=1.1, zorder=5, label=ref_l)
            if name.lower() == "grw":
                default_xlabel = (
                    r"NLL  $\sum_{s,t}-\log\,\mathcal{N}"
                    r"(r_{s,t}\mid\tilde\mu_s,\tilde\sigma_s^2)$"
                )
            else:
                default_xlabel = "Per-trajectory score (lower = more likely)"
            ax_hist.set_xlabel(sc.get("x_label") or default_xlabel, fontsize=fs_label)
            ax_hist.set_ylabel("Density", fontsize=fs_label)
            ax_hist.tick_params(labelsize=fs_tick)
            ax_hist.set_title(f"{disp_scorer}-scored — {score_kind} histogram",
                              fontsize=fs_title)
            ax_hist.legend(fontsize=fs_legend)

            # ---- Empirical CDFs + W1 areas -------------------------------
            cdf_real = _ecdf(nll_real, x_grid)
            ax_cdf.plot(x_grid, cdf_real, color=real_color, linewidth=1.6,
                        label=r"$F_{\mathrm{real}}$")
            for bi, bn in enumerate(baseline_names):
                cdf_gen = _ecdf(nll_gen_dict[bn], x_grid)
                w1 = float(_w1(nll_real, nll_gen_dict[bn]))
                disp = format_baseline_display_name(bn)
                ax_cdf.plot(x_grid, cdf_gen, color=color_map[bn], linewidth=1.6,
                            label=fr"$F_{{\mathrm{{{disp}}}}}$")
                ax_cdf.fill_between(
                    x_grid, cdf_real, cdf_gen, alpha=0.20,
                    color=fill_colors[bi % len(fill_colors)],
                    label=fr"$W_1$={w1:.2f}",
                )
            if ref_v is not None:
                ax_cdf.axvline(float(ref_v), color="0.35", linestyle=":",
                               linewidth=1.1, zorder=5, label=ref_l)
            if use_log:
                ax_cdf.set_xscale("log")
                ax_cdf.xaxis.set_minor_formatter(NullFormatter())  # avoid collisions
            ax_cdf.set_xlabel(
                sc.get("x_label") or r"Per-trajectory NLL  $\ell$", fontsize=fs_label,
            )
            ax_cdf.set_ylabel("Empirical CDF", fontsize=fs_label)
            ax_cdf.tick_params(labelsize=fs_tick)
            ax_cdf.set_ylim(0, 1)
            ax_cdf.set_title(f"{disp_scorer}-scored — {score_kind} CDF",
                             fontsize=fs_title)
            ax_cdf.legend(fontsize=fs_legend)

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        try:
            fig.tight_layout()
        except Exception:
            pass
        fig.savefig(save_path, dpi=_dpi, bbox_inches="tight")
        plt.close(fig)
    print(f"  [Paper] Fig 3 NLL comparison saved to: {save_path}")


def plot_cumulative_nll_histograms_comparison(
    cum_nll_real: Any,
    cum_nll_gen_dict: Dict[str, Any],
    save_path: str,
    bins: int = 60,
    dpi: int = 150,
    log_x: bool = True,
    scoring_model_name: Optional[str] = None,
    baseline_colors: Optional[Dict[str, str]] = None,
) -> None:
    """Per-horizon cumulative NLL histograms + CDFs with all baselines overlaid.

    Produces a 2-row by *T*-column grid:

    * **Top row** — overlaid density histograms of per-trajectory
      cumulative NLL at each horizon *t* for the real data and every
      baseline's generated returns, all scored under the same model.
    * **Bottom row** — empirical CDFs with per-baseline
      :math:`|F_{\\text{real}} - F_{\\text{baseline}}|` areas shaded
      and :math:`W_1` values annotated.

    Args:
        cum_nll_real: ``[B, T]`` numpy array — cumulative NLL of real data.
        cum_nll_gen_dict: ``{baseline_name: [B_i, T] numpy array}``.
        save_path: File path for the saved figure.
        bins: Number of histogram bins per panel.
        dpi: Figure DPI.
        log_x: If ``True`` (default), use a logarithmic x-axis on both
            rows.  Cumulative NLL values are always positive and typically
            right-skewed, so log scale reveals the bulk and tail
            simultaneously.  Set to ``False`` for a linear x-axis.
        scoring_model_name: Name of the model used to score NLL (for title).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import wasserstein_distance as _w1

    baseline_names = list(cum_nll_gen_dict.keys())
    color_map = _resolve_baseline_color_map(
        baseline_names,
        baseline_colors=baseline_colors,
    )
    T = cum_nll_real.shape[1]

    fig, axes = plt.subplots(
        2, T,
        figsize=(6.5 * T, 10),
        gridspec_kw={"hspace": 0.38, "wspace": 0.28},
    )
    if T == 1:
        axes = axes.reshape(2, 1)

    def _ecdf(s, x):
        return np.searchsorted(np.sort(s), x, side="right") / len(s)

    fill_colors = [
        "mediumpurple", "lightcoral", "khaki",
        "palegreen", "lightsalmon", "lightskyblue",
    ]

    for t in range(T):
        real_t = cum_nll_real[:, t]

        # Compute shared bin edges from all distributions at this horizon
        all_t = [real_t]
        for name in baseline_names:
            all_t.append(cum_nll_gen_dict[name][:, t])
        combined = np.concatenate(all_t)
        lo, hi = np.percentile(combined, 0.5), np.percentile(combined, 99.5)
        use_log = log_x and lo > 0
        if use_log:
            bin_edges = np.logspace(np.log10(lo), np.log10(hi), bins + 1)
        else:
            bin_edges = np.linspace(lo, hi, bins + 1)

        # ---- Top: density histograms ---------------------------------
        ax_h = axes[0, t]
        ax_h.hist(
            real_t, bins=bin_edges, density=True, alpha=0.50,
            color="steelblue",
            label=(
                r"real  $\mu$=" + f"{real_t.mean():.1f}"
                + r",  $\sigma$=" + f"{real_t.std():.1f}"
            ),
        )
        for bi, name in enumerate(baseline_names):
            display_name = format_baseline_display_name(name)
            color = color_map[name]
            gen_t = cum_nll_gen_dict[name][:, t]
            ax_h.hist(
                gen_t, bins=bin_edges, density=True, alpha=0.40,
                color=color,
                label=(
                    f"{display_name}  "
                    r"$\mu$=" + f"{gen_t.mean():.1f}"
                    + r",  $\sigma$=" + f"{gen_t.std():.1f}"
                ),
            )
        if use_log:
            ax_h.set_xscale("log")
        ax_h.set_title(f"$t = {t + 1}$", fontsize=12)
        ax_h.set_xlabel(
            r"Cumulative NLL  $\ell_{b," + str(t + 1) + r"}$",
            fontsize=11,
        )
        if t == 0:
            ax_h.set_ylabel("Density", fontsize=11)
        ax_h.tick_params(axis="both", which="both", labelsize=8)
        ax_h.legend(fontsize=8)

        # ---- Bottom: empirical CDFs ----------------------------------
        ax_c = axes[1, t]
        x_lo = float(combined.min())
        x_hi = float(combined.max())
        if use_log:
            x_grid = np.logspace(np.log10(max(x_lo, 1e-12)), np.log10(x_hi), 2000)
        else:
            x_grid = np.linspace(x_lo, x_hi, 2000)
        cdf_real = _ecdf(real_t, x_grid)

        ax_c.plot(
            x_grid, cdf_real, color="steelblue", linewidth=1.8,
            label=r"$F_{\mathrm{real}}$",
        )
        for bi, name in enumerate(baseline_names):
            display_name = format_baseline_display_name(name)
            color = color_map[name]
            fc = fill_colors[bi % len(fill_colors)]
            gen_t = cum_nll_gen_dict[name][:, t]
            cdf_gen = _ecdf(gen_t, x_grid)
            w1 = float(_w1(real_t, gen_t))
            ax_c.plot(
                x_grid, cdf_gen, color=color, linewidth=1.8,
                label=f"$F_{{\\mathrm{{{display_name}}}}}$",
            )
            ax_c.fill_between(
                x_grid, cdf_real, cdf_gen, alpha=0.20, color=fc,
                label=(
                    f"$|F_{{\\mathrm{{real}}}} - F_{{\\mathrm{{{display_name}}}}}|$"
                    f"  W$_1$={w1:.1f}"
                ),
            )
        if use_log:
            ax_c.set_xscale("log")
        ax_c.set_xlabel(
            r"Cumulative NLL  $\ell_{b," + str(t + 1) + r"}$",
            fontsize=11,
        )
        if t == 0:
            ax_c.set_ylabel("Empirical CDF", fontsize=11)
        ax_c.set_ylim(0, 1)
        ax_c.tick_params(axis="both", which="both", labelsize=8)
        ax_c.legend(fontsize=8)

    _scoring_label = (
        f"scored under {scoring_model_name}" if scoring_model_name
        else "shared scoring model"
    )
    fig.suptitle(
        f"Cumulative NLL histograms — real vs generated ({_scoring_label})",
        fontsize=13, y=1.01,
    )

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [Comparison] Cumulative NLL histogram figure saved to: {save_path}")
