"""
K-bandlimited spectral leverage scores vs. learned node-selection frequencies.

This is a **standalone, CPU-only** analysis: it runs NO diffusion sampling and
loads NO model checkpoint. It reads two artifacts produced separately:

  1. The aggregate selector-sampling diagnostics JSON dumped by an eval /
     ``compare_baselines`` run (per-node, per-level, per-timestep selection
     frequencies pooled over all eval batches). See
     ``DiffusionTrainer._dump_aggregate_selector_sampling_json``.
  2. The static graph for the dataset (``processed/graph_static.pt`` or
     ``raw/adj.npy`` under the dataset root), from which the symmetric weighted
     adjacency ``W`` is built.

For each learned selector level it computes the K-bandlimited leverage score
``ℓ_v = Σ_{k<K} u_k(v)^2`` (smallest-K eigenvectors of the normalized symmetric
Laplacian by default), with ``K`` set to that level's realized kept-node budget,
and compares ``ℓ_v`` against the level's mean ``P(selected|active)`` via Spearman
/ Pearson correlation and top-K rank (Jaccard) overlap.

Example
-------
    python -m graph_signal_diffusion.analysis.leverage_scores \
        --diagnostics-json outputs/.../selector_diagnostics/selector_sampling_aggregate_test_epoch_4501.json \
        --dataset-root data/sp500/cleaned_drop_incomplete_min_coverage_0.95_corr_0.7_sector_bonus_0.05
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Dense eigh is used below this many nodes (the SP500 corr graph is ~471 nodes,
# well under this); above it, a sparse scipy.sparse.linalg.eigsh path is used.
_DENSE_EIGH_MAX_NODES = 2000


# --------------------------------------------------------------------------- #
# Core spectral quantity
# --------------------------------------------------------------------------- #
def leverage_scores(
    W: np.ndarray,
    K: int,
    normalized: bool = True,
) -> np.ndarray:
    """K-bandlimited leverage scores ``ℓ_v = Σ_{k<K} u_k(v)^2``.

    Args:
        W: ``(N, N)`` dense weighted adjacency. Symmetrised defensively.
        K: spectral bandwidth (number of smallest-eigenvalue eigenvectors).
            Clamped to ``[1, N]``.
        normalized: if True use the normalized symmetric Laplacian
            ``L = I - D^-1/2 W D^-1/2``; else the combinatorial ``L = D - W``.

    Returns:
        ``(N,)`` non-negative leverage scores. ``Σ_v ℓ_v == K`` (orthonormal
        eigenvectors), so the vector is a per-node share of K units of
        "spectral mass" concentrated in the low-frequency band.
    """
    W = np.asarray(W, dtype=float)
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        raise ValueError(f"W must be a square 2-D matrix, got shape {W.shape}")
    W = (W + W.T) / 2.0  # symmetrise defensively
    n = W.shape[0]
    K = int(max(1, min(int(K), n)))

    d = W.sum(axis=1)
    if normalized:
        dinv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-12))
        L = np.eye(n) - (dinv_sqrt[:, None] * W * dinv_sqrt[None, :])
        L = (L + L.T) / 2.0  # enforce exact symmetry for eigh
    else:
        L = np.diag(d) - W

    if n <= _DENSE_EIGH_MAX_NODES or K >= n:
        # Ascending eigenvalues; take the K smallest.
        _vals, U = np.linalg.eigh(L)
        U_K = U[:, :K]
    else:  # pragma: no cover - large-graph fallback
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla

        vals, U = spla.eigsh(sp.csr_matrix(L), k=K, which="SM")
        U_K = U[:, np.argsort(vals)]

    return (U_K ** 2).sum(axis=1)


# --------------------------------------------------------------------------- #
# Graph loading
# --------------------------------------------------------------------------- #
def dense_W_from_edges(
    edge_index: np.ndarray,
    edge_weight: Optional[np.ndarray],
    num_nodes: int,
) -> np.ndarray:
    """Build a dense ``(N, N)`` adjacency from a COO edge list."""
    ei = np.asarray(edge_index)
    if ei.shape[0] != 2:
        raise ValueError(f"edge_index must be (2, E), got {ei.shape}")
    w = (
        np.ones(ei.shape[1], dtype=float)
        if edge_weight is None
        else np.asarray(edge_weight, dtype=float).reshape(ei.shape[1], -1)[:, 0]
    )
    W = np.zeros((num_nodes, num_nodes), dtype=float)
    W[ei[0].astype(int), ei[1].astype(int)] = w
    return W


def load_static_graph(
    dataset_root: Optional[str] = None,
    graph_path: Optional[str] = None,
) -> Tuple[np.ndarray, Dict[str, Any], Optional[List[str]]]:
    """Load the static graph and return ``(W, info, symbols)``.

    Resolves, in order: an explicit ``graph_path``; ``<root>/processed/graph_static.pt``;
    ``<root>/raw/adj.npy``. ``symbols`` is a best-effort per-node label list (or
    None if unavailable).
    """
    candidates: List[str] = []
    if graph_path is not None:
        candidates.append(graph_path)
    if dataset_root is not None:
        candidates.append(os.path.join(dataset_root, "processed", "graph_static.pt"))
        candidates.append(os.path.join(dataset_root, "raw", "adj.npy"))
    path = next((c for c in candidates if os.path.exists(c)), None)
    if path is None:
        raise FileNotFoundError(
            f"No static graph found. Tried: {candidates or '[]'}"
        )

    info: Dict[str, Any] = {}
    symbols: Optional[List[str]] = None

    if path.endswith(".npy"):
        adj = np.load(path)
        if adj.ndim == 3:  # (N, N, F) -> collapse trailing feature dim
            adj = adj[..., 0]
        W = np.asarray(adj, dtype=float)
    else:
        import torch

        g = torch.load(path, map_location="cpu", weights_only=False)
        info = dict(g.get("info", {})) if isinstance(g, dict) else {}
        edge_index = g["edge_index"].cpu().numpy()
        ew = g.get("edge_weight", None)
        edge_weight = None if ew is None else ew.cpu().numpy()
        num_nodes = _resolve_num_nodes(info, edge_index)
        W = dense_W_from_edges(edge_index, edge_weight, num_nodes)
        symbols = _extract_symbols(info, g, W.shape[0])

    if symbols is None:
        symbols = _extract_symbols(info, None, W.shape[0])
    return W, info, symbols


def _resolve_num_nodes(info: Dict[str, Any], edge_index: np.ndarray) -> int:
    for key in ("Num_nodes", "num_nodes", "n_nodes", "N"):
        v = info.get(key) if isinstance(info, dict) else None
        if isinstance(v, (int, np.integer)) and int(v) > 0:
            return int(v)
    return int(edge_index.max()) + 1 if edge_index.size else 0


def _extract_symbols(
    info: Optional[Dict[str, Any]],
    graph_obj: Any,
    num_nodes: int,
) -> Optional[List[str]]:
    """Best-effort per-node label list of length ``num_nodes`` (else None)."""
    sources: List[Any] = []
    if isinstance(info, dict):
        for key in ("symbols", "tickers", "stocks", "stock_symbols", "node_names"):
            if key in info:
                sources.append(info[key])
    if isinstance(graph_obj, dict):
        for key in ("symbols", "tickers", "stocks"):
            if key in graph_obj:
                sources.append(graph_obj[key])
    for src in sources:
        try:
            labels = [str(s) for s in list(src)]
        except TypeError:
            continue
        if len(labels) == num_nodes:
            return labels
    return None


def _read_symbols_file(path: str) -> Optional[List[str]]:
    """Read a per-node label list from .csv / .json / .txt (in node order).

    - ``.csv``: uses a ``Symbol``/``symbol``/``ticker`` column if present, else
      the first column (header row skipped).
    - ``.json``: a list of labels, or a dict mapping (stringified) node index ->
      label, ordered by integer key.
    - anything else (``.txt``): one label per non-empty line.
    """
    try:
        if path.endswith(".json"):
            with open(path, "r") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                try:
                    return [str(obj[k]) for k in sorted(obj, key=lambda x: int(x))]
                except (TypeError, ValueError):
                    return [str(v) for v in obj.values()]
            return [str(s) for s in obj]
        if path.endswith(".csv"):
            import csv

            with open(path, "r", newline="") as f:
                rows = list(csv.reader(f))
            if not rows:
                return None
            header = [h.strip() for h in rows[0]]
            col = 0
            for name in ("Symbol", "symbol", "Ticker", "ticker"):
                if name in header:
                    col = header.index(name)
                    break
            return [r[col].strip() for r in rows[1:] if len(r) > col and r[col].strip()]
        with open(path, "r") as f:
            return [line.strip() for line in f if line.strip()]
    except Exception as e:  # pragma: no cover - defensive
        warnings.warn(f"Failed to read symbols file {path!r}: {e}", stacklevel=2)
        return None


def load_symbols(
    num_nodes: int,
    symbols_path: Optional[str] = None,
    dataset_root: Optional[str] = None,
) -> Optional[List[str]]:
    """Resolve per-node labels of length ``num_nodes`` (else None).

    Priority: explicit ``symbols_path`` -> ``<dataset_root>/raw/stocks.csv``. A
    source whose length does not match ``num_nodes`` is rejected (with a warning)
    to avoid silently mislabeling nodes — the node order must match the
    adjacency's row order (verified for SP500: stocks.csv row order == node order).
    """
    candidates: List[str] = []
    if symbols_path is not None:
        candidates.append(symbols_path)
    if dataset_root is not None:
        candidates.append(os.path.join(dataset_root, "raw", "stocks.csv"))
    for path in candidates:
        if not os.path.exists(path):
            continue
        syms = _read_symbols_file(path)
        if not syms:
            continue
        if len(syms) == num_nodes:
            return syms
        warnings.warn(
            f"symbols source {path!r} has {len(syms)} entries != {num_nodes} nodes; "
            "ignoring (cannot guarantee node-order alignment).",
            stacklevel=2,
        )
    return None


# --------------------------------------------------------------------------- #
# Learned selection-frequency reduction
# --------------------------------------------------------------------------- #
def low_noise_probe_indices(
    probe_timesteps: Optional[List[Any]],
    n_rows: int,
    probe_timestep_fraction: float,
) -> Tuple[List[int], Optional[float]]:
    """Probe-row indices in the low-noise window ``t <= fraction * t_max``.

    Mirrors the WRA ``probe_timestep_fraction`` knob: timesteps near the end of
    generation (small ``t``, low noise) are the meaningful ones for interpreting
    downsampling. Returns ``(indices, t_threshold)``.

    - ``fraction >= 1.0`` selects every timestep (since all ``t <= t_max``).
    - If no timestep falls in the window, falls back to the single lowest ``t``.
    - If ``probe_timesteps`` is missing or length-mismatched, returns all rows
      with ``t_threshold = None``.
    """
    if not isinstance(probe_timesteps, (list, tuple)) or len(probe_timesteps) != n_rows:
        return list(range(n_rows)), None
    ts = [float(t) for t in probe_timesteps]
    t_max = max(ts)
    t_threshold = float(probe_timestep_fraction) * t_max
    idx = [i for i, t in enumerate(ts) if t <= t_threshold]
    if not idx:
        idx = [int(np.argmin(ts))]
    return idx, t_threshold


def level_learned_importance(
    network_diag: Dict[str, Any],
    level_key: str,
    probe_timestep_fraction: float = 1.0,
    measure: str = "conditional",
    min_active_frac: float = 0.0,
) -> Tuple[np.ndarray, int]:
    """Per-node learned importance and realized kept-budget K for one level.

    ``measure`` selects the learned signal:

    - ``"conditional"`` (default): mean ``P(selected | active)`` — isolates this
      selector level's preference, conditioned on the node reaching its pool.
    - ``"freq"``: mean unconditional ``P(selected)`` (``== P(active) *
      P(selected | active)``) — cumulative retention through this level. Defined
      over all nodes and matches leverage's all-node / K-budget framing, so it is
      the more natural target for "what does the whole pyramid keep" at deep
      levels.

    The non-requested signal is used as a fallback if the requested one is absent
    from the diagnostics. The chosen signal is averaged (ignoring NaN) over the
    **low-noise probe window** ``t <= probe_timestep_fraction * t_max`` — the
    timesteps closest to the end of generation. ``probe_timestep_fraction=1.0``
    (the default) averages over all probe timesteps.

    ``min_active_frac`` (default 0.0 = off): nodes active in fewer than this
    fraction of probed graphs (pooled over the window) have their importance set
    to NaN so they drop out of the downstream correlation. This tames the
    high-variance tail of the conditional at deep levels, where many nodes are
    active in only a handful of graphs. The mask never changes K.

    K is the realized kept-node budget over the same window:
    ``round(mean_t Σ_node selected[t] / graph_count[t])`` — the average number
    of nodes the selector keeps at this level. When per-node counts are present
    K is clamped to ``>= 1``; only when counts are entirely absent does it fall
    back to the count of finite-importance nodes (computed BEFORE any
    availability mask, so the budget is mask-independent).

    Returns ``(importance (N,), K)``.
    """
    if measure not in ("conditional", "freq"):
        raise ValueError(f"measure must be 'conditional' or 'freq', got {measure!r}")
    cond = network_diag.get("selection_given_available_by_level", {}) or {}
    freq = network_diag.get("selection_freq_by_level", {}) or {}
    # Primary source per requested measure; fall back to the other so a
    # diagnostics file carrying only one of the two still works.
    primary, fallback = (cond, freq) if measure == "conditional" else (freq, cond)
    matrix = primary.get(level_key, None)
    if matrix is None:
        matrix = fallback.get(level_key, None)
    if matrix is None:
        raise KeyError(f"level {level_key!r} not present in diagnostics")
    mat = np.asarray(matrix, dtype=float)  # (num_probe, N)
    if mat.ndim != 2:
        raise ValueError(f"level {level_key!r} matrix must be 2-D, got {mat.shape}")

    idx, _t_threshold = low_noise_probe_indices(
        network_diag.get("probe_timesteps", None), mat.shape[0], probe_timestep_fraction
    )

    with warnings.catch_warnings():
        # All-NaN columns (nodes never available at this level) legitimately
        # produce NaN; suppress numpy's "Mean of empty slice" RuntimeWarning.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        importance = np.nanmean(mat[idx, :], axis=0)  # (N,)

    # Realized kept-budget K from raw counts (over the same window) when
    # available. Computed on the UNMASKED importance so the availability mask
    # below (which only affects which nodes we correlate) never changes K.
    selected = network_diag.get("selected_count_by_level", {}).get(level_key, None)
    graph_count = network_diag.get("probe_graph_count", None)
    k_val: Optional[int] = None
    if selected is not None and graph_count is not None:
        sel = np.asarray(selected, dtype=float)  # (num_probe, N)
        gc = np.asarray(graph_count, dtype=float).reshape(-1)  # (num_probe,)
        win = [i for i in idx if i < sel.shape[0] and i < gc.size]
        if win:
            denom = np.clip(gc[win], 1.0, None)
            kept_per_graph = sel[win].sum(axis=1) / denom  # (len(win),)
            k_val = max(1, int(round(float(np.nanmean(kept_per_graph)))))
    if k_val is None or k_val <= 0:
        # Counts absent -> fall back to the number of finite-importance nodes.
        k_val = int(np.isfinite(importance).sum())

    # Optionally mask low-availability nodes -> NaN (dropped from correlation).
    if min_active_frac and min_active_frac > 0.0:
        importance = _apply_min_active_mask(
            importance, network_diag, level_key, idx, float(min_active_frac)
        )
    return importance, max(1, k_val)


def _apply_min_active_mask(
    importance: np.ndarray,
    network_diag: Dict[str, Any],
    level_key: str,
    window_idx: List[int],
    min_active_frac: float,
) -> np.ndarray:
    """Set importance to NaN for nodes active in < ``min_active_frac`` of graphs.

    Availability fraction is pooled over the probe window:
    ``Σ_t available_count[t, v] / Σ_t graph_count[t]``. Requires
    ``available_count_by_level`` and ``probe_graph_count``; if either is missing
    (or shape-mismatched) the importance is returned unchanged, with a warning.
    """
    avail = network_diag.get("available_count_by_level", {}).get(level_key, None)
    graph_count = network_diag.get("probe_graph_count", None)
    if avail is None or graph_count is None:
        warnings.warn(
            f"min_active_frac={min_active_frac:g} requested but "
            "available_count_by_level / probe_graph_count is missing for level "
            f"{level_key!r}; skipping availability mask.",
            stacklevel=2,
        )
        return importance
    a = np.asarray(avail, dtype=float)  # (num_probe, N)
    gc = np.asarray(graph_count, dtype=float).reshape(-1)  # (num_probe,)
    if a.ndim != 2 or a.shape[1] != importance.shape[0]:
        warnings.warn(
            f"available_count for level {level_key!r} has shape {a.shape}, "
            f"incompatible with {importance.shape[0]} nodes; skipping mask.",
            stacklevel=2,
        )
        return importance
    win = [i for i in window_idx if i < a.shape[0] and i < gc.size]
    total_graphs = float(gc[win].sum()) if win else 0.0
    if not win or total_graphs <= 0.0:
        return importance
    active_frac = a[win].sum(axis=0) / total_graphs  # (N,)
    out = importance.copy()
    out[active_frac < min_active_frac] = np.nan
    return out


# --------------------------------------------------------------------------- #
# Comparison metrics
# --------------------------------------------------------------------------- #
def rank_agreement(
    leverage: np.ndarray,
    importance: np.ndarray,
    top_k: int,
) -> Dict[str, float]:
    """Spearman / Pearson correlation + top-K Jaccard between two node scores.

    NaN entries in ``importance`` are dropped pairwise. The top-K sets are the K
    highest-scoring nodes under each measure (restricted to the finite subset).
    """
    lev = np.asarray(leverage, dtype=float)
    imp = np.asarray(importance, dtype=float)
    mask = np.isfinite(lev) & np.isfinite(imp)
    n = int(mask.sum())
    out: Dict[str, float] = {"n_nodes": float(n)}
    if n < 2:
        out.update({"spearman": float("nan"), "pearson": float("nan"), "topk_jaccard": float("nan")})
        return out

    lev_m, imp_m = lev[mask], imp[mask]
    from scipy.stats import pearsonr, spearmanr

    out["spearman"] = float(spearmanr(lev_m, imp_m).correlation)
    # Pearson is undefined when either input is constant.
    if np.ptp(lev_m) == 0 or np.ptp(imp_m) == 0:
        out["pearson"] = float("nan")
    else:
        out["pearson"] = float(pearsonr(lev_m, imp_m)[0])

    k = int(max(1, min(top_k, n)))
    idx = np.flatnonzero(mask)
    top_lev = set(idx[np.argsort(lev_m)[::-1][:k]].tolist())
    top_imp = set(idx[np.argsort(imp_m)[::-1][:k]].tolist())
    union = top_lev | top_imp
    out["topk_jaccard"] = float(len(top_lev & top_imp) / len(union)) if union else float("nan")
    out["top_k"] = float(k)
    return out


# --------------------------------------------------------------------------- #
# Orchestration + plotting
# --------------------------------------------------------------------------- #
def _select_network(diagnostics: Dict[str, Any], network_key: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    by_network = diagnostics.get("by_network", None)
    if isinstance(by_network, dict) and by_network:
        if network_key is not None:
            if network_key not in by_network:
                raise KeyError(f"network_key {network_key!r} not in {sorted(by_network)}")
            return network_key, by_network[network_key]
        # Default: the network with the most probed graphs.
        def _total_graphs(d: Dict[str, Any]) -> float:
            return float(np.nansum(np.asarray(d.get("probe_graph_count", [0.0]), dtype=float)))

        key = max(by_network.keys(), key=lambda k: _total_graphs(by_network[k]))
        return key, by_network[key]
    # No by_network — treat the top-level dict as the (only) network bucket.
    return (network_key or "aggregate"), diagnostics


def compare_leverage_vs_selection(
    diagnostics: Dict[str, Any],
    W: np.ndarray,
    *,
    normalized: bool = True,
    network_key: Optional[str] = None,
    symbols: Optional[List[str]] = None,
    out_dir: Optional[str] = None,
    annotate_top: int = 10,
    probe_timestep_fraction: float = 0.1,
    measure: str = "conditional",
    min_active_frac: float = 0.0,
) -> Dict[str, Any]:
    """Compute per-level leverage-vs-selection agreement and (optionally) plot.

    ``probe_timestep_fraction`` (default 0.1) restricts the learned importance to
    the low-noise probe window ``t <= fraction * t_max`` — the timesteps closest
    to the end of generation, which are the meaningful ones for downsampling
    interpretation (mirrors the WRA selector-rate knob).

    ``measure`` (``"conditional"`` | ``"freq"``) and ``min_active_frac`` are
    forwarded to :func:`level_learned_importance` — see that function for their
    semantics. With ``measure="freq"`` the output filenames gain a ``_freq``
    suffix so conditional and frequency runs do not overwrite each other.

    Returns a stats dict: ``{"network_key", "normalized", "num_nodes",
    "probe_timestep_fraction", "measure", "min_active_frac", "levels":
    {level_key: {"K", "num_probe_used", "t_threshold", **rank_agreement}}}``.
    """
    key, network_diag = _select_network(diagnostics, network_key)
    n_nodes = int(W.shape[0])

    cond = network_diag.get("selection_given_available_by_level", {}) or {}
    freq = network_diag.get("selection_freq_by_level", {}) or {}
    level_keys = sorted(
        set(cond.keys()) | set(freq.keys()),
        key=lambda k: (int(k) if str(k).lstrip("-").isdigit() else 0, str(k)),
    )

    stats: Dict[str, Any] = {
        "network_key": key,
        "normalized": bool(normalized),
        "num_nodes": n_nodes,
        "probe_timestep_fraction": float(probe_timestep_fraction),
        "measure": str(measure),
        "min_active_frac": float(min_active_frac),
        "levels": {},
    }
    per_level_arrays: List[Tuple[str, np.ndarray, np.ndarray, int]] = []

    for level_key in level_keys:
        importance, k_val = level_learned_importance(
            network_diag,
            level_key,
            probe_timestep_fraction=probe_timestep_fraction,
            measure=measure,
            min_active_frac=min_active_frac,
        )
        if importance.size != n_nodes:
            # Node-count mismatch between diagnostics and graph -> skip safely.
            stats["levels"][level_key] = {"error": f"node mismatch {importance.size} vs {n_nodes}"}
            continue
        # Record which probe timesteps fed the importance reduction.
        matrix = cond.get(level_key, freq.get(level_key))
        n_rows = int(np.asarray(matrix, dtype=float).shape[0])
        win_idx, t_threshold = low_noise_probe_indices(
            network_diag.get("probe_timesteps", None), n_rows, probe_timestep_fraction
        )
        lev = leverage_scores(W, K=k_val, normalized=normalized)
        agree = rank_agreement(lev, importance, top_k=k_val)
        agree["K"] = int(k_val)
        agree["num_probe_used"] = int(len(win_idx))
        agree["t_threshold"] = (None if t_threshold is None else float(t_threshold))
        stats["levels"][level_key] = agree
        per_level_arrays.append((level_key, lev, importance, k_val))

    if out_dir is not None and per_level_arrays:
        os.makedirs(out_dir, exist_ok=True)
        _plot_comparison(
            per_level_arrays, key, normalized, symbols, out_dir, annotate_top,
            probe_timestep_fraction=probe_timestep_fraction,
            measure=measure, min_active_frac=min_active_frac,
        )
        with open(os.path.join(out_dir, _stats_filename(key, normalized, measure)), "w") as f:
            json.dump(stats, f, indent=2)
    return stats


def _artifact_tag(network_key: str, normalized: bool, measure: str) -> str:
    """Shared filename stem for the JSON/PDF artifacts of one comparison run.

    ``conditional`` (the default) keeps the legacy ``..._{lap}`` stem so existing
    artifacts/callers are unaffected; other measures append ``_{measure}`` so the
    two do not overwrite each other.
    """
    tag = "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(network_key)).strip("_") or "net"
    lap = "norm" if normalized else "comb"
    meas = "" if measure == "conditional" else f"_{measure}"
    return f"leverage_vs_selection_{tag}_{lap}{meas}"


def _stats_filename(network_key: str, normalized: bool, measure: str = "conditional") -> str:
    return f"{_artifact_tag(network_key, normalized, measure)}.json"


def _plot_comparison(
    per_level_arrays: List[Tuple[str, np.ndarray, np.ndarray, int]],
    network_key: str,
    normalized: bool,
    symbols: Optional[List[str]],
    out_dir: str,
    annotate_top: int,
    probe_timestep_fraction: float = 0.1,
    measure: str = "conditional",
    min_active_frac: float = 0.0,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    window_desc = (
        "all timesteps"
        if probe_timestep_fraction >= 1.0
        else f"low-noise window t≤{probe_timestep_fraction:g}·t_max"
    )
    meas_desc = "P(selected | active)" if measure == "conditional" else "selection freq P(selected)"
    n_levels = len(per_level_arrays)
    fig, axes = plt.subplots(1, n_levels, figsize=(max(4.0, 4.5 * n_levels), 4.2), squeeze=False)
    for ax, (level_key, lev, imp, k_val) in zip(axes[0], per_level_arrays):
        mask = np.isfinite(lev) & np.isfinite(imp)
        ax.scatter(lev[mask], imp[mask], s=12, alpha=0.6, edgecolors="none")
        rho = float(spearmanr(lev[mask], imp[mask]).correlation) if mask.sum() >= 2 else float("nan")
        ax.set_title(f"Level {level_key} | K={k_val} | Spearman={rho:.3f}", fontsize=10)
        ax.set_xlabel(f"Leverage score ({'norm.' if normalized else 'comb.'} Laplacian, K={k_val})")
        # Window info (low-noise vs all-timesteps) lives in the suptitle, not here.
        ax.set_ylabel(f"Mean {meas_desc}")
        ax.grid(alpha=0.2)
        # Annotate the union of top-N nodes by each axis (leverage / importance).
        idx = np.flatnonzero(mask)
        if idx.size:
            top = set(idx[np.argsort(imp[mask])[::-1][:annotate_top]].tolist())
            top |= set(idx[np.argsort(lev[mask])[::-1][:annotate_top]].tolist())
            for node in top:
                label = symbols[node] if (symbols and node < len(symbols)) else str(node)
                ax.annotate(label, (lev[node], imp[node]), fontsize=6, alpha=0.8)

    active_desc = "" if min_active_frac <= 0.0 else f" | active≥{min_active_frac:g}"
    fig.suptitle(
        f"K-bandlimited leverage vs learned {meas_desc} | {network_key} | "
        f"{window_desc}{active_desc}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(out_dir, f"{_artifact_tag(network_key, normalized, measure)}.pdf"),
        dpi=150,
    )
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _load_diagnostics(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)  # json.load accepts NaN tokens by default


def main(argv: Optional[List[str]] = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--diagnostics-json", required=True, help="Aggregate selector-sampling diagnostics JSON.")
    p.add_argument("--dataset-root", default=None, help="Dataset root (loads processed/graph_static.pt or raw/adj.npy).")
    p.add_argument("--graph-path", default=None, help="Explicit path to graph_static.pt or adj.npy (overrides --dataset-root).")
    p.add_argument("--normalized", type=lambda s: str(s).lower() not in ("0", "false", "no"), default=True,
                   help="Use the normalized symmetric Laplacian (default true).")
    p.add_argument("--network-key", default=None, help="Which by_network key to analyse (default: most-probed).")
    p.add_argument("--out-dir", default=None, help="Output dir (default: <diagnostics dir>/leverage_comparison).")
    p.add_argument(
        "--symbols", default=None,
        help="Per-node label source for scatter annotations (.csv with a Symbol "
             "column / .json list / .txt one-per-line). If omitted, falls back to "
             "<dataset-root>/raw/stocks.csv, then node index.",
    )
    p.add_argument("--annotate-top", type=int, default=10, help="Top-N nodes per measure to label on the scatter.")
    p.add_argument(
        "--probe-timestep-fraction", type=float, default=0.1,
        help="Average selection freq only over the low-noise window t<=fraction*t_max "
             "(timesteps near end of generation). 1.0 => all timesteps. Default 0.1.",
    )
    p.add_argument(
        "--importance-measure", choices=["conditional", "freq"], default="conditional",
        help="Learned signal compared against leverage: 'conditional' = mean "
             "P(selected|active), isolates each selector level [default]; 'freq' = mean "
             "unconditional P(selected) = cumulative pyramid retention (matches leverage's "
             "all-node / K-budget framing; better headline at deep levels). 'freq' artifacts "
             "get a _freq filename suffix.",
    )
    p.add_argument(
        "--min-active-frac", type=float, default=0.0,
        help="Mask nodes active in fewer than this fraction of probed graphs (per level, over "
             "the probe window) -> dropped from the correlation. Tames the high-variance tail of "
             "the conditional at deep levels. Never changes K. Default 0.0 (no masking).",
    )
    args = p.parse_args(argv)

    if args.dataset_root is None and args.graph_path is None:
        p.error("one of --dataset-root or --graph-path is required")

    diagnostics = _load_diagnostics(args.diagnostics_json)
    W, _info, graph_symbols = load_static_graph(dataset_root=args.dataset_root, graph_path=args.graph_path)
    # Priority: explicit --symbols -> raw/stocks.csv -> graph's own labels -> node index.
    symbols = load_symbols(W.shape[0], symbols_path=args.symbols, dataset_root=args.dataset_root) or graph_symbols
    out_dir = args.out_dir or os.path.join(os.path.dirname(os.path.abspath(args.diagnostics_json)), "leverage_comparison")

    stats = compare_leverage_vs_selection(
        diagnostics,
        W,
        normalized=args.normalized,
        network_key=args.network_key,
        symbols=symbols,
        out_dir=out_dir,
        annotate_top=args.annotate_top,
        probe_timestep_fraction=args.probe_timestep_fraction,
        measure=args.importance_measure,
        min_active_frac=args.min_active_frac,
    )
    print(json.dumps(stats, indent=2))
    print(f"\nSaved comparison to: {out_dir}")


if __name__ == "__main__":
    main()
