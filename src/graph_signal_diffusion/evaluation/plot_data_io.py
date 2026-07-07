"""Sidecar plot-data dump/load helpers for the compare-baselines pipeline.

Each polished ``.pdf`` produced under ``eval_viz/`` gets a sibling ``.npz``
holding the *minimum* set of arrays needed to re-run the corresponding plot
function.  The companion :mod:`graph_signal_diffusion.cli.replot_baselines`
script reads these files and regenerates the figures without re-running
evaluation.

Layout (collocated with their plots):

* ``eval_viz/{baseline}/{split}/structural_data.npz`` — drives the 5
  per-baseline structural plots (raw ``cat_real``/``cat_gen`` + precomputed
  cumulative-std arrays).
* ``eval_viz/{baseline}/{split}/nll_data.npz`` — drives the 2 per-baseline
  NLL plots (only baselines that score NLL, e.g. GRW).
* ``eval_viz/{baseline}/{split}/stock_prices/t{ts}_b{bi}.npz`` — one
  sidecar per per-window ``stock_prices_*.pdf``.
* ``eval_viz/comparison/structural_data.npz`` — drives the comparison
  structural + cumulative-std plots.  Heavy ``cat_real``/``cat_gen``
  tensors are *referenced* by relative path back to the per-baseline
  files instead of duplicated; only the small precomputed cum-std
  arrays are stored inline.
* ``eval_viz/comparison/nll_data__{scorer}.npz`` — one per scoring model.

All readers tolerate missing optional keys.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Mapping, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_NONE_SENTINEL = "__none__"


def _to_np(x: Any) -> np.ndarray:
    """Best-effort tensor → numpy conversion (CPU, contiguous)."""
    if x is None:
        return np.array(_NONE_SENTINEL)
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def _is_none_sentinel(x: np.ndarray) -> bool:
    return x.dtype.kind in ("U", "S", "O") and x.size == 1 and str(x.item()) == _NONE_SENTINEL


def _opt(x: np.ndarray) -> Optional[np.ndarray]:
    return None if _is_none_sentinel(x) else x


def _safe_key(name: str) -> str:
    """Sanitize a baseline name for use as an npz key suffix."""
    return "".join(c if (c.isalnum() or c in "_-") else "_" for c in str(name))


def _atomic_savez(path: str, data: Mapping[str, Any]) -> None:
    """Write ``np.savez_compressed`` atomically (tmp file + rename).

    ``np.savez_compressed`` auto-appends ``.npz`` if missing, so we
    pass a tmp name that already ends in ``.npz`` to avoid surprises.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **data)
    os.replace(tmp, path)


def _rel(path: str, start: str) -> str:
    """Relative path with forward slashes (portable across OSes for storage)."""
    return os.path.relpath(path, start).replace(os.sep, "/")


# ---------------------------------------------------------------------------
# Per-baseline structural dump / load
# ---------------------------------------------------------------------------

def dump_per_baseline_structural(
    out_path: str,
    *,
    cat_real: Any,
    cat_gen: Any,
    model_name: str,
    max_lag: int,
    n_eigenvalues: int,
    window_stride: int,
    window_start_indices: Any = None,
    cum_std_real: Any = None,
    cum_std_gen: Any = None,
    cum_std_real_tn: Any = None,
    cum_std_gen_tn: Any = None,
    theoretical: Any = None,
    theoretical_label: Optional[str] = None,
    dpi: int = 150,
) -> None:
    """Dump the consolidated per-baseline structural plot data."""
    payload: Dict[str, Any] = {
        "schema_version": np.int64(1),
        "cat_real": _to_np(cat_real).astype(np.float32, copy=False),
        "cat_gen": _to_np(cat_gen).astype(np.float32, copy=False),
        "model_name": np.array(str(model_name)),
        "max_lag": np.int64(int(max_lag)),
        "n_eigenvalues": np.int64(int(n_eigenvalues)),
        "window_stride": np.int64(int(window_stride)),
        "window_start_indices": _to_np(window_start_indices),
        "cum_std_real": _to_np(cum_std_real),
        "cum_std_gen": _to_np(cum_std_gen),
        "cum_std_real_tn": _to_np(cum_std_real_tn),
        "cum_std_gen_tn": _to_np(cum_std_gen_tn),
        "theoretical": _to_np(theoretical),
        "theoretical_label": np.array(str(theoretical_label) if theoretical_label is not None else _NONE_SENTINEL),
        "dpi": np.int64(int(dpi)),
    }
    _atomic_savez(out_path, payload)


def load_per_baseline_structural(path: str) -> Dict[str, Any]:
    """Inverse of :func:`dump_per_baseline_structural`."""
    with np.load(path, allow_pickle=False) as f:
        out: Dict[str, Any] = {
            "cat_real": f["cat_real"],
            "cat_gen": f["cat_gen"],
            "model_name": str(f["model_name"].item()),
            "max_lag": int(f["max_lag"].item()),
            "n_eigenvalues": int(f["n_eigenvalues"].item()),
            "window_stride": int(f["window_stride"].item()),
            "window_start_indices": _opt(f["window_start_indices"]),
            "cum_std_real": _opt(f["cum_std_real"]),
            "cum_std_gen": _opt(f["cum_std_gen"]),
            "cum_std_real_tn": _opt(f["cum_std_real_tn"]),
            "cum_std_gen_tn": _opt(f["cum_std_gen_tn"]),
            "theoretical": _opt(f["theoretical"]),
            "dpi": int(f["dpi"].item()),
        }
        tl = str(f["theoretical_label"].item())
        out["theoretical_label"] = None if tl == _NONE_SENTINEL else tl
    return out


# ---------------------------------------------------------------------------
# Comparison structural dump / load
# ---------------------------------------------------------------------------

def dump_comparison_structural(
    out_path: str,
    *,
    baseline_names: List[str],
    per_baseline_paths: Mapping[str, str],
    max_lag: int,
    n_eigenvalues: int,
    cum_std_real: Any,
    cum_std_real_per_stock: Any,
    cum_std_gen_dict: Mapping[str, Any],
    cum_std_gen_per_stock_dict: Mapping[str, Any],
    theoretical: Any = None,
    baseline_colors: Optional[Mapping[str, str]] = None,
) -> None:
    """Dump the consolidation file for comparison structural plots.

    Heavy tensors (``cat_real``/``cat_gen``) are *not* duplicated — they
    are referenced by ``per_baseline_paths`` (relative paths to each
    baseline's ``structural_data.npz``).  The replot CLI follows the
    references at load time.
    """
    out_dir = os.path.dirname(out_path) or "."
    rel_paths = {
        name: _rel(p, out_dir) for name, p in per_baseline_paths.items()
    }
    payload: Dict[str, Any] = {
        "schema_version": np.int64(1),
        "baseline_names": np.array([str(n) for n in baseline_names]),
        "per_baseline_paths_json": np.array(json.dumps(rel_paths)),
        "max_lag": np.int64(int(max_lag)),
        "n_eigenvalues": np.int64(int(n_eigenvalues)),
        "cum_std_real": _to_np(cum_std_real),
        "cum_std_real_per_stock": _to_np(cum_std_real_per_stock),
        "theoretical": _to_np(theoretical),
        "baseline_colors_json": np.array(
            json.dumps(dict(baseline_colors)) if baseline_colors else _NONE_SENTINEL
        ),
    }
    for name in baseline_names:
        key = _safe_key(name)
        payload[f"cum_std_gen__{key}"] = _to_np(cum_std_gen_dict[name])
        payload[f"cum_std_gen_per_stock__{key}"] = _to_np(
            cum_std_gen_per_stock_dict[name]
        )
    _atomic_savez(out_path, payload)


def load_comparison_structural(path: str) -> Dict[str, Any]:
    base_dir = os.path.dirname(path) or "."
    with np.load(path, allow_pickle=False) as f:
        baseline_names = [str(n) for n in f["baseline_names"]]
        rel_paths = json.loads(str(f["per_baseline_paths_json"].item()))
        per_baseline_abs = {
            name: os.path.normpath(os.path.join(base_dir, rel_paths[name]))
            for name in baseline_names
        }
        cum_std_gen_dict: Dict[str, np.ndarray] = {}
        cum_std_gen_per_stock_dict: Dict[str, np.ndarray] = {}
        for name in baseline_names:
            key = _safe_key(name)
            cum_std_gen_dict[name] = f[f"cum_std_gen__{key}"]
            cum_std_gen_per_stock_dict[name] = f[f"cum_std_gen_per_stock__{key}"]
        bc_raw = str(f["baseline_colors_json"].item())
        baseline_colors = None if bc_raw == _NONE_SENTINEL else json.loads(bc_raw)
        out: Dict[str, Any] = {
            "baseline_names": baseline_names,
            "per_baseline_paths": per_baseline_abs,
            "max_lag": int(f["max_lag"].item()),
            "n_eigenvalues": int(f["n_eigenvalues"].item()),
            "cum_std_real": f["cum_std_real"],
            "cum_std_real_per_stock": f["cum_std_real_per_stock"],
            "cum_std_gen_dict": cum_std_gen_dict,
            "cum_std_gen_per_stock_dict": cum_std_gen_per_stock_dict,
            "theoretical": _opt(f["theoretical"]),
            "baseline_colors": baseline_colors,
        }

    # Resolve heavy tensors lazily by reading the per-baseline files.
    # Replot consumers expect the canonical structure used by the plot
    # functions, so we eagerly load the small bits we need here.
    cat_real = None
    cat_gen_dict: Dict[str, np.ndarray] = {}
    window_start_indices = None
    window_stride = 1
    window_start_indices_gen: Dict[str, Optional[np.ndarray]] = {}
    for name in baseline_names:
        bp = per_baseline_abs[name]
        if not os.path.exists(bp):
            raise FileNotFoundError(
                f"comparison file references missing per-baseline data: {bp}"
            )
        bd = load_per_baseline_structural(bp)
        if cat_real is None:
            cat_real = bd["cat_real"]
            window_start_indices = bd["window_start_indices"]
            window_stride = bd["window_stride"]
        cat_gen_dict[name] = bd["cat_gen"]
        window_start_indices_gen[name] = bd["window_start_indices"]
    out.update({
        "cat_real": cat_real,
        "cat_gen_dict": cat_gen_dict,
        "window_start_indices": window_start_indices,
        "window_stride": window_stride,
        "window_start_indices_gen": window_start_indices_gen,
    })
    return out


# ---------------------------------------------------------------------------
# Per-baseline NLL dump / load (one file holds both nll + cum-nll arrays)
# ---------------------------------------------------------------------------

def dump_per_baseline_nll(
    out_path: str,
    *,
    nll_real: Any,
    nll_gen: Any,
    cum_nll_real_bt: Any = None,
    cum_nll_gen_bt: Any = None,
    bins: int = 60,
    dpi: int = 150,
    log_x: bool = True,
    scoring_model_name: Optional[str] = None,
) -> None:
    payload: Dict[str, Any] = {
        "schema_version": np.int64(1),
        "nll_real": _to_np(nll_real),
        "nll_gen": _to_np(nll_gen),
        "cum_nll_real_bt": _to_np(cum_nll_real_bt),
        "cum_nll_gen_bt": _to_np(cum_nll_gen_bt),
        "bins": np.int64(int(bins)),
        "dpi": np.int64(int(dpi)),
        "log_x": np.bool_(bool(log_x)),
        "scoring_model_name": np.array(
            str(scoring_model_name) if scoring_model_name is not None else _NONE_SENTINEL
        ),
    }
    _atomic_savez(out_path, payload)


def load_per_baseline_nll(path: str) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as f:
        sn = str(f["scoring_model_name"].item())
        return {
            "nll_real": f["nll_real"],
            "nll_gen": f["nll_gen"],
            "cum_nll_real_bt": _opt(f["cum_nll_real_bt"]),
            "cum_nll_gen_bt": _opt(f["cum_nll_gen_bt"]),
            "bins": int(f["bins"].item()),
            "dpi": int(f["dpi"].item()),
            "log_x": bool(f["log_x"].item()),
            "scoring_model_name": None if sn == _NONE_SENTINEL else sn,
        }


# ---------------------------------------------------------------------------
# Comparison NLL dump / load (one file per scoring model)
# ---------------------------------------------------------------------------

def dump_comparison_nll(
    out_path: str,
    *,
    nll_real: Any,
    nll_gen_dict: Mapping[str, Any],
    cum_nll_real: Any = None,
    cum_nll_gen_dict: Optional[Mapping[str, Any]] = None,
    scoring_model_name: str,
    bins: int = 60,
    log_x: bool = True,
    baseline_colors: Optional[Mapping[str, str]] = None,
) -> None:
    baseline_names = list(nll_gen_dict.keys())
    payload: Dict[str, Any] = {
        "schema_version": np.int64(1),
        "scoring_model_name": np.array(str(scoring_model_name)),
        "baseline_names": np.array([str(n) for n in baseline_names]),
        "nll_real": _to_np(nll_real),
        "cum_nll_real": _to_np(cum_nll_real),
        "bins": np.int64(int(bins)),
        "log_x": np.bool_(bool(log_x)),
        "baseline_colors_json": np.array(
            json.dumps(dict(baseline_colors)) if baseline_colors else _NONE_SENTINEL
        ),
    }
    cum_dict = dict(cum_nll_gen_dict) if cum_nll_gen_dict else {}
    has_cum_names: List[str] = []
    for name in baseline_names:
        key = _safe_key(name)
        payload[f"nll_gen__{key}"] = _to_np(nll_gen_dict[name])
        if name in cum_dict:
            payload[f"cum_nll_gen__{key}"] = _to_np(cum_dict[name])
            has_cum_names.append(name)
    payload["cum_nll_baseline_names"] = np.array([str(n) for n in has_cum_names])
    _atomic_savez(out_path, payload)


def load_comparison_nll(path: str) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as f:
        baseline_names = [str(n) for n in f["baseline_names"]]
        cum_names = [str(n) for n in f["cum_nll_baseline_names"]]
        nll_gen_dict: Dict[str, np.ndarray] = {}
        cum_nll_gen_dict: Dict[str, np.ndarray] = {}
        for name in baseline_names:
            key = _safe_key(name)
            nll_gen_dict[name] = f[f"nll_gen__{key}"]
        for name in cum_names:
            key = _safe_key(name)
            cum_nll_gen_dict[name] = f[f"cum_nll_gen__{key}"]
        bc_raw = str(f["baseline_colors_json"].item())
        return {
            "scoring_model_name": str(f["scoring_model_name"].item()),
            "baseline_names": baseline_names,
            "nll_real": f["nll_real"],
            "nll_gen_dict": nll_gen_dict,
            "cum_nll_real": _opt(f["cum_nll_real"]),
            "cum_nll_gen_dict": cum_nll_gen_dict if cum_nll_gen_dict else None,
            "bins": int(f["bins"].item()),
            "log_x": bool(f["log_x"].item()),
            "baseline_colors": None if bc_raw == _NONE_SENTINEL else json.loads(bc_raw),
        }


# ---------------------------------------------------------------------------
# Per-window stock_prices dump / load
# ---------------------------------------------------------------------------

def dump_stock_prices_window(
    out_path: str,
    *,
    pred_mean: Any,
    target_mean: Any,
    gen_ensemble: Any = None,
    hist_samples: Any = None,
    stock_indices: Any,
    stock_symbols: Optional[List[str]] = None,
    batch_idx: int = 0,
    time_start: Optional[int] = None,
) -> None:
    payload: Dict[str, Any] = {
        "schema_version": np.int64(1),
        "pred_mean": _to_np(pred_mean).astype(np.float32, copy=False),
        "target_mean": _to_np(target_mean).astype(np.float32, copy=False),
        "gen_ensemble": _to_np(gen_ensemble),
        "hist_samples": _to_np(hist_samples),
        "stock_indices": np.asarray(list(stock_indices), dtype=np.int64),
        "stock_symbols": np.array(
            [str(s) for s in (stock_symbols or [])], dtype=object
        ).astype(str) if stock_symbols else np.array([_NONE_SENTINEL]),
        "batch_idx": np.int64(int(batch_idx)),
        "time_start": np.int64(int(time_start)) if time_start is not None else np.array(_NONE_SENTINEL),
    }
    _atomic_savez(out_path, payload)


def load_stock_prices_window(path: str) -> Dict[str, Any]:
    with np.load(path, allow_pickle=False) as f:
        ts_arr = f["time_start"]
        time_start: Optional[int]
        if ts_arr.dtype.kind in ("U", "S", "O"):
            time_start = None
        else:
            time_start = int(ts_arr.item())
        symbols_raw = f["stock_symbols"]
        if symbols_raw.size == 1 and str(symbols_raw[0]) == _NONE_SENTINEL:
            stock_symbols = None
        else:
            stock_symbols = [str(s) for s in symbols_raw]
        return {
            "pred_mean": f["pred_mean"],
            "target_mean": f["target_mean"],
            "gen_ensemble": _opt(f["gen_ensemble"]),
            "hist_samples": _opt(f["hist_samples"]),
            "stock_indices": [int(x) for x in f["stock_indices"]],
            "stock_symbols": stock_symbols,
            "batch_idx": int(f["batch_idx"].item()),
            "time_start": time_start,
        }


# ---------------------------------------------------------------------------
# Paper Fig 1 forecast-column dump / load (the already-sliced n_stocks panel)
# ---------------------------------------------------------------------------

def dump_paper_fig1_column(out_path: str, *, column: Mapping[str, Any]) -> None:
    """Dump one paper Fig 1 forecast column (the sliced ``n_stocks`` panel data).

    ``column`` is the dict produced by
    ``StockPriceForecastingTaskV2._select_window_column`` — already sliced to the
    chosen stocks for a single window, so this is a *tiny* sidecar (n_stocks
    rows, not all stocks).  Re-loaded by :func:`load_paper_fig1_column` and
    rendered by ``_plot_paper_forecast_column``, so a replot reproduces the
    panel exactly (same window, same stocks).
    """
    c = dict(column)

    def _arr_or_none(x: Any) -> np.ndarray:
        return (
            _to_np(x).astype(np.float32, copy=False) if x is not None
            else _to_np(None)
        )

    symbols = c.get("stock_symbols")
    date_labels = c.get("date_labels")
    time_start = c.get("time_start")
    payload: Dict[str, Any] = {
        "schema_version": np.int64(1),
        "pred": _to_np(c["pred"]).astype(np.float32, copy=False),       # [T_f, n]
        "target": _to_np(c["target"]).astype(np.float32, copy=False),   # [T_f, n]
        "ensemble": _arr_or_none(c.get("ensemble")),                    # [K,T_f,n]|none
        "hist": _arr_or_none(c.get("hist")),                            # [T_hist,n]|none
        "stock_indices": np.asarray(
            list(c.get("stock_indices") or []), dtype=np.int64,
        ),
        "stock_symbols": (
            np.array([str(s) for s in symbols], dtype=object).astype(str)
            if symbols else np.array([_NONE_SENTINEL])
        ),
        "date_labels": (
            np.asarray(list(date_labels), dtype=np.int64)
            if date_labels is not None else np.array([_NONE_SENTINEL])
        ),
        "time_start": (
            np.int64(int(time_start)) if time_start is not None
            else np.array(_NONE_SENTINEL)
        ),
    }
    _atomic_savez(out_path, payload)


def load_paper_fig1_column(path: str) -> Dict[str, Any]:
    """Inverse of :func:`dump_paper_fig1_column` → column dict for the plotter."""
    with np.load(path, allow_pickle=False) as f:
        sym_raw = f["stock_symbols"]
        if sym_raw.size == 1 and str(sym_raw[0]) == _NONE_SENTINEL:
            symbols = None
        else:
            symbols = [str(s) for s in sym_raw]
        dl_raw = f["date_labels"]
        if dl_raw.dtype.kind in ("U", "S", "O"):
            date_labels = None
        else:
            date_labels = [int(x) for x in dl_raw]
        ts_raw = f["time_start"]
        time_start = (
            None if ts_raw.dtype.kind in ("U", "S", "O") else int(ts_raw.item())
        )
        return {
            "pred": f["pred"],
            "target": f["target"],
            "ensemble": _opt(f["ensemble"]),
            "hist": _opt(f["hist"]),
            "stock_indices": [int(x) for x in f["stock_indices"]],
            "stock_symbols": symbols,
            "date_labels": date_labels,
            "time_start": time_start,
        }
