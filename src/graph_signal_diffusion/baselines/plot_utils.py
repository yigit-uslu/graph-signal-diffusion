"""Shared plotting utilities for baseline diagnostics."""
from __future__ import annotations

import numpy as np
from typing import Any


_BASELINE_DISPLAY_ALIASES: dict[str, str] = {
    "fp": "FP",
    "ap": "AP",
    "grw": "GRW",
    "wmmse": "WMMSE",
    "diffusion": "U-GNN",
    "ddpm": "DDPM",
    "ddim": "DDIM",
    "mlp": "MLP",
    "gnn": "GNN",
    "lstm": "LSTM",
    "rnn": "RNN",
    "cnn": "CNN",
    "arima": "ARIMA",
}

_BASELINE_DISPLAY_ALIASES_OVERRIDE: dict[str, str] = {}


def set_baseline_display_aliases(aliases: dict[str, Any] | None) -> None:
    """Override baseline display aliases used by :func:`format_baseline_display_name`.

    Passing ``None`` or an empty mapping resets to built-in defaults.
    """
    global _BASELINE_DISPLAY_ALIASES_OVERRIDE
    if not aliases:
        _BASELINE_DISPLAY_ALIASES_OVERRIDE = {}
        return
    _BASELINE_DISPLAY_ALIASES_OVERRIDE = {
        str(k).strip().lower(): str(v)
        for k, v in aliases.items()
        if str(k).strip()
    }


def format_baseline_display_name(name: Any) -> str:
    """Format baseline names for plot legends/labels.

    Known abbreviated baseline IDs are rendered in uppercase, while other
    names are returned unchanged.
    """
    raw = "" if name is None else str(name)
    key = raw.strip().lower()
    if not key:
        return raw
    if key in _BASELINE_DISPLAY_ALIASES_OVERRIDE:
        return _BASELINE_DISPLAY_ALIASES_OVERRIDE[key]
    return _BASELINE_DISPLAY_ALIASES.get(key, raw)


def ecdf(s: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Evaluate the empirical CDF of *s* at each point in *x*.

    Args:
        s: 1-D sample array.
        x: Points at which to evaluate the CDF.

    Returns:
        Array of the same shape as *x* with values in ``[0, 1]``.
    """
    return np.searchsorted(np.sort(s), x, side="right") / len(s)
