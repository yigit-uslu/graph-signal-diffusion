"""OmegaConf resolvers for RevIN-related config interpolation.

Currently provides ``${revin_alpha:w,b}`` which computes the bias-compensating
σ correction from the blend weight ``w`` and the training-set bias
``b = E[σ_cond_w]``:

    α(w, b) = 1 / (1 - w · (1 - b))

This is the closed-form derived in
``docs/agent_summaries/revin_blend_weight_2026-05-26.md``. Letting the YAML
do this math means the user only has to pick ``w``; ``α`` follows
deterministically from training-set statistics.

Typical usage in a diffusion config:

    revin_blend_weight: 0.7
    revin_sigma_correction: ${revin_alpha:${.revin_blend_weight},${oc.select:dataset_info.sigma_cond_mean_train,0.82}}

The ``oc.select:...,0.82`` provides a literal-fallback so the config still
resolves when no dataset_info has been injected (e.g., for quick prototyping
without the train CLI's auto-injection step).

The resolver fires at config-resolution time, so the resolved numeric value
is what gets saved to ``.hydra/config.yaml`` for the run. At checkpoint-load
time no resolver is needed — the literal ``revin_sigma_correction`` is
already in the saved config.
"""
from __future__ import annotations

from typing import Any

from omegaconf import OmegaConf


def revin_alpha_resolver(w: Any, b: Any) -> float:
    """Compute α(w, b) = 1 / (1 - w·(1 - b)).

    Args:
        w: blend weight in [0, 1]. 1.0 = pure per-window σ_cond_w;
           0.0 = constant 1.0 (RevIN's σ becomes a no-op).
        b: training-set mean of σ_cond_w (≈ 0.82 for SP500 with past_window=20).
           When w=1.0 and b=0.82, α=1.22; when w=0.0, α=1.0 regardless of b.

    Returns:
        The σ scale correction α as a Python float.

    Raises:
        ValueError: if the denominator ``1 - w·(1 - b)`` is non-positive
            (would require b > 1 + (1-b)·(w-1)/w; never the case for
            valid SP500-style data, but guarded against silent NaN/inf).
    """
    w_f = float(w)
    b_f = float(b)
    denom = 1.0 - w_f * (1.0 - b_f)
    if denom <= 0.0:
        raise ValueError(
            f"revin_alpha: denominator {denom:.6f} ≤ 0 for w={w_f}, b={b_f}. "
            f"Expected 0 ≤ w ≤ 1 and 0 < b ≤ 1 (typical b ≈ 0.8 for SP500)."
        )
    return 1.0 / denom


def register_revin_resolvers() -> None:
    """Register the ``revin_alpha`` resolver with OmegaConf.

    Idempotent — safe to call multiple times (silently ignores if the
    resolver is already registered, matching the WRA resolver pattern).
    """
    try:
        OmegaConf.register_new_resolver("revin_alpha", revin_alpha_resolver)
    except ValueError:
        # Already registered (likely because multiple modules import this).
        pass
