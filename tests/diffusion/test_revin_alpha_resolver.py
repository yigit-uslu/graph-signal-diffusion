"""Tests for the ``${revin_alpha:w,b}`` OmegaConf resolver.

Computes α(w, b) = 1 / (1 − w·(1 − b)) — the closed-form σ correction
for the RevIN blend that pairs with ``revin_blend_weight``. See
docs/agent_summaries/revin_blend_weight_2026-05-26.md for the derivation.
"""
from __future__ import annotations

import math

import pytest
from omegaconf import OmegaConf

# Importing the diffusion package registers the resolver as a side effect.
import graph_signal_diffusion.diffusion  # noqa: F401
from graph_signal_diffusion.diffusion.revin_resolvers import (
    revin_alpha_resolver,
    register_revin_resolvers,
)


# ---------------------------------------------------------------------------
# Closed-form values (no Hydra/OmegaConf involved)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "w, b, expected",
    [
        # w=0 → α=1.0 regardless of b
        (0.0, 0.82, 1.0),
        (0.0, 0.50, 1.0),
        # w=1, b=0.82 → SP500 baseline α=1.22
        (1.0, 0.82, 1.0 / 0.82),
        # w=0.5, b=0.82 → α=1.10
        (0.5, 0.82, 1.0 / 0.91),
        # w=0.7, b=0.82 → α=1.14
        (0.7, 0.82, 1.0 / 0.874),
        # b=1.0 → α=1.0 regardless of w (already unbiased)
        (0.5, 1.0, 1.0),
        (1.0, 1.0, 1.0),
    ],
)
def test_alpha_closed_form(w, b, expected):
    got = revin_alpha_resolver(w, b)
    assert got == pytest.approx(expected, rel=1e-8, abs=1e-10), (
        f"α({w}, {b}) = {got}, expected {expected}"
    )


def test_alpha_raises_on_invalid_denominator():
    """For pathological inputs (w > 1 with b < 0) the denominator could
    go non-positive; resolver must fail loudly, not silently produce
    inf/NaN."""
    with pytest.raises(ValueError, match="denominator"):
        revin_alpha_resolver(w=2.0, b=0.0)  # 1 - 2*(1-0) = -1


# ---------------------------------------------------------------------------
# OmegaConf integration (the actual yaml-side use case)
# ---------------------------------------------------------------------------

def test_resolver_registered_on_diffusion_package_import():
    """Importing graph_signal_diffusion.diffusion registers ${revin_alpha}.
    We re-call register to verify idempotence."""
    register_revin_resolvers()  # must not raise

    cfg = OmegaConf.create({
        "blend": 0.7,
        "b": 0.82,
        "alpha": "${revin_alpha:${blend},${b}}",
    })
    resolved_alpha = OmegaConf.to_container(cfg, resolve=True)["alpha"]
    assert resolved_alpha == pytest.approx(1.0 / 0.874, rel=1e-8)


def test_resolver_with_oc_select_fallback():
    """The typical config pattern uses oc.select to fall back to a literal
    b when dataset_info isn't injected.  Verify the fallback path works.
    """
    cfg = OmegaConf.create({
        "blend": 0.5,
        # No dataset_info present, so this should fall back to 0.82.
        "alpha": "${revin_alpha:${blend},${oc.select:dataset_info.sigma_cond_mean_train,0.82}}",
    })
    resolved_alpha = OmegaConf.to_container(cfg, resolve=True)["alpha"]
    assert resolved_alpha == pytest.approx(1.0 / 0.91, rel=1e-8)


def test_resolver_reads_injected_dataset_info():
    """When dataset_info.sigma_cond_mean_train IS present in the cfg,
    the resolver picks it up via oc.select."""
    cfg = OmegaConf.create({
        "blend": 0.7,
        "dataset_info": {"sigma_cond_mean_train": 0.78},
        "alpha": "${revin_alpha:${blend},${oc.select:dataset_info.sigma_cond_mean_train,0.82}}",
    })
    resolved_alpha = OmegaConf.to_container(cfg, resolve=True)["alpha"]
    # b=0.78, w=0.7 → 1 / (1 - 0.7·0.22) = 1 / 0.846 ≈ 1.182
    expected = 1.0 / (1.0 - 0.7 * (1.0 - 0.78))
    assert resolved_alpha == pytest.approx(expected, rel=1e-8)


# ---------------------------------------------------------------------------
# Sanity: monotonicity properties of α(w)
# ---------------------------------------------------------------------------

def test_alpha_monotone_in_w_for_fixed_b():
    """For fixed b < 1, α(w) is monotone non-decreasing in w (since
    smaller blend → less bias correction needed)."""
    b = 0.82
    alphas = [revin_alpha_resolver(w, b) for w in (0.0, 0.25, 0.5, 0.75, 1.0)]
    for prev, nxt in zip(alphas[:-1], alphas[1:]):
        assert nxt >= prev - 1e-12, f"α(w) not monotone in w: {alphas}"


def test_alpha_at_w_one_inverse_of_b():
    """α(1, b) = 1 / b — the simplest sanity identity."""
    for b in (0.5, 0.7, 0.82, 0.95, 1.0):
        assert revin_alpha_resolver(1.0, b) == pytest.approx(1.0 / b, rel=1e-8)
