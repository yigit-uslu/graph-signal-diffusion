"""Tests for SP500 Fig 2 — paper structural comparison.

Covers:
  * the new ``normalize=False`` path of ``compute_autocorr_profile_per_stock``
    (raw autocovariance γ_l instead of the normalized ACF ρ_l), and
  * a smoke test of ``plot_structural_comparison_paper`` (1×2: twin-axis
    combined ACF + eigenvalue comparison).
"""
import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for tests

from graph_signal_diffusion.evaluation import structural_metrics as sm


def _returns(B=24, T=8, N=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, T, N, generator=g)


def test_autocorr_per_stock_normalize_false_is_autocovariance():
    r = _returns(seed=0)
    norm = sm.compute_autocorr_profile_per_stock(
        r, max_lag=3, squared=True, normalize=True,
    ).numpy()
    raw = sm.compute_autocorr_profile_per_stock(
        r, max_lag=3, squared=True, normalize=False,
    ).numpy()

    # Normalized lag-0 is identically 1; raw lag-0 is the variance (>0, ≠ 1).
    assert np.allclose(norm[0], 1.0, atol=1e-5)
    assert np.all(raw[0] > 0)
    assert not np.allclose(raw[0], 1.0)

    # raw[lag] == norm[lag] * gamma_0  (raw[0] == gamma_0)
    for lag in range(1, 4):
        assert np.allclose(raw[lag], norm[lag] * raw[0], rtol=1e-4, atol=1e-7)


def test_autocorr_per_stock_normalize_defaults_true():
    r = _returns(seed=1)
    a = sm.compute_autocorr_profile_per_stock(r, max_lag=2, squared=False).numpy()
    # Default behavior unchanged: normalized → row 0 is identically 1.
    assert np.allclose(a[0], 1.0, atol=1e-5)


def test_autocorr_normalize_false_phased_matches_unphased_relationship():
    """Phased averaging must still produce autocovariances (linear in phases)."""
    r = _returns(B=24, T=8, N=5, seed=2)
    wsi = torch.arange(24)  # distinct start indices → phased estimator engages
    raw = sm.compute_autocorr_profile_per_stock(
        r, max_lag=2, squared=True, window_stride=8,
        window_start_indices=wsi, normalize=False,
    ).numpy()
    norm = sm.compute_autocorr_profile_per_stock(
        r, max_lag=2, squared=True, window_stride=8,
        window_start_indices=wsi, normalize=True,
    ).numpy()
    assert raw.shape == norm.shape == (3, 5)
    assert np.all(raw[0] > 0)
    assert np.allclose(norm[0], 1.0, atol=1e-5)


def test_plot_structural_comparison_paper_writes_pdf(tmp_path):
    real = _returns(seed=1)
    gen = {"grw": _returns(seed=2), "diffusion": _returns(seed=3)}
    out = str(tmp_path / "fig2.pdf")
    sm.plot_structural_comparison_paper(
        real, gen, save_path=out,
        max_lag=3, n_eigenvalues=4, dpi=100,
        baseline_colors={"grw": "darkgreen", "diffusion": "#ff7f0e"},
        normalize=False, include_lag0=True, twin_axis=True, show_iqr=True,
    )
    assert os.path.exists(out) and os.path.getsize(out) > 0


def test_plot_structural_comparison_paper_no_lag0_single_baseline(tmp_path):
    # Branch coverage: drop lag 0, single baseline, no twin axis, no IQR bars.
    real = _returns(seed=1)
    gen = {"diffusion": _returns(seed=3)}
    out = str(tmp_path / "fig2b.pdf")
    sm.plot_structural_comparison_paper(
        real, gen, save_path=out,
        max_lag=3, n_eigenvalues=4, dpi=100,
        normalize=False, include_lag0=False, twin_axis=False, show_iqr=False,
    )
    assert os.path.exists(out) and os.path.getsize(out) > 0


def test_plot_structural_comparison_paper_normalized_variant(tmp_path):
    # normalize=True should also render (ρ_l in [-1,1], lag-0 ≡ 1).
    real = _returns(seed=4)
    gen = {"diffusion": _returns(seed=5)}
    out = str(tmp_path / "fig2c.pdf")
    sm.plot_structural_comparison_paper(
        real, gen, save_path=out, max_lag=2, n_eigenvalues=3, dpi=100,
        normalize=True, include_lag0=True, twin_axis=True,
    )
    assert os.path.exists(out) and os.path.getsize(out) > 0


def test_plot_structural_comparison_paper_default_normalized_shared(tmp_path):
    # New SP500 Fig 2 defaults: normalized ACF, lag 0 dropped, single shared
    # y-axis (no twin axis).  Must render with all-default flags.
    real = _returns(seed=8)
    gen = {"grw": _returns(seed=9), "diffusion": _returns(seed=10)}
    out = str(tmp_path / "fig2_default.pdf")
    sm.plot_structural_comparison_paper(
        real, gen, save_path=out, max_lag=2, n_eigenvalues=4, dpi=100,
    )
    assert os.path.exists(out) and os.path.getsize(out) > 0


def test_resolve_paper_style_merges_global_per_figure_and_colors():
    from graph_signal_diffusion.evaluation.plot_style_apply import resolve_paper_style
    ps = {
        "global": {"rc_params": {"axes.facecolor": "#EAEAF2"}, "default_dpi": 150},
        "baseline": {"colors": {"real": "steelblue", "grw": "darkgreen"}},
        "plots": {"structural_comparison": {"dpi": 300, "real_color": "navy"}},
    }
    rc, pfig, colors = resolve_paper_style(
        ps, "structural_comparison", {"diffusion": "#ff7f0e"},
    )
    assert rc["axes.facecolor"] == "#EAEAF2"
    assert rc["figure.dpi"] == 150.0 and rc["savefig.dpi"] == 150.0
    assert pfig["dpi"] == 300 and pfig["real_color"] == "navy"
    # baseline.colors merged with the explicit override (override wins/adds).
    assert colors["grw"] == "darkgreen" and colors["diffusion"] == "#ff7f0e"
    # None -> all empty (plotter falls back to its built-in defaults).
    rc0, pf0, c0 = resolve_paper_style(None, "structural_comparison")
    assert rc0 == {} and pf0 == {} and c0 == {}


def test_plot_structural_comparison_paper_with_plot_style(tmp_path):
    # Exercises the self-styling path: rc_context(global) + plots.structural_comparison.
    real = _returns(seed=1)
    gen = {"grw": _returns(seed=2), "diffusion": _returns(seed=3)}
    style = {
        "global": {"rc_params": {"axes.facecolor": "#EAEAF2", "axes.grid": True}},
        "baseline": {"colors": {"real": "steelblue", "grw": "darkgreen",
                                "diffusion": "#ff7f0e"}},
        "plots": {"structural_comparison": {"figure_size": [7.16, 3.3], "dpi": 100,
                                            "fonts": {"title": 10, "label": 9}}},
    }
    out = str(tmp_path / "fig2_styled.pdf")
    sm.plot_structural_comparison_paper(
        real, gen, save_path=out, max_lag=3, n_eigenvalues=4, plot_style=style,
    )
    assert os.path.exists(out) and os.path.getsize(out) > 0
