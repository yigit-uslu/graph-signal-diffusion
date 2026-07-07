"""Tests for SP500 Fig 3 — paper dual-scorer NLL comparison.

``plot_nll_comparison_paper`` builds an ``n_scorers × 2`` grid (rows = scoring
models, cols = {density histogram, empirical CDF}).  Smoke-tested for the 2×2
(both scorers), the degraded 1×2 (single scorer), linear-x, and empty/no-op
cases.  NLL arrays are synthetic positive values (no model required).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for tests

from graph_signal_diffusion.evaluation import structural_metrics as sm


def _nll(n=400, loc=10.0, seed=0):
    rng = np.random.default_rng(seed)
    return np.abs(rng.normal(loc, 3.0, size=n)) + 0.1  # strictly positive


def test_plot_nll_comparison_paper_two_scorers_2x2(tmp_path):
    scorers = [
        {"name": "grw", "nll_real": _nll(seed=1),
         "nll_gen_dict": {"grw": _nll(seed=2), "diffusion": _nll(seed=3)}},
        # Different scale for the diffusion scorer -> per-row x-range exercised.
        {"name": "diffusion", "nll_real": _nll(loc=40, seed=4),
         "nll_gen_dict": {"diffusion": _nll(loc=40, seed=5),
                          "grw": _nll(loc=40, seed=6)}},
    ]
    out = str(tmp_path / "fig3.pdf")
    sm.plot_nll_comparison_paper(
        scorers, out, bins=30, dpi=100, log_x=True,
        baseline_colors={"grw": "darkgreen", "diffusion": "#ff7f0e"},
    )
    assert os.path.exists(out) and os.path.getsize(out) > 0


def test_plot_nll_comparison_paper_single_scorer_linear(tmp_path):
    scorers = [
        {"name": "grw", "nll_real": _nll(seed=1),
         "nll_gen_dict": {"grw": _nll(seed=2), "diffusion": _nll(seed=3)}},
    ]
    out = str(tmp_path / "fig3b.pdf")
    sm.plot_nll_comparison_paper(scorers, out, bins=30, dpi=100, log_x=False)
    assert os.path.exists(out) and os.path.getsize(out) > 0


def test_plot_nll_comparison_paper_empty_is_noop(tmp_path):
    out = str(tmp_path / "none.pdf")
    sm.plot_nll_comparison_paper([], out)
    assert not os.path.exists(out)  # n == 0 returns early, writes nothing


def test_plot_nll_comparison_paper_skips_empty_gen_dict(tmp_path):
    # A scorer with an empty gen-dict is filtered out; remaining one still plots.
    scorers = [
        {"name": "diffusion", "nll_real": _nll(seed=7), "nll_gen_dict": {}},
        {"name": "grw", "nll_real": _nll(seed=1),
         "nll_gen_dict": {"grw": _nll(seed=2), "diffusion": _nll(seed=3)}},
    ]
    out = str(tmp_path / "fig3c.pdf")
    sm.plot_nll_comparison_paper(scorers, out, bins=20, dpi=100)
    assert os.path.exists(out) and os.path.getsize(out) > 0


def test_nll_use_log_rule():
    # Log only when requested, positive, and spanning >= 1 decade (ratio >= 10).
    assert sm._nll_use_log(100.0, 280.0, True) is False    # ratio 2.8 < 10 -> linear
    assert sm._nll_use_log(1.0, 1000.0, True) is True      # ratio 1000 -> log
    assert sm._nll_use_log(1.0, 1000.0, False) is False    # log disabled
    assert sm._nll_use_log(-5.0, 1000.0, True) is False    # non-positive min


def test_robust_density_ylim_top_excludes_self_scored():
    # The self-scored series (key == scorer) has a huge spike; it must NOT
    # drive the y-top.  Reference = real + the OTHER baselines.
    heights = {
        "__real__": np.array([1.0, 2.0, 1.5]),
        "grw": np.array([100.0, 90.0]),       # self-scored spike (scorer="grw")
        "diffusion": np.array([3.0, 2.0]),
    }
    top = sm._robust_density_ylim_top(heights, "grw", headroom=1.25)
    assert abs(top - 3.0 * 1.25) < 1e-9          # max(real=2, diffusion=3)=3
    # Case-insensitive scorer match.
    assert abs(sm._robust_density_ylim_top(heights, "GRW", 1.25) - 3.75) < 1e-9


def test_robust_density_ylim_top_none_when_only_self():
    # Only the self-scored series present → nothing to clip against → None.
    heights = {"grw": np.array([100.0, 90.0])}
    assert sm._robust_density_ylim_top(heights, "grw", 1.25) is None
    # Empty / all-empty arrays → None.
    assert sm._robust_density_ylim_top({}, "grw") is None
    assert sm._robust_density_ylim_top({"__real__": np.array([])}, "grw") is None


def test_plot_nll_comparison_paper_self_scored_spike_renders(tmp_path):
    # End-to-end: a self-scored GRW spike must not crash the robust-y-top path.
    spike = np.full(300, 50.0) + np.random.default_rng(0).normal(0, 0.2, 300)
    scorers = [{
        "name": "grw", "nll_real": _nll(loc=50, seed=1),
        "nll_gen_dict": {"grw": spike, "diffusion": _nll(loc=50, seed=2)},
    }]
    out = str(tmp_path / "fig3_spike.pdf")
    sm.plot_nll_comparison_paper(scorers, out, bins=40, dpi=100, log_x=False)
    assert os.path.exists(out) and os.path.getsize(out) > 0


def test_plot_nll_comparison_paper_wide_range_log(tmp_path):
    # A >= 1-decade range exercises the log-x branch + minor-label suppression.
    rng = np.random.default_rng(11)
    wide = np.abs(rng.normal(50.0, 200.0, size=400)) + 1.0  # spans many decades
    scorers = [{
        "name": "grw", "nll_real": wide,
        "nll_gen_dict": {"grw": wide * 1.1, "diffusion": wide * 0.9},
    }]
    out = str(tmp_path / "fig3_wide.pdf")
    sm.plot_nll_comparison_paper(scorers, out, bins=30, dpi=100, log_x=True)
    assert os.path.exists(out) and os.path.getsize(out) > 0


def test_annotate_grw_per_element_nats():
    import math
    grw = {"name": "grw", "nll_real": _nll(seed=1),
           "nll_gen_dict": {"grw": _nll(seed=2), "diffusion": _nll(seed=3)}}
    out = sm.annotate_grw_per_element_nats(grw, 2340)
    assert out["x_divisor"] == 2340.0
    assert "nats" in out["x_label"]
    assert abs(out["reference_value"] - 0.5 * math.log(2 * math.pi * math.e)) < 1e-9
    assert out is not grw and "x_divisor" not in grw  # original untouched
    # Case-insensitive name match.
    assert sm.annotate_grw_per_element_nats({**grw, "name": "GRW"}, 2340)["x_divisor"] == 2340.0
    # Non-GRW scorer (e.g. the diffusion proxy) is returned unchanged.
    diff = {"name": "diffusion", "nll_real": _nll(seed=4), "nll_gen_dict": {"diffusion": _nll(seed=5)}}
    assert sm.annotate_grw_per_element_nats(diff, 2340) is diff
    # Missing / invalid n_elements → no-op.
    assert sm.annotate_grw_per_element_nats(grw, None) is grw
    assert sm.annotate_grw_per_element_nats(grw, 0) is grw


def test_annotate_diffusion_per_step_mse():
    diff = {"name": "diffusion", "nll_real": _nll(seed=1),
            "nll_gen_dict": {"diffusion": _nll(seed=2), "grw": _nll(seed=3)}}
    out = sm.annotate_diffusion_per_step_mse(diff, 500)
    assert out["x_divisor"] == 500.0
    assert "MSE" in out["x_label"]
    assert "reference_value" not in out          # no floor line for MSE units
    assert out is not diff and "x_divisor" not in diff
    # Case-insensitive name match.
    assert sm.annotate_diffusion_per_step_mse({**diff, "name": "Diffusion"}, 500)["x_divisor"] == 500.0
    # Non-diffusion scorer (GRW) untouched.
    grw = {"name": "grw", "nll_real": _nll(seed=4), "nll_gen_dict": {"grw": _nll(seed=5)}}
    assert sm.annotate_diffusion_per_step_mse(grw, 500) is grw
    # Missing / invalid num_timesteps → no-op.
    assert sm.annotate_diffusion_per_step_mse(diff, None) is diff
    assert sm.annotate_diffusion_per_step_mse(diff, 0) is diff


def test_plot_nll_comparison_paper_with_divisor_and_reference(tmp_path):
    # GRW row carries x_divisor + reference line; must render and the W1 (now in
    # nats) stays finite.  Use raw GRW-scale values to exercise the rescale.
    raw = np.abs(np.random.default_rng(0).normal(3000, 400, 200)) + 1.0
    scorers = [sm.annotate_grw_per_element_nats(
        {"name": "grw", "nll_real": raw,
         "nll_gen_dict": {"grw": raw * 1.05, "diffusion": raw * 0.95}},
        2340,
    )]
    out = str(tmp_path / "fig3_nats.pdf")
    sm.plot_nll_comparison_paper(scorers, out, bins=30, dpi=100, log_x=False)
    assert os.path.exists(out) and os.path.getsize(out) > 0


def test_plot_nll_comparison_paper_with_plot_style(tmp_path):
    # Exercises the self-styling path: rc_context(global) + plots.nll_comparison.
    scorers = [{"name": "grw", "nll_real": _nll(seed=1),
                "nll_gen_dict": {"grw": _nll(seed=2), "diffusion": _nll(seed=3)}}]
    style = {
        "global": {"rc_params": {"axes.facecolor": "#EAEAF2", "axes.grid": True}},
        "baseline": {"colors": {"grw": "darkgreen", "diffusion": "#ff7f0e"}},
        "plots": {"nll_comparison": {"fig_width": 7.16, "row_height": 3.0, "dpi": 100,
                                     "fonts": {"title": 9, "label": 8}}},
    }
    out = str(tmp_path / "fig3_styled.pdf")
    sm.plot_nll_comparison_paper(scorers, out, plot_style=style)
    assert os.path.exists(out) and os.path.getsize(out) > 0


def test_annotate_diffusion_per_step_mse_label_is_elbo():
    # The diffusion (U-GNN) row labels its per-element denoising-MSE axis with the
    # ELBO term; this single label is shared by both the histogram and CDF panels.
    diff = {"name": "diffusion", "nll_real": _nll(seed=1),
            "nll_gen_dict": {"diffusion": _nll(seed=2)}}
    out = sm.annotate_diffusion_per_step_mse(diff, 500)
    assert out["x_label"] == r"Denoising MSE per element  (ELBO)"
    assert "ELBO" in out["x_label"] and "mathcal" not in out["x_label"]


def test_panel_titles_per_scorer_kind(tmp_path, monkeypatch):
    # Per-scorer panel kinds: GRW row → "NLL histogram"/"NLL CDF"; U-GNN
    # (diffusion) row → "ELBO histogram"/"ELBO CDF".  "Empirical CDF" lives on the
    # CDF y-label, not the title.  Capture titles + y-labels at fig-close since the
    # plotter saves & closes internally.  startswith/endswith avoids the em-dash.
    import matplotlib.pyplot as plt
    captured = {}
    orig_close = plt.close

    def _capture_close(fig=None):
        if fig is not None and hasattr(fig, "axes"):
            captured["titles"] = [ax.get_title() for ax in fig.axes]
            captured["ylabels"] = [ax.get_ylabel() for ax in fig.axes]
        return orig_close(fig)

    monkeypatch.setattr(plt, "close", _capture_close)
    scorers = [
        {"name": "grw", "nll_real": _nll(seed=1),
         "nll_gen_dict": {"grw": _nll(seed=2), "diffusion": _nll(seed=3)}},
        {"name": "diffusion", "nll_real": _nll(seed=4),
         "nll_gen_dict": {"grw": _nll(seed=5), "diffusion": _nll(seed=6)}},
    ]
    sm.plot_nll_comparison_paper(
        scorers, str(tmp_path / "fig3_titles.pdf"), bins=20, dpi=80,
    )
    titles = captured.get("titles", [])
    ylabels = captured.get("ylabels", [])
    # Histogram panels.
    assert any(t.startswith("GRW-scored") and t.endswith("NLL histogram")
               for t in titles)
    assert any(t.startswith("U-GNN-scored") and t.endswith("ELBO histogram")
               for t in titles)
    # CDF panels: kind-specific titles; "Empirical CDF" is the y-label, not title.
    assert any(t.startswith("GRW-scored") and t.endswith("NLL CDF")
               for t in titles)
    assert any(t.startswith("U-GNN-scored") and t.endswith("ELBO CDF")
               for t in titles)
    assert not any("Empirical CDF" in t for t in titles)
    assert sum(y == "Empirical CDF" for y in ylabels) == 2
