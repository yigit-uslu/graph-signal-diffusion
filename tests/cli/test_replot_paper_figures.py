"""Tests for Phase 2c — ``replot_baselines`` regenerates the SP500 paper
figures (Fig 2 / Fig 3) from comparison sidecars, self-styled from the run's
``.hydra/config.yaml`` snapshot.

Everything here is synthetic (no GPU, no model): we hand-build the per-baseline
+ comparison structural sidecars (and the per-scorer NLL sidecars) with
``plot_data_io.dump_*`` and a minimal snapshot, then exercise

  * ``load_plot_style_from_run_dir`` (full subtree / missing → None),
  * ``_load_paper_cfg`` (behavioral cfg + scorer-row order),
  * ``_replot_paper_figures`` (both figures, gated-off, Fig-2-only), and
  * an end-to-end ``main()`` smoke (the os-shadowing bug taught us to
    exercise ``main()`` — unit tests alone missed it).
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for tests

from omegaconf import OmegaConf

from graph_signal_diffusion.cli import replot_baselines as rb
from graph_signal_diffusion.evaluation.plot_data_io import (
    dump_comparison_nll,
    dump_comparison_structural,
    dump_per_baseline_structural,
)
from graph_signal_diffusion.evaluation.plot_style_apply import (
    load_plot_style_from_run_dir,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_PLOT_STYLE = {
    "global": {
        "backend": "Agg",
        "default_dpi": 150,
        "rc_params": {"axes.facecolor": "#EAEAF2", "axes.grid": True,
                      "font.family": "serif"},
    },
    "baseline": {"colors": {"real": "steelblue", "grw": "darkgreen",
                            "diffusion": "#ff7f0e"}},
    "plots": {
        "structural_comparison": {"figure_size": [7.16, 3.3], "dpi": 100,
                                  "real_color": "steelblue",
                                  "fonts": {"title": 10, "label": 9}},
        "nll_comparison": {"fig_width": 7.16, "row_height": 3.0, "dpi": 100,
                           "real_color": "steelblue",
                           "fonts": {"title": 9, "label": 8}},
    },
}

_PAPER_FIGURES = {
    "enabled": True,
    "fig2": {"max_lag": None, "n_eigenvalues": None, "normalize": True,
             "include_lag0": False, "show_iqr": True, "twin_axis": False},
    "fig3": {"log_x": None, "bins": 30},
}


def _returns(B=24, T=8, N=6, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((B, T, N)).astype(np.float32)


def _nll(n=200, loc=10.0, seed=0):
    rng = np.random.default_rng(seed)
    return (np.abs(rng.normal(loc, 3.0, size=n)) + 0.1).astype(np.float32)


def _build_eval_viz(eval_viz, *, with_nll=True, baselines=("grw", "diffusion")):
    """Create per-baseline + comparison structural sidecars (+ optional NLL).

    Mirrors the real on-disk layout: ``eval_viz/{baseline}/test/`` per-baseline
    files referenced (by relative path) from ``eval_viz/comparison/``.
    """
    comp_dir = os.path.join(eval_viz, "comparison")
    os.makedirs(comp_dir, exist_ok=True)
    T, N = 8, 6
    cat_real = _returns(seed=1)
    per_paths, cum_gen, cum_gen_ps = {}, {}, {}
    for i, name in enumerate(baselines):
        bdir = os.path.join(eval_viz, name, "test")
        os.makedirs(bdir, exist_ok=True)
        p = os.path.join(bdir, "structural_data.npz")
        dump_per_baseline_structural(
            p, cat_real=cat_real, cat_gen=_returns(seed=10 + i),
            model_name=name, max_lag=3, n_eigenvalues=4, window_stride=8,
        )
        per_paths[name] = p
        cum_gen[name] = np.linspace(0.1, 1.0, T).astype(np.float32)
        cum_gen_ps[name] = np.ones((T, N), dtype=np.float32)
    dump_comparison_structural(
        os.path.join(comp_dir, "structural_data.npz"),
        baseline_names=list(baselines), per_baseline_paths=per_paths,
        max_lag=3, n_eigenvalues=4,
        cum_std_real=np.linspace(0.1, 1.0, T).astype(np.float32),
        cum_std_real_per_stock=np.ones((T, N), dtype=np.float32),
        cum_std_gen_dict=cum_gen, cum_std_gen_per_stock_dict=cum_gen_ps,
        baseline_colors={"grw": "darkgreen", "diffusion": "#ff7f0e"},
    )
    if with_nll:
        for i, scorer in enumerate(baselines):
            gen = {b: _nll(loc=10 + 5 * i, seed=100 + i * 10 + j)
                   for j, b in enumerate(baselines)}
            dump_comparison_nll(
                os.path.join(comp_dir, f"nll_data__{scorer}.npz"),
                nll_real=_nll(loc=10 + 5 * i, seed=200 + i),
                nll_gen_dict=gen, scoring_model_name=scorer,
                bins=30, log_x=False,
                baseline_colors={"grw": "darkgreen", "diffusion": "#ff7f0e"},
            )


def _write_snapshot(run_dir, *, plot_style=_PLOT_STYLE, paper=_PAPER_FIGURES,
                    order=("grw", "diffusion")):
    hydra_dir = os.path.join(run_dir, ".hydra")
    os.makedirs(hydra_dir, exist_ok=True)
    cfg = OmegaConf.create({
        "plot_style": plot_style,
        "paper_figures": paper,
        "baselines_to_compare": list(order),
    })
    OmegaConf.save(cfg, os.path.join(hydra_dir, "config.yaml"))


# ---------------------------------------------------------------------------
# load_plot_style_from_run_dir
# ---------------------------------------------------------------------------

def test_load_plot_style_from_run_dir_full_subtree(tmp_path):
    run = str(tmp_path / "run")
    _write_snapshot(run)
    ps = load_plot_style_from_run_dir(run)
    assert isinstance(ps, dict)
    # Full subtree (not just `.global`): baseline + plots present too.
    assert set(ps) >= {"global", "baseline", "plots"}
    assert ps["plots"]["structural_comparison"]["dpi"] == 100
    assert ps["baseline"]["colors"]["diffusion"] == "#ff7f0e"


def test_load_plot_style_from_run_dir_missing_returns_none(tmp_path):
    # No snapshot at all.
    assert load_plot_style_from_run_dir(str(tmp_path / "nope")) is None
    # Snapshot present but without a plot_style node.
    run = str(tmp_path / "run2")
    os.makedirs(os.path.join(run, ".hydra"))
    OmegaConf.save(OmegaConf.create({"foo": 1}),
                   os.path.join(run, ".hydra", "config.yaml"))
    assert load_plot_style_from_run_dir(run) is None


# ---------------------------------------------------------------------------
# _load_paper_cfg
# ---------------------------------------------------------------------------

def test_load_paper_cfg(tmp_path):
    run = str(tmp_path / "run")
    _write_snapshot(run)
    paper, order = rb._load_paper_cfg(run)
    assert paper.get("enabled") is True
    assert paper["fig3"]["bins"] == 30
    assert order == ["grw", "diffusion"]  # scorer-row order = baselines order
    # Missing snapshot → empty/neutral.
    assert rb._load_paper_cfg(str(tmp_path / "none")) == ({}, [])


# ---------------------------------------------------------------------------
# _replot_paper_figures
# ---------------------------------------------------------------------------

def test_replot_paper_figures_emits_both(tmp_path):
    run = str(tmp_path / "run")
    eval_viz = os.path.join(run, "eval_viz")
    _build_eval_viz(eval_viz, with_nll=True)
    _write_snapshot(run)
    ps = load_plot_style_from_run_dir(run)
    paper, order = rb._load_paper_cfg(run)
    written = rb._replot_paper_figures(eval_viz, ps, paper, order)
    assert "paper/fig2_structural_comparison.pdf" in written
    assert "paper/fig3_nll_comparison.pdf" in written
    for w in written:
        assert os.path.getsize(os.path.join(eval_viz, w)) > 0


def test_replot_paper_figures_diffusion_num_timesteps(tmp_path):
    # The diffusion-proxy row rescale (÷ num_timesteps) flows through the new
    # _replot_paper_figures param; Fig 3 must still emit.
    run = str(tmp_path / "run")
    eval_viz = os.path.join(run, "eval_viz")
    _build_eval_viz(eval_viz, with_nll=True)
    _write_snapshot(run)
    ps = load_plot_style_from_run_dir(run)
    paper, order = rb._load_paper_cfg(run)
    written = rb._replot_paper_figures(eval_viz, ps, paper, order, 500)
    assert "paper/fig3_nll_comparison.pdf" in written
    assert os.path.getsize(
        os.path.join(eval_viz, "paper", "fig3_nll_comparison.pdf")
    ) > 0


def test_replot_paper_figures_gated_off(tmp_path):
    run = str(tmp_path / "run")
    eval_viz = os.path.join(run, "eval_viz")
    _build_eval_viz(eval_viz, with_nll=True)
    paper = dict(_PAPER_FIGURES)
    paper["enabled"] = False
    _write_snapshot(run, paper=paper)
    ps = load_plot_style_from_run_dir(run)
    pc, order = rb._load_paper_cfg(run)
    written = rb._replot_paper_figures(eval_viz, ps, pc, order)
    assert written == []
    # Nothing written → no paper/ dir created.
    assert not os.path.isdir(os.path.join(eval_viz, "paper"))


def test_replot_paper_figures_emits_fig1(tmp_path):
    # Fig 1 regenerates from its own tiny per-panel sidecar (paper/fig1_*.npz);
    # no comparison sidecars required.
    from graph_signal_diffusion.evaluation.plot_data_io import (
        dump_paper_fig1_column,
    )
    run = str(tmp_path / "run")
    eval_viz = os.path.join(run, "eval_viz")
    paper_dir = os.path.join(eval_viz, "paper")
    os.makedirs(paper_dir)
    rng = np.random.default_rng(5)
    T_f, n, K, T_hist = 5, 3, 6, 12
    column = {
        "stock_indices": [1, 4, 9],
        "stock_symbols": ["AAA", "BBB", "CCC"],
        "pred": rng.standard_normal((T_f, n)).astype(np.float32) + 100,
        "target": rng.standard_normal((T_f, n)).astype(np.float32) + 100,
        "ensemble": rng.standard_normal((K, T_f, n)).astype(np.float32) + 100,
        "hist": rng.standard_normal((T_hist, n)).astype(np.float32) + 100,
        "time_start": 208,
        "date_labels": list(range(208, 208 + T_hist + T_f)),
    }
    dump_paper_fig1_column(
        os.path.join(paper_dir, "fig1_forecast_column_test_w00_t208.npz"),
        column=column,
    )
    _write_snapshot(run)
    ps = load_plot_style_from_run_dir(run)
    pc, order = rb._load_paper_cfg(run)
    written = rb._replot_paper_figures(eval_viz, ps, pc, order)
    assert "paper/fig1_forecast_column_test_w00_t208.pdf" in written
    assert os.path.getsize(
        os.path.join(paper_dir, "fig1_forecast_column_test_w00_t208.pdf")
    ) > 0


def test_replot_paper_figures_fig2_only_when_no_nll(tmp_path):
    run = str(tmp_path / "run")
    eval_viz = os.path.join(run, "eval_viz")
    _build_eval_viz(eval_viz, with_nll=False)
    _write_snapshot(run)
    ps = load_plot_style_from_run_dir(run)
    pc, order = rb._load_paper_cfg(run)
    written = rb._replot_paper_figures(eval_viz, ps, pc, order)
    assert written == ["paper/fig2_structural_comparison.pdf"]
    assert not os.path.exists(
        os.path.join(eval_viz, "paper", "fig3_nll_comparison.pdf")
    )


# ---------------------------------------------------------------------------
# End-to-end main() smoke (guards the wiring in main())
# ---------------------------------------------------------------------------

def test_main_emits_paper_figures(tmp_path, monkeypatch):
    run = str(tmp_path / "run")
    eval_viz = os.path.join(run, "eval_viz")
    _build_eval_viz(eval_viz, with_nll=True)
    _write_snapshot(run)
    monkeypatch.setattr(sys, "argv", ["replot_baselines", run])
    assert rb.main() == 0
    assert os.path.exists(
        os.path.join(eval_viz, "paper", "fig2_structural_comparison.pdf")
    )
    assert os.path.exists(
        os.path.join(eval_viz, "paper", "fig3_nll_comparison.pdf")
    )
