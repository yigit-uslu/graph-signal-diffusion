"""Round-trip tests for plot_data_io sidecar dump/load + replot.

These tests verify that the sidecar files written next to polished PDFs
can be loaded and fed back into the plot functions to regenerate
nonzero-sized PDFs.  No real data is needed — synthetic [B, T, N]
return tensors are sufficient because we only check the IO + plot
plumbing, not the statistical content.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from graph_signal_diffusion.evaluation import structural_metrics as sm
from graph_signal_diffusion.evaluation.plot_data_io import (
    dump_comparison_nll,
    dump_comparison_structural,
    dump_per_baseline_nll,
    dump_per_baseline_structural,
    dump_stock_prices_window,
    load_comparison_nll,
    load_comparison_structural,
    load_per_baseline_nll,
    load_per_baseline_structural,
    load_stock_prices_window,
)


@pytest.fixture
def synth_returns():
    """Tiny synthetic dataset: B=4 windows, T=5 steps, N=8 stocks."""
    torch.manual_seed(0)
    cat_real = torch.randn(4, 5, 8)
    cat_gen = torch.randn(4 * 3, 5, 8)  # n_samples=3 per window
    return cat_real, cat_gen


def _nonzero_pdf(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 1024


# ---------------------------------------------------------------------------
# Per-baseline structural round-trip
# ---------------------------------------------------------------------------

def test_per_baseline_structural_roundtrip(tmp_path, synth_returns):
    cat_real, cat_gen = synth_returns
    npz = str(tmp_path / "structural_data.npz")

    cum_std_real = sm.compute_cumulative_return_std(cat_real).numpy()
    cum_std_gen = sm.compute_cumulative_return_std(cat_gen).numpy()
    cum_std_real_tn = sm.compute_cumulative_return_std(cat_real, per_stock=True).numpy()
    cum_std_gen_tn = sm.compute_cumulative_return_std(cat_gen, per_stock=True).numpy()
    theoretical = np.sqrt(np.arange(1, 6, dtype=np.float32))

    dump_per_baseline_structural(
        npz,
        cat_real=cat_real,
        cat_gen=cat_gen,
        model_name="GRW",
        max_lag=2,
        n_eigenvalues=3,
        window_stride=5,
        window_start_indices=None,
        cum_std_real=cum_std_real,
        cum_std_gen=cum_std_gen,
        cum_std_real_tn=cum_std_real_tn,
        cum_std_gen_tn=cum_std_gen_tn,
        theoretical=theoretical,
        dpi=72,
    )
    assert os.path.exists(npz)

    data = load_per_baseline_structural(npz)
    assert data["model_name"] == "GRW"
    assert data["max_lag"] == 2
    assert data["n_eigenvalues"] == 3
    assert data["window_start_indices"] is None
    np.testing.assert_allclose(data["cat_real"], cat_real.numpy())
    np.testing.assert_allclose(data["cat_gen"], cat_gen.numpy())
    np.testing.assert_allclose(data["cum_std_real"], cum_std_real)

    # Replot via the same code path the CLI uses.
    from graph_signal_diffusion.cli import replot_baselines

    written = replot_baselines._replot_per_baseline_structural(npz)
    assert "structural_metrics_grw.pdf" in written
    assert "autocorr_per_stock_grw.pdf" in written
    assert "autocorr_violin_grw.pdf" in written
    assert "autocorr_heatmap_grw.pdf" in written
    assert "cumulative_return_std_grw.pdf" in written
    for fname in written:
        assert _nonzero_pdf(str(tmp_path / fname)), f"missing/empty: {fname}"


# ---------------------------------------------------------------------------
# Comparison structural round-trip with the reference-link trick
# ---------------------------------------------------------------------------

def test_comparison_structural_roundtrip_with_references(tmp_path, synth_returns):
    cat_real, cat_gen = synth_returns
    # Build a fake eval_viz layout: comparison/ + grw/test/ + diffusion/test/
    eval_viz = tmp_path / "eval_viz"
    grw_dir = eval_viz / "grw" / "test"
    diff_dir = eval_viz / "diffusion" / "test"
    comp_dir = eval_viz / "comparison"
    for d in (grw_dir, diff_dir, comp_dir):
        d.mkdir(parents=True)

    grw_npz = str(grw_dir / "structural_data.npz")
    diff_npz = str(diff_dir / "structural_data.npz")
    comp_npz = str(comp_dir / "structural_data.npz")

    cum_std_real = sm.compute_cumulative_return_std(cat_real).numpy()
    cum_std_gen = sm.compute_cumulative_return_std(cat_gen).numpy()
    cum_std_real_tn = sm.compute_cumulative_return_std(cat_real, per_stock=True).numpy()
    cum_std_gen_tn = sm.compute_cumulative_return_std(cat_gen, per_stock=True).numpy()

    for path, name in [(grw_npz, "GRW"), (diff_npz, "Diffusion")]:
        dump_per_baseline_structural(
            path,
            cat_real=cat_real,
            cat_gen=cat_gen,
            model_name=name,
            max_lag=2,
            n_eigenvalues=3,
            window_stride=5,
            window_start_indices=None,
            cum_std_real=cum_std_real,
            cum_std_gen=cum_std_gen,
            cum_std_real_tn=cum_std_real_tn,
            cum_std_gen_tn=cum_std_gen_tn,
            theoretical=None,
            dpi=72,
        )

    dump_comparison_structural(
        comp_npz,
        baseline_names=["grw", "diffusion"],
        per_baseline_paths={"grw": grw_npz, "diffusion": diff_npz},
        max_lag=2,
        n_eigenvalues=3,
        cum_std_real=cum_std_real,
        cum_std_real_per_stock=cum_std_real_tn,
        cum_std_gen_dict={"grw": cum_std_gen, "diffusion": cum_std_gen},
        cum_std_gen_per_stock_dict={"grw": cum_std_gen_tn, "diffusion": cum_std_gen_tn},
        theoretical=None,
        baseline_colors={"grw": "tab:orange", "diffusion": "tab:green"},
    )

    # Reference resolution: heavy tensors loaded from per-baseline files.
    loaded = load_comparison_structural(comp_npz)
    assert loaded["baseline_names"] == ["grw", "diffusion"]
    assert loaded["cat_real"] is not None
    np.testing.assert_allclose(loaded["cat_real"], cat_real.numpy())
    assert set(loaded["cat_gen_dict"].keys()) == {"grw", "diffusion"}
    np.testing.assert_allclose(loaded["cat_gen_dict"]["grw"], cat_gen.numpy())

    # Sanity-check that a missing per-baseline file raises clearly.
    os.remove(grw_npz)
    with pytest.raises(FileNotFoundError):
        load_comparison_structural(comp_npz)


# ---------------------------------------------------------------------------
# Per-baseline NLL round-trip
# ---------------------------------------------------------------------------

def test_per_baseline_nll_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    nll_real = np.abs(rng.standard_normal(50)) + 1.0
    nll_gen = np.abs(rng.standard_normal(50)) + 1.5
    cum_real = np.abs(rng.standard_normal((50, 5)))
    cum_gen = np.abs(rng.standard_normal((50, 5)))

    npz = str(tmp_path / "nll_data.npz")
    dump_per_baseline_nll(
        npz,
        nll_real=nll_real,
        nll_gen=nll_gen,
        cum_nll_real_bt=cum_real,
        cum_nll_gen_bt=cum_gen,
        bins=20,
        dpi=72,
        log_x=True,
        scoring_model_name="GRW",
    )
    data = load_per_baseline_nll(npz)
    assert data["scoring_model_name"] == "GRW"
    np.testing.assert_allclose(data["nll_real"], nll_real)
    np.testing.assert_allclose(data["cum_nll_gen_bt"], cum_gen)

    from graph_signal_diffusion.cli import replot_baselines

    written = replot_baselines._replot_per_baseline_nll(npz)
    assert "nll_histogram_grw.pdf" in written
    assert "cumulative_nll_histogram_grw.pdf" in written
    for fname in written:
        assert _nonzero_pdf(str(tmp_path / fname)), f"missing/empty: {fname}"


# ---------------------------------------------------------------------------
# Comparison NLL round-trip
# ---------------------------------------------------------------------------

def test_comparison_nll_roundtrip(tmp_path):
    rng = np.random.default_rng(1)
    nll_real = np.abs(rng.standard_normal(60)) + 1.0
    nll_gen_dict = {
        "grw": np.abs(rng.standard_normal(60)) + 1.2,
        "diffusion": np.abs(rng.standard_normal(60)) + 0.9,
    }
    cum_nll_real = np.abs(rng.standard_normal((60, 5)))
    cum_nll_gen_dict = {
        "grw": np.abs(rng.standard_normal((60, 5))),
        "diffusion": np.abs(rng.standard_normal((60, 5))),
    }
    npz = str(tmp_path / "nll_data__GRW.npz")
    dump_comparison_nll(
        npz,
        nll_real=nll_real,
        nll_gen_dict=nll_gen_dict,
        cum_nll_real=cum_nll_real,
        cum_nll_gen_dict=cum_nll_gen_dict,
        scoring_model_name="GRW",
        bins=20,
        log_x=True,
        baseline_colors={"grw": "tab:orange", "diffusion": "tab:green"},
    )
    data = load_comparison_nll(npz)
    assert data["scoring_model_name"] == "GRW"
    assert set(data["nll_gen_dict"].keys()) == {"grw", "diffusion"}
    assert set(data["cum_nll_gen_dict"].keys()) == {"grw", "diffusion"}

    from graph_signal_diffusion.cli import replot_baselines

    written = replot_baselines._replot_comparison_nll(npz)
    assert "nll_histogram_GRW_scored.pdf" in written
    assert "cumulative_nll_histogram_GRW_scored.pdf" in written
    for fname in written:
        assert _nonzero_pdf(str(tmp_path / fname))


# ---------------------------------------------------------------------------
# Per-window stock_prices round-trip
# ---------------------------------------------------------------------------

def test_stock_prices_window_roundtrip(tmp_path):
    rng = np.random.default_rng(2)
    pred_mean = rng.standard_normal((5, 4, 1)).astype(np.float32) + 100.0
    target_mean = rng.standard_normal((5, 4, 1)).astype(np.float32) + 100.0
    gen_ensemble = rng.standard_normal((6, 5, 4, 1)).astype(np.float32) + 100.0
    hist_samples = rng.standard_normal((20, 4, 1)).astype(np.float32) + 100.0

    sidecar_dir = tmp_path / "stock_prices"
    sidecar_dir.mkdir()
    npz = str(sidecar_dir / "t100_b0.npz")
    dump_stock_prices_window(
        npz,
        pred_mean=pred_mean,
        target_mean=target_mean,
        gen_ensemble=gen_ensemble,
        hist_samples=hist_samples,
        stock_indices=[0, 1, 2, 3],
        stock_symbols=["A", "B", "C", "D"],
        batch_idx=0,
        time_start=100,
    )
    data = load_stock_prices_window(npz)
    assert data["batch_idx"] == 0
    assert data["time_start"] == 100
    assert data["stock_indices"] == [0, 1, 2, 3]
    assert data["stock_symbols"] == ["A", "B", "C", "D"]
    np.testing.assert_allclose(data["pred_mean"], pred_mean)

    from graph_signal_diffusion.cli import replot_baselines

    written = replot_baselines._replot_stock_prices_window(npz)
    assert written == ["stock_prices_t100_b0.pdf"]
    pdf_path = str(tmp_path / "stock_prices_t100_b0.pdf")
    assert _nonzero_pdf(pdf_path)


# ---------------------------------------------------------------------------
# Paper Fig 1 forecast-column round-trip (the tiny already-sliced panel)
# ---------------------------------------------------------------------------

def test_paper_fig1_column_roundtrip(tmp_path):
    from graph_signal_diffusion.evaluation.plot_data_io import (
        dump_paper_fig1_column,
        load_paper_fig1_column,
    )
    rng = np.random.default_rng(3)
    T_f, n, K, T_hist = 5, 3, 6, 20
    column = {
        "stock_indices": [2, 5, 7],
        "stock_symbols": ["AAA", "BBB", "CCC"],
        "pred": rng.standard_normal((T_f, n)).astype(np.float32) + 100.0,
        "target": rng.standard_normal((T_f, n)).astype(np.float32) + 100.0,
        "ensemble": rng.standard_normal((K, T_f, n)).astype(np.float32) + 100.0,
        "hist": rng.standard_normal((T_hist, n)).astype(np.float32) + 100.0,
        "time_start": 208,
        "date_labels": list(range(208, 208 + T_hist + T_f)),
    }
    npz = str(tmp_path / "fig1_forecast_column_test_w00_t208.npz")
    dump_paper_fig1_column(npz, column=column)

    out = load_paper_fig1_column(npz)
    assert out["stock_indices"] == [2, 5, 7]
    assert out["stock_symbols"] == ["AAA", "BBB", "CCC"]
    assert out["time_start"] == 208
    assert len(out["date_labels"]) == T_hist + T_f and out["date_labels"][0] == 208
    np.testing.assert_allclose(out["pred"], column["pred"])
    np.testing.assert_allclose(out["ensemble"], column["ensemble"])
    np.testing.assert_allclose(out["hist"], column["hist"])

    # Render via the exact staticmethod the replot CLI calls.
    from graph_signal_diffusion.tasks.stock_price_forecasting.evaluator import (
        StockPriceForecastingTaskV2,
    )
    pdf = str(tmp_path / "fig1_forecast_column_test_w00_t208.pdf")
    StockPriceForecastingTaskV2._plot_paper_forecast_column(out, pdf, None)
    assert _nonzero_pdf(pdf)


def test_paper_fig1_column_roundtrip_none_ensemble_hist(tmp_path):
    # None ensemble + None hist + None symbols/time_start must round-trip to
    # None (the none-sentinel must not leak through as a string array).
    from graph_signal_diffusion.evaluation.plot_data_io import (
        dump_paper_fig1_column,
        load_paper_fig1_column,
    )
    rng = np.random.default_rng(4)
    column = {
        "stock_indices": [0],
        "stock_symbols": None,
        "pred": rng.standard_normal((4, 1)).astype(np.float32),
        "target": rng.standard_normal((4, 1)).astype(np.float32),
        "ensemble": None,
        "hist": None,
        "time_start": None,
        "date_labels": None,
    }
    npz = str(tmp_path / "fig1_b0.npz")
    dump_paper_fig1_column(npz, column=column)
    out = load_paper_fig1_column(npz)
    assert out["ensemble"] is None
    assert out["hist"] is None
    assert out["stock_symbols"] is None
    assert out["time_start"] is None
    assert out["date_labels"] is None
