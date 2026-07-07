"""Unit tests for the paper-ready Fig 1 forecast-column plotters.

Covers the two pure/static helpers that build SP500 Fig 1:
  * ``_select_window_column`` — slices one window + chosen stocks into a
    plain-numpy 'column' dict.
  * ``_plot_paper_forecast_column`` — renders a single-column (n stock rows,
    shared x) paper figure.

These run without a model or destandardization: synthetic ``[B,T,N,F]``
tensors are sliced and plotted, so the tests are fast and deterministic.
"""
import os
import torch
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for tests

from graph_signal_diffusion.tasks.stock_price_forecasting.evaluator import (
    StockPriceForecastingTaskV2,
)

# A valid plot_style dict (global rcParams + per-figure block) so the tests
# exercise the self-styling path: rc_context(global) + plots.stock_prices.
_STYLE = {
    "global": {"rc_params": {"axes.facecolor": "#EAEAF2", "axes.grid": True,
                             "font.family": "serif"}},
    "plots": {"stock_prices": {"panel_width": 2.4, "panel_aspect": 1.0,
                               "dpi": 100, "colors": {"target": "tab:green"},
                               "fonts": {"title": 7, "label": 6}}},
}


def _toy_prices(B=2, T_f=5, N=10, K=4, T_hist=8):
    """Positive-valued synthetic price tensors mimicking the metadata layout."""
    pred = torch.randn(B, T_f, N, 1).abs() + 1.0          # [B, T_f, N, 1]
    target = torch.randn(B, T_f, N, 1).abs() + 1.0        # [B, T_f, N, 1]
    ensemble = torch.randn(B, K, T_f, N, 1).abs() + 1.0   # [B, K, T_f, N, 1]
    hist = torch.randn(B, T_hist, N, 1).abs() + 1.0       # [B, T_hist, N, 1]
    return pred, target, ensemble, hist


def test_select_window_column_shapes_symbols_and_dates():
    pred, target, ens, hist = _toy_prices(B=2, T_f=5, N=10, K=4, T_hist=8)
    col = StockPriceForecastingTaskV2._select_window_column(
        window_idx=1,
        stock_idx=[0, 3, 7],
        pred_prices=pred,
        target_prices=target,
        gen_ensemble_prices=ens,
        hist_prices=hist,
        stock_symbols=[f"S{i}" for i in range(10)],
        time_start=100,
    )
    # Arrays are sliced to the 3 chosen stocks.
    assert col["pred"].shape == (5, 3)
    assert col["target"].shape == (5, 3)
    assert col["ensemble"].shape == (4, 5, 3)
    assert col["hist"].shape == (8, 3)
    # Symbols follow the requested stock indices.
    assert col["stock_symbols"] == ["S0", "S3", "S7"]
    assert col["stock_indices"] == [0, 3, 7]
    # Date labels span history + forecast, starting at time_start.
    assert col["date_labels"][0] == 100
    assert len(col["date_labels"]) == 8 + 5


def test_select_window_column_missing_optionals():
    pred, target, _, _ = _toy_prices(B=1, T_f=4, N=5)
    col = StockPriceForecastingTaskV2._select_window_column(
        window_idx=0,
        stock_idx=[2],
        pred_prices=pred,
        target_prices=target,
        gen_ensemble_prices=None,   # no ensemble
        hist_prices=None,           # no history
        stock_symbols=None,         # fall back to "Stock {i}"
        time_start=None,            # integer x-axis (no date labels)
    )
    assert col["ensemble"] is None
    assert col["hist"] is None
    assert col["date_labels"] is None
    assert col["stock_symbols"] == ["Stock 2"]


def test_plot_paper_forecast_column_writes_pdf(tmp_path):
    pred, target, ens, hist = _toy_prices(B=1, T_f=5, N=6, K=4, T_hist=8)
    col = StockPriceForecastingTaskV2._select_window_column(
        window_idx=0,
        stock_idx=[0, 1, 2],
        pred_prices=pred,
        target_prices=target,
        gen_ensemble_prices=ens,
        hist_prices=hist,
        stock_symbols=None,
        time_start=50,
    )
    out = str(tmp_path / "col.pdf")
    ret = StockPriceForecastingTaskV2._plot_paper_forecast_column(col, out, _STYLE)
    assert ret == out
    assert os.path.exists(out) and os.path.getsize(out) > 0


def test_plot_paper_forecast_column_degenerate(tmp_path):
    """Single stock, no ensemble, no history, integer x-axis must still plot."""
    pred = torch.randn(1, 4, 5, 1).abs() + 1.0
    target = torch.randn(1, 4, 5, 1).abs() + 1.0
    col = StockPriceForecastingTaskV2._select_window_column(
        window_idx=0,
        stock_idx=[2],
        pred_prices=pred,
        target_prices=target,
        gen_ensemble_prices=None,
        hist_prices=None,
        stock_symbols=None,
        time_start=None,
    )
    out = str(tmp_path / "deg.pdf")
    StockPriceForecastingTaskV2._plot_paper_forecast_column(col, out, _STYLE)
    assert os.path.exists(out) and os.path.getsize(out) > 0


def test_plot_paper_forecast_column_clamps_stocks_when_few(tmp_path):
    """Asking for more stocks than exist yields a column over only what exists."""
    pred, target, ens, hist = _toy_prices(B=1, T_f=5, N=2, K=3, T_hist=6)
    # Only 2 stocks available; request both.
    col = StockPriceForecastingTaskV2._select_window_column(
        window_idx=0,
        stock_idx=[0, 1],
        pred_prices=pred,
        target_prices=target,
        gen_ensemble_prices=ens,
        hist_prices=hist,
        stock_symbols=["AAA", "BBB"],
        time_start=0,
    )
    assert col["pred"].shape == (5, 2)
    out = str(tmp_path / "few.pdf")
    StockPriceForecastingTaskV2._plot_paper_forecast_column(col, out, _STYLE)
    assert os.path.exists(out) and os.path.getsize(out) > 0
