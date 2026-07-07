"""Shared fixtures for ``StockPriceForecastingTask.visualize_predictions`` tests.

``visualize_predictions`` was refactored to a **metadata-only** signature: it no
longer takes ``pred=``/``target=``; predictions/targets/history all arrive inside
the ``metadata`` dict (``gen_ensemble_log_returns``, ``gen_ensemble_prices``,
``target_prices``, ``last_prices``, ``hist_prices``, ``hist_log_returns``, ...).

Rather than hand-build every key (and drift from the real contract), these helpers
populate ``metadata`` by driving the actual evaluator pipeline
(``_process_ensemble`` -> ``_convert_to_prices``). That is exactly what
``evaluate_samples`` runs before visualization, so the fixtures stay faithful to
production metadata by construction.
"""
import torch

from graph_signal_diffusion.tasks.stock_price_forecasting.evaluator import (
    StockPriceForecastingTask,
)


def make_viz_metadata(
    B: int = 2,
    T: int = 10,
    N: int = 15,
    F: int = 1,
    n: int = 5,
    with_history: bool = True,
    T_hist: int = 20,
    seed: int = 0,
):
    """Build fully-populated metadata for ``visualize_predictions``.

    Parameters mirror the shapes the evaluator expects. ``n`` is the ensemble
    size (use ``n=1`` for a single/deterministic forecast, ``n>1`` for an
    ensemble). When ``with_history`` is True, ``hist_prices``
    ([B, T_hist, N, 1]) and ``hist_log_returns`` ([B, T_hist, N, F]) are added.

    Returns the metadata dict after the real pipeline has populated
    ``gen_ensemble``, ``gen_ensemble_log_returns``, ``gen_ensemble_prices``,
    ``target_prices`` and ``pred_prices``.
    """
    torch.manual_seed(seed)
    task = StockPriceForecastingTask()

    # Ensemble generated log returns, in the flattened [B*n, T, N, F] layout that
    # _process_ensemble expects; and ground-truth log returns [B, T, N, F].
    gen = torch.randn(B * n, T, N, F) * 0.02
    target_log_returns = torch.randn(B, T, N, F) * 0.02
    last_prices = torch.rand(B, N, 1) * 200 + 50  # realistic $50-$250

    metadata = {
        "batch_size": B,
        "num_stocks": N,
        "num_timesteps": T,
        "num_features": F,
        "n_samples_per_input": n,
        "last_prices": last_prices,
        "timestamp": torch.zeros(B, dtype=torch.long),
    }
    if with_history:
        metadata["hist_log_returns"] = torch.randn(B, T_hist, N, F) * 0.02
        metadata["hist_prices"] = torch.rand(B, T_hist, N, 1) * 200 + 50

    # Drive the real pipeline: _process_ensemble sets gen_ensemble + pred_log_returns;
    # _convert_to_prices sets target_prices, gen_ensemble_log_returns,
    # gen_ensemble_prices and pred_prices.
    task._process_ensemble(gen, metadata)
    task._convert_to_prices(target_log_returns, metadata)
    return metadata
