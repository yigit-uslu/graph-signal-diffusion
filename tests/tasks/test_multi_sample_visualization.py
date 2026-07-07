"""
Test multi-sample prediction visualization with price conversion.

visualize_predictions is metadata-only; metadata is built via the shared
make_viz_metadata fixture, which drives the real evaluator pipeline.
"""
import os

import torch
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for tests
import matplotlib.pyplot as plt

from graph_signal_diffusion.tasks.stock_price_forecasting.evaluator import StockPriceForecastingTask
from tests.tasks._stock_viz_fixtures import make_viz_metadata


def test_multi_sample_visualization(tmp_path):
    """evaluate_samples runs the full multi-sample pipeline incl. price-space viz."""
    B, n, T, N, F = 2, 8, 20, 30, 1  # 8 samples per input
    task = StockPriceForecastingTask()
    torch.manual_seed(42)

    # Ground-truth raw log returns and an ensemble of predictions (flattened B*n).
    real_log_returns = 0.0002 + 0.015 * torch.randn(B, T, N, F)
    ensemble = torch.stack(
        [0.0002 + 0.015 * torch.randn(B, T, N, F) for _ in range(n)], dim=1
    )  # (B, n, T, N, F)
    generated_samples = ensemble.reshape(B * n, T, N, F)

    # History + last prices for price conversion.
    T_hist = 10
    hist_log_returns = 0.0002 + 0.015 * torch.randn(B, T_hist, N, F)
    initial_prices = torch.full((B, N, 1), 100.0)
    hist_prices = torch.zeros(B, T_hist, N, 1)
    current_price = initial_prices.clone()
    for t in range(T_hist):
        current_price = current_price * torch.exp(hist_log_returns[:, t, :, :])
        hist_prices[:, t, :, :] = current_price
    last_prices = hist_prices[:, -1, :, :]

    metadata = {
        'n_samples_per_input': n,
        'hist_log_returns': hist_log_returns,
        'hist_prices': hist_prices,
        'last_prices': last_prices,
    }

    out_dir = str(tmp_path / "multi_sample_viz")
    os.makedirs(out_dir, exist_ok=True)

    metrics = task.evaluate_samples(
        generated_samples=generated_samples,
        real_samples=real_log_returns,
        metadata=metadata,
        viz_save_dir=out_dir,
    )

    # Primary (price) metrics computed, ensemble metrics present (n>1).
    assert 'price_rmse' in metrics
    assert 'sample_diversity' in metrics
    assert 'crps_mean' in metrics

    # evaluate_samples should have populated the viz metadata.
    assert 'gen_ensemble_prices' in metadata
    assert 'gen_ensemble_log_returns' in metadata
    assert 'target_prices' in metadata

    # Visualization files were written.
    saved = [f for f in os.listdir(out_dir) if f.endswith(('.png', '.pdf'))]
    assert len(saved) > 0, "expected at least one visualization file"


def test_single_vs_multi_comparison(tmp_path):
    """visualize_predictions handles both single (n=1) and ensemble (n>1) metadata."""
    task = StockPriceForecastingTask()
    out_dir = tmp_path / "single_vs_multi"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Single/deterministic forecast (n=1).
    metadata_single = make_viz_metadata(B=1, T=15, N=20, n=1, with_history=True, seed=42)
    fig1 = task.visualize_predictions(
        metadata=metadata_single,
        stocks=[0, 5],
        batch_index=0,
        figsize=(12, 8),
        plot_cumulative=True,
    )
    assert fig1 is not None
    fig1.savefig(str(out_dir / "single_sample.pdf"), dpi=150, bbox_inches='tight')
    plt.close(fig1)

    # Multi-sample ensemble (n=5) with confidence bands.
    metadata_multi = make_viz_metadata(B=1, T=15, N=20, n=5, with_history=True, seed=42)
    fig2 = task.visualize_predictions(
        metadata=metadata_multi,
        stocks=[0, 5],
        batch_index=0,
        figsize=(12, 8),
        plot_cumulative=True,
        show_confidence_bands=True,
        confidence_alpha=0.1,
    )
    assert fig2 is not None
    fig2.savefig(str(out_dir / "multi_sample.pdf"), dpi=150, bbox_inches='tight')
    plt.close(fig2)


if __name__ == "__main__":
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        test_multi_sample_visualization(Path(d))
        test_single_vs_multi_comparison(Path(d))
    print("All multi-sample visualization tests passed.")
