"""Unit tests for StockPriceForecastingTask visualization helper.

visualize_predictions is metadata-only: predictions/targets/history arrive inside
the metadata dict (see tests/tasks/_stock_viz_fixtures.make_viz_metadata).
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for tests

from graph_signal_diffusion.tasks.stock_price_forecasting.evaluator import (
    StockPriceForecastingTask,
)
from tests.tasks._stock_viz_fixtures import make_viz_metadata


# Create output directory for test visualizations
OUTPUT_DIR = Path(__file__).parent / "stock_forecasting_viz"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


def test_visualize_predictions_saves_file(tmp_path: Path):
    task = StockPriceForecastingTask()
    metadata = make_viz_metadata(B=1, T=12, N=6, n=5)

    out_path = OUTPUT_DIR / "stock_viz_test.pdf"
    fig = task.visualize_predictions(
        metadata=metadata,
        stocks=None,
        batch_index=0,
        n_examples=3,
        save_dir=str(out_path.parent),
        plot_cumulative=True,
    )
    fig.savefig(str(out_path))

    assert fig is not None
    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_visualize_predictions_with_explicit_stocks(tmp_path: Path):
    """visualize_predictions honours an explicit stock subset via the metadata API."""
    task = StockPriceForecastingTask()
    metadata = make_viz_metadata(B=1, T=8, N=5, n=5)

    out_path = OUTPUT_DIR / "stock_viz_flat_test.pdf"
    fig = task.visualize_predictions(
        metadata=metadata,
        stocks=[0, 2],
        batch_index=0,
        save_dir=str(out_path.parent),
    )
    fig.savefig(str(out_path))

    assert fig is not None
    assert out_path.exists()
    assert out_path.stat().st_size > 0


if __name__ == "__main__":
    test_visualize_predictions_saves_file(tmp_path=Path('.'))
    test_visualize_predictions_with_explicit_stocks(tmp_path=Path('.'))
    print("All stock price forecasting viz tests passed.")
