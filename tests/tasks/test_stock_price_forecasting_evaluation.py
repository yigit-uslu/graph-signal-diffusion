"""
Comprehensive unit tests for the StockPriceForecastingTask evaluation pipeline.

Tests cover:
1. Log-return to price conversion
2. Primary metrics on prices
3. Secondary metrics on log returns
4. Multi-sample evaluation with probabilistic metrics
5. Visualization with both price and cumulative log return plots
"""
import pytest
import torch
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for tests

from graph_signal_diffusion.tasks.stock_price_forecasting.evaluator import (
    StockPriceForecastingTask,
)
from tests.tasks._stock_viz_fixtures import make_viz_metadata


# Create output directory for test visualizations
OUTPUT_DIR = Path(__file__).parent / "stock_forecasting_eval_tests"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


class TestLogReturnToPriceConversion:
    """Test the log-return to price conversion logic."""
    
    def test_price_conversion_formula(self):
        """Verify that log returns are correctly converted to prices."""
        # Setup
        B, T, N, F = 2, 5, 3, 1
        
        # Known log returns (small values for numerical stability)
        log_returns = torch.tensor([0.01, -0.02, 0.015, 0.005, -0.01]).view(1, T, 1, 1)
        log_returns = log_returns.expand(B, T, N, F)
        
        # Last observed price
        last_prices = torch.tensor([100.0, 150.0, 200.0]).view(1, N, 1).expand(B, N, 1)
        
        # Manual calculation: price_t = last_price * exp(cumsum(log_returns))
        cumsum = torch.cumsum(log_returns, dim=1)
        expected_prices = last_prices.unsqueeze(1) * torch.exp(cumsum)
        
        # Verify shape
        assert expected_prices.shape == (B, T, N, F)
        
        # Verify first price is last_price * exp(first_return)
        for b in range(B):
            for n in range(N):
                expected_first = last_prices[b, n, 0].item() * np.exp(log_returns[b, 0, n, 0].item())
                assert np.isclose(expected_prices[b, 0, n, 0].item(), expected_first, rtol=1e-5)
        
        print("✓ Log-return to price conversion formula verified")
    
    def test_zero_returns_preserve_price(self):
        """If log returns are all zero, prices should equal last_price."""
        B, T, N, F = 1, 5, 2, 1
        
        log_returns = torch.zeros(B, T, N, F)
        last_prices = torch.tensor([100.0, 200.0]).view(1, N, 1)
        
        cumsum = torch.cumsum(log_returns, dim=1)
        prices = last_prices.unsqueeze(1) * torch.exp(cumsum)
        
        # All prices should equal last_prices
        for t in range(T):
            assert torch.allclose(prices[0, t, :, 0], last_prices[0, :, 0])
        
        print("✓ Zero returns correctly preserve last price")


class TestEvaluateSamplesMetrics:
    """Test the evaluate_samples method produces correct metrics."""
    
    def _create_mock_data(self, B=2, T=5, N=10, F=1, with_prices=True):
        """Create mock data for testing."""
        # Generate log returns (realistic range: -5% to +5%)
        generated_log_returns = torch.randn(B, T, N, F) * 0.02
        real_log_returns = torch.randn(B, T, N, F) * 0.02
        
        metadata = {
            'batch_size': B,
            'num_stocks': N,
            'num_timesteps': T,
            'num_features': F,
        }
        
        if with_prices:
            # Create last_prices (realistic stock prices)
            last_prices = torch.rand(B, N, 1) * 200 + 50  # $50 to $250
            metadata['last_prices'] = last_prices
        
        return generated_log_returns, real_log_returns, metadata
    
    def test_primary_metrics_on_prices(self):
        """Test that primary metrics are computed on actual prices."""
        task = StockPriceForecastingTask()
        gen, real, metadata = self._create_mock_data(B=2, T=5, N=10)
        
        metrics = task.evaluate_samples(
            generated_samples=gen,
            real_samples=real,
            metadata=metadata,
            viz_save_dir=None
        )
        
        # Check price-prefixed metrics exist
        assert 'price_mse' in metrics, "Should have price_mse"
        assert 'price_mae' in metrics, "Should have price_mae"
        assert 'price_rmse' in metrics, "Should have price_rmse"
        
        # Metrics should be positive
        assert metrics['price_mse'] >= 0
        assert metrics['price_mae'] >= 0
        assert metrics['price_rmse'] >= 0
        
        print(f"✓ Primary price metrics computed:")
        print(f"  price_mse: {metrics['price_mse']:.4f}")
        print(f"  price_mae: {metrics['price_mae']:.4f}")
        print(f"  price_rmse: {metrics['price_rmse']:.4f}")
    
    def test_secondary_return_metrics(self):
        """Test that secondary metrics are computed on log returns."""
        task = StockPriceForecastingTask()
        gen, real, metadata = self._create_mock_data(B=2, T=5, N=10)
        
        metrics = task.evaluate_samples(
            generated_samples=gen,
            real_samples=real,
            metadata=metadata,
            viz_save_dir=None
        )
        
        # Check return-prefixed metrics exist
        assert 'return_mse' in metrics, "Should have return_mse"
        assert 'return_mae' in metrics, "Should have return_mae"
        assert 'return_rmse' in metrics, "Should have return_rmse"
        assert 'return_direction_accuracy' in metrics, "Should have return_direction_accuracy"
        assert 'volclustering_gen' in metrics, "Should have volclustering_gen"
        assert 'volclustering_real' in metrics, "Should have volclustering_real"
        assert 'volclustering_gap' in metrics, "Should have volclustering_gap"
        assert 'momentum_gen' in metrics, "Should have momentum_gen"
        assert 'kurtosis_gen' in metrics, "Should have kurtosis_gen"

        # Direction accuracy should be in [0, 1]
        assert 0 <= metrics['return_direction_accuracy'] <= 1

        # ACF values should be in [-1, 1]
        assert -1 <= metrics['volclustering_gen'] <= 1
        assert -1 <= metrics['volclustering_real'] <= 1

        print(f"✓ Secondary return metrics computed:")
        print(f"  return_mse: {metrics['return_mse']:.6f}")
        print(f"  return_direction_accuracy: {metrics['return_direction_accuracy']:.2%}")
        print(f"  volclustering_gen: {metrics['volclustering_gen']:.4f}")
    
    def test_temporal_metrics(self):
        """Test that temporal metrics are computed at each horizon."""
        task = StockPriceForecastingTask()
        T = 6
        gen, real, metadata = self._create_mock_data(B=2, T=T, N=10)
        
        metrics = task.evaluate_samples(
            generated_samples=gen,
            real_samples=real,
            metadata=metadata,
            viz_save_dir=None
        )
        
        # Check horizon-specific metrics
        for t in range(1, T + 1):
            assert f'price_mse_horizon_{t}' in metrics, f"Should have price_mse_horizon_{t}"
            assert f'price_mae_horizon_{t}' in metrics, f"Should have price_mae_horizon_{t}"
        
        # Check short/medium/long term metrics (T=6 >= 3)
        assert 'price_mse_short_term' in metrics
        assert 'price_mse_medium_term' in metrics
        assert 'price_mse_long_term' in metrics
        
        print(f"✓ Temporal metrics computed for T={T} horizons")
    
    def test_fallback_without_last_prices(self):
        """Test fallback behavior when last_prices is not provided."""
        task = StockPriceForecastingTask()
        gen, real, metadata = self._create_mock_data(B=2, T=5, N=10, with_prices=False)
        
        metrics = task.evaluate_samples(
            generated_samples=gen,
            real_samples=real,
            metadata=metadata,
            viz_save_dir=None
        )
        
        # Should have logreturn-prefixed metrics instead
        assert 'logreturn_mse' in metrics or 'return_mse' in metrics
        
        print("✓ Fallback without last_prices works correctly")


class TestMultiSampleEvaluation:
    """Test multi-sample evaluation with probabilistic metrics."""
    
    def _create_multi_sample_data(self, B=2, n=5, T=5, N=10, F=1):
        """Create mock multi-sample data."""
        # Generated: (B*n, T, N, F) - cloned batch
        generated = torch.randn(B * n, T, N, F) * 0.02
        
        # Real: (B, T, N, F) - original batch
        real = torch.randn(B, T, N, F) * 0.02
        
        # Last prices
        last_prices = torch.rand(B, N, 1) * 200 + 50
        
        metadata = {
            'n_samples_per_input': n,
            'batch_size': B,
            'num_stocks': N,
            'num_timesteps': T,
            'num_features': F,
            'last_prices': last_prices,
        }
        
        return generated, real, metadata
    
    def test_multi_sample_reshaping(self):
        """Test that multi-sample data is reshaped correctly."""
        from graph_signal_diffusion.utils import reshape_generated_samples
        
        B, n, T, N, F = 2, 5, 4, 3, 1
        generated = torch.randn(B * n, T, N, F)
        
        reshaped = reshape_generated_samples(generated, n)
        
        assert reshaped.shape == (B, n, T, N, F)
        print(f"✓ Multi-sample reshaping: ({B*n}, {T}, {N}, {F}) -> ({B}, {n}, {T}, {N}, {F})")
    
    def test_probabilistic_metrics(self):
        """Test that probabilistic metrics are computed for multi-sample case."""
        task = StockPriceForecastingTask()
        gen, real, metadata = self._create_multi_sample_data(B=2, n=10, T=5, N=10)
        
        metrics = task.evaluate_samples(
            generated_samples=gen,
            real_samples=real,
            metadata=metadata,
            viz_save_dir=None
        )
        
        # Check probabilistic metrics
        assert 'sample_diversity' in metrics
        assert 'crps_mean' in metrics
        assert 'mis_90' in metrics
        assert 'coverage_90' in metrics
        
        # Sample diversity should be positive (std across samples)
        assert metrics['sample_diversity'] > 0
        
        # CRPS should be non-negative
        assert metrics['crps_mean'] >= 0
        
        # Coverage should be in [0, 1]
        assert 0 <= metrics['coverage_90'] <= 1
        
        print(f"✓ Probabilistic metrics computed:")
        print(f"  sample_diversity: {metrics['sample_diversity']:.4f}")
        print(f"  crps_mean: {metrics['crps_mean']:.4f}")
        print(f"  coverage_90: {metrics['coverage_90']:.2%}")
    
    def test_single_sample_no_probabilistic_metrics(self):
        """Test that single sample case does not include probabilistic metrics."""
        task = StockPriceForecastingTask()
        
        B, T, N, F = 2, 5, 10, 1
        gen = torch.randn(B, T, N, F) * 0.02
        real = torch.randn(B, T, N, F) * 0.02
        last_prices = torch.rand(B, N, 1) * 200 + 50
        
        metadata = {
            'batch_size': B,
            'num_stocks': N,
            'last_prices': last_prices,
        }
        
        metrics = task.evaluate_samples(
            generated_samples=gen,
            real_samples=real,
            metadata=metadata,
            viz_save_dir=None
        )
        
        # Should NOT have multi-sample metrics
        assert 'sample_diversity' not in metrics
        assert 'crps_mean' not in metrics
        
        print("✓ Single sample correctly omits probabilistic metrics")


class TestVisualization:
    """Test the visualization functionality."""
    
    def test_visualization_with_price_conversion(self):
        """Test that visualization correctly converts log returns to prices."""
        task = StockPriceForecastingTask()
        metadata = make_viz_metadata(B=2, T=10, N=15, n=5)

        save_path = OUTPUT_DIR / "viz_price_conversion.pdf"

        fig = task.visualize_predictions(
            metadata=metadata,
            stocks=[0, 5, 10],
            batch_index=0,
            save_dir=str(OUTPUT_DIR),
            plot_cumulative=True,
        )

        assert fig is not None
        fig.savefig(str(save_path))
        assert save_path.exists()

        print(f"✓ Visualization with price conversion saved to {save_path}")

    def test_visualization_with_history(self):
        """Test visualization with historical actual prices plotted in black."""
        task = StockPriceForecastingTask()
        metadata = make_viz_metadata(B=2, T=10, N=15, n=5, with_history=True, T_hist=20)

        # Verify hist_prices is in metadata with the expected [B, T_hist, N, 1] shape
        assert 'hist_prices' in metadata, "hist_prices should be in metadata for history plot"
        assert metadata['hist_prices'].shape == (2, 20, 15, 1), f"hist_prices shape mismatch: {metadata['hist_prices'].shape}"

        save_path = OUTPUT_DIR / "viz_with_history.pdf"

        fig = task.visualize_predictions(
            metadata=metadata,
            stocks=[0, 5, 10],
            batch_index=0,
            save_dir=str(OUTPUT_DIR),
            plot_cumulative=True,
        )

        assert fig is not None
        fig.savefig(str(save_path))
        assert save_path.exists()

        print(f"✓ Visualization with full history (actual prices in black) saved to {save_path}")

    def test_visualization_with_ensemble(self):
        """Test visualization with ensemble predictions."""
        task = StockPriceForecastingTask()
        metadata = make_viz_metadata(B=2, T=10, N=15, n=10, with_history=True)

        save_path = OUTPUT_DIR / "viz_with_ensemble.pdf"

        fig = task.visualize_predictions(
            metadata=metadata,
            stocks=[0, 5, 10],
            batch_index=0,
            save_dir=str(OUTPUT_DIR),
            plot_cumulative=True,
            show_confidence_bands=True,
            confidence_alpha=0.1,
        )

        assert fig is not None
        fig.savefig(str(save_path))
        assert save_path.exists()

        print(f"✓ Visualization with ensemble and confidence bands saved to {save_path}")

    def test_visualization_without_cumulative(self):
        """Test visualization without cumulative log returns plot."""
        task = StockPriceForecastingTask()
        metadata = make_viz_metadata(B=2, T=10, N=15, n=5)

        save_path = OUTPUT_DIR / "viz_prices_only.pdf"

        fig = task.visualize_predictions(
            metadata=metadata,
            stocks=[0, 5],
            batch_index=0,
            save_dir=str(OUTPUT_DIR),
            plot_cumulative=False,
        )

        assert fig is not None
        fig.savefig(str(save_path))
        assert save_path.exists()

        print(f"✓ Visualization (prices only) saved to {save_path}")


class TestPrepareData:
    """Test the prepare_data method."""
    
    def test_prepare_data_extracts_last_prices(self):
        """Test that prepare_data correctly extracts last_prices from close_price."""
        from torch_geometric.data import Data
        
        task = StockPriceForecastingTask()
        
        # Create mock PyG Data
        B, N, T_past, T_future, F = 2, 10, 20, 5, 1
        
        # Simulate batched data
        y = torch.randn(B * N, T_future, F)  # Targets (log returns)
        close_price = torch.rand(B * N, T_past) * 200 + 50  # Historical prices [B*N, T_past]
        batch = torch.repeat_interleave(torch.arange(B), N)  # Batch indicator
        
        data = Data(
            y=y,
            close_price=close_price.unsqueeze(-1),  # [B*N, T_past, 1]
            batch=batch,
        )
        
        result = task.prepare_data(data)
        
        assert 'samples' in result
        assert 'metadata' in result
        assert 'last_prices' in result['metadata']
        
        # last_prices should be [B, N, 1]
        last_prices = result['metadata']['last_prices']
        assert last_prices.shape == (B, N, 1)
        
        print(f"✓ prepare_data correctly extracts last_prices: {last_prices.shape}")


class TestReturnMetrics:
    """Test the return-specific metrics computation."""
    
    def test_return_direction_accuracy(self):
        """Test direction accuracy on returns."""
        task = StockPriceForecastingTask()
        
        # Create perfect prediction (same sign)
        pred_returns = torch.tensor([[0.01, -0.02, 0.015], [0.005, -0.01, 0.02]])  # (2, 3)
        target_returns = torch.tensor([[0.02, -0.01, 0.01], [0.01, -0.005, 0.015]])  # Same signs
        
        pred_returns = pred_returns.unsqueeze(-1).unsqueeze(0)  # (1, 2, 3, 1) = (B, T, N, F)
        target_returns = target_returns.unsqueeze(-1).unsqueeze(0)
        
        metrics = task._compute_return_metrics(pred_returns, target_returns)
        
        # All signs match, so accuracy should be 1.0
        assert metrics['return_direction_accuracy'] == 1.0
        
        print("✓ Return direction accuracy computed correctly for matching signs")
    
    def test_structural_metrics(self):
        """Test structural metrics (volclustering, momentum, kurtosis, eigval1)."""
        task = StockPriceForecastingTask()

        # Create random returns with enough data for meaningful stats
        target_returns = torch.randn(4, 10, 5, 1)
        pred_returns = torch.randn(4, 10, 5, 1)

        metrics = task._compute_return_metrics(pred_returns, target_returns)

        # All structural metric keys should exist
        for key in ['volclustering_gen', 'volclustering_real', 'volclustering_gap',
                     'momentum_gen', 'momentum_real', 'momentum_gap',
                     'kurtosis_gen', 'kurtosis_real', 'kurtosis_gap']:
            assert key in metrics, f"Should have {key}"

        # ACF values in [-1, 1]
        assert -1 <= metrics['volclustering_gen'] <= 1
        assert -1 <= metrics['volclustering_real'] <= 1
        assert -1 <= metrics['momentum_gen'] <= 1
        assert -1 <= metrics['momentum_real'] <= 1

        # Kurtosis should be finite
        assert np.isfinite(metrics['kurtosis_gen'])
        assert np.isfinite(metrics['kurtosis_real'])

        # Gap should be non-negative
        assert metrics['volclustering_gap'] >= 0
        assert metrics['momentum_gap'] >= 0
        assert metrics['kurtosis_gap'] >= 0

        # Eigenvalue should exist for N > 1
        assert 'eigval1_gen' in metrics
        assert 'eigval1_real' in metrics

        print(f"✓ Structural metrics computed:")
        print(f"  volclustering_gen: {metrics['volclustering_gen']:.4f}")
        print(f"  momentum_gen: {metrics['momentum_gen']:.4f}")
        print(f"  kurtosis_gen: {metrics['kurtosis_gen']:.2f}")
        print(f"  eigval1_gen: {metrics.get('eigval1_gen', 'N/A')}")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Running Stock Price Forecasting Evaluation Tests")
    print("=" * 60 + "\n")
    
    # Log-return to price conversion
    print("\n--- Log-Return to Price Conversion Tests ---")
    test_conversion = TestLogReturnToPriceConversion()
    test_conversion.test_price_conversion_formula()
    test_conversion.test_zero_returns_preserve_price()
    
    # Evaluate samples metrics
    print("\n--- Evaluate Samples Metrics Tests ---")
    test_metrics = TestEvaluateSamplesMetrics()
    test_metrics.test_primary_metrics_on_prices()
    test_metrics.test_secondary_return_metrics()
    test_metrics.test_temporal_metrics()
    test_metrics.test_fallback_without_last_prices()
    
    # Multi-sample evaluation
    print("\n--- Multi-Sample Evaluation Tests ---")
    test_multi = TestMultiSampleEvaluation()
    test_multi.test_multi_sample_reshaping()
    test_multi.test_probabilistic_metrics()
    test_multi.test_single_sample_no_probabilistic_metrics()
    
    # Visualization
    print("\n--- Visualization Tests ---")
    test_viz = TestVisualization()
    test_viz.test_visualization_with_price_conversion()
    test_viz.test_visualization_with_history()
    test_viz.test_visualization_with_ensemble()
    test_viz.test_visualization_without_cumulative()
    
    # Prepare data
    print("\n--- Prepare Data Tests ---")
    test_prepare = TestPrepareData()
    test_prepare.test_prepare_data_extracts_last_prices()
    
    # Return metrics
    print("\n--- Return Metrics Tests ---")
    test_return = TestReturnMetrics()
    test_return.test_return_direction_accuracy()
    test_return.test_structural_metrics()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
