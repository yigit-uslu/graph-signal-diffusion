"""
Unit test to diagnose evaluation issues:
1. NaN/Infinity metrics in price conversion
2. Missing data points in visualizations
3. Missing generated samples per input in visualization

ROOT CAUSE IDENTIFIED:
======================
The raw CSV 'DailyLogReturn' column is ALREADY STANDARDIZED (z-scored, std≈1).
When converting to prices: price_t = last_price * exp(cumsum(log_returns))
With z-scored data: cumsum can reach ~5-10, making exp(10) ≈ 22000!

This causes:
- Inf/NaN in price metrics
- Unrealistic visualization values
- All price-based metrics are meaningless

SOLUTIONS:
1. Store original normalization params (mean, std) with the data
2. Denormalize before price conversion: real_r = z * original_std + original_mean  
3. Or compute metrics on standardized returns only (no price conversion)

Run with: pytest tests/tasks/test_evaluation_issues.py -v -s
"""
import pytest
import torch
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from graph_signal_diffusion.tasks.stock_price_forecasting.evaluator import StockPriceForecastingTask
from graph_signal_diffusion.utils import reshape_generated_samples
from tests.tasks._stock_viz_fixtures import make_viz_metadata

OUTPUT_DIR = Path(__file__).parent / "evaluation_issues_tests"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


class TestNaNInfinityMetrics:
    """Diagnose why metrics become NaN or Infinity."""
    
    def test_large_log_returns_cause_inf(self):
        """
        ISSUE: Large generated log returns cause exp() overflow -> Inf prices -> Inf metrics.
        
        If diffusion model generates log returns outside reasonable range (e.g., |r| > 5),
        exp(cumsum(r)) can overflow to Inf.
        """
        task = StockPriceForecastingTask()
        B, T, N, F = 2, 10, 5, 1
        
        # Realistic last prices
        last_prices = torch.tensor([100.0, 150.0, 200.0, 120.0, 80.0]).view(1, N, 1).expand(B, N, 1)
        
        # Case 1: Normal log returns (typical range -0.1 to +0.1)
        gen_normal = torch.randn(B, T, N, F) * 0.02  # ~2% daily returns
        real_normal = torch.randn(B, T, N, F) * 0.02
        
        metadata_normal = {
            'batch_size': B, 'num_stocks': N, 'num_timesteps': T, 'num_features': F,
            'last_prices': last_prices.clone(),
        }
        
        metrics_normal = task.evaluate_samples(gen_normal, real_normal, metadata_normal, viz_save_dir=None)
        print("\n=== Normal log returns (std=0.02) ===")
        print(f"Generated log returns range: [{gen_normal.min():.4f}, {gen_normal.max():.4f}]")
        print(f"price_mse: {metrics_normal.get('price_mse', 'N/A')}")
        print(f"price_mae: {metrics_normal.get('price_mae', 'N/A')}")
        
        # Case 2: Abnormally large log returns (simulating bad diffusion output)
        gen_large = torch.randn(B, T, N, F) * 5.0  # 500% daily returns (UNREALISTIC!)
        real_large = torch.randn(B, T, N, F) * 0.02
        
        metadata_large = {
            'batch_size': B, 'num_stocks': N, 'num_timesteps': T, 'num_features': F,
            'last_prices': last_prices.clone(),
        }
        
        metrics_large = task.evaluate_samples(gen_large, real_large, metadata_large, viz_save_dir=None)
        print("\n=== Large log returns (std=5.0) - PROBLEMATIC ===")
        print(f"Generated log returns range: [{gen_large.min():.4f}, {gen_large.max():.4f}]")
        print(f"price_mse: {metrics_large.get('price_mse', 'N/A')}")
        print(f"price_mae: {metrics_large.get('price_mae', 'N/A')}")
        
        # Demonstrate the math
        cumsum = torch.cumsum(gen_large, dim=1)
        prices = last_prices.unsqueeze(1) * torch.exp(cumsum)
        print(f"Cumulative log returns max: {cumsum.max():.2f}")
        print(f"exp(cumsum) max: {torch.exp(cumsum).max():.2e}")
        print(f"Price max: {prices.max():.2e}")
        print(f"Has Inf in prices: {torch.isinf(prices).any()}")
        
        # Check for NaN/Inf
        has_nan = np.isnan(metrics_large.get('price_mse', 0))
        has_inf = np.isinf(metrics_large.get('price_mse', 0))
        
        if has_nan or has_inf:
            print("\n⚠️  ISSUE CONFIRMED: Large log returns cause NaN/Inf metrics!")
            print("   ROOT CAUSE: exp(cumsum(large_values)) overflows to Inf")
            print("   SOLUTION: Clamp generated log returns to reasonable range [-0.5, 0.5]")
        
        # Suggest fix: clamp log returns
        gen_clamped = torch.clamp(gen_large, -0.5, 0.5)
        metadata_clamped = {
            'batch_size': B, 'num_stocks': N, 'num_timesteps': T, 'num_features': F,
            'last_prices': last_prices.clone(),
        }
        metrics_clamped = task.evaluate_samples(gen_clamped, real_large, metadata_clamped, viz_save_dir=None)
        print("\n=== After clamping to [-0.5, 0.5] ===")
        print(f"price_mse: {metrics_clamped.get('price_mse', 'N/A')}")
        print(f"price_mae: {metrics_clamped.get('price_mae', 'N/A')}")
    
    def test_missing_last_prices(self):
        """
        ISSUE: If last_prices is missing, price conversion fails gracefully but
        metrics on log returns may have different scale.
        """
        task = StockPriceForecastingTask()
        B, T, N, F = 2, 5, 3, 1
        
        gen = torch.randn(B, T, N, F) * 0.02
        real = torch.randn(B, T, N, F) * 0.02
        
        # Without last_prices
        metadata_no_prices = {
            'batch_size': B, 'num_stocks': N, 'num_timesteps': T, 'num_features': F,
            # NO last_prices!
        }
        
        print("\n=== Missing last_prices ===")
        metrics = task.evaluate_samples(gen, real, metadata_no_prices, viz_save_dir=None)
        
        # Should fall back to log return metrics
        assert 'logreturn_mse' in metrics or 'price_mse' in metrics, \
            "Should have either logreturn or price metrics"
        print(f"Metrics keys: {list(metrics.keys())[:5]}...")


class TestVisualizationMissingData:
    """Diagnose missing data points in visualization."""
    
    def test_hist_prices_shape_mismatch(self):
        """
        ISSUE: hist_prices shape may not match expectations, causing missing history in plots.
        """
        task = StockPriceForecastingTask()
        B, T, N, F = 1, 5, 3, 1
        T_hist = 10

        metadata = make_viz_metadata(B=B, T=T, N=N, F=F, n=5, with_history=True, T_hist=T_hist)

        # History must be present with the expected [B, T_hist, N, 1] shape so the
        # history segment renders rather than silently dropping.
        assert 'hist_prices' in metadata
        assert metadata['hist_prices'].shape == (B, T_hist, N, 1), \
            f"hist_prices shape mismatch: {metadata['hist_prices'].shape}"

        save_path = OUTPUT_DIR / "test_hist_prices.pdf"
        fig = task.visualize_predictions(
            metadata=metadata,
            stocks=[0, 1, 2],
            batch_index=0,
            save_dir=str(OUTPUT_DIR),
            plot_cumulative=True,
        )
        assert fig is not None
        plt.close(fig)


class TestMultiSampleVisualization:
    """Diagnose missing ensemble samples in visualization."""
    
    def test_ensemble_samples_not_passed_to_viz(self):
        """
        ISSUE: gen_ensemble may not be correctly passed to visualize_predictions.
        
        When n_samples_per_input > 1:
        1. trainer.evaluate() generates (B*n, T, N, F) samples
        2. evaluate_samples() reshapes to (B, n, T, N, F) and stores in metadata['gen_ensemble']
        3. visualize_predictions() should receive this and plot individual samples
        """
        task = StockPriceForecastingTask()
        B, T, N, F = 2, 5, 4, 1
        n_samples = 5

        # make_viz_metadata drives the real pipeline, so metadata['gen_ensemble']
        # is the reshaped (B, n, T, N, F) ensemble that visualize_predictions reads.
        metadata = make_viz_metadata(B=B, T=T, N=N, F=F, n=n_samples, with_history=True, T_hist=10)

        # Verify the ensemble is present with the expected n_samples on the ensemble axis.
        assert 'gen_ensemble' in metadata, "Ensemble should be in metadata"
        assert metadata['gen_ensemble'].shape[1] == n_samples, \
            f"Ensemble should have {n_samples} samples, got {metadata['gen_ensemble'].shape[1]}"

        fig = task.visualize_predictions(
            metadata=metadata,
            stocks=[0, 1, 2, 3],
            batch_index=0,
            save_dir=str(OUTPUT_DIR),
            plot_cumulative=True,
            show_confidence_bands=True,
        )
        assert fig is not None
        plt.close(fig)

    def test_trainer_evaluation_flow(self):
        """
        Simulate the exact flow in trainer.evaluate() to verify shapes.
        """
        print("\n=== Simulating trainer.evaluate() flow ===")
        
        B, T, N, F = 4, 10, 80, 1  # Typical batch: 4 graphs, 10 timesteps, 80 stocks
        n_samples_per_input = 3
        
        # Step 1: prepare_data returns real_samples [B, T, N, F]
        real_samples = torch.randn(B, T, N, F) * 0.02
        print(f"1. real_samples from prepare_data: {real_samples.shape}")
        
        # Step 2: Clone batch (as in trainer)
        from torch_geometric.data import Data, Batch
        
        # Create mock PyG data
        single_data = Data(
            x=torch.randn(N, 10, 5),  # [N, T_past, F_in]
            y=torch.randn(N, T, F),   # [N, T, F]
            edge_index=torch.randint(0, N, (2, 200)),
        )
        data_batch = Batch.from_data_list([single_data] * B)
        print(f"2. Original batch: {data_batch.num_graphs} graphs, {data_batch.num_nodes} nodes")
        
        # Clone for multi-sample
        if n_samples_per_input > 1:
            data_list = data_batch.to_data_list()
            data_cloned = Batch.from_data_list(
                [g for g in data_list for _ in range(n_samples_per_input)]
            )
            B_cloned = B * n_samples_per_input
            sample_shape = (B_cloned, T, N, F)
        else:
            data_cloned = data_batch
            sample_shape = real_samples.shape
        
        print(f"3. Cloned batch: {data_cloned.num_graphs} graphs")
        print(f"4. sample_shape for diffusion.sample(): {sample_shape}")
        
        # Step 3: Generate samples (mock)
        generated_samples_normalized = torch.randn(*sample_shape) * 0.02
        print(f"5. generated_samples shape: {generated_samples_normalized.shape}")
        
        # Step 4: Pass to evaluate_samples
        metadata = {
            'batch_size': B,
            'num_stocks': N,
            'num_timesteps': T,
            'num_features': F,
            'last_prices': torch.rand(B, N, 1) * 100 + 50,
            'n_samples_per_input': n_samples_per_input,
        }
        
        # Check shapes match what evaluate_samples expects
        if n_samples_per_input > 1:
            # evaluate_samples expects (B*n, T, N, F) and will reshape
            assert generated_samples_normalized.shape[0] == B * n_samples_per_input, \
                f"Expected B*n={B*n_samples_per_input}, got {generated_samples_normalized.shape[0]}"
            print(f"✓ Shape matches: (B*n={B}*{n_samples_per_input}, T={T}, N={N}, F={F})")
        
        print("✓ Trainer evaluation flow simulation complete")


class TestPrepareDataShapes:
    """Test that prepare_data produces correct shapes for the evaluator."""
    
    def test_3d_input_reshape(self):
        """
        ISSUE: When samples come as [B*N, T, F], reshape to [B, T, N, F] may fail
        if batch size inference is wrong.
        """
        from torch_geometric.data import Data, Batch
        
        task = StockPriceForecastingTask()
        B, T, N, F = 2, 5, 10, 1
        
        # Create mock PyG batch
        single_data = Data(
            x=torch.randn(N, 20, 8),  # [N, T_past, F_in]
            y=torch.randn(N, T, F),   # [N, T, F]  - will be flattened by PyG
            edge_index=torch.randint(0, N, (2, 50)),
            close_price=torch.rand(N, 20, 1) * 100 + 50,
            close_price_y=torch.rand(N, T, 1) * 100 + 50,
            stocks_index=torch.arange(N),
            timestamp=torch.tensor([0] * N),
        )
        
        data_batch = Batch.from_data_list([single_data] * B)
        
        print("\n=== Testing prepare_data with PyG batch ===")
        print(f"Batch y shape (flattened by PyG): {data_batch.y.shape}")
        print(f"Batch close_price shape: {data_batch.close_price.shape}")
        print(f"Batch batch tensor: {data_batch.batch.shape}")
        print(f"Batch num_graphs: {data_batch.num_graphs}")
        
        # prepare_data should correctly reshape
        try:
            result = task.prepare_data(data_batch)
            print(f"Prepared samples shape: {result['samples'].shape}")
            print(f"Metadata batch_size: {result['metadata'].get('batch_size')}")
            print(f"Metadata num_stocks: {result['metadata'].get('num_stocks')}")
            
            assert result['samples'].dim() == 4, "Should be 4D [B, T, N, F]"
            assert result['samples'].shape == (B, T, N, F), \
                f"Expected {(B, T, N, F)}, got {result['samples'].shape}"
            print("✓ prepare_data correctly reshapes to [B, T, N, F]")
            
        except Exception as e:
            print(f"✗ prepare_data failed: {e}")
            import traceback
            traceback.print_exc()
            raise


def run_all_diagnostics():
    """Run all diagnostic tests and summarize findings."""
    print("=" * 70)
    print("EVALUATION ISSUES DIAGNOSTIC TESTS")
    print("=" * 70)
    
    test_classes = [
        TestNaNInfinityMetrics(),
        TestVisualizationMissingData(),
        TestMultiSampleVisualization(),
        TestPrepareDataShapes(),
    ]
    
    issues_found = []
    
    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"\n{'='*70}")
        print(f"Running {class_name}")
        print("=" * 70)
        
        for method_name in dir(test_class):
            if method_name.startswith('test_'):
                method = getattr(test_class, method_name)
                print(f"\n--- {method_name} ---")
                try:
                    method()
                except AssertionError as e:
                    issues_found.append(f"{class_name}.{method_name}: {e}")
                except Exception as e:
                    issues_found.append(f"{class_name}.{method_name}: {e}")
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if issues_found:
        print(f"\n⚠️  {len(issues_found)} issues found:")
        for issue in issues_found:
            print(f"  - {issue}")
    else:
        print("\n✓ All tests passed!")
    
    print("\nKEY FINDINGS:")
    print("1. NaN/Inf metrics: Caused by large log returns -> exp() overflow")
    print("   FIX: Clamp generated log returns to [-0.5, 0.5] before price conversion")
    print("")
    print("2. Missing viz data: Check hist_prices and past_samples are passed correctly")
    print("   FIX: Ensure prepare_data extracts and reshapes hist_prices to [B, T_hist, N, 1]")
    print("")
    print("3. Missing ensemble samples: Verify gen_ensemble is stored in metadata")
    print("   FIX: Ensure evaluate_samples stores reshaped ensemble before visualization")


if __name__ == "__main__":
    run_all_diagnostics()
