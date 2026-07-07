"""
Test evaluation and visualization with multiple samples per input.
Tests price conversion, de-scaling, and probabilistic metrics.
"""
import pytest
import torch
from graph_signal_diffusion.tasks.stock_price_forecasting.evaluator import StockPriceForecastingTask


def test_multi_sample_evaluation():
    """Test that evaluator correctly handles multiple samples per input with price conversion."""
    
    print("\n" + "="*70)
    print("Testing Multi-Sample Evaluation with Price Conversion")
    print("="*70)
    
    # Setup
    B, n, T, N, F = 2, 5, 10, 20, 1
    task = StockPriceForecastingTask()
    
    # Create synthetic RAW log returns data
    torch.manual_seed(42)
    
    # Generated: (B*n, T, N, F) - output from diffusion model after cloning
    generated_log_returns = 0.0002 + 0.015 * torch.randn(B * n, T, N, F)
    
    # Real: (B, T, N, F) - ground truth log returns
    real_log_returns = 0.0002 + 0.015 * torch.randn(B, T, N, F)
    
    # Create historical prices for conversion (start at $100)
    initial_prices = torch.full((B, N, 1), 100.0)
    last_prices = initial_prices  # Simplified - assume last observed price is $100
    
    # Metadata with price information
    metadata = {
        'n_samples_per_input': n,
        'edge_index': torch.randint(0, N, (2, N * 3)),  # Dummy graph
        'last_prices': last_prices,  # Required for price conversion
    }
    
    print(f"Setup:")
    print(f"  Batch size: {B}")
    print(f"  Samples per input: {n}")
    print(f"  Timesteps: {T}")
    print(f"  Stocks: {N}")
    print(f"  Log return std: {real_log_returns.std().item():.4f}")
    print(f"  Starting price: ${last_prices[0, 0, 0].item():.2f}")
    
    # Evaluate
    print(f"\nRunning evaluation...")
    metrics = task.evaluate_samples(
        generated_samples=generated_log_returns,
        real_samples=real_log_returns,
        metadata=metadata,
        viz_save_dir=None  # Don't save during test
    )
    
    print(f"✓ Evaluation completed!")
    
    # Verify PRIMARY metrics exist (computed on PRICES)
    assert 'price_rmse' in metrics, "Should have price RMSE"
    assert 'price_mae' in metrics, "Should have price MAE"
    assert 'price_mape' in metrics, "Should have price MAPE"
    
    # Verify ENSEMBLE metrics exist
    assert 'sample_diversity' in metrics, "Should have diversity metric"
    
    # Verify PROBABILISTIC metrics exist (computed on PRICES)
    assert 'crps_mean' in metrics, "Should have CRPS metric"
    assert 'mis_90' in metrics, "Should have MIS at 90% confidence"
    assert 'mis_80' in metrics, "Should have MIS at 80% confidence"
    assert 'coverage_90' in metrics, "Should have coverage at 90% confidence"
    assert 'coverage_80' in metrics, "Should have coverage at 80% confidence"
    assert 'ensemble_spread_mean' in metrics, "Should have ensemble spread"
    
    # Verify RETURN metrics exist (computed on LOG RETURNS)
    assert 'volclustering_gen' in metrics, "Should have volclustering_gen"
    
    # Verify values are reasonable
    assert metrics['sample_diversity'] >= 0, "Diversity should be non-negative"
    assert metrics['crps_mean'] >= 0, "CRPS should be non-negative"
    assert 0 <= metrics['coverage_90'] <= 1, "Coverage should be in [0, 1]"
    assert 0 <= metrics['coverage_80'] <= 1, "Coverage should be in [0, 1]"
    assert metrics['price_rmse'] > 0, "Price RMSE should be positive"
    assert metrics['price_mae'] > 0, "Price MAE should be positive"
    
    print("\n✓ Multi-sample evaluation test passed!")
    print(f"\nPrice Metrics:")
    print(f"  RMSE: ${metrics['price_rmse']:.2f}")
    print(f"  MAE: ${metrics['price_mae']:.2f}")
    print(f"  MAPE: {metrics['price_mape']:.2%}")
    print(f"\nEnsemble Metrics:")
    print(f"  CRPS: {metrics['crps_mean']:.4f}")
    print(f"  MIS-90: {metrics['mis_90']:.4f}")
    print(f"  Coverage-90: {metrics['coverage_90']*100:.1f}%")
    print(f"  Diversity: {metrics['sample_diversity']:.4f}")
    print(f"\nReturn Metrics:")
    print(f"  VolClustering: {metrics['volclustering_gen']:.4f}")


def test_single_sample_evaluation():
    """Test that evaluator still works with single sample (n=1) and price conversion."""
    
    print("\n" + "="*70)
    print("Testing Single-Sample Evaluation with Price Conversion")
    print("="*70)
    
    # Setup
    B, T, N, F = 2, 10, 20, 1
    task = StockPriceForecastingTask()
    
    torch.manual_seed(123)
    
    # Create synthetic data - raw log returns
    generated_log_returns = 0.0002 + 0.015 * torch.randn(B, T, N, F)
    real_log_returns = 0.0002 + 0.015 * torch.randn(B, T, N, F)
    
    # Price information for conversion
    last_prices = torch.full((B, N, 1), 100.0)
    
    # Metadata (no n_samples_per_input means n=1)
    metadata = {
        'edge_index': torch.randint(0, N, (2, N * 3)),
        'last_prices': last_prices,
    }
    
    print(f"Setup:")
    print(f"  Batch size: {B}")
    print(f"  Single sample evaluation (n=1)")
    print(f"  Timesteps: {T}")
    print(f"  Stocks: {N}")
    
    # Evaluate
    print(f"\nRunning evaluation...")
    metrics = task.evaluate_samples(
        generated_samples=generated_log_returns,
        real_samples=real_log_returns,
        metadata=metadata,
        viz_save_dir=None
    )
    
    print(f"✓ Evaluation completed!")
    
    # Verify metrics exist (computed on PRICES)
    assert 'price_rmse' in metrics, "Should have price RMSE"
    assert 'price_mae' in metrics, "Should have price MAE"
    assert 'volclustering_gen' in metrics, "Should have volclustering_gen"
    
    # Should NOT have multi-sample metrics
    assert 'sample_diversity' not in metrics, "Should not have diversity for single sample"
    assert 'crps_mean' not in metrics, "Should not have CRPS for single sample"
    
    print("\n✓ Single-sample evaluation test passed!")
    print(f"\nPrice Metrics:")
    print(f"  RMSE: ${metrics['price_rmse']:.2f}")
    print(f"  MAE: ${metrics['price_mae']:.2f}")
    print(f"  MAPE: {metrics['price_mape']:.2%}")
    print(f"\nReturn Metrics:")
    print(f"  VolClustering: {metrics['volclustering_gen']:.4f}")


if __name__ == "__main__":
    test_multi_sample_evaluation()
    test_single_sample_evaluation()
    print("\n✓ All evaluation tests passed!")
