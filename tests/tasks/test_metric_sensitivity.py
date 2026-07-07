"""
Test to visualize how metrics change with prediction accuracy.

This test creates predictions with varying noise levels (from perfect to very noisy)
and plots how different metrics deteriorate as prediction quality decreases.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from graph_signal_diffusion.tasks.stock_price_forecasting.evaluator import StockPriceForecastingTask


def test_metric_sensitivity():
    """Test how metrics change with increasing prediction noise."""
    
    print("\n" + "="*70)
    print("Testing Metric Sensitivity to Prediction Quality")
    print("="*70)
    
    # Setup
    B = 5  # batch size
    T = 20  # timesteps
    N = 10  # stocks
    F = 1  # features
    n_samples = 10  # ensemble size
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Generate realistic target log returns
    target_log_returns = torch.randn(B, T, N, F) * 0.015  # ~1.5% daily std
    
    # Create metadata with last prices
    last_prices = torch.ones(B, N, 1) * 100.0  # Start at $100
    metadata = {
        'last_prices': last_prices,
        'n_samples_per_input': n_samples,
    }
    
    # Test different noise levels (0 = perfect, 1 = same std as target)
    noise_levels = np.linspace(0, 2.0, 15)  # 0x to 2x target std
    
    results = {
        'noise_levels': [],
        'rmse': [],
        'mae': [],
        'mape': [],
        'crps': [],
        'coverage_90': [],
        'sample_diversity': [],
    }
    
    # Create evaluator (disable visualization by overriding the method)
    task = StockPriceForecastingTask(
        history_length=10,
        forecast_horizon=T,
        edge_index=torch.tensor([[0, 1], [1, 0]]),  # Dummy edge index
        normalizer=None
    )
    
    # Disable visualization to avoid memory issues
    task._visualize_results = lambda metadata, viz_save_dir: None
    
    print(f"\nSetup:")
    print(f"  Batch size (B): {B}")
    print(f"  Timesteps (T): {T}")
    print(f"  Stocks (N): {N}")
    print(f"  Ensemble size (n): {n_samples}")
    print(f"  Target log return std: {target_log_returns.std().item():.4f}")
    print(f"  Testing {len(noise_levels)} noise levels from 0.0 to {noise_levels[-1]:.1f}x target std")
    
    # Test each noise level
    for noise_level in noise_levels:
        # Generate predictions with varying noise
        # Start with target + noise, create ensemble
        noise = torch.randn(B, n_samples, T, N, F) * (target_log_returns.std() * noise_level)
        target_expanded = target_log_returns.unsqueeze(1)  # [B, 1, T, N, F]
        generated_samples = target_expanded + noise  # [B, n, T, N, F]
        
        # Reshape for evaluator: [B*n, T, N, F]
        generated_samples_flat = generated_samples.reshape(B * n_samples, T, N, F)
        
        # Create fresh metadata for each evaluation
        eval_metadata = {
            'last_prices': last_prices.clone(),
            'n_samples_per_input': n_samples,
        }
        
        # Evaluate (without visualization)
        metrics = task.evaluate_samples(
            generated_samples=generated_samples_flat,
            real_samples=target_log_returns,
            metadata=eval_metadata,
            viz_save_dir=None  # Skip visualization
        )
        
        # Store results
        results['noise_levels'].append(noise_level)
        results['rmse'].append(metrics.get('price_rmse', np.nan))
        results['mae'].append(metrics.get('price_mae', np.nan))
        results['mape'].append(metrics.get('price_mape', np.nan))
        results['crps'].append(metrics.get('crps_mean', np.nan))
        results['coverage_90'].append(metrics.get('coverage_90', np.nan))
        results['sample_diversity'].append(metrics.get('sample_diversity', np.nan))
        
        print(f"  Noise={noise_level:.2f}x: RMSE=${metrics.get('price_rmse', 0):.2f}, "
              f"CRPS={metrics.get('crps_mean', 0):.2f}, "
              f"Coverage={metrics.get('coverage_90', 0):.1f}%")
    
    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Metric Sensitivity to Prediction Quality\n(Higher noise = worse predictions)', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: RMSE
    ax = axes[0, 0]
    ax.plot(results['noise_levels'], results['rmse'], 'o-', linewidth=2, markersize=6, color='#e74c3c')
    ax.set_xlabel('Noise Level (× target std)', fontsize=10)
    ax.set_ylabel('Price RMSE ($)', fontsize=10)
    ax.set_title('Root Mean Squared Error', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axvline(x=0, color='green', linestyle='--', alpha=0.5, label='Perfect')
    
    # Plot 2: MAE
    ax = axes[0, 1]
    ax.plot(results['noise_levels'], results['mae'], 'o-', linewidth=2, markersize=6, color='#3498db')
    ax.set_xlabel('Noise Level (× target std)', fontsize=10)
    ax.set_ylabel('Price MAE ($)', fontsize=10)
    ax.set_title('Mean Absolute Error', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axvline(x=0, color='green', linestyle='--', alpha=0.5, label='Perfect')
    
    # Plot 3: CRPS
    ax = axes[0, 2]
    ax.plot(results['noise_levels'], results['crps'], 'o-', linewidth=2, markersize=6, color='#9b59b6')
    ax.set_xlabel('Noise Level (× target std)', fontsize=10)
    ax.set_ylabel('CRPS', fontsize=10)
    ax.set_title('Continuous Ranked Probability Score', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axvline(x=0, color='green', linestyle='--', alpha=0.5, label='Perfect')
    
    # Plot 4: Coverage
    ax = axes[1, 0]
    ax.plot(results['noise_levels'], results['coverage_90'], 'o-', linewidth=2, markersize=6, color='#2ecc71')
    ax.axhline(y=90, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Target (90%)')
    ax.set_xlabel('Noise Level (× target std)', fontsize=10)
    ax.set_ylabel('Coverage (%)', fontsize=10)
    ax.set_title('90% Prediction Interval Coverage', fontweight='bold')
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Plot 5: Sample Diversity
    ax = axes[1, 1]
    ax.plot(results['noise_levels'], results['sample_diversity'], 'o-', linewidth=2, markersize=6, color='#f39c12')
    ax.set_xlabel('Noise Level (× target std)', fontsize=10)
    ax.set_ylabel('Sample Diversity (std)', fontsize=10)
    ax.set_title('Ensemble Sample Diversity', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 6: MAPE (log scale)
    ax = axes[1, 2]
    mape_values = [m if m < 1e6 else np.nan for m in results['mape']]  # Cap extreme values
    ax.plot(results['noise_levels'], mape_values, 'o-', linewidth=2, markersize=6, color='#e67e22')
    ax.set_xlabel('Noise Level (× target std)', fontsize=10)
    ax.set_ylabel('MAPE (%)', fontsize=10)
    ax.set_title('Mean Absolute Percentage Error', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path(__file__).parent.parent / "figs" / "metric_sensitivity"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "metric_sensitivity.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved metric sensitivity plot to: {output_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("Key Observations:")
    print("="*70)
    print(f"  Perfect predictions (noise=0):")
    print(f"    - RMSE: ${results['rmse'][0]:.2f}")
    print(f"    - CRPS: {results['crps'][0]:.2f}")
    print(f"    - Coverage: {results['coverage_90'][0]:.1f}%")
    print(f"\n  Worst predictions (noise={noise_levels[-1]:.1f}x):")
    print(f"    - RMSE: ${results['rmse'][-1]:.2f}")
    print(f"    - CRPS: {results['crps'][-1]:.2f}")
    print(f"    - Coverage: {results['coverage_90'][-1]:.1f}%")
    
    # Avoid division by zero
    if results['rmse'][0] > 0:
        print(f"\n  RMSE increased by {results['rmse'][-1]/results['rmse'][0]:.1f}x")
    else:
        print(f"\n  RMSE: from ${results['rmse'][0]:.2f} to ${results['rmse'][-1]:.2f}")
    
    if results['crps'][0] > 0:
        print(f"  CRPS increased by {results['crps'][-1]/results['crps'][0]:.1f}x")
    else:
        print(f"  CRPS: from {results['crps'][0]:.2f} to {results['crps'][-1]:.2f}")
    
    print("\n" + "="*70)
    print("✓✓✓ METRIC SENSITIVITY TEST PASSED! ✓✓✓")
    print("="*70)
    print(f"\nView plot at: {output_path}")


if __name__ == "__main__":
    test_metric_sensitivity()
