"""
Guide: Handling Multiple Samples in Task.evaluate_samples()

After the trainer clones batches for n_samples_per_input > 1,
your task's evaluate_samples() receives different shapes:

INPUT SHAPES:
- generated_samples: (B*n, T, N, F)  if n_samples_per_input > 1
                      (B, T, N, F)     if n_samples_per_input = 1
- real_samples:      (B, T, N, F)     always
- metadata['n_samples_per_input']: n  (will be in metadata if > 1)
"""

import torch
from graph_signal_diffusion.utils import reshape_generated_samples, repeat_real_samples


def example_evaluate_samples(generated_samples, real_samples, metadata, viz_save_dir=None):
    """
    Example implementation of evaluate_samples() that handles multiple samples per input.
    
    Args:
        generated_samples: (B*n, T, N, F) or (B, T, N, F)
        real_samples: (B, T, N, F)
        metadata: dict with 'n_samples_per_input' if > 1
        viz_save_dir: optional directory for visualizations
    
    Returns:
        dict of metrics
    """
    # Check if multiple samples were generated per input
    n_samples_per_input = metadata.get('n_samples_per_input', 1)
    
    if n_samples_per_input > 1:
        print(f"Handling {n_samples_per_input} samples per input...")
        return evaluate_multiple_samples(
            generated_samples, real_samples, n_samples_per_input, metadata, viz_save_dir
        )
    else:
        # Single sample per input - standard evaluation
        return evaluate_single_sample(generated_samples, real_samples, metadata, viz_save_dir)


def evaluate_single_sample(generated_samples, real_samples, metadata, viz_save_dir):
    """
    Standard evaluation when n_samples_per_input = 1.
    
    Shapes:
        generated_samples: (B, T, N, F)
        real_samples: (B, T, N, F)
    """
    assert generated_samples.shape == real_samples.shape
    
    metrics = {}
    
    # Standard metrics
    metrics['mse'] = ((generated_samples - real_samples) ** 2).mean().item()
    metrics['mae'] = (generated_samples - real_samples).abs().mean().item()
    
    # Visualization (optional)
    if viz_save_dir:
        import matplotlib.pyplot as plt
        import os
        os.makedirs(viz_save_dir, exist_ok=True)
        
        # Plot first batch item
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(real_samples[0, :, 0, 0].cpu(), label='Real')
        plt.plot(generated_samples[0, :, 0, 0].cpu(), label='Generated')
        plt.legend()
        plt.title('Sample Prediction')
        plt.savefig(f"{viz_save_dir}/prediction.png")
        plt.close()
    
    return metrics


def evaluate_multiple_samples(generated_samples, real_samples, n_samples_per_input, metadata, viz_save_dir):
    """
    Enhanced evaluation when n_samples_per_input > 1.
    
    Shapes:
        generated_samples: (B*n, T, N, F)
        real_samples: (B, T, N, F)
    
    Strategies:
        1. Reshape and compute ensemble/diversity metrics
        2. Select best sample per input
        3. Compute per-sample metrics
    """
    B = real_samples.shape[0]
    n = n_samples_per_input
    
    metrics = {}
    
    # ===== Strategy 1: Reshape to (B, n, T, N, F) for per-input analysis =====
    gen_reshaped = reshape_generated_samples(generated_samples, n_samples_per_input)
    # gen_reshaped: (B, n, T, N, F)
    
    # Compute ensemble mean (average across n samples)
    gen_mean = gen_reshaped.mean(dim=1)  # (B, T, N, F)
    metrics['mse_ensemble'] = ((gen_mean - real_samples) ** 2).mean().item()
    metrics['mae_ensemble'] = (gen_mean - real_samples).abs().mean().item()
    
    # Compute sample diversity (std across n samples)
    gen_std = gen_reshaped.std(dim=1)  # (B, T, N, F)
    metrics['diversity_mean'] = gen_std.mean().item()
    metrics['diversity_std'] = gen_std.std().item()
    
    # ===== Strategy 2: Find best sample per input =====
    # Compute MSE for each of the n samples per input
    mse_per_sample = ((gen_reshaped - real_samples.unsqueeze(1)) ** 2).mean(dim=(2, 3, 4))
    # mse_per_sample: (B, n) - MSE for each sample
    
    best_indices = mse_per_sample.argmin(dim=1)  # (B,)
    best_samples = gen_reshaped[torch.arange(B), best_indices]  # (B, T, N, F)
    
    metrics['mse_best'] = ((best_samples - real_samples) ** 2).mean().item()
    metrics['mae_best'] = (best_samples - real_samples).abs().mean().item()
    
    # Oracle improvement (best vs mean)
    metrics['oracle_improvement'] = (metrics['mse_ensemble'] - metrics['mse_best']) / metrics['mse_ensemble']
    
    # ===== Strategy 3: Per-sample metrics (treat as extended batch) =====
    # Repeat real samples to match generated shape
    real_repeated = repeat_real_samples(real_samples, n_samples_per_input)
    # real_repeated: (B*n, T, N, F)
    
    metrics['mse_all'] = ((generated_samples - real_repeated) ** 2).mean().item()
    metrics['mae_all'] = (generated_samples - real_repeated).abs().mean().item()
    
    # ===== Visualization (optional) =====
    if viz_save_dir:
        import matplotlib.pyplot as plt
        import os
        os.makedirs(viz_save_dir, exist_ok=True)
        
        # Plot first input with all n samples
        plt.figure(figsize=(15, 5))
        
        # Subplot 1: All samples + real
        plt.subplot(1, 3, 1)
        for i in range(n):
            plt.plot(gen_reshaped[0, i, :, 0, 0].cpu(), alpha=0.5, label=f'Sample {i+1}')
        plt.plot(real_samples[0, :, 0, 0].cpu(), 'k--', linewidth=2, label='Real')
        plt.legend()
        plt.title(f'All {n} Generated Samples vs Real')
        
        # Subplot 2: Ensemble mean + uncertainty
        plt.subplot(1, 3, 2)
        mean_curve = gen_mean[0, :, 0, 0].cpu()
        std_curve = gen_std[0, :, 0, 0].cpu()
        t = torch.arange(len(mean_curve))
        plt.plot(mean_curve, label='Ensemble Mean')
        plt.fill_between(t, mean_curve - std_curve, mean_curve + std_curve, alpha=0.3)
        plt.plot(real_samples[0, :, 0, 0].cpu(), 'k--', linewidth=2, label='Real')
        plt.legend()
        plt.title('Ensemble Prediction with Uncertainty')
        
        # Subplot 3: Best sample
        plt.subplot(1, 3, 3)
        best_idx = best_indices[0].item()
        plt.plot(gen_reshaped[0, best_idx, :, 0, 0].cpu(), label=f'Best (Sample {best_idx+1})')
        plt.plot(real_samples[0, :, 0, 0].cpu(), 'k--', linewidth=2, label='Real')
        plt.legend()
        plt.title(f'Best Sample (MSE={mse_per_sample[0, best_idx].item():.4f})')
        
        plt.tight_layout()
        plt.savefig(f"{viz_save_dir}/multi_sample_prediction.png", dpi=150)
        plt.close()
        
        # Diversity heatmap (optional)
        if gen_reshaped.shape[3] > 1:  # If multiple nodes
            plt.figure(figsize=(12, 4))
            diversity_map = gen_std[0, 0, :, 0].cpu()  # (N,) - diversity per node
            plt.bar(range(len(diversity_map)), diversity_map)
            plt.xlabel('Node')
            plt.ylabel('Sample Std')
            plt.title('Prediction Diversity per Node')
            plt.savefig(f"{viz_save_dir}/diversity_per_node.png", dpi=150)
            plt.close()
    
    return metrics


# ===== Example Task Implementation =====

class ExampleTaskWithMultiSample:
    """
    Example task that properly handles n_samples_per_input.
    """
    
    def __init__(self):
        self.n_samples_per_input = 5  # Generate 5 samples per input
        self.eval_every_n_epochs = 1
        self.normalizer = None  # Add your normalizer here if needed
    
    def evaluate_samples(self, generated_samples, real_samples, metadata, viz_save_dir=None):
        """
        Evaluate generated samples against real samples.
        
        This method is called by the trainer and must handle:
        - generated_samples: (B*n, T, N, F) if n_samples_per_input > 1
        - real_samples: (B, T, N, F) always
        """
        n_samples_per_input = metadata.get('n_samples_per_input', 1)
        
        if n_samples_per_input > 1:
            return evaluate_multiple_samples(
                generated_samples, real_samples, 
                n_samples_per_input, metadata, viz_save_dir
            )
        else:
            return evaluate_single_sample(
                generated_samples, real_samples, 
                metadata, viz_save_dir
            )


if __name__ == "__main__":
    print("="*70)
    print("Testing multi-sample evaluation")
    print("="*70)
    
    # Setup
    B, n, T, N, F = 4, 5, 10, 100, 1
    
    real_samples = torch.randn(B, T, N, F)
    generated_samples = torch.randn(B * n, T, N, F)
    
    metadata = {
        'n_samples_per_input': n,
        'batch_size': B,
        'batch_size_cloned': B * n,
    }
    
    # Evaluate
    metrics = evaluate_multiple_samples(
        generated_samples, real_samples, n, metadata, viz_save_dir=None
    )
    
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")
    
    print("\n" + "="*70)
    print("Key Insights:")
    print("="*70)
    print(f"""
    1. Ensemble MSE ({metrics['mse_ensemble']:.6f}) - Average prediction quality
    2. Best Sample MSE ({metrics['mse_best']:.6f}) - Oracle performance
    3. Oracle Improvement: {metrics['oracle_improvement']*100:.2f}% - How much better best is than ensemble
    4. Sample Diversity: {metrics['diversity_mean']:.6f} - Prediction uncertainty
    
    If diversity is too low: Model is overconfident (all samples similar)
    If diversity is too high: Model is underconfident (samples too varied)
    """)
