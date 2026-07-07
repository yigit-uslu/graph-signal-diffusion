"""
Example: How to use n_samples_per_input in your task evaluation.

This example shows how to configure your task to generate multiple samples
per real sample and properly evaluate them.
"""

import torch
from graph_signal_diffusion.utils.batch_cloning import (
    reshape_generated_samples,
    repeat_real_samples
)


def example_task_with_multiple_samples():
    """
    Example showing how to configure a task with multiple samples per input.
    """
    
    class ExampleTask:
        def __init__(self):
            # Set this to generate multiple samples per real sample
            self.n_samples_per_input = 5
            self.eval_every_n_epochs = 1
            
        def evaluate_samples(self, generated_samples, real_samples, metadata, viz_save_dir=None):
            """
            Evaluate generated samples against real samples.
            
            Args:
                generated_samples: Shape (B*n_samples_per_input, T, N, F) or (B, T, N, F)
                real_samples: Shape (B, T, N, F)
                metadata: Dictionary with batch info
                viz_save_dir: Optional directory for visualizations
                
            Returns:
                Dictionary of metrics
            """
            B_real = real_samples.shape[0]
            B_gen = generated_samples.shape[0]
            
            metrics = {}
            
            # Check if multiple samples were generated
            if B_gen > B_real:
                n_samples = B_gen // B_real
                print(f"Evaluating {n_samples} samples per input...")
                
                # Approach 1: Reshape and compute per-input statistics
                gen_reshaped = reshape_generated_samples(
                    generated_samples, 
                    n_samples_per_input=n_samples
                )  # (B, n, T, N, F)
                
                # Compute mean and std across samples
                gen_mean = gen_reshaped.mean(dim=1)  # (B, T, N, F)
                gen_std = gen_reshaped.std(dim=1)    # (B, T, N, F)
                
                # Evaluate mean prediction
                metrics['mse_mean'] = ((gen_mean - real_samples) ** 2).mean().item()
                metrics['mae_mean'] = (gen_mean - real_samples).abs().mean().item()
                
                # Diversity metrics
                metrics['sample_diversity'] = gen_std.mean().item()
                
                # Approach 2: Evaluate best sample per input
                # Compute MSE for each sample
                mse_per_sample = ((gen_reshaped - real_samples.unsqueeze(1)) ** 2).mean(dim=(2, 3, 4))  # (B, n)
                best_sample_indices = mse_per_sample.argmin(dim=1)  # (B,)
                best_samples = gen_reshaped[torch.arange(B_real), best_sample_indices]  # (B, T, N, F)
                
                metrics['mse_best'] = ((best_samples - real_samples) ** 2).mean().item()
                
                # Approach 3: Compute per-sample metrics (treat as extended batch)
                real_repeated = repeat_real_samples(real_samples, n_samples_per_input=n_samples)
                metrics['mse_all'] = ((generated_samples - real_repeated) ** 2).mean().item()
                
            else:
                # Single sample per input
                metrics['mse'] = ((generated_samples - real_samples) ** 2).mean().item()
                metrics['mae'] = (generated_samples - real_samples).abs().mean().item()
            
            return metrics
    
    # Usage in trainer
    print("="*60)
    print("Example: Task with multiple samples per input")
    print("="*60)
    
    task = ExampleTask()
    
    # Simulate real samples
    B, T, N, F = 4, 10, 100, 1
    real_samples = torch.randn(B, T, N, F)
    
    # Simulate generated samples (5 per input)
    n_samples = task.n_samples_per_input
    generated_samples = torch.randn(B * n_samples, T, N, F)
    
    print(f"\nReal samples shape: {real_samples.shape}")
    print(f"Generated samples shape: {generated_samples.shape}")
    print(f"Samples per input: {n_samples}")
    
    # Evaluate
    metadata = {"batch_size": B}
    metrics = task.evaluate_samples(generated_samples, real_samples, metadata)
    
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    return task, metrics


def example_sampling_in_diffusion():
    """
    Example showing how the trainer handles batch cloning for multiple samples.
    """
    print("\n" + "="*60)
    print("Example: Trainer handling batch cloning")
    print("="*60)
    
    from torch_geometric.data import Data, Batch
    
    # Create sample graph data
    B = 4  # Batch size
    N = 100  # Nodes per graph
    
    # Create batched graphs
    graphs = []
    for i in range(B):
        data = Data(
            x=torch.randn(N, 32),
            edge_index=torch.randint(0, N, (2, 500)),
            edge_weight=torch.rand(500)
        )
        graphs.append(data)
    
    batch = Batch.from_data_list(graphs)
    
    print(f"\nOriginal batch:")
    print(f"  Num graphs: {batch.num_graphs}")
    print(f"  Total nodes: {batch.num_nodes}")
    print(f"  Total edges: {batch.num_edges}")
    
    # Simulate what happens in trainer.evaluate() with n_samples_per_input=3
    n_samples_per_input = 3
    
    # Trainer clones the batch BEFORE calling diffusion.sample()
    data_list = batch.to_data_list()
    data_cloned = Batch.from_data_list(
        [g for g in data_list for _ in range(n_samples_per_input)]
    )
    
    print(f"\nCloned batch (n_samples_per_input={n_samples_per_input}):")
    print(f"  Num graphs: {data_cloned.num_graphs}")
    print(f"  Total nodes: {data_cloned.num_nodes}")
    print(f"  Total edges: {data_cloned.num_edges}")
    
    # Trainer adjusts the shape
    real_samples_shape = (B, 10, N, 1)  # (B, T, N, F)
    sample_shape = (B * n_samples_per_input, *real_samples_shape[1:])
    
    print(f"\nShape adjustment:")
    print(f"  Real samples shape: {real_samples_shape}")
    print(f"  Generated samples shape: {sample_shape}")
    
    # Then trainer calls: diffusion.sample(shape=sample_shape, data=data_cloned)
    # Diffusion model doesn't need to know about cloning!
    
    print(f"\nAdvantages:")
    print(f"  ✓ Diffusion model interface unchanged")
    print(f"  ✓ Cloning logic explicit and debuggable in trainer")
    print(f"  ✓ Better separation of concerns")
    print(f"  ✓ Can inspect data_cloned before sampling")


if __name__ == "__main__":
    # Run examples
    task, metrics = example_task_with_multiple_samples()
    example_sampling_in_diffusion()
    
    print("\n" + "="*60)
    print("Summary: To use this in your code")
    print("="*60)
    print("""
1. In your task class, add:
   self.n_samples_per_input = 5  # or any number > 1

2. The trainer will automatically:
   - Clone the data object (graphs) n times
   - Adjust the shape passed to diffusion.sample()
   - Call: diffusion.sample(shape=(B*n, T, N, F), data=data_cloned)

3. Diffusion model interface stays unchanged:
   - No need to modify sample() signature
   - Cleaner separation of concerns
   - Easier to debug and understand

4. In your task's evaluate_samples(), handle the larger batch:
   - Use reshape_generated_samples() to get (B, n, T, N, F) shape
   - Compute statistics like mean, std, best sample, etc.
   - Optionally repeat real samples with repeat_real_samples()

5. Graph batching is handled automatically by PyG's Batch:
   - Edge indices are properly offset for each graph
   - Edge weights are preserved
   - All graph attributes are maintained
    """)
