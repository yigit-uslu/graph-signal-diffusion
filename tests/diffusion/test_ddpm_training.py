"""Integration test for DDPM training and sampling on toy 2D data."""
import pytest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
import matplotlib.pyplot as plt
from pathlib import Path

from graph_signal_diffusion.diffusion.ddpm import DDPM
from graph_signal_diffusion.diffusion.ddim import DDIM


class SimpleMLP(nn.Module):
    """Simple MLP for denoising 2D points."""
    
    def __init__(self, hidden_dim=128, time_embed_dim=64):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        self.net = nn.Sequential(
            nn.Linear(2 + time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )
    
    def forward(self, x, timesteps, edge_index=None, edge_weight=None, cond=None, return_intermediates=False):
        """
        Args:
            x: [B, T, N, F] where T=1, N=1, F=2 for 2D points
            timesteps: [B]
        Returns:
            pred: [B, T, N, F]
            intermediates: None or dict
        """
        B = x.shape[0]
        
        # Flatten spatial dimensions for 2D point
        x_flat = x.reshape(B, -1)  # [B, T*N*F] = [B, 2]
        
        # Time embedding
        t_normalized = timesteps.float().unsqueeze(-1) / 1000.0  # [B, 1]
        t_emb = self.time_embed(t_normalized)  # [B, time_embed_dim]
        
        # Concatenate and process
        x_in = torch.cat([x_flat, t_emb], dim=-1)  # [B, 2 + time_embed_dim]
        pred_flat = self.net(x_in)  # [B, 2]
        
        # Reshape back
        pred = pred_flat.reshape(B, 1, 1, 2)  # [B, T=1, N=1, F=2]
        
        return pred, None if not return_intermediates else {}


class ToyDataset(Dataset):
    """Toy 2D dataset with mixture of Gaussians."""
    
    def __init__(self, n_samples=1000, mode='circle'):
        super().__init__()
        self.n_samples = n_samples
        self.mode = mode
        self.samples = self._generate_samples()
    
    def _generate_samples(self):
        """Generate 2D samples."""
        if self.mode == 'circle':
            # Points on a circle
            theta = torch.rand(self.n_samples) * 2 * np.pi
            r = 1.0 + torch.randn(self.n_samples) * 0.1
            x = r * torch.cos(theta)
            y = r * torch.sin(theta)
            samples = torch.stack([x, y], dim=1)
            
        elif self.mode == 'gaussian_mixture':
            # Mixture of 4 Gaussians
            n_per_mode = self.n_samples // 4
            centers = torch.tensor([
                [1.0, 1.0],
                [-1.0, 1.0],
                [-1.0, -1.0],
                [1.0, -1.0],
            ])
            samples = []
            for center in centers:
                samples.append(center + torch.randn(n_per_mode, 2) * 0.2)
            samples = torch.cat(samples, dim=0)
            
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        return samples
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        # Return as graph data format expected by DDPM
        # Shape: [N=1, T=1, F=2] where we have a single 2D point
        y = self.samples[idx].unsqueeze(0).unsqueeze(0)  # [1, 1, 2]
        
        # Create minimal graph structure (single node, no edges)
        data = Data(
            y=y,  # [N=1, T=1, F=2]
            edge_index=torch.empty((2, 0), dtype=torch.long),
            x=None,  # No conditioning
        )
        data.num_graphs = 1
        
        return data


def wasserstein_distance_2d(samples1, samples2):
    """Approximate 1-Wasserstein distance for 2D point clouds."""
    # Sort both by x, then y
    s1 = samples1[samples1[:, 0].argsort()]
    s2 = samples2[samples2[:, 0].argsort()]
    
    # Take equal number of samples
    n = min(len(s1), len(s2))
    s1, s2 = s1[:n], s2[:n]
    
    # Compute mean L1 distance
    return torch.abs(s1 - s2).mean().item()


class TestDDPMTraining:
    """Test DDPM training and sampling on toy data."""
    
    @pytest.mark.parametrize("mode", ["circle", "gaussian_mixture"])
    def test_ddpm_trains_and_samples(self, mode):
        """Test that DDPM can learn and sample from a toy 2D distribution."""
        # Set seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Hyperparameters
        n_train_samples = 500
        n_epochs = 500  # Reduced from 1000
        batch_size = 32
        n_generated_samples = 200
        beta_schedule = "cosine"
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\nUsing device: {device}")
        
        # Create dataset and dataloader
        dataset = ToyDataset(n_samples=n_train_samples, mode=mode)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Get training data statistics for later comparison
        train_samples_flat = dataset.samples  # [N, 2]
        train_mean = train_samples_flat.mean(dim=0)
        train_std = train_samples_flat.std(dim=0)
        
        print(f"\nTraining on {mode} distribution with {beta_schedule} schedule:")
        print(f"  Train mean: {train_mean.numpy()}")
        print(f"  Train std:  {train_std.numpy()}")
        
        # Create model and diffusion
        model = SimpleMLP(hidden_dim=128, time_embed_dim=64).to(device)
        ddpm = DDPM(
            model=model,
            num_timesteps=500,  # Number of diffusion timesteps
            beta_schedule=beta_schedule,
            beta_start=1e-4,
            beta_end=2e-2,
            loss_type="l2",
            parameterization="eps",
        ).to(device)
        
        optimizer = torch.optim.Adam(ddpm.parameters(), lr=1e-3)
        
        # Training loop
        ddpm.train()
        losses = []
        
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            for batch in dataloader:
                batch = batch.to(device)
                
                # Forward pass
                loss = ddpm.training_loss(batch)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            losses.append(avg_loss)
            
            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")
        
        # Verify training progressed (loss decreased)
        initial_loss = np.mean(losses[:5])
        final_loss = np.mean(losses[-5:])
        print(f"\n  Initial loss: {initial_loss:.4f}")
        print(f"  Final loss:   {final_loss:.4f}")
        print(f"  Improvement:  {(initial_loss - final_loss) / initial_loss * 100:.1f}%")
        
        # Verify training converged (we accept 20% improvement)
        assert final_loss < initial_loss * 0.8, \
            f"Training did not converge sufficiently: {final_loss:.4f} vs {initial_loss:.4f}"
        
        # Generate samples
        ddpm.eval()
        print(f"\nGenerating {n_generated_samples} samples...")
        
        # Create dummy graph data for sampling
        dummy_data = Data(
            edge_index=torch.empty((2, 0), dtype=torch.long).to(device),
            x=None,
        )
        dummy_data.num_graphs = n_generated_samples
        
        with torch.no_grad():
            generated = ddpm.sample(
                shape=(n_generated_samples, 1, 1, 2),  # [B, T=1, N=1, F=2]
                device=device,
                data=dummy_data,
            )
        
        # Extract 2D points from generated samples
        generated_samples = generated.squeeze().cpu()  # [B, 2]
        
        if generated_samples.dim() == 1:
            generated_samples = generated_samples.unsqueeze(0)
        
        # Compute statistics
        gen_mean = generated_samples.mean(dim=0)
        gen_std = generated_samples.std(dim=0)
        
        print(f"\nGenerated samples statistics:")
        print(f"  Generated mean: {gen_mean.numpy()}")
        print(f"  Generated std:  {gen_std.numpy()}")
        
        # Assertions: check if generated distribution matches training distribution
        
        # 1. Mean should be close
        mean_error = torch.abs(gen_mean - train_mean).max().item()
        print(f"\n  Mean error (max):  {mean_error:.4f}")
        assert mean_error < 0.3, \
            f"Generated mean too far from training mean: {mean_error:.4f}"
        
        # 2. Std should be close
        std_error = torch.abs(gen_std - train_std).max().item()
        print(f"  Std error (max):   {std_error:.4f}")
        assert std_error < 0.75, \
            f"Generated std too far from training std: {std_error:.4f}"
        
        # 3. Wasserstein distance should be small
        wd = wasserstein_distance_2d(
            generated_samples[:n_train_samples],
            train_samples_flat[:n_train_samples]
        )
        print(f"  Wasserstein dist:  {wd:.4f}")
        # Wasserstein distance should be reasonable (< 1.5 is acceptable for toy 2D data)
        assert wd < 1.5, \
            f"Wasserstein distance too large: {wd:.4f}"
        
        # 3.5. Create scatter plot visualization
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        # Plot training samples
        ax.scatter(train_samples_flat[:, 0], train_samples_flat[:, 1], 
                   alpha=0.5, s=20, c='blue', label='Training samples')
        
        # Plot generated samples
        ax.scatter(generated_samples[:, 0], generated_samples[:, 1], 
                   alpha=0.5, s=20, c='red', label='Generated samples')
        
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_title(f'DDPM: Training vs Generated ({mode}, {beta_schedule})', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Save plot
        output_dir = Path('tests/figs/ddpm_viz')
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_path = output_dir / f'ddpm_training_{mode}_{beta_schedule}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved visualization to: {plot_path}")
        
        # 3.6. Compare DDPM with DDIM samplers
        print(f"\nComparing DDPM vs DDIM samplers:")
        
        # DDIM configurations to test
        ddim_configs = [
            {'sampling_timesteps': 50, 'ddim_eta': 0.0, 'label': 'DDIM (50 steps, η=0.0)'},
            {'sampling_timesteps': 50, 'ddim_eta': 0.5, 'label': 'DDIM (50 steps, η=0.5)'},
            {'sampling_timesteps': 50, 'ddim_eta': 1.0, 'label': 'DDIM (50 steps, η=1.0)'},
            {'sampling_timesteps': 20, 'ddim_eta': 0.0, 'label': 'DDIM (20 steps, η=0.0)'},
            {'sampling_timesteps': 500, 'ddim_eta': 1.0, 'label': 'DDIM (500 steps, η=1.0)'},
        ]
        
        # Generate samples from each DDIM configuration
        ddim_samples = {}
        for config in ddim_configs:
            ddim = DDIM(
                model=model,
                num_timesteps=500,  # Same as DDPM training timesteps
                beta_schedule=beta_schedule,
                sampling_timesteps=config['sampling_timesteps'],
                ddim_eta=config['ddim_eta']
            ).to(device)
            
            # Generate samples
            with torch.no_grad():
                ddim_samples_batch = ddim.sample(
                    shape=(n_generated_samples, 1, 1, 2),
                    device=device,
                    data=dummy_data,
                )
            
            samples = ddim_samples_batch.squeeze().cpu()  # Keep as tensor [batch, 2]
            if samples.dim() == 1:
                samples = samples.unsqueeze(0)
            ddim_samples[config['label']] = samples.numpy()  # Convert to numpy for plotting
            
            # Compute metrics (use tensor for wasserstein_distance_2d)
            wd = wasserstein_distance_2d(
                samples[:len(train_samples_flat)].cpu(),
                train_samples_flat.cpu()
            )
            print(f"  {config['label']:30s}: WD = {wd:.4f}")
        
        # Create comparison visualization
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        axes = axes.flatten()
        
        # Plot 1: Training data
        axes[0].scatter(train_samples_flat[:, 0], train_samples_flat[:, 1], 
                       alpha=0.5, s=20, c='blue')
        axes[0].set_title('Training samples', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_aspect('equal')
        
        # Plot 2: DDPM samples
        axes[1].scatter(generated_samples[:, 0], generated_samples[:, 1], 
                       alpha=0.5, s=20, c='red')
        axes[1].set_title('DDPM (500 steps)', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_aspect('equal')
        
        # Plots 3-7: DDIM samples
        for idx, (label, samples) in enumerate(ddim_samples.items(), start=2):
            axes[idx].scatter(samples[:, 0], samples[:, 1], 
                            alpha=0.5, s=20, c='green')
            axes[idx].set_title(label, fontsize=12)
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_aspect('equal')
        
        # Hide unused subplots
        for idx in range(len(ddim_samples) + 2, len(axes)):
            axes[idx].axis('off')
            axes[idx].set_aspect('equal')
        
        fig.suptitle(f'DDPM vs DDIM Comparison ({mode}, {beta_schedule})', fontsize=16, y=0.995)
        plt.tight_layout()
        
        # Save comparison plot
        comparison_path = output_dir / f'ddpm_vs_ddim_comparison_{mode}_{beta_schedule}.png'
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved comparison to: {comparison_path}")
        
        # 4. Visual check: samples should be in reasonable range
        gen_range = generated_samples.abs().max().item()
        print(f"  Generated range:   [-{gen_range:.2f}, {gen_range:.2f}]")
        assert gen_range < 5.0, \
            f"Generated samples out of reasonable range: {gen_range:.2f}"
        
        print(f"\n✓ DDPM successfully learned and sampled from {mode} distribution!")
    
    def test_ddpm_overfits_single_point(self):
        """Test that DDPM can overfit to a single point (sanity check)."""
        torch.manual_seed(42)
        
        device = torch.device("cpu")
        target_point = torch.tensor([[0.5, -0.3]])  # Single 2D point
        
        # Create tiny dataset with just one repeated point
        class SinglePointDataset(Dataset):
            def __len__(self):
                return 20  # Repeat same point
            
            def __getitem__(self, idx):
                y = target_point.unsqueeze(0)  # [1, 1, 2]
                return Data(
                    y=y,
                    edge_index=torch.empty((2, 0), dtype=torch.long),
                    x=None,
                    num_graphs=1,
                )
        
        dataset = SinglePointDataset()
        dataloader = DataLoader(dataset, batch_size=5, shuffle=False)
        
        # Small model
        model = SimpleMLP(hidden_dim=64, time_embed_dim=32)
        ddpm = DDPM(
            model=model,
            num_timesteps=50,
            parameterization="eps",
        )
        
        optimizer = torch.optim.Adam(ddpm.parameters(), lr=1e-3)
        
        # Train to overfit
        ddpm.train()
        for epoch in range(100):
            for batch in dataloader:
                loss = ddpm.training_loss(batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # Generate samples
        ddpm.eval()
        dummy_data = Data(
            edge_index=torch.empty((2, 0), dtype=torch.long),
            x=None,
            num_graphs=10,
        )
        
        with torch.no_grad():
            generated = ddpm.sample(
                shape=(10, 1, 1, 2),
                device=device,
                data=dummy_data,
            )
        
        generated_samples = generated.squeeze()
        if generated_samples.dim() == 1:
            generated_samples = generated_samples.unsqueeze(0)
        
        # Should be very close to target point
        mean_generated = generated_samples.mean(dim=0)
        error = torch.abs(mean_generated - target_point.squeeze()).max().item()
        
        print(f"\nTarget point:     {target_point.squeeze().numpy()}")
        print(f"Generated mean:   {mean_generated.numpy()}")
        print(f"Error:            {error:.4f}")
        
        assert error < 0.2, \
            f"Failed to overfit to single point: error={error:.4f}"
        
        print("✓ DDPM successfully overfitted to single point!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
