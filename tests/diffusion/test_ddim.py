"""Unit tests for DDIM (Denoising Diffusion Implicit Model) sampling."""
import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

from graph_signal_diffusion.diffusion.ddpm import DDPM
from graph_signal_diffusion.diffusion.ddim import DDIM


class SimpleModel(nn.Module):
    """Simple model for testing diffusion sampling."""
    
    def __init__(self, channels=1):
        super().__init__()
        self.channels = channels
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.time_embed = nn.Embedding(1000, 128)
        self.proj = nn.Linear(128, channels)
    
    def forward(self, x, timesteps, edge_index=None, edge_weight=None, cond=None, return_intermediates=False):
        """
        Args:
            x: [B, T, N, F]
            timesteps: [B]
        Returns:
            pred: [B, T, N, F]
            intermediates: None or dict
        """
        B, T, N, F = x.shape
        
        # Simple identity + time embedding
        t_emb = self.time_embed(timesteps)  # [B, 128]
        t_emb = self.proj(t_emb)  # [B, F]
        t_emb = t_emb.view(B, 1, 1, F)  # [B, 1, 1, F]
        
        # Simple prediction: slight modification of input
        pred = x + 0.1 * t_emb
        
        return pred, None if not return_intermediates else {}


class ProbeSelectorStub(nn.Module):
    """Selector stub exposing sampling probe hooks used by DDPM/DDIM diagnostics."""

    def __init__(self):
        super().__init__()
        self.collect_sampling_probe = False
        self._last_diag = {"selector_level": 0}

    def set_collect_sampling_probe(self, enabled: bool) -> None:
        self.collect_sampling_probe = bool(enabled)

    def get_selector_diagnostics(self):
        return self._last_diag

    def update_probe_masks(self, active_in: torch.Tensor, active_out: torch.Tensor) -> None:
        diag = {"selector_level": 0}
        if self.collect_sampling_probe:
            diag["probe_active_mask_in"] = active_in.detach()
            diag["probe_active_mask_out"] = active_out.detach()
        self._last_diag = diag


class ProbeAwareModel(SimpleModel):
    """Simple model augmented with a probe-capable selector diagnostics module."""

    def __init__(self, channels=1):
        super().__init__(channels=channels)
        self.selector_probe = ProbeSelectorStub()

    def forward(self, x, timesteps, edge_index=None, edge_weight=None, cond=None, return_intermediates=False):
        pred, intermediates = super().forward(
            x=x,
            timesteps=timesteps,
            edge_index=edge_index,
            edge_weight=edge_weight,
            cond=cond,
            return_intermediates=return_intermediates,
        )
        B, _, N, _ = x.shape
        idx = torch.arange(N, device=x.device)
        shift = int(timesteps[0].item()) % max(1, N)
        selected = (((idx + shift) % 2) == 0).float()
        active_in = torch.ones((B, N), device=x.device, dtype=x.dtype)
        active_out = selected.view(1, N).expand(B, -1).to(dtype=x.dtype)
        self.selector_probe.update_probe_masks(active_in=active_in, active_out=active_out)
        return pred, intermediates


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    return SimpleModel(channels=2)


@pytest.fixture
def probe_model():
    """Create a probe-aware model for selector sampling diagnostics tests."""
    return ProbeAwareModel(channels=2)


@pytest.fixture
def simple_graph_data():
    """Create simple graph data for testing."""
    B, N, T, F = 2, 10, 8, 2
    
    # Create node features (conditioning)
    x = torch.randn(B * N, T, F)
    
    # Create target signals
    y = torch.randn(B * N, T, F)
    
    # Create simple edge index (fully connected per graph)
    edge_indices = []
    for b in range(B):
        offset = b * N
        for i in range(N):
            for j in range(N):
                if i != j:
                    edge_indices.append([offset + i, offset + j])
    
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t()
    
    # Create batch
    batch = torch.repeat_interleave(torch.arange(B), N)
    
    data = Data(x=x, y=y, edge_index=edge_index, batch=batch)
    data.num_graphs = B
    
    return data


class TestDDIMSampling:
    """Test suite for DDIM sampling functionality."""
    
    @pytest.mark.parametrize("beta_schedule", ["linear", "cosine"])
    def test_ddim_initialization(self, simple_model, beta_schedule):
        """Test DDIM initialization with different parameters."""
        # Standard initialization
        ddim = DDIM(
            model=simple_model,
            num_timesteps=100,
            sampling_timesteps=10,
            ddim_eta=0.0,
            beta_schedule=beta_schedule,
        )
        
        assert ddim.num_timesteps == 100
        assert ddim.sampling_timesteps == 10
        assert ddim.ddim_eta == 0.0
        assert len(ddim.ddim_timesteps) <= 11  # 10 sampling steps + potentially endpoint
        assert ddim.get_speedup() == 10.0
        
        print(f"✓ DDIM initialization ({beta_schedule}): PASSED")
    
    @pytest.mark.parametrize("beta_schedule", ["linear", "cosine"])
    def test_ddim_timestep_schedule(self, simple_model, beta_schedule):
        """Test that DDIM creates proper timestep subsequences."""
        # Test different acceleration factors
        for num_t, sample_t in [(1000, 50), (1000, 100), (500, 25)]:
            ddim = DDIM(
                model=simple_model,
                num_timesteps=num_t,
                sampling_timesteps=sample_t,
                beta_schedule=beta_schedule,
            )
            
            timesteps = ddim.ddim_timesteps
            
            # Check timesteps are in descending order
            assert torch.all(timesteps[:-1] >= timesteps[1:]), \
                f"Timesteps should be descending: {timesteps[:10]}"
            
            # Check we end at or near 0
            assert timesteps[-1] <= 10, f"Should end near 0, got {timesteps[-1]}"
            
            # Check we start near max
            assert timesteps[0] >= num_t - num_t // sample_t, \
                f"Should start near {num_t}, got {timesteps[0]}"
        
        print(f"✓ DDIM timestep schedule ({beta_schedule}): PASSED")
    
    @pytest.mark.parametrize("beta_schedule", ["linear", "cosine"])
    def test_ddim_no_acceleration(self, simple_model, beta_schedule):
        """Test DDIM with sampling_timesteps=num_timesteps (no acceleration)."""
        ddim = DDIM(
            model=simple_model,
            num_timesteps=100,
            sampling_timesteps=100,
            beta_schedule=beta_schedule,
        )
        
        # Should use num_timesteps-1 timesteps (excluding t=0, which is handled specially)
        # Timesteps go from 99 down to 1, final step to 0 is implicit
        assert len(ddim.ddim_timesteps) == 99
        assert ddim.ddim_timesteps[0].item() == 99  # Starts at T-1
        assert ddim.ddim_timesteps[-1].item() == 1  # Ends at 1 (step to 0 is final)
        
        print(f"✓ DDIM no acceleration ({beta_schedule}): PASSED")
    
    @pytest.mark.parametrize("beta_schedule", ["linear", "cosine"])
    def test_ddim_deterministic_sampling(self, simple_model, simple_graph_data, beta_schedule):
        """Test deterministic DDIM sampling (eta=0.0)."""
        ddim = DDIM(
            model=simple_model,
            num_timesteps=50,
            sampling_timesteps=10,
            ddim_eta=0.0,
            parameterization="eps",
            beta_schedule=beta_schedule,
        )
        
        B, N, T, F = 2, 10, 8, 2
        shape = (B, T, N, F)
        device = torch.device("cpu")
        
        # Sample twice with same seed - should be identical
        torch.manual_seed(42)
        sample1 = ddim.sample(shape, device, simple_graph_data)
        
        torch.manual_seed(42)
        sample2 = ddim.sample(shape, device, simple_graph_data)
        
        assert torch.allclose(sample1, sample2, atol=1e-5), \
            "Deterministic sampling should produce identical results"
        
        assert sample1.shape == shape
        
        print(f"✓ DDIM deterministic sampling ({beta_schedule}): PASSED")
    
    @pytest.mark.parametrize("beta_schedule", ["linear", "cosine"])
    def test_ddim_stochastic_sampling(self, simple_model, simple_graph_data, beta_schedule):
        """Test stochastic DDIM sampling (eta > 0)."""
        ddim = DDIM(
            model=simple_model,
            num_timesteps=50,
            sampling_timesteps=10,
            ddim_eta=0.5,
            parameterization="eps",
            beta_schedule=beta_schedule,
        )
        
        B, N, T, F = 2, 10, 8, 2
        shape = (B, T, N, F)
        device = torch.device("cpu")
        
        # Sample twice with different seeds - should be different
        torch.manual_seed(42)
        sample1 = ddim.sample(shape, device, simple_graph_data)
        
        torch.manual_seed(123)
        sample2 = ddim.sample(shape, device, simple_graph_data)
        
        assert not torch.allclose(sample1, sample2, atol=1e-3), \
            "Stochastic sampling should produce different results"
        
        print(f"✓ DDIM stochastic sampling ({beta_schedule}): PASSED")
    
    @pytest.mark.parametrize("beta_schedule", ["linear", "cosine"])
    def test_ddim_vs_ddpm_with_eta_1(self, simple_model, simple_graph_data, beta_schedule):
        """Test that DDIM with eta=1.0 and full timesteps behaves similarly to DDPM."""
        num_timesteps = 50
        
        # Create DDPM
        ddpm = DDPM(
            model=simple_model,
            num_timesteps=num_timesteps,
            parameterization="eps",
            beta_schedule=beta_schedule,
        )
        
        # Create DDIM with eta=1.0 and full timesteps (should approximate DDPM)
        ddim = DDIM(
            model=simple_model,
            num_timesteps=num_timesteps,
            sampling_timesteps=num_timesteps,
            ddim_eta=1.0,
            parameterization="eps",
            beta_schedule=beta_schedule,
        )
        
        B, N, T, F = 2, 10, 8, 2
        shape = (B, T, N, F)
        device = torch.device("cpu")
        
        # Sample from both
        torch.manual_seed(42)
        ddpm_sample = ddpm.sample(shape, device, simple_graph_data)
        
        torch.manual_seed(42)
        ddim_sample = ddim.sample(shape, device, simple_graph_data)
        
        # They won't be exactly identical due to implementation differences,
        # but should be reasonably close
        mse = torch.mean((ddpm_sample - ddim_sample) ** 2).item()
        print(f"   MSE between DDPM and DDIM(eta=1.0): {mse:.6f}")
        
        # Check that they're in similar range and not completely different
        assert mse < 10.0, \
            f"DDIM with eta=1.0 should approximate DDPM, but MSE={mse:.6f} is too large"
        
        # Check distributions are similar
        ddpm_mean, ddpm_std = ddpm_sample.mean(), ddpm_sample.std()
        ddim_mean, ddim_std = ddim_sample.mean(), ddim_sample.std()
        
        print(f"   DDPM: mean={ddpm_mean:.4f}, std={ddpm_std:.4f}")
        print(f"   DDIM: mean={ddim_mean:.4f}, std={ddim_std:.4f}")
        
        assert abs(ddpm_mean - ddim_mean) < 0.5, "Means should be similar"
        assert abs(ddpm_std - ddim_std) < 0.5, "Stds should be similar"
        
        print(f"✓ DDIM vs DDPM with eta=1.0 ({beta_schedule}): PASSED")
    
    @pytest.mark.parametrize("beta_schedule", ["linear", "cosine"])
    def test_ddim_different_sampling_rates(self, simple_model, simple_graph_data, beta_schedule):
        """Test DDIM with various sampling rates."""
        num_timesteps = 100
        B, N, T, F = 2, 10, 8, 2
        shape = (B, T, N, F)
        device = torch.device("cpu")
        
        sampling_rates = [10, 20, 50, 100]
        samples = {}
        
        for sampling_timesteps in sampling_rates:
            ddim = DDIM(
                model=simple_model,
                num_timesteps=num_timesteps,
                sampling_timesteps=sampling_timesteps,
                ddim_eta=0.0,
                parameterization="eps",
                beta_schedule=beta_schedule,
            )
            
            speedup = ddim.get_speedup()
            expected_speedup = num_timesteps / sampling_timesteps
            assert speedup == expected_speedup, \
                f"Speedup should be {expected_speedup}x, got {speedup}x"
            
            torch.manual_seed(42)
            sample = ddim.sample(shape, device, simple_graph_data)
            samples[sampling_timesteps] = sample
            
            print(f"   Sampling rate {sampling_timesteps}: speedup={speedup}x, "
                  f"mean={sample.mean():.4f}, std={sample.std():.4f}")
        
        # All samples should be valid (not NaN or Inf)
        for rate, sample in samples.items():
            assert torch.isfinite(sample).all(), \
                f"Sample with rate {rate} contains NaN or Inf"
        
        print(f"✓ DDIM different sampling rates ({beta_schedule}): PASSED")
    
    @pytest.mark.parametrize("beta_schedule", ["linear", "cosine"])
    def test_ddim_faster_than_ddpm(self, simple_model, beta_schedule):
        """Verify that DDIM with fewer timesteps requires fewer model calls."""
        import time
        
        num_timesteps = 100
        B, N, T, F = 1, 10, 8, 2
        shape = (B, T, N, F)
        device = torch.device("cpu")
        
        # Create simple graph data matching the shape
        x = torch.randn(B * N, T, F)
        y = torch.randn(B * N, T, F)
        edge_indices = []
        for b in range(B):
            offset = b * N
            for i in range(N):
                for j in range(N):
                    if i != j:
                        edge_indices.append([offset + i, offset + j])
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t() if edge_indices else torch.empty((2, 0), dtype=torch.long)
        batch = torch.repeat_interleave(torch.arange(B), N)
        data = Data(x=x, y=y, edge_index=edge_index, batch=batch)
        data.num_graphs = B
        
        # DDPM with full timesteps
        ddpm = DDPM(
            model=simple_model,
            num_timesteps=num_timesteps,
            parameterization="eps",
            beta_schedule=beta_schedule,
        )
        
        # DDIM with 10x acceleration
        ddim = DDIM(
            model=simple_model,
            num_timesteps=num_timesteps,
            sampling_timesteps=10,
            ddim_eta=0.0,
            parameterization="eps",
            beta_schedule=beta_schedule,
        )
        
        # Time DDPM
        torch.manual_seed(42)
        start = time.time()
        ddpm_sample = ddpm.sample(shape, device, data)
        ddpm_time = time.time() - start
        
        # Time DDIM
        torch.manual_seed(42)
        start = time.time()
        ddim_sample = ddim.sample(shape, device, data)
        ddim_time = time.time() - start
        
        print(f"   DDPM time: {ddpm_time:.4f}s")
        print(f"   DDIM time: {ddim_time:.4f}s")
        print(f"   Speedup: {ddpm_time / ddim_time:.2f}x")
        
        # DDIM should be significantly faster (at least 3x due to 10x fewer steps)
        assert ddim_time < ddpm_time * 0.5, \
            f"DDIM should be faster: {ddim_time:.4f}s vs {ddpm_time:.4f}s"
        
        print(f"✓ DDIM faster than DDPM ({beta_schedule}): PASSED")
    
    @pytest.mark.parametrize("beta_schedule", ["linear", "cosine"])
    def test_ddim_eta_variations(self, simple_model, simple_graph_data, beta_schedule):
        """Test DDIM with different eta values."""
        B, N, T, F = 2, 10, 8, 2
        shape = (B, T, N, F)
        device = torch.device("cpu")
        
        etas = [0.0, 0.25, 0.5, 0.75, 1.0]
        
        for eta in etas:
            ddim = DDIM(
                model=simple_model,
                num_timesteps=50,
                sampling_timesteps=10,
                ddim_eta=eta,
                parameterization="eps",
                beta_schedule=beta_schedule,
            )
            
            torch.manual_seed(42)
            sample = ddim.sample(shape, device, simple_graph_data)
            
            assert torch.isfinite(sample).all(), \
                f"Sample with eta={eta} contains NaN or Inf"
            
            print(f"   eta={eta}: mean={sample.mean():.4f}, std={sample.std():.4f}")
        
        print(f"✓ DDIM eta variations ({beta_schedule}): PASSED")
    
    @pytest.mark.parametrize("beta_schedule", ["linear", "cosine"])
    def test_ddim_training_loss_same_as_ddpm(self, simple_model, simple_graph_data, beta_schedule):
        """Verify that DDIM uses the same training loss as DDPM."""
        # DDIM and DDPM should have identical training losses
        # since DDIM only changes the sampling procedure
        
        ddpm = DDPM(
            model=simple_model,
            num_timesteps=100,
            parameterization="eps",
            beta_schedule=beta_schedule,
        )
        
        ddim = DDIM(
            model=simple_model,
            num_timesteps=100,
            sampling_timesteps=10,
            ddim_eta=0.0,
            parameterization="eps",
            beta_schedule=beta_schedule,
        )
        
        # Compute loss on same data with same random seed
        torch.manual_seed(42)
        ddpm_loss = ddpm.training_loss(simple_graph_data)
        
        torch.manual_seed(42)
        ddim_loss = ddim.training_loss(simple_graph_data)
        
        print(f"   DDPM loss: {ddpm_loss:.6f}")
        print(f"   DDIM loss: {ddim_loss:.6f}")
        
        assert torch.allclose(ddpm_loss, ddim_loss, atol=1e-6), \
            "DDIM and DDPM should have identical training losses"
        
        print(f"✓ DDIM training loss same as DDPM ({beta_schedule}): PASSED")

    @pytest.mark.parametrize("beta_schedule", ["linear", "cosine"])
    def test_ddim_selector_sampling_probe_diagnostics(
        self,
        probe_model,
        simple_graph_data,
        beta_schedule,
    ):
        """DDIM should support DDPM-style selector sampling diagnostics payloads."""
        ddim = DDIM(
            model=probe_model,
            num_timesteps=50,
            sampling_timesteps=10,
            ddim_eta=0.0,
            parameterization="eps",
            beta_schedule=beta_schedule,
        )

        B, N, T, F = 2, 10, 8, 2
        shape = (B, T, N, F)
        device = torch.device("cpu")
        probe_steps = [49, 30, 10, 0]

        simple_graph_data.network_id = torch.tensor([0], dtype=torch.long)
        simple_graph_data.dataset_name = "sp500"

        generated, selector_sampling_diagnostics = ddim.sample(
            shape=shape,
            device=device,
            data=simple_graph_data,
            return_selector_sampling_diagnostics=True,
            selector_probe_timesteps=probe_steps,
        )

        assert tuple(generated.shape) == shape
        assert isinstance(selector_sampling_diagnostics, dict)

        resolved_steps = selector_sampling_diagnostics.get("probe_timesteps", [])
        assert len(resolved_steps) > 0
        assert set(resolved_steps).issubset(set(int(t) for t in ddim.ddim_timesteps.tolist()))
        assert resolved_steps == sorted(resolved_steps, reverse=True)

        graph_count = selector_sampling_diagnostics.get("probe_graph_count", [])
        assert len(graph_count) == len(resolved_steps)
        assert all(abs(float(c) - float(B)) < 1e-6 for c in graph_count)

        selection_cond = selector_sampling_diagnostics.get("selection_given_available_by_level", {})
        assert isinstance(selection_cond, dict)
        assert "0" in selection_cond
        level0 = selection_cond["0"]
        assert len(level0) == len(resolved_steps)
        assert all(len(row) == N for row in level0)

        by_network = selector_sampling_diagnostics.get("by_network", {})
        assert isinstance(by_network, dict)
        assert "sp500::network_0" in by_network


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
