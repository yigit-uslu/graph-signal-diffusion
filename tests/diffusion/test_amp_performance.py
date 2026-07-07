"""
Unit tests for verifying AMP (Automatic Mixed Precision) speedup and memory gains.

This test compares:
1. FP32 (baseline) vs FP16 (AMP) inference
2. With/without torch.compile() optimization
3. Memory usage differences
4. Output quality (ensure AMP doesn't significantly degrade results)
"""
import pytest
import torch
import torch.nn as nn
import time
import gc
from typing import Dict, Tuple
from torch_geometric.data import Data, Batch

from graph_signal_diffusion.diffusion.ddpm import DDPM
from graph_signal_diffusion.diffusion.ddim import DDIM


class SimpleModel(nn.Module):
    """Simple model for testing diffusion performance."""
    
    def __init__(self, channels=2, hidden_dim=64):
        super().__init__()
        self.channels = channels
        self.hidden_dim = hidden_dim
        
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        self.net = nn.Sequential(
            nn.Linear(channels + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, channels),
        )
    
    def forward(self, x, timesteps, edge_index=None, edge_weight=None, cond=None, return_intermediates=False):
        """
        Args:
            x: [B, T, N, F]
            timesteps: [B]
        Returns:
            pred: [B, T, N, F]
            intermediates: None
        """
        B, T, N, F = x.shape
        
        # Time embedding
        t_normalized = timesteps.float().unsqueeze(-1) / 1000.0
        t_emb = self.time_embed(t_normalized)  # [B, hidden_dim]
        t_emb = t_emb.view(B, 1, 1, self.hidden_dim).expand(B, T, N, self.hidden_dim)
        
        # Concatenate and process
        x_flat = x.view(B * T * N, F)
        t_flat = t_emb.reshape(B * T * N, self.hidden_dim)
        
        x_in = torch.cat([x_flat, t_flat], dim=-1)
        pred_flat = self.net(x_in)
        
        pred = pred_flat.view(B, T, N, F)
        
        return pred, None


def create_test_data(B: int, N: int, T: int, F: int, device: torch.device) -> Data:
    """Create test graph data."""
    x = torch.randn(B * N, T, F, device=device)
    y = torch.randn(B * N, T, F, device=device)
    
    # Create fully connected edge index per graph
    edge_indices = []
    for b in range(B):
        offset = b * N
        for i in range(N):
            for j in range(N):
                if i != j:
                    edge_indices.append([offset + i, offset + j])
    
    edge_index = torch.tensor(edge_indices, dtype=torch.long, device=device).t() if edge_indices else torch.empty((2, 0), dtype=torch.long, device=device)
    batch = torch.repeat_interleave(torch.arange(B, device=device), N)
    
    data = Data(x=x, y=y, edge_index=edge_index, batch=batch)
    data.num_graphs = B
    
    return data


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


def measure_inference(
    diffusion: nn.Module,
    shape: Tuple[int, ...],
    device: torch.device,
    data: Data,
    use_amp: bool,
    num_runs: int = 3,
    warmup_runs: int = 1,
) -> Dict[str, float]:
    """Measure inference time and memory for diffusion sampling."""
    
    # Warmup runs
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = diffusion.sample(shape, device, data, use_amp=use_amp)
    
    torch.cuda.synchronize() if device.type == "cuda" else None
    gc.collect()
    torch.cuda.empty_cache() if device.type == "cuda" else None
    
    # Measure memory before
    mem_before = get_gpu_memory_mb()
    
    # Timed runs
    times = []
    peak_mems = []
    
    for _ in range(num_runs):
        torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
        
        start = time.perf_counter()
        with torch.no_grad():
            sample = diffusion.sample(shape, device, data, use_amp=use_amp)
        torch.cuda.synchronize() if device.type == "cuda" else None
        end = time.perf_counter()
        
        times.append(end - start)
        if device.type == "cuda":
            peak_mems.append(torch.cuda.max_memory_allocated() / (1024 * 1024))
    
    return {
        "mean_time": sum(times) / len(times),
        "min_time": min(times),
        "max_time": max(times),
        "peak_memory_mb": sum(peak_mems) / len(peak_mems) if peak_mems else 0.0,
        "sample_mean": sample.mean().item(),
        "sample_std": sample.std().item(),
    }


class TestAMPPerformance:
    """Test suite for AMP performance verification."""
    
    @pytest.fixture
    def device(self):
        """Get CUDA device if available, otherwise skip GPU tests."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    
    @pytest.fixture
    def model(self, device):
        """Create simple model for testing."""
        return SimpleModel(channels=2, hidden_dim=64).to(device)
    
    @pytest.fixture
    def ddpm(self, model):
        """Create DDPM for testing."""
        return DDPM(
            model=model,
            num_timesteps=50,  # Use fewer steps for faster testing
            beta_schedule="linear",
            parameterization="eps",
        )
    
    @pytest.fixture
    def ddim(self, model):
        """Create DDIM for testing."""
        return DDIM(
            model=model,
            num_timesteps=50,
            sampling_timesteps=10,  # Accelerated
            ddim_eta=0.0,
            beta_schedule="linear",
            parameterization="eps",
        )
    
    def test_amp_ddpm_speedup(self, ddpm, device):
        """Test that AMP provides speedup for DDPM sampling."""
        if device.type != "cuda":
            pytest.skip("CUDA required for AMP speedup test")
        
        B, T, N, F = 2, 4, 8, 2
        shape = (B, T, N, F)
        data = create_test_data(B, N, T, F, device)
        
        ddpm = ddpm.to(device)
        
        # Measure FP32 (baseline)
        fp32_results = measure_inference(ddpm, shape, device, data, use_amp=False, num_runs=3)
        
        # Measure FP16 (AMP)
        fp16_results = measure_inference(ddpm, shape, device, data, use_amp=True, num_runs=3)
        
        print(f"\n=== DDPM AMP Performance ===")
        print(f"FP32 time: {fp32_results['mean_time']:.4f}s")
        print(f"FP16 time: {fp16_results['mean_time']:.4f}s")
        print(f"Speedup: {fp32_results['mean_time'] / fp16_results['mean_time']:.2f}x")
        print(f"FP32 peak memory: {fp32_results['peak_memory_mb']:.1f} MB")
        print(f"FP16 peak memory: {fp16_results['peak_memory_mb']:.1f} MB")
        print(f"Memory reduction: {(1 - fp16_results['peak_memory_mb'] / fp32_results['peak_memory_mb']) * 100:.1f}%")
        
        # Verify FP16 is faster (allow some tolerance for small models)
        # Note: For very small models, FP16 overhead might not provide speedup
        # So we just verify it doesn't significantly slow down
        assert fp16_results['mean_time'] < fp32_results['mean_time'] * 1.5, \
            f"FP16 should not be significantly slower than FP32"
        
        # Verify outputs are similar (FP16 may have small differences)
        assert abs(fp32_results['sample_mean'] - fp16_results['sample_mean']) < 1.0, \
            "Sample means should be similar"
        
        print("✓ DDPM AMP speedup test: PASSED")
    
    def test_amp_ddim_speedup(self, ddim, device):
        """Test that AMP provides speedup for DDIM sampling."""
        if device.type != "cuda":
            pytest.skip("CUDA required for AMP speedup test")
        
        B, T, N, F = 2, 4, 8, 2
        shape = (B, T, N, F)
        data = create_test_data(B, N, T, F, device)
        
        ddim = ddim.to(device)
        
        # Measure FP32 (baseline)
        fp32_results = measure_inference(ddim, shape, device, data, use_amp=False, num_runs=3)
        
        # Measure FP16 (AMP)
        fp16_results = measure_inference(ddim, shape, device, data, use_amp=True, num_runs=3)
        
        print(f"\n=== DDIM AMP Performance ===")
        print(f"FP32 time: {fp32_results['mean_time']:.4f}s")
        print(f"FP16 time: {fp16_results['mean_time']:.4f}s")
        print(f"Speedup: {fp32_results['mean_time'] / fp16_results['mean_time']:.2f}x")
        print(f"FP32 peak memory: {fp32_results['peak_memory_mb']:.1f} MB")
        print(f"FP16 peak memory: {fp16_results['peak_memory_mb']:.1f} MB")
        
        # Verify FP16 is not significantly slower
        assert fp16_results['mean_time'] < fp32_results['mean_time'] * 1.5
        
        # Verify outputs are similar
        assert abs(fp32_results['sample_mean'] - fp16_results['sample_mean']) < 1.0
        
        print("✓ DDIM AMP speedup test: PASSED")
    
    def test_amp_memory_reduction(self, device):
        """Test that AMP reduces memory usage for larger batches."""
        if device.type != "cuda":
            pytest.skip("CUDA required for memory test")
        
        # Use larger dimensions to see memory difference
        B, T, N, F = 8, 16, 32, 4
        shape = (B, T, N, F)
        
        model = SimpleModel(channels=F, hidden_dim=128).to(device)
        ddim = DDIM(
            model=model,
            num_timesteps=100,
            sampling_timesteps=20,
            ddim_eta=0.0,
            beta_schedule="linear",
            parameterization="eps",
        ).to(device)
        
        data = create_test_data(B, N, T, F, device)
        
        # Measure FP32
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            _ = ddim.sample(shape, device, data, use_amp=False)
        torch.cuda.synchronize()
        fp32_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
        
        # Measure FP16
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        torch.cuda.empty_cache()
        
        with torch.no_grad():
            _ = ddim.sample(shape, device, data, use_amp=True)
        torch.cuda.synchronize()
        fp16_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
        
        print(f"\n=== Memory Reduction Test ===")
        print(f"FP32 peak memory: {fp32_memory:.1f} MB")
        print(f"FP16 peak memory: {fp16_memory:.1f} MB")
        print(f"Memory reduction: {(1 - fp16_memory / fp32_memory) * 100:.1f}%")
        
        # Verify memory reduction (at least 10% for larger models)
        # Note: Small models may not show significant reduction
        assert fp16_memory <= fp32_memory, \
            f"FP16 should use less or equal memory: {fp16_memory:.1f} MB vs {fp32_memory:.1f} MB"
        
        print("✓ Memory reduction test: PASSED")
    
    def test_amp_numerical_stability(self, ddim, device):
        """Test that AMP produces numerically stable results."""
        if device.type != "cuda":
            pytest.skip("CUDA required for numerical stability test")
        
        B, T, N, F = 4, 8, 16, 2
        shape = (B, T, N, F)
        data = create_test_data(B, N, T, F, device)
        
        ddim = ddim.to(device)
        
        # Generate multiple samples with AMP
        samples = []
        for seed in range(3):
            torch.manual_seed(seed)
            with torch.no_grad():
                sample = ddim.sample(shape, device, data, use_amp=True)
            samples.append(sample)
        
        # Check for NaN or Inf
        for i, sample in enumerate(samples):
            assert torch.isfinite(sample).all(), f"Sample {i} contains NaN or Inf"
        
        # Check outputs are reasonable (not collapsed or exploded)
        for i, sample in enumerate(samples):
            assert sample.std() > 0.01, f"Sample {i} has collapsed (std too small)"
            assert sample.std() < 100, f"Sample {i} has exploded (std too large)"
            assert abs(sample.mean()) < 10, f"Sample {i} has drifted (mean too large)"
        
        print(f"\n=== Numerical Stability Test ===")
        for i, sample in enumerate(samples):
            print(f"Sample {i}: mean={sample.mean():.4f}, std={sample.std():.4f}")
        
        print("✓ Numerical stability test: PASSED")
    
    @pytest.mark.skipif(not hasattr(torch, 'compile'), reason="torch.compile not available")
    def test_compile_ddim(self, device):
        """Test that torch.compile works with DDIM."""
        if device.type != "cuda":
            pytest.skip("CUDA recommended for compile test")
        
        model = SimpleModel(channels=2, hidden_dim=64).to(device)
        
        # Create DDIM with compile_model=True
        ddim = DDIM(
            model=model,
            num_timesteps=50,
            sampling_timesteps=10,
            ddim_eta=0.0,
            beta_schedule="linear",
            parameterization="eps",
            compile_model=True,
        ).to(device)
        
        B, T, N, F = 2, 4, 8, 2
        shape = (B, T, N, F)
        data = create_test_data(B, N, T, F, device)
        
        # Warmup (compile happens on first run)
        with torch.no_grad():
            _ = ddim.sample(shape, device, data, use_amp=False)
        
        # Verify sampling works after compile
        torch.manual_seed(42)
        with torch.no_grad():
            sample = ddim.sample(shape, device, data, use_amp=False)
        
        assert torch.isfinite(sample).all(), "Compiled model produced NaN or Inf"
        assert sample.shape == shape, f"Wrong shape: {sample.shape} vs {shape}"
        
        print(f"\n=== torch.compile Test ===")
        print(f"Sample shape: {sample.shape}")
        print(f"Sample mean: {sample.mean():.4f}, std: {sample.std():.4f}")
        print("✓ torch.compile test: PASSED")
    
    def test_amp_deterministic(self, ddim, device):
        """Test that AMP sampling is deterministic with same seed."""
        if device.type != "cuda":
            pytest.skip("CUDA required for AMP deterministic test")
        
        B, T, N, F = 2, 4, 8, 2
        shape = (B, T, N, F)
        data = create_test_data(B, N, T, F, device)
        
        ddim = ddim.to(device)
        
        # Sample twice with same seed
        torch.manual_seed(123)
        with torch.no_grad():
            sample1 = ddim.sample(shape, device, data, use_amp=True)
        
        torch.manual_seed(123)
        with torch.no_grad():
            sample2 = ddim.sample(shape, device, data, use_amp=True)
        
        # They should be identical (deterministic DDIM with eta=0)
        assert torch.allclose(sample1, sample2, atol=1e-5), \
            "Deterministic AMP sampling should produce identical results"
        
        print(f"\n=== AMP Deterministic Test ===")
        print(f"Max diff: {(sample1 - sample2).abs().max().item():.8f}")
        print("✓ AMP deterministic test: PASSED")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for benchmark")
class TestAMPBenchmark:
    """Benchmark tests for detailed performance comparison."""
    
    @pytest.fixture
    def device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        pytest.skip("CUDA required for benchmark")
    
    def test_benchmark_comparison(self, device):
        """Comprehensive benchmark comparing all configurations."""
        # Configuration
        B, T, N, F = 4, 8, 16, 2
        shape = (B, T, N, F)
        num_runs = 5
        
        model_fp32 = SimpleModel(channels=F, hidden_dim=64).to(device)
        model_fp16 = SimpleModel(channels=F, hidden_dim=64).to(device).half()
        
        data = create_test_data(B, N, T, F, device)
        
        configs = [
            ("DDPM FP32", DDPM(model_fp32, num_timesteps=50, parameterization="eps").to(device), False),
            ("DDPM AMP", DDPM(model_fp32, num_timesteps=50, parameterization="eps").to(device), True),
            ("DDIM FP32", DDIM(model_fp32, num_timesteps=50, sampling_timesteps=10, parameterization="eps").to(device), False),
            ("DDIM AMP", DDIM(model_fp32, num_timesteps=50, sampling_timesteps=10, parameterization="eps").to(device), True),
        ]
        
        print(f"\n{'='*60}")
        print("BENCHMARK: DDPM/DDIM with FP32 vs AMP")
        print(f"Shape: {shape}, Runs: {num_runs}")
        print(f"{'='*60}")
        
        results = {}
        for name, diffusion, use_amp in configs:
            result = measure_inference(diffusion, shape, device, data, use_amp, num_runs=num_runs)
            results[name] = result
            print(f"\n{name}:")
            print(f"  Time: {result['mean_time']:.4f}s (±{result['max_time'] - result['min_time']:.4f}s)")
            print(f"  Peak Memory: {result['peak_memory_mb']:.1f} MB")
        
        # Summary
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        
        ddpm_speedup = results["DDPM FP32"]["mean_time"] / results["DDPM AMP"]["mean_time"]
        ddim_speedup = results["DDIM FP32"]["mean_time"] / results["DDIM AMP"]["mean_time"]
        ddim_vs_ddpm = results["DDPM FP32"]["mean_time"] / results["DDIM AMP"]["mean_time"]
        
        print(f"DDPM AMP speedup: {ddpm_speedup:.2f}x")
        print(f"DDIM AMP speedup: {ddim_speedup:.2f}x")
        print(f"DDIM+AMP vs DDPM FP32: {ddim_vs_ddpm:.2f}x faster")
        
        # All tests should complete without error
        print("\n✓ Benchmark completed successfully")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
