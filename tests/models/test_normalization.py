"""Unit tests for normalization components."""

import pytest
import torch
import torch.nn as nn

# ✅ Use absolute imports (not relative)
from graph_signal_diffusion.models.components.normalization import (
    LayerNorm,
    GroupNorm,
    InstanceNorm,
    GraphNorm,
    AdaptiveLayerNorm,
    AdaptiveGroupNorm,
    get_normalization,
)


class TestLayerNorm:
    """Test suite for LayerNorm."""
    
    def test_forward_4d(self):
        """Test with 4D input (B, T, N, F)."""
        norm = LayerNorm(num_features=64)
        x = torch.randn(4, 10, 100, 64)
        out = norm(x)
        
        assert out.shape == (4, 10, 100, 64)
        print("✓ LayerNorm 4D input: PASSED")
    
    def test_forward_3d(self):
        """Test with 3D input (B*N, T, F)."""
        norm = LayerNorm(num_features=64)
        x = torch.randn(400, 10, 64)
        out = norm(x)
        
        assert out.shape == (400, 10, 64)
        print("✓ LayerNorm 3D input: PASSED")
    
    def test_forward_2d(self):
        """Test with 2D input (B*N, F)."""
        norm = LayerNorm(num_features=64)
        x = torch.randn(400, 64)
        out = norm(x)
        
        assert out.shape == (400, 64)
        print("✓ LayerNorm 2D input: PASSED")
    
    def test_normalization_properties(self):
        """Test that output is normalized."""
        norm = LayerNorm(num_features=64, elementwise_affine=False)
        x = torch.randn(4, 10, 100, 64) * 10 + 5  # Non-normalized
        out = norm(x)
        
        # Check mean ≈ 0 and std ≈ 1 along feature dimension
        mean = out.mean(dim=-1)
        # Use unbiased=False to match layer_norm's computation
        std = out.std(dim=-1, unbiased=False)
        
        assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5), \
            f"LayerNorm failed to center properly. Max mean: {mean.abs().max()}"
        assert torch.allclose(std, torch.ones_like(std), atol=1e-4), \
            f"LayerNorm failed to normalize properly. Std range: [{std.min()}, {std.max()}]"
        print("✓ LayerNorm normalization properties: PASSED")
    
    def test_affine_parameters(self):
        """Test learnable affine parameters."""
        norm = LayerNorm(num_features=64, elementwise_affine=True)
        
        assert norm.weight is not None
        assert norm.bias is not None
        assert norm.weight.shape == (64,)
        assert norm.bias.shape == (64,)
        print("✓ LayerNorm affine parameters: PASSED")
    
    def test_no_affine_parameters(self):
        """Test that no parameters exist when elementwise_affine=False."""
        norm = LayerNorm(num_features=64, elementwise_affine=False)
        
        assert norm.weight is None
        assert norm.bias is None
        print("✓ LayerNorm no affine parameters: PASSED")


class TestGroupNorm:
    """Test suite for GroupNorm."""
    
    def test_forward_4d(self):
        """Test with 4D input."""
        norm = GroupNorm(num_features=64, num_groups=8)
        x = torch.randn(4, 10, 100, 64)
        out = norm(x)
        
        assert out.shape == (4, 10, 100, 64)
        print("✓ GroupNorm 4D input: PASSED")
    
    def test_forward_3d(self):
        """Test with 3D input."""
        norm = GroupNorm(num_features=64, num_groups=8)
        x = torch.randn(400, 10, 64)
        out = norm(x)
        
        assert out.shape == (400, 10, 64)
        print("✓ GroupNorm 3D input: PASSED")
    
    def test_forward_2d(self):
        """Test with 2D input."""
        norm = GroupNorm(num_features=64, num_groups=8)
        x = torch.randn(400, 64)
        out = norm(x)
        
        assert out.shape == (400, 64)
        print("✓ GroupNorm 2D input: PASSED")
    
    def test_group_divisibility(self):
        """Test that num_features must be divisible by num_groups."""
        with pytest.raises(AssertionError):
            GroupNorm(num_features=64, num_groups=7)  # 64 not divisible by 7
        print("✓ GroupNorm divisibility check: PASSED")
    
    def test_different_group_sizes(self):
        """Test different numbers of groups."""
        for num_groups in [1, 2, 4, 8, 16, 32]:
            norm = GroupNorm(num_features=64, num_groups=num_groups)
            x = torch.randn(4, 10, 100, 64)
            out = norm(x)
            assert out.shape == (4, 10, 100, 64)
        print("✓ GroupNorm different group sizes: PASSED")
    
    def test_independence_from_batch_size(self):
        """Test that output doesn't depend on batch size."""
        norm = GroupNorm(num_features=64, num_groups=8)
        
        # Single sample
        x1 = torch.randn(1, 10, 100, 64)
        out1 = norm(x1)
        
        # Batch of 8
        x8 = x1.repeat(8, 1, 1, 1)
        out8 = norm(x8)
        
        # First sample should be identical
        torch.testing.assert_close(out1[0], out8[0], rtol=1e-5, atol=1e-5)
        print("✓ GroupNorm batch independence: PASSED")
    
    def test_normalization_properties(self):
        """Test basic normalization properties."""
        norm = GroupNorm(num_features=64, num_groups=8, affine=False)
        x = torch.randn(4, 10, 100, 64) * 5 + 10  # Non-normalized
        out = norm(x)
        
        # Output should have reasonable statistics
        assert out.mean().abs() < 0.1, f"GroupNorm mean too large: {out.mean()}"
        assert 0.8 < out.std() < 1.2, f"GroupNorm std out of range: {out.std()}"
        print("✓ GroupNorm normalization properties: PASSED")


class TestInstanceNorm:
    """Test suite for InstanceNorm."""
    
    def test_forward_4d(self):
        """Test with 4D input."""
        norm = InstanceNorm(num_features=64)
        x = torch.randn(4, 10, 100, 64)
        out = norm(x)
        
        assert out.shape == (4, 10, 100, 64)
        print("✓ InstanceNorm 4D input: PASSED")
    
    def test_forward_3d(self):
        """Test with 3D input."""
        norm = InstanceNorm(num_features=64)
        x = torch.randn(400, 10, 64)
        out = norm(x)
        
        assert out.shape == (400, 10, 64)
        print("✓ InstanceNorm 3D input: PASSED")
    
    def test_per_instance_normalization(self):
        """Test that each instance is normalized independently."""
        norm = InstanceNorm(num_features=64, affine=False)
        x = torch.randn(4, 10, 100, 64)
        out = norm(x)
        
        # Each batch item should have its own statistics
        for b in range(4):
            sample = out[b]  # (T, N, F)
            # Flatten spatial-temporal dimensions
            sample_flat = sample.reshape(-1, 64)
            mean = sample_flat.mean(dim=0)
            std = sample_flat.std(dim=0, unbiased=False)
            
            # Should be approximately normalized
            # (more lenient because of finite sample effects)
            assert torch.allclose(mean, torch.zeros_like(mean), atol=0.15), \
                f"InstanceNorm mean too large for batch {b}: {mean.abs().max()}"
            assert torch.allclose(std, torch.ones_like(std), atol=0.15), \
                f"InstanceNorm std out of range for batch {b}: [{std.min()}, {std.max()}]"
        print("✓ InstanceNorm per-instance properties: PASSED")


class TestGraphNorm:
    """Test suite for GraphNorm."""
    
    def test_forward_4d(self):
        """Test with 4D input."""
        norm = GraphNorm(num_features=64)
        x = torch.randn(4, 10, 100, 64)
        out = norm(x)
        
        assert out.shape == (4, 10, 100, 64)
        print("✓ GraphNorm 4D input: PASSED")
    
    def test_forward_3d(self):
        """Test with 3D input."""
        norm = GraphNorm(num_features=64)
        x = torch.randn(400, 10, 64)
        out = norm(x)
        
        assert out.shape == (400, 10, 64)
        print("✓ GraphNorm 3D input: PASSED")
    
    def test_normalization_over_nodes(self):
        """Test that normalization is computed over node dimension."""
        norm = GraphNorm(num_features=64, affine=False)
        x = torch.randn(4, 10, 100, 64)
        out = norm(x)
        
        # For each (batch, time), check normalization over nodes
        for b in range(2):  # Test first 2 batches
            for t in range(0, 10, 5):  # Test every 5th timestep
                nodes = out[b, t]  # (N, F)
                mean = nodes.mean(dim=0)
                std = nodes.std(dim=0, unbiased=False)
                
                # Should be normalized over nodes
                assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-4), \
                    f"GraphNorm mean too large at b={b}, t={t}: {mean.abs().max()}"
                assert torch.allclose(std, torch.ones_like(std), atol=0.15), \
                    f"GraphNorm std out of range at b={b}, t={t}: [{std.min()}, {std.max()}]"
        print("✓ GraphNorm normalization over nodes: PASSED")


class TestAdaptiveLayerNorm:
    """Test suite for AdaptiveLayerNorm."""
    
    def test_forward_4d_2d_cond(self):
        """Test with 4D input and 2D conditioning."""
        norm = AdaptiveLayerNorm(num_features=64, cond_dim=128)
        x = torch.randn(4, 10, 100, 64)  # (B, T, N, F)
        cond = torch.randn(4, 128)  # (B, cond_dim)
        
        out = norm(x, cond)
        assert out.shape == (4, 10, 100, 64)
        print("✓ AdaptiveLayerNorm 4D input + 2D conditioning: PASSED")
    
    def test_forward_4d_3d_cond(self):
        """Test with 4D input and 3D conditioning (per-node)."""
        norm = AdaptiveLayerNorm(num_features=64, cond_dim=128)
        x = torch.randn(4, 10, 100, 64)  # (B, T, N, F)
        cond = torch.randn(4, 100, 128)  # (B, N, cond_dim)
        
        out = norm(x, cond)
        assert out.shape == (4, 10, 100, 64)
        print("✓ AdaptiveLayerNorm 4D input + 3D conditioning: PASSED")
    
    def test_forward_3d(self):
        """Test with 3D input."""
        norm = AdaptiveLayerNorm(num_features=64, cond_dim=128)
        x = torch.randn(400, 10, 64)  # (B*N, T, F)
        cond = torch.randn(400, 128)  # (B*N, cond_dim)
        
        out = norm(x, cond)
        assert out.shape == (400, 10, 64)
        print("✓ AdaptiveLayerNorm 3D input: PASSED")
    
    def test_modulation_effect(self):
        """Test that conditioning actually modulates the output."""
        norm = AdaptiveLayerNorm(num_features=64, cond_dim=128)
        x = torch.randn(4, 10, 100, 64)
        
        # Different conditioning should give different outputs
        cond1 = torch.randn(4, 128)
        cond2 = torch.randn(4, 128)
        
        out1 = norm(x, cond1)
        out2 = norm(x, cond2)
        
        # Outputs should be different
        diff = (out1 - out2).abs().mean()
        assert diff > 0.01, f"Conditioning has no effect! Difference: {diff}"
        print("✓ AdaptiveLayerNorm modulation effect: PASSED")
    
    def test_gradients_flow_to_conditioning(self):
        """Test that gradients flow through conditioning."""
        norm = AdaptiveLayerNorm(num_features=64, cond_dim=128)
        x = torch.randn(4, 10, 100, 64, requires_grad=True)
        cond = torch.randn(4, 128, requires_grad=True)
        
        out = norm(x, cond)
        loss = out.sum()
        loss.backward()
        
        # Both x and cond should have gradients
        assert cond.grad is not None, "No gradient for conditioning!"
        assert x.grad is not None, "No gradient for input!"
        assert cond.grad.abs().sum() > 0, "Zero gradient for conditioning!"
        print("✓ AdaptiveLayerNorm gradient flow: PASSED")


class TestAdaptiveGroupNorm:
    """Test suite for AdaptiveGroupNorm."""
    
    def test_forward(self):
        """Test basic forward pass."""
        norm = AdaptiveGroupNorm(num_features=64, num_groups=8, cond_dim=128)
        x = torch.randn(4, 10, 100, 64)
        cond = torch.randn(4, 128)
        
        out = norm(x, cond)
        assert out.shape == (4, 10, 100, 64)
        print("✓ AdaptiveGroupNorm forward: PASSED")
    
    def test_with_different_batch_sizes(self):
        """Test independence from batch size."""
        norm = AdaptiveGroupNorm(num_features=64, num_groups=8, cond_dim=128)
        
        # Different batch sizes should work
        for batch_size in [1, 4, 16]:
            x = torch.randn(batch_size, 10, 100, 64)
            cond = torch.randn(batch_size, 128)
            out = norm(x, cond)
            assert out.shape == (batch_size, 10, 100, 64)
        print("✓ AdaptiveGroupNorm different batch sizes: PASSED")
    
    def test_time_conditioning_varies_output(self):
        """Test that different timesteps produce different outputs."""
        norm = AdaptiveGroupNorm(num_features=64, num_groups=8, cond_dim=128)
        x = torch.randn(4, 10, 100, 64)
        
        # Simulate different timestep embeddings
        time_emb_early = torch.randn(4, 128)
        time_emb_late = torch.randn(4, 128)
        
        out_early = norm(x, time_emb_early)
        out_late = norm(x, time_emb_late)
        
        # Different timesteps should produce different outputs
        diff = (out_early - out_late).abs().mean()
        assert diff > 0.01, f"Time conditioning has no effect! Difference: {diff}"
        print("✓ AdaptiveGroupNorm time-varying output: PASSED")
    
    def test_gradients_flow(self):
        """Test gradient flow through adaptive norm."""
        norm = AdaptiveGroupNorm(num_features=64, num_groups=8, cond_dim=128)
        x = torch.randn(4, 10, 100, 64, requires_grad=True)
        cond = torch.randn(4, 128, requires_grad=True)
        
        out = norm(x, cond)
        loss = out.sum()
        loss.backward()
        
        assert cond.grad is not None
        assert x.grad is not None
        assert cond.grad.abs().sum() > 0
        print("✓ AdaptiveGroupNorm gradient flow: PASSED")


class TestGetNormalization:
    """Test suite for normalization factory function."""
    
    def test_all_norm_types(self):
        """Test that all normalization types can be created."""
        for norm_type in ['layer', 'group', 'instance', 'graph', 'none']:
            norm = get_normalization(norm_type, num_features=64)
            assert norm is not None
            
            x = torch.randn(4, 10, 100, 64)
            out = norm(x)
            assert out.shape == (4, 10, 100, 64)
        print("✓ get_normalization all types: PASSED")
    
    def test_invalid_norm_type(self):
        """Test that invalid norm_type raises error."""
        with pytest.raises(ValueError):
            get_normalization('invalid', num_features=64)
        print("✓ get_normalization invalid type handling: PASSED")
    
    def test_group_norm_parameters(self):
        """Test that GroupNorm gets correct num_groups."""
        norm = get_normalization('group', num_features=64, num_groups=16)
        assert isinstance(norm, GroupNorm)
        assert norm.num_groups == 16
        print("✓ get_normalization GroupNorm parameters: PASSED")


# Run all tests
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing LayerNorm")
    print("="*60)
    test_ln = TestLayerNorm()
    test_ln.test_forward_4d()
    test_ln.test_forward_3d()
    test_ln.test_forward_2d()
    test_ln.test_normalization_properties()
    test_ln.test_affine_parameters()
    test_ln.test_no_affine_parameters()
    
    print("\n" + "="*60)
    print("Testing GroupNorm")
    print("="*60)
    test_gn = TestGroupNorm()
    test_gn.test_forward_4d()
    test_gn.test_forward_3d()
    test_gn.test_forward_2d()
    test_gn.test_group_divisibility()
    test_gn.test_different_group_sizes()
    test_gn.test_independence_from_batch_size()
    test_gn.test_normalization_properties()
    
    print("\n" + "="*60)
    print("Testing InstanceNorm")
    print("="*60)
    test_in = TestInstanceNorm()
    test_in.test_forward_4d()
    test_in.test_forward_3d()
    test_in.test_per_instance_normalization()
    
    print("\n" + "="*60)
    print("Testing GraphNorm")
    print("="*60)
    test_grnorm = TestGraphNorm()
    test_grnorm.test_forward_4d()
    test_grnorm.test_forward_3d()
    test_grnorm.test_normalization_over_nodes()
    
    print("\n" + "="*60)
    print("Testing AdaptiveLayerNorm")
    print("="*60)
    test_aln = TestAdaptiveLayerNorm()
    test_aln.test_forward_4d_2d_cond()
    test_aln.test_forward_4d_3d_cond()
    test_aln.test_forward_3d()
    test_aln.test_modulation_effect()
    test_aln.test_gradients_flow_to_conditioning()
    
    print("\n" + "="*60)
    print("Testing AdaptiveGroupNorm")
    print("="*60)
    test_agn = TestAdaptiveGroupNorm()
    test_agn.test_forward()
    test_agn.test_with_different_batch_sizes()
    test_agn.test_time_conditioning_varies_output()
    test_agn.test_gradients_flow()
    
    print("\n" + "="*60)
    print("Testing get_normalization Factory")
    print("="*60)
    test_factory = TestGetNormalization()
    test_factory.test_all_norm_types()
    test_factory.test_invalid_norm_type()
    test_factory.test_group_norm_parameters()
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED! ✓")
    print("="*60)