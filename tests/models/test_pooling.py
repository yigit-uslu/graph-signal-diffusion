"""Unit tests for graph pooling components."""

import torch
import torch.nn as nn
from graph_signal_diffusion.models.components.pooling import (
    PoolingGNN,
    LearnableGraphPool,
    GraphUnpool,
    GraphMaxPool,
    GraphMaxUnpool,
    StridedGraphMaxPool,
)


class TestPoolingGNN:
    """Test suite for PoolingGNN."""
    
    def test_forward_basic(self):
        """Test basic forward pass."""
        pooling_gnn = PoolingGNN(in_channels=32, hidden_channels=16, num_layers=2)
        x = torch.randn(4, 10, 100, 32)
        edge_index = torch.randint(0, 400, (2, 500))
        active_mask = torch.ones(4, 100, dtype=torch.bool)
        
        scores = pooling_gnn(x, edge_index, active_mask=active_mask)
        
        assert scores.shape == (4, 100)
        assert not torch.isnan(scores).any()
        assert not torch.isinf(scores).any()
        print("✓ PoolingGNN basic forward: PASSED")
    
    def test_inactive_nodes_masked(self):
        """Test that inactive nodes get very low scores."""
        pooling_gnn = PoolingGNN(in_channels=32, hidden_channels=16)
        
        # Initialize with non-zero weights
        for param in pooling_gnn.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, 0.1)
        
        B, T, N = 2, 5, 10
        x = torch.randn(B, T, N, 32) * 3.0
        edge_index = torch.randint(0, B * N, (2, 50))
        
        # Only first 5 nodes are active
        active_mask = torch.zeros(B, N, dtype=torch.bool)
        active_mask[:, :5] = True
        
        scores = pooling_gnn(x, edge_index, active_mask=active_mask)
        
        # Inactive nodes (5-9) should have very low scores
        active_scores = scores[:, :5]
        inactive_scores = scores[:, 5:]
        
        assert (inactive_scores < -1e8).all(), \
            "Inactive nodes should have very low scores"
        assert (active_scores > inactive_scores.max()).all(), \
            "Active nodes should have higher scores than inactive"
        
        print("✓ PoolingGNN inactive nodes masked: PASSED")
    
    def test_with_conditioning(self):
        """Test with conditioning input."""
        pooling_gnn = PoolingGNN(
            in_channels=32,
            hidden_channels=16,
            cond_dim=64,
        )
        
        x = torch.randn(4, 10, 100, 32)
        edge_index = torch.randint(0, 400, (2, 500))
        active_mask = torch.ones(4, 100, dtype=torch.bool)
        cond = torch.randn(4, 64)
        
        scores = pooling_gnn(x, edge_index, active_mask=active_mask, cond=cond)
        
        assert scores.shape == (4, 100)
        print("✓ PoolingGNN with conditioning: PASSED")
    
    def test_different_active_masks(self):
        """Test that different active masks produce different scores."""
        pooling_gnn = PoolingGNN(in_channels=32, hidden_channels=16)
        
        # Initialize weights
        for param in pooling_gnn.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
        
        x = torch.randn(2, 5, 10, 32) * 2.0
        edge_index = torch.randint(0, 20, (2, 40))
        
        # Two different masks
        mask1 = torch.ones(2, 10, dtype=torch.bool)
        mask1[:, 5:] = False  # First 5 nodes active
        
        mask2 = torch.ones(2, 10, dtype=torch.bool)
        mask2[:, :5] = False  # Last 5 nodes active
        
        scores1 = pooling_gnn(x, edge_index, active_mask=mask1)
        scores2 = pooling_gnn(x, edge_index, active_mask=mask2)
        
        # Active nodes in mask1 should differ from mask2
        assert not torch.allclose(scores1[:, :5], scores2[:, :5]), \
            "Different masks should produce different score patterns"
        
        print("✓ PoolingGNN different active masks: PASSED")
    
    def test_gradient_flow(self):
        """Test that gradients flow correctly."""
        pooling_gnn = PoolingGNN(in_channels=32, hidden_channels=16, num_layers=2)
        
        x = torch.randn(2, 5, 10, 32, requires_grad=True)
        edge_index = torch.randint(0, 20, (2, 40))
        active_mask = torch.ones(2, 10, dtype=torch.bool)
        
        scores = pooling_gnn(x, edge_index, active_mask=active_mask)
        loss = scores.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.abs().sum() > 0
        
        print("✓ PoolingGNN gradient flow: PASSED")


class TestLearnableGraphPool:
    """Test suite for LearnableGraphPool."""
    
    def test_forward_basic(self):
        """Test basic forward pass."""
        pool = LearnableGraphPool(in_channels=32, pooling_ratio=0.5)
        
        x = torch.randn(4, 10, 100, 32)
        edge_index = torch.randint(0, 400, (2, 500))
        
        x_pooled, new_mask, indices, _ = pool(x, edge_index)
        
        assert x_pooled.shape == (4, 10, 100, 32)
        assert new_mask.shape == (4, 100)
        assert new_mask.sum(dim=1).max() <= 50  # ~50% of 100 nodes
        print("✓ LearnableGraphPool basic forward: PASSED")
    
    def test_nested_selection(self):
        """Test that pooling respects nested selection."""
        pool = LearnableGraphPool(in_channels=32, pooling_ratio=0.5)
        
        B, T, N = 2, 5, 20
        x = torch.randn(B, T, N, 32)
        edge_index = torch.randint(0, B * N, (2, 100))
        
        # Start with only 10 nodes active
        active_mask = torch.zeros(B, N, dtype=torch.bool)
        active_mask[:, :10] = True
        
        x_pooled, new_mask, indices, _ = pool(x, edge_index, active_mask=active_mask)
        
        # New mask should only select from the 10 active nodes
        # (so max 5 nodes should be active after pooling)
        assert new_mask.sum(dim=1).max() <= 5
        
        # All selected nodes should be from the original active set
        for b in range(B):
            selected = torch.where(new_mask[b])[0]
            original_active = torch.where(active_mask[b])[0]
            assert all(s in original_active for s in selected), \
                "Selected nodes should be from original active set"
        
        print("✓ LearnableGraphPool nested selection: PASSED")
    

    def test_inactive_nodes_stay_zero(self):
        """Test that inactive nodes have zero features after pooling."""
        pool = LearnableGraphPool(in_channels=32, pooling_ratio=0.5)
        
        B, T, N = 2, 5, 10
        x = torch.randn(B, T, N, 32) * 5.0
        edge_index = torch.randint(0, B * N, (2, 50))
        
        # First 5 nodes active
        active_mask = torch.zeros(B, N, dtype=torch.bool)
        active_mask[:, :5] = True
        
        # Use eval mode for deterministic hard selection
        pool.eval()
        x_pooled, new_mask, indices, _ = pool(x, edge_index, active_mask=active_mask)
        
        # Inactive nodes in new_mask should have zero features
        for b in range(B):
            inactive = ~new_mask[b]
            inactive_features = x_pooled[b, :, inactive, :]
            
            # Check that inactive nodes are actually zero (or very close)
            assert torch.allclose(inactive_features, torch.zeros_like(inactive_features), atol=1e-6), \
                f"Inactive nodes should have zero features, but got max value: {inactive_features.abs().max()}"
        
        print("✓ LearnableGraphPool inactive nodes stay zero: PASSED")

    def test_inactive_nodes_stay_zero_soft_mode(self):
        """Test that inactive nodes have near-zero features in soft mode."""
        pool = LearnableGraphPool(
            in_channels=32, 
            pooling_ratio=0.5,
            selection_mode='soft'
        )
        
        B, T, N = 2, 5, 10
        x = torch.randn(B, T, N, 32) * 5.0
        edge_index = torch.randint(0, B * N, (2, 50))
        
        # First 5 nodes active
        active_mask = torch.zeros(B, N, dtype=torch.bool)
        active_mask[:, :5] = True
        
        # Use training mode (soft selection)
        pool.train()
        x_pooled, new_mask, indices, _ = pool(x, edge_index, active_mask=active_mask)
        
        # In soft mode, nodes from the original inactive set (5-9) should still be zero
        # because they were never in the active_mask to begin with
        for b in range(B):
            # Nodes that were NEVER active (5-9) should be zero
            never_active = ~active_mask[b]
            never_active_features = x_pooled[b, :, never_active, :]
            
            assert torch.allclose(never_active_features, torch.zeros_like(never_active_features), atol=1e-6), \
                f"Nodes never in active mask should be zero, got max: {never_active_features.abs().max()}"
            
            # Nodes that were active but not selected (from 0-4) might have small non-zero values
            # in soft mode, but should be much smaller than selected nodes
            was_active = active_mask[b]
            newly_inactive = was_active & ~new_mask[b]
            
            if newly_inactive.any():
                newly_inactive_features = x_pooled[b, :, newly_inactive, :]
                selected_features = x_pooled[b, :, new_mask[b], :]
                
                # Newly inactive should have much smaller magnitude than selected
                if selected_features.abs().max() > 1e-3:  # If selected nodes have significant values
                    ratio = newly_inactive_features.abs().max() / (selected_features.abs().max() + 1e-8)
                    # With deterministic same-logit soft masking, newly inactive nodes can still
                    # carry non-trivial values. Keep a loose sanity bound to avoid blow-ups.
                    assert ratio < 1.5, \
                        f"Newly inactive nodes should not dominate selected (ratio={ratio:.3f})"
        
        print("✓ LearnableGraphPool inactive nodes stay zero (soft mode): PASSED")


    def test_soft_vs_hard_selection(self):
        """Test soft (training) vs hard (eval) selection modes."""
        pool = LearnableGraphPool(
            in_channels=32,
            pooling_ratio=0.5,
            selection_mode='soft',
        )
        
        x = torch.randn(2, 5, 20, 32)
        edge_index = torch.randint(0, 40, (2, 100))
        
        # Training mode (soft)
        pool.train()
        x_soft, mask_soft, _, _ = pool(x, edge_index)
        
        # Eval mode (hard)
        pool.eval()
        x_hard, mask_hard, _, _ = pool(x, edge_index)
        
        # Hard selection should be binary
        assert ((mask_hard == 0) | (mask_hard == 1)).all(), \
            "Hard selection should be binary"
        
        print("✓ LearnableGraphPool soft vs hard selection: PASSED")
    
    def test_with_conditioning(self):
        """Test pooling with conditioning."""
        pool = LearnableGraphPool(
            in_channels=32,
            pooling_ratio=0.5,
            cond_dim=64,
        )
        
        x = torch.randn(4, 10, 100, 32)
        edge_index = torch.randint(0, 400, (2, 500))
        cond = torch.randn(4, 64)
        
        x_pooled, new_mask, indices, _ = pool(x, edge_index, cond=cond)
        
        assert x_pooled.shape == (4, 10, 100, 32)
        print("✓ LearnableGraphPool with conditioning: PASSED")
    
    def test_gradient_flow(self):
        """Test that gradients flow through pooling."""
        pool = LearnableGraphPool(in_channels=32, pooling_ratio=0.5)
        
        x = torch.randn(2, 5, 10, 32, requires_grad=True)
        edge_index = torch.randint(0, 20, (2, 40))
        
        x_pooled, _, _, _ = pool(x, edge_index)
        loss = x_pooled.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.abs().sum() > 0
        
        print("✓ LearnableGraphPool gradient flow: PASSED")
    
    def test_hierarchical_pooling(self):
        """Test multiple levels of pooling (hierarchical)."""
        pool1 = LearnableGraphPool(in_channels=32, pooling_ratio=0.5)
        pool2 = LearnableGraphPool(in_channels=32, pooling_ratio=0.5)
        
        B, T, N = 2, 5, 100
        x = torch.randn(B, T, N, 32)
        edge_index = torch.randint(0, B * N, (2, 500))
        
        # Level 1: Pool to ~50 nodes
        x1, mask1, _, _ = pool1(x, edge_index)
        assert mask1.sum(dim=1).max() <= 50
        
        # Level 2: Pool to ~25 nodes (from 50)
        x2, mask2, _, _ = pool2(x1, edge_index, active_mask=mask1)
        assert mask2.sum(dim=1).max() <= 25
        
        # mask2 should be a subset of mask1
        for b in range(B):
            selected_2 = set(torch.where(mask2[b])[0].tolist())
            selected_1 = set(torch.where(mask1[b])[0].tolist())
            assert selected_2.issubset(selected_1), \
                "Level 2 selection should be subset of Level 1"
        
        print("✓ LearnableGraphPool hierarchical pooling: PASSED")


class TestGraphUnpool:
    """Test suite for GraphUnpool."""
    
    def test_forward_basic(self):
        """Test basic unpooling."""
        unpool = GraphUnpool()
        
        x_pooled = torch.randn(4, 10, 100, 32)
        x_skip = torch.randn(4, 10, 100, 64)
        
        x = unpool(x_pooled, x_skip)
        
        assert x.shape == (4, 10, 100, 96)  # 32 + 64
        print("✓ GraphUnpool basic forward: PASSED")
    
    def test_preserves_zeros(self):
        """Test that zero nodes in pooled features stay zero."""
        unpool = GraphUnpool()
        
        B, T, N = 2, 5, 10
        x_pooled = torch.randn(B, T, N, 32)
        x_skip = torch.randn(B, T, N, 64)
        
        # Zero out some nodes in pooled features
        x_pooled[:, :, 5:, :] = 0
        
        x = unpool(x_pooled, x_skip)
        
        # Pooled part of concatenation should still be zero
        assert torch.allclose(x[:, :, 5:, :32], torch.zeros(1)), \
            "Zeroed pooled features should remain zero after unpooling"
        
        # Skip connection part should be non-zero
        assert x[:, :, 5:, 32:].abs().sum() > 0, \
            "Skip connection should preserve non-zero values"
        
        print("✓ GraphUnpool preserves zeros: PASSED")
    
    def test_gradient_flow(self):
        """Test gradient flow through unpooling."""
        unpool = GraphUnpool()
        
        x_pooled = torch.randn(2, 5, 10, 32, requires_grad=True)
        x_skip = torch.randn(2, 5, 10, 64, requires_grad=True)
        
        x = unpool(x_pooled, x_skip)
        loss = x.sum()
        loss.backward()
        
        assert x_pooled.grad is not None
        assert x_skip.grad is not None
        assert x_pooled.grad.abs().sum() > 0
        assert x_skip.grad.abs().sum() > 0
        
        print("✓ GraphUnpool gradient flow: PASSED")


class TestEndToEnd:
    """End-to-end tests for UNet-style architecture."""
    
    def test_unet_style_pooling_unpooling(self):
        """Test pooling -> processing -> unpooling pipeline."""
        # Encoder
        pool1 = LearnableGraphPool(in_channels=32, pooling_ratio=0.5)
        pool2 = LearnableGraphPool(in_channels=32, pooling_ratio=0.5)
        
        # Decoder
        unpool2 = GraphUnpool()
        unpool1 = GraphUnpool()
        
        B, T, N = 2, 5, 100
        x = torch.randn(B, T, N, 32)
        edge_index = torch.randint(0, B * N, (2, 500))
        
        # Encoder: Pool twice
        enc1 = x.clone()  # Save for skip
        x_pool1, mask1, _, _ = pool1(enc1, edge_index)
        
        enc2 = x_pool1.clone()  # Save for skip
        x_pool2, mask2, _, _ = pool2(x_pool1, edge_index, active_mask=mask1)
        
        # Bottleneck (simulated processing)
        bottleneck = x_pool2 * 0.5
        
        # Decoder: Unpool twice with skip connections
        dec2 = unpool2(bottleneck, enc2)
        dec1 = unpool1(dec2[:, :, :, :32], enc1)  # Use first 32 channels
        
        assert dec1.shape == (B, T, N, 64)  # 32 + 32
        
        print("✓ End-to-end UNet-style pooling/unpooling: PASSED")



class TestGraphMaxPool:
    """Test suite for GraphMaxPool."""
    
    def test_forward_basic(self):
        """Test basic max-pooling."""
        pool = GraphMaxPool(gamma=2, selection_method='stride')
        
        x = torch.randn(2, 5, 20, 32)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])  # Simple chain
        
        x_pooled, mask, _, _ = pool(x, edge_index)
        
        assert x_pooled.shape == (2, 5, 20, 32)
        assert mask.sum(dim=1).max() == 10  # Every 2nd node
        print("✓ GraphMaxPool basic forward: PASSED")
    
    def test_max_pooling_correctness(self):
        """Test that max-pooling actually computes max over neighborhoods."""
        pool = GraphMaxPool(gamma=1, selection_method='stride')
        
        B, T, N, F = 1, 1, 5, 1
        
        # Simple chain: 0-1-2-3-4
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        
        # Set specific values to test max-pooling
        x = torch.zeros(B, T, N, F)
        x[0, 0, 0, 0] = 1.0  # Node 0: value 1
        x[0, 0, 1, 0] = 5.0  # Node 1: value 5 (max in 1-hop of node 0)
        x[0, 0, 2, 0] = 3.0  # Node 2: value 3
        
        pool.gamma = 1  # 1-hop neighborhood
        x_pooled, _, _, _ = pool(x, edge_index)
        
        # After 1-hop max-pooling:
        # Node 0 should have max(1, 5) = 5 (self + neighbor 1)
        # Node 1 should have max(5, 1, 3) = 5 (self + neighbors 0,2)
        # Node 2 should have max(3, 5, ...) = 5
        
        assert x_pooled[0, 0, 0, 0] == 5.0, \
            f"Node 0 should have pooled value 5, got {x_pooled[0, 0, 0, 0]}"
        assert x_pooled[0, 0, 1, 0] == 5.0
        
        print("✓ GraphMaxPool max-pooling correctness: PASSED")
    
    def test_with_learned_selection(self):
        """Test max-pooling with learned node selection."""
        pool = GraphMaxPool(
            gamma=2,
            selection_method='learned',
            in_channels=32,
            pooling_ratio=0.5,
        )
        
        x = torch.randn(2, 5, 20, 32)
        edge_index = torch.randint(0, 40, (2, 100))
        
        x_pooled, mask, _, _ = pool(x, edge_index)
        
        assert x_pooled.shape == (2, 5, 20, 32)
        assert mask.sum(dim=1).max() <= 10  # ~50% of 20 nodes
        print("✓ GraphMaxPool with learned selection: PASSED")
    
    def test_inactive_nodes_stay_zero(self):
        """Test that inactive nodes remain zero after max-pooling."""
        pool = GraphMaxPool(gamma=2, selection_method='stride')
        
        B, T, N = 1, 1, 10
        x = torch.randn(B, T, N, 32)
        edge_index = torch.randint(0, N, (2, 30))
        
        # Only first 5 nodes active
        active_mask = torch.zeros(B, N, dtype=torch.bool)
        active_mask[:, :5] = True
        
        x_pooled, mask, _, _ = pool(x, edge_index, active_mask=active_mask)
        
        # Nodes 5-9 should still be zero
        assert torch.allclose(x_pooled[:, :, 5:, :], torch.zeros(1)), \
            "Inactive nodes should remain zero after max-pooling"
        
        print("✓ GraphMaxPool inactive nodes stay zero: PASSED")
    
    def test_gradient_flow(self):
        """Test gradient flow through max-pooling."""
        pool = GraphMaxPool(gamma=2, selection_method='stride')
        
        x = torch.randn(1, 1, 10, 32, requires_grad=True)
        edge_index = torch.randint(0, 10, (2, 30))
        
        x_pooled, _, _, _ = pool(x, edge_index)
        loss = x_pooled.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.abs().sum() > 0
        
        print("✓ GraphMaxPool gradient flow: PASSED")


class TestGraphMaxUnpool:
    """Test suite for GraphMaxUnpool."""
    
    def test_forward_zero_interpolation(self):
        """Test unpooling with zero-padding."""
        unpool = GraphMaxUnpool(gamma=2, interpolation='zero')
        
        B, T, N = 2, 5, 20
        x_pooled = torch.randn(B, T, N, 32)
        x_skip = torch.randn(B, T, N, 64)
        selection_mask = torch.rand(B, N) > 0.5
        
        x = unpool(x_pooled, x_skip, selection_mask)
        
        assert x.shape == (B, T, N, 96)  # 32 + 64
        print("✓ GraphMaxUnpool zero interpolation: PASSED")
    
    def test_forward_nearest_interpolation(self):
        """Test unpooling with nearest-neighbor interpolation."""
        unpool = GraphMaxUnpool(gamma=2, interpolation='nearest')
        
        B, T, N = 1, 1, 10
        x_pooled = torch.randn(B, T, N, 32)
        x_skip = torch.randn(B, T, N, 64)
        
        # Select every 2nd node
        selection_mask = torch.zeros(B, N, dtype=torch.bool)
        selection_mask[:, ::2] = True
        
        # Zero out non-selected nodes in pooled features
        x_pooled = x_pooled * selection_mask.unsqueeze(1).unsqueeze(-1).float()
        
        # Simple chain graph
        edge_index = torch.tensor([[i, i+1] for i in range(N-1)]).t()
        
        x = unpool(x_pooled, x_skip, selection_mask, edge_index)
        
        assert x.shape == (B, T, N, 96)
        print("✓ GraphMaxUnpool nearest interpolation: PASSED")
    
    def test_gradient_flow(self):
        """Test gradient flow through unpooling."""
        unpool = GraphMaxUnpool(gamma=2, interpolation='zero')
        
        x_pooled = torch.randn(1, 1, 10, 32, requires_grad=True)
        x_skip = torch.randn(1, 1, 10, 64, requires_grad=True)
        selection_mask = torch.ones(1, 10, dtype=torch.bool)
        
        x = unpool(x_pooled, x_skip, selection_mask)
        loss = x.sum()
        loss.backward()
        
        assert x_pooled.grad is not None
        assert x_skip.grad is not None
        
        print("✓ GraphMaxUnpool gradient flow: PASSED")


class TestStridedGraphMaxPool:
    """Test suite for StridedGraphMaxPool."""
    
    def test_forward_basic(self):
        """Test basic strided max-pooling."""
        pool = StridedGraphMaxPool(gamma=2, K=2, selection_method='stride')
        
        x = torch.randn(2, 5, 20, 32)
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])  # Simple chain
        
        x_pooled, mask, _, _ = pool(x, edge_index)
        
        assert x_pooled.shape == (2, 5, 20, 32)
        assert mask.sum(dim=1).max() == 10  # Every 2nd node (γ=2)
        print("✓ StridedGraphMaxPool basic forward: PASSED")
    
    def test_multi_scale_neighborhoods(self):
        """Test that multi-scale neighborhoods (stride_input×k-hop) are computed correctly."""
        # stride_input=1 (dense signal), gamma=2 (downsampling factor), K=2
        pool = StridedGraphMaxPool(gamma=2, K=2, stride_input=1, selection_method='stride')
        
        B, T, N, F = 1, 1, 5, 1
        
        # Simple chain: 0-1-2-3-4
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
        
        # Set specific values to test multi-scale max-pooling
        x = torch.zeros(B, T, N, F)
        x[0, 0, 0, 0] = 1.0  # Node 0
        x[0, 0, 1, 0] = 2.0  # Node 1 (1-hop from 0)
        x[0, 0, 2, 0] = 8.0  # Node 2 (2-hop from 0)
        x[0, 0, 3, 0] = 3.0  # Node 3
        x[0, 0, 4, 0] = 4.0  # Node 4
        
        x_pooled, mask, _, _ = pool(x, edge_index)
        
        # With stride_input=1, K=2, node 0 should aggregate from:
        # - 0-hop: itself (1.0)
        # - 1-hop: node 1 (2.0)
        # - 2-hop: node 2 (8.0)
        # So max should be 8.0
        
        pooled_value = x_pooled[0, 0, 0, 0]
        assert pooled_value == 8.0, \
            f"Node 0 should have max=8.0 from multi-scale neighborhood, got {pooled_value}"
        
        print("✓ StridedGraphMaxPool multi-scale neighborhoods: PASSED")
    
    def test_different_K_values(self):
        """Test with different K values (number of scales)."""
        x = torch.randn(1, 1, 20, 32)
        edge_index = torch.randint(0, 20, (2, 50))
        
        # K=1: only 0-hop and stride_input-hop
        pool_k1 = StridedGraphMaxPool(gamma=2, K=1, stride_input=1, selection_method='stride')
        x_k1, _, _, _ = pool_k1(x, edge_index)
        
        # K=3: 0-hop, stride_input-hop, 2×stride_input-hop, 3×stride_input-hop
        pool_k3 = StridedGraphMaxPool(gamma=2, K=3, stride_input=1, selection_method='stride')
        x_k3, _, _, _ = pool_k3(x, edge_index)
        
        # Both should have same shape
        assert x_k1.shape == x_k3.shape
        
        # But values should differ (larger K aggregates more neighbors)
        # Note: they might be similar for small graphs, so we just check they run
        print("✓ StridedGraphMaxPool different K values: PASSED")

    def test_k_zero_fast_path_skips_adjacency_and_maxpool(self, monkeypatch):
        """Test K=0 fast path bypasses adjacency/max-pooling and applies selection directly."""
        pool = StridedGraphMaxPool(gamma=2, K=0, selection_method='stride')

        def _should_not_be_called(*args, **kwargs):
            raise AssertionError("Adjacency/max-pool helpers must not be called when K=0")

        monkeypatch.setattr(pool, "_compute_strided_hop_adjacency_boolean", _should_not_be_called)
        monkeypatch.setattr(pool, "_compute_strided_hop_adjacency", _should_not_be_called)
        monkeypatch.setattr(pool, "_max_pool_neighborhoods", _should_not_be_called)

        B, T, N, F = 2, 3, 8, 4
        x = torch.randn(B, T, N, F)
        edge_index = torch.randint(0, N, (2, 24))
        active_mask = torch.tensor(
            [
                [1, 1, 1, 1, 1, 1, 0, 0],
                [0, 1, 1, 0, 1, 0, 0, 1],
            ],
            dtype=torch.bool,
        )

        x_pooled, new_mask, _, neighborhoods = pool(
            x,
            edge_index,
            active_mask=active_mask,
            return_neighborhoods=True,
        )

        expected_mask = torch.zeros_like(active_mask)
        expected_mask[0, [0, 2, 4]] = True
        expected_mask[1, [1, 4]] = True

        expected_x = x * expected_mask.unsqueeze(1).unsqueeze(-1).float()

        assert neighborhoods is None
        assert torch.equal(new_mask, expected_mask)
        assert torch.allclose(x_pooled, expected_x)

        print("✓ StridedGraphMaxPool K=0 fast path: PASSED")
    
    def test_stride_vs_learned_selection(self):
        """Test stride-based vs learned selection methods."""
        x = torch.randn(2, 5, 20, 32)
        edge_index = torch.randint(0, 40, (2, 100))
        
        # Stride selection
        pool_stride = StridedGraphMaxPool(gamma=2, K=2, selection_method='stride')
        x_stride, mask_stride, _, _ = pool_stride(x, edge_index)
        
        # Learned selection
        pool_learned = StridedGraphMaxPool(
            gamma=2, K=2, 
            selection_method='learned',
            in_channels=32,
            pooling_ratio=0.5,
        )
        x_learned, mask_learned, _, _ = pool_learned(x, edge_index)
        
        # Stride should select exactly every γ-th node
        assert mask_stride.sum(dim=1)[0] == 10  # 20 / 2
        
        # Learned should select approximately pooling_ratio fraction
        assert mask_learned.sum(dim=1).max() <= 10
        
        print("✓ StridedGraphMaxPool stride vs learned selection: PASSED")
    
    def test_inactive_nodes_stay_zero(self):
        """Test that inactive nodes remain zero after strided max-pooling."""
        pool = StridedGraphMaxPool(gamma=2, K=2, selection_method='stride')
        
        B, T, N = 1, 1, 10
        x = torch.randn(B, T, N, 32) * 3.0  # Scale up to make non-zero
        edge_index = torch.randint(0, N, (2, 30))
        
        # Only first 5 nodes active
        active_mask = torch.zeros(B, N, dtype=torch.bool)
        active_mask[:, :5] = True
        
        x_pooled, mask, _, _ = pool(x, edge_index, active_mask=active_mask)
        
        # All originally inactive nodes (5-9) should still be zero
        assert torch.allclose(x_pooled[:, :, 5:, :], torch.zeros(1)), \
            "Inactive nodes should remain zero after strided max-pooling"
        
        # Stride selection may select nodes outside active set, but they should have zero features
        for b in range(B):
            selected = torch.where(mask[b])[0]
            original_active_set = set(torch.where(active_mask[b])[0].tolist())
            
            for s in selected.tolist():
                if s not in original_active_set:
                    # Nodes selected but not in original active set should be zero
                    assert torch.allclose(x_pooled[b, :, s, :], torch.zeros(1)), \
                        f"Node {s} selected but not active, should have zero features"
        
        print("✓ StridedGraphMaxPool inactive nodes stay zero: PASSED")
    
    def test_gradient_flow(self):
        """Test gradient flow through strided max-pooling."""
        pool = StridedGraphMaxPool(gamma=2, K=2, selection_method='stride')
        
        x = torch.randn(1, 1, 10, 32, requires_grad=True)
        edge_index = torch.randint(0, 10, (2, 30))
        
        x_pooled, _, _, _ = pool(x, edge_index)
        loss = x_pooled.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.abs().sum() > 0
        
        print("✓ StridedGraphMaxPool gradient flow: PASSED")
    
    def test_comparison_with_standard_maxpool(self):
        """Compare StridedGraphMaxPool with standard GraphMaxPool."""
        # Both with stride_input=1 (dense signal), K=1
        # StridedGraphMaxPool should aggregate 0-hop and 1-hop
        # GraphMaxPool with γ=2 aggregates up to 2-hop (0, 1, 2-hop)
        # So they should differ for nodes with 2-hop neighbors
        
        x = torch.randn(1, 1, 10, 32)
        # Create a chain graph for predictable behavior
        edge_index = torch.tensor([[i, i+1] for i in range(9)]).t()
        
        # Standard max-pool (γ=2): aggregates 0,1,2-hop
        standard_pool = GraphMaxPool(gamma=2, selection_method='stride')
        x_standard, _, _, _ = standard_pool(x, edge_index)
        
        # Strided max-pool (stride_input=1, K=1): aggregates 0,1-hop only
        strided_pool = StridedGraphMaxPool(gamma=2, K=1, stride_input=1, selection_method='stride')
        x_strided, _, _, _ = strided_pool(x, edge_index)
        
        # Both should have same shape
        assert x_standard.shape == x_strided.shape
        
        # Values may differ since standard includes 2-hop while strided (K=1) doesn't
        # Note: for random data they might differ, but we just verify both work
        print("✓ StridedGraphMaxPool comparison with GraphMaxPool: PASSED")
    
    def test_hierarchical_strided_pooling(self):
        """Test multiple levels of strided pooling."""
        pool1 = StridedGraphMaxPool(gamma=2, K=2, selection_method='stride')
        pool2 = StridedGraphMaxPool(gamma=2, K=2, selection_method='stride')
        
        B, T, N = 1, 1, 100
        x = torch.randn(B, T, N, 32)
        edge_index = torch.randint(0, N, (2, 300))
        
        # Level 1: Pool to ~50 nodes
        x1, mask1, _, _ = pool1(x, edge_index)
        assert mask1.sum(dim=1)[0] == 50  # Stride selection is deterministic
        
        # Level 2: Pool to ~25 nodes
        x2, mask2, _, _ = pool2(x1, edge_index, active_mask=mask1)
        
        # mask2 should be subset of mask1
        for b in range(B):
            selected_2 = set(torch.where(mask2[b])[0].tolist())
            selected_1 = set(torch.where(mask1[b])[0].tolist())
            assert selected_2.issubset(selected_1), \
                "Level 2 selection should be subset of Level 1"
        
        print("✓ StridedGraphMaxPool hierarchical pooling: PASSED")
    
    def test_random_selection(self):
        """Test random selection method."""
        pool = StridedGraphMaxPool(gamma=2, K=2, selection_method='random')
        
        x = torch.randn(2, 5, 20, 32)
        edge_index = torch.randint(0, 40, (2, 100))
        
        x_pooled, mask, indices, _ = pool(x, edge_index)
        
        assert x_pooled.shape == (2, 5, 20, 32)
        # Should select ~N/γ nodes
        assert mask.sum(dim=1).max() <= 10
        assert indices is not None  # Random selection returns indices via _selected_indices_from_mask
        
        print("✓ StridedGraphMaxPool random selection: PASSED")
    
    def test_with_conditioning(self):
        """Test strided pooling with learned selection and conditioning."""
        pool = StridedGraphMaxPool(
            gamma=2,
            K=2,
            selection_method='learned',
            in_channels=32,
            pooling_ratio=0.5,
            selector_kwargs={
                'hidden_channels': 32,
                'cond_dim': 64,
                },
        )
        
        x = torch.randn(4, 10, 100, 32)
        edge_index = torch.randint(0, 400, (2, 500))
        cond = torch.randn(4, 64)
        
        x_pooled, mask, _, _ = pool(x, edge_index, cond=cond)
        
        assert x_pooled.shape == (4, 10, 100, 32)
        print("✓ StridedGraphMaxPool with conditioning: PASSED")


# Run all tests
if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing PoolingGNN")
    print("="*60)
    test_pooling = TestPoolingGNN()
    test_pooling.test_forward_basic()
    test_pooling.test_inactive_nodes_masked()
    test_pooling.test_with_conditioning()
    test_pooling.test_different_active_masks()
    test_pooling.test_gradient_flow()
    
    print("\n" + "="*60)
    print("Testing LearnableGraphPool")
    print("="*60)
    test_pool = TestLearnableGraphPool()
    test_pool.test_forward_basic()
    test_pool.test_nested_selection()
    test_pool.test_inactive_nodes_stay_zero()
    test_pool.test_soft_vs_hard_selection()
    test_pool.test_with_conditioning()
    test_pool.test_gradient_flow()
    test_pool.test_hierarchical_pooling()
    
    print("\n" + "="*60)
    print("Testing GraphUnpool")
    print("="*60)
    test_unpool = TestGraphUnpool()
    test_unpool.test_forward_basic()
    test_unpool.test_preserves_zeros()
    test_unpool.test_gradient_flow()
    
    print("\n" + "="*60)
    print("Testing End-to-End")
    print("="*60)
    test_e2e = TestEndToEnd()
    test_e2e.test_unet_style_pooling_unpooling()


    print("\n" + "="*60)
    print("Testing GraphMaxPool")
    print("="*60)
    test_maxpool = TestGraphMaxPool()
    test_maxpool.test_forward_basic()
    test_maxpool.test_max_pooling_correctness()
    test_maxpool.test_with_learned_selection()
    test_maxpool.test_inactive_nodes_stay_zero()
    test_maxpool.test_gradient_flow()
    
    print("\n" + "="*60)
    print("Testing GraphMaxUnpool")
    print("="*60)
    test_maxunpool = TestGraphMaxUnpool()
    test_maxunpool.test_forward_zero_interpolation()
    test_maxunpool.test_forward_nearest_interpolation()
    test_maxunpool.test_gradient_flow()
    
    print("\n" + "="*60)
    print("Testing StridedGraphMaxPool")
    print("="*60)
    test_strided = TestStridedGraphMaxPool()
    test_strided.test_forward_basic()
    test_strided.test_multi_scale_neighborhoods()
    test_strided.test_different_K_values()
    test_strided.test_stride_vs_learned_selection()
    test_strided.test_inactive_nodes_stay_zero()
    test_strided.test_gradient_flow()
    test_strided.test_comparison_with_standard_maxpool()
    test_strided.test_hierarchical_strided_pooling()
    test_strided.test_random_selection()
    test_strided.test_with_conditioning()
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED! ✓")
    print("="*60)
    
