"""Unit tests for UGNN (U-Net Graph Neural Network)."""
import pytest
import torch
import torch.nn as nn
from graph_signal_diffusion.models.ugnn import (
    EncoderBlock,
    DecoderBlock,
    UGNNEncoder,
    UGNNDecoder,
    UGNN,
    GNNConfig,
    PoolingConfig,
    UpsamplingConfig,
    EmbeddingConfig,
    UGNNConfig,
)


class TestEncoderBlock:
    """Test suite for EncoderBlock."""
    
    def test_basic_forward(self):
        """Test basic forward pass without conditioning."""
        gnn_config = GNNConfig()
        pooling_config = PoolingConfig(gamma=2)
        embedding_config = EmbeddingConfig(time_embed_dim=128)
        
        block = EncoderBlock(
            in_channels=32,
            out_channels=64,
            stride_pre=1,
            stride_post=2,
            gnn_config=gnn_config,
            pooling_config=pooling_config,
            embedding_config=embedding_config,
        )
        
        B, T, N = 4, 10, 100
        x = torch.randn(B, T, N, 32)
        timesteps = torch.randint(0, 1000, (B,))
        time_emb = torch.randn(B, 128)
        edge_index = torch.randint(0, B * N, (2, 500))
        
        x_pooled, new_mask, x_skip, _ = block(
            x=x,
            timesteps=timesteps,
            edge_index=edge_index,
            time_emb=time_emb,
        )
        
        # Check shapes
        assert x_pooled.shape[0] == B
        assert x_pooled.shape[1] == T
        assert x_pooled.shape[2] == N  # Same N, but fewer active
        assert x_pooled.shape[3] == 64
        
        assert new_mask.shape == (B, N)
        assert x_skip.shape == (B, T, N, 64)
        
        # Check that pooling reduced active nodes
        num_active = new_mask.sum(dim=1).float().mean()
        assert num_active < N, f"Pooling should reduce active nodes: {num_active} vs {N}"
        
        print("✓ EncoderBlock basic forward: PASSED")
    
    def test_with_conditioning(self):
        """Test forward pass with conditional embeddings."""
        gnn_config = GNNConfig()
        pooling_config = PoolingConfig(gamma=2)
        embedding_config = EmbeddingConfig(
            time_embed_dim=128,
            cond_channels=8,
            cond_embed_dim=32,
        )
        
        block = EncoderBlock(
            in_channels=32,
            out_channels=64,
            stride_pre=1,
            stride_post=2,
            gnn_config=gnn_config,
            pooling_config=pooling_config,
            embedding_config=embedding_config,
        )
        
        B, T, N = 2, 5, 50
        x = torch.randn(B, T, N, 32)
        timesteps = torch.randint(0, 1000, (B,))
        time_emb = torch.randn(B, 128)
        # EncoderBlock expects pre-computed conditional embeddings
        cond_emb = torch.randn(B, N, 32)  # Already encoded
        edge_index = torch.randint(0, B * N, (2, 200))
        
        x_pooled, new_mask, x_skip, _ = block(
            x=x,
            timesteps=timesteps,
            edge_index=edge_index,
            time_emb=time_emb,
            cond_emb=cond_emb,  # Pass pre-computed embeddings
        )
        
        assert x_pooled.shape == (B, T, N, 64)
        assert new_mask.shape == (B, N)
        assert x_skip.shape == (B, T, N, 64)
        
        print("✓ EncoderBlock with conditioning: PASSED")
    
    def test_with_active_mask(self):
        """Test that active mask is properly respected."""
        gnn_config = GNNConfig()
        pooling_config = PoolingConfig(gamma=2)
        embedding_config = EmbeddingConfig(time_embed_dim=64)
        
        block = EncoderBlock(
            in_channels=16,
            out_channels=32,
            stride_pre=1,
            stride_post=2,
            gnn_config=gnn_config,
            pooling_config=pooling_config,
            embedding_config=embedding_config,
        )
        
        B, T, N = 2, 3, 20
        x = torch.randn(B, T, N, 16)
        timesteps = torch.randint(0, 1000, (B,))
        time_emb = torch.randn(B, 64)
        edge_index = torch.randint(0, B * N, (2, 80))
        
        # Create active mask - only half the nodes active
        active_mask = torch.zeros(B, N, dtype=torch.bool)
        active_mask[:, :N//2] = True
        
        x_pooled, new_mask, x_skip, _ = block(
            x=x,
            timesteps=timesteps,
            edge_index=edge_index,
            time_emb=time_emb,
            active_mask=active_mask,
        )
        
        # New mask should have even fewer active nodes
        new_active_count = new_mask.sum(dim=1).float().mean()
        old_active_count = active_mask.sum(dim=1).float().mean()
        
        assert new_active_count <= old_active_count, \
            f"Pooling should reduce active nodes: {new_active_count} vs {old_active_count}"
        
        print("✓ EncoderBlock with active mask: PASSED")
    
    def test_gradient_flow(self):
        """Test that gradients flow correctly."""
        gnn_config = GNNConfig()
        pooling_config = PoolingConfig(gamma=2)
        embedding_config = EmbeddingConfig(time_embed_dim=64)
        
        block = EncoderBlock(
            in_channels=16,
            out_channels=32,
            stride_pre=1,
            stride_post=2,
            gnn_config=gnn_config,
            pooling_config=pooling_config,
            embedding_config=embedding_config,
        )
        
        B, T, N = 2, 3, 20
        x = torch.randn(B, T, N, 16, requires_grad=True)
        timesteps = torch.randint(0, 1000, (B,))
        time_emb = torch.randn(B, 64, requires_grad=True)
        edge_index = torch.randint(0, B * N, (2, 80))
        
        x_pooled, new_mask, x_skip, _ = block(
            x=x,
            timesteps=timesteps,
            edge_index=edge_index,
            time_emb=time_emb,
        )
        
        loss = x_pooled.sum()
        loss.backward()
        
        assert x.grad is not None
        assert time_emb.grad is not None
        assert x.grad.abs().sum() > 0
        assert time_emb.grad.abs().sum() > 0
        
        print("✓ EncoderBlock gradient flow: PASSED")


class TestUGNNEncoder:
    """Test suite for UGNNEncoder (contracting path)."""
    
    def test_basic_encoder(self):
        """Test encoder with multiple levels."""
        config = UGNNConfig(
            in_channels=16,
            out_channels=16,
            base_channels=32,
            channel_multipliers=[1, 2, 4],
            embedding_config=EmbeddingConfig(time_embed_dim=64),
            pooling_config=PoolingConfig(gamma=2),
        )
        encoder = UGNNEncoder(
            in_channels=16,
            config=config,
        )
        
        B, T, N = 2, 5, 64
        x = torch.randn(B, T, N, 16)
        timesteps = torch.randint(0, 1000, (B,))
        time_emb = torch.randn(B, 64)
        edge_index = torch.randint(0, B * N, (2, 300))
        
        _, skip_features, active_masks, _ = encoder(
            x=x,
            timesteps=timesteps,
            edge_index=edge_index,
            time_emb=time_emb,
        )
        
        # Check that we have features at each level
        assert len(skip_features) == 3
        assert len(active_masks) == 3
        
        # Check shapes at each level
        expected_channels = [32, 64, 128]
        for i, (feats, mask) in enumerate(zip(skip_features, active_masks)):
            assert feats.shape[0] == B
            assert feats.shape[1] == T
            assert feats.shape[2] == N
            assert feats.shape[3] == expected_channels[i]
            assert mask.shape == (B, N)
            
            # Check progressive downsampling
            num_active = mask.sum(dim=1).float().mean()
            print(f"  Level {i}: ~{num_active:.1f} active nodes (out of {N})")
        
        print("✓ UGNNEncoder basic: PASSED")
    
    def test_encoder_with_conditioning(self):
        """Test encoder with conditional signals."""
        config = UGNNConfig(
            in_channels=8,
            out_channels=8,
            base_channels=16,
            channel_multipliers=[1, 2],
            embedding_config=EmbeddingConfig(
                time_embed_dim=32,
                cond_channels=4,
                cond_embed_dim=16,
            ),
            pooling_config=PoolingConfig(gamma=2),
        )
        encoder = UGNNEncoder(
            in_channels=8,
            config=config,
        )
        
        B, T, N = 2, 3, 32
        x = torch.randn(B, T, N, 8)
        timesteps = torch.randint(0, 1000, (B,))
        time_emb = torch.randn(B, 32)
        cond = torch.randn(B, T, N, 4)
        edge_index = torch.randint(0, B * N, (2, 100))
        
        _, skip_features, active_masks, _ = encoder(
            x=x,
            timesteps=timesteps,
            edge_index=edge_index,
            time_emb=time_emb,
            cond=cond,
        )
        
        assert len(skip_features) == 2
        assert len(active_masks) == 2
        
        print("✓ UGNNEncoder with conditioning: PASSED")
    
    def test_encoder_gradient_flow(self):
        """Test gradient flow through encoder."""
        config = UGNNConfig(
            in_channels=8,
            out_channels=8,
            base_channels=16,
            channel_multipliers=[1, 2],
            embedding_config=EmbeddingConfig(time_embed_dim=32),
            pooling_config=PoolingConfig(gamma=2),
        )
        encoder = UGNNEncoder(
            in_channels=8,
            config=config,
        )
        
        B, T, N = 2, 3, 32
        x = torch.randn(B, T, N, 8, requires_grad=True)
        timesteps = torch.randint(0, 1000, (B,))
        time_emb = torch.randn(B, 32, requires_grad=True)
        edge_index = torch.randint(0, B * N, (2, 100))
        
        _, skip_features, active_masks, _ = encoder(
            x=x,
            timesteps=timesteps,
            edge_index=edge_index,
            time_emb=time_emb,
        )
        
        # Compute loss from final features
        loss = skip_features[-1].sum()
        loss.backward()
        
        assert x.grad is not None
        assert time_emb.grad is not None
        assert x.grad.abs().sum() > 0
        assert time_emb.grad.abs().sum() > 0
        
        # Check encoder parameters have gradients
        total_grad = 0
        for param in encoder.parameters():
            if param.requires_grad and param.grad is not None:
                total_grad += param.grad.abs().sum().item()
        
        assert total_grad > 0, "Encoder parameters should have gradients"
        
        print("✓ UGNNEncoder gradient flow: PASSED")
    
    def test_variable_gamma(self):
        """Test encoder with variable gamma per level."""
        config = UGNNConfig(
            in_channels=8,
            out_channels=8,
            base_channels=16,
            channel_multipliers=[1, 2, 4],
            embedding_config=EmbeddingConfig(time_embed_dim=32),
            pooling_config=PoolingConfig(gamma=[2, 2, 3]),  # Variable gamma
        )
        encoder = UGNNEncoder(
            in_channels=8,
            config=config,
        )
        
        # Check stride tracking
        assert encoder.gammas == [2, 2, 3]
        assert encoder.stride_pre == [1, 2, 4]  # Accumulated: 1, 2, 2*2=4
        assert encoder.stride_post == [2, 4, 12]  # Accumulated: 2, 4, 4*3=12
        
        B, T, N = 2, 5, 64
        x = torch.randn(B, T, N, 8)
        timesteps = torch.randint(0, 1000, (B,))
        time_emb = torch.randn(B, 32)
        edge_index = torch.randint(0, B * N, (2, 300))
        
        _, skip_features, active_masks, _ = encoder(
            x=x,
            timesteps=timesteps,
            edge_index=edge_index,
            time_emb=time_emb,
        )
        
        assert len(skip_features) == 3
        assert len(active_masks) == 3
        
        # Check that active nodes decrease according to gamma
        for i, mask in enumerate(active_masks):
            num_active = mask.sum(dim=1).float().mean()
            print(f"  Level {i}: gamma={encoder.gammas[i]}, stride={encoder.stride_post[i]}, ~{num_active:.1f} active")
        
        print("✓ UGNNEncoder variable gamma: PASSED")
    
    def test_stride_tracking(self):
        """Test that stride_pre and stride_post are correctly computed."""
        # Test with uniform gamma
        config1 = UGNNConfig(
            in_channels=8,
            out_channels=8,
            base_channels=16,
            channel_multipliers=[1, 2, 4],
            pooling_config=PoolingConfig(gamma=2),
        )
        encoder1 = UGNNEncoder(in_channels=8, config=config1)
        assert encoder1.stride_pre == [1, 2, 4]
        assert encoder1.stride_post == [2, 4, 8]
        
        # Test with variable gamma
        config2 = UGNNConfig(
            in_channels=8,
            out_channels=8,
            base_channels=16,
            channel_multipliers=[1, 2, 4],
            pooling_config=PoolingConfig(gamma=[2, 3, 2]),
        )
        encoder2 = UGNNEncoder(in_channels=8, config=config2)
        assert encoder2.stride_pre == [1, 2, 6]
        assert encoder2.stride_post == [2, 6, 12]
        
        # Test with single level
        config3 = UGNNConfig(
            in_channels=8,
            out_channels=8,
            base_channels=16,
            channel_multipliers=[1],
            pooling_config=PoolingConfig(gamma=3),
        )
        encoder3 = UGNNEncoder(in_channels=8, config=config3)
        assert encoder3.stride_pre == [1]
        assert encoder3.stride_post == [3]
        
        print("✓ UGNNEncoder stride tracking: PASSED")


class TestDecoderBlock:
    """Test suite for DecoderBlock."""
    
    def test_basic_forward_concat(self):
        """Test basic forward pass with concatenation skip fusion."""
        gnn_config = GNNConfig()
        upsampling_config = UpsamplingConfig(method='zero')
        embedding_config = EmbeddingConfig(time_embed_dim=128)
        
        block = DecoderBlock(
            in_channels=64,
            skip_channels=32,
            out_channels=32,
            stride_pre=2,  # After unpooling
            gnn_config=gnn_config,
            upsampling_config=upsampling_config,
            embedding_config=embedding_config,
            skip_connection_mode='concat',
        )
        
        B, T, N = 4, 10, 100
        x = torch.randn(B, T, N, 64)  # From previous decoder level
        skip_features = torch.randn(B, T, N, 32)  # From encoder
        timesteps = torch.randint(0, 1000, (B,))
        time_emb = torch.randn(B, 128)
        edge_index = torch.randint(0, B * N, (2, 500))
        
        # Create active mask (simulate unpooling target)
        active_mask_target = torch.ones(B, N, dtype=torch.bool)
        active_mask_target[:, ::2] = True  # Half nodes active
        # Simulate currently active nodes (previous decoder level) as empty set
        active_mask_input = torch.zeros(B, N, dtype=torch.bool)

        x_out, _ = block(
            x=x,
            skip_features=skip_features,
            timesteps=timesteps,
            edge_index=edge_index,
            active_mask_target=active_mask_target,
            active_mask_input=active_mask_input,
            time_emb=time_emb,
        )
        
        # Check shape
        assert x_out.shape == (B, T, N, 32)
        
        print("✓ DecoderBlock basic forward (concat): PASSED")
    
    def test_forward_add_mode(self):
        """Test forward pass with addition skip fusion."""
        gnn_config = GNNConfig()
        upsampling_config = UpsamplingConfig(method='zero')
        embedding_config = EmbeddingConfig(time_embed_dim=128)
        
        # For 'add' mode, in_channels and skip_channels should match
        block = DecoderBlock(
            in_channels=64,
            skip_channels=64,
            out_channels=32,
            stride_pre=2,
            gnn_config=gnn_config,
            upsampling_config=upsampling_config,
            embedding_config=embedding_config,
            skip_connection_mode='add',
        )
        
        B, T, N = 2, 5, 50
        x = torch.randn(B, T, N, 64)
        skip_features = torch.randn(B, T, N, 64)
        timesteps = torch.randint(0, 1000, (B,))
        time_emb = torch.randn(B, 128)
        edge_index = torch.randint(0, B * N, (2, 200))
        active_mask_target = torch.ones(B, N, dtype=torch.bool)
        active_mask_input = torch.zeros(B, N, dtype=torch.bool)
        
        x_out, _ = block(
            x=x,
            skip_features=skip_features,
            timesteps=timesteps,
            edge_index=edge_index,
            active_mask_target=active_mask_target,
            active_mask_input=active_mask_input,
            time_emb=time_emb,
        )
        
        assert x_out.shape == (B, T, N, 32)
        
        print("✓ DecoderBlock forward (add mode): PASSED")
    
    def test_with_conditioning(self):
        """Test forward pass with conditional embeddings."""
        gnn_config = GNNConfig()
        upsampling_config = UpsamplingConfig(method='zero')
        embedding_config = EmbeddingConfig(
            time_embed_dim=128,
            cond_channels=8,
            cond_embed_dim=32,
        )
        
        block = DecoderBlock(
            in_channels=64,
            skip_channels=32,
            out_channels=32,
            stride_pre=1,
            gnn_config=gnn_config,
            upsampling_config=upsampling_config,
            embedding_config=embedding_config,
            skip_connection_mode='concat',
        )
        
        B, T, N = 2, 5, 50
        x = torch.randn(B, T, N, 64)
        skip_features = torch.randn(B, T, N, 32)
        timesteps = torch.randint(0, 1000, (B,))
        time_emb = torch.randn(B, 128)
        cond_emb = torch.randn(B, N, 32)
        edge_index = torch.randint(0, B * N, (2, 200))
        active_mask_target = torch.ones(B, N, dtype=torch.bool)
        active_mask_input = torch.zeros(B, N, dtype=torch.bool)

        x_out, _ = block(
            x=x,
            skip_features=skip_features,
            timesteps=timesteps,
            edge_index=edge_index,
            active_mask_target=active_mask_target,
            active_mask_input=active_mask_input,
            cond_emb=cond_emb,
            time_emb=time_emb,
        )
        
        assert x_out.shape == (B, T, N, 32)
        
        print("✓ DecoderBlock with conditioning: PASSED")
    
    def test_gradient_flow(self):
        """Test that gradients flow properly through decoder block."""
        gnn_config = GNNConfig()
        upsampling_config = UpsamplingConfig(method='zero')
        embedding_config = EmbeddingConfig(time_embed_dim=128)
        
        block = DecoderBlock(
            in_channels=64,
            skip_channels=32,
            out_channels=32,
            stride_pre=1,
            gnn_config=gnn_config,
            upsampling_config=upsampling_config,
            embedding_config=embedding_config,
        )
        
        B, T, N = 2, 3, 20
        x = torch.randn(B, T, N, 64, requires_grad=True)
        skip_features = torch.randn(B, T, N, 32, requires_grad=True)
        timesteps = torch.randint(0, 1000, (B,))
        time_emb = torch.randn(B, 128)
        edge_index = torch.randint(0, B * N, (2, 80))
        active_mask_target = torch.ones(B, N, dtype=torch.bool)
        active_mask_input = torch.zeros(B, N, dtype=torch.bool)
        
        x_out, _ = block(
            x=x,
            skip_features=skip_features,
            timesteps=timesteps,
            edge_index=edge_index,
            active_mask_target=active_mask_target,
            active_mask_input=active_mask_input,
            time_emb=time_emb,
        )
        
        loss = x_out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert skip_features.grad is not None
        assert not torch.isnan(x.grad).any()
        assert not torch.isnan(skip_features.grad).any()
        
        print("✓ DecoderBlock gradient flow: PASSED")


class TestUGNNDecoder:
    """Test suite for UGNNDecoder."""
    
    def test_basic_decoder(self):
        """Test basic decoder forward pass."""
        config = UGNNConfig(
            in_channels=8,
            out_channels=8,
            base_channels=16,
            channel_multipliers=[1, 2, 4],
            embedding_config=EmbeddingConfig(time_embed_dim=32),
            skip_connection_mode='concat',
        )
        
        decoder = UGNNDecoder(out_channels=8, config=config)
        
        B, T, N = 2, 5, 64
        # Simulate bottleneck input (coarsest resolution)
        x_bottleneck = torch.randn(B, T, N, 64)  # base_channels * channel_multipliers[-1]
        
        # Simulate encoder skip features (finest to coarsest)
        skip_features = [
            torch.randn(B, T, N, 16),  # Level 0: base_channels * 1
            torch.randn(B, T, N, 32),  # Level 1: base_channels * 2
            torch.randn(B, T, N, 64),  # Level 2: base_channels * 4
        ]
        
        # Simulate encoder active masks
        active_masks = [
            torch.ones(B, N, dtype=torch.bool),  # Level 0: all active
            torch.ones(B, N, dtype=torch.bool),  # Level 1: half active
            torch.ones(B, N, dtype=torch.bool),  # Level 2: quarter active
        ]
        active_masks[1][:, ::2] = False
        active_masks[2][:, ::4] = False
        
        timesteps = torch.randint(0, 1000, (B,))
        time_emb = torch.randn(B, 32)
        edge_index = torch.randint(0, B * N, (2, 300))
        
        out, _ = decoder(
            x=x_bottleneck,
            skip_features=skip_features,
            active_masks=active_masks,
            timesteps=timesteps,
            edge_index=edge_index,
            time_emb=time_emb,
        )
        
        assert out.shape == (B, T, N, 8)
        
        print("✓ UGNNDecoder basic forward: PASSED")
    
    def test_decoder_with_conditioning(self):
        """Test decoder with conditional inputs."""
        config = UGNNConfig(
            in_channels=8,
            out_channels=8,
            base_channels=16,
            channel_multipliers=[1, 2],
            embedding_config=EmbeddingConfig(
                time_embed_dim=32,
                cond_channels=4,
                cond_embed_dim=16,
            ),
        )
        
        decoder = UGNNDecoder(out_channels=8, config=config)
        
        B, T, N = 2, 5, 32
        x_bottleneck = torch.randn(B, T, N, 32)
        skip_features = [
            torch.randn(B, T, N, 16),
            torch.randn(B, T, N, 32),
        ]
        active_masks = [
            torch.ones(B, N, dtype=torch.bool),
            torch.ones(B, N, dtype=torch.bool),
        ]
        
        timesteps = torch.randint(0, 1000, (B,))
        time_emb = torch.randn(B, 32)
        cond = torch.randn(B, T, N, 4)
        cond_emb_precomputed = decoder.cond_encoder(cond, target_timesteps=T)
        edge_index = torch.randint(0, B * N, (2, 150))
        
        out, _ = decoder(
            x=x_bottleneck,
            skip_features=skip_features,
            active_masks=active_masks,
            timesteps=timesteps,
            edge_index=edge_index,
            cond_emb_precomputed=cond_emb_precomputed,
            time_emb=time_emb,
        )
        
        assert out.shape == (B, T, N, 8)
        
        print("✓ UGNNDecoder with conditioning: PASSED")
    
    def test_decoder_gradient_flow(self):
        """Test gradient flow through decoder."""
        config = UGNNConfig(
            in_channels=8,
            out_channels=8,
            base_channels=16,
            channel_multipliers=[1, 2],
            embedding_config=EmbeddingConfig(time_embed_dim=32),
        )
        
        decoder = UGNNDecoder(out_channels=8, config=config)
        
        B, T, N = 2, 3, 32
        x_bottleneck = torch.randn(B, T, N, 32, requires_grad=True)
        skip_features = [
            torch.randn(B, T, N, 16, requires_grad=True),
            torch.randn(B, T, N, 32, requires_grad=True),
        ]
        active_masks = [
            torch.ones(B, N, dtype=torch.bool),
            torch.ones(B, N, dtype=torch.bool),
        ]
        
        timesteps = torch.randint(0, 1000, (B,))
        time_emb = torch.randn(B, 32)
        edge_index = torch.randint(0, B * N, (2, 150))
        
        out, _ = decoder(
            x=x_bottleneck,
            skip_features=skip_features,
            active_masks=active_masks,
            timesteps=timesteps,
            edge_index=edge_index,
            time_emb=time_emb,
        )
        
        loss = out.sum()
        loss.backward()
        
        assert x_bottleneck.grad is not None
        assert all(sf.grad is not None for sf in skip_features)
        assert not torch.isnan(x_bottleneck.grad).any()
        
        print("✓ UGNNDecoder gradient flow: PASSED")


class TestUGNN:
    """Test suite for full UGNN model."""
    
    def test_basic_forward(self):
        """Test basic UGNN forward pass (encoder only for now)."""
        config = UGNNConfig(
            in_channels=1,
            out_channels=1,
            base_channels=32,
            channel_multipliers=[1, 2, 4],
            embedding_config=EmbeddingConfig(time_embed_dim=64),
            pooling_config=PoolingConfig(gamma=2),
        )
        ugnn = UGNN(config=config)
        
        B, T, N = 2, 5, 64
        x = torch.randn(B, T, N, 1)
        timesteps = torch.randint(0, 1000, (B,))
        edge_index = torch.randint(0, B * N, (2, 300))
        
        out, _ = ugnn(x, timesteps, edge_index)
        
        # Output should have same shape as input
        assert out.shape == (B, T, N, 1)
        
        print("✓ UGNN basic forward: PASSED")
    
    def test_with_conditioning(self):
        """Test UGNN with conditional signals."""
        config = UGNNConfig(
            in_channels=1,
            out_channels=1,
            base_channels=16,
            channel_multipliers=[1, 2],
            embedding_config=EmbeddingConfig(
                time_embed_dim=32,
                cond_channels=2,
                cond_embed_dim=16,
            ),
            pooling_config=PoolingConfig(gamma=2),
        )
        ugnn = UGNN(config=config)
        
        B, T, N = 2, 3, 32
        x = torch.randn(B, T, N, 1)
        timesteps = torch.randint(0, 1000, (B,))
        cond = torch.randn(B, T, N, 2)
        edge_index = torch.randint(0, B * N, (2, 100))
        
        out, _ = ugnn(x, timesteps, edge_index, cond=cond)
        
        assert out.shape == (B, T, N, 1)
        
        print("✓ UGNN with conditioning: PASSED")
    
    def test_gradient_flow(self):
        """Test gradient flow through UGNN."""
        config = UGNNConfig(
            in_channels=1,
            out_channels=1,
            base_channels=16,
            channel_multipliers=[1, 2],
            embedding_config=EmbeddingConfig(time_embed_dim=32),
            pooling_config=PoolingConfig(gamma=2),
        )
        ugnn = UGNN(config=config)
        
        B, T, N = 2, 3, 32
        x = torch.randn(B, T, N, 1, requires_grad=True)
        timesteps = torch.randint(0, 1000, (B,))
        edge_index = torch.randint(0, B * N, (2, 100))
        
        out, _ = ugnn(x, timesteps, edge_index)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.abs().sum() > 0
        
        # Check model parameters have gradients
        total_grad = 0
        for param in ugnn.parameters():
            if param.requires_grad and param.grad is not None:
                total_grad += param.grad.abs().sum().item()
        
        assert total_grad > 0, "UGNN parameters should have gradients"
        
        print("✓ UGNN gradient flow: PASSED")
    
    def test_learned_pooling(self):
        """Test UGNN with learned pooling selection."""
        config = UGNNConfig(
            in_channels=1,
            out_channels=1,
            base_channels=16,
            channel_multipliers=[1, 2],
            embedding_config=EmbeddingConfig(time_embed_dim=32),
            pooling_config=PoolingConfig(gamma=2, selection_method='learned'),
        )
        ugnn = UGNN(config=config)
        
        B, T, N = 2, 3, 32
        x = torch.randn(B, T, N, 1)
        timesteps = torch.randint(0, 1000, (B,))
        edge_index = torch.randint(0, B * N, (2, 100))
        
        out, _ = ugnn(x, timesteps, edge_index)
        
        assert out.shape == (B, T, N, 1)
        
        # Check that encoder blocks have pooling with learned selection
        for i, block in enumerate(ugnn.encoder.encoder_blocks):
            assert block.pool.selection_method == 'learned'
            # Verify stride_input was passed correctly
            expected_stride = ugnn.encoder.stride_pre[i]
            assert block.pool.stride_input == expected_stride, \
                f"Level {i}: stride_input={block.pool.stride_input}, expected {expected_stride}"
        
        print("✓ UGNN with learned pooling: PASSED")
    
    def test_variable_gamma_full(self):
        """Test full UGNN with variable gamma."""
        config = UGNNConfig(
            in_channels=1,
            out_channels=1,
            base_channels=16,
            channel_multipliers=[1, 2, 4],
            embedding_config=EmbeddingConfig(time_embed_dim=32),
            pooling_config=PoolingConfig(gamma=[2, 3, 2]),  # Variable downsampling
        )
        ugnn = UGNN(config=config)
        
        # Verify encoder has correct stride tracking
        assert ugnn.encoder.gammas == [2, 3, 2]
        assert ugnn.encoder.stride_pre == [1, 2, 6]
        assert ugnn.encoder.stride_post == [2, 6, 12]
        
        B, T, N = 2, 3, 48
        x = torch.randn(B, T, N, 1)
        timesteps = torch.randint(0, 1000, (B,))
        edge_index = torch.randint(0, B * N, (2, 150))
        
        out, _ = ugnn(x, timesteps, edge_index)
        
        assert out.shape == (B, T, N, 1)
        
        print("✓ UGNN with variable gamma: PASSED")
    
    def test_no_bottleneck(self):
        """Test UGNN with bottleneck disabled."""
        config = UGNNConfig(
            in_channels=1,
            out_channels=1,
            base_channels=16,
            channel_multipliers=[1, 2, 4],
            num_bottleneck_layers=0,  # Disable bottleneck
        )
        ugnn = UGNN(config=config)
        
        # Verify bottleneck is Identity
        assert isinstance(ugnn.bottleneck, nn.Identity)
        
        B, T, N = 2, 3, 48
        x = torch.randn(B, T, N, 1)
        timesteps = torch.randint(0, 1000, (B,))
        edge_index = torch.randint(0, B * N, (2, 150))
        
        out, _ = ugnn(x, timesteps, edge_index)
        
        assert out.shape == (B, T, N, 1)
        
        print("✓ UGNN with no bottleneck: PASSED")
    
    def test_skip_connection_add_mode(self):
        """Test UGNN with addition skip connections."""
        config = UGNNConfig(
            in_channels=1,
            out_channels=1,
            base_channels=16,
            channel_multipliers=[1, 1, 1],  # Same channels for 'add' mode
            skip_connection_mode='add',
            num_bottleneck_layers=1,
        )
        ugnn = UGNN(config=config)
        
        B, T, N = 2, 3, 32
        x = torch.randn(B, T, N, 1)
        timesteps = torch.randint(0, 1000, (B,))
        edge_index = torch.randint(0, B * N, (2, 120))
        
        out, _ = ugnn(x, timesteps, edge_index)
        
        assert out.shape == (B, T, N, 1)
        
        # Verify decoder uses 'add' mode
        for block in ugnn.decoder.decoder_blocks:
            assert block.skip_connection_mode == 'add'
        
        print("✓ UGNN with skip connection add mode: PASSED")

    def test_return_intermediates_disabled(self):
        """Verify UGNN behavior when return_intermediates=False.

        The model should return a tuple `(output, None)` (intermediates are None)
        and gradients should flow through the network when computing a loss.
        """
        config = UGNNConfig(
            in_channels=1,
            out_channels=1,
            base_channels=16,
            channel_multipliers=[1, 2],
            embedding_config=EmbeddingConfig(time_embed_dim=32),
            pooling_config=PoolingConfig(gamma=2),
        )
        ugnn = UGNN(config=config)

        B, T, N = 2, 3, 32
        x = torch.randn(B, T, N, 1, requires_grad=True)
        timesteps = torch.randint(0, 1000, (B,))
        edge_index = torch.randint(0, B * N, (2, 100))

        out, intermediates = ugnn(x, timesteps, edge_index, return_intermediates=False)

        # Check outputs
        assert isinstance(out, torch.Tensor)
        assert intermediates is None, "Intermediates should be None when return_intermediates=False"
        assert out.shape == (B, T, N, 1)

        # Check gradient flow
        loss = out.sum()
        loss.backward()
        assert x.grad is not None and x.grad.abs().sum() > 0

        print("✓ UGNN return_intermediates=False: PASSED")
    
    def test_encoder_decoder_symmetry(self):
        """Test that encoder and decoder have symmetric structure."""
        config = UGNNConfig(
            in_channels=1,
            out_channels=1,
            base_channels=16,
            channel_multipliers=[1, 2, 4, 8],
            num_bottleneck_layers=1,
        )
        ugnn = UGNN(config=config)
        
        # Check that encoder and decoder have same number of levels
        assert len(ugnn.encoder.encoder_blocks) == len(ugnn.decoder.decoder_blocks)
        
        # Check stride symmetry
        assert ugnn.encoder.stride_pre == ugnn.decoder.stride_pre
        assert ugnn.encoder.stride_post == ugnn.decoder.stride_post
        
        # Check channel symmetry (decoder reverses encoder)
        encoder_channels = [block.out_channels for block in ugnn.encoder.encoder_blocks]
        decoder_channels = [block.out_channels for block in ugnn.decoder.decoder_blocks]
        assert decoder_channels == list(reversed(encoder_channels))
        
        print("✓ UGNN encoder-decoder symmetry: PASSED")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing EncoderBlock")
    print("="*60)
    test_enc = TestEncoderBlock()
    test_enc.test_basic_forward()
    test_enc.test_with_conditioning()
    test_enc.test_with_active_mask()
    test_enc.test_gradient_flow()
    
    print("\n" + "="*60)
    print("Testing UGNNEncoder")
    print("="*60)
    test_encoder = TestUGNNEncoder()
    test_encoder.test_basic_encoder()
    test_encoder.test_encoder_with_conditioning()
    test_encoder.test_encoder_gradient_flow()
    test_encoder.test_variable_gamma()
    test_encoder.test_stride_tracking()
    
    print("\n" + "="*60)
    print("Testing DecoderBlock")
    print("="*60)
    test_dec = TestDecoderBlock()
    test_dec.test_basic_forward_concat()
    test_dec.test_forward_add_mode()
    test_dec.test_with_conditioning()
    test_dec.test_gradient_flow()
    
    print("\n" + "="*60)
    print("Testing UGNNDecoder")
    print("="*60)
    test_decoder = TestUGNNDecoder()
    test_decoder.test_basic_decoder()
    test_decoder.test_decoder_with_conditioning()
    test_decoder.test_decoder_gradient_flow()
    
    print("\n" + "="*60)
    print("Testing UGNN (Full U-Net)")
    print("="*60)
    test_ugnn = TestUGNN()
    test_ugnn.test_basic_forward()
    test_ugnn.test_with_conditioning()
    test_ugnn.test_gradient_flow()
    test_ugnn.test_learned_pooling()
    test_ugnn.test_variable_gamma_full()
    test_ugnn.test_no_bottleneck()
    test_ugnn.test_skip_connection_add_mode()
    test_ugnn.test_encoder_decoder_symmetry()
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)
