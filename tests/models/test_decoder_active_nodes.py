"""Unit test for decoder active node propagation logic."""
import sys
import torch
from pathlib import Path

# Import UGNN modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from graph_signal_diffusion.models.ugnn import (
    UGNN,
    GNNConfig,
    PoolingConfig,
    EmbeddingConfig,
    UGNNConfig,
)


def test_decoder_active_node_propagation():
    """
    Test that decoder correctly propagates active nodes through upsampling.
    
    Expected behavior with gamma=2, 3 levels, starting with 512 nodes:
    - Encoder level 0: 512 active -> pool -> 256 active
    - Encoder level 1: 256 active -> pool -> 128 active
    - Encoder level 2: 128 active -> pool -> 64 active (bottleneck)
    
    Decoder should restore in reverse:
    - Decoder level 2 input: 64, output: 128
    - Decoder level 1 input: 128, output: 256
    - Decoder level 0 input: 256, output: 512
    """
    print("\n" + "="*60)
    print("Testing Decoder Active Node Propagation")
    print("="*60 + "\n")
    
    # Create UGNN with 3 levels
    n_nodes = 512
    config = UGNNConfig(
        in_channels=1,
        out_channels=1,
        base_channels=16,
        channel_multipliers=[1, 2, 4],
        gnn_config=GNNConfig(
            K=2,
            num_layers=1,
            norm_type='layer',
            dropout=0.0,
            activation='silu',
            use_strided_conv=True,
            use_pre_activation=False,
        ),
        pooling_config=PoolingConfig(
            gamma=2,
            pool_K=1,
            selection_method='stride',
        ),
        embedding_config=EmbeddingConfig(
            time_embed_dim=64,
            num_timesteps=1000,
        ),
    )
    
    ugnn = UGNN(config=config)
    ugnn.eval()
    
    # Create simple input
    B, T, F = 1, 1, 1
    x = torch.randn(B, T, n_nodes, F)
    timesteps = torch.zeros(B, dtype=torch.long)
    
    # Create simple graph (ring lattice)
    edge_list = []
    for i in range(n_nodes):
        edge_list.append([i, (i + 1) % n_nodes])
        edge_list.append([i, (i - 1) % n_nodes])
    edge_index = torch.tensor(edge_list).t().contiguous()
    
    print(f"Initial setup:")
    print(f"  Nodes: {n_nodes}")
    print(f"  Encoder levels: {config.num_levels}")
    print(f"  Gamma: 2")
    print(f"  Selection method: stride\n")
    
    # Run encoder
    with torch.no_grad():
        time_emb = ugnn.time_embed(timesteps)
        x_enc, skip_features, active_masks, encoder_intermediates = ugnn.encoder(
            x=x,
            timesteps=timesteps,
            edge_index=edge_index,
            time_emb=time_emb,
            return_intermediates=True,
        )
    
    print("Encoder output (skip features for decoder):")
    for i, mask in enumerate(active_masks):
        num_active = mask.sum().item()
        print(f"  Level {i}: {num_active}/{n_nodes} active nodes")
    
    # Get bottleneck features (output of last encoder level)
    x_bottleneck = skip_features[-1]  # Most coarse features
    bottleneck_mask = active_masks[-1]
    num_bottleneck_active = bottleneck_mask.sum().item()
    print(f"\nBottleneck: {num_bottleneck_active}/{n_nodes} active nodes")
    
    # Expected encoder active nodes (with gamma=2, stride selection)
    expected_encoder = [256, 128, 64]
    
    print(f"\nVerifying encoder active nodes:")
    for i, (actual, expected) in enumerate(zip([m.sum().item() for m in active_masks], expected_encoder)):
        status = "✓" if actual == expected else "✗"
        print(f"  {status} Level {i}: expected {expected}, got {actual}")
    
    # Run decoder with intermediates
    with torch.no_grad():
        output, all_intermediates = ugnn.decoder(
            x=x_bottleneck,
            skip_features=skip_features,
            active_masks=active_masks,
            timesteps=timesteps,
            edge_index=edge_index,
            time_emb=time_emb,
            return_intermediates=True,
        )
    
    print(f"\n" + "-"*60)
    print("Decoder intermediate active nodes:")
    print("-"*60)
    
    # Expected decoder progression
    # Level 2: input 64 -> output 128
    # Level 1: input 128 -> output 256
    # Level 0: input 256 -> output 512
    expected_decoder_input = [256, 128, 64]  # Listed by level (0, 1, 2)
    expected_decoder_output = [512, 256, 128]  # Listed by level (0, 1, 2)
    
    all_pass = True
    
    for level in range(config.num_levels):
        level_key = f'decoder_level_{level}'
        
        if level_key not in all_intermediates:
            print(f"\n✗ Level {level}: NOT FOUND in intermediates!")
            all_pass = False
            continue
        
        level_data = all_intermediates[level_key]
        
        # Check input mask
        if 'active_mask_in' in level_data:
            num_input_active = level_data['active_mask_in'].sum().item()
        else:
            num_input_active = "MISSING"
            all_pass = False
        
        # Check output mask
        if 'active_mask_out' in level_data:
            num_output_active = level_data['active_mask_out'].sum().item()
        else:
            num_output_active = "MISSING"
            all_pass = False
        
        # Check after_upsample features
        if 'after_upsample' in level_data:
            upsample_feats = level_data['after_upsample']
            # Count nodes matching output mask (these should be the "active" ones)
            if 'active_mask_out' in level_data:
                out_mask = level_data['active_mask_out']
                num_expected_active = out_mask.sum().item()
                # Note: all N nodes may have non-zero values due to skip connection fusion,
                # but only active_mask_out nodes are considered "active"
                num_upsample_info = f"{num_expected_active} (by mask, tensor shape: {upsample_feats.shape})"
            else:
                num_upsample_info = f"shape: {upsample_feats.shape}"
        else:
            num_upsample_info = "MISSING"
        
        print(f"\nDecoder Level {level}:")
        print(f"  Input mask:      {num_input_active}/{n_nodes} active")
        print(f"  After upsample:  {num_upsample_info}")
        print(f"  Output mask:     {num_output_active}/{n_nodes} active")
        
        # Verify expectations
        exp_in = expected_decoder_input[level]
        exp_out = expected_decoder_output[level]
        
        input_match = num_input_active == exp_in
        output_match = num_output_active == exp_out
        
        status_in = "✓" if input_match else "✗"
        status_out = "✓" if output_match else "✗"
        
        print(f"  {status_in} Expected input:  {exp_in}")
        print(f"  {status_out} Expected output: {exp_out}")
        
        if not input_match or not output_match:
            all_pass = False
    
    # Check final output
    print(f"\n" + "-"*60)
    print("Final decoder output:")
    final_nonzero = (output.abs().sum(dim=(1, -1)) > 1e-6).sum().item()
    print(f"  Non-zero nodes: {final_nonzero}/{n_nodes}")
    final_match = final_nonzero == n_nodes
    status_final = "✓" if final_match else "✗"
    print(f"  {status_final} Expected: {n_nodes}")
    
    if not final_match:
        all_pass = False
    
    print(f"\n" + "="*60)
    if all_pass:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print("="*60 + "\n")
    
    return all_pass


if __name__ == "__main__":
    success = test_decoder_active_node_propagation()
    exit(0 if success else 1)
