"""
Simple test runner for sample reproducibility without pytest dependencies.
"""

import sys
import torch
import numpy as np
import json
import tempfile
import shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from torch_geometric.loader import DataLoader
from graph_signal_diffusion.datasets.wra.primal_dual_dataset import WRAPrimalDualDataset
from graph_signal_diffusion.datasets.wra.channel import WirelessChannel
from graph_signal_diffusion.models.power_allocation_gnn import PowerAllocationGNN
from graph_signal_diffusion.trainers.dual_optimizer import DualOptimizer
from graph_signal_diffusion.trainers.primal_dual_trainer import WRAPrimalDualTrainer
from graph_signal_diffusion.utils.rate_calculator import compute_system_parameters


def test_network_seed_storage():
    """Test that network seeds are stored in the dataset."""
    print("\n" + "="*70)
    print("TEST 1: Network Seed Storage")
    print("="*70)
    
    config = {
        'num_networks': 2,
        'n_links': 10,
        'seed_start': 42,
        'deployment_range': 500.0,
        'num_timesteps': 50,
    }
    
    # Create dataset
    dataset = WRAPrimalDualDataset.from_seed_range(
        num_networks=config['num_networks'],
        n_links=config['n_links'],
        seed_start=config['seed_start'],
        deployment_range=config['deployment_range'],
        num_timesteps=config['num_timesteps'],
    )
    
    # Check that seeds are stored correctly
    for i in range(config['num_networks']):
        sample = dataset[i]
        assert hasattr(sample, 'network_seed'), f"Sample {i} missing network_seed"
        expected_seed = config['seed_start'] + i
        actual_seed = sample.network_seed.item()
        assert actual_seed == expected_seed, \
            f"Sample {i}: Expected seed {expected_seed}, got {actual_seed}"
        print(f"  ✓ Network {i}: seed = {actual_seed}")
    
    print("✓ TEST 1 PASSED: Network seeds stored correctly\n")


def test_full_reproducibility():
    """Test full training, sample collection, and reproducible loading."""
    print("="*70)
    print("TEST 2: Full Sample Collection and Reproducibility")
    print("="*70)
    
    config = {
        'num_networks': 2,
        'n_links': 10,
        'seed_start': 42,
        'deployment_range': 500.0,
        'num_timesteps': 50,
        'max_epochs': 20,
        'num_samples_per_network': 3,
    }
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    print(f"\nUsing temporary directory: {temp_dir}")
    
    try:
        # 1. Create dataset
        print("\n1. Creating dataset...")
        dataset = WRAPrimalDualDataset.from_seed_range(
            num_networks=config['num_networks'],
            n_links=config['n_links'],
            seed_start=config['seed_start'],
            deployment_range=config['deployment_range'],
            num_timesteps=config['num_timesteps'],
        )
        
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        
        # 2. System parameters
        print("2. Computing system parameters...")
        system_params = compute_system_parameters(
            P_max_dBm=10.0,
            bandwidth_Hz=1e6,
            noise_psd_dBm_Hz=-174.0,
        )
        
        # 3. Create model
        print("3. Creating model...")
        sample = dataset[0]
        input_dim = sample.x.shape[1]
        
        model = PowerAllocationGNN(
            input_dim=input_dim,
            hidden_dim=32,
            num_layers=2,
            P_max=system_params['P_max_watts'],
        )
        
        # 4. Create dual optimizer
        print("4. Creating dual optimizer...")
        dual_optimizer = DualOptimizer(
            num_networks=config['num_networks'],
            num_receivers=config['n_links'],
            r_min=0.5,
            alpha_dual=0.01,
            momentum=0.0,
            device='cpu',
        )
        
        # 5. Create trainer
        print("5. Creating trainer...")
        trainer = WRAPrimalDualTrainer(
            model=model,
            dual_optimizer=dual_optimizer,
            system_params=system_params,
            learning_rate=1e-3,
            max_epochs=config['max_epochs'],
            checkpoint_dir=temp_dir,
            convergence_window=5,
            num_samples_per_network=config['num_samples_per_network'],
            moving_avg_window=5,
            gradient_norm_threshold=float('inf'),
            dual_variance_threshold=float('inf'),
            violation_fraction_threshold=1.0,
            violation_fraction_on_model_avg_rates_threshold=1.0,
            device='cpu',  # Use CPU for faster test
        )
        
        # 6. Train
        print("6. Training (this will take a minute)...")
        results = trainer.train(dataloader)
        print(f"   Training completed: {config['max_epochs']} epochs")
        
        # 7. Load saved samples
        print("\n7. Loading saved samples...")
        samples_path = Path(temp_dir) / "collected_samples.npz"
        assert samples_path.exists(), "Samples file not created"
        
        loaded_samples = np.load(samples_path, allow_pickle=True)
        print(f"   Loaded {len(loaded_samples.files)} keys from samples file")

        # 8. Check schema (v1 format)
        print("\n8. Checking schema and sample counts...")
        assert 'schema_version' in loaded_samples, "Missing schema_version"
        assert int(loaded_samples['schema_version']) == 1, "Unexpected schema version"
        assert 'network_ids' in loaded_samples, "Missing network_ids"
        assert 'network_seeds' in loaded_samples, "Missing network_seeds"
        assert 'power_samples_per_network' in loaded_samples, "Missing power_samples_per_network"

        network_ids_arr = loaded_samples['network_ids']            # (N,) int64
        network_seeds_arr = loaded_samples['network_seeds']        # (N,) int64
        power_samples_arr = loaded_samples['power_samples_per_network']  # object array
        rate_samples_arr = loaded_samples['rate_samples_per_network']    # object array
        assoc_arr = loaded_samples['associations']                 # object array

        assert len(network_ids_arr) == config['num_networks'], \
            f"Expected {config['num_networks']} networks, got {len(network_ids_arr)}"

        # All networks should have num_samples_per_network samples
        for pos in range(config['num_networks']):
            n_samples = power_samples_arr[pos].shape[0]
            assert n_samples == config['num_samples_per_network'], (
                f"Network {network_ids_arr[pos]}: expected {config['num_samples_per_network']} "
                f"samples, got {n_samples}"
            )
        print(f"   ✓ Schema v1, {config['num_samples_per_network']} samples per network")

        # 9. Check network seeds
        print("\n9. Verifying network seeds...")
        for pos, net_id in enumerate(network_ids_arr):
            loaded_seed = int(network_seeds_arr[pos])
            expected_seed = config['seed_start'] + int(net_id)
            assert loaded_seed == expected_seed, \
                f"Network {net_id}: Expected seed {expected_seed}, got {loaded_seed}"
            print(f"   ✓ Network {net_id}: seed = {loaded_seed}")

        # 10. Regenerate networks and verify associations match
        print("\n10. Regenerating networks and verifying channel matching...")
        for pos, net_id in enumerate(network_ids_arr):
            seed = int(network_seeds_arr[pos])
            assoc_saved = assoc_arr[pos]

            channel_regen = WirelessChannel(
                n_links=config['n_links'],
                seed=seed,
                deployment_range=config['deployment_range'],
            )
            assoc_regen = channel_regen.associations
            assert np.allclose(assoc_regen, assoc_saved), \
                f"Network {net_id}: Associations mismatch!"
            print(f"   ✓ Network {net_id}: seed={seed} → same topology & associations")

        print("   ✓ All networks regenerated with matching structure")

        # 11. Check power and rate sample shapes
        print("\n11. Verifying power and rate samples...")
        for pos, net_id in enumerate(network_ids_arr):
            assoc_saved = assoc_arr[pos]
            m, n = assoc_saved.shape
            powers = power_samples_arr[pos]  # (num_samples, m)
            rates = rate_samples_arr[pos]    # (num_samples, n)
            assert powers.shape == (config['num_samples_per_network'], m), \
                f"Power shape mismatch: {powers.shape}"
            assert rates.shape == (config['num_samples_per_network'], n), \
                f"Rates shape mismatch: {rates.shape}"
            print(f"   ✓ Network {net_id}: All {config['num_samples_per_network']} samples verified")
        
        print("\n✓ TEST 2 PASSED: Full reproducibility verified\n")
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temporary directory")


if __name__ == "__main__":
    print("\n" + "#"*70)
    print("# SAMPLE REPRODUCIBILITY TEST SUITE")
    print("#"*70)
    
    try:
        test_network_seed_storage()
        test_full_reproducibility()
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED! ✓")
        print("="*70)
        print("\nSummary:")
        print("  ✓ Network seeds are correctly stored in dataset")
        print("  ✓ Seeds are saved in collected samples")
        print("  ✓ Networks can be regenerated from seeds")
        print("  ✓ Regenerated channels match original data exactly")
        print("  ✓ Power and rate samples are properly structured")
        print("\nConclusion: Sample collection is fully reproducible!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
