"""
Unit test for verifying sample collection reproducibility.

This test ensures that:
1. Network seeds are properly stored during training
2. Samples can be loaded from saved files
3. Networks can be regenerated from seeds with exact matching
"""

import pytest
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path
from torch_geometric.loader import DataLoader

from graph_signal_diffusion.datasets.wra.primal_dual_dataset import WRAPrimalDualDataset
from graph_signal_diffusion.datasets.wra.channel import WirelessChannel
from graph_signal_diffusion.models.power_allocation_gnn import PowerAllocationGNN
from graph_signal_diffusion.trainers.dual_optimizer import DualOptimizer
from graph_signal_diffusion.trainers.primal_dual_trainer import WRAPrimalDualTrainer
from graph_signal_diffusion.utils.rate_calculator import compute_system_parameters


class TestSampleReproducibility:
    """Test suite for sample collection reproducibility."""
    
    @pytest.fixture
    def setup_training(self):
        """Setup a minimal training configuration."""
        # Configuration
        config = {
            'num_networks': 2,
            'n_links': 10,
            'seed_start': 42,
            'deployment_range': 500.0,
            'num_timesteps': 50,
            'max_epochs': 50,
            'num_samples_per_network': 4,
        }
        
        # Create temporary directory for outputs
        temp_dir = tempfile.mkdtemp()
        
        # System parameters
        system_params = compute_system_parameters(
            P_max_dBm=10.0,
            bandwidth_Hz=1e6,
            noise_psd_dBm_Hz=-174.0,
        )
        
        yield config, temp_dir, system_params
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_network_seed_storage(self, setup_training):
        """Test that network seeds are stored in the dataset."""
        config, temp_dir, system_params = setup_training
        
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
            assert sample.network_seed.item() == expected_seed, \
                f"Sample {i}: Expected seed {expected_seed}, got {sample.network_seed.item()}"
        
        print(f"✓ Network seeds stored correctly in dataset")
    
    def test_sample_collection_and_loading(self, setup_training):
        """Test full training, sample collection, and reproducible loading."""
        config, temp_dir, system_params = setup_training
        
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
        
        # 2. Create model
        print("2. Creating model...")
        sample = dataset[0]
        input_dim = sample.x.shape[1]
        
        model = PowerAllocationGNN(
            input_dim=input_dim,
            hidden_dim=32,
            num_layers=2,
        )

        # 3. Create dual optimizer
        print("3. Creating dual optimizer...")
        dual_optimizer = DualOptimizer(
            num_networks=config['num_networks'],
            num_receivers=config['n_links'],
            r_min=0.5,
            alpha_dual=0.01,
            momentum=0.0,
            device='cpu',
        )

        # 4. Create trainer
        print("4. Creating trainer...")
        trainer = WRAPrimalDualTrainer(
            model=model,
            dual_optimizer=dual_optimizer,
            system_params=system_params,
            learning_rate=1e-3,
            max_epochs=config['max_epochs'],
            checkpoint_dir=temp_dir,
            convergence_window=10,
            num_samples_per_network=config['num_samples_per_network'],
            moving_avg_window=5,
            gradient_norm_threshold=float('inf'),
            dual_variance_threshold=float('inf'),
            violation_fraction_threshold=1.0,
            violation_fraction_on_model_avg_rates_threshold=1.0,
            device='cpu',
        )

        # 5. Train
        print("5. Training...")
        results = trainer.train(dataloader)
        
        # 6. Load saved samples
        print("6. Loading saved samples...")
        samples_path = Path(temp_dir) / "collected_samples.npz"
        assert samples_path.exists(), "Samples file not created"
        
        loaded_samples = np.load(samples_path, allow_pickle=True)

        # 7. Check schema (v1 format)
        print("7. Checking schema...")
        assert 'schema_version' in loaded_samples, "Missing schema_version"
        assert int(loaded_samples['schema_version']) == 1
        network_ids_arr = loaded_samples['network_ids']        # (N,) int64
        network_seeds_arr = loaded_samples['network_seeds']    # (N,) int64
        assoc_arr = loaded_samples['associations']             # object array
        power_samples_arr = loaded_samples['power_samples_per_network']  # object array
        rate_samples_arr = loaded_samples['rate_samples_per_network']    # object array

        # 8. Check network seeds
        print("8. Checking network seeds...")
        for pos, net_id in enumerate(network_ids_arr):
            loaded_seed = int(network_seeds_arr[pos])
            expected_seed = config['seed_start'] + int(net_id)
            assert loaded_seed == expected_seed, \
                f"Network {net_id}: Expected seed {expected_seed}, got {loaded_seed}"

        print("✓ Network seeds saved and loaded correctly")

        # 9. Verify associations match when network is regenerated from its seed
        print("9. Regenerating networks and verifying associations...")
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
            print(f"  ✓ Network {net_id}: Associations match (deterministic from seed)")

        # 10. Check power and rate sample shapes
        print("10. Checking power and rate samples...")
        for pos, net_id in enumerate(network_ids_arr):
            assoc_saved = assoc_arr[pos]
            m, n = assoc_saved.shape
            powers = power_samples_arr[pos]  # (num_samples, m)
            rates = rate_samples_arr[pos]    # (num_samples, n)
            assert powers.shape == (config['num_samples_per_network'], m), \
                f"Power shape mismatch: {powers.shape}"
            assert rates.shape == (config['num_samples_per_network'], n), \
                f"Rates shape mismatch: {rates.shape}"

        print(f"✓ All {config['num_samples_per_network']} samples per network saved correctly")
        
        print("\n" + "="*70)
        print("ALL REPRODUCIBILITY TESTS PASSED!")
        print("="*70)
    
    def test_load_samples_function(self, setup_training):
        """Test helper function for loading samples."""
        config, temp_dir, system_params = setup_training
        
        # Create and train (abbreviated version)
        dataset = WRAPrimalDualDataset.from_seed_range(
            num_networks=config['num_networks'],
            n_links=config['n_links'],
            seed_start=config['seed_start'],
            deployment_range=config['deployment_range'],
            num_timesteps=config['num_timesteps'],
        )
        
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        sample = dataset[0]
        
        model = PowerAllocationGNN(
            input_dim=sample.x.shape[1],
            hidden_dim=32,
            num_layers=2,
        )

        dual_optimizer = DualOptimizer(
            num_networks=config['num_networks'],
            num_receivers=config['n_links'],
            r_min=0.5,
            alpha_dual=0.01,
            device='cpu',
        )

        trainer = WRAPrimalDualTrainer(
            model=model,
            dual_optimizer=dual_optimizer,
            system_params=system_params,
            learning_rate=1e-3,
            max_epochs=10,
            checkpoint_dir=temp_dir,
            num_samples_per_network=config['num_samples_per_network'],
            device='cpu',
        )
        
        trainer.train(dataloader)
        
        # Test loading v1-schema NPZ
        def load_samples_v1(samples_path: Path):
            """Load v1-schema PD samples, returning a dict keyed by network_id."""
            npz = np.load(samples_path, allow_pickle=True)
            assert int(npz['schema_version']) == 1, "Unexpected schema version"
            network_ids = npz['network_ids']
            network_seeds = npz['network_seeds']
            associations = npz['associations']
            power_samples = npz['power_samples_per_network']
            rate_samples = npz['rate_samples_per_network']

            results = {}
            for pos, net_id in enumerate(network_ids):
                results[int(net_id)] = {
                    'seed': int(network_seeds[pos]),
                    'associations': associations[pos],
                    'power_samples': list(power_samples[pos]),   # (num_samples, m) → list of rows
                    'rate_samples': list(rate_samples[pos]),
                }
            return results

        samples_path = Path(temp_dir) / "collected_samples.npz"
        loaded_results = load_samples_v1(samples_path)

        assert len(loaded_results) == config['num_networks']

        for net_id, data in loaded_results.items():
            assert 'seed' in data
            assert len(data['power_samples']) == config['num_samples_per_network'], \
                f"Network {net_id}: expected {config['num_samples_per_network']} power samples"
            assert len(data['rate_samples']) == config['num_samples_per_network'], \
                f"Network {net_id}: expected {config['num_samples_per_network']} rate samples"

        print("✓ Sample loading helper function works correctly")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
