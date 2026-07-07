"""
Integration test for WRA task evaluator with real dataset.
"""
import torch
import pytest
from pathlib import Path

from graph_signal_diffusion.tasks.wireless_resource_allocation.evaluator import (
    WirelessResourceAllocationTask
)
from graph_signal_diffusion.datasets.wra.datamodule import WRABuilder
from graph_signal_diffusion.datasets.normalizer import Normalizer


@pytest.mark.slow
class TestWRAIntegration:
    """Integration tests for the complete WRA pipeline."""

    def test_end_to_end_evaluation(self):
        """Test complete evaluation pipeline with real dataset."""
        # Skip if data doesn't exist
        data_root = Path('data/wra')
        if not (data_root / 'raw' / 'small').exists():
            pytest.skip("WRA dataset not found - run conversion script first")

        # Build dataset
        builder = WRABuilder()
        cfg = type('Config', (), {
            'root': str(data_root),
            'datasets': ['small'],
            'batch_size': 4,
            'normalize': True,
            'train_fraction': 0.8,
            'val_fraction': 0.1,
            'test_fraction': 0.1,
        })()

        datasets = builder.build_datasets(cfg)
        dataset_info = builder.get_dataset_info()

        # Create task evaluator
        task = WirelessResourceAllocationTask(num_channel_realizations=50)
        task.set_dataset_info(dataset_info)

        # Create normalizer (mock fitted state)
        normalizer = Normalizer()
        normalizer.fitted = True
        normalizer.mean_ = torch.tensor([0.0])
        normalizer.std_ = torch.tensor([0.5])
        task.set_normalizer(normalizer)

        # Get a small batch
        test_loader = builder.build_loaders(cfg, datasets)['test']
        batch = next(iter(test_loader))

        # Prepare data
        data = task.prepare_data(batch)

        # Create dummy generated samples (slightly perturbed real samples)
        generated = data['samples'] + 0.1 * torch.randn_like(data['samples'])
        real = data['samples']

        # Evaluate
        metrics = task.evaluate_samples(generated, real, data['metadata'])

        # Check that we got reasonable metrics
        assert 'sum_rate_generated' in metrics
        assert 'sum_rate_real' in metrics
        assert 'fairness_generated' in metrics
        assert 'fairness_real' in metrics
        assert 'num_networks_evaluated' in metrics

        # Basic sanity checks
        assert metrics['num_networks_evaluated'] > 0
        assert metrics['sum_rate_generated'] > 0
        assert metrics['sum_rate_real'] > 0
        assert 0 <= metrics['fairness_generated'] <= 1
        assert 0 <= metrics['fairness_real'] <= 1

        print(f"Integration test passed: {metrics['num_networks_evaluated']} networks evaluated")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])