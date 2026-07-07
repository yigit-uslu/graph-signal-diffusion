"""
Tests for WirelessResourceAllocationTask evaluator.
"""
import torch
import numpy as np
import pytest
from unittest.mock import MagicMock

from graph_signal_diffusion.tasks.wireless_resource_allocation.evaluator import (
    WirelessResourceAllocationTask
)
from graph_signal_diffusion.datasets.normalizer import Normalizer


class TestWRATaskEvaluator:
    """Test cases for wireless resource allocation task evaluator."""
    
    def test_task_initialization(self):
        """Test that task can be initialized."""
        task = WirelessResourceAllocationTask(num_channel_realizations=100)
        assert task.num_channel_realizations == 100
        assert task.normalizer is None
        assert task.dataset_info is None
    
    def test_normalizer_injection(self):
        """Test normalizer injection."""
        task = WirelessResourceAllocationTask()
        normalizer = MagicMock(spec=Normalizer)
        
        task.set_normalizer(normalizer)
        assert task.normalizer is normalizer
    
    def test_dataset_info_injection(self):
        """Test dataset info injection."""
        task = WirelessResourceAllocationTask()
        dataset_info = {
            'system_params': {'P_max': 1.0, 'noise_var': 1e-10, 'r_min': 0.5},
            'channel_params': {'deployment_range': 1000.0},
            'network_seeds': {0: 42},
            'associations': {0: np.eye(10)},
        }
        
        task.set_dataset_info(dataset_info)
        assert task.dataset_info == dataset_info


class TestWRAUtilities:
    """Test utility functions used by the task."""
    
    def test_receiver_to_transmitter_power(self):
        """Test power conversion."""
        from graph_signal_diffusion.datasets.wra.utils import receiver_to_transmitter_power
        
        # Simple case: 2 transmitters, 2 receivers, identity association
        power_rx = torch.tensor([1.0, 2.0])
        associations = torch.eye(2)
        
        power_tx = receiver_to_transmitter_power(power_rx, associations)
        
        assert torch.allclose(power_tx, power_rx)
    
    def test_clamp_power(self):
        """Test power clamping."""
        from graph_signal_diffusion.datasets.wra.utils import clamp_power
        
        power = torch.tensor([0.5, 1.5, -0.5, 0.8])
        P_max = 1.0
        
        clamped = clamp_power(power, P_max)
        
        assert torch.all(clamped >= 0)
        assert torch.all(clamped <= P_max)
        assert clamped[0].item() == 0.5
        assert clamped[1].item() == 1.0
        assert clamped[2].item() == 0.0
    
    def test_jains_fairness_index(self):
        """Test Jain's fairness index computation."""
        from graph_signal_diffusion.datasets.wra.utils import jains_fairness_index
        
        # Perfect fairness: all equal
        rates_equal = torch.tensor([1.0, 1.0, 1.0, 1.0])
        fairness_equal = jains_fairness_index(rates_equal)
        assert abs(fairness_equal - 1.0) < 1e-6
        
        # Unfair: one gets everything
        rates_unfair = torch.tensor([4.0, 0.0, 0.0, 0.0])
        fairness_unfair = jains_fairness_index(rates_unfair)
        assert fairness_unfair < 1.0
        assert fairness_unfair == 0.25  # Should be 1/n for this case
    
    def test_compute_violation_rate(self):
        """Test constraint violation rate."""
        from graph_signal_diffusion.datasets.wra.utils import compute_violation_rate
        
        values = torch.tensor([0.5, 1.5, 0.8, 2.0, 0.3])
        threshold = 1.0
        
        # Upper bound violations (values > threshold)
        violation_rate = compute_violation_rate(values, threshold, lower_bound=False)
        assert abs(violation_rate - 0.4) < 1e-6  # 2 out of 5 violate
        
        # Lower bound violations (values < threshold)
        violation_rate_lower = compute_violation_rate(values, threshold, lower_bound=True)
        assert abs(violation_rate_lower - 0.6) < 1e-6  # 3 out of 5 violate


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
