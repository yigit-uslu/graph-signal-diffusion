from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any
import torch

from graph_signal_diffusion.core.registry import Registry

TASK_REGISTRY = Registry("tasks")

class BaseTask(ABC):
    """Base class for downstream evaluation tasks."""
    def __init__(self, **kwargs):
        super().__init__()
        self.debug_print = lambda msg: None  # Injected by trainer
    
    @abstractmethod
    def evaluate_samples(
        self, 
        generated_samples: torch.Tensor,
        real_samples: torch.Tensor,
        metadata: Dict[str, Any],
        **kwargs
    ) -> Dict[str, float]:
        """
        Evaluate generated samples on the downstream task.
        
        Args:
            generated_samples: Samples from the diffusion model
            real_samples: Ground truth samples from validation set
            metadata: Additional info (e.g., graph structure, timestamps)
            **kwargs: Additional arguments for specific tasks
        Returns:
            Dictionary of metrics (e.g., {'mse': 0.5, 'mae': 0.3})
        """
        pass
    
    @abstractmethod
    def prepare_data(self, data_batch) -> Dict[str, Any]:
        """
        Extract relevant data from batch for evaluation.
        
        Returns:
            Dictionary with 'samples', 'metadata', etc.
        """
        pass