from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import torch
import torch.nn as nn
from torch_geometric.data import Data

class BaseDiffusion(ABC, nn.Module):
    """
    Base class for diffusion models.
    
    All diffusion models should implement:
    - add_noise: Forward diffusion process
    - training_loss: Loss computation for training
    - sample: Reverse diffusion (generation)
    """
    
    def __init__(self, model: nn.Module, **kwargs):
        super().__init__()
        self.model = model
        self.debug_print = lambda msg: None  # Injected by trainer
    
    @abstractmethod
    def add_noise(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Add noise to clean samples according to the forward diffusion process.
        
        Args:
            x0: Clean samples [batch, channels, nodes, features]
            t: Timesteps [batch]
            noise: Optional pre-sampled noise
            
        Returns:
            Noisy samples x_t
        """
        pass

    @abstractmethod
    def training_loss(self, data: Data) -> torch.Tensor:
        """
        Compute training loss for the diffusion model.
        
        Args:
            data: Batch of graph data
            
        Returns:
            Scalar loss tensor
        """
        pass
    
    @abstractmethod
    def sample(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
        data: Optional[Data] = None
    ) -> torch.Tensor:
        """
        Generate samples using reverse diffusion.
        
        Args:
            shape: Shape of samples to generate
            device: Device to generate on
            data: Optional conditioning data
            
        Returns:
            Generated samples
        """
        pass
    
    def forward(self, *args, **kwargs):
        """Forward pass - delegates to model."""
        return self.model(*args, **kwargs)
    

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model})"