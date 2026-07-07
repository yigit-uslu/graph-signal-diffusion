from abc import ABC, abstractmethod
from typing import Optional
from torch_geometric.loader import DataLoader
import torch

class BaseTrainer(ABC):
    """Base class for all trainers (diffusion, GNN, etc.)."""
    
    def __init__(self, device: torch.device, save_dir: Optional[str] = None):
        self.device = device
        self.save_dir = save_dir
    
    @abstractmethod
    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, **kwargs):
        """Train the model."""
        pass
    
    @abstractmethod
    def evaluate(self, loader: DataLoader) -> dict:
        """Evaluate the model."""
        pass
    
    @abstractmethod
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        pass