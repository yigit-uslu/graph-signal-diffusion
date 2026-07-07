from abc import ABC, abstractmethod
from typing import Dict, Optional
import torch
from torch_geometric.loader import DataLoader

class BaseBaseline(ABC):
    """Base class for all baseline methods."""

    # Whether the dataloader must wrap eval splits with ReplicatedDataset
    # (providing n_samples copies of each input).  Default ``False`` — most
    # baselines (e.g. GRW) generate multiple samples internally.
    # Override to ``True`` in baselines that rely on the dataloader for
    # input replication (e.g. DiffusionBaseline).
    needs_replicated_loader: bool = False

    # Whether this baseline provides its own ensemble evaluation via
    # ``evaluate()``.  UnifiedEvaluator dispatches on this flag.
    # Override to ``True`` in baselines that call task.evaluate_samples()
    # internally (e.g. GRW, FullPower, WMMSE, DiffusionBaseline).
    has_ensemble_evaluate: bool = False

    def __init__(self, device: torch.device, save_dir: Optional[str] = None, **kwargs):
        self.device = device
        self.save_dir = save_dir

    @abstractmethod
    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, **kwargs):
        """Train/fit the baseline model."""
        pass

    @abstractmethod
    def predict(self, data) -> torch.Tensor:
        """Generate predictions for input data."""
        pass

    @abstractmethod
    def evaluate(
        self,
        loader: DataLoader,
        task,
        max_batches: Optional[int] = None,
        viz_save_dir: Optional[str] = None,
        eval_split_name: str = "eval",
    ) -> Dict[str, float]:
        """Evaluate on a dataset using task-specific metrics."""
        pass
    
    def save_checkpoint(self, path: str):
        """Save model state (if applicable)."""
        pass
    
    def load_checkpoint(self, path: str):
        """Load model state (if applicable)."""
        pass