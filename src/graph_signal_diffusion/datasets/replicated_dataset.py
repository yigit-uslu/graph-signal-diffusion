"""
Wrapper dataset for replicating samples multiple times.

Useful for validation/test sets where we want to generate n_samples_per_input
predictions from the same input for computing statistics (mean, std, etc.).
"""

from typing import Any
from torch.utils.data import Dataset


class ReplicatedDataset(Dataset):
    """
    Wraps a dataset to replicate each sample n times.
    
    This allows generating multiple predictions per input during evaluation
    without manual duplication in the trainer. The dataloader will naturally
    batch replicated samples together.
    
    Example:
        Original dataset: [sample_0, sample_1, sample_2]  (length=3)
        ReplicatedDataset(original, n_replicas=10):
            - Length: 30
            - Indices 0-9 → sample_0
            - Indices 10-19 → sample_1
            - Indices 20-29 → sample_2
    
    Args:
        dataset: The original dataset to wrap
        n_replicas: Number of times to replicate each sample (default: 1, no replication)
        
    Usage:
        # Training: use original dataset
        train_dataset = SP500Stocks(...)
        
        # Validation/Test: wrap with replication
        val_dataset = ReplicatedDataset(val_dataset_original, n_replicas=10)
        test_dataset = ReplicatedDataset(test_dataset_original, n_replicas=10)
        
        # Dataloader batches naturally include replicas
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    """
    
    def __init__(self, dataset: Dataset, n_replicas: int = 1):
        self.dataset = dataset
        self.n_replicas = n_replicas
        self._original_length = len(dataset)
    
    def __len__(self) -> int:
        return self._original_length * self.n_replicas
    
    def __getitem__(self, idx: int) -> Any:
        # Map replicated index back to original index
        original_idx = idx // self.n_replicas
        return self.dataset[original_idx]
    
    def __repr__(self) -> str:
        return (
            f"ReplicatedDataset(\n"
            f"  dataset={self.dataset.__class__.__name__},\n"
            f"  original_length={self._original_length},\n"
            f"  n_replicas={self.n_replicas},\n"
            f"  total_length={len(self)}\n"
            f")"
        )
