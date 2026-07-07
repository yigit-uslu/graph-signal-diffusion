"""Custom batch sampler for ReplicatedDataset that groups replicas together."""

import random
import torch
from typing import Iterator, List, Optional
from torch.utils.data import Sampler

from .replicated_dataset import ReplicatedDataset


class ReplicatedGroupedBatchSampler(Sampler):
    """Batch sampler that groups replicas of the same sample together.
    
    Designed to work with ReplicatedDataset where consecutive indices map to
    replicas of the same original sample. Each batch contains replicas from
    `originals_per_batch` original samples, for a total batch size of
    `originals_per_batch × n_replicas`.
    
    This guarantees that all replicas of each original sample are kept together
    within the batch, enabling efficient statistics computation (mean, std, CI).
    
    Parameters
    ----------
    dataset : ReplicatedDataset
        ReplicatedDataset with n_replicas per original sample
    originals_per_batch : int, default=1
        Number of original samples to include in each batch.
        Actual batch size will be originals_per_batch × n_replicas
    shuffle : bool, default=False
        Whether to shuffle the order of original samples each epoch
    seed : int, optional
        Random seed for reproducibility (creates deterministic shuffle)
    generator : torch.Generator, optional
        Stateful random generator for reproducible but varying shuffles across epochs.
        If provided, takes precedence over seed. Generator state advances each epoch,
        producing different shuffle orders while remaining reproducible.
    
    Examples
    --------
    >>> # Original dataset: 100 samples
    >>> # ReplicatedDataset: 100 × 10 = 1000 samples
    >>> dataset = ReplicatedDataset(original_dataset, n_replicas=10)
    >>> 
    >>> # Batch 1 original sample at a time (batch_size = 10)
    >>> sampler = ReplicatedGroupedBatchSampler(dataset, originals_per_batch=1)
    >>> loader = DataLoader(dataset, batch_sampler=sampler)
    >>> for batch in loader:
    ...     assert batch.shape[0] == 10  # 1 original × 10 replicas
    >>> 
    >>> # Batch 4 original samples at a time (batch_size = 40)
    >>> sampler = ReplicatedGroupedBatchSampler(dataset, originals_per_batch=4)
    >>> loader = DataLoader(dataset, batch_sampler=sampler)
    >>> for batch in loader:
    ...     assert batch.shape[0] == 40  # 4 originals × 10 replicas each
    
    Notes
    -----
    - Actual batch_size = originals_per_batch × n_replicas
    - Replicas of each original sample are kept consecutive in the batch
    - When using batch_sampler, DataLoader's batch_size, shuffle, and drop_last
      arguments must NOT be specified (they're controlled by the sampler)
    - ReplicatedDataset maps indices: [0..n_replicas-1] → sample_0,
      [n_replicas..2*n_replicas-1] → sample_1, etc.
    """
    
    def __init__(
        self, 
        dataset: ReplicatedDataset,
        originals_per_batch: int = 1,
        shuffle: bool = False, 
        seed: Optional[int] = None,
        generator: Optional[torch.Generator] = None
    ):
        if not isinstance(dataset, ReplicatedDataset):
            raise TypeError(
                f"Expected ReplicatedDataset, got {type(dataset).__name__}. "
                f"Use ReplicatedGroupedBatchSampler only with ReplicatedDataset."
            )
        
        if originals_per_batch < 1:
            raise ValueError(
                f"originals_per_batch must be >= 1, got {originals_per_batch}"
            )
        
        self.dataset = dataset
        self.n_replicas = dataset.n_replicas
        self.original_length = dataset._original_length
        self.originals_per_batch = originals_per_batch
        self.shuffle = shuffle
        self.seed = seed
        self.generator = generator
    
    def __iter__(self) -> Iterator[List[int]]:
        """Generate batches of replica groups."""
        # Create list of original sample indices
        original_indices = list(range(self.original_length))
        
        # Shuffle order of original samples if requested
        if self.shuffle:
            if self.generator is not None:
                # Use stateful generator for shuffling (advances state each epoch)
                # This produces different shuffle orders across epochs while being reproducible
                indices_tensor = torch.randperm(self.original_length, generator=self.generator)
                original_indices = indices_tensor.tolist()
            else:
                # Fall back to seed-based shuffling (same shuffle each epoch if seed is set)
                if self.seed is not None:
                    random.seed(self.seed)
                random.shuffle(original_indices)
        
        # Batch originals_per_batch original samples together (each with their replicas)
        for i in range(0, len(original_indices), self.originals_per_batch):
            batch_originals = original_indices[i:i + self.originals_per_batch]
            batch_indices = []
            
            for orig_idx in batch_originals:
                # Map original index to replica indices
                # ReplicatedDataset stores replicas consecutively:
                # indices [orig_idx * n_replicas : (orig_idx + 1) * n_replicas]
                start_idx = orig_idx * self.n_replicas
                replica_indices = list(range(start_idx, start_idx + self.n_replicas))
                batch_indices.extend(replica_indices)
            
            yield batch_indices
    
    def __len__(self) -> int:
        """Number of batches per epoch."""
        return (self.original_length + self.originals_per_batch - 1) // self.originals_per_batch
    
    def __repr__(self) -> str:
        batch_size = self.originals_per_batch * self.n_replicas
        return (
            f"ReplicatedGroupedBatchSampler(\n"
            f"  original_samples={self.original_length},\n"
            f"  n_replicas={self.n_replicas},\n"
            f"  originals_per_batch={self.originals_per_batch},\n"
            f"  batches={len(self)},\n"
            f"  batch_size={batch_size},\n"
            f"  shuffle={self.shuffle}\n"
            f")"
        )
