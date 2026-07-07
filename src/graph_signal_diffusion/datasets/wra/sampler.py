"""Custom batch sampler for WRA dataset that groups samples by network."""

import random
import torch
from collections import defaultdict
from typing import Iterator, List, Optional
from torch.utils.data import Sampler


class NetworkGroupedBatchSampler(Sampler):
    """Batch sampler that groups samples by network for efficient evaluation.
    
    Each batch contains samples_per_network samples from each of
    batch_size/samples_per_network different (dataset, network) keys.
    This avoids collisions when multiple WRA datasets reuse the same numeric
    network_id and keeps dataset tagging consistent across diagnostics.
    
    Parameters
    ----------
    dataset : Dataset
        Dataset with samples attribute containing (dataset_name, network_id, sample_idx) tuples
    batch_size : int
        Total number of samples per batch
    samples_per_network : int
        Number of samples to draw from each network (K)
    shuffle : bool
        Whether to shuffle network order each epoch
    seed : int, optional
        Random seed for reproducibility (creates deterministic shuffle)
    generator : torch.Generator, optional
        Stateful random generator for reproducible but varying shuffles across epochs.
        If provided, takes precedence over seed. Generator state advances each epoch,
        producing different shuffle orders while remaining reproducible.
    
    Examples
    --------
    >>> # Create sampler for evaluation: 16 samples per batch, 4 from each network
    >>> sampler = NetworkGroupedBatchSampler(
    ...     dataset=val_dataset,
    ...     batch_size=16,
    ...     samples_per_network=4,
    ...     shuffle=False
    ... )
    >>> # Each batch will have 4 networks × 4 samples = 16 total
    >>> # Network grouping: [net_0, net_0, net_0, net_0, net_1, net_1, ...]
    """
    
    def __init__(
        self, 
        dataset, 
        batch_size: int, 
        samples_per_network: int, 
        shuffle: bool = True, 
        seed: Optional[int] = None,
        generator: Optional[torch.Generator] = None
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.samples_per_network = samples_per_network
        self.shuffle = shuffle
        self.seed = seed
        self.generator = generator
        # Isolated Python RNG — never touches the global random state.
        self._rng = random.Random(seed)

        if batch_size % samples_per_network != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be divisible by "
                f"samples_per_network ({samples_per_network})"
            )
        
        self.networks_per_batch = batch_size // samples_per_network
        
        # Group samples by composite key (dataset_name, network_id)
        self.network_to_indices = defaultdict(list)
        for idx, (dataset_name, network_id, sample_idx) in enumerate(dataset.samples):
            key = (str(dataset_name), int(network_id))
            self.network_to_indices[key].append(idx)
        
        self.networks = list(self.network_to_indices.keys())
        self.num_networks = len(self.networks)
    
    def __iter__(self) -> Iterator[List[int]]:
        """Generate batches grouped by network."""
        # Shuffle composite key order at start of each epoch
        networks = self.networks.copy()
        if self.shuffle:
            if self.generator is not None:
                # Use stateful generator for shuffling (advances state each epoch)
                # This produces different shuffle orders across epochs while being reproducible
                indices_tensor = torch.randperm(self.num_networks, generator=self.generator)
                networks = [self.networks[i] for i in indices_tensor.tolist()]
            else:
                # Fall back to seed-based shuffling via the instance RNG.
                # Produces the same shuffle every epoch when seed is fixed,
                # without poisoning the global Python random state.
                self._rng.shuffle(networks)
        
        # Generate batches
        for i in range(0, len(networks), self.networks_per_batch):
            batch_networks = networks[i:i + self.networks_per_batch]
            batch_indices = []
            
            for net_key in batch_networks:
                available_indices = self.network_to_indices[net_key]
                
                # Sample K indices from this network
                if len(available_indices) >= self.samples_per_network:
                    # Have enough samples - random sample without replacement
                    selected = self._rng.sample(available_indices, self.samples_per_network)
                else:
                    # Not enough samples - sample with replacement
                    selected = self._rng.choices(available_indices, k=self.samples_per_network)
                
                batch_indices.extend(selected)
            
            yield batch_indices
    
    def __len__(self) -> int:
        """Number of batches per epoch."""
        return (self.num_networks + self.networks_per_batch - 1) // self.networks_per_batch
