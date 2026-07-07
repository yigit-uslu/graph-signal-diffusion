"""
Dataset wrapper for Wireless Resource Allocation with Primal-Dual training.

This module provides a PyTorch Dataset that wraps WirelessChannel instances
and prepares data for primal-dual GNN training.
"""

import logging
import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import Data
from typing import List, Dict, Tuple, Sequence, Optional
from pathlib import Path

import tqdm

from ..wra.channel import WirelessChannel

logger = logging.getLogger(__name__)
from ..wra.channel_factory import get_channel_class, normalize_channel_version
from ...utils.graph_builder import build_interference_graph


class WirelessData(Data):
    """
    Custom PyG Data object for wireless power allocation that handles batching logic.
    """
    def __init__(self, 
                 x=None, 
                 edge_index=None, 
                 edge_weight=None,
                 H_instantaneous=None,
                 H_l=None,
                 associations=None,
                 network_id=None,
                 network_seed=None):
        super().__init__(x=x, edge_index=edge_index, edge_weight=edge_weight)
        self.H_instantaneous = H_instantaneous
        self.H_l = H_l
        self.associations = associations
        self.network_id = network_id
        self.network_seed = network_seed
    
    def __inc__(self, key, value, *args, **kwargs):
        # Default increment behavior for most attributes
        return super().__inc__(key, value, *args, **kwargs)
    
    def __cat_dim__(self, key, value, *args, **kwargs):
        """
        Specify concatenation dimension for each attribute during batching.
        - None: Store in a list to support variable graph sizes (m, n can differ)
        - Default: Use PyG's default batching for graph attributes
        """
        if key in ['H_instantaneous', 'H_l', 'associations', 'network_id', 'network_seed']:
            # Store in list - each network can have different (m, n)
            # H_instantaneous: list of (T, m, n) tensors
            # H_l: list of (m, n) tensors
            # associations: list of (m, n) tensors
            # network_id: list of scalars
            # network_seed: list of scalars
            return None
        
        # Use default behavior for standard graph attributes (x, edge_index, edge_weight)
        return super().__cat_dim__(key, value, *args, **kwargs)


class WRAPrimalDualDataset(Dataset):
    """
    Dataset for primal-dual power allocation training.
    
    Each sample contains:
    - Graph representation (built from large-scale channel gains)
    - Instantaneous channel realizations over time (for rate computation)
    - Association matrix (TX-RX pairing)
    - Network ID (for dual variable indexing)
    
    Parameters
    ----------
    channels : list of WirelessChannel
        List of wireless channel instances
    num_timesteps : int
        Number of timesteps to sample per channel (default: 200)
    t_chunk : int, optional
        Timesteps returned by ``__getitem__`` via a random consecutive slice
        ``H_instantaneous[t0:t0+t_chunk]``. If None, defaults to
        ``num_timesteps`` (full sequence).
    top_k : int
        Number of edges to keep per node in sparsified graph (default: 10)
    min_degree : int
        Minimum degree for each node (default: 1)
    spectral_normalize_edge_weights : bool
        If True, scale edge weights so adjacency spectral radius is 1 (default: False)
    disable_small_scale_fading : bool
        If True, use only large-scale fading (no time variation) (default: False)
    
    Examples
    --------
    >>> from graph_signal_diffusion.datasets.wra.channel import WirelessChannel
    >>> channels = [WirelessChannel(n_links=10, seed=i) for i in range(100)]
    >>> dataset = WRAPrimalDualDataset(channels, num_timesteps=200)
    >>> print(len(dataset))
    100
    >>> sample = dataset[0]
    >>> print(sample.keys())
    dict_keys(['graph', 'H_instantaneous', 'associations', 'network_id'])
    """
    
    def __init__(
        self,
        channels: List[WirelessChannel],
        num_timesteps: int = 200,
        t_chunk: Optional[int] = None,
        top_k: int = 10,
        min_degree: int = 1,
        disable_small_scale_fading: bool = False,
        spectral_normalize_edge_weights: bool = False,
    ):
        self.channels = channels
        self.num_timesteps = num_timesteps
        self.t_chunk = self._resolve_t_chunk(t_chunk=t_chunk, num_timesteps=num_timesteps)
        self.top_k = top_k
        self.min_degree = min_degree
        self.spectral_normalize_edge_weights = spectral_normalize_edge_weights
        self.disable_small_scale_fading = disable_small_scale_fading
        
        # Pre-sample all channel realizations and build graphs
        logger.info("Preparing dataset with %d networks...", len(channels))
        self.samples = []
        
        for net_id, channel in tqdm.tqdm(enumerate(channels), total=len(channels), desc="Preparing dataset"):
            # Sample channel realization
            realization = channel.sample_realization(
                num_timesteps=num_timesteps,
                disable_small_scale_fading=disable_small_scale_fading
            )
            
            # Build graph from large-scale channel gains
            graph_data = build_interference_graph(
                H_ls=realization['H_l'],
                associations=realization['associations'],
                top_k=top_k,
                min_degree=min_degree,
                normalize_method='log1p',  # Use log(1+x) to keep edge weights non-negative
                spectral_normalize_edge_weights=spectral_normalize_edge_weights,
            )
            
            # Extract instantaneous channel gains (already power gains from channel.sample_realization())
            # Note: realization['H'] is already |H|^2 (power gains), no need to square again
            H_instantaneous = realization['H']  # (m, n, T) power gains
            H_instantaneous = np.transpose(H_instantaneous, (2, 0, 1))  # (T, m, n)
            
            # Extract large-scale channel gains
            H_l = realization['H_l']  # (m, n) power gains
            
            logger.debug(
                "Network %d: H shape %s  H_l shape %s  H max=%.3e  H_l max=%.3e",
                net_id,
                realization['H'].shape,
                realization['H_l'].shape,
                float(realization['H'].max()),
                float(realization['H_l'].max()),
            )

            # Create WirelessData object with all attributes
            # The __cat_dim__ logic is handled in the WirelessData class
            graph = WirelessData(
                x=graph_data.x,
                edge_index=graph_data.edge_index,
                edge_weight=graph_data.edge_weight,
                H_instantaneous=torch.tensor(H_instantaneous, dtype=torch.float32),
                H_l=torch.tensor(H_l, dtype=torch.float32),
                associations=torch.tensor(realization['associations'], dtype=torch.float32),
                network_id=torch.tensor(net_id, dtype=torch.long),
                network_seed=torch.tensor(channel.seed, dtype=torch.long)
            )
            
            self.samples.append(graph)

        logger.info("Dataset prepared: %d networks", len(self.samples))

    @staticmethod
    def _resolve_t_chunk(t_chunk: Optional[int], num_timesteps: int) -> int:
        """Resolve the timesteps-per-sample chunk length used in __getitem__."""
        total = int(num_timesteps)
        if total <= 0:
            raise ValueError(f"num_timesteps must be > 0, got {num_timesteps}.")
        if t_chunk is None:
            return total
        chunk = int(t_chunk)
        if chunk <= 0:
            raise ValueError(f"t_chunk must be > 0, got {t_chunk}.")
        if chunk > total:
            raise ValueError(
                f"t_chunk must be <= num_timesteps ({total}), got {chunk}."
            )
        return chunk
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Data:
        """
        Get a single network sample.
        
        Returns
        -------
        graph : Data
            PyTorch Geometric Data object with attributes:
            - x: Node features (n, 2)
            - edge_index: Edge connectivity (2, E)
            - edge_weight: Edge weights (E,)
            - H_instantaneous: Instantaneous channel gains (T, m, n)
            - H_l: Large-scale channel gains (m, n)
            - associations: Association matrix (m, n)
            - network_id: Network identifier (scalar)
        """
        sample = self.samples[idx]
        H_inst = getattr(sample, 'H_instantaneous', None)
        if not torch.is_tensor(H_inst):
            return sample

        total_timesteps = int(H_inst.shape[0])
        t_chunk = int(getattr(self, 't_chunk', total_timesteps))

        if t_chunk == total_timesteps:
            return sample
        if t_chunk <= 0:
            raise ValueError(f"t_chunk must be > 0, got {t_chunk}.")
        if t_chunk > total_timesteps:
            raise ValueError(
                f"t_chunk={t_chunk} exceeds available timesteps={total_timesteps} for idx={idx}."
            )

        max_start = total_timesteps - t_chunk
        start = int(torch.randint(0, max_start + 1, (1,)).item())
        end = start + t_chunk

        chunked_sample = sample.__class__()
        for key, value in sample:
            if key == 'H_instantaneous':
                chunked_sample[key] = value[start:end]
            else:
                chunked_sample[key] = value
        return chunked_sample
    
    @classmethod
    def from_seed_range(
        cls,
        num_networks: int,
        n_links: int,
        seed_start: int = 0,
        num_timesteps: int = 200,
        t_chunk: Optional[int] = None,
        top_k: int = 10,
        min_degree: int = 1,
        disable_small_scale_fading: bool = False,
        spectral_normalize_edge_weights: bool = False,
        channel_version: str = 'v2',
        **channel_kwargs
    ) -> 'WRAPrimalDualDataset':
        """
        Create dataset from a range of random seeds.
        
        Parameters
        ----------
        num_networks : int
            Number of networks to generate
        n_links : int
            Number of links per network
        seed_start : int
            Starting seed (default: 0)
        num_timesteps : int
            Number of timesteps for channel sampling
        t_chunk : int, optional
            Timesteps returned per ``__getitem__`` via random consecutive
            chunking. If None, uses full ``num_timesteps``.
        top_k : int
            Number of edges per node in graph
        min_degree : int
            Minimum degree per node
        disable_small_scale_fading : bool
            Whether to disable small-scale fading
        spectral_normalize_edge_weights : bool
            Whether to spectral-normalize graph edge weights
        channel_version : str
            Channel version: 'v1' for WirelessChannel, 'v2' for WirelessChannelV2 (default: 'v2')
        **channel_kwargs
            Additional keyword arguments for WirelessChannel (e.g., min_tx_rx_distance, max_tx_rx_distance)
        
        Returns
        -------
        dataset : WRAPrimalDualDataset
            Dataset instance
        """
        normalized_channel_version = normalize_channel_version(channel_version, default='v2')
        ChannelClass = get_channel_class(normalized_channel_version, default='v2')
        
        print(f"Generating {num_networks} wireless channels with {n_links} links each...")
        print(f"  Using {ChannelClass.__name__}")
        if channel_kwargs:
            print(f"  Channel parameters: {channel_kwargs}")
        
        channels = []
        
        for i in range(num_networks):
            channel = ChannelClass(
                n_links=n_links,
                seed=seed_start + i,
                **channel_kwargs
            )
            channels.append(channel)
        
        return cls(
            channels=channels,
            num_timesteps=num_timesteps,
            t_chunk=t_chunk,
            top_k=top_k,
            min_degree=min_degree,
            disable_small_scale_fading=disable_small_scale_fading,
            spectral_normalize_edge_weights=spectral_normalize_edge_weights,
        )
    
    @classmethod
    def from_seed_range_with_merging(
        cls,
        num_networks: int,
        n_links_per_subnet: int,
        num_subnets: int,
        seed_start: int = 0,
        num_timesteps: int = 200,
        t_chunk: Optional[int] = None,
        top_k: int = 10,
        min_degree: int = 1,
        disable_small_scale_fading: bool = False,
        spectral_normalize_edge_weights: bool = False,
        subnet_spacing: float = 150.0,
        layout: str = 'circular',
        channel_version: str = 'v2',
        **channel_kwargs
    ) -> 'WRAPrimalDualDataset':
        """
        Create dataset by merging smaller subnetworks into larger networks.
        
        This approach is more reliable for large networks (e.g., 200 links) because:
        - Smaller subnetworks deploy successfully with high probability
        - Spatial separation naturally preserves interference dominance
        - Final network maintains realistic heterogeneous structure
        
        Parameters
        ----------
        num_networks : int
            Number of merged networks to generate
        n_links_per_subnet : int
            Number of links in each subnetwork (e.g., 50)
        num_subnets : int
            Number of subnetworks to merge (e.g., 4 for 200 total links)
        seed_start : int
            Starting seed for subnetwork generation
        num_timesteps : int
            Number of timesteps per channel realization
        t_chunk : int, optional
            Timesteps returned per ``__getitem__`` via random consecutive
            chunking. If None, uses full ``num_timesteps``.
        top_k : int
            Number of edges per node in graph
        min_degree : int
            Minimum degree per node
        disable_small_scale_fading : bool
            Whether to disable small-scale fading
        spectral_normalize_edge_weights : bool
            Whether to spectral-normalize graph edge weights
        subnet_spacing : float
            Spacing between subnetworks in meters (default: 150m)
        layout : str
            Merging layout: 'circular' or 'linear' (default: 'circular')
        channel_version : str
            Channel version: 'v1' for WirelessChannel, 'v2' for WirelessChannelV2 (default: 'v2')
        **channel_kwargs
            Additional keyword arguments for WirelessChannel (e.g., min_tx_rx_distance, max_tx_rx_distance)
        
        Returns
        -------
        dataset : WRAPrimalDualDataset
            Dataset instance with merged networks
            
        Examples
        --------
        >>> # Create 32 networks of 200 links each by merging 4 subnetworks of 50 links
        >>> dataset = WRAPrimalDualDataset.from_seed_range_with_merging(
        ...     num_networks=32,
        ...     n_links_per_subnet=50,
        ...     num_subnets=4,
        ...     seed_start=42,
        ...     deployment_range=700.0,
        ...     subnet_spacing=150.0,
        ...     layout='circular'
        ... )
        """
        from graph_signal_diffusion.datasets.wra.channel import merge_channels
        
        normalized_channel_version = normalize_channel_version(channel_version, default='v2')
        ChannelClass = get_channel_class(normalized_channel_version, default='v2')
        
        total_links = n_links_per_subnet * num_subnets
        print(f"Generating {num_networks} wireless channels with {total_links} links each...")
        print(f"  Strategy: Merge {num_subnets} subnetworks of {n_links_per_subnet} links")
        print(f"  Layout: {layout} with {subnet_spacing}m spacing")
        print(f"  Using {ChannelClass.__name__}")
        if channel_kwargs:
            print(f"  Channel parameters: {channel_kwargs}")
        
        channels = []
        
        for i in range(num_networks):
            # Create subnetworks
            subnets = []
            for j in range(num_subnets):
                subnet_seed = seed_start + i * num_subnets + j
                subnet = ChannelClass(
                    n_links=n_links_per_subnet,
                    seed=subnet_seed,
                    **channel_kwargs
                )
                subnets.append(subnet)
            
            # Merge subnetworks
            merged_channel = merge_channels(
                subnets,
                spacing=subnet_spacing,
                layout=layout
            )
            channels.append(merged_channel)
            
            print(f"  Network {i+1}/{num_networks}: {total_links} links (merged from {num_subnets} subnets)")
        
        return cls(
            channels=channels,
            num_timesteps=num_timesteps,
            t_chunk=t_chunk,
            top_k=top_k,
            min_degree=min_degree,
            disable_small_scale_fading=disable_small_scale_fading,
            spectral_normalize_edge_weights=spectral_normalize_edge_weights,
        )
    
    def save(self, path: str) -> None:
        """
        Save dataset to disk.
        
        Parameters
        ----------
        path : str
            Path to save dataset
        """
        import pickle
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Dataset saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'WRAPrimalDualDataset':
        """
        Load dataset from disk.
        
        Parameters
        ----------
        path : str
            Path to load dataset from
        
        Returns
        -------
        dataset : WRAPrimalDualDataset
            Loaded dataset
        """
        import pickle
        with open(path, 'rb') as f:
            dataset = pickle.load(f)
        print(f"Dataset loaded from {path}")
        return dataset


class WRAConstraintProfilePrimalDualDataset(Dataset):
    """
    Dataset wrapper that expands base networks across constraint profiles.

    Expansion invariant (base-major):
      expanded_id = base_id * num_profiles + profile_id

    Each expanded sample appends a per-receiver r_min feature channel:
      x: (n, 2) -> (n, 3)
    """

    def __init__(
        self,
        base_dataset: WRAPrimalDualDataset,
        constraint_profiles: Sequence[float | Sequence[float]],
        *,
        r_min_feature_scale: float = 1.0,
        profile_names: Optional[Sequence[str]] = None,
    ):
        if len(base_dataset) == 0:
            raise ValueError("base_dataset must contain at least one network.")
        if len(constraint_profiles) == 0:
            raise ValueError("constraint_profiles must be non-empty.")

        self.base_dataset = base_dataset
        self.num_base_networks = len(base_dataset)
        self.r_min_feature_scale = float(r_min_feature_scale)

        sample0 = base_dataset[0]
        self.num_receivers = int(sample0.x.shape[0])

        self.constraint_profile_table = self._canonicalize_profiles(
            profiles=constraint_profiles,
            num_receivers=self.num_receivers,
        )  # (P, n)
        self.num_profiles = int(self.constraint_profile_table.shape[0])

        if profile_names is None:
            self.profile_names = [f"profile_{i}" for i in range(self.num_profiles)]
        else:
            if len(profile_names) != self.num_profiles:
                raise ValueError(
                    "profile_names length must match number of profiles: "
                    f"{len(profile_names)} vs {self.num_profiles}."
                )
            self.profile_names = [str(name) for name in profile_names]

    @staticmethod
    def _canonicalize_profiles(
        *,
        profiles: Sequence[float | Sequence[float]],
        num_receivers: int,
    ) -> torch.Tensor:
        canonical_rows: list[torch.Tensor] = []
        for idx, profile in enumerate(profiles):
            row = torch.as_tensor(profile, dtype=torch.float32)
            if row.ndim == 0:
                row = row.repeat(num_receivers)
            elif row.ndim == 1 and row.shape[0] == num_receivers:
                pass
            else:
                raise ValueError(
                    f"Profile {idx} must be scalar or length-{num_receivers} vector; "
                    f"got shape {tuple(row.shape)}."
                )
            canonical_rows.append(row)
        return torch.stack(canonical_rows, dim=0)

    def __len__(self) -> int:
        return self.num_base_networks * self.num_profiles

    def decode_expanded_id(self, expanded_id: int) -> tuple[int, int]:
        expanded = int(expanded_id)
        if expanded < 0 or expanded >= len(self):
            raise IndexError(f"expanded_id out of range: {expanded_id}")
        base_id = expanded // self.num_profiles
        profile_id = expanded % self.num_profiles
        return base_id, profile_id

    def encode_expanded_id(self, base_id: int, profile_id: int) -> int:
        if base_id < 0 or base_id >= self.num_base_networks:
            raise IndexError(f"base_id out of range: {base_id}")
        if profile_id < 0 or profile_id >= self.num_profiles:
            raise IndexError(f"profile_id out of range: {profile_id}")
        return int(base_id) * self.num_profiles + int(profile_id)

    def get_profile_vector(self, profile_id: int) -> np.ndarray:
        if profile_id < 0 or profile_id >= self.num_profiles:
            raise IndexError(f"profile_id out of range: {profile_id}")
        return self.constraint_profile_table[profile_id].cpu().numpy().copy()

    def get_profile_name(self, profile_id: int) -> str:
        if profile_id < 0 or profile_id >= self.num_profiles:
            raise IndexError(f"profile_id out of range: {profile_id}")
        return self.profile_names[profile_id]

    def get_r_min_table(self) -> torch.Tensor:
        """
        Return expanded per-network/per-receiver threshold table.

        Shape: (num_base_networks * num_profiles, num_receivers)
        Ordering is base-major to match expanded network IDs.
        """
        return self.constraint_profile_table.repeat(self.num_base_networks, 1).clone()

    def __getitem__(self, idx: int) -> Data:
        base_id, profile_id = self.decode_expanded_id(idx)
        base_sample = self.base_dataset[base_id].clone()

        profile_vec = self.constraint_profile_table[profile_id].to(
            device=base_sample.x.device,
            dtype=base_sample.x.dtype,
        )
        conditioned_channel = (profile_vec * self.r_min_feature_scale).unsqueeze(1)  # (n, 1)
        base_sample.x = torch.cat([base_sample.x, conditioned_channel], dim=1)

        expanded_id = self.encode_expanded_id(base_id, profile_id)
        base_sample.network_id = torch.tensor(expanded_id, dtype=torch.long)
        return base_sample


def create_train_val_split(
    dataset: WRAPrimalDualDataset,
    val_fraction: float = 0.2,
    seed: int = 42
) -> Tuple[WRAPrimalDualDataset, WRAPrimalDualDataset]:
    """
    Split dataset into train and validation sets.
    
    Parameters
    ----------
    dataset : WRAPrimalDualDataset
        Full dataset
    val_fraction : float
        Fraction of data to use for validation (default: 0.2)
    seed : int
        Random seed for split (default: 42)
    
    Returns
    -------
    train_dataset : WRAPrimalDualDataset
        Training dataset
    val_dataset : WRAPrimalDualDataset
        Validation dataset
    """
    rng = np.random.RandomState(seed)
    num_samples = len(dataset)
    indices = rng.permutation(num_samples)
    
    num_val = int(num_samples * val_fraction)
    val_indices = indices[:num_val]
    train_indices = indices[num_val:]
    
    # Create subset datasets
    train_dataset = WRAPrimalDualDataset.__new__(WRAPrimalDualDataset)
    train_dataset.channels = [dataset.channels[i] for i in train_indices]
    train_dataset.samples = [dataset.samples[i] for i in train_indices]
    train_dataset.num_timesteps = dataset.num_timesteps
    train_dataset.t_chunk = getattr(dataset, 't_chunk', dataset.num_timesteps)
    train_dataset.top_k = dataset.top_k
    train_dataset.min_degree = dataset.min_degree
    train_dataset.disable_small_scale_fading = dataset.disable_small_scale_fading
    train_dataset.spectral_normalize_edge_weights = getattr(
        dataset, 'spectral_normalize_edge_weights', False
    )
    
    val_dataset = WRAPrimalDualDataset.__new__(WRAPrimalDualDataset)
    val_dataset.channels = [dataset.channels[i] for i in val_indices]
    val_dataset.samples = [dataset.samples[i] for i in val_indices]
    val_dataset.num_timesteps = dataset.num_timesteps
    val_dataset.t_chunk = getattr(dataset, 't_chunk', dataset.num_timesteps)
    val_dataset.top_k = dataset.top_k
    val_dataset.min_degree = dataset.min_degree
    val_dataset.disable_small_scale_fading = dataset.disable_small_scale_fading
    val_dataset.spectral_normalize_edge_weights = getattr(
        dataset, 'spectral_normalize_edge_weights', False
    )
    
    print(f"Split dataset: {len(train_dataset)} train, {len(val_dataset)} val")
    
    return train_dataset, val_dataset
