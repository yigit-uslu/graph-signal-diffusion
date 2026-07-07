"""
Dataset module for Wireless Resource Allocation diffusion training.

This module provides PyTorch Geometric Dataset classes for loading power allocation
samples from primal-dual training outputs and preparing them for diffusion model training.
"""

import torch
import numpy as np
from torch_geometric.data import Data, Dataset
from typing import Dict, List, Optional, Tuple
import os
import os.path as osp
import json
from pathlib import Path
import logging
import inspect


logger = logging.getLogger(__name__)

# Compatibility shim: older PyTorch versions do not accept weights_only in torch.load.
if "weights_only" not in inspect.signature(torch.load).parameters:
    _torch_load_original = torch.load

    def _torch_load_compat(*args, **kwargs):
        kwargs.pop("weights_only", None)
        return _torch_load_original(*args, **kwargs)

    torch.load = _torch_load_compat


def _safe_torch_load(path: str, weights_only: Optional[bool] = None):
    """Torch load compatible with older PyTorch versions lacking weights_only."""
    if weights_only is None:
        return torch.load(path)
    try:
        return torch.load(path, weights_only=weights_only)
    except TypeError:
        return torch.load(path)


def _normalize_precomputed_h_tmn(
    h_arr: np.ndarray,
    expected_m: int,
    expected_n: int,
    source_name: str,
) -> np.ndarray:
    """Normalize precomputed H arrays to shape (T, m, n)."""
    h_arr = np.asarray(h_arr, dtype=np.float32)

    if h_arr.ndim != 3:
        raise ValueError(f"Expected 3D H array in {source_name}, got shape {h_arr.shape}")

    # Preferred convention: (T, m, n).
    if h_arr.shape[1] == expected_m and h_arr.shape[2] == expected_n:
        return h_arr

    # Backward support: (m, n, T) -> transpose to (T, m, n).
    if h_arr.shape[0] == expected_m and h_arr.shape[1] == expected_n:
        return np.transpose(h_arr, (2, 0, 1))

    raise ValueError(
        f"H shape mismatch in {source_name}: got {h_arr.shape}, expected (T,{expected_m},{expected_n}) "
        f"or ({expected_m},{expected_n},T)."
    )


def _load_precomputed_h_tmn(raw_h_file: str, expected_m: int, expected_n: int) -> np.ndarray:
    """Load precomputed H_instantaneous from per-network file and normalize to shape (T, m, n)."""
    with np.load(raw_h_file, allow_pickle=False) as h_npz:
        if "H" not in h_npz:
            raise ValueError(f"Missing 'H' key in precomputed H file: {raw_h_file}")
        h_arr = h_npz["H"]
    return _normalize_precomputed_h_tmn(
        h_arr=h_arr,
        expected_m=expected_m,
        expected_n=expected_n,
        source_name=raw_h_file,
    )


class WirelessDataDiffusion(Data):
    """
    Custom PyTorch Geometric Data object for wireless power allocation.
    
    This class extends PyG's Data class to handle wireless-specific attributes
    during batching, similar to StocksDataDiffusion for stock data.
    
    Attributes
    ----------
    x : torch.Tensor
        Node features (n, T, F) where T=1, F=2 [direct_signal, total_interference]
    edge_index : torch.Tensor
        Graph connectivity (2, E)
    edge_weight : torch.Tensor
        Edge weights (E,)
    y : torch.Tensor
        Target power allocations (n, T, 1) where T=1, rescaled to [-0.5, 0.5]
    network_id : int
        Network identifier (for debugging/tracking)
    info : dict
        Metadata dictionary with channel parameters, system params, etc.
    """
    
    def __init__(
        self,
        x=None,
        edge_index=None,
        edge_weight=None,
        y=None,
        network_id=None,
        dataset_name=None,
        info=None,
    ):
        super().__init__(x=x, edge_index=edge_index, edge_weight=edge_weight)
        self.y = y
        self.network_id = network_id
        self.dataset_name = dataset_name
        self.info = info
    
    def __inc__(self, key, value, *args, **kwargs):
        """Define how to increment indices when batching."""
        if key in ['network_id', 'dataset_name']:
            # Don't increment categorical labels
            return 0
        return super().__inc__(key, value, *args, **kwargs)
    
    def __cat_dim__(self, key, value, *args, **kwargs):
        """Define concatenation dimension for each attribute."""
        if key == 'y':
            # Concatenate along node dimension (dim 0)
            return 0
        elif key in ['network_id', 'dataset_name']:
            # Stack into a list/tensor
            return 0
        elif key == 'info':
            # Don't concatenate info - keep as list
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)


class WRADataset(Dataset):
    """
    Dataset for wireless resource allocation diffusion training.
    
    Loads power allocation samples from processed .pt files and prepares them
    for UGNN-based diffusion model training.
    
    Parameters
    ----------
    root : str
        Root directory containing processed data
    dataset_names : list of str
        Names of sub-datasets to load (e.g., ["N_10_density_500_seed_0_Pmax_0.01_rmin_0.5"])
    split : str
        Data split: 'train', 'val', or 'test'
    train_fraction : float
        Fraction of networks to use for training (default: 0.8)
    val_fraction : float
        Fraction for validation (default: 0.1)
    test_fraction : float
        Fraction for testing (default: 0.1)
    cache_graphs : bool
        Whether to cache graph.pt files in memory (default: True)
    augment : bool
        Whether to apply data augmentation (default: False)
    augment_noise_std : float
        Std of Gaussian noise for augmentation (default: 0.0)
    augment_channel_noise : float
        Std of noise for channel features (default: 0.0)
    augment_edge_dropout : float
        Probability of dropping edges (default: 0.0)
    top_k : int, optional
        Keep top-K strongest interference edges per node during graph construction (default: 10)
    min_degree : int, optional
        Ensure at least this many edges per node during graph construction (default: 1)
    edge_normalize_method : str, optional
        Method to normalize edge weights: 'log', 'log1p', or 'none' (default: 'log1p')
    spectral_normalize_edge_weights : bool
        If True, spectral-normalize graph edge weights during processing so the
        weighted adjacency has maximum eigenvalue 1 (default: False)
    max_networks_per_split : int, optional
        Maximum networks per split (for debugging)
    max_samples_per_network : int, optional
        Maximum samples per network (for debugging)
    model_cond_channels : int, optional
        Configured conditioning feature width expected by the model.
        If provided, r_min conditioning is skipped when appending it would
        exceed this width.
    """
    
    def __init__(
        self,
        root: str,
        dataset_names: List[str],
        split: str = 'train',
        train_fraction: float = 0.8,
        val_fraction: float = 0.1,
        test_fraction: float = 0.1,
        cache_graphs: bool = True,
        augment: bool = False,
        augment_noise_std: float = 0.0,
        augment_channel_noise: float = 0.0,
        augment_edge_dropout: float = 0.0,
        top_k: int = 10,
        min_degree: int = 1,
        edge_normalize_method: str = 'log1p',
        spectral_normalize_edge_weights: bool = False,
        max_networks_per_split: Optional[int] = None,
        max_samples_per_network: Optional[int] = None,
        model_cond_channels: Optional[int] = None,
    ):
        self.root = root
        self.dataset_names = list(dataset_names)
        self.split = split
        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.cache_graphs = cache_graphs
        self.augment = augment and (split == 'train')  # Only augment during training
        self.augment_noise_std = augment_noise_std
        self.augment_channel_noise = augment_channel_noise
        self.augment_edge_dropout = augment_edge_dropout
        self.top_k = top_k
        self.min_degree = min_degree
        self.edge_normalize_method = edge_normalize_method
        self.spectral_normalize_edge_weights = spectral_normalize_edge_weights
        self.max_networks_per_split = max_networks_per_split
        self.max_samples_per_network = max_samples_per_network
        if model_cond_channels is None:
            self.model_cond_channels = None
        else:
            self.model_cond_channels = int(model_cond_channels)
            if self.model_cond_channels <= 0:
                raise ValueError(
                    f"model_cond_channels must be positive when set, got {self.model_cond_channels}."
                )
        
        # Initialize
        self.samples = []  # List of (dataset_name, network_id, sample_id) tuples
        self.graph_cache = {}  # Cache for graph.pt files
        self.metadata_cache = {}  # Cache for metadata.json files
        self._graph_rebuild_logged = set()  # Avoid repeated mismatch logs per network
        self._r_min_concat_skip_logged = set()
        self._feature_cap_mismatch_logged = set()
        
        super().__init__(root)
        
        # Build sample list
        self._build_sample_list()
        
        logger.info(f"WRADataset [{split}]: {len(self.samples)} samples from {len(dataset_names)} sub-dataset(s)")

    @staticmethod
    def _to_numpy_2d(value, key_name: str) -> np.ndarray:
        """Convert tensor/array-like value to a 2D numpy array."""
        if isinstance(value, torch.Tensor):
            arr = value.detach().cpu().numpy()
        else:
            arr = np.asarray(value)
        if arr.ndim != 2:
            raise ValueError(f"Expected {key_name} to be 2D, got shape {arr.shape}")
        return arr

    def _graph_params_match_current_config(self, graph_data: dict) -> bool:
        """Return whether cached graph parameters match current dataset config."""
        if not isinstance(graph_data, dict):
            return False

        saved_top_k = graph_data.get("top_k", None)
        saved_min_degree = graph_data.get("min_degree", None)
        saved_edge_norm = graph_data.get("edge_normalize_method", graph_data.get("normalize_method", None))
        saved_spectral_norm = graph_data.get("spectral_normalize_edge_weights", None)

        if saved_top_k is None or saved_min_degree is None or saved_edge_norm is None or saved_spectral_norm is None:
            return False

        return (
            int(saved_top_k) == int(self.top_k)
            and int(saved_min_degree) == int(self.min_degree)
            and str(saved_edge_norm) == str(self.edge_normalize_method)
            and bool(saved_spectral_norm) == bool(self.spectral_normalize_edge_weights)
        )

    def _maybe_rebuild_graph_for_current_config(self, graph_data: dict, cache_key: Tuple[str, int]) -> dict:
        """
        Rebuild graph tensors from H_ls/associations when config changed.

        This keeps WRA diffusion graph construction configurable via Hydra
        even when processed graph.pt files already exist on disk.
        """
        if self._graph_params_match_current_config(graph_data):
            return graph_data

        if "H_ls" not in graph_data or "associations" not in graph_data:
            if cache_key not in self._graph_rebuild_logged:
                logger.warning(
                    "Graph config mismatch for %s, but H_ls/associations are missing. "
                    "Using stored edge_index/edge_weight as-is.",
                    cache_key,
                )
                self._graph_rebuild_logged.add(cache_key)
            return graph_data

        from ...utils.graph_builder import build_interference_graph

        H_ls = self._to_numpy_2d(graph_data["H_ls"], "H_ls").astype(np.float32)
        associations = self._to_numpy_2d(graph_data["associations"], "associations").astype(np.float32)

        rebuilt_graph = build_interference_graph(
            H_ls=H_ls,
            associations=associations,
            top_k=int(self.top_k),
            min_degree=int(self.min_degree),
            normalize_method=str(self.edge_normalize_method),
            spectral_normalize_edge_weights=bool(self.spectral_normalize_edge_weights),
        )

        rebuilt_data = dict(graph_data)
        rebuilt_data["x"] = rebuilt_graph.x
        rebuilt_data["edge_index"] = rebuilt_graph.edge_index
        rebuilt_data["edge_weight"] = rebuilt_graph.edge_weight
        rebuilt_data["num_nodes"] = int(rebuilt_graph.num_nodes)
        rebuilt_data["top_k"] = int(self.top_k)
        rebuilt_data["min_degree"] = int(self.min_degree)
        rebuilt_data["edge_normalize_method"] = str(self.edge_normalize_method)
        rebuilt_data["spectral_normalize_edge_weights"] = bool(self.spectral_normalize_edge_weights)

        if cache_key not in self._graph_rebuild_logged:
            logger.info(
                "Rebuilt WRA graph for %s using current config: top_k=%d, min_degree=%d, "
                "edge_normalize_method=%s, spectral_normalize_edge_weights=%s",
                cache_key,
                int(self.top_k),
                int(self.min_degree),
                str(self.edge_normalize_method),
                bool(self.spectral_normalize_edge_weights),
            )
            self._graph_rebuild_logged.add(cache_key)

        return rebuilt_data
    
    @property
    def raw_file_names(self) -> List[str]:
        """Return list of raw files (checked during initialization)."""
        raw_files = []
        for dataset_name in self.dataset_names:
            raw_dir = osp.join(self.root, dataset_name, 'raw')
            raw_files.append(osp.join(raw_dir, 'collected_samples.npz'))
            raw_files.append(osp.join(raw_dir, 'config.yaml'))
            raw_files.append(osp.join(raw_dir, 'network_info.json'))
        return raw_files
    
    @property
    def processed_file_names(self) -> List[str]:
        """Return list of processed files (checked during initialization)."""
        # We check for existence of processed directories, not individual files
        processed_dirs = []
        for dataset_name in self.dataset_names:
            processed_dir = osp.join(self.root, dataset_name, 'processed')
            processed_dirs.append(processed_dir)
        return processed_dirs
    
    def _build_sample_list(self):
        """Build flat list of all samples from specified sub-datasets and split."""
        for dataset_name in self.dataset_names:
            # Load metadata
            metadata_path = osp.join(self.root, dataset_name, 'processed', 'metadata.json')
            
            if not osp.exists(metadata_path):
                raise FileNotFoundError(
                    f"Metadata not found: {metadata_path}. "
                    f"Please run the conversion script to process raw data."
                )
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.metadata_cache[dataset_name] = metadata
            
            network_ids = metadata.get('network_ids', list(range(metadata['num_networks'])))
            network_ids = [int(nid) for nid in network_ids]
            num_networks = len(network_ids)

            per_network_counts = metadata.get('samples_per_network_by_network', {})

            # Compute split indices for this sub-dataset
            train_end = int(num_networks * self.train_fraction)
            val_end = int(num_networks * (self.train_fraction + self.val_fraction))
            
            if self.split == 'train':
                split_network_ids = network_ids[:train_end]
            elif self.split == 'val':
                split_network_ids = network_ids[train_end:val_end]
            elif self.split == 'test':
                split_network_ids = network_ids[val_end:]
            else:
                raise ValueError(f"Unknown split: {self.split}")
            
            # Apply network limit if debugging
            if self.max_networks_per_split is not None:
                split_network_ids = split_network_ids[:self.max_networks_per_split]
            
            # Add all samples from these networks
            for network_id in split_network_ids:
                samples_per_network = int(
                    per_network_counts.get(str(network_id), metadata['samples_per_network'])
                )
                if self.max_samples_per_network is not None:
                    samples_per_network = min(samples_per_network, self.max_samples_per_network)
                for sample_id in range(samples_per_network):
                    self.samples.append((dataset_name, network_id, sample_id))
    
    def process(self):
        """
        Process raw data into .pt files.
        
        This method is called automatically by PyG if processed files don't exist.
        It converts collected_samples.npz into network_X/graph.pt and sample_Y.pt files.
        """
        logger.info("Processing raw data...")

        # Track canonical processed dirs for H_instantaneous deduplication.
        # Maps realpath(raw_h_dir) -> processed_dir of the first dataset that
        # materialized H timestep files from this raw source.  Subsequent
        # datasets whose raw_h_dir resolves to the same real path will create
        # symlinks instead of duplicating ~300 MB/network of timestep files.
        _h_canonical_processed_base: Dict[str, str] = {}

        for dataset_name in self.dataset_names:
            raw_dir = osp.join(self.root, dataset_name, 'raw')
            processed_dir = osp.join(self.root, dataset_name, 'processed')
            raw_h_dir = osp.join(raw_dir, 'h_instantaneous')
            raw_h_legacy_per_network_dir = osp.join(raw_dir, 'wra_h_instantaneous')
            raw_h_legacy_npz_path = osp.join(raw_dir, 'wra_h_instantaneous.npz')
            logger.info(
                f"Processing {dataset_name} with top_k={self.top_k}, min_degree={self.min_degree}, "
                f"edge_normalize_method={self.edge_normalize_method}, "
                f"spectral_normalize_edge_weights={self.spectral_normalize_edge_weights}"
            )
            
            # Load raw data
            samples_path = osp.join(raw_dir, 'collected_samples.npz')
            config_path = osp.join(raw_dir, 'config.yaml')
            network_info_path = osp.join(raw_dir, 'network_info.json')
            
            if not all(osp.exists(p) for p in [samples_path, config_path, network_info_path]):
                raise FileNotFoundError(
                    f"Raw data incomplete for {dataset_name}. "
                    f"Please run the conversion script first."
                )
            
            # Load samples into dict so keys are still available after file handle closes
            with np.load(samples_path, allow_pickle=True) as npz:
                data = {k: npz[k] for k in npz.files}
            
            # Load network info
            with open(network_info_path, 'r') as f:
                network_info = json.load(f)
            
            global_channel_version = str(network_info.get('channel_version', 'v2'))
            global_channel_params = network_info.get('channel_params', {})

            # Import here to avoid circular dependency
            from ...utils.graph_builder import build_interference_graph
            
            # Process each network
            os.makedirs(processed_dir, exist_ok=True)

            if 'network_ids' in data:
                network_ids = [int(nid) for nid in np.asarray(data['network_ids'], dtype=np.int64).tolist()]
            else:
                network_ids = sorted(
                    int(k.split('_')[1])
                    for k in data.keys()
                    if k.startswith('network_') and k.count('_') == 1
                )

            if len(network_ids) == 0:
                raise ValueError(
                    f"No network entries found in {samples_path}. "
                    "Expected keys like 'network_0' and optional 'network_ids'."
                )

            samples_per_network_by_network: Dict[str, int] = {}
            h_timesteps_by_network: Dict[str, int] = {}
            has_any_precomputed_h = False
            has_any_constraint_conditioning = False

            raw_h_npz = np.load(raw_h_legacy_npz_path, allow_pickle=False) if osp.exists(raw_h_legacy_npz_path) else None
            try:
                for net_id in network_ids:
                    net_id_str = f'network_{net_id}'
                    if net_id_str not in data:
                        raise ValueError(
                            f"Missing '{net_id_str}' entry in {samples_path}. "
                            f"network_ids includes {net_id} but payload is incomplete."
                        )

                    net_data_item = data[net_id_str]
                    net_data = net_data_item.item() if isinstance(net_data_item, np.ndarray) else net_data_item
                    if not isinstance(net_data, dict):
                        raise ValueError(
                            f"Entry '{net_id_str}' is not a dict payload. "
                            f"Found type={type(net_data)}"
                        )

                    network_dir = osp.join(processed_dir, f'network_{net_id}')
                    os.makedirs(network_dir, exist_ok=True)

                    # Get network info
                    info = network_info.get(net_id_str, {})
                    if not isinstance(info, dict):
                        raise ValueError(f"Malformed network_info entry for {net_id_str}")

                    # Build graph from large-scale fading
                    if 'H_ls' not in net_data or 'associations' not in net_data:
                        raise ValueError(
                            f"{net_id_str} missing required fields in converted raw data. "
                            "Required: H_ls, associations."
                        )
                    H_ls = np.asarray(net_data['H_ls'], dtype=np.float32)  # (m, n)
                    associations = np.asarray(net_data['associations'], dtype=np.float32)  # (m, n)
                    if H_ls.ndim != 2 or associations.ndim != 2:
                        raise ValueError(
                            f"{net_id_str} malformed matrices: H_ls{H_ls.shape}, associations{associations.shape}."
                        )

                    # Build graph using dataset config parameters (not raw data settings)
                    # This allows reprocessing with different graph construction settings
                    # independent of how the primal-dual data was originally generated.
                    graph_data = build_interference_graph(
                        H_ls=H_ls,
                        associations=associations,
                        top_k=self.top_k,
                        min_degree=self.min_degree,
                        normalize_method=self.edge_normalize_method,
                        spectral_normalize_edge_weights=self.spectral_normalize_edge_weights,
                    )

                    channel_version = str(info.get('channel_version', global_channel_version))
                    channel_params = info.get('channel_params', global_channel_params)

                    # Save graph.pt with metadata
                    graph_save_data = {
                        'x': graph_data.x,  # (n, 2)
                        'edge_index': graph_data.edge_index,  # (2, E)
                        'edge_weight': graph_data.edge_weight,  # (E,)
                        'num_nodes': graph_data.num_nodes,
                        # Metadata for evaluation
                        'H_ls': torch.tensor(H_ls, dtype=torch.float32),
                        'associations': torch.tensor(associations, dtype=torch.float32),
                        'tx_locations': torch.tensor(info.get('tx_locations'), dtype=torch.float32) if info.get('tx_locations') is not None else None,
                        'rx_locations': torch.tensor(info.get('rx_locations'), dtype=torch.float32) if info.get('rx_locations') is not None else None,
                        'network_seed': info.get('network_seed', net_data.get('network_seed')),
                        'n_links': info.get('n_links', int(associations.shape[0])),
                        'deployment_range': info.get('deployment_range', global_channel_params.get('deployment_range', 1000.0)),
                        'channel_params': channel_params,
                        'channel_version': channel_version,
                        # Graph construction parameters (controlled by dataset config)
                        'top_k': int(self.top_k),
                        'min_degree': int(self.min_degree),
                        'edge_normalize_method': str(self.edge_normalize_method),
                        'spectral_normalize_edge_weights': bool(self.spectral_normalize_edge_weights),
                    }

                    r_min_per_receiver = net_data.get('r_min_per_receiver')
                    if r_min_per_receiver is None:
                        # Fallback: construct uniform r_min vector from metadata
                        # when the build pipeline omitted per-receiver vectors
                        # (e.g. source_r_min_is_global_constant was True).
                        per_net_sp = info.get('system_params', {})
                        global_sp = network_info.get('system_params', {})
                        scalar_r_min = per_net_sp.get('r_min', global_sp.get('r_min'))
                        if scalar_r_min is not None:
                            n_receivers = int(H_ls.shape[1])
                            r_min_per_receiver = np.full(n_receivers, float(scalar_r_min), dtype=np.float32)
                    if r_min_per_receiver is not None:
                        r_min_per_receiver = np.asarray(r_min_per_receiver, dtype=np.float32).reshape(-1)
                        if r_min_per_receiver.shape[0] != H_ls.shape[1]:
                            raise ValueError(
                                f"{net_id_str}.r_min_per_receiver has length {r_min_per_receiver.shape[0]} "
                                f"but expected {H_ls.shape[1]} receivers."
                            )
                        graph_save_data['r_min_per_receiver'] = torch.tensor(
                            r_min_per_receiver, dtype=torch.float32
                        )
                        has_any_constraint_conditioning = True

                    # Optionally materialize precomputed H_instantaneous by timeslot.
                    h_num_timesteps = 0
                    h_timeslots_dir = osp.join(network_dir, 'H_instantaneous')

                    # Check if another dataset already materialized H from
                    # the same raw source (i.e. raw_h_dir is a symlink or
                    # resolves to the same real path).  If so, create a
                    # symlink instead of loading + re-saving ~300 MB.
                    real_raw_h = osp.realpath(raw_h_dir)
                    canonical_base = _h_canonical_processed_base.get(real_raw_h)
                    canonical_h_dir = (
                        osp.join(canonical_base, f'network_{net_id}', 'H_instantaneous')
                        if canonical_base is not None else None
                    )

                    if canonical_h_dir is not None and osp.isdir(canonical_h_dir):
                        # Symlink to the already-materialized canonical copy.
                        rel_target = osp.relpath(canonical_h_dir, network_dir)
                        os.symlink(rel_target, h_timeslots_dir)
                        h_num_timesteps = len([
                            f for f in os.listdir(canonical_h_dir)
                            if f.startswith('timestep_') and f.endswith('.pt')
                        ])
                        has_any_precomputed_h = True
                        logger.info(
                            "Symlinked H_instantaneous for %s -> %s (T=%d)",
                            net_id_str, rel_target, h_num_timesteps,
                        )
                    else:
                        # Load and materialize H timestep files.
                        h_tmn = None

                        # Current format: raw/h_instantaneous/network_<id>.npz
                        raw_h_file = osp.join(raw_h_dir, f'network_{net_id}.npz')
                        if osp.exists(raw_h_file):
                            h_tmn = _load_precomputed_h_tmn(
                                raw_h_file=raw_h_file,
                                expected_m=int(H_ls.shape[0]),
                                expected_n=int(H_ls.shape[1]),
                            )

                        # Legacy per-network format: raw/wra_h_instantaneous/network_<id>.npz
                        if h_tmn is None:
                            raw_h_file = osp.join(raw_h_legacy_per_network_dir, f'network_{net_id}.npz')
                            if osp.exists(raw_h_file):
                                h_tmn = _load_precomputed_h_tmn(
                                    raw_h_file=raw_h_file,
                                    expected_m=int(H_ls.shape[0]),
                                    expected_n=int(H_ls.shape[1]),
                                )

                        # Legacy single-file format: raw/wra_h_instantaneous.npz
                        if h_tmn is None and raw_h_npz is not None:
                            combined_key = f'network_{net_id}_H'
                            if combined_key in raw_h_npz:
                                h_tmn = _normalize_precomputed_h_tmn(
                                    h_arr=raw_h_npz[combined_key],
                                    expected_m=int(H_ls.shape[0]),
                                    expected_n=int(H_ls.shape[1]),
                                    source_name=f"{raw_h_legacy_npz_path}:{combined_key}",
                                )

                        if h_tmn is None:
                            logger.warning(
                                "No precomputed H_instantaneous found for network %s. "
                                "Searched: '%s' (current per-network dir), '%s' (legacy per-network dir), "
                                "'%s' (legacy combined npz). Diffusion training will proceed without "
                                "instantaneous channel data for this network.",
                                net_id_str,
                                raw_h_dir,
                                raw_h_legacy_per_network_dir,
                                raw_h_legacy_npz_path,
                            )

                        if h_tmn is not None:
                            os.makedirs(h_timeslots_dir, exist_ok=True)
                            for t_idx in range(h_tmn.shape[0]):
                                h_t = torch.tensor(h_tmn[t_idx], dtype=torch.float32)
                                torch.save(h_t, osp.join(h_timeslots_dir, f'timestep_{t_idx}.pt'))

                            h_num_timesteps = int(h_tmn.shape[0])
                            has_any_precomputed_h = True
                            # Record as canonical so later datasets can symlink.
                            if real_raw_h not in _h_canonical_processed_base:
                                _h_canonical_processed_base[real_raw_h] = processed_dir
                            logger.info(
                                f"Loaded precomputed H_instantaneous for {net_id_str}: "
                                f"T={h_num_timesteps}, shape per slot={tuple(h_tmn.shape[1:])}"
                            )

                    # Update graph metadata now that optional H materialization is resolved.
                    graph_save_data['has_h_instantaneous'] = bool(h_num_timesteps > 0)
                    graph_save_data['h_num_timesteps'] = int(h_num_timesteps)
                    torch.save(graph_save_data, osp.join(network_dir, 'graph.pt'))

                    # Save each power allocation sample
                    power_samples = net_data['power_samples']
                    if not isinstance(power_samples, list):
                        raise ValueError(
                            f"{net_id_str}.power_samples must be a list of sample dicts, "
                            f"found type={type(power_samples)}."
                        )
                    if len(power_samples) == 0:
                        raise ValueError(f"{net_id_str}.power_samples is empty.")

                    for sample_idx, sample in enumerate(power_samples):
                        if not isinstance(sample, dict):
                            raise ValueError(
                                f"{net_id_str}.power_samples[{sample_idx}] must be dict, "
                                f"found type={type(sample)}."
                            )

                        if 'power_per_transmitter' not in sample:
                            raise ValueError(
                                f"{net_id_str}.power_samples[{sample_idx}] missing power_per_transmitter."
                            )
                        power_tx = np.asarray(sample['power_per_transmitter'], dtype=np.float32)
                        if 'power_per_receiver' in sample and sample['power_per_receiver'] is not None:
                            power_rx = np.asarray(sample['power_per_receiver'], dtype=np.float32)
                        else:
                            power_rx = associations.T @ power_tx

                        sample_save_data = {
                            'power_alloc_per_receiver': power_rx,  # (n,)
                            'power_alloc_per_transmitter': power_tx,  # (m,)
                            'ergodic_rates': sample.get('rates', None),  # (n,) optional
                            'source_step': sample.get('source_step', None),
                        }
                        torch.save(sample_save_data, osp.join(network_dir, f'sample_{sample_idx}.pt'))

                    samples_per_network_by_network[str(net_id)] = len(power_samples)
                    h_timesteps_by_network[str(net_id)] = int(h_num_timesteps)
            finally:
                if raw_h_npz is not None:
                    raw_h_npz.close()
            
            # Save metadata
            sample_counts = list(samples_per_network_by_network.values())
            if len(sample_counts) == 0:
                raise ValueError(f"No processed samples found for dataset {dataset_name}.")

            has_effective_constraint_conditioning = bool(
                has_any_constraint_conditioning
                and (
                    self.model_cond_channels is None
                    or int(self.model_cond_channels) >= 3
                )
            )
            if has_any_constraint_conditioning and not has_effective_constraint_conditioning:
                logger.info(
                    "Detected r_min_per_receiver in raw data, but model_cond_channels=%s. "
                    "Keeping effective conditioning feature dim at 2 for %s.",
                    str(self.model_cond_channels),
                    dataset_name,
                )

            metadata = {
                'channel_version': global_channel_version,
                'num_networks': len(network_ids),
                'network_ids': network_ids,
                'samples_per_network': int(min(sample_counts)),
                'samples_per_network_by_network': samples_per_network_by_network,
                'has_h_instantaneous': bool(has_any_precomputed_h),
                'h_num_timesteps_by_network': h_timesteps_by_network,
                'P_max': network_info['system_params']['P_max'],
                'noise_var': network_info['system_params']['noise_var'],
                'BW': network_info['system_params']['BW'],
                'r_min': network_info['system_params'].get('r_min', 0.5),
                'has_constraint_conditioning': has_effective_constraint_conditioning,
                'has_constraint_conditioning_available': bool(has_any_constraint_conditioning),
                'feature_dim': 3 if has_effective_constraint_conditioning else 2,
                'model_cond_channels': self.model_cond_channels,
                'dataset_name': dataset_name,
            }
            
            with open(osp.join(processed_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(
                f"Processed {dataset_name}: {metadata['num_networks']} networks, "
                f"samples/network range=[{min(sample_counts)}, {max(sample_counts)}]"
            )
    
    def len(self) -> int:
        """Return number of samples in this split."""
        return len(self.samples)
    
    def get(self, idx: int) -> Data:
        """Load and return a single sample."""
        dataset_name, network_id, sample_id = self.samples[idx]
        
        # Load graph.pt (with caching)
        cache_key = (dataset_name, network_id)
        if self.cache_graphs and cache_key in self.graph_cache:
            graph_data = self.graph_cache[cache_key]
        else:
            graph_path = osp.join(
                self.root, dataset_name, 'processed', f'network_{network_id}', 'graph.pt'
            )
            graph_data = _safe_torch_load(graph_path, weights_only=True)

        graph_data = self._maybe_rebuild_graph_for_current_config(graph_data, cache_key)

        if self.cache_graphs:
            self.graph_cache[cache_key] = graph_data
        
        # Load sample.pt (weights_only=False needed for numpy arrays)
        sample_path = osp.join(
            self.root, dataset_name, 'processed', f'network_{network_id}', f'sample_{sample_id}.pt'
        )
        sample_data = _safe_torch_load(sample_path, weights_only=False)
        
        # Prepare node features:
        # - default: (n, 2) -> (n, 1, 2)
        # - conditioned: append r_min_per_receiver channel -> (n, 3) -> (n, 1, 3)
        x_base = graph_data['x'].clone()
        r_min_available = graph_data.get('r_min_per_receiver') is not None
        r_min_included = False
        if r_min_available:
            next_dim = int(x_base.shape[1]) + 1
            if self.model_cond_channels is not None and next_dim > int(self.model_cond_channels):
                if cache_key not in self._r_min_concat_skip_logged:
                    logger.warning(
                        "Skipping r_min_per_receiver concatenation for %s because "
                        "feature dim would become %d > model_cond_channels=%d.",
                        cache_key,
                        next_dim,
                        int(self.model_cond_channels),
                    )
                    self._r_min_concat_skip_logged.add(cache_key)
            else:
                r_min_vec = graph_data['r_min_per_receiver']
                if not isinstance(r_min_vec, torch.Tensor):
                    r_min_vec = torch.tensor(r_min_vec, dtype=x_base.dtype)
                r_min_vec = r_min_vec.to(dtype=x_base.dtype).reshape(-1)
                if r_min_vec.shape[0] != x_base.shape[0]:
                    raise ValueError(
                        f"r_min_per_receiver length {r_min_vec.shape[0]} does not match "
                        f"num_nodes {x_base.shape[0]} for dataset={dataset_name}, network={network_id}."
                    )
                x_base = torch.cat([x_base, r_min_vec.unsqueeze(1)], dim=1)
                r_min_included = True

        if (
            self.model_cond_channels is not None
            and int(x_base.shape[1]) > int(self.model_cond_channels)
            and cache_key not in self._feature_cap_mismatch_logged
        ):
            logger.warning(
                "Node feature dim %d exceeds model_cond_channels=%d for %s.",
                int(x_base.shape[1]),
                int(self.model_cond_channels),
                cache_key,
            )
            self._feature_cap_mismatch_logged.add(cache_key)

        x = x_base.unsqueeze(1)  # Add temporal dimension

        # Prepare target: (n,) -> (n, 1, 1)
        power_per_receiver = sample_data['power_alloc_per_receiver']

        # Normalize to [-0.5, 0.5]: y = p / P_max - 0.5
        # The task evaluator reverses this via the injected normalizer.
        metadata = self.metadata_cache[dataset_name]
        P_max = metadata['P_max']
        y = (power_per_receiver / P_max) - 0.5

        y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)  # (n, 1, 1)

        # Apply augmentation if enabled
        if self.augment:
            if self.augment_noise_std > 0:
                y = y + torch.randn_like(y) * self.augment_noise_std

            if self.augment_channel_noise > 0:
                x = x + torch.randn_like(x) * self.augment_channel_noise

        # Prepare info dict
        info = {
            'dataset_name': dataset_name,
            'network_id': network_id,
            'sample_id': sample_id,
            'P_max': self.metadata_cache[dataset_name]['P_max'],
            'network_seed': graph_data['network_seed'],
            'channel_version': graph_data.get('channel_version', self.metadata_cache[dataset_name].get('channel_version', 'v2')),
            'has_constraint_conditioning': bool(r_min_included),
            'has_constraint_conditioning_available': bool(r_min_available),
            'model_cond_channels': self.model_cond_channels,
        }
        
        # Create WirelessDataDiffusion object
        data = WirelessDataDiffusion(
            x=x,
            edge_index=graph_data['edge_index'],
            edge_weight=graph_data['edge_weight'],
            y=y,
            network_id=network_id,
            dataset_name=dataset_name,  # Add as separate attribute for batching
            info=info,
        )
        
        # Apply edge dropout if enabled
        if self.augment and self.augment_edge_dropout > 0:
            # Keep self-loops, randomly drop other edges
            num_edges = data.edge_index.shape[1]
            is_self_loop = data.edge_index[0] == data.edge_index[1]
            keep_mask = is_self_loop | (torch.rand(num_edges) > self.augment_edge_dropout)
            
            data.edge_index = data.edge_index[:, keep_mask]
            data.edge_weight = data.edge_weight[keep_mask]
        
        return data
