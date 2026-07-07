"""
Data module builder for Wireless Resource Allocation dataset.

This module provides the WRABuilder class that implements the dataset builder
interface for creating train/val/test datasets and dataloaders.
"""

from __future__ import annotations
import torch
import json
import os.path as osp
from typing import Dict, Optional
from torch_geometric.loader import DataLoader

from graph_signal_diffusion.datasets import DATASET_REGISTRY
from graph_signal_diffusion.datasets.base import DatasetConfig
from graph_signal_diffusion.datasets.wra.dataset import WRADataset
from graph_signal_diffusion.datasets.wra.sampler import NetworkGroupedBatchSampler

import math
import logging
logger = logging.getLogger(__name__)


def _merge_r_min_per_dataset(*dicts):
    """Merge r_min_per_dataset dicts from train/val/test splits with conflict detection."""
    merged = {}
    for d in dicts:
        for ds_name, r_val in d.items():
            if ds_name in merged and not math.isclose(merged[ds_name], r_val, rel_tol=1e-6, abs_tol=1e-9):
                raise ValueError(
                    f"r_min conflict for dataset '{ds_name}': "
                    f"train/val/test splits report {merged[ds_name]} vs {r_val}"
                )
            merged[ds_name] = r_val
    return merged


@DATASET_REGISTRY.register("wra")
class WRABuilder:
    """
    Builder for Wireless Resource Allocation dataset.
    
    Implements the dataset builder interface to create datasets and dataloaders
    for diffusion model training on power allocation samples.
    """
    
    def __init__(self):
        self.normalizer: Optional[object] = None  # WRA handles normalization internally
        self.dataset_info = None
    
    def build_datasets(self, cfg: DatasetConfig) -> Dict[str, WRADataset]:
        """
        Build train, validation, and test datasets.
        
        Parameters
        ----------
        cfg : DatasetConfig
            Configuration object with dataset parameters
        
        Returns
        -------
        datasets : dict
            Dictionary with keys 'train', 'val', 'test', 'train-val'
        """
        if not cfg.datasets:
            raise ValueError(
                "cfg.datasets is empty. Populate it with at least one content-addressed path "
                "produced by build_diffusion_dataset.py, e.g.:\n"
                "  datasets: [\"debug/pdc_npz_abc123def456\"]\n"
                "The scenario component is the training config name with the 'wra_' prefix stripped."
            )

        logger.info(f"Building WRA datasets from: {cfg.root}")
        logger.info(f"Sub-datasets: {cfg.datasets}")

        # Graph construction knobs (Hydra-configurable).
        # These control how PyG graphs are built from H_ls + associations.
        top_k_cfg = cfg.get("top_k", 10)
        min_degree_cfg = cfg.get("min_degree", 1)
        edge_norm_cfg = cfg.get("edge_normalize_method", "log1p")
        spectral_norm_cfg = cfg.get("spectral_normalize_edge_weights", False)

        graph_top_k = 10 if top_k_cfg is None else int(top_k_cfg)
        graph_min_degree = 1 if min_degree_cfg is None else int(min_degree_cfg)
        graph_edge_normalize_method = "log1p" if edge_norm_cfg is None else str(edge_norm_cfg)
        graph_spectral_normalize_edge_weights = bool(spectral_norm_cfg)
        model_cond_channels_cfg = cfg.get("model_cond_channels", None)
        model_cond_channels = (
            None if model_cond_channels_cfg is None else int(model_cond_channels_cfg)
        )

        logger.info(
            "WRA graph config: top_k=%d, min_degree=%d, edge_normalize_method=%s, "
            "spectral_normalize_edge_weights=%s, model_cond_channels=%s",
            graph_top_k,
            graph_min_degree,
            graph_edge_normalize_method,
            graph_spectral_normalize_edge_weights,
            str(model_cond_channels),
        )

        # Create datasets for each split
        train_dataset = WRADataset(
            root=cfg.root,
            dataset_names=cfg.datasets,
            split='train',
            train_fraction=cfg.train_fraction,
            val_fraction=cfg.val_fraction,
            test_fraction=cfg.test_fraction,
            cache_graphs=cfg.cache_graphs,
            augment=cfg.augment,
            augment_noise_std=cfg.augment_noise_std,
            augment_channel_noise=cfg.augment_channel_noise,
            augment_edge_dropout=cfg.augment_edge_dropout,
            top_k=graph_top_k,
            min_degree=graph_min_degree,
            edge_normalize_method=graph_edge_normalize_method,
            spectral_normalize_edge_weights=graph_spectral_normalize_edge_weights,
            model_cond_channels=model_cond_channels,
            max_networks_per_split=cfg.get('max_networks_per_split', None),
            max_samples_per_network=cfg.get('max_samples_per_network', None),
        )

        val_dataset = WRADataset(
            root=cfg.root,
            dataset_names=cfg.datasets,
            split='val',
            train_fraction=cfg.train_fraction,
            val_fraction=cfg.val_fraction,
            test_fraction=cfg.test_fraction,
            cache_graphs=cfg.cache_graphs,
            augment=False,
            top_k=graph_top_k,
            min_degree=graph_min_degree,
            edge_normalize_method=graph_edge_normalize_method,
            spectral_normalize_edge_weights=graph_spectral_normalize_edge_weights,
            model_cond_channels=model_cond_channels,
            max_networks_per_split=cfg.get('max_networks_per_split', None),
            max_samples_per_network=cfg.get('max_samples_per_network', None),
        )

        test_dataset = WRADataset(
            root=cfg.root,
            dataset_names=cfg.datasets,
            split='test',
            train_fraction=cfg.train_fraction,
            val_fraction=cfg.val_fraction,
            test_fraction=cfg.test_fraction,
            cache_graphs=cfg.cache_graphs,
            augment=False,
            top_k=graph_top_k,
            min_degree=graph_min_degree,
            edge_normalize_method=graph_edge_normalize_method,
            spectral_normalize_edge_weights=graph_spectral_normalize_edge_weights,
            model_cond_channels=model_cond_channels,
            max_networks_per_split=cfg.get('max_networks_per_split', None),
            max_samples_per_network=cfg.get('max_samples_per_network', None),
        )
        
        # Train-val dataset: full train split; sampler picks any train-split network
        train_val_dataset = train_dataset

        logger.info(f"Train dataset: {len(train_dataset)} samples")
        logger.info(f"Validation dataset: {len(val_dataset)} samples")
        logger.info(f"Test dataset: {len(test_dataset)} samples")
        logger.info(f"Train-Val dataset: {len(train_val_dataset)} samples (full train split)")
        
        # Build comprehensive dataset info from ALL splits
        logger.info("Extracting metadata from all dataset splits...")
        train_info = self._extract_dataset_info(train_dataset)
        val_info = self._extract_dataset_info(val_dataset)
        test_info = self._extract_dataset_info(test_dataset)
        
        # Merge metadata using composite keys (dataset_name, network_id)
        self.dataset_info = {
            'system_params': train_info['system_params'],    # Same across all splits
            'channel_params': train_info['channel_params'],  # Same across all splits
            'channel_version': train_info.get('channel_version', 'v2'),
            'network_seeds': {
                **train_info['network_seeds'],
                **val_info['network_seeds'],
                **test_info['network_seeds'],
            },
            'associations': {
                **train_info['associations'],
                **val_info['associations'],
                **test_info['associations'],
            },
            'h_ls_gains': {
                **train_info.get('h_ls_gains', {}),
                **val_info.get('h_ls_gains', {}),
                **test_info.get('h_ls_gains', {}),
            },
            'channel_versions': {
                **train_info.get('channel_versions', {}),
                **val_info.get('channel_versions', {}),
                **test_info.get('channel_versions', {}),
            },
            'h_timeslot_dirs': {
                **train_info.get('h_timeslot_dirs', {}),
                **val_info.get('h_timeslot_dirs', {}),
                **test_info.get('h_timeslot_dirs', {}),
            },
            'h_num_timesteps': {
                **train_info.get('h_num_timesteps', {}),
                **val_info.get('h_num_timesteps', {}),
                **test_info.get('h_num_timesteps', {}),
            },
            'channel_cache_info': {
                **train_info.get('channel_cache_info', {}),
                **val_info.get('channel_cache_info', {}),
                **test_info.get('channel_cache_info', {}),
            },
            'r_min_per_dataset': _merge_r_min_per_dataset(
                train_info.get('r_min_per_dataset', {}),
                val_info.get('r_min_per_dataset', {}),
                test_info.get('r_min_per_dataset', {}),
            ),
        }
        
        logger.info(f"Total unique networks in metadata: {len(self.dataset_info['network_seeds'])}")
        
        return {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
            "train-val": train_val_dataset,
        }
    
    def _extract_dataset_info(self, dataset: WRADataset) -> Dict:
        """Extract comprehensive metadata for task evaluation.
        
        Parameters
        ----------
        dataset : WRADataset
            Dataset to extract info from (typically train_dataset)
        
        Returns
        -------
        dict
            Dictionary containing:
            - system_params: P_max, noise_var, BW, r_min
            - channel_params: deployment_range, path_loss_exponent_short/long, etc.
            - network_seeds: {network_id: seed} mapping
            - associations: {network_id: associations_matrix} mapping
        """
        logger.info("Extracting dataset info for task evaluation...")
        
        # Get system parameters from metadata
        if len(dataset.metadata_cache) == 0:
            logger.warning("No metadata cached - returning empty dataset info")
            return {}
        
        first_metadata = list(dataset.metadata_cache.values())[0]
        
        system_params = {
            'P_max': first_metadata['P_max'],
            'noise_var': first_metadata['noise_var'],
            'BW': first_metadata['BW'],
            'r_min': first_metadata.get('r_min', 0.5),
        }
        
        # Load r_min from network_info.json
        dataset_name = list(dataset.metadata_cache.keys())[0]
        network_info_path = osp.join(dataset.root, dataset_name, 'raw', 'network_info.json')
        
        if osp.exists(network_info_path):
            with open(network_info_path, 'r') as f:
                network_info = json.load(f)
            system_params['r_min'] = network_info['system_params'].get('r_min', system_params['r_min'])
        else:
            logger.warning(f"Network info not found: {network_info_path}")
            system_params['r_min'] = system_params.get('r_min', 0.5)  # Default value
        
        # Extract network seeds, associations, and H_ls from all cached graphs
        network_seeds = {}
        associations = {}
        h_ls_gains = {}
        channel_versions = {}
        h_timeslot_dirs = {}
        h_num_timesteps = {}
        channel_cache_info = {}
        channel_params = None
        channel_version_default = first_metadata.get('channel_version', 'v2')

        # Iterate through all samples to extract unique networks
        unique_networks = set()
        unique_dataset_names = set()
        for dataset_name, network_id, _ in dataset.samples:
            unique_networks.add((dataset_name, network_id))
            unique_dataset_names.add(dataset_name)

        # Load channel_cache_info and per-dataset r_min from network_info.json
        r_min_per_dataset = {}
        for ds_name in unique_dataset_names:
            ni_path = osp.join(dataset.root, ds_name, 'raw', 'network_info.json')
            if not osp.exists(ni_path):
                continue
            try:
                with open(ni_path, 'r') as f:
                    ni = json.load(f)
                cci = ni.get('channel_cache_info')
                if isinstance(cci, dict) and cci.get('channel_cache_file'):
                    channel_cache_info[ds_name] = cci
                # Extract per-dataset r_min
                ds_r_min = ni.get('system_params', {}).get('r_min')
                if ds_r_min is not None:
                    try:
                        r_min_per_dataset[ds_name] = float(ds_r_min)
                    except (TypeError, ValueError):
                        logger.warning(
                            "Non-numeric r_min=%r in network_info.json for dataset '%s'; skipping.",
                            ds_r_min, ds_name,
                        )
            except Exception as exc:
                logger.warning("Failed to read channel_cache_info from %s: %s", ni_path, exc)

        missing = unique_dataset_names - set(r_min_per_dataset.keys())
        if missing:
            logger.warning(
                "r_min not found in network_info.json for %d sub-dataset(s): %s. "
                "These will fall back to system_params['r_min']=%s. "
                "This may produce incorrect violation metrics.",
                len(missing), sorted(missing), system_params.get('r_min'),
            )
        
        logger.info(f"Loading metadata from {len(unique_networks)} unique networks...")
        
        for dataset_name, network_id in unique_networks:
            # Load graph.pt for this network
            cache_key = (dataset_name, network_id)
            network_dir = osp.join(dataset.root, dataset_name, 'processed', f'network_{network_id}')
            
            if cache_key in dataset.graph_cache:
                graph_data = dataset.graph_cache[cache_key]
            else:
                # Load from disk
                graph_path = osp.join(network_dir, 'graph.pt')
                
                if not osp.exists(graph_path):
                    logger.warning(f"Graph not found: {graph_path}")
                    continue
                
                graph_data = torch.load(graph_path)
            
            # Extract seed and associations using composite key
            composite_key = (dataset_name, network_id)
            network_seeds[composite_key] = graph_data.get('network_seed')
            associations[composite_key] = graph_data.get('associations')
            h_ls_gains[composite_key] = graph_data.get('H_ls')
            channel_versions[composite_key] = graph_data.get('channel_version', channel_version_default)
            if bool(graph_data.get('has_h_instantaneous', False)):
                h_timeslot_dirs[composite_key] = osp.join(network_dir, 'H_instantaneous')
                h_num_timesteps[composite_key] = int(graph_data.get('h_num_timesteps', 0))
            
            # Extract channel params (same for all networks in a dataset)
            if channel_params is None and 'channel_params' in graph_data:
                channel_params = graph_data['channel_params']
        
        logger.info(f"Extracted seeds and associations for {len(network_seeds)} networks")
        
        # Default channel params if not found
        if channel_params is None:
            logger.warning("Channel params not found in graph data - using defaults")
            channel_params = {
                'deployment_range': 1000.0,
                'path_loss_exponent_short': 2.0,
                'path_loss_exponent_long': 4.0,
                'shadowing_std': 7.0,
            }
        
        return {
            'system_params': system_params,
            'channel_params': channel_params,
            'channel_version': channel_version_default,
            'network_seeds': network_seeds,
            'associations': associations,
            'h_ls_gains': h_ls_gains,
            'channel_versions': channel_versions,
            'h_timeslot_dirs': h_timeslot_dirs,
            'h_num_timesteps': h_num_timesteps,
            'channel_cache_info': channel_cache_info,
            'r_min_per_dataset': r_min_per_dataset,
        }
    
    def get_dataset_info(self) -> Dict:
        """Return dataset metadata."""
        return self.dataset_info or {}
    
    def build_loaders(
        self,
        cfg: DatasetConfig,
        datasets: Dict[str, WRADataset],
        accelerator=None
    ) -> Dict[str, DataLoader]:
        """
        Build dataloaders for each dataset split.
        
        Uses regular random sampling for training (diversity) and NetworkGroupedBatchSampler
        for evaluation (efficiency - groups K samples from same network together).
        
        Parameters
        ----------
        cfg : DatasetConfig
            Configuration object
        datasets : dict
            Dictionary of datasets from build_datasets()
        accelerator : Accelerator, optional
            HuggingFace Accelerator for distributed training
        
        Returns
        -------
        loaders : dict
            Dictionary with keys 'train', 'val', 'test', 'train-val'
        """
        import math
        
        g = torch.Generator()
        g.manual_seed(0)
        
        # Get n_samples_per_input from config (K samples per network for evaluation)
        # This should match trainer.n_samples_per_input
        n_samples_per_input = cfg.get('n_samples_per_input', 10)
        
        # Calculate networks per batch for evaluation loaders
        # batch_size_val = networks_per_batch × samples_per_network
        networks_per_batch = math.ceil(cfg.batch_size_val / n_samples_per_input)
        eval_batch_size = networks_per_batch * n_samples_per_input

        # Optional context for logging only: how many raw samples exist per network on disk.
        # This is a pool-size statistic and is NOT the divisor used for eval grouping.
        raw_samples_per_network = None
        val_meta_cache = getattr(datasets.get("val"), "metadata_cache", None)
        if isinstance(val_meta_cache, dict) and len(val_meta_cache) > 0:
            first_meta = next(iter(val_meta_cache.values()))
            if isinstance(first_meta, dict):
                raw_samples_per_network = first_meta.get("samples_per_network", None)
        
        logger.info(f"Dataloader configuration:")
        logger.info(f"  Training: batch_size={cfg.batch_size}, regular sampling (diversity)")
        logger.info(f"  Evaluation: batch_size={eval_batch_size}, grouped sampling")
        logger.info(
            "    networks_per_batch = ceil(batch_size_val / n_samples_per_input) = "
            "ceil(%s / %s) = %s",
            cfg.batch_size_val,
            n_samples_per_input,
            networks_per_batch,
        )
        logger.info(f"    (K={n_samples_per_input} samples × {networks_per_batch} networks/batch)")
        if raw_samples_per_network is not None:
            logger.info(
                "    raw samples/network on disk=%s (pool size only; not used in the divisor)",
                raw_samples_per_network,
            )
        
        # Training loader with shuffling (regular sampling for diversity)
        train_loader = DataLoader(
            datasets["train"],
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
            drop_last=cfg.drop_last,
            generator=g,
            follow_batch=["y"],
            exclude_keys=["info"],
        )
        
        # Create stateful generators for evaluation loaders
        # These will produce different shuffle orders each epoch while remaining reproducible
        val_generator = torch.Generator()
        val_generator.manual_seed(43)
        
        train_val_generator = torch.Generator()
        train_val_generator.manual_seed(44)  # Different seed from val
        
        # Validation loader with NetworkGroupedBatchSampler
        val_sampler = NetworkGroupedBatchSampler(
            dataset=datasets["val"],
            batch_size=eval_batch_size,
            samples_per_network=n_samples_per_input,
            shuffle=True,
            generator=val_generator,  # Stateful: different shuffles each epoch
        )
        val_loader = DataLoader(
            datasets["val"],
            batch_sampler=val_sampler,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
            follow_batch=["y"],
            exclude_keys=["info"],
        )
        
        # Test loader with NetworkGroupedBatchSampler
        test_sampler = NetworkGroupedBatchSampler(
            dataset=datasets["test"],
            batch_size=eval_batch_size,
            samples_per_network=n_samples_per_input,
            shuffle=False,
            seed=42,
        )
        test_loader = DataLoader(
            datasets["test"],
            batch_sampler=test_sampler,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
            follow_batch=["y"],
            exclude_keys=["info"],
        )
        
        # Train-val loader with NetworkGroupedBatchSampler
        train_val_sampler = NetworkGroupedBatchSampler(
            dataset=datasets["train-val"],
            batch_size=eval_batch_size,
            samples_per_network=n_samples_per_input,
            shuffle=True,
            generator=train_val_generator,  # Stateful: different shuffles each epoch
        )
        train_val_loader = DataLoader(
            datasets["train-val"],
            batch_sampler=train_val_sampler,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
            follow_batch=["y"],
            exclude_keys=["info"],
        )
        
        # Prepare with accelerator if provided
        if accelerator:
            train_loader, val_loader, test_loader, train_val_loader = accelerator.prepare(
                train_loader, val_loader, test_loader, train_val_loader
            )
        
        logger.info("DataLoaders created successfully")
        
        return {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
            "train-val": train_val_loader,
        }
