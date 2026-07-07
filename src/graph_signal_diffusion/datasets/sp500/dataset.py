import os.path as osp
from typing import Callable, Optional

import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Dataset, Data
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
from torch_geometric.utils import coalesce
from tqdm import tqdm
from .utils import get_graph_in_pyg_format, get_column_idx, compute_periodic_dynamic_adjacencies
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

import logging
log = logging.getLogger(__name__)


def _spectral_normalize_edge_weights(
	edge_index: torch.Tensor,
	edge_weight: torch.Tensor,
	num_nodes: int,
	eps: float = 1e-12,
) -> torch.Tensor:
	"""Scale edge weights so weighted adjacency spectral radius is 1."""
	if edge_weight.numel() == 0:
		return edge_weight

	adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float64)
	src = edge_index[0].detach().cpu().numpy().astype(np.int64)
	dst = edge_index[1].detach().cpu().numpy().astype(np.int64)
	weights = edge_weight.detach().cpu().numpy().astype(np.float64)
	np.add.at(adjacency, (src, dst), weights)

	eigenvalues = np.linalg.eigvals(adjacency)
	spectral_radius = float(np.max(np.abs(eigenvalues))) if eigenvalues.size > 0 else 0.0
	if spectral_radius <= eps:
		return edge_weight

	return edge_weight / spectral_radius


def _compute_edge_stats(
	edge_index: torch.Tensor,
	edge_weight: torch.Tensor,
	num_nodes: int,
	include_self_loops: bool = False,
) -> dict:
	"""
	Compute simple edge statistics for logging.

	Returns directed edge density based on PyG edge list cardinality.
	"""
	num_edges = int(edge_index.shape[1]) if edge_index.ndim == 2 else 0
	possible_edges = int(num_nodes * num_nodes) if include_self_loops else int(num_nodes * max(num_nodes - 1, 0))
	edge_density = float(num_edges / possible_edges) if possible_edges > 0 else 0.0
	avg_edge_weight = float(edge_weight.mean().item()) if edge_weight.numel() > 0 else 0.0
	return {
		"num_edges": num_edges,
		"edge_density": edge_density,
		"avg_edge_weight": avg_edge_weight,
	}


class StocksDataDiffusion(Data):
	def __init__(
		self,
		x=None,
		edge_index=None,
		edge_weight=None,
		close_price=None,
		y=None,
		close_price_y=None,
		timestamp=None,
		stocks_index=None,
		target_column_idx=None,
		network_id=None,
		dataset_name=None,
		info=None,
	):
		super().__init__(
			x=x,
			edge_index=edge_index,
			edge_weight=edge_weight,
			close_price=close_price,
			y=y,
			close_price_y=close_price_y,
			timestamp=timestamp,
			stocks_index=stocks_index,
			target_column_idx=target_column_idx,
		)
		self.network_id = network_id
		self.dataset_name = dataset_name
		if info is not None:
			self.info = info
			# for key in info:
			# 	setattr(self, key, info[key])
			# self.info = info

	def __inc__(self, key, value, *args, **kwargs):
		if key in ["timestamp", "stocks_index", "target_column_idx", "network_id", "dataset_name"]:
			return 0
		
		if key == "info":
			return 0
		# Don't increment any info-related attributes
		# if key in ["Target", "Features", "Num_nodes"]:
		# 	return 0
		return super().__inc__(key, value, *args, **kwargs)


	def __cat_dim__(self, key, value, *args, **kwargs):
		# Default behavior for specific keys
		# if key in ['adj']:
		# 	return 1 # concatenate along the node dimension

		if key in ["timestamp", "stocks_index", "target_column_idx", "network_id", "dataset_name"]:
			return 0  # Batch timestamps and stocks_index
		
		# Don't batch info dictionary or any of its keys
		if key == "info":
			return None
		
		if key in ["Features", "Target"]:
			return 0

		# if key in self.info:
		# 	return None  # Don't batch - keeps only first occurrence
		
		return super().__cat_dim__(key, value, *args, **kwargs)

	
import torch.serialization
# At the top of your module, after imports
# Register safe globals for torch.load with weights_only=True (PyTorch 2.0+)
try:
    torch.serialization.add_safe_globals([StocksDataDiffusion, Data, DataEdgeAttr, DataTensorAttr, GlobalStorage])
except AttributeError:
    # Older PyTorch versions don't have add_safe_globals
    pass

from graph_signal_diffusion.datasets.normalizer import Normalizer # add normalization support for the dataset



class SP500Stocks(Dataset):
	"""
	Stock price data for the S&P 500 companies.
	"""

	def __init__(self, root: str = "../data/SP500/", values_file_name: str = "values.csv", adj_file_name: str = "adj.npy", past_window: int = 25, future_window: int = 1,
			  target_column_name: str = "DailyLogReturn",
			  pool_ratio: float = 0.5,
			  force_reload: bool = False, transform: Callable = None,
			  per_stock_stats: dict = None,
			  feature_names: list = None,
			  standardized_features: list = None,
			  log_transform_features: list = None,
		  sp500_scale_factor: float = 100.0,
		  top_k: int = None,
		  min_degree: int = None,
		  edge_weight_norm: str = "none",
		  spectral_normalize_edge_weights: bool = False,
		  dynamic_graph_period: int = None,
		  dynamic_graph_top_k: int = None,
		  dynamic_graph_min_degree: int = None,
		  dynamic_graph_method: str = "pearson",
		  dynamic_graph_edge_weight_threshold: Optional[float] = None,
		  dynamic_graph_edge_budget_ratio: Optional[float] = None,
		  standardize_target_in_x_for_revin: bool = False):
		self.values_file_name = values_file_name
		self.adj_file_name = adj_file_name
		self.past_window = past_window
		self.future_window = future_window
		self.target_column_name = target_column_name
		self.pool_ratio = pool_ratio
		self.top_k = top_k
		self.min_degree = min_degree
		self.edge_weight_norm = edge_weight_norm
		self.spectral_normalize_edge_weights = spectral_normalize_edge_weights
		
		# Validate edge_weight_norm
		_VALID_EDGE_NORMS = {"none", "l1_col", "l2_col", "symmetric"}
		if edge_weight_norm not in _VALID_EDGE_NORMS:
			raise ValueError(
				f"Invalid edge_weight_norm='{edge_weight_norm}'. "
				f"Choose from: {sorted(_VALID_EDGE_NORMS)}. "
				f"Consider using spectral_normalize_edge_weights=true instead."
			)
		
		# Dynamic graph parameters
		self.dynamic_graph_period = dynamic_graph_period
		self.dynamic_graph_top_k = dynamic_graph_top_k
		self.dynamic_graph_min_degree = dynamic_graph_min_degree
		self.dynamic_graph_method = dynamic_graph_method
		self.dynamic_graph_edge_weight_threshold = dynamic_graph_edge_weight_threshold
		self.dynamic_graph_edge_budget_ratio = dynamic_graph_edge_budget_ratio

		if self.dynamic_graph_edge_budget_ratio is not None and self.dynamic_graph_edge_budget_ratio <= 0:
			raise ValueError(
				f"dynamic_graph_edge_budget_ratio must be > 0 or None, "
				f"got {self.dynamic_graph_edge_budget_ratio}"
			)

		if self.dynamic_graph_edge_weight_threshold is not None and self.dynamic_graph_edge_weight_threshold < 0:
			raise ValueError(
				f"dynamic_graph_edge_weight_threshold must be >= 0 or None, "
				f"got {self.dynamic_graph_edge_weight_threshold}"
			)
		
		# Validate: degree-dependent norms are incompatible with dynamic graph concatenation
		_DEGREE_DEPENDENT_NORMS = {"l1_col", "l2_col", "symmetric"}
		if dynamic_graph_period is not None and edge_weight_norm in _DEGREE_DEPENDENT_NORMS:
			raise ValueError(
				f"edge_weight_norm='{edge_weight_norm}' is a degree-dependent normalization "
				f"that is incompatible with dynamic_graph_period={dynamic_graph_period}. "
				f"When dynamic graphs are enabled, static and dynamic edges are concatenated "
				f"at retrieval time, which invalidates degree-dependent normalization invariants. "
				f"Use edge_weight_norm='none' with spectral_normalize_edge_weights=true instead."
			)
		
		# NEW: Normalization support 
		self.normalizer = None
		self._normalize_targets = True  # Whether to normalize target values
		
		# NEW: SP500 per-stock standardization support
		self.per_stock_stats = per_stock_stats
		
		# Feature names MUST be provided when using per-stock standardization
		if per_stock_stats is not None and feature_names is None:
			raise ValueError(
				"feature_names must be provided when per_stock_stats is specified. "
				"This ensures correct mapping between feature indices and statistics."
			)
		
		self.feature_names = feature_names
		self.standardized_features = standardized_features or []
		self.log_transform_features = log_transform_features or []
		self.sp500_scale_factor = sp500_scale_factor
		self._apply_sp500_standardization = per_stock_stats is not None
		self.standardize_target_in_x_for_revin = standardize_target_in_x_for_revin

		# Store target-specific stats for easy access during evaluation
		self.target_stats = None
		if per_stock_stats is not None and target_column_name in per_stock_stats:
			self.target_stats = {
				'mean': torch.from_numpy(per_stock_stats[target_column_name]['mean']).float(),
				'std': torch.from_numpy(per_stock_stats[target_column_name]['std']).float(),
				'scale_factor': sp500_scale_factor
			}
			log.info(f"Target stats stored: mean shape={self.target_stats['mean'].shape}, "
			        f"std shape={self.target_stats['std'].shape}")

		if self._apply_sp500_standardization:
			log.info(f"SP500Stocks: Will apply feature standardization during preprocessing")
			log.info(f"Feature order: {self.feature_names}")
			log.info(f"Standardizing features: {self.standardized_features}")
			log.info(f"Log-transforming: {self.log_transform_features}")
			log.info(f"Target standardization applied at runtime in get()")

		super().__init__(root, force_reload=force_reload, transform=transform)
	
	@property
	def raw_file_names(self) -> list[str]:
		"""Required by PyTorch Geometric Dataset."""
		return [self.values_file_name, self.adj_file_name]
	
	def _get_processing_config(self) -> dict:
		"""Get parameters that affect processed data contents."""
		config = {
			'past_window': self.past_window,
			'future_window': self.future_window,
			'target_column_name': self.target_column_name,
			'pool_ratio': self.pool_ratio,
			'top_k': self.top_k,
			'min_degree': self.min_degree,
			'edge_weight_norm': self.edge_weight_norm,
			'spectral_normalize_edge_weights': self.spectral_normalize_edge_weights,
			'values_file_name': self.values_file_name,
			'adj_file_name': self.adj_file_name,
			'apply_standardization': self._apply_sp500_standardization,
			'standardized_features': tuple(sorted(self.standardized_features)) if self.standardized_features else None,
			'log_transform_features': tuple(sorted(self.log_transform_features)) if self.log_transform_features else None,
			'sp500_scale_factor': self.sp500_scale_factor,
		}
		# Only include dynamic graph params when enabled, so that
		# the hash remains identical to pre-dynamic-graph code when disabled.
		if self.dynamic_graph_period is not None:
			config['dynamic_graph_period'] = self.dynamic_graph_period
			config['dynamic_graph_top_k'] = self.dynamic_graph_top_k
			config['dynamic_graph_min_degree'] = self.dynamic_graph_min_degree
			config['dynamic_graph_method'] = self.dynamic_graph_method
			config['dynamic_graph_edge_weight_threshold'] = self.dynamic_graph_edge_weight_threshold
			if self.dynamic_graph_edge_budget_ratio is not None:
				config['dynamic_graph_edge_budget_ratio'] = self.dynamic_graph_edge_budget_ratio
		return config
	
	def _get_cache_suffix(self) -> str:
		"""Generate cache directory suffix from processing parameters."""
		import hashlib
		import json
		
		config = self._get_processing_config()
		# Sort dict for consistent hashing
		config_str = json.dumps(config, sort_keys=True)
		hash_val = hashlib.md5(config_str.encode()).hexdigest()[:8]
		return f"_{hash_val}"
	
	@property
	def processed_dir(self) -> str:
		"""Override to add parameter-specific suffix for cache isolation."""
		suffix = self._get_cache_suffix()
		processed_path = osp.join(self.root, f'processed{suffix}')
		return processed_path
	
	@property
	def processed_file_names(self) -> list[str]:
		files = ['graph_static.pt'] + [
			f'timestep_{idx}.pt' for idx in range(len(self))
		]
		if self.dynamic_graph_period is not None:
			# Include dynamic graph metadata so PyG re-runs process() if missing
			files.append('dynamic_graph_meta.json')
		return files

	def download(self) -> None:
		pass

	def process(self) -> None:
		"""Process raw data and save to disk with parameter-specific caching."""
		import json
		
		# Save processing config for transparency and validation
		config = self._get_processing_config()
		config_path = osp.join(self.processed_dir, 'processing_config.json')
		with open(config_path, 'w') as f:
			json.dump(config, f, indent=2, sort_keys=True)
		log.info(f"Saved processing config to {config_path}")
		log.info(f"Processing parameters hash: {self._get_cache_suffix()}")
		
		x, close_prices, edge_index, edge_weight, info_dict = get_graph_in_pyg_format(
			values_path=osp.join(self.root, f"raw/{self.values_file_name}"),
			adj_path=osp.join(self.root, f"raw/{self.adj_file_name}"),
			target_column_name=self.target_column_name,
			pool_ratio=self.pool_ratio,
			top_k=self.top_k,
			min_degree=self.min_degree,
			edge_weight_norm=self.edge_weight_norm
		)

		if self.spectral_normalize_edge_weights:
			edge_weight = _spectral_normalize_edge_weights(
				edge_index=edge_index,
				edge_weight=edge_weight,
				num_nodes=x.shape[0],
			)
			log.info("Applied spectral normalization to SP500 edge weights (spectral radius -> 1)")

		info_dict["spectral_normalize_edge_weights"] = bool(self.spectral_normalize_edge_weights)
		static_edge_stats = _compute_edge_stats(
			edge_index=edge_index,
			edge_weight=edge_weight,
			num_nodes=x.shape[0],
			include_self_loops=False,
		)
		log.info(
			"Static graph stats (final): avg_edge_weight=%.6f, edge_density=%.6f, num_edges=%d",
			static_edge_stats["avg_edge_weight"],
			static_edge_stats["edge_density"],
			static_edge_stats["num_edges"],
		)

		target_column_idx = get_column_idx(values_path_or_df=osp.join(self.root, f"raw/{self.values_file_name}"),
											column_name=info_dict["Target"]
											) - 1 # subtract one because we dropped the first column from values
		log.info(f"Target column name / index: {info_dict['Target']} / {target_column_idx}")

		# Save target column index for later use
		info_dict["Target_column_idx"] = target_column_idx
		self.info = info_dict
		log.info(f"Info dictionary keys: {list(info_dict.keys())}")
		
		# Apply feature standardization during preprocessing (EXCLUDING target)
		# x shape: [N_stocks, N_features, T_timesteps]
		if self._apply_sp500_standardization:
			log.info(f"\nApplying feature standardization during preprocessing...")
			x = self._preprocess_features(x, feature_names=self.feature_names, target_column_idx=target_column_idx)
			log.info(f"Feature standardization complete (target excluded)")
		
		# Create timesteps with static graph structure
		timestamps = [
			StocksDataDiffusion(
				x=x[:, :, idx:idx + self.past_window].clone(),
				edge_index=edge_index,
				edge_weight=edge_weight,
				close_price=close_prices[:, idx:idx + self.past_window].clone(),
				y=x[:, target_column_idx, idx + self.past_window:idx + self.past_window + self.future_window].clone(),
				close_price_y=close_prices[:, idx + self.past_window:idx + self.past_window + self.future_window].clone(),
				stocks_index=torch.arange(x.shape[0]),
				timestamp=torch.LongTensor([idx]).repeat(x.shape[0]),
				info=info_dict
			)
			for idx in range(x.shape[2] - self.past_window - self.future_window)
		]
		
		# Save static graph data once (edge_index, edge_weight, info)
		static_graph_data = {
			'edge_index': edge_index,
			'edge_weight': edge_weight,
			'info': info_dict,
			'stocks_index': torch.arange(x.shape[0])  # Save once since it's the same for all timesteps
		}
		torch.save(static_graph_data, osp.join(self.processed_dir, 'graph_static.pt'))
		log.info(f"Saved static graph data: {edge_index.shape[1]} edges, {x.shape[0]} nodes")
		
		# Compute and save periodic dynamic graphs if enabled
		if self.dynamic_graph_period is not None:
			log.info(f"\n🔄 Computing periodic dynamic graphs (period={self.dynamic_graph_period})...")
			dyn_top_k = self.dynamic_graph_top_k if self.dynamic_graph_top_k is not None else self.top_k
			dyn_min_deg = self.dynamic_graph_min_degree if self.dynamic_graph_min_degree is not None else self.min_degree
			dyn_threshold = (
				float(self.dynamic_graph_edge_weight_threshold)
				if self.dynamic_graph_edge_weight_threshold is not None
				else 0.0
			)
			
			# Compute adaptive edge budget from static graph size
			static_num_edges = int(edge_index.shape[1])
			dyn_max_edges = None
			if self.dynamic_graph_edge_budget_ratio is not None:
				dyn_max_edges = int(self.dynamic_graph_edge_budget_ratio * static_num_edges)
				log.info(
					f"Edge budget: ratio={self.dynamic_graph_edge_budget_ratio}, "
					f"static_edges={static_num_edges}, max_dynamic_edges={dyn_max_edges}"
				)
			
			log.info(
				f"Dynamic graph settings: threshold={self.dynamic_graph_edge_weight_threshold} "
				f"(effective={dyn_threshold}), top_k={dyn_top_k}, min_degree={dyn_min_deg}, "
				f"max_edges={dyn_max_edges}"
			)
			
			dynamic_graphs = compute_periodic_dynamic_adjacencies(
				values_path=osp.join(self.root, f"raw/{self.values_file_name}"),
				period=self.dynamic_graph_period,
				target_column_name=self.target_column_name,
				correlation_threshold=dyn_threshold,
				method=self.dynamic_graph_method,
				top_k=dyn_top_k,
				min_degree=dyn_min_deg,
				max_edges=dyn_max_edges,
			)
			
			# Apply spectral normalization to each dynamic graph independently
			if self.spectral_normalize_edge_weights:
				for dg in dynamic_graphs:
					dg['edge_weight'] = _spectral_normalize_edge_weights(
						edge_index=dg['edge_index'],
						edge_weight=dg['edge_weight'],
						num_nodes=x.shape[0],
					)
				log.info("Applied spectral normalization to all dynamic graphs")

			dynamic_edge_stats = [
				_compute_edge_stats(
					edge_index=dg['edge_index'],
					edge_weight=dg['edge_weight'],
					num_nodes=x.shape[0],
					include_self_loops=False,
				)
				for dg in dynamic_graphs
			]
			if dynamic_edge_stats:
				dynamic_avg_weights = np.array([s["avg_edge_weight"] for s in dynamic_edge_stats], dtype=np.float64)
				dynamic_densities = np.array([s["edge_density"] for s in dynamic_edge_stats], dtype=np.float64)
				dynamic_num_edges = np.array([s["num_edges"] for s in dynamic_edge_stats], dtype=np.int64)
				log.info(
					"Dynamic graph stats (post-thresholding/final across %d periods): "
					"avg_edge_weight(mean/min/max)=%.6f/%.6f/%.6f, "
					"edge_density(mean/min/max)=%.6f/%.6f/%.6f, "
					"num_edges(mean/min/max)=%.1f/%d/%d",
					len(dynamic_edge_stats),
					float(dynamic_avg_weights.mean()),
					float(dynamic_avg_weights.min()),
					float(dynamic_avg_weights.max()),
					float(dynamic_densities.mean()),
					float(dynamic_densities.min()),
					float(dynamic_densities.max()),
					float(dynamic_num_edges.mean()),
					int(dynamic_num_edges.min()),
					int(dynamic_num_edges.max()),
				)
			
			# Save each dynamic graph as a separate .pt file
			for k, dg in enumerate(dynamic_graphs):
				torch.save(dg, osp.join(self.processed_dir, f'dynamic_graph_{k}.pt'))
			
			# Save metadata: mapping from timestep index → dynamic graph file index
			import json
			dynamic_meta = {
				'period': self.dynamic_graph_period,
				'num_periods': len(dynamic_graphs),
				'method': self.dynamic_graph_method,
				'edge_weight_threshold': self.dynamic_graph_edge_weight_threshold,
				'effective_edge_weight_threshold': dyn_threshold,
				'top_k': dyn_top_k,
				'min_degree': dyn_min_deg,
				'edge_budget_ratio': self.dynamic_graph_edge_budget_ratio,
				'edge_budget_max_edges': dyn_max_edges,
				'static_num_edges': static_num_edges,
				'periods': [
					{
						'file_index': k,
						'start_timestep': dg['start_timestep'],
						'end_timestep': dg['end_timestep'],
						'num_edges': int(dg['edge_index'].shape[1]),
						'adapted_threshold': dg.get('adapted_threshold', dyn_threshold),
					}
					for k, dg in enumerate(dynamic_graphs)
				],
			}
			with open(osp.join(self.processed_dir, 'dynamic_graph_meta.json'), 'w') as f:
				json.dump(dynamic_meta, f, indent=2)
			log.info(f"Saved {len(dynamic_graphs)} dynamic graph files + metadata")
			
			# Visualize dynamic graphs
			self._visualize_dynamic_graphs(dynamic_graphs, num_nodes=x.shape[0])
		
		# Visualize the static PyG graph
		self._visualize_graph(edge_index, edge_weight, x.shape[0])
		
		# Save only timestep-specific data (x, y, close_price, timestamp)
		for t, timestep in tqdm(enumerate(timestamps), desc="Processing timesteps", total=len(timestamps)):
			timestep_dict = {
				'x': timestep.x,
				'y': timestep.y,
				'close_price': timestep.close_price,
				'close_price_y': timestep.close_price_y,
				'timestamp': timestep.timestamp,
			}
			torch.save(timestep_dict, osp.join(self.processed_dir, f"timestep_{t}.pt"))
			

	def len(self) -> int:
		values = pd.read_csv(self.raw_paths[0]).set_index(['Symbol', 'Date'])
		return len(values.loc[values.index[0][0]]) - self.past_window - self.future_window
	

	def _visualize_graph(self, edge_index: torch.Tensor, edge_weight: torch.Tensor, num_nodes: int) -> None:
		"""
		Visualize the PyG correlation graph with circular sector-grouped layout.
		
		Matches WRA visualization style:
		- Nodes colored by sector
		- Edges colored/sized by correlation strength
		- Circular layout with same-sector stocks adjacent
		"""
		try:
			# Load sector layout info (shared by static and dynamic visualizations)
			layout = self._get_sector_layout(num_nodes)
			if layout is None:
				return
			
			# Build filename with sparsification parameters and edge normalization
			filename_parts = ['graph_visualization']
			if self.top_k is not None:
				filename_parts.append(f'top_k_{self.top_k}')
			if self.min_degree is not None:
				filename_parts.append(f'min_degree_{self.min_degree}')
			if self.edge_weight_norm != 'none':
				filename_parts.append(f'norm_{self.edge_weight_norm}')
			if self.spectral_normalize_edge_weights:
				filename_parts.append('spectral_norm')
			filename = '_'.join(filename_parts) + '.pdf'
			
			# Summary text
			if self.top_k or self.min_degree:
				sparsification_info = f'Sparsification: top_k={self.top_k}, min_degree={self.min_degree}'
			else:
				sparsification_info = 'No sparsification applied'
			norm_info = f'Edge norm: {self.edge_weight_norm}, spectral_norm: {self.spectral_normalize_edge_weights}'
			subtitle = f'{len(layout["unique_sectors"])} sectors | Circular layout with sector grouping | {sparsification_info} | {norm_info}'
			
			title = f'SP500 Correlation Graph: {num_nodes} stocks, {edge_index.shape[1]} edges'
			
			self._plot_graph_circular(
				edge_index=edge_index,
				edge_weight=edge_weight,
				layout=layout,
				title=title,
				subtitle=subtitle,
				filename=filename,
			)
			
		except Exception as e:
			log.info(f"WARNING: Failed to create graph visualization: {e}")
			import traceback
			traceback.print_exc()

	def _visualize_dynamic_graphs(self, dynamic_graphs: list, num_nodes: int) -> None:
		"""Visualize at most 4 periodic dynamic graphs evenly sampled across timeline."""
		try:
			layout = self._get_sector_layout(num_nodes)
			if layout is None:
				return
			
			dyn_top_k = self.dynamic_graph_top_k if self.dynamic_graph_top_k is not None else self.top_k
			dyn_min_deg = self.dynamic_graph_min_degree if self.dynamic_graph_min_degree is not None else self.min_degree
			
			# Sample at most 4 dynamic graphs evenly across the timeline
			max_visualizations = 4
			n_periods = len(dynamic_graphs)
			
			if n_periods <= max_visualizations:
				# Visualize all if we have 4 or fewer
				indices_to_visualize = list(range(n_periods))
			else:
				# Sample evenly: first, ~1/3, ~2/3, last
				indices_to_visualize = [
					0,
					n_periods // 3,
					2 * n_periods // 3,
					n_periods - 1
				]
			
			log.info(f"Visualizing {len(indices_to_visualize)} of {n_periods} dynamic graphs: periods {indices_to_visualize}")
			
			for k in indices_to_visualize:
				dg = dynamic_graphs[k]
				start_t = dg['start_timestep']
				end_t = dg['end_timestep']
				n_edges = dg['edge_index'].shape[1]
				
				filename = f'dynamic_graph_period_{k}_t{start_t}-{end_t}.pdf'
				title = f'Dynamic Graph Period {k}: timesteps [{start_t}, {end_t}), {n_edges} edges'
				
				parts = [f'Period {k} of {n_periods}']
				if dyn_top_k is not None:
					parts.append(f'top_k={dyn_top_k}')
				if dyn_min_deg is not None:
					parts.append(f'min_degree={dyn_min_deg}')
				parts.append(f'spectral_norm={self.spectral_normalize_edge_weights}')
				subtitle = ' | '.join(parts)
				
				self._plot_graph_circular(
					edge_index=dg['edge_index'],
					edge_weight=dg['edge_weight'],
					layout=layout,
					title=title,
					subtitle=subtitle,
					filename=filename,
				)
			
			log.info(f"Saved {len(indices_to_visualize)} dynamic graph visualizations (sampled from {n_periods} total periods)")
			
		except Exception as e:
			log.info(f"WARNING: Failed to create dynamic graph visualizations: {e}")
			import traceback
			traceback.print_exc()

	def _get_sector_layout(self, num_nodes: int) -> dict:
		"""
		Load sector info and compute circular layout positions.
		
		Returns dict with 'node_pos', 'sector_groups', 'unique_sectors',
		'sector_color_map', or None if stocks.csv is not found.
		"""
		stocks_path = osp.join(self.root, 'raw', 'stocks.csv')
		if not osp.exists(stocks_path):
			log.info("WARNING: stocks.csv not found, skipping graph visualization")
			return None
		
		stocks_df = pd.read_csv(stocks_path)
		
		values_path = osp.join(self.root, f"raw/{self.values_file_name}")
		values = pd.read_csv(values_path).set_index(['Symbol', 'Date'])
		symbols = values.index.get_level_values('Symbol').unique().tolist()
		
		sector_map = dict(zip(stocks_df['Symbol'], stocks_df['Sector']))
		sectors = [sector_map.get(sym, 'Unknown') for sym in symbols]
		unique_sectors = sorted(set(sectors))
		
		# Group stocks by sector
		sector_groups = {}
		for idx, sector in enumerate(sectors):
			if sector not in sector_groups:
				sector_groups[sector] = []
			sector_groups[sector].append(idx)
		
		# Compute circular positions
		node_pos = np.zeros((num_nodes, 2))
		angle_offset = 0
		for sector in unique_sectors:
			group = sector_groups[sector]
			group_size = len(group)
			angle_span = 2 * np.pi * (group_size / num_nodes)
			for i, node_idx in enumerate(group):
				angle = angle_offset + (i / group_size) * angle_span
				node_pos[node_idx] = [np.cos(angle), np.sin(angle)]
			angle_offset += angle_span
		
		# Sector colors
		sector_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_sectors)))
		sector_color_map = {sector: sector_colors[i] for i, sector in enumerate(unique_sectors)}
		
		return {
			'node_pos': node_pos,
			'sector_groups': sector_groups,
			'unique_sectors': unique_sectors,
			'sector_color_map': sector_color_map,
			'num_nodes': num_nodes,
		}

	def _plot_graph_circular(
		self,
		edge_index: torch.Tensor,
		edge_weight: torch.Tensor,
		layout: dict,
		title: str,
		subtitle: str,
		filename: str,
	) -> None:
		"""
		Core graph visualization with circular sector-grouped layout.
		
		Used by both static and dynamic graph visualization methods.
		"""
		node_pos = layout['node_pos']
		sector_groups = layout['sector_groups']
		unique_sectors = layout['unique_sectors']
		sector_color_map = layout['sector_color_map']
		num_nodes = layout['num_nodes']
		
		fig, ax = plt.subplots(1, 1, figsize=(16, 16))
		
		edge_index_np = edge_index.numpy()
		edge_weight_np = edge_weight.numpy()
		edge_weight_original = edge_weight_np.copy()
		
		# Normalize edge weights for visualization alpha/linewidth
		if len(edge_weight_np) > 0:
			edge_weights_norm = (np.abs(edge_weight_np) - np.abs(edge_weight_np).min()) / \
			                     (np.abs(edge_weight_np).max() - np.abs(edge_weight_np).min() + 1e-10)
		else:
			edge_weights_norm = edge_weight_np
		
		# Draw edges
		for i in range(edge_index_np.shape[1]):
			src, tgt = edge_index_np[0, i], edge_index_np[1, i]
			if src == tgt:
				continue
			weight = edge_weights_norm[i]
			color = plt.cm.Reds(0.3 + 0.7 * weight)
			alpha = 0.2 + 0.6 * weight
			linewidth = 0.3 + 1.2 * weight
			ax.plot(
				[node_pos[src, 0], node_pos[tgt, 0]],
				[node_pos[src, 1], node_pos[tgt, 1]],
				color=color, alpha=alpha, linewidth=linewidth, zorder=1
			)
		
		# Draw nodes colored by sector
		for sector in unique_sectors:
			group = sector_groups[sector]
			pos = node_pos[group]
			ax.scatter(
				pos[:, 0], pos[:, 1],
				c=[sector_color_map[sector]] * len(group),
				s=100, marker='o',
				edgecolors='black', linewidths=1.5,
				label=sector, zorder=3
			)
		
		# Add sector dividers
		angle_offset = 0
		for sector in unique_sectors:
			group_size = len(sector_groups[sector])
			angle_span = 2 * np.pi * (group_size / num_nodes)
			angle = angle_offset
			ax.plot(
				[0, 1.15 * np.cos(angle)], [0, 1.15 * np.sin(angle)],
				color='gray', linestyle=':', linewidth=1, alpha=0.3, zorder=0
			)
			angle_offset += angle_span
		
		ax.set_xlabel('', fontsize=12)
		ax.set_ylabel('', fontsize=12)
		ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
		
		ax.legend(
			loc='lower left', fontsize=7, ncol=2,
			frameon=True, fancybox=True, shadow=True, framealpha=0.9
		)
		
		ax.set_xticks([])
		ax.set_yticks([])
		for spine in ax.spines.values():
			spine.set_visible(False)
		
		ax.set_aspect('equal')
		ax.set_xlim(-1.3, 1.3)
		ax.set_ylim(-1.3, 1.3)
		
		# Add colorbar in upper right corner
		if len(edge_weight_original) > 0:
			from matplotlib.cm import ScalarMappable
			from matplotlib.colors import Normalize as MplNormalize
			
			# Extend colorbar to 0 to show full correlation scale and visualize thresholding
			# (edges with correlation < threshold are absent from graph)
			norm = MplNormalize(vmin=0.0, vmax=edge_weight_original.max())
			sm = ScalarMappable(cmap=plt.cm.Reds, norm=norm)
			sm.set_array([])
			
			# Place colorbar using fig.add_axes with explicit position [left, bottom, width, height]
			# in figure-fraction coordinates — upper right corner of the plot area
			cbar_ax = fig.add_axes([0.82, 0.68, 0.02, 0.18])
			cbar = fig.colorbar(sm, cax=cbar_ax, orientation='vertical')
			cbar.set_label('Edge Weight', rotation=270, labelpad=15, fontsize=10)
			cbar.ax.tick_params(labelsize=8)
		
		# Subtitle
		fig.text(0.5, 0.02, subtitle, ha='center', fontsize=11, style='italic')
		
		# Save
		plots_dir = osp.join(self.root, 'plots')
		import os
		os.makedirs(plots_dir, exist_ok=True)
		figure_path = osp.join(plots_dir, filename)
		plt.savefig(figure_path, dpi=300, bbox_inches='tight')
		plt.close()
		log.info(f"Saved graph visualization: {figure_path}")

	def set_normalizer(self, normalizer: Normalizer) -> None:
		"""Set the normalizer for the dataset."""
		self.normalizer = normalizer
		log.info(f"Normalizer set for SP500Stocks dataset")
	
	def get_target_standardization_stats(self) -> dict:
		"""
		Get target standardization stats for evaluation-time destandardization.
		
		Returns:
			dict with keys:
				- 'mean': [N,] per-stock means
				- 'std': [N,] per-stock stds
				- 'scale_factor': float (e.g., 100.0 for return features)
				- 'target_name': str (e.g., 'DailyLogReturn')
		"""
		if self.target_stats is None:
			raise ValueError(
				"Target stats not available. Dataset must be initialized with per_stock_stats."
			)
		
		return {
			'mean': self.target_stats['mean'],
			'std': self.target_stats['std'],
			'scale_factor': self.target_stats['scale_factor'],
			'target_name': self.target_column_name
		}
	
	@staticmethod
	def destandardize_target(standardized: torch.Tensor, stats: dict) -> torch.Tensor:
		"""
		Destandardize target predictions/samples for evaluation.
		
		Usage:
			stats = dataset.get_target_standardization_stats()
			raw_predictions = SP500Stocks.destandardize_target(standardized_predictions, stats)
		
		Args:
			standardized: Standardized tensor of shape [N, ...] or [batch, N, ...]
			stats: Dict from get_target_standardization_stats() with keys:
				   'mean' [N,], 'std' [N,], 'scale_factor'
		
		Returns:
			Destandardized tensor in original scale
		"""
		mean = stats['mean'].to(standardized.device)
		std = stats['std'].to(standardized.device)
		scale_factor = stats['scale_factor']
		
		# Reshape for broadcasting based on input shape
		# Handle both [N, ...] and [batch, N, ...] cases
		if standardized.dim() == 2:  # [N, T]
			mean = mean.view(-1, 1)
			std = std.view(-1, 1)
		elif standardized.dim() == 3:  # [N, T, F] or [batch, N, T]
			if standardized.shape[-1] == 1:  # [N, T, 1] or [batch, N, 1]
				mean = mean.view(-1, 1, 1)
				std = std.view(-1, 1, 1)
			else:  # [batch, N, T]
				mean = mean.view(1, -1, 1)
				std = std.view(1, -1, 1)
		elif standardized.dim() == 4:  # [batch, N, T, F]
			mean = mean.view(1, -1, 1, 1)
			std = std.view(1, -1, 1, 1)
		
		# Destandardize: x = z * std + mean, then multiply by scale_factor
		destandardized = (standardized * (std + 1e-8) + mean) * scale_factor
		
		return destandardized


	def get(self, idx: int) -> Data:
		# Load static graph data (cached after first load)
		if not hasattr(self, '_static_graph_cache'):
			self._static_graph_cache = torch.load(
				osp.join(self.processed_dir, 'graph_static.pt'),
				weights_only=True
			)
		
		# Load dynamic graph metadata (cached after first load)
		if not hasattr(self, '_dynamic_graph_meta'):
			meta_path = osp.join(self.processed_dir, 'dynamic_graph_meta.json')
			if osp.exists(meta_path):
				import json
				with open(meta_path, 'r') as f:
					self._dynamic_graph_meta = json.load(f)
			else:
				self._dynamic_graph_meta = None  # No dynamic graphs
		
		# Load timestep-specific data
		timestep_dict = torch.load(
			osp.join(self.processed_dir, f'timestep_{idx}.pt'),
			weights_only=True
		)
		
		# Get fused (static + dynamic) graph, coalesced and cached per period
		edge_index, edge_weight = self._get_fused_graph(idx)
		
		# Combine static and timestep data
		save_dict = {
			**timestep_dict,
			'edge_index': edge_index,
			'edge_weight': edge_weight,
			'stocks_index': self._static_graph_cache['stocks_index'],
			'network_id': 0,
			'dataset_name': 'sp500',
			'info': self._static_graph_cache['info'],
		}
		
		# Store target column index as a top-level scalar so it survives PyG batching
		# (info dict is excluded from dataloaders via exclude_keys)
		save_dict['target_column_idx'] = self._static_graph_cache['info'].get(
			'Target_column_idx', None
		)

		# Permute x from [N, F, T] to [N, T, F]
		save_dict['x'] = save_dict['x'].permute(0, 2, 1)  # E.g., [N, 8, 20] -> [N, 20, 8]
		
		# Add feature dimension to y: [N, T] -> [N, T, 1]
		save_dict['y'] = save_dict['y'].unsqueeze(-1)  # [N, 5] -> [N, 5, 1]

		# NEW: Apply target standardization at runtime (features already standardized during preprocessing)
		if self._apply_sp500_standardization:
			# Get target feature index (DailyLogReturn)
			target_feat_idx = self.feature_names.index(self.target_column_name)

			# Standardize target y (future log returns)
			# Note: Features in x were already standardized during process()
			save_dict['y'] = self._standardize_target(save_dict['y'], feature_idx=target_feat_idx)

			# Optionally standardize target channel in x (conditioning) for RevIN space consistency.
			# When enabled, x[..., target_idx] is transformed to the same standardized space as y,
			# so RevIN stats computed from conditioning are in the correct space.
			if self.standardize_target_in_x_for_revin:
				target_slice = save_dict['x'][:, :, target_feat_idx:target_feat_idx+1]  # [N, T_obs, 1]
				save_dict['x'][:, :, target_feat_idx:target_feat_idx+1] = \
					self._standardize_target(target_slice, feature_idx=target_feat_idx)
		
		# Fallback: Apply generic normalization if normalizer is set (e.g., for SP100)
		elif self.normalizer is not None and self._normalize_targets:
			# Normalize y using the normalizer
			save_dict['y'] = self.normalizer.normalize(save_dict['y'])

		# Also permute close_price and close_price_y if needed
		if save_dict['close_price'].dim() == 2:
			save_dict['close_price'] = save_dict['close_price'].unsqueeze(-1)  # [N, T] -> [N, T, 1]
		if save_dict['close_price_y'].dim() == 2:
			save_dict['close_price_y'] = save_dict['close_price_y'].unsqueeze(-1)  # [N, T] -> [N, T, 1]

		return StocksDataDiffusion(**save_dict)
	
	
	def _get_fused_graph(self, idx: int) -> tuple:
		"""
		Get the fused (static + dynamic) graph for a given timestep, coalesced and cached.
		
		On the first call for each dynamic graph period, concatenates static and dynamic
		edges, deduplicates via coalesce (summing weights for duplicate pairs), and caches
		the result. Subsequent calls for timesteps in the same period return the cached
		tensors directly.
		
		Parameters
		----------
		idx : int
			Timestep index.
		
		Returns
		-------
		tuple of (edge_index, edge_weight)
			Fused, coalesced edge tensors.
		"""
		# Initialize fused graph cache
		if not hasattr(self, '_fused_graph_cache'):
			self._fused_graph_cache = {}
			self._fused_graph_cache_logged = False
		
		# Determine cache key: period file_index or -1 for static-only
		period_key = self._get_period_key_for_timestep(idx)
		cache_key = period_key if period_key is not None else -1
		
		if cache_key not in self._fused_graph_cache:
			# Cache miss: build fused graph
			static_ei = self._static_graph_cache['edge_index']
			static_ew = self._static_graph_cache['edge_weight']
			num_nodes = int(self._static_graph_cache['stocks_index'].shape[0])
			
			if cache_key >= 0:
				# Load and concatenate dynamic graph
				dyn_graph = self._load_dynamic_graph(cache_key)
				fused_ei = torch.cat([static_ei, dyn_graph['edge_index']], dim=1)
				fused_ew = torch.cat([static_ew, dyn_graph['edge_weight']], dim=0)
				edges_before = fused_ei.shape[1]
				
				# Coalesce: sort, deduplicate, sum weights for duplicate (i,j) pairs
				fused_ei, fused_ew = coalesce(fused_ei, fused_ew, num_nodes=num_nodes)
				edges_after = fused_ei.shape[1]
				
				if not self._fused_graph_cache_logged:
					log.info(
						f"Fused graph cache miss (period={cache_key}): "
						f"static={static_ei.shape[1]} + dynamic={dyn_graph['edge_index'].shape[1]} "
						f"= {edges_before} edges → coalesced to {edges_after} "
						f"({edges_before - edges_after} duplicates removed, "
						f"{(edges_before - edges_after) / max(edges_before, 1) * 100:.1f}% reduction)"
					)
					self._fused_graph_cache_logged = True
			else:
				# Static-only: no concatenation needed, but still coalesce for consistency
				fused_ei = static_ei
				fused_ew = static_ew
			
			self._fused_graph_cache[cache_key] = (fused_ei, fused_ew)
		
		return self._fused_graph_cache[cache_key]
	
	
	def _get_period_key_for_timestep(self, idx: int) -> Optional[int]:
		"""
		Get the dynamic graph period file_index for a given timestep.
		
		Returns None if no dynamic graph is available (either disabled or
		timestep is before the end of the first complete period).
		"""
		meta = self._dynamic_graph_meta
		if meta is None:
			return None
		
		selected_period = None
		for period_info in meta['periods']:
			if idx >= period_info['end_timestep']:
				selected_period = period_info
			else:
				break  # Periods are sorted by start_timestep
		
		if selected_period is None:
			return None
		
		return selected_period['file_index']
	
	
	def _load_dynamic_graph(self, file_index: int) -> dict:
		"""
		Load a dynamic graph .pt file, with size-1 LRU cache.
		
		Parameters
		----------
		file_index : int
			Index of the dynamic graph file to load.
		
		Returns
		-------
		dict with 'edge_index' and 'edge_weight' tensors.
		"""
		if not hasattr(self, '_dynamic_graph_file_cache'):
			self._dynamic_graph_file_cache = (None, None)
		
		cached_idx, cached_data = self._dynamic_graph_file_cache
		if cached_idx != file_index:
			cached_data = torch.load(
				osp.join(self.processed_dir, f'dynamic_graph_{file_index}.pt'),
				weights_only=True
			)
			self._dynamic_graph_file_cache = (file_index, cached_data)
		
		return cached_data
	
	
	def _get_dynamic_graph_for_timestep(self, idx: int) -> dict:
		"""
		DEPRECATED: Use _get_fused_graph() instead.
		
		Kept for backward compatibility. Returns the raw dynamic graph
		for a given timestep (without coalescing with static graph).
		"""
		period_key = self._get_period_key_for_timestep(idx)
		if period_key is None:
			return None
		return self._load_dynamic_graph(period_key)
	
	
	def _preprocess_features(self, x: torch.Tensor, feature_names: list, target_column_idx: int) -> torch.Tensor:
		"""
		Apply per-stock standardization to features during preprocessing.
		
		This is called ONCE during process() to standardize all features except the target.
		Features are standardized in-place to save memory.
		
		Args:
			x: [N_stocks, N_features, T_timesteps] feature tensor
			feature_names: List of feature names in order
			target_column_idx: Index of target feature (will be skipped)
		
		Returns:
			x with features standardized (target unchanged)
		"""
		N_stocks, N_features, T_timesteps = x.shape
		log.info(f"  Input shape: [N={N_stocks}, F={N_features}, T={T_timesteps}]")
		
		for feat_idx, feat_name in enumerate(feature_names):
			# Skip target column (will be standardized at runtime)
			if feat_idx == target_column_idx:
				log.info(f"  [{feat_idx}] {feat_name} (target) - SKIPPED (runtime standardization)")
				continue
			
			# Skip features not in standardization list
			if feat_name not in self.standardized_features:
				log.info(f"  [{feat_idx}] {feat_name} - SKIPPED (excluded from standardization)")
				continue
			
			# Get per-stock stats
			if feat_name not in self.per_stock_stats:
				log.info(f"  [{feat_idx}] {feat_name} - WARNING: no stats found")
				continue
			
			stats = self.per_stock_stats[feat_name]
			means = torch.from_numpy(stats['mean']).float()  # [N,]
			stds = torch.from_numpy(stats['std']).float()    # [N,]
			
			# Reshape for broadcasting: [N,] -> [N, 1] for [N, T] slicing
			means = means.view(-1, 1)
			stds = stds.view(-1, 1)
			
			# Apply preprocessing based on feature type
			feat_data = x[:, feat_idx, :]  # [N, T]
			
			# 1. Return features: divide by 100 first
			if 'Return' in feat_name or feat_name.startswith('ALR'):
				feat_data = feat_data / self.sp500_scale_factor
			
			# 2. Log-transform if needed
			elif feat_name in self.log_transform_features:
				feat_data = torch.log1p(feat_data)
			
			# 3. Standardize: z = (x - mean) / std
			x[:, feat_idx, :] = (feat_data - means) / (stds + 1e-8)
			
			log.info(f"  [{feat_idx}] {feat_name} - STANDARDIZED")
		
		return x
	
	
	def _standardize_target(self, tensor: torch.Tensor, feature_idx: int) -> torch.Tensor:
		"""
		Apply per-stock standardization to target variable at runtime.
		
		This is called in get() to standardize the target (y) which needs to be
		inverted during evaluation, so it must remain unstandardized in saved data.
		
		Args:
			tensor: [N, T, 1] or [N, T] target data to standardize
			feature_idx: Index into self.feature_names to get correct stats
		
		Returns:
			Standardized tensor with same shape
		"""
		if not self._apply_sp500_standardization:
			return tensor
		
		feat_name = self.feature_names[feature_idx]
		
		if feat_name not in self.per_stock_stats:
			log.info(f"WARNING: Feature '{feat_name}' not in per_stock_stats - skipping standardization")
			return tensor
		
		stats = self.per_stock_stats[feat_name]
		means = torch.from_numpy(stats['mean']).to(tensor.device).float()  # [N,]
		stds = torch.from_numpy(stats['std']).to(tensor.device).float()    # [N,]
		
		# Verify tensor size matches stats size
		if tensor.shape[0] != means.shape[0]:
			raise ValueError(
				f"Stock count mismatch: tensor has {tensor.shape[0]} stocks, "
				f"but per_stock_stats has {means.shape[0]}."
			)
		
		# Apply preprocessing: divide by 100 for return features
		if 'Return' in feat_name or feat_name.startswith('ALR'):
			tensor = tensor / self.sp500_scale_factor
		
		# Reshape for broadcasting: [N,] -> [N, 1, 1] for [N, T, 1] tensor
		if tensor.dim() == 3:  # [N, T, 1]
			means = means.view(-1, 1, 1)
			stds = stds.view(-1, 1, 1)
		elif tensor.dim() == 2:  # [N, T]
			means = means.view(-1, 1)
			stds = stds.view(-1, 1)
		else:
			raise ValueError(f"Unexpected tensor shape for standardization: {tensor.shape}")
		
		# Standardize: z = (x - mean) / std
		standardized = (tensor - means) / (stds + 1e-8)
		
		return standardized

	
	def _standardize_tensor(self, tensor: torch.Tensor, feature_idx: int) -> torch.Tensor:
		"""
		DEPRECATED: Use _standardize_target() instead.
		
		This method is kept for backward compatibility but is no longer used.
		Feature standardization is now done once during preprocessing via _preprocess_features(),
		and target standardization is done at runtime via _standardize_target().
		"""
		# Redirect to new method
		return self._standardize_target(tensor, feature_idx)

	
