import os.path as osp
from typing import Callable

import pandas as pd
import torch
from torch_geometric.data import Dataset, Data
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from torch_geometric.data.storage import GlobalStorage
# from notebooks.datasets.utils import get_graph_in_pyg_format
from .utils import combine_edge_features, get_graph_in_pyg_format, get_column_idx
from .utils_covariance_graph import target_to_graph


class StocksDataDiffusion(Data):
	"""
	Custom PyG Data class for stock price diffusion.
	
	The `info` attribute contains dataset-wide properties (not per-sample)
	and is preserved without duplication during batching via custom collate_fn.
	
	Attributes:
		x: Node features [N, T, F]
		edge_index: Graph connectivity [2, E]
		edge_weight: Edge weights [E] or [E, num_edge_features]
		close_price: Historical closing prices [N, T, 1]
		y: Target log returns [N, T_future, 1]
		close_price_y: Future closing prices [N, T_future, 1]
		timestamp: Time index [N]
		stocks_index: Stock indices [N]
		info: Dataset-wide metadata dict (NOT per-sample, preserved without duplication)
	"""
	
	def __init__(self, x=None, edge_index=None, edge_weight=None, close_price=None, y=None, close_price_y=None, timestamp=None, stocks_index=None, info=None):
		super().__init__(x=x, edge_index=edge_index, edge_weight=edge_weight, close_price=close_price, y=y, close_price_y=close_price_y, timestamp=timestamp, stocks_index=stocks_index)
		if info is not None:
			self.info = info

	def __inc__(self, key, value, *args, **kwargs):
		if key in ["timestamp", "stocks_index", "info"]:
			return 0
		return super().__inc__(key, value, *args, **kwargs)

	def __cat_dim__(self, key, value, *args, **kwargs):
		if key in ["timestamp", "stocks_index"]:
			return 0  # Batch timestamps and stocks_index along dim 0
		
		if key in ["Features", "Target"]:
			return 0

		return super().__cat_dim__(key, value, *args, **kwargs)
	
	# @staticmethod
	# def collate_fn(data_list):
	# 	"""
	# 	Custom collate function that preserves `info` without duplication.
		
	# 	Use this as collate_fn in Pytorch DataLoader:
	# 		DataLoader(dataset, collate_fn=StocksDataDiffusion.collate_fn, ...)
	# 	Remark that PyG's default collate_fn, which cannot be overwritten, would duplicate `info` for each sample.
	# 	"""
	# 	from torch_geometric.data import Batch
		
	# 	# Use PyG's batch but exclude info to prevent duplication
	# 	batch = Batch.from_data_list(data_list, exclude_keys=['info'])
		
	# 	# Add info from first sample (dataset-wide, same for all samples)
	# 	if data_list and hasattr(data_list[0], 'info'):
	# 		batch.info = data_list[0].info
		
	# 	return batch

	
import torch.serialization
# At the top of your module, after imports
torch.serialization.add_safe_globals([StocksDataDiffusion, Data, DataEdgeAttr, DataTensorAttr, GlobalStorage])

from graph_signal_diffusion.datasets.normalizer import Normalizer # add normalization support for the dataset



class SP100Stocks(Dataset):
	"""
	Stock price data for the S&P 100 companies.
	"""

	def __init__(self, root: str = "../data/SP100/", values_file_name: str = "values.csv", adj_file_name: str = "adj.npy", past_window: int = 25, future_window: int = 1,
			  target_column_name: str = "DailyLogReturn",
			  corr_threshold: float = None,
			  pool_ratio: float = 0.5,
			  force_reload: bool = False, transform: Callable = None):
		self.values_file_name = values_file_name
		self.adj_file_name = adj_file_name
		self.past_window = past_window
		self.future_window = future_window
		self.target_column_name = target_column_name
		self.corr_threshold = corr_threshold
		self.pool_ratio = pool_ratio 

		# NEW: Normalization support 
		self.normalizer: Normalizer = None
		self._normalize_targets: bool = True  # Whether to normalize target values

		super().__init__(root, force_reload=force_reload, transform=transform)

	@property
	def raw_file_names(self) -> list[str]:
		return [
			self.values_file_name, self.adj_file_name
		]

	@property
	def processed_file_names(self) -> list[str]:
		return [
			f'timestep_{idx}.pt' for idx in range(len(self))
		]

	def download(self) -> None:
		pass

	def process(self) -> None:
		x, close_prices, edge_index, edge_weight, info_dict = get_graph_in_pyg_format(
			values_path=osp.join(self.root, f"raw/{self.values_file_name}"),
			adj_path=osp.join(self.root, f"raw/{self.adj_file_name}"),
			target_column_name=self.target_column_name,
			pool_ratio=self.pool_ratio
		)


		target_column_idx = get_column_idx(values_path_or_df=osp.join(self.root, f"raw/{self.values_file_name}"),
											column_name=info_dict["Target"]
											) - 1 # subtract one because we dropped the first column from values
		print("Target column name / index:", f"{info_dict['Target']} / {target_column_idx}")

		# Save target column index for later use
		info_dict["Target_column_idx"] = target_column_idx
		self.info = info_dict
		

		
		timestamps = []
		for idx in range(x.shape[2] - self.past_window - self.future_window):
			if self.corr_threshold is not None:	
				# Recompute edge_index and edge_weight based on correlation of closing prices in the current window
				# window_close_prices = close_prices[:, idx:idx + self.past_window]
				past_window_y = x[:, target_column_idx, idx:idx + self.past_window]

				# print(f"Last stock target values in all window: {x[-1, target_column_idx, :]}")

				# print(f"At time index {idx}, past window {info_dict['Target']} is {past_window_y}.")
				edge_index_t_corr, edge_weight_t_corr = target_to_graph(
					past_window_y,
					method="correlation", 
					threshold=self.corr_threshold,
					use_absolute=True,
					remove_self_loops=True,
					normalize=True,
					save_path=self.root + f"/raw/temporal_corr_{idx}.pdf" if idx < 10 else None
				)

				# Make sure none of the edge weights are NaN
				assert not torch.isnan(edge_weight_t_corr).any(), "Edge weights contain {} many NaN values.".format(torch.isnan(edge_weight_t_corr).sum().item())

				print(f"At time index {idx}, computed temporal correlation graph with {edge_weight_t_corr.size(0)} edges based on closing prices in the window [{idx}, {idx + self.past_window}).") if idx < 10 else None

				# Combine your graphs
				combined_edge_index, combined_edge_weight = combine_edge_features(
					edge_index, edge_weight,           # Existing graph
					edge_index_t_corr, edge_weight_t_corr,  # Windowed temporal correlation graph
					fill_value=0.0,                     # Value for missing edges
					debug_print= idx < 10               # Print debug info for first 10 time steps
				)

				assert combined_edge_weight.dim() == 2, "combined_edge_weight should have shape (num_edges, num_features)"
				print(f"Temporal correlation graph added {combined_edge_weight.size(0) - edge_weight.size(0)} additional edges to multi-graph.") if idx < 10 else None

				# Make sure none of the edge weights are NaN
				assert not torch.isnan(combined_edge_weight).any(), "Edge weights contain {} many NaN values.".format(torch.isnan(combined_edge_weight).sum().item())

			else:
				combined_edge_index, combined_edge_weight = edge_index, edge_weight
			
			timestamps.append(
				StocksDataDiffusion(
					x=x[:, :, idx:idx + self.past_window],
					edge_index=combined_edge_index,
					edge_weight=combined_edge_weight,
					close_price=close_prices[:, idx:idx + self.past_window],
					y=x[:, target_column_idx, idx + self.past_window:idx + self.past_window + self.future_window],
					close_price_y=close_prices[:, idx + self.past_window:idx + self.past_window + self.future_window],
					stocks_index=torch.arange(x.shape[0]),
					timestamp=torch.LongTensor([idx]).repeat(x.shape[0]), # Repeat for each node in the batch
					info=info_dict
				)
			)
		

		# Original code without dynamic graph computation
		# timestamps = [
		# 	StocksDataDiffusion(
		# 		x=x[:, :, idx:idx + self.past_window],
		# 		edge_index=edge_index,
		# 		edge_weight=edge_weight,
		# 		close_price=close_prices[:, idx:idx + self.past_window],
		# 		y=x[:, target_column_idx, idx + self.past_window:idx + self.past_window + self.future_window],
		# 		close_price_y=close_prices[:, idx + self.past_window:idx + self.past_window + self.future_window],
		# 		stocks_index=torch.arange(x.shape[0]),
		# 		timestamp=torch.LongTensor([idx]).repeat(x.shape[0]), # Repeat for each node in the batch
		# 		info=info_dict
		# 	) for idx in range(x.shape[2] - self.past_window - self.future_window)
		# ]



		# # Save processed data objects
		# for t, timestep in enumerate(timestamps):
		# 	torch.save(
		# 		timestep, osp.join(self.processed_dir, f"timestep_{t}.pt")
		# 	)

		# For future compatibility with torch.load(), save only tensors/primitives, reconstruct objects on load.
		for t, timestep in enumerate(timestamps):
			# Save as dict of tensors instead of Data object
			save_dict = {
				'x': timestep.x,
				'edge_index': timestep.edge_index,
				'edge_weight': timestep.edge_weight,
				'close_price': timestep.close_price,
				'y': timestep.y,
				'close_price_y': timestep.close_price_y,
				'timestamp': timestep.timestamp,
				'stocks_index': timestep.stocks_index,
				'info': timestep.info,
			}
			torch.save(save_dict, osp.join(self.processed_dir, f"timestep_{t}.pt"))
			

	def len(self) -> int:
		values = pd.read_csv(self.raw_paths[0]).set_index(['Symbol', 'Date'])
		return len(values.loc[values.index[0][0]]) - self.past_window - self.future_window
	

	def set_normalizer(self, normalizer: Normalizer) -> None:
		"""Set the normalizer for the dataset."""
		self.normalizer = normalizer
		print(f"✅ Normalizer set for SP100Stocks dataset")


	def get(self, idx: int) -> Data:
		save_dict = torch.load(osp.join(self.processed_dir, f'timestep_{idx}.pt'), weights_only=True)
		
		# Permute x from [N, F, T] to [N, T, F]
		save_dict['x'] = save_dict['x'].permute(0, 2, 1)  # E.g., [N, 8, 20] -> [N, 20, 8]
		
		# Add feature dimension to y: [N, T] -> [N, T, 1]
		save_dict['y'] = save_dict['y'].unsqueeze(-1)  # [N, 5] -> [N, 5, 1]


		# NEW: Apply normalization to target y if normalizer is set
		if self.normalizer is not None and self._normalize_targets:
			# Normalize y using the normalizer
			save_dict['y'] = self.normalizer.normalize(save_dict['y'])
		
			# # Optionally, normalize close_price_y as well
			# save_dict['close_price_y'] = self.normalizer.normalize(save_dict['close_price_y'])

		
		# Also permute close_price and close_price_y if needed
		if save_dict['close_price'].dim() == 2:
			save_dict['close_price'] = save_dict['close_price'].unsqueeze(-1)  # [N, T] -> [N, T, 1]
		if save_dict['close_price_y'].dim() == 2:
			save_dict['close_price_y'] = save_dict['close_price_y'].unsqueeze(-1)  # [N, T] -> [N, T, 1]


		# # ← NEW: Normalize close prices too
		# if self.normalizer is not None and self._normalize_targets:
		# 	save_dict['close_price'] = self.normalizer.normalize(save_dict['close_price'])
		# 	save_dict['close_price_y'] = self.normalizer.normalize(save_dict['close_price_y'])

		
		return StocksDataDiffusion(**save_dict)

	

# class SP100Stocks(Dataset):
# 	"""
# 	Stock price data for the S&P 100 companies.
# 	"""

# 	def __init__(self, root: str = "../data/SP100/", values_file_name: str = "values.csv", adj_file_name: str = "adj.npy", past_window: int = 25, future_window: int = 1,
# 			  target_column_name: str = "NormClose",
# 			  corr_threshold: float = None,
# 			  pool_ratio: float = 0.5,
# 			  force_reload: bool = False, transform: Callable = None):
# 		self.values_file_name = values_file_name
# 		self.adj_file_name = adj_file_name
# 		self.past_window = past_window
# 		self.future_window = future_window
# 		self.target_column_name = target_column_name
# 		self.corr_threshold = corr_threshold
# 		self.pool_ratio = pool_ratio 
# 		super().__init__(root, force_reload=force_reload, transform=transform)

# 	@property
# 	def raw_file_names(self) -> list[str]:
# 		return [
# 			self.values_file_name, self.adj_file_name
# 		]

# 	@property
# 	def processed_file_names(self) -> list[str]:
# 		return [
# 			f'timestep_{idx}.pt' for idx in range(len(self))
# 		]

# 	def download(self) -> None:
# 		pass

# 	def process(self) -> None:
# 		x, close_prices, edge_index, edge_weight, info_dict = get_graph_in_pyg_format(
# 			values_path=osp.join(self.root, f"raw/{self.values_file_name}"),
# 			adj_path=osp.join(self.root, f"raw/{self.adj_file_name}"),
# 			target_column_name=self.target_column_name,
# 			pool_ratio=self.pool_ratio
# 		)

# 		self.info = info_dict
# 		target_column_idx = get_column_idx(values_path_or_df=osp.join(self.root, f"raw/{self.values_file_name}"),
# 											column_name=info_dict["Target"]
# 											) - 1 # subtract one because we dropped the first column from values
# 		print("Target column name / index:", f"{info_dict['Target']} / {target_column_idx}")

		
# 		timestamps = []
# 		for idx in range(x.shape[2] - self.past_window - self.future_window):
# 			if self.corr_threshold is not None:	
# 				# Recompute edge_index and edge_weight based on correlation of closing prices in the current window
# 				# window_close_prices = close_prices[:, idx:idx + self.past_window]
# 				past_window_y = x[:, target_column_idx, idx:idx + self.past_window]

# 				# print(f"Last stock target values in all window: {x[-1, target_column_idx, :]}")

# 				# print(f"At time index {idx}, past window {info_dict['Target']} is {past_window_y}.")
# 				edge_index_t_corr, edge_weight_t_corr = target_to_graph(
# 					past_window_y,
# 					method="correlation", 
# 					threshold=self.corr_threshold,
# 					use_absolute=True,
# 					remove_self_loops=True,
# 					normalize=True,
# 					save_path=self.root + f"/raw/temporal_corr_{idx}.pdf" if idx < 10 else None
# 				)

# 				# Make sure none of the edge weights are NaN
# 				assert not torch.isnan(edge_weight_t_corr).any(), "Edge weights contain {} many NaN values.".format(torch.isnan(edge_weight_t_corr).sum().item())

# 				print(f"At time index {idx}, computed temporal correlation graph with {edge_weight_t_corr.size(0)} edges based on closing prices in the window [{idx}, {idx + self.past_window}).") if idx < 10 else None

# 				# Combine your graphs
# 				combined_edge_index, combined_edge_weight = combine_edge_features(
# 					edge_index, edge_weight,           # Existing graph
# 					edge_index_t_corr, edge_weight_t_corr,  # Windowed temporal correlation graph
# 					fill_value=0.0,                     # Value for missing edges
# 					debug_print= idx < 10               # Print debug info for first 10 time steps
# 				)

# 				assert combined_edge_weight.dim() == 2, "combined_edge_weight should have shape (num_edges, num_features)"
# 				print(f"Temporal correlation graph added {combined_edge_weight.size(0) - edge_weight.size(0)} additional edges to multi-graph.") if idx < 10 else None

# 				# Make sure none of the edge weights are NaN
# 				assert not torch.isnan(combined_edge_weight).any(), "Edge weights contain {} many NaN values.".format(torch.isnan(combined_edge_weight).sum().item())

# 			else:
# 				combined_edge_index, combined_edge_weight = edge_index, edge_weight
			
# 			timestamps.append(
# 				StocksDataDiffusion(
# 					x=x[:, :, idx:idx + self.past_window],
# 					edge_index=combined_edge_index,
# 					edge_weight=combined_edge_weight,
# 					close_price=close_prices[:, idx:idx + self.past_window],
# 					y=x[:, target_column_idx, idx + self.past_window:idx + self.past_window + self.future_window],
# 					close_price_y=close_prices[:, idx + self.past_window:idx + self.past_window + self.future_window],
# 					stocks_index=torch.arange(x.shape[0]),
# 					timestamp=torch.LongTensor([idx]).repeat(x.shape[0]), # Repeat for each node in the batch
# 					info=info_dict
# 				)
# 			)
		

# 		# Original code without dynamic graph computation
# 		# timestamps = [
# 		# 	StocksDataDiffusion(
# 		# 		x=x[:, :, idx:idx + self.past_window],
# 		# 		edge_index=edge_index,
# 		# 		edge_weight=edge_weight,
# 		# 		close_price=close_prices[:, idx:idx + self.past_window],
# 		# 		y=x[:, target_column_idx, idx + self.past_window:idx + self.past_window + self.future_window],
# 		# 		close_price_y=close_prices[:, idx + self.past_window:idx + self.past_window + self.future_window],
# 		# 		stocks_index=torch.arange(x.shape[0]),
# 		# 		timestamp=torch.LongTensor([idx]).repeat(x.shape[0]), # Repeat for each node in the batch
# 		# 		info=info_dict
# 		# 	) for idx in range(x.shape[2] - self.past_window - self.future_window)
# 		# ]



# 		# # Save processed data objects
# 		# for t, timestep in enumerate(timestamps):
# 		# 	torch.save(
# 		# 		timestep, osp.join(self.processed_dir, f"timestep_{t}.pt")
# 		# 	)

# 		# For future compatibility with torch.load(), save only tensors/primitives, reconstruct objects on load.
# 		for t, timestep in enumerate(timestamps):
# 			# Save as dict of tensors instead of Data object
# 			save_dict = {
# 				'x': timestep.x,
# 				'edge_index': timestep.edge_index,
# 				'edge_weight': timestep.edge_weight,
# 				'close_price': timestep.close_price,
# 				'y': timestep.y,
# 				'close_price_y': timestep.close_price_y,
# 				'timestamp': timestep.timestamp,
# 				'stocks_index': timestep.stocks_index,
# 				'info': timestep.info,
# 			}
# 			torch.save(save_dict, osp.join(self.processed_dir, f"timestep_{t}.pt"))
			

# 	def len(self) -> int:
# 		values = pd.read_csv(self.raw_paths[0]).set_index(['Symbol', 'Date'])
# 		return len(values.loc[values.index[0][0]]) - self.past_window - self.future_window
	

# 	# def get(self, idx: int) -> Data:
# 	# 	save_dict = torch.load(osp.join(self.processed_dir, f'timestep_{idx}.pt'), weights_only=True)
# 	# 	return StocksDataDiffusion(**save_dict)

# 	def get(self, idx: int) -> Data:
# 		save_dict = torch.load(osp.join(self.processed_dir, f'timestep_{idx}.pt'), weights_only=True)
		
# 		# Permute x from [N, F, T] to [N, T, F]
# 		save_dict['x'] = save_dict['x'].permute(0, 2, 1)  # E.g., [N, 8, 20] -> [N, 20, 8]
		
# 		# Add feature dimension to y: [N, T] -> [N, T, 1]
# 		save_dict['y'] = save_dict['y'].unsqueeze(-1)  # [N, 5] -> [N, 5, 1]
		
# 		# Also permute close_price and close_price_y if needed
# 		if save_dict['close_price'].dim() == 2:
# 			save_dict['close_price'] = save_dict['close_price'].unsqueeze(-1)  # [N, T] -> [N, T, 1]
# 		if save_dict['close_price_y'].dim() == 2:
# 			save_dict['close_price_y'] = save_dict['close_price_y'].unsqueeze(-1)  # [N, T] -> [N, T, 1]
		
# 		return StocksDataDiffusion(**save_dict)

	


