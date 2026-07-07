from typing import Union
import numpy as np
import pandas as pd
import torch
from torch_geometric.utils import dense_to_sparse, to_dense_adj
from torch_geometric.data import Data

# from core.GRW import GeometricRandomWalk as GRW
# from core.GRW import CorrelatedGeometricRandomWalk as CGRW
# from models.unet_utils import get_pooling_selection_matrices
from .utils_pooling import get_pooling_selection_matrices
from pathlib import Path


def get_graph_in_pyg_format(values_path: str, adj_path: Union[str, list[str]],
							target_column_name: str = "DailyLogReturn",
							pool_ratio: float = 0.5, pool_depth: int = 3,
							) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
	"""
	Creates the PyTorch Geometric graph data from the stock price data and adjacency matrix.
	:param values_path: Path of the CSV file containing the stock price data
	:param adj_path: Path of the NumPy file containing the adjacency matrix
	:return: The graph data in PyTorch Geometric format
		# x: Node features (nodes_nb, timestamps_nb, features_nb)
		x: Node features (nodes_nb, features_nb, timestamps_nb)
		close_prices: Close prices (nodes_nb, timestamps_nb)
		edge_index: Edge index (2, edge_nb)
		edge_weight: Edge weight (edge_nb, edge_features_nb) or (edge_nb,) if single feature
		info: Additional info dictionary
	"""
	values = pd.read_csv(values_path).set_index(['Symbol', 'Date'])
	adj = np.load(adj_path)

	assert adj.shape[0] == adj.shape[1], "Adjacency matrix must be square."
	if adj.ndim == 2:
		nodes_nb, edge_nb = len(adj), np.count_nonzero(adj)
		adj = np.expand_dims(adj, axis=-1) # (nodes_nb, nodes_nb, 1)
	elif adj.ndim == 3:
		print("Nonzero edges per feature:", [np.count_nonzero(adj[..., i]) for i in range(adj.shape[-1])])
		nodes_nb, edge_nb = len(adj), np.count_nonzero(np.any(adj, axis = -1, keepdims=False), axis = (0, 1))
		print(f"Total number of edges (nonzero in any feature): {edge_nb}")
	else:
		raise ValueError("Adjacency matrix must be 2D or 3D numpy array.")
	
	x = torch.tensor(
		values.drop(columns=["Close"]).to_numpy().reshape((nodes_nb, -1, values.shape[1] - 1)), dtype=torch.float32
	) # shape (nodes_nb, timestamps_nb, features_nb)
	x = x.transpose(1, 2) # shape (nodes_nb, features_nb, timestamps_nb)
	close_prices = torch.tensor(
		values[["Close"]].to_numpy().reshape((nodes_nb, -1)), dtype=torch.float32
	)

	edge_index, edge_weight = torch.zeros((2, edge_nb), dtype=torch.long), torch.zeros((edge_nb, adj.shape[-1]), dtype=torch.float32)
	count = 0
	for i in range(nodes_nb):
		for j in range(nodes_nb):
			# if (weight := adj[i, j]) != 0:
			# 	edge_index[0, count], edge_index[1, count] = i, j
			# 	edge_weight[count] = weight
			# 	count += 1
			if np.any(weight := adj[i, j] != 0):
				edge_index[0, count], edge_index[1, count] = i, j
				edge_weight[count] = torch.FloatTensor(weight)
				count += 1

	print(f"Graph loaded with {nodes_nb} nodes, {edge_nb} edges with {edge_weight.shape[1]} features per edge, {x.shape[1]} features per node, {x.shape[2]} timestamps per node.")
	edge_weight = edge_weight.squeeze(1) # (edge_nb,) if edge_weight.shape[1] == 1 else (edge_nb, edge_features_nb)


	# Compute node degrees using edge_index and edge_weight
	adj = to_dense_adj(edge_index, edge_attr=edge_weight, max_num_nodes=nodes_nb).squeeze(0)
	orig_degrees = adj.abs().sum(dim=1)  # Sum of absolute weights of edges connected to each node
	orig_degrees = orig_degrees.sum(dim=1) if edge_weight.ndim == 2 else orig_degrees
	
	degrees = orig_degrees.clone()
	print("Original node degrees:", degrees.tolist())
	# unsorted_indices = range(len(degrees))
	depth = pool_depth


	selection_matrices, num_pooled_nodes = get_pooling_selection_matrices(degrees, pool_ratio, depth)


	# pool_ratio = 0.5
	# sorted, indices = torch.sort(degrees, descending=True)
	# selected_node_indices = indices[:int(len(indices) // (1 / pool_ratio))]
	# selected_node_degrees = sorted[:int(len(indices) // (1 / pool_ratio))]

	# all_selected_nodes = [selected_node_indices]
	# all_selected_degrees = [selected_node_degrees]

	# selection_matrix = torch.zeros((len(sorted), len(selected_node_indices)), dtype=torch.long)
	# for col, node in enumerate(selected_node_indices):
	# 	selection_matrix[node, col] = 1

	
	print('Values_path: ', values_path)
	root = Path(values_path).parent # one directory up from values_path
	print('Root path: ', root)

	for i, selection_matrix in enumerate(selection_matrices):
		selection_matrix = selection_matrix.t().detach().cpu().numpy()
		N_in, N_out = selection_matrix.shape
		# np.save(f"{root}/raw/node_selection_matrices/C_{N_in}_{N_out}.npy", selection_matrix)
		save_path = root / "node_selection_matrices" / f"C_{N_in}_{N_out}.npy"
		save_path.parent.mkdir(parents=True, exist_ok=True) # Create directory if it doesn't exist
		np.save(save_path, selection_matrix)

    # num_nodes = len(degrees)
	# all_selection_matrices = []
	# all_selected_nodes = []
	# all_selected_degrees = []

	# for _ in range(depth):
	# 	sorted, indices = torch.sort(degrees, descending=True)
	# 	selected_node_indices = indices[:(len(indices) // 2)]
	# 	selected_node_degrees = sorted[:(len(indices) // 2)]

	# 	for idx in selected_node_indices:
	# 		degrees[idx] = -1  # Mark selected nodes with -1 to exclude them in the next iteration

	# 	all_selected_nodes.append(selected_node_indices)
	# 	all_selected_degrees.append(selected_node_degrees)


	# sorted_indices = torch.argsort(degrees, descending=True)
	# selected_nodes = sorted_indices[:(len(sorted_indices) // 2)]
	# selected_degrees = degrees[selected_nodes]

	# # Create a nodes_nb x len(selected_nodes) selection matrix
	# selection_matrix = torch.zeros((len(sorted_indices), len(selected_nodes)), dtype=torch.float32)
	# for col, node in enumerate(selected_nodes):
	# 	selection_matrix[node, col] = 1.0

	# 	# Remove selected nodes from degrees
	# 	degrees = 

	# all_selection_matrices.append(selection_matrix)
	# all_selected_nodes.append(selected_nodes)
	# all_selected_degrees.append(selected_degrees)


	# for _ in range(depth):
	# 	sorted_indices = torch.argsort(degrees, descending=True)
	# 	selected_nodes = sorted_indices[:(len(sorted_indices) // 2)]

	# 	# Create a nodes_nb x len(selected_nodes) selection matrix
	# 	selection_matrix = torch.zeros((len(sorted_indices), len(selected_nodes)), dtype=torch.float32)
	# 	for col, node in enumerate(selected_nodes):
	# 		selection_matrix[node, col] = 1.0

	# 	all_selection_matrices.append(selection_matrix)
	# 	all_selected_nodes.append(selected_nodes)
	# 	unsorted_indices = sorted_indices[(len(sorted_indices) // 2):]

	# 	selected_degrees = degrees[selected_nodes]
	# 	all_selected_degrees.append(selected_degrees)

	# 	unsorted_indices = torch.sort(unsorted_indices).values
	# 	degrees = orig_degrees[unsorted_indices]

	# for nodes, degrees in zip(all_selected_nodes, all_selected_degrees):
	# 	print("*********************************\n\n\nNext set of nodes\n\n\n********************************")
	# 	for node, degree in zip(nodes, degrees):
	# 		print(f"Node {node.item()}: Degree {degree.item()}")

	info_dict = {"Features": values.drop(columns=["Close"]).columns.tolist(),
			  		"Target": target_column_name,
				  	"Num_nodes": nodes_nb
	}

	# Compute per-stock log return scale factors for de-scaling during evaluation
	# The raw CSV DailyLogReturn is pre-scaled: DailyLogReturn = raw_log_return * scale_factor
	# where scale_factor ≈ 1/per_stock_std (making std ≈ 1 for each stock)
	if target_column_name == "DailyLogReturn":
		log_return_scale_factors = compute_log_return_scale_factors(values_path)
		info_dict["log_return_scale_factors"] = log_return_scale_factors
		print(f"✅ Computed log return scale factors for {len(log_return_scale_factors)} stocks")
		print(f"   Scale factor range: [{min(log_return_scale_factors.values()):.2f}, {max(log_return_scale_factors.values()):.2f}]")

	return x, close_prices, edge_index, edge_weight, info_dict



def daily_log_returns_to_log_returns(daily_log_returns: torch.Tensor, time_dim: int = 0) -> torch.Tensor:
	"""
	Maps daily log returns back to cumulative log returns.
	"""
	return daily_log_returns.cumsum(dim=time_dim)


def daily_log_returns_to_closing_prices(log_returns: torch.Tensor, init_closing_prices: torch.Tensor, time_dim: int = 0) -> torch.Tensor:
	"""
	Maps daily log returns back to closing prices.
	"""
	return daily_log_returns_to_log_returns(log_returns, time_dim).exp() * init_closing_prices


def get_column_names(values_path_or_df: Union[str, pd.DataFrame]) -> list[str]:
	"""
	Retrieves the column names (features) of the dataset stocks
	"""
	if isinstance(values_path_or_df, str):
		values = pd.read_csv(values_path_or_df).set_index(['Symbol', 'Date'])
	elif isinstance(values_path_or_df, pd.DataFrame):
		values = values_path_or_df
	else:
		raise NotImplementedError(f"Unsupported type for values_path: {type(values_path_or_df)}")

	return values.columns.tolist()


def get_column_idx(values_path_or_df: Union[str, pd.DataFrame], column_name: str) -> int:
	"""
	Retrieves the index of a specific column in the dataset stocks
	"""
	if isinstance(values_path_or_df, str):
		values = pd.read_csv(values_path_or_df).set_index(['Symbol', 'Date'])
	elif isinstance(values_path_or_df, pd.DataFrame):
		values = values_path_or_df
	else:
		raise NotImplementedError(f"Unsupported type for values_path: {type(values_path_or_df)}")

	return values.columns.get_loc(column_name)


def get_stocks_labels(values_path: str) -> list[str]:
	"""
	Retrieves the labels (symbols) of the dataset stocks
	:return: The list of stock labels
	"""
	return pd.read_csv(values_path)["Symbol"].unique().tolist()



def test_load_adj(adj_path: str) -> np.ndarray:
	"""
	Test the loading of the adjacency matrix and print it.
	"""
	adj = np.load(adj_path)
	assert adj.ndim == 2, "Adjacency matrix must be 2-dimensional."
	assert adj.shape[0] == adj.shape[1], "Adjacency matrix must be square."

	print("Adj: ", adj)
	return adj


def get_graph_from_covariance(values_path: str, target_column_name: str = "DailyLogReturn", 
							  method: str = "covariance", cov_method: str = "sample",
							  threshold: float = 0.0, use_absolute: bool = True,
							  remove_self_loops: bool = True, normalize: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
	"""
	Creates the PyTorch Geometric graph data from stock price data using covariance/correlation of closing prices.
	
	:param values_path: Path of the CSV file containing the stock price data
	:param target_column_name: Name of the target column for prediction
	:param method: "covariance" or "correlation" - method to compute relationships
	:param cov_method: "sample" or "population" (only used when method="covariance")
	:param threshold: Minimum absolute value for an edge to be included
	:param use_absolute: Whether to use absolute values of covariance/correlation
	:param remove_self_loops: Whether to remove self-loops (diagonal entries)
	:param normalize: Whether to normalize edge weights to [0, 1]
	:return: The graph data in PyTorch Geometric format
		x: Node features (nodes_nb, features_nb, timestamps_nb)
		close_prices: Close prices (nodes_nb, timestamps_nb)
		edge_index: Edge index (2, edge_nb)
		edge_weight: Edge weight (edge_nb,)
		info: Additional info dictionary
	"""
	values = pd.read_csv(values_path).set_index(['Symbol', 'Date'])
	nodes_nb = len(values.index.get_level_values('Symbol').unique())
	
	# Create node features
	x = torch.tensor(
		values.drop(columns=["Close"]).to_numpy().reshape((nodes_nb, -1, values.shape[1] - 1)), dtype=torch.float32
	) # shape (nodes_nb, timestamps_nb, features_nb)
	x = x.transpose(1, 2) # shape (nodes_nb, features_nb, timestamps_nb)
	
	# Extract closing prices
	close_prices = torch.tensor(
		values[["Close"]].to_numpy().reshape((nodes_nb, -1)), dtype=torch.float32
	)
	
	# Compute covariance or correlation matrix
	if method == "covariance":
		if cov_method == "sample":
			relationship_matrix = torch.cov(close_prices)
		elif cov_method == "population":
			mean_prices = close_prices.mean(dim=1, keepdim=True)
			centered_prices = close_prices - mean_prices
			relationship_matrix = torch.mm(centered_prices, centered_prices.t()) / close_prices.shape[1]
		else:
			raise ValueError("cov_method must be 'sample' or 'population'")
	elif method == "correlation":
		relationship_matrix = torch.corrcoef(close_prices)
	else:
		raise ValueError("method must be 'covariance' or 'correlation'")
	
	# Process the relationship matrix to create adjacency matrix
	adj_matrix = relationship_matrix.clone()
	
	# Remove self-loops if requested
	if remove_self_loops:
		adj_matrix.fill_diagonal_(0.0)
	
	# Use absolute values if requested
	if use_absolute:
		adj_matrix = torch.abs(adj_matrix)
	
	# Apply threshold
	adj_matrix = adj_matrix * (adj_matrix > threshold)
	
	# Normalize if requested
	if normalize and adj_matrix.max() > 0:
		adj_matrix = adj_matrix / adj_matrix.max()
	
	# Convert to sparse format
	edge_index, edge_weight = dense_to_sparse(adj_matrix)
	
	print(f"Graph loaded with {nodes_nb} nodes, {edge_index.shape[1]} edges, {x.shape[1]} features per node, {x.shape[2]} timestamps per node.")
	print(f"Graph created using {method} method with threshold={threshold}")
	
	info_dict = {"Features": values.drop(columns=["Close"]).columns.tolist(),
				"Target": target_column_name,
				"Num_nodes": nodes_nb,
				"Graph_method": method,
				"Threshold": threshold
	}

	return x, close_prices, edge_index, edge_weight, info_dict


def combine_edge_features(edge_index1: torch.Tensor, edge_weight1: torch.Tensor,
						  edge_index2: torch.Tensor, edge_weight2: torch.Tensor,
						  fill_value: float = 0.0,
						  debug_print: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
	"""
	Combine two edge graphs by merging their edge indices and stacking edge weights as features.
	This function answers the specific question about combining edge_index_t_corr and edge_weight_t
	with existing edge_index and edge_weight.
	
	Args:
		edge_index1: First edge index tensor of shape (2, num_edges1) 
		edge_weight1: First edge weights of shape (num_edges1,) or (num_edges1, num_features1)
		edge_index2: Second edge index tensor of shape (2, num_edges2)
		edge_weight2: Second edge weights of shape (num_edges2,) or (num_edges2, num_features2)  
		fill_value: Value to use for missing edges in either graph
		
	Returns:
		combined_edge_index: Combined edge index of shape (2, num_combined_edges)
		combined_edge_weight: Combined edge weights of shape (num_combined_edges, total_features)
	"""
	
	# Ensure edge weights are 2D
	if edge_weight1.dim() == 1:
		edge_weight1 = edge_weight1.unsqueeze(-1)
	if edge_weight2.dim() == 1:
		edge_weight2 = edge_weight2.unsqueeze(-1)
	
	# Get dimensions
	num_features1 = edge_weight1.shape[1]
	num_features2 = edge_weight2.shape[1] 
	total_features = num_features1 + num_features2

	if debug_print:
		print(f"Combining edge features: Graph1 has {num_features1} features, Graph2 has {num_features2} features, total combined features will be {total_features}.")
	
	# Convert edge indices to tuples for easier comparison
	edges1_set = set(zip(edge_index1[0].tolist(), edge_index1[1].tolist()))
	edges2_set = set(zip(edge_index2[0].tolist(), edge_index2[1].tolist()))
	
	# Find all unique edges
	all_edges = edges1_set.union(edges2_set)
	
	# Create combined edge index
	combined_edge_list = list(all_edges)
	combined_edge_index = torch.tensor(combined_edge_list).t().long()
	
	# Create combined edge weights
	num_combined_edges = len(combined_edge_list)
	combined_edge_weight = torch.full((num_combined_edges, total_features), fill_value, dtype=torch.float32)
	
	# Map original edges to combined indices
	edge_to_idx = {edge: idx for idx, edge in enumerate(combined_edge_list)}
	
	# Fill in weights from first graph
	for i, (src, tgt) in enumerate(zip(edge_index1[0].tolist(), edge_index1[1].tolist())):
		combined_idx = edge_to_idx[(src, tgt)]
		combined_edge_weight[combined_idx, :num_features1] = edge_weight1[i]
	
	# Fill in weights from second graph  
	for i, (src, tgt) in enumerate(zip(edge_index2[0].tolist(), edge_index2[1].tolist())):
		combined_idx = edge_to_idx[(src, tgt)]
		combined_edge_weight[combined_idx, num_features1:] = edge_weight2[i]

	if debug_print:
		print(f"Combined graph has {num_combined_edges} edges with {combined_edge_weight.shape[1]} features per edge.")
	
	return combined_edge_index, combined_edge_weight



def gather_target_preds_observations_and_timestamps(
	data: Data,
	ygen: torch.Tensor,
	timestamps: torch.Tensor,
	stocks_index: torch.Tensor,
	target_column_idx: int = 1,
	debug_assert: bool = True,
	debug_print: bool = False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
	"""
	Gathers and reshapes the target, predictions, observations, and timestamps for evaluation and plotting.
	Args:
		data: The input graph data containing x and y
		ygen: The generated predictions from the model of shape (num_timestamps * nsamples, num_stocks, future_window)
		timestamps: The timestamps tensor of shape (num_timestamps * nsamples, num_stocks, 1)
		stocks_index: The stocks index tensor of shape (num_timestamps * nsamples, num_stocks, 1)
	Returns:
		preds: Reshaped predictions of shape (num_timestamps, num_stocks, future_window, nsamples).

		target: Reshaped target of shape (num_timestamps, num_stocks, future_window, 1).

		observations: Observations of shape (num_timestamps, num_stocks, past_window).

		obs_and_pred_timestamps: Timestamps for observations and predictions of shape (num_timestamps, past_window + future_window).
	"""

	num_timestamps = len(data.ptr) - 1
	num_stocks = data.x.shape[0] // num_timestamps
	num_features, past_window = data.x.shape[1], data.x.shape[2]  # num_features, past_window

	target = data.y.reshape(len(data.ptr) - 1, num_stocks, -1) # num_timestamps, num_stocks, future_window

	preds = ygen.reshape(-1, num_stocks, *target.shape[2:]) # [num_timestamps * nsamples, num_stocks, future_window]    
	nsamples = preds.shape[0] // num_timestamps

	preds_stocks_index = stocks_index.reshape(-1, num_stocks, 
												#   *target.shape[2:-1],
													1) # [num_timestamps * nsamples, num_stocks, 1]

	if debug_assert:
		assert torch.all(preds_stocks_index[:, 0] == 0), "Stocks index should be 0 for all samples."
		assert torch.all(preds_stocks_index[:, 2] == 2), "Stocks index should be 2 for all samples."
		assert torch.all(preds_stocks_index[:, -1] == num_stocks - 1), f"Stocks index should be {num_stocks - 1} for all samples."

	# Split ygen into nsamples chunks and stack them across the last dimension
	preds = torch.stack(torch.chunk(preds, nsamples, dim = 0), dim = -1) # [num_timestamps, num_stocks, future_window, nsamples]

	preds_timestamps = timestamps.reshape(-1, num_stocks, 
												#   *target.shape[2:-1],
												1) # [num_timestamps * nsamples, num_stocks, 1]
	preds_timestamps = torch.stack(torch.chunk(preds_timestamps, nsamples, dim = 0), dim = -1) # [num_timestamps, num_stocks, 1, nsamples]
	target.unsqueeze_(-1) # [num_timestamps, num_stocks, future_window, 1]

	if debug_assert:
		assert torch.all(preds_timestamps[0] == preds_timestamps[0, 0, 0, 0]), f"Timestamps index should be {preds_timestamps[0, 0, 0, 0]} for all samples but it is {preds_timestamps[0]}."
		assert torch.all(preds_timestamps[2] == preds_timestamps[2, 0, 0, 0]), f"Timestamps index should be {preds_timestamps[2, 0, 0, 0]} for all samples but it is {preds_timestamps[2]}."
		assert torch.all(preds_timestamps[-1] == preds_timestamps[-1, 0, 0, 0]), f"Timestamps index should be {preds_timestamps[-1, 0, 0, 0]} for all samples but it is {preds_timestamps[-1]}."

		assert torch.all(preds_timestamps[3, 0] == preds_timestamps[3, 0, 0, 0]), f"Timestamps index should be {preds_timestamps[3, 0, 0, 0]} for all samples but it is {preds_timestamps[3, 0]}."
		assert torch.all(preds_timestamps[3, 2] == preds_timestamps[3, 2, 0, 0]), f"Timestamps index should be {preds_timestamps[3, 2, 0, 0]} for all samples but it is {preds_timestamps[3, 2]}."
		assert torch.all(preds_timestamps[3, -1] == preds_timestamps[3, -1, 0, 0]), f"Timestamps index should be {preds_timestamps[3, -1, 0, 0]} for all samples but it is {preds_timestamps[3, -1]}."

		assert preds.shape[:3] == target.shape[:3], f"Shape mismatch: {preds.shape} != {target.shape}"
	

	# target_column_idx = 1 # NormClose
	observations = data.x[:, target_column_idx, :].reshape(-1, num_stocks, past_window)

	future_window = target.shape[2]
	
	
	obs_and_pred_timestamps = torch.arange(-past_window, future_window, device = preds_timestamps.device).view(1, -1).repeat(num_timestamps, 1) # [num_timestamps, past_window + future_window]

	# Choose the first stock wlog because all stocks have the same timestamps
	stock_idx = 0
	obs_and_pred_timestamps = obs_and_pred_timestamps + preds_timestamps[:, stock_idx, 0, 0].view(-1, 1) # Shift by the prediction start timestamp

	if debug_print:
		## Assert shifted observations match the target start and end values
		delta_obs_time = obs_and_pred_timestamps[1,0] - obs_and_pred_timestamps[0,0]
		print(f"Delta observation time: {delta_obs_time}")
		print("Observations[0:2, :]: ", observations[0:2, stock_idx, :])
		print("Target[0:2, stock]: ", target[0:2, stock_idx, :, 0])

	return (preds, target, observations, obs_and_pred_timestamps)




# def run_geometric_random_walk_baseline(observations: torch.Tensor,
# 									   obs_and_pred_timestamps: torch.Tensor,
# 									   nsamples: int = 100) -> torch.Tensor:
	
# 	num_timestamps, num_stocks, past_window = observations.shape
# 	future_window = obs_and_pred_timestamps.shape[1] - past_window

# 	# dt = 1 / past_window
# 	dt = 1
# 	grw_list = [GRW(mu = 0, sigma = 1, t0 = 0, t1 = dt * past_window, dt = dt) for _ in range(num_stocks)]

# 	grw_pred_log_returns = torch.zeros((num_timestamps, num_stocks, future_window, nsamples), device=observations.device)
# 	initial_log_returns = daily_log_returns_to_log_returns(observations, time_dim = 2).detach().cpu().numpy()[..., -1:]

# 	for stock_idx in range(num_stocks):
# 		grw = grw_list[stock_idx]

# 		for window_idx in range(len(obs_and_pred_timestamps)):
# 			# Fit the model to the past observations
# 			mu_fit, sigma_fit = grw.fit(observations[window_idx:window_idx + 1, stock_idx].detach().cpu().numpy(), paths_type="log-increments").values()
# 			grw = GRW(mu = mu_fit, sigma = sigma_fit, t0 = grw.t1, t1 = grw.t1 + dt * future_window, dt = dt)

# 			grw_sample_paths = grw.generate_paths(num_paths = nsamples, return_type = "log-returns",
# 										 initial_log_return = initial_log_returns[window_idx:window_idx+1, stock_idx])[:, -future_window:] # [nsamples, future_window]
# 			grw_pred_log_returns[window_idx, stock_idx, :, :] \
# 				= torch.from_numpy(grw_sample_paths).to(device = grw_pred_log_returns.device).T # [future_window, nsamples]

# 	return grw_pred_log_returns



# def run_correlated_geometric_random_walk_baseline(observations: torch.Tensor,
# 									   			  obs_and_pred_timestamps: torch.Tensor,
# 									   			  nsamples: int = 100
# 												  ) -> torch.Tensor:

# 	num_timestamps, num_stocks, past_window = observations.shape
# 	future_window = obs_and_pred_timestamps.shape[1] - past_window

# 	# dt = 1 / past_window
# 	dt = 1
# 	grw = CGRW(mu_vector=np.zeros(num_stocks), sigma_vector=np.ones(num_stocks), t0=0, t1=dt*past_window, dt=dt)
	

# 	grw_pred_log_returns = torch.zeros((num_timestamps, num_stocks, future_window, nsamples), device=observations.device)
# 	for window_idx in range(len(obs_and_pred_timestamps)):
# 		# Fit the model to the past observations
# 		mu_vector_fit, sigma_vector_fit, correlation_matrix_fit, covariance_matrix_fit, _ \
# 			= grw.fit(torch.permute(observations[window_idx:window_idx + 1], (0, 2, 1)).detach().cpu().numpy(), is_log_paths=True).values()

# 		grw = CGRW(mu_vector=mu_vector_fit, sigma_vector=sigma_vector_fit,
# 					covariance_matrix=covariance_matrix_fit,
# 					t0=grw.t1, t1=grw.t1 + dt * future_window, dt=dt
# 					)

# 		grw_sample_paths = grw.generate_paths(num_paths = nsamples) # [nsamples, future_window, num_stocks]
		
# 		grw_pred_log_returns[window_idx] \
# 			= torch.from_numpy(grw_sample_paths).to(device = grw_pred_log_returns.device).transpose(0, 2) # [num_stocks, future_window, nsamples]

# 	return grw_pred_log_returns


def compute_log_return_scale_factors(values_path: Union[str, pd.DataFrame]) -> dict:
	"""
	Compute per-stock scale factors for DailyLogReturn de-scaling.
	
	The raw CSV contains pre-scaled DailyLogReturn values where:
		DailyLogReturn = scale_factor * log(Close_t / Close_{t-1})
	
	This function reverse-engineers the scale factor for each stock by fitting:
		DailyLogReturn = scale * raw_log_return  (no offset, as verified empirically)
	
	Args:
		values_path: Path to values.csv or DataFrame with Symbol, Date, Close, DailyLogReturn columns
		
	Returns:
		Dictionary mapping stock symbol -> scale factor
		
	Usage in evaluation:
		actual_log_return = DailyLogReturn / scale_factor
		price_t = last_price * exp(cumsum(actual_log_returns))
	"""
	if isinstance(values_path, str):
		values = pd.read_csv(values_path).set_index(['Symbol', 'Date'])
	elif isinstance(values_path, pd.DataFrame):
		values = values_path
	else:
		raise NotImplementedError(f"Unsupported type for values_path: {type(values_path)}")
	
	# Reset index to have Symbol and Date as columns for groupby
	df = values.reset_index()
	df = df.sort_values(['Symbol', 'Date']).reset_index(drop=True)
	
	# Compute raw log returns from Close prices
	df['raw_log_return'] = df.groupby('Symbol')['Close'].transform(
		lambda x: np.log(x / x.shift(1))
	)
	
	scale_factors = {}
	for symbol in df['Symbol'].unique():
		stock_df = df[df['Symbol'] == symbol].dropna(subset=['raw_log_return'])
		
		raw_lr = stock_df['raw_log_return'].values
		scaled_lr = stock_df['DailyLogReturn'].values
		
		# Fit scale factor using least squares: DailyLogReturn = scale * raw_log_return
		# Minimizes sum((scaled_lr - scale * raw_lr)^2)
		# Solution: scale = sum(raw_lr * scaled_lr) / sum(raw_lr^2)
		scale = np.sum(raw_lr * scaled_lr) / np.sum(raw_lr * raw_lr)
		scale_factors[symbol] = float(scale)
	
	return scale_factors


def descale_log_returns(
	scaled_log_returns: torch.Tensor,
	scale_factors: Union[dict, torch.Tensor],
	stock_symbols: list = None,
	stock_dim: int = -2
) -> torch.Tensor:
	"""
	De-scale pre-scaled DailyLogReturn values back to raw log returns.
	
	Args:
		scaled_log_returns: Tensor of scaled log returns [..., N, ...] where N is num_stocks
		scale_factors: Either dict mapping symbol->scale or tensor of shape [N]
		stock_symbols: List of stock symbols (required if scale_factors is dict)
		stock_dim: Dimension corresponding to stocks (default: -2 for [..., N, F])
		
	Returns:
		Tensor of raw log returns with same shape as input
		
	Example:
		# Using dict scale factors
		raw_lr = descale_log_returns(scaled_lr, scale_factors_dict, stock_symbols=['AAPL', 'MSFT', ...])
		
		# Using tensor scale factors  
		raw_lr = descale_log_returns(scaled_lr, scale_tensor)  # scale_tensor shape [N]
	"""
	if isinstance(scale_factors, dict):
		if stock_symbols is None:
			raise ValueError("stock_symbols required when scale_factors is a dict")
		# Convert to tensor in correct order
		scale_tensor = torch.tensor(
			[scale_factors[sym] for sym in stock_symbols],
			dtype=scaled_log_returns.dtype,
			device=scaled_log_returns.device
		)
	else:
		scale_tensor = scale_factors
	
	# Reshape scale_tensor to broadcast correctly
	# For input shape [B, T, N, F] with stock_dim=-2, we need scale shape [1, 1, N, 1]
	ndim = scaled_log_returns.ndim
	shape = [1] * ndim
	shape[stock_dim] = len(scale_tensor)
	scale_tensor = scale_tensor.view(*shape)
	
	return scaled_log_returns / scale_tensor

