#!/usr/bin/env python3
# DEPRECATED: moved to src/graph_signal_diffusion/cli/stock/clean.py
# Use: python -m graph_signal_diffusion.cli.stock.clean  or  stock-clean
"""
Clean SP500 data and regenerate all derived files with consistent dimensions.

This script:
1. Cleans values.csv using chosen method (common_range, drop_incomplete, forward_fill)
2. Recomputes fundamentals-based adjacency matrix on cleaned data
3. Filters stocks.csv to match cleaned stock list
4. Filters and renormalizes fundamentals.csv
5. Computes graph diagnostics (degree stats/distribution, connectivity)
6. Saves all outputs to a parameterized cleaned dataset directory under data/sp500/

Usage:
    # Basic usage with default parameters
    python scripts/clean_sp500_data.py --method drop_incomplete --min-coverage 0.95
    python scripts/clean_sp500_data.py --method common_range
    python scripts/clean_sp500_data.py --method forward_fill
    
    # Optional edge-weight thresholding (default: disabled)
    python scripts/clean_sp500_data.py --method drop_incomplete --min-coverage 0.95 --edge-weight-threshold 0.7
    
    # With sector bonus weighting (default: 0.0)
    python scripts/clean_sp500_data.py --method drop_incomplete --min-coverage 0.95 --sector-bonus 0.2
    
    # Combined: optional threshold + sector bonus
    python scripts/clean_sp500_data.py --method drop_incomplete --min-coverage 0.95 --edge-weight-threshold 0.6 --sector-bonus 0.3
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
from datetime import datetime
import shutil
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Disable LaTeX text rendering to avoid missing font errors
matplotlib.rcParams['text.usetex'] = False

import networkx as nx
from matplotlib.gridspec import GridSpec

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def clean_values_common_range(df: pd.DataFrame) -> pd.DataFrame:
    """
    Method 1: Keep all stocks, use only dates common to ALL stocks.
    """
    print("\n" + "="*80)
    print("Method: Common Date Range")
    print("="*80)
    
    # Get date coverage per stock
    date_counts = df.groupby('Symbol')['Date'].nunique()
    all_dates = df['Date'].unique()
    
    print(f"Total stocks: {len(date_counts)}")
    print(f"Total dates: {len(all_dates)}")
    
    # Find stocks with full coverage
    max_dates = date_counts.max()
    stocks_with_max = date_counts[date_counts == max_dates].index.tolist()
    
    if len(stocks_with_max) == len(date_counts):
        print("✓ All stocks already have same number of dates!")
        return df
    
    # Get dates that exist for ALL stocks
    common_dates = None
    for symbol in date_counts.index:
        symbol_dates = set(df[df['Symbol'] == symbol]['Date'].unique())
        if common_dates is None:
            common_dates = symbol_dates
        else:
            common_dates = common_dates.intersection(symbol_dates)
    
    common_dates = sorted(list(common_dates))
    
    print(f"\nCommon dates: {len(common_dates)} ({len(common_dates)/len(all_dates)*100:.1f}% retention)")
    print(f"Date range: {common_dates[0]} to {common_dates[-1]}")
    
    # Filter to common dates
    df_clean = df[df['Date'].isin(common_dates)].copy()
    
    # Verify
    verify_counts = df_clean.groupby('Symbol')['Date'].nunique()
    assert verify_counts.nunique() == 1, "Stocks still have different date counts!"
    
    print(f"\n✓ Cleaned dataset: {len(verify_counts)} stocks × {verify_counts.iloc[0]} dates")
    
    return df_clean


def clean_values_drop_incomplete(df: pd.DataFrame, min_coverage: float = 0.90) -> pd.DataFrame:
    """
    Method 2: Drop stocks below coverage threshold, then use common range.
    """
    print("\n" + "="*80)
    print(f"Method: Drop Incomplete Stocks (min_coverage={min_coverage})")
    print("="*80)
    
    # Get date coverage per stock
    all_dates = df['Date'].unique()
    date_counts = df.groupby('Symbol')['Date'].nunique()
    max_dates = len(all_dates)
    
    print(f"Total stocks: {len(date_counts)}")
    print(f"Total dates: {max_dates}")
    
    # Calculate coverage
    coverage = date_counts / max_dates
    
    # Filter stocks
    min_dates = int(max_dates * min_coverage)
    stocks_to_keep = coverage[coverage >= min_coverage].index.tolist()
    stocks_to_drop = coverage[coverage < min_coverage].index.tolist()
    
    print(f"\nCoverage threshold: {min_coverage*100:.0f}% ({min_dates} dates)")
    print(f"Stocks to keep: {len(stocks_to_keep)} ({len(stocks_to_keep)/len(date_counts)*100:.1f}%)")
    print(f"Stocks to drop: {len(stocks_to_drop)}")
    
    if stocks_to_drop:
        print("\nDropped stocks:")
        for symbol in stocks_to_drop:
            print(f"  - {symbol}: {date_counts[symbol]} dates ({coverage[symbol]*100:.1f}%)")
    
    # Filter dataset
    df_filtered = df[df['Symbol'].isin(stocks_to_keep)].copy()
    
    # Now apply common range on remaining stocks
    df_clean = clean_values_common_range(df_filtered)
    
    return df_clean


def clean_values_forward_fill(df: pd.DataFrame) -> pd.DataFrame:
    """
    Method 3: Forward fill missing dates for each stock.
    """
    print("\n" + "="*80)
    print("Method: Forward Fill")
    print("="*80)
    
    # Get all unique dates
    all_dates = sorted(df['Date'].unique())
    all_symbols = df['Symbol'].unique()
    
    print(f"Total stocks: {len(all_symbols)}")
    print(f"Total dates: {len(all_dates)}")
    print(f"Date range: {all_dates[0]} to {all_dates[-1]}")
    
    # Process each stock
    cleaned_dfs = []
    synthetic_count = 0
    
    for symbol in all_symbols:
        stock_data = df[df['Symbol'] == symbol].copy()
        stock_data = stock_data.set_index('Date').sort_index()
        
        # Reindex to full date range
        original_count = len(stock_data)
        stock_data = stock_data.reindex(all_dates)
        
        # Forward fill numeric columns
        numeric_cols = stock_data.select_dtypes(include=[np.number]).columns
        stock_data[numeric_cols] = stock_data[numeric_cols].fillna(method='ffill')
        
        # Backfill any remaining NaNs at the start
        stock_data[numeric_cols] = stock_data[numeric_cols].fillna(method='bfill')
        
        # Restore Symbol column
        stock_data['Symbol'] = symbol
        stock_data = stock_data.reset_index()
        
        synthetic_count += (len(all_dates) - original_count)
        cleaned_dfs.append(stock_data)
    
    df_clean = pd.concat(cleaned_dfs, ignore_index=True)
    
    print(f"\n✓ Filled {synthetic_count} missing date-stock combinations")
    print(f"  Average per stock: {synthetic_count / len(all_symbols):.1f} dates")
    print(f"  Synthetic data: {synthetic_count / len(df_clean) * 100:.2f}%")
    
    # Verify
    verify_counts = df_clean.groupby('Symbol')['Date'].nunique()
    assert verify_counts.nunique() == 1, "Stocks still have different date counts!"
    
    print(f"\n✓ Cleaned dataset: {len(all_symbols)} stocks × {len(all_dates)} dates")
    
    return df_clean


def compute_sector_adjacency(stocks_df: pd.DataFrame, stock_order: list) -> np.ndarray:
    """
    Compute binary adjacency matrix based on sector membership.
    Stocks in the same sector get an edge.
    """
    print("\n" + "="*80)
    print("Computing Sector Adjacency Matrix")
    print("="*80)
    
    n_stocks = len(stock_order)
    sector_adj = np.zeros((n_stocks, n_stocks), dtype=float)
    
    # Create symbol to index mapping
    symbol_to_idx = {symbol: i for i, symbol in enumerate(stock_order)}
    
    # Create symbol to sector mapping
    symbol_to_sector = dict(zip(stocks_df['Symbol'], stocks_df['Sector']))
    
    # Build sector adjacency
    for i, symbol_i in enumerate(stock_order):
        sector_i = symbol_to_sector.get(symbol_i, None)
        if sector_i is None:
            continue
        
        for j, symbol_j in enumerate(stock_order):
            if i == j:
                continue
            sector_j = symbol_to_sector.get(symbol_j, None)
            if sector_j is None:
                continue
            
            # Same sector -> edge
            if sector_i == sector_j:
                sector_adj[i, j] = 1.0
    
    # Count sectors
    sectors = [symbol_to_sector.get(s, 'Unknown') for s in stock_order]
    sector_counts = pd.Series(sectors).value_counts()
    
    print(f"Sectors found: {len(sector_counts)}")
    print(f"Number of sector edges: {np.count_nonzero(sector_adj)}")
    print(f"Sector edge density: {np.count_nonzero(sector_adj) / (n_stocks * n_stocks):.4f}")
    
    print("\nSector distribution:")
    for sector, count in sector_counts.head(10).items():
        print(f"  {sector}: {count} stocks")
    
    return sector_adj


def compute_fundamentals_adjacency(
    fundamentals_df: pd.DataFrame,
    stocks_df: pd.DataFrame,
    stock_order: list,
    edge_weight_threshold: float = 0.0,
    sector_bonus: float = 0.0,
    corr_method: str = 'spearman',
) -> tuple:
    """
    Compute weighted adjacency matrix from fundamentals and sector information.
    
    Final adjacency (SP100 notebook style):
        adj = abs(corr(fundamentals)) + sector_bonus * sector_adjacency
        adj = adj * (adj >= edge_weight_threshold)   # optional
        adj = adj / adj.max()                        # if max > 0
    
    Args:
        fundamentals_df: Fundamentals dataframe (must include Symbol)
        stocks_df: Stocks dataframe with Sector column
        stock_order: Ordered list of stock symbols
        edge_weight_threshold: Optional post-processing threshold.
            If > 0, edges with weights below this value are zeroed.
            If <= 0, thresholding is disabled.
        sector_bonus: Weight bonus for stocks in the same sector
        corr_method: Correlation method ('spearman' or 'pearson')
    
    Returns:
        adj_matrix: Final weighted adjacency matrix
        corr_np: Absolute fundamentals correlation matrix (diag=0)
        sector_adj: Binary sector adjacency matrix (diag=0)
    """
    print("\n" + "="*80)
    print("Computing Fundamentals-Based Weighted Adjacency Matrix")
    print(f"  Correlation method: {corr_method}")
    if edge_weight_threshold > 0:
        print(f"  Edge-weight threshold: {edge_weight_threshold}")
    else:
        print("  Edge-weight threshold: disabled")
    print(f"  Sector bonus: {sector_bonus}")
    print("="*80)

    fundamentals = fundamentals_df.set_index('Symbol').reindex(stock_order)
    numeric_cols = fundamentals.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric fundamentals columns found to build adjacency.")

    missing_rows = int(fundamentals[numeric_cols].isna().all(axis=1).sum())
    if missing_rows > 0:
        print(f"Warning: {missing_rows} symbols have missing/empty fundamentals rows; filling with 0.")

    fundamentals_numeric = fundamentals[numeric_cols].fillna(0.0)
    print(f"Fundamentals matrix: {fundamentals_numeric.shape[0]} stocks × {fundamentals_numeric.shape[1]} features")

    # Match notebook approach: correlation across stocks from all fundamentals.
    corr_matrix = fundamentals_numeric.transpose().corr(method=corr_method).fillna(0.0)
    corr_np = np.abs(corr_matrix.to_numpy(dtype=float))
    np.fill_diagonal(corr_np, 0.0)

    print(f"\nFundamentals correlation matrix: {corr_np.shape[0]} × {corr_np.shape[1]}")
    print(f"Correlation stats: min={corr_np.min():.4f}, max={corr_np.max():.4f}, mean={corr_np.mean():.4f}")

    # Always compute sector adjacency for diagnostics/visualization when stocks metadata exists.
    if stocks_df is not None:
        sector_adj = compute_sector_adjacency(stocks_df, stock_order)
    else:
        sector_adj = np.zeros_like(corr_np)
        if sector_bonus > 0:
            print("Warning: sector_bonus > 0 but stocks.csv is missing. Sector bonus is skipped.")

    adj_matrix = corr_np.copy()
    if sector_bonus > 0:
        adj_matrix = adj_matrix + sector_bonus * sector_adj

    print(f"\nCombined adjacency:")
    print(f"  Formula: adj = |fundamentals_corr| + {sector_bonus} × sector_adjacency")
    print(f"  Value range (pre-threshold): [{adj_matrix.min():.4f}, {adj_matrix.max():.4f}]")
    print(f"  Mean weight (pre-threshold): {adj_matrix.mean():.4f}")
    
    # Optionally apply thresholding.
    nonzero_before = np.count_nonzero(adj_matrix)
    if edge_weight_threshold > 0:
        print(f"\nApplying edge-weight threshold {edge_weight_threshold}:")
        print(f"  Before: {nonzero_before} nonzero edges")
        adj_matrix = adj_matrix * (adj_matrix >= edge_weight_threshold)
        nonzero_after = np.count_nonzero(adj_matrix)
        print(f"  After:  {nonzero_after} nonzero edges")
        print(f"  Edge density: {nonzero_after / (adj_matrix.shape[0] * adj_matrix.shape[1]):.4f}")
        print(f"  Sparsification: {(1 - nonzero_after / (adj_matrix.shape[0] * adj_matrix.shape[1])) * 100:.1f}% of edges removed")
    else:
        print("\nEdge-weight thresholding disabled (keeping all positive weighted edges).")
        print(f"  Nonzero edges: {nonzero_before}")
        print(f"  Edge density: {nonzero_before / (adj_matrix.shape[0] * adj_matrix.shape[1]):.4f}")

    # Remove self loops and max-normalize, matching notebook behavior.
    np.fill_diagonal(adj_matrix, 0.0)
    max_val = float(adj_matrix.max()) if adj_matrix.size > 0 else 0.0
    if max_val > 0:
        adj_matrix = adj_matrix / max_val

    print(f"\nFinal adjacency stats (post-normalization): min={adj_matrix.min():.4f}, max={adj_matrix.max():.4f}, mean={adj_matrix.mean():.4f}")

    return adj_matrix, corr_np, sector_adj


def compute_graph_metadata(adj_matrix: np.ndarray, stock_order: list) -> dict:
    """
    Compute graph-level diagnostics from a weighted adjacency matrix.

    Connectivity metrics are computed on an unweighted graph where edge exists
    when adjacency weight > 0. Degree stats are reported for both unweighted
    and weighted degrees.
    """
    print("\n" + "="*80)
    print("Computing Graph Metadata")
    print("="*80)

    if adj_matrix.ndim != 2 or adj_matrix.shape[0] != adj_matrix.shape[1]:
        raise ValueError(f"adj_matrix must be square, got shape {adj_matrix.shape}")

    n = int(adj_matrix.shape[0])
    adj = np.array(adj_matrix, dtype=float, copy=True)
    np.fill_diagonal(adj, 0.0)

    binary_adj = (adj > 0).astype(np.int8)
    degrees = binary_adj.sum(axis=1).astype(int)
    weighted_degrees = adj.sum(axis=1).astype(float)

    unique_deg, counts_deg = np.unique(degrees, return_counts=True)
    degree_distribution = {
        str(int(k)): int(v) for k, v in zip(unique_deg.tolist(), counts_deg.tolist())
    }

    undirected_edges = int(np.count_nonzero(np.triu(binary_adj, k=1)))
    directed_nonzero_edges = int(np.count_nonzero(binary_adj))
    possible_undirected_edges = (n * (n - 1)) // 2 if n > 1 else 0
    possible_directed_edges = n * (n - 1) if n > 1 else 0

    G = nx.from_numpy_array(binary_adj)
    components = list(nx.connected_components(G))
    component_sizes = sorted([int(len(c)) for c in components], reverse=True)
    num_components = int(len(component_sizes))
    largest_component_size = int(component_sizes[0]) if component_sizes else 0
    largest_component_fraction = float(largest_component_size / n) if n > 0 else 0.0

    isolated_indices = np.where(degrees == 0)[0].tolist()
    isolated_symbols = [stock_order[i] for i in isolated_indices]

    lcc_radius = None
    lcc_diameter = None
    lcc_avg_shortest_path = None
    if components:
        largest_component_nodes = max(components, key=len)
        G_lcc = G.subgraph(largest_component_nodes).copy()
        if G_lcc.number_of_nodes() == 1:
            lcc_radius = 0
            lcc_diameter = 0
            lcc_avg_shortest_path = 0.0
        elif G_lcc.number_of_nodes() > 1:
            lcc_radius = int(nx.radius(G_lcc))
            lcc_diameter = int(nx.diameter(G_lcc))
            lcc_avg_shortest_path = float(nx.average_shortest_path_length(G_lcc))

    graph_metadata = {
        'num_nodes': n,
        'num_edges_undirected': undirected_edges,
        'num_edges_directed_nonzero': directed_nonzero_edges,
        'edge_density_undirected': (
            float(undirected_edges / possible_undirected_edges)
            if possible_undirected_edges > 0 else 0.0
        ),
        'edge_density_directed': (
            float(directed_nonzero_edges / possible_directed_edges)
            if possible_directed_edges > 0 else 0.0
        ),
        'degree_stats': {
            'min': int(degrees.min()) if n > 0 else 0,
            'max': int(degrees.max()) if n > 0 else 0,
            'mean': float(degrees.mean()) if n > 0 else 0.0,
            'median': float(np.median(degrees)) if n > 0 else 0.0,
            'std': float(degrees.std()) if n > 0 else 0.0,
        },
        'weighted_degree_stats': {
            'min': float(weighted_degrees.min()) if n > 0 else 0.0,
            'max': float(weighted_degrees.max()) if n > 0 else 0.0,
            'mean': float(weighted_degrees.mean()) if n > 0 else 0.0,
            'median': float(np.median(weighted_degrees)) if n > 0 else 0.0,
            'std': float(weighted_degrees.std()) if n > 0 else 0.0,
        },
        'degree_distribution': degree_distribution,
        'num_connected_components': num_components,
        'connected_component_sizes_desc': component_sizes,
        'largest_component_size': largest_component_size,
        'largest_component_fraction': largest_component_fraction,
        'num_isolated_nodes': int(len(isolated_indices)),
        'isolated_symbols': isolated_symbols,
        'largest_component_radius': lcc_radius,
        'largest_component_diameter': lcc_diameter,
        'largest_component_avg_shortest_path': lcc_avg_shortest_path,
    }

    print(f"Nodes: {graph_metadata['num_nodes']}")
    print(f"Undirected edges: {graph_metadata['num_edges_undirected']}")
    print(f"Connected components: {graph_metadata['num_connected_components']}")
    print(f"Degree min/max/mean: {graph_metadata['degree_stats']['min']}/{graph_metadata['degree_stats']['max']}/{graph_metadata['degree_stats']['mean']:.2f}")
    print(f"Avg weighted degree: {graph_metadata['weighted_degree_stats']['mean']:.4f}")

    return graph_metadata


def filter_stocks_csv(stocks_df: pd.DataFrame, valid_symbols: list) -> pd.DataFrame:
    """
    Filter stocks.csv to only include valid symbols in correct order.
    """
    print("\n" + "="*80)
    print("Filtering stocks.csv")
    print("="*80)
    
    print(f"Original stocks: {len(stocks_df)}")
    print(f"Valid symbols: {len(valid_symbols)}")
    
    # Filter to valid symbols
    filtered = stocks_df[stocks_df['Symbol'].isin(valid_symbols)].copy()
    
    # Reorder to match valid_symbols order (important for graph node indexing)
    filtered['_order'] = filtered['Symbol'].map({s: i for i, s in enumerate(valid_symbols)})
    filtered = filtered.sort_values('_order').drop(columns=['_order'])
    
    print(f"Filtered stocks: {len(filtered)}")
    
    return filtered


def filter_and_renormalize_fundamentals(fund_df: pd.DataFrame, valid_symbols: list) -> pd.DataFrame:
    """
    Filter fundamentals.csv to valid symbols and renormalize.
    """
    print("\n" + "="*80)
    print("Filtering and Renormalizing fundamentals.csv")
    print("="*80)
    
    print(f"Original fundamentals: {len(fund_df)}")
    print(f"Valid symbols: {len(valid_symbols)}")
    
    # Filter to valid symbols
    filtered = fund_df[fund_df['Symbol'].isin(valid_symbols)].copy()
    
    # Reorder to match valid_symbols order
    filtered['_order'] = filtered['Symbol'].map({s: i for i, s in enumerate(valid_symbols)})
    filtered = filtered.sort_values('_order').drop(columns=['_order'])
    
    print(f"Filtered fundamentals: {len(filtered)}")
    
    # Renormalize numeric columns (z-score normalization)
    numeric_cols = filtered.select_dtypes(include=[np.number]).columns
    
    print(f"\nRenormalizing {len(numeric_cols)} numeric columns...")
    for col in numeric_cols:
        mean_val = filtered[col].mean()
        std_val = filtered[col].std()
        if std_val > 1e-8:  # Avoid division by zero
            filtered[col] = (filtered[col] - mean_val) / std_val
        else:
            filtered[col] = 0
    
    # Fill NaN with 0
    filtered = filtered.fillna(0)
    
    print("✓ Renormalization complete")
    
    return filtered


def visualize_adjacency_matrices(corr_matrix: np.ndarray, sector_matrix: np.ndarray, 
                                 combined_matrix: np.ndarray, stocks_df: pd.DataFrame,
                                 stock_order: list, output_path: Path,
                                 visualization_threshold: float, sector_bonus: float):
    """
    Visualize fundamentals-correlation, sector, and combined adjacency matrices as networkx graphs.
    
    Args:
        corr_matrix: Fundamentals-correlation adjacency (absolute correlations)
        sector_matrix: Sector-based binary adjacency (0 or 1)
        combined_matrix: Combined weighted adjacency
        stocks_df: Dataframe with stock sector information
        stock_order: Ordered list of stock symbols
        output_path: Path to save the figure
        visualization_threshold: Threshold used only for drawing edges in plots
        sector_bonus: Sector bonus weight
    """
    print("\n" + "="*80)
    print("Visualizing Adjacency Matrices")
    print("="*80)
    
    # Create sector color mapping
    symbol_to_sector = dict(zip(stocks_df['Symbol'], stocks_df['Sector']))
    unique_sectors = sorted(set(symbol_to_sector.values()))
    sector_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_sectors)))
    sector_to_color = dict(zip(unique_sectors, sector_colors))
    
    # Sort stocks by sector to group them together on the circle
    stocks_with_sectors = [(symbol, symbol_to_sector.get(symbol, 'Unknown')) for symbol in stock_order]
    sorted_stocks_with_sectors = sorted(stocks_with_sectors, key=lambda x: (x[1], x[0]))  # Sort by sector, then by symbol
    sorted_stock_order = [s[0] for s in sorted_stocks_with_sectors]
    
    # Create mapping from original index to sorted index
    original_to_sorted_idx = {stock_order.index(s): sorted_stock_order.index(s) for s in stock_order}
    
    node_colors = [sector_to_color.get(symbol_to_sector.get(s, 'Unknown'), [0.7, 0.7, 0.7, 1.0]) 
                   for s in sorted_stock_order]
    
    # Calculate dynamic figure size based on number of nodes
    # Scale figure size to ensure nodes are well-spaced
    n_nodes = len(stock_order)
    base_width = 10
    width_per_node = 0.15  # Additional width per node
    total_width = max(20, base_width + n_nodes * width_per_node)
    fig_height = max(8, total_width / 3)  # Maintain aspect ratio
    
    print(f"Visualizing full graph with {n_nodes} stocks")
    print(f"Figure size: {total_width:.1f} x {fig_height:.1f} inches")
    print(f"Stocks grouped by {len(unique_sectors)} sectors on circle")
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(total_width, fig_height))
    gs = GridSpec(1, 3, figure=fig, wspace=0.3)
    
    # Use full graph (no subsetting)
    subset_stocks = sorted_stock_order
    subset_corr = corr_matrix
    subset_sector = sector_matrix
    subset_combined = combined_matrix
    subset_colors = node_colors
    
    # 1. Correlation graph (threshold-based edges for visualization only)
    ax1 = fig.add_subplot(gs[0])
    G_corr = nx.Graph()
    G_corr.add_nodes_from(range(len(subset_stocks)))
    
    # Add edges using mapping from original to sorted indices
    for i in range(len(stock_order)):
        for j in range(i+1, len(stock_order)):
            if subset_corr[i, j] >= visualization_threshold:
                # Map original indices to sorted circular layout indices
                sorted_i = original_to_sorted_idx[i]
                sorted_j = original_to_sorted_idx[j]
                G_corr.add_edge(sorted_i, sorted_j, weight=subset_corr[i, j])
    
    # Calculate node size based on number of nodes
    n_nodes = len(subset_stocks)
    node_size = max(50, min(300, 15000 / n_nodes))  # Scale node size inversely with number of nodes
    
    # Use circular layout with sector-grouped order
    pos = nx.circular_layout(G_corr)
    nx.draw_networkx_nodes(G_corr, pos, node_color=subset_colors, 
                          node_size=node_size, alpha=0.8, ax=ax1)
    edges = G_corr.edges()
    if edges:
        weights = [G_corr[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G_corr, pos, alpha=0.3, width=1.0, ax=ax1)
    ax1.set_title(f'Fundamentals Corr Graph\n(|corr| >= {visualization_threshold}, {G_corr.number_of_edges()} edges)', 
                 fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # 2. Sector graph
    ax2 = fig.add_subplot(gs[1])
    G_sector = nx.Graph()
    G_sector.add_nodes_from(range(len(subset_stocks)))
    
    # Add edges using mapping from original to sorted indices
    for i in range(len(stock_order)):
        for j in range(i+1, len(stock_order)):
            if subset_sector[i, j] > 0:
                sorted_i = original_to_sorted_idx[i]
                sorted_j = original_to_sorted_idx[j]
                G_sector.add_edge(sorted_i, sorted_j)
    
    nx.draw_networkx_nodes(G_sector, pos, node_color=subset_colors, 
                          node_size=node_size, alpha=0.8, ax=ax2)
    if G_sector.number_of_edges() > 0:
        nx.draw_networkx_edges(G_sector, pos, alpha=0.3, width=1.0, ax=ax2)
    ax2.set_title(f'Sector Graph\n(same sector, {G_sector.number_of_edges()} edges)', 
                 fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    # 3. Combined weighted graph
    ax3 = fig.add_subplot(gs[2])
    G_combined = nx.Graph()
    G_combined.add_nodes_from(range(len(subset_stocks)))
    
    # Add edges using mapping from original to sorted indices
    for i in range(len(stock_order)):
        for j in range(i+1, len(stock_order)):
            if subset_combined[i, j] >= visualization_threshold:
                sorted_i = original_to_sorted_idx[i]
                sorted_j = original_to_sorted_idx[j]
                G_combined.add_edge(sorted_i, sorted_j, weight=subset_combined[i, j])
    
    nx.draw_networkx_nodes(G_combined, pos, node_color=subset_colors, 
                          node_size=node_size, alpha=0.8, ax=ax3)
    edges = G_combined.edges()
    if edges:
        weights = [G_combined[u][v]['weight'] for u, v in edges]
        max_weight = max(weights) if weights else 1.0
        edge_widths = [2.0 * w / max_weight for w in weights]
        nx.draw_networkx_edges(G_combined, pos, alpha=0.4, width=edge_widths, ax=ax3)
    ax3.set_title(f'Combined Graph\n(|fund_corr| + {sector_bonus}xsector, {G_combined.number_of_edges()} edges)', 
                 fontsize=12, fontweight='bold')
    ax3.axis('off')
    
    # Add legend for top sectors
    top_sectors = list(unique_sectors)[:10]
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor=sector_to_color[s], markersize=8, label=s)
                      for s in top_sectors]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, 
              bbox_to_anchor=(0.5, -0.05), frameon=False)
    
    plt.suptitle('SP500 Stock Network Visualization', fontsize=14, fontweight='bold', y=1.02)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.close()
    
    print(f"✓ Saved visualization to {output_path}")
    print(f"  Correlation edges: {G_corr.number_of_edges()}")
    print(f"  Sector edges: {G_sector.number_of_edges()}")
    print(f"  Combined edges: {G_combined.number_of_edges()}")


def main():
    parser = argparse.ArgumentParser(
        description="Clean SP500 data and regenerate all derived files with fundamentals-based static graph"
    )
    parser.add_argument(
        '--method',
        type=str,
        choices=['common_range', 'drop_incomplete', 'forward_fill'],
        default='drop_incomplete',
        help='Cleaning method to use'
    )
    parser.add_argument(
        '--min-coverage',
        type=float,
        default=0.95,
        help='Minimum coverage threshold for drop_incomplete method (default: 0.95)'
    )
    parser.add_argument(
        '--edge-weight-threshold',
        type=float,
        default=0.0,
        help='Optional threshold for pruning weighted adjacency edges. '
             'Edges with weight < threshold are set to 0. Use 0 to disable (default: 0.0).'
    )
    parser.add_argument(
        '--correlation-threshold',
        dest='correlation_threshold_legacy',
        type=float,
        default=None,
        help=argparse.SUPPRESS
    )
    parser.add_argument(
        '--viz-threshold',
        type=float,
        default=0.7,
        help='Threshold used only for adjacency visualization plots (default: 0.7)'
    )
    parser.add_argument(
        '--sector-bonus',
        type=float,
        default=0.0,
        help='Weight bonus added for stocks in same sector (default: 0.0). '
             'Final adjacency = |fundamentals_correlations| + sector_bonus * sector_adjacency'
    )
    parser.add_argument(
        '--corr-method',
        type=str,
        choices=['spearman', 'pearson'],
        default='spearman',
        help='Fundamentals correlation method (default: spearman)'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='data/sp500/raw',
        help='Input directory containing raw data'
    )
    
    args = parser.parse_args()

    if args.correlation_threshold_legacy is not None:
        print("Warning: --correlation-threshold is deprecated. Use --edge-weight-threshold instead.")
        args.edge_weight_threshold = float(args.correlation_threshold_legacy)

    if args.edge_weight_threshold < 0:
        raise ValueError(f"--edge-weight-threshold must be >= 0, got {args.edge_weight_threshold}")
    if args.viz_threshold < 0:
        raise ValueError(f"--viz-threshold must be >= 0, got {args.viz_threshold}")
    if args.sector_bonus < 0:
        raise ValueError(f"--sector-bonus must be >= 0, got {args.sector_bonus}")
    
    # Set up paths
    input_dir = Path(args.input_dir)
    
    # Create output directory name
    if args.method == 'drop_incomplete':
        output_dir_name = f"cleaned_{args.method}_min_coverage_{args.min_coverage}"
    else:
        output_dir_name = f"cleaned_{args.method}"
    
    # Append edge-weight threshold and sector bonus
    if args.edge_weight_threshold > 0:
        output_dir_name += f"_corr_{args.edge_weight_threshold}"
    else:
        output_dir_name += "_corr_none"
    if args.sector_bonus > 0:
        output_dir_name += f"_sector_bonus_{args.sector_bonus}"

    output_dir_name = output_dir_name + "/raw" # to match original structure
    
    output_dir = input_dir.parent / output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("SP500 Data Cleaning and Reprocessing")
    print("="*80)
    print(f"Method: {args.method}")
    if args.method == 'drop_incomplete':
        print(f"Min coverage: {args.min_coverage}")
    if args.edge_weight_threshold > 0:
        print(f"Edge-weight threshold: {args.edge_weight_threshold}")
    else:
        print("Edge-weight threshold: disabled")
    print(f"Visualization threshold: {args.viz_threshold}")
    print(f"Correlation method: {args.corr_method}")
    print(f"Sector bonus: {args.sector_bonus}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load raw data
    print("\n" + "="*80)
    print("Loading Raw Data")
    print("="*80)
    
    values_path = input_dir / 'values.csv'
    stocks_path = input_dir / 'stocks.csv'
    fundamentals_path = input_dir / 'fundamentals.csv'
    
    if not values_path.exists():
        print(f"Error: {values_path} not found!")
        return
    
    df_values = pd.read_csv(values_path)
    print(f"Loaded values.csv: {len(df_values)} rows, {df_values['Symbol'].nunique()} stocks")
    
    # Load metadata files
    df_stocks = None
    df_fundamentals = None
    
    if stocks_path.exists():
        df_stocks = pd.read_csv(stocks_path)
        print(f"Loaded stocks.csv: {len(df_stocks)} stocks")
    else:
        print(f"Warning: {stocks_path} not found, will skip stocks.csv processing")
    
    if fundamentals_path.exists():
        df_fundamentals = pd.read_csv(fundamentals_path)
        print(f"Loaded fundamentals.csv: {len(df_fundamentals)} stocks")
    else:
        raise FileNotFoundError(
            f"{fundamentals_path} not found. Fundamentals are required to build static adjacency."
        )
    
    # Step 1: Clean values.csv (defines valid stock universe and order)
    if args.method == 'common_range':
        df_clean = clean_values_common_range(df_values)
    elif args.method == 'drop_incomplete':
        df_clean = clean_values_drop_incomplete(df_values, args.min_coverage)
    elif args.method == 'forward_fill':
        df_clean = clean_values_forward_fill(df_values)
    else:
        raise ValueError(f"Unknown method: {args.method}")
    
    # Get final stock list (in order from cleaned values)
    final_symbols = df_clean['Symbol'].unique().tolist()

    # Step 2: Filter stocks.csv
    df_stocks_clean = None
    if df_stocks is not None:
        df_stocks_clean = filter_stocks_csv(df_stocks, final_symbols)

    # Step 3: Filter and renormalize fundamentals.csv
    df_fundamentals_clean = filter_and_renormalize_fundamentals(df_fundamentals, final_symbols)

    # Step 4: Compute weighted adjacency from cleaned fundamentals + optional sector bonus
    adj_matrix, corr_only, sector_only = compute_fundamentals_adjacency(
        fundamentals_df=df_fundamentals_clean,
        stocks_df=df_stocks_clean,
        stock_order=final_symbols,
        edge_weight_threshold=args.edge_weight_threshold,
        sector_bonus=args.sector_bonus,
        corr_method=args.corr_method,
    )

    # Step 4.5: Compute graph diagnostics metadata
    graph_metadata = compute_graph_metadata(adj_matrix, final_symbols)

    # Step 5: Save all outputs
    print("\n" + "="*80)
    print("Saving Cleaned Data")
    print("="*80)
    
    # Save values.csv
    values_out = output_dir / 'values.csv'
    df_clean.to_csv(values_out, index=False)
    print(f"✓ Saved values.csv: {len(df_clean)} rows, {len(final_symbols)} stocks")
    
    # Save adjacency matrix
    adj_out = output_dir / 'adj.npy'
    np.save(adj_out, adj_matrix)
    print(f"✓ Saved adj.npy: {adj_matrix.shape}")
    
    # Save stocks.csv
    if df_stocks is not None:
        stocks_out = output_dir / 'stocks.csv'
        df_stocks_clean.to_csv(stocks_out, index=False)
        print(f"✓ Saved stocks.csv: {len(df_stocks_clean)} stocks")
    
    # Save fundamentals.csv
    fundamentals_out = output_dir / 'fundamentals.csv'
    df_fundamentals_clean.to_csv(fundamentals_out, index=False)
    print(f"✓ Saved fundamentals.csv: {len(df_fundamentals_clean)} stocks")

    # Step 6: Visualize adjacency matrices
    if df_stocks_clean is not None:
        viz_path = output_dir / 'adjacency_visualization.pdf'
        visualize_adjacency_matrices(
            corr_matrix=corr_only,
            sector_matrix=sector_only,
            combined_matrix=adj_matrix,
            stocks_df=df_stocks_clean,
            stock_order=final_symbols,
            output_path=viz_path,
            visualization_threshold=args.viz_threshold,
            sector_bonus=args.sector_bonus
        )
    
    # Save metadata
    metadata = {
        'method': args.method,
        'min_coverage': args.min_coverage if args.method == 'drop_incomplete' else None,
        'edge_weight_threshold': float(args.edge_weight_threshold),
        'correlation_threshold': float(args.edge_weight_threshold),  # Backward-compatible key
        'corr_method': args.corr_method,
        'sector_bonus': args.sector_bonus,
        'adjacency_formula': f'|fundamentals_correlations| + {args.sector_bonus} * sector_adjacency',
        'adjacency_thresholding_enabled': bool(args.edge_weight_threshold > 0),
        'visualization_threshold': float(args.viz_threshold),
        'num_stocks': len(final_symbols),
        'num_dates': df_clean['Date'].nunique(),
        'num_nonzero_edges': int(np.count_nonzero(adj_matrix)),
        'nonzero_edge_density': float(np.count_nonzero(adj_matrix) / (adj_matrix.shape[0] * adj_matrix.shape[1])),
        'graph_metadata_file': 'graph_metadata.json',
        'num_connected_components': graph_metadata['num_connected_components'],
        'largest_component_fraction': graph_metadata['largest_component_fraction'],
        'min_degree': graph_metadata['degree_stats']['min'],
        'max_degree': graph_metadata['degree_stats']['max'],
        'avg_weighted_degree': graph_metadata['weighted_degree_stats']['mean'],
        'num_edges_above_threshold': (
            int(np.count_nonzero(adj_matrix >= args.edge_weight_threshold))
            if args.edge_weight_threshold > 0 else None
        ),
        'edge_density_above_threshold': (
            float(np.count_nonzero(adj_matrix >= args.edge_weight_threshold) / (adj_matrix.shape[0] * adj_matrix.shape[1]))
            if args.edge_weight_threshold > 0 else None
        ),
        'adj_matrix_stats': {
            'min': float(adj_matrix.min()),
            'max': float(adj_matrix.max()),
            'mean': float(adj_matrix.mean()),
            'std': float(adj_matrix.std()),
        },
        'date_range': [df_clean['Date'].min(), df_clean['Date'].max()],
        'timestamp': datetime.now().isoformat(),
    }
    
    import json
    metadata_out = output_dir / 'metadata.json'
    with open(metadata_out, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata.json")

    graph_metadata_out = output_dir / 'graph_metadata.json'
    with open(graph_metadata_out, 'w') as f:
        json.dump(graph_metadata, f, indent=2)
    print(f"✓ Saved graph_metadata.json")
    
    # Print summary
    print("\n" + "="*80)
    print("Cleaning Complete!")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print(f"Files created:")
    print(f"  - values.csv: {len(df_clean)} rows")
    print(f"  - adj.npy: {adj_matrix.shape}")
    if df_stocks_clean is not None:
        print(f"  - stocks.csv: {len(df_stocks_clean)} rows")
    print(f"  - fundamentals.csv: {len(df_fundamentals_clean)} rows")
    print(f"  - metadata.json")
    print(f"  - graph_metadata.json")
    
    print(f"\nDataset summary:")
    print(f"  Stocks: {len(final_symbols)}")
    print(f"  Dates: {df_clean['Date'].nunique()}")
    print(f"  Date range: {df_clean['Date'].min()} to {df_clean['Date'].max()}")
    print(f"  Adjacency formula: {metadata['adjacency_formula']}")
    print(f"  Adjacency stats: min={metadata['adj_matrix_stats']['min']:.4f}, max={metadata['adj_matrix_stats']['max']:.4f}, mean={metadata['adj_matrix_stats']['mean']:.4f}")
    print(f"  Nonzero edges: {metadata['num_nonzero_edges']}")
    print(f"  Nonzero edge density: {metadata['nonzero_edge_density']:.4f}")
    print(f"  Connected components: {graph_metadata['num_connected_components']}")
    print(f"  Degree min/max: {graph_metadata['degree_stats']['min']}/{graph_metadata['degree_stats']['max']}")
    print(f"  Avg weighted degree: {graph_metadata['weighted_degree_stats']['mean']:.4f}")
    if args.edge_weight_threshold > 0:
        print(f"  Edges above threshold {args.edge_weight_threshold}: {metadata['num_edges_above_threshold']}")
        print(f"  Edge density (threshold {args.edge_weight_threshold}): {metadata['edge_density_above_threshold']:.4f}")
    else:
        print("  Edge thresholding disabled")
    
    print(f"\nTo use this cleaned dataset, update your config:")
    print(f"  dataset.data_dir: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()





# """
# Clean SP500 data and regenerate all derived files with consistent dimensions.

# This script:
# 1. Cleans values.csv using chosen method (common_range, drop_incomplete, forward_fill)
# 2. Recomputes fundamentals-based adjacency matrix on cleaned data
# 3. Filters stocks.csv to match cleaned stock list
# 4. Filters and renormalizes fundamentals.csv
# 5. Computes graph diagnostics (degree stats/distribution, connectivity)
# 6. Saves all outputs to a parameterized cleaned dataset directory under data/sp500/

# Usage:
#     # Basic usage with default parameters
#     python scripts/clean_sp500_data.py --method drop_incomplete --min-coverage 0.95
#     python scripts/clean_sp500_data.py --method common_range
#     python scripts/clean_sp500_data.py --method forward_fill
    
#     # Optional edge-weight thresholding (default: disabled)
#     python scripts/clean_sp500_data.py --method drop_incomplete --min-coverage 0.95 --edge-weight-threshold 0.7
    
#     # With sector bonus weighting (default: 0.0)
#     python scripts/clean_sp500_data.py --method drop_incomplete --min-coverage 0.95 --sector-bonus 0.2
    
#     # Combined: optional threshold + sector bonus
#     python scripts/clean_sp500_data.py --method drop_incomplete --min-coverage 0.95 --edge-weight-threshold 0.6 --sector-bonus 0.3
# """

# import sys
# from pathlib import Path
# import pandas as pd
# import numpy as np
# import argparse
# from datetime import datetime
# import shutil
# import matplotlib
# matplotlib.use('Agg')  # Non-interactive backend
# import matplotlib.pyplot as plt

# # Disable LaTeX text rendering to avoid missing font errors
# matplotlib.rcParams['text.usetex'] = False

# import networkx as nx
# from matplotlib.gridspec import GridSpec

# # Add project root to path
# project_root = Path(__file__).parent.parent
# sys.path.insert(0, str(project_root))


# def clean_values_common_range(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Method 1: Keep all stocks, use only dates common to ALL stocks.
#     """
#     print("\n" + "="*80)
#     print("Method: Common Date Range")
#     print("="*80)
    
#     # Get date coverage per stock
#     date_counts = df.groupby('Symbol')['Date'].nunique()
#     all_dates = df['Date'].unique()
    
#     print(f"Total stocks: {len(date_counts)}")
#     print(f"Total dates: {len(all_dates)}")
    
#     # Find stocks with full coverage
#     max_dates = date_counts.max()
#     stocks_with_max = date_counts[date_counts == max_dates].index.tolist()
    
#     if len(stocks_with_max) == len(date_counts):
#         print("✓ All stocks already have same number of dates!")
#         return df
    
#     # Get dates that exist for ALL stocks
#     common_dates = None
#     for symbol in date_counts.index:
#         symbol_dates = set(df[df['Symbol'] == symbol]['Date'].unique())
#         if common_dates is None:
#             common_dates = symbol_dates
#         else:
#             common_dates = common_dates.intersection(symbol_dates)
    
#     common_dates = sorted(list(common_dates))
    
#     print(f"\nCommon dates: {len(common_dates)} ({len(common_dates)/len(all_dates)*100:.1f}% retention)")
#     print(f"Date range: {common_dates[0]} to {common_dates[-1]}")
    
#     # Filter to common dates
#     df_clean = df[df['Date'].isin(common_dates)].copy()
    
#     # Verify
#     verify_counts = df_clean.groupby('Symbol')['Date'].nunique()
#     assert verify_counts.nunique() == 1, "Stocks still have different date counts!"
    
#     print(f"\n✓ Cleaned dataset: {len(verify_counts)} stocks × {verify_counts.iloc[0]} dates")
    
#     return df_clean


# def clean_values_drop_incomplete(df: pd.DataFrame, min_coverage: float = 0.90) -> pd.DataFrame:
#     """
#     Method 2: Drop stocks below coverage threshold, then use common range.
#     """
#     print("\n" + "="*80)
#     print(f"Method: Drop Incomplete Stocks (min_coverage={min_coverage})")
#     print("="*80)
    
#     # Get date coverage per stock
#     all_dates = df['Date'].unique()
#     date_counts = df.groupby('Symbol')['Date'].nunique()
#     max_dates = len(all_dates)
    
#     print(f"Total stocks: {len(date_counts)}")
#     print(f"Total dates: {max_dates}")
    
#     # Calculate coverage
#     coverage = date_counts / max_dates
    
#     # Filter stocks
#     min_dates = int(max_dates * min_coverage)
#     stocks_to_keep = coverage[coverage >= min_coverage].index.tolist()
#     stocks_to_drop = coverage[coverage < min_coverage].index.tolist()
    
#     print(f"\nCoverage threshold: {min_coverage*100:.0f}% ({min_dates} dates)")
#     print(f"Stocks to keep: {len(stocks_to_keep)} ({len(stocks_to_keep)/len(date_counts)*100:.1f}%)")
#     print(f"Stocks to drop: {len(stocks_to_drop)}")
    
#     if stocks_to_drop:
#         print("\nDropped stocks:")
#         for symbol in stocks_to_drop:
#             print(f"  - {symbol}: {date_counts[symbol]} dates ({coverage[symbol]*100:.1f}%)")
    
#     # Filter dataset
#     df_filtered = df[df['Symbol'].isin(stocks_to_keep)].copy()
    
#     # Now apply common range on remaining stocks
#     df_clean = clean_values_common_range(df_filtered)
    
#     return df_clean


# def clean_values_forward_fill(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Method 3: Forward fill missing dates for each stock.
#     """
#     print("\n" + "="*80)
#     print("Method: Forward Fill")
#     print("="*80)
    
#     # Get all unique dates
#     all_dates = sorted(df['Date'].unique())
#     all_symbols = df['Symbol'].unique()
    
#     print(f"Total stocks: {len(all_symbols)}")
#     print(f"Total dates: {len(all_dates)}")
#     print(f"Date range: {all_dates[0]} to {all_dates[-1]}")
    
#     # Process each stock
#     cleaned_dfs = []
#     synthetic_count = 0
    
#     for symbol in all_symbols:
#         stock_data = df[df['Symbol'] == symbol].copy()
#         stock_data = stock_data.set_index('Date').sort_index()
        
#         # Reindex to full date range
#         original_count = len(stock_data)
#         stock_data = stock_data.reindex(all_dates)
        
#         # Forward fill numeric columns
#         numeric_cols = stock_data.select_dtypes(include=[np.number]).columns
#         stock_data[numeric_cols] = stock_data[numeric_cols].fillna(method='ffill')
        
#         # Backfill any remaining NaNs at the start
#         stock_data[numeric_cols] = stock_data[numeric_cols].fillna(method='bfill')
        
#         # Restore Symbol column
#         stock_data['Symbol'] = symbol
#         stock_data = stock_data.reset_index()
        
#         synthetic_count += (len(all_dates) - original_count)
#         cleaned_dfs.append(stock_data)
    
#     df_clean = pd.concat(cleaned_dfs, ignore_index=True)
    
#     print(f"\n✓ Filled {synthetic_count} missing date-stock combinations")
#     print(f"  Average per stock: {synthetic_count / len(all_symbols):.1f} dates")
#     print(f"  Synthetic data: {synthetic_count / len(df_clean) * 100:.2f}%")
    
#     # Verify
#     verify_counts = df_clean.groupby('Symbol')['Date'].nunique()
#     assert verify_counts.nunique() == 1, "Stocks still have different date counts!"
    
#     print(f"\n✓ Cleaned dataset: {len(all_symbols)} stocks × {len(all_dates)} dates")
    
#     return df_clean


# def compute_sector_adjacency(stocks_df: pd.DataFrame, stock_order: list) -> np.ndarray:
#     """
#     Compute binary adjacency matrix based on sector membership.
#     Stocks in the same sector get an edge.
#     """
#     print("\n" + "="*80)
#     print("Computing Sector Adjacency Matrix")
#     print("="*80)
    
#     n_stocks = len(stock_order)
#     sector_adj = np.zeros((n_stocks, n_stocks), dtype=float)
    
#     # Create symbol to index mapping
#     symbol_to_idx = {symbol: i for i, symbol in enumerate(stock_order)}
    
#     # Create symbol to sector mapping
#     symbol_to_sector = dict(zip(stocks_df['Symbol'], stocks_df['Sector']))
    
#     # Build sector adjacency
#     for i, symbol_i in enumerate(stock_order):
#         sector_i = symbol_to_sector.get(symbol_i, None)
#         if sector_i is None:
#             continue
        
#         for j, symbol_j in enumerate(stock_order):
#             if i == j:
#                 continue
#             sector_j = symbol_to_sector.get(symbol_j, None)
#             if sector_j is None:
#                 continue
            
#             # Same sector -> edge
#             if sector_i == sector_j:
#                 sector_adj[i, j] = 1.0
    
#     # Count sectors
#     sectors = [symbol_to_sector.get(s, 'Unknown') for s in stock_order]
#     sector_counts = pd.Series(sectors).value_counts()
    
#     print(f"Sectors found: {len(sector_counts)}")
#     print(f"Number of sector edges: {np.count_nonzero(sector_adj)}")
#     print(f"Sector edge density: {np.count_nonzero(sector_adj) / (n_stocks * n_stocks):.4f}")
    
#     print("\nSector distribution:")
#     for sector, count in sector_counts.head(10).items():
#         print(f"  {sector}: {count} stocks")
    
#     return sector_adj


# def compute_fundamentals_adjacency(
#     fundamentals_df: pd.DataFrame,
#     stocks_df: pd.DataFrame,
#     stock_order: list,
#     edge_weight_threshold: float = 0.0,
#     sector_bonus: float = 0.0,
#     corr_method: str = 'spearman',
# ) -> tuple:
#     """
#     Compute weighted adjacency matrix from fundamentals and sector information.
    
#     Final adjacency (SP100 notebook style):
#         adj = abs(corr(fundamentals)) + sector_bonus * sector_adjacency
#         adj = adj * (adj >= edge_weight_threshold)   # optional
#         adj = adj / adj.max()                        # if max > 0
    
#     Args:
#         fundamentals_df: Fundamentals dataframe (must include Symbol)
#         stocks_df: Stocks dataframe with Sector column
#         stock_order: Ordered list of stock symbols
#         edge_weight_threshold: Optional post-processing threshold.
#             If > 0, edges with weights below this value are zeroed.
#             If <= 0, thresholding is disabled.
#         sector_bonus: Weight bonus for stocks in the same sector
#         corr_method: Correlation method ('spearman' or 'pearson')
    
#     Returns:
#         adj_matrix: Final weighted adjacency matrix
#         corr_np: Absolute fundamentals correlation matrix (diag=0)
#         sector_adj: Binary sector adjacency matrix (diag=0)
#     """
#     print("\n" + "="*80)
#     print("Computing Fundamentals-Based Weighted Adjacency Matrix")
#     print(f"  Correlation method: {corr_method}")
#     if edge_weight_threshold > 0:
#         print(f"  Edge-weight threshold: {edge_weight_threshold}")
#     else:
#         print("  Edge-weight threshold: disabled")
#     print(f"  Sector bonus: {sector_bonus}")
#     print("="*80)

#     fundamentals = fundamentals_df.set_index('Symbol').reindex(stock_order)
#     numeric_cols = fundamentals.select_dtypes(include=[np.number]).columns.tolist()
#     if not numeric_cols:
#         raise ValueError("No numeric fundamentals columns found to build adjacency.")

#     missing_rows = int(fundamentals[numeric_cols].isna().all(axis=1).sum())
#     if missing_rows > 0:
#         print(f"Warning: {missing_rows} symbols have missing/empty fundamentals rows; filling with 0.")

#     fundamentals_numeric = fundamentals[numeric_cols].fillna(0.0)
#     print(f"Fundamentals matrix: {fundamentals_numeric.shape[0]} stocks × {fundamentals_numeric.shape[1]} features")

#     # Match notebook approach: correlation across stocks from all fundamentals.
#     corr_matrix = fundamentals_numeric.transpose().corr(method=corr_method).fillna(0.0)
#     corr_np = np.abs(corr_matrix.to_numpy(dtype=float))
#     np.fill_diagonal(corr_np, 0.0)

#     print(f"\nFundamentals correlation matrix: {corr_np.shape[0]} × {corr_np.shape[1]}")
#     print(f"Correlation stats: min={corr_np.min():.4f}, max={corr_np.max():.4f}, mean={corr_np.mean():.4f}")

#     # Always compute sector adjacency for diagnostics/visualization when stocks metadata exists.
#     if stocks_df is not None:
#         sector_adj = compute_sector_adjacency(stocks_df, stock_order)
#     else:
#         sector_adj = np.zeros_like(corr_np)
#         if sector_bonus > 0:
#             print("Warning: sector_bonus > 0 but stocks.csv is missing. Sector bonus is skipped.")

#     adj_matrix = corr_np.copy()
#     if sector_bonus > 0:
#         adj_matrix = adj_matrix + sector_bonus * sector_adj

#     print(f"\nCombined adjacency:")
#     print(f"  Formula: adj = |fundamentals_corr| + {sector_bonus} × sector_adjacency")
#     print(f"  Value range (pre-threshold): [{adj_matrix.min():.4f}, {adj_matrix.max():.4f}]")
#     print(f"  Mean weight (pre-threshold): {adj_matrix.mean():.4f}")
    
#     # Optionally apply thresholding.
#     nonzero_before = np.count_nonzero(adj_matrix)
#     if edge_weight_threshold > 0:
#         print(f"\nApplying edge-weight threshold {edge_weight_threshold}:")
#         print(f"  Before: {nonzero_before} nonzero edges")
#         adj_matrix = adj_matrix * (adj_matrix >= edge_weight_threshold)
#         nonzero_after = np.count_nonzero(adj_matrix)
#         print(f"  After:  {nonzero_after} nonzero edges")
#         print(f"  Edge density: {nonzero_after / (adj_matrix.shape[0] * adj_matrix.shape[1]):.4f}")
#         print(f"  Sparsification: {(1 - nonzero_after / (adj_matrix.shape[0] * adj_matrix.shape[1])) * 100:.1f}% of edges removed")
#     else:
#         print("\nEdge-weight thresholding disabled (keeping all positive weighted edges).")
#         print(f"  Nonzero edges: {nonzero_before}")
#         print(f"  Edge density: {nonzero_before / (adj_matrix.shape[0] * adj_matrix.shape[1]):.4f}")

#     # Remove self loops and max-normalize, matching notebook behavior.
#     np.fill_diagonal(adj_matrix, 0.0)
#     max_val = float(adj_matrix.max()) if adj_matrix.size > 0 else 0.0
#     if max_val > 0:
#         adj_matrix = adj_matrix / max_val

#     print(f"\nFinal adjacency stats (post-normalization): min={adj_matrix.min():.4f}, max={adj_matrix.max():.4f}, mean={adj_matrix.mean():.4f}")

#     return adj_matrix, corr_np, sector_adj


# def compute_graph_metadata(adj_matrix: np.ndarray, stock_order: list) -> dict:
#     """
#     Compute graph-level diagnostics from a weighted adjacency matrix.

#     Connectivity metrics are computed on an unweighted graph where edge exists
#     when adjacency weight > 0. Degree stats are reported for both unweighted
#     and weighted degrees.
#     """
#     print("\n" + "="*80)
#     print("Computing Graph Metadata")
#     print("="*80)

#     if adj_matrix.ndim != 2 or adj_matrix.shape[0] != adj_matrix.shape[1]:
#         raise ValueError(f"adj_matrix must be square, got shape {adj_matrix.shape}")

#     n = int(adj_matrix.shape[0])
#     adj = np.array(adj_matrix, dtype=float, copy=True)
#     np.fill_diagonal(adj, 0.0)

#     binary_adj = (adj > 0).astype(np.int8)
#     degrees = binary_adj.sum(axis=1).astype(int)
#     weighted_degrees = adj.sum(axis=1).astype(float)

#     unique_deg, counts_deg = np.unique(degrees, return_counts=True)
#     degree_distribution = {
#         str(int(k)): int(v) for k, v in zip(unique_deg.tolist(), counts_deg.tolist())
#     }

#     undirected_edges = int(np.count_nonzero(np.triu(binary_adj, k=1)))
#     directed_nonzero_edges = int(np.count_nonzero(binary_adj))
#     possible_undirected_edges = (n * (n - 1)) // 2 if n > 1 else 0
#     possible_directed_edges = n * (n - 1) if n > 1 else 0

#     G = nx.from_numpy_array(binary_adj)
#     components = list(nx.connected_components(G))
#     component_sizes = sorted([int(len(c)) for c in components], reverse=True)
#     num_components = int(len(component_sizes))
#     largest_component_size = int(component_sizes[0]) if component_sizes else 0
#     largest_component_fraction = float(largest_component_size / n) if n > 0 else 0.0

#     isolated_indices = np.where(degrees == 0)[0].tolist()
#     isolated_symbols = [stock_order[i] for i in isolated_indices]

#     lcc_radius = None
#     lcc_diameter = None
#     lcc_avg_shortest_path = None
#     if components:
#         largest_component_nodes = max(components, key=len)
#         G_lcc = G.subgraph(largest_component_nodes).copy()
#         if G_lcc.number_of_nodes() == 1:
#             lcc_radius = 0
#             lcc_diameter = 0
#             lcc_avg_shortest_path = 0.0
#         elif G_lcc.number_of_nodes() > 1:
#             lcc_radius = int(nx.radius(G_lcc))
#             lcc_diameter = int(nx.diameter(G_lcc))
#             lcc_avg_shortest_path = float(nx.average_shortest_path_length(G_lcc))

#     graph_metadata = {
#         'num_nodes': n,
#         'num_edges_undirected': undirected_edges,
#         'num_edges_directed_nonzero': directed_nonzero_edges,
#         'edge_density_undirected': (
#             float(undirected_edges / possible_undirected_edges)
#             if possible_undirected_edges > 0 else 0.0
#         ),
#         'edge_density_directed': (
#             float(directed_nonzero_edges / possible_directed_edges)
#             if possible_directed_edges > 0 else 0.0
#         ),
#         'degree_stats': {
#             'min': int(degrees.min()) if n > 0 else 0,
#             'max': int(degrees.max()) if n > 0 else 0,
#             'mean': float(degrees.mean()) if n > 0 else 0.0,
#             'median': float(np.median(degrees)) if n > 0 else 0.0,
#             'std': float(degrees.std()) if n > 0 else 0.0,
#         },
#         'weighted_degree_stats': {
#             'min': float(weighted_degrees.min()) if n > 0 else 0.0,
#             'max': float(weighted_degrees.max()) if n > 0 else 0.0,
#             'mean': float(weighted_degrees.mean()) if n > 0 else 0.0,
#             'median': float(np.median(weighted_degrees)) if n > 0 else 0.0,
#             'std': float(weighted_degrees.std()) if n > 0 else 0.0,
#         },
#         'degree_distribution': degree_distribution,
#         'num_connected_components': num_components,
#         'connected_component_sizes_desc': component_sizes,
#         'largest_component_size': largest_component_size,
#         'largest_component_fraction': largest_component_fraction,
#         'num_isolated_nodes': int(len(isolated_indices)),
#         'isolated_symbols': isolated_symbols,
#         'largest_component_radius': lcc_radius,
#         'largest_component_diameter': lcc_diameter,
#         'largest_component_avg_shortest_path': lcc_avg_shortest_path,
#     }

#     print(f"Nodes: {graph_metadata['num_nodes']}")
#     print(f"Undirected edges: {graph_metadata['num_edges_undirected']}")
#     print(f"Connected components: {graph_metadata['num_connected_components']}")
#     print(f"Degree min/max/mean: {graph_metadata['degree_stats']['min']}/{graph_metadata['degree_stats']['max']}/{graph_metadata['degree_stats']['mean']:.2f}")
#     print(f"Avg weighted degree: {graph_metadata['weighted_degree_stats']['mean']:.4f}")

#     return graph_metadata


# def filter_stocks_csv(stocks_df: pd.DataFrame, valid_symbols: list) -> pd.DataFrame:
#     """
#     Filter stocks.csv to only include valid symbols in correct order.
#     """
#     print("\n" + "="*80)
#     print("Filtering stocks.csv")
#     print("="*80)
    
#     print(f"Original stocks: {len(stocks_df)}")
#     print(f"Valid symbols: {len(valid_symbols)}")
    
#     # Filter to valid symbols
#     filtered = stocks_df[stocks_df['Symbol'].isin(valid_symbols)].copy()
    
#     # Reorder to match valid_symbols order (important for graph node indexing)
#     filtered['_order'] = filtered['Symbol'].map({s: i for i, s in enumerate(valid_symbols)})
#     filtered = filtered.sort_values('_order').drop(columns=['_order'])
    
#     print(f"Filtered stocks: {len(filtered)}")
    
#     return filtered


# def filter_and_renormalize_fundamentals(fund_df: pd.DataFrame, valid_symbols: list) -> pd.DataFrame:
#     """
#     Filter fundamentals.csv to valid symbols and renormalize.
#     """
#     print("\n" + "="*80)
#     print("Filtering and Renormalizing fundamentals.csv")
#     print("="*80)
    
#     print(f"Original fundamentals: {len(fund_df)}")
#     print(f"Valid symbols: {len(valid_symbols)}")
    
#     # Filter to valid symbols
#     filtered = fund_df[fund_df['Symbol'].isin(valid_symbols)].copy()
    
#     # Reorder to match valid_symbols order
#     filtered['_order'] = filtered['Symbol'].map({s: i for i, s in enumerate(valid_symbols)})
#     filtered = filtered.sort_values('_order').drop(columns=['_order'])
    
#     print(f"Filtered fundamentals: {len(filtered)}")
    
#     # Renormalize numeric columns (z-score normalization)
#     numeric_cols = filtered.select_dtypes(include=[np.number]).columns
    
#     print(f"\nRenormalizing {len(numeric_cols)} numeric columns...")
#     for col in numeric_cols:
#         mean_val = filtered[col].mean()
#         std_val = filtered[col].std()
#         if std_val > 1e-8:  # Avoid division by zero
#             filtered[col] = (filtered[col] - mean_val) / std_val
#         else:
#             filtered[col] = 0
    
#     # Fill NaN with 0
#     filtered = filtered.fillna(0)
    
#     print("✓ Renormalization complete")
    
#     return filtered


# def visualize_adjacency_matrices(corr_matrix: np.ndarray, sector_matrix: np.ndarray, 
#                                  combined_matrix: np.ndarray, stocks_df: pd.DataFrame,
#                                  stock_order: list, output_path: Path,
#                                  visualization_threshold: float, sector_bonus: float):
#     """
#     Visualize fundamentals-correlation, sector, and combined adjacency matrices as networkx graphs.
    
#     Args:
#         corr_matrix: Fundamentals-correlation adjacency (absolute correlations)
#         sector_matrix: Sector-based binary adjacency (0 or 1)
#         combined_matrix: Combined weighted adjacency
#         stocks_df: Dataframe with stock sector information
#         stock_order: Ordered list of stock symbols
#         output_path: Path to save the figure
#         visualization_threshold: Threshold used only for drawing edges in plots
#         sector_bonus: Sector bonus weight
#     """
#     print("\n" + "="*80)
#     print("Visualizing Adjacency Matrices")
#     print("="*80)
    
#     # Create sector color mapping
#     symbol_to_sector = dict(zip(stocks_df['Symbol'], stocks_df['Sector']))
#     unique_sectors = sorted(set(symbol_to_sector.values()))
#     sector_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_sectors)))
#     sector_to_color = dict(zip(unique_sectors, sector_colors))
    
#     # Sort stocks by sector to group them together on the circle
#     stocks_with_sectors = [(symbol, symbol_to_sector.get(symbol, 'Unknown')) for symbol in stock_order]
#     sorted_stocks_with_sectors = sorted(stocks_with_sectors, key=lambda x: (x[1], x[0]))  # Sort by sector, then by symbol
#     sorted_stock_order = [s[0] for s in sorted_stocks_with_sectors]
    
#     # Create mapping from original index to sorted index
#     original_to_sorted_idx = {stock_order.index(s): sorted_stock_order.index(s) for s in stock_order}
    
#     node_colors = [sector_to_color.get(symbol_to_sector.get(s, 'Unknown'), [0.7, 0.7, 0.7, 1.0]) 
#                    for s in sorted_stock_order]
    
#     # Calculate dynamic figure size based on number of nodes
#     # Scale figure size to ensure nodes are well-spaced
#     n_nodes = len(stock_order)
#     base_width = 10
#     width_per_node = 0.15  # Additional width per node
#     total_width = max(20, base_width + n_nodes * width_per_node)
#     fig_height = max(8, total_width / 3)  # Maintain aspect ratio
    
#     print(f"Visualizing full graph with {n_nodes} stocks")
#     print(f"Figure size: {total_width:.1f} x {fig_height:.1f} inches")
#     print(f"Stocks grouped by {len(unique_sectors)} sectors on circle")
    
#     # Create figure with 3 subplots
#     fig = plt.figure(figsize=(total_width, fig_height))
#     gs = GridSpec(1, 3, figure=fig, wspace=0.3)
    
#     # Use full graph (no subsetting)
#     subset_stocks = sorted_stock_order
#     subset_corr = corr_matrix
#     subset_sector = sector_matrix
#     subset_combined = combined_matrix
#     subset_colors = node_colors
    
#     # 1. Correlation graph (threshold-based edges for visualization only)
#     ax1 = fig.add_subplot(gs[0])
#     G_corr = nx.Graph()
#     G_corr.add_nodes_from(range(len(subset_stocks)))
    
#     # Add edges using mapping from original to sorted indices
#     for i in range(len(stock_order)):
#         for j in range(i+1, len(stock_order)):
#             if subset_corr[i, j] >= visualization_threshold:
#                 # Map original indices to sorted circular layout indices
#                 sorted_i = original_to_sorted_idx[i]
#                 sorted_j = original_to_sorted_idx[j]
#                 G_corr.add_edge(sorted_i, sorted_j, weight=subset_corr[i, j])
    
#     # Calculate node size based on number of nodes
#     n_nodes = len(subset_stocks)
#     node_size = max(50, min(300, 15000 / n_nodes))  # Scale node size inversely with number of nodes
    
#     # Use circular layout with sector-grouped order
#     pos = nx.circular_layout(G_corr)
#     nx.draw_networkx_nodes(G_corr, pos, node_color=subset_colors, 
#                           node_size=node_size, alpha=0.8, ax=ax1)
#     edges = G_corr.edges()
#     if edges:
#         weights = [G_corr[u][v]['weight'] for u, v in edges]
#         nx.draw_networkx_edges(G_corr, pos, alpha=0.3, width=1.0, ax=ax1)
#     ax1.set_title(f'Fundamentals Corr Graph\n(|corr| >= {visualization_threshold}, {G_corr.number_of_edges()} edges)', 
#                  fontsize=12, fontweight='bold')
#     ax1.axis('off')
    
#     # 2. Sector graph
#     ax2 = fig.add_subplot(gs[1])
#     G_sector = nx.Graph()
#     G_sector.add_nodes_from(range(len(subset_stocks)))
    
#     # Add edges using mapping from original to sorted indices
#     for i in range(len(stock_order)):
#         for j in range(i+1, len(stock_order)):
#             if subset_sector[i, j] > 0:
#                 sorted_i = original_to_sorted_idx[i]
#                 sorted_j = original_to_sorted_idx[j]
#                 G_sector.add_edge(sorted_i, sorted_j)
    
#     nx.draw_networkx_nodes(G_sector, pos, node_color=subset_colors, 
#                           node_size=node_size, alpha=0.8, ax=ax2)
#     if G_sector.number_of_edges() > 0:
#         nx.draw_networkx_edges(G_sector, pos, alpha=0.3, width=1.0, ax=ax2)
#     ax2.set_title(f'Sector Graph\n(same sector, {G_sector.number_of_edges()} edges)', 
#                  fontsize=12, fontweight='bold')
#     ax2.axis('off')
    
#     # 3. Combined weighted graph
#     ax3 = fig.add_subplot(gs[2])
#     G_combined = nx.Graph()
#     G_combined.add_nodes_from(range(len(subset_stocks)))
    
#     # Add edges using mapping from original to sorted indices
#     for i in range(len(stock_order)):
#         for j in range(i+1, len(stock_order)):
#             if subset_combined[i, j] >= visualization_threshold:
#                 sorted_i = original_to_sorted_idx[i]
#                 sorted_j = original_to_sorted_idx[j]
#                 G_combined.add_edge(sorted_i, sorted_j, weight=subset_combined[i, j])
    
#     nx.draw_networkx_nodes(G_combined, pos, node_color=subset_colors, 
#                           node_size=node_size, alpha=0.8, ax=ax3)
#     edges = G_combined.edges()
#     if edges:
#         weights = [G_combined[u][v]['weight'] for u, v in edges]
#         max_weight = max(weights) if weights else 1.0
#         edge_widths = [2.0 * w / max_weight for w in weights]
#         nx.draw_networkx_edges(G_combined, pos, alpha=0.4, width=edge_widths, ax=ax3)
#     ax3.set_title(f'Combined Graph\n(|fund_corr| + {sector_bonus}xsector, {G_combined.number_of_edges()} edges)', 
#                  fontsize=12, fontweight='bold')
#     ax3.axis('off')
    
#     # Add legend for top sectors
#     top_sectors = list(unique_sectors)[:10]
#     legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
#                                   markerfacecolor=sector_to_color[s], markersize=8, label=s)
#                       for s in top_sectors]
#     fig.legend(handles=legend_elements, loc='lower center', ncol=5, 
#               bbox_to_anchor=(0.5, -0.05), frameon=False)
    
#     plt.suptitle('SP500 Stock Network Visualization', fontsize=14, fontweight='bold', y=1.02)
#     plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
#     plt.close()
    
#     print(f"✓ Saved visualization to {output_path}")
#     print(f"  Correlation edges: {G_corr.number_of_edges()}")
#     print(f"  Sector edges: {G_sector.number_of_edges()}")
#     print(f"  Combined edges: {G_combined.number_of_edges()}")


# def main():
#     parser = argparse.ArgumentParser(
#         description="Clean SP500 data and regenerate all derived files with fundamentals-based static graph"
#     )
#     parser.add_argument(
#         '--method',
#         type=str,
#         choices=['common_range', 'drop_incomplete', 'forward_fill'],
#         default='drop_incomplete',
#         help='Cleaning method to use'
#     )
#     parser.add_argument(
#         '--min-coverage',
#         type=float,
#         default=0.95,
#         help='Minimum coverage threshold for drop_incomplete method (default: 0.95)'
#     )
#     parser.add_argument(
#         '--edge-weight-threshold',
#         type=float,
#         default=0.0,
#         help='Optional threshold for pruning weighted adjacency edges. '
#              'Edges with weight < threshold are set to 0. Use 0 to disable (default: 0.0).'
#     )
#     parser.add_argument(
#         '--correlation-threshold',
#         dest='correlation_threshold_legacy',
#         type=float,
#         default=None,
#         help=argparse.SUPPRESS
#     )
#     parser.add_argument(
#         '--viz-threshold',
#         type=float,
#         default=0.5,
#         help='Threshold used only for adjacency visualization plots (default: 0.5)'
#     )
#     parser.add_argument(
#         '--sector-bonus',
#         type=float,
#         default=0.0,
#         help='Weight bonus added for stocks in same sector (default: 0.0). '
#              'Final adjacency = |fundamentals_correlations| + sector_bonus * sector_adjacency'
#     )
#     parser.add_argument(
#         '--corr-method',
#         type=str,
#         choices=['spearman', 'pearson'],
#         default='spearman',
#         help='Fundamentals correlation method (default: spearman)'
#     )
#     parser.add_argument(
#         '--input-dir',
#         type=str,
#         default='data/sp500/raw',
#         help='Input directory containing raw data'
#     )
    
#     args = parser.parse_args()

#     if args.correlation_threshold_legacy is not None:
#         print("Warning: --correlation-threshold is deprecated. Use --edge-weight-threshold instead.")
#         args.edge_weight_threshold = float(args.correlation_threshold_legacy)

#     if args.edge_weight_threshold < 0:
#         raise ValueError(f"--edge-weight-threshold must be >= 0, got {args.edge_weight_threshold}")
#     if args.viz_threshold < 0:
#         raise ValueError(f"--viz-threshold must be >= 0, got {args.viz_threshold}")
#     if args.sector_bonus < 0:
#         raise ValueError(f"--sector-bonus must be >= 0, got {args.sector_bonus}")
    
#     # Set up paths
#     input_dir = Path(args.input_dir)
    
#     # Create output directory name
#     if args.method == 'drop_incomplete':
#         output_dir_name = f"cleaned_{args.method}_min_coverage_{args.min_coverage}"
#     else:
#         output_dir_name = f"cleaned_{args.method}"
    
#     # Append edge-weight threshold and sector bonus
#     if args.edge_weight_threshold > 0:
#         output_dir_name += f"_corr_{args.edge_weight_threshold}"
#     else:
#         output_dir_name += "_corr_none"
#     if args.sector_bonus > 0:
#         output_dir_name += f"_sector_bonus_{args.sector_bonus}"

#     output_dir_name = output_dir_name + "/raw" # to match original structure
    
#     output_dir = input_dir.parent / output_dir_name
#     output_dir.mkdir(parents=True, exist_ok=True)
    
#     print("="*80)
#     print("SP500 Data Cleaning and Reprocessing")
#     print("="*80)
#     print(f"Method: {args.method}")
#     if args.method == 'drop_incomplete':
#         print(f"Min coverage: {args.min_coverage}")
#     if args.edge_weight_threshold > 0:
#         print(f"Edge-weight threshold: {args.edge_weight_threshold}")
#     else:
#         print("Edge-weight threshold: disabled")
#     print(f"Visualization threshold: {args.viz_threshold}")
#     print(f"Correlation method: {args.corr_method}")
#     print(f"Sector bonus: {args.sector_bonus}")
#     print(f"Input directory: {input_dir}")
#     print(f"Output directory: {output_dir}")
    
#     # Load raw data
#     print("\n" + "="*80)
#     print("Loading Raw Data")
#     print("="*80)
    
#     values_path = input_dir / 'values.csv'
#     stocks_path = input_dir / 'stocks.csv'
#     fundamentals_path = input_dir / 'fundamentals.csv'
    
#     if not values_path.exists():
#         print(f"Error: {values_path} not found!")
#         return
    
#     df_values = pd.read_csv(values_path)
#     print(f"Loaded values.csv: {len(df_values)} rows, {df_values['Symbol'].nunique()} stocks")
    
#     # Load metadata files
#     df_stocks = None
#     df_fundamentals = None
    
#     if stocks_path.exists():
#         df_stocks = pd.read_csv(stocks_path)
#         print(f"Loaded stocks.csv: {len(df_stocks)} stocks")
#     else:
#         print(f"Warning: {stocks_path} not found, will skip stocks.csv processing")
    
#     if fundamentals_path.exists():
#         df_fundamentals = pd.read_csv(fundamentals_path)
#         print(f"Loaded fundamentals.csv: {len(df_fundamentals)} stocks")
#     else:
#         raise FileNotFoundError(
#             f"{fundamentals_path} not found. Fundamentals are required to build static adjacency."
#         )
    
#     # Step 1: Clean values.csv (defines valid stock universe and order)
#     if args.method == 'common_range':
#         df_clean = clean_values_common_range(df_values)
#     elif args.method == 'drop_incomplete':
#         df_clean = clean_values_drop_incomplete(df_values, args.min_coverage)
#     elif args.method == 'forward_fill':
#         df_clean = clean_values_forward_fill(df_values)
#     else:
#         raise ValueError(f"Unknown method: {args.method}")
    
#     # Get final stock list (in order from cleaned values)
#     final_symbols = df_clean['Symbol'].unique().tolist()

#     # Step 2: Filter stocks.csv
#     df_stocks_clean = None
#     if df_stocks is not None:
#         df_stocks_clean = filter_stocks_csv(df_stocks, final_symbols)

#     # Step 3: Filter and renormalize fundamentals.csv
#     df_fundamentals_clean = filter_and_renormalize_fundamentals(df_fundamentals, final_symbols)

#     # Step 4: Compute weighted adjacency from cleaned fundamentals + optional sector bonus
#     adj_matrix, corr_only, sector_only = compute_fundamentals_adjacency(
#         fundamentals_df=df_fundamentals_clean,
#         stocks_df=df_stocks_clean,
#         stock_order=final_symbols,
#         edge_weight_threshold=args.edge_weight_threshold,
#         sector_bonus=args.sector_bonus,
#         corr_method=args.corr_method,
#     )

#     # Step 4.5: Compute graph diagnostics metadata
#     graph_metadata = compute_graph_metadata(adj_matrix, final_symbols)

#     # Step 5: Save all outputs
#     print("\n" + "="*80)
#     print("Saving Cleaned Data")
#     print("="*80)
    
#     # Save values.csv
#     values_out = output_dir / 'values.csv'
#     df_clean.to_csv(values_out, index=False)
#     print(f"✓ Saved values.csv: {len(df_clean)} rows, {len(final_symbols)} stocks")
    
#     # Save adjacency matrix
#     adj_out = output_dir / 'adj.npy'
#     np.save(adj_out, adj_matrix)
#     print(f"✓ Saved adj.npy: {adj_matrix.shape}")
    
#     # Save stocks.csv
#     if df_stocks is not None:
#         stocks_out = output_dir / 'stocks.csv'
#         df_stocks_clean.to_csv(stocks_out, index=False)
#         print(f"✓ Saved stocks.csv: {len(df_stocks_clean)} stocks")
    
#     # Save fundamentals.csv
#     fundamentals_out = output_dir / 'fundamentals.csv'
#     df_fundamentals_clean.to_csv(fundamentals_out, index=False)
#     print(f"✓ Saved fundamentals.csv: {len(df_fundamentals_clean)} stocks")

#     # Step 6: Visualize adjacency matrices
#     if df_stocks_clean is not None:
#         viz_path = output_dir / 'adjacency_visualization.pdf'
#         visualize_adjacency_matrices(
#             corr_matrix=corr_only,
#             sector_matrix=sector_only,
#             combined_matrix=adj_matrix,
#             stocks_df=df_stocks_clean,
#             stock_order=final_symbols,
#             output_path=viz_path,
#             visualization_threshold=args.viz_threshold,
#             sector_bonus=args.sector_bonus
#         )
    
#     # Save metadata
#     metadata = {
#         'method': args.method,
#         'min_coverage': args.min_coverage if args.method == 'drop_incomplete' else None,
#         'edge_weight_threshold': float(args.edge_weight_threshold),
#         'correlation_threshold': float(args.edge_weight_threshold),  # Backward-compatible key
#         'corr_method': args.corr_method,
#         'sector_bonus': args.sector_bonus,
#         'adjacency_formula': f'|fundamentals_correlations| + {args.sector_bonus} * sector_adjacency',
#         'adjacency_thresholding_enabled': bool(args.edge_weight_threshold > 0),
#         'visualization_threshold': float(args.viz_threshold),
#         'num_stocks': len(final_symbols),
#         'num_dates': df_clean['Date'].nunique(),
#         'num_nonzero_edges': int(np.count_nonzero(adj_matrix)),
#         'nonzero_edge_density': float(np.count_nonzero(adj_matrix) / (adj_matrix.shape[0] * adj_matrix.shape[1])),
#         'graph_metadata_file': 'graph_metadata.json',
#         'num_connected_components': graph_metadata['num_connected_components'],
#         'largest_component_fraction': graph_metadata['largest_component_fraction'],
#         'min_degree': graph_metadata['degree_stats']['min'],
#         'max_degree': graph_metadata['degree_stats']['max'],
#         'avg_weighted_degree': graph_metadata['weighted_degree_stats']['mean'],
#         'num_edges_above_threshold': (
#             int(np.count_nonzero(adj_matrix >= args.edge_weight_threshold))
#             if args.edge_weight_threshold > 0 else None
#         ),
#         'edge_density_above_threshold': (
#             float(np.count_nonzero(adj_matrix >= args.edge_weight_threshold) / (adj_matrix.shape[0] * adj_matrix.shape[1]))
#             if args.edge_weight_threshold > 0 else None
#         ),
#         'adj_matrix_stats': {
#             'min': float(adj_matrix.min()),
#             'max': float(adj_matrix.max()),
#             'mean': float(adj_matrix.mean()),
#             'std': float(adj_matrix.std()),
#         },
#         'date_range': [df_clean['Date'].min(), df_clean['Date'].max()],
#         'timestamp': datetime.now().isoformat(),
#     }
    
#     import json
#     metadata_out = output_dir / 'metadata.json'
#     with open(metadata_out, 'w') as f:
#         json.dump(metadata, f, indent=2)
#     print(f"✓ Saved metadata.json")

#     graph_metadata_out = output_dir / 'graph_metadata.json'
#     with open(graph_metadata_out, 'w') as f:
#         json.dump(graph_metadata, f, indent=2)
#     print(f"✓ Saved graph_metadata.json")
    
#     # Print summary
#     print("\n" + "="*80)
#     print("Cleaning Complete!")
#     print("="*80)
#     print(f"\nOutput directory: {output_dir}")
#     print(f"Files created:")
#     print(f"  - values.csv: {len(df_clean)} rows")
#     print(f"  - adj.npy: {adj_matrix.shape}")
#     if df_stocks_clean is not None:
#         print(f"  - stocks.csv: {len(df_stocks_clean)} rows")
#     print(f"  - fundamentals.csv: {len(df_fundamentals_clean)} rows")
#     print(f"  - metadata.json")
#     print(f"  - graph_metadata.json")
    
#     print(f"\nDataset summary:")
#     print(f"  Stocks: {len(final_symbols)}")
#     print(f"  Dates: {df_clean['Date'].nunique()}")
#     print(f"  Date range: {df_clean['Date'].min()} to {df_clean['Date'].max()}")
#     print(f"  Adjacency formula: {metadata['adjacency_formula']}")
#     print(f"  Adjacency stats: min={metadata['adj_matrix_stats']['min']:.4f}, max={metadata['adj_matrix_stats']['max']:.4f}, mean={metadata['adj_matrix_stats']['mean']:.4f}")
#     print(f"  Nonzero edges: {metadata['num_nonzero_edges']}")
#     print(f"  Nonzero edge density: {metadata['nonzero_edge_density']:.4f}")
#     print(f"  Connected components: {graph_metadata['num_connected_components']}")
#     print(f"  Degree min/max: {graph_metadata['degree_stats']['min']}/{graph_metadata['degree_stats']['max']}")
#     print(f"  Avg weighted degree: {graph_metadata['weighted_degree_stats']['mean']:.4f}")
#     if args.edge_weight_threshold > 0:
#         print(f"  Edges above threshold {args.edge_weight_threshold}: {metadata['num_edges_above_threshold']}")
#         print(f"  Edge density (threshold {args.edge_weight_threshold}): {metadata['edge_density_above_threshold']:.4f}")
#     else:
#         print("  Edge thresholding disabled")
    
#     print(f"\nTo use this cleaned dataset, update your config:")
#     print(f"  dataset.data_dir: {output_dir}")
#     print("="*80)


# if __name__ == "__main__":
#     main()
