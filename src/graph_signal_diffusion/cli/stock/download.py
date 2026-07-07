#!/usr/bin/env python3
"""
Script to download S&P 500 stock data using yfinance.
Downloads daily data for all S&P 500 constituents over a configurable lookback window,
then builds a static graph from fundamentals similarity (SP100 notebook style):

    adj = abs(spearman_corr(fundamentals)) + sector_bonus * sector_relation_adj
    adj = adj * (adj >= edge_weight_threshold)   # optional threshold
    adj = adj / adj.max()                        # if max > 0
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import sector info function
from graph_signal_diffusion.cli.stock.update_stocks import fetch_sp500_sector_info


def get_sp500_tickers():
    """
    Get list of current S&P 500 constituents from Wikipedia.
    Returns tuple of (tickers_list, stocks_dataframe)
    """
    stocks_info = fetch_sp500_sector_info()
    if stocks_info is None:
        return None, None
    
    tickers = stocks_info['Symbol'].tolist()
    return tickers, stocks_info


def download_price_data(tickers, start_date, end_date, output_dir):
    """
    Download historical price data for given tickers.
    """
    print(f"\nDownloading price data from {start_date} to {end_date}...")
    
    # Download all tickers at once (faster)
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        progress=True,
        threads=True,
        group_by='ticker'
    )
    
    if data.empty:
        print("No data downloaded!")
        return None
    
    # Process data into long format
    records = []
    failed_tickers = []
    
    print("\nProcessing downloaded data...")
    for ticker in tqdm(tickers):
        try:
            if len(tickers) == 1:
                ticker_data = data
            else:
                ticker_data = data[ticker]
            
            # Check if data exists and is not empty
            if ticker_data.empty or ticker_data['Close'].isna().all():
                failed_tickers.append(ticker)
                continue
            
            # Reset index to get dates
            ticker_data = ticker_data.reset_index()
            
            # Add ticker symbol
            ticker_data['Symbol'] = ticker
            
            # Keep only necessary columns
            ticker_data = ticker_data[['Symbol', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            
            # Drop rows with NaN values
            ticker_data = ticker_data.dropna()
            
            if len(ticker_data) > 0:
                records.append(ticker_data)
            else:
                failed_tickers.append(ticker)
                
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            failed_tickers.append(ticker)
    
    if failed_tickers:
        print(f"\nFailed to download data for {len(failed_tickers)} tickers:")
        print(failed_tickers)
    
    if not records:
        print("No valid data to save!")
        return None
    
    # Combine all records
    df = pd.concat(records, ignore_index=True)
    
    # Save original data
    output_file = output_dir / 'stocks_orig.csv'
    df.to_csv(output_file, index=False)
    print(f"\nSaved original data to {output_file}")
    print(f"Total records: {len(df)}")
    print(f"Unique stocks: {df['Symbol'].nunique()}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    return df


def calculate_technical_indicators(df):
    """
    Calculate technical indicators for each stock.
    """
    print("\nCalculating technical indicators...")
    
    # Sort by symbol and date
    df = df.sort_values(['Symbol', 'Date'])
    
    # Calculate indicators for each stock
    results = []
    
    for symbol in tqdm(df['Symbol'].unique()):
        stock_data = df[df['Symbol'] == symbol].copy()
        stock_data = stock_data.sort_values('Date')
        
        # Calculate log returns
        stock_data['Close_lag1'] = stock_data['Close'].shift(1)
        stock_data['DailyLogReturn'] = np.log(stock_data['Close'] / stock_data['Close_lag1']) * 100
        
        # Average Log Returns (ALR) over different periods
        stock_data['ALR1W'] = stock_data['DailyLogReturn'].rolling(window=5).mean()
        stock_data['ALR2W'] = stock_data['DailyLogReturn'].rolling(window=10).mean()
        stock_data['ALR1M'] = stock_data['DailyLogReturn'].rolling(window=21).mean()
        stock_data['ALR2M'] = stock_data['DailyLogReturn'].rolling(window=42).mean()
        
        # RSI (Relative Strength Index)
        delta = stock_data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        stock_data['RSI'] = (1 - (1 / (1 + rs)))
        
        # MACD (Moving Average Convergence Divergence)
        exp1 = stock_data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = stock_data['Close'].ewm(span=26, adjust=False).mean()
        stock_data['MACD'] = (exp1 - exp2) / stock_data['Close']
        
        # Normalized Close (z-score normalization)
        close_mean = stock_data['Close'].mean()
        close_std = stock_data['Close'].std()
        stock_data['NormClose'] = (stock_data['Close'] - close_mean) / (close_std + 1e-8)
        
        # Drop intermediate columns and NaN rows
        stock_data = stock_data.drop(columns=['Close_lag1'])
        stock_data = stock_data.dropna()
        
        results.append(stock_data)
    
    df_processed = pd.concat(results, ignore_index=True)
    
    return df_processed


def download_fundamentals(tickers, output_dir):
    """
    Download fundamental data for given tickers.
    """
    print("\nDownloading fundamental data...")
    
    fundamentals = []
    failed_tickers = []
    
    for ticker in tqdm(tickers):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Extract key fundamental metrics
            fundamental_data = {
                'Symbol': ticker,
                'marketCap': info.get('marketCap', np.nan),
                'trailingPE': info.get('trailingPE', np.nan),
                'forwardPE': info.get('forwardPE', np.nan),
                'pegRatio': info.get('pegRatio', np.nan),
                'priceToBook': info.get('priceToBook', np.nan),
                'trailingEps': info.get('trailingEps', np.nan),
                'forwardEps': info.get('forwardEps', np.nan),
                'bookValue': info.get('bookValue', np.nan),
                'payoutRatio': info.get('payoutRatio', np.nan),
                'beta': info.get('beta', np.nan),
                'fiveYearAvgDividendYield': info.get('fiveYearAvgDividendYield', np.nan),
                '52WeekChange': info.get('52WeekChange', np.nan),
                'averageVolume': info.get('averageVolume', np.nan),
                'enterpriseToRevenue': info.get('enterpriseToRevenue', np.nan),
                'profitMargins': info.get('profitMargins', np.nan),
            }
            
            fundamentals.append(fundamental_data)
            
        except Exception as e:
            print(f"Error downloading fundamentals for {ticker}: {e}")
            failed_tickers.append(ticker)
    
    if failed_tickers:
        print(f"\nFailed to download fundamentals for {len(failed_tickers)} tickers:")
        print(failed_tickers)
    
    df_fundamentals = pd.DataFrame(fundamentals)
    
    # Save original fundamentals
    output_file = output_dir / 'fundamentals_orig.csv'
    df_fundamentals.to_csv(output_file, index=False)
    print(f"\nSaved original fundamentals to {output_file}")
    
    return df_fundamentals


def normalize_fundamentals(df_fundamentals):
    """
    Normalize fundamental data using z-score normalization.
    """
    print("\nNormalizing fundamental data...")
    
    df_norm = df_fundamentals.copy()
    
    # Get numerical columns (exclude Symbol)
    numerical_cols = df_norm.select_dtypes(include=[np.number]).columns
    
    # Z-score normalization
    for col in numerical_cols:
        mean_val = df_norm[col].mean()
        std_val = df_norm[col].std()
        df_norm[col] = (df_norm[col] - mean_val) / (std_val + 1e-8)
    
    # Fill NaN with 0
    df_norm = df_norm.fillna(0)
    
    return df_norm


def get_valid_stocks_by_return_coverage(
    df: pd.DataFrame,
    min_coverage: float = 0.8,
    target_column: str = "DailyLogReturn",
) -> list[str]:
    """
    Select stocks with at least `min_coverage` non-NaN target coverage.
    """
    pivot_data = df.pivot(index='Date', columns='Symbol', values=target_column)
    valid_stocks = pivot_data.columns[pivot_data.notna().mean() > min_coverage]
    return valid_stocks.tolist()


def filter_stocks_info_by_order(stocks_info: pd.DataFrame, stock_order: list[str]) -> pd.DataFrame:
    """
    Filter/reorder stocks metadata to match `stock_order`.
    """
    filtered = stocks_info[stocks_info['Symbol'].isin(stock_order)].copy()
    filtered['_order'] = filtered['Symbol'].map({s: i for i, s in enumerate(stock_order)})
    filtered = filtered.sort_values('_order').drop(columns=['_order'])
    return filtered


def compute_sector_relation_adjacency(stocks_df: pd.DataFrame, stock_order: list[str]) -> np.ndarray:
    """
    Build binary sector relation adjacency (1 for same sector, 0 otherwise), diagonal=0.
    """
    stocks = stocks_df.set_index('Symbol').reindex(stock_order)
    sectors = stocks['Sector']
    n = len(stock_order)
    adj = np.zeros((n, n), dtype=float)
    for i in range(n):
        si = sectors.iloc[i]
        if pd.isna(si):
            continue
        for j in range(n):
            if i == j:
                continue
            sj = sectors.iloc[j]
            if pd.isna(sj):
                continue
            adj[i, j] = float(si == sj)
    return adj


def compute_fundamentals_adjacency(
    fundamentals_df: pd.DataFrame,
    stocks_df: pd.DataFrame,
    stock_order: list[str],
    sector_bonus: float = 0.0,
    edge_weight_threshold: float = 0.0,
    corr_method: str = "spearman",
):
    """
    Compute static adjacency from fundamentals (SP100 notebook style):
      adj = abs(corr(fundamentals)) + sector_bonus * sector_relation_adj
      optional thresholding, then max-normalization.
    """
    print("\nComputing fundamentals-based weighted adjacency...")
    print(f"  Correlation method: {corr_method}")
    print(f"  Sector bonus: {sector_bonus}")
    if edge_weight_threshold > 0:
        print(f"  Edge-weight threshold: {edge_weight_threshold}")
    else:
        print("  Edge-weight threshold: disabled")

    fundamentals = fundamentals_df.set_index('Symbol').reindex(stock_order)
    numeric_cols = fundamentals.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric fundamentals columns found to build adjacency.")

    missing_symbols = int(fundamentals.index.isna().sum()) + int(fundamentals[numeric_cols].isna().all(axis=1).sum())
    if missing_symbols > 0:
        print(f"Warning: {missing_symbols} symbols have missing/empty fundamentals rows; filling with 0.")

    fundamentals_numeric = fundamentals[numeric_cols].fillna(0.0)

    # Match notebook approach: corr across stocks using all fundamentals features.
    fundamentals_corr = fundamentals_numeric.transpose().corr(method=corr_method).fillna(0.0)
    fundamentals_corr_np = fundamentals_corr.to_numpy(dtype=float)
    np.fill_diagonal(fundamentals_corr_np, 0.0)

    adj = np.abs(fundamentals_corr_np)

    if stocks_df is not None and sector_bonus > 0:
        sector_adj = compute_sector_relation_adjacency(stocks_df, stock_order)
        adj = adj + sector_adj * float(sector_bonus)

    if edge_weight_threshold > 0:
        adj = adj * (adj >= float(edge_weight_threshold))

    np.fill_diagonal(adj, 0.0)
    max_val = float(adj.max()) if adj.size > 0 else 0.0
    if max_val > 0:
        adj = adj / max_val

    print(f"Adjacency matrix shape: {adj.shape}")
    print(f"Number of nonzero edges: {np.count_nonzero(adj)}")
    print(f"Edge density: {np.count_nonzero(adj) / (adj.shape[0] * adj.shape[1]):.4f}")
    print(f"Edge weight stats: min={adj.min():.4f}, max={adj.max():.4f}, mean={adj.mean():.4f}")

    return adj


def filter_data_by_valid_stocks(df, valid_stocks):
    """
    Filter dataframe to only include valid stocks.
    """
    return df[df['Symbol'].isin(valid_stocks)].copy()


def main():
    default_end_date = "2026-02-14"

    parser = argparse.ArgumentParser(
        description="Download SP500 data and build fundamentals-based weighted adjacency."
    )
    parser.add_argument(
        "--years-back",
        type=int,
        default=10,
        help="Lookback window in years from end date (default: 10).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=default_end_date,
        help=f"End date in YYYY-MM-DD format (default: {default_end_date}).",
    )
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=0.8,
        help="Minimum DailyLogReturn coverage to keep a stock in values/fundamentals (default: 0.8).",
    )
    parser.add_argument(
        "--sector-bonus",
        type=float,
        default=0.0,
        help="Bonus added to adjacency when two stocks share sector (default: 0.0).",
    )
    parser.add_argument(
        "--edge-weight-threshold",
        type=float,
        default=0.0,
        help="Optional threshold on combined adjacency before normalization (default: 0.0 = disabled).",
    )
    parser.add_argument(
        "--corr-method",
        type=str,
        choices=["spearman", "pearson"],
        default="spearman",
        help="Fundamentals correlation method (default: spearman).",
    )
    args = parser.parse_args()

    if args.years_back <= 0:
        raise ValueError(f"--years-back must be > 0, got {args.years_back}")
    if not (0 < args.coverage_threshold <= 1):
        raise ValueError(f"--coverage-threshold must be in (0, 1], got {args.coverage_threshold}")
    if args.edge_weight_threshold < 0:
        raise ValueError(f"--edge-weight-threshold must be >= 0, got {args.edge_weight_threshold}")
    if args.sector_bonus < 0:
        raise ValueError(f"--sector-bonus must be >= 0, got {args.sector_bonus}")
    
    # Set up paths
    project_root = Path(__file__).parents[4]
    data_dir = project_root / 'data' / 'sp500' / 'raw'
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("S&P 500 Data Download Script")
    print("="*80)
    
    # Calculate date range
    try:
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    except ValueError as err:
        raise ValueError(
            f"Invalid --end-date '{args.end_date}'. Expected format YYYY-MM-DD."
        ) from err

    years_back = int(args.years_back)
    start_date = end_date - timedelta(days=years_back * 365 + 30)  # Add buffer for weekends/holidays
    
    print(f"\nDate range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Lookback years: {years_back}")
    print(f"Output directory: {data_dir}")
    
    # Step 1: Get S&P 500 tickers and sector information
    tickers, stocks_info = get_sp500_tickers()
    if tickers is None:
        print("Failed to retrieve tickers. Exiting.")
        return
    
    # Save original stocks info with sectors
    stocks_info.to_csv(data_dir / 'stocks_orig.csv', index=False)
    print(f"Saved original stock information to {data_dir / 'stocks_orig.csv'}")
    
    # Step 2: Download price data
    df_prices = download_price_data(tickers, start_date, end_date, data_dir)
    if df_prices is None:
        print("Failed to download price data. Exiting.")
        return
    
    # Step 3: Calculate technical indicators
    df_processed = calculate_technical_indicators(df_prices)
    
    # Step 4: Select valid stocks by return coverage and filter values
    valid_stocks = get_valid_stocks_by_return_coverage(
        df_processed,
        min_coverage=float(args.coverage_threshold),
        target_column="DailyLogReturn",
    )
    df_final = filter_data_by_valid_stocks(df_processed, valid_stocks)
    
    print(f"\nFinal dataset contains {len(valid_stocks)} stocks")
    print(f"Total records: {len(df_final)}")
    
    # Step 5: Download fundamentals (only for valid stocks)
    df_fundamentals = download_fundamentals(valid_stocks, data_dir)
    
    # Step 6: Normalize fundamentals
    df_fundamentals_norm = normalize_fundamentals(df_fundamentals)

    # Step 7: Build filtered stocks metadata in the same order as valid_stocks
    stocks_filtered = filter_stocks_info_by_order(stocks_info, valid_stocks)

    # Step 8: Compute adjacency from fundamentals (+ optional sector bonus)
    adj_matrix = compute_fundamentals_adjacency(
        fundamentals_df=df_fundamentals_norm,
        stocks_df=stocks_filtered,
        stock_order=valid_stocks,
        sector_bonus=float(args.sector_bonus),
        edge_weight_threshold=float(args.edge_weight_threshold),
        corr_method=str(args.corr_method),
    )
    
    # Step 9: Save processed data
    print("\nSaving processed data...")
    
    # Save values
    df_final.to_csv(data_dir / 'values.csv', index=False)
    print(f"Saved processed values to {data_dir / 'values.csv'}")
    
    # Save adjacency matrix
    np.save(data_dir / 'adj_orig.npy', adj_matrix)
    np.save(data_dir / 'adj.npy', adj_matrix)
    print(f"Saved adjacency matrix to {data_dir / 'adj.npy'}")
    
    # Save fundamentals
    df_fundamentals_norm.to_csv(data_dir / 'fundamentals.csv', index=False)
    print(f"Saved normalized fundamentals to {data_dir / 'fundamentals.csv'}")
    
    # Save stocks metadata with sectors, ordered to match adjacency/values stock order
    stocks_filtered.to_csv(data_dir / 'stocks.csv', index=False)
    print(f"Saved stock metadata to {data_dir / 'stocks.csv'}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("Download Complete!")
    print("="*80)
    print(f"\nFinal Statistics:")
    print(f"  - Number of stocks: {len(valid_stocks)}")
    print(f"  - Total records: {len(df_final)}")
    print(f"  - Date range: {df_final['Date'].min()} to {df_final['Date'].max()}")
    print(f"  - Features per stock: {len(df_final.columns) - 2}")  # Exclude Symbol and Date
    print(f"  - Adjacency matrix shape: {adj_matrix.shape}")
    print(f"  - Number of edges: {np.count_nonzero(adj_matrix)}")
    print(f"  - Coverage threshold: {args.coverage_threshold}")
    print(f"  - Correlation method: {args.corr_method}")
    print(f"  - Sector bonus: {args.sector_bonus}")
    if args.edge_weight_threshold > 0:
        print(f"  - Edge-weight threshold: {args.edge_weight_threshold}")
    else:
        print("  - Edge-weight threshold: disabled")
    print("  - Adjacency construction: abs(fundamentals corr) + sector bonus, then max-normalized")
    
    print("\nOutput files:")
    print(f"  - {data_dir / 'values.csv'}")
    print(f"  - {data_dir / 'adj.npy'}")
    print(f"  - {data_dir / 'fundamentals.csv'}")
    print(f"  - {data_dir / 'stocks.csv'}")
    
    print("\nTo use this data, update your config to use dataset: sp500")
    print("="*80)


if __name__ == "__main__":
    main()
