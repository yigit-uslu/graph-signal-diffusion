#!/usr/bin/env python3
"""
Script to fetch S&P 500 sector information and create stocks.csv file.
This can be run independently without redownloading price data.
"""

import sys
from pathlib import Path
import pandas as pd



def fetch_sp500_sector_info():
    """
    Fetch S&P 500 stock information including sectors from Wikipedia.
    Returns DataFrame with columns: Symbol, Name, Sector
    """
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    try:
        # Add user agent to avoid 403 errors
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        tables = pd.read_html(url, storage_options=headers)
        sp500_table = tables[0]
        
        # Extract symbol, name, and sector information
        stocks_info = sp500_table[['Symbol', 'Security', 'GICS Sector']].copy()
        stocks_info.columns = ['Symbol', 'Name', 'Sector']
        
        # Replace dots with dashes for Yahoo Finance compatibility
        stocks_info['Symbol'] = stocks_info['Symbol'].str.replace('.', '-', regex=False)
        
        print(f"Successfully retrieved sector information for {len(stocks_info)} S&P 500 stocks")
        return stocks_info
    except Exception as e:
        print(f"Error fetching S&P 500 sector information: {e}")
        return None


def create_stocks_csv(data_dir, stocks_info=None):
    """
    Create stocks.csv file with sector information for stocks in the dataset.
    
    Args:
        data_dir: Path to the data directory (e.g., data/sp500/raw)
        stocks_info: Optional DataFrame with stock info. If None, will fetch from Wikipedia.
    """
    data_dir = Path(data_dir)
    
    # Fetch sector info if not provided
    if stocks_info is None:
        print("Fetching S&P 500 sector information from Wikipedia...")
        stocks_info = fetch_sp500_sector_info()
        if stocks_info is None:
            print("Failed to fetch sector information. Exiting.")
            return False
    
    # Check if values.csv exists to determine which stocks are in the dataset
    values_file = data_dir / 'values.csv'
    if not values_file.exists():
        print(f"Warning: {values_file} not found.")
        print("Creating stocks.csv with all S&P 500 stocks...")
        output_file = data_dir / 'stocks.csv'
        stocks_info.to_csv(output_file, index=False)
        print(f"✓ Created {output_file} with {len(stocks_info)} stocks")
        return True
    
    # Read values.csv to get the actual stocks in the dataset
    print(f"Reading {values_file} to identify stocks in dataset...")
    df_values = pd.read_csv(values_file)
    
    if 'Symbol' in df_values.columns:
        valid_stocks = df_values['Symbol'].unique()
    else:
        print("Error: 'Symbol' column not found in values.csv")
        return False
    
    print(f"Found {len(valid_stocks)} stocks in dataset")
    
    # Filter stocks_info to only include stocks in the dataset
    valid_stocks_info = stocks_info[stocks_info['Symbol'].isin(valid_stocks)].copy()
    
    # Sort by symbol for consistency
    valid_stocks_info = valid_stocks_info.sort_values('Symbol').reset_index(drop=True)
    
    # Save to stocks.csv
    output_file = data_dir / 'stocks.csv'
    valid_stocks_info.to_csv(output_file, index=False)
    
    print(f"\n✓ Successfully created {output_file}")
    print(f"  - Total stocks: {len(valid_stocks_info)}")
    print(f"  - Sectors: {valid_stocks_info['Sector'].nunique()}")
    print(f"\nSector distribution:")
    sector_counts = valid_stocks_info['Sector'].value_counts()
    for sector, count in sector_counts.items():
        print(f"  - {sector}: {count}")
    
    return True


def main():
    # Set up paths
    project_root = Path(__file__).parents[4]
    data_dir = project_root / 'data' / 'sp500' / 'raw'
    
    if not data_dir.exists():
        print(f"Error: Directory {data_dir} does not exist.")
        print("Please run download_sp500_data.py first to download the dataset.")
        return
    
    print("="*80)
    print("S&P 500 Stocks Information Update")
    print("="*80)
    print(f"\nData directory: {data_dir}\n")
    
    # Create stocks.csv
    success = create_stocks_csv(data_dir)
    
    if success:
        print("\n" + "="*80)
        print("Update Complete!")
        print("="*80)
    else:
        print("\nUpdate failed. Please check the errors above.")


if __name__ == "__main__":
    main()
