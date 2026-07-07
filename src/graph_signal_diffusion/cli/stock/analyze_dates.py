"""
Analyze SP500 data quality and date coverage.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import sys

def analyze_sp500_dates(data_path: str = "data/sp500/raw/values.csv"):
    """Analyze date coverage for each stock."""
    
    print(f"Loading data from {data_path}...")
    
    # Load data
    df = pd.read_csv(data_path)
    df = df.set_index(['Symbol', 'Date'])
    
    # Get unique stocks and dates
    symbols = df.index.get_level_values('Symbol').unique()
    all_dates = df.index.get_level_values('Date').unique()
    
    print(f"\n{'='*80}")
    print(f"SP500 Dataset Overview")
    print(f"{'='*80}")
    print(f"Total stocks: {len(symbols)}")
    print(f"Total unique dates: {len(all_dates)}")
    print(f"Date range: {all_dates.min()} to {all_dates.max()}")
    
    # Analyze each stock
    stock_info = []
    for symbol in symbols:
        stock_dates = df.loc[symbol].index.unique()
        stock_info.append({
            'symbol': symbol,
            'num_dates': len(stock_dates),
            'first_date': stock_dates.min(),
            'last_date': stock_dates.max(),
            'missing_ratio': 1 - len(stock_dates) / len(all_dates),
            'coverage': len(stock_dates) / len(all_dates)
        })
    
    stock_df = pd.DataFrame(stock_info).sort_values('num_dates')
    
    print(f"\n{'='*80}")
    print(f"Stock Coverage Analysis")
    print(f"{'='*80}")
    print(f"Min dates per stock: {stock_df['num_dates'].min()}")
    print(f"Max dates per stock: {stock_df['num_dates'].max()}")
    print(f"Mean dates per stock: {stock_df['num_dates'].mean():.1f}")
    print(f"Median dates per stock: {stock_df['num_dates'].median():.1f}")
    
    # Show stocks with least coverage
    print(f"\n{'='*80}")
    print(f"10 Stocks with Least Coverage")
    print(f"{'='*80}")
    print(stock_df[['symbol', 'num_dates', 'coverage', 'missing_ratio', 'first_date', 'last_date']].head(10).to_string(index=False))
    
    # Show stocks with most coverage
    print(f"\n{'='*80}")
    print(f"10 Stocks with Most Coverage")
    print(f"{'='*80}")
    print(stock_df[['symbol', 'num_dates', 'coverage', 'missing_ratio', 'first_date', 'last_date']].tail(10).to_string(index=False))
    
    # Find common date range
    latest_start = stock_df['first_date'].max()
    earliest_end = stock_df['last_date'].min()
    common_dates = [d for d in all_dates if latest_start <= d <= earliest_end]
    
    print(f"\n{'='*80}")
    print(f"Common Date Range Analysis (All Stocks)")
    print(f"{'='*80}")
    print(f"Latest start date: {latest_start}")
    print(f"Earliest end date: {earliest_end}")
    print(f"Common date range: {len(common_dates)} dates")
    print(f"Data retention: {len(common_dates) / len(all_dates) * 100:.1f}%")
    
    # Thresholds analysis
    print(f"\n{'='*80}")
    print(f"Drop Incomplete Stock Analysis (Option 2)")
    print(f"{'='*80}")
    print(f"{'Threshold':<15} {'Stocks Kept':<15} {'% Stocks':<15} {'Common Dates':<15} {'% Dates':<15}")
    print(f"{'-'*80}")
    
    for threshold in [0.95, 0.90, 0.85, 0.80, 0.75]:
        min_dates = int(len(all_dates) * threshold)
        stocks_kept = stock_df[stock_df['num_dates'] >= min_dates]
        
        if len(stocks_kept) > 0:
            # Find common range for these stocks
            kept_symbols = stocks_kept['symbol'].tolist()
            kept_latest_start = stocks_kept['first_date'].max()
            kept_earliest_end = stocks_kept['last_date'].min()
            kept_common_dates = [d for d in all_dates if kept_latest_start <= d <= kept_earliest_end]
            
            print(f"{threshold*100:.0f}% coverage  {len(stocks_kept):<15} {len(stocks_kept)/len(symbols)*100:<14.1f}% {len(kept_common_dates):<15} {len(kept_common_dates)/len(all_dates)*100:<14.1f}%")
        else:
            print(f"{threshold*100:.0f}% coverage  0               0.0%            0               0.0%")
    
    # Distribution of missing dates
    print(f"\n{'='*80}")
    print(f"Missing Data Distribution")
    print(f"{'='*80}")
    
    missing_buckets = [
        (0, 0.01, "< 1% missing"),
        (0.01, 0.05, "1-5% missing"),
        (0.05, 0.10, "5-10% missing"),
        (0.10, 0.20, "10-20% missing"),
        (0.20, 0.50, "20-50% missing"),
        (0.50, 1.0, "> 50% missing")
    ]
    
    for low, high, label in missing_buckets:
        count = ((stock_df['missing_ratio'] >= low) & (stock_df['missing_ratio'] < high)).sum()
        if low == 0.50:  # Last bucket includes upper bound
            count = (stock_df['missing_ratio'] >= low).sum()
        pct = count / len(symbols) * 100
        print(f"{label:<20} {count:>4} stocks ({pct:>5.1f}%)")
    
    return stock_df, all_dates, common_dates


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Analyze SP500 dataset date coverage")
    parser.add_argument("--input", default="data/sp500/raw/values.csv",
                       help="Input CSV path")
    parser.add_argument("--output", default="data/sp500/stock_coverage_analysis.csv",
                       help="Output CSV for detailed analysis")

    args = parser.parse_args()

    try:
        stock_df, all_dates, common_dates = analyze_sp500_dates(args.input)

        # Save detailed analysis
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        stock_df.to_csv(output_path, index=False)
        print(f"\n{'='*80}")
        print(f"✅ Saved detailed analysis to {output_path}")
        print(f"{'='*80}")

    except FileNotFoundError:
        print(f"\n❌ Error: File not found: {args.input}")
        print(f"Make sure the SP500 dataset exists at the specified path.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
