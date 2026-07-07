#!/usr/bin/env python3
"""
Example script showing how to extract and save metrics from baseline_comparison.csv

This demonstrates all the available methods for working with baseline comparison metrics.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from graph_signal_diffusion.evaluation import (
    display_metrics_table,
    save_metrics_table_markdown,
    extract_metrics_table,
)
from graph_signal_diffusion.baselines.stock_price_forecasting.grw import GeometricRandomWalk


# Example CSV path - adjust to your actual path
CSV_PATH = "outputs/stock_price_forecasting_v2-sp500_cleaned/comparison/2026-02-16/14-15-25/baseline_comparison.csv"


def example_1_display_default():
    """Example 1: Display metrics with default settings."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Display with default metrics (RMSE, CRPS, MAE, MSE)")
    print("="*80)
    
    df = display_metrics_table(CSV_PATH)
    return df


def example_2_custom_metrics():
    """Example 2: Display custom metrics."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Display custom metrics (CRPS and Coverage)")
    print("="*80)
    
    df = display_metrics_table(
        CSV_PATH,
        metrics=['crps_mean', 'crps_std', 'coverage_90', 'coverage_80', 'mis_90'],
        decimals=4
    )
    return df


def example_3_save_markdown():
    """Example 3: Save to markdown file."""
    print("\n" + "="*80)
    print("EXAMPLE 3: Save to markdown file")
    print("="*80)
    
    output_path = Path(CSV_PATH).parent / "custom_metrics.md"
    
    save_metrics_table_markdown(
        CSV_PATH,
        str(output_path),
        metrics=['rmse', 'mae', 'crps_mean'],
        decimals=3
    )
    
    print(f"Saved to: {output_path}")


def example_4_grw_convenience():
    """Example 4: Use GRW class convenience methods."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Use GRW class convenience methods")
    print("="*80)
    
    # Display
    df = GeometricRandomWalk.display_comparison_metrics(CSV_PATH)
    
    # Save to markdown (automatically saves to same directory)
    GeometricRandomWalk.save_comparison_metrics_markdown(CSV_PATH)
    
    return df


def example_5_extract_for_processing():
    """Example 5: Extract metrics for further processing."""
    print("\n" + "="*80)
    print("EXAMPLE 5: Extract metrics DataFrame for further processing")
    print("="*80)
    
    df = extract_metrics_table(
        CSV_PATH,
        metrics=['rmse', 'mae', 'crps_mean'],
        splits=['val', 'test']
    )
    
    print("\nExtracted DataFrame:")
    print(df)
    
    # Now you can do custom analysis
    print("\n\nCustom analysis - Performance degradation from val to test:")
    for metric in ['rmse', 'mae', 'crps_mean']:
        val_value = df[('val', metric)].iloc[0]
        test_value = df[('test', metric)].iloc[0]
        degradation = ((test_value - val_value) / val_value) * 100
        print(f"  {metric:12s}: {degradation:+.1f}%")
    
    return df


def main():
    """Run all examples."""
    csv_path = Path(CSV_PATH)
    
    if not csv_path.exists():
        print(f"Error: CSV file not found: {CSV_PATH}")
        print("\nPlease update CSV_PATH in this script to point to your baseline_comparison.csv file")
        sys.exit(1)
    
    print(f"\nUsing CSV: {CSV_PATH}\n")
    
    # Run examples
    example_1_display_default()
    example_2_custom_metrics()
    example_3_save_markdown()
    example_4_grw_convenience()
    example_5_extract_for_processing()
    
    print("\n" + "="*80)
    print("✓ All examples completed successfully!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
