#!/usr/bin/env python3
"""
Script to display key metrics from baseline_comparison.csv in a readable table format.

Usage:
    python scripts/display_metrics_table.py <path_to_csv> [--markdown <output_path>]
    
Example:
    # Display in terminal
    python scripts/display_metrics_table.py outputs/stock_price_forecasting_v2-sp500_cleaned/comparison/2026-02-16/14-15-25/baseline_comparison.csv
    
    # Save to markdown
    python scripts/display_metrics_table.py outputs/stock_price_forecasting_v2-sp500_cleaned/comparison/2026-02-16/14-15-25/baseline_comparison.csv --markdown metrics.md
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from graph_signal_diffusion.evaluation import display_metrics_table, save_metrics_table_markdown
from graph_signal_diffusion.baselines.stock_price_forecasting.grw import GeometricRandomWalk


def main():
    parser = argparse.ArgumentParser(
        description="Display and export metrics from baseline_comparison.csv"
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to baseline_comparison.csv file"
    )
    parser.add_argument(
        "--markdown", "-md",
        type=str,
        default=None,
        help="Save formatted table to markdown file (optional)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["grouped", "flat"],
        default="grouped",
        help="Table format: 'grouped' (by split) or 'flat' (single table)"
    )
    
    args = parser.parse_args()
    
    csv_path = args.csv_path
    
    if not Path(csv_path).exists():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)
    
    print(f"Reading metrics from: {csv_path}\n")
    
    # Display metrics in terminal
    print("=" * 80)
    print("DISPLAYING METRICS")
    print("=" * 80)
    df = display_metrics_table(csv_path, format=args.format)
    
    # Save to markdown if requested
    if args.markdown:
        output_path = args.markdown
        print(f"\nSaving to markdown: {output_path}")
        save_metrics_table_markdown(
            csv_path,
            output_path,
            format=args.format
        )
        print(f"✓ Markdown table saved to: {output_path}")
    else:
        # Offer to save using default name
        default_md = str(Path(csv_path).parent / "metrics_summary.md")
        print(f"\n💡 Tip: Add --markdown flag to save as markdown")
        print(f"   Example: --markdown {default_md}")
    
    print("\n✓ Successfully processed metrics from baseline_comparison.csv")


if __name__ == "__main__":
    main()
