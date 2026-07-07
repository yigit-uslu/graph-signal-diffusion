#!/bin/bash
# Stage 3: Analyse date coverage and missing-data distribution.
# Run from the project root after downloading raw data.

python -m graph_signal_diffusion.cli.stock.analyze_dates \
    --input data/sp500/raw/values.csv \
    --output data/sp500/stock_coverage_analysis.csv
