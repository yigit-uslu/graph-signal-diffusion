#!/bin/bash
# Stage 1: Download raw SP500 price + fundamental data via yfinance.
# Produces: data/sp500/raw/{values.csv, adj.npy, fundamentals.csv, stocks.csv}
# Run from the project root.

python -m graph_signal_diffusion.cli.stock.download \
    --years-back 10 \
    --end-date 2026-02-14 \
    --coverage-threshold 0.8 \
    --sector-bonus 0.0 \
    --corr-method spearman
