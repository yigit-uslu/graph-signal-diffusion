#!/bin/bash
# Stage 2: Refresh S&P 500 sector metadata (stocks.csv) from Wikipedia.
# Run from the project root when sector info needs updating without a full re-download.

python -m graph_signal_diffusion.cli.stock.update_stocks
