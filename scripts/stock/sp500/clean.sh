#!/bin/bash
# Stage 4: Clean raw SP500 data and regenerate adjacency + graph diagnostics.
# Produces: data/sp500/cleaned_drop_incomplete_min_coverage_0.95_corr_0.7_sector_bonus_0.05/
# Run from the project root.

python -m graph_signal_diffusion.cli.stock.clean \
    --method drop_incomplete \
    --min-coverage 0.95 \
    --edge-weight-threshold 0.7 \
    --sector-bonus 0.05
