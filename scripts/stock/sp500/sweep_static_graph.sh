#!/bin/bash
# Stage 5a: Sweep (threshold, top_k, min_degree) on the static cleaned adjacency.
# Run from the project root after cleaning.

CLEANED_DIR=data/sp500/cleaned_drop_incomplete_min_coverage_0.95_corr_0.7_sector_bonus_0.05

python -m graph_signal_diffusion.cli.stock.sweep_static_graph \
    --input-dir "$CLEANED_DIR" \
    --thresholds "0.0,0.01,0.1,0.2,0.5,0.6,0.7" \
    --top-k-values "none,20,50" \
    --min-degree-values "none,1,2,10"
