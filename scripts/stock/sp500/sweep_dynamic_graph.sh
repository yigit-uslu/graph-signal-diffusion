#!/bin/bash
# Stage 5b: Sweep dynamic-graph (correlation threshold, top_k, min_degree) for
# monthly and quarterly periods.  Run from the project root after cleaning.

CLEANED_DIR=data/sp500/cleaned_drop_incomplete_min_coverage_0.95_corr_0.7_sector_bonus_0.05

python -m graph_signal_diffusion.cli.stock.sweep_dynamic_graph \
    --input-dir "$CLEANED_DIR" \
    --periods "21,63" \
    --thresholds "0.0,0.2,0.5,0.7" \
    --top-k-values "none,20" \
    --min-degree-values "none,2"
