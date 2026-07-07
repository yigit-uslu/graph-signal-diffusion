#!/bin/bash
# Compare GRW and diffusion baselines on SP100 (default config).
# Run from the project root.

python -m graph_signal_diffusion.cli.compare_baselines \
    baselines_to_compare='[grw,diffusion]'

# GRW only:
# python -m graph_signal_diffusion.cli.compare_baselines baselines_to_compare='[grw]'
#
# Tune GRW parameters:
# python -m graph_signal_diffusion.cli.compare_baselines \
#     baselines_to_compare='[grw]' \
#     baselines.grw.n_samples=50 \
#     baselines.grw.shrinkage_strength=20.0
