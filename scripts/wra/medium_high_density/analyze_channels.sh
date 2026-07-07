#!/bin/bash
# Analyse wra_medium_high_density channel scenario: generates networks, visualises deployments,
# and computes full-power rate statistics to guide r_min selection.

export HYDRA_FULL_ERROR=1  # Enable full Hydra error traces for easier debugging.
CUDA_VISIBLE_DEVICES=0 python -m graph_signal_diffusion.cli.wra.analyze_channels \
    --config-name=channel_analysis/wra_medium_high_density \
    # --cfg job --resolve
