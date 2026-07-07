#!/bin/bash
# Analyse wra_large_outdoor_low_density channel scenario: generates networks, visualizes deployments,
# and computes full-power rate statistics to guide r_min selection.

export HYDRA_FULL_ERROR=1  # Enable full Hydra error traces for easier debugging.
CUDA_VISIBLE_DEVICES=0 python -m graph_signal_diffusion.cli.wra.analyze_channels \
    --config-name=channel_analysis/wra_large_outdoor_low_density \
    # --cfg job --resolve
