#!/bin/bash
# Analyse wra_small channel scenario: generates networks, visualises deployments,
# and computes full-power rate statistics to guide r_min selection.
CUDA_VISIBLE_DEVICES=1 python -m graph_signal_diffusion.cli.wra.analyze_channels \
    --config-name=channel_analysis/wra_small \
    num_examples=10
