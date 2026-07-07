#!/bin/bash
# Train PD expert on wra_medium_high_density scenario (200 links, high-density).
# Collected samples feed into diffusion model training.
CUDA_VISIBLE_DEVICES=0 python -m graph_signal_diffusion.cli.wra.train_pd \
    --config-name=pd_training/wra_medium_high_density
