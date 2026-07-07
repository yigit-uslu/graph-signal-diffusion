#!/bin/bash
# Train PD expert on wra_small scenario (50 links, dense).
# Collected samples feed into diffusion model training.
CUDA_VISIBLE_DEVICES=0 python -m graph_signal_diffusion.cli.wra.train_pd \
    --config-name=pd_training/wra_small \
    training.r_min=0.25
