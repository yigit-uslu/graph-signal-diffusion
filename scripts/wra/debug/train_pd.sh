#!/bin/bash
# Train PD expert on wra_debug scenario (50 links, sparse).
# Collected samples feed into diffusion model training.
CUDA_VISIBLE_DEVICES=1 python -m graph_signal_diffusion.cli.wra.train_pd \
    --config-name=pd_training/wra_debug
