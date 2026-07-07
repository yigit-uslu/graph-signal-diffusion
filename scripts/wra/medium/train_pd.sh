#!/bin/bash
# Train PD expert on wra_medium scenario (200 links, medium-density).
# Collected samples feed into diffusion model training.

export HYDRA_FULL_ERROR=1  # For debugging Hydra configs. Remove or set to 0 for cleaner error messages once configs are debugged.
CUDA_VISIBLE_DEVICES=1 python -m graph_signal_diffusion.cli.wra.train_pd \
    --config-name=pd_training/wra_medium
