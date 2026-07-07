# !/bin/bash

CUDA_VISIBLE_DEVICES=1 python scripts/convert_pd_samples_to_diffusion.py \
  --input outputs/primal_dual/2026-02-01/18-04-47/collected_samples.npz \
  --output-root data/wra

# CUDA_VISIBLE_DEVICES=1 python scripts/convert_pd_samples_to_diffusion.py \
#   --input outputs/primal_dual/2026-01-31/22-30-15/collected_samples.npz \
#   --output-root data/wra