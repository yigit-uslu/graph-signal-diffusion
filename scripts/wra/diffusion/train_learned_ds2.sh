#!/bin/bash
# Train WRA diffusion model with UGNN learned downsampling
# (gamma=[1,2,2,2], STE selector).
#
# Uncomment the LD_LIBRARY_PATH block below if you encounter
# GLIBCXX / GLIBC / PyTorch library loading issues on Linux.
# --------------------------------------------------------------------------
# if [ -z "$CONDA_PREFIX" ]; then
#     echo "Error: Conda environment not activated."
#     echo "Run: conda activate graph-signal-diffusion"
#     exit 1
# fi
# export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH}"
# export PYTHONWARNINGS="ignore::UserWarning"
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# --------------------------------------------------------------------------

CUDA_VISIBLE_DEVICES=0 python -m graph_signal_diffusion.cli.train \
    task=wireless_resource_allocation \
    model@task.model=ugnn_wra_learned_ds2
