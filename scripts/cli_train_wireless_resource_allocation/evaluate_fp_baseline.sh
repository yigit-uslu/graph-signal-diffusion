# DEPRECATED: use scripts/wra/diffusion/evaluate_baselines.sh instead.
#!/bin/bash

rm -rf src/**/__pycache__/*.pyc

export HYDRA_FULL_ERROR=1  # Enable full Hydra error tracebacks for easier debugging


CUDA_VISIBLE_DEVICES=1 python -m graph_signal_diffusion.cli.evaluate \
  dataset=wra task=wireless_resource_allocation baseline=fp
  

CUDA_VISIBLE_DEVICES=1 python -m graph_signal_diffusion.cli.evaluate \
  dataset=wra task=wireless_resource_allocation baseline=wmmse
