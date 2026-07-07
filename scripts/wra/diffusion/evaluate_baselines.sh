#!/bin/bash
# Evaluate full-power and WMMSE baselines on the WRA task.

export HYDRA_FULL_ERROR=1

CUDA_VISIBLE_DEVICES=1 python -m graph_signal_diffusion.cli.evaluate \
    dataset=wra task=wireless_resource_allocation baseline=fp

CUDA_VISIBLE_DEVICES=1 python -m graph_signal_diffusion.cli.evaluate \
    dataset=wra task=wireless_resource_allocation baseline=wmmse
