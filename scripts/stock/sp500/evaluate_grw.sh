#!/bin/bash
# Evaluate GRW baseline only on SP500-cleaned.
# IMPORTANT: dataset.n_samples_per_input=1 — GRW handles multi-sample replication
# internally; setting it higher would cause double-replication (n × n).
# Run from the project root.

export HYDRA_FULL_ERROR=1

batch_size=64
batch_size_val=64

CUDA_VISIBLE_DEVICES=0 python -m graph_signal_diffusion.cli.compare_baselines \
    --config-name compare_baselines_sp500 \
    baselines_to_compare='[grw]' \
    dataset.batch_size=$batch_size \
    dataset.batch_size_val=$batch_size_val \
    dataset.n_samples_per_input=1 \
    baselines.grw.n_samples=20 \
    dataset.past_window=20 \
    dataset.future_window=5 \
    wandb.enabled=False
