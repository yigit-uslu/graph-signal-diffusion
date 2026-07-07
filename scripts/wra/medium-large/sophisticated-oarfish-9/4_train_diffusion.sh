#!/bin/bash
# Stage 4: Train the diffusion model on all 20 subdatasets
# (4 densities x 5 r_min = 128,000 samples).
#
# This is the "Ablation 1: Baseline" configuration that produced the
# sophisticated-oarfish-9 checkpoint.  Architecture mirrors abiding-potoo-545
# (the single-density predecessor) — only the dataset changes.
#
# Key hyperparameters:
#   - UGNN v3 DS4: base_channels=64, channel_multipliers=[1,1,1], gamma=[1,2,2]
#   - GNN: K=2, num_layers=2, max_gnn_stride=2
#   - 3 bottleneck layers, learned pooling (STE, temp_min=0.50, linear anneal)
#   - DDIM: 500 timesteps, 100 sampling steps, eta=0.2
#   - AdamW: lr=1e-4, betas=[0.9,0.98], weight_decay=1e-4
#   - Cosine LR: warmup 1%, decay 80%, min_lr_ratio=0.05
#   - 5000 epochs, best model selected by composite metric
#
# Usage:
#   bash scripts/wra/medium-large/sophisticated-oarfish-9/4_train_diffusion.sh
#
#   # Override GPU:
#   GPU_ID=1 bash scripts/wra/medium-large/sophisticated-oarfish-9/4_train_diffusion.sh

set -euo pipefail
export HYDRA_FULL_ERROR=1

GPU_ID="${GPU_ID:-0}"
PRESET="medium-large_outdoor_all_density_all_rmin"

CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m graph_signal_diffusion.cli.train \
    task=wireless_resource_allocation \
    "dataset@task.dataset=wra_${PRESET}" \
    "trainer.name=gdm_wra_${PRESET}" \
    model@task.model=ugnn_wra_v3_ds4 \
    trainer@task.trainer=trainer_wra_v3_learned_ds4 \
    dataset.num_workers=4 \
    dataset.persistent_workers=false \
    dataset.pin_memory=false \
    'model.config.channel_multipliers=[1,1,1]' \
    'model.config.pooling_config.gamma=[1,2,2]' \
    model.config.gnn_config.max_gnn_stride=2 \
    model.config.num_bottleneck_layers=3 \
    model.config.pooling_config.entropy_reg_weight=0.0 \
    model.config.pooling_config.selector_kwargs.temperature_min=0.50 \
    model.config.pooling_config.selector_kwargs.temperature_schedule=linear \
    trainer.selector_temperature_schedule.anneal_ratio=0.90 \
    trainer.eval_every_n_epochs=50 \
    trainer.eval_schedule.type=multi_phase \
    trainer.eval_schedule.eval_on_first_epoch=false \
    trainer.eval_schedule.eval_on_last_epoch=true \
    'trainer.eval_schedule.phases=[{period: 10, until_epoch: 100}, {period: 50, until_epoch: -100}, {period: 20}]' \
    trainer.lr_scheduler.min_lr_ratio=0.05 \
    +task.plot_style.plots.task_rate_evolution.enabled=false \
    +task.plot_style.plots.task_power_scatter.enabled=false \
    +task.plot_style.plots.selector_rate_correlation.max_networks_per_dataset=1 \
    wandb.enabled=true
