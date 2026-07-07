#!/bin/bash
# Stage 4 (optional / expensive) — DIFFUSION TRAINING (the rigorous-quoll-131 recipe).
#
# Trains the U-GNN diffusion model on the all-density (single r_min=0.6) dataset.
# Recipe recovered VERBATIM from the run's .hydra/overrides.yaml (+ explicit seed=0
# from .hydra/config.yaml). 5000 epochs; best model selected by the tail-weighted
# composite metric (rank-1 = best_model_epoch_1600.pt, the shipped checkpoint).
#
# STATISTICAL, not bitwise: seed=0 is pinned, but AMP + cuDNN are not bitwise
# deterministic, so a fresh run yields a statistically equivalent model — NOT the
# identical checkpoint or the exact test_summary.json numbers. For the exact
# numbers, use 05_evaluate_checkpoint.sh on the shipped checkpoint instead.
#
# Model: ugnn_wra_v3_ds8_norm_act_head (DS8, learned STE pooling, model_cond_channels=2).
#
# Usage:
#   ./04_train_diffusion.sh
#   WANDB=true ./04_train_diffusion.sh          # enable wandb logging (off by default)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/00_config.sh"
export HYDRA_FULL_ERROR=1

WANDB_ENABLED="${WANDB:-false}"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" $PYRUN -m graph_signal_diffusion.cli.train \
  seed=0 \
  task="$TASK" \
  "dataset@task.dataset=$DATASET_CFG" \
  "trainer.name=$TRAINER_NAME" \
  "model@task.model=$MODEL" \
  "trainer@task.trainer=$TRAINER" \
  dataset.num_workers=4 \
  dataset.persistent_workers=false \
  dataset.pin_memory=false \
  model.config.pooling_config.entropy_reg_weight=0.0 \
  model.config.pooling_config.selector_kwargs.temperature_min=0.50 \
  model.config.pooling_config.selector_kwargs.temperature_schedule=linear \
  model.config.pooling_config.selector_exploration_noise=1.0 \
  model.config.pooling_config.selector_exploration_noise_min=0.0 \
  model.config.pooling_config.selector_exploration_noise_schedule=linear \
  trainer.selector_temperature_schedule.warmup_ratio=0.02 \
  trainer.selector_temperature_schedule.anneal_ratio=0.73 \
  +trainer.selector_exploration_schedule.enabled=true \
  +trainer.selector_exploration_schedule.warmup_ratio=0.02 \
  +trainer.selector_exploration_schedule.anneal_ratio=0.73 \
  trainer.eval_every_n_epochs=50 \
  trainer.eval_schedule.type=multi_phase \
  trainer.eval_schedule.eval_on_first_epoch=false \
  trainer.eval_schedule.eval_on_last_epoch=true \
  'trainer.eval_schedule.phases=[{period: 10, until_epoch: 100}, {period: 50, until_epoch: -100}, {period: 20}]' \
  trainer.lr_scheduler.min_lr_ratio=0.05 \
  ++task.plot_style.plots.task_rate_evolution.enabled=false \
  ++task.plot_style.plots.task_power_scatter.enabled=false \
  ++task.plot_style.plots.selector_rate_correlation.max_networks_per_dataset=1 \
  "wandb.enabled=${WANDB_ENABLED}"

echo ""
echo "Stage 4 complete. Best checkpoints land in:"
echo "  $EXPERIMENT_DIR/trainer_chkpts/best_models/"
echo "This is a STATISTICAL reproduction — for the exact paper numbers eval the"
echo "shipped checkpoint with ./05_evaluate_checkpoint.sh."
