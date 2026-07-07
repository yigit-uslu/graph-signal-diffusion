#!/bin/bash
# Diffusion training on ALL 4 medium-large outdoor densities at a SINGLE
# reference r_min (0.6) — NO r_min sweep. 4 sub-datasets, 25,600 samples.
# (Companion to train_dev_all_density_all_rmin.sh, which sweeps 5 r_min.)
#
# =========================================================================
# MERGED U-GNN ARM: sophisticated-oarfish-9 (WRA task) + sociable-frigatebird-619
# (the most recent U-GNN, non-temporal innovations only).
# =========================================================================
#
# Base (from sophisticated-oarfish-9):
#   WRA task + dataset, learned node downsampling, DDIM (500 steps, 100
#   sampling, eta=0.2), AdamW lr=1e-4, eps/L2, the selector recipe
#   (temperature_min=0.50, linear temperature schedule, entropy_reg_weight=0).
#   Difference here: single fixed r_min -> model_cond_channels=2 (set in the
#   dataset config), and the ds8 backbone below.
#
# Ported from sociable-frigatebird-619 (NON-TEMPORAL — these act on WRA's
# single snapshot; the temporal features are dropped, see below):
#   1. ds8 backbone (model config ugnn_wra_v3_ds8_norm_act_head):
#        downsample by 2 at every level (gamma=[2,2,2], 8x compression),
#        max_gnn_stride=2, num_bottleneck_layers=2, dropout=0.10.
#   2. norm_act_head output head: LayerNorm -> SiLU -> Linear, zero-init
#        (eps-parameterization friendly). Baked into the model config.
#   3. Selector Gumbel exploration: selector_exploration_noise=1.0 (linear),
#        annealed via trainer.selector_exploration_schedule (warmup 2%,
#        anneal 73%). Prevents premature learned-selector collapse.
#   4. Fraction-based selector schedules at frigatebird's 0.02 / 0.73 shape
#        (longer explore-then-exploit) for BOTH temperature and exploration.
#
# DROPPED (temporal / time-series — inert or inapplicable at WRA T=1):
#   cond-encoder gated + temporal attention, backbone temporal-mixer
#   self-attention, cross-attention conditioning over a past window, RevIN,
#   and leverage_scores (an offline diagnostic, not a training feature).
#
# Usage:
#   bash scripts/wra/diffusion/train_dev_all_density.sh
#   WRA_CUDA_DEVICE=1 bash scripts/wra/diffusion/train_dev_all_density.sh
#   # dry-run the resolved config without training:
#   bash scripts/wra/diffusion/train_dev_all_density.sh --cfg job --resolve

set -euo pipefail
export HYDRA_FULL_ERROR=1

WRA_PRESET=medium-large_outdoor_all_density \
WRA_CUDA_DEVICE="${WRA_CUDA_DEVICE:-1}" \
bash scripts/wra/diffusion/run_train.sh \
    model@task.model=ugnn_wra_v3_ds8_norm_act_head \
    trainer@task.trainer=trainer_wra_v3_learned_ds4 \
    dataset.num_workers=4 \
    dataset.persistent_workers=false \
    dataset.pin_memory=false \
    \
    `# ── selector recipe (sophisticated-oarfish-9) ──` \
    model.config.pooling_config.entropy_reg_weight=0.0 \
    model.config.pooling_config.selector_kwargs.temperature_min=0.50 \
    model.config.pooling_config.selector_kwargs.temperature_schedule=linear \
    \
    `# ── NEW: selector Gumbel exploration (sociable-frigatebird-619) ──` \
    model.config.pooling_config.selector_exploration_noise=1.0 \
    model.config.pooling_config.selector_exploration_noise_min=0.0 \
    model.config.pooling_config.selector_exploration_noise_schedule=linear \
    \
    `# ── NEW: fraction-based schedules, frigatebird 0.02 / 0.73 shape ──` \
    trainer.selector_temperature_schedule.warmup_ratio=0.02 \
    trainer.selector_temperature_schedule.anneal_ratio=0.73 \
    +trainer.selector_exploration_schedule.enabled=true \
    +trainer.selector_exploration_schedule.warmup_ratio=0.02 \
    +trainer.selector_exploration_schedule.anneal_ratio=0.73 \
    \
    `# ── eval schedule + plotting (sophisticated-oarfish-9) ──` \
    trainer.eval_every_n_epochs=50 \
    trainer.eval_schedule.type=multi_phase \
    trainer.eval_schedule.eval_on_first_epoch=false \
    trainer.eval_schedule.eval_on_last_epoch=true \
    'trainer.eval_schedule.phases=[{period: 10, until_epoch: 100}, {period: 50, until_epoch: -100}, {period: 20}]' \
    trainer.lr_scheduler.min_lr_ratio=0.05 \
    ++task.plot_style.plots.task_rate_evolution.enabled=false \
    ++task.plot_style.plots.task_power_scatter.enabled=false \
    ++task.plot_style.plots.selector_rate_correlation.max_networks_per_dataset=1 \
    wandb.enabled=true \
    "$@"
