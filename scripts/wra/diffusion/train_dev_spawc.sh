#!/bin/bash
# This is the script that I actually run for quick model/trainer ablations.

WRA_PRESET=medium-large_outdoor_low_density WRA_CUDA_DEVICE=0 \
bash scripts/wra/diffusion/run_train.sh \
    model@task.model=ugnn_wra_v3_ds4 \
    trainer@task.trainer=trainer_wra_v3_learned_ds4 \
    dataset.num_workers=8 \
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
    trainer.eval_every_n_epochs=20 \
    trainer.lr_scheduler.min_lr_ratio=0.05 \
    wandb.enabled=false \