#!/bin/bash
# Diffusion training on ALL density scenarios (ultra-low, low, mid, high)
# across all 5 r_min datasets (r_min = 0.4, 0.5, 0.6, 0.7, 0.8).
# 20 sub-datasets total (4 densities x 5 r_min), 128,000 samples.


## ***************** Ablation 1: Baseline (GPU 0) ************************** #
# Mirrors abiding-potoo-545 hyperparameters exactly.
# Only change: dataset is all_density_all_rmin.
# WRA_PRESET=medium-large_outdoor_all_density_all_rmin WRA_CUDA_DEVICE=0 \
# bash scripts/wra/diffusion/run_train.sh \
#     model@task.model=ugnn_wra_v3_ds4 \
#     trainer@task.trainer=trainer_wra_v3_learned_ds4 \
#     dataset.num_workers=4 \
#     dataset.persistent_workers=false \
#     dataset.pin_memory=false \
#     'model.config.channel_multipliers=[1,1,1]' \
#     'model.config.pooling_config.gamma=[1,2,2]' \
#     model.config.gnn_config.max_gnn_stride=2 \
#     model.config.num_bottleneck_layers=3 \
#     model.config.pooling_config.entropy_reg_weight=0.0 \
#     model.config.pooling_config.selector_kwargs.temperature_min=0.50 \
#     model.config.pooling_config.selector_kwargs.temperature_schedule=linear \
#     trainer.selector_temperature_schedule.anneal_ratio=0.90 \
#     trainer.eval_every_n_epochs=50 \
#     trainer.eval_schedule.type=multi_phase \
#     trainer.eval_schedule.eval_on_first_epoch=false \
#     trainer.eval_schedule.eval_on_last_epoch=true \
#     'trainer.eval_schedule.phases=[{period: 10, until_epoch: 100}, {period: 50, until_epoch: -100}, {period: 20}]' \
#     trainer.lr_scheduler.min_lr_ratio=0.05 \
#     +task.plot_style.plots.task_rate_evolution.enabled=false \
#     +task.plot_style.plots.task_power_scatter.enabled=false \
#     +task.plot_style.plots.selector_rate_correlation.max_networks_per_dataset=1 \
#     wandb.enabled=true


## ***************** Ablation 2: Depth over Width (GPU 1) ****************** #
# Narrower U-Net (32 base, [1,1,2]) + deeper GNN (K=3, num_layers=3),
# 4 bottleneck layers, lower temperature floor (0.20).
# Lower LR (3e-5), faster decay (decay_fraction=0.6).
# 2000 max-epochs (LR floor ~epoch 1200, 800 epochs of low-LR settling).
WRA_PRESET=medium-large_outdoor_all_density_all_rmin WRA_CUDA_DEVICE=1 \
bash scripts/wra/diffusion/run_train.sh \
    model@task.model=ugnn_wra_v3_ds4 \
    trainer@task.trainer=trainer_wra_v3_learned_ds4 \
    dataset.num_workers=4 \
    dataset.persistent_workers=false \
    dataset.pin_memory=false \
    'model.config.channel_multipliers=[1,1,2]' \
    model.config.base_channels=32 \
    'model.config.pooling_config.gamma=[1,2,2]' \
    model.config.gnn_config.max_gnn_stride=2 \
    model.config.gnn_config.num_layers=3 \
    model.config.gnn_config.K=3 \
    model.config.num_bottleneck_layers=4 \
    model.config.pooling_config.entropy_reg_weight=0.0 \
    model.config.pooling_config.selector_kwargs.temperature_min=0.20 \
    model.config.pooling_config.selector_kwargs.temperature_schedule=linear \
    trainer.selector_temperature_schedule.anneal_ratio=0.90 \
    trainer.max_epochs=2000 \
    trainer.eval_every_n_epochs=50 \
    trainer.eval_schedule.type=multi_phase \
    trainer.eval_schedule.eval_on_first_epoch=false \
    trainer.eval_schedule.eval_on_last_epoch=true \
    'trainer.eval_schedule.phases=[{period: 20, until_epoch: 100}, {period: 50, until_epoch: -200}, {period: 20}]' \
    trainer.optimizer.learning_rate=3e-5 \
    trainer.lr_scheduler.decay_fraction=0.6 \
    trainer.lr_scheduler.min_lr_ratio=0.05 \
    +task.plot_style.plots.task_rate_evolution.enabled=false \
    +task.plot_style.plots.task_power_scatter.enabled=false \
    +task.plot_style.plots.selector_rate_correlation.max_networks_per_dataset=1 \
    wandb.enabled=true
