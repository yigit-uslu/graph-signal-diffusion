#!/bin/bash
# This is the script that I actually run for quick model/trainer ablations.

## ***************** ARM L: 4-level [1,2,2,2] + 3-layer bottleneck + linear anneal ***************** #
WRA_PRESET=large_outdoor_low_density WRA_CUDA_DEVICE=1 \
bash scripts/wra/diffusion/run_train.sh \
    dataset@task.dataset=wra_large_outdoor_low_density \
    model@task.model=ugnn_wra_v3_ds4 \
    trainer@task.trainer=trainer_wra_v3_learned_ds4 \
    dataset.num_workers=8 \
    dataset.persistent_workers=false \
    dataset.pin_memory=false \
    'model.config.channel_multipliers=[1,1,1,1]' \
    'model.config.pooling_config.gamma=[1,2,2,2]' \
    model.config.gnn_config.max_gnn_stride=2 \
    model.config.num_bottleneck_layers=3 \
    model.config.pooling_config.entropy_reg_weight=0.0 \
    model.config.pooling_config.selector_kwargs.temperature_min=0.20 \
    model.config.pooling_config.selector_kwargs.temperature_schedule=linear \
    trainer.selector_temperature_schedule.anneal_ratio=0.90 \
    trainer.eval_every_n_epochs=100 \
    trainer.lr_scheduler.min_lr_ratio=0.05


## ***************** ARM K: 3-level [1,2,2] + 3-layer bottleneck + linear anneal * #
# Redesigned architecture motivated by image U-Net principles:
#   - gamma=[1,2,2] with channel_multipliers=[1,1,1]: 3 encoder levels instead of 4.
#     Level 0 (γ=1): process@800, Level 1 (γ=2): process@800+pool→400,
#     Level 2 (γ=2): process@400+pool→200. Same node-ops as [1,2,1,2] but shifts
#     compute from mid-resolution processing to bottleneck global reasoning.
#   - num_bottleneck_layers=3: heavy processing at 200 nodes (cheap) for global
#     coordination, matching image U-Net bottleneck design.
#   - Linear temperature schedule with anneal_ratio=0.90: slower, constant-rate
#     decay stabilizes L0 gradient cascade from interleaved architecture.
#     T>0.40 for ~75% of training vs ~40% with cosine anneal_ratio=0.60.
#   - No entropy reg, post L1 fix, T_min=0.20.
# WRA_PRESET=large_outdoor_low_density WRA_CUDA_DEVICE=0 \
# bash scripts/wra/diffusion/run_train.sh \
#     dataset@task.dataset=wra_large_outdoor_low_density \
#     model@task.model=ugnn_wra_v3_ds4 \
#     trainer@task.trainer=trainer_wra_v3_learned_ds4 \
#     dataset.num_workers=8 \
#     dataset.persistent_workers=false \
#     dataset.pin_memory=false \
#     'model.config.channel_multipliers=[1,1,1]' \
#     'model.config.pooling_config.gamma=[1,2,2]' \
#     model.config.gnn_config.max_gnn_stride=2 \
#     model.config.num_bottleneck_layers=3 \
#     model.config.pooling_config.entropy_reg_weight=0.0 \
#     model.config.pooling_config.selector_kwargs.temperature_min=0.20 \
#     model.config.pooling_config.selector_kwargs.temperature_schedule=linear \
#     trainer.selector_temperature_schedule.anneal_ratio=0.90 \
#     trainer.eval_every_n_epochs=100 \
#     trainer.lr_scheduler.min_lr_ratio=0.05


## ***************** ARM J: ARM I + no temperature annealing (T=1.0 constant) ** #
# Same architecture as ARM I (gamma=[1,2,1,2], max_gnn_stride=2, no entropy reg,
# post L1 fix) but disables the temperature schedule entirely.
# Selector stays at T=1.0 for the full run — purely stochastic selection driven
# only by diffusion loss gradients through the STE. Tests whether the selector
# can learn useful structure without any temperature-driven commitment.
# WRA_PRESET=large_outdoor_low_density WRA_CUDA_DEVICE=0 \
# bash scripts/wra/diffusion/run_train.sh \
#     dataset@task.dataset=wra_large_outdoor_low_density \
#     model@task.model=ugnn_wra_v3_ds4 \
#     trainer@task.trainer=trainer_wra_v3_learned_ds4 \
#     dataset.num_workers=4 \
#     dataset.persistent_workers=false \
#     dataset.pin_memory=false \
#     model.config.pooling_config.entropy_reg_weight=0.0 \
#     'model.config.pooling_config.gamma=[1,2,1,2]' \
#     model.config.gnn_config.max_gnn_stride=2 \
#     trainer.selector_temperature_schedule.enabled=false \
#     trainer.eval_every_n_epochs=200 \
#     trainer.lr_scheduler.min_lr_ratio=0.05


## ***************** ARM I: ARM H + gamma rearrangement [1,2,1,2] + stride caps * #
# Same config as ARM H (warm floor, slow anneal, no entropy reg, post L1 fix)
# but rearranges gamma from [1,1,2,2] to [1,2,1,2] so that:
#   Level 0 (γ=1): process at full resolution → Level 1 (γ=2, L0): select+pool
#   Level 2 (γ=1): process on L0 subset      → Level 3 (γ=2, L1): select+pool
# This places one processing block before each selection, giving each selector
# GNN-refined features rather than raw/just-selected features.
# In [1,1,2,2], both selections are back-to-back with no intermediate processing.
# max_gnn_stride=2 and max_pool_stride=2 cap the effective receptive field to
# prevent over-reach at deeper levels (L1 stride_pre=4 would otherwise give
# 8-hop GNN reach, covering the entire 800-node graph).
# WRA_PRESET=large_outdoor_low_density WRA_CUDA_DEVICE=1 \
# bash scripts/wra/diffusion/run_train.sh \
#     dataset@task.dataset=wra_large_outdoor_low_density \
#     model@task.model=ugnn_wra_v3_ds4 \
#     trainer@task.trainer=trainer_wra_v3_learned_ds4 \
#     dataset.num_workers=4 \
#     dataset.persistent_workers=false \
#     dataset.pin_memory=false \
#     model.config.pooling_config.entropy_reg_weight=0.0 \
#     'model.config.pooling_config.gamma=[1,2,1,2]' \
#     model.config.pooling_config.selector_kwargs.temperature_min=0.20 \
#     model.config.gnn_config.max_gnn_stride=2 \
#     trainer.selector_temperature_schedule.anneal_ratio=0.60 \
#     trainer.eval_every_n_epochs=50 \
#     trainer.lr_scheduler.min_lr_ratio=0.05


## ***************** ARM H: warm floor + slow anneal (post-fix) ***************** #
# Tests whether maintaining selector flexibility improves generation quality
# now that L1 actually receives diffusion loss gradients (post-fix).
# ARM G's heatmaps showed rich timestep-aware L1 selection fading as T→0.05.
# T_min=0.20 keeps the selector in the regime where temporal structure was
# strongest (sigmoid(score/0.20) ≈ 0.92 at score=0.5 — committed preferences
# respected, but marginal nodes retain meaningful stochasticity).
# anneal_ratio=0.60 gives L1 more exploration time before reaching floor.
# vs ARM G: T_min 0.05→0.20, anneal_ratio 0.35→0.60, all else identical.
# WRA_PRESET=large_outdoor_low_density WRA_CUDA_DEVICE=0 \
# bash scripts/wra/diffusion/run_train.sh \
#     dataset@task.dataset=wra_large_outdoor_low_density \
#     model@task.model=ugnn_wra_v3_ds4 \
#     trainer@task.trainer=trainer_wra_v3_learned_ds4 \
#     dataset.num_workers=4 \
#     dataset.persistent_workers=false \
#     dataset.pin_memory=false \
#     model.config.pooling_config.entropy_reg_weight=0.0 \
#     model.config.pooling_config.selector_kwargs.temperature_min=0.20 \
#     trainer.selector_temperature_schedule.anneal_ratio=0.60 \
#     trainer.eval_every_n_epochs=100 \
#     trainer.lr_scheduler.min_lr_ratio=0.05


## ***************** ARM G: ARM F rerun after L1 selector gradient fix ***************** #
# Identical config to ARM F (skink without entropy penalty).
# ARM F ran with a bug where the deepest (L1) selector received no gradient
# from the diffusion loss — only the entropy penalty. The encoder returned
# skip_features[-1] (pre-pooling) to the bottleneck instead of x_enc
# (post-pooling), severing the STE gradient path for L1's selection_mask.
# ARM G reruns the same config to measure the impact of the fix.
# WRA_PRESET=large_outdoor_low_density WRA_CUDA_DEVICE=1 \
# bash scripts/wra/diffusion/run_train.sh \
#     dataset@task.dataset=wra_large_outdoor_low_density \
#     model@task.model=ugnn_wra_v3_ds4 \
#     trainer@task.trainer=trainer_wra_v3_learned_ds4 \
#     dataset.num_workers=4 \
#     dataset.persistent_workers=false \
#     dataset.pin_memory=false \
#     model.config.pooling_config.entropy_reg_weight=0.0 \
#     trainer.selector_temperature_schedule.anneal_ratio=0.35 \
#     trainer.eval_every_n_epochs=100 \
#     trainer.lr_scheduler.min_lr_ratio=0.05


## ***************** ARM F: skink without entropy penalty ***************** #
# Identical to skink (Arm E) but with entropy_reg_weight=0.
# Tests whether the entropy penalty contributes anything beyond what the
# temperature anneal already provides. If Arm F matches skink, the penalty
# is dead weight and can be removed from the hyperparameter space.
# NOTE: Ran with L1 selector gradient bug — see ARM G for the fixed rerun.
# WRA_PRESET=large_outdoor_low_density WRA_CUDA_DEVICE=0 \
# bash scripts/wra/diffusion/run_train.sh \
#     dataset@task.dataset=wra_large_outdoor_low_density \
#     model@task.model=ugnn_wra_v3_ds4 \
#     trainer@task.trainer=trainer_wra_v3_learned_ds4 \
#     dataset.num_workers=4 \
#     dataset.persistent_workers=false \
#     dataset.pin_memory=false \
#     model.config.pooling_config.entropy_reg_weight=0.0 \
#     trainer.selector_temperature_schedule.anneal_ratio=0.35 \
#     trainer.eval_every_n_epochs=100 \
#     trainer.lr_scheduler.min_lr_ratio=0.05


## ***************** ARM E: moose schedule + compressed anneal ***************** #
# Keep moose's T_init=1.0 and warmup (preserves beneficial random-mask dropout
# phase in ep 0-700), but compress the anneal window (0.60 → 0.35) so the
# selector commits at ep ~1740 with 58% LR remaining (vs moose's ep 3040 at 16%).
# T: 1.0 → warmup → 0.05 (cosine, anneal over 35% of remaining steps)
# Only ONE parameter change from moose: anneal_ratio=0.35
# WRA_PRESET=large_outdoor_low_density WRA_CUDA_DEVICE=1 \
# bash scripts/wra/diffusion/run_train.sh \
#     dataset@task.dataset=wra_large_outdoor_low_density \
#     model@task.model=ugnn_wra_v3_ds4 \
#     trainer@task.trainer=trainer_wra_v3_learned_ds4 \
#     dataset.num_workers=4 \
#     dataset.persistent_workers=false \
#     dataset.pin_memory=false \
#     trainer.selector_temperature_schedule.anneal_ratio=0.35 \
#     trainer.eval_every_n_epochs=100 \
#     trainer.lr_scheduler.min_lr_ratio=0.05


## ***************** ARM D: low T_init + no warmup + cold floor ***************** #
# Revised after three-way analysis: warm floor (T_min=0.30) produces entropy
# plateau at ~0.23 and worse post-floor validation than moose (T_min=0.05).
# # Arm D keeps moose's cold floor but starts lower (T_init=0.7), eliminates
# # warmup, and uses a 50% anneal window so the transition happens earlier
# # while LR is still high.
# # T: 0.7 → 0.05 (cosine, no warmup, anneal over 50% of steps → floor at ~ep 2100)
# # vs moose: T_init=1.0, warmup=0.02, anneal=0.60, T_min=0.05, depth_scale=1.0
# WRA_PRESET=large_outdoor_low_density WRA_CUDA_DEVICE=0 \
# bash scripts/wra/diffusion/run_train.sh \
#     dataset@task.dataset=wra_large_outdoor_low_density \
#     model@task.model=ugnn_wra_v3_ds4 \
#     trainer@task.trainer=trainer_wra_v3_learned_ds4 \
#     dataset.num_workers=4 \
#     dataset.persistent_workers=false \
#     dataset.pin_memory=false \
#     model.config.pooling_config.selector_kwargs.temperature=0.7 \
#     model.config.pooling_config.selector_kwargs.temperature_min=0.05 \
#     model.config.pooling_config.entropy_reg_weight=1e-2 \
#     model.config.pooling_config.entropy_reg_depth_scale=1.0 \
#     trainer.selector_temperature_schedule.warmup_ratio=0.0 \
#     trainer.selector_temperature_schedule.anneal_ratio=0.50 \
#     trainer.eval_every_n_epochs=100


# ## ***************** "eccentric-bird-496" ablation: ARM C ***************** #
# WRA_PRESET=large_outdoor_low_density WRA_CUDA_DEVICE=1 \
# bash scripts/wra/diffusion/run_train.sh \
#     dataset@task.dataset=wra_large_outdoor_low_density \
#     model@task.model=ugnn_wra_v3_ds4 \
#     trainer@task.trainer=trainer_wra_v3_learned_ds4 \
#     dataset.num_workers=4 \
#     dataset.persistent_workers=false \
#     dataset.pin_memory=false \
#     model.config.pooling_config.selector_kwargs.temperature_min=0.3 \
#     model.config.pooling_config.entropy_reg_depth_scale=0.5 \
#     trainer.selector_temperature_schedule.warmup_ratio=0.01 \
#     trainer.selector_temperature_schedule.anneal_ratio=0.35 \
#     trainer.eval_every_n_epochs=100




# ## ***************** "mutant-moose-562 ablation: ARM B ***************** #
# WRA_PRESET=large_outdoor_low_density WRA_CUDA_DEVICE=0 \
# bash scripts/wra/diffusion/run_train.sh \
#     dataset@task.dataset=wra_large_outdoor_low_density \
#     model@task.model=ugnn_wra_v3_ds4 \
#     trainer@task.trainer=trainer_wra_v3_learned_ds4 \
#     dataset.num_workers=4 \
#     dataset.persistent_workers=false \
#     dataset.pin_memory=false \
#     model.config.pooling_config.selector_kwargs.temperature_min=0.3 \
#     model.config.pooling_config.entropy_reg_weight=5e-3 \
#     model.config.pooling_config.entropy_reg_depth_scale=0.5



# ***************** "monumental-mushroom-152" ***************** #
# # Baseline: no pooling, no compactness, learned selector with stride-based candidate generation, no selector temperature annealing, no packed score mode, no sparse pointwise execution, no compact skip cache.
# WRA_PRESET=large_outdoor_low_density WRA_CUDA_DEVICE=0 \
# bash scripts/wra/diffusion/run_train.sh \
#   dataset@task.dataset=wra_large_outdoor_low_density \
#   model@task.model=ugnn_wra_v3_ds4 \
#   trainer@task.trainer=trainer_wra_v3_learned_ds4 \
#   trainer.name=gdm_wra_large_outdoor_v3_nods_nocompact \
#   model.config.pooling_config.gamma='[1,1,1,1]' \
#   model.config.pooling_config.selection_method=stride \
#   trainer.selector_temperature_schedule.enabled=false \
#   model.config.gnn_config.pointwise_sparse_mode=off \
#   model.config.compact_skip_cache=false \
#   model.config.pooling_config.selector_packed_score_mode=off \
#   dataset.num_workers=4 \
#   dataset.persistent_workers=false \
#   dataset.pin_memory=false \

# # ***************** "mutant-moose-562" ***************** #
# WRA_PRESET=large_outdoor_low_density WRA_CUDA_DEVICE=1 \
# bash scripts/wra/diffusion/run_train.sh \
#     dataset@task.dataset=wra_large_outdoor_low_density \
#     model@task.model=ugnn_wra_v3_ds4 \
#     trainer@task.trainer=trainer_wra_v3_learned_ds4 \
#     dataset.num_workers=4 \
#     dataset.persistent_workers=false \
#     dataset.pin_memory=false \


# WRA_PRESET=medium WRA_CUDA_DEVICE=1 \
# bash scripts/wra/diffusion/run_train.sh \
#     model@task.model=ugnn_wra_v3_nopool_learned_ds2 \
#     trainer@task.trainer=trainer_wra_debug_v3_learned_ds2 \
#     trainer.name=gdm_wra_medium_v3_learned_ds2
