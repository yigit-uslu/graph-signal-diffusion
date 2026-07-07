#!/bin/bash
# SP500 experiment launch script for learned downsampling ablations.
# Analogous to scripts/wra/diffusion/train_dev.sh.
# Run from the project root:
#   bash scripts/stock/sp500/train_dev.sh

export HYDRA_FULL_ERROR=1

# Graph variant: correlation threshold 0.7 with sector bonus 0.05.
# Override dataset.root to point to the 0.7 variant (sp500_cleaned.yaml defaults to 0.6).
# SP500_ROOT=data/sp500/cleaned_drop_incomplete_min_coverage_0.95_corr_0.7_sector_bonus_0.05
SP500_ROOT=data/sp500/cleaned_drop_incomplete_min_coverage_0.95_corr_0.6_sector_bonus_0.05



# ***************** SP500 ARM F T=5: lightweight denoiser, same conditioning ************* #
# Tests whether D-v2f's 6M-param diamond denoiser is overparameterized.
# The 5-way checkpoint analysis of stoic-polecat-491 shows forecast quality
# peaks at ep 275 and degrades steadily — the model fits early then overfits.
#
# Changes vs D-v2f (stoic-polecat-491):
#   - 3-level U-Net [1,2,1] instead of 5-level [1,2,4,2,1]:
#     channels [64, 128, 64] vs [64, 128, 256, 128, 64].
#     Without graph pooling the removed levels were redundant
#     (same resolution, just extra sequential processing).
#   - 2 attention heads (temporal mixer + cross-attention) instead of 4.
#   - batch_size=64 (from 32): peak 128ch fits more under AMP.
#   - max_epochs=1000 (from 2000): D-v2f's best CRPS was ep 275/2000;
#     1000 epochs is generous headroom with faster convergence expected.
#   - Denser early eval schedule to catch the CRPS minimum precisely.
#
# Kept from D-v2f: 3-chunk split, RevIN, DDIM (eta=0.2, 500 timesteps,
# 100 sampling steps), conditioning encoder (4x128ch, dilations [1,2,4,8]),
# time_embed_dim=128, K=2 ChebConv, 2 GNN layers per block, per_layer
# temporal mixer schedule.
#
# Estimated params: ~1.5M (vs 6.03M). Training ~3-4x faster.
#
# Hypothesis: the lightweight denoiser reaches comparable early-epoch
# forecast quality (CRPS, MAE) since the conditioning path — which
# dominates the useful capacity — is unchanged. Late-epoch degradation
# is reduced because there is less backbone capacity to overfit with.
# Stylized-fact fidelity (kurtosis, eigval1) may peak lower than D-v2f's
# ep 2000 since the denoiser has less capacity to memorize the marginal.
#
# Control: D-v2f (stoic-polecat-491).
# Runs: pragmatic-trout-751
# dropout=0.01
# CUDA_VISIBLE_DEVICES=0 python -m graph_signal_diffusion.cli.train \
#     task=stock_price_forecasting_v3 \
#     dataset.root=${SP500_ROOT} \
#     dataset.past_window=20 \
#     dataset.future_window=5 \
#     dataset.batch_size=64 \
#     dataset.batch_size_val=200 \
#     dataset.num_workers=4 \
#     dataset.persistent_workers=true \
#     dataset.pin_memory=true \
#     dataset.n_split_chunks=3 \
#     dataset.shuffle_val=false \
#     dataset.standardize_target_in_x_for_revin=true \
#     diffusion.revin=true \
#     model.config.gnn_config.dropout=$dropout \
#     model.config.gnn_config.temporal_mixer.schedule=per_layer \
#     model.config.gnn_config.temporal_mixer.kernel_size=3 \
#     'model.config.gnn_config.temporal_mixer.dilations=[1,1]' \
#     model.config.gnn_config.temporal_mixer.use_pointwise=true \
#     model.config.gnn_config.temporal_mixer.causal=false \
#     model.config.gnn_config.temporal_mixer.attention.enabled=true \
#     model.config.gnn_config.temporal_mixer.attention.num_heads=2 \
#     model.config.gnn_config.temporal_mixer.attention.dropout=$dropout \
#     model.config.gnn_config.temporal_mixer.attention.max_timesteps=5 \
#     'model.config.embedding_config.cond.shared_encoder.temporal.hidden_channels=[128,128,128,128]' \
#     model.config.embedding_config.cond.shared_encoder.temporal.num_layers=4 \
#     model.config.embedding_config.cond.shared_encoder.temporal.mixer.causal=false \
#     'model.config.embedding_config.cond.shared_encoder.temporal.mixer.dilations=[1,2,4,8]' \
#     model.config.embedding_config.cond.shared_encoder.temporal.output.mode=time_varying \
#     model.config.embedding_config.cond.shared_encoder.temporal.output.time_varying.method=none \
#     model.config.embedding_config.cond.block_fusion.mode=cross_attention \
#     model.config.embedding_config.cond.block_fusion.cross_attention.heads=2 \
#     model.config.embedding_config.cond.block_fusion.cross_attention.dropout=$dropout \
#     model.config.embedding_config.cond.block_fusion.cross_attention.bias=true \
#     model.config.embedding_config.cond.block_fusion.cross_attention.causal=false \
#     'model.config.channel_multipliers=[1,2,1]' \
#     'model.config.pooling_config.gamma=[1,1,1]' \
#     model.config.pooling_config.selection_method=stride \
#     model.config.num_bottleneck_layers=1 \
#     model.config.compact_skip_cache=false \
#     model.config.gnn_config.pointwise_sparse_mode=off \
#     model.config.pooling_config.selector_packed_score_mode=off \
#     trainer.selector_temperature_schedule.enabled=false \
#     trainer.max_epochs=5000 \
#     trainer.eval_every_n_epochs=50 \
#     trainer.eval_schedule.phases='[{period: 50, until_epoch: 500}, {period: 100, until_epoch: 1000}, {period: 250}]' \
#     trainer.save_checkpoint_every_n_epochs=500 \
#     trainer.use_amp=true \
#     trainer.n_samples_per_input=10 \
#     trainer.max_num_val_batches=2 trainer.max_num_train_val_batches=1 \
#     wandb.enabled=true


# ***************** SP500 ARM G T=10: ARM F-v2 + longer horizon + dense eval ************ #
# ARM F-v2 compare-baselines (6 checkpoints, n_samples=20) showed forecast quality
# peaks at ep 500–1500, with coverage degrading monotonically after ep 1000.
# ARM G reruns the same config with:
#   - future_window=10 (vs 5): longer forecast horizon.
#   - Denser eval schedule in the [500, 1500] sweet spot (every 25 epochs).
#   - Checkpoint saves every 100 epochs (vs 500) to enable fine-grained selection.
#   - max_epochs=2500 (ARM F-v2 plateaued by ep 1500; 2500 is generous headroom).
#
# Everything else identical to ARM F-v2 (amiable-rattlesnake-49): corr_0.7 graph,
# [1,2,1] denoiser, 1.75M params, dropout=0.01, RevIN, DDIM, cross-attention.
#
# Eval schedule: [{period: 50, until: 500}, {period: 25, until: 1500}, {period: 250}]
#
# Control: ARM F-v2 (amiable-rattlesnake-49).
# Runs: fiery-capuchin-409
# SP500_ROOT_07=data/sp500/cleaned_drop_incomplete_min_coverage_0.95_corr_0.7_sector_bonus_0.05
# dropout=0.01
# CUDA_VISIBLE_DEVICES=0 python -m graph_signal_diffusion.cli.train \
#     task=stock_price_forecasting_v3 \
#     dataset.root=${SP500_ROOT_07} \
#     dataset.past_window=20 \
#     dataset.future_window=10 \
#     dataset.batch_size=32 \
#     dataset.batch_size_val=200 \
#     dataset.num_workers=4 \
#     dataset.persistent_workers=true \
#     dataset.pin_memory=true \
#     dataset.n_split_chunks=3 \
#     dataset.shuffle_val=false \
#     dataset.standardize_target_in_x_for_revin=true \
#     diffusion.revin=true \
#     model.config.gnn_config.dropout=$dropout \
#     model.config.gnn_config.temporal_mixer.schedule=per_layer \
#     model.config.gnn_config.temporal_mixer.kernel_size=3 \
#     'model.config.gnn_config.temporal_mixer.dilations=[1,1]' \
#     model.config.gnn_config.temporal_mixer.use_pointwise=true \
#     model.config.gnn_config.temporal_mixer.causal=false \
#     model.config.gnn_config.temporal_mixer.attention.enabled=true \
#     model.config.gnn_config.temporal_mixer.attention.num_heads=2 \
#     model.config.gnn_config.temporal_mixer.attention.dropout=$dropout \
#     model.config.gnn_config.temporal_mixer.attention.max_timesteps=10 \
#     'model.config.embedding_config.cond.shared_encoder.temporal.hidden_channels=[128,128,128,128]' \
#     model.config.embedding_config.cond.shared_encoder.temporal.num_layers=4 \
#     model.config.embedding_config.cond.shared_encoder.temporal.mixer.causal=false \
#     'model.config.embedding_config.cond.shared_encoder.temporal.mixer.dilations=[1,2,4,8]' \
#     model.config.embedding_config.cond.shared_encoder.temporal.output.mode=time_varying \
#     model.config.embedding_config.cond.shared_encoder.temporal.output.time_varying.method=none \
#     model.config.embedding_config.cond.block_fusion.mode=cross_attention \
#     model.config.embedding_config.cond.block_fusion.cross_attention.heads=2 \
#     model.config.embedding_config.cond.block_fusion.cross_attention.dropout=$dropout \
#     model.config.embedding_config.cond.block_fusion.cross_attention.bias=true \
#     model.config.embedding_config.cond.block_fusion.cross_attention.causal=false \
#     'model.config.channel_multipliers=[1,2,1]' \
#     'model.config.pooling_config.gamma=[1,1,1]' \
#     model.config.pooling_config.selection_method=stride \
#     model.config.num_bottleneck_layers=1 \
#     model.config.compact_skip_cache=false \
#     model.config.gnn_config.pointwise_sparse_mode=off \
#     model.config.pooling_config.selector_packed_score_mode=off \
#     trainer.selector_temperature_schedule.enabled=false \
#     trainer.max_epochs=2500 \
#     trainer.eval_every_n_epochs=25 \
#     trainer.eval_schedule.phases='[{period: 50, until_epoch: 250}, {period: 25, until_epoch: 750}, {period: 100}]' \
#     trainer.save_checkpoint_every_n_epochs=100 \
#     trainer.use_amp=true \
#     trainer.n_samples_per_input=10 \
#     trainer.max_num_val_batches=2 trainer.max_num_train_val_batches=1 \
#     wandb.enabled=true








# ---- ARM F-v2 (amethyst-perch-521 and steel-goshawk-355) - REVISITED amiable-rattlesnake-49 after SPAWC branch merged -------------------- #
#
# Revisiting ARM F-v2's corr_0.7 variant with the latest codebase after merging the SPAWC branch.
# Changed to base_channels=32 and channel_multipliers=[4,2,1] (64,32,16 channels) to test the inverted-pyramid denoiser architecture.
#
# Control: ARM F-v2 (amiable-rattlesnake-49) on corr_0.7.
# Runs: amethyst-perch-521 and steel-goshawk-355 (latter with ddim_eta=1.0 instead of 0.2 to test if more stochasticity helps with the sparser graph).
# SP500_ROOT_07=data/sp500/cleaned_drop_incomplete_min_coverage_0.95_corr_0.7_sector_bonus_0.05
# dropout=0.01
# ddim_eta=1.0
# CUDA_VISIBLE_DEVICES=1 python -m graph_signal_diffusion.cli.train \
#     task=stock_price_forecasting_v3 \
#     dataset.root=${SP500_ROOT_07} \
#     dataset.past_window=20 \
#     dataset.future_window=5 \
#     dataset.batch_size=64 \
#     dataset.batch_size_val=200 \
#     dataset.num_workers=4 \
#     dataset.persistent_workers=true \
#     dataset.pin_memory=true \
#     dataset.n_split_chunks=3 \
#     dataset.shuffle_val=false \
#     dataset.standardize_target_in_x_for_revin=true \
#     diffusion.revin=true \
#     diffusion.ddim_eta=$ddim_eta \
#     model.config.gnn_config.dropout=$dropout \
#     model.config.gnn_config.temporal_mixer.schedule=per_layer \
#     model.config.gnn_config.temporal_mixer.kernel_size=3 \
#     'model.config.gnn_config.temporal_mixer.dilations=[1,1]' \
#     model.config.gnn_config.temporal_mixer.use_pointwise=true \
#     model.config.gnn_config.temporal_mixer.causal=false \
#     model.config.gnn_config.temporal_mixer.attention.enabled=true \
#     model.config.gnn_config.temporal_mixer.attention.num_heads=2 \
#     model.config.gnn_config.temporal_mixer.attention.dropout=$dropout \
#     model.config.gnn_config.temporal_mixer.attention.max_timesteps=5 \
#     'model.config.embedding_config.cond.shared_encoder.temporal.hidden_channels=[128,128,128,128]' \
#     model.config.embedding_config.cond.shared_encoder.temporal.num_layers=4 \
#     model.config.embedding_config.cond.shared_encoder.temporal.mixer.causal=false \
#     'model.config.embedding_config.cond.shared_encoder.temporal.mixer.dilations=[1,2,4,8]' \
#     model.config.embedding_config.cond.shared_encoder.temporal.output.mode=time_varying \
#     model.config.embedding_config.cond.shared_encoder.temporal.output.time_varying.method=none \
#     model.config.embedding_config.cond.block_fusion.mode=cross_attention \
#     model.config.embedding_config.cond.block_fusion.cross_attention.heads=2 \
#     model.config.embedding_config.cond.block_fusion.cross_attention.dropout=$dropout \
#     model.config.embedding_config.cond.block_fusion.cross_attention.bias=true \
#     model.config.embedding_config.cond.block_fusion.cross_attention.causal=false \
#     model.config.base_channels=32 \
#     'model.config.channel_multipliers=[4,2,1]' \
#     'model.config.pooling_config.gamma=[1,1,1]' \
#     model.config.pooling_config.selection_method=stride \
#     model.config.num_bottleneck_layers=1 \
#     model.config.compact_skip_cache=false \
#     model.config.gnn_config.pointwise_sparse_mode=off \
#     model.config.pooling_config.selector_packed_score_mode=off \
#     trainer.selector_temperature_schedule.enabled=false \
#     trainer.max_epochs=2500 \
#     trainer.eval_every_n_epochs=25 \
#     trainer.eval_schedule.phases='[{period: 50, until_epoch: 250}, {period: 25, until_epoch: 750}, {period: 100}]' \
#     trainer.save_checkpoint_every_n_epochs=100 \
#     trainer.use_amp=true \
#     trainer.n_samples_per_input=10 \
#     trainer.max_num_val_batches=2 trainer.max_num_train_val_batches=1 \
#     wandb.enabled=true



# ---- ARM F-v2-d10: amethyst-perch-521 with higher dropout + calibration tracker --- #
# Same architecture / data / DDIM config as amethyst-perch-521 (η=0.2 variant),
# but with two changes targeting per-window probabilistic calibration:
#   1. dropout=0.1 (10× higher than amethyst's 0.01). Earlier diagnosis showed
#      amethyst's narrow predictive distributions (Cov@90 ≈ 0.65 vs nominal 0.90,
#      spread ~58% of GRW's) trace back to the denoiser memorizing the conditional
#      mean. DDPM/EDM literature uses dropout in [0.1, 0.13]; ARM F-v2's 0.01 was
#      unusually low.
#   2. Best-model tracker swapped from CRPS+MAE (sharpness-biased) to the
#      generalization-gap-aware composite identified in the amethyst/steel-goshawk
#      tracker-composite-sweep analysis:
#        val_loss + |val − train-val gap| + 0.3·val_return_crps
#      α=0.3 sits in the stable region where the gap-zero crossing remains the
#      global minimum. Requires the `val_loss_gap` derived metric committed
#      2026-05-14 (see docs/agent_summaries/val_loss_gap_metric_2026-05-14.md).
#
# Also bumped save_checkpoint_every_n_epochs from 100 → 50 so the new tracker's
# top picks land on saved DDIM checkpoints (avoids the "rank-1 ep not on disk"
# situation we hit with amethyst).
#
# Control: amethyst-perch-521 (same model, dropout=0.01, default CRPS+MAE tracker).
# Runs: gray-galago-398
# SP500_ROOT_07=data/sp500/cleaned_drop_incomplete_min_coverage_0.95_corr_0.7_sector_bonus_0.05
# dropout=0.1
# ddim_eta=0.2
# CUDA_VISIBLE_DEVICES=0 python -m graph_signal_diffusion.cli.train \
#     task=stock_price_forecasting_v3 \
#     dataset.root=${SP500_ROOT_07} \
#     dataset.past_window=20 \
#     dataset.future_window=5 \
#     dataset.batch_size=64 \
#     dataset.batch_size_val=200 \
#     dataset.num_workers=4 \
#     dataset.persistent_workers=true \
#     dataset.pin_memory=true \
#     dataset.n_split_chunks=3 \
#     dataset.shuffle_val=false \
#     dataset.standardize_target_in_x_for_revin=true \
#     diffusion.revin=true \
#     diffusion.ddim_eta=$ddim_eta \
#     model.config.gnn_config.dropout=$dropout \
#     model.config.gnn_config.temporal_mixer.schedule=per_layer \
#     model.config.gnn_config.temporal_mixer.kernel_size=3 \
#     'model.config.gnn_config.temporal_mixer.dilations=[1,1]' \
#     model.config.gnn_config.temporal_mixer.use_pointwise=true \
#     model.config.gnn_config.temporal_mixer.causal=false \
#     model.config.gnn_config.temporal_mixer.attention.enabled=true \
#     model.config.gnn_config.temporal_mixer.attention.num_heads=2 \
#     model.config.gnn_config.temporal_mixer.attention.dropout=$dropout \
#     model.config.gnn_config.temporal_mixer.attention.max_timesteps=5 \
#     'model.config.embedding_config.cond.shared_encoder.temporal.hidden_channels=[128,128,128,128]' \
#     model.config.embedding_config.cond.shared_encoder.temporal.num_layers=4 \
#     model.config.embedding_config.cond.shared_encoder.temporal.mixer.causal=false \
#     'model.config.embedding_config.cond.shared_encoder.temporal.mixer.dilations=[1,2,4,8]' \
#     model.config.embedding_config.cond.shared_encoder.temporal.output.mode=time_varying \
#     model.config.embedding_config.cond.shared_encoder.temporal.output.time_varying.method=none \
#     model.config.embedding_config.cond.block_fusion.mode=cross_attention \
#     model.config.embedding_config.cond.block_fusion.cross_attention.heads=2 \
#     model.config.embedding_config.cond.block_fusion.cross_attention.dropout=$dropout \
#     model.config.embedding_config.cond.block_fusion.cross_attention.bias=true \
#     model.config.embedding_config.cond.block_fusion.cross_attention.causal=false \
#     model.config.base_channels=32 \
#     'model.config.channel_multipliers=[4,2,1]' \
#     'model.config.pooling_config.gamma=[1,1,1]' \
#     model.config.pooling_config.selection_method=stride \
#     model.config.num_bottleneck_layers=1 \
#     model.config.compact_skip_cache=false \
#     model.config.gnn_config.pointwise_sparse_mode=off \
#     model.config.pooling_config.selector_packed_score_mode=off \
#     trainer.selector_temperature_schedule.enabled=false \
#     trainer.max_epochs=2500 \
#     trainer.eval_every_n_epochs=25 \
#     trainer.eval_schedule.phases='[{period: 50, until_epoch: 250}, {period: 25, until_epoch: 750}, {period: 100}]' \
#     trainer.save_checkpoint_every_n_epochs=50 \
#     trainer.best_model.metrics='[{name: val_loss, weight: 1.0, direction: minimize}, {name: val_loss_gap, weight: 1.0, direction: minimize}, {name: val_return_crps, weight: 0.3, direction: minimize}]' \
#     trainer.use_amp=true \
#     trainer.n_samples_per_input=10 \
#     trainer.max_num_val_batches=2 trainer.max_num_train_val_batches=1 \
#     wandb.enabled=true



# ---- ARM F-v2-d05: dropout-middle sweep (gray-galago dropout halved) ---------------- #
# Same architecture / data / tracker as gray-galago-398, but with dropout=0.05
# (geometric midpoint between amethyst-perch-521's 0.01 and gray-galago-398's 0.10).
#
# Motivation: gray-galago at dropout=0.10 produced a two-regime sampling pattern
# at test time — "wide" checkpoints (e.g. ep 475: Cov@90≈0.80, signed NLL gap +9.9,
# eigval ratio 0.38) and "narrow" checkpoints (e.g. ep 675/1250: Cov@90≈0.65,
# NLL gap −31, eigval ratio 0.53–0.61). amethyst at dropout=0.01 preserved factor
# structure but had weak tail calibration. This run searches the Pareto middle:
# enough noise-induction to widen the ensemble, not enough to break the
# top-eigenvalue alignment with the empirical SP500 factor.
#
# Control: gray-galago-398 (same model + tracker, dropout=0.10).
# Pair:    amethyst-perch-521 (same model + tracker, dropout=0.01).
# Runs: nice-stoat-49
# SP500_ROOT_07=data/sp500/cleaned_drop_incomplete_min_coverage_0.95_corr_0.7_sector_bonus_0.05
# dropout=0.05
# ddim_eta=0.2
# CUDA_VISIBLE_DEVICES=0 python -m graph_signal_diffusion.cli.train \
#     task=stock_price_forecasting_v3 \
#     dataset.root=${SP500_ROOT_07} \
#     dataset.past_window=20 \
#     dataset.future_window=5 \
#     dataset.batch_size=64 \
#     dataset.batch_size_val=200 \
#     dataset.num_workers=4 \
#     dataset.persistent_workers=true \
#     dataset.pin_memory=true \
#     dataset.n_split_chunks=3 \
#     dataset.shuffle_val=false \
#     dataset.standardize_target_in_x_for_revin=true \
#     diffusion.revin=true \
#     diffusion.ddim_eta=$ddim_eta \
#     model.config.gnn_config.dropout=$dropout \
#     model.config.gnn_config.temporal_mixer.schedule=per_layer \
#     model.config.gnn_config.temporal_mixer.kernel_size=3 \
#     'model.config.gnn_config.temporal_mixer.dilations=[1,1]' \
#     model.config.gnn_config.temporal_mixer.use_pointwise=true \
#     model.config.gnn_config.temporal_mixer.causal=false \
#     model.config.gnn_config.temporal_mixer.attention.enabled=true \
#     model.config.gnn_config.temporal_mixer.attention.num_heads=2 \
#     model.config.gnn_config.temporal_mixer.attention.dropout=$dropout \
#     model.config.gnn_config.temporal_mixer.attention.max_timesteps=5 \
#     'model.config.embedding_config.cond.shared_encoder.temporal.hidden_channels=[128,128,128,128]' \
#     model.config.embedding_config.cond.shared_encoder.temporal.num_layers=4 \
#     model.config.embedding_config.cond.shared_encoder.temporal.mixer.causal=false \
#     'model.config.embedding_config.cond.shared_encoder.temporal.mixer.dilations=[1,2,4,8]' \
#     model.config.embedding_config.cond.shared_encoder.temporal.output.mode=time_varying \
#     model.config.embedding_config.cond.shared_encoder.temporal.output.time_varying.method=none \
#     model.config.embedding_config.cond.block_fusion.mode=cross_attention \
#     model.config.embedding_config.cond.block_fusion.cross_attention.heads=2 \
#     model.config.embedding_config.cond.block_fusion.cross_attention.dropout=$dropout \
#     model.config.embedding_config.cond.block_fusion.cross_attention.bias=true \
#     model.config.embedding_config.cond.block_fusion.cross_attention.causal=false \
#     model.config.base_channels=32 \
#     'model.config.channel_multipliers=[4,2,1]' \
#     'model.config.pooling_config.gamma=[1,1,1]' \
#     model.config.pooling_config.selection_method=stride \
#     model.config.num_bottleneck_layers=1 \
#     model.config.compact_skip_cache=false \
#     model.config.gnn_config.pointwise_sparse_mode=off \
#     model.config.pooling_config.selector_packed_score_mode=off \
#     trainer.selector_temperature_schedule.enabled=false \
#     trainer.max_epochs=2500 \
#     trainer.eval_every_n_epochs=25 \
#     trainer.eval_schedule.phases='[{period: 50, until_epoch: 250}, {period: 25, until_epoch: 750}, {period: 100}]' \
#     trainer.save_checkpoint_every_n_epochs=50 \
#     trainer.best_model.metrics='[{name: val_loss, weight: 1.0, direction: minimize}, {name: val_loss_gap, weight: 1.0, direction: minimize}, {name: val_return_crps, weight: 0.3, direction: minimize}]' \
#     trainer.use_amp=true \
#     trainer.n_samples_per_input=10 \
#     trainer.max_num_val_batches=2 trainer.max_num_train_val_batches=1 \
#     wandb.enabled=true



# ---- ARM F-v2-d10-L4: gray-galago + one extra U-GNN level, fixed-width 64ch ------ #
# Same dropout=0.10 + tracker as gray-galago-398, but extends the U-GNN with one
# extra level at uniform width: channel_multipliers=[1,1,1,1] at
# base_channels=64 → channels [64, 64, 64, 64] across all four levels (vs
# gray-galago's inverted pyramid [128, 64, 32] at base=32).
# pooling_config.gamma=[1,1,1,1] (4-entry list to match the new depth).
# batch_size lowered 64 → 48 to keep training under the AMP memory ceiling
# (driven by the wider deep-end activations and the extra pooling level, not
# by total parameter count).
#
# Capacity comparison (measured from saved checkpoints):
#   outstanding-fox-517  total = 1,356,484 params   (encoder 575K / decoder 608K / bottleneck 42K)
#   gray-galago-398      total = 1,531,044 params   (encoder 811K / decoder 578K / bottleneck 11K)
# Outstanding-fox is actually ~11% LIGHTER overall (no wide 128-ch top level),
# but the bottleneck is ~3.9× larger (64ch deepest vs 32ch). Capacity has been
# redistributed from the top level to the deeper, graph-coarsened levels —
# not added on top.
#
# Motivation: with dropout=0.10 alone, gray-galago produced two distinct
# sampling regimes (wide vs narrow) that the val-time tracker can't
# distinguish. The hypothesis here is that gray-galago's 3-level pyramid
# concentrates capacity at the shallowest (128ch) level while the deeper
# graph-coarsened levels — where global factor-structure mixing happens —
# are starved (64ch and 32ch). A fixed-width 4-level stack reallocates that
# top-level capacity into the bottleneck, giving the deep-end enough headroom
# to represent the full conditional distribution under high dropout rather
# than collapsing onto one of two simpler modes.
#
# Control: gray-galago-398 (same dropout + tracker, 3 levels [4,2,1] @ base=32).
# Runs: outstanding-fox-517
# SP500_ROOT_07=data/sp500/cleaned_drop_incomplete_min_coverage_0.95_corr_0.7_sector_bonus_0.05
# dropout=0.1
# ddim_eta=0.2
# CUDA_VISIBLE_DEVICES=1 python -m graph_signal_diffusion.cli.train \
#     task=stock_price_forecasting_v3 \
#     dataset.root=${SP500_ROOT_07} \
#     dataset.past_window=20 \
#     dataset.future_window=5 \
#     dataset.batch_size=48 \
#     dataset.batch_size_val=200 \
#     dataset.num_workers=4 \
#     dataset.persistent_workers=true \
#     dataset.pin_memory=true \
#     dataset.n_split_chunks=3 \
#     dataset.shuffle_val=false \
#     dataset.standardize_target_in_x_for_revin=true \
#     diffusion.revin=true \
#     diffusion.ddim_eta=$ddim_eta \
#     model.config.gnn_config.dropout=$dropout \
#     model.config.gnn_config.temporal_mixer.schedule=per_layer \
#     model.config.gnn_config.temporal_mixer.kernel_size=3 \
#     'model.config.gnn_config.temporal_mixer.dilations=[1,1]' \
#     model.config.gnn_config.temporal_mixer.use_pointwise=true \
#     model.config.gnn_config.temporal_mixer.causal=false \
#     model.config.gnn_config.temporal_mixer.attention.enabled=true \
#     model.config.gnn_config.temporal_mixer.attention.num_heads=2 \
#     model.config.gnn_config.temporal_mixer.attention.dropout=$dropout \
#     model.config.gnn_config.temporal_mixer.attention.max_timesteps=5 \
#     'model.config.embedding_config.cond.shared_encoder.temporal.hidden_channels=[128,128,128,128]' \
#     model.config.embedding_config.cond.shared_encoder.temporal.num_layers=4 \
#     model.config.embedding_config.cond.shared_encoder.temporal.mixer.causal=false \
#     'model.config.embedding_config.cond.shared_encoder.temporal.mixer.dilations=[1,2,4,8]' \
#     model.config.embedding_config.cond.shared_encoder.temporal.output.mode=time_varying \
#     model.config.embedding_config.cond.shared_encoder.temporal.output.time_varying.method=none \
#     model.config.embedding_config.cond.block_fusion.mode=cross_attention \
#     model.config.embedding_config.cond.block_fusion.cross_attention.heads=2 \
#     model.config.embedding_config.cond.block_fusion.cross_attention.dropout=$dropout \
#     model.config.embedding_config.cond.block_fusion.cross_attention.bias=true \
#     model.config.embedding_config.cond.block_fusion.cross_attention.causal=false \
#     model.config.base_channels=64 \
#     'model.config.channel_multipliers=[1,1,1,1]' \
#     'model.config.pooling_config.gamma=[1,1,1,1]' \
#     model.config.pooling_config.selection_method=stride \
#     model.config.num_bottleneck_layers=1 \
#     model.config.compact_skip_cache=false \
#     model.config.gnn_config.pointwise_sparse_mode=off \
#     model.config.pooling_config.selector_packed_score_mode=off \
#     trainer.selector_temperature_schedule.enabled=false \
#     trainer.max_epochs=2500 \
#     trainer.eval_every_n_epochs=25 \
#     trainer.eval_schedule.phases='[{period: 50, until_epoch: 250}, {period: 25, until_epoch: 750}, {period: 100}]' \
#     trainer.save_checkpoint_every_n_epochs=50 \
#     trainer.best_model.metrics='[{name: val_loss, weight: 1.0, direction: minimize}, {name: val_loss_gap, weight: 1.0, direction: minimize}, {name: val_return_crps, weight: 0.3, direction: minimize}]' \
#     trainer.use_amp=true \
#     trainer.n_samples_per_input=10 \
#     trainer.max_num_val_batches=2 trainer.max_num_train_val_batches=1 \
#     wandb.enabled=true



# ---- DS8 (4-level [1,2,2,2] learned downsampling): outstanding-fox + LDS ----------- #
# All architectural + training settings now live in:
#   - conf/task/stock_price_forecasting_v3_learned_ds8.yaml
#   - conf/model/ugnn_sp500_v3_ds8.yaml
#   - conf/trainer/trainer_sp500_v3_learned_ds8.yaml
#
# Summary of what's baked into those configs (vs outstanding-fox-517):
#   - 4-level encoder with gamma=[1,2,2,2]; node counts 468 → 234 → 117 → 58
#   - Learned NodeSelector (STE), selector_skip_gamma1_levels=true, packed
#     score path auto, linear temperature anneal (warmup_ratio=0.02,
#     anneal_ratio=0.90) — inherited from learned_ds4
#   - max_gnn_stride=2 caps StridedTAGConv γ per level
#   - num_bottleneck_layers=2 (matching encoder gnn_config.num_layers=2)
#   - dropout=0.10 across GNN, temporal-attn, cross-attn cond fusion
#   - Per-layer temporal mixer with self-attention (heads=2, T_max=5)
#   - Wide cond encoder (4 × 128ch, dilations=[1,2,4,8]); cross-attention fusion
#   - max_epochs=2500, eval every 25, save every 50, 3-term tracker composite
#
# Watch-signals (eig1_ratio correction note 2026-05-18):
#   - Over-smoothing canary: val_eigval1_ratio → ~0.9 + val_return_spread ↓
#   - Under-mixing canary: val_eigval1_ratio < 0.5
#   - Selector collapse: entropy aux loss → 0 quickly
#
# Control: outstanding-fox-517 (gamma=[1,1,1,1] fixed-stride, bottleneck=1).
# Planned follow-up ablations (override gamma at the CLI):
#   - gamma=[2,2,2,2]  no full-res block, 16x compression
#   - gamma=[1,2,2]   3 levels, matches sophisticated-oarfish-9 in WRA
#   - gamma=[1,1,2,2] delayed pooling (non-canonical)
# Runs: charming-nuthatch-603 (failed: selector cond T mismatch, fixed in ds8 yaml), tomato-mule-215
# SP500_ROOT_07=data/sp500/cleaned_drop_incomplete_min_coverage_0.95_corr_0.7_sector_bonus_0.05
# ddim_eta=0.2
# CUDA_VISIBLE_DEVICES=0 python -m graph_signal_diffusion.cli.train \
#     task=stock_price_forecasting_v3_learned_ds8 \
#     dataset.root=${SP500_ROOT_07} \
#     dataset.past_window=20 \
#     dataset.future_window=5 \
#     dataset.batch_size=48 \
#     dataset.batch_size_val=200 \
#     dataset.num_workers=4 \
#     dataset.persistent_workers=true \
#     dataset.pin_memory=true \
#     dataset.n_split_chunks=3 \
#     dataset.shuffle_val=false \
#     dataset.standardize_target_in_x_for_revin=true \
#     diffusion.revin=true \
#     diffusion.ddim_eta=$ddim_eta \
#     wandb.enabled=true



# ---- DS8-pyramid: canonical U-Net 3-level [2,2,2] with channel doubling ----------- #
# Sibling of tomato-mule-215 (DS8, gamma=[1,2,2,2], uniform 64ch). Drops the
# full-resolution L0 block and replaces uniform-width with DDPM/Stable
# Diffusion-style channel doubling at each pool.
#
# Architecture diffs vs the DS8 yaml defaults (which tomato-mule-215 uses):
#   - pooling_config.gamma=[2,2,2]   (was [1,2,2,2])
#       3 encoder levels, pool at every transition (canonical U-Net).
#       Node counts: 468 → 234 → 117 → 58. Same 8x bottleneck compression as
#       tomato-mule but reached one level shallower.
#   - channel_multipliers=[1,2,4]    (was [1,1,1,1])
#       Channels per level: [32, 64, 128]. Concentrates capacity at the
#       coarser, more semantically-rich levels — matches the image U-Net
#       trade-off where the bottleneck is widest.
#   - base_channels=32               (was 64)
#       Keeps the deepest level at 128ch (32 × mult[-1]=4). Top level is
#       lighter (32ch on 468 nodes) than tomato-mule's 64ch on 468 — net
#       activation memory is comparable or slightly lower.
#
# Everything else identical to DS8 / tomato-mule-215:
#   - selection_method=learned, max_gnn_stride=2, num_bottleneck_layers=2
#   - dropout=0.10, temperature_min=0.5, linear anneal, 3-term tracker
#   - max_epochs=2500, learned_projection cond at past_window=20 → 5
#
# What this ablation isolates:
#   - "Is the full-resolution L0 block doing useful work, or is canonical
#     pool-at-every-level enough?" (the gamma question)
#   - "Does channel doubling at the bottleneck outperform uniform width?"
#     (the capacity-distribution question)
# Both axes change together, so a single comparison vs tomato-mule answers
# both jointly. If DS8-pyramid wins, future SP500 configs adopt the
# canonical U-Net pattern; if tomato-mule wins, the uniform-width / full-res-
# first design has merit specifically for graph signals.
#
# Control: tomato-mule-215 (DS8, gamma=[1,2,2,2], uniform 64ch, base=64).
# Runs: infrared-coyote-480
# SP500_ROOT_07=data/sp500/cleaned_drop_incomplete_min_coverage_0.95_corr_0.7_sector_bonus_0.05
# ddim_eta=0.2
# CUDA_VISIBLE_DEVICES=1 python -m graph_signal_diffusion.cli.train \
#     task=stock_price_forecasting_v3_learned_ds8 \
#     dataset.root=${SP500_ROOT_07} \
#     dataset.past_window=20 \
#     dataset.future_window=5 \
#     dataset.batch_size=48 \
#     dataset.batch_size_val=200 \
#     dataset.num_workers=4 \
#     dataset.persistent_workers=true \
#     dataset.pin_memory=true \
#     dataset.n_split_chunks=3 \
#     dataset.shuffle_val=false \
#     dataset.standardize_target_in_x_for_revin=true \
#     diffusion.revin=true \
#     diffusion.ddim_eta=$ddim_eta \
#     model.config.base_channels=32 \
#     'model.config.channel_multipliers=[1,2,4]' \
#     'model.config.pooling_config.gamma=[2,2,2]' \
#     wandb.enabled=true



# ---- DS8-wide-uniform-128: 3-level uniform-128ch, 3 layers/depth ----------------- #
# Builds on outstanding-fox-517 (uniform-width 4-level beats inverted pyramid) and
# tomato-mule-215 (DS8 with learned NodeSelector, 8x bottleneck compression) with a
# substantial capacity bump targeted at closing tomato-mule's eig1_ratio gap
# (eig1_r=0.596 vs outstanding-fox 0.671 vs real-data target ≈1.0).
#
# Architecture diffs vs the DS8 yaml defaults (which tomato-mule-215 uses):
#   - base_channels=128                (was 64)            channels doubled
#   - gnn_config.num_layers=3          (was 2)             one more GNN layer per
#                                                          encoder/decoder block
#   - num_bottleneck_layers=3          (was 2)             matching encoder num_layers
#   - pooling_config.gamma=[2,2,2]     (was [1,2,2,2])
#       3 encoder levels, pool at every transition (canonical U-Net topology).
#       Drops the full-resolution γ=1 block from the front of the stack.
#       Node counts: 468 → 234 → 117 → 58. Same 8x bottleneck compression as
#       tomato-mule, reached one level shallower.
#   - channel_multipliers=[1,1,1]      (was [1,1,1,1])
#       Uniform-width 128ch across all 3 levels. Echoes outstanding-fox's
#       finding that uniform-width beats the inverted pyramid.
#       Direct pairing with infrared-coyote-480 (γ=[2,2,2] + [1,2,4] @ base=32 →
#       channels [32,64,128]): same graph schedule + same 128ch bottleneck,
#       different capacity allocation at the input side.
#   - batch_size=24                    (was 48)
#       Activation memory grows ~2x per element (channels 2x, layers 1.5x,
#       levels 0.75x); bs=48 → 24 keeps total activations roughly constant.
#
# Param estimate: ~6M params (vs tomato-mule 1.4M). ~4.5x bump.
# Per-epoch wallclock ~10x tomato-mule (5x per-step compute × 2x more steps
# from halved batch size). At 2500 epochs, plan accordingly.
#
# Everything else inherited from the DS8 yaml chain (tomato-mule-215 settings):
#   - selection_method=learned, NodeSelector v3, T_min=0.50, linear anneal_ratio=0.90
#   - max_gnn_stride=2 (caps StridedTAGConv γ at the bottleneck)
#   - dropout=0.10 across GNN, temporal-attn, cross-attn cond fusion
#   - Per-layer temporal mixer with self-attention (heads=2, T_max=5)
#   - Wide cond encoder (4 × 128ch, dilations=[1,2,4,8]); cross-attention fusion
#   - learned_projection cond (REQUIRED under selection_method=learned)
#   - max_epochs=2500, eval every 25, save every 50, 3-term tracker composite
#     (val_loss + |val_loss_gap| + 0.3·val_return_crps)
#
# Hypotheses to test:
#   1. Does doubled bottleneck-mixing capacity close the eig1_ratio gap
#      (tomato-mule 0.60 → ~0.75 or beyond, approaching real-data target 1.0)?
#   2. Does coverage lift toward nominal (cov_90 0.67 → ~0.78)?
#   3. Does val_loss beat outstanding-fox's 0.335?
#
# If yes → DS8's 8x compression was capacity-bound, not topology-bound.
# If no  → under-mixing is structural; next try γ=[1,1,2,2] (delayed pooling)
#          or γ=[1,2,2,1] (intermediate compression).
#
# Watch-signals (eig1_ratio correction note 2026-05-18):
#   - Over-smoothing canary: val_eigval1_ratio → ~0.9 + val_return_spread ↓
#   - Selector collapse: entropy aux loss → 0 quickly
#   - Memorization risk at 6M params on ~13.5K supervised forecast targets:
#       train_val gap blowout → tracker's |gap| term should keep best-pick stable
#
# Controls:
#   - tomato-mule-215   (same DS8 yaml; γ=[1,2,2,2], base=64, num_layers=2)
#   - outstanding-fox-517 (γ=[1,1,1,1] stride; base=64; uniform-width baseline)
#   - infrared-coyote-480 (γ=[2,2,2] pyramid [1,2,4] @ base=32; same graph
#       schedule + bottleneck width but pyramid capacity allocation)
#
# Runs: cyber-doberman-15
# SP500_ROOT_07=data/sp500/cleaned_drop_incomplete_min_coverage_0.95_corr_0.7_sector_bonus_0.05
# ddim_eta=0.2
# CUDA_VISIBLE_DEVICES=0 python -m graph_signal_diffusion.cli.train \
#     task=stock_price_forecasting_v3_learned_ds8 \
#     dataset.root=${SP500_ROOT_07} \
#     dataset.past_window=20 \
#     dataset.future_window=5 \
#     dataset.batch_size=24 \
#     dataset.batch_size_val=200 \
#     dataset.num_workers=4 \
#     dataset.persistent_workers=true \
#     dataset.pin_memory=true \
#     dataset.n_split_chunks=3 \
#     dataset.shuffle_val=false \
#     dataset.standardize_target_in_x_for_revin=true \
#     diffusion.revin=true \
#     diffusion.ddim_eta=$ddim_eta \
#     model.config.base_channels=128 \
#     'model.config.channel_multipliers=[1,1,1]' \
#     'model.config.pooling_config.gamma=[2,2,2]' \
#     model.config.gnn_config.num_layers=3 \
#     'model.config.gnn_config.temporal_mixer.dilations=[1,1,1]' \
#     model.config.num_bottleneck_layers=3 \
#     wandb.enabled=true



# ---- DS8-wide-uniform-128 + DDPM-style output head (QUEUED) -------------- #
# Sibling of cyber-doberman-15. SAME DS8-wide-uniform-128 backbone (3-level
# uniform 128ch, gamma=[2,2,2], num_layers=3, num_bottleneck_layers=3, ~6M
# params). The only architectural change is the output head; training budget
# is also shortened.
#
# Output head diff (vs cyber-doberman-15's default bare-Linear head):
#   model=ugnn_sp500_v3_ds8_norm_act_head replaces UGNNDecoder.output_proj
#   with nn.Sequential(LayerNorm(128), SiLU, Linear(128, 1)) and
#   zero-initializes the final Linear's weight + bias.
#
#   What this changes about training:
#     - Zero-init under parameterization='eps' makes the untrained model
#       emit ε̂≈0, so the very first DDIM step is approximately the identity.
#       This is the canonical convergence-friendly init used across DDPM,
#       EDM, ADM, and DiT — removes the "unlearn random ε̂ predictions"
#       phase at the start of training (typically hundreds of wasted epochs).
#     - LayerNorm(128) + SiLU before the final Linear stabilizes the input
#       scale to the head regardless of how the terminal GNN.output_proj
#       (a 128→128 Linear) shifts the feature distribution. Matches the
#       DDPM/EDM/ADM convention of Norm→Activation→Projection at the head.
#     - LayerNorm is CHANNEL-ONLY: it normalizes per-(B, T, N) position
#       across the C=128 feature dim. T (timesteps) and N (nodes) are NOT
#       mixed by the norm — same semantics as DiT.
#     - Adds ~256 params (2×128 LayerNorm γ/β) on top of cyber-doberman-15's
#       ~6M — negligible.
#
# Training budget diff:
#   trainer.max_epochs=1000 (vs cyber-doberman-15's inherited 2500).
#   Rationale: the amethyst/cyber-doberman lineage shows CRPS plateaus by
#   ep ~500-1500 and degrades after; with a more convergence-friendly init
#   we expect the useful range to peak earlier still. 1000 epochs covers
#   the full expected useful window with headroom. The inherited
#   eval_schedule [{period:50 until:250}, {period:25 until:750}, {period:100}]
#   from the DS8 trainer yaml is unchanged and gives dense coverage in the
#   [25, 750] sweet spot, plus 3 evals (ep 850, 950, 1000) in the tail.
#
# Implementation reference:
#   - OutputHeadConfig dataclass + _build_output_head helper:
#       src/graph_signal_diffusion/models/ugnn/ugnn.py
#   - Preset:
#       src/graph_signal_diffusion/conf/model/ugnn_sp500_v3_ds8_norm_act_head.yaml
#   - Design, tests, edge cases:
#       docs/agent_summaries/output_head_config_2026-05-24.md
#
# Hypotheses:
#   1. Zero-init eliminates the early-epoch CRPS instability seen in
#      cyber-doberman-15's first few hundred epochs.
#   2. The LayerNorm+SiLU+zero-init head closes some of cyber-doberman-15's
#      remaining eig1_ratio gap vs the real-data target ≈1.0 by giving the
#      optimizer a better-conditioned starting point.
#   3. Best-checkpoint val_loss improves vs cyber-doberman-15 at fewer
#      total training epochs, or holds with substantially less compute.
#
# Control: cyber-doberman-15 (same backbone, default bare-Linear head, 2500
# epochs). Pairs on GPU 1 with cyber-doberman-15 on GPU 0 — both can run
# concurrently. To launch this ARM in isolation, comment out the
# cyber-doberman-15 block above.
#
# Runs: fervent-cicada-140 (3 splits) and turquoise-chimpanzee-597 (5 splits)
# SP500_ROOT_07=data/sp500/cleaned_drop_incomplete_min_coverage_0.95_corr_0.7_sector_bonus_0.05
# ddim_eta=0.2
# n_split_chunks=5
# CUDA_VISIBLE_DEVICES=1 python -m graph_signal_diffusion.cli.train \
#     task=stock_price_forecasting_v3_learned_ds8 \
#     model@task.model=ugnn_sp500_v3_ds8_norm_act_head \
#     dataset.root=${SP500_ROOT_07} \
#     dataset.past_window=20 \
#     dataset.future_window=5 \
#     dataset.batch_size=24 \
#     dataset.batch_size_val=200 \
#     dataset.num_workers=4 \
#     dataset.persistent_workers=true \
#     dataset.pin_memory=true \
#     dataset.n_split_chunks=$n_split_chunks \
#     dataset.shuffle_val=false \
#     dataset.standardize_target_in_x_for_revin=true \
#     diffusion.revin=true \
#     diffusion.ddim_eta=$ddim_eta \
#     model.config.base_channels=128 \
#     'model.config.channel_multipliers=[1,1,1]' \
#     'model.config.pooling_config.gamma=[2,2,2]' \
#     model.config.gnn_config.num_layers=3 \
#     'model.config.gnn_config.temporal_mixer.dilations=[1,1,1]' \
#     model.config.num_bottleneck_layers=3 \
#     trainer.max_epochs=1000 \
#     wandb.enabled=true



# ---- DS8 + norm-act head + RevIN-bias correction: F3 ablation ----------------- #
# Ablates the new RevIN σ-bias-compensation knobs (revin_blend_weight + the
# ${revin_alpha:w,b} auto-α resolver) on top of the F3 backbone. The intent
# is to isolate the RevIN fix as the single moving variable vs F3.
#
# Changes vs fervent-cicada-140 (F3):
#   - diffusion.revin_blend_weight=0.7 (default 1.0 in conf/diffusion/ddim.yaml):
#       σ'_w = 0.7·σ_cond_w + 0.3·1.0
#     Blends the past-window σ_cond_w estimator with the per-stock long-run
#     σ=1.0 (=1.0 by construction after upstream per-stock standardization).
#     Structurally reduces the ~18% downward bias in 20-day σ estimates
#     (where the small-sample Gaussian factor ~0.96 compounds with
#     volatility-clustering bias ~15% on top).
#   - diffusion.revin_sigma_correction auto-resolved via ${revin_alpha:w,b}:
#       b ≈ 0.82 = E[σ_cond_w] on the SP500 train set, injected by the
#       SP500 builder as dataset_info.sigma_cond_mean_train.
#       α(0.7, 0.82) = 1 / (1 - 0.7·(1 - 0.82)) ≈ 1.143
#       The resolver fires at config-resolution time; the literal numeric
#       value is saved in .hydra/config.yaml so checkpoint reload does not
#       refire the resolver. No CLI override needed — the resolver chain
#       in ddim.yaml automatically picks up the new w.
#   - eval_schedule middle phase period changed 25 → 50.
#       Two-phase schedule [{period:50, until:750}, {period:100}] preserves
#       dense coverage around F3's best-epoch window (ep 475) but saves
#       ~10 evals across the run since val_loss is already smooth there.
#   - CUDA_VISIBLE_DEVICES=0 (F3 trained on GPU 1). Lets this run in
#       parallel with the turquoise-chimpanzee-597 block above on GPU 1.
#
# Kept from F3 verbatim:
#   - Backbone: DS8-wide-uniform-128 (base=128, gamma=[2,2,2],
#     channel_multipliers=[1,1,1], num_layers=3, num_bottleneck_layers=3,
#     temporal_mixer.dilations=[1,1,1]) → ~6M params
#   - Output head: norm-act-head (LayerNorm+SiLU+Linear, zero-init)
#     via model@task.model=ugnn_sp500_v3_ds8_norm_act_head
#   - Data: corr_0.7 graph, past_window=20, future_window=5, bs=24,
#     n_split_chunks=3, standardize_target_in_x_for_revin=true
#   - Diffusion: revin=true, ddim_eta=0.2, 500 train / 100 sampling steps
#   - Trainer: max_epochs=1000, dropout=0.10 (inherited), AMP, n_samples=10,
#     val_loss + |val_loss_gap| + 0.3·val_return_crps composite tracker
#
# Hypotheses to test:
#   1. Spread ratio (gen/real) lifts from F3 ep475's ~0.58 toward 1.0.
#   2. Cov@90 lifts from F3 ep475's 0.66 toward nominal 0.90.
#   3. val_loss and val_loss_gap stay comparable to F3's (the σ correction
#      is symmetric — applied at single source so norm and denorm match;
#      training-time MSE in RevIN space is approximately scale-invariant).
#   4. val_kurtosis_gen unchanged ~13-15 (no new tail mechanism introduced;
#      RevIN affects scale, not shape).
#
# Control: fervent-cicada-140 (same backbone, no RevIN bias correction).
# Pair:    turquoise-chimpanzee-597 (same backbone, n_split_chunks=5).
#
# Runs: colorful-wapiti-739
# SP500_ROOT_07=data/sp500/cleaned_drop_incomplete_min_coverage_0.95_corr_0.7_sector_bonus_0.05
# ddim_eta=0.2
# n_split_chunks=3
# revin_blend_weight=0.7
# CUDA_VISIBLE_DEVICES=0 python -m graph_signal_diffusion.cli.train \
#     task=stock_price_forecasting_v3_learned_ds8 \
#     model@task.model=ugnn_sp500_v3_ds8_norm_act_head \
#     dataset.root=${SP500_ROOT_07} \
#     dataset.past_window=20 \
#     dataset.future_window=5 \
#     dataset.batch_size=24 \
#     dataset.batch_size_val=200 \
#     dataset.num_workers=4 \
#     dataset.persistent_workers=true \
#     dataset.pin_memory=true \
#     dataset.n_split_chunks=$n_split_chunks \
#     dataset.shuffle_val=false \
#     dataset.standardize_target_in_x_for_revin=true \
#     diffusion.revin=true \
#     diffusion.ddim_eta=$ddim_eta \
#     diffusion.revin_blend_weight=$revin_blend_weight \
#     model.config.base_channels=128 \
#     'model.config.channel_multipliers=[1,1,1]' \
#     'model.config.pooling_config.gamma=[2,2,2]' \
#     model.config.gnn_config.num_layers=3 \
#     'model.config.gnn_config.temporal_mixer.dilations=[1,1,1]' \
#     model.config.num_bottleneck_layers=3 \
#     trainer.max_epochs=1000 \
#     'trainer.eval_schedule.phases=[{period: 50, until_epoch: 500}, {period: 100}]' \
#     wandb.enabled=true



# ---- colorful-wapiti-739 + selector-temperature-anneal DISABLED: GPU 1 pair ---- #
# Pair-ablation of colorful-wapiti-739. Backbone, data, diffusion, RevIN-fix
# settings IDENTICAL — only difference is the selector temperature schedule.
#
# Change vs colorful-wapiti-739:
#   - trainer.selector_temperature_schedule.enabled=false
#       Short-circuits _resolve_selector_temperature_schedule() so
#       temperature_warmup_steps and temperature_anneal_steps are NOT written
#       to model.config.pooling_config.selector_kwargs. NodeSelector falls
#       back to its yaml defaults (temperature_anneal_steps=0 from
#       ugnn_sp500_v3_ds4.yaml), which routes _current_temperature() to
#       return self.temperature (=1.0) on every forward — no annealing.
#       Selector stays in fully-soft (T=1.0) mode for the entire 1000 epochs.
#   - CUDA_VISIBLE_DEVICES=1 (wapiti runs on GPU 0). Both can run in parallel.
#
# Everything else verbatim from colorful-wapiti-739 (see comment block above):
#   - DS8-wide-uniform-128 backbone + norm-act head
#   - corr_0.7 graph, past_window=20, future_window=5, n_split_chunks=3
#   - bs=24, max_epochs=1000, AMP, RevIN, ddim_eta=0.2
#   - diffusion.revin_blend_weight=0.7 → α≈1.143 via auto-resolver
#   - Two-phase eval schedule [{50, until:500}, {100}]
#
# Hypothesis: colorful-wapiti's anneal drives T from 1.0 → 0.5 over 90% of
# training; the late-stage hard-selection regime (T=0.5, ep ≥ 902) is what
# F3 spent its last ~100 epochs in. The question this arm answers:
#   - Is the RevIN-fix benefit independent of the selector anneal? If this
#     arm matches wapiti's spread-ratio/cov_90 lift, then YES — the σ
#     correction is the dominant lever and the selector schedule is
#     orthogonal.
#   - If this arm lags wapiti's spread/coverage but matches at val_loss,
#     then the selector hardening at T=0.5 was buying calibration that
#     wapiti gets "for free" from its anneal — the anneal matters.
#   - If this arm BEATS wapiti, soft selection (T=1.0 throughout) lets the
#     selector keep more diverse pooling patterns, which preserves more
#     conditioning signal end-to-end. Suggests F3's anneal was harmful.
#
# Control: colorful-wapiti-739 (same arm with anneal enabled, GPU 0).
#
# Runs: charming-rat-237
# SP500_ROOT_07=data/sp500/cleaned_drop_incomplete_min_coverage_0.95_corr_0.7_sector_bonus_0.05
# ddim_eta=0.2
# n_split_chunks=3
# revin_blend_weight=0.7
# CUDA_VISIBLE_DEVICES=1 python -m graph_signal_diffusion.cli.train \
#     task=stock_price_forecasting_v3_learned_ds8 \
#     model@task.model=ugnn_sp500_v3_ds8_norm_act_head \
#     dataset.root=${SP500_ROOT_07} \
#     dataset.past_window=20 \
#     dataset.future_window=5 \
#     dataset.batch_size=24 \
#     dataset.batch_size_val=200 \
#     dataset.num_workers=4 \
#     dataset.persistent_workers=true \
#     dataset.pin_memory=true \
#     dataset.n_split_chunks=$n_split_chunks \
#     dataset.shuffle_val=false \
#     dataset.standardize_target_in_x_for_revin=true \
#     diffusion.revin=true \
#     diffusion.ddim_eta=$ddim_eta \
#     diffusion.revin_blend_weight=$revin_blend_weight \
#     model.config.base_channels=128 \
#     'model.config.channel_multipliers=[1,1,1]' \
#     'model.config.pooling_config.gamma=[2,2,2]' \
#     model.config.gnn_config.num_layers=3 \
#     'model.config.gnn_config.temporal_mixer.dilations=[1,1,1]' \
#     model.config.num_bottleneck_layers=3 \
#     trainer.selector_temperature_schedule.enabled=false \
#     trainer.max_epochs=1000 \
#     'trainer.eval_schedule.phases=[{period: 50, until_epoch: 500}, {period: 100}]' \
#     wandb.enabled=true



# ---- colorful-wapiti-739 + cond-encoder boost + Gumbel exploration: GPU 1 ---- #
# Ablation of colorful-wapiti-739. RevIN, backbone, dataset, selector temperature
# anneal IDENTICAL to wapiti. Three changes — all targeted at giving the model
# more temporal/exploration capacity without adding GNN layers:
#
#   (1) Cond encoder gated activation (WaveNet/DiffWave-style):
#       cond.shared_encoder.temporal.mixer.gated=true
#       The cond encoder is 4 layers × kernel=5 × dilations=[1,2,4,8] over the
#       past_window=20 conditioning signal. Switching gated on replaces the
#       SiLU activation with tanh(W_f·x) * sigmoid(W_g·x) and adds a 1x1
#       gate_proj per mixer. The sigmoid acts as a learned per-channel
#       per-timestep soft selector; tanh provides the bounded feature.
#       Per cond mixer: +16,768 params. 8 cond mixers × ~16.8K = +134K params.
#
#   (2) Cond encoder temporal self-attention spanning the full past window:
#       cond.shared_encoder.temporal.mixer.attention={enabled: true,
#         num_heads: 2, dropout: 0.1, max_timesteps: 20}
#       Bidirectional self-attention applied per-node over the T_cond=20
#       past-window timesteps inside every cond mixer. Pairwise time-context
#       complements the dilated TCN's local-then-pooled receptive field.
#       max_timesteps=20 must match past_window (positional embedding size).
#       Per cond mixer: +68,864 params (q/k/v/o + LayerNorm + pos_emb at
#       C=128, heads=2). 8 cond mixers × ~68.9K = +551K params.
#
#   (3) Selector Gumbel exploration noise with linear anneal:
#       pooling_config.selector_exploration_noise=1.0
#       pooling_config.selector_exploration_noise_min=0.0
#       pooling_config.selector_exploration_noise_schedule=linear
#       trainer.selector_exploration_schedule.enabled=true (warmup/anneal
#       ratios mirror selector_temperature_schedule: 2% warmup, 90% anneal).
#       Adds ε·gumbel to selector scores during training only, where ε starts
#       at 1.0 (comparable to score std ≈ 1.0 in wapiti's metrics) and linearly
#       anneals to 0 by the end of the anneal window. Active only in
#       soft/ste selection modes — diffusion sampling and eval are unaffected.
#       No new parameters.
#
#   (4) Learning rate bump 5×:
#       trainer.optimizer.learning_rate=5e-4  (wapiti: 1e-4)
#       The added params from (1)+(2) and the exploration noise from (3) both
#       mean the model has more to learn AND a noisier loss landscape. A 5×
#       lr is the cleanest matching response — keeps the cosine schedule's
#       warmup_ratio=0.01 and min_lr_ratio=0.05 unchanged (both are ratios
#       that scale with the base lr, so peak/min lr both scale 5×). Risk:
#       early-training instability with the same 1% warmup; watch grad_norm
#       and train_loss in the first ~50 epochs.
#
# Combined: ~+685K params (~+13% on top of wapiti's ~5M model-only count).
#
# Everything else verbatim from colorful-wapiti-739 (see comment block above):
#   - DS8-wide-uniform-128 backbone + norm-act head, 3 layers/depth
#   - corr_0.7 graph, past_window=20, future_window=5, n_split_chunks=3
#   - bs=24, max_epochs=1000, AMP, RevIN, ddim_eta=0.2
#   - diffusion.revin_blend_weight=0.7 → α≈1.115 via auto-resolver
#   - Selector temperature anneal ENABLED (same as wapiti, NOT rat)
#   - Two-phase eval schedule [{50, until:500}, {100}]
#
# Hypotheses to test:
#   1. Cond encoder gated+attention lifts the multi-day signal that wapiti
#      already shows a small edge on (cum_T_direction_accuracy ~0.515 at
#      ep 700, vs F3's ~0.50). The cond encoder is the deepest temporal
#      stack in the model — most leverage per parameter added.
#   2. Gumbel exploration prevents the selector from collapsing onto a
#      single high-score subset early in training. Wapiti's selector
#      entropy at ep 700 was ~0.74 (decreasing as T anneals); without
#      exploration the selector commits to its first-found basin.
#   3. The combination shouldn't hurt val_loss / val_loss_gap (RevIN
#      correction is unchanged; cond capacity is additive). Risk:
#      overfitting given SP500's ~1860 training samples — watch
#      val_loss_gap drift after ep 300.
#
# Control: colorful-wapiti-739 (cond mixers ungated, no cond self-attn, no
#          exploration noise; same selector anneal).
#
# GPU note: wapiti training is still occupying GPU 0; this arm runs on
# GPU 1 (rat freed it). Both can run in parallel.
#
# Plumbing references:
#   - docs/agent_summaries/cond_encoder_gated_attention_wiring_2026-05-27.md
#     (cond gated + cond attention wiring through EmbeddingConfig)
#   - cli/train.py:_resolve_selector_exploration_schedule (translates
#     trainer.selector_exploration_schedule.{warmup_ratio,anneal_ratio} to
#     concrete step counts based on total_steps)
#
# Runs: carmine-moose-32
# SP500_ROOT_07=data/sp500/cleaned_drop_incomplete_min_coverage_0.95_corr_0.7_sector_bonus_0.05
# ddim_eta=0.2
# n_split_chunks=3
# revin_blend_weight=0.7
# CUDA_VISIBLE_DEVICES=1 python -m graph_signal_diffusion.cli.train \
#     task=stock_price_forecasting_v3_learned_ds8 \
#     model@task.model=ugnn_sp500_v3_ds8_norm_act_head \
#     dataset.root=${SP500_ROOT_07} \
#     dataset.past_window=20 \
#     dataset.future_window=5 \
#     dataset.batch_size=24 \
#     dataset.batch_size_val=200 \
#     dataset.num_workers=4 \
#     dataset.persistent_workers=true \
#     dataset.pin_memory=true \
#     dataset.n_split_chunks=$n_split_chunks \
#     dataset.shuffle_val=false \
#     dataset.standardize_target_in_x_for_revin=true \
#     diffusion.revin=true \
#     diffusion.ddim_eta=$ddim_eta \
#     diffusion.revin_blend_weight=$revin_blend_weight \
#     model.config.base_channels=128 \
#     'model.config.channel_multipliers=[1,1,1]' \
#     'model.config.pooling_config.gamma=[2,2,2]' \
#     model.config.gnn_config.num_layers=3 \
#     'model.config.gnn_config.temporal_mixer.dilations=[1,1,1]' \
#     model.config.num_bottleneck_layers=3 \
#     +model.config.embedding_config.cond.shared_encoder.temporal.mixer.gated=true \
#     +model.config.embedding_config.cond.shared_encoder.temporal.mixer.attention.enabled=true \
#     +model.config.embedding_config.cond.shared_encoder.temporal.mixer.attention.num_heads=2 \
#     +model.config.embedding_config.cond.shared_encoder.temporal.mixer.attention.dropout=0.1 \
#     +model.config.embedding_config.cond.shared_encoder.temporal.mixer.attention.max_timesteps=20 \
#     model.config.pooling_config.selector_exploration_noise=1.0 \
#     model.config.pooling_config.selector_exploration_noise_min=0.0 \
#     model.config.pooling_config.selector_exploration_noise_schedule=linear \
#     +trainer.selector_exploration_schedule.enabled=true \
#     +trainer.selector_exploration_schedule.warmup_ratio=0.02 \
#     +trainer.selector_exploration_schedule.anneal_ratio=0.90 \
#     trainer.optimizer.learning_rate=5e-4 \
#     trainer.max_epochs=1000 \
#     'trainer.eval_schedule.phases=[{period: 50, until_epoch: 500}, {period: 100}]' \
#     wandb.enabled=true



# ---- carmine-moose-32 + lr DOWN 5× (= wapiti/5 = 2e-5): GPU 1 pair ---- #
# Pair-ablation of carmine-moose-32. Backbone, data, diffusion, RevIN-fix,
# cond gated, cond attention, Gumbel exploration settings IDENTICAL —
# only difference is the optimizer learning rate.
#
# lr bracketing across the three ARMs (all otherwise wapiti + cond boost
# + Gumbel exploration):
#   colorful-wapiti-739:           lr = 1e-4   (1×, baseline; no cond boost,
#                                              no Gumbel exploration)
#   carmine-moose-32:              lr = 5e-4   (5× UP from wapiti)
#   THIS ARM:                      lr = 2e-5   (5× DOWN from wapiti,
#                                               25× DOWN from carmine-moose)
#
# Change vs carmine-moose-32:
#   - trainer.optimizer.learning_rate=2e-5  (carmine-moose: 5e-4)
#     The cosine schedule's warmup_ratio=0.01 and min_lr_ratio=0.05 are
#     unchanged (both are ratios that scale with the base lr). Effective
#     min_lr at decay end: 0.05 × 2e-5 = 1.0e-6.
#   - CUDA_VISIBLE_DEVICES=0 (launched once GPU 0 freed up;
#     carmine-moose-32 still occupies GPU 1).
#
# Everything else verbatim from carmine-moose-32 (see comment block above):
#   - DS8-wide-uniform-128 backbone + norm-act head
#   - corr_0.7 graph, past_window=20, future_window=5, n_split_chunks=3
#   - bs=24, max_epochs=1000, AMP, RevIN, ddim_eta=0.2
#   - diffusion.revin_blend_weight=0.7 → α≈1.1151 via auto-resolver
#   - cond shared encoder: gated=true, attention {enabled=true, heads=2,
#     dropout=0.1, max_timesteps=20}
#   - Gumbel exploration noise: 1.0 → 0 linear, warmup 2%, anneal 90%
#   - Selector temperature anneal enabled (T: 1.0 → 0.5)
#   - Two-phase eval schedule [{50, until:500}, {100}]
#
# Hypothesis: carmine-moose-32 pushes lr UP 5× to match the added capacity
# (cond gated + attn) and the noisier loss landscape (Gumbel exploration).
# This ARM pushes lr DOWN 5× from wapiti, hypothesizing the opposite: the
# added capacity may need MORE careful learning, not less. Specifically:
#   - If carmine-moose composite beats both wapiti and this ARM, 5× UP is
#     the right response — capacity needs faster movement to use it.
#   - If this ARM beats both, the added capacity needs careful learning;
#     wapiti's 1× was already too fast for the boosted model.
#   - If neither beats wapiti, the cond-boost / Gumbel changes are not
#     net-positive at any lr in this bracket — the lr knob is not the
#     bottleneck.
#
# Risk vs carmine-moose-32: lr=2e-5 is 0.2× wapiti's. Training is ~5× slower
# per gradient step in expectation. With 1000 epochs and the same cosine
# decay shape, the late phase may underfit (lr too low to escape local
# minima). Watch composite trajectory: if it's still descending steeply
# at ep 900-1000, max_epochs is too tight at this lr; for follow-ups bump
# to max_epochs=2000 or stretch the eval schedule.
#
# Control: carmine-moose-32 (same arm with lr 5e-4).
# Reference baseline: colorful-wapiti-739 (no cond boost, no Gumbel, lr 1e-4).
#
# Plumbing references:
#   - docs/agent_summaries/cond_encoder_gated_attention_wiring_2026-05-27.md
#   - docs/agent_summaries/wapiti_cond_boost_exploration_launcher_2026-05-27.md
#   - docs/agent_summaries/selector_exploration_noise_logging_2026-05-27.md
#     (NodeSelector exploration application + diag emission; ensures
#     `selector_exploration_noise` is logged to epoch_summaries / JSONL /
#     train.log per epoch for this new run)
#
# Runs: pompous-pigeon-214
# SP500_ROOT_07=data/sp500/cleaned_drop_incomplete_min_coverage_0.95_corr_0.7_sector_bonus_0.05
# ddim_eta=0.2
# n_split_chunks=3
# revin_blend_weight=0.7
# CUDA_VISIBLE_DEVICES=0 python -m graph_signal_diffusion.cli.train \
#     task=stock_price_forecasting_v3_learned_ds8 \
#     model@task.model=ugnn_sp500_v3_ds8_norm_act_head \
#     dataset.root=${SP500_ROOT_07} \
#     dataset.past_window=20 \
#     dataset.future_window=5 \
#     dataset.batch_size=24 \
#     dataset.batch_size_val=200 \
#     dataset.num_workers=4 \
#     dataset.persistent_workers=true \
#     dataset.pin_memory=true \
#     dataset.n_split_chunks=$n_split_chunks \
#     dataset.shuffle_val=false \
#     dataset.standardize_target_in_x_for_revin=true \
#     diffusion.revin=true \
#     diffusion.ddim_eta=$ddim_eta \
#     diffusion.revin_blend_weight=$revin_blend_weight \
#     model.config.base_channels=128 \
#     'model.config.channel_multipliers=[1,1,1]' \
#     'model.config.pooling_config.gamma=[2,2,2]' \
#     model.config.gnn_config.num_layers=3 \
#     'model.config.gnn_config.temporal_mixer.dilations=[1,1,1]' \
#     model.config.num_bottleneck_layers=3 \
#     +model.config.embedding_config.cond.shared_encoder.temporal.mixer.gated=true \
#     +model.config.embedding_config.cond.shared_encoder.temporal.mixer.attention.enabled=true \
#     +model.config.embedding_config.cond.shared_encoder.temporal.mixer.attention.num_heads=2 \
#     +model.config.embedding_config.cond.shared_encoder.temporal.mixer.attention.dropout=0.1 \
#     +model.config.embedding_config.cond.shared_encoder.temporal.mixer.attention.max_timesteps=20 \
#     model.config.pooling_config.selector_exploration_noise=1.0 \
#     model.config.pooling_config.selector_exploration_noise_min=0.0 \
#     model.config.pooling_config.selector_exploration_noise_schedule=linear \
#     +trainer.selector_exploration_schedule.enabled=true \
#     +trainer.selector_exploration_schedule.warmup_ratio=0.02 \
#     +trainer.selector_exploration_schedule.anneal_ratio=0.90 \
#     trainer.optimizer.learning_rate=2e-5 \
#     trainer.max_epochs=1000 \
#     'trainer.eval_schedule.phases=[{period: 50, until_epoch: 500}, {period: 100}]' \
#     wandb.enabled=true



# ---- carmine-moose-32 + capacity reduction (W:128→64, L:3→2, B:3→2, bs:24→64, chunks:3→10): GPU 1 ---- #
# Capacity-reduction + finer-grained-chronological-split pair-ablation of
# carmine-moose-32. Cond-boost (gated + attention), Gumbel exploration
# schedule, RevIN-fix, eval schedule, and the lr=5e-4 IDENTICAL to carmine.
# Two clusters of knobs change: (a) the backbone size and training batch
# size, (b) the chronological dataset split granularity.
#
# Motivation: carmine-moose-32 overfit hard at lr=5e-4 — train_loss dropped
# to 0.254 at ep 892 while val_loss climbed to 0.325 (gap +0.067), and the
# composite-leaderboard froze at ep 350. The sibling pompous-pigeon-214
# (same arm with lr=2e-5) caught up to wapiti at ep 600 (composite ≈ 0.2791),
# showing the cond-boost + Gumbel stack itself isn't the problem. This ARM
# stacks TWO probes of the carmine overfit:
#
#   (a) Capacity-reduction probe — does reducing model size at lr=5e-4
#       prevent overfit directly? Is the overfit memorization-driven
#       (smaller model can't memorize) or optimization-driven (lr too high
#       for any capacity)?
#   (b) Distributional-shift probe — does finer-grained chronological
#       splitting (n_split_chunks 3 → 10) close the train/val gap by
#       interleaving train and val samples more frequently across the
#       full timeline? If carmine's gap was partially from temporal
#       regime shift (val from a different market period than train),
#       n_split_chunks=10 should narrow it.
#
# These knobs are confounded — this is NOT a clean single-knob ablation.
# If train/val gap closes vs carmine, we cannot disentangle whether the
# fix was capacity or split granularity from this run alone. A follow-up
# arm holding capacity at 128/3/3 and only changing chunks→10 would
# isolate the split knob; this run is the combined cheaper probe.
#
# Capacity deltas vs carmine-moose-32:
#   - model.config.base_channels=64           (carmine: 128) — half width
#   - model.config.gnn_config.num_layers=2    (carmine:   3) — fewer GNN
#                                                              layers per block
#   - model.config.gnn_config.temporal_mixer.dilations=[1,1]
#                                             (carmine: [1,1,1]) — MUST match
#                                                              num_layers length
#   - model.config.num_bottleneck_layers=2    (carmine:   3) — shallower
#                                                              bottleneck
#
# Combined: dominant activation cost scales roughly as base_channels ×
# num_layers, so activation memory drops to ≈0.5 × 0.67 ≈ 0.35× of carmine.
#
# Training delta to use the freed memory:
#   - dataset.batch_size=64                   (carmine:  24) — ~2.7× bump,
#                                                              matches the
#                                                              activation cut.
#                                                              batch_size_val=200
#                                                              unchanged (val
#                                                              has no grad
#                                                              memory).
#
# Dataset-split delta (distributional-shift probe):
#   - dataset.n_split_chunks=10               (carmine:   3) — finer-grained
#                                                              interleaved
#                                                              chronological
#                                                              split. Each
#                                                              chunk is ~10%
#                                                              of timeline (vs
#                                                              ~33% with 3
#                                                              chunks). Train,
#                                                              val, test all
#                                                              cycle through
#                                                              the timeline
#                                                              10 times vs 3.
#                                                              Total sample
#                                                              counts ≈ same
#                                                              (split fractions
#                                                              unchanged: 0.8/
#                                                              0.1/0.1).
#                                                              Train and val
#                                                              are now
#                                                              temporally
#                                                              interleaved
#                                                              more densely,
#                                                              reducing the
#                                                              chance that val
#                                                              samples come
#                                                              from a market
#                                                              regime the
#                                                              train set
#                                                              never sees.
#                                                              BREAKS apples-
#                                                              to-apples with
#                                                              carmine/wapiti/
#                                                              pompous (all
#                                                              n_split_chunks=
#                                                              3); compare_
#                                                              baselines test
#                                                              splits will
#                                                              also differ.
#
# NOT changed (intentional):
#   - lr=5e-4: single-knob delta on capacity (linear-scaling lr to 1.33e-3
#     would compound knobs and is already known to overfit at carmine's lr).
#     The larger batch already reduces gradient noise, which acts as a mild
#     implicit regularizer for the smaller model — that's a feature, not a
#     bug, for this probe.
#   - channel_multipliers=[1,1,1]: still 3 blocks (network DEPTH unchanged;
#     only WIDTH and per-block-layer count reduced).
#   - pooling_config.gamma=[2,2,2]: same downsampling.
#   - cond.shared_encoder.temporal.mixer.num_layers=4: internal cond-encoder
#     mixer untouched. The cond encoder is a separate small subnet whose
#     params (~134K from gated + ~551K from attention) are not the main
#     capacity knob being probed.
#   - cond gated, cond attention, Gumbel exploration kwargs: verbatim from
#     carmine.
#   - max_epochs=1000, eval schedule, RevIN α=1.1150 auto-resolver, ddim_eta=0.2.
#
# Param-count sanity check vs carmine (~5M model-only + cond-boost ~685K):
#   - base_channels² scaling in GNN convs: 64²/128² = 0.25 → most GNN params
#     drop to ~25% of carmine
#   - num_layers 2/3 = 0.67: further reduces per-block param share
#   - bottleneck 2/3 = 0.67: same for bottleneck
#   - Approximate total backbone: ~5M × 0.25 × ((2+2)/(3+3)) ≈ ~0.83M, plus
#     cond-encoder boost (~685K unchanged) → ~1.5M total. ~30% of carmine.
#
# Hypotheses to test:
#   1. train_loss does NOT drop as low as 0.254 (less capacity to memorize).
#      Expected floor: ~0.27-0.29.
#   2. val_loss does NOT diverge from train_loss as widely (gap stays in
#      the +0.01 to +0.03 band, well under carmine's +0.067 at ep 800).
#      With n_split_chunks=10 interleaving, val samples are temporally
#      closer to train samples → smaller distributional shift contribution
#      to the gap. If the gap is now ≤+0.01 even at late epochs, the
#      distributional shift was a significant component of carmine's gap;
#      if the gap is still +0.03-0.05, capacity / lr is the main driver.
#   3. Composite leaderboard updates LATER than carmine's ep 350 — i.e.,
#      the smaller model continues to improve into the lr-decay window
#      instead of plateauing early.
#   4. Best composite plausibly beats carmine's 0.2825 but is UNLIKELY to
#      beat wapiti's 0.2791 — the smaller model has less raw expressive
#      power, even if it generalizes better per-parameter. Note: composite
#      values are NOT apples-to-apples with carmine/wapiti/pompous since
#      val_loss is computed on a different val split (10-chunk interleave
#      vs 3-chunk). Treat absolute composite improvements vs the 3-chunk
#      family as suggestive, not decisive.
#
# Risks:
#   - Smaller model may underfit at lr=5e-4 (lr now overshooting reduced
#     capacity). Watch train_loss in the first ~50 epochs: if it stays
#     above ~0.32 after warmup, capacity is the bottleneck, not lr.
#   - bs=64 + n_split_chunks=10 (~1860 train samples — total count is
#     similar to 3-chunk; split fractions are unchanged) → ~29 batches/
#     epoch. Step count drops from carmine's ~78 batches/epoch to ~29 →
#     fewer optimizer steps per epoch at the same lr → effective lr-per-
#     sample stays similar but the cosine schedule fires fewer steps. The
#     cosine total_steps is computed from epochs × batches, so the
#     schedule shape adapts automatically.
#   - n_split_chunks=10 makes the test set ALSO interleaved, so the
#     compare_baselines test split for this run will not align with the
#     3-chunk family's test split. Direct test-set composite comparison
#     across families is invalid. A clean cross-family comparison requires
#     either (a) re-evaluating older arms with n_split_chunks=10, or
#     (b) re-evaluating this arm with n_split_chunks=3 (the saved checkpoint
#     would need to be re-applied to the 3-chunk test split — feasible
#     since dataset.n_split_chunks is a CLI override).
#   - Cond encoder (~685K params) is now ~45% of total params instead of
#     ~12%. The exploration noise and selector temperature schedules may
#     interact differently with a cond-dominated model — watch
#     selector_entropy_mean_norm and exploration_noise anneal curves.
#
# Control: carmine-moose-32 (same lr, same exploration/cond knobs, FULL
#          capacity at 128/3/3, n_split_chunks=3).
# Sibling: pompous-pigeon-214 (full capacity, lr=2e-5, n_split_chunks=3)
#          — the other "fix carmine's overfit" probe via the lr knob.
# Caveat:  this arm changes TWO knob clusters (capacity + split granularity)
#          vs carmine. Effects are confounded; clean separation needs a
#          follow-up.
#
# Plumbing references:
#   - docs/agent_summaries/cond_encoder_gated_attention_wiring_2026-05-27.md
#   - docs/agent_summaries/wapiti_cond_boost_exploration_launcher_2026-05-27.md
#   - docs/agent_summaries/exploration_noise_aggregator_allowlist_fix_2026-05-28.md
#     (this run benefits from the JSONL surface fix — `selector_exploration_noise`
#     will appear in epoch_summaries.jsonl, not just Wandb)
#
# Runs: spotted-catfish-602  (FINISHED 1000/1000 — commented out post-launch)
# SP500_ROOT_07=data/sp500/cleaned_drop_incomplete_min_coverage_0.95_corr_0.7_sector_bonus_0.05
# ddim_eta=0.2
# n_split_chunks=10
# revin_blend_weight=0.7
# CUDA_VISIBLE_DEVICES=1 python -m graph_signal_diffusion.cli.train \
#     task=stock_price_forecasting_v3_learned_ds8 \
#     model@task.model=ugnn_sp500_v3_ds8_norm_act_head \
#     dataset.root=${SP500_ROOT_07} \
#     dataset.past_window=20 \
#     dataset.future_window=5 \
#     dataset.batch_size=64 \
#     dataset.batch_size_val=200 \
#     dataset.num_workers=4 \
#     dataset.persistent_workers=true \
#     dataset.pin_memory=true \
#     dataset.n_split_chunks=$n_split_chunks \
#     dataset.shuffle_val=false \
#     dataset.standardize_target_in_x_for_revin=true \
#     diffusion.revin=true \
#     diffusion.ddim_eta=$ddim_eta \
#     diffusion.revin_blend_weight=$revin_blend_weight \
#     model.config.base_channels=64 \
#     'model.config.channel_multipliers=[1,1,1]' \
#     'model.config.pooling_config.gamma=[2,2,2]' \
#     model.config.gnn_config.num_layers=2 \
#     'model.config.gnn_config.temporal_mixer.dilations=[1,1]' \
#     model.config.num_bottleneck_layers=2 \
#     +model.config.embedding_config.cond.shared_encoder.temporal.mixer.gated=true \
#     +model.config.embedding_config.cond.shared_encoder.temporal.mixer.attention.enabled=true \
#     +model.config.embedding_config.cond.shared_encoder.temporal.mixer.attention.num_heads=2 \
#     +model.config.embedding_config.cond.shared_encoder.temporal.mixer.attention.dropout=0.1 \
#     +model.config.embedding_config.cond.shared_encoder.temporal.mixer.attention.max_timesteps=20 \
#     model.config.pooling_config.selector_exploration_noise=1.0 \
#     model.config.pooling_config.selector_exploration_noise_min=0.0 \
#     model.config.pooling_config.selector_exploration_noise_schedule=linear \
#     +trainer.selector_exploration_schedule.enabled=true \
#     +trainer.selector_exploration_schedule.warmup_ratio=0.02 \
#     +trainer.selector_exploration_schedule.anneal_ratio=0.90 \
#     trainer.optimizer.learning_rate=5e-4 \
#     trainer.max_epochs=1000 \
#     'trainer.eval_schedule.phases=[{period: 50, until_epoch: 500}, {period: 100}]' \
#     wandb.enabled=true



# ---- full-capacity cond-boost + 10-chunk + 2000 epochs @ lr=1e-4 (kurtosis-gap fix): GPU 1 ---- #
# Full-capacity (128/3/3) cond-boost + Gumbel arm at lr=1e-4 (= wapiti's lr),
# with the 10-chunk split and a 2000-epoch budget. Started from the pompous-
# pigeon-214 recipe but with the learning rate bumped 5× (2e-5 → 1e-4).
#
# WHY lr=1e-4 (not pompous's 2e-5): pompous's KURTOSIS GAP is too large — its
# generated 5-day-return distribution is far more peaked/heavy-tailed than
# real. Measured val_kurtosis_gap (gen − real, real ≈ 3.4) across the run:
#   pompous (lr 2e-5): median 26.7, never below 18 (ep999 still 18.4) — BAD
#   catfish (lr 5e-4): median 14.0, ep999 11.0
#   wapiti  (lr 1e-4): median 11.4, down to ~7 — BEST of the family
# The kurtosis gap tracks lr inversely-U: the very low 2e-5 leaves the model
# under-trained on the spread/tail structure (collapsed, over-peaked samples),
# while wapiti's 1e-4 produced the lowest gap. lr=1e-4 is the sweet spot — high
# enough to learn the tail structure, low enough to avoid carmine's 5e-4
# overfit. This arm tests whether 1e-4 + cond-boost (a point never run before:
# wapiti had 1e-4 but NO cond-boost; carmine had cond-boost but 5e-4; pompous
# had cond-boost but 2e-5) fixes the kurtosis gap while keeping pompous's other
# gains.
#
# Capacity stays full 128/3/3 (5.05M params); bs stays 24 (bs=64 OOMs at full
# capacity). cond gated + attention + Gumbel exploration all kept.
#
# Deltas vs pompous-pigeon-214:
#   - trainer.optimizer.learning_rate=1e-4  (pompous: 2e-5) — 5× UP; kurtosis-
#                                                gap fix (see above). Now equals
#                                                wapiti's lr.
#   - dataset.n_split_chunks=10   (pompous: 3) — finer interleaved
#                                                chronological split (the
#                                                spotted-catfish change;
#                                                reduces train/val regime
#                                                shift).
#   - trainer.max_epochs=2000     (pompous: 1000) — 2× budget; the cosine decay
#                                                finishes over the full 2000.
#
# SCHEDULE NOTES for max_epochs=2000:
#   1. Cosine LR decay stretches over the full 2000 epochs — INTENDED (the
#      whole point of more epochs). min_lr = 0.05×1e-4 = 5e-6.
#   2. Selector temperature anneal AND Gumbel exploration anneal are both set
#      to COMPLETE at ep 1500 (0.75 fraction of max_epochs), NOT stretched to
#      the end:
#        warmup_ratio=0.02, anneal_ratio=0.73  →  done at (0.02+0.73)=0.75×2000=ep1500.
#      Completion fraction = warmup_ratio + anneal_ratio, because both resolve
#      to total_steps × ratio (verified: cli/train.py _resolve_steps; the yaml
#      "remaining steps" comment is misleading). After ep 1500 both hold at
#      their floor (T_min=0.5; exploration=0) for the final 500 epochs — the
#      last 25% is pure low-temp exploitation with the lr still decaying.
#   3. vs pompous: pompous left the temperature schedule at its ds8 default
#      (anneal_ratio=0.90 → completes at 0.92). THIS arm overrides BOTH the
#      temperature schedule (plain override; key exists in the ds8 trainer
#      config) and the exploration schedule (already +appended) to 0.73.
#
# Eval schedule kept verbatim from pompous: [{period:50, until_epoch:500},
# {period:100}]. NOTE: dense (every-50) eval now covers only the first 25% of
# training; the back 1500 epochs get every-100 (15 evals). If you want denser
# late-phase coverage, bump until_epoch to 1000 or add a third phase.
#
# RevIN: revin_blend_weight=0.7 unchanged; revin_sigma_correction AUTO-resolves
# at training time from the 10-chunk sigma_cond_mean_train (≈1.107, matching
# spotted-catfish — NOT pompous's 3-chunk 1.1150). Materialized to .hydra at
# launch; no action needed.
#
# Comparisons:
#   - vs pompous-pigeon-214 (3-chunk, 1000ep, lr2e-5): 3 knobs differ (lr 5×UP,
#     split 3→10, 2× epochs). Primary read: is the kurtosis gap fixed (target
#     ~7-11 like wapiti, vs pompous's ~18-27)? (NOTE: composite/val metrics are
#     on the 10-chunk val split, NOT comparable to pompous's 3-chunk composites
#     — see n_split_chunks leakage note. Compare gap behavior + GRW-relative
#     test metrics, not raw composite.)
#   - vs colorful-wapiti-739 (3-chunk, 1000ep, lr1e-4, NO cond-boost): SAME lr
#     now. Adds cond-boost + Gumbel + 10-chunk + 2× epochs. Tests whether the
#     cond-boost stack helps at the lr that already gave wapiti the family's
#     lowest kurtosis gap.
#   - vs spotted-catfish-602 (10-chunk, 1000ep, lr5e-4, 64/2/2): higher capacity
#     + lower lr (1e-4 vs 5e-4) + longer train, same split. Does full capacity
#     reproduce/improve catfish's apparent cum_T directional skill (0.51-0.53)?
#
# Risk: lr=1e-4 + full 128/3/3 capacity + cond-boost + 2000 epochs is the most
# overfit-prone combination of the family on raw capacity×lr×budget — but it's
# 5× BELOW carmine's 5e-4 (which overfit at 3-chunk/1000ep), AND the 10-chunk
# split is itself anti-overfit (kept catfish's gap ≈0). wapiti ran lr=1e-4 at
# 3-chunk/1000ep without overfitting. Net: moderate risk. Watch the train/val
# gap after ~ep 1000 (the 2× budget is the new stressor); if it drifts past
# ~+0.03, the long budget at 1e-4 is overfitting and an earlier checkpoint wins.
#
# Plumbing references:
#   - docs/agent_summaries/carmine_moose_lr_down_5x_launcher_2026-05-27.md (pompous)
#   - docs/agent_summaries/carmine_moose_capacity_reduction_launcher_2026-05-28.md (catfish; 10-chunk rationale)
#   - memory: reference_n_split_chunks_cross_split_leakage (why 10-chunk metrics aren't comparable to 3-chunk arms)
#
# Runs: fierce-squirrel-222  (LAUNCHED/running on GPU 1 — commented out post-launch)
# SP500_ROOT_07=data/sp500/cleaned_drop_incomplete_min_coverage_0.95_corr_0.7_sector_bonus_0.05
# ddim_eta=0.2
# n_split_chunks=10
# revin_blend_weight=0.7
# CUDA_VISIBLE_DEVICES=1 python -m graph_signal_diffusion.cli.train \
#     task=stock_price_forecasting_v3_learned_ds8 \
#     model@task.model=ugnn_sp500_v3_ds8_norm_act_head \
#     dataset.root=${SP500_ROOT_07} \
#     dataset.past_window=20 \
#     dataset.future_window=5 \
#     dataset.batch_size=24 \
#     dataset.batch_size_val=200 \
#     dataset.num_workers=4 \
#     dataset.persistent_workers=true \
#     dataset.pin_memory=true \
#     dataset.n_split_chunks=$n_split_chunks \
#     dataset.shuffle_val=false \
#     dataset.standardize_target_in_x_for_revin=true \
#     diffusion.revin=true \
#     diffusion.ddim_eta=$ddim_eta \
#     diffusion.revin_blend_weight=$revin_blend_weight \
#     model.config.base_channels=128 \
#     'model.config.channel_multipliers=[1,1,1]' \
#     'model.config.pooling_config.gamma=[2,2,2]' \
#     model.config.gnn_config.num_layers=3 \
#     'model.config.gnn_config.temporal_mixer.dilations=[1,1,1]' \
#     model.config.num_bottleneck_layers=3 \
#     +model.config.embedding_config.cond.shared_encoder.temporal.mixer.gated=true \
#     +model.config.embedding_config.cond.shared_encoder.temporal.mixer.attention.enabled=true \
#     +model.config.embedding_config.cond.shared_encoder.temporal.mixer.attention.num_heads=2 \
#     +model.config.embedding_config.cond.shared_encoder.temporal.mixer.attention.dropout=0.1 \
#     +model.config.embedding_config.cond.shared_encoder.temporal.mixer.attention.max_timesteps=20 \
#     model.config.pooling_config.selector_exploration_noise=1.0 \
#     model.config.pooling_config.selector_exploration_noise_min=0.0 \
#     model.config.pooling_config.selector_exploration_noise_schedule=linear \
#     +trainer.selector_exploration_schedule.enabled=true \
#     +trainer.selector_exploration_schedule.warmup_ratio=0.02 \
#     +trainer.selector_exploration_schedule.anneal_ratio=0.73 \
#     trainer.selector_temperature_schedule.warmup_ratio=0.02 \
#     trainer.selector_temperature_schedule.anneal_ratio=0.73 \
#     trainer.optimizer.learning_rate=1e-4 \
#     trainer.max_epochs=2000 \
#     'trainer.eval_schedule.phases=[{period: 50, until_epoch: 500}, {period: 100}]' \
#     wandb.enabled=true



# ---- spotted-catfish-602 extended to 5000 epochs + lr DOWN to 1e-4 (small 64/2/2, 10-chunk, bs=64) + 0.75-frac anneals: GPU 0 ---- #
# Long-training extension of spotted-catfish-602 (the reduced-capacity arm,
# 64/2/2, 1.88M params) at the family's sweet-spot lr=1e-4 (= wapiti /
# fierce-squirrel), with 5× the epoch budget and the temperature + exploration
# anneals updated to fierce-squirrel-222's 0.75-fraction-completion shape.
#
# Deltas vs spotted-catfish-602:
#   - trainer.optimizer.learning_rate=1e-4  (catfish: 5e-4) — 0.2× DOWN. catfish
#                                                ran 5e-4 and did not overfit
#                                                through 1000 epochs, but 5000
#                                                epochs of sustained 5e-4 would
#                                                be the riskiest lr exposure in
#                                                the family. 1e-4 is the safe
#                                                sweet spot (wapiti's lowest
#                                                kurtosis gap) and de-risks the
#                                                long run.
#   - trainer.max_epochs=5000     (catfish: 1000) — 5× budget. catfish's
#                                                composite leaderboard rank-1
#                                                was EARLY (ep 300, 0.2519) and
#                                                it stayed flat/healthy through
#                                                ep 999 (gap ≈ 0). Tests whether
#                                                5× budget at the safe lr lets
#                                                the small model improve past
#                                                that, or just plateaus.
#   - exploration anneal_ratio 0.90 → 0.73, AND temperature schedule made
#     explicit at warmup_ratio=0.02 / anneal_ratio=0.73 (catfish left
#     temperature at the ds8 default 0.02/0.90). Both now COMPLETE at
#     (0.02+0.73)=0.75 × 5000 = ep 3750, matching fierce-squirrel-222's
#     0.75-fraction shape. After ep 3750 both hold at floor (T_min=0.5;
#     exploration=0) for the final 1250 epochs.
#
# Unchanged from catfish: base_channels=64, num_layers=2, dilations=[1,1],
# num_bottleneck_layers=2 (1.88M params), bs=64, n_split_chunks=10,
# cond gated + attention, Gumbel exploration (1.0 linear), RevIN
# (revin_blend_weight=0.7; α auto-resolves ≈1.107 for 10-chunk), ddim_eta=0.2,
# eval schedule [{period:50, until_epoch:500}, {period:100}].
#
# SCHEDULE NOTES for max_epochs=5000:
#   1. Cosine LR decay stretches over the full 5000 epochs. min_lr = 0.05×1e-4
#      = 5e-6. (At lr=1e-4 — 5× below catfish's 5e-4 — the stretched decay is no
#      longer a sustained-high-lr concern: wapiti ran 1e-4 cleanly, and this
#      model is far smaller.)
#   2. Temperature + exploration anneals complete at ep 3750 (0.75 fraction),
#      then hold floor for the final 1250 epochs (pure low-temp exploitation,
#      lr still decaying).
#
# Hypothesis: catfish was compute-limited, not capacity-limited — 5× budget at
# the safe lr=1e-4 + a long exploration phase (to ep 3750) lets the 1.88M model
# keep improving past its ep-300 best. Counter-hypothesis: the small model
# already converged by ep 300 and the extra 4700 epochs add nothing.
#
# Risk: much lower now that lr=1e-4 (the 5e-4 version's "highest sustained-lr
# exposure" concern drove this lr-down change). Main remaining risk is
# diminishing returns — 5000 epochs on a 1.88M model may plateau early. Watch
# the composite from ~ep 500; if flat, the budget bought nothing (kill + revert
# to an early checkpoint). Mild overfit risk persists only in the long tail —
# watch the train/val gap after ~ep 2000; if it drifts past ~+0.03 the
# best_models tracker will surface the earlier optimum.
#
# Pair context: fierce-squirrel-222 (FULL 128/3/3, 2000ep) and this arm (SMALL
# 64/2/2, 5000ep) now SHARE lr=1e-4 + 10-chunk + 0.75-fraction anneals — so this
# is closer to a capacity comparison (full vs small) at the sweet-spot lr,
# though the epoch budgets still differ (2000 vs 5000).
#
# Compute: bs=64 on the 10-chunk train set (~1857 samples) → ~30 batches/epoch
# × 5000 = ~150k steps on the small 1.88M model — comparable step count to
# fierce-squirrel but cheaper per step. GPU 0 (fierce-squirrel occupies GPU 1).
#
# Plumbing references:
#   - docs/agent_summaries/carmine_moose_capacity_reduction_launcher_2026-05-28.md (catfish, the base)
#   - docs/agent_summaries/pompous_10chunk_2000ep_launcher_2026-05-29.md (fierce-squirrel; the 0.75-fraction anneal shape this matches)
#
# Runs: sociable-frigatebird-619  (FINISHED 5000/5000 — commented out post-launch)
# SP500_ROOT_07=data/sp500/cleaned_drop_incomplete_min_coverage_0.95_corr_0.7_sector_bonus_0.05
# ddim_eta=0.2
# n_split_chunks=10
# revin_blend_weight=0.7
# CUDA_VISIBLE_DEVICES=0 python -m graph_signal_diffusion.cli.train \
#     task=stock_price_forecasting_v3_learned_ds8 \
#     model@task.model=ugnn_sp500_v3_ds8_norm_act_head \
#     dataset.root=${SP500_ROOT_07} \
#     dataset.past_window=20 \
#     dataset.future_window=5 \
#     dataset.batch_size=64 \
#     dataset.batch_size_val=200 \
#     dataset.num_workers=4 \
#     dataset.persistent_workers=true \
#     dataset.pin_memory=true \
#     dataset.n_split_chunks=$n_split_chunks \
#     dataset.shuffle_val=false \
#     dataset.standardize_target_in_x_for_revin=true \
#     diffusion.revin=true \
#     diffusion.ddim_eta=$ddim_eta \
#     diffusion.revin_blend_weight=$revin_blend_weight \
#     model.config.base_channels=64 \
#     'model.config.channel_multipliers=[1,1,1]' \
#     'model.config.pooling_config.gamma=[2,2,2]' \
#     model.config.gnn_config.num_layers=2 \
#     'model.config.gnn_config.temporal_mixer.dilations=[1,1]' \
#     model.config.num_bottleneck_layers=2 \
#     +model.config.embedding_config.cond.shared_encoder.temporal.mixer.gated=true \
#     +model.config.embedding_config.cond.shared_encoder.temporal.mixer.attention.enabled=true \
#     +model.config.embedding_config.cond.shared_encoder.temporal.mixer.attention.num_heads=2 \
#     +model.config.embedding_config.cond.shared_encoder.temporal.mixer.attention.dropout=0.1 \
#     +model.config.embedding_config.cond.shared_encoder.temporal.mixer.attention.max_timesteps=20 \
#     model.config.pooling_config.selector_exploration_noise=1.0 \
#     model.config.pooling_config.selector_exploration_noise_min=0.0 \
#     model.config.pooling_config.selector_exploration_noise_schedule=linear \
#     +trainer.selector_exploration_schedule.enabled=true \
#     +trainer.selector_exploration_schedule.warmup_ratio=0.02 \
#     +trainer.selector_exploration_schedule.anneal_ratio=0.73 \
#     trainer.selector_temperature_schedule.warmup_ratio=0.02 \
#     trainer.selector_temperature_schedule.anneal_ratio=0.73 \
#     trainer.optimizer.learning_rate=1e-4 \
#     trainer.max_epochs=5000 \
#     'trainer.eval_schedule.phases=[{period: 50, until_epoch: 500}, {period: 100}]' \
#     wandb.enabled=true



# ---- HIGH-LR variant of sociable-frigatebird-619: lr UP to 5e-4 (small 64/2/2, 10-chunk, bs=64, 5000ep) + K=20 eval: GPU 0 ---- #
# One-knob* LR ablation on sociable-frigatebird-619 (the small 1.88M arm that WON
# the 10-chunk capacity comparison on held-out test). Raises base lr 5× (1e-4 →
# 5e-4); nothing architectural changes. (*plus eval-only K / cap bumps below,
# which do not affect the trained model.)
#
# WHY 5e-4 — the LR-floor finding:
#   frigatebird's cosine (min_lr_ratio=0.05 × 1e-4) hit its 5e-6 FLOOR by ~ep4500
#   and its selector anneals complete at ep3750, so the back ~1250 epochs trained
#   near-frozen. Yet kurtosis_gen was STILL falling on test (ep2800 17.6 → ep4500
#   16.3, real ~7.74). Raising base lr to 5e-4 lifts the whole schedule so the
#   floor becomes 0.05 × 5e-4 = 2.5e-5 (5× higher) — it UNFREEZES that tail
#   WITHOUT extending the horizon, directly testing whether the small model's
#   kurtosis gap keeps closing.
#
# Deltas vs sociable-frigatebird-619:
#   - trainer.optimizer.learning_rate=5e-4   (was 1e-4) — 5× UP. = spotted-
#     catfish-602's native lr. Completes the small-model (lr, budget) 2×2:
#       catfish (5e-4, 1000ep) · THIS (5e-4, 5000ep) · frigatebird (1e-4, 5000ep).
#   - trainer.n_samples_per_input=20         (was 10) — eval-time K doubled for
#     tighter CRPS/coverage/MIS → more reliable best_models selection. EVAL-ONLY
#     (no effect on training speed or the 30 train-batches/epoch).
#   - trainer.max_num_val_batches=4          (was 2) AND
#     trainer.max_num_train_val_batches=2    (was 1) — REQUIRED with K=20:
#     originals_per_batch = ceil(batch_size_val / K) = ceil(200/20) = 10 (was 20),
#     so the batch-unit caps must double to PRESERVE window coverage (val: 40
#     windows/eval; train-val: 20). test is uncapped (-1) → still all 240.
#
# Unchanged from frigatebird: base_channels=64, num_layers=2, dilations=[1,1],
# num_bottleneck_layers=2 (1.88M), bs=64, n_split_chunks=10, cond gated +
# attention, Gumbel exploration (1.0 linear), anneals warmup 0.02 / anneal 0.73
# (complete at ep3750 = 0.75 fraction), RevIN (blend 0.7; α auto ≈1.107 for
# 10-chunk), ddim_eta=0.2, eval_schedule [{period:50, until 500},{period:100}].
#
# Validation cadence (best-model selection granularity): every 50ep to ep500,
# every 100ep to ep5000 (+ final) → ~55 heavy evals; min_warmup_evals=3 so
# best_models start being saved ~ep150.
#
# RISK — this deliberately REVERSES frigatebird's documented lr choice (it chose
# 1e-4 because "5000 epochs of sustained 5e-4 would be the riskiest lr exposure
# in the family"). We take that exposure on purpose. WATCH: catfish (5e-4) peaked
# on val composite EARLY (ep300 of 1000); this arm may likewise peak early and
# overfit the back half on val_loss — acceptable, since kurtosis is the target
# and best_models pins the val optimum regardless. Kill criterion: if by ~ep1500
# BOTH val composite is worse than frigatebird's 0.2513 AND kurtosis_gen has not
# improved past 16.3, the higher lr bought nothing — revert to frigatebird.
#
# Compute: bs=64 on the 10-chunk train set (~1857 samples) → ~30 batches/epoch ×
# 5000 = ~150k steps on the 1.88M model. GPU 0 (both GPUs free — frigatebird and
# fierce-squirrel both finished). Each heavy eval ~2× frigatebird's (K=20 +
# doubled caps); ~55 evals → modest overall slowdown.
#
# Plumbing references:
#   - docs/agent_summaries/catfish_extended_5000ep_launcher_2026-05-29.md (frigatebird, the base)
#   - docs/agent_summaries/frigatebird_highlr_5e4_arm_2026-06-05.md (this arm: 5e-4 + K=20 rationale)
#   - memory: reference_sp500_capacity_10chunk_small_vs_full (frigatebird won the capacity verdict)
#
# Runs: exuberant-lemming-913  (LAUNCHED/running on GPU 0 — commented out post-launch)
# SP500_ROOT_07=data/sp500/cleaned_drop_incomplete_min_coverage_0.95_corr_0.7_sector_bonus_0.05
# ddim_eta=0.2
# n_split_chunks=10
# revin_blend_weight=0.7
# CUDA_VISIBLE_DEVICES=0 python -m graph_signal_diffusion.cli.train \
#     task=stock_price_forecasting_v3_learned_ds8 \
#     model@task.model=ugnn_sp500_v3_ds8_norm_act_head \
#     dataset.root=${SP500_ROOT_07} \
#     dataset.past_window=20 \
#     dataset.future_window=5 \
#     dataset.batch_size=64 \
#     dataset.batch_size_val=200 \
#     dataset.num_workers=4 \
#     dataset.persistent_workers=true \
#     dataset.pin_memory=true \
#     dataset.n_split_chunks=$n_split_chunks \
#     dataset.shuffle_val=false \
#     dataset.standardize_target_in_x_for_revin=true \
#     diffusion.revin=true \
#     diffusion.ddim_eta=$ddim_eta \
#     diffusion.revin_blend_weight=$revin_blend_weight \
#     model.config.base_channels=64 \
#     'model.config.channel_multipliers=[1,1,1]' \
#     'model.config.pooling_config.gamma=[2,2,2]' \
#     model.config.gnn_config.num_layers=2 \
#     'model.config.gnn_config.temporal_mixer.dilations=[1,1]' \
#     model.config.num_bottleneck_layers=2 \
#     +model.config.embedding_config.cond.shared_encoder.temporal.mixer.gated=true \
#     +model.config.embedding_config.cond.shared_encoder.temporal.mixer.attention.enabled=true \
#     +model.config.embedding_config.cond.shared_encoder.temporal.mixer.attention.num_heads=2 \
#     +model.config.embedding_config.cond.shared_encoder.temporal.mixer.attention.dropout=0.1 \
#     +model.config.embedding_config.cond.shared_encoder.temporal.mixer.attention.max_timesteps=20 \
#     model.config.pooling_config.selector_exploration_noise=1.0 \
#     model.config.pooling_config.selector_exploration_noise_min=0.0 \
#     model.config.pooling_config.selector_exploration_noise_schedule=linear \
#     +trainer.selector_exploration_schedule.enabled=true \
#     +trainer.selector_exploration_schedule.warmup_ratio=0.02 \
#     +trainer.selector_exploration_schedule.anneal_ratio=0.73 \
#     trainer.selector_temperature_schedule.warmup_ratio=0.02 \
#     trainer.selector_temperature_schedule.anneal_ratio=0.73 \
#     trainer.optimizer.learning_rate=5e-4 \
#     trainer.n_samples_per_input=20 \
#     trainer.max_num_val_batches=4 \
#     trainer.max_num_train_val_batches=2 \
#     trainer.max_epochs=5000 \
#     'trainer.eval_schedule.phases=[{period: 50, until_epoch: 500}, {period: 100}]' \
#     wandb.enabled=true



# ---- 1-STEP (T=1) variant of sociable-frigatebird-619: future_window 5->1 (small 64/2/2, 10-chunk, bs=64, 5000ep): GPU 0 ---- #
# Horizon ablation on sociable-frigatebird-619 (the small 1.88M arm that WON the
# 10-chunk capacity comparison). The ONLY data change is the prediction horizon:
# dataset.future_window 5 -> 1. lr stays 1e-4. Eval K=20 + caps 4/2 (lemming's
# refinement, added at launch) -- so the TRAINING composite is comparable to
# exuberant-lemming-913 (K=20), NOT frigatebird (K=10): val_return_crps is
# K-biased. TEST eval runs n_samples=20 for ALL arms via compare_baselines -> clean.
#
# TEMPORAL-PROCESSING CHANGE (the one architectural delta):
#   The denoiser's per-layer temporal mixer operates over the prediction axis (T).
#   At T=1 it is DEGENERATE: the depthwise Conv1d (kernel=3, pad=1) over a length-1
#   sequence collapses to a single learned per-channel scalar (the two outer taps
#   hit zero-padding) and the self-attention AUTO-SKIPS (graph_conv.py guards on
#   x_seq.size(-1) > 1). So we DISABLE it -- temporal_mixer.enabled=false (replaces
#   frigatebird's dilations=[1,1] override). ugnn.py:296-299 maps enabled=false ->
#   schedule 'off' -> mixers=None -> cleanly skipped in the forward pass. The GNN
#   layers still do ALL spatial graph processing; only the dead T-axis mixing is
#   removed. This is the honest single-horizon architecture (slightly fewer params).
#
# Unchanged from frigatebird: base_channels=64, num_layers=2, num_bottleneck=2,
#   channel_multipliers=[1,1,1], gamma=[2,2,2], bs=64, n_split_chunks=10, COND
#   encoder gated+attention (max_timesteps=20 -- that is the PAST window, NOT the
#   horizon, so it is unaffected), learned_projection auto-retargets 20->1 (cond
#   temporal dim still matches x's T), Gumbel exploration (1.0 linear), anneals
#   0.02/0.73 (complete ep3750), RevIN (blend 0.7; alpha auto ~1.107, computed from
#   the PAST window so horizon-independent), ddim_eta=0.2, eval_schedule
#   [{50, until 500},{100}], lr 1e-4, K=20 eval (caps val=4 / train-val=2).
#
# WHAT T=1 MEANS (read before comparing across arms): this is NEXT-DAY return
#   forecasting, not 5-day-ahead. The cum_T_* metrics (5-day CUMULATIVE return)
#   collapse to the 1-day return == the single-step metrics, so this arm is NOT a
#   drop-in vs the T=5 arms on the headline cumulative metrics. The clean
#   comparison is "T=1 specialist vs frigatebird's FIRST horizon step (t=1 of 5)"
#   on per-step CRPS/coverage/kurtosis -- does specializing to a single step sharpen
#   the next-day distribution? Bonus: T=1 sidesteps the phase-distribution bug
#   (datamodule guard only fires for future_window>1).
#
# RISK: low. Same proven backbone/recipe; only the horizon and the (degenerate)
#   T-axis mixer change. If the model underfits next-day structure relative to its
#   t=1 slice under T=5, the horizon specialization bought nothing -- revert.
#
# Plumbing references:
#   - docs/agent_summaries/frigatebird_highlr_5e4_arm_2026-06-05.md (the base arm + recipe)
#   - docs/agent_summaries/frigatebird_t1_horizon_arm_2026-06-19.md (this arm: T=1 + temporal-mixer disable)
#   - memory: reference_sp500_capacity_10chunk_small_vs_full (frigatebird won the capacity verdict)
#
# Runs: attractive-manticore-346  (LAUNCHED/running on GPU 0 — commented out post-launch)
# NOTE: launched WITH lemming's K=20 eval refinement (n_samples_per_input=20,
# max_num_val_batches=4, max_num_train_val_batches=2) — added at launch, NOT the
# K=10 originally drafted. Consequence: this arm's TRAINING composite is comparable
# to exuberant-lemming-913 (K=20), NOT frigatebird (K=10), because val_return_crps
# is K-biased (biased 1/(2M^2) estimator). TEST eval via compare_baselines runs
# n_samples=20 for ALL arms → still clean. lr stayed 1e-4. See doc for details.
# SP500_ROOT_07=data/sp500/cleaned_drop_incomplete_min_coverage_0.95_corr_0.7_sector_bonus_0.05
# ddim_eta=0.2
# n_split_chunks=10
# revin_blend_weight=0.7
# CUDA_VISIBLE_DEVICES=0 python -m graph_signal_diffusion.cli.train \
#     task=stock_price_forecasting_v3_learned_ds8 \
#     model@task.model=ugnn_sp500_v3_ds8_norm_act_head \
#     dataset.root=${SP500_ROOT_07} \
#     dataset.past_window=20 \
#     dataset.future_window=1 \
#     dataset.batch_size=64 \
#     dataset.batch_size_val=200 \
#     dataset.num_workers=4 \
#     dataset.persistent_workers=true \
#     dataset.pin_memory=true \
#     dataset.n_split_chunks=$n_split_chunks \
#     dataset.shuffle_val=false \
#     dataset.standardize_target_in_x_for_revin=true \
#     diffusion.revin=true \
#     diffusion.ddim_eta=$ddim_eta \
#     diffusion.revin_blend_weight=$revin_blend_weight \
#     model.config.base_channels=64 \
#     'model.config.channel_multipliers=[1,1,1]' \
#     'model.config.pooling_config.gamma=[2,2,2]' \
#     model.config.gnn_config.num_layers=2 \
#     model.config.gnn_config.temporal_mixer.enabled=false \
#     model.config.num_bottleneck_layers=2 \
#     +model.config.embedding_config.cond.shared_encoder.temporal.mixer.gated=true \
#     +model.config.embedding_config.cond.shared_encoder.temporal.mixer.attention.enabled=true \
#     +model.config.embedding_config.cond.shared_encoder.temporal.mixer.attention.num_heads=2 \
#     +model.config.embedding_config.cond.shared_encoder.temporal.mixer.attention.dropout=0.1 \
#     +model.config.embedding_config.cond.shared_encoder.temporal.mixer.attention.max_timesteps=20 \
#     model.config.pooling_config.selector_exploration_noise=1.0 \
#     model.config.pooling_config.selector_exploration_noise_min=0.0 \
#     model.config.pooling_config.selector_exploration_noise_schedule=linear \
#     +trainer.selector_exploration_schedule.enabled=true \
#     +trainer.selector_exploration_schedule.warmup_ratio=0.02 \
#     +trainer.selector_exploration_schedule.anneal_ratio=0.73 \
#     trainer.selector_temperature_schedule.warmup_ratio=0.02 \
#     trainer.selector_temperature_schedule.anneal_ratio=0.73 \
#     trainer.optimizer.learning_rate=1e-4 \
#     trainer.n_samples_per_input=20 \
#     trainer.max_num_val_batches=4 \
#     trainer.max_num_train_val_batches=2 \
#     trainer.max_epochs=5000 \
#     'trainer.eval_schedule.phases=[{period: 50, until_epoch: 500}, {period: 100}]' \
#     wandb.enabled=true



# ***************** SP500 ARM G-v2 T=10: inverted-pyramid denoiser, longer horizon ****** #
# Pairs with ARM G (fiery-capuchin-409): same T=10 horizon and corr_0.7 graph,
# but with an inverted-pyramid denoiser [4,2,1] at base_channels=16
# (channels [64,32,16]) instead of ARM G's diamond [1,2,1] at base=64 ([64,128,64]).
#
# Changes vs ARM G (fiery-capuchin-409):
#   - base_channels=16 (from 64), channel_multipliers=[4,2,1] (from [1,2,1]).
#     Channels: [64, 32, 16] vs [64, 128, 64]. Much smaller model.
#   - batch_size=64 (from 32): smaller model fits more under AMP even at T=10.
#
# Kept from ARM G: corr_0.7 graph, T=10, past_window=20, 2 GNN layers/block,
# K=2 ChebConv, 2 attention heads, time_embed=128, conditioning (4×128ch,
# cross-attention with 2 heads), dropout=0.01, RevIN, DDIM, dense eval schedule.
#
# Hypothesis: the 64-ch input level captures local structure; the narrow deeper
# levels can't memorize the marginal, reducing overfitting. If forecast quality
# holds, this validates that denoiser depth capacity is not the bottleneck.
#
# Control: ARM G (fiery-capuchin-409).
# Runs: convivial-serpent-360
# SP500_ROOT_07=data/sp500/cleaned_drop_incomplete_min_coverage_0.95_corr_0.7_sector_bonus_0.05
# dropout=0.01
# CUDA_VISIBLE_DEVICES=1 python -m graph_signal_diffusion.cli.train \
#     task=stock_price_forecasting_v3 \
#     dataset.root=${SP500_ROOT_07} \
#     dataset.past_window=20 \
#     dataset.future_window=10 \
#     dataset.batch_size=64 \
#     dataset.batch_size_val=200 \
#     dataset.num_workers=4 \
#     dataset.persistent_workers=true \
#     dataset.pin_memory=true \
#     dataset.n_split_chunks=3 \
#     dataset.shuffle_val=false \
#     dataset.standardize_target_in_x_for_revin=true \
#     diffusion.revin=true \
#     model.config.gnn_config.dropout=$dropout \
#     model.config.gnn_config.temporal_mixer.schedule=per_layer \
#     model.config.gnn_config.temporal_mixer.kernel_size=3 \
#     'model.config.gnn_config.temporal_mixer.dilations=[1,1]' \
#     model.config.gnn_config.temporal_mixer.use_pointwise=true \
#     model.config.gnn_config.temporal_mixer.causal=false \
#     model.config.gnn_config.temporal_mixer.attention.enabled=true \
#     model.config.gnn_config.temporal_mixer.attention.num_heads=2 \
#     model.config.gnn_config.temporal_mixer.attention.dropout=$dropout \
#     model.config.gnn_config.temporal_mixer.attention.max_timesteps=10 \
#     'model.config.embedding_config.cond.shared_encoder.temporal.hidden_channels=[128,128,128,128]' \
#     model.config.embedding_config.cond.shared_encoder.temporal.num_layers=4 \
#     model.config.embedding_config.cond.shared_encoder.temporal.mixer.causal=false \
#     'model.config.embedding_config.cond.shared_encoder.temporal.mixer.dilations=[1,2,4,8]' \
#     model.config.embedding_config.cond.shared_encoder.temporal.output.mode=time_varying \
#     model.config.embedding_config.cond.shared_encoder.temporal.output.time_varying.method=none \
#     model.config.embedding_config.cond.block_fusion.mode=cross_attention \
#     model.config.embedding_config.cond.block_fusion.cross_attention.heads=2 \
#     model.config.embedding_config.cond.block_fusion.cross_attention.dropout=$dropout \
#     model.config.embedding_config.cond.block_fusion.cross_attention.bias=true \
#     model.config.embedding_config.cond.block_fusion.cross_attention.causal=false \
#     model.config.base_channels=16 \
#     'model.config.channel_multipliers=[4,2,1]' \
#     'model.config.pooling_config.gamma=[1,1,1]' \
#     model.config.pooling_config.selection_method=stride \
#     model.config.num_bottleneck_layers=1 \
#     model.config.compact_skip_cache=false \
#     model.config.gnn_config.pointwise_sparse_mode=off \
#     model.config.pooling_config.selector_packed_score_mode=off \
#     trainer.selector_temperature_schedule.enabled=false \
#     trainer.max_epochs=2500 \
#     trainer.eval_every_n_epochs=25 \
#     trainer.eval_schedule.phases='[{period: 50, until_epoch: 250}, {period: 25, until_epoch: 750}, {period: 100}]' \
#     trainer.save_checkpoint_every_n_epochs=50 \
#     trainer.use_amp=true \
#     trainer.n_samples_per_input=10 \
#     trainer.max_num_val_batches=2 trainer.max_num_train_val_batches=1 \
#     wandb.enabled=true



# ---- ARM F-v2 (amiable-rattlesnake-49) — completed, commented out -------------------- #
# Identical to ARM F (pragmatic-trout-751) except uses the more heavily sparsified
# correlation graph (threshold 0.7 vs 0.6). The sparser graph has fewer edges per
# stock — each ChebConv K=2 hop aggregates over a smaller neighborhood. This tests
# whether the denser 0.6-threshold graph provides useful information for the GNN
# or just adds noise.
#
# Only change vs ARM F: dataset.root points to corr_0.7 variant.
# Runs on GPU 1 (ARM F runs on GPU 0).
#
# Control: ARM F (pragmatic-trout-751) on corr_0.6.
# Runs: amiable-rattlesnake-49
# SP500_ROOT_07=data/sp500/cleaned_drop_incomplete_min_coverage_0.95_corr_0.7_sector_bonus_0.05
# dropout=0.01
# CUDA_VISIBLE_DEVICES=1 python -m graph_signal_diffusion.cli.train \
#     task=stock_price_forecasting_v3 \
#     dataset.root=${SP500_ROOT_07} \
#     dataset.past_window=20 \
#     dataset.future_window=5 \
#     dataset.batch_size=64 \
#     dataset.batch_size_val=200 \
#     dataset.num_workers=4 \
#     dataset.persistent_workers=true \
#     dataset.pin_memory=true \
#     dataset.n_split_chunks=3 \
#     dataset.shuffle_val=false \
#     dataset.standardize_target_in_x_for_revin=true \
#     diffusion.revin=true \
#     model.config.gnn_config.dropout=$dropout \
#     model.config.gnn_config.temporal_mixer.schedule=per_layer \
#     model.config.gnn_config.temporal_mixer.kernel_size=3 \
#     'model.config.gnn_config.temporal_mixer.dilations=[1,1]' \
#     model.config.gnn_config.temporal_mixer.use_pointwise=true \
#     model.config.gnn_config.temporal_mixer.causal=false \
#     model.config.gnn_config.temporal_mixer.attention.enabled=true \
#     model.config.gnn_config.temporal_mixer.attention.num_heads=2 \
#     model.config.gnn_config.temporal_mixer.attention.dropout=$dropout \
#     model.config.gnn_config.temporal_mixer.attention.max_timesteps=5 \
#     'model.config.embedding_config.cond.shared_encoder.temporal.hidden_channels=[128,128,128,128]' \
#     model.config.embedding_config.cond.shared_encoder.temporal.num_layers=4 \
#     model.config.embedding_config.cond.shared_encoder.temporal.mixer.causal=false \
#     'model.config.embedding_config.cond.shared_encoder.temporal.mixer.dilations=[1,2,4,8]' \
#     model.config.embedding_config.cond.shared_encoder.temporal.output.mode=time_varying \
#     model.config.embedding_config.cond.shared_encoder.temporal.output.time_varying.method=none \
#     model.config.embedding_config.cond.block_fusion.mode=cross_attention \
#     model.config.embedding_config.cond.block_fusion.cross_attention.heads=2 \
#     model.config.embedding_config.cond.block_fusion.cross_attention.dropout=$dropout \
#     model.config.embedding_config.cond.block_fusion.cross_attention.bias=true \
#     model.config.embedding_config.cond.block_fusion.cross_attention.causal=false \
#     'model.config.channel_multipliers=[1,2,1]' \
#     'model.config.pooling_config.gamma=[1,1,1]' \
#     model.config.pooling_config.selection_method=stride \
#     model.config.num_bottleneck_layers=1 \
#     model.config.compact_skip_cache=false \
#     model.config.gnn_config.pointwise_sparse_mode=off \
#     model.config.pooling_config.selector_packed_score_mode=off \
#     trainer.selector_temperature_schedule.enabled=false \
#     trainer.max_epochs=5000 \
#     trainer.eval_every_n_epochs=50 \
#     trainer.eval_schedule.phases='[{period: 50, until_epoch: 500}, {period: 100, until_epoch: 1000}, {period: 250}]' \
#     trainer.save_checkpoint_every_n_epochs=500 \
#     trainer.use_amp=true \
#     trainer.n_samples_per_input=10 \
#     trainer.max_num_val_batches=2 trainer.max_num_train_val_batches=1 \
#     wandb.enabled=true









# ***************** SP500 ARM C T=5: gated + self-attention + pos_emb, no downsampling ****** #
# Builds on ARM B T=5 temporal setup (per_layer, bidirectional, pointwise, dilations=[1,1])
# and adds the three new temporal processing features:
#   - gated=true: WaveNet-style tanh(filter)*sigmoid(gate) after depthwise conv.
#   - attention.enabled=true: multi-head self-attention over T (bidirectional).
#   - attention.max_timesteps=5: learnable positional embeddings for forecast horizon.
# Downsampling disabled (same as ARM B) to isolate temporal improvements.
# Runs: fragrant-orca-216
# CUDA_VISIBLE_DEVICES=0 python -m graph_signal_diffusion.cli.train \
#     task=stock_price_forecasting_v3 \
#     dataset.root=${SP500_ROOT} \
#     dataset.past_window=20 \
#     dataset.future_window=5 \
#     dataset.batch_size=64 \
#     dataset.batch_size_val=200 \
#     dataset.num_workers=4 \
#     dataset.persistent_workers=true \
#     dataset.pin_memory=true \
#     model.config.gnn_config.temporal_mixer.schedule=per_layer \
#     model.config.gnn_config.temporal_mixer.kernel_size=3 \
#     'model.config.gnn_config.temporal_mixer.dilations=[1,1]' \
#     model.config.gnn_config.temporal_mixer.use_pointwise=true \
#     model.config.gnn_config.temporal_mixer.causal=false \
#     model.config.gnn_config.temporal_mixer.gated=true \
#     model.config.gnn_config.temporal_mixer.attention.enabled=true \
#     model.config.gnn_config.temporal_mixer.attention.num_heads=4 \
#     model.config.gnn_config.temporal_mixer.attention.dropout=0.0 \
#     model.config.gnn_config.temporal_mixer.attention.max_timesteps=5 \
#     'model.config.channel_multipliers=[1,1,1,1]' \
#     'model.config.pooling_config.gamma=[1,1,1,1]' \
#     model.config.pooling_config.selection_method=stride \
#     model.config.num_bottleneck_layers=1 \
#     model.config.compact_skip_cache=false \
#     model.config.gnn_config.pointwise_sparse_mode=off \
#     model.config.pooling_config.selector_packed_score_mode=off \
#     trainer.selector_temperature_schedule.enabled=false \
#     trainer.use_amp=true \
#     trainer.n_samples_per_input=10 \
#     trainer.max_num_val_batches=1 trainer.max_num_train_val_batches=1 \
#     wandb.enabled=true


# ***************** SP500 ARM D T=5: larger cond encoder, no downsampling ****************** #
# Tests whether the conditioning encoder is the bottleneck.
# ARM B/C stuck at ~50% direction accuracy — temporal processing improvements
# made no difference. Hypothesis: the 35K-param cond encoder (2-layer, 64-wide,
# causal) can't extract useful signal from 20 days × 12 features.
#
# Changes vs ARM B T=5:
#   - hidden_channels: [64,64] → [256,256,256,256] (4 layers, 4× wider)
#   - causal: true → false (bidirectional; past window is fully observed)
#   - dilations: [1,4] → [1,2,4,8] (dense multi-scale coverage of 20-step past)
#   - kernel_size: 5 → 5 (unchanged, RF per layer = 5/9/17/33 with dilations)
# Cond encoder grows from ~35K to ~322K params (9.2×). Total model ~1.48M (+24% vs ARM C).
# Denoiser temporal setup matches ARM B (per_layer, bidirectional, dilations=[1,1]).
# Runs: defiant-python-6
# CUDA_VISIBLE_DEVICES=1 python -m graph_signal_diffusion.cli.train \
#     task=stock_price_forecasting_v3 \
#     dataset.root=${SP500_ROOT} \
#     dataset.past_window=20 \
#     dataset.future_window=5 \
#     dataset.batch_size=64 \
#     dataset.batch_size_val=200 \
#     dataset.num_workers=4 \
#     dataset.persistent_workers=true \
#     dataset.pin_memory=true \
#     model.config.gnn_config.temporal_mixer.schedule=per_layer \
#     model.config.gnn_config.temporal_mixer.kernel_size=3 \
#     'model.config.gnn_config.temporal_mixer.dilations=[1,1]' \
#     model.config.gnn_config.temporal_mixer.use_pointwise=true \
#     model.config.gnn_config.temporal_mixer.causal=false \
#     'model.config.embedding_config.cond.shared_encoder.temporal.hidden_channels=[256,256,256,256]' \
#     model.config.embedding_config.cond.shared_encoder.temporal.num_layers=4 \
#     model.config.embedding_config.cond.shared_encoder.temporal.mixer.causal=false \
#     'model.config.embedding_config.cond.shared_encoder.temporal.mixer.dilations=[1,2,4,8]' \
#     'model.config.channel_multipliers=[1,1,1,1]' \
#     'model.config.pooling_config.gamma=[1,1,1,1]' \
#     model.config.pooling_config.selection_method=stride \
#     model.config.num_bottleneck_layers=1 \
#     model.config.compact_skip_cache=false \
#     model.config.gnn_config.pointwise_sparse_mode=off \
#     model.config.pooling_config.selector_packed_score_mode=off \
#     trainer.selector_temperature_schedule.enabled=false \
#     trainer.use_amp=true \
#     trainer.n_samples_per_input=10 \
#     trainer.max_num_val_batches=1 trainer.max_num_train_val_batches=1 \
#     wandb.enabled=true


# ***************** SP500 ARM D-v2 T=5: cross-attention full past, no downsampling ********* #
# Removes temporal collapse from the conditioning path entirely.
# ARM D still collapses cond to (B, N, 128) via learned projection + temporal pooling.
# ARM D-v2 outputs (B, T=20, N, embed_dim) and uses cross-attention at each denoiser
# level: Q=denoiser features (T=5), KV=full encoded past (T=20).
# Each forecast day dynamically attends to whichever past days are relevant.
#
# Changes vs ARM D:
#   - time_varying.method: none (skip learned projection, pass full T=20 through)
#   - block_fusion.mode: cross_attention (replaces concat fusion)
#   - Cross-attention: 4 heads, no dropout, bidirectional (causal=false)
# Cond encoder architecture same as ARM D (4 layers, 256-wide, bidirectional, [1,2,4,8]).
# Denoiser temporal setup matches ARM B (per_layer, bidirectional, dilations=[1,1]).
# Runs: aquamarine-dragon-614
# CUDA_VISIBLE_DEVICES=0 python -m graph_signal_diffusion.cli.train \
#     task=stock_price_forecasting_v3 \
#     dataset.root=${SP500_ROOT} \
#     dataset.past_window=20 \
#     dataset.future_window=5 \
#     dataset.batch_size=64 \
#     dataset.batch_size_val=200 \
#     dataset.num_workers=4 \
#     dataset.persistent_workers=true \
#     dataset.pin_memory=true \
#     model.config.gnn_config.temporal_mixer.schedule=per_layer \
#     model.config.gnn_config.temporal_mixer.kernel_size=3 \
#     'model.config.gnn_config.temporal_mixer.dilations=[1,1]' \
#     model.config.gnn_config.temporal_mixer.use_pointwise=true \
#     model.config.gnn_config.temporal_mixer.causal=false \
#     'model.config.embedding_config.cond.shared_encoder.temporal.hidden_channels=[256,256,256,256]' \
#     model.config.embedding_config.cond.shared_encoder.temporal.num_layers=4 \
#     model.config.embedding_config.cond.shared_encoder.temporal.mixer.causal=false \
#     'model.config.embedding_config.cond.shared_encoder.temporal.mixer.dilations=[1,2,4,8]' \
#     model.config.embedding_config.cond.shared_encoder.temporal.output.mode=time_varying \
#     model.config.embedding_config.cond.shared_encoder.temporal.output.time_varying.method=none \
#     model.config.embedding_config.cond.block_fusion.mode=cross_attention \
#     model.config.embedding_config.cond.block_fusion.cross_attention.heads=4 \
#     model.config.embedding_config.cond.block_fusion.cross_attention.dropout=0.0 \
#     model.config.embedding_config.cond.block_fusion.cross_attention.bias=true \
#     model.config.embedding_config.cond.block_fusion.cross_attention.causal=false \
#     'model.config.channel_multipliers=[1,1,1,1]' \
#     'model.config.pooling_config.gamma=[1,1,1,1]' \
#     model.config.pooling_config.selection_method=stride \
#     model.config.num_bottleneck_layers=1 \
#     model.config.compact_skip_cache=false \
#     model.config.gnn_config.pointwise_sparse_mode=off \
#     model.config.pooling_config.selector_packed_score_mode=off \
#     trainer.selector_temperature_schedule.enabled=false \
#     trainer.use_amp=true \
#     trainer.n_samples_per_input=10 \
#     trainer.max_num_val_batches=1 trainer.max_num_train_val_batches=1 \
#     wandb.enabled=true


# ***************** SP500 ARM D-v2b T=5: diamond U-Net + cross-attention full past ********* #
# Builds on D-v2's cross-attention conditioning (method=none, full T=20 past) with
# a larger 5-level diamond-shaped denoiser and diamond-shaped cond encoder.
# Total: ~8.5M params (vs ~1.5M in D-v2, ~1.19M in ARM B/C).
#
# Denoiser (5-level diamond, base_channels=64, no downsampling):
#   Level 0: 64ch   (input/output — lightweight)
#   Level 1: 128ch  (feature expansion)
#   Level 2: 256ch  (deepest — maximum capacity)
#   Level 3: 128ch  (feature contraction)
#   Level 4: 64ch   (output refinement)
#
# Cond encoder (4-layer diamond, bidirectional, dilations=[1,2,4,8]):
#   Layer 0: 64ch   (local patterns, dilation=1)
#   Layer 1: 256ch  (mid-range, dilation=2)
#   Layer 2: 1024ch (widest — complex temporal features, dilation=4, RF=17)
#   Layer 3: 256ch  (long-range, dilation=8)
#   → output_proj 256→256 (cond_embed_dim), no compression
#   Cond encoder alone: ~1.87M params.
#
# Cross-attention at each level: Q=denoiser (T=5), KV=full encoded past (T=20).
# Scaled embeddings: time_embed_dim=256, cond_embed_dim=256.
#
# Training: max_epochs=5000, batch_size=24 (~12-15GB on 24GB GPU).
#   ~2.7× more gradient updates per epoch vs B=64 arms.
# Eval schedule: sparse mid-training (period=50), denser at end.
#   3 val batches per eval for more reliable metric estimation.
# Runs:
# CUDA_VISIBLE_DEVICES=1 python -m graph_signal_diffusion.cli.train \
#     task=stock_price_forecasting_v3 \
#     dataset.root=${SP500_ROOT} \
#     dataset.past_window=20 \
#     dataset.future_window=5 \
#     dataset.batch_size=24 \
#     dataset.batch_size_val=100 \
#     dataset.num_workers=4 \
#     dataset.persistent_workers=true \
#     dataset.pin_memory=true \
#     model.config.gnn_config.temporal_mixer.schedule=per_layer \
#     model.config.gnn_config.temporal_mixer.kernel_size=3 \
#     'model.config.gnn_config.temporal_mixer.dilations=[1,1]' \
#     model.config.gnn_config.temporal_mixer.use_pointwise=true \
#     model.config.gnn_config.temporal_mixer.causal=false \
#     model.config.embedding_config.time_embed_dim=256 \
#     model.config.embedding_config.cond.embed_dim=256 \
#     'model.config.embedding_config.cond.shared_encoder.temporal.hidden_channels=[64,256,1024,256]' \
#     model.config.embedding_config.cond.shared_encoder.temporal.num_layers=4 \
#     model.config.embedding_config.cond.shared_encoder.temporal.mixer.causal=false \
#     'model.config.embedding_config.cond.shared_encoder.temporal.mixer.dilations=[1,2,4,8]' \
#     model.config.embedding_config.cond.shared_encoder.temporal.output.mode=time_varying \
#     model.config.embedding_config.cond.shared_encoder.temporal.output.time_varying.method=none \
#     model.config.embedding_config.cond.block_fusion.mode=cross_attention \
#     model.config.embedding_config.cond.block_fusion.cross_attention.heads=4 \
#     model.config.embedding_config.cond.block_fusion.cross_attention.dropout=0.0 \
#     model.config.embedding_config.cond.block_fusion.cross_attention.bias=true \
#     model.config.embedding_config.cond.block_fusion.cross_attention.causal=false \
#     'model.config.channel_multipliers=[1,2,4,2,1]' \
#     'model.config.pooling_config.gamma=[1,1,1,1,1]' \
#     model.config.pooling_config.selection_method=stride \
#     model.config.num_bottleneck_layers=1 \
#     model.config.compact_skip_cache=false \
#     model.config.gnn_config.pointwise_sparse_mode=off \
#     model.config.pooling_config.selector_packed_score_mode=off \
#     trainer.selector_temperature_schedule.enabled=false \
#     trainer.max_epochs=5000 \
#     trainer.eval_every_n_epochs=50 \
#     trainer.eval_schedule.phases='[{period: 200, until_epoch: 200}, {period: 50, until_epoch: -200}, {period: 50}]' \
#     trainer.use_amp=true \
#     trainer.n_samples_per_input=10 \
#     trainer.max_num_val_batches=3 trainer.max_num_train_val_batches=1 \
#     wandb.enabled=true








# ***************** SP500 ARM D-v2c T=5: diamond denoiser + regularized cross-attention **** #
# Middle ground between D-v2 (~600K params, stable but no signal) and D-v2b (~8.5M, overfits).
# Estimated ~1.5-2M params.
#
# Denoiser (5-level diamond, base_channels=64, no downsampling):
#   Level 0: 64ch, Level 1: 128ch, Level 2: 256ch, Level 3: 128ch, Level 4: 64ch
#   Tests whether multi-scale channel hierarchy helps without the param explosion of D-v2b.
#
# Cond encoder (4-layer flat [128,128,128,128], bidirectional, dilations=[1,2,4,8]):
#   D-v2 style flat encoder but narrower (128 vs D-v2's 256). Avoids D-v2b's 1.87M
#   diamond cond encoder which dominated its param budget.
#
# Regularization (new vs D-v2/D-v2b):
#   - cross_attention.dropout=0.05 (was 0.0): regularizes the Q=T5, KV=T20 attention path
#   - temporal self-attention enabled with dropout=0.05: global temporal reasoning in denoiser
#     (ARM C added this but saw no improvement — hypothesis: it helps when paired with
#     cross-attention conditioning and sufficient denoiser capacity)
#   - GNN dropout=0.05 (unchanged, from yaml default)
#
# Training: batch_size=32, max_epochs=5000, eval schedule same as D-v2b.
# Runs: scrupulous-dove-578
# dropout=0.1
# CUDA_VISIBLE_DEVICES=0 python -m graph_signal_diffusion.cli.train \
#     task=stock_price_forecasting_v3 \
#     dataset.root=${SP500_ROOT} \
#     dataset.past_window=20 \
#     dataset.future_window=5 \
#     dataset.batch_size=32 \
#     dataset.batch_size_val=200 \
#     dataset.num_workers=4 \
#     dataset.persistent_workers=true \
#     dataset.pin_memory=true \
#     model.config.gnn_config.dropout=$dropout \
#     model.config.gnn_config.temporal_mixer.schedule=per_layer \
#     model.config.gnn_config.temporal_mixer.kernel_size=3 \
#     'model.config.gnn_config.temporal_mixer.dilations=[1,1]' \
#     model.config.gnn_config.temporal_mixer.use_pointwise=true \
#     model.config.gnn_config.temporal_mixer.causal=false \
#     model.config.gnn_config.temporal_mixer.attention.enabled=true \
#     model.config.gnn_config.temporal_mixer.attention.num_heads=4 \
#     model.config.gnn_config.temporal_mixer.attention.dropout=$dropout \
#     model.config.gnn_config.temporal_mixer.attention.max_timesteps=5 \
#     'model.config.embedding_config.cond.shared_encoder.temporal.hidden_channels=[128,128,128,128]' \
#     model.config.embedding_config.cond.shared_encoder.temporal.num_layers=4 \
#     model.config.embedding_config.cond.shared_encoder.temporal.mixer.causal=false \
#     'model.config.embedding_config.cond.shared_encoder.temporal.mixer.dilations=[1,2,4,8]' \
#     model.config.embedding_config.cond.shared_encoder.temporal.output.mode=time_varying \
#     model.config.embedding_config.cond.shared_encoder.temporal.output.time_varying.method=none \
#     model.config.embedding_config.cond.block_fusion.mode=cross_attention \
#     model.config.embedding_config.cond.block_fusion.cross_attention.heads=4 \
#     model.config.embedding_config.cond.block_fusion.cross_attention.dropout=$dropout \
#     model.config.embedding_config.cond.block_fusion.cross_attention.bias=true \
#     model.config.embedding_config.cond.block_fusion.cross_attention.causal=false \
#     'model.config.channel_multipliers=[1,2,4,2,1]' \
#     'model.config.pooling_config.gamma=[1,1,1,1,1]' \
#     model.config.pooling_config.selection_method=stride \
#     model.config.num_bottleneck_layers=1 \
#     model.config.compact_skip_cache=false \
#     model.config.gnn_config.pointwise_sparse_mode=off \
#     model.config.pooling_config.selector_packed_score_mode=off \
#     trainer.selector_temperature_schedule.enabled=false \
#     trainer.max_epochs=5000 \
#     trainer.eval_every_n_epochs=50 \
#     trainer.eval_schedule.phases='[{period: 100, until_epoch: 200}, {period: 50, until_epoch: -200}, {period: 50}]' \
#     trainer.use_amp=true \
#     trainer.n_samples_per_input=10 \
#     trainer.max_num_val_batches=2 trainer.max_num_train_val_batches=1 \
#     wandb.enabled=true


# ***************** SP500 ARM D-v2d T=5: D-v2c + truncated 500-day training window ******** #
# Identical architecture to D-v2c (diamond denoiser, regularized cross-attention, dropout=0.1).
# Tests the hypothesis that training on 8 years of data (2016-2024) hurts because the
# recent regime (low volatility clustering, weak market factor) differs from the historical
# average. Truncates training to the most recent 500 days (~2 years) before the val split.
#
# Distributional comparison showed:
#   - Train-val (2023-04 to 2024-04) and Val (2024-03 to 2025-03) are nearly identical
#     (KS rejection 4.5%) but full train (2016-2024) differs (KS rejection 15.4%)
#   - Volatility clustering: train=0.227, train-val=0.032, val=0.030
#   - Full training data includes COVID crash, rate hikes — very different regime
#
# Expected outcome: better val metrics (especially structural) due to regime-matched training.
# Risk: fewer training samples (500 vs 1870) may hurt capacity-hungry model.
# dropout=0.1
# CUDA_VISIBLE_DEVICES=1 python -m graph_signal_diffusion.cli.train \
#     task=stock_price_forecasting_v3 \
#     dataset.root=${SP500_ROOT} \
#     dataset.past_window=20 \
#     dataset.future_window=5 \
#     dataset.batch_size=32 \
#     dataset.batch_size_val=200 \
#     dataset.num_workers=4 \
#     dataset.persistent_workers=true \
#     dataset.pin_memory=true \
#     dataset.train_window_days=500 \
#     model.config.gnn_config.dropout=$dropout \
#     model.config.gnn_config.temporal_mixer.schedule=per_layer \
#     model.config.gnn_config.temporal_mixer.kernel_size=3 \
#     'model.config.gnn_config.temporal_mixer.dilations=[1,1]' \
#     model.config.gnn_config.temporal_mixer.use_pointwise=true \
#     model.config.gnn_config.temporal_mixer.causal=false \
#     model.config.gnn_config.temporal_mixer.attention.enabled=true \
#     model.config.gnn_config.temporal_mixer.attention.num_heads=4 \
#     model.config.gnn_config.temporal_mixer.attention.dropout=$dropout \
#     model.config.gnn_config.temporal_mixer.attention.max_timesteps=5 \
#     'model.config.embedding_config.cond.shared_encoder.temporal.hidden_channels=[128,128,128,128]' \
#     model.config.embedding_config.cond.shared_encoder.temporal.num_layers=4 \
#     model.config.embedding_config.cond.shared_encoder.temporal.mixer.causal=false \
#     'model.config.embedding_config.cond.shared_encoder.temporal.mixer.dilations=[1,2,4,8]' \
#     model.config.embedding_config.cond.shared_encoder.temporal.output.mode=time_varying \
#     model.config.embedding_config.cond.shared_encoder.temporal.output.time_varying.method=none \
#     model.config.embedding_config.cond.block_fusion.mode=cross_attention \
#     model.config.embedding_config.cond.block_fusion.cross_attention.heads=4 \
#     model.config.embedding_config.cond.block_fusion.cross_attention.dropout=$dropout \
#     model.config.embedding_config.cond.block_fusion.cross_attention.bias=true \
#     model.config.embedding_config.cond.block_fusion.cross_attention.causal=false \
#     'model.config.channel_multipliers=[1,2,4,2,1]' \
#     'model.config.pooling_config.gamma=[1,1,1,1,1]' \
#     model.config.pooling_config.selection_method=stride \
#     model.config.num_bottleneck_layers=1 \
#     model.config.compact_skip_cache=false \
#     model.config.gnn_config.pointwise_sparse_mode=off \
#     model.config.pooling_config.selector_packed_score_mode=off \
#     trainer.selector_temperature_schedule.enabled=false \
#     trainer.max_epochs=5000 \
#     trainer.eval_every_n_epochs=50 \
#     trainer.eval_schedule.phases='[{period: 100, until_epoch: 200}, {period: 50, until_epoch: -200}, {period: 50}]' \
#     trainer.use_amp=true \
#     trainer.n_samples_per_input=10 \
#     trainer.max_num_val_batches=2 trainer.max_num_train_val_batches=1 \
#     wandb.enabled=true


# ***************** SP500 ARM D-v2e T=5: D-v2d + RevIN ******************************** #
# Identical to D-v2d (truncated 500-day window, diamond denoiser, regularized cross-attention)
# plus RevIN (Reversible Instance Normalization).
#
# RevIN removes per-stock scale/shift from the diffusion target x0 using conditioning window
# statistics, so the model can focus on cross-stock structure and temporal dynamics instead of
# learning per-stock normalization. Both flags are required:
#   - diffusion.revin=true: enables normalize-before-noise / denormalize-after-sampling
#   - dataset.standardize_target_in_x_for_revin=true: aligns conditioning target channel
#     to the same standardized space as y (otherwise RevIN stats are in wrong scale)
#
# Hypothesis: RevIN frees model capacity that was wasted learning per-stock normalization,
# delaying the onset of overfitting (D-v2c/D-v2d both peaked at epoch 450).
# If D-v2e peaks later (e.g. epoch 800+), RevIN is effective and a follow-up with reduced
# capacity is warranted. If it still peaks at ~450, the overfitting is not capacity-driven.
#
# max_epochs reduced from 5000 to 2000 (both D-v2c/D-v2d peaked at 450, no point training 5000).
# Eval schedule: every 50 epochs from the start for finer-grained peak detection.
#
# Control: D-v2d (hypnotic-sunfish-345) continues on GPU 1 as the no-RevIN baseline.
# Runs: economic-tapir-531
# dropout=0.1
# CUDA_VISIBLE_DEVICES=0 python -m graph_signal_diffusion.cli.train \
#     task=stock_price_forecasting_v3 \
#     dataset.root=${SP500_ROOT} \
#     dataset.past_window=20 \
#     dataset.future_window=5 \
#     dataset.batch_size=32 \
#     dataset.batch_size_val=200 \
#     dataset.num_workers=4 \
#     dataset.persistent_workers=true \
#     dataset.pin_memory=true \
#     dataset.train_window_days=500 \
#     dataset.standardize_target_in_x_for_revin=true \
#     diffusion.revin=true \
#     model.config.gnn_config.dropout=$dropout \
#     model.config.gnn_config.temporal_mixer.schedule=per_layer \
#     model.config.gnn_config.temporal_mixer.kernel_size=3 \
#     'model.config.gnn_config.temporal_mixer.dilations=[1,1]' \
#     model.config.gnn_config.temporal_mixer.use_pointwise=true \
#     model.config.gnn_config.temporal_mixer.causal=false \
#     model.config.gnn_config.temporal_mixer.attention.enabled=true \
#     model.config.gnn_config.temporal_mixer.attention.num_heads=4 \
#     model.config.gnn_config.temporal_mixer.attention.dropout=$dropout \
#     model.config.gnn_config.temporal_mixer.attention.max_timesteps=5 \
#     'model.config.embedding_config.cond.shared_encoder.temporal.hidden_channels=[128,128,128,128]' \
#     model.config.embedding_config.cond.shared_encoder.temporal.num_layers=4 \
#     model.config.embedding_config.cond.shared_encoder.temporal.mixer.causal=false \
#     'model.config.embedding_config.cond.shared_encoder.temporal.mixer.dilations=[1,2,4,8]' \
#     model.config.embedding_config.cond.shared_encoder.temporal.output.mode=time_varying \
#     model.config.embedding_config.cond.shared_encoder.temporal.output.time_varying.method=none \
#     model.config.embedding_config.cond.block_fusion.mode=cross_attention \
#     model.config.embedding_config.cond.block_fusion.cross_attention.heads=4 \
#     model.config.embedding_config.cond.block_fusion.cross_attention.dropout=$dropout \
#     model.config.embedding_config.cond.block_fusion.cross_attention.bias=true \
#     model.config.embedding_config.cond.block_fusion.cross_attention.causal=false \
#     'model.config.channel_multipliers=[1,2,4,2,1]' \
#     'model.config.pooling_config.gamma=[1,1,1,1,1]' \
#     model.config.pooling_config.selection_method=stride \
#     model.config.num_bottleneck_layers=1 \
#     model.config.compact_skip_cache=false \
#     model.config.gnn_config.pointwise_sparse_mode=off \
#     model.config.pooling_config.selector_packed_score_mode=off \
#     trainer.selector_temperature_schedule.enabled=false \
#     trainer.max_epochs=2000 \
#     trainer.eval_every_n_epochs=50 \
#     trainer.eval_schedule.phases='[{period: 50, until_epoch: 500}, {period: 100, until_epoch: -200}, {period: 50}]' \
#     trainer.use_amp=true \
#     trainer.n_samples_per_input=10 \
#     trainer.max_num_val_batches=2 trainer.max_num_train_val_batches=1 \
#     wandb.enabled=true


# ***************** SP500 ARM D-v2f T=5: D-v2e + 3-chunk interleaved split ************ #
# Identical to D-v2e (RevIN, temporal attention) but with two key changes:
#   1. Interleaved 3-chunk chronological split (n_split_chunks=3): the ~2328-sample
#      timeline is divided into 3 equal chunks (~776 each), each split 80-10-10
#      internally, then concatenated. Val/test windows are spread across the full
#      10-year history instead of concentrated in the final 20%.
#   2. Full training data (train_window_days removed): with interleaved chunks, each
#      chunk's train portion is already close in time to its val/test, so the 500-day
#      truncation is no longer needed. This gives ~1862 train samples (vs ~500 in D-v2e).
#
# Also enables shuffle_val=false for deterministic structural metric evaluation.
#
# Boundary leakage: ~24 samples per boundary (~31% of each 78-sample val/test chunk),
# acceptable for 3 chunks.
#
# Hypothesis: Interleaved eval windows + full training data will produce more robust
# val metrics and better generalization. If D-v2e's val metrics were inflated by only
# evaluating on the most recent 10%, D-v2f will show more honest scores.
#
# Control: D-v2e (economic-tapir-531) as the single-chunk, 500-day-window baseline.
# Runs: stoic-polecat-491
# dropout=0.1
# CUDA_VISIBLE_DEVICES=1 python -m graph_signal_diffusion.cli.train \
#     task=stock_price_forecasting_v3 \
#     dataset.root=${SP500_ROOT} \
#     dataset.past_window=20 \
#     dataset.future_window=5 \
#     dataset.batch_size=32 \
#     dataset.batch_size_val=200 \
#     dataset.num_workers=4 \
#     dataset.persistent_workers=true \
#     dataset.pin_memory=true \
#     dataset.n_split_chunks=3 \
#     dataset.shuffle_val=false \
#     dataset.standardize_target_in_x_for_revin=true \
#     diffusion.revin=true \
#     model.config.gnn_config.dropout=$dropout \
#     model.config.gnn_config.temporal_mixer.schedule=per_layer \
#     model.config.gnn_config.temporal_mixer.kernel_size=3 \
#     'model.config.gnn_config.temporal_mixer.dilations=[1,1]' \
#     model.config.gnn_config.temporal_mixer.use_pointwise=true \
#     model.config.gnn_config.temporal_mixer.causal=false \
#     model.config.gnn_config.temporal_mixer.attention.enabled=true \
#     model.config.gnn_config.temporal_mixer.attention.num_heads=4 \
#     model.config.gnn_config.temporal_mixer.attention.dropout=$dropout \
#     model.config.gnn_config.temporal_mixer.attention.max_timesteps=5 \
#     'model.config.embedding_config.cond.shared_encoder.temporal.hidden_channels=[128,128,128,128]' \
#     model.config.embedding_config.cond.shared_encoder.temporal.num_layers=4 \
#     model.config.embedding_config.cond.shared_encoder.temporal.mixer.causal=false \
#     'model.config.embedding_config.cond.shared_encoder.temporal.mixer.dilations=[1,2,4,8]' \
#     model.config.embedding_config.cond.shared_encoder.temporal.output.mode=time_varying \
#     model.config.embedding_config.cond.shared_encoder.temporal.output.time_varying.method=none \
#     model.config.embedding_config.cond.block_fusion.mode=cross_attention \
#     model.config.embedding_config.cond.block_fusion.cross_attention.heads=4 \
#     model.config.embedding_config.cond.block_fusion.cross_attention.dropout=$dropout \
#     model.config.embedding_config.cond.block_fusion.cross_attention.bias=true \
#     model.config.embedding_config.cond.block_fusion.cross_attention.causal=false \
#     'model.config.channel_multipliers=[1,2,4,2,1]' \
#     'model.config.pooling_config.gamma=[1,1,1,1,1]' \
#     model.config.pooling_config.selection_method=stride \
#     model.config.num_bottleneck_layers=1 \
#     model.config.compact_skip_cache=false \
#     model.config.gnn_config.pointwise_sparse_mode=off \
#     model.config.pooling_config.selector_packed_score_mode=off \
#     trainer.selector_temperature_schedule.enabled=false \
#     trainer.max_epochs=2000 \
#     trainer.eval_every_n_epochs=50 \
#     trainer.eval_schedule.phases='[{period: 100, until_epoch: 200}, {period: 25, until_epoch: 600}, {period: 50}]' \
#     trainer.use_amp=true \
#     trainer.n_samples_per_input=20 \
#     trainer.max_num_val_batches=2 trainer.max_num_train_val_batches=1 \
#     wandb.enabled=true





# ***************** SP500 ARM E T=5: longer past + lean model ***************************** #
# New direction: tests whether direction accuracy (stuck at ~50% across all D-v2 arms)
# is limited by information input rather than model capacity.
#
# Two changes vs D-v2f (stoic-polecat-491):
#   1. past_window=40 (vs 20): ~2 months of history captures medium-term momentum.
#      Cross-attention path (Q=T5, KV=T40) handles longer context natively.
#   2. Lean model: flat [1,1,1,1] denoiser (vs diamond [1,2,4,2,1]), narrower cond
#      encoder [64,64,64,64,64] (vs [128,128,128,128]). 1.25M params vs 6.03M.
#      Faster training for quicker iteration.
#
# Cond encoder: 5 layers with dilations=[1,2,4,8,16] and kernel_size=5.
#   RF = 1 + 4*(1+2+4+8+16) = 125, easily covering 40-step past window.
#
# Kept from D-v2f: 3-chunk interleaved split, RevIN, DDIM (η=0.2, linear schedule),
# per-layer temporal mixer with self-attention, cross-attention full-past conditioning.
#
# Note: cosine beta_schedule was tried (fragrant-toad-616) but produced NaN val metrics
# and degenerate (constant) samples — reverted to linear.
#
# Hypothesis: D-v2f's diamond denoiser learns good marginal return distributions
# (val CRPS=0.921 beats GRW 1.293 by 29%) but 20-day past is too short for
# trend detection. Doubling the context window should help directional signal.
# If direction accuracy improves, it's the information (not the capacity) that matters.
#
# Control: D-v2f (stoic-polecat-491) as the capacity+short-context baseline.
# Runs: big-dragon-28 (OOM at batch_size=64), fragrant-toad-616 (cosine schedule → NaN metrics), enchanted-vole-122 (linear schedule)
# dropout=0.1
# CUDA_VISIBLE_DEVICES=0 python -m graph_signal_diffusion.cli.train \
#     task=stock_price_forecasting_v3 \
#     dataset.root=${SP500_ROOT} \
#     dataset.past_window=40 \
#     dataset.future_window=5 \
#     dataset.batch_size=32 \
#     dataset.batch_size_val=200 \
#     dataset.num_workers=4 \
#     dataset.persistent_workers=true \
#     dataset.pin_memory=true \
#     dataset.n_split_chunks=3 \
#     dataset.shuffle_val=true \
#     dataset.standardize_target_in_x_for_revin=true \
#     diffusion.revin=true \
#     model.config.gnn_config.dropout=$dropout \
#     model.config.gnn_config.temporal_mixer.schedule=per_layer \
#     model.config.gnn_config.temporal_mixer.kernel_size=3 \
#     'model.config.gnn_config.temporal_mixer.dilations=[1,1]' \
#     model.config.gnn_config.temporal_mixer.use_pointwise=true \
#     model.config.gnn_config.temporal_mixer.causal=false \
#     model.config.gnn_config.temporal_mixer.attention.enabled=true \
#     model.config.gnn_config.temporal_mixer.attention.num_heads=4 \
#     model.config.gnn_config.temporal_mixer.attention.dropout=$dropout \
#     model.config.gnn_config.temporal_mixer.attention.max_timesteps=5 \
#     'model.config.embedding_config.cond.shared_encoder.temporal.hidden_channels=[64,64,64,64,64]' \
#     model.config.embedding_config.cond.shared_encoder.temporal.num_layers=5 \
#     model.config.embedding_config.cond.shared_encoder.temporal.mixer.causal=false \
#     model.config.embedding_config.cond.shared_encoder.temporal.mixer.kernel_size=5 \
#     'model.config.embedding_config.cond.shared_encoder.temporal.mixer.dilations=[1,2,4,8,16]' \
#     model.config.embedding_config.cond.shared_encoder.temporal.output.mode=time_varying \
#     model.config.embedding_config.cond.shared_encoder.temporal.output.time_varying.method=none \
#     model.config.embedding_config.cond.block_fusion.mode=cross_attention \
#     model.config.embedding_config.cond.block_fusion.cross_attention.heads=4 \
#     model.config.embedding_config.cond.block_fusion.cross_attention.dropout=$dropout \
#     model.config.embedding_config.cond.block_fusion.cross_attention.bias=true \
#     model.config.embedding_config.cond.block_fusion.cross_attention.causal=false \
#     'model.config.channel_multipliers=[1,1,1,1]' \
#     'model.config.pooling_config.gamma=[1,1,1,1]' \
#     model.config.pooling_config.selection_method=stride \
#     model.config.num_bottleneck_layers=1 \
#     model.config.compact_skip_cache=false \
#     model.config.gnn_config.pointwise_sparse_mode=off \
#     model.config.pooling_config.selector_packed_score_mode=off \
#     trainer.selector_temperature_schedule.enabled=false \
#     trainer.max_epochs=2000 \
#     trainer.eval_every_n_epochs=50 \
#     trainer.eval_schedule.phases='[{period: 50, until_epoch: 100}, {period: 25, until_epoch: 600}, {period: 50}]' \
#     trainer.use_amp=true \
#     trainer.n_samples_per_input=20 \
#     trainer.max_num_val_batches=2 trainer.max_num_train_val_batches=1 \
#     wandb.enabled=true






## ***************** SP500 ARM B T=20: improved temporal, no downsampling ****************** #
# Isolates temporal processing improvements from learned downsampling.
# ARM A's selector is still soft/uniform at epoch 570 (entropy ~0.70), so disabling
# DS removes a confounder without losing meaningful structure.
#
# Temporal changes vs ARM A:
#   - schedule: per_layer (interleaved with each GNN layer, not just once per block).
#     With num_layers=2, dilations list must have length 2.
#   - dilations: [1,3] with kernel=3 gives per-layer RF = 5 and 7.
#     Stacked across 2 layers in each of 3 encoder levels + bottleneck + decoder,
#     the effective RF compounds well beyond T=20. Critically, dilation=1 sees
#     adjacent timesteps (lag-1 autocorrelation), which dilation=5 in ARM A misses.
#   - use_pointwise: true enables cross-channel temporal reasoning in each mixer.
#     ARM A's pointwise=false means each of the 64 channels learns an independent
#     temporal filter with zero interaction — too thin for directional signal.
#   - causal: false — the denoiser sees the entire noisy future window at once,
#     so bidirectional mixing is correct. Causal masking only makes sense for the
#     conditioning encoder (past observations) or autoregressive generation.
#     With ARM A's causal=true + dilation=5, the first ~4 timesteps in T=20 had
#     zero temporal context (RF didn't reach any neighbor).
#
# Downsampling disabled (clean temporal-only ablation):
#   - gamma=[1,1,1,1], selection_method=stride, no temperature schedule.
#   - num_bottleneck_layers=1 (no multi-layer bottleneck without pooled graph).
#   - compact_skip_cache/sparse pointwise/packed scores off (not needed without DS).
# CUDA_VISIBLE_DEVICES=0 python -m graph_signal_diffusion.cli.train \
#     task=stock_price_forecasting_v3 \
#     dataset.root=${SP500_ROOT} \
#     dataset.past_window=20 \
#     dataset.future_window=20 \
#     dataset.batch_size=64 \
#     dataset.batch_size_val=200 \
#     dataset.num_workers=4 \
#     dataset.persistent_workers=true \
#     dataset.pin_memory=true \
#     model.config.gnn_config.temporal_mixer.schedule=per_layer \
#     model.config.gnn_config.temporal_mixer.kernel_size=3 \
#     'model.config.gnn_config.temporal_mixer.dilations=[1,3]' \
#     model.config.gnn_config.temporal_mixer.use_pointwise=true \
#     model.config.gnn_config.temporal_mixer.causal=false \
#     'model.config.channel_multipliers=[1,1,1,1]' \
#     'model.config.pooling_config.gamma=[1,1,1,1]' \
#     model.config.pooling_config.selection_method=stride \
#     model.config.num_bottleneck_layers=1 \
#     model.config.compact_skip_cache=false \
#     model.config.gnn_config.pointwise_sparse_mode=off \
#     model.config.pooling_config.selector_packed_score_mode=off \
#     trainer.selector_temperature_schedule.enabled=false \
#     trainer.use_amp=true \
#     trainer.n_samples_per_input=10 \
#     trainer.max_num_val_batches=1 trainer.max_num_train_val_batches=0 \
#     wandb.enabled=true


## ***************** SP500 ARM B T=10: improved temporal, no downsampling ****************** #
# Same temporal + no-DS setup as ARM B T=20, adapted for T=10:
#   - dilations: [1,2] with kernel=3 gives per-layer RF = 5 and 5.
#     Dense coverage of the 10-step window via stacking; dilation=1 preserves
#     lag-1 adjacency, dilation=2 covers every-other-day patterns.
# CUDA_VISIBLE_DEVICES=1 python -m graph_signal_diffusion.cli.train \
#     task=stock_price_forecasting_v3 \
#     dataset.root=${SP500_ROOT} \
#     dataset.past_window=20 \
#     dataset.future_window=10 \
#     dataset.batch_size=64 \
#     dataset.batch_size_val=200 \
#     dataset.num_workers=4 \
#     dataset.persistent_workers=true \
#     dataset.pin_memory=true \
#     model.config.gnn_config.temporal_mixer.schedule=per_layer \
#     model.config.gnn_config.temporal_mixer.kernel_size=3 \
#     'model.config.gnn_config.temporal_mixer.dilations=[1,2]' \
#     model.config.gnn_config.temporal_mixer.use_pointwise=true \
#     model.config.gnn_config.temporal_mixer.causal=false \
#     'model.config.channel_multipliers=[1,1,1,1]' \
#     'model.config.pooling_config.gamma=[1,1,1,1]' \
#     model.config.pooling_config.selection_method=stride \
#     model.config.num_bottleneck_layers=1 \
#     model.config.compact_skip_cache=false \
#     model.config.gnn_config.pointwise_sparse_mode=off \
#     model.config.pooling_config.selector_packed_score_mode=off \
#     trainer.selector_temperature_schedule.enabled=false \
#     trainer.use_amp=true \
#     trainer.n_samples_per_input=10 \
#     trainer.max_num_val_batches=1 trainer.max_num_train_val_batches=0 \
#     wandb.enabled=true




# ***************** SP500 ARM B T=5: improved temporal, no downsampling ******************* #
# Same temporal + no-DS setup as ARM B T=20/T=10, adapted for T=5:
#   - dilations: [1,1] with kernel=3 gives per-layer RF = 3 and 3.
#     Two stacked layers with dilation=1 cover the full T=5 window with
#     dense adjacent-timestep coverage — no gaps.
# Runs: proud-frog-316 (crashed: bottleneck dilations mismatch), super-hoatzin-319 (killed), noisy-ant-108
# CUDA_VISIBLE_DEVICES=1 python -m graph_signal_diffusion.cli.train \
#     task=stock_price_forecasting_v3 \
#     dataset.root=${SP500_ROOT} \
#     dataset.past_window=20 \
#     dataset.future_window=5 \
#     dataset.batch_size=64 \
#     dataset.batch_size_val=200 \
#     dataset.num_workers=4 \
#     dataset.persistent_workers=true \
#     dataset.pin_memory=true \
#     model.config.gnn_config.temporal_mixer.schedule=per_layer \
#     model.config.gnn_config.temporal_mixer.kernel_size=3 \
#     'model.config.gnn_config.temporal_mixer.dilations=[1,1]' \
#     model.config.gnn_config.temporal_mixer.use_pointwise=true \
#     model.config.gnn_config.temporal_mixer.causal=false \
#     'model.config.channel_multipliers=[1,1,1,1]' \
#     'model.config.pooling_config.gamma=[1,1,1,1]' \
#     model.config.pooling_config.selection_method=stride \
#     model.config.num_bottleneck_layers=1 \
#     model.config.compact_skip_cache=false \
#     model.config.gnn_config.pointwise_sparse_mode=off \
#     model.config.pooling_config.selector_packed_score_mode=off \
#     trainer.selector_temperature_schedule.enabled=false \
#     trainer.use_amp=true \
#     trainer.n_samples_per_input=10 \
#     trainer.max_num_val_batches=1 trainer.max_num_train_val_batches=1 \
#     wandb.enabled=true


# # wooden-smilodon-684: Not an improvement over noisy-ant-108 (launcher defined above), so can be ignored for ARM C experiments.
# CUDA_VISIBLE_DEVICES=0 python -m graph_signal_diffusion.cli.train \
#     task=stock_price_forecasting_v3 \
#     dataset.root=${SP500_ROOT} \
#     dataset.past_window=20 \
#     dataset.future_window=5 \
#     dataset.spectral_normalize_edge_weights=false \
#     dataset.batch_size=64 \
#     dataset.batch_size_val=200 \
#     dataset.num_workers=4 \
#     dataset.persistent_workers=true \
#     dataset.pin_memory=true \
#     model.config.gnn_config.temporal_mixer.schedule=per_layer \
#     model.config.gnn_config.temporal_mixer.kernel_size=3 \
#     'model.config.gnn_config.temporal_mixer.dilations=[1,1]' \
#     model.config.gnn_config.temporal_mixer.use_pointwise=true \
#     model.config.gnn_config.temporal_mixer.causal=false \
#     'model.config.channel_multipliers=[1,1,1]' \
#     'model.config.pooling_config.gamma=[1,1,1]' \
#     model.config.pooling_config.selection_method=stride \
#     model.config.num_bottleneck_layers=1 \
#     model.config.compact_skip_cache=false \
#     model.config.gnn_config.pointwise_sparse_mode=off \
#     model.config.pooling_config.selector_packed_score_mode=off \
#     trainer.selector_temperature_schedule.enabled=false \
#     trainer.use_amp=true \
#     trainer.n_samples_per_input=10 \
#     trainer.optimizer.learning_rate=1e-5 trainer.optimizer.weight_decay=0.0 \
#     trainer.max_num_val_batches=1 trainer.max_num_train_val_batches=1 \
#     wandb.enabled=true


## ***************** SP500 ARM A T=20: per-block temporal mixer baseline ****************** #
# 3-level [1,2,2] ARM K architecture with future_window=20.
# Per-block temporal mixer: kernel=5, dilation=5 (RF=21, covers T=20).
#   - Sparse sampling: sees every 5th timestep, misses lag-1 autocorrelation.
#   - Pointwise off: no cross-channel temporal interaction.
# Runs: vigorous-hound-219
# CUDA_VISIBLE_DEVICES=0 python -m graph_signal_diffusion.cli.train \
#     task=stock_price_forecasting_v3 \
#     dataset.root=${SP500_ROOT} \
#     dataset.past_window=20 \
#     dataset.future_window=20 \
#     dataset.batch_size=64 \
#     dataset.batch_size_val=200 \
#     dataset.num_workers=4 \
#     dataset.persistent_workers=true \
#     dataset.pin_memory=true \
#     model.config.gnn_config.temporal_mixer.kernel_size=5 \
#     'model.config.gnn_config.temporal_mixer.dilations=[5]' \
#     trainer.use_amp=true \
#     trainer.n_samples_per_input=10 \
#     trainer.max_num_val_batches=1 trainer.max_num_train_val_batches=0 \
#     wandb.enabled=true


## ***************** SP500 ARM A T=10: per-block temporal mixer baseline ****************** #
# Same architecture as ARM A T=20 but with future_window=10.
# Per-block temporal mixer: kernel=5, dilation=3 (RF=13, covers T=10).
# Runs: inescapable-toad-245
# CUDA_VISIBLE_DEVICES=1 python -m graph_signal_diffusion.cli.train \
#     task=stock_price_forecasting_v3 \
#     dataset.root=${SP500_ROOT} \
#     dataset.past_window=20 \
#     dataset.future_window=10 \
#     dataset.batch_size=64 \
#     dataset.batch_size_val=200 \
#     dataset.num_workers=4 \
#     dataset.persistent_workers=false \
#     dataset.pin_memory=true \
#     model.config.gnn_config.temporal_mixer.kernel_size=5 \
#     'model.config.gnn_config.temporal_mixer.dilations=[3]' \
#     trainer.use_amp=true \
#     trainer.n_samples_per_input=10 \
#     trainer.max_num_val_batches=1 trainer.max_num_train_val_batches=0 \
#     wandb.enabled=true


## ***************** SP500 ARM K: 3-level [1,2,2] + 3-layer bottleneck + linear anneal *** #
# Direct port of WRA ARM K (neat-chupacabra-681) to SP500-cleaned.
# Architecture is functionally identical modulo temporal processing:
#   - gamma=[1,2,2] with channel_multipliers=[1,1,1]: 3 encoder levels.
#     Level 0 (gamma=1): GNN@468, Level 1 (gamma=2): GNN@468+pool->234,
#     Level 2 (gamma=2): GNN@234+pool->117.
#   - num_bottleneck_layers=3: heavy processing at 117 nodes for global
#     cross-stock reasoning (matching image U-Net bottleneck principle).
#   - Linear temperature schedule with anneal_ratio=0.90: T_min=0.20
#     reached at epoch ~900 (of 1000), ~100 epochs at the floor.
#   - No entropy reg (diffusion loss alone suffices with slow linear anneal).
#   - Per-block temporal mixer: kernel=3, dilation=2, causal.
#     RF = 1 + (3-1)*2 = 5 = T_out. Covers full forecast window.
#   - Time-varying conditioning: T_cond=20 -> T_out=5 learned projection.
#   - Conditioning fusion in selector: 'add' mode (same as WRA ARM K).
# CUDA_VISIBLE_DEVICES=1 python -m graph_signal_diffusion.cli.train \
#     task=stock_price_forecasting_v3 \
#     dataset.root=${SP500_ROOT} \
#     dataset.past_window=20 \
#     dataset.future_window=5 \
#     dataset.batch_size=64 \
#     dataset.batch_size_val=50 \
#     dataset.num_workers=8 \
#     dataset.persistent_workers=true \
#     dataset.pin_memory=true \
#     trainer.use_amp=true \
#     wandb.enabled=true


## ***************** SP500 no-DS baseline: no downsampling, same temporal setup ********** #
# Baseline comparison: identical temporal processing (v3 temporal mixer +
# time-varying conditioning) but no learned downsampling.
# Analogous to WRA "monumental-mushroom-152" (no downsampling baseline).
# CUDA_VISIBLE_DEVICES=0 python -m graph_signal_diffusion.cli.train \
#     task=stock_price_forecasting_v3 \
#     dataset.root=${SP500_ROOT} \
#     dataset.past_window=20 \
#     dataset.future_window=5 \
#     dataset.batch_size=64 \
#     dataset.batch_size_val=50 \
#     dataset.num_workers=8 \
#     dataset.persistent_workers=true \
#     dataset.pin_memory=true \
#     'model.config.channel_multipliers=[1,1,1,1]' \
#     'model.config.pooling_config.gamma=[1,1,1,1]' \
#     model.config.pooling_config.selection_method=stride \
#     model.config.num_bottleneck_layers=1 \
#     model.config.compact_skip_cache=false \
#     model.config.gnn_config.pointwise_sparse_mode=off \
#     model.config.pooling_config.selector_packed_score_mode=off \
#     trainer.selector_temperature_schedule.enabled=false \
#     trainer.use_amp=true \
#     wandb.enabled=true


## ***************** SP500 ARM K + cosine anneal: test linear vs cosine ****************** #
# Same architecture as SP500 ARM K but with cosine temperature schedule
# (anneal_ratio=0.60) to test whether the linear anneal advantage transfers
# from WRA to SP500. Direct analogue of WRA ARM H (able-jaybird-407).
# CUDA_VISIBLE_DEVICES=1 python -m graph_signal_diffusion.cli.train \
#     task=stock_price_forecasting_v3 \
#     dataset.root=${SP500_ROOT} \
#     dataset.past_window=20 \
#     dataset.future_window=5 \
#     dataset.batch_size=64 \
#     dataset.batch_size_val=50 \
#     dataset.num_workers=8 \
#     dataset.persistent_workers=true \
#     dataset.pin_memory=true \
#     model.config.pooling_config.selector_kwargs.temperature_schedule=cosine \
#     trainer.selector_temperature_schedule.anneal_ratio=0.60 \
#     trainer.use_amp=true \
#     wandb.enabled=true


## ***************** SP500 ARM K + 4-level [1,2,2,2]: deeper downsampling **************** #
# Tests whether the extra downsampling level helps on 468-node graphs.
# 468 -> 234 -> 117 -> 59 bottleneck (59 nodes, ~12.5% of graph).
# Analogous to WRA ARM L (4-level variant).
# CUDA_VISIBLE_DEVICES=1 python -m graph_signal_diffusion.cli.train \
#     task=stock_price_forecasting_v3 \
#     dataset.root=${SP500_ROOT} \
#     dataset.past_window=20 \
#     dataset.future_window=5 \
#     dataset.batch_size=64 \
#     dataset.batch_size_val=50 \
#     dataset.num_workers=8 \
#     dataset.persistent_workers=true \
#     dataset.pin_memory=true \
#     'model.config.channel_multipliers=[1,1,1,1]' \
#     'model.config.pooling_config.gamma=[1,2,2,2]' \
#     trainer.use_amp=true \
#     wandb.enabled=true


