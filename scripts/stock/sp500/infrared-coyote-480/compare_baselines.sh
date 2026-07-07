#!/bin/bash
# Compare GRW vs infrared-coyote-480 on the SP500-cleaned test split.
#
# infrared-coyote-480 config (DS8-pyramid, gamma=[2,2,2]):
#   past_window=20, future_window=5, 3-chunk interleaved split,
#   3-level U-GNN denoiser at base_channels=32, channel_multipliers=[1,2,4]
#   (channels [32,64,128] doubling at each pool — DDPM/Stable Diffusion-style
#   pyramid), gamma=[2,2,2] (canonical U-Net pool-at-every-transition,
#   8x compression at bottleneck = 58 nodes), max_gnn_stride=2,
#   num_bottleneck_layers=2, learned NodeSelector (STE, T_min=0.5, linear
#   anneal anneal_ratio=0.90), dropout=0.10, RevIN, DDIM (η=0.2, 500 train /
#   100 sampling timesteps), per-layer temporal mixer + self-attention,
#   cross-attention cond fusion, corr_0.7 graph, bs=48. Trained with
#   task=stock_price_forecasting_v3_learned_ds8 + the three model-level
#   overrides (base_channels=32, multipliers=[1,2,4], gamma=[2,2,2]).
#
# Top-5 best-model checkpoints (FINAL, ep 2499 / 2500):
#   ep  525 — rank 1  composite 0.2974  val_loss=0.3496  |gap|=0.0002  CRPS=1.1137  cov90=0.689  cov99=0.746  eig1_r=0.557
#   ep  200 — rank 2  composite 0.2979  val_loss=0.3552  |gap|=0.0027  CRPS=1.0908 (top-5 min)  cov90=0.666  cov99=0.725  eig1_r=0.516
#   ep  375 — rank 3  composite 0.2986  val_loss=0.3505  |gap|=0.0003  CRPS=1.1198  cov90=0.706 (top-5 max)  cov99=0.763  eig1_r=0.390
#   ep  325 — rank 4  composite 0.2999  val_loss=0.3530  |gap|=0.0096  CRPS=1.0906  cov90=0.662  cov99=0.723  eig1_r=0.383
#   ep 2450 — rank 5  composite 0.3006  val_loss=0.3496  |gap|=0.0082  CRPS=1.1119  cov90=0.683  cov99=0.739  eig1_r=0.529 (late post-anneal)
#
# Notable: ep 525 wins the α-sweep at α ∈ [0, 0.3] (the SP500 tracker setting).
# ep 200 takes over at α ≥ 0.5 — it has the lowest CRPS in top-5 by a clear
# margin (1.0908 vs 1.1137). So if test-time CRPS is the priority, ep 200 is
# the alternate operating point. The eig1_ratio range (0.39-0.56) remains
# markedly lower than tomato-mule's (0.60-0.68) — confirms the pyramid
# architecture's conservative factor capture observed throughout training.
#
# Eval setup: 20 samples per window, full DDIM sampling (100 steps,
# η=0.2 from training), AMP enabled.

export HYDRA_FULL_ERROR=1

experiment_name=infrared-coyote-480
experiment_dir=outputs/stock_price_forecasting_v3_learned_ds8-sp500_cleaned/ugnn_sp500_v3_ds8-ddim-gdm_sp500_v3_learned_ds8/$experiment_name
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_525.pt   # rank-1: α=0.3 tracker winner, highest eig1_ratio in top-5
# Alternate operating points:
checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_200.pt   # rank-2, lowest CRPS, α≥0.5 tracker winner
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_375.pt   # rank-3, highest cov90 in top-5
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_2450.pt  # rank-5, late post-anneal

# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_325.pt   # rank-4



batch_size=100
batch_size_val=100
n_samples=20 # 50 for full eval; 20 for quicker evaluation.
test_subsample_n=50

# Inference-time DDIM stochasticity. Training used η=0.2 (nearly
# deterministic). Earlier sweep on amethyst showed η=1.0 actually narrows
# spread on this model family — keep at 0.2 unless deliberately probing.
ddim_eta=0.2
sampling_timesteps=100

# Output dir override: comparison/<date>/<time>_<experiment>_epoch-<epoch>.
# Epoch is parsed from the active checkpoint filename (NA for the best_model.pt
# symlink or other names without an epoch_<N> token).
epoch_label=$(basename "$checkpoint_path" | grep -oE 'epoch_[0-9]+' | grep -oE '[0-9]+' | head -1)
run_dir="$(dirname "$(dirname "$experiment_dir")")/comparison/$(date +%Y-%m-%d/%H-%M-%S)_${experiment_name}_epoch-${epoch_label:-NA}"

CUDA_VISIBLE_DEVICES=1 python -m graph_signal_diffusion.cli.compare_baselines \
    --config-name compare_baselines_sp500 \
    hydra.run.dir="$run_dir" \
    task=stock_price_forecasting_v3_learned_ds8 \
    baselines_to_compare='[grw,diffusion]' \
    baselines.diffusion.checkpoint_path=$checkpoint_path \
    baselines.diffusion.n_samples=$n_samples \
    ++baselines.diffusion.use_amp=true \
    ++baselines.diffusion.diffusion_overrides.ddim_eta=$ddim_eta \
    ++baselines.diffusion.diffusion_overrides.sampling_timesteps=$sampling_timesteps \
    baselines.grw.n_samples=$n_samples \
    dataset.root=data/sp500/cleaned_drop_incomplete_min_coverage_0.95_corr_0.7_sector_bonus_0.05 \
    dataset.batch_size=$batch_size \
    dataset.batch_size_val=$batch_size_val \
    dataset.n_samples_per_input=1 \
    dataset.past_window=20 \
    dataset.future_window=5 \
    dataset.n_split_chunks=3 \
    dataset.standardize_target_in_x_for_revin=true \
    dataset.test_subsample_n=$test_subsample_n \
    'eval_splits={test: null}' \
    wandb.enabled=False

# Notes:
# - dataset.root points to the corr_0.7 graph variant (same as gray-galago-398 /
#   outstanding-fox-517 / tomato-mule-215).
# - dataset.future_window=5 (T=5) matches training. GRW also forecasts 5
#   steps ahead for a fair comparison.
# - dataset.n_split_chunks=3 must match the training-time split.
# - dataset.standardize_target_in_x_for_revin=true must match training (RevIN).
# - n_samples_per_input=1 because compare_baselines rebuilds the loader per-baseline.
# - dataset.test_subsample_n=50 picks 50 windows uniformly along test_idx via
#   np.linspace (deterministic, covers all 3 chunks). Set to null for full
#   (~234 windows) at ~3-4× the runtime.
# - For tail-coverage reads (cov_95 / cov_99 emitted by the 2026-05-14 evaluator
#   extension), bump n_samples to 50 (cov_95 stable) or 200 (cov_99 stable).
# - eval_splits={test: null} processes all (subsampled) test batches.
# - The diffusion baseline auto-loads model architecture from the experiment's
#   .hydra/config.yaml, so the task config we pass is mostly cosmetic — using
#   stock_price_forecasting_v3_learned_ds8 to match the trained task.
# - CUDA_VISIBLE_DEVICES=1 — change to 0 if running in parallel with another arm.
# - GPU runtime estimate at n_samples=20 + 100 DDIM steps (T=5) + L=3 levels +
#   bottleneck at 128ch (comparable to tomato-mule):
#     test_subsample_n=50  → ~45min
#     test_subsample_n=100 → ~1h 30min
# - Training is COMPLETE (ep 2499 / 2500). Selector reached T=0.50 floor as
#   designed. Top-5 leaderboard is final.
