#!/bin/bash
# Compare GRW vs outstanding-fox-517 on the SP500-cleaned test split.
#
# outstanding-fox-517 config:
#   past_window=20, future_window=5 (T=5), 3-chunk interleaved split,
#   fixed-width 4-level U-GNN denoiser at base_channels=64 with
#   channel_multipliers=[1,1,1,1] (channels [64,64,64,64], gamma=[1,1,1,1]),
#   DDIM (linear schedule, η=0.2, 500 train timesteps / 100 sampling
#   timesteps), RevIN, cross-attention conditioning, corr_0.7 graph, bs=48.
#
# Differences vs gray-galago-398 (control):
#   1. base_channels=64 (vs 32) with channel_multipliers=[1,1,1,1] (vs [4,2,1])
#      → uniform 64-channel width across 4 graph-coarsened levels, instead of
#      gray-galago's 3-level inverted pyramid [128,64,32]. Distributes capacity
#      evenly across spatial scales rather than concentrating it at the top.
#   2. pooling_config.gamma=[1,1,1,1] (one more pooling step than gray-galago).
#   3. batch_size=48 (vs 64) to keep training under the AMP memory ceiling.
#
# Tracker, dropout (0.10), DDIM η, schedule, RevIN, T=5 horizon, save cadence
# all identical to gray-galago-398.
#
# Top-5 best-model checkpoints saved by the trainer (final, ep 2499 / 2500):
#   ep 1950 — rank 1  composite 0.2901  val_loss=0.3350 (run min)  |gap|=0.0029  CRPS=1.098  cov90=0.685 (top-5 max)  eig1_ratio=0.671
#   ep 1450 — rank 2  composite 0.2946  val_loss=0.3382             |gap|=0.0058  CRPS=1.112  cov90=0.662  eig1_ratio=0.682 (top-5 max)
#   ep 1650 — rank 3  composite 0.2956  val_loss=0.3390             |gap|=0.0089  CRPS=1.107  cov90=0.671  eig1_ratio=0.604
#   ep 2250 — rank 4  composite 0.2971  val_loss=0.3372             |gap|=0.0117  CRPS=1.115  cov90=0.668  eig1_ratio=0.635
#   ep  350 — rank 5  composite 0.2980  val_loss=0.3453             |gap|=0.0045  CRPS=1.118  cov90=0.672  eig1_ratio=0.503 (early)
#
# Notable: ep 1950 is the unique α-sweep rank-1 across α ∈ [0, 1.0] — it's the
# global minimum of val_loss AND a near-zero gap AND the lowest CRPS in the
# top-5 AND the highest Cov@90 in the top-5. This is the cleanest training-time
# signal seen since amethyst-perch-521 ep 950.
#
# Eval setup: 20 samples per window (vs 10 in training), full DDIM
# sampling (100 steps, η=0.2 from training), AMP enabled.

export HYDRA_FULL_ERROR=1

experiment_name=outstanding-fox-517
experiment_dir=outputs/stock_price_forecasting_v3-sp500_cleaned/ugnn_sp500_v3_ds4-ddim-gdm_sp500_v3_learned_ds4/$experiment_name
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_1950.pt  # rank-1: dominant on every val-time metric
# Alternate operating points:
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_1450.pt  # rank-2, highest eig1_ratio in top-5 (factor structure preserved)
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_1650.pt  # rank-3, mid-band
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_2250.pt  # rank-4, late-stage operating point
checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_350.pt   # rank-5, early checkpoint (low eig1_ratio)


batch_size=100
batch_size_val=250 # 100 by default 
n_samples=50 # 50 for full eval; 20 for quicker evaluation.
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
    task=stock_price_forecasting_v3 \
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
#   amethyst-perch-521 / steel-goshawk-355 / ARM G / G-v2).
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
# - CUDA_VISIBLE_DEVICES=1 — change to 0 if running in parallel with another arm.
# - Estimated runtime at n_samples=20 + 100 DDIM steps (T=5) on the 4-level
#   fixed-width 64ch architecture is similar to gray-galago (slightly more
#   forward-pass cost due to the extra pooling level):
#     test_subsample_n=50  → ~45min
#     test_subsample_n=100 → ~1h 30min
#     test_subsample_n=null (full ~234)  → ~3.5-4.5h
