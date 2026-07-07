#!/bin/bash
# Compare GRW vs ARM F (pragmatic-trout-751) on the SP500-cleaned test split.
#
# ARM F config: past_window=20, future_window=5, 3-chunk interleaved split,
# lightweight [1,2,1] denoiser (1.75M params / 1.65M trainable), DDIM (linear
# schedule, η=0.2, 500 train timesteps / 100 sampling timesteps), RevIN,
# cross-attention conditioning, dropout=0.01, corr_0.6 graph.
#
# Uses checkpoint at epoch 1500 — the best composite (CRPS+MAE) checkpoint.
# Other top-5 checkpoints saved by the trainer:
#   ep 1500 — best composite score (1.261)          ← default
#   ep 3250 — rank 2 (1.271)
#   ep  200 — rank 3 (1.271)
#   ep 1000 — rank 4 (1.272)
#   ep 4250 — rank 5 (1.278)
#
# Eval setup: 20 samples per window (vs 10 in training), full DDIM sampling
# (100 steps, η=0.2 from training), AMP enabled.

export HYDRA_FULL_ERROR=1

experiment_name=pragmatic-trout-751
experiment_dir=outputs/stock_price_forecasting_v3-sp500_cleaned/ugnn_sp500_v3_ds4-ddim-gdm_sp500_v3_learned_ds4/$experiment_name
checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_1500.pt

batch_size=100
batch_size_val=100
n_samples=20 # 50 for full eval; 20 for quicker evaluation.
test_subsample_n=50

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
    baselines.grw.n_samples=$n_samples \
    dataset.root=data/sp500/cleaned_drop_incomplete_min_coverage_0.95_corr_0.6_sector_bonus_0.05 \
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
# - dataset.root points to the corr_0.6 graph variant (same as stoic-polecat-491).
# - dataset.n_split_chunks=3 must match the training-time split.
# - dataset.standardize_target_in_x_for_revin=true must match training (RevIN).
# - n_samples_per_input=1 because compare_baselines rebuilds the loader per-baseline.
# - dataset.test_subsample_n=50 picks 50 windows uniformly along test_idx via
#   np.linspace (deterministic, covers all 3 chunks).
# - eval_splits={test: null} processes all (subsampled) test batches.
# - No diffusion_overrides needed — default η=0.2, 100 sampling steps from training.
# - CUDA_VISIBLE_DEVICES=1 so it can run in parallel with ARM F-v2 on GPU 0.
# - Estimated runtime at n_samples=20 + 100 DDIM steps:
#     test_subsample_n=50  → ~40min
#     test_subsample_n=100 → ~1h 20min
#     test_subsample_n=null (full ~234)  → ~3-4h
