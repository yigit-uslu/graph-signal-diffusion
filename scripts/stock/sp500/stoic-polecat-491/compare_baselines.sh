#!/bin/bash
# Compare GRW vs ARM D-v2f (stoic-polecat-491) on the SP500-cleaned test split.
#
# D-v2f config: past_window=20, future_window=5, 3-chunk interleaved split,
# diamond [1,2,4,2,1] denoiser (6.03M params), DDIM (linear schedule, η=0.2,
# 500 train timesteps / 100 sampling timesteps), RevIN, cross-attention conditioning.
#
# Uses checkpoint at epoch 500 — the best return CRPS checkpoint with the
# tightest train-val/val gap (no significant overfitting yet).  Alternative
# checkpoints saved by the trainer:
#   ep 275  — early CRPS minimum
#   ep 475  — best return MAE / per-step direction
#   ep 500  — best return CRPS              ← default DONE
#   ep 850  — best cumulative-T direction MV (partially overfit per train-val gap)
#   ep 1250 — late checkpoint DONE
#
# Eval setup: 50 samples per window (vs 20 in training) for tighter ensemble
# statistics, full DDIM sampling (100 steps, η=0.2 from training), AMP enabled.

export HYDRA_FULL_ERROR=1

experiment_name=stoic-polecat-491
experiment_dir=outputs/stock_price_forecasting_v3-sp500_cleaned/ugnn_sp500_v3_ds4-ddim-gdm_sp500_v3_learned_ds4/$experiment_name
checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_500.pt
# checkpoint_path=$experiment_dir/trainer_chkpts/DDIM_epoch_2000.pt

batch_size=100
batch_size_val=100
n_samples=50 # smoke run with 5 samples/window for quick testing; increase to 50 for full eval (see notes below)
test_subsample_n=10 # uniform-stride linspace subsample of the test split (deterministic, covers all 3 chunks). null = full split.

# Output dir override: comparison/<date>/<time>_<experiment>_epoch-<epoch>.
# Epoch is parsed from the active checkpoint filename (NA for the best_model.pt
# symlink or other names without an epoch_<N> token).
epoch_label=$(basename "$checkpoint_path" | grep -oE 'epoch_[0-9]+' | grep -oE '[0-9]+' | head -1)
run_dir="$(dirname "$(dirname "$experiment_dir")")/comparison/$(date +%Y-%m-%d/%H-%M-%S)_${experiment_name}_epoch-${epoch_label:-NA}"

CUDA_VISIBLE_DEVICES=0 python -m graph_signal_diffusion.cli.compare_baselines \
    --config-name compare_baselines_sp500 \
    hydra.run.dir="$run_dir" \
    task=stock_price_forecasting_v3 \
    baselines_to_compare='[grw,diffusion]' \
    baselines.diffusion.checkpoint_path=$checkpoint_path \
    baselines.diffusion.n_samples=$n_samples \
    ++baselines.diffusion.use_amp=true \
    '++baselines.diffusion.diffusion_overrides={ddim_eta: 1.0, sampling_timesteps: 500}' \
    baselines.grw.n_samples=$n_samples \
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
# - task=stock_price_forecasting_v3 overrides the v2 default in
#   compare_baselines_sp500.yaml so that v3-only metrics (cumulative-T
#   direction accuracy, majority-vote direction, etc.) are reported.
# - dataset.n_split_chunks=3 must match the training-time split, otherwise
#   the test windows will not correspond to D-v2f's actual held-out set.
# - dataset.standardize_target_in_x_for_revin=true must also match training;
#   D-v2f's RevIN expects pre-standardized inputs.
# - n_samples_per_input=1 here because compare_baselines rebuilds the loader
#   per-baseline using baselines.<name>.n_samples for the ensemble.
# - dataset.test_subsample_n=50 picks 50 windows uniformly along test_idx via
#   np.linspace (deterministic, includes first & last test windows, ~17 from
#   each of the 3 chronological chunks).  Set to null for the full ~234-window
#   test split.
# - eval_splits={test: null} now means "process all (subsampled) test batches".
# - Estimated runtime at n_samples=50 + 100 DDIM steps:
#     test_subsample_n=50  → ~1h 40min   (50 unique windows; current default)
#     test_subsample_n=100 → ~3h 20min
#     test_subsample_n=null (full ~234)  → ~6-9h
