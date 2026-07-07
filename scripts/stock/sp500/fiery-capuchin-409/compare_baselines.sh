#!/bin/bash
# Compare GRW vs ARM G (fiery-capuchin-409) on the SP500-cleaned test split.
#
# ARM G config: past_window=20, future_window=10 (T=10), 3-chunk interleaved
# split, diamond [1,2,1] denoiser (base_channels=64 → channels [64,128,64]),
# DDIM (linear schedule, η=0.2, 500 train timesteps / 100 sampling timesteps),
# RevIN, cross-attention conditioning, dropout=0.01, corr_0.7 graph, bs=32.
#
# Uses checkpoint at epoch 725 — the best composite (CRPS+MAE) checkpoint.
# Current top-5 checkpoints saved by the trainer:
#   ep  725 — rank 1 (1.2267)                         ← default
#   ep  650 — rank 2 (1.2333)
#   ep  850 — rank 3 (1.2348)
#   ep 1050 — rank 4 (1.2361)
#   ep  525 — rank 5 (1.2376)
#
# Eval setup: 20 samples per window (vs 10 in training), full DDIM sampling
# (100 steps, η=0.2 from training), AMP enabled.

export HYDRA_FULL_ERROR=1

experiment_name=fiery-capuchin-409
experiment_dir=outputs/stock_price_forecasting_v3-sp500_cleaned/ugnn_sp500_v3_ds4-ddim-gdm_sp500_v3_learned_ds4/$experiment_name
checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_650.pt
# checkpoint_path=$experiment_dir/trainer_chkpts/DDIM_epoch_850.pt

batch_size=100
batch_size_val=200
n_samples=10 # 50 # 50 for full eval; 20 for quicker evaluation.
test_subsample_n=100

# DDIM stochasticity knob. Training used η=0.2 (nearly deterministic).
# Bumping to 1.0 at inference widens the ensemble (DDPM-equivalent limit),
# trading sharpness for coverage — useful when the model is undercovered
# (ARM G: pCov90≈0.65 vs nominal 0.90, spread ~55% of GRW's).
ddim_eta=1.0
sampling_timesteps=250 # 100 # keep at training default for fair comparison

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
    ++baselines.diffusion.diffusion_overrides.ddim_eta=$ddim_eta \
    ++baselines.diffusion.diffusion_overrides.sampling_timesteps=$sampling_timesteps \
    ++baselines.diffusion.evaluation.elbo_n_mc_samples=500 \
    baselines.grw.n_samples=$n_samples \
    dataset.root=data/sp500/cleaned_drop_incomplete_min_coverage_0.95_corr_0.7_sector_bonus_0.05 \
    dataset.batch_size=$batch_size \
    dataset.batch_size_val=$batch_size_val \
    dataset.n_samples_per_input=1 \
    dataset.past_window=20 \
    dataset.future_window=10 \
    dataset.n_split_chunks=3 \
    dataset.standardize_target_in_x_for_revin=true \
    dataset.test_subsample_n=$test_subsample_n \
    'eval_splits={test: null}' \
    wandb.enabled=False

# Notes:
# - dataset.root points to the corr_0.7 graph variant (same as ARM F-v2/G-v2).
# - dataset.future_window=10 (T=10) must match training; GRW will also forecast
#   10 steps ahead for a fair comparison.
# - dataset.n_split_chunks=3 must match the training-time split.
# - dataset.standardize_target_in_x_for_revin=true must match training (RevIN).
# - n_samples_per_input=1 because compare_baselines rebuilds the loader per-baseline.
# - dataset.test_subsample_n=50 picks 50 windows uniformly along test_idx via
#   np.linspace (deterministic, covers all 3 chunks).
# - eval_splits={test: null} processes all (subsampled) test batches.
# - diffusion_overrides.ddim_eta=1.0 overrides the training-time η=0.2 to inject
#   full DDPM stochasticity at inference (wider ensemble → higher coverage).
#   Sampling steps stay at the training default (100).
# - CUDA_VISIBLE_DEVICES=0 so it can run in parallel with ARM G-v2 on GPU 1.
# - Estimated runtime at n_samples=20 + 100 DDIM steps (T=10 doubles per-sample work vs T=5):
#     test_subsample_n=50  → ~1h 20min
#     test_subsample_n=100 → ~2h 40min
#     test_subsample_n=null (full ~234)  → ~6-8h
