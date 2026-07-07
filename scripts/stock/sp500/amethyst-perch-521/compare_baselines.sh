#!/bin/bash
# Compare GRW vs amethyst-perch-521 on the SP500-cleaned test split.
#
# amethyst-perch-521 config:
#   past_window=20, future_window=5 (T=5), 3-chunk interleaved split,
#   inverted-pyramid [4,2,1] denoiser at base_channels=32 → channels
#   [128,64,32], DDIM (linear schedule, η=0.2, 500 train timesteps /
#   100 sampling timesteps), RevIN, cross-attention conditioning,
#   dropout=0.01, corr_0.7 graph, bs=64.
#
# Pair partner: steel-goshawk-355, identical config except ddim_eta=1.0
# (still running). Train-loss agreement between the two confirms the
# weights are effectively identical; only the validation sampler differs.
#
# Top-5 best-model checkpoints saved by the trainer:
#   ep  350 — rank 1 (1.2608)                              ← default
#   ep 1650 — rank 2 (1.2629)   ← late peak, higher coverage (0.637)
#   ep  375 — rank 3 (1.2632)
#   ep  725 — rank 4 (1.2660)
#   ep  700 — rank 5 (1.2692)
#
# Eval setup: 20 samples per window (vs 10 in training), full DDIM
# sampling (100 steps, η=0.2 from training), AMP enabled.

export HYDRA_FULL_ERROR=1

experiment_name=amethyst-perch-521
experiment_dir=outputs/stock_price_forecasting_v3-sp500_cleaned/ugnn_sp500_v3_ds4-ddim-gdm_sp500_v3_learned_ds4/$experiment_name
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_350.pt
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_725.pt
# Alternate late-peak checkpoint with higher coverage (cov90=0.637, spread=3.34):
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_1650.pt
checkpoint_path=$experiment_dir/trainer_chkpts/DDIM_epoch_2000.pt

batch_size=100
batch_size_val=100
n_samples=20 # 50 for full eval; 20 for quicker evaluation.
test_subsample_n=50

# Inference-time DDIM stochasticity. Training used η=0.2 (nearly
# deterministic). Raise to widen the ensemble (a/b test of sampler
# stochasticity at fixed weights).
ddim_eta=0.2
sampling_timesteps=100

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
# - dataset.root points to the corr_0.7 graph variant (same as ARM F-v2 / G / G-v2).
# - dataset.future_window=5 (T=5) matches training. GRW also forecasts 5
#   steps ahead for a fair comparison.
# - dataset.n_split_chunks=3 must match the training-time split.
# - dataset.standardize_target_in_x_for_revin=true must match training (RevIN).
# - n_samples_per_input=1 because compare_baselines rebuilds the loader per-baseline.
# - dataset.test_subsample_n=50 picks 50 windows uniformly along test_idx via
#   np.linspace (deterministic, covers all 3 chunks). Set to null for full
#   (~234 windows) at ~3-4× the runtime.
# - eval_splits={test: null} processes all (subsampled) test batches.
# - diffusion_overrides.ddim_eta is exposed as a knob; set to 1.0 to inject
#   full DDPM stochasticity at inference (wider ensemble). Earlier sweep
#   (steel-goshawk-355) showed η=1.0 narrows spread by ~2.5% on this model
#   rather than widening it — keep at 0.2 unless deliberately probing.
# - CUDA_VISIBLE_DEVICES=0 — steel-goshawk-355 still training on GPU 1.
# - Estimated runtime at n_samples=20 + 100 DDIM steps (T=5):
#     test_subsample_n=50  → ~40min
#     test_subsample_n=100 → ~1h 20min
#     test_subsample_n=null (full ~234)  → ~3-4h
