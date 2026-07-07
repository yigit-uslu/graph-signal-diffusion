#!/bin/bash
# Compare GRW vs gray-galago-398 on the SP500-cleaned test split.
#
# gray-galago-398 config:
#   past_window=20, future_window=5 (T=5), 3-chunk interleaved split,
#   inverted-pyramid [4,2,1] denoiser at base_channels=32 → channels
#   [128,64,32], DDIM (linear schedule, η=0.2, 500 train timesteps /
#   100 sampling timesteps), RevIN, cross-attention conditioning,
#   corr_0.7 graph, bs=64.
#
# Differences vs amethyst-perch-521 (control):
#   1. dropout=0.10 (10× higher than amethyst's 0.01) at all three sites
#      (GNN, temporal-mixer attn, cross-attn cond fusion).
#   2. Best-model tracker = val_loss + |val−train-val gap| + 0.3·val_return_crps
#      (vs amethyst's CRPS+MAE default). Uses the val_loss_gap derived metric
#      committed 2026-05-14.
#   3. save_checkpoint_every_n_epochs=50 (vs amethyst's 100), so the tracker's
#      picks land on saved DDIM checkpoints reliably.
#
# Top-5 best-model checkpoints saved by the trainer (as of ep 1547 / 2500):
#   ep 675 — rank 1  composite 0.2969  val_loss=0.3425 (run min)  CRPS=1.112  cov90=0.640   ← default
#   ep 450 — rank 2  composite 0.2971  val_loss=0.3456            CRPS=1.109  cov90=0.734
#   ep 475 — rank 3  composite 0.2974  val_loss=0.3560  |gap|=0.0009 (run min)  CRPS=1.091 (run min)  cov90=0.747
#   ep 600 — rank 4  composite 0.2974  val_loss=0.3507            CRPS=1.103  cov90=0.674
#   ep 625 — rank 5  composite 0.2976  val_loss=0.3492            CRPS=1.099  cov90=0.620
#
# Notable: ep 475 has BOTH the smallest |gap| AND the lowest CRPS in the run,
# with the highest cov90 of the top-5 — a strong alternate operating point if
# coverage is the priority.
#
# Eval setup: 20 samples per window (vs 10 in training), full DDIM
# sampling (100 steps, η=0.2 from training), AMP enabled.

export HYDRA_FULL_ERROR=1

experiment_name=gray-galago-398
experiment_dir=outputs/stock_price_forecasting_v3-sp500_cleaned/ugnn_sp500_v3_ds4-ddim-gdm_sp500_v3_learned_ds4/$experiment_name
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_675.pt
# Alternate operating points:
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_475.pt  # smallest |gap|, best CRPS, highest cov90
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_450.pt  # second-highest cov90
checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_2450.pt  # late-stage plateau
# checkpoint_path=$experiment_dir/trainer_chkpts/DDIM_epoch_750.pt # calibration extremum in the mid-late band
# checkpoint_path=$experiment_dir/trainer_chkpts/DDIM_epoch_1250.pt # the strongest "balanced" mid-late candidate


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
# - dataset.root points to the corr_0.7 graph variant (same as amethyst-perch-521 /
#   steel-goshawk-355 / ARM G / G-v2).
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
# - CUDA_VISIBLE_DEVICES=0 — change to 1 if running in parallel with another arm.
# - Estimated runtime at n_samples=20 + 100 DDIM steps (T=5):
#     test_subsample_n=50  → ~40min
#     test_subsample_n=100 → ~1h 20min
#     test_subsample_n=null (full ~234)  → ~3-4h
