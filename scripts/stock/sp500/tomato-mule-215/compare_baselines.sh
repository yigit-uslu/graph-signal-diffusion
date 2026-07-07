#!/bin/bash
# Compare GRW vs tomato-mule-215 on the SP500-cleaned test split.
#
# tomato-mule-215 config (DS8, gamma=[1,2,2,2]):
#   past_window=20, future_window=5, 3-chunk interleaved split,
#   4-level U-GNN denoiser at base_channels=64, channel_multipliers=[1,1,1,1]
#   (uniform 64ch across all 4 levels), gamma=[1,2,2,2] (8x compression at
#   bottleneck = 58 nodes), max_gnn_stride=2, num_bottleneck_layers=2,
#   learned NodeSelector (STE, T_min=0.5, linear anneal anneal_ratio=0.90),
#   dropout=0.10, RevIN, DDIM (η=0.2, 500 train / 100 sampling timesteps),
#   per-layer temporal mixer + self-attention, cross-attention cond fusion,
#   corr_0.7 graph, bs=48. Trained with task=stock_price_forecasting_v3_learned_ds8.
#
# Top-5 best-model checkpoints (FINAL, ep 2499 / 2500):
#   ep  850 — rank 1  composite 0.2929  val_loss=0.3419  |gap|=0.0018  CRPS=1.0997  cov90=0.672  cov99=0.731  eig1_r=0.596
#   ep  600 — rank 2  composite 0.2952  val_loss=0.3467  |gap|=0.0005  CRPS=1.1055  cov90=0.623  cov99=0.684  eig1_r=0.659
#   ep  550 — rank 3  composite 0.2954  val_loss=0.3479  |gap|=0.0000 (run min)  CRPS=1.1053  cov90=0.650  cov99=0.710  eig1_r=0.670
#   ep 1450 — rank 4  composite 0.2956  val_loss=0.3483  |gap|=0.0002  CRPS=1.1048  cov90=0.682 (top-5 max)  cov99=0.738  eig1_r=0.651
#   ep 2499 — rank 5  composite 0.2960  val_loss=0.3484  |gap|=0.0008  CRPS=1.1050  cov90=0.676  cov99=0.735  eig1_r=0.678 (top-5 max)
#
# Notable: ep 850 is the UNIQUE α-sweep rank-1 across α ∈ [0, 1.0] in the
# tracker_composite_sweep analysis — same dominance pattern as outstanding-fox
# ep 1950. ep 2499 (late post-anneal) has the highest eig1_ratio in top-5, so
# it's the strongest factor-capture checkpoint and worth a test-time read for
# the "post-anneal hard-selection regime" vs ep 850's "high-T soft regime"
# question.
#
# Eval setup: 20 samples per window (vs 10 in training), full DDIM
# sampling (100 steps, η=0.2 from training), AMP enabled.

export HYDRA_FULL_ERROR=1

experiment_name=tomato-mule-215
experiment_dir=outputs/stock_price_forecasting_v3_learned_ds8-sp500_cleaned/ugnn_sp500_v3_ds8-ddim-gdm_sp500_v3_learned_ds8/$experiment_name
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_850.pt   # rank-1: unique α-sweep winner
# Alternate operating points:
checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_2499.pt  # rank-5, late post-anneal, highest eig1_ratio in top-5
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_600.pt   # rank-2
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_550.pt   # rank-3, zero |gap|
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_1450.pt  # rank-4, highest cov90 in top-5 — test-time CRPS / loss / eig1 winner of the 4-ckpt 2026-05-20 sweep


batch_size=100
batch_size_val=200 # 100
n_samples=100 # 50 for full eval; 20 for quicker evaluation.
test_subsample_n=50

# Inference-time DDIM stochasticity. Training used η=0.2 (nearly
# deterministic). Earlier sweep on amethyst showed η=1.0 actually narrows
# spread on this model family — keep at 0.2 unless deliberately probing.
ddim_eta=0.2
sampling_timesteps=100

# ── Post-hoc variance correction (eval-side) ─────────────────────────
# Rescales generated samples in standardized space, BEFORE batch_hook fires
# (so both task-level metrics AND structural diagnostics see corrected
# samples). Default mode=off is a bitwise no-op; set mode=scalar to activate.
# See docs/agent_summaries/variance_correction_plan_2026-05-20.md and
# docs/agent_summaries/variance_correction_implementation_2026-05-20.md.
#
# pivot:
#   ensemble_mean — x' = μ_t + α(x - μ_t). Preserves per-horizon ensemble
#                   mean (point-forecast invariant per (b, t, s)), but
#                   structural metrics drift modestly because the t-varying
#                   intercept μ_t breaks γ_0-normalization invariance.
#                   Right α: σ_real_marginal / σ_gen_ensemble ≈ 2.01/1.40 ≈ 1.44.
#                   We used α = 1.5 (slight over-correction).
#   zero          — x' = α·x in standardized space. STRICTLY scale-invariant
#                   for structural metrics (autocorr, eigval1_ratio, etc.).
#                   Per-horizon point forecast shifts by (α-1)·μ̂, small for
#                   daily log returns. Justified when under-spread is
#                   systematic across stocks/windows (per-stock analysis
#                   2026-05-20: 97.4% of 468 stocks under-spread).
#                   Right α: σ_real_marginal / σ_gen_marginal ≈ 1/0.783 ≈ 1.28.
#                   This matches pooled marginal std exactly. α=1.5 here would
#                   over-amplify Var(μ̂) by 1.17x.
variance_correction_mode=scalar         # off | scalar | per_horizon
variance_correction_alpha=1.28          # pivot=zero: 1.28 (marginal match); pivot=ensemble_mean: 1.5 (coverage match)
variance_correction_pivot=ensemble_mean          # ensemble_mean | zero

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
    ++baselines.diffusion.variance_correction.mode=$variance_correction_mode \
    ++baselines.diffusion.variance_correction.alpha=$variance_correction_alpha \
    ++baselines.diffusion.variance_correction.pivot=$variance_correction_pivot \
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
#   outstanding-fox-517 / infrared-coyote-480).
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
# - CUDA_VISIBLE_DEVICES=0 — change to 1 if running in parallel with another arm.
# - GPU runtime estimate at n_samples=20 + 100 DDIM steps (T=5) + L=4 levels
#   (similar profile to outstanding-fox):
#     test_subsample_n=50  → ~45min
#     test_subsample_n=100 → ~1h 30min
#     test_subsample_n=null (full ~234)  → ~3.5-4.5h
# - Training is COMPLETE (ep 2499 / 2500). Selector reached T=0.50 floor as
#   designed. Top-5 leaderboard is final.
