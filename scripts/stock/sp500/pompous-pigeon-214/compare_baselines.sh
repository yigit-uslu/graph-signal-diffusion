#!/bin/bash
# Compare GRW vs pompous-pigeon-214 on the SP500-cleaned test split.
#
# pompous-pigeon-214 architecture (DS8-wide-uniform-128 + norm-act head, ~5.05M params):
#   Same backbone + cond-boost + Gumbel exploration as carmine-moose-32:
#   base_channels=128, gamma=[2,2,2], num_layers=3, num_bottleneck_layers=3,
#   learned NodeSelector STE T_min=0.5 linear anneal_ratio=0.90, dropout=0.10,
#   RevIN, DDIM eta=0.2, 500 train / 100 sampling timesteps, corr_0.7 graph,
#   bs=24, T=5, n_split_chunks=3.
#
#   cond-boost (same as carmine):
#     - cond.shared_encoder.temporal.mixer.gated=true
#     - cond.shared_encoder.temporal.mixer.attention.enabled=true
#       (num_heads=2, dropout=0.1, max_timesteps=20)
#     - selector_exploration_noise=1.0 linear, warmup 2%, anneal 90%
#
#   The ONLY delta vs carmine-moose-32: learning_rate=2e-5 (carmine: 5e-4;
#   wapiti: 1e-4). This is the "lr DOWN 5× from wapiti, 25× DOWN from
#   carmine" arm of the lr bracket.
#
#   RevIN bias-compensation (same mechanism as carmine/wapiti):
#     - revin_blend_weight=0.7
#     - revin_sigma_correction = 1.1150246904679009, materialized as a
#       literal in .hydra/config.yaml (auto-loaded at eval — NO CLI override).
#
# Task: stock_price_forecasting_v3_learned_ds8, n_split_chunks=3 (NATIVE,
# identical to F3 / wapiti / carmine). Pair ablations (all 3-chunk, directly
# comparable on THIS test split):
#   - colorful-wapiti-739 (lr=1e-4, no cond boost, no Gumbel) — REFERENCE,
#     composite 0.27915 @ ep 300
#   - carmine-moose-32    (lr=5e-4, same cond boost + Gumbel) — composite
#     0.2825 @ ep 200, overfit hard (frozen since ep 350)
#
# Training status as of 2026-05-29: FINISHED (1000/1000 epochs).
# Tracker leaderboard (composite = val_loss + |val_loss_gap| + 0.3·val_return_crps),
# top-5 of run:
#   ep 999 — rank 1  composite 0.2786  val_loss=0.3074  gap=+0.0100  val_crps=1.1084  cum_T_dir=0.4980
#   ep 600 — rank 2  composite 0.2791  val_loss=0.3117  gap=+0.0226  val_crps=1.0982  cum_T_dir=0.4943
#   ep 700 — rank 3  composite 0.2806  val_loss=0.3079  gap=+0.0060  val_crps=1.1169  cum_T_dir=0.4922
#   ep 200 — rank 4  composite 0.2807  val_loss=0.3090  gap=+0.0101  val_crps=1.1129  cum_T_dir=0.4853
#   ep 900 — rank 5  composite 0.2808  val_loss=0.3013  gap=+0.0088  val_crps=1.1111  cum_T_dir=0.4877
#
# Headline: pompous (lr 0.2× wapiti) caught up to and very slightly edged
# wapiti (composite 0.2786 vs 0.27915), confirming lr=2e-5 ≈ lr=1e-4 ≫
# lr=5e-4 for this architecture. The cond-boost + Gumbel stack is fine at
# a sane lr; carmine's 5e-4 was the mistake.
#
# cum_T_direction_accuracy hovers ~0.49 across the whole run — NO directional
# skill on the 5-day cumulative-return signal (same as F3 / wapiti). The
# cond-boost did not unlock directional skill at this lr.
#
# Checkpoint selection:
#
#   ep 999 (default) — tracker rank-1, best composite, best cum_T_crps
#     (2.6951) and best cum_T_dir (0.4980, still ≈ no-skill) of the run.
#     Terminal state of the cosine schedule (fully decayed lr).
#     best_models/best_model_epoch_999.pt
#
#   ep 600 — tracker rank-2; LOWEST val_return_crps of top-5 (1.0982),
#     second-best cum_T_crps (2.7124):
#     best_models/best_model_epoch_600.pt
#
#   ep 700 — tracker rank-3; SMALLEST val_loss_gap of top-5 (+0.0060):
#     best_models/best_model_epoch_700.pt
#
#   ep 900 — tracker rank-5; lowest val_loss of top-5 (0.3013):
#     best_models/best_model_epoch_900.pt
#
# Hypotheses to test on the (3-chunk) test split:
#   1. Test composite / val_return_crps land in the wapiti band (~0.30 /
#      ~1.10) — pompous and wapiti are statistically tied; the lr knob is
#      not the bottleneck once it's in a sane range.
#   2. cum_T_direction_accuracy ≈ 0.50 (no directional skill) — matches the
#      training-time signal; the cond-boost did not help here.
#   3. Spread ratio (gen/real), Cov@90 in the wapiti band (~0.6, ~0.65) —
#      same RevIN-fix mechanism.
#
# Eval setup: 20 samples per window (matching wapiti/carmine launchers),
# full DDIM sampling (100 steps, eta=0.2 matching training), AMP enabled,
# test_subsample_n=50 (deterministic linspace over all 3 chunks).

export HYDRA_FULL_ERROR=1

experiment_name=pompous-pigeon-214
experiment_dir=outputs/stock_price_forecasting_v3_learned_ds8-sp500_cleaned/ugnn_sp500_v3_ds8_norm_act_head-ddim-gdm_sp500_v3_learned_ds8/$experiment_name
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_999.pt

# Alternate operating points:
checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_600.pt   # rank 2; lowest val_return_crps in top-5
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_700.pt   # rank 3; smallest val_loss_gap in top-5
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_900.pt   # rank 5; lowest val_loss in top-5
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model.pt             # tracker rank-1 symlink (= ep 999)

batch_size=100
batch_size_val=200
n_samples=20 # 50 for full eval; 20 for quicker evaluation. Wapiti/carmine launchers use 20.
test_subsample_n=50

# Inference-time DDIM stochasticity. Training used eta=0.2.
ddim_eta=0.2
sampling_timesteps=100

# ── RevIN σ scale correction ─────────────────────────────────────────
# pompous's saved .hydra/config.yaml has revin_sigma_correction=1.1150246904679009
# materialized as a literal. The diffusion baseline auto-loads this — NO CLI
# override required for default eval.
#
# To probe sensitivity to α at eval time, uncomment one of:
# revin_sigma_correction_override=1.1150246904679009 # explicit trained value
# revin_sigma_correction_override=1.0                # disable correction entirely
# And add to the python call:
#   ++baselines.diffusion.diffusion_overrides.revin_sigma_correction=$revin_sigma_correction_override
# IMPORTANT: do NOT combine with the legacy task-level variance_correction below.

# Legacy task-level variance correction — KEEP OFF.
variance_correction_mode=off               # off | scalar | per_horizon
variance_correction_alpha=1.0              # ignored when mode=off
variance_correction_pivot=ensemble_mean    # ignored when mode=off

# Output dir override: comparison/<date>/<time>_<experiment>_epoch-<epoch>.
# Epoch is parsed from the active checkpoint filename (NA for the best_model.pt
# symlink or other names without an epoch_<N> token).
epoch_label=$(basename "$checkpoint_path" | grep -oE 'epoch_[0-9]+' | grep -oE '[0-9]+' | head -1)
run_dir="$(dirname "$(dirname "$experiment_dir")")/comparison/$(date +%Y-%m-%d/%H-%M-%S)_${experiment_name}_epoch-${epoch_label:-NA}"

CUDA_VISIBLE_DEVICES=0 python -m graph_signal_diffusion.cli.compare_baselines \
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
# - dataset.n_split_chunks=3 MUST match training (pompous trained 3-chunk).
#   Do NOT re-eval on a different chunk count: 80% of the 10-chunk test
#   windows were in pompous's 3-chunk TRAIN set (verified) — cross-split
#   re-eval leaks train data into test. Cross-arm comparison vs the 10-chunk
#   spotted-catfish-602 requires a retrained single-knob control, not a re-eval.
# - dataset.root = corr_0.7 graph variant (same as the whole DS8-norm-act-head family).
# - dataset.future_window=5 (T=5) matches training; GRW forecasts T=5 too.
# - dataset.standardize_target_in_x_for_revin=true must match training (RevIN).
# - dataset.test_subsample_n=50 picks 50 windows uniformly (deterministic);
#   null for full (~234 windows).
# - The diffusion baseline auto-loads architecture + diffusion config from
#   .hydra/config.yaml — including α=1.1150, cond gated/attention flags, and
#   selector exploration kwargs.
# - CUDA_VISIBLE_DEVICES=0 — both training runs finished; this eval uses GPU 0.
#   The sibling spotted-catfish-602 launcher uses GPU 1 so both can run in parallel.
# - GPU runtime at n_samples=20 + 100 DDIM steps on the 128ch denoiser:
#     test_subsample_n=50 → ~1-2h; null (~234) → ~5-8h.
# - For tail-coverage (cov_95 / cov_99): bump n_samples to 50 / 200.
