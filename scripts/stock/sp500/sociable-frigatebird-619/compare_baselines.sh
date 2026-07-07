#!/bin/bash
# Compare GRW vs sociable-frigatebird-619 on the SP500-cleaned test split.
#
# ============================ SPLIT WARNING ============================
# sociable-frigatebird-619 was trained with dataset.n_split_chunks=10. This
# launcher evaluates on the NATIVE 10-chunk test split. This is the only
# valid eval split for this checkpoint:
#   - GRW vs diffusion HERE is valid (GRW is recomputed on the same 10-chunk
#     test windows; both baselines see the same data).
#   - Results ARE directly comparable to spotted-catfish-602 and
#     fierce-squirrel-222 — all three are n_split_chunks=10 on the SAME
#     dataset, so they share the identical 10-chunk test windows. This is the
#     SMALL-vs-FULL capacity comparison on held-out test (frigatebird +
#     catfish = small 1.88M; fierce-squirrel = full 5.05M).
#   - Cross-arm comparison vs the 3-chunk family (wapiti / carmine /
#     pompous-pigeon) is NOT valid. The test windows differ, AND re-evaluating
#     across chunk counts LEAKS: ~80% of the 3-chunk test windows were in this
#     run's 10-chunk TRAIN set (verified by index overlap). Do NOT set
#     dataset.n_split_chunks=3 here to "compare" — that evaluates the model on
#     its own training data.
# =======================================================================
#
# sociable-frigatebird-619 architecture (DS8 norm-act head, ~1.88M params —
# the SMALL backbone, ~37% of the full 128/3/3 model):
#   base_channels=64, gamma=[2,2,2], num_layers=2 (dilations=[1,1]),
#   num_bottleneck_layers=2, learned NodeSelector STE T_min=0.5, dropout=0.10,
#   RevIN, DDIM eta=0.2, 500 train / 100 sampling timesteps, corr_0.7 graph,
#   bs=64, T=5.
#   cond-boost: cond.shared_encoder.temporal.mixer gated=true + attention
#     (heads=2, dropout=0.1, max_timesteps=20).
#   Gumbel exploration: selector_exploration_noise=1.0 linear.
#   Schedules (temperature + exploration) complete at ep 3750 (0.75 fraction
#     of max_epochs=5000): warmup_ratio=0.02, anneal_ratio=0.73.
#
#   This is the LONG-TRAINING extension of spotted-catfish-602 at the
#   sweet-spot lr=1e-4: same small 64/2/2 capacity + 10-chunk split, but
#   lr DOWN to 1e-4 (catfish: 5e-4) and 5× the epoch budget (5000 vs 1000).
#
#   RevIN bias-compensation:
#     - revin_blend_weight=0.7
#     - revin_sigma_correction = 1.107108709910147, materialized as a literal
#       in .hydra/config.yaml (auto-loaded at eval — NO CLI override). The
#       10-chunk α (split-dependent; same as catfish/fierce-squirrel, NOT the
#       3-chunk family's 1.1150).
#
# Task: stock_price_forecasting_v3_learned_ds8, n_split_chunks=10 (NATIVE).
# Comparable 10-chunk cohort (share the test set):
#   - spotted-catfish-602   (SMALL 64/2/2, lr=5e-4, 1000ep) — the base
#   - fierce-squirrel-222    (FULL 128/3/3, lr=1e-4, 2000ep) — the full-capacity
#     counterpart; on val it LOST to frigatebird (composite 0.2544 vs 0.2513)
#     and overfit its back half. This launcher's test result is the capacity verdict.
#
# Training status: FINISHED (5000/5000 epochs). Unlike fierce-squirrel (full
# capacity, peaked ep 450 then overfit), the SMALL model KEPT IMPROVING and did
# NOT overfit: the train/val gap oscillated around zero the whole run (final
# stretch ep4000-5000: gap in [-0.036, +0.038], no systematic widening). The
# leaderboard rank-1 is LATE (ep 4500) — the 5× budget genuinely paid off.
#
# Tracker leaderboard (composite = val_loss + |val_loss_gap| + 0.3·val_return_crps;
# 10-chunk val split — comparable to catfish/fierce-squirrel, NOT the 3-chunk family):
#   ep 4500 — rank 1  comp 0.2513  val_loss=0.2891 gap=-0.0095 vcrps=0.9629 mis90=9.99 kurt=7.36 cumTdir=0.5101
#   ep 1800 — rank 2  comp 0.2533  val_loss=0.2924 gap=+0.0142 vcrps=0.9592 mis90=9.76 kurt=8.88 cumTdir=0.5002
#   ep 2800 — rank 3  comp 0.2536  val_loss=0.2936 gap=+0.0064 vcrps=0.9625 mis90=9.67 kurt=9.11 cumTdir=0.5086
#   ep 4200 — rank 4  comp 0.2549  val_loss=0.2866 gap=-0.0361 vcrps=0.9626 mis90=9.85 kurt=7.55 cumTdir=0.5166
#   ep 1600 — rank 5  comp 0.2551  val_loss=0.2911 gap=-0.0164 vcrps=0.9708 mis90=9.95 kurt=11.91 cumTdir=0.4930
#
# This arm is the STRONGEST on val composite of the entire 10-chunk cohort
# (0.2513 < fierce-squirrel's full-capacity 0.2544) — the test-set eval decides
# whether small-beats-full carries to held-out data.
#
# Checkpoint selection:
#
#   ep 4500 (default) — tracker rank-1, the genuine best of the 5000-epoch run
#     (the long budget paid off; it beat the earlier ep1800). val_loss=0.2891,
#     gap=-0.0095, kurtosis 7.36 (excellent), cumTdir 0.5101.
#     best_models/best_model_epoch_4500.pt
#
#   ep 4200 — rank 4; LOWEST val_loss of the run (0.2866), BEST cum_T_dir of
#     top-5 (0.5166), kurtosis 7.55. gap very negative (-0.036) — that eval's
#     val draw was easier than train, so read it as noisy-but-strong.
#     best_models/best_model_epoch_4200.pt
#
#   ep 1800 — rank 2; the EARLY best (reached at ~1/3 the compute). Useful as
#     the "did the extra 3200 epochs actually help on TEST?" reference point —
#     eval ep4500 vs ep1800 to see if the val-side late gain generalizes.
#     best_models/best_model_epoch_1800.pt
#
#   ep 2800 — rank 3; smallest |val_loss_gap| of the top tier (+0.0064).
#     best_models/best_model_epoch_2800.pt
#
# Hypotheses to test on the (10-chunk) test split:
#   1. CAPACITY verdict on held-out test: frigatebird (small 1.88M) beat
#      fierce-squirrel (full 5.05M) on val (0.2513 vs 0.2544). Does it carry to
#      the shared test set? If yes, capacity is genuinely counterproductive on
#      this split — the headline finding.
#   2. Did the LATE gain generalize? ep4500 (0.2513) beat ep1800 (0.2533) on
#      val. Compare both on test — if ep4500 still wins, the 5× budget bought
#      real generalization, not val overfit.
#   3. cum_T_direction_accuracy (~0.51) — survives on held-out test or collapses
#      to ~0.50?
#   4. Kurtosis (7.36) generalization — the small model at lr=1e-4 has the
#      lowest kurtosis gap of the cohort; should hold on test.
#
# Eval setup: 20 samples per window (matching the family launchers), full DDIM
# sampling (100 steps, eta=0.2 matching training), AMP enabled,
# test_subsample_n=50 (deterministic linspace over all 10 chunks).

export HYDRA_FULL_ERROR=1

experiment_name=sociable-frigatebird-619
experiment_dir=outputs/stock_price_forecasting_v3_learned_ds8-sp500_cleaned/ugnn_sp500_v3_ds8_norm_act_head-ddim-gdm_sp500_v3_learned_ds8/$experiment_name
checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_4500.pt

# Alternate operating points:
# checkpoint_path=$experiment_dir/trainer_chkpts/DDIM_epoch_4950.pt
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_4200.pt   # lowest val_loss (0.2866) + best cum_T_dir of top-5 (0.5166)
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_1800.pt   # the EARLY best; eval vs ep4500 to test if the late gain generalizes
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_2800.pt   # smallest val_loss_gap of top tier (+0.0064)
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model.pt              # tracker rank-1 symlink (= ep 4500)

batch_size=100
batch_size_val=200
n_samples=100 # 50 for full eval; 20 for quicker evaluation. Family launchers use 20.
test_subsample_n=null # instead of 50 to avoid the possible phase bug.

# Inference-time DDIM stochasticity. Training used eta=0.2.
ddim_eta=0.2
sampling_timesteps=100

# ── RevIN σ scale correction ─────────────────────────────────────────
# frigatebird's saved .hydra/config.yaml has revin_sigma_correction=1.107108709910147
# materialized as a literal (the 10-chunk α). The diffusion baseline auto-loads
# this — NO CLI override required for default eval.
#
# To probe sensitivity to α at eval time, uncomment one of:
# revin_sigma_correction_override=1.107108709910147 # explicit trained value
# revin_sigma_correction_override=1.0               # disable correction entirely
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
    dataset.n_split_chunks=10 \
    dataset.standardize_target_in_x_for_revin=true \
    dataset.test_subsample_n=$test_subsample_n \
    ++paper_figures.fig1.max_panels=20 \
    'eval_splits={test: null}' \
    wandb.enabled=False

# Notes:
# - dataset.n_split_chunks=10 MUST match training (see SPLIT WARNING above).
# - dataset.root = corr_0.7 graph variant (same as the whole DS8-norm-act-head family).
# - dataset.future_window=5 (T=5) matches training; GRW forecasts T=5 too.
# - dataset.standardize_target_in_x_for_revin=true must match training (RevIN).
# - dataset.test_subsample_n=50 picks 50 windows uniformly across all 10
#   chunks (deterministic); null for full (~240 windows).
# - The diffusion baseline auto-loads architecture + diffusion config from
#   .hydra/config.yaml — including α=1.1071, base_channels=64, num_layers=2,
#   num_bottleneck_layers=2, cond gated/attention flags, exploration kwargs.
# - CUDA_VISIBLE_DEVICES=0 — frigatebird training FINISHED (freed GPU 0);
#   the fierce-squirrel-222 eval launcher uses GPU 1, so both can run in parallel.
# - GPU runtime at n_samples=20 + 100 DDIM steps on the SMALL 64ch denoiser:
#     test_subsample_n=50 → <1h; null (~240) → ~3-5h.
# - For tail-coverage (cov_95 / cov_99): bump n_samples to 50 / 200.
