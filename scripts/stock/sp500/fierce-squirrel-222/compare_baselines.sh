#!/bin/bash
# Compare GRW vs fierce-squirrel-222 on the SP500-cleaned test split.
#
# ============================ SPLIT WARNING ============================
# fierce-squirrel-222 was trained with dataset.n_split_chunks=10. This
# launcher evaluates on the NATIVE 10-chunk test split. This is the only
# valid eval split for this checkpoint:
#   - GRW vs diffusion HERE is valid (GRW is recomputed on the same 10-chunk
#     test windows; both baselines see the same data).
#   - Results ARE directly comparable to spotted-catfish-602 and
#     sociable-frigatebird-619 — all three are n_split_chunks=10 on the SAME
#     dataset, so they share the identical 10-chunk test windows (this is the
#     full-vs-small CAPACITY comparison on held-out test).
#   - Cross-arm comparison vs the 3-chunk family (wapiti / carmine /
#     pompous-pigeon) is NOT valid. The test windows differ, AND re-evaluating
#     across chunk counts LEAKS: ~80% of the 3-chunk test windows were in this
#     run's 10-chunk TRAIN set (verified by index overlap). Do NOT set
#     dataset.n_split_chunks=3 here to "compare" — that evaluates the model on
#     its own training data.
# =======================================================================
#
# fierce-squirrel-222 architecture (DS8-wide-uniform-128 + norm-act head,
# ~5.05M params — the FULL backbone):
#   base_channels=128, gamma=[2,2,2], num_layers=3, num_bottleneck_layers=3,
#   learned NodeSelector STE T_min=0.5, dropout=0.10, RevIN, DDIM eta=0.2,
#   500 train / 100 sampling timesteps, corr_0.7 graph, bs=24, T=5.
#   cond-boost: cond.shared_encoder.temporal.mixer gated=true + attention
#     (heads=2, dropout=0.1, max_timesteps=20).
#   Gumbel exploration: selector_exploration_noise=1.0 linear.
#   Schedules (temperature + exploration) complete at ep 1500 (0.75 fraction
#     of max_epochs=2000): warmup_ratio=0.02, anneal_ratio=0.73.
#
#   This is the lr=1e-4 "kurtosis-gap fix" arm — the pompous-pigeon recipe
#   (cond-boost + Gumbel, full capacity) but lr bumped 5× (2e-5 → 1e-4) +
#   10-chunk split + 2000 epochs. The lr bump worked: kurtosis gap fell from
#   pompous's persistent ~18-27 to ~8-11 (wapiti band) — see below.
#
#   RevIN bias-compensation:
#     - revin_blend_weight=0.7
#     - revin_sigma_correction = 1.107108709910147, materialized as a literal
#       in .hydra/config.yaml (auto-loaded at eval — NO CLI override). This is
#       the 10-chunk α (split-dependent; same as catfish/frigatebird, NOT the
#       3-chunk family's 1.1150).
#
# Task: stock_price_forecasting_v3_learned_ds8, n_split_chunks=10 (NATIVE).
# Comparable 10-chunk cohort (share the test set):
#   - spotted-catfish-602    (SMALL 64/2/2, lr=5e-4, 1000ep) — the base
#   - sociable-frigatebird-619 (SMALL 64/2/2, lr=1e-4, 5000ep) — small sibling
#     at THIS lr; on val it edged fierce-squirrel (composite 0.2533 vs 0.2544)
#     and did NOT overfit over its long run.
#
# Training status: FINISHED (2000/2000 epochs). The full model PLATEAUED at
# ep 450 (composite rank-1) and then OVERFIT the entire back half: train_loss
# 0.288 → 0.266, val_loss 0.300 → 0.312, gap +0.009 → +0.046 by ep 2000.
# Nothing past ep 450 beat it. The best_models tracker correctly pinned the
# early checkpoint. (Lesson: full capacity at lr=1e-4 on the 10-chunk split
# needs hard early-stopping; the extra 1550 epochs were wasted/harmful.)
#
# Tracker leaderboard (composite = val_loss + |val_loss_gap| + 0.3·val_return_crps;
# 10-chunk val split — comparable to catfish/frigatebird, NOT the 3-chunk family):
#   ep 450 — rank 1  comp 0.2544  val_loss=0.2918 gap=+0.0093 vcrps=0.9711 mis90=10.25 kurt=10.91 cumTdir=0.5318
#   ep 350 — rank 2  comp 0.2546  val_loss=0.2894 gap=+0.0082 vcrps=0.9621 mis90= 9.90 kurt=16.68 cumTdir=0.4906
#   ep 150 — rank 3  comp 0.2550  val_loss=0.2932 gap=-0.0054 vcrps=0.9546 mis90= 9.97 kurt=23.19 cumTdir=0.5038
#   ep 600 — rank 4  comp 0.2566  val_loss=0.2901 gap=+0.0002 vcrps=0.9991 mis90=10.32 kurt= 9.45 cumTdir=0.4998
#   ep 700 — rank 5  comp 0.2578  val_loss=0.3008 gap=+0.0123 vcrps=0.9563 mis90= 9.58 kurt= 8.15 cumTdir=0.5370
#
# Checkpoint selection:
#
#   ep 450 (default) — tracker rank-1, the genuine best of the 2000-epoch run.
#     Strong cum_T_direction_accuracy (0.5318) and a healthy gap (+0.009);
#     kurtosis gap 10.9 (wapiti band). best_models/best_model_epoch_450.pt
#
#   ep 700 — composite rank-5, BUT the directional / calibration pick: BEST
#     cum_T_direction_accuracy (0.5370) and BEST cum_T_crps (2.1151) and LOWEST
#     kurtosis gap (8.15) of the saved best_models. val_loss has started to
#     climb (0.3008, gap +0.012 — early overfit onset), so it trades a touch of
#     val_loss for the best multi-day-return behavior.
#     best_models/best_model_epoch_700.pt
#
#   ep 350 — rank 2; BEST val_return_crps (0.9621) and BEST val_return_mis_90
#     (9.90), but NO directional skill (cumTdir 0.49) and higher kurtosis
#     (16.7). The sharpness/calibration pick.
#     best_models/best_model_epoch_350.pt
#
#   ep 600 — rank 4; near-zero val_loss_gap (+0.0002), kurtosis 9.45.
#     best_models/best_model_epoch_600.pt
#
#   NOTE: the late checkpoints (ep 1200-2000) reach even lower kurtosis (~6.8)
#   but are OVERFIT on val_loss (gap up to +0.046) and were NOT saved as
#   best_models — do not use them.
#
# Hypotheses to test on the (10-chunk) test split:
#   1. CAPACITY verdict on held-out test: does fierce-squirrel (full 5.05M)
#      match or beat the small arms (catfish/frigatebird, 1.88M) on the shared
#      10-chunk test set? On val the small model won — if that carries to test,
#      capacity is genuinely counterproductive on this split.
#   2. cum_T_direction_accuracy at ep 450 (0.5318) / ep 700 (0.5370): does the
#      apparent directional skill SURVIVE on held-out test windows, or collapse
#      to ~0.50 (a val artifact)?
#   3. Kurtosis / calibration: ep 450's well-shaped gen distribution (kurt 10.9,
#      vs pompous's 3-chunk ~50) should hold on test — confirms the lr=1e-4 fix
#      generalizes, not just a val-set effect.
#
# Eval setup: 20 samples per window (matching the family launchers), full DDIM
# sampling (100 steps, eta=0.2 matching training), AMP enabled,
# test_subsample_n=50 (deterministic linspace over all 10 chunks).

export HYDRA_FULL_ERROR=1

experiment_name=fierce-squirrel-222
experiment_dir=outputs/stock_price_forecasting_v3_learned_ds8-sp500_cleaned/ugnn_sp500_v3_ds8_norm_act_head-ddim-gdm_sp500_v3_learned_ds8/$experiment_name
checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_450.pt

# Alternate operating points:
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_700.pt   # best cum_T_dir (0.5370) + cum_T_crps (2.115) + lowest kurtosis (8.15); val_loss climbing
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_350.pt   # best val_return_crps (0.9621) + mis_90 (9.90); no directional skill
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_600.pt   # near-zero val_loss_gap (+0.0002)
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model.pt             # tracker rank-1 symlink (= ep 450)

batch_size=100
batch_size_val=200
n_samples=20 # 50 for full eval; 20 for quicker evaluation. Family launchers use 20.
test_subsample_n=50

# Inference-time DDIM stochasticity. Training used eta=0.2.
ddim_eta=0.2
sampling_timesteps=100

# ── RevIN σ scale correction ─────────────────────────────────────────
# fierce-squirrel's saved .hydra/config.yaml has revin_sigma_correction=1.107108709910147
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
#   .hydra/config.yaml — including α=1.1071, base_channels=128, num_layers=3,
#   num_bottleneck_layers=3, cond gated/attention flags, exploration kwargs.
# - CUDA_VISIBLE_DEVICES=1 — fierce-squirrel training FINISHED (freed GPU 1);
#   sociable-frigatebird-619 still trains on GPU 0.
# - GPU runtime at n_samples=20 + 100 DDIM steps on the FULL 128ch denoiser:
#     test_subsample_n=50 → ~1-2h; null (~240) → ~5-8h.
# - For tail-coverage (cov_95 / cov_99): bump n_samples to 50 / 200.
