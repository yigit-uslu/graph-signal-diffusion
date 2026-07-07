#!/bin/bash
# Compare GRW vs spotted-catfish-602 on the SP500-cleaned test split.
#
# ============================ SPLIT WARNING ============================
# spotted-catfish-602 was trained with dataset.n_split_chunks=10. This
# launcher evaluates on the NATIVE 10-chunk test split. This is the only
# valid eval split for this checkpoint:
#   - GRW vs diffusion HERE is valid (GRW is recomputed on the same 10-chunk
#     test windows; both baselines see the same data).
#   - Cross-arm comparison vs the 3-chunk family (wapiti / carmine /
#     pompous-pigeon) is NOT valid from this run. The test windows differ,
#     AND re-evaluating across chunk counts LEAKS: 79.5% of the 3-chunk test
#     windows were in spotted-catfish's 10-chunk TRAIN set (verified by index
#     overlap). Do NOT set dataset.n_split_chunks=3 here to "compare" — that
#     evaluates the model on its own training data.
#   - To cleanly attribute spotted-catfish's gains to capacity-vs-split,
#     retrain a single-knob control (full 128/3/3 capacity + n_split_chunks=10,
#     OR reduced 64/2/2 capacity + n_split_chunks=3), not a re-eval.
# =======================================================================
#
# spotted-catfish-602 architecture (DS8 norm-act head, ~1.88M params — ~37%
# of the full 128/3/3 model):
#   Capacity-reduced + 10-chunk-split ablation of carmine-moose-32:
#     - base_channels=64            (carmine: 128)
#     - gnn_config.num_layers=2     (carmine:   3), dilations=[1,1]
#     - num_bottleneck_layers=2     (carmine:   3)
#     - dataset.batch_size=64       (carmine:  24)
#     - dataset.n_split_chunks=10   (carmine:   3)
#   Kept from carmine: cond-boost (gated + attention heads=2/dropout=0.1/
#     max_timesteps=20), Gumbel exploration (1.0 linear, warmup 2%, anneal
#     90%), selector temperature anneal (1.0→0.5), lr=5e-4, RevIN, DDIM
#     eta=0.2, 500 train / 100 sampling timesteps, corr_0.7 graph, T=5,
#     dropout=0.10, max_epochs=1000.
#
#   RevIN bias-compensation:
#     - revin_blend_weight=0.7
#     - revin_sigma_correction = 1.107108709910147, materialized as a literal
#       in .hydra/config.yaml (auto-loaded at eval — NO CLI override).
#       NOTE: this α differs slightly from the 3-chunk family's 1.1150 — the
#       resolver reads sigma_cond_mean_train, which is split-dependent. Yet
#       another reason cross-arm metric comparison is not apples-to-apples.
#
# Task: stock_price_forecasting_v3_learned_ds8, n_split_chunks=10 (NATIVE).
# Control (3-chunk, NOT directly comparable): carmine-moose-32 (full capacity,
# same lr=5e-4, 3-chunk). Sibling lr-bracket: pompous-pigeon-214.
#
# Training status as of 2026-05-29: FINISHED (1000/1000 epochs). Unlike
# carmine (which overfit hard: train→0.254, val→0.325, gap +0.067), this arm
# did NOT overfit — train floored ~0.27-0.29 (final 0.287), val stayed
# ~0.29-0.31, and the train/val gap oscillated around ZERO (-0.038 to
# +0.029). The negative gaps are partly the 10-chunk split making val
# in-regime (easier), partly the reduced capacity not memorizing.
#
# Tracker leaderboard (composite = val_loss + |val_loss_gap| + 0.3·val_return_crps),
# top-5 of run (composite values are on the 10-chunk val split — NOT
# comparable to the 3-chunk family's composites):
#   ep 300 — rank 1  composite 0.2519  val_loss=0.2922  gap=-0.0282  val_crps=0.9525  cum_T_dir=0.5107
#   ep 350 — rank 2  composite 0.2537  val_loss=0.2794  gap=-0.0096  val_crps=0.9758  cum_T_dir=0.5190
#   ep 600 — rank 3  composite 0.2543  val_loss=0.2909  gap=-0.0053  val_crps=0.9530  cum_T_dir=0.5209
#   ep 800 — rank 4  composite 0.2548  val_loss=0.2937  gap=+0.0142  val_crps=0.9532  cum_T_dir=0.5267
#   ep 500 — rank 5  composite 0.2564  val_loss=0.2962  gap=+0.0135  val_crps=0.9749  cum_T_dir=0.5305
#
# Notable: cum_T_direction_accuracy is consistently > 0.51 on the 10-chunk
# val set (peaks 0.5320 at ep 450) — apparent directional skill on the 5-day
# cumulative-return signal, UNLIKE the 3-chunk family (~0.49, no skill). This
# is the single most interesting signal in the run, BUT it may be a split
# artifact (in-regime val is easier to call directionally). The test-split
# eval here measures whether it holds on held-out 10-chunk test windows.
#
# Checkpoint selection:
#
#   ep 300 (default) — tracker rank-1, lowest val_crps (0.9525), low
#     cum_T_crps (2.13), cum_T_dir 0.5107.
#     best_models/best_model_epoch_300.pt
#
#   ep 350 — tracker rank-2; BEST val_return_mis_90 (9.152) of the run;
#     cum_T_dir 0.5190:
#     best_models/best_model_epoch_350.pt
#
#   ep 450 — BEST cum_T_direction_accuracy of the run (0.5320); NOT in
#     best_models (composite rank > 5), so use the periodic DDIM checkpoint:
#     DDIM_epoch_450.pt
#
#   ep 400 — BEST cum_T_crps of the run (2.0726); also periodic-only:
#     DDIM_epoch_400.pt
#
# Hypotheses to test on the (10-chunk) test split:
#   1. GRW-vs-diffusion gap is wider than the 3-chunk family's — i.e. the
#      diffusion model beats GRW by more on the easier in-regime test set.
#   2. cum_T_direction_accuracy > 0.51 SURVIVES on held-out test windows
#      (not just val). If it holds, the in-regime training is unlocking real
#      directional skill; if it collapses to ~0.50, it was a val artifact.
#   3. Calibration (Cov@90, spread ratio) is TIGHTER than the 3-chunk family
#      because val/test are in-regime (less distribution shift to cover).
#
# Eval setup: 20 samples per window (matching the family launchers), full
# DDIM sampling (100 steps, eta=0.2 matching training), AMP enabled,
# test_subsample_n=50 (deterministic linspace over all 10 chunks).

export HYDRA_FULL_ERROR=1

experiment_name=spotted-catfish-602
experiment_dir=outputs/stock_price_forecasting_v3_learned_ds8-sp500_cleaned/ugnn_sp500_v3_ds8_norm_act_head-ddim-gdm_sp500_v3_learned_ds8/$experiment_name
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_300.pt

# Alternate operating points:
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_800.pt     # rank 4; best cum_T_direction_accuracy (0.5267) and best val_loss (0.2937) of the run, but higher val_crps (0.9532) and worse val_return_mis_90 (9.652) than rank 1
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_600.pt     # rank 3; best cum_T_crps (2.0726) of the run, but higher val_loss (0.2909) and worse val_return_mis_90 (9.152) than rank 1 and 2
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_350.pt   # rank 2; best val_return_mis_90 of run
# checkpoint_path=$experiment_dir/trainer_chkpts/DDIM_epoch_450.pt                     # best cum_T_direction_accuracy (0.5320)
# checkpoint_path=$experiment_dir/trainer_chkpts/DDIM_epoch_400.pt                     # best cum_T_crps (2.0726)
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model.pt             # tracker rank-1 symlink (= ep 300)
checkpoint_path=$experiment_dir/trainer_chkpts/DDIM_epoch_1000.pt 

batch_size=100
batch_size_val=200
n_samples=20 # 50 for full eval; 20 for quicker evaluation. Family launchers use 20.
test_subsample_n=50

# Inference-time DDIM stochasticity. Training used eta=0.2.
ddim_eta=0.2
sampling_timesteps=100

# ── RevIN σ scale correction ─────────────────────────────────────────
# spotted-catfish's saved .hydra/config.yaml has revin_sigma_correction=1.107108709910147
# materialized as a literal. The diffusion baseline auto-loads this — NO CLI
# override required for default eval.
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
#   This is the single most important difference from the family launchers.
# - dataset.root = corr_0.7 graph variant (same as the whole DS8-norm-act-head family).
# - dataset.future_window=5 (T=5) matches training; GRW forecasts T=5 too.
# - dataset.standardize_target_in_x_for_revin=true must match training (RevIN).
# - dataset.test_subsample_n=50 picks 50 windows uniformly across all 10
#   chunks (deterministic); null for full (~240 windows).
# - The diffusion baseline auto-loads architecture + diffusion config from
#   .hydra/config.yaml — including α=1.1071, base_channels=64, num_layers=2,
#   num_bottleneck_layers=2, cond gated/attention flags, exploration kwargs.
# - CUDA_VISIBLE_DEVICES=1 — both training runs finished; this eval uses GPU 1.
#   The sibling pompous-pigeon-214 launcher uses GPU 0 so both can run in parallel.
# - GPU runtime at n_samples=20 + 100 DDIM steps on the 64ch denoiser:
#     test_subsample_n=50 → <1h (smaller model than the family); null (~240) → ~3-5h.
# - For tail-coverage (cov_95 / cov_99): bump n_samples to 50 / 200.
