#!/bin/bash
# Compare GRW vs colorful-wapiti-739 on the SP500-cleaned test split.
#
# colorful-wapiti-739 architecture (DS8-wide-uniform-128 + norm-act head, ~6M params):
#   Same backbone as fervent-cicada-140 (F3): base_channels=128, gamma=[2,2,2],
#   num_layers=3, num_bottleneck_layers=3, learned NodeSelector STE T_min=0.5
#   linear anneal_ratio=0.90, dropout=0.10, RevIN, DDIM eta=0.2, 500 train /
#   100 sampling timesteps, corr_0.7 graph, bs=24, T=5, n_split_chunks=3.
#
#   ONLY difference vs F3: RevIN bias-compensation knobs.
#     F3:     revin_blend_weight implicit 1.0; legacy task-level
#             variance_correction (alpha=1.22 applied asymmetrically at sample
#             time only — ELBO/NLL inconsistent with the corrected distribution).
#     wapiti: revin_blend_weight=0.7; revin_sigma_correction auto-resolved via
#             ${revin_alpha:w,b} resolver →
#             α = 1 / (1 - 0.7·(1 - 0.8526)) = 1.1151
#             (where b=0.8526 is sigma_cond_mean_train injected at training time
#             from SP500 dataset metadata).
#             Applied symmetrically in _compute_revin_stats — both forward
#             normalize (training, ELBO) and reverse denormalize (sampling)
#             use the same corrected scale. ELBO/NLL is consistent with the
#             corrected generative distribution.
#
#   The resolved α=1.1151 is materialized as a literal in
#   .hydra/config.yaml (via _persist_resolved_revin_sigma_correction in
#   cli/train.py — see docs/agent_summaries/persist_revin_sigma_correction_
#   2026-05-26.md). The diffusion baseline reloads architecture + diffusion
#   config from the experiment's .hydra/config.yaml, so the trained-with α is
#   automatically used at eval time. NO CLI override required.
#
# Task: stock_price_forecasting_v3_learned_ds8, n_split_chunks=3 (identical to
# F3's split). Sibling: charming-rat-237 (same arm with selector_temperature_
# schedule.enabled=false; T=1.0 throughout).
#
# Training status as of 2026-05-27: in progress (still running on GPU 0,
# ep ≥ 750/1000). Tracker leaderboard (raw composite =
# val_loss + |val_loss_gap| + 0.3·val_return_crps), best 5 of run so far:
#   ep 300 — rank 1  composite 0.27915  val_loss=0.302  |gap|=0.0044  cum_T_dir=0.494
#   ep 500 — rank 2  composite 0.28412  val_loss=0.305  |gap|=0.016   cum_T_dir=0.493
#   ep 450 — rank 3  composite 0.28413  val_loss=0.307  |gap|=0.011   cum_T_dir=0.491
#   ep 250 — rank 4  composite 0.28643  val_loss=0.310  |gap|=0.011   cum_T_dir=0.499
#   ep 150 — rank 5  composite 0.28553  val_loss=0.300  |gap|=0.011   cum_T_dir=0.507
#
# Checkpoint selection — three operating points to probe across the run:
#
#   ep 300 (default) — tracker rank-1, near-zero val_loss_gap (~0.004), comparable
#     to F3 ep475's gap. Best composite of the run so far.
#     best_models/best_model_epoch_300.pt
#
#   ep 500 — tracker rank-2:
#     best_models/best_model_epoch_500.pt
#     val_loss=0.305, val_return_crps=1.107 (lowest of top-5), eig1_r=0.725
#
#   ep 700 — latest available; best cum_T metrics in the run (post-leaderboard):
#     DDIM_epoch_700.pt
#     val_loss=0.320, val_cum_T_crps=2.651 (lowest of run, -2% vs ep 300),
#     val_cum_T_direction_accuracy=0.514 (highest of run, +2pp vs ep 300).
#     Selector temperature has annealed to T=0.622 (post-warmup, still 30%
#     more anneal remaining at ep 1000).
#
# Hypotheses to test on the test split:
#   1. Spread ratio (gen/real) lifts from F3 ep475's ~0.58 toward 1.0 because
#      the symmetric σ correction enters training (no longer just a sample-
#      time hack). The new σ scaling expands the gen distribution at train and
#      ensures the noise predictor sees the same scale at eval.
#   2. Cov@90 lifts from F3 ep475's 0.66 toward nominal 0.90.
#   3. val_loss / val_return_crps stay comparable to F3 ep475's (~0.35 / ~1.11).
#   4. cum_T_direction_accuracy at ep 700 (~0.514) translates to a real test-set
#      directional skill gain — F3 ep475's was ~0.50 (no skill).
#
# Pair comparison: turquoise-chimpanzee-597 (n_split_chunks=5) and
# charming-rat-237 (selector anneal disabled). Both share corr_0.7 + DS8 + RevIN-fix.
#
# Eval setup: 20 samples per window (matching F3's launcher), full DDIM
# sampling (100 steps, eta=0.2 matching training), AMP enabled,
# test_subsample_n=50 (deterministic linspace over all 3 chunks).

export HYDRA_FULL_ERROR=1

experiment_name=colorful-wapiti-739
experiment_dir=outputs/stock_price_forecasting_v3_learned_ds8-sp500_cleaned/ugnn_sp500_v3_ds8_norm_act_head-ddim-gdm_sp500_v3_learned_ds8/$experiment_name
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_300.pt

# Alternate operating points:
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_500.pt   # composite rank 2; lowest return CRPS in top-5
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_450.pt   # composite rank 3
checkpoint_path=$experiment_dir/trainer_chkpts/DDIM_epoch_700.pt                     # latest; best cum_T_crps and cum_T_direction_accuracy
# checkpoint_path=$experiment_dir/trainer_chkpts/DDIM_epoch_600.pt                     # also strong on cum_T metrics
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model.pt             # tracker rank-1 symlink (currently ep 300)

batch_size=100
batch_size_val=200
n_samples=20 # 50 for full eval; 20 for quicker evaluation. F3 launcher uses 20.
test_subsample_n=50

# Inference-time DDIM stochasticity. Training used eta=0.2.
ddim_eta=0.2
sampling_timesteps=100

# ── RevIN σ scale correction ─────────────────────────────────────────
# wapiti's saved .hydra/config.yaml has revin_sigma_correction=1.1151
# materialized as a literal (via _persist_resolved_revin_sigma_correction
# at training time). The diffusion baseline auto-loads this — NO CLI
# override required for default eval.
#
# To probe sensitivity to α at eval time, uncomment one of:
# revin_sigma_correction_override=1.1151   # explicit trained value
# revin_sigma_correction_override=1.22     # F3's asymmetric arithmetic-mean compensation
# revin_sigma_correction_override=1.0      # disable correction entirely
#
# To activate the override, also add to the python call:
#   ++baselines.diffusion.diffusion_overrides.revin_sigma_correction=$revin_sigma_correction_override
#
# IMPORTANT: do NOT combine with the legacy task-level variance_correction
# block below (they compound multiplicatively).

# Legacy task-level variance correction — KEEP OFF (the symmetric in-model
# correction is now the canonical mechanism). Documented for reference only.
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
    dataset.n_split_chunks=3 \
    dataset.standardize_target_in_x_for_revin=true \
    dataset.test_subsample_n=$test_subsample_n \
    'eval_splits={test: null}' \
    wandb.enabled=False

# Notes:
# - dataset.root points to the corr_0.7 graph variant (same as F3, CD, and
#   the rest of the DS8-norm-act-head family).
# - dataset.future_window=5 (T=5) matches training; GRW forecasts T=5 too.
# - dataset.n_split_chunks=3 MUST match the training-time split. Sibling
#   turquoise-chimpanzee-597 used 5 chunks — DO NOT cross-reference.
# - dataset.standardize_target_in_x_for_revin=true must match training (RevIN).
# - dataset.test_subsample_n=50 picks 50 windows uniformly (deterministic).
#   Set null for full (~234 windows).
# - The diffusion baseline auto-loads architecture and diffusion config from
#   the experiment's .hydra/config.yaml — including the materialized α=1.1151.
# - CUDA_VISIBLE_DEVICES=1 — wapiti training still runs on GPU 0; eval uses
#   GPU 1 (freed when charming-rat-237 was paused 2026-05-27).
# - GPU runtime estimate at n_samples=20 + 100 DDIM steps on the 128ch denoiser:
#     test_subsample_n=50  → ~1-2h
#     test_subsample_n=null (full ~234) → ~5-8h
# - For tail-coverage reads (cov_95 / cov_99): bump n_samples to 50 (cov_95
#   stable) or 200 (cov_99 stable).
