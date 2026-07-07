#!/bin/bash
# Compare GRW vs carmine-moose-32 on the SP500-cleaned test split.
#
# carmine-moose-32 architecture (DS8-wide-uniform-128 + norm-act head):
#   Same backbone as colorful-wapiti-739: base_channels=128, gamma=[2,2,2],
#   num_layers=3, num_bottleneck_layers=3, learned NodeSelector STE T_min=0.5
#   linear anneal_ratio=0.90, dropout=0.10, RevIN, DDIM eta=0.2, 500 train /
#   100 sampling timesteps, corr_0.7 graph, bs=24, T=5, n_split_chunks=3.
#
#   Deltas vs wapiti (the cond-boost + Gumbel + lr-UP composite ablation):
#     - cond.shared_encoder.temporal.mixer.gated=true (vs wapiti=false)
#     - cond.shared_encoder.temporal.mixer.attention.enabled=true
#       (num_heads=2, dropout=0.1, max_timesteps=20)
#     - selector_exploration_noise=1.0 with linear schedule, warmup 2%,
#       anneal 90% (wapiti has selector_exploration_noise=0 — Gumbel off)
#     - learning_rate=5e-4 (5× UP from wapiti's 1e-4)
#     - selector temperature anneal: 1.0 → 0.5 (same as wapiti)
#
#   RevIN bias-compensation is the SAME as wapiti (also as F3's "fix"):
#     - revin_blend_weight=0.7
#     - revin_sigma_correction auto-resolved via ${revin_alpha:w,b} →
#       α = 1.1150246904679009, materialized as literal in .hydra/config.yaml
#       (via _persist_resolved_revin_sigma_correction at training time)
#     - Applied symmetrically in _compute_revin_stats (both forward
#       normalize at train/ELBO and reverse denormalize at sampling).
#
#   The diffusion baseline reloads architecture + diffusion config from
#   .hydra/config.yaml, so the trained-with α is automatically used at
#   eval time. NO CLI override required.
#
# Task: stock_price_forecasting_v3_learned_ds8, n_split_chunks=3 (identical
# to F3 / wapiti / pompous-pigeon's split). Pair ablations:
#   - colorful-wapiti-739 (lr=1e-4, no cond boost, no Gumbel) — REFERENCE
#   - pompous-pigeon-214  (lr=2e-5, same cond boost + Gumbel; lr DOWN 5×)
#
# Training status as of 2026-05-28: FINISHED (1000/1000 epochs, ep 999).
# Tracker leaderboard (composite = val_loss + |val_loss_gap| + 0.3·val_return_crps),
# top-5 of run (note: NONE updated past ep 350 — leaderboard frozen since
# epoch 350, the model overfit hard at this lr — train_loss dropped to 0.254
# at ep 892 while val_loss climbed to 0.325, gap +0.067):
#   ep 200 — rank 1  composite 0.2825  val_loss=0.3015  gap=+0.0210  val_crps=1.1004  cum_T_dir=0.5029
#   ep 150 — rank 2  composite 0.2884  val_loss=0.3065  gap=+0.0162  val_crps=1.1347  cum_T_dir=0.5013
#   ep 300 — rank 3  composite 0.2956  val_loss=0.3079  gap=+0.0212  val_crps=1.1335  cum_T_dir=0.5019
#   ep 350 — rank 4  composite 0.2965  val_loss=0.3091  gap=+0.0149  val_crps=1.1341  cum_T_dir=0.4794
#   ep 250 — rank 5  composite 0.2977  val_loss=0.3148  gap=+0.0286  val_crps=1.1174  cum_T_dir=0.4800
#
# Late-training cum_T metrics (NOT in leaderboard — composite-driven):
#   ep 800 — val_loss=0.3246 gap=+0.0666  cum_T_crps=2.8787  cum_T_dir=0.5087
#   ep 900 — val_loss=0.3152 gap=+0.0561  cum_T_crps=2.8515  cum_T_dir=0.5101 ← BEST cum_T_dir of run
#   ep 999 — val_loss=0.3247 gap=+0.0611  cum_T_crps=2.8837  cum_T_dir=0.5087
#
# Composite-best ep 200 also has BEST cum_T_crps (2.7390) among epochs
# with non-degenerate val_loss — late-epoch cum_T_dir gains coexist with
# severe overfit on val_loss. The composite rank-1 (ep 200) is the clean
# operating point; ep 900 is the "cum_T-direction-best, val-overfit" probe.
#
# Checkpoint selection — three operating points to probe across the run:
#
#   ep 200 (default) — tracker rank-1, best composite, lowest cum_T_crps
#     among non-degenerate epochs, train/val gap +0.021.
#     best_models/best_model_epoch_200.pt
#
#   ep 350 — tracker rank-4, smallest val_loss_gap of top-5 (+0.015):
#     best_models/best_model_epoch_350.pt
#     val_loss=0.3091, val_return_crps=1.1341 (worst CRPS in top-5).
#
#   ep 900 — best cum_T_direction_accuracy (0.5101) of the entire run;
#     val_loss heavily overfit (0.3152, gap +0.056). Probe to see if late-
#     epoch directional skill on the cumulative-return signal survives
#     the val_loss degradation:
#     DDIM_epoch_900.pt
#
#   DDIM_epoch_1000.pt — final checkpoint (ep 999, terminal state of cosine
#     schedule). val_loss=0.3247, gap=+0.0611, cum_T_crps=2.8837. Probe to
#     measure how much terminal overfit hurts test-set generalization.
#
# Hypotheses to test on the test split:
#   1. Composite rank-1 (ep 200) matches wapiti's best on val_loss / val_return_crps
#      (~0.30 / ~1.10) — i.e. the cond-boost + Gumbel + lr-UP combo does NOT
#      improve over wapiti's lr=1e-4 baseline DESPITE the added capacity.
#      If true, the 5× lr UP burned the cond-boost / Gumbel capacity gains.
#   2. cum_T_direction_accuracy at ep 200 (~0.50) is no better than F3's
#      ~0.50; ep 900 (~0.51) MIGHT show a small directional edge, but at
#      the cost of probabilistic calibration (val_return_mis_90 ≥ 12).
#   3. Spread ratio (gen/real), Cov@90 stay in the wapiti band (~0.6,
#      ~0.65) — the RevIN-fix mechanism is the same, no architectural
#      reason for a calibration shift.
#   4. Comparing test composite vs train_loss=0.254 (ep 892) confirms the
#      overfit: large train-test gap reproduces on the test split. The
#      best operating point should be EARLY (ep 200), not late.
#
# Pair comparison (sibling lr-bracket arms, same architecture):
#   - colorful-wapiti-739 (lr 1e-4, no cond boost, no Gumbel) — composite
#     0.27915 @ ep 300 (current sweep leader; pompous-pigeon TIED at ep 600).
#   - pompous-pigeon-214  (lr 2e-5, same cond boost + Gumbel as carmine, lr
#     DOWN 5× from wapiti) — composite 0.2791 @ ep 600 (currently still
#     training; matched wapiti's lead this morning).
#
# The pompous-pigeon vs carmine pair isolates the lr knob holding all other
# carmine-deltas fixed. Pompous's surge to 0.2791 vs carmine's stale 0.2825
# at this point in training is strong evidence that the cond-boost + Gumbel
# stack is fine — it was the lr=5e-4 that destroyed it.
#
# Eval setup: 20 samples per window (matching wapiti's launcher), full DDIM
# sampling (100 steps, eta=0.2 matching training), AMP enabled,
# test_subsample_n=50 (deterministic linspace over all 3 chunks).

export HYDRA_FULL_ERROR=1

experiment_name=carmine-moose-32
experiment_dir=outputs/stock_price_forecasting_v3_learned_ds8-sp500_cleaned/ugnn_sp500_v3_ds8_norm_act_head-ddim-gdm_sp500_v3_learned_ds8/$experiment_name
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_200.pt

# Alternate operating points:
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_150.pt   # composite rank 2
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_350.pt   # composite rank 4; smallest gap in top-5
checkpoint_path=$experiment_dir/trainer_chkpts/DDIM_epoch_900.pt                     # best cum_T_direction_accuracy (0.5101); val_loss overfit
# checkpoint_path=$experiment_dir/trainer_chkpts/DDIM_epoch_1000.pt                    # final ep 999; terminal overfit probe
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model.pt             # tracker rank-1 symlink (= ep 200)

batch_size=100
batch_size_val=200
n_samples=20 # 50 for full eval; 20 for quicker evaluation. Wapiti launcher uses 20.
test_subsample_n=50

# Inference-time DDIM stochasticity. Training used eta=0.2.
ddim_eta=0.2
sampling_timesteps=100

# ── RevIN σ scale correction ─────────────────────────────────────────
# carmine's saved .hydra/config.yaml has revin_sigma_correction=1.1150246904679009
# materialized as a literal (via _persist_resolved_revin_sigma_correction
# at training time). The diffusion baseline auto-loads this — NO CLI
# override required for default eval. This is the SAME α as wapiti.
#
# To probe sensitivity to α at eval time, uncomment one of:
# revin_sigma_correction_override=1.1150246904679009 # explicit trained value
# revin_sigma_correction_override=1.22               # F3's asymmetric arithmetic-mean compensation
# revin_sigma_correction_override=1.0                # disable correction entirely
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
# - dataset.root points to the corr_0.7 graph variant (same as F3, wapiti,
#   pompous-pigeon, and the rest of the DS8-norm-act-head family).
# - dataset.future_window=5 (T=5) matches training; GRW forecasts T=5 too.
# - dataset.n_split_chunks=3 MUST match the training-time split.
# - dataset.standardize_target_in_x_for_revin=true must match training (RevIN).
# - dataset.test_subsample_n=50 picks 50 windows uniformly (deterministic).
#   Set null for full (~234 windows).
# - The diffusion baseline auto-loads architecture and diffusion config from
#   the experiment's .hydra/config.yaml — including the materialized α=1.1150,
#   the cond gated/attention flags, and selector exploration kwargs.
# - CUDA_VISIBLE_DEVICES=1 — carmine training finished (freed GPU 1);
#   pompous-pigeon-214 still occupies GPU 0.
# - GPU runtime estimate at n_samples=20 + 100 DDIM steps on the 128ch denoiser:
#     test_subsample_n=50  → ~1-2h
#     test_subsample_n=null (full ~234) → ~5-8h
# - For tail-coverage reads (cov_95 / cov_99): bump n_samples to 50 (cov_95
#   stable) or 200 (cov_99 stable).
