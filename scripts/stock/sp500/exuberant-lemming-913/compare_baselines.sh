#!/bin/bash
# Compare GRW vs exuberant-lemming-913 on the SP500-cleaned test split.
#
# ============================ SPLIT WARNING ============================
# exuberant-lemming-913 was trained with dataset.n_split_chunks=10. This
# launcher evaluates on the NATIVE 10-chunk test split — the only valid eval
# split for this checkpoint:
#   - GRW vs diffusion HERE is valid (GRW recomputed on the same 10-chunk test
#     windows; both baselines see the same data).
#   - Results ARE directly comparable to spotted-catfish-602,
#     sociable-frigatebird-619, and fierce-squirrel-222 — all four are
#     n_split_chunks=10 on the SAME dataset → identical 10-chunk test windows.
#     This launcher completes the small-model (LR, budget) 2×2:
#         catfish (5e-4, 1000ep) · LEMMING (5e-4, 5000ep) · frigatebird (1e-4, 5000ep)
#     and the small-vs-full capacity axis vs fierce-squirrel (FULL 128/3/3).
#   - NOT comparable to the 3-chunk family (wapiti / carmine / pompous) — the
#     test windows differ AND re-splitting LEAKS ~80% train into test
#     (memory: reference_n_split_chunks_cross_split_leakage). Do NOT set
#     dataset.n_split_chunks=3 here.
# =======================================================================
#
# ===================== K / n_samples COMPARABILITY =====================
# exuberant-lemming-913 was TRAINED with trainer.n_samples_per_input=20 (K=20),
# whereas catfish / frigatebird / fierce-squirrel trained with K=10. The CRPS
# estimator is the biased 1/(2M^2) form (evaluation/probabilistic_metrics.py),
# whose bias shrinks with M — so the TRAINING-time val composite / val_return_crps
# are NOT cross-arm comparable (K differs).
#   BUT compare_baselines runs n_samples=20 for ALL arms (this launcher and the
#   frigatebird / fierce-squirrel ones — verified: their comparison runs used
#   baselines.diffusion.n_samples=20). So this TEST-set eval IS clean and directly
#   comparable across the cohort. Keep n_samples=20 below to preserve that.
# =======================================================================
#
# exuberant-lemming-913 architecture (DS8 norm-act head, ~1.88M params — the
# SMALL backbone, identical to sociable-frigatebird-619):
#   base_channels=64, gamma=[2,2,2], num_layers=2 (dilations=[1,1]),
#   num_bottleneck_layers=2, learned NodeSelector STE T_min=0.5, dropout=0.10,
#   RevIN, DDIM eta=0.2, 500 train / 100 sampling timesteps, corr_0.7 graph,
#   bs=64, T=5.
#   cond-boost: cond.shared_encoder.temporal.mixer gated=true + attention
#     (heads=2, dropout=0.1, max_timesteps=20).
#   Gumbel exploration: selector_exploration_noise=1.0 linear.
#   Schedules (temperature + exploration) complete at ep 3750 (0.75 fraction of
#     max_epochs=5000): warmup_ratio=0.02, anneal_ratio=0.73.
#
#   THE DISTINGUISHING KNOB: base lr = 5e-4 (frigatebird: 1e-4). This is the
#   high-LR variant built to test the "LR-floor" hypothesis — frigatebird's
#   cosine hit its 5e-6 floor by ~ep4500 (back ~1250 epochs near-frozen), yet
#   its kurtosis was still falling. Raising lr to 5e-4 lifts the floor to
#   0.05*5e-4 = 2.5e-5 (5x), unfreezing the tail. See
#   docs/agent_summaries/frigatebird_highlr_5e4_arm_2026-06-05.md.
#
#   RevIN bias-compensation (auto-loaded from .hydra/config.yaml — NO CLI override):
#     - revin_blend_weight=0.7
#     - revin_sigma_correction=1.107108709910147 (10-chunk α; same as
#       catfish/frigatebird/fierce-squirrel, NOT the 3-chunk family's 1.1150).
#
# Task: stock_price_forecasting_v3_learned_ds8, n_split_chunks=10 (NATIVE).
#
# Training status: FINISHED (5000/5000). Stable (grad_norm 0.07-0.19, no 5x-LR
# blowup). KEY RESULT — the high LR worked as intended on its target metric:
#   - val_loss SATURATED early (~0.290 from ep350; best 0.2903 @ep1800) and never
#     beat frigatebird's 0.2866 — so on the clean K-independent metric, full
#     frigatebird stays marginally ahead.
#   - kurtosis KEPT IMPROVING long after val_loss plateaued: val_kurtosis_gen
#     fell 40 -> ~6-9 (real ~2.44 in val-space), gap roughly HALVED vs
#     frigatebird's ~+9-10. Lowest gen 6.46 @ep1300.
#   - The composite optimum is FRONT-LOADED (rank-1 ep600; nothing after ep1800
#     made the top-5) — high LR finds its composite optimum early, then trades
#     further epochs for tail-shape, not loss.
#
# Final tracker leaderboard (composite = val_loss + |val_loss_gap| + 0.3*val_return_crps;
#   K=20 — within-run valid, NOT cross-arm comparable, see note above):
#   ep  600 — rank 1  comp 0.2487  val_loss=0.2927 gap=+0.0039 vcrps=0.9178 kurt_gen=13.50 cumTdir=0.5037
#   ep  800 — rank 2  comp 0.2495  val_loss=0.2977 gap=+0.0046 vcrps=0.9053 kurt_gen=12.96 cumTdir=0.5233
#   ep  300 — rank 3  comp 0.2498  val_loss=0.2958 gap=+0.0026 vcrps=0.9205 kurt_gen=18.30 cumTdir=0.5013
#   ep 1100 — rank 4  comp 0.2499  val_loss=0.2963 gap=+0.0005 vcrps=0.9268 kurt_gen=10.05 cumTdir=0.5082
#   ep 1800 — rank 5  comp 0.2512  val_loss=0.2903 gap=+0.0125 vcrps=0.9167 kurt_gen= 8.01 cumTdir=0.5035
#
# Checkpoint selection — NOTE the composite-best and kurtosis-best DIVERGE for
# this arm (high LR decoupled them), so pick by what you want to measure:
#
#   ep 600 (default) — composite rank-1; the principled cohort-comparison point
#     (matches how frigatebird/fierce are compared at THEIR composite-best). BUT
#     its kurtosis (13.50) is NOT representative of the arm's tail-shape win.
#     best_models/best_model_epoch_600.pt
#
#   ep 1800 — best val_loss of the whole run (0.2903) AND good kurtosis (8.01);
#     a saved best_model (rank-5). The best all-rounder / "deployable best" on
#     the clean metric. Recommended SECOND eval.
#     best_models/best_model_epoch_1800.pt
#
#   ep 1300 — the KURTOSIS SHOWCASE: lowest val_kurtosis_gen (6.46, gap +4.0).
#     NOT a best_model (composite not top-5; val_loss 0.3045 is worse) — use the
#     PERIODIC checkpoint. Eval this to capture the arm's headline tail-shape
#     result on test.
#     DDIM_epoch_1300.pt
#
#   ep 1100 — rank 4; smallest |val_loss_gap| (+0.0005) + good kurtosis (10.05).
#     best_models/best_model_epoch_1100.pt
#
# Hypotheses to test on the (10-chunk) test split:
#   1. KURTOSIS on held-out test: does the val-side tail-shape win (gap halved
#      vs frigatebird) carry to test? Eval ep1300/ep1800, compare test kurtosis
#      to frigatebird's ~16.3 (test) at ep4500. This is the arm's headline.
#   2. SHARPNESS cost: val_loss plateaued above frigatebird — does lemming give
#      up test return_crps for the kurtosis gain, or hold even?
#   3. (LR, budget) 2x2: lemming (5e-4, 5000) vs catfish (5e-4, 1000) vs
#      frigatebird (1e-4, 5000) on the shared test windows — did 5x budget at
#      high LR beat catfish, and did high LR beat frigatebird on anything but
#      kurtosis?
#   4. cum_T_direction_accuracy (~0.51) + coverage_90 survival on test.
#
# Eval setup: 20 samples/window (cohort-matched → clean cross-arm), full DDIM
# sampling (100 steps, eta=0.2 matching training), AMP, test_subsample_n=50.

export HYDRA_FULL_ERROR=1

experiment_name=exuberant-lemming-913
experiment_dir=outputs/stock_price_forecasting_v3_learned_ds8-sp500_cleaned/ugnn_sp500_v3_ds8_norm_act_head-ddim-gdm_sp500_v3_learned_ds8/$experiment_name
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_600.pt

# Alternate operating points (composite-best and kurtosis-best DIVERGE here):
checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_1800.pt  # best val_loss (0.2903) + good kurt (8.01); deployable best. RECOMMENDED 2nd
# checkpoint_path=$experiment_dir/trainer_chkpts/DDIM_epoch_1300.pt                    # KURTOSIS showcase: lowest val_kurtosis_gen (6.46). Periodic (not a best_model)
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_1100.pt  # smallest |gap| (+0.0005) + kurt 10.05
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model.pt             # tracker rank-1 symlink (= ep 600)
# checkpoint_path=$experiment_dir/trainer_chkpts/DDIM_epoch_5000.pt           # FINAL checkpoint — test if the tail-shape win keeps improving after ep600, or if it overfits (val_kurtosis_gen rose to 8.46 by ep5000, gap +5.5 vs frigatebird).
# checkpoint_path=$experiment_dir/trainer_chkpts/DDIM_epoch_2600.pt 
# checkpoint_path=$experiment_dir/trainer_chkpts/DDIM_epoch_4400.pt 

batch_size=100
batch_size_val=200
n_samples=20 # KEEP 20 — cohort-matched (frigatebird/fierce ran n_samples=20) → clean cross-arm test comparison. See K/n_samples note above.
test_subsample_n=null # 50 (NOT 51) to MATCH the actual cohort comparison runs (frigatebird ep4500 / fierce sweep were run at 50 → same 50 windows = comparable). null for full (~240 windows, ~3-5h; only if you re-run the WHOLE cohort at full).

# Inference-time DDIM stochasticity. Training used eta=0.2.
ddim_eta=0.2
sampling_timesteps=100

# ── RevIN σ scale correction ─────────────────────────────────────────
# lemming's saved .hydra/config.yaml has revin_sigma_correction=1.107108709910147
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
# Epoch parsed from the active checkpoint filename (NA for the best_model.pt
# symlink or names without an epoch_<N> / DDIM_epoch_<N> token).
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
# - dataset.test_subsample_n=50 = same 50 windows as the cohort runs (comparable);
#   null for the full ~240-window test set (only if re-running the whole cohort at full).
# - The diffusion baseline auto-loads architecture + diffusion config from
#   .hydra/config.yaml — including α=1.1071, base_channels=64, num_layers=2,
#   num_bottleneck_layers=2, cond gated/attention flags, exploration kwargs.
# - CUDA_VISIBLE_DEVICES=0 — lemming training FINISHED (freed GPU 0). Switch to 1
#   if you have launched another job on GPU 0.
# - GPU runtime at n_samples=20 + 100 DDIM steps on the SMALL 64ch denoiser:
#     test_subsample_n=50 → <1h; null (~240) → ~3-5h.
# - For tail-coverage (cov_95 / cov_99): bump n_samples to 50 / 200.
