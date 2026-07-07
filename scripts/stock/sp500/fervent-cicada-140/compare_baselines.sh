#!/bin/bash
# Compare GRW vs fervent-cicada-140 on the SP500-cleaned test split.
#
# fervent-cicada-140 architecture (DS8-wide-uniform-128 + norm-act head, ~6M params):
#   Same backbone as cyber-doberman-15 (3-level uniform 128ch, gamma=[2,2,2],
#   num_layers=3, num_bottleneck_layers=3, max_gnn_stride=2, learned NodeSelector
#   STE T_min=0.5 linear anneal_ratio=0.90, dropout=0.10, RevIN, DDIM eta=0.2,
#   500 train / 100 sampling timesteps, corr_0.7 graph, bs=24, T=5).
#
#   ONLY architectural difference vs cyber-doberman-15: output head.
#     CD : UGNNDecoder.output_proj = nn.Linear(128, 1)             (default)
#     F3 : UGNNDecoder.output_proj = nn.Sequential(LayerNorm(128),
#                                                  SiLU,
#                                                  Linear(128, 1)) (zero-init)
#   See docs/agent_summaries/output_head_config_2026-05-24.md for the
#   OutputHeadConfig dataclass + helper + tests.
#
#   Training-budget difference: max_epochs=1000 (CD ran 2500). With the same
#   anneal_ratio=0.90, F3 hits T_min=0.500 at ep 902 (CD hits it at ep 2250).
#   F3 saw 97 epochs of fully-engaged STE hard-selection regime (ep 902-999).
#
# Task: stock_price_forecasting_v3_learned_ds8, n_split_chunks=3 (identical to
# cyber-doberman-15's split). Paired with turquoise-chimpanzee-597 which used
# n_split_chunks=5 (different val regime — see that arm's launcher).
#
# Training COMPLETED 2026-05-25 (ep 999 / 1000, ~24h wallclock). Final tracker
# leaderboard (raw composite = val_loss + |val_loss_gap| + 0.3·val_return_crps):
#   ep 475 — rank 1  composite 0.2970  val_loss=0.350  |gap|=0.0008  eig1_r=0.685  cov90=0.660
#   ep 575 — rank 2  composite 0.2971  val_loss=0.346  |gap|=0.006   eig1_r=0.600  cov90=0.660
#   ep 525 — rank 3  composite 0.2979  val_loss=0.349  |gap|=0.004   eig1_r=0.689  cov90=0.654
#   ep 375 — rank 4  composite 0.2998  val_loss=0.339  |gap|=0.015   eig1_r=0.707  cov90=0.645
#   ep 400 — rank 5  composite 0.2999  val_loss=0.343  |gap|=0.008   eig1_r=0.683  cov90=0.640
#
# Checkpoint selection — five operating points worth probing across the run:
#
#   ep 475 (default) — tracker rank-1, near-zero RevIN-space gap:
#     best_models/best_model_epoch_475.pt
#     val_loss=0.350  val_loss_gap=0.0008 (~6× better than CD's all-time best)
#     val_return_crps=1.109  eig1_r=0.685  kurtosis_gen=15.4
#     The norm-act-head's ONLY clean win over CD — the head's zero-init
#     achieves substantially tighter train/val agreement in RevIN-space.
#
#   ep 375 — best val_loss in run (best_model_epoch_375.pt):
#     val_loss=0.339 (lowest), val_return_crps=1.119, eig1_r=0.707
#     Mid-train pick; structural metrics still oscillating heavily.
#
#   ep 1000 — final state (DDIM_epoch_1000.pt):
#     val_loss=0.348  val_return_crps=1.098  eig1_r=0.687  kurt_gen=13.0
#     Post-anneal hard-selection regime; selector at T_min=0.500.
#     Useful for comparing to CD's ep 2499 final-state pick.
#
#   ep 600 — best kurtosis_gap (DDIM_epoch_600.pt):
#     val_kurtosis_gap=9.04 (best of run), eig1_r=0.706, val_loss=0.350
#     The structural-mode optimum.
#
#   ep 200 — best eig1_r (DDIM_epoch_200.pt):
#     eig1_r=0.836 (highest of run), val_loss=0.365, val_return_crps=1.115
#     Early-train factor-structure peak; noisy elsewhere.
#
# Run-level summary vs cyber-doberman-15 (head ablation, matched-epoch read):
#   - val_loss:        F3 best 0.339 vs CD best 0.340 — TIED
#   - val_return_crps: F3 best 1.090 vs CD best 1.095 — TIED (±0.005)
#   - val_eigval1_ratio: F3 sustained 0.65-0.75 vs CD sustained 0.72-0.77 — CD slight edge
#   - val_kurtosis_gen: F3 ~13-15 late-train vs CD ~7-10 late-train — CD WIN (~1.5-2×)
#   - val_loss_gap:    F3 min 0.0008 vs CD min 0.005 — F3 WIN (~6×)
#
# Full final diagnostics: docs/agent_summaries/norm_act_head_diagnostics_2026-05-25.md
#
# Eval setup: 50 samples per window (vs 10 in training), full DDIM sampling
# (100 steps, eta=0.2 matching training), AMP enabled.

export HYDRA_FULL_ERROR=1

experiment_name=fervent-cicada-140
experiment_dir=outputs/stock_price_forecasting_v3_learned_ds8-sp500_cleaned/ugnn_sp500_v3_ds8_norm_act_head-ddim-gdm_sp500_v3_learned_ds8/$experiment_name
checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_475.pt

# Alternate operating points:
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_375.pt  # best val_loss=0.339, eig1_r=0.707
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_525.pt  # composite rank 3
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_575.pt  # composite rank 2
# checkpoint_path=$experiment_dir/trainer_chkpts/DDIM_epoch_1000.pt                   # FINAL: post-anneal STE, val_loss=0.348
# checkpoint_path=$experiment_dir/trainer_chkpts/DDIM_epoch_950.pt                    # post-anneal mid-window, eig1_r=0.750
# checkpoint_path=$experiment_dir/trainer_chkpts/DDIM_epoch_600.pt                    # best kurtosis_gap=9.04
# checkpoint_path=$experiment_dir/trainer_chkpts/DDIM_epoch_200.pt                    # best eig1_r=0.836 (early)

batch_size=100
batch_size_val=200
n_samples=20 # 50 for full eval; 20 for quicker evaluation.
test_subsample_n=50

# Inference-time DDIM stochasticity. Training used eta=0.2 (nearly deterministic).
# For first read of this arm, keep eta=0.2 matching training.
ddim_eta=0.2
sampling_timesteps=100

# ── Symmetric RevIN σ scale correction ───────────────────────────────
# Replaces the asymmetric task-level variance_correction. Multiplies σ_w
# inside _compute_revin_stats by this factor so that both training-style
# normalize and sampling-style denormalize use the corrected scale. Effect
# on samples is roughly identical to the old variance_correction=scalar,
# alpha=1.22, pivot=ensemble_mean, but ALSO makes ELBO/NLL evaluation
# consistent with the corrected generative distribution (the old task-level
# version did not).
#
# α = 1 / E[σ_cond_w] on the F3 train set ≈ 1.22 (arithmetic-mean
# compensation). Geomean compensation would be α ≈ 1.28. See
# docs/agent_summaries/revin_sigma_correction_2026-05-26.md.
#
# IMPORTANT: do NOT combine with the old variance_correction CLI flags
# (they compound multiplicatively). The variance_correction block below
# is kept at mode=off for that reason.
revin_sigma_correction=1.22

# Legacy task-level variance correction — KEEP OFF when using
# revin_sigma_correction. Documented for reference only.
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
    ++baselines.diffusion.diffusion_overrides.revin_sigma_correction=$revin_sigma_correction \
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
# - dataset.root points to the corr_0.7 graph variant (same as cyber-doberman-15
#   and the rest of the DS8-norm-act-head family).
# - dataset.future_window=5 (T=5) matches training; GRW forecasts T=5 too.
# - dataset.n_split_chunks=3 MUST match the training-time split (F3 trained
#   on 3 chunks). The sibling arm turquoise-chimpanzee-597 used 5 chunks and
#   produces a different val regime — DO NOT cross-reference its test metrics.
# - dataset.standardize_target_in_x_for_revin=true must match training (RevIN).
# - n_samples_per_input=1 — compare_baselines rebuilds the loader per-baseline.
# - dataset.test_subsample_n=50 picks 50 windows uniformly via np.linspace
#   (deterministic, covers all 3 chunks). Set null for full (~234 windows).
# - For tail-coverage reads (cov_95 / cov_99): bump n_samples to 50 (cov_95
#   stable) or 200 (cov_99 stable).
# - The diffusion baseline auto-loads architecture from the experiment's
#   .hydra/config.yaml, so the task config passed here is mostly cosmetic.
# - CUDA_VISIBLE_DEVICES=0 — fervent-cicada-140 training ran on GPU 1; eval
#   uses GPU 0. The sibling turquoise-chimpanzee-597 launcher uses GPU 1,
#   so both compare_baselines runs can launch in parallel.
# - GPU runtime estimate at n_samples=50 + 100 DDIM steps on the 128ch
#   denoiser (per-step compute matches cyber-doberman):
#     test_subsample_n=50  → ~2-3h
#     test_subsample_n=100 → ~4-6h
#     test_subsample_n=null (full ~234) → ~10-15h
# - Architectural-comparison context: the norm-act-head is mildly net-negative
#   on tail-fidelity vs CD (kurtosis_gen ~1.5-2× higher at matched epochs) and
#   net-positive on RevIN-space gap (val_loss_gap min 0.0008 vs CD's 0.005).
#   The compare-baselines read will show whether this trade-off translates
#   into a test-set sample-quality win.
