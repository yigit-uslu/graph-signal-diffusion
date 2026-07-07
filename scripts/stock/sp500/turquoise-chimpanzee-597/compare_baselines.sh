#!/bin/bash
# Compare GRW vs turquoise-chimpanzee-597 on the SP500-cleaned test split.
#
# turquoise-chimpanzee-597 architecture (DS8-wide-uniform-128 + norm-act head,
# ~6M params):
#   IDENTICAL to fervent-cicada-140 (see that arm's launcher for full spec).
#   Same backbone, same training overrides, same head. The ONLY difference is
#   dataset.n_split_chunks=5 (vs F3's n_split_chunks=3).
#
# Why this matters — REGIME WARNING:
#   The 5-chunk interleaved split lands one val chunk directly on the COVID
#   crash window (2020-02-03 → 2020-04-07). F3's 3-chunk split avoids that
#   window entirely. The COVID slice carries the highest realized vol of the
#   dataset and inflates T5's val metrics dramatically:
#     - val_kurtosis_real:    T5 = 6.58   vs F3 = 3.40  (data property, not model)
#     - val_volclustering_real: T5 = 0.29 vs F3 = 0.07  (data property)
#     - val_eigval1_real:     T5 = 250    vs F3 = 163   (data property)
#     - val_return_crps:      T5 ≈ 1.70   vs F3 ≈ 1.10  (~55% gap, all from regime)
#     - val_return_mse:       T5 ≈ 14     vs F3 ≈ 4.5   (~3× gap, quadratic in tails)
#   This is NOT a model-quality difference — F3 and T5 are bit-identical on
#   train_loss (both 0.3343 at ep 999). See:
#     docs/agent_summaries/n_split_chunks_eval_gap_2026-05-25.md
#
# Task: stock_price_forecasting_v3_learned_ds8, n_split_chunks=5.
#
# Training COMPLETED 2026-05-25 (ep 999 / 1000, ~24h wallclock). Final tracker
# leaderboard (raw composite = val_loss + |val_loss_gap| + 0.3·val_return_crps):
#   ep 575 — rank 1  composite 0.3763  val_loss=0.359  |gap|=0.001   eig1_r=0.598
#   ep 675 — rank 2  composite 0.3796  val_loss=0.365  |gap|=0.001   eig1_r=0.655  ← best eig1_r
#   ep 300 — rank 3  composite 0.3794  val_loss=0.364  |gap|=0.003   eig1_r=0.609
#   ep 550 — rank 4  composite 0.3801  val_loss=0.362  |gap|=0.006   eig1_r=0.529
#   ep 400 — rank 5  composite 0.3804  val_loss=0.364  |gap|=0.003   eig1_r=0.436
#
# T5's composite is ~0.08 above F3's at every epoch because the val_return_crps
# term is regime-inflated. NOT a tracker malfunction — it's correctly identifying
# T5's val state as harder. Composite comparisons across F3 and T5 are
# meaningless; only intra-T5 comparisons are.
#
# Checkpoint selection — five operating points worth probing across the run:
#
#   ep 575 (default) — tracker rank-1 (best_model_epoch_575.pt):
#     val_loss=0.359  val_loss_gap=0.001  val_return_crps=1.684  eig1_r=0.598
#     The "generalizes best on val" pick under T5's split. Lowest |gap| in
#     run (after the brief ep 725 = 0.001 spike).
#
#   ep 675 — best eig1_r in run (best_model_epoch_675.pt):
#     eig1_r=0.6554 (highest), val_loss=0.365, val_return_crps=1.692
#     Structural-mode peak under the COVID-window regime.
#
#   ep 250 — best val_loss in run (DDIM_epoch_250.pt):
#     val_loss=0.3519 (lowest), val_return_crps=1.719
#     Early-train val_loss min; structural metrics still poor (eig1_r≈0.46).
#
#   ep 1000 — final state + best kurtosis_gap (DDIM_epoch_1000.pt):
#     val_loss=0.379  val_loss_gap=0.014  val_return_crps=1.717
#     val_kurtosis_gap=11.88 (BEST of run, set at ep 999!), eig1_r=0.564
#     Post-anneal hard-selection regime delivered T5's only late-train
#     improvement: tail fidelity. Selector at T_min=0.500.
#
#   ep 100 — best raw val_return_crps in run (DDIM_epoch_100.pt):
#     val_return_crps=1.6570 (lowest), val_loss=0.363, eig1_r=0.393
#     Very early-train pick. Worth probing to see whether the early
#     under-fit checkpoint produces sharper-but-undercommitted forecasts.
#
# Run-level summary vs cyber-doberman-15 (head ablation under T5 split):
#   - val_loss trajectory: T5 drifted up +0.018 in the post-anneal regime
#     (ep 700 → 999), the only run that did. F3 was stable. Likely COVID-window
#     overfitting under hard STE selection.
#   - val_kurtosis_gap: improved from 17.1 → 11.9 across ep 700 → 999 (best
#     ever at ep 999). The anneal helped tails for the harder regime.
#   - vs CD: head-to-head metric comparison is not valid (different splits).
#
# Full final diagnostics: docs/agent_summaries/norm_act_head_diagnostics_2026-05-25.md
#
# Eval setup: 50 samples per window (vs 10 in training), full DDIM sampling
# (100 steps, eta=0.2 matching training), AMP enabled.

export HYDRA_FULL_ERROR=1

experiment_name=turquoise-chimpanzee-597
experiment_dir=outputs/stock_price_forecasting_v3_learned_ds8-sp500_cleaned/ugnn_sp500_v3_ds8_norm_act_head-ddim-gdm_sp500_v3_learned_ds8/$experiment_name
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_575.pt

# Alternate operating points:
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_675.pt  # best eig1_r=0.6554
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_300.pt  # composite rank 3
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_550.pt  # composite rank 4
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_200.pt  # early structural snapshot
# checkpoint_path=$experiment_dir/trainer_chkpts/DDIM_epoch_250.pt                    # best val_loss=0.3519
# checkpoint_path=$experiment_dir/trainer_chkpts/DDIM_epoch_1000.pt                   # FINAL: post-anneal, best kurt_gap=11.88
checkpoint_path=$experiment_dir/trainer_chkpts/DDIM_epoch_950.pt                    # post-anneal mid-window
# checkpoint_path=$experiment_dir/trainer_chkpts/DDIM_epoch_100.pt                    # best raw CRPS=1.657 (early underfit)

batch_size=100
batch_size_val=200
n_samples=20 # 50 for full eval; 20 for quicker evaluation.
test_subsample_n=50

# Inference-time DDIM stochasticity. Training used eta=0.2 (nearly deterministic).
# For first read of this arm, keep eta=0.2 matching training.
ddim_eta=0.2
sampling_timesteps=100

# ── Post-hoc variance correction (eval-side) ─────────────────────────
# Default off for the first read on turquoise-chimpanzee-597. The COVID-window
# val regime may need MORE correction than F3 (T5's val_return_spread_std is
# 1.10 vs F3's 0.83), but a calibration pass on the val split should be done
# first to estimate alpha. See variance_correction_implementation_2026-05-20.md.
#
# To activate after calibration:
#   variance_correction_mode=scalar
#   variance_correction_alpha=1.5           # likely higher than F3's 1.28
#   variance_correction_pivot=ensemble_mean # or zero
variance_correction_mode=off            # off | scalar | per_horizon
variance_correction_alpha=1.0           # only used when mode != off
variance_correction_pivot=ensemble_mean # ensemble_mean | zero

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
    dataset.n_split_chunks=5 \
    dataset.standardize_target_in_x_for_revin=true \
    dataset.test_subsample_n=$test_subsample_n \
    'eval_splits={test: null}' \
    wandb.enabled=False

# Notes:
# - dataset.root points to the corr_0.7 graph variant (same as F3 and the
#   rest of the DS8-norm-act-head family).
# - dataset.future_window=5 (T=5) matches training; GRW forecasts T=5 too.
# - dataset.n_split_chunks=5 MUST match the training-time split (T5 trained
#   on 5 chunks). Do NOT compare absolute test metrics here against F3's
#   compare_baselines results — different test partitions (T5's test windows
#   span 5 narrow bands incl. one near the COVID crash, F3's span 3 wider
#   bands that avoid it).
# - dataset.standardize_target_in_x_for_revin=true must match training (RevIN).
# - n_samples_per_input=1 — compare_baselines rebuilds the loader per-baseline.
# - dataset.test_subsample_n=50 picks 50 windows uniformly via np.linspace
#   (deterministic, covers all 5 chunks). Set null for full (~235 windows).
# - For tail-coverage reads (cov_95 / cov_99): bump n_samples to 50 (cov_95
#   stable) or 200 (cov_99 stable).
# - The diffusion baseline auto-loads architecture from the experiment's
#   .hydra/config.yaml, so the task config passed here is mostly cosmetic.
# - CUDA_VISIBLE_DEVICES=1 — turquoise-chimpanzee-597 training ran on GPU 1;
#   reusing the same GPU for eval. The sibling fervent-cicada-140 launcher
#   uses GPU 0, so both compare_baselines runs can launch in parallel.
# - GPU runtime estimate at n_samples=50 + 100 DDIM steps on the 128ch
#   denoiser (per-step compute matches cyber-doberman):
#     test_subsample_n=50  → ~2-3h
#     test_subsample_n=100 → ~4-6h
#     test_subsample_n=null (full ~235) → ~10-15h
# - F3-vs-T5 head-to-head interpretation: NOT an architectural ablation.
#   Same architecture, same data, only n_split_chunks differs. Differences
#   in test metrics will reflect the regime asymmetry (T5 val/test includes
#   COVID-window slices, F3 does not). Use F3's compare_baselines result
#   as the cleaner forecast-quality read; treat T5's as a stress test on
#   high-volatility regimes.
