#!/bin/bash
# Compare GRW vs cyber-doberman-15 on the SP500-cleaned test split.
#
# cyber-doberman-15 architecture (DS8-wide-uniform-128, ~6M params):
#   past_window=20, future_window=5, 3-chunk interleaved split,
#   3-level U-GNN denoiser at base_channels=128, channel_multipliers=[1,1,1]
#   (uniform 128ch on all 3 levels), gamma=[2,2,2] (8x bottleneck compression
#   at 58 nodes), max_gnn_stride=2, num_bottleneck_layers=3,
#   gnn_config.num_layers=3 with temporal_mixer dilations=[1,1,1]
#   (RF=7 covering T=5), learned NodeSelector (STE, T_min=0.5, linear anneal
#   anneal_ratio=0.90), dropout=0.10, RevIN, DDIM (eta=0.2, 500 train / 100
#   sampling timesteps), per-layer temporal mixer + self-attention, cross-
#   attention cond fusion, corr_0.7 graph, bs=24. Task=
#   stock_price_forecasting_v3_learned_ds8.
#
# Controls:
#   - tomato-mule-215    (DS8 yaml; gamma=[1,2,2,2], base=64, num_layers=2; ~1.4M params)
#   - outstanding-fox-517 (gamma=[1,1,1,1] stride; base=64; uniform-width baseline)
#   - infrared-coyote-480 (gamma=[2,2,2] pyramid [1,2,4] @ base=32; same graph
#       schedule + bottleneck width but pyramid capacity allocation)
#
# Training COMPLETED 2026-05-23 (ep 2499 / 2500, ~36h wallclock). Final
# tracker leaderboard — composite rank-1 stayed at ep 475 throughout the
# late-train regime because every late checkpoint had |gap| > 0.04 vs
# ep 475's 0.005:
#   ep 475 — rank 1  composite 0.2981  val_loss=0.3514  |gap|=0.005  eig1_r=0.659  cov90=0.674
#   ep 375 — rank 2  composite 0.2999  val_loss=0.3396  |gap|=0.017  eig1_r=0.668  cov90=0.695
#   ep 150 — rank 3  composite 0.3009  val_loss=0.3446  |gap|=0.013  eig1_r=0.555  cov90=0.630
#   ep 400 — rank 4  composite 0.3011  val_loss=0.3442  |gap|=0.011  eig1_r=0.602  cov90=0.643
#   ep 575 — rank 5  composite 0.3012  val_loss=0.3491  |gap|=0.012  eig1_r=0.686  cov90=0.688
#
# Checkpoint selection — three structural / quality optima emerged across the
# full 2500-epoch trajectory (none of these are best_models picks; they are
# save_every_n_epochs=50 snapshots in trainer_chkpts/ root):
#
#   ep 700 (default) — cov_90 + spread optimum (DDIM_epoch_700.pt):
#     val_loss=0.348  eig1_r=0.717  cov90=0.698 (highest in run)
#     spread=1.46 (highest)  |gap|=0.033  kurt=9.6
#     Best-balanced checkpoint: above outstanding-fox on both eig1_r AND
#     cov_90, widest predictive distribution.
#
#   ep 1650 — structural peak (DDIM_epoch_1650.pt):
#     val_loss=0.364  eig1_r=0.774 (highest in run)  cov90=0.665
#     spread=1.37  |gap|=0.049  kurt=8.0
#     Highest factor-structure alignment, but narrower spread + higher |gap|.
#
#   ep 2499 — final state (DDIM_epoch_2499.pt):
#     val_loss=0.357 (best in 1500+ epochs)  eig1_r=0.761  cov90=0.675
#     spread=1.40  |gap|=0.048  kurt=7.4
#     Post-anneal hard-selection regime; selector at T_min=0.50. Strong
#     structural metrics + recovered val_loss after mid-train overfit dip.
#
#   ep 475 — tracker rank-1 (best_model_epoch_475.pt):
#     val_loss=0.351  eig1_r=0.659  cov90=0.674  |gap|=0.005 (lowest)
#     The "generalizes best on val" pick; trades structural strength for
#     train/val agreement.
#
# Run-level summary vs controls:
#   - eig1_ratio:   cyber-doberman 0.72–0.77 sustained late-train (WIN;
#                   outstanding-fox 0.671, tomato-mule 0.596)
#   - val_loss:     cyber-doberman min 0.340 at ep 375; outstanding-fox 0.335
#                   (LOSS — capacity didn't lower val_loss further)
#   - cov_90:       cyber-doberman peak 0.698 (small WIN; outstanding-fox 0.685)
#   - kurtosis_gen: cyber-doberman 7–9 (BIG WIN; tomato-mule ≈21, much closer
#                   to real-data ≈3–7)
#
# Eval setup: 20 samples per window (vs 10 in training), full DDIM sampling
# (100 steps, eta=0.2 matching training), AMP enabled.

export HYDRA_FULL_ERROR=1

experiment_name=cyber-doberman-15
experiment_dir=outputs/stock_price_forecasting_v3_learned_ds8-sp500_cleaned/ugnn_sp500_v3_ds8-ddim-gdm_sp500_v3_learned_ds8/$experiment_name
checkpoint_path=$experiment_dir/trainer_chkpts/DDIM_epoch_50.pt
# checkpoint_path=$experiment_dir/trainer_chkpts/DDIM_epoch_800.pt
# checkpoint_path=$experiment_dir/trainer_chkpts/DDIM_epoch_1200.pt 

# Alternate operating points:
# checkpoint_path=$experiment_dir/trainer_chkpts/DDIM_epoch_2500.pt           # FINAL state: best late val_loss=0.357 + eig1_r=0.761, |gap|=0.048
# checkpoint_path=$experiment_dir/trainer_chkpts/DDIM_epoch_1650.pt           # structural peak: eig1_r=0.774 (highest in run), val_loss=0.364
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_475.pt  # tracker rank-1: |gap|=0.005, weaker structural (eig1_r=0.659)
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_375.pt  # tracker rank-2: val_loss min (0.340), eig1_r=0.668
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_575.pt  # tracker rank-5: latest pick, eig1_r=0.686
# checkpoint_path=$experiment_dir/trainer_chkpts/DDIM_epoch_200.pt            # structural extreme (early): eig1_r=0.825, val_loss=0.364 (higher)

batch_size=100
batch_size_val=100
n_samples=50 # 50 for full eval; 20 for quicker evaluation.
test_subsample_n=50

# Inference-time DDIM stochasticity. Training used eta=0.2 (nearly deterministic).
# For first read of this arm, keep eta=0.2 matching training. The tomato-mule
# launcher's eta=0.5 was an exploratory setting; we don't yet know how this
# 6M model responds to higher eval-time stochasticity.
ddim_eta=0.2
sampling_timesteps=100

# ── Post-hoc variance correction (eval-side) ─────────────────────────
# Default off for the first read on cyber-doberman-15. Val-time spread at
# ep 700 (1.46) is meaningfully larger than tomato-mule's uncorrected ~1.40,
# so this model may need less / no correction.
#
# To activate (see variance_correction_implementation_2026-05-20.md):
#   variance_correction_mode=scalar
#   variance_correction_alpha=1.28          # marginal-match (pivot=zero)
#                       or =1.5             # coverage-match (pivot=ensemble_mean)
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
    dataset.n_split_chunks=3 \
    dataset.standardize_target_in_x_for_revin=true \
    dataset.test_subsample_n=$test_subsample_n \
    'eval_splits={test: null}' \
    wandb.enabled=False

# Notes:
# - dataset.root points to the corr_0.7 graph variant (same as all DS8 arms).
# - dataset.future_window=5 (T=5) matches training; GRW forecasts T=5 too.
# - dataset.n_split_chunks=3 must match the training-time split.
# - dataset.standardize_target_in_x_for_revin=true must match training (RevIN).
# - n_samples_per_input=1 — compare_baselines rebuilds the loader per-baseline.
# - dataset.test_subsample_n=50 picks 50 windows uniformly via np.linspace
#   (deterministic, covers all 3 chunks). Set null for full (~234 windows).
# - For tail-coverage reads (cov_95 / cov_99): bump n_samples to 50 (cov_95
#   stable) or 200 (cov_99 stable).
# - The diffusion baseline auto-loads architecture from the experiment's
#   .hydra/config.yaml, so the task config passed here is mostly cosmetic.
# - CUDA_VISIBLE_DEVICES=1 — cyber-doberman-15 training is on GPU 0; eval
#   uses GPU 1. If infrared-coyote-480 is still training on GPU 1, this
#   will collide — check `nvidia-smi` before launch.
# - GPU runtime estimate at n_samples=20 + 100 DDIM steps on the wider 128ch
#   denoiser (per-step compute ~5× tomato-mule):
#     test_subsample_n=50  → ~2-3h    (vs tomato-mule's ~45min)
#     test_subsample_n=100 → ~4-6h
#     test_subsample_n=null (full ~234) → ~10-15h
# - Training COMPLETED 2026-05-23 (ep 2499/2500). Final tracker leaderboard
#   shown in the header is the authoritative end state. The structural mode
#   crystallized late-train (sustained eig1_r ≈ 0.74–0.77 from ep 850 onward),
#   but the composite tracker's |gap| penalty kept rank-1 at ep 475 throughout
#   — that's expected behavior, not a bug. Manual probing of ep 700 / 2499 /
#   1650 captures the structural axis the tracker can't promote.
