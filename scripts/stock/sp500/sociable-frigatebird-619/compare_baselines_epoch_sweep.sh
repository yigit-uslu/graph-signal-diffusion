#!/bin/bash
# Sequential EPOCH-SWEEP variant of compare_baselines.sh for sociable-frigatebird-619.
#
# Runs the SAME compare_baselines pipeline as compare_baselines.sh, but loops
# over a predefined list of training epochs, one checkpoint at a time, on the
# SAME GPU (the loop is inherently sequential). Each epoch's run lands in its
# own labeled output directory so nothing clobbers anything else.
#
# This is a LOOP WRAPPER ONLY — it does not change the per-epoch
# compare_baselines CLI invocation. All eval params (n_samples, DDIM eta,
# sampling steps, split, dataset root, etc.) are identical to the
# single-checkpoint launcher so epochs are compared apples-to-apples.
#
# Checkpoint source: the tracker's TOP-5 best_models picks (NOT uniform periodic
# snapshots) — run the strongest checkpoints first. best_models/ holds exactly
# these 5: ep 4500(r1) 1800(r2) 2800(r3) 4200(r4) 1600(r5). The diffusion
# baseline walks up from the checkpoint to load the experiment's .hydra/config.yaml
# (see baselines/diffusion/baseline.py:_find_experiment_dir), so RevIN alpha=1.1071,
# base_channels=64, num_layers=2, num_bottleneck_layers=2, cond gated/attention
# flags, exploration kwargs, etc. auto-load correctly for EVERY epoch — no
# per-epoch overrides.
#
#   To do a UNIFORM periodic sweep instead, point ckpt_dir/ckpt_prefix at the
#   DDIM snapshots and set a uniform epochs list (periodic DDIM_epoch_*.pt exist
#   every 50 epochs, 50..5000):
#     ckpt_dir=trainer_chkpts ; ckpt_prefix=DDIM_epoch ; epochs=(1000 2000 3000 4000 5000)
#
# ============================ SPLIT WARNING ============================
# sociable-frigatebird-619 was trained with dataset.n_split_chunks=10. This sweep
# evaluates on the NATIVE 10-chunk test split (the only valid split for this
# checkpoint). Results ARE directly comparable to spotted-catfish-602 and
# fierce-squirrel-222 (same dataset + 10-chunk split => identical test windows;
# this is the SMALL-vs-FULL capacity comparison — frigatebird + catfish small
# 1.88M, fierce-squirrel full 5.05M). Do NOT set dataset.n_split_chunks=3 to
# "compare" against the 3-chunk family — that leaks ~80% of those test windows
# into this model's TRAIN set. See compare_baselines.sh for the full rationale.
# =======================================================================
#
# Why best_models (not uniform) first: unlike fierce-squirrel (full capacity,
# peaked ep 450 then overfit), this SMALL model KEPT IMPROVING over its 5000-epoch
# run and did NOT overfit — the tracker rank-1 is LATE (ep 4500). The top-5 are
# scattered across the run (1600, 1800, 2800, 4200, 4500), so a uniform every-N
# grid would miss them; running the tracker picks directly is the cleanest first
# read of this arm's best operating points on the test split.
#
# Eval setup (matches compare_baselines.sh): 20 samples per window, full DDIM
# sampling (100 steps, eta=0.2), AMP on, test_subsample_n=50, GRW recomputed on
# the same 10-chunk windows each run. GRW is checkpoint-independent (so its
# numbers will be identical every run), but keeping it makes each epoch dir a
# self-contained GRW-vs-diffusion comparison; it is cheap vs 100-step DDIM.
#
# Runtime: at n_samples=20 + 100 DDIM steps on the SMALL 64ch denoiser, each
# checkpoint is <1h (test_subsample_n=50). A 5-checkpoint sweep is therefore
# ~3-5h, sequential on one GPU.
#
# Usage:
#   ./compare_baselines_epoch_sweep.sh           # run the full sweep
#   DRY_RUN=1 ./compare_baselines_epoch_sweep.sh # print each command, run nothing
#
# Failure policy: if one run exits non-zero (CUDA OOM, bad checkpoint, ...), it
# is logged and the sweep CONTINUES to the next checkpoint. A success / fail /
# skip summary is printed at the end.

export HYDRA_FULL_ERROR=1

# ── Sweep configuration (edit these) ─────────────────────────────────
# Checkpoint source (best_models picks by default; see header for the uniform
# DDIM alternative). Path = $experiment_dir/$ckpt_dir/${ckpt_prefix}_${epoch}.pt
ckpt_dir=trainer_chkpts/best_models             # or: trainer_chkpts  (for DDIM_epoch_*)
ckpt_prefix=best_model_epoch                    # or: DDIM_epoch
epochs=(1800 2800 4200 1600)               # tracker top-5: 4500(r1) 1800(r2) 2800(r3) 4200(r4) 1600(r5)
gpu=0                                           # same GPU for the whole sequential sweep
dry_run=${DRY_RUN:-0}                           # DRY_RUN=1 -> print commands, execute nothing

experiment_name=sociable-frigatebird-619
experiment_dir=outputs/stock_price_forecasting_v3_learned_ds8-sp500_cleaned/ugnn_sp500_v3_ds8_norm_act_head-ddim-gdm_sp500_v3_learned_ds8/$experiment_name

# Where each run's comparison output goes. One shared launch timestamp groups
# the whole sweep under comparison/<date>/, one leaf per epoch — matching the
# family single-run launchers' layout:
#   comparison/<date>/<time>_<experiment>_epoch-<epoch>
# comparison_root is derived from experiment_dir (= outputs/<task>-<dataset>/comparison),
# the same root the default hydra.run.dir produces.
run_date="$(date +%Y-%m-%d)"
run_time="$(date +%H-%M-%S)"
comparison_root="$(dirname "$(dirname "$experiment_dir")")/comparison"
run_prefix="$comparison_root/$run_date/${run_time}_${experiment_name}"

# ── Eval params (identical to compare_baselines.sh) ──────────────────
batch_size=100
batch_size_val=200
n_samples=20            # 50 for full eval; 20 for quicker eval. Family launchers use 20.
test_subsample_n=null
ddim_eta=0.2            # inference-time DDIM stochasticity; training used 0.2
sampling_timesteps=100

# Legacy task-level variance correction — KEEP OFF (RevIN sigma correction is
# auto-loaded from the checkpoint's .hydra/config.yaml).
variance_correction_mode=off            # off | scalar | per_horizon
variance_correction_alpha=1.0           # ignored when mode=off
variance_correction_pivot=ensemble_mean # ignored when mode=off

# ── Sequential sweep ─────────────────────────────────────────────────
declare -a succeeded=() failed=() skipped=()

echo "=================================================================="
echo "sociable-frigatebird-619 epoch sweep"
echo "  checkpoints: $ckpt_dir/${ckpt_prefix}_<e>.pt"
echo "  epochs:      ${epochs[*]}"
echo "  gpu:         $gpu"
echo "  output:      ${run_prefix}_epoch-<e>"
echo "  dry run:     $dry_run"
echo "=================================================================="

for epoch in "${epochs[@]}"; do
    checkpoint_path="$experiment_dir/$ckpt_dir/${ckpt_prefix}_${epoch}.pt"
    run_dir="${run_prefix}_epoch-${epoch}"

    if [[ ! -f "$checkpoint_path" ]]; then
        echo "[epoch $epoch] SKIP — checkpoint not found: $checkpoint_path"
        skipped+=("$epoch")
        continue
    fi

    # Build the per-epoch command as an array so DRY_RUN can print it verbatim
    # and the real run reuses exactly the same argv (no duplication / drift).
    cmd=(python -m graph_signal_diffusion.cli.compare_baselines
        --config-name compare_baselines_sp500
        task=stock_price_forecasting_v3_learned_ds8
        baselines_to_compare='[grw,diffusion]'
        baselines.diffusion.checkpoint_path="$checkpoint_path"
        baselines.diffusion.n_samples=$n_samples
        ++baselines.diffusion.use_amp=true
        ++baselines.diffusion.diffusion_overrides.ddim_eta=$ddim_eta
        ++baselines.diffusion.diffusion_overrides.sampling_timesteps=$sampling_timesteps
        ++baselines.diffusion.variance_correction.mode=$variance_correction_mode
        ++baselines.diffusion.variance_correction.alpha=$variance_correction_alpha
        ++baselines.diffusion.variance_correction.pivot=$variance_correction_pivot
        baselines.grw.n_samples=$n_samples
        dataset.root=data/sp500/cleaned_drop_incomplete_min_coverage_0.95_corr_0.7_sector_bonus_0.05
        dataset.batch_size=$batch_size
        dataset.batch_size_val=$batch_size_val
        dataset.n_samples_per_input=1
        dataset.past_window=20
        dataset.future_window=5
        dataset.n_split_chunks=10
        dataset.standardize_target_in_x_for_revin=true
        dataset.test_subsample_n=$test_subsample_n
        ++paper_figures.fig1.max_panels=20
        'eval_splits={test: null}'
        hydra.run.dir="$run_dir"
        wandb.enabled=False
    )

    echo "=================================================================="
    echo "[epoch $epoch] checkpoint: $checkpoint_path"
    echo "[epoch $epoch] output dir: $run_dir"
    echo "=================================================================="

    if [[ "$dry_run" == "1" ]]; then
        printf '[epoch %s] DRY RUN — would execute:\n  CUDA_VISIBLE_DEVICES=%s ' "$epoch" "$gpu"
        printf '%q ' "${cmd[@]}"
        printf '\n'
        succeeded+=("$epoch")
        continue
    fi

    mkdir -p "$run_dir"
    CUDA_VISIBLE_DEVICES=$gpu "${cmd[@]}" 2>&1 | tee "$run_dir/launcher.log"
    status=${PIPESTATUS[0]}   # python's exit code, not tee's

    if [[ $status -eq 0 ]]; then
        echo "[epoch $epoch] OK"
        succeeded+=("$epoch")
    else
        echo "[epoch $epoch] FAILED (exit $status) — see $run_dir/launcher.log"
        failed+=("$epoch")
    fi
done

echo
echo "===================== EPOCH SWEEP SUMMARY ========================"
echo "Output dirs: ${run_prefix}_epoch-<epoch>"
echo "Succeeded (${#succeeded[@]}): ${succeeded[*]:-none}"
echo "Failed    (${#failed[@]}): ${failed[*]:-none}"
echo "Skipped   (${#skipped[@]}): ${skipped[*]:-none}"
echo "=================================================================="

# Exit non-zero if any run failed, so callers / CI can detect partial sweeps.
[[ ${#failed[@]} -eq 0 ]]
