#!/bin/bash
# Stage 8 (PAPER COMPARISON) — GRW vs. diffusion on the native 10-chunk test split.
#
# This is the source of the paper's baseline-comparison figures/tables: it
# evaluates the shipped checkpoint (best_model_epoch_4500.pt) against the
# geometric-random-walk (GRW) baseline on the SAME held-out test windows, so the
# two are directly comparable. Distinct from 05_evaluate_checkpoint.sh, which
# reproduces the trainer's own test_summary.json.
#
# Faithful to scripts/stock/sp500/sociable-frigatebird-619/compare_baselines.sh
# (the canonical launcher — see it for alternate checkpoints, the RevIN-alpha
# sensitivity probe, and the full leaderboard notes). Knobs below match it.
#
#   *** SPLIT DISCIPLINE ***  n_split_chunks=10 is the ONLY valid eval split for
#   this checkpoint. A different chunk count leaks ~80% of train windows into
#   "test." Do not change it.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/00_config.sh"

export HYDRA_FULL_ERROR=1

# --- Eval knobs (match the canonical launcher) ------------------------------
batch_size=100
batch_size_val=200
n_samples=100                 # tail coverage; 20 = quicker (family launchers use 20)
test_subsample_n=null         # full test (~240 windows)
ddim_eta=0.2                  # matches training
sampling_timesteps=100
# Legacy task-level variance correction — KEEP OFF (RevIN alpha auto-loads from ckpt).
variance_correction_mode=off
variance_correction_alpha=1.0
variance_correction_pivot=ensemble_mean

if [ ! -f "$CHECKPOINT" ]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  echo "Restore the shipped checkpoint, or retrain via ./04_train.sh." >&2
  exit 1
fi
if [ ! -d "$DATASET_ROOT/raw" ]; then
  echo "Missing cleaned data root '$DATASET_ROOT/raw'. Run ./03_clean.sh first." >&2
  exit 1
fi

# --- Output dir: comparison/<date>/<time>_<experiment>_epoch-<epoch> ---------
epoch_label=$(basename "$CHECKPOINT" | grep -oE 'epoch_[0-9]+' | grep -oE '[0-9]+' | head -1)
run_dir="$(dirname "$(dirname "$EXPERIMENT_DIR")")/comparison/$(date +%Y-%m-%d/%H-%M-%S)_${EXPERIMENT_NAME}_epoch-${epoch_label:-NA}"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" $PYRUN -m graph_signal_diffusion.cli.compare_baselines \
    --config-name compare_baselines_sp500 \
    hydra.run.dir="$run_dir" \
    task="$TASK" \
    baselines_to_compare='[grw,diffusion]' \
    baselines.diffusion.checkpoint_path="$CHECKPOINT" \
    baselines.diffusion.n_samples=$n_samples \
    ++baselines.diffusion.use_amp=true \
    ++baselines.diffusion.diffusion_overrides.ddim_eta=$ddim_eta \
    ++baselines.diffusion.diffusion_overrides.sampling_timesteps=$sampling_timesteps \
    ++baselines.diffusion.variance_correction.mode=$variance_correction_mode \
    ++baselines.diffusion.variance_correction.alpha=$variance_correction_alpha \
    ++baselines.diffusion.variance_correction.pivot=$variance_correction_pivot \
    baselines.grw.n_samples=$n_samples \
    dataset.root="$DATASET_ROOT" \
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

echo ""
echo "Comparison written under: $run_dir"
