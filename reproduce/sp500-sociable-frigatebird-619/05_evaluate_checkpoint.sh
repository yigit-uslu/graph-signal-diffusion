#!/bin/bash
# Stage 7 (PRIMARY RESULTS PATH) — Evaluate the shipped checkpoint to reproduce
# the paper's test metrics (test_summary.json).
#
# This runs the same held-out test evaluation the trainer ran at the end of the
# 5000-epoch run, on the SAME checkpoint (best_model_epoch_4500.pt) and the SAME
# native 10-chunk test split. It reproduces the exact numbers up to DDIM sampler
# stochasticity (eta=0.2), which is pinned by seed=0 and n_samples below.
#
#   Match to test_summary.json (the reference in the run's output dir):
#     - n_samples_per_input = 10   (the trainer's test-eval ensemble size)
#     - sampling_timesteps  = 100, ddim_eta = 0.2   (auto-loaded from checkpoint)
#     - n_split_chunks      = 10   (NATIVE split — the ONLY valid eval split;
#                                   other chunk counts leak train into test)
#     - full test set (test_subsample_n=null, eval_splits.test=null)
#
# Point metrics (MAE/RMSE/MSE, price_*) reproduce essentially exactly; the
# probabilistic metrics (return_crps, mis_*, coverage_*) reproduce up to the
# seed-pinned sampler draw. Runtime: ~1-3h on one GPU (small 64ch denoiser,
# 100 DDIM steps, ~240 windows x 10 samples).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/00_config.sh"

export HYDRA_FULL_ERROR=1

if [ ! -f "$CHECKPOINT" ]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  echo "Either restore the shipped checkpoint, or retrain via ./04_train.sh." >&2
  exit 1
fi
if [ ! -d "$DATASET_ROOT/raw" ]; then
  echo "Missing cleaned data root '$DATASET_ROOT/raw'. Run ./03_clean.sh first." >&2
  exit 1
fi

# The diffusion baseline auto-loads architecture + diffusion config (base_channels,
# num_layers, ddim_eta=0.2, sampling_timesteps=100, RevIN alpha=1.1071, cond
# gated/attention flags, ...) from the checkpoint's .hydra/config.yaml — so only
# the dataset/split/ensemble knobs need to be specified here.
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" $PYRUN -m graph_signal_diffusion.cli.evaluate \
    seed=0 \
    baseline=diffusion \
    task="$TASK" \
    dataset=sp500_cleaned \
    dataset.root="$DATASET_ROOT" \
    checkpoint_path="$CHECKPOINT" \
    baseline.n_samples=10 \
    ++baseline.use_amp=true \
    dataset.batch_size=100 \
    dataset.batch_size_val=200 \
    dataset.past_window=20 \
    dataset.future_window=5 \
    dataset.n_split_chunks=10 \
    dataset.standardize_target_in_x_for_revin=true \
    dataset.test_subsample_n=null \
    'eval_splits={test: null}' \
    wandb.enabled=false

echo ""
echo "Compare your metrics against the reference:"
echo "  $EXPERIMENT_DIR/test_summary.json"
echo ""
echo "For the GRW-vs-diffusion paper comparison (n_samples=100, tail coverage), see:"
echo "  scripts/stock/sp500/sociable-frigatebird-619/compare_baselines.sh"
