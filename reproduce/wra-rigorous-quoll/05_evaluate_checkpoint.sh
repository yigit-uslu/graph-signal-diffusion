#!/bin/bash
# Stage 5 (PRIMARY RESULTS PATH) — Evaluate the shipped checkpoint to reproduce
# the paper's test metrics (test_summary.json).
#
# Runs the same held-out test evaluation the trainer ran at the end of training,
# on the SAME rank-1 checkpoint (best_model_epoch_1600.pt) and the SAME native
# 5:1:2 network-ID split (test = last 1/4 of networks per density). cli.test
# rebuilds the model / diffusion / task / loaders from the run's bundled
# .hydra/config.yaml (DDIM 100 steps, ddim_eta=0.2, model_cond_channels=2 — all
# auto-loaded), so only the checkpoint + split need to be named here.
#
# Point metrics reproduce essentially exactly; the channel-simulated rate metrics
# (sum_rate, fairness, rate percentiles) reproduce up to the seed-pinned DDIM
# sampler draw + channel realizations (num_channel_realizations=500). Reference:
# the run's test_summary.json (best_model_epoch_1600.pt): sum_rate_generated≈1122,
# fairness≈0.66. Runtime: ~10-40 min on one GPU (sampling-heavy).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/00_config.sh"
export HYDRA_FULL_ERROR=1

EVAL_ON="${EVAL_ON:-test}"
OUTPUT_DIR="${OUTPUT_DIR:-${EXPERIMENT_DIR}/reproduce_eval}"

# --- Preconditions ----------------------------------------------------------
if [ ! -f "$CHECKPOINT" ]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  echo "Restore the shipped checkpoint, or retrain via ./04_train_diffusion.sh." >&2
  exit 1
fi
if [ ! -f "$CONFIG_DIR/config.yaml" ]; then
  echo "Hydra config not found: $CONFIG_DIR/config.yaml" >&2
  exit 1
fi
if [ ! -d "$DATASET_ROOT" ]; then
  echo "Dataset root '$DATASET_ROOT' not found." >&2
  echo "The ~136 GB expert dataset is referenced, not bundled. Point DATASET_ROOT at" >&2
  echo "your copy, or symlink it into place:  ln -s /path/to/wra data/wra" >&2
  echo "(cli.test resolves the dataset at <repo>/data/wra — see checksums/dataset_manifest.txt)" >&2
  exit 1
fi

echo "Evaluating checkpoint (epoch ${CHECKPOINT_EPOCH}) on the '${EVAL_ON}' split..."
echo "  checkpoint: $CHECKPOINT"
echo "  config-dir: $CONFIG_DIR"
echo "  output-dir: $OUTPUT_DIR"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" $PYRUN -m graph_signal_diffusion.cli.test \
  --config-dir "$CONFIG_DIR" \
  --checkpoint "$CHECKPOINT" \
  --eval-on "$EVAL_ON" \
  --output-dir "$OUTPUT_DIR" \
  "$@"

echo ""
echo "Compare your metrics against the reference:"
echo "  $EXPERIMENT_DIR/test_summary.json   (sum_rate_generated≈1122.15, fairness≈0.6603)"
