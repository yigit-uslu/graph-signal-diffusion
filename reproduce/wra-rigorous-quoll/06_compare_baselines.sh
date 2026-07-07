#!/bin/bash
# Stage 6 — BASELINES & PAPER FIGURES from the shipped checkpoint.
#
# Reproduces the paper's WRA figures by comparing the diffusion model against the
# full-power (FP) and adaptive-power (AP) baselines on the test split, plus the
# size/density transferability sweep. Thin wrapper over the canonical launchers
# (which carry the full option notes):
#   scripts/wra/diffusion/rigorous_quoll_131/compare_baselines.sh       (rate figures)
#   scripts/wra/diffusion/rigorous_quoll_131/visualize_network_panels.sh (anatomy panels)
#
# Uses the bundled rank-1 checkpoint (epoch 1600) by default. Needs the referenced
# dataset (data/wra) and — for the transferability sweep — the transfer channel
# cache (data/wra_channel_cache_transfer/), both part of the frozen artifact set.
#
# Usage:
#   ./06_compare_baselines.sh                 # baseline comparison + transferability
#   PANELS=true ./06_compare_baselines.sh     # also render the network-anatomy panels
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/00_config.sh"

CANON="scripts/wra/diffusion/rigorous_quoll_131"

if [ ! -f "$CHECKPOINT" ]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2; exit 1
fi

echo "== Baseline comparison (FP / AP / diffusion) + transferability — epoch ${CHECKPOINT_EPOCH} =="
CHECKPOINT_PATH="$CHECKPOINT" \
CHECKPOINT_EPOCH="$CHECKPOINT_EPOCH" \
GPU_ID="${CUDA_VISIBLE_DEVICES}" \
CONDA_ENV="$CONDA_ENV" \
  bash "$CANON/compare_baselines.sh" "$@"

if [ "${PANELS:-false}" = "true" ]; then
  echo "== Network-anatomy panels (PD-expert scenario views) =="
  CONDA_ENV="$CONDA_ENV" bash "$CANON/visualize_network_panels.sh"
fi

echo ""
echo "Stage 6 complete. Figures under:"
echo "  outputs/${TASK}-wra/comparison/<date>/<time>_${EXPERIMENT_NAME}_epoch-${CHECKPOINT_EPOCH}/"
echo "  (network panels, if PANELS=true) $EXPERIMENT_DIR/network_panels/"
