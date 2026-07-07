#!/bin/bash
# Regenerate selector-rate-correlation plots for the merged U-GNN arm
# rigorous-quoll-131 from a saved checkpoint — WITHOUT retraining.
#
# Produces, under <OUTPUT_DIR>/eval_viz/epoch_<E>/:
#   - per-network:  selector_rate_correlation_d{density}_..._n{net}.pdf
#   - per-density aggregate (pooled over all networks of a density):
#                   selector_rate_correlation_aggregate_d{density}.pdf
#   - global aggregate (pooled over all densities):
#                   selector_rate_correlation_aggregate_global.pdf
#
# How it works: cli/test.py loads the run's .hydra/config.yaml + checkpoint,
# rebuilds the model/diffusion/task/loaders, then (with --selector-rate-correlation)
# runs a forward-only selector probe to collect per-network deep-level hard-mask
# frequencies and feeds them into the WRA evaluator, which correlates them against
# per-node ergodic rates. The aggregate accumulates across eval batches and flushes
# on the final batch, so per-density/global pools cover the full split.
#
# NOTE on epoch folder: cli/test.py parses the 1-based epoch from the checkpoint
# filename and stores it 0-based, so best_model_epoch_1600.pt -> eval_viz/epoch_1599.
#
# Usage:
#   bash scripts/wra/diffusion/rigorous_quoll_131/regenerate_selector_rate_correlation.sh
#   CHECKPOINT_EPOCH=2400 GPU_ID=1 bash .../regenerate_selector_rate_correlation.sh
#   EVAL_ON=test bash .../regenerate_selector_rate_correlation.sh

set -euo pipefail
export HYDRA_FULL_ERROR=1

# Env libstdc++ for matplotlib (CXXABI_1.3.15 / GLIBCXX_3.4.29).
export LD_LIBRARY_PATH="${HOME}/miniconda3/envs/graph-signal-diffusion/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Reduce allocator fragmentation for the sampling-heavy eval (n_samples cloning).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

GPU_ID="${GPU_ID:-0}"
CONDA_ENV="${CONDA_ENV:-graph-signal-diffusion}"
EVAL_ON="${EVAL_ON:-val}"                      # val | test | train-val (comma-sep ok)
# By default use the run's NATIVE eval config (the val loader already groups
# 50 samples/network; task.n_samples_per_input=1 → no extra cloning), which
# reproduces the training-time plots and fits in memory. Overriding this
# re-clones the already-grouped batch and will OOM (e.g. 100 → 800×100 graphs).
# Set N_SAMPLES_PER_INPUT only if you deliberately want a different clone factor.
N_SAMPLES_PER_INPUT="${N_SAMPLES_PER_INPUT:-}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-rigorous-quoll-131}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-outputs/wireless_resource_allocation-wra/ugnn_wra_v3_ds8_norm_act_head-ddim_wra-gdm_wra_medium-large_outdoor_all_density/${EXPERIMENT_NAME}}"
# Top-5 best-model leaderboard epochs: 1600 (#1) 2400 (#2) 1650 (#3) 2200 (#4) 3300 (#5).
CHECKPOINT_EPOCH="${CHECKPOINT_EPOCH:-1600}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${EXPERIMENT_DIR}/trainer_chkpts/best_models/best_model_epoch_${CHECKPOINT_EPOCH}.pt}"
CONFIG_DIR="${CONFIG_DIR:-${EXPERIMENT_DIR}/.hydra}"
OUTPUT_DIR="${OUTPUT_DIR:-${EXPERIMENT_DIR}/selector_rate_correlation_regen}"

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT_PATH}" >&2
  exit 1
fi
if [[ ! -f "${CONFIG_DIR}/config.yaml" ]]; then
  echo "Hydra config not found: ${CONFIG_DIR}/config.yaml" >&2
  exit 1
fi

cmd=(
  conda run -n "${CONDA_ENV}" python -m graph_signal_diffusion.cli.test
  --config-dir "${CONFIG_DIR}"
  --checkpoint "${CHECKPOINT_PATH}"
  --eval-on "${EVAL_ON}"
  --selector-rate-correlation
  --output-dir "${OUTPUT_DIR}"
)
if [[ -n "${N_SAMPLES_PER_INPUT}" ]]; then
  cmd+=(--n-samples-per-input "${N_SAMPLES_PER_INPUT}")
fi

echo "Regenerating selector-rate-correlation for ${EXPERIMENT_NAME} on GPU ${GPU_ID}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Eval split: ${EVAL_ON}  (n_samples_per_input=${N_SAMPLES_PER_INPUT:-<native>})"
echo "Output:     ${OUTPUT_DIR}/eval_viz/epoch_<E>/  (plots; <E> is 0-based)"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${cmd[@]}" "$@"
