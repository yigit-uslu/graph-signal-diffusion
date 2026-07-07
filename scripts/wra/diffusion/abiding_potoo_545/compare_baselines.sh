#!/bin/bash
# Compare WRA baselines with abiding-potoo-545 diffusion checkpoint.
# Multi-r_min experiment (medium-large outdoor low-density, all r_min).
#
# Default run (FP + AP + Diffusion on test split):
#   bash scripts/wra/diffusion/abiding_potoo_545/compare_baselines.sh
#
# Example overrides:
#   # Diffusion-only, validation split, 100 samples per network:
#   bash scripts/wra/diffusion/abiding_potoo_545/compare_baselines.sh \
#     baselines_to_compare='[diffusion]' \
#     eval_splits='{val: null}' \
#     n_samples_per_network=100
#
#   # Include WMMSE:
#   bash scripts/wra/diffusion/abiding_potoo_545/compare_baselines.sh \
#     baselines_to_compare='[fp,ap,wmmse,diffusion]'
#
#   # Enable r_min sweep:
#   bash scripts/wra/diffusion/abiding_potoo_545/compare_baselines.sh \
#     r_min_sweep.enabled=true

set -euo pipefail

export HYDRA_FULL_ERROR=1

# Ensure conda's libstdc++ is found (provides GLIBCXX_3.4.29 needed by numpy)
export LD_LIBRARY_PATH="${CONDA_PREFIX:-${HOME}/miniconda3/envs/graph-signal-diffusion}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

GPU_ID="${GPU_ID:-1}"
CONFIG_NAME="${CONFIG_NAME:-compare_baselines_wra}"
DATASET_CFG="${DATASET_CFG:-wra_medium-large_outdoor_low_density_all_rmin}"
N_SAMPLES_PER_NETWORK="${N_SAMPLES_PER_NETWORK:-50}"
CONDA_ENV="${CONDA_ENV:-graph-signal-diffusion}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-abiding-potoo-545}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-outputs/wireless_resource_allocation-wra/ugnn_wra_v3_ds4-ddim_wra-gdm_wra_medium-large_outdoor_low_density_all_rmin/${EXPERIMENT_NAME}}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${EXPERIMENT_DIR}/trainer_chkpts/best_models/best_model_epoch_2950.pt}"

if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT_PATH}" >&2
  exit 1
fi

cmd=(
  conda run -n "${CONDA_ENV}" python -m graph_signal_diffusion.cli.compare_baselines
  --config-name="${CONFIG_NAME}"
  dataset="${DATASET_CFG}"
  dataset@task.dataset="${DATASET_CFG}"
  baselines_to_compare='[fp,ap,diffusion]'
  baselines.diffusion.checkpoint_path="${CHECKPOINT_PATH}"
  n_samples_per_network="${N_SAMPLES_PER_NETWORK}"
  r_min_sweep.enabled=true
)

if [[ "$#" -gt 0 ]]; then
  cmd+=("$@")
fi

echo "Launching compare_baselines on GPU ${GPU_ID}"
echo "Checkpoint: ${CHECKPOINT_PATH}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${cmd[@]}"
