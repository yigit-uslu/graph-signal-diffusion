#!/bin/bash
# Compare WRA baselines with sophisticated-oarfish-9 diffusion checkpoint.
# Multi-density, multi-r_min experiment (medium-large outdoor, all density, all r_min).
#
# Default run (FP + AP + Diffusion on test split, r_min sweep + transferability):
#   bash scripts/wra/diffusion/sophisticated_oarfish_9/compare_baselines.sh
#
# Example overrides:
#   # Diffusion-only, validation split, 100 samples per network:
#   bash scripts/wra/diffusion/sophisticated_oarfish_9/compare_baselines.sh \
#     baselines_to_compare='[diffusion]' \
#     eval_splits='{val: null}' \
#     n_samples_per_network=100
#
#   # Include WMMSE:
#   bash scripts/wra/diffusion/sophisticated_oarfish_9/compare_baselines.sh \
#     baselines_to_compare='[fp,ap,wmmse,diffusion]'
#
#   # Disable r_min sweep or transferability:
#   bash scripts/wra/diffusion/sophisticated_oarfish_9/compare_baselines.sh \
#     r_min_sweep.enabled=false
#   bash scripts/wra/diffusion/sophisticated_oarfish_9/compare_baselines.sh \
#     transferability_sweep.enabled=false

set -euo pipefail

export HYDRA_FULL_ERROR=1

# Ensure the *env* libstdc++ is found (provides CXXABI_1.3.15 needed by matplotlib).
# CONDA_PREFIX may point to the base env when running outside an activated env,
# so we hardcode the target env path instead.
export LD_LIBRARY_PATH="${HOME}/miniconda3/envs/graph-signal-diffusion/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

GPU_ID="${GPU_ID:-0}"
CONFIG_NAME="${CONFIG_NAME:-compare_baselines_wra}"
DATASET_CFG="${DATASET_CFG:-wra_medium-large_outdoor_all_density_all_rmin}"
N_SAMPLES_PER_NETWORK="${N_SAMPLES_PER_NETWORK:-100}"
CONDA_ENV="${CONDA_ENV:-graph-signal-diffusion}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-sophisticated-oarfish-9}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-outputs/wireless_resource_allocation-wra/ugnn_wra_v3_ds4-ddim_wra-gdm_wra_medium-large_outdoor_all_density_all_rmin/${EXPERIMENT_NAME}}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${EXPERIMENT_DIR}/trainer_chkpts/best_models/best_model_epoch_2400.pt}"

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

  # ── r_min sweep (all 4 densities) ──
  r_min_sweep.enabled=true
  '+r_min_sweep.reference_datasets=[medium-large_outdoor_ultra-low_density/wrpc_v1_primal_history_k200_h0dd7afd393f9,medium-large_outdoor_low_density/wrpc_v1_primal_history_k200_h43d4a26a4203,medium-large_outdoor_mid_density/wrpc_v1_primal_history_k200_hc1f8f7a25432,medium-large_outdoor_high_density/wrpc_v1_primal_history_k200_ha6c7c432ee13]'

  # ── Size/density transferability sweep ──
  transferability_sweep.enabled=true
)

if [[ "$#" -gt 0 ]]; then
  cmd+=("$@")
fi

echo "Launching compare_baselines on GPU ${GPU_ID}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Phases: baseline comparison + r_min sweep (4 densities) + transferability (medium + large)"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${cmd[@]}"
