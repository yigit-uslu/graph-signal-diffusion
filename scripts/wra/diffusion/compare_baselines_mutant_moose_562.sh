#!/bin/bash
# Compare WRA baselines on GPU 1 with mutant-moose-562 diffusion checkpoint.
#
# Default run (FP + AP + Diffusion on test split):
#   bash scripts/wra/diffusion/compare_baselines_mutant_moose_562.sh
#
# Example overrides:
#   # Diffusion-only, validation split, 100 samples per network:
#   bash scripts/wra/diffusion/compare_baselines_mutant_moose_562.sh \
#     baselines_to_compare='[diffusion]' \
#     eval_splits='{val: null}' \
#     n_samples_per_network=100
#
#   # Re-add WMMSE and ablate diffusion sampler:
#   bash scripts/wra/diffusion/compare_baselines_mutant_moose_562.sh \
#     baselines_to_compare='[fp,ap,wmmse,diffusion]' \
#     baselines.diffusion.diffusion_overrides.sampling_timesteps=50 \
#     baselines.diffusion.diffusion_overrides.ddim_eta=0.0

set -euo pipefail

export HYDRA_FULL_ERROR=1

GPU_ID="${GPU_ID:-1}"
CONFIG_NAME="${CONFIG_NAME:-compare_baselines_wra}"
DATASET_CFG="${DATASET_CFG:-wra_large_outdoor_low_density}"
N_SAMPLES_PER_NETWORK="${N_SAMPLES_PER_NETWORK:-50}"
CONDA_ENV="${CONDA_ENV:-torch_env}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-mutant-moose-562}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-outputs/wireless_resource_allocation-wra/ugnn_wra_v3_ds4-ddim_wra-gdm_wra_large_outdoor_low_density/${EXPERIMENT_NAME}}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${EXPERIMENT_DIR}/trainer_chkpts/best_models/best_model_epoch_1500.pt}"

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
)

if [[ "$#" -gt 0 ]]; then
  cmd+=("$@")
fi

echo "Launching compare_baselines on GPU ${GPU_ID}"
echo "Checkpoint: ${CHECKPOINT_PATH}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${cmd[@]}"
