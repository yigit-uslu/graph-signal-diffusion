#!/bin/bash
# Generate diffusion overlay samples for the first 2 training networks.
#
# Runs compare_baselines in minimal mode (diffusion-only, train-val split,
# max_networks_per_split=2) to produce per-network NPZ files under
#   outputs/.../comparison/<date>/<time>/overlay/
# that can be fed to visualize_trace via overlay.generated_powers_path.
#
# Usage:
#   bash scripts/wra/diffusion/sophisticated_oarfish_9/generate_train_overlays.sh
#
#   # Override GPU or sample count:
#   GPU_ID=1 N_SAMPLES_PER_NETWORK=50 \
#     bash scripts/wra/diffusion/sophisticated_oarfish_9/generate_train_overlays.sh

set -euo pipefail

export HYDRA_FULL_ERROR=1
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

  # Diffusion-only — no need for FP/AP/WMMSE
  baselines_to_compare='[diffusion]'
  baselines.diffusion.checkpoint_path="${CHECKPOINT_PATH}"
  n_samples_per_network="${N_SAMPLES_PER_NETWORK}"

  # Train-val split (grouped batching), first 2 networks only.
  +dataset.max_networks_per_split=2
  '~eval_splits'
  '+eval_splits={train-val: null}'

  # Disable all optional phases
  r_min_sweep.enabled=false
  transferability_sweep.enabled=false
)

if [[ "$#" -gt 0 ]]; then
  cmd+=("$@")
fi

echo "Generating diffusion overlays for train networks 0 & 1 on GPU ${GPU_ID}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Samples per network: ${N_SAMPLES_PER_NETWORK}"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${cmd[@]}"
