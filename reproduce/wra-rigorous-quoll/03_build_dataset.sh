#!/bin/bash
# Stage 3 (regeneration / provenance) — DIFFUSION-DATASET BUILD (sample collection).
#
# Converts each density's PD primal history (from Stage 2) into the raw/ dataset
# format consumed by WRADataset.process(). rigorous-quoll-131 is single-r_min, so
# every density is built at r_min=0.6 with H_instantaneous ENABLED (each density is
# its own reference — no cross-r_min symlinking, unlike the oarfish-9 sweep).
#
# Collection settings (paper): primal_history source, window_size=1000,
# refine_feasible_subset=true, target_samples_per_network=200
#   -> 4 densities × 32 networks × 200 = 25,600 samples (5:1:2 split).
#
# The resulting content-addressed sub-dataset dirs are pinned in
#   src/graph_signal_diffusion/conf/dataset/wra_medium-large_outdoor_all_density.yaml
# (wrpc_v1_primal_history_k200_h<hash>) and listed in checksums/dataset_manifest.txt.
#
# INPUT: the 4 PD output dirs from Stage 2. They are run-specific (content-addressed
# + dated), so pass them via env vars (a dir containing collected_samples.npz):
#   PD_RUN_ULTRA_LOW=outputs/wra_medium-large_outdoor_ultra-low_density/wrpd_..._r0.6_.../<date>/<time> \
#   PD_RUN_LOW=... PD_RUN_MID=... PD_RUN_HIGH=... ./03_build_dataset.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/00_config.sh"
export HYDRA_FULL_ERROR=1

R_MIN="${R_MIN:-0.6}"
COMMON_ARGS=(
  collection.sample_source=primal_history
  collection.primal_history.window_size=1000
  collection.primal_history.refine_feasible_subset=true
  collection.target_samples_per_network=200
  output.h_instantaneous.enabled=true
)

# density | env var holding its PD output dir
build_one() {
  local density="$1" input_dir="$2"
  if [[ -z "$input_dir" ]]; then
    echo "ERROR: PD output dir for '${density}' not set (see header for the PD_RUN_* env vars)." >&2
    exit 1
  fi
  if [[ ! -d "$input_dir" ]]; then
    echo "ERROR: PD output dir not found: $input_dir" >&2; exit 1
  fi
  echo "====== Build dataset: ${density} density | r_min=${R_MIN} ======"
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" $PYRUN -m graph_signal_diffusion.cli.wra.build_diffusion_dataset \
    --config-name="pd_collection/wra_medium-large_outdoor_${density}_density" \
    input_dir="$input_dir" \
    "training.r_min=${R_MIN}" \
    "${COMMON_ARGS[@]}"
  echo "====== Done: ${density} density ======"; echo
}

build_one ultra-low "${PD_RUN_ULTRA_LOW:-}"
build_one low        "${PD_RUN_LOW:-}"
build_one mid        "${PD_RUN_MID:-}"
build_one high       "${PD_RUN_HIGH:-}"

echo "Stage 3 complete: 4 sub-datasets built at r_min=${R_MIN}."
echo "Verify the content-addressed paths against checksums/dataset_manifest.txt."
