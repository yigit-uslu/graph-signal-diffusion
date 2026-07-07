#!/bin/bash
# Stage 2 (regeneration / provenance) — PRIMAL-DUAL EXPERT TRAINING.
#
# Trains the primal-dual (PD) expert power-allocation policy for the 4 densities
# at the SINGLE reference r_min=0.6 (rigorous-quoll-131 is a fixed-r_min arm — cf.
# sophisticated-oarfish-9, which swept 5 r_min values × 4 densities = 20 runs).
# Here that is just 4 runs.
#
# Per-density epoch budget (higher density → more interference → larger budget),
# mirroring the r_min=0.6 column of the oarfish-9 matrix:
#   ultra-low 10K | low 10K | mid 20K | high 30K   (K = 1000 epochs)
#
# EXPENSIVE (many GPU-hours per density) and part of the regeneration path only.
# Each run writes a PD output dir containing collected_samples.npz + primal history,
# consumed by Stage 3. Not needed for the exact-numbers path (Stage 5).
#
# Usage:
#   ./02_train_pd.sh                  # all 4 densities at r_min=0.6
#   GPU_ID=1 ./02_train_pd.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/00_config.sh"
export HYDRA_FULL_ERROR=1

R_MIN="${R_MIN:-0.6}"
declare -A MAX_EPOCHS=( [ultra-low]=10000 [low]=10000 [mid]=20000 [high]=30000 )
DENSITIES=(ultra-low low mid high)

for density in "${DENSITIES[@]}"; do
  max_epochs="${MAX_EPOCHS[$density]}"
  echo "====== PD expert: ${density} density | r_min=${R_MIN} | max_epochs=${max_epochs} ======"
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" $PYRUN -m graph_signal_diffusion.cli.wra.train_pd \
    --config-name="pd_training/wra_medium-large_outdoor_${density}_density" \
    "training.r_min=${R_MIN}" \
    "training.max_epochs=${max_epochs}"
  echo "====== Done: ${density} density ======"; echo
done
echo "Stage 2 complete: 4 PD experts trained at r_min=${R_MIN}."
echo "Note the 4 PD output dirs (containing collected_samples.npz) for Stage 3."
