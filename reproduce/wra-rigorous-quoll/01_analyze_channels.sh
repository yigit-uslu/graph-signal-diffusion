#!/bin/bash
# Stage 1 (regeneration / provenance) — CHANNEL & SCENARIO GENERATION.
#
# Generates the wireless networks and channel statistics for the 4 medium-large
# outdoor density scenarios (ultra-low / low / mid / high) that rigorous-quoll-131
# trains on. Produces per-scenario deployment visualizations and full-power rate
# CDFs (used to guide r_min selection).
#
# This is the FIRST stage of regenerating the expert dataset from scratch. For the
# primary (exact-numbers) reproduction you do NOT need to run this — point
# DATASET_ROOT at the frozen ~136 GB dataset and go straight to
# 05_evaluate_checkpoint.sh. Channel generation is seeded (seed=42 in the configs)
# but is part of the expensive regeneration path.
#
# Usage:
#   ./01_analyze_channels.sh          # 32 networks/scenario (paper setting)
#   ./01_analyze_channels.sh 8        # override networks/scenario (quick smoke)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/00_config.sh"
export HYDRA_FULL_ERROR=1

DENSITIES=(ultra-low low mid high)

NUM_NETWORKS_OVERRIDE="${1:-}"
OVERRIDE_ARGS=()
if [[ -n "$NUM_NETWORKS_OVERRIDE" ]]; then
  OVERRIDE_ARGS=("dataset.num_networks=${NUM_NETWORKS_OVERRIDE}")
  echo "Overriding dataset.num_networks=${NUM_NETWORKS_OVERRIDE}"
fi

for density in "${DENSITIES[@]}"; do
  echo "====== Analyzing channels: medium-large outdoor ${density} density ======"
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" $PYRUN -m graph_signal_diffusion.cli.wra.analyze_channels \
    --config-name="channel_analysis/wra_medium-large_outdoor_${density}_density" \
    ${OVERRIDE_ARGS[@]+"${OVERRIDE_ARGS[@]}"}
  echo "====== Done: ${density} density ======"; echo
done
echo "Stage 1 complete: channels analyzed for all 4 densities."
