#!/bin/bash
# Stage 2: Train primal-dual experts for all 4 densities x 5 r_min values.
#
# Runs 20 PD training jobs sequentially on a single GPU.  Each (density, r_min)
# pair has a tuned max_epochs budget reflecting two factors:
#   - Higher density  → more interference → harder optimisation → larger base
#   - Higher r_min    → tighter feasibility  → 2x multiplier for r_min >= 0.7
#
# Epoch matrix (K = 1000 epochs):
#
#   density\r_min │  0.4   0.5   0.6   0.7   0.8
#   ──────────────┼──────────────────────────────
#   ultra-low     │  10K   10K   10K   20K   20K
#   low           │  10K   10K   10K   20K   20K
#   mid           │  20K   20K   20K   40K   40K
#   high          │  30K   30K   30K   60K   60K
#
# Usage:
#   bash scripts/wra/medium-large/sophisticated-oarfish-9/2_train_pd.sh
#
#   # Override GPU or start from a specific density:
#   GPU_ID=1 bash scripts/wra/medium-large/sophisticated-oarfish-9/2_train_pd.sh

set -euo pipefail
export HYDRA_FULL_ERROR=1

GPU_ID="${GPU_ID:-0}"

# ── Density → base max_epochs mapping ──
# Mirrors the values in the per-density pd_training/*.yaml configs.
declare -A BASE_EPOCHS=(
    [ultra-low]=10000
    [low]=10000
    [mid]=20000
    [high]=30000
)

DENSITIES=(ultra-low low mid high)
R_MIN_VALUES=(0.4 0.5 0.6 0.7 0.8)

# Higher r_min tightens the rate constraint → harder to converge → 2x budget.
get_max_epochs() {
    local base="$1"
    local r_min="$2"
    case "${r_min}" in
        0.7|0.8) echo $(( base * 2 )) ;;
        *)       echo "${base}" ;;
    esac
}

total=$(( ${#DENSITIES[@]} * ${#R_MIN_VALUES[@]} ))
counter=0

for density in "${DENSITIES[@]}"; do
    base="${BASE_EPOCHS[${density}]}"
    config="pd_training/wra_medium-large_outdoor_${density}_density"

    for r_min in "${R_MIN_VALUES[@]}"; do
        max_epochs=$(get_max_epochs "${base}" "${r_min}")
        counter=$(( counter + 1 ))
        echo "====== [${counter}/${total}] ${density} density | r_min=${r_min} | max_epochs=${max_epochs} ======"

        CUDA_VISIBLE_DEVICES="${GPU_ID}" python -m graph_signal_diffusion.cli.wra.train_pd \
            --config-name="${config}" \
            "training.r_min=${r_min}" \
            "training.max_epochs=${max_epochs}"

        echo "====== Done: ${density} density | r_min=${r_min} ======"
        echo
    done
done

echo "All 20 PD training runs complete."
