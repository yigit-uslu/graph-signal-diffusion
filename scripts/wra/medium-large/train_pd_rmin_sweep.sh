#!/bin/bash
# Sweep r_min values for the medium-large outdoor low_density and mid_density scenarios.
# Runs sequentially on GPU 0 with r_min in {0.4, 0.5, 0.7, 0.8}.
# Usage: bash scripts/wra/medium-large/train_pd_rmin_sweep.sh [num_networks_override]

set -euo pipefail
export HYDRA_FULL_ERROR=1

cleanup() {
    jobs -pr | xargs -r kill -TERM
}
trap cleanup INT TERM EXIT

NUM_NETWORKS_OVERRIDE="${1:-}"
OVERRIDE_ARGS=()
if [[ -n "${NUM_NETWORKS_OVERRIDE}" ]]; then
    OVERRIDE_ARGS=("dataset.num_networks=${NUM_NETWORKS_OVERRIDE}")
    echo "Overriding dataset.num_networks=${NUM_NETWORKS_OVERRIDE} for all runs."
else
    echo "Using dataset.num_networks default from scenario config."
fi

R_MIN_VALUES=(0.4 0.5 0.7 0.8)

# Higher r_min values are harder to optimise — allow more epochs.
get_max_epochs() {
    local r_min="$1"
    local base="$2"
    case "${r_min}" in
        0.7|0.8) echo $(( base * 2 )) ;;
        *)       echo "${base}" ;;
    esac
}

# # ── Low-density sweep (base 10000 epochs) ──
# echo "############################################"
# echo "# Scenario: low_density  (base 10000 eps) #"
# echo "############################################"
# for R_MIN in "${R_MIN_VALUES[@]}"; do
#     MAX_EPOCHS=$(get_max_epochs "${R_MIN}" 10000)
#     echo "=========================================="
#     echo "Starting r_min=${R_MIN}  max_epochs=${MAX_EPOCHS}"
#     echo "=========================================="
#     CUDA_VISIBLE_DEVICES=0 python -m graph_signal_diffusion.cli.wra.train_pd \
#         --config-name=pd_training/wra_medium-large_outdoor_low_density \
#         "training.r_min=${R_MIN}" \
#         "training.max_epochs=${MAX_EPOCHS}" \
#         "${OVERRIDE_ARGS[@]}"
#     echo "Finished r_min=${R_MIN}"
# done

# # ── Mid-density sweep (base 20000 epochs) ──
# echo "############################################"
# echo "# Scenario: mid_density  (base 20000 eps) #"
# echo "############################################"
# for R_MIN in "${R_MIN_VALUES[@]}"; do
#     MAX_EPOCHS=$(get_max_epochs "${R_MIN}" 20000)
#     echo "=========================================="
#     echo "Starting r_min=${R_MIN}  max_epochs=${MAX_EPOCHS}"
#     echo "=========================================="
#     CUDA_VISIBLE_DEVICES=0 python -m graph_signal_diffusion.cli.wra.train_pd \
#         --config-name=pd_training/wra_medium-large_outdoor_mid_density \
#         "training.r_min=${R_MIN}" \
#         "training.max_epochs=${MAX_EPOCHS}" \
#         "${OVERRIDE_ARGS[@]}"
#     echo "Finished r_min=${R_MIN}"
# done

# # ── Ultra-low-density sweep (base 10000 epochs) ──
# echo "############################################"
# echo "# Scenario: ultra-low_density  (base 10000 eps) #"
# echo "############################################"
# for R_MIN in "${R_MIN_VALUES[@]}"; do
#     MAX_EPOCHS=$(get_max_epochs "${R_MIN}" 10000)
#     echo "=========================================="
#     echo "Starting r_min=${R_MIN}  max_epochs=${MAX_EPOCHS}"
#     echo "=========================================="
#     CUDA_VISIBLE_DEVICES=1 python -m graph_signal_diffusion.cli.wra.train_pd \
#         --config-name=pd_training/wra_medium-large_outdoor_ultra-low_density \
#         "training.r_min=${R_MIN}" \
#         "training.max_epochs=${MAX_EPOCHS}" \
#         "${OVERRIDE_ARGS[@]}"
#     echo "Finished r_min=${R_MIN}"
# done


# ── High-density sweep (base 30000 epochs) ──
echo "############################################"
echo "# Scenario: high_density  (base 30000 eps) #"
echo "############################################"
for R_MIN in "${R_MIN_VALUES[@]}"; do
    MAX_EPOCHS=$(get_max_epochs "${R_MIN}" 30000)
    echo "=========================================="
    echo "Starting r_min=${R_MIN}  max_epochs=${MAX_EPOCHS}"
    echo "=========================================="
    CUDA_VISIBLE_DEVICES=1 python -m graph_signal_diffusion.cli.wra.train_pd \
        --config-name=pd_training/wra_medium-large_outdoor_high_density \
        "training.r_min=${R_MIN}" \
        "training.max_epochs=${MAX_EPOCHS}" \
        "${OVERRIDE_ARGS[@]}"
    echo "Finished r_min=${R_MIN}"
done


echo "All r_min sweep runs complete."
