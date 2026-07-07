#!/bin/bash
# Train PD expert on wra_medium-large_outdoor_*_density scenario where * = low, mid, high, ultra-low.
# Collected samples feed into diffusion model training.

set -euo pipefail
export HYDRA_FULL_ERROR=1  # Enable full Hydra error traces for easier debugging.

cleanup() {
    # Ensure spawned background workers do not survive script exit/signals.
    jobs -pr | xargs -r kill -TERM
}
trap cleanup INT TERM EXIT

NUM_NETWORKS_OVERRIDE="${1:-}"
OVERRIDE_ARGS=()
if [[ -n "${NUM_NETWORKS_OVERRIDE}" ]]; then
    OVERRIDE_ARGS=("dataset.num_networks=${NUM_NETWORKS_OVERRIDE}")
    echo "Overriding dataset.num_networks=${NUM_NETWORKS_OVERRIDE} for all scenarios."
else
    echo "Using dataset.num_networks defaults from each scenario config."
fi

# CUDA_VISIBLE_DEVICES=0 python -m graph_signal_diffusion.cli.wra.train_pd \
#     --config-name=pd_training/wra_medium-large_outdoor_low_density \
#     "${OVERRIDE_ARGS[@]}" & p1=$!

# CUDA_VISIBLE_DEVICES=1 python -m graph_signal_diffusion.cli.wra.train_pd \
#     --config-name=pd_training/wra_medium-large_outdoor_mid_density \
#     "${OVERRIDE_ARGS[@]}" & p2=$!

# wait "$p1" # exits script if low_density failed
# wait "$p2" # exits script if mid_density failed


CUDA_VISIBLE_DEVICES=1 python -m graph_signal_diffusion.cli.wra.train_pd \
    --config-name=pd_training/wra_medium-large_outdoor_high_density \
    "${OVERRIDE_ARGS[@]}" & p3=$!

# CUDA_VISIBLE_DEVICES=1 python -m graph_signal_diffusion.cli.wra.train_pd \
#     --config-name=pd_training/wra_medium-large_outdoor_ultra-low_density \
#     "${OVERRIDE_ARGS[@]}" & p4=$!

wait "$p3" # exits script if high_density failed
# wait "$p4" # exits script if ultra-low_density failed
