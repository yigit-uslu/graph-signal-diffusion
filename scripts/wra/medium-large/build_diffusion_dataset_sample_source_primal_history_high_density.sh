#!/bin/bash
# Build diffusion-ready raw WRA dataset from completed PD runs (medium-large outdoor high density)
# using primal history as sample source, for r_min sweep [0.4, 0.5, 0.6, 0.7, 0.8].
#
# r_min=0.6 is built first as the H_instantaneous reference; the others symlink to it.
set -euo pipefail
export HYDRA_FULL_ERROR=1

SCENARIO_ROOT=data/wra/medium-large_outdoor_high_density
LINK_HELPER=scripts/wra/link_h_instantaneous_ref.sh
CONFIG=pd_collection/wra_medium-large_outdoor_high_density
COMMON_ARGS=(
    collection.sample_source=primal_history
    collection.primal_history.window_size=1000
    collection.primal_history.refine_feasible_subset=true
    collection.target_samples_per_network=200
)

# Helper: find the most recently created raw/ dir under SCENARIO_ROOT
find_latest_raw() {
    find "$SCENARIO_ROOT" -maxdepth 2 -name raw -type d -printf '%T@ %p\n' \
        | sort -n | tail -1 | cut -d' ' -f2-
}

# =============================================================================
# 1) Reference: r_min=0.6 (with H_instantaneous enabled)
# =============================================================================
INPUT_DIR=outputs/wra_medium-large_outdoor_high_density/wrpd_v1_wrach_v1_s42_D32_N400_R5800_v3_h2dda4a04ca7b_r0.6_a0.2_h9361961eadf0/2026-03-16/16-41-22
python -m graph_signal_diffusion.cli.wra.build_diffusion_dataset \
    --config-name="$CONFIG" \
    input_dir=$INPUT_DIR \
    training.r_min=0.6 \
    output.h_instantaneous.enabled=true \
    "${COMMON_ARGS[@]}" \
    "$@"

# Create the stable anchor symlink
REF_RAW=$(find_latest_raw)
bash "$LINK_HELPER" create-anchor "$SCENARIO_ROOT" "$REF_RAW"

# =============================================================================
# 2) r_min=0.4
# =============================================================================
INPUT_DIR=outputs/wra_medium-large_outdoor_high_density/wrpd_v1_wrach_v1_s42_D32_N400_R5800_v3_h2dda4a04ca7b_r0.4_a0.2_haeeb262dc512/2026-03-17/00-34-17
python -m graph_signal_diffusion.cli.wra.build_diffusion_dataset \
    --config-name="$CONFIG" \
    input_dir=$INPUT_DIR \
    training.r_min=0.4 \
    "${COMMON_ARGS[@]}" \
    "$@"

bash "$LINK_HELPER" link "$SCENARIO_ROOT" "$(find_latest_raw)"

# =============================================================================
# 3) r_min=0.5
# =============================================================================
INPUT_DIR=outputs/wra_medium-large_outdoor_high_density/wrpd_v1_wrach_v1_s42_D32_N400_R5800_v3_h2dda4a04ca7b_r0.5_a0.2_haf3bf83302a0/2026-03-17/12-07-32
python -m graph_signal_diffusion.cli.wra.build_diffusion_dataset \
    --config-name="$CONFIG" \
    input_dir=$INPUT_DIR \
    training.r_min=0.5 \
    "${COMMON_ARGS[@]}" \
    "$@"

bash "$LINK_HELPER" link "$SCENARIO_ROOT" "$(find_latest_raw)"

# =============================================================================
# 4) r_min=0.7
# =============================================================================
INPUT_DIR=outputs/wra_medium-large_outdoor_high_density/wrpd_v1_wrach_v1_s42_D32_N400_R5800_v3_h2dda4a04ca7b_r0.7_a0.2_h4f814ddbf57b/2026-03-17/22-59-38
python -m graph_signal_diffusion.cli.wra.build_diffusion_dataset \
    --config-name="$CONFIG" \
    input_dir=$INPUT_DIR \
    training.r_min=0.7 \
    "${COMMON_ARGS[@]}" \
    "$@"

bash "$LINK_HELPER" link "$SCENARIO_ROOT" "$(find_latest_raw)"

# =============================================================================
# 5) r_min=0.8
# =============================================================================
INPUT_DIR=outputs/wra_medium-large_outdoor_high_density/wrpd_v1_wrach_v1_s42_D32_N400_R5800_v3_h2dda4a04ca7b_r0.8_a0.2_h310269b3a34c/2026-03-18/16-09-37
python -m graph_signal_diffusion.cli.wra.build_diffusion_dataset \
    --config-name="$CONFIG" \
    input_dir=$INPUT_DIR \
    training.r_min=0.8 \
    "${COMMON_ARGS[@]}" \
    "$@"

bash "$LINK_HELPER" link "$SCENARIO_ROOT" "$(find_latest_raw)"
