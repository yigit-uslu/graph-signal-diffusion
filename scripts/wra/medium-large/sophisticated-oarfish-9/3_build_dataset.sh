#!/bin/bash
# Stage 3: Build diffusion-ready datasets from completed PD expert runs.
#
# Converts primal history from all 20 PD runs (4 densities x 5 r_min) into
# the raw/ format consumed by WRADataset.process().
#
# For each density, r_min=0.6 is built first with H_instantaneous enabled
# (the reference).  The remaining r_min values symlink their h_instantaneous/
# directories to that reference via link_h_instantaneous_ref.sh.
#
# After Stage 2 completes, fill in the INPUT_DIR paths below from the PD
# output directories (the ones containing collected_samples.npz).
#
# Usage:
#   bash scripts/wra/medium-large/sophisticated-oarfish-9/3_build_dataset.sh

set -euo pipefail
export HYDRA_FULL_ERROR=1

LINK_HELPER=scripts/wra/link_h_instantaneous_ref.sh

COMMON_ARGS=(
    collection.sample_source=primal_history
    collection.primal_history.window_size=1000
    collection.primal_history.refine_feasible_subset=true
    collection.target_samples_per_network=200
)

# ── PD output paths lookup table ──
# Each entry: "density|r_min|input_dir"
# Fill in the input_dir after Stage 2 completes.
# r_min=0.6 MUST appear first within each density block (H_instantaneous reference).
read -r -d '' PD_RUNS <<'TABLE' || true
ultra-low|0.6|outputs/wra_medium-large_outdoor_ultra-low_density/wrpd_v1_wrach_v1_s42_D32_N400_R7800_v3_h83a5ca655906_r0.6_a0.2_ha8f90c3f236f/2026-03-15/11-54-13
ultra-low|0.4|outputs/wra_medium-large_outdoor_ultra-low_density/wrpd_v1_wrach_v1_s42_D32_N400_R7800_v3_h83a5ca655906_r0.4_a0.2_hee178290f551/2026-03-15/15-34-39
ultra-low|0.5|outputs/wra_medium-large_outdoor_ultra-low_density/wrpd_v1_wrach_v1_s42_D32_N400_R7800_v3_h83a5ca655906_r0.5_a0.2_h7afcdff41256/2026-03-15/18-52-45
ultra-low|0.7|outputs/wra_medium-large_outdoor_ultra-low_density/wrpd_v1_wrach_v1_s42_D32_N400_R7800_v3_h83a5ca655906_r0.7_a0.2_hd71a1624c374/2026-03-15/22-16-54
ultra-low|0.8|outputs/wra_medium-large_outdoor_ultra-low_density/wrpd_v1_wrach_v1_s42_D32_N400_R7800_v3_h83a5ca655906_r0.8_a0.2_h99b12fea8846/2026-03-16/05-37-41
low|0.6|outputs/wra_medium-large_outdoor_low_density/wrpd_v1_wrach_v1_s42_D32_N400_R7000_v3_h1682e1bdff71_r0.6_a0.2_h864914b22c7b/2026-03-14/18-44-55
low|0.4|outputs/wra_medium-large_outdoor_low_density/wrpd_v1_wrach_v1_s42_D32_N400_R7000_v3_h1682e1bdff71_r0.4_a0.2_hb3fbcf492398/2026-03-14/23-34-00
low|0.5|outputs/wra_medium-large_outdoor_low_density/wrpd_v1_wrach_v1_s42_D32_N400_R7000_v3_h1682e1bdff71_r0.5_a0.2_h24a49f6c4171/2026-03-15/01-42-50
low|0.7|outputs/wra_medium-large_outdoor_low_density/wrpd_v1_wrach_v1_s42_D32_N400_R7000_v3_h1682e1bdff71_r0.7_a0.2_h0c588eac76d8/2026-03-15/04-41-34
low|0.8|outputs/wra_medium-large_outdoor_low_density/wrpd_v1_wrach_v1_s42_D32_N400_R7000_v3_h1682e1bdff71_r0.8_a0.2_h7a9ca5beeec2/2026-03-15/10-53-04
mid|0.6|outputs/wra_medium-large_outdoor_mid_density/wrpd_v1_wrach_v1_s42_D32_N400_R6300_v3_h68582761d5df_r0.6_a0.2_hf183faf6339f/2026-03-14/19-04-11
mid|0.4|outputs/wra_medium-large_outdoor_mid_density/wrpd_v1_wrach_v1_s42_D32_N400_R6300_v3_h68582761d5df_r0.4_a0.2_h278d2d05ae5c/2026-03-15/18-54-57
mid|0.5|outputs/wra_medium-large_outdoor_mid_density/wrpd_v1_wrach_v1_s42_D32_N400_R6300_v3_h68582761d5df_r0.5_a0.2_h694589f8a766/2026-03-16/00-07-27
mid|0.7|outputs/wra_medium-large_outdoor_mid_density/wrpd_v1_wrach_v1_s42_D32_N400_R6300_v3_h68582761d5df_r0.7_a0.2_h1cbee1e461e9/2026-03-16/07-33-42
mid|0.8|outputs/wra_medium-large_outdoor_mid_density/wrpd_v1_wrach_v1_s42_D32_N400_R6300_v3_h68582761d5df_r0.8_a0.2_h4a47f7f0dbca/2026-03-16/23-50-48
high|0.6|outputs/wra_medium-large_outdoor_high_density/wrpd_v1_wrach_v1_s42_D32_N400_R5800_v3_h2dda4a04ca7b_r0.6_a0.2_h9361961eadf0/2026-03-16/16-41-22
high|0.4|outputs/wra_medium-large_outdoor_high_density/wrpd_v1_wrach_v1_s42_D32_N400_R5800_v3_h2dda4a04ca7b_r0.4_a0.2_haeeb262dc512/2026-03-17/00-34-17
high|0.5|outputs/wra_medium-large_outdoor_high_density/wrpd_v1_wrach_v1_s42_D32_N400_R5800_v3_h2dda4a04ca7b_r0.5_a0.2_haf3bf83302a0/2026-03-17/12-07-32
high|0.7|outputs/wra_medium-large_outdoor_high_density/wrpd_v1_wrach_v1_s42_D32_N400_R5800_v3_h2dda4a04ca7b_r0.7_a0.2_h4f814ddbf57b/2026-03-17/22-59-38
high|0.8|outputs/wra_medium-large_outdoor_high_density/wrpd_v1_wrach_v1_s42_D32_N400_R5800_v3_h2dda4a04ca7b_r0.8_a0.2_h310269b3a34c/2026-03-18/16-09-37
TABLE

# Helper: find the most recently created raw/ dir under a scenario root.
find_latest_raw() {
    local scenario_root="$1"
    find "$scenario_root" -maxdepth 2 -name raw -type d -printf '%T@ %p\n' \
        | sort -n | tail -1 | cut -d' ' -f2-
}

prev_density=""
total=$(echo "$PD_RUNS" | grep -c '|')
counter=0

while IFS='|' read -r density r_min input_dir; do
    [[ -z "$density" ]] && continue

    scenario_root="data/wra/medium-large_outdoor_${density}_density"
    config="pd_collection/wra_medium-large_outdoor_${density}_density"
    counter=$(( counter + 1 ))

    # Validate input dir exists
    if [[ ! -d "$input_dir" ]]; then
        echo "ERROR: input_dir not found: $input_dir" >&2
        exit 1
    fi

    is_reference=false
    if [[ "$density" != "$prev_density" ]]; then
        # First entry for this density must be r_min=0.6 (the reference)
        if [[ "$r_min" != "0.6" ]]; then
            echo "ERROR: first r_min for ${density} density must be 0.6, got ${r_min}" >&2
            exit 1
        fi
        is_reference=true
        prev_density="$density"
    fi

    echo "====== [${counter}/${total}] ${density} density | r_min=${r_min} ======"

    if $is_reference; then
        # Build reference with H_instantaneous enabled
        python -m graph_signal_diffusion.cli.wra.build_diffusion_dataset \
            --config-name="$config" \
            input_dir="$input_dir" \
            training.r_min="$r_min" \
            output.h_instantaneous.enabled=true \
            "${COMMON_ARGS[@]}"

        # Create the stable anchor symlink for this density
        ref_raw=$(find_latest_raw "$scenario_root")
        bash "$LINK_HELPER" create-anchor "$scenario_root" "$ref_raw"
    else
        # Build without H_instantaneous, then symlink to reference
        python -m graph_signal_diffusion.cli.wra.build_diffusion_dataset \
            --config-name="$config" \
            input_dir="$input_dir" \
            training.r_min="$r_min" \
            "${COMMON_ARGS[@]}"

        bash "$LINK_HELPER" link "$scenario_root" "$(find_latest_raw "$scenario_root")"
    fi

    echo "====== Done: ${density} density | r_min=${r_min} ======"
    echo
done <<< "$PD_RUNS"

echo "All 20 diffusion datasets built."
