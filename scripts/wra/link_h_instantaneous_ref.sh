#!/bin/bash
# Helper script to manage H_instantaneous symlinks across r_min sweep datasets.
#
# Usage:
#   # After building the reference (r_min=0.6) dataset:
#   bash scripts/wra/link_h_instantaneous_ref.sh create-anchor <scenario_data_root> <ref_raw_dir>
#
#   # After building any non-reference r_min dataset:
#   bash scripts/wra/link_h_instantaneous_ref.sh link <scenario_data_root> <target_raw_dir>
#
# Examples:
#   bash scripts/wra/link_h_instantaneous_ref.sh create-anchor \
#       data/wra/medium-large_outdoor_low_density \
#       data/wra/medium-large_outdoor_low_density/wrpc_v1_..._h<hash>/raw
#
#   bash scripts/wra/link_h_instantaneous_ref.sh link \
#       data/wra/medium-large_outdoor_low_density \
#       data/wra/medium-large_outdoor_low_density/wrpc_v1_..._h<hash>/raw

set -euo pipefail

MODE="${1:-}"
SCENARIO_ROOT="${2:-}"
RAW_DIR="${3:-}"

if [[ -z "$MODE" || -z "$SCENARIO_ROOT" || -z "$RAW_DIR" ]]; then
    echo "Usage: $0 {create-anchor|link} <scenario_data_root> <raw_dir>" >&2
    exit 1
fi

ANCHOR="$SCENARIO_ROOT/h_instantaneous_ref"

case "$MODE" in
    create-anchor)
        H_INST_DIR="$RAW_DIR/h_instantaneous"
        if [[ ! -d "$H_INST_DIR" ]]; then
            echo "ERROR: h_instantaneous directory not found: $H_INST_DIR" >&2
            echo "       Build the reference r_min=0.6 dataset with output.h_instantaneous.enabled=true first." >&2
            exit 1
        fi
        if [[ -L "$H_INST_DIR" ]]; then
            echo "ERROR: $H_INST_DIR is a symlink, not a real directory. Cannot use as reference." >&2
            exit 1
        fi

        # Create anchor as a relative symlink from scenario root to the real h_instantaneous dir
        REL_PATH=$(realpath --relative-to="$SCENARIO_ROOT" "$H_INST_DIR")
        ln -sfn "$REL_PATH" "$ANCHOR"
        echo "Created anchor: $ANCHOR -> $REL_PATH"
        ;;

    link)
        if [[ ! -L "$ANCHOR" && ! -d "$ANCHOR" ]]; then
            echo "ERROR: Anchor not found: $ANCHOR" >&2
            echo "       Run 'create-anchor' after building the reference r_min=0.6 dataset first." >&2
            exit 1
        fi

        TARGET="$RAW_DIR/h_instantaneous"

        # Skip if already correctly linked
        if [[ -L "$TARGET" ]]; then
            echo "Symlink already exists: $TARGET -> $(readlink "$TARGET") (skipping)"
            exit 0
        fi

        # Warn if a real directory exists (leftover from a build with h_instantaneous enabled)
        if [[ -d "$TARGET" ]]; then
            echo "WARNING: Real h_instantaneous directory exists at $TARGET" >&2
            echo "         This may be a leftover. Remove it manually and re-run if you want a symlink." >&2
            exit 1
        fi

        # Create symlink: raw/h_instantaneous -> ../../h_instantaneous_ref
        REL_PATH=$(realpath --relative-to="$RAW_DIR" "$ANCHOR")
        ln -s "$REL_PATH" "$TARGET"
        echo "Linked: $TARGET -> $REL_PATH"
        ;;

    *)
        echo "Unknown mode: $MODE. Use 'create-anchor' or 'link'." >&2
        exit 1
        ;;
esac
