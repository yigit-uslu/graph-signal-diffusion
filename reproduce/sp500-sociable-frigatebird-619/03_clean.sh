#!/bin/bash
# Stage 4 — Clean the frozen raw into the corr_0.7 cleaned data root.
#
# This step is DETERMINISTIC given a fixed raw/ input. Verified (see README §3):
# it regenerates $CLEANED_ROOT/raw/ byte-for-byte identical to the paper's data
# root for every data file (values.csv, adj.npy, stocks.csv, fundamentals.csv,
# graph_metadata.json) — the ONLY difference is the wall-clock `timestamp` field
# inside metadata.json. So this is the true reproducible entry point: run it on
# the frozen raw and you land on the identical dataset.
#
# The large processed_<hash>/ tensor caches are NOT written here — the dataset
# loader builds them lazily on first train/eval from this cleaned raw/.
#
# Produces: $CLEANED_ROOT/raw/{values.csv, adj.npy, stocks.csv, fundamentals.csv,
#                              metadata.json, graph_metadata.json, adjacency_visualization.pdf}
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/00_config.sh"

if [ ! -d "$RAW_DIR" ]; then
  echo "Missing '$RAW_DIR'. Restore the frozen raw (README §2) before cleaning." >&2
  exit 1
fi

# Exactly the command that produced the paper's cleaned root (verified equivalent).
$PYRUN -m graph_signal_diffusion.cli.stock.clean \
    --input-dir "$RAW_DIR" \
    --method "$METHOD" \
    --min-coverage "$MIN_COVERAGE" \
    --edge-weight-threshold "$EDGE_WEIGHT_THRESHOLD" \
    --sector-bonus "$SECTOR_BONUS"

echo ""
echo "Cleaned data root -> $CLEANED_ROOT/raw"
echo "Validate byte-equivalence (from repo root):"
echo "    sha256sum -c \"$REPRO_DIR/checksums/cleaned_raw.sha256\""
echo "(or run ./verify_data_equivalence.sh for the full move-aside / diff / restore check)"
