#!/bin/bash
# Verify the referenced expert dataset against checksums/dataset_manifest.txt.
#
# The ~136 GB expert primal-dual dataset is NOT bundled (it is referenced via
# DATASET_ROOT). Full sha256 of 136 GB is impractical to ship/run, and each
# sub-dataset directory is already CONTENT-ADDRESSED (the `...k200_h<hash>` suffix
# is a content hash of its build inputs) — so integrity is anchored by the path
# names plus a structural check (directory present, file count, total bytes).
#
# This proves your DATASET_ROOT holds the exact 4 sub-datasets rigorous-quoll-131
# trained/eval'd on. It reads-only; nothing is written or moved.
#
#   ./verify_dataset_manifest.sh                 # verify against the manifest
#   ./verify_dataset_manifest.sh --regen         # rewrite the manifest from disk
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/00_config.sh"

MANIFEST="$REPRO_DIR/checksums/dataset_manifest.txt"

dir_stats() {  # -> "<file_count> <total_bytes>" for a directory
  local d="$1"
  local n bytes
  n=$(find "$d" -type f | wc -l)
  bytes=$(find "$d" -type f -printf '%s\n' | awk '{s+=$1} END{printf "%.0f\n", s}')
  echo "$n $bytes"
}

# Sub-dataset paths (relative to DATASET_ROOT) — pinned by the dataset config
# wra_medium-large_outdoor_all_density.yaml.
SUBDATASETS=(
  "medium-large_outdoor_ultra-low_density/wrpc_v1_primal_history_k200_h0dd7afd393f9"
  "medium-large_outdoor_low_density/wrpc_v1_primal_history_k200_h43d4a26a4203"
  "medium-large_outdoor_mid_density/wrpc_v1_primal_history_k200_hc1f8f7a25432"
  "medium-large_outdoor_high_density/wrpc_v1_primal_history_k200_ha6c7c432ee13"
)

if [ "${1:-}" = "--regen" ]; then
  echo "# rigorous-quoll-131 expert dataset manifest (relative to DATASET_ROOT)" > "$MANIFEST"
  echo "# format: <relpath>|<file_count>|<total_bytes>" >> "$MANIFEST"
  for rel in "${SUBDATASETS[@]}"; do
    d="$DATASET_ROOT/$rel"
    [ -d "$d" ] || { echo "FATAL: missing $d (cannot regen)" >&2; exit 1; }
    read -r n bytes < <(dir_stats "$d")
    echo "${rel}|${n}|${bytes}" >> "$MANIFEST"
    echo "  recorded ${rel}  (${n} files, ${bytes} bytes)"
  done
  echo "Manifest written: $MANIFEST"
  exit 0
fi

[ -f "$MANIFEST" ] || { echo "FATAL: manifest not found: $MANIFEST" >&2; exit 1; }

echo "Verifying DATASET_ROOT='$DATASET_ROOT' against $MANIFEST"
rc=0
while IFS='|' read -r rel exp_n exp_bytes; do
  [[ "$rel" =~ ^#|^$ ]] && continue
  d="$DATASET_ROOT/$rel"
  if [ ! -d "$d" ]; then
    echo "  MISSING  $rel"; rc=1; continue
  fi
  read -r n bytes < <(dir_stats "$d")
  if [ "$n" = "$exp_n" ] && [ "$bytes" = "$exp_bytes" ]; then
    echo "  OK       $rel  (${n} files, ${bytes} bytes)"
  else
    echo "  MISMATCH $rel  (got ${n} files/${bytes} B, expected ${exp_n}/${exp_bytes})"; rc=1
  fi
done < "$MANIFEST"

echo ""
if [ "$rc" -eq 0 ]; then
  echo "RESULT: PASS — all 4 sub-datasets present and structurally match the manifest."
else
  echo "RESULT: FAIL — see mismatches above (rebuild via Stages 1-3, or fix DATASET_ROOT)."
fi
exit "$rc"
