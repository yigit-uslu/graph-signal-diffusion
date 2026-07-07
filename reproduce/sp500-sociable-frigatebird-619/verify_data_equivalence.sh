#!/bin/bash
# Verify that Stage 4 (clean) is byte-reproducible: regenerate $CLEANED_ROOT/raw/
# from the frozen raw and prove it is identical to the current cleaned root.
#
# Strategy (safe + reversible): snapshot checksums -> move the current cleaned
# raw/ aside as a backup -> re-run clean -> compare -> RESTORE the original.
# The original is never deleted until the backup is confirmed in place, and the
# pristine original (incl. downstream node_selection_matrices/ and original
# timestamps) is restored at the end regardless of outcome.
#
# The big processed_<hash>/ caches and plots/ are untouched (clean.py does not
# write them). Expected result: all data files identical; metadata.json differs
# ONLY in its wall-clock `timestamp`; node_selection_matrices/ exists only in the
# original (a downstream artifact, not clean.py output).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/00_config.sh"

RAW_OUT="$CLEANED_ROOT/raw"
BACKUP="$CLEANED_ROOT/raw.orig_verify"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

DATA_FILES="values.csv adj.npy stocks.csv fundamentals.csv graph_metadata.json"

# --- Preconditions ----------------------------------------------------------
[ -d "$RAW_OUT" ]   || { echo "FATAL: '$RAW_OUT' not present — run ./03_clean.sh first." >&2; exit 1; }
[ ! -e "$BACKUP" ]  || { echo "FATAL: backup '$BACKUP' already exists — resolve it first." >&2; exit 1; }

echo "== snapshot original checksums =="
( cd "$RAW_OUT" && sha256sum $DATA_FILES ) > "$WORK/orig.sha256"
cp "$RAW_OUT/metadata.json" "$WORK/orig_metadata.json"
cat "$WORK/orig.sha256"

echo "== move original aside (backup) =="
mv "$RAW_OUT" "$BACKUP"
[ -d "$BACKUP" ] && [ ! -e "$RAW_OUT" ] || { echo "FATAL: move failed" >&2; exit 1; }

# From here on, always try to restore the original on the way out.
restore() {
  if [ -d "$BACKUP" ]; then
    rm -rf "$RAW_OUT" 2>/dev/null || true
    mv "$BACKUP" "$RAW_OUT"
    echo "== restored pristine original -> $RAW_OUT =="
  fi
}
trap 'restore; rm -rf "$WORK"' EXIT

echo "== re-run clean =="
$PYRUN -m graph_signal_diffusion.cli.stock.clean \
    --input-dir "$RAW_DIR" \
    --method "$METHOD" \
    --min-coverage "$MIN_COVERAGE" \
    --edge-weight-threshold "$EDGE_WEIGHT_THRESHOLD" \
    --sector-bonus "$SECTOR_BONUS" > "$WORK/clean.log" 2>&1 || { echo "clean FAILED (see below)"; tail -20 "$WORK/clean.log"; exit 1; }

echo "== compare data files (regenerated vs original) =="
rc=0
( cd "$RAW_OUT" && sha256sum -c "$WORK/orig.sha256" ) || rc=1

echo "== compare metadata.json (ignoring wall-clock timestamp) =="
if diff <(grep -v '"timestamp"' "$WORK/orig_metadata.json") \
        <(grep -v '"timestamp"' "$RAW_OUT/metadata.json") >/dev/null; then
  echo "metadata.json: identical apart from timestamp (expected)"
else
  echo "metadata.json: DIFFERS beyond timestamp"; rc=1
fi

echo ""
if [ "$rc" -eq 0 ]; then
  echo "RESULT: PASS — clean stage is byte-reproducible for all data files."
else
  echo "RESULT: FAIL — see mismatches above."
fi
# EXIT trap restores the pristine original.
exit "$rc"
