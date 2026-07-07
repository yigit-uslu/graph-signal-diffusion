#!/bin/bash
# Compute (and verify) the content id embedded in the dataset release tag.
#
# WHY THIS EXISTS
#   rigorous-quoll-131 trains on FOUR content-addressed sub-datasets (one per
#   network density). Each sub-dataset dir is named `..._h<hash>`, where <hash> is
#   md5[:12] of that sub-dataset's canonicalized build inputs — scenario/channel
#   generation + primal-dual trainer + sample collection + dataset build (see
#   src/graph_signal_diffusion/datasets/wra/channel_factory.py). So the dataset's
#   identity is FOUR hashes, not one. To tag the *release* (a single string) we fold
#   the four into one deterministic "bundle id".
#
# DEFINITION (canonical — reproducible offline from the checked-in manifest)
#   1. Take the four sub-dataset relpaths from checksums/dataset_manifest.txt
#      (field 1, "|"-separated).
#   2. Extract each trailing "_h<hex>" token -> the bare lowercase 12-hex string
#      (drops the leading "h"): e.g. 0dd7afd393f9.
#   3. Sort the tokens bytewise with LC_ALL=C, so the id depends only on the SET of
#      sub-datasets — not on their listing order, nor on the host locale.
#   4. Canonical byte stream = the sorted tokens, one per line, each terminated by
#      "\n" (a trailing newline after the last token — exactly `printf '%s\n'`).
#   5. bundle_id = md5(stream) truncated to the first 12 hex chars — the SAME
#      md5[:12] convention used for the per-sub-dataset hashes (md5 is used here as
#      a content fingerprint, not a security primitive).
#
#   release_tag = "<DATASET_TAG_STEM>-<bundle_id>"   (stem default: wra-N400-gsd)
#
#   Reproduce by hand:
#     printf '%s\n' 0dd7afd393f9 43d4a26a4203 c1f8f7a25432 a6c7c432ee13 \
#       | LC_ALL=C sort | md5sum | cut -c1-12          # -> 85faf506ec70
#
# USAGE
#   ./dataset_bundle_id.sh            # print the bundle id      (e.g. 85faf506ec70)
#   ./dataset_bundle_id.sh --id       # same as above
#   ./dataset_bundle_id.sh --tag      # print the full tag       (wra-N400-gsd-<id>)
#   ./dataset_bundle_id.sh --check    # assert tag == DATASET_RELEASE_TAG; exit 1 on drift
#   TOKEN_SOURCE=disk ./dataset_bundle_id.sh --check   # also require the 4 dirs to exist
#
# See DATASET_TAG.md for the full write-up and design notes.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/00_config.sh"

STEM="${DATASET_TAG_STEM:-wra-N400-gsd}"
MANIFEST="$REPRO_DIR/checksums/dataset_manifest.txt"
[ -f "$MANIFEST" ] || { echo "FATAL: manifest not found: $MANIFEST" >&2; exit 1; }

# 1. Read the sub-dataset relpaths (field 1) from the manifest (skip #comments / blanks).
mapfile -t RELPATHS < <(awk -F'|' '!/^#/ && NF {print $1}' "$MANIFEST")
[ "${#RELPATHS[@]}" -ge 1 ] || { echo "FATAL: no sub-dataset rows in $MANIFEST" >&2; exit 1; }

# 2. Extract the trailing _h<12-hex> token from each relpath. With TOKEN_SOURCE=disk,
#    also require the dir to physically exist under DATASET_ROOT (proves the tag
#    matches the dataset present, not just the manifest text).
TOKENS=()
for rel in "${RELPATHS[@]}"; do
  if [ "${TOKEN_SOURCE:-manifest}" = "disk" ]; then
    [ -d "$DATASET_ROOT/$rel" ] || { echo "FATAL: missing sub-dataset dir: $DATASET_ROOT/$rel" >&2; exit 1; }
  fi
  tok="$(printf '%s\n' "$rel" | sed -n 's/.*_h\([0-9a-f]\{12\}\)$/\1/p')"
  [ -n "$tok" ] || { echo "FATAL: no _h<12-hex> token in relpath: $rel" >&2; exit 1; }
  TOKENS+=("$tok")
done
[ "${#TOKENS[@]}" -eq "${#RELPATHS[@]}" ] || { echo "FATAL: token/relpath count mismatch" >&2; exit 1; }

# 3-5. Sorted (LC_ALL=C) tokens, one per line -> md5 -> first 12 hex.
BUNDLE_ID="$(printf '%s\n' "${TOKENS[@]}" | LC_ALL=C sort | md5sum | cut -c1-12)"
TAG="${STEM}-${BUNDLE_ID}"

case "${1:-}" in
  ""|--id) echo "$BUNDLE_ID" ;;
  --tag)   echo "$TAG" ;;
  --check)
    if [ "$TAG" = "$DATASET_RELEASE_TAG" ]; then
      echo "OK   recomputed tag matches DATASET_RELEASE_TAG: $TAG  (bundle id $BUNDLE_ID)"
    else
      echo "FAIL recomputed '$TAG' != pinned DATASET_RELEASE_TAG '$DATASET_RELEASE_TAG'" >&2
      echo "     update DATASET_RELEASE_TAG in 00_config.sh (or re-check the manifest)." >&2
      exit 1
    fi
    ;;
  *) echo "usage: $(basename "$0") [--id | --tag | --check]   (env: TOKEN_SOURCE=manifest|disk, DATASET_TAG_STEM)" >&2; exit 2 ;;
esac
