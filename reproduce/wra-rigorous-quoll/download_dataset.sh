#!/bin/bash
# CONSUMER — fetch the referenced expert dataset from the PUBLIC shared data repo
# (a GitHub Release in `gsd-dataset`) and restore it under DATASET_ROOT (data/wra).
#
# No git-LFS and NO GitHub account/token: the ~39 GB dataset is hosted as split
# (<=2 GB) tar parts on a PUBLIC release, fetched with plain curl (anonymous).
# Needs curl, tar, sha256sum and ~78 GB free disk (downloaded parts + extracted).
# Point DATASET_ROOT elsewhere / symlink if data/wra is on a small volume.
#
# Usage:
#   ./download_dataset.sh
#   GH_REPO=owner/repo DATASET_RELEASE_TAG=some-tag ./download_dataset.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/00_config.sh"

REPO="$GH_REPO"                       # shared public data repo (from 00_config.sh)
TAG="$DATASET_RELEASE_TAG"            # this dataset's release tag (from 00_config.sh)
BASE_URL="${DATASET_DOWNLOAD_BASE:-https://github.com/${REPO}/releases/download/${TAG}}"
WORK="${DOWNLOAD_WORK:-${REPRO_DIR}/.download}"

command -v curl >/dev/null || { echo "FATAL: curl not found." >&2; exit 1; }
mkdir -p "$DATASET_ROOT" "$WORK"

echo "== 1/4 fetching part manifest from ${BASE_URL} =="
curl -fL --retry 3 -o "$WORK/dataset_archives.sha256" "${BASE_URL}/dataset_archives.sha256"

# The sha256 manifest lists every part ("<hash>  <name>"); drive the download from
# it so we never have to guess how many parts each sub-dataset split into. Each part
# is verified right after download, so a re-run resumes (cached parts are skipped).
echo "== 2/4 downloading + verifying parts (anonymous curl, no auth) =="
n=0
while read -r want name; do
  [ -n "${name:-}" ] || continue
  n=$((n + 1))
  dst="$WORK/$name"
  if [ -f "$dst" ] && (cd "$WORK" && printf '%s  %s\n' "$want" "$name" | sha256sum -c --status -); then
    echo "   ok   $name  (cached)"
    continue
  fi
  echo "   get  $name"
  curl -fL --retry 3 -o "$dst" "${BASE_URL}/${name}" </dev/null
  (cd "$WORK" && printf '%s  %s\n' "$want" "$name" | sha256sum -c --status -) \
    || { echo "FATAL: checksum mismatch after download: $name" >&2; exit 1; }
done < "$WORK/dataset_archives.sha256"
[ "$n" -gt 0 ] || { echo "FATAL: no parts listed in the manifest." >&2; exit 1; }

echo "== 3/4 reassembling + extracting each sub-dataset → ${DATASET_ROOT} =="
shopt -s nullglob
for first in "$WORK"/*.tar.part-00; do
  base="${first%.part-00}"
  name="$(basename "$base")"
  echo "   ${name}  ($(ls "${base}".part-* | wc -l) parts)"
  cat "${base}".part-* | tar -x -C "$DATASET_ROOT"
done
shopt -u nullglob

echo "== 4/4 verifying against the dataset manifest =="
bash "$REPRO_DIR/verify_dataset_manifest.sh"

echo ""
echo "Done. Reclaim space with: rm -rf '$WORK'"
