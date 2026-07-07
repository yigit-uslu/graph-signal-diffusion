#!/bin/bash
# CONSUMER — fetch the frozen S&P 500 raw from the PUBLIC shared data repo
# (a GitHub Release in `gsd-dataset`) and restore it under data/sp500/raw.
#
# This is the LFS-FREE acquisition path: the ~0.3 GB frozen raw is hosted as
# gzip-compressed tar parts on a PUBLIC release, fetched with plain curl (anonymous
# — no GitHub account or token). `git lfs pull` still works as a fallback while the
# LFS pointer is retained. Needs curl, tar, gzip, sha256sum and ~0.5 GB free disk
# (compressed part + extracted files).
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
mkdir -p "$RAW_DIR" "$WORK"

echo "== 1/4 fetching part manifest from ${BASE_URL} =="
curl -fL --retry 3 -o "$WORK/dataset_archives.sha256" "${BASE_URL}/dataset_archives.sha256"

# The sha256 manifest lists every part ("<hash>  <name>"); drive the download from
# it so we never have to guess how many parts the archive split into. Each part is
# verified right after download, so a re-run resumes (cached parts are skipped).
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

echo "== 3/4 reassembling + extracting → ${RAW_DIR} =="
shopt -s nullglob
parts=( "$WORK"/*.tgz.part-* )        # zero-padded suffixes -> glob sorts in order
[ "${#parts[@]}" -gt 0 ] || { echo "FATAL: no tar parts found in $WORK." >&2; exit 1; }
cat "${parts[@]}" | tar -xz -C "$RAW_DIR"
shopt -u nullglob

echo "== 4/4 verifying restored files against the frozen checksums =="
# checksums/raw.sha256 pins the exact bytes sociable-frigatebird-619 trained on;
# paths are relative to the project root (00_config.sh already cd'd us there).
sha256sum -c "$REPRO_DIR/checksums/raw.sha256"

echo ""
echo "Done. The frozen raw is restored at ${RAW_DIR}."
echo "Next: ${REPRO_DIR#$PROJECT_ROOT/}/03_clean.sh   (deterministic clean stage)"
echo "Reclaim space with: rm -rf '$WORK'"
