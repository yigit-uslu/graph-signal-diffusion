#!/bin/bash
# PRODUCER (dataset owner only) — package the frozen S&P 500 raw into gzip tar parts
# and upload them to a GitHub Release on the SHARED PUBLIC `gsd-dataset` repo. No
# git-LFS; the parts are release assets and never enter the git tree.
#
# The raw is small (~0.3 GB, ~0.1 GB gzipped), so this is a single tar split into
# <=PARTSIZE parts (one part in practice). Requires the `gh` CLI authenticated with
# write access to the repo.
#
# Usage:
#   ./package_dataset.sh                 # package + upload the frozen raw
#   PARTSIZE=1900M ./package_dataset.sh  # tune the split size (< 2 GB GitHub limit)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/00_config.sh"

# GH_REPO and DATASET_RELEASE_TAG come from 00_config.sh (single source of truth).
REPO="$GH_REPO"
TAG="$DATASET_RELEASE_TAG"
PARTSIZE="${PARTSIZE:-1900M}"
WORK="${PACKAGE_WORK:-${REPRO_DIR}/.package}"
RAW_MANIFEST="$REPRO_DIR/checksums/raw.sha256"

command -v gh >/dev/null || { echo "FATAL: gh CLI not found." >&2; exit 1; }
[ -f "$RAW_MANIFEST" ] || { echo "FATAL: raw manifest not found: $RAW_MANIFEST" >&2; exit 1; }
mkdir -p "$WORK"

# The files we publish are EXACTLY those pinned in checksums/raw.sha256 (the frozen
# paper raw). Refuse to publish if the on-disk raw does not match — never ship data
# that differs from what sociable-frigatebird-619 trained on. (This content check is
# the SP500 analog of WRA's dataset_bundle_id.sh --check guard.)
echo "== verifying on-disk raw matches the frozen checksums =="
sha256sum -c "$RAW_MANIFEST" \
  || { echo "FATAL: raw != checksums/raw.sha256 — refusing to publish." >&2; exit 1; }

# Manifest paths are relative to the project root; store them in the tar relative to
# RAW_DIR (bare names) so `tar -x -C RAW_DIR` restores data/sp500/raw/<file>.
mapfile -t FILES < <(awk 'NF {print $2}' "$RAW_MANIFEST")
[ "${#FILES[@]}" -ge 1 ] || { echo "FATAL: no files listed in $RAW_MANIFEST" >&2; exit 1; }
REL=()
for f in "${FILES[@]}"; do REL+=("${f#"$RAW_DIR"/}"); done

# Ensure the shared PUBLIC data repo exists (idempotent; it already does — WRA created it).
if ! gh repo view "$REPO" >/dev/null 2>&1; then
  echo "== creating PUBLIC data repo '${REPO}' =="
  gh repo create "$REPO" --public \
    --description "Large diffusion datasets for graph-signal-diffusion (GitHub Release assets; no git-LFS)."
fi

# A release needs a target commit; a freshly-created repo is empty (release create
# 422s with "Repository is empty"). Seed an initial README commit via the API if the
# repo has no commits yet. No-op once the repo is populated (it is).
if ! gh api "repos/${REPO}/commits" -q '.[0].sha' >/dev/null 2>&1; then
  echo "== seeding empty repo '${REPO}' with an initial commit =="
  README_B64="$(printf '%s\n' \
    "# ${REPO##*/}" "" \
    "Large diffusion datasets for **graph-signal-diffusion**," \
    "hosted as GitHub **Release** assets (split tar parts; no git-LFS)." "" \
    "Each dataset is a separate release tag; see the consuming repo's" \
    "reproduce/*/download_dataset.sh to fetch + verify one." | base64 -w0)"
  gh api --method PUT "repos/${REPO}/contents/README.md" \
    -f message="Initialize ${REPO##*/}" \
    -f content="$README_B64" >/dev/null
fi

read -r -d '' RELEASE_NOTES <<'NOTES' || true
This is the frozen raw S&P 500 dataset used to reproduce the sociable-frigatebird-619 diffusion-forecasting run in graph-signal-diffusion. It bundles the exact bytes the paper cleaned and trained on: daily OHLCV plus engineered return columns (values.csv), S&P 500 constituent + sector metadata (stocks.csv), fundamentals (fundamentals.csv), and a precomputed adjacency (adj.npy). The prices are sourced from Yahoo Finance (yfinance) and cover 2016-03-18 through 2026-02-13. The data is distributed as gzip tar parts (no git-LFS) and restores the frozen data/sp500/raw/ tree; downstream, reproduce/sp500-sociable-frigatebird-619/03_clean.sh deterministically rebuilds the cleaned data root from it. You can download and verify it anonymously with plain curl (no GitHub account or token required) using reproduce/sp500-sociable-frigatebird-619/download_dataset.sh.
NOTES

# Ensure the release exists (idempotent).
if ! gh release view "$TAG" --repo "$REPO" >/dev/null 2>&1; then
  echo "== creating release '${TAG}' =="
  gh release create "$TAG" --repo "$REPO" \
    --title "S&P 500 frozen raw (sociable-frigatebird-619; yfinance thru 2026-02-13)" \
    --notes "$RELEASE_NOTES"
fi

name="sp500-raw"
echo "== packing ${#REL[@]} files from ${RAW_DIR} (gzip) =="
: > "$WORK/dataset_archives.sha256"
# tar relative to RAW_DIR so extraction restores <RAW_DIR>/<file>.
tar -czf - -C "$RAW_DIR" "${REL[@]}" | split -b "$PARTSIZE" -d -a 2 - "$WORK/${name}.tgz.part-"
( cd "$WORK" && sha256sum "${name}".tgz.part-* >> dataset_archives.sha256 )
echo "   uploading $(ls "$WORK/${name}".tgz.part-* | wc -l) part(s) ..."
gh release upload "$TAG" --repo "$REPO" --clobber "$WORK/${name}".tgz.part-*

echo "== uploading part checksum manifest =="
gh release upload "$TAG" --repo "$REPO" --clobber "$WORK/dataset_archives.sha256"
cp "$WORK/dataset_archives.sha256" "$REPRO_DIR/checksums/dataset_archives.sha256"
rm -f "$WORK/${name}".tgz.part-*        # free disk

echo ""
echo "Release '${TAG}' populated. Consumers restore with ./download_dataset.sh"
echo "The part-checksum manifest is also committed at checksums/dataset_archives.sha256"
