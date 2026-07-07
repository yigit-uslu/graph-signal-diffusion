#!/bin/bash
# PRODUCER (dataset owner only) — package the referenced expert dataset into split
# (<=2 GB) tar parts and upload them to a GitHub Release. No git-LFS; the parts are
# release assets, never entering the git tree.
#
# Streams ONE sub-dataset at a time (tar → split → upload → delete parts) so peak
# extra disk is ~one sub-dataset (~9 GB), not the full ~39 GB — important on a tight
# volume. Requires the `gh` CLI authenticated with write access to the repo.
#
# Usage:
#   ./package_dataset.sh                 # package + upload all 4 sub-datasets
#   PARTSIZE=1900M ./package_dataset.sh  # tune the split size (< 2 GB GitHub limit)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/00_config.sh"

# GH_REPO and DATASET_RELEASE_TAG come from 00_config.sh (single source of truth).
REPO="$GH_REPO"
TAG="$DATASET_RELEASE_TAG"
PARTSIZE="${PARTSIZE:-1900M}"
WORK="${PACKAGE_WORK:-${REPRO_DIR}/.package}"

command -v gh >/dev/null || { echo "FATAL: gh CLI not found." >&2; exit 1; }
mkdir -p "$WORK"

# Refuse to publish under a mislabeled tag: recompute the bundle id from the on-disk
# sub-datasets and assert it equals DATASET_RELEASE_TAG (see DATASET_TAG.md).
echo "== verifying release tag matches dataset content =="
TOKEN_SOURCE=disk bash "$REPRO_DIR/dataset_bundle_id.sh" --check

# density → config-pinned sub-dataset relpath (the exact dirs rigorous-quoll-131 uses)
declare -A SUB=(
  [ultra-low]="medium-large_outdoor_ultra-low_density/wrpc_v1_primal_history_k200_h0dd7afd393f9"
  [low]="medium-large_outdoor_low_density/wrpc_v1_primal_history_k200_h43d4a26a4203"
  [mid]="medium-large_outdoor_mid_density/wrpc_v1_primal_history_k200_hc1f8f7a25432"
  [high]="medium-large_outdoor_high_density/wrpc_v1_primal_history_k200_ha6c7c432ee13"
)
ORDER=(ultra-low low mid high)

# Ensure the shared PUBLIC data repo exists (idempotent; create it public if missing).
if ! gh repo view "$REPO" >/dev/null 2>&1; then
  echo "== creating PUBLIC data repo '${REPO}' =="
  gh repo create "$REPO" --public \
    --description "Large diffusion datasets for graph-signal-diffusion (GitHub Release assets; no git-LFS)."
fi

# A release needs a target commit; a freshly-created repo is empty, so release create
# 422s with "Repository is empty". Seed an initial README commit via the API (no clone
# needed) if the repo has no commits yet.
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

# Human-readable release notes (full sentences). Kept here so a fresh
# `gh release create` uses them; the live release is also updated with the same
# text via `gh release edit --notes-file` when the description changes.
read -r -d '' RELEASE_NOTES <<'NOTES' || true
This is a diffusion-ready dataset of expert primal-dual (PD) algorithm rollouts for wireless resource allocation (WRA). It bundles four content-addressed sub-datasets, one per network density (ultra-low, low, mid, and high), all collected at a single QoS target of r_min=0.6. Each sample is a trajectory from a PD expert solving the power-allocation problem, formatted so that a graph diffusion model can be trained directly to imitate the expert. The data is distributed as split tar parts totaling roughly 39 GB and does not use git-LFS. It reproduces the rigorous-quoll-131 experiment in the WRA experiments of the graph-signal-diffusion repository. You can download and verify it anonymously with plain curl (no GitHub account or token required) using reproduce/wra-rigorous-quoll/download_dataset.sh.
NOTES

# Ensure the release exists (idempotent).
if ! gh release view "$TAG" --repo "$REPO" >/dev/null 2>&1; then
  echo "== creating release '${TAG}' =="
  gh release create "$TAG" --repo "$REPO" \
    --title "WRA expert diffusion dataset (rigorous-quoll-131)" \
    --notes "$RELEASE_NOTES"
fi

: > "$WORK/dataset_archives.sha256"
for d in "${ORDER[@]}"; do
  rel="${SUB[$d]}"
  src="$DATASET_ROOT/$rel"
  [ -d "$src" ] || { echo "FATAL: missing sub-dataset: $src" >&2; exit 1; }
  name="wra-rq_${d}"
  echo "== packing ${d}: ${rel} =="
  # tar relative to DATASET_ROOT so extraction restores <root>/<density>/<hash>/...
  tar -cf - -C "$DATASET_ROOT" "$rel" | split -b "$PARTSIZE" -d -a 2 - "$WORK/${name}.tar.part-"
  ( cd "$WORK" && sha256sum "${name}".tar.part-* >> dataset_archives.sha256 )
  echo "   uploading $(ls "$WORK/${name}".tar.part-* | wc -l) parts ..."
  gh release upload "$TAG" --repo "$REPO" --clobber "$WORK/${name}".tar.part-*
  rm -f "$WORK/${name}".tar.part-*        # free disk before the next sub-dataset
  echo "   done ${d}"
done

echo "== uploading part checksum manifest =="
gh release upload "$TAG" --repo "$REPO" --clobber "$WORK/dataset_archives.sha256"
cp "$WORK/dataset_archives.sha256" "$REPRO_DIR/checksums/dataset_archives.sha256"
echo ""
echo "Release '${TAG}' populated. Consumers restore with ./download_dataset.sh"
echo "The part-checksum manifest is also committed at checksums/dataset_archives.sha256"
