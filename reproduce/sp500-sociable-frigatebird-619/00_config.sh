#!/bin/bash
# Shared configuration for the sociable-frigatebird-619 reproduction pipeline.
#
# Every stage script sources this file. You can also `source 00_config.sh`
# directly to print the resolved paths. Every value is overridable from the
# environment, e.g.:
#     CONDA_ENV=myenv CUDA_VISIBLE_DEVICES=1 ./05_evaluate_checkpoint.sh
#
# NOTE: scripts are location-independent — they resolve the project root from
# this file's path (the repro folder lives two levels below the repo root).

# --- Resolve the project root from this file's location ---------------------
_REPRO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPRO_DIR="$_REPRO_DIR"
export PROJECT_ROOT="$(cd "$_REPRO_DIR/../.." && pwd)"
cd "$PROJECT_ROOT" || { echo "FATAL: cannot cd to project root '$PROJECT_ROOT'" >&2; return 1 2>/dev/null || exit 1; }

# --- Python environment -----------------------------------------------------
# All CLIs run through the pinned conda env. Override CONDA_ENV (or PYRUN wholesale).
export CONDA_ENV="${CONDA_ENV:-torch_env}"
export PYRUN="${PYRUN:-conda run -n ${CONDA_ENV} python}"

# --- Dataset parameters (the corr_0.7 DS8 family) ---------------------------
# These reproduce the exact cleaned data-root NAME the run trained on, so do not
# change them unless you intend to build a different dataset.
export METHOD="${METHOD:-drop_incomplete}"
export MIN_COVERAGE="${MIN_COVERAGE:-0.95}"
export EDGE_WEIGHT_THRESHOLD="${EDGE_WEIGHT_THRESHOLD:-0.7}"
export SECTOR_BONUS="${SECTOR_BONUS:-0.05}"

export RAW_DIR="${RAW_DIR:-data/sp500/raw}"
export CLEANED_ROOT="${CLEANED_ROOT:-data/sp500/cleaned_${METHOD}_min_coverage_${MIN_COVERAGE}_corr_${EDGE_WEIGHT_THRESHOLD}_sector_bonus_${SECTOR_BONUS}}"
# `dataset.root` passed to train/eval (the loader appends /raw and /processed_<hash>):
export DATASET_ROOT="${DATASET_ROOT:-$CLEANED_ROOT}"

# --- Public dataset hosting (shared 'gsd-dataset' GitHub repo) ---------------
# The frozen raw (RAW_DIR, ~0.3 GB, Yahoo-Finance-derived) is published as gzip tar
# parts on a PUBLIC GitHub Release in the SHARED `gsd-dataset` data repo, so it can
# be fetched with anonymous curl (no account/token) — the LFS-free acquisition path
# (download_dataset.sh). `git lfs pull` still works as a fallback while the LFS
# pointer is retained. One release TAG per dataset; the tag encodes the yfinance
# data vintage (prices through 2026-02-13). SP500 uses a PLAIN versioned tag (no
# content-hash), since the raw is a single, non-content-addressed directory — the
# frozen bytes are instead pinned by checksums/raw.sha256, which package/download
# both verify against.
export GH_REPO="${GH_REPO:-yigit-uslu/gsd-dataset}"
export DATASET_TAG_STEM="${DATASET_TAG_STEM:-sp500-gsd}"
export DATASET_RELEASE_TAG="${DATASET_RELEASE_TAG:-${DATASET_TAG_STEM}-2026-02-13}"

# --- The frigatebird run (exact experiment identity) ------------------------
export EXPERIMENT_NAME="sociable-frigatebird-619"
export TASK="stock_price_forecasting_v3_learned_ds8"
export MODEL="ugnn_sp500_v3_ds8_norm_act_head"
export EXPERIMENT_DIR="outputs/${TASK}-sp500_cleaned/${MODEL}-ddim-gdm_sp500_v3_learned_ds8/${EXPERIMENT_NAME}"
# Checkpoint (tracker rank-1, ep4500 — what test_summary.json was written from).
# Prefer the 19MB copy bundled in this folder (self-contained; committed via
# REGULAR git — .pt is not LFS-tracked); fall back to the canonical training
# output if the bundle is absent. An explicit CHECKPOINT=... env override wins.
_REPRO_REL="${REPRO_DIR#$PROJECT_ROOT/}"
_BUNDLED_CKPT="${_REPRO_REL}/checkpoint/best_model_epoch_4500.pt"
_CANONICAL_CKPT="${EXPERIMENT_DIR}/trainer_chkpts/best_models/best_model_epoch_4500.pt"
if [ -z "${CHECKPOINT:-}" ]; then
  if [ -f "$_BUNDLED_CKPT" ]; then CHECKPOINT="$_BUNDLED_CKPT"; else CHECKPOINT="$_CANONICAL_CKPT"; fi
fi
export CHECKPOINT

# --- GPU --------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# --- Banner when sourced directly (not by another script) -------------------
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  echo "PROJECT_ROOT   = $PROJECT_ROOT"
  echo "CONDA_ENV      = $CONDA_ENV   (PYRUN='$PYRUN')"
  echo "RAW_DIR        = $RAW_DIR"
  echo "CLEANED_ROOT   = $CLEANED_ROOT"
  echo "DATASET_ROOT   = $DATASET_ROOT"
  echo "GH_REPO        = $GH_REPO"
  echo "RELEASE_TAG    = $DATASET_RELEASE_TAG"
  echo "EXPERIMENT_DIR = $EXPERIMENT_DIR"
  echo "CHECKPOINT     = $CHECKPOINT"
fi
