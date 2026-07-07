#!/bin/bash
# Shared configuration for the rigorous-quoll-131 (WRA / TSP) reproduction pipeline.
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
# WRA uses `graph-signal-diffusion` (this project's env for wireless work).
export CONDA_ENV="${CONDA_ENV:-graph-signal-diffusion}"
export PYRUN="${PYRUN:-conda run -n ${CONDA_ENV} python}"

# The env libstdc++ must be found for matplotlib (CXXABI_1.3.15 / GLIBCXX_3.4.29).
# Prepend the env lib dir (NOT $CONDA_PREFIX) — required by the WRA plotting CLIs.
export LD_LIBRARY_PATH="${HOME}/miniconda3/envs/${CONDA_ENV}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Reduce allocator fragmentation for sampling-heavy eval (n_samples cloning).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# --- Experiment identity (the rigorous-quoll-131 run) -----------------------
export EXPERIMENT_NAME="rigorous-quoll-131"
export TASK="wireless_resource_allocation"
export MODEL="ugnn_wra_v3_ds8_norm_act_head"
export TRAINER="trainer_wra_v3_learned_ds4"
# The all-density (single r_min=0.6) dataset config — 4 sub-datasets, model_cond_channels=2.
export DATASET_CFG="${DATASET_CFG:-wra_medium-large_outdoor_all_density}"
export TRAINER_NAME="gdm_wra_medium-large_outdoor_all_density"
export EXPERIMENT_DIR="outputs/${TASK}-wra/${MODEL}-ddim_wra-${TRAINER_NAME}/${EXPERIMENT_NAME}"

# --- Dataset root (REFERENCED, not bundled) ---------------------------------
# The 4 diffusion sub-datasets the run consumes (4 densities x 32 networks x 200
# samples at r_min=0.6) total ~39 GB and are NOT shipped in git; the full data/wra
# tree (with the PD/channel regeneration intermediates) is ~136 GB. cli.test
# resolves the dataset at <repo>/data/wra, so keep DATASET_ROOT=data/wra (symlink
# your copy into place if it lives elsewhere: `ln -s /path/to/wra data/wra`). The
# 4 content-addressed sub-dataset dirs are pinned in
#   src/graph_signal_diffusion/conf/dataset/wra_medium-large_outdoor_all_density.yaml
# and listed in checksums/dataset_manifest.txt. See §Dataset in README.md.
export DATASET_ROOT="${DATASET_ROOT:-data/wra}"

# --- Public dataset hosting (shared 'gsd-dataset' GitHub repo) ---------------
# The ~39 GB dataset is published as split tar parts on a PUBLIC GitHub Release in
# a SHARED data repo (one release TAG per dataset; SP500 etc. can coexist under
# their own tags). Because the repo is public, consumers download anonymously with
# plain curl — no GitHub account or token. Both download_dataset.sh (consumer) and
# package_dataset.sh (owner) read these values, so the tag lives in ONE place.
#
# The tag is "<stem>-<bundle id>". The bundle id (85faf506ec70) is a content
# fingerprint of the 4 sub-datasets: md5[:12] over their sorted _h<hash> tokens.
# It is DERIVED + VERIFIED by dataset_bundle_id.sh and documented in DATASET_TAG.md;
# `dataset_bundle_id.sh --check` asserts the literal below still matches the data.
# Rebuild the dataset -> recompute (dataset_bundle_id.sh) -> bump the id here.
export GH_REPO="${GH_REPO:-yigit-uslu/gsd-dataset}"
export DATASET_TAG_STEM="${DATASET_TAG_STEM:-wra-N400-gsd}"
export DATASET_RELEASE_TAG="${DATASET_RELEASE_TAG:-${DATASET_TAG_STEM}-85faf506ec70}"

# --- Checkpoint (rank-1 best model, ep1600 — what test_summary.json used) ----
# Prefer the 8 MB copy bundled in this folder (self-contained; committed via
# REGULAR git — .pt is not LFS-tracked); fall back to the canonical training
# output if the bundle is absent. An explicit CHECKPOINT=... env override wins.
export CHECKPOINT_EPOCH="${CHECKPOINT_EPOCH:-1600}"
_REPRO_REL="${REPRO_DIR#$PROJECT_ROOT/}"
_BUNDLED_CKPT="${_REPRO_REL}/checkpoint/best_model_epoch_${CHECKPOINT_EPOCH}.pt"
_CANONICAL_CKPT="${EXPERIMENT_DIR}/trainer_chkpts/best_models/best_model_epoch_${CHECKPOINT_EPOCH}.pt"
if [ -z "${CHECKPOINT:-}" ]; then
  if [ -f "$_BUNDLED_CKPT" ]; then CHECKPOINT="$_BUNDLED_CKPT"; else CHECKPOINT="$_CANONICAL_CKPT"; fi
fi
export CHECKPOINT

# --- Hydra config dir (for cli.test --config-dir) ---------------------------
# cli.test rebuilds model/diffusion/task/loaders from the run's saved .hydra/.
# Prefer the bundled copy (self-contained), fall back to the canonical run dir.
_BUNDLED_HYDRA="${_REPRO_REL}/config/.hydra"
_CANONICAL_HYDRA="${EXPERIMENT_DIR}/.hydra"
if [ -z "${CONFIG_DIR:-}" ]; then
  if [ -f "$_BUNDLED_HYDRA/config.yaml" ]; then CONFIG_DIR="$_BUNDLED_HYDRA"; else CONFIG_DIR="$_CANONICAL_HYDRA"; fi
fi
export CONFIG_DIR

# --- GPU --------------------------------------------------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# --- Banner when sourced directly (not by another script) -------------------
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  echo "PROJECT_ROOT   = $PROJECT_ROOT"
  echo "CONDA_ENV      = $CONDA_ENV   (PYRUN='$PYRUN')"
  echo "EXPERIMENT_DIR = $EXPERIMENT_DIR"
  echo "DATASET_CFG    = $DATASET_CFG"
  echo "DATASET_ROOT   = $DATASET_ROOT"
  echo "GH_REPO        = $GH_REPO"
  echo "RELEASE_TAG    = $DATASET_RELEASE_TAG"
  echo "CHECKPOINT     = $CHECKPOINT"
  echo "CONFIG_DIR     = $CONFIG_DIR"
fi
