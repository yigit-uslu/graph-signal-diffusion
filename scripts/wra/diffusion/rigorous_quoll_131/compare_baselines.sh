#!/bin/bash
# TENTATIVE compare-baselines launcher for the merged U-GNN arm rigorous-quoll-131
# (ugnn_wra_v3_ds8_norm_act_head, all-density @ single r_min=0.6, model_cond_channels=2).
#
# Target paper-ready figures:
#   (i)  Cumulative ergodic percentile-rate evolution  — main comparison, test split.
#        -> <comparison_dir>/wra_percentile_evolution_test.pdf
#           (+ task_rate_cdf ergodic-rate CDF). On by default; no flag needed.
#   (ii) Size transferability figure                   — transferability_sweep phase.
#        -> <comparison_dir>/wra_percentile_evolution_transfer_size_*.pdf
#           (plus per-target and per-density variants). Requires the phase ON.
#
# NO r_min-sweep phase here: this arm is trained at a single fixed r_min=0.6
# (cf. sophisticated-oarfish-9, which swept r_min). r_min_sweep stays disabled.
#
# Default run (FP + AP + Diffusion on test, + size transferability):
#   bash scripts/wra/diffusion/rigorous_quoll_131/compare_baselines.sh
#   # ... or evaluate a different checkpoint by epoch (best-model OR periodic, note 3):
#   CHECKPOINT_EPOCH=2400 bash scripts/wra/diffusion/rigorous_quoll_131/compare_baselines.sh
#   # late-training periodic snapshots, two cards in parallel:
#   GPU_ID=0 CHECKPOINT_EPOCH=4000 bash scripts/wra/diffusion/rigorous_quoll_131/compare_baselines.sh &
#   GPU_ID=1 CHECKPOINT_EPOCH=5000 bash scripts/wra/diffusion/rigorous_quoll_131/compare_baselines.sh &
#
# Validate config composition first (use --cfg job WITHOUT --resolve; --resolve
# eagerly evaluates the runtime-only 'revin_alpha' resolver and errors):
#   bash scripts/wra/diffusion/rigorous_quoll_131/compare_baselines.sh --cfg job
#
# ─────────────────────────────────────────────────────────────────────────────
# NOTES (verified against the transfer cache + transferability code):
#   1. Transfer artifacts are REUSED, not regenerated. The channel cache
#      (data/wra_channel_cache_transfer/, all 24 targets) is keyed by scenario +
#      channel params — independent of the model and of cond_channels. The raw
#      transfer datasets (data/wra_transfer/transfer_*_n8_r0.60/raw/) are reused
#      when present (_wra_transferability.py:722). Both were already built by
#      sophisticated-oarfish-9 for these exact targets at r_min=0.6 — so there is
#      NO channel generation and NO expert-sample cloning (transfer eval is
#      expert-free: y placeholder + reference_policy=none). The sweep is just
#      DDIM sampling + FP on the already-cached targets.
#   2. cond_channels=2 is auto-handled. Target conditioning is built in-memory per
#      the model's cond_channels — r_min is appended ONLY when ==3
#      (_wra_transferability.py:550), so oarfish-9's on-disk 3-channel processed/
#      graphs are never reused here. cond_channels is resolved by walking up from
#      the checkpoint to its .hydra/config.yaml (compare_baselines.py:992 ->
#      dataset.model_cond_channels=2). No override needed; just keep
#      rigorous-quoll-131/.hydra/config.yaml in place.
#   3. Checkpoint selection. Training is COMPLETE (5000 epochs). Choose a top-5
#      best-model checkpoint with CHECKPOINT_EPOCH (default 1600 = the run's #1 and
#      the frozen best_models/best_model.pt). The leaderboard composite is
#      0.5·mean_viol_gap% + 0.25·1%_rate_gap% + 0.25·5%_rate_gap% (lower=better,
#      meaned over the 4 densities); all five live in
#      trainer_chkpts/best_models/best_model_epoch_<E>.pt:
#          rank 1  epoch 1600  composite 3.87   <- DEFAULT (best fairness/throughput)
#          rank 2  epoch 2400  composite 4.74   (most fairness-leaning of the five)
#          rank 3  epoch 1650  composite 4.93
#          rank 4  epoch 2200  composite 4.97
#          rank 5  epoch 3300  composite 6.14
#      The composite is tail-weighted: later epochs trade worst-user (1%/5%/min)
#      rate for higher sum-rate, so 1600 is the best operating point for this task.
#      LATE-TRAINING periodic snapshots (the only checkpoints past 3300) are also
#      selectable by epoch: CHECKPOINT_EPOCH falls back to trainer_chkpts/
#      DDIM_epoch_<E>.pt for E in {1000,2000,3000,4000,5000}. The final 5000 beats
#      the expert sum-rate by ~2.7% but degrades rate@1% (val composite 8.79); 4000
#      sits in the degraded plateau. A full CHECKPOINT_PATH always overrides.
#   4. GPU. Defaults to GPU 1 (set GPU_ID=0 for the other card); training is done so
#      both are free. Output goes to a labeled, collision-free dir
#      (<root>/comparison/<date>/<time>_<exp>_epoch-<E>) so parallel per-epoch runs
#      don't clash; override the location with RUN_DIR=... .
#
# Optional fast smoke before the full 24-target sweep (one small target):
#   bash scripts/wra/diffusion/rigorous_quoll_131/compare_baselines.sh \
#     '++transferability_sweep.targets=[{scenario: wra_large_outdoor_high_density, eval_num_networks: 2, batch_size_val: 200}]'
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
export HYDRA_FULL_ERROR=1

# Ensure the *env* libstdc++ is found (CXXABI_1.3.15 for matplotlib).
export LD_LIBRARY_PATH="${HOME}/miniconda3/envs/graph-signal-diffusion/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

GPU_ID="${GPU_ID:-1}"
CONFIG_NAME="${CONFIG_NAME:-compare_baselines_wra}"
DATASET_CFG="${DATASET_CFG:-wra_medium-large_outdoor_all_density}"
N_SAMPLES_PER_NETWORK="${N_SAMPLES_PER_NETWORK:-100}"
CONDA_ENV="${CONDA_ENV:-graph-signal-diffusion}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-rigorous-quoll-131}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-outputs/wireless_resource_allocation-wra/ugnn_wra_v3_ds8_norm_act_head-ddim_wra-gdm_wra_medium-large_outdoor_all_density/${EXPERIMENT_NAME}}"
# CHECKPOINT_EPOCH resolves to a best-model leaderboard checkpoint when one exists
# (1600/1650/2200/2400/3300), else a periodic snapshot trainer_chkpts/DDIM_epoch_<E>.pt
# (1000/2000/3000/4000/5000 — the only late-training checkpoints past 3300). A full
# CHECKPOINT_PATH always wins.
CHECKPOINT_EPOCH="${CHECKPOINT_EPOCH:-3300}"
_best_ckpt="${EXPERIMENT_DIR}/trainer_chkpts/best_models/best_model_epoch_${CHECKPOINT_EPOCH}.pt"
_ddim_ckpt="${EXPERIMENT_DIR}/trainer_chkpts/DDIM_epoch_${CHECKPOINT_EPOCH}.pt"
if [[ -n "${CHECKPOINT_PATH:-}" ]]; then
  :  # explicit full path wins
elif [[ -f "${_best_ckpt}" ]]; then
  CHECKPOINT_PATH="${_best_ckpt}"
elif [[ -f "${_ddim_ckpt}" ]]; then
  CHECKPOINT_PATH="${_ddim_ckpt}"
else
  echo "No checkpoint for epoch ${CHECKPOINT_EPOCH}:" >&2
  echo "  tried ${_best_ckpt}" >&2
  echo "  tried ${_ddim_ckpt}" >&2
  echo "  (best_models: 1600 1650 2200 2400 3300 | periodic DDIM: 1000 2000 3000 4000 5000)" >&2
  exit 1
fi
if [[ ! -f "${CHECKPOINT_PATH}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT_PATH}" >&2
  exit 1
fi

# Labeled, collision-free output dir so parallel per-epoch runs are identifiable
# (e.g. <root>/2026-06-10/14-05-22_rigorous-quoll-131_epoch-4000). Override with RUN_DIR.
COMPARISON_ROOT="${COMPARISON_ROOT:-outputs/wireless_resource_allocation-wra/comparison}"
RUN_DIR="${RUN_DIR:-${COMPARISON_ROOT}/$(date +%Y-%m-%d)/$(date +%H-%M-%S)_${EXPERIMENT_NAME}_epoch-${CHECKPOINT_EPOCH}}"

cmd=(
  conda run -n "${CONDA_ENV}" python -m graph_signal_diffusion.cli.compare_baselines
  --config-name="${CONFIG_NAME}"
  dataset="${DATASET_CFG}"
  dataset@task.dataset="${DATASET_CFG}"
  baselines_to_compare='[fp,ap,diffusion]'
  baselines.diffusion.checkpoint_path="${CHECKPOINT_PATH}"
  n_samples_per_network="${N_SAMPLES_PER_NETWORK}"

  # ── (i) cumulative ergodic percentile-rate + CDF: on by default, kept explicit ──
  plot_style.plots.percentile_evolution_comparison.enabled=true
  plot_style.plots.task_rate_cdf.enabled=true

  # ── No r_min sweep for this single-r_min arm ──
  r_min_sweep.enabled=false

  # ── (ii) size/density transferability sweep (targets default to r_min=0.6) ──
  transferability_sweep.enabled=true
)

# Use the labeled run dir unless the caller passed their own hydra.run.dir.
_user_rundir=false
for _a in "$@"; do [[ "${_a}" == hydra.run.dir=* ]] && _user_rundir=true; done
if [[ "${_user_rundir}" == false ]]; then
  cmd+=( "hydra.run.dir=${RUN_DIR}" )
fi

# Forward extra Hydra/CLI args (e.g. --cfg job --resolve, or target subsets).
if [[ "$#" -gt 0 ]]; then
  cmd+=("$@")
fi

echo "Launching compare_baselines for ${EXPERIMENT_NAME} (epoch ${CHECKPOINT_EPOCH}) on GPU ${GPU_ID}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
if [[ "${_user_rundir}" == false ]]; then
  echo "Output dir: ${RUN_DIR}"
else
  echo "Output dir: <caller-supplied hydra.run.dir>"
fi
echo "Dataset:    ${DATASET_CFG}  (n_samples_per_network=${N_SAMPLES_PER_NETWORK})"
echo "Phases:     baseline comparison (test) + size transferability  |  r_min sweep DISABLED"

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${cmd[@]}"
