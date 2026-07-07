#!/bin/bash
# Canonical WRA diffusion training launcher.
#
# This script assembles Hydra overrides and calls:
#   python -m graph_signal_diffusion.cli.train task=wireless_resource_allocation ...
#
# Supported presets:
#   - debug  -> dataset=wra_debug
#   - medium -> dataset=wra_medium
#   - <name> -> dataset=wra_<name>
#
# Main environment variables:
#   WRA_PRESET         : debug | medium | <name> (default: debug)
#   WRA_CUDA_DEVICE    : CUDA device id (default: 0)
#   WRA_WANDB_ENABLED  : true | false (default: true)
#   Preset mapping expects config files like:
#     conf/dataset/wra_debug.yaml
#     conf/dataset/wra_medium.yaml
#     conf/dataset/wra_<name>.yaml
#
# Examples:
#   1) Debug preset (default):
#      bash scripts/wra/diffusion/train.sh
#   2) Medium preset:
#      WRA_PRESET=medium bash scripts/wra/diffusion/train.sh
#   3) Custom preset (expects conf/dataset/wra_large.yaml):
#      WRA_PRESET=large bash scripts/wra/diffusion/train.sh
#   4) Medium with wandb off:
#      WRA_PRESET=medium WRA_WANDB_ENABLED=false \
#      bash scripts/wra/diffusion/train.sh
#   5) Resolve config only (no training):
#      WRA_PRESET=medium bash scripts/wra/diffusion/train.sh --cfg job --resolve
#
# Uncomment the LD_LIBRARY_PATH block below only if you hit
# GLIBCXX / GLIBC / PyTorch shared-library loading issues.
# --------------------------------------------------------------------------
# if [ -z "$CONDA_PREFIX" ]; then
#     echo "Error: Conda environment not activated."
#     echo "Run: conda activate graph-signal-diffusion"
#     exit 1
# fi
# export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib/python3.11/site-packages/torch/lib:${LD_LIBRARY_PATH}"
# export PYTHONWARNINGS="ignore::UserWarning"
# --------------------------------------------------------------------------

WRA_PRESET="${WRA_PRESET:-debug}"   # maps to dataset config wra_<preset>
WRA_CUDA_DEVICE="${WRA_CUDA_DEVICE:-1}"
WRA_WANDB_ENABLED="${WRA_WANDB_ENABLED:-true}"
WRA_DATASET_CONFIG="wra_${WRA_PRESET}"

CMD=(
    python -m graph_signal_diffusion.cli.train
    task=wireless_resource_allocation
    "dataset@task.dataset=${WRA_DATASET_CONFIG}"
    "trainer.name=gdm_wra_${WRA_PRESET}"
    "wandb.enabled=${WRA_WANDB_ENABLED}"
)

# Forward any extra Hydra/CLI flags (e.g., --cfg job --resolve).
CMD+=("$@")

CUDA_VISIBLE_DEVICES="${WRA_CUDA_DEVICE}" "${CMD[@]}"
