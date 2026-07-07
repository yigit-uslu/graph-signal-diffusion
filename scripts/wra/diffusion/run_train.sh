#!/bin/bash
# Thin convenience wrapper around scripts/wra/diffusion/train.sh.
#
# This script is for quick launches from torch_env with opinionated defaults:
#   - preset: medium
#   - wandb: enabled
#
# Pass-through:
#   Any CLI args are forwarded to train.sh (for example: --cfg job --resolve).
#
# Examples:
#   1) Default medium run:
#      bash scripts/wra/diffusion/run_train.sh
#   2) Resolve config only:
#      bash scripts/wra/diffusion/run_train.sh --cfg job --resolve
#   3) Override to debug preset:
#      WRA_PRESET=debug bash scripts/wra/diffusion/run_train.sh
#   4) Custom preset (expects conf/dataset/wra_large.yaml):
#      WRA_PRESET=large bash scripts/wra/diffusion/run_train.sh

WRA_PRESET="${WRA_PRESET:-medium}"
WRA_WANDB_ENABLED="${WRA_WANDB_ENABLED:-true}"

WRA_PRESET="${WRA_PRESET}" WRA_WANDB_ENABLED="${WRA_WANDB_ENABLED}" \
bash scripts/wra/diffusion/train.sh "$@"
