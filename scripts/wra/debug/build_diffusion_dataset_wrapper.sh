#!/bin/bash
# Build diffusion-ready raw WRA dataset from a completed PD run (wra_debug).
# Usage:
#   scripts/wra/debug/build_diffusion_dataset.sh \
#     input_dir=outputs/wra_debug/<pd_run_key>/<date>/<time> \
#     [extra Hydra overrides...]

python -m graph_signal_diffusion.cli.wra.build_diffusion_dataset \
    --config-name=pd_collection/wra_debug \
    "$@"