#!/bin/bash
# Load and verify collected PD expert samples for the wra_debug scenario.
# Edit --output-dir to point to the desired PD training run output directory.

CUDA_VISIBLE_DEVICES=1 python -m graph_signal_diffusion.cli.wra.verify_pd_samples \
    --output-dir outputs/wra_debug/<pd_run_key>/<date>/<time> \
    --verbose --skip-network-graph-viz
