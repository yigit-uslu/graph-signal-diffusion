#!/bin/bash
# Evaluate GRW baseline on SP100 (default config). Run from the project root.
# To evaluate a diffusion checkpoint add:  baseline=diffusion checkpoint_path=<path>

python -m graph_signal_diffusion.cli.evaluate baseline=grw
