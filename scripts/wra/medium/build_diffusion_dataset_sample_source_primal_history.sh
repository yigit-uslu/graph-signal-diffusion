#!/bin/bash
# Build diffusion-ready raw WRA dataset from a completed PD run (wra_medium) using possibly whole primal history as sample source.
# Usage:
#   scripts/wra/medium/build_diffusion_dataset_wrapper.sh \
#     input_dir=outputs/wra_medium/<pd_run_key>/<date>/<time> \
#     collection.sample_source=primal_history \
#     [extra Hydra overrides...]
# This forwards the overrides to the build_diffusion_dataset_wrapper, which controls dataset generation parameters like target_samples_per_network and primal_history refinement settings.
# Note that the input_dir must point to a completed PD run with collected samples, as the script reads from the trainer_chkpts/best_models/best_model_epoch_*.pt files to extract the samples for diffusion dataset construction. The script does not re-run PD or sample collection; it only
export HYDRA_FULL_ERROR=1  # Enable full Hydra error tracebacks for easier debugging
# INPUT_DIR=outputs/wra_medium/wrpd_v1_wrach_v1_s42_D128_N200_R3200_v3_h930120ef88b8_r0.8_a0.1_h9d6b12eb1bae/2026-03-05/19-26-57
INPUT_DIR=outputs/wra_medium/wrpd_v1_wrach_v1_s42_D128_N200_R3200_v3_h930120ef88b8_r0.8_a0.1_he13d1ac3ae99/2026-03-03/18-45-05
bash ./scripts/wra/medium/build_diffusion_dataset_wrapper.sh \
    input_dir=$INPUT_DIR \
    collection.sample_source=primal_history \
    collection.primal_history.window_size=1000 \
    collection.primal_history.refine_feasible_subset=true \
    collection.target_samples_per_network=200 \
    "$@"
