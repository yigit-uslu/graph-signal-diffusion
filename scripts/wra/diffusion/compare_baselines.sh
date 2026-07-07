#!/bin/bash
# Compare full-power and WMMSE baselines on the WRA task.
# Uncomment diffusion baseline in compare_baselines_wra if you have already evaluated it and want to include it in the comparison.
export HYDRA_FULL_ERROR=1

experiment_name=amaranth-bear-829
experiment_dir=outputs/wireless_resource_allocation-wra/ugnn_wra-ddim_wra-gdm_wra_debug/$experiment_name
checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_3949.pt

CUDA_VISIBLE_DEVICES=1 python -m graph_signal_diffusion.cli.compare_baselines \
  --config-name=compare_baselines_wra \
  baselines.diffusion.checkpoint_path=${checkpoint_path} \
  baselines.diffusion.n_samples=100
  
