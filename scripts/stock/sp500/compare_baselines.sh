#!/bin/bash
# Compare GRW and diffusion baselines side-by-side on SP500-cleaned.
# Requires a trained diffusion checkpoint. Run from the project root.

export HYDRA_FULL_ERROR=1

# experiment_name=nods-dynamic-graph-21-0.7-budget_ratio-1.0-lr1e-4-Fh64-cond_embed_dim256-num_layers3-diffusion-ddim
# experiment_dir=outputs/stock_price_forecasting_v2-sp500_cleaned/ugnn_sp500_v2-ddim-gdm_sp500/$experiment_name
# checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_899.pt

experiment_name=lush-jacamar-155
experiment_dir=outputs/stock_price_forecasting_v2-sp500_cleaned/ugnn_sp500_v2-ddim-gdm_sp500/$experiment_name
checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_399.pt

batch_size=100
batch_size_val=50
n_samples=50 # 20

CUDA_VISIBLE_DEVICES=1 python -m graph_signal_diffusion.cli.compare_baselines \
    --config-name compare_baselines_sp500 \
    baselines_to_compare='[grw,diffusion]' \
    baselines.diffusion.checkpoint_path=$checkpoint_path \
    baselines.diffusion.n_samples=$n_samples \
    ++baselines.diffusion.use_amp=true \
    baselines.grw.n_samples=$n_samples \
    dataset.batch_size=$batch_size \
    dataset.batch_size_val=$batch_size_val \
    dataset.n_samples_per_input=1 \
    dataset.past_window=20 \
    dataset.future_window=5 \
    'eval_splits={test: 1}' \
    wandb.enabled=False

# Diffusion-override examples:
# Halve sampling steps + deterministic:
#   "++baselines.diffusion.diffusion_overrides={sampling_timesteps: 50, ddim_eta: 0.0}"
# Full DDPM (all 500 steps):
#   "++baselines.diffusion.diffusion_overrides={_target_: graph_signal_diffusion.diffusion.ddpm.DDPM}"
