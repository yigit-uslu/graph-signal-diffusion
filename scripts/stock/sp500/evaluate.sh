#!/bin/bash
# Evaluate a trained SP500 diffusion checkpoint.
# Edit experiment_name and checkpoint_path to match your run. Run from project root.

export HYDRA_FULL_ERROR=1

experiment_name=nods-dynamic-graph-21-0.7-budget_ratio-1.0-lr1e-4-Fh64-cond_embed_dim256-num_layers3-diffusion-ddim
experiment_dir=outputs/stock_price_forecasting_v2-sp500_cleaned/ugnn_sp500_v2-ddim-gdm_sp500/$experiment_name
checkpoint_path=$experiment_dir/trainer_chkpts/best_models/best_model_epoch_899.pt

CUDA_VISIBLE_DEVICES=0 python -m graph_signal_diffusion.cli.evaluate \
    baseline=diffusion \
    dataset=sp500_cleaned \
    task=stock_price_forecasting_v2 \
    checkpoint_path=$checkpoint_path \
    baseline.n_samples=20 \
    ++baseline.use_amp=true \
    dataset.batch_size=100 \
    dataset.batch_size_val=100 \
    eval_splits.test=5

# Possible overrides:
# Evaluate on both val and test:  eval_splits.val=null eval_splits.test=null
# Cap batches for quick sanity check:  eval_splits.test=5
# Use W&B logging:  wandb.enabled=true wandb.project=my-project
