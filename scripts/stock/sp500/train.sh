#!/bin/bash
# Stage 6: Train SP500 diffusion model (no downsampling, supports static or dynamic graph).
# Run from the project root. Edit variables below to customise the run.
# source ./scripts/setup_training_env.sh  # Uncomment if you need GLIBCXX fixes.

model="ugnn_sp500_v3"

diffusion_type="ddim"
base_channels=64
cond_embed_dim=$((2 * ${base_channels}))
batch_size=32 # Reduced from 64 to 32 for v3 due to increased memory usage from temporal cross-attention and/or temporal projection. Note that the optimal batch size may differ across model variants, so this is not necessarily a regression compared to v2.
batch_size_val=200

dynamic_graph_period=21    # Set to null for static-graph only.
corr_thresh=0.7
dynamic_graph_edge_weight_threshold=0.7
dynamic_graph_edge_budget_ratio=1.0

if [ "$dynamic_graph_period" == "null" ]; then
    graph_type=static-graph
    dynamic_graph_edge_weight_threshold=null
    dynamic_graph_edge_budget_ratio=null
else
    graph_type=dynamic-graph-${dynamic_graph_period}-${dynamic_graph_edge_weight_threshold}-budget_ratio-${dynamic_graph_edge_budget_ratio}
fi

max_epochs=2000
optim__lr=1e-5 # used to be 1e-4
lr_scheduler__warmup_ratio=0.001
use_amp=True
eval_every_n_epochs=50 # This is now resolved to a multi-phase eval_schedule.

CUDA_VISIBLE_DEVICES=0 python -m graph_signal_diffusion.cli.train \
    task=stock_price_forecasting_v2 \
    dataset@task.dataset=sp500_cleaned \
    dataset.root=data/sp500/cleaned_drop_incomplete_min_coverage_0.95_corr_${corr_thresh}_sector_bonus_0.05 \
    model@task.model=${model} \
    diffusion@task.diffusion=${diffusion_type} \
    dataset.past_window=20 \
    dataset.future_window=5 \
    dataset.batch_size=${batch_size} \
    dataset.batch_size_val=${batch_size_val} \
    dataset.dynamic_graph_period=${dynamic_graph_period} \
    dataset.dynamic_graph_edge_weight_threshold=${dynamic_graph_edge_weight_threshold} \
    dataset.dynamic_graph_edge_budget_ratio=${dynamic_graph_edge_budget_ratio} \
    trainer.optimizer.learning_rate=${optim__lr} \
    trainer.lr_scheduler.warmup_ratio=${lr_scheduler__warmup_ratio} \
    model.config.base_channels=${base_channels} \
    model.config.embedding_config.time_embed_dim=${cond_embed_dim} \
    model.config.embedding_config.cond.embed_dim=${cond_embed_dim} \
    trainer.eval_every_n_epochs=${eval_every_n_epochs} \
    trainer.max_epochs=${max_epochs} \
    trainer.use_amp=${use_amp} \
    wandb.enabled=True

    # run_name="nods-${graph_type}-lr${optim__lr}-Fh${base_channels}-cond_embed_dim${cond_embed_dim}-num_layers${num_layers}-diffusion-${diffusion_type}"
