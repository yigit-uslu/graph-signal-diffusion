#!/bin/bash
# Stage 6: Train SP500 diffusion model (no downsampling, supports static or dynamic graph).
# Run from the project root. Edit variables below to customise the run.
# source ./scripts/setup_training_env.sh  # Uncomment if you need GLIBCXX fixes.

diffusion_type="ddim"
base_channels=64
cond_embed_dim=$((2 * ${base_channels}))
batch_size=64
batch_size_val=100
optim__lr=1e-4

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

max_epochs=5000
eval_every_n_epochs=50 # This is now resolved to a multi-phase eval_schedule.

CUDA_VISIBLE_DEVICES=0 python -m graph_signal_diffusion.cli.train \
    task=stock_price_forecasting_v2 \
    dataset@task.dataset=sp500_cleaned \
    dataset.root=data/sp500/cleaned_drop_incomplete_min_coverage_0.95_corr_${corr_thresh}_sector_bonus_0.05 \
    model@task.model=ugnn_sp500_v2 \
    diffusion@task.diffusion=${diffusion_type} \
    dataset.past_window=20 \
    dataset.future_window=5 \
    dataset.batch_size=${batch_size} \
    dataset.batch_size_val=${batch_size_val} \
    dataset.dynamic_graph_period=${dynamic_graph_period} \
    dataset.dynamic_graph_edge_weight_threshold=${dynamic_graph_edge_weight_threshold} \
    dataset.dynamic_graph_edge_budget_ratio=${dynamic_graph_edge_budget_ratio} \
    trainer.optimizer.learning_rate=${optim__lr} \
    model.config.base_channels=${base_channels} \
    model.config.embedding_config.time_embed_dim=${cond_embed_dim} \
    model.config.embedding_config.cond.embed_dim=${cond_embed_dim} \
    model.config.embedding_config.cond.shared_encoder.temporal.output.mode=static \
    model.config.embedding_config.cond.block_fusion.mode=concat \
    trainer.eval_every_n_epochs=${eval_every_n_epochs} \
    trainer.max_epochs=${max_epochs} \
    wandb.enabled=True \
    # run_name="nods-${graph_type}-lr${optim__lr}-Fh${base_channels}-cond_embed_dim${cond_embed_dim}-num_layers${num_layers}-diffusion-${diffusion_type}"
