# DEPRECATED: use scripts/stock/ launchers instead.
# DEPRECATED: use scripts/stock/ launchers instead.
# !/bin/bash

source ./scripts/setup_training_env.sh

# We will run static graph and dynamic graph versions of a complex model with low lr.

diffusion_type="ddim" # Switch to DDPM for better stability with larger model.
base_channels=32 # Increase model capacity
cond_embed_dim=$((4 * ${base_channels})) # Four times the base_channels
batch_size=64 # reduce the batch size to mitigate potential GPU memory issues with larger model.
batch_size_val=20
num_layers=2 # Decrease the number of layers.
K_hops=1 # Limit message passing to 2-hop neighbors to mitigate score attenuation issues with dynamic graphs, can be increased if learnable score gain proves effective
lr=1e-3 # Learning rate peaks at 1e-4 and dies down to 1e-7.
lr_scheduler__name="cosine" # Linear warmup + linear decay to zero.
lr_scheduler__decay_fraction=0.5
dropout=0.05 # A lot of dropout

dynamic_graph_period=21  # Period in trading days for dynamic graph updates (null = static only)

corr_thresh=0.7 # static graph
dynamic_graph_edge_weight_threshold=0.7   # Edge weight threshold for dynamic graph construction. Tune this to balance graph connectivity and noise. 0.2 is a starting point based on preliminary experiments, but it may require adjustment based on the distribution of correlation values in the cleaned SP500 dataset.
dynamic_graph_edge_budget_ratio=1.0 # Cap dynamic graph edges to at most the same number as the static graph. This prevents overly dense dynamic graphs that can cause score attenuation and training instability, while still allowing flexibility in edge selection based on correlation strength.
# Set graph_type to 'static' if dynamic_graph_period is null, 'dynamic' otherwise
if [ "$dynamic_graph_period" == "null" ]; then
    graph_type=static-graph
    dynamic_graph_edge_weight_threshold=null # Ensure this is null when dynamic graphs are disabled
    dynamic_graph_edge_budget_ratio=null # Ensure this is null when dynamic graphs are disabled
else
    graph_type=dynamic-graph-${dynamic_graph_period}-${dynamic_graph_edge_weight_threshold}-budget_ratio-${dynamic_graph_edge_budget_ratio}
fi

max_epochs=5000 # Set a high max_epochs to allow training to run until convergence, but we will rely on early stopping based on validation performance to prevent overfitting and unnecessary long training times.
eval_every_n_epochs=50 # Increase evaluation frequency to better monitor training progress without dynamic graphs, can be reduced if it causes too much overhead


echo "Training UGNN with cleaned SP500 dataset, base_channels=${base_channels}, cond_embed_dim=${cond_embed_dim}, num_layers=${num_layers}, lr=${lr}"
echo "Graph type: ${graph_type}"
# diagnose_selector_every_n_epochs=50 # Increase selector diagnosis frequency to monitor selector behavior without dynamic graphs, can be reduced if it causes too much overhead
# learnable_score_gain=true # Enable learnable score gain to help mitigate potential score attenuation
# score_gain_per_level=\[1.0,10.0,100.0\] # Warm-start gain values to offset deep-level score attenuation, can be fine-tuned if learnable_score_gain is true
# entropy_reg_depth_scale=1.0 # Even more aggressive than before. Scale down entropy regularization weight for deeper selectors to allow more confident selection when scores are attenuated
# selector_temperature_schedule__warmup_steps=250 # previously 1000
# selector_temperature_schedule__anneal_steps=7500 # previously 20000.

CUDA_VISIBLE_DEVICES=1 python -m graph_signal_diffusion.cli.train \
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
    wandb.enabled=True \
    trainer.learning_rate=${lr} \
    trainer.lr_scheduler.name=${lr_scheduler__name} \
    trainer.lr_scheduler.decay_fraction=${lr_scheduler__decay_fraction} \
    model.config.base_channels=${base_channels} \
    model.config.embedding_config.time_embed_dim=${cond_embed_dim} \
    model.config.embedding_config.cond_embed_dim=${cond_embed_dim} \
    model.config.gnn_config.K=${K_hops} \
    model.config.gnn_config.num_layers=${num_layers} \
    model.config.gnn_config.dropout=${dropout} \
    trainer.eval_every_n_epochs=${eval_every_n_epochs} \
    trainer.max_epochs=${max_epochs} \
    run_name="nods-${graph_type}-lr${lr}-Fh${base_channels}-cond_embed_dim${cond_embed_dim}-num_layers${num_layers}-diffusion-${diffusion_type}"
