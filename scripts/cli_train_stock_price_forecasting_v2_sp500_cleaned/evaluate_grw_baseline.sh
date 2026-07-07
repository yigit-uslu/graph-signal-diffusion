# DEPRECATED: use scripts/stock/ launchers instead.
# DEPRECATED: use scripts/stock/ launchers instead.
#!/bin/bash

rm -rf src/**/__pycache__/*.pyc

export HYDRA_FULL_ERROR=1  # Enable full Hydra error tracebacks for easier debugging

# # Basic evaluation with default parameters
# CUDA_VISIBLE_DEVICES=1 python -m graph_signal_diffusion.cli.compare_baselines \
#     --config-name compare_baselines_sp500

batch_size=64
batch_size_val=64
# Override dataset parameters (batch sizes, windows, etc.)
# IMPORTANT: dataset.n_samples_per_input=1 because GRW handles its own
# multi-sample replication internally (via predict_ensemble / _replicate_metadata).
# Without this override, ${trainer.n_samples_per_input} in sp500.yaml fails
# (no trainer section in compare_baselines), and if it DID resolve to >1 the
# ReplicatedDataset would duplicate the data BEFORE GRW adds its own copies,
# causing double-replication (n × n).
CUDA_VISIBLE_DEVICES=0 conda run -n torch_env python -m graph_signal_diffusion.cli.compare_baselines \
    --config-name compare_baselines_sp500 \
    dataset.batch_size=$batch_size \
    dataset.batch_size_val=$batch_size_val \
    dataset.n_samples_per_input=1 \
    baselines.grw.n_samples=20 \
    dataset.past_window=20 \
    dataset.future_window=5 \
    wandb.enabled=False

# # Override GRW baseline parameters
# CUDA_VISIBLE_DEVICES=1 conda run -n torch_env python -m graph_signal_diffusion.cli.compare_baselines \
#     --config-name compare_baselines_sp500 \
#     dataset.n_samples_per_input=1 \
#     baselines.grw.shrinkage_strength=100 \
#     baselines.grw.n_samples=50

# # Override both dataset and baseline parameters
# CUDA_VISIBLE_DEVICES=1 conda run -n torch_env python -m graph_signal_diffusion.cli.compare_baselines \
#     --config-name compare_baselines_sp500 \
#     dataset.n_samples_per_input=1 \
#     dataset.batch_size_val=8 \
#     baselines.grw.shrinkage_strength=1000 \
#     baselines.grw.n_samples=20 \
#     task.n_samples_per_input=20