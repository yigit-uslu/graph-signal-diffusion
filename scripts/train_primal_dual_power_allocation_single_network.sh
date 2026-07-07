#!/bin/bash
#  Configuration
# batch_size = 16
# n_links = 50
# num_timesteps = 200
# P_max_dBm = 10.0
# deployment_range = 700.0  # Reduced from 1000m for higher interference
# seed_start = 42
# Basic single network run
CUDA_VISIBLE_DEVICES=1 python scripts/train_primal_dual_power_allocation.py \
    dataset.num_networks=1 \
    dataset.n_links=50 \
    system.P_max_dBm=10.0 \
    dataset.deployment_range=700.0 \
    model.type=gnn \
    training.batch_size=1 \
    training.max_epochs=10000 \
    device=cuda \
    seed=42 \
    training.violation_rate_threshold=0.2 \
    training.dual_momentum=0.9 \
    training.alpha_dual=1.0 \
    training.dual_update_frequency=10

# Override parameters as needed 
# python scripts/train_primal_dual_power_allocation.py model.type=mlp training.learning_rate=5e-4

# Multi-run sweep example (uncomment to use)
# python scripts/train_primal_dual_power_allocation.py -m training.alpha_dual=0.01,0.02,0.05 training.learning_rate=1e-3,5e-4