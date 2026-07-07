#!/bin/bash


#  Configuration
# Multi-network run, short training for quick testing of conversion of pd-samples to diffusion pipeline.
CUDA_VISIBLE_DEVICES=1 python scripts/train_primal_dual_power_allocation.py \
    dataset.num_networks=16 \
    dataset.n_links=50 \
    system.P_max_dBm=10.0 \
    dataset.deployment_range=700.0 \
    model.type=gnn \
    training.batch_size=4 \
    training.max_epochs=1000 \
    device=cuda \
    seed=42 \
    training.violation_rate_threshold=0.2 \
    training.dual_momentum=0.9 \
    training.alpha_dual=1.0 \
    training.dual_update_mode=step \
    training.dual_update_frequency=10 \
    training.convergence_patience=200 \
    training.convergence_window=200

# Override parameters as needed 
# python scripts/train_primal_dual_power_allocation.py model.type=mlp training.learning_rate=5e-4

# Multi-run sweep example (uncomment to use)
# python scripts/train_primal_dual_power_allocation.py -m training.alpha_dual=0.01,0.02,0.05 training.learning_rate=1e-3,5e-4