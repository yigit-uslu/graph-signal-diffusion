#!/bin/bash

# Train primal-dual power allocation GNN on large (200-link) networks
# using subnetwork merging strategy for reliable deployment

# Configuration for 200-link networks via merging
# - 4 subnetworks of 50 links each
# - Same deployment_range per subnet as the working 50-link case
# - Circular layout with 150m spacing between subnets

python scripts/train_primal_dual_power_allocation_merged.py \
    dataset.num_networks=32 \
    dataset.n_links_per_subnet=50 \
    dataset.num_subnets=4 \
    dataset.deployment_range=700.0 \
    dataset.subnet_spacing=150.0 \
    dataset.layout=circular \
    training.batch_size=8 \
    training.learning_rate=5e-4 \
    training.max_epochs=20000 \
    system.P_max_dBm=10.0 \
    system.bandwidth_Hz=10e6 \
    system.noise_psd_dBm_Hz=-174.0
