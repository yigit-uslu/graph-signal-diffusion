# Primal-Dual GNN for Wireless Power Allocation

Implementation of a primal-dual algorithm that learns power allocation policies for wireless interference networks using Graph Neural Networks (GNNs).

## Overview

This implementation solves a **constrained optimization problem** for each wireless network:

**Objective**: Maximize sum-ergodic-rate across all receivers

**Constraints**:
- Min-rate requirement: `R_i ≥ r_min_i` for each receiver i
  (scalar `r_min` or per-network/per-receiver profile)
- Max power budget: `0 ≤ p_i ≤ P_max` for each transmitter i (default: 10 mW)

The problem is high-dimensional and nonconvex due to interference coupling between users.

## Architecture

### Primal Policy (GNN)
- **Model**: `PowerAllocationGNN` - TAGConv-based GNN with residual connections
- **Input**: Graph representation of wireless network
  - Nodes = receivers
  - Self-loops weighted by direct channel gains
  - Edges weighted by interference channel gains
  - Node features: [direct_signal_strength, total_interference_potential]
- **Output**: Power allocation per transmitter (constrained to [0, P_max] via sigmoid)
- **Layers**: 3 ResidualGNNBlocks with K=2 hop TAGConv aggregation

### Dual Variables (Lagrange Multipliers)
- **Optimizer**: `DualOptimizer` - Projected subgradient ascent
- **Variables**: λ ∈ R^(N_networks × n) - one per receiver per network
- **Thresholds**: Canonical `r_min_table` with shape `(num_networks, num_receivers)`
  (scalar / per-network vector / per-network-per-receiver matrix accepted at input)
- **Update**: λ ← max(0, λ + α × (r_min[net_id, i] - R_i)) every K=10 gradient steps
- **Tracking**: Dual trajectories saved for oscillation analysis

### Training Algorithm
- **Method**: Primal-dual optimization with Lagrangian loss
- **Loss**: `L(p, λ) = -sum(R_i) + sum(λ_i × max(0, r_min_i - R_i))`
- **Primal update**: Adam optimizer on GNN parameters
- **Dual update**: Subgradient ascent on constraint violations
- **Convergence**: Joint criteria on gradient norm, dual variance, violation rate
- **Sample collection**: After convergence, collect 20 power allocations per network from oscillating trajectory

## Implementation Files

### Core Modules

1. **Graph Construction** - `src/graph_signal_diffusion/utils/graph_builder.py`
   - Extract large-scale channel gains from WirelessChannel
   - Build PyG graphs with self-loops and interference edges
   - Apply top-K sparsification ensuring minimum degree
   - Compute node features from channel statistics

2. **PowerAllocationGNN** - `src/graph_signal_diffusion/models/power_allocation_gnn.py`
   - TAGConv-based GNN (K=2 hops, L=3 layers)
   - Input projection, residual blocks, output MLP
   - Sigmoid × P_max for automatic power constraint satisfaction
   - Scatter-add to map receiver embeddings → transmitter powers

3. **DualOptimizer** - `src/graph_signal_diffusion/trainers/dual_optimizer.py`
   - Maintains per-network dual variables
   - Projected subgradient updates
   - Oscillation analysis for convergence detection
   - Checkpoint save/load support

4. **PrimalDualTrainer** - `src/graph_signal_diffusion/trainers/primal_dual_trainer.py`
   - Lagrangian loss computation over batches
   - Joint convergence checking (4 criteria)
   - Sample collection after convergence or max_epochs
   - Per-network quality analysis

5. **Rate Calculator** - `src/graph_signal_diffusion/utils/rate_calculator.py`
   - Shannon capacity computation with interference
   - Ergodic rate averaging over time
   - System parameter conversion (dBm ↔ watts)
   - Jain's fairness index

6. **Dataset** - `src/graph_signal_diffusion/datasets/wra/primal_dual_dataset.py`
   - Wraps WirelessChannel instances
   - Pre-builds graphs from large-scale gains
   - Provides instantaneous channel realizations for rate computation
   - Custom collate function for batching

7. **Training Script** - `scripts/train_primal_dual_power_allocation.py`
   - End-to-end training pipeline
   - Command-line argument parsing
   - Checkpoint and result saving

## Usage

### Basic Training

```bash
python scripts/train_primal_dual_power_allocation.py \
    --num_networks 100 \
    --n_links 10 \
    --num_timesteps 200 \
    --max_epochs 1000 \
    --r_min 0.7 \
    --checkpoint_dir checkpoints/primal_dual
```

### Advanced Configuration

```bash
python scripts/train_primal_dual_power_allocation.py \
    --num_networks 200 \
    --n_links 20 \
    --hidden_dim 128 \
    --num_layers 3 \
    --K 2 \
    --batch_size 16 \
    --learning_rate 1e-3 \
    --alpha_dual 0.01 \
    --r_min 0.7 \
    --P_max_dBm 10.0 \
    --device cuda
```

### Key Parameters

**Dataset**:
- `--num_networks`: Number of training networks (default: 100)
- `--n_links`: Number of TX-RX pairs per network (default: 10)
- `--num_timesteps`: Time steps for rate averaging (default: 200)
- `--deployment_range`: Deployment area size in meters (default: 600)

**Model**:
- `--hidden_dim`: GNN hidden dimension (default: 64)
- `--num_layers`: Number of ResidualGNNBlocks (default: 3)
- `--K`: TAGConv hop parameter (default: 2)

**Training**:
- `--max_epochs`: Maximum training epochs (default: 1000)
- `--learning_rate`: GNN optimizer learning rate (default: 1e-3)
- `--alpha_dual`: Dual step size (default: 0.01)
- `--dual_update_frequency`: Update duals every K steps (default: 10)

**Constraints**:
- `--r_min`: Minimum rate constraint in bits/s/Hz (default: 0.7)
- `--P_max_dBm`: Maximum transmit power in dBm (default: 10.0)

## Outputs

After training, the following are saved to `checkpoint_dir`:

1. **Pre-collection checkpoint** (`pre_collection_checkpoint.pt`):
   - Model state at convergence (before sample collection)
   - Dual variable states
   - Training history
   - For reproducibility

2. **Training history** (`training_history.json`):
   - Loss, sum-rate, min-rate per epoch
   - Constraint violation rates
   - Dual variable statistics
   - Gradient norms

3. **Collected samples** (`collected_samples.npz`):
   - 20 power allocations per network
   - Corresponding ergodic rates
   - Sum-rates and min-rates per sample

4. **Quality report** (`quality_report.json`):
   - Per-network constraint satisfaction rates
   - Violation severity statistics
   - Mean/min rates across samples

5. **Regular checkpoints** (`checkpoint_epoch_N.pt`):
   - Saved every 100 epochs
   - Full training state for resumption

## Convergence Criteria

Training terminates when **all** of the following are met for 10 consecutive epochs:

1. **Gradient norm**: Mean ||∇_θ L|| < 1e-4 over 50-epoch window
2. **Dual variance**: Change in std(λ) < 1% over 50-epoch window
3. **Violation rate**: Fraction of violated constraints < 5%
4. **Objective quality**: Mean sum-rate > 50% of WMMSE baseline (placeholder)

If convergence not reached by `max_epochs=1000`, samples are collected anyway.

## Sample Quality Analysis

After collection, each network's samples are analyzed:
- **Constraint satisfaction rate**: Fraction of receivers meeting r_min
- **Violation severity**: Mean (r_min - R_i)_+ when violated
- **Rate statistics**: Mean and min rates per user across samples

Networks with < 80% satisfaction may need hyperparameter tuning.

## Next Steps: Diffusion Model Training

The collected samples form a dataset for training a UGNN diffusion model:
1. Each sample is a power allocation vector
2. Samples capture the distributional optimal policy (from dual oscillations)
3. Diffusion model learns to generate diverse, constraint-satisfying allocations

## Design Choices

### Why receivers as nodes?
- Receivers aggregate interference from multiple transmitters
- Natural for computing constraint violations (min-rate is per-receiver)
- Scatter-add maps receiver embeddings → transmitter powers via associations

### Why per-network duals?
- Each network has unique topology and constraint tightness
- Independent dual updates allow personalized constraint enforcement
- Enables heterogeneous training across diverse network realizations

### Why joint convergence?
- Ensures uniform quality across all training networks
- Simplifies dataset preparation for diffusion training
- Avoids mix of converged and non-converged samples

### Why oscillating duals indicate distributional policy?
- Nonconvex problem → multiple local optima
- Periodic dual oscillations → GNN explores different solutions
- Captures uncertainty and diversity for generative modeling

## Future Enhancements

1. **WMMSE baseline**: Implement WMMSE solver for convergence threshold
2. **Adaptive step sizes**: Tune α_dual based on constraint violation severity
3. **Warm-start**: Pre-train on unconstrained sum-rate maximization
4. **Heterogeneous networks**: Support variable network sizes (m ≠ n)
5. **Distributed inference**: Batch multiple graphs efficiently in forward pass
6. **Visualization**: Plot dual trajectories, rate distributions, power allocations

## References

- **TAGConv**: Du et al. "Topology Adaptive Graph Convolutional Networks" (2017)
- **Primal-Dual Methods**: Boyd & Vandenberghe "Convex Optimization" (2004)
- **WMMSE**: Shi et al. "An iteratively weighted MMSE approach..." (2011)
- **Shannon Capacity**: Shannon "A Mathematical Theory of Communication" (1948)
