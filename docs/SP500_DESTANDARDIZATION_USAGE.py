"""
Example: How to use the simplified destandardization API in trainer/evaluation.

The SP500 dataset now stores only target-specific standardization stats and
provides a simple static method for destandardization during evaluation.
"""

import torch
from graph_signal_diffusion.datasets.sp500.dataset import SP500Stocks

# ============================================================================
# SETUP: Dataset initialization (done once)
# ============================================================================

# Assume dataset is already created with per_stock_stats
# (this happens in the datamodule during build_datasets)
dataset = ...  # Your SP500Stocks dataset instance

# Get target standardization stats (done once, store in task/trainer)
target_stats = dataset.get_target_standardization_stats()

print(f"Target: {target_stats['target_name']}")
print(f"Per-stock means shape: {target_stats['mean'].shape}")  # [N_stocks,]
print(f"Per-stock stds shape: {target_stats['std'].shape}")    # [N_stocks,]
print(f"Scale factor: {target_stats['scale_factor']}")         # 100.0


# ============================================================================
# TRAINING: Model receives standardized data
# ============================================================================

# During training, get() returns standardized targets
sample = dataset.get(idx)
x = sample.x                    # [N, T_past, F] - features (already standardized)
y_standardized = sample.y       # [N, T_future, 1] - target (standardized)

# Model is trained on standardized data
predictions_standardized = model(x)  # [N, T_future, 1]

# Loss computed on standardized scale
loss = loss_fn(predictions_standardized, y_standardized)


# ============================================================================
# EVALUATION: Destandardize predictions and targets
# ============================================================================

# At evaluation time, destandardize both predictions and targets
# to compute metrics on original scale

with torch.no_grad():
    # Get standardized predictions from model
    predictions_standardized = model(x)  # [N, T_future, 1] or [batch, N, T_future, 1]
    
    # Destandardize predictions using static method
    predictions_raw = SP500Stocks.destandardize_target(
        predictions_standardized, 
        target_stats
    )
    
    # Destandardize ground truth targets
    targets_raw = SP500Stocks.destandardize_target(
        y_standardized,
        target_stats
    )
    
    # Now compute metrics on raw scale
    # predictions_raw and targets_raw are in original units (percentage log returns)
    mae = torch.abs(predictions_raw - targets_raw).mean()
    mse = ((predictions_raw - targets_raw) ** 2).mean()
    
    print(f"MAE (raw scale): {mae:.4f}%")
    print(f"RMSE (raw scale): {torch.sqrt(mse):.4f}%")


# ============================================================================
# BATCH OPERATIONS: Destandardize works with batched data
# ============================================================================

# destandardize_target handles various tensor shapes:
# - [N, T]: per-stock, per-timestep
# - [N, T, 1]: per-stock, per-timestep, single feature
# - [batch, N, T]: batched predictions
# - [batch, N, T, 1]: batched predictions with feature dim

batch_predictions = torch.randn(32, 466, 5, 1)  # [batch, N_stocks, T_future, 1]
batch_predictions_raw = SP500Stocks.destandardize_target(
    batch_predictions,
    target_stats
)
print(f"Batch destandardized shape: {batch_predictions_raw.shape}")


# ============================================================================
# SUMMARY
# ============================================================================

"""
Benefits of the new API:

1. Simplified: Only target stats are stored (not full per_stock_stats dict)
2. Efficient: Static method, no need for dataset instance during evaluation
3. Flexible: Works with various tensor shapes (handles batching automatically)
4. Clear: Single method call, explicit about what's being destandardized
5. Separation: Features standardized once in preprocessing, never need destandardization
              Target standardized at runtime, destandardized for evaluation

Usage pattern:
--------------
1. Setup: stats = dataset.get_target_standardization_stats()
2. Training: work with standardized data from dataset.get()
3. Evaluation: raw = SP500Stocks.destandardize_target(standardized, stats)
"""
