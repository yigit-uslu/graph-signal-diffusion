"""
SP500 Target Destandardization Integration - Full Pipeline

This document shows how the simplified destandardization API integrates with
the training pipeline, maintaining backwards compatibility.
"""

# ============================================================================
# 1. DATASET INITIALIZATION (datamodule.py)
# ============================================================================

# Dataset stores target-specific stats
from graph_signal_diffusion.datasets.sp500.dataset import SP500Stocks

dataset = SP500Stocks(
    root='data/sp500/cleaned_...',
    per_stock_stats=per_stock_stats,  # Computed by builder
    feature_names=feature_names,
    standardized_features=[...],
    target_column_name='DailyLogReturn',
    sp500_scale_factor=100.0,
)

# Dataset automatically stores target stats for easy access:
# dataset.target_stats = {
#     'mean': torch.Tensor[N_stocks],
#     'std': torch.Tensor[N_stocks],
#     'scale_factor': 100.0,
#     'target_name': 'DailyLogReturn'
# }


# ============================================================================
# 2. TRAINING INITIALIZATION (train.py)
# ============================================================================

# After building datasets, CLI injects destandardization into task
if task is not None and dataset_name in ['sp500', 'sp500_cleaned']:
    full_dataset = datasets.get('full')
    if hasattr(full_dataset, 'get_target_standardization_stats'):
        # Get stats from dataset
        target_stats = full_dataset.get_target_standardization_stats()
        
        # Inject both stats and destandardization method into task
        if hasattr(task, 'set_target_destandardization'):
            from graph_signal_diffusion.datasets.sp500.dataset import SP500Stocks
            task.set_target_destandardization(
                target_stats,
                SP500Stocks.destandardize_target
            )

# Task now has:
# - task.target_stats: Dict with mean, std, scale_factor
# - task.destandardize_target_fn: Static method for destandardization


# ============================================================================
# 3. TASK EVALUATOR (evaluator.py)
# ============================================================================

class StockPriceForecastingTaskV2:
    def __init__(self, ...):
        # Initialize destandardization attributes
        self.target_stats = None
        self.destandardize_target_fn = None
    
    def set_target_destandardization(self, target_stats, destandardize_fn):
        """Injected by CLI during training initialization."""
        self.target_stats = target_stats
        self.destandardize_target_fn = destandardize_fn
    
    def _destandardize_both_samples(self, generated, real, metadata):
        """
        Destandardize predictions and targets.
        
        NEW: Uses injected method if available (simpler, cleaner)
        FALLBACK: Uses legacy per_stock_stats if new method not injected
        """
        # Try new API first
        if self.destandardize_target_fn is not None and self.target_stats is not None:
            logger.info("Using injected destandardization (new API)")
            generated_destd = self.destandardize_target_fn(generated, self.target_stats)
            real_destd = self.destandardize_target_fn(real, self.target_stats)
            return generated_destd, real_destd
        
        # Fall back to legacy
        logger.info("Using legacy per_stock_stats destandardization")
        # ... complex legacy code ...


# ============================================================================
# 4. EVALUATION FLOW
# ============================================================================

# During training epoch:
# trainer.evaluate_v2() 
#   → task.evaluate_samples(generated, real, metadata)
#     → task._destandardize_both_samples(generated, real, metadata)
#       → Uses injected destandardize_target_fn(samples, target_stats)
#       → Returns destandardized predictions and targets
#     → task._process_ensemble_v2() - works on destandardized data
#     → task._convert_log_returns_to_prices_v2() - converts to prices
#     → task._compute_all_metrics_v2() - computes metrics on prices
#     → Returns metrics dict


# ============================================================================
# 5. BACKWARDS COMPATIBILITY
# ============================================================================

# Old approach (still works):
# - Task uses dataset_info['per_stock_stats'] 
# - Manual destandardization: (z * std + mean) * scale_factor
# - Complex feature-specific logic in evaluator

# New approach (cleaner):
# - Dataset provides get_target_standardization_stats()
# - Static method handles all destandardization logic
# - Task just calls destandardize_target_fn()
# - One-liner: destd = fn(standardized, stats)

# Compatibility:
# - If new method injected → uses it
# - If not → falls back to legacy
# - No breaking changes for existing code


# ============================================================================
# 6. BENEFITS
# ============================================================================

"""
1. Separation of Concerns:
   - Dataset owns standardization/destandardization logic
   - Task just needs to call it
   
2. Simplified Task Code:
   - From ~200 lines of destandardization logic
   - To: 2 lines calling injected method
   
3. Type Safety:
   - Static method with clear signature
   - Handles various tensor shapes automatically
   
4. Testability:
   - Destandardization tested independently in dataset tests
   - Task tests can mock the injection
   
5. Maintainability:
   - Single source of truth for destandardization
   - Changes only needed in dataset.py
   
6. Performance:
   - No redundant stat lookups
   - Stats pre-computed and cached
   - Efficient tensor operations
"""


# ============================================================================
# 7. MIGRATION PATH
# ============================================================================

"""
For new datasets:
1. Implement get_target_standardization_stats() in dataset
2. Implement static destandardize_target() method  
3. CLI will auto-inject into task
4. Task uses new API automatically

For old datasets (SP100, etc.):
1. No changes needed
2. Task falls back to legacy per_stock_stats
3. Works as before

For hybrid (SP500 with new processing):
1. Dataset has both APIs
2. New API preferred when available
3. Full backwards compatibility maintained
"""
