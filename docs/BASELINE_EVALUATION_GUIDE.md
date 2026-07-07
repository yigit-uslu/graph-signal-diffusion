# Baseline Evaluation Guide

## Overview

This guide shows how to evaluate baselines using the CLI infrastructure. All baselines are registered in `BASELINE_REGISTRY` and can be evaluated using standardized CLI commands.

## Architecture

### Components

1. **Baseline Registry** - `BASELINE_REGISTRY` in `src/graph_signal_diffusion/baselines/__init__.py`
   - Auto-discovers and registers baselines using `@BASELINE_REGISTRY.register(name)` decorator
   - All baselines inherit from `BaseBaseline` with standardized interface

2. **CLI Scripts**
   - `cli.evaluate` - Evaluate a single baseline
   - `cli.compare_baselines` - Compare multiple baselines

3. **Configuration**
   - Baseline configs: `conf/baseline/<baseline_name>.yaml`
   - Comparison configs: `conf/compare_baselines.yaml`
   - Evaluation configs: `conf/evaluate.yaml`

4. **Unified Evaluator** - `evaluation.UnifiedEvaluator`
   - Standardized evaluation pipeline
   - Automatic metric computation
   - Visualization generation

## Usage

### Compare Multiple Baselines

```bash
# Compare GRW and Diffusion on SP100
python -m graph_signal_diffusion.cli.compare_baselines \
    dataset=sp100 \
    task=stock_price_forecasting \
    baselines_to_compare=[grw,diffusion] \
    max_epochs=100

# Compare on SP500
python -m graph_signal_diffusion.cli.compare_baselines \
    dataset=sp500 \
    baselines_to_compare=[grw,diffusion]

# Override GRW parameters
python -m graph_signal_diffusion.cli.compare_baselines \
    baselines_to_compare=[grw] \
    baselines.grw.n_samples=50 \
    baselines.grw.shrinkage_strength=20.0
```

**Output:**
- `outputs/<date>/<time>/baseline_comparison.csv` - Metrics table
- `outputs/<date>/<time>/<baseline_name>/` - Visualizations per baseline
- `outputs/<date>/<time>/comparison_*.png` - Comparison plots

### Evaluate Single Baseline

```bash
# Evaluate GRW only
python -m graph_signal_diffusion.cli.evaluate \
    baseline=grw \
    dataset=sp100 \
    task=stock_price_forecasting

# Load from checkpoint
python -m graph_signal_diffusion.cli.evaluate \
    baseline=diffusion \
    checkpoint_path=checkpoints/model_epoch_100.pt
```

## Baseline Interface

All baselines must implement:

```python
from graph_signal_diffusion.baselines import BASELINE_REGISTRY
from graph_signal_diffusion.baselines.base import BaseBaseline

@BASELINE_REGISTRY.register("my_baseline")
class MyBaseline(BaseBaseline):
    def __init__(self, device, **kwargs):
        super().__init__(device=device, **kwargs)
        # Initialize parameters
    
    def fit(self, train_loader, val_loader=None, **kwargs):
        """Train/fit the baseline model."""
        pass
    
    def predict(self, data) -> torch.Tensor:
        """
        Generate predictions for input data.
        
        Args:
            data: PyG Data object
            
        Returns:
            predictions: [B*n_samples, T, N, F] tensor
        """
        pass
    
    def evaluate(self, loader, task) -> Dict[str, float]:
        """Evaluate on dataset using task metrics."""
        all_predictions = []
        all_targets = []
        all_metadata = []
        
        for data in loader:
            data = data.to(self.device)
            predictions = self.predict(data)
            data_dict = task.prepare_data(data)
            
            all_predictions.append(predictions)
            all_targets.append(data_dict["samples"])
            all_metadata.append(data_dict["metadata"])
        
        predictions_all = torch.cat(all_predictions)
        targets_all = torch.cat(all_targets)
        metadata = self._merge_metadata(all_metadata)
        
        return task.evaluate_samples(predictions_all, targets_all, metadata)
```

## Configuration Structure

### Baseline Config (`conf/baseline/grw.yaml`)

```yaml
# @package baseline
name: grw
_target_: graph_signal_diffusion.baselines.stock_price_forecasting.grw.GeometricRandomWalk
needs_training: true

# Baseline-specific parameters
device: cuda
n_samples: 20
shrinkage_strength: 10.0
min_samples_per_stock: 20
```

### Comparison Config (`conf/compare_baselines.yaml`)

```yaml
# @package _global_
defaults:
  - dataset: sp100
  - task: stock_price_forecasting
  - baseline@baselines.diffusion: diffusion
  - baseline@baselines.grw: grw
  - _self_

baselines_to_compare: [grw, diffusion]
max_epochs: 100
seed: 0
```

## GRW Baseline Details

The Geometric Random Walk baseline uses per-stock parameters with empirical Bayes shrinkage:

### Parameters

- `n_samples` (default: 20) - Number of samples to generate per prediction
- `shrinkage_strength` (default: 10.0) - Controls shrinkage toward global mean
- `min_samples_per_stock` (default: 20) - Minimum samples for per-stock estimation

### Shrinkage Formula

For each stock s:
- λ = n_s / (n_s + k) where k = `shrinkage_strength`
- μ_final_s = λ × μ_s + (1-λ) × μ_global
- σ_final_s = λ × σ_s + (1-λ) × σ_global

### Output Format

Predictions are in **log return space**: `[B*n_samples, T_future, N, 1]`

The task evaluator automatically converts to price space for metrics.

## Workflow

### Standard Evaluation Flow

1. **Discovery** - `discover_baselines()` imports and registers all baselines
2. **Configuration** - Hydra loads configs and instantiates baselines
3. **Fitting** - `baseline.fit(train_loader)` estimates parameters
4. **Prediction** - `baseline.predict(data)` generates samples
5. **Evaluation** - `task.evaluate_samples()` computes metrics
6. **Visualization** - `UnifiedEvaluator` creates comparison plots

### Adding New Baselines

1. Create baseline class in `src/graph_signal_diffusion/baselines/<task>/`
2. Register with `@BASELINE_REGISTRY.register("name")`
3. Implement `fit()`, `predict()`, and `evaluate()` methods
4. Create config in `conf/baseline/<name>.yaml`
5. Add to `baselines_to_compare` list

## Examples

### Quick GRW Evaluation

```bash
# Just evaluate GRW (no training needed for diffusion)
python -m graph_signal_diffusion.cli.compare_baselines \
    baselines_to_compare=[grw]
```

### Hyperparameter Sweep

```bash
# Test different shrinkage strengths
for k in 5.0 10.0 20.0 50.0; do
    python -m graph_signal_diffusion.cli.evaluate \
        baseline=grw \
        baselines.grw.shrinkage_strength=$k \
        hydra.run.dir=outputs/grw_k${k}
done
```

### Compare with Trained Diffusion

```bash
# First train diffusion model
python train.py # Your normal training script

# Then compare with GRW
python -m graph_signal_diffusion.cli.compare_baselines \
    baselines_to_compare=[grw,diffusion] \
    baselines.diffusion.checkpoint_path=checkpoints/best_model.pt
```

## Best Practices

1. **Use CLI for consistency** - Don't write separate evaluation scripts
2. **Configure via Hydra** - Override parameters via command line
3. **Leverage UnifiedEvaluator** - Automatic metrics and visualization
4. **Register all baselines** - Makes them discoverable and comparable
5. **Follow BaseBaseline interface** - Ensures compatibility

## Troubleshooting

### "Baseline not found in registry"
- Check if baseline file is in `baselines/<task>/` directory
- Verify `@BASELINE_REGISTRY.register()` decorator is present
- Ensure `discover_baselines()` is called

### "needs_training parameter"
- Set `needs_training: true` if baseline needs `fit()`
- Set `needs_training: false` for analytical baselines

### Shape mismatches
- Ensure predictions are `[B*n_samples, T, N, F]`
- Check that batch dimension includes `num_graphs`
- Verify feature dimension matches target

## Related Documentation

- [TASK_EVALUATION_GUIDE.py](TASK_EVALUATION_GUIDE.py) - Task-specific evaluation
- [UGNN_PIPELINE_ANALYSIS.md](UGNN_PIPELINE_ANALYSIS.md) - Diffusion model details
