# Output Directory Structure

## Overview

All outputs are now organized hierarchically by task and method for easy navigation and comparison.

## Run Naming Behavior

The directory naming depends on whether you provide an explicit `run_name` and whether wandb is enabled:

### 1. **Explicit run name** (recommended for reproducibility)
```bash
python -m graph_signal_diffusion.cli.train run_name=my_experiment_v1
```
- **Local directory**: `outputs/.../my_experiment_v1/`
- **Wandb run name**: `my_experiment_v1`
- ✅ Perfect correlation between local and wandb

### 2. **No run name + wandb enabled** (automatic naming)
```bash
python -m graph_signal_diffusion.cli.train wandb.enabled=true
```
- **Wandb generates**: `happy-forest-42` (or similar)
- **Local directory**: `outputs/.../2026-01-27/14-30-45/` (timestamp)
- **Symlink created**: `outputs/.../happy-forest-42 -> 2026-01-27/14-30-45/`
- ✅ Can navigate by wandb name or timestamp

### 3. **No run name + wandb disabled** (default for local debugging)
```bash
python -m graph_signal_diffusion.cli.train
```
- **Local directory**: `outputs/.../2026-01-27/14-30-45/` (timestamp)
- No wandb run created
- ✅ Chronological organization

### Finding Runs

**By wandb name:**
```bash
ls outputs/stock_price_forecasting/sp500/ddpm/happy-forest-42/
```

**By timestamp:**
```bash
ls outputs/stock_price_forecasting/sp500/ddpm/2026-01-27/14-30-45/
```

**By explicit name:**
```bash
ls outputs/stock_price_forecasting/sp500/ddpm/my_experiment_v1/
```

## Directory Structure

```
outputs/
├── stock_price_forecasting/          # Task name
│   ├── diffusion/                     # Training outputs
│   │   ├── 2026-01-27/
│   │   │   └── 14-30-45/
│   │   │       ├── trainer_chkpts/    # Model checkpoints
│   │   │       │   ├── checkpoint_epoch_100.pt
│   │   │       │   └── checkpoint_epoch_200.pt
│   │   │       ├── .hydra/            # Config used
│   │   │       └── train.log
│   │   └── multirun/                  # Hyperparameter sweeps
│   │       └── 2026-01-27/
│   │           └── 15-00-00/
│   │               ├── 0/             # Run with params[0]
│   │               ├── 1/             # Run with params[1]
│   │               └── 2/             # Run with params[2]
│   │
│   ├── grw/                           # GRW baseline evaluation
│   │   └── 2026-01-27/
│   │       └── 16-00-00/
│   │           ├── evaluation_results.json
│   │           ├── predictions_vs_actual.png
│   │           └── .hydra/
│   │
│   └── comparison/                    # Baseline comparison
│       └── 2026-01-27/
│           └── 16-30-00/
│               ├── baseline_comparison.csv    # ← Main results table
│               ├── diffusion/                 # Diffusion visualizations
│               │   ├── predictions_vs_actual.png
│               │   ├── error_distribution.png
│               │   └── temporal_error.png
│               ├── grw/                       # GRW visualizations
│               │   └── ...
│               ├── comparison_price_rmse.png  # Comparison plots
│               ├── comparison_price_mae.png
│               └── .hydra/
│
├── primal_dual_power_allocation/     # Different task
│   ├── diffusion/
│   │   └── 2026-01-27/
│   └── comparison/
│       └── 2026-01-27/
│
└── traffic_forecasting/              # Another task
    ├── diffusion/
    └── comparison/
```

## Output Paths by Command

### Training Diffusion Model

**Command**:
```bash
python -m graph_signal_diffusion.cli.train
```

**Output path**:
```
outputs/stock_price_forecasting/diffusion/2026-01-27/14-30-45/
```

**Override path**:
```bash
python -m graph_signal_diffusion.cli.train \
    hydra.run.dir=outputs/my_custom_path
```

---

### Evaluating Single Baseline

**Command**:
```bash
python -m graph_signal_diffusion.cli.evaluate baseline=grw
```

**Output path**:
```
outputs/stock_price_forecasting/grw/2026-01-27/16-00-00/
```

**Files created**:
- `evaluation_results.json` - All metrics
- Visualization plots

---

### Comparing Baselines

**Command**:
```bash
python -m graph_signal_diffusion.cli.compare_baselines \
    baselines_to_compare=[diffusion,grw]
```

**Output path**:
```
outputs/stock_price_forecasting/comparison/2026-01-27/16-30-00/
```

**Files created**:
- `baseline_comparison.csv` - **Main results table**
- `<baseline_name>/` - Per-baseline visualizations
- `comparison_*.png` - Side-by-side comparison plots

---

### Hyperparameter Sweep (Multirun)

**Command**:
```bash
python -m graph_signal_diffusion.cli.train --multirun \
    trainer.learning_rate=0.0001,0.0002,0.0005 \
    seed=42,43,44
```

**Output path**:
```
outputs/stock_price_forecasting/diffusion/multirun/2026-01-27/17-00-00/
├── 0/  # lr=0.0001, seed=42
├── 1/  # lr=0.0001, seed=43
├── 2/  # lr=0.0001, seed=44
├── 3/  # lr=0.0002, seed=42
...
└── 8/  # lr=0.0005, seed=44
```

---

## Benefits

### 1. **Easy Navigation**
```bash
# All stock price forecasting results
ls outputs/stock_price_forecasting/

# All diffusion training runs
ls outputs/stock_price_forecasting/diffusion/

# All comparison results
ls outputs/stock_price_forecasting/comparison/
```

### 2. **Clear Separation**
- Training outputs separate from evaluation
- Each baseline has its own directory
- Comparisons in dedicated folder

### 3. **Task-Based Organization**
- Easy to find all results for a specific task
- Compare different tasks side-by-side
- No mixing of unrelated experiments

### 4. **Chronological History**
```bash
# See all experiments on a specific date
ls outputs/stock_price_forecasting/diffusion/2026-01-27/

# Find most recent run
ls -lt outputs/stock_price_forecasting/diffusion/2026-01-27/ | head -n 2
```

---

## Finding Checkpoints

### Most Recent Training Run

```bash
# Get the latest checkpoint
LATEST=$(ls -t outputs/stock_price_forecasting/diffusion/*/*/trainer_chkpts/checkpoint_*.pt | head -n 1)
echo $LATEST

# Use in evaluation
python -m graph_signal_diffusion.cli.compare_baselines \
    baselines.diffusion.checkpoint_path=$LATEST
```

### Specific Date/Time

```bash
# Checkpoint from specific run
CHECKPOINT="outputs/stock_price_forecasting/diffusion/2026-01-27/14-30-45/trainer_chkpts/checkpoint_epoch_100.pt"
```

### Best Checkpoint (by validation loss)

```python
import os
import glob
import torch

# Find all checkpoints
pattern = "outputs/stock_price_forecasting/diffusion/*/*/trainer_chkpts/checkpoint_*.pt"
checkpoints = glob.glob(pattern)

# Find best by validation loss
best_ckpt = None
best_val_loss = float('inf')

for ckpt in checkpoints:
    try:
        data = torch.load(ckpt, map_location='cpu')
        val_loss = data.get('val_loss', float('inf'))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_ckpt = ckpt
    except:
        continue

print(f"Best checkpoint: {best_ckpt}")
print(f"Validation loss: {best_val_loss:.6f}")
```

---

## Configuration Details

### Training Config ([config.yaml](../src/graph_signal_diffusion/conf/config.yaml))

```yaml
hydra:
  run:
    dir: outputs/${task.name}/diffusion/${now:%Y-%m-%d}/${now:%H-%M-%S}
  sweep:
    dir: outputs/${task.name}/diffusion/multirun/${now:%Y-%m-%d}/${now:%H-%M-%S}
    subdir: ${hydra.job.num}
```

- `${task.name}` - Resolved from task config (e.g., "stock_price_forecasting")
- `${now:%Y-%m-%d}` - Current date
- `${now:%H-%M-%S}` - Current time
- `${hydra.job.num}` - Job number in multirun sweep

### Evaluation Config ([evaluate.yaml](../src/graph_signal_diffusion/conf/evaluate.yaml))

```yaml
hydra:
  run:
    dir: outputs/${task.name}/${baseline.name}/${now:%Y-%m-%d}/${now:%H-%M-%S}
```

- `${baseline.name}` - Baseline being evaluated (e.g., "grw", "diffusion")

### Comparison Config ([compare_baselines.yaml](../src/graph_signal_diffusion/conf/compare_baselines.yaml))

```yaml
hydra:
  run:
    dir: outputs/${task.name}/comparison/${now:%Y-%m-%d}/${now:%H-%M-%S}
```

---

## Custom Output Paths

### Override at Runtime

```bash
# Custom path for training
python -m graph_signal_diffusion.cli.train \
    hydra.run.dir=outputs/ablation_study/model_v2

# Custom path for comparison
python -m graph_signal_diffusion.cli.compare_baselines \
    hydra.run.dir=outputs/paper_results/final_comparison
```

### Organize by Experiment

```bash
# Group by experiment name
python -m graph_signal_diffusion.cli.train \
    hydra.run.dir=outputs/experiments/exp_001_large_model

python -m graph_signal_diffusion.cli.train \
    hydra.run.dir=outputs/experiments/exp_002_small_model
```

---

## Migration from Old Structure

If you have existing outputs in the old format:

```bash
# Old structure
outputs/2026-01-27/14-30-45/

# New structure
outputs/stock_price_forecasting/diffusion/2026-01-27/14-30-45/
```

**Migration script**:
```bash
#!/bin/bash
# Move old outputs to new structure

# Assuming old outputs are diffusion training runs
for dir in outputs/20*/*/; do
    date=$(basename $(dirname "$dir"))
    time=$(basename "$dir")
    
    # Create new structure
    mkdir -p "outputs/stock_price_forecasting/diffusion/$date/"
    
    # Move directory
    mv "$dir" "outputs/stock_price_forecasting/diffusion/$date/$time"
done

echo "Migration complete!"
```

---

## Gitignore Recommendations

Add to `.gitignore`:
```gitignore
# Outputs (but keep structure)
outputs/*/*/*/*/
!outputs/*/*/*/*/*/.gitkeep

# Or ignore everything except comparison CSVs
outputs/
!outputs/**/baseline_comparison.csv
!outputs/**/evaluation_results.json
```

---

## Examples

### Complete Workflow with New Structure

```bash
# 1. Train model
python -m graph_signal_diffusion.cli.train
# → outputs/stock_price_forecasting/diffusion/2026-01-27/14-30-45/

# 2. Note checkpoint path
CHECKPOINT="outputs/stock_price_forecasting/diffusion/2026-01-27/14-30-45/trainer_chkpts/checkpoint_epoch_100.pt"

# 3. Compare with baselines
python -m graph_signal_diffusion.cli.compare_baselines \
    baselines.diffusion.checkpoint_path=$CHECKPOINT
# → outputs/stock_price_forecasting/comparison/2026-01-27/16-30-00/

# 4. Results for paper
cat outputs/stock_price_forecasting/comparison/2026-01-27/16-30-00/baseline_comparison.csv
```

### Organize Multiple Experiments

```bash
# Experiment 1: Standard model
python -m graph_signal_diffusion.cli.train
# → outputs/stock_price_forecasting/diffusion/2026-01-27/14-00-00/

# Experiment 2: Large model
python -m graph_signal_diffusion.cli.train model=ugnn_large
# → outputs/stock_price_forecasting/diffusion/2026-01-27/15-00-00/

# Experiment 3: Different diffusion steps
python -m graph_signal_diffusion.cli.train diffusion.num_timesteps=500
# → outputs/stock_price_forecasting/diffusion/2026-01-27/16-00-00/

# Compare all three
python -m graph_signal_diffusion.cli.compare_baselines \
    baselines_to_compare=[diffusion,diffusion,diffusion] \
    baselines.diffusion.checkpoint_path=outputs/stock_price_forecasting/diffusion/2026-01-27/14-00-00/trainer_chkpts/checkpoint_epoch_100.pt \
    baselines.diffusion2.checkpoint_path=outputs/stock_price_forecasting/diffusion/2026-01-27/15-00-00/trainer_chkpts/checkpoint_epoch_100.pt \
    baselines.diffusion3.checkpoint_path=outputs/stock_price_forecasting/diffusion/2026-01-27/16-00-00/trainer_chkpts/checkpoint_epoch_100.pt
```

---

## Tips

1. **Use tab completion**: The hierarchical structure makes it easy to navigate with tab completion in the shell

2. **Grep for results**: `grep -r "price_rmse" outputs/stock_price_forecasting/comparison/`

3. **Compare dates**: `diff outputs/stock_price_forecasting/comparison/2026-01-27/*/baseline_comparison.csv`

4. **Archive old results**: `tar -czf archive_2026-01.tar.gz outputs/*/*/2026-01-*/`

5. **Symbolic links**: Create links to important experiments
   ```bash
   ln -s outputs/stock_price_forecasting/diffusion/2026-01-27/14-30-45 best_model
   ```
