# Research Workflow: Training, Evaluation, and Publication

Complete workflow for training diffusion models, comparing with baselines, and generating publication-ready results.

## Overview

**Goal**: Train diffusion model → Evaluate on test set → Compare with baselines → Generate tables for paper

**Steps**:
1. Configure and train diffusion model
2. Evaluate trained model and save checkpoint
3. Fit and evaluate all baselines on same data
4. Compare results and generate publication tables

---

## Step 1: Train Diffusion Model

### Configure Hyperparameters

Edit configs or override via command line:

**Key config files**:
- `conf/config.yaml` - Main training config
- `conf/dataset/sp100.yaml` - Dataset settings
- `conf/model/ugnn.yaml` - Model architecture
- `conf/diffusion/ddpm.yaml` - Diffusion process
- `conf/trainer/default.yaml` - Training hyperparameters

**Important hyperparameters**:
```yaml
# conf/trainer/default.yaml
max_epochs: 100
learning_rate: 0.0001
max_grad_norm: 1.0
log_every_n_steps: 50

# conf/diffusion/ddpm.yaml
num_timesteps: 1000
beta_schedule: "linear"
beta_start: 0.0001
beta_end: 0.02

# conf/model/ugnn.yaml
hidden_channels: 128
num_layers: 4
dropout: 0.1
```

### Train Model

**Standard training**:
```bash
# Default config (SP100, UGNN, DDPM, 100 epochs)
python -m graph_signal_diffusion.cli.train

# Or use your existing train.py
python train.py
```

**Override hyperparameters**:
```bash
python -m graph_signal_diffusion.cli.train \
    trainer.max_epochs=200 \
    trainer.learning_rate=0.0002 \
    model.hidden_channels=256 \
    diffusion.num_timesteps=500 \
    seed=42
```

**Train on different dataset**:
```bash
python -m graph_signal_diffusion.cli.train dataset=sp500
```

**Custom experiment name**:
```bash
python -m graph_signal_diffusion.cli.train \
    hydra.run.dir=outputs/exp_large_model \
    trainer.max_epochs=200 \
    model.hidden_channels=256
```

### Output

Training produces:
- `outputs/<date>/<time>/trainer_chkpts/` - Model checkpoints
- `outputs/<date>/<time>/.hydra/` - Config used
- Console logs with training/validation metrics

**Save the checkpoint path for later!**
```
outputs/2026-01-27/14-30-45/trainer_chkpts/checkpoint_epoch_100.pt
```

---

## Step 2: Compare with Baselines

Now compare your trained diffusion model with GRW and other baselines.

### Option A: Compare All Baselines Together

**Compare diffusion + GRW**:
```bash
python -m graph_signal_diffusion.cli.compare_baselines \
    baselines_to_compare=[diffusion,grw] \
    baselines.diffusion.checkpoint_path=outputs/stock_price_forecasting/diffusion/2026-01-27/14-30-45/trainer_chkpts/checkpoint_epoch_100.pt
```

**Customize baseline parameters**:
```bash
python -m graph_signal_diffusion.cli.compare_baselines \
    baselines_to_compare=[diffusion,grw] \
    baselines.diffusion.checkpoint_path=outputs/.../checkpoint_epoch_100.pt \
    baselines.grw.n_samples=50 \
    baselines.grw.shrinkage_strength=20.0
```

**Output**:
- `outputs/stock_price_forecasting/comparison/<date>/<time>/baseline_comparison.csv` - **Metrics table for paper!**
- `outputs/stock_price_forecasting/comparison/<date>/<time>/diffusion/` - Diffusion visualizations
- `outputs/stock_price_forecasting/comparison/<date>/<time>/grw/` - GRW visualizations
- `outputs/stock_price_forecasting/comparison/<date>/<time>/comparison_*.png` - Side-by-side comparison plots

### Option B: Evaluate Baselines Separately

If you need more control or want to run experiments independently:

**1. Evaluate GRW**:
```bash
python -m graph_signal_diffusion.cli.evaluate baseline=grw
# → outputs/stock_price_forecasting/grw/2026-01-27/16-00-00/
```

**2. Evaluate diffusion**:
```bash
python -m graph_signal_diffusion.cli.evaluate \
    baseline=diffusion \
    baseline.checkpoint_path=outputs/stock_price_forecasting/diffusion/.../checkpoint_epoch_100.pt
# → outputs/stock_price_forecasting/diffusion/2026-01-27/16-15-00/
```

**3. Manually combine results** (use scripts/compare_results.py or Python)

### Option C: Add More Baselines

To add additional baselines (ARIMA, VAR, etc.):

1. Create baseline class in `src/graph_signal_diffusion/baselines/stock_price_forecasting/`
2. Register with `@BASELINE_REGISTRY.register("name")`
3. Create config in `conf/baseline/name.yaml`
4. Add to comparison:

```bash
python -m graph_signal_diffusion.cli.compare_baselines \
    baselines_to_compare=[diffusion,grw,arima,var]
```

---

## Step 3: Generate Publication Tables

### Metrics Computed

The evaluation pipeline automatically computes:

**Primary metrics (price space)**:
- `price_mse` - Mean Squared Error
- `price_mae` - Mean Absolute Error  
- `price_rmse` - Root Mean Squared Error
- `price_mape` - Mean Absolute Percentage Error
- `price_smape` - Symmetric MAPE

**Probabilistic metrics**:
- `crps` - Continuous Ranked Probability Score
- `log_likelihood` - Gaussian log-likelihood
- `coverage_50`, `coverage_90` - Calibration
- `interval_width_50`, `interval_width_90`

**Secondary metrics (log return space)**:
- `return_mse`, `return_mae`, `return_rmse`
- `direction_accuracy` - Sign prediction
- `return_correlation` - Correlation coefficient

**Temporal metrics**:
- `temporal_mse_t1`, `temporal_mse_t5` - Error by forecast horizon

### Format for Paper

The `baseline_comparison.csv` contains results in this format:

```csv
baseline,price_mse,price_mae,price_rmse,price_mape,crps,...
diffusion,0.0234,0.1123,0.1531,2.45,0.0567,...
grw,0.0456,0.1567,0.2136,3.21,0.0823,...
```

### Create Publication Table

**Option 1: Use pandas to format**:

```python
import pandas as pd

# Load results
df = pd.read_csv("outputs/2026-01-27/14-30-45/baseline_comparison.csv", index_col=0)

# Select key metrics for paper
paper_metrics = [
    'price_rmse', 'price_mae', 'price_mape', 
    'crps', 'direction_accuracy'
]
table = df[paper_metrics]

# Round and format
table = table.round(4)

# Add improvement row
table.loc['Improvement (%)'] = (
    (table.loc['grw'] - table.loc['diffusion']) / table.loc['grw'] * 100
)

# Export to LaTeX
print(table.to_latex(escape=False))

# Export to CSV for Excel/Google Sheets
table.to_csv("paper_table.csv")
```

**Option 2: Manual LaTeX table**:

```latex
\begin{table}[h]
\centering
\caption{Baseline Comparison on S\&P100 Stock Price Forecasting}
\begin{tabular}{lccccc}
\toprule
Method & RMSE $\downarrow$ & MAE $\downarrow$ & MAPE $\downarrow$ & CRPS $\downarrow$ & Dir. Acc. $\uparrow$ \\
\midrule
GRW & 0.2136 & 0.1567 & 3.21 & 0.0823 & 0.512 \\
Diffusion (Ours) & \textbf{0.1531} & \textbf{0.1123} & \textbf{2.45} & \textbf{0.0567} & \textbf{0.587} \\
\midrule
Improvement & 28.3\% & 28.3\% & 23.7\% & 31.1\% & 14.6\% \\
\bottomrule
\end{tabular}
\label{tab:results}
\end{table}
```

---

## Complete Example: End-to-End

```bash
# ============================================================
# STEP 1: Train diffusion model
# ============================================================

# Train with your chosen hyperparameters
python -m graph_signal_diffusion.cli.train \
    trainer.max_epochs=100 \
    trainer.learning_rate=0.0001 \
    model.hidden_channels=128 \
    diffusion.num_timesteps=1000 \
    seed=42

# Output automatically saved to:
# outputs/stock_price_forecasting/diffusion/2026-01-27/14-30-45/

# Save the checkpoint path!
# outputs/stock_price_forecasting/diffusion/2026-01-27/14-30-45/trainer_chkpts/checkpoint_epoch_100.pt

# ============================================================
# STEP 2: Compare with baselines
# ============================================================

# Compare diffusion + GRW
python -m graph_signal_diffusion.cli.compare_baselines \
    baselines_to_compare=[diffusion,grw] \
    baselines.diffusion.checkpoint_path=$CHECKPOINT \
    baselines.grw.n_samples=20 \
    baselines.grw.shrinkage_strength=10.0 \
    hydra.run.dir=outputs/comparison_exp1

# ============================================================
# STEP 3: Generate publication table
# ============================================================

# Results are in: outputs/comparison_exp1/baseline_comparison.csv
python << EOF
import pandas as pd
df = pd.read_csv("outputs/comparison_exp1/baseline_comparison.csv", index_col=0)

# Select metrics for paper
metrics = ['price_rmse', 'price_mae', 'price_mape', 'crps', 'direction_accuracy']
table = df[metrics].round(4)

# Calculate improvement
improvement = ((table.loc['grw'] - table.loc['diffusion']) / table.loc['grw'] * 100).round(1)
table.loc['Improvement (%)'] = improvement

print("\nResults for Paper:")
print("=" * 80)
print(table)
print("\nLaTeX Table:")
print(table.to_latex(escape=False))
EOF
```

---

## Hyperparameter Sweeps

To find best hyperparameters:

### Grid Search Example

```bash
# Test different learning rates
for lr in 0.0001 0.0002 0.0005 0.001; do
    python -m graph_signal_diffusion.cli.train \
        trainer.learning_rate=$lr \
        seed=42 \
        hydra.run.dir=outputs/sweep_lr/lr_${lr}
done

# Compare all trained models
python -m graph_signal_diffusion.cli.compare_baselines \
    baselines_to_compare=[diffusion,grw] \
    baselines.diffusion.checkpoint_path=outputs/sweep_lr/lr_0.0001/trainer_chkpts/checkpoint_epoch_100.pt
# ... repeat for each lr
```

### Hydra Multirun

```bash
# Automatic sweep over multiple parameters
python -m graph_signal_diffusion.cli.train \
    --multirun \
    trainer.learning_rate=0.0001,0.0002,0.0005 \
    model.hidden_channels=64,128,256 \
    seed=42,43,44
```

---

## Ablation Studies

### Architecture Ablation

```bash
# Compare different models
python -m graph_signal_diffusion.cli.train model=ugnn hydra.run.dir=outputs/ablation_ugnn
python -m graph_signal_diffusion.cli.train model=unet hydra.run.dir=outputs/ablation_unet

# Compare results
python -m graph_signal_diffusion.cli.compare_baselines \
    baselines_to_compare=[diffusion,diffusion] \
    baselines.diffusion.checkpoint_path=outputs/ablation_ugnn/...pt \
    baselines.diffusion2.checkpoint_path=outputs/ablation_unet/...pt
```

### Diffusion Steps Ablation

```bash
# Test different timesteps
for T in 100 500 1000 2000; do
    python -m graph_signal_diffusion.cli.train \
        diffusion.num_timesteps=$T \
        hydra.run.dir=outputs/ablation_T/T_${T}
done
```

---

## Best Practices

### 1. **Version Control Your Configs**
```bash
# Before training, commit your config
git add conf/
git commit -m "Config for experiment: large model, lr=0.0001"
git tag exp_v1

# After training, note the commit hash
git rev-parse HEAD > outputs/diffusion_exp1/git_hash.txt
```

### 2. **Use Descriptive Output Directories**
```bash
python -m graph_signal_diffusion.cli.train \
    hydra.run.dir=outputs/sp100_ugnn128_lr1e4_T1000_seed42
```

### 3. **Track Multiple Seeds**
```bash
# Train with different seeds
for seed in 42 43 44 45 46; do
    python -m graph_signal_diffusion.cli.train \
        seed=$seed \
        hydra.run.dir=outputs/diffusion_seed${seed}
done

# Compare average performance
python scripts/aggregate_seeds.py outputs/diffusion_seed*
```

### 4. **Save Experiment Metadata**
```bash
# After training, save experiment info
cat > outputs/diffusion_exp1/experiment.json << EOF
{
  "description": "UGNN with 128 channels, 1000 timesteps",
  "dataset": "SP100",
  "train_date": "2026-01-27",
  "hyperparameters": {
    "learning_rate": 0.0001,
    "epochs": 100,
    "hidden_channels": 128,
    "diffusion_steps": 1000
  }
}
EOF
```

### 5. **Validation-Based Model Selection**
```python
# Select best checkpoint based on validation loss
import os
import torch

checkpoint_dir = "outputs/diffusion_exp1/trainer_chkpts"
checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]

best_checkpoint = None
best_val_loss = float('inf')

for ckpt in checkpoints:
    data = torch.load(os.path.join(checkpoint_dir, ckpt))
    val_loss = data.get('val_loss', float('inf'))
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_checkpoint = ckpt

print(f"Best checkpoint: {best_checkpoint}")
print(f"Val loss: {best_val_loss}")
```

---

## Troubleshooting

### "Out of memory" during training
```bash
# Reduce batch size
python -m graph_signal_diffusion.cli.train dataset.batch_size=16

# Use gradient accumulation
python -m graph_signal_diffusion.cli.train trainer.gradient_accumulation_steps=4
```

### "GRW not fitted" error
Make sure `needs_training: true` in `conf/baseline/grw.yaml`

### Different train/test splits
Both `cli.train` and `cli.evaluate` use the same dataset config → same splits.
Verify with `--cfg job` flag to print resolved config.

### Checkpoint not found
Use absolute path or relative to project root:
```bash
python -m graph_signal_diffusion.cli.compare_baselines \
    baselines.diffusion.checkpoint_path=$(pwd)/outputs/diffusion_exp1/trainer_chkpts/checkpoint_epoch_100.pt
```

---

## Quick Reference

```bash
# Train diffusion
python -m graph_signal_diffusion.cli.train [overrides]

# Compare with baselines
python -m graph_signal_diffusion.cli.compare_baselines \
    baselines_to_compare=[diffusion,grw] \
    baselines.diffusion.checkpoint_path=path/to/checkpoint.pt

# Evaluate single baseline
python -m graph_signal_diffusion.cli.evaluate \
    baseline=grw

# Check config before running
python -m graph_signal_diffusion.cli.train --cfg job

# Hyperparameter sweep
python -m graph_signal_diffusion.cli.train --multirun \
    trainer.learning_rate=0.0001,0.0002 \
    seed=42,43,44
```

## Related Documentation

- [BASELINE_EVALUATION_GUIDE.md](BASELINE_EVALUATION_GUIDE.md) - Baseline evaluation details
- [TASK_EVALUATION_GUIDE.py](TASK_EVALUATION_GUIDE.py) - Task-specific metrics
- [UGNN_PIPELINE_ANALYSIS.md](UGNN_PIPELINE_ANALYSIS.md) - Model architecture
