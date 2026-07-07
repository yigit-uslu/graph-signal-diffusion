# Reproducibility Guide

This document explains how to achieve exact reproducibility of experimental results.

## For Paper Authors

When publishing results, include:

1. **Exact Python version**: `python --version`
2. **requirements.txt with pinned versions** (already in repo)
3. **Git commit hash** of the code used
4. **Random seeds** used (stored in config files)
5. **Hardware specifications** (GPU model, driver version)

### Generating Pinned Requirements

Before publishing:

```bash
# Ensure your environment is up-to-date
pip install -e .

# Generate pinned requirements from your working environment
python scripts/generate_requirements.py

# Commit the updated requirements.txt
git add requirements.txt
git commit -m "Update requirements.txt for reproducibility"
```

## For Users Reproducing Results

### Step 1: Match Python Version

Check the Python version used:
```bash
cat .python-version  # If available
# Or check the paper/documentation
```

Create environment with matching Python version:
```bash
# Using conda
conda create -n graph-signal-diffusion python=3.9
conda activate graph-signal-diffusion

# Or using pyenv
pyenv install 3.9.12
pyenv virtualenv 3.9.12 graph-signal-diffusion
pyenv activate graph-signal-diffusion
```

### Step 2: Install Exact Dependencies

```bash
# Install exact versions from requirements.txt
pip install -r requirements.txt

# Install project in editable mode
pip install -e .
```

### Step 3: Verify Installation

```bash
# Run verification script
python scripts/verify_install.py

# Check package versions match
pip list
```

### Step 4: Use Same Random Seeds

Random seeds are typically specified in config files:
- Check `configs/*.yaml` for seed values
- Do not modify seed values
- If running multiple times, use the same seed for comparison

### Step 5: Verify Data

```bash
# Check data directory structure
ls -la data/

# If data is missing, follow data preparation instructions in docs/
# Some datasets may need to be downloaded or generated
```

## Common Reproducibility Issues

### Issue 1: Different Results with Same Code

**Possible causes:**
- Different hardware (CPU vs GPU, different GPU models)
- Non-deterministic algorithms (some PyTorch operations)
- Different CUDA/cuDNN versions
- Numerical precision differences

**Solutions:**
```python
# In your training script, add:
import torch
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Make PyTorch deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

### Issue 2: Version Conflicts

**Solution:**
```bash
# Start fresh
pip freeze > old_requirements.txt  # Backup current env
pip uninstall -y -r old_requirements.txt  # Remove all packages
pip install -r requirements.txt  # Install exact versions
```

### Issue 3: CUDA Version Mismatch

Check CUDA version:
```bash
nvcc --version
nvidia-smi
```

Install matching PyTorch:
```bash
# For CUDA 11.3
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113

# For CUDA 11.8
pip install torch==1.12.1+cu118 torchvision==0.13.1+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118
```

## Docker for Maximum Reproducibility

For absolute reproducibility, use Docker:

```dockerfile
# Dockerfile
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

WORKDIR /workspace

# Copy requirements
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy project
COPY . .
RUN pip install -e .

# Set reproducibility options
ENV PYTHONHASHSEED=0
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8

CMD ["python", "-m", "graph_signal_diffusion.cli.train"]
```

Build and run:
```bash
docker build -t graph-signal-diffusion .
docker run --gpus all -v $(pwd)/outputs:/workspace/outputs graph-signal-diffusion
```

## Continuous Integration

Use GitHub Actions to ensure reproducibility:

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -e .
      - name: Verify installation
        run: python scripts/verify_install.py
      - name: Run tests
        run: pytest tests/
```

## Experiment Tracking

Use experiment tracking tools to record:
- Exact package versions used
- Hardware specifications
- Random seeds
- Hyperparameters
- Git commit hash

Example using Weights & Biases:
```python
import wandb
import sys
import torch

wandb.init(project="graph-signal-diffusion")
wandb.config.update({
    "python_version": sys.version,
    "pytorch_version": torch.__version__,
    "cuda_version": torch.version.cuda,
    "git_commit": os.popen("git rev-parse HEAD").read().strip(),
})
```

## Checklist for Reproducibility

- [ ] Exact Python version specified
- [ ] All dependencies pinned in requirements.txt
- [ ] Random seeds documented and fixed
- [ ] Data preparation steps documented
- [ ] Git commit hash recorded
- [ ] Hardware specifications documented
- [ ] Installation verified with verify_install.py
- [ ] Results can be reproduced by independent party
- [ ] Experiment tracking enabled (wandb/tensorboard)
- [ ] Config files version controlled
