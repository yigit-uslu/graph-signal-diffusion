# Installation Guide

This document provides systematic installation instructions for reproducible research.

## Quick Start

### Option 1: Conda Environment (Recommended)

**🎯 For most users (new systems with different CUDA versions):**
```bash
# Step 1: Create minimal environment (no PyTorch yet)
conda env create -f environment-minimal.yml
conda activate graph-signal-diffusion

# Step 2: Check your CUDA version
nvcc --version  # or nvidia-smi

# Step 3: Install PyTorch for your CUDA version
# For CUDA 11.8:
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# For CUDA 12.1:
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# For CUDA 12.4:
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

# For CPU only (no GPU):
conda install pytorch torchvision torchaudio cpuonly -c pytorch

# Step 4: Install PyTorch Geometric
conda install pyg -c pyg

# Step 5: Install remaining packages
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv
pip install jaxtyping beartype accelerate wandb yfinance
pip install -e .

# Step 6: Verify installation
python -c "import torch; import torch_geometric; print('✓ Installation successful')"
```

**⚡ For exact reproducibility** (matching development system with CUDA 12.4):
```bash
# Use the pinned environment file (requires CUDA 12.4)
conda env create -f environment.yml

# Activate environment
conda activate graph-signal-diffusion

# Verify installation
python -c "import torch; import torch_geometric; print('✓ Installation successful')"
```

### Option 2: pip with Pinned Versions (Exact Reproducibility)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install exact versions
pip install -r requirements.txt

# Install project in editable mode
pip install -e .
```

### Option 3: pip with Flexible Versions (Development)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install with flexible versions
pip install -e .

# Optional: Install development tools
pip install -r requirements-dev.txt
```

## Dependency Management Philosophy

This project uses a **multi-layered approach** to dependency management:

### 1. `pyproject.toml` (Source of Truth)
- Contains **core dependencies** with **minimum version requirements**
- Used for development and distribution
- Allows users flexibility to use newer compatible versions
- Command: `pip install -e .`

### 2. `requirements.txt` (Exact Reproducibility)
- **Pinned versions** of all dependencies
- Generated from actual working environment
- Use this to **replicate exact results** from papers/experiments
- Command: `pip install -r requirements.txt`

### 3. `environment.yml` (Conda - Exact Reproducibility)
- Conda-specific environment with **pinned CUDA version** (12.4)
- For exact reproduction of development environment
- Command: `conda env create -f environment.yml`

### 4. `environment-flexible.yml` (Conda - Attempted Auto-Detection)
- ⚠️ **May fail on some systems** due to missing CUDA packages in conda channels
- Tries to auto-detect CUDA version but conda solver can be fragile
- If this fails, use `environment-minimal.yml` instead
- Command: `conda env create -f environment-flexible.yml`

### 5. `environment-minimal.yml` (Conda - Most Reliable)
- **Recommended for new systems** where other methods fail
- Installs only non-PyTorch dependencies first
- Then you manually install PyTorch for your specific CUDA version
- Most flexible and reliable approach
- See Quick Start guide above for step-by-step instructions

## Updating Dependencies

### For Maintainers: Generate New requirements.txt

When you update your environment and want others to use your exact versions:

```bash
# Generate pinned requirements from current environment
python scripts/generate_requirements.py

# This creates/updates:
# - requirements.txt (pinned versions)
# - requirements-dev.txt (development tools)
```

### Alternative: Using pip freeze

```bash
# Export all packages (may include unnecessary dependencies)
pip freeze > requirements-full.txt

# Or use pipreqs to analyze imports (install first: pip install pipreqs)
pipreqs . --force
```

### Alternative: Using conda

```bash
# Export current conda environment
conda env export > environment-full.yml

# Export without builds (more portable)
conda env export --no-builds > environment-portable.yml

# Export from history (minimal spec)
conda env export --from-history > environment.yml
```

## CUDA Compatibility Guide

### Understanding CUDA Versions

This project provides multiple conda environment files:

1. **`environment.yml`**: Pinned to CUDA 12.4 (exact reproducibility)
2. **`environment-flexible.yml`**: Attempts auto-detection (may fail on some systems)
3. **`environment-minimal.yml`**: Most reliable (manual PyTorch installation)

**Which should you use?**
- **New system with conda installation errors?** → Use `environment-minimal.yml` (see Quick Start)
- **New system, want to try one-command install?** → Try `environment-flexible.yml` (may fail)
- **Reproducing exact results from paper?** → Use `environment.yml` (requires CUDA 12.4)

### Checking Your CUDA Version

```bash
# Check system CUDA version
nvcc --version

# Alternative: Check via nvidia-smi
nvidia-smi  # Look at top right for "CUDA Version"

# After PyTorch installation, check what PyTorch sees:
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

### Troubleshooting Conda Installation Failures

If `conda env create` fails with errors about missing CUDA packages:

**Problem**: Conda can't find compatible CUDA packages for PyTorch  
**Solution**: Use the step-by-step approach with `environment-minimal.yml`

```bash
# 1. Create minimal environment without PyTorch
conda env create -f environment-minimal.yml
conda activate graph-signal-diffusion

# 2. Manually install PyTorch for your CUDA version
# This is more reliable than letting conda auto-detect
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 3. Install PyTorch Geometric
conda install pyg -c pyg

# 4. Install remaining packages via pip
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv
pip install jaxtyping beartype accelerate wandb yfinance
pip install -e .
```

**Alternative**: If conda still fails, use pip entirely:
```bash
# Create empty conda environment
conda create -n graph-signal-diffusion python=3.11
conda activate graph-signal-diffusion

# Install via pip (replace cu121 with your CUDA version: cu118, cu121, cu124, or cpu)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
pip install numpy scipy pandas matplotlib seaborn networkx scikit-learn
pip install hydra-core omegaconf pyyaml tqdm pillow
pip install jaxtyping beartype accelerate wandb yfinance
pip install -e .
```

### Conda Installation (Auto-Detection)

When using `environment-flexible.yml`, conda will attempt to:
1. Detect your system's CUDA version
2. Install compatible PyTorch binaries
3. Build or download matching torch-scatter/sparse/cluster packages

```bash
# Conda attempts CUDA auto-detection
conda env create -f environment-flexible.yml
```

⚠️ **Note**: This may fail with errors about missing `cuda-cudart` or similar packages. If it fails, use `environment-minimal.yml` instead.

### Manual pip Installation for Specific CUDA Versions

If you prefer pip or need specific CUDA versions:

**CUDA 11.8:**
```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu118.html
```

**CUDA 12.1:**
```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
```

**CUDA 12.4:**
```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cu124.html
```

**CPU-only:**
```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.5.1+cpu.html
```

## System-Specific Notes

### Checking Your CUDA Version

```bash
# Check installed CUDA version
nvcc --version

# Check CUDA version PyTorch sees
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

For CPU-only:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
```

### Apple Silicon (M1/M2)

```bash
# Use conda for best performance
conda install pytorch torchvision -c pytorch

# Or use pip with appropriate version
pip install torch torchvision
```

## Verification

After installation, verify your setup:

```bash
# Test imports
python -c "
import torch
import torch_geometric
import hydra
import numpy as np
import pandas as pd
print(f'PyTorch: {torch.__version__}')
print(f'PyG: {torch_geometric.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print('✓ All imports successful')
"

# Run tests
pytest tests/

# Run a quick training test
python -m graph_signal_diffusion.cli.train --help
```

## Troubleshooting

### Issue: PyTorch Geometric Installation Fails

**Solution**: Install PyTorch first, then install PyG with matching versions:

```bash
# Check PyTorch version and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"

# Install PyG with matching version
pip install torch-geometric -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html
```

### Issue: Version Conflicts

**Solution**: Use the pinned requirements or create a fresh environment:

```bash
# Remove existing environment
conda env remove -n graph-signal-diffusion

# Recreate from scratch
conda env create -f environment.yml
```

### Issue: Missing CUDA Libraries

**Solution**: Ensure CUDA toolkit is installed and environment variables are set:

```bash
# Check CUDA
nvcc --version

# Set environment variables (add to ~/.bashrc)
export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH
```

## For Paper Reproducibility

To exactly reproduce results from a specific paper/experiment:

1. Check the paper's `requirements.txt` or commit hash
2. Use the exact pinned versions:
   ```bash
   pip install -r requirements.txt
   ```
3. Verify Python version matches (check `python --version`)
4. Use the same random seeds (specified in config files)

## Contributing

If you're contributing to this project:

1. Install development dependencies:
   ```bash
   pip install -e .
   pip install -r requirements-dev.txt
   ```

2. After adding new imports, update dependencies:
   ```bash
   # Update pyproject.toml with new package
   # Then regenerate requirements
   python scripts/generate_requirements.py
   ```

3. Run tests before submitting:
   ```bash
   pytest tests/
   black src/ tests/
   ruff check src/ tests/
   ```
