# Dependency Management Setup Summary

## Overview

This project now implements a **principled, systematic, and flexible** dependency management system that balances reproducibility with development flexibility.

## Files Created/Updated

### 1. Core Dependency Files

#### `pyproject.toml` (Updated)
- **Purpose**: Source of truth for project dependencies
- **Versions**: Minimum version requirements (e.g., `>=2.0.0`)
- **Use case**: Development and package distribution
- **Installation**: `pip install -e .`
- **Advantages**: 
  - Allows users to use newer compatible versions
  - Flexible for different environments
  - Standard Python packaging format

#### `requirements.txt` (Created)
- **Purpose**: Exact reproducibility of research results
- **Versions**: Pinned versions (e.g., `==1.12.1`)
- **Use case**: Replicating paper results exactly
- **Installation**: `pip install -r requirements.txt`
- **Advantages**:
  - Guarantees same package versions
  - Eliminates version-related variability
  - Essential for scientific reproducibility

#### `requirements-dev.txt` (Created)
- **Purpose**: Development tools (linters, formatters, testing)
- **Use case**: Contributors and maintainers
- **Installation**: `pip install -r requirements-dev.txt`
- **Includes**: pytest, black, ruff, mypy, jupyter

#### `environment.yml` (Created)
- **Purpose**: Conda environment specification
- **Use case**: Users preferring Conda, GPU users
- **Installation**: `conda env create -f environment.yml`
- **Advantages**:
  - Handles CUDA/PyTorch compatibility automatically
  - Better for managing system-level dependencies
  - Cross-platform compatibility

### 2. Utility Scripts

#### `scripts/generate_requirements.py` (Created)
- **Purpose**: Generate pinned requirements from current environment
- **Usage**: `python scripts/generate_requirements.py`
- **What it does**:
  - Analyzes project imports
  - Extracts installed versions
  - Generates requirements.txt with pinned versions
- **When to use**: Before publishing results or major releases

#### `scripts/verify_install.py` (Created)
- **Purpose**: Verify installation completeness
- **Usage**: `python scripts/verify_install.py`
- **What it checks**:
  - All required packages are installed
  - Package versions
  - Python version compatibility
  - CUDA availability
- **When to use**: After installation, before running experiments

### 3. Documentation

#### `INSTALL.md` (Created)
- **Comprehensive installation guide**
- **Covers**:
  - Multiple installation methods (conda, pip, exact/flexible)
  - System-specific instructions (CUDA, Apple Silicon, etc.)
  - Troubleshooting common issues
  - Verification steps
  - Dependency management philosophy

#### `REPRODUCIBILITY.md` (Created)
- **Guide for exact result reproduction**
- **Covers**:
  - How to publish reproducible results
  - How to reproduce published results
  - Common reproducibility issues and solutions
  - Docker setup for maximum reproducibility
  - Experiment tracking best practices
  - Reproducibility checklist

#### `README.md` (Updated)
- **Added comprehensive installation section**
- **Links to detailed guides**
- **Explains dependency management approach**

### 4. Supporting Files

#### `Makefile` (Created)
- **Convenient shortcuts for common tasks**
- **Commands**:
  - `make install`: Flexible installation
  - `make install-exact`: Exact versions
  - `make install-dev`: With dev tools
  - `make verify`: Verify installation
  - `make test`: Run tests
  - `make lint`/`make format`: Code quality
  - `make requirements`: Regenerate requirements.txt
  - `make clean`: Remove build artifacts

#### `.python-version` (Created)
- **Specifies recommended Python version**: 3.9.12
- **Used by**: pyenv and other version managers
- **Purpose**: Ensure consistent Python version across team

## Dependency Management Workflow

### For Users (Reproducing Results)

```bash
# 1. Clone repository
git clone <repo-url>
cd graph-signal-diffusion

# 2. Install with exact versions
pip install -r requirements.txt
pip install -e .

# 3. Verify installation
python scripts/verify_install.py

# 4. Run experiments with same seeds/configs
python -m graph_signal_diffusion.cli.train --config-name <config>
```

### For Developers (Contributing)

```bash
# 1. Clone repository
git clone <repo-url>
cd graph-signal-diffusion

# 2. Install in development mode
pip install -e .
pip install -r requirements-dev.txt

# 3. Verify installation
make verify

# 4. Make changes, run tests
make test
make lint
make format
```

### For Maintainers (Publishing Results)

```bash
# 1. Ensure environment is clean and up-to-date
pip install -e .

# 2. Run experiments
python -m graph_signal_diffusion.cli.train ...

# 3. Generate exact requirements
python scripts/generate_requirements.py

# 4. Commit updated requirements
git add requirements.txt
git commit -m "Update requirements for paper submission"

# 5. Document experiment details
# - Git commit hash
# - Python version
# - Hardware specs
# - Random seeds
```

## Three-Tier Dependency Strategy

### Tier 1: Development (pyproject.toml)
```bash
pip install -e .
```
- **Flexibility**: High
- **Reproducibility**: Medium
- **Use case**: Daily development, testing new features
- **Version constraints**: Minimum versions with `>=`

### Tier 2: Exact Reproducibility (requirements.txt)
```bash
pip install -r requirements.txt
```
- **Flexibility**: None
- **Reproducibility**: Maximum
- **Use case**: Reproducing published results
- **Version constraints**: Pinned with `==`

### Tier 3: Conda Environment (environment.yml)
```bash
conda env create -f environment.yml
```
- **Flexibility**: Medium
- **Reproducibility**: High
- **Use case**: GPU users, system compatibility
- **Version constraints**: Mix of pinned and flexible

## Package Discovery Process

The dependencies were identified through:

1. **Static analysis**: Scanning all Python files for import statements
2. **Pylance analysis**: Using VS Code's Python language server
3. **Manual verification**: Checking which packages are actually used
4. **Version extraction**: Getting installed versions from environment

### Core Dependencies Identified

- **Deep Learning**: torch, torchvision, torch-geometric, torch-scatter
- **Scientific**: numpy, scipy, pandas
- **Visualization**: matplotlib, seaborn, networkx
- **ML Tools**: scikit-learn
- **Config**: hydra-core, omegaconf, pyyaml
- **Type Safety**: jaxtyping, beartype
- **Training**: accelerate, tqdm
- **Tracking**: wandb
- **Data**: yfinance
- **Images**: pillow

## Key Advantages of This Approach

1. **Systematic**: Uses tools to automatically discover dependencies
2. **Flexible**: Multiple installation options for different use cases
3. **Reproducible**: Pinned versions for exact result replication
4. **Documented**: Comprehensive guides for all scenarios
5. **Maintainable**: Easy to update and regenerate requirements
6. **User-friendly**: Clear instructions for different user types
7. **Best practices**: Follows Python packaging standards

## Continuous Maintenance

### Regular Updates
```bash
# Quarterly or before major releases:
python scripts/generate_requirements.py
git add requirements.txt
git commit -m "Update dependencies"
```

### Security Updates
```bash
# Check for security vulnerabilities
pip-audit requirements.txt

# Or use safety
safety check -r requirements.txt
```

### Dependency Graph Analysis
```bash
# Visualize dependency tree
pipdeptree

# Check for conflicts
pip check
```

## Comparison with Alternatives

### ❌ Only conda env export
- **Problem**: Includes ALL packages (too many dependencies)
- **Problem**: Platform-specific (not portable)
- **Problem**: Includes build numbers (brittle)

### ❌ Only pip freeze
- **Problem**: Includes transitive dependencies
- **Problem**: No flexibility for development
- **Problem**: Hard to maintain

### ✅ Our Multi-Layered Approach
- **Advantage**: Separates direct from transitive dependencies
- **Advantage**: Provides flexibility AND reproducibility
- **Advantage**: Well-documented and maintainable
- **Advantage**: Supports multiple installation methods

## Next Steps for Users

1. **Read**: Start with [INSTALL.md](INSTALL.md)
2. **Choose**: Pick installation method based on your needs
3. **Install**: Follow the instructions
4. **Verify**: Run `python scripts/verify_install.py`
5. **Experiment**: Run the examples in README.md
6. **Reproduce**: Follow [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for exact results

## Next Steps for Maintainers

1. **Review**: Check that all dependencies are correctly listed
2. **Test**: Try fresh installation in clean environment
3. **Document**: Add any project-specific dependency notes
4. **Automate**: Consider adding CI/CD to verify installations
5. **Update**: Regenerate requirements before releases
