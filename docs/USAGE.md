# Usage & Development Guide

Setup, configuration, and usage reference for the Graph Signal Generative Diffusion
Modeling framework. This detail was moved out of the top-level [README](../README.md)
(which now focuses on the applications) — nothing here is new, just relocated.

## Project structure

```
├── configs/              # Task-specific training configurations
│   ├── wra_*.yaml        # Wireless resource allocation configs
│   └── pd_*.yaml         # Primal-dual optimization configs
├── data/                 # Dataset storage (raw data tracked in git)
│   ├── sp100/            # S&P 100 stock data
│   ├── sp500/            # S&P 500 stock data
│   └── wra/              # Wireless resource allocation data
├── docs/                 # Documentation and implementation guides
├── reproduce/            # End-to-end reproduction packages (per experiment)
├── assets/               # Figures for the README (regular git, not LFS)
├── scripts/              # Training and evaluation scripts
├── src/
│   └── graph_signal_diffusion/
│       ├── cli/          # Command-line interfaces
│       ├── datasets/     # Dataset implementations
│       ├── diffusion/    # Diffusion model implementations
│       ├── models/       # Neural network architectures (incl. ugnn/)
│       ├── tasks/        # Task definitions
│       ├── trainers/     # Training logic
│       └── conf/         # Hydra configuration schemas
└── tests/                # Unit and integration tests
```

## Supported datasets, models, and tasks

- **Datasets**: `SP100`, `SP500` (stock price time series), `CIFAR10` (testing),
  `WRA` (wireless resource allocation, custom-generated).
- **Models**: `UGNN` — U-Net-style Graph Neural Network with hierarchical
  pooling/unpooling, encoder–decoder architecture with skip connections, multi-hop
  graph convolutions with configurable strides, and time-embedding / conditional
  generation support. (See the [U-GNN section](../assets/ugnn/README.md) and
  [docs/UGNN_ARCHITECTURE.md](UGNN_ARCHITECTURE.md).)
- **Tasks**: **Stock Price Forecasting** (predict future returns; graph from stock
  correlations; configurable horizons) and **Wireless Resource Allocation**
  (primal-dual power allocation with GNN policies under QoS constraints).

## Installation

This project uses a **multi-layered dependency management system** that balances
reproducibility with development flexibility.

**Prerequisites**: Python 3.9+ (3.11.11 in development), CUDA 12.4 for GPU
(CPU-only also supported), Conda recommended for PyTorch + CUDA.

### For paper reproducibility (exact versions)

```bash
git clone https://github.com/yigit-uslu/graph-signal-diffusion.git
cd graph-signal-diffusion
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt                     # PyTorch 2.5.1, CUDA 12.4
pip install -e .
python scripts/verify_install.py
```

### For development (flexible versions)

```bash
git clone <repo-url> && cd graph-signal-diffusion
python -m venv .venv && source .venv/bin/activate
pip install -e .
pip install -r requirements-dev.txt                 # optional dev tools
```

### For Conda users (recommended for GPU)

```bash
git clone <repo-url> && cd graph-signal-diffusion
conda env create -f environment.yml
conda activate graph-signal-diffusion
python scripts/verify_install.py
```

### Dependency files

| File | Purpose | When to use | Constraints |
|------|---------|-------------|-------------|
| `requirements.txt` | Exact reproducibility | Replicating paper results | Pinned (`==`) |
| `pyproject.toml` | Development & distribution | Daily development | Minimum (`>=`) |
| `environment.yml` | Conda environment | GPU users | Mixed |

Key development versions: PyTorch 2.5.1+cu124, PyTorch Geometric 2.5.2,
Python 3.11.11, CUDA 12.4. For comprehensive instructions and troubleshooting see
[INSTALL.md](../INSTALL.md), [REPRODUCIBILITY.md](../REPRODUCIBILITY.md), and
[DEPENDENCY_SETUP.md](../DEPENDENCY_SETUP.md).

### Verification & Makefile

```bash
python scripts/verify_install.py
python -c "import torch, torch_geometric; print('✓ Installation successful')"
python -m graph_signal_diffusion.cli.train --help

make install | install-exact | install-dev | verify | test   # see `make help`
```

Common issues: CUDA version mismatch, torch-geometric install order, and
paper-mismatch → use pinned `requirements.txt`; see
[INSTALL.md](../INSTALL.md) / [REPRODUCIBILITY.md](../REPRODUCIBILITY.md).

## Quick start

```bash
# Stock price forecasting (SP500)
python -m graph_signal_diffusion.cli.train \
    task=stock_price_forecasting_v2 dataset=sp500 model=ugnn trainer.epochs=100

# Wireless resource allocation
python -m graph_signal_diffusion.cli.train \
    --config-path ../../../configs --config-name wra_medium_pd training.r_min=0.6

# Primal-dual power allocation script
python scripts/train_primal_dual_power_allocation.py \
    --config-path ../configs --config-name wra_small_pd
```

## Configuration

The framework uses [Hydra](https://hydra.cc/) for hierarchical configuration
(`src/graph_signal_diffusion/conf/{dataset,model,task,trainer}/`). Compose via the
defaults list and override from the CLI:

```yaml
defaults:
  - dataset: sp500
  - model: ugnn
  - task: stock_price_forecasting_v2
  - trainer: default
```

```bash
python -m graph_signal_diffusion.cli.train \
    dataset.batch_size=64 trainer.learning_rate=1e-3 model.base_channels=128
```

## Advanced usage

**Custom dataset**: create a class in `datasets/your_dataset/`, implement the
`Dataset` interface, register it in `__init__.py`, add a config under
`conf/dataset/your_dataset.yaml`.

**Custom model**: create it under `models/your_model/`, inherit `nn.Module`,
register in `models/__init__.py`, add `conf/model/your_model.yaml`.

## Output structure

```
outputs/
└── task_name/dataset_name/model_name/YYYY-MM-DD/HH-MM-SS/
    ├── checkpoints/     # Model checkpoints
    ├── samples/         # Generated samples
    ├── train.log        # Training logs
    └── .hydra/          # Configuration snapshots
```

## Testing

```bash
pytest                              # all tests
pytest tests/test_datasets.py       # a specific suite
pytest --cov=graph_signal_diffusion # with coverage
```
