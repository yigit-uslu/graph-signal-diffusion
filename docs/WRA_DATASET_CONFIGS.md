# WRA Dataset Configuration System

## Overview

The WRA dataset now supports a flexible configuration system that allows you to:
- Use friendly names ('small', 'medium', 'large') from a registry
- Create custom configurations programmatically
- Support multiple versions/variants of the same base configuration
- Maintain independent processing for each dataset
- **Automatic version detection** to prevent accidental overwrites

## Conversion Script Versioning

The conversion script now includes automatic version management:

### Auto-versioning (Recommended)

Automatically detects existing data and creates new versions (v1, v2, v3, ...):

```bash
python scripts/convert_pd_samples_to_diffusion.py \
    --input outputs/2026-01-23/18-17-30/collected_samples.npz \
    --output-root data/wra \
    --auto-version
```

**Behavior:**
- If no data exists: saves to `N_50_.../collected_samples.npz`
- If data exists: detects highest version (e.g., v2) and saves to `N_50_.../v3/`
- **Never overwrites** existing data

### Explicit Versioning

Specify a custom version string:

```bash
python scripts/convert_pd_samples_to_diffusion.py \
    --input outputs/2026-01-23/18-17-30/collected_samples.npz \
    --output-root data/wra \
    --version primal_dual_lr0.001
```

Saves to: `data/wra/raw/N_50_.../primal_dual_lr0.001/`

### Force Overwrite

Overwrite existing data (use with caution):

```bash
python scripts/convert_pd_samples_to_diffusion.py \
    --input outputs/2026-01-23/18-17-30/collected_samples.npz \
    --output-root data/wra \
    --force
```

## Usage Examples

### 1. Using Friendly Names

```python
from src.graph_signal_diffusion.datasets.wra import WRADataset

# Use predefined 'small' configuration
dataset = WRADataset(
    root='data/wra',
    dataset_names=['small'],  # Resolves to full name automatically
    split='train'
)

# Use multiple standard configs
dataset = WRADataset(
    root='data/wra',
    dataset_names=['small', 'medium'],
    split='train'
)
```

### 2. Using Full Dataset Names (backward compatible)

```python
dataset = WRADataset(
    root='data/wra',
    dataset_names=['N_50_density_102.0_seed_42_Pmax_dBm_10.0_rmin_0.5'],
    split='train'
)
```

### 3. Creating from Configuration Objects

```python
from src.graph_signal_diffusion.datasets.wra import WRAConfig, WRADataset

# Define custom configuration
config = WRAConfig(
    n_links=75,
    density=120.0,
    network_seed=42,
    P_max_dBm=12.0,
    r_min=0.8,
    deployment_range=1200.0,
    path_loss_exponent=3.5
)

# Create dataset from config
dataset = WRADataset.from_config(config, split='train')
```

### 4. Using Versions for Different Training Runs

```python
# Different training runs of the same network configuration
dataset = WRADataset(
    root='data/wra',
    dataset_names=[
        'N_50_density_102.0_seed_42_Pmax_dBm_10.0_rmin_0.5/v1',
        'N_50_density_102.0_seed_42_Pmax_dBm_10.0_rmin_0.5/v2',
        'N_50_density_102.0_seed_42_Pmax_dBm_10.0_rmin_0.5/primal_dual_lr0.001',
    ],
    split='train'
)

# Or using from_config with version
config = WRAConfig(n_links=50, density=102.0, network_seed=42, 
                   P_max_dBm=10.0, r_min=0.5)
dataset = WRADataset.from_config(config, version='v1', split='train')
```

### 5. Listing Available Configs

```python
from src.graph_signal_diffusion.datasets.wra import list_standard_configs

# See all predefined configs
configs = list_standard_configs()
print(configs)
# {'small': 'N_50_density_102.0_seed_42_Pmax_dBm_10.0_rmin_0.5',
#  'medium': 'N_100_density_150.0_seed_42_Pmax_dBm_15.0_rmin_1.0',
#  'large': 'N_200_density_200.0_seed_42_Pmax_dBm_20.0_rmin_1.5'}
```

### 6. Parsing Existing Dataset Names

```python
from src.graph_signal_diffusion.datasets.wra import WRAConfig

# Parse parameters from dataset name
name = "N_50_density_102.0_seed_42_Pmax_dBm_10.0_rmin_0.5"
config = WRAConfig.from_dataset_name(name)

print(config.n_links)        # 50
print(config.density)        # 102.0
print(config.network_seed)   # 42
```

## Directory Structure

The system supports flexible organization:

```
data/wra/
├── raw/
│   ├── N_50_density_102.0_seed_42_Pmax_dBm_10.0_rmin_0.5/
│   │   ├── collected_samples.npz           # Default version
│   │   ├── v1/
│   │   │   └── collected_samples.npz       # Version 1
│   │   └── primal_dual_lr0.001/
│   │       └── collected_samples.npz       # Custom training run
│   └── N_100_density_150.0_seed_42_Pmax_dBm_15.0_rmin_1.0/
│       └── collected_samples.npz
└── processed/
    ├── N_50_density_102.0_seed_42_Pmax_dBm_10.0_rmin_0.5/
    │   ├── metadata.json
    │   └── network_*/
    ├── N_50_density_102.0_seed_42_Pmax_dBm_10.0_rmin_0.5/v1/
    │   └── ...
    └── N_100_density_150.0_seed_42_Pmax_dBm_15.0_rmin_1.0/
        └── ...
```

## Standard Configurations

Three predefined configurations are available:

| Name   | n_links | density | P_max_dBm | r_min | deployment_range |
|--------|---------|---------|-----------|-------|------------------|
| small  | 50      | 102.0   | 10.0      | 0.5   | 1000.0          |
| medium | 100     | 150.0   | 15.0      | 1.0   | 1500.0          |
| large  | 200     | 200.0   | 20.0      | 1.5   | 2000.0          |

All use `network_seed=42` and `path_loss_exponent=3.5` by default.

## Adding Custom Configurations

To add your own standard configuration:

```python
# In configs.py, add to STANDARD_CONFIGS:
STANDARD_CONFIGS['my_custom'] = WRAConfig(
    n_links=150,
    density=175.0,
    network_seed=42,
    P_max_dBm=18.0,
    r_min=1.2,
    deployment_range=1800.0,
)
```

## Benefits

✅ **Clean separation**: Each dataset name processes independently  
✅ **Flexible versioning**: Support multiple training runs of same config  
✅ **Discoverable**: Registry provides standard configs  
✅ **Backward compatible**: Existing code still works  
✅ **Self-documenting**: Names encode all parameters  
✅ **Type-safe**: WRAConfig validates parameters
