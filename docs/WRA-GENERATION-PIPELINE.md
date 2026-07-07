# WRA Generation Pipeline: Configuration and Data Flow

This document describes the three-stage WRA (Wireless Resource Allocation) generative dataset pipeline: channel analysis → PD training → diffusion dataset collection. It covers configuration inheritance, content-addressed hashing, and how outputs flow between stages.

## Overview

The pipeline has three stages, each with a distinct responsibility:

1. **Channel Analysis** (`analyze_channels.py`) — generates wireless channels and caches them
2. **PD Training** (`train_pd.py`) — trains a primal-dual expert on fixed channels
3. **Collection** (`build_diffusion_dataset.py`) — converts PD samples into raw WRA diffusion datasets

Each stage uses Hydra configuration files with scenario-specific settings (e.g., `wra_debug`, `wra_small`, `wra_medium`).

---

## Configuration Inheritance Chain

Configurations are composed hierarchically. The example below shows the `wra_debug` scenario:

```
pd_collection/wra_debug.yaml
  ├─ /pd_training/wra_debug.yaml
  │   ├─ /scenario/wra_debug.yaml
  │   │   ├─ /scenario/defaults.yaml
  │   │   │   └─ [defines channel_cache_key resolver]
  │   │   └─ [16 networks, 50 links, 2500m, seed=42, v3 channels]
  │   │
  │   ├─ /pd_training/defaults.yaml
  │   │   └─ [defines pd_run_key resolver, model/training hyperparams]
  │   │
  │   └─ _self_ [training overrides: r_min=0.7, alpha_dual=0.05, etc.]
  │
  ├─ /pd_collection/defaults.yaml
  │   └─ [defines pd_collection_key resolver, collection parameters]
  │
  └─ _self_ [no scenario-specific overrides]
```

### Key Points

- **Scenario parameters** (dataset, channel, system) are defined once in `/scenario/wra_{scenario}.yaml` and inherited by all three stages
- **Custom Hydra resolvers** are registered once at module import via `register_wra_hydra_resolvers()` in `channel_factory.py`
- **Each stage has its own defaults** that inherit the scenario params and add stage-specific parameters (e.g., model architecture for training)
- **Config keys** (`channel_cache_key`, `pd_run_key`, `pd_collection_key`) are available to all stages that inherit them

### Available Scenarios

Scenarios with `pd_collection` configs (and thus full-pipeline support):
- `wra_debug` — 16 networks, 50 links (lightweight, good for testing)
- `wra_small` — 8 networks, 20 links
- `wra_medium` — 32 networks, 100 links
- `wra_medium_low_density` — 32 networks, 20 links (sparse)
- `wra_medium_high_qos` — 32 networks, 100 links (high QoS variant)
- `wra_large_low_density` — 64 networks, 50 links (large, sparse)

---

## Content-Addressed Hashing

The pipeline uses three content-addressed keys that form a hierarchy. Each key hashes a set of parameters and becomes part of the output directory path.

### 1. `channel_cache_key` — Channel Configuration Hash

**Defined in:** `/scenario/defaults.yaml:44`

**Resolver:** `_wra_channel_cache_key_resolver()` in `channel_factory.py:513-553`

**Function:** `build_wra_channel_cache_metadata()` in `channel_factory.py:195-228`

**Parameters hashed (17 total):**
```
seed                          (int)
num_networks                  (int)
n_links                       (int)
deployment_range              (float)
channel_version               (str: v1, v2, or v3)
channel.min_tx_tx_distance    (float)
channel.min_tx_rx_distance    (float)
channel.max_tx_rx_distance    (float)
channel.path_loss_exponent_short  (float)
channel.path_loss_exponent_long   (float)
channel.shadowing_std         (float)
channel.carrier_freq          (float)
channel.speed                 (float)
channel.num_fading_paths      (int)
channel.delta_t               (float)
channel.max_deployment_attempts (int)
channel.max_recursion_depth   (int)
```

**Output format:**
```
wrach_v1_s{seed}_D{num_networks}_N{n_links}_R{deployment_range}_{version}_h{md5[:12]}
```

**Example for wra_debug:**
```
wrach_v1_s42_D16_N50_R2500_v3_h7a3f8e2c1b9d
```

**Used by:** All three stages (to ensure they use the same channels)

**Cache location:** `data/wra_channel_cache/{scenario_name}/{channel_cache_key}.pt`

---

### 2. `pd_run_key` — Training Configuration Hash

**Defined in:** `/pd_training/defaults.yaml:57`

**Resolver:** `_wra_pd_run_key_resolver()` in `channel_factory.py`

**Function:** `build_wra_pd_run_key()` in `channel_factory.py`

`pd_run_key` hashes system/model/training settings plus the upstream `channel_cache_key`.
For `constraint_profile`, hashing is **active-branch only**:

- `source=scalar`: hashes scalar value (`constraint_profile.scalar.value`, or `training.r_min` fallback)
- `source=explicit`: hashes `constraint_profile.explicit.profiles` only
- `source=sampled`: hashes `constraint_profile.sampled.{min,max,count,seed}`

Inactive branches are excluded from payload hashing.  
`constraint_profile.explicit.names` is cosmetic only and excluded from hashing.

Other hashed training fields include:
`alpha_dual`, `dual_momentum`, `learning_rate`, `batch_size`, `max_epochs`,
convergence settings, dual-update settings, and sample-collection settings.

**Output format**
```
wrpd_v1_{channel_cache_key}_r{r_token}_a{alpha_dual:g}_h{md5[:12]}
```

Where:
- `r_token = scalar r_min` for scalar mode
- `r_token = source` for non-scalar modes (`explicit` or `sampled`)

**Example (scalar, r_min=0.7, alpha_dual=0.05):**
```
wrpd_v1_wrach_v1_s42_D16_N50_R2500_v3_h7a3f8e2c1b9d_r0.7_a0.05_h9e4c5b1a2f3d
```

**Example (explicit profile mode):**
```
wrpd_v1_wrach_v1_s42_D16_N50_R2500_v3_h7a3f8e2c1b9d_rexplicit_a0.05_hxxxxxxxxxxxx
```

**Uniqueness:** Two PD training runs with different hyperparameters will have different `pd_run_key` values.

**Used by:** `train_pd.py` and `build_diffusion_dataset.py`

---

### 3. `pd_collection_key` — Sample Collection Configuration Hash

**Defined in:** `/pd_collection/defaults.yaml:47`

**Resolver:** `_wra_pd_collection_key_resolver()` in `channel_factory.py:627-646`

**Function:** `build_wra_pd_collection_key()` in `channel_factory.py:471-506`

**Parameters hashed (8 total, but conditional):**
```
pd_run_key                        (str, from upstream)
sample_source                     (str: "npz" or "primal_history")
target_samples_per_network        (int)

# Only included when sample_source == "primal_history":
window_size                       (int or null)
refine_feasible_subset            (bool)
refine_objective                  (str)
subset_feasibility_tolerance      (float)
subset_bottleneck_nodes           (int)
```

**Important:** When `sample_source == "npz"` (the default), the primal_history parameters are **excluded from the hash**. This ensures that two collection runs using NPZ with different windowing settings still get the same key (because the NPZ path doesn't use those parameters).

**Output format:**
```
wrpc_v1_{sample_source}_k{target_samples_per_network}_h{md5[:12]}
```

**Example for wra_debug (npz source, target=100 samples):**
```
wrpc_v1_npz_k100_h3e7a9c2b1d5f
```

**Used by:** `build_diffusion_dataset.py` (collection stage)

---

## Output Directory Structure

The three stages write to different locations, forming a hierarchy keyed on the content-addressed hashes.

### Stage 1: Channel Analysis

**Script:** `scripts/wra/debug/analyze_channels.sh`

**Hydra output directory:**
```
outputs/{scenario_name}/{channel_cache_key}/{date}/{time}/
```

**Example:**
```
outputs/wra_debug/wrach_v1_s42_D16_N50_R2500_v3_h7a3f8e2c1b9d/2026-02-25/14-32-15/
```

**Files written:**
- `.hydra/config.yaml` — Hydra config (resolved)
- `.hydra/hydra.yaml` — Hydra runtime config
- `analysis_report.html` — Channel visualization (if enabled)

**Channel cache written to:**
```
data/wra_channel_cache/{scenario_name}/{channel_cache_key}.pt
```

**Example:**
```
data/wra_channel_cache/wra_debug/wrach_v1_s42_D16_N50_R2500_v3_h7a3f8e2c1b9d.pt
```

### Stage 2: PD Training

**Script:** `scripts/wra/debug/train_pd.sh`

**Hydra output directory:**
```
outputs/{scenario_name}/{pd_run_key}/{date}/{time}/
```

**Example:**
```
outputs/wra_debug/wrpd_v1_wrach_v1_s42_D16_N50_R2500_v3_h7a3f8e2c1b9d_r0.7_a0.05_h9e4c5b1a2f3d/2026-02-25/14-45-22/
```

**Files written:**
- `.hydra/config.yaml` — Hydra config (resolved) — **this is crucial for Stage 3**
- `.hydra/hydra.yaml` — Hydra runtime config
- `checkpoints/` — model checkpoints
- `collected_samples.npz` — Polyak-averaged samples (schema v2)
- `primal_history.jsonl` — raw training trajectory (line-delimited JSON)
- `collection_metadata.json` — metadata about collected samples

**Channel cache read from:**
```
data/wra_channel_cache/{scenario_name}/{channel_cache_key}.pt
```

(Same cache file as Stage 1)

### Stage 3: Diffusion Dataset Collection

**Script:** `scripts/wra/debug/build_diffusion_dataset.sh input_dir=outputs/wra_debug/...`

**Hydra output directory (logs only):**
```
outputs/{scenario_name}/{pd_collection_key}/{date}/{time}/
```

**Example:**
```
outputs/wra_debug/wrpc_v1_npz_k100_h3e7a9c2b1d5f/2026-02-25/14-50-10/
```

**Raw WRA dataset written to:**
```
data/wra/{scenario}/{pd_collection_key}/raw/
```

**Where:**
- `{scenario}` = scenario name with `wra_` prefix stripped (e.g., `wra_debug` → `debug`)
- `{pd_collection_key}` = content-addressed key (e.g., `pdc_npz_3e7a9c2b1d5f`)

**Example:**
```
data/wra/debug/pdc_npz_3e7a9c2b1d5f/raw/
```

**Files written:**
- `.hydra/config.yaml` — copy of Stage 2's config
- `collected_samples.npz` — raw WRA dataset
- `config.yaml` — same as above (for convenience)
- `network_info.json` — collection metadata and statistics

---

## How to Hand Off Between Stages

### Quick Start Example (wra_debug)

```bash
# Stage 1: Analyze channels (creates cache, one-time per scenario)
bash scripts/wra/debug/analyze_channels.sh

# Stage 2: Train PD expert
bash scripts/wra/debug/train_pd.sh
# Output will print:
#   ✓ Training complete
#   Output: outputs/wra_debug/<pd_run_key>/<date>/<time>

# Copy the output path from Stage 2, use as input to Stage 3:
bash scripts/wra/debug/build_diffusion_dataset.sh \
    input_dir=outputs/wra_debug/<pd_run_key>/<date>/<time>

# Raw WRA dataset is now in: data/wra/debug/<pd_collection_key>/raw/
```

### Detailed Workflow

#### Step 1: Channel Analysis (Optional, One-Time)

Run this once per scenario to pre-generate and cache channels:

```bash
bash scripts/wra/debug/analyze_channels.sh
```

**What happens:**
1. Loads scenario config (`wra_debug`)
2. Computes `channel_cache_key` from scenario params
3. Checks if `data/wra_channel_cache/wra_debug/{cache_key}.pt` exists
4. If not, generates 5 networks and saves cache file
5. Outputs to `outputs/wra_debug/{cache_key}/<date>/<time>/`

**Output:** Cache file is ready for Steps 2 and 3.

#### Step 2: Train PD Expert

```bash
bash scripts/wra/debug/train_pd.sh
```

**What happens:**
1. Loads scenario config + training config (`pd_training/wra_debug`)
2. Computes `channel_cache_key` (same as Step 1)
3. Loads pre-generated channels from `data/wra_channel_cache/wra_debug/{cache_key}.pt`
4. Computes `pd_run_key` from all training hyperparameters
5. Trains GNN with primal-dual optimization for 20,000 epochs
6. Saves to `outputs/wra_debug/{pd_run_key}/<date>/<time>/`:
   - `.hydra/config.yaml` — the resolved config
   - `collected_samples.npz` — Polyak samples
   - `primal_history.jsonl` — full trajectory
7. Prints output directory at completion

**Next step:** Note the output directory path for Step 3.

#### Step 3: Build Diffusion Dataset

```bash
PD_RUN_DIR="outputs/wra_debug/<pd_run_key>/<date>/<time>"
bash scripts/wra/debug/build_diffusion_dataset.sh \
    input_dir="$PD_RUN_DIR"
```

**What happens:**
1. Loads `$PD_RUN_DIR/.hydra/config.yaml` saved by Stage 2
2. Computes `channel_cache_key` from the saved config (same as Stages 1 & 2)
3. Loads channels from `data/wra_channel_cache/wra_debug/{cache_key}.pt`
4. Loads samples from `$PD_RUN_DIR/collected_samples.npz`
5. Computes `pd_collection_key` from sample source (default: npz) and target count (default: 100)
6. Converts samples to raw WRA dataset format
7. Writes to `data/wra/debug/<pd_collection_key>/raw/`:
   - `collected_samples.npz` — raw dataset
   - `network_info.json` — metadata and statistics

**Output:** Raw WRA dataset is ready for diffusion model training.

### Optional Overrides

Each script accepts Hydra parameter overrides:

```bash
# Force overwrite if dataset already exists
bash scripts/wra/debug/build_diffusion_dataset.sh \
    input_dir=outputs/wra_debug/... \
    output.force=true

# Sample only 50 networks instead of all
bash scripts/wra/debug/build_diffusion_dataset.sh \
    input_dir=outputs/wra_debug/... \
    collection.target_samples_per_network=50

# Write to an explicit directory (bypasses content-addressed layout)
bash scripts/wra/debug/build_diffusion_dataset.sh \
    input_dir=outputs/wra_debug/... \
    output.raw_wra_dir=/path/to/custom/dir
```

---

## Consistency Guarantees

The three-stage pipeline ensures reproducibility and consistency through content-addressing:

1. **Same channels across all stages:** `channel_cache_key` is computed identically in `analyze_channels.py`, `train_pd.py`, and `build_diffusion_dataset.py` using the same `build_wra_channel_cache_metadata()` function. If parameters change, the key changes, so cached channels are invalidated.

2. **Unique training runs:** `pd_run_key` is determined by 34 parameters (including the upstream `channel_cache_key`). Different hyperparameters = different keys = different output directories = no overwrite risk.

3. **Collection variants from one training run:** `pd_collection_key` depends on the sample source and collection parameters. One PD training run can produce multiple collection variants (e.g., npz with k=50, npz with k=100, primal_history with windowing) with different keys and different output datasets.

4. **Directory isolation:** Each stage writes to a uniquely-keyed subdirectory within `outputs/{scenario_name}/`, so multiple runs don't interfere:
   ```
   outputs/wra_debug/
     ├─ wrach_v1_.../  (cache key namespace)
     ├─ wrpd_v1_.../   (training key namespace)
     └─ wrpc_v1_.../   (collection key namespace)
   ```

---

## Troubleshooting

### Channel cache not found

If `train_pd.sh` or `build_diffusion_dataset.sh` prints:
```
Channel cache not found at data/wra_channel_cache/...; will reconstruct channels.
```

This is **normal** if you skipped `analyze_channels.sh`. The stages will reconstruct channels on-demand (slower, but works). To pre-cache for future runs, run `analyze_channels.sh` once.

### Config mismatch

If you see:
```
Channel cache metadata mismatch; will reconstruct channels
```

The saved cache was created with different scenario parameters. This usually means:
- You changed `wra_debug` scenario definition after creating the cache
- Or you're pointing at a cache from a different scenario

**Fix:** Delete the mismatched cache and re-run `analyze_channels.sh`.

### Missing input_dir

```
Usage: build_diffusion_dataset.sh input_dir=<path> [...]
```

You forgot to provide the PD training output directory. Use:
```bash
bash scripts/wra/debug/build_diffusion_dataset.sh \
    input_dir=outputs/wra_debug/<pd_run_key>/<date>/<time>
```

### Dataset already exists

```
Output files already exist in: data/wra/debug/<pd_collection_key>/raw/
```

The content-addressed output directory already exists. Since the key is deterministic,
re-running with the same inputs will always map to the same directory. Choose one:
- Force overwrite: `output.force=true`
- Write to a custom directory: `output.raw_wra_dir=/path/to/custom/dir`

---

## References

**Configuration files:**
- Scenarios: `src/graph_signal_diffusion/conf/wra_generation/scenario/`
- Channel analysis: `src/graph_signal_diffusion/conf/wra_generation/channel_analysis/`
- PD training: `src/graph_signal_diffusion/conf/wra_generation/pd_training/`
- Collection: `src/graph_signal_diffusion/conf/wra_generation/pd_collection/`

**Scripts:**
- `scripts/wra/{scenario}/analyze_channels.sh`
- `scripts/wra/{scenario}/train_pd.sh`
- `scripts/wra/{scenario}/build_diffusion_dataset.sh`

**Source code:**
- `src/graph_signal_diffusion/cli/wra/analyze_channels.py`
- `src/graph_signal_diffusion/cli/wra/train_pd.py`
- `src/graph_signal_diffusion/cli/wra/build_diffusion_dataset.py`
- `src/graph_signal_diffusion/datasets/wra/channel_factory.py` (key resolvers and hashing logic)
