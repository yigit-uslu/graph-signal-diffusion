# Reproducing `rigorous-quoll-131` (WRA / TSP)

End-to-end pipeline to reproduce the wireless-resource-allocation diffusion run
`rigorous-quoll-131` — the **U-GNN DS8** model (`ugnn_wra_v3_ds8_norm_act_head`,
learned STE pooling, `model_cond_channels=2`) trained to imitate a primal-dual
(PD) expert across 4 network densities at a single QoS level `r_min=0.6`.
Reproducibility anchor: **`repro/wra-tsp` @ `f53f7b0`**.

Every script sources [`00_config.sh`](00_config.sh) and is location-independent
(it resolves the repo root from its own path). Run them from anywhere; each `cd`s
to the project root. Python always goes through the pinned conda env (`CONDA_ENV`,
default **`graph-signal-diffusion`**) with the `LD_LIBRARY_PATH` preamble the WRA
plotting CLIs need.

---

## Pipeline at a glance

| # | Script | Stage | Reproducible? |
|---|--------|-------|---------------|
| 1 | `01_analyze_channels.sh` | channel / scenario generation (4 densities) | ~ seeded, but part of the expensive **regeneration** path (§2) |
| 2 | `02_train_pd.sh` | primal-dual expert training (4 densities @ r_min=0.6) | ❌ expensive; regeneration only (§2) |
| 3 | `03_build_dataset.sh` | collect PD samples → diffusion sub-datasets | ~ deterministic transform of the PD outputs (§2) |
| 4 | `04_train_diffusion.sh` | diffusion training, 5000 ep, seed=0 | ~ **statistical** only — expensive (§3) |
| 5 | `05_evaluate_checkpoint.sh` | eval shipped checkpoint → `test_summary.json` | ✅ **exact numbers** (§3) |
| 6 | `06_compare_baselines.sh` | FP/AP vs diffusion + transferability (paper figs) | ✅ checkpoint-based (§3) |
| — | `verify_dataset_manifest.sh` | structural integrity of the referenced dataset | the §2 check, re-runnable |

## Quick start (the reproducible path)

```bash
cd <repo-root>
REPRO=reproduce/wra-rigorous-quoll

# 0. Obtain the referenced ~39 GB dataset (public GitHub Release, no LFS, no auth)
#    and restore it under data/wra, then confirm it matches the manifest.
$REPRO/download_dataset.sh                        # curl download + verify + extract
#    (already have the data? skip the download and just run:)
#    $REPRO/verify_dataset_manifest.sh            # expect: PASS
sha256sum -c $REPRO/checksums/checkpoint.sha256   # expect: OK

# 1. Reproduce the paper's test metrics from the shipped rank-1 checkpoint (§3).
$REPRO/05_evaluate_checkpoint.sh
#    -> compare against outputs/.../rigorous-quoll-131/test_summary.json

# 2. (optional) Reproduce the baseline comparison + transferability figures.
$REPRO/06_compare_baselines.sh
```

Prerequisites: the `graph-signal-diffusion` conda env and one CUDA GPU. The 8 MB
rank-1 checkpoint and the run's `.hydra/` config **travel with the repo** (§4); the
expert dataset is referenced, not bundled (§2).

---

## §2 — The expert dataset is referenced, not bundled

`rigorous-quoll-131` trains on 4 **content-addressed** sub-datasets (one per
density) built from a primal-dual expert:

```
data/wra/medium-large_outdoor_{ultra-low,low,mid,high}_density/wrpc_v1_primal_history_k200_h<hash>/
```

Pinned in
[`conf/dataset/wra_medium-large_outdoor_all_density.yaml`](../../src/graph_signal_diffusion/conf/dataset/wra_medium-large_outdoor_all_density.yaml).
Together they are **~39 GB** (the full `data/wra` tree, including the PD/channel
**regeneration** intermediates, is ~136 GB). That is far too large for the git tree,
and we do not use git-LFS — so the dataset is hosted as **split tar parts on a
public GitHub Release** in the shared **`gsd-dataset`** data repo (≤2 GB each, plain
release assets) and fetched by `download_dataset.sh` with anonymous `curl` (no
GitHub account or token needed):

```bash
reproduce/wra-rigorous-quoll/download_dataset.sh          # curl download + verify + extract (no auth)
# already have a copy elsewhere? symlink it instead of downloading:
ln -s /path/to/your/wra data/wra && \
  reproduce/wra-rigorous-quoll/verify_dataset_manifest.sh # PASS
```

`cli.test` resolves the dataset at `<repo>/data/wra`, so keep it there (or symlink).
The dataset owner (re)publishes the Release into the shared public **`gsd-dataset`**
repo with `package_dataset.sh` (streams one sub-dataset at a time so peak disk is
~9 GB; parts are verified against `checksums/dataset_archives.sha256`; the exact
release tag is pinned in `00_config.sh`, its content id derived + verified by
`dataset_bundle_id.sh` — see [`DATASET_TAG.md`](DATASET_TAG.md)). Consumers need ~78 GB free (parts + extracted).

The `...k200_h<hash>` suffix is a **content hash** of each sub-dataset's build
inputs, so the path names themselves anchor identity; `verify_dataset_manifest.sh`
adds a structural check (present, file count, total bytes) against
[`checksums/dataset_manifest.txt`](checksums/dataset_manifest.txt).

**Regenerating it from scratch (Stages 1–3, expensive).** `01_analyze_channels.sh`
generates the seeded (seed=42) networks + channels; `02_train_pd.sh` trains the 4
PD experts at r_min=0.6 (per-density epoch budgets: ultra-low/low 10K, mid 20K,
high 30K — many GPU-hours each); `03_build_dataset.sh` collects each expert's
primal history (window 1000, 200 samples/network, feasible-subset refined) into
the diffusion sub-datasets. Because PD solves and channel draws are seeded but not
guaranteed bitwise across hardware, treat regeneration as **provenance** — the
exact-numbers path (§3) uses the referenced frozen dataset. These stages mirror the
SPAWC pipeline `scripts/wra/medium-large/sophisticated-oarfish-9/` (which sweeps 5
r_min values; this single-r_min arm needs only r_min=0.6).

## §3 — Results: exact (checkpoint) vs. statistical (retrain)

**Exact numbers → `05_evaluate_checkpoint.sh` (recommended).** Runs the same
held-out test evaluation the trainer ran, on the shipped `best_model_epoch_1600.pt`
(the run's rank-1 by the tail-weighted composite), native 5:1:2 network-ID split
(test = last 1/4 of networks per density). Architecture + DDIM knobs (100 steps,
`ddim_eta=0.2`, `model_cond_channels=2`) auto-load from the bundled
`.hydra/config.yaml`. Point metrics reproduce essentially exactly; the
channel-simulated rate metrics reproduce up to the seed-pinned DDIM sampler draw +
`num_channel_realizations=500`. Reference: the run's `test_summary.json`. Headline
targets:

| metric | value |
|---|---|
| `loss` | 0.0546 |
| `sum_rate_generated` | 1122.15 |
| `mean_rate_generated` | 2.805 |
| `min_rate_generated` | 0.304 |
| `rate_1pct_generated` | 0.510 |
| `rate_5pct_generated` | 0.743 |
| `fairness_generated` | 0.6603 |

**Retrain → `04_train_diffusion.sh` (optional, expensive).** `seed=0` is pinned,
but AMP + cuDNN are not bitwise-deterministic, so a fresh 5000-epoch run (days on
one GPU) yields a *statistically equivalent* model — **not** the identical
checkpoint or the exact table above. The recipe is recovered verbatim from the
run's `.hydra/overrides.yaml`. Use it to reproduce the *method*, not the *numbers*.

**Baselines/figures → `06_compare_baselines.sh`.** Compares the diffusion model
against the full-power (FP) and adaptive-power (AP) baselines on the test split and
runs the size/density transferability sweep, on the shipped checkpoint. Thin
wrapper over the canonical launchers under
`scripts/wra/diffusion/rigorous_quoll_131/` (which carry the full option notes and
checkpoint leaderboard). `PANELS=true` also renders the network-anatomy figures.

> ⚠️ **Checkpoint choice.** `best_model_epoch_1600.pt` is the rank-1 operating point
> (best worst-user rate / fairness) — the one `test_summary.json` was written from.
> Later epochs trade tail rate for sum-rate; the periodic snapshots (ep 4000/5000)
> beat the expert sum-rate but degrade rate@1%. Bundle ships ep1600 only.

---

## §4 — Artifacts & distribution

| Artifact | Size | In git as | Get it with |
|---|---|---|---|
| Rank-1 checkpoint `checkpoint/best_model_epoch_1600.pt` | 8 MB | **regular git blob** | plain clone |
| Run config `config/.hydra/{config,overrides}.yaml` | ~10 KB | regular git | plain clone |
| Expert sub-datasets (4 × ~9.8 GB) | ~39 GB | **public GitHub Release** in `gsd-dataset` (split tar parts, not LFS) | `download_dataset.sh` — curl, no auth (§2) |

The checkpoint is a **plain git blob, not LFS**: the root `.gitignore` excludes
`*.pt`, and a scoped [`checkpoint/.gitignore`](checkpoint/.gitignore) re-includes
just this file. Verify after cloning:

```bash
sha256sum -c reproduce/wra-rigorous-quoll/checksums/checkpoint.sha256
```

## Files

```
00_config.sh                 shared paths / env / experiment identity / DATASET_ROOT / CHECKPOINT
01_analyze_channels.sh       Stage 1  channel & scenario generation   (regeneration)
02_train_pd.sh               Stage 2  primal-dual expert training      (regeneration, expensive)
03_build_dataset.sh          Stage 3  collect PD samples -> datasets   (regeneration)
04_train_diffusion.sh        Stage 4  diffusion training seed=0        (optional/expensive)
05_evaluate_checkpoint.sh    Stage 5  eval shipped ckpt                (exact numbers)
06_compare_baselines.sh      Stage 6  FP/AP vs diffusion + figures     (paper comparison)
verify_dataset_manifest.sh   §2 structural integrity check (read-only)
download_dataset.sh          fetch the dataset from the public Release via curl (consumer, no auth)
package_dataset.sh           split + upload the dataset to the shared gsd-dataset Release (owner only)
dataset_bundle_id.sh         compute/verify the release-tag content id (see DATASET_TAG.md)
checkpoint/best_model_epoch_1600.pt   rank-1 checkpoint (8 MB, regular git)
checkpoint/.gitignore                 re-includes the .pt past the root *.pt rule
config/.hydra/config.yaml             resolved run config (for cli.test --config-dir)
config/.hydra/overrides.yaml          the verbatim training overrides
checksums/checkpoint.sha256           checkpoint integrity manifest
checksums/dataset_manifest.txt        referenced-dataset structural manifest
DATASET_TAG.md                        how the release-tag content id (…-85faf506ec70) is derived
```

Stage 4's Hydra recipe is the verbatim `.hydra/overrides.yaml` (+ explicit
`seed=0`). Stages 1–3 mirror the proven `sophisticated-oarfish-9` launchers.
