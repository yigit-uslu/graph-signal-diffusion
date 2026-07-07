# Reproducing `sociable-frigatebird-619` (S&P 500)

End-to-end pipeline to reproduce the S&P 500 diffusion-forecasting run
`sociable-frigatebird-619` — the **small ~1.88M-parameter DS8 model** (the
long-training, `lr=1e-4`, 5000-epoch extension of `spotted-catfish-602`) on the
native **10-chunk** split. All commands are recovered verbatim from the run's
saved `.hydra/overrides.yaml` and cross-checked against the surviving output dir.

Every script sources [`00_config.sh`](00_config.sh) and is location-independent
(it resolves the repo root from its own path). Run them from anywhere; each
`cd`s to the project root. Python always goes through the pinned conda env
(`CONDA_ENV`, default `torch_env`).

---

## Pipeline at a glance

| # | Script | Stage | Reproducible? |
|---|--------|-------|---------------|
| 0 | `download_dataset.sh` | fetch frozen raw (public Release, curl) | ✅ verified vs `checksums/raw.sha256` |
| 1 | `01_download_raw.sh` | yfinance raw pull | ❌ live source — **provenance only** (§2) |
| 2 | `02_update_stocks.sh` | Wikipedia sectors | ❌ live source — optional, provenance only |
| 4 | `03_clean.sh` | clean → cleaned data root | ✅ **byte-identical** (verified, §3) |
| 6 | `04_train.sh` | train 5000 epochs (seed=0) | ~ statistical only — expensive (§4) |
| 7 | `05_evaluate_checkpoint.sh` | eval shipped checkpoint → `test_summary.json` | ✅ **exact numbers** (§4) |
| 8 | `06_compare_baselines.sh` | GRW vs. diffusion on test (paper figs) | ✅ checkpoint-based comparison (§4) |
| — | `verify_data_equivalence.sh` | move-aside / re-clean / diff / restore | the §3 proof, re-runnable |

(Stages 3 and 5 in the source tree — `analyze_dates`, graph sweeps — are
diagnostic/exploratory and are not part of this reproduction.)

## Quick start (the reproducible path)

```bash
cd <repo-root>
REPRO=reproduce/sp500-sociable-frigatebird-619

# 0. Get the frozen raw — fetch from the public Release (curl, no auth). It lands at
#    data/sp500/raw and is verified against the paper's exact bytes on the way in.
$REPRO/download_dataset.sh
#    (already have it via git-LFS? skip the download and just verify:)
#    sha256sum -c $REPRO/checksums/raw.sha256

# 1. Build the cleaned data root from the frozen raw (deterministic, §3).
$REPRO/03_clean.sh
sha256sum -c $REPRO/checksums/cleaned_raw.sha256      # expect all OK

# 2. Reproduce the paper's test metrics from the shipped checkpoint (§4).
$REPRO/05_evaluate_checkpoint.sh
#    -> compare against outputs/.../sociable-frigatebird-619/test_summary.json

# 3. (optional) Reproduce the GRW-vs-diffusion comparison behind the paper figures.
$REPRO/06_compare_baselines.sh
```

Prerequisites: the `torch_env` conda env and one CUDA GPU. **Everything the pipeline
needs travels with the repo or a public Release** (see §5): the frozen
`data/sp500/raw/` via [`download_dataset.sh`](download_dataset.sh) (anonymous curl,
no auth — the LFS-free path) or `git lfs pull` as a fallback, and the ep4500
checkpoint bundled here in `checkpoint/` via regular git. The large cleaned data
root is *not* stored — `03_clean.sh` regenerates it deterministically (§3).

---

## §2 — Raw data is frozen (why download isn't the entry point)

`01_download_raw.sh` pulls from **yfinance, a live source**. A fresh pull will
**not** byte-reproduce `data/sp500/raw/`: adjusted-close prices are retroactively
restated on splits/dividends, index membership drifts, and tickers delist over
time. (The repo already carries evidence of this — a superseded `raw-v001/`
alongside `raw/`.)

So the reproducible pipeline **starts from the frozen `raw/`**.
`checksums/raw.sha256` pins the exact bytes the paper used
(`values.csv`, `stocks.csv`, `fundamentals.csv`, `adj.npy`). Fetch that frozen raw
with [`download_dataset.sh`](download_dataset.sh) (public Release, anonymous curl)
or `git lfs pull` — both land it at `data/sp500/raw` and verify against the manifest
(§5). `01_download_raw.sh` is kept only to document — and audit — how that frozen
raw was originally produced (it refuses to overwrite an existing `raw/`).

## §3 — The clean stage is byte-reproducible (verified)

`03_clean.sh` is a **deterministic** transform of the frozen raw. This was
verified by moving the paper's cleaned root aside, re-running clean, and diffing:

```
values.csv         OK
adj.npy            OK
stocks.csv         OK
fundamentals.csv   OK
graph_metadata.json OK
metadata.json      identical except its wall-clock `timestamp` field
```

Every **data** file is byte-identical; the only difference is the `datetime.now()`
stamp written into `metadata.json`. `node_selection_matrices/` and the sibling
`processed_<hash>/` tensor caches are **downstream artifacts** (the dataset loader
builds the caches lazily at train/eval time), not `clean.py` output, and are
outside the equivalence claim. Re-run the proof any time:

```bash
reproduce/sp500-sociable-frigatebird-619/verify_data_equivalence.sh   # PASS
```

It is safe: the original is backed up before clean writes, and the pristine
original (caches, `node_selection_matrices/`, timestamps) is restored on exit.

## §4 — Results: exact (checkpoint) vs. statistical (retrain)

**Exact numbers → `05_evaluate_checkpoint.sh` (recommended).** It runs the same
held-out test evaluation the trainer ran, on the shipped
`best_model_epoch_4500.pt`, native 10-chunk test split, ensemble size **10**,
100 DDIM steps at `eta=0.2` (auto-loaded from the checkpoint). Point metrics
(MAE/RMSE/MSE, `price_*`) reproduce essentially exactly; probabilistic metrics
(`return_crps`, `mis_*`, `coverage_*`) reproduce up to the seed-pinned sampler
draw. Reference: the run's `test_summary.json`. Headline targets:

| metric | value |
|---|---|
| `loss` | 0.3204 |
| `return_mae` / `return_rmse` | 1.4125 / 2.0538 |
| `return_crps` | 1.0790 |
| `return_mis_90` / `return_coverage_90` | 11.385 / 0.6908 |
| `direction_accuracy` | 0.5024 |
| `cum_T_direction_accuracy` | 0.5161 |
| `kurtosis_gen` (gap) | 10.569 (9.12) |
| `price_mae` / `price_rmse` | 3.744 / 10.716 |
| `eigval1_ratio` | 0.6431 |

**Retrain → `04_train.sh` (optional, expensive).** `seed=0` is pinned, but AMP +
cuDNN are not bitwise-deterministic, so a fresh 5000-epoch run (~days on one GPU)
yields a *statistically equivalent* model — **not** the identical checkpoint, and
not the exact table above. Use it to reproduce the *method*, not the *numbers*.

**Baseline comparison → `06_compare_baselines.sh`.** The paper's figures/tables
come from comparing the diffusion model against the GRW baseline on the *same*
10-chunk test windows. `06` runs that comparison on the shipped checkpoint
(ensemble 100, full test) and writes to `outputs/…/comparison/<date>/…`. It is a
thin, config-wired mirror of the canonical
`scripts/stock/sp500/sociable-frigatebird-619/compare_baselines.sh` (which also
documents alternate checkpoints and the RevIN-α sensitivity probe).

> ⚠️ **Split discipline.** This checkpoint was trained with `n_split_chunks=10`;
> that is the **only** valid eval split. Re-evaluating at a different chunk count
> leaks ~80% of the train windows into "test." Both `04`/`05` hard-code 10.

---

## §5 — Artifacts & distribution (what travels, and how)

| Artifact | Size | Distributed as | Get it with |
|---|---|---|---|
| Frozen raw `data/sp500/raw/` | ~0.3GB (≈0.1GB gz) | **public GitHub Release** in `gsd-dataset` (gzip tar parts, not LFS) — also still git-LFS today | `./download_dataset.sh` (curl, no auth) — or `git lfs pull` (fallback) |
| ep4500 checkpoint `checkpoint/best_model_epoch_4500.pt` | 19MB | **regular git blob** | plain clone |
| Cleaned data root | ~5GB | not stored | `./03_clean.sh` (deterministic) |

The checkpoint is deliberately committed as a **plain git blob, not LFS**: the
root `.gitignore` excludes `*.pt`, and a scoped
[`checkpoint/.gitignore`](checkpoint/.gitignore) re-includes just this file. At
19MB and frozen it is a one-time cost, and keeping it out of LFS is consistent
with retiring the LFS dependency later. Verify it after cloning:

```bash
sha256sum -c reproduce/sp500-sociable-frigatebird-619/checksums/checkpoint.sha256
```

The frozen raw's **LFS-free path is now live**: it is hosted on a public GitHub
Release in the shared `gsd-dataset` repo and fetched by
[`download_dataset.sh`](download_dataset.sh) with anonymous `curl` (no account or
token). The owner (re)publishes it with
[`package_dataset.sh`](package_dataset.sh), which verifies the on-disk raw against
`checksums/raw.sha256`, gzip-tars it, uploads to the tag pinned in `00_config.sh`
(`sp500-gsd-2026-02-13`, encoding the yfinance vintage), and records per-part
checksums in `checksums/dataset_archives.sha256`. The git-LFS pointer is **retained
as a fallback for now**; actually stripping it from history is deferred to the later
public-release pass.

## Files

```
00_config.sh                 shared paths / env / experiment identity
01_download_raw.sh           Stage 1  yfinance pull        (provenance only)
02_update_stocks.sh          Stage 2  Wikipedia sectors    (optional)
03_clean.sh                  Stage 4  clean -> data root   (deterministic)
04_train.sh                  Stage 6  train 5000ep seed=0  (optional/expensive)
05_evaluate_checkpoint.sh    Stage 7  eval shipped ckpt    (exact numbers)
06_compare_baselines.sh      Stage 8  GRW vs diffusion     (paper comparison)
verify_data_equivalence.sh   §3 proof harness (safe, reversible)
download_dataset.sh          fetch frozen raw from the public Release (curl, no auth)
package_dataset.sh           publish frozen raw to the Release (owner; needs gh)
checkpoint/best_model_epoch_4500.pt   the ep4500 checkpoint (19MB, regular git)
checkpoint/.gitignore                 re-includes the .pt past the root *.pt rule
checksums/raw.sha256         frozen raw input manifest
checksums/cleaned_raw.sha256 cleaned data-root manifest
checksums/checkpoint.sha256  checkpoint integrity manifest
checksums/dataset_archives.sha256     per-part checksums of the released tar parts
```

All three Hydra invocations (`04`, `05`, `06`) were dry-validated with `--cfg job`
(config composes, all overrides legal). `06` mirrors the canonical launcher
`scripts/stock/sp500/sociable-frigatebird-619/compare_baselines.sh`, which carries
the extended notes (alternate checkpoints, RevIN-α sensitivity probe).
