# `assets/` — figures for the top-level README

Presentation figures for the project's public-facing `README.md`, which covers
both applications of the graph-signal generative-diffusion framework:

| Subdir | Experiment | Reproduction pipeline |
|---|---|---|
| `sp500-sociable-frigatebird-619/` | S&P 500 forecasting | `reproduce/sp500-sociable-frigatebird-619/` |
| `wra-rigorous-quoll/` | Wireless resource allocation | `reproduce/wra-rigorous-quoll/` (other machine) |

## Versioning policy — REGULAR git, not LFS

Everything under `assets/` is a **plain git blob**, never git-LFS. This is a
deliberate exception enforced by [`assets/.gitattributes`](.gitattributes), which
overrides the repo-root `*.pdf filter=lfs …` rule for this subtree — the same
spirit as the ep4500 checkpoint bundled in `reproduce/.../checkpoint/` (re-included
past the root `*.pt` ignore).

Why: these figures are small, are embedded inline in the README (**PNG renders on
GitHub; an LFS-tracked PDF would not**), and keeping them out of LFS is consistent
with retiring the LFS dependency. Verified: a `.png`/`.pdf` here resolves to
`filter: unset` (regular git), while the same extensions elsewhere still use LFS.

## Conventions

- **Format: PNG.** Embed PNG exports (they render inline and are regular-git). Keep
  print-quality PDFs in `outputs/…/comparison/…` as masters; don't commit them here.
- **Provenance.** These figures are *isolated from* a `06_compare_baselines.sh`
  run under `outputs/…/comparison/<date>/<time>_<experiment>_epoch-<N>_…/`. Record
  the exact source run per experiment (a short `SOURCE.md` in the subdir, or a note
  in the README caption) so a figure can always be regenerated.
- **Curate.** A handful of headline figures per experiment (e.g. the radar summary
  + the key distributional panels), not the full comparison dump.

## Adding a figure

```bash
# 1. copy the chosen PNG(s) out of the (gitignored) comparison output
cp outputs/.../comparison/<run>/.../comparison_radar.png \
   assets/sp500-sociable-frigatebird-619/

# 2. it is regular-git automatically (assets/.gitattributes) — verify:
git check-attr filter -- assets/sp500-sociable-frigatebird-619/comparison_radar.png
#   -> filter: unset   (NOT lfs)

# 3. embed in the top-level README:  ![](assets/sp500-sociable-frigatebird-619/comparison_radar.png)
```

The WRA machine drops its `rigorous-quoll` figures into
`assets/wra-rigorous-quoll/`; the shared `assets/.gitattributes` already makes them
regular-git — no per-machine setup needed.
