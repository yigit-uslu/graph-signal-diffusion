#!/usr/bin/env python3
"""Training diagnostics for ds8 / all-density (single-r_min) WRA diffusion arms.

Analogous to scripts/analyze_oarfish9_diagnostics.py, but adapted for the
"merged U-GNN" family (sophisticated-oarfish-9 task x sociable-frigatebird-619
architecture):
  - resolves a run by NAME under outputs/wireless_resource_allocation-wra/*/<run>
    (so it is agnostic to the model subdir: ds8_norm_act_head, ds4, ...),
  - works for the single-r_min all-density datasets (4 sub-datasets, one per
    density) as well as multi-r_min ones,
  - uses the VAL split (the best-model tracker's objective split),
  - emits a text health report + a CSV + three evolution PDFs:
      <run>/per_density_gap_evolution.pdf       (the headline: gaps vs epoch)
      <run>/per_density_rate_percentiles.pdf     (generated vs expert tail rates)
      <run>/training_selector_health.pdf         (loss/grad/lr + selector telemetry)
      <run>/per_density_metrics.csv

Usage:
    python scripts/analyze_ds8_alldensity_diagnostics.py [RUN_NAME ...]
    # default: rigorous-quoll-131
"""

import csv
import glob
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DEFAULT_RUNS = ["rigorous-quoll-131"]
SEARCH_GLOB = "outputs/wireless_resource_allocation-wra/*/{run}/epoch_summaries.jsonl"

# Tracker composite weights (from trainer best_model config).
COMPOSITE_WEIGHTS = {
    "rate_mean_violation_gap_pct_generated": 0.5,
    "rate_1pct_gap_pct": 0.25,
    "rate_5pct_gap_pct": 0.25,
}

# hash -> (density, r_min) for the medium-large outdoor all-density datasets.
HASH_TO_LABEL = {
    "h51d52d48c355": ("ultra-low", 0.4), "h54416c2fba47": ("ultra-low", 0.5),
    "h0dd7afd393f9": ("ultra-low", 0.6), "h56b70909ae82": ("ultra-low", 0.7),
    "he1da74a5635c": ("ultra-low", 0.8),
    "hd8d168fe5810": ("low", 0.4), "h26dde690eca5": ("low", 0.5),
    "h43d4a26a4203": ("low", 0.6), "h0f9e518a51ae": ("low", 0.7),
    "h391ad7cf511b": ("low", 0.8),
    "haa4d4fdc8221": ("mid", 0.4), "h3fddacd0eadc": ("mid", 0.5),
    "hc1f8f7a25432": ("mid", 0.6), "h4ab03934d654": ("mid", 0.7),
    "h943a82aae11f": ("mid", 0.8),
    "h9c4284c3b7ea": ("high", 0.4), "h5b791d7f2908": ("high", 0.5),
    "ha6c7c432ee13": ("high", 0.6), "h26614d2f6640": ("high", 0.7),
    "h9681f9e137e7": ("high", 0.8),
}
DENSITY_ORDER = ["ultra-low", "low", "mid", "high"]
DENSITY_COLORS = {"ultra-low": "tab:cyan", "low": "tab:blue",
                  "mid": "tab:green", "high": "tab:red"}


# ───────────────────────── loading / helpers ──────────────────────────
def find_run_dir(run):
    hits = glob.glob(SEARCH_GLOB.format(run=run))
    if not hits:
        return None
    # Prefer the most-recently-modified summaries if a name collides.
    hits.sort(key=os.path.getmtime)
    return os.path.dirname(hits[-1])


def load_records(jsonl_path):
    recs = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # tolerate a partially-written final line on a live run
    return recs


def detect_split(recs):
    """Prefer 'val_' (tracker split); fall back to 'train-val_'."""
    for pref in ("val_", "train-val_"):
        if any(k.startswith(pref + "subdataset/") for r in recs for k in r):
            return pref
    return "val_"


def finite(x):
    return isinstance(x, (int, float)) and not (math.isnan(x) or math.isinf(x))


def ser(recs, key):
    return ([r["epoch"] for r in recs if key in r and finite(r[key])],
            [r[key] for r in recs if key in r and finite(r[key])])


def detect_present(recs, split):
    """Return {hash: (density, r_min, subdataset_name)} actually in the data."""
    present = {}
    prefix = split + "subdataset/"
    for r in recs:
        keys = [k for k in r if k.startswith(prefix)]
        if not keys:
            continue
        for k in keys:
            name = k[len(prefix):].split("/")[0]
            h = name.rsplit("_", 1)[-1]
            if h in HASH_TO_LABEL:
                d, rm = HASH_TO_LABEL[h]
                present[h] = (d, rm, name)
        break  # one eval record carries the full sub-dataset set
    return present


def subkey(split, name, metric):
    return f"{split}subdataset/{name}/{metric}"


# ───────────────────────── text health report ─────────────────────────
def text_report(recs, split, run):
    last = recs[-1]
    ep = last["epoch"]
    evals = [r for r in recs if any("gap_pct" in k for k in r)]
    print(f"\n{'='*80}\n{run}  —  epoch {ep}  ({100*ep/5000:.0f}% of 5000)  |  "
          f"{len(evals)} eval ckpts, latest @ {evals[-1]['epoch']}  |  split='{split}'\n{'='*80}")

    def line(key, direction="min", p=4, label=None):
        eps, vals = ser(recs, key)
        label = label or key
        if not vals:
            print(f"  {label:46s}  (absent)")
            return
        bi = (min if direction == "min" else max)(range(len(vals)), key=lambda i: vals[i])
        tail = " ".join(f"{v:.2f}" for v in vals[-5:])
        print(f"  {label:46s} first={vals[0]:.{p}f}  latest@{eps[-1]}={vals[-1]:.{p}f}  "
              f"best={vals[bi]:.{p}f}@{eps[bi]}  last5=[{tail}]")

    print("\n[ TRAINING HEALTH ]")
    line("train_loss"); line("train_grad_norm"); line("train_lr", p=7)
    sk_e, sk_v = ser(recs, "num_skipped_steps")
    su_e, su_v = ser(recs, "num_successful_steps")
    if sk_v:
        print(f"  optimizer steps: skipped(sum)={sum(sk_v):.0f}  "
              f"successful(sum)={sum(su_v):.0f}  latest_epoch_skipped={sk_v[-1]:.0f}")

    print("\n[ VAL COMPOSITE (tracker objective; lower=better) ]")
    line("best_model_composite_score"); line("best_model_raw_composite_score")

    print("\n[ VAL COMPONENT GAPS % (lower=better) ]")
    for m in ["rate_mean_violation_gap_pct_generated", "rate_1pct_gap_pct",
              "rate_5pct_gap_pct", "min_rate_gap_pct", "sum_rate_gap_pct"]:
        line(split + m, p=2)

    print("\n[ FEASIBILITY / VIOLATIONS (val) ]")
    line(split + "feasibility_rate", "max")
    line(split + "rate_violation_percentage_generated", p=2)

    print("\n[ GENERALIZATION: val vs train-val ]")
    for base in ["rate_mean_violation_gap_pct_generated", "rate_1pct_gap_pct", "feasibility_rate"]:
        _, v = ser(recs, "val_" + base)
        _, tv = ser(recs, "train-val_" + base)
        if v and tv:
            print(f"  {base:44s} val={v[-1]:.2f}  train-val={tv[-1]:.2f}")
    line("val_loss_gap")

    print("\n[ SELECTOR HEALTH (Gumbel-exploration features) ]")
    line("selector_temperature"); line("selector_exploration_noise")
    line("selector_entropy_mean_norm")
    for lv in range(3):
        line(f"selector_level{lv}_selected_ratio_mean", p=3)


# ───────────────────────── series for plots / CSV ─────────────────────
def build_series(recs, split, present):
    """series[(density, r_min)][metric] -> (epochs, values); '+ composite'."""
    metrics = ["rate_mean_violation_gap_pct_generated", "rate_1pct_gap_pct",
               "rate_5pct_gap_pct", "min_rate_gap_pct", "feasibility_rate",
               "min_rate_generated", "min_rate_real",
               "rate_0.1pct_generated", "rate_0.1pct_real",
               "rate_1pct_generated", "rate_1pct_real",
               "rate_5pct_generated", "rate_5pct_real"]
    out = {}
    for h, (d, rm, name) in present.items():
        s = {m: ([], []) for m in metrics}
        comp = ([], [])
        for r in recs:
            for m in metrics:
                v = r.get(subkey(split, name, m))
                if finite(v):
                    s[m][0].append(r["epoch"]); s[m][1].append(v)
            # per-density composite (tracker weighting)
            parts = {w: r.get(subkey(split, name, w)) for w in COMPOSITE_WEIGHTS}
            if all(finite(v) for v in parts.values()):
                comp[0].append(r["epoch"])
                comp[1].append(sum(COMPOSITE_WEIGHTS[w] * parts[w] for w in parts))
        s["composite"] = comp
        out[(d, rm)] = s
    return out


def best_composite_epoch(recs):
    eps, vals = ser(recs, "best_model_raw_composite_score")
    if not vals:
        eps, vals = ser(recs, "best_model_composite_score")
    if not vals:
        return None
    return eps[min(range(len(vals)), key=lambda i: vals[i])]


# ───────────────────────── plots ──────────────────────────────────────
def _mark_peak(ax, peak_ep):
    if peak_ep is not None:
        ax.axvline(peak_ep, color="0.4", ls=":", lw=1.2)


def plot_gap_evolution(recs, split, present, series, out_path, run):
    peak = best_composite_epoch(recs)
    panels = [("Composite (0.5·mean+0.25·1p+0.25·5p)", "composite"),
              ("Mean violation gap %", "rate_mean_violation_gap_pct_generated"),
              ("1-pct rate gap %", "rate_1pct_gap_pct"),
              ("5-pct rate gap %", "rate_5pct_gap_pct"),
              ("Min-rate gap %", "min_rate_gap_pct"),
              ("Feasibility rate", "feasibility_rate")]
    fig, axes = plt.subplots(3, 2, figsize=(15, 13), sharex=True)
    fig.suptitle(f"Per-density metric evolution — {run}  "
                 f"(dotted line = best-composite epoch {peak})", fontsize=13, y=0.995)
    for idx, (title, metric) in enumerate(panels):
        ax = axes[idx // 2, idx % 2]
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3)
        for (d, rm), s in sorted(series.items(), key=lambda kv: DENSITY_ORDER.index(kv[0][0])):
            ep, v = s[metric]
            if ep:
                ax.plot(ep, v, color=DENSITY_COLORS[d], lw=1.6, alpha=0.9,
                        label=f"{d} (r={rm})")
        # aggregate val_ curve (black dashed) where a top-level metric exists
        agg_key = "best_model_raw_composite_score" if metric == "composite" else split + metric
        aep, av = ser(recs, agg_key)
        if aep:
            ax.plot(aep, av, color="black", ls="--", lw=1.3, alpha=0.7, label="aggregate")
        _mark_peak(ax, peak)
    for c in range(2):
        axes[-1, c].set_xlabel("Epoch")
    axes[0, 0].legend(fontsize=8, loc="best")
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_rate_percentiles(recs, split, present, series, out_path, run):
    peak = best_composite_epoch(recs)
    panels = [("Min rate", "min_rate"), ("0.1-pct rate", "rate_0.1pct"),
              ("1-pct rate", "rate_1pct"), ("5-pct rate", "rate_5pct")]
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    fig.suptitle(f"Per-density tail rates: generated (solid) vs expert (faded) — {run}",
                 fontsize=13, y=0.995)
    for idx, (title, base) in enumerate(panels):
        ax = axes[idx // 2, idx % 2]
        ax.set_title(f"{title} (bps/Hz)", fontsize=11)
        ax.grid(True, alpha=0.3)
        gkey = base + "_generated" if base.startswith("rate") else "min_rate_generated"
        rkey = base + "_real" if base.startswith("rate") else "min_rate_real"
        for (d, rm), s in sorted(series.items(), key=lambda kv: DENSITY_ORDER.index(kv[0][0])):
            ge, gv = s.get(gkey, ([], []))
            re_, rv = s.get(rkey, ([], []))
            if ge:
                ax.plot(ge, gv, color=DENSITY_COLORS[d], lw=1.6, alpha=0.9, label=d)
            if re_:
                ax.plot(re_, rv, color=DENSITY_COLORS[d], lw=1.1, alpha=0.3)
        _mark_peak(ax, peak)
    for c in range(2):
        axes[-1, c].set_xlabel("Epoch")
    axes[0, 0].legend(fontsize=8, loc="best")
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_training_selector_health(recs, out_path, run):
    peak = best_composite_epoch(recs)
    fig, axes = plt.subplots(3, 2, figsize=(15, 13), sharex=True)
    fig.suptitle(f"Training & selector health — {run}", fontsize=13, y=0.995)

    def draw(ax, items, title, ylab, logy=False):
        ax.set_title(title, fontsize=11); ax.set_ylabel(ylab); ax.grid(True, alpha=0.3)
        for key, lab, kw in items:
            ep, v = ser(recs, key)
            if ep:
                ax.plot(ep, v, label=lab, **kw)
        if logy:
            ax.set_yscale("log")
        if len(items) > 1:
            ax.legend(fontsize=8)
        _mark_peak(ax, peak)

    draw(axes[0, 0], [("train_loss", "train", dict(color="tab:blue", lw=1.4)),
                      ("val_loss", "val", dict(color="tab:orange", lw=1.2, alpha=0.8))],
         "Loss (eps / L2)", "loss")
    draw(axes[0, 1], [("train_grad_norm", "grad_norm", dict(color="tab:red", lw=1.2))],
         "Grad norm (clip=1.0)", "norm")
    draw(axes[1, 0], [("train_lr", "lr", dict(color="tab:green", lw=1.4))],
         "Learning rate", "lr", logy=True)
    draw(axes[1, 1], [("selector_temperature", "temperature", dict(color="tab:purple", lw=1.4)),
                      ("selector_exploration_noise", "exploration", dict(color="tab:brown", lw=1.4))],
         "Selector temperature & exploration (annealing)", "value")
    draw(axes[2, 0], [("selector_entropy_mean_norm", "entropy_norm", dict(color="tab:gray", lw=1.4))],
         "Selector normalized entropy", "entropy")
    draw(axes[2, 1], [(f"selector_level{lv}_selected_ratio_mean", f"level{lv}",
                       dict(lw=1.4)) for lv in range(3)],
         "Selected ratio per level (target ~0.5 for gamma=2)", "ratio")
    axes[2, 1].axhline(0.5, color="0.5", ls=":", lw=1)
    for c in range(2):
        axes[-1, c].set_xlabel("Epoch")
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def write_csv(series, recs, out_path):
    epochs = sorted({e for s in series.values()
                     for e in s["composite"][0]})
    cols = ["mean_gap", "1pct_gap", "5pct_gap", "min_gap", "composite", "feasibility"]
    cmap = {"mean_gap": "rate_mean_violation_gap_pct_generated",
            "1pct_gap": "rate_1pct_gap_pct", "5pct_gap": "rate_5pct_gap_pct",
            "min_gap": "min_rate_gap_pct", "feasibility": "feasibility_rate"}
    # epoch-indexed lookup per (density,metric)
    idx = {}
    for (d, rm), s in series.items():
        for col, metric in list(cmap.items()) + [("composite", "composite")]:
            ep, v = s[metric] if col == "composite" else s.get(cmap.get(col, ""), ([], []))
            idx[(d, col)] = dict(zip(ep, v))
    header = ["epoch"]
    for d in DENSITY_ORDER:
        if any(k[0] == d for k in {(dd, rr) for (dd, rr) in series}):
            header += [f"{d}_{c}" for c in cols]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header)
        for e in epochs:
            row = [e]
            for d in DENSITY_ORDER:
                if not any(k[0] == d for k in series):
                    continue
                for c in cols:
                    val = idx.get((d, c), {}).get(e, float("nan"))
                    row.append(round(val, 3) if finite(val) else "")
            w.writerow(row)
    print(f"  wrote {out_path} ({len(epochs)} eval epochs)")


# ───────────────────────── driver ─────────────────────────────────────
def run_for(run):
    rd = find_run_dir(run)
    if rd is None:
        print(f"ERROR: could not find epoch_summaries.jsonl for run '{run}'")
        return
    recs = load_records(os.path.join(rd, "epoch_summaries.jsonl"))
    if not recs:
        print(f"ERROR: no records for '{run}'")
        return
    split = detect_split(recs)
    present = detect_present(recs, split)
    text_report(recs, split, run)
    if not present:
        print("  (no per-subdataset metrics found; skipping plots/CSV)")
        return
    series = build_series(recs, split, present)
    print(f"\n[ writing artifacts to {rd} ]")
    plot_gap_evolution(recs, split, present, series,
                       os.path.join(rd, "per_density_gap_evolution.pdf"), run)
    plot_rate_percentiles(recs, split, present, series,
                          os.path.join(rd, "per_density_rate_percentiles.pdf"), run)
    plot_training_selector_health(recs, os.path.join(rd, "training_selector_health.pdf"), run)
    write_csv(series, recs, os.path.join(rd, "per_density_metrics.csv"))


def main():
    runs = sys.argv[1:] or DEFAULT_RUNS
    for run in runs:
        run_for(run)


if __name__ == "__main__":
    main()
