"""Regenerate compare-baselines plots from saved sidecar data.

Usage::

    python -m graph_signal_diffusion.cli.replot_baselines <run_dir>

where ``<run_dir>`` is a comparison output directory containing an
``eval_viz/`` subtree (e.g.
``outputs/.../comparison/2026-04-08/12-57-02``).

The script walks ``eval_viz/`` and rewrites every polished ``.pdf`` whose
matching ``*.npz`` sidecar exists.  Aesthetic edits live in
``structural_metrics.py`` (or the GRW baseline's private NLL plotters);
this CLI just calls the plot functions with the loaded data.

It does *not* re-run any evaluation, dataset loading, or model inference.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import Dict, List, Optional

import torch

from graph_signal_diffusion.evaluation import structural_metrics as sm
from graph_signal_diffusion.evaluation.plot_data_io import (
    load_comparison_nll,
    load_comparison_structural,
    load_per_baseline_nll,
    load_per_baseline_structural,
    load_stock_prices_window,
)


# ---------------------------------------------------------------------------
# Per-baseline plots
# ---------------------------------------------------------------------------

def _replot_per_baseline_structural(npz_path: str) -> List[str]:
    """Regenerate the 5 per-baseline structural pdfs from a sidecar."""
    data = load_per_baseline_structural(npz_path)
    out_dir = os.path.dirname(npz_path)
    model_name = data["model_name"]
    suffix = model_name.lower()  # "grw" / "diffusion"

    cat_real = torch.as_tensor(data["cat_real"])
    cat_gen = torch.as_tensor(data["cat_gen"])
    wsi = data["window_start_indices"]
    wsi_t = torch.as_tensor(wsi) if wsi is not None else None

    written: List[str] = []

    sm.plot_structural_metrics(
        cat_real, cat_gen,
        save_path=os.path.join(out_dir, f"structural_metrics_{suffix}.pdf"),
        max_lag=data["max_lag"],
        n_eigenvalues=data["n_eigenvalues"],
        dpi=data["dpi"],
        model_name=model_name,
        window_stride=data["window_stride"],
        window_start_indices=wsi_t,
    )
    written.append(f"structural_metrics_{suffix}.pdf")

    sm.plot_autocorr_per_stock(
        cat_real, cat_gen,
        save_path=os.path.join(out_dir, f"autocorr_per_stock_{suffix}.pdf"),
        max_lag=data["max_lag"],
        dpi=data["dpi"],
        model_name=model_name,
        window_stride=data["window_stride"],
        window_start_indices=wsi_t,
    )
    written.append(f"autocorr_per_stock_{suffix}.pdf")

    sm.plot_autocorr_violin(
        cat_real, cat_gen,
        save_path=os.path.join(out_dir, f"autocorr_violin_{suffix}.pdf"),
        max_lag=data["max_lag"],
        dpi=data["dpi"],
        model_name=model_name,
        window_stride=data["window_stride"],
        window_start_indices=wsi_t,
    )
    written.append(f"autocorr_violin_{suffix}.pdf")

    sm.plot_autocorr_heatmap(
        cat_gen,
        save_path=os.path.join(out_dir, f"autocorr_heatmap_{suffix}.pdf"),
        max_lag=data["max_lag"],
        dpi=data["dpi"],
        model_name=model_name,
        window_stride=data["window_stride"],
        window_start_indices=wsi_t,
        real_returns=cat_real,
    )
    written.append(f"autocorr_heatmap_{suffix}.pdf")

    if data["cum_std_real"] is not None and data["cum_std_gen"] is not None:
        sm.plot_cumulative_return_std(
            data["cum_std_real"], data["cum_std_gen"],
            save_path=os.path.join(out_dir, f"cumulative_return_std_{suffix}.pdf"),
            theoretical=data["theoretical"],
            std_real_tn=data["cum_std_real_tn"],
            std_gen_tn=data["cum_std_gen_tn"],
            dpi=data["dpi"],
            model_name=model_name,
            theoretical_label=data["theoretical_label"],
        )
        written.append(f"cumulative_return_std_{suffix}.pdf")

    return written


def _replot_per_baseline_nll(npz_path: str) -> List[str]:
    """Regenerate the 2 per-baseline NLL pdfs from a sidecar.

    The per-baseline NLL plotters live as private staticmethods on the
    GRW baseline class; we call them directly here.
    """
    from graph_signal_diffusion.baselines.stock_price_forecasting.grw import (
        GeometricRandomWalk,
    )

    data = load_per_baseline_nll(npz_path)
    out_dir = os.path.dirname(npz_path)
    suffix = (data.get("scoring_model_name") or "grw").lower()

    written: List[str] = []
    GeometricRandomWalk._plot_nll_histograms(
        data["nll_real"], data["nll_gen"],
        save_path=os.path.join(out_dir, f"nll_histogram_{suffix}.pdf"),
        bins=data["bins"],
        dpi=data["dpi"],
        log_x=data["log_x"],
    )
    written.append(f"nll_histogram_{suffix}.pdf")

    if data["cum_nll_real_bt"] is not None and data["cum_nll_gen_bt"] is not None:
        GeometricRandomWalk._plot_cumulative_nll_histograms(
            data["cum_nll_real_bt"], data["cum_nll_gen_bt"],
            save_path=os.path.join(out_dir, f"cumulative_nll_histogram_{suffix}.pdf"),
            bins=data["bins"],
            dpi=data["dpi"],
        )
        written.append(f"cumulative_nll_histogram_{suffix}.pdf")

    return written


# ---------------------------------------------------------------------------
# Comparison plots
# ---------------------------------------------------------------------------

def _replot_comparison_structural(npz_path: str) -> List[str]:
    data = load_comparison_structural(npz_path)
    out_dir = os.path.dirname(npz_path)

    cat_real = torch.as_tensor(data["cat_real"])
    gen_returns_dict = {
        name: torch.as_tensor(arr) for name, arr in data["cat_gen_dict"].items()
    }
    wsi = data["window_start_indices"]
    wsi_t = torch.as_tensor(wsi) if wsi is not None else None
    wsi_gen = {
        name: (torch.as_tensor(v) if v is not None else None)
        for name, v in data["window_start_indices_gen"].items()
    }

    sm.plot_structural_metrics_comparison(
        cat_real, gen_returns_dict,
        save_path=os.path.join(out_dir, "structural_metrics_comparison.pdf"),
        max_lag=data["max_lag"],
        n_eigenvalues=data["n_eigenvalues"],
        window_stride=data["window_stride"],
        window_start_indices=wsi_t,
        window_start_indices_gen=wsi_gen,
        baseline_colors=data["baseline_colors"],
    )

    sm.plot_cumulative_return_std_comparison(
        data["cum_std_real"], data["cum_std_gen_dict"],
        save_path=os.path.join(out_dir, "cumulative_return_std_comparison.pdf"),
        theoretical=data["theoretical"],
        std_real_per_stock=data["cum_std_real_per_stock"],
        std_gen_per_stock_dict=data["cum_std_gen_per_stock_dict"],
        baseline_colors=data["baseline_colors"],
    )
    return [
        "structural_metrics_comparison.pdf",
        "cumulative_return_std_comparison.pdf",
    ]


def _replot_comparison_nll(npz_path: str) -> List[str]:
    data = load_comparison_nll(npz_path)
    out_dir = os.path.dirname(npz_path)
    scoring = data["scoring_model_name"]

    written: List[str] = []
    sm.plot_nll_histograms_comparison(
        data["nll_real"], data["nll_gen_dict"],
        save_path=os.path.join(out_dir, f"nll_histogram_{scoring}_scored.pdf"),
        bins=data["bins"],
        log_x=data["log_x"],
        scoring_model_name=scoring,
        baseline_colors=data["baseline_colors"],
    )
    written.append(f"nll_histogram_{scoring}_scored.pdf")

    if data["cum_nll_real"] is not None and data["cum_nll_gen_dict"]:
        sm.plot_cumulative_nll_histograms_comparison(
            data["cum_nll_real"], data["cum_nll_gen_dict"],
            save_path=os.path.join(
                out_dir, f"cumulative_nll_histogram_{scoring}_scored.pdf",
            ),
            bins=data["bins"],
            log_x=data["log_x"],
            scoring_model_name=scoring,
            baseline_colors=data["baseline_colors"],
        )
        written.append(f"cumulative_nll_histogram_{scoring}_scored.pdf")
    return written


# ---------------------------------------------------------------------------
# Per-window stock_prices
# ---------------------------------------------------------------------------

def _replot_stock_prices_window(npz_path: str) -> List[str]:
    """Regenerate one stock_prices_t*_b*.pdf from its sidecar."""
    from graph_signal_diffusion.tasks.stock_price_forecasting.evaluator import (
        StockPriceForecastingTaskV2,
    )

    data = load_stock_prices_window(npz_path)
    # Sidecars live in <viz_save_dir>/stock_prices/, the parent dir is
    # the actual viz dir where the PDF should land.
    sidecar_dir = os.path.dirname(npz_path)
    viz_save_dir = os.path.dirname(sidecar_dir)

    pred_mean = torch.as_tensor(data["pred_mean"])
    target_mean = torch.as_tensor(data["target_mean"])
    gen_ensemble = (
        torch.as_tensor(data["gen_ensemble"])
        if data["gen_ensemble"] is not None else None
    )
    hist_samples = (
        torch.as_tensor(data["hist_samples"])
        if data["hist_samples"] is not None else None
    )

    StockPriceForecastingTaskV2._plot_stock_predictions_v2(
        pred_mean=pred_mean,
        target_mean=target_mean,
        gen_ensemble=gen_ensemble,
        hist_samples=hist_samples,
        stock_indices=data["stock_indices"],
        stock_symbols=data["stock_symbols"],
        batch_idx=data["batch_idx"],
        time_start=data["time_start"],
        viz_save_dir=viz_save_dir,
    )
    ts = data["time_start"]
    bi = data["batch_idx"]
    fname = (
        f"stock_prices_t{ts}_b{bi}.pdf" if ts is not None
        else f"stock_prices_b{bi}.pdf"
    )
    return [fname]


# ---------------------------------------------------------------------------
# Paper figures (Fig 2 / Fig 3) — regenerated from the comparison sidecars
# ---------------------------------------------------------------------------

def _load_paper_cfg(run_dir: str) -> "tuple[dict, list]":
    """Read ``paper_figures`` + ``baselines_to_compare`` from the snapshot.

    ``paper_figures`` carries the behavioral flags the live run used (which
    baseline, lags, ``normalize``, …) so the replot reproduces the same paper
    figures; ``baselines_to_compare`` fixes the Fig 3 scorer-row order to match
    the live ``baselines.items()`` iteration.  Returns ``({}, [])`` when the
    snapshot (or either node) is absent — i.e. paper figures are skipped.
    """
    cfg_path = os.path.join(run_dir, ".hydra", "config.yaml")
    if not os.path.exists(cfg_path):
        return {}, []
    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(cfg_path)

        def _sel(path, default):
            node = OmegaConf.select(cfg, path)
            if node is None:
                return default
            try:
                c = OmegaConf.to_container(node, resolve=True)
            except Exception:
                c = OmegaConf.to_container(node, resolve=False)
            return c if c is not None else default

        paper = _sel("paper_figures", {})
        order = _sel("baselines_to_compare", [])
        return (
            paper if isinstance(paper, dict) else {},
            list(order) if isinstance(order, (list, tuple)) else [],
        )
    except Exception:
        return {}, []


def _load_diffusion_num_timesteps(run_dir: str) -> Optional[int]:
    """Read ``diffusion.num_timesteps`` from the run snapshot.

    Used to express the Fig 3 diffusion-proxy row in per-element denoising-MSE
    units (proxy ÷ num_timesteps).  Returns ``None`` when unavailable (the
    rescale is then skipped and the row stays on its raw proxy scale).
    """
    cfg_path = os.path.join(run_dir, ".hydra", "config.yaml")
    if not os.path.exists(cfg_path):
        return None
    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(cfg_path)
        v = OmegaConf.select(cfg, "diffusion.num_timesteps")
        return int(v) if v is not None else None
    except Exception:
        return None


def _replot_paper_figures(
    eval_viz_dir: str,
    plot_style: Optional[dict],
    paper_cfg: dict,
    baseline_order: List[str],
    diffusion_num_timesteps: Optional[int] = None,
) -> List[str]:
    """Regenerate the SP500 paper figures (Fig 2, Fig 3) from comparison sidecars.

    Mirrors the live ``compare_baselines`` paper-figure calls: same behavioral
    flags (``paper_figures.fig2`` / ``fig3``) and the same self-styling
    ``plot_style``, so the from-sidecars output matches a live GPU run.  Writes
    to ``<eval_viz>/paper/``.  Skips entirely unless ``paper_figures.enabled``
    (only regenerates figures the run actually produced).

    Fig 3 is aggregated across *all* ``comparison/nll_data__*.npz`` scorer
    sidecars (rows = scorers), which is why this runs as a post-walk pass rather
    than per-file.  Fig 1 (per-window forecast columns) is regenerated from its
    own tiny per-panel sidecars (``paper/fig1_*.npz``) written by the live run.
    """
    if not (paper_cfg or {}).get("enabled", False):
        return []
    comp_dir = os.path.join(eval_viz_dir, "comparison")
    paper_dir = os.path.join(eval_viz_dir, "paper")
    written: List[str] = []
    # N·T per trajectory (from the [B, T, N] structural sidecar), used to express
    # the GRW row in per-(stock, step) nats in Fig 3.  Set in the Fig 2 block.
    n_elements: Optional[int] = None

    # ── Fig 1: per-window forecast columns (one tiny sidecar per panel) ──
    fig1_files = (
        sorted(
            f for f in os.listdir(paper_dir)
            if f.startswith("fig1_") and f.endswith(".npz")
        )
        if os.path.isdir(paper_dir) else []
    )
    if fig1_files:
        try:
            from graph_signal_diffusion.evaluation.plot_data_io import (
                load_paper_fig1_column,
            )
            from graph_signal_diffusion.tasks.stock_price_forecasting.evaluator import (
                StockPriceForecastingTaskV2,
            )
            for fn in fig1_files:
                try:
                    col = load_paper_fig1_column(os.path.join(paper_dir, fn))
                    stem = os.path.splitext(fn)[0]
                    StockPriceForecastingTaskV2._plot_paper_forecast_column(
                        col, os.path.join(paper_dir, stem + ".pdf"), plot_style,
                    )
                    written.append(f"paper/{stem}.pdf")
                except Exception:
                    print(
                        f"[replot] FAILED paper/{fn}\n" + traceback.format_exc()
                    )
        except Exception:
            print(
                "[replot] FAILED importing the Fig 1 plotter\n"
                + traceback.format_exc()
            )

    # ── Fig 2: structural comparison (combined ACF + eigenvalue spectrum) ──
    struct_npz = os.path.join(comp_dir, "structural_data.npz")
    if os.path.exists(struct_npz):
        try:
            f2 = dict(paper_cfg.get("fig2", {}) or {})
            d = load_comparison_structural(struct_npz)
            cat_real = torch.as_tensor(d["cat_real"])
            n_elements = int(cat_real.shape[1]) * int(cat_real.shape[2])  # N·T
            gen = {n: torch.as_tensor(a) for n, a in d["cat_gen_dict"].items()}
            wsi = d["window_start_indices"]
            wsi_t = torch.as_tensor(wsi) if wsi is not None else None
            wsi_gen = {
                n: (torch.as_tensor(v) if v is not None else None)
                for n, v in d["window_start_indices_gen"].items()
            }
            os.makedirs(paper_dir, exist_ok=True)
            sm.plot_structural_comparison_paper(
                cat_real, gen,
                save_path=os.path.join(paper_dir, "fig2_structural_comparison.pdf"),
                max_lag=(
                    f2["max_lag"] if f2.get("max_lag") is not None else d["max_lag"]
                ),
                n_eigenvalues=(
                    f2["n_eigenvalues"]
                    if f2.get("n_eigenvalues") is not None else d["n_eigenvalues"]
                ),
                window_stride=d["window_stride"],
                window_start_indices=wsi_t,
                window_start_indices_gen=wsi_gen,
                baseline_colors=d["baseline_colors"],
                normalize=bool(f2.get("normalize", True)),
                include_lag0=bool(f2.get("include_lag0", False)),
                show_iqr=bool(f2.get("show_iqr", True)),
                twin_axis=bool(f2.get("twin_axis", False)),
                plot_style=plot_style,
            )
            written.append("paper/fig2_structural_comparison.pdf")
        except Exception:
            print(
                "[replot] FAILED paper/fig2_structural_comparison.pdf\n"
                + traceback.format_exc()
            )

    # ── Fig 3: dual-scorer NLL (rows = scorers, cols = {hist, CDF}) ──
    scorer_files: Dict[str, str] = {}
    if os.path.isdir(comp_dir):
        for fn in os.listdir(comp_dir):
            if fn.startswith("nll_data__") and fn.endswith(".npz"):
                name = fn[len("nll_data__"):-len(".npz")]
                scorer_files[name] = os.path.join(comp_dir, fn)
    if scorer_files:
        try:
            f3 = dict(paper_cfg.get("fig3", {}) or {})

            # Order scorer rows to match the live baselines.items() iteration
            # (= baselines_to_compare order); unknown names sort last by name.
            def _rank(name: str) -> "tuple[int, str]":
                try:
                    return (baseline_order.index(name), name)
                except ValueError:
                    return (len(baseline_order), name)

            ordered = sorted(scorer_files, key=_rank)
            scorers: List[dict] = []
            side_log_x, side_bins, side_colors = True, 60, None
            for i, name in enumerate(ordered):
                nd = load_comparison_nll(scorer_files[name])
                scorers.append({
                    "name": nd["scoring_model_name"],
                    "nll_real": nd["nll_real"],
                    "nll_gen_dict": nd["nll_gen_dict"],
                })
                if i == 0:  # recover the live log_x/bins/colors from the sidecar
                    side_log_x = bool(nd["log_x"])
                    side_bins = int(nd["bins"])
                    side_colors = nd["baseline_colors"]
            # Express the GRW (summed-NLL) row in per-(stock, step) nats so it
            # reads against the i.i.d. Gaussian floor; mirrors the live run.
            if f3.get("grw_per_element_nats", True):
                scorers = [
                    sm.annotate_grw_per_element_nats(s, n_elements) for s in scorers
                ]
            # Express the diffusion-proxy row in per-element denoising-MSE units.
            if f3.get("diffusion_per_step_mse", True):
                scorers = [
                    sm.annotate_diffusion_per_step_mse(s, diffusion_num_timesteps)
                    for s in scorers
                ]
            if scorers:
                os.makedirs(paper_dir, exist_ok=True)
                f3_logx = f3.get("log_x")
                sm.plot_nll_comparison_paper(
                    scorers,
                    save_path=os.path.join(paper_dir, "fig3_nll_comparison.pdf"),
                    bins=int(f3.get("bins", side_bins)),
                    log_x=(side_log_x if f3_logx is None else bool(f3_logx)),
                    baseline_colors=side_colors,
                    plot_style=plot_style,
                )
                written.append("paper/fig3_nll_comparison.pdf")
        except Exception:
            print(
                "[replot] FAILED paper/fig3_nll_comparison.pdf\n"
                + traceback.format_exc()
            )

    return written


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _walk_eval_viz(eval_viz_dir: str) -> int:
    """Find every sidecar under eval_viz/ and regenerate its plot.

    Returns the total number of pdfs (re)written.
    """
    n_written = 0
    for dirpath, dirnames, filenames in os.walk(eval_viz_dir):
        for fname in filenames:
            full = os.path.join(dirpath, fname)
            try:
                if fname == "structural_data.npz":
                    if os.path.basename(dirpath) == "comparison":
                        written = _replot_comparison_structural(full)
                    else:
                        written = _replot_per_baseline_structural(full)
                elif fname == "nll_data.npz":
                    written = _replot_per_baseline_nll(full)
                elif fname.startswith("nll_data__") and fname.endswith(".npz"):
                    written = _replot_comparison_nll(full)
                elif (
                    os.path.basename(dirpath) == "stock_prices"
                    and fname.endswith(".npz")
                ):
                    written = _replot_stock_prices_window(full)
                else:
                    continue
            except Exception:
                print(
                    f"[replot] FAILED {os.path.relpath(full, eval_viz_dir)}\n"
                    + traceback.format_exc()
                )
                continue
            for w in written:
                print(f"  ↺ {os.path.relpath(os.path.join(dirpath, w), eval_viz_dir)}")
            n_written += len(written)
    return n_written


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument(
        "run_dir",
        help="Comparison output directory containing eval_viz/ "
        "(e.g. outputs/.../comparison/2026-04-08/12-57-02).",
    )
    args = p.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    eval_viz = os.path.join(run_dir, "eval_viz")
    if not os.path.isdir(eval_viz):
        # Allow passing eval_viz directly.
        if os.path.isdir(run_dir) and os.path.basename(run_dir) == "eval_viz":
            eval_viz = run_dir
            run_dir = os.path.dirname(run_dir)
        else:
            print(f"[replot] no eval_viz/ under {run_dir}", file=sys.stderr)
            return 2

    # Apply the same global plot style the pipeline used.  The canonical
    # source is `plot_style.global` in <run_dir>/.hydra/config.yaml,
    # which Hydra wrote when the pipeline ran.  No runtime snapshot is
    # required because no module mutates rcParams as an import-time
    # side effect any more — the entire style lives in the yaml.
    from graph_signal_diffusion.evaluation.plot_style_apply import (
        apply_global_plot_style_from_run_dir,
        load_plot_style_from_run_dir,
    )
    if apply_global_plot_style_from_run_dir(run_dir):
        print(
            f"[replot] applied plot_style.global from "
            f"{run_dir}/.hydra/config.yaml"
        )
    else:
        print(
            f"[replot] WARNING: no plot_style.global in "
            f"{run_dir}/.hydra/config.yaml; using matplotlib defaults"
        )

    # Full plot_style subtree + behavioral paper-figure cfg, recovered from the
    # snapshot so the self-styling paper figures (Fig 2 / Fig 3) replot
    # identically to the live run.
    paper_plot_style = load_plot_style_from_run_dir(run_dir)
    paper_cfg, baseline_order = _load_paper_cfg(run_dir)
    diffusion_steps = _load_diffusion_num_timesteps(run_dir)

    print(f"[replot] walking {eval_viz}")
    n = _walk_eval_viz(eval_viz)

    # Paper figures aggregate the comparison sidecars (Fig 3 needs every scorer
    # file together), so they run as a post-walk pass.
    paper_written = _replot_paper_figures(
        eval_viz, paper_plot_style, paper_cfg, baseline_order, diffusion_steps,
    )
    for w in paper_written:
        print(f"  ↺ {w}")
    n += len(paper_written)

    print(f"[replot] regenerated {n} pdfs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
