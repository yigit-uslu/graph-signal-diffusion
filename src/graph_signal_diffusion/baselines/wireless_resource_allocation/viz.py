"""Shared visualisation utilities for WRA baselines.

Provides:

* :func:`plot_power_allocation_histogram` — two-panel histogram + empirical-CDF
  figure for per-baseline power-allocation distributions.
* :func:`plot_wra_percentile_evolution_comparison` — cross-baseline overlay of
  cumulative-average ergodic-rate trajectories (p1, p5, mean).
"""
from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from graph_signal_diffusion.baselines.plot_utils import (
    ecdf as _ecdf,
    format_baseline_display_name,
)

logger = logging.getLogger(__name__)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _style_get(style_cfg: dict[str, Any], path: str, default: Any) -> Any:
    current: Any = style_cfg
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_tuple2(value: Any, default: tuple[float, float]) -> tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (_as_float(value[0], default[0]), _as_float(value[1], default[1]))
    return default


def _canonical_percentile_panel(value: Any) -> str | None:
    """Map user/config panel labels to canonical keys {'p1','p5','mean'}."""
    raw = str(value).strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    if raw in {"p1", "1", "q1", "percentile1", "1st", "first"}:
        return "p1"
    if raw in {"p5", "5", "q5", "percentile5", "5th"}:
        return "p5"
    if raw in {"mean", "avg", "average"}:
        return "mean"
    return None


def _resolve_panel_order(style_cfg: dict[str, Any]) -> list[str]:
    """Resolve plot panel order from style cfg, always returning p1/p5/mean."""
    default_order = ["p1", "p5", "mean"]
    raw = _style_get(style_cfg, "panel_order", default_order)

    if not isinstance(raw, (list, tuple)):
        logger.warning(
            "Invalid panel_order=%r; expected list/tuple. Using default %s.",
            raw,
            default_order,
        )
        return default_order

    ordered: list[str] = []
    seen: set[str] = set()
    invalid: list[str] = []
    for item in raw:
        canonical = _canonical_percentile_panel(item)
        if canonical is None:
            invalid.append(str(item))
            continue
        if canonical in seen:
            continue
        ordered.append(canonical)
        seen.add(canonical)

    if invalid:
        logger.warning(
            "Ignoring unsupported percentile panel labels: %s. "
            "Supported labels map to {p1, p5, mean}.",
            invalid,
        )

    # Ensure all three panels are always present.
    for key in default_order:
        if key not in seen:
            ordered.append(key)

    return ordered


def _infer_r_min_from_records(
    records_by_baseline: dict[str, list[dict]],
    baseline_order: list[str],
) -> float | None:
    """Infer a shared r_min from WRA evaluation records."""
    r_min_vals: list[float] = []
    for baseline_name in baseline_order:
        for rec in records_by_baseline.get(baseline_name, []):
            value = rec.get("r_min", None)
            if value is None:
                continue
            try:
                r_min_vals.append(float(value))
            except (TypeError, ValueError):
                logger.warning(
                    "Ignoring non-numeric r_min=%r in baseline '%s'.",
                    value,
                    baseline_name,
                )

    if not r_min_vals:
        return None

    unique_vals = sorted({round(v, 10) for v in r_min_vals})
    if len(unique_vals) > 1:
        logger.warning(
            "Multiple r_min values found in WRA records %s; using %.6f for plot.",
            unique_vals,
            unique_vals[0],
        )
    return float(unique_vals[0])


def _build_baseline_color_map(
    baseline_order: list[str],
    baseline_colors: dict[str, str] | None,
    default_cycle: list[str],
) -> dict[str, str]:
    """Build deterministic baseline->color mapping."""
    configured = baseline_colors or {}
    color_map: dict[str, str] = {}

    for baseline_name in baseline_order:
        color = configured.get(baseline_name, None)
        if color:
            color_map[baseline_name] = str(color)

    used = {c.lower() for c in color_map.values()}
    fallback_cycle = [c for c in default_cycle if c.lower() not in used]
    if not fallback_cycle:
        fallback_cycle = list(default_cycle)

    next_idx = 0
    for baseline_name in baseline_order:
        if baseline_name in color_map:
            continue
        color_map[baseline_name] = fallback_cycle[next_idx % len(fallback_cycle)]
        next_idx += 1

    return color_map


def plot_power_allocation_histogram(
    power_real: Any,
    power_gen: Any,
    save_path: str,
    baseline_name: str = "Baseline",
    bins: int = 60,
    dpi: int = 150,
    plot_style: dict[str, Any] | None = None,
) -> float:
    """Two-panel histogram + empirical-CDF figure for WRA power distributions.

    Computes and plots the distribution of **per-sample mean normalised power**
    — i.e. for each sample ``b`` the scalar

    .. math::

        \\bar{y}_b = \\frac{1}{N} \\sum_{i=1}^{N} y_{b,i},
        \\quad y_{b,i} = \\frac{p_{b,i}}{P_{\\max}} - 0.5 \\in [-0.5,\\,0.5]

    for both the real and generated allocations, then plots:

    * **Left panel** — overlaid density histograms of ``ȳ_real`` and
      ``ȳ_gen`` with sample mean/std annotated in the legend.  A dashed
      vertical line marks ``y = 0.5`` (full-power, ``P_max``).
    * **Right panel** — empirical CDFs with the ``|ΔF|`` region shaded in
      purple; the area of that region equals the Wasserstein-1 distance W₁.

    This parallels the per-trajectory NLL histogram produced by
    :meth:`GeometricRandomWalk._plot_nll_histograms` for the stock-price
    forecasting task.

    Args:
        power_real: 1-D numpy array of per-sample mean normalised power for
            real allocations.
        power_gen:  1-D numpy array of per-sample mean normalised power for
            generated allocations.
        save_path: Absolute or relative file path (PDF / PNG) to save the
            figure.  Parent directory is created if it does not exist.
        baseline_name: Short label for the generated distribution, e.g.
            ``"Full-Power"`` or ``"WMMSE"``.  Used in axis titles and legend.
        bins: Number of histogram bins (default 60).
        dpi: Figure DPI (default 150).
        plot_style: Optional style-override mapping loaded from
            ``cfg.plot_style.plots.power_allocation_histogram``.

    Returns:
        Wasserstein-1 distance between the real and generated distributions.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import wasserstein_distance as _w1

    style_cfg = _as_dict(plot_style)
    bins = max(1, _as_int(_style_get(style_cfg, "bins", bins), bins))
    dpi = max(1, _as_int(_style_get(style_cfg, "dpi", dpi), dpi))
    fig_size = _as_tuple2(_style_get(style_cfg, "figure_size", [14.0, 5.0]), (14.0, 5.0))
    width_ratios_raw = _style_get(style_cfg, "width_ratios", [1.2, 1.0])
    if isinstance(width_ratios_raw, (list, tuple)) and len(width_ratios_raw) == 2:
        width_ratios = [
            _as_float(width_ratios_raw[0], 1.2),
            _as_float(width_ratios_raw[1], 1.0),
        ]
    else:
        width_ratios = [1.2, 1.0]
    x_padding = _as_float(_style_get(style_cfg, "x_padding", 0.01), 0.01)
    full_power_x = _as_float(_style_get(style_cfg, "full_power_reference.x", 0.5), 0.5)
    full_power_color = str(_style_get(style_cfg, "full_power_reference.color", "gray"))
    full_power_linestyle = str(_style_get(style_cfg, "full_power_reference.linestyle", "--"))
    full_power_linewidth = _as_float(
        _style_get(style_cfg, "full_power_reference.linewidth", 0.9), 0.9
    )
    full_power_alpha = _as_float(_style_get(style_cfg, "full_power_reference.alpha", 0.7), 0.7)
    real_color = str(_style_get(style_cfg, "real_distribution.color", "steelblue"))
    real_alpha = _as_float(_style_get(style_cfg, "real_distribution.alpha", 0.55), 0.55)
    real_linewidth = _as_float(
        _style_get(style_cfg, "real_distribution.linewidth", 1.8), 1.8
    )
    gen_color = str(_style_get(style_cfg, "generated_distribution.color", "tab:orange"))
    gen_alpha = _as_float(_style_get(style_cfg, "generated_distribution.alpha", 0.50), 0.50)
    gen_linewidth = _as_float(
        _style_get(style_cfg, "generated_distribution.linewidth", 1.8), 1.8
    )
    cdf_gap_color = str(_style_get(style_cfg, "cdf_gap.color", "mediumpurple"))
    cdf_gap_alpha = _as_float(_style_get(style_cfg, "cdf_gap.alpha", 0.25), 0.25)
    x_label_left_fontsize = _as_float(_style_get(style_cfg, "fonts.x_label_left", 10), 10.0)
    x_label_right_fontsize = _as_float(_style_get(style_cfg, "fonts.x_label_right", 11), 11.0)
    y_label_fontsize = _as_float(_style_get(style_cfg, "fonts.y_label", 11), 11.0)
    title_fontsize = _as_float(_style_get(style_cfg, "fonts.title", 12), 12.0)
    legend_fontsize = _as_float(_style_get(style_cfg, "fonts.legend", 9), 9.0)

    baseline_display_name = format_baseline_display_name(baseline_name)
    w1 = float(_w1(power_real, power_gen))

    # Bin range: always include the full normalised range [-0.5, 0.5] plus
    # any data that falls outside (shouldn't happen in practice).
    all_vals = np.concatenate([power_real, power_gen])
    plot_min = min(float(all_vals.min()), -0.5) - x_padding
    plot_max = max(float(all_vals.max()), 0.5) + x_padding
    x_grid   = np.linspace(plot_min, plot_max, 2000)
    hist_bins = np.linspace(plot_min, plot_max, bins + 1)

    cdf_real = _ecdf(power_real, x_grid)
    cdf_gen  = _ecdf(power_gen,  x_grid)

    fig, (ax_hist, ax_cdf) = plt.subplots(
        1, 2, figsize=fig_size,
        gridspec_kw={"width_ratios": width_ratios},
    )

    # ── Left panel: density histograms ───────────────────────────────
    ax_hist.axvline(
        full_power_x,
        color=full_power_color,
        linestyle=full_power_linestyle,
        linewidth=full_power_linewidth,
        label=r"$P_{\max}$ ($y = 0.5$)", alpha=full_power_alpha,
    )
    ax_hist.hist(
        power_real, bins=hist_bins, density=True, alpha=real_alpha,
        color=real_color,
        label=(
            rf"Real  $\mu={power_real.mean():.3f}$,"
            rf"  $\sigma={power_real.std():.3f}$"
        ),
        linewidth=real_linewidth,
    )
    ax_hist.hist(
        power_gen, bins=hist_bins, density=True, alpha=gen_alpha,
        color=gen_color,
        label=(
            rf"{baseline_display_name}  $\mu={power_gen.mean():.3f}$,"
            rf"  $\sigma={power_gen.std():.3f}$"
        ),
        linewidth=gen_linewidth,
    )
    ax_hist.set_xlabel(
        r"Mean normalised power per sample"
        r"  $\bar{y}_b = \frac{1}{N}\sum_i\,\left(\frac{p_{b,i}}{P_{\max}}-0.5\right)$",
        fontsize=x_label_left_fontsize,
    )
    ax_hist.set_ylabel("Density", fontsize=y_label_fontsize)
    ax_hist.set_title(
        f"Power Allocation Distribution  (real vs {baseline_display_name})",
        fontsize=title_fontsize,
    )
    ax_hist.legend(fontsize=legend_fontsize)
    ax_hist.set_xlim(plot_min, plot_max)

    # ── Right panel: empirical CDFs ──────────────────────────────────
    ax_cdf.plot(
        x_grid, cdf_real, color=real_color, linewidth=real_linewidth,
        label=r"$F_{\mathrm{real}}$",
    )
    ax_cdf.plot(
        x_grid, cdf_gen, color=gen_color, linewidth=gen_linewidth,
        label=rf"$F_{{\mathrm{{{baseline_display_name}}}}}$",
    )
    ax_cdf.fill_between(
        x_grid, cdf_real, cdf_gen,
        alpha=cdf_gap_alpha, color=cdf_gap_color,
        label=(
            r"$|F_{\mathrm{real}} - F_{\mathrm{gen}}|$"
            rf"  (area $= W_1 = {w1:.4f}$)"
        ),
    )
    ax_cdf.set_xlabel(
        r"Mean normalised power  $\bar{y}$", fontsize=x_label_right_fontsize,
    )
    ax_cdf.set_ylabel("Empirical CDF", fontsize=y_label_fontsize)
    ax_cdf.set_title(
        rf"Empirical CDFs — $W_1 = {w1:.4f}$", fontsize=title_fontsize,
    )
    ax_cdf.legend(fontsize=legend_fontsize)
    ax_cdf.set_ylim(0, 1)
    ax_cdf.set_xlim(plot_min, plot_max)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{baseline_display_name}] power histogram saved to: {save_path}")
    return w1


# ---------------------------------------------------------------------------
# Cross-baseline percentile-evolution overlay
# ---------------------------------------------------------------------------

def plot_wra_percentile_evolution_comparison(
    records_by_baseline: dict[str, list[dict]],
    save_path: str,
    baseline_order: list[str] | None = None,
    baseline_colors: dict[str, str] | None = None,
    r_min: float | None = None,
    marker_every: int = 5,
    error_display: str = "bars",
    dpi: int = 150,
    plot_style: dict[str, Any] | None = None,
) -> None:
    r"""Overlay cumulative-average ergodic-rate percentile trajectories across baselines.

    For each baseline and the shared expert (real policy), computes how the
    low-tail ergodic-rate statistics evolve as more time-sharing slots are
    averaged.  Specifically, for slot index *t* (1-based) and baseline *b*,
    the plotted value is:

    .. math::

        q_t^{(b)} = \\text{percentile}_q\\Bigl(
            \\bigcup_{r \\in \\text{records}(b)}
            \\frac{1}{t}\\sum_{s=1}^{t} r_{\\text{gen\_rates\_per\_slot}}[s,:]
        \\Bigr)

    where the union pools all receiver rates from all network records.

    Three statistics are plotted (one subplot each): 1st percentile,
    5th percentile, and mean.

    For each realization index (``eval_batch_idx``), statistics are computed
    across all receivers from all test networks, then averaged across
    realizations. Variability across realizations for expert (real) and
    diffusion (U-GNN) can be rendered either as vertical error bars or as a
    shaded ±1σ band.

    The expert (real-policy) curve is computed identically from
    ``real_rates_per_slot`` of diffusion records when available, otherwise
    from the first baseline in plotting order. Baselines share the same test
    set, so real rates should match across them.

    Args:
        records_by_baseline: ``{baseline_name: [record, ...]}``.  Each record
            must contain ``gen_rates_per_slot`` and ``real_rates_per_slot``
            numpy arrays of shape ``[num_time_shares, n]``.
        save_path: Destination file path (PDF / PNG).
        baseline_order: Optional list controlling line-plot order.  Defaults
            to the insertion order of *records_by_baseline*.
        baseline_colors: Optional mapping ``{baseline_name: color}`` for fixed
            baseline colors across runs/orderings. Use key ``"expert"`` to
            override the expert (real) curve color.
        r_min: Optional explicit minimum-rate threshold. If omitted, attempts
            to infer from records.
        marker_every: Frequency (in slots) for plotting line markers.
        error_display: Variability style for expert/diffusion. One of
            ``{"bars", "band"}``.
        dpi: Figure DPI.
        plot_style: Optional style-override mapping loaded from
            ``cfg.plot_style.plots.percentile_evolution_comparison``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8-darkgrid")

    style_cfg = _as_dict(plot_style)
    marker_every = _as_int(_style_get(style_cfg, "marker_every", marker_every), marker_every)
    error_display = str(_style_get(style_cfg, "error_display", error_display))
    dpi = max(1, _as_int(_style_get(style_cfg, "dpi", dpi), dpi))
    figure_size = _as_tuple2(
        _style_get(style_cfg, "figure_size", [24.0, 5.0]),
        (24.0, 5.0),
    )
    suptitle_fontsize = _as_float(_style_get(style_cfg, "suptitle_fontsize", 13), 13.0)
    show_suptitle = bool(_style_get(style_cfg, "show_suptitle", False))
    wspace = _as_float(_style_get(style_cfg, "wspace", 0.25), 0.25)
    expert_label = str(_style_get(style_cfg, "expert_label", "Expert") or "Expert")
    qos_label_symbol = str(
        _style_get(style_cfg, "line_styles.qos.label_symbol", r"f_{\min}") or r"f_{\min}"
    )
    axis_label_fontsize = _as_float(_style_get(style_cfg, "fonts.axis_label", 16), 16.0)
    tick_fontsize = _as_float(_style_get(style_cfg, "fonts.tick", 16), 16.0)
    legend_fontsize = _as_float(_style_get(style_cfg, "fonts.legend", 14), 14.0)
    grid_alpha = _as_float(_style_get(style_cfg, "grid.alpha", 0.3), 0.3)
    expert_linewidth = _as_float(
        _style_get(style_cfg, "line_styles.expert.linewidth", 2.0), 2.0
    )
    expert_linestyle = str(_style_get(style_cfg, "line_styles.expert.linestyle", "-"))
    expert_marker = str(_style_get(style_cfg, "line_styles.expert.marker", "o"))
    expert_markersize = _as_float(
        _style_get(style_cfg, "line_styles.expert.markersize", 4.0), 4.0
    )
    baseline_linewidth = _as_float(
        _style_get(style_cfg, "line_styles.baseline.linewidth", 1.6), 1.6
    )
    baseline_marker_other = str(_style_get(style_cfg, "line_styles.baseline.marker_other", "o"))
    baseline_marker_diffusion = str(
        _style_get(style_cfg, "line_styles.baseline.marker_diffusion", "s")
    )
    baseline_markersize = _as_float(
        _style_get(style_cfg, "line_styles.baseline.markersize", 3.8), 3.8
    )
    qos_color = str(_style_get(style_cfg, "line_styles.qos.color", "red"))
    qos_linestyle = str(_style_get(style_cfg, "line_styles.qos.linestyle", "--"))
    qos_linewidth = _as_float(_style_get(style_cfg, "line_styles.qos.linewidth", 1.6), 1.6)
    qos_alpha = _as_float(_style_get(style_cfg, "line_styles.qos.alpha", 0.8), 0.8)
    band_alpha = _as_float(_style_get(style_cfg, "variability.band_alpha", 0.18), 0.18)
    band_linewidth = _as_float(
        _style_get(style_cfg, "variability.band_linewidth", 0.0), 0.0
    )
    bars_elinewidth = _as_float(
        _style_get(style_cfg, "variability.bars.elinewidth", 1.0), 1.0
    )
    bars_capsize = _as_float(_style_get(style_cfg, "variability.bars.capsize", 2), 2.0)
    bars_alpha = _as_float(_style_get(style_cfg, "variability.bars.alpha", 0.9), 0.9)
    bars_markerfacecolor = str(
        _style_get(style_cfg, "variability.bars.markerfacecolor", "white")
    )
    bars_markeredgewidth = _as_float(
        _style_get(style_cfg, "variability.bars.markeredgewidth", 1.0), 1.0
    )
    bars_linewidth = _as_float(_style_get(style_cfg, "variability.bars.linewidth", 0.0), 0.0)

    if not records_by_baseline:
        logger.warning("plot_wra_percentile_evolution_comparison: no records — skipping.")
        return

    order = baseline_order or list(records_by_baseline.keys())
    # Filter to baselines that actually have records
    order = [b for b in order if b in records_by_baseline and records_by_baseline[b]]

    if not order:
        logger.warning("plot_wra_percentile_evolution_comparison: all baselines empty — skipping.")
        return

    configured_colors = dict(baseline_colors or {})
    configured_colors.update(_as_dict(_style_get(style_cfg, "baseline_colors", {})))
    default_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    style_cycle = _style_get(style_cfg, "fallback_cycle", None)
    if isinstance(style_cycle, (list, tuple)) and style_cycle:
        default_cycle = [str(c) for c in style_cycle]
    if not default_cycle:
        default_cycle = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    color_map = _build_baseline_color_map(order, configured_colors, default_cycle)
    expert_color = str(
        configured_colors.get(
            "expert",
            _style_get(style_cfg, "line_styles.expert.color", "black"),
        )
    )

    try:
        marker_every_int = int(marker_every)
    except (TypeError, ValueError):
        logger.warning("Invalid marker_every=%r; using default marker_every=5.", marker_every)
        marker_every_int = 5
    if marker_every_int <= 0:
        logger.warning("Non-positive marker_every=%r; using marker_every=1.", marker_every)
        marker_every_int = 1

    error_display_mode = str(error_display).strip().lower()
    if error_display_mode not in {"bars", "band"}:
        logger.warning(
            "Invalid error_display=%r; expected one of {'bars','band'}. "
            "Falling back to 'bars'.",
            error_display,
        )
        error_display_mode = "bars"

    inferred_r_min = _infer_r_min_from_records(records_by_baseline, order)
    resolved_r_min: float | None
    if r_min is None:
        resolved_r_min = inferred_r_min
    else:
        try:
            resolved_r_min = float(r_min)
        except (TypeError, ValueError):
            logger.warning("Invalid explicit r_min=%r; skipping r_min overlay.", r_min)
            resolved_r_min = None
    if resolved_r_min is not None and not np.isfinite(resolved_r_min):
        logger.warning("Non-finite r_min=%r; skipping r_min overlay.", resolved_r_min)
        resolved_r_min = None
    if (
        resolved_r_min is not None
        and inferred_r_min is not None
        and abs(resolved_r_min - inferred_r_min) > 1e-8
    ):
        logger.warning(
            "Explicit r_min=%.6f differs from record-inferred r_min=%.6f; "
            "using explicit value.",
            resolved_r_min,
            inferred_r_min,
        )

    # ------------------------------------------------------------------
    # Determine S_min — truncate to shortest slot count across all records
    # ------------------------------------------------------------------
    all_slot_counts = [
        rec["gen_rates_per_slot"].shape[0]
        for b in order
        for rec in records_by_baseline[b]
    ]
    S_min = min(all_slot_counts)
    S_max = max(all_slot_counts)
    if S_min != S_max:
        logger.warning(
            "plot_wra_percentile_evolution_comparison: slot counts differ "
            "across records (%d–%d); truncating to S_min=%d.",
            S_min, S_max, S_min,
        )

    # ------------------------------------------------------------------
    # Helper: realization-aware trajectories over slots 1..S_min
    # ------------------------------------------------------------------
    def _percentile_trajectory_single_realization(
        records: list[dict],
        key: str,
    ) -> np.ndarray:
        """Return ``[4, S_min]`` array of [min, p1, p5, mean] for one realization."""
        traj = np.zeros((4, S_min))
        for t in range(S_min):
            pooled = np.concatenate([
                rec[key][:t + 1].mean(axis=0)   # [n]
                for rec in records
            ])                                   # [total_n]
            traj[0, t] = pooled.min()
            traj[1, t] = np.percentile(pooled, 1)
            traj[2, t] = np.percentile(pooled, 5)
            traj[3, t] = pooled.mean()
        return traj

    def _trajectory_mean_std_over_realizations(
        records: list[dict],
        key: str,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Return (mean, std, M) with shape ``[4, S_min]`` over realization groups.

        Realizations are grouped by ``record['eval_batch_idx']`` when available.
        For legacy records without this field, all records are treated as one
        realization.
        """
        by_eval_batch: dict[int, list[dict]] = {}
        missing_eval_idx = False
        for rec in records:
            if "eval_batch_idx" not in rec:
                missing_eval_idx = True
                eval_idx = 0
            else:
                try:
                    eval_idx = int(rec["eval_batch_idx"])
                except (TypeError, ValueError):
                    missing_eval_idx = True
                    eval_idx = 0
            by_eval_batch.setdefault(eval_idx, []).append(rec)

        if missing_eval_idx:
            logger.warning(
                "Some records are missing eval_batch_idx; grouped them into one "
                "realization for variability estimates."
            )

        eval_ids = sorted(by_eval_batch.keys())
        per_realization = np.stack(
            [
                _percentile_trajectory_single_realization(by_eval_batch[idx], key)
                for idx in eval_ids
            ],
            axis=0,
        )  # [M, 4, S_min]
        return per_realization.mean(axis=0), per_realization.std(axis=0), len(eval_ids)

    def _count_total_receivers(records: list[dict]) -> int:
        """Count receivers across unique test networks in a baseline's records."""
        per_network_rx: dict[tuple[Any, Any], int] = {}
        for idx, rec in enumerate(records):
            dataset_name = rec.get("dataset_name", "__unknown_dataset__")
            network_id = rec.get("network_id", f"__unknown_network_{idx}__")
            key = (dataset_name, network_id)
            rates = rec.get("real_rates_per_slot", None)
            if rates is None:
                rates = rec.get("gen_rates_per_slot", None)
            if rates is None or not hasattr(rates, "shape") or len(rates.shape) < 2:
                continue
            n_rx = int(rates.shape[1])
            if key in per_network_rx and per_network_rx[key] != n_rx:
                logger.warning(
                    "Receiver-count mismatch for network key %r: %d vs %d. "
                    "Using max.",
                    key,
                    per_network_rx[key],
                    n_rx,
                )
            per_network_rx[key] = max(per_network_rx.get(key, 0), n_rx)
        return int(sum(per_network_rx.values()))

    panel_order = _resolve_panel_order(style_cfg)
    panel_specs = {
        "p1": (1, "P1", "Ergodic p1-rate (bits/s/Hz)"),
        "p5": (2, "P5", "Ergodic p5-rate (bits/s/Hz)"),
        "mean": (3, "Mean", "Ergodic mean-rate (bits/s/Hz)"),
    }
    ordered_panels = [panel_specs[k] for k in panel_order]
    slot_axis = np.arange(1, S_min + 1)
    marker_indices = np.arange(0, S_min, marker_every_int, dtype=int)
    if marker_indices.size == 0:
        marker_indices = np.array([0], dtype=int)
    elif marker_indices[-1] != S_min - 1:
        marker_indices = np.append(marker_indices, S_min - 1)
    marker_slots = slot_axis[marker_indices]

    # Expert curve: use diffusion records when available (pairs naturally with
    # the stochastic diffusion curve), otherwise fall back to the first baseline.
    # With reference_policy="none" (expert-free transfer mode), records lack
    # real_rates_per_slot; in that case, skip the expert curve entirely.
    expert_source = "diffusion" if "diffusion" in order else order[0]
    expert_records = records_by_baseline[expert_source]
    _has_real_rates = any(
        "real_rates_per_slot" in rec for rec in expert_records
    )
    if _has_real_rates:
        expert_mean, expert_std, expert_m = _trajectory_mean_std_over_realizations(
            expert_records, "real_rates_per_slot"
        )
    else:
        expert_mean = np.zeros((4, S_min))
        expert_std = np.zeros((4, S_min))
        expert_m = 0
    num_total_rx = _count_total_receivers(expert_records)

    # Per-baseline generated trajectories (mean/std over realizations)
    baseline_stats = {
        b: _trajectory_mean_std_over_realizations(
            records_by_baseline[b], "gen_rates_per_slot"
        )
        for b in order
    }
    for b, (_, _, m_b) in baseline_stats.items():
        if m_b != expert_m:
            logger.warning(
                "Realization-count mismatch: expert source '%s' has M=%d but "
                "baseline '%s' has M=%d.",
                expert_source,
                expert_m,
                b,
                m_b,
            )

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, len(ordered_panels), figsize=figure_size, sharey=False,
                             layout=None)
    fig.set_layout_engine("none")
    fig.subplots_adjust(wspace=wspace)
    axes = np.atleast_1d(axes)
    if show_suptitle:
        fig.suptitle(
            f"WRA Ergodic-Rate Evolution (cumulative average across {num_total_rx} rx)",
            fontsize=suptitle_fontsize,
        )

    def _plot_variability(
        ax: Any,
        mean_vals: np.ndarray,
        std_vals: np.ndarray,
        color: str,
        marker: str,
        marker_size: float,
    ) -> None:
        if error_display_mode == "band":
            ax.fill_between(
                slot_axis,
                mean_vals - std_vals,
                mean_vals + std_vals,
                color=color,
                alpha=band_alpha,
                linewidth=band_linewidth,
            )
            return

        ax.errorbar(
            marker_slots,
            mean_vals[marker_indices],
            yerr=std_vals[marker_indices],
            fmt=marker,
            ecolor=color,
            color=color,
            elinewidth=bars_elinewidth,
            capsize=bars_capsize,
            alpha=bars_alpha,
            markersize=marker_size,
            markerfacecolor=bars_markerfacecolor,
            markeredgewidth=bars_markeredgewidth,
            linewidth=bars_linewidth,
        )

    for panel_idx, (ax, (stat_idx, stat_label, ylabel)) in enumerate(zip(axes, ordered_panels)):
        # Expert line (skipped in expert-free transfer mode)
        if _has_real_rates:
            ax.plot(
                slot_axis, expert_mean[stat_idx],
                color=expert_color,
                linewidth=expert_linewidth,
                linestyle=expert_linestyle,
                marker=expert_marker,
                markevery=marker_indices.tolist(),
                markersize=expert_markersize,
                label=expert_label,
            )
            _plot_variability(
                ax=ax,
                mean_vals=expert_mean[stat_idx],
                std_vals=expert_std[stat_idx],
                color=expert_color,
                marker=expert_marker,
                marker_size=expert_markersize,
            )
        # Baseline lines
        for b in order:
            display_name = format_baseline_display_name(b)
            b_mean, b_std, _ = baseline_stats[b]
            is_diffusion = str(b).strip().lower() == "diffusion"
            curve_marker = baseline_marker_diffusion if is_diffusion else baseline_marker_other
            ax.plot(
                slot_axis, b_mean[stat_idx],
                color=color_map[b], linewidth=baseline_linewidth,
                marker=curve_marker,
                markevery=marker_indices.tolist(),
                markersize=baseline_markersize,
                label=display_name,
            )
            # Show variability bars for diffusion (stochastic baseline).
            if is_diffusion:
                _plot_variability(
                    ax=ax,
                    mean_vals=b_mean[stat_idx],
                    std_vals=b_std[stat_idx],
                    color=color_map[b],
                    marker=curve_marker,
                    marker_size=baseline_markersize,
                )
        # Draw QoS threshold only for low-tail panels; skip Mean panel.
        if resolved_r_min is not None and stat_label != "Mean":
            ax.axhline(
                resolved_r_min,
                color=qos_color,
                linestyle=qos_linestyle,
                linewidth=qos_linewidth,
                alpha=qos_alpha,
                label=rf"${qos_label_symbol}$",
            )
        ax.set_xlabel("Time slots", fontsize=axis_label_fontsize)
        ax.set_ylabel(ylabel, fontsize=axis_label_fontsize)
        ax.tick_params(axis="both", labelsize=tick_fontsize)
        if panel_idx == 0:
            _leg = ax.legend(fontsize=legend_fontsize, frameon=True, framealpha=0.5,
                             edgecolor="gray", fancybox=True, facecolor="white")
            _leg.get_frame().set_linewidth(1.0)
        ax.grid(True, alpha=grid_alpha)

    # fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [WRA] percentile-evolution comparison saved to: {save_path}")
