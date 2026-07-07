#!/usr/bin/env python3
"""
Sweep SP500 dynamic-graph sparsification hyperparameters and plot diagnostics.

This script mirrors SP500 dynamic graph creation (periodic correlation graphs),
then sweeps sparsification settings:
1) correlation threshold (pre-sparsification),
2) top_k,
3) min_degree.

It evaluates multiple period lengths (e.g., monthly and quarterly trading-day
cycles), computes per-window graph metrics, aggregates them per configuration,
and saves both tables and plots.

Optional: sample only `n_periods` random windows per (period, threshold) to
reduce runtime. Default `n_periods=null` keeps full-window behavior.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch



from graph_signal_diffusion.datasets.sp500.utils import (  # noqa: E402
    _sparsify_graph,
    compute_periodic_dynamic_adjacencies,
)


def parse_float_list(raw: str) -> list[float]:
    vals = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            vals.append(float(token))
    if not vals:
        raise ValueError("Expected at least one float value.")
    return vals


def parse_int_list(raw: str) -> list[int]:
    vals = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            v = int(token)
            if v <= 0:
                raise ValueError(f"Expected positive integer, got {v}")
            vals.append(v)
    if not vals:
        raise ValueError("Expected at least one integer value.")
    return vals


def parse_int_or_none_list(raw: str) -> list[Optional[int]]:
    vals: list[Optional[int]] = []
    for token in raw.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token in {"none", "null", "na"}:
            vals.append(None)
        else:
            v = int(token)
            if v < 0:
                raise ValueError(f"Expected non-negative integer or none, got {v}")
            vals.append(v)
    if not vals:
        raise ValueError("Expected at least one value.")
    return vals


def parse_optional_positive_int(raw: str) -> Optional[int]:
    token = raw.strip().lower()
    if token in {"none", "null", "na"}:
        return None
    value = int(token)
    if value <= 0:
        raise ValueError(f"Expected positive integer or null, got {value}")
    return value


def label_opt_int(v: Optional[int]) -> str:
    return "none" if v is None else str(v)


def sample_window_indices(
    total_windows: int,
    n_periods: Optional[int],
    rng: np.random.Generator,
) -> np.ndarray:
    if n_periods is None or n_periods >= total_windows:
        return np.arange(total_windows, dtype=np.int64)
    selected = rng.choice(total_windows, size=n_periods, replace=False)
    selected = np.sort(selected.astype(np.int64))
    return selected


def build_undirected_graph(
    num_nodes: int,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))
    if edge_index.numel() == 0:
        return g

    src = edge_index[0].detach().cpu().numpy().astype(np.int64)
    dst = edge_index[1].detach().cpu().numpy().astype(np.int64)
    wgt = edge_weight.detach().cpu().numpy().astype(np.float64)

    undirected_weights: dict[tuple[int, int], float] = {}
    for u, v, w in zip(src, dst, wgt):
        if u == v:
            continue
        a, b = (u, v) if u < v else (v, u)
        key = (a, b)
        prev = undirected_weights.get(key)
        if prev is None or w > prev:
            undirected_weights[key] = float(w)

    for (u, v), w in undirected_weights.items():
        g.add_edge(u, v, weight=w)
    return g


def compute_window_metrics(
    num_nodes: int,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
) -> dict:
    src = edge_index[0].detach().cpu().numpy().astype(np.int64) if edge_index.numel() > 0 else np.array([], dtype=np.int64)
    dst = edge_index[1].detach().cpu().numpy().astype(np.int64) if edge_index.numel() > 0 else np.array([], dtype=np.int64)
    wgt = edge_weight.detach().cpu().numpy().astype(np.float64) if edge_weight.numel() > 0 else np.array([], dtype=np.float64)

    out_deg = np.zeros(num_nodes, dtype=np.int64)
    out_wdeg = np.zeros(num_nodes, dtype=np.float64)
    if src.size > 0:
        np.add.at(out_deg, src, 1)
        np.add.at(out_wdeg, src, wgt)

    g = build_undirected_graph(num_nodes=num_nodes, edge_index=edge_index, edge_weight=edge_weight)
    undeg = np.array([d for _, d in g.degree()], dtype=np.int64)
    unwdeg = np.array([d for _, d in g.degree(weight="weight")], dtype=np.float64)

    components = list(nx.connected_components(g))
    component_sizes = sorted([len(c) for c in components], reverse=True)
    num_components = len(component_sizes)
    largest_component_size = component_sizes[0] if component_sizes else 0
    largest_component_fraction = largest_component_size / num_nodes if num_nodes > 0 else 0.0
    num_isolated_nodes = int(np.sum(undeg == 0)) if undeg.size > 0 else 0

    graph_radius = None
    graph_diameter = None
    largest_component_radius = None
    largest_component_diameter = None

    if num_nodes == 1:
        graph_radius = 0
        graph_diameter = 0
        largest_component_radius = 0
        largest_component_diameter = 0
    elif num_nodes > 1 and num_components == 1:
        graph_radius = int(nx.radius(g))
        graph_diameter = int(nx.diameter(g))
        largest_component_radius = graph_radius
        largest_component_diameter = graph_diameter
    elif component_sizes:
        lcc_nodes = max(components, key=len)
        g_lcc = g.subgraph(lcc_nodes).copy()
        if g_lcc.number_of_nodes() == 1:
            largest_component_radius = 0
            largest_component_diameter = 0
        else:
            largest_component_radius = int(nx.radius(g_lcc))
            largest_component_diameter = int(nx.diameter(g_lcc))

    return {
        "num_nodes": int(num_nodes),
        "num_edges_directed": int(src.size),
        "num_edges_undirected": int(g.number_of_edges()),
        "avg_out_degree": float(out_deg.mean()) if out_deg.size > 0 else 0.0,
        "avg_node_degree": float(undeg.mean()) if undeg.size > 0 else 0.0,
        "min_node_degree": int(undeg.min()) if undeg.size > 0 else 0,
        "max_node_degree": int(undeg.max()) if undeg.size > 0 else 0,
        "avg_weighted_out_degree": float(out_wdeg.mean()) if out_wdeg.size > 0 else 0.0,
        "avg_weighted_degree": float(unwdeg.mean()) if unwdeg.size > 0 else 0.0,
        "num_connected_components": int(num_components),
        "largest_component_fraction": float(largest_component_fraction),
        "num_isolated_nodes": int(num_isolated_nodes),
        "graph_radius": graph_radius,
        "graph_diameter": graph_diameter,
        "largest_component_radius": largest_component_radius,
        "largest_component_diameter": largest_component_diameter,
    }


def _nan_stats(values: list[Optional[float]]) -> tuple[float, float, float]:
    arr = np.array([np.nan if v is None else float(v) for v in values], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    return float(np.nanmean(arr)), float(np.nanmin(arr)), float(np.nanmax(arr))


def aggregate_window_metrics(window_rows: list[dict]) -> dict:
    n = len(window_rows)
    if n == 0:
        return {
            "num_windows": 0,
            "connected_window_fraction": float("nan"),
            "avg_node_degree_mean": float("nan"),
            "avg_node_degree_min": float("nan"),
            "avg_node_degree_max": float("nan"),
            "avg_weighted_degree_mean": float("nan"),
            "num_connected_components_mean": float("nan"),
            "num_connected_components_min": float("nan"),
            "num_connected_components_max": float("nan"),
            "largest_component_fraction_mean": float("nan"),
            "largest_component_fraction_min": float("nan"),
            "num_isolated_nodes_mean": float("nan"),
            "num_isolated_nodes_max": float("nan"),
            "graph_radius_mean_connected": float("nan"),
            "graph_radius_min_connected": float("nan"),
            "graph_radius_max_connected": float("nan"),
            "largest_component_radius_mean": float("nan"),
            "largest_component_diameter_mean": float("nan"),
        }

    connected = [int(r["num_connected_components"] == 1) for r in window_rows]
    avg_deg = [r["avg_node_degree"] for r in window_rows]
    avg_wdeg = [r["avg_weighted_degree"] for r in window_rows]
    comps = [r["num_connected_components"] for r in window_rows]
    lcc_frac = [r["largest_component_fraction"] for r in window_rows]
    iso = [r["num_isolated_nodes"] for r in window_rows]
    radius = [r["graph_radius"] for r in window_rows]
    lcc_radius = [r["largest_component_radius"] for r in window_rows]
    lcc_diam = [r["largest_component_diameter"] for r in window_rows]

    r_mean, r_min, r_max = _nan_stats(radius)
    lcc_r_mean, _, _ = _nan_stats(lcc_radius)
    lcc_d_mean, _, _ = _nan_stats(lcc_diam)

    return {
        "num_windows": n,
        "connected_window_fraction": float(np.mean(connected)),
        "avg_node_degree_mean": float(np.mean(avg_deg)),
        "avg_node_degree_min": float(np.min(avg_deg)),
        "avg_node_degree_max": float(np.max(avg_deg)),
        "avg_weighted_degree_mean": float(np.mean(avg_wdeg)),
        "num_connected_components_mean": float(np.mean(comps)),
        "num_connected_components_min": float(np.min(comps)),
        "num_connected_components_max": float(np.max(comps)),
        "largest_component_fraction_mean": float(np.mean(lcc_frac)),
        "largest_component_fraction_min": float(np.min(lcc_frac)),
        "num_isolated_nodes_mean": float(np.mean(iso)),
        "num_isolated_nodes_max": float(np.max(iso)),
        "graph_radius_mean_connected": r_mean,
        "graph_radius_min_connected": r_min,
        "graph_radius_max_connected": r_max,
        "largest_component_radius_mean": lcc_r_mean,
        "largest_component_diameter_mean": lcc_d_mean,
    }


def get_base_dynamic_graphs(
    values_path: Path,
    periods: list[int],
    thresholds: list[float],
    target_column_name: str,
    method: str,
) -> dict[tuple[int, float], list[dict]]:
    """
    Cache base dynamic graphs (without top_k/min_degree) for each period/threshold.
    """
    cache: dict[tuple[int, float], list[dict]] = {}
    for period in periods:
        for threshold in thresholds:
            print(
                f"Building base dynamic graphs: period={period}, threshold={threshold:g}"
            )
            graphs = compute_periodic_dynamic_adjacencies(
                values_path=str(values_path),
                period=int(period),
                target_column_name=target_column_name,
                correlation_threshold=float(threshold),
                method=method,
                top_k=None,
                min_degree=None,
            )
            cache[(int(period), float(threshold))] = graphs
    return cache


def make_sparse_dynamic_graphs(
    base_graphs: list[dict],
    num_nodes: int,
    top_k: Optional[int],
    min_degree: Optional[int],
) -> list[dict]:
    if top_k is None and min_degree is None:
        return base_graphs

    out = []
    for g in base_graphs:
        ei, ew = _sparsify_graph(
            edge_index=g["edge_index"],
            edge_weight=g["edge_weight"],
            num_nodes=num_nodes,
            top_k=top_k,
            min_degree=min_degree,
        )
        out.append(
            {
                "edge_index": ei,
                "edge_weight": ew,
                "start_timestep": int(g["start_timestep"]),
                "end_timestep": int(g["end_timestep"]),
                "num_dates_in_window": int(g["num_dates_in_window"]),
            }
        )
    return out


def sweep_dynamic_graphs(
    input_dir: Path,
    output_dir: Path,
    periods: list[int],
    thresholds: list[float],
    top_k_values: list[Optional[int]],
    min_degree_values: list[Optional[int]],
    target_column_name: str = "DailyLogReturn",
    method: str = "pearson",
    n_periods: Optional[int] = None,
    sampling_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    values_path = input_dir / "values.csv"
    if not values_path.exists():
        raise FileNotFoundError(f"{values_path} not found.")

    values = pd.read_csv(values_path).set_index(["Symbol", "Date"])
    num_nodes = len(values.index.get_level_values("Symbol").unique())

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SP500 Dynamic Graph Sparsity Sweep")
    print("=" * 80)
    print(f"Input dir: {input_dir}")
    print(f"Values path: {values_path}")
    print(f"Num nodes: {num_nodes}")
    print(f"Periods: {periods}")
    print(f"Thresholds: {thresholds}")
    print(f"top_k values: {[label_opt_int(v) for v in top_k_values]}")
    print(f"min_degree values: {[label_opt_int(v) for v in min_degree_values]}")
    print(f"n_periods sampling: {'full' if n_periods is None else n_periods}")
    print(f"sampling_seed: {sampling_seed}")

    base_cache = get_base_dynamic_graphs(
        values_path=values_path,
        periods=periods,
        thresholds=thresholds,
        target_column_name=target_column_name,
        method=method,
    )

    rng = np.random.default_rng(int(sampling_seed))
    sampled_indices_cache: dict[tuple[int, float], np.ndarray] = {}
    for key, graphs in base_cache.items():
        selected = sample_window_indices(
            total_windows=len(graphs),
            n_periods=n_periods,
            rng=rng,
        )
        sampled_indices_cache[key] = selected
        period, threshold = key
        print(
            f"Sampling windows: period={period} threshold={threshold:g} "
            f"used={len(selected)}/{len(graphs)}"
        )

    window_rows: list[dict] = []
    agg_rows: list[dict] = []
    total = len(periods) * len(thresholds) * len(top_k_values) * len(min_degree_values)
    counter = 0

    for period in periods:
        for threshold in thresholds:
            base_graphs = base_cache[(int(period), float(threshold))]
            selected_indices = sampled_indices_cache[(int(period), float(threshold))]
            selected_graphs = [base_graphs[int(i)] for i in selected_indices.tolist()]
            for top_k in top_k_values:
                for min_degree in min_degree_values:
                    counter += 1
                    print(
                        f"[{counter:>4}/{total}] period={period} threshold={threshold:g} "
                        f"top_k={label_opt_int(top_k)} min_degree={label_opt_int(min_degree)}"
                    )
                    dyn_graphs = make_sparse_dynamic_graphs(
                        base_graphs=selected_graphs,
                        num_nodes=num_nodes,
                        top_k=top_k,
                        min_degree=min_degree,
                    )

                    combo_window_rows: list[dict] = []
                    for wi, g in enumerate(dyn_graphs):
                        original_wi = int(selected_indices[wi])
                        metrics = compute_window_metrics(
                            num_nodes=num_nodes,
                            edge_index=g["edge_index"],
                            edge_weight=g["edge_weight"],
                        )
                        row = {
                            "period": int(period),
                            "threshold": float(threshold),
                            "top_k": top_k,
                            "min_degree": min_degree,
                            "top_k_label": label_opt_int(top_k),
                            "min_degree_label": label_opt_int(min_degree),
                            "window_idx": int(wi),
                            "window_idx_original": original_wi,
                            "start_timestep": int(g["start_timestep"]),
                            "end_timestep": int(g["end_timestep"]),
                            "num_dates_in_window": int(g["num_dates_in_window"]),
                        }
                        row.update(metrics)
                        combo_window_rows.append(row)
                        window_rows.append(row)

                    agg = aggregate_window_metrics(combo_window_rows)
                    agg_row = {
                        "period": int(period),
                        "threshold": float(threshold),
                        "top_k": top_k,
                        "min_degree": min_degree,
                        "top_k_label": label_opt_int(top_k),
                        "min_degree_label": label_opt_int(min_degree),
                        "num_windows_available": int(len(base_graphs)),
                        "num_windows_sampled": int(len(selected_graphs)),
                    }
                    agg_row.update(agg)
                    agg_rows.append(agg_row)

    df_windows = pd.DataFrame(window_rows)
    df_agg = pd.DataFrame(agg_rows)

    windows_jsonl = output_dir / "window_metrics.jsonl"
    windows_csv = output_dir / "window_metrics.csv"
    agg_jsonl = output_dir / "aggregate_metrics.jsonl"
    agg_csv = output_dir / "aggregate_metrics.csv"
    summary_json = output_dir / "summary.json"

    df_windows.to_json(windows_jsonl, orient="records", lines=True)
    df_windows.to_csv(windows_csv, index=False)
    df_agg.to_json(agg_jsonl, orient="records", lines=True)
    df_agg.to_csv(agg_csv, index=False)

    # Best configs prioritize connectivity, then larger components, then sparsity-aware degree.
    ranked = df_agg.sort_values(
        by=[
            "num_connected_components_mean",
            "largest_component_fraction_mean",
            "avg_node_degree_mean",
        ],
        ascending=[True, False, False],
    )
    summary = {
        "input_dir": str(input_dir),
        "target_column_name": target_column_name,
        "method": method,
        "periods": periods,
        "thresholds": thresholds,
        "n_periods": n_periods,
        "sampling_seed": int(sampling_seed),
        "top_k_values": [None if v is None else int(v) for v in top_k_values],
        "min_degree_values": [None if v is None else int(v) for v in min_degree_values],
        "num_combinations": int(total),
        "top_10_configs": ranked.head(10)[
            [
                "period",
                "threshold",
                "top_k",
                "min_degree",
                "num_windows",
                "connected_window_fraction",
                "num_connected_components_mean",
                "largest_component_fraction_mean",
                "avg_node_degree_mean",
                "largest_component_radius_mean",
            ]
        ].to_dict(orient="records"),
    }
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved window metrics:   {windows_jsonl}")
    print(f"Saved aggregate metrics:{agg_jsonl}")
    print(f"Saved summary:          {summary_json}")

    return df_windows, df_agg


def _heatmap_plot(
    df: pd.DataFrame,
    metric: str,
    thresholds: list[float],
    top_k_labels: list[str],
    min_degree_labels: list[str],
    out_path: Path,
    title_prefix: str,
) -> None:
    n_panels = len(thresholds)
    n_cols = min(4, n_panels)
    n_rows = int(math.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    cmap = "viridis"
    vmin = np.nanmin(df[metric].to_numpy(dtype=float))
    vmax = np.nanmax(df[metric].to_numpy(dtype=float))
    if not np.isfinite(vmin):
        vmin, vmax = 0.0, 1.0
    if vmin == vmax:
        vmax = vmin + 1e-9

    im = None
    for idx, threshold in enumerate(thresholds):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        sub = df[df["threshold"] == threshold]
        pivot = sub.pivot(index="min_degree_label", columns="top_k_label", values=metric)
        pivot = pivot.reindex(index=min_degree_labels, columns=top_k_labels)
        arr = pivot.to_numpy(dtype=float)
        im = ax.imshow(arr, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)

        ax.set_title(f"threshold={threshold:g}")
        ax.set_xlabel("top_k")
        ax.set_ylabel("min_degree")
        ax.set_xticks(np.arange(len(top_k_labels)))
        ax.set_xticklabels(top_k_labels, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(min_degree_labels)))
        ax.set_yticklabels(min_degree_labels)

        if arr.size <= 100:
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    val = arr[i, j]
                    txt = "nan" if np.isnan(val) else f"{val:.2f}"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="white")

    for idx in range(n_panels, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].axis("off")

    if im is not None:
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95)
        cbar.set_label(metric)
    fig.suptitle(f"{title_prefix}: {metric}", fontsize=14, fontweight="bold")
    fig.subplots_adjust(top=0.90, wspace=0.35, hspace=0.35)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _threshold_summary_plot(
    df: pd.DataFrame,
    metric: str,
    thresholds: list[float],
    out_path: Path,
    title: str,
) -> None:
    rows = []
    for t in thresholds:
        vals = df.loc[df["threshold"] == t, metric].to_numpy(dtype=float)
        rows.append(
            {
                "threshold": t,
                "min": float(np.nanmin(vals)),
                "median": float(np.nanmedian(vals)),
                "max": float(np.nanmax(vals)),
            }
        )
    s = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(s["threshold"], s["median"], marker="o", label="median")
    ax.fill_between(s["threshold"], s["min"], s["max"], alpha=0.25, label="min..max")
    ax.set_xlabel("threshold")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def make_plots(
    df_agg: pd.DataFrame,
    periods: list[int],
    thresholds: list[float],
    top_k_values: list[Optional[int]],
    min_degree_values: list[Optional[int]],
    output_dir: Path,
) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    top_k_labels = [label_opt_int(v) for v in top_k_values]
    min_degree_labels = [label_opt_int(v) for v in min_degree_values]

    heatmap_metrics = [
        "avg_node_degree_mean",
        "num_connected_components_mean",
        "largest_component_fraction_mean",
        "connected_window_fraction",
        "largest_component_radius_mean",
        "largest_component_diameter_mean",
        "num_isolated_nodes_mean",
    ]

    for period in periods:
        sub = df_agg[df_agg["period"] == period].copy()
        prefix = f"period_{period}"
        for metric in heatmap_metrics:
            out = plots_dir / f"{prefix}_heatmap_{metric}.png"
            _heatmap_plot(
                df=sub,
                metric=metric,
                thresholds=thresholds,
                top_k_labels=top_k_labels,
                min_degree_labels=min_degree_labels,
                out_path=out,
                title_prefix=f"Dynamic Sweep period={period}",
            )

        # Threshold-only summary across top_k/min_degree combinations.
        threshold_metrics = [
            "avg_node_degree_mean",
            "num_connected_components_mean",
            "largest_component_fraction_mean",
            "connected_window_fraction",
            "largest_component_radius_mean",
        ]
        for metric in threshold_metrics:
            out = plots_dir / f"{prefix}_threshold_summary_{metric}.png"
            _threshold_summary_plot(
                df=sub,
                metric=metric,
                thresholds=thresholds,
                out_path=out,
                title=f"period={period}: {metric} vs threshold",
            )

    print(f"Saved plots to: {plots_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep dynamic SP500 graph sparsification for monthly/quarterly periods "
            "and plot connectivity/degree/radius diagnostics."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Cleaned dataset raw dir containing values.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output dir (default: <input-dir>/dynamic_sparsity_sweep)",
    )
    parser.add_argument(
        "--periods",
        type=str,
        default="21,63",
        help=(
            "Comma-separated period lengths in trading days. "
            "Defaults include monthly (~21) and quarterly (~63) cycles."
        ),
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.0,0.2,0.5,0.7",
        help="Comma-separated dynamic correlation thresholds.",
    )
    parser.add_argument(
        "--top-k-values",
        type=str,
        default="none,20",
        help="Comma-separated top_k values. Use 'none' for no top-k sparsification.",
    )
    parser.add_argument(
        "--min-degree-values",
        type=str,
        default="none,2",
        help="Comma-separated min_degree values. Use 'none' for no min-degree constraint.",
    )
    parser.add_argument(
        "--n-periods",
        type=str,
        default="null",
        help=(
            "Number of dynamic windows to sample randomly per (period, threshold). "
            "Use 'null' to evaluate all windows (default behavior)."
        ),
    )
    parser.add_argument(
        "--sampling-seed",
        type=int,
        default=42,
        help="Random seed used for window sampling when --n-periods is set.",
    )
    parser.add_argument(
        "--target-column-name",
        type=str,
        default="DailyLogReturn",
        help="Column used to compute dynamic correlations.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="pearson",
        choices=["pearson"],
        help="Dynamic correlation method (current implementation supports pearson).",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "dynamic_sparsity_sweep"
    periods = parse_int_list(args.periods)
    thresholds = parse_float_list(args.thresholds)
    top_k_values = parse_int_or_none_list(args.top_k_values)
    min_degree_values = parse_int_or_none_list(args.min_degree_values)
    n_periods = parse_optional_positive_int(args.n_periods)

    for t in thresholds:
        if t < 0:
            raise ValueError(f"Threshold must be >= 0, got {t}")

    _, df_agg = sweep_dynamic_graphs(
        input_dir=input_dir,
        output_dir=output_dir,
        periods=periods,
        thresholds=thresholds,
        top_k_values=top_k_values,
        min_degree_values=min_degree_values,
        target_column_name=args.target_column_name,
        method=args.method,
        n_periods=n_periods,
        sampling_seed=args.sampling_seed,
    )
    make_plots(
        df_agg=df_agg,
        periods=periods,
        thresholds=thresholds,
        top_k_values=top_k_values,
        min_degree_values=min_degree_values,
        output_dir=output_dir,
    )

    print("\nDone.")
    print(f"Aggregate metrics: {output_dir / 'aggregate_metrics.csv'}")
    print(f"Window metrics:    {output_dir / 'window_metrics.csv'}")
    print(f"Plots:             {output_dir / 'plots'}")


if __name__ == "__main__":
    main()
