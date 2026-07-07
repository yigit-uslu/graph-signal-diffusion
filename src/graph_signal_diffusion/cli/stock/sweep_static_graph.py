#!/usr/bin/env python3
"""
Sweep graph sparsity hyperparameters on a cleaned SP500 adjacency and plot diagnostics.

This script is intended to run after `scripts/clean_sp500_data.py`.
It evaluates the sparsity factors that affect SP500 PyG graph creation:

1) edge thresholding on the weighted adjacency (pre-PyG conversion),
2) top-k sparsification (incoming edges per node),
3) minimum degree constraint.

For each (threshold, top_k, min_degree) combination, it computes graph metrics and
saves both tabular outputs and plots.
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
from torch_geometric.utils import dense_to_sparse


# Add project root to path for internal imports

from graph_signal_diffusion.datasets.sp500.utils import _sparsify_graph


def parse_float_list(raw: str) -> list[float]:
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError("At least one threshold value must be provided.")
    return values


def parse_int_or_none_list(raw: str) -> list[Optional[int]]:
    values: list[Optional[int]] = []
    for token in raw.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token in {"none", "null", "na"}:
            values.append(None)
        else:
            iv = int(token)
            if iv < 0:
                raise ValueError(f"Expected non-negative integer, got {iv}.")
            values.append(iv)
    if not values:
        raise ValueError("At least one top_k/min_degree value must be provided.")
    return values


def label_opt_int(value: Optional[int]) -> str:
    return "none" if value is None else str(value)


def apply_threshold(adj: np.ndarray, threshold: float) -> np.ndarray:
    out = np.array(adj, copy=True, dtype=np.float32)
    np.fill_diagonal(out, 0.0)
    if threshold > 0:
        out[out < threshold] = 0.0
    return out


def adjacency_to_sparse(
    adj: np.ndarray,
    top_k: Optional[int],
    min_degree: Optional[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    num_nodes = adj.shape[0]
    edge_index, edge_weight = dense_to_sparse(torch.from_numpy(adj).float())
    if top_k is not None or min_degree is not None:
        edge_index, edge_weight = _sparsify_graph(
            edge_index=edge_index,
            edge_weight=edge_weight,
            num_nodes=num_nodes,
            top_k=top_k,
            min_degree=min_degree,
        )
    return edge_index, edge_weight


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

    # Merge directed pairs into one undirected edge (keep max weight).
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


def compute_graph_metrics(
    num_nodes: int,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
) -> dict:
    src = edge_index[0].detach().cpu().numpy().astype(np.int64) if edge_index.numel() > 0 else np.array([], dtype=np.int64)
    dst = edge_index[1].detach().cpu().numpy().astype(np.int64) if edge_index.numel() > 0 else np.array([], dtype=np.int64)
    wgt = edge_weight.detach().cpu().numpy().astype(np.float64) if edge_weight.numel() > 0 else np.array([], dtype=np.float64)

    # Directed (PyG) degree stats
    out_deg = np.zeros(num_nodes, dtype=np.int64)
    in_deg = np.zeros(num_nodes, dtype=np.int64)
    out_wdeg = np.zeros(num_nodes, dtype=np.float64)
    in_wdeg = np.zeros(num_nodes, dtype=np.float64)

    if src.size > 0:
        np.add.at(out_deg, src, 1)
        np.add.at(in_deg, dst, 1)
        np.add.at(out_wdeg, src, wgt)
        np.add.at(in_wdeg, dst, wgt)

    g = build_undirected_graph(num_nodes=num_nodes, edge_index=edge_index, edge_weight=edge_weight)
    undeg = np.array([d for _, d in g.degree()], dtype=np.int64) if num_nodes > 0 else np.array([], dtype=np.int64)
    unwdeg = np.array([d for _, d in g.degree(weight="weight")], dtype=np.float64) if num_nodes > 0 else np.array([], dtype=np.float64)

    num_edges_undirected = int(g.number_of_edges())
    num_edges_directed = int(src.size)
    possible_undirected = (num_nodes * (num_nodes - 1)) // 2 if num_nodes > 1 else 0
    possible_directed = num_nodes * (num_nodes - 1) if num_nodes > 1 else 0

    components = list(nx.connected_components(g))
    component_sizes = sorted([len(c) for c in components], reverse=True)
    num_components = len(component_sizes)
    largest_component_size = component_sizes[0] if component_sizes else 0
    largest_component_fraction = (largest_component_size / num_nodes) if num_nodes > 0 else 0.0

    isolated_nodes = int(np.sum(undeg == 0)) if undeg.size > 0 else 0

    graph_radius = None
    graph_diameter = None
    lcc_radius = None
    lcc_diameter = None

    if num_nodes == 1:
        graph_radius = 0
        graph_diameter = 0
        lcc_radius = 0
        lcc_diameter = 0
    elif num_nodes > 1 and num_components == 1:
        graph_radius = int(nx.radius(g))
        graph_diameter = int(nx.diameter(g))
        lcc_radius = graph_radius
        lcc_diameter = graph_diameter
    elif component_sizes:
        largest_component_nodes = max(components, key=len)
        g_lcc = g.subgraph(largest_component_nodes).copy()
        if g_lcc.number_of_nodes() == 1:
            lcc_radius = 0
            lcc_diameter = 0
        else:
            lcc_radius = int(nx.radius(g_lcc))
            lcc_diameter = int(nx.diameter(g_lcc))

    unique_deg, counts_deg = np.unique(undeg, return_counts=True) if undeg.size > 0 else (np.array([], dtype=int), np.array([], dtype=int))
    degree_distribution = {str(int(k)): int(v) for k, v in zip(unique_deg.tolist(), counts_deg.tolist())}

    return {
        "num_nodes": int(num_nodes),
        "num_edges_directed": num_edges_directed,
        "num_edges_undirected": num_edges_undirected,
        "edge_density_directed": float(num_edges_directed / possible_directed) if possible_directed > 0 else 0.0,
        "edge_density_undirected": float(num_edges_undirected / possible_undirected) if possible_undirected > 0 else 0.0,
        "avg_out_degree": float(out_deg.mean()) if out_deg.size > 0 else 0.0,
        "avg_in_degree": float(in_deg.mean()) if in_deg.size > 0 else 0.0,
        "avg_node_degree": float(undeg.mean()) if undeg.size > 0 else 0.0,
        "min_node_degree": int(undeg.min()) if undeg.size > 0 else 0,
        "max_node_degree": int(undeg.max()) if undeg.size > 0 else 0,
        "avg_weighted_out_degree": float(out_wdeg.mean()) if out_wdeg.size > 0 else 0.0,
        "avg_weighted_in_degree": float(in_wdeg.mean()) if in_wdeg.size > 0 else 0.0,
        "avg_weighted_degree": float(unwdeg.mean()) if unwdeg.size > 0 else 0.0,
        "num_connected_components": int(num_components),
        "connected_component_sizes_desc": component_sizes,
        "largest_component_size": int(largest_component_size),
        "largest_component_fraction": float(largest_component_fraction),
        "num_isolated_nodes": isolated_nodes,
        "graph_radius": graph_radius,
        "graph_diameter": graph_diameter,
        "largest_component_radius": lcc_radius,
        "largest_component_diameter": lcc_diameter,
        "degree_distribution": degree_distribution,
    }


def _heatmap_plot(
    df: pd.DataFrame,
    metric: str,
    thresholds: list[float],
    top_k_labels: list[str],
    min_degree_labels: list[str],
    out_path: Path,
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

    for idx, threshold in enumerate(thresholds):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        sub = df[df["threshold"] == threshold].copy()
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
                    text = "nan" if np.isnan(val) else f"{val:.2f}"
                    ax.text(j, i, text, ha="center", va="center", fontsize=8, color="white")

    # Hide unused subplots
    for idx in range(n_panels, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].axis("off")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.95)
    cbar.set_label(metric)
    fig.suptitle(f"SP500 Sparsity Sweep: {metric}", fontsize=14, fontweight="bold")
    fig.subplots_adjust(top=0.90, wspace=0.35, hspace=0.35)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _threshold_summary_plot(
    df: pd.DataFrame,
    metric: str,
    thresholds: list[float],
    out_path: Path,
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
    ax.set_title(f"{metric} vs threshold (across top_k/min_degree)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_sweep(
    input_dir: Path,
    thresholds: list[float],
    top_k_values: list[Optional[int]],
    min_degree_values: list[Optional[int]],
    output_dir: Path,
) -> pd.DataFrame:
    adj_path = input_dir / "adj.npy"
    if not adj_path.exists():
        raise FileNotFoundError(f"{adj_path} not found.")

    adj = np.load(adj_path)
    if adj.ndim != 2:
        raise ValueError(f"Expected 2D adjacency in {adj_path}, got shape {adj.shape}.")
    if adj.shape[0] != adj.shape[1]:
        raise ValueError(f"Adjacency must be square, got shape {adj.shape}.")

    num_nodes = adj.shape[0]
    results: list[dict] = []
    total = len(thresholds) * len(top_k_values) * len(min_degree_values)
    counter = 0

    print("=" * 80)
    print("SP500 Graph Sparsity Sweep")
    print("=" * 80)
    print(f"Input dir: {input_dir}")
    print(f"Adjacency shape: {adj.shape}")
    print(f"Combinations: {total}")

    for threshold in thresholds:
        adj_thr = apply_threshold(adj=adj, threshold=threshold)
        for top_k in top_k_values:
            for min_degree in min_degree_values:
                counter += 1
                print(
                    f"[{counter:>4}/{total}] threshold={threshold:g}, top_k={label_opt_int(top_k)}, "
                    f"min_degree={label_opt_int(min_degree)}"
                )

                edge_index, edge_weight = adjacency_to_sparse(
                    adj=adj_thr,
                    top_k=top_k,
                    min_degree=min_degree,
                )
                metrics = compute_graph_metrics(
                    num_nodes=num_nodes,
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                )
                row = {
                    "threshold": float(threshold),
                    "top_k": top_k,
                    "min_degree": min_degree,
                    "top_k_label": label_opt_int(top_k),
                    "min_degree_label": label_opt_int(min_degree),
                }
                row.update(metrics)
                results.append(row)

    df = pd.DataFrame(results)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_csv = output_dir / "sweep_results.csv"
    results_jsonl = output_dir / "sweep_results.jsonl"
    df.to_csv(results_csv, index=False)
    df.to_json(results_jsonl, orient="records", lines=True)
    print(f"\nSaved results CSV: {results_csv}")
    print(f"Saved results JSONL: {results_jsonl}")

    summary = {
        "input_dir": str(input_dir),
        "adjacency_shape": list(adj.shape),
        "thresholds": thresholds,
        "top_k_values": [None if v is None else int(v) for v in top_k_values],
        "min_degree_values": [None if v is None else int(v) for v in min_degree_values],
        "num_combinations": int(total),
        "best_by_connectivity": df.sort_values(
            by=["num_connected_components", "largest_component_fraction", "avg_node_degree"],
            ascending=[True, False, False],
        )
        .head(10)[
            [
                "threshold",
                "top_k",
                "min_degree",
                "num_connected_components",
                "largest_component_fraction",
                "avg_node_degree",
                "graph_radius",
                "largest_component_radius",
            ]
        ]
        .to_dict(orient="records"),
    }

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary JSON: {summary_path}")

    return df


def make_plots(
    df: pd.DataFrame,
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
        "avg_node_degree",
        "avg_weighted_degree",
        "num_connected_components",
        "largest_component_fraction",
        "graph_radius",
        "largest_component_radius",
        "largest_component_diameter",
        "num_isolated_nodes",
    ]

    for metric in heatmap_metrics:
        out = plots_dir / f"heatmap_{metric}.png"
        _heatmap_plot(
            df=df,
            metric=metric,
            thresholds=thresholds,
            top_k_labels=top_k_labels,
            min_degree_labels=min_degree_labels,
            out_path=out,
        )

    threshold_metrics = [
        "avg_node_degree",
        "num_connected_components",
        "largest_component_fraction",
        "largest_component_radius",
    ]
    for metric in threshold_metrics:
        out = plots_dir / f"threshold_summary_{metric}.png"
        _threshold_summary_plot(df=df, metric=metric, thresholds=thresholds, out_path=out)

    print(f"Saved plots to: {plots_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep threshold/top_k/min_degree on cleaned SP500 graph and plot "
            "sparsity diagnostics."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to cleaned dataset raw directory containing adj.npy",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for sweep tables/plots (default: <input-dir>/sparsity_sweep)",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.0,0.01,0.1,0.2,0.5,0.6,0.7",
        help="Comma-separated threshold values applied to adjacency weights.",
    )
    parser.add_argument(
        "--top-k-values",
        type=str,
        default="none,20",
        help="Comma-separated top_k values. Use 'none' for no top-k pruning.",
    )
    parser.add_argument(
        "--min-degree-values",
        type=str,
        default="none,1,2,10",
        help="Comma-separated min_degree values. Use 'none' for no min-degree constraint.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir is not None else input_dir / "sparsity_sweep"

    thresholds = parse_float_list(args.thresholds)
    top_k_values = parse_int_or_none_list(args.top_k_values)
    min_degree_values = parse_int_or_none_list(args.min_degree_values)

    for t in thresholds:
        if t < 0:
            raise ValueError(f"Threshold must be >= 0, got {t}")

    df = run_sweep(
        input_dir=input_dir,
        thresholds=thresholds,
        top_k_values=top_k_values,
        min_degree_values=min_degree_values,
        output_dir=output_dir,
    )
    make_plots(
        df=df,
        thresholds=thresholds,
        top_k_values=top_k_values,
        min_degree_values=min_degree_values,
        output_dir=output_dir,
    )

    print("\nDone.")
    print(f"Results: {output_dir / 'sweep_results.csv'}")
    print(f"Plots:   {output_dir / 'plots'}")


if __name__ == "__main__":
    main()
