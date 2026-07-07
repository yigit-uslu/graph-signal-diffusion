#!/usr/bin/env python
"""Quick no-channel sweep to choose convert_pd_samples_to_diffusion arguments.

This script intentionally avoids channel instantiation/caching. It reads only
primal_history trajectory rates and runs the same subset refinement objective
used by convert_pd_samples_to_diffusion.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import yaml

import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from convert_pd_samples_to_diffusion import (  # noqa: E402
    _load_pd_samples_from_primal_history,
    _resolve_input_paths,
    _summarize_rate_array,
    select_feasible_subset_refined,
)


TABLE_COLUMNS = [
    "window_size",
    "objective",
    "count",
    "min",
    "p1",
    "p5",
    "p10",
    "p25",
    "p50",
    "mean",
    "p75",
    "p90",
    "p95",
    "max",
    "below_r_min_fraction",
    "avg_violating_receivers_per_network",
    "feasible_networks",
]


def _parse_int_list(value: str) -> List[int]:
    out: List[int] = []
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    if not out:
        raise ValueError("Expected at least one integer value.")
    return out


def _parse_str_list(value: str) -> List[str]:
    out: List[str] = []
    for token in str(value).split(","):
        token = token.strip()
        if token:
            out.append(token)
    if not out:
        raise ValueError("Expected at least one string value.")
    return out


def _format_cell(col: str, value: Any) -> str:
    if isinstance(value, float):
        if col in {"below_r_min_fraction"}:
            return f"{value:.6f}"
        return f"{value:.6f}"
    return str(value)


def _render_markdown_table(rows: List[Dict[str, Any]]) -> str:
    header = "| " + " | ".join(TABLE_COLUMNS) + " |"
    sep = "| " + " | ".join("---" for _ in TABLE_COLUMNS) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(_format_cell(col, row[col]) for col in TABLE_COLUMNS) + " |")
    return "\n".join([header, sep] + body)


def run_sweep(
    *,
    input_path: str,
    window_sizes: List[int],
    refine_objectives: List[str],
    target_samples_per_network: int,
    subset_feasibility_tolerance: float,
    subset_bottleneck_nodes: int,
) -> Dict[str, Any]:
    input_path_obj = Path(input_path)
    if not input_path_obj.exists():
        raise FileNotFoundError(f"Input path not found: {input_path_obj}")

    run_dir, primal_history_path, _ = _resolve_input_paths(input_path_obj)
    config_path = run_dir / ".hydra" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not primal_history_path.exists():
        raise FileNotFoundError(f"primal_history.jsonl not found: {primal_history_path}")

    with open(config_path, "r") as f:
        config: Mapping[str, Any] = yaml.safe_load(f)

    canonical = _load_pd_samples_from_primal_history(
        primal_history_path=primal_history_path,
        config=config,
    )
    r_min = float(config["training"]["r_min"])

    rows: List[Dict[str, Any]] = []
    for window_size in window_sizes:
        for objective in refine_objectives:
            avg_rates_across_networks: List[np.ndarray] = []
            violating_receivers_per_network: List[int] = []
            feasible_networks = 0
            effective_window_sizes: List[int] = []

            for net_id in canonical["network_ids"]:
                net_data = canonical["networks"][net_id]
                rate_samples = net_data.get("rate_samples")
                if rate_samples is None:
                    raise ValueError(
                        f"Network {net_id} is missing rate samples in primal_history view; "
                        "cannot run no-channel smoke sweep."
                    )
                rates = np.asarray(rate_samples, dtype=np.float64)
                if rates.ndim != 2 or rates.shape[0] == 0:
                    raise ValueError(f"Network {net_id} has malformed rates with shape {rates.shape}")

                if window_size > 0 and window_size < rates.shape[0]:
                    rates_window = rates[-window_size:]
                else:
                    rates_window = rates
                effective_window_sizes.append(int(rates_window.shape[0]))

                selected_idx, summary = select_feasible_subset_refined(
                    rates=rates_window,
                    target_size=target_samples_per_network,
                    r_min=r_min,
                    feasibility_tolerance=subset_feasibility_tolerance,
                    num_bottleneck_nodes=subset_bottleneck_nodes,
                    objective=objective,
                )
                selected_rates = rates_window[selected_idx]
                avg_selected_rates = selected_rates.mean(axis=0)
                avg_rates_across_networks.append(avg_selected_rates)

                violating_receivers_per_network.append(int(summary["num_violating_receivers"]))
                feasible_networks += int(bool(summary["is_feasible"]))

            global_stats = _summarize_rate_array(
                np.concatenate(avg_rates_across_networks, axis=0),
                r_min=r_min,
            )

            rows.append(
                {
                    "window_size": int(window_size),
                    "objective": str(objective),
                    "count": int(global_stats["count"]),
                    "min": float(global_stats["min"]),
                    "p1": float(global_stats["p1"]),
                    "p5": float(global_stats["p5"]),
                    "p10": float(global_stats["p10"]),
                    "p25": float(global_stats["p25"]),
                    "p50": float(global_stats["p50"]),
                    "mean": float(global_stats["mean"]),
                    "p75": float(global_stats["p75"]),
                    "p90": float(global_stats["p90"]),
                    "p95": float(global_stats["p95"]),
                    "max": float(global_stats["max"]),
                    "below_r_min_fraction": float(global_stats["below_r_min_fraction"]),
                    "avg_violating_receivers_per_network": float(np.mean(violating_receivers_per_network)),
                    "feasible_networks": f"{feasible_networks}/{len(canonical['network_ids'])}",
                    "effective_window_size_min": int(min(effective_window_sizes)),
                    "effective_window_size_max": int(max(effective_window_sizes)),
                }
            )

    return {
        "input_path": str(input_path_obj),
        "run_dir": str(run_dir),
        "primal_history_path": str(primal_history_path),
        "network_count": int(len(canonical["network_ids"])),
        "target_samples_per_network": int(target_samples_per_network),
        "r_min": float(r_min),
        "subset_feasibility_tolerance": float(subset_feasibility_tolerance),
        "subset_bottleneck_nodes": int(subset_bottleneck_nodes),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "No-channel smoke sweep for conversion argument selection. "
            "Reads rates from primal_history only."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Run directory or primal_history.jsonl path.",
    )
    parser.add_argument(
        "--window-sizes",
        type=str,
        default="100,200,500,1000",
        help="Comma-separated window sizes to sweep.",
    )
    parser.add_argument(
        "--refine-objectives",
        type=str,
        default="min_rate,p1_rate,p5_rate",
        help="Comma-separated refinement objectives to sweep.",
    )
    parser.add_argument(
        "--target-samples-per-network",
        type=int,
        default=100,
        help="Target selected samples per network in each sweep point.",
    )
    parser.add_argument(
        "--subset-feasibility-tolerance",
        type=float,
        default=0.0,
        help="Allowed slack for feasibility threshold: r_min - tolerance.",
    )
    parser.add_argument(
        "--subset-bottleneck-nodes",
        type=int,
        default=5,
        help="Number of bottleneck receivers used for diversity feature space.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional path to write table rows as CSV.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to write full sweep payload as JSON.",
    )

    args = parser.parse_args()

    window_sizes = _parse_int_list(args.window_sizes)
    refine_objectives = _parse_str_list(args.refine_objectives)
    allowed_objectives = {"min_rate", "p1_rate", "p5_rate", "composite"}
    invalid = [obj for obj in refine_objectives if obj not in allowed_objectives]
    if invalid:
        raise ValueError(
            f"Invalid refine objective(s): {invalid}. Allowed: {sorted(allowed_objectives)}"
        )

    result = run_sweep(
        input_path=args.input,
        window_sizes=window_sizes,
        refine_objectives=refine_objectives,
        target_samples_per_network=args.target_samples_per_network,
        subset_feasibility_tolerance=args.subset_feasibility_tolerance,
        subset_bottleneck_nodes=args.subset_bottleneck_nodes,
    )

    print(
        f"Loaded {result['network_count']} networks from {result['primal_history_path']} "
        "(no channel instantiation)."
    )
    print(
        f"target_samples_per_network={result['target_samples_per_network']}, "
        f"r_min={result['r_min']}, "
        f"subset_feasibility_tolerance={result['subset_feasibility_tolerance']}, "
        f"subset_bottleneck_nodes={result['subset_bottleneck_nodes']}"
    )
    print("")
    print(_render_markdown_table(result["rows"]))

    if args.output_csv:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(result["rows"][0].keys()))
            writer.writeheader()
            writer.writerows(result["rows"])
        print(f"\nSaved CSV: {output_csv}")

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved JSON: {output_json}")


if __name__ == "__main__":
    main()
