#!/usr/bin/env python3
"""Single-forward UGNN selector diagnostics smoke script.

Builds a small graph signal batch (cloned graph topology, different signals), runs one
UGNN forward pass, aggregates selector diagnostics using the same trainer helpers, and
renders selector diagnostic figures.

Outputs are written to:
    <output_dir>/selector_diagnostics/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from graph_signal_diffusion.models.ugnn import (
    EmbeddingConfig,
    GNNConfig,
    PoolingConfig,
    UGNN,
    UGNNConfig,
)
from graph_signal_diffusion.trainers.trainer import DiffusionTrainer
from graph_signal_diffusion.utils.plotting import (
    plot_selector_diagnostics_jsonl,
    plot_selector_network_hard_mask_statistics,
    plot_selector_network_node_statistics,
    plot_selector_network_selection_frequency_evolution,
)


def _ring_edge_index(num_nodes: int, device: torch.device) -> torch.Tensor:
    edges = []
    for i in range(num_nodes):
        j = (i + 1) % num_nodes
        edges.append((i, j))
        edges.append((j, i))
    edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()
    return edge_index


def _generate_node_index_mean_signals(
    batch_size: int,
    timesteps: int,
    num_nodes: int,
    num_features: int,
    device: torch.device,
    noise_std: float = 1.0,
) -> torch.Tensor:
    """
    Generate i.i.d. Gaussian node signals with node-index-dependent mean.

    For node i, each (batch, time, feature) sample is drawn from:
        x[..., i, :] ~ N(mean=i, std=noise_std).
    """
    noise = torch.randn(batch_size, timesteps, num_nodes, num_features, device=device)
    node_means = torch.arange(num_nodes, device=device, dtype=noise.dtype).view(1, 1, num_nodes, 1)
    return node_means + noise_std * noise


def _build_ugnn(num_timesteps: int) -> UGNN:
    # Matches learned downsampling setup:
    # - gamma=[1,2,2,2]
    # - pool_K=0 (feature max-pooling identity)
    # - learned selector in STE mode
    config = UGNNConfig(
        in_channels=1,
        out_channels=1,
        base_channels=32,
        channel_multipliers=[1, 1, 1, 1],
        gnn_config=GNNConfig(
            K=2,
            num_layers=2,
            norm_type="layer",
            dropout=0.0,
            activation="silu",
            normalize=False,
            use_strided_conv=False,
            use_pre_activation=False,
            use_temporal_mixer=False,
        ),
        pooling_config=PoolingConfig(
            gamma=[1, 2, 2, 2],
            pool_K=0,
            selection_method="learned",
            selector_kwargs={
                "use_pooling_gnn": False,
                "hidden_channels": 16,
                "selection_mode": "ste",
                "temperature": 1.0,
                "temperature_schedule": "linear",
                "temperature_min": 0.3,
                "temperature_anneal_steps": 8000,
                "temperature_warmup_steps": 500,
                "entropy_reg_weight": 1.0e-3,
            },
        ),
        embedding_config=EmbeddingConfig(
            time_embed_dim=64,
            num_timesteps=num_timesteps,
            cond_embed_dim=32,
            cond_channels=None,
        ),
        num_bottleneck_layers=1,
        skip_connection_mode="concat",
    )
    return UGNN(config=config)


def _extract_selector_level_diags(model: torch.nn.Module) -> List[dict]:
    selector_level_diags: List[dict] = []
    for module in model.modules():
        get_diag = getattr(module, "get_selector_diagnostics", None)
        if not callable(get_diag):
            continue
        diag = get_diag()
        if not isinstance(diag, dict) or len(diag) == 0:
            continue

        level_diag = {"selector_level": len(selector_level_diags)}
        for k, v in diag.items():
            if isinstance(v, torch.Tensor):
                level_diag[k] = v.detach().cpu()
            elif isinstance(v, (float, int, str, bool)):
                level_diag[k] = v
        selector_level_diags.append(level_diag)
    return selector_level_diags


def _aggregate_selector_diag(selector_level_diags: List[dict]) -> dict:
    if not selector_level_diags:
        return {"num_selector_levels": 0, "selector_aux_total": 0.0, "stats_by_level": {}}

    # Mirrors graph_signal_diffusion.diffusion.ddpm._collect_selector_losses_and_diagnostics
    numeric_keys = [
        "temperature",
        "exploration_noise",
        "entropy_mean_norm",
        "entropy_reg",
        "entropy_reg_weight",
        "num_active_mean",
        "num_selected_mean",
        "selected_ratio_mean",
        "score_mean_active",
        "score_std_active",
        "soft_mask_mean_active",
        "soft_mask_std_active",
    ]
    stats_by_level = {k: {} for k in numeric_keys}
    agg = {k: 0.0 for k in numeric_keys}

    for level_idx, d in enumerate(selector_level_diags):
        for k in numeric_keys:
            v = float(d.get(k, 0.0))
            agg[k] += v
            stats_by_level[k][int(level_idx)] = v

    n = float(len(selector_level_diags))
    for k in numeric_keys:
        agg[k] /= n

    agg["num_selector_levels"] = int(len(selector_level_diags))
    agg["selector_aux_total"] = 0.0
    agg["stats_by_level"] = stats_by_level
    return agg


def _parse_graph_key(graph_key: str) -> Tuple[Optional[str], str]:
    if "::" in graph_key:
        dataset_name, network_tag = graph_key.split("::", 1)
        return dataset_name, network_tag
    return None, graph_key


def _write_selector_diagnostic_jsonl(
    out_jsonl: Path,
    epoch: int,
    avg_selector_diags: Dict[str, float],
    avg_selector_stats_by_level: Dict[str, Dict[str, float]],
    selector_level_freqs: Dict[int, torch.Tensor],
    selector_network_stats: Dict[str, Dict[str, dict]],
) -> None:
    global_row = {
        "scope": "global",
        "epoch": int(epoch),
        **{
            str(k): float(v) if isinstance(v, (int, float)) else float("nan")
            for k, v in avg_selector_diags.items()
        },
        "stats_by_level": avg_selector_stats_by_level,
        "selection_freq_mean_by_level": {
            str(level_idx): float(freqs.mean().item())
            for level_idx, freqs in selector_level_freqs.items()
        },
        "selection_freq_std_by_level": {
            str(level_idx): float(freqs.std(unbiased=False).item())
            for level_idx, freqs in selector_level_freqs.items()
        },
    }

    with open(out_jsonl, "w", encoding="utf-8") as f:
        f.write(json.dumps(global_row) + "\n")
        for graph_key, level_stats in selector_network_stats.items():
            dataset_name, network_tag = _parse_graph_key(graph_key)
            row = {
                "scope": "network",
                "epoch": int(epoch),
                "network_key": graph_key,
                "dataset_name": dataset_name,
                "network_id": network_tag,
                "selection_freq_mean_by_level": {
                    str(level_idx): float(level_data.get("freq_mean", float("nan")))
                    for level_idx, level_data in level_stats.items()
                },
                "selection_freq_std_by_level": {
                    str(level_idx): float(level_data.get("freq_std", float("nan")))
                    for level_idx, level_data in level_stats.items()
                },
                "num_nodes_by_level": {
                    str(level_idx): int(level_data.get("num_nodes", 0))
                    for level_idx, level_data in level_stats.items()
                },
                "num_graphs_by_level": {
                    str(level_idx): float(level_data.get("num_graphs", float("nan")))
                    for level_idx, level_data in level_stats.items()
                },
            }
            f.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="UGNN single-forward selector diagnostics smoke script")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/selector_single_forward_smoke"))
    parser.add_argument("--num-graphs", type=int, default=20, help="Batch size (graph clones)")
    parser.add_argument("--num-nodes", type=int, default=24)
    parser.add_argument("--timesteps", type=int, default=8)
    parser.add_argument("--num-diffusion-steps", type=int, default=500)
    parser.add_argument("--training-step", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--dataset-name", type=str, default="toy")
    parser.add_argument("--network-id", type=str, default="network_0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--noise-std", type=float, default=1.0)
    parser.add_argument("--save-pdf", action="store_true", help="Save PDF instead of PNG")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        print("CUDA requested but unavailable; falling back to CPU.")

    model = _build_ugnn(num_timesteps=args.num_diffusion_steps).to(device)
    model.train()

    # Apply step-based schedules (e.g., selector temperature annealing).
    for module in model.modules():
        set_step = getattr(module, "set_training_step", None)
        if callable(set_step):
            set_step(args.training_step)

    B = args.num_graphs
    T = args.timesteps
    N = args.num_nodes
    F = model.config.in_channels

    x = _generate_node_index_mean_signals(
        batch_size=B,
        timesteps=T,
        num_nodes=N,
        num_features=F,
        device=device,
        noise_std=args.noise_std,
    )
    diffusion_t = torch.randint(0, args.num_diffusion_steps, (B,), dtype=torch.long, device=device)
    edge_index = _ring_edge_index(N, device=device)

    # Reset selector state to guarantee diagnostics come from this forward only.
    for module in model.modules():
        reset = getattr(module, "reset_selector_state", None)
        if callable(reset):
            reset()

    with torch.no_grad():
        _out, _inter = model(
            x=x,
            timesteps=diffusion_t,
            edge_index=edge_index,
            edge_weight=None,
            cond=None,
            return_intermediates=False,
        )

    selector_level_diags = _extract_selector_level_diags(model)
    selector_diag = _aggregate_selector_diag(selector_level_diags)

    if selector_diag.get("num_selector_levels", 0) == 0:
        raise RuntimeError(
            "No selector diagnostics were found. Ensure learned selection is enabled at at least one UGNN level."
        )

    graph_key = f"{args.dataset_name}::{args.network_id}"
    batch_graph_keys = [graph_key for _ in range(B)]

    selector_diag_state = DiffusionTrainer._init_selector_diag_state()
    DiffusionTrainer._accumulate_selector_diag(selector_diag_state, selector_diag)
    DiffusionTrainer._accumulate_selector_level_diags(
        selector_diag_state,
        selector_level_diags,
        batch_graph_keys,
    )
    (
        avg_selector_diags,
        avg_selector_stats_by_level,
        selector_network_stats,
        selector_level_freqs,
        selector_network_node_stats,
    ) = DiffusionTrainer._finalize_selector_diag_state(selector_diag_state)

    out_dir = args.output_dir / "selector_diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    diag_jsonl_path = out_dir / "selector_diagnostic_summaries.jsonl"

    _write_selector_diagnostic_jsonl(
        out_jsonl=diag_jsonl_path,
        epoch=args.epoch,
        avg_selector_diags=avg_selector_diags,
        avg_selector_stats_by_level=avg_selector_stats_by_level,
        selector_level_freqs=selector_level_freqs,
        selector_network_stats=selector_network_stats,
    )

    plot_selector_diagnostics_jsonl(
        epoch_jsonl_path=diag_jsonl_path,
        out_dir=out_dir,
        save_pdf=args.save_pdf,
    )
    plot_selector_network_node_statistics(
        selector_network_node_stats=selector_network_node_stats,
        epoch=args.epoch,
        out_dir=out_dir,
        save_pdf=args.save_pdf,
    )
    plot_selector_network_hard_mask_statistics(
        selector_network_node_stats=selector_network_node_stats,
        epoch=args.epoch,
        out_dir=out_dir,
        save_pdf=args.save_pdf,
    )
    plot_selector_network_selection_frequency_evolution(
        diagnostic_jsonl_path=diag_jsonl_path,
        out_dir=out_dir,
        save_pdf=args.save_pdf,
    )

    print(f"Saved selector diagnostics to: {out_dir}")
    print(f"JSONL: {diag_jsonl_path}")
    print("Generated files:")
    for p in sorted(out_dir.glob("*")):
        print(f"  - {p.name}")


if __name__ == "__main__":
    main()
