"""Trainer-level DDIM sampling diagnostics heatmap test.

This test runs real DDIM sampling with a UGNN model that has learned downsampling
at 3 selector levels (gamma=[1,2,2,2] -> skip first gamma=1 level), requests
sampling-phase selector probes at explicit DDPM timesteps, and verifies heatmap output.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import torch
from torch_geometric.data import Data

from graph_signal_diffusion.diffusion.ddim import DDIM
from graph_signal_diffusion.models.ugnn import (
    EmbeddingConfig,
    GNNConfig,
    PoolingConfig,
    UGNN,
    UGNNConfig,
)
from graph_signal_diffusion.trainers.trainer import DiffusionTrainer


def _ring_edge_index(num_nodes: int, device: torch.device) -> torch.Tensor:
    edges = []
    for i in range(num_nodes):
        j = (i + 1) % num_nodes
        edges.append((i, j))
        edges.append((j, i))
    return torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()


def _build_ugnn_for_sampling(num_diffusion_steps: int) -> UGNN:
    config = UGNNConfig(
        in_channels=1,
        out_channels=1,
        base_channels=8,
        channel_multipliers=[1, 1, 1, 1],
        gnn_config=GNNConfig(
            K=1,
            num_layers=1,
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
            temporal_score_agg="mean",
            selector_kwargs={
                "use_pooling_gnn": False,
                "hidden_channels": 8,
                "selection_mode": "ste",
                "temperature": 1.0,
                "temperature_schedule": "linear",
                "temperature_min": 0.2,
                "temperature_anneal_steps": 1000,
                "temperature_warmup_steps": 0,
                "entropy_reg_weight": 1e-3,
            },
        ),
        embedding_config=EmbeddingConfig(
            time_embed_dim=32,
            num_timesteps=num_diffusion_steps,
            cond_embed_dim=16,
            cond_channels=None,
        ),
        num_bottleneck_layers=1,
        skip_connection_mode="concat",
    )
    return UGNN(config=config)


def _expected_ddim_probe_timesteps(diffusion: DDIM, requested_steps: list[int]) -> list[int]:
    requested = sorted(
        {
            int(max(0, min(diffusion.num_timesteps - 1, int(t))))
            for t in requested_steps
        },
        reverse=True,
    )
    visited = [int(t) for t in diffusion.ddim_timesteps.tolist()]
    out = []
    seen = set()
    for t_req in requested:
        nearest = min(visited, key=lambda t_seen: (abs(t_seen - t_req), -t_seen))
        if nearest not in seen:
            seen.add(nearest)
            out.append(nearest)
    return sorted(out, reverse=True)


def _run_ddim_sampling_selector_heatmap(
    output_dir: Path,
    *,
    patch_tqdm: bool = True,
) -> Path:
    """Run DDIM sampling diagnostics and return created heatmap path."""
    torch.manual_seed(0)
    device = torch.device("cpu")

    num_diffusion_steps = 500
    sampling_timesteps = 50
    num_nodes = 24
    num_samples = 20
    probe_steps = [0, 5, 10, 25, 50, 100, 200, 300, 400, 450]

    # Reduce tqdm overhead/noise for tests/script runs.
    tqdm_ctx = (
        patch("graph_signal_diffusion.diffusion.ddim.tqdm.tqdm", lambda it, *args, **kwargs: it)
        if patch_tqdm
        else nullcontext()
    )

    with tqdm_ctx:
        model = _build_ugnn_for_sampling(num_diffusion_steps=num_diffusion_steps).to(device).eval()
        diffusion = DDIM(
            model=model,
            num_timesteps=num_diffusion_steps,
            sampling_timesteps=sampling_timesteps,
            ddim_eta=0.0,
            beta_schedule="linear",
            beta_start=1e-4,
            beta_end=2e-2,
            loss_type="l2",
            parameterization="eps",
        ).to(device)

        optimizer = torch.optim.Adam(diffusion.parameters(), lr=1e-3)
        trainer = DiffusionTrainer(
            diffusion=diffusion,
            optimizer=optimizer,
            device=device,
            save_dir=str(output_dir / "trainer_chkpts"),
        )
        trainer.epoch = 0

        data = Data(edge_index=_ring_edge_index(num_nodes=num_nodes, device=device))
        data.network_id = torch.tensor([0], dtype=torch.long, device=device)
        data.dataset_name = "sp500"

        shape = (num_samples, 1, num_nodes, 1)
        generated_samples, selector_sampling_diagnostics = diffusion.sample(
            shape=shape,
            device=device,
            data=data,
            return_selector_sampling_diagnostics=True,
            selector_probe_timesteps=probe_steps,
        )

        assert tuple(generated_samples.shape) == shape
        assert isinstance(selector_sampling_diagnostics, dict)

        expected_resolved_probe_steps = _expected_ddim_probe_timesteps(diffusion, probe_steps)
        resolved_probe_steps = selector_sampling_diagnostics.get("probe_timesteps", [])
        assert [int(t) for t in resolved_probe_steps] == expected_resolved_probe_steps

        # Ensure sampling diagnostics are aggregated for exactly 3 selector levels.
        sel_cond = selector_sampling_diagnostics.get("selection_given_available_by_level", {})
        assert isinstance(sel_cond, dict)
        assert len(sel_cond) == 3

        # One static graph replicated across 20 samples; each probe should count all 20.
        probe_graph_count = selector_sampling_diagnostics.get("probe_graph_count", [])
        assert len(probe_graph_count) == len(expected_resolved_probe_steps)
        assert all(abs(float(c) - float(num_samples)) < 1e-6 for c in probe_graph_count)

        by_network = selector_sampling_diagnostics.get("by_network", {})
        assert isinstance(by_network, dict)
        assert "sp500::network_0" in by_network

        output_dir.mkdir(parents=True, exist_ok=True)
        trainer._plot_selector_sampling_diagnostics(
            selector_sampling_diagnostics=selector_sampling_diagnostics,
            data=data,
            viz_save_dir=str(output_dir),
            eval_split="val",
        )

    heatmap_path = (
        output_dir
        / "selector_diagnostics"
        / "selector_sampling_heatmap_val_dsp500_nnetwork_0_epoch_0001.pdf"
    )
    return heatmap_path


def test_ddim_sampling_selector_probe_heatmap_ugnn_gamma1222(tmp_path: Path) -> None:
    heatmap_path = _run_ddim_sampling_selector_heatmap(
        output_dir=tmp_path / "selector_diagnostics_ddim_sampling",
        patch_tqdm=True,
    )
    assert heatmap_path.exists()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DDIM sampling selector-heatmap diagnostics test workflow."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tests/figs/selector_diagnostics_ddim_sampling"),
        help="Directory where selector diagnostics visualizations are saved.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    out = _run_ddim_sampling_selector_heatmap(
        output_dir=args.output_dir,
        patch_tqdm=True,
    )
    print(f"Saved heatmap to: {out}")
