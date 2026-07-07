"""Smoke test: instantiate UGNN from YAML config and run a forward pass on a small Watts-Strogatz graph."""
import yaml
from pathlib import Path
import torch
import networkx as nx
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate

# Import UGNN and config dataclasses
from graph_signal_diffusion.models.ugnn import (
    UGNN,
    UGNNConfig,
    GNNConfig,
    PoolingConfig,
    UpsamplingConfig,
    EmbeddingConfig,
)


def _load_ugnn_from_yaml(yaml_path: Path) -> UGNN:
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    # Use Hydra's instantiate to create the UGNN model directly from the config dict
    model = instantiate(data)
    return model


def _compose_model_cfg(config_name: str):
    repo_root = Path(__file__).resolve().parents[2]
    config_dir = repo_root / "src" / "graph_signal_diffusion" / "conf" / "model"
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name=config_name)


def _build_ring_graph(n_nodes: int) -> torch.Tensor:
    src = []
    dst = []
    for i in range(n_nodes):
        src.extend([i, i])
        dst.extend([(i + 1) % n_nodes, i])
    return torch.tensor([src, dst], dtype=torch.long)


def test_ugnn_from_config_forward_smoke(tmp_path: Path):
    """Load a small UGNN config and run a forward pass to ensure no runtime errors and correct output shape."""
    # Locate the small config file in the repo
    repo_root = Path(__file__).resolve().parents[2]
    yaml_path = repo_root / 'src' / 'graph_signal_diffusion' / 'conf' / 'model' / 'ugnn.yaml'
    assert yaml_path.exists(), f"Config not found: {yaml_path}"

    # Build model using Hydra instantiate (canonical configuration)
    model = _load_ugnn_from_yaml(yaml_path)
    model.eval()

    # Create a small Watts-Strogatz graph
    n_nodes = 64
    k = 4
    p = 0.1
    G = nx.watts_strogatz_graph(n_nodes, k, p, seed=0)
    # Build undirected edge_index
    edges = list(G.edges())
    if len(edges) == 0:
        # Fallback: make a simple chain
        edges = [(i, i+1) for i in range(n_nodes-1)]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    # Create dummy input (B, T, N, F)
    B, T = 1, 4
    F_in = model.config.in_channels
    x = torch.randn(B, T, n_nodes, F_in)
    timesteps = torch.zeros(B, dtype=torch.long)

    with torch.no_grad():
        out, intermediates = model(x=x, timesteps=timesteps, edge_index=edge_index, return_intermediates=True)

    # Basic assertions
    assert isinstance(out, torch.Tensor)
    assert out.shape == (B, T, n_nodes, model.config.out_channels)
    # intermediates should be a dict when requested
    assert isinstance(intermediates, dict)
    # Check at least encoder intermediates exist
    encoder_keys = [k for k in intermediates.keys() if 'encoder' in k.lower()]
    assert len(encoder_keys) >= 1

    # Print skip connection tensor shapes
    print("Skip connection tensor shapes:")
    for key in sorted(intermediates.keys()):
        if 'encoder_level' in key and isinstance(intermediates[key], dict):
            level_data = intermediates[key]
            if 'after_pool' in level_data:
                shape = level_data['after_pool'].shape
                print(f"  {key}: {shape}")


@pytest.mark.parametrize(
    "config_name,checkpoint_name",
    [
        ("ugnn_sp500_v3", "ugnn_temporal_projected.pt"),
        ("ugnn_sp500_v3_temporal_cross_attn", "ugnn_temporal_cross_attn.pt"),
    ],
)
def test_ugnn_temporal_projected_config_train_infer_checkpoint_smoke(
    tmp_path: Path,
    config_name: str,
    checkpoint_name: str,
):
    cfg = _compose_model_cfg(config_name)

    # Make dataset-derived interpolation values explicit for this unit test.
    cfg.model.config.embedding_config.cond.channels = 2
    cfg.model.config.embedding_config.cond.shared_encoder.temporal.output.time_varying.learned_projection.source_timesteps = 20
    cfg.model.config.embedding_config.cond.shared_encoder.temporal.output.time_varying.learned_projection.target_timesteps = 5

    # Keep runtime small while preserving the temporal-projected code path.
    cfg.model.config.base_channels = 8
    cfg.model.config.channel_multipliers = [1, 1]
    cfg.model.config.pooling_config.gamma = [1, 1]
    cfg.model.config.num_bottleneck_layers = 0
    cfg.model.config.gnn_config.num_layers = 1
    cfg.model.config.gnn_config.K = 1

    model = instantiate(cfg.model)
    model.train()

    B, T, N = 2, 5, 12
    x = torch.randn(B, T, N, model.config.in_channels)
    cond = torch.randn(B, 20, N, 2)
    timesteps = torch.randint(0, model.config.embedding_config.num_timesteps, (B,))
    edge_index = _build_ring_graph(N)
    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32)

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3,
    )

    out, _ = model(
        x=x,
        timesteps=timesteps,
        edge_index=edge_index,
        edge_weight=edge_weight,
        cond=cond,
        return_intermediates=False,
    )
    assert out.shape == (B, T, N, model.config.out_channels)
    loss = out.pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    ckpt_path = tmp_path / checkpoint_name
    torch.save(model.state_dict(), ckpt_path)

    reloaded = instantiate(cfg.model).eval()
    reloaded.load_state_dict(torch.load(ckpt_path, weights_only=True), strict=True)

    with torch.no_grad():
        out_reload, _ = reloaded(
            x=x,
            timesteps=timesteps,
            edge_index=edge_index,
            edge_weight=edge_weight,
            cond=cond,
            return_intermediates=False,
        )

    assert out_reload.shape == (B, T, N, model.config.out_channels)
    assert torch.isfinite(out_reload).all()



if __name__ == "__main__":
    test_ugnn_from_config_forward_smoke(tmp_path=Path('.'))
    print("UGNN config smoke test passed.")
