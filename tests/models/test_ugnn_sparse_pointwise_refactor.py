import time

import torch
import pytest

from graph_signal_diffusion.models.components.graph_conv import GNN
from graph_signal_diffusion.models.ugnn import (
    UGNN,
    UGNNConfig,
    GNNConfig,
    PoolingConfig,
    UpsamplingConfig,
    EmbeddingConfig,
)


def _chain_edge_index(num_nodes: int) -> torch.Tensor:
    edges = [(i, i + 1) for i in range(num_nodes - 1)]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return torch.cat([edge_index, edge_index.flip(0)], dim=1)


def _mask_with_ratio(batch_size: int, num_nodes: int, ratio: float) -> torch.Tensor:
    keep = max(0, min(num_nodes, int(round(num_nodes * ratio))))
    mask = torch.zeros(batch_size, num_nodes, dtype=torch.bool)
    if keep > 0:
        mask[:, :keep] = True
    return mask


def _build_test_gnn(
    pointwise_mode: str,
    pointwise_threshold: float = 0.75,
    *,
    temporal_mixer_schedule: str = "per_layer",
    use_pre_activation: bool = False,
) -> GNN:
    return GNN(
        in_channels=4,
        hidden_channels=8,
        out_channels=5,
        num_layers=2,
        K=2,
        norm_type="layer",
        dropout=0.0,
        activation="silu",
        use_temporal_mixer=True,
        temporal_mixer_schedule=temporal_mixer_schedule,
        temporal_kernel_size=3,
        temporal_causal=False,
        temporal_use_pointwise=True,
        use_pre_activation=use_pre_activation,
        use_strided=False,
        normalize=False,
        pointwise_sparse_mode=pointwise_mode,
        pointwise_sparse_threshold=pointwise_threshold,
    )


def _build_ugnn_config(
    *,
    pointwise_mode: str,
    compact_skip_cache: bool,
    cond_channels: int | None = None,
    cond_fusion_mode: str = "concat",
) -> UGNNConfig:
    return UGNNConfig(
        in_channels=1,
        out_channels=1,
        base_channels=8,
        channel_multipliers=[1, 2],
        gnn_config=GNNConfig(
            K=1,
            num_layers=1,
            norm_type="none",
            dropout=0.0,
            activation="relu",
            use_strided_conv=False,
            use_temporal_mixer=False,
            pointwise_sparse_mode=pointwise_mode,
            pointwise_sparse_threshold=0.75,
        ),
        pooling_config=PoolingConfig(
            gamma=[2, 2],
            pool_K=0,
            selection_method="stride",
            selector_version="v3",
        ),
        upsampling_config=UpsamplingConfig(method="zero"),
        embedding_config=EmbeddingConfig(
            time_embed_dim=16,
            num_timesteps=100,
            cond_channels=cond_channels,
            cond_embed_dim=8,
            cond_fusion_mode=cond_fusion_mode,  # type: ignore[arg-type]
        ),
        num_bottleneck_layers=1,
        skip_connection_mode="concat",
        compact_skip_cache=compact_skip_cache,
    )


def _collect_grads(model: torch.nn.Module) -> dict[str, torch.Tensor | None]:
    grads: dict[str, torch.Tensor | None] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        grads[name] = None if param.grad is None else param.grad.detach().clone()
    return grads


def _build_perf_gnn(pointwise_mode: str) -> GNN:
    return GNN(
        in_channels=16,
        hidden_channels=32,
        out_channels=16,
        num_layers=3,
        K=2,
        norm_type="layer",
        dropout=0.0,
        activation="silu",
        use_temporal_mixer=True,
        temporal_mixer_schedule="per_layer",
        temporal_kernel_size=3,
        temporal_causal=False,
        temporal_use_pointwise=True,
        use_strided=False,
        normalize=False,
        pointwise_sparse_mode=pointwise_mode,
        pointwise_sparse_threshold=0.75,
    )


def _benchmark_forward_cuda_ms(
    model: GNN,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    active_mask: torch.Tensor,
    *,
    warmup: int = 5,
    repeats: int = 20,
) -> tuple[float, float]:
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x, edge_index, active_mask=active_mask)
        torch.cuda.synchronize(device=x.device)

        torch.cuda.reset_peak_memory_stats(device=x.device)
        timings_ms: list[float] = []
        for _ in range(repeats):
            torch.cuda.synchronize(device=x.device)
            t0 = time.perf_counter()
            _ = model(x, edge_index, active_mask=active_mask)
            torch.cuda.synchronize(device=x.device)
            timings_ms.append((time.perf_counter() - t0) * 1000.0)

    median_ms = float(torch.tensor(timings_ms).median().item())
    peak_mb = float(torch.cuda.max_memory_allocated(device=x.device) / (1024.0 ** 2))
    return median_ms, peak_mb


@pytest.mark.parametrize("active_ratio", [1.0, 0.5, 0.25])
def test_gnn_packed_pointwise_matches_dense_forward(active_ratio: float):
    torch.manual_seed(0)
    B, T, N, F = 2, 4, 12, 4
    edge_index = _chain_edge_index(N)

    dense = _build_test_gnn("off")
    packed = _build_test_gnn("on")
    packed.load_state_dict(dense.state_dict(), strict=True)
    dense.eval()
    packed.eval()

    mask = _mask_with_ratio(B, N, active_ratio)
    x = torch.randn(B, T, N, F)
    x = x * mask.unsqueeze(1).unsqueeze(-1).to(dtype=x.dtype)

    out_dense = dense(x, edge_index, active_mask=mask)
    out_packed = packed(x, edge_index, active_mask=mask)

    max_abs_err = float((out_dense - out_packed).abs().max().item())
    assert max_abs_err <= 1e-6


@pytest.mark.parametrize("all_inactive", [False, True])
def test_gnn_packed_pointwise_matches_dense_gradients(all_inactive: bool):
    torch.manual_seed(1)
    B, T, N, F = 2, 3, 10, 4
    edge_index = _chain_edge_index(N)

    dense = _build_test_gnn("off")
    packed = _build_test_gnn("on")
    packed.load_state_dict(dense.state_dict(), strict=True)
    dense.train()
    packed.train()

    if all_inactive:
        mask = torch.zeros(B, N, dtype=torch.bool)
    else:
        mask = _mask_with_ratio(B, N, 0.5)

    x_base = torch.randn(B, T, N, F)
    x_base = x_base * mask.unsqueeze(1).unsqueeze(-1).to(dtype=x_base.dtype)

    x_dense = x_base.clone().requires_grad_(True)
    x_packed = x_base.clone().requires_grad_(True)

    out_dense = dense(x_dense, edge_index, active_mask=mask)
    out_packed = packed(x_packed, edge_index, active_mask=mask)

    target = torch.randn_like(out_dense)
    loss_dense = torch.nn.functional.mse_loss(out_dense, target)
    loss_packed = torch.nn.functional.mse_loss(out_packed, target)

    loss_dense.backward()
    loss_packed.backward()

    grads_dense = _collect_grads(dense)
    grads_packed = _collect_grads(packed)
    assert grads_dense.keys() == grads_packed.keys()

    for name in grads_dense:
        g_dense = grads_dense[name]
        g_packed = grads_packed[name]
        if g_dense is None or g_packed is None:
            assert g_dense is None and g_packed is None
            continue
        torch.testing.assert_close(g_dense, g_packed, atol=1e-6, rtol=1e-6)


def test_gnn_auto_mode_routes_by_batch_average_active_ratio():
    torch.manual_seed(2)
    B, T, N, F = 2, 3, 12, 4
    edge_index = _chain_edge_index(N)

    gnn = _build_test_gnn("auto", pointwise_threshold=0.75)
    gnn.eval()
    x = torch.randn(B, T, N, F)

    mask_dense = _mask_with_ratio(B, N, 1.0)
    x_dense = x * mask_dense.unsqueeze(1).unsqueeze(-1).to(dtype=x.dtype)
    _ = gnn(x_dense, edge_index, active_mask=mask_dense)
    assert gnn._last_pointwise_sparse_used is False
    assert abs(gnn._last_pointwise_sparse_active_ratio - 1.0) <= 1e-8

    mask_sparse = _mask_with_ratio(B, N, 0.5)
    x_sparse = x * mask_sparse.unsqueeze(1).unsqueeze(-1).to(dtype=x.dtype)
    _ = gnn(x_sparse, edge_index, active_mask=mask_sparse)
    assert gnn._last_pointwise_sparse_used is True
    assert abs(gnn._last_pointwise_sparse_active_ratio - 0.5) <= 1e-8


def test_gnn_packed_per_layer_mixer_fast_path_used_when_supported():
    torch.manual_seed(6)
    B, T, N, F = 2, 3, 12, 4
    edge_index = _chain_edge_index(N)
    gnn = _build_test_gnn("on", temporal_mixer_schedule="per_layer", use_pre_activation=False)
    gnn.eval()

    mask = _mask_with_ratio(B, N, 0.5)
    x = torch.randn(B, T, N, F)
    x = x * mask.unsqueeze(1).unsqueeze(-1).to(dtype=x.dtype)
    _ = gnn(x, edge_index, active_mask=mask)
    assert gnn._last_packed_layer_mixer_path_used is True


@pytest.mark.parametrize(
    ("temporal_mixer_schedule", "use_pre_activation"),
    [("per_layer", True), ("per_block", False)],
)
def test_gnn_packed_per_layer_mixer_fast_path_falls_back_when_unsupported(
    temporal_mixer_schedule: str,
    use_pre_activation: bool,
):
    torch.manual_seed(7)
    B, T, N, F = 2, 3, 12, 4
    edge_index = _chain_edge_index(N)
    gnn = _build_test_gnn(
        "on",
        temporal_mixer_schedule=temporal_mixer_schedule,
        use_pre_activation=use_pre_activation,
    )
    gnn.eval()

    mask = _mask_with_ratio(B, N, 0.5)
    x = torch.randn(B, T, N, F)
    x = x * mask.unsqueeze(1).unsqueeze(-1).to(dtype=x.dtype)
    _ = gnn(x, edge_index, active_mask=mask)
    assert gnn._last_packed_layer_mixer_path_used is False


def test_ugnn_supported_paths_use_packed_pre_gnn_and_skip_fusion_then_debug_falls_back():
    torch.manual_seed(8)
    N = 8
    edge_index = _chain_edge_index(N)
    cfg = _build_ugnn_config(pointwise_mode="on", compact_skip_cache=False)
    model = UGNN(cfg).eval()

    x = torch.randn(1, 3, N, 1)
    timesteps = torch.tensor([5], dtype=torch.long)

    out, _ = model(x, timesteps, edge_index, return_intermediates=False)
    assert out.shape == x.shape
    assert model.encoder.encoder_blocks[0]._last_packed_pre_gnn_fusion_used is False
    assert model.encoder.encoder_blocks[1]._last_packed_pre_gnn_fusion_used is True
    assert model.decoder.decoder_blocks[0]._last_packed_skip_fusion_used is True
    assert model.decoder.decoder_blocks[0]._last_packed_pre_gnn_fusion_used is True

    out_debug, _ = model(x, timesteps, edge_index, return_intermediates=True)
    assert out_debug.shape == x.shape
    assert all(
        not block._last_packed_pre_gnn_fusion_used
        for block in model.encoder.encoder_blocks
    )
    assert all(
        (not block._last_packed_pre_gnn_fusion_used) and (not block._last_packed_skip_fusion_used)
        for block in model.decoder.decoder_blocks
    )


@pytest.mark.parametrize("cond_fusion_mode", ["film", "cross_attention"])
def test_ugnn_film_and_cross_attention_force_dense_pre_gnn_fusion(cond_fusion_mode: str):
    torch.manual_seed(9)
    N = 8
    edge_index = _chain_edge_index(N)
    cfg = _build_ugnn_config(
        pointwise_mode="on",
        compact_skip_cache=False,
        cond_channels=2,
        cond_fusion_mode=cond_fusion_mode,
    )
    model = UGNN(cfg).eval()

    x = torch.randn(1, 3, N, 1)
    cond = torch.randn(1, 3, N, 2)
    timesteps = torch.tensor([6], dtype=torch.long)

    out, _ = model(x, timesteps, edge_index, cond=cond, return_intermediates=False)
    assert out.shape == x.shape
    assert all(
        not block._last_packed_pre_gnn_fusion_used
        for block in model.encoder.encoder_blocks
    )
    assert all(
        not block._last_packed_pre_gnn_fusion_used
        for block in model.decoder.decoder_blocks
    )


def test_ugnn_strict_checkpoint_load_compatible_across_sparse_modes():
    torch.manual_seed(3)
    N = 8
    edge_index = _chain_edge_index(N)

    cfg_off = _build_ugnn_config(pointwise_mode="off", compact_skip_cache=False)
    cfg_on = _build_ugnn_config(pointwise_mode="on", compact_skip_cache=False)

    model_off = UGNN(cfg_off)
    model_on = UGNN(cfg_on)

    state = model_off.state_dict()
    model_on.load_state_dict(state, strict=True)

    x = torch.randn(1, 3, N, 1)
    timesteps = torch.tensor([4], dtype=torch.long)

    out, _ = model_on(x, timesteps, edge_index)
    assert out.shape == x.shape


def test_ugnn_compact_skip_cache_matches_dense_and_keeps_debug_dense():
    torch.manual_seed(4)
    N = 8
    edge_index = _chain_edge_index(N)

    cfg_dense = _build_ugnn_config(pointwise_mode="off", compact_skip_cache=False)
    cfg_compact = _build_ugnn_config(pointwise_mode="off", compact_skip_cache=True)

    model_dense = UGNN(cfg_dense)
    model_compact = UGNN(cfg_compact)
    model_compact.load_state_dict(model_dense.state_dict(), strict=True)
    model_dense.eval()
    model_compact.eval()

    x = torch.randn(1, 3, N, 1)
    timesteps = torch.tensor([7], dtype=torch.long)

    out_dense, _ = model_dense(x, timesteps, edge_index, return_intermediates=False)
    out_compact, _ = model_compact(x, timesteps, edge_index, return_intermediates=False)
    torch.testing.assert_close(out_dense, out_compact, atol=1e-6, rtol=1e-6)

    time_emb = model_compact.time_embed(timesteps)
    _, skip_features_compact, _, _ = model_compact.encoder(
        x=x,
        timesteps=timesteps,
        edge_index=edge_index,
        time_emb=time_emb,
        return_intermediates=False,
    )
    assert isinstance(skip_features_compact[0], torch.Tensor)
    assert any(isinstance(s, dict) and s.get("kind") == "packed_skip_v1" for s in skip_features_compact[1:])

    _, skip_features_debug, _, _ = model_compact.encoder(
        x=x,
        timesteps=timesteps,
        edge_index=edge_index,
        time_emb=time_emb,
        return_intermediates=True,
    )
    assert all(isinstance(s, torch.Tensor) for s in skip_features_debug)


def test_gnn_config_validates_sparse_settings_without_temporal_mixer_override():
    with pytest.raises(ValueError, match="Unknown pointwise_sparse_mode"):
        GNNConfig(pointwise_sparse_mode="invalid_mode")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="pointwise_sparse_threshold must be in \\[0, 1\\]"):
        GNNConfig(pointwise_sparse_threshold=1.5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for performance smoke")
def test_gnn_sparse_pointwise_performance_smoke_cuda():
    torch.manual_seed(5)
    device = torch.device("cuda")

    B, T, N, C = 2, 4, 800, 16
    edge_index = _chain_edge_index(N).to(device)
    active_mask = _mask_with_ratio(B, N, 0.25).to(device)
    x = torch.randn(B, T, N, C, device=device)
    x = x * active_mask.unsqueeze(1).unsqueeze(-1).to(dtype=x.dtype)

    model_off = _build_perf_gnn("off").to(device).eval()
    model_auto = _build_perf_gnn("auto").to(device).eval()
    model_on = _build_perf_gnn("on").to(device).eval()
    model_auto.load_state_dict(model_off.state_dict(), strict=True)
    model_on.load_state_dict(model_off.state_dict(), strict=True)

    results: dict[str, tuple[float, float]] = {}
    for mode, model in (("off", model_off), ("auto", model_auto), ("on", model_on)):
        torch.cuda.empty_cache()
        median_ms, peak_mb = _benchmark_forward_cuda_ms(
            model,
            x,
            edge_index,
            active_mask,
            warmup=5,
            repeats=20,
        )
        results[mode] = (median_ms, peak_mb)

    print(
        "\nGNN sparse-pointwise performance smoke (CUDA):\n"
        f"| mode | median_forward_ms | peak_cuda_mb |\n"
        f"|---|---:|---:|\n"
        f"| off | {results['off'][0]:.3f} | {results['off'][1]:.2f} |\n"
        f"| auto | {results['auto'][0]:.3f} | {results['auto'][1]:.2f} |\n"
        f"| on | {results['on'][0]:.3f} | {results['on'][1]:.2f} |"
    )

    assert model_auto._last_pointwise_sparse_used is True
    assert model_on._last_pointwise_sparse_used is True
    assert all(results[mode][0] > 0.0 for mode in ("off", "auto", "on"))
    assert all(results[mode][1] >= 0.0 for mode in ("off", "auto", "on"))
