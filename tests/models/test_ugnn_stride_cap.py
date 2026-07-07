"""Tests for max_gnn_stride and max_pool_stride caps on UGNN."""
import pytest
import torch
import networkx as nx

from graph_signal_diffusion.models.ugnn import (
    UGNN,
    UGNNConfig,
    GNNConfig,
    PoolingConfig,
    UpsamplingConfig,
    EmbeddingConfig,
)


def _make_config(
    max_gnn_stride=None,
    max_pool_stride=None,
    use_strided_conv=True,
    gammas=None,
    pool_K=0,
    selection_method='stride',
):
    """Build a minimal UGNNConfig with stride caps."""
    if gammas is None:
        gammas = [1, 1, 2, 2]
    return UGNNConfig(
        in_channels=1,
        out_channels=1,
        base_channels=16,
        channel_multipliers=[1] * len(gammas),
        gnn_config=GNNConfig(
            K=2,
            num_layers=1,
            norm_type='layer',
            dropout=0.0,
            activation='relu',
            normalize=False,
            use_strided_conv=use_strided_conv,
            strided_power_mode='a_plus_i',
            max_gnn_stride=max_gnn_stride,
        ),
        pooling_config=PoolingConfig(
            gamma=gammas,
            pool_K=pool_K,
            selection_method=selection_method,
            max_pool_stride=max_pool_stride,
        ),
        upsampling_config=UpsamplingConfig(method='zero'),
        embedding_config=EmbeddingConfig(
            time_embed_dim=16,
            num_timesteps=10,
        ),
        num_bottleneck_layers=1,
        skip_connection_mode='concat',
    )


def _get_gnn_gammas(model):
    """Extract the gamma from each encoder GNN, decoder GNN, and bottleneck GNN."""
    enc_gammas = []
    for block in model.encoder.encoder_blocks:
        # GNN stores layers, each ResidualGNNBlock has a TAGConvLayer with .gamma
        layer0 = block.gnn.layers[0]
        enc_gammas.append(layer0.conv.gamma)

    dec_gammas = []
    for block in model.decoder.decoder_blocks:
        layer0 = block.gnn.layers[0]
        dec_gammas.append(layer0.conv.gamma)

    bottleneck_gamma = model.bottleneck.layers[0].conv.gamma

    return enc_gammas, dec_gammas, bottleneck_gamma


def _get_pool_stride_inputs(model):
    """Extract stride_input from each encoder pool's NeighborhoodFeaturePooling."""
    strides = []
    for block in model.encoder.encoder_blocks:
        pool = block.pool
        if hasattr(pool, 'neighborhood_agg') and pool.neighborhood_agg is not None:
            strides.append(pool.neighborhood_agg.stride)
        else:
            strides.append(None)
    return strides


# ── GNNConfig validation ────────────────────────────────────────────


def test_max_gnn_stride_rejects_zero():
    with pytest.raises(ValueError, match="max_gnn_stride must be >= 1"):
        GNNConfig(max_gnn_stride=0)


def test_max_gnn_stride_rejects_negative():
    with pytest.raises(ValueError, match="max_gnn_stride must be >= 1"):
        GNNConfig(max_gnn_stride=-1)


def test_max_gnn_stride_accepts_one():
    cfg = GNNConfig(max_gnn_stride=1)
    assert cfg.max_gnn_stride == 1


def test_max_gnn_stride_none_is_default():
    cfg = GNNConfig()
    assert cfg.max_gnn_stride is None


# ── PoolingConfig validation ────────────────────────────────────────


def test_max_pool_stride_rejects_zero():
    with pytest.raises(ValueError, match="max_pool_stride must be >= 1"):
        PoolingConfig(max_pool_stride=0)


def test_max_pool_stride_none_is_default():
    cfg = PoolingConfig()
    assert cfg.max_pool_stride is None


# ── Encoder/Decoder/Bottleneck GNN gamma capping ────────────────────


def test_uncapped_gammas_match_stride_pre():
    """Without cap, GNN gammas equal stride_pre at each level."""
    config = _make_config(max_gnn_stride=None, gammas=[1, 1, 2, 2])
    model = UGNN(config)
    enc, dec, bn = _get_gnn_gammas(model)
    # stride_pre[i] = product(gammas[0..i-1]):  [1, 1, 1, 2], bottleneck_stride = 4
    assert enc == [1, 1, 1, 2]
    assert bn == 4


def test_capped_gammas():
    """With max_gnn_stride=2, levels with stride_pre > 2 are capped."""
    config = _make_config(max_gnn_stride=2, gammas=[1, 1, 2, 2])
    model = UGNN(config)
    enc, dec, bn = _get_gnn_gammas(model)
    # stride_pre = [1, 1, 1, 2] → all <= 2, so no encoder change
    assert enc == [1, 1, 1, 2]
    assert bn == 2  # bottleneck_stride=4, capped to 2


def test_cap_at_one_degrades_to_standard():
    """max_gnn_stride=1 makes all levels use gamma=1 (standard TAGConv behavior)."""
    config = _make_config(max_gnn_stride=1, gammas=[1, 1, 2, 2])
    model = UGNN(config)
    enc, dec, bn = _get_gnn_gammas(model)
    assert enc == [1, 1, 1, 1]
    assert bn == 1


def test_cap_higher_than_any_stride_is_noop():
    """max_gnn_stride=100 never activates for gammas=[1,1,2,2]."""
    config_capped = _make_config(max_gnn_stride=100, gammas=[1, 1, 2, 2])
    config_none = _make_config(max_gnn_stride=None, gammas=[1, 1, 2, 2])
    model_capped = UGNN(config_capped)
    model_none = UGNN(config_none)
    assert _get_gnn_gammas(model_capped) == _get_gnn_gammas(model_none)


def test_capped_gammas_larger_strides():
    """With gammas=[2,2,2,2] and max_gnn_stride=3, deeper levels are capped."""
    config = _make_config(max_gnn_stride=3, gammas=[2, 2, 2, 2])
    model = UGNN(config)
    enc, dec, bn = _get_gnn_gammas(model)
    # stride_pre = [1, 2, 4, 8] → capped to [1, 2, 3, 3]
    assert enc == [1, 2, 3, 3]
    assert bn == 3  # bottleneck_stride=16, capped to 3


def test_decoder_gammas_mirror_encoder():
    """Decoder GNN gammas should be capped the same way as encoder."""
    config = _make_config(max_gnn_stride=2, gammas=[1, 1, 2, 2])
    model = UGNN(config)
    enc, dec, bn = _get_gnn_gammas(model)
    # Decoder reverses the encoder levels
    assert dec == list(reversed(enc))


# ── Pool stride_input capping ────────────────────────────────────────


def test_pool_stride_input_capped(tmp_path):
    """max_pool_stride caps stride_input passed to NeighborhoodFeaturePooling."""
    config = _make_config(
        max_pool_stride=2,
        gammas=[1, 1, 2, 2],
        pool_K=1,  # Enable neighborhood pooling so NFP is created
    )
    model = UGNN(config)
    strides = _get_pool_stride_inputs(model)
    # stride_pre = [1, 1, 1, 2] → all <= 2, so no change
    assert strides == [1, 1, 1, 2]


def test_pool_stride_input_uncapped():
    """Without cap, stride_input equals stride_pre."""
    config = _make_config(
        max_pool_stride=None,
        gammas=[1, 1, 2, 2],
        pool_K=1,
    )
    model = UGNN(config)
    strides = _get_pool_stride_inputs(model)
    assert strides == [1, 1, 1, 2]


def test_pool_stride_input_cap_activates_on_large_strides():
    """With deeper cumulative stride, max_pool_stride should clip stride_input."""
    config = _make_config(
        max_pool_stride=2,
        gammas=[2, 2, 2, 2],
        pool_K=1,
    )
    model = UGNN(config)
    strides = _get_pool_stride_inputs(model)
    # stride_pre = [1, 2, 4, 8] -> clipped to [1, 2, 2, 2]
    assert strides == [1, 2, 2, 2]


def test_pool_stride_input_cap_at_one():
    """max_pool_stride=1 forces all pooling stride_input values to 1."""
    config = _make_config(
        max_pool_stride=1,
        gammas=[2, 2, 2, 2],
        pool_K=1,
    )
    model = UGNN(config)
    strides = _get_pool_stride_inputs(model)
    assert strides == [1, 1, 1, 1]


def test_pool_stride_input_large_cap_is_noop():
    """Very large cap should behave like uncapped pooling stride_input."""
    config_capped = _make_config(
        max_pool_stride=100,
        gammas=[2, 2, 2, 2],
        pool_K=1,
    )
    config_none = _make_config(
        max_pool_stride=None,
        gammas=[2, 2, 2, 2],
        pool_K=1,
    )
    model_capped = UGNN(config_capped)
    model_none = UGNN(config_none)
    assert _get_pool_stride_inputs(model_capped) == _get_pool_stride_inputs(model_none)


def test_pool_stride_cap_updates_legacy_learned_selector_gamma():
    """
    In legacy learned-selection path, cap propagates via stride_input -> selector gamma.
    """
    config_uncapped = _make_config(
        max_pool_stride=None,
        gammas=[2, 2, 2, 2],
        pool_K=1,
        selection_method='learned',
    )
    config_capped = _make_config(
        max_pool_stride=1,
        gammas=[2, 2, 2, 2],
        pool_K=1,
        selection_method='learned',
    )
    model_uncapped = UGNN(config_uncapped)
    model_capped = UGNN(config_capped)

    # Deepest encoder level has stride_pre=8 for gammas=[2,2,2,2].
    uncapped_selector_gnn = model_uncapped.encoder.encoder_blocks[-1].pool.selector.pooling_gnn
    capped_selector_gnn = model_capped.encoder.encoder_blocks[-1].pool.selector.pooling_gnn

    assert type(uncapped_selector_gnn).__name__ == 'StridedPoolingGNN'
    assert type(capped_selector_gnn).__name__ == 'PoolingGNN'


# ── use_strided_conv=False ignores cap ───────────────────────────────


def test_cap_ignored_when_strided_conv_disabled():
    """When use_strided_conv=False, max_gnn_stride has no effect (gamma is always 2)."""
    config = _make_config(
        max_gnn_stride=1,
        use_strided_conv=False,
        gammas=[1, 1, 2, 2],
    )
    model = UGNN(config)
    enc, dec, bn = _get_gnn_gammas(model)
    # All gammas should be 2 (the default for non-strided conv)
    assert enc == [2, 2, 2, 2]
    assert bn == 2


# ── Forward pass smoke test ─────────────────────────────────────────


def test_capped_model_forward_pass():
    """Capped model runs a forward pass without errors and produces correct shapes."""
    config = _make_config(max_gnn_stride=2, gammas=[1, 1, 2, 2])
    model = UGNN(config)
    model.eval()

    n_nodes = 32
    G = nx.watts_strogatz_graph(n_nodes, 4, 0.1, seed=42)
    edges = list(G.edges())
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    B, T = 2, 1
    x = torch.randn(B, T, n_nodes, config.in_channels)
    timesteps = torch.zeros(B, dtype=torch.long)

    with torch.no_grad():
        out, intermediates = model(
            x=x, timesteps=timesteps, edge_index=edge_index,
            return_intermediates=True,
        )

    assert out.shape == (B, T, n_nodes, config.out_channels)
