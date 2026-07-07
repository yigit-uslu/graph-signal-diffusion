"""Tests for selector_skip_gamma1_levels opt-in knob.

When enabled with selector_version='v3', gamma=1 levels should NOT
instantiate a NodeSelector (saving memory/optimizer state), while
gamma>1 levels should still have one.
"""

import torch
import pytest

from graph_signal_diffusion.models.ugnn import (
    UGNN,
    UGNNConfig,
    GNNConfig,
    PoolingConfig,
    UpsamplingConfig,
    EmbeddingConfig,
)
from graph_signal_diffusion.models.components.pooling import StridedGraphMaxPool


def _chain_edge_index(num_nodes: int) -> torch.Tensor:
    edges = [(i, i + 1) for i in range(num_nodes - 1)]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return torch.cat([edge_index, edge_index.flip(0)], dim=1)


def _make_config(
    *,
    gamma,
    selection_method="learned",
    selector_version="v3",
    selector_skip_gamma1_levels=False,
    pool_K=0,
    in_channels=1,
    out_channels=1,
):
    return UGNNConfig(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=8,
        channel_multipliers=[1] * (len(gamma) if isinstance(gamma, list) else 4),
        gnn_config=GNNConfig(
            num_layers=1,
            use_strided_conv=False,
        ),
        pooling_config=PoolingConfig(
            gamma=gamma,
            pool_K=pool_K,
            selection_method=selection_method,
            selector_version=selector_version,
            selector_skip_gamma1_levels=selector_skip_gamma1_levels,
            selector_use_cond_fusion=False,
            selector_time_emb_dim=None,
            selector_kwargs={
                "selection_mode": "hard",
                "cond_fusion": False,
            },
        ),
        upsampling_config=UpsamplingConfig(),
        embedding_config=EmbeddingConfig(time_embed_dim=32),
    )


def _count_params(model):
    return sum(p.numel() for p in model.parameters())


class TestSelectorSkipConstruction:
    """Verify selector presence/absence at each encoder level."""

    def test_knob_false_all_levels_have_selector(self):
        cfg = _make_config(
            gamma=[1, 1, 2, 1],
            selector_skip_gamma1_levels=False,
        )
        model = UGNN(cfg)
        for i, block in enumerate(model.encoder.encoder_blocks):
            assert block.pool.selector is not None, (
                f"Level {i}: selector should exist when knob is False"
            )

    def test_knob_true_skips_gamma1_levels(self):
        cfg = _make_config(
            gamma=[1, 1, 2, 1],
            selector_skip_gamma1_levels=True,
        )
        model = UGNN(cfg)
        for i, block in enumerate(model.encoder.encoder_blocks):
            gamma_i = cfg.pooling_config.gamma[i]
            if gamma_i == 1:
                assert block.pool.selector is None, (
                    f"Level {i} (gamma=1): selector should be skipped"
                )
            else:
                assert block.pool.selector is not None, (
                    f"Level {i} (gamma={gamma_i}): selector should exist"
                )

    def test_knob_true_preserves_selection_method(self):
        """selection_method should remain 'learned' even when selector is skipped."""
        cfg = _make_config(
            gamma=[1, 1, 2, 1],
            selector_skip_gamma1_levels=True,
        )
        model = UGNN(cfg)
        for block in model.encoder.encoder_blocks:
            assert block.pool.selection_method == "learned"

    def test_selector_packed_score_knobs_propagate_to_all_active_selectors(self):
        cfg = _make_config(
            gamma=[1, 2, 2, 2],
            selector_skip_gamma1_levels=True,
        )
        cfg.pooling_config.selector_packed_score_mode = "auto"
        cfg.pooling_config.selector_packed_score_threshold = 0.75

        model = UGNN(cfg)
        for i, block in enumerate(model.encoder.encoder_blocks):
            gamma_i = cfg.pooling_config.gamma[i]
            if gamma_i == 1:
                assert block.pool.selector is None
                continue

            assert block.pool.selector is not None
            assert block.pool.selector.packed_score_mode == "auto"
            assert abs(block.pool.selector.packed_score_threshold - 0.75) <= 1e-8


class TestSelectorSkipParameterCount:
    """Knob-enabled model should have fewer parameters."""

    def test_fewer_params_with_knob(self):
        cfg_off = _make_config(
            gamma=[1, 1, 2, 1],
            selector_skip_gamma1_levels=False,
        )
        cfg_on = _make_config(
            gamma=[1, 1, 2, 1],
            selector_skip_gamma1_levels=True,
        )
        model_off = UGNN(cfg_off, num_nodes=32, in_channels=1, out_channels=1)
        model_on = UGNN(cfg_on, num_nodes=32, in_channels=1, out_channels=1)
        assert _count_params(model_on) < _count_params(model_off)


class TestSelectorSkipForwardEquivalence:
    """Both knob states should produce identical forward outputs."""

    def test_forward_outputs_match(self):
        torch.manual_seed(42)
        N = 32
        B, T, F = 2, 1, 1
        gamma = [1, 1, 2, 1]

        cfg_off = _make_config(gamma=gamma, selector_skip_gamma1_levels=False)
        cfg_on = _make_config(gamma=gamma, selector_skip_gamma1_levels=True)

        model_off = UGNN(cfg_off)
        model_on = UGNN(cfg_on)

        # Copy all shared weights from model_off to model_on.
        state = model_off.state_dict()
        model_on.load_state_dict(state, strict=False)

        model_off.eval()
        model_on.eval()

        x = torch.randn(B, T, N, F)
        edge_index = _chain_edge_index(N)
        timesteps = torch.zeros(B, dtype=torch.long)

        with torch.no_grad():
            out_off = model_off(x, timesteps, edge_index)
            out_on = model_on(x, timesteps, edge_index)

        torch.testing.assert_close(out_off, out_on, atol=1e-5, rtol=1e-5)


class TestSelectorSkipRegression:
    """Knob should NOT affect legacy selector or non-learned methods."""

    def test_legacy_selector_unaffected(self):
        cfg = _make_config(
            gamma=[1, 1, 2, 1],
            selector_version="legacy",
            selector_skip_gamma1_levels=True,
        )
        model = UGNN(cfg)
        # Legacy + knob=True: selectors should still exist on all levels
        # because the knob only applies to v3.
        for i, block in enumerate(model.encoder.encoder_blocks):
            assert block.pool.selector is not None, (
                f"Level {i}: legacy selector should NOT be skipped"
            )

    def test_stride_method_unaffected(self):
        cfg = _make_config(
            gamma=[1, 1, 2, 1],
            selection_method="stride",
            selector_skip_gamma1_levels=True,
        )
        model = UGNN(cfg)
        # stride method never creates selectors regardless of knob.
        for block in model.encoder.encoder_blocks:
            assert block.pool.selector is None

    def test_invalid_skip_selector_with_learned_gamma_gt1_raises(self):
        with pytest.raises(
            ValueError,
            match="skip_selector=True is invalid with selection_method='learned' and gamma>1",
        ):
            StridedGraphMaxPool(
                gamma=2,
                K=1,
                selection_method="learned",
                in_channels=8,
                skip_selector=True,
            )


class TestSelectorSkipValidation:
    """skip_selector=True with gamma>1 and learned selection must raise."""

    def test_raises_for_gamma_gt1_learned(self):
        from graph_signal_diffusion.models.components.pooling import StridedGraphMaxPool

        with pytest.raises(ValueError, match="skip_selector=True is invalid"):
            StridedGraphMaxPool(
                gamma=2,
                K=0,
                selection_method="learned",
                in_channels=8,
                skip_selector=True,
            )

    def test_no_raise_for_gamma1_learned(self):
        from graph_signal_diffusion.models.components.pooling import StridedGraphMaxPool

        # gamma=1 + learned + skip_selector is the intended usage; must not raise.
        pool = StridedGraphMaxPool(
            gamma=1,
            K=0,
            selection_method="learned",
            in_channels=8,
            skip_selector=True,
        )
        assert pool.selector is None

    def test_no_raise_for_stride_skip(self):
        from graph_signal_diffusion.models.components.pooling import StridedGraphMaxPool

        # stride method + skip_selector is harmless (selector is None anyway).
        StridedGraphMaxPool(gamma=2, K=0, selection_method="stride", skip_selector=True)
