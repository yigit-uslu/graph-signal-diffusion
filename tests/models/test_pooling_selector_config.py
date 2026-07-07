"""Tests that pooling selector kwargs propagate into the LearnableGraphPool selector."""
import torch
import pytest

from graph_signal_diffusion.models.components.pooling import PoolingGNN, StridedGraphMaxPool, StridedPoolingGNN
from graph_signal_diffusion.models.components.pooling import TemporalMLPSelectorHead


def test_selector_temperature_forwarding():
    selector_kwargs = {'temperature': 0.42}
    pool = StridedGraphMaxPool(
        gamma=2,
        K=1,
        selection_method='learned',
        in_channels=8,
        selector_kwargs=selector_kwargs,
        stride_input=1,
    )

    assert pool.selector is not None
    # temperature should be forwarded to the selector
    assert hasattr(pool.selector, 'temperature')
    assert pool.selector.temperature == 0.42

    # basic forward should run (no crash) on a tiny random input
    B, T, N, F = 1, 2, 16, 8
    x = torch.randn(B, T, N, F)
    edge_index = torch.tensor([[i for i in range(N) if i+1 < N for _ in (0,1)],
                               [i+1 for i in range(N-1)] + [i for i in range(N-1)]]).long()
    # Ensure edge_index shape is 2 x E (neighbors duplicated etc.) — for test simplicity make a chain
    # Rebuild edge_index properly
    edges = [(i, i+1) for i in range(N-1)]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    x_pooled, new_mask, indices, scores = pool(x, edge_index)
    assert x_pooled.shape == (B, T, N, F)
    assert new_mask.shape == (B, N)
    # assert scores.shape == (B, N)


def test_selector_ste_kwargs_forwarding():
    selector_kwargs = {
        'selection_mode': 'ste',
        'temperature': 1.5,
        'temperature_schedule': 'linear',
        'temperature_min': 0.5,
        'temperature_anneal_steps': 10,
        'temperature_warmup_steps': 2,
        'entropy_reg_weight': 1e-3,
    }
    pool = StridedGraphMaxPool(
        gamma=2,
        K=0,
        selection_method='learned',
        in_channels=8,
        selector_kwargs=selector_kwargs,
        stride_input=1,
    )

    assert pool.selector is not None
    assert pool.selector.selection_mode == "ste"
    assert pool.selector.temperature == 1.5
    assert pool.selector.temperature_schedule == "linear"
    assert pool.selector.temperature_min == 0.5
    assert pool.selector.temperature_anneal_steps == 10
    assert pool.selector.temperature_warmup_steps == 2
    assert pool.selector.entropy_reg_weight == 1e-3

    pool.selector.set_training_step(7)
    assert pool.selector.get_current_temperature() < 1.5
    assert pool.selector.get_current_temperature() > 0.5


def test_selector_packed_score_kwargs_forwarding():
    selector_kwargs = {
        'selector_version': 'v3',
        'packed_score_mode': 'auto',
        'packed_score_threshold': 0.35,
    }
    pool = StridedGraphMaxPool(
        gamma=2,
        K=0,
        selection_method='learned',
        in_channels=8,
        selector_kwargs=selector_kwargs,
        stride_input=1,
    )

    assert pool.selector is not None
    assert pool.selector.packed_score_mode == 'auto'
    assert abs(pool.selector.packed_score_threshold - 0.35) <= 1e-8


def test_selector_legacy_epoch_kwargs_alias_to_steps():
    selector_kwargs = {
        'selection_mode': 'ste',
        'temperature': 1.0,
        'temperature_schedule': 'linear',
        'temperature_min': 0.5,
        'temperature_anneal_epochs': 20,
        'temperature_warmup_epochs': 3,
    }
    pool = StridedGraphMaxPool(
        gamma=2,
        K=0,
        selection_method='learned',
        in_channels=8,
        selector_kwargs=selector_kwargs,
        stride_input=1,
    )

    assert pool.selector.temperature_anneal_steps == 20
    assert pool.selector.temperature_warmup_steps == 3
    # Deprecated aliases should mirror step values for compatibility.
    assert pool.selector.temperature_anneal_epochs == 20
    assert pool.selector.temperature_warmup_epochs == 3


def test_selector_can_disable_strided_pooling_gnn():
    selector_kwargs = {
        'use_strided_pooling_gnn': False,
        'selection_mode': 'ste',
    }
    pool = StridedGraphMaxPool(
        gamma=2,
        K=0,
        selection_method='learned',
        in_channels=8,
        selector_kwargs=selector_kwargs,
        stride_input=1,
    )

    assert isinstance(pool.selector.pooling_gnn, PoolingGNN)
    assert not isinstance(pool.selector.pooling_gnn, StridedPoolingGNN)


def test_selector_can_disable_pooling_gnn_entirely():
    selector_kwargs = {
        'use_pooling_gnn': False,
        'hidden_channels': 12,
        'selection_mode': 'ste',
    }
    pool = StridedGraphMaxPool(
        gamma=2,
        K=0,
        selection_method='learned',
        in_channels=8,
        selector_kwargs=selector_kwargs,
        stride_input=1,
    )

    assert pool.selector.pooling_gnn is None
    assert isinstance(pool.selector.selector_head, TemporalMLPSelectorHead)

    # Basic forward should work with MLP selector head only.
    B, T, N, F = 1, 3, 16, 8
    x = torch.randn(B, T, N, F)
    edges = [(i, i + 1) for i in range(N - 1)]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    x_pooled, new_mask, selected_indices, _ = pool(x, edge_index)
    assert x_pooled.shape == (B, T, N, F)
    assert new_mask.shape == (B, N)
    assert selected_indices is not None
    scores = pool.selector.compute_scores(x, edge_index)
    assert scores.shape == (B, N)


@pytest.mark.parametrize("null_value", [None, "null"])
def test_neighborhood_pooling_null_allowed_when_k_zero(null_value):
    pool = StridedGraphMaxPool(
        gamma=2,
        K=0,
        selection_method='learned',
        in_channels=8,
        selector_kwargs={'selection_mode': 'ste'},
        stride_input=1,
        neighborhood_pooling=null_value,
    )
    assert pool.neighborhood_agg is None


@pytest.mark.parametrize("null_value", [None, "null"])
def test_neighborhood_pooling_null_rejected_when_k_positive(null_value):
    with pytest.raises(ValueError, match="neighborhood_pooling must be 'max' or 'avg' when K > 0"):
        StridedGraphMaxPool(
            gamma=2,
            K=1,
            selection_method='learned',
            in_channels=8,
            selector_kwargs={'selection_mode': 'ste'},
            stride_input=1,
            neighborhood_pooling=null_value,
        )



if __name__ == "__main__":
    test_selector_temperature_forwarding()
    test_selector_ste_kwargs_forwarding()
    test_selector_legacy_epoch_kwargs_alias_to_steps()
    test_selector_can_disable_strided_pooling_gnn()
    test_selector_can_disable_pooling_gnn_entirely()
    print("Pooling selector config test passed.")
