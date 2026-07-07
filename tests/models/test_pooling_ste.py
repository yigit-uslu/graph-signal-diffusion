"""Unit tests for STE-based learned pooling behavior."""

import torch

from graph_signal_diffusion.models.components.pooling import LearnableGraphPool


def _chain_edge_index(num_nodes: int) -> torch.Tensor:
    edges = [(i, i + 1) for i in range(num_nodes - 1)]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return torch.cat([edge_index, edge_index.flip(0)], dim=1)


def test_ste_selection_respects_per_sample_topk():
    torch.manual_seed(0)
    B, T, N, F = 2, 4, 10, 8
    x = torch.randn(B, T, N, F)
    edge_index = _chain_edge_index(N)

    active_mask = torch.zeros(B, N, dtype=torch.bool)
    active_mask[0, :8] = True  # k=4 when ratio=0.5
    active_mask[1, :5] = True  # k=2 when ratio=0.5

    pool = LearnableGraphPool(
        in_channels=F,
        pooling_ratio=0.5,
        selection_mode="ste",
        temperature=1.0,
    )
    pool.train()

    x_pooled, new_mask, _, _ = pool(
        x=x,
        edge_index=edge_index,
        active_mask=active_mask,
    )

    assert x_pooled.shape == (B, T, N, F)
    assert new_mask.shape == (B, N)
    assert int(new_mask[0].sum().item()) == 4
    assert int(new_mask[1].sum().item()) == 2


def test_ste_allows_gradient_flow_to_selector_parameters():
    torch.manual_seed(0)
    B, T, N, F = 2, 3, 12, 6
    x = torch.randn(B, T, N, F, requires_grad=True)
    edge_index = _chain_edge_index(N)

    pool = LearnableGraphPool(
        in_channels=F,
        pooling_ratio=0.5,
        selection_mode="ste",
        temperature=1.0,
        entropy_reg_weight=1e-3,
    )
    pool.train()

    x_pooled, _, _, _ = pool(x=x, edge_index=edge_index)
    aux_loss = pool.get_selector_aux_loss()
    assert aux_loss is not None

    loss = x_pooled.sum() + aux_loss
    loss.backward()

    assert x.grad is not None
    assert x.grad.abs().sum() > 0

    # Ensure selector parameters receive gradients under STE.
    has_nonzero_grad = False
    for p in pool.parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            has_nonzero_grad = True
            break
    assert has_nonzero_grad


def test_temperature_schedule_linear_step_based():
    pool = LearnableGraphPool(
        in_channels=8,
        pooling_ratio=0.5,
        selection_mode="ste",
        temperature=2.0,
        temperature_schedule="linear",
        temperature_min=0.5,
        temperature_anneal_steps=10,
        temperature_warmup_steps=2,
    )

    pool.set_training_step(0)
    assert abs(pool.get_current_temperature() - 2.0) < 1e-6

    pool.set_training_step(2)  # warmup boundary
    assert abs(pool.get_current_temperature() - 2.0) < 1e-6

    pool.set_training_step(7)  # halfway through anneal
    expected_mid = 2.0 + (5.0 / 10.0) * (0.5 - 2.0)
    assert abs(pool.get_current_temperature() - expected_mid) < 1e-6

    pool.set_training_step(30)  # after anneal
    assert abs(pool.get_current_temperature() - 0.5) < 1e-6


def test_entropy_regularization_diagnostics_are_exposed():
    torch.manual_seed(0)
    B, T, N, F = 2, 4, 10, 8
    x = torch.randn(B, T, N, F)
    edge_index = _chain_edge_index(N)

    pool = LearnableGraphPool(
        in_channels=F,
        pooling_ratio=0.5,
        selection_mode="ste",
        temperature=1.0,
        entropy_reg_weight=5e-4,
    )
    pool.train()

    _ = pool(x=x, edge_index=edge_index)
    aux_loss = pool.get_selector_aux_loss()
    diag = pool.get_selector_diagnostics()

    assert aux_loss is not None
    assert aux_loss.item() >= 0.0
    assert "entropy_reg" in diag
    assert "entropy_mean_norm" in diag
    assert "temperature" in diag
    assert "selected_node_counts_per_graph" in diag
    assert tuple(diag["selected_node_counts_per_graph"].shape) == (B, N)
    assert "scores_per_graph" in diag
    assert tuple(diag["scores_per_graph"].shape) == (B, N)
    assert "active_mask_per_graph" in diag
    assert tuple(diag["active_mask_per_graph"].shape) == (B, N)
    assert "soft_mask_per_graph" in diag
    assert tuple(diag["soft_mask_per_graph"].shape) == (B, N)


def test_heavy_selector_diagnostics_can_be_disabled():
    torch.manual_seed(0)
    B, T, N, F = 2, 4, 10, 8
    x = torch.randn(B, T, N, F)
    edge_index = _chain_edge_index(N)

    pool = LearnableGraphPool(
        in_channels=F,
        pooling_ratio=0.5,
        selection_mode="ste",
        temperature=1.0,
    )
    pool.train()

    pool.set_collect_heavy_diagnostics(False)
    _ = pool(x=x, edge_index=edge_index)
    diag = pool.get_selector_diagnostics()

    assert "temperature" in diag
    assert "entropy_reg" in diag
    assert "selected_node_counts_per_graph" not in diag
    assert "scores_per_graph" not in diag
    assert "active_mask_per_graph" not in diag
    assert "soft_mask_per_graph" not in diag

    pool.set_collect_heavy_diagnostics(True)
    _ = pool(x=x, edge_index=edge_index)
    diag_heavy = pool.get_selector_diagnostics()
    assert "selected_node_counts_per_graph" in diag_heavy
    assert "scores_per_graph" in diag_heavy
