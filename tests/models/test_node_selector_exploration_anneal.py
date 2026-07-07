"""Unit + smoke tests for NodeSelector Gumbel exploration_noise annealing."""

import math

import pytest
import torch

from graph_signal_diffusion.models.components.node_selector import NodeSelector


def _chain_edge_index(num_nodes: int) -> torch.Tensor:
    edges = [(i, i + 1) for i in range(num_nodes - 1)]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return torch.cat([edge_index, edge_index.flip(0)], dim=1)


def _make_selector(**kwargs) -> NodeSelector:
    defaults = dict(
        in_channels=8,
        pooling_ratio=0.5,
        selection_mode='ste',
        cond_fusion=False,
    )
    defaults.update(kwargs)
    return NodeSelector(**defaults)


# --- _current_exploration_noise() schedule shapes -------------------------


def test_constant_schedule_returns_initial_value_regardless_of_step():
    sel = _make_selector(
        exploration_noise=0.7,
        exploration_noise_min=0.05,
        exploration_noise_schedule='constant',
        exploration_noise_anneal_steps=100,
    )
    for step in (0, 1, 50, 99, 100, 1000):
        sel.set_training_step(step)
        assert sel.get_current_exploration_noise() == pytest.approx(0.7)


def test_anneal_steps_zero_disables_anneal():
    sel = _make_selector(
        exploration_noise=0.7,
        exploration_noise_min=0.05,
        exploration_noise_schedule='linear',
        exploration_noise_anneal_steps=0,
    )
    for step in (0, 10, 1000):
        sel.set_training_step(step)
        assert sel.get_current_exploration_noise() == pytest.approx(0.7)


def test_linear_anneal_interpolates_then_clamps_to_min():
    initial, floor = 1.0, 0.05
    anneal_steps = 100
    sel = _make_selector(
        exploration_noise=initial,
        exploration_noise_min=floor,
        exploration_noise_schedule='linear',
        exploration_noise_anneal_steps=anneal_steps,
    )
    # Start of anneal
    sel.set_training_step(0)
    assert sel.get_current_exploration_noise() == pytest.approx(initial)
    # Midpoint
    sel.set_training_step(50)
    expected = initial + 0.5 * (floor - initial)
    assert sel.get_current_exploration_noise() == pytest.approx(expected)
    # 90% through
    sel.set_training_step(90)
    expected = initial + 0.9 * (floor - initial)
    assert sel.get_current_exploration_noise() == pytest.approx(expected)
    # End of anneal
    sel.set_training_step(anneal_steps)
    assert sel.get_current_exploration_noise() == pytest.approx(floor)
    # Past end stays at floor
    sel.set_training_step(anneal_steps * 5)
    assert sel.get_current_exploration_noise() == pytest.approx(floor)


def test_linear_anneal_respects_warmup_steps():
    initial, floor = 1.0, 0.05
    warmup = 20
    anneal_steps = 100
    sel = _make_selector(
        exploration_noise=initial,
        exploration_noise_min=floor,
        exploration_noise_schedule='linear',
        exploration_noise_warmup_steps=warmup,
        exploration_noise_anneal_steps=anneal_steps,
    )
    # During warmup: holds at initial
    for step in (0, 5, 19):
        sel.set_training_step(step)
        assert sel.get_current_exploration_noise() == pytest.approx(initial)
    # First step of anneal (progress 0)
    sel.set_training_step(warmup)
    assert sel.get_current_exploration_noise() == pytest.approx(initial)
    # Midpoint after warmup
    sel.set_training_step(warmup + 50)
    expected = initial + 0.5 * (floor - initial)
    assert sel.get_current_exploration_noise() == pytest.approx(expected)
    # End
    sel.set_training_step(warmup + anneal_steps)
    assert sel.get_current_exploration_noise() == pytest.approx(floor)


def test_cosine_anneal_monotone_and_hits_endpoints():
    initial, floor = 1.0, 0.05
    anneal_steps = 100
    sel = _make_selector(
        exploration_noise=initial,
        exploration_noise_min=floor,
        exploration_noise_schedule='cosine',
        exploration_noise_anneal_steps=anneal_steps,
    )
    sel.set_training_step(0)
    assert sel.get_current_exploration_noise() == pytest.approx(initial)
    sel.set_training_step(anneal_steps)
    assert sel.get_current_exploration_noise() == pytest.approx(floor)
    # Midpoint cosine = (initial + floor) / 2
    sel.set_training_step(anneal_steps // 2)
    expected = floor + 0.5 * (initial - floor) * (1.0 + math.cos(math.pi * 0.5))
    assert sel.get_current_exploration_noise() == pytest.approx(expected)

    # Monotone non-increasing across the anneal window
    prev = math.inf
    for step in range(0, anneal_steps + 1, 5):
        sel.set_training_step(step)
        cur = sel.get_current_exploration_noise()
        assert cur <= prev + 1e-9
        prev = cur


def test_invalid_schedule_raises():
    with pytest.raises(ValueError, match="exploration_noise_schedule"):
        _make_selector(exploration_noise_schedule='exponential')


def test_negative_min_raises():
    with pytest.raises(ValueError, match="exploration_noise_min"):
        _make_selector(exploration_noise_min=-0.1)


def test_negative_anneal_steps_raises():
    with pytest.raises(ValueError, match="exploration_noise_anneal_steps"):
        _make_selector(exploration_noise_anneal_steps=-5)


# --- Forward pass smoke tests ---------------------------------------------


def test_forward_with_gumbel_on_runs_and_uses_annealed_value():
    torch.manual_seed(0)
    B, T, N, F = 2, 4, 12, 8
    sel = _make_selector(
        in_channels=F,
        exploration_noise=1.0,
        exploration_noise_min=0.05,
        exploration_noise_schedule='linear',
        exploration_noise_anneal_steps=10,
    )
    sel.train()
    x = torch.randn(B, T, N, F)
    edge_index = _chain_edge_index(N)
    active_mask = torch.ones(B, N, dtype=torch.bool)

    sel.set_training_step(0)
    out_pooled, new_active, _idx, scores = sel(x, edge_index, active_mask=active_mask)
    diag_start = sel.get_selector_diagnostics()
    assert out_pooled.shape == x.shape
    assert new_active.shape == (B, N)
    assert scores.shape == (B, N)
    assert diag_start["exploration_noise"] == pytest.approx(1.0)

    sel.set_training_step(10)
    sel(x, edge_index, active_mask=active_mask)
    diag_end = sel.get_selector_diagnostics()
    assert diag_end["exploration_noise"] == pytest.approx(0.05)


def test_forward_with_exploration_off_is_deterministic_when_seeded():
    """exploration_noise=0 → no Gumbel injection → forward is deterministic."""
    B, T, N, F = 2, 4, 12, 8
    edge_index = _chain_edge_index(N)
    active_mask = torch.ones(B, N, dtype=torch.bool)
    torch.manual_seed(123)
    x = torch.randn(B, T, N, F)

    def run_once():
        torch.manual_seed(7)
        sel = _make_selector(in_channels=F, exploration_noise=0.0)
        sel.train()
        _, new_active, idx, _ = sel(x, edge_index, active_mask=active_mask)
        return new_active.clone(), idx.clone()

    mask_a, idx_a = run_once()
    mask_b, idx_b = run_once()
    assert torch.equal(mask_a, mask_b)
    assert torch.equal(idx_a, idx_b)


def test_forward_with_gumbel_on_produces_different_selections_across_seeds():
    """exploration_noise > 0 → Gumbel injection → selection varies with RNG."""
    B, T, N, F = 2, 4, 32, 8
    edge_index = _chain_edge_index(N)
    active_mask = torch.ones(B, N, dtype=torch.bool)

    torch.manual_seed(0)
    x = torch.randn(B, T, N, F)

    def run_once(seed: int):
        torch.manual_seed(seed)
        sel = NodeSelector(
            in_channels=F,
            pooling_ratio=0.25,  # K=8 of 32 → leaves room for divergence
            selection_mode='ste',
            cond_fusion=False,
            exploration_noise=1.0,
        )
        # Re-init score head with a fixed seed so only Gumbel noise differs.
        torch.manual_seed(99)
        torch.nn.init.normal_(sel.score_proj.weight)
        torch.nn.init.zeros_(sel.score_proj.bias)
        sel.train()
        torch.manual_seed(seed)
        _, new_active, _, _ = sel(x, edge_index, active_mask=active_mask)
        return new_active.clone()

    mask_a = run_once(1)
    mask_b = run_once(2)
    # At least one batch row should disagree under different Gumbel draws.
    assert not torch.equal(mask_a, mask_b)


def test_forward_eval_mode_disables_gumbel_even_when_noise_positive():
    """sel.eval() → Gumbel branch is skipped (training-only guard)."""
    B, T, N, F = 2, 4, 16, 8
    edge_index = _chain_edge_index(N)
    active_mask = torch.ones(B, N, dtype=torch.bool)
    torch.manual_seed(0)
    x = torch.randn(B, T, N, F)

    sel = NodeSelector(
        in_channels=F,
        pooling_ratio=0.5,
        selection_mode='ste',
        cond_fusion=False,
        exploration_noise=1.0,
    )
    sel.eval()
    torch.manual_seed(1)
    _, mask_a, _, _ = sel(x, edge_index, active_mask=active_mask)
    torch.manual_seed(2)
    _, mask_b, _, _ = sel(x, edge_index, active_mask=active_mask)
    assert torch.equal(mask_a, mask_b)
