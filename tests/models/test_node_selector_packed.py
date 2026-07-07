"""Tests for NodeSelector packed score computation parity."""

import pytest
import torch

from graph_signal_diffusion.models.components.node_selector import NodeSelector


B, T, N, C = 2, 4, 20, 16
TIME_DIM = 8


def _make_inputs(cond_fusion_mode: str, active_ratio: float = 0.3):
    """Create test inputs with a given active ratio."""
    num_active = max(1, int(N * active_ratio))
    active_mask = torch.zeros(B, N, dtype=torch.bool)
    active_mask[:, :num_active] = True

    x = torch.randn(B, T, N, C)
    # Zero out inactive positions (as forward() does)
    x = x * active_mask.unsqueeze(1).unsqueeze(-1).float()

    time_emb = torch.randn(B, TIME_DIM)
    cond = torch.randn(B, T, N, C) if cond_fusion_mode != 'none' else None
    edge_index = torch.zeros(2, 0, dtype=torch.long)
    return x, edge_index, active_mask, cond, time_emb


@pytest.mark.parametrize("cond_fusion_mode", ['none', 'film', 'add', 'cross_attention'])
@pytest.mark.parametrize("temporal_reduce", ['mean', 'last', 'attn'])
def test_packed_dense_score_parity(cond_fusion_mode, temporal_reduce):
    """Packed and dense paths must produce identical scores at active positions."""
    sel = NodeSelector(
        in_channels=C,
        cond_fusion_mode=cond_fusion_mode,
        temporal_reduce=temporal_reduce,
        time_emb_dim=TIME_DIM,
        packed_score_mode='off',  # start with dense
    )
    sel.eval()

    x, edge_index, active_mask, cond, time_emb = _make_inputs(cond_fusion_mode)

    with torch.no_grad():
        scores_dense = sel.compute_scores(
            x, edge_index, active_mask=active_mask, cond=cond, time_emb=time_emb,
        )

    # Switch to packed
    sel.packed_score_mode = 'auto'
    sel.packed_score_threshold = 1.0  # force packed path (30% < 100%)

    with torch.no_grad():
        scores_packed = sel.compute_scores(
            x, edge_index, active_mask=active_mask, cond=cond, time_emb=time_emb,
        )

    assert torch.allclose(
        scores_dense[active_mask], scores_packed[active_mask], atol=1e-5,
    ), f"Active scores differ: max delta = {(scores_dense[active_mask] - scores_packed[active_mask]).abs().max()}"
    assert torch.allclose(
        scores_dense[~active_mask], scores_packed[~active_mask], atol=1e-5,
    ), "Inactive scores should both be min/4"


@pytest.mark.parametrize("cond_fusion_mode", ['none', 'film', 'add'])
def test_packed_forward_parity(cond_fusion_mode):
    """Full forward() outputs must match between packed and dense paths."""
    sel = NodeSelector(
        in_channels=C,
        cond_fusion_mode=cond_fusion_mode,
        temporal_reduce='mean',
        time_emb_dim=TIME_DIM,
        packed_score_mode='off',
    )
    sel.eval()

    x, edge_index, active_mask, cond, time_emb = _make_inputs(cond_fusion_mode)

    with torch.no_grad():
        x_pooled_d, mask_d, idx_d, scores_d = sel(
            x, edge_index, active_mask=active_mask, cond=cond, time_emb=time_emb,
        )

    sel.packed_score_mode = 'auto'
    sel.packed_score_threshold = 1.0

    with torch.no_grad():
        x_pooled_p, mask_p, idx_p, scores_p = sel(
            x, edge_index, active_mask=active_mask, cond=cond, time_emb=time_emb,
        )

    assert torch.allclose(scores_d[active_mask], scores_p[active_mask], atol=1e-5)
    assert torch.equal(mask_d, mask_p), "new_active_mask must be identical"
    assert torch.equal(idx_d, idx_p), "top_k_indices must be identical"
    assert torch.allclose(x_pooled_d, x_pooled_p, atol=1e-5)


def test_packed_not_triggered_above_threshold():
    """Packed path should NOT trigger when active_ratio >= threshold."""
    sel = NodeSelector(
        in_channels=C,
        packed_score_mode='auto',
        packed_score_threshold=0.2,  # threshold = 20%
    )
    sel.eval()

    # 50% active → above threshold → should use dense path
    active_mask = torch.zeros(B, N, dtype=torch.bool)
    active_mask[:, : N // 2] = True
    x = torch.randn(B, T, N, C) * active_mask.unsqueeze(1).unsqueeze(-1).float()
    edge_index = torch.zeros(2, 0, dtype=torch.long)

    # Verify it runs without error (dense path)
    with torch.no_grad():
        scores = sel.compute_scores(x, edge_index, active_mask=active_mask)
    assert scores.shape == (B, N)


def test_packed_all_inactive():
    """Edge case: no active nodes should return min/4 everywhere."""
    sel = NodeSelector(
        in_channels=C,
        packed_score_mode='auto',
        packed_score_threshold=1.0,
    )
    sel.eval()

    active_mask = torch.zeros(B, N, dtype=torch.bool)
    x = torch.zeros(B, T, N, C)
    edge_index = torch.zeros(2, 0, dtype=torch.long)

    with torch.no_grad():
        scores = sel.compute_scores(x, edge_index, active_mask=active_mask)
    # All scores should be min/4 (masked_fill)
    expected = torch.finfo(scores.dtype).min / 4
    assert torch.allclose(scores, torch.full_like(scores, expected))


def test_packed_gradient_flow():
    """Verify gradients flow back through the packed path."""
    sel = NodeSelector(
        in_channels=C,
        cond_fusion_mode='film',
        time_emb_dim=TIME_DIM,
        packed_score_mode='auto',
        packed_score_threshold=1.0,
    )
    sel.train()

    x, edge_index, active_mask, cond, time_emb = _make_inputs('film')
    x = x.requires_grad_(True)

    x_pooled, _, _, scores = sel(
        x, edge_index, active_mask=active_mask, cond=cond, time_emb=time_emb,
    )
    loss = x_pooled.sum() + scores[active_mask].sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    # Gradients at inactive positions should be zero
    inactive_grad = x.grad * (~active_mask).unsqueeze(1).unsqueeze(-1).float()
    assert inactive_grad.abs().max() == 0.0, "Inactive nodes should have zero gradient"


def test_packed_add_path_does_not_materialize_dense_cond_tokens():
    """Packed add mode should not depend on _prepare_cond_tokens dense expansion."""
    sel = NodeSelector(
        in_channels=C,
        cond_fusion_mode='add',
        temporal_reduce='mean',
        time_emb_dim=TIME_DIM,
        packed_score_mode='off',
    )
    sel.eval()

    x, edge_index, active_mask, cond, time_emb = _make_inputs('add')

    with torch.no_grad():
        scores_dense = sel.compute_scores(
            x, edge_index, active_mask=active_mask, cond=cond, time_emb=time_emb,
        )

    sel.packed_score_mode = 'auto'
    sel.packed_score_threshold = 1.0  # force packed

    def _boom(*args, **kwargs):
        raise AssertionError("_prepare_cond_tokens should not be called in packed add path")

    sel._prepare_cond_tokens = _boom  # type: ignore[method-assign]

    with torch.no_grad():
        scores_packed = sel.compute_scores(
            x, edge_index, active_mask=active_mask, cond=cond, time_emb=time_emb,
        )

    torch.testing.assert_close(scores_packed, scores_dense, atol=1e-5, rtol=1e-5)
