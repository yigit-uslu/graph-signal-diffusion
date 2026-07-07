"""Verify that L1 (deepest-level) NodeSelector receives gradients from the
diffusion loss, not just from the entropy penalty.

This test catches the bug where the encoder discarded its final x_pooled
(post-selector) and fed skip_features[-1] (pre-selector) into the bottleneck,
severing the gradient path for the deepest selector.
"""

import torch
import pytest

from graph_signal_diffusion.models.ugnn import (
    UGNN,
    UGNNConfig,
    GNNConfig,
    PoolingConfig,
    EmbeddingConfig,
)


def _make_2level_ugnn(N: int = 32, selection_mode: str = "ste") -> UGNN:
    """Build a minimal 2-level UGNN with learned selectors at both levels."""
    config = UGNNConfig(
        in_channels=1,
        out_channels=1,
        base_channels=8,
        channel_multipliers=[1, 2],
        num_bottleneck_layers=1,
        gnn_config=GNNConfig(K=1, num_layers=1),
        pooling_config=PoolingConfig(
            gamma=2,
            pool_K=0,
            selection_method="learned",
            selector_version="v3",
            selector_kwargs={"selection_mode": selection_mode},
        ),
        embedding_config=EmbeddingConfig(
            time_embed_dim=16,
            num_timesteps=100,
        ),
    )
    return UGNN(config=config)


def _build_inputs(B: int, T: int, N: int, device: str = "cpu"):
    x = torch.randn(B, T, N, 1, device=device)
    timesteps = torch.randint(0, 100, (B,), device=device)
    # Simple cycle graph
    src = torch.arange(N, device=device)
    dst = (src + 1) % N
    edge_index = torch.stack([
        torch.cat([src, dst]),
        torch.cat([dst, src]),
    ])
    return x, timesteps, edge_index


def _collect_selector_params(ugnn: UGNN, exclude_cond: bool = True):
    """Return {level_idx: list_of_(name, param)} for each learned selector.

    When exclude_cond=True, skip conditioning-related parameters (cond_attention,
    cond_add_bias) that only participate when conditioning is provided.
    """
    cond_prefixes = ("cond_attention.", "cond_add_bias")
    selectors = {}
    for i, block in enumerate(ugnn.encoder.encoder_blocks):
        if hasattr(block.pool, "selector") and block.pool.selector is not None:
            params = [
                (name, p)
                for name, p in block.pool.selector.named_parameters()
                if p.requires_grad
                and not (exclude_cond and name.startswith(cond_prefixes))
            ]
            selectors[i] = params
    return selectors


@pytest.mark.parametrize("selection_mode", ["ste", "soft"])
def test_l1_selector_receives_gradient(selection_mode: str):
    """Both L0 and L1 selectors must get non-zero grad from a simple MSE loss."""
    N = 32
    ugnn = _make_2level_ugnn(N=N, selection_mode=selection_mode)
    ugnn.train()

    x, timesteps, edge_index = _build_inputs(B=2, T=4, N=N)

    out, _ = ugnn(x, timesteps, edge_index)
    loss = out.pow(2).mean()  # simple loss, no entropy penalty
    loss.backward()

    selectors = _collect_selector_params(ugnn)
    assert len(selectors) == 2, f"Expected 2 selectors (L0, L1), got {len(selectors)}"

    for level, params in selectors.items():
        assert len(params) > 0, f"Level {level}: no trainable selector params"
        for name, p in params:
            assert p.grad is not None, (
                f"Level {level} param '{name}': grad is None — "
                f"no gradient path from loss to this parameter"
            )
            grad_norm = p.grad.norm().item()
            assert grad_norm > 0, (
                f"Level {level} param '{name}': grad_norm = 0.0 — "
                f"gradient path exists but gradient is exactly zero"
            )
            print(f"  L{level} {name}: grad_norm = {grad_norm:.6e}")


@pytest.mark.parametrize("selection_mode", ["ste", "soft"])
def test_l1_gradient_magnitude_reasonable(selection_mode: str):
    """L1 selector gradient should be within ~100x of L0's (not orders of
    magnitude smaller, which would suggest a near-break)."""
    N = 32
    ugnn = _make_2level_ugnn(N=N, selection_mode=selection_mode)
    ugnn.train()

    x, timesteps, edge_index = _build_inputs(B=2, T=4, N=N)

    out, _ = ugnn(x, timesteps, edge_index)
    loss = out.pow(2).mean()
    loss.backward()

    selectors = _collect_selector_params(ugnn)

    def _total_grad_norm(params):
        grads = [p.grad.flatten() for _, p in params if p.grad is not None]
        if not grads:
            return 0.0
        return torch.cat(grads).norm().item()

    l0_norm = _total_grad_norm(selectors[0])
    l1_norm = _total_grad_norm(selectors[1])

    print(f"  L0 total grad norm: {l0_norm:.6e}")
    print(f"  L1 total grad norm: {l1_norm:.6e}")

    assert l0_norm > 0, "L0 grad norm should be positive"
    assert l1_norm > 0, "L1 grad norm should be positive"

    ratio = l0_norm / l1_norm if l1_norm > 0 else float("inf")
    print(f"  L0/L1 ratio: {ratio:.2f}")
    assert ratio < 100, (
        f"L0/L1 grad ratio = {ratio:.1f} — L1 gradient is suspiciously small"
    )
