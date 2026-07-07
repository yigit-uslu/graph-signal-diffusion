"""
Tests that RevIN fails fast when conditioning is missing.

Covers: training_loss, training_loss_stratified_t, compute_elbo_per_trajectory,
DDPM.sample, DDIM.sample — all must raise ValueError when revin=True and
conditioning (data.x) is None.
"""

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

from graph_signal_diffusion.diffusion.ddpm import DDPM
from graph_signal_diffusion.diffusion.ddim import DDIM


class _Stub(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x, timesteps, edge_index=None, edge_weight=None,
                cond=None, return_intermediates=False):
        return x, None


def _make_batch_no_x(B=2, T_fut=5, N=4, F=1):
    """Batch with y but no x (conditioning)."""
    graphs = []
    for _ in range(B):
        g = Data(
            y=torch.randn(N, T_fut, F),
            edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        )
        g.num_nodes = N
        # Explicitly set x to None
        g.x = None
        graphs.append(g)
    return Batch.from_data_list(graphs)


def _make_ddpm_revin():
    return DDPM(model=_Stub(), num_timesteps=10, revin=True)


def _make_ddim_revin():
    return DDIM(model=_Stub(), num_timesteps=10, sampling_timesteps=5, revin=True)


class TestTrainingLossRequiresConditioning:

    def test_training_loss_raises(self):
        ddpm = _make_ddpm_revin()
        batch = _make_batch_no_x()
        with pytest.raises(ValueError, match="RevIN requires conditioning"):
            ddpm.training_loss(batch)

    def test_stratified_loss_raises(self):
        ddpm = _make_ddpm_revin()
        batch = _make_batch_no_x()
        with pytest.raises(ValueError, match="RevIN requires conditioning"):
            ddpm.training_loss_stratified_t(batch, num_bins=2)

    def test_elbo_raises(self):
        ddpm = _make_ddpm_revin()
        batch = _make_batch_no_x()
        with pytest.raises(ValueError, match="RevIN requires conditioning"):
            ddpm.compute_elbo_per_trajectory(batch, n_mc_samples=1)


class TestSampleRequiresConditioning:

    def test_ddpm_sample_raises(self):
        ddpm = _make_ddpm_revin()
        batch = _make_batch_no_x()
        with pytest.raises(ValueError, match="RevIN requires conditioning"):
            ddpm.sample((2, 5, 4, 1), device=torch.device("cpu"), data=batch)

    def test_ddim_sample_raises(self):
        ddim = _make_ddim_revin()
        batch = _make_batch_no_x()
        with pytest.raises(ValueError, match="RevIN requires conditioning"):
            ddim.sample((2, 5, 4, 1), device=torch.device("cpu"), data=batch)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
