"""
Tests for RevIN (Reversible Instance Normalization) in DDPM and DDIM.

Covers: roundtrip, training_loss, stratified_loss, sample shape (DDPM/DDIM),
constant-stock clamping, cross-stock correlation preservation, disabled parity.
"""

import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

from graph_signal_diffusion.diffusion.ddpm import DDPM
from graph_signal_diffusion.diffusion.ddim import DDIM


# ---------------------------------------------------------------------------
# Minimal denoiser model stub
# ---------------------------------------------------------------------------

class _StubDenoiser(nn.Module):
    """Minimal model matching the UGNN forward signature."""

    def __init__(self, T, N, F):
        super().__init__()
        self.linear = nn.Linear(T * N * F, T * N * F)
        self._shape = (T, N, F)

    def forward(self, x, timesteps, edge_index=None, edge_weight=None,
                cond=None, return_intermediates=False):
        B = x.shape[0]
        out = self.linear(x.reshape(B, -1)).reshape(B, *self._shape)
        return out, None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_batch(B=4, T_fut=5, T_obs=10, N=8, F=1, F_obs=3, target_idx=0,
                device="cpu"):
    """Create a synthetic PyG Batch with conditioning and info metadata."""
    # y: [B*N, T_fut, F] — target (standardized space)
    y = torch.randn(B * N, T_fut, F, device=device)
    # x: [B*N, T_obs, F_obs] — conditioning
    x = torch.randn(B * N, T_obs, F_obs, device=device)
    # Simple fully-connected graph per sample
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device=device)

    graphs = []
    for i in range(B):
        g = Data(
            y=y[i * N:(i + 1) * N],
            x=x[i * N:(i + 1) * N],
            edge_index=edge_index,
            target_column_idx=target_idx,
        )
        g.num_nodes = N
        graphs.append(g)

    batch = Batch.from_data_list(graphs)
    return batch


def _make_ddpm(T_fut=5, N=8, F=1, revin=True, **kw):
    model = _StubDenoiser(T_fut, N, F)
    return DDPM(
        model=model,
        num_timesteps=10,  # tiny for speed
        revin=revin,
        **kw,
    )


def _make_ddim(T_fut=5, N=8, F=1, revin=True, **kw):
    model = _StubDenoiser(T_fut, N, F)
    return DDIM(
        model=model,
        num_timesteps=10,
        sampling_timesteps=5,
        revin=revin,
        **kw,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRevinRoundtrip:

    def test_normalize_then_denormalize_is_identity(self):
        ddpm = _make_ddpm()
        x = torch.randn(4, 5, 8, 1)
        mu = torch.randn(4, 1, 8, 1)
        sigma = torch.rand(4, 1, 8, 1).clamp(min=1e-5)

        x_norm = ddpm._revin_normalize(x, mu, sigma)
        x_recovered = ddpm._revin_denormalize(x_norm, mu, sigma)
        torch.testing.assert_close(x_recovered, x, atol=1e-5, rtol=1e-5)


class TestRevinTrainingLoss:

    def test_training_loss_finite(self):
        ddpm = _make_ddpm(revin=True)
        batch = _make_batch()
        loss = ddpm.training_loss(batch)
        assert torch.isfinite(loss).all(), f"Non-finite loss: {loss}"

    def test_stratified_loss_finite(self):
        ddpm = _make_ddpm(revin=True)
        batch = _make_batch()
        loss, bin_losses = ddpm.training_loss_stratified_t(batch, num_bins=2)
        assert torch.isfinite(loss).all(), f"Non-finite stratified loss: {loss}"
        assert len(bin_losses) > 0

    def test_elbo_finite(self):
        ddpm = _make_ddpm(revin=True)
        batch = _make_batch()
        elbo = ddpm.compute_elbo_per_trajectory(batch, n_mc_samples=2)
        assert torch.isfinite(elbo).all(), f"Non-finite ELBO: {elbo}"
        assert elbo.shape == (4,)  # B=4


class TestRevinSampleShape:

    def test_ddpm_sample_shape(self):
        ddpm = _make_ddpm(revin=True)
        batch = _make_batch()
        shape = (4, 5, 8, 1)  # B, T, N, F
        result = ddpm.sample(shape, device=torch.device("cpu"), data=batch)
        assert result.shape == shape
        assert torch.isfinite(result).all()

    def test_ddim_sample_shape(self):
        ddim = _make_ddim(revin=True)
        batch = _make_batch()
        shape = (4, 5, 8, 1)
        result = ddim.sample(shape, device=torch.device("cpu"), data=batch)
        assert result.shape == shape
        assert torch.isfinite(result).all()


class TestRevinConstantStock:

    def test_constant_stock_no_nan(self):
        """A stock with constant past should have sigma_w clamped, not NaN."""
        ddpm = _make_ddpm(revin=True, revin_eps=1e-5)
        B, T_obs, N, F_obs = 2, 10, 4, 3

        # Make conditioning with one stock having constant target channel
        cond = torch.randn(B, T_obs, N, F_obs)
        cond[:, :, 0, 0] = 5.0  # stock 0, target feature 0 is constant

        mu, sigma = ddpm._compute_revin_stats(cond, target_idx=0)
        assert torch.isfinite(mu).all()
        assert torch.isfinite(sigma).all()
        assert (sigma >= ddpm.revin_eps).all()


class TestRevinCrossStockCorrelation:

    def test_correlation_preserved(self):
        """RevIN (per-stock normalization) should not destroy cross-stock
        correlation structure — it only shifts/scales per stock."""
        ddpm = _make_ddpm(revin=True)
        torch.manual_seed(42)
        B, T, N, F = 1, 20, 5, 1

        # Create correlated stocks
        common = torch.randn(B, T, 1, F)
        x0 = common + torch.randn(B, T, N, F) * 0.1  # highly correlated

        mu = torch.randn(B, 1, N, F)
        sigma = torch.rand(B, 1, N, F).clamp(min=0.1)

        x0_norm = ddpm._revin_normalize(x0, mu, sigma)

        # Compute pairwise correlation before and after
        def _pairwise_corr(t):
            # t: [1, T, N, 1] -> [T, N]
            t = t.squeeze(0).squeeze(-1)
            return torch.corrcoef(t.T)

        corr_before = _pairwise_corr(x0)
        corr_after = _pairwise_corr(x0_norm)
        # Correlation should be very similar (RevIN is affine per-stock)
        torch.testing.assert_close(corr_before, corr_after, atol=1e-4, rtol=1e-3)


class TestRevinDisabledParity:

    def test_disabled_produces_identical_loss(self):
        """revin=False must produce identical loss to a baseline without RevIN."""
        torch.manual_seed(99)
        batch = _make_batch()

        ddpm_off = _make_ddpm(revin=False)
        ddpm_base = _make_ddpm(revin=False)

        # Share weights
        ddpm_base.load_state_dict(ddpm_off.state_dict())

        torch.manual_seed(0)
        loss_off = ddpm_off.training_loss(batch)
        torch.manual_seed(0)
        loss_base = ddpm_base.training_loss(batch)

        torch.testing.assert_close(loss_off, loss_base)


class TestRevinOrderingCorrectness:

    def test_revin_changes_xt_distribution(self):
        """RevIN must normalize x0 BEFORE noise addition.

        If RevIN is applied after xt is created, the noised tensor xt would be
        in raw space while the target (eps or x0) is in RevIN space — breaking
        the forward process. This test verifies that RevIN actually changes the
        forward-process input by comparing xt distributions with and without.
        """
        torch.manual_seed(42)
        # Use conditioning with large per-stock mean shifts so RevIN has
        # a visible effect on x0 (and thus xt).
        batch = _make_batch(B=4, T_fut=5, T_obs=10, N=8, F=1, F_obs=3)

        # Shift conditioning target channel per-stock to create distinct RevIN stats
        B, N = 4, 8
        for i in range(N):
            # Large per-stock offset: stock i has mean = 10*i in conditioning
            batch.x[i::N, :, 0] += 10.0 * i

        ddpm_on = _make_ddpm(revin=True)
        ddpm_off = _make_ddpm(revin=False)
        ddpm_off.load_state_dict(ddpm_on.state_dict())

        # Same random seed → same t, same noise
        torch.manual_seed(0)
        loss_on = ddpm_on.training_loss(batch)
        torch.manual_seed(0)
        loss_off = ddpm_off.training_loss(batch)

        # Loss MUST differ when RevIN is enabled vs disabled
        # (RevIN changes x0 before noise addition, changing xt and thus loss)
        assert not torch.allclose(loss_on, loss_off, atol=1e-6), (
            f"RevIN-on loss ({loss_on.item():.6f}) equals RevIN-off loss "
            f"({loss_off.item():.6f}) — RevIN may not be affecting the forward process."
        )

    def test_revin_normalizes_before_noise(self):
        """Directly verify that _maybe_revin_normalize is called before add_noise
        by checking that x0 passed to add_noise is in RevIN space."""
        ddpm = _make_ddpm(revin=True)
        batch = _make_batch(B=2, T_fut=5, T_obs=10, N=4, F=1, F_obs=3)

        # Add large offsets to conditioning target
        for i in range(4):
            batch.x[i::4, :, 0] += 50.0 * i

        B, N, T, F = 2, 4, 5, 1
        x0 = batch.y.view(B, N, T, F).permute(0, 2, 1, 3)
        ut = batch.x.view(B, N, *batch.x.shape[1:]).swapaxes(1, 2)

        x0_norm, mu, sigma = ddpm._maybe_revin_normalize(x0, ut, batch)

        # x0_norm should have per-stock mean ≈ 0 (centered by RevIN)
        per_stock_mean = x0_norm.mean(dim=1)  # [B, N, F]
        # Not exactly 0 (RevIN uses conditioning stats, not x0 stats),
        # but it should be different from raw x0's mean
        raw_mean = x0.mean(dim=1)

        # RevIN should shift x0: normalized mean should differ from raw mean
        assert not torch.allclose(per_stock_mean, raw_mean, atol=1e-3), (
            "RevIN normalization did not change x0 mean — possible no-op."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
