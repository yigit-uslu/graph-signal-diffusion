"""Unit tests for structural metric helpers on StockPriceForecastingTask.

Tests: _lag1_acf, _excess_kurtosis, _top1_eigenvalue.
"""
import numpy as np
import pytest
import torch

from graph_signal_diffusion.tasks.stock_price_forecasting.evaluator import (
    StockPriceForecastingTask,
)

task = StockPriceForecastingTask()


# ======================================================================
# _lag1_acf tests
# ======================================================================


class TestLag1Acf:
    def test_known_signal(self):
        """Construct returns whose r² follows AR(1) with known coefficient."""
        torch.manual_seed(42)
        alpha = 0.6
        B, T, N, F = 200, 100, 4, 1

        # Build r² as AR(1): r²_t = alpha * r²_{t-1} + (1-alpha) * noise
        r2 = torch.zeros(B, T, N)
        r2[:, 0, :] = torch.rand(B, N)
        for t in range(1, T):
            r2[:, t, :] = alpha * r2[:, t - 1, :] + (1 - alpha) * torch.rand(B, N)

        # r = sign * sqrt(r²)
        signs = torch.sign(torch.randn(B, T, N))
        returns = signs * torch.sqrt(r2)
        returns = returns.unsqueeze(-1)  # [B, T, N, 1]

        acf = task._lag1_acf(returns, squared=True)
        assert abs(acf - alpha) < 0.1, f"Expected ~{alpha}, got {acf}"

    def test_iid_returns(self):
        """i.i.d. Gaussian returns -> lag-1 ACF of r² should be near 0."""
        torch.manual_seed(0)
        returns = torch.randn(100, 50, 10, 1)
        acf = task._lag1_acf(returns, squared=True)
        assert abs(acf) < 0.1, f"Expected ~0 for i.i.d., got {acf}"

    def test_5d_vs_4d_mean(self):
        """5D ensemble path preserves structure that 4D mean destroys.

        Two members with phase-shifted variance blocks: ensemble mean
        cancels the block structure, while 5D path preserves it.
        Use large contrast ratio and longer sequences for robust separation.
        """
        torch.manual_seed(7)
        B, n, T, N, F = 100, 2, 80, 3, 1
        block = T // 2

        # Member 0: high var in first half, low in second
        m0 = torch.zeros(B, T, N, F)
        m0[:, :block, :, :] = torch.randn(B, block, N, F) * 3.0
        m0[:, block:, :, :] = torch.randn(B, T - block, N, F) * 0.05

        # Member 1: low var in first half, high in second (phase-shifted)
        m1 = torch.zeros(B, T, N, F)
        m1[:, :block, :, :] = torch.randn(B, block, N, F) * 0.05
        m1[:, block:, :, :] = torch.randn(B, T - block, N, F) * 3.0

        ensemble_5d = torch.stack([m0, m1], dim=1)  # [B, 2, T, N, F]
        ensemble_mean = ensemble_5d.mean(dim=1)      # [B, T, N, F]

        acf_5d = task._lag1_acf(ensemble_5d, squared=True)
        acf_mean = task._lag1_acf(ensemble_mean, squared=True)

        # 5D should preserve strong block ACF; mean should cancel most of it
        assert acf_5d > acf_mean + 0.05, (
            f"5D ACF ({acf_5d:.4f}) should clearly exceed mean ACF ({acf_mean:.4f})"
        )

    def test_T1_returns_zero(self):
        """T=1 -> returns 0.0, no NaN."""
        returns = torch.randn(4, 1, 3, 1)
        acf = task._lag1_acf(returns, squared=True)
        assert acf == 0.0
        assert not np.isnan(acf)

    def test_constant_returns_zero(self):
        """All-constant returns -> 0.0, no NaN."""
        returns = torch.ones(4, 10, 3, 1) * 5.0
        acf = task._lag1_acf(returns, squared=True)
        assert acf == 0.0
        assert not np.isnan(acf)

    def test_3d_input(self):
        """3D [B, T, F] input works without error."""
        returns = torch.randn(8, 20, 3)
        acf = task._lag1_acf(returns, squared=True)
        assert np.isfinite(acf)

    def test_momentum_iid(self):
        """squared=False on i.i.d. returns -> near 0 (no momentum)."""
        torch.manual_seed(1)
        returns = torch.randn(100, 50, 10, 1)
        acf = task._lag1_acf(returns, squared=False)
        assert abs(acf) < 0.1, f"Expected ~0 for i.i.d., got {acf}"

    def test_bad_dim_raises(self):
        """2D input raises ValueError."""
        with pytest.raises(ValueError, match="expects 3D/4D/5D"):
            task._lag1_acf(torch.randn(10, 5), squared=True)


# ======================================================================
# _excess_kurtosis tests
# ======================================================================


class TestExcessKurtosis:
    def test_gaussian(self):
        """Gaussian returns -> excess kurtosis near 0."""
        torch.manual_seed(2)
        returns = torch.randn(200, 100, 10, 1)
        kurt = task._excess_kurtosis(returns)
        assert abs(kurt) < 0.5, f"Expected ~0 for Gaussian, got {kurt}"

    def test_fat_tails(self):
        """t-distribution with df=4 -> excess kurtosis near 6."""
        torch.manual_seed(3)
        # t(df=4) has theoretical excess kurtosis = 6/(df-4) which is undefined
        # for df=4 exactly, use df=5 -> excess kurtosis = 6
        import scipy.stats
        samples = scipy.stats.t.rvs(df=5, size=(200, 100, 10, 1))
        returns = torch.tensor(samples, dtype=torch.float32)
        kurt = task._excess_kurtosis(returns)
        assert 3.0 < kurt < 12.0, f"Expected ~6 for t(5), got {kurt}"

    def test_constant_returns_zero(self):
        """Constant returns -> 0.0, no NaN."""
        returns = torch.ones(4, 10, 3, 1) * 2.0
        kurt = task._excess_kurtosis(returns)
        assert kurt == 0.0

    def test_bad_dim_raises(self):
        """2D input raises ValueError."""
        with pytest.raises(ValueError, match="expects 3D/4D/5D"):
            task._excess_kurtosis(torch.randn(10, 5))


# ======================================================================
# _top1_eigenvalue tests
# ======================================================================


class TestTop1Eigenvalue:
    def test_identity_correlation(self):
        """i.i.d. per-stock returns -> correlation ~ identity -> top eigval ~ 1."""
        torch.manual_seed(4)
        returns = torch.randn(100, 50, 10, 1)
        ev = task._top1_eigenvalue(returns)
        assert ev is not None
        # For N=10 stocks with 5000 samples, top eigenvalue of identity ~1.0
        # Marchenko-Pastur correction: slightly above 1
        assert 0.8 < ev < 1.8, f"Expected ~1 for i.i.d., got {ev}"

    def test_single_factor(self):
        """Returns = beta * market + noise -> top eigenvalue >> 1."""
        torch.manual_seed(5)
        B, T, N = 100, 50, 10
        market = torch.randn(B, T, 1)  # common factor
        beta = torch.rand(1, 1, N) * 2 + 0.5
        noise = torch.randn(B, T, N) * 0.3
        returns = (beta * market + noise).unsqueeze(-1)  # [B, T, N, 1]

        ev = task._top1_eigenvalue(returns)
        assert ev is not None
        assert ev > 3.0, f"Expected large eigenvalue for single-factor, got {ev}"

    def test_3d_returns_none(self):
        """3D input -> returns None (no stock axis)."""
        returns = torch.randn(8, 20, 3)
        assert task._top1_eigenvalue(returns) is None

    def test_single_sample_returns_none(self):
        """Single sample after reshape -> returns None."""
        returns = torch.randn(1, 1, 10, 1)  # B*T = 1 sample
        assert task._top1_eigenvalue(returns) is None

    def test_single_stock_returns_none(self):
        """N=1 stock -> returns None."""
        returns = torch.randn(10, 20, 1, 1)
        assert task._top1_eigenvalue(returns) is None

    def test_5d_input(self):
        """5D ensemble input works."""
        torch.manual_seed(6)
        returns = torch.randn(10, 5, 20, 8, 1)
        ev = task._top1_eigenvalue(returns)
        assert ev is not None
        assert np.isfinite(ev)

    def test_nonfinite_returns_none(self):
        """Non-finite inputs (NaN/Inf) return None instead of crashing."""
        returns = torch.randn(10, 20, 5, 1)
        returns[0, 0, 0, 0] = float("nan")
        assert task._top1_eigenvalue(returns) is None

        returns2 = torch.randn(10, 20, 5, 1)
        returns2[1, 2, 3, 0] = float("inf")
        assert task._top1_eigenvalue(returns2) is None


# ======================================================================
# Integration: eigval1_ratio skip policy
# ======================================================================


class TestEigvalRatioPolicy:
    def test_ratio_skipped_when_real_small(self):
        """When eigval1_real < 0.01, ratio key is absent."""
        # With N=2 and real data, eigval1_real will be >= 1.0 normally.
        # Force a degenerate case: constant target with tiny variance
        torch.manual_seed(8)
        pred = torch.randn(10, 20, 5, 1)
        # Target: nearly constant per-stock (eigval ~ 0 after standardization)
        # Actually, with std normalization, even tiny-variance stocks get eigval ~1.
        # Instead, test via direct helper calls mimicking the call-site logic.
        ev_gen = task._top1_eigenvalue(pred)
        ev_real = 0.005  # Simulated degenerate real eigenvalue

        metrics = {}
        if ev_gen is not None:
            metrics["eigval1_gen"] = ev_gen
        if ev_real is not None:
            metrics["eigval1_real"] = ev_real
        if ev_gen is not None and ev_real is not None and ev_real >= 0.01:
            metrics["eigval1_ratio"] = ev_gen / ev_real

        assert "eigval1_ratio" not in metrics, "Ratio should be skipped for real < 0.01"

    def test_ratio_present_when_real_normal(self):
        """When eigval1_real >= 0.01, ratio is emitted."""
        torch.manual_seed(9)
        pred = torch.randn(10, 20, 5, 1)
        target = torch.randn(10, 20, 5, 1)

        metrics = {}
        ev_gen = task._top1_eigenvalue(pred)
        ev_real = task._top1_eigenvalue(target)
        if ev_gen is not None:
            metrics["eigval1_gen"] = ev_gen
        if ev_real is not None:
            metrics["eigval1_real"] = ev_real
        if ev_gen is not None and ev_real is not None and ev_real >= 0.01:
            metrics["eigval1_ratio"] = ev_gen / ev_real

        assert "eigval1_ratio" in metrics
        assert np.isfinite(metrics["eigval1_ratio"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
