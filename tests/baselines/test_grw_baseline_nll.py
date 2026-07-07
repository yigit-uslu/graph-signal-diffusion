"""Unit tests for the static Geometric Random Walk baseline."""
from __future__ import annotations

import math
import pytest
import torch
from collections import namedtuple
from unittest.mock import MagicMock
from torch_geometric.data import Data, Batch

from graph_signal_diffusion.baselines.stock_price_forecasting.grw import (
    GeometricRandomWalk,
)

""" 
# Defaults (N=10, T=20, B=5000)
python tests/baselines/test_grw_baseline_nll.py --out-dir figures/

# Custom parameters
python tests/baselines/test_grw_baseline_nll.py \
    --n-stocks 500 \
    --n-steps 20 \
    --n-trajectories 10000 \
    --mean-range -0.1 0.1 \
    --std-range 0.9 1.1 \
    --out-dir outputs/nll_histograms \
    --seed 0
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_batch(B: int, N: int, T: int, F: int = 1) -> Batch:
    """Create a minimal PyG Batch that mimics the SP500 dataloader output.

    ``y`` is filled with samples from a *known* distribution so that the
    fitted parameters can be verified.
    """
    graphs = []
    for _ in range(B):
        y = torch.randn(N, T, F)  # std≈1, mean≈0 (standardised space)
        x = torch.randn(N, 10, 12)  # dummy features
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        close_price = torch.rand(N, 10, 1) * 100 + 50
        close_price_y = torch.rand(N, T, 1) * 100 + 50
        stocks_index = torch.arange(N)
        timestamp = torch.zeros(N, dtype=torch.long)

        data = Data(
            x=x, y=y, edge_index=edge_index,
            close_price=close_price,
            close_price_y=close_price_y,
            stocks_index=stocks_index,
            timestamp=timestamp,
        )
        graphs.append(data)

    return Batch.from_data_list(graphs)


def _make_loader(n_batches: int = 4, B: int = 4, N: int = 8, T: int = 5):
    """Return a list-based 'loader' of batches (quacks like a DataLoader)."""
    return [_make_batch(B, N, T) for _ in range(n_batches)]


# ---------------------------------------------------------------------------
# Tests: fit()
# ---------------------------------------------------------------------------

class TestFit:
    def test_global_stats_near_standard_normal(self):
        loader = _make_loader(n_batches=8, B=8, N=4, T=5)
        grw = GeometricRandomWalk(device="cpu", n_samples=5, shrinkage_strength=10.0)
        grw.fit(loader)

        # With enough samples, global stats should be close to N(0,1)
        assert abs(grw.global_mean) < 0.15, f"global_mean={grw.global_mean}"
        assert abs(grw.global_std - 1.0) < 0.15, f"global_std={grw.global_std}"

    def test_per_stock_count(self):
        N = 6
        loader = _make_loader(n_batches=4, B=4, N=N, T=5)
        grw = GeometricRandomWalk(device="cpu", n_samples=3)
        grw.fit(loader)

        assert len(grw.per_stock_means) == N
        assert len(grw.per_stock_stds) == N
        assert grw.n_stocks == N

    def test_shrinkage_toward_global(self):
        """With very high shrinkage_strength, per-stock params should ≈ global."""
        loader = _make_loader(n_batches=4, B=4, N=4, T=5)
        grw = GeometricRandomWalk(device="cpu", shrinkage_strength=1e6)
        grw.fit(loader)

        for s in range(4):
            assert abs(grw.per_stock_means[s] - grw.global_mean) < 1e-4
            assert abs(grw.per_stock_stds[s] - grw.global_std) < 1e-4

    def test_no_shrinkage(self):
        """With shrinkage_strength=0, per-stock params should be pure MLE."""
        loader = _make_loader(n_batches=4, B=4, N=4, T=5)
        grw = GeometricRandomWalk(device="cpu", shrinkage_strength=0.0)
        grw.fit(loader)

        # With k=0, lambda=1 → no shrinkage → just check it doesn't crash
        for s in range(4):
            assert grw.per_stock_means[s] is not None


# ---------------------------------------------------------------------------
# Tests: predict()
# ---------------------------------------------------------------------------

class TestPredict:
    @pytest.fixture(autouse=True)
    def _setup(self):
        loader = _make_loader(n_batches=4, B=4, N=6, T=5)
        self.grw = GeometricRandomWalk(device="cpu", n_samples=10)
        self.grw.fit(loader)
        self.batch = _make_batch(B=2, N=6, T=5)

    def test_predict_shape(self):
        out = self.grw.predict(self.batch)
        assert out.shape == (2, 5, 6, 1), f"Expected [B,T,N,1], got {out.shape}"

    def test_predict_is_deterministic_mean(self):
        """predict() returns the analytical mean, so two calls should match."""
        out1 = self.grw.predict(self.batch)
        out2 = self.grw.predict(self.batch)
        assert torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# Tests: predict_ensemble()
# ---------------------------------------------------------------------------

class TestPredictEnsemble:
    @pytest.fixture(autouse=True)
    def _setup(self):
        loader = _make_loader(n_batches=4, B=4, N=6, T=5)
        self.grw = GeometricRandomWalk(device="cpu", n_samples=7)
        self.grw.fit(loader)
        self.batch = _make_batch(B=3, N=6, T=5)

    def test_ensemble_shape(self):
        out = self.grw.predict_ensemble(self.batch)
        # B*n_samples, T, N, F
        assert out.shape == (3 * 7, 5, 6, 1), f"Got {out.shape}"

    def test_ensemble_is_stochastic(self):
        out1 = self.grw.predict_ensemble(self.batch)
        out2 = self.grw.predict_ensemble(self.batch)
        assert not torch.allclose(out1, out2), "Two draws should differ"


# ---------------------------------------------------------------------------
# Tests: evaluate() with a mock task
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_evaluate_runs_without_error(self):
        N, T, F = 4, 5, 1
        loader = _make_loader(n_batches=3, B=2, N=N, T=T)
        grw = GeometricRandomWalk(device="cpu", n_samples=3)
        grw.fit(loader)

        # Build a mock task whose prepare_data returns the right shapes
        mock_task = MagicMock()

        def _prepare_data(data):
            B = data.num_graphs
            samples = data.y.squeeze(-1) if data.y.dim() > 2 else data.y
            # Reshape to [B, T, N, F]
            samples = samples.view(B, N, T, F).permute(0, 2, 1, 3)
            return {
                "samples": samples,
                "metadata": {
                    "batch_size": B,
                    "num_timesteps": T,
                    "num_stocks": N,
                    "num_features": F,
                    "close_price": data.close_price,
                    "close_price_y": data.close_price_y,
                    "stocks_index": data.stocks_index,
                    "timestamp": data.timestamp,
                },
            }

        mock_task.prepare_data.side_effect = _prepare_data
        mock_task.evaluate_samples.return_value = {"price_mse": 0.42}

        n_test_batches = 2
        test_loader = _make_loader(n_batches=n_test_batches, B=2, N=N, T=T)
        metrics = grw.evaluate(test_loader, mock_task)

        assert "price_mse" in metrics
        # Metrics should be averaged across batches
        assert metrics["price_mse"] == 0.42  # constant per batch → average is the same

        # evaluate_samples is called once per batch (not once with all data)
        assert mock_task.evaluate_samples.call_count == n_test_batches

        # Verify each per-batch call has correct shapes
        for call in mock_task.evaluate_samples.call_args_list:
            gen, real, meta = call[0]
            expected_per_batch = 2 * 3  # B * n_samples
            assert gen.shape[0] == expected_per_batch
            assert real.shape[0] == expected_per_batch
            assert meta["n_samples_per_input"] == 3
            assert meta["batch_size"] == expected_per_batch


# ---------------------------------------------------------------------------
# Tests: metadata helpers
# ---------------------------------------------------------------------------

class TestMetadataHelpers:
    def test_replicate_metadata_tensors(self):
        meta = {
            "close_price": torch.randn(4, 10, 1),
            "batch_size": 4,
            "n_samples_per_input": 1,
        }
        out = GeometricRandomWalk._replicate_metadata(meta, n=3)
        assert out["close_price"].shape[0] == 12
        assert out["batch_size"] == 4  # scalar — not replicated


# ---------------------------------------------------------------------------
# Tests: get_params
# ---------------------------------------------------------------------------

class TestGetParams:
    def test_not_fitted(self):
        grw = GeometricRandomWalk(device="cpu")
        assert grw.get_params()["status"] == "not_fitted"

    def test_fitted(self):
        loader = _make_loader(n_batches=2, B=2, N=4, T=5)
        grw = GeometricRandomWalk(device="cpu")
        grw.fit(loader)
        params = grw.get_params()
        assert "global_mean" in params
        assert params["n_stocks"] == 4


# ---------------------------------------------------------------------------
# Helpers for NLL histogram tests
# ---------------------------------------------------------------------------

def _make_grw_with_known_params(
    true_means: torch.Tensor,
    true_stds: torch.Tensor,
) -> GeometricRandomWalk:
    """Return a GeometricRandomWalk with parameters injected directly.

    Bypasses fitting so that tests are fast and fully controlled.  The
    injected parameters are treated as if they came from a full ``fit()``
    call.

    Args:
        true_means: Per-stock means, shape ``[N]``.
        true_stds:  Per-stock standard deviations, shape ``[N]``.

    Returns:
        A fitted ``GeometricRandomWalk`` instance.
    """
    N = true_means.size(0)
    grw = GeometricRandomWalk(device="cpu", n_samples=1)

    global_mean = true_means.mean().item()
    global_std = true_stds.mean().item()

    grw.global_mean = global_mean
    grw.global_std = global_std
    grw.n_stocks = N
    grw.per_stock_means = {s: true_means[s].item() for s in range(N)}
    grw.per_stock_stds = {s: true_stds[s].item() for s in range(N)}

    return grw


def _sample_from_params(
    B: int,
    T: int,
    true_means: torch.Tensor,
    true_stds: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample standardised log-returns from known per-stock Gaussians.

    Returns:
        Tensor ``[B, T, N]`` drawn i.i.d. from
        ``r_{b,t,s} ~ N(true_means[s], true_stds[s]^2)``.
    """
    N = true_means.size(0)
    z = torch.randn(B, T, N, generator=generator)
    return z * true_stds.unsqueeze(0).unsqueeze(0) + true_means.unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Tests: compute_nll
# ---------------------------------------------------------------------------

class TestComputeNLL:
    """Unit tests for the analytical per-trajectory NLL computation."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        torch.manual_seed(0)
        N, T = 8, 15
        self.N, self.T = N, T

        # Heterogeneous true parameters: means in [-0.3, 0.3], stds in [0.7, 1.3]
        self.true_means = torch.linspace(-0.3, 0.3, N)
        self.true_stds = torch.linspace(0.7, 1.3, N)

        self.grw = _make_grw_with_known_params(self.true_means, self.true_stds)

    def test_output_shape(self):
        """compute_nll returns a 1-D tensor of length B."""
        B = 32
        gen = torch.Generator()
        gen.manual_seed(1)
        r = _sample_from_params(B, self.T, self.true_means, self.true_stds, gen)
        nll = self.grw.compute_nll(r)
        assert nll.shape == (B,), f"Expected [B], got {nll.shape}"

    def test_accepts_4d_input(self):
        """compute_nll squeezes the trailing feature dimension correctly."""
        B = 10
        gen = torch.Generator()
        gen.manual_seed(2)
        r = _sample_from_params(B, self.T, self.true_means, self.true_stds, gen)
        nll_3d = self.grw.compute_nll(r)
        nll_4d = self.grw.compute_nll(r.unsqueeze(-1))  # [B, T, N, 1]
        assert torch.allclose(nll_3d, nll_4d)

    def test_nll_is_positive(self):
        """NLL values must be positive (negative log of a proper density)."""
        B = 50
        gen = torch.Generator()
        gen.manual_seed(3)
        r = _sample_from_params(B, self.T, self.true_means, self.true_stds, gen)
        nll = self.grw.compute_nll(r)
        assert (nll > 0).all(), "All per-trajectory NLLs should be positive"

    def test_mean_nll_matches_gaussian_entropy(self):
        """E[NLL] should converge to N*T * average Gaussian differential entropy.

        For r_{b,t,s} ~ N(μ_s, σ_s²):
            E[-log p(r)] = 0.5*log(2*pi*e*σ_s²)  per (t, s) pair
        Summed over N and T:
            E[NLL_b] = T * Σ_s  0.5 * log(2*pi*e*σ_s²)
        """
        B = 2000  # many samples → accurate empirical mean
        gen = torch.Generator()
        gen.manual_seed(4)
        r = _sample_from_params(B, self.T, self.true_means, self.true_stds, gen)
        nll = self.grw.compute_nll(r)
        empirical_mean = nll.mean().item()

        # Theoretical expected NLL (sum over all N stocks and T timesteps)
        expected_mean = self.T * sum(
            0.5 * math.log(2 * math.pi * math.e * self.true_stds[s].item() ** 2)
            for s in range(self.N)
        )
        rel_error = abs(empirical_mean - expected_mean) / abs(expected_mean)
        assert rel_error < 0.02, (
            f"Mean NLL {empirical_mean:.3f} deviates from theoretical "
            f"{expected_mean:.3f} by {rel_error*100:.1f}% (threshold 2%)"
        )

    def test_higher_nll_for_outlier_trajectory(self):
        """A trajectory with extreme returns should have higher NLL than a typical one."""
        gen = torch.Generator()
        gen.manual_seed(5)
        typical = _sample_from_params(1, self.T, self.true_means, self.true_stds, gen)
        # Outlier: returns offset by +5 standard deviations for all stocks
        outlier = typical + 5.0
        nll_typical = self.grw.compute_nll(typical)
        nll_outlier = self.grw.compute_nll(outlier)
        assert nll_outlier.item() > nll_typical.item(), (
            "Outlier trajectory should have higher NLL than a typical one"
        )


# ---------------------------------------------------------------------------
# Tests: NLL histogram comparison (real vs GRW-generated)
# ---------------------------------------------------------------------------

class TestNLLHistograms:
    """Compare NLL_GRW(r_real) vs NLL_GRW(r_GRW) histograms.

    Both sets of trajectories are drawn from the **same** GRW distribution
    (by construction), so the two histograms should be statistically
    indistinguishable.  We verify this by checking that the means and
    standard deviations are close.

    The test also saves an annotated histogram figure so the user can
    visually inspect the overlap quality.

    Histogram semantics
    -------------------
    Each histogram entry is the per-trajectory NLL
    ``NLL_b = Σ_{s,t} -log p(r_{b,t,s})``, which is the **sum** (not
    average) over all T timesteps and N stocks.  There is no normalisation
    by B, T, or N — the raw sum is used so that the histogram displays the
    full joint log-likelihood of the trajectory under the GRW model.
    """

    N: int = 10
    T: int = 20
    B_TRAIN: int = 5_000  # trajectories used to draw r_real
    B_GRW: int = 5_000   # trajectories sampled from the GRW model

    @pytest.fixture(autouse=True)
    def _setup(self):
        torch.manual_seed(42)
        # Heterogeneous per-stock parameters
        self.true_means = torch.linspace(-0.25, 0.25, self.N)
        self.true_stds = torch.linspace(0.8, 1.2, self.N)
        self.grw = _make_grw_with_known_params(self.true_means, self.true_stds)

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _compute_nll_real(self) -> torch.Tensor:
        """Draw r_real from the true distribution and compute per-traj NLL."""
        gen = torch.Generator()
        gen.manual_seed(100)
        r_real = _sample_from_params(
            self.B_TRAIN, self.T, self.true_means, self.true_stds, gen
        )
        return self.grw.compute_nll(r_real)

    def _compute_nll_grw(self) -> torch.Tensor:
        """Sample r_GRW from the fitted GRW model and compute per-traj NLL."""
        r_grw = self.grw._sample_returns(self.B_GRW, self.T, self.N)
        # _sample_returns returns [B, T, N, 1]; squeeze the feature dim
        return self.grw.compute_nll(r_grw.squeeze(-1))

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_nll_real_and_grw_shapes(self):
        """Both NLL vectors should have length equal to the batch size."""
        nll_real = self._compute_nll_real()
        nll_grw = self._compute_nll_grw()
        assert nll_real.shape == (self.B_TRAIN,)
        assert nll_grw.shape == (self.B_GRW,)

    def test_nll_means_close(self):
        """Mean NLL should match within 2% — same Gaussian distribution."""
        nll_real = self._compute_nll_real()
        nll_grw = self._compute_nll_grw()
        mean_real = nll_real.mean().item()
        mean_grw = nll_grw.mean().item()
        rel_diff = abs(mean_real - mean_grw) / abs(mean_real)
        assert rel_diff < 0.02, (
            f"Mean NLL real={mean_real:.3f}, GRW={mean_grw:.3f}; "
            f"relative difference {rel_diff*100:.2f}% exceeds 2% threshold"
        )

    def test_nll_stds_close(self):
        """Standard deviation of NLL should match within 5%."""
        nll_real = self._compute_nll_real()
        nll_grw = self._compute_nll_grw()
        std_real = nll_real.std().item()
        std_grw = nll_grw.std().item()
        rel_diff = abs(std_real - std_grw) / abs(std_real)
        assert rel_diff < 0.05, (
            f"Std NLL real={std_real:.3f}, GRW={std_grw:.3f}; "
            f"relative difference {rel_diff*100:.2f}% exceeds 5% threshold"
        )

    def test_nll_wasserstein_close(self):
        """W1 between NLL_real and NLL_GRW should be small relative to the NLL scale.

        For 1-D distributions the Wasserstein-1 distance equals the L1 norm
        of the difference between the two empirical CDFs:

            W1 = integral |F_real(x) - F_GRW(x)| dx

        Both NLL vectors are drawn from the *same* distribution (by
        construction), so W1 should be much smaller than the typical NLL
        value.  We use a 1% relative threshold against the mean NLL as a
        sanity check.
        """
        from scipy.stats import wasserstein_distance

        nll_real = self._compute_nll_real().numpy()
        nll_grw = self._compute_nll_grw().numpy()

        w1 = wasserstein_distance(nll_real, nll_grw)
        mean_nll = float(nll_real.mean())
        rel_w1 = w1 / abs(mean_nll)
        assert rel_w1 < 0.01, (
            f"W1={w1:.3f} nats is {rel_w1*100:.2f}% of mean NLL={mean_nll:.1f}; "
            f"exceeds 1% threshold"
        )

    def test_nll_histogram_plot(self, tmp_path):
        """Save overlapping NLL histograms for visual inspection.

        The figure is written to ``tmp_path/nll_histogram_grw.png``.
        Pass ``--histogram-dir=<path>`` as a pytest option (or inspect
        ``tmp_path`` directly) to retrieve the figure after the run.
        """
        import matplotlib
        matplotlib.use("Agg")  # headless — no display required
        import matplotlib.pyplot as plt

        nll_real = self._compute_nll_real().numpy()
        nll_grw = self._compute_nll_grw().numpy()

        # Theoretical expected NLL (sum over N*T Gaussian terms)
        expected_mean = self.T * sum(
            0.5 * math.log(2 * math.pi * math.e * self.true_stds[s].item() ** 2)
            for s in range(self.N)
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        bins = 60

        ax.hist(
            nll_real, bins=bins, density=True, alpha=0.55,
            label=f"$\\mathrm{{NLL}}_{{\\mathrm{{GRW}}}}(r^{{\\mathrm{{real}}}})$  "
                  f"μ={nll_real.mean():.1f}, σ={nll_real.std():.1f}",
            color="steelblue",
        )
        ax.hist(
            nll_grw, bins=bins, density=True, alpha=0.55,
            label=f"$\\mathrm{{NLL}}_{{\\mathrm{{GRW}}}}(r^{{\\mathrm{{GRW}}}})$  "
                  f"μ={nll_grw.mean():.1f}, σ={nll_grw.std():.1f}",
            color="darkorange",
        )
        ax.axvline(
            expected_mean, color="black", linestyle="--", linewidth=1.5,
            label=f"Theoretical $E[\\mathrm{{NLL}}]$ = {expected_mean:.1f}",
        )
        ax.set_xlabel(
            f"Per-trajectory NLL  "
            r"$= \sum_{s,t} -\log\,\mathcal{N}(r_{s,t}\mid\tilde{\mu}_s,\tilde{\sigma}_s^2)$",
            fontsize=11,
        )
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(
            f"GRW NLL Histograms  (N={self.N} stocks, T={self.T} steps, "
            f"B={self.B_TRAIN} trajectories)",
            fontsize=12,
        )
        ax.legend(fontsize=10)
        fig.tight_layout()

        out_path = tmp_path / "nll_histogram_grw.pdf"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"\n[TestNLLHistograms] histogram saved to: {out_path}")

        assert out_path.exists(), "Histogram figure was not created"


# ---------------------------------------------------------------------------
# Structural metrics helpers
# ---------------------------------------------------------------------------

def compute_lag1_autocorr(returns: torch.Tensor) -> float:
    """Lag-1 autocorrelation (standard ACF) of squared returns, pooled over all stocks.

    For each stock *s* we compute the standard ACF at lag 1:
    ρ(1) = γ(1) / γ(0), using a single global mean μ = E[r²] and
    normalising by the lag-0 autocovariance γ(0) = Var(r²).

    Under i.i.d. sampling (GRW assumption) this should be ≈ 0.
    Under volatility clustering (ARCH/GARCH) it is positive.

    Args:
        returns: Standardised log-returns ``[B, T, N]``.

    Returns:
        Mean lag-1 ACF of ``r²`` across all N stocks (scalar).
    """
    B, T, N = returns.shape
    r2 = returns ** 2  # [B, T, N]
    r2_flat = r2.reshape(-1, N)
    mu = r2_flat.mean(0)                      # [N] global mean
    gamma_0 = ((r2_flat - mu) ** 2).mean(0)   # [N] lag-0 autocovariance

    # Lag-1 pairs
    r2_t  = r2[:, :-1, :].reshape(-1, N)  # r²_t
    r2_t1 = r2[:, 1:,  :].reshape(-1, N)  # r²_{t+1}

    gamma_1 = ((r2_t - mu) * (r2_t1 - mu)).mean(0)  # [N]
    rho_1 = gamma_1 / (gamma_0 + 1e-8)               # [N]
    return rho_1.mean().item()


def compute_autocorr_profile(
    returns: torch.Tensor,
    max_lag: int = 10,
) -> torch.Tensor:
    """Lag-0..max_lag autocorrelation (standard ACF) of squared returns, averaged over stocks.

    For each lag ``l`` and each stock ``s``:

    .. math::

        \rho_l^{(s)} = \gamma_l^{(s)} / \gamma_0^{(s)}

    Returns the mean over all stocks at each lag.

    Args:
        returns: ``[B, T, N]`` standardised log-returns.
        max_lag: Maximum lag to compute (inclusive).

    Returns:
        ``[max_lag + 1]`` tensor of autocorrelations at lags 0…max_lag.
    """
    B, T, N = returns.shape
    r2 = returns ** 2  # [B, T, N]
    r2_flat = r2.reshape(-1, N)
    mu = r2_flat.mean(0)                      # [N] global mean
    gamma_0 = ((r2_flat - mu) ** 2).mean(0)   # [N] lag-0 autocovariance

    profile = torch.zeros(max_lag + 1)
    profile[0] = 1.0  # ρ(0) = 1
    for lag in range(1, max_lag + 1):
        r2_t  = r2[:, :T - lag, :].reshape(-1, N)  # [B*(T-lag), N]
        r2_tl = r2[:, lag:,     :].reshape(-1, N)  # [B*(T-lag), N]

        gamma_l = ((r2_t - mu) * (r2_tl - mu)).mean(0)  # [N]
        rho_l = gamma_l / (gamma_0 + 1e-8)               # [N]
        profile[lag] = rho_l.mean()

    return profile


def compute_top_eigenvalues(returns: torch.Tensor, k: int = 5) -> torch.Tensor:
    """Top-k eigenvalues of the empirical return correlation matrix.

    Observations are pooled over B batches and T timesteps, giving an
    ``[B*T, N]`` design matrix from which the ``[N, N]`` empirical
    correlation is estimated.

    Under i.i.d. independent stocks (GRW) all eigenvalues are ≈ 1.
    Under a factor model the first eigenvalue is ≫ 1 (market factor).

    Args:
        returns: ``[B, T, N]`` standardised log-returns.
        k: Number of top eigenvalues to return.

    Returns:
        Eigenvalues in descending order, shape ``[k]``.
    """
    B, T, N = returns.shape
    r = returns.reshape(-1, N).float()  # [B*T, N]

    # Z-score each stock across observations
    r = r - r.mean(0)
    r = r / (r.std(0) + 1e-8)

    # Empirical correlation matrix [N, N]
    C = (r.T @ r) / r.shape[0]

    # Eigenvalues of real-symmetric matrix (ascending from eigvalsh)
    vals = torch.linalg.eigvalsh(C)  # [N], ascending
    return vals.flip(0)[:k]          # [k], descending


def _make_arch1_returns(
    B: int, T: int, N: int,
    alpha: float = 0.5, omega: float = 0.5,
    seed: int = 0,
) -> torch.Tensor:
    """ARCH(1) returns with known positive lag-1 autocorrelation in r².

    Model: r_t = σ_t * ε_t,  σ_t² = ω + α * r_{t-1}²

    The theoretical lag-1 autocorrelation of ``r²`` is approximately
    ``α`` (exact for conditional Gaussian ARCH(1)).

    Args:
        B: Batch size (independent trajectories).
        T: Sequence length.
        N: Number of stocks (each modelled independently, same params).
        alpha: ARCH coefficient; controls autocorrelation strength.
        omega: Constant variance term; controls unconditional variance
               ``E[r²] = ω / (1 − α)`` (requires α < 1 for stationarity).
        seed: RNG seed.

    Returns:
        ``[B, T, N]`` returns.
    """
    gen = torch.Generator()
    gen.manual_seed(seed)

    returns = torch.zeros(B, T, N)
    sigma2 = torch.ones(B, N)  # initial conditional variance

    for t in range(T):
        eps = torch.randn(B, N, generator=gen)
        r_t = sigma2.sqrt() * eps
        returns[:, t, :] = r_t
        sigma2 = omega + alpha * r_t ** 2

    return returns


def _make_factor_model_returns(
    B: int, T: int, N: int,
    n_factors: int = 1, factor_loading: float = 0.8,
    seed: int = 0,
) -> torch.Tensor:
    """Factor-model returns with a dominant common factor.

    Model: r_{b,t,s} = a * f_{b,t} + √(1-a²) * ε_{b,t,s}

    All stocks share the same loading ``a`` on a single standard-normal
    factor ``f``, plus independent idiosyncratic noise ``ε``.  The
    return variance per stock is 1 and the pairwise correlation is ``a²``.

    The theoretical correlation matrix is
        C = a² * 11ᵀ + (1 - a²) * I
    with top eigenvalue ``N * a² + (1 - a²)`` and remaining N-1
    eigenvalues equal to ``1 - a²``.

    Args:
        B: Batch size.
        T: Sequence length.
        N: Number of stocks.
        n_factors: Number of independent common factors (each with loading ``a``).
        factor_loading: Loading coefficient ``a``; pairwise corr = ``a²``.
        seed: RNG seed.

    Returns:
        ``[B, T, N]`` returns with unit variance per stock.
    """
    gen = torch.Generator()
    gen.manual_seed(seed)

    factors = torch.randn(B, T, n_factors, generator=gen)      # [B, T, k]
    loadings = factor_loading * torch.ones(N, n_factors)        # [N, k]
    eps = torch.randn(B, T, N, generator=gen)                   # [B, T, N]
    idio_scale = math.sqrt(max(1.0 - n_factors * factor_loading ** 2, 1e-6))

    # [B, T, N] = [B, T, k] @ [k, N] + idio
    return (factors @ loadings.T) + idio_scale * eps


# ---------------------------------------------------------------------------
# Tests: lag-1 autocorrelation of squared returns
# ---------------------------------------------------------------------------

class TestLag1Autocorrelation:
    """Check lag-1 autocorrelation of r² for GRW (i.i.d.) vs ARCH(1) data.

    GRW samples are i.i.d. by construction → autocorr ≈ 0.
    ARCH(1) samples have volatility clustering → autocorr ≈ α > 0.

    This is the key test for temporal structure that GRW misses:
    real financial returns exhibit positive autocorrelation in squared
    returns (volatility clustering) that a well-trained diffusion model
    should reproduce.
    """

    B: int = 500
    T: int = 20
    N: int = 10
    ARCH_ALPHA: float = 0.5  # theoretical lag-1 autocorr of r² ≈ 0.5

    @pytest.fixture(autouse=True)
    def _setup(self):
        torch.manual_seed(0)
        self.true_means = torch.zeros(self.N)
        self.true_stds  = torch.ones(self.N)
        self.grw = _make_grw_with_known_params(self.true_means, self.true_stds)

    def test_helper_output_is_scalar(self):
        """compute_lag1_autocorr returns a python float."""
        gen = torch.Generator(); gen.manual_seed(10)
        r = _sample_from_params(self.B, self.T, self.true_means, self.true_stds, gen)
        result = compute_lag1_autocorr(r)
        assert isinstance(result, float), f"Expected float, got {type(result)}"

    def test_grw_autocorr_near_zero(self):
        """GRW samples are i.i.d. → lag-1 autocorr of r² ≈ 0 (|ρ| < 0.05)."""
        gen = torch.Generator(); gen.manual_seed(11)
        r_grw = _sample_from_params(self.B, self.T, self.true_means, self.true_stds, gen)
        rho = compute_lag1_autocorr(r_grw)
        assert abs(rho) < 0.05, (
            f"GRW lag-1 autocorr = {rho:.4f}; expected |ρ| < 0.05 for i.i.d. samples"
        )

    def test_arch1_autocorr_positive(self):
        """ARCH(1) samples have positive lag-1 autocorr of r² ≈ α={ARCH_ALPHA}."""
        r_arch = _make_arch1_returns(self.B, self.T, self.N, alpha=self.ARCH_ALPHA, seed=12)
        rho = compute_lag1_autocorr(r_arch)
        # Allow wide tolerance: empirical estimate converges slowly
        # Theoretical value is α; we just check it is clearly positive
        assert rho > 0.2, (
            f"ARCH(1) lag-1 autocorr = {rho:.4f}; expected > 0.2 "
            f"(theoretical ≈ {self.ARCH_ALPHA})"
        )

    def test_arch1_autocorr_exceeds_grw(self):
        """ARCH(1) autocorr should be clearly larger than GRW autocorr."""
        gen = torch.Generator(); gen.manual_seed(13)
        r_grw  = _sample_from_params(self.B, self.T, self.true_means, self.true_stds, gen)
        r_arch = _make_arch1_returns(self.B, self.T, self.N, alpha=self.ARCH_ALPHA, seed=14)

        rho_grw  = compute_lag1_autocorr(r_grw)
        rho_arch = compute_lag1_autocorr(r_arch)

        assert rho_arch > rho_grw + 0.1, (
            f"ARCH(1) ρ={rho_arch:.4f} should exceed GRW ρ={rho_grw:.4f} by > 0.1"
        )

    def test_theoretical_convergence(self):
        """With many samples, empirical autocorr should be within 0.1 of α."""
        B_large = 2000
        r_arch = _make_arch1_returns(B_large, self.T, self.N, alpha=self.ARCH_ALPHA, seed=15)
        rho = compute_lag1_autocorr(r_arch)
        assert abs(rho - self.ARCH_ALPHA) < 0.1, (
            f"Empirical ARCH(1) autocorr {rho:.4f} differs from "
            f"theoretical {self.ARCH_ALPHA} by more than 0.1"
        )


# ---------------------------------------------------------------------------
# Tests: top eigenvalues of the empirical correlation matrix
# ---------------------------------------------------------------------------

class TestTopEigenvalues:
    """Check the top eigenvalues of the empirical correlation matrix.

    GRW generates independent stocks → all eigenvalues ≈ 1 (identity
    correlation matrix), so the top eigenvalue should be close to 1.

    A factor model with a strong common factor → top eigenvalue ≈ N*a²,
    which is much larger than 1.

    This is the key test for cross-stock structure:
    real equity returns have a dominant market factor (first eigenvalue
    >> 1) and sector factors (next few eigenvalues > 1) that a diffusion
    model should reproduce but GRW cannot.
    """

    B: int = 500
    T: int = 20
    N: int = 15
    FACTOR_LOADING: float = 0.8
    # Theoretical top eigenvalue for factor model: N*a² + (1-a²)
    # = 15*0.64 + 0.36 = 9.96

    @pytest.fixture(autouse=True)
    def _setup(self):
        torch.manual_seed(0)
        self.true_means = torch.zeros(self.N)
        self.true_stds  = torch.ones(self.N)

    def test_helper_output_shape(self):
        """compute_top_eigenvalues returns a tensor of the requested length."""
        gen = torch.Generator(); gen.manual_seed(20)
        r = _sample_from_params(self.B, self.T, self.true_means, self.true_stds, gen)
        vals = compute_top_eigenvalues(r, k=3)
        assert vals.shape == (3,), f"Expected shape (3,), got {vals.shape}"

    def test_eigenvalues_descending(self):
        """Returned eigenvalues should be in descending order."""
        gen = torch.Generator(); gen.manual_seed(21)
        r = _sample_from_params(self.B, self.T, self.true_means, self.true_stds, gen)
        vals = compute_top_eigenvalues(r, k=5)
        assert (vals[:-1] >= vals[1:]).all(), (
            f"Eigenvalues not in descending order: {vals.tolist()}"
        )

    def test_grw_top_eigenvalue_near_one(self):
        """Independent GRW stocks → top eigenvalue should be close to 1.

        With many observations the sample correlation matrix converges to
        identity.  We allow up to 1.5 to account for finite-sample bias
        (Marchenko-Pastur upper edge for q = N/(B*T) ≈ 0.0015 is ≈ 1.08).
        """
        gen = torch.Generator(); gen.manual_seed(22)
        r_grw = _sample_from_params(self.B, self.T, self.true_means, self.true_stds, gen)
        top_val = compute_top_eigenvalues(r_grw, k=1).item()
        assert top_val < 1.5, (
            f"GRW top eigenvalue = {top_val:.3f}; expected < 1.5 for independent stocks"
        )

    def test_factor_model_top_eigenvalue_large(self):
        """Factor model → top eigenvalue should be >> 1 (≈ N*a² + (1-a²))."""
        theoretical_top = self.N * self.FACTOR_LOADING ** 2 + (1 - self.FACTOR_LOADING ** 2)
        r_factor = _make_factor_model_returns(
            self.B, self.T, self.N,
            factor_loading=self.FACTOR_LOADING, seed=23,
        )
        top_val = compute_top_eigenvalues(r_factor, k=1).item()
        # Allow 20% tolerance around theoretical value
        assert top_val > theoretical_top * 0.8, (
            f"Factor model top eigenvalue = {top_val:.3f}; "
            f"expected ≈ {theoretical_top:.2f} (within 20%)"
        )

    def test_factor_top_eigenvalue_exceeds_grw(self):
        """Factor model top eigenvalue should be much larger than GRW's."""
        gen = torch.Generator(); gen.manual_seed(24)
        r_grw    = _sample_from_params(self.B, self.T, self.true_means, self.true_stds, gen)
        r_factor = _make_factor_model_returns(
            self.B, self.T, self.N,
            factor_loading=self.FACTOR_LOADING, seed=25,
        )
        top_grw    = compute_top_eigenvalues(r_grw,    k=1).item()
        top_factor = compute_top_eigenvalues(r_factor, k=1).item()
        assert top_factor > top_grw * 4, (
            f"Factor top={top_factor:.3f} should be at least 4× GRW top={top_grw:.3f}"
        )

    def test_grw_eigenvalue_spectrum_is_flat(self):
        """GRW eigenvalues should be roughly equal — no dominant factor.

        The ratio top/second should be close to 1 (flat spectrum).
        """
        B_large = 2000
        gen = torch.Generator(); gen.manual_seed(26)
        r_grw = _sample_from_params(B_large, self.T, self.true_means, self.true_stds, gen)
        vals  = compute_top_eigenvalues(r_grw, k=5)
        ratio = (vals[0] / vals[-1]).item()
        assert ratio < 1.5, (
            f"GRW eigenvalue ratio top/5th = {ratio:.3f}; "
            f"expected < 1.5 for a flat spectrum (independent stocks)"
        )

    def test_factor_model_eigenvalue_gap(self):
        """Factor model should have a large gap between 1st and 2nd eigenvalue."""
        r_factor = _make_factor_model_returns(
            self.B, self.T, self.N,
            factor_loading=self.FACTOR_LOADING, seed=27,
        )
        vals = compute_top_eigenvalues(r_factor, k=2)
        gap_ratio = (vals[0] / vals[1]).item()
        assert gap_ratio > 5, (
            f"Factor model eigenvalue gap ratio (λ₁/λ₂) = {gap_ratio:.2f}; "
            f"expected > 5 (dominant single market factor)"
        )


# ---------------------------------------------------------------------------
# Tests: evaluate() with structural / NLL metrics from config
# ---------------------------------------------------------------------------

class TestEvaluateWithDiagnostics:
    """Verify that evaluate() respects the nested evaluation config dict
    and correctly returns structural and NLL metric keys."""

    N, T, F = 4, 5, 1

    def _make_grw(self, **eval_overrides):
        """Create a GRW with specific evaluation settings."""
        eval_cfg = {
            "structural_metrics": False,
            "nll_metrics": False,
            "structural_max_lag": 2,
            "structural_n_eigenvalues": 3,
            "log_every_n_batches": 0,  # quiet
        }
        eval_cfg.update(eval_overrides)
        return GeometricRandomWalk(
            device="cpu", n_samples=3, evaluation=eval_cfg,
        )

    def _fit_and_evaluate(self, grw, n_batches=3):
        loader = _make_loader(n_batches=n_batches, B=2, N=self.N, T=self.T)
        grw.fit(loader)

        mock_task = MagicMock()

        def _prepare(data):
            B = data.num_graphs
            samples = data.y.squeeze(-1) if data.y.dim() > 2 else data.y
            samples = samples.view(B, self.N, self.T, self.F).permute(0, 2, 1, 3)
            return {
                "samples": samples,
                "metadata": {
                    "batch_size": B,
                    "num_timesteps": self.T,
                    "num_stocks": self.N,
                    "num_features": self.F,
                    "close_price": data.close_price,
                    "close_price_y": data.close_price_y,
                    "stocks_index": data.stocks_index,
                    "timestamp": data.timestamp,
                },
            }

        mock_task.prepare_data.side_effect = _prepare
        mock_task.evaluate_samples.return_value = {"price_mse": 0.5}

        test_loader = _make_loader(n_batches=2, B=2, N=self.N, T=self.T)
        return grw.evaluate(test_loader, mock_task)

    # ----- backward compatibility -----

    def test_defaults_return_only_task_metrics(self):
        grw = self._make_grw()
        metrics = self._fit_and_evaluate(grw)
        assert "price_mse" in metrics
        assert "nll_mean_real" not in metrics
        assert "autocorr_lag1_real" not in metrics

    # ----- evaluation config parsing -----

    def test_evaluation_none_is_safe(self):
        """Passing evaluation=None should use all defaults (off)."""
        grw = GeometricRandomWalk(device="cpu", n_samples=3, evaluation=None)
        assert grw.structural_metrics is False
        assert grw.nll_metrics is False
        assert grw.log_every_n_batches == 10

    def test_evaluation_partial_dict(self):
        """Only the provided keys override defaults."""
        grw = GeometricRandomWalk(
            device="cpu", n_samples=3,
            evaluation={"nll_metrics": True},
        )
        assert grw.nll_metrics is True
        assert grw.structural_metrics is False
        assert grw.structural_max_lag == 2

    # ----- structural metrics -----

    def test_structural_metrics_keys(self):
        grw = self._make_grw(structural_metrics=True)
        metrics = self._fit_and_evaluate(grw)

        # Task metric still present
        assert "price_mse" in metrics

        # Autocorrelation keys for lags 1..2
        for l in (1, 2):
            assert f"autocorr_lag{l}_real" in metrics, f"Missing autocorr_lag{l}_real"
            assert f"autocorr_lag{l}_gen" in metrics
            assert f"autocorr_lag{l}_diff" in metrics

        # Eigenvalue keys for ranks 1..3
        for i in (1, 2, 3):
            assert f"eigenvalue_{i}_real" in metrics, f"Missing eigenvalue_{i}_real"
            assert f"eigenvalue_{i}_gen" in metrics
            assert f"eigenvalue_{i}_ratio" in metrics

    def test_structural_eigenvalues_positive(self):
        grw = self._make_grw(structural_metrics=True)
        metrics = self._fit_and_evaluate(grw)
        for i in (1, 2, 3):
            assert metrics[f"eigenvalue_{i}_real"] > 0
            assert metrics[f"eigenvalue_{i}_gen"] > 0

    # ----- NLL metrics -----

    def test_nll_metrics_keys(self):
        grw = self._make_grw(nll_metrics=True)
        metrics = self._fit_and_evaluate(grw)

        assert "price_mse" in metrics
        for key in ("nll_mean_real", "nll_mean_gen",
                     "nll_std_real", "nll_std_gen", "nll_w1"):
            assert key in metrics, f"Missing NLL key: {key}"

    def test_nll_values_positive(self):
        grw = self._make_grw(nll_metrics=True)
        metrics = self._fit_and_evaluate(grw)
        assert metrics["nll_mean_real"] > 0
        assert metrics["nll_mean_gen"] > 0
        assert metrics["nll_w1"] >= 0

    # ----- both together -----

    def test_both_metrics_together(self):
        grw = self._make_grw(structural_metrics=True, nll_metrics=True)
        metrics = self._fit_and_evaluate(grw)

        assert "autocorr_lag1_real" in metrics
        assert "eigenvalue_1_real" in metrics
        assert "nll_mean_real" in metrics
        assert "nll_w1" in metrics

    # ----- get_params includes evaluation settings -----

    def test_get_params_includes_evaluation(self):
        grw = self._make_grw(structural_metrics=True, nll_metrics=True)
        loader = _make_loader(n_batches=2, B=2, N=self.N, T=self.T)
        grw.fit(loader)
        params = grw.get_params()
        assert "evaluation" in params
        assert params["evaluation"]["structural_metrics"] is True
        assert params["evaluation"]["nll_metrics"] is True
        assert params["evaluation"]["structural_max_lag"] == 2
        assert params["evaluation"]["log_every_n_batches"] == 0


# ---------------------------------------------------------------------------
# Tests: static structural helpers on the class
# ---------------------------------------------------------------------------

class TestClassStaticHelpers:
    """Verify that the static helpers ported to GeometricRandomWalk
    produce the same results as the module-level test helpers."""

    B, T, N = 200, 20, 8

    def test_autocorr_profile_matches(self):
        torch.manual_seed(0)
        r = torch.randn(self.B, self.T, self.N)
        module_result = compute_autocorr_profile(r, max_lag=3)
        class_result = GeometricRandomWalk.compute_autocorr_profile(r, max_lag=3)
        assert torch.allclose(module_result, class_result, atol=1e-6)

    def test_top_eigenvalues_matches(self):
        torch.manual_seed(1)
        r = torch.randn(self.B, self.T, self.N)
        module_result = compute_top_eigenvalues(r, k=4)
        class_result = GeometricRandomWalk.compute_top_eigenvalues(r, k=4)
        assert torch.allclose(module_result, class_result, atol=1e-6)

    def test_compute_structural_metrics_keys(self):
        torch.manual_seed(2)
        real = torch.randn(self.B, self.T, self.N)
        gen = torch.randn(self.B, self.T, self.N)
        result = GeometricRandomWalk.compute_structural_metrics(
            real, gen, max_lag=2, n_eigenvalues=3,
        )
        expected_keys = set()
        for l in (0, 1, 2):
            for prefix in ("", "ptraj_"):
                expected_keys |= {
                    f"{prefix}autocorr_lag{l}_real",
                    f"{prefix}autocorr_lag{l}_gen",
                    f"{prefix}autocorr_lag{l}_diff",
                }
        for i in (1, 2, 3):
            for prefix in ("", "ptraj_"):
                expected_keys |= {
                    f"{prefix}eigenvalue_{i}_real",
                    f"{prefix}eigenvalue_{i}_gen",
                    f"{prefix}eigenvalue_{i}_ratio",
                }
        assert set(result.keys()) == expected_keys


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _build_parser():
    import argparse

    p = argparse.ArgumentParser(
        description=(
            "Compute and plot GRW NLL histograms for synthetic data.\n\n"
            "Generates two sets of trajectories:\n"
            "  r_real  – drawn directly from the true per-stock Gaussians\n"
            "  r_GRW   – sampled via the fitted GRW model\n\n"
            "Both are scored under NLL_GRW and their histograms are overlaid."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--n-stocks", type=int, default=10, metavar="N",
        help="Number of stocks (default: %(default)s)",
    )
    p.add_argument(
        "--n-steps", type=int, default=20, metavar="T",
        help="Forecast horizon / trajectory length (default: %(default)s)",
    )
    p.add_argument(
        "--n-trajectories", type=int, default=5_000, metavar="B",
        help="Number of trajectories per split (default: %(default)s)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Global random seed (default: %(default)s)",
    )
    p.add_argument(
        "--mean-range", type=float, nargs=2, default=[-0.25, 0.25],
        metavar=("LO", "HI"),
        help="Range for per-stock true means (default: %(default)s)",
    )
    p.add_argument(
        "--std-range", type=float, nargs=2, default=[0.8, 1.2],
        metavar=("LO", "HI"),
        help="Range for per-stock true std-devs (default: %(default)s)",
    )
    p.add_argument(
        "--bins", type=int, default=60,
        help="Number of histogram bins (default: %(default)s)",
    )
    p.add_argument(
        "--out-dir", type=str, default=".",
        metavar="DIR",
        help="Directory to save the histogram PNG (default: current directory)",
    )
    p.add_argument(
        "--dpi", type=int, default=150,
        help="Figure DPI (default: %(default)s)",
    )
    p.add_argument(
        "--max-lag", type=int, default=10, metavar="L",
        help="Maximum lag for autocorrelation profile (default: %(default)s)",
    )
    p.add_argument(
        "--n-eigenvalues", type=int, default=10, metavar="K",
        help="Number of top eigenvalues to show in spectrum plot (default: %(default)s)",
    )
    return p


def main():
    import argparse
    import os
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = _build_parser()
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    N = args.n_stocks
    T = args.n_steps
    B = args.n_trajectories

    true_means = torch.linspace(args.mean_range[0], args.mean_range[1], N)
    true_stds = torch.linspace(args.std_range[0], args.std_range[1], N)

    grw = _make_grw_with_known_params(true_means, true_stds)

    # --- r_real: draw from the true distribution ---------------------------
    gen_real = torch.Generator()
    gen_real.manual_seed(args.seed + 1)
    r_real = _sample_from_params(B, T, true_means, true_stds, gen_real)
    nll_real = grw.compute_nll(r_real).numpy()

    # --- r_GRW: sample from the fitted model --------------------------------
    r_grw = grw._sample_returns(B, T, N).squeeze(-1)  # [B, T, N]
    nll_grw = grw.compute_nll(r_grw).numpy()

    # --- Theoretical expected NLL -------------------------------------------
    expected_mean = T * sum(
        0.5 * math.log(2 * math.pi * math.e * true_stds[s].item() ** 2)
        for s in range(N)
    )

    # --- Print summary -------------------------------------------------------
    print(f"Settings:  N={N} stocks,  T={T} steps,  B={B} trajectories,  seed={args.seed}")
    print(f"True means: [{true_means[0]:.3f} … {true_means[-1]:.3f}]")
    print(f"True stds:  [{true_stds[0]:.3f} … {true_stds[-1]:.3f}]")
    print()
    print(f"Theoretical E[NLL]     = {expected_mean:.3f}")
    print(f"Empirical  E[NLL_real] = {nll_real.mean():.3f}  ±  {nll_real.std():.3f}")
    print(f"Empirical  E[NLL_GRW]  = {nll_grw.mean():.3f}  ±  {nll_grw.std():.3f}")

    from scipy.stats import wasserstein_distance
    import numpy as np

    rel_diff_mean = abs(nll_real.mean() - nll_grw.mean()) / abs(nll_real.mean())
    rel_diff_std = abs(nll_real.std() - nll_grw.std()) / abs(nll_real.std())
    w1 = wasserstein_distance(nll_real, nll_grw)
    rel_w1 = w1 / abs(nll_real.mean())
    print()
    print(f"Mean relative difference:  {rel_diff_mean * 100:.2f}%  (threshold 2%)")
    print(f"Std  relative difference:  {rel_diff_std  * 100:.2f}%  (threshold 5%)")
    print(f"Wasserstein-1 distance:    {w1:.3f} nats  ({rel_w1 * 100:.2f}% of mean NLL)")
    print()
    print("W1 = integral |F_real(x) - F_GRW(x)| dx  (L1 norm of empirical CDF difference)")

    # --- Build empirical CDFs for visualisation ------------------------------
    all_vals = np.concatenate([nll_real, nll_grw])
    x_grid = np.linspace(all_vals.min(), all_vals.max(), 2000)

    def _ecdf(samples: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Evaluate the empirical CDF of *samples* at each point in *x*."""
        return np.searchsorted(np.sort(samples), x, side="right") / len(samples)

    cdf_real = _ecdf(nll_real, x_grid)
    cdf_grw = _ecdf(nll_grw, x_grid)
    cdf_diff = np.abs(cdf_real - cdf_grw)

    # --- Plot ----------------------------------------------------------------
    fig, (ax_hist, ax_cdf) = plt.subplots(
        1, 2, figsize=(14, 5),
        gridspec_kw={"width_ratios": [1.2, 1]},
    )

    # Left panel — density histograms
    ax_hist.hist(
        nll_real, bins=args.bins, density=True, alpha=0.55,
        label=(
            r"$\mathrm{NLL}_{\mathrm{GRW}}(r^{\mathrm{real}})$  "
            f"  μ={nll_real.mean():.1f},  σ={nll_real.std():.1f}"
        ),
        color="steelblue",
    )
    ax_hist.hist(
        nll_grw, bins=args.bins, density=True, alpha=0.55,
        label=(
            r"$\mathrm{NLL}_{\mathrm{GRW}}(r^{\mathrm{GRW}})$  "
            f"  μ={nll_grw.mean():.1f},  σ={nll_grw.std():.1f}"
        ),
        color="darkorange",
    )
    ax_hist.axvline(
        expected_mean, color="black", linestyle="--", linewidth=1.5,
        label=f"Theoretical $E[\\mathrm{{NLL}}]$ = {expected_mean:.1f}",
    )
    ax_hist.set_xlabel(
        r"Per-trajectory NLL  $= \sum_{s,t} -\log\,\mathcal{N}(r_{s,t}\mid\tilde{\mu}_s,\tilde{\sigma}_s^2)$",
        fontsize=11,
    )
    ax_hist.set_ylabel("Density", fontsize=11)
    ax_hist.set_title(
        f"NLL Histograms  (N={N}, T={T}, B={B})",
        fontsize=12,
    )
    ax_hist.legend(fontsize=9)

    # Right panel — empirical CDFs + |ΔCDF| shaded (= W1 integrand)
    ax_cdf.plot(x_grid, cdf_real, color="steelblue", linewidth=1.8,
                label=r"$F_{\mathrm{real}}$")
    ax_cdf.plot(x_grid, cdf_grw, color="darkorange", linewidth=1.8,
                label=r"$F_{\mathrm{GRW}}$")
    ax_cdf.fill_between(
        x_grid, cdf_real, cdf_grw,
        alpha=0.25, color="mediumpurple",
        label=fr"$|F_{{\rm real}} - F_{{\rm GRW}}|$  (area = W$_1$ = {w1:.2f})",
    )
    ax_cdf.set_xlabel(
        r"Per-trajectory NLL  $\ell$",
        fontsize=11,
    )
    ax_cdf.set_ylabel("Empirical CDF", fontsize=11)
    ax_cdf.set_title(
        fr"Empirical CDFs  —  $W_1 = \int |F_{{\rm real}} - F_{{\rm GRW}}|\,dx = {w1:.2f}$ nats",
        fontsize=12,
    )
    ax_cdf.legend(fontsize=9)
    ax_cdf.set_ylim(0, 1)

    fig.tight_layout()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "nll_histogram_grw.pdf")
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    print(f"\nHistogram saved to: {out_path}")

    # =========================================================================
    # Cumulative-return NLL histograms (2-row grid, one column per horizon)
    # =========================================================================
    cum_nll_real_bt = grw.compute_cumulative_nll(r_real).numpy()  # [B, T]
    cum_nll_grw_bt  = grw.compute_cumulative_nll(r_grw).numpy()
    cum_nll_path = os.path.join(args.out_dir, "cumulative_nll_histogram_grw.pdf")
    w1_per_t = GeometricRandomWalk._plot_cumulative_nll_histograms(
        cum_nll_real_bt, cum_nll_grw_bt,
        save_path=cum_nll_path,
        bins=args.bins,
        dpi=args.dpi,
    )
    print(f"Cumulative NLL histogram saved to: {cum_nll_path}")
    if w1_per_t:
        for ti, w in enumerate(w1_per_t):
            print(f"  W1(t={ti+1}) = {w:.4f} nats")

    # =========================================================================
    # Cumulative log-return std vs prediction horizon
    # =========================================================================
    sigma_bar = float(np.mean(
        [grw.per_stock_stds.get(s, grw.global_std) for s in range(N)]
    ))
    theoretical_std = sigma_bar * np.sqrt(np.arange(1, T + 1))
    cum_std_path = os.path.join(args.out_dir, "cumulative_return_std_grw.pdf")
    GeometricRandomWalk._plot_cumulative_return_std(
        GeometricRandomWalk.compute_cumulative_return_std(r_real).numpy(),
        GeometricRandomWalk.compute_cumulative_return_std(r_grw).numpy(),
        theoretical_std,
        save_path=cum_std_path,
        std_real_tn=GeometricRandomWalk.compute_cumulative_return_std(r_real, per_stock=True).numpy(),
        std_gen_tn=GeometricRandomWalk.compute_cumulative_return_std(r_grw, per_stock=True).numpy(),
        dpi=args.dpi,
    )
    print(f"Cumulative-return std figure saved to: {cum_std_path}")

    # =========================================================================
    # Structural metrics: lag-1 autocorrelation profile + eigenvalue spectrum
    # =========================================================================

    max_lag = args.max_lag
    k_eigs  = min(args.n_eigenvalues, N)  # can't ask for more than N eigenvalues

    # --- Autocorrelation profiles -------------------------------------------
    profile_real = compute_autocorr_profile(r_real, max_lag=max_lag)  # [L+1]
    profile_grw  = compute_autocorr_profile(r_grw,  max_lag=max_lag)  # [L+1]

    # --- Top-k eigenvalues --------------------------------------------------
    eigs_real = compute_top_eigenvalues(r_real, k=k_eigs).numpy()  # [K]
    eigs_grw  = compute_top_eigenvalues(r_grw,  k=k_eigs).numpy()  # [K]

    # --- Print structural summary -------------------------------------------
    print("\n--- Structural metrics ---")
    print(f"Lag-1 ACF of r²   real={profile_real[1]:.4f}   GRW={profile_grw[1]:.4f}")
    print(f"Top-1 eigenvalue       real={eigs_real[0]:.3f}       GRW={eigs_grw[0]:.3f}")
    if k_eigs >= 2:
        print(f"Top-2 eigenvalue       real={eigs_real[1]:.3f}       GRW={eigs_grw[1]:.3f}")
        print(f"Eigenvalue gap (λ₁/λ₂) real={eigs_real[0]/eigs_real[1]:.2f}        "
              f"GRW={eigs_grw[0]/eigs_grw[1]:.2f}")

    # --- Figure: structural metrics -----------------------------------------
    fig2, (ax_ac, ax_ev) = plt.subplots(1, 2, figsize=(14, 5))

    lags = np.arange(0, max_lag + 1)
    ax_ac.bar(
        lags - 0.2, profile_real.numpy(), width=0.38,
        color="steelblue", alpha=0.8,
        label=r"$r^{\mathrm{real}}$",
    )
    ax_ac.bar(
        lags + 0.2, profile_grw.numpy(), width=0.38,
        color="darkorange", alpha=0.8,
        label=r"$r^{\mathrm{GRW}}$",
    )
    ax_ac.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_ac.set_xlabel("Lag  $l$", fontsize=11)
    ax_ac.set_ylabel(r"ACF of $r^2$  $\rho_l = \gamma_l / \gamma_0$",
                     fontsize=11)
    ax_ac.set_title(
        r"ACF profile of $r^2$  "
        f"(N={N}, T={T}, B={B})",
        fontsize=12,
    )
    ax_ac.set_xticks(lags)
    ax_ac.legend(fontsize=10)
    ax_ac.set_ylim(
        min(profile_real.min().item(), profile_grw.min().item()) - 0.05,
        max(profile_real.max().item(), profile_grw.max().item()) + 0.08,
    )

    # Annotate lag-1 values
    ax_ac.annotate(
        f"{profile_real[1]:.3f}",
        xy=(lags[1] - 0.2, profile_real[1].item()),
        xytext=(0, 6), textcoords="offset points",
        ha="center", fontsize=8, color="steelblue",
    )
    ax_ac.annotate(
        f"{profile_grw[1]:.3f}",
        xy=(lags[1] + 0.2, profile_grw[1].item()),
        xytext=(0, 6), textcoords="offset points",
        ha="center", fontsize=8, color="darkorange",
    )

    # Right panel — eigenvalue spectrum
    idx = np.arange(1, k_eigs + 1)
    ax_ev.bar(
        idx - 0.2, eigs_real, width=0.38,
        color="steelblue", alpha=0.8,
        label=r"$r^{\mathrm{real}}$",
    )
    ax_ev.bar(
        idx + 0.2, eigs_grw, width=0.38,
        color="darkorange", alpha=0.8,
        label=r"$r^{\mathrm{GRW}}$",
    )
    ax_ev.axhline(1.0, color="black", linewidth=0.8, linestyle="--",
                  label="Marchenko-Pastur mean (=1)")
    ax_ev.set_xlabel("Eigenvalue rank", fontsize=11)
    ax_ev.set_ylabel(r"Eigenvalue  $\lambda_k$", fontsize=11)
    ax_ev.set_title(
        fr"Top-{k_eigs} eigenvalues of $\hat{{C}}$  "
        f"(N={N}, T={T}, B={B})",
        fontsize=12,
    )
    ax_ev.set_xticks(idx)
    ax_ev.legend(fontsize=10)

    fig2.tight_layout()
    struct_path = os.path.join(args.out_dir, "structural_metrics_grw.pdf")
    fig2.savefig(struct_path, dpi=args.dpi)
    plt.close(fig2)
    print(f"Structural metrics figure saved to: {struct_path}")

    # =========================================================================
    # Per-stock autocorrelation profiles
    # =========================================================================
    autocorr_path = os.path.join(args.out_dir, "autocorr_per_stock_grw.pdf")
    GeometricRandomWalk._plot_autocorr_per_stock(
        r_real, r_grw,
        save_path=autocorr_path,
        max_lag=max_lag,
        dpi=args.dpi,
    )
    print(f"Per-stock autocorr figure saved to: {autocorr_path}")


if __name__ == "__main__":
    main()




