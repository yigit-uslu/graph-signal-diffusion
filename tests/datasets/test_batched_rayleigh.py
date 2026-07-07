"""Regression tests for batched Rayleigh power generation.

Verifies determinism, cross-chunk consistency, and statistical correctness
of ``WirelessChannel._generate_batched_rayleigh_power``.
"""

import numpy as np
import pytest

from graph_signal_diffusion.datasets.wra.channel import WirelessChannelV2


# Small network for fast tests.  seed=42 gives reliable single-attempt deployment.
SEED = 42
N_LINKS = 4
DEPLOYMENT_RANGE = 200.0
NUM_TIMESTEPS = 200


def _make_channel():
    return WirelessChannelV2(
        n_links=N_LINKS, deployment_range=DEPLOYMENT_RANGE, seed=SEED
    )


class TestBatchedRayleighPower:
    """Regression suite for _generate_batched_rayleigh_power."""

    def test_determinism(self):
        """Same RNG state must produce bitwise-identical output."""
        ch = _make_channel()

        state = ch._rng.get_state()
        result_a = ch._generate_batched_rayleigh_power(NUM_TIMESTEPS)

        ch._rng.set_state(state)
        result_b = ch._generate_batched_rayleigh_power(NUM_TIMESTEPS)

        np.testing.assert_array_equal(result_a, result_b)

    @pytest.mark.parametrize("chunk_size", [1, 3, 7, N_LINKS * N_LINKS])
    def test_cross_chunk_consistency(self, chunk_size):
        """Different chunk sizes must produce identical results up to FP rounding.

        chunk_size=1 processes each link individually (simplest batching);
        chunk_size=N*N processes everything in one shot.  Any indexing or
        RNG-order bug in the chunking logic will show up as a mismatch.

        Different batch dimensions cause BLAS to sum matrix products in a
        different order, so we allow rtol=1e-12 (well above the observed
        worst-case of ~6e-14).
        """
        ch = _make_channel()

        state = ch._rng.get_state()
        reference = ch._generate_batched_rayleigh_power(
            NUM_TIMESTEPS, chunk_size=N_LINKS * N_LINKS
        )

        ch._rng.set_state(state)
        chunked = ch._generate_batched_rayleigh_power(
            NUM_TIMESTEPS, chunk_size=chunk_size
        )

        np.testing.assert_allclose(
            chunked,
            reference,
            rtol=1e-12,
            atol=0,
            err_msg=f"chunk_size={chunk_size} diverged from single-chunk reference",
        )

    def test_output_shape(self):
        """Output shape must be (m, n, T)."""
        ch = _make_channel()
        result = ch._generate_batched_rayleigh_power(NUM_TIMESTEPS)
        assert result.shape == (N_LINKS, N_LINKS, NUM_TIMESTEPS)

    def test_non_negative_power(self):
        """Channel power must be non-negative everywhere."""
        ch = _make_channel()
        result = ch._generate_batched_rayleigh_power(NUM_TIMESTEPS)
        assert np.all(result >= 0)

    def test_mean_power_scales_with_large_scale_fading(self):
        """Time-averaged power per link should approximate large_scale_fading^2.

        For unit-mean Rayleigh envelope, E[|h|^2] = 1, so
        E[H_power[i,j,:]] ≈ large_scale_fading[i,j]^2.
        We use many timesteps and a generous tolerance (20%) since
        the sum-of-sinusoids model has finite N0.
        """
        ch = _make_channel()
        T_long = 5000
        result = ch._generate_batched_rayleigh_power(T_long)

        expected_power = ch.large_scale_fading ** 2
        empirical_mean = result.mean(axis=2)

        # Per-link relative error; skip near-zero entries to avoid division issues
        mask = expected_power > 1e-20
        rel_error = np.abs(empirical_mean[mask] - expected_power[mask]) / expected_power[mask]
        assert np.mean(rel_error) < 0.20, (
            f"Mean relative error {np.mean(rel_error):.3f} exceeds 20% tolerance"
        )

    def test_sample_realization_matches_direct_call(self):
        """sample_realization() must use _generate_batched_rayleigh_power."""
        ch = _make_channel()

        state = ch._rng.get_state()
        direct = ch._generate_batched_rayleigh_power(NUM_TIMESTEPS)

        ch._rng.set_state(state)
        via_api = ch.sample_realization(NUM_TIMESTEPS)['H']

        np.testing.assert_array_equal(direct, via_api)
