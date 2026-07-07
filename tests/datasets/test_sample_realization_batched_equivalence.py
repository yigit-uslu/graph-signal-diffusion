import numpy as np

from graph_signal_diffusion.datasets.wra.channel import WirelessChannel


def _legacy_sample_realization_power(channel: WirelessChannel, num_timesteps: int) -> np.ndarray:
    """Reference implementation matching the original nested-loop logic."""
    m, n = channel.large_scale_fading.shape
    H_complex = np.zeros((m, n, num_timesteps), dtype=complex)

    for i in range(m):
        for j in range(n):
            rayleigh_coeff = channel._generate_rayleigh_fading(num_timesteps)
            H_complex[i, j, :] = channel.large_scale_fading[i, j] * rayleigh_coeff

    return np.abs(H_complex) ** 2


def test_sample_realization_batched_matches_legacy_with_fixed_rng():
    """Smoke test: batched Rayleigh realization should match legacy numerically."""
    tx_locations = np.array(
        [[-120.0, -120.0], [120.0, -120.0], [-120.0, 120.0], [120.0, 120.0], [0.0, 0.0]],
        dtype=float,
    )
    rx_locations = np.array(
        [[-95.0, -120.0], [120.0, -95.0], [-145.0, 120.0], [120.0, 95.0], [0.0, 25.0]],
        dtype=float,
    )

    channel = WirelessChannel(
        n_links=5,
        deployment_range=500.0,
        min_tx_rx_distance=10.0,
        max_tx_rx_distance=70.0,
        seed=123,
        skip_deployment=True,
        tx_locations=tx_locations,
        rx_locations=rx_locations,
    )

    num_timesteps = 64

    # Fix and snapshot RNG state so both implementations consume identical draws.
    np.random.seed(20260303)
    rng_state = np.random.get_state()
    expected_H = _legacy_sample_realization_power(channel, num_timesteps=num_timesteps)

    np.random.set_state(rng_state)
    actual = channel.sample_realization(num_timesteps=num_timesteps)
    actual_H = actual["H"]

    assert expected_H.shape == actual_H.shape
    assert np.allclose(actual_H, expected_H, rtol=1e-12, atol=1e-12), (
        f"Batched sample_realization deviates from legacy reference "
        f"(max abs diff={np.max(np.abs(actual_H - expected_H)):.3e})"
    )
