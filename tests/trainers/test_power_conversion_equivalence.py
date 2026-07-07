"""
Smoke test: verify that the three power-conversion approaches used in
PrimalDualTrainer all produce identical transmitter-power tensors for
one-to-one TX-RX pairings (the only pairing type used in WRA).

The three approaches under test:
  1. receiver_to_transmitter_power(p_rx, assoc)          -- used in compute_lagrangian_loss
  2. scatter_add_ via associations_to_transmitter_index   -- used in per-epoch rate collection
  3. assoc @ p_rx  (direct matrix multiply)               -- used in collect_samples

Finding: approach 1 is a thin wrapper around approach 3 (see utils.py line 44).
All three are equivalent for bijective (permutation-matrix) associations.

Also tests the power-conversion formula unification:
    10 ** ((dBm - 30) / 10)  ==  10 ** (dBm / 10) / 1000
"""

import pytest
import torch
import numpy as np


# ---------------------------------------------------------------------------
# Helpers imported from the package
# ---------------------------------------------------------------------------
from graph_signal_diffusion.datasets.wra.utils import receiver_to_transmitter_power


def _associations_to_transmitter_index(
    assoc_list: list[torch.Tensor], device: torch.device
) -> torch.Tensor:
    """Build per-receiver global transmitter indices for batched scatter_add_.

    Each association matrix is shaped (num_tx, num_rx) with one active transmitter
    per receiver column (the one-to-one pairing used in WRA).
    """
    index_parts: list[torch.Tensor] = []
    tx_offset = 0
    for assoc in assoc_list:
        if assoc.dim() != 2:
            raise ValueError(f"association must be rank-2, got shape {tuple(assoc.shape)}")
        num_tx, _num_rx = assoc.shape
        # Per receiver column, choose the associated transmitter row.
        tx_idx_local = assoc.to(device=device).argmax(dim=0).to(dtype=torch.long)
        index_parts.append(tx_idx_local + tx_offset)
        tx_offset += num_tx
    return torch.cat(index_parts, dim=0) if index_parts else torch.empty(0, dtype=torch.long, device=device)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _random_permutation_matrix(n: int, seed: int = 0) -> torch.Tensor:
    """Return a random (n, n) permutation matrix."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    perm = torch.randperm(n, generator=rng)
    A = torch.zeros(n, n)
    A[torch.arange(n), perm] = 1.0
    return A


def _identity_assoc(n: int) -> torch.Tensor:
    return torch.eye(n)


# ---------------------------------------------------------------------------
# Power conversion formula equivalence
# ---------------------------------------------------------------------------

class TestPowerConversionFormula:
    """Verify that the two algebraically equivalent dBm→Watts formulae agree."""

    @pytest.mark.parametrize("dbm", [-30.0, 0.0, 10.0, 20.0, 30.0, -10.0])
    def test_dbm_to_watts_formulas_equivalent(self, dbm: float):
        """10**((dBm-30)/10)  ==  10**(dBm/10) / 1000"""
        formula_canonical = 10 ** ((dbm - 30) / 10)
        formula_alt = 10 ** (dbm / 10) / 1000
        assert abs(formula_canonical - formula_alt) < 1e-12, (
            f"dBm={dbm}: canonical={formula_canonical}, alt={formula_alt}"
        )

    def test_10dbm_equals_10mW(self):
        """Sanity: 10 dBm == 10 mW == 0.01 W."""
        watts = 10 ** ((10.0 - 30) / 10)
        assert abs(watts - 0.01) < 1e-12, f"Expected 0.01 W, got {watts}"

    def test_0dbm_equals_1mW(self):
        """Sanity: 0 dBm == 1 mW == 0.001 W."""
        watts = 10 ** ((0.0 - 30) / 10)
        assert abs(watts - 0.001) < 1e-12


# ---------------------------------------------------------------------------
# Receiver-to-transmitter power mapping equivalence
# ---------------------------------------------------------------------------

class TestRxToTxPowerEquivalence:
    """All three mapping methods must agree for one-to-one (permutation) associations."""

    def _method_matmul(self, assoc: torch.Tensor, p_rx: torch.Tensor) -> torch.Tensor:
        """Method 3 / Method 1 (both are associations @ p_rx)."""
        return assoc @ p_rx

    def _method_helper(self, assoc: torch.Tensor, p_rx: torch.Tensor) -> torch.Tensor:
        """Method 1: receiver_to_transmitter_power utility."""
        return receiver_to_transmitter_power(p_rx, assoc)

    def _method_scatter(self, assoc_list: list, p_rx_batched: torch.Tensor) -> torch.Tensor:
        """Method 2: scatter_add_ via associations_to_transmitter_index (batched)."""
        device = p_rx_batched.device
        transmitter_index = _associations_to_transmitter_index(assoc_list, device)
        num_tx = sum(a.shape[0] for a in assoc_list)
        p_tx = torch.zeros(num_tx, device=device)
        p_tx.scatter_add_(0, transmitter_index, p_rx_batched)
        return p_tx

    @pytest.mark.parametrize("n,seed", [(5, 0), (10, 1), (50, 42), (1, 99)])
    def test_identity_assoc(self, n: int, seed: int):
        """Identity associations: all three methods return identical receiver powers."""
        torch.manual_seed(seed)
        assoc = _identity_assoc(n)
        p_rx = torch.rand(n)

        p_tx_matmul = self._method_matmul(assoc, p_rx)
        p_tx_helper = self._method_helper(assoc, p_rx)
        p_tx_scatter = self._method_scatter([assoc], p_rx)

        assert torch.allclose(p_tx_matmul, p_rx), "Identity assoc should be a no-op"
        assert torch.allclose(p_tx_helper, p_tx_matmul, atol=1e-6)
        assert torch.allclose(p_tx_scatter, p_tx_matmul, atol=1e-6)

    @pytest.mark.parametrize("n,seed", [(5, 10), (10, 20), (50, 30)])
    def test_permutation_assoc_single_graph(self, n: int, seed: int):
        """Random permutation associations: method 1 == method 3."""
        torch.manual_seed(seed)
        assoc = _random_permutation_matrix(n, seed)
        p_rx = torch.rand(n)

        p_tx_helper = self._method_helper(assoc, p_rx)
        p_tx_matmul = self._method_matmul(assoc, p_rx)

        assert torch.allclose(p_tx_helper, p_tx_matmul, atol=1e-6), (
            f"Method 1 vs 3 disagree for n={n}, seed={seed}"
        )

    @pytest.mark.parametrize("batch_sizes,seed", [
        ([5, 5], 0),
        ([3, 7, 4], 1),
        ([10, 10, 10], 2),
    ])
    def test_batched_scatter_matches_per_graph(self, batch_sizes: list, seed: int):
        """Batched scatter_add_ result must equal concatenation of per-graph matmuls."""
        torch.manual_seed(seed)
        assoc_list = [_random_permutation_matrix(n, seed + i) for i, n in enumerate(batch_sizes)]
        p_rx_list = [torch.rand(n) for n in batch_sizes]
        p_rx_batched = torch.cat(p_rx_list)

        # Per-graph reference
        p_tx_ref = torch.cat([self._method_matmul(a, p) for a, p in zip(assoc_list, p_rx_list)])

        # Batched scatter
        p_tx_scatter = self._method_scatter(assoc_list, p_rx_batched)

        assert torch.allclose(p_tx_scatter, p_tx_ref, atol=1e-6), (
            f"Batched scatter disagrees with per-graph matmul for sizes={batch_sizes}"
        )

    def test_helper_is_matmul(self):
        """Explicitly confirm receiver_to_transmitter_power is associations @ p_rx."""
        n = 8
        torch.manual_seed(7)
        assoc = _random_permutation_matrix(n, seed=7)
        p_rx = torch.rand(n)

        from_helper = receiver_to_transmitter_power(p_rx, assoc)
        from_matmul = assoc @ p_rx

        assert torch.allclose(from_helper, from_matmul, atol=1e-7), (
            "receiver_to_transmitter_power must equal assoc @ p_rx"
        )


# ---------------------------------------------------------------------------
# Recommendation note (printed when run with -v)
# ---------------------------------------------------------------------------

def test_unified_approach_recommendation():
    """
    Documents the recommended unified approach for all three code paths.

    FINDING:
      - Method 1 (receiver_to_transmitter_power) == Method 3 (assoc @ p_rx) by definition.
      - Method 2 (scatter_add_) is equivalent for one-to-one pairings but requires the
        batched associations_to_transmitter_index helper and an extra zero-init tensor.

    RECOMMENDATION:
      Use receiver_to_transmitter_power(power_per_receiver_graph, associations) everywhere:
        * compute_lagrangian_loss  -- already uses this  ✓
        * collect_samples          -- currently uses bare assoc @ p_rx  (identical, but
                                      replace for clarity and single source of truth)
        * per-epoch rate collection (train_epoch) -- currently uses scatter_add_;
                                      simplify to a per-graph loop with the helper,
                                      or keep as-is with a comment explaining equivalence.

      Either approach is numerically identical for bijective associations.
    """
    assert True, "This test documents the recommendation; it always passes."
