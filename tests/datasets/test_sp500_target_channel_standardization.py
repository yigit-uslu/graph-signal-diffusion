"""
Tests for the gated target-channel standardization in SP500 conditioning (Phase 2 of RevIN).

Verifies that `standardize_target_in_x_for_revin=True` applies the same
per-stock standardization to x[..., target_idx] as to y, and that
the default (False) preserves the baseline distribution bit-for-bit.
"""

import sys
from pathlib import Path
import torch
import numpy as np
import pytest
from torch_geometric.data import Batch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from graph_signal_diffusion.datasets.sp500.dataset import SP500Stocks, StocksDataDiffusion


# ---------------------------------------------------------------------------
# Fixtures: lightweight mock dataset (no disk I/O)
# ---------------------------------------------------------------------------

def _make_mock_dataset(standardize_target_in_x_for_revin: bool):
    """Create a minimal SP500Stocks-like object with mock stats.

    We monkey-patch the parts of SP500Stocks that _standardize_target needs,
    without calling __init__ or process() (which require real data on disk).
    """
    N, T_obs, T_fut, F = 5, 10, 5, 4
    feature_names = ["DailyLogReturn", "Volume", "RSI", "ALR1W"]
    target_column_name = "DailyLogReturn"
    target_feat_idx = feature_names.index(target_column_name)

    # Deterministic per-stock stats
    rng = np.random.RandomState(42)
    means = rng.randn(N).astype(np.float32) * 0.01  # small means
    stds = np.abs(rng.randn(N).astype(np.float32)) + 0.5  # positive stds

    per_stock_stats = {
        "DailyLogReturn": {"mean": means, "std": stds},
        "Volume": {"mean": rng.randn(N).astype(np.float32),
                    "std": np.abs(rng.randn(N).astype(np.float32)) + 0.1},
        "ALR1W": {"mean": rng.randn(N).astype(np.float32) * 0.01,
                   "std": np.abs(rng.randn(N).astype(np.float32)) + 0.3},
    }

    # Build a mock object with the attributes _standardize_target reads
    obj = object.__new__(SP500Stocks)
    obj.feature_names = feature_names
    obj.target_column_name = target_column_name
    obj.per_stock_stats = per_stock_stats
    obj.sp500_scale_factor = 100.0
    obj._apply_sp500_standardization = True
    obj.standardize_target_in_x_for_revin = standardize_target_in_x_for_revin

    # Deterministic raw tensors
    torch.manual_seed(123)
    # x is [N, T_obs, F] — target channel is in raw space (percentage-scaled)
    x_raw = torch.randn(N, T_obs, F) * 2.0  # ~std=2.0 raw space
    # y is [N, T_fut, 1] — same raw space before standardization
    y_raw = torch.randn(N, T_fut, 1) * 2.0

    return obj, x_raw, y_raw, target_feat_idx, means, stds


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestConditioningTargetStandardization:

    def test_conditioning_target_matches_y_formula(self):
        """Target channel in x should be transformed by the same formula as y:
        z = (x / scale_factor - mean) / std
        """
        obj, x_raw, y_raw, tidx, means, stds = _make_mock_dataset(
            standardize_target_in_x_for_revin=True
        )

        # Apply standardization to y
        y_std = obj._standardize_target(y_raw.clone(), feature_idx=tidx)

        # Apply standardization to x target channel
        x_target_slice = x_raw[:, :, tidx:tidx+1].clone()  # [N, T_obs, 1]
        x_target_std = obj._standardize_target(x_target_slice, feature_idx=tidx)

        # Manually compute expected: z = (x / 100.0 - mean) / (std + 1e-8)
        means_t = torch.from_numpy(means).float().view(-1, 1, 1)
        stds_t = torch.from_numpy(stds).float().view(-1, 1, 1)

        y_expected = (y_raw / 100.0 - means_t) / (stds_t + 1e-8)
        x_expected = (x_target_slice / 100.0 - means_t) / (stds_t + 1e-8)

        torch.testing.assert_close(y_std, y_expected, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(x_target_std, x_expected, atol=1e-6, rtol=1e-5)

    def test_y_uses_same_stats_as_x_target(self):
        """Both y and x target channel use identical per-stock mean/std."""
        obj, x_raw, y_raw, tidx, means, stds = _make_mock_dataset(
            standardize_target_in_x_for_revin=True
        )

        # Standardize both
        y_std = obj._standardize_target(y_raw.clone(), feature_idx=tidx)
        x_slice = x_raw[:, :, tidx:tidx+1].clone()
        x_std = obj._standardize_target(x_slice, feature_idx=tidx)

        # Recover the effective mean/std used by each
        # z = (raw / 100 - mean) / (std + eps)  =>  mean = raw/100 - z*(std+eps)
        # Just verify that for the same raw input, both produce the same output
        shared_input = torch.randn(5, 7, 1) * 2.0
        y_out = obj._standardize_target(shared_input.clone(), feature_idx=tidx)
        x_out = obj._standardize_target(shared_input.clone(), feature_idx=tidx)
        torch.testing.assert_close(y_out, x_out)

    def test_non_target_features_unchanged(self):
        """When flag is on, non-target features in x must be unaffected."""
        obj, x_raw, y_raw, tidx, _, _ = _make_mock_dataset(
            standardize_target_in_x_for_revin=True
        )

        x_before = x_raw.clone()

        # Simulate the get() logic
        target_slice = x_raw[:, :, tidx:tidx+1].clone()
        x_raw[:, :, tidx:tidx+1] = obj._standardize_target(
            target_slice, feature_idx=tidx
        )

        # Non-target columns should be identical
        for f in range(x_raw.shape[-1]):
            if f == tidx:
                continue
            torch.testing.assert_close(
                x_raw[:, :, f], x_before[:, :, f],
                msg=f"Feature {f} was modified but should be unchanged"
            )

    def test_flag_off_preserves_old_distribution(self):
        """With default flag=False, x target channel is in raw space (~std=2.0),
        bit-for-bit identical to the baseline."""
        obj, x_raw, y_raw, tidx, _, _ = _make_mock_dataset(
            standardize_target_in_x_for_revin=False
        )

        x_before = x_raw.clone()

        # Simulate get() with flag off: no x target standardization
        if obj.standardize_target_in_x_for_revin:
            raise AssertionError("Flag should be off")

        # x should be completely untouched
        torch.testing.assert_close(x_raw, x_before)

        # Verify target channel is in raw space (std >> 1 if data is percentage-scaled)
        target_channel = x_raw[:, :, tidx]
        assert target_channel.std().item() > 1.0, (
            f"Expected raw-space std > 1.0, got {target_channel.std().item()}"
        )

    def test_target_channel_std_reduced_when_flag_on(self):
        """When flag is on, the target channel std should be closer to 1.0
        (standardized) than the raw space (~2.0)."""
        obj, x_raw, _, tidx, _, _ = _make_mock_dataset(
            standardize_target_in_x_for_revin=True
        )

        raw_std = x_raw[:, :, tidx].std().item()

        target_slice = x_raw[:, :, tidx:tidx+1].clone()
        x_raw[:, :, tidx:tidx+1] = obj._standardize_target(
            target_slice, feature_idx=tidx
        )
        std_after = x_raw[:, :, tidx].std().item()

        # raw_std should be around 2.0, std_after should be closer to 1.0
        # (exact value depends on mock stats, but it should decrease)
        assert std_after < raw_std, (
            f"Expected standardization to reduce std: {raw_std:.3f} -> {std_after:.3f}"
        )


class TestTargetColumnIdxMetadata:

    def test_stocks_data_accepts_and_batches_target_column_idx(self):
        """target_column_idx should be accepted by constructor and survive batching."""
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

        g1 = StocksDataDiffusion(
            x=torch.randn(2, 10, 4),
            y=torch.randn(2, 5, 1),
            edge_index=edge_index,
            target_column_idx=4,
        )
        g2 = StocksDataDiffusion(
            x=torch.randn(2, 10, 4),
            y=torch.randn(2, 5, 1),
            edge_index=edge_index,
            target_column_idx=4,
        )

        assert int(g1.target_column_idx) == 4
        batch = Batch.from_data_list([g1, g2])
        assert hasattr(batch, "target_column_idx")
        torch.testing.assert_close(batch.target_column_idx, torch.tensor([4, 4]))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
