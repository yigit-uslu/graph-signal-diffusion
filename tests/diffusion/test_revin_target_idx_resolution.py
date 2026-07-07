"""
Tests for RevIN target feature index resolution from metadata.

Covers: top-level attr > canonical info key > legacy info key > config fallback,
batched tensors, mixed batch raises, exclude_keys simulation.
"""

import pytest
import torch

from graph_signal_diffusion.diffusion.ddpm import DDPM
import torch.nn as nn


class _Stub(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x, timesteps, edge_index=None, edge_weight=None,
                cond=None, return_intermediates=False):
        return x, None


def _make_ddpm(**kw):
    return DDPM(model=_Stub(), num_timesteps=10, revin=True, **kw)


class _FakeData:
    """Minimal stand-in for PyG Data with configurable attrs."""
    def __init__(self, info=None, target_column_idx=None):
        self.info = info
        if target_column_idx is not None:
            self.target_column_idx = target_column_idx


class TestResolveTargetIdx:

    def test_top_level_attr_wins_over_info(self):
        """data.target_column_idx (top-level) takes priority over info dict."""
        ddpm = _make_ddpm()
        data = _FakeData(
            target_column_idx=4,
            info={'target_feature_idx': 99, 'Target_column_idx': 0},
        )
        assert ddpm._resolve_revin_target_idx(data) == 4

    def test_top_level_batched_tensor(self):
        """PyG collates scalar attrs into tensors across batch."""
        ddpm = _make_ddpm()
        data = _FakeData(target_column_idx=torch.tensor([4, 4, 4]))
        assert ddpm._resolve_revin_target_idx(data) == 4

    def test_top_level_scalar_tensor(self):
        """Single-item tensor metadata should also resolve correctly."""
        ddpm = _make_ddpm()
        data = _FakeData(target_column_idx=torch.tensor(4))
        assert ddpm._resolve_revin_target_idx(data) == 4

    def test_top_level_mixed_batch_raises(self):
        ddpm = _make_ddpm()
        data = _FakeData(target_column_idx=torch.tensor([4, 2, 4]))
        with pytest.raises(ValueError, match="Mixed target feature indices"):
            ddpm._resolve_revin_target_idx(data)

    def test_canonical_info_over_legacy(self):
        """When top-level attr is absent, canonical info key wins over legacy."""
        ddpm = _make_ddpm()
        data = _FakeData(info={
            'target_feature_idx': 3,
            'Target_column_idx': 0,
        })
        assert ddpm._resolve_revin_target_idx(data) == 3

    def test_legacy_fallback(self):
        ddpm = _make_ddpm()
        data = _FakeData(info={'Target_column_idx': 2})
        assert ddpm._resolve_revin_target_idx(data) == 2

    def test_config_fallback(self):
        ddpm = _make_ddpm(revin_cond_feature_idx=7)
        data = _FakeData(info={})
        assert ddpm._resolve_revin_target_idx(data) == 7

    def test_config_fallback_no_info(self):
        ddpm = _make_ddpm(revin_cond_feature_idx=5)
        data = _FakeData(info=None)
        assert ddpm._resolve_revin_target_idx(data) == 5

    def test_batched_info_tensor(self):
        ddpm = _make_ddpm()
        data = _FakeData(info={
            'Target_column_idx': torch.tensor([2, 2, 2, 2])
        })
        assert ddpm._resolve_revin_target_idx(data) == 2

    def test_batched_info_tensor_single(self):
        ddpm = _make_ddpm()
        data = _FakeData(info={
            'Target_column_idx': torch.tensor([4])
        })
        assert ddpm._resolve_revin_target_idx(data) == 4

    def test_mixed_info_batch_raises(self):
        ddpm = _make_ddpm()
        data = _FakeData(info={
            'Target_column_idx': torch.tensor([0, 1, 0, 0])
        })
        with pytest.raises(ValueError, match="Mixed target feature indices"):
            ddpm._resolve_revin_target_idx(data)

    def test_list_metadata(self):
        ddpm = _make_ddpm()
        data = _FakeData(info={'Target_column_idx': [3, 3, 3]})
        assert ddpm._resolve_revin_target_idx(data) == 3

    def test_mixed_list_raises(self):
        ddpm = _make_ddpm()
        data = _FakeData(info={'Target_column_idx': [0, 2]})
        with pytest.raises(ValueError, match="Mixed target feature indices"):
            ddpm._resolve_revin_target_idx(data)

    def test_exclude_keys_simulation(self):
        """Simulates what happens when info is excluded from batching
        (as in SP500 dataloaders) but target_column_idx is a top-level attr."""
        ddpm = _make_ddpm(revin_cond_feature_idx=0)  # wrong fallback
        # After PyG batching with exclude_keys=["info"], info is absent
        # but target_column_idx survives as a batched tensor
        data = _FakeData(target_column_idx=torch.tensor([4, 4]))
        # Should NOT fall back to config (0), should use top-level attr (4)
        assert ddpm._resolve_revin_target_idx(data) == 4


class TestComputeRevinStatsBounds:

    def test_target_idx_out_of_bounds_raises(self):
        ddpm = _make_ddpm()
        cond = torch.randn(2, 10, 4, 3)  # F_obs = 3
        with pytest.raises(IndexError, match="out of bounds"):
            ddpm._compute_revin_stats(cond, target_idx=5)

    def test_target_idx_negative_raises(self):
        ddpm = _make_ddpm()
        cond = torch.randn(2, 10, 4, 3)
        with pytest.raises(IndexError, match="out of bounds"):
            ddpm._compute_revin_stats(cond, target_idx=-1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
