"""Tests for per-sub-dataset r_min support in WRA datamodule and evaluator."""
import math
import pytest

from graph_signal_diffusion.datasets.wra.datamodule import _merge_r_min_per_dataset


# ---------------------------------------------------------------------------
# _merge_r_min_per_dataset tests
# ---------------------------------------------------------------------------


class TestMergeRminPerDataset:
    """Tests for the module-level merge helper."""

    def test_disjoint_dicts(self):
        result = _merge_r_min_per_dataset(
            {"ds_a": 0.4},
            {"ds_b": 0.6},
            {"ds_c": 0.8},
        )
        assert result == {"ds_a": 0.4, "ds_b": 0.6, "ds_c": 0.8}

    def test_same_key_same_value(self):
        """Same dataset in multiple splits with identical r_min should merge fine."""
        result = _merge_r_min_per_dataset(
            {"ds_a": 0.5},
            {"ds_a": 0.5},
        )
        assert result == {"ds_a": 0.5}

    def test_conflict_raises(self):
        """Different r_min for same dataset across splits must raise."""
        with pytest.raises(ValueError, match="r_min conflict"):
            _merge_r_min_per_dataset(
                {"ds_a": 0.4},
                {"ds_a": 0.8},
            )

    def test_tolerance_allows_tiny_diff(self):
        """Float representation differences within tolerance should not conflict."""
        result = _merge_r_min_per_dataset(
            {"ds_a": 0.6},
            {"ds_a": 0.6 + 1e-10},
        )
        assert math.isclose(result["ds_a"], 0.6, rel_tol=1e-6)

    def test_tolerance_rejects_real_diff(self):
        """Actual different values (0.6 vs 0.61) must raise."""
        with pytest.raises(ValueError, match="r_min conflict"):
            _merge_r_min_per_dataset(
                {"ds_a": 0.6},
                {"ds_a": 0.61},
            )

    def test_empty_dicts(self):
        result = _merge_r_min_per_dataset({}, {}, {})
        assert result == {}

    def test_single_dict(self):
        result = _merge_r_min_per_dataset({"ds_a": 0.4, "ds_b": 0.8})
        assert result == {"ds_a": 0.4, "ds_b": 0.8}
