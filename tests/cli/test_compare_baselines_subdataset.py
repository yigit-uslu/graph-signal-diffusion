"""Tests for per-subdataset comparison outputs in compare_baselines pipeline."""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import pytest

from graph_signal_diffusion.cli.compare_baselines import (
    _build_short_name_to_r_min,
    _save_per_subdataset_comparison,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wra_record(dataset_name: str, r_min: float, n: int = 4, T: int = 3) -> dict:
    """Create a minimal WRA evaluation record."""
    return {
        "dataset_name": dataset_name,
        "r_min": r_min,
        "network_id": 0,
        "eval_batch_idx": 0,
        "gen_rates_per_slot": np.random.rand(T, n).astype(np.float32),
        "real_rates_per_slot": np.random.rand(T, n).astype(np.float32),
    }


def _make_results_df(
    baselines: list[str],
    splits: list[str],
    subdatasets: list[str],
    metrics: list[str],
) -> pd.DataFrame:
    """Build a synthetic results DataFrame with subdataset/ columns."""
    data: dict[str, list[float]] = {}
    for split in splits:
        for sd in subdatasets:
            for metric in metrics:
                col = f"{split}/subdataset/{sd}/{metric}"
                data[col] = [float(i + 1) for i in range(len(baselines))]
        # Also add a global metric
        data[f"{split}/global_rate"] = [1.0] * len(baselines)
    return pd.DataFrame(data, index=baselines)


# ---------------------------------------------------------------------------
# Tests: _build_short_name_to_r_min
# ---------------------------------------------------------------------------

class TestBuildShortNameToRmin:
    def test_basic_mapping(self) -> None:
        records = {
            "test": {
                "fp": [
                    _make_wra_record("scenario/wrpc_hAAA", 0.4),
                    _make_wra_record("scenario/wrpc_hBBB", 0.6),
                ],
            },
        }
        result = _build_short_name_to_r_min(records)
        assert result == {("test", "wrpc_hAAA"): 0.4, ("test", "wrpc_hBBB"): 0.6}

    def test_consistent_r_min_no_warning(self) -> None:
        """Same short_name with same r_min from multiple baselines — no warning."""
        records = {
            "test": {
                "fp": [_make_wra_record("s/hAAA", 0.5)],
                "ap": [_make_wra_record("s/hAAA", 0.5)],
            },
        }
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = _build_short_name_to_r_min(records)
        assert result == {("test", "hAAA"): 0.5}

    def test_inconsistent_r_min_warns_and_uses_min(self) -> None:
        """Same short_name with different r_min values must warn and use min."""
        records = {
            "test": {
                "fp": [_make_wra_record("s/hAAA", 0.6)],
                "ap": [_make_wra_record("s/hAAA", 0.4)],
            },
        }
        with pytest.warns(RuntimeWarning, match="inconsistent r_min"):
            result = _build_short_name_to_r_min(records)
        assert result[("test", "hAAA")] == pytest.approx(0.4)

    def test_split_aware_mapping(self) -> None:
        """Same short_name in different splits should be keyed separately."""
        records = {
            "val": {"fp": [_make_wra_record("s/hAAA", 0.3)]},
            "test": {"fp": [_make_wra_record("s/hAAA", 0.5)]},
        }
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = _build_short_name_to_r_min(records)
        assert result[("val", "hAAA")] == pytest.approx(0.3)
        assert result[("test", "hAAA")] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Tests: _save_per_subdataset_comparison
# ---------------------------------------------------------------------------

class TestSavePerSubdatasetComparison:
    def test_generates_md_and_csv(self, tmp_path: os.PathLike) -> None:
        """With subdataset columns, should generate .md and .csv files."""
        df = _make_results_df(
            baselines=["fp", "diffusion"],
            splits=["test"],
            subdatasets=["hAAA", "hBBB"],
            metrics=["mean_rate_generated", "rate_1pct_generated"],
        )
        wra_records = {
            "test": {
                "fp": [
                    _make_wra_record("sc/hAAA", 0.4),
                    _make_wra_record("sc/hBBB", 0.6),
                ],
            },
        }
        _save_per_subdataset_comparison(df, str(tmp_path), wra_records)

        md_path = tmp_path / "per_subdataset_comparison.md"
        csv_path = tmp_path / "per_subdataset_comparison.csv"
        assert md_path.exists()
        assert csv_path.exists()

        md_content = md_path.read_text()
        assert "hAAA" in md_content
        assert "hBBB" in md_content
        assert "r_min=0.40" in md_content
        assert "r_min=0.60" in md_content

    def test_preserves_split_dimension(self, tmp_path: os.PathLike) -> None:
        """Metrics from different splits must not be mixed."""
        df = _make_results_df(
            baselines=["fp"],
            splits=["val", "test"],
            subdatasets=["hAAA"],
            metrics=["mean_rate_generated"],
        )
        wra_records = {
            "val": {"fp": [_make_wra_record("s/hAAA", 0.5)]},
            "test": {"fp": [_make_wra_record("s/hAAA", 0.5)]},
        }
        _save_per_subdataset_comparison(df, str(tmp_path), wra_records)

        md_content = (tmp_path / "per_subdataset_comparison.md").read_text()
        assert "val / hAAA" in md_content
        assert "test / hAAA" in md_content

        csv_df = pd.read_csv(tmp_path / "per_subdataset_comparison.csv")
        splits_in_csv = set(csv_df["split"].unique())
        assert splits_in_csv == {"val", "test"}

    def test_skips_when_no_subdataset_columns(self, tmp_path: os.PathLike) -> None:
        """Single-subdataset run has no subdataset/ columns — nothing generated."""
        df = pd.DataFrame(
            {"test/global_rate": [1.0, 2.0]},
            index=["fp", "ap"],
        )
        _save_per_subdataset_comparison(df, str(tmp_path), {})

        assert not (tmp_path / "per_subdataset_comparison.md").exists()
        assert not (tmp_path / "per_subdataset_comparison.csv").exists()


# ---------------------------------------------------------------------------
# Tests: per-subdataset plot graceful degradation
# ---------------------------------------------------------------------------

class TestPerSubdatasetPlotGracefulDegradation:
    def test_missing_baseline_for_subdataset(self, tmp_path: os.PathLike) -> None:
        """When a baseline has records for only a subset of subdatasets, plotting
        should still succeed for all subdatasets without errors."""
        import matplotlib
        matplotlib.use("Agg")
        from graph_signal_diffusion.baselines.wireless_resource_allocation.viz import (
            plot_wra_percentile_evolution_comparison,
        )

        ds_x = "scenario/wrpc_hXXX"
        ds_y = "scenario/wrpc_hYYY"

        # Baseline A has records for both X and Y
        recs_a = [
            _make_wra_record(ds_x, 0.4),
            _make_wra_record(ds_y, 0.6),
        ]
        # Baseline B has records only for X
        recs_b = [
            _make_wra_record(ds_x, 0.4),
        ]

        all_records = {"baseline_a": recs_a, "baseline_b": recs_b}

        # Group by dataset_name
        from collections import defaultdict
        grouped: dict[str, dict[str, list]] = {}
        for bl_name, bl_recs in all_records.items():
            by_ds: dict[str, list] = defaultdict(list)
            for rec in bl_recs:
                by_ds[str(rec["dataset_name"])].append(rec)
            grouped[bl_name] = dict(by_ds)

        all_ds = {ds_x, ds_y}
        for ds_name in sorted(all_ds):
            short = ds_name.rsplit("/", 1)[-1]
            ds_recs: dict[str, list] = {}
            for bl_name, bl_ds_map in grouped.items():
                if ds_name in bl_ds_map:
                    ds_recs[bl_name] = bl_ds_map[ds_name]

            save_path = str(tmp_path / f"plot_{short}.pdf")
            # Should not raise even when baseline_b is missing for ds_y
            plot_wra_percentile_evolution_comparison(
                ds_recs,
                save_path=save_path,
                baseline_order=["baseline_a", "baseline_b"],
            )
            assert os.path.exists(save_path)
