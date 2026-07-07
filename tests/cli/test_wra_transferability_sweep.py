"""Tests for WRA size/density transferability sweep helpers."""
from __future__ import annotations

import json
import math
import os
import os.path as osp

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeChannel:
    """Minimal picklable stand-in for WirelessChannel."""

    def __init__(self, *, n_links: int, seed: int):
        self.large_scale_fading = np.random.rand(n_links, n_links).astype(np.float32)
        self.associations = np.eye(n_links, dtype=np.float32)
        self.seed = seed


def _make_mock_channel(*, n_links: int, seed: int):
    """Create a minimal mock WirelessChannel with required attributes."""
    return _FakeChannel(n_links=n_links, seed=seed)


# ---------------------------------------------------------------------------
# Tests: transfer_dataset_name
# ---------------------------------------------------------------------------


class TestTransferDatasetName:
    def test_base_name(self):
        from graph_signal_diffusion.cli._wra_transferability import (
            transfer_dataset_name,
        )

        name = transfer_dataset_name(
            "large_outdoor_mid", "abcd1234efgh",
            eval_num_networks=8, base_r_min=0.6,
        )
        assert name == "transfer_large_outdoor_mid_abcd1234_n8_r0.60"

    def test_with_swept_r_min(self):
        from graph_signal_diffusion.cli._wra_transferability import (
            transfer_dataset_name,
        )

        name = transfer_dataset_name(
            "large_outdoor_mid", "abcd1234efgh",
            eval_num_networks=8, base_r_min=0.6, swept_r_min=0.4,
        )
        assert name == "transfer_large_outdoor_mid_abcd1234_n8_r0.60_rmin0.40"

    def test_different_eval_num_produces_different_name(self):
        from graph_signal_diffusion.cli._wra_transferability import (
            transfer_dataset_name,
        )

        name_8 = transfer_dataset_name(
            "sc", "key12345",
            eval_num_networks=8, base_r_min=0.6,
        )
        name_16 = transfer_dataset_name(
            "sc", "key12345",
            eval_num_networks=16, base_r_min=0.6,
        )
        assert name_8 != name_16

    def test_different_base_r_min_produces_different_name(self):
        from graph_signal_diffusion.cli._wra_transferability import (
            transfer_dataset_name,
        )

        name_a = transfer_dataset_name(
            "sc", "key12345",
            eval_num_networks=8, base_r_min=0.4,
        )
        name_b = transfer_dataset_name(
            "sc", "key12345",
            eval_num_networks=8, base_r_min=0.6,
        )
        assert name_a != name_b


# ---------------------------------------------------------------------------
# Tests: select_tail_network_ids
# ---------------------------------------------------------------------------


class TestSelectTailNetworkIds:
    def test_tail_selection(self):
        from graph_signal_diffusion.cli._wra_transferability import (
            select_tail_network_ids,
        )

        ids = select_tail_network_ids(32, 8)
        assert ids == list(range(24, 32))

    def test_full_selection(self):
        from graph_signal_diffusion.cli._wra_transferability import (
            select_tail_network_ids,
        )

        ids = select_tail_network_ids(8, 8)
        assert ids == list(range(8))

    def test_exceeds_total_raises(self):
        from graph_signal_diffusion.cli._wra_transferability import (
            select_tail_network_ids,
        )

        with pytest.raises(ValueError, match="eval_num_networks.*> total_networks"):
            select_tail_network_ids(4, 8)

    def test_zero_raises(self):
        from graph_signal_diffusion.cli._wra_transferability import (
            select_tail_network_ids,
        )

        with pytest.raises(ValueError, match="must be positive"):
            select_tail_network_ids(8, 0)

    def test_negative_raises(self):
        from graph_signal_diffusion.cli._wra_transferability import (
            select_tail_network_ids,
        )

        with pytest.raises(ValueError, match="must be positive"):
            select_tail_network_ids(8, -1)


# ---------------------------------------------------------------------------
# Tests: synthesize_transfer_raw_dir
# ---------------------------------------------------------------------------


class TestSynthesizeTransferRawDir:
    def test_creates_raw_directory(self, tmp_path):
        from graph_signal_diffusion.cli._wra_transferability import (
            synthesize_transfer_raw_dir,
        )

        channels = [_make_mock_channel(n_links=4, seed=100 + i) for i in range(8)]
        selected_ids = [6, 7]  # tail of 8
        ds_dir = str(tmp_path / "transfer_test")

        raw_dir = synthesize_transfer_raw_dir(
            channels=channels,
            selected_network_ids=selected_ids,
            dataset_dir=ds_dir,
            system_params={"P_max": 0.01, "noise_var": 1e-10, "BW": 10e6, "r_min": 0.6},
            channel_params={"version": "v2"},
            channel_version="v2",
            channel_cache_info_entry={"channel_cache_key": "test123", "channel_cache_file": "/tmp/test.pt"},
            scenario_name="test_scenario",
        )

        assert osp.isdir(raw_dir)
        assert osp.isfile(osp.join(raw_dir, "collected_samples.npz"))
        assert osp.isfile(osp.join(raw_dir, "network_info.json"))
        assert osp.isfile(osp.join(raw_dir, "config.yaml"))
        assert osp.isdir(osp.join(raw_dir, "h_instantaneous"))

    def test_npz_contains_renumbered_networks(self, tmp_path):
        from graph_signal_diffusion.cli._wra_transferability import (
            synthesize_transfer_raw_dir,
        )

        channels = [_make_mock_channel(n_links=4, seed=100 + i) for i in range(8)]
        selected_ids = [6, 7]
        ds_dir = str(tmp_path / "transfer_test")

        synthesize_transfer_raw_dir(
            channels=channels,
            selected_network_ids=selected_ids,
            dataset_dir=ds_dir,
            system_params={"P_max": 0.01, "noise_var": 1e-10, "BW": 10e6, "r_min": 0.6},
            channel_params={"version": "v2"},
            channel_version="v2",
            channel_cache_info_entry={},
            scenario_name="test",
        )

        npz_path = osp.join(ds_dir, "raw", "collected_samples.npz")
        with np.load(npz_path, allow_pickle=True) as npz:
            assert "network_0" in npz.files
            assert "network_1" in npz.files
            assert "network_6" not in npz.files  # renumbered, not original IDs

            net0 = npz["network_0"].item()
            assert "H_ls" in net0
            assert "associations" in net0
            assert "power_samples" in net0
            assert len(net0["power_samples"]) == 1

    def test_network_info_preserves_seeds(self, tmp_path):
        from graph_signal_diffusion.cli._wra_transferability import (
            synthesize_transfer_raw_dir,
        )

        channels = [_make_mock_channel(n_links=4, seed=100 + i) for i in range(8)]
        selected_ids = [6, 7]
        ds_dir = str(tmp_path / "transfer_test")

        synthesize_transfer_raw_dir(
            channels=channels,
            selected_network_ids=selected_ids,
            dataset_dir=ds_dir,
            system_params={"P_max": 0.01, "noise_var": 1e-10, "BW": 10e6, "r_min": 0.6},
            channel_params={},
            channel_version="v2",
            channel_cache_info_entry={},
            scenario_name="test",
        )

        info_path = osp.join(ds_dir, "raw", "network_info.json")
        with open(info_path) as f:
            info = json.load(f)

        # Seeds should be from cache IDs 6 and 7
        assert info["network_0"]["network_seed"] == 106
        assert info["network_1"]["network_seed"] == 107

    def test_idempotent(self, tmp_path):
        """Running synthesize twice should not error."""
        from graph_signal_diffusion.cli._wra_transferability import (
            synthesize_transfer_raw_dir,
        )

        channels = [_make_mock_channel(n_links=4, seed=42)]
        ds_dir = str(tmp_path / "transfer_idem")
        kwargs = dict(
            channels=channels,
            selected_network_ids=[0],
            dataset_dir=ds_dir,
            system_params={"P_max": 0.01, "noise_var": 1e-10, "BW": 10e6, "r_min": 0.5},
            channel_params={},
            channel_version="v2",
            channel_cache_info_entry={},
            scenario_name="test",
        )

        synthesize_transfer_raw_dir(**kwargs)
        synthesize_transfer_raw_dir(**kwargs)  # should not raise


# ---------------------------------------------------------------------------
# Tests: create_transfer_r_min_variant
# ---------------------------------------------------------------------------


class TestCreateTransferRMinVariant:
    def test_creates_symlinks_and_modified_info(self, tmp_path):
        from graph_signal_diffusion.cli._wra_transferability import (
            create_transfer_r_min_variant,
            synthesize_transfer_raw_dir,
        )

        channels = [_make_mock_channel(n_links=4, seed=42)]
        base_dir = str(tmp_path / "base")
        synthesize_transfer_raw_dir(
            channels=channels,
            selected_network_ids=[0],
            dataset_dir=base_dir,
            system_params={"P_max": 0.01, "noise_var": 1e-10, "BW": 10e6, "r_min": 0.6},
            channel_params={},
            channel_version="v2",
            channel_cache_info_entry={},
            scenario_name="test",
        )

        variant_dir = str(tmp_path / "variant_rmin0.40")
        create_transfer_r_min_variant(
            base_raw_dir=osp.join(base_dir, "raw"),
            variant_dataset_dir=variant_dir,
            r_min=0.4,
        )

        variant_raw = osp.join(variant_dir, "raw")
        assert osp.islink(osp.join(variant_raw, "collected_samples.npz"))
        assert osp.islink(osp.join(variant_raw, "config.yaml"))
        assert osp.islink(osp.join(variant_raw, "h_instantaneous"))
        assert not osp.islink(osp.join(variant_raw, "network_info.json"))

        with open(osp.join(variant_raw, "network_info.json")) as f:
            info = json.load(f)
        assert info["system_params"]["r_min"] == pytest.approx(0.4)
        assert info["network_0"]["system_params"]["r_min"] == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# Tests: find_compatible_channel_cache
# ---------------------------------------------------------------------------


class TestFindCompatibleCache:
    def test_finds_larger_cache(self, tmp_path):
        from graph_signal_diffusion.cli._wra_transferability import (
            find_compatible_channel_cache,
        )

        # Create a cache with 32 networks
        scenario_dir = tmp_path / "scenario_x"
        scenario_dir.mkdir()
        channels = [_make_mock_channel(n_links=4, seed=i) for i in range(32)]
        metadata = {
            "seed": 42,
            "n_links": 4,
            "deployment_range": 1000.0,
            "channel_version": "v2",
            "channel_params": {},
            "num_networks": 32,
            "channel_cache_key": "hash32",
        }
        torch.save(
            {"metadata": metadata, "channels": channels},
            scenario_dir / "hash32.pt",
        )

        result = find_compatible_channel_cache(
            scenario_name="scenario_x",
            cache_root=tmp_path,
            expected_metadata={
                "seed": 42,
                "n_links": 4,
                "deployment_range": 1000.0,
                "channel_version": "v2",
                "channel_params": {},
                "num_networks": 8,  # we only need 8
            },
            min_num_networks=8,
        )

        assert result is not None
        assert len(result.channels) == 32

    def test_returns_none_when_too_few(self, tmp_path):
        from graph_signal_diffusion.cli._wra_transferability import (
            find_compatible_channel_cache,
        )

        scenario_dir = tmp_path / "scenario_y"
        scenario_dir.mkdir()
        channels = [_make_mock_channel(n_links=4, seed=i) for i in range(4)]
        metadata = {
            "seed": 42,
            "n_links": 4,
            "deployment_range": 1000.0,
            "channel_version": "v2",
            "channel_params": {},
            "num_networks": 4,
            "channel_cache_key": "hash4",
        }
        torch.save(
            {"metadata": metadata, "channels": channels},
            scenario_dir / "hash4.pt",
        )

        result = find_compatible_channel_cache(
            scenario_name="scenario_y",
            cache_root=tmp_path,
            expected_metadata={
                "seed": 42,
                "n_links": 4,
                "deployment_range": 1000.0,
                "channel_version": "v2",
                "channel_params": {},
                "num_networks": 8,
            },
            min_num_networks=8,
        )

        assert result is None

    def test_returns_none_when_params_mismatch(self, tmp_path):
        from graph_signal_diffusion.cli._wra_transferability import (
            find_compatible_channel_cache,
        )

        scenario_dir = tmp_path / "scenario_z"
        scenario_dir.mkdir()
        channels = [_make_mock_channel(n_links=4, seed=i) for i in range(32)]
        metadata = {
            "seed": 42,
            "n_links": 4,  # different from expected 8
            "deployment_range": 1000.0,
            "channel_version": "v2",
            "channel_params": {},
            "num_networks": 32,
            "channel_cache_key": "hash32",
        }
        torch.save(
            {"metadata": metadata, "channels": channels},
            scenario_dir / "hash32.pt",
        )

        result = find_compatible_channel_cache(
            scenario_name="scenario_z",
            cache_root=tmp_path,
            expected_metadata={
                "seed": 42,
                "n_links": 8,  # different!
                "deployment_range": 1000.0,
                "channel_version": "v2",
                "channel_params": {},
                "num_networks": 8,
            },
            min_num_networks=8,
        )

        assert result is None


# ---------------------------------------------------------------------------
# Tests: AP exclusion guard
# ---------------------------------------------------------------------------


class TestAPExclusionGuard:
    def test_rejects_ap_in_expert_free_mode(self):
        from graph_signal_diffusion.cli._wra_transferability import (
            validate_transfer_baselines,
        )

        with pytest.raises(ValueError, match="ap"):
            validate_transfer_baselines(["fp", "ap", "diffusion"], reference_policy="none")

    def test_allows_non_ap_baselines(self):
        from graph_signal_diffusion.cli._wra_transferability import (
            validate_transfer_baselines,
        )

        # Should not raise
        validate_transfer_baselines(["fp", "wmmse", "diffusion"], reference_policy="none")


# ---------------------------------------------------------------------------
# Tests: save_transferability_summary
# ---------------------------------------------------------------------------


class TestSaveTransferabilitySummary:
    def test_generates_csv_and_md(self, tmp_path):
        from graph_signal_diffusion.cli._wra_transferability import (
            save_transferability_summary,
        )

        results = [
            {"target_scenario": "large_mid", "baseline": "fp", "r_min": 0.6,
             "mean_rate_generated": 2.5, "rate_1pct_generated": 0.8},
            {"target_scenario": "large_mid", "baseline": "diffusion", "r_min": 0.6,
             "mean_rate_generated": 3.1, "rate_1pct_generated": 1.2},
            {"target_scenario": "large_high", "baseline": "fp", "r_min": 0.6,
             "mean_rate_generated": 1.8, "rate_1pct_generated": 0.5},
        ]

        save_transferability_summary(
            sweep_results=results,
            output_dir=str(tmp_path),
        )

        csv_path = tmp_path / "transferability_sweep_summary.csv"
        md_path = tmp_path / "transferability_sweep_summary.md"
        assert csv_path.exists()
        assert md_path.exists()

        import pandas as pd
        df = pd.read_csv(csv_path)
        assert len(df) == 3


# ---------------------------------------------------------------------------
# Tests: plot_transferability_results
# ---------------------------------------------------------------------------


class TestPlotTransferabilityResults:
    def test_generates_plot_files(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")

        from graph_signal_diffusion.cli._wra_transferability import (
            plot_transferability_results,
        )

        results = [
            {"target_scenario": "large_mid", "baseline": "fp", "r_min": 0.6,
             "mean_rate_generated": 2.5, "rate_1pct_generated": 0.8,
             "rate_5pct_generated": 1.0, "rate_violation_percentage_generated": 10.0,
             "rate_mean_violation_gap_pct_generated": 5.0},
            {"target_scenario": "large_mid", "baseline": "diffusion", "r_min": 0.6,
             "mean_rate_generated": 3.1, "rate_1pct_generated": 1.2,
             "rate_5pct_generated": 1.5, "rate_violation_percentage_generated": 3.0,
             "rate_mean_violation_gap_pct_generated": 1.5},
            {"target_scenario": "large_high", "baseline": "fp", "r_min": 0.6,
             "mean_rate_generated": 1.8, "rate_1pct_generated": 0.5,
             "rate_5pct_generated": 0.7, "rate_violation_percentage_generated": 20.0,
             "rate_mean_violation_gap_pct_generated": 10.0},
            {"target_scenario": "large_high", "baseline": "diffusion", "r_min": 0.6,
             "mean_rate_generated": 2.2, "rate_1pct_generated": 0.9,
             "rate_5pct_generated": 1.1, "rate_violation_percentage_generated": 8.0,
             "rate_mean_violation_gap_pct_generated": 4.0},
        ]

        plot_transferability_results(
            sweep_results=results,
            output_dir=str(tmp_path),
            r_min_values=[0.6],
        )

        plot_dir = tmp_path / "transferability_sweep_plots"
        assert plot_dir.exists()
        # Combined plot (single r_min, no suffix)
        assert (plot_dir / "transferability_sweep_all_metrics.pdf").exists()
        # Individual plots
        assert (plot_dir / "transferability_sweep_mean_rate_generated.pdf").exists()

    def test_2d_sweep_generates_per_rmin_plots(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")

        from graph_signal_diffusion.cli._wra_transferability import (
            plot_transferability_results,
        )

        results = [
            {"target_scenario": "large_mid", "baseline": "fp", "r_min": 0.4,
             "mean_rate_generated": 2.5},
            {"target_scenario": "large_mid", "baseline": "fp", "r_min": 0.6,
             "mean_rate_generated": 2.0},
        ]

        plot_transferability_results(
            sweep_results=results,
            output_dir=str(tmp_path),
            r_min_values=[0.4, 0.6],
        )

        plot_dir = tmp_path / "transferability_sweep_plots"
        assert (plot_dir / "transferability_sweep_all_metrics_rmin0.40.pdf").exists()
        assert (plot_dir / "transferability_sweep_all_metrics_rmin0.60.pdf").exists()
