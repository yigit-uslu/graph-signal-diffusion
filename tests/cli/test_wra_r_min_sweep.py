"""Tests for r_min sweep helper module."""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_reference_raw_dir(tmp_path, *, num_networks=2, include_r_min_per_receiver=False):
    """Create a minimal reference raw directory with NPZ, config, network_info."""
    raw_dir = tmp_path / "reference" / "raw"
    raw_dir.mkdir(parents=True)

    # network_info.json
    network_info = {
        "system_params": {"r_min": 0.5, "P_max_dBm": 23.0},
        "channel_params": {"version": "v2"},
    }
    for net_id in range(num_networks):
        network_info[f"network_{net_id}"] = {
            "system_params": {"r_min": 0.5},
            "n_links": 4,
            "network_seed": 42 + net_id,
        }
    (raw_dir / "network_info.json").write_text(json.dumps(network_info, indent=2))

    # config.yaml
    (raw_dir / "config.yaml").write_text("dummy: true\n")

    # collected_samples.npz
    npz_data = {}
    for net_id in range(num_networks):
        net_dict = {
            "H_ls": np.random.rand(4, 4).astype(np.float32),
            "associations": np.eye(4, dtype=np.float32),
            "power_samples": [
                {"power_allocation": np.random.rand(4).astype(np.float32)}
                for _ in range(3)
            ],
        }
        if include_r_min_per_receiver:
            net_dict["r_min_per_receiver"] = np.full(4, 0.5, dtype=np.float32)
        npz_data[f"network_{net_id}"] = net_dict

    np.savez(str(raw_dir / "collected_samples.npz"), **npz_data)

    # h_instantaneous directory (empty — H is optional for sweep tests)
    (raw_dir / "h_instantaneous").mkdir()

    return str(raw_dir)


# ---------------------------------------------------------------------------
# Tests: Guards
# ---------------------------------------------------------------------------


class TestGuards:
    def test_npz_no_r_min_per_receiver_passes(self, tmp_path):
        """Should pass when NPZ has no r_min_per_receiver."""
        from graph_signal_diffusion.cli._wra_r_min_sweep import (
            check_npz_no_r_min_per_receiver,
        )
        raw_dir = _make_reference_raw_dir(tmp_path, include_r_min_per_receiver=False)
        # Should not raise
        check_npz_no_r_min_per_receiver(raw_dir)

    def test_npz_r_min_per_receiver_raises(self, tmp_path):
        """Should raise when NPZ contains r_min_per_receiver."""
        from graph_signal_diffusion.cli._wra_r_min_sweep import (
            check_npz_no_r_min_per_receiver,
        )
        raw_dir = _make_reference_raw_dir(tmp_path, include_r_min_per_receiver=True)
        with pytest.raises(ValueError, match="r_min_per_receiver"):
            check_npz_no_r_min_per_receiver(raw_dir)

    def test_npz_guard_skips_legacy_flat_keys(self, tmp_path):
        """Should not crash on NPZ with flat keys like network_0_H_instantaneous."""
        from graph_signal_diffusion.cli._wra_r_min_sweep import (
            check_npz_no_r_min_per_receiver,
        )
        raw_dir = _make_reference_raw_dir(tmp_path, include_r_min_per_receiver=False)
        # Add a legacy flat key that is not a dict
        npz_path = os.path.join(raw_dir, "collected_samples.npz")
        with np.load(npz_path, allow_pickle=True) as npz:
            data = dict(npz)
        data["network_0_H_instantaneous"] = np.zeros((10, 4, 4), dtype=np.float32)
        np.savez(npz_path, **data)
        # Should not raise
        check_npz_no_r_min_per_receiver(raw_dir)

    def test_model_cond_channels_too_low(self):
        """Should raise when model_cond_channels < 3."""
        from graph_signal_diffusion.cli._wra_r_min_sweep import (
            check_model_cond_channels,
        )
        with pytest.raises(ValueError, match="model_cond_channels >= 3"):
            check_model_cond_channels(2)

    def test_model_cond_channels_ok(self):
        """Should not raise when model_cond_channels >= 3."""
        from graph_signal_diffusion.cli._wra_r_min_sweep import (
            check_model_cond_channels,
        )
        check_model_cond_channels(3)


# ---------------------------------------------------------------------------
# Tests: create_sweep_dataset_dirs
# ---------------------------------------------------------------------------


class TestCreateSweepDatasetDirs:
    def test_creates_symlinks_and_network_info(self, tmp_path):
        """Verify correct symlinks and r_min override in network_info.json."""
        from graph_signal_diffusion.cli._wra_r_min_sweep import (
            create_sweep_dataset_dirs,
        )

        raw_dir = _make_reference_raw_dir(tmp_path, num_networks=2)
        output_root = str(tmp_path / "sweep_data")

        ds_names = create_sweep_dataset_dirs(
            reference_raw_dir=raw_dir,
            output_root=output_root,
            r_min_values=[0.3, 0.9],
        )

        assert len(ds_names) == 2
        assert ds_names[0] == "sweep_rmin_0.30"
        assert ds_names[1] == "sweep_rmin_0.90"

        for ds_name, expected_r_min in zip(ds_names, [0.3, 0.9]):
            ds_raw = os.path.join(output_root, ds_name, "raw")

            # Symlinks exist
            assert os.path.islink(os.path.join(ds_raw, "collected_samples.npz"))
            assert os.path.islink(os.path.join(ds_raw, "config.yaml"))
            assert os.path.islink(os.path.join(ds_raw, "h_instantaneous"))

            # network_info.json is a regular file (not symlink)
            info_path = os.path.join(ds_raw, "network_info.json")
            assert os.path.isfile(info_path)
            assert not os.path.islink(info_path)

            # r_min overridden at all levels
            with open(info_path) as f:
                info = json.load(f)
            assert info["system_params"]["r_min"] == pytest.approx(expected_r_min)
            assert info["network_0"]["system_params"]["r_min"] == pytest.approx(expected_r_min)
            assert info["network_1"]["system_params"]["r_min"] == pytest.approx(expected_r_min)

    def test_replaces_real_directory_with_symlink(self, tmp_path):
        """A stale real directory at dst should be removed and replaced by a symlink."""
        from graph_signal_diffusion.cli._wra_r_min_sweep import (
            create_sweep_dataset_dirs,
        )

        raw_dir = _make_reference_raw_dir(tmp_path, num_networks=2)
        output_root = str(tmp_path / "sweep_data")

        # Pre-create a real directory where h_instantaneous symlink should go
        stale_dir = os.path.join(output_root, "sweep_rmin_0.30", "raw", "h_instantaneous")
        os.makedirs(stale_dir, exist_ok=True)
        # Put a file inside to ensure rmtree is needed
        with open(os.path.join(stale_dir, "leftover.txt"), "w") as f:
            f.write("stale")

        # Should not raise IsADirectoryError
        ds_names = create_sweep_dataset_dirs(
            reference_raw_dir=raw_dir,
            output_root=output_root,
            r_min_values=[0.3],
        )

        dst = os.path.join(output_root, ds_names[0], "raw", "h_instantaneous")
        assert os.path.islink(dst), "Expected a symlink, got a real directory"

    def test_idempotent(self, tmp_path):
        """Running create_sweep_dataset_dirs twice should not error."""
        from graph_signal_diffusion.cli._wra_r_min_sweep import (
            create_sweep_dataset_dirs,
        )

        raw_dir = _make_reference_raw_dir(tmp_path)
        output_root = str(tmp_path / "sweep_data")

        create_sweep_dataset_dirs(
            reference_raw_dir=raw_dir,
            output_root=output_root,
            r_min_values=[0.5],
        )
        # Run again — should not raise
        create_sweep_dataset_dirs(
            reference_raw_dir=raw_dir,
            output_root=output_root,
            r_min_values=[0.5],
        )


# ---------------------------------------------------------------------------
# Tests: Summary table
# ---------------------------------------------------------------------------


class TestSweepSummary:
    def test_generates_csv_and_md(self, tmp_path):
        """Summary table should produce both CSV and markdown files."""
        from graph_signal_diffusion.cli._wra_r_min_sweep import save_sweep_summary

        results = [
            {"r_min": 0.4, "mean_rate_generated": 2.5, "sum_rate_generated": 10.0},
            {"r_min": 0.6, "mean_rate_generated": 1.8, "sum_rate_generated": 7.2},
            {"r_min": 0.45, "mean_rate_generated": 2.1, "sum_rate_generated": 8.4},
        ]
        save_sweep_summary(
            sweep_results=results,
            native_r_min_values=[0.4, 0.6],
            output_dir=str(tmp_path),
        )

        csv_path = tmp_path / "r_min_sweep_summary.csv"
        md_path = tmp_path / "r_min_sweep_summary.md"
        assert csv_path.exists()
        assert md_path.exists()

        import pandas as pd
        df = pd.read_csv(csv_path)
        assert len(df) == 3
        # Should be sorted by r_min
        assert list(df["r_min"]) == pytest.approx([0.4, 0.45, 0.6])
        # Native flag
        assert list(df["is_native"]) == [True, False, True]

        md_content = md_path.read_text()
        assert "0.40" in md_content
        assert "0.45" in md_content
        assert "0.60" in md_content


# ---------------------------------------------------------------------------
# Tests: Plots
# ---------------------------------------------------------------------------


class TestSweepPlots:
    def test_generates_plot_files(self, tmp_path):
        """plot_r_min_sweep should produce individual and combined PDFs."""
        import matplotlib
        matplotlib.use("Agg")

        from graph_signal_diffusion.cli._wra_r_min_sweep import plot_r_min_sweep

        results = [
            {"r_min": 0.4, "mean_rate_generated": 2.5,
             "rate_0.1pct_generated": 0.3, "rate_0.5pct_generated": 0.4,
             "rate_1pct_generated": 0.5, "rate_5pct_generated": 0.7,
             "rate_10pct_generated": 0.9},
            {"r_min": 0.6, "mean_rate_generated": 1.8,
             "rate_0.1pct_generated": 0.5, "rate_0.5pct_generated": 0.55,
             "rate_1pct_generated": 0.6, "rate_5pct_generated": 0.8,
             "rate_10pct_generated": 1.0},
        ]

        plot_r_min_sweep(
            sweep_results=results,
            native_r_min_values=[0.4],
            output_dir=str(tmp_path),
        )

        plot_dir = tmp_path / "r_min_sweep_plots"
        assert plot_dir.exists()
        # Combined plot
        assert (plot_dir / "r_min_sweep_all_percentiles.pdf").exists()
        # Individual plots
        assert (plot_dir / "r_min_sweep_rate_1pct_generated.pdf").exists()
        assert (plot_dir / "r_min_sweep_mean_rate_generated.pdf").exists()

    def test_handles_missing_metrics(self, tmp_path):
        """Plot should handle None/missing metric values gracefully."""
        import matplotlib
        matplotlib.use("Agg")

        from graph_signal_diffusion.cli._wra_r_min_sweep import plot_r_min_sweep

        results = [
            {"r_min": 0.4, "mean_rate_generated": 2.5},
            {"r_min": 0.6},  # No metrics
        ]

        # Should not raise
        plot_r_min_sweep(
            sweep_results=results,
            native_r_min_values=[0.4],
            output_dir=str(tmp_path),
        )


# ---------------------------------------------------------------------------
# Tests: dataset_info extraction
# ---------------------------------------------------------------------------


class TestBuildSweepDatasetInfo:
    def test_extracts_metadata_from_processed_dir(self, tmp_path):
        """build_sweep_dataset_info should extract metadata from graph.pt files."""
        import torch
        from graph_signal_diffusion.cli._wra_r_min_sweep import (
            build_sweep_dataset_info,
        )

        ds_name = "sweep_rmin_0.30"
        root = str(tmp_path)

        # Create raw dir with network_info.json
        raw_dir = os.path.join(root, ds_name, "raw")
        os.makedirs(raw_dir)
        network_info = {
            "system_params": {"r_min": 0.3},
            "network_0": {"system_params": {"r_min": 0.3}, "network_seed": 42},
        }
        with open(os.path.join(raw_dir, "network_info.json"), "w") as f:
            json.dump(network_info, f)

        # Create processed dir with graph.pt
        proc_dir = os.path.join(root, ds_name, "processed", "network_0")
        os.makedirs(proc_dir)
        graph_data = {
            "associations": np.eye(4, dtype=np.float32),
            "H_ls": np.random.rand(4, 4).astype(np.float32),
            "network_seed": 42,
            "channel_version": "v2",
        }
        torch.save(graph_data, os.path.join(proc_dir, "graph.pt"))

        # Create H_instantaneous dir with timestep files
        h_dir = os.path.join(proc_dir, "H_instantaneous")
        os.makedirs(h_dir)
        for t in range(5):
            torch.save(
                torch.randn(4, 4),
                os.path.join(h_dir, f"timestep_{t}.pt"),
            )

        info = build_sweep_dataset_info(
            dataset_root=root,
            dataset_name=ds_name,
            r_min=0.3,
            system_params={"P_max": 1.0, "noise_var": 1e-10, "r_min": 0.5},
        )

        # Check r_min overridden in system_params
        assert info["system_params"]["r_min"] == pytest.approx(0.3)

        # Check r_min_per_dataset
        assert info["r_min_per_dataset"] == {ds_name: pytest.approx(0.3)}

        # Check associations extracted
        key = (ds_name, 0)
        assert key in info["associations"]
        assert info["associations"][key].shape == (4, 4)

        # Check H timeslot info
        assert key in info["h_timeslot_dirs"]
        assert key in info["h_num_timesteps"]
        assert info["h_num_timesteps"][key] == 5

        # Check network seeds
        assert info["network_seeds"][key] == 42

    def test_propagates_channel_cache_info(self, tmp_path):
        """channel_cache_info from network_info.json should appear in dataset_info."""
        import torch
        from graph_signal_diffusion.cli._wra_r_min_sweep import (
            build_sweep_dataset_info,
        )

        ds_name = "sweep_rmin_0.50"
        root = str(tmp_path)

        raw_dir = os.path.join(root, ds_name, "raw")
        os.makedirs(raw_dir)
        network_info = {
            "system_params": {"r_min": 0.5},
            "channel_cache_info": {
                "channel_cache_file": "/some/path/cache.pt",
                "channel_cache_key": "abc123",
            },
            "network_0": {"system_params": {"r_min": 0.5}},
        }
        with open(os.path.join(raw_dir, "network_info.json"), "w") as f:
            json.dump(network_info, f)

        proc_dir = os.path.join(root, ds_name, "processed", "network_0")
        os.makedirs(proc_dir)
        torch.save(
            {"associations": np.eye(4, dtype=np.float32),
             "H_ls": np.random.rand(4, 4).astype(np.float32)},
            os.path.join(proc_dir, "graph.pt"),
        )

        info = build_sweep_dataset_info(
            dataset_root=root,
            dataset_name=ds_name,
            r_min=0.5,
            system_params={"P_max": 1.0, "noise_var": 1e-10, "r_min": 0.5},
        )

        assert ds_name in info["channel_cache_info"]
        assert info["channel_cache_info"][ds_name]["channel_cache_file"] == "/some/path/cache.pt"

    def test_empty_channel_cache_info_when_absent(self, tmp_path):
        """channel_cache_info should be empty when not in network_info.json."""
        import torch
        from graph_signal_diffusion.cli._wra_r_min_sweep import (
            build_sweep_dataset_info,
        )

        ds_name = "sweep_rmin_0.50"
        root = str(tmp_path)

        raw_dir = os.path.join(root, ds_name, "raw")
        os.makedirs(raw_dir)
        network_info = {
            "system_params": {"r_min": 0.5},
            "network_0": {"system_params": {"r_min": 0.5}},
        }
        with open(os.path.join(raw_dir, "network_info.json"), "w") as f:
            json.dump(network_info, f)

        proc_dir = os.path.join(root, ds_name, "processed", "network_0")
        os.makedirs(proc_dir)
        torch.save(
            {"associations": np.eye(4, dtype=np.float32),
             "H_ls": np.random.rand(4, 4).astype(np.float32)},
            os.path.join(proc_dir, "graph.pt"),
        )

        info = build_sweep_dataset_info(
            dataset_root=root,
            dataset_name=ds_name,
            r_min=0.5,
            system_params={"P_max": 1.0, "noise_var": 1e-10, "r_min": 0.5},
        )

        assert info["channel_cache_info"] == {}
