"""
Pytest for the npz collection pipeline in build_diffusion_dataset.py.

Creates a self-contained dummy PD run directory (collected_samples.npz +
.hydra/config.yaml) with seed=9999 so no real channel cache exists, then
verifies that _collect_from_npz() correctly:
  - runs to completion (channel reconstructed via create_wireless_channel)
  - writes collected_samples.npz, config.yaml, and network_info.json
  - populates tx_locations for every network
  - does NOT write a channel_reconstruction_fallback field
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from graph_signal_diffusion.cli.wra.build_diffusion_dataset import _collect_from_npz


NUM_NETWORKS = 2
N_LINKS = 5
NUM_SAMPLES = 8
SEED = 9999  # deliberately unusual — no channel cache will exist for this seed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _object_array(arrays):
    """Build a 1-D NumPy object array from a list of ndarrays."""
    out = np.empty(len(arrays), dtype=object)
    for i, a in enumerate(arrays):
        out[i] = a
    return out


def _build_dummy_npz(path: Path, *, include_constraint_metadata: bool = False) -> None:
    """Write a minimal schema-v2 collected_samples.npz with deterministic data."""
    rng = np.random.default_rng(0)

    network_ids = np.arange(NUM_NETWORKS, dtype=np.int64)
    # seeds: seed_start + net_id
    network_seeds = np.array([SEED + i for i in range(NUM_NETWORKS)], dtype=np.int64)

    assoc_list = []
    power_list = []
    rate_list = []
    step_list = []
    r_min_list = []
    base_network_ids = []
    constraint_profile_ids = []
    constraint_profile_names = []

    for _ in range(NUM_NETWORKS):
        # Association matrix: identity-like (each TX serves its paired RX)
        assoc = np.eye(N_LINKS, dtype=np.float64)
        assoc_list.append(assoc)

        # Power samples: (NUM_SAMPLES, n_links) — values in [0, 1]
        power_list.append(rng.random((NUM_SAMPLES, N_LINKS)).astype(np.float64))

        # Rate samples: (NUM_SAMPLES, n_links) — non-negative
        rate_list.append(rng.random((NUM_SAMPLES, N_LINKS)).astype(np.float64) * 2.0)

        step_list.append(np.arange(NUM_SAMPLES, dtype=np.int64))
        if include_constraint_metadata:
            r_min_list.append(np.linspace(0.3, 0.7, N_LINKS, dtype=np.float64))
            base_network_ids.append(100 + _)
            constraint_profile_ids.append(_)
            constraint_profile_names.append(f"profile_{_}")

    payload = {
        "schema_version": np.array(2, dtype=np.int64),
        "channel_version": np.array("v2", dtype="<U2"),
        "network_ids": network_ids,
        "network_seeds": network_seeds,
        "associations": _object_array(assoc_list),
        "power_samples_per_network": _object_array(power_list),
        "rate_samples_per_network": _object_array(rate_list),
        "sample_steps_per_network": _object_array(step_list),
    }
    if include_constraint_metadata:
        payload["r_min_per_receiver_per_network"] = _object_array(r_min_list)
        payload["base_network_ids"] = np.asarray(base_network_ids, dtype=np.int64)
        payload["constraint_profile_ids"] = np.asarray(constraint_profile_ids, dtype=np.int64)
        payload["constraint_profile_names"] = _object_array(constraint_profile_names)
    np.savez_compressed(path, **payload)


def _build_config_yaml(path: Path) -> None:
    """Write a minimal Hydra config.yaml that triggers a cache miss."""
    config = {
        "name": "test_build_diffusion_dataset",
        "seed": SEED,
        "dataset": {
            "num_networks": NUM_NETWORKS,
            "n_links": N_LINKS,
            "deployment_range": 500.0,
        },
        "channel": {
            "version": "v2",
        },
        "system": {
            "P_max_dBm": 10.0,
            "bandwidth_Hz": 10_000_000.0,
            "noise_psd_dBm_Hz": -174.0,
        },
        "training": {
            "r_min": 0.5,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def dummy_pd_run(tmp_path):
    """Create a minimal PD run directory and return (run_dir, out_dir)."""
    run_dir = tmp_path / "pd_run"
    run_dir.mkdir()

    _build_config_yaml(run_dir / ".hydra" / "config.yaml")
    _build_dummy_npz(run_dir / "collected_samples.npz")

    out_dir = tmp_path / "raw_out"
    return run_dir, out_dir


@pytest.fixture
def dummy_pd_run_with_constraint_metadata(tmp_path):
    """Create a minimal PD run directory with optional per-network r_min metadata."""
    run_dir = tmp_path / "pd_run_conditional"
    run_dir.mkdir()

    _build_config_yaml(run_dir / ".hydra" / "config.yaml")
    _build_dummy_npz(run_dir / "collected_samples.npz", include_constraint_metadata=True)

    out_dir = tmp_path / "raw_out_conditional"
    return run_dir, out_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_collect_from_npz_writes_expected_files(dummy_pd_run):
    """`_collect_from_npz` must produce the three canonical output files."""
    run_dir, out_dir = dummy_pd_run

    result_dir = _collect_from_npz(
        input_path=str(run_dir),
        output_root=str(run_dir),      # not used when output_raw_wra_dir is set
        output_raw_wra_dir=str(out_dir),
        force=True,
    )

    assert result_dir == out_dir
    assert (out_dir / "collected_samples.npz").exists(), "collected_samples.npz missing"
    assert (out_dir / "config.yaml").exists(), "config.yaml missing"
    assert (out_dir / "network_info.json").exists(), "network_info.json missing"


def test_collect_from_npz_tx_locations_populated(dummy_pd_run):
    """Every network must have non-null tx_locations (channel was reconstructed)."""
    run_dir, out_dir = dummy_pd_run

    _collect_from_npz(
        input_path=str(run_dir),
        output_root=str(run_dir),
        output_raw_wra_dir=str(out_dir),
        force=True,
    )

    with open(out_dir / "network_info.json") as f:
        info = json.load(f)

    for net_id in range(NUM_NETWORKS):
        net_key = f"network_{net_id}"
        assert net_key in info, f"{net_key} not found in network_info.json"
        tx = info[net_key]["tx_locations"]
        assert tx is not None, f"tx_locations is null for {net_key}"
        assert len(tx) == N_LINKS, f"Expected {N_LINKS} tx rows for {net_key}, got {len(tx)}"


def test_collect_from_npz_no_fallback_key(dummy_pd_run):
    """The H_instantaneous fallback path was removed; no selection_summary should reference it."""
    run_dir, out_dir = dummy_pd_run

    _collect_from_npz(
        input_path=str(run_dir),
        output_root=str(run_dir),
        output_raw_wra_dir=str(out_dir),
        force=True,
    )

    with open(out_dir / "network_info.json") as f:
        info = json.load(f)

    for net_id in range(NUM_NETWORKS):
        net_key = f"network_{net_id}"
        summary = info[net_key]["selection_summary"]
        assert "channel_reconstruction_fallback" not in summary, (
            f"Unexpected fallback key in {net_key}.selection_summary"
        )


def test_collect_from_npz_sample_counts(dummy_pd_run):
    """Without a target_samples_per_network limit, all samples should be kept."""
    run_dir, out_dir = dummy_pd_run

    _collect_from_npz(
        input_path=str(run_dir),
        output_root=str(run_dir),
        output_raw_wra_dir=str(out_dir),
        force=True,
    )

    with open(out_dir / "network_info.json") as f:
        info = json.load(f)

    for net_id in range(NUM_NETWORKS):
        net_key = f"network_{net_id}"
        assert info[net_key]["num_samples_selected"] == NUM_SAMPLES
        assert info[net_key]["selection_summary"]["selection_strategy"] == "all_samples"


def test_collect_from_npz_target_subsampling(dummy_pd_run):
    """With target_samples_per_network=3, evenly-spaced downsampling should occur."""
    run_dir, out_dir = dummy_pd_run

    target = 3
    _collect_from_npz(
        input_path=str(run_dir),
        output_root=str(run_dir),
        output_raw_wra_dir=str(out_dir),
        force=True,
        target_samples_per_network=target,
    )

    with open(out_dir / "network_info.json") as f:
        info = json.load(f)

    for net_id in range(NUM_NETWORKS):
        net_key = f"network_{net_id}"
        assert info[net_key]["num_samples_selected"] == target
        assert info[net_key]["selection_summary"]["selection_strategy"] == "evenly_spaced"


def test_collect_from_npz_rate_stats_present(dummy_pd_run):
    """Rate statistics should be populated since the dummy NPZ includes rate samples."""
    run_dir, out_dir = dummy_pd_run

    _collect_from_npz(
        input_path=str(run_dir),
        output_root=str(run_dir),
        output_raw_wra_dir=str(out_dir),
        force=True,
    )

    with open(out_dir / "network_info.json") as f:
        info = json.load(f)

    for net_id in range(NUM_NETWORKS):
        net_key = f"network_{net_id}"
        rate_stats = info[net_key]["selected_rate_stats"]
        assert rate_stats is not None, f"rate_stats missing for {net_key}"
        assert "stochastic_avg_receiver_rates" in rate_stats

    final = info.get("final_dataset_rate_stats", {})
    assert isinstance(final, dict)
    assert "stochastic_avg_receiver_rates" in final


def test_collect_from_npz_force_flag(dummy_pd_run):
    """Running twice without force=True should raise FileExistsError."""
    run_dir, out_dir = dummy_pd_run

    _collect_from_npz(
        input_path=str(run_dir),
        output_root=str(run_dir),
        output_raw_wra_dir=str(out_dir),
        force=True,
    )

    with pytest.raises(FileExistsError):
        _collect_from_npz(
            input_path=str(run_dir),
            output_root=str(run_dir),
            output_raw_wra_dir=str(out_dir),
            force=False,
        )


def test_collect_from_npz_propagates_constraint_metadata(dummy_pd_run_with_constraint_metadata):
    run_dir, out_dir = dummy_pd_run_with_constraint_metadata

    _collect_from_npz(
        input_path=str(run_dir),
        output_root=str(run_dir),
        output_raw_wra_dir=str(out_dir),
        force=True,
    )

    with open(out_dir / "network_info.json") as f:
        info = json.load(f)

    assert info["has_per_network_r_min"] is True
    assert info["constraint_mode"] == "per_network_r_min"
    for net_id in range(NUM_NETWORKS):
        key = f"network_{net_id}"
        assert "r_min_per_receiver" in info[key]
        assert "base_network_id" in info[key]
        assert "constraint_profile_id" in info[key]
        assert "constraint_profile_name" in info[key]
