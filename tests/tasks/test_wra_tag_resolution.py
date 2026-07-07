import numpy as np
import torch
from unittest.mock import MagicMock

from graph_signal_diffusion.tasks.wireless_resource_allocation.evaluator import (
    WirelessResourceAllocationTask,
)


def test_prepare_data_resolves_alias_dataset_tag_to_canonical_metadata_key():
    task = WirelessResourceAllocationTask()
    canonical_dataset_name = "N_648_density_15.8_seed_42_Pmax_dBm_10.0_rmin_1.0"
    dataset_alias = "wra-large-low-density"

    associations = np.eye(2, dtype=np.float32)
    dataset_info = {
        "system_params": {"P_max": 1.0, "noise_var": 1e-10, "r_min": 0.5},
        "associations": {(canonical_dataset_name, 0): associations},
        "h_timeslot_dirs": {(canonical_dataset_name, 0): "/tmp/fake_h_slots"},
        "h_num_timesteps": {(canonical_dataset_name, 0): 200},
    }
    task.set_dataset_info(dataset_info)

    batch = MagicMock()
    batch.y = torch.randn(4, 1, 1)  # 2 graphs x 2 nodes
    batch.network_id = torch.tensor([0, 0])
    batch.dataset_name = [dataset_alias, dataset_alias]
    batch.batch = torch.tensor([0, 0, 1, 1])

    prepared = task.prepare_data(batch)
    metadata = prepared["metadata"]

    assert metadata["dataset_names"] == [canonical_dataset_name, canonical_dataset_name]
    assert metadata["dataset_name_tags"] == [dataset_alias, dataset_alias]
    assert metadata["dataset_aliases"] == [dataset_alias, dataset_alias]
    assert np.array_equal(metadata["associations"][0], associations)
    assert np.array_equal(metadata["associations"][1], associations)
    assert metadata["h_timeslot_dirs"] == ["/tmp/fake_h_slots", "/tmp/fake_h_slots"]
    assert metadata["h_num_timesteps"] == [200, 200]


def test_resolve_selector_network_key_matches_alias_and_network_id():
    task = WirelessResourceAllocationTask()
    selector_stats = {
        "N_648_density_15.8_seed_42_Pmax_dBm_10.0_rmin_1.0::network_0": {}
    }
    resolved = task._resolve_selector_network_key(
        selector_network_node_stats=selector_stats,
        dataset_name="wra-large-low-density",
        network_id=0,
    )
    assert resolved == "N_648_density_15.8_seed_42_Pmax_dBm_10.0_rmin_1.0::network_0"


def test_selector_rate_correlation_plot_created(tmp_path):
    task = WirelessResourceAllocationTask()
    selector_stats = {
        "wra-large-low-density::network_0": {
            0: {"hard_mask_mean": torch.tensor([0.9, 0.7, 0.4, 0.2], dtype=torch.float32)},
            1: {"hard_mask_mean": torch.tensor([0.8, 0.5, 0.3, 0.1], dtype=torch.float32)},
            2: {"hard_mask_mean": torch.tensor([0.7, 0.6, 0.2, 0.05], dtype=torch.float32)},
        }
    }
    metadata = {"selector_network_node_stats": selector_stats}
    evaluation_records = [
        {
            "dataset_name": "wra-large-low-density",
            "network_id": 0,
            "real_rates_final": np.array([0.2, 0.35, 0.55, 0.8], dtype=float),
            "gen_rates_final": np.array([0.25, 0.33, 0.58, 0.78], dtype=float),
        },
        {
            "dataset_name": "wra-large-low-density",
            "network_id": 0,
            "real_rates_final": np.array([0.18, 0.37, 0.5, 0.82], dtype=float),
            "gen_rates_final": np.array([0.22, 0.31, 0.56, 0.8], dtype=float),
        },
    ]

    task._visualize_selector_rate_correlation(
        evaluation_records=evaluation_records,
        metadata=metadata,
        viz_save_dir=str(tmp_path),
    )

    expected = tmp_path / "selector_rate_correlation_dwra-large-low-density_n0.pdf"
    assert expected.exists()


def test_visualize_results_sanitizes_dataset_name_in_output_filename(tmp_path):
    task = WirelessResourceAllocationTask()

    generated_power = torch.tensor([[0.2, 0.8]], dtype=torch.float32)
    real_power = torch.tensor([[0.25, 0.75]], dtype=torch.float32)
    metadata = {
        "system_params": {"P_max": 1.0},
        "batch_size": 1,
        "num_nodes": 2,
        "network_ids": [12],
        "dataset_names": ["debug/wrpc_v1_primal_history"],
    }

    task._visualize_results(
        generated_power=generated_power,
        real_power=real_power,
        metadata=metadata,
        viz_save_dir=str(tmp_path),
    )

    expected = tmp_path / "power_allocation_scatter_ddebug_wrpc_v1_primal_history_n12.pdf"
    assert expected.exists()
