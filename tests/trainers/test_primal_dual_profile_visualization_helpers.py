import tempfile

import numpy as np
import pytest
import torch
import logging

from graph_signal_diffusion.trainers.dual_optimizer import DualOptimizer
from graph_signal_diffusion.trainers.primal_dual_trainer import (
    WRAConditionalPrimalDualTrainer,
    _extract_scalar_r_min_from_summary,
    visualize_power_allocations,
)


class _DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, *args, **kwargs):
        return self.weight


class _DummyConstraintDataset:
    def __init__(self):
        self._profile_vectors = {
            0: np.array([0.5, 0.5], dtype=np.float32),
            1: np.array([0.8, 0.8], dtype=np.float32),
        }
        self._profile_names = {0: "easy", 1: "hard"}

    def decode_expanded_id(self, expanded_id: int) -> tuple[int, int]:
        # base-major, num_profiles=2
        return int(expanded_id) // 2, int(expanded_id) % 2

    def get_profile_name(self, profile_id: int) -> str:
        return self._profile_names[int(profile_id)]

    def get_profile_vector(self, profile_id: int) -> np.ndarray:
        return self._profile_vectors[int(profile_id)]


def _build_conditional_trainer() -> WRAConditionalPrimalDualTrainer:
    dual = DualOptimizer(
        num_networks=4,
        num_receivers=2,
        r_min=torch.tensor(
            [
                [0.5, 0.5],  # base0/profile0
                [0.8, 0.8],  # base0/profile1
                [0.5, 0.5],  # base1/profile0
                [0.8, 0.8],  # base1/profile1
            ],
            dtype=torch.float32,
        ),
        alpha_dual=0.1,
        update_frequency=1,
        momentum=0.0,
        device="cpu",
    )
    checkpoint_dir = tempfile.mkdtemp(prefix="pd_profile_viz_")
    return WRAConditionalPrimalDualTrainer(
        constraint_profile_dataset=_DummyConstraintDataset(),
        model=_DummyModel(),
        dual_optimizer=dual,
        system_params={"noise_var": 1.0},
        learning_rate=1e-3,
        max_epochs=1,
        checkpoint_dir=checkpoint_dir,
        device="cpu",
    )


def test_extract_scalar_r_min_collapses_constant_range():
    summary = {
        "r_min": None,
        "r_min_is_scalar": False,
        "r_min_min": 0.65,
        "r_min_max": 0.65,
    }
    assert _extract_scalar_r_min_from_summary(summary) == pytest.approx(0.65)


def test_conditional_profile_metrics_and_base_major_selection():
    trainer = _build_conditional_trainer()

    epoch_network_ids = [0, 1, 2, 3]
    epoch_rates = [
        torch.tensor([0.62, 0.66]),  # profile 0
        torch.tensor([0.79, 0.82]),  # profile 1
        torch.tensor([0.58, 0.63]),  # profile 0
        torch.tensor([0.81, 0.83]),  # profile 1
    ]
    epoch_slacks = [
        torch.tensor([0.5, 0.5]) - epoch_rates[0],
        torch.tensor([0.8, 0.8]) - epoch_rates[1],
        torch.tensor([0.5, 0.5]) - epoch_rates[2],
        torch.tensor([0.8, 0.8]) - epoch_rates[3],
    ]

    metrics = trainer._build_constraint_profile_epoch_metrics(
        epoch_network_ids=epoch_network_ids,
        epoch_rates=epoch_rates,
        epoch_slacks=epoch_slacks,
    )
    assert [entry["constraint_profile_id"] for entry in metrics] == [0, 1]
    assert metrics[0]["constraint_profile_name"] == "easy"
    assert metrics[1]["constraint_profile_name"] == "hard"
    assert metrics[0]["r_min_is_scalar"] is True
    assert metrics[1]["r_min_is_scalar"] is True
    assert metrics[0]["r_min"] == pytest.approx(0.5)
    assert metrics[1]["r_min"] == pytest.approx(0.8)

    trainer._primal_metadata_entries = {0: [], 1: [], 2: [], 3: []}
    assert trainer._select_visualization_network_ids(max_base_networks=1) == [0, 1]
    assert trainer._select_visualization_network_ids(max_base_networks=2) == [0, 1, 2, 3]


def test_visualize_power_allocations_skips_invalid_associations(caplog, tmp_path):
    entries = [
        {
            "epoch": 1,
            "network_id": 0,
            "power_allocations": [0.2, 0.8],
            "rates": [0.4, 0.5],
        },
        {
            "epoch": 2,
            "network_id": 0,
            "power_allocations": [0.3, 0.7],
            "rates": [0.45, 0.55],
        },
    ]
    metadata_entries = {0: [1, 0, 1]}  # invalid: not a 2D association matrix

    with caplog.at_level(logging.WARNING):
        visualize_power_allocations(
            output_dir=str(tmp_path),
            P_max=1.0,
            r_min=0.5,
            network_id=0,
            all_entries=entries,
            metadata_entries=metadata_entries,
        )

    assert any(
        "expected 2D associations matrix" in record.message for record in caplog.records
    )
