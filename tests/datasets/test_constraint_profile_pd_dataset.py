import numpy as np
import torch

from graph_signal_diffusion.datasets.wra.primal_dual_dataset import (
    WRAPrimalDualDataset,
    WRAConstraintProfilePrimalDualDataset,
    WirelessData,
)


def _make_base_dataset(num_networks: int = 2, n_links: int = 3) -> WRAPrimalDualDataset:
    dataset = WRAPrimalDualDataset.__new__(WRAPrimalDualDataset)
    dataset.channels = []
    dataset.samples = []
    for net_id in range(num_networks):
        x = torch.stack(
            [
                torch.linspace(0.1, 0.3, n_links),
                torch.linspace(0.4, 0.6, n_links),
            ],
            dim=1,
        )
        sample = WirelessData(
            x=x,
            edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            edge_weight=torch.tensor([1.0, 1.0], dtype=torch.float32),
            H_instantaneous=torch.ones((2, n_links, n_links), dtype=torch.float32),
            H_l=torch.eye(n_links, dtype=torch.float32),
            associations=torch.eye(n_links, dtype=torch.float32),
            network_id=torch.tensor(net_id, dtype=torch.long),
            network_seed=torch.tensor(100 + net_id, dtype=torch.long),
        )
        dataset.samples.append(sample)
    return dataset


def test_constraint_profile_dataset_base_major_expansion_and_features():
    base_dataset = _make_base_dataset(num_networks=2, n_links=3)
    wrapped = WRAConstraintProfilePrimalDualDataset(
        base_dataset=base_dataset,
        constraint_profiles=[0.5, [0.2, 0.4, 0.6]],
        profile_names=["uniform", "vector"],
        r_min_feature_scale=2.0,
    )

    assert len(wrapped) == 4

    s0 = wrapped[0]  # base 0, profile 0
    s1 = wrapped[1]  # base 0, profile 1
    s2 = wrapped[2]  # base 1, profile 0
    assert int(s0.network_id.item()) == 0
    assert int(s1.network_id.item()) == 1
    assert int(s2.network_id.item()) == 2

    assert s0.x.shape == (3, 3)
    assert torch.allclose(s0.x[:, :2], base_dataset[0].x)
    assert torch.allclose(s0.x[:, 2], torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32))
    assert torch.allclose(s1.x[:, 2], torch.tensor([0.4, 0.8, 1.2], dtype=torch.float32))

    base_id, profile_id = wrapped.decode_expanded_id(3)
    assert (base_id, profile_id) == (1, 1)
    assert wrapped.encode_expanded_id(base_id, profile_id) == 3
    assert wrapped.get_profile_name(1) == "vector"
    assert np.allclose(wrapped.get_profile_vector(0), np.array([0.5, 0.5, 0.5], dtype=np.float32))


def test_constraint_profile_dataset_r_min_table_base_major_order():
    base_dataset = _make_base_dataset(num_networks=3, n_links=2)
    wrapped = WRAConstraintProfilePrimalDualDataset(
        base_dataset=base_dataset,
        constraint_profiles=[0.1, 0.9],
    )
    table = wrapped.get_r_min_table()
    assert table.shape == (6, 2)
    expected = torch.tensor(
        [
            [0.1, 0.1],  # base 0, profile 0
            [0.9, 0.9],  # base 0, profile 1
            [0.1, 0.1],  # base 1, profile 0
            [0.9, 0.9],  # base 1, profile 1
            [0.1, 0.1],  # base 2, profile 0
            [0.9, 0.9],  # base 2, profile 1
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(table, expected)
