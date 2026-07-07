import torch

from graph_signal_diffusion.trainers.dual_optimizer import DualOptimizer


def test_dual_optimizer_scalar_rmin_update_matches_expected_slack():
    opt = DualOptimizer(
        num_networks=2,
        num_receivers=3,
        r_min=0.5,
        alpha_dual=1.0,
        update_frequency=1,
        momentum=0.0,
        device="cpu",
    )
    network_ids = torch.tensor([0, 1], dtype=torch.long)
    rates = torch.tensor(
        [[0.4, 0.6, 0.5], [0.7, 0.2, 0.5]],
        dtype=torch.float32,
    )
    opt.update(network_ids, rates)

    expected = torch.tensor(
        [[1.1, 0.9, 1.0], [0.8, 1.3, 1.0]],
        dtype=torch.float32,
    )
    assert torch.allclose(opt.lambdas, expected, atol=1e-6)
    assert opt.is_scalar_r_min()
    assert opt.scalar_r_min_or_none() == 0.5


def test_dual_optimizer_per_network_vector_rmin():
    opt = DualOptimizer(
        num_networks=2,
        num_receivers=3,
        r_min=torch.tensor([0.5, 0.9], dtype=torch.float32),
        alpha_dual=1.0,
        update_frequency=1,
        momentum=0.0,
        device="cpu",
    )
    network_ids = torch.tensor([0, 1], dtype=torch.long)
    rates = torch.tensor(
        [[0.4, 0.6, 0.5], [0.7, 0.2, 0.5]],
        dtype=torch.float32,
    )
    opt.update(network_ids, rates)

    expected = torch.tensor(
        [[1.1, 0.9, 1.0], [1.2, 1.7, 1.4]],
        dtype=torch.float32,
    )
    assert torch.allclose(opt.lambdas, expected, atol=1e-6)
    assert not opt.is_scalar_r_min()
    assert opt.scalar_r_min_or_none() is None


def test_dual_optimizer_per_receiver_matrix_rmin_roundtrip():
    r_min_table = torch.tensor(
        [[0.5, 0.6, 0.7], [0.8, 0.9, 1.0]],
        dtype=torch.float32,
    )
    opt = DualOptimizer(
        num_networks=2,
        num_receivers=3,
        r_min=r_min_table,
        alpha_dual=0.25,
        update_frequency=2,
        momentum=0.1,
        device="cpu",
    )
    state = opt.state_dict()

    opt2 = DualOptimizer(
        num_networks=2,
        num_receivers=3,
        r_min=0.0,
        alpha_dual=0.25,
        update_frequency=2,
        momentum=0.1,
        device="cpu",
    )
    opt2.load_state_dict(state)
    assert torch.allclose(opt2.r_min_table, r_min_table)
    assert not opt2.is_scalar_r_min()


def test_dual_optimizer_constant_table_detected_as_scalar():
    opt = DualOptimizer(
        num_networks=2,
        num_receivers=3,
        r_min=torch.full((2, 3), 0.75),
        alpha_dual=0.1,
        update_frequency=1,
        momentum=0.0,
        device="cpu",
    )
    assert opt.is_scalar_r_min()
    assert opt.scalar_r_min_or_none() == 0.75
