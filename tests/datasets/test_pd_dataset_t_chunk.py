import torch

from graph_signal_diffusion.datasets.wra.primal_dual_dataset import (
    WRAPrimalDualDataset,
    WRAConstraintProfilePrimalDualDataset,
    WirelessData,
)


def _make_base_dataset(*, num_networks: int = 1, n_links: int = 3, T: int = 10, t_chunk: int | None = None):
    dataset = WRAPrimalDualDataset.__new__(WRAPrimalDualDataset)
    dataset.channels = []
    dataset.samples = []
    dataset.num_timesteps = T
    dataset.t_chunk = T if t_chunk is None else int(t_chunk)

    for net_id in range(num_networks):
        # Timestep signature: H[t, :, :] == t, so chunk boundaries are easy to verify.
        h_ts = torch.arange(T, dtype=torch.float32).view(T, 1, 1).expand(T, n_links, n_links).clone()
        sample = WirelessData(
            x=torch.ones((n_links, 2), dtype=torch.float32),
            edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            edge_weight=torch.tensor([1.0, 1.0], dtype=torch.float32),
            H_instantaneous=h_ts,
            H_l=torch.eye(n_links, dtype=torch.float32),
            associations=torch.eye(n_links, dtype=torch.float32),
            network_id=torch.tensor(net_id, dtype=torch.long),
            network_seed=torch.tensor(100 + net_id, dtype=torch.long),
        )
        dataset.samples.append(sample)
    return dataset


def test_getitem_returns_full_sequence_when_chunk_equals_num_timesteps():
    ds = _make_base_dataset(T=10, t_chunk=10)
    sample = ds[0]
    assert sample.H_instantaneous.shape[0] == 10
    assert torch.allclose(sample.H_instantaneous[:, 0, 0], torch.arange(10, dtype=torch.float32))


def test_getitem_returns_random_consecutive_chunk_and_preserves_base_tensor():
    ds = _make_base_dataset(T=10, t_chunk=4)
    torch.manual_seed(0)
    sample = ds[0]

    chunk_sig = sample.H_instantaneous[:, 0, 0]
    assert sample.H_instantaneous.shape[0] == 4
    assert torch.allclose(chunk_sig[1:] - chunk_sig[:-1], torch.ones(3))
    assert 0 <= int(chunk_sig[0].item()) <= 6

    base_sig = ds.samples[0].H_instantaneous[:, 0, 0]
    assert base_sig.shape[0] == 10
    assert torch.allclose(base_sig, torch.arange(10, dtype=torch.float32))


def test_constraint_profile_wrapper_respects_base_dataset_chunking():
    base = _make_base_dataset(num_networks=2, n_links=3, T=8, t_chunk=3)
    wrapped = WRAConstraintProfilePrimalDualDataset(
        base_dataset=base,
        constraint_profiles=[0.5],
    )
    sample = wrapped[0]
    assert sample.H_instantaneous.shape[0] == 3
    assert sample.x.shape[1] == 3  # Original 2 channels + r_min channel


def test_resolve_t_chunk_validates_bounds():
    assert WRAPrimalDualDataset._resolve_t_chunk(None, 10) == 10
    assert WRAPrimalDualDataset._resolve_t_chunk(4, 10) == 4

    try:
        WRAPrimalDualDataset._resolve_t_chunk(11, 10)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for t_chunk > num_timesteps")

