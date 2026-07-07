import random

from graph_signal_diffusion.datasets.wra.configs import dataset_name_to_alias
from graph_signal_diffusion.datasets.wra.sampler import NetworkGroupedBatchSampler


class _MockWRADataset:
    def __init__(self, samples):
        self.samples = samples


def test_dataset_name_to_alias_for_standard_wra_configs():
    # build_diffusion_dataset.py strips the "wra_" prefix from the scenario name, so
    # content-addressed paths look like "debug/pdc_npz_...", "large/pdc_npz_...", etc.
    assert dataset_name_to_alias("debug/pdc_npz_abc123") == "wra-debug"
    assert dataset_name_to_alias("large_low_density/pdc_npz_xyz456") == "wra-large-low-density"
    assert dataset_name_to_alias("small/pdc_npz_def789") == "wra-small"
    # Already alias-like strings pass through unchanged.
    assert dataset_name_to_alias("wra-large") == "wra-large"
    assert dataset_name_to_alias("wra-large-low-density") == "wra-large-low-density"
    # Empty / non-string input returns sentinel.
    assert dataset_name_to_alias("") == "wra-unknown"


def test_network_grouped_sampler_uses_raw_dataset_name_and_network_id():
    # Two datasets share the same numeric network_id=0; they must be treated separately.
    # Paths use stripped scenario names (no wra_ prefix) as produced by build_diffusion_dataset.py.
    samples = [
        ("large/pdc_npz_aaa", 0, 0),
        ("large/pdc_npz_aaa", 0, 1),
        ("large_low_density/pdc_npz_bbb", 0, 0),
        ("large_low_density/pdc_npz_bbb", 0, 1),
        ("large_low_density/pdc_npz_bbb", 1, 0),
        ("large_low_density/pdc_npz_bbb", 1, 1),
    ]
    dataset = _MockWRADataset(samples)

    sampler = NetworkGroupedBatchSampler(
        dataset=dataset,
        batch_size=4,
        samples_per_network=2,
        shuffle=False,
    )

    keys = set(sampler.network_to_indices.keys())
    assert ("large/pdc_npz_aaa", 0) in keys
    assert ("large_low_density/pdc_npz_bbb", 0) in keys
    assert ("large_low_density/pdc_npz_bbb", 1) in keys
    assert len(keys) == 3

    # With shuffle disabled, first batch should contain first two composite keys only.
    random.seed(0)
    batch0 = next(iter(sampler))
    batch0_keys = {(dataset.samples[i][0], dataset.samples[i][1]) for i in batch0}
    assert len(batch0_keys) == 2
    assert ("large/pdc_npz_aaa", 0) in batch0_keys
    assert ("large_low_density/pdc_npz_bbb", 0) in batch0_keys


def test_sampler_separates_same_scenario_different_hash():
    """Sub-datasets with the same scenario prefix but different content hashes
    must produce separate grouping keys (not merge into one)."""
    samples = [
        ("medium-large_outdoor_low_density/wrpc_v1_k200_hAAA", 0, 0),
        ("medium-large_outdoor_low_density/wrpc_v1_k200_hAAA", 0, 1),
        ("medium-large_outdoor_low_density/wrpc_v1_k200_hBBB", 0, 0),
        ("medium-large_outdoor_low_density/wrpc_v1_k200_hBBB", 0, 1),
    ]
    dataset = _MockWRADataset(samples)

    sampler = NetworkGroupedBatchSampler(
        dataset=dataset,
        batch_size=2,
        samples_per_network=2,
        shuffle=False,
    )

    keys = set(sampler.network_to_indices.keys())
    assert len(keys) == 2, f"Expected 2 distinct keys, got {len(keys)}: {keys}"
    assert ("medium-large_outdoor_low_density/wrpc_v1_k200_hAAA", 0) in keys
    assert ("medium-large_outdoor_low_density/wrpc_v1_k200_hBBB", 0) in keys

    # Each key should have exactly 2 sample indices (not 4 merged).
    for key in keys:
        assert len(sampler.network_to_indices[key]) == 2
