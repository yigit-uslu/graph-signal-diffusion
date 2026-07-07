"""Verify that dataset.shuffle_val wires through to the val/train-val batch samplers."""

import math
import pytest
import torch
from omegaconf import OmegaConf
from unittest.mock import MagicMock, patch, call

from graph_signal_diffusion.datasets.replicated_dataset import ReplicatedDataset
from graph_signal_diffusion.datasets.replicated_sampler import ReplicatedGroupedBatchSampler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sampler(shuffle: bool, n_originals: int = 50, n_replicas: int = 10,
                  originals_per_batch: int = 5) -> ReplicatedGroupedBatchSampler:
    """Build a sampler around a tiny mock ReplicatedDataset."""
    base = MagicMock()
    base.__len__ = MagicMock(return_value=n_originals)
    base.__getitem__ = MagicMock(side_effect=lambda i: i)
    ds = ReplicatedDataset(base, n_replicas=n_replicas)

    generator = torch.Generator().manual_seed(0) if shuffle else None
    return ReplicatedGroupedBatchSampler(
        dataset=ds,
        originals_per_batch=originals_per_batch,
        shuffle=shuffle,
        generator=generator,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestShuffleValKnob:
    """Test that shuffle_val=false produces deterministic batch order."""

    def test_deterministic_when_shuffle_false(self):
        """Two iterations with shuffle=False yield identical batch order."""
        sampler = _make_sampler(shuffle=False)
        batches_1 = list(sampler)
        batches_2 = list(sampler)
        assert batches_1 == batches_2

    def test_varies_when_shuffle_true(self):
        """Two iterations with shuffle=True yield different batch orders
        (generator state advances)."""
        sampler = _make_sampler(shuffle=True)
        batches_1 = list(sampler)
        batches_2 = list(sampler)
        # With 50 originals the probability of identical order is ~1/50! ≈ 0
        assert batches_1 != batches_2

    def test_first_batch_indices_stable_when_no_shuffle(self):
        """First batch contains the first originals_per_batch originals (0..4),
        each expanded to n_replicas consecutive indices."""
        sampler = _make_sampler(shuffle=False, n_originals=20,
                                n_replicas=10, originals_per_batch=3)
        first_batch = list(sampler)[0]
        # Originals 0,1,2 → replica indices 0..9, 10..19, 20..29
        expected = list(range(0, 30))
        assert first_batch == expected

    def test_config_default_is_true(self):
        """sp500.yaml default for shuffle_val is true (backward compat)."""
        import importlib.resources
        from omegaconf import OmegaConf
        import pathlib

        yaml_path = (
            pathlib.Path(__file__).resolve().parents[2]
            / "src" / "graph_signal_diffusion" / "conf" / "dataset" / "sp500.yaml"
        )
        cfg = OmegaConf.load(yaml_path)
        assert cfg.shuffle_val is True
