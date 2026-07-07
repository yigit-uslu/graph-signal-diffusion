"""Smoke tests for average-power WRA baseline evaluation behavior."""
from __future__ import annotations

import pytest
import torch

from graph_signal_diffusion.baselines.wireless_resource_allocation.average_power import (
    AveragePowerBaseline,
)


class _FakeData:
    def __init__(self, num_graphs: int, *, y: torch.Tensor | None = None):
        self.num_graphs = num_graphs
        if y is None:
            y = torch.zeros(num_graphs, 1, 1)
        self.y = y

    def to(self, _device):
        return self


def _make_wra_metadata(batch_size: int, n_links: int) -> dict:
    return {
        "batch_size": batch_size,
        "dataset_names": ["wra"] * batch_size,
        "network_ids": list(range(batch_size)),
        "system_params": {"P_max": 1.0, "noise_var": 1.0e-10},
        "associations": [torch.eye(n_links)] * batch_size,
    }


def test_average_power_uses_per_network_per_receiver_means() -> None:
    """AP should average expert powers per (dataset, network_id) and receiver."""
    B, T, N, F = 4, 2, 3, 1
    targets = torch.tensor(
        [
            [[[0.1], [0.2], [0.3]], [[0.3], [0.4], [0.5]]],  # net 10
            [[[0.5], [0.6], [0.7]], [[0.7], [0.8], [0.9]]],  # net 10
            [[[-0.2], [-0.1], [0.0]], [[0.0], [0.1], [0.2]]],  # net 11
            [[[0.2], [0.3], [0.4]], [[0.4], [0.5], [0.6]]],  # net 11
        ],
        dtype=torch.float32,
    )  # [B, T, N, F]

    metadata = _make_wra_metadata(batch_size=B, n_links=N)
    metadata["network_ids"] = [10, 10, 11, 11]

    captured: dict[str, object] = {}

    class _Task:
        @staticmethod
        def prepare_data(_data):
            return {"samples": targets, "metadata": metadata}

        @staticmethod
        def evaluate_samples(predictions, _targets, _meta, viz_save_dir=None):
            _ = viz_save_dir
            captured["predictions"] = predictions.detach().cpu()
            captured["meta"] = dict(_meta)
            return {"rate_mean": 1.0}

    baseline = AveragePowerBaseline(device="cpu", n_samples=4, log_every_n_batches=0)
    metrics = baseline.evaluate([_FakeData(num_graphs=B)], _Task(), max_batches=1)

    assert metrics["rate_mean"] == pytest.approx(1.0)
    assert baseline.needs_replicated_loader is True
    assert captured["meta"]["n_samples_per_input"] == 4

    observed = captured["predictions"][:, :, :, 0]  # [B, T, N]
    expected_net10 = torch.tensor([0.4, 0.5, 0.6], dtype=torch.float32)
    expected_net11 = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)

    assert torch.allclose(observed[0], expected_net10.unsqueeze(0).expand(T, -1))
    assert torch.allclose(observed[1], expected_net10.unsqueeze(0).expand(T, -1))
    assert torch.allclose(observed[2], expected_net11.unsqueeze(0).expand(T, -1))
    assert torch.allclose(observed[3], expected_net11.unsqueeze(0).expand(T, -1))


def test_average_power_raises_for_multi_feature_targets() -> None:
    """AP must fail loudly if WRA target schema changes to F != 1."""
    B, T, N, F = 1, 2, 2, 2
    targets = torch.zeros(B, T, N, F)
    metadata = _make_wra_metadata(batch_size=B, n_links=N)

    class _Task:
        @staticmethod
        def prepare_data(_data):
            return {"samples": targets, "metadata": metadata}

        @staticmethod
        def evaluate_samples(*_args, **_kwargs):
            raise AssertionError("evaluate_samples should not be called when F != 1")

    baseline = AveragePowerBaseline(device="cpu", n_samples=2, log_every_n_batches=0)
    with pytest.raises(ValueError, match="F==1"):
        baseline.evaluate([_FakeData(num_graphs=B)], _Task(), max_batches=1)


def test_average_power_per_key_metric_counting() -> None:
    """subdataset/ metrics appearing in only some batches must not be diluted."""
    B, T, N, F = 1, 2, 3, 1
    targets = torch.zeros(B, T, N, F)
    call_count = 0

    class _Task:
        @staticmethod
        def prepare_data(_data):
            return {
                "samples": targets,
                "metadata": _make_wra_metadata(batch_size=B, n_links=N),
            }

        @staticmethod
        def evaluate_samples(_pred, _tgt, _meta, viz_save_dir=None):
            nonlocal call_count
            call_count += 1
            metrics = {"global_rate": 4.0}
            if call_count == 1:
                metrics["subdataset/hAAA/rate"] = 10.0
            else:
                metrics["subdataset/hBBB/rate"] = 20.0
            return metrics

    baseline = AveragePowerBaseline(device="cpu", n_samples=1, log_every_n_batches=0)
    loader = [_FakeData(num_graphs=B), _FakeData(num_graphs=B)]
    result = baseline.evaluate(loader, _Task(), max_batches=2)

    assert result["global_rate"] == pytest.approx(4.0)
    assert result["subdataset/hAAA/rate"] == pytest.approx(10.0)
    assert result["subdataset/hBBB/rate"] == pytest.approx(20.0)
