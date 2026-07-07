"""Smoke tests for full-power WRA baseline evaluation behavior."""
from __future__ import annotations

import pytest
import torch

from graph_signal_diffusion.baselines.wireless_resource_allocation.full_power import (
    FullPowerBaseline,
)


class _FakeData:
    def __init__(self, num_graphs: int, *, y: torch.Tensor | None = None):
        self.num_graphs = num_graphs
        if y is None:
            y = torch.zeros(num_graphs, 1, 1)
        self.y = y

    def to(self, _device):
        return self


def test_full_power_propagates_loader_sample_count_to_task_metadata() -> None:
    """Full-power baseline should pass configured n_samples_per_input to task."""
    B, T, N, F = 1, 2, 3, 1
    targets = torch.zeros(B, T, N, F)
    captured: dict[str, object] = {}

    class _Task:
        @staticmethod
        def prepare_data(_data):
            return {
                "samples": targets,
                "metadata": {
                    "batch_size": B,
                    "dataset_names": ["wra"],
                    "network_ids": [0],
                    "system_params": {"P_max": 1.0, "noise_var": 1.0e-10},
                    "associations": [torch.eye(N)],
                },
            }

        @staticmethod
        def evaluate_samples(predictions, _targets, _meta, viz_save_dir=None):
            _ = viz_save_dir
            captured["predictions"] = predictions.detach().cpu()
            captured["meta"] = dict(_meta)
            return {"rate_mean": 1.0}

    baseline = FullPowerBaseline(device="cpu", n_samples=4, log_every_n_batches=0)
    metrics = baseline.evaluate([_FakeData(num_graphs=B)], _Task(), max_batches=1)

    assert metrics["rate_mean"] == pytest.approx(1.0)
    assert baseline.needs_replicated_loader is True
    assert captured["meta"]["n_samples_per_input"] == 4
    observed = captured["predictions"][0, :, :, 0]  # [T, N]
    assert observed.shape == (T, N)
    assert torch.allclose(observed, torch.full((T, N), 0.5))


def test_full_power_per_key_metric_counting() -> None:
    """subdataset/ metrics appearing in only some batches must not be diluted."""
    B, T, N, F = 1, 2, 3, 1
    targets = torch.zeros(B, T, N, F)
    call_count = 0

    class _Task:
        @staticmethod
        def prepare_data(_data):
            return {
                "samples": targets,
                "metadata": {
                    "batch_size": B,
                    "dataset_names": ["wra"],
                    "network_ids": [0],
                    "system_params": {"P_max": 1.0, "noise_var": 1e-10},
                    "associations": [torch.eye(N)],
                },
            }

        @staticmethod
        def evaluate_samples(_pred, _tgt, _meta, viz_save_dir=None):
            nonlocal call_count
            call_count += 1
            # Global metric in both batches; subdataset keys only in one each
            metrics = {"global_rate": 4.0}
            if call_count == 1:
                metrics["subdataset/hAAA/rate"] = 10.0
            else:
                metrics["subdataset/hBBB/rate"] = 20.0
            return metrics

    baseline = FullPowerBaseline(device="cpu", n_samples=1, log_every_n_batches=0)
    loader = [_FakeData(num_graphs=B), _FakeData(num_graphs=B)]
    result = baseline.evaluate(loader, _Task(), max_batches=2)

    # Global: (4+4)/2 = 4.0
    assert result["global_rate"] == pytest.approx(4.0)
    # Each subdataset key appeared in exactly 1 batch -> value / 1, not / 2
    assert result["subdataset/hAAA/rate"] == pytest.approx(10.0)
    assert result["subdataset/hBBB/rate"] == pytest.approx(20.0)
