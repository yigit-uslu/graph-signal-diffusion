"""Smoke tests for WMMSE baseline evaluation behavior."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from graph_signal_diffusion.baselines.wireless_resource_allocation.wmmse import (
    WMMSEBaseline,
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
        "h_ls_gains": [np.eye(n_links, dtype=np.float64) for _ in range(batch_size)],
        "associations": [np.eye(n_links, dtype=np.float64) for _ in range(batch_size)],
        "system_params": {"P_max": 1.0, "noise_var": 1.0e-10},
    }


def test_wmmse_raises_for_multi_feature_targets() -> None:
    """WMMSE must fail loudly if WRA target schema changes to F != 1."""
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

    baseline = WMMSEBaseline(device="cpu", num_wmmse_iters=1, log_every_n_batches=0)
    with pytest.raises(ValueError, match="F==1"):
        baseline.evaluate([_FakeData(num_graphs=B)], _Task(), max_batches=1)


def test_wmmse_broadcasts_static_allocation_across_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """Static WMMSE allocation should be repeated for all T steps."""
    B, T, N, F = 1, 3, 2, 1
    targets = torch.zeros(B, T, N, F)
    metadata = _make_wra_metadata(batch_size=B, n_links=N)
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

    monkeypatch.setattr(
        WMMSEBaseline,
        "_wmmse_solve",
        staticmethod(
            lambda *_args, **_kwargs: np.array([0.2, 0.8], dtype=np.float64)
        ),
    )

    baseline = WMMSEBaseline(
        device="cpu",
        n_samples=3,
        num_wmmse_iters=1,
        log_every_n_batches=0,
    )
    metrics = baseline.evaluate([_FakeData(num_graphs=B)], _Task(), max_batches=1)

    assert metrics["rate_mean"] == pytest.approx(1.0)
    assert "predictions" in captured
    observed = captured["predictions"][0, :, :, 0]  # [T, N]
    expected_per_node = torch.tensor([-0.3, 0.3], dtype=torch.float32)
    expected = expected_per_node.unsqueeze(0).expand(T, -1)
    assert observed.shape == (T, N)
    assert torch.allclose(observed, expected)
    assert captured["meta"]["n_samples_per_input"] == 3
    assert baseline.needs_replicated_loader is True


def test_wmmse_per_key_metric_counting(monkeypatch: pytest.MonkeyPatch) -> None:
    """subdataset/ metrics appearing in only some batches must not be diluted."""
    B, T, N, F = 1, 2, 2, 1
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

    monkeypatch.setattr(
        WMMSEBaseline,
        "_wmmse_solve",
        staticmethod(lambda *_a, **_kw: np.array([0.5, 0.5], dtype=np.float64)),
    )

    baseline = WMMSEBaseline(
        device="cpu", n_samples=1, num_wmmse_iters=1, log_every_n_batches=0,
    )
    loader = [_FakeData(num_graphs=B), _FakeData(num_graphs=B)]
    result = baseline.evaluate(loader, _Task(), max_batches=2)

    assert result["global_rate"] == pytest.approx(4.0)
    assert result["subdataset/hAAA/rate"] == pytest.approx(10.0)
    assert result["subdataset/hBBB/rate"] == pytest.approx(20.0)
