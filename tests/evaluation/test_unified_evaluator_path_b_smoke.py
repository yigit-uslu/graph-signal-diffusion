"""Smoke tests for UnifiedEvaluator Path-B (non-ensemble baselines)."""
from __future__ import annotations

import pytest
import torch

from graph_signal_diffusion.evaluation.evaluator import UnifiedEvaluator


class _FakeData:
    def __init__(self, y: torch.Tensor, num_graphs: int, batch_idx: int):
        self.y = y
        self.num_graphs = num_graphs
        self.batch_idx = batch_idx

    def to(self, _device):
        return self


class _PathBBaseline:
    def __init__(self):
        self.device = torch.device("cpu")

    @staticmethod
    def predict(data) -> torch.Tensor:
        # Return [B*N, T, F], matching the evaluator's expected Path-B contract.
        return data.y.clone()


class _TrackingTask:
    def __init__(self):
        self.seen_batch_indices: list[int] = []

    @staticmethod
    def prepare_data(data):
        return {
            "samples": data.y.clone(),  # [B*N, T, F]
            "metadata": {"batch_idx": data.batch_idx, "batch_size": data.num_graphs},
        }

    def evaluate_samples(self, predictions, targets, metadata, viz_save_dir=None):
        _ = predictions, targets, viz_save_dir
        self.seen_batch_indices.append(int(metadata["batch_idx"]))
        # Per-batch scalar used to validate averaging behavior.
        return {"meta_metric": float(metadata["batch_idx"] + 1)}


def test_path_b_uses_per_batch_metadata_and_averages(monkeypatch: pytest.MonkeyPatch) -> None:
    """Path-B must evaluate each batch with its own metadata dict."""
    B, N, T, F = 1, 2, 3, 1
    y0 = torch.arange(B * N * T * F, dtype=torch.float32).view(B * N, T, F)
    y1 = y0 + 100.0
    loader = [
        _FakeData(y=y0, num_graphs=B, batch_idx=0),
        _FakeData(y=y1, num_graphs=B, batch_idx=1),
    ]

    # Keep this smoke test focused on Path-B metadata handling.
    monkeypatch.setattr(
        "graph_signal_diffusion.evaluation.evaluator.compute_all_metrics",
        lambda predictions, targets, prefix="": {"full_tensor_metric": float(predictions.shape[0])},
    )

    task = _TrackingTask()
    baseline = _PathBBaseline()
    evaluator = UnifiedEvaluator(task=task, loaders={"test": loader}, save_dir=None)

    metrics = evaluator.evaluate_baseline_on_split(
        baseline=baseline,
        baseline_name="path_b_smoke",
        split="test",
    )

    # Ensure per-batch metadata was consumed in order.
    assert task.seen_batch_indices == [0, 1]
    # Averaged from batch metrics {1.0, 2.0}
    assert metrics["meta_metric"] == pytest.approx(1.5)
    # compute_all_metrics ran on full concatenated [B_total, T, N, F]
    assert metrics["full_tensor_metric"] == pytest.approx(2.0)
