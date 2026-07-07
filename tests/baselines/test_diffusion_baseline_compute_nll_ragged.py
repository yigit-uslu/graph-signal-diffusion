"""Regression test for `DiffusionBaseline.compute_nll` with ragged cached batches.

The pre-fix implementation required `B % B_template == 0` where `B_template`
was the size of the *first* cached batch. In practice the final cached batch
is typically smaller (dataset length not divisible by `batch_size_val`), which
caused `compute_nll` to raise `ValueError: compute_nll expects B divisible by
template batch size` when a cross-scoring baseline (e.g. GRW) evaluated the
full test split.

The rewrite iterates cached batches by their own `num_graphs`, so this test
drives `compute_nll` with:
  * Uniform cached batches of size 4, total B=234 (the exact numbers from the
    user's failing run on the full SP500 test split: cached = [4]*58 + [2]).
  * Short inputs (B < cached_total), to verify early termination.
  * Over-long inputs (B > cached_total), to verify NaN padding.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import List

import pytest
import torch
from torch_geometric.data import Batch, Data

from graph_signal_diffusion.baselines.diffusion.baseline import DiffusionBaseline


def _make_data(num_graphs: int, N: int, T: int) -> Batch:
    """Build a PyG Batch with `num_graphs` trivial graphs of N nodes.

    Always returns a Batch (even for num_graphs=1) — mirrors the post-fix
    invariant in baseline.py's hook caching path. Pre-fix this helper
    returned a raw Data for num_graphs=1, which fed the bug rather than
    catching it; the production code now always wraps in Batch.
    """
    data_list: List[Data] = []
    for _ in range(num_graphs):
        data_list.append(
            Data(
                x=torch.randn(N, 4),
                y=torch.zeros(N, T, 1),
                edge_index=torch.tensor([[0], [0]], dtype=torch.long),
            )
        )
    return Batch.from_data_list(data_list)


class _FakeDiffusion:
    """Stand-in for `trainer.diffusion` that returns per-trajectory scores
    shaped `[num_graphs]` and records inputs for assertions."""
    def __init__(self):
        self.calls: List[int] = []

    def compute_elbo_per_trajectory(self, data, n_mc_samples: int = 1):
        self.calls.append(int(data.num_graphs))
        return torch.full((int(data.num_graphs),), float(data.num_graphs))


class _FakeTrainer:
    def __init__(self):
        self.diffusion = _FakeDiffusion()


def _make_baseline(cached: List, trainer: _FakeTrainer) -> DiffusionBaseline:
    """Construct a DiffusionBaseline object without running __init__."""
    bl = DiffusionBaseline.__new__(DiffusionBaseline)
    bl._cached_eval_data_batches = cached
    bl.elbo_n_mc_samples = 1
    bl.device = torch.device("cpu")
    bl.trainer = trainer
    bl._require_trainer = lambda: trainer  # short-circuit pipeline build
    return bl


def test_ragged_final_batch_reproduces_full_test_split_case():
    """Cached batches mirror the user's failing run: 58 batches of 4 + 1 batch of 2."""
    N, T = 5, 3
    cached = [_make_data(4, N, T) for _ in range(58)] + [_make_data(2, N, T)]
    bl = _make_baseline(cached, _FakeTrainer())
    returns = torch.randn(234, T, N)

    nll = bl.compute_nll(returns)

    assert nll.shape == (234,)
    assert torch.isfinite(nll).all(), "No NaNs expected: cached covers all 234 windows"
    # Fake diffusion returns num_graphs for each chunk; expected sequence:
    assert bl.trainer.diffusion.calls == [4] * 58 + [2]


def test_uniform_cached_batches_still_work():
    """Regression guard against the previous uniform-size assumption."""
    N, T = 4, 2
    cached = [_make_data(8, N, T) for _ in range(3)]  # 24 total
    bl = _make_baseline(cached, _FakeTrainer())
    returns = torch.randn(24, T, N)

    nll = bl.compute_nll(returns)

    assert nll.shape == (24,)
    assert torch.isfinite(nll).all()
    assert bl.trainer.diffusion.calls == [8, 8, 8]


def test_short_input_terminates_early_without_nans():
    N, T = 3, 2
    cached = [_make_data(5, N, T) for _ in range(4)]  # 20 total
    bl = _make_baseline(cached, _FakeTrainer())
    returns = torch.randn(12, T, N)  # fewer than cached_total

    nll = bl.compute_nll(returns)

    assert nll.shape == (12,)
    assert torch.isfinite(nll).all()
    # Consumes 5 + 5 + 2 (partial third batch), never touches the fourth
    assert bl.trainer.diffusion.calls == [5, 5, 2]


def test_over_long_input_pads_remainder_with_nan():
    """If the caller passes more windows than cached batches cover, the
    missing tail is filled with NaN instead of crashing."""
    N, T = 3, 2
    cached = [_make_data(4, N, T)]  # only 4 cached windows
    bl = _make_baseline(cached, _FakeTrainer())
    returns = torch.randn(10, T, N)

    nll = bl.compute_nll(returns)

    assert nll.shape == (10,)
    assert torch.isfinite(nll[:4]).all(), "First 4 should score cleanly"
    assert torch.isnan(nll[4:]).all(), "Uncovered tail should be NaN-padded"


def test_accepts_4d_returns_with_trailing_feature_dim():
    N, T = 3, 2
    cached = [_make_data(4, N, T)]
    bl = _make_baseline(cached, _FakeTrainer())
    returns_4d = torch.randn(4, T, N, 1)

    nll = bl.compute_nll(returns_4d)

    assert nll.shape == (4,)
    assert torch.isfinite(nll).all()


def test_single_graph_cached_batches_work_for_compute_nll():
    """Cached batches with num_graphs=1 must not break compute_nll.

    Reproduces the cyber-doberman-15 failure mode: with n_samples=50 ==
    batch_size_val=50 on a ReplicatedDataset, every loader batch contains
    exactly one unique window (50 replicas), so the hook's dedup produces a
    single-element list per batch. Pre-fix the hook stored a raw Data here
    and compute_nll's `b.num_graphs` accumulation raised AttributeError.
    """
    N, T = 5, 3
    cached = [_make_data(1, N, T) for _ in range(50)]
    bl = _make_baseline(cached, _FakeTrainer())
    returns = torch.randn(50, T, N)

    nll = bl.compute_nll(returns)

    assert nll.shape == (50,)
    assert torch.isfinite(nll).all()
    assert bl.trainer.diffusion.calls == [1] * 50


def test_compute_elbo_nll_tensors_handles_single_graph_batches():
    """Self-scoring NLL path (post-evaluate cleanup) must accept single-graph
    cached batches. This is the *specific* code path that raised in the
    cyber-doberman-15 compare-baselines log:
        "Diffusion ELBO self-scoring failed; skipping NLL proxy."
        AttributeError: 'GlobalStorage' object has no attribute 'num_graphs'
    """
    N, T = 5, 3
    n_batches = 4
    cached = [_make_data(1, N, T) for _ in range(n_batches)]
    all_real = [torch.randn(1, T, N) for _ in range(n_batches)]
    all_gen = [torch.randn(1, T, N) for _ in range(n_batches)]

    bl = _make_baseline(cached, _FakeTrainer())
    bl._compute_elbo_nll_tensors(all_real, all_gen, cached)

    assert bl.last_nll_tensors is not None
    assert bl.last_nll_tensors["nll_real"].shape == (n_batches,)
    assert bl.last_nll_tensors["nll_gen"].shape == (n_batches,)
    # 2 calls per cached batch (one for real, one for gen), each on 1 graph.
    assert bl.trainer.diffusion.calls == [1] * (2 * n_batches)


def test_missing_cache_raises():
    bl = DiffusionBaseline.__new__(DiffusionBaseline)
    bl._cached_eval_data_batches = []
    with pytest.raises(RuntimeError, match="compute_nll requires cached evaluation data"):
        bl.compute_nll(torch.randn(4, 2, 3))
