"""Unit tests for ``cfg.test_subsample_n`` linspace subsampling.

Validates the algorithm used by ``sp500/datamodule.py`` to deterministically
pick K equi-spaced indices into ``test_idx`` so that ``eval_splits={test: 1}``
no longer biases the eval toward chunk 0 of an interleaved chronological
split.

The test exercises the algorithm in isolation rather than constructing a
full SP500 dataset, since the slicing is a 2-line numpy expression and
the rest of the datamodule is unrelated.
"""

from __future__ import annotations

import numpy as np
import torch


def _subsample(test_idx: torch.Tensor, k: int) -> torch.Tensor:
    """Mirror of the slicing block in sp500/datamodule.py."""
    pick = np.linspace(0, len(test_idx) - 1, num=k).round().astype(np.int64)
    pick = np.unique(pick)
    return test_idx[torch.from_numpy(pick)]


def test_subsample_basic_50_of_234():
    # Mimic SP500 test_idx with 234 unique chronological windows.
    test_idx = torch.arange(234)
    out = _subsample(test_idx, 50)

    # Length: linspace at K=50 over N=234 gives no rounding collisions
    assert len(out) == 50

    # Endpoints included
    assert out[0].item() == 0
    assert out[-1].item() == 233

    # Strictly monotonic
    diffs = (out[1:] - out[:-1]).tolist()
    assert all(d > 0 for d in diffs), f"non-monotonic picks: {diffs}"

    # Roughly equi-spaced (max gap and min gap differ by at most 1 step)
    assert max(diffs) - min(diffs) <= 1


def test_subsample_full_coverage_across_chunks():
    # Construct a test_idx that mimics the n_split_chunks=3 layout:
    # 3 narrow bands at ~28-33%, 62-67%, 95-100% of the timeline.
    N = 1000
    chunk0 = torch.arange(280, 330)
    chunk1 = torch.arange(620, 670)
    chunk2 = torch.arange(950, 1000)
    test_idx = torch.cat([chunk0, chunk1, chunk2])  # 150 windows total
    assert len(test_idx) == 150

    # Subsample 50 → should pull ~17 windows from each chunk.
    out = _subsample(test_idx, 50)
    assert len(out) == 50

    # Each chunk's representation should be > 10 (much better than the
    # 0 windows you'd get with the front-loaded slicing test_idx[:50],
    # which would all come from chunk 0).
    in_chunk0 = ((out >= 280) & (out < 330)).sum().item()
    in_chunk1 = ((out >= 620) & (out < 670)).sum().item()
    in_chunk2 = ((out >= 950) & (out < 1000)).sum().item()
    assert in_chunk0 + in_chunk1 + in_chunk2 == 50
    assert in_chunk0 >= 10, f"chunk 0 underrepresented: {in_chunk0}"
    assert in_chunk1 >= 10, f"chunk 1 underrepresented: {in_chunk1}"
    assert in_chunk2 >= 10, f"chunk 2 underrepresented: {in_chunk2}"

    # First/last test windows always included
    assert out[0].item() == chunk0[0].item()
    assert out[-1].item() == chunk2[-1].item()


def test_subsample_deterministic():
    test_idx = torch.arange(234)
    out_a = _subsample(test_idx, 50)
    out_b = _subsample(test_idx, 50)
    assert torch.equal(out_a, out_b)


def test_subsample_K_geq_N_returns_full():
    """When K >= len(test_idx), the linspace pick covers every index."""
    test_idx = torch.arange(40)
    out = _subsample(test_idx, 100)  # K > N
    # np.unique collapses the duplicates that linspace produces.
    assert len(out) == 40
    assert torch.equal(out, test_idx)


def test_subsample_K_equals_N():
    test_idx = torch.arange(50)
    out = _subsample(test_idx, 50)
    assert len(out) == 50
    assert torch.equal(out, test_idx)


# ---------------------------------------------------------------------------
# Phase-distribution sanity check (mirrors the warning in datamodule.py)
# ---------------------------------------------------------------------------

def _phase_counts(test_idx: torch.Tensor, future_window: int) -> list:
    """Same logic the datamodule uses for the warning."""
    phases = (test_idx.numpy() % future_window).astype(np.int64)
    return np.bincount(phases, minlength=future_window).tolist()


def test_phase_coverage_full_under_default_subsample():
    """K=50, N=234, T=5: linspace stride is non-integer → all 5 phases hit."""
    # Realistic mock of test_idx after the n_split_chunks=3 split:
    # 3 contiguous bands at chunk-0/1/2 test slices, each ~78 windows.
    test_idx = torch.cat([
        torch.arange(280, 358),
        torch.arange(640, 718),
        torch.arange(1000, 1078),
    ])
    assert len(test_idx) == 234

    out = _subsample(test_idx, 50)
    counts = _phase_counts(out, future_window=5)
    assert all(c > 0 for c in counts), f"empty phase: {counts}"
    # Reasonably balanced
    assert max(counts) <= 2 * min(counts), f"unbalanced: {counts}"


def test_phase_coverage_degenerate_when_stride_multiple_of_T():
    """Pathological K such that stride is exactly a multiple of T_steps."""
    # N=200, T=5, K=41 → linspace stride = 199/40 = 4.975, not exact multiple.
    # To force degeneracy, build a test_idx whose values + stride collapse to
    # one residue class. Easiest: take strided picks from a uniform range
    # whose stride is exactly T.
    test_idx = torch.arange(0, 200, 1)  # 200 contiguous test windows
    # K=40 → linspace stride = 199/39 ≈ 5.103 (close to 5 but not exact).
    # K=41 → 199/40 = 4.975. Neither is exactly 5.
    # K=40, but apply linspace then check phases:
    out = _subsample(test_idx, 40)
    counts = _phase_counts(out, future_window=5)
    # 40 picks across 5 phases on a stride-1 contiguous test_idx:
    # linspace gives picks at 0, 5.10, 10.21, 15.31, ... → rounded {0, 5,
    # 10, 15, 20, 26, 31, ...}.  All 5 phases reachable.
    assert all(c > 0 for c in counts), f"phase coverage broken: {counts}"

    # Now construct a *truly* degenerate case: pick K such that linspace
    # stride is exact 5.0 → K = N/T + 1 with appropriate N.
    # N=201, K=41: stride = 200/40 = 5.0 exactly.
    test_idx = torch.arange(0, 201)
    out = _subsample(test_idx, 41)
    counts = _phase_counts(out, future_window=5)
    # All picks land at indices {0, 5, 10, ..., 200}, all phase 0.
    assert counts[0] == len(out)
    assert sum(c > 0 for c in counts) == 1, f"expected single-phase: {counts}"


