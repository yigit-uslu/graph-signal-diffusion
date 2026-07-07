"""Tests for n_split_chunks interleaved chronological splitting."""

import torch
import pytest
from omegaconf import OmegaConf


# ---------------------------------------------------------------------------
# Helpers — replicate the split logic in isolation (no dataset I/O needed)
# ---------------------------------------------------------------------------

def _split(N: int, train_frac: float = 0.8, n_chunks: int = 1):
    """Reproduce the datamodule split logic for a dataset of length N."""
    val_frac = (1 - train_frac) / 2

    if n_chunks == 1:
        train_idx = torch.arange(0, int(N * train_frac))
        val_idx = torch.arange(train_idx[-1] + 1,
                               int(N * (train_frac + val_frac)))
        test_idx = torch.arange(val_idx[-1] + 1, N)
        train_val_idx = torch.arange(
            int(N * (train_frac - val_frac)),
            int(N * train_frac),
        )
    else:
        chunk_size = N // n_chunks
        train_ids, val_ids, test_ids = [], [], []
        for c in range(n_chunks):
            start = c * chunk_size
            end = (start + chunk_size) if c < n_chunks - 1 else N
            n = end - start
            t_end = start + int(n * train_frac)
            v_end = start + int(n * (train_frac + val_frac))
            train_ids.append(torch.arange(start, t_end))
            val_ids.append(torch.arange(t_end, v_end))
            test_ids.append(torch.arange(v_end, end))
        train_idx = torch.cat(train_ids)
        val_idx = torch.cat(val_ids)
        test_idx = torch.cat(test_ids)
        train_val_idx = train_ids[-1]

    return train_idx, val_idx, test_idx, train_val_idx


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSplitChunks:

    def test_single_chunk_matches_legacy(self):
        """n_chunks=1 produces the same indices as the original split."""
        N = 1000
        t1, v1, te1, tv1 = _split(N, n_chunks=1)
        # Legacy hand-computed:
        assert t1[0] == 0 and t1[-1] == 799
        assert v1[0] == 800 and v1[-1] == 899
        assert te1[0] == 900 and te1[-1] == 999
        assert tv1[0] == 700 and tv1[-1] == 799

    def test_no_overlap_single_chunk(self):
        t, v, te, tv = _split(1000, n_chunks=1)
        all_idx = torch.cat([t, v, te])
        assert len(all_idx) == len(all_idx.unique()), "Overlap detected"

    def test_no_overlap_multi_chunk(self):
        for n_chunks in [2, 3, 5, 10]:
            t, v, te, tv = _split(1000, n_chunks=n_chunks)
            all_idx = torch.cat([t, v, te])
            assert len(all_idx) == len(all_idx.unique()), \
                f"Overlap with {n_chunks} chunks"

    def test_full_coverage_multi_chunk(self):
        """All indices 0..N-1 are assigned to exactly one split."""
        N = 1000
        for n_chunks in [1, 3, 5]:
            t, v, te, tv = _split(N, n_chunks=n_chunks)
            all_idx = torch.cat([t, v, te]).sort().values
            assert torch.equal(all_idx, torch.arange(N)), \
                f"Missing indices with {n_chunks} chunks"

    def test_ratio_approximately_80_10_10(self):
        """Each split should be approximately 80-10-10 of total."""
        N = 2000
        for n_chunks in [1, 2, 4, 5]:
            t, v, te, tv = _split(N, n_chunks=n_chunks)
            assert abs(len(t) / N - 0.8) < 0.02, f"Train ratio off with {n_chunks} chunks"
            assert abs(len(v) / N - 0.1) < 0.02, f"Val ratio off with {n_chunks} chunks"
            assert abs(len(te) / N - 0.1) < 0.02, f"Test ratio off with {n_chunks} chunks"

    def test_interleaving_order(self):
        """Within each chunk, train < val < test indices (chronological)."""
        N = 1200
        n_chunks = 3
        chunk_size = N // n_chunks
        t, v, te, tv = _split(N, n_chunks=n_chunks)
        for c in range(n_chunks):
            start = c * chunk_size
            end = (start + chunk_size) if c < n_chunks - 1 else N
            t_in_chunk = t[(t >= start) & (t < end)]
            v_in_chunk = v[(v >= start) & (v < end)]
            te_in_chunk = te[(te >= start) & (te < end)]
            if len(t_in_chunk) > 0 and len(v_in_chunk) > 0:
                assert t_in_chunk.max() < v_in_chunk.min(), \
                    f"Chunk {c}: train not before val"
            if len(v_in_chunk) > 0 and len(te_in_chunk) > 0:
                assert v_in_chunk.max() < te_in_chunk.min(), \
                    f"Chunk {c}: val not before test"

    def test_train_val_is_last_chunk_train(self):
        """train_val should equal the train portion of the last chunk."""
        N = 1000
        for n_chunks in [2, 3, 5]:
            t, v, te, tv = _split(N, n_chunks=n_chunks)
            # Reconstruct last chunk train manually
            chunk_size = N // n_chunks
            last_start = (n_chunks - 1) * chunk_size
            last_end = N
            n = last_end - last_start
            last_t_end = last_start + int(n * 0.8)
            expected_tv = torch.arange(last_start, last_t_end)
            assert torch.equal(tv, expected_tv), \
                f"train_val mismatch with {n_chunks} chunks"

    def test_config_default_is_one(self):
        """sp500.yaml default for n_split_chunks is 1."""
        import pathlib
        yaml_path = (
            pathlib.Path(__file__).resolve().parents[2]
            / "src" / "graph_signal_diffusion" / "conf" / "dataset" / "sp500.yaml"
        )
        cfg = OmegaConf.load(yaml_path)
        assert cfg.n_split_chunks == 1
