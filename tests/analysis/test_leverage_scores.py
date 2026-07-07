import numpy as np
import pytest

from graph_signal_diffusion.analysis.leverage_scores import (
    compare_leverage_vs_selection,
    dense_W_from_edges,
    level_learned_importance,
    leverage_scores,
    load_symbols,
    low_noise_probe_indices,
    rank_agreement,
)


def _two_cluster_W():
    """Two triangles (nodes 0-1-2 and 3-4-5) joined by a single bridge 2-3."""
    W = np.zeros((6, 6))
    for i, j in [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5), (2, 3)]:
        W[i, j] = W[j, i] = 1.0
    return W


# --------------------------------------------------------------------------- #
# leverage_scores invariants
# --------------------------------------------------------------------------- #
def test_leverage_scores_sum_equals_K():
    """Orthonormal eigenvectors => Σ_v ℓ_v == K, for both Laplacians."""
    W = _two_cluster_W()
    for normalized in (True, False):
        for K in (1, 2, 3, 6):
            lev = leverage_scores(W, K=K, normalized=normalized)
            assert lev.shape == (6,)
            assert np.all(lev >= -1e-9)
            np.testing.assert_allclose(lev.sum(), K, atol=1e-6)


def test_leverage_scores_K1_normalized_is_degree_distribution():
    """Normalized symmetric Laplacian: λ=0 eigenvector ∝ sqrt(d) =>
    ℓ_v(K=1) == d_v / Σ d."""
    W = _two_cluster_W()
    d = W.sum(axis=1)
    lev = leverage_scores(W, K=1, normalized=True)
    np.testing.assert_allclose(lev, d / d.sum(), atol=1e-6)


def test_leverage_scores_K1_combinatorial_is_uniform():
    """Combinatorial Laplacian on a connected graph: λ=0 eigenvector is
    constant => ℓ_v(K=1) == 1/N uniform."""
    W = _two_cluster_W()
    lev = leverage_scores(W, K=1, normalized=False)
    np.testing.assert_allclose(lev, np.full(6, 1.0 / 6.0), atol=1e-6)


def test_leverage_scores_K_clamped_to_N():
    W = _two_cluster_W()
    lev = leverage_scores(W, K=999, normalized=True)
    # K clamps to N=6 => every node has full unit leverage (Σ_k u_k(v)^2 = 1).
    np.testing.assert_allclose(lev, np.ones(6), atol=1e-6)


def test_leverage_scores_symmetrises_asymmetric_input():
    W = _two_cluster_W()
    W_asym = W.copy()
    W_asym[0, 1] = 5.0  # break symmetry; (1,0) stays 1.0
    lev_sym = leverage_scores((W_asym + W_asym.T) / 2.0, K=2, normalized=True)
    lev_auto = leverage_scores(W_asym, K=2, normalized=True)
    np.testing.assert_allclose(lev_auto, lev_sym, atol=1e-9)


def test_leverage_scores_rejects_non_square():
    with pytest.raises(ValueError):
        leverage_scores(np.zeros((3, 4)), K=1)


# --------------------------------------------------------------------------- #
# dense_W_from_edges
# --------------------------------------------------------------------------- #
def test_dense_W_from_edges_weighted():
    edge_index = np.array([[0, 1, 2], [1, 0, 0]])
    edge_weight = np.array([0.7, 0.7, 0.4])
    W = dense_W_from_edges(edge_index, edge_weight, num_nodes=3)
    assert W.shape == (3, 3)
    assert W[0, 1] == 0.7 and W[1, 0] == 0.7 and W[2, 0] == 0.4
    assert W[0, 2] == 0.0  # only (2,0) was given


def test_dense_W_from_edges_unweighted_defaults_to_ones():
    edge_index = np.array([[0, 1], [1, 2]])
    W = dense_W_from_edges(edge_index, None, num_nodes=3)
    assert W[0, 1] == 1.0 and W[1, 2] == 1.0


# --------------------------------------------------------------------------- #
# level_learned_importance
# --------------------------------------------------------------------------- #
def test_level_learned_importance_mean_and_realized_K():
    network_diag = {
        "probe_timesteps": [10, 0],
        "probe_graph_count": [4.0, 4.0],
        # 4 nodes; node 3 never available -> NaN conditional -> dropped from mean.
        "selection_given_available_by_level": {
            "0": [
                [1.0, 0.5, 0.0, float("nan")],
                [0.5, 0.5, 0.0, float("nan")],
            ]
        },
        "selected_count_by_level": {
            "0": [
                [4.0, 2.0, 0.0, 0.0],
                [2.0, 2.0, 0.0, 0.0],
            ]
        },
    }
    importance, K = level_learned_importance(network_diag, "0")
    np.testing.assert_allclose(importance[:3], [0.75, 0.5, 0.0], atol=1e-9)
    assert np.isnan(importance[3])
    # kept per graph: t=10 -> 6/4=1.5 ; t=0 -> 4/4=1.0 ; mean=1.25 -> round=1.
    assert K == 1


def test_low_noise_probe_indices_window_and_fallback():
    ts = [99, 66, 33, 0]
    # fraction 0.1 -> threshold 9.9 -> only t=0 (index 3).
    idx, thr = low_noise_probe_indices(ts, 4, 0.1)
    assert idx == [3] and thr == pytest.approx(9.9)
    # fraction 1.0 -> all timesteps.
    idx, thr = low_noise_probe_indices(ts, 4, 1.0)
    assert idx == [0, 1, 2, 3] and thr == pytest.approx(99.0)
    # No timestep in-window -> fall back to single lowest (argmin).
    idx, thr = low_noise_probe_indices([99, 66, 33], 3, 0.1)
    assert idx == [2]  # t=33 is the lowest
    # Missing / mismatched probe_timesteps -> all rows, no threshold.
    assert low_noise_probe_indices(None, 4, 0.1) == ([0, 1, 2, 3], None)
    assert low_noise_probe_indices([10, 0], 4, 0.1) == ([0, 1, 2, 3], None)


def test_level_learned_importance_low_noise_window():
    network_diag = {
        "probe_timesteps": [100, 50, 10, 0],
        "probe_graph_count": [4.0, 4.0, 4.0, 4.0],
        "selection_given_available_by_level": {
            "0": [
                [0.1, 0.1, 0.1],
                [0.2, 0.2, 0.2],
                [0.9, 0.8, 0.7],
                [1.0, 0.9, 0.8],
            ]
        },
        "selected_count_by_level": {
            "0": [
                [3.0, 3.0, 3.0],
                [3.0, 3.0, 3.0],
                [2.0, 2.0, 2.0],
                [2.0, 2.0, 1.0],
            ]
        },
    }
    # fraction 0.1 -> window {t=10, t=0} -> mean of last two rows.
    imp, K = level_learned_importance(network_diag, "0", probe_timestep_fraction=0.1)
    np.testing.assert_allclose(imp, [0.95, 0.85, 0.75], atol=1e-9)
    # kept/graph over window: t=10 -> 6/4=1.5 ; t=0 -> 5/4=1.25 ; mean=1.375 -> 1.
    assert K == 1

    # fraction 1.0 -> all four timesteps averaged (different result).
    imp_all, _ = level_learned_importance(network_diag, "0", probe_timestep_fraction=1.0)
    np.testing.assert_allclose(imp_all, [0.55, 0.5, 0.45], atol=1e-9)


def test_level_learned_importance_falls_back_to_freq():
    network_diag = {
        "probe_timesteps": [0],
        "selection_freq_by_level": {"0": [[0.2, 0.8]]},
    }
    importance, K = level_learned_importance(network_diag, "0")
    np.testing.assert_allclose(importance, [0.2, 0.8])
    assert K == 2  # no counts -> fall back to count of finite-importance nodes


def test_level_learned_importance_measure_selects_matrix():
    """measure='conditional' reads the conditional matrix; 'freq' reads the
    unconditional frequency matrix."""
    network_diag = {
        "probe_timesteps": [0],
        "selection_given_available_by_level": {"0": [[0.1, 0.2, 0.3]]},
        "selection_freq_by_level": {"0": [[0.9, 0.8, 0.7]]},
    }
    imp_c, _ = level_learned_importance(network_diag, "0", measure="conditional")
    np.testing.assert_allclose(imp_c, [0.1, 0.2, 0.3])
    imp_f, _ = level_learned_importance(network_diag, "0", measure="freq")
    np.testing.assert_allclose(imp_f, [0.9, 0.8, 0.7])


def test_level_learned_importance_measure_falls_back_to_other():
    """Requested measure absent -> falls back to the other available signal."""
    nd_cond_only = {  # freq requested, only conditional present
        "probe_timesteps": [0],
        "selection_given_available_by_level": {"0": [[0.1, 0.2]]},
    }
    imp, _ = level_learned_importance(nd_cond_only, "0", measure="freq")
    np.testing.assert_allclose(imp, [0.1, 0.2])
    nd_freq_only = {  # conditional requested, only freq present
        "probe_timesteps": [0],
        "selection_freq_by_level": {"0": [[0.3, 0.4]]},
    }
    imp2, _ = level_learned_importance(nd_freq_only, "0", measure="conditional")
    np.testing.assert_allclose(imp2, [0.3, 0.4])


def test_level_learned_importance_invalid_measure_raises():
    nd = {"probe_timesteps": [0], "selection_freq_by_level": {"0": [[0.1]]}}
    with pytest.raises(ValueError):
        level_learned_importance(nd, "0", measure="bogus")


def test_level_learned_importance_min_active_frac_masks_and_preserves_K():
    """min_active_frac NaNs out low-availability nodes but leaves K unchanged."""
    network_diag = {
        "probe_timesteps": [10, 0],
        "probe_graph_count": [10.0, 10.0],  # 20 graphs over the window
        "selection_given_available_by_level": {
            "0": [[1.0, 0.5, 0.2, 0.9], [1.0, 0.5, 0.2, 0.9]],
        },
        "available_count_by_level": {
            # node 3 active in only 1 of 20 graphs -> frac 0.05; others 1.0.
            "0": [[10, 10, 10, 1], [10, 10, 10, 0]],
        },
        "selected_count_by_level": {"0": [[10, 5, 2, 1], [10, 5, 2, 0]]},
    }
    imp0, K0 = level_learned_importance(network_diag, "0", min_active_frac=0.0)
    np.testing.assert_allclose(imp0, [1.0, 0.5, 0.2, 0.9])
    # kept/graph: row0=18/10=1.8, row1=17/10=1.7 -> mean 1.75 -> round 2.
    assert K0 == 2

    imp, K = level_learned_importance(network_diag, "0", min_active_frac=0.1)
    np.testing.assert_allclose(imp[:3], [1.0, 0.5, 0.2])
    assert np.isnan(imp[3])  # 0.05 < 0.1 -> masked
    assert K == 2  # mask does not change the budget


def test_level_learned_importance_min_active_frac_missing_available_warns():
    """min_active_frac>0 without available_count -> warn, return unchanged."""
    nd = {
        "probe_timesteps": [0],
        "selection_given_available_by_level": {"0": [[0.1, 0.2, 0.3]]},
    }
    with pytest.warns(UserWarning):
        imp, _ = level_learned_importance(nd, "0", min_active_frac=0.5)
    np.testing.assert_allclose(imp, [0.1, 0.2, 0.3])


# --------------------------------------------------------------------------- #
# rank_agreement
# --------------------------------------------------------------------------- #
def test_rank_agreement_perfect_monotone():
    lev = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    imp = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = rank_agreement(lev, imp, top_k=2)
    assert out["spearman"] == pytest.approx(1.0)
    assert out["pearson"] == pytest.approx(1.0)
    assert out["topk_jaccard"] == pytest.approx(1.0)  # same top-2 nodes
    assert out["n_nodes"] == 5.0


def test_rank_agreement_drops_nan_pairwise():
    lev = np.array([0.1, 0.2, 0.3, 0.4])
    imp = np.array([1.0, np.nan, 3.0, 4.0])
    out = rank_agreement(lev, imp, top_k=2)
    assert out["n_nodes"] == 3.0
    assert out["spearman"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# load_symbols
# --------------------------------------------------------------------------- #
def test_load_symbols_from_stocks_csv_in_root(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "stocks.csv").write_text("Symbol,Name,Sector\nA,Agilent,Health\nAAPL,Apple,Tech\nABBV,AbbVie,Health\n")
    syms = load_symbols(3, dataset_root=str(tmp_path))
    assert syms == ["A", "AAPL", "ABBV"]


def test_load_symbols_explicit_path_takes_priority(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "stocks.csv").write_text("Symbol\nA\nAAPL\nABBV\n")
    explicit = tmp_path / "mine.json"
    explicit.write_text('["X", "Y", "Z"]')
    syms = load_symbols(3, symbols_path=str(explicit), dataset_root=str(tmp_path))
    assert syms == ["X", "Y", "Z"]


def test_load_symbols_txt_one_per_line(tmp_path):
    f = tmp_path / "s.txt"
    f.write_text("AAA\nBBB\nCCC\n")
    assert load_symbols(3, symbols_path=str(f)) == ["AAA", "BBB", "CCC"]


def test_load_symbols_length_mismatch_rejected(tmp_path):
    f = tmp_path / "s.txt"
    f.write_text("AAA\nBBB\n")
    with pytest.warns(UserWarning):
        assert load_symbols(3, symbols_path=str(f)) is None


def test_load_symbols_missing_returns_none(tmp_path):
    assert load_symbols(3, dataset_root=str(tmp_path)) is None


# --------------------------------------------------------------------------- #
# End-to-end (no disk / no model)
# --------------------------------------------------------------------------- #
def test_compare_leverage_vs_selection_end_to_end(tmp_path):
    W = _two_cluster_W()
    # Learned importance that mirrors degree -> should correlate with K=1 normalized leverage.
    d = W.sum(axis=1)
    imp_row = (d / d.max()).tolist()
    diagnostics = {
        "by_network": {
            "sp500::network_0": {
                "network_key": "sp500::network_0",
                "dataset_name": "sp500",
                "network_id": "network_0",
                "probe_timesteps": [10, 0],
                "probe_graph_count": [3.0, 3.0],
                "selection_given_available_by_level": {"0": [imp_row, imp_row]},
                "selected_count_by_level": {"0": [[1, 1, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0]]},
                "available_count_by_level": {"0": [[3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3]]},
            }
        }
    }
    stats = compare_leverage_vs_selection(
        diagnostics, W, normalized=True, out_dir=str(tmp_path)
    )
    assert stats["network_key"] == "sp500::network_0"
    assert stats["num_nodes"] == 6
    assert "0" in stats["levels"]
    lvl = stats["levels"]["0"]
    assert "spearman" in lvl and "K" in lvl and "topk_jaccard" in lvl
    # Default measure is recorded.
    assert stats["measure"] == "conditional"
    assert stats["min_active_frac"] == 0.0
    # Artifacts written.
    assert (tmp_path / "leverage_vs_selection_sp500__network_0_norm.pdf").exists()
    assert (tmp_path / "leverage_vs_selection_sp500__network_0_norm.json").exists()


def test_compare_leverage_vs_selection_freq_measure(tmp_path):
    """measure='freq' records the measure in stats and writes _freq-suffixed
    artifacts; the default 'conditional' keeps the legacy (unsuffixed) names and
    does not collide with the freq run."""
    W = _two_cluster_W()
    d = W.sum(axis=1)
    imp_row = (d / d.max()).tolist()
    diagnostics = {
        "by_network": {
            "sp500::network_0": {
                "network_key": "sp500::network_0",
                "dataset_name": "sp500",
                "network_id": "network_0",
                "probe_timesteps": [10, 0],
                "probe_graph_count": [3.0, 3.0],
                "selection_given_available_by_level": {"0": [imp_row, imp_row]},
                "selection_freq_by_level": {"0": [imp_row, imp_row]},
                "selected_count_by_level": {"0": [[1, 1, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0]]},
                "available_count_by_level": {"0": [[3, 3, 3, 3, 3, 3], [3, 3, 3, 3, 3, 3]]},
            }
        }
    }
    stats = compare_leverage_vs_selection(
        diagnostics, W, normalized=True, measure="freq", min_active_frac=0.0, out_dir=str(tmp_path)
    )
    assert stats["measure"] == "freq"
    assert stats["min_active_frac"] == 0.0
    assert (tmp_path / "leverage_vs_selection_sp500__network_0_norm_freq.pdf").exists()
    assert (tmp_path / "leverage_vs_selection_sp500__network_0_norm_freq.json").exists()

    # Default conditional keeps the legacy filename (no suffix).
    stats_c = compare_leverage_vs_selection(diagnostics, W, normalized=True, out_dir=str(tmp_path))
    assert stats_c["measure"] == "conditional"
    assert (tmp_path / "leverage_vs_selection_sp500__network_0_norm.pdf").exists()
    assert (tmp_path / "leverage_vs_selection_sp500__network_0_norm.json").exists()
