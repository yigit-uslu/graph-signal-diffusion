import json

import numpy as np

from graph_signal_diffusion.utils.plotting import (
    _select_sampling_heatmap_node_indices,
    _select_sampling_heatmap_nodes_from_reward,
    aggregate_selector_sampling_diagnostics,
    plot_selector_diagnostics_jsonl,
    plot_selector_sampling_phase_heatmaps,
    plot_selector_network_hard_mask_statistics,
    plot_selector_network_node_statistics,
    plot_selector_network_selection_frequency_evolution,
)


def test_plot_selector_diagnostics_jsonl_creates_files(tmp_path):
    outdir = tmp_path / "selector_diagnostics"
    outdir.mkdir(parents=True, exist_ok=True)
    jsonl = tmp_path / "epoch_summaries.jsonl"

    rows = [
        {
            "epoch": 1,
            "train_selector_temperature": 1.0,
            "train_selector_entropy_mean_norm": 0.62,
            "train_selector_selector_aux_total": 0.015,
        },
        {
            "epoch": 2,
            "train_selector_temperature": 0.9,
            "train_selector_entropy_mean_norm": 0.68,
            "train_selector_selector_aux_total": 0.010,
        },
        {
            "epoch": 3,
            "train_selector_temperature": 0.8,
            "train_selector_entropy_mean_norm": 0.71,
            "train_selector_selector_aux_total": 0.008,
        },
    ]
    with open(jsonl, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    plot_selector_diagnostics_jsonl(jsonl, outdir, save_pdf=True)

    assert (outdir / "selector_temperature.pdf").exists()
    assert (outdir / "selector_aux_loss.pdf").exists()
    assert (outdir / "selector_entropy.pdf").exists()


def test_plot_selector_network_node_statistics_creates_network_tagged_file(tmp_path):
    outdir = tmp_path / "selector_diagnostics"
    outdir.mkdir(parents=True, exist_ok=True)

    stats = {
        "setA::network_7": {
            0: {
                "score_mean": np.array([0.2, 0.4, 0.1, 0.3], dtype=float),
                "score_std": np.array([0.05, 0.06, 0.04, 0.05], dtype=float),
                "soft_mask_mean": np.array([0.7, 0.4, 0.2, 0.6], dtype=float),
                "soft_mask_std": np.array([0.1, 0.12, 0.08, 0.09], dtype=float),
            },
            1: {
                "score_mean": np.array([0.1, 0.3, 0.2, 0.25], dtype=float),
                "score_std": np.array([0.03, 0.04, 0.05, 0.03], dtype=float),
                "soft_mask_mean": np.array([0.5, 0.55, 0.35, 0.4], dtype=float),
                "soft_mask_std": np.array([0.08, 0.07, 0.06, 0.05], dtype=float),
            },
        }
    }

    plot_selector_network_node_statistics(
        selector_network_node_stats=stats,
        epoch=4,
        out_dir=outdir,
        save_pdf=True,
    )

    assert (outdir / "selector_node_stats_dsetA_nnetwork_7_epoch_0005.pdf").exists()


def test_plot_selector_network_hard_mask_statistics_creates_network_tagged_file(tmp_path):
    outdir = tmp_path / "selector_diagnostics"
    outdir.mkdir(parents=True, exist_ok=True)

    stats = {
        "setA::network_7": {
            0: {
                "hard_mask_mean": np.array([0.5, 0.25, 0.75, 0.0], dtype=float),
                "hard_mask_std": np.array([0.5, 0.43, 0.43, 0.0], dtype=float),
            },
            1: {
                "hard_mask_mean": np.array([0.4, 0.6, 0.5, 0.2], dtype=float),
                "hard_mask_std": np.array([0.49, 0.49, 0.5, 0.4], dtype=float),
            },
        }
    }

    plot_selector_network_hard_mask_statistics(
        selector_network_node_stats=stats,
        epoch=4,
        out_dir=outdir,
        save_pdf=True,
    )

    assert (outdir / "selector_hard_mask_dsetA_nnetwork_7_epoch_0005.pdf").exists()


def test_plot_selector_network_selection_frequency_evolution_creates_file(tmp_path):
    outdir = tmp_path / "selector_diagnostics"
    outdir.mkdir(parents=True, exist_ok=True)
    diagnostic_jsonl = outdir / "selector_diagnostic_summaries.jsonl"

    rows = [
        {
            "scope": "global",
            "epoch": 0,
            "temperature": 1.0,
            "selector_aux_total": 0.01,
            "entropy_mean_norm": 0.6,
        },
        {
            "scope": "network",
            "epoch": 0,
            "network_key": "setA::network_7",
            "dataset_name": "setA",
            "network_id": "network_7",
            "selection_freq_mean_by_level": {"0": 0.50, "1": 0.35},
            "selection_freq_std_by_level": {"0": 0.10, "1": 0.08},
        },
        {
            "scope": "network",
            "epoch": 1,
            "network_key": "setA::network_7",
            "dataset_name": "setA",
            "network_id": "network_7",
            "selection_freq_mean_by_level": {"0": 0.48, "1": 0.37},
            "selection_freq_std_by_level": {"0": 0.09, "1": 0.07},
        },
    ]
    with open(diagnostic_jsonl, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    plot_selector_network_selection_frequency_evolution(
        diagnostic_jsonl_path=diagnostic_jsonl,
        out_dir=outdir,
        save_pdf=True,
    )

    assert (outdir / "selector_freq_evolution_dsetA_nnetwork_7.pdf").exists()


def test_plot_selector_network_selection_frequency_evolution_supports_per_node_ema(tmp_path):
    outdir = tmp_path / "selector_diagnostics"
    outdir.mkdir(parents=True, exist_ok=True)
    diagnostic_jsonl = outdir / "selector_diagnostic_summaries.jsonl"

    rows = [
        {"scope": "global", "epoch": 0, "temperature": 1.0, "selector_aux_total": 0.01, "entropy_mean_norm": 0.6},
        {
            "scope": "network",
            "epoch": 0,
            "network_key": "setA::network_7",
            "dataset_name": "setA",
            "network_id": "network_7",
            "selection_freq_tracked_node_indices_by_level": {
                "0": [0, 3, 5, 7],
                "1": [1, 2, 8, 10],
            },
            "selection_freq_tracked_node_ma_by_level": {
                "0": [0.01, 0.28, 0.44, 0.63],
                "1": [0.07, 0.12, 0.40, 0.51],
            },
        },
        {
            "scope": "network",
            "epoch": 1,
            "network_key": "setA::network_7",
            "dataset_name": "setA",
            "network_id": "network_7",
            "selection_freq_tracked_node_indices_by_level": {
                "0": [0, 3, 5, 7],
                "1": [1, 2, 8, 10],
            },
            "selection_freq_tracked_node_ma_by_level": {
                "0": [0.03, 0.30, 0.47, 0.66],
                "1": [0.08, 0.15, 0.43, 0.55],
            },
        },
    ]
    with open(diagnostic_jsonl, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    plot_selector_network_selection_frequency_evolution(
        diagnostic_jsonl_path=diagnostic_jsonl,
        out_dir=outdir,
        save_pdf=True,
    )

    assert (outdir / "selector_freq_evolution_dsetA_nnetwork_7.pdf").exists()


def test_plot_selector_diagnostics_jsonl_supports_selector_diagnostic_summaries(tmp_path):
    outdir = tmp_path / "selector_diagnostics"
    outdir.mkdir(parents=True, exist_ok=True)
    diagnostic_jsonl = outdir / "selector_diagnostic_summaries.jsonl"

    rows = [
        {"scope": "global", "epoch": 0, "temperature": 1.0, "selector_aux_total": 0.02, "entropy_mean_norm": 0.58},
        {"scope": "global", "epoch": 1, "temperature": 0.9, "selector_aux_total": 0.015, "entropy_mean_norm": 0.63},
        {"scope": "network", "epoch": 1, "network_key": "setA::network_7", "selection_freq_mean_by_level": {"0": 0.5}},
    ]
    with open(diagnostic_jsonl, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    plot_selector_diagnostics_jsonl(diagnostic_jsonl, outdir, save_pdf=True)

    assert (outdir / "selector_temperature.pdf").exists()
    assert (outdir / "selector_aux_loss.pdf").exists()
    assert (outdir / "selector_entropy.pdf").exists()


def test_plot_selector_sampling_phase_heatmaps_creates_file(tmp_path):
    outdir = tmp_path / "selector_diagnostics"
    outdir.mkdir(parents=True, exist_ok=True)

    selector_sampling_diagnostics = {
        "probe_timesteps": [499, 400, 300, 200, 100, 0],
        "selection_given_available_by_level": {
            "0": [
                [0.90, 0.85, 0.80, 0.75],
                [0.88, 0.84, 0.79, 0.74],
                [0.86, 0.82, 0.78, 0.73],
                [0.84, 0.80, 0.76, 0.71],
                [0.82, 0.78, 0.74, 0.69],
                [0.80, 0.76, 0.72, 0.67],
            ],
            "1": [
                [0.70, 0.62, 0.55, 0.48],
                [0.68, 0.60, 0.53, 0.46],
                [0.66, 0.58, 0.51, 0.44],
                [0.64, 0.56, 0.49, 0.42],
                [0.62, 0.54, 0.47, 0.40],
                [0.60, 0.52, 0.45, 0.38],
            ],
        },
    }

    plot_selector_sampling_phase_heatmaps(
        selector_sampling_diagnostics=selector_sampling_diagnostics,
        out_dir=outdir,
        epoch=4,
        split="val",
        dataset_name="sp500",
        network_id="network_0",
        save_pdf=True,
    )

    assert (outdir / "selector_sampling_heatmap_val_dsp500_nnetwork_0_epoch_0005.pdf").exists()


def test_plot_selector_sampling_phase_heatmaps_batch_idx_in_filename(tmp_path):
    """batch_idx must be woven into the filename so per-batch heatmaps for the
    same (split, dataset, network, epoch) do not overwrite one another."""
    outdir = tmp_path / "selector_diagnostics"
    outdir.mkdir(parents=True, exist_ok=True)

    selector_sampling_diagnostics = {
        "probe_timesteps": [499, 300, 100, 0],
        "selection_given_available_by_level": {
            "0": [
                [0.90, 0.85, 0.80, 0.75],
                [0.88, 0.84, 0.79, 0.74],
                [0.86, 0.82, 0.78, 0.73],
                [0.84, 0.80, 0.76, 0.71],
            ],
        },
    }

    # Two distinct batch indices -> two distinct files (no overwrite).
    for b in (0, 12):
        plot_selector_sampling_phase_heatmaps(
            selector_sampling_diagnostics=selector_sampling_diagnostics,
            out_dir=outdir,
            epoch=0,
            split="test",
            dataset_name="sp500",
            network_id="network_0",
            save_pdf=True,
            batch_idx=b,
        )

    assert (outdir / "selector_sampling_heatmap_test_dsp500_nnetwork_0_b000_epoch_0001.pdf").exists()
    assert (outdir / "selector_sampling_heatmap_test_dsp500_nnetwork_0_b012_epoch_0001.pdf").exists()
    # The un-indexed (batch_idx=None) filename must NOT be produced here.
    assert not (outdir / "selector_sampling_heatmap_test_dsp500_nnetwork_0_epoch_0001.pdf").exists()


def test_plot_selector_sampling_phase_heatmaps_no_batch_idx_keeps_legacy_filename(tmp_path):
    """Default (batch_idx=None) keeps the legacy filename for back-compat."""
    outdir = tmp_path / "selector_diagnostics"
    outdir.mkdir(parents=True, exist_ok=True)

    selector_sampling_diagnostics = {
        "probe_timesteps": [499, 300, 100, 0],
        "selection_given_available_by_level": {
            "0": [
                [0.90, 0.85, 0.80, 0.75],
                [0.88, 0.84, 0.79, 0.74],
                [0.86, 0.82, 0.78, 0.73],
                [0.84, 0.80, 0.76, 0.71],
            ],
        },
    }

    plot_selector_sampling_phase_heatmaps(
        selector_sampling_diagnostics=selector_sampling_diagnostics,
        out_dir=outdir,
        epoch=0,
        split="test",
        dataset_name="sp500",
        network_id="network_0",
        save_pdf=True,
    )

    assert (outdir / "selector_sampling_heatmap_test_dsp500_nnetwork_0_epoch_0001.pdf").exists()


def test_aggregate_selector_sampling_diagnostics_pools_counts_and_aligns_by_timestep():
    """Pooling sums selected/available/graph counts across batches, recomputes
    frequencies, aligns rows by probe-timestep VALUE (not position), and yields
    NaN for never-available nodes."""
    # Batch A: probe order [10, 0]; node 1 is never available (-> NaN conditional).
    batch_a = {
        "probe_timesteps": [10, 0],
        "probe_graph_count": [4.0, 4.0],
        "selected_count_by_level": {"0": [[2.0, 0.0], [1.0, 0.0]]},
        "available_count_by_level": {"0": [[4.0, 0.0], [2.0, 0.0]]},
        "by_network": {
            "sp500::network_0": {
                "network_key": "sp500::network_0",
                "dataset_name": "sp500",
                "network_id": "network_0",
                "probe_timesteps": [10, 0],
                "probe_graph_count": [4.0, 4.0],
                "selected_count_by_level": {"0": [[2.0, 0.0], [1.0, 0.0]]},
                "available_count_by_level": {"0": [[4.0, 0.0], [2.0, 0.0]]},
            }
        },
    }
    # Batch B: REVERSED probe order [0, 10] -> must be aligned by value.
    batch_b = {
        "probe_timesteps": [0, 10],
        "probe_graph_count": [4.0, 4.0],
        "selected_count_by_level": {"0": [[3.0, 0.0], [1.0, 0.0]]},
        "available_count_by_level": {"0": [[4.0, 0.0], [4.0, 0.0]]},
        "by_network": {
            "sp500::network_0": {
                "network_key": "sp500::network_0",
                "dataset_name": "sp500",
                "network_id": "network_0",
                "probe_timesteps": [0, 10],
                "probe_graph_count": [4.0, 4.0],
                "selected_count_by_level": {"0": [[3.0, 0.0], [1.0, 0.0]]},
                "available_count_by_level": {"0": [[4.0, 0.0], [4.0, 0.0]]},
            }
        },
    }

    agg = aggregate_selector_sampling_diagnostics([batch_a, batch_b])
    assert agg is not None
    # Reference grid is the first batch's order [10, 0].
    assert agg["probe_timesteps"] == [10, 0]
    assert agg["probe_graph_count"] == [8.0, 8.0]

    selected = np.asarray(agg["selected_count_by_level"]["0"])
    available = np.asarray(agg["available_count_by_level"]["0"])
    # t=10 row: A[2,0] + B(t=10 is B row 1)[1,0] = [3,0]; t=0: A[1,0] + B[3,0] = [4,0]
    np.testing.assert_allclose(selected, [[3.0, 0.0], [4.0, 0.0]])
    np.testing.assert_allclose(available, [[8.0, 0.0], [6.0, 0.0]])

    freq = np.asarray(agg["selection_freq_by_level"]["0"])
    np.testing.assert_allclose(freq, [[3 / 8, 0.0], [4 / 8, 0.0]])

    cond = np.asarray(agg["selection_given_available_by_level"]["0"])
    np.testing.assert_allclose(cond[:, 0], [3 / 8, 4 / 6])
    assert np.isnan(cond[:, 1]).all()  # node 1 never available -> NaN

    # by_network pooled identically.
    net = agg["by_network"]["sp500::network_0"]
    assert net["network_id"] == "network_0"
    assert net["dataset_name"] == "sp500"
    np.testing.assert_allclose(
        np.asarray(net["selected_count_by_level"]["0"]), [[3.0, 0.0], [4.0, 0.0]]
    )


def test_aggregate_selector_sampling_diagnostics_empty_returns_none():
    assert aggregate_selector_sampling_diagnostics([]) is None
    assert aggregate_selector_sampling_diagnostics([{}]) is None


def test_plot_selector_sampling_phase_heatmaps_aggregate_filename(tmp_path):
    """aggregate=True uses the _aggregate token and takes precedence over batch_idx."""
    outdir = tmp_path / "selector_diagnostics"
    outdir.mkdir(parents=True, exist_ok=True)

    selector_sampling_diagnostics = {
        "probe_timesteps": [499, 300, 100, 0],
        "selection_given_available_by_level": {
            "0": [
                [0.90, 0.85, 0.80, 0.75],
                [0.88, 0.84, 0.79, 0.74],
                [0.86, 0.82, 0.78, 0.73],
                [0.84, 0.80, 0.76, 0.71],
            ],
        },
    }

    plot_selector_sampling_phase_heatmaps(
        selector_sampling_diagnostics=selector_sampling_diagnostics,
        out_dir=outdir,
        epoch=0,
        split="test",
        dataset_name="sp500",
        network_id="network_0",
        save_pdf=True,
        batch_idx=5,  # should be ignored in favor of the aggregate token
        aggregate=True,
    )

    assert (outdir / "selector_sampling_heatmap_test_dsp500_nnetwork_0_aggregate_epoch_0001.pdf").exists()
    assert not (outdir / "selector_sampling_heatmap_test_dsp500_nnetwork_0_b005_epoch_0001.pdf").exists()


def test_select_sampling_heatmap_node_indices_prefers_low_and_high_activity():
    activity = np.arange(30, dtype=float)  # Increasing activity by node index.
    picked = _select_sampling_heatmap_node_indices(activity, max_nodes=20)

    assert picked.shape == (20,)
    assert set(picked.tolist()) == (set(range(10)) | set(range(20, 30)))


def test_select_sampling_heatmap_node_indices_returns_all_when_small():
    activity = np.array([0.3, 0.1, 0.9, 0.5], dtype=float)
    picked = _select_sampling_heatmap_node_indices(activity, max_nodes=20)
    assert np.array_equal(picked, np.arange(4, dtype=int))


def test_select_sampling_heatmap_nodes_from_reward_uses_shared_weighted_reward():
    # 30 nodes, two selector levels.
    n_nodes = 30
    level_keys = [(0, "0"), (1, "1")]

    # Level-0 favors nodes 0..9, level-1 strongly favors 20..29.
    probs_l0 = np.full((3, n_nodes), 0.1, dtype=float)
    probs_l0[:, 0:10] = 0.9
    probs_l1 = np.full((3, n_nodes), 0.1, dtype=float)
    probs_l1[:, 20:30] = 0.9

    matrix_by_level = {
        "0": probs_l0.tolist(),
        "1": probs_l1.tolist(),
    }
    diagnostics = {
        "cumulative_downsampling_factor_by_level": {"0": 1.0, "1": 4.0},
    }

    picked = _select_sampling_heatmap_nodes_from_reward(
        selector_sampling_diagnostics=diagnostics,
        level_keys=level_keys,
        matrix_by_level=matrix_by_level,
        max_nodes=20,
    )

    # Bottom-10 reward: nodes 10..19; Top-10 reward: nodes 20..29.
    assert picked.shape == (20,)
    assert set(picked.tolist()) == (set(range(10, 20)) | set(range(20, 30)))
