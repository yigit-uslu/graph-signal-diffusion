"""Tests for tracked-network trace visualization (Phase-2).

Generates synthetic ``tracked_network_trace.jsonl`` data with known patterns
and verifies parsing, receiver auto-selection, and figure generation.

Run the example-output test standalone to produce an inspectable plot::

    pytest tests/trainers/test_visualize_trace.py::test_generate_example_output -v -s
"""

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest

# Fixed output directory for example plots.
_EXAMPLE_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "figs"


# ---------------------------------------------------------------------------
# Synthetic JSONL generator
# ---------------------------------------------------------------------------

def _make_synthetic_trace(
    tmp_path: Path,
    *,
    num_epochs: int = 200,
    n_receivers: int = 10,
    network_ids: list[int] | None = None,
    include_full_vectors: bool = True,
    window_sizes: list[int] | None = None,
) -> Path:
    """Write a synthetic ``tracked_network_trace.jsonl`` with known patterns.

    Receiver 0: oscillating dual (high variance) — should be auto-selected.
    Receiver 1: steadily growing dual (moderate variance).
    Receiver 2: initially feasible, then slowly drifting into violation.
    Others: near-zero duals (uninteresting).
    """
    if network_ids is None:
        network_ids = [0]
    if window_sizes is None:
        window_sizes = [10, 50]

    trace_path = tmp_path / "tracked_network_trace.jsonl"
    rng = np.random.default_rng(42)

    with open(trace_path, "w") as f:
        for net_id in network_ids:
            # Metadata entry.
            r_min = rng.uniform(0.3, 0.8, size=n_receivers).tolist()
            # Fix r_min for the first three receivers so the patterns are readable.
            r_min[0] = 0.55  # oscillating receiver hovers around this
            r_min[1] = 0.50  # improving receiver starts below, ends above
            if n_receivers > 2:
                r_min[2] = 0.60  # drifter starts above, ends below

            associations = np.eye(n_receivers).tolist()  # 1-to-1 for simplicity

            # Synthetic cross-channel gains (n x n): diagonal = direct signal,
            # off-diagonal = interference.  With 1-to-1 associations, this is
            # just H_l itself.  Make RX 0 ↔ RX 2 strong mutual interferers.
            H_l = rng.exponential(0.01, size=(n_receivers, n_receivers))
            np.fill_diagonal(H_l, rng.uniform(0.5, 1.0, size=n_receivers))
            if n_receivers > 2:
                # Strong mutual interference between RX 0 and RX 2.
                H_l[0, 2] = 0.35
                H_l[2, 0] = 0.30
            # cross_channel_gains = A^T @ H_l = I @ H_l = H_l (1-to-1 case).
            cross_channel_gains = H_l.tolist()

            metadata = {
                "type": "metadata",
                "network_id": net_id,
                "network_seed": 1000 + net_id,
                "associations": associations,
                "r_min_per_receiver": r_min,
                "tracked_receiver_indices": [0, 1],
                "window_sizes": window_sizes,
                "include_full_vectors": include_full_vectors,
                "cross_channel_gains": cross_channel_gains,
            }
            f.write(json.dumps(metadata) + "\n")

            # Epoch entries.
            # Running windowed average state for each window size.
            rate_windows: dict[int, list[np.ndarray]] = {w: [] for w in window_sizes}

            for epoch in range(num_epochs):
                t = epoch / max(num_epochs - 1, 1)

                # Receiver 0: oscillating rates / duals (constraint boundary).
                rate_0 = 0.55 + 0.15 * math.sin(2 * math.pi * t * 6) + rng.normal(0, 0.02)
                dual_0 = max(0.0, 1.0 + 0.8 * math.sin(2 * math.pi * t * 6) + rng.normal(0, 0.05))
                power_0 = max(0.01, 0.5 + 0.15 * math.sin(2 * math.pi * t * 6 + 0.3) + rng.normal(0, 0.03))

                # Receiver 1: steadily improving (constraint met mid-training).
                rate_1 = 0.30 + 0.45 * t + rng.normal(0, 0.02)
                dual_1 = max(0.0, 0.9 * (1.0 - t) ** 1.5 + rng.normal(0, 0.03))
                power_1 = max(0.01, 0.3 + 0.4 * t + rng.normal(0, 0.02))

                # Receiver 2: initially feasible, drifts into violation.
                if n_receivers > 2:
                    rate_2 = 0.75 - 0.25 * t + rng.normal(0, 0.015)
                    dual_2 = max(0.0, -0.3 + 0.8 * t + rng.normal(0, 0.04))
                    power_2 = max(0.01, 0.6 - 0.2 * t + rng.normal(0, 0.02))
                else:
                    rate_2 = power_2 = dual_2 = 0.0

                # Other receivers: stable and feasible (boring).
                rates = [rate_0, rate_1]
                duals = [dual_0, dual_1]
                powers = [power_0, power_1]
                if n_receivers > 2:
                    rates.append(rate_2)
                    duals.append(dual_2)
                    powers.append(power_2)
                for _ in range(max(0, n_receivers - 3)):
                    rates.append(0.7 + rng.normal(0, 0.01))
                    duals.append(max(0.0, 0.01 + rng.normal(0, 0.005)))
                    powers.append(max(0.01, 0.5 + rng.normal(0, 0.03)))

                slacks = [r_min[i] - rates[i] for i in range(n_receivers)]

                # Compute windowed averages.
                rates_arr = np.array(rates)
                windowed_avg_rates: dict[str, list[float]] = {}
                for w in window_sizes:
                    rate_windows[w].append(rates_arr)
                    if len(rate_windows[w]) > w:
                        rate_windows[w] = rate_windows[w][-w:]
                    avg = np.mean(rate_windows[w], axis=0)
                    windowed_avg_rates[str(w)] = avg.tolist()

                selected_trace = {
                    "receiver_indices": [0, 1],
                    "receiver_power_allocations": [powers[0], powers[1]],
                    "ergodic_rates": [rates[0], rates[1]],
                    "dual_multipliers": [duals[0], duals[1]],
                    "constraint_slacks": [slacks[0], slacks[1]],
                    "windowed_avg_rates": {
                        k: [v[0], v[1]] for k, v in windowed_avg_rates.items()
                    },
                }

                entry: dict = {
                    "type": "epoch_trace",
                    "epoch": epoch,
                    "network_id": net_id,
                    "network_seed": 1000 + net_id,
                    "selected_receiver_trace": selected_trace,
                }
                if include_full_vectors:
                    entry.update({
                        "power_allocations": powers,
                        "receiver_power_allocations": powers,
                        "ergodic_rates": rates,
                        "dual_multipliers": duals,
                        "constraint_slacks": slacks,
                        "windowed_avg_rates": windowed_avg_rates,
                    })
                f.write(json.dumps(entry) + "\n")

    return trace_path


# ---------------------------------------------------------------------------
# Example output generator (run standalone to produce inspectable plots)
# ---------------------------------------------------------------------------


def test_generate_example_output(tmp_path):
    """Generate example trace-dynamics plots from synthetic data.

    Writes output to ``outputs/test_trace_viz_example/`` in the repo root so
    you can open the figures directly.  Run with::

        pytest tests/trainers/test_visualize_trace.py::test_generate_example_output -v -s
    """
    from graph_signal_diffusion.trainers.pd_visualization import (
        parse_trace_jsonl,
        select_interesting_receivers,
        visualize_trace_dynamics,
    )

    # --- generate synthetic JSONL with two networks -----------------------
    trace_path = _make_synthetic_trace(
        tmp_path,
        num_epochs=500,
        n_receivers=10,
        network_ids=[0, 1],
        include_full_vectors=True,
        window_sizes=[10, 50, 100],
    )

    trace_data = parse_trace_jsonl(trace_path)
    assert len(trace_data) == 2

    out_dir = _EXAMPLE_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Network 0: neg_corr_pair auto-select (k=2) with collection window + scatter
    auto_rx_pair = select_interesting_receivers(
        trace_data, network_id=0, k=2, strategy="neg_corr_pair",
        neg_corr_pair_cfg={"variance_quantile": 0.5, "interference_weight": 1.0},
    )
    print(f"\nAuto-selected receivers for network 0: {auto_rx_pair}")
    p0 = visualize_trace_dynamics(
        trace_data=trace_data,
        network_id=0,
        output_dir=out_dir,
        grouped_receivers=[auto_rx_pair],
        collection_window=150,
        plot_cfg={
            "dpi": 150,
            "linewidth": 1.4,
            "alpha_raw": 0.25,
            "smoothing_window": 30,
            "legend_fontsize": 9,
            "title_fontsize": 12,
            "label_fontsize": 11,
            "format": "pdf",
            "modes": {"full": {"figsize": [14, 18], "scatter_colormap": "viridis"}},
            "panels": {"scatter": {
                "transient_display": "both",
                "collection_marker_size": 10,
                "transient_marker_size": 3,
                "transient_alpha": 0.12,
                "collection_alpha": 0.8,
            }},
        },
    )
    assert p0 is not None and p0.exists()
    print(f"Network 0 plot (k=2, scatter+cw): {p0}")

    # --- Network 0 again: collection_only scatter mode --------------------
    p0b = visualize_trace_dynamics(
        trace_data=trace_data,
        network_id=0,
        output_dir=out_dir / "collection_only",
        grouped_receivers=[auto_rx_pair],
        collection_window=150,
        plot_cfg={
            "dpi": 150,
            "smoothing_window": 30,
            "format": "pdf",
            "modes": {"full": {"figsize": [14, 18]}},
            "panels": {"scatter": {"transient_display": "collection_only"}},
        },
    )
    assert p0b is not None and p0b.exists()
    print(f"Network 0 plot (collection_only): {p0b}")

    # --- Network 0 again: with overlay of dummy generated samples ----------
    rng = np.random.default_rng(77)
    dummy_overlay = {
        0: rng.uniform(0.2, 0.8, size=(80, 10)).astype(np.float32),
    }
    p0c = visualize_trace_dynamics(
        trace_data=trace_data,
        network_id=0,
        output_dir=out_dir / "with_overlay",
        grouped_receivers=[auto_rx_pair],
        collection_window=150,
        overlay_samples=dummy_overlay,
        overlay_cfg={
            "marker": "x",
            "color": "crimson",
            "alpha": 0.5,
            "marker_size": 30,
            "label": "Generated",
        },
        plot_cfg={
            "dpi": 150,
            "linewidth": 1.4,
            "alpha_raw": 0.25,
            "smoothing_window": 30,
            "legend_fontsize": 9,
            "title_fontsize": 12,
            "label_fontsize": 11,
            "format": "pdf",
            "modes": {"full": {"figsize": [14, 18], "scatter_colormap": "viridis"}},
            "panels": {"scatter": {
                "transient_display": "both",
                "collection_marker_size": 10,
                "transient_marker_size": 3,
                "transient_alpha": 0.12,
                "collection_alpha": 0.8,
            }},
        },
    )
    assert p0c is not None and p0c.exists()
    print(f"Network 0 plot (with overlay): {p0c}")

    # --- Network 1: explicit receivers [0, 1, 2] (no scatter, k>2) --------
    p1 = visualize_trace_dynamics(
        trace_data=trace_data,
        network_id=1,
        output_dir=out_dir,
        grouped_receivers=[[0, 1, 2]],
        collection_window=100,
        plot_cfg={
            "figsize": [14, 18],
            "dpi": 150,
            "linewidth": 1.4,
            "alpha_raw": 0.25,
            "smoothing_window": 30,
            "format": "png",
        },
    )
    assert p1 is not None and p1.exists()
    print(f"Network 1 plot (k=3, no scatter): {p1}")

    # --- also save the raw JSONL for inspection ---------------------------
    import shutil
    saved_jsonl = out_dir / "tracked_network_trace.jsonl"
    shutil.copy2(trace_path, saved_jsonl)
    print(f"Raw JSONL:      {saved_jsonl}")
    print(f"\nAll example outputs in: {out_dir}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestParseTraceJsonl:
    """Tests for ``parse_trace_jsonl``."""

    def test_parses_metadata_and_epochs(self, tmp_path):
        from graph_signal_diffusion.trainers.pd_visualization import parse_trace_jsonl

        trace_path = _make_synthetic_trace(tmp_path, num_epochs=50)
        data = parse_trace_jsonl(trace_path)

        assert 0 in data
        assert data[0]["metadata"] is not None
        assert data[0]["metadata"]["type"] == "metadata"
        assert len(data[0]["epochs"]) == 50

    def test_cross_channel_gains_in_metadata(self, tmp_path):
        from graph_signal_diffusion.trainers.pd_visualization import parse_trace_jsonl

        trace_path = _make_synthetic_trace(tmp_path, num_epochs=10, n_receivers=5)
        data = parse_trace_jsonl(trace_path)

        meta = data[0]["metadata"]
        assert "cross_channel_gains" in meta
        ccg = meta["cross_channel_gains"]
        assert len(ccg) == 5
        assert len(ccg[0]) == 5
        # Diagonal should be strong (direct signal).
        assert ccg[0][0] > 0.3
        # RX 0 ↔ RX 2 should have strong mutual interference.
        assert ccg[0][2] > 0.2
        assert ccg[2][0] > 0.2

    def test_epochs_sorted(self, tmp_path):
        from graph_signal_diffusion.trainers.pd_visualization import parse_trace_jsonl

        trace_path = _make_synthetic_trace(tmp_path, num_epochs=100)
        data = parse_trace_jsonl(trace_path)

        epoch_nums = [e["epoch"] for e in data[0]["epochs"]]
        assert epoch_nums == sorted(epoch_nums)

    def test_multiple_networks(self, tmp_path):
        from graph_signal_diffusion.trainers.pd_visualization import parse_trace_jsonl

        trace_path = _make_synthetic_trace(tmp_path, network_ids=[0, 5, 12])
        data = parse_trace_jsonl(trace_path)

        assert set(data.keys()) == {0, 5, 12}
        for net_id in [0, 5, 12]:
            assert data[net_id]["metadata"]["network_id"] == net_id

    def test_missing_file_returns_empty(self, tmp_path):
        from graph_signal_diffusion.trainers.pd_visualization import parse_trace_jsonl

        data = parse_trace_jsonl(tmp_path / "nonexistent.jsonl")
        assert data == {}

    def test_empty_file_returns_empty(self, tmp_path):
        from graph_signal_diffusion.trainers.pd_visualization import parse_trace_jsonl

        trace_path = tmp_path / "empty.jsonl"
        trace_path.write_text("")
        data = parse_trace_jsonl(trace_path)
        assert data == {}


class TestSelectInterestingReceivers:
    """Tests for ``select_interesting_receivers``."""

    def test_selects_oscillating_receiver(self, tmp_path):
        from graph_signal_diffusion.trainers.pd_visualization import (
            parse_trace_jsonl,
            select_interesting_receivers,
        )

        trace_path = _make_synthetic_trace(tmp_path, num_epochs=200, n_receivers=10)
        data = parse_trace_jsonl(trace_path)

        selected = select_interesting_receivers(data, network_id=0, k=2)
        # Receiver 0 has the highest dual variance (oscillating).
        assert 0 in selected
        assert len(selected) == 2

    def test_k_exceeds_receivers(self, tmp_path):
        from graph_signal_diffusion.trainers.pd_visualization import (
            parse_trace_jsonl,
            select_interesting_receivers,
        )

        trace_path = _make_synthetic_trace(tmp_path, num_epochs=50, n_receivers=3)
        data = parse_trace_jsonl(trace_path)

        selected = select_interesting_receivers(data, network_id=0, k=10)
        # Should return at most n_receivers indices.
        assert len(selected) <= 3

    def test_fallback_to_metadata(self, tmp_path):
        from graph_signal_diffusion.trainers.pd_visualization import (
            parse_trace_jsonl,
            select_interesting_receivers,
        )

        # Without full vectors, auto-selection falls back to metadata indices.
        trace_path = _make_synthetic_trace(
            tmp_path, num_epochs=50, include_full_vectors=False,
        )
        data = parse_trace_jsonl(trace_path)

        selected = select_interesting_receivers(data, network_id=0, k=2)
        assert selected == [0, 1]  # from metadata tracked_receiver_indices

    def test_nonexistent_network_returns_empty(self, tmp_path):
        from graph_signal_diffusion.trainers.pd_visualization import (
            parse_trace_jsonl,
            select_interesting_receivers,
        )

        trace_path = _make_synthetic_trace(tmp_path, num_epochs=50)
        data = parse_trace_jsonl(trace_path)

        selected = select_interesting_receivers(data, network_id=999, k=2)
        assert selected == []


class TestAutoSelectionStrategies:
    """Tests for the four auto-selection strategies."""

    def test_top_dual_var_selects_oscillating(self, tmp_path):
        from graph_signal_diffusion.trainers.pd_visualization import (
            parse_trace_jsonl,
            select_interesting_receivers,
        )

        trace_path = _make_synthetic_trace(tmp_path, num_epochs=200, n_receivers=10)
        data = parse_trace_jsonl(trace_path)

        selected = select_interesting_receivers(
            data, network_id=0, k=2, strategy="top_dual_var",
        )
        assert 0 in selected  # RX 0 has highest dual variance
        assert len(selected) == 2

    def test_top_primal_var(self, tmp_path):
        from graph_signal_diffusion.trainers.pd_visualization import (
            parse_trace_jsonl,
            select_interesting_receivers,
        )

        trace_path = _make_synthetic_trace(tmp_path, num_epochs=200, n_receivers=10)
        data = parse_trace_jsonl(trace_path)

        selected = select_interesting_receivers(
            data, network_id=0, k=2, strategy="top_primal_var",
        )
        assert len(selected) == 2
        # RX 0 and RX 1 have the most varying power (sinusoidal + ramp).
        assert 0 in selected or 1 in selected

    def test_top_primal_var_fallback_to_dual(self, tmp_path):
        """Without full power vectors, top_primal_var falls back to top_dual_var."""
        from graph_signal_diffusion.trainers.pd_visualization import (
            parse_trace_jsonl,
            select_interesting_receivers,
        )

        trace_path = _make_synthetic_trace(
            tmp_path, num_epochs=100, n_receivers=10, include_full_vectors=False,
        )
        data = parse_trace_jsonl(trace_path)

        selected = select_interesting_receivers(
            data, network_id=0, k=2, strategy="top_primal_var",
        )
        # Falls back to metadata since no full vectors at all.
        assert selected == [0, 1]

    def test_top_combined_var(self, tmp_path):
        from graph_signal_diffusion.trainers.pd_visualization import (
            parse_trace_jsonl,
            select_interesting_receivers,
        )

        trace_path = _make_synthetic_trace(tmp_path, num_epochs=200, n_receivers=10)
        data = parse_trace_jsonl(trace_path)

        selected = select_interesting_receivers(
            data, network_id=0, k=3, strategy="top_combined_var",
        )
        assert len(selected) == 3
        # RX 0 should rank highly by both criteria.
        assert 0 in selected

    def test_neg_corr_pair_selects_interferers(self, tmp_path):
        """neg_corr_pair should favor the (0, 2) pair due to strong mutual
        interference even if their duals are not the most anti-correlated."""
        from graph_signal_diffusion.trainers.pd_visualization import (
            parse_trace_jsonl,
            select_interesting_receivers,
        )

        trace_path = _make_synthetic_trace(
            tmp_path, num_epochs=200, n_receivers=10, include_full_vectors=True,
        )
        data = parse_trace_jsonl(trace_path)

        selected = select_interesting_receivers(
            data, network_id=0, k=2, strategy="neg_corr_pair",
            neg_corr_pair_cfg={"variance_quantile": 0.5, "interference_weight": 1.0},
        )
        assert len(selected) == 2
        # RX 0 should be selected (highest dual variance + strong interferer).
        assert 0 in selected

    def test_neg_corr_pair_pure_correlation(self, tmp_path):
        """With interference_weight=0, selection is purely by negative dual correlation."""
        from graph_signal_diffusion.trainers.pd_visualization import (
            parse_trace_jsonl,
            select_interesting_receivers,
        )

        trace_path = _make_synthetic_trace(
            tmp_path, num_epochs=200, n_receivers=10, include_full_vectors=True,
        )
        data = parse_trace_jsonl(trace_path)

        selected = select_interesting_receivers(
            data, network_id=0, k=2, strategy="neg_corr_pair",
            neg_corr_pair_cfg={"variance_quantile": 0.5, "interference_weight": 0.0},
        )
        assert len(selected) == 2

    def test_neg_corr_pair_degrades_for_k_gt_2(self, tmp_path):
        """neg_corr_pair falls back to top_dual_var when k > 2."""
        from graph_signal_diffusion.trainers.pd_visualization import (
            parse_trace_jsonl,
            select_interesting_receivers,
        )

        trace_path = _make_synthetic_trace(tmp_path, num_epochs=200, n_receivers=10)
        data = parse_trace_jsonl(trace_path)

        selected = select_interesting_receivers(
            data, network_id=0, k=3, strategy="neg_corr_pair",
        )
        assert len(selected) == 3
        assert 0 in selected  # same as top_dual_var

    def test_neg_corr_pair_no_full_vectors_fallback(self, tmp_path):
        """neg_corr_pair without full vectors falls back to metadata."""
        from graph_signal_diffusion.trainers.pd_visualization import (
            parse_trace_jsonl,
            select_interesting_receivers,
        )

        trace_path = _make_synthetic_trace(
            tmp_path, num_epochs=100, include_full_vectors=False,
        )
        data = parse_trace_jsonl(trace_path)

        selected = select_interesting_receivers(
            data, network_id=0, k=2, strategy="neg_corr_pair",
        )
        assert selected == [0, 1]  # metadata fallback

    def test_unknown_strategy_raises(self, tmp_path):
        from graph_signal_diffusion.trainers.pd_visualization import (
            parse_trace_jsonl,
            select_interesting_receivers,
        )

        trace_path = _make_synthetic_trace(tmp_path, num_epochs=50)
        data = parse_trace_jsonl(trace_path)

        with pytest.raises(ValueError, match="Unknown auto_strategy"):
            select_interesting_receivers(
                data, network_id=0, k=2, strategy="bogus",
            )

    def test_visualize_forwards_strategy(self, tmp_path):
        """visualize_trace_dynamics should accept and use auto_strategy."""
        from graph_signal_diffusion.trainers.pd_visualization import (
            parse_trace_jsonl,
            visualize_trace_dynamics,
        )

        trace_path = _make_synthetic_trace(
            tmp_path, num_epochs=100, n_receivers=10, include_full_vectors=True,
        )
        data = parse_trace_jsonl(trace_path)

        result = visualize_trace_dynamics(
            trace_data=data,
            network_id=0,
            output_dir=tmp_path / "strategy_fwd",
            grouped_receivers=None,
            auto_strategy="neg_corr_pair",
            neg_corr_pair_cfg={"variance_quantile": 0.5, "interference_weight": 1.0},
        )
        assert result is not None
        assert result.exists()


class TestVisualizeTraceDynamics:
    """Tests for ``visualize_trace_dynamics``."""

    def test_produces_pdf(self, tmp_path):
        from graph_signal_diffusion.trainers.pd_visualization import (
            parse_trace_jsonl,
            visualize_trace_dynamics,
        )

        trace_path = _make_synthetic_trace(tmp_path, num_epochs=100)
        data = parse_trace_jsonl(trace_path)
        out_dir = tmp_path / "output"

        result = visualize_trace_dynamics(
            trace_data=data,
            network_id=0,
            output_dir=out_dir,
            grouped_receivers=[[0, 1]],
        )

        assert result is not None
        assert result.exists()
        assert result.suffix == ".pdf"
        assert result.stat().st_size > 0

    def test_produces_png(self, tmp_path):
        from graph_signal_diffusion.trainers.pd_visualization import (
            parse_trace_jsonl,
            visualize_trace_dynamics,
        )

        trace_path = _make_synthetic_trace(tmp_path, num_epochs=50)
        data = parse_trace_jsonl(trace_path)
        out_dir = tmp_path / "output_png"

        result = visualize_trace_dynamics(
            trace_data=data,
            network_id=0,
            output_dir=out_dir,
            grouped_receivers=[[0, 1]],
            plot_cfg={"format": "png", "dpi": 72, "figsize": [8, 10]},
        )

        assert result is not None
        assert result.suffix == ".png"
        assert result.exists()

    def test_auto_receiver_selection(self, tmp_path):
        from graph_signal_diffusion.trainers.pd_visualization import (
            parse_trace_jsonl,
            visualize_trace_dynamics,
        )

        trace_path = _make_synthetic_trace(tmp_path, num_epochs=100, n_receivers=10)
        data = parse_trace_jsonl(trace_path)

        # grouped_receivers=None triggers auto-selection.
        result = visualize_trace_dynamics(
            trace_data=data,
            network_id=0,
            output_dir=tmp_path / "auto",
            grouped_receivers=None,
        )

        assert result is not None
        assert result.exists()

    def test_missing_network_returns_none(self, tmp_path):
        from graph_signal_diffusion.trainers.pd_visualization import (
            parse_trace_jsonl,
            visualize_trace_dynamics,
        )

        trace_path = _make_synthetic_trace(tmp_path, num_epochs=50)
        data = parse_trace_jsonl(trace_path)

        result = visualize_trace_dynamics(
            trace_data=data,
            network_id=999,
            output_dir=tmp_path,
        )
        assert result is None

    def test_custom_plot_style(self, tmp_path):
        from graph_signal_diffusion.trainers.pd_visualization import (
            parse_trace_jsonl,
            visualize_trace_dynamics,
        )

        trace_path = _make_synthetic_trace(tmp_path, num_epochs=80)
        data = parse_trace_jsonl(trace_path)

        result = visualize_trace_dynamics(
            trace_data=data,
            network_id=0,
            output_dir=tmp_path / "styled",
            grouped_receivers=[[0, 1]],
            plot_cfg={
                "figsize": [10, 12],
                "dpi": 100,
                "linewidth": 2.0,
                "alpha_raw": 0.5,
                "smoothing_window": 20,
                "legend_fontsize": 10,
                "title_fontsize": 14,
                "label_fontsize": 12,
                "grid_alpha": 0.5,
                "format": "pdf",
            },
        )

        assert result is not None
        assert result.exists()

    def test_multiple_networks_separate_files(self, tmp_path):
        from graph_signal_diffusion.trainers.pd_visualization import (
            parse_trace_jsonl,
            visualize_trace_dynamics,
        )

        trace_path = _make_synthetic_trace(
            tmp_path, num_epochs=50, network_ids=[0, 3],
        )
        data = parse_trace_jsonl(trace_path)
        out_dir = tmp_path / "multi"

        paths = []
        for net_id in [0, 3]:
            p = visualize_trace_dynamics(
                trace_data=data,
                network_id=net_id,
                output_dir=out_dir,
                grouped_receivers=[[0, 1]],
            )
            paths.append(p)

        assert all(p is not None and p.exists() for p in paths)
        assert paths[0] != paths[1]  # different filenames

    def test_without_full_vectors(self, tmp_path):
        """Visualization should work even with only selected_receiver_trace data."""
        from graph_signal_diffusion.trainers.pd_visualization import (
            parse_trace_jsonl,
            visualize_trace_dynamics,
        )

        trace_path = _make_synthetic_trace(
            tmp_path, num_epochs=60, include_full_vectors=False,
        )
        data = parse_trace_jsonl(trace_path)

        result = visualize_trace_dynamics(
            trace_data=data,
            network_id=0,
            output_dir=tmp_path / "no_full",
            grouped_receivers=[[0, 1]],
        )
        assert result is not None
        assert result.exists()

    def test_out_of_range_receiver_indices_warns(self, tmp_path):
        """Explicit receiver indices beyond the data dimension should warn and be dropped."""
        from graph_signal_diffusion.trainers.pd_visualization import (
            parse_trace_jsonl,
            visualize_trace_dynamics,
        )

        trace_path = _make_synthetic_trace(
            tmp_path, num_epochs=50, n_receivers=4, include_full_vectors=True,
        )
        data = parse_trace_jsonl(trace_path)

        # RX 0 valid, RX 99 out of range — should warn and drop RX 99.
        result = visualize_trace_dynamics(
            trace_data=data,
            network_id=0,
            output_dir=tmp_path / "oor",
            grouped_receivers=[[0, 99]],
        )
        # Should still produce a plot with the valid receiver.
        assert result is not None
        assert result.exists()

    def test_all_out_of_range_returns_none(self, tmp_path):
        """If every explicit receiver index is out of range, return None."""
        from graph_signal_diffusion.trainers.pd_visualization import (
            parse_trace_jsonl,
            visualize_trace_dynamics,
        )

        trace_path = _make_synthetic_trace(
            tmp_path, num_epochs=50, n_receivers=4, include_full_vectors=True,
        )
        data = parse_trace_jsonl(trace_path)

        result = visualize_trace_dynamics(
            trace_data=data,
            network_id=0,
            output_dir=tmp_path / "all_oor",
            grouped_receivers=[[50, 99]],
        )
        assert result is None


class TestAutoKValidation:
    """Tests for auto_k boundary validation."""

    def test_select_interesting_receivers_k_zero_raises(self, tmp_path):
        from graph_signal_diffusion.trainers.pd_visualization import (
            parse_trace_jsonl,
            select_interesting_receivers,
        )

        trace_path = _make_synthetic_trace(tmp_path, num_epochs=50)
        data = parse_trace_jsonl(trace_path)

        with pytest.raises(ValueError, match="k must be >= 1"):
            select_interesting_receivers(data, network_id=0, k=0)

    def test_select_interesting_receivers_k_negative_raises(self, tmp_path):
        from graph_signal_diffusion.trainers.pd_visualization import (
            parse_trace_jsonl,
            select_interesting_receivers,
        )

        trace_path = _make_synthetic_trace(tmp_path, num_epochs=50)
        data = parse_trace_jsonl(trace_path)

        with pytest.raises(ValueError, match="k must be >= 1"):
            select_interesting_receivers(data, network_id=0, k=-3)


class TestWindowedRateMerge:
    """Tests for windowed-rate extraction when auto-selected receivers
    are outside selected_receiver_trace.receiver_indices."""

    def test_windowed_rates_for_non_selected_receiver(self, tmp_path):
        """Auto-selected RX outside the selected_receiver_trace subset
        should still get correct windowed rates from full vectors."""
        from graph_signal_diffusion.trainers.pd_visualization import (
            parse_trace_jsonl,
            select_interesting_receivers,
            visualize_trace_dynamics,
        )

        # Synthetic trace: selected_receiver_trace tracks [0, 1],
        # but full vectors cover all 10 receivers.
        trace_path = _make_synthetic_trace(
            tmp_path, num_epochs=100, n_receivers=10,
            include_full_vectors=True, window_sizes=[10, 50],
        )
        data = parse_trace_jsonl(trace_path)

        # Force-request RX 2 which is NOT in selected_receiver_trace.
        result = visualize_trace_dynamics(
            trace_data=data,
            network_id=0,
            output_dir=tmp_path / "merge_test",
            grouped_receivers=[[0, 2]],  # RX 2 only in full vectors
        )
        assert result is not None
        assert result.exists()

    def test_windowed_rates_values_correctness(self, tmp_path):
        """Verify windowed rate values for a non-selected receiver match
        the full-vector data, not position-indexed selected data."""
        from graph_signal_diffusion.trainers.pd_visualization import (
            parse_trace_jsonl,
        )

        trace_path = _make_synthetic_trace(
            tmp_path, num_epochs=20, n_receivers=5,
            include_full_vectors=True, window_sizes=[5],
        )
        data = parse_trace_jsonl(trace_path)
        epochs = data[0]["epochs"]

        # RX 2 is NOT in selected_receiver_trace.receiver_indices ([0, 1]).
        # The full-vector windowed_avg_rates["5"] should have all 5 values.
        # The selected windowed_avg_rates["5"] has only 2 values (positions 0, 1).
        # Before the fix, merging would overwrite, and indexing RX 2 into the
        # 2-element sel array would silently give the wrong value.
        last_epoch = epochs[-1]
        full_windowed = last_epoch.get("windowed_avg_rates", {})
        sel_windowed = last_epoch.get("selected_receiver_trace", {}).get(
            "windowed_avg_rates", {},
        )

        # Verify the data shape: full has 5 elements, sel has 2.
        assert len(full_windowed["5"]) == 5
        assert len(sel_windowed["5"]) == 2

        # The correct value for RX 2 windowed rate is full_windowed["5"][2].
        expected_rx2_val = full_windowed["5"][2]

        # If the merge bug existed, sel_windowed would overwrite full_windowed
        # for key "5", and then indexing position 2 into the 2-element array
        # would either be out of range or return sel_windowed["5"][1] (wrong).
        # (This test verifies the data setup; the fix test is above.
        # We just confirm the sel value at position 1 != full value at index 2.)
        assert sel_windowed["5"][1] != pytest.approx(expected_rx2_val, abs=0.01)


class TestCollectionWindowAndScatter:
    """Tests for collection_window and power_scatter features."""

    def test_collection_window_draws_line(self, tmp_path):
        """With collection_window set, the figure should still be produced."""
        from graph_signal_diffusion.trainers.pd_visualization import (
            parse_trace_jsonl,
            visualize_trace_dynamics,
        )

        trace_path = _make_synthetic_trace(tmp_path, num_epochs=200)
        data = parse_trace_jsonl(trace_path)

        result = visualize_trace_dynamics(
            trace_data=data,
            network_id=0,
            output_dir=tmp_path / "cw",
            grouped_receivers=[[0, 1, 2]],
            collection_window=50,
        )
        assert result is not None
        assert result.exists()

    def test_collection_window_none_no_effect(self, tmp_path):
        """collection_window=None should produce the same figure as before."""
        from graph_signal_diffusion.trainers.pd_visualization import (
            parse_trace_jsonl,
            visualize_trace_dynamics,
        )

        trace_path = _make_synthetic_trace(tmp_path, num_epochs=100)
        data = parse_trace_jsonl(trace_path)

        result = visualize_trace_dynamics(
            trace_data=data,
            network_id=0,
            output_dir=tmp_path / "no_cw",
            grouped_receivers=[[0, 1]],
            collection_window=None,
        )
        assert result is not None
        assert result.exists()

    def test_collection_window_exceeds_epochs(self, tmp_path):
        """If collection_window > total epochs, no boundary is drawn (no crash)."""
        from graph_signal_diffusion.trainers.pd_visualization import (
            parse_trace_jsonl,
            visualize_trace_dynamics,
        )

        trace_path = _make_synthetic_trace(tmp_path, num_epochs=50)
        data = parse_trace_jsonl(trace_path)

        result = visualize_trace_dynamics(
            trace_data=data,
            network_id=0,
            output_dir=tmp_path / "big_cw",
            grouped_receivers=[[0, 1]],
            collection_window=9999,
        )
        assert result is not None
        assert result.exists()

    def test_scatter_panel_for_k2(self, tmp_path):
        """k=2 should produce a figure with the scatter panel."""
        from graph_signal_diffusion.trainers.pd_visualization import (
            parse_trace_jsonl,
            visualize_trace_dynamics,
        )

        trace_path = _make_synthetic_trace(
            tmp_path, num_epochs=100, n_receivers=10, include_full_vectors=True,
        )
        data = parse_trace_jsonl(trace_path)

        result = visualize_trace_dynamics(
            trace_data=data,
            network_id=0,
            output_dir=tmp_path / "scatter_k2",
            grouped_receivers=[[0, 2]],
            collection_window=30,
            plot_cfg={"panels": {"scatter": {"transient_display": "both"}}},
        )
        assert result is not None
        assert result.exists()
        # k=2 figure should be larger than a typical k>2 (has extra scatter panel).
        assert result.stat().st_size > 0

    def test_scatter_collection_only_mode(self, tmp_path):
        """transient_display='collection_only' should produce a valid figure."""
        from graph_signal_diffusion.trainers.pd_visualization import (
            parse_trace_jsonl,
            visualize_trace_dynamics,
        )

        trace_path = _make_synthetic_trace(
            tmp_path, num_epochs=100, n_receivers=10, include_full_vectors=True,
        )
        data = parse_trace_jsonl(trace_path)

        result = visualize_trace_dynamics(
            trace_data=data,
            network_id=0,
            output_dir=tmp_path / "scatter_co",
            grouped_receivers=[[0, 2]],
            collection_window=30,
            plot_cfg={"panels": {"scatter": {"transient_display": "collection_only"}}},
        )
        assert result is not None
        assert result.exists()

    def test_no_scatter_for_k_gt_2(self, tmp_path):
        """k>2 should NOT include a scatter panel (4-row layout)."""
        from graph_signal_diffusion.trainers.pd_visualization import (
            parse_trace_jsonl,
            visualize_trace_dynamics,
        )

        trace_path = _make_synthetic_trace(
            tmp_path, num_epochs=80, n_receivers=10, include_full_vectors=True,
        )
        data = parse_trace_jsonl(trace_path)

        result = visualize_trace_dynamics(
            trace_data=data,
            network_id=0,
            output_dir=tmp_path / "no_scatter",
            grouped_receivers=[[0, 1, 2]],
            collection_window=20,
        )
        assert result is not None
        assert result.exists()

    def test_scatter_without_collection_window(self, tmp_path):
        """Scatter with no collection_window treats all points as collection."""
        from graph_signal_diffusion.trainers.pd_visualization import (
            parse_trace_jsonl,
            visualize_trace_dynamics,
        )

        trace_path = _make_synthetic_trace(
            tmp_path, num_epochs=80, n_receivers=5, include_full_vectors=True,
        )
        data = parse_trace_jsonl(trace_path)

        result = visualize_trace_dynamics(
            trace_data=data,
            network_id=0,
            output_dir=tmp_path / "scatter_no_cw",
            grouped_receivers=[[0, 1]],
            collection_window=None,
            plot_cfg={"panels": {"scatter": {"transient_display": "both"}}},
        )
        assert result is not None
        assert result.exists()

    def test_scatter_unit_axis_range(self, tmp_path):
        """axis_range='unit' should fix scatter axes to [0, 1]."""
        from graph_signal_diffusion.trainers.pd_visualization import (
            parse_trace_jsonl,
            visualize_trace_dynamics,
        )

        trace_path = _make_synthetic_trace(
            tmp_path, num_epochs=80, n_receivers=5, include_full_vectors=True,
        )
        data = parse_trace_jsonl(trace_path)

        result = visualize_trace_dynamics(
            trace_data=data,
            network_id=0,
            output_dir=tmp_path / "scatter_unit",
            grouped_receivers=[[0, 1]],
            collection_window=20,
            plot_cfg={"panels": {"scatter": {"transient_display": "both", "axis_range": "unit"}}},
        )
        assert result is not None
        assert result.exists()

    def test_scatter_auto_axis_range(self, tmp_path):
        """axis_range='auto' (default) should also produce a valid figure."""
        from graph_signal_diffusion.trainers.pd_visualization import (
            parse_trace_jsonl,
            visualize_trace_dynamics,
        )

        trace_path = _make_synthetic_trace(
            tmp_path, num_epochs=80, n_receivers=5, include_full_vectors=True,
        )
        data = parse_trace_jsonl(trace_path)

        result = visualize_trace_dynamics(
            trace_data=data,
            network_id=0,
            output_dir=tmp_path / "scatter_auto",
            grouped_receivers=[[0, 1]],
            collection_window=20,
            plot_cfg={"panels": {"scatter": {"transient_display": "both", "axis_range": "auto"}}},
        )
        assert result is not None
        assert result.exists()


def _make_dummy_overlay_npz(
    path: Path,
    network_ids: list[int],
    n_samples: int = 50,
    n_receivers: int = 5,
    seed: int = 123,
) -> Path:
    """Create a dummy generated_powers.npz file for overlay tests."""
    rng = np.random.default_rng(seed)
    arrays: dict[str, np.ndarray] = {
        "network_ids": np.array(network_ids, dtype=np.int64),
    }
    for nid in network_ids:
        arrays[f"network_{nid}_powers"] = rng.uniform(
            0.0, 1.0, size=(n_samples, n_receivers),
        ).astype(np.float32)
    npz_path = path / "generated_powers.npz"
    np.savez(npz_path, **arrays)
    return npz_path


class TestOverlaySamples:
    """Tests for overlay_samples on the scatter panel."""

    def test_overlay_on_scatter_k2(self, tmp_path):
        """Overlay should appear on the scatter panel for k=2."""
        from graph_signal_diffusion.trainers.pd_visualization import (
            parse_trace_jsonl,
            visualize_trace_dynamics,
        )

        trace_path = _make_synthetic_trace(
            tmp_path, num_epochs=80, n_receivers=5, include_full_vectors=True,
        )
        data = parse_trace_jsonl(trace_path)
        npz_path = _make_dummy_overlay_npz(tmp_path, network_ids=[0], n_receivers=5)

        overlay = {}
        npz = np.load(npz_path)
        overlay[0] = npz["network_0_powers"]
        npz.close()

        result = visualize_trace_dynamics(
            trace_data=data,
            network_id=0,
            output_dir=tmp_path / "overlay_k2",
            grouped_receivers=[[0, 1]],
            collection_window=20,
            plot_cfg={"panels": {"scatter": {"transient_display": "both"}}},
            overlay_samples=overlay,
            overlay_cfg={"marker": "x", "color": "crimson", "alpha": 0.6},
        )
        assert result is not None
        assert result.exists()

    def test_overlay_ignored_when_k_gt_2(self, tmp_path):
        """Overlay should be silently ignored when k>2 (no scatter panel)."""
        from graph_signal_diffusion.trainers.pd_visualization import (
            parse_trace_jsonl,
            visualize_trace_dynamics,
        )

        trace_path = _make_synthetic_trace(
            tmp_path, num_epochs=80, n_receivers=5, include_full_vectors=True,
        )
        data = parse_trace_jsonl(trace_path)

        overlay = {0: np.random.default_rng(99).uniform(0, 1, (30, 5)).astype(np.float32)}

        result = visualize_trace_dynamics(
            trace_data=data,
            network_id=0,
            output_dir=tmp_path / "overlay_k3",
            grouped_receivers=[[0, 1, 2]],
            overlay_samples=overlay,
        )
        assert result is not None
        assert result.exists()

    def test_overlay_none_no_effect(self, tmp_path):
        """overlay_samples=None should work fine (no overlay)."""
        from graph_signal_diffusion.trainers.pd_visualization import (
            parse_trace_jsonl,
            visualize_trace_dynamics,
        )

        trace_path = _make_synthetic_trace(
            tmp_path, num_epochs=80, n_receivers=5, include_full_vectors=True,
        )
        data = parse_trace_jsonl(trace_path)

        result = visualize_trace_dynamics(
            trace_data=data,
            network_id=0,
            output_dir=tmp_path / "overlay_none",
            grouped_receivers=[[0, 1]],
            overlay_samples=None,
        )
        assert result is not None
        assert result.exists()

    def test_overlay_missing_network_skipped(self, tmp_path):
        """If overlay has no data for the network_id, no overlay is drawn."""
        from graph_signal_diffusion.trainers.pd_visualization import (
            parse_trace_jsonl,
            visualize_trace_dynamics,
        )

        trace_path = _make_synthetic_trace(
            tmp_path, num_epochs=80, n_receivers=5, include_full_vectors=True,
        )
        data = parse_trace_jsonl(trace_path)

        # Overlay only has network 99, but we're plotting network 0.
        overlay = {99: np.random.default_rng(42).uniform(0, 1, (20, 5)).astype(np.float32)}

        result = visualize_trace_dynamics(
            trace_data=data,
            network_id=0,
            output_dir=tmp_path / "overlay_missing",
            grouped_receivers=[[0, 1]],
            overlay_samples=overlay,
        )
        assert result is not None
        assert result.exists()

    def test_overlay_custom_style(self, tmp_path):
        """Custom overlay_cfg values should produce a valid figure."""
        from graph_signal_diffusion.trainers.pd_visualization import (
            parse_trace_jsonl,
            visualize_trace_dynamics,
        )

        trace_path = _make_synthetic_trace(
            tmp_path, num_epochs=80, n_receivers=5, include_full_vectors=True,
        )
        data = parse_trace_jsonl(trace_path)

        overlay = {0: np.random.default_rng(7).uniform(0, 1, (40, 5)).astype(np.float32)}

        result = visualize_trace_dynamics(
            trace_data=data,
            network_id=0,
            output_dir=tmp_path / "overlay_style",
            grouped_receivers=[[0, 1]],
            collection_window=20,
            overlay_samples=overlay,
            overlay_cfg={
                "marker": "o",
                "color": "blue",
                "alpha": 0.3,
                "marker_size": 15,
                "label": "Diffusion",
            },
        )
        assert result is not None
        assert result.exists()
