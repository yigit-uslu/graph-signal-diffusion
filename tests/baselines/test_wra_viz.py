import numpy as np

from graph_signal_diffusion.baselines.wireless_resource_allocation.viz import (
    _build_baseline_color_map,
    plot_power_allocation_histogram,
    plot_wra_percentile_evolution_comparison,
)
from graph_signal_diffusion.baselines.plot_utils import format_baseline_display_name


def test_plot_power_allocation_histogram_saves_pdf(tmp_path):
    power_real = np.linspace(-0.5, 0.5, 200, dtype=np.float32)
    power_gen = np.linspace(-0.4, 0.4, 200, dtype=np.float32)
    save_path = tmp_path / "power_allocation_histogram_test.pdf"

    w1 = plot_power_allocation_histogram(
        power_real=power_real,
        power_gen=power_gen,
        save_path=str(save_path),
        baseline_name="FP",
    )

    assert np.isfinite(w1)
    assert save_path.exists()


def test_build_baseline_color_map_respects_configured_colors() -> None:
    order_a = ["fp", "wmmse"]
    order_b = ["wmmse", "fp"]
    configured = {"fp": "#123456", "wmmse": "#abcdef"}
    default_cycle = ["tab:blue", "tab:orange", "tab:green"]

    cmap_a = _build_baseline_color_map(order_a, configured, default_cycle)
    cmap_b = _build_baseline_color_map(order_b, configured, default_cycle)

    assert cmap_a["fp"] == "#123456"
    assert cmap_b["fp"] == "#123456"
    assert cmap_a["wmmse"] == "#abcdef"
    assert cmap_b["wmmse"] == "#abcdef"


def test_format_baseline_display_name_uppercases_common_abbreviations() -> None:
    assert format_baseline_display_name("fp") == "FP"
    assert format_baseline_display_name("grw") == "GRW"
    assert format_baseline_display_name("wmmse") == "WMMSE"
    assert format_baseline_display_name("diffusion") == "U-GNN"


def test_plot_wra_percentile_evolution_uses_fixed_colors_and_rmin_line(
    tmp_path,
    monkeypatch,
):
    import matplotlib.axes

    base = np.array(
        [
            [0.9, 1.1, 1.0],
            [1.0, 1.2, 1.1],
            [1.1, 1.3, 1.2],
        ],
        dtype=np.float32,
    )
    records_by_baseline = {
        "fp": [
            {
                "gen_rates_per_slot": base * 0.95,
                "real_rates_per_slot": base,
                "r_min": 0.7,
            }
        ],
        "wmmse": [
            {
                "gen_rates_per_slot": base * 1.02,
                "real_rates_per_slot": base,
                "r_min": 0.7,
            }
        ],
    }
    baseline_colors = {
        "expert": "#010101",
        "fp": "#111111",
        "wmmse": "#222222",
    }

    plot_calls: list[tuple[str | None, str | None]] = []
    hline_calls: list[tuple[float, str | None, str | None]] = []
    original_plot = matplotlib.axes.Axes.plot
    original_axhline = matplotlib.axes.Axes.axhline

    def _capture_plot(self, *args, **kwargs):
        label = kwargs.get("label")
        color = kwargs.get("color")
        if label in {"Expert (real)", "FP", "WMMSE"}:
            plot_calls.append((label, color))
        return original_plot(self, *args, **kwargs)

    def _capture_axhline(self, y=0, xmin=0, xmax=1, **kwargs):
        hline_calls.append((float(y), kwargs.get("color"), kwargs.get("linestyle")))
        return original_axhline(self, y=y, xmin=xmin, xmax=xmax, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "plot", _capture_plot)
    monkeypatch.setattr(matplotlib.axes.Axes, "axhline", _capture_axhline)

    save_path = tmp_path / "wra_percentile_evolution_test.pdf"
    plot_wra_percentile_evolution_comparison(
        records_by_baseline=records_by_baseline,
        save_path=str(save_path),
        baseline_order=["wmmse", "fp"],
        baseline_colors=baseline_colors,
    )

    assert save_path.exists()

    fp_colors = {color for label, color in plot_calls if label == "FP"}
    wmmse_colors = {color for label, color in plot_calls if label == "WMMSE"}
    expert_colors = {color for label, color in plot_calls if label == "Expert (real)"}
    assert fp_colors == {"#111111"}
    assert wmmse_colors == {"#222222"}
    assert expert_colors == {"#010101"}

    assert len(hline_calls) == 3
    assert all(np.isclose(y, 0.7) for y, _, _ in hline_calls)
    assert all(color == "red" for _, color, _ in hline_calls)
    assert all(linestyle == "--" for _, _, linestyle in hline_calls)


def test_plot_wra_percentile_evolution_aggregates_by_realization_with_errorbars(
    tmp_path,
    monkeypatch,
) -> None:
    import matplotlib.axes

    records_by_baseline = {
        "diffusion": [
            {
                "gen_rates_per_slot": np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32),
                "real_rates_per_slot": np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.float32),
                "eval_batch_idx": 0,
                "r_min": 0.5,
            },
            {
                "gen_rates_per_slot": np.array([[3.0, 3.0], [3.0, 3.0]], dtype=np.float32),
                "real_rates_per_slot": np.array([[6.0, 6.0], [6.0, 6.0]], dtype=np.float32),
                "eval_batch_idx": 1,
                "r_min": 0.5,
            },
        ],
        "fp": [
            {
                "gen_rates_per_slot": np.array([[10.0, 10.0], [10.0, 10.0]], dtype=np.float32),
                "real_rates_per_slot": np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.float32),
                "eval_batch_idx": 0,
                "r_min": 0.5,
            },
            {
                "gen_rates_per_slot": np.array([[20.0, 20.0], [20.0, 20.0]], dtype=np.float32),
                "real_rates_per_slot": np.array([[6.0, 6.0], [6.0, 6.0]], dtype=np.float32),
                "eval_batch_idx": 1,
                "r_min": 0.5,
            },
        ],
    }
    baseline_colors = {
        "expert": "#010101",
        "diffusion": "#ff8800",
        "fp": "#008800",
    }

    plot_y_by_label: dict[str, list[np.ndarray]] = {}
    errorbar_calls: list[tuple[np.ndarray, np.ndarray, str | None]] = []
    original_plot = matplotlib.axes.Axes.plot
    original_errorbar = matplotlib.axes.Axes.errorbar

    def _capture_plot(self, *args, **kwargs):
        label = kwargs.get("label")
        if isinstance(label, str):
            y = np.asarray(args[1], dtype=np.float64)
            plot_y_by_label.setdefault(label, []).append(y)
        return original_plot(self, *args, **kwargs)

    def _capture_errorbar(self, *args, **kwargs):
        y = np.asarray(args[1], dtype=np.float64)
        yerr = np.asarray(kwargs.get("yerr"), dtype=np.float64)
        ecolor = kwargs.get("ecolor")
        errorbar_calls.append((y, yerr, ecolor))
        return original_errorbar(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "plot", _capture_plot)
    monkeypatch.setattr(matplotlib.axes.Axes, "errorbar", _capture_errorbar)

    save_path = tmp_path / "wra_percentile_realization_aggregation.pdf"
    plot_wra_percentile_evolution_comparison(
        records_by_baseline=records_by_baseline,
        save_path=str(save_path),
        baseline_order=["diffusion", "fp"],
        baseline_colors=baseline_colors,
    )
    assert save_path.exists()

    assert "U-GNN" in plot_y_by_label
    assert "Expert (real)" in plot_y_by_label
    assert "FP" in plot_y_by_label
    assert all(np.allclose(y, np.array([2.0, 2.0])) for y in plot_y_by_label["U-GNN"])
    assert all(np.allclose(y, np.array([4.0, 4.0])) for y in plot_y_by_label["Expert (real)"])
    assert all(np.allclose(y, np.array([15.0, 15.0])) for y in plot_y_by_label["FP"])

    expert_err = [(y, yerr) for y, yerr, c in errorbar_calls if c == "#010101"]
    diffusion_err = [(y, yerr) for y, yerr, c in errorbar_calls if c == "#ff8800"]
    fp_err = [(y, yerr) for y, yerr, c in errorbar_calls if c == "#008800"]

    assert len(expert_err) == 4
    assert len(diffusion_err) == 4
    assert len(fp_err) == 0
    assert all(np.allclose(y, np.array([4.0, 4.0])) for y, _ in expert_err)
    assert all(np.allclose(yerr, np.array([2.0, 2.0])) for _, yerr in expert_err)
    assert all(np.allclose(y, np.array([2.0, 2.0])) for y, _ in diffusion_err)
    assert all(np.allclose(yerr, np.array([1.0, 1.0])) for _, yerr in diffusion_err)


def test_plot_wra_percentile_evolution_markers_and_errorbars_every_five_points(
    tmp_path,
    monkeypatch,
) -> None:
    import matplotlib.axes

    slots = 11
    marker_x = np.array([1.0, 6.0, 11.0], dtype=np.float64)
    expected_markevery = [0, 5, 10]
    curve = np.tile(np.array([[1.0, 1.2]], dtype=np.float32), (slots, 1))

    records_by_baseline = {
        "diffusion": [
            {
                "gen_rates_per_slot": curve,
                "real_rates_per_slot": curve * 1.1,
                "eval_batch_idx": 0,
                "r_min": 0.5,
            }
        ],
        "fp": [
            {
                "gen_rates_per_slot": curve * 0.9,
                "real_rates_per_slot": curve * 1.1,
                "eval_batch_idx": 0,
                "r_min": 0.5,
            }
        ],
    }

    markevery_by_label: dict[str, list[int] | None] = {}
    errorbar_x_by_color: dict[str, list[np.ndarray]] = {}
    original_plot = matplotlib.axes.Axes.plot
    original_errorbar = matplotlib.axes.Axes.errorbar

    def _capture_plot(self, *args, **kwargs):
        label = kwargs.get("label")
        if isinstance(label, str):
            markevery = kwargs.get("markevery")
            markevery_by_label[label] = list(markevery) if markevery is not None else None
        return original_plot(self, *args, **kwargs)

    def _capture_errorbar(self, *args, **kwargs):
        ecolor = kwargs.get("ecolor")
        x = np.asarray(args[0], dtype=np.float64)
        if isinstance(ecolor, str):
            errorbar_x_by_color.setdefault(ecolor, []).append(x)
        return original_errorbar(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "plot", _capture_plot)
    monkeypatch.setattr(matplotlib.axes.Axes, "errorbar", _capture_errorbar)

    save_path = tmp_path / "wra_percentile_markers_every_five.pdf"
    plot_wra_percentile_evolution_comparison(
        records_by_baseline=records_by_baseline,
        save_path=str(save_path),
        baseline_order=["diffusion", "fp"],
        baseline_colors={"expert": "#010101", "diffusion": "#ff8800", "fp": "#008800"},
    )
    assert save_path.exists()

    assert markevery_by_label["Expert (real)"] == expected_markevery
    assert markevery_by_label["U-GNN"] == expected_markevery
    assert markevery_by_label["FP"] == expected_markevery

    assert len(errorbar_x_by_color["#010101"]) == 4
    assert len(errorbar_x_by_color["#ff8800"]) == 4
    assert all(np.allclose(x, marker_x) for x in errorbar_x_by_color["#010101"])
    assert all(np.allclose(x, marker_x) for x in errorbar_x_by_color["#ff8800"])


def test_plot_wra_percentile_evolution_band_mode_and_total_rx_title(
    tmp_path,
    monkeypatch,
) -> None:
    import matplotlib.axes
    import matplotlib.figure

    slots = 6
    curve_a = np.tile(np.array([[1.0, 1.1]], dtype=np.float32), (slots, 1))
    curve_b = np.tile(np.array([[1.2, 1.3]], dtype=np.float32), (slots, 1))

    records_by_baseline = {
        "diffusion": [
            {
                "dataset_name": "testset",
                "network_id": 0,
                "gen_rates_per_slot": curve_a,
                "real_rates_per_slot": curve_a * 1.05,
                "eval_batch_idx": 0,
                "r_min": 0.5,
            },
            {
                "dataset_name": "testset",
                "network_id": 1,
                "gen_rates_per_slot": curve_b,
                "real_rates_per_slot": curve_b * 1.05,
                "eval_batch_idx": 0,
                "r_min": 0.5,
            },
        ],
        "fp": [
            {
                "dataset_name": "testset",
                "network_id": 0,
                "gen_rates_per_slot": curve_a * 0.95,
                "real_rates_per_slot": curve_a * 1.05,
                "eval_batch_idx": 0,
                "r_min": 0.5,
            },
            {
                "dataset_name": "testset",
                "network_id": 1,
                "gen_rates_per_slot": curve_b * 0.95,
                "real_rates_per_slot": curve_b * 1.05,
                "eval_batch_idx": 0,
                "r_min": 0.5,
            },
        ],
    }

    fill_between_calls: list[np.ndarray] = []
    errorbar_calls: list[np.ndarray] = []
    suptitle_texts: list[str] = []
    original_fill_between = matplotlib.axes.Axes.fill_between
    original_errorbar = matplotlib.axes.Axes.errorbar
    original_suptitle = matplotlib.figure.Figure.suptitle

    def _capture_fill_between(self, *args, **kwargs):
        x = np.asarray(args[0], dtype=np.float64)
        fill_between_calls.append(x)
        return original_fill_between(self, *args, **kwargs)

    def _capture_errorbar(self, *args, **kwargs):
        x = np.asarray(args[0], dtype=np.float64)
        errorbar_calls.append(x)
        return original_errorbar(self, *args, **kwargs)

    def _capture_suptitle(self, t, *args, **kwargs):
        suptitle_texts.append(str(t))
        return original_suptitle(self, t, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "fill_between", _capture_fill_between)
    monkeypatch.setattr(matplotlib.axes.Axes, "errorbar", _capture_errorbar)
    monkeypatch.setattr(matplotlib.figure.Figure, "suptitle", _capture_suptitle)

    save_path = tmp_path / "wra_percentile_band_mode.pdf"
    plot_wra_percentile_evolution_comparison(
        records_by_baseline=records_by_baseline,
        save_path=str(save_path),
        baseline_order=["diffusion", "fp"],
        baseline_colors={"expert": "#010101", "diffusion": "#ff8800", "fp": "#008800"},
        marker_every=3,
        error_display="band",
    )
    assert save_path.exists()

    # 4 panels x 2 variability curves (expert + diffusion)
    assert len(fill_between_calls) == 8
    assert len(errorbar_calls) == 0
    assert all(np.allclose(x, np.arange(1, slots + 1, dtype=np.float64)) for x in fill_between_calls)
    assert any("cumulative average across 4 rx" in t for t in suptitle_texts)
