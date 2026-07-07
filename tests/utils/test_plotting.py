import json
from pathlib import Path
import tempfile

from graph_signal_diffusion.utils.plotting import plot_epoch_summaries_jsonl


def test_plot_epoch_summaries_jsonl_creates_pdfs(tmp_path):
    outdir = tmp_path / "out"
    outdir.mkdir()
    jsonl = outdir / "epoch_summaries.jsonl"

    rows = [
        {"epoch": 1, "train_loss": 1.0, "train_grad_norm": 10.0, "mse": 0.5},
        {"epoch": 2, "train_loss": 0.8, "train_grad_norm": 5.0, "mse": 0.4, "mae": 0.2},
        {"epoch": 3, "train_loss": 0.6, "train_grad_norm": 2.0, "mse": 0.35, "rmse": 0.6},
    ]

    with open(jsonl, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    plot_epoch_summaries_jsonl(jsonl, outdir, save_pdf=True, grad_logscale=True)

    assert (outdir / "training_loss.pdf").exists()
    assert (outdir / "grad_norm.pdf").exists()
    assert (outdir / "mae.pdf").exists()


def test_plot_epoch_summaries_jsonl_wra_percentile_gap_plot(tmp_path):
    outdir = tmp_path / "out_wra"
    outdir.mkdir()
    jsonl = outdir / "epoch_summaries.jsonl"

    rows = [
        {"epoch": 1, "train_loss": 1.0, "train_grad_norm": 10.0, "val_rate_1pct_gap_pct": 20.0, "val_rate_5pct_gap_pct": 15.0},
        {"epoch": 2, "train_loss": 0.8, "train_grad_norm": 5.0, "val_rate_1pct_gap_pct": 18.0, "val_rate_5pct_gap_pct": 13.0},
        {"epoch": 3, "train_loss": 0.6, "train_grad_norm": 2.0, "val_rate_1pct_gap_pct": 16.0, "val_rate_5pct_gap_pct": 12.0},
    ]

    with open(jsonl, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    plot_epoch_summaries_jsonl(jsonl, outdir, save_pdf=True, grad_logscale=True)

    assert (outdir / "wra_performance_gaps.pdf").exists()


def test_plot_epoch_summaries_jsonl_wra_violations_and_mean_gap_plot(tmp_path):
    outdir = tmp_path / "out_wra_violations"
    outdir.mkdir()
    jsonl = outdir / "epoch_summaries.jsonl"

    rows = [
        {
            "epoch": 1,
            "train_loss": 1.0,
            "train_grad_norm": 10.0,
            "val_power_violation_percentage_generated": 12.0,
            "val_rate_violation_percentage_generated": 35.0,
            "val_rate_mean_violation_gap_pct_generated": 28.0,
            "val_rate_mean_violation_gap_pct_real": 12.0,
            "train-val_power_violation_percentage_generated": 10.0,
            "train-val_rate_violation_percentage_generated": 30.0,
            "train-val_rate_mean_violation_gap_pct_generated": 24.0,
            "train-val_rate_mean_violation_gap_pct_real": 10.0,
        },
        {
            "epoch": 2,
            "train_loss": 0.8,
            "train_grad_norm": 5.0,
            "val_power_violation_percentage_generated": 10.0,
            "val_rate_violation_percentage_generated": 28.0,
            "val_rate_mean_violation_gap_pct_generated": 22.0,
            "val_rate_mean_violation_gap_pct_real": 10.0,
            "train-val_power_violation_percentage_generated": 8.0,
            "train-val_rate_violation_percentage_generated": 24.0,
            "train-val_rate_mean_violation_gap_pct_generated": 20.0,
            "train-val_rate_mean_violation_gap_pct_real": 8.0,
        },
        {
            "epoch": 3,
            "train_loss": 0.6,
            "train_grad_norm": 2.0,
            "val_power_violation_percentage_generated": 9.0,
            "val_rate_violation_percentage_generated": 21.0,
            "val_rate_mean_violation_gap_pct_generated": 16.0,
            "val_rate_mean_violation_gap_pct_real": 6.0,
            "train-val_power_violation_percentage_generated": 7.0,
            "train-val_rate_violation_percentage_generated": 18.0,
            "train-val_rate_mean_violation_gap_pct_generated": 14.0,
            "train-val_rate_mean_violation_gap_pct_real": 6.0,
        },
    ]

    with open(jsonl, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    plot_epoch_summaries_jsonl(jsonl, outdir, save_pdf=True, grad_logscale=True)

    assert (outdir / "wra_violations.pdf").exists()
