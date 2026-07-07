"""Unit tests for selector-temperature restoration from experiment artifacts."""
from __future__ import annotations

import json

import pytest
import torch.nn as nn

from graph_signal_diffusion.utils.selector_temperature_restore import (
    resolve_selector_temperature_from_records,
    restore_selector_temperature_from_records,
)


class _DummySelector(nn.Module):
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.selection_mode = "ste"
        self.temperature = float(temperature)


class _DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.selector_a = _DummySelector(temperature=1.0)
        self.selector_b = _DummySelector(temperature=1.0)
        self.linear = nn.Linear(1, 1)


def _write_base_experiment_tree(tmp_path):
    exp_dir = tmp_path / "exp"
    (exp_dir / ".hydra").mkdir(parents=True)
    (exp_dir / ".hydra" / "config.yaml").write_text(
        "model:\n"
        "  config:\n"
        "    pooling_config:\n"
        "      selector_kwargs:\n"
        "        temperature: 1.0\n"
        "        temperature_schedule: linear\n"
        "        temperature_min: 0.2\n"
        "        temperature_warmup_steps: 2\n"
        "        temperature_anneal_steps: 6\n"
    )
    return exp_dir


def test_restore_uses_epoch_summaries_logged_selector_temperature(tmp_path):
    exp_dir = _write_base_experiment_tree(tmp_path)
    ckpt_path = exp_dir / "trainer_chkpts" / "best_models" / "best_model_epoch_1500.pt"
    ckpt_path.parent.mkdir(parents=True)
    ckpt_path.touch()

    (exp_dir / "epoch_summaries.jsonl").write_text(
        json.dumps(
            {
                "epoch": 1500,
                "selector_temperature": 0.5747092166969194,
                "num_successful_steps": 17,
            }
        )
        + "\n"
    )

    model = _DummyModel()
    info = restore_selector_temperature_from_records(
        root_module=model,
        checkpoint_path=ckpt_path,
        checkpoint_epoch=1500,
        experiment_dir=exp_dir,
    )

    assert info is not None
    assert info.temperature == pytest.approx(0.5747092166969194)
    assert "epoch_summaries.jsonl" in info.source
    assert info.modules_updated == 2
    assert model.selector_a.temperature == pytest.approx(0.5747092166969194)
    assert model.selector_b.temperature == pytest.approx(0.5747092166969194)


def test_restore_falls_back_to_best_models_manifest(tmp_path):
    exp_dir = _write_base_experiment_tree(tmp_path)
    ckpt_path = exp_dir / "trainer_chkpts" / "best_models" / "best_model_epoch_2300.pt"
    ckpt_path.parent.mkdir(parents=True)
    ckpt_path.touch()

    manifest = {
        "config": {},
        "leaderboard": [
            {
                "rank": 1,
                "epoch": 2300,
                "path": str(ckpt_path.resolve()),
                "raw_metrics": {
                    "selector_temperature": 0.2073369699645704,
                },
                "ema_metrics": {},
            }
        ],
        "total_evaluations": 1,
        "last_updated_epoch": 2300,
    }
    manifest_path = exp_dir / "trainer_chkpts" / "best_models" / "best_models_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    model = _DummyModel()
    info = restore_selector_temperature_from_records(
        root_module=model,
        checkpoint_path=ckpt_path,
        checkpoint_epoch=2300,
        experiment_dir=exp_dir,
    )

    assert info is not None
    assert info.temperature == pytest.approx(0.2073369699645704)
    assert "best_models_manifest.json" in info.source
    assert info.modules_updated == 2


def test_resolve_computes_schedule_when_epoch_row_lacks_temperature(tmp_path):
    exp_dir = _write_base_experiment_tree(tmp_path)
    ckpt_path = exp_dir / "trainer_chkpts" / "DDIM_epoch_2.pt"
    ckpt_path.parent.mkdir(parents=True)
    ckpt_path.touch()

    # epoch 0 contributes 3 successful steps; epoch 1 has 2 successful steps.
    # For linear schedule (T0=1.0, Tmin=0.2, warmup=2, anneal=6):
    # step 3 -> 0.866666..., step 4 -> 0.733333..., mean -> 0.8.
    rows = [
        {"epoch": 0, "num_successful_steps": 3},
        {"epoch": 1, "num_successful_steps": 2},
    ]
    with (exp_dir / "epoch_summaries.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    record = resolve_selector_temperature_from_records(
        checkpoint_path=ckpt_path,
        checkpoint_epoch=1,
        experiment_dir=exp_dir,
    )

    assert record is not None
    assert "computed_from_schedule" in record.source
    assert record.temperature == pytest.approx(0.8)


def test_resolve_accepts_filename_epoch_fallback_to_minus_one(tmp_path):
    exp_dir = _write_base_experiment_tree(tmp_path)
    ckpt_path = exp_dir / "trainer_chkpts" / "DDIM_epoch_350.pt"
    ckpt_path.parent.mkdir(parents=True)
    ckpt_path.touch()

    (exp_dir / "epoch_summaries.jsonl").write_text(
        json.dumps(
            {
                "epoch": 349,
                "selector_temperature": 0.444,
                "num_successful_steps": 17,
            }
        )
        + "\n"
    )

    record = resolve_selector_temperature_from_records(
        checkpoint_path=ckpt_path,
        checkpoint_epoch=None,
        experiment_dir=exp_dir,
    )

    assert record is not None
    assert record.temperature == pytest.approx(0.444)
    assert record.epoch == 349

