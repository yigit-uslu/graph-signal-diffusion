"""Restore selector temperature from run artifacts for inference diagnostics.

Checkpoint state dicts do not persist selector schedule state (for example,
``current_step``), so post-hoc inference often resets selector temperature to
its config default.  This helper recovers the checkpoint-era selector
temperature from experiment logs and applies it to selector modules.
"""
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from omegaconf import OmegaConf

log = logging.getLogger(__name__)


@dataclass
class SelectorTemperatureRecord:
    """Resolved selector temperature from experiment records."""

    temperature: float
    source: str
    epoch: Optional[int] = None


@dataclass
class SelectorTemperatureRestoreInfo(SelectorTemperatureRecord):
    """Selector-temperature restore result after applying to model modules."""

    modules_updated: int = 0


def _coerce_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None:
            return default
        out = float(value)
        if not math.isfinite(out):
            return default
        return out
    except (TypeError, ValueError):
        return default


def _iter_candidate_epochs(
    checkpoint_path: Path,
    checkpoint_epoch: Optional[int],
) -> list[int]:
    if checkpoint_epoch is not None:
        return [int(checkpoint_epoch)]

    match = re.search(r"epoch[_-](\d+)", checkpoint_path.name, re.IGNORECASE)
    if not match:
        return []

    parsed = int(match.group(1))
    # Support both zero-based and one-based filename conventions.
    epochs = [parsed]
    if parsed > 0:
        epochs.append(parsed - 1)
    # De-duplicate while preserving order.
    deduped: list[int] = []
    for e in epochs:
        if e not in deduped:
            deduped.append(e)
    return deduped


def _find_experiment_dir(
    checkpoint_path: Path,
    experiment_dir: Optional[str | Path] = None,
    max_depth: int = 6,
) -> Optional[Path]:
    if experiment_dir is not None:
        exp = Path(experiment_dir).resolve()
        if (exp / ".hydra" / "config.yaml").is_file():
            return exp
        return None

    current = checkpoint_path.resolve().parent
    for _ in range(max_depth):
        if (current / ".hydra" / "config.yaml").is_file():
            return current
        current = current.parent
    return None


def _load_epoch_rows_by_epoch(path: Path) -> dict[int, dict[str, Any]]:
    rows_by_epoch: dict[int, dict[str, Any]] = {}
    if not path.is_file():
        return rows_by_epoch

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            payload = line.strip()
            if not payload:
                continue
            try:
                row = json.loads(payload)
            except json.JSONDecodeError:
                log.debug("Skipping malformed JSONL line %d in %s", line_no, path)
                continue

            epoch = _coerce_int(row.get("epoch"), default=None)
            if epoch is None:
                continue
            # Keep the last record for an epoch if duplicates exist.
            rows_by_epoch[epoch] = row

    return rows_by_epoch


def _load_selector_schedule_config(experiment_dir: Path) -> Optional[dict[str, Any]]:
    cfg_path = experiment_dir / ".hydra" / "config.yaml"
    if not cfg_path.is_file():
        return None

    try:
        cfg = OmegaConf.load(str(cfg_path))
    except Exception:
        return None

    selector_kwargs = OmegaConf.select(cfg, "model.config.pooling_config.selector_kwargs")
    if selector_kwargs is None:
        return None
    if OmegaConf.is_config(selector_kwargs):
        selector_kwargs = OmegaConf.to_container(selector_kwargs, resolve=True)
    if not isinstance(selector_kwargs, dict):
        return None

    schedule = str(selector_kwargs.get("temperature_schedule", "constant")).lower()
    if schedule not in {"constant", "linear", "cosine"}:
        schedule = "constant"

    temp0 = _coerce_float(selector_kwargs.get("temperature"), default=1.0)
    temp_min = _coerce_float(selector_kwargs.get("temperature_min"), default=0.1)
    warmup = _coerce_int(selector_kwargs.get("temperature_warmup_steps"), default=0)
    anneal = _coerce_int(selector_kwargs.get("temperature_anneal_steps"), default=0)

    if temp0 is None or temp_min is None:
        return None
    if temp_min <= 0:
        return None
    if warmup is None or warmup < 0:
        warmup = 0
    if anneal is None or anneal < 0:
        anneal = 0

    return {
        "temperature": float(temp0),
        "temperature_min": float(temp_min),
        "temperature_schedule": schedule,
        "temperature_warmup_steps": int(warmup),
        "temperature_anneal_steps": int(anneal),
    }


def _scheduled_temperature_at_step(step: int, schedule_cfg: dict[str, Any]) -> float:
    t0 = float(schedule_cfg["temperature"])
    tmin = float(schedule_cfg["temperature_min"])
    warmup = int(schedule_cfg["temperature_warmup_steps"])
    anneal = int(schedule_cfg["temperature_anneal_steps"])
    schedule = str(schedule_cfg["temperature_schedule"])

    if schedule == "constant" or anneal <= 0:
        return t0
    if step < warmup:
        return t0

    progress = (step - warmup) / max(anneal, 1)
    progress = min(max(progress, 0.0), 1.0)

    if schedule == "linear":
        temp = t0 + progress * (tmin - t0)
    elif schedule == "cosine":
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        temp = tmin + (t0 - tmin) * cosine
    else:
        temp = t0
    return float(max(temp, tmin))


def _compute_epoch_mean_temperature(
    schedule_cfg: dict[str, Any],
    start_step: int,
    num_steps: int,
) -> float:
    if num_steps <= 0:
        return _scheduled_temperature_at_step(start_step, schedule_cfg)
    temps = (
        _scheduled_temperature_at_step(start_step + i, schedule_cfg)
        for i in range(num_steps)
    )
    return float(sum(temps) / float(num_steps))


def _resolve_from_epoch_summaries(
    experiment_dir: Path,
    candidate_epochs: Iterable[int],
) -> Optional[SelectorTemperatureRecord]:
    epoch_rows = _load_epoch_rows_by_epoch(experiment_dir / "epoch_summaries.jsonl")
    if not epoch_rows:
        return None

    schedule_cfg: Optional[dict[str, Any]] = None
    ordered_epochs = sorted(epoch_rows.keys())

    for target_epoch in candidate_epochs:
        row = epoch_rows.get(int(target_epoch))
        if row is None:
            continue

        logged_temp = _coerce_float(row.get("selector_temperature"), default=None)
        if logged_temp is not None:
            return SelectorTemperatureRecord(
                temperature=logged_temp,
                source=f"epoch_summaries.jsonl:epoch={target_epoch}:selector_temperature",
                epoch=int(target_epoch),
            )

        if schedule_cfg is None:
            schedule_cfg = _load_selector_schedule_config(experiment_dir)
        if schedule_cfg is None:
            continue

        start_step = 0
        for epoch in ordered_epochs:
            if epoch >= int(target_epoch):
                break
            start_step += max(
                _coerce_int(epoch_rows[epoch].get("num_successful_steps"), default=0) or 0,
                0,
            )
        num_steps = max(_coerce_int(row.get("num_successful_steps"), default=0) or 0, 0)
        temp = _compute_epoch_mean_temperature(schedule_cfg, start_step, num_steps)
        return SelectorTemperatureRecord(
            temperature=temp,
            source=(
                "epoch_summaries.jsonl:computed_from_schedule"
                f":epoch={target_epoch}:start_step={start_step}:num_steps={num_steps}"
            ),
            epoch=int(target_epoch),
        )

    return None


def _resolve_from_best_models_manifest(
    experiment_dir: Path,
    checkpoint_path: Path,
    candidate_epochs: Iterable[int],
) -> Optional[SelectorTemperatureRecord]:
    manifest_path = experiment_dir / "trainer_chkpts" / "best_models" / "best_models_manifest.json"
    if not manifest_path.is_file():
        return None

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    leaderboard = payload.get("leaderboard", [])
    if not isinstance(leaderboard, list):
        return None

    target_resolved = checkpoint_path.resolve(strict=False)

    for row in leaderboard:
        if not isinstance(row, dict):
            continue
        raw_path = row.get("path")
        if raw_path is None:
            continue
        try:
            row_path = Path(str(raw_path)).resolve(strict=False)
        except Exception:
            continue
        if row_path != target_resolved:
            continue
        raw_metrics = row.get("raw_metrics", {})
        temp = _coerce_float(
            raw_metrics.get("selector_temperature") if isinstance(raw_metrics, dict) else None,
            default=None,
        )
        if temp is None:
            continue
        return SelectorTemperatureRecord(
            temperature=temp,
            source="best_models_manifest.json:path_match:raw_metrics.selector_temperature",
            epoch=_coerce_int(row.get("epoch"), default=None),
        )

    for target_epoch in candidate_epochs:
        matches = [
            row for row in leaderboard
            if isinstance(row, dict) and _coerce_int(row.get("epoch"), default=None) == int(target_epoch)
        ]
        for row in matches:
            raw_metrics = row.get("raw_metrics", {})
            temp = _coerce_float(
                raw_metrics.get("selector_temperature") if isinstance(raw_metrics, dict) else None,
                default=None,
            )
            if temp is None:
                continue
            return SelectorTemperatureRecord(
                temperature=temp,
                source=(
                    "best_models_manifest.json:epoch_match:raw_metrics.selector_temperature"
                    f":epoch={target_epoch}"
                ),
                epoch=int(target_epoch),
            )

    return None


def resolve_selector_temperature_from_records(
    checkpoint_path: str | Path,
    checkpoint_epoch: Optional[int] = None,
    experiment_dir: Optional[str | Path] = None,
) -> Optional[SelectorTemperatureRecord]:
    """Resolve checkpoint-era selector temperature from experiment records."""
    ckpt_path = Path(checkpoint_path)
    exp_dir = _find_experiment_dir(ckpt_path, experiment_dir=experiment_dir)
    if exp_dir is None:
        return None

    candidate_epochs = _iter_candidate_epochs(ckpt_path, checkpoint_epoch)
    if not candidate_epochs:
        log.debug("No candidate epochs resolved for selector-temperature restore: %s", ckpt_path)

    resolved = _resolve_from_epoch_summaries(exp_dir, candidate_epochs)
    if resolved is not None:
        return resolved

    resolved = _resolve_from_best_models_manifest(
        exp_dir, checkpoint_path=ckpt_path, candidate_epochs=candidate_epochs
    )
    if resolved is not None:
        return resolved

    return None


def apply_selector_temperature(root_module: Any, temperature: float) -> int:
    """Apply ``temperature`` to selector modules found in ``root_module``."""
    if root_module is None or not hasattr(root_module, "modules"):
        return 0

    updated = 0
    for module in root_module.modules():
        if not hasattr(module, "selection_mode"):
            continue
        if not hasattr(module, "temperature"):
            continue
        try:
            module.temperature = float(temperature)
            updated += 1
        except Exception:
            continue
    return updated


def restore_selector_temperature_from_records(
    *,
    root_module: Any,
    checkpoint_path: str | Path,
    checkpoint_epoch: Optional[int] = None,
    experiment_dir: Optional[str | Path] = None,
) -> Optional[SelectorTemperatureRestoreInfo]:
    """Resolve and apply selector temperature from experiment records."""
    record = resolve_selector_temperature_from_records(
        checkpoint_path=checkpoint_path,
        checkpoint_epoch=checkpoint_epoch,
        experiment_dir=experiment_dir,
    )
    if record is None:
        return None

    modules_updated = apply_selector_temperature(root_module, record.temperature)
    return SelectorTemperatureRestoreInfo(
        temperature=record.temperature,
        source=record.source,
        epoch=record.epoch,
        modules_updated=modules_updated,
    )
