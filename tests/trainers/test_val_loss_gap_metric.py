"""Regression tests for the derived ``val_loss_gap`` metric.

`_inject_val_loss_gap` is called from `DiffusionTrainer` after merging
the val and train-val per-epoch metrics. It surfaces
``|val_loss − train-val_loss|`` so that best-model trackers can favour
epochs with small generalization gaps (e.g. the
`val_loss + |gap| + α·CRPS` composite identified in the
amethyst-perch-521 / steel-goshawk-355 sweep analyses).

Covers:
  1. Both halves present → gap is absolute difference.
  2. Equal losses → gap is exactly 0.
  3. Either half missing → metric absent (no spurious zero).
  4. Non-finite (NaN/Inf) → metric absent.
  5. Non-numeric → metric absent (no exception).
  6. Idempotence — re-running on the same dict updates rather than appends.
"""
from __future__ import annotations

import math

import pytest

from graph_signal_diffusion.trainers.trainer import _inject_val_loss_gap


def test_basic_positive_gap():
    m = {"val_loss": 0.3504, "train-val_loss": 0.3507}
    _inject_val_loss_gap(m)
    assert m["val_loss_gap"] == pytest.approx(0.0003, abs=1e-9)


def test_basic_negative_signed_gap_takes_abs():
    """val_loss > train-val_loss is the typical overfit direction; verify
    abs() is taken regardless of which is larger."""
    m1 = {"val_loss": 0.40, "train-val_loss": 0.30}
    m2 = {"val_loss": 0.30, "train-val_loss": 0.40}
    _inject_val_loss_gap(m1)
    _inject_val_loss_gap(m2)
    assert m1["val_loss_gap"] == pytest.approx(0.10, abs=1e-9)
    assert m2["val_loss_gap"] == pytest.approx(0.10, abs=1e-9)


def test_equal_losses_yield_zero_gap():
    m = {"val_loss": 0.35, "train-val_loss": 0.35}
    _inject_val_loss_gap(m)
    assert m["val_loss_gap"] == 0.0


def test_missing_val_loss_is_noop():
    m = {"train-val_loss": 0.30}
    _inject_val_loss_gap(m)
    assert "val_loss_gap" not in m


def test_missing_train_val_loss_is_noop():
    """Eval schedule sometimes skips train-val on a given epoch — make sure
    we don't inject a misleading zero in that case."""
    m = {"val_loss": 0.35}
    _inject_val_loss_gap(m)
    assert "val_loss_gap" not in m


def test_neither_half_present_is_noop():
    m = {"train_loss": 0.40, "train_lr": 1e-4}  # neither key present
    before = dict(m)
    _inject_val_loss_gap(m)
    assert m == before


@pytest.mark.parametrize("val,tv", [
    (float("nan"), 0.30),
    (0.30, float("nan")),
    (float("inf"), 0.30),
    (0.30, float("-inf")),
])
def test_non_finite_inputs_skipped(val, tv):
    m = {"val_loss": val, "train-val_loss": tv}
    _inject_val_loss_gap(m)
    assert "val_loss_gap" not in m, (
        f"Non-finite input ({val}, {tv}) should not yield a stored gap"
    )


def test_non_numeric_inputs_skipped():
    m = {"val_loss": "0.35", "train-val_loss": None}
    _inject_val_loss_gap(m)
    assert "val_loss_gap" not in m


def test_accepts_string_numerics_via_float_coerce():
    """If upstream code happens to stringify floats, the coercion should
    succeed rather than raising."""
    m = {"val_loss": "0.35", "train-val_loss": "0.30"}
    _inject_val_loss_gap(m)
    assert m["val_loss_gap"] == pytest.approx(0.05, abs=1e-9)


def test_idempotent_updates_in_place():
    m = {"val_loss": 0.40, "train-val_loss": 0.30}
    _inject_val_loss_gap(m)
    assert m["val_loss_gap"] == pytest.approx(0.10, abs=1e-9)
    # Update upstream value, re-inject — value should update, not duplicate.
    m["val_loss"] = 0.35
    _inject_val_loss_gap(m)
    assert m["val_loss_gap"] == pytest.approx(0.05, abs=1e-9)
    assert sum(1 for k in m if k == "val_loss_gap") == 1


def test_does_not_modify_unrelated_keys():
    m = {"val_loss": 0.35, "train-val_loss": 0.30, "train_loss": 0.34,
         "val_return_crps": 1.10}
    _inject_val_loss_gap(m)
    assert m["train_loss"] == 0.34
    assert m["val_return_crps"] == 1.10
    assert m["val_loss"] == 0.35
    assert m["train-val_loss"] == 0.30


def test_finite_check_via_math_isfinite():
    """Sanity: math.isfinite is the gate, not numpy."""
    assert math.isfinite(0.5)
    assert not math.isfinite(float("nan"))
    assert not math.isfinite(float("inf"))
