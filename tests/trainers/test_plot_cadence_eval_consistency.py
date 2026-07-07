"""Tests for the cadence consistency between the eval schedule and the
selector-diagnostic plot / heavy-collection cadences.

Bug context (2026-05-27): two sibling functions in ``DiffusionTrainer``
used a 1-indexed convention ``(epoch + 1) % period == 0`` while
``UniformEvalSchedule.should_eval`` (and the phased variant) uses 0-indexed
``epoch % period == 0``. Both were only invoked from inside the eval
branch, so the *effective* trigger was ``eval_epochs ∩ cadence_epochs``.

Affected functions:
  1. ``_should_plot_on_epoch`` (gates per-epoch plot rendering)
  2. ``_should_collect_heavy_selector_diagnostics`` (gates per-network
     row writes into ``selector_diagnostic_summaries.jsonl``, which the
     freq_evolution plot consumes)

For any pair where ``period_eval`` is a multiple of ``period_cadence`` (the
common case — eval every 50/100 epochs, plot fallback hardcoded to 5,
heavy-diag cadence 25), the intersection was empty for ``epoch > 0`` and
only the forced ``eval_on_last_epoch=True`` final-epoch trigger fired.

The fix aligns both with eval semantics: ``epoch > 0 and epoch % n == 0``.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from graph_signal_diffusion.trainers.eval_schedule import (
    MultiPhaseEvalSchedule,
    UniformEvalSchedule,
)
from graph_signal_diffusion.trainers.trainer import DiffusionTrainer


# ---------------------------------------------------------------------------
# Direct unit tests on _should_plot_on_epoch
# ---------------------------------------------------------------------------

def test_should_plot_on_epoch_skips_epoch_zero():
    """Plot cadence intentionally skips ep 0 to avoid rendering a single-row PDF."""
    assert DiffusionTrainer._should_plot_on_epoch(epoch=0, plot_every_n_epochs=5) is False


def test_should_plot_on_epoch_fires_on_first_multiple():
    """First plot lands on epoch == period (5), not 4."""
    assert DiffusionTrainer._should_plot_on_epoch(epoch=5, plot_every_n_epochs=5) is True
    # Pre-fix would have fired at 4: this test fails before the fix.
    assert DiffusionTrainer._should_plot_on_epoch(epoch=4, plot_every_n_epochs=5) is False


def test_should_plot_on_epoch_matches_eval_modular_arithmetic():
    """For any period, ``epoch % period == 0`` (excluding 0) ⇔ plot fires."""
    period = 5
    expected_plot_epochs = {n for n in range(1, 1000) if n % period == 0}
    actual_plot_epochs = {
        e for e in range(1000)
        if DiffusionTrainer._should_plot_on_epoch(epoch=e, plot_every_n_epochs=period)
    }
    assert actual_plot_epochs == expected_plot_epochs


def test_should_plot_on_epoch_disabled_when_period_is_none_or_nonpositive():
    """Backward compat: None / <=0 disables plotting."""
    assert DiffusionTrainer._should_plot_on_epoch(epoch=50, plot_every_n_epochs=None) is False
    assert DiffusionTrainer._should_plot_on_epoch(epoch=50, plot_every_n_epochs=0) is False
    assert DiffusionTrainer._should_plot_on_epoch(epoch=50, plot_every_n_epochs=-3) is False


# ---------------------------------------------------------------------------
# Cross-consistency with the eval schedule (the root cause of the bug)
# ---------------------------------------------------------------------------

def test_uniform_eval_period_50_plot_period_5_intersect_at_eval_epochs():
    """With eval every 50 epochs and plot every 5, every NATURAL eval epoch > 0
    should also be a plot epoch (since 50 is a multiple of 5).

    Pre-fix plot fired at {4, 9, 14, ...} which intersects {0, 50, 100, ...}
    only via the forced last-epoch eval. Post-fix the natural cadence aligns;
    the forced last-epoch eval (max_epochs - 1, often not on the plot grid) is
    handled separately by ``_plot_epoch_artifacts`` (last-epoch force), not by
    ``_should_plot_on_epoch``."""
    eval_sched = UniformEvalSchedule(period=50, eval_on_last_epoch=True)
    max_epochs = 1000

    eval_epochs = {e for e in range(max_epochs) if eval_sched.should_eval(e, max_epochs)}
    plot_epochs_at_5 = {
        e for e in range(max_epochs)
        if DiffusionTrainer._should_plot_on_epoch(epoch=e, plot_every_n_epochs=5)
    }
    # Natural eval epochs (excluding the forced last) should all be plot epochs.
    natural_eval_epochs = {e for e in eval_epochs if e > 0 and e != max_epochs - 1}
    missing = natural_eval_epochs - plot_epochs_at_5
    assert not missing, f"Natural eval epochs not triggering plot: {sorted(missing)[:10]}"


def test_phased_eval_and_plot_cadence_5_intersect_at_all_eval_epochs():
    """Same property for the canonical SP500 DS8 phased schedule
    `[{50, until:500}, {100}]` — every natural eval epoch > 0 should fire
    plotting. Forced last-epoch handled separately at the call site."""
    phases = [
        {"period": 50, "until_epoch": 500},
        {"period": 100, "until_epoch": None},
    ]
    eval_sched = MultiPhaseEvalSchedule(phases=phases, eval_on_last_epoch=True)
    max_epochs = 1000

    eval_epochs = {e for e in range(max_epochs) if eval_sched.should_eval(e, max_epochs)}
    plot_epochs_at_5 = {
        e for e in range(max_epochs)
        if DiffusionTrainer._should_plot_on_epoch(epoch=e, plot_every_n_epochs=5)
    }
    natural_eval_epochs = {e for e in eval_epochs if e > 0 and e != max_epochs - 1}
    missing = natural_eval_epochs - plot_epochs_at_5
    assert not missing, (
        f"Natural eval epochs not triggering plot in phased schedule (period=5): "
        f"{sorted(missing)[:10]}"
    )


def test_phased_eval_plot_cadence_count_is_reasonable():
    """Sanity check: with the SP500 phased eval + plot_period=5, we should get
    ~14 plot renders during training (every eval epoch except 0). Pre-fix would
    yield just 1 — the forced last-epoch render."""
    phases = [
        {"period": 50, "until_epoch": 500},
        {"period": 100, "until_epoch": None},
    ]
    eval_sched = MultiPhaseEvalSchedule(phases=phases, eval_on_last_epoch=True)
    max_epochs = 1000
    eval_epochs = [e for e in range(max_epochs) if eval_sched.should_eval(e, max_epochs)]
    # eval epochs (0-indexed): 0, 50, 100, ..., 500, 600, ..., 900, plus 999 forced
    # → 16 total; plot fires on the 15 positive ones (skipping 0).
    plot_fires = sum(
        1 for e in eval_epochs
        if e > 0 and DiffusionTrainer._should_plot_on_epoch(epoch=e, plot_every_n_epochs=5)
    )
    assert plot_fires >= 10, f"Expected ~14 plot renders, got {plot_fires}"


# ---------------------------------------------------------------------------
# Regression: pre-fix behavior would have left these failing
# ---------------------------------------------------------------------------

def test_eval_period_multiple_of_plot_period_general_property():
    """For any eval period that is a positive multiple of the plot period,
    every eval epoch > 0 should trigger plotting. This is the general property
    the bug fix establishes."""
    for plot_period in [2, 5, 10]:
        for multiplier in [1, 2, 5, 10, 20]:
            eval_period = plot_period * multiplier
            eval_sched = UniformEvalSchedule(period=eval_period, eval_on_last_epoch=False)
            for max_epochs in [100, 1000]:
                eval_epochs = {
                    e for e in range(max_epochs) if eval_sched.should_eval(e, max_epochs)
                }
                for e in eval_epochs:
                    if e == 0:
                        continue
                    assert DiffusionTrainer._should_plot_on_epoch(
                        epoch=e, plot_every_n_epochs=plot_period,
                    ), (
                        f"Inconsistency at eval_period={eval_period}, "
                        f"plot_period={plot_period}, epoch={e}"
                    )


# ---------------------------------------------------------------------------
# Sibling bug: _should_collect_heavy_selector_diagnostics had the same
# off-by-one. This gates per-network row writes to the diagnostic JSONL,
# which the freq_evolution plot consumes — when the gate never fires, the
# JSONL has zero network rows and the plot has nothing to render.
# ---------------------------------------------------------------------------


def _make_heavy_diag_stand_in(
    *,
    eval_schedule,
    max_epochs: int,
    diagnose_selector_every_n_epochs: int,
    diagnose_model_enabled: bool = True,
):
    """Build a SimpleNamespace with just the attributes
    ``_should_collect_heavy_selector_diagnostics`` accesses.

    Avoids instantiating a full ``DiffusionTrainer`` (which requires a
    diffusion model, optimizer, data loaders, etc.). We call the unbound
    method via ``.__func__(stand_in, ...)``.
    """
    return SimpleNamespace(
        diagnose_model_enabled=diagnose_model_enabled,
        diagnose_selector_every_n_epochs=diagnose_selector_every_n_epochs,
        max_epochs=max_epochs,
        eval_schedule=eval_schedule,
    )


def _should_collect(stand_in, epoch: int, *, is_last_epoch: bool = False) -> bool:
    return DiffusionTrainer._should_collect_heavy_selector_diagnostics(
        stand_in, epoch, is_last_epoch=is_last_epoch,
    )


# Direct unit tests on _should_collect_heavy_selector_diagnostics ------------

def test_heavy_diag_skips_epoch_zero():
    """Heavy diag intentionally skips ep 0 (no diagnostic value at step 0)."""
    sched = UniformEvalSchedule(period=25, eval_on_last_epoch=False)
    s = _make_heavy_diag_stand_in(
        eval_schedule=sched, max_epochs=1000,
        diagnose_selector_every_n_epochs=25,
    )
    assert _should_collect(s, epoch=0) is False


def test_heavy_diag_fires_on_first_natural_multiple():
    """First heavy diag at epoch == n (not n - 1, as the pre-fix did)."""
    sched = UniformEvalSchedule(period=25, eval_on_last_epoch=False)
    s = _make_heavy_diag_stand_in(
        eval_schedule=sched, max_epochs=1000,
        diagnose_selector_every_n_epochs=25,
    )
    assert _should_collect(s, epoch=25) is True
    # Pre-fix fired at 24: this assertion fails before the fix.
    assert _should_collect(s, epoch=24) is False


def test_heavy_diag_disabled_when_n_is_zero_or_negative():
    sched = UniformEvalSchedule(period=25, eval_on_last_epoch=False)
    for n in (0, -1, -3):
        s = _make_heavy_diag_stand_in(
            eval_schedule=sched, max_epochs=1000,
            diagnose_selector_every_n_epochs=n,
        )
        assert _should_collect(s, epoch=25) is False, f"n={n} should disable"


def test_heavy_diag_disabled_when_diagnose_model_disabled():
    """Global gate: diagnose_model_enabled=False blocks everything."""
    sched = UniformEvalSchedule(period=25, eval_on_last_epoch=False)
    s = _make_heavy_diag_stand_in(
        eval_schedule=sched, max_epochs=1000,
        diagnose_selector_every_n_epochs=25,
        diagnose_model_enabled=False,
    )
    assert _should_collect(s, epoch=25) is False
    # Even last-epoch force respects the global gate.
    assert _should_collect(s, epoch=999, is_last_epoch=True) is False


def test_heavy_diag_last_epoch_force_overrides_cadence():
    """is_last_epoch=True forces collection regardless of cadence (existing
    behavior at line 590-591; must remain intact after the fix)."""
    sched = UniformEvalSchedule(period=25, eval_on_last_epoch=False)
    s = _make_heavy_diag_stand_in(
        eval_schedule=sched, max_epochs=1000,
        diagnose_selector_every_n_epochs=25,
    )
    # Epoch 999 is not a multiple of 25 and not an eval epoch under period=25,
    # but is_last_epoch=True forces True.
    assert _should_collect(s, epoch=999, is_last_epoch=True) is True


def test_heavy_diag_skipped_when_not_eval_epoch():
    """Pass eval gate first: non-eval epochs never collect."""
    sched = UniformEvalSchedule(period=50, eval_on_last_epoch=False)
    s = _make_heavy_diag_stand_in(
        eval_schedule=sched, max_epochs=1000,
        diagnose_selector_every_n_epochs=25,
    )
    # epoch=25: multiple of 25 but NOT a multiple of 50 → eval doesn't fire.
    assert _should_collect(s, epoch=25) is False
    # epoch=50: both eval and heavy gate match → True.
    assert _should_collect(s, epoch=50) is True


# Cross-consistency: heavy diag fires on natural eval epochs -----------------

def test_heavy_diag_uniform_eval_25_n_25_fires_at_every_eval_epoch():
    """The carmine-moose case: eval=25, heavy n=25 → every eval epoch > 0
    should collect heavy diag (post-fix). Pre-fix this set was empty for
    eval_period == n (the most common config)."""
    sched = UniformEvalSchedule(period=25, eval_on_last_epoch=True)
    max_epochs = 1000
    s = _make_heavy_diag_stand_in(
        eval_schedule=sched, max_epochs=max_epochs,
        diagnose_selector_every_n_epochs=25,
    )
    eval_epochs = [e for e in range(max_epochs) if sched.should_eval(e, max_epochs)]
    natural_eval = [e for e in eval_epochs if e > 0 and e != max_epochs - 1]
    missing = [e for e in natural_eval if not _should_collect(s, epoch=e)]
    assert not missing, (
        f"Natural eval epochs not collecting heavy diag: {missing[:10]}"
    )


def test_heavy_diag_phased_eval_50_100_n_25_fires_at_every_eval_epoch():
    """The SP500 DS8 case: phased eval [{50, until:500}, {100}], heavy n=25.
    50 and 100 are both multiples of 25 → every natural eval epoch > 0 should
    fire heavy diag post-fix. Pre-fix only ep 999 fired (via is_last_epoch)."""
    sched = MultiPhaseEvalSchedule(
        phases=[
            {"period": 50, "until_epoch": 500},
            {"period": 100, "until_epoch": None},
        ],
        eval_on_last_epoch=True,
    )
    max_epochs = 1000
    s = _make_heavy_diag_stand_in(
        eval_schedule=sched, max_epochs=max_epochs,
        diagnose_selector_every_n_epochs=25,
    )
    eval_epochs = [e for e in range(max_epochs) if sched.should_eval(e, max_epochs)]
    natural_eval = [e for e in eval_epochs if e > 0 and e != max_epochs - 1]
    missing = [e for e in natural_eval if not _should_collect(s, epoch=e)]
    assert not missing, (
        f"Phased natural eval epochs not collecting heavy diag: {missing[:10]}"
    )


def test_heavy_diag_n_multiple_of_eval_general_property():
    """For any heavy cadence n that is a divisor of the eval period, every
    eval epoch > 0 should fire heavy diag (mirrors the plot-cadence
    consistency test)."""
    for eval_period in [25, 50, 100]:
        for n in [d for d in [1, 5, 10, 25, 50] if eval_period % d == 0]:
            sched = UniformEvalSchedule(period=eval_period, eval_on_last_epoch=False)
            for max_epochs in [200, 1000]:
                s = _make_heavy_diag_stand_in(
                    eval_schedule=sched, max_epochs=max_epochs,
                    diagnose_selector_every_n_epochs=n,
                )
                eval_epochs = [
                    e for e in range(max_epochs) if sched.should_eval(e, max_epochs)
                ]
                for e in eval_epochs:
                    if e == 0:
                        continue
                    assert _should_collect(s, epoch=e), (
                        f"Inconsistency at eval_period={eval_period}, n={n}, "
                        f"epoch={e}"
                    )
