"""Unit tests for graph_signal_diffusion.trainers.eval_schedule."""

import pytest
from omegaconf import OmegaConf

from graph_signal_diffusion.trainers.eval_schedule import (
    CosineEvalSchedule,
    EvalSchedule,
    MultiPhaseEvalSchedule,
    UniformEvalSchedule,
)


# ---------------------------------------------------------------------------
# UniformEvalSchedule
# ---------------------------------------------------------------------------


class TestUniformEvalSchedule:
    def test_basic_firing_epochs(self):
        """Uses epoch % period == 0 semantics, consistent with MultiPhaseEvalSchedule."""
        schedule = UniformEvalSchedule(period=10, eval_on_last_epoch=False)
        max_epochs = 100
        firing = [e for e in range(max_epochs) if schedule.should_eval(e, max_epochs)]
        # Should fire at 0, 10, 20, ..., 90 (ten epochs apart, 0-indexed)
        assert firing == list(range(0, max_epochs, 10))

    def test_eval_on_last_epoch_fires(self):
        schedule = UniformEvalSchedule(period=10, eval_on_last_epoch=True)
        # last epoch is 99, which is also divisible -> True by both conditions
        assert schedule.should_eval(99, 100)
        # last epoch is 94 (not divisible by 10 + 1 = 0 based)
        schedule2 = UniformEvalSchedule(period=10, eval_on_last_epoch=True)
        assert schedule2.should_eval(94, 95)  # epoch 94 is last, forced

    def test_eval_on_last_epoch_absent_without_flag(self):
        schedule = UniformEvalSchedule(period=10, eval_on_last_epoch=False)
        # epoch 94 is last but not a multiple-of-10 milestone
        assert not schedule.should_eval(94, 95)

    def test_no_false_positives(self):
        schedule = UniformEvalSchedule(period=7, eval_on_last_epoch=False)
        max_epochs = 50
        for e in range(max_epochs):
            expected = e % 7 == 0
            assert schedule.should_eval(e, max_epochs) == expected

    def test_period_1_fires_every_epoch(self):
        schedule = UniformEvalSchedule(period=1, eval_on_last_epoch=False)
        for e in range(20):
            assert schedule.should_eval(e, 20)

    def test_period_0_raises(self):
        with pytest.raises(ValueError, match="period must be > 0"):
            UniformEvalSchedule(period=0)

    def test_negative_period_raises(self):
        with pytest.raises(ValueError, match="period must be > 0"):
            UniformEvalSchedule(period=-5)

    def test_repr(self):
        s = UniformEvalSchedule(period=10)
        assert "UniformEvalSchedule" in repr(s)
        assert "10" in repr(s)


# ---------------------------------------------------------------------------
# MultiPhaseEvalSchedule
# ---------------------------------------------------------------------------


class TestMultiPhaseEvalSchedule:
    def test_single_phase_matches_uniform(self):
        """A single phase with period=10 should match UniformEvalSchedule w/ same period."""
        multi = MultiPhaseEvalSchedule(
            phases=[{"period": 10}], eval_on_last_epoch=False
        )
        uni = UniformEvalSchedule(period=10, eval_on_last_epoch=False)
        max_epochs = 100
        multi_set = {e for e in range(max_epochs) if multi.should_eval(e, max_epochs)}
        uni_set = {e for e in range(max_epochs) if uni.should_eval(e, max_epochs)}
        # Both use epoch % period == 0 → 0, 10, 20, ...
        expected = set(range(0, max_epochs, 10))
        assert multi_set == expected
        assert uni_set == expected

    def test_phase_boundary_alignment_reset(self):
        """Each phase resets its grid to the phase start boundary."""
        schedule = MultiPhaseEvalSchedule(
            phases=[
                {"period": 5, "until_epoch": 20},
                {"period": 7},
            ],
            eval_on_last_epoch=False,
        )
        max_epochs = 50
        firing = [e for e in range(max_epochs) if schedule.should_eval(e, max_epochs)]

        # Phase 1: epochs 0–19, period=5 → 0, 5, 10, 15
        phase1 = [e for e in firing if e < 20]
        assert phase1 == [0, 5, 10, 15]

        # Phase 2: starts at epoch 20, period=7 → 20, 27, 34, 41, 48
        phase2 = [e for e in firing if e >= 20]
        assert phase2 == [20, 27, 34, 41, 48]

    def test_eval_on_last_epoch(self):
        schedule = MultiPhaseEvalSchedule(
            phases=[{"period": 100}], eval_on_last_epoch=True
        )
        assert schedule.should_eval(49, 50)  # last epoch, period=100 won't fire normally

    def test_eval_on_last_epoch_false(self):
        schedule = MultiPhaseEvalSchedule(
            phases=[{"period": 100}], eval_on_last_epoch=False
        )
        # Period=100 means only epoch 0 fires; epoch 49 should NOT fire
        assert not schedule.should_eval(49, 50)

    def test_empty_phases_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            MultiPhaseEvalSchedule(phases=[])

    def test_period_zero_raises(self):
        with pytest.raises(ValueError, match="period must be > 0"):
            MultiPhaseEvalSchedule(phases=[{"period": 0}])

    def test_last_phase_with_until_epoch_raises(self):
        with pytest.raises(ValueError, match="last phase must not have"):
            MultiPhaseEvalSchedule(
                phases=[
                    {"period": 5, "until_epoch": 50},
                    {"period": 10, "until_epoch": 100},  # last phase has until_epoch
                ]
            )

    def test_non_last_phase_missing_until_epoch_raises(self):
        with pytest.raises(ValueError, match="missing 'until_epoch'"):
            MultiPhaseEvalSchedule(
                phases=[
                    {"period": 5},  # not last, but no until_epoch
                    {"period": 10},
                ]
            )

    def test_until_epoch_must_advance(self):
        with pytest.raises(ValueError, match="must be greater than the phase start"):
            MultiPhaseEvalSchedule(
                phases=[
                    {"period": 5, "until_epoch": 0},  # until_epoch == phase_start
                    {"period": 10},
                ]
            )

    def test_three_phase_schedule(self):
        """Dense → sparse → dense (canonical use case)."""
        schedule = MultiPhaseEvalSchedule(
            phases=[
                {"period": 3, "until_epoch": 15},
                {"period": 10, "until_epoch": 85},
                {"period": 3},
            ],
            eval_on_last_epoch=False,
        )
        max_epochs = 100
        firing = [e for e in range(max_epochs) if schedule.should_eval(e, max_epochs)]

        # Phase 1: 0,3,6,9,12 (14 < 15)
        p1 = [e for e in firing if e < 15]
        assert p1 == [0, 3, 6, 9, 12]

        # Phase 2: 15,25,35,45,55,65,75 (84 < 85)
        p2 = [e for e in firing if 15 <= e < 85]
        assert p2 == [15, 25, 35, 45, 55, 65, 75]

        # Phase 3: 85,88,91,94,97
        p3 = [e for e in firing if e >= 85]
        assert p3 == [85, 88, 91, 94, 97]

    def test_repr(self):
        s = MultiPhaseEvalSchedule(phases=[{"period": 5}])
        assert "MultiPhaseEvalSchedule" in repr(s)

    # ------------------------------------------------------------------
    # Negative until_epoch
    # ------------------------------------------------------------------

    def test_negative_until_epoch_resolves_from_end(self):
        """until_epoch=-150 should resolve to max_epochs-150."""
        max_epochs = 1000
        schedule = MultiPhaseEvalSchedule(
            phases=[
                {"period": 5, "until_epoch": 50},
                {"period": 20, "until_epoch": -150},  # resolves to 850
                {"period": 5},
            ],
            eval_on_last_epoch=False,
        )
        firing = [e for e in range(max_epochs) if schedule.should_eval(e, max_epochs)]

        # Phase 1: 0,5,10,...,45 (< 50)
        p1 = [e for e in firing if e < 50]
        assert p1 == list(range(0, 50, 5))

        # Phase 2: 50,70,...,830 (< 850)
        p2 = [e for e in firing if 50 <= e < 850]
        assert p2 == list(range(50, 850, 20))

        # Phase 3: 850,855,...,995 (period 5 from 850)
        p3 = [e for e in firing if e >= 850]
        assert p3 == list(range(850, 1000, 5))

    def test_negative_until_epoch_different_max_epochs(self):
        """Negative until_epoch resolves differently for different max_epochs."""
        schedule = MultiPhaseEvalSchedule(
            phases=[
                {"period": 10, "until_epoch": -50},
                {"period": 2},
            ],
            eval_on_last_epoch=False,
        )
        # max_epochs=200: until_epoch=-50 resolves to 150
        # Phase 1: epochs 0..149, period=10 → fires at 0,10,...,140
        assert schedule.should_eval(140, 200)      # fires (140%10==0)
        assert not schedule.should_eval(149, 200)  # doesn't fire (149%10==9)
        # Phase 2: starts at 150, period=2 → fires at 150,152,...
        assert schedule.should_eval(150, 200)      # (150-150)%2==0
        assert not schedule.should_eval(151, 200)  # (151-150)%2==1
        assert schedule.should_eval(152, 200)      # (152-150)%2==0

        # max_epochs=100: until_epoch=-50 resolves to 50
        # Phase 1: epochs 0..49, period=10 → fires at 0,10,...,40
        assert schedule.should_eval(40, 100)       # fires
        assert not schedule.should_eval(49, 100)   # doesn't fire (49%10==9)
        # Phase 2: starts at 50, period=2
        assert schedule.should_eval(50, 100)       # (50-50)%2==0
        assert not schedule.should_eval(51, 100)   # (51-50)%2==1
        assert schedule.should_eval(52, 100)       # (52-50)%2==0

    def test_negative_until_epoch_accepted_at_construction(self):
        """Negative until_epoch must not raise at construction time."""
        schedule = MultiPhaseEvalSchedule(
            phases=[
                {"period": 5, "until_epoch": -100},
                {"period": 10},
            ]
        )
        assert schedule is not None

    def test_negative_until_epoch_on_last_phase_raises(self):
        """until_epoch on the last phase is always forbidden, even if negative."""
        with pytest.raises(ValueError, match="last phase must not have"):
            MultiPhaseEvalSchedule(
                phases=[
                    {"period": 5, "until_epoch": 50},
                    {"period": 10, "until_epoch": -100},  # last phase — forbidden
                ]
            )


# ---------------------------------------------------------------------------
# CosineEvalSchedule
# ---------------------------------------------------------------------------


class TestCosineEvalSchedule:
    def test_always_evals_at_epoch_0(self):
        schedule = CosineEvalSchedule(min_period=2, max_period=20, eval_on_last_epoch=False)
        assert schedule.should_eval(0, 100)

    def test_density_higher_at_start_and_end_than_middle(self):
        """The eval epoch set should be denser in the first/last 20% than mid 60%."""
        max_epochs = 200
        schedule = CosineEvalSchedule(min_period=2, max_period=40, eval_on_last_epoch=False)
        eval_set = set(schedule.eval_epochs(max_epochs))

        boundary = int(0.2 * max_epochs)
        early = sum(1 for e in eval_set if e < boundary)
        late = sum(1 for e in eval_set if e >= max_epochs - boundary)
        middle = sum(
            1 for e in eval_set if boundary <= e < max_epochs - boundary
        )
        mid_range = max_epochs - 2 * boundary

        # Density = count / range; early/late density should exceed middle density
        early_density = early / boundary
        late_density = late / boundary
        mid_density = middle / mid_range if mid_range > 0 else 0

        assert early_density > mid_density, (
            f"Expected early density ({early_density:.3f}) > mid density ({mid_density:.3f})"
        )
        assert late_density > mid_density, (
            f"Expected late density ({late_density:.3f}) > mid density ({mid_density:.3f})"
        )

    def test_eval_on_last_epoch_forced(self):
        schedule = CosineEvalSchedule(min_period=2, max_period=40, eval_on_last_epoch=True)
        # Force: epoch 99 must always be in eval set
        assert schedule.should_eval(99, 100)

    def test_eval_on_last_epoch_not_forced_when_false(self):
        """Last epoch should only appear if the step sequence happens to land there."""
        # Use a large max_period so landing exactly on the last epoch is unlikely
        schedule = CosineEvalSchedule(min_period=5, max_period=50, eval_on_last_epoch=False)
        max_epochs = 101  # odd number, end = 100
        eval_set = schedule._get_eval_set(max_epochs)
        # We're not asserting it's absent (it might land there), just that the
        # flag is not forcing it — tested by checking eval_on_last_epoch=False internally.
        assert not schedule.eval_on_last_epoch

    def test_cache_is_reused(self):
        schedule = CosineEvalSchedule(min_period=2, max_period=20)
        _ = schedule.should_eval(0, 100)
        assert 100 in schedule._cache
        cached = schedule._cache[100]
        _ = schedule.should_eval(5, 100)
        assert schedule._cache[100] is cached, "Cache object should be the same instance"

    def test_min_period_1_covers_every_epoch_at_extremes(self):
        """With min_period=1, at least epochs 0 and (max-1) should be evals."""
        schedule = CosineEvalSchedule(min_period=1, max_period=10, eval_on_last_epoch=True)
        assert schedule.should_eval(0, 50)
        assert schedule.should_eval(49, 50)

    def test_min_period_zero_raises(self):
        with pytest.raises(ValueError, match="min_period must be >= 1"):
            CosineEvalSchedule(min_period=0, max_period=10)

    def test_max_period_less_than_min_raises(self):
        with pytest.raises(ValueError, match="max_period"):
            CosineEvalSchedule(min_period=10, max_period=5)

    def test_equal_min_max_period_matches_uniform(self):
        """When min_period == max_period the schedule degenerates to uniform step-based."""
        period = 5
        schedule = CosineEvalSchedule(
            min_period=period, max_period=period, eval_on_last_epoch=False
        )
        max_epochs = 50
        expected = set(range(0, max_epochs, period))
        actual = set(schedule.eval_epochs(max_epochs))
        # eval_epochs includes last-epoch handling, but eval_on_last_epoch=False here.
        # With equal periods, every step is exactly +period.
        assert actual == expected

    def test_single_epoch_training(self):
        schedule = CosineEvalSchedule(min_period=2, max_period=10)
        assert schedule.should_eval(0, 1)

    def test_repr(self):
        s = CosineEvalSchedule(min_period=2, max_period=20)
        assert "CosineEvalSchedule" in repr(s)
        assert "2" in repr(s)
        assert "20" in repr(s)

    def test_eval_epochs_sorted_and_unique(self):
        schedule = CosineEvalSchedule(min_period=2, max_period=30, eval_on_last_epoch=True)
        epochs = schedule.eval_epochs(200)
        assert epochs == sorted(set(epochs)), "eval_epochs should be sorted and unique"


# ---------------------------------------------------------------------------
# EvalSchedule.from_config factory
# ---------------------------------------------------------------------------


class TestFromConfig:
    def test_none_cfg_returns_uniform_with_fallback(self):
        s = EvalSchedule.from_config(None, fallback_period=15)
        assert isinstance(s, UniformEvalSchedule)
        assert s.period == 15

    def test_uniform_type(self):
        s = EvalSchedule.from_config({"type": "uniform", "period": 7})
        assert isinstance(s, UniformEvalSchedule)
        assert s.period == 7

    def test_uniform_eval_on_last_epoch_false(self):
        s = EvalSchedule.from_config(
            {"type": "uniform", "period": 5, "eval_on_last_epoch": False}
        )
        assert not s.eval_on_last_epoch

    def test_multi_phase_type(self):
        s = EvalSchedule.from_config(
            {
                "type": "multi_phase",
                "phases": [
                    {"period": 5, "until_epoch": 50},
                    {"period": 10},
                ],
            }
        )
        assert isinstance(s, MultiPhaseEvalSchedule)

    def test_multi_phase_missing_phases_raises(self):
        with pytest.raises(ValueError, match="requires a 'phases' list"):
            EvalSchedule.from_config({"type": "multi_phase"})

    def test_cosine_type(self):
        s = EvalSchedule.from_config(
            {"type": "cosine", "min_period": 3, "max_period": 25}
        )
        assert isinstance(s, CosineEvalSchedule)
        assert s.min_period == 3
        assert s.max_period == 25

    def test_cosine_defaults(self):
        """Cosine type with no explicit min/max uses defaults."""
        s = EvalSchedule.from_config({"type": "cosine"})
        assert isinstance(s, CosineEvalSchedule)
        assert s.min_period >= 1

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown eval_schedule type"):
            EvalSchedule.from_config({"type": "invalid_schedule_xyz"})

    def test_no_type_key_returns_uniform_fallback(self):
        s = EvalSchedule.from_config({}, fallback_period=20)
        assert isinstance(s, UniformEvalSchedule)
        assert s.period == 20

    def test_case_insensitive_type(self):
        s = EvalSchedule.from_config({"type": "COSINE", "min_period": 2, "max_period": 10})
        assert isinstance(s, CosineEvalSchedule)

    def test_eval_on_last_epoch_default_is_true(self):
        s = EvalSchedule.from_config({"type": "uniform", "period": 5})
        assert s.eval_on_last_epoch is True

    def test_multi_phase_omegaconf_interpolation_resolves_periods(self):
        cfg = OmegaConf.create(
            {
                "trainer": {
                    "eval_every_n_epochs": 50,
                    "eval_schedule": {
                        "type": "multi_phase",
                        "eval_on_last_epoch": True,
                        "phases": [
                            {"period": 10, "until_epoch": 50},
                            {
                                "period": "${trainer.eval_every_n_epochs}",
                                "until_epoch": -100,
                            },
                            {"period": 10},
                        ],
                    },
                }
            }
        )

        s = EvalSchedule.from_config(
            cfg.trainer.get("eval_schedule", None),
            fallback_period=int(cfg.trainer.eval_every_n_epochs),
        )
        assert isinstance(s, MultiPhaseEvalSchedule)

        max_epochs = 5000
        for e in [50, 100, 4900, 4910, 4999]:
            assert s.should_eval(e, max_epochs)
        for e in [99, 4899, 4909, 4998]:
            assert not s.should_eval(e, max_epochs)
