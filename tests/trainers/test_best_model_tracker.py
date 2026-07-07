"""Unit tests for BestModelTracker."""

import json
import os
import tempfile

import pytest

from graph_signal_diffusion.trainers.best_model_tracker import (
    BestModelTracker,
    MetricSpec,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_specs():
    return [
        MetricSpec(name="val_rate_mean_violation_gap_pct_generated", weight=0.5, direction="minimize"),
        MetricSpec(name="val_rate_1pct_gap_pct", weight=0.25, direction="minimize"),
        MetricSpec(name="val_rate_5pct_gap_pct", weight=0.25, direction="minimize"),
    ]


@pytest.fixture
def tracker(default_specs):
    return BestModelTracker(
        metrics=default_specs,
        ema_alpha=0.3,
        top_k=5,
        min_warmup_evals=3,
    )


def _make_metrics(violation_gap_pct: float, rate_1pct: float, rate_5pct: float):
    """Helper to build a raw metrics dict."""
    return {
        "val_rate_mean_violation_gap_pct_generated": violation_gap_pct,
        "val_rate_1pct_gap_pct": rate_1pct,
        "val_rate_5pct_gap_pct": rate_5pct,
    }


# ---------------------------------------------------------------------------
# MetricSpec validation
# ---------------------------------------------------------------------------


class TestMetricSpec:

    def test_valid(self):
        spec = MetricSpec(name="foo", weight=1.0, direction="minimize")
        assert spec.name == "foo"

    def test_invalid_direction(self):
        with pytest.raises(ValueError, match="direction"):
            MetricSpec(name="foo", weight=1.0, direction="decreasing")

    def test_negative_weight(self):
        with pytest.raises(ValueError, match="weight"):
            MetricSpec(name="foo", weight=-1.0, direction="minimize")


# ---------------------------------------------------------------------------
# EMA computation
# ---------------------------------------------------------------------------


class TestEMAComputation:

    def test_first_observation_sets_ema(self, tracker):
        raw = _make_metrics(50.0, 80.0, 60.0)
        tracker.update(0, raw)
        ema = tracker.get_ema_metrics()
        assert ema["val_rate_mean_violation_gap_pct_generated"] == pytest.approx(50.0)
        assert ema["val_rate_1pct_gap_pct"] == pytest.approx(80.0)
        assert ema["val_rate_5pct_gap_pct"] == pytest.approx(60.0)

    def test_ema_decay(self, tracker):
        alpha = 0.3
        # First observation
        tracker.update(0, _make_metrics(100.0, 100.0, 100.0))
        # Second observation
        tracker.update(1, _make_metrics(0.0, 0.0, 0.0))
        ema = tracker.get_ema_metrics()
        expected = alpha * 0.0 + (1 - alpha) * 100.0  # 70.0
        assert ema["val_rate_mean_violation_gap_pct_generated"] == pytest.approx(expected)

    def test_ema_converges(self, tracker):
        """Feed constant values — EMA should converge to that value."""
        for i in range(50):
            tracker.update(i, _make_metrics(42.0, 42.0, 42.0))
        ema = tracker.get_ema_metrics()
        for v in ema.values():
            assert v == pytest.approx(42.0, abs=0.01)


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------


class TestCompositeScore:

    def test_composite_weighted_sum(self, default_specs):
        tracker = BestModelTracker(
            metrics=default_specs, ema_alpha=1.0, top_k=5, min_warmup_evals=0
        )
        raw = _make_metrics(20.0, 80.0, 40.0)
        score = tracker.update(0, raw)
        # Weights: 0.5, 0.25, 0.25 (already normalized)
        # EMA with alpha=1 means EMA == raw value
        expected = 0.5 * 20.0 + 0.25 * 80.0 + 0.25 * 40.0  # 10 + 20 + 10 = 40
        assert score == pytest.approx(expected)

    def test_returns_none_during_warmup(self, tracker):
        for i in range(2):
            score = tracker.update(i, _make_metrics(50.0, 50.0, 50.0))
            assert score is None  # warmup = 3
        score = tracker.update(2, _make_metrics(50.0, 50.0, 50.0))
        assert score is not None

    def test_missing_metric_returns_none(self, default_specs):
        tracker = BestModelTracker(
            metrics=default_specs, ema_alpha=1.0, top_k=5, min_warmup_evals=0,
            missing_metric_grace_evals=5,
        )
        # Only provide 2 of 3 metrics
        score = tracker.update(0, {
            "val_rate_mean_violation_gap_pct_generated": 20.0,
            "val_rate_1pct_gap_pct": 80.0,
        })
        assert score is None  # rate_5pct_gap_pct never observed, still within grace

    def test_raw_composite_score_uses_current_epoch_metrics(self, default_specs):
        tracker = BestModelTracker(
            metrics=default_specs, ema_alpha=0.3, top_k=5, min_warmup_evals=3
        )
        raw = _make_metrics(20.0, 80.0, 40.0)
        score = tracker.raw_composite_score(raw)
        expected = 0.5 * 20.0 + 0.25 * 80.0 + 0.25 * 40.0  # 40.0
        assert score == pytest.approx(expected)

    def test_raw_composite_score_renormalizes_when_metrics_missing(self, default_specs):
        tracker = BestModelTracker(
            metrics=default_specs, ema_alpha=0.3, top_k=5, min_warmup_evals=3
        )
        raw_partial = {
            "val_rate_mean_violation_gap_pct_generated": 20.0,
            "val_rate_1pct_gap_pct": 80.0,
            # val_rate_5pct_gap_pct missing
        }
        # Weighted partial sum = 0.5*20 + 0.25*80 = 30 with observed weight 0.75
        # Renormalized score = 30 / 0.75 = 40
        score = tracker.raw_composite_score(raw_partial)
        assert score == pytest.approx(40.0)


# ---------------------------------------------------------------------------
# Missing-metric grace period fallback
# ---------------------------------------------------------------------------


class TestMissingMetricFallback:
    """After ``missing_metric_grace_evals``, composite score falls back to observed metrics."""

    def test_returns_none_before_grace(self):
        specs = [
            MetricSpec(name="a", weight=0.5, direction="minimize"),
            MetricSpec(name="b", weight=0.5, direction="minimize"),
        ]
        tracker = BestModelTracker(
            metrics=specs, ema_alpha=1.0, top_k=5,
            min_warmup_evals=0, missing_metric_grace_evals=3,
        )
        # Provide only "a" for 2 updates (< grace=3)
        for i in range(2):
            score = tracker.update(i, {"a": 10.0})
            assert score is None

    def test_falls_back_after_grace(self):
        specs = [
            MetricSpec(name="a", weight=0.6, direction="minimize"),
            MetricSpec(name="b", weight=0.4, direction="minimize"),
        ]
        tracker = BestModelTracker(
            metrics=specs, ema_alpha=1.0, top_k=5,
            min_warmup_evals=0, missing_metric_grace_evals=3,
        )
        # Provide only "a" for 3+ updates (>= grace=3)
        for i in range(3):
            score = tracker.update(i, {"a": 50.0})

        # After grace, "b" is missing → fallback re-normalizes weights.
        # Only "a" observed with weight 0.6/(0.6) = 1.0, so score = 50.0
        assert score is not None
        assert score == pytest.approx(50.0)

    def test_fallback_renormalizes_weights(self):
        specs = [
            MetricSpec(name="a", weight=1.0, direction="minimize"),
            MetricSpec(name="b", weight=1.0, direction="minimize"),
            MetricSpec(name="c", weight=1.0, direction="minimize"),
        ]
        tracker = BestModelTracker(
            metrics=specs, ema_alpha=1.0, top_k=5,
            min_warmup_evals=0, missing_metric_grace_evals=0,  # immediate fallback
        )
        # Provide "a" and "b" but not "c"
        score = tracker.update(0, {"a": 30.0, "b": 60.0})
        # Norm weights are 1/3 each. Observed weight = 2/3.
        # Raw sum = (1/3)*30 + (1/3)*60 = 30.  Re-normalized: 30 / (2/3) = 45.
        assert score == pytest.approx(45.0)

    def test_fallback_with_maximize(self):
        specs = [
            MetricSpec(name="error", weight=0.5, direction="minimize"),
            MetricSpec(name="accuracy", weight=0.5, direction="maximize"),
        ]
        tracker = BestModelTracker(
            metrics=specs, ema_alpha=1.0, top_k=5,
            min_warmup_evals=0, missing_metric_grace_evals=0,
        )
        # Only "accuracy" observed
        score = tracker.update(0, {"accuracy": 90.0})
        # weight 0.5, direction=maximize → -90 * 0.5 = -45 / 0.5 = -90
        assert score == pytest.approx(-90.0)

    def test_all_missing_returns_none(self):
        specs = [MetricSpec(name="a", weight=1.0, direction="minimize")]
        tracker = BestModelTracker(
            metrics=specs, ema_alpha=1.0, top_k=5,
            min_warmup_evals=0, missing_metric_grace_evals=0,
        )
        # No metrics provided at all
        score = tracker.update(0, {"unrelated": 42.0})
        assert score is None

    def test_fallback_logs_once(self, caplog):
        """Warning about missing-metric fallback should only appear once."""
        import logging
        specs = [
            MetricSpec(name="a", weight=1.0, direction="minimize"),
            MetricSpec(name="b", weight=1.0, direction="minimize"),
        ]
        tracker = BestModelTracker(
            metrics=specs, ema_alpha=1.0, top_k=5,
            min_warmup_evals=0, missing_metric_grace_evals=0,
        )
        with caplog.at_level(logging.WARNING):
            tracker.update(0, {"a": 10.0})
            tracker.update(1, {"a": 10.0})
            tracker.update(2, {"a": 10.0})

        warning_msgs = [r.message for r in caplog.records if "never observed" in r.message]
        assert len(warning_msgs) == 1  # logged exactly once

    def test_all_observed_ignores_grace(self):
        """When all metrics are present, grace period is irrelevant."""
        specs = [
            MetricSpec(name="a", weight=1.0, direction="minimize"),
            MetricSpec(name="b", weight=1.0, direction="minimize"),
        ]
        tracker = BestModelTracker(
            metrics=specs, ema_alpha=1.0, top_k=5,
            min_warmup_evals=0, missing_metric_grace_evals=999,
        )
        score = tracker.update(0, {"a": 20.0, "b": 40.0})
        assert score is not None
        assert score == pytest.approx(30.0)  # (0.5*20 + 0.5*40)


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------


class TestWarmup:

    def test_no_save_during_warmup(self, tracker):
        for i in range(tracker.min_warmup_evals - 1):
            score = tracker.update(i, _make_metrics(10.0, 10.0, 10.0))
            assert not tracker.should_save(score)
        # At warmup boundary
        score = tracker.update(tracker.min_warmup_evals - 1, _make_metrics(10.0, 10.0, 10.0))
        assert tracker.should_save(score)

    def test_warmed_up_property(self, tracker):
        assert not tracker.warmed_up
        for i in range(tracker.min_warmup_evals):
            tracker.update(i, _make_metrics(50.0, 50.0, 50.0))
        assert tracker.warmed_up


# ---------------------------------------------------------------------------
# Top-K eviction
# ---------------------------------------------------------------------------


class TestTopKEviction:

    def test_keeps_top_k(self, default_specs):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = BestModelTracker(
                metrics=default_specs, ema_alpha=1.0, top_k=3, min_warmup_evals=0
            )
            paths = []
            for i in range(5):
                score_val = 50 - i * 5  # 50, 45, 40, 35, 30 (lower=better)
                score = tracker.update(i, _make_metrics(score_val, score_val, score_val))
                path = os.path.join(tmpdir, f"model_{i}.pt")
                open(path, "w").close()  # create empty file
                paths.append(path)
                tracker.register_checkpoint(i, score, path)

            assert tracker.leaderboard_size == 3
            # Best 3 scores: 30.0, 35.0, 40.0 (entries 4, 3, 2)
            assert tracker.best_score == pytest.approx(30.0)
            # Evicted files should be deleted
            assert not os.path.exists(paths[0])  # score 50
            assert not os.path.exists(paths[1])  # score 45
            assert os.path.exists(paths[2])       # score 40
            assert os.path.exists(paths[3])       # score 35
            assert os.path.exists(paths[4])       # score 30

    def test_duplicate_scores_handled(self, default_specs):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = BestModelTracker(
                metrics=default_specs, ema_alpha=1.0, top_k=2, min_warmup_evals=0
            )
            for i in range(3):
                score = tracker.update(i, _make_metrics(50.0, 50.0, 50.0))
                path = os.path.join(tmpdir, f"model_{i}.pt")
                open(path, "w").close()
                tracker.register_checkpoint(i, score, path)
            assert tracker.leaderboard_size == 2

    def test_should_save_when_not_full(self, default_specs):
        tracker = BestModelTracker(
            metrics=default_specs, ema_alpha=1.0, top_k=5, min_warmup_evals=0
        )
        score = tracker.update(0, _make_metrics(99.0, 99.0, 99.0))
        assert tracker.should_save(score)

    def test_should_not_save_when_worse_than_all(self, default_specs):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = BestModelTracker(
                metrics=default_specs, ema_alpha=1.0, top_k=2, min_warmup_evals=0
            )
            for i in range(2):
                score_val = 10.0 + i
                score = tracker.update(i, _make_metrics(score_val, score_val, score_val))
                path = os.path.join(tmpdir, f"model_{i}.pt")
                open(path, "w").close()
                tracker.register_checkpoint(i, score, path)
            # Now try a worse score
            worse_score = tracker.update(2, _make_metrics(99.0, 99.0, 99.0))
            assert not tracker.should_save(worse_score)


# ---------------------------------------------------------------------------
# Symlink
# ---------------------------------------------------------------------------


class TestSymlink:

    def test_creates_symlink(self, default_specs):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = BestModelTracker(
                metrics=default_specs, ema_alpha=1.0, top_k=5, min_warmup_evals=0
            )
            ckpt_path = os.path.join(tmpdir, "best_model_epoch_100.pt")
            open(ckpt_path, "w").close()
            score = tracker.update(0, _make_metrics(30.0, 30.0, 30.0))
            tracker.register_checkpoint(0, score, ckpt_path)
            tracker.update_symlink(tmpdir)

            link = os.path.join(tmpdir, "best_model.pt")
            assert os.path.islink(link)
            assert os.readlink(link) == "best_model_epoch_100.pt"

    def test_updates_symlink(self, default_specs):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = BestModelTracker(
                metrics=default_specs, ema_alpha=1.0, top_k=5, min_warmup_evals=0
            )
            # First checkpoint
            p1 = os.path.join(tmpdir, "model_1.pt")
            open(p1, "w").close()
            s1 = tracker.update(0, _make_metrics(50.0, 50.0, 50.0))
            tracker.register_checkpoint(0, s1, p1)
            tracker.update_symlink(tmpdir)

            # Better checkpoint
            p2 = os.path.join(tmpdir, "model_2.pt")
            open(p2, "w").close()
            s2 = tracker.update(1, _make_metrics(10.0, 10.0, 10.0))
            tracker.register_checkpoint(1, s2, p2)
            tracker.update_symlink(tmpdir)

            link = os.path.join(tmpdir, "best_model.pt")
            assert os.readlink(link) == "model_2.pt"


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


class TestManifest:

    def test_manifest_schema(self, default_specs):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = BestModelTracker(
                metrics=default_specs, ema_alpha=1.0, top_k=5, min_warmup_evals=0
            )
            ckpt_path = os.path.join(tmpdir, "model_0.pt")
            open(ckpt_path, "w").close()
            score = tracker.update(0, _make_metrics(30.0, 40.0, 35.0))
            tracker.register_checkpoint(
                0, score, ckpt_path,
                raw_metrics=_make_metrics(30.0, 40.0, 35.0),
            )
            manifest_path = tracker.save_manifest(tmpdir)

            assert os.path.exists(manifest_path)
            with open(manifest_path) as f:
                data = json.load(f)

            assert "config" in data
            assert data["config"]["top_k"] == 5
            assert data["config"]["ema_alpha"] == 1.0
            assert len(data["config"]["metrics"]) == 3

            assert "leaderboard" in data
            assert len(data["leaderboard"]) == 1
            entry = data["leaderboard"][0]
            assert entry["rank"] == 1
            assert entry["epoch"] == 0
            assert "composite_score" in entry
            assert "raw_metrics" in entry
            assert "ema_metrics" in entry

            assert data["total_evaluations"] == 1
            assert data["last_updated_epoch"] == 0

    def test_manifest_leaderboard_ordered(self, default_specs):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = BestModelTracker(
                metrics=default_specs, ema_alpha=1.0, top_k=5, min_warmup_evals=0
            )
            for i in range(3):
                val = 30.0 - i * 5  # 30, 25, 20 → scores decrease (improve)
                score = tracker.update(i, _make_metrics(val, val, val))
                p = os.path.join(tmpdir, f"model_{i}.pt")
                open(p, "w").close()
                tracker.register_checkpoint(i, score, p)

            manifest_path = tracker.save_manifest(tmpdir)
            with open(manifest_path) as f:
                data = json.load(f)

            scores = [e["composite_score"] for e in data["leaderboard"]]
            assert scores == sorted(scores)  # best (lowest) first


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestFromConfig:

    def test_from_config(self):
        cfg = {
            "enabled": True,
            "top_k": 3,
            "ema_alpha": 0.2,
            "min_warmup_evals": 5,
            "metrics": [
                {"name": "metric_a", "weight": 1.0, "direction": "minimize"},
                {"name": "metric_b", "weight": 2.0, "direction": "minimize"},
            ],
        }
        tracker = BestModelTracker.from_config(cfg)
        assert tracker.top_k == 3
        assert tracker.ema_alpha == pytest.approx(0.2)
        assert tracker.min_warmup_evals == 5
        assert len(tracker.metrics) == 2

    def test_from_config_missing_metrics(self):
        with pytest.raises(ValueError, match="metrics"):
            BestModelTracker.from_config({"metrics": []})

    def test_from_config_defaults(self):
        cfg = {
            "metrics": [{"name": "m1"}],
        }
        tracker = BestModelTracker.from_config(cfg)
        assert tracker.top_k == 5
        assert tracker.ema_alpha == pytest.approx(0.3)
        assert tracker.min_warmup_evals == 3


# ---------------------------------------------------------------------------
# Direction: maximize
# ---------------------------------------------------------------------------


class TestMaximizeDirection:

    def test_maximize_negates(self):
        specs = [MetricSpec(name="accuracy", weight=1.0, direction="maximize")]
        tracker = BestModelTracker(
            metrics=specs, ema_alpha=1.0, top_k=5, min_warmup_evals=0
        )
        # Higher accuracy = better → lower composite = better (negated)
        s1 = tracker.update(0, {"accuracy": 90.0})
        assert s1 == pytest.approx(-90.0)
        s2 = tracker.update(1, {"accuracy": 95.0})
        assert s2 == pytest.approx(-95.0)
        # s2 < s1 → 95% accuracy is "better" in composite space
        assert s2 < s1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:

    def test_constructor_validations(self, default_specs):
        with pytest.raises(ValueError, match="MetricSpec"):
            BestModelTracker(metrics=[])

        with pytest.raises(ValueError, match="ema_alpha"):
            BestModelTracker(metrics=default_specs, ema_alpha=0.0)

        with pytest.raises(ValueError, match="top_k"):
            BestModelTracker(metrics=default_specs, top_k=0)

        with pytest.raises(ValueError, match="min_warmup_evals"):
            BestModelTracker(metrics=default_specs, min_warmup_evals=-1)

    def test_should_save_none(self, tracker):
        assert not tracker.should_save(None)

    def test_properties_empty(self, tracker):
        assert tracker.best_score is None
        assert tracker.best_epoch is None
        assert tracker.best_path is None
        assert tracker.leaderboard_size == 0


# ---------------------------------------------------------------------------
# SP500 Stock Price Forecasting Integration
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Subdataset aggregation
# ---------------------------------------------------------------------------


class TestSubdatasetAggregation:
    """Test per-subdataset composite score aggregation."""

    @pytest.fixture
    def wra_specs(self):
        return [
            MetricSpec(name="val_rate_mean_violation_gap_pct_generated", weight=0.5, direction="minimize"),
            MetricSpec(name="val_rate_1pct_gap_pct", weight=0.25, direction="minimize"),
            MetricSpec(name="val_rate_5pct_gap_pct", weight=0.25, direction="minimize"),
        ]

    def _make_subdataset_metrics(
        self, subdatasets: dict[str, tuple[float, float, float]]
    ) -> dict[str, float]:
        """Build epoch_metrics with per-subdataset keys.

        subdatasets: {name: (violation_gap, 1p_gap, 5p_gap)}
        """
        metrics: dict[str, float] = {}
        for name, (viol, p1, p5) in subdatasets.items():
            prefix = f"val_subdataset/{name}/"
            metrics[f"{prefix}rate_mean_violation_gap_pct_generated"] = viol
            metrics[f"{prefix}rate_1pct_gap_pct"] = p1
            metrics[f"{prefix}rate_5pct_gap_pct"] = p5
        return metrics

    def test_mean_aggregation(self, wra_specs):
        tracker = BestModelTracker(
            metrics=wra_specs, ema_alpha=1.0, top_k=5,
            min_warmup_evals=0, subdataset_aggregation="mean",
        )
        # Two subdatasets with different difficulties
        raw = self._make_subdataset_metrics({
            "ds_easy": (2.0, 10.0, 5.0),   # composite = 0.5*2 + 0.25*10 + 0.25*5 = 4.75
            "ds_hard": (8.0, 30.0, 20.0),   # composite = 0.5*8 + 0.25*30 + 0.25*20 = 16.5
        })
        score = tracker.update(0, raw)
        expected = (4.75 + 16.5) / 2  # 10.625
        assert score == pytest.approx(expected)

    def test_worst_aggregation(self, wra_specs):
        tracker = BestModelTracker(
            metrics=wra_specs, ema_alpha=1.0, top_k=5,
            min_warmup_evals=0, subdataset_aggregation="worst",
        )
        raw = self._make_subdataset_metrics({
            "ds_easy": (2.0, 10.0, 5.0),   # composite = 4.75
            "ds_hard": (8.0, 30.0, 20.0),   # composite = 16.5
        })
        score = tracker.update(0, raw)
        expected = 16.5  # worst (highest)
        assert score == pytest.approx(expected)

    def test_global_ignores_subdataset_keys(self, wra_specs):
        tracker = BestModelTracker(
            metrics=wra_specs, ema_alpha=1.0, top_k=5,
            min_warmup_evals=0, subdataset_aggregation="global",
        )
        # Provide both global and subdataset keys
        raw = {
            "val_rate_mean_violation_gap_pct_generated": 5.0,
            "val_rate_1pct_gap_pct": 20.0,
            "val_rate_5pct_gap_pct": 10.0,
            "val_subdataset/ds_easy/rate_mean_violation_gap_pct_generated": 2.0,
            "val_subdataset/ds_easy/rate_1pct_gap_pct": 10.0,
            "val_subdataset/ds_easy/rate_5pct_gap_pct": 5.0,
        }
        score = tracker.update(0, raw)
        # Should use global keys only
        expected = 0.5 * 5.0 + 0.25 * 20.0 + 0.25 * 10.0  # 10.0
        assert score == pytest.approx(expected)

    def test_mean_falls_back_to_global_when_no_subdatasets(self, wra_specs):
        """Single-dataset training: no subdataset/ keys exist."""
        tracker = BestModelTracker(
            metrics=wra_specs, ema_alpha=1.0, top_k=5,
            min_warmup_evals=0, subdataset_aggregation="mean",
        )
        raw = _make_metrics(5.0, 20.0, 10.0)
        score = tracker.update(0, raw)
        expected = 0.5 * 5.0 + 0.25 * 20.0 + 0.25 * 10.0  # 10.0
        assert score == pytest.approx(expected)

    def test_ema_smoothing_on_aggregated_composite(self, wra_specs):
        tracker = BestModelTracker(
            metrics=wra_specs, ema_alpha=0.5, top_k=5,
            min_warmup_evals=0, subdataset_aggregation="mean",
        )
        # Epoch 0: composite = (4.75 + 16.5) / 2 = 10.625
        raw0 = self._make_subdataset_metrics({
            "ds_easy": (2.0, 10.0, 5.0),
            "ds_hard": (8.0, 30.0, 20.0),
        })
        score0 = tracker.update(0, raw0)
        assert score0 == pytest.approx(10.625)

        # Epoch 1: composite = (4.75 + 4.75) / 2 = 4.75
        raw1 = self._make_subdataset_metrics({
            "ds_easy": (2.0, 10.0, 5.0),
            "ds_hard": (2.0, 10.0, 5.0),
        })
        score1 = tracker.update(1, raw1)
        # EMA: 0.5 * 4.75 + 0.5 * 10.625 = 7.6875
        assert score1 == pytest.approx(7.6875)

    def test_raw_composite_score_with_mean(self, wra_specs):
        tracker = BestModelTracker(
            metrics=wra_specs, ema_alpha=1.0, top_k=5,
            min_warmup_evals=0, subdataset_aggregation="mean",
        )
        raw = self._make_subdataset_metrics({
            "ds_easy": (2.0, 10.0, 5.0),   # 4.75
            "ds_hard": (8.0, 30.0, 20.0),   # 16.5
        })
        score = tracker.raw_composite_score(raw)
        assert score == pytest.approx(10.625)

    def test_raw_composite_score_with_worst(self, wra_specs):
        tracker = BestModelTracker(
            metrics=wra_specs, ema_alpha=1.0, top_k=5,
            min_warmup_evals=0, subdataset_aggregation="worst",
        )
        raw = self._make_subdataset_metrics({
            "ds_easy": (2.0, 10.0, 5.0),   # 4.75
            "ds_hard": (8.0, 30.0, 20.0),   # 16.5
        })
        score = tracker.raw_composite_score(raw)
        assert score == pytest.approx(16.5)

    def test_invalid_aggregation_mode_raises(self, wra_specs):
        with pytest.raises(ValueError, match="subdataset_aggregation"):
            BestModelTracker(
                metrics=wra_specs, subdataset_aggregation="invalid",
            )

    def test_from_config_with_subdataset_aggregation(self):
        cfg = {
            "metrics": [{"name": "val_loss", "weight": 1.0, "direction": "minimize"}],
            "subdataset_aggregation": "worst",
        }
        tracker = BestModelTracker.from_config(cfg)
        assert tracker.subdataset_aggregation == "worst"

    def test_from_config_default_is_global(self):
        cfg = {
            "metrics": [{"name": "val_loss", "weight": 1.0, "direction": "minimize"}],
        }
        tracker = BestModelTracker.from_config(cfg)
        assert tracker.subdataset_aggregation == "global"

    def test_manifest_includes_aggregation_mode(self, wra_specs):
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = BestModelTracker(
                metrics=wra_specs, ema_alpha=1.0, top_k=5,
                min_warmup_evals=0, subdataset_aggregation="mean",
            )
            raw = self._make_subdataset_metrics({
                "ds_a": (5.0, 20.0, 10.0),
            })
            score = tracker.update(0, raw)
            path = os.path.join(tmpdir, "model_0.pt")
            open(path, "w").close()
            tracker.register_checkpoint(0, score, path)
            manifest_path = tracker.save_manifest(tmpdir)
            with open(manifest_path) as f:
                data = json.load(f)
            assert data["config"]["subdataset_aggregation"] == "mean"

    def test_grace_period_enforced_in_subdataset_mode(self, wra_specs):
        """Before grace period, missing metrics in a subdataset should return None."""
        tracker = BestModelTracker(
            metrics=wra_specs, ema_alpha=1.0, top_k=5,
            min_warmup_evals=0, missing_metric_grace_evals=3,
            subdataset_aggregation="mean",
        )
        # Provide only 2 of 3 metrics for each subdataset
        raw = {
            "val_subdataset/ds_a/rate_mean_violation_gap_pct_generated": 5.0,
            "val_subdataset/ds_a/rate_1pct_gap_pct": 20.0,
            # rate_5pct_gap_pct missing for ds_a
            "val_subdataset/ds_b/rate_mean_violation_gap_pct_generated": 8.0,
            "val_subdataset/ds_b/rate_1pct_gap_pct": 30.0,
            # rate_5pct_gap_pct missing for ds_b
        }
        # Before grace (update 1 of 3), should return None
        score = tracker.update(0, raw)
        assert score is None

    def test_grace_period_falls_back_after_threshold(self, wra_specs):
        """After grace period, missing metrics should renormalize."""
        tracker = BestModelTracker(
            metrics=wra_specs, ema_alpha=1.0, top_k=5,
            min_warmup_evals=0, missing_metric_grace_evals=2,
            subdataset_aggregation="mean",
        )
        raw = {
            "val_subdataset/ds_a/rate_mean_violation_gap_pct_generated": 5.0,
            "val_subdataset/ds_a/rate_1pct_gap_pct": 20.0,
            # rate_5pct_gap_pct missing
        }
        tracker.update(0, raw)  # update 1
        score = tracker.update(1, raw)  # update 2 >= grace=2
        assert score is not None

    def test_register_checkpoint_includes_subdataset_ema(self, wra_specs):
        """Manifest should contain the subdataset composite EMA value."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tracker = BestModelTracker(
                metrics=wra_specs, ema_alpha=1.0, top_k=5,
                min_warmup_evals=0, subdataset_aggregation="mean",
            )
            raw = self._make_subdataset_metrics({
                "ds_a": (5.0, 20.0, 10.0),
            })
            score = tracker.update(0, raw)
            path = os.path.join(tmpdir, "model_0.pt")
            open(path, "w").close()
            tracker.register_checkpoint(0, score, path, raw_metrics=raw)

            # Check the leaderboard entry has the subdataset composite EMA
            entry = tracker._leaderboard[0]
            assert "_subdataset_composite" in entry.ema_metrics
            assert entry.ema_metrics["_subdataset_composite"] == pytest.approx(score)

    def test_many_subdatasets(self, wra_specs):
        """Simulate 20 subdatasets (4 densities x 5 r_min)."""
        tracker = BestModelTracker(
            metrics=wra_specs, ema_alpha=1.0, top_k=5,
            min_warmup_evals=0, subdataset_aggregation="mean",
        )
        subdatasets = {}
        composites = []
        for i in range(20):
            viol = 2.0 + i * 0.5
            p1 = 10.0 + i * 1.0
            p5 = 5.0 + i * 0.5
            name = f"sd_{i:02d}"
            subdatasets[name] = (viol, p1, p5)
            composites.append(0.5 * viol + 0.25 * p1 + 0.25 * p5)
        raw = self._make_subdataset_metrics(subdatasets)
        score = tracker.update(0, raw)
        expected = sum(composites) / len(composites)
        assert score == pytest.approx(expected)


class TestSP500Metrics:
    """Test best-model tracking with SP500 stock price forecasting metrics."""

    @pytest.fixture
    def sp500_specs(self):
        return [
            MetricSpec(name="val_price_mse", weight=0.5, direction="minimize"),
            MetricSpec(name="val_price_mae", weight=0.25, direction="minimize"),
            MetricSpec(name="val_crps_mean", weight=0.25, direction="minimize"),
        ]

    @pytest.fixture
    def sp500_tracker(self, sp500_specs):
        return BestModelTracker(
            metrics=sp500_specs,
            ema_alpha=0.3,
            top_k=5,
            min_warmup_evals=3,
        )

    def _make_sp500_metrics(self, mse: float, mae: float, crps: float):
        """Helper to build SP500 metrics dict."""
        return {
            "val_price_mse": mse,
            "val_price_mae": mae,
            "val_crps_mean": crps,
        }

    def test_sp500_initialization(self, sp500_tracker):
        """Verify tracker initializes correctly with SP500 metrics."""
        assert sp500_tracker.leaderboard_size == 0
        assert sp500_tracker.best_score is None
        assert len(sp500_tracker.metrics) == 3

    def test_sp500_composite_score(self, sp500_tracker):
        """Verify composite score calculation with SP500 metrics."""
        # First update to initialize EMA
        metrics1 = self._make_sp500_metrics(mse=100.0, mae=8.0, crps=5.0)
        sp500_tracker.update(epoch=1, raw_metrics=metrics1)

        # All metrics minimize, so composite score = 0.5*100 + 0.25*8 + 0.25*5 = 53.25
        ema = sp500_tracker.get_ema_metrics()
        expected_score = 0.5 * ema["val_price_mse"] + 0.25 * ema["val_price_mae"] + 0.25 * ema["val_crps_mean"]
        
        # Second update to get score (after warmup we need 3 updates)
        sp500_tracker.update(epoch=2, raw_metrics=metrics1)
        score = sp500_tracker.update(epoch=3, raw_metrics=metrics1)
        assert score is not None
        assert abs(score - expected_score) < 0.1  # Allow small EMA variation

    def test_sp500_ema_decay(self, sp500_tracker):
        """Verify EMA smoothing works with SP500 metrics."""
        # Update 1: mse=100, mae=10, crps=5
        metrics1 = self._make_sp500_metrics(mse=100.0, mae=10.0, crps=5.0)
        sp500_tracker.update(epoch=1, raw_metrics=metrics1)

        # EMA should match raw values on first update
        ema = sp500_tracker.get_ema_metrics()
        assert ema["val_price_mse"] == 100.0
        assert ema["val_price_mae"] == 10.0
        assert ema["val_crps_mean"] == 5.0

        # Update 2: mse=80, mae=8, crps=4
        # EMA = 0.3*new + 0.7*old
        metrics2 = self._make_sp500_metrics(mse=80.0, mae=8.0, crps=4.0)
        sp500_tracker.update(epoch=2, raw_metrics=metrics2)

        expected_mse = 0.3 * 80.0 + 0.7 * 100.0  # 94.0
        expected_mae = 0.3 * 8.0 + 0.7 * 10.0     # 9.4
        expected_crps = 0.3 * 4.0 + 0.7 * 5.0     # 4.7

        ema = sp500_tracker.get_ema_metrics()
        assert abs(ema["val_price_mse"] - expected_mse) < 1e-6
        assert abs(ema["val_price_mae"] - expected_mae) < 1e-6
        assert abs(ema["val_crps_mean"] - expected_crps) < 1e-6

    def test_sp500_checkpoint_ranking(self, sp500_tracker):
        """Verify checkpoints are ranked correctly by composite score."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Warmup: 3 updates
            for i in range(1, 4):
                m = self._make_sp500_metrics(mse=100.0 - i*5, mae=10.0 - i*0.5, crps=5.0 - i*0.2)
                sp500_tracker.update(epoch=i, raw_metrics=m)

            # Save checkpoint after warmup
            metrics4 = self._make_sp500_metrics(mse=80.0, mae=8.0, crps=4.0)
            score4 = sp500_tracker.update(epoch=4, raw_metrics=metrics4)
            assert sp500_tracker.should_save(score4)

            ckpt4 = os.path.join(tmpdir, "checkpoint_epoch_4.pt")
            with open(ckpt4, "w") as f:
                f.write("dummy")
            sp500_tracker.register_checkpoint(epoch=4, score=score4, path=ckpt4)

            # Better checkpoint (lower errors)
            metrics5 = self._make_sp500_metrics(mse=70.0, mae=7.0, crps=3.5)
            score5 = sp500_tracker.update(epoch=5, raw_metrics=metrics5)
            assert sp500_tracker.should_save(score5)

            ckpt5 = os.path.join(tmpdir, "checkpoint_epoch_5.pt")
            with open(ckpt5, "w") as f:
                f.write("dummy")
            sp500_tracker.register_checkpoint(epoch=5, score=score5, path=ckpt5)

            # Check leaderboard: epoch 5 should be best (lowest composite)
            assert sp500_tracker.leaderboard_size == 2
            assert sp500_tracker.best_epoch == 5  # Best model (lowest errors)
            assert sp500_tracker.best_score == score5
            assert score5 < score4  # Verify epoch 5 is better than epoch 4

    def test_sp500_config_round_trip(self, sp500_specs):
        """Verify SP500 config can be saved and loaded."""
        config = {
            "enabled": True,
            "top_k": 5,
            "ema_alpha": 0.3,
            "min_warmup_evals": 3,
            "metrics": [
                {"name": "val_price_mse", "weight": 0.5, "direction": "minimize"},
                {"name": "val_price_mae", "weight": 0.25, "direction": "minimize"},
                {"name": "val_crps_mean", "weight": 0.25, "direction": "minimize"},
            ],
        }

        tracker = BestModelTracker.from_config(config)
        assert len(tracker.metrics) == 3
        assert tracker.metrics[0].name == "val_price_mse"
        assert tracker.metrics[0].weight == 0.5
        assert tracker.metrics[1].name == "val_price_mae"
        assert tracker.metrics[2].name == "val_crps_mean"
