"""Best-model checkpoint tracker with EMA-smoothed composite scoring.

Maintains a top-K leaderboard of the best model checkpoints observed during
training, using an exponential moving average (EMA) of configurable validation
metrics to produce a composite score.  Checkpoints are automatically managed:
new best models are saved and the worst checkpoint is evicted (and deleted from
disk) when the leaderboard is full.

A ``best_model.pt`` symlink is kept pointing at the current rank-1 checkpoint
so that post-training loading uses a stable path.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public dataclass for per-metric configuration
# ---------------------------------------------------------------------------


@dataclass
class MetricSpec:
    """Specification for one metric tracked by :class:`BestModelTracker`.

    Parameters
    ----------
    name : str
        Metric key as it appears in ``epoch_metrics`` (e.g.
        ``"val_rate_mean_violation_gap_pct_generated"``).
    weight : float
        Relative weight in the composite score (will be normalized to sum
        to 1 across all specs).
    direction : str
        ``"minimize"`` or ``"maximize"``.  Currently only ``"minimize"`` is
        used, but the field is kept for forward-compatibility.
    """

    name: str
    weight: float = 1.0
    direction: str = "minimize"

    def __post_init__(self) -> None:
        if self.direction not in ("minimize", "maximize"):
            raise ValueError(
                f"MetricSpec.direction must be 'minimize' or 'maximize', "
                f"got {self.direction!r}"
            )
        if self.weight < 0:
            raise ValueError(f"MetricSpec.weight must be >= 0, got {self.weight}")


# ---------------------------------------------------------------------------
# Internal leaderboard entry
# ---------------------------------------------------------------------------


@dataclass
class _LeaderboardEntry:
    epoch: int
    composite_score: float
    path: str
    raw_metrics: Dict[str, float] = field(default_factory=dict)
    ema_metrics: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main tracker
# ---------------------------------------------------------------------------


class BestModelTracker:
    """Track best model checkpoints during training.

    Parameters
    ----------
    metrics : list of MetricSpec
        Metrics to monitor.  Weights are normalized to sum to 1.
    ema_alpha : float
        EMA smoothing coefficient in ``(0, 1]``.  ``alpha=1`` disables
        smoothing.
    top_k : int
        Number of best checkpoints to retain on disk.
    min_warmup_evals : int
        Number of ``update`` calls before the tracker starts comparing scores.
        During warmup the EMA is still updated but ``should_save`` always
        returns ``False``.
    missing_metric_grace_evals : int
        After this many ``update`` calls, if some metrics have never been
        observed the composite score falls back to using only the observed
        metrics (re-normalizing weights).  Before the grace period, missing
        metrics cause ``_composite_score`` to return ``None``.
        Set to 0 to always tolerate missing metrics after warmup.
    """

    # Supported subdataset aggregation modes.
    _SUBDATASET_AGGREGATION_MODES = ("global", "mean", "worst")

    def __init__(
        self,
        metrics: List[MetricSpec],
        *,
        ema_alpha: float = 0.3,
        top_k: int = 5,
        min_warmup_evals: int = 3,
        missing_metric_grace_evals: int = 5,
        subdataset_aggregation: str = "global",
    ) -> None:
        if not metrics:
            raise ValueError("At least one MetricSpec must be provided.")
        if not (0.0 < ema_alpha <= 1.0):
            raise ValueError(f"ema_alpha must be in (0, 1], got {ema_alpha}")
        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {top_k}")
        if min_warmup_evals < 0:
            raise ValueError(
                f"min_warmup_evals must be >= 0, got {min_warmup_evals}"
            )
        if missing_metric_grace_evals < 0:
            raise ValueError(
                f"missing_metric_grace_evals must be >= 0, got {missing_metric_grace_evals}"
            )
        if subdataset_aggregation not in self._SUBDATASET_AGGREGATION_MODES:
            raise ValueError(
                f"subdataset_aggregation must be one of "
                f"{self._SUBDATASET_AGGREGATION_MODES}, got {subdataset_aggregation!r}"
            )

        self.metrics = list(metrics)
        # Normalize weights to sum to 1.
        total_w = sum(m.weight for m in self.metrics)
        if total_w <= 0:
            raise ValueError("Sum of metric weights must be > 0.")
        self._norm_weights = [m.weight / total_w for m in self.metrics]

        self.ema_alpha = ema_alpha
        self.top_k = top_k
        self.min_warmup_evals = min_warmup_evals
        self.missing_metric_grace_evals = missing_metric_grace_evals
        self.subdataset_aggregation = subdataset_aggregation

        # Per-metric EMA state: ``None`` until first observation.
        # In "global" mode these track the global metric values.
        # In "mean"/"worst" mode these track the *aggregated* composite
        # (single EMA value keyed by a synthetic name).
        self._ema_values: Dict[str, Optional[float]] = {
            m.name: None for m in self.metrics
        }
        # EMA for the aggregated per-subdataset composite (mean/worst modes).
        self._ema_subdataset_composite: Optional[float] = None
        self._num_updates: int = 0
        self._logged_missing_fallback: bool = False  # avoid log spam

        # Leaderboard: kept sorted best-first (lowest composite score first
        # when all metrics are "minimize").
        self._leaderboard: List[_LeaderboardEntry] = []

    # ------------------------------------------------------------------
    # EMA + composite score
    # ------------------------------------------------------------------

    def update(
        self, epoch: int, raw_metrics: Dict[str, Any]
    ) -> Optional[float]:
        """Apply EMA to each tracked metric and return composite score.

        Returns ``None`` during the warmup period.

        When ``subdataset_aggregation`` is ``"mean"`` or ``"worst"``, the
        composite is computed per-subdataset first (using
        ``val_subdataset/{name}/{base_metric}`` keys), then aggregated.
        EMA smoothing is applied to the aggregated composite.
        """
        self._num_updates += 1

        if self.subdataset_aggregation != "global":
            return self._update_subdataset(raw_metrics)

        # --- Global mode (original behavior) ---
        for spec in self.metrics:
            value = raw_metrics.get(spec.name)
            if value is None or not isinstance(value, (int, float)):
                continue
            prev = self._ema_values[spec.name]
            if prev is None:
                self._ema_values[spec.name] = float(value)
            else:
                self._ema_values[spec.name] = (
                    self.ema_alpha * float(value) + (1.0 - self.ema_alpha) * prev
                )

        if self._num_updates < self.min_warmup_evals:
            return None
        return self._composite_score()

    def _update_subdataset(self, raw_metrics: Dict[str, Any]) -> Optional[float]:
        """Compute per-subdataset composites, aggregate, apply EMA."""
        # Discover subdataset names from keys like "val_subdataset/{name}/{metric}"
        subdataset_names = self._discover_subdatasets(raw_metrics)

        # Before the grace period, require all tracked metrics per subdataset.
        # After grace, allow partial metrics (renormalized).
        past_grace = self._num_updates >= self.missing_metric_grace_evals
        require_all = not past_grace

        if not subdataset_names:
            # No subdataset keys found — fall back to global metrics as a
            # single "subdataset" so the tracker still works for
            # single-dataset training runs.
            raw_composite = self._single_composite(
                raw_metrics, prefix="", require_all=require_all,
            )
            if raw_composite is None:
                if self._num_updates < self.min_warmup_evals:
                    return None
                return self._ema_subdataset_composite
            aggregated = raw_composite
        else:
            # Compute composite per subdataset
            per_sd_scores: List[float] = []
            for sd_name in subdataset_names:
                prefix = f"val_subdataset/{sd_name}/"
                sd_score = self._single_composite(
                    raw_metrics, prefix=prefix, require_all=require_all,
                )
                if sd_score is not None:
                    per_sd_scores.append(sd_score)

            if not per_sd_scores:
                if self._num_updates < self.min_warmup_evals:
                    return None
                return self._ema_subdataset_composite

            if self.subdataset_aggregation == "mean":
                aggregated = sum(per_sd_scores) / len(per_sd_scores)
            else:  # "worst"
                aggregated = max(per_sd_scores)  # higher = worse for minimize

        # Apply EMA to the aggregated composite
        prev = self._ema_subdataset_composite
        if prev is None:
            self._ema_subdataset_composite = aggregated
        else:
            self._ema_subdataset_composite = (
                self.ema_alpha * aggregated + (1.0 - self.ema_alpha) * prev
            )

        if self._num_updates < self.min_warmup_evals:
            return None
        return self._ema_subdataset_composite

    def _discover_subdatasets(self, raw_metrics: Dict[str, Any]) -> List[str]:
        """Extract unique subdataset names from ``val_subdataset/{name}/...`` keys."""
        names: set = set()
        prefix = "val_subdataset/"
        for key in raw_metrics:
            if not isinstance(key, str) or not key.startswith(prefix):
                continue
            rest = key[len(prefix):]
            slash_idx = rest.find("/")
            if slash_idx > 0:
                names.add(rest[:slash_idx])
        return sorted(names)

    def _single_composite(
        self,
        raw_metrics: Dict[str, Any],
        *,
        prefix: str,
        require_all: bool = False,
    ) -> Optional[float]:
        """Compute a single weighted composite from metrics with *prefix*.

        For global mode (prefix=""), metric spec names are used directly.
        For subdataset mode (prefix="val_subdataset/{name}/"), the spec's
        base metric name (after stripping the ``val_`` prefix) is looked up
        under the subdataset prefix.

        When *require_all* is ``True``, returns ``None`` if any tracked
        metric is missing (enforces grace-period semantics).
        """
        observed_score = 0.0
        observed_weight = 0.0
        missing: List[str] = []
        for spec, w in zip(self.metrics, self._norm_weights):
            if prefix:
                # Spec name: "val_rate_1pct_gap_pct"
                # → base: "rate_1pct_gap_pct"
                # → lookup: "val_subdataset/{name}/rate_1pct_gap_pct"
                base_name = spec.name
                if base_name.startswith("val_"):
                    base_name = base_name[4:]
                key = prefix + base_name
            else:
                key = spec.name
            value = raw_metrics.get(key)
            if value is None or not isinstance(value, (int, float)):
                missing.append(key)
                continue
            signed = float(value) if spec.direction == "minimize" else -float(value)
            observed_score += w * signed
            observed_weight += w
        if require_all and missing:
            return None
        if observed_weight <= 0:
            return None
        return observed_score / observed_weight

    def _composite_score(self) -> Optional[float]:
        """Weighted sum of current EMA values.

        Before the grace period (``missing_metric_grace_evals``), returns
        ``None`` if any metric has never been observed.  After the grace
        period, missing metrics are skipped and the remaining weights are
        re-normalized — as long as at least one metric has been observed.
        """
        past_grace = self._num_updates >= self.missing_metric_grace_evals

        observed_score = 0.0
        observed_weight = 0.0
        missing: List[str] = []

        for spec, w in zip(self.metrics, self._norm_weights):
            v = self._ema_values[spec.name]
            if v is None:
                missing.append(spec.name)
                continue
            signed = v if spec.direction == "minimize" else -v
            observed_score += w * signed
            observed_weight += w

        if missing:
            if not past_grace:
                # Still within the grace period — require all metrics.
                return None
            # Past grace: fall back to observed-only score.
            if observed_weight <= 0:
                return None  # nothing observed at all
            if not self._logged_missing_fallback:
                log.warning(
                    "Best-model tracker: metrics %s were never observed after "
                    "%d evaluations. Falling back to composite score using "
                    "only the %d observed metric(s) (weights re-normalized).",
                    missing, self._num_updates,
                    len(self.metrics) - len(missing),
                )
                self._logged_missing_fallback = True
            # Re-normalize: observed_score currently uses weights that sum
            # to observed_weight (< 1.0).  Scale up so they sum to 1.0.
            return observed_score / observed_weight

        return observed_score

    def raw_composite_score(self, raw_metrics: Dict[str, Any]) -> Optional[float]:
        """Compute composite score from *raw* metrics of a single epoch.

        This does **not** use EMA state or warmup logic. It computes a weighted
        score directly from the metric values present in ``raw_metrics``.
        Missing metrics are skipped and remaining weights are re-normalized.
        Returns ``None`` only when none of the tracked metrics are present.

        Respects ``subdataset_aggregation``: when ``"mean"`` or ``"worst"``,
        computes per-subdataset composites and aggregates them.
        """
        if self.subdataset_aggregation != "global":
            subdataset_names = self._discover_subdatasets(raw_metrics)
            if subdataset_names:
                per_sd: List[float] = []
                for sd_name in subdataset_names:
                    s = self._single_composite(
                        raw_metrics, prefix=f"val_subdataset/{sd_name}/",
                    )
                    if s is not None:
                        per_sd.append(s)
                if not per_sd:
                    return None
                if self.subdataset_aggregation == "mean":
                    return sum(per_sd) / len(per_sd)
                return max(per_sd)  # worst

        # Global fallback
        return self._single_composite(raw_metrics, prefix="")

    # ------------------------------------------------------------------
    # Save / eviction logic
    # ------------------------------------------------------------------

    @property
    def warmed_up(self) -> bool:
        """Whether the tracker has completed the warmup period."""
        return self._num_updates >= self.min_warmup_evals

    def should_save(self, score: Optional[float]) -> bool:
        """Return ``True`` if *score* is good enough to enter the leaderboard.

        Always returns ``False`` during warmup (when *score* is ``None``).
        """
        if score is None:
            return False
        if len(self._leaderboard) < self.top_k:
            return True
        # Worst entry is at the end (leaderboard is sorted best-first).
        return score < self._leaderboard[-1].composite_score

    def register_checkpoint(
        self,
        epoch: int,
        score: float,
        path: str,
        raw_metrics: Optional[Dict[str, float]] = None,
    ) -> Optional[str]:
        """Insert a checkpoint into the leaderboard.

        Returns the path of the evicted checkpoint (deleted from disk) if the
        leaderboard was full, else ``None``.
        """
        ema_snapshot = {k: v for k, v in self._ema_values.items() if v is not None}
        if self.subdataset_aggregation != "global" and self._ema_subdataset_composite is not None:
            ema_snapshot["_subdataset_composite"] = self._ema_subdataset_composite
        entry = _LeaderboardEntry(
            epoch=epoch,
            composite_score=score,
            path=path,
            raw_metrics=dict(raw_metrics or {}),
            ema_metrics=ema_snapshot,
        )

        self._leaderboard.append(entry)
        self._leaderboard.sort(key=lambda e: e.composite_score)

        evicted_path: Optional[str] = None
        if len(self._leaderboard) > self.top_k:
            worst = self._leaderboard.pop()  # remove worst (last)
            evicted_path = worst.path
            if os.path.isfile(evicted_path):
                try:
                    os.remove(evicted_path)
                    log.info("Evicted checkpoint: %s", evicted_path)
                except OSError as e:
                    log.warning("Failed to delete evicted checkpoint %s: %s", evicted_path, e)

        return evicted_path

    # ------------------------------------------------------------------
    # Symlink management
    # ------------------------------------------------------------------

    def update_symlink(self, save_dir: str) -> None:
        """Create/update ``best_model.pt`` symlink to current rank-1 checkpoint."""
        if not self._leaderboard:
            return
        best = self._leaderboard[0]
        link_path = os.path.join(save_dir, "best_model.pt")
        target = os.path.basename(best.path)
        try:
            if os.path.islink(link_path) or os.path.exists(link_path):
                os.remove(link_path)
            os.symlink(target, link_path)
            log.info("Symlink best_model.pt -> %s", target)
        except OSError as e:
            log.warning("Failed to update best_model.pt symlink: %s", e)

    # ------------------------------------------------------------------
    # Manifest persistence
    # ------------------------------------------------------------------

    def save_manifest(self, save_dir: str) -> str:
        """Write ``best_models_manifest.json`` to *save_dir*.

        Returns the path of the manifest file.
        """
        os.makedirs(save_dir, exist_ok=True)
        manifest = {
            "config": {
                "top_k": self.top_k,
                "ema_alpha": self.ema_alpha,
                "min_warmup_evals": self.min_warmup_evals,
                "subdataset_aggregation": self.subdataset_aggregation,
                "metrics": [asdict(m) for m in self.metrics],
            },
            "leaderboard": [
                {
                    "rank": rank + 1,
                    "epoch": entry.epoch,
                    "composite_score": entry.composite_score,
                    "path": entry.path,
                    "raw_metrics": entry.raw_metrics,
                    "ema_metrics": entry.ema_metrics,
                }
                for rank, entry in enumerate(self._leaderboard)
            ],
            "total_evaluations": self._num_updates,
            "last_updated_epoch": (
                self._leaderboard[0].epoch if self._leaderboard else None
            ),
        }
        path = os.path.join(save_dir, "best_models_manifest.json")
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        log.info("Saved best-model manifest to %s", path)
        return path

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def best_score(self) -> Optional[float]:
        return self._leaderboard[0].composite_score if self._leaderboard else None

    @property
    def best_epoch(self) -> Optional[int]:
        return self._leaderboard[0].epoch if self._leaderboard else None

    @property
    def best_path(self) -> Optional[str]:
        return self._leaderboard[0].path if self._leaderboard else None

    @property
    def leaderboard_size(self) -> int:
        return len(self._leaderboard)

    def get_ema_metrics(self) -> Dict[str, Optional[float]]:
        """Return current EMA values (may contain ``None`` for unobserved metrics)."""
        out = dict(self._ema_values)
        if self.subdataset_aggregation != "global":
            out["_subdataset_composite_ema"] = self._ema_subdataset_composite
        return out

    # ------------------------------------------------------------------
    # Factory from config dict (YAML)
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "BestModelTracker":
        """Create a ``BestModelTracker`` from a config dict (e.g. Hydra).

        Expected schema::

            best_model:
              enabled: true
              top_k: 5
              ema_alpha: 0.3
              min_warmup_evals: 3
              metrics:
                - name: <val_metric_name>   # key as it appears in epoch_metrics
                  weight: 0.5
                  direction: minimize             # or "maximize"
                - name: <val_another_metric>
                  weight: 0.25
                  direction: minimize
        """
        metrics_cfg = cfg.get("metrics", [])
        if not metrics_cfg:
            raise ValueError("best_model.metrics must be a non-empty list.")
        metric_specs = [
            MetricSpec(
                name=m["name"],
                weight=float(m.get("weight", 1.0)),
                direction=m.get("direction", "minimize"),
            )
            for m in metrics_cfg
        ]
        return cls(
            metrics=metric_specs,
            ema_alpha=float(cfg.get("ema_alpha", 0.3)),
            top_k=int(cfg.get("top_k", 5)),
            min_warmup_evals=int(cfg.get("min_warmup_evals", 3)),
            missing_metric_grace_evals=int(cfg.get("missing_metric_grace_evals", 5)),
            subdataset_aggregation=str(cfg.get("subdataset_aggregation", "global")),
        )
