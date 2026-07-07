"""Evaluation scheduling strategies for DiffusionTrainer.

Each schedule decides, given the current epoch and total number of epochs,
whether an evaluation pass should be run.  All concrete schedules share an
``eval_on_last_epoch`` flag that forces evaluation unconditionally on the
final training epoch (``epoch == max_epochs - 1``).

Usage::

    schedule = EvalSchedule.from_config(
        {"type": "cosine", "min_period": 2, "max_period": 30},
        fallback_period=10,
    )

    for epoch in range(max_epochs):
        if schedule.should_eval(epoch, max_epochs):
            ...

Config schema (``eval_schedule:`` YAML block)::

    # Uniform (default / backward-compat)
    eval_schedule:
      type: uniform
      period: 10                # eval every 10 epochs (1-indexed: epochs 9, 19, ...)
      eval_on_last_epoch: true  # always eval on the final epoch

    # Multi-phase (different period per training phase)
    eval_schedule:
      type: multi_phase
      eval_on_last_epoch: true
      phases:
        - period: 5             # eval every 5 epochs from epoch 0 ...
          until_epoch: 50       # ... until epoch 50 (exclusive)
        - period: 20            # eval every 20 epochs from epoch 50 ...
          until_epoch: 150      # ... until epoch 150 (exclusive)
        - period: 5             # eval every 5 epochs from epoch 150 to end

    # Cosine density (higher density at start and end, lower at midpoint)
    eval_schedule:
      type: cosine
      min_period: 2             # smallest gap between evals (at start / end)
      max_period: 30            # largest gap between evals (at midpoint)
      eval_on_last_epoch: true
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple


class EvalSchedule(ABC):
    """Abstract base class for evaluation schedules."""

    @abstractmethod
    def should_eval(self, epoch: int, max_epochs: int) -> bool:
        """Return True if evaluation should run at *epoch*.

        Args:
            epoch: Current epoch index (0-based).
            max_epochs: Total number of training epochs.

        Returns:
            True when the evaluation pipeline should be triggered.
        """

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @staticmethod
    def from_config(
        cfg: Optional[Any],
        fallback_period: int = 10,
    ) -> "EvalSchedule":
        """Construct an :class:`EvalSchedule` from a config object or dict.

        Args:
            cfg: A dict-like object with at least a ``type`` key, or ``None``.
                When ``None`` (or when ``cfg`` is missing a ``type`` key),
                falls back to :class:`UniformEvalSchedule` with
                *fallback_period*.
            fallback_period: Period used when *cfg* is absent or invalid.

        Returns:
            A concrete :class:`EvalSchedule` instance.

        Raises:
            ValueError: If *cfg* specifies an unknown schedule type or
                contains invalid parameters.
        """
        def _get(key: str, default: Any = None) -> Any:
            if cfg is None:
                return default
            if isinstance(cfg, dict):
                return cfg.get(key, default)
            get_fn = getattr(cfg, "get", None)
            if callable(get_fn):
                try:
                    return get_fn(key, default)
                except Exception:
                    pass
            return getattr(cfg, key, default)

        if cfg is None:
            return UniformEvalSchedule(period=fallback_period)

        schedule_type = _get("type", None)
        if schedule_type is None:
            return UniformEvalSchedule(period=fallback_period)

        schedule_type = str(schedule_type).strip().lower()
        eval_on_last = bool(_get("eval_on_last_epoch", True))

        if schedule_type == "uniform":
            period = int(_get("period", fallback_period))
            return UniformEvalSchedule(period=period, eval_on_last_epoch=eval_on_last)

        if schedule_type == "multi_phase":
            phases_raw = _get("phases", None)
            if phases_raw is None:
                raise ValueError(
                    "eval_schedule type='multi_phase' requires a 'phases' list."
                )
            phases: List[Dict[str, Any]] = []
            for p in phases_raw:
                if isinstance(p, dict):
                    phases.append(p)
                else:
                    # OmegaConf structured node -> convert to dict
                    try:
                        from omegaconf import OmegaConf  # type: ignore
                        phases.append(OmegaConf.to_container(p, resolve=True))  # type: ignore[arg-type]
                    except Exception:
                        phases.append(dict(p))
            return MultiPhaseEvalSchedule(phases=phases, eval_on_last_epoch=eval_on_last)

        if schedule_type == "cosine":
            min_period = int(_get("min_period", 1))
            max_period = int(_get("max_period", fallback_period))
            return CosineEvalSchedule(
                min_period=min_period,
                max_period=max_period,
                eval_on_last_epoch=eval_on_last,
            )

        raise ValueError(
            f"Unknown eval_schedule type '{schedule_type}'. "
            "Expected one of: 'uniform', 'multi_phase', 'cosine'."
        )


# ---------------------------------------------------------------------------
# Concrete implementations
# ---------------------------------------------------------------------------


class UniformEvalSchedule(EvalSchedule):
    """Evaluate at uniformly-spaced epochs.

    Uses ``epoch % period == 0`` semantics so that a ``period=10`` schedule
    fires at epochs 0, 10, 20, … This is consistent with
    :class:`MultiPhaseEvalSchedule` which uses the same 0-indexed convention.
    Epoch 0 evaluation is controlled by ``eval_on_first_epoch`` in the
    training loop.

    Args:
        period: Number of epochs between evaluations. Must be > 0.
        eval_on_last_epoch: When True, also force evaluation on
            ``epoch == max_epochs - 1``.
    """

    def __init__(self, period: int = 10, eval_on_last_epoch: bool = True) -> None:
        if period <= 0:
            raise ValueError(
                f"UniformEvalSchedule: period must be > 0, got {period}."
            )
        self.period = int(period)
        self.eval_on_last_epoch = bool(eval_on_last_epoch)

    def should_eval(self, epoch: int, max_epochs: int) -> bool:
        if self.eval_on_last_epoch and epoch == max_epochs - 1:
            return True
        return epoch % self.period == 0

    def __repr__(self) -> str:
        return (
            f"UniformEvalSchedule(period={self.period}, "
            f"eval_on_last_epoch={self.eval_on_last_epoch})"
        )


class MultiPhaseEvalSchedule(EvalSchedule):
    """Evaluate at different frequencies during distinct training phases.

    Each phase covers a contiguous epoch range and has its own ``period``.
    Within a phase, evals fire when ``(epoch - phase_start) % period == 0``,
    so each phase starts a fresh grid from its own start boundary.

    Args:
        phases: Ordered list of phase dicts.  Each dict must contain:
            - ``period`` (int > 0): eval period for this phase.
            - ``until_epoch`` (int, optional): first epoch *not* in this
              phase.  The last phase must omit this key (it runs until
              training ends).  **Negative values are allowed** and are
              resolved relative to ``max_epochs`` at call time:
              ``until_epoch = max_epochs + until_epoch``.
              For example, ``until_epoch=-150`` on a phase means
              "this phase ends 150 epochs before training ends".
        eval_on_last_epoch: When True, force evaluation on the final epoch.

    Example — dense early, sparse mid, dense for last 150 epochs::

        MultiPhaseEvalSchedule(phases=[
            {"period": 5,  "until_epoch": 50},
            {"period": 20, "until_epoch": -150},
            {"period": 5},
        ])

    Note:
        Ordering of resolved ``until_epoch`` values is checked lazily at
        ``should_eval()`` time when negative values are involved.
    """

    def __init__(
        self,
        phases: List[Dict[str, Any]],
        eval_on_last_epoch: bool = True,
    ) -> None:
        if not phases:
            raise ValueError("MultiPhaseEvalSchedule: 'phases' must not be empty.")

        self.eval_on_last_epoch = bool(eval_on_last_epoch)

        # Parse and validate phases -------------------------------------------
        # Each tuple entry: (raw_phase_start, period, raw_until_epoch)
        # Both raw_phase_start and raw_until_epoch may be negative; they are
        # resolved against max_epochs lazily inside should_eval().
        parsed: List[Tuple[int, int, Optional[int]]] = []
        prev_until: int = 0  # raw until of the previous phase = raw start of this one
        for i, phase in enumerate(phases):
            is_last = i == len(phases) - 1

            period = int(phase.get("period", 0))
            if period <= 0:
                raise ValueError(
                    f"MultiPhaseEvalSchedule: phases[{i}].period must be > 0, "
                    f"got {period}."
                )

            until = phase.get("until_epoch", None)
            if until is not None:
                until = int(until)
                if is_last:
                    raise ValueError(
                        "MultiPhaseEvalSchedule: the last phase must not have "
                        "'until_epoch'; it should run until training ends."
                    )
                # For positive values where both boundaries are known we can
                # validate ordering eagerly.  When either boundary is negative
                # (relative to max_epochs) we skip the ordering check and leave
                # it to the caller to supply a coherent max_epochs.
                if until >= 0 and prev_until >= 0 and until <= prev_until:
                    raise ValueError(
                        f"MultiPhaseEvalSchedule: phases[{i}].until_epoch={until} must "
                        f"be greater than the phase start ({prev_until})."
                    )
            else:
                if not is_last:
                    raise ValueError(
                        f"MultiPhaseEvalSchedule: phases[{i}] is not the last phase "
                        "but is missing 'until_epoch'."
                    )

            # phase_start is the raw until of the previous phase so that
            # negative boundaries propagate correctly into later phases.
            parsed.append((prev_until, period, until))
            if until is not None:
                prev_until = until

        self._phases: List[Tuple[int, int, Optional[int]]] = parsed

    @staticmethod
    def _resolve_until(until: Optional[int], max_epochs: int) -> Optional[int]:
        """Resolve a raw until_epoch value against max_epochs.

        Negative values use Python-style offset from the end:
        ``-k`` resolves to ``max_epochs - k``.
        Non-negative values and ``None`` pass through unchanged.
        """
        if until is None or until >= 0:
            return until
        return max_epochs + until

    def should_eval(self, epoch: int, max_epochs: int) -> bool:
        if self.eval_on_last_epoch and epoch == max_epochs - 1:
            return True
        for raw_start, period, raw_until in self._phases:
            start = self._resolve_until(raw_start, max_epochs)
            until = self._resolve_until(raw_until, max_epochs)
            if until is None or epoch < until:
                # epoch is in this phase
                return (epoch - start) % period == 0
        # epoch is beyond the last defined phase end — use last phase
        raw_last_start, last_period, _ = self._phases[-1]
        last_start = self._resolve_until(raw_last_start, max_epochs)
        return (epoch - last_start) % last_period == 0

    def __repr__(self) -> str:
        return (
            f"MultiPhaseEvalSchedule(phases={self._phases!r}, "
            f"eval_on_last_epoch={self.eval_on_last_epoch})"
        )


class CosineEvalSchedule(EvalSchedule):
    """Higher eval density at the start and end of training, lower at midpoint.

    The instantaneous period at epoch ``e`` is::

        p(e) = round(min_period + (max_period - min_period)
                     * sin²(π * e / (max_epochs - 1)))

    Starting from epoch 0, the next eval is scheduled at
    ``e_next = e + max(1, p(e))``, then at
    ``e_next = e_next + max(1, p(e_next))``, and so on.  The resulting
    eval epoch set is pre-computed once per ``max_epochs`` value and cached.

    Args:
        min_period: Minimum gap between evaluations (used near start / end
            of training). Must be >= 1.
        max_period: Maximum gap between evaluations (used near the midpoint
            of training). Must be >= ``min_period``.
        eval_on_last_epoch: When True, force evaluation on the final epoch.
    """

    def __init__(
        self,
        min_period: int = 1,
        max_period: int = 10,
        eval_on_last_epoch: bool = True,
    ) -> None:
        if min_period < 1:
            raise ValueError(
                f"CosineEvalSchedule: min_period must be >= 1, got {min_period}."
            )
        if max_period < min_period:
            raise ValueError(
                f"CosineEvalSchedule: max_period ({max_period}) must be >= "
                f"min_period ({min_period})."
            )
        self.min_period = int(min_period)
        self.max_period = int(max_period)
        self.eval_on_last_epoch = bool(eval_on_last_epoch)

        # Cache: max_epochs -> frozenset of eval epoch indices
        self._cache: Dict[int, FrozenSet[int]] = {}

    def _compute_eval_epochs(self, max_epochs: int) -> FrozenSet[int]:
        """Pre-compute the full set of eval epoch indices for *max_epochs*."""
        if max_epochs <= 0:
            return frozenset()
        if max_epochs == 1:
            return frozenset({0})

        denom = float(max_epochs - 1)
        delta = float(self.max_period - self.min_period)

        def _period_at(e: int) -> int:
            sin2 = math.sin(math.pi * e / denom) ** 2
            return max(1, round(self.min_period + delta * sin2))

        epochs: Set[int] = set()
        e = 0
        while e < max_epochs:
            epochs.add(e)
            e += _period_at(e)

        return frozenset(epochs)

    def _get_eval_set(self, max_epochs: int) -> FrozenSet[int]:
        if max_epochs not in self._cache:
            self._cache[max_epochs] = self._compute_eval_epochs(max_epochs)
        return self._cache[max_epochs]

    def should_eval(self, epoch: int, max_epochs: int) -> bool:
        if self.eval_on_last_epoch and epoch == max_epochs - 1:
            return True
        return epoch in self._get_eval_set(max_epochs)

    def eval_epochs(self, max_epochs: int) -> List[int]:
        """Return sorted list of eval epoch indices for inspection / logging."""
        base = sorted(self._get_eval_set(max_epochs))
        if self.eval_on_last_epoch and (max_epochs - 1) not in self._get_eval_set(max_epochs):
            base = sorted(set(base) | {max_epochs - 1})
        return base

    def __repr__(self) -> str:
        return (
            f"CosineEvalSchedule(min_period={self.min_period}, "
            f"max_period={self.max_period}, "
            f"eval_on_last_epoch={self.eval_on_last_epoch})"
        )
