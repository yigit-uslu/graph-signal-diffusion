"""Phase-2 regression test: no module should mutate matplotlib.rcParams
as an import-time side effect during the compare-baselines startup path.

Before Phase 2, importing ``graph_signal_diffusion.datasets`` (via
``discover_datasets()``) transitively loaded ``sp100/utils_graph.py``,
which mutated ~40 rcParams at module load.  This test locks in the fix:
``discover_datasets() + discover_baselines() + discover_tasks()`` must
leave matplotlib.rcParams unchanged.
"""

from __future__ import annotations

import matplotlib


# Keys that are OK to change.  ``backend`` and ``backend_fallback`` are
# both mutated as a side effect of ``matplotlib.use('Agg')``, which a
# few modules call intentionally because they need a non-interactive
# backend.  Neither affects rendering aesthetics.
_ALLOWED_CHANGES = {"backend", "backend_fallback"}


def test_discover_does_not_mutate_rcparams():
    before = dict(matplotlib.rcParams)

    from graph_signal_diffusion.cli.compare_baselines import (
        discover_baselines,
        discover_datasets,
        discover_tasks,
    )
    discover_datasets()
    discover_baselines()
    discover_tasks()

    after = dict(matplotlib.rcParams)

    diffs = {
        k: (before.get(k), after.get(k))
        for k in set(before) | set(after)
        if k not in _ALLOWED_CHANGES and before.get(k) != after.get(k)
    }
    assert not diffs, (
        "Import-time rcParams mutations detected during discover_*:\n"
        + "\n".join(f"  {k}: {a!r} -> {b!r}" for k, (a, b) in sorted(diffs.items()))
    )
