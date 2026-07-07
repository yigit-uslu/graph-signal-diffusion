"""Regression test for the paper-figure config plumbing in compare_baselines.

Guards the bug where ``_paper_out_dir`` was set inside ``main()`` using ``os``,
which ``main``'s function-local ``import os`` shadowed — raising
``UnboundLocalError: cannot access local variable 'os'``.  The plumbing now
lives in the module-level helper ``_attach_paper_figure_cfg`` (where ``os`` is
the top-level import), so importing + calling it must succeed and set both
attributes.
"""
import types
from omegaconf import OmegaConf

from graph_signal_diffusion.cli.compare_baselines import _attach_paper_figure_cfg


def test_attach_paper_figure_cfg_sets_attrs():
    task = types.SimpleNamespace()
    cfg = OmegaConf.create(
        {"paper_figures": {"enabled": True, "fig1": {"n_stocks": 3}}}
    )
    # Must NOT raise (the original bug raised UnboundLocalError here).
    _attach_paper_figure_cfg(task, cfg, "/tmp/run123")
    assert task._paper_out_dir == "/tmp/run123/eval_viz/paper"
    assert task.paper_figures_cfg["enabled"] is True
    # Nested config is resolved to a plain dict by _as_dict.
    assert task.paper_figures_cfg["fig1"]["n_stocks"] == 3


def test_attach_paper_figure_cfg_missing_block():
    task = types.SimpleNamespace()
    _attach_paper_figure_cfg(task, OmegaConf.create({}), "/tmp/run")
    assert task.paper_figures_cfg == {}            # _as_dict of missing -> {}
    assert task._paper_out_dir == "/tmp/run/eval_viz/paper"


def test_attach_paper_figure_cfg_none_task_is_noop():
    # Should simply return without error.
    _attach_paper_figure_cfg(None, OmegaConf.create({"paper_figures": {}}), "/tmp/x")
