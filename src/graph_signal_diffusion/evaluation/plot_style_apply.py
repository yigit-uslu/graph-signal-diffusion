"""Shared global matplotlib style application.

Both ``compare_baselines.py`` and ``replot_baselines.py`` apply the same
``plot_style.global`` (backend, rcParams, default DPI) so pipeline- and
replot-generated PDFs share aesthetics.

The canonical style for SP500 baseline comparison lives in
``conf/plot_style/sp500_compare.yaml`` and is loaded as a Hydra default
of ``compare_baselines_sp500.yaml``.  Both entry points read from the
same yaml — there is no longer a separate runtime snapshot.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Mapping, Optional


def _ps_as_dict(value: Any) -> dict:
    """Best-effort coerce a (possibly OmegaConf) mapping to a plain dict."""
    if isinstance(value, Mapping):
        return dict(value)
    try:  # OmegaConf DictConfig is not a Mapping subclass on all versions
        from omegaconf import DictConfig, OmegaConf
        if isinstance(value, DictConfig):
            out = OmegaConf.to_container(value, resolve=True)
            return out if isinstance(out, dict) else {}
    except Exception:
        pass
    return {}


def resolve_paper_style(
    plot_style: Optional[Mapping[str, Any]],
    plot_name: str,
    baseline_colors: Optional[Mapping[str, str]] = None,
) -> "tuple[dict, dict, dict]":
    """Resolve ``(global_rc, per_figure_style, baseline_colors)`` from a plot_style.

    Lets a paper-figure plotter style itself **identically on every code path**
    (a live CLI run *or* a sidecar replot): apply ``global_rc`` via
    ``matplotlib.rc_context`` and read per-figure params from
    ``per_figure_style`` (``plot_style.plots[plot_name]``).  ``baseline_colors``
    merges ``plot_style.baseline.colors`` with the optional explicit override
    (override wins).  Safe with ``plot_style=None`` → all three empty, so the
    plotter falls back to its built-in defaults (e.g. in unit tests).
    """
    ps = _ps_as_dict(plot_style)
    g = _ps_as_dict(ps.get("global"))
    rc = dict(_ps_as_dict(g.get("rc_params")))
    dpi = g.get("default_dpi")
    if dpi is not None:
        try:
            rc.setdefault("savefig.dpi", float(dpi))
            rc.setdefault("figure.dpi", float(dpi))
        except (TypeError, ValueError):
            pass
    per_fig = _ps_as_dict(_ps_as_dict(ps.get("plots")).get(plot_name))
    colors = dict(_ps_as_dict(_ps_as_dict(ps.get("baseline")).get("colors")))
    if baseline_colors:
        colors.update(dict(baseline_colors))
    return rc, per_fig, colors


def apply_global_plot_style(global_cfg: Optional[Mapping[str, Any]]) -> None:
    """Apply ``plot_style.global`` (``backend`` / ``rc_params`` / ``default_dpi``).

    Safe to call with ``None`` or an empty dict — becomes a no-op.  This
    is the *only* runtime mutation point for global rcParams in the
    compare-baselines / replot pipelines; all per-plot tweaks live
    inside the plot functions themselves (typically as
    ``with matplotlib.rc_context({...}):`` blocks).
    """
    if not global_cfg:
        return

    import matplotlib

    backend = global_cfg.get("backend")
    if backend:
        backend_str = str(backend)
        try:
            current = str(matplotlib.get_backend())
            if current.lower() != backend_str.lower():
                matplotlib.use(backend_str, force=True)
        except Exception as exc:
            warnings.warn(
                f"Could not apply matplotlib backend '{backend_str}': {exc}",
                RuntimeWarning, stacklevel=2,
            )

    rc_params = dict(global_cfg.get("rc_params") or {})
    if rc_params:
        try:
            matplotlib.rcParams.update(rc_params)
        except Exception as exc:
            warnings.warn(
                f"Could not apply rc_params from plot_style.global: {exc}",
                RuntimeWarning, stacklevel=2,
            )

    default_dpi = global_cfg.get("default_dpi")
    if default_dpi is not None:
        try:
            dpi_val = float(default_dpi)
            matplotlib.rcParams["savefig.dpi"] = dpi_val
            matplotlib.rcParams["figure.dpi"] = dpi_val
        except (TypeError, ValueError):
            warnings.warn(
                f"Invalid plot_style.global.default_dpi={default_dpi!r}; ignoring.",
                RuntimeWarning, stacklevel=2,
            )


def load_plot_style_from_run_dir(run_dir: str) -> Optional[dict]:
    """Load the **full** ``plot_style`` subtree from ``<run_dir>/.hydra/config.yaml``.

    Unlike :func:`apply_global_plot_style_from_run_dir` (which applies only
    ``plot_style.global`` to the *global* rcParams), this returns the entire
    ``plot_style`` mapping (``global`` + ``baseline`` + ``plots``) as a plain,
    resolved dict so it can be handed to the self-styling paper-figure plotters
    via their ``plot_style=`` argument — making a sidecar replot render
    identically to the live run that produced it.

    Returns ``None`` if the snapshot or the ``plot_style`` node is absent.
    Best-effort on resolution: if interpolation resolution fails (e.g. a
    ``${hydra:...}`` resolver unavailable outside a live hydra context), falls
    back to the raw/unresolved container — the SP500 ``plot_style`` values are
    all literals, so the fallback is loss-free for them.
    """
    cfg_path = os.path.join(run_dir, ".hydra", "config.yaml")
    if not os.path.exists(cfg_path):
        return None
    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(cfg_path)
        node = OmegaConf.select(cfg, "plot_style")
        if node is None:
            return None
        try:
            container = OmegaConf.to_container(node, resolve=True)
        except Exception:
            container = OmegaConf.to_container(node, resolve=False)
        return container if isinstance(container, dict) else None
    except Exception as exc:
        warnings.warn(
            f"Could not load plot_style from {cfg_path}: {exc}",
            RuntimeWarning, stacklevel=2,
        )
        return None


def apply_global_plot_style_from_run_dir(run_dir: str) -> bool:
    """Load ``<run_dir>/.hydra/config.yaml`` and apply ``plot_style.global``.

    Returns True if a config was found and the style was applied,
    False otherwise.  Used by ``replot_baselines`` so it can recover
    the exact style the pipeline used without rerunning hydra.
    """
    cfg_path = os.path.join(run_dir, ".hydra", "config.yaml")
    if not os.path.exists(cfg_path):
        return False
    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(cfg_path)
        # Resolve only the plot_style.global subtree — the rest of the
        # snapshot may contain hydra interpolations that only work inside
        # a live hydra context.
        global_node = OmegaConf.select(cfg, "plot_style.global")
        if global_node is None:
            return False
        global_dict = OmegaConf.to_container(global_node, resolve=True) or {}
        if not isinstance(global_dict, dict):
            return False
    except Exception as exc:
        warnings.warn(
            f"Could not load plot_style.global from {cfg_path}: {exc}",
            RuntimeWarning, stacklevel=2,
        )
        return False
    apply_global_plot_style(global_dict)
    return True
