from __future__ import annotations
import importlib
import pkgutil

from graph_signal_diffusion.core.registry import Registry
from graph_signal_diffusion.baselines.base import BaseBaseline

BASELINE_REGISTRY = Registry("baselines")

def discover_baselines():
    """Import all baseline subpackages so they can register themselves."""
    pkg_name = __name__
    for m in pkgutil.iter_modules(__path__):
        if m.ispkg:
            importlib.import_module(f"{pkg_name}.{m.name}")

def get_baseline(name: str):
    return BASELINE_REGISTRY.get(name)

__all__ = ["BASELINE_REGISTRY", "discover_baselines", "get_baseline", "BaseBaseline"]