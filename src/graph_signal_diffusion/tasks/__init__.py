from __future__ import annotations
import importlib
import pkgutil
from graph_signal_diffusion.core.registry import Registry
from graph_signal_diffusion.tasks.base import BaseTask

TASK_REGISTRY = Registry("tasks")

def discover_tasks():
    """Import all task subpackages."""
    pkg_name = __name__
    for m in pkgutil.iter_modules(__path__):
        if m.ispkg:
            importlib.import_module(f"{pkg_name}.{m.name}")

def get_task(name: str):
    return TASK_REGISTRY.get(name)

__all__ = ["TASK_REGISTRY", "discover_tasks", "get_task", "BaseTask"]