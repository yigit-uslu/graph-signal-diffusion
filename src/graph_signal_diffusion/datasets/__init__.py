# This file is for auto-discovery of datasets in the datasets module.
# datasets/__init__.py
from __future__ import annotations
import importlib
import pkgutil

from graph_signal_diffusion.core.registry import Registry
from .replicated_dataset import ReplicatedDataset
from .replicated_sampler import ReplicatedGroupedBatchSampler

DATASET_REGISTRY: Registry = Registry("datasets")

def discover_datasets():
    """
    Imports all subpackages under ./datasets so they can register themselves.
    Call this once at program start.
    """
    pkg_name = __name__  # "datasets"
    for m in pkgutil.iter_modules(__path__):
        if m.ispkg:
            importlib.import_module(f"{pkg_name}.{m.name}")

# convenience
def get_dataset_builder(name: str):
    return DATASET_REGISTRY.get(name)
