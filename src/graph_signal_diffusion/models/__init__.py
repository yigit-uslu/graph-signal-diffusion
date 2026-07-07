from __future__ import annotations
import importlib
import pkgutil
from graph_signal_diffusion.core.registry import Registry

MODEL_REGISTRY: Registry = Registry("models")

def discover_models():
    """Import all model subpackages."""
    pkg_name = __name__
    for m in pkgutil.iter_modules(__path__):
        if m.ispkg:
            importlib.import_module(f"{pkg_name}.{m.name}")

def get_model(name: str):
    return MODEL_REGISTRY.get(name)

__all__ = ["MODEL_REGISTRY", "discover_models", "get_model"]


# """
# Graph neural network models for diffusion.

# Main Models:
#     - UNet: Dummy UNet model
#     - UGNN: GNN-based U-Net with graph signal downsampling and upsampling
#     - SimpleUNet: Simplified version for ablation
#     - PureGNN: Pure GNN baseline

# Components:
#     All reusable components are available via:
#     from graph_signal_diffusion.models.components import ...
# """

# from __future__ import annotations


# # NOTE: Avoid importing model submodules at package import time.
# # Importing model modules here causes them to register into MODEL_REGISTRY
# # eagerly which makes it hard to control which models are available during
# # a Hydra run and can lead to accidental loading of multiple model presets.
# # Prefer explicit imports (e.g., `from graph_signal_diffusion.models.ugnn import UGNN`)
# # or use `discover_models(cfg.get('model', None))` from the CLI to import only
# # the model necessary for the current run.

# # Registry utilities
# from .registry import MODEL_REGISTRY, discover_models, get_model

# # Expose components for convenience
# from . import components

# __all__ = [
#     "components",
#     "MODEL_REGISTRY",
#     "discover_models",
#     "get_model",
# ]