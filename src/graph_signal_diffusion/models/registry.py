# # Registry for models
# from __future__ import annotations
# import importlib
# import pkgutil
# from graph_signal_diffusion.core.registry import Registry

# MODEL_REGISTRY = Registry("models")

# def discover_models(model_name: str | None = None):
#     """Import model submodules to register them.

#     Simplified behaviour:
#       - If `model_name` is provided (e.g. "ugnn"), import only
#         `graph_signal_diffusion.models.{model_name}`.
#       - If `model_name` is None, import all modules under the models package.

#     This keeps discovery deterministic and avoids accidentally importing
#     unrelated model modules (which can register themselves at import time).
#     """

#     print(f"Discovering models for model '{model_name}'...")

#     pkg_name = __name__.rsplit(".", 1)[0]

#     def _import_single(name: str) -> bool:
#         module_path = f"{pkg_name}.{name}"
#         try:
#             importlib.import_module(module_path)
#             return True
#         except Exception:
#             return False

#     if model_name:
#         base = str(model_name).split('_')[0]
#         _import_single(base)
#         return

#     # Default: import all modules under the package
#     try:
#         pkg = importlib.import_module(pkg_name)
#     except Exception:
#         return

#     failures = {}
#     modules = sorted(list(pkgutil.iter_modules(pkg.__path__)), key=lambda x: x.name)
#     for m in modules:
#         module_path = f"{pkg_name}.{m.name}"
#         try:
#             importlib.import_module(module_path)
#         except Exception as e:
#             # Record the failure and continue
#             import traceback
#             failures[m.name] = traceback.format_exc()
#             continue

#     if failures:
#         # Print a concise summary to help debugging — do not raise to keep discovery robust
#         print(f"Warning: some model modules failed to import ({len(failures)}): {sorted(failures.keys())}")
#         for name, tb in failures.items():
#             # Show only the first non-empty line from the traceback for brevity
#             first_line = tb.splitlines()[-1] if tb else "<no traceback>"
#             print(f"  - {name}: {first_line}")

#     # pkg_name = __name__  # "models"
#     # for m in pkgutil.iter_modules(__path__):
#     #     if m.ispkg:
#     #         importlib.import_module(f"{pkg_name}.{m.name}")

# def get_model(name: str):
#     return MODEL_REGISTRY.get(name)

# # __all__ = ["MODEL_REGISTRY", "discover_models", "get_model"]