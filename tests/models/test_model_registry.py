"""Tests for model registry discovery behavior."""
import importlib


def test_discover_models_imports_only_target():
    """Ensure discover_models('ugnn') imports and registers UGNN but does not import/register UNet."""
    # Reload the registry module to get a fresh MODEL_REGISTRY for the test
    reg_mod = importlib.reload(__import__('graph_signal_diffusion.models.registry', fromlist=['']))
    MODEL_REGISTRY = reg_mod.MODEL_REGISTRY

    # Start from a clean slate (best-effort); no 'ugnn' or 'unet' should be registered
    keys_before = set(MODEL_REGISTRY.keys())

    assert 'ugnn' not in keys_before

    # Import only 'ugnn' module via the simplified discover_models
    reg_mod.discover_models('ugnn')

    keys_after = set(MODEL_REGISTRY.keys())

    # UGNN should now be registered
    assert 'ugnn' in keys_after, f"Expected 'ugnn' to be registered; registry keys: {sorted(keys_after)}"

    # UNet should NOT have been newly registered by this operation
    newly_registered = keys_after - keys_before
    assert 'unet' not in newly_registered, (
        "discover_models('ugnn') unexpectedly caused 'unet' to register; "
        f"newly registered keys: {sorted(newly_registered)}"
    )


def test_discover_models_none_registers_all():
    """discover_models(None) should import and register all model modules."""
    reg_mod = importlib.reload(__import__('graph_signal_diffusion.models.registry', fromlist=['']))
    MODEL_REGISTRY = reg_mod.MODEL_REGISTRY

    # Start clean
    keys_before = set(MODEL_REGISTRY.keys())

    # Call discover_models with None to import all model modules
    reg_mod.discover_models(None)

    keys_after = set(MODEL_REGISTRY.keys())

    # Expect both 'ugnn' and 'unet' to be present after importing all modules
    assert 'ugnn' in keys_after, f"'ugnn' not registered after discover_models(None): {sorted(keys_after)}"
    assert 'unet' in keys_after, f"'unet' not registered after discover_models(None): {sorted(keys_after)}"




if __name__ == "__main__":
    test_discover_models_imports_only_target()
    print("test_discover_models_imports_only_target passed.")

    test_discover_models_none_registers_all()
    print("test_discover_models_none_registers_all passed.")
