"""Regression test: _resolve_scalar_r_min_from_config must not silently
fall back to default=0.5 when constraint_profile.scalar.value is an
unresolved Hydra interpolation string like '${training.r_min}'.
"""

from graph_signal_diffusion.cli.wra.build_diffusion_dataset import (
    _resolve_scalar_r_min_from_config,
)


def _make_config(r_min: float = 0.6, scalar_value=None):
    """Build a minimal config dict matching the structure of a saved Hydra config."""
    if scalar_value is None:
        scalar_value = r_min
    return {
        "training": {
            "r_min": r_min,
            "constraint_profile": {
                "source": "scalar",
                "scalar": {"value": scalar_value},
            },
        },
    }


def test_resolved_value():
    """When scalar.value is already a number, it should be returned directly."""
    config = _make_config(r_min=0.6, scalar_value=0.6)
    assert _resolve_scalar_r_min_from_config(config) == 0.6


def test_unresolved_hydra_interpolation_falls_back_to_training_r_min():
    """When scalar.value is an unresolved Hydra interpolation string,
    the function must fall back to training.r_min (0.6), NOT the default (0.5)."""
    config = _make_config(r_min=0.6, scalar_value="${training.r_min}")
    result = _resolve_scalar_r_min_from_config(config)
    assert result == 0.6, f"Expected 0.6 but got {result} (likely fell back to default=0.5)"


def test_unresolved_interpolation_top_level_fallback():
    """If training.r_min is also missing, fall back to top-level config r_min."""
    config = {
        "r_min": 0.7,
        "training": {
            "constraint_profile": {
                "source": "scalar",
                "scalar": {"value": "${training.r_min}"},
            },
        },
    }
    result = _resolve_scalar_r_min_from_config(config)
    assert result == 0.7, f"Expected 0.7 but got {result}"


def test_all_missing_returns_default():
    """If no r_min is found anywhere, return the default."""
    config = {
        "training": {
            "constraint_profile": {
                "source": "scalar",
                "scalar": {"value": "${training.r_min}"},
            },
        },
    }
    result = _resolve_scalar_r_min_from_config(config, default=0.5)
    assert result == 0.5


def test_non_scalar_source_with_interpolation():
    """Non-scalar source with an unresolved interpolation in training.r_min."""
    config = {
        "training": {
            "r_min": "${some.other.ref}",
            "constraint_profile": {
                "source": "sampled",
            },
        },
        "r_min": 0.8,
    }
    result = _resolve_scalar_r_min_from_config(config)
    assert result == 0.8, f"Expected 0.8 but got {result}"
