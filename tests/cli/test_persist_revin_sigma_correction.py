"""Tests for ``_persist_resolved_revin_sigma_correction``.

The function resolves ``cfg.diffusion.revin_sigma_correction`` (normally a
``${revin_alpha:...}`` interpolation) and writes the literal value back to
both the cfg and the on-disk ``.hydra/config.yaml``. This makes the
checkpoint reload path agnostic to whether ``dataset_info`` was re-injected
— the RevIN σ scale correction always matches the value used at training
time.

See docs/agent_summaries/persist_revin_sigma_correction_2026-05-26.md for
the design rationale.
"""
from __future__ import annotations

import logging
import os
import time
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

# Importing the diffusion package registers the ${revin_alpha:...} resolver.
import graph_signal_diffusion.diffusion  # noqa: F401
from graph_signal_diffusion.cli.train import _persist_resolved_revin_sigma_correction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cfg(*, revin=True, blend_weight=0.7, sigma_mean=None, sigma_corr_literal=None):
    """Construct a minimal cfg that mirrors the post-Hydra-merge state."""
    diffusion = {"revin": revin, "revin_blend_weight": blend_weight}
    if sigma_corr_literal is not None:
        diffusion["revin_sigma_correction"] = sigma_corr_literal
    else:
        diffusion["revin_sigma_correction"] = (
            "${revin_alpha:${.revin_blend_weight},"
            "${oc.select:dataset_info.sigma_cond_mean_train,0.82}}"
        )
    cfg = OmegaConf.create({"diffusion": diffusion})
    if sigma_mean is not None:
        cfg.dataset_info = OmegaConf.create({"sigma_cond_mean_train": sigma_mean})
    return cfg


def _write_initial_hydra_yaml(tmp_path, cfg):
    """Mimic Hydra's startup-time save (interpolations preserved)."""
    hydra_dir = tmp_path / ".hydra"
    hydra_dir.mkdir(exist_ok=True)
    config_path = hydra_dir / "config.yaml"
    OmegaConf.save(cfg, str(config_path))
    return str(config_path)


def _hydra_stub(tmp_path):
    """Minimal HydraConfig-shaped object exposing runtime.output_dir."""
    return SimpleNamespace(runtime=SimpleNamespace(output_dir=str(tmp_path)))


# ---------------------------------------------------------------------------
# Core behavior
# ---------------------------------------------------------------------------

def test_materializes_resolved_alpha_in_cfg(tmp_path):
    """After the call, cfg.diffusion.revin_sigma_correction is a literal
    float matching the closed-form α(0.7, 0.8526)."""
    cfg = _make_cfg(sigma_mean=0.8526)
    _write_initial_hydra_yaml(tmp_path, cfg)

    _persist_resolved_revin_sigma_correction(
        cfg, _hydra_stub(tmp_path), logging.getLogger("test")
    )

    expected = 1.0 / (1.0 - 0.7 * (1.0 - 0.8526))
    assert float(cfg.diffusion.revin_sigma_correction) == pytest.approx(expected, rel=1e-6)
    # And the node is now a literal, not an interpolation.
    assert not OmegaConf.is_interpolation(cfg.diffusion, "revin_sigma_correction")


def test_writes_literal_to_hydra_config_yaml(tmp_path):
    """The on-disk .hydra/config.yaml has the literal, not ${revin_alpha:...}."""
    cfg = _make_cfg(sigma_mean=0.8526)
    config_path = _write_initial_hydra_yaml(tmp_path, cfg)

    # Before: file contains the interpolation string.
    with open(config_path) as f:
        content_before = f.read()
    assert "${revin_alpha" in content_before, \
        "test setup: initial yaml should hold the interpolation"

    _persist_resolved_revin_sigma_correction(
        cfg, _hydra_stub(tmp_path), logging.getLogger("test")
    )

    # After: file holds the literal value.
    with open(config_path) as f:
        content_after = f.read()
    assert "${revin_alpha" not in content_after, \
        f"interpolation still in saved yaml:\n{content_after}"
    saved = OmegaConf.load(config_path)
    expected = 1.0 / (1.0 - 0.7 * (1.0 - 0.8526))
    assert float(saved.diffusion.revin_sigma_correction) == pytest.approx(expected, rel=1e-6)


def test_falls_back_to_default_b_when_no_dataset_info(tmp_path):
    """If dataset_info isn't injected, the resolver falls back to 0.82 and
    the persist function materializes that fallback as a literal."""
    cfg = _make_cfg(sigma_mean=None)  # no dataset_info
    _write_initial_hydra_yaml(tmp_path, cfg)

    _persist_resolved_revin_sigma_correction(
        cfg, _hydra_stub(tmp_path), logging.getLogger("test")
    )

    expected_fallback = 1.0 / (1.0 - 0.7 * (1.0 - 0.82))
    assert float(cfg.diffusion.revin_sigma_correction) == pytest.approx(expected_fallback, rel=1e-6)


# ---------------------------------------------------------------------------
# No-op guards
# ---------------------------------------------------------------------------

def test_noop_when_revin_disabled(tmp_path):
    """When diffusion.revin=false, the function leaves cfg AND the file
    untouched."""
    cfg = _make_cfg(revin=False, sigma_mean=0.85)
    config_path = _write_initial_hydra_yaml(tmp_path, cfg)
    mtime_before = os.path.getmtime(config_path)

    time.sleep(0.05)  # ensure any rewrite would bump mtime
    _persist_resolved_revin_sigma_correction(
        cfg, _hydra_stub(tmp_path), logging.getLogger("test")
    )

    assert os.path.getmtime(config_path) == mtime_before, \
        "config.yaml was rewritten despite revin=false"
    assert OmegaConf.is_interpolation(cfg.diffusion, "revin_sigma_correction"), \
        "cfg field was materialized despite revin=false"


def test_skip_when_diffusion_missing(tmp_path):
    """When cfg has no diffusion section, the function is a clean no-op
    (no crash even if .hydra dir doesn't exist)."""
    cfg = OmegaConf.create({"trainer": {"max_epochs": 100}})
    _persist_resolved_revin_sigma_correction(
        cfg, _hydra_stub(tmp_path), logging.getLogger("test")
    )
    # No assertion needed — just must not raise.


def test_skip_when_revin_sigma_correction_missing(tmp_path):
    """When diffusion exists with revin=true but revin_sigma_correction
    field is missing (e.g., older config), the function is a clean no-op."""
    cfg = OmegaConf.create({
        "diffusion": {"revin": True, "revin_blend_weight": 1.0},
    })
    _persist_resolved_revin_sigma_correction(
        cfg, _hydra_stub(tmp_path), logging.getLogger("test")
    )
    # No assertion needed — just must not raise.


# ---------------------------------------------------------------------------
# Idempotence + already-literal handling
# ---------------------------------------------------------------------------

def test_idempotent_when_already_literal(tmp_path):
    """If revin_sigma_correction was explicitly overridden to a literal
    (no interpolation), the function should still produce a valid post-
    state — cfg + file both hold the same literal."""
    cfg = _make_cfg(sigma_corr_literal=1.5)
    _write_initial_hydra_yaml(tmp_path, cfg)

    _persist_resolved_revin_sigma_correction(
        cfg, _hydra_stub(tmp_path), logging.getLogger("test")
    )

    assert float(cfg.diffusion.revin_sigma_correction) == 1.5
    saved = OmegaConf.load(str(tmp_path / ".hydra" / "config.yaml"))
    assert float(saved.diffusion.revin_sigma_correction) == 1.5


def test_repeat_invocation_is_safe(tmp_path):
    """Calling the function twice produces the same state as calling once."""
    cfg = _make_cfg(sigma_mean=0.8526)
    _write_initial_hydra_yaml(tmp_path, cfg)

    _persist_resolved_revin_sigma_correction(
        cfg, _hydra_stub(tmp_path), logging.getLogger("test")
    )
    first_value = float(cfg.diffusion.revin_sigma_correction)
    _persist_resolved_revin_sigma_correction(
        cfg, _hydra_stub(tmp_path), logging.getLogger("test")
    )
    second_value = float(cfg.diffusion.revin_sigma_correction)

    assert first_value == second_value


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def test_emits_log_with_resolved_value(tmp_path, caplog):
    """The function logs the materialized α at INFO level."""
    cfg = _make_cfg(sigma_mean=0.8526)
    _write_initial_hydra_yaml(tmp_path, cfg)

    with caplog.at_level(logging.INFO):
        _persist_resolved_revin_sigma_correction(
            cfg, _hydra_stub(tmp_path), logging.getLogger("test")
        )

    # Some log line mentions revin_sigma_correction AND the resolved value.
    expected = 1.0 / (1.0 - 0.7 * (1.0 - 0.8526))
    matching = [
        rec for rec in caplog.records
        if "revin_sigma_correction" in rec.message
        and f"{expected:.4f}" in rec.message
    ]
    assert matching, \
        f"expected log message mentioning revin_sigma_correction and ~{expected:.4f}, got:\n" \
        + "\n".join(rec.message for rec in caplog.records)
