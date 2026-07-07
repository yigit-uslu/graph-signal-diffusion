"""Unit tests for the post-hoc variance correction feature.

Covers:
  1. ``StockPriceForecastingTaskV2.apply_variance_correction`` math
     (no-op modes, scalar rescale, per-horizon rescale, ensemble mean
     preservation, shape-contract violation, missing metadata key).
  2. ``DiffusionBaseline._parse_variance_correction_config`` validation
     (mode whitelist, missing-alpha errors, dict + OmegaConf inputs,
     YAML 1.1 off → bool coercion).
  3. ``DiffusionBaseline._maybe_promote_alpha_ema`` checkpoint-embedded
     auto-promotion (legacy checkpoints stay mode='off'; CLI override
     always wins; tensor and list payloads both supported).

Design + rationale: see
``docs/agent_summaries/variance_correction_plan_2026-05-20.md``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf

from graph_signal_diffusion.baselines.diffusion.baseline import (
    DiffusionBaseline,
    VarianceCorrectionConfig,
)
from graph_signal_diffusion.tasks.stock_price_forecasting.evaluator import (
    StockPriceForecastingTaskV2,
)


# ── Fixtures ───────────────────────────────────────────────────────────


def _make_task() -> StockPriceForecastingTaskV2:
    """Construct a minimal V2 task instance for math tests.

    No dataset_info / target_destandardization injection — the rescale
    operates purely on standardized tensors.
    """
    return StockPriceForecastingTaskV2(
        forecast_horizon=5,
        n_samples_per_input=4,
    )


def _make_gaussian_ensemble(
    B: int, n: int, T: int, N: int, F: int = 1, seed: int = 0,
) -> torch.Tensor:
    """``[B*n, T, N, F]`` Gaussian ensemble with stable seed."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B * n, T, N, F, generator=g)


def _ensemble_view(samples: torch.Tensor, B: int, n: int) -> torch.Tensor:
    """Reshape ``[B*n, T, N, F] → [B, n, T, N, F]``."""
    Bn, T, N, F = samples.shape
    assert Bn == B * n, f"Bn={Bn} != B*n={B*n}"
    return samples.view(B, n, T, N, F)


# ── apply_variance_correction math ─────────────────────────────────────


def test_mode_off_is_bitwise_noop():
    """``mode=off`` returns the input tensor with bitwise equality."""
    task = _make_task()
    samples = _make_gaussian_ensemble(B=3, n=4, T=5, N=7)
    metadata = {"n_samples_per_input": 4}

    task.variance_correction = VarianceCorrectionConfig(mode="off")
    out = task.apply_variance_correction(samples, metadata)
    assert torch.equal(out, samples), "mode=off must be bitwise no-op"


def test_variance_correction_none_field_is_noop():
    """``self.variance_correction is None`` → return input unchanged."""
    task = _make_task()
    samples = _make_gaussian_ensemble(B=2, n=4, T=5, N=3)
    metadata = {"n_samples_per_input": 4}

    task.variance_correction = None
    out = task.apply_variance_correction(samples, metadata)
    assert torch.equal(out, samples)


def test_scalar_alpha_1_is_numerical_noop():
    """``mode=scalar, alpha=1.0`` is a no-op modulo fp roundoff."""
    task = _make_task()
    samples = _make_gaussian_ensemble(B=3, n=4, T=5, N=7)
    metadata = {"n_samples_per_input": 4}

    task.variance_correction = VarianceCorrectionConfig(
        mode="scalar", alpha=1.0,
    )
    out = task.apply_variance_correction(samples, metadata)
    assert torch.allclose(out, samples, atol=1e-6, rtol=1e-6)


def test_scalar_alpha_2_doubles_ensemble_spread():
    """At α=2.0, per-window per-stock per-horizon ensemble std doubles
    and ensemble mean is preserved exactly.
    """
    task = _make_task()
    B, n, T, N = 5, 8, 6, 4
    samples = _make_gaussian_ensemble(B=B, n=n, T=T, N=N, seed=42)
    metadata = {"n_samples_per_input": n}

    task.variance_correction = VarianceCorrectionConfig(
        mode="scalar", alpha=2.0,
    )
    out = task.apply_variance_correction(samples, metadata)

    pre = _ensemble_view(samples, B, n)   # [B, n, T, N, 1]
    post = _ensemble_view(out, B, n)

    # Per-window ensemble mean preserved.
    assert torch.allclose(
        post.mean(dim=1), pre.mean(dim=1), atol=1e-6, rtol=1e-6,
    ), "ensemble mean must be preserved under rescale"

    # Per-window ensemble std doubles.
    std_pre = pre.std(dim=1, unbiased=False)    # [B, T, N, 1]
    std_post = post.std(dim=1, unbiased=False)
    assert torch.allclose(std_post, 2.0 * std_pre, atol=1e-5, rtol=1e-5), (
        "ensemble std must scale by alpha=2.0"
    )


def test_per_horizon_alpha_unit_vector_is_noop():
    """``mode=per_horizon, alpha_per_horizon=[1]*T`` is a no-op."""
    task = _make_task()
    samples = _make_gaussian_ensemble(B=3, n=4, T=5, N=7)
    metadata = {"n_samples_per_input": 4}

    task.variance_correction = VarianceCorrectionConfig(
        mode="per_horizon", alpha_per_horizon=[1.0] * 5,
    )
    out = task.apply_variance_correction(samples, metadata)
    assert torch.allclose(out, samples, atol=1e-6, rtol=1e-6)


def test_per_horizon_alpha_scales_per_timestep():
    """Per-horizon alpha scales each timestep's ensemble std independently."""
    task = _make_task()
    B, n, T, N = 3, 8, 5, 4
    samples = _make_gaussian_ensemble(B=B, n=n, T=T, N=N, seed=1)
    metadata = {"n_samples_per_input": n}

    alphas = [1.0, 2.0, 1.0, 2.0, 1.0]
    task.variance_correction = VarianceCorrectionConfig(
        mode="per_horizon", alpha_per_horizon=alphas,
    )
    out = task.apply_variance_correction(samples, metadata)

    pre = _ensemble_view(samples, B, n)
    post = _ensemble_view(out, B, n)

    # Mean preserved at every horizon.
    assert torch.allclose(
        post.mean(dim=1), pre.mean(dim=1), atol=1e-6, rtol=1e-6,
    )

    # Per-horizon std scales by per-horizon alpha.
    std_pre = pre.std(dim=1, unbiased=False)   # [B, T, N, 1]
    std_post = post.std(dim=1, unbiased=False)
    alpha_tensor = torch.tensor(alphas).view(1, T, 1, 1)
    assert torch.allclose(
        std_post, alpha_tensor * std_pre, atol=1e-5, rtol=1e-5,
    )


def test_per_horizon_wrong_length_raises():
    """``len(alpha_per_horizon) != T`` raises ValueError at evaluate-time."""
    task = _make_task()
    samples = _make_gaussian_ensemble(B=2, n=4, T=5, N=3)
    metadata = {"n_samples_per_input": 4}

    task.variance_correction = VarianceCorrectionConfig(
        mode="per_horizon",
        alpha_per_horizon=[1.0, 2.0, 1.0],   # len=3, but T=5
    )
    with pytest.raises(ValueError, match="alpha_per_horizon has length"):
        task.apply_variance_correction(samples, metadata)


def test_bn_not_divisible_by_n_raises():
    """Shape contract: ``Bn % n != 0`` raises ValueError."""
    task = _make_task()
    samples = torch.randn(7, 5, 3, 1)  # Bn=7, not divisible by n=4
    metadata = {"n_samples_per_input": 4}

    task.variance_correction = VarianceCorrectionConfig(
        mode="scalar", alpha=1.5,
    )
    with pytest.raises(ValueError, match="not divisible by"):
        task.apply_variance_correction(samples, metadata)


def test_missing_n_samples_per_input_raises():
    """Missing metadata key with ``mode != off`` is a hard error."""
    task = _make_task()
    samples = _make_gaussian_ensemble(B=2, n=4, T=5, N=3)

    task.variance_correction = VarianceCorrectionConfig(
        mode="scalar", alpha=1.5,
    )
    with pytest.raises(ValueError, match="n_samples_per_input"):
        task.apply_variance_correction(samples, metadata={})


def test_invalid_n_samples_per_input_zero_raises():
    """``n_samples_per_input < 1`` is a hard error (no silent fallback)."""
    task = _make_task()
    samples = _make_gaussian_ensemble(B=2, n=4, T=5, N=3)

    task.variance_correction = VarianceCorrectionConfig(
        mode="scalar", alpha=1.5,
    )
    with pytest.raises(ValueError, match="n_samples_per_input"):
        task.apply_variance_correction(samples, metadata={"n_samples_per_input": 0})


def test_n_equals_1_collapses_to_zero_variance():
    """When n=1 the ensemble has a single member; the rescaled output
    equals the input exactly (no spread to scale).
    """
    task = _make_task()
    samples = _make_gaussian_ensemble(B=4, n=1, T=5, N=3)
    metadata = {"n_samples_per_input": 1}

    task.variance_correction = VarianceCorrectionConfig(
        mode="scalar", alpha=2.0,
    )
    out = task.apply_variance_correction(samples, metadata)
    # Per-window ensemble = single sample; mean = self, deviation = 0.
    assert torch.allclose(out, samples, atol=1e-6, rtol=1e-6)


def test_dtype_and_device_preserved():
    """Output preserves input ``dtype`` and ``device``."""
    task = _make_task()
    samples = _make_gaussian_ensemble(B=2, n=4, T=5, N=3).to(torch.float64)
    metadata = {"n_samples_per_input": 4}

    task.variance_correction = VarianceCorrectionConfig(
        mode="scalar", alpha=1.5,
    )
    out = task.apply_variance_correction(samples, metadata)
    assert out.dtype == samples.dtype
    assert out.device == samples.device
    assert out.shape == samples.shape


# ── pivot=zero (global rescale) ────────────────────────────────────────


def test_pivot_zero_mode_off_is_bitwise_noop():
    """``pivot=zero, mode=off`` still no-op."""
    task = _make_task()
    samples = _make_gaussian_ensemble(B=3, n=4, T=5, N=7)
    task.variance_correction = VarianceCorrectionConfig(mode="off", pivot="zero")
    out = task.apply_variance_correction(samples, metadata={})
    assert torch.equal(out, samples)


def test_pivot_zero_scalar_alpha_2_multiplies_globally():
    """``pivot=zero, alpha=2.0`` simply multiplies the tensor by 2."""
    task = _make_task()
    samples = _make_gaussian_ensemble(B=3, n=4, T=5, N=7, seed=11)
    task.variance_correction = VarianceCorrectionConfig(
        mode="scalar", alpha=2.0, pivot="zero",
    )
    out = task.apply_variance_correction(samples, metadata={})
    assert torch.allclose(out, 2.0 * samples, atol=1e-6, rtol=1e-6)


def test_pivot_zero_does_not_need_n_samples_per_input():
    """``pivot=zero`` must work even when n_samples_per_input is absent."""
    task = _make_task()
    samples = _make_gaussian_ensemble(B=3, n=4, T=5, N=7)
    task.variance_correction = VarianceCorrectionConfig(
        mode="scalar", alpha=1.5, pivot="zero",
    )
    # Intentionally missing the key; should NOT raise.
    out = task.apply_variance_correction(samples, metadata={})
    assert torch.allclose(out, 1.5 * samples, atol=1e-6, rtol=1e-6)


def test_pivot_zero_alpha_1_is_numerical_noop():
    task = _make_task()
    samples = _make_gaussian_ensemble(B=3, n=4, T=5, N=7)
    task.variance_correction = VarianceCorrectionConfig(
        mode="scalar", alpha=1.0, pivot="zero",
    )
    out = task.apply_variance_correction(samples, metadata={})
    assert torch.allclose(out, samples, atol=1e-6, rtol=1e-6)


def test_pivot_zero_per_horizon_scales_per_timestep_globally():
    """``pivot=zero, per_horizon`` scales each horizon by its own α with no mean preservation."""
    task = _make_task()
    B, n, T, N = 2, 4, 5, 3
    samples = _make_gaussian_ensemble(B=B, n=n, T=T, N=N, seed=7)
    alphas = [1.0, 2.0, 1.0, 2.0, 1.0]
    task.variance_correction = VarianceCorrectionConfig(
        mode="per_horizon", alpha_per_horizon=alphas, pivot="zero",
    )
    out = task.apply_variance_correction(samples, metadata={})
    alpha_tensor = torch.tensor(alphas).view(1, T, 1, 1)
    expected = alpha_tensor * samples
    assert torch.allclose(out, expected, atol=1e-5, rtol=1e-5)


def test_pivot_zero_preserves_global_marginal_mean_when_centered():
    """If samples are already zero-mean, pivot=zero with α=1.5 keeps mean=0
    (because there's nothing to shift around)."""
    task = _make_task()
    g = torch.Generator().manual_seed(42)
    samples = torch.randn(8, 5, 4, 1, generator=g)
    samples = samples - samples.mean()   # center
    task.variance_correction = VarianceCorrectionConfig(
        mode="scalar", alpha=1.5, pivot="zero",
    )
    out = task.apply_variance_correction(samples, metadata={})
    assert abs(out.mean().item()) < 1e-5
    # Spread scaled by 1.5.
    assert torch.allclose(out.std(), 1.5 * samples.std(), atol=1e-5, rtol=1e-5)


def test_pivot_zero_structural_invariance_correlation_along_t():
    """Under pivot=zero, the lag-1 autocorrelation of x² along t is
    EXACTLY preserved (textbook scale invariance of γ_k/γ_0 under global α).
    """
    g = torch.Generator().manual_seed(0)
    B, n, T, N = 4, 3, 10, 2     # T=10 for a less-noisy estimate
    # Construct an AR(1)-like trajectory with non-trivial autocorr.
    x = torch.zeros(B * n, T, N, 1)
    eps = torch.randn(B * n, T, N, 1, generator=g)
    rho = 0.7
    x[:, 0] = eps[:, 0]
    for t in range(1, T):
        x[:, t] = rho * x[:, t - 1] + eps[:, t]

    task = _make_task()
    task.variance_correction = VarianceCorrectionConfig(
        mode="scalar", alpha=2.5, pivot="zero",
    )
    out = task.apply_variance_correction(x, metadata={})

    # Compute lag-1 autocorr of x² along t for both tensors.
    def _lag1_autocorr_r2(t):
        r2 = (t ** 2).squeeze(-1)              # [B*n, T, N]
        mu = r2.mean(dim=1, keepdim=True)
        d = r2 - mu
        gamma_0 = (d ** 2).mean(dim=1)
        gamma_1 = (d[:, :-1] * d[:, 1:]).mean(dim=1)
        rho_1 = gamma_1 / (gamma_0 + 1e-12)
        return rho_1.mean().item()

    ac_pre = _lag1_autocorr_r2(x)
    ac_post = _lag1_autocorr_r2(out)
    # Strict invariance: γ_k(α²x²)/γ_0(α²x²) = α⁴γ_k(x²) / α⁴γ_0(x²) = ρ_k(x²).
    assert abs(ac_pre - ac_post) < 1e-5, (
        f"pivot=zero should preserve lag-1 autocorr of r² exactly; "
        f"pre={ac_pre:.6f}, post={ac_post:.6f}"
    )


def test_pivot_ensemble_mean_default_is_unchanged():
    """Without setting pivot, default is ensemble_mean (backward compat)."""
    task = _make_task()
    samples = _make_gaussian_ensemble(B=2, n=4, T=5, N=3)
    metadata = {"n_samples_per_input": 4}

    task.variance_correction = VarianceCorrectionConfig(
        mode="scalar", alpha=2.0,  # pivot defaults to ensemble_mean
    )
    out = task.apply_variance_correction(samples, metadata)
    # Should match the prior ensemble-mean behavior — mean preserved per (b,t,s).
    pre = _ensemble_view(samples, 2, 4)
    post = _ensemble_view(out, 2, 4)
    assert torch.allclose(post.mean(dim=1), pre.mean(dim=1), atol=1e-6, rtol=1e-6)


def test_pivot_invalid_raises():
    task = _make_task()
    samples = _make_gaussian_ensemble(B=2, n=4, T=5, N=3)
    metadata = {"n_samples_per_input": 4}
    # Set up manually since the parser would reject this earlier; here we
    # exercise the apply_variance_correction defensive check.
    task.variance_correction = VarianceCorrectionConfig(
        mode="scalar", alpha=1.5, pivot="some_other_pivot",
    )
    with pytest.raises(ValueError, match="pivot must be"):
        task.apply_variance_correction(samples, metadata)


# ── _parse_variance_correction_config validation ───────────────────────


def test_parse_default_when_none():
    """Default config when raw is None: mode='off', no validation errors."""
    cfg = DiffusionBaseline._parse_variance_correction_config(None)
    assert cfg.mode == "off"
    assert cfg.alpha is None
    assert cfg.alpha_per_horizon is None
    assert cfg.source == "default"


def test_parse_dict_scalar_mode():
    raw = {"mode": "scalar", "alpha": 1.5}
    cfg = DiffusionBaseline._parse_variance_correction_config(raw)
    assert cfg.mode == "scalar"
    assert cfg.alpha == 1.5
    assert cfg.source == "cli"


def test_parse_omegaconf_dictconfig():
    """OmegaConf DictConfig inputs are accepted (path used by hydra)."""
    raw = OmegaConf.create({"mode": "scalar", "alpha": 1.5})
    cfg = DiffusionBaseline._parse_variance_correction_config(raw)
    assert cfg.mode == "scalar"
    assert cfg.alpha == 1.5


def test_parse_off_string_stays_off():
    raw = {"mode": "off", "alpha": None, "alpha_per_horizon": None}
    cfg = DiffusionBaseline._parse_variance_correction_config(raw)
    assert cfg.mode == "off"


def test_parse_yaml11_off_bool_coerced_to_off():
    """YAML 1.1 maps unquoted off/no/false → bool False; coerce defensively."""
    raw = {"mode": False, "alpha": None}
    cfg = DiffusionBaseline._parse_variance_correction_config(raw)
    assert cfg.mode == "off"


def test_parse_invalid_mode_raises():
    raw = {"mode": "linear", "alpha": 1.5}
    with pytest.raises(ValueError, match="mode must be one of"):
        DiffusionBaseline._parse_variance_correction_config(raw)


def test_parse_scalar_missing_alpha_raises():
    raw = {"mode": "scalar", "alpha": None}
    with pytest.raises(ValueError, match="mode='scalar' requires"):
        DiffusionBaseline._parse_variance_correction_config(raw)


def test_parse_per_horizon_missing_list_raises():
    raw = {"mode": "per_horizon", "alpha_per_horizon": None}
    with pytest.raises(ValueError, match="mode='per_horizon' requires"):
        DiffusionBaseline._parse_variance_correction_config(raw)


def test_parse_per_horizon_empty_list_raises():
    raw = {"mode": "per_horizon", "alpha_per_horizon": []}
    with pytest.raises(ValueError, match="must be non-empty"):
        DiffusionBaseline._parse_variance_correction_config(raw)


def test_parse_per_horizon_floats_coerced():
    raw = {"mode": "per_horizon", "alpha_per_horizon": [1.0, 1.5, 2.0]}
    cfg = DiffusionBaseline._parse_variance_correction_config(raw)
    assert cfg.alpha_per_horizon == [1.0, 1.5, 2.0]


def test_parse_off_ignores_alpha():
    """When mode='off', alpha/alpha_per_horizon are discarded."""
    raw = {"mode": "off", "alpha": 1.5, "alpha_per_horizon": [1.0, 2.0]}
    cfg = DiffusionBaseline._parse_variance_correction_config(raw)
    assert cfg.mode == "off"
    assert cfg.alpha is None
    assert cfg.alpha_per_horizon is None


def test_parse_pivot_default_is_ensemble_mean():
    raw = {"mode": "scalar", "alpha": 1.5}
    cfg = DiffusionBaseline._parse_variance_correction_config(raw)
    assert cfg.pivot == "ensemble_mean"


def test_parse_pivot_zero():
    raw = {"mode": "scalar", "alpha": 1.5, "pivot": "zero"}
    cfg = DiffusionBaseline._parse_variance_correction_config(raw)
    assert cfg.pivot == "zero"


def test_parse_pivot_invalid_raises():
    raw = {"mode": "scalar", "alpha": 1.5, "pivot": "median"}
    with pytest.raises(ValueError, match="pivot must be"):
        DiffusionBaseline._parse_variance_correction_config(raw)


def test_parse_pivot_case_insensitive():
    raw = {"mode": "scalar", "alpha": 1.5, "pivot": "ZERO"}
    cfg = DiffusionBaseline._parse_variance_correction_config(raw)
    assert cfg.pivot == "zero"


# ── _maybe_promote_alpha_ema (checkpoint-embedded) ─────────────────────


def _make_baseline_stub(
    initial_vc: VarianceCorrectionConfig,
) -> "_BaselineStub":
    """Minimal stub exposing the two attributes/methods used by
    _maybe_promote_alpha_ema, so we can test it without going through
    the full pipeline build.
    """
    stub = SimpleNamespace()
    stub.variance_correction = initial_vc
    stub._maybe_promote_alpha_ema = (
        DiffusionBaseline._maybe_promote_alpha_ema.__get__(stub)
    )
    return stub


def test_alpha_ema_promotion_from_tensor_when_off(tmp_path: Path):
    """Checkpoint with ``alpha_ema`` tensor + mode=off → promotion."""
    ckpt_path = tmp_path / "model.pt"
    # Use float64 so the tensor → list roundtrip is exact (float32 would
    # drop 1.4 → 1.399999976...).
    expected = [1.4, 1.5, 1.6, 1.5, 1.4]
    torch.save(
        {"epoch": 100, "alpha_ema": torch.tensor(expected, dtype=torch.float64)},
        ckpt_path,
    )

    stub = _make_baseline_stub(VarianceCorrectionConfig(mode="off"))
    stub._maybe_promote_alpha_ema(str(ckpt_path))

    assert stub.variance_correction.mode == "per_horizon"
    assert stub.variance_correction.alpha_per_horizon == pytest.approx(expected)
    assert stub.variance_correction.source == "checkpoint_alpha_ema"


def test_alpha_ema_promotion_from_list_payload(tmp_path: Path):
    """Checkpoint with ``alpha_ema`` as plain list → also promoted."""
    ckpt_path = tmp_path / "model.pt"
    torch.save({"epoch": 100, "alpha_ema": [1.2, 1.3]}, ckpt_path)

    stub = _make_baseline_stub(VarianceCorrectionConfig(mode="off"))
    stub._maybe_promote_alpha_ema(str(ckpt_path))

    assert stub.variance_correction.mode == "per_horizon"
    assert stub.variance_correction.alpha_per_horizon == [1.2, 1.3]


def test_old_checkpoint_without_alpha_ema_stays_off(tmp_path: Path):
    """Legacy checkpoint (no alpha_ema key) → mode stays 'off'."""
    ckpt_path = tmp_path / "old.pt"
    torch.save({"epoch": 100, "model": {}}, ckpt_path)

    stub = _make_baseline_stub(VarianceCorrectionConfig(mode="off"))
    stub._maybe_promote_alpha_ema(str(ckpt_path))

    assert stub.variance_correction.mode == "off"
    assert stub.variance_correction.alpha_per_horizon is None


def test_cli_override_blocks_alpha_ema_promotion(tmp_path: Path):
    """CLI override (mode != off) wins; alpha_ema is ignored."""
    ckpt_path = tmp_path / "model.pt"
    torch.save(
        {"epoch": 100, "alpha_ema": torch.tensor([1.4, 1.5, 1.6, 1.5, 1.4])},
        ckpt_path,
    )

    cli_cfg = VarianceCorrectionConfig(
        mode="scalar", alpha=1.5, source="cli",
    )
    stub = _make_baseline_stub(cli_cfg)
    stub._maybe_promote_alpha_ema(str(ckpt_path))

    # Unchanged: CLI wins.
    assert stub.variance_correction.mode == "scalar"
    assert stub.variance_correction.alpha == 1.5
    assert stub.variance_correction.source == "cli"


def test_alpha_ema_empty_payload_ignored(tmp_path: Path):
    """Empty alpha_ema list → ignored with a warning, mode stays 'off'."""
    ckpt_path = tmp_path / "model.pt"
    torch.save({"epoch": 100, "alpha_ema": []}, ckpt_path)

    stub = _make_baseline_stub(VarianceCorrectionConfig(mode="off"))
    stub._maybe_promote_alpha_ema(str(ckpt_path))

    assert stub.variance_correction.mode == "off"


def test_alpha_ema_unparseable_payload_ignored(tmp_path: Path):
    """Non-numeric alpha_ema payload → ignored with a warning."""
    ckpt_path = tmp_path / "model.pt"
    torch.save({"epoch": 100, "alpha_ema": "not a tensor"}, ckpt_path)

    stub = _make_baseline_stub(VarianceCorrectionConfig(mode="off"))
    stub._maybe_promote_alpha_ema(str(ckpt_path))

    # Unparseable string is iterable char-by-char; that's coerced to floats
    # only if char-values are numeric. "not a tensor" fails float(c), so
    # we hit the except → mode stays 'off'. Both outcomes (unparseable and
    # ignored) are acceptable, but specifically check mode is still off.
    assert stub.variance_correction.mode == "off"
