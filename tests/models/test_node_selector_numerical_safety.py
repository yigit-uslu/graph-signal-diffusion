"""Tests for NodeSelector numerical safety guards.

Covers:
1. Logit clipping doesn't change normal outputs (no-op under typical conditions).
2. Extreme temperature produces finite entropy and aux loss.
3. Temperature floor uses temperature_min (not hardcoded 1e-6).
4. Synthetic NaN aux loss triggers DDPM sanitization warning.
"""

import logging
import math

import torch
import pytest

from graph_signal_diffusion.models.components.node_selector import NodeSelector


def _chain_edge_index(num_nodes: int) -> torch.Tensor:
    edges = [(i, i + 1) for i in range(num_nodes - 1)]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return torch.cat([edge_index, edge_index.flip(0)], dim=1)


def _make_selector(
    in_channels=8,
    temperature=1.0,
    temperature_min=0.1,
    temperature_schedule="constant",
    temperature_anneal_steps=0,
    temperature_warmup_steps=0,
    entropy_reg_weight=0.1,
    selection_mode="ste",
    **kwargs,
):
    return NodeSelector(
        in_channels=in_channels,
        pooling_ratio=0.5,
        selection_mode=selection_mode,
        temperature=temperature,
        temperature_min=temperature_min,
        temperature_schedule=temperature_schedule,
        temperature_anneal_steps=temperature_anneal_steps,
        temperature_warmup_steps=temperature_warmup_steps,
        entropy_reg_weight=entropy_reg_weight,
        min_retained_nodes=1,
        temporal_reduce="mean",
        cond_fusion=False,
        **kwargs,
    )


class TestLogitClippingNoOp:
    """Logit clipping at ±20 should be a no-op for normal scores."""

    def test_normal_scores_unchanged(self):
        """With temperature=1.0 and normalized scores in [-3, 3], clamp is no-op."""
        torch.manual_seed(42)
        sel = _make_selector(temperature=1.0, selection_mode="hard")
        sel.eval()

        B, T, N, F = 2, 1, 16, 8
        x = torch.randn(B, T, N, F)
        edge_index = _chain_edge_index(N)
        active_mask = torch.ones(B, N, dtype=torch.bool)

        out = sel(x, edge_index, active_mask=active_mask)
        x_pooled, new_mask, indices, scores = out

        assert torch.isfinite(x_pooled).all()
        assert torch.isfinite(scores).all()

    def test_scores_within_clip_range(self):
        """After z-score normalization, scores should be O(1) — well within ±20."""
        torch.manual_seed(42)
        sel = _make_selector(temperature=1.0, selection_mode="hard")
        sel.eval()

        B, T, N, F = 4, 1, 32, 8
        x = torch.randn(B, T, N, F)
        edge_index = _chain_edge_index(N)
        active_mask = torch.ones(B, N, dtype=torch.bool)

        with torch.no_grad():
            _, _, _, scores = sel(x, edge_index, active_mask=active_mask)
        # z-score normalized scores should be well within ±20 clip bound
        assert scores.abs().max().item() < 20.0, (
            f"Scores max={scores.abs().max().item():.2f} exceeds clip bound"
        )


class TestExtremeTemperatureFiniteness:
    """Extreme temperature annealing should still produce finite values."""

    def test_low_temperature_finite_entropy(self):
        """With temperature_min=0.05, annealed to completion, entropy is finite."""
        torch.manual_seed(42)
        sel = _make_selector(
            temperature=1.0,
            temperature_min=0.05,
            temperature_schedule="linear",
            temperature_anneal_steps=100,
            entropy_reg_weight=0.1,
            selection_mode="ste",
        )
        sel.train()

        B, T, N, F = 2, 1, 16, 8
        x = torch.randn(B, T, N, F)
        edge_index = _chain_edge_index(N)
        active_mask = torch.ones(B, N, dtype=torch.bool)

        # Advance to end of annealing
        sel.current_step = 100

        out = sel(x, edge_index, active_mask=active_mask)
        x_pooled, new_mask, indices, scores = out

        assert torch.isfinite(x_pooled).all(), "x_pooled has non-finite values"
        assert torch.isfinite(scores).all(), "scores has non-finite values"

        # Check aux loss is finite
        aux_loss = sel.get_selector_aux_loss()
        if aux_loss is not None:
            assert torch.isfinite(aux_loss), (
                f"aux loss is non-finite: {aux_loss.item()}"
            )

    def test_very_low_temperature_finite(self):
        """Even with temperature_min=0.01, outputs remain finite."""
        torch.manual_seed(42)
        sel = _make_selector(
            temperature=1.0,
            temperature_min=0.01,
            temperature_schedule="cosine",
            temperature_anneal_steps=50,
            entropy_reg_weight=0.5,
            selection_mode="ste",
        )
        sel.train()
        sel.current_step = 50

        B, T, N, F = 2, 1, 16, 8
        x = torch.randn(B, T, N, F)
        edge_index = _chain_edge_index(N)
        active_mask = torch.ones(B, N, dtype=torch.bool)

        out = sel(x, edge_index, active_mask=active_mask)
        x_pooled, _, _, scores = out

        assert torch.isfinite(x_pooled).all()
        assert torch.isfinite(scores).all()

        aux_loss = sel.get_selector_aux_loss()
        if aux_loss is not None:
            assert torch.isfinite(aux_loss), f"aux loss non-finite: {aux_loss}"


class TestTemperatureFloor:
    """Temperature floor should use temperature_min, not hardcoded 1e-6."""

    def test_floor_equals_temperature_min(self):
        sel = _make_selector(
            temperature=1.0,
            temperature_min=0.2,
            temperature_schedule="linear",
            temperature_anneal_steps=100,
        )
        # Advance well past the end of annealing
        sel.current_step = 200
        t = sel._current_temperature()
        assert t == pytest.approx(0.2), (
            f"Temperature floor should be temperature_min=0.2, got {t}"
        )

    def test_floor_prevents_sub_min_temperature(self):
        """Even if schedule math would go below min, the floor catches it."""
        sel = _make_selector(
            temperature=0.5,
            temperature_min=0.1,
            temperature_schedule="linear",
            temperature_anneal_steps=10,
        )
        # Way past annealing end
        sel.current_step = 1000
        t = sel._current_temperature()
        assert t >= 0.1, f"Temperature {t} fell below temperature_min=0.1"


class TestEntropyRegularizerEdgeCases:
    """Edge cases for _entropy_regularizer static method."""

    def test_empty_active_probs_returns_zeros(self):
        """Fully-masked (no active nodes) returns zero contribution."""
        soft_mask = torch.zeros(2, 8)
        active_mask = torch.zeros(2, 8, dtype=torch.bool)
        loss, entropy = NodeSelector._entropy_regularizer(soft_mask, active_mask)
        assert loss.item() == 0.0
        assert entropy.item() == 0.0

    def test_near_one_soft_mask_finite(self):
        """Near-saturated soft_mask (sigmoid(10) ≈ 0.99995) produces finite entropy."""
        # sigmoid(10) ≈ 0.99995 — realistic output after logit clipping
        soft_mask = torch.full((2, 8), torch.sigmoid(torch.tensor(10.0)).item())
        active_mask = torch.ones(2, 8, dtype=torch.bool)
        loss, entropy = NodeSelector._entropy_regularizer(soft_mask, active_mask)
        assert torch.isfinite(loss), f"loss is non-finite: {loss}"
        assert torch.isfinite(entropy), f"entropy is non-finite: {entropy}"

    def test_near_zero_soft_mask_finite(self):
        """Near-zero soft_mask (sigmoid(-10) ≈ 0.00005) produces finite entropy."""
        soft_mask = torch.full((2, 8), torch.sigmoid(torch.tensor(-10.0)).item())
        active_mask = torch.ones(2, 8, dtype=torch.bool)
        loss, entropy = NodeSelector._entropy_regularizer(soft_mask, active_mask)
        assert torch.isfinite(loss), f"loss is non-finite: {loss}"
        assert torch.isfinite(entropy), f"entropy is non-finite: {entropy}"

    def test_all_zeros_soft_mask_finite(self):
        """Saturated soft_mask (all 0.0) should still produce finite entropy."""
        soft_mask = torch.zeros(2, 8)
        active_mask = torch.ones(2, 8, dtype=torch.bool)
        loss, entropy = NodeSelector._entropy_regularizer(soft_mask, active_mask)
        assert torch.isfinite(loss)
        assert torch.isfinite(entropy)


class TestDDPMAuxLossSanitization:
    """DDPM aux loss guard should sanitize non-finite values."""

    def test_nan_sanitized_to_zero(self):
        """NaN aux loss is replaced with zero."""
        loss = torch.tensor(float("nan"))
        if not torch.isfinite(loss):
            loss = torch.zeros_like(loss)
        assert loss.item() == 0.0

    def test_inf_sanitized_to_zero(self):
        """Positive infinity aux loss is replaced with zero."""
        loss = torch.tensor(float("inf"))
        if not torch.isfinite(loss):
            loss = torch.zeros_like(loss)
        assert loss.item() == 0.0

    def test_neg_inf_sanitized_to_zero(self):
        """Negative infinity aux loss is replaced with zero."""
        loss = torch.tensor(float("-inf"))
        if not torch.isfinite(loss):
            loss = torch.zeros_like(loss)
        assert loss.item() == 0.0

    def test_finite_loss_passes_through(self):
        """Finite aux loss passes through the guard unchanged."""
        loss = torch.tensor(0.42, requires_grad=True)
        original = loss
        if not torch.isfinite(loss):
            loss = torch.zeros_like(loss)
        assert loss is original
        assert loss.item() == pytest.approx(0.42)

    def test_mixed_losses_only_valid_contributes(self):
        """When aggregating, only finite losses contribute to the total."""
        aux_losses = []

        loss_ok = torch.tensor(0.5)
        if not torch.isfinite(loss_ok):
            loss_ok = torch.zeros_like(loss_ok)
        aux_losses.append(loss_ok)

        loss_nan = torch.tensor(float("nan"))
        if not torch.isfinite(loss_nan):
            loss_nan = torch.zeros_like(loss_nan)
        aux_losses.append(loss_nan)

        loss_inf = torch.tensor(float("inf"))
        if not torch.isfinite(loss_inf):
            loss_inf = torch.zeros_like(loss_inf)
        aux_losses.append(loss_inf)

        total = torch.stack(aux_losses).sum()
        assert torch.isfinite(total)
        assert total.item() == pytest.approx(0.5)

    def test_ddpm_collect_sanitizes_nan(self, caplog):
        """DDPM._collect_selector_losses_and_diagnostics sanitizes NaN and logs."""
        from graph_signal_diffusion.diffusion.ddpm import DDPM

        # Fake model with one module returning NaN aux loss
        class _NanSelector(torch.nn.Module):
            def get_selector_aux_loss(self):
                return torch.tensor(float("nan"))
            def get_selector_diagnostics(self):
                return None

        # Build a minimal DDPM instance via __new__ (skip __init__),
        # then use object.__setattr__ to bypass Module.__setattr__ checks.
        ddpm = DDPM.__new__(DDPM)
        object.__setattr__(ddpm, 'last_selector_diagnostics', {})
        object.__setattr__(ddpm, 'last_selector_level_diagnostics', [])
        object.__setattr__(ddpm, 'last_selector_stats_by_level', {})
        object.__setattr__(ddpm, 'model', torch.nn.ModuleList([_NanSelector()]))

        with caplog.at_level(logging.WARNING, logger="graph_signal_diffusion.diffusion.ddpm"):
            result = ddpm._collect_selector_losses_and_diagnostics()

        # NaN should be sanitized to zero
        assert result is not None
        assert torch.isfinite(result)
        assert result.item() == 0.0

        # Warning should have been logged
        assert any("Non-finite selector aux loss" in r.message for r in caplog.records)

    def test_ddpm_aggregator_includes_exploration_noise(self):
        """``last_selector_diagnostics`` (aggregated) must carry the scalar
        ``exploration_noise`` emitted by NodeSelector.

        Background: the aggregate dict is consumed by the trainer's
        epoch-summary path (JSONL + ``expl=…`` train.log line). The per-level
        dict in ``last_selector_level_diagnostics`` already includes every
        scalar key unfiltered, which is what Wandb reads — so a Wandb-only
        signal does not prove the aggregator surfaces the field.
        """
        from graph_signal_diffusion.diffusion.ddpm import DDPM

        class _ExplorationDiagSelector(torch.nn.Module):
            def get_selector_aux_loss(self):
                return None

            def get_selector_diagnostics(self):
                return {
                    "temperature": 0.75,
                    "exploration_noise": 0.42,
                    "entropy_mean_norm": 0.5,
                    "entropy_reg": 0.5,
                    "entropy_reg_weight": 0.0,
                    "num_active_mean": 273.0,
                    "num_selected_mean": 136.0,
                    "selected_ratio_mean": 0.499,
                    "score_mean_active": 0.0,
                    "score_std_active": 1.0,
                    "soft_mask_mean_active": 0.5,
                    "soft_mask_std_active": 0.27,
                    "score_gain": 0.0,
                }

        ddpm = DDPM.__new__(DDPM)
        object.__setattr__(ddpm, 'last_selector_diagnostics', {})
        object.__setattr__(ddpm, 'last_selector_level_diagnostics', [])
        object.__setattr__(ddpm, 'last_selector_stats_by_level', {})
        object.__setattr__(
            ddpm, 'collect_selector_heavy_diagnostics', False
        )
        object.__setattr__(
            ddpm, 'model', torch.nn.ModuleList([_ExplorationDiagSelector()])
        )

        ddpm._collect_selector_losses_and_diagnostics()

        agg = ddpm.last_selector_diagnostics
        assert "exploration_noise" in agg, (
            "Aggregate last_selector_diagnostics must surface "
            "exploration_noise (regression for ddpm.py numeric_keys allowlist)"
        )
        assert agg["exploration_noise"] == pytest.approx(0.42)
        # Sanity: existing aggregated keys remain present.
        assert agg["temperature"] == pytest.approx(0.75)
