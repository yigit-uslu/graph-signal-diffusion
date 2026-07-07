"""Targeted tests for AMP runtime gating, gradient zeroing, and eval autocast scope.

These tests verify the behavior described in AMP_COUNTER_PLAN_2026-03-03.md:
- self._amp_enabled is the CUDA-gated runtime flag; self.use_amp is preserved as user config.
- Exactly one optimizer.zero_grad() call per training step; diffusion.zero_grad() never called.
- Eval autocast context wraps only loss computation, not sampling.
"""
import logging
import tempfile
from unittest.mock import MagicMock, patch, call
from pathlib import Path

import torch
import torch.nn as nn
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from graph_signal_diffusion.trainers.trainer import DiffusionTrainer
from graph_signal_diffusion.diffusion.base import BaseDiffusion


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x):
        return self.linear(x)


class DummyDiffusion(BaseDiffusion):
    """Minimal diffusion: training_loss is connected to model params so backward() works."""

    def __init__(self, model):
        super().__init__(model)

    def training_loss(self, data, return_pred_stats=False):
        # Use the model so loss has a real grad_fn through model parameters.
        dummy_input = torch.zeros(1, 4)
        loss = self.model(dummy_input).sum()
        if return_pred_stats:
            return loss, {}
        return loss

    def sample(self, shape, device, data=None, use_amp=False, return_selector_sampling_diagnostics=False):
        if return_selector_sampling_diagnostics:
            return torch.zeros(shape), {}
        return torch.zeros(shape)

    def add_noise(self, x, t, noise=None):
        return x


class DummyData:
    """Minimal data object that satisfies fit()'s access patterns."""

    num_graphs = 1
    num_nodes = 4

    def to(self, device):
        return self


def _make_trainer(use_amp: bool, device: torch.device, tmpdir: str) -> DiffusionTrainer:
    model = DummyModel()
    diffusion = DummyDiffusion(model)
    optimizer = torch.optim.SGD(diffusion.parameters(), lr=0.01)
    return DiffusionTrainer(
        diffusion=diffusion,
        optimizer=optimizer,
        device=device,
        save_dir=tmpdir,
        use_amp=use_amp,
    )


# ---------------------------------------------------------------------------
# Phase 1 tests: _amp_enabled runtime gating
# ---------------------------------------------------------------------------

class TestAmpRuntimeGating:
    def test_amp_requested_on_cpu_disables_runtime_amp(self, tmp_path):
        """use_amp=True on CPU: _amp_enabled is False, scaler is None, use_amp is preserved."""
        trainer = _make_trainer(use_amp=True, device=torch.device("cpu"), tmpdir=str(tmp_path))

        assert trainer.use_amp is True, "user config flag must be preserved"
        assert trainer._amp_enabled is False, "_amp_enabled must be False on CPU"
        assert trainer.scaler is None, "scaler must be None on CPU"

    def test_amp_false_on_cpu_stays_disabled(self, tmp_path):
        """use_amp=False: _amp_enabled is False, scaler is None."""
        trainer = _make_trainer(use_amp=False, device=torch.device("cpu"), tmpdir=str(tmp_path))

        assert trainer.use_amp is False
        assert trainer._amp_enabled is False
        assert trainer.scaler is None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_amp_requested_on_cuda_enables_runtime_amp(self, tmp_path):
        """use_amp=True on CUDA: _amp_enabled is True, scaler is created."""
        trainer = _make_trainer(use_amp=True, device=torch.device("cuda"), tmpdir=str(tmp_path))

        assert trainer._amp_enabled is True
        assert trainer.scaler is not None

    def test_amp_requested_on_cpu_emits_warning(self, tmp_path, caplog):
        """use_amp=True on CPU emits a log.warning about AMP requiring CUDA."""
        with caplog.at_level(logging.WARNING, logger="graph_signal_diffusion.trainers.trainer"):
            _make_trainer(use_amp=True, device=torch.device("cpu"), tmpdir=str(tmp_path))

        warning_texts = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("AMP" in t and "CUDA" in t for t in warning_texts), (
            f"Expected AMP/CUDA warning; got: {warning_texts}"
        )

    def test_amp_false_on_cpu_no_warning(self, tmp_path, caplog):
        """use_amp=False: no AMP warning is emitted."""
        with caplog.at_level(logging.WARNING, logger="graph_signal_diffusion.trainers.trainer"):
            _make_trainer(use_amp=False, device=torch.device("cpu"), tmpdir=str(tmp_path))

        warning_texts = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        amp_warnings = [t for t in warning_texts if "AMP" in t]
        assert not amp_warnings, f"Unexpected AMP warnings: {amp_warnings}"


# ---------------------------------------------------------------------------
# Phase 2 tests: single zero_grad per training step
# ---------------------------------------------------------------------------

class TestTrainingStepZeroGrad:
    def test_optimizer_zero_grad_called_exactly_once_per_step(self, tmp_path):
        """fit() must call optimizer.zero_grad exactly once per successful training step."""
        device = torch.device("cpu")
        model = DummyModel()
        diffusion = DummyDiffusion(model)
        optimizer = torch.optim.SGD(diffusion.parameters(), lr=0.01)

        trainer = DiffusionTrainer(
            diffusion=diffusion,
            optimizer=optimizer,
            device=device,
            save_dir=str(tmp_path),
            use_amp=False,
        )

        zero_grad_call_count = 0
        original_zero_grad = optimizer.zero_grad

        def counting_zero_grad(**kwargs):
            nonlocal zero_grad_call_count
            zero_grad_call_count += 1
            return original_zero_grad(**kwargs)

        optimizer.zero_grad = counting_zero_grad

        train_loader = [DummyData()]  # 1 batch
        trainer.fit(train_loader, val_loader=None, max_epochs=1)

        assert zero_grad_call_count == 1, (
            f"Expected exactly 1 optimizer.zero_grad() call per step, got {zero_grad_call_count}"
        )

    def test_diffusion_zero_grad_never_called(self, tmp_path):
        """fit() must never call diffusion.zero_grad() directly."""
        device = torch.device("cpu")
        model = DummyModel()
        diffusion = DummyDiffusion(model)
        optimizer = torch.optim.SGD(diffusion.parameters(), lr=0.01)

        trainer = DiffusionTrainer(
            diffusion=diffusion,
            optimizer=optimizer,
            device=device,
            save_dir=str(tmp_path),
            use_amp=False,
        )

        diffusion_zero_grad_called = False
        original_zero_grad = diffusion.zero_grad

        def spy_zero_grad(*args, **kwargs):
            nonlocal diffusion_zero_grad_called
            diffusion_zero_grad_called = True
            return original_zero_grad(*args, **kwargs)

        diffusion.zero_grad = spy_zero_grad

        train_loader = [DummyData()]
        trainer.fit(train_loader, val_loader=None, max_epochs=1)

        assert not diffusion_zero_grad_called, (
            "diffusion.zero_grad() must not be called directly from fit()"
        )


# ---------------------------------------------------------------------------
# Sampling helper AMP flag propagation
# ---------------------------------------------------------------------------

class TestSamplingAmpFlagPropagation:
    def test_sample_receives_amp_enabled_not_use_amp(self, tmp_path):
        """_sample_with_selector_sampling_diagnostics must forward _amp_enabled, not use_amp.

        On CPU with use_amp=True: _amp_enabled=False, use_amp=True.
        diffusion.sample() must be called with use_amp=False (_amp_enabled).
        """
        device = torch.device("cpu")
        model = DummyModel()
        diffusion = DummyDiffusion(model)
        optimizer = torch.optim.SGD(diffusion.parameters(), lr=0.01)

        trainer = DiffusionTrainer(
            diffusion=diffusion,
            optimizer=optimizer,
            device=device,
            save_dir=str(tmp_path),
            use_amp=True,   # requested, but CPU — so _amp_enabled must be False
        )

        assert trainer.use_amp is True
        assert trainer._amp_enabled is False

        # Record the use_amp value passed to diffusion.sample()
        captured_use_amp: list = []
        original_sample = diffusion.sample

        def spy_sample(*args, **kwargs):
            captured_use_amp.append(kwargs.get("use_amp", None))
            return original_sample(*args, **kwargs)

        diffusion.sample = spy_sample

        trainer._sample_with_selector_sampling_diagnostics(
            shape=(1, 4),
            data=DummyData(),
        )

        assert len(captured_use_amp) == 1, "diffusion.sample() was not called"
        assert captured_use_amp[0] == trainer._amp_enabled, (
            f"diffusion.sample() received use_amp={captured_use_amp[0]!r}, "
            f"expected _amp_enabled={trainer._amp_enabled!r}"
        )


# ---------------------------------------------------------------------------
# Eval autocast scope: loss inside, sampling outside
# ---------------------------------------------------------------------------

class TestEvalAutoCastScope:
    def test_eval_loss_inside_autocast_sampling_outside(self, tmp_path):
        """evaluate() must call training_loss inside the autocast ctx and sampling outside it.

        Uses a mock context manager instead of torch.is_autocast_enabled() so the
        assertion is CPU-compatible and independent of CUDA hardware.
        """

        class _TrackingCtx:
            """Lightweight context manager that exposes whether it is currently active."""
            def __init__(self):
                self.active = False
            def __enter__(self):
                self.active = True
                return self
            def __exit__(self, *args):
                self.active = False
                return False

        tracking_ctx = _TrackingCtx()

        class _MinimalTask:
            """Minimal task that triggers the sampling branch inside evaluate()."""
            n_samples_per_input = 1
            def prepare_data(self, data):
                return {"samples": torch.zeros(1, 4), "metadata": {}}
            def evaluate_samples(self, generated_samples, real_samples, metadata, viz_save_dir=None):
                return {}

        device = torch.device("cpu")
        model = DummyModel()
        diffusion = DummyDiffusion(model)
        optimizer = torch.optim.SGD(diffusion.parameters(), lr=0.01)

        trainer = DiffusionTrainer(
            diffusion=diffusion,
            optimizer=optimizer,
            device=device,
            save_dir=str(tmp_path),
            use_amp=False,
            task=_MinimalTask(),
        )
        # Force _amp_enabled=True so evaluate() uses torch.amp.autocast (not nullcontext).
        trainer._amp_enabled = True

        autocast_during_loss = []
        autocast_during_sample = []

        def spy_loss(data):
            autocast_during_loss.append(tracking_ctx.active)
            return torch.tensor(0.5)  # detached scalar — no backward in eval

        original_sample = diffusion.sample
        def spy_sample(*args, **kwargs):
            autocast_during_sample.append(tracking_ctx.active)
            return original_sample(*args, **kwargs)

        diffusion.training_loss = spy_loss
        diffusion.sample = spy_sample

        # Patch torch.amp.autocast so it returns our tracking context manager.
        # evaluate() creates: autocast_ctx = torch.amp.autocast(...) if _amp_enabled else nullcontext()
        # With the patch, autocast_ctx becomes tracking_ctx, which records enter/exit.
        with patch.object(torch.amp, "autocast", return_value=tracking_ctx):
            trainer.evaluate([DummyData()], eval_split="val", num_max_eval_batches=1)

        assert autocast_during_loss, "training_loss was never called during evaluate()"
        assert autocast_during_sample, "diffusion.sample was never called during evaluate()"
        assert all(autocast_during_loss), (
            "training_loss must run INSIDE the autocast context during evaluate()"
        )
        assert not any(autocast_during_sample), (
            "diffusion.sample must run OUTSIDE the autocast context during evaluate()"
        )

    def test_stratified_loss_inside_autocast(self, tmp_path):
        """evaluate() must also call training_loss_stratified_t inside the autocast ctx.

        When the diffusion model exposes training_loss_stratified_t, evaluate() uses it
        in preference to training_loss.  The autocast scope must still bracket only that
        call; sampling must remain outside.
        """

        class _TrackingCtx:
            def __init__(self):
                self.active = False
            def __enter__(self):
                self.active = True
                return self
            def __exit__(self, *args):
                self.active = False
                return False

        tracking_ctx = _TrackingCtx()

        class _StratifiedDiffusion(DummyDiffusion):
            """DummyDiffusion extended with training_loss_stratified_t."""
            def training_loss_stratified_t(self, data, return_pred_stats=False):
                # Returns (scalar_loss, per-bin dict) — plus pred_stats when requested,
                # matching the real training_loss_stratified_t contract used by evaluate().
                loss, bins = torch.tensor(0.5), {"t_bin_0": [0.5]}
                if return_pred_stats:
                    return loss, bins, {}
                return loss, bins

        class _MinimalTask:
            n_samples_per_input = 1
            def prepare_data(self, data):
                return {"samples": torch.zeros(1, 4), "metadata": {}}
            def evaluate_samples(self, generated_samples, real_samples, metadata, viz_save_dir=None):
                return {}

        device = torch.device("cpu")
        model = DummyModel()
        diffusion = _StratifiedDiffusion(model)
        optimizer = torch.optim.SGD(diffusion.parameters(), lr=0.01)

        trainer = DiffusionTrainer(
            diffusion=diffusion,
            optimizer=optimizer,
            device=device,
            save_dir=str(tmp_path),
            use_amp=False,
            task=_MinimalTask(),
        )
        trainer._amp_enabled = True

        autocast_during_stratified = []
        autocast_during_sample = []

        original_stratified = diffusion.training_loss_stratified_t
        def spy_stratified(data, return_pred_stats=False):
            autocast_during_stratified.append(tracking_ctx.active)
            return original_stratified(data, return_pred_stats=return_pred_stats)

        original_sample = diffusion.sample
        def spy_sample(*args, **kwargs):
            autocast_during_sample.append(tracking_ctx.active)
            return original_sample(*args, **kwargs)

        diffusion.training_loss_stratified_t = spy_stratified
        diffusion.sample = spy_sample

        with patch.object(torch.amp, "autocast", return_value=tracking_ctx):
            trainer.evaluate([DummyData()], eval_split="val", num_max_eval_batches=1)

        assert autocast_during_stratified, "training_loss_stratified_t was never called"
        assert autocast_during_sample, "diffusion.sample was never called"
        assert all(autocast_during_stratified), (
            "training_loss_stratified_t must run INSIDE the autocast context"
        )
        assert not any(autocast_during_sample), (
            "diffusion.sample must run OUTSIDE the autocast context"
        )

    def test_evaluate_v2_loss_inside_autocast_sampling_outside(self, tmp_path):
        """evaluate_v2() must mirror evaluate(): loss inside autocast, sampling outside."""

        class _TrackingCtx:
            def __init__(self):
                self.active = False
            def __enter__(self):
                self.active = True
                return self
            def __exit__(self, *args):
                self.active = False
                return False

        tracking_ctx = _TrackingCtx()

        class _MinimalTaskV2:
            n_samples_per_input = 1
            def prepare_data(self, data):
                return {"samples": torch.zeros(1, 4), "metadata": {}}
            def evaluate_samples(self, generated_samples, real_samples, metadata, viz_save_dir=None):
                return {}

        device = torch.device("cpu")
        model = DummyModel()
        diffusion = DummyDiffusion(model)
        optimizer = torch.optim.SGD(diffusion.parameters(), lr=0.01)

        trainer = DiffusionTrainer(
            diffusion=diffusion,
            optimizer=optimizer,
            device=device,
            save_dir=str(tmp_path),
            use_amp=False,
            task=_MinimalTaskV2(),
        )
        trainer._amp_enabled = True

        autocast_during_loss = []
        autocast_during_sample = []

        def spy_loss(data):
            autocast_during_loss.append(tracking_ctx.active)
            return torch.tensor(0.5)

        original_sample = diffusion.sample
        def spy_sample(*args, **kwargs):
            autocast_during_sample.append(tracking_ctx.active)
            return original_sample(*args, **kwargs)

        diffusion.training_loss = spy_loss
        diffusion.sample = spy_sample

        with patch.object(torch.amp, "autocast", return_value=tracking_ctx):
            trainer.evaluate_v2([DummyData()], eval_split="val", num_max_eval_batches=1)

        assert autocast_during_loss, "training_loss was never called during evaluate_v2()"
        assert autocast_during_sample, "diffusion.sample was never called during evaluate_v2()"
        assert all(autocast_during_loss), (
            "training_loss must run INSIDE the autocast context during evaluate_v2()"
        )
        assert not any(autocast_during_sample), (
            "diffusion.sample must run OUTSIDE the autocast context during evaluate_v2()"
        )


# ---------------------------------------------------------------------------
# Zero-grad behaviour on skip paths
# ---------------------------------------------------------------------------

class TestZeroGradOnSkipPaths:
    def test_zero_grad_called_once_on_nonfinite_loss_skip(self, tmp_path):
        """When non-AMP loss is non-finite the step is skipped but zero_grad is called
        exactly once (at the top of the step — the skip path must not add a second call).
        """
        device = torch.device("cpu")
        model = DummyModel()

        class _InfLossDiffusion(DummyDiffusion):
            def training_loss(self, data, return_pred_stats=False):
                loss = torch.tensor(float("inf"))
                if return_pred_stats:
                    return loss, {}
                return loss

        diffusion = _InfLossDiffusion(model)
        optimizer = torch.optim.SGD(diffusion.parameters(), lr=0.01)

        trainer = DiffusionTrainer(
            diffusion=diffusion,
            optimizer=optimizer,
            device=device,
            save_dir=str(tmp_path),
            use_amp=False,
        )

        zero_grad_call_count = 0
        original_zero_grad = optimizer.zero_grad

        def counting_zero_grad(**kwargs):
            nonlocal zero_grad_call_count
            zero_grad_call_count += 1
            return original_zero_grad(**kwargs)

        optimizer.zero_grad = counting_zero_grad

        trainer.fit([DummyData()], val_loader=None, max_epochs=1)

        assert zero_grad_call_count == 1, (
            f"Expected exactly 1 zero_grad call even on a skipped (inf loss) step, "
            f"got {zero_grad_call_count}"
        )

    def test_zero_grad_called_once_on_nonfinite_grad_skip(self, tmp_path):
        """When non-AMP gradients are non-finite the step is skipped but zero_grad is
        called exactly once (skip path must not add a second call after backward).
        """
        device = torch.device("cpu")

        class _InfGradDiffusion(DummyDiffusion):
            """Produces finite loss but injects inf into the model gradient after backward."""
            def training_loss(self, data, return_pred_stats=False):
                # Connect to params so backward() populates .grad
                dummy_input = torch.zeros(1, 4)
                loss = self.model(dummy_input).sum()
                if return_pred_stats:
                    return loss, {}
                return loss

            def parameters(self):
                # After super().parameters(), we return the same params but will
                # inject inf into the first one inside the test via a hook.
                return super().parameters()

        model = DummyModel()
        diffusion = _InfGradDiffusion(model)
        optimizer = torch.optim.SGD(diffusion.parameters(), lr=0.01)

        trainer = DiffusionTrainer(
            diffusion=diffusion,
            optimizer=optimizer,
            device=device,
            save_dir=str(tmp_path),
            use_amp=False,
        )

        # Inject inf into the gradient of the first parameter after every backward pass.
        first_param = next(diffusion.parameters())

        def _inject_inf_grad(grad):
            return torch.full_like(grad, float("inf"))

        first_param.register_hook(_inject_inf_grad)

        zero_grad_call_count = 0
        original_zero_grad = optimizer.zero_grad

        def counting_zero_grad(**kwargs):
            nonlocal zero_grad_call_count
            zero_grad_call_count += 1
            return original_zero_grad(**kwargs)

        optimizer.zero_grad = counting_zero_grad

        trainer.fit([DummyData()], val_loader=None, max_epochs=1)

        assert zero_grad_call_count == 1, (
            f"Expected exactly 1 zero_grad call even on a skipped (inf grad) step, "
            f"got {zero_grad_call_count}"
        )


class TestSkipEvalAtEpochZero:
    def test_skip_epoch_zero_eval_enabled_by_default(self, tmp_path):
        device = torch.device("cpu")
        model = DummyModel()
        diffusion = DummyDiffusion(model)
        optimizer = torch.optim.SGD(diffusion.parameters(), lr=0.01)

        trainer = DiffusionTrainer(
            diffusion=diffusion,
            optimizer=optimizer,
            device=device,
            save_dir=str(tmp_path),
            use_amp=False,
            eval_schedule={"type": "uniform", "period": 1, "eval_on_last_epoch": True},
        )

        eval_call_count = 0

        def _spy_evaluate(*args, **kwargs):
            nonlocal eval_call_count
            eval_call_count += 1
            return {"loss": 0.0}

        trainer.evaluate = _spy_evaluate
        trainer.fit([DummyData()], val_loader=[DummyData()], max_epochs=3)

        # With period=1 we'd normally eval at epochs 0,1,2. The epoch-0 skip keeps only 1,2.
        assert eval_call_count == 2

    def test_skip_epoch_zero_eval_can_be_disabled(self, tmp_path):
        device = torch.device("cpu")
        model = DummyModel()
        diffusion = DummyDiffusion(model)
        optimizer = torch.optim.SGD(diffusion.parameters(), lr=0.01)

        trainer = DiffusionTrainer(
            diffusion=diffusion,
            optimizer=optimizer,
            device=device,
            save_dir=str(tmp_path),
            use_amp=False,
            eval_schedule={
                "type": "uniform",
                "period": 1,
                "eval_on_first_epoch": True,
                "eval_on_last_epoch": True,
            },
        )

        eval_call_count = 0

        def _spy_evaluate(*args, **kwargs):
            nonlocal eval_call_count
            eval_call_count += 1
            return {"loss": 0.0}

        trainer.evaluate = _spy_evaluate
        trainer.fit([DummyData()], val_loader=[DummyData()], max_epochs=3)

        # Epoch-0 skip disabled => evaluate at every epoch for period=1.
        assert eval_call_count == 3


class TestBestModelRawCompositeLogging:
    def test_raw_composite_is_injected_and_logged_to_wandb(self, tmp_path):
        class _MinimalTask:
            n_samples_per_input = 1

            def prepare_data(self, data):
                return {"samples": torch.zeros(1, 4), "metadata": {}}

            def evaluate_samples(self, generated_samples, real_samples, metadata, viz_save_dir=None):
                return {"metric_a": 2.5}

        class _WandbRun:
            def __init__(self):
                self.logged = []
                self.summary = {}

            def log(self, payload, step=None):
                self.logged.append((payload, step))

        device = torch.device("cpu")
        model = DummyModel()
        diffusion = DummyDiffusion(model)
        optimizer = torch.optim.SGD(diffusion.parameters(), lr=0.01)
        wandb_run = _WandbRun()

        trainer = DiffusionTrainer(
            diffusion=diffusion,
            optimizer=optimizer,
            device=device,
            save_dir=str(tmp_path),
            use_amp=False,
            task=_MinimalTask(),
            wandb_run=wandb_run,
            eval_schedule={
                "type": "uniform",
                "period": 1,
                "eval_on_first_epoch": True,
                "eval_on_last_epoch": True,
            },
            best_model={
                "enabled": True,
                "top_k": 5,
                "ema_alpha": 0.3,
                "min_warmup_evals": 1000,  # avoid checkpoint save path in this unit test
                "metrics": [
                    {"name": "val_metric_a", "weight": 1.0, "direction": "minimize"},
                ],
            },
        )

        trainer.fit([DummyData()], val_loader=[DummyData()], max_epochs=1)

        assert wandb_run.logged, "Expected at least one wandb.log call."
        payload, step = wandb_run.logged[-1]
        assert step == 0
        assert "best_model_raw_composite_score" in payload
        assert payload["best_model_raw_composite_score"] == pytest.approx(2.5)
