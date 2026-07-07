import torch
import tempfile
import pytest

from graph_signal_diffusion.trainers.dual_optimizer import DualOptimizer
from graph_signal_diffusion.trainers.primal_dual_trainer import PrimalDualTrainer


class _DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1))

    def forward(self, *args, **kwargs):
        return self.weight


class _DummyTrainer(PrimalDualTrainer):
    def primal_forward(self, batch):
        return torch.zeros((batch.num_graphs, 1), device=self.device), None

    def compute_constraints(self, primal_vars, forward_ctx, batch):
        batch_size = batch.num_graphs
        objective = torch.zeros((batch_size,), device=self.device)
        g = torch.zeros((batch_size, self.dual_optimizer.num_receivers), device=self.device)
        per_user = torch.zeros((batch_size, self.dual_optimizer.num_receivers), device=self.device)
        return objective, g, per_user

    def collect_samples(self, dataloader):
        return {}

    def analyze_sample_quality(self, samples, dataloader):
        return {}


def _build_trainer(dual_optimizer: DualOptimizer) -> _DummyTrainer:
    checkpoint_dir = tempfile.mkdtemp(prefix="pd_rmin_summary_")
    return _DummyTrainer(
        model=_DummyModel(),
        dual_optimizer=dual_optimizer,
        system_params={"noise_var": 1.0},
        learning_rate=1e-3,
        max_epochs=1,
        checkpoint_dir=checkpoint_dir,
        device="cpu",
    )


def test_r_min_summary_scalar_mode():
    dual = DualOptimizer(
        num_networks=2,
        num_receivers=3,
        r_min=0.7,
        alpha_dual=0.1,
        update_frequency=1,
        momentum=0.0,
        device="cpu",
    )
    trainer = _build_trainer(dual)
    summary = trainer._r_min_summary()
    assert summary["r_min"] == pytest.approx(0.7)
    assert summary["r_min_is_scalar"] is True
    assert summary["r_min_min"] == pytest.approx(0.7)
    assert summary["r_min_max"] == pytest.approx(0.7)
    assert summary["r_min_mean"] == pytest.approx(0.7)


def test_r_min_summary_vector_mode():
    dual = DualOptimizer(
        num_networks=2,
        num_receivers=3,
        r_min=torch.tensor([0.4, 0.9], dtype=torch.float32),
        alpha_dual=0.1,
        update_frequency=1,
        momentum=0.0,
        device="cpu",
    )
    trainer = _build_trainer(dual)
    summary = trainer._r_min_summary()
    assert summary["r_min"] is None
    assert summary["r_min_is_scalar"] is False
    assert summary["r_min_min"] == pytest.approx(0.4)
    assert summary["r_min_max"] == pytest.approx(0.9)
    assert summary["r_min_mean"] == pytest.approx(0.65)
