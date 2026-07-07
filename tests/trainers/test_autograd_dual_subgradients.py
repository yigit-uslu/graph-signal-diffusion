"""
Smoke tests for autograd-based dual subgradients.

Verifies:
1. DualOptimizer.update_from_gradients produces the same lambda updates as
   update(rates=...) when given g = r_min - rates.
2. After compute_lagrangian_loss + loss.backward(), lambdas.grad * B equals
   the constraint slacks g = r_min - ergodic_rates, up to floating-point tolerance.
3. WRAPrimalDualTrainer.primal_forward and compute_constraints compose correctly.
"""

import torch
import pytest
from collections import defaultdict
from torch_geometric.data import Batch

from graph_signal_diffusion.trainers.dual_optimizer import DualOptimizer
from graph_signal_diffusion.trainers.primal_dual_trainer import (
    PrimalDualTrainer,
    WRAPrimalDualTrainer,
)
from graph_signal_diffusion.models.power_allocation_gnn import PowerAllocationGNN
from graph_signal_diffusion.datasets.wra.primal_dual_dataset import WirelessData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dual_optimizer(num_networks=4, n=6, r_min=0.5, alpha=0.05, momentum=0.0):
    opt = DualOptimizer(
        num_networks=num_networks,
        num_receivers=n,
        r_min=r_min,
        alpha_dual=alpha,
        update_frequency=1,
        momentum=momentum,
        device="cpu",
    )
    # Give non-trivial initial lambdas so updates are non-zero
    opt.lambdas = torch.rand(num_networks, n)
    return opt


def _make_trainer(
    m=8,
    n=8,
    T=10,
    num_networks=4,
    r_min=0.5,
    device="cpu",
    **trainer_kwargs,
):
    """Return a WRAPrimalDualTrainer with a tiny GNN, ready for forward passes."""
    model = PowerAllocationGNN(
        input_dim=2,
        hidden_dim=16,
        num_layers=2,
        K=1,
        P_max=1.0,
        norm_type="none",
        num_groups=1,
        dropout=0.0,
        activation="relu",
    )
    dual_opt = _make_dual_optimizer(num_networks=num_networks, n=n, r_min=r_min)
    trainer = WRAPrimalDualTrainer(
        model=model,
        dual_optimizer=dual_opt,
        system_params={"noise_var": 1e-2, "P_max_watts": 1.0},
        learning_rate=1e-3,
        max_epochs=10,
        checkpoint_dir="/tmp/test_wra_trainer",
        num_samples_per_network=2,
        moving_avg_window=5,
        device=device,
        **trainer_kwargs,
    )
    return trainer, m, n, T


def _make_batch(B=3, m=8, n=8, T=10, num_networks=4, device="cpu"):
    """Build a minimal PyG Batch that WRAPrimalDualTrainer expects.

    Uses WirelessData (not plain Data) so that __cat_dim__ returns None for
    associations, H_instantaneous, and network_id — causing PyG to store them
    as Python lists in the batch, matching what the real dataset produces.
    """
    graphs = []
    for b in range(B):
        net_id = b % num_networks
        # Node features: (n, 2) — each receiver gets [mean_H, noise_var]
        x = torch.rand(n, 2)
        # Chain edges among receiver nodes (minimal connectivity)
        src = torch.arange(n - 1)
        dst = torch.arange(1, n)
        edge_index = torch.stack([
            torch.cat([src, dst]),
            torch.cat([dst, src]),
        ])
        edge_weight = torch.rand(edge_index.shape[1])
        # One-to-one association (identity, m==n)
        associations = torch.eye(m, n)
        # Instantaneous channel gains: (T, m, n)
        H_instantaneous = torch.rand(T, m, n) * 0.1
        g = WirelessData(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            associations=associations,
            H_instantaneous=H_instantaneous,
            network_id=torch.tensor(net_id, dtype=torch.long),
        )
        graphs.append(g)

    batch = Batch.from_data_list(graphs)
    return batch.to(device)


class _GenericConvergenceTrainer(PrimalDualTrainer):
    """Minimal non-WRA trainer for testing base convergence semantics."""

    def __init__(self, fixed_g: torch.Tensor, **kwargs):
        self.fixed_g = fixed_g
        super().__init__(**kwargs)

    def primal_forward(self, batch: Batch):
        return torch.zeros((batch.num_graphs, 1), dtype=torch.float32, device=self.device), None

    def compute_constraints(self, primal_vars: torch.Tensor, forward_ctx: object, batch: Batch):
        B = batch.num_graphs
        g = self.fixed_g[:B].to(self.device)
        objective = torch.zeros(B, dtype=torch.float32, device=self.device)
        per_user_metrics = torch.zeros_like(g)
        return objective, g, per_user_metrics

    def collect_samples(self, dataloader):
        return {}

    def analyze_sample_quality(self, samples, dataloader):
        return {}


def _make_generic_trainer(
    fixed_g: torch.Tensor,
    n: int,
    num_networks: int = 4,
    alpha_dual: float = 0.1,
):
    dual_opt = _make_dual_optimizer(
        num_networks=num_networks,
        n=n,
        r_min=0.5,
        alpha=alpha_dual,
        momentum=0.0,
    )
    model = torch.nn.Linear(1, 1)
    return _GenericConvergenceTrainer(
        fixed_g=fixed_g,
        model=model,
        dual_optimizer=dual_opt,
        system_params={"noise_var": 1e-2, "P_max_watts": 1.0},
        learning_rate=1e-3,
        max_epochs=5,
        checkpoint_dir="/tmp/test_generic_convergence_trainer",
        convergence_window=3,
        convergence_patience=2,
        gradient_norm_threshold=0.1,
        dual_variance_threshold=0.1,
        dual_stationarity_threshold=0.05,
        num_samples_per_network=2,
        moving_avg_window=3,
        device="cpu",
    )


# ---------------------------------------------------------------------------
# Test 1: update_from_gradients == update(rates) numerically
# ---------------------------------------------------------------------------

class TestUpdateFromGradients:

    def test_identity_associations_no_momentum(self):
        """With identity associations, update_from_gradients(g) == update(rates)."""
        B, n = 4, 6
        r_min = 0.5
        rates = torch.rand(B, n)
        g = r_min - rates  # constraint slacks

        opt1 = _make_dual_optimizer(num_networks=B, n=n, r_min=r_min, momentum=0.0)
        opt2 = _make_dual_optimizer(num_networks=B, n=n, r_min=r_min, momentum=0.0)
        initial_lambdas = torch.rand(B, n)
        opt1.lambdas = initial_lambdas.clone()
        opt2.lambdas = initial_lambdas.clone()

        network_ids = torch.arange(B)
        opt1.update(network_ids=network_ids, rates=rates)
        opt2.update_from_gradients(network_ids=network_ids, gradients=g)

        assert torch.allclose(opt1.lambdas, opt2.lambdas, atol=1e-6), (
            f"Lambda mismatch: max diff = {(opt1.lambdas - opt2.lambdas).abs().max():.2e}"
        )

    def test_with_momentum(self):
        """update_from_gradients and update produce the same result with momentum."""
        B, n = 3, 8
        r_min = 0.7
        momentum = 0.9
        rates = torch.rand(B, n)
        g = r_min - rates

        opt1 = _make_dual_optimizer(num_networks=B, n=n, r_min=r_min, momentum=momentum)
        opt2 = _make_dual_optimizer(num_networks=B, n=n, r_min=r_min, momentum=momentum)
        initial_lambdas = torch.rand(B, n)
        initial_velocity = torch.rand(B, n) * 0.1
        opt1.lambdas = initial_lambdas.clone()
        opt2.lambdas = initial_lambdas.clone()
        opt1.velocity = initial_velocity.clone()
        opt2.velocity = initial_velocity.clone()

        network_ids = torch.arange(B)
        opt1.update(network_ids=network_ids, rates=rates)
        opt2.update_from_gradients(network_ids=network_ids, gradients=g)

        assert torch.allclose(opt1.lambdas, opt2.lambdas, atol=1e-6), (
            f"Lambda mismatch (momentum): max diff = {(opt1.lambdas - opt2.lambdas).abs().max():.2e}"
        )

    def test_projection_clamp_matches(self):
        """Projection to λ≥0 behaves identically in both paths."""
        B, n = 2, 4
        r_min = 0.3
        # Use rates > r_min so g is negative (all constraints satisfied)
        rates = torch.ones(B, n) * 0.9
        g = r_min - rates  # all negative

        opt1 = _make_dual_optimizer(num_networks=B, n=n, r_min=r_min, momentum=0.0)
        opt2 = _make_dual_optimizer(num_networks=B, n=n, r_min=r_min, momentum=0.0)
        opt1.lambdas = torch.ones(B, n) * 0.01  # near zero — will be clamped
        opt2.lambdas = torch.ones(B, n) * 0.01

        network_ids = torch.arange(B)
        opt1.update(network_ids=network_ids, rates=rates)
        opt2.update_from_gradients(network_ids=network_ids, gradients=g)

        assert torch.allclose(opt1.lambdas, opt2.lambdas, atol=1e-6)
        assert (opt1.lambdas >= 0).all(), "Lambdas should remain non-negative"

    def test_stats_match(self):
        """Both methods return the same diagnostic stats dict."""
        B, n = 3, 5
        r_min = 0.6
        rates = torch.rand(B, n)
        g = r_min - rates

        opt1 = _make_dual_optimizer(num_networks=B, n=n, r_min=r_min, momentum=0.0)
        opt2 = _make_dual_optimizer(num_networks=B, n=n, r_min=r_min, momentum=0.0)
        lam0 = torch.rand(B, n)
        opt1.lambdas = lam0.clone()
        opt2.lambdas = lam0.clone()

        network_ids = torch.arange(B)
        s1 = opt1.update(network_ids=network_ids, rates=rates)
        s2 = opt2.update_from_gradients(network_ids=network_ids, gradients=g)

        for key in s1:
            assert abs(s1[key] - s2[key]) < 1e-5, (
                f"Stats mismatch for '{key}': {s1[key]} vs {s2[key]}"
            )


# ---------------------------------------------------------------------------
# Test 2: autograd subgradient == g from compute_lagrangian_loss
# ---------------------------------------------------------------------------

class TestAutogradSubgradientEquivalence:

    def test_lambdas_grad_equals_g_over_B(self):
        """After loss.backward(), lambdas.grad * B == g (constraint slacks)."""
        B, m, n, T = 3, 8, 8, 10
        trainer, _, _, _ = _make_trainer(m=m, n=n, T=T, num_networks=4)
        batch = _make_batch(B=B, m=m, n=n, T=T, num_networks=4)

        loss, lambdas, g, per_user_metrics, primal_vars, stats = (
            trainer.compute_lagrangian_loss(batch)
        )
        loss.backward()

        assert lambdas.grad is not None, "lambdas.grad is None after backward()"

        # lambdas.grad = ∂L/∂λ = g / B  (due to .mean() in loss)
        recovered_g = lambdas.grad * B
        assert torch.allclose(recovered_g, g, atol=1e-5), (
            f"Autograd g mismatch: max diff = {(recovered_g - g).abs().max():.2e}"
        )

    def test_lambdas_grad_matches_r_min_minus_rates(self):
        """lambdas.grad * B == r_min - per_user_metrics."""
        B, m, n, T = 4, 6, 6, 8
        r_min = 0.4
        trainer, _, _, _ = _make_trainer(m=m, n=n, T=T, num_networks=4, r_min=r_min)
        batch = _make_batch(B=B, m=m, n=n, T=T, num_networks=4)

        loss, lambdas, g, per_user_metrics, primal_vars, stats = (
            trainer.compute_lagrangian_loss(batch)
        )
        loss.backward()

        # g is defined as r_min - rates
        expected_g = r_min - per_user_metrics
        recovered_g = lambdas.grad * B

        assert torch.allclose(recovered_g, expected_g, atol=1e-5), (
            f"r_min - rates mismatch: max diff = {(recovered_g - expected_g).abs().max():.2e}"
        )

    def test_model_grads_unaffected_by_leaf_lambda(self):
        """Making lambdas a leaf tensor must not break gradient flow to model params."""
        B, m, n, T = 2, 8, 8, 10
        trainer, _, _, _ = _make_trainer(m=m, n=n, T=T, num_networks=4)
        batch = _make_batch(B=B, m=m, n=n, T=T, num_networks=4)

        trainer.model.zero_grad()
        loss, lambdas, g, per_user_metrics, primal_vars, stats = (
            trainer.compute_lagrangian_loss(batch)
        )
        loss.backward()

        grads = [p.grad for p in trainer.model.parameters() if p.grad is not None]
        assert len(grads) > 0, "No model parameter gradients found after backward()"
        assert all((grad.abs() > 0).any() for grad in grads), (
            "Some model parameter gradients are all-zero — gradient flow broken"
        )

    def test_dual_update_via_autograd_matches_manual(self):
        """
        Full round-trip: update_from_gradients(lambdas.grad * B) produces the
        same lambda values as the old update(rates=per_user_metrics).
        """
        B, m, n, T = 3, 8, 8, 10
        r_min = 0.5
        trainer, _, _, _ = _make_trainer(m=m, n=n, T=T, num_networks=4, r_min=r_min)
        batch = _make_batch(B=B, m=m, n=n, T=T, num_networks=4)

        # --- Autograd path ---
        initial_lambdas = trainer.dual_optimizer.lambdas.clone()
        loss, lambdas, g, per_user_metrics, primal_vars, stats = (
            trainer.compute_lagrangian_loss(batch)
        )
        loss.backward()
        network_ids = torch.tensor(
            [nid.item() for nid in batch.network_id], dtype=torch.long
        )
        trainer.dual_optimizer.update_from_gradients(
            network_ids=network_ids, gradients=lambdas.grad * B
        )
        lambdas_autograd = trainer.dual_optimizer.lambdas.clone()

        # --- Manual (rates) path with identical initial state ---
        trainer.dual_optimizer.lambdas = initial_lambdas.clone()
        trainer.dual_optimizer.velocity.zero_()
        trainer.dual_optimizer.update(
            network_ids=network_ids, rates=per_user_metrics
        )
        lambdas_manual = trainer.dual_optimizer.lambdas.clone()

        assert torch.allclose(lambdas_autograd, lambdas_manual, atol=1e-5), (
            f"Autograd vs manual lambda mismatch: "
            f"max diff = {(lambdas_autograd - lambdas_manual).abs().max():.2e}"
        )


# ---------------------------------------------------------------------------
# Test 3: WRAPrimalDualTrainer.primal_forward / compute_constraints shapes
# ---------------------------------------------------------------------------

class TestWRATrainerShapes:

    def test_primal_forward_shapes(self):
        B, m, n, T = 4, 8, 8, 10
        trainer, _, _, _ = _make_trainer(m=m, n=n, T=T, num_networks=4)
        batch = _make_batch(B=B, m=m, n=n, T=T, num_networks=4)
        batch = batch.to(trainer.device)

        power_batch, assoc_batch = trainer.primal_forward(batch)

        assert power_batch.shape == (B, m), f"Expected ({B}, {m}), got {power_batch.shape}"
        assert assoc_batch.shape == (B, m, n), f"Expected ({B}, {m}, {n}), got {assoc_batch.shape}"
        assert (power_batch >= 0).all(), "Powers should be non-negative (model uses sigmoid)"

    def test_compute_constraints_shapes(self):
        B, m, n, T = 4, 8, 8, 10
        trainer, _, _, _ = _make_trainer(m=m, n=n, T=T, num_networks=4)
        batch = _make_batch(B=B, m=m, n=n, T=T, num_networks=4)
        batch = batch.to(trainer.device)

        power_batch, assoc_batch = trainer.primal_forward(batch)
        objective, g, per_user_metrics = trainer.compute_constraints(
            power_batch, assoc_batch, batch
        )

        assert objective.shape == (B,), f"objective shape: {objective.shape}"
        assert g.shape == (B, n), f"g shape: {g.shape}"
        assert per_user_metrics.shape == (B, n), f"per_user_metrics shape: {per_user_metrics.shape}"

    def test_g_equals_r_min_minus_rates(self):
        """g from compute_constraints must equal r_min - per_user_metrics."""
        B, m, n, T = 3, 8, 8, 10
        r_min = 0.5
        trainer, _, _, _ = _make_trainer(m=m, n=n, T=T, num_networks=4, r_min=r_min)
        batch = _make_batch(B=B, m=m, n=n, T=T, num_networks=4)
        batch = batch.to(trainer.device)

        power_batch, assoc_batch = trainer.primal_forward(batch)
        objective, g, per_user_metrics = trainer.compute_constraints(
            power_batch, assoc_batch, batch
        )

        expected_g = r_min - per_user_metrics
        assert torch.allclose(g, expected_g, atol=1e-6), (
            f"g != r_min - rates: max diff = {(g - expected_g).abs().max():.2e}"
        )

    def test_objective_equals_sum_rates(self):
        """objective[b] == per_user_metrics[b].sum() for all b."""
        B, m, n, T = 3, 8, 8, 10
        trainer, _, _, _ = _make_trainer(m=m, n=n, T=T, num_networks=4)
        batch = _make_batch(B=B, m=m, n=n, T=T, num_networks=4)
        batch = batch.to(trainer.device)

        power_batch, assoc_batch = trainer.primal_forward(batch)
        objective, g, per_user_metrics = trainer.compute_constraints(
            power_batch, assoc_batch, batch
        )

        expected_objective = per_user_metrics.sum(dim=1)
        assert torch.allclose(objective, expected_objective, atol=1e-6), (
            f"objective != sum(rates): max diff = "
            f"{(objective - expected_objective).abs().max():.2e}"
        )


class TestConvergenceSemantics:

    def test_projected_dual_residual_zero_for_negative_g_at_boundary(self):
        """If λ=0 and g<0, projection is inactive and projected residual is zero."""
        B, n = 2, 4
        fixed_g = torch.full((B, n), -0.5)
        trainer = _make_generic_trainer(fixed_g=fixed_g, n=n, alpha_dual=0.2)
        trainer.dual_optimizer.lambdas.zero_()

        batch = _make_batch(B=B, m=n, n=n, T=2, num_networks=4)
        _, _, _, _, _, stats = trainer.compute_lagrangian_loss(batch)
        assert stats['projected_dual_residual'] == pytest.approx(0.0, abs=1e-8)

    def test_projected_dual_residual_positive_for_positive_g_at_boundary(self):
        """If λ=0 and g>0, projected residual equals mean positive subgradient magnitude."""
        B, n = 2, 4
        fixed_g = torch.full((B, n), 0.3)
        trainer = _make_generic_trainer(fixed_g=fixed_g, n=n, alpha_dual=0.2)
        trainer.dual_optimizer.lambdas.zero_()

        batch = _make_batch(B=B, m=n, n=n, T=2, num_networks=4)
        _, _, _, _, _, stats = trainer.compute_lagrangian_loss(batch)
        assert stats['projected_dual_residual'] == pytest.approx(0.3, abs=1e-6)

    def test_base_convergence_ignores_violation_metrics(self):
        """Base trainer convergence should depend only on core generic criteria."""
        trainer = _make_generic_trainer(fixed_g=torch.zeros(3, 4), n=4, alpha_dual=0.2)
        trainer.training_history['gradient_norm'] = [0.01, 0.01, 0.01]
        trainer.training_history['std_lambda'] = [1.0, 1.0, 1.0]
        trainer.training_history['projected_dual_residual'] = [0.01, 0.02, 0.01]
        # Populate ergodic buffers (λ=0, g=0 → residual=0).
        n_recv = trainer.dual_optimizer.lambdas.shape[0] * trainer.dual_optimizer.lambdas.shape[1]
        for _ in range(trainer.convergence_window):
            trainer._ergodic_dual_buffer.append(torch.zeros(n_recv))
            trainer._ergodic_slack_buffer.append(torch.zeros(n_recv))
        trainer._ergodic_buffers_updated_this_epoch = True
        # Intentionally poor violation metrics should not affect base convergence.
        trainer.training_history['violation_fraction'] = [1.0, 1.0, 1.0]
        trainer.training_history['violation_fraction_on_model_avg_rates'] = [1.0, 1.0, 1.0]
        trainer.training_history['mean_violation_slack_on_model_avg_rates'] = [1.0, 1.0, 1.0]

        converged, status = trainer.check_convergence()
        assert converged is True
        assert 'dual_stationarity' in status
        assert 'violation_fraction' not in status

    def test_dual_stationarity_uses_abs_ergodic_complementary_slackness(self):
        """Dual stationarity should threshold mean_i|lambda_bar_i * g_bar_i|, not signed value."""
        trainer = _make_generic_trainer(fixed_g=torch.zeros(3, 4), n=4, alpha_dual=0.2)
        trainer.training_history['gradient_norm'] = [0.01, 0.01, 0.01]
        trainer.training_history['std_lambda'] = [1.0, 1.0, 1.0]

        # Make ergodic complementary slackness large in magnitude:
        # mean_i |1.0 * -0.5| = 0.5 > 0.05.
        n_recv = trainer.dual_optimizer.lambdas.shape[0] * trainer.dual_optimizer.lambdas.shape[1]
        for _ in range(trainer.convergence_window):
            trainer._ergodic_dual_buffer.append(torch.ones(n_recv))
            trainer._ergodic_slack_buffer.append(torch.full((n_recv,), -0.5))
        trainer._ergodic_buffers_updated_this_epoch = True

        converged, status = trainer.check_convergence()
        assert converged is False
        assert status['dual_stationarity']['value'] == pytest.approx(0.5, abs=1e-8)
        assert status['dual_stationarity']['raw_value'] == pytest.approx(-0.5, abs=1e-8)
        assert status['dual_stationarity']['converged'] is False

    def test_dual_stationarity_abs_is_applied_before_averaging(self):
        """Positive/negative component cancellation must not hide nonzero complementary slackness."""
        trainer = _make_generic_trainer(fixed_g=torch.zeros(3, 4), n=4, alpha_dual=0.2)
        trainer.training_history['gradient_norm'] = [0.01, 0.01, 0.01]
        trainer.training_history['std_lambda'] = [1.0, 1.0, 1.0]

        n_recv = trainer.dual_optimizer.lambdas.shape[0] * trainer.dual_optimizer.lambdas.shape[1]
        alternating = torch.tensor(
            [0.5 if i % 2 == 0 else -0.5 for i in range(n_recv)],
            dtype=torch.float32,
        )
        for _ in range(trainer.convergence_window):
            trainer._ergodic_dual_buffer.append(torch.ones(n_recv))
            trainer._ergodic_slack_buffer.append(alternating)
        trainer._ergodic_buffers_updated_this_epoch = True

        converged, status = trainer.check_convergence()
        # Signed mean is ~0, but mean absolute product is 0.5 and should fail threshold=0.05.
        assert converged is False
        assert status['dual_stationarity']['raw_value'] == pytest.approx(0.0, abs=1e-8)
        assert status['dual_stationarity']['value'] == pytest.approx(0.5, abs=1e-8)
        assert status['dual_stationarity']['converged'] is False

    def test_wra_hook_keeps_legacy_violation_criteria(self):
        """WRA trainer keeps violation-based criteria as problem-specific checks."""
        trainer, _, _, _ = _make_trainer(
            m=4,
            n=4,
            T=2,
            num_networks=4,
            r_min=0.5,
            convergence_window=3,
            gradient_norm_threshold=0.1,
            dual_variance_threshold=0.1,
            dual_stationarity_threshold=0.05,
            violation_fraction_threshold=0.2,
            violation_fraction_on_model_avg_rates_threshold=0.2,
            mean_violation_slack_on_model_avg_rates_threshold=0.1,
        )
        trainer.training_history['gradient_norm'] = [0.01, 0.01, 0.01]
        trainer.training_history['std_lambda'] = [1.0, 1.0, 1.0]
        trainer.training_history['projected_dual_residual'] = [0.01, 0.01, 0.01]
        # Populate ergodic buffers (λ=0, g=0 → residual=0).
        n_recv = trainer.dual_optimizer.lambdas.shape[0] * trainer.dual_optimizer.lambdas.shape[1]
        for _ in range(trainer.convergence_window):
            trainer._ergodic_dual_buffer.append(torch.zeros(n_recv))
            trainer._ergodic_slack_buffer.append(torch.zeros(n_recv))
        trainer._ergodic_buffers_updated_this_epoch = True
        trainer.training_history['violation_fraction'] = [0.8, 0.8, 0.8]
        trainer.training_history['violation_fraction_on_model_avg_rates'] = [0.8, 0.8, 0.8]
        trainer.training_history['mean_violation_slack_on_model_avg_rates'] = [0.5, 0.5, 0.5]

        converged, status = trainer.check_convergence()
        assert converged is False
        assert 'violation_fraction' in status
        assert 'violation_fraction_on_model_avg_rates' in status
        assert 'mean_violation_slack_on_model_avg_rates' in status

        trainer.training_history['violation_fraction'] = [0.0, 0.0, 0.0]
        trainer.training_history['violation_fraction_on_model_avg_rates'] = [0.0, 0.0, 0.0]
        trainer.training_history['mean_violation_slack_on_model_avg_rates'] = [0.0, 0.0, 0.0]
        trainer._ergodic_buffers_updated_this_epoch = True
        converged, status = trainer.check_convergence()
        assert converged is True
