"""Tests for selector diagnostics organization helpers in the trainer."""

import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from graph_signal_diffusion.trainers.trainer import DiffusionTrainer


def test_extract_graph_keys_uses_dataset_and_network_ids():
    batch = SimpleNamespace(
        num_graphs=3,
        network_id=torch.tensor([2, 3, 5], dtype=torch.long),
        dataset_name=["setA", "setA", "setB"],
    )

    keys = DiffusionTrainer._extract_graph_keys(batch)
    assert keys == [
        "setA::network_2",
        "setA::network_3",
        "setB::network_5",
    ]


def test_extract_graph_keys_falls_back_to_index_without_ids():
    batch = SimpleNamespace(num_graphs=2)
    keys = DiffusionTrainer._extract_graph_keys(batch)
    assert keys == ["graph_0", "graph_1"]


def test_extract_graph_keys_broadcasts_singleton_metadata():
    batch = SimpleNamespace(
        num_graphs=4,
        network_id=torch.tensor([0], dtype=torch.long),
        dataset_name=["sp500"],
    )

    keys = DiffusionTrainer._extract_graph_keys(batch)
    assert keys == [
        "sp500::network_0",
        "sp500::network_0",
        "sp500::network_0",
        "sp500::network_0",
    ]


class _DummyModelWithStats(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 3)

    def get_architecture_stats(self):
        return {
            "model_name": "DummyModelWithStats",
            "params_total": int(sum(p.numel() for p in self.parameters())),
        }


class _DummyDiffusionWithModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _DummyModelWithStats()


class _DummyDiffusionNoModelStats(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(5, 2)


class _DummyDiffusionWithSelectorToggle(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 3)
        self.collect_selector_heavy_diagnostics = True
        self.toggle_calls = []

    def set_collect_selector_heavy_diagnostics(self, enabled: bool):
        enabled = bool(enabled)
        self.collect_selector_heavy_diagnostics = enabled
        self.toggle_calls.append(enabled)


class _DummySelectorPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.selector = nn.Linear(1, 1, bias=False)


class _DummySelectorBlock(nn.Module):
    def __init__(self, gamma: int):
        super().__init__()
        self.gamma = int(gamma)
        self.pool = _DummySelectorPool()


class _DummySelectorEncoder(nn.Module):
    def __init__(self, gammas):
        super().__init__()
        self.encoder_blocks = nn.ModuleList(
            [_DummySelectorBlock(gamma=g) for g in gammas]
        )


class _DummySelectorModel(nn.Module):
    def __init__(self, gammas):
        super().__init__()
        self.encoder = _DummySelectorEncoder(gammas=gammas)


class _DummySelectorDiffusion(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def training_loss(self, data, return_pred_stats: bool = False):
        loss = torch.zeros((), dtype=torch.float32)
        for block in self.model.encoder.encoder_blocks:
            if getattr(block, "gamma", 1) > 1:
                weight = block.pool.selector.weight
                loss = loss + (weight ** 2).sum()
        # Mirror DDPM.training_loss: (loss, pred_stats) when requested, else loss.
        if return_pred_stats:
            return loss, {}
        return loss

    def sample(self, shape, device, data=None, use_amp=False, return_selector_sampling_diagnostics=False):
        samples = torch.zeros(shape, device=device)
        if return_selector_sampling_diagnostics:
            return samples, {}
        return samples

    def add_noise(self, x, t, noise=None):
        return x


class _DummyTrainBatch:
    num_graphs = 1
    num_nodes = 1

    def to(self, device):
        return self


def test_model_architecture_stats_written_to_json(tmp_path):
    diffusion = _DummyDiffusionWithModel()
    optimizer = torch.optim.SGD(diffusion.parameters(), lr=1e-3)
    trainer = DiffusionTrainer(
        diffusion=diffusion,
        optimizer=optimizer,
        device=torch.device("cpu"),
        save_dir=str(tmp_path),
    )

    path = trainer._log_and_save_model_architecture_stats()
    assert path is not None

    stats_path = tmp_path / "model_architecture_stats.json"
    assert stats_path.exists()
    loaded = json.loads(stats_path.read_text())
    assert loaded["model_name"] == "DummyModelWithStats"
    assert loaded["params_total"] > 0
    assert loaded["diffusion_wrapper"] == "_DummyDiffusionWithModel"


def test_collect_model_architecture_stats_fallback_without_model_method(tmp_path):
    diffusion = _DummyDiffusionNoModelStats()
    optimizer = torch.optim.SGD(diffusion.parameters(), lr=1e-3)
    trainer = DiffusionTrainer(
        diffusion=diffusion,
        optimizer=optimizer,
        device=torch.device("cpu"),
        save_dir=str(tmp_path),
    )

    stats = trainer._collect_model_architecture_stats()
    assert stats["model_name"] == "_DummyDiffusionNoModelStats"
    assert stats["params_total"] > 0
    assert stats["params_trainable"] > 0
    assert "size_total_mb" in stats


def test_selector_heavy_diagnostic_cadence():
    diffusion = _DummyDiffusionWithSelectorToggle()
    optimizer = torch.optim.SGD(diffusion.parameters(), lr=1e-3)
    trainer = DiffusionTrainer(
        diffusion=diffusion,
        optimizer=optimizer,
        device=torch.device("cpu"),
        diagnose_selector_every_n_epochs=3,
    )

    # 0-indexed cadence: epoch > 0 and epoch % n == 0 (matches
    # EvalSchedule.should_eval). For n=3: fires at epochs {3, 6, 9, ...}.
    # Epoch 0 always skipped (no diagnostic history yet to collect).
    assert trainer._should_collect_heavy_selector_diagnostics(0) is False
    assert trainer._should_collect_heavy_selector_diagnostics(1) is False
    assert trainer._should_collect_heavy_selector_diagnostics(2) is False
    assert trainer._should_collect_heavy_selector_diagnostics(3) is True
    assert trainer._should_collect_heavy_selector_diagnostics(5) is False
    assert trainer._should_collect_heavy_selector_diagnostics(6) is True
    # is_last_epoch forces collection regardless of cadence
    assert trainer._should_collect_heavy_selector_diagnostics(2, is_last_epoch=True) is True


def test_selector_heavy_diagnostic_cadence_zero_disables():
    diffusion = _DummyDiffusionWithSelectorToggle()
    optimizer = torch.optim.SGD(diffusion.parameters(), lr=1e-3)
    trainer = DiffusionTrainer(
        diffusion=diffusion,
        optimizer=optimizer,
        device=torch.device("cpu"),
        diagnose_selector_every_n_epochs=0,
    )

    for epoch in range(10):
        assert trainer._should_collect_heavy_selector_diagnostics(epoch) is False


def test_selector_heavy_diagnostics_toggle_propagates_to_diffusion():
    diffusion = _DummyDiffusionWithSelectorToggle()
    optimizer = torch.optim.SGD(diffusion.parameters(), lr=1e-3)
    trainer = DiffusionTrainer(
        diffusion=diffusion,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )

    trainer._set_selector_heavy_diagnostics_enabled(False)
    trainer._set_selector_heavy_diagnostics_enabled(True)

    assert diffusion.toggle_calls == [False, True]
    assert diffusion.collect_selector_heavy_diagnostics is True


def test_diagnose_model_config_disables_selector_diagnostics():
    diffusion = _DummyDiffusionWithSelectorToggle()
    optimizer = torch.optim.SGD(diffusion.parameters(), lr=1e-3)
    trainer = DiffusionTrainer(
        diffusion=diffusion,
        optimizer=optimizer,
        device=torch.device("cpu"),
        diagnose_model={
            "enabled": False,
            "diagnose_selector_every_n_epochs": 1,
            "selector_sampling_max_network_plots": 2,
        },
    )

    assert trainer.diagnose_model_enabled is False
    for epoch in range(10):
        assert trainer._should_collect_heavy_selector_diagnostics(epoch) is False

    trainer._set_selector_heavy_diagnostics_enabled(True)
    # Disabled diagnose_model should force heavy diagnostics OFF in diffusion.
    assert diffusion.toggle_calls == [False]
    assert diffusion.collect_selector_heavy_diagnostics is False


def test_compute_selector_grad_norms_uses_active_selector_level_indexing():
    model = _DummySelectorModel(gammas=[1, 2, 1, 3])
    blocks = model.encoder.encoder_blocks

    # gamma=1 levels should be ignored entirely.
    blocks[0].pool.selector.weight.grad = torch.tensor([[9.0]])
    blocks[2].pool.selector.weight.grad = torch.tensor([[8.0]])
    # active selector levels (gamma>1) should map to level0, level1.
    blocks[1].pool.selector.weight.grad = torch.tensor([[3.0]])
    blocks[3].pool.selector.weight.grad = torch.tensor([[4.0]])

    norms = DiffusionTrainer._compute_selector_grad_norms(model)

    assert set(norms.keys()) == {
        "selector_grad_norm_level0",
        "selector_grad_norm_level1",
        "selector_grad_norm_total",
    }
    assert norms["selector_grad_norm_level0"] == pytest.approx(3.0)
    assert norms["selector_grad_norm_level1"] == pytest.approx(4.0)
    assert norms["selector_grad_norm_total"] == pytest.approx(5.0)


def test_compute_selector_grad_norms_returns_empty_when_no_active_selector_grads():
    model = _DummySelectorModel(gammas=[2, 3])
    norms = DiffusionTrainer._compute_selector_grad_norms(model)
    assert norms == {}


def test_selector_grad_norm_metrics_emitted_when_diagnose_model_disabled(tmp_path):
    class _WandbRun:
        def __init__(self):
            self.logged = []
            self.summary = {}

        def log(self, payload, step=None):
            self.logged.append((payload, step))

    model = _DummySelectorModel(gammas=[1, 2])
    diffusion = _DummySelectorDiffusion(model)
    optimizer = torch.optim.SGD(diffusion.parameters(), lr=1e-2)
    wandb_run = _WandbRun()
    trainer = DiffusionTrainer(
        diffusion=diffusion,
        optimizer=optimizer,
        device=torch.device("cpu"),
        save_dir=str(tmp_path),
        wandb_run=wandb_run,
        diagnose_model={"enabled": False},
        use_amp=False,
    )

    trainer.fit([_DummyTrainBatch()], val_loader=None, max_epochs=1)

    assert wandb_run.logged
    payload, step = wandb_run.logged[-1]
    assert step == 0
    assert "selector_grad_norm_level0" in payload
    assert "selector_grad_norm_total" in payload


def test_sampling_selector_plotting_uses_network_split_and_cap(tmp_path, monkeypatch):
    diffusion = _DummyDiffusionWithSelectorToggle()
    optimizer = torch.optim.SGD(diffusion.parameters(), lr=1e-3)
    trainer = DiffusionTrainer(
        diffusion=diffusion,
        optimizer=optimizer,
        device=torch.device("cpu"),
        save_dir=str(tmp_path),
        selector_sampling_max_network_plots=2,
    )

    calls = []

    def _fake_plot(
        selector_sampling_diagnostics,
        out_dir,
        epoch,
        split,
        dataset_name=None,
        network_id=None,
        save_pdf=True,
        batch_idx=None,
        aggregate=False,
    ):
        calls.append(
            {
                "dataset_name": dataset_name,
                "network_id": network_id,
                "split": split,
                "epoch": epoch,
            }
        )

    monkeypatch.setattr(
        "graph_signal_diffusion.utils.plotting.plot_selector_sampling_phase_heatmaps",
        _fake_plot,
    )

    selector_sampling_diagnostics = {
        "by_network": {
            "setA::network_2": {"probe_timesteps": [10, 0], "selection_given_available_by_level": {"0": [[0.4], [0.5]]}},
            "setA::network_1": {"probe_timesteps": [10, 0], "selection_given_available_by_level": {"0": [[0.6], [0.7]]}},
            "setA::network_3": {"probe_timesteps": [10, 0], "selection_given_available_by_level": {"0": [[0.2], [0.3]]}},
        }
    }

    trainer._plot_selector_sampling_diagnostics(
        selector_sampling_diagnostics=selector_sampling_diagnostics,
        data=None,
        viz_save_dir=str(tmp_path),
        eval_split="val",
    )

    # Sorted keys -> network_1, network_2, network_3; capped to first two.
    assert len(calls) == 2
    assert [c["network_id"] for c in calls] == ["network_1", "network_2"]
    assert all(c["split"] == "val" for c in calls)


def test_aggregate_sampling_selector_plotting_ignores_network_cap(tmp_path, monkeypatch):
    """The end-of-eval aggregate heatmap plots every network regardless of the
    max_network_plots cap that constrains per-batch plots."""
    diffusion = _DummyDiffusionWithSelectorToggle()
    optimizer = torch.optim.SGD(diffusion.parameters(), lr=1e-3)
    trainer = DiffusionTrainer(
        diffusion=diffusion,
        optimizer=optimizer,
        device=torch.device("cpu"),
        save_dir=str(tmp_path),
        selector_sampling_max_network_plots=2,  # would cap per-batch to 2
    )

    calls = []

    def _fake_plot(
        selector_sampling_diagnostics,
        out_dir,
        epoch,
        split,
        dataset_name=None,
        network_id=None,
        save_pdf=True,
        batch_idx=None,
        aggregate=False,
    ):
        calls.append({"network_id": network_id, "aggregate": aggregate})

    monkeypatch.setattr(
        "graph_signal_diffusion.utils.plotting.plot_selector_sampling_phase_heatmaps",
        _fake_plot,
    )

    def _net_diag(net_id, sel):
        return {
            "network_key": f"setA::{net_id}",
            "dataset_name": "setA",
            "network_id": net_id,
            "probe_timesteps": [10, 0],
            "probe_graph_count": [4.0, 4.0],
            "selected_count_by_level": {"0": [[sel], [sel]]},
            "available_count_by_level": {"0": [[4.0], [4.0]]},
        }

    per_batch = [
        {
            "probe_timesteps": [10, 0],
            "probe_graph_count": [4.0, 4.0],
            "selected_count_by_level": {"0": [[1.0], [1.0]]},
            "available_count_by_level": {"0": [[4.0], [4.0]]},
            "by_network": {
                "setA::network_2": _net_diag("network_2", 2.0),
                "setA::network_1": _net_diag("network_1", 1.0),
                "setA::network_3": _net_diag("network_3", 3.0),
            },
        }
    ]

    trainer._plot_aggregate_selector_sampling_diagnostics(
        per_batch,
        viz_save_dir=str(tmp_path),
        eval_split="val",
    )

    # All three networks plotted (cap of 2 ignored), and all flagged aggregate.
    assert len(calls) == 3
    assert sorted(c["network_id"] for c in calls) == ["network_1", "network_2", "network_3"]
    assert all(c["aggregate"] for c in calls)


def test_sampling_selector_plotting_skipped_when_diagnose_model_disabled(tmp_path, monkeypatch):
    diffusion = _DummyDiffusionWithSelectorToggle()
    optimizer = torch.optim.SGD(diffusion.parameters(), lr=1e-3)
    trainer = DiffusionTrainer(
        diffusion=diffusion,
        optimizer=optimizer,
        device=torch.device("cpu"),
        save_dir=str(tmp_path),
        diagnose_model={"enabled": False},
    )

    calls = []

    def _fake_plot(*args, **kwargs):
        calls.append(1)

    monkeypatch.setattr(
        "graph_signal_diffusion.utils.plotting.plot_selector_sampling_phase_heatmaps",
        _fake_plot,
    )

    trainer._plot_selector_sampling_diagnostics(
        selector_sampling_diagnostics={
            "probe_timesteps": [10, 0],
            "selection_given_available_by_level": {"0": [[0.6], [0.7]]},
        },
        data=None,
        viz_save_dir=str(tmp_path),
        eval_split="val",
    )
    assert calls == []


def test_accumulate_selector_level_diags_respects_tracked_network_keys():
    diffusion = _DummyDiffusionWithSelectorToggle()
    optimizer = torch.optim.SGD(diffusion.parameters(), lr=1e-3)
    trainer = DiffusionTrainer(
        diffusion=diffusion,
        optimizer=optimizer,
        device=torch.device("cpu"),
        diagnose_model={"selector_summary_max_network_rows": 2},
    )

    state = DiffusionTrainer._init_selector_diag_state()
    batch_graph_keys = [
        "setA::network_0",
        "setA::network_1",
        "setA::network_2",
    ]
    tracked_keys = trainer._resolve_selector_diag_tracked_network_keys(batch_graph_keys)
    assert tracked_keys == {"setA::network_0", "setA::network_1"}

    selector_level_diags = [
        {
            "selector_level": 0,
            "num_graphs": 3.0,
            "selected_node_counts": torch.tensor([2.0, 2.0]),
            "selected_node_counts_per_graph": torch.tensor(
                [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
            ),
            "scores_per_graph": torch.tensor(
                [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
            ),
            "soft_mask_per_graph": torch.tensor(
                [[0.2, 0.8], [0.6, 0.4], [0.5, 0.5]]
            ),
            "active_mask_per_graph": torch.tensor(
                [[True, True], [True, True], [True, True]]
            ),
        }
    ]

    DiffusionTrainer._accumulate_selector_level_diags(
        state,
        selector_level_diags,
        batch_graph_keys,
        tracked_network_keys=tracked_keys,
    )
    _, _, selector_network_stats, _, _ = DiffusionTrainer._finalize_selector_diag_state(state)
    assert sorted(selector_network_stats.keys()) == ["setA::network_0", "setA::network_1"]


def test_selector_summary_network_rows_written_only_when_enabled(tmp_path):
    diffusion = _DummyDiffusionWithSelectorToggle()
    optimizer = torch.optim.SGD(diffusion.parameters(), lr=1e-3)
    trainer = DiffusionTrainer(
        diffusion=diffusion,
        optimizer=optimizer,
        device=torch.device("cpu"),
        save_dir=str(tmp_path),
    )

    network_keys = ["setA::network_0", "setA::network_1", "setA::network_2"]
    selector_network_stats = {
        k: {"0": {"freq_mean": 0.5, "freq_std": 0.1, "num_nodes": 4, "num_graphs": 2.0}}
        for k in network_keys
    }
    selector_freq_ma_state = {
        k: {0: torch.tensor([0.2, 0.4, 0.6, 0.8])}
        for k in network_keys
    }
    selector_freq_tracked_nodes = {
        k: {0: [0, 2]}
        for k in network_keys
    }

    summary_path = trainer._append_selector_diagnostic_summaries_jsonl(
        epoch=0,
        avg_selector_diags={"temperature": 1.0, "selector_aux_total": 0.01},
        avg_selector_stats_by_level={"entropy_mean_norm": {"0": 0.5}},
        selector_level_freqs={0: torch.tensor([0.5, 0.75])},
        selector_network_stats=selector_network_stats,
        selector_freq_ma_state=selector_freq_ma_state,
        selector_freq_tracked_nodes=selector_freq_tracked_nodes,
        write_network_rows=False,
    )
    assert summary_path is not None

    summary_file = tmp_path / "selector_diagnostics" / "selector_diagnostic_summaries.jsonl"
    rows = [json.loads(line) for line in summary_file.read_text().splitlines()]
    assert [r["scope"] for r in rows] == ["global"]

    trainer._append_selector_diagnostic_summaries_jsonl(
        epoch=1,
        avg_selector_diags={"temperature": 0.9, "selector_aux_total": 0.02},
        avg_selector_stats_by_level={"entropy_mean_norm": {"0": 0.4}},
        selector_level_freqs={0: torch.tensor([0.5, 0.6])},
        selector_network_stats=selector_network_stats,
        selector_freq_ma_state=selector_freq_ma_state,
        selector_freq_tracked_nodes=selector_freq_tracked_nodes,
        write_network_rows=True,
    )
    rows = [json.loads(line) for line in summary_file.read_text().splitlines()]
    assert sum(1 for r in rows if r["scope"] == "global") == 2
    assert sum(1 for r in rows if r["scope"] == "network") == len(network_keys)


def test_selector_summary_network_rows_respect_cap(tmp_path):
    diffusion = _DummyDiffusionWithSelectorToggle()
    optimizer = torch.optim.SGD(diffusion.parameters(), lr=1e-3)
    trainer = DiffusionTrainer(
        diffusion=diffusion,
        optimizer=optimizer,
        device=torch.device("cpu"),
        save_dir=str(tmp_path),
        diagnose_model={"selector_summary_max_network_rows": 2},
    )

    selector_network_stats = {
        "setA::network_2": {"0": {"freq_mean": 0.5, "freq_std": 0.1, "num_nodes": 4, "num_graphs": 2.0}},
        "setA::network_1": {"0": {"freq_mean": 0.5, "freq_std": 0.1, "num_nodes": 4, "num_graphs": 2.0}},
        "setA::network_3": {"0": {"freq_mean": 0.5, "freq_std": 0.1, "num_nodes": 4, "num_graphs": 2.0}},
    }
    selector_freq_ma_state = {
        k: {0: torch.tensor([0.2, 0.4, 0.6, 0.8])}
        for k in selector_network_stats.keys()
    }

    trainer._append_selector_diagnostic_summaries_jsonl(
        epoch=0,
        avg_selector_diags={"temperature": 1.0, "selector_aux_total": 0.01},
        avg_selector_stats_by_level={"entropy_mean_norm": {"0": 0.5}},
        selector_level_freqs={0: torch.tensor([0.5, 0.75])},
        selector_network_stats=selector_network_stats,
        selector_freq_ma_state=selector_freq_ma_state,
        selector_freq_tracked_nodes=None,
        write_network_rows=True,
    )

    summary_file = tmp_path / "selector_diagnostics" / "selector_diagnostic_summaries.jsonl"
    rows = [json.loads(line) for line in summary_file.read_text().splitlines()]
    network_rows = [r for r in rows if r["scope"] == "network"]
    assert len(network_rows) == 2
    # Fallback ordering is sorted when no tracked-network order is available.
    assert [r["network_key"] for r in network_rows] == [
        "setA::network_1",
        "setA::network_2",
    ]


def test_sampling_selector_plotting_respects_nested_toggle_and_cap(tmp_path, monkeypatch):
    diffusion = _DummyDiffusionWithSelectorToggle()
    optimizer = torch.optim.SGD(diffusion.parameters(), lr=1e-3)
    trainer = DiffusionTrainer(
        diffusion=diffusion,
        optimizer=optimizer,
        device=torch.device("cpu"),
        save_dir=str(tmp_path),
        diagnose_model={
            "plot_sampling_heatmap": {
                "enabled": True,
                "max_network_plots": 1,
                "plot_every_n_epochs": None,
            }
        },
    )
    trainer.epoch = 0

    calls = []

    def _fake_plot(
        selector_sampling_diagnostics,
        out_dir,
        epoch,
        split,
        dataset_name=None,
        network_id=None,
        save_pdf=True,
        batch_idx=None,
        aggregate=False,
    ):
        calls.append(network_id)

    monkeypatch.setattr(
        "graph_signal_diffusion.utils.plotting.plot_selector_sampling_phase_heatmaps",
        _fake_plot,
    )

    selector_sampling_diagnostics = {
        "by_network": {
            "setA::network_2": {"probe_timesteps": [10, 0], "selection_given_available_by_level": {"0": [[0.4], [0.5]]}},
            "setA::network_1": {"probe_timesteps": [10, 0], "selection_given_available_by_level": {"0": [[0.6], [0.7]]}},
            "setA::network_3": {"probe_timesteps": [10, 0], "selection_given_available_by_level": {"0": [[0.2], [0.3]]}},
        }
    }
    trainer._plot_selector_sampling_diagnostics(
        selector_sampling_diagnostics=selector_sampling_diagnostics,
        data=None,
        viz_save_dir=str(tmp_path),
        eval_split="val",
    )
    assert calls == ["network_1"]

    trainer_disabled = DiffusionTrainer(
        diffusion=diffusion,
        optimizer=optimizer,
        device=torch.device("cpu"),
        save_dir=str(tmp_path),
        diagnose_model={"plot_sampling_heatmap": {"enabled": False}},
    )
    trainer_disabled.epoch = 0
    trainer_disabled._plot_selector_sampling_diagnostics(
        selector_sampling_diagnostics=selector_sampling_diagnostics,
        data=None,
        viz_save_dir=str(tmp_path),
        eval_split="val",
    )
    assert calls == ["network_1"]


def test_sampling_selector_plotting_zero_period_means_every_eval(tmp_path, monkeypatch):
    diffusion = _DummyDiffusionWithSelectorToggle()
    optimizer = torch.optim.SGD(diffusion.parameters(), lr=1e-3)
    trainer = DiffusionTrainer(
        diffusion=diffusion,
        optimizer=optimizer,
        device=torch.device("cpu"),
        save_dir=str(tmp_path),
        diagnose_model={
            "plot_sampling_heatmap": {
                "enabled": True,
                "max_network_plots": 1,
                "plot_every_n_epochs": 0,
            }
        },
    )
    trainer.epoch = 4

    calls = []

    def _fake_plot(
        selector_sampling_diagnostics,
        out_dir,
        epoch,
        split,
        dataset_name=None,
        network_id=None,
        save_pdf=True,
        batch_idx=None,
        aggregate=False,
    ):
        calls.append(network_id)

    monkeypatch.setattr(
        "graph_signal_diffusion.utils.plotting.plot_selector_sampling_phase_heatmaps",
        _fake_plot,
    )

    trainer._plot_selector_sampling_diagnostics(
        selector_sampling_diagnostics={
            "by_network": {
                "setA::network_7": {
                    "probe_timesteps": [10, 0],
                    "selection_given_available_by_level": {"0": [[0.4], [0.5]]},
                }
            }
        },
        data=None,
        viz_save_dir=str(tmp_path),
        eval_split="val",
    )
    assert calls == ["network_7"]


def test_epoch_selector_plotting_null_node_cadence_falls_back_to_selector_heavy_cadence(
    tmp_path,
    monkeypatch,
):
    diffusion = _DummyDiffusionWithSelectorToggle()
    optimizer = torch.optim.SGD(diffusion.parameters(), lr=1e-3)
    trainer = DiffusionTrainer(
        diffusion=diffusion,
        optimizer=optimizer,
        device=torch.device("cpu"),
        save_dir=str(tmp_path),
        diagnose_model={
            "diagnose_selector_every_n_epochs": 2,
            "plot_node_stats": {
                "enabled": True,
                "max_network_plots": 1,
                "plot_every_n_epochs": None,
            },
            "plot_hard_mask": {
                "enabled": False,
                "plot_every_n_epochs": 1,
            },
            "plot_freq_evolution": {
                "enabled": False,
                "plot_every_n_epochs": 1,
            },
            "plot_scalar_series": {
                "enabled": False,
                "plot_every_n_epochs": 1,
            },
        },
    )

    calls = []

    def _fake_node_stats(selector_network_node_stats, *args, **kwargs):
        calls.append(len(selector_network_node_stats))

    monkeypatch.setattr(
        "graph_signal_diffusion.utils.plotting.plot_selector_network_node_statistics",
        _fake_node_stats,
    )

    selector_network_node_stats = {
        "setA::network_2": {0: {"hard_mask_mean": torch.tensor([0.1, 0.2])}},
        "setA::network_1": {0: {"hard_mask_mean": torch.tensor([0.3, 0.4])}},
    }

    # Plot cadence semantics: epoch > 0 and epoch % period == 0 (matches
    # EvalSchedule.should_eval). With heavy cadence = 2 and node_stats
    # falling back to it, only epoch 2 fires (epoch 0 always skipped, epoch 1
    # not a multiple of 2).
    trainer._plot_epoch_artifacts(
        epoch=0,
        epoch_summary_path=None,
        selector_diag_summary_path=str(tmp_path / "selector_diagnostic_summaries.jsonl"),
        selector_network_node_stats=selector_network_node_stats,
    )
    trainer._plot_epoch_artifacts(
        epoch=1,
        epoch_summary_path=None,
        selector_diag_summary_path=str(tmp_path / "selector_diagnostic_summaries.jsonl"),
        selector_network_node_stats=selector_network_node_stats,
    )
    trainer._plot_epoch_artifacts(
        epoch=2,
        epoch_summary_path=None,
        selector_diag_summary_path=str(tmp_path / "selector_diagnostic_summaries.jsonl"),
        selector_network_node_stats=selector_network_node_stats,
    )
    assert calls == [1]


def test_epoch_selector_plotting_respects_nested_toggles_and_caps(tmp_path, monkeypatch):
    diffusion = _DummyDiffusionWithSelectorToggle()
    optimizer = torch.optim.SGD(diffusion.parameters(), lr=1e-3)
    trainer = DiffusionTrainer(
        diffusion=diffusion,
        optimizer=optimizer,
        device=torch.device("cpu"),
        save_dir=str(tmp_path),
        diagnose_model={
            "plot_node_stats": {
                "enabled": True,
                "max_network_plots": 1,
                "plot_every_n_epochs": 1,
            },
            "plot_hard_mask": {
                "enabled": False,
                "max_network_plots": 1,
                "plot_every_n_epochs": 1,
            },
            "plot_freq_evolution": {
                "enabled": True,
                "max_network_plots": 2,
                "plot_every_n_epochs": 1,
            },
            "plot_scalar_series": {
                "enabled": False,
                "plot_every_n_epochs": 1,
            },
        },
    )

    calls = {
        "scalar_series": 0,
        "freq_evolution": [],
        "node_stats": [],
        "hard_mask": 0,
    }

    def _fake_scalar_series(*args, **kwargs):
        calls["scalar_series"] += 1

    def _fake_freq_evolution(*args, **kwargs):
        calls["freq_evolution"].append(kwargs.get("max_network_plots"))

    def _fake_node_stats(selector_network_node_stats, *args, **kwargs):
        calls["node_stats"].append(len(selector_network_node_stats))

    def _fake_hard_mask(*args, **kwargs):
        calls["hard_mask"] += 1

    monkeypatch.setattr(
        "graph_signal_diffusion.utils.plotting.plot_selector_diagnostics_jsonl",
        _fake_scalar_series,
    )
    monkeypatch.setattr(
        "graph_signal_diffusion.utils.plotting.plot_selector_network_selection_frequency_evolution",
        _fake_freq_evolution,
    )
    monkeypatch.setattr(
        "graph_signal_diffusion.utils.plotting.plot_selector_network_node_statistics",
        _fake_node_stats,
    )
    monkeypatch.setattr(
        "graph_signal_diffusion.utils.plotting.plot_selector_network_hard_mask_statistics",
        _fake_hard_mask,
    )

    selector_network_node_stats = {
        "setA::network_2": {0: {"hard_mask_mean": torch.tensor([0.1, 0.2])}},
        "setA::network_1": {0: {"hard_mask_mean": torch.tensor([0.3, 0.4])}},
        "setA::network_3": {0: {"hard_mask_mean": torch.tensor([0.5, 0.6])}},
    }

    # Plot cadence skips epoch 0 (epoch > 0 and epoch % period == 0); use
    # epoch 1 with period 1 so all enabled plots fire.
    trainer._plot_epoch_artifacts(
        epoch=1,
        epoch_summary_path=None,
        selector_diag_summary_path=str(tmp_path / "selector_diagnostic_summaries.jsonl"),
        selector_network_node_stats=selector_network_node_stats,
    )

    assert calls["scalar_series"] == 0
    assert calls["hard_mask"] == 0
    assert calls["node_stats"] == [1]
    assert calls["freq_evolution"] == [2]
