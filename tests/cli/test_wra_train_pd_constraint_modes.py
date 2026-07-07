import torch
from omegaconf import OmegaConf

from graph_signal_diffusion.cli.wra import train_pd
from graph_signal_diffusion.datasets.wra.primal_dual_dataset import (
    WRAPrimalDualDataset,
    WirelessData,
)
from graph_signal_diffusion.trainers.dual_optimizer import DualOptimizer


def _make_base_dataset(num_networks: int = 2, n_links: int = 3) -> WRAPrimalDualDataset:
    dataset = WRAPrimalDualDataset.__new__(WRAPrimalDualDataset)
    dataset.channels = []
    dataset.samples = []
    for net_id in range(num_networks):
        sample = WirelessData(
            x=torch.randn(n_links, 2),
            edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
            edge_weight=torch.tensor([1.0, 1.0], dtype=torch.float32),
            H_instantaneous=torch.ones((2, n_links, n_links), dtype=torch.float32),
            H_l=torch.eye(n_links, dtype=torch.float32),
            associations=torch.eye(n_links, dtype=torch.float32),
            network_id=torch.tensor(net_id, dtype=torch.long),
            network_seed=torch.tensor(100 + net_id, dtype=torch.long),
        )
        dataset.samples.append(sample)
    return dataset


def _base_cfg(*, n_links: int, source: str, payload: dict):
    training_cfg = {
        "r_min": 0.5,
        "constraint_profile": {
            "type": "min_rate",
            "source": source,
            "r_min_feature_scale": 1.0,
            "scalar": {"value": 0.5},
            "explicit": {"profiles": [], "names": []},
            "sampled": {"min": None, "max": None, "count": None, "seed": None},
        },
    }
    training_cfg["constraint_profile"][source] = payload
    return OmegaConf.create(
        {
            "seed": 123,
            "dataset": {"n_links": n_links},
            "training": training_cfg,
        }
    )


def test_constraint_mode_scalar_keeps_base_dataset_and_scalar_dual_rmin():
    n_links = 3
    base_dataset = _make_base_dataset(num_networks=2, n_links=n_links)
    cfg = _base_cfg(n_links=n_links, source="scalar", payload={"value": 0.55})

    profile_spec = train_pd._resolve_constraint_profile_spec(cfg)
    conditioning = train_pd._build_constraint_conditioned_dataset(base_dataset, profile_spec)

    assert conditioning["constraint_source"] == "scalar"
    assert conditioning["constraint_dataset"] is None
    assert len(conditioning["train_dataset"]) == 2
    assert conditioning["model_input_dim"] == 2

    dual = DualOptimizer(
        num_networks=len(conditioning["train_dataset"]),
        num_receivers=n_links,
        r_min=conditioning["dual_r_min_input"],
        alpha_dual=0.1,
        update_frequency=1,
        momentum=0.0,
        device="cpu",
    )
    assert dual.r_min_table.shape == (2, n_links)
    assert dual.is_scalar_r_min()


def test_constraint_mode_explicit_expands_networks_and_dual_table():
    n_links = 3
    base_dataset = _make_base_dataset(num_networks=2, n_links=n_links)
    cfg = _base_cfg(
        n_links=n_links,
        source="explicit",
        payload={"profiles": [0.3, [0.4, 0.5, 0.6]], "names": ["low", "med"]},
    )

    profile_spec = train_pd._resolve_constraint_profile_spec(cfg)
    conditioning = train_pd._build_constraint_conditioned_dataset(base_dataset, profile_spec)

    assert conditioning["constraint_source"] == "explicit"
    assert conditioning["constraint_dataset"] is not None
    assert len(conditioning["train_dataset"]) == 4  # 2 base networks x 2 profiles
    assert conditioning["model_input_dim"] == 3
    assert conditioning["dual_r_min_input"].shape == (4, n_links)

    dual = DualOptimizer(
        num_networks=len(conditioning["train_dataset"]),
        num_receivers=n_links,
        r_min=conditioning["dual_r_min_input"],
        alpha_dual=0.1,
        update_frequency=1,
        momentum=0.0,
        device="cpu",
    )
    assert dual.r_min_table.shape == (4, n_links)
    assert not dual.is_scalar_r_min()


def test_constraint_mode_sampled_expands_networks_and_dual_table():
    n_links = 4
    base_dataset = _make_base_dataset(num_networks=3, n_links=n_links)
    cfg = _base_cfg(
        n_links=n_links,
        source="sampled",
        payload={"min": 0.2, "max": 0.8, "count": 3, "seed": 7},
    )

    profile_spec = train_pd._resolve_constraint_profile_spec(cfg)
    conditioning = train_pd._build_constraint_conditioned_dataset(base_dataset, profile_spec)

    assert conditioning["constraint_source"] == "sampled"
    assert conditioning["constraint_dataset"] is not None
    assert len(conditioning["train_dataset"]) == 9  # 3 base networks x 3 sampled profiles
    assert conditioning["model_input_dim"] == 3
    assert conditioning["dual_r_min_input"].shape == (9, n_links)

    dual = DualOptimizer(
        num_networks=len(conditioning["train_dataset"]),
        num_receivers=n_links,
        r_min=conditioning["dual_r_min_input"],
        alpha_dual=0.1,
        update_frequency=1,
        momentum=0.0,
        device="cpu",
    )
    assert dual.r_min_table.shape == (9, n_links)


def test_trainer_selection_uses_conditional_subclass_only_for_wrapped_dataset(monkeypatch):
    class ScalarTrainerSpy:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class ConditionalTrainerSpy:
        def __init__(self, *, constraint_profile_dataset, **kwargs):
            self.constraint_profile_dataset = constraint_profile_dataset
            self.kwargs = kwargs

    monkeypatch.setattr(train_pd, "WRAPrimalDualTrainer", ScalarTrainerSpy)
    monkeypatch.setattr(train_pd, "WRAConditionalPrimalDualTrainer", ConditionalTrainerSpy)

    scalar_trainer = train_pd._build_pd_trainer(None, {"foo": 1})
    assert isinstance(scalar_trainer, ScalarTrainerSpy)
    assert scalar_trainer.kwargs["foo"] == 1

    dataset_marker = object()
    conditional_trainer = train_pd._build_pd_trainer(dataset_marker, {"bar": 2})
    assert isinstance(conditional_trainer, ConditionalTrainerSpy)
    assert conditional_trainer.constraint_profile_dataset is dataset_marker
    assert conditional_trainer.kwargs["bar"] == 2
