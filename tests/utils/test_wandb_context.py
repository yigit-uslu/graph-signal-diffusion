"""Unit tests for wandb context derivation utilities."""
from __future__ import annotations

import pytest
from omegaconf import OmegaConf

from graph_signal_diffusion.utils.wandb_context import (
    build_wandb_context,
    merge_wandb_context,
    shorten,
)


# ---------------------------------------------------------------------------
# shorten()
# ---------------------------------------------------------------------------


def test_shorten_known_alias():
    assert shorten("stock_price_forecasting_v2") == "stock"
    assert shorten("stock_price_forecasting") == "stock"
    assert shorten("wireless_resource_allocation") == "wra"
    assert shorten("sp500_cleaned") == "sp500c"


def test_shorten_passthrough():
    assert shorten("sp500") == "sp500"
    assert shorten("ddim") == "ddim"


def test_shorten_custom_aliases():
    aliases = {"foo": "f", "bar_baz": "bb"}
    assert shorten("foo", aliases) == "f"
    assert shorten("bar_baz", aliases) == "bb"
    # Known default alias should NOT apply when custom aliases are given
    assert shorten("stock_price_forecasting_v2", aliases) == "stock_price_forecasting_v2"


# ---------------------------------------------------------------------------
# build_wandb_context() — training stage
# ---------------------------------------------------------------------------


def _make_train_cfg(
    task_name: str = "stock_price_forecasting_v2",
    dataset_name: str = "sp500",
    model_name: str = "ugnn_sp500_v2_learned_ds2",
    diffusion_name: str = "ddim",
    selection_method: str | None = "learned",
    scenario: str | None = None,
) -> OmegaConf:
    d: dict = {
        "task": {"name": task_name},
        "dataset": {"name": dataset_name},
        "model": {"name": model_name, "config": {}},
        "diffusion": {"name": diffusion_name},
    }
    if selection_method is not None:
        d["model"]["config"]["pooling_config"] = {"selection_method": selection_method}
    if scenario is not None:
        d["dataset"]["scenario"] = scenario
    return OmegaConf.create(d)


def test_build_context_train_sp500():
    cfg = _make_train_cfg()
    ctx = build_wandb_context(cfg, stage="train")

    assert ctx["group"] == "train/stock/sp500"
    assert ctx["job_type"] == "train"

    tags = ctx["auto_tags"]
    assert "stage:train" in tags
    assert "task:stock" in tags
    assert "dataset:sp500" in tags
    assert "model:ugnn_sp500_v2_learned_ds2" in tags
    assert "diffusion:ddim" in tags
    assert "pool:learned" in tags


def test_build_context_train_wra_scenario():
    cfg = _make_train_cfg(
        task_name="wireless_resource_allocation",
        dataset_name="wra",
        model_name="ugnn_wra",
        diffusion_name="ddim_wra",
        selection_method=None,
        scenario="large_low_density",
    )
    ctx = build_wandb_context(cfg, stage="train")

    assert ctx["group"] == "train/wra/wra-large-low-density"
    assert ctx["job_type"] == "train"

    tags = ctx["auto_tags"]
    assert "scenario:large-low-density" in tags
    assert "model:ugnn_wra" in tags
    assert "diffusion:ddim_wra" in tags
    # No pool tag when selection_method is absent
    assert not any(t.startswith("pool:") for t in tags)


def test_build_context_train_no_pool_tag_for_stride():
    """When selection_method is 'stride' (identity), no pool tag is emitted."""
    cfg = _make_train_cfg(selection_method="stride")
    ctx = build_wandb_context(cfg, stage="train")
    assert not any(t.startswith("pool:") for t in ctx["auto_tags"])


# ---------------------------------------------------------------------------
# build_wandb_context() — eval stage
# ---------------------------------------------------------------------------


def test_build_context_eval():
    cfg = OmegaConf.create({
        "task": {"name": "stock_price_forecasting_v2"},
        "dataset": {"name": "sp500"},
    })
    ctx = build_wandb_context(cfg, stage="eval", baseline_name="diffusion")

    assert ctx["group"] == "eval/stock/sp500"
    assert ctx["job_type"] == "baseline-eval"

    tags = ctx["auto_tags"]
    assert "stage:eval" in tags
    assert "task:stock" in tags
    assert "dataset:sp500" in tags
    assert "baseline:diffusion" in tags
    # No model/diffusion/pool tags in eval
    assert not any(t.startswith("model:") for t in tags)
    assert not any(t.startswith("diffusion:") for t in tags)
    assert not any(t.startswith("pool:") for t in tags)


# ---------------------------------------------------------------------------
# build_wandb_context() — compare stage
# ---------------------------------------------------------------------------


def test_build_context_compare_includes_baseline_set_and_job_type():
    cfg = OmegaConf.create({
        "task": {"name": "stock_price_forecasting_v2"},
        "dataset": {"name": "sp500"},
    })
    ctx = build_wandb_context(
        cfg,
        stage="compare",
        baseline_names=["grw", "diffusion"],
    )

    assert ctx["group"] == "compare/stock/sp500/diffusion+grw"
    assert ctx["job_type"] == "baseline-compare"

    tags = ctx["auto_tags"]
    assert "stage:compare" in tags
    assert "baseline:diffusion" in tags
    assert "baseline:grw" in tags
    assert "baselines:diffusion+grw" in tags


def test_build_context_compare_is_order_invariant():
    cfg = OmegaConf.create({
        "task": {"name": "stock_price_forecasting_v2"},
        "dataset": {"name": "sp500"},
    })
    ctx_a = build_wandb_context(
        cfg,
        stage="compare",
        baseline_names=["grw", "diffusion"],
    )
    ctx_b = build_wandb_context(
        cfg,
        stage="compare",
        baseline_names=["diffusion", "grw"],
    )

    assert ctx_a["group"] == ctx_b["group"]
    assert ctx_a["job_type"] == ctx_b["job_type"]
    assert ctx_a["auto_tags"] == ctx_b["auto_tags"]


# ---------------------------------------------------------------------------
# merge_wandb_context()
# ---------------------------------------------------------------------------


def test_merge_explicit_group_wins():
    cfg = OmegaConf.create({
        "task": {"name": "stock_price_forecasting_v2"},
        "dataset": {"name": "sp500"},
        "wandb": {"group": "my-custom-group", "tags": []},
    })
    ctx = build_wandb_context(cfg, stage="train")
    merged = merge_wandb_context(cfg, ctx)

    assert merged["group"] == "my-custom-group"


def test_merge_tags_extend():
    cfg = OmegaConf.create({
        "task": {"name": "stock_price_forecasting_v2"},
        "dataset": {"name": "sp500"},
        "wandb": {"group": None, "tags": ["custom-tag", "stage:train"]},
    })
    ctx = build_wandb_context(cfg, stage="train")
    merged = merge_wandb_context(cfg, ctx)

    # custom-tag should be present
    assert "custom-tag" in merged["tags"]
    # auto-tags should be present
    assert "task:stock" in merged["tags"]
    assert "dataset:sp500" in merged["tags"]
    # Duplicates should be removed (stage:train appears in both auto and user)
    assert merged["tags"].count("stage:train") == 1


def test_merge_null_group_uses_auto():
    cfg = OmegaConf.create({
        "task": {"name": "stock_price_forecasting_v2"},
        "dataset": {"name": "sp500"},
        "wandb": {"group": None, "tags": []},
    })
    ctx = build_wandb_context(cfg, stage="train")
    merged = merge_wandb_context(cfg, ctx)

    assert merged["group"] == "train/stock/sp500"


def test_merge_no_wandb_section():
    """When cfg has no wandb section, merge still works using auto defaults."""
    cfg = OmegaConf.create({
        "task": {"name": "wireless_resource_allocation"},
        "dataset": {"name": "wra"},
    })
    ctx = build_wandb_context(cfg, stage="train")
    merged = merge_wandb_context(cfg, ctx)

    assert merged["group"] == "train/wra/wra"
    assert "stage:train" in merged["tags"]
