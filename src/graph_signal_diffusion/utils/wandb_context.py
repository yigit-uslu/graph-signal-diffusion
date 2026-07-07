"""Shared W&B context derivation for training and evaluation CLIs.

Provides deterministic auto-generation of ``group``, ``tags``, and
``job_type`` from the Hydra config, with explicit config values always
winning over auto-derived defaults.
"""
from __future__ import annotations

from omegaconf import DictConfig, OmegaConf


# ---------------------------------------------------------------------------
# Display aliases — shorten verbose config names for tags / group paths
# ---------------------------------------------------------------------------

_DEFAULT_DISPLAY_ALIASES: dict[str, str] = {
    "stock_price_forecasting": "stock",
    "stock_price_forecasting_v2": "stock",
    "wireless_resource_allocation": "wra",
    "sp500_cleaned": "sp500c",
}


def shorten(name: str, aliases: dict[str, str] | None = None) -> str:
    """Shorten a config name using the alias mapping.

    Returns *name* unchanged when no alias is defined for it.
    """
    lookup = aliases if aliases is not None else _DEFAULT_DISPLAY_ALIASES
    return lookup.get(name, name)


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------


def build_wandb_context(
    cfg: DictConfig,
    stage: str,
    *,
    baseline_name: str | None = None,
    baseline_names: list[str] | None = None,
    display_aliases: dict[str, str] | None = None,
) -> dict:
    """Derive W&B metadata from the Hydra config.

    Parameters
    ----------
    cfg:
        Resolved Hydra config (must contain at least ``task`` and ``dataset``).
    stage:
        One of ``"train"``, ``"eval"``, or ``"compare"``.
    baseline_name:
        Name of the baseline being evaluated (only used when *stage* is
        ``"eval"``).
    baseline_names:
        Names of all baselines in a comparison run (only used when *stage*
        is ``"compare"``). The set is canonicalized (sorted, deduplicated)
        so grouping is order-invariant.
    display_aliases:
        Optional override for the shortening map.  ``None`` uses the
        module-level ``_DEFAULT_DISPLAY_ALIASES``.

    Returns
    -------
    dict with keys ``group``, ``job_type``, ``auto_tags``.
    """
    if stage not in {"train", "eval", "compare"}:
        raise ValueError(f"Unsupported wandb context stage: {stage}")

    # ── Read config values ────────────────────────────────────────────
    task_name = str(OmegaConf.select(cfg, "task.name", default="unknown"))
    dataset_name = str(OmegaConf.select(cfg, "dataset.name", default="unknown"))
    model_name = OmegaConf.select(cfg, "model.name", default=None)
    diffusion_name = OmegaConf.select(cfg, "diffusion.name", default=None)
    pool_method = OmegaConf.select(
        cfg, "model.config.pooling_config.selection_method", default=None,
    )
    scenario = OmegaConf.select(cfg, "dataset.scenario", default=None)

    # ── Shortened display names ───────────────────────────────────────
    task_short = shorten(task_name, display_aliases)
    dataset_short = shorten(dataset_name, display_aliases)

    # Canonicalized baseline names for compare/eval tagging.
    names_raw = baseline_names or ([] if baseline_name is None else [baseline_name])
    canonical_baselines = sorted({str(n) for n in names_raw if n is not None})
    baseline_set = "+".join(canonical_baselines) if canonical_baselines else "none"

    # ── Group ─────────────────────────────────────────────────────────
    dataset_part = dataset_short
    if scenario is not None:
        dataset_part = f"{dataset_short}-{str(scenario).replace('_', '-')}"
    if stage == "compare":
        group = f"compare/{task_short}/{dataset_part}/{baseline_set}"
    else:
        group = f"{stage}/{task_short}/{dataset_part}"

    # ── Job type ──────────────────────────────────────────────────────
    if stage == "train":
        job_type = "train"
    elif stage == "eval":
        job_type = "baseline-eval"
    else:
        job_type = "baseline-compare"

    # ── Auto tags ─────────────────────────────────────────────────────
    auto_tags: list[str] = [
        f"stage:{stage}",
        f"task:{task_short}",
        f"dataset:{dataset_short}",
    ]
    if scenario is not None:
        auto_tags.append(f"scenario:{str(scenario).replace('_', '-')}")

    if stage == "train":
        if model_name is not None:
            auto_tags.append(f"model:{model_name}")
        if diffusion_name is not None:
            auto_tags.append(f"diffusion:{diffusion_name}")
        if pool_method == "learned":
            auto_tags.append("pool:learned")
    elif stage == "eval":
        if baseline_name is not None:
            auto_tags.append(f"baseline:{baseline_name}")
    elif stage == "compare":
        if canonical_baselines:
            auto_tags.extend([f"baseline:{bn}" for bn in canonical_baselines])
            auto_tags.append(f"baselines:{baseline_set}")

    return {"group": group, "job_type": job_type, "auto_tags": auto_tags}


# ---------------------------------------------------------------------------
# Merge helper
# ---------------------------------------------------------------------------


def merge_wandb_context(cfg: DictConfig, context: dict) -> dict:
    """Merge auto-derived context with explicit config; config wins.

    Returns a dict with keys ``group``, ``job_type``, ``tags`` ready for
    passing to ``wandb.init``.

    *   ``group``: uses ``cfg.wandb.group`` when non-null, otherwise the
        auto-derived group from *context*.
    *   ``job_type``: always from *context* (no config field yet).
    *   ``tags``: *context["auto_tags"]* extended with
        ``cfg.wandb.tags``, deduplicated while preserving order.
    """
    # Group: explicit config wins
    explicit_group = OmegaConf.select(cfg, "wandb.group", default=None)
    group = explicit_group if explicit_group is not None else context["group"]

    # Tags: auto first, then user, deduplicated
    user_tags = list(OmegaConf.select(cfg, "wandb.tags", default=[]) or [])
    seen: set[str] = set()
    merged_tags: list[str] = []
    for tag in context["auto_tags"] + user_tags:
        if tag not in seen:
            seen.add(tag)
            merged_tags.append(tag)

    return {
        "group": group,
        "job_type": context["job_type"],
        "tags": merged_tags,
    }
