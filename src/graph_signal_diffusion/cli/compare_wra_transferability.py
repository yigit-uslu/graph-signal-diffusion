"""Standalone WRA transferability evaluation CLI.

Thin wrapper around the shared transferability helpers used by the
``compare_baselines`` integration.  Supports the same disk-based
``WRADataset.process()`` path for graph construction parity with training.

Usage:
    python -m graph_signal_diffusion.cli.compare_wra_transferability \
        source.checkpoint_path=outputs/.../best_model.pt

    # Override targets:
    python -m graph_signal_diffusion.cli.compare_wra_transferability \
        source.checkpoint_path=outputs/.../best_model.pt \
        'targets=[{scenario: wra_mega_outdoor_low_density, eval_num_networks: 4}]'
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import hydra
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from graph_signal_diffusion.baselines import discover_baselines
from graph_signal_diffusion.tasks import discover_tasks


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True, throw_on_missing=False)
    return value if isinstance(value, dict) else {}


def _instantiate_baseline(baseline_cfg_section: DictConfig, device: torch.device):
    """Instantiate a baseline (same deferred-checkpoint logic as compare_baselines)."""
    baseline_dict = OmegaConf.to_container(baseline_cfg_section, resolve=True)
    if not isinstance(baseline_dict, dict):
        raise ValueError("Baseline config section must resolve to a dictionary.")

    post_init: dict[str, Any] = {}
    for key in ("diffusion_overrides",):
        value = baseline_dict.pop(key, None)
        if value:
            post_init[key] = value

    deferred_checkpoint = None
    if post_init and baseline_dict.get("checkpoint_path"):
        deferred_checkpoint = baseline_dict.pop("checkpoint_path")

    baseline_cfg = OmegaConf.create(baseline_dict)
    baseline = instantiate(baseline_cfg, device=device)

    for key, value in post_init.items():
        setattr(baseline, key, value)
    if deferred_checkpoint is not None:
        baseline.load_checkpoint(deferred_checkpoint)

    if bool(baseline_cfg.get("needs_training", False)):
        raise ValueError(
            f"Baseline '{baseline_cfg.get('name', '?')}' requires training, "
            "which is not supported in transferability evaluation."
        )
    return baseline


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="wra_generation/transferability/defaults",
)
def main(cfg: DictConfig) -> None:
    discover_tasks()
    discover_baselines()

    from graph_signal_diffusion.cli._wra_transferability import (
        as_dict,
        build_channel_cache_info_entry,
        build_or_load_transfer_channel_cache,
        build_target_system_params,
        create_transfer_r_min_variant,
        enforce_pmax_compatibility,
        extract_graph_build_cfg,
        find_compatible_channel_cache,
        load_checkpoint_hydra_config,
        load_wra_scenario_config,
        plot_transferability_results,
        resolve_model_cond_channels,
        resolve_path_from_runtime_cwd,
        resolve_source_system_params,
        save_transferability_summary,
        select_tail_network_ids,
        synthesize_transfer_raw_dir,
        transfer_dataset_name,
        validate_transfer_baselines,
    )
    from graph_signal_diffusion.cli._wra_r_min_sweep import build_sweep_dataset_info
    from graph_signal_diffusion.datasets.wra.dataset import WRADataset
    from graph_signal_diffusion.datasets.wra.sampler import NetworkGroupedBatchSampler
    from graph_signal_diffusion.datasets.wra.channel_factory import (
        build_wra_channel_cache_metadata,
        normalize_channel_version,
        sanitize_channel_params,
    )
    from torch_geometric.loader import DataLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    hydra_cfg = HydraConfig.get()
    runtime_cwd = Path(hydra_cfg.runtime.cwd).resolve()
    output_dir = Path(hydra_cfg.runtime.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    conf_root = Path(__file__).resolve().parents[1] / "conf"

    # --- Source checkpoint ---
    source_ckpt_raw = cfg.source.get("checkpoint_path", None)
    if not source_ckpt_raw:
        raise ValueError("source.checkpoint_path must be provided.")
    source_ckpt_path = resolve_path_from_runtime_cwd(str(source_ckpt_raw), runtime_cwd)
    source_ckpt_cfg = load_checkpoint_hydra_config(source_ckpt_path)
    source_system_params = resolve_source_system_params(source_ckpt_cfg)
    model_cond_channels = resolve_model_cond_channels(
        source_ckpt_cfg, cfg.source.get("model_cond_channels", None),
    )
    graph_cfg = extract_graph_build_cfg(source_ckpt_cfg)
    source_p_max_dBm = source_system_params.get("P_max_dBm")

    # --- Baselines ---
    baseline_names = [str(n) for n in cfg.baselines_to_compare]
    validate_transfer_baselines(baseline_names, reference_policy="none")

    missing_baselines = [n for n in baseline_names if n not in cfg.baselines]
    if missing_baselines:
        raise ValueError(
            f"Baselines {missing_baselines} are listed in baselines_to_compare "
            f"but missing from cfg.baselines."
        )

    # Inject checkpoint path into diffusion baseline
    if "diffusion" in baseline_names:
        OmegaConf.update(
            cfg, "baselines.diffusion.checkpoint_path",
            str(source_ckpt_path), force_add=True,
        )

    baselines = {
        name: _instantiate_baseline(cfg.baselines[name], device=device)
        for name in baseline_names
    }

    # --- Config values ---
    targets = list(OmegaConf.to_container(cfg.get("targets", []), resolve=True))
    r_min_values = list(cfg.get("r_min_values", [0.6]))
    if not r_min_values:
        raise ValueError("r_min_values must be non-empty.")
    base_r_min = float(r_min_values[0])
    n_samples_per_network = int(cfg.get("n_samples_per_network", 50))
    batch_size_val = int(cfg.get("batch_size_val", 8))

    transfer_cache_root = resolve_path_from_runtime_cwd(
        str(cfg.transfer_cache.get("root", "data/wra_channel_cache_transfer")),
        runtime_cwd,
    )
    auto_generate = bool(cfg.transfer_cache.get("auto_generate", True))
    require_existing = bool(cfg.transfer_cache.get("require_existing_cache", False))
    transfer_data_root = str(resolve_path_from_runtime_cwd(
        str(cfg.get("transfer_data_root", "data/wra_transfer")),
        runtime_cwd,
    ))

    plot_style_cfg = as_dict(OmegaConf.to_container(cfg.get("plot_style", {}), resolve=True))

    # --- Prepare datasets ---
    all_dataset_names: list[str] = []
    eval_plan: list[dict[str, Any]] = []

    print(f"\nTransferability evaluation: {len(targets)} target(s), "
          f"r_min_values={r_min_values}, baselines={baseline_names}")

    for target_entry in targets:
        target = target_entry if isinstance(target_entry, dict) else {"scenario": str(target_entry)}
        scenario_name = str(target.get("scenario", ""))
        if not scenario_name:
            continue

        eval_num_networks = int(target.get("eval_num_networks", 8))
        cache_num_networks_cfg = target.get("cache_num_networks", None)
        cache_num_networks = int(cache_num_networks_cfg) if cache_num_networks_cfg is not None else eval_num_networks

        scenario_cfg = load_wra_scenario_config(conf_root, scenario_name)
        target_sys_params = build_target_system_params(scenario_cfg, resolved_r_min=base_r_min)

        if source_p_max_dBm is not None:
            enforce_pmax_compatibility(
                source_p_max_dBm=source_p_max_dBm,
                target_p_max_dBm=float(target_sys_params["P_max_dBm"]),
                allow_mismatch=False,
            )

        # Channel cache: fuzzy scan then auto-generate
        seed = int(OmegaConf.select(scenario_cfg, "seed", default=42))
        n_links = int(OmegaConf.select(scenario_cfg, "dataset.n_links", default=50))
        deployment_range = float(OmegaConf.select(scenario_cfg, "dataset.deployment_range", default=1000.0))
        channel_cfg_raw = OmegaConf.to_container(
            OmegaConf.select(scenario_cfg, "channel", default=OmegaConf.create({})),
            resolve=True,
        )
        channel_cfg_dict = channel_cfg_raw if isinstance(channel_cfg_raw, dict) else {}
        channel_version = normalize_channel_version(channel_cfg_dict.get("version"), default="v2")
        channel_params = sanitize_channel_params(channel_cfg_dict)

        expected_meta = build_wra_channel_cache_metadata(
            seed=seed, num_networks=cache_num_networks, n_links=n_links,
            deployment_range=deployment_range, channel_version=channel_version,
            channel_params=channel_params, channel_config_name=f"transfer_{scenario_name}",
        )

        cache_artifact = find_compatible_channel_cache(
            scenario_name=scenario_name, cache_root=transfer_cache_root,
            expected_metadata=expected_meta, min_num_networks=eval_num_networks,
        )
        if cache_artifact is None:
            cache_artifact = build_or_load_transfer_channel_cache(
                scenario_cfg=scenario_cfg, scenario_name=scenario_name,
                cache_root=transfer_cache_root, cache_num_networks=cache_num_networks,
                auto_generate=auto_generate, require_existing_cache=require_existing,
            )

        selected_ids = select_tail_network_ids(len(cache_artifact.channels), eval_num_networks)
        scenario_short = scenario_name.replace("wra_", "")
        cache_info_entry = build_channel_cache_info_entry(
            cache_file=cache_artifact.cache_file,
            cache_metadata=cache_artifact.metadata,
        )

        print(f"\n  Target: {scenario_name} ({eval_num_networks} networks, "
              f"cache={cache_artifact.cache_key})")

        # Base directory
        base_ds_name = transfer_dataset_name(
            scenario_short, cache_artifact.cache_key,
            eval_num_networks=eval_num_networks, base_r_min=base_r_min,
        )
        base_ds_dir = os.path.join(transfer_data_root, base_ds_name)

        synthesize_transfer_raw_dir(
            channels=cache_artifact.channels,
            selected_network_ids=selected_ids,
            dataset_dir=base_ds_dir,
            system_params=target_sys_params,
            channel_params=dict(cache_artifact.channel_params),
            channel_version=cache_artifact.channel_version,
            channel_cache_info_entry=cache_info_entry,
            scenario_name=scenario_name,
        )
        all_dataset_names.append(base_ds_name)
        eval_plan.append({
            "target_scenario": scenario_name,
            "r_min": base_r_min,
            "dataset_name": base_ds_name,
            "system_params": dict(target_sys_params),
            "channel_params": dict(cache_artifact.channel_params),
            "n_links": n_links,
            "eval_num_networks": eval_num_networks,
        })

        for r_val in r_min_values[1:]:
            variant_ds_name = transfer_dataset_name(
                scenario_short, cache_artifact.cache_key,
                eval_num_networks=eval_num_networks, base_r_min=base_r_min,
                swept_r_min=r_val,
            )
            variant_ds_dir = os.path.join(transfer_data_root, variant_ds_name)
            create_transfer_r_min_variant(
                base_raw_dir=os.path.join(base_ds_dir, "raw"),
                variant_dataset_dir=variant_ds_dir,
                r_min=r_val,
            )
            all_dataset_names.append(variant_ds_name)
            variant_sys = dict(target_sys_params)
            variant_sys["r_min"] = float(r_val)
            eval_plan.append({
                "target_scenario": scenario_name,
                "r_min": r_val,
                "dataset_name": variant_ds_name,
                "system_params": variant_sys,
                "channel_params": dict(cache_artifact.channel_params),
                "n_links": n_links,
                "eval_num_networks": eval_num_networks,
            })

    if not eval_plan:
        raise ValueError("No transfer targets prepared.")

    # --- Process all datasets ---
    print(f"\n  Processing {len(all_dataset_names)} dataset(s)...")
    _all_ds = WRADataset(
        root=transfer_data_root,
        dataset_names=all_dataset_names,
        split="test",
        train_fraction=0.0, val_fraction=0.0, test_fraction=1.0,
        top_k=int(graph_cfg["top_k"]),
        min_degree=int(graph_cfg["min_degree"]),
        edge_normalize_method=str(graph_cfg["edge_normalize_method"]),
        spectral_normalize_edge_weights=bool(graph_cfg["spectral_normalize_edge_weights"]),
        model_cond_channels=model_cond_channels,
    )
    del _all_ds

    # --- Evaluate ---
    task = instantiate(cfg.task, reference_policy="none")
    if hasattr(task, "set_plot_style"):
        task.set_plot_style(plot_style_cfg)

    sweep_results: list[dict[str, Any]] = []
    # WRA evaluation records keyed by dataset_name: {ds_name: {bl_name: [records]}}
    _wra_records: dict[str, dict[str, list]] = {}

    for plan_entry in eval_plan:
        ds_name = plan_entry["dataset_name"]
        r_val = plan_entry["r_min"]
        target_name = plan_entry["target_scenario"]
        print(f"\n  Evaluating {target_name} (r_min={r_val:.2f})...")

        per_ds = WRADataset(
            root=transfer_data_root,
            dataset_names=[ds_name],
            split="test",
            train_fraction=0.0, val_fraction=0.0, test_fraction=1.0,
            top_k=int(graph_cfg["top_k"]),
            min_degree=int(graph_cfg["min_degree"]),
            edge_normalize_method=str(graph_cfg["edge_normalize_method"]),
            spectral_normalize_edge_weights=bool(graph_cfg["spectral_normalize_edge_weights"]),
            model_cond_channels=model_cond_channels,
        )
        if len(per_ds) == 0:
            print(f"    No samples, skipping.")
            continue

        networks_per_batch = max(1, math.ceil(batch_size_val / n_samples_per_network))
        sampler = NetworkGroupedBatchSampler(
            dataset=per_ds,
            batch_size=networks_per_batch * n_samples_per_network,
            samples_per_network=n_samples_per_network,
            shuffle=False, seed=42,
        )
        loader = DataLoader(
            per_ds, batch_sampler=sampler,
            num_workers=0, pin_memory=False, persistent_workers=False,
            follow_batch=["y"], exclude_keys=["info"],
        )

        dataset_info = build_sweep_dataset_info(
            dataset_root=transfer_data_root,
            dataset_name=ds_name,
            r_min=r_val,
            system_params=plan_entry["system_params"],
            channel_params=plan_entry["channel_params"],
        )
        task.set_dataset_info(dataset_info)

        for bl_name, baseline in baselines.items():
            viz_dir = os.path.join(str(output_dir), "eval_viz", ds_name, bl_name)
            os.makedirs(viz_dir, exist_ok=True)

            metrics = baseline.evaluate(
                loader, task, max_batches=None,
                viz_save_dir=viz_dir, eval_split_name="test",
            )
            entry: dict[str, Any] = {
                "target_scenario": target_name,
                "baseline": bl_name,
                "r_min": r_val,
                "n_links": plan_entry["n_links"],
                "eval_num_networks": plan_entry["eval_num_networks"],
            }
            entry.update(metrics)
            sweep_results.append(entry)

            mean_rate = metrics.get("mean_rate_generated", float("nan"))
            viol_pct = metrics.get("rate_violation_percentage_generated", float("nan"))
            print(f"    [{bl_name}] mean_rate={mean_rate:.4f}, viol%={viol_pct:.2f}")

            # Collect WRA evaluation records for percentile evolution plots
            recs = getattr(baseline, "last_wra_evaluation_records", None)
            if recs:
                _wra_records.setdefault(ds_name, {})[bl_name] = recs

    # --- Outputs ---
    save_transferability_summary(sweep_results=sweep_results, output_dir=str(output_dir))

    plot_transferability_results(
        sweep_results=sweep_results, output_dir=str(output_dir),
        r_min_values=r_min_values, plot_style=plot_style_cfg,
    )

    # --- Per-(target, r_min) percentile evolution plots ---
    if _wra_records and getattr(task, "is_power_allocation_task", False):
        from graph_signal_diffusion.baselines.wireless_resource_allocation.viz import (
            plot_wra_percentile_evolution_comparison,
        )
        comp_dir = os.path.join(str(output_dir), "eval_viz", "comparison")
        os.makedirs(comp_dir, exist_ok=True)

        _wra_style_cfg = _as_dict(
            _as_dict(plot_style_cfg.get("plots", {})).get(
                "percentile_evolution_comparison", {}
            )
        )
        _baseline_colors = _as_dict(
            _as_dict(plot_style_cfg.get("baseline", {})).get("colors", {})
        )

        for rec_key, records_by_bl in _wra_records.items():
            if not records_by_bl:
                continue
            try:
                plot_wra_percentile_evolution_comparison(
                    records_by_bl,
                    save_path=os.path.join(
                        comp_dir,
                        f"wra_percentile_evolution_transfer_{rec_key}.pdf",
                    ),
                    baseline_order=list(baselines.keys()),
                    baseline_colors=_baseline_colors or None,
                    plot_style=_wra_style_cfg,
                )
            except Exception as exc:
                print(f"    [WARNING] Percentile plot for {rec_key} failed: {exc}")

    # Manifest
    manifest = {
        "source_checkpoint": str(source_ckpt_path),
        "source_scenario": str(cfg.source.get("source_scenario_name", "")),
        "model_cond_channels": model_cond_channels,
        "graph_build_cfg": graph_cfg,
        "targets": [str(t.get("scenario", t) if isinstance(t, dict) else t) for t in targets],
        "r_min_values": r_min_values,
        "baselines": baseline_names,
    }
    with open(output_dir / "transferability_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
