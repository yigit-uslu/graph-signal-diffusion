from __future__ import annotations
import contextlib
import os
import warnings
from typing import Any
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig
import torch

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore[assignment]

from graph_signal_diffusion.datasets import discover_datasets, get_dataset_builder
from graph_signal_diffusion.tasks import discover_tasks
from graph_signal_diffusion.baselines import discover_baselines
from graph_signal_diffusion.evaluation import UnifiedEvaluator
from graph_signal_diffusion.cli._baseline_shared import (
    setup_task,
    log_results_to_wandb,
)


def _to_plain(value: Any) -> Any:
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(
            value, resolve=True, throw_on_missing=False,
        )
    return value


def _as_dict(value: Any) -> dict[str, Any]:
    value = _to_plain(value)
    return value if isinstance(value, dict) else {}


def _normalize_color_map(value: Any) -> dict[str, str]:
    raw = _as_dict(value)
    out: dict[str, str] = {}
    for k, v in raw.items():
        key = str(k).strip()
        if not key:
            continue
        out[key] = str(v)
    return out


def _apply_global_plot_style(plot_style_cfg: dict[str, Any]) -> None:
    """Thin wrapper around the shared style helper.

    Kept for backwards compatibility with existing call sites; the actual
    logic lives in :mod:`graph_signal_diffusion.evaluation.plot_style_apply`
    so the replot CLI can apply the same style.
    """
    from graph_signal_diffusion.evaluation.plot_style_apply import (
        apply_global_plot_style,
    )
    apply_global_plot_style(_as_dict(plot_style_cfg.get("global", {})))


def _apply_baseline_display_aliases(plot_style_cfg: dict[str, Any]) -> None:
    baseline_cfg = _as_dict(plot_style_cfg.get("baseline", {}))
    aliases = _as_dict(baseline_cfg.get("display_aliases", {}))
    from graph_signal_diffusion.baselines.plot_utils import (
        set_baseline_display_aliases,
    )
    set_baseline_display_aliases(aliases or None)


def _attach_paper_figure_cfg(task, cfg: DictConfig, output_dir: str) -> None:
    """Expose paper-figure config + a stable output dir to *task*.

    Task-side paper-figure plotters (e.g. SP500 Fig 1) read ``paper_figures_cfg``
    and ``_paper_out_dir``; a no-op for tasks that ignore them.  The dedicated
    ``eval_viz/paper/`` dir sidesteps the diffusion eval_viz nesting so all paper
    figures land in one predictable place regardless of baseline.

    Defined at module scope (NOT inside ``main``) so ``os`` resolves to the
    module-level import — ``main`` has function-local ``import os`` statements
    that would otherwise shadow it and raise ``UnboundLocalError`` here.
    """
    if task is None:
        return
    setattr(task, "paper_figures_cfg", _as_dict(cfg.get("paper_figures", {})))
    setattr(task, "_paper_out_dir", os.path.join(output_dir, "eval_viz", "paper"))


def _init_wandb(cfg: DictConfig, hydra_cfg, baseline_names: list[str]):
    """Build the wandb context manager for baseline comparison.

    Returns a context manager that yields the wandb Run when enabled,
    or ``contextlib.nullcontext()`` (yields None) when disabled.

    Auto-derives ``group``, ``tags``, and ``job_type`` from the Hydra
    config.  Explicit ``wandb.group`` / ``wandb.tags`` values in the
    config take precedence over auto-derived defaults.
    """
    if wandb is None:
        return contextlib.nullcontext()
    if not cfg.get("wandb", {}).get("enabled", False):
        return contextlib.nullcontext()

    from graph_signal_diffusion.utils.wandb_context import (
        build_wandb_context,
        merge_wandb_context,
    )

    context = build_wandb_context(
        cfg, stage="compare", baseline_names=baseline_names,
    )
    merged = merge_wandb_context(cfg, context)

    wandb_config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=False
    )

    # Auto-generate a descriptive run name
    dataset_name = cfg.dataset.get("name", "unknown")
    baselines_str = "+".join(sorted(set(baseline_names))) if baseline_names else "none"
    run_name = (
        cfg.wandb.get("name", None)
        or f"baseline-compare-{baselines_str}-{dataset_name}"
    )

    return wandb.init(
        project=cfg.wandb.get("project", "graph-signal-diffusion"),
        entity=cfg.wandb.get("entity", None),
        group=merged["group"],
        job_type=merged["job_type"],
        tags=merged["tags"],
        notes=cfg.wandb.get("notes", None),
        name=run_name,
        mode=cfg.wandb.get("mode", "online"),
        config=wandb_config,
        dir=hydra_cfg.runtime.output_dir,
    )


def _build_short_name_to_r_min(
    wra_records: dict[str, dict[str, list]],
) -> dict[tuple[str, str], float | None]:
    """Build ``{(split, short_name): r_min}`` mapping from WRA evaluation records.

    Keys are ``(split, short_name)`` so that r_min is resolved per-split,
    avoiding cross-split conflation when the same short_name appears in
    different splits with different r_min values.

    Emits a warning if a single ``(split, short_name)`` maps to multiple
    distinct r_min values across baselines' records within that split.
    """
    mapping: dict[tuple[str, str], set[float]] = {}
    for split, split_recs in wra_records.items():
        for _bl_recs in split_recs.values():
            for rec in _bl_recs:
                ds = rec.get("dataset_name")
                r = rec.get("r_min")
                if ds is None or r is None:
                    continue
                short = str(ds).rsplit("/", 1)[-1] if "/" in str(ds) else str(ds)
                mapping.setdefault((split, short), set()).add(float(r))

    result: dict[tuple[str, str], float | None] = {}
    for key, vals in mapping.items():
        if len(vals) > 1:
            warnings.warn(
                f"Subdataset '{key[1]}' (split='{key[0]}') has inconsistent "
                f"r_min values {sorted(vals)}; using min={min(vals):.6f}.",
                RuntimeWarning,
                stacklevel=2,
            )
        result[key] = min(vals) if vals else None
    return result


def _save_per_subdataset_comparison(
    results_df,
    output_dir: str,
    wra_records: dict[str, dict[str, list]],
) -> None:
    """Generate per-subdataset comparison markdown and CSV from results_df.

    Skips silently when no ``subdataset/`` columns are present (single-dataset
    backwards compatibility).
    """
    import os
    import pandas as pd

    # Identify subdataset columns: {split}/subdataset/{short_name}/{metric}
    sd_cols = [c for c in results_df.columns if "/subdataset/" in c]
    if not sd_cols:
        return

    short_to_r_min = _build_short_name_to_r_min(wra_records)

    # Focused metrics to include in the table
    _DISPLAY_METRICS = [
        ("mean_rate_generated", "Mean Rate"),
        ("rate_1pct_generated", "1p Rate"),
        ("rate_5pct_generated", "5p Rate"),
        ("rate_violation_percentage_generated", "Viol%"),
        ("rate_mean_violation_gap_pct_generated", "Mean Viol Gap%"),
        ("rate_1pct_gap_pct", "1p Gap%"),
        ("rate_5pct_gap_pct", "5p Gap%"),
    ]
    metric_keys = [m[0] for m in _DISPLAY_METRICS]
    metric_labels = [m[1] for m in _DISPLAY_METRICS]

    # Parse columns into structured data
    # {(split, short_name): {baseline: {metric: value}}}
    structured: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
    for col in sd_cols:
        parts = col.split("/subdataset/", 1)
        if len(parts) != 2:
            continue
        split = parts[0]
        remainder = parts[1]  # short_name/metric
        slash_idx = remainder.find("/")
        if slash_idx < 0:
            continue
        short_name = remainder[:slash_idx]
        metric = remainder[slash_idx + 1:]
        if metric not in metric_keys:
            continue
        key = (split, short_name)
        for baseline in results_df.index:
            val = results_df.loc[baseline, col]
            if pd.isna(val):
                continue
            structured.setdefault(key, {}).setdefault(
                baseline, {}
            )[metric] = float(val)

    if not structured:
        return

    # Build markdown
    lines: list[str] = ["# Per-subdataset Baseline Comparison\n"]

    # Long-form CSV rows
    csv_rows: list[dict] = []

    for (split, short_name) in sorted(structured.keys()):
        r_min = short_to_r_min.get((split, short_name))
        r_min_str = f" (r_min={r_min:.2f})" if r_min is not None else ""
        lines.append(f"## {split} / {short_name}{r_min_str}\n")

        # Header
        header = "| Baseline | " + " | ".join(metric_labels) + " |"
        sep = "|" + "|".join(["---"] * (len(metric_labels) + 1)) + "|"
        lines.append(header)
        lines.append(sep)

        baselines_data = structured[(split, short_name)]
        for bl_name in sorted(baselines_data.keys()):
            bl_metrics = baselines_data[bl_name]
            cells = []
            for mk in metric_keys:
                v = bl_metrics.get(mk)
                cells.append(f"{v:.3f}" if v is not None else "—")
                if v is not None:
                    csv_rows.append({
                        "split": split,
                        "subdataset": short_name,
                        "baseline": bl_name,
                        "metric": mk,
                        "value": v,
                    })
            lines.append(f"| {bl_name} | " + " | ".join(cells) + " |")

        lines.append("")

    md_path = os.path.join(output_dir, "per_subdataset_comparison.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    csv_path = os.path.join(output_dir, "per_subdataset_comparison.csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    print(f"  Per-subdataset comparison saved to {md_path}")
    print(f"  Per-subdataset comparison CSV saved to {csv_path}")


def _save_wra_receiver_rates(
    wra_records: dict[str, dict[str, list]],
    output_dir: str,
    *,
    prefix: str = "receiver_rates",
    scenario_map: dict[str, str] | None = None,
) -> None:
    """Save per-receiver ergodic rates and per-slot rate arrays from WRA records.

    Produces two files:
      - ``{prefix}_ergodic.csv`` — one row per (baseline, scenario, network,
        receiver) with the ergodic (final) generated and real rates plus r_min.
      - ``{prefix}_per_slot.npz`` — per-slot rate arrays keyed by
        ``{baseline}/{ds_key}/{eval_batch_idx}`` with ``gen_rates_per_slot``
        and optionally ``real_rates_per_slot``.

    Parameters
    ----------
    wra_records
        ``{ds_key: {baseline_name: [record, ...]}}``
    output_dir
        Directory to write files into.
    prefix
        Filename prefix for the output files.
    scenario_map
        Optional ``{ds_key: scenario_name}`` for adding a scenario column.
    """
    import os
    import numpy as np
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)

    # --- Ergodic rates CSV ---
    rows: list[dict[str, Any]] = []
    # --- Per-slot arrays for npz ---
    npz_data: dict[str, Any] = {}

    for ds_key, records_by_bl in wra_records.items():
        scenario = (scenario_map or {}).get(ds_key, ds_key)
        for bl_name, recs in records_by_bl.items():
            for rec in recs:
                network_id = rec.get("network_id", -1)
                batch_idx = rec.get("eval_batch_idx", 0)
                r_min = rec.get("r_min", float("nan"))
                dataset_name = rec.get("dataset_name", ds_key)

                gen_final = rec.get("gen_rates_final")  # [n]
                real_final = rec.get("real_rates_final")  # [n] or None
                if gen_final is not None:
                    gen_final = np.asarray(gen_final).ravel()
                    real_final_arr = (
                        np.asarray(real_final).ravel()
                        if real_final is not None
                        else np.full_like(gen_final, float("nan"))
                    )
                    for rx_idx in range(len(gen_final)):
                        rows.append({
                            "scenario": scenario,
                            "dataset_name": dataset_name,
                            "baseline": bl_name,
                            "network_id": int(network_id),
                            "eval_batch_idx": int(batch_idx),
                            "receiver_idx": int(rx_idx),
                            "r_min": float(r_min),
                            "gen_rate": float(gen_final[rx_idx]),
                            "real_rate": float(real_final_arr[rx_idx]),
                        })

                # Per-slot arrays
                npz_key = f"{bl_name}/{ds_key}/batch{batch_idx}_net{network_id}"
                gen_ps = rec.get("gen_rates_per_slot")
                if gen_ps is not None:
                    npz_data[f"{npz_key}/gen_rates_per_slot"] = np.asarray(gen_ps)
                real_ps = rec.get("real_rates_per_slot")
                if real_ps is not None:
                    npz_data[f"{npz_key}/real_rates_per_slot"] = np.asarray(real_ps)

    if rows:
        csv_path = os.path.join(output_dir, f"{prefix}_ergodic.csv")
        pd.DataFrame(rows).to_csv(csv_path, index=False)
        print(f"  Per-receiver ergodic rates saved to {csv_path} ({len(rows)} rows)")

    if npz_data:
        npz_path = os.path.join(output_dir, f"{prefix}_per_slot.npz")
        np.savez_compressed(npz_path, **npz_data)
        print(f"  Per-slot rate arrays saved to {npz_path} ({len(npz_data)} arrays)")


def _density_label_from_dataset(dataset_path: str) -> str:
    """Extract a short density label from a dataset path like
    'medium-large_outdoor_low_density/wrpc_v1_..._h...' -> 'low'.
    Falls back to the full path if pattern doesn't match."""
    import re
    m = re.search(r"outdoor_([a-z_-]+?)_density", dataset_path)
    return m.group(1) if m else dataset_path.split("/")[0]


def _save_diffusion_overlay_npz(
    wra_records: dict[str, dict[str, list]],
    output_dir: str,
    *,
    network_ids: list[int] | None = None,
    baseline_name: str = "diffusion",
    scenario_map: dict[str, str] | None = None,
) -> None:
    """Save diffusion-generated power samples in viz_trace overlay format.

    Produces one ``.npz`` per (scenario, network_id) under
    ``{output_dir}/overlay/``.  Each file contains:
      - ``network_ids``: ``[network_id]``
      - ``network_{id}_powers``: ``[num_samples, n_links]`` (generated)
      - ``network_{id}_real_powers``: ``[num_samples, n_links]`` (expert, if available)

    Parameters
    ----------
    wra_records
        ``{ds_key: {baseline_name: [record, ...]}}``
    output_dir
        Root directory; files written to ``{output_dir}/overlay/``.
    network_ids
        If given, only save these network IDs. ``None`` saves all.
    baseline_name
        Which baseline's records to use (default: ``"diffusion"``).
    scenario_map
        Optional ``{ds_key: scenario_name}`` for directory naming.
    """
    import os
    import numpy as np

    overlay_dir = os.path.join(output_dir, "overlay")
    os.makedirs(overlay_dir, exist_ok=True)
    net_id_set = set(network_ids) if network_ids is not None else None
    count = 0

    for ds_key, records_by_bl in wra_records.items():
        recs = records_by_bl.get(baseline_name, [])
        if not recs:
            continue
        scenario = (scenario_map or {}).get(ds_key, ds_key)
        # Sanitize scenario for filename
        scenario_safe = scenario.replace("/", "_").replace(" ", "_")

        # Group records by network_id
        by_net_gen: dict[int, list[np.ndarray]] = {}
        by_net_real: dict[int, list[np.ndarray]] = {}
        for rec in recs:
            nid = rec.get("network_id")
            if nid is None:
                continue
            nid = int(nid)
            if net_id_set is not None and nid not in net_id_set:
                continue
            gen = rec.get("gen_powers_sampled")  # [num_time_shares, m]
            if gen is not None:
                by_net_gen.setdefault(nid, []).append(np.asarray(gen))
            real = rec.get("real_powers_sampled")  # [num_time_shares, m]
            if real is not None:
                by_net_real.setdefault(nid, []).append(np.asarray(real))

        for nid, power_arrays in by_net_gen.items():
            # Stack all eval batches: each is [T, m], concatenate along axis 0
            all_gen = np.concatenate(power_arrays, axis=0)  # [total_samples, m]
            npz_data = {
                "network_ids": np.array([nid]),
                f"network_{nid}_powers": all_gen,
            }
            # Include real (expert) powers when available
            if nid in by_net_real:
                all_real = np.concatenate(by_net_real[nid], axis=0)
                npz_data[f"network_{nid}_real_powers"] = all_real
            npz_path = os.path.join(
                overlay_dir,
                f"generated_powers_{scenario_safe}_net{nid}.npz",
            )
            np.savez_compressed(npz_path, **npz_data)
            count += 1

    if count > 0:
        print(f"  Diffusion overlay files saved to {overlay_dir}/ ({count} files)")


def _per_network_stats(
    records: list[dict], r_min: float,
) -> dict[str, float]:
    """Compute std of per-network metrics (for error bars).

    Groups records by ``network_id``, computes each metric per network,
    then returns the standard deviation across networks.
    """
    import numpy as np
    from collections import defaultdict as _ddict
    by_net: dict[Any, list[np.ndarray]] = _ddict(list)
    for rec in records:
        gf = rec.get("gen_rates_final")
        if gf is not None:
            by_net[rec.get("network_id", id(rec))].append(
                np.asarray(gf).ravel()
            )
    if len(by_net) < 2:
        return {}
    _pcts = {"0.1": 0.1, "0.5": 0.5, "1": 1, "5": 5, "10": 10}
    net_means: list[float] = []
    net_pcts: dict[str, list[float]] = {k: [] for k in _pcts}
    net_viol: list[float] = []
    net_viol_gap: list[float] = []
    for _nid, arrs in by_net.items():
        pooled = np.concatenate(arrs)
        net_means.append(float(np.mean(pooled)))
        for k, p in _pcts.items():
            net_pcts[k].append(float(np.percentile(pooled, p)))
        v = pooled < r_min
        net_viol.append(float(100.0 * np.sum(v) / len(pooled)) if len(pooled) else 0.0)
        vg = np.where(v, r_min - pooled, 0.0)
        net_viol_gap.append(
            float(100.0 * np.mean(vg / np.maximum(r_min, 1e-12))) if len(pooled) else 0.0
        )
    out: dict[str, float] = {
        "mean_rate_generated_std": float(np.std(net_means)),
    }
    for k in _pcts:
        out[f"rate_{k}pct_generated_std"] = float(np.std(net_pcts[k]))
    out["rate_violation_percentage_generated_std"] = float(np.std(net_viol))
    out["rate_mean_violation_gap_pct_generated_std"] = float(np.std(net_viol_gap))
    return out


def _run_r_min_sweep_single(
    *,
    cfg: DictConfig,
    reference_dataset: str,
    r_min_values: list[float],
    baseline: Any,
    baseline_name: str,
    task: Any,
    n_samples_per_network: int,
    batch_size_val: int,
    cfg_mcc: int,
    output_dir: str,
    density_label: str,
) -> tuple[list[dict[str, Any]], list[list[dict]]]:
    """Run r_min sweep for a single reference dataset.

    Returns (sweep_results, per_rmin_wra_records) where per_rmin_wra_records[i]
    is the list of WRA evaluation records for r_min_values[i].
    """
    import math
    import os
    import json

    from graph_signal_diffusion.cli._wra_r_min_sweep import (
        check_npz_no_r_min_per_receiver,
        create_sweep_dataset_dirs,
        build_sweep_dataset_info,
        sweep_dataset_name,
    )
    from graph_signal_diffusion.datasets.wra.dataset import WRADataset
    from graph_signal_diffusion.datasets.wra.sampler import NetworkGroupedBatchSampler
    from torch_geometric.loader import DataLoader

    reference_raw_dir = os.path.join(str(cfg.dataset.root), reference_dataset, "raw")
    check_npz_no_r_min_per_receiver(reference_raw_dir)

    sweep_root = os.path.join(output_dir, "_r_min_sweep_data", density_label)
    dataset_names = create_sweep_dataset_dirs(
        reference_raw_dir=reference_raw_dir,
        output_root=sweep_root,
        r_min_values=r_min_values,
    )

    print(f"    Created {len(dataset_names)} sweep dataset dirs under {sweep_root}")

    # Process all sweep datasets in one WRADataset instance (H dedup)
    _all_sweep_ds = WRADataset(
        root=sweep_root,
        dataset_names=dataset_names,
        split="test",
        top_k=int(cfg.dataset.get("top_k", 10)),
        min_degree=int(cfg.dataset.get("min_degree", 1)),
        edge_normalize_method=str(cfg.dataset.get("edge_normalize_method", "log1p")),
        spectral_normalize_edge_weights=bool(
            cfg.dataset.get("spectral_normalize_edge_weights", False)
        ),
        model_cond_channels=cfg_mcc,
    )
    del _all_sweep_ds  # Only needed to trigger process()

    # Extract system_params and channel_params from reference
    ref_info_path = os.path.join(reference_raw_dir, "network_info.json")
    with open(ref_info_path) as f:
        ref_network_info = json.load(f)
    ref_system_params = ref_network_info.get("system_params", {})
    ref_channel_params = ref_network_info.get("channel_params", {})

    sweep_results: list[dict[str, Any]] = []
    per_rmin_wra_records: list[list[dict]] = []

    for r_min in r_min_values:
        ds_name = sweep_dataset_name(r_min)
        print(f"\n    [{density_label}] Evaluating r_min={r_min:.2f} ({ds_name})...")

        per_ds = WRADataset(
            root=sweep_root,
            dataset_names=[ds_name],
            split="test",
            top_k=int(cfg.dataset.get("top_k", 10)),
            min_degree=int(cfg.dataset.get("min_degree", 1)),
            edge_normalize_method=str(cfg.dataset.get("edge_normalize_method", "log1p")),
            spectral_normalize_edge_weights=bool(
                cfg.dataset.get("spectral_normalize_edge_weights", False)
            ),
            model_cond_channels=cfg_mcc,
        )

        if len(per_ds) == 0:
            print(f"      No samples for {ds_name}, skipping.")
            per_rmin_wra_records.append([])
            continue

        networks_per_batch = max(1, math.ceil(batch_size_val / n_samples_per_network))
        eval_batch_size = networks_per_batch * n_samples_per_network
        sampler = NetworkGroupedBatchSampler(
            dataset=per_ds,
            batch_size=eval_batch_size,
            samples_per_network=n_samples_per_network,
            shuffle=False,
            seed=42,
        )
        loader = DataLoader(
            per_ds,
            batch_sampler=sampler,
            num_workers=int(cfg.dataset.get("num_workers", 0)),
            pin_memory=bool(cfg.dataset.get("pin_memory", False)),
            persistent_workers=False,
            follow_batch=["y"],
            exclude_keys=["info"],
        )

        dataset_info = build_sweep_dataset_info(
            dataset_root=sweep_root,
            dataset_name=ds_name,
            r_min=r_min,
            system_params=ref_system_params,
            channel_params=ref_channel_params,
        )

        original_dataset_info = getattr(task, "dataset_info", None)
        original_reference_policy = getattr(task, "reference_policy", "expert")
        task.set_dataset_info(dataset_info)
        task.reference_policy = "none"

        try:
            viz_dir = os.path.join(
                output_dir, "eval_viz", "r_min_sweep", density_label, ds_name
            )
            os.makedirs(viz_dir, exist_ok=True)

            metrics = baseline.evaluate(
                loader, task,
                max_batches=None,
                viz_save_dir=viz_dir,
                eval_split_name="test",
            )

            entry = {"r_min": r_min, "density": density_label}
            entry.update(metrics)
            sweep_results.append(entry)

            # Capture WRA evaluation records for percentile plots
            recs = getattr(baseline, "last_wra_evaluation_records", None)
            per_rmin_wra_records.append(list(recs) if recs else [])

            mean_rate = metrics.get("mean_rate_generated", float("nan"))
            viol_pct = metrics.get("rate_violation_percentage_generated", float("nan"))
            print(f"      mean_rate={mean_rate:.4f}, viol%={viol_pct:.2f}")

        finally:
            if original_dataset_info is not None:
                task.set_dataset_info(original_dataset_info)
            else:
                task.dataset_info = None
            task.reference_policy = original_reference_policy

    return sweep_results, per_rmin_wra_records


def _run_r_min_sweep(
    *,
    cfg: DictConfig,
    sweep_cfg: dict[str, Any],
    baselines: dict[str, Any],
    task: Any,
    device: torch.device,
    output_dir: str,
    plot_style_cfg: dict[str, Any],
) -> None:
    """Execute the optional r_min sweep phase after standard comparison.

    Supports multiple reference datasets (one per density).  When
    ``reference_datasets`` is a list, runs a separate sweep per density and
    produces both per-density and aggregated cross-density outputs.  The
    legacy singular ``reference_dataset`` key is still supported for backward
    compatibility.
    """
    import os
    from omegaconf import OmegaConf

    from graph_signal_diffusion.cli._wra_r_min_sweep import (
        check_model_cond_channels,
        save_sweep_summary,
        plot_r_min_sweep,
    )

    print(f"\n{'='*60}")
    print("r_min SWEEP")
    print(f"{'='*60}")

    r_min_values = list(sweep_cfg["r_min_values"])
    native_r_min_values = list(sweep_cfg.get("native_r_min_values", []))
    baseline_name = str(sweep_cfg.get("baseline", "diffusion"))
    n_samples_per_network = int(sweep_cfg.get("n_samples_per_network", 50))
    batch_size_val = int(sweep_cfg.get("batch_size_val", 1000))

    # Resolve reference dataset(s) — support both singular and list forms
    ref_datasets_raw = sweep_cfg.get("reference_datasets", None)
    if ref_datasets_raw is not None:
        reference_datasets = list(ref_datasets_raw)
    else:
        reference_datasets = [str(sweep_cfg["reference_dataset"])]

    print(f"  r_min values: {r_min_values}")
    print(f"  native: {native_r_min_values}")
    print(f"  reference datasets ({len(reference_datasets)}):")
    for rd in reference_datasets:
        print(f"    - {rd}")
    print(f"  baseline: {baseline_name}")

    if baseline_name not in baselines:
        print(f"  [WARNING] Baseline '{baseline_name}' not in evaluated baselines. "
              "Skipping r_min sweep.")
        return
    baseline = baselines[baseline_name]

    # --- Guards ---
    cfg_mcc = int(cfg.dataset.get("model_cond_channels", 2))
    diffusion_ckpt = None
    bl_cfg = _as_dict(OmegaConf.to_container(cfg.baselines.get(baseline_name, {}), resolve=True))
    if bl_cfg.get("checkpoint_path"):
        diffusion_ckpt = str(bl_cfg["checkpoint_path"])
    check_model_cond_channels(cfg_mcc, checkpoint_path=diffusion_ckpt)

    # --- Run sweep for each reference dataset ---
    all_sweep_results: list[dict[str, Any]] = []
    # {r_min_value: [wra_records_across_densities]}
    aggregated_wra_records: dict[float, list[dict]] = {rv: [] for rv in r_min_values}

    for ref_dataset in reference_datasets:
        density_label = _density_label_from_dataset(ref_dataset)
        print(f"\n  --- Density: {density_label} ({ref_dataset}) ---")

        density_results, per_rmin_records = _run_r_min_sweep_single(
            cfg=cfg,
            reference_dataset=ref_dataset,
            r_min_values=r_min_values,
            baseline=baseline,
            baseline_name=baseline_name,
            task=task,
            n_samples_per_network=n_samples_per_network,
            batch_size_val=batch_size_val,
            cfg_mcc=cfg_mcc,
            output_dir=output_dir,
            density_label=density_label,
        )

        all_sweep_results.extend(density_results)

        # Accumulate WRA records per r_min across densities
        for i, r_min in enumerate(r_min_values):
            if i < len(per_rmin_records) and per_rmin_records[i]:
                aggregated_wra_records[r_min].extend(per_rmin_records[i])

        # Per-density summary and plots
        if density_results:
            density_out = os.path.join(output_dir, f"r_min_sweep_{density_label}")
            os.makedirs(density_out, exist_ok=True)
            save_sweep_summary(
                sweep_results=density_results,
                native_r_min_values=native_r_min_values,
                output_dir=density_out,
            )
            plot_r_min_sweep(
                sweep_results=density_results,
                native_r_min_values=native_r_min_values,
                output_dir=density_out,
                plot_style=plot_style_cfg,
            )
            print(f"    Per-density results saved to {density_out}/")

    if not all_sweep_results:
        print("  No sweep results collected.")
        return

    # --- Aggregated summary (all densities) ---
    # Recompute QoS metrics from pooled per-receiver rates across densities.
    import numpy as np

    def _metrics_from_pooled_records(
        records: list[dict], r_min: float,
    ) -> dict[str, Any]:
        """Compute QoS metrics by pooling all receivers across records."""
        all_gen_rates: list[np.ndarray] = []
        for rec in records:
            gen_final = rec.get("gen_rates_final")
            if gen_final is not None:
                all_gen_rates.append(np.asarray(gen_final).ravel())
        if not all_gen_rates:
            return {}
        pooled = np.concatenate(all_gen_rates)
        n = len(pooled)
        violations = pooled < r_min
        violation_gaps = np.where(violations, r_min - pooled, 0.0)
        out: dict[str, Any] = {
            "mean_rate_generated": float(np.mean(pooled)),
            "sum_rate_generated": float(np.sum(pooled)),
            "rate_0.1pct_generated": float(np.percentile(pooled, 0.1)),
            "rate_0.5pct_generated": float(np.percentile(pooled, 0.5)),
            "rate_1pct_generated": float(np.percentile(pooled, 1)),
            "rate_5pct_generated": float(np.percentile(pooled, 5)),
            "rate_10pct_generated": float(np.percentile(pooled, 10)),
            "rate_violation_percentage_generated": float(
                100.0 * np.sum(violations) / n
            ) if n > 0 else 0.0,
            "rate_mean_violation_gap_pct_generated": float(
                100.0 * np.mean(violation_gaps / np.maximum(r_min, 1e-12))
            ) if n > 0 else 0.0,
            "feasibility_rate": float(
                1.0 - np.sum(violations) / n
            ) if n > 0 else 0.0,
        }
        out.update(_per_network_stats(records, r_min))
        return out

    aggregated_results: list[dict[str, Any]] = []
    for r_min in r_min_values:
        recs = aggregated_wra_records.get(r_min, [])
        if not recs:
            continue
        agg: dict[str, Any] = {"r_min": r_min}
        agg.update(_metrics_from_pooled_records(recs, r_min))
        aggregated_results.append(agg)

    save_sweep_summary(
        sweep_results=aggregated_results,
        native_r_min_values=native_r_min_values,
        output_dir=output_dir,
    )
    plot_r_min_sweep(
        sweep_results=aggregated_results,
        native_r_min_values=native_r_min_values,
        output_dir=output_dir,
        plot_style=plot_style_cfg,
    )

    # --- Per-r_min percentile evolution plots (aggregated across densities) ---
    if any(aggregated_wra_records.values()):
        from graph_signal_diffusion.baselines.wireless_resource_allocation.viz import (
            plot_wra_percentile_evolution_comparison,
        )
        _comp_dir = os.path.join(output_dir, "eval_viz", "r_min_sweep", "comparison")
        os.makedirs(_comp_dir, exist_ok=True)

        _wra_style_cfg = _as_dict(
            _as_dict(plot_style_cfg.get("plots", {})).get(
                "percentile_evolution_comparison", {}
            )
        )
        _baseline_colors = _normalize_color_map(
            _as_dict(plot_style_cfg.get("baseline", {})).get("colors", {})
        )

        for r_min in r_min_values:
            recs = aggregated_wra_records.get(r_min, [])
            if not recs:
                continue
            records_by_baseline = {baseline_name: recs}
            try:
                plot_wra_percentile_evolution_comparison(
                    records_by_baseline,
                    save_path=os.path.join(
                        _comp_dir,
                        f"wra_percentile_evolution_sweep_rmin_{r_min:.2f}.pdf",
                    ),
                    baseline_order=[baseline_name],
                    baseline_colors=_baseline_colors or None,
                    r_min=r_min,
                    plot_style=_wra_style_cfg,
                )
            except Exception as exc:
                print(f"    [WARNING] Percentile plot for r_min={r_min:.2f} failed: {exc}")

        print(f"  Per-r_min percentile evolution plots saved to {_comp_dir}/")

    # --- Save per-receiver rates for replotting ---
    if any(aggregated_wra_records.values()):
        _rmin_wra_for_save: dict[str, dict[str, list]] = {}
        for r_min, recs in aggregated_wra_records.items():
            if recs:
                ds_key = f"r_min_{r_min:.2f}"
                _rmin_wra_for_save[ds_key] = {baseline_name: recs}
        if _rmin_wra_for_save:
            _save_wra_receiver_rates(
                _rmin_wra_for_save,
                output_dir,
                prefix="r_min_sweep_rates",
            )

    total_evals = len(all_sweep_results)
    n_densities = len(reference_datasets)
    print(f"\n  r_min sweep complete. {total_evals} evaluations "
          f"across {n_densities} density level(s).")


def _run_transferability_sweep(
    *,
    cfg: DictConfig,
    sweep_cfg: dict[str, Any],
    baselines: dict[str, Any],
    task: Any,
    device: torch.device,
    output_dir: str,
    plot_style_cfg: dict[str, Any],
) -> None:
    """Execute the optional size/density transferability sweep phase."""
    import math
    import os
    from pathlib import Path

    from graph_signal_diffusion.cli._wra_transferability import (
        build_channel_cache_info_entry,
        build_or_load_transfer_channel_cache,
        build_target_system_params,
        enforce_pmax_compatibility,
        extract_graph_build_cfg,
        find_compatible_channel_cache,
        load_checkpoint_hydra_config,
        load_wra_scenario_config,
        plot_transferability_results,
        resolve_model_cond_channels,
        resolve_source_system_params,
        save_transferability_summary,
        select_tail_network_ids,
        synthesize_transfer_raw_dir,
        transfer_dataset_name,
        validate_transfer_baselines,
    )
    from graph_signal_diffusion.cli._wra_transferability import (
        create_transfer_r_min_variant,
    )
    from graph_signal_diffusion.cli._wra_r_min_sweep import build_sweep_dataset_info
    from graph_signal_diffusion.datasets.wra.dataset import WRADataset
    from graph_signal_diffusion.datasets.wra.sampler import NetworkGroupedBatchSampler
    from torch_geometric.loader import DataLoader

    print(f"\n{'='*60}")
    print("TRANSFERABILITY SWEEP")
    print(f"{'='*60}")

    targets = list(sweep_cfg.get("targets", []))
    r_min_values = list(sweep_cfg.get("r_min_values", [0.6]))
    n_samples_per_network = int(sweep_cfg.get("n_samples_per_network", 50))
    batch_size_val = int(sweep_cfg.get("batch_size_val", 8))

    # --- Guard: r_min_values non-empty ---
    if not r_min_values:
        print("  [ERROR] r_min_values is empty. Skipping transferability sweep.")
        return

    # --- Guard: resolve baselines (subset of parent, auto-exclude AP) ---
    override_bl = sweep_cfg.get("baselines_to_compare", None)
    if override_bl is not None:
        bl_names = [str(n) for n in override_bl]
    else:
        bl_names = list(baselines.keys())
    bl_names = [n for n in bl_names if n.lower() != "ap"]
    missing = [n for n in bl_names if n not in baselines]
    if missing:
        print(f"  [ERROR] Baselines {missing} not instantiated in parent run. Skipping.")
        return
    if not bl_names:
        print("  [ERROR] No baselines remain after AP exclusion. Skipping.")
        return
    sweep_baselines = {n: baselines[n] for n in bl_names}

    print(f"  targets: {[t.get('scenario', t) if isinstance(t, dict) else t for t in targets]}")
    print(f"  baselines: {bl_names}")
    print(f"  r_min_values: {r_min_values}")

    # --- Extract source info from diffusion checkpoint ---
    diffusion_bl_cfg = _as_dict(
        OmegaConf.to_container(cfg.baselines.get("diffusion", {}), resolve=True)
    )
    ckpt_path = diffusion_bl_cfg.get("checkpoint_path")
    if not ckpt_path:
        print("  [WARNING] No diffusion checkpoint found. Skipping transferability sweep.")
        return
    source_ckpt_cfg = load_checkpoint_hydra_config(Path(str(ckpt_path)))
    model_cond_channels = resolve_model_cond_channels(source_ckpt_cfg, None)
    graph_cfg = extract_graph_build_cfg(source_ckpt_cfg)
    source_system_params = resolve_source_system_params(source_ckpt_cfg)
    source_p_max_dBm = source_system_params.get("P_max_dBm")

    conf_root = Path(__file__).resolve().parents[1] / "conf"
    transfer_cache_root = Path(
        str(sweep_cfg.get("transfer_cache", {}).get("root", "data/wra_channel_cache_transfer"))
    )
    auto_generate = bool(sweep_cfg.get("transfer_cache", {}).get("auto_generate", True))
    transfer_data_root = str(sweep_cfg.get("transfer_data_root", "data/wra_transfer"))
    base_r_min = float(r_min_values[0])

    # --- Prepare all (target, r_min) dataset directories ---
    all_dataset_names: list[str] = []
    eval_plan: list[dict[str, Any]] = []  # [{target, r_min, dataset_name, dataset_dir, system_params}]

    for target_entry in targets:
        target = _as_dict(target_entry) if isinstance(target_entry, dict) else {"scenario": str(target_entry)}
        scenario_name = str(target.get("scenario", ""))
        if not scenario_name:
            print(f"  [WARNING] Empty scenario name in target: {target}. Skipping.")
            continue

        eval_num_networks = int(target.get("eval_num_networks", 8))
        target_batch_size = int(target.get("batch_size_val", batch_size_val))
        target_is_native = bool(target.get("is_native", False))
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

        # --- Channel cache: fuzzy scan then auto-generate ---
        from graph_signal_diffusion.datasets.wra.channel_factory import (
            build_wra_channel_cache_metadata,
            normalize_channel_version,
            sanitize_channel_params,
        )
        from graph_signal_diffusion.cli._wra_transferability import (
            _as_dict as _t_as_dict,
            _select as _t_select,
            _to_float as _t_to_float,
            _to_int as _t_to_int,
        )

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

        # Build expected metadata for fuzzy lookup
        expected_meta = build_wra_channel_cache_metadata(
            seed=seed,
            num_networks=cache_num_networks,
            n_links=n_links,
            deployment_range=deployment_range,
            channel_version=channel_version,
            channel_params=channel_params,
            channel_config_name=f"transfer_{scenario_name}",
        )

        # Fuzzy scan first
        cache_artifact = find_compatible_channel_cache(
            scenario_name=scenario_name,
            cache_root=transfer_cache_root,
            expected_metadata=expected_meta,
            min_num_networks=eval_num_networks,
        )

        if cache_artifact is None:
            # Fall back to exact match + auto-generate
            cache_artifact = build_or_load_transfer_channel_cache(
                scenario_cfg=scenario_cfg,
                scenario_name=scenario_name,
                cache_root=transfer_cache_root,
                cache_num_networks=cache_num_networks,
                auto_generate=auto_generate,
                require_existing_cache=bool(
                    sweep_cfg.get("transfer_cache", {}).get("require_existing_cache", False)
                ),
            )

        total_in_cache = len(cache_artifact.channels)
        selected_ids = select_tail_network_ids(total_in_cache, eval_num_networks)
        print(f"\n  Target: {scenario_name}")
        print(f"    cache: {cache_artifact.cache_key} ({total_in_cache} networks)")
        print(f"    evaluating networks: {selected_ids[0]}..{selected_ids[-1]} ({eval_num_networks} total)")

        # Short scenario name for directory naming
        scenario_short = scenario_name.replace("wra_", "")

        cache_info_entry = build_channel_cache_info_entry(
            cache_file=cache_artifact.cache_file,
            cache_metadata=cache_artifact.metadata,
        )

        # --- Base directory (first r_min) ---
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

        # Plan for base r_min
        all_dataset_names.append(base_ds_name)
        eval_plan.append({
            "target_scenario": scenario_name,
            "r_min": base_r_min,
            "dataset_name": base_ds_name,
            "dataset_root": transfer_data_root,
            "system_params": dict(target_sys_params),
            "channel_params": dict(cache_artifact.channel_params),
            "n_links": n_links,
            "deployment_range": deployment_range,
            "eval_num_networks": eval_num_networks,
            "batch_size_val": target_batch_size,
            "is_native": target_is_native,
        })

        # --- r_min variants (additional values beyond the base) ---
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
            variant_sys_params = dict(target_sys_params)
            variant_sys_params["r_min"] = float(r_val)
            eval_plan.append({
                "target_scenario": scenario_name,
                "r_min": r_val,
                "dataset_name": variant_ds_name,
                "dataset_root": transfer_data_root,
                "system_params": variant_sys_params,
                "channel_params": dict(cache_artifact.channel_params),
                "n_links": n_links,
                "deployment_range": deployment_range,
                "eval_num_networks": eval_num_networks,
                "batch_size_val": target_batch_size,
                "is_native": target_is_native,
            })

    if not eval_plan:
        print("  No transfer targets prepared. Skipping.")
        return

    # --- Process all transfer datasets in one WRADataset (H dedup) ---
    print(f"\n  Processing {len(all_dataset_names)} transfer dataset(s)...")
    _all_ds = WRADataset(
        root=transfer_data_root,
        dataset_names=all_dataset_names,
        split="test",
        train_fraction=0.0,
        val_fraction=0.0,
        test_fraction=1.0,
        top_k=int(graph_cfg["top_k"]),
        min_degree=int(graph_cfg["min_degree"]),
        edge_normalize_method=str(graph_cfg["edge_normalize_method"]),
        spectral_normalize_edge_weights=bool(graph_cfg["spectral_normalize_edge_weights"]),
        model_cond_channels=model_cond_channels,
    )
    print(f"  All-transfer dataset: {len(_all_ds)} samples")
    del _all_ds

    # --- Evaluate each (target, r_min) ---
    sweep_results: list[dict[str, Any]] = []
    _wra_transfer_records: dict[str, dict[str, list]] = {}
    # Map ds_name → target scenario for aggregation
    _ds_to_scenario: dict[str, str] = {
        e["dataset_name"]: e["target_scenario"] for e in eval_plan
    }

    for plan_entry in eval_plan:
        ds_name = plan_entry["dataset_name"]
        r_val = plan_entry["r_min"]
        target_name = plan_entry["target_scenario"]
        print(f"\n  Evaluating {target_name} (r_min={r_val:.2f}, ds={ds_name})...")

        per_ds = WRADataset(
            root=transfer_data_root,
            dataset_names=[ds_name],
            split="test",
            train_fraction=0.0,
            val_fraction=0.0,
            test_fraction=1.0,
            top_k=int(graph_cfg["top_k"]),
            min_degree=int(graph_cfg["min_degree"]),
            edge_normalize_method=str(graph_cfg["edge_normalize_method"]),
            spectral_normalize_edge_weights=bool(graph_cfg["spectral_normalize_edge_weights"]),
            model_cond_channels=model_cond_channels,
        )

        if len(per_ds) == 0:
            print(f"    No samples for {ds_name}, skipping.")
            continue

        entry_batch_size = int(plan_entry.get("batch_size_val", batch_size_val))
        networks_per_batch = max(1, math.ceil(entry_batch_size / n_samples_per_network))
        eval_batch_size = networks_per_batch * n_samples_per_network
        sampler = NetworkGroupedBatchSampler(
            dataset=per_ds,
            batch_size=eval_batch_size,
            samples_per_network=n_samples_per_network,
            shuffle=False,
            seed=42,
        )
        loader = DataLoader(
            per_ds,
            batch_sampler=sampler,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            follow_batch=["y"],
            exclude_keys=["info"],
        )

        dataset_info = build_sweep_dataset_info(
            dataset_root=transfer_data_root,
            dataset_name=ds_name,
            r_min=r_val,
            system_params=plan_entry["system_params"],
            channel_params=plan_entry["channel_params"],
        )

        original_dataset_info = getattr(task, "dataset_info", None)
        original_reference_policy = getattr(task, "reference_policy", "expert")
        task.set_dataset_info(dataset_info)
        task.reference_policy = "none"

        try:
            for bl_name, baseline in sweep_baselines.items():
                viz_dir = os.path.join(
                    output_dir, "eval_viz", "transferability_sweep", ds_name, bl_name,
                )
                os.makedirs(viz_dir, exist_ok=True)

                metrics = baseline.evaluate(
                    loader, task,
                    max_batches=None,
                    viz_save_dir=viz_dir,
                    eval_split_name="test",
                )

                entry: dict[str, Any] = {
                    "target_scenario": target_name,
                    "baseline": bl_name,
                    "r_min": r_val,
                    "n_links": plan_entry["n_links"],
                    "deployment_range": plan_entry.get("deployment_range"),
                    "eval_num_networks": plan_entry["eval_num_networks"],
                    "is_native": plan_entry.get("is_native", False),
                }
                entry.update(metrics)

                # Compute per-network std for error bars in size-scaling plots
                _t_recs = getattr(baseline, "last_wra_evaluation_records", None)
                if _t_recs:
                    entry.update(_per_network_stats(list(_t_recs), r_val))

                sweep_results.append(entry)

                mean_rate = metrics.get("mean_rate_generated", float("nan"))
                viol_pct = metrics.get("rate_violation_percentage_generated", float("nan"))
                print(f"    [{bl_name}] mean_rate={mean_rate:.4f}, viol%={viol_pct:.2f}")

                # Collect WRA records for percentile evolution plots
                recs = _t_recs
                if recs:
                    rec_key = f"{ds_name}"
                    _wra_transfer_records.setdefault(rec_key, {})[bl_name] = recs

        finally:
            if original_dataset_info is not None:
                task.set_dataset_info(original_dataset_info)
            else:
                task.dataset_info = None
            task.reference_policy = original_reference_policy

    if not sweep_results:
        print("  No transferability sweep results collected.")
        return

    # --- Helper to parse size/density from scenario name ---
    def _parse_size_density(scenario_name: str) -> tuple[str, str]:
        """Extract (size, density) from scenario name like wra_medium_outdoor_low_density."""
        name = scenario_name.replace("wra_", "").replace("_density", "")
        for size_tag in (
            "medium-large-plus", "medium-large", "extra-large",
            "medium", "large", "mega", "small",
        ):
            if name.startswith(size_tag):
                rest = name[len(size_tag):].lstrip("_")
                for env in ("outdoor_", "indoor_"):
                    if rest.startswith(env):
                        rest = rest[len(env):]
                        break
                return size_tag, rest or "default"
        return "unknown", "unknown"

    # --- Save per-scenario summary ---
    save_transferability_summary(
        sweep_results=sweep_results,
        output_dir=output_dir,
    )

    # --- Save by-size and by-density aggregated summaries ---
    # Recompute QoS metrics from pooled per-receiver rates (not averaged metrics).
    import pandas as pd
    import numpy as np

    def _transfer_metrics_from_pooled(
        records: list[dict], r_min: float,
    ) -> dict[str, Any]:
        """Compute QoS metrics by pooling all receivers across records."""
        all_gen: list[np.ndarray] = []
        all_real: list[np.ndarray] = []
        for rec in records:
            gf = rec.get("gen_rates_final")
            if gf is not None:
                all_gen.append(np.asarray(gf).ravel())
            rf = rec.get("real_rates_final")
            if rf is not None:
                all_real.append(np.asarray(rf).ravel())
        if not all_gen:
            return {}
        pooled = np.concatenate(all_gen)
        n = len(pooled)
        violations = pooled < r_min
        viol_gaps = np.where(violations, r_min - pooled, 0.0)
        out: dict[str, Any] = {
            "mean_rate_generated": float(np.mean(pooled)),
            "rate_0.1pct_generated": float(np.percentile(pooled, 0.1)),
            "rate_0.5pct_generated": float(np.percentile(pooled, 0.5)),
            "rate_1pct_generated": float(np.percentile(pooled, 1)),
            "rate_5pct_generated": float(np.percentile(pooled, 5)),
            "rate_10pct_generated": float(np.percentile(pooled, 10)),
            "rate_violation_percentage_generated": float(
                100.0 * np.sum(violations) / n
            ) if n > 0 else 0.0,
            "rate_mean_violation_gap_pct_generated": float(
                100.0 * np.mean(viol_gaps / np.maximum(r_min, 1e-12))
            ) if n > 0 else 0.0,
            "feasibility_rate": float(1.0 - np.sum(violations) / n) if n > 0 else 0.0,
            "n_receivers_pooled": int(n),
        }
        if all_real:
            pooled_real = np.concatenate(all_real)
            out["mean_rate_real"] = float(np.mean(pooled_real))
            out["rate_1pct_real"] = float(np.percentile(pooled_real, 1))
            out["rate_5pct_real"] = float(np.percentile(pooled_real, 5))
        out.update(_per_network_stats(records, r_min))
        return out

    # Build mapping: (ds_key) → (size, density, baseline, r_min, records)
    _agg_groups_size: dict[tuple[str, str, float], list[dict]] = {}
    _agg_groups_density: dict[tuple[str, str, float], list[dict]] = {}

    for ds_key, records_by_bl in _wra_transfer_records.items():
        scenario = _ds_to_scenario.get(ds_key, "")
        if not scenario:
            continue
        size, density = _parse_size_density(scenario)
        # Find r_min for this ds_key from eval_plan
        r_val = r_min_values[0]  # default
        for pe in eval_plan:
            if pe["dataset_name"] == ds_key:
                r_val = pe["r_min"]
                break
        for bl_name, recs in records_by_bl.items():
            _agg_groups_size.setdefault((size, bl_name, r_val), []).extend(recs)
            _agg_groups_density.setdefault((density, bl_name, r_val), []).extend(recs)

    # By-size CSV
    if _agg_groups_size:
        by_size_rows: list[dict[str, Any]] = []
        for (size, bl_name, r_val), recs in sorted(_agg_groups_size.items()):
            row: dict[str, Any] = {"size": size, "baseline": bl_name, "r_min": r_val}
            row.update(_transfer_metrics_from_pooled(recs, r_val))
            by_size_rows.append(row)
        by_size_csv = os.path.join(output_dir, "transferability_sweep_by_size.csv")
        pd.DataFrame(by_size_rows).to_csv(by_size_csv, index=False)
        print(f"  By-size summary (pooled) saved to {by_size_csv}")

    # By-density CSV
    if _agg_groups_density:
        by_density_rows: list[dict[str, Any]] = []
        for (density, bl_name, r_val), recs in sorted(_agg_groups_density.items()):
            row = {"density": density, "baseline": bl_name, "r_min": r_val}
            row.update(_transfer_metrics_from_pooled(recs, r_val))
            by_density_rows.append(row)
        by_density_csv = os.path.join(output_dir, "transferability_sweep_by_density.csv")
        pd.DataFrame(by_density_rows).to_csv(by_density_csv, index=False)
        print(f"  By-density summary (pooled) saved to {by_density_csv}")

    # --- Save per-receiver rates for replotting ---
    if _wra_transfer_records:
        _save_wra_receiver_rates(
            _wra_transfer_records,
            output_dir,
            prefix="transferability_rates",
            scenario_map=_ds_to_scenario,
        )

    # --- Save diffusion overlay files for viz_trace (networks 0 and 1) ---
    if _wra_transfer_records:
        _save_diffusion_overlay_npz(
            _wra_transfer_records,
            output_dir,
            network_ids=[0, 1],
            scenario_map=_ds_to_scenario,
        )

    # --- Cross-target line plots ---
    plot_transferability_results(
        sweep_results=sweep_results,
        output_dir=output_dir,
        r_min_values=r_min_values,
        plot_style=plot_style_cfg,
    )

    # --- Per-(target, r_min) percentile evolution plots ---
    if _wra_transfer_records and getattr(task, "is_power_allocation_task", False):
        import os as _os
        from graph_signal_diffusion.baselines.wireless_resource_allocation.viz import (
            plot_wra_percentile_evolution_comparison,
        )
        _comp_dir = _os.path.join(output_dir, "eval_viz", "transferability_sweep", "comparison")
        _os.makedirs(_comp_dir, exist_ok=True)

        _wra_style_cfg = _as_dict(
            _as_dict(plot_style_cfg.get("plots", {})).get(
                "percentile_evolution_comparison", {}
            )
        )
        _baseline_colors = _normalize_color_map(
            _as_dict(plot_style_cfg.get("baseline", {})).get("colors", {})
        )

        single_r_min = len(r_min_values) == 1
        for rec_key, records_by_bl in _wra_transfer_records.items():
            if not records_by_bl:
                continue
            suffix = ""
            if not single_r_min:
                # Extract r_min from dataset name if present
                for rv in r_min_values:
                    if f"rmin{rv:.2f}" in rec_key:
                        suffix = f"_rmin{rv:.2f}"
                        break

            try:
                plot_wra_percentile_evolution_comparison(
                    records_by_bl,
                    save_path=_os.path.join(
                        _comp_dir,
                        f"wra_percentile_evolution_transfer_{rec_key}.pdf",
                    ),
                    baseline_order=list(sweep_baselines.keys()),
                    baseline_colors=_baseline_colors or None,
                    plot_style=_wra_style_cfg,
                )
            except Exception as exc:
                print(f"    [WARNING] Percentile plot for {rec_key} failed: {exc}")

        # --- Aggregated percentile evolution plots (by size / by density) ---
        # Build aggregated records: by size and by density
        _by_size: dict[str, dict[str, list]] = {}   # {size: {bl_name: [records...]}}
        _by_density: dict[str, dict[str, list]] = {} # {density: {bl_name: [records...]}}
        for ds_key, records_by_bl in _wra_transfer_records.items():
            scenario = _ds_to_scenario.get(ds_key, "")
            if not scenario:
                continue
            size, density = _parse_size_density(scenario)
            for bl_name, recs in records_by_bl.items():
                _by_size.setdefault(size, {}).setdefault(bl_name, []).extend(recs)
                _by_density.setdefault(density, {}).setdefault(bl_name, []).extend(recs)

        # Per-size aggregated plots (pool all densities)
        for size_label, agg_records_by_bl in sorted(_by_size.items()):
            if not agg_records_by_bl:
                continue
            try:
                plot_wra_percentile_evolution_comparison(
                    agg_records_by_bl,
                    save_path=_os.path.join(
                        _comp_dir,
                        f"wra_percentile_evolution_transfer_size_{size_label}.pdf",
                    ),
                    baseline_order=list(sweep_baselines.keys()),
                    baseline_colors=_baseline_colors or None,
                    plot_style=_wra_style_cfg,
                )
            except Exception as exc:
                print(f"    [WARNING] Aggregated size plot for {size_label} failed: {exc}")

        # Per-density aggregated plots (pool all sizes)
        for density_label, agg_records_by_bl in sorted(_by_density.items()):
            if not agg_records_by_bl:
                continue
            try:
                plot_wra_percentile_evolution_comparison(
                    agg_records_by_bl,
                    save_path=_os.path.join(
                        _comp_dir,
                        f"wra_percentile_evolution_transfer_density_{density_label}.pdf",
                    ),
                    baseline_order=list(sweep_baselines.keys()),
                    baseline_colors=_baseline_colors or None,
                    plot_style=_wra_style_cfg,
                )
            except Exception as exc:
                print(f"    [WARNING] Aggregated density plot for {density_label} failed: {exc}")

        n_size = len(_by_size)
        n_density = len(_by_density)
        print(f"  Aggregated percentile plots: {n_size} size level(s), {n_density} density level(s)")

    # --- Size-scaling plots (metric vs n_links, one curve per density) ---
    from graph_signal_diffusion.cli._wra_transferability import (
        plot_transferability_size_scaling,
    )

    # Native (training-size) targets are identified by is_native=True in sweep_results.
    _density_labels = _as_dict(sweep_cfg.get("density_labels", {}))

    # Generate size-scaling plots for each baseline that was evaluated
    _sweep_bl_names = sorted({str(r.get("baseline", "")) for r in sweep_results})
    for _bl_name in _sweep_bl_names:
        try:
            plot_transferability_size_scaling(
                sweep_results=sweep_results,
                output_dir=output_dir,
                baseline_name=_bl_name,
                density_labels=_density_labels or None,
                plot_style=plot_style_cfg,
            )
        except Exception as exc:
            print(f"  [WARNING] Size-scaling plots for {_bl_name} failed: {exc}")

    print(f"\n  Transferability sweep complete. {len(sweep_results)} evaluations.")


@hydra.main(version_base=None, config_path="../conf", config_name="compare_baselines")
def main(cfg: DictConfig):
    # Discover plugins
    discover_datasets()
    discover_tasks()
    discover_baselines()

    # Load centralized plotting style config and apply global style after
    # plugin discovery (which may mutate matplotlib rcParams via imports).
    _plot_style_cfg = _as_dict(cfg.get("plot_style", {}))
    _apply_global_plot_style(_plot_style_cfg)
    _apply_baseline_display_aliases(_plot_style_cfg)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data (initial build with default n_samples_per_input)
    print(f"Loading dataset: {cfg.dataset.name}")
    builder = get_dataset_builder(cfg.dataset.name)()
    datasets = builder.build_datasets(cfg.dataset)
    loaders = builder.build_loaders(cfg.dataset, datasets)

    # Load task with proper dataset-specific initialisation
    task = setup_task(cfg, builder, datasets)
    print(f"Using task: {cfg.task.name if task else 'None'}")
    if task is not None:
        if hasattr(task, "set_plot_style"):
            task.set_plot_style(_plot_style_cfg)
        else:
            setattr(task, "plot_style", _plot_style_cfg)

    hydra_cfg = HydraConfig.get()
    output_dir = hydra_cfg.runtime.output_dir

    # Expose paper-figure config + a stable output dir to the task (module-level
    # helper so ``os`` is the top-level import, not main()'s function-local one).
    _attach_paper_figure_cfg(task, cfg, output_dir)

    # Load and train baselines (training always uses base loaders)
    baselines = {}
    for baseline_name in cfg.baselines_to_compare:
        print(f"\n{'='*60}")
        print(f"Setting up: {baseline_name}")
        print(f"{'='*60}")

        # Convert to plain dict so we can safely extract fields that
        # contain _target_ keys meant as *data* (e.g. diffusion_overrides)
        # without Hydra recursively trying to instantiate them.
        # We also extract checkpoint_path so we can control the init order:
        # set diffusion_overrides *before* the checkpoint triggers model build.
        baseline_dict = OmegaConf.to_container(
            cfg.baselines[baseline_name], resolve=True,
        )
        _post_init: dict = {}
        for _key in ("diffusion_overrides",):
            val = baseline_dict.pop(_key, None)
            if val:
                _post_init[_key] = val

        # If we have overrides that must be set before checkpoint loading,
        # defer the checkpoint load to after instantiation.
        _checkpoint = None
        if _post_init and baseline_dict.get("checkpoint_path"):
            _checkpoint = baseline_dict.pop("checkpoint_path")

        baseline_cfg = OmegaConf.create(baseline_dict)
        baseline = instantiate(baseline_cfg, device=device)

        # Inject fields that couldn't go through Hydra instantiate, then
        # trigger the deferred checkpoint load.
        for k, v in _post_init.items():
            setattr(baseline, k, v)
        if _checkpoint is not None:
            baseline.load_checkpoint(_checkpoint)

        # Train if needed
        if baseline_cfg.get('needs_training', True):
            print(f"Training {baseline_name}...")
            baseline.fit(loaders['train'], loaders['val'])

            # Save checkpoint
            if baseline.save_dir:
                import os
                checkpoint_path = os.path.join(output_dir, f"{baseline_name}_final.pt")
                baseline.save_checkpoint(checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")

        baselines[baseline_name] = baseline

    # Compare all baselines on multiple splits
    # Read eval_splits from config (with sensible fallback)
    if "eval_splits" in cfg:
        eval_splits = OmegaConf.to_container(cfg.eval_splits, resolve=True)
    else:
        eval_splits = {"test": None}

    # ── Per-baseline loader rebuilding ────────────────────────────────
    # GRW generates multi-sample internally (needs n_samples_per_input=1);
    # Diffusion relies on ReplicatedDataset (needs n_samples_per_input=N).
    # We cache loaders by n_samples_per_input to avoid redundant rebuilds.
    _loader_cache: dict[int, dict] = {}
    _original_n_spi = cfg.dataset.get("n_samples_per_input", 1)

    def _get_loaders_for_baseline(baseline, baseline_name: str) -> dict:
        """Return data loaders with the correct n_samples_per_input for *baseline*."""
        n_spi = 1  # default: baseline handles its own replication
        if getattr(baseline, "needs_replicated_loader", False):
            n_spi = getattr(baseline, "n_samples", None) or 1
            if n_spi <= 1:
                warnings.warn(
                    f"[{baseline_name}] needs_replicated_loader=True but "
                    f"n_samples={n_spi}.  Set baselines.{baseline_name}."
                    f"n_samples to a value > 1 (e.g. 20) for ensemble "
                    f"evaluation.",
                    stacklevel=2,
                )
            print(f"  [{baseline_name}] needs ReplicatedDataset → "
                  f"n_samples_per_input={n_spi}")
        else:
            print(f"  [{baseline_name}] generates samples internally → "
                  f"n_samples_per_input=1")

        if n_spi in _loader_cache:
            return _loader_cache[n_spi]

        # Rebuild datasets + loaders with the required n_samples_per_input
        OmegaConf.update(cfg, "dataset.n_samples_per_input", n_spi)
        ds = builder.build_datasets(cfg.dataset)
        lds = builder.build_loaders(cfg.dataset, ds)
        _loader_cache[n_spi] = lds

        # Restore original so the next baseline starts from a clean state
        OmegaConf.update(cfg, "dataset.n_samples_per_input", _original_n_spi)
        return lds

    # ── W&B initialisation ────────────────────────────────────────────
    baseline_names = list(baselines.keys())
    with _init_wandb(cfg, hydra_cfg, baseline_names) as wandb_run:
        _skip_comparison = bool(cfg.get("skip_comparison", False))

        print(f"\n{'='*60}")
        if _skip_comparison:
            print("SKIPPING BASELINE COMPARISON (skip_comparison=true)")
        else:
            print("COMPARING BASELINES")
        print(f"  Splits: { {s: f'max_batches={m}' if m else 'all' for s, m in eval_splits.items()} }")
        if wandb_run is not None:
            print(f"  W&B run: {wandb_run.name} ({wandb_run.url})")
        print(f"{'='*60}")

        # Evaluate each baseline with its own loaders
        all_results: dict[str, dict[str, float]] = {}
        # WRA percentile-evolution records: {split: {baseline_name: [records]}}
        _wra_records: dict[str, dict[str, list]] = {}
        for name, baseline in baselines.items():
            if _skip_comparison:
                break
            bl_loaders = _get_loaders_for_baseline(baseline, name)

            # Also update task.n_samples_per_input so TaskV2 reshaping is correct
            n_spi = getattr(baseline, "n_samples", None) or 1
            if hasattr(task, "n_samples_per_input"):
                task.n_samples_per_input = n_spi

            # Replication factor *in the loader* (only for replicated-loader
            # baselines like Diffusion).  GRW handles multi-sample generation
            # internally and uses n_samples_per_input=1 in the loader, so its
            # loader_n_spi stays at 1.
            # When batch_size_val=B and n_replicas=K, the datamodule creates
            # batches of B items = B/K unique windows.  Without correction,
            # max_batches=M would yield M·B unique-window-batches for GRW but
            # only M·(B/K) unique-window-batches for Diffusion — an M·K:M ratio.
            # Multiplying max_batches by loader_n_spi restores parity.
            loader_n_spi = (
                (getattr(baseline, "n_samples", None) or 1)
                if getattr(baseline, "needs_replicated_loader", False)
                else 1
            )

            evaluator = UnifiedEvaluator(
                task, bl_loaders, save_dir=output_dir, plot_style=_plot_style_cfg,
            )
            row: dict[str, float] = {}
            for split, max_batches in eval_splits.items():
                effective_max_batches = (
                    None if max_batches is None else max_batches * loader_n_spi
                )
                if loader_n_spi > 1 and max_batches is not None:
                    print(f"  [{name}] scaling max_batches: {max_batches} → "
                          f"{effective_max_batches} (loader_n_spi={loader_n_spi})")
                metrics = evaluator.evaluate_baseline_on_split(
                    baseline, name, split=split, max_batches=effective_max_batches,
                )
                for k, v in metrics.items():
                    row[f"{split}/{k}"] = v

                # Capture WRA evaluation records for the percentile-evolution plot.
                # last_wra_evaluation_records is set by has_ensemble_evaluate baselines
                # after each evaluate() call; read it immediately before the next split
                # overwrites it.
                recs = getattr(baseline, "last_wra_evaluation_records", None)
                if recs:
                    _wra_records.setdefault(split, {})[name] = recs

            all_results[name] = row

        # ── Cross-baseline structural metric plots ─────────────────────
        # Collect last_structural_tensors exposed by each baseline.
        # (skipped when skip_comparison=true since no eval was run)
        _legacy_comparison_plot_cfg = _as_dict(cfg.get("comparison_plot", {}))
        _legacy_comparison_colors = _normalize_color_map(
            _legacy_comparison_plot_cfg.get("baseline_colors", {}),
        )
        _style_baseline_cfg = _as_dict(_plot_style_cfg.get("baseline", {}))
        _style_baseline_colors_all = _normalize_color_map(
            _style_baseline_cfg.get("colors", {}),
        )
        _style_comparison_colors = {
            name: color
            for name, color in _style_baseline_colors_all.items()
            if name.lower() not in {"expert", "real", "generated"}
        }
        _comparison_baseline_colors = None
        if _legacy_comparison_colors:
            warnings.warn(
                "cfg.comparison_plot.* is deprecated. Use "
                "cfg.plot_style.baseline.colors instead.",
                FutureWarning,
                stacklevel=2,
            )
            _comparison_baseline_colors = dict(_legacy_comparison_colors)
        if _style_comparison_colors:
            if _comparison_baseline_colors is None:
                _comparison_baseline_colors = {}
            _comparison_baseline_colors.update(_style_comparison_colors)

        _struct_tensors: dict[str, dict] = {}
        for name, baseline in baselines.items():
            if not bool(getattr(baseline, "structural_metrics", False)):
                continue
            st = getattr(baseline, "last_structural_tensors", None)
            if st is not None:
                _struct_tensors[name] = st

        if output_dir and len(_struct_tensors) >= 2:
            import os, numpy as np
            from graph_signal_diffusion.evaluation import structural_metrics as sm

            comp_dir = os.path.join(output_dir, "eval_viz", "comparison")
            os.makedirs(comp_dir, exist_ok=True)

            # Use the real-data tensor from the first baseline (all should
            # be identical since they share the same test set).
            first = next(iter(_struct_tensors.values()))
            cat_real = first["real"]
            window_stride = first["window_stride"]
            window_start_indices = first["window_start_indices"]

            # Sanity-check: all baselines' real tensors should have the
            # same shape (they all evaluate on the same underlying test set).
            for _bname, _bst in _struct_tensors.items():
                if _bst["real"].shape != cat_real.shape:
                    warnings.warn(
                        f"Real-tensor shape mismatch: first baseline has "
                        f"{cat_real.shape} but '{_bname}' has "
                        f"{_bst['real'].shape}.  Cross-baseline comparison "
                        f"may be unreliable — check n_samples_per_input / "
                        f"ReplicatedDataset deduplication."
                    )
                    break

            gen_returns_dict = {
                name: st["gen"] for name, st in _struct_tensors.items()
            }
            # Per-baseline window_start_indices (batch sizes may differ between
            # baselines due to different n_samples_per_input in their loaders).
            window_start_indices_gen = {
                name: st["window_start_indices"]
                for name, st in _struct_tensors.items()
            }

            # Infer max_lag / n_eigenvalues from the first structural baseline
            _any_name = next(iter(_struct_tensors.keys()))
            _any_bl = baselines[_any_name]
            max_lag = getattr(_any_bl, "structural_max_lag", 2)
            n_eigenvalues = getattr(_any_bl, "structural_n_eigenvalues", 5)

            sm.plot_structural_metrics_comparison(
                cat_real, gen_returns_dict,
                save_path=os.path.join(comp_dir, "structural_metrics_comparison.pdf"),
                max_lag=max_lag,
                n_eigenvalues=n_eigenvalues,
                window_stride=window_stride,
                window_start_indices=window_start_indices,
                window_start_indices_gen=window_start_indices_gen,
                baseline_colors=_comparison_baseline_colors,
            )

            # Paper-ready 2-panel structural comparison (Fig 2): combined
            # autocovariance ACF (r & r² on twin axes, un-normalized) +
            # eigenvalue comparison.  Gated by paper_figures.enabled.
            _pf = _as_dict(cfg.get("paper_figures", {}))
            if _pf.get("enabled", False):
                _f2 = _as_dict(_pf.get("fig2", {}))
                _paper_dir = os.path.join(output_dir, "eval_viz", "paper")
                os.makedirs(_paper_dir, exist_ok=True)
                sm.plot_structural_comparison_paper(
                    cat_real, gen_returns_dict,
                    save_path=os.path.join(
                        _paper_dir, "fig2_structural_comparison.pdf",
                    ),
                    max_lag=(
                        _f2.get("max_lag")
                        if _f2.get("max_lag") is not None else max_lag
                    ),
                    n_eigenvalues=(
                        _f2.get("n_eigenvalues")
                        if _f2.get("n_eigenvalues") is not None else n_eigenvalues
                    ),
                    window_stride=window_stride,
                    window_start_indices=window_start_indices,
                    window_start_indices_gen=window_start_indices_gen,
                    baseline_colors=_comparison_baseline_colors,
                    normalize=bool(_f2.get("normalize", True)),
                    include_lag0=bool(_f2.get("include_lag0", False)),
                    show_iqr=bool(_f2.get("show_iqr", True)),
                    twin_axis=bool(_f2.get("twin_axis", False)),
                    plot_style=_plot_style_cfg,  # aesthetics: plot_style.plots.structural_comparison
                )

            # Cumulative-return std comparison
            std_real = sm.compute_cumulative_return_std(cat_real).numpy()
            std_real_per_stock = sm.compute_cumulative_return_std(
                cat_real, per_stock=True,
            ).numpy()  # [T, N]
            std_gen_dict = {
                name: sm.compute_cumulative_return_std(st["gen"]).numpy()
                for name, st in _struct_tensors.items()
            }
            std_gen_per_stock_dict = {
                name: sm.compute_cumulative_return_std(
                    st["gen"], per_stock=True,
                ).numpy()  # [T, N]
                for name, st in _struct_tensors.items()
            }
            T_steps = cat_real.shape[1]
            sigma_bar = float(cat_real.std(dim=(0, 1)).mean().item())
            theoretical = sigma_bar * np.sqrt(np.arange(1, T_steps + 1))

            sm.plot_cumulative_return_std_comparison(
                std_real, std_gen_dict,
                save_path=os.path.join(comp_dir, "cumulative_return_std_comparison.pdf"),
                theoretical=theoretical,
                std_real_per_stock=std_real_per_stock,
                std_gen_per_stock_dict=std_gen_per_stock_dict,
                baseline_colors=_comparison_baseline_colors,
            )

            # Dump consolidated comparison plot-data sidecar.
            _dump_plot_data = bool(cfg.get("viz", {}).get("dump_plot_data", True))
            if _dump_plot_data:
                try:
                    from graph_signal_diffusion.evaluation.plot_data_io import (
                        dump_comparison_structural,
                    )
                    # Heavy cat_real / cat_gen tensors are referenced by
                    # relative path back to each baseline's structural_data.npz
                    # to avoid duplication.
                    _per_baseline_paths = {
                        name: os.path.join(
                            output_dir, "eval_viz", name, "test", "structural_data.npz",
                        )
                        for name in _struct_tensors.keys()
                    }
                    dump_comparison_structural(
                        os.path.join(comp_dir, "structural_data.npz"),
                        baseline_names=list(_struct_tensors.keys()),
                        per_baseline_paths=_per_baseline_paths,
                        max_lag=max_lag,
                        n_eigenvalues=n_eigenvalues,
                        cum_std_real=std_real,
                        cum_std_real_per_stock=std_real_per_stock,
                        cum_std_gen_dict=std_gen_dict,
                        cum_std_gen_per_stock_dict=std_gen_per_stock_dict,
                        theoretical=theoretical,
                        baseline_colors=_comparison_baseline_colors,
                    )
                except Exception:
                    import traceback as _dump_tb
                    warnings.warn(
                        "Failed to dump comparison structural plot data sidecar:\n"
                        + _dump_tb.format_exc()
                    )

            print(f"\n  Comparative structural plots saved to {comp_dir}/")

        # ── Cross-baseline NLL comparison plots ────────────────────────
        # Collect ALL baselines that can act as NLL scorers, then produce
        # one histogram per scorer.
        import traceback as _tb

        all_scorers = [
            (name, bl) for name, bl in baselines.items()
            if hasattr(bl, "compute_nll")
            and getattr(bl, "last_nll_tensors", None) is not None
        ]

        if output_dir and all_scorers:
            import os
            from graph_signal_diffusion.evaluation import structural_metrics as sm

            comp_dir = os.path.join(output_dir, "eval_viz", "comparison")
            os.makedirs(comp_dir, exist_ok=True)
            nll_log_x = bool(cfg.get("viz", {}).get("nll_log_x", True))

            # Accumulator for the paper 2×2 dual-scorer NLL figure (Fig 3).
            _paper_nll_scorers: list[dict] = []

            for scoring_name, scoring_bl in all_scorers:
                try:
                    scoring_nll = scoring_bl.last_nll_tensors
                    nll_real = scoring_nll["nll_real"]
                    nll_gen_dict = {scoring_name: scoring_nll["nll_gen"]}

                    # Score each other baseline's generated returns under
                    # this scoring model.
                    for name, baseline in baselines.items():
                        if name == scoring_name:
                            continue
                        st = getattr(baseline, "last_structural_tensors", None)
                        if st is not None:
                            gen_ret = st["gen"]  # [B, T, N]
                            nll_other = scoring_bl.compute_nll(gen_ret)
                            nll_other = nll_other.detach().cpu().numpy()
                            nll_gen_dict[name] = nll_other

                    if len(nll_gen_dict) >= 2:
                        sm.plot_nll_histograms_comparison(
                            nll_real, nll_gen_dict,
                            save_path=os.path.join(
                                comp_dir,
                                f"nll_histogram_{scoring_name}_scored.pdf",
                            ),
                            log_x=nll_log_x,
                            scoring_model_name=scoring_name,
                            baseline_colors=_comparison_baseline_colors,
                        )
                        print(
                            f"  NLL histogram ({scoring_name}-scored) "
                            f"saved to {comp_dir}/"
                        )
                        # Stash this scorer's data for the paper Fig 3 (2×2).
                        _paper_nll_scorers.append({
                            "name": scoring_name,
                            "nll_real": nll_real,
                            "nll_gen_dict": dict(nll_gen_dict),
                        })

                    # ── Cumulative NLL comparison (per-horizon) ────────
                    cum_nll_real = scoring_nll.get("cum_nll_real")
                    if cum_nll_real is not None and len(nll_gen_dict) >= 2:
                        cum_nll_gen_dict = {}
                        cum_nll_own = scoring_nll.get("cum_nll_gen")
                        if cum_nll_own is not None:
                            cum_nll_gen_dict[scoring_name] = cum_nll_own
                        for name, baseline in baselines.items():
                            if name == scoring_name:
                                continue
                            st = getattr(baseline, "last_structural_tensors", None)
                            if (
                                st is not None
                                and hasattr(scoring_bl, "compute_cumulative_nll")
                            ):
                                gen_ret = st["gen"]
                                cum_nll_other = scoring_bl.compute_cumulative_nll(
                                    gen_ret,
                                ).numpy()
                                cum_nll_gen_dict[name] = cum_nll_other
                        if len(cum_nll_gen_dict) >= 2:
                            sm.plot_cumulative_nll_histograms_comparison(
                                cum_nll_real, cum_nll_gen_dict,
                                save_path=os.path.join(
                                    comp_dir,
                                    f"cumulative_nll_histogram_{scoring_name}_scored.pdf",
                                ),
                                scoring_model_name=scoring_name,
                                baseline_colors=_comparison_baseline_colors,
                            )
                            print(
                                f"  Cumulative NLL histogram ({scoring_name}-scored) "
                                f"saved to {comp_dir}/"
                            )

                    # Dump consolidated comparison NLL plot data sidecar
                    # (one file per scoring model).
                    if bool(cfg.get("viz", {}).get("dump_plot_data", True)):
                        try:
                            from graph_signal_diffusion.evaluation.plot_data_io import (
                                dump_comparison_nll,
                            )
                            dump_comparison_nll(
                                os.path.join(
                                    comp_dir, f"nll_data__{scoring_name}.npz",
                                ),
                                nll_real=nll_real,
                                nll_gen_dict=nll_gen_dict,
                                cum_nll_real=cum_nll_real,
                                cum_nll_gen_dict=cum_nll_gen_dict if cum_nll_gen_dict else None,
                                scoring_model_name=scoring_name,
                                bins=int(cfg.get("viz", {}).get("nll_bins", 60)),
                                log_x=nll_log_x,
                                baseline_colors=_comparison_baseline_colors,
                            )
                        except Exception:
                            print(
                                f"  [WARNING] Failed to dump NLL plot data sidecar "
                                f"for scorer {scoring_name}.\n{_tb.format_exc()}"
                            )
                except Exception:
                    print(
                        f"  [WARNING] NLL scoring under {scoring_name} failed; "
                        f"skipping.\n{_tb.format_exc()}"
                    )

            # Paper-ready dual-scorer NLL comparison (Fig 3): rows = scorers,
            # cols = {histogram, CDF}.  Gated by paper_figures.enabled.  With
            # both GRW and U-GNN scorers this is 2×2; if diffusion ELBO
            # self-scoring failed it degrades to a single 1×2 row.
            _pf_nll = _as_dict(cfg.get("paper_figures", {}))
            if _pf_nll.get("enabled", False) and _paper_nll_scorers:
                _f3 = _as_dict(_pf_nll.get("fig3", {}))
                _paper_dir = os.path.join(output_dir, "eval_viz", "paper")
                os.makedirs(_paper_dir, exist_ok=True)
                _f3_logx = _f3.get("log_x")
                # GRW's NLL is summed over N·T (stock, timestep) pairs; express
                # that row in per-element nats so it reads against the i.i.d.
                # Gaussian floor (≈1.42).  N·T comes from the [B, T, N] real
                # tensor; the diffusion-proxy row is left unchanged.
                _fig3_scorers = list(_paper_nll_scorers)
                if _f3.get("grw_per_element_nats", True):
                    _nt = None
                    try:
                        _r = next(iter(_struct_tensors.values()))["real"]
                        _nt = int(_r.shape[1]) * int(_r.shape[2])
                    except Exception:
                        _nt = None
                    _fig3_scorers = [
                        sm.annotate_grw_per_element_nats(_s, _nt)
                        for _s in _fig3_scorers
                    ]
                # Diffusion proxy row → per-element denoising MSE (÷ num_timesteps).
                if _f3.get("diffusion_per_step_mse", True):
                    try:
                        _dts = int(cfg.get("diffusion", {}).get("num_timesteps"))
                    except Exception:
                        _dts = None
                    _fig3_scorers = [
                        sm.annotate_diffusion_per_step_mse(_s, _dts)
                        for _s in _fig3_scorers
                    ]
                sm.plot_nll_comparison_paper(
                    _fig3_scorers,
                    save_path=os.path.join(_paper_dir, "fig3_nll_comparison.pdf"),
                    bins=int(_f3.get("bins", cfg.get("viz", {}).get("nll_bins", 60))),
                    log_x=(nll_log_x if _f3_logx is None else bool(_f3_logx)),
                    baseline_colors=_comparison_baseline_colors,
                    plot_style=_plot_style_cfg,  # aesthetics: plot_style.plots.nll_comparison
                )
                if len(_paper_nll_scorers) < 2:
                    print(
                        "  [Fig 3] Only one NLL scorer available (diffusion ELBO "
                        "self-scoring may have failed); emitted 1×2 instead of 2×2."
                    )

        # ── Cross-baseline WRA percentile-evolution plots ──────────────
        if output_dir and _wra_records and getattr(task, "is_power_allocation_task", False):
            import os as _os
            from graph_signal_diffusion.baselines.wireless_resource_allocation.viz import (
                plot_wra_percentile_evolution_comparison,
            )
            _comp_dir = _os.path.join(output_dir, "eval_viz", "comparison")
            _os.makedirs(_comp_dir, exist_ok=True)
            _wra_style_cfg = _as_dict(
                _as_dict(_plot_style_cfg.get("plots", {})).get(
                    "percentile_evolution_comparison", {}
                )
            )
            _wra_plot_cfg = _as_dict(cfg.get("wra_percentile_plot", {}))
            if _wra_plot_cfg:
                warnings.warn(
                    "cfg.wra_percentile_plot.* is deprecated. Use "
                    "cfg.plot_style.plots.percentile_evolution_comparison.*",
                    FutureWarning,
                    stacklevel=2,
                )
            _baseline_colors = None
            _r_min_override = None
            _marker_every = 5
            _error_display = "bars"
            if isinstance(_wra_plot_cfg, dict):
                _colors_cfg = _wra_plot_cfg.get("baseline_colors", None)
                if isinstance(_colors_cfg, dict):
                    _baseline_colors = {
                        str(name): str(color)
                        for name, color in _colors_cfg.items()
                    }
                if _wra_plot_cfg.get("r_min", None) is not None:
                    _r_min_override = _wra_plot_cfg.get("r_min")
                if _wra_plot_cfg.get("marker_every", None) is not None:
                    _marker_every = _wra_plot_cfg.get("marker_every")
                if _wra_plot_cfg.get("error_display", None) is not None:
                    _error_display = str(_wra_plot_cfg.get("error_display"))
            _wra_style_colors = _normalize_color_map(_wra_style_cfg.get("baseline_colors", {}))
            if not _wra_style_colors:
                _wra_style_colors = dict(_style_baseline_colors_all)
            if _wra_style_colors:
                if _baseline_colors is None:
                    _baseline_colors = {}
                _baseline_colors.update(_wra_style_colors)
            if _wra_style_cfg.get("r_min", None) is not None:
                _r_min_override = _wra_style_cfg.get("r_min")
            if _wra_style_cfg.get("marker_every", None) is not None:
                _marker_every = _wra_style_cfg.get("marker_every")
            if _wra_style_cfg.get("error_display", None) is not None:
                _error_display = str(_wra_style_cfg.get("error_display"))

            for split, records_by_baseline in _wra_records.items():
                missing = [n for n in baselines if n not in records_by_baseline]
                if missing:
                    print(
                        f"  [WRA percentile plot] baselines without records for "
                        f"split '{split}': {missing} — they will be omitted."
                    )
                if not records_by_baseline:
                    continue

                # --- Global pooled plot (always) ---
                plot_wra_percentile_evolution_comparison(
                    records_by_baseline,
                    save_path=_os.path.join(
                        _comp_dir, f"wra_percentile_evolution_{split}.pdf"
                    ),
                    baseline_order=list(baselines.keys()),
                    baseline_colors=_baseline_colors,
                    r_min=_r_min_override,
                    marker_every=_marker_every,
                    error_display=_error_display,
                    plot_style=_wra_style_cfg,
                )

                # --- Per-subdataset plots (when >1 subdataset) ---
                all_ds_names: set[str] = set()
                for _recs in records_by_baseline.values():
                    for _rec in _recs:
                        ds = _rec.get("dataset_name")
                        if ds:
                            all_ds_names.add(str(ds))

                if len(all_ds_names) > 1:
                    from collections import defaultdict as _defaultdict
                    # Group each baseline's records by dataset_name
                    _grouped: dict[str, dict[str, list]] = {}
                    for bl_name, bl_recs in records_by_baseline.items():
                        _by_ds: dict[str, list] = _defaultdict(list)
                        for _rec in bl_recs:
                            ds = str(_rec.get("dataset_name", ""))
                            _by_ds[ds].append(_rec)
                        _grouped[bl_name] = dict(_by_ds)

                    for ds_name in sorted(all_ds_names):
                        short_name = ds_name.rsplit("/", 1)[-1] if "/" in ds_name else ds_name
                        ds_records_by_baseline: dict[str, list] = {}
                        for bl_name, bl_ds_map in _grouped.items():
                            if ds_name in bl_ds_map:
                                ds_records_by_baseline[bl_name] = bl_ds_map[ds_name]
                        if ds_records_by_baseline:
                            plot_wra_percentile_evolution_comparison(
                                ds_records_by_baseline,
                                save_path=_os.path.join(
                                    _comp_dir,
                                    f"wra_percentile_evolution_{split}_{short_name}.pdf",
                                ),
                                baseline_order=list(baselines.keys()),
                                baseline_colors=_baseline_colors,
                                r_min=_r_min_override,
                                marker_every=_marker_every,
                                error_display=_error_display,
                                plot_style=_wra_style_cfg,
                            )
                    print(
                        f"  Per-subdataset percentile plots ({len(all_ds_names)} "
                        f"subdatasets) saved to {_comp_dir}/"
                    )

                    # --- Per-r_min plots (aggregate across densities) ---
                    # Group records by r_min, pooling all densities together.
                    all_r_min_vals: set[float] = set()
                    for _recs in records_by_baseline.values():
                        for _rec in _recs:
                            _rm = _rec.get("r_min")
                            if _rm is not None:
                                all_r_min_vals.add(float(_rm))

                    if len(all_r_min_vals) > 1:
                        for r_min_val in sorted(all_r_min_vals):
                            rmin_records_by_baseline: dict[str, list] = {}
                            for bl_name, bl_recs in records_by_baseline.items():
                                filtered = [
                                    _rec for _rec in bl_recs
                                    if _rec.get("r_min") is not None
                                    and abs(float(_rec["r_min"]) - r_min_val) < 1e-6
                                ]
                                if filtered:
                                    rmin_records_by_baseline[bl_name] = filtered
                            if rmin_records_by_baseline:
                                plot_wra_percentile_evolution_comparison(
                                    rmin_records_by_baseline,
                                    save_path=_os.path.join(
                                        _comp_dir,
                                        f"wra_percentile_evolution_{split}_rmin_{r_min_val:.2f}.pdf",
                                    ),
                                    baseline_order=list(baselines.keys()),
                                    baseline_colors=_baseline_colors,
                                    r_min=r_min_val,
                                    marker_every=_marker_every,
                                    error_display=_error_display,
                                    plot_style=_wra_style_cfg,
                                )
                        print(
                            f"  Per-r_min percentile plots ({len(all_r_min_vals)} "
                            f"r_min levels) saved to {_comp_dir}/"
                        )

        import pandas as pd
        results_df = pd.DataFrame(all_results).T if all_results else pd.DataFrame()

        # Save comparison CSV, markdown summary, and cross-baseline plots
        if output_dir and not results_df.empty:
            from graph_signal_diffusion.evaluation.evaluator import save_comparison_results
            save_comparison_results(
                results_df,
                save_dir=output_dir,
                visualize=True,
                plot_style=_plot_style_cfg,
            )
            # Per-subdataset comparison table (only when subdataset/ keys exist)
            if _wra_records:
                _save_per_subdataset_comparison(results_df, output_dir, _wra_records)

            # Per-receiver rates for replotting (one file per eval split)
            # Regroup {bl_name: [recs]} → {dataset_name: {bl_name: [recs]}}
            # so per-dataset granularity is preserved for reconstruction.
            for _split, _split_records in _wra_records.items():
                if _split_records:
                    _by_dataset: dict[str, dict[str, list]] = {}
                    for _bl_name, _bl_recs in _split_records.items():
                        for _rec in _bl_recs:
                            _ds = _rec.get("dataset_name", "unknown")
                            _by_dataset.setdefault(_ds, {}).setdefault(
                                _bl_name, []
                            ).append(_rec)
                    _save_wra_receiver_rates(
                        _by_dataset,
                        output_dir,
                        prefix=f"comparison_rates_{_split}",
                    )

            # Diffusion overlay files for viz_trace (networks 0 and 1)
            for _split, _split_records in _wra_records.items():
                if _split_records and "diffusion" in _split_records:
                    _by_dataset_ov: dict[str, dict[str, list]] = {}
                    for _bl_name, _bl_recs in _split_records.items():
                        for _rec in _bl_recs:
                            _ds = _rec.get("dataset_name", "unknown")
                            _by_dataset_ov.setdefault(_ds, {}).setdefault(
                                _bl_name, []
                            ).append(_rec)
                    _save_diffusion_overlay_npz(
                        _by_dataset_ov,
                        output_dir,
                        network_ids=[0, 1],
                    )

        # ── Optional r_min sweep phase ─────────────────────────────────
        _sweep_cfg = _as_dict(cfg.get("r_min_sweep", {}))
        if _sweep_cfg.get("enabled", False) and output_dir:
            _run_r_min_sweep(
                cfg=cfg,
                sweep_cfg=_sweep_cfg,
                baselines=baselines,
                task=task,
                device=device,
                output_dir=output_dir,
                plot_style_cfg=_plot_style_cfg,
            )

        # ── Optional transferability sweep phase ──────────────────────
        _transfer_cfg = _as_dict(cfg.get("transferability_sweep", {}))
        if _transfer_cfg.get("enabled", False) and output_dir:
            _run_transferability_sweep(
                cfg=cfg,
                sweep_cfg=_transfer_cfg,
                baselines=baselines,
                task=task,
                device=device,
                output_dir=output_dir,
                plot_style_cfg=_plot_style_cfg,
            )

        # ── Log metrics to W&B ────────────────────────────────────────
        if not results_df.empty:
            log_results_to_wandb(wandb_run, results_df)

            print(f"\n{'='*60}")
            print("FINAL RESULTS")
            print(f"{'='*60}")
            print(results_df.to_string())
        print(f"\nResults saved to {output_dir}/")
        print(f"Visualisations saved to {output_dir}/eval_viz/")


if __name__ == "__main__":
    main()
