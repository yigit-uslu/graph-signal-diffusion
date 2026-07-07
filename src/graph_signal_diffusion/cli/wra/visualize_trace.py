#!/usr/bin/env python
"""Post-training visualization of tracked-network trace dynamics.

Reads ``tracked_network_trace.jsonl`` produced during PD training and
generates per-network multi-panel figures showing the evolution of power
allocations, ergodic rates (with windowed averages), dual multipliers,
and constraint slacks for selected receivers.

Usage
-----
    python -m graph_signal_diffusion.cli.wra.visualize_trace \
        --config-name=pd_trace_visualization/wra_medium \
        input_dir=/path/to/pd_training_output
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import yaml
import hydra
from omegaconf import DictConfig, OmegaConf

from graph_signal_diffusion.datasets.wra.channel_factory import (
    register_wra_hydra_resolvers,
)
from graph_signal_diffusion.trainers.pd_visualization import (
    parse_trace_jsonl,
    select_receivers_by_groups,
    visualize_trace_dynamics,
    visualize_trace_row,
)

register_wra_hydra_resolvers()

log = logging.getLogger(__name__)


def _resolve_output_dir(cfg: DictConfig, input_dir: Path) -> Path:
    """Determine where to write figures."""
    explicit = cfg.output.get("dir")
    if explicit is not None:
        out = Path(explicit)
    else:
        out = input_dir
    out.mkdir(parents=True, exist_ok=True)
    return out


def _resolve_plot_cfg(cfg: DictConfig) -> dict:
    """Build a unified plot-config dict from Hydra config groups.

    Merges ``plot_style``, ``modes``, and ``panels`` into a single dict
    so the visualisation functions receive everything through one parameter.
    """
    raw = OmegaConf.to_container(cfg.plot_style, resolve=True)
    if not isinstance(raw, dict):
        raw = {}
    # Merge output format so the plotting function can use it.
    raw.setdefault("format", str(cfg.output.get("format", "pdf")))

    # Merge mode-specific overrides.
    modes_raw = OmegaConf.to_container(cfg.get("modes", {}), resolve=True)
    if isinstance(modes_raw, dict):
        raw["modes"] = modes_raw

    # Merge per-panel config.
    panels_raw = OmegaConf.to_container(cfg.get("panels", {}), resolve=True)
    if isinstance(panels_raw, dict):
        raw["panels"] = panels_raw

    return raw


@hydra.main(
    version_base=None,
    config_path="../../conf/wra_generation",
    config_name="pd_trace_visualization/wra_debug",
)
def main(cfg: DictConfig) -> None:
    input_dir = Path(cfg.input_dir)
    trace_path = input_dir / cfg.trace_filename

    if not trace_path.exists():
        log.error("Trace file not found: %s", trace_path)
        return

    log.info("Parsing trace file: %s", trace_path)
    trace_data = parse_trace_jsonl(trace_path)
    if not trace_data:
        log.warning("No network data found in %s", trace_path)
        return

    output_dir = _resolve_output_dir(cfg, input_dir)
    plot_cfg = _resolve_plot_cfg(cfg)

    # Resolve receiver selection settings.
    rx_cfg = cfg.receiver_selection
    rx_mode = str(rx_cfg.get("mode", "auto")).strip().lower()
    auto_strategy = str(rx_cfg.get("auto_strategy", "neg_corr_pair")).strip().lower()
    neg_corr_pair_cfg = OmegaConf.to_container(
        rx_cfg.get("neg_corr_pair", {}), resolve=True,
    )
    if not isinstance(neg_corr_pair_cfg, dict):
        neg_corr_pair_cfg = {}
    seed_raw = rx_cfg.get("seed")
    viz_seed: int | None = int(seed_raw) if seed_raw is not None else None
    explicit_indices = OmegaConf.to_container(
        rx_cfg.get("receiver_indices", []), resolve=True,
    )
    if not isinstance(explicit_indices, list):
        explicit_indices = []

    # Resolve quantile groups.
    groups_raw = OmegaConf.to_container(rx_cfg.get("groups", []), resolve=True)
    if not isinstance(groups_raw, list) or not groups_raw:
        # Fallback: single top-k group for backward compatibility.
        auto_k = int(rx_cfg.get("auto_k", 2))
        groups_raw = [{"quantile": [0.75, 1.0], "count": auto_k,
                       "palette": "PuRd", "palette_range": [0.35, 0.85],
                       "scatter_pairs": min(auto_k // 2, 4)}]

    # Collection window.
    cw_raw = cfg.get("collection_window", None)
    collection_window: int | None = int(cw_raw) if cw_raw is not None else None

    # Auto-parse P_max from the PD training run's saved Hydra config
    # and inject into the scatter panel config.
    scatter_pcfg = plot_cfg.setdefault("panels", {}).setdefault("scatter", {})
    if "p_max" not in scatter_pcfg:
        hydra_cfg_path = input_dir / ".hydra" / "config.yaml"
        if hydra_cfg_path.exists():
            try:
                with open(hydra_cfg_path) as _f:
                    run_cfg = yaml.safe_load(_f)
                p_max_dbm = run_cfg.get("system", {}).get("P_max_dBm")
                if p_max_dbm is not None:
                    p_max_linear = 10.0 ** (float(p_max_dbm) / 10.0) / 1000.0
                    scatter_pcfg["p_max"] = p_max_linear
                    log.info("Auto-detected P_max = %.6g W (%.1f dBm)", p_max_linear, p_max_dbm)
            except Exception as exc:
                log.warning("Could not parse P_max from %s: %s", hydra_cfg_path, exc)

    # Overlay settings (generated power samples on scatter panel).
    overlay_raw = OmegaConf.to_container(
        cfg.get("overlay", {}), resolve=True,
    )
    if not isinstance(overlay_raw, dict):
        overlay_raw = {}
    overlay_cfg = {
        k: v for k, v in overlay_raw.items()
        if k in ("marker", "color", "alpha", "marker_size", "label")
    }
    overlay_samples: dict[int, np.ndarray] | None = None
    overlay_path_str = overlay_raw.get("generated_powers_path")
    if overlay_path_str is not None:
        overlay_path = Path(overlay_path_str)
        if overlay_path.exists():
            log.info("Loading overlay samples from: %s", overlay_path)
            npz = np.load(overlay_path, allow_pickle=False)
            net_ids = npz.get("network_ids")
            if net_ids is not None:
                overlay_samples = {}
                for nid in net_ids:
                    key = f"network_{int(nid)}_powers"
                    if key in npz:
                        overlay_samples[int(nid)] = npz[key]
            npz.close()
        else:
            log.warning("Overlay file not found: %s", overlay_path)

    # Visualization mode: "full" (4-panel column), "row" (2-column), or "both".
    viz_mode = str(cfg.get("visualization_mode", "full")).strip().lower()
    run_full = viz_mode in ("full", "both")
    run_row = viz_mode in ("row", "both")
    if not run_full and not run_row:
        log.error("Unknown visualization_mode '%s'; expected full, row, or both.", viz_mode)
        return

    log.info(
        "Found %d tracked network(s): %s  (mode=%s)",
        len(trace_data),
        sorted(trace_data.keys()),
        viz_mode,
    )

    # Common kwargs shared by both visualisation functions.
    common_kwargs = dict(
        auto_strategy=auto_strategy,
        neg_corr_pair_cfg=neg_corr_pair_cfg,
        collection_window=collection_window,
        overlay_samples=overlay_samples,
        overlay_cfg=overlay_cfg if overlay_cfg else None,
        plot_cfg=plot_cfg,
        seed=viz_seed,
    )

    for network_id in sorted(trace_data.keys()):
        net_info = trace_data[network_id]
        n_epochs = len(net_info.get("epochs", []))
        if n_epochs == 0:
            log.warning("Network %d has no epoch data; skipping.", network_id)
            continue

        if rx_mode == "explicit":
            grouped_receivers = [[int(i) for i in explicit_indices]]
            group_cfgs = [{"palette": "tab10", "palette_range": [0.0, 1.0],
                           "scatter_pairs": len(explicit_indices) // 2}]
        else:
            grouped_receivers = select_receivers_by_groups(
                trace_data, network_id, groups=groups_raw,
                strategy=auto_strategy,
                neg_corr_pair_cfg=neg_corr_pair_cfg,
                collection_window=collection_window,
                seed=viz_seed,
            )
            group_cfgs = groups_raw

        total_rx = sum(len(g) for g in grouped_receivers)
        log.info(
            "Visualizing network %d: %d epochs, %d receivers in %d groups (strategy=%s)",
            network_id, n_epochs, total_rx, len(grouped_receivers), auto_strategy,
        )
        for gi, grx in enumerate(grouped_receivers):
            q = group_cfgs[gi].get("quantile", [0, 1])
            log.info("  Group %d [%.0f-%.0f%%]: %s", gi, q[0] * 100, q[1] * 100, grx)

        if run_full:
            path = visualize_trace_dynamics(
                trace_data=trace_data,
                network_id=network_id,
                output_dir=output_dir,
                grouped_receivers=grouped_receivers,
                group_cfgs=group_cfgs,
                **common_kwargs,
            )
            if path is not None:
                log.info("  [full] %s", path)

        if run_row:
            path = visualize_trace_row(
                trace_data=trace_data,
                network_id=network_id,
                output_dir=output_dir,
                grouped_receivers=grouped_receivers,
                group_cfgs=group_cfgs,
                **common_kwargs,
            )
            if path is not None:
                log.info("  [row]  %s", path)

    log.info("Trace visualization complete.")


if __name__ == "__main__":
    main()
