from __future__ import annotations
from typing import Dict, Any, Optional, List, Hashable, Tuple
import copy
import torch
import numpy as np
import logging
import os
import re
import uuid
from collections import OrderedDict
from pathlib import Path
import matplotlib.pyplot as plt

from graph_signal_diffusion.tasks import TASK_REGISTRY
from graph_signal_diffusion.tasks.base import BaseTask
from graph_signal_diffusion.datasets.normalizer import Normalizer
from graph_signal_diffusion.datasets.wra.utils import (
    compute_ergodic_rates_batched,
    jains_fairness_index,
    clamp_power,
    compute_violation_rate,
)
from graph_signal_diffusion.datasets.wra.configs import dataset_name_to_alias
from graph_signal_diffusion.datasets.wra.channel_factory import (
    canonicalize_channel_cache_metadata,
    find_channel_cache_metadata_mismatches,
)

logger = logging.getLogger(__name__)


@TASK_REGISTRY.register("wireless_resource_allocation")
class WirelessResourceAllocationTask(BaseTask):
    """Task evaluator for wireless resource allocation.
    
    Evaluates generated power allocations by:
    1. Denormalizing from [-0.5, 0.5] to [0, P_max]
    2. Loading precomputed H_instantaneous timeslots from processed dataset files
    3. Computing ergodic rates with time-sharing across samples
    4. Comparing against ground truth (primal-dual) allocations
    
    Time-Sharing Evaluation:
    - Divides T channel realizations into num_time_shares = T/T_0 windows
    - For each evaluation batch, randomly samples power allocations and channel windows
    - Computes stochastic ergodic rate per slot, then averages across slots
    - Repeats for num_eval_batches to measure sensitivity to sampling
    
    Parameters
    ----------
    num_channel_realizations : int, optional
        Number of H_instantaneous timeslots required per network during evaluation
        (default: 200)
    ergodic_window_size : int, optional
        Window size T_0 for computing stochastic ergodic rate per slot (default: 10)
    num_eval_batches : int, optional
        Number of random samplings for statistical robustness (default: 10)
    normalizer : Normalizer, optional
        Normalizer for denormalizing power allocations (injected by trainer)
    dataset_info : dict, optional
        Dataset metadata including network seeds, system params (injected by trainer)
    **kwargs
        Additional arguments (valid_datasets, etc.)
    """

    # Opt-in marker: tells UnifiedEvaluator to produce a power-distribution
    # histogram and power_dist_w1 metric after evaluation.  Tasks that deal
    # with normalised power allocations (WRA) set this True; all others
    # (stock forecasting, etc.) leave it at the default False inherited from
    # BaseTask so that the evaluator never generates spurious power plots.
    is_power_allocation_task: bool = True

    def __init__(
        self,
        num_channel_realizations: int = 200,
        ergodic_window_size: int = 10,
        num_eval_batches: int = 10,
        eval_num_realizations: Optional[int] = None,
        eval_h_chunk_size: int = 32,
        eval_h_cache_budget_gb: float = 4.0,
        h_io_mode: str = "auto",
        h_sidecar_filename: str = "H_instantaneous_tmn.npy",
        reference_policy: str = "expert",
        normalizer: Optional[Normalizer] = None,
        dataset_info: Optional[Dict] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_channel_realizations = int(num_channel_realizations)
        self.eval_num_realizations = int(eval_num_realizations) if eval_num_realizations is not None else self.num_channel_realizations
        self.ergodic_window_size = ergodic_window_size
        self.num_time_shares = self.eval_num_realizations // ergodic_window_size
        if self.num_time_shares <= 0:
            raise ValueError(
                f"Invalid evaluation setup: eval_num_realizations={self.eval_num_realizations}, "
                f"ergodic_window_size={ergodic_window_size}."
            )
        self.eval_h_chunk_size = int(eval_h_chunk_size)
        if self.eval_h_chunk_size <= 0:
            raise ValueError(
                f"eval_h_chunk_size must be positive, got {eval_h_chunk_size}."
            )
        self.eval_h_cache_budget_gb = float(eval_h_cache_budget_gb)
        self.h_io_mode = str(h_io_mode).strip().lower()
        if self.h_io_mode not in {"auto", "legacy"}:
            raise ValueError(
                f"h_io_mode must be one of ['auto', 'legacy'], got {h_io_mode!r}."
            )
        sidecar_name = str(h_sidecar_filename).strip()
        self.h_sidecar_filename = sidecar_name if sidecar_name else "H_instantaneous_tmn.npy"
        self.num_eval_batches = num_eval_batches
        self.normalizer = normalizer
        self.dataset_info = dataset_info
        self.plot_style = self._as_dict(kwargs.get("plot_style", {}))
        self.valid_datasets = kwargs.get('valid_datasets', ['wra', 'wra_small_low_qos'])
        self.reference_policy = str(reference_policy).strip().lower()
        if self.reference_policy not in {"expert", "none"}:
            raise ValueError(
                "reference_policy must be one of {'expert', 'none'}, "
                f"got {reference_policy!r}."
            )

        # --- Channel-cache-on-demand state ---
        # Loaded channel cache files: {dataset_name: list[WirelessChannel]}
        self._channel_cache_files: Dict[str, list] = {}
        # Deep-copied channel objects for deterministic H regeneration:
        # {(dataset_name, network_id): WirelessChannel}  (frozen RNG state)
        self._channel_object_cache: Dict[Tuple[str, int], Any] = {}
        # LRU cache of generated H arrays: {(dataset_name, network_id): np.ndarray (m, n, T)}
        self._h_pool_cache: OrderedDict[Tuple[str, int], np.ndarray] = OrderedDict()
        self._h_pool_cache_bytes: int = 0

        logger.info(f"Initialized WirelessResourceAllocationTask")
        logger.info(f"  Channel realizations (base): {self.num_channel_realizations}")
        logger.info(f"  Channel realizations (eval): {self.eval_num_realizations}")
        logger.info(f"  Ergodic window size (T_0): {self.ergodic_window_size}")
        logger.info(f"  Num time shares: {self.num_time_shares}")
        logger.info(f"  Eval H transfer chunk size: {self.eval_h_chunk_size}")
        logger.info(f"  Eval H cache budget: {self.eval_h_cache_budget_gb:.1f} GB")
        logger.info(f"  Num eval batches: {self.num_eval_batches}")
        logger.info(f"  H I/O mode: {self.h_io_mode}")
        logger.info(f"  H sidecar filename: {self.h_sidecar_filename}")
        logger.info(f"  Reference policy: {self.reference_policy}")
        logger.info("  Evaluation channel source: precomputed H first, channel-cache fallback")
        logger.info(f"  Valid datasets: {self.valid_datasets}")
    
    def set_normalizer(self, normalizer: Normalizer) -> None:
        """Inject normalizer from dataset builder."""
        self.normalizer = normalizer
        logger.info(f"✅ Normalizer injected into WirelessResourceAllocationTask")
    
    def set_dataset_info(self, dataset_info: Dict) -> None:
        """Inject dataset metadata from dataset builder."""
        self.dataset_info = dataset_info
        logger.info(f"✅ Dataset info injected into WirelessResourceAllocationTask")
        logger.info(f"   Networks: {len(dataset_info.get('associations', {}))}")
        logger.info("   Requires precomputed H_instantaneous in processed dataset")

    def set_plot_style(self, plot_style: Optional[Dict[str, Any]]) -> None:
        """Inject centralized plotting style config from the compare/eval CLI."""
        self.plot_style = self._as_dict(plot_style)

    @staticmethod
    def _as_dict(value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return value
        try:
            from omegaconf import DictConfig
            if isinstance(value, DictConfig):
                from omegaconf import OmegaConf
                return OmegaConf.to_container(value, resolve=True)
        except ImportError:
            pass
        return {}

    @staticmethod
    def _style_get(style_cfg: Dict[str, Any], path: str, default: Any) -> Any:
        current: Any = style_cfg
        for key in path.split("."):
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]
        return current

    @staticmethod
    def _as_float(value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _as_int(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _as_tuple2(value: Any, default: tuple[float, float]) -> tuple[float, float]:
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return (
                WirelessResourceAllocationTask._as_float(value[0], default[0]),
                WirelessResourceAllocationTask._as_float(value[1], default[1]),
            )
        return default

    def _plot_style_section(self, name: str) -> Dict[str, Any]:
        plots_cfg = self._as_dict(self._as_dict(self.plot_style).get("plots", {}))
        return self._as_dict(plots_cfg.get(name, {}))

    @staticmethod
    def _normalize_network_id(network_id: Any) -> Any:
        """Best-effort normalization of network IDs to Python ints when possible."""
        try:
            return int(network_id)
        except (TypeError, ValueError):
            return network_id

    @staticmethod
    def _sanitize_filename_component(value: Any) -> str:
        """Convert arbitrary values into filesystem-safe filename components."""
        text = "unknown" if value is None else str(value).strip()
        if not text:
            return "unknown"
        for sep in (os.sep, os.altsep):
            if sep:
                text = text.replace(sep, "_")
        text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
        text = text.strip("._")
        return text or "unknown"

    @classmethod
    def _network_sort_key(cls, network_id: Any) -> tuple[int, Any]:
        """Stable ordering key for network IDs (ints first, then strings)."""
        network_id_norm = cls._normalize_network_id(network_id)
        if isinstance(network_id_norm, (int, np.integer)) and not isinstance(network_id_norm, bool):
            return (0, int(network_id_norm))
        return (1, str(network_id_norm))

    @classmethod
    def _select_network_keys_for_plotting(
        cls,
        network_keys: List[tuple[Any, Any]],
        style_cfg: Optional[Dict[str, Any]],
        plot_tag: str,
    ) -> List[tuple[Any, Any]]:
        """
        Select at most K network keys per dataset for plotting.

        Config:
          - max_networks_per_dataset: int (default=5)
            <=0 disables clamping.
        """
        style_cfg = cls._as_dict(style_cfg)
        max_networks_per_dataset = cls._as_int(
            cls._style_get(style_cfg, "max_networks_per_dataset", 5),
            5,
        )
        if max_networks_per_dataset <= 0 or len(network_keys) <= 1:
            return list(network_keys)

        from collections import defaultdict

        keys_by_dataset = defaultdict(list)
        for composite_key in network_keys:
            dataset_name = (
                composite_key[0]
                if isinstance(composite_key, tuple) and len(composite_key) == 2
                else None
            )
            keys_by_dataset[dataset_name].append(composite_key)

        selected_keys: List[tuple[Any, Any]] = []
        skipped_count = 0
        dataset_names = sorted(
            keys_by_dataset.keys(),
            key=lambda value: "" if value is None else str(value),
        )
        for dataset_name in dataset_names:
            dataset_keys = sorted(
                keys_by_dataset[dataset_name],
                key=lambda key: cls._network_sort_key(
                    key[1] if isinstance(key, tuple) and len(key) == 2 else None
                ),
            )
            selected_keys.extend(dataset_keys[:max_networks_per_dataset])
            skipped_count += max(0, len(dataset_keys) - max_networks_per_dataset)

        if skipped_count > 0:
            logger.info(
                "%s: plotting %d/%d networks (max_networks_per_dataset=%d).",
                plot_tag,
                len(selected_keys),
                len(network_keys),
                max_networks_per_dataset,
            )
        return selected_keys

    @classmethod
    def _resolve_metadata_key(
        cls,
        metadata_map: Dict[Hashable, Any],
        dataset_name_tag: Any,
        network_id: Any,
    ) -> Optional[Hashable]:
        """
        Resolve metadata map keys for both canonical and alias dataset tags.

        Supports:
          1) direct composite key: (dataset_name, network_id)
          2) legacy keying by network_id only
          3) alias-based composite-key matching
          4) unique network_id fallback when unambiguous
        """
        if not isinstance(metadata_map, dict) or len(metadata_map) == 0:
            return None

        net_id_norm = cls._normalize_network_id(network_id)
        ds_tag = None if dataset_name_tag is None else str(dataset_name_tag)

        direct_candidates = []
        if ds_tag is not None:
            direct_candidates.extend([
                (ds_tag, network_id),
                (ds_tag, net_id_norm),
            ])
        direct_candidates.extend([
            network_id,
            net_id_norm,
        ])

        for candidate in direct_candidates:
            if candidate in metadata_map:
                return candidate

        tuple_keys_for_network: List[Hashable] = []
        for key in metadata_map.keys():
            if not (isinstance(key, tuple) and len(key) == 2):
                continue
            _, key_network = key
            if cls._normalize_network_id(key_network) == net_id_norm:
                tuple_keys_for_network.append(key)

        if len(tuple_keys_for_network) == 1:
            return tuple_keys_for_network[0]

        if ds_tag is not None and len(tuple_keys_for_network) > 0:
            ds_alias = dataset_name_to_alias(ds_tag)
            alias_matches = [
                key for key in tuple_keys_for_network
                if dataset_name_to_alias(str(key[0])) == ds_alias
            ]
            if len(alias_matches) == 1:
                return alias_matches[0]
            if len(alias_matches) > 1:
                ds_base = ds_tag.split("/")[0]
                base_matches = [
                    key for key in alias_matches
                    if str(key[0]).split("/")[0] == ds_base
                ]
                if len(base_matches) == 1:
                    return base_matches[0]

        return None

    @classmethod
    def _parse_selector_graph_key(cls, graph_key: Any) -> tuple[Optional[str], Any]:
        """Parse trainer selector graph key into (dataset_name, network_id)."""
        if graph_key is None:
            return None, None
        key_str = str(graph_key)
        if "::" in key_str:
            dataset_name, network_tag = key_str.split("::", 1)
        else:
            dataset_name, network_tag = None, key_str

        if isinstance(network_tag, str) and network_tag.startswith("network_"):
            network_tag = network_tag[len("network_") :]
        return dataset_name, cls._normalize_network_id(network_tag)

    @classmethod
    def _resolve_selector_network_key(
        cls,
        selector_network_node_stats: Dict[str, Any],
        dataset_name: Any,
        network_id: Any,
    ) -> Optional[str]:
        """
        Resolve selector-network stats key for a given (dataset_name, network_id).

        Supports matching across canonical/alias dataset names and legacy key styles.
        """
        if not isinstance(selector_network_node_stats, dict) or len(selector_network_node_stats) == 0:
            return None

        ds_name = None if dataset_name is None else str(dataset_name)
        ds_alias = dataset_name_to_alias(ds_name) if ds_name is not None else None
        net_norm = cls._normalize_network_id(network_id)

        candidates: List[str] = []
        if ds_name is not None:
            candidates.append(f"{ds_name}::network_{net_norm}")
            candidates.append(f"{ds_name}::network_{network_id}")
        if ds_alias is not None and ds_alias != ds_name:
            candidates.append(f"{ds_alias}::network_{net_norm}")
            candidates.append(f"{ds_alias}::network_{network_id}")
        candidates.append(f"network_{net_norm}")
        candidates.append(str(net_norm))
        for candidate in candidates:
            if candidate in selector_network_node_stats:
                return candidate

        exact_ds_matches: List[str] = []
        alias_matches: List[str] = []
        net_only_matches: List[str] = []
        for graph_key in selector_network_node_stats.keys():
            ds_key, net_key = cls._parse_selector_graph_key(graph_key)
            if cls._normalize_network_id(net_key) != net_norm:
                continue
            if ds_key is None:
                net_only_matches.append(str(graph_key))
                continue
            if ds_name is not None and str(ds_key) == ds_name:
                exact_ds_matches.append(str(graph_key))
                continue
            if ds_alias is not None and dataset_name_to_alias(str(ds_key)) == ds_alias:
                alias_matches.append(str(graph_key))
                continue
            net_only_matches.append(str(graph_key))

        for bucket in (exact_ds_matches, alias_matches, net_only_matches):
            if len(bucket) > 0:
                return sorted(bucket)[0]
        return None

    @staticmethod
    def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
        """Compute Spearman rank correlation without scipy dependency."""
        if x.size < 2 or y.size < 2:
            return float("nan")
        x_rank = np.argsort(np.argsort(x))
        y_rank = np.argsort(np.argsort(y))
        if np.std(x_rank) < 1e-12 or np.std(y_rank) < 1e-12:
            return float("nan")
        return float(np.corrcoef(x_rank, y_rank)[0, 1])

    @staticmethod
    def _extract_deep_level_freq_and_rates(
        level_stats: Dict[Any, Any],
        records: List[Dict],
    ) -> Optional[tuple]:
        """Extract deepest-level selection frequency + per-node ergodic rates.

        Returns ``(deep_level, sel, real, gen)`` where ``sel`` is the deepest
        pooling level's per-node hard-mask selection frequency and ``real`` /
        ``gen`` are the per-node expert / generated ergodic rates (mean over the
        network's records), all trimmed to a common node count. Returns ``None``
        when the stats are missing, degenerate, or rates are unavailable.
        """
        deep_level = None
        for level_id in level_stats.keys():
            try:
                level_int = int(level_id)
            except (TypeError, ValueError):
                continue
            if deep_level is None or level_int > deep_level:
                deep_level = level_int
        if deep_level is None:
            return None

        deep_stats = level_stats.get(deep_level, level_stats.get(str(deep_level), {}))
        if not isinstance(deep_stats, dict):
            return None
        hard_freq = deep_stats.get("hard_mask_mean", None)
        if isinstance(hard_freq, torch.Tensor):
            hard_freq_np = hard_freq.detach().float().cpu().numpy()
        else:
            hard_freq_np = np.asarray(hard_freq, dtype=float) if hard_freq is not None else np.array([])
        if hard_freq_np.ndim != 1 or hard_freq_np.size <= 1:
            return None

        real_rates = [np.asarray(r["real_rates_final"], dtype=float) for r in records if "real_rates_final" in r]
        gen_rates = [np.asarray(r["gen_rates_final"], dtype=float) for r in records if "gen_rates_final" in r]
        if len(real_rates) == 0:
            return None
        real_rates_mean = np.mean(np.stack(real_rates, axis=0), axis=0)
        gen_rates_mean = (
            np.mean(np.stack(gen_rates, axis=0), axis=0)
            if len(gen_rates) > 0
            else np.full_like(real_rates_mean, np.nan)
        )

        num_nodes = int(min(hard_freq_np.size, real_rates_mean.size, gen_rates_mean.size))
        if num_nodes <= 1:
            return None
        return (
            deep_level,
            hard_freq_np[:num_nodes],
            real_rates_mean[:num_nodes],
            gen_rates_mean[:num_nodes],
        )

    def _visualize_selector_rate_correlation(
        self,
        evaluation_records: List[Dict],
        metadata: Dict[str, Any],
        viz_save_dir: Optional[str] = None,
    ) -> None:
        """
        Visualize deep-level selector frequency vs per-node ergodic rates.

        One figure is created per unique network in `evaluation_records` when
        selector per-network diagnostics are available in metadata.
        """
        if viz_save_dir is None:
            return
        style_cfg = self._plot_style_section("selector_rate_correlation")
        if not bool(style_cfg.get("enabled", True)):
            return
        selector_network_node_stats = metadata.get("selector_network_node_stats", None)
        if not isinstance(selector_network_node_stats, dict) or len(selector_network_node_stats) == 0:
            return

        fig_size = self._as_tuple2(
            self._style_get(style_cfg, "figure_size", [12.0, 8.0]),
            (12.0, 8.0),
        )
        height_ratios_raw = self._style_get(style_cfg, "height_ratios", [2.0, 1.25])
        if isinstance(height_ratios_raw, (list, tuple)) and len(height_ratios_raw) == 2:
            height_ratios = [
                self._as_float(height_ratios_raw[0], 2.0),
                self._as_float(height_ratios_raw[1], 1.25),
            ]
        else:
            height_ratios = [2.0, 1.25]

        sel_color = str(self._style_get(style_cfg, "selection_curve.color", "tab:blue"))
        sel_linewidth = self._as_float(
            self._style_get(style_cfg, "selection_curve.linewidth", 1.8), 1.8
        )
        sel_marker = str(self._style_get(style_cfg, "selection_curve.marker", "o"))
        sel_markersize = self._as_float(
            self._style_get(style_cfg, "selection_curve.markersize", 3), 3.0
        )
        sel_ylabel_color = str(
            self._style_get(style_cfg, "selection_curve.y_label_color", "tab:blue")
        )
        sel_y_limits_raw = self._style_get(style_cfg, "selection_curve.y_limits", [-0.02, 1.02])
        sel_y_limits = (
            self._as_float(sel_y_limits_raw[0], -0.02),
            self._as_float(sel_y_limits_raw[1], 1.02),
        ) if isinstance(sel_y_limits_raw, (list, tuple)) and len(sel_y_limits_raw) == 2 else (-0.02, 1.02)

        real_color = str(self._style_get(style_cfg, "real_curve.color", "tab:orange"))
        real_linewidth = self._as_float(self._style_get(style_cfg, "real_curve.linewidth", 1.6), 1.6)
        real_linestyle = str(self._style_get(style_cfg, "real_curve.linestyle", "--"))
        gen_color = str(self._style_get(style_cfg, "generated_curve.color", "tab:green"))
        gen_linewidth = self._as_float(self._style_get(style_cfg, "generated_curve.linewidth", 1.4), 1.4)
        gen_linestyle = str(self._style_get(style_cfg, "generated_curve.linestyle", ":"))

        scatter_color = str(self._style_get(style_cfg, "scatter.color", "tab:purple"))
        scatter_size = self._as_float(self._style_get(style_cfg, "scatter.size", 25), 25.0)
        scatter_alpha = self._as_float(self._style_get(style_cfg, "scatter.alpha", 0.75), 0.75)
        scatter_edgecolors = self._style_get(style_cfg, "scatter.edgecolors", "none")

        legend_loc = str(self._style_get(style_cfg, "legend.loc", "upper left"))
        legend_fontsize = self._as_float(self._style_get(style_cfg, "legend.fontsize", 9), 9.0)
        legend_framealpha = self._as_float(
            self._style_get(style_cfg, "legend.framealpha", 0.9), 0.9
        )

        corr_x = self._as_float(self._style_get(style_cfg, "corr_text.x", 0.99), 0.99)
        corr_y = self._as_float(self._style_get(style_cfg, "corr_text.y", 0.02), 0.02)
        corr_fontsize = self._as_float(self._style_get(style_cfg, "corr_text.fontsize", 9), 9.0)
        corr_ha = str(self._style_get(style_cfg, "corr_text.ha", "right"))
        corr_va = str(self._style_get(style_cfg, "corr_text.va", "bottom"))
        corr_bbox = self._as_dict(self._style_get(style_cfg, "corr_text.bbox", {}))
        corr_bbox_style = corr_bbox.get("boxstyle", "round")
        corr_bbox_facecolor = corr_bbox.get("facecolor", "white")
        corr_bbox_alpha = self._as_float(corr_bbox.get("alpha", 0.85), 0.85)
        corr_bbox_edgecolor = corr_bbox.get("edgecolor", "gray")

        suptitle_fontsize = self._as_float(self._style_get(style_cfg, "fonts.suptitle", 13), 13.0)
        suptitle_fontweight = str(
            self._style_get(style_cfg, "fonts.suptitle_weight", "bold")
        )
        grid_alpha = self._as_float(self._style_get(style_cfg, "grid.alpha", 0.25), 0.25)
        grid_linestyle = str(self._style_get(style_cfg, "grid.linestyle", "--"))
        dpi = max(1, self._as_int(self._style_get(style_cfg, "dpi", 150), 150))
        filename_template = str(
            self._style_get(
                style_cfg,
                "filename_template",
                "selector_rate_correlation_d{dataset_name}_n{network_id}.pdf",
            )
        )

        os.makedirs(viz_save_dir, exist_ok=True)
        from collections import defaultdict

        network_records = defaultdict(list)
        for record in evaluation_records:
            composite_key = (record.get("dataset_name"), record.get("network_id"))
            network_records[composite_key].append(record)

        selected_keys = self._select_network_keys_for_plotting(
            list(network_records.keys()),
            style_cfg=style_cfg,
            plot_tag="selector_rate_correlation",
        )

        for (dataset_name, network_id) in selected_keys:
            records = network_records[(dataset_name, network_id)]
            selector_key = self._resolve_selector_network_key(
                selector_network_node_stats=selector_network_node_stats,
                dataset_name=dataset_name,
                network_id=network_id,
            )
            if selector_key is None:
                logger.debug(
                    "Selector-rate correlation: no selector stats key found for (%s, %s).",
                    dataset_name,
                    network_id,
                )
                continue

            level_stats = selector_network_node_stats.get(selector_key, {})
            if not isinstance(level_stats, dict) or len(level_stats) == 0:
                continue

            extracted = self._extract_deep_level_freq_and_rates(level_stats, records)
            if extracted is None:
                continue
            deep_level, sel, real, gen = extracted
            num_nodes = int(sel.size)

            order = np.argsort(real)
            x = np.arange(num_nodes)
            sel_sorted = sel[order]
            real_sorted = real[order]
            gen_sorted = gen[order]

            pearson = float("nan")
            if np.std(real) > 1e-12 and np.std(sel) > 1e-12:
                pearson = float(np.corrcoef(real, sel)[0, 1])
            spearman = self._spearman_corr(real, sel)

            fig, axes = plt.subplots(
                2,
                1,
                figsize=fig_size,
                gridspec_kw={"height_ratios": height_ratios},
            )

            ax = axes[0]
            ax.plot(
                x,
                sel_sorted,
                color=sel_color,
                linewidth=sel_linewidth,
                marker=sel_marker,
                markersize=sel_markersize,
                label=f"Deep-level selection frequency (L{deep_level})",
            )
            ax.set_ylabel("Selection Frequency", color=sel_ylabel_color)
            ax.tick_params(axis="y", labelcolor=sel_ylabel_color)
            ax.set_ylim(sel_y_limits[0], sel_y_limits[1])
            ax.grid(True, alpha=grid_alpha, linestyle=grid_linestyle)

            ax_rates = ax.twinx()
            ax_rates.plot(
                x,
                real_sorted,
                color=real_color,
                linewidth=real_linewidth,
                linestyle=real_linestyle,
                label="Expert ergodic rate",
            )
            ax_rates.plot(
                x,
                gen_sorted,
                color=gen_color,
                linewidth=gen_linewidth,
                linestyle=gen_linestyle,
                label="Generated ergodic rate",
            )
            ax_rates.set_ylabel("Ergodic Rate (bits/s/Hz)")

            lines_1, labels_1 = ax.get_legend_handles_labels()
            lines_2, labels_2 = ax_rates.get_legend_handles_labels()
            ax.legend(
                lines_1 + lines_2,
                labels_1 + labels_2,
                loc=legend_loc,
                fontsize=legend_fontsize,
                framealpha=legend_framealpha,
            )

            corr_txt = (
                f"Pearson(real, sel)={pearson:.3f}\n"
                f"Spearman(real, sel)={spearman:.3f}\n"
                f"Selector key={selector_key}"
            )
            ax.text(
                corr_x,
                corr_y,
                corr_txt,
                transform=ax.transAxes,
                ha=corr_ha,
                va=corr_va,
                fontsize=corr_fontsize,
                bbox=dict(
                    boxstyle=corr_bbox_style,
                    facecolor=corr_bbox_facecolor,
                    alpha=corr_bbox_alpha,
                    edgecolor=corr_bbox_edgecolor,
                ),
            )
            ax.set_xlabel("Node Index (sorted by real ergodic rate)")

            ax_scatter = axes[1]
            ax_scatter.scatter(
                real,
                sel,
                s=scatter_size,
                alpha=scatter_alpha,
                color=scatter_color,
                edgecolors=scatter_edgecolors,
            )
            ax_scatter.set_xlabel("Real Ergodic Rate (bits/s/Hz)")
            ax_scatter.set_ylabel("Deep-Level Selection Frequency")
            ax_scatter.grid(True, alpha=grid_alpha, linestyle=grid_linestyle)

            fig.suptitle(
                f"Selector vs Ergodic Rates | Dataset={dataset_name}, Network={network_id}",
                fontsize=suptitle_fontsize,
                fontweight=suptitle_fontweight,
            )

            safe_dataset_name = self._sanitize_filename_component(dataset_name)
            safe_network_id = self._sanitize_filename_component(network_id)
            filename = filename_template.format(
                dataset_name=safe_dataset_name,
                network_id=safe_network_id,
            )
            save_path = os.path.join(viz_save_dir, filename)
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
            plt.close()
            logger.info("📊 Saved selector-rate correlation plot to %s", save_path)
    
    def _visualize_selector_rate_correlation_aggregate(
        self,
        evaluation_records: List[Dict],
        metadata: Dict[str, Any],
        viz_save_dir: Optional[str] = None,
    ) -> None:
        """Aggregate deep-level selector frequency vs ergodic-rate correlation.

        Unlike the per-network plots, this pools the per-node (deep-level
        selection frequency, expert ergodic rate) pairs across ALL networks —
        one figure per density (dataset_name), plus an optional global figure
        across densities. The pooled correlation is far more robust than the
        per-network one (hundreds–thousands of nodes instead of ~50–400).

        Controlled by the ``selector_rate_correlation.aggregate`` plot-style
        block; reuses the same data path (selector_network_node_stats +
        per-node real/gen rates) as the per-network visualization.
        """
        if viz_save_dir is None:
            return
        style_cfg = self._plot_style_section("selector_rate_correlation")
        if not bool(style_cfg.get("enabled", True)):
            return
        agg_cfg = self._as_dict(self._style_get(style_cfg, "aggregate", {}))
        if not bool(self._style_get(agg_cfg, "enabled", True)):
            return
        selector_network_node_stats = metadata.get("selector_network_node_stats", None)
        if not isinstance(selector_network_node_stats, dict) or len(selector_network_node_stats) == 0:
            return

        from collections import OrderedDict, defaultdict

        # evaluate_samples is called once per eval batch; a density's networks may
        # span batches. Accumulate this batch's per-node (sel, real, gen) pairs into
        # a per-pass buffer keyed by (dataset_name, network_id) and only render on
        # the final batch so per-density / global pools cover the whole split.
        batch_idx = self._as_int(metadata.get("eval_batch_idx", 0), 0)
        num_batches = max(1, self._as_int(metadata.get("eval_num_batches", 1), 1))
        if batch_idx <= 0 or not isinstance(getattr(self, "_selector_rate_agg_buffer", None), dict):
            self._selector_rate_agg_buffer = OrderedDict()
        buffer: "OrderedDict[tuple, tuple]" = self._selector_rate_agg_buffer

        network_records = defaultdict(list)
        for record in evaluation_records:
            composite_key = (record.get("dataset_name"), record.get("network_id"))
            network_records[composite_key].append(record)

        for (dataset_name, network_id), records in network_records.items():
            selector_key = self._resolve_selector_network_key(
                selector_network_node_stats=selector_network_node_stats,
                dataset_name=dataset_name,
                network_id=network_id,
            )
            if selector_key is None:
                continue
            level_stats = selector_network_node_stats.get(selector_key, {})
            if not isinstance(level_stats, dict) or len(level_stats) == 0:
                continue
            extracted = self._extract_deep_level_freq_and_rates(level_stats, records)
            if extracted is None:
                continue
            deep_level, sel, real, gen = extracted
            # Latest extraction wins (networks are not expected to repeat across batches).
            buffer[(dataset_name, network_id)] = (deep_level, sel, real, gen)

        # Wait until the final batch of the pass before rendering.
        if batch_idx < num_batches - 1:
            return

        # Pool the buffer by density (dataset_name) across ALL networks.
        per_dataset: "OrderedDict[Any, Dict[str, Any]]" = OrderedDict()
        for (dataset_name, network_id), (deep_level, sel, real, gen) in buffer.items():
            entry = per_dataset.setdefault(
                dataset_name,
                {"sel": [], "real": [], "gen": [], "networks": set(), "deep_levels": set()},
            )
            entry["sel"].append(sel)
            entry["real"].append(real)
            entry["gen"].append(gen)
            entry["networks"].add(network_id)
            entry["deep_levels"].add(deep_level)

        if not per_dataset:
            return

        num_bins = max(2, self._as_int(self._style_get(agg_cfg, "num_bins", 15), 15))
        min_networks = max(1, self._as_int(self._style_get(agg_cfg, "min_networks", 1), 1))
        density_tmpl = str(self._style_get(
            agg_cfg, "filename_template_density",
            "selector_rate_correlation_aggregate_d{dataset_name}.pdf",
        ))
        global_tmpl = str(self._style_get(
            agg_cfg, "filename_template_global",
            "selector_rate_correlation_aggregate_global.pdf",
        ))
        global_enabled = bool(self._style_get(agg_cfg, "global_enabled", True))

        os.makedirs(viz_save_dir, exist_ok=True)
        global_sel: List[np.ndarray] = []
        global_real: List[np.ndarray] = []
        global_gen: List[np.ndarray] = []
        global_networks = 0
        global_deep_levels: set = set()

        for dataset_name, entry in per_dataset.items():
            if len(entry["networks"]) < min_networks:
                continue
            sel = np.concatenate(entry["sel"])
            real = np.concatenate(entry["real"])
            gen = np.concatenate(entry["gen"])
            global_sel.append(sel)
            global_real.append(real)
            global_gen.append(gen)
            global_networks += len(entry["networks"])
            global_deep_levels |= entry["deep_levels"]
            self._render_aggregate_correlation_figure(
                sel=sel, real=real, gen=gen,
                style_cfg=style_cfg, num_bins=num_bins,
                deep_levels=entry["deep_levels"],
                n_networks=len(entry["networks"]),
                title=f"Selector vs Ergodic Rates (aggregate) | Dataset={dataset_name}",
                save_path=os.path.join(
                    viz_save_dir,
                    density_tmpl.format(dataset_name=self._sanitize_filename_component(dataset_name)),
                ),
            )

        if global_enabled and len(per_dataset) > 1 and global_sel:
            self._render_aggregate_correlation_figure(
                sel=np.concatenate(global_sel),
                real=np.concatenate(global_real),
                gen=np.concatenate(global_gen),
                style_cfg=style_cfg, num_bins=num_bins,
                deep_levels=global_deep_levels,
                n_networks=global_networks,
                n_datasets=len(per_dataset),
                title="Selector vs Ergodic Rates (aggregate) | ALL densities",
                save_path=os.path.join(viz_save_dir, global_tmpl),
            )

    def _render_aggregate_correlation_figure(
        self,
        *,
        sel: np.ndarray,
        real: np.ndarray,
        gen: np.ndarray,
        style_cfg: Dict[str, Any],
        num_bins: int,
        deep_levels: set,
        n_networks: int,
        title: str,
        save_path: str,
        n_datasets: Optional[int] = None,
    ) -> None:
        """Render one pooled scatter + binned-trend correlation figure."""
        fig_size = self._as_tuple2(self._style_get(style_cfg, "figure_size", [12.0, 8.0]), (12.0, 8.0))
        height_ratios_raw = self._style_get(style_cfg, "height_ratios", [2.0, 1.25])
        height_ratios = (
            [self._as_float(height_ratios_raw[0], 2.0), self._as_float(height_ratios_raw[1], 1.25)]
            if isinstance(height_ratios_raw, (list, tuple)) and len(height_ratios_raw) == 2
            else [2.0, 1.25]
        )
        scatter_color = str(self._style_get(style_cfg, "scatter.color", "tab:purple"))
        scatter_size = self._as_float(self._style_get(style_cfg, "scatter.size", 25), 25.0)
        scatter_alpha = self._as_float(self._style_get(style_cfg, "scatter.alpha", 0.75), 0.75)
        trend_color = str(self._style_get(style_cfg, "selection_curve.color", "tab:blue"))
        grid_alpha = self._as_float(self._style_get(style_cfg, "grid.alpha", 0.25), 0.25)
        grid_linestyle = str(self._style_get(style_cfg, "grid.linestyle", "--"))
        legend_loc = str(self._style_get(style_cfg, "legend.loc", "upper left"))
        legend_fontsize = self._as_float(self._style_get(style_cfg, "legend.fontsize", 9), 9.0)
        corr_x = self._as_float(self._style_get(style_cfg, "corr_text.x", 0.99), 0.99)
        corr_y = self._as_float(self._style_get(style_cfg, "corr_text.y", 0.02), 0.02)
        corr_fontsize = self._as_float(self._style_get(style_cfg, "corr_text.fontsize", 9), 9.0)
        suptitle_fontsize = self._as_float(self._style_get(style_cfg, "fonts.suptitle", 13), 13.0)
        suptitle_fontweight = str(self._style_get(style_cfg, "fonts.suptitle_weight", "bold"))
        dpi = max(1, self._as_int(self._style_get(style_cfg, "dpi", 150), 150))

        # Pooled correlation on finite pairs.
        finite = np.isfinite(real) & np.isfinite(sel)
        rr = real[finite]
        ss = sel[finite]
        pearson = float("nan")
        spearman = float("nan")
        if rr.size > 1 and np.std(rr) > 1e-12 and np.std(ss) > 1e-12:
            pearson = float(np.corrcoef(rr, ss)[0, 1])
            spearman = self._spearman_corr(rr, ss)

        # Binned mean ± std selection frequency across the rate range.
        centers: List[float] = []
        means: List[float] = []
        stds: List[float] = []
        if rr.size > 0 and np.ptp(rr) > 1e-12:
            edges = np.linspace(rr.min(), rr.max(), num_bins + 1)
            idx = np.clip(np.digitize(rr, edges[1:-1]), 0, num_bins - 1)
            for b in range(num_bins):
                mask = idx == b
                if np.any(mask):
                    centers.append(float(rr[mask].mean()))
                    means.append(float(ss[mask].mean()))
                    stds.append(float(ss[mask].std()))

        fig, axes = plt.subplots(2, 1, figsize=fig_size, gridspec_kw={"height_ratios": height_ratios})

        ax = axes[0]
        ax.scatter(real, sel, s=scatter_size, alpha=scatter_alpha, color=scatter_color, edgecolors="none")
        if centers:
            ax.plot(centers, means, color=trend_color, linewidth=2.0, marker="o", markersize=4,
                    label="Binned mean selection frequency")
            ax.legend(loc=legend_loc, fontsize=legend_fontsize, framealpha=0.9)
        ax.set_xlabel("Expert Ergodic Rate (bits/s/Hz)")
        ax.set_ylabel("Deep-Level Selection Frequency")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=grid_alpha, linestyle=grid_linestyle)
        scope = (f"N_datasets={n_datasets}, " if n_datasets is not None else "")
        corr_txt = (
            f"Pearson(real, sel)={pearson:.3f}\n"
            f"Spearman(real, sel)={spearman:.3f}\n"
            f"{scope}N_networks={n_networks}, N_nodes={int(sel.size)}\n"
            f"Deep level(s)=L{','.join(str(d) for d in sorted(deep_levels))}"
        )
        ax.text(
            corr_x, corr_y, corr_txt, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=corr_fontsize,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"),
        )

        ax_bin = axes[1]
        if centers:
            ax_bin.errorbar(centers, means, yerr=stds, fmt="o-", color=trend_color,
                            ecolor="gray", elinewidth=1.0, capsize=2, markersize=4)
        ax_bin.set_xlabel("Expert Ergodic Rate bin (bits/s/Hz)")
        ax_bin.set_ylabel("Mean Selection Frequency")
        ax_bin.set_ylim(-0.02, 1.02)
        ax_bin.grid(True, alpha=grid_alpha, linestyle=grid_linestyle)

        fig.suptitle(title, fontsize=suptitle_fontsize, fontweight=suptitle_fontweight)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        logger.info("📊 Saved aggregate selector-rate correlation plot to %s", save_path)

    def prepare_data(self, data_batch) -> Dict[str, Any]:
        """Extract wireless resource allocation data from batch.
        
        Parameters
        ----------
        data_batch : WirelessDataDiffusion batch
            Batch with attributes:
                - y: [B*N, 1, 1] normalized power allocations (PyG format)
                - network_id: [B] network identifiers
                - info: dict with per-graph metadata
                - batch: [B*N] batch assignment vector
                - edge_index: [2, E] graph structure
                - edge_weight: [E] interference weights
        
        Returns
        -------
        dict
            Dictionary with 'samples' [B, T, N, F] and 'metadata' for evaluation
        """
        logger.debug(f"Preparing WRA data from batch")
        
        # Extract normalized power allocations in PyG format
        y_pyg = data_batch.y  # [B*N, 1, 1]
        
        # Extract network IDs (one per graph in batch)
        network_ids = []
        dataset_names = []
        if hasattr(data_batch, 'network_id'):
            if isinstance(data_batch.network_id, torch.Tensor):
                network_ids = data_batch.network_id.cpu().tolist()
            else:
                network_ids = data_batch.network_id
        
        if hasattr(data_batch, 'dataset_name'):
            if isinstance(data_batch.dataset_name, list):
                dataset_names = data_batch.dataset_name
            elif isinstance(data_batch.dataset_name, torch.Tensor):
                # If batched as tensor, convert back
                dataset_names = [data_batch.info[i]['dataset_name'] for i in range(len(network_ids))]
            else:
                dataset_names = [data_batch.dataset_name] * len(network_ids)
        else:
            # Fallback: extract from info if available
            dataset_names = [data_batch.info[i]['dataset_name'] if hasattr(data_batch, 'info') else 'unknown' 
                           for i in range(len(network_ids))]
        
        # Get batch assignment for unbatching
        batch_vector = data_batch.batch if hasattr(data_batch, 'batch') else None
        
        # Unbatch PyG format to tensor format [B, T, N, F]
        if batch_vector is not None:
            batch_size = len(network_ids)
            # Count nodes per graph (assumes all graphs have same number of nodes)
            nodes_per_graph = (batch_vector == 0).sum().item()
            
            # Reshape from [B*N, 1, 1] to [B, 1, N, 1]
            samples = y_pyg.view(batch_size, nodes_per_graph, 1, 1).transpose(1, 2)  # [B, 1, N, 1]
        else:
            # Single graph case
            samples = y_pyg.unsqueeze(0).transpose(1, 2)  # [1, 1, N, 1]
            batch_size = 1
        
        logger.debug(f"  Reshaped samples from {y_pyg.shape} to {samples.shape}")
        
        # Extract metadata for each network in batch using composite keys
        associations_list = []
        h_ls_gains_list = []
        h_timeslot_dirs = []
        h_num_timesteps = []
        resolved_dataset_names = []
        dataset_aliases = []

        associations_map = self.dataset_info.get('associations', {})
        h_ls_gains_map = self.dataset_info.get('h_ls_gains', {})
        h_timeslot_dirs_map = self.dataset_info.get('h_timeslot_dirs', {})
        h_num_timesteps_map = self.dataset_info.get('h_num_timesteps', {})
        
        for dataset_name, net_id in zip(dataset_names, network_ids):
            assoc_key = self._resolve_metadata_key(associations_map, dataset_name, net_id)
            if assoc_key is None:
                net_id_norm = self._normalize_network_id(net_id)
                available_keys = [
                    str(key) for key in associations_map.keys()
                    if (
                        isinstance(key, tuple)
                        and len(key) == 2
                        and self._normalize_network_id(key[1]) == net_id_norm
                    ) or key == net_id_norm
                ]
                raise RuntimeError(
                    "Failed to resolve WRA dataset metadata key for "
                    f"dataset_name='{dataset_name}', network_id={net_id}. "
                    f"Available keys for this network: {available_keys[:8]}"
                )

            assoc = associations_map.get(assoc_key)
            associations_list.append(assoc)

            h_ls_key = self._resolve_metadata_key(h_ls_gains_map, dataset_name, net_id)
            if h_ls_key is None and assoc_key in h_ls_gains_map:
                h_ls_key = assoc_key
            h_ls_gains_list.append(h_ls_gains_map.get(h_ls_key) if h_ls_key is not None else None)

            h_key = self._resolve_metadata_key(h_timeslot_dirs_map, dataset_name, net_id)
            if h_key is None and assoc_key in h_timeslot_dirs_map:
                h_key = assoc_key
            h_dir = h_timeslot_dirs_map.get(h_key) if h_key is not None else None

            h_num_key = self._resolve_metadata_key(h_num_timesteps_map, dataset_name, net_id)
            if h_num_key is None and assoc_key in h_num_timesteps_map:
                h_num_key = assoc_key
            h_t = h_num_timesteps_map.get(h_num_key) if h_num_key is not None else None

            h_timeslot_dirs.append(h_dir)
            h_num_timesteps.append(h_t)

            if isinstance(assoc_key, tuple) and len(assoc_key) == 2:
                resolved_dataset_name = str(assoc_key[0])
            elif dataset_name is not None:
                resolved_dataset_name = str(dataset_name)
            else:
                resolved_dataset_name = "unknown"
            resolved_dataset_names.append(resolved_dataset_name)
            # Derive alias from the incoming tag (which may already be alias-like,
            # e.g. "wra-large-low-density") rather than the resolved canonical
            # name (e.g. "N_648_density_15.8_...") which produces wrong aliases.
            alias_source = str(dataset_name) if dataset_name is not None else resolved_dataset_name
            dataset_aliases.append(dataset_name_to_alias(alias_source))
        
        # Get system parameters (same for all networks in dataset)
        system_params = self.dataset_info.get('system_params', {})
        
        metadata = {
            'network_ids': network_ids,
            'dataset_names': resolved_dataset_names,
            'dataset_name_tags': [str(name) for name in dataset_names],
            'dataset_aliases': dataset_aliases,
            'associations': associations_list,
            'h_ls_gains': h_ls_gains_list,
            'h_timeslot_dirs': h_timeslot_dirs,
            'h_num_timesteps': h_num_timesteps,
            'system_params': system_params,
            'batch_vector': batch_vector,
            'batch_size': batch_size,
            'num_nodes': nodes_per_graph if batch_vector is not None else y_pyg.shape[0],
        }
        
        logger.debug(f"  Prepared {len(network_ids)} networks")
        logger.debug(f"  Samples shape: {samples.shape} [B, T, N, F]")
        
        return {
            'samples': samples,
            'metadata': metadata,
        }

    @staticmethod
    def _resolve_timeslot_files(
        *,
        timeslot_path: Path,
        num_available: Optional[int],
    ) -> tuple[list[Path], int]:
        """Resolve ordered per-timestep H files and reported available count."""
        available = int(num_available) if num_available is not None else 0
        if available <= 0:
            timeslot_files = sorted(timeslot_path.glob("timestep_*.pt"))
            if len(timeslot_files) == 0:
                timeslot_files = sorted(timeslot_path.glob("h_timeslot_*.pt"))
            available = len(timeslot_files)
            return timeslot_files, available

        timeslot_files = []
        for idx in range(available):
            new_path = timeslot_path / f"timestep_{idx}.pt"
            old_path = timeslot_path / f"h_timeslot_{idx}.pt"
            if new_path.exists() or not old_path.exists():
                timeslot_files.append(new_path)
            else:
                timeslot_files.append(old_path)
        return timeslot_files, available

    @staticmethod
    def _load_single_h_slot(
        slot_file: Path,
        *,
        dataset_name: str,
        network_id: int,
    ) -> np.ndarray:
        """Load one precomputed H slot and normalize to float32 numpy (m, n)."""
        if not slot_file.exists():
            raise FileNotFoundError(
                f"Missing precomputed H timeslot file for ({dataset_name}, {network_id}): {slot_file}"
            )
        slot_tensor = torch.load(slot_file, map_location='cpu')
        if isinstance(slot_tensor, dict):
            if 'H' in slot_tensor:
                slot_tensor = slot_tensor['H']
            elif 'H_inst' in slot_tensor:
                slot_tensor = slot_tensor['H_inst']
        if not isinstance(slot_tensor, torch.Tensor):
            slot_tensor = torch.as_tensor(slot_tensor)
        if slot_tensor.ndim != 2:
            raise ValueError(
                f"Malformed precomputed H slot {slot_file} with shape {tuple(slot_tensor.shape)}. "
                "Expected rank-2 [m, n] tensor."
            )
        return slot_tensor.cpu().numpy().astype(np.float32, copy=False)

    def _load_selected_h_from_timeslots(
        self,
        *,
        timeslot_files: list[Path],
        selected_indices: np.ndarray,
        dataset_name: str,
        network_id: int,
    ) -> np.ndarray:
        """Load selected timesteps from per-slot files and return (m, n, T)."""
        h_slots = [
            self._load_single_h_slot(
                timeslot_files[int(idx)],
                dataset_name=dataset_name,
                network_id=network_id,
            )
            for idx in selected_indices
        ]
        h_tmn = np.stack(h_slots, axis=0)  # (T, m, n)
        return np.transpose(h_tmn, (1, 2, 0))

    def _build_h_sidecar_from_timeslots(
        self,
        *,
        sidecar_path: Path,
        timeslot_files: list[Path],
        dataset_name: str,
        network_id: int,
    ) -> None:
        """Materialize contiguous sidecar (T, m, n) with atomic rename."""
        if len(timeslot_files) == 0:
            raise RuntimeError(
                f"Cannot build H sidecar for ({dataset_name}, {network_id}) with no timeslot files."
            )

        h_slots = [
            self._load_single_h_slot(
                slot_file,
                dataset_name=dataset_name,
                network_id=network_id,
            )
            for slot_file in timeslot_files
        ]
        h_tmn = np.stack(h_slots, axis=0).astype(np.float32, copy=False)  # (T, m, n)

        tmp_path = sidecar_path.parent / f"{sidecar_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        try:
            with open(tmp_path, "wb") as tmp_f:
                np.save(tmp_f, h_tmn, allow_pickle=False)
            os.replace(tmp_path, sidecar_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def _load_precomputed_h_samples(
        self,
        *,
        timeslot_dir: Optional[str],
        num_available: Optional[int],
        dataset_name: str,
        network_id: int,
    ) -> Optional[np.ndarray]:
        """Load precomputed H samples and return shape (m, n, T_selected).

        Returns None when precomputed H is unavailable (caller should fall
        back to channel-cache-on-demand generation).
        """
        if timeslot_dir is None:
            return None

        timeslot_path = Path(timeslot_dir)
        if not timeslot_path.exists():
            logger.info(
                "Precomputed H directory missing for (%s, %s): %s; will try channel-cache fallback.",
                dataset_name, network_id, timeslot_path,
            )
            return None

        timeslot_files, available = self._resolve_timeslot_files(
            timeslot_path=timeslot_path,
            num_available=num_available,
        )
        if available <= 0:
            logger.info(
                "No H timeslot files for (%s, %s) in %s; will try channel-cache fallback.",
                dataset_name, network_id, timeslot_path,
            )
            return None

        required = int(self.eval_num_realizations)
        if available < required:
            raise ValueError(
                f"Precomputed H for ({dataset_name}, {network_id}) has only {available} timeslots, "
                f"but evaluator requires {required}. Regenerate H_instantaneous with at least "
                f"{required} timesteps."
            )

        # Use contiguous temporal indices to preserve channel correlation.
        selected_indices = np.arange(required, dtype=np.int64)

        if self.h_io_mode == "auto":
            sidecar_path = timeslot_path / self.h_sidecar_filename
            try:
                if not sidecar_path.exists():
                    self._build_h_sidecar_from_timeslots(
                        sidecar_path=sidecar_path,
                        timeslot_files=timeslot_files,
                        dataset_name=dataset_name,
                        network_id=network_id,
                    )
                h_tmn_all = np.load(sidecar_path, mmap_mode="r")
                if h_tmn_all.ndim != 3:
                    raise ValueError(
                        f"Malformed H sidecar for ({dataset_name}, {network_id}) at {sidecar_path}: "
                        f"expected rank-3 (T, m, n), got shape {tuple(h_tmn_all.shape)}."
                    )
                if int(h_tmn_all.shape[0]) < available:
                    raise ValueError(
                        f"H sidecar for ({dataset_name}, {network_id}) has only "
                        f"{int(h_tmn_all.shape[0])} timesteps, expected at least {available}."
                    )
                selected = np.asarray(h_tmn_all[selected_indices], dtype=np.float32)
                return np.transpose(selected, (1, 2, 0))
            except Exception as exc:
                logger.warning(
                    "Failed sidecar H load for (%s, %s) at %s; falling back to legacy per-timeslot loading. "
                    "Reason: %s",
                    dataset_name,
                    network_id,
                    sidecar_path,
                    exc,
                )

        return self._load_selected_h_from_timeslots(
            timeslot_files=timeslot_files,
            selected_indices=selected_indices,
            dataset_name=dataset_name,
            network_id=network_id,
        )

    # ------------------------------------------------------------------
    # Channel-cache-on-demand: lazy H generation with LRU eviction
    # ------------------------------------------------------------------

    def _load_channel_cache_file(self, dataset_name: str) -> list:
        """Load and cache channel objects from a channel cache .pt file.

        Uses ``self.dataset_info['channel_cache_info']`` to locate the file.
        Raises RuntimeError with a clear remediation message on failure.
        """
        if dataset_name in self._channel_cache_files:
            return self._channel_cache_files[dataset_name]

        cache_info_map = (self.dataset_info or {}).get("channel_cache_info", {})
        cci = cache_info_map.get(dataset_name)
        if not isinstance(cci, dict) or not cci.get("channel_cache_file"):
            raise RuntimeError(
                f"No channel_cache_info for dataset '{dataset_name}' in dataset_info. "
                "Either regenerate the raw dataset with a recent build_diffusion_dataset.py "
                "or ensure precomputed H_instantaneous files exist in the processed dataset."
            )

        cache_file = Path(cci["channel_cache_file"])
        if not cache_file.exists():
            raise RuntimeError(
                f"Channel cache file not found: {cache_file}. "
                f"Expected for dataset '{dataset_name}'. "
                "Re-run PD training to regenerate the channel cache, or ensure "
                "precomputed H_instantaneous files exist in the processed dataset."
            )

        logger.info("Loading channel cache for dataset '%s' from %s", dataset_name, cache_file)
        try:
            cached = torch.load(cache_file, map_location="cpu", weights_only=False)
        except TypeError:
            # Older PyTorch versions don't support weights_only kwarg.
            cached = torch.load(cache_file, map_location="cpu")
        if not isinstance(cached, dict) or "channels" not in cached:
            raise RuntimeError(
                f"Invalid channel cache format at {cache_file}. "
                "Expected a dict with 'channels' key."
            )

        # Validate cache metadata against expected metadata from collection.
        expected_meta_raw = cci.get("expected_metadata")
        if isinstance(expected_meta_raw, dict) and expected_meta_raw:
            cached_meta_raw = cached.get("metadata")
            try:
                expected_meta = canonicalize_channel_cache_metadata(
                    {**expected_meta_raw, "channel_cache_key": cci.get("channel_cache_key", "")},
                    default_channel_version="v2",
                )
                cached_meta = canonicalize_channel_cache_metadata(
                    cached_meta_raw, default_channel_version="v2",
                )
                mismatches = find_channel_cache_metadata_mismatches(expected_meta, cached_meta)
                if mismatches:
                    mismatch_details = "; ".join(
                        f"{k}: expected={exp!r}, actual={act!r}"
                        for k, (exp, act) in mismatches.items()
                    )
                    raise RuntimeError(
                        f"Channel cache metadata mismatch for dataset '{dataset_name}' "
                        f"at {cache_file}: {mismatch_details}. "
                        "The cache file may be stale. Re-run PD training to regenerate it."
                    )
            except RuntimeError:
                raise
            except Exception as exc:
                logger.warning(
                    "Could not validate channel cache metadata for '%s': %s. "
                    "Proceeding without validation.",
                    dataset_name, exc,
                )

        channels = cached["channels"]
        self._channel_cache_files[dataset_name] = channels
        logger.info(
            "Loaded and validated %d channel objects from cache for dataset '%s'.",
            len(channels), dataset_name,
        )
        return channels

    def _resolve_channel_object(
        self,
        dataset_name: str,
        network_id: int,
    ) -> Any:
        """Get a deep-copied channel object with frozen RNG state for deterministic H.

        On first call for a (dataset_name, network_id), loads the channel cache,
        resolves the channel by seed match (preferred) or index fallback,
        deep-copies it, and stores the copy.  Subsequent calls return the same
        frozen copy.

        Resolution order:
          1. Match by ``network_seed`` (authoritative — works for expanded/
             profile IDs where network_id != cache index).
          2. Fall back to ``channels[network_id]`` only when seed info is
             unavailable (legacy datasets).
        """
        cache_key = (dataset_name, network_id)
        if cache_key in self._channel_object_cache:
            return self._channel_object_cache[cache_key]

        channels = self._load_channel_cache_file(dataset_name)
        net_id_int = self._normalize_network_id(network_id)

        # Preferred: match by network_seed.
        network_seeds_map = (self.dataset_info or {}).get("network_seeds", {})
        seed_key = self._resolve_metadata_key(network_seeds_map, dataset_name, network_id)
        target_seed = network_seeds_map.get(seed_key) if seed_key is not None else None

        if target_seed is not None:
            target_seed_int = int(target_seed)
            for ch in channels:
                ch_seed = getattr(ch, "seed", None)
                if ch_seed is not None and int(ch_seed) == target_seed_int:
                    frozen = copy.deepcopy(ch)
                    self._channel_object_cache[cache_key] = frozen
                    return frozen
            # Seed was known but no channel matched — fall through to index.
            logger.warning(
                "Seed-based channel lookup failed for (%s, %s): target_seed=%d "
                "not found among %d cached channels. Trying index fallback.",
                dataset_name, network_id, target_seed_int, len(channels),
            )

        # Fallback: index-based lookup (legacy datasets without seed metadata).
        if isinstance(net_id_int, int) and 0 <= net_id_int < len(channels):
            channel = channels[net_id_int]
            frozen = copy.deepcopy(channel)
            self._channel_object_cache[cache_key] = frozen
            return frozen

        raise RuntimeError(
            f"Cannot resolve channel object for ({dataset_name}, network_id={network_id}). "
            f"Cache has {len(channels)} channels. "
            f"Seed-based lookup (target_seed={target_seed}) found no match and "
            f"index {net_id_int} is out of range. "
            "Ensure the channel cache matches the dataset."
        )

    def _evict_h_pool_lru(self) -> None:
        """Evict oldest entries from ``_h_pool_cache`` until within budget."""
        budget_bytes = int(self.eval_h_cache_budget_gb * 1e9)
        while self._h_pool_cache_bytes > budget_bytes and self._h_pool_cache:
            evicted_key, evicted_arr = self._h_pool_cache.popitem(last=False)
            self._h_pool_cache_bytes -= evicted_arr.nbytes
            logger.debug(
                "Evicted H pool entry %s (%.1f MB); cache now %.1f MB.",
                evicted_key,
                evicted_arr.nbytes / 1e6,
                self._h_pool_cache_bytes / 1e6,
            )

    def _get_or_generate_h_from_cache(
        self,
        *,
        dataset_name: str,
        network_id: int,
    ) -> np.ndarray:
        """Return H array (m, n, T) from LRU pool, generating on cache miss.

        On miss: deep-copies the frozen channel object, calls
        ``sample_realization`` on the copy (so the frozen copy's RNG is never
        advanced), caches the result, and evicts LRU entries if over budget.
        """
        cache_key = (dataset_name, self._normalize_network_id(network_id))

        if cache_key in self._h_pool_cache:
            # Move to end (most recently used).
            self._h_pool_cache.move_to_end(cache_key)
            return self._h_pool_cache[cache_key]

        # Generate from a fresh copy of the frozen channel object.
        frozen_channel = self._resolve_channel_object(dataset_name, int(cache_key[1]))
        channel_copy = copy.deepcopy(frozen_channel)
        realization = channel_copy.sample_realization(
            num_timesteps=int(self.eval_num_realizations),
        )
        # channel returns H with shape (m, n, T); store as-is.
        h_mnt = np.asarray(realization["H"], dtype=np.float32)

        self._h_pool_cache[cache_key] = h_mnt
        self._h_pool_cache_bytes += h_mnt.nbytes
        logger.info(
            "Generated H from channel cache for (%s, %s): shape %s, %.1f MB. "
            "Pool total: %.1f MB / %.1f MB budget.",
            dataset_name, network_id, h_mnt.shape,
            h_mnt.nbytes / 1e6,
            self._h_pool_cache_bytes / 1e6,
            self.eval_h_cache_budget_gb * 1e3,
        )

        self._evict_h_pool_lru()
        return h_mnt

    def evaluate_samples(
        self, 
        generated_samples: torch.Tensor,
        real_samples: torch.Tensor,
        metadata: Dict[str, Any],
        viz_save_dir: Optional[str] = None,
        **kwargs
    ) -> Dict[str, float]:
        """Evaluate generated power allocations against ground truth.
        
        Parameters
        ----------
        generated_samples : torch.Tensor
            Generated power allocations from diffusion model, shape [B, T, N, F]
            Normalized to [-0.5, 0.5]
        real_samples : torch.Tensor
            Ground truth power allocations from primal-dual, shape [B, T, N, F]
            Normalized to [-0.5, 0.5]
        metadata : dict
            Metadata from prepare_data() containing network info
        viz_save_dir : Optional directory to save visualizations. By default, saves to Hydra output dir if active.
        
        Returns
        -------
        dict
            Dictionary of performance metrics
        """
        logger.info(f"Evaluating wireless resource allocation...")
        logger.debug(f"  Generated samples shape: {generated_samples.shape}")
        logger.debug(f"  Real samples shape: {real_samples.shape}")
        has_reference = self.reference_policy != "none"
        
        # Reshape from [B, T, N, F] to [B*N] for per-node processing
        B, T, N, F = generated_samples.shape
        generated_power_flat = generated_samples.squeeze(-1).squeeze(1)  # [B, N]
        real_power_flat = real_samples.squeeze(-1).squeeze(1)  # [B, N]
        
        # Denormalize power allocations
        # Note: WRA dataset uses hardcoded normalization, so we need to reverse it
        # y_normalized = (power / P_max) - 0.5, so power = (y_normalized + 0.5) * P_max
        system_params = metadata['system_params']
        P_max = system_params.get('P_max', 1.0)
        noise_var = system_params.get('noise_var', 1e-10)
        r_min = system_params.get('r_min', 0.5)
        r_min_per_dataset = self.dataset_info.get('r_min_per_dataset', {}) if self.dataset_info else {}

        generated_power_denorm = (generated_power_flat + 0.5) * P_max  # [B, N]
        real_power_denorm = (real_power_flat + 0.5) * P_max  # [B, N]
        
        # Ensure both tensors are on the same device
        if real_power_denorm.device != generated_power_denorm.device:
            real_power_denorm = real_power_denorm.to(generated_power_denorm.device)
        
        logger.debug(f"  Denormalized power range: [{generated_power_denorm.min():.4f}, {generated_power_denorm.max():.4f}]")
        
        # Clamp generated power to feasible range [0, P_max]
        generated_power_clamped = clamp_power(generated_power_denorm, P_max)  # [B, N]
        power_clamp_rate = (generated_power_denorm != generated_power_clamped).float().mean().item()
        
        # Get batch info
        batch_size = metadata['batch_size']
        num_nodes = metadata['num_nodes']
        
        # Group samples by unique (dataset_name, network_id) for efficient evaluation
        # With NetworkGroupedBatchSampler, samples are already grouped: [net_0, net_0, ..., net_1, net_1, ...]
        from collections import defaultdict
        network_to_samples = defaultdict(list)
        
        for batch_idx in range(batch_size):
            dataset_name = metadata['dataset_names'][batch_idx]
            network_id = metadata['network_ids'][batch_idx]
            composite_key = (dataset_name, network_id)
            network_to_samples[composite_key].append(batch_idx)
        
        unique_networks = list(network_to_samples.keys())
        logger.debug(f"Evaluating {len(unique_networks)} unique networks with "
                    f"{batch_size} total samples")
        
        # Initialize metrics collectors
        all_metrics = []
        
        # Store detailed evaluation records for visualization
        evaluation_records = []
        
        # Process each unique network using precomputed H_instantaneous only.
        r_min_fallback_datasets = set()
        for composite_key in unique_networks:
            dataset_name, network_id = composite_key
            network_r_min = r_min_per_dataset.get(dataset_name, r_min)
            if dataset_name not in r_min_per_dataset:
                r_min_fallback_datasets.add(dataset_name)
            sample_indices = network_to_samples[composite_key]
            associations = metadata['associations'][sample_indices[0]]
            h_timeslot_dir = None
            if 'h_timeslot_dirs' in metadata and len(metadata['h_timeslot_dirs']) > sample_indices[0]:
                h_timeslot_dir = metadata['h_timeslot_dirs'][sample_indices[0]]
            h_num_timesteps = None
            if 'h_num_timesteps' in metadata and len(metadata['h_num_timesteps']) > sample_indices[0]:
                h_num_timesteps = metadata['h_num_timesteps'][sample_indices[0]]
            
            if associations is None:
                raise RuntimeError(
                    f"Missing associations for ({dataset_name}, {network_id}) in evaluator metadata."
                )
            
            # Convert associations to torch tensor
            if isinstance(associations, np.ndarray):
                associations_torch = torch.from_numpy(associations).float()
            elif isinstance(associations, torch.Tensor):
                associations_torch = associations.float()
            else:
                associations_torch = torch.tensor(associations, dtype=torch.float32)
            
            # Get device from first sample
            device = generated_power_clamped[sample_indices[0]].device
            associations_torch = associations_torch.to(device)
            
            n_links = associations_torch.shape[0]
            H_samples = self._load_precomputed_h_samples(
                timeslot_dir=h_timeslot_dir,
                num_available=h_num_timesteps,
                dataset_name=dataset_name,
                network_id=network_id,
            )
            if H_samples is None:
                # Fallback: generate H on-demand from channel cache.
                H_samples = self._get_or_generate_h_from_cache(
                    dataset_name=dataset_name,
                    network_id=network_id,
                )

            if H_samples.shape[0] != n_links:
                raise ValueError(
                    f"H_instantaneous shape mismatch for ({dataset_name}, {network_id}): "
                    f"H has m={H_samples.shape[0]} but associations has m={n_links}."
                )
            if H_samples.shape[1] != int(associations_torch.shape[1]):
                raise ValueError(
                    f"H_instantaneous shape mismatch for ({dataset_name}, {network_id}): "
                    f"H has n={H_samples.shape[1]} but associations has n={int(associations_torch.shape[1])}."
                )
            # Keep H on CPU; _evaluate_time_shared streams only window slices to device.
            # Shape normalization: (m, n, T) -> (T, m, n)
            H_samples_torch = torch.from_numpy(H_samples).float().permute(2, 0, 1)
            
            # Collect all K samples for this network
            gen_power_network = generated_power_clamped[sample_indices]  # [K, N]
            real_tx_powers_all: Optional[torch.Tensor] = None
            if has_reference:
                real_power_network = real_power_denorm[sample_indices]  # [K, N]
                # Ensure same device
                if real_power_network.device != device:
                    real_power_network = real_power_network.to(device)
            
            # Convert all per-receiver powers to per-transmitter powers.
            # Vectorized form avoids Python loops over K samples.
            assoc_t = associations_torch.transpose(0, 1)  # (n, m)
            gen_tx_powers_all = gen_power_network @ assoc_t  # [K, m]
            if has_reference:
                real_tx_powers_all = real_power_network @ assoc_t  # [K, m]
            
            # Evaluate with multiple random samplings for statistical robustness
            eval_batch_metrics = []
            for eval_batch_idx in range(self.num_eval_batches):
                metrics, record = self._evaluate_time_shared(
                    gen_tx_powers_all,
                    real_tx_powers_all,
                    H_samples_torch,
                    associations_torch,
                    noise_var,
                    P_max,
                    network_r_min,
                    network_id,
                    dataset_name,
                    eval_batch_idx=eval_batch_idx,
                    include_reference=has_reference,
                )
                eval_batch_metrics.append(metrics)
                evaluation_records.append(record)
            
            # Average metrics across evaluation batches
            aggregated_network = self._aggregate_eval_batches(eval_batch_metrics)
            aggregated_network['_dataset_name'] = dataset_name
            all_metrics.append(aggregated_network)
            
            logger.debug(f"Network ({dataset_name}, {network_id}): Evaluated {len(sample_indices)} samples across "
                        f"{self.num_eval_batches} eval batches, avg sum rate gen={aggregated_network['sum_rate_generated']:.2f}")
        
        if r_min_fallback_datasets:
            logger.warning(
                "r_min_per_dataset lookup missed %d dataset(s): %s. "
                "Used scalar fallback r_min=%s for these.",
                len(r_min_fallback_datasets), sorted(r_min_fallback_datasets), r_min,
            )

        # Aggregate metrics across all networks
        if not all_metrics:
            raise RuntimeError("No networks were evaluated. Check dataset metadata and H_instantaneous files.")
        
        aggregated = self._aggregate_metrics(all_metrics)

        # Per-sub-dataset aggregation (only when multiple sub-datasets present)
        unique_ds_names = set(m['_dataset_name'] for m in all_metrics)
        if len(unique_ds_names) > 1:
            from collections import defaultdict
            ds_groups = defaultdict(list)
            for m in all_metrics:
                ds_groups[m['_dataset_name']].append(m)
            for ds_name, ds_metrics in ds_groups.items():
                ds_agg = self._aggregate_metrics(ds_metrics)
                # Use a short readable prefix: last path component (the hash key)
                short_name = ds_name.rsplit('/', 1)[-1] if '/' in ds_name else ds_name
                for k, v in ds_agg.items():
                    aggregated[f'subdataset/{short_name}/{k}'] = v

        # Add MSE for sanity check (using denormalized power)
        if has_reference:
            aggregated['mse_power'] = torch.nn.functional.mse_loss(
                generated_power_clamped,
                real_power_denorm,
            ).item()
        aggregated['power_clamp_rate'] = power_clamp_rate
        aggregated['num_networks_evaluated'] = len(all_metrics)

        logger.info(f"Evaluation complete: {len(all_metrics)} networks")
        if has_reference and 'sum_rate_real' in aggregated:
            logger.info(
                f"  Sum rate: gen={aggregated['sum_rate_generated']:.2f}, "
                f"real={aggregated['sum_rate_real']:.2f}, "
                f"gap={aggregated.get('sum_rate_gap_pct', 0.0):.1f}%"
            )
        else:
            logger.info(
                f"  Sum rate (generated only): {aggregated['sum_rate_generated']:.2f}"
            )
        
        if has_reference:
            # Visualize results
            self._visualize_results(
                generated_power_clamped,
                real_power_denorm,
                metadata,
                viz_save_dir
            )

            # Visualize CDF of ergodic rates
            unique_r_mins = sorted(set(rec['r_min'] for rec in evaluation_records))
            self._visualize_rate_cdf(
                evaluation_records,
                unique_r_mins,
                viz_save_dir
            )

            # Visualize time evolution of percentile rates per network
            self._visualize_per_slot_rate_evolution(
                evaluation_records,
                unique_r_mins,
                viz_save_dir
            )

            # Visualize correlation between deep-level selector frequency and
            # per-node ergodic rates (when trainer injects selector diagnostics).
            self._visualize_selector_rate_correlation(
                evaluation_records=evaluation_records,
                metadata=metadata,
                viz_save_dir=viz_save_dir,
            )

            # Aggregate (pooled across networks): per-density + optional global.
            self._visualize_selector_rate_correlation_aggregate(
                evaluation_records=evaluation_records,
                metadata=metadata,
                viz_save_dir=viz_save_dir,
            )
        
        # Store evaluation records for potential future visualization
        self.last_evaluation_records = evaluation_records
        
        return aggregated
    
    def _evaluate_time_shared(
        self,
        gen_tx_powers_all: torch.Tensor,
        real_tx_powers_all: Optional[torch.Tensor],
        H_samples: torch.Tensor,
        associations: torch.Tensor,
        noise_var: float,
        P_max: float,
        r_min: float,
        network_id: int,
        dataset_name: str,
        eval_batch_idx: int,
        include_reference: bool = True,
    ) -> tuple:
        """Evaluate with time-sharing across randomly sampled powers and channel windows.
        
        Parameters
        ----------
        gen_tx_powers_all : torch.Tensor
            All generated transmitter powers for this network, shape [K, m]
        real_tx_powers_all : Optional[torch.Tensor]
            All reference transmitter powers for this network, shape [K, m].
            Required when include_reference=True.
        H_samples : torch.Tensor
            Channel realizations, shape [T, m, n]
        associations : torch.Tensor
            TX-RX pairing matrix, shape [m, n]
        noise_var : float
            Noise variance
        P_max : float
            Maximum power constraint
        r_min : float
            Minimum rate constraint
        network_id : int
            Network identifier
        dataset_name : str
            Dataset name
        eval_batch_idx : int
            Realization index in [0, num_eval_batches-1].
        include_reference : bool
            Whether to compute reference/expert comparison metrics.
        
        Returns
        -------
        metrics : dict
            Performance metrics for this evaluation batch
        record : dict
            Detailed evaluation record for visualization
        """
        K = gen_tx_powers_all.shape[0]
        T = H_samples.shape[0]
        if include_reference and real_tx_powers_all is None:
            raise ValueError(
                "real_tx_powers_all must be provided when include_reference=True."
            )
        
        # Randomly sample power indices independently for generated and real samples
        # (with replacement from K samples). Independent draws give each distribution
        # its own unbiased ergodic rate estimate; there is no natural index correspondence
        # between generated and real samples that would motivate a paired design.
        gen_power_indices = torch.randint(0, K, (self.num_time_shares,), device=gen_tx_powers_all.device)
        real_power_indices = torch.randint(0, K, (self.num_time_shares,), device=gen_tx_powers_all.device)

        # Use sequential channel windows (not random sampling)
        # Each slot gets a consecutive window: [0:T_0], [T_0:2*T_0], ..., [(num_time_shares-1)*T_0:num_time_shares*T_0]
        window_starts = torch.arange(self.num_time_shares) * self.ergodic_window_size

        # Sample powers: [num_time_shares, m]
        gen_powers_sampled = gen_tx_powers_all[gen_power_indices]
        real_powers_sampled = (
            real_tx_powers_all[real_power_indices] if include_reference else None
        )
        
        # Compute ergodic rates slot-wise. H can reside on CPU; only slot windows
        # are transferred to device to avoid materializing full-T tensors on GPU.
        target_device = gen_tx_powers_all.device
        transfer_chunk_size = max(int(self.eval_h_chunk_size), int(self.ergodic_window_size))
        cached_chunk_start = -1
        cached_chunk_end = -1
        cached_chunk_gpu: Optional[torch.Tensor] = None

        gen_rates_per_slot = []
        real_rates_per_slot = []
        
        for slot_idx in range(self.num_time_shares):
            # Get power and channel for this slot
            gen_power_slot = gen_powers_sampled[slot_idx]  # [m]
            start = int(window_starts[slot_idx].item())
            end = start + self.ergodic_window_size

            if H_samples.device != target_device:
                if (
                    cached_chunk_gpu is None
                    or start < cached_chunk_start
                    or end > cached_chunk_end
                ):
                    cached_chunk_start = start
                    cached_chunk_end = min(T, start + transfer_chunk_size)
                    cached_chunk_gpu = H_samples[cached_chunk_start:cached_chunk_end].to(
                        target_device,
                        non_blocking=True,
                    )
                local_start = start - cached_chunk_start
                local_end = end - cached_chunk_start
                H_window = cached_chunk_gpu[local_start:local_end]
            else:
                H_window = H_samples[start:end]

            if include_reference:
                real_power_slot = real_powers_sampled[slot_idx]  # [m]
                # Shared pass for generated + reference powers over the same H_window.
                rates_slot = compute_ergodic_rates_batched(
                    powers=torch.stack([gen_power_slot, real_power_slot], dim=0),  # (2, m)
                    H_samples=H_window,
                    associations=associations,
                    noise_var=noise_var,
                )  # (2, n)
                gen_rate_slot = rates_slot[0]
                real_rate_slot = rates_slot[1]
            else:
                rates_slot = compute_ergodic_rates_batched(
                    powers=gen_power_slot.unsqueeze(0),  # (1, m)
                    H_samples=H_window,
                    associations=associations,
                    noise_var=noise_var,
                )  # (1, n)
                gen_rate_slot = rates_slot[0]
            
            gen_rates_per_slot.append(gen_rate_slot)
            if include_reference:
                real_rates_per_slot.append(real_rate_slot)
        
        # Stack to get [num_time_shares, n]
        gen_rates_per_slot = torch.stack(gen_rates_per_slot)
        if include_reference:
            real_rates_per_slot = torch.stack(real_rates_per_slot)
        
        # Average across time slots: [n]
        gen_rates = gen_rates_per_slot.mean(dim=0)
        real_rates = real_rates_per_slot.mean(dim=0) if include_reference else None
        
        # Compute metrics
        # Percentile quantiles to report for ergodic rates
        _rate_pct_qs = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10]
        _rate_pct_labels = ["0.1pct", "0.2pct", "0.5pct", "1pct", "2pct", "5pct", "10pct"]
        gen_rate_pcts = {
            label: torch.quantile(gen_rates, q).item()
            for q, label in zip(_rate_pct_qs, _rate_pct_labels)
        }
        power_violation_rate_generated = compute_violation_rate(
            gen_powers_sampled.flatten(), P_max, lower_bound=False
        )
        rate_violation_rate_generated = compute_violation_rate(
            gen_rates, r_min, lower_bound=True
        )
        rate_slack_generated = torch.clamp(r_min - gen_rates, min=0.0)
        metrics = {
            'sum_rate_generated': gen_rates.sum().item(),
            'min_rate_generated': gen_rates.min().item(),
            **{f'rate_{label}_generated': val for label, val in gen_rate_pcts.items()},
            'mean_rate_generated': gen_rates.mean().item(),
            'fairness_generated': jains_fairness_index(gen_rates),
            # New naming: percentage-scaled metrics for easier reporting/plotting.
            'power_violation_percentage_generated': power_violation_rate_generated * 100.0,
            'rate_violation_percentage_generated': rate_violation_rate_generated * 100.0,
            'rate_mean_violation_gap_pct_generated': rate_slack_generated.mean().item() * 100.0 / r_min if r_min > 0 else 0.0,
        }
        if include_reference:
            real_rate_pcts = {
                label: torch.quantile(real_rates, q).item()
                for q, label in zip(_rate_pct_qs, _rate_pct_labels)
            }
            rate_violation_rate_real = compute_violation_rate(
                real_rates, r_min, lower_bound=True
            )
            rate_slack_real = torch.clamp(r_min - real_rates, min=0.0)
            metrics.update(
                {
                    'sum_rate_real': real_rates.sum().item(),
                    'min_rate_real': real_rates.min().item(),
                    **{f'rate_{label}_real': val for label, val in real_rate_pcts.items()},
                    'mean_rate_real': real_rates.mean().item(),
                    'fairness_real': jains_fairness_index(real_rates),
                    'rate_violation_percentage_real': rate_violation_rate_real * 100.0,
                    'rate_mean_violation_gap_pct_real': rate_slack_real.mean().item() * 100.0 / r_min if r_min > 0 else 0.0,
                }
            )
        
        # Create detailed record for visualization
        record = {
            'network_id': network_id,
            'dataset_name': dataset_name,
            'eval_batch_idx': int(eval_batch_idx),
            'r_min': float(r_min),
            'power_indices': gen_power_indices.cpu().numpy(),
            'window_starts': window_starts.cpu().numpy(),
            'gen_powers_sampled': gen_powers_sampled.cpu().numpy(),  # [num_time_shares, m]
            'gen_rates_per_slot': gen_rates_per_slot.cpu().numpy(),  # [num_time_shares, n]
            'gen_rates_final': gen_rates.cpu().numpy(),  # [n]
            'power_violation_percentage_generated': metrics['power_violation_percentage_generated'],
            'rate_violation_percentage_generated': metrics['rate_violation_percentage_generated'],
            'rate_mean_violation_gap_pct_generated': metrics['rate_mean_violation_gap_pct_generated'],
        }
        if include_reference:
            record.update(
                {
                    'real_powers_sampled': real_powers_sampled.cpu().numpy(),  # [num_time_shares, m]
                    'real_rates_per_slot': real_rates_per_slot.cpu().numpy(),  # [num_time_shares, n]
                    'real_rates_final': real_rates.cpu().numpy(),  # [n]
                    'rate_violation_percentage_real': metrics['rate_violation_percentage_real'],
                    'rate_mean_violation_gap_pct_real': metrics['rate_mean_violation_gap_pct_real'],
                }
            )
        
        return metrics, record
    
    def _aggregate_eval_batches(self, eval_batch_metrics: List[Dict]) -> Dict[str, float]:
        """Aggregate metrics across multiple evaluation batches.
        
        Parameters
        ----------
        eval_batch_metrics : list of dict
            Metrics from each evaluation batch
        
        Returns
        -------
        dict
            Aggregated metrics (mean and std across batches)
        """
        aggregated = {}
        
        # Get all metric keys
        keys = eval_batch_metrics[0].keys()
        
        # Compute mean and std for each metric
        for key in keys:
            values = [m[key] for m in eval_batch_metrics]
            aggregated[key] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)
        
        return aggregated
    
    def _visualize_rate_cdf(
        self,
        evaluation_records: List[Dict],
        r_min,
        viz_save_dir: Optional[str] = None
    ) -> None:
        """Visualize CDF of ergodic rates for generated vs real samples.

        Parameters
        ----------
        evaluation_records : list of dict
            Evaluation records containing rate information for each network/batch
        r_min : float or list of float
            Minimum rate constraint(s) for visualization
        viz_save_dir : str, optional
            Directory to save visualization
        """
        if viz_save_dir is None:
            return
        # Normalize r_min to a sorted list of unique values
        if isinstance(r_min, (int, float)):
            r_min_values = [float(r_min)]
        else:
            r_min_values = sorted(set(float(v) for v in r_min))
        style_cfg = self._plot_style_section("task_rate_cdf")
        if not bool(style_cfg.get("enabled", True)):
            return

        fig_size = self._as_tuple2(
            self._style_get(style_cfg, "figure_size", [10.0, 7.0]),
            (10.0, 7.0),
        )
        real_color = str(self._style_get(style_cfg, "real_curve.color", "tab:orange"))
        real_linewidth = self._as_float(
            self._style_get(style_cfg, "real_curve.linewidth", 2.5), 2.5
        )
        real_alpha = self._as_float(self._style_get(style_cfg, "real_curve.alpha", 0.9), 0.9)
        real_linestyle = str(self._style_get(style_cfg, "real_curve.linestyle", "-"))
        gen_color = str(self._style_get(style_cfg, "generated_curve.color", "tab:blue"))
        gen_linewidth = self._as_float(
            self._style_get(style_cfg, "generated_curve.linewidth", 2.5), 2.5
        )
        gen_alpha = self._as_float(self._style_get(style_cfg, "generated_curve.alpha", 0.9), 0.9)
        gen_linestyle = str(self._style_get(style_cfg, "generated_curve.linestyle", "--"))
        rmin_color = str(self._style_get(style_cfg, "r_min_line.color", "red"))
        rmin_linestyle = str(self._style_get(style_cfg, "r_min_line.linestyle", ":"))
        rmin_linewidth = self._as_float(
            self._style_get(style_cfg, "r_min_line.linewidth", 2.0), 2.0
        )
        rmin_alpha = self._as_float(self._style_get(style_cfg, "r_min_line.alpha", 0.8), 0.8)
        stats_x = self._as_float(self._style_get(style_cfg, "stats_box.x", 0.98), 0.98)
        stats_y = self._as_float(self._style_get(style_cfg, "stats_box.y", 0.02), 0.02)
        stats_fontsize = self._as_float(
            self._style_get(style_cfg, "stats_box.fontsize", 10), 10.0
        )
        stats_va = str(
            self._style_get(style_cfg, "stats_box.verticalalignment", "bottom")
        )
        stats_ha = str(
            self._style_get(style_cfg, "stats_box.horizontalalignment", "right")
        )
        stats_bbox = self._as_dict(self._style_get(style_cfg, "stats_box.bbox", {}))
        stats_bbox_style = stats_bbox.get("boxstyle", "round")
        stats_bbox_facecolor = stats_bbox.get("facecolor", "white")
        stats_bbox_alpha = self._as_float(stats_bbox.get("alpha", 0.9), 0.9)
        stats_bbox_edgecolor = stats_bbox.get("edgecolor", "gray")
        x_label_fontsize = self._as_float(self._style_get(style_cfg, "fonts.x_label", 13), 13.0)
        y_label_fontsize = self._as_float(self._style_get(style_cfg, "fonts.y_label", 13), 13.0)
        title_fontsize = self._as_float(self._style_get(style_cfg, "fonts.title", 15), 15.0)
        title_weight = str(self._style_get(style_cfg, "fonts.title_weight", "bold"))
        label_weight = str(self._style_get(style_cfg, "fonts.label_weight", "bold"))
        legend_loc = str(self._style_get(style_cfg, "legend.loc", "upper left"))
        legend_fontsize = self._as_float(self._style_get(style_cfg, "fonts.legend", 11), 11.0)
        legend_framealpha = self._as_float(
            self._style_get(style_cfg, "legend.framealpha", 0.9), 0.9
        )
        grid_alpha = self._as_float(self._style_get(style_cfg, "grid.alpha", 0.3), 0.3)
        grid_linestyle = str(self._style_get(style_cfg, "grid.linestyle", "--"))
        y_limits_raw = self._style_get(style_cfg, "y_limits", [0.0, 1.05])
        if isinstance(y_limits_raw, (list, tuple)) and len(y_limits_raw) == 2:
            y_limits = (
                self._as_float(y_limits_raw[0], 0.0),
                self._as_float(y_limits_raw[1], 1.05),
            )
        else:
            y_limits = (0.0, 1.05)
        min_scale = self._as_float(self._style_get(style_cfg, "x_limits.min_scale", 0.8), 0.8)
        max_percentile = self._as_float(
            self._style_get(style_cfg, "x_limits.max_percentile", 95), 95.0
        )
        max_scale = self._as_float(self._style_get(style_cfg, "x_limits.max_scale", 1.1), 1.1)
        filename = str(self._style_get(style_cfg, "filename", "ergodic_rates_cdf.pdf"))
        dpi = max(1, self._as_int(self._style_get(style_cfg, "dpi", 150), 150))
        
        os.makedirs(viz_save_dir, exist_ok=True)
        
        def _render_cdf(records, r_mins, title_suffix, out_filename):
            """Render a single CDF plot from *records* and save to *out_filename*."""
            gen_rates_list, real_rates_list = [], []
            for rec in records:
                gen_rates_list.extend(rec['gen_rates_final'].tolist())
                real_rates_list.extend(rec['real_rates_final'].tolist())
            all_gen = np.array(gen_rates_list)
            all_real = np.array(real_rates_list)
            if len(all_gen) == 0 or len(all_real) == 0:
                return

            fig, ax = plt.subplots(1, 1, figsize=fig_size)

            gen_sorted = np.sort(all_gen)
            real_sorted = np.sort(all_real)
            gen_cdf = np.arange(1, len(gen_sorted) + 1) / len(gen_sorted)
            real_cdf = np.arange(1, len(real_sorted) + 1) / len(real_sorted)

            ax.plot(real_sorted, real_cdf, linewidth=real_linewidth, color=real_color,
                    label='Expert', alpha=real_alpha, linestyle=real_linestyle)
            ax.plot(gen_sorted, gen_cdf, linewidth=gen_linewidth, color=gen_color,
                    label='Generated', alpha=gen_alpha, linestyle=gen_linestyle)

            for i, r_val in enumerate(r_mins):
                ax.axvline(
                    r_val, color=rmin_color, linestyle=rmin_linestyle,
                    linewidth=rmin_linewidth, alpha=rmin_alpha,
                    label=f'$r_{{\\mathrm{{min}}}}$ = {r_val:.2f}' if len(r_mins) <= 5 or i == 0 else None,
                )

            gen_mean, real_mean = np.mean(all_gen), np.mean(all_real)
            gen_min_val, real_min_val = np.min(all_gen), np.min(all_real)
            gen_5th, real_5th = np.percentile(all_gen, 5), np.percentile(all_real, 5)

            stats_text = (
                f"Generated:\n  Mean: {gen_mean:.3f}\n  Min: {gen_min_val:.3f}\n"
                f"  5th %ile: {gen_5th:.3f}\n\n"
                f"Real:\n  Mean: {real_mean:.3f}\n  Min: {real_min_val:.3f}\n"
                f"  5th %ile: {real_5th:.3f}"
            )
            ax.text(stats_x, stats_y, stats_text, transform=ax.transAxes,
                    fontsize=stats_fontsize, verticalalignment=stats_va,
                    horizontalalignment=stats_ha,
                    bbox=dict(boxstyle=stats_bbox_style, facecolor=stats_bbox_facecolor,
                              alpha=stats_bbox_alpha, edgecolor=stats_bbox_edgecolor))

            ax.set_xlabel('Ergodic Rate (bits/s/Hz)', fontsize=x_label_fontsize, fontweight=label_weight)
            ax.set_ylabel('CDF', fontsize=y_label_fontsize, fontweight=label_weight)
            ax.set_title(f'CDF of Ergodic Rates: Generated vs Real{title_suffix}',
                         fontsize=title_fontsize, fontweight=title_weight)
            ax.legend(loc=legend_loc, fontsize=legend_fontsize, framealpha=legend_framealpha)
            ax.grid(True, alpha=grid_alpha, linestyle=grid_linestyle)
            ax.set_ylim([y_limits[0], y_limits[1]])

            x_lo = min(gen_min_val, real_min_val, min(r_mins)) * min_scale
            x_hi = max(np.percentile(all_gen, max_percentile),
                       np.percentile(all_real, max_percentile)) * max_scale
            ax.set_xlim([x_lo, x_hi])

            save_path = os.path.join(viz_save_dir, out_filename)
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            plt.close()
            logger.info(f"📊 Saved ergodic rate CDF to {save_path}")

        # --- Global CDF (all sub-datasets combined) ---
        _render_cdf(evaluation_records, r_min_values, "", filename)

        # --- Per-sub-dataset CDFs ---
        ds_groups: Dict[str, List[Dict]] = {}
        for record in evaluation_records:
            ds_groups.setdefault(record['dataset_name'], []).append(record)

        if len(ds_groups) > 1:
            stem, ext = os.path.splitext(filename)
            for ds_name, ds_records in sorted(ds_groups.items()):
                short_name = ds_name.rsplit('/', 1)[-1] if '/' in ds_name else ds_name
                ds_r_mins = sorted(set(float(rec['r_min']) for rec in ds_records))
                _render_cdf(
                    ds_records,
                    ds_r_mins,
                    f'\n({short_name})',
                    f"{stem}_{short_name}{ext}",
                )
    
    def _visualize_per_slot_rate_evolution(
        self,
        evaluation_records: List[Dict],
        r_min,
        viz_save_dir: Optional[str] = None
    ) -> None:
        """Visualize time evolution of percentile ergodic rates per network.

        For each unique network, creates a 3-subplot figure showing:
        - 5th percentile ergodic rates over time
        - 1st percentile ergodic rates over time
        - Network-wide minimum ergodic rates over time

        Ergodic rates at time slot t are computed as cumulative average of per-slot
        rates from slot 0 to t. Percentiles are computed across all receivers.

        For generated (diffusion) policy: plots mean across M evaluation batches
        with error bars showing standard deviation.

        Parameters
        ----------
        evaluation_records : list of dict
            Evaluation records containing per-slot rate information
        r_min : float or list of float
            Minimum rate constraint(s) for visualization (per-network r_min
            is read from the records; this parameter is unused but kept for
            backward compatibility)
        viz_save_dir : str, optional
            Directory to save visualization
        """
        if viz_save_dir is None:
            return
        style_cfg = self._plot_style_section("task_rate_evolution")
        if not bool(style_cfg.get("enabled", True)):
            return
        
        os.makedirs(viz_save_dir, exist_ok=True)
        
        # Group records by unique network (dataset_name, network_id)
        from collections import defaultdict
        network_records = defaultdict(list)
        
        for record in evaluation_records:
            composite_key = (record['dataset_name'], record['network_id'])
            network_records[composite_key].append(record)
        
        selected_network_keys = self._select_network_keys_for_plotting(
            list(network_records.keys()),
            style_cfg=style_cfg,
            plot_tag="task_rate_evolution",
        )

        # Create visualization for each selected network
        for composite_key in selected_network_keys:
            records = network_records[composite_key]
            dataset_name, network_id = composite_key
            # Read per-network r_min from records with consistency check
            record_r_mins = set(rec['r_min'] for rec in records)
            if len(record_r_mins) != 1:
                raise ValueError(
                    f"Inconsistent r_min in records for network "
                    f"({dataset_name}, {network_id}): {record_r_mins}"
                )
            network_r_min = records[0]['r_min']
            self._visualize_network_slot_evolution(
                records,
                dataset_name,
                network_id,
                network_r_min,
                viz_save_dir,
                style_cfg=style_cfg,
            )
    
    def _visualize_network_slot_evolution(
        self,
        records: List[Dict],
        dataset_name: str,
        network_id: int,
        r_min: float,
        viz_save_dir: str,
        style_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Create time evolution plot for a single network.
        
        Parameters
        ----------
        records : list of dict
            All evaluation records (M batches) for this network
        dataset_name : str
            Dataset name
        network_id : int
            Network ID
        r_min : float
            Minimum rate constraint
        viz_save_dir : str
            Directory to save visualization
        """
        style_cfg = self._as_dict(style_cfg)
        M = len(records)  # Number of evaluation batches
        num_time_shares = records[0]['gen_rates_per_slot'].shape[0]  # T_0
        n_receivers = records[0]['gen_rates_per_slot'].shape[1]  # n

        fig_size = self._as_tuple2(
            self._style_get(style_cfg, "figure_size", [12.0, 14.0]),
            (12.0, 14.0),
        )
        n_rows = max(1, self._as_int(self._style_get(style_cfg, "rows", 3), 3))
        dpi = max(1, self._as_int(self._style_get(style_cfg, "dpi", 150), 150))
        filename_template = str(
            self._style_get(
                style_cfg,
                "filename_template",
                "rate_evolution_d{dataset_name}_n{network_id}.pdf",
            )
        )

        real_color = str(self._style_get(style_cfg, "real_curve.color", "tab:orange"))
        real_linewidth = self._as_float(
            self._style_get(style_cfg, "real_curve.linewidth", 2.5), 2.5
        )
        real_marker = str(self._style_get(style_cfg, "real_curve.marker", "o"))
        real_markersize = self._as_float(
            self._style_get(style_cfg, "real_curve.markersize", 6), 6.0
        )
        real_alpha = self._as_float(self._style_get(style_cfg, "real_curve.alpha", 0.9), 0.9)

        gen_color = str(self._style_get(style_cfg, "generated_curve.color", "tab:blue"))
        gen_linewidth = self._as_float(
            self._style_get(style_cfg, "generated_curve.linewidth", 2.5), 2.5
        )
        gen_marker = str(self._style_get(style_cfg, "generated_curve.marker", "s"))
        gen_markersize = self._as_float(
            self._style_get(style_cfg, "generated_curve.markersize", 6), 6.0
        )
        gen_alpha = self._as_float(self._style_get(style_cfg, "generated_curve.alpha", 0.9), 0.9)
        gen_linestyle = str(self._style_get(style_cfg, "generated_curve.linestyle", "--"))
        gen_capsize = self._as_float(self._style_get(style_cfg, "generated_curve.capsize", 4), 4.0)
        gen_capthick = self._as_float(
            self._style_get(style_cfg, "generated_curve.capthick", 1.5), 1.5
        )

        rmin_color = str(self._style_get(style_cfg, "r_min_line.color", "red"))
        rmin_linestyle = str(self._style_get(style_cfg, "r_min_line.linestyle", ":"))
        rmin_linewidth = self._as_float(
            self._style_get(style_cfg, "r_min_line.linewidth", 2.0), 2.0
        )
        rmin_alpha = self._as_float(self._style_get(style_cfg, "r_min_line.alpha", 0.7), 0.7)

        x_label_fontsize = self._as_float(self._style_get(style_cfg, "fonts.x_label", 12), 12.0)
        y_label_fontsize = self._as_float(self._style_get(style_cfg, "fonts.y_label", 12), 12.0)
        subplot_title_fontsize = self._as_float(
            self._style_get(style_cfg, "fonts.subplot_title", 13), 13.0
        )
        legend_fontsize = self._as_float(self._style_get(style_cfg, "fonts.legend", 10), 10.0)
        suptitle_fontsize = self._as_float(self._style_get(style_cfg, "fonts.suptitle", 14), 14.0)
        title_weight = str(self._style_get(style_cfg, "fonts.title_weight", "bold"))
        label_weight = str(self._style_get(style_cfg, "fonts.label_weight", "bold"))
        legend_loc = str(self._style_get(style_cfg, "legend.loc", "best"))
        legend_framealpha = self._as_float(
            self._style_get(style_cfg, "legend.framealpha", 0.9), 0.9
        )
        grid_alpha = self._as_float(self._style_get(style_cfg, "grid.alpha", 0.3), 0.3)
        grid_linestyle = str(self._style_get(style_cfg, "grid.linestyle", "--"))
        x_left_margin = self._as_float(
            self._style_get(style_cfg, "x_limits.left_margin", 0.5), 0.5
        )
        x_right_margin = self._as_float(
            self._style_get(style_cfg, "x_limits.right_margin", 0.5), 0.5
        )
        suptitle_y = self._as_float(self._style_get(style_cfg, "suptitle.y", 0.995), 0.995)
        
        # Initialize storage for cumulative percentile rates
        # Shape: [M, num_time_shares] for each percentile/metric
        gen_5th_pct = np.zeros((M, num_time_shares))
        real_5th_pct = np.zeros((M, num_time_shares))
        gen_1st_pct = np.zeros((M, num_time_shares))
        real_1st_pct = np.zeros((M, num_time_shares))
        gen_min = np.zeros((M, num_time_shares))
        real_min = np.zeros((M, num_time_shares))
        
        # Compute cumulative percentiles for each evaluation batch
        for m, record in enumerate(records):
            gen_rates_per_slot = record['gen_rates_per_slot']  # [num_time_shares, n]
            real_rates_per_slot = record['real_rates_per_slot']  # [num_time_shares, n]
            
            # For each time slot t, compute cumulative average and percentile
            for t in range(num_time_shares):
                # Cumulative average from slot 0 to t (inclusive)
                gen_cumulative_avg = np.mean(gen_rates_per_slot[:t+1, :], axis=0)  # [n]
                real_cumulative_avg = np.mean(real_rates_per_slot[:t+1, :], axis=0)  # [n]
                
                # Compute percentiles across receivers
                gen_5th_pct[m, t] = np.percentile(gen_cumulative_avg, 5)
                real_5th_pct[m, t] = np.percentile(real_cumulative_avg, 5)
                gen_1st_pct[m, t] = np.percentile(gen_cumulative_avg, 1)
                real_1st_pct[m, t] = np.percentile(real_cumulative_avg, 1)
                gen_min[m, t] = np.min(gen_cumulative_avg)
                real_min[m, t] = np.min(real_cumulative_avg)
        
        # Compute mean and std across M evaluation batches
        gen_5th_mean = np.mean(gen_5th_pct, axis=0)
        gen_5th_std = np.std(gen_5th_pct, axis=0)
        real_5th_mean = np.mean(real_5th_pct, axis=0)
        
        gen_1st_mean = np.mean(gen_1st_pct, axis=0)
        gen_1st_std = np.std(gen_1st_pct, axis=0)
        real_1st_mean = np.mean(real_1st_pct, axis=0)
        
        gen_min_mean = np.mean(gen_min, axis=0)
        gen_min_std = np.std(gen_min, axis=0)
        real_min_mean = np.mean(real_min, axis=0)
        
        # Create figure with 3 subplots (one for each metric)
        fig, axes = plt.subplots(n_rows, 1, figsize=fig_size)
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = np.array([axes])
        if len(axes) < 3:
            logger.warning("task_rate_evolution rows=%d < 3; using 3.", len(axes))
            fig, axes = plt.subplots(3, 1, figsize=fig_size)
            axes = axes.flatten()
        
        time_slots = np.arange(1, num_time_shares + 1)  # 1-indexed for display
        
        # Subplot 1: 5th percentile
        ax = axes[0]
        ax.plot(
            time_slots,
            real_5th_mean,
            linewidth=real_linewidth,
            color=real_color,
            label='Expert',
            marker=real_marker,
            markersize=real_markersize,
            alpha=real_alpha,
        )
        ax.errorbar(time_slots, gen_5th_mean, yerr=gen_5th_std, 
                   linewidth=gen_linewidth, color=gen_color, label='Generated',
                   marker=gen_marker, markersize=gen_markersize, alpha=gen_alpha,
                   capsize=gen_capsize, capthick=gen_capthick,
                   linestyle=gen_linestyle)
        ax.axhline(
            r_min,
            color=rmin_color,
            linestyle=rmin_linestyle,
            linewidth=rmin_linewidth,
            label=f'$r_{{\\mathrm{{min}}}}$ = {r_min:.2f}',
            alpha=rmin_alpha,
        )
        ax.set_xlabel('Time Slot', fontsize=x_label_fontsize, fontweight=label_weight)
        ax.set_ylabel(
            '5th Percentile Rate (bits/s/Hz)',
            fontsize=y_label_fontsize,
            fontweight=label_weight,
        )
        ax.set_title(
            '5th Percentile Ergodic Rate Evolution',
            fontsize=subplot_title_fontsize,
            fontweight=title_weight,
        )
        ax.legend(loc=legend_loc, fontsize=legend_fontsize, framealpha=legend_framealpha)
        ax.grid(True, alpha=grid_alpha, linestyle=grid_linestyle)
        ax.set_xlim([x_left_margin, num_time_shares + x_right_margin])
        
        # Subplot 2: 1st percentile
        ax = axes[1]
        ax.plot(
            time_slots,
            real_1st_mean,
            linewidth=real_linewidth,
            color=real_color,
            label='Expert',
            marker=real_marker,
            markersize=real_markersize,
            alpha=real_alpha,
        )
        ax.errorbar(time_slots, gen_1st_mean, yerr=gen_1st_std,
                   linewidth=gen_linewidth, color=gen_color, label='Generated',
                   marker=gen_marker, markersize=gen_markersize, alpha=gen_alpha,
                   capsize=gen_capsize, capthick=gen_capthick,
                   linestyle=gen_linestyle)
        ax.axhline(
            r_min,
            color=rmin_color,
            linestyle=rmin_linestyle,
            linewidth=rmin_linewidth,
            label=f'$r_{{\\mathrm{{min}}}}$ = {r_min:.2f}',
            alpha=rmin_alpha,
        )
        ax.set_xlabel('Time Slot', fontsize=x_label_fontsize, fontweight=label_weight)
        ax.set_ylabel(
            '1st Percentile Rate (bits/s/Hz)',
            fontsize=y_label_fontsize,
            fontweight=label_weight,
        )
        ax.set_title(
            '1st Percentile Ergodic Rate Evolution',
            fontsize=subplot_title_fontsize,
            fontweight=title_weight,
        )
        ax.legend(loc=legend_loc, fontsize=legend_fontsize, framealpha=legend_framealpha)
        ax.grid(True, alpha=grid_alpha, linestyle=grid_linestyle)
        ax.set_xlim([x_left_margin, num_time_shares + x_right_margin])
        
        # Subplot 3: Network-wide minimum
        ax = axes[2]
        ax.plot(
            time_slots,
            real_min_mean,
            linewidth=real_linewidth,
            color=real_color,
            label='Expert',
            marker=real_marker,
            markersize=real_markersize,
            alpha=real_alpha,
        )
        ax.errorbar(time_slots, gen_min_mean, yerr=gen_min_std,
                   linewidth=gen_linewidth, color=gen_color, label='Generated',
                   marker=gen_marker, markersize=gen_markersize, alpha=gen_alpha,
                   capsize=gen_capsize, capthick=gen_capthick,
                   linestyle=gen_linestyle)
        ax.axhline(
            r_min,
            color=rmin_color,
            linestyle=rmin_linestyle,
            linewidth=rmin_linewidth,
            label=f'$r_{{\\mathrm{{min}}}}$ = {r_min:.2f}',
            alpha=rmin_alpha,
        )
        ax.set_xlabel('Time Slot', fontsize=x_label_fontsize, fontweight=label_weight)
        ax.set_ylabel(
            'Minimum Rate (bits/s/Hz)',
            fontsize=y_label_fontsize,
            fontweight=label_weight,
        )
        ax.set_title(
            'Network-Wide Minimum Ergodic Rate Evolution',
            fontsize=subplot_title_fontsize,
            fontweight=title_weight,
        )
        ax.legend(loc=legend_loc, fontsize=legend_fontsize, framealpha=legend_framealpha)
        ax.grid(True, alpha=grid_alpha, linestyle=grid_linestyle)
        ax.set_xlim([x_left_margin, num_time_shares + x_right_margin])
        
        # Overall title
        fig.suptitle(
            f'Percentile Rate Evolution: Dataset={dataset_name}, Network={network_id}\n'
            f'(Cumulative Average, M={M} eval batches, n={n_receivers} receivers)',
            fontsize=suptitle_fontsize,
            fontweight=title_weight,
            y=suptitle_y,
        )

        for idx in range(3, len(axes)):
            axes[idx].axis("off")
        
        # plt.tight_layout(rect=[0, 0, 1, 0.985])
        
        # Save figure
        safe_dataset_name = self._sanitize_filename_component(dataset_name)
        safe_network_id = self._sanitize_filename_component(network_id)
        filename = filename_template.format(
            dataset_name=safe_dataset_name,
            network_id=safe_network_id,
        )
        save_path = os.path.join(viz_save_dir, filename)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Saved per-slot rate evolution to {save_path}")
    
    def _visualize_results(
        self,
        generated_power: torch.Tensor,
        real_power: torch.Tensor,
        metadata: Dict[str, Any],
        viz_save_dir: Optional[str] = None
    ) -> None:
        """Visualize power allocation results.
        
        Creates scatter plots showing joint distributions of power allocations
        for pairs of consecutive receiver nodes (node_i vs node_i+1).
        Saves one plot per unique network with all samples from that network.
        
        Parameters
        ----------
        generated_power : torch.Tensor
            Generated power allocations [B, N] in [0, P_max] range
        real_power : torch.Tensor
            Real power allocations [B, N] in [0, P_max] range
        metadata : dict
            Metadata containing system parameters, network_ids, dataset_names
        viz_save_dir : str, optional
            Directory to save visualization
        """
        if viz_save_dir is None:
            return
        style_cfg = self._plot_style_section("task_power_scatter")
        if not bool(style_cfg.get("enabled", True)):
            return
        
        os.makedirs(viz_save_dir, exist_ok=True)
        
        # Extract parameters
        P_max = metadata['system_params'].get('P_max', 1.0)
        batch_size = metadata['batch_size']
        num_nodes = metadata['num_nodes']
        network_ids = metadata.get('network_ids', list(range(batch_size)))
        dataset_names = metadata.get('dataset_names', ['unknown'] * batch_size)
        
        # Group samples by unique (dataset_name, network_id)
        from collections import defaultdict
        network_to_samples = defaultdict(list)
        
        for batch_idx in range(batch_size):
            dataset_name = dataset_names[batch_idx]
            network_id = network_ids[batch_idx]
            composite_key = (dataset_name, network_id)
            network_to_samples[composite_key].append(batch_idx)
        
        selected_network_keys = self._select_network_keys_for_plotting(
            list(network_to_samples.keys()),
            style_cfg=style_cfg,
            plot_tag="task_power_scatter",
        )

        # Create one plot per selected network with all its samples
        for composite_key in selected_network_keys:
            sample_indices = network_to_samples[composite_key]
            dataset_name, network_id = composite_key
            
            # Collect all samples for this network
            gen_power_network = generated_power[sample_indices]  # [K, N]
            real_power_network = real_power[sample_indices]  # [K, N]
            
            self._visualize_network_samples(
                gen_power_network,
                real_power_network,
                P_max,
                num_nodes,
                network_id,
                dataset_name,
                len(sample_indices),
                viz_save_dir,
                style_cfg=style_cfg,
            )
    
    def _visualize_network_samples(
        self,
        gen_power_network: torch.Tensor,
        real_power_network: torch.Tensor,
        P_max: float,
        num_nodes: int,
        network_id: int,
        dataset_name: str,
        num_samples: int,
        viz_save_dir: str,
        style_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Visualize power allocation for all samples from a single network.
        
        Parameters
        ----------
        gen_power_network : torch.Tensor
            Generated power allocations [K, N] in [0, P_max] range (K samples from this network)
        real_power_network : torch.Tensor
            Real power allocations [K, N] in [0, P_max] range
        P_max : float
            Maximum power constraint
        num_nodes : int
            Number of nodes in the network
        network_id : int
            Network ID for filename
        dataset_name : str
            Dataset name for filename
        num_samples : int
            Number of samples for this network
        viz_save_dir : str
            Directory to save visualization
        """
        style_cfg = self._as_dict(style_cfg)
        max_pairs_cfg = max(
            1, self._as_int(self._style_get(style_cfg, "max_pairs", 10), 10)
        )
        random_pair_sampling = bool(
            self._style_get(style_cfg, "random_pair_sampling", True)
        )
        subplot_grid_raw = self._style_get(style_cfg, "subplot_grid", [2, 5])
        if isinstance(subplot_grid_raw, (list, tuple)) and len(subplot_grid_raw) == 2:
            n_rows = max(1, self._as_int(subplot_grid_raw[0], 2))
            n_cols = max(1, self._as_int(subplot_grid_raw[1], 5))
        else:
            n_rows, n_cols = 2, 5
        n_axes_total = n_rows * n_cols
        fig_size = self._as_tuple2(
            self._style_get(style_cfg, "figure_size", [20.0, 8.0]),
            (20.0, 8.0),
        )
        dpi = max(1, self._as_int(self._style_get(style_cfg, "dpi", 150), 150))
        filename_template = str(
            self._style_get(
                style_cfg,
                "filename_template",
                "power_allocation_scatter_d{dataset_name}_n{network_id}.pdf",
            )
        )

        gen_color = str(self._style_get(style_cfg, "generated_scatter.color", "tab:blue"))
        gen_edgecolor = str(
            self._style_get(style_cfg, "generated_scatter.edgecolor", "navy")
        )
        gen_alpha = self._as_float(
            self._style_get(style_cfg, "generated_scatter.alpha", 0.6), 0.6
        )
        gen_size = self._as_float(self._style_get(style_cfg, "generated_scatter.size", 50), 50.0)
        gen_linewidth = self._as_float(
            self._style_get(style_cfg, "generated_scatter.linewidth", 0.5), 0.5
        )
        real_color = str(self._style_get(style_cfg, "real_scatter.color", "tab:orange"))
        real_edgecolor = str(self._style_get(style_cfg, "real_scatter.edgecolor", "darkred"))
        real_alpha = self._as_float(
            self._style_get(style_cfg, "real_scatter.alpha", 0.6), 0.6
        )
        real_size = self._as_float(self._style_get(style_cfg, "real_scatter.size", 50), 50.0)
        real_linewidth = self._as_float(
            self._style_get(style_cfg, "real_scatter.linewidth", 0.5), 0.5
        )

        bounds_color = str(self._style_get(style_cfg, "bounds_line.color", "gray"))
        bounds_linestyle = str(self._style_get(style_cfg, "bounds_line.linestyle", ":"))
        bounds_alpha = self._as_float(
            self._style_get(style_cfg, "bounds_line.alpha", 0.5), 0.5
        )
        bounds_linewidth = self._as_float(
            self._style_get(style_cfg, "bounds_line.linewidth", 1.5), 1.5
        )

        limits_raw = self._style_get(style_cfg, "limits", [-0.05, 1.05])
        if isinstance(limits_raw, (list, tuple)) and len(limits_raw) == 2:
            axis_limits = (
                self._as_float(limits_raw[0], -0.05),
                self._as_float(limits_raw[1], 1.05),
            )
        else:
            axis_limits = (-0.05, 1.05)

        tick_format = str(self._style_get(style_cfg, "tick_format", "%.2f"))
        x_label_fontsize = self._as_float(self._style_get(style_cfg, "fonts.x_label", 10), 10.0)
        y_label_fontsize = self._as_float(self._style_get(style_cfg, "fonts.y_label", 10), 10.0)
        subplot_title_fontsize = self._as_float(
            self._style_get(style_cfg, "fonts.subplot_title", 11), 11.0
        )
        title_weight = str(self._style_get(style_cfg, "fonts.title_weight", "bold"))
        legend_fontsize = self._as_float(self._style_get(style_cfg, "fonts.legend", 8), 8.0)
        corr_text_fontsize = self._as_float(
            self._style_get(style_cfg, "fonts.correlation_text", 9), 9.0
        )
        suptitle_fontsize = self._as_float(self._style_get(style_cfg, "fonts.suptitle", 14), 14.0)

        legend_loc = str(self._style_get(style_cfg, "legend.loc", "upper right"))
        first_subplot_only = bool(
            self._style_get(style_cfg, "legend.first_subplot_only", True)
        )

        corr_text_x = self._as_float(self._style_get(style_cfg, "correlation_text.x", 0.05), 0.05)
        corr_text_y = self._as_float(self._style_get(style_cfg, "correlation_text.y", 0.95), 0.95)
        corr_text_va = str(
            self._style_get(style_cfg, "correlation_text.verticalalignment", "top")
        )
        corr_text_ha = str(
            self._style_get(style_cfg, "correlation_text.horizontalalignment", "left")
        )
        corr_bbox = self._as_dict(self._style_get(style_cfg, "correlation_text.bbox", {}))
        corr_bbox_style = corr_bbox.get("boxstyle", "round")
        corr_bbox_facecolor = corr_bbox.get("facecolor", "white")
        corr_bbox_alpha = self._as_float(corr_bbox.get("alpha", 0.8), 0.8)

        grid_alpha = self._as_float(self._style_get(style_cfg, "grid.alpha", 0.3), 0.3)
        grid_linestyle = str(self._style_get(style_cfg, "grid.linestyle", "--"))
        suptitle_y = self._as_float(self._style_get(style_cfg, "suptitle.y", 0.995), 0.995)

        # Select consecutive node pairs (node_i, node_i+1)
        max_pairs = min(max_pairs_cfg, num_nodes - 1, n_axes_total)
        
        if max_pairs == 0:
            logger.warning(f"Not enough nodes for pair visualization (network {network_id})")
            return
        
        # Node-pair selection
        if random_pair_sampling:
            node_starts = np.random.choice(num_nodes - 1, size=max_pairs, replace=False)
        else:
            node_starts = np.arange(max_pairs)
        
        # Create figure grid
        fig, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)
        axes = np.atleast_1d(axes).flatten()
        
        # Convert to numpy for plotting
        gen_power_np = gen_power_network.cpu().numpy()  # [K, N]
        real_power_np = real_power_network.cpu().numpy()  # [K, N]
        
        # Plot each node pair
        for idx, (ax, node_i) in enumerate(zip(axes[:max_pairs], node_starts)):
            node_j = node_i + 1
            
            # Extract power values for node pair across all samples
            gen_power_i = gen_power_np[:, node_i]  # [K]
            gen_power_j = gen_power_np[:, node_j]  # [K]
            real_power_i = real_power_np[:, node_i]  # [K]
            real_power_j = real_power_np[:, node_j]  # [K]
            
            # Normalize by P_max for axis range [-0.05, 1.05]
            gen_power_i_norm = gen_power_i / P_max
            gen_power_j_norm = gen_power_j / P_max
            real_power_i_norm = real_power_i / P_max
            real_power_j_norm = real_power_j / P_max
            
            # Scatter plot: generated samples (blue) and real samples (red/orange)
            ax.scatter(
                gen_power_i_norm,
                gen_power_j_norm,
                alpha=gen_alpha,
                s=gen_size,
                c=gen_color,
                edgecolors=gen_edgecolor,
                linewidth=gen_linewidth,
                label='Generated',
            )
            ax.scatter(
                real_power_i_norm,
                real_power_j_norm,
                alpha=real_alpha,
                s=real_size,
                c=real_color,
                edgecolors=real_edgecolor,
                linewidth=real_linewidth,
                label='Expert',
            )
            
            # Add feasibility bounds (normalized: 0 and 1)
            ax.axhline(y=0, color=bounds_color, linestyle=bounds_linestyle, alpha=bounds_alpha, linewidth=bounds_linewidth)
            ax.axhline(y=1, color=bounds_color, linestyle=bounds_linestyle, alpha=bounds_alpha, linewidth=bounds_linewidth)
            ax.axvline(x=0, color=bounds_color, linestyle=bounds_linestyle, alpha=bounds_alpha, linewidth=bounds_linewidth)
            ax.axvline(x=1, color=bounds_color, linestyle=bounds_linestyle, alpha=bounds_alpha, linewidth=bounds_linewidth)
            
            # Formatting
            ax.set_xlabel(
                f'Power at Node {node_i} / $P_{{\\mathrm{{max}}}}$',
                fontsize=x_label_fontsize,
            )
            ax.set_ylabel(
                f'Power at Node {node_j} / $P_{{\\mathrm{{max}}}}$',
                fontsize=y_label_fontsize,
            )
            ax.set_title(
                f'Node Pair ({node_i}, {node_j})',
                fontsize=subplot_title_fontsize,
                fontweight=title_weight,
            )
            ax.set_xlim([axis_limits[0], axis_limits[1]])
            ax.set_ylim([axis_limits[0], axis_limits[1]])
            
            # Format ticks to 2 decimals
            from matplotlib.ticker import FormatStrFormatter
            ax.xaxis.set_major_formatter(FormatStrFormatter(tick_format))
            ax.yaxis.set_major_formatter(FormatStrFormatter(tick_format))
            
            ax.grid(alpha=grid_alpha, linestyle=grid_linestyle)
            if idx == 0 or not first_subplot_only:
                ax.legend(loc=legend_loc, fontsize=legend_fontsize)
            
            # Add correlation coefficients for both generated and real
            if num_samples > 1:  # Only compute correlation if we have multiple samples
                corr_gen = np.corrcoef(gen_power_i, gen_power_j)[0, 1]
                corr_real = np.corrcoef(real_power_i, real_power_j)[0, 1]
                text = r'$\rho_{\mathrm{gen}}$' + f' = {corr_gen:.3f}\n' + r'$\rho_{\mathrm{real}}$' + f' = {corr_real:.3f}'
                ax.text(
                    corr_text_x,
                    corr_text_y,
                    text,
                    transform=ax.transAxes,
                    fontsize=corr_text_fontsize,
                    verticalalignment=corr_text_va,
                    horizontalalignment=corr_text_ha,
                    bbox=dict(
                        boxstyle=corr_bbox_style,
                        facecolor=corr_bbox_facecolor,
                        alpha=corr_bbox_alpha,
                    ),
                )
        
        # Hide unused subplots
        for idx in range(max_pairs, n_axes_total):
            axes[idx].axis('off')
        
        plt.suptitle(
            f'Power Allocations: Dataset={dataset_name}, Network={network_id} ({num_samples} samples)',
            fontsize=suptitle_fontsize,
            fontweight=title_weight,
            y=suptitle_y,
        )
        
        # Save figure with dataset_name and network_id in filename (no batch index)
        safe_dataset_name = self._sanitize_filename_component(dataset_name)
        safe_network_id = self._sanitize_filename_component(network_id)
        filename = filename_template.format(
            dataset_name=safe_dataset_name,
            network_id=safe_network_id,
        )
        save_path = os.path.join(viz_save_dir, filename)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Saved power allocation visualization to {save_path}")
    
    def _aggregate_metrics(self, all_metrics: List[Dict]) -> Dict[str, float]:
        """Aggregate metrics across multiple networks."""
        aggregated = {}
        
        # Get all metric keys (skip internal metadata keys prefixed with '_')
        keys = [k for k in all_metrics[0].keys() if not k.startswith('_')]

        # Average each metric
        for key in keys:
            values = [m[key] for m in all_metrics if key in m]
            aggregated[key] = np.mean(values)
        
        # Compute performance gaps
        has_reference = (
            'sum_rate_real' in aggregated and
            'min_rate_real' in aggregated and
            'rate_1pct_real' in aggregated and
            'rate_5pct_real' in aggregated
        )
        if has_reference:
            if aggregated['sum_rate_real'] > 0:
                aggregated['sum_rate_gap_pct'] = (
                    (aggregated['sum_rate_real'] - aggregated['sum_rate_generated']) /
                    aggregated['sum_rate_real'] * 100
                )
            else:
                aggregated['sum_rate_gap_pct'] = 0.0
            
            if aggregated['min_rate_real'] > 0:
                aggregated['min_rate_gap_pct'] = (
                    (aggregated['min_rate_real'] - aggregated['min_rate_generated']) /
                    aggregated['min_rate_real'] * 100
                )
            else:
                aggregated['min_rate_gap_pct'] = 0.0

            if aggregated.get('rate_1pct_real', 0.0) > 0:
                aggregated['rate_1pct_gap_pct'] = (
                    (aggregated['rate_1pct_real'] - aggregated['rate_1pct_generated']) /
                    aggregated['rate_1pct_real'] * 100
                )
            else:
                aggregated['rate_1pct_gap_pct'] = 0.0

            if aggregated.get('rate_5pct_real', 0.0) > 0:
                aggregated['rate_5pct_gap_pct'] = (
                    (aggregated['rate_5pct_real'] - aggregated['rate_5pct_generated']) /
                    aggregated['rate_5pct_real'] * 100
                )
            else:
                aggregated['rate_5pct_gap_pct'] = 0.0
        
        # Feasibility rate (both power and rate constraints satisfied).
        # Prefer legacy fractional keys if present; otherwise derive from percentages.
        power_violation_rate = aggregated.get('power_violation_rate_generated')
        if power_violation_rate is None:
            power_violation_pct = aggregated.get('power_violation_percentage_generated')
            power_violation_rate = 0.0 if power_violation_pct is None else float(power_violation_pct) / 100.0

        rate_violation_rate = aggregated.get('rate_violation_rate_generated')
        if rate_violation_rate is None:
            rate_violation_pct = aggregated.get('rate_violation_percentage_generated')
            rate_violation_rate = 0.0 if rate_violation_pct is None else float(rate_violation_pct) / 100.0

        aggregated['feasibility_rate'] = (1.0 - power_violation_rate) * (1.0 - rate_violation_rate)
        
        return aggregated
