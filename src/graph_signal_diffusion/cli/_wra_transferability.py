"""Helpers for WRA transferability evaluation CLI."""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import os.path as osp
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

logger = logging.getLogger(__name__)

from graph_signal_diffusion.datasets.wra.channel_factory import (
    build_wra_channel_cache_metadata,
    canonicalize_channel_cache_metadata,
    create_wireless_channel,
    find_channel_cache_metadata_mismatches,
    normalize_channel_version,
    sanitize_channel_params,
)
from graph_signal_diffusion.datasets.wra.dataset import WirelessDataDiffusion
from graph_signal_diffusion.datasets.wra.sampler import NetworkGroupedBatchSampler
from graph_signal_diffusion.utils.graph_builder import build_interference_graph


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True, throw_on_missing=False)
    return value if isinstance(value, dict) else {}


def _select(cfg: Any, key_path: str, default: Any = None) -> Any:
    if OmegaConf.is_config(cfg):
        return OmegaConf.select(cfg, key_path, default=default)

    current = cfg
    for key in key_path.split("."):
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def _to_float(value: Any, *, key: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected numeric value for '{key}', got {value!r}.") from exc


def as_dict(value: Any) -> dict[str, Any]:
    """Public wrapper for dictionary coercion."""
    return _as_dict(value)


def to_float(value: Any, *, key: str) -> float:
    """Public wrapper for float coercion with contextual error key."""
    return _to_float(value, key=key)


def _to_int(value: Any, *, key: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected integer value for '{key}', got {value!r}.") from exc


def resolve_path_from_runtime_cwd(path_value: str, runtime_cwd: Path) -> Path:
    """Resolve a user path relative to Hydra runtime cwd."""
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = runtime_cwd / path
    return path.resolve()


def find_experiment_config_for_checkpoint(checkpoint_path: Path) -> Path:
    """Walk up from a checkpoint until `.hydra/config.yaml` is found."""
    current = checkpoint_path.resolve().parent
    while True:
        candidate = current / ".hydra" / "config.yaml"
        if candidate.exists():
            return candidate
        if current.parent == current:
            break
        current = current.parent
    raise FileNotFoundError(
        "Could not find '.hydra/config.yaml' by walking up from checkpoint: "
        f"{checkpoint_path}"
    )


def load_checkpoint_hydra_config(checkpoint_path: Path) -> DictConfig:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    cfg_path = find_experiment_config_for_checkpoint(checkpoint_path)
    return OmegaConf.load(cfg_path)


def resolve_inherited_r_min(source_checkpoint_cfg: Any) -> tuple[float, str]:
    """Resolve inherited r_min from source checkpoint config.

    Resolution order:
      1) training.constraint_profile.scalar.value
      2) training.r_min
    """
    scalar_from_profile = _select(
        source_checkpoint_cfg,
        "training.constraint_profile.scalar.value",
        default=None,
    )
    if scalar_from_profile is not None:
        try:
            return _to_float(
                scalar_from_profile,
                key="training.constraint_profile.scalar.value",
            ), "training.constraint_profile.scalar.value"
        except ValueError:
            # Unresolved interpolation strings (e.g. "${training.r_min}")
            # are treated as unavailable and fall through to the legacy key.
            pass

    fallback_r_min = _select(source_checkpoint_cfg, "training.r_min", default=None)
    if fallback_r_min is not None:
        try:
            return _to_float(fallback_r_min, key="training.r_min"), "training.r_min"
        except ValueError:
            pass

    raise ValueError(
        "Could not resolve inherited target r_min from source checkpoint config. "
        "Missing both 'training.constraint_profile.scalar.value' and 'training.r_min'."
    )


def resolve_target_r_min(
    target_r_min: Any,
    source_checkpoint_cfg: Any,
) -> tuple[float, str]:
    """Resolve target r_min (`float` or string `inherit`)."""
    if isinstance(target_r_min, str) and target_r_min.strip().lower() == "inherit":
        return resolve_inherited_r_min(source_checkpoint_cfg)

    return _to_float(target_r_min, key="target.r_min"), "target.r_min"


def resolve_model_cond_channels(
    source_checkpoint_cfg: Any,
    source_override: Optional[Any] = None,
) -> int:
    """Resolve source model conditioning width for transfer dataset features."""
    if source_override is not None:
        value = _to_int(source_override, key="source.model_cond_channels")
    else:
        value = _to_int(
            _select(source_checkpoint_cfg, "dataset.model_cond_channels", default=None),
            key="dataset.model_cond_channels",
        )

    if value not in (2, 3):
        raise ValueError(
            "WRA transferability currently supports model_cond_channels in {2, 3}; "
            f"got {value}."
        )
    return value


def resolve_source_system_params(source_checkpoint_cfg: Any) -> dict[str, Optional[float]]:
    """Extract source system params from source checkpoint config."""
    out: dict[str, Optional[float]] = {}
    for key in ("P_max_dBm", "bandwidth_Hz", "noise_psd_dBm_Hz"):
        raw = _select(source_checkpoint_cfg, f"system.{key}", default=None)
        out[key] = None if raw is None else _to_float(raw, key=f"system.{key}")
    return out


def enforce_pmax_compatibility(
    *,
    source_p_max_dBm: Optional[float],
    target_p_max_dBm: float,
    allow_mismatch: bool,
) -> None:
    """Guard against source/target P_max mismatch by default."""
    if source_p_max_dBm is None:
        return

    if math.isclose(
        float(source_p_max_dBm),
        float(target_p_max_dBm),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        return

    if allow_mismatch:
        return

    raise ValueError(
        "Source/target P_max_dBm mismatch. "
        f"source={float(source_p_max_dBm):g}, target={float(target_p_max_dBm):g}. "
        "Set experiment.allow_pmax_mismatch=true to override."
    )


def inject_diffusion_checkpoint_path(cfg: DictConfig, checkpoint_path: str) -> None:
    """Wire source checkpoint into `baselines.diffusion.checkpoint_path`."""
    if "baselines" not in cfg or "diffusion" not in cfg.baselines:
        raise ValueError("Missing `baselines.diffusion` section in transferability config.")
    OmegaConf.update(
        cfg,
        "baselines.diffusion.checkpoint_path",
        str(checkpoint_path),
        force_add=True,
    )


def validate_transfer_baselines(
    baselines_to_compare: Sequence[str],
    reference_policy: str,
) -> None:
    """Reject AP in expert-free transfer mode."""
    baseline_names = {str(name).strip().lower() for name in baselines_to_compare}
    if str(reference_policy).strip().lower() == "none" and "ap" in baseline_names:
        raise ValueError(
            "AveragePowerBaseline ('ap') requires expert target power allocations and "
            "is not supported in expert-free transfer mode (transfer.reference_policy=none)."
        )


def select_eval_network_ids(cache_num_networks: int, eval_num_networks: int) -> list[int]:
    """Deterministic first-N network selection with bounds checking."""
    if cache_num_networks <= 0:
        raise ValueError(f"cache_num_networks must be positive, got {cache_num_networks}.")
    if eval_num_networks <= 0:
        raise ValueError(f"eval_num_networks must be positive, got {eval_num_networks}.")
    if eval_num_networks > cache_num_networks:
        raise ValueError(
            "eval_num_networks must be <= cache_num_networks, got "
            f"{eval_num_networks} > {cache_num_networks}."
        )
    return list(range(int(eval_num_networks)))


def load_wra_scenario_config(conf_root: Path, scenario_name: str) -> DictConfig:
    """Load `wra_generation/scenario/defaults + scenario_name` as one config."""
    defaults_path = conf_root / "wra_generation" / "scenario" / "defaults.yaml"
    scenario_path = conf_root / "wra_generation" / "scenario" / f"{scenario_name}.yaml"

    if not defaults_path.exists():
        raise FileNotFoundError(f"Scenario defaults config not found: {defaults_path}")
    if not scenario_path.exists():
        raise FileNotFoundError(f"Scenario config not found: {scenario_path}")

    defaults_cfg = OmegaConf.load(defaults_path)
    scenario_cfg = OmegaConf.load(scenario_path)
    if isinstance(scenario_cfg, DictConfig) and "defaults" in scenario_cfg:
        scenario_cfg = OmegaConf.create(
            {k: v for k, v in scenario_cfg.items() if k != "defaults"}
        )
    return OmegaConf.merge(defaults_cfg, scenario_cfg)


def extract_graph_build_cfg(source_checkpoint_cfg: Any) -> dict[str, Any]:
    """Read graph-construction knobs from source checkpoint dataset config."""
    return {
        "top_k": _to_int(_select(source_checkpoint_cfg, "dataset.top_k", default=10), key="dataset.top_k"),
        "min_degree": _to_int(
            _select(source_checkpoint_cfg, "dataset.min_degree", default=1),
            key="dataset.min_degree",
        ),
        "edge_normalize_method": str(
            _select(
                source_checkpoint_cfg,
                "dataset.edge_normalize_method",
                default="log1p",
            )
        ),
        "self_loop_weight_scale": _to_float(
            _select(
                source_checkpoint_cfg,
                "dataset.self_loop_weight_scale",
                default=1.0,
            ),
            key="dataset.self_loop_weight_scale",
        ),
        "spectral_normalize_edge_weights": bool(
            _select(
                source_checkpoint_cfg,
                "dataset.spectral_normalize_edge_weights",
                default=False,
            )
        ),
    }


def build_target_system_params(
    scenario_cfg: Any,
    *,
    resolved_r_min: float,
) -> dict[str, float]:
    """Convert scenario system config to task-level `system_params`."""
    p_max_dBm = _to_float(_select(scenario_cfg, "system.P_max_dBm", default=None), key="system.P_max_dBm")
    bandwidth_Hz = _to_float(
        _select(scenario_cfg, "system.bandwidth_Hz", default=None),
        key="system.bandwidth_Hz",
    )
    noise_psd_dBm_Hz = _to_float(
        _select(scenario_cfg, "system.noise_psd_dBm_Hz", default=None),
        key="system.noise_psd_dBm_Hz",
    )

    p_max = 10.0 ** ((p_max_dBm - 30.0) / 10.0)
    noise_power_dBm = noise_psd_dBm_Hz + 10.0 * math.log10(bandwidth_Hz)
    noise_var = 10.0 ** ((noise_power_dBm - 30.0) / 10.0)

    return {
        "P_max": float(p_max),
        "P_max_dBm": float(p_max_dBm),
        "BW": float(bandwidth_Hz),
        "noise_var": float(noise_var),
        "noise_psd_dBm_Hz": float(noise_psd_dBm_Hz),
        "r_min": float(resolved_r_min),
    }


def build_channel_cache_info_entry(
    *,
    cache_file: Path,
    cache_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Build evaluator-consumable channel cache info entry."""
    abs_path = cache_file.expanduser().resolve()
    return {
        "channel_cache_key": str(cache_metadata.get("channel_cache_key", "")),
        "channel_cache_file": str(abs_path),
        "expected_metadata": {
            k: v
            for k, v in dict(cache_metadata).items()
            if k != "channel_cache_key"
        },
    }


@dataclass(frozen=True)
class TransferCacheArtifact:
    """Transfer channel cache resolution output."""

    cache_key: str
    cache_file: Path
    metadata: dict[str, Any]
    channels: list[Any]
    channel_version: str
    channel_params: dict[str, Any]


def build_or_load_transfer_channel_cache(
    *,
    scenario_cfg: Any,
    scenario_name: str,
    cache_root: Path,
    cache_num_networks: int,
    auto_generate: bool,
    require_existing_cache: bool,
) -> TransferCacheArtifact:
    """Load transfer channel cache or generate it under transfer cache root."""
    seed = _to_int(_select(scenario_cfg, "seed", default=None), key="seed")
    n_links = _to_int(_select(scenario_cfg, "dataset.n_links", default=None), key="dataset.n_links")
    deployment_range = _to_float(
        _select(scenario_cfg, "dataset.deployment_range", default=None),
        key="dataset.deployment_range",
    )
    channel_cfg = _as_dict(_select(scenario_cfg, "channel", default={}))
    channel_version = normalize_channel_version(channel_cfg.get("version"), default="v2")
    channel_params = sanitize_channel_params(channel_cfg)

    cache_metadata = build_wra_channel_cache_metadata(
        seed=seed,
        num_networks=int(cache_num_networks),
        n_links=n_links,
        deployment_range=deployment_range,
        channel_version=channel_version,
        channel_params=channel_params,
        channel_config_name=f"transfer_{scenario_name}",
    )
    cache_key = str(cache_metadata["channel_cache_key"])

    cache_dir = cache_root.expanduser().resolve() / scenario_name
    cache_file = cache_dir / f"{cache_key}.pt"

    if cache_file.exists():
        try:
            loaded = torch.load(cache_file, map_location="cpu", weights_only=False)
        except TypeError:
            loaded = torch.load(cache_file, map_location="cpu")
        if not isinstance(loaded, dict) or "channels" not in loaded:
            raise RuntimeError(f"Invalid channel cache file format: {cache_file}")

        cached_meta = canonicalize_channel_cache_metadata(
            loaded.get("metadata"),
            default_channel_version="v2",
        )
        expected_meta = canonicalize_channel_cache_metadata(
            cache_metadata,
            default_channel_version="v2",
        )
        mismatches = find_channel_cache_metadata_mismatches(expected_meta, cached_meta)
        if mismatches:
            mismatch_details = "; ".join(
                f"{key}: expected={exp!r}, actual={act!r}"
                for key, (exp, act) in mismatches.items()
            )
            raise RuntimeError(
                f"Transfer channel cache metadata mismatch at {cache_file}: {mismatch_details}"
            )

        channels = loaded["channels"]
        return TransferCacheArtifact(
            cache_key=cache_key,
            cache_file=cache_file.resolve(),
            metadata=dict(cache_metadata),
            channels=list(channels),
            channel_version=channel_version,
            channel_params=dict(channel_params),
        )

    if (not auto_generate) or require_existing_cache:
        reason = (
            "auto-generation disabled"
            if not auto_generate
            else "require_existing_cache=true"
        )
        raise FileNotFoundError(
            f"Transfer channel cache missing ({reason}): {cache_file}"
        )

    cache_dir.mkdir(parents=True, exist_ok=True)
    channels = [
        create_wireless_channel(
            n_links=n_links,
            seed=seed + int(network_id),
            channel_version=channel_version,
            deployment_range=deployment_range,
            channel_params=channel_params,
        )
        for network_id in range(int(cache_num_networks))
    ]
    torch.save({"metadata": cache_metadata, "channels": channels}, cache_file)

    return TransferCacheArtifact(
        cache_key=cache_key,
        cache_file=cache_file.resolve(),
        metadata=dict(cache_metadata),
        channels=list(channels),
        channel_version=channel_version,
        channel_params=dict(channel_params),
    )


class TransferWRADataset(Dataset):
    """Minimal in-memory WRA dataset for transferability evaluation."""

    def __init__(
        self,
        *,
        dataset_name: str,
        network_ids: Sequence[int],
        templates: Mapping[int, WirelessDataDiffusion],
    ):
        self.dataset_name = str(dataset_name)
        self.samples = [
            (self.dataset_name, int(network_id), 0)
            for network_id in network_ids
        ]
        self._templates = {int(k): v for k, v in templates.items()}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> WirelessDataDiffusion:
        _, network_id, _ = self.samples[idx]
        return copy.deepcopy(self._templates[int(network_id)])


@dataclass
class TransferEvaluationBundle:
    """Transfer target evaluation payload."""

    loader: DataLoader
    dataset_info: dict[str, Any]
    evaluated_network_ids: list[int]


def build_transfer_eval_bundle(
    *,
    channels: Sequence[Any],
    evaluated_network_ids: Sequence[int],
    dataset_name: str,
    model_cond_channels: int,
    system_params: Mapping[str, Any],
    graph_cfg: Mapping[str, Any],
    channel_cache_info_entry: Mapping[str, Any],
    channel_version: str,
    channel_params: Mapping[str, Any],
    n_samples_per_network: int,
    batch_size_val: int,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
) -> TransferEvaluationBundle:
    """Create in-memory transfer dataset + grouped eval loader + task metadata."""
    if model_cond_channels not in (2, 3):
        raise ValueError(
            f"Unsupported model_cond_channels={model_cond_channels}; expected 2 or 3."
        )
    r_min = _to_float(system_params.get("r_min"), key="system_params.r_min")

    templates: dict[int, WirelessDataDiffusion] = {}
    associations_map: dict[tuple[str, int], np.ndarray] = {}
    h_ls_map: dict[tuple[str, int], np.ndarray] = {}
    network_seeds_map: dict[tuple[str, int], int] = {}
    channel_versions_map: dict[tuple[str, int], str] = {}

    for network_id in evaluated_network_ids:
        channel = channels[int(network_id)]
        h_ls = np.asarray(channel.large_scale_fading, dtype=np.float32)
        associations = np.asarray(channel.associations, dtype=np.float32)

        graph = build_interference_graph(
            H_ls=h_ls,
            associations=associations,
            top_k=int(graph_cfg["top_k"]),
            min_degree=int(graph_cfg["min_degree"]),
            normalize_method=str(graph_cfg["edge_normalize_method"]),
            self_loop_weight_scale=float(graph_cfg["self_loop_weight_scale"]),
            spectral_normalize_edge_weights=bool(
                graph_cfg["spectral_normalize_edge_weights"]
            ),
        )

        x = graph.x.to(dtype=torch.float32)
        if model_cond_channels == 3:
            r_min_vec = torch.full(
                (x.shape[0], 1),
                float(r_min),
                dtype=x.dtype,
            )
            x = torch.cat([x, r_min_vec], dim=1)
        x = x.unsqueeze(1)  # [N, T=1, F]

        # Expert-free transfer mode: no expert target policy is available.
        # Keep a deterministic placeholder target to satisfy dataset schema;
        # task-level reference_policy=none suppresses expert-dependent metrics.
        y = torch.full((x.shape[0], 1, 1), 0.5, dtype=torch.float32)

        seed_val = getattr(channel, "seed", None)
        if seed_val is None:
            seed_val = int(network_id)
        seed_int = _to_int(seed_val, key=f"channel.seed(network_id={network_id})")

        sample = WirelessDataDiffusion(
            x=x,
            edge_index=graph.edge_index,
            edge_weight=graph.edge_weight,
            y=y,
            network_id=int(network_id),
            dataset_name=str(dataset_name),
            info={
                "dataset_name": str(dataset_name),
                "network_id": int(network_id),
                "network_seed": int(seed_int),
                "channel_version": str(channel_version),
                "has_constraint_conditioning": bool(model_cond_channels == 3),
                "model_cond_channels": int(model_cond_channels),
            },
        )
        templates[int(network_id)] = sample

        key = (str(dataset_name), int(network_id))
        associations_map[key] = associations
        h_ls_map[key] = h_ls
        network_seeds_map[key] = int(seed_int)
        channel_versions_map[key] = str(channel_version)

    eval_dataset = TransferWRADataset(
        dataset_name=str(dataset_name),
        network_ids=[int(v) for v in evaluated_network_ids],
        templates=templates,
    )

    samples_per_network = int(max(1, n_samples_per_network))
    networks_per_batch = int(
        max(1, math.ceil(float(batch_size_val) / float(samples_per_network)))
    )
    eval_batch_size = int(networks_per_batch * samples_per_network)

    sampler = NetworkGroupedBatchSampler(
        dataset=eval_dataset,
        batch_size=eval_batch_size,
        samples_per_network=samples_per_network,
        shuffle=False,
        seed=42,
    )
    loader = DataLoader(
        eval_dataset,
        batch_sampler=sampler,
        num_workers=int(num_workers),
        pin_memory=bool(pin_memory),
        persistent_workers=bool(persistent_workers) and int(num_workers) > 0,
        follow_batch=["y"],
        exclude_keys=["info"],
    )

    dataset_info = {
        "system_params": dict(system_params),
        "channel_params": dict(channel_params),
        "channel_version": str(channel_version),
        "network_seeds": network_seeds_map,
        "associations": associations_map,
        "h_ls_gains": h_ls_map,
        "channel_versions": channel_versions_map,
        "h_timeslot_dirs": {},
        "h_num_timesteps": {},
        "channel_cache_info": {
            str(dataset_name): dict(channel_cache_info_entry),
        },
    }

    return TransferEvaluationBundle(
        loader=loader,
        dataset_info=dataset_info,
        evaluated_network_ids=[int(v) for v in evaluated_network_ids],
    )


# ---------------------------------------------------------------------------
# Disk-based transfer dataset synthesis
# ---------------------------------------------------------------------------


def transfer_dataset_name(
    scenario_short: str,
    cache_key: str,
    *,
    eval_num_networks: int,
    base_r_min: float,
    swept_r_min: Optional[float] = None,
) -> str:
    """Canonical directory name for a transfer dataset.

    The name encodes all parameters that affect the synthesized data so that
    a change in any of them creates a new directory (preventing stale reuse).

    Base:  ``transfer_{scenario_short}_{cache_key[:8]}_n{N}_r{base_r_min:.2f}``
    With swept r_min: ``...._rmin{swept_r_min:.2f}``
    """
    base = f"transfer_{scenario_short}_{cache_key[:8]}_n{eval_num_networks}_r{base_r_min:.2f}"
    if swept_r_min is not None:
        return f"{base}_rmin{swept_r_min:.2f}"
    return base


def select_tail_network_ids(
    total_networks: int,
    eval_num_networks: int,
) -> list[int]:
    """Select the last *eval_num_networks* from a cache of *total_networks*.

    The tail-selection convention mirrors the standard train/val/test split
    where the test fraction occupies the tail of network IDs.
    """
    if eval_num_networks <= 0:
        raise ValueError(
            f"eval_num_networks must be positive, got {eval_num_networks}."
        )
    if eval_num_networks > total_networks:
        raise ValueError(
            f"eval_num_networks ({eval_num_networks}) > total_networks "
            f"({total_networks}) in cache."
        )
    start = total_networks - eval_num_networks
    return list(range(start, total_networks))


def synthesize_transfer_raw_dir(
    *,
    channels: Sequence[Any],
    selected_network_ids: Sequence[int],
    dataset_dir: str,
    system_params: Mapping[str, Any],
    channel_params: Mapping[str, Any],
    channel_version: str,
    channel_cache_info_entry: Mapping[str, Any],
    scenario_name: str,
) -> str:
    """Create a WRADataset-compatible ``raw/`` directory from channel objects.

    Network IDs are renumbered 0..len(selected_network_ids)-1 in the NPZ
    and network_info.json.  ``network_seed`` preserves the original seed
    for channel-cache matching during evaluation.

    Returns the path to ``raw/``.
    """
    raw_dir = osp.join(dataset_dir, "raw")

    # Idempotency: skip if raw/ already has all expected files.
    info_path = osp.join(raw_dir, "network_info.json")
    npz_path = osp.join(raw_dir, "collected_samples.npz")
    if (
        osp.isfile(info_path)
        and osp.isfile(npz_path)
        and osp.isfile(osp.join(raw_dir, "config.yaml"))
    ):
        logger.info("Reusing existing synthesized raw dir: %s", raw_dir)
        return raw_dir

    os.makedirs(raw_dir, exist_ok=True)

    # --- collected_samples.npz ---
    npz_data: dict[str, Any] = {}
    network_info: dict[str, Any] = {
        "system_params": dict(system_params),
        "channel_params": dict(channel_params),
        "channel_version": str(channel_version),
        "channel_cache_info": dict(channel_cache_info_entry),
    }

    for local_id, cache_id in enumerate(selected_network_ids):
        channel = channels[int(cache_id)]
        h_ls = np.asarray(channel.large_scale_fading, dtype=np.float32)
        associations = np.asarray(channel.associations, dtype=np.float32)
        n_links = int(h_ls.shape[0])

        seed_val = getattr(channel, "seed", None)
        if seed_val is None:
            seed_val = int(cache_id)

        net_key = f"network_{local_id}"
        npz_data[net_key] = {
            "H_ls": h_ls,
            "associations": associations,
            "power_samples": [
                {"power_per_transmitter": np.full(n_links, 0.5, dtype=np.float32)}
            ],
        }
        network_info[net_key] = {
            "system_params": {"r_min": float(system_params.get("r_min", 0.5))},
            "network_seed": int(seed_val),
            "n_links": n_links,
        }

    np.savez(npz_path, **npz_data)

    with open(info_path, "w") as f:
        json.dump(network_info, f, indent=2)

    # Minimal config.yaml
    config_path = osp.join(raw_dir, "config.yaml")
    with open(config_path, "w") as f:
        f.write(f"# Auto-generated transfer dataset for {scenario_name}\n")
        f.write("transfer: true\n")

    # Empty h_instantaneous directory
    h_dir = osp.join(raw_dir, "h_instantaneous")
    os.makedirs(h_dir, exist_ok=True)

    logger.info(
        "Synthesized transfer raw dir: %s (%d networks)",
        raw_dir,
        len(selected_network_ids),
    )
    return raw_dir


def create_transfer_r_min_variant(
    *,
    base_raw_dir: str,
    variant_dataset_dir: str,
    r_min: float,
) -> str:
    """Create a symlinked r_min variant of a base transfer dataset.

    Symlinks ``collected_samples.npz``, ``config.yaml``, ``h_instantaneous/``
    from *base_raw_dir*.  Writes a new ``network_info.json`` with r_min
    overridden at all levels.

    Returns the variant's ``raw/`` path.
    """
    from graph_signal_diffusion.cli._wra_r_min_sweep import (
        _override_r_min_in_network_info,
    )

    base_raw_dir = osp.realpath(base_raw_dir)
    variant_raw = osp.join(variant_dataset_dir, "raw")
    os.makedirs(variant_raw, exist_ok=True)

    # Symlink raw artifacts
    for fname in ("collected_samples.npz", "config.yaml", "h_instantaneous"):
        src = osp.join(base_raw_dir, fname)
        dst = osp.join(variant_raw, fname)
        if osp.exists(dst) or osp.islink(dst):
            os.remove(dst)
        if osp.exists(src) or osp.isdir(src):
            os.symlink(src, dst)

    # Write modified network_info.json
    with open(osp.join(base_raw_dir, "network_info.json")) as f:
        ref_info = json.load(f)
    modified_info = _override_r_min_in_network_info(ref_info, r_min)
    with open(osp.join(variant_raw, "network_info.json"), "w") as f:
        json.dump(modified_info, f, indent=2)

    return variant_raw


# ---------------------------------------------------------------------------
# Fuzzy channel cache lookup (broadens build_or_load_transfer_channel_cache)
# ---------------------------------------------------------------------------


def find_compatible_channel_cache(
    *,
    scenario_name: str,
    cache_root: Path,
    expected_metadata: Mapping[str, Any],
    min_num_networks: int,
) -> Optional[TransferCacheArtifact]:
    """Scan ``{cache_root}/{scenario_name}/`` for a compatible cache.

    A cache is compatible if its metadata matches *expected_metadata* on all
    fields except ``num_networks``, and has ``num_networks >= min_num_networks``.
    If multiple matches, prefer the smallest ``num_networks`` (least waste).
    """
    scenario_dir = cache_root.expanduser().resolve() / scenario_name
    if not scenario_dir.is_dir():
        return None

    best: Optional[tuple[int, Path, dict, list, str, dict]] = None

    for cache_file in scenario_dir.glob("*.pt"):
        try:
            loaded = torch.load(cache_file, map_location="cpu", weights_only=False)
        except Exception:
            try:
                loaded = torch.load(cache_file, map_location="cpu")
            except Exception:
                continue
        if not isinstance(loaded, dict) or "channels" not in loaded:
            continue

        cached_meta = loaded.get("metadata", {})
        cached_num = int(cached_meta.get("num_networks", 0))
        if cached_num < min_num_networks:
            continue

        # Compare all fields except num_networks
        compatible = True
        for key, expected_val in expected_metadata.items():
            if key in ("num_networks", "channel_cache_key"):
                continue
            cached_val = cached_meta.get(key)
            if cached_val != expected_val:
                compatible = False
                break
        if not compatible:
            continue

        if best is None or cached_num < best[0]:
            channel_cfg = _as_dict(cached_meta.get("channel_params", {}))
            ch_version = str(cached_meta.get("channel_version", "v2"))
            best = (
                cached_num,
                cache_file,
                dict(cached_meta),
                list(loaded["channels"]),
                ch_version,
                dict(channel_cfg),
            )

    if best is None:
        return None

    _, cache_file, metadata, channels, ch_version, ch_params = best
    cache_key = str(metadata.get("channel_cache_key", ""))
    return TransferCacheArtifact(
        cache_key=cache_key,
        cache_file=cache_file.resolve(),
        metadata=metadata,
        channels=channels,
        channel_version=ch_version,
        channel_params=ch_params,
    )


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def save_transferability_summary(
    *,
    sweep_results: Sequence[Mapping[str, Any]],
    output_dir: str,
) -> None:
    """Write transferability sweep results to CSV and markdown."""
    import pandas as pd

    if not sweep_results:
        return

    df = pd.DataFrame(list(sweep_results))

    # Sort by target_scenario, baseline, r_min
    sort_cols = [c for c in ("target_scenario", "r_min", "baseline") if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    csv_path = osp.join(output_dir, "transferability_sweep_summary.csv")
    df.to_csv(csv_path, index=False)

    # Markdown
    md_lines = ["# Transferability Sweep Summary\n"]
    md_lines.append(df.to_markdown(index=False, floatfmt=".4f"))
    md_path = osp.join(output_dir, "transferability_sweep_summary.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))

    logger.info("Transferability summary saved to %s and %s", csv_path, md_path)


# ---------------------------------------------------------------------------
# Cross-target line plots
# ---------------------------------------------------------------------------

_TRANSFER_PLOT_METRICS = [
    ("mean_rate_generated", "Mean Rate (bps/Hz)"),
    ("rate_1pct_generated", "1st Percentile Rate"),
    ("rate_5pct_generated", "5th Percentile Rate"),
    ("rate_violation_percentage_generated", "Violation %"),
    ("rate_mean_violation_gap_pct_generated", "Mean Violation Gap %"),
]


def _style_get(style: dict, *keys, default=None):
    """Navigate nested style dict."""
    current = style
    for k in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(k, default)
    return current


def plot_transferability_results(
    *,
    sweep_results: Sequence[Mapping[str, Any]],
    output_dir: str,
    r_min_values: Sequence[float],
    plot_style: Optional[Mapping[str, Any]] = None,
) -> None:
    """Generate cross-target line plots for the transferability sweep.

    One figure per r_min value.  Each figure has one panel per metric with
    x = target scenarios (categorical), one line per baseline.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8-darkgrid")

    style = _as_dict(
        _as_dict(plot_style or {}).get("plots", {})
    ).get("transferability_sweep", {})
    style = _as_dict(style)

    dpi = int(_style_get(style, "dpi", default=300))
    fig_size = _style_get(style, "figure_size", default=[7.5, 5.0])
    combined_size = _style_get(style, "combined_figure_size", default=[20.0, 12.0])
    line_width = float(_style_get(style, "line", "linewidth", default=2.0))
    marker = str(_style_get(style, "line", "marker", default="o"))
    marker_size = float(_style_get(style, "line", "markersize", default=8.0))
    font_axis = int(_style_get(style, "fonts", "axis_label", default=16))
    font_title = int(_style_get(style, "fonts", "title", default=16))
    font_legend = int(_style_get(style, "fonts", "legend", default=14))
    font_tick = int(_style_get(style, "fonts", "tick", default=16))
    grid_alpha = float(_style_get(style, "grid", "alpha", default=0.3))
    grid_ls = str(_style_get(style, "grid", "linestyle", default="--"))

    # Baseline colors from parent style
    baseline_colors = _as_dict(
        _as_dict(plot_style or {}).get("baseline", {})
    ).get("colors", {})
    fallback_cycle = ["tab:blue", "tab:orange", "tab:green", "tab:red",
                      "tab:purple", "tab:brown", "tab:pink"]

    plot_dir = osp.join(output_dir, "transferability_sweep_plots")
    os.makedirs(plot_dir, exist_ok=True)

    single_r_min = len(r_min_values) == 1

    for r_val in r_min_values:
        # Filter results for this r_min
        r_results = [r for r in sweep_results if abs(r.get("r_min", 0) - r_val) < 1e-6]
        if not r_results:
            continue

        # Discover baselines and targets
        baselines_seen = sorted({str(r["baseline"]) for r in r_results if "baseline" in r})
        targets_seen = []
        seen_set: set[str] = set()
        for r in r_results:
            t = str(r.get("target_scenario", ""))
            if t and t not in seen_set:
                targets_seen.append(t)
                seen_set.add(t)

        if not baselines_seen or not targets_seen:
            continue

        def _color(bl_name: str, idx: int) -> str:
            c = baseline_colors.get(bl_name)
            if c:
                return str(c)
            return fallback_cycle[idx % len(fallback_cycle)]

        suffix = "" if single_r_min else f"_rmin{r_val:.2f}"

        # --- Combined multi-panel figure ---
        n_metrics = len(_TRANSFER_PLOT_METRICS)
        n_cols = min(3, n_metrics)
        n_rows = math.ceil(n_metrics / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=combined_size, squeeze=False)

        for panel_idx, (metric_key, metric_label) in enumerate(_TRANSFER_PLOT_METRICS):
            ax = axes[panel_idx // n_cols][panel_idx % n_cols]
            x_positions = list(range(len(targets_seen)))

            for bl_idx, bl_name in enumerate(baselines_seen):
                y_vals = []
                for t in targets_seen:
                    match = [
                        r for r in r_results
                        if r.get("baseline") == bl_name and r.get("target_scenario") == t
                    ]
                    if match and metric_key in match[0]:
                        y_vals.append(float(match[0][metric_key]))
                    else:
                        y_vals.append(float("nan"))

                ax.plot(
                    x_positions, y_vals,
                    color=_color(bl_name, bl_idx),
                    linewidth=line_width,
                    marker=marker,
                    markersize=marker_size,
                    label=bl_name,
                )

            ax.set_xticks(x_positions)
            short_targets = [t.rsplit("_", 2)[-2] + "_" + t.rsplit("_", 1)[-1]
                             if "_" in t else t for t in targets_seen]
            ax.set_xticklabels(short_targets, rotation=30, ha="right", fontsize=font_tick - 2)
            ax.tick_params(axis="y", labelsize=font_tick - 2)
            ax.set_ylabel(metric_label, fontsize=font_axis)
            ax.set_title(metric_label, fontsize=font_title)
            ax.grid(alpha=grid_alpha, linestyle=grid_ls)
            if panel_idx == 0:
                _leg = ax.legend(fontsize=font_legend, frameon=True, framealpha=0.5,
                                 edgecolor="gray", fancybox=True, facecolor="white")
                _leg.get_frame().set_linewidth(1.0)

        # Hide unused panels
        for idx in range(n_metrics, n_rows * n_cols):
            axes[idx // n_cols][idx % n_cols].set_visible(False)

        r_min_title = f" (r_min={r_val:.2f})" if not single_r_min else ""
        fig.suptitle(f"Transferability Sweep{r_min_title}", fontsize=font_title + 2)
        fig.tight_layout()
        fig.savefig(
            osp.join(plot_dir, f"transferability_sweep_all_metrics{suffix}.pdf"),
            dpi=dpi, bbox_inches="tight",
        )
        plt.close(fig)

        # --- Individual per-metric plots ---
        for metric_key, metric_label in _TRANSFER_PLOT_METRICS:
            fig_s, ax_s = plt.subplots(figsize=fig_size)
            x_positions = list(range(len(targets_seen)))

            for bl_idx, bl_name in enumerate(baselines_seen):
                y_vals = []
                for t in targets_seen:
                    match = [
                        r for r in r_results
                        if r.get("baseline") == bl_name and r.get("target_scenario") == t
                    ]
                    if match and metric_key in match[0]:
                        y_vals.append(float(match[0][metric_key]))
                    else:
                        y_vals.append(float("nan"))

                ax_s.plot(
                    x_positions, y_vals,
                    color=_color(bl_name, bl_idx),
                    linewidth=line_width,
                    marker=marker,
                    markersize=marker_size,
                    label=bl_name,
                )

            ax_s.set_xticks(x_positions)
            short_targets = [t.rsplit("_", 2)[-2] + "_" + t.rsplit("_", 1)[-1]
                             if "_" in t else t for t in targets_seen]
            ax_s.set_xticklabels(short_targets, rotation=30, ha="right", fontsize=font_tick)
            ax_s.tick_params(axis="y", labelsize=font_tick)
            ax_s.set_ylabel(metric_label, fontsize=font_axis)
            ax_s.set_title(f"{metric_label}{r_min_title}", fontsize=font_title)
            _leg_s = ax_s.legend(fontsize=font_legend, frameon=True, framealpha=0.5,
                                 edgecolor="gray", fancybox=True, facecolor="white")
            _leg_s.get_frame().set_linewidth(1.0)
            ax_s.grid(alpha=grid_alpha, linestyle=grid_ls)
            fig_s.tight_layout()
            fig_s.savefig(
                osp.join(plot_dir, f"transferability_sweep_{metric_key}{suffix}.pdf"),
                dpi=dpi, bbox_inches="tight",
            )
            plt.close(fig_s)

    logger.info("Transferability plots saved to %s", plot_dir)


_SIZE_SCALING_METRICS = [
    ("mean_rate_generated", "Ergodic mean-rate (bits/s/Hz)"),
    ("rate_1pct_generated", "Ergodic p1-rate (bits/s/Hz)"),
    ("rate_5pct_generated", "Ergodic p5-rate (bits/s/Hz)"),
    ("rate_10pct_generated", "Ergodic p10-rate (bits/s/Hz)"),
    ("rate_violation_percentage_generated", "Violation %"),
]


def plot_transferability_size_scaling(
    *,
    sweep_results: Sequence[Mapping[str, Any]],
    output_dir: str,
    baseline_name: str = "diffusion",
    density_labels: Optional[Mapping[str, str]] = None,
    plot_style: Optional[Mapping[str, Any]] = None,
) -> None:
    """Generate size-scaling plots: metric vs n_links, one curve per density.

    Entries with ``is_native=True`` are highlighted with a diamond marker.

    Parameters
    ----------
    sweep_results
        Transferability sweep results (each entry has ``n_links``,
        ``deployment_range``, ``target_scenario``, ``baseline``,
        ``is_native`` (optional), and metrics).
    output_dir
        Directory to write plot files.
    baseline_name
        Which baseline to plot (default: ``"diffusion"``).
    density_labels
        Optional ``{density_tag: display_label}`` mapping, e.g.
        ``{"ultra-low": "6.6"}``.  If ``None``, labels are
        auto-computed from ``n_links / deployment_range²``.
    plot_style
        Style config dict (from ``plot_style.plots.transferability_sweep``).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-v0_8-darkgrid")

    style = _as_dict(
        _as_dict(plot_style or {}).get("plots", {})
    ).get("transferability_sweep", {})
    style = _as_dict(style)

    dpi = int(_style_get(style, "dpi", default=300))
    fig_size = _style_get(style, "figure_size", default=[7.5, 5.0])
    line_width = float(_style_get(style, "line", "linewidth", default=2.0))
    marker = str(_style_get(style, "line", "marker", default="o"))
    marker_size = float(_style_get(style, "line", "markersize", default=8.0))
    font_axis = int(_style_get(style, "fonts", "axis_label", default=16))
    font_title = int(_style_get(style, "fonts", "title", default=16))
    font_legend = int(_style_get(style, "fonts", "legend", default=14))
    font_tick = int(_style_get(style, "fonts", "tick", default=16))
    eb_capsize = float(_style_get(style, "error_bar", "capsize", default=3.0))
    eb_capthick = float(_style_get(style, "error_bar", "capthick", default=1.2))
    eb_elinewidth = float(_style_get(style, "error_bar", "elinewidth", default=1.0))
    eb_alpha = float(_style_get(style, "error_bar", "alpha", default=0.6))

    density_labels = _as_dict(density_labels)

    # Parse density from scenario name
    def _density_tag(scenario: str) -> str:
        name = scenario.replace("wra_", "").replace("_density", "")
        # Order matters: longest prefixes first to avoid partial matches
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
                return rest or "default"
        return "unknown"

    # Filter to requested baseline
    bl_results = [
        r for r in sweep_results
        if str(r.get("baseline", "")).lower() == baseline_name.lower()
    ]
    if not bl_results:
        logger.warning("No results for baseline '%s'; skipping size-scaling plots.", baseline_name)
        return

    # Group by density tag
    by_density: dict[str, list[dict]] = {}
    for r in bl_results:
        tag = _density_tag(str(r.get("target_scenario", "")))
        by_density.setdefault(tag, []).append(dict(r))

    # Compute display labels for each density
    density_display: dict[str, str] = {}
    for tag, entries in by_density.items():
        if tag in density_labels and density_labels[tag]:
            density_display[tag] = str(density_labels[tag])
        else:
            # Auto-compute from first entry with deployment_range
            for e in entries:
                dr = e.get("deployment_range")
                nl = e.get("n_links")
                if dr and nl:
                    d_val = float(nl) / (float(dr) ** 2)
                    density_display[tag] = f"{d_val:.1e}"
                    break
            else:
                density_display[tag] = tag

    # Color cycle for densities: blue (lowest) → red (highest)
    density_colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728",
                      "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    # Sort by actual density value (ascending) so blue=lowest, red=highest
    def _density_sort_key(tag: str) -> float:
        entries = by_density.get(tag, [])
        for e in entries:
            dr = e.get("deployment_range")
            nl = e.get("n_links")
            if dr and nl:
                return float(nl) / (float(dr) ** 2)
        # Fallback: alphabetical
        return {"ultra-low": 0, "low": 1, "mid": 2, "high": 3}.get(tag, 99)
    density_order = sorted(by_density.keys(), key=_density_sort_key)
    color_map = {tag: density_colors[i % len(density_colors)]
                 for i, tag in enumerate(density_order)}

    plot_dir = osp.join(output_dir, "transferability_size_scaling", baseline_name)
    os.makedirs(plot_dir, exist_ok=True)

    # Build a global n_links → equally-spaced index mapping
    all_n_links = sorted({int(e.get("n_links", 0))
                          for entries in by_density.values()
                          for e in entries if e.get("n_links")})
    n_links_to_idx = {n: i for i, n in enumerate(all_n_links)}

    # Determine the std key suffix for each metric
    def _std_key(metric_key: str) -> str:
        return metric_key.replace("_generated", "_generated_std")

    from matplotlib.legend_handler import HandlerErrorbar
    _eb_handler = {matplotlib.container.ErrorbarContainer: HandlerErrorbar(xerr_size=0, yerr_size=0)}

    for metric_key, metric_label in _SIZE_SCALING_METRICS:
        fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
        density_handles: list = []
        density_labels: list[str] = []
        aux_handles: list = []
        aux_labels: list[str] = []

        for tag in density_order:
            entries = by_density[tag]
            # Sort by n_links
            entries_sorted = sorted(entries, key=lambda e: int(e.get("n_links", 0)))
            filtered = [(e, n_links_to_idx[int(e["n_links"])]) for e in entries_sorted
                        if metric_key in e and int(e.get("n_links", 0)) in n_links_to_idx]

            if not filtered:
                continue

            x_pos = [idx for _, idx in filtered]
            y_vals = [float(e[metric_key]) for e, _ in filtered]
            y_errs = [float(e.get(_std_key(metric_key), 0.0)) for e, _ in filtered]
            is_native = [bool(e.get("is_native", False)) for e, _ in filtered]
            has_errs = any(e > 0 for e in y_errs)

            color = color_map[tag]
            label = density_display.get(tag, tag)

            # Line through all points with error bars (equally spaced)
            if has_errs:
                h = ax.errorbar(x_pos, y_vals, yerr=y_errs, color=color,
                                linewidth=line_width, marker=marker,
                                markersize=marker_size, zorder=2,
                                capsize=eb_capsize, capthick=eb_capthick,
                                elinewidth=eb_elinewidth, alpha=eb_alpha)
            else:
                h, = ax.plot(x_pos, y_vals, color=color, linewidth=line_width,
                             marker=marker, markersize=marker_size, zorder=2)
            density_handles.append(h)
            density_labels.append(label)

            # Highlight native points with a different marker
            x_nat = [x for x, n in zip(x_pos, is_native) if n]
            y_nat = [y for y, n in zip(y_vals, is_native) if n]
            if x_nat:
                ax.scatter(x_nat, y_nat, marker="D", s=marker_size ** 2 * 1.5,
                           color=color, edgecolors="black", linewidths=1.0,
                           zorder=3)

        # QoS threshold line for rate metrics (skip for violation %)
        if "violation" not in metric_key:
            _r_min_val = None
            for entries in by_density.values():
                for e in entries:
                    rm = e.get("r_min")
                    if rm is not None:
                        _r_min_val = float(rm)
                        break
                if _r_min_val is not None:
                    break
            if _r_min_val is not None:
                h_qos = ax.axhline(
                    _r_min_val, color="red", linestyle="--", linewidth=1.6,
                    alpha=0.8, zorder=1,
                )
                aux_handles.append(h_qos)
                aux_labels.append(r"$f_{\min}$")

        # Add a single legend entry for native marker if any native points exist
        has_native = any(
            e.get("is_native") for entries in by_density.values() for e in entries
        )
        if has_native:
            h_nat = ax.scatter([], [], marker="D", s=marker_size ** 2 * 1.5,
                               color="gray", edgecolors="black", linewidths=1.0)
            aux_handles.append(h_nat)
            aux_labels.append("Native (training size)")

        ax.set_xlabel("Network size (number of tx-rx pairs)", fontsize=font_axis)
        ax.set_ylabel(metric_label, fontsize=font_axis)
        ax.tick_params(axis="both", labelsize=font_tick)
        # Equally spaced x-ticks labeled with actual n_links values
        ax.set_xticks(list(range(len(all_n_links))))
        ax.set_xticklabels([str(n) for n in all_n_links])

        # Primary legend: density curves (with title)
        _leg1 = ax.legend(density_handles, density_labels,
                          fontsize=font_legend, frameon=True, framealpha=0.5,
                          edgecolor="gray", fancybox=True, facecolor="white",
                          title=r"Density (tx-rx pairs/km$^2$)", title_fontsize=font_legend - 1,
                          handler_map=_eb_handler, loc="upper right", ncol=2)
        _leg1.get_frame().set_linewidth(1.0)
        ax.add_artist(_leg1)
        # Auxiliary legend: QoS line + native marker (no title)
        if aux_handles:
            _leg2 = ax.legend(aux_handles, aux_labels,
                              fontsize=font_legend, frameon=True, framealpha=0.5,
                              edgecolor="gray", fancybox=True, facecolor="white",
                              loc="upper left")
            _leg2.get_frame().set_linewidth(1.0)
        fig.tight_layout()
        safe_name = metric_key.replace(".", "_")
        fig.savefig(osp.join(plot_dir, f"size_scaling_{safe_name}.pdf"),
                    dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    logger.info("Size-scaling plots saved to %s", plot_dir)
