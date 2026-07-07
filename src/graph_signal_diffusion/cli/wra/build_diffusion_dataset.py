#!/usr/bin/env python
"""Convert primal-dual trajectory samples into raw WRA diffusion datasets."""

from __future__ import annotations

import json
import shutil
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import hydra
import numpy as np
import torch
import tqdm
import yaml
from omegaconf import DictConfig, OmegaConf

from graph_signal_diffusion.datasets.wra.channel_factory import (
    build_wra_channel_cache_metadata,
    canonicalize_channel_cache_metadata,
    create_wireless_channel,
    find_channel_cache_metadata_mismatches,
    normalize_channel_version,
    register_wra_hydra_resolvers,
    sanitize_channel_params,
)
from graph_signal_diffusion.datasets.wra.sample_schema import (
    RAW_WRA_SCHEMA_VERSION,
    load_pd_samples_npz,
)

register_wra_hydra_resolvers()


def _resolve_input_paths(input_path: Path) -> Tuple[Path, Path, Path]:
    """Resolve run directory and expected source files from a user input path."""
    if input_path.is_dir():
        run_dir = input_path
        primal_history_path = run_dir / "primal_history.jsonl"
        samples_npz_path = run_dir / "collected_samples.npz"
        return run_dir, primal_history_path, samples_npz_path

    run_dir = input_path.parent
    if input_path.suffix.lower() == ".jsonl":
        primal_history_path = input_path
        samples_npz_path = run_dir / "collected_samples.npz"
    elif input_path.suffix.lower() == ".npz":
        primal_history_path = run_dir / "primal_history.jsonl"
        samples_npz_path = input_path
    else:
        primal_history_path = run_dir / "primal_history.jsonl"
        samples_npz_path = run_dir / "collected_samples.npz"

    return run_dir, primal_history_path, samples_npz_path


def _load_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r") as f:
        loaded = json.load(f)
    return loaded if isinstance(loaded, dict) else {}


def _infer_channel_version_from_config(config: Mapping[str, Any], default: str = "v2") -> str:
    top_level = config.get("channel_version")
    if top_level is not None:
        return normalize_channel_version(top_level, default=default)

    channel_cfg = config.get("channel", {})
    if isinstance(channel_cfg, Mapping):
        return normalize_channel_version(channel_cfg.get("version"), default=default)

    dataset_cfg = config.get("dataset", {})
    if isinstance(dataset_cfg, Mapping):
        return normalize_channel_version(dataset_cfg.get("channel_version"), default=default)

    return normalize_channel_version(None, default=default)


def _load_pd_samples_from_primal_history(
    *,
    primal_history_path: Path,
    config: Mapping[str, Any],
    default_channel_version: str = "v2",
) -> Dict[str, Any]:
    """Load checkpoint trajectory samples from primal_history.jsonl into canonical in-memory format."""
    if not primal_history_path.exists():
        raise FileNotFoundError(f"primal_history.jsonl not found: {primal_history_path}")

    collection_metadata = _load_json_if_exists(primal_history_path.parent / "collection_metadata.json")

    associations_by_network: Dict[int, np.ndarray] = {}
    network_seed_by_network: Dict[int, int] = {}
    r_min_by_network: Dict[int, np.ndarray] = {}
    base_network_id_by_network: Dict[int, int] = {}
    constraint_profile_id_by_network: Dict[int, int] = {}
    constraint_profile_name_by_network: Dict[int, str] = {}
    samples_by_network: Dict[int, Dict[int, Dict[str, Any]]] = {}

    with open(primal_history_path, "r") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {primal_history_path} at line {line_idx}: {exc}") from exc

            if not isinstance(entry, dict):
                continue

            network_id_raw = entry.get("network_id")
            if network_id_raw is None:
                continue
            try:
                network_id = int(network_id_raw)
            except (TypeError, ValueError):
                continue

            if entry.get("type") == "metadata":
                if "associations" in entry:
                    associations_by_network[network_id] = np.asarray(entry["associations"], dtype=np.float64)
                if entry.get("network_seed") is not None:
                    try:
                        network_seed_by_network[network_id] = int(entry["network_seed"])
                    except (TypeError, ValueError):
                        pass
                if entry.get("r_min_per_receiver") is not None:
                    r_min_by_network[network_id] = np.asarray(
                        entry["r_min_per_receiver"], dtype=np.float64
                    ).reshape(-1)
                if entry.get("base_network_id") is not None:
                    try:
                        base_network_id_by_network[network_id] = int(entry["base_network_id"])
                    except (TypeError, ValueError):
                        pass
                if entry.get("constraint_profile_id") is not None:
                    try:
                        constraint_profile_id_by_network[network_id] = int(
                            entry["constraint_profile_id"]
                        )
                    except (TypeError, ValueError):
                        pass
                if entry.get("constraint_profile_name") is not None:
                    constraint_profile_name_by_network[network_id] = str(
                        entry["constraint_profile_name"]
                    )
                continue

            if "epoch" not in entry or "power_allocations" not in entry:
                continue

            try:
                epoch = int(entry["epoch"])
            except (TypeError, ValueError):
                continue

            power = np.asarray(entry["power_allocations"], dtype=np.float64)
            rates_raw = entry.get("rates")
            rates = np.asarray(rates_raw, dtype=np.float64) if rates_raw is not None else None

            per_network_epochs = samples_by_network.setdefault(network_id, {})
            # Keep latest record if duplicate epoch exists.
            per_network_epochs[epoch] = {"power": power, "rates": rates}

    if not samples_by_network:
        raise ValueError(
            f"No trajectory samples were found in {primal_history_path}. "
            "Expected entries with keys: epoch, network_id, power_allocations, rates."
        )

    seed_start = get_config_param(config, "seed", None)
    try:
        seed_start = int(seed_start) if seed_start is not None else None
    except (TypeError, ValueError):
        seed_start = None

    channel_version = collection_metadata.get("channel_version")
    if channel_version is None:
        channel_version = _infer_channel_version_from_config(config, default=default_channel_version)
    else:
        channel_version = normalize_channel_version(channel_version, default=default_channel_version)

    networks: Dict[int, Dict[str, Any]] = {}
    for net_id in sorted(samples_by_network.keys()):
        if net_id not in associations_by_network:
            raise ValueError(
                f"Missing associations metadata for network {net_id} in {primal_history_path}. "
                "Ensure primal_history.jsonl includes metadata entries."
            )

        epoch_map = samples_by_network[net_id]
        sorted_epochs = sorted(epoch_map.keys())
        powers = [np.asarray(epoch_map[e]["power"], dtype=np.float64) for e in sorted_epochs]
        power_arr = np.stack(powers, axis=0)

        if all(epoch_map[e]["rates"] is not None for e in sorted_epochs):
            rates = [np.asarray(epoch_map[e]["rates"], dtype=np.float64) for e in sorted_epochs]
            rate_arr: Optional[np.ndarray] = np.stack(rates, axis=0)
        else:
            rate_arr = None

        network_seed = network_seed_by_network.get(net_id)
        if network_seed is None and seed_start is not None:
            network_seed = seed_start + net_id

        networks[net_id] = {
            "network_seed": network_seed,
            "associations": np.asarray(associations_by_network[net_id], dtype=np.float64),
            "power_samples": power_arr,
            "rate_samples": rate_arr,
            "sample_steps": np.asarray(sorted_epochs, dtype=np.int64),
        }
        if net_id in r_min_by_network:
            networks[net_id]["r_min_per_receiver"] = np.asarray(
                r_min_by_network[net_id], dtype=np.float64
            ).reshape(-1)
        if net_id in base_network_id_by_network:
            networks[net_id]["base_network_id"] = int(base_network_id_by_network[net_id])
        if net_id in constraint_profile_id_by_network:
            networks[net_id]["constraint_profile_id"] = int(
                constraint_profile_id_by_network[net_id]
            )
        if net_id in constraint_profile_name_by_network:
            networks[net_id]["constraint_profile_name"] = str(
                constraint_profile_name_by_network[net_id]
            )

    return {
        "schema_version": 2,
        "channel_version": channel_version,
        "network_ids": sorted(networks.keys()),
        "networks": networks,
        "source_format": "primal_history_jsonl",
    }


def _analyze_primal_history_r_min_variability(
    canonical: Mapping[str, Any],
    *,
    atol: float = 1e-8,
) -> Dict[str, Any]:
    """
    Decide whether per-network r_min vectors should be kept as conditioning.

    For primal-history input, r_min_per_receiver can be present even in scalar
    runs. Keeping it as an extra feature is only useful when it actually varies.
    """
    network_ids = canonical.get("network_ids", [])
    networks = canonical.get("networks", {})
    if not isinstance(networks, Mapping):
        networks = {}

    vectors: List[np.ndarray] = []
    for net_id in network_ids:
        net_data = networks.get(int(net_id), {})
        if not isinstance(net_data, Mapping):
            continue
        r_min_vec = net_data.get("r_min_per_receiver")
        if r_min_vec is None:
            continue
        arr = np.asarray(r_min_vec, dtype=np.float64).reshape(-1)
        if arr.size == 0:
            continue
        vectors.append(arr)

    if not vectors:
        return {
            "has_vectors": False,
            "is_constant_global": False,
            "should_include_vectors": False,
            "constant_value": None,
            "num_networks_with_vectors": 0,
        }

    reference_value = float(vectors[0][0])
    is_constant_global = bool(
        all(np.allclose(vec, reference_value, rtol=0.0, atol=atol) for vec in vectors)
    )

    return {
        "has_vectors": True,
        "is_constant_global": is_constant_global,
        "should_include_vectors": not is_constant_global,
        "constant_value": reference_value if is_constant_global else None,
        "num_networks_with_vectors": int(len(vectors)),
    }


def get_config_param(config: Mapping[str, Any], param_name: str, default: Any = None) -> Any:
    """Find a parameter in common Hydra sections."""
    if param_name in config:
        return config[param_name]

    sections = ["dataset", "system", "training", "channel"]
    for section in sections:
        section_obj = config.get(section, {})
        if isinstance(section_obj, Mapping) and param_name in section_obj:
            return section_obj[param_name]

    return default


def _resolve_scalar_r_min_from_config(
    config: Mapping[str, Any],
    *,
    default: Optional[float] = 0.5,
) -> Optional[float]:
    """
    Resolve scalar r_min when available.

    For `constraint_profile.source=scalar`, this prefers
    `training.constraint_profile.scalar.value` and falls back to `training.r_min`.
    For non-scalar profile sources, returns None unless a legacy scalar is explicitly set.
    """
    training_cfg = config.get("training", {}) if isinstance(config, Mapping) else {}
    if not isinstance(training_cfg, Mapping):
        training_cfg = {}

    profile_cfg = training_cfg.get("constraint_profile", {})
    if not isinstance(profile_cfg, Mapping):
        profile_cfg = {}

    source = str(profile_cfg.get("source", "scalar")).strip().lower()

    def _try_float(val: Any) -> Optional[float]:
        """Convert to float, returning None for unresolvable values
        (e.g. unresolved Hydra interpolations like '${training.r_min}')."""
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    if source == "scalar":
        scalar_cfg = profile_cfg.get("scalar", {})
        scalar_candidate: Optional[float] = None
        if isinstance(scalar_cfg, Mapping):
            scalar_candidate = _try_float(scalar_cfg.get("value"))
        if scalar_candidate is None:
            scalar_candidate = _try_float(training_cfg.get("r_min"))
        if scalar_candidate is None:
            scalar_candidate = _try_float(config.get("r_min"))
        if scalar_candidate is None:
            return None if default is None else float(default)
        return scalar_candidate
    else:
        scalar_candidate = _try_float(training_cfg.get("r_min"))
        if scalar_candidate is None:
            scalar_candidate = _try_float(config.get("r_min"))
        if scalar_candidate is None:
            return None
        return scalar_candidate


def _build_threshold_vector(
    *,
    num_receivers: int,
    r_min: Any,
    feasibility_tolerance: float = 0.0,
) -> np.ndarray:
    """
    Canonicalize scalar/vector r_min into a per-receiver threshold vector.
    """
    if r_min is None:
        raise ValueError("r_min cannot be None when threshold vector is required.")

    arr = np.asarray(r_min, dtype=np.float64).reshape(-1)
    if arr.size == 1:
        thresholds = np.full(num_receivers, float(arr[0]), dtype=np.float64)
    elif arr.size == num_receivers:
        thresholds = arr.astype(np.float64, copy=False)
    else:
        raise ValueError(
            "r_min must be scalar or length num_receivers. "
            f"Got length {arr.size} for num_receivers={num_receivers}."
        )
    return thresholds - float(feasibility_tolerance)



def _pairwise_euclidean_distances(features: np.ndarray) -> np.ndarray:
    squared_norm = np.sum(features * features, axis=1, keepdims=True)
    sq_dist = squared_norm + squared_norm.T - 2.0 * (features @ features.T)
    np.maximum(sq_dist, 0.0, out=sq_dist)
    return np.sqrt(sq_dist)


def _summarize_rate_array(rates: np.ndarray, r_min: Optional[Any] = None) -> Dict[str, Any]:
    """Compute standard percentile statistics for a 1D rate array."""
    rates = np.asarray(rates, dtype=np.float64).reshape(-1)
    if rates.size == 0:
        return {}

    summary: Dict[str, Any] = {
        "count": int(rates.size),
        "min": float(np.min(rates)),
        "p1": float(np.percentile(rates, 1)),
        "p5": float(np.percentile(rates, 5)),
        "p10": float(np.percentile(rates, 10)),
        "p25": float(np.percentile(rates, 25)),
        "p50": float(np.percentile(rates, 50)),
        "p75": float(np.percentile(rates, 75)),
        "p90": float(np.percentile(rates, 90)),
        "p95": float(np.percentile(rates, 95)),
        "mean": float(np.mean(rates)),
        "max": float(np.max(rates)),
    }
    if r_min is not None:
        thresholds = np.asarray(r_min, dtype=np.float64).reshape(-1)
        if thresholds.size == 1:
            summary["below_r_min_fraction"] = float(np.mean(rates < float(thresholds[0])))
        elif thresholds.size == rates.size:
            summary["below_r_min_fraction"] = float(np.mean(rates < thresholds))
    return summary


def _summarize_power_samples(power: np.ndarray) -> Dict[str, Any]:
    """Compute percentile statistics for power allocation values (flattened across samples and links)."""
    vals = np.asarray(power, dtype=np.float64).reshape(-1)
    if vals.size == 0:
        return {}
    return {
        "count": int(vals.size),
        "min": float(np.min(vals)),
        "p1": float(np.percentile(vals, 1)),
        "p5": float(np.percentile(vals, 5)),
        "p50": float(np.percentile(vals, 50)),
        "p95": float(np.percentile(vals, 95)),
        "p99": float(np.percentile(vals, 99)),
        "max": float(np.max(vals)),
        "mean": float(np.mean(vals)),
    }


def _to_h_dtype(dtype_name: str):
    name = str(dtype_name).lower().strip()
    if name == "float16":
        return np.float16
    if name == "float64":
        return np.float64
    return np.float32


def _save_network_h_instantaneous(
    *,
    h_inst_dir: Path,
    net_id: int,
    network_seed: int,
    channel: Any,
    channel_version: str,
    num_timesteps: int,
    dtype,
) -> int:
    """Sample H_instantaneous for one network and write network_{net_id}.npz.

    Returns the uncompressed byte count of the H array.
    H convention: (T, m, n) — one small-scale fading matrix per timestep.
    """
    sampled = channel.sample_realization(num_timesteps=num_timesteps)
    # channel returns H with shape (m, n, T); store as (T, m, n)
    h_tmn = np.transpose(np.asarray(sampled["H"], dtype=dtype), (2, 0, 1))
    np.savez_compressed(
        h_inst_dir / f"network_{net_id}.npz",
        H=h_tmn,
        network_id=np.array(net_id, dtype=np.int64),
        network_seed=np.array(network_seed, dtype=np.int64),
        channel_version=np.array(channel_version),
        num_timesteps=np.array(num_timesteps, dtype=np.int64),
        shape_convention=np.array("H[T,m,n]"),
    )
    return int(h_tmn.nbytes)


def _compute_refinement_objective(avg_rates: np.ndarray, objective: str) -> float:
    objective = objective.lower()
    if objective == "min_rate":
        return float(np.min(avg_rates))
    if objective == "p1_rate":
        return float(np.percentile(avg_rates, 1))
    if objective == "p5_rate":
        return float(np.percentile(avg_rates, 5))
    # composite
    min_rate = float(np.min(avg_rates))
    p1_rate = float(np.percentile(avg_rates, 1))
    p5_rate = float(np.percentile(avg_rates, 5))
    return 0.5 * min_rate + 0.25 * p1_rate + 0.25 * p5_rate


def _select_evenly_spaced_indices(num_samples: int, target_size: int) -> np.ndarray:
    if target_size >= num_samples:
        return np.arange(num_samples, dtype=np.int64)
    if target_size <= 1:
        return np.array([num_samples - 1], dtype=np.int64)

    idx = np.linspace(0, num_samples - 1, num=target_size)
    idx = np.round(idx).astype(np.int64)
    idx = np.unique(idx)
    if idx.size < target_size:
        missing = target_size - idx.size
        extras = np.setdiff1d(np.arange(num_samples, dtype=np.int64), idx, assume_unique=True)
        idx = np.concatenate([idx, extras[-missing:]])
    return np.sort(idx[:target_size])


def select_feasible_subset_refined(
    *,
    rates: np.ndarray,
    target_size: int,
    r_min: Any,
    feasibility_tolerance: float,
    num_bottleneck_nodes: int,
    objective: str,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """Greedy subset refinement balancing feasibility, objective, and diversity."""
    rates = np.asarray(rates, dtype=np.float64)
    if rates.ndim != 2:
        raise ValueError(f"Expected rates shape [num_samples, num_receivers], got {rates.shape}")

    num_samples, num_receivers = rates.shape
    target_size = max(1, min(int(target_size), num_samples))
    thresholds = _build_threshold_vector(
        num_receivers=num_receivers,
        r_min=r_min,
        feasibility_tolerance=feasibility_tolerance,
    )

    if num_samples == 0:
        return np.array([], dtype=np.int64), {
            "is_feasible": False,
            "num_violating_receivers": 0,
            "objective_value": float("nan"),
            "selection_strategy": "empty",
        }

    avg_initial = rates.mean(axis=0)
    num_bottleneck_nodes = max(1, min(int(num_bottleneck_nodes), num_receivers))
    bottleneck_nodes = np.argsort(avg_initial)[:num_bottleneck_nodes]
    features = rates[:, bottleneck_nodes]

    dist_matrix = _pairwise_euclidean_distances(features) if num_samples > 1 else np.zeros((1, 1), dtype=np.float64)

    selected_mask = np.ones(num_samples, dtype=bool)
    selected_count = num_samples
    sum_rates = rates.sum(axis=0)
    diversity_row_sum = dist_matrix.sum(axis=1)

    while selected_count > target_size:
        selected_indices = np.flatnonzero(selected_mask)
        best_candidate: Optional[tuple[float, float, float, int]] = None

        for idx in selected_indices:
            new_count = selected_count - 1
            if new_count <= 0:
                continue

            avg_after = (sum_rates - rates[idx]) / new_count
            num_violations = int(np.sum(avg_after < thresholds))
            objective_after = _compute_refinement_objective(avg_after, objective)
            diversity_loss = float(diversity_row_sum[idx] / max(1, new_count))

            # Lexicographic: minimize violations, maximize objective, then keep diversity.
            candidate = (float(num_violations), -objective_after, diversity_loss, int(idx))
            if best_candidate is None or candidate < best_candidate:
                best_candidate = candidate

        if best_candidate is None:
            break

        idx_remove = best_candidate[3]
        selected_mask[idx_remove] = False
        sum_rates -= rates[idx_remove]
        selected_count -= 1

        selected_indices = np.flatnonzero(selected_mask)
        if selected_indices.size > 0:
            diversity_row_sum[selected_indices] -= dist_matrix[selected_indices, idx_remove]
        diversity_row_sum[idx_remove] = 0.0

    selected = np.flatnonzero(selected_mask)
    final_avg = rates[selected].mean(axis=0)
    violations = np.where(final_avg < thresholds)[0]

    summary = {
        "selection_strategy": "feasible_diverse_refinement",
        "objective": objective,
        "target_size": int(target_size),
        "selected_size": int(selected.size),
        "is_feasible": bool(violations.size == 0),
        "num_violating_receivers": int(violations.size),
        "objective_value": _compute_refinement_objective(final_avg, objective),
        "min_rate": float(np.min(final_avg)),
        "p1_rate": float(np.percentile(final_avg, 1)),
        "p5_rate": float(np.percentile(final_avg, 5)),
        "bottleneck_nodes": bottleneck_nodes.tolist(),
        "threshold_min": float(np.min(thresholds)),
        "threshold_max": float(np.max(thresholds)),
    }
    return selected.astype(np.int64), summary


def _resolve_output_dir(
    *,
    output_root: Path,
    output_raw_wra_dir: Optional[Path],
    scenario_name: str,
    pd_collection_key: str,
) -> Path:
    if output_raw_wra_dir is not None:
        return output_raw_wra_dir
    return output_root / scenario_name / pd_collection_key / "raw"


def _try_load_channel_cache(
    config: Mapping[str, Any], run_dir: Path
) -> Tuple[Optional[list], Dict[str, Any]]:
    """Try to load the pre-generated channel cache used by PD training.

    Replicates the cache lookup pattern from train_pd.py using the same
    content-addressed key so the channels are guaranteed to match.

    Returns (channels, cache_info) where channels is the list of
    WirelessChannel objects on success (or None on failure), and cache_info
    is a dict with the resolved cache file path + key + expected metadata
    (always populated when derivable, even if the cache file is absent).
    """
    cache_info: Dict[str, Any] = {}
    try:
        seed = int(get_config_param(config, "seed", 0))
        num_networks = int(get_config_param(config, "num_networks", 0))
        n_links = int(get_config_param(config, "n_links", 0))
        deployment_range = float(get_config_param(config, "deployment_range", 1000.0))
        name = config.get("name", "")
        channel_config_name = config.get("channel_config_name", f"default_{seed}")

        channel_version = _infer_channel_version_from_config(config, default="v2")
        channel_params_cfg = sanitize_channel_params(config.get("channel", {}))
        channel_params_cfg.pop("version", None)

        raw_meta = build_wra_channel_cache_metadata(
            seed=seed,
            num_networks=num_networks,
            n_links=n_links,
            deployment_range=deployment_range,
            channel_version=channel_version,
            channel_params=channel_params_cfg,
            channel_config_name=channel_config_name,
        )
        expected_meta = canonicalize_channel_cache_metadata(raw_meta, default_channel_version="v2")
        cache_key = expected_meta["channel_cache_key"]

        cache_file = Path("data/wra_channel_cache") / name / f"{cache_key}.pt"
        cache_info = {
            "channel_cache_key": cache_key,
            "channel_cache_file": str(cache_file),
            "scenario_name": name,
            "expected_metadata": {
                k: v for k, v in expected_meta.items()
                if k != "channel_cache_key"
            },
        }

        if not cache_file.exists():
            print(f"Channel cache not found at {cache_file}; will reconstruct channels.")
            return None, cache_info

        print(f"Found channel cache at {cache_file} — validating …")
        cached = torch.load(cache_file, map_location="cpu", weights_only=False)
        if not isinstance(cached, dict) or "channels" not in cached:
            print("Channel cache format invalid; will reconstruct channels.")
            return None, cache_info

        cached_meta = canonicalize_channel_cache_metadata(
            cached.get("metadata"), default_channel_version="v2"
        )
        mismatches = find_channel_cache_metadata_mismatches(expected_meta, cached_meta)
        if mismatches:
            print("Channel cache metadata mismatch; will reconstruct channels:")
            for key, (exp, act) in mismatches.items():
                print(f"  {key}: expected={exp!r}  actual={act!r}")
            return None, cache_info

        channels = cached["channels"]
        print(f"✓ Loaded {len(channels)} pre-generated channels from cache.")
        return channels, cache_info

    except Exception as exc:
        print(f"Warning: channel cache lookup failed ({exc}); will reconstruct channels.")
        return None, cache_info


def _reconstruct_channel(
    *,
    net_id: int,
    network_seed: int,
    n_links: int,
    channel_version: str,
    channel_params_cfg: Dict[str, Any],
    channel_cache: Optional[list],
) -> Tuple[Any, str]:
    """Resolve a WirelessChannel from cache or per-seed reconstruction.

    Returns (channel, effective_channel_version). Raises RuntimeError if
    both the cache lookup and reconstruction fail.
    """
    if channel_cache is not None and net_id < len(channel_cache):
        return channel_cache[net_id], channel_version

    last_error: Optional[Exception] = None
    candidate_versions = [channel_version, "v3", "v2", "v1"]
    dedup_versions: List[str] = []
    for vc in candidate_versions:
        norm = normalize_channel_version(vc, default=channel_version)
        if norm not in dedup_versions:
            dedup_versions.append(norm)

    for vc in dedup_versions:
        try:
            channel = create_wireless_channel(
                n_links=n_links,
                seed=network_seed,
                channel_version=vc,
                channel_params=channel_params_cfg,
                default_version="v2",
            )
            return channel, vc
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        f"Failed to reconstruct channel for network {net_id}. Last error: {last_error}"
    )


def _print_conversion_summary(
    *,
    raw_dir: Path,
    processed_network_ids: List[int],
    skipped_networks: List[int],
    selected_counts: List[int],
    network_info: Dict[str, Any],
) -> None:
    print("✓ Conversion complete")
    print(f"  Output: {raw_dir}")
    print(f"  Networks processed: {len(processed_network_ids)}")
    print(f"  Networks skipped: {len(skipped_networks)}")
    print(f"  Avg selected samples/network: {float(np.mean(selected_counts)):.2f}")
    power_stats = network_info.get("final_dataset_power_stats", {})
    if power_stats:
        p_max = network_info.get("system_params", {}).get("P_max")
        print("  Final dataset power allocation stats (W):")
        print(
            "    min={min:.4e}, p1={p1:.4e}, p5={p5:.4e}, p50={p50:.4e}, "
            "p95={p95:.4e}, p99={p99:.4e}, max={max:.4e}".format(**power_stats)
        )
        if p_max:
            print("  Final dataset power allocation stats (fraction of P_max):")
            print(
                f"    min={power_stats['min']/p_max:.4f}, p1={power_stats['p1']/p_max:.4f}, "
                f"p5={power_stats['p5']/p_max:.4f}, p50={power_stats['p50']/p_max:.4f}, "
                f"p95={power_stats['p95']/p_max:.4f}, p99={power_stats['p99']/p_max:.4f}, "
                f"max={power_stats['max']/p_max:.4f}"
            )
    final_stats = network_info.get("final_dataset_rate_stats", {})
    stochastic_stats = final_stats.get("stochastic_avg_receiver_rates", {}) if isinstance(final_stats, dict) else {}
    if stochastic_stats:
        print("  Final dataset stochastic-average receiver-rate stats:")
        print(
            "    min={min:.4f}, p1={p1:.4f}, p5={p5:.4f}, p50={p50:.4f}, mean={mean:.4f}, max={max:.4f}".format(
                **stochastic_stats
            )
        )
        if "below_r_min_fraction" in stochastic_stats:
            print(f"    below_r_min_fraction={stochastic_stats['below_r_min_fraction']:.4%}")
    else:
        print("  Final dataset rate stats unavailable (no rate samples present).")
    h_inst = network_info.get("h_instantaneous", {})
    if h_inst.get("generated"):
        print(
            f"  H_instantaneous: {h_inst['num_timesteps']} timesteps, "
            f"dtype={h_inst['dtype']}, "
            f"~{h_inst['estimated_uncompressed_gb']:.3f} GB uncompressed"
        )
        print(f"    Saved to: {raw_dir / h_inst['subdir']}/")


def _collect_from_npz(
    *,
    input_path: str,
    output_root: str,
    output_raw_wra_dir: Optional[str] = None,
    force: bool = False,
    target_samples_per_network: Optional[int] = None,
    pd_collection_key: str = "",
    h_instantaneous_timesteps: Optional[int] = None,
    h_instantaneous_dtype: str = "float32",
) -> Path:
    """Collect samples from a completed PD run NPZ checkpoint (no windowing or greedy refinement)."""
    input_path = Path(input_path)
    output_root = Path(output_root)
    output_raw_wra_dir_path = Path(output_raw_wra_dir) if output_raw_wra_dir else None

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    run_dir, _, samples_npz_path = _resolve_input_paths(input_path)
    config_path = run_dir / ".hydra" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if not samples_npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {samples_npz_path}")

    canonical = load_pd_samples_npz(samples_npz_path, config=config, default_channel_version="v2")
    channel_version = canonical["channel_version"]

    p_max_dbm = float(get_config_param(config, "P_max_dBm", 10.0))
    r_min_scalar = _resolve_scalar_r_min_from_config(config, default=0.5)
    bandwidth_hz = float(get_config_param(config, "bandwidth_Hz", 10e6))
    noise_psd_dbm_hz = float(get_config_param(config, "noise_psd_dBm_Hz", -174.0))

    noise_power_dbm = noise_psd_dbm_hz + 10 * np.log10(bandwidth_hz)
    noise_var = 10 ** ((noise_power_dbm - 30) / 10)
    p_max = 10 ** ((p_max_dbm - 30) / 10)

    system_params = {
        "P_max": float(p_max),
        "P_max_dBm": float(p_max_dbm),
        "noise_var": float(noise_var),
        "BW": float(bandwidth_hz),
        "r_min": None if r_min_scalar is None else float(r_min_scalar),
    }

    scenario_name = str(config.get("name", "unknown")).removeprefix("wra_")
    raw_dir = _resolve_output_dir(
        output_root=output_root,
        output_raw_wra_dir=output_raw_wra_dir_path,
        scenario_name=scenario_name,
        pd_collection_key=pd_collection_key,
    )

    output_files = [raw_dir / n for n in ("collected_samples.npz", "config.yaml", "network_info.json")]
    existing_files = [f for f in output_files if f.exists()]
    if existing_files and not force:
        print(f"Output files already exist in: {raw_dir}")
        for path in existing_files:
            print(f"  - {path.name}")
        print("Use force=True or a different pd_collection_key to avoid overwrite.")
        raise FileExistsError(f"Output already exists: {raw_dir}")

    raw_dir.mkdir(parents=True, exist_ok=True)

    channel_params_cfg: Dict[str, Any] = {}
    dataset_cfg = config.get("dataset", {}) if isinstance(config.get("dataset", {}), Mapping) else {}
    if isinstance(dataset_cfg.get("channel_params"), Mapping):
        channel_params_cfg.update(dataset_cfg["channel_params"])
    channel_cfg = config.get("channel", {}) if isinstance(config.get("channel", {}), Mapping) else {}
    channel_params_cfg.update({k: v for k, v in channel_cfg.items() if k != "version"})
    if get_config_param(config, "deployment_range", None) is not None:
        channel_params_cfg["deployment_range"] = float(get_config_param(config, "deployment_range"))
    channel_params_cfg = sanitize_channel_params(channel_params_cfg)

    channel_cache, channel_cache_info = _try_load_channel_cache(config, run_dir)

    network_info: Dict[str, Any] = {
        "schema_version": RAW_WRA_SCHEMA_VERSION,
        "channel_version": channel_version,
        "system_params": system_params,
        "channel_params": channel_params_cfg,
        "conversion": {
            "sample_source": "npz",
            "target_samples_per_network": int(target_samples_per_network) if target_samples_per_network is not None else None,
            "refine_feasible_subset": False,
        },
        "source_samples_file": str(samples_npz_path),
        "source_run_dir": str(run_dir),
        "source_format": canonical.get("source_format"),
        "constraint_mode": "scalar",
        "has_per_network_r_min": False,
        "channel_cache_info": channel_cache_info,
    }

    save_dict: Dict[str, Any] = {
        "schema_version": np.array(RAW_WRA_SCHEMA_VERSION, dtype=np.int64),
        "channel_version": np.array(channel_version, dtype="<U2"),
    }

    h_inst_dir: Optional[Path] = None
    h_inst_dtype = np.float32
    resolved_h_ts = 0
    total_h_inst_bytes = 0
    if h_instantaneous_timesteps is not None:
        resolved_h_ts = (
            h_instantaneous_timesteps
            if h_instantaneous_timesteps > 0
            else int(get_config_param(config, "num_timesteps", 200))
        )
        h_inst_dtype = _to_h_dtype(h_instantaneous_dtype)
        h_inst_dir = raw_dir / "h_instantaneous"
        h_inst_dir.mkdir(parents=True, exist_ok=True)

    processed_network_ids: List[int] = []
    skipped_networks: List[int] = []
    selected_counts: List[int] = []
    selected_avg_rates_across_networks: List[np.ndarray] = []
    selected_all_rates_across_networks: List[np.ndarray] = []
    selected_all_power_across_networks: List[np.ndarray] = []
    has_per_network_r_min = False

    for net_id in tqdm.tqdm(canonical["network_ids"], desc="Processing networks"):
        net_data = canonical["networks"][net_id]
        network_seed = net_data.get("network_seed")
        associations = np.asarray(net_data["associations"], dtype=np.float64)
        power_samples = np.asarray(net_data["power_samples"], dtype=np.float64)
        rate_samples = net_data.get("rate_samples")
        if rate_samples is not None:
            rate_samples = np.asarray(rate_samples, dtype=np.float64)
        sample_steps = net_data.get("sample_steps")
        if sample_steps is None:
            sample_steps = np.arange(power_samples.shape[0], dtype=np.int64)
        else:
            sample_steps = np.asarray(sample_steps, dtype=np.int64)

        r_min_per_receiver = net_data.get("r_min_per_receiver")
        if r_min_per_receiver is not None:
            r_min_per_receiver = np.asarray(r_min_per_receiver, dtype=np.float64).reshape(-1)
            has_per_network_r_min = True
        base_network_id = net_data.get("base_network_id")
        constraint_profile_id = net_data.get("constraint_profile_id")
        constraint_profile_name = net_data.get("constraint_profile_name")

        if associations.ndim != 2 or power_samples.ndim != 2 or power_samples.shape[0] == 0:
            skipped_networks.append(net_id)
            print(f"Skipping network {net_id}: malformed associations/power samples")
            continue

        if network_seed is None:
            skipped_networks.append(net_id)
            print(f"Skipping network {net_id}: missing network_seed")
            continue

        total_samples = power_samples.shape[0]
        target_k = total_samples
        if target_samples_per_network is not None and target_samples_per_network > 0:
            target_k = min(int(target_samples_per_network), total_samples)

        if target_k < total_samples:
            selected_local_idx = _select_evenly_spaced_indices(total_samples, target_k)
            selection_summary: Dict[str, Any] = {
                "selection_strategy": "evenly_spaced",
                "target_size": int(target_k),
                "selected_size": int(selected_local_idx.size),
                "is_feasible": None,
            }
        else:
            selected_local_idx = np.arange(total_samples, dtype=np.int64)
            selection_summary = {
                "selection_strategy": "all_samples",
                "target_size": int(target_k),
                "selected_size": int(selected_local_idx.size),
                "is_feasible": None,
            }

        power_selected = power_samples[selected_local_idx]
        rates_selected = rate_samples[selected_local_idx] if rate_samples is not None else None
        steps_selected = sample_steps[selected_local_idx]

        selected_power_stats = _summarize_power_samples(power_selected)
        selected_all_power_across_networks.append(power_selected.reshape(-1))

        selected_rate_stats: Optional[Dict[str, Any]] = None
        if rates_selected is not None and rates_selected.size > 0:
            rates_selected = np.asarray(rates_selected, dtype=np.float64)
            avg_rates_selected = rates_selected.mean(axis=0)
            if r_min_per_receiver is not None and r_min_per_receiver.shape[0] == rates_selected.shape[1]:
                stochastic_thresholds = r_min_per_receiver
                all_thresholds = np.broadcast_to(
                    r_min_per_receiver[None, :], rates_selected.shape
                ).reshape(-1)
            else:
                stochastic_thresholds = r_min_scalar
                all_thresholds = r_min_scalar
            selected_rate_stats = {
                "stochastic_avg_receiver_rates": _summarize_rate_array(
                    avg_rates_selected, r_min=stochastic_thresholds
                ),
                "all_selected_receiver_rates": _summarize_rate_array(
                    rates_selected.reshape(-1), r_min=all_thresholds
                ),
                "num_selected_samples": int(rates_selected.shape[0]),
                "num_receivers": int(rates_selected.shape[1]) if rates_selected.ndim == 2 else int(avg_rates_selected.size),
            }
            selected_avg_rates_across_networks.append(avg_rates_selected)
            selected_all_rates_across_networks.append(rates_selected.reshape(-1))

        n_links = int(associations.shape[0])
        deployment_range_value = float(channel_params_cfg.get("deployment_range", get_config_param(config, "deployment_range", 1000.0)))

        channel, effective_channel_version = _reconstruct_channel(
            net_id=net_id,
            network_seed=int(network_seed),
            n_links=n_links,
            channel_version=channel_version,
            channel_params_cfg=channel_params_cfg,
            channel_cache=channel_cache,
        )
        h_ls = np.asarray(channel.large_scale_fading, dtype=np.float64)
        tx_locations = np.asarray(channel.tx_locations, dtype=np.float64)
        rx_locations = np.asarray(channel.rx_locations, dtype=np.float64)
        deployment_range_value = float(channel.deployment_range)

        if h_inst_dir is not None:
            total_h_inst_bytes += _save_network_h_instantaneous(
                h_inst_dir=h_inst_dir,
                net_id=net_id,
                network_seed=int(network_seed),
                channel=channel,
                channel_version=effective_channel_version,
                num_timesteps=resolved_h_ts,
                dtype=h_inst_dtype,
            )

        converted_samples: List[Dict[str, Any]] = []
        for i in range(power_selected.shape[0]):
            power_tx = np.asarray(power_selected[i], dtype=np.float64)
            power_rx = associations.T @ power_tx
            rates_i = None
            if rates_selected is not None and i < rates_selected.shape[0]:
                rates_i = np.asarray(rates_selected[i], dtype=np.float64)
            converted_samples.append({
                "power_per_transmitter": power_tx,
                "power_per_receiver": power_rx,
                "rates": rates_i,
                "source_step": int(steps_selected[i]),
            })

        net_key = f"network_{net_id}"
        save_dict[net_key] = {
            "network_id": int(net_id),
            "network_seed": int(network_seed),
            "channel_version": effective_channel_version,
            "H_ls": h_ls,
            "associations": associations,
            "power_samples": converted_samples,
            "sample_steps": steps_selected,
        }
        if r_min_per_receiver is not None:
            save_dict[net_key]["r_min_per_receiver"] = r_min_per_receiver
        if base_network_id is not None:
            save_dict[net_key]["base_network_id"] = int(base_network_id)
        if constraint_profile_id is not None:
            save_dict[net_key]["constraint_profile_id"] = int(constraint_profile_id)
        if constraint_profile_name is not None:
            save_dict[net_key]["constraint_profile_name"] = str(constraint_profile_name)
        network_info[net_key] = {
            "network_seed": int(network_seed),
            "n_links": n_links,
            "deployment_range": deployment_range_value,
            "tx_locations": tx_locations.tolist(),
            "rx_locations": rx_locations.tolist(),
            "top_k": int(get_config_param(config, "top_k", 10)),
            "min_degree": int(get_config_param(config, "min_degree", 1)),
            "normalize_method": str(get_config_param(config, "normalize_method", "log1p")),
            "channel_version": effective_channel_version,
            "channel_params": channel_params_cfg,
            "system_params": system_params,
            "num_samples_source": int(total_samples),
            "num_samples_selected": int(power_selected.shape[0]),
            "selection_summary": selection_summary,
            "selected_power_stats": selected_power_stats,
            "selected_rate_stats": selected_rate_stats,
        }
        if r_min_per_receiver is not None:
            network_info[net_key]["r_min_per_receiver"] = r_min_per_receiver.tolist()
        if base_network_id is not None:
            network_info[net_key]["base_network_id"] = int(base_network_id)
        if constraint_profile_id is not None:
            network_info[net_key]["constraint_profile_id"] = int(constraint_profile_id)
        if constraint_profile_name is not None:
            network_info[net_key]["constraint_profile_name"] = str(constraint_profile_name)
        processed_network_ids.append(int(net_id))
        selected_counts.append(int(power_selected.shape[0]))

    if not processed_network_ids:
        raise RuntimeError("No valid networks were processed during conversion.")

    save_dict["network_ids"] = np.asarray(processed_network_ids, dtype=np.int64)

    if selected_all_power_across_networks:
        network_info["final_dataset_power_stats"] = _summarize_power_samples(
            np.concatenate(selected_all_power_across_networks)
        )

    if selected_avg_rates_across_networks:
        global_stochastic_rates = np.concatenate(selected_avg_rates_across_networks, axis=0)
        global_all_selected_rates = np.concatenate(selected_all_rates_across_networks, axis=0)
        global_r_min_threshold = None if has_per_network_r_min else r_min_scalar
        network_info["final_dataset_rate_stats"] = {
            "stochastic_avg_receiver_rates": _summarize_rate_array(
                global_stochastic_rates, r_min=global_r_min_threshold
            ),
            "all_selected_receiver_rates": _summarize_rate_array(
                global_all_selected_rates, r_min=global_r_min_threshold
            ),
            "num_networks_with_rates": int(len(selected_avg_rates_across_networks)),
        }
    else:
        network_info["final_dataset_rate_stats"] = {
            "available": False,
            "message": "Rate statistics unavailable because selected samples do not include rates.",
        }
    network_info["has_per_network_r_min"] = bool(has_per_network_r_min)
    network_info["constraint_mode"] = "per_network_r_min" if has_per_network_r_min else "scalar"

    if h_inst_dir is not None:
        h_inst_metadata = {
            "generated": True,
            "num_timesteps": int(resolved_h_ts),
            "dtype": str(h_inst_dtype),
            "shape_convention": "H[T,m,n]",
            "subdir": "h_instantaneous",
            "estimated_uncompressed_gb": float(total_h_inst_bytes / 1e9),
        }
        with open(h_inst_dir / "metadata.json", "w") as f:
            json.dump(h_inst_metadata, f, indent=2)
        network_info["h_instantaneous"] = h_inst_metadata
    else:
        network_info["h_instantaneous"] = {"generated": False}

    np.savez_compressed(raw_dir / "collected_samples.npz", **save_dict)
    shutil.copy(config_path, raw_dir / "config.yaml")
    with open(raw_dir / "network_info.json", "w") as f:
        json.dump(network_info, f, indent=2)

    _print_conversion_summary(
        raw_dir=raw_dir,
        processed_network_ids=processed_network_ids,
        skipped_networks=skipped_networks,
        selected_counts=selected_counts,
        network_info=network_info,
    )
    return raw_dir


def _collect_from_primal_history(
    *,
    input_path: str,
    output_root: str,
    output_raw_wra_dir: Optional[str] = None,
    force: bool = False,
    target_samples_per_network: Optional[int] = None,
    pd_collection_key: str = "",
    window_size: Optional[int] = None,
    refine_feasible_subset: bool = False,
    refine_objective: str = "min_rate",
    subset_feasibility_tolerance: float = 0.0,
    subset_bottleneck_nodes: int = 5,
    h_instantaneous_timesteps: Optional[int] = None,
    h_instantaneous_dtype: str = "float32",
) -> Path:
    """Collect samples from a primal_history.jsonl trajectory (supports windowing and greedy refinement)."""
    input_path = Path(input_path)
    output_root = Path(output_root)
    output_raw_wra_dir_path = Path(output_raw_wra_dir) if output_raw_wra_dir else None

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    run_dir, primal_history_path, _ = _resolve_input_paths(input_path)
    config_path = run_dir / ".hydra" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    canonical = _load_pd_samples_from_primal_history(
        primal_history_path=primal_history_path,
        config=config,
        default_channel_version="v2",
    )
    channel_version = canonical["channel_version"]
    r_min_variability = _analyze_primal_history_r_min_variability(canonical)
    include_per_network_r_min = bool(r_min_variability["should_include_vectors"])

    p_max_dbm = float(get_config_param(config, "P_max_dBm", 10.0))
    r_min_scalar = _resolve_scalar_r_min_from_config(config, default=0.5)
    if r_min_scalar is None and r_min_variability["constant_value"] is not None:
        r_min_scalar = float(r_min_variability["constant_value"])
    bandwidth_hz = float(get_config_param(config, "bandwidth_Hz", 10e6))
    noise_psd_dbm_hz = float(get_config_param(config, "noise_psd_dBm_Hz", -174.0))

    if r_min_variability["has_vectors"] and not include_per_network_r_min:
        print(
            "Detected constant r_min_per_receiver across primal-history networks; "
            "omitting per-network r_min vectors from converted raw dataset."
        )
    elif include_per_network_r_min:
        print(
            "Detected varying r_min_per_receiver across primal-history networks; "
            "preserving per-network r_min vectors in converted raw dataset."
        )

    noise_power_dbm = noise_psd_dbm_hz + 10 * np.log10(bandwidth_hz)
    noise_var = 10 ** ((noise_power_dbm - 30) / 10)
    p_max = 10 ** ((p_max_dbm - 30) / 10)

    system_params = {
        "P_max": float(p_max),
        "P_max_dBm": float(p_max_dbm),
        "noise_var": float(noise_var),
        "BW": float(bandwidth_hz),
        "r_min": None if r_min_scalar is None else float(r_min_scalar),
    }

    scenario_name = str(config.get("name", "unknown")).removeprefix("wra_")
    raw_dir = _resolve_output_dir(
        output_root=output_root,
        output_raw_wra_dir=output_raw_wra_dir_path,
        scenario_name=scenario_name,
        pd_collection_key=pd_collection_key,
    )

    output_files = [raw_dir / n for n in ("collected_samples.npz", "config.yaml", "network_info.json")]
    existing_files = [f for f in output_files if f.exists()]
    if existing_files and not force:
        print(f"Output files already exist in: {raw_dir}")
        for path in existing_files:
            print(f"  - {path.name}")
        print("Use force=True or a different pd_collection_key to avoid overwrite.")
        raise FileExistsError(f"Output already exists: {raw_dir}")

    raw_dir.mkdir(parents=True, exist_ok=True)

    channel_params_cfg: Dict[str, Any] = {}
    dataset_cfg = config.get("dataset", {}) if isinstance(config.get("dataset", {}), Mapping) else {}
    if isinstance(dataset_cfg.get("channel_params"), Mapping):
        channel_params_cfg.update(dataset_cfg["channel_params"])
    channel_cfg = config.get("channel", {}) if isinstance(config.get("channel", {}), Mapping) else {}
    channel_params_cfg.update({k: v for k, v in channel_cfg.items() if k != "version"})
    if get_config_param(config, "deployment_range", None) is not None:
        channel_params_cfg["deployment_range"] = float(get_config_param(config, "deployment_range"))
    channel_params_cfg = sanitize_channel_params(channel_params_cfg)

    channel_cache, channel_cache_info = _try_load_channel_cache(config, run_dir)

    network_info: Dict[str, Any] = {
        "schema_version": RAW_WRA_SCHEMA_VERSION,
        "channel_version": channel_version,
        "system_params": system_params,
        "channel_params": channel_params_cfg,
        "conversion": {
            "sample_source": "primal_history",
            "window_size": int(window_size) if window_size is not None else None,
            "target_samples_per_network": int(target_samples_per_network) if target_samples_per_network is not None else None,
            "refine_feasible_subset": bool(refine_feasible_subset),
            "refine_objective": refine_objective,
            "subset_feasibility_tolerance": float(subset_feasibility_tolerance),
            "subset_bottleneck_nodes": int(subset_bottleneck_nodes),
        },
        "source_samples_file": str(primal_history_path),
        "source_run_dir": str(run_dir),
        "source_format": canonical.get("source_format"),
        "constraint_mode": "scalar",
        "has_per_network_r_min": False,
        "source_has_per_network_r_min": bool(r_min_variability["has_vectors"]),
        "source_r_min_is_global_constant": bool(r_min_variability["is_constant_global"]),
        "included_per_network_r_min": bool(include_per_network_r_min),
        "channel_cache_info": channel_cache_info,
    }

    save_dict: Dict[str, Any] = {
        "schema_version": np.array(RAW_WRA_SCHEMA_VERSION, dtype=np.int64),
        "channel_version": np.array(channel_version, dtype="<U2"),
    }

    h_inst_dir: Optional[Path] = None
    h_inst_dtype = np.float32
    resolved_h_ts = 0
    total_h_inst_bytes = 0
    if h_instantaneous_timesteps is not None:
        resolved_h_ts = (
            h_instantaneous_timesteps
            if h_instantaneous_timesteps > 0
            else int(get_config_param(config, "num_timesteps", 200))
        )
        h_inst_dtype = _to_h_dtype(h_instantaneous_dtype)
        h_inst_dir = raw_dir / "h_instantaneous"
        h_inst_dir.mkdir(parents=True, exist_ok=True)

    processed_network_ids: List[int] = []
    skipped_networks: List[int] = []
    selected_counts: List[int] = []
    selected_avg_rates_across_networks: List[np.ndarray] = []
    selected_all_rates_across_networks: List[np.ndarray] = []
    selected_all_power_across_networks: List[np.ndarray] = []
    has_per_network_r_min = False

    for net_id in tqdm.tqdm(canonical["network_ids"], desc="Processing networks"):
        net_data = canonical["networks"][net_id]
        network_seed = net_data.get("network_seed")
        associations = np.asarray(net_data["associations"], dtype=np.float64)
        power_samples = np.asarray(net_data["power_samples"], dtype=np.float64)
        rate_samples = net_data.get("rate_samples")
        if rate_samples is not None:
            rate_samples = np.asarray(rate_samples, dtype=np.float64)
        sample_steps = net_data.get("sample_steps")
        if sample_steps is None:
            sample_steps = np.arange(power_samples.shape[0], dtype=np.int64)
        else:
            sample_steps = np.asarray(sample_steps, dtype=np.int64)

        raw_r_min_per_receiver = net_data.get("r_min_per_receiver")
        if raw_r_min_per_receiver is not None:
            raw_r_min_per_receiver = np.asarray(raw_r_min_per_receiver, dtype=np.float64).reshape(-1)
        r_min_per_receiver = (
            raw_r_min_per_receiver
            if include_per_network_r_min and raw_r_min_per_receiver is not None
            else None
        )
        if r_min_per_receiver is not None:
            has_per_network_r_min = True
        base_network_id = net_data.get("base_network_id")
        constraint_profile_id = net_data.get("constraint_profile_id")
        constraint_profile_name = net_data.get("constraint_profile_name")

        if associations.ndim != 2 or power_samples.ndim != 2 or power_samples.shape[0] == 0:
            skipped_networks.append(net_id)
            print(f"Skipping network {net_id}: malformed associations/power samples")
            continue

        if network_seed is None:
            skipped_networks.append(net_id)
            print(f"Skipping network {net_id}: missing network_seed")
            continue

        total_samples = power_samples.shape[0]

        # Apply trailing window (primal_history-specific).
        if window_size is not None and window_size > 0:
            if window_size > total_samples:
                warnings.warn(
                    f"window_size={window_size} exceeds the {total_samples} epoch entries "
                    f"available in primal_history.jsonl (network {net_id}). "
                    f"The trainer keeps a bounded rolling window of recent epochs; earlier "
                    f"entries are not persisted. Using all {total_samples} available entries.",
                    UserWarning,
                    stacklevel=2,
                )
            if window_size < total_samples:
                start = total_samples - int(window_size)
                power_window = power_samples[start:]
                rate_window = rate_samples[start:] if rate_samples is not None else None
                steps_window = sample_steps[start:]
            else:
                power_window = power_samples
                rate_window = rate_samples
                steps_window = sample_steps
        else:
            power_window = power_samples
            rate_window = rate_samples
            steps_window = sample_steps

        target_k = power_window.shape[0]
        if target_samples_per_network is not None and target_samples_per_network > 0:
            target_k = min(int(target_samples_per_network), power_window.shape[0])

        if target_k < power_window.shape[0]:
            if refine_feasible_subset and rate_window is not None and rate_window.size > 0:
                if r_min_per_receiver is not None and r_min_per_receiver.shape[0] == rate_window.shape[1]:
                    refine_threshold = r_min_per_receiver
                else:
                    refine_threshold = r_min_scalar
                if refine_threshold is None:
                    warnings.warn(
                        "No scalar or per-receiver r_min available for refinement; "
                        "falling back to 0.5.",
                        UserWarning,
                        stacklevel=2,
                    )
                    refine_threshold = 0.5
                selected_local_idx, selection_summary = select_feasible_subset_refined(
                    rates=rate_window,
                    target_size=target_k,
                    r_min=refine_threshold,
                    feasibility_tolerance=subset_feasibility_tolerance,
                    num_bottleneck_nodes=subset_bottleneck_nodes,
                    objective=refine_objective,
                )
            else:
                selected_local_idx = _select_evenly_spaced_indices(power_window.shape[0], target_k)
                selection_summary: Dict[str, Any] = {
                    "selection_strategy": "evenly_spaced",
                    "target_size": int(target_k),
                    "selected_size": int(selected_local_idx.size),
                    "is_feasible": None,
                }
        else:
            selected_local_idx = np.arange(power_window.shape[0], dtype=np.int64)
            selection_summary = {
                "selection_strategy": "all_samples",
                "target_size": int(target_k),
                "selected_size": int(selected_local_idx.size),
                "is_feasible": None,
            }

        power_selected = power_window[selected_local_idx]
        rates_selected = rate_window[selected_local_idx] if rate_window is not None else None
        steps_selected = steps_window[selected_local_idx]

        selected_power_stats = _summarize_power_samples(power_selected)
        selected_all_power_across_networks.append(power_selected.reshape(-1))

        selected_rate_stats: Optional[Dict[str, Any]] = None
        if rates_selected is not None and rates_selected.size > 0:
            rates_selected = np.asarray(rates_selected, dtype=np.float64)
            avg_rates_selected = rates_selected.mean(axis=0)
            if r_min_per_receiver is not None and r_min_per_receiver.shape[0] == rates_selected.shape[1]:
                stochastic_thresholds = r_min_per_receiver
                all_thresholds = np.broadcast_to(
                    r_min_per_receiver[None, :], rates_selected.shape
                ).reshape(-1)
            else:
                stochastic_thresholds = r_min_scalar
                all_thresholds = r_min_scalar
            selected_rate_stats = {
                "stochastic_avg_receiver_rates": _summarize_rate_array(
                    avg_rates_selected, r_min=stochastic_thresholds
                ),
                "all_selected_receiver_rates": _summarize_rate_array(
                    rates_selected.reshape(-1), r_min=all_thresholds
                ),
                "num_selected_samples": int(rates_selected.shape[0]),
                "num_receivers": int(rates_selected.shape[1]) if rates_selected.ndim == 2 else int(avg_rates_selected.size),
            }
            selected_avg_rates_across_networks.append(avg_rates_selected)
            selected_all_rates_across_networks.append(rates_selected.reshape(-1))

        n_links = int(associations.shape[0])
        deployment_range_value = float(channel_params_cfg.get("deployment_range", get_config_param(config, "deployment_range", 1000.0)))

        channel, effective_channel_version = _reconstruct_channel(
            net_id=net_id,
            network_seed=int(network_seed),
            n_links=n_links,
            channel_version=channel_version,
            channel_params_cfg=channel_params_cfg,
            channel_cache=channel_cache,
        )
        h_ls = np.asarray(channel.large_scale_fading, dtype=np.float64)
        tx_locations = np.asarray(channel.tx_locations, dtype=np.float64)
        rx_locations = np.asarray(channel.rx_locations, dtype=np.float64)
        deployment_range_value = float(channel.deployment_range)

        if h_inst_dir is not None:
            total_h_inst_bytes += _save_network_h_instantaneous(
                h_inst_dir=h_inst_dir,
                net_id=net_id,
                network_seed=int(network_seed),
                channel=channel,
                channel_version=effective_channel_version,
                num_timesteps=resolved_h_ts,
                dtype=h_inst_dtype,
            )

        converted_samples: List[Dict[str, Any]] = []
        for i in range(power_selected.shape[0]):
            power_tx = np.asarray(power_selected[i], dtype=np.float64)
            power_rx = associations.T @ power_tx
            rates_i = None
            if rates_selected is not None and i < rates_selected.shape[0]:
                rates_i = np.asarray(rates_selected[i], dtype=np.float64)
            converted_samples.append({
                "power_per_transmitter": power_tx,
                "power_per_receiver": power_rx,
                "rates": rates_i,
                "source_step": int(steps_selected[i]),
            })

        net_key = f"network_{net_id}"
        save_dict[net_key] = {
            "network_id": int(net_id),
            "network_seed": int(network_seed),
            "channel_version": effective_channel_version,
            "H_ls": h_ls,
            "associations": associations,
            "power_samples": converted_samples,
            "sample_steps": steps_selected,
        }
        if r_min_per_receiver is not None:
            save_dict[net_key]["r_min_per_receiver"] = r_min_per_receiver
        if base_network_id is not None:
            save_dict[net_key]["base_network_id"] = int(base_network_id)
        if constraint_profile_id is not None:
            save_dict[net_key]["constraint_profile_id"] = int(constraint_profile_id)
        if constraint_profile_name is not None:
            save_dict[net_key]["constraint_profile_name"] = str(constraint_profile_name)
        network_info[net_key] = {
            "network_seed": int(network_seed),
            "n_links": n_links,
            "deployment_range": deployment_range_value,
            "tx_locations": tx_locations.tolist(),
            "rx_locations": rx_locations.tolist(),
            "top_k": int(get_config_param(config, "top_k", 10)),
            "min_degree": int(get_config_param(config, "min_degree", 1)),
            "normalize_method": str(get_config_param(config, "normalize_method", "log1p")),
            "channel_version": effective_channel_version,
            "channel_params": channel_params_cfg,
            "system_params": system_params,
            "num_samples_source": int(total_samples),
            "num_samples_selected": int(power_selected.shape[0]),
            "selection_summary": selection_summary,
            "selected_power_stats": selected_power_stats,
            "selected_rate_stats": selected_rate_stats,
        }
        if r_min_per_receiver is not None:
            network_info[net_key]["r_min_per_receiver"] = r_min_per_receiver.tolist()
        if base_network_id is not None:
            network_info[net_key]["base_network_id"] = int(base_network_id)
        if constraint_profile_id is not None:
            network_info[net_key]["constraint_profile_id"] = int(constraint_profile_id)
        if constraint_profile_name is not None:
            network_info[net_key]["constraint_profile_name"] = str(constraint_profile_name)
        processed_network_ids.append(int(net_id))
        selected_counts.append(int(power_selected.shape[0]))

    if not processed_network_ids:
        raise RuntimeError("No valid networks were processed during conversion.")

    save_dict["network_ids"] = np.asarray(processed_network_ids, dtype=np.int64)

    if selected_all_power_across_networks:
        network_info["final_dataset_power_stats"] = _summarize_power_samples(
            np.concatenate(selected_all_power_across_networks)
        )

    if selected_avg_rates_across_networks:
        global_stochastic_rates = np.concatenate(selected_avg_rates_across_networks, axis=0)
        global_all_selected_rates = np.concatenate(selected_all_rates_across_networks, axis=0)
        global_r_min_threshold = None if has_per_network_r_min else r_min_scalar
        network_info["final_dataset_rate_stats"] = {
            "stochastic_avg_receiver_rates": _summarize_rate_array(
                global_stochastic_rates, r_min=global_r_min_threshold
            ),
            "all_selected_receiver_rates": _summarize_rate_array(
                global_all_selected_rates, r_min=global_r_min_threshold
            ),
            "num_networks_with_rates": int(len(selected_avg_rates_across_networks)),
        }
    else:
        network_info["final_dataset_rate_stats"] = {
            "available": False,
            "message": "Rate statistics unavailable because selected samples do not include rates.",
        }
    network_info["has_per_network_r_min"] = bool(has_per_network_r_min)
    network_info["constraint_mode"] = "per_network_r_min" if has_per_network_r_min else "scalar"

    if h_inst_dir is not None:
        h_inst_metadata = {
            "generated": True,
            "num_timesteps": int(resolved_h_ts),
            "dtype": str(h_inst_dtype),
            "shape_convention": "H[T,m,n]",
            "subdir": "h_instantaneous",
            "estimated_uncompressed_gb": float(total_h_inst_bytes / 1e9),
        }
        with open(h_inst_dir / "metadata.json", "w") as f:
            json.dump(h_inst_metadata, f, indent=2)
        network_info["h_instantaneous"] = h_inst_metadata
    else:
        network_info["h_instantaneous"] = {"generated": False}

    np.savez_compressed(raw_dir / "collected_samples.npz", **save_dict)
    shutil.copy(config_path, raw_dir / "config.yaml")
    with open(raw_dir / "network_info.json", "w") as f:
        json.dump(network_info, f, indent=2)

    _print_conversion_summary(
        raw_dir=raw_dir,
        processed_network_ids=processed_network_ids,
        skipped_networks=skipped_networks,
        selected_counts=selected_counts,
        network_info=network_info,
    )
    return raw_dir


def convert_samples(
    *,
    input_path: str,
    sample_source: str = "npz",
    output_root: str,
    output_raw_wra_dir: Optional[str] = None,
    force: bool = False,
    target_samples_per_network: Optional[int] = None,
    pd_collection_key: str,
    window_size: Optional[int] = None,
    refine_feasible_subset: bool = False,
    refine_objective: str = "min_rate",
    subset_feasibility_tolerance: float = 0.0,
    subset_bottleneck_nodes: int = 5,
    h_instantaneous_timesteps: Optional[int] = None,
    h_instantaneous_dtype: str = "float32",
) -> Path:
    """Dispatch to the appropriate collection pipeline based on sample_source.

    Args:
        sample_source: ``"npz"`` loads from the Polyak-checkpoint NPZ file
            (no windowing or greedy refinement). ``"primal_history"`` loads from
            the raw JSONL trajectory and supports both windowing and greedy
            feasibility/diversity refinement.
        h_instantaneous_timesteps: If None, skip H_instantaneous generation.
            If 0, read num_timesteps from the training config. If positive, use
            that many timesteps.
        h_instantaneous_dtype: Stored dtype for H tensors (float16/float32/float64).
    """
    source = str(sample_source).strip().lower()
    if source == "npz":
        return _collect_from_npz(
            input_path=input_path,
            output_root=output_root,
            output_raw_wra_dir=output_raw_wra_dir,
            force=force,
            target_samples_per_network=target_samples_per_network,
            pd_collection_key=pd_collection_key,
            h_instantaneous_timesteps=h_instantaneous_timesteps,
            h_instantaneous_dtype=h_instantaneous_dtype,
        )
    if source == "primal_history":
        return _collect_from_primal_history(
            input_path=input_path,
            output_root=output_root,
            output_raw_wra_dir=output_raw_wra_dir,
            force=force,
            target_samples_per_network=target_samples_per_network,
            pd_collection_key=pd_collection_key,
            window_size=window_size,
            refine_feasible_subset=refine_feasible_subset,
            refine_objective=refine_objective,
            subset_feasibility_tolerance=subset_feasibility_tolerance,
            subset_bottleneck_nodes=subset_bottleneck_nodes,
            h_instantaneous_timesteps=h_instantaneous_timesteps,
            h_instantaneous_dtype=h_instantaneous_dtype,
        )
    raise ValueError(
        f"Unsupported sample_source '{sample_source}'. Use 'npz' or 'primal_history'."
    )


def _cfg_get_optional(cfg: DictConfig, key: str, default: Any = None) -> Any:
    """Return ``cfg[key]`` unless the key is absent or MISSING."""
    if key not in cfg or OmegaConf.is_missing(cfg, key):
        return default
    return cfg[key]


@hydra.main(version_base=None, config_path="../../conf/wra_generation", config_name="pd_collection/wra_small")
def main(cfg: DictConfig) -> None:
    if OmegaConf.is_missing(cfg, "input_dir"):
        raise ValueError(
            "Missing required Hydra override: `input_dir`. "
            "Example: python -m graph_signal_diffusion.cli.wra.build_diffusion_dataset "
            "--config-name=pd_collection/wra_medium input_dir=/path/to/pd_run"
        )

    raw_wra_dir = _cfg_get_optional(cfg.output, "raw_wra_dir")
    ph = cfg.collection.primal_history
    h_inst_cfg = cfg.output.h_instantaneous
    h_inst_enabled = bool(h_inst_cfg.enabled)
    if h_inst_enabled:
        raw_h_ts = _cfg_get_optional(h_inst_cfg, "num_timesteps")
        h_ts = 0 if raw_h_ts is None else int(raw_h_ts)
        h_dtype = str(h_inst_cfg.dtype)
    else:
        h_ts = None
        h_dtype = "float32"
    convert_samples(
        input_path=hydra.utils.to_absolute_path(str(cfg.input_dir)),
        sample_source=str(cfg.collection.sample_source),
        output_root=hydra.utils.to_absolute_path(str(cfg.output.root)),
        output_raw_wra_dir=hydra.utils.to_absolute_path(str(raw_wra_dir)) if raw_wra_dir else None,
        force=bool(_cfg_get_optional(cfg.output, "force", False)),
        target_samples_per_network=_cfg_get_optional(cfg.collection, "target_samples_per_network"),
        pd_collection_key=str(cfg.pd_collection_key),
        window_size=_cfg_get_optional(ph, "window_size"),
        refine_feasible_subset=bool(ph.refine_feasible_subset),
        refine_objective=str(ph.refine_objective),
        subset_feasibility_tolerance=float(ph.subset_feasibility_tolerance),
        subset_bottleneck_nodes=int(ph.subset_bottleneck_nodes),
        h_instantaneous_timesteps=h_ts,
        h_instantaneous_dtype=h_dtype,
    )


if __name__ == "__main__":
    main()
