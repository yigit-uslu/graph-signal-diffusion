"""Schema adapters for primal-dual and converted WRA sample artifacts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import numpy as np

from .channel_factory import normalize_channel_version


PD_TRAJECTORY_SCHEMA_VERSION = 1
RAW_WRA_SCHEMA_VERSION = 1
PD_NUMERIC_DTYPE = np.float32


def _maybe_item(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    return value


def _to_numpy(value: Any, dtype: Optional[np.dtype] = None) -> np.ndarray:
    arr = np.asarray(_maybe_item(value))
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def _to_2d(array_like: Any, dtype: np.dtype = PD_NUMERIC_DTYPE) -> np.ndarray:
    arr = _to_numpy(array_like, dtype=dtype)
    if arr.ndim == 0:
        return np.empty((0, 0), dtype=dtype)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def _object_array(values: Iterable[Any]) -> np.ndarray:
    values = list(values)
    out = np.empty(len(values), dtype=object)
    for i, value in enumerate(values):
        out[i] = value
    return out


def _infer_channel_version(config: Optional[Mapping[str, Any]], default: str = "v2") -> str:
    if not config:
        return default
    if "channel_version" in config:
        return normalize_channel_version(config.get("channel_version"), default=default)
    channel_cfg = config.get("channel", {}) if isinstance(config.get("channel", {}), Mapping) else {}
    return normalize_channel_version(channel_cfg.get("version"), default=default)


def _parse_sample_list_from_network_dict(network_data: Mapping[str, Any]) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    powers: list[np.ndarray] = []
    rates: list[np.ndarray] = []
    steps: list[int] = []

    sample_list = network_data.get("power_samples", [])
    if isinstance(sample_list, np.ndarray):
        sample_list = _maybe_item(sample_list)

    if isinstance(sample_list, np.ndarray) and sample_list.dtype != object:
        powers_arr = _to_2d(sample_list)
        rate_samples = network_data.get("rate_samples")
        rates_arr = _to_2d(rate_samples) if rate_samples is not None else None
        return powers_arr, rates_arr, None

    if isinstance(sample_list, list):
        for idx, sample in enumerate(sample_list):
            if isinstance(sample, Mapping):
                if "power" in sample and sample["power"] is not None:
                    powers.append(_to_numpy(sample["power"], dtype=PD_NUMERIC_DTYPE))
                if "rates" in sample and sample["rates"] is not None:
                    rates.append(_to_numpy(sample["rates"], dtype=PD_NUMERIC_DTYPE))
                sample_step = sample.get("checkpoint_epoch", sample.get("sample_step", sample.get("step", idx)))
                steps.append(int(sample_step))
            elif sample is not None:
                powers.append(_to_numpy(sample, dtype=PD_NUMERIC_DTYPE))
                steps.append(idx)

    rate_samples = network_data.get("rate_samples")
    if len(rates) == 0 and rate_samples is not None:
        rate_samples_arr = _to_2d(rate_samples)
        if rate_samples_arr.shape[0] == len(powers):
            rates = [rate_samples_arr[i] for i in range(rate_samples_arr.shape[0])]

    power_arr = np.stack(powers, axis=0) if powers else np.empty((0, 0), dtype=PD_NUMERIC_DTYPE)
    rate_arr: Optional[np.ndarray]
    if rates and len(rates) == len(powers):
        rate_arr = np.stack(rates, axis=0)
    else:
        rate_arr = None

    step_arr = np.asarray(steps, dtype=np.int64) if steps and len(steps) == len(powers) else None
    return power_arr, rate_arr, step_arr


def canonicalize_pd_samples_dict(
    raw_samples: Mapping[str, Any],
    *,
    config: Optional[Mapping[str, Any]] = None,
    default_channel_version: str = "v2",
) -> Dict[str, Any]:
    """Normalize primal-dual sample files (legacy or canonical) into one in-memory format."""
    raw = {k: _maybe_item(v) for k, v in raw_samples.items()}

    schema_version_raw = raw.get("schema_version")
    if schema_version_raw is not None:
        try:
            schema_version = int(_maybe_item(schema_version_raw))
        except (TypeError, ValueError):
            schema_version = -1
    else:
        schema_version = -1

    if schema_version >= PD_TRAJECTORY_SCHEMA_VERSION and "power_samples_per_network" in raw:
        network_ids = _to_numpy(raw.get("network_ids"), dtype=np.int64).tolist()
        network_seeds = _to_numpy(raw.get("network_seeds", np.full(len(network_ids), -1)), dtype=np.int64)
        associations_arr = _to_numpy(raw.get("associations"), dtype=object)
        has_h_instantaneous = "H_instantaneous" in raw
        h_arr = _to_numpy(raw.get("H_instantaneous"), dtype=object) if has_h_instantaneous else None
        power_arr = _to_numpy(raw.get("power_samples_per_network"), dtype=object)
        rate_arr = _to_numpy(raw.get("rate_samples_per_network"), dtype=object) if "rate_samples_per_network" in raw else None
        steps_arr = _to_numpy(raw.get("sample_steps_per_network"), dtype=object) if "sample_steps_per_network" in raw else None
        r_min_arr = _to_numpy(raw.get("r_min_per_receiver_per_network"), dtype=object) if "r_min_per_receiver_per_network" in raw else None
        base_ids_arr = _to_numpy(raw.get("base_network_ids"), dtype=np.int64) if "base_network_ids" in raw else None
        profile_ids_arr = _to_numpy(raw.get("constraint_profile_ids"), dtype=np.int64) if "constraint_profile_ids" in raw else None
        profile_names_arr = _to_numpy(raw.get("constraint_profile_names"), dtype=object) if "constraint_profile_names" in raw else None

        channel_version = normalize_channel_version(raw.get("channel_version"), default=default_channel_version)
        seed_start = None
        if config is not None and isinstance(config, Mapping):
            seed_start = config.get("seed")
        networks: Dict[int, Dict[str, Any]] = {}
        for i, net_id in enumerate(network_ids):
            seed_i = int(network_seeds[i]) if i < len(network_seeds) else -1
            network_seed = None if seed_i < 0 else seed_i
            if network_seed is None and seed_start is not None:
                network_seed = int(seed_start) + int(net_id)
            powers_i = _to_2d(power_arr[i], dtype=PD_NUMERIC_DTYPE)
            rates_i = _to_2d(rate_arr[i], dtype=PD_NUMERIC_DTYPE) if rate_arr is not None else None
            if rates_i is not None and rates_i.shape[0] != powers_i.shape[0]:
                rates_i = None
            steps_i = _to_numpy(steps_arr[i], dtype=np.int64) if steps_arr is not None else None
            if steps_i is not None and steps_i.shape[0] != powers_i.shape[0]:
                steps_i = None
            associations_i = _to_numpy(associations_arr[i], dtype=PD_NUMERIC_DTYPE)
            if has_h_instantaneous:
                h_i = _to_numpy(h_arr[i], dtype=PD_NUMERIC_DTYPE)
            elif associations_i.ndim == 2:
                h_i = np.empty((0, associations_i.shape[0], associations_i.shape[1]), dtype=PD_NUMERIC_DTYPE)
            else:
                h_i = np.empty((0, 0, 0), dtype=PD_NUMERIC_DTYPE)

            networks[int(net_id)] = {
                "network_seed": network_seed,
                "H_instantaneous": h_i,
                "associations": associations_i,
                "power_samples": powers_i,
                "rate_samples": rates_i,
                "sample_steps": steps_i,
            }
            if r_min_arr is not None and i < len(r_min_arr):
                networks[int(net_id)]["r_min_per_receiver"] = _to_numpy(
                    r_min_arr[i], dtype=PD_NUMERIC_DTYPE
                ).reshape(-1)
            if base_ids_arr is not None and i < len(base_ids_arr):
                base_id = int(base_ids_arr[i])
                if base_id >= 0:
                    networks[int(net_id)]["base_network_id"] = base_id
            if profile_ids_arr is not None and i < len(profile_ids_arr):
                profile_id = int(profile_ids_arr[i])
                if profile_id >= 0:
                    networks[int(net_id)]["constraint_profile_id"] = profile_id
            if profile_names_arr is not None and i < len(profile_names_arr):
                profile_name = _maybe_item(profile_names_arr[i])
                if profile_name is not None and str(profile_name) != "":
                    networks[int(net_id)]["constraint_profile_name"] = str(profile_name)

        return {
            "schema_version": PD_TRAJECTORY_SCHEMA_VERSION,
            "channel_version": channel_version,
            "network_ids": sorted(networks.keys()),
            "networks": networks,
            "source_format": "canonical_v2",
            "source_has_channel_version": True,
            "source_has_h_instantaneous": has_h_instantaneous,
        }

    # Legacy adapters
    pattern_base = re.compile(r"^network_(\d+)$")
    pattern_h = re.compile(r"^network_(\d+)_H_instantaneous$")
    pattern_assoc = re.compile(r"^network_(\d+)_associations$")
    pattern_seed = re.compile(r"^network_(\d+)_seed$")
    pattern_power_idx = re.compile(r"^network_(\d+)_power_(\d+)$")
    pattern_rate_idx = re.compile(r"^network_(\d+)_rates_(\d+)$")
    pattern_power_samples = re.compile(r"^network_(\d+)_power_samples$")
    pattern_rate_samples = re.compile(r"^network_(\d+)_rate_samples$")
    pattern_step_samples = re.compile(r"^network_(\d+)_sample_steps$")

    state: Dict[int, Dict[str, Any]] = {}

    def get_state(net_id: int) -> Dict[str, Any]:
        if net_id not in state:
            state[net_id] = {
                "network_seed": None,
                "H_instantaneous": None,
                "associations": None,
                "power_by_idx": {},
                "rate_by_idx": {},
                "step_by_idx": {},
            }
        return state[net_id]

    for key, value in raw.items():
        if not key.startswith("network_"):
            continue

        if m := pattern_base.match(key):
            net_id = int(m.group(1))
            st = get_state(net_id)
            maybe_dict = _maybe_item(value)
            if isinstance(maybe_dict, Mapping):
                if maybe_dict.get("network_seed") is not None:
                    st["network_seed"] = int(maybe_dict["network_seed"])
                if maybe_dict.get("H_instantaneous") is not None:
                    st["H_instantaneous"] = _to_numpy(maybe_dict["H_instantaneous"], dtype=PD_NUMERIC_DTYPE)
                if maybe_dict.get("associations") is not None:
                    st["associations"] = _to_numpy(maybe_dict["associations"], dtype=PD_NUMERIC_DTYPE)
                if maybe_dict.get("r_min_per_receiver") is not None:
                    st["r_min_per_receiver"] = _to_numpy(
                        maybe_dict["r_min_per_receiver"], dtype=PD_NUMERIC_DTYPE
                    ).reshape(-1)
                if maybe_dict.get("base_network_id") is not None:
                    st["base_network_id"] = int(maybe_dict["base_network_id"])
                if maybe_dict.get("constraint_profile_id") is not None:
                    st["constraint_profile_id"] = int(maybe_dict["constraint_profile_id"])
                if maybe_dict.get("constraint_profile_name") is not None:
                    st["constraint_profile_name"] = str(maybe_dict["constraint_profile_name"])

                power_arr, rates_arr, steps_arr = _parse_sample_list_from_network_dict(maybe_dict)
                for idx in range(power_arr.shape[0]):
                    st["power_by_idx"][idx] = power_arr[idx]
                    if rates_arr is not None:
                        st["rate_by_idx"][idx] = rates_arr[idx]
                    if steps_arr is not None:
                        st["step_by_idx"][idx] = int(steps_arr[idx])
            continue

        if m := pattern_h.match(key):
            get_state(int(m.group(1)))["H_instantaneous"] = _to_numpy(value, dtype=PD_NUMERIC_DTYPE)
            continue

        if m := pattern_assoc.match(key):
            get_state(int(m.group(1)))["associations"] = _to_numpy(value, dtype=PD_NUMERIC_DTYPE)
            continue

        if m := pattern_seed.match(key):
            get_state(int(m.group(1)))["network_seed"] = int(_maybe_item(value))
            continue

        if m := pattern_power_idx.match(key):
            net_id = int(m.group(1))
            idx = int(m.group(2))
            get_state(net_id)["power_by_idx"][idx] = _to_numpy(value, dtype=PD_NUMERIC_DTYPE)
            continue

        if m := pattern_rate_idx.match(key):
            net_id = int(m.group(1))
            idx = int(m.group(2))
            get_state(net_id)["rate_by_idx"][idx] = _to_numpy(value, dtype=PD_NUMERIC_DTYPE)
            continue

        if m := pattern_power_samples.match(key):
            net_id = int(m.group(1))
            st = get_state(net_id)
            arr = _to_2d(value, dtype=PD_NUMERIC_DTYPE)
            for idx in range(arr.shape[0]):
                st["power_by_idx"][idx] = arr[idx]
            continue

        if m := pattern_rate_samples.match(key):
            net_id = int(m.group(1))
            st = get_state(net_id)
            arr = _to_2d(value, dtype=PD_NUMERIC_DTYPE)
            for idx in range(arr.shape[0]):
                st["rate_by_idx"][idx] = arr[idx]
            continue

        if m := pattern_step_samples.match(key):
            net_id = int(m.group(1))
            st = get_state(net_id)
            arr = _to_numpy(value, dtype=np.int64)
            for idx in range(arr.shape[0]):
                st["step_by_idx"][idx] = int(arr[idx])
            continue

    if len(state) == 0:
        raise ValueError("No network_* entries were found in collected_samples file.")

    inferred_channel_version = _infer_channel_version(config, default=default_channel_version)
    seed_start = None
    if config is not None:
        seed_start = config.get("seed") if isinstance(config, Mapping) else None

    networks: Dict[int, Dict[str, Any]] = {}
    for net_id in sorted(state.keys()):
        st = state[net_id]
        if st["H_instantaneous"] is None or st["associations"] is None:
            raise ValueError(f"Network {net_id} missing required H_instantaneous or associations.")

        indices = sorted(set(st["power_by_idx"].keys()) | set(st["rate_by_idx"].keys()))
        powers = [st["power_by_idx"][idx] for idx in indices if idx in st["power_by_idx"]]
        power_arr = np.stack(powers, axis=0) if powers else np.empty((0, 0), dtype=PD_NUMERIC_DTYPE)

        if indices and all(idx in st["rate_by_idx"] for idx in indices):
            rate_arr = np.stack([st["rate_by_idx"][idx] for idx in indices], axis=0)
        else:
            rate_arr = None

        if indices and all(idx in st["step_by_idx"] for idx in indices):
            step_arr = np.asarray([st["step_by_idx"][idx] for idx in indices], dtype=np.int64)
        else:
            step_arr = None

        network_seed = st["network_seed"]
        if network_seed is None and seed_start is not None:
            network_seed = int(seed_start) + int(net_id)

        networks[net_id] = {
            "network_seed": network_seed,
            "H_instantaneous": st["H_instantaneous"],
            "associations": st["associations"],
            "power_samples": power_arr,
            "rate_samples": rate_arr,
            "sample_steps": step_arr,
        }
        if st.get("r_min_per_receiver") is not None:
            networks[net_id]["r_min_per_receiver"] = _to_numpy(
                st["r_min_per_receiver"], dtype=PD_NUMERIC_DTYPE
            ).reshape(-1)
        if st.get("base_network_id") is not None:
            networks[net_id]["base_network_id"] = int(st["base_network_id"])
        if st.get("constraint_profile_id") is not None:
            networks[net_id]["constraint_profile_id"] = int(st["constraint_profile_id"])
        if st.get("constraint_profile_name") is not None:
            networks[net_id]["constraint_profile_name"] = str(st["constraint_profile_name"])

    return {
        "schema_version": PD_TRAJECTORY_SCHEMA_VERSION,
        "channel_version": inferred_channel_version,
        "network_ids": sorted(networks.keys()),
        "networks": networks,
        "source_format": "legacy_adapted",
        "source_has_channel_version": False,
    }


def load_pd_samples_npz(
    samples_path: str | Path,
    *,
    config: Optional[Mapping[str, Any]] = None,
    default_channel_version: str = "v2",
) -> Dict[str, Any]:
    """Load and canonicalize primal-dual sample file."""
    samples_path = Path(samples_path)
    with np.load(samples_path, allow_pickle=True) as npz:
        raw = {k: npz[k] for k in npz.files}
    return canonicalize_pd_samples_dict(raw, config=config, default_channel_version=default_channel_version)


def build_pd_samples_npz_payload(
    samples_by_network: Mapping[int, Mapping[str, Any]],
    *,
    channel_version: Optional[str] = None,
    include_h_instantaneous: bool = False,
) -> Dict[str, Any]:
    """Build canonical v2 primal-dual NPZ payload from in-memory sample dicts."""
    network_ids = sorted(int(k) for k in samples_by_network.keys())

    seeds: list[int] = []
    associations: list[np.ndarray] = []
    h_inst: list[np.ndarray] = []
    power_samples: list[np.ndarray] = []
    rate_samples: list[np.ndarray] = []
    sample_steps: list[np.ndarray] = []
    r_min_vectors: list[np.ndarray] = []
    base_network_ids: list[int] = []
    constraint_profile_ids: list[int] = []
    constraint_profile_names: list[Any] = []

    has_any_r_min_vector = False
    has_any_base_network_id = False
    has_any_constraint_profile_id = False
    has_any_constraint_profile_name = False

    for net_id in network_ids:
        network_data = samples_by_network[net_id]

        seed = network_data.get("network_seed")
        seeds.append(int(seed) if seed is not None else -1)

        associations.append(_to_numpy(network_data["associations"], dtype=PD_NUMERIC_DTYPE))
        if include_h_instantaneous:
            if "H_instantaneous" not in network_data:
                raise ValueError("include_h_instantaneous=True but network_data is missing H_instantaneous")
            h_inst.append(_to_numpy(network_data["H_instantaneous"], dtype=PD_NUMERIC_DTYPE))

        powers_arr, rates_arr, steps_arr = _parse_sample_list_from_network_dict(network_data)
        power_samples.append(powers_arr)

        if rates_arr is None:
            rates_arr = np.empty((powers_arr.shape[0], 0), dtype=PD_NUMERIC_DTYPE)
        rate_samples.append(rates_arr)

        if steps_arr is None:
            steps_arr = np.arange(powers_arr.shape[0], dtype=np.int64)
        sample_steps.append(steps_arr)

        r_min_vec = network_data.get("r_min_per_receiver")
        if r_min_vec is None:
            r_min_vectors.append(np.empty((0,), dtype=PD_NUMERIC_DTYPE))
        else:
            has_any_r_min_vector = True
            r_min_vectors.append(_to_numpy(r_min_vec, dtype=PD_NUMERIC_DTYPE).reshape(-1))

        base_network_id = network_data.get("base_network_id")
        if base_network_id is None:
            base_network_ids.append(-1)
        else:
            has_any_base_network_id = True
            base_network_ids.append(int(base_network_id))

        constraint_profile_id = network_data.get("constraint_profile_id")
        if constraint_profile_id is None:
            constraint_profile_ids.append(-1)
        else:
            has_any_constraint_profile_id = True
            constraint_profile_ids.append(int(constraint_profile_id))

        constraint_profile_name = network_data.get("constraint_profile_name")
        if constraint_profile_name is None:
            constraint_profile_names.append("")
        else:
            has_any_constraint_profile_name = True
            constraint_profile_names.append(str(constraint_profile_name))

    payload = {
        "schema_version": np.array(PD_TRAJECTORY_SCHEMA_VERSION, dtype=np.int64),
        "channel_version": np.array(normalize_channel_version(channel_version, default="v2"), dtype="<U2"),
        "network_ids": np.asarray(network_ids, dtype=np.int64),
        "network_seeds": np.asarray(seeds, dtype=np.int64),
        "associations": _object_array(associations),
        "power_samples_per_network": _object_array(power_samples),
        "rate_samples_per_network": _object_array(rate_samples),
        "sample_steps_per_network": _object_array(sample_steps),
    }
    if include_h_instantaneous:
        payload["H_instantaneous"] = _object_array(h_inst)
    if has_any_r_min_vector:
        payload["r_min_per_receiver_per_network"] = _object_array(r_min_vectors)
    if has_any_base_network_id:
        payload["base_network_ids"] = np.asarray(base_network_ids, dtype=np.int64)
    if has_any_constraint_profile_id:
        payload["constraint_profile_ids"] = np.asarray(constraint_profile_ids, dtype=np.int64)
    if has_any_constraint_profile_name:
        payload["constraint_profile_names"] = _object_array(constraint_profile_names)
    return payload


def save_pd_samples_npz(
    output_path: str | Path,
    samples_by_network: Mapping[int, Mapping[str, Any]],
    *,
    channel_version: Optional[str] = None,
    include_h_instantaneous: bool = False,
) -> Path:
    """Write canonical v2 primal-dual sample artifact to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_pd_samples_npz_payload(
        samples_by_network,
        channel_version=channel_version,
        include_h_instantaneous=include_h_instantaneous,
    )
    np.savez_compressed(output_path, **payload)
    return output_path
