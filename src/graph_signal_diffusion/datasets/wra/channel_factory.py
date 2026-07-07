"""Shared channel factory utilities for WRA pipelines."""

from __future__ import annotations

import hashlib
import json
import math
import ast
from typing import Any, Dict, Mapping, Optional, Tuple, Type

from .channel import WirelessChannel, WirelessChannelV2, WirelessChannelV3


CHANNEL_VERSION_ALIASES: Dict[str, str] = {
    "v1": "v1",
    "1": "v1",
    "wirelesschannel": "v1",
    "default": "v1",
    "v2": "v2",
    "2": "v2",
    "wirelesschannelv2": "v2",
    "v3": "v3",
    "3": "v3",
    "wirelesschannelv3": "v3",
}

CHANNEL_PARAM_KEYS = {
    "deployment_range",
    "min_tx_tx_distance",
    "min_tx_rx_distance",
    "max_tx_rx_distance",
    "path_loss_exponent_short",
    "path_loss_exponent_long",
    "shadowing_std",
    "carrier_freq",
    "speed",
    "num_fading_paths",
    "delta_t",
    "max_recursion_depth",
    "max_deployment_attempts",
}

# Parameters where an explicit YAML null carries meaning and must be preserved.
CHANNEL_PARAM_NULLABLE_KEYS = {
    "max_tx_rx_distance",  # null = no upper TX-RX distance constraint
}

# Parameters that directly or indirectly affect spatial geometry and large-scale fading.
# We hash all constructor-relevant channel params for safety; this subset is a documented minimum.
GEOMETRY_LARGE_SCALE_CHANNEL_PARAM_KEYS = {
    "min_tx_tx_distance",
    "min_tx_rx_distance",
    "max_tx_rx_distance",
    "path_loss_exponent_short",
    "path_loss_exponent_long",
    "shadowing_std",
    "max_recursion_depth",
    "max_deployment_attempts",
}

CHANNEL_PARAM_DEFAULTS: Dict[str, Any] = {
    "min_tx_tx_distance": 35.0,
    "min_tx_rx_distance": 10.0,
    "max_tx_rx_distance": 50.0,
    "path_loss_exponent_short": 2.0,
    "path_loss_exponent_long": 4.0,
    "shadowing_std": 7.0,
    "carrier_freq": 2.4e9,
    "speed": 1.0,
    "num_fading_paths": 100,
    "delta_t": 0.01,  # 10 ms — matches scenario/defaults.yaml
    "max_deployment_attempts": 1000,
    "max_recursion_depth": 3,  # V3 only
}

CHANNEL_PARAM_FLOAT_KEYS = {
    "min_tx_tx_distance",
    "min_tx_rx_distance",
    "max_tx_rx_distance",
    "path_loss_exponent_short",
    "path_loss_exponent_long",
    "shadowing_std",
    "carrier_freq",
    "speed",
    "delta_t",
}

CHANNEL_PARAM_INT_KEYS = {
    "num_fading_paths",
    "max_recursion_depth",
    "max_deployment_attempts",
}

WRA_CHANNEL_CACHE_KEY_SCHEMA_VERSION = 1
WRA_PD_RUN_KEY_SCHEMA_VERSION = 1
WRA_PD_COLLECTION_KEY_SCHEMA_VERSION = 1


def normalize_channel_version(channel_version: Optional[Any], default: str = "v2") -> str:
    """Normalize channel version labels into {'v1','v2','v3'}.

    Parameters
    ----------
    channel_version : Any
        Raw channel version value from config/metadata.
    default : str
        Fallback channel version when value is missing.
    """
    if channel_version is None:
        return default
    key = str(channel_version).strip().lower()
    return CHANNEL_VERSION_ALIASES.get(key, default)


def get_channel_class(channel_version: Optional[Any], default: str = "v2") -> Type[WirelessChannel]:
    """Return channel class for a version string."""
    version = normalize_channel_version(channel_version, default=default)
    if version == "v3":
        return WirelessChannelV3
    if version == "v2":
        return WirelessChannelV2
    return WirelessChannel


def sanitize_channel_params(channel_params: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Filter channel params to constructor-supported keys."""
    if not channel_params:
        return {}
    return {
        k: v
        for k, v in dict(channel_params).items()
        if k in CHANNEL_PARAM_KEYS and (v is not None or k in CHANNEL_PARAM_NULLABLE_KEYS)
    }


def canonicalize_wra_channel_params(
    channel_params: Optional[Mapping[str, Any]],
    *,
    channel_version: Optional[Any],
    default_version: str = "v2",
    include_defaults: bool = True,
) -> Dict[str, Any]:
    """Canonicalize WRA channel constructor params for cache keys/metadata.

    Parameters
    ----------
    channel_params : mapping or None
        Raw channel parameters (usually from config or channel kwargs).
    channel_version : Any
        Channel version label; aliases are normalized.
    default_version : str
        Fallback version when missing.
    include_defaults : bool
        If True, fill omitted constructor params with current defaults so
        explicit-default and omitted-default configs hash the same.
    """
    version = normalize_channel_version(channel_version, default=default_version)
    params = sanitize_channel_params(channel_params)
    params.pop("deployment_range", None)  # Cache metadata stores this separately.

    if include_defaults:
        for key, value in CHANNEL_PARAM_DEFAULTS.items():
            if key == "max_recursion_depth" and version != "v3":
                continue
            params.setdefault(key, value)

    if version != "v3":
        params.pop("max_recursion_depth", None)

    canonical: Dict[str, Any] = {}
    for key in sorted(params):
        value = params[key]
        if key in CHANNEL_PARAM_FLOAT_KEYS and value is not None:
            canonical[key] = float(value)
        elif key in CHANNEL_PARAM_INT_KEYS and value is not None:
            canonical[key] = int(value)
        else:
            canonical[key] = value
    return canonical


def build_wra_channel_cache_metadata(
    *,
    seed: int,
    num_networks: int,
    n_links: int,
    deployment_range: float,
    channel_version: Optional[Any],
    channel_params: Optional[Mapping[str, Any]] = None,
    channel_config_name: Optional[str] = None,
    default_version: str = "v2",
) -> Dict[str, Any]:
    """Build canonical metadata + hashed cache key for WRA channel caches.

    The hash payload includes the full set of canonicalized channel constructor
    parameters (not just geometry/large-scale parameters) so cached channel
    objects remain unambiguous when temporal fading parameters differ.
    """
    normalized_channel_version = normalize_channel_version(channel_version, default=default_version)
    canonical_channel_params = canonicalize_wra_channel_params(
        channel_params,
        channel_version=normalized_channel_version,
        default_version=default_version,
        include_defaults=True,
    )

    base_metadata: Dict[str, Any] = {
        "n_links": int(n_links),
        "deployment_range": float(deployment_range),
        "channel_version": normalized_channel_version,
        "channel_params": canonical_channel_params,
        "num_networks": int(num_networks),
        "seed": int(seed),
    }

    hash_payload = {
        "schema_version": WRA_CHANNEL_CACHE_KEY_SCHEMA_VERSION,
        **base_metadata,
    }
    hash_str = json.dumps(hash_payload, sort_keys=True, separators=(",", ":"))
    hash_val = hashlib.md5(hash_str.encode()).hexdigest()[:12]

    # Keep the filename prefix short and readable; uniqueness comes from the hash.
    deployment_token = f"{float(deployment_range):g}"
    channel_cache_key = (
        f"wrach_v{WRA_CHANNEL_CACHE_KEY_SCHEMA_VERSION}"
        f"_s{int(seed)}_D{int(num_networks)}_N{int(n_links)}"
        f"_R{deployment_token}_{normalized_channel_version}_h{hash_val}"
    )

    metadata: Dict[str, Any] = {
        "channel_cache_key": channel_cache_key,
        "channel_cache_key_hash": hash_val,
        "channel_cache_key_schema_version": WRA_CHANNEL_CACHE_KEY_SCHEMA_VERSION,
        **base_metadata,
        # Backward-compatible top-level mirrors for older tooling/logging.
        "min_tx_rx_distance": canonical_channel_params.get("min_tx_rx_distance"),
        "max_tx_rx_distance": canonical_channel_params.get("max_tx_rx_distance"),
        "max_recursion_depth": canonical_channel_params.get("max_recursion_depth"),
    }
    if channel_config_name is not None:
        metadata["channel_config_name"] = channel_config_name
    return metadata


def build_wra_channel_cache_key(**kwargs: Any) -> str:
    """Return only the hashed WRA channel cache key."""
    return build_wra_channel_cache_metadata(**kwargs)["channel_cache_key"]


# ---------------------------------------------------------------------------
# Cache metadata canonicalization & comparison (shared by all WRA scripts)
# ---------------------------------------------------------------------------

def _as_int(value: Any) -> Any:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def _as_float(value: Any) -> Any:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def canonicalize_channel_cache_metadata(
    metadata: Any,
    default_channel_version: str = "v2",
) -> Optional[Dict[str, Any]]:
    """Normalize cache metadata into a comparable canonical form.

    Parameters
    ----------
    metadata : dict or Any
        Raw metadata dict loaded from a ``.pt`` cache file, or the output of
        :func:`build_wra_channel_cache_metadata`.
    default_channel_version : str
        Version to assume when ``channel_version`` is absent.

    Returns
    -------
    dict or None
        Canonicalized metadata dict, or ``None`` if *metadata* is not a dict.
    """
    if not isinstance(metadata, dict):
        return None

    channel_version = normalize_channel_version(
        metadata.get("channel_version"),
        default=default_channel_version,
    )

    # --- canonicalize nested channel_params ---
    raw_channel_params = metadata.get("channel_params")
    channel_params_source = (
        raw_channel_params if isinstance(raw_channel_params, dict) else metadata
    )
    params = sanitize_channel_params(channel_params_source)
    params.pop("deployment_range", None)  # stored separately in metadata
    if channel_version != "v3":
        params.pop("max_recursion_depth", None)

    canonical_params: Dict[str, Any] = {}
    for key in sorted(params):
        value = params[key]
        if key in CHANNEL_PARAM_FLOAT_KEYS and value is not None:
            canonical_params[key] = _as_float(value)
        elif key in CHANNEL_PARAM_INT_KEYS and value is not None:
            canonical_params[key] = _as_int(value)
        else:
            canonical_params[key] = value

    # --- top-level backward-compat mirrors ---
    min_tx_rx_distance = _as_float(metadata.get("min_tx_rx_distance"))
    max_tx_rx_distance = _as_float(metadata.get("max_tx_rx_distance"))
    if channel_version == "v3":
        raw_mrd = metadata.get("max_recursion_depth")
        max_recursion_depth: Any = 3 if raw_mrd is None else _as_int(raw_mrd)
    else:
        max_recursion_depth = None

    return {
        "channel_cache_key": metadata.get("channel_cache_key"),
        "channel_config_name": metadata.get("channel_config_name"),
        "n_links": _as_int(metadata.get("n_links")),
        "deployment_range": _as_float(metadata.get("deployment_range")),
        "channel_version": channel_version,
        "min_tx_rx_distance": min_tx_rx_distance,
        "max_tx_rx_distance": max_tx_rx_distance,
        "max_recursion_depth": max_recursion_depth,
        "channel_params": canonical_params,
        "num_networks": _as_int(metadata.get("num_networks")),
        "seed": _as_int(metadata.get("seed")),
    }


def find_channel_cache_metadata_mismatches(
    expected: Dict[str, Any],
    actual: Optional[Dict[str, Any]],
) -> Dict[str, Tuple[Any, Any]]:
    """Return ``{key: (expected_value, actual_value)}`` for mismatched fields.

    Keys that are purely informational (``channel_config_name``) are skipped.
    ``channel_cache_key`` mismatches are ignored when the *actual* value is
    ``None`` (backward compatibility with pre-hashing caches).
    """
    if actual is None:
        return {"metadata": (expected, None)}

    mismatches: Dict[str, Tuple[Any, Any]] = {}
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if key in {"channel_config_name"}:
            continue
        if key in {"channel_cache_key"} and actual_value is None:
            continue

        if isinstance(expected_value, float) or isinstance(actual_value, float):
            if expected_value is None or actual_value is None:
                if expected_value != actual_value:
                    mismatches[key] = (expected_value, actual_value)
            elif not math.isclose(
                float(expected_value), float(actual_value), rel_tol=0.0, abs_tol=1e-12
            ):
                mismatches[key] = (expected_value, actual_value)
        else:
            if expected_value != actual_value:
                mismatches[key] = (expected_value, actual_value)

    return mismatches


def _none_if_null(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"none", "null"}:
        return None
    return value


def _parse_maybe_literal(value: Any) -> Any:
    """Parse strings like '[1,2]' or '{"a":1}' into Python objects when possible."""
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if stripped == "":
        return value
    try:
        return json.loads(stripped)
    except Exception:
        pass
    try:
        return ast.literal_eval(stripped)
    except Exception:
        return value


def _canonicalize_sample_collection_interval(value: Any) -> Any:
    """
    Canonicalize sample_collection_interval for run-key hashing.

    Returns one of:
    - None
    - int (for finite integer-like values)
    - "inf" / "-inf" (for non-finite values)
    - float (fallback for finite non-integer numerics)
    """
    value = _none_if_null(value)
    if value is None:
        return None

    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"", "none", "null", "~"}:
            return None
        if token in {"inf", ".inf", "+inf", "+.inf"}:
            return "inf"
        if token in {"-inf", "-.inf"}:
            return "-inf"
        try:
            value = float(token)
        except ValueError as exc:
            raise ValueError(
                "training.sample_collection_interval must be numeric, null, or inf-like."
            ) from exc
    elif not isinstance(value, (int, float)):
        raise TypeError(
            "training.sample_collection_interval must be numeric, null, or inf-like."
        )

    value_f = float(value)
    if math.isinf(value_f):
        return "inf" if value_f > 0 else "-inf"
    if value_f.is_integer():
        return int(value_f)
    return value_f


def _canonicalize_explicit_profiles(raw_profiles: Any) -> list[Any]:
    """Canonicalize explicit constraint profiles for hashing payloads."""
    parsed = _parse_maybe_literal(_none_if_null(raw_profiles))
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(parsed):
            parsed = OmegaConf.to_container(parsed, resolve=True)
    except Exception:
        pass
    if parsed is None:
        return []
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(
            "constraint_profile.explicit.profiles must be a list of scalars or vectors; "
            f"got type {type(parsed)}."
        )
    canonical: list[Any] = []
    for idx, profile in enumerate(parsed):
        if isinstance(profile, (list, tuple)):
            canonical.append([float(v) for v in profile])
        else:
            try:
                canonical.append(float(profile))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Explicit profile at index {idx} must be numeric or numeric list; got {profile!r}."
                ) from exc
    return canonical


def build_wra_pd_run_key(
    *,
    channel_cache_key: str,
    seed: int,
    P_max_dBm: float,
    bandwidth_Hz: float,
    noise_psd_dBm_Hz: float,
    model_type: str,
    model_hidden_dim: int,
    model_num_layers: int,
    model_K: Optional[int],
    model_norm_type: str,
    model_dropout: float,
    model_activation: str,
    training_r_min: Optional[float],
    training_constraint_profile_type: str,
    training_constraint_profile_source: str,
    training_constraint_profile_scalar_value: Optional[float],
    training_constraint_profile_explicit_profiles: Optional[Any],
    training_constraint_profile_sampled_min: Optional[float],
    training_constraint_profile_sampled_max: Optional[float],
    training_constraint_profile_sampled_count: Optional[int],
    training_constraint_profile_sampled_seed: Optional[int],
    training_constraint_profile_r_min_feature_scale: float,
    training_alpha_dual: float,
    training_dual_momentum: float,
    training_learning_rate: float,
    training_batch_size: int,
    training_max_epochs: int,
    training_convergence_window: int,
    training_convergence_warmup_epochs: Optional[int],
    training_convergence_patience: int,
    training_gradient_norm_threshold: float,
    training_dual_variance_threshold: float,
    training_dual_stationarity_threshold: float,
    training_violation_fraction_threshold: float,
    training_violation_fraction_on_model_avg_rates_threshold: float,
    training_mean_violation_slack_on_model_avg_rates_threshold: float,
    training_dual_update_mode: str,
    training_dual_update_frequency: int,
    training_num_samples_per_network: int,
    training_moving_avg_window: int,
    training_sample_collection_interval: Any,
) -> str:
    """
    Build a hashed run key for primal-dual training outputs.

    Constraint-profile hashing is source-aware:
    - scalar: only scalar value is included
    - explicit: only explicit profile vectors are included
    - sampled: only sampled range/count/seed fields are included

    Inactive branches are excluded by design.
    """
    source = str(training_constraint_profile_source).strip().lower()
    profile_type = str(training_constraint_profile_type).strip().lower()
    if profile_type != "min_rate":
        raise ValueError(
            "Unsupported constraint_profile.type for PD run key hashing: "
            f"{training_constraint_profile_type!r}. Expected 'min_rate'."
        )

    scalar_value: Optional[float] = None
    constraint_payload: Dict[str, Any] = {
        "type": "min_rate",
        "source": source,
        "r_min_feature_scale": float(training_constraint_profile_r_min_feature_scale),
    }
    if source == "scalar":
        if training_constraint_profile_scalar_value is not None:
            scalar_value = float(training_constraint_profile_scalar_value)
        elif training_r_min is not None:
            scalar_value = float(training_r_min)
        else:
            raise ValueError(
                "Scalar constraint profile selected but neither training.r_min "
                "nor constraint_profile.scalar.value is provided."
            )
        constraint_payload["scalar"] = {"value": scalar_value}
    elif source == "explicit":
        explicit_profiles = _canonicalize_explicit_profiles(training_constraint_profile_explicit_profiles)
        if len(explicit_profiles) == 0:
            raise ValueError(
                "constraint_profile.source=explicit requires non-empty "
                "constraint_profile.explicit.profiles."
            )
        # Explicit names are cosmetic by design and excluded from hash payload.
        constraint_payload["explicit"] = {
            "profiles": explicit_profiles,
        }
    elif source == "sampled":
        constraint_payload["sampled"] = {
            "min": None if training_constraint_profile_sampled_min is None else float(training_constraint_profile_sampled_min),
            "max": None if training_constraint_profile_sampled_max is None else float(training_constraint_profile_sampled_max),
            "count": None if training_constraint_profile_sampled_count is None else int(training_constraint_profile_sampled_count),
            "seed": None if training_constraint_profile_sampled_seed is None else int(training_constraint_profile_sampled_seed),
        }
    else:
        raise ValueError(
            "Unsupported constraint_profile.source for PD run key hashing: "
            f"{training_constraint_profile_source!r}."
        )

    canonical_sample_collection_interval = _canonicalize_sample_collection_interval(
        training_sample_collection_interval
    )

    payload = {
        "schema_version": WRA_PD_RUN_KEY_SCHEMA_VERSION,
        "channel_cache_key": str(channel_cache_key),
        "seed": int(seed),
        "system": {
            "P_max_dBm": float(P_max_dBm),
            "bandwidth_Hz": float(bandwidth_Hz),
            "noise_psd_dBm_Hz": float(noise_psd_dBm_Hz),
        },
        "model": {
            "type": str(model_type),
            "hidden_dim": int(model_hidden_dim),
            "num_layers": int(model_num_layers),
            "K": None if model_K is None else int(model_K),
            "norm_type": str(model_norm_type),
            "dropout": float(model_dropout),
            "activation": str(model_activation),
        },
        "training": {
            "r_min": None if scalar_value is None else float(scalar_value),
            "constraint_profile": constraint_payload,
            "alpha_dual": float(training_alpha_dual),
            "dual_momentum": float(training_dual_momentum),
            "learning_rate": float(training_learning_rate),
            "batch_size": int(training_batch_size),
            "max_epochs": int(training_max_epochs),
            "convergence_window": int(training_convergence_window),
            "convergence_warmup_epochs": None if training_convergence_warmup_epochs is None else int(training_convergence_warmup_epochs),
            "convergence_patience": int(training_convergence_patience),
            "gradient_norm_threshold": float(training_gradient_norm_threshold),
            "dual_variance_threshold": float(training_dual_variance_threshold),
            "dual_stationarity_threshold": float(training_dual_stationarity_threshold),
            "violation_fraction_threshold": float(training_violation_fraction_threshold),
            "violation_fraction_on_model_avg_rates_threshold": float(training_violation_fraction_on_model_avg_rates_threshold),
            "mean_violation_slack_on_model_avg_rates_threshold": float(
                training_mean_violation_slack_on_model_avg_rates_threshold
            ),
            "dual_update_mode": str(training_dual_update_mode),
            "dual_update_frequency": int(training_dual_update_frequency),
            "num_samples_per_network": int(training_num_samples_per_network),
            "moving_avg_window": int(training_moving_avg_window),
            "sample_collection_interval": canonical_sample_collection_interval,
        },
    }
    hash_str = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    hash_val = hashlib.md5(hash_str.encode()).hexdigest()[:12]

    r_token = (
        f"{float(scalar_value):g}"
        if scalar_value is not None
        else str(source)
    )
    return (
        f"wrpd_v{WRA_PD_RUN_KEY_SCHEMA_VERSION}"
        f"_{str(channel_cache_key)}"
        f"_r{r_token}"
        f"_a{float(training_alpha_dual):g}"
        f"_h{hash_val}"
    )


def build_wra_pd_collection_key(
    *,
    pd_run_key: str,
    sample_source: str,
    target_samples_per_network: Optional[int],
    window_size: Optional[int] = None,
    refine_feasible_subset: bool = False,
    refine_objective: str = "min_rate",
    subset_feasibility_tolerance: float = 0.0,
    subset_bottleneck_nodes: int = 5,
) -> str:
    """Build a hashed key identifying a PD sample collection configuration.

    Hashes over ``pd_run_key`` plus all collection parameters.  When
    ``sample_source`` is ``"npz"`` the primal_history-specific parameters
    (windowing, refinement) are excluded from the hash because they have no
    effect on the output; two npz configs that differ only in those parameters
    will correctly map to the same key.
    """
    source = str(sample_source).strip().lower()
    payload: Dict[str, Any] = {
        "schema_version": WRA_PD_COLLECTION_KEY_SCHEMA_VERSION,
        "pd_run_key": str(pd_run_key),
        "sample_source": source,
        "target_samples_per_network": None if target_samples_per_network is None else int(target_samples_per_network),
    }
    if source == "primal_history":
        payload.update({
            "window_size": None if window_size is None else int(window_size),
            "refine_feasible_subset": bool(refine_feasible_subset),
            "refine_objective": str(refine_objective),
            "subset_feasibility_tolerance": float(subset_feasibility_tolerance),
            "subset_bottleneck_nodes": int(subset_bottleneck_nodes),
        })
    hash_str = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    hash_val = hashlib.md5(hash_str.encode()).hexdigest()[:12]

    return (
        f"wrpc_v{WRA_PD_COLLECTION_KEY_SCHEMA_VERSION}"
        f"_{source}"
        f"_k{target_samples_per_network if target_samples_per_network is not None else 'all'}"
        f"_h{hash_val}"
    )


def register_wra_hydra_resolvers() -> None:
    """Register OmegaConf resolvers used by WRA configs for hashed run/cache keys."""
    from omegaconf import OmegaConf

    def _wra_channel_cache_key_resolver(
        seed: Any,
        num_networks: Any,
        n_links: Any,
        deployment_range: Any,
        channel_version: Any,
        min_tx_tx_distance: Any,
        min_tx_rx_distance: Any,
        max_tx_rx_distance: Any,
        path_loss_exponent_short: Any,
        path_loss_exponent_long: Any,
        shadowing_std: Any,
        carrier_freq: Any,
        speed: Any,
        num_fading_paths: Any,
        delta_t: Any,
        max_deployment_attempts: Any,
        max_recursion_depth: Any,
    ) -> str:
        channel_params = {
            "min_tx_tx_distance": min_tx_tx_distance,
            "min_tx_rx_distance": min_tx_rx_distance,
            "max_tx_rx_distance": max_tx_rx_distance,
            "path_loss_exponent_short": path_loss_exponent_short,
            "path_loss_exponent_long": path_loss_exponent_long,
            "shadowing_std": shadowing_std,
            "carrier_freq": carrier_freq,
            "speed": speed,
            "num_fading_paths": num_fading_paths,
            "delta_t": delta_t,
            "max_deployment_attempts": max_deployment_attempts,
            "max_recursion_depth": max_recursion_depth,
        }
        return build_wra_channel_cache_key(
            seed=int(seed),
            num_networks=int(num_networks),
            n_links=int(n_links),
            deployment_range=float(deployment_range),
            channel_version=channel_version,
            channel_params=channel_params,
        )

    def _wra_pd_run_key_resolver(
        channel_cache_key: Any,
        seed: Any,
        P_max_dBm: Any,
        bandwidth_Hz: Any,
        noise_psd_dBm_Hz: Any,
        model_type: Any,
        model_hidden_dim: Any,
        model_num_layers: Any,
        model_K: Any,
        model_norm_type: Any,
        model_dropout: Any,
        model_activation: Any,
        training_r_min: Any,
        training_constraint_profile_type: Any,
        training_constraint_profile_source: Any,
        training_constraint_profile_scalar_value: Any,
        training_constraint_profile_explicit_profiles: Any,
        training_constraint_profile_sampled_min: Any,
        training_constraint_profile_sampled_max: Any,
        training_constraint_profile_sampled_count: Any,
        training_constraint_profile_sampled_seed: Any,
        training_constraint_profile_r_min_feature_scale: Any,
        training_alpha_dual: Any,
        training_dual_momentum: Any,
        training_learning_rate: Any,
        training_batch_size: Any,
        training_max_epochs: Any,
        training_convergence_window: Any,
        training_convergence_warmup_epochs: Any,
        training_convergence_patience: Any,
        training_gradient_norm_threshold: Any,
        training_dual_variance_threshold: Any,
        training_dual_stationarity_threshold: Any,
        training_violation_fraction_threshold: Any,
        training_violation_fraction_on_model_avg_rates_threshold: Any,
        training_mean_violation_slack_on_model_avg_rates_threshold: Any,
        training_dual_update_mode: Any,
        training_dual_update_frequency: Any,
        training_num_samples_per_network: Any,
        training_moving_avg_window: Any,
        training_sample_collection_interval: Any,
    ) -> str:
        return build_wra_pd_run_key(
            channel_cache_key=str(channel_cache_key),
            seed=int(seed),
            P_max_dBm=float(P_max_dBm),
            bandwidth_Hz=float(bandwidth_Hz),
            noise_psd_dBm_Hz=float(noise_psd_dBm_Hz),
            model_type=str(model_type),
            model_hidden_dim=int(model_hidden_dim),
            model_num_layers=int(model_num_layers),
            model_K=None if _none_if_null(model_K) is None else int(model_K),
            model_norm_type=str(model_norm_type),
            model_dropout=float(model_dropout),
            model_activation=str(model_activation),
            training_r_min=None if _none_if_null(training_r_min) is None else float(training_r_min),
            training_constraint_profile_type=str(training_constraint_profile_type),
            training_constraint_profile_source=str(training_constraint_profile_source),
            training_constraint_profile_scalar_value=None if _none_if_null(training_constraint_profile_scalar_value) is None else float(training_constraint_profile_scalar_value),
            training_constraint_profile_explicit_profiles=_none_if_null(training_constraint_profile_explicit_profiles),
            training_constraint_profile_sampled_min=None if _none_if_null(training_constraint_profile_sampled_min) is None else float(training_constraint_profile_sampled_min),
            training_constraint_profile_sampled_max=None if _none_if_null(training_constraint_profile_sampled_max) is None else float(training_constraint_profile_sampled_max),
            training_constraint_profile_sampled_count=None if _none_if_null(training_constraint_profile_sampled_count) is None else int(training_constraint_profile_sampled_count),
            training_constraint_profile_sampled_seed=None if _none_if_null(training_constraint_profile_sampled_seed) is None else int(training_constraint_profile_sampled_seed),
            training_constraint_profile_r_min_feature_scale=float(training_constraint_profile_r_min_feature_scale),
            training_alpha_dual=float(training_alpha_dual),
            training_dual_momentum=float(training_dual_momentum),
            training_learning_rate=float(training_learning_rate),
            training_batch_size=int(training_batch_size),
            training_max_epochs=int(training_max_epochs),
            training_convergence_window=int(training_convergence_window),
            training_convergence_warmup_epochs=None if _none_if_null(training_convergence_warmup_epochs) is None else int(training_convergence_warmup_epochs),
            training_convergence_patience=int(training_convergence_patience),
            training_gradient_norm_threshold=float(training_gradient_norm_threshold),
            training_dual_variance_threshold=float(training_dual_variance_threshold),
            training_dual_stationarity_threshold=float(training_dual_stationarity_threshold),
            training_violation_fraction_threshold=float(training_violation_fraction_threshold),
            training_violation_fraction_on_model_avg_rates_threshold=float(training_violation_fraction_on_model_avg_rates_threshold),
            training_mean_violation_slack_on_model_avg_rates_threshold=float(
                training_mean_violation_slack_on_model_avg_rates_threshold
            ),
            training_dual_update_mode=str(training_dual_update_mode),
            training_dual_update_frequency=int(training_dual_update_frequency),
            training_num_samples_per_network=int(training_num_samples_per_network),
            training_moving_avg_window=int(training_moving_avg_window),
            training_sample_collection_interval=training_sample_collection_interval,
        )

    def _wra_pd_collection_key_resolver(
        pd_run_key: Any,
        sample_source: Any,
        target_samples_per_network: Any,
        window_size: Any,
        refine_feasible_subset: Any,
        refine_objective: Any,
        subset_feasibility_tolerance: Any,
        subset_bottleneck_nodes: Any,
    ) -> str:
        return build_wra_pd_collection_key(
            pd_run_key=str(pd_run_key),
            sample_source=str(sample_source),
            target_samples_per_network=None if target_samples_per_network is None else int(target_samples_per_network),
            window_size=None if window_size is None else int(window_size),
            refine_feasible_subset=bool(refine_feasible_subset),
            refine_objective=str(refine_objective),
            subset_feasibility_tolerance=float(subset_feasibility_tolerance),
            subset_bottleneck_nodes=int(subset_bottleneck_nodes),
        )

    for name, func in (
        ("wra_channel_cache_key", _wra_channel_cache_key_resolver),
        ("wra_pd_run_key", _wra_pd_run_key_resolver),
        ("wra_pd_collection_key", _wra_pd_collection_key_resolver),
    ):
        try:
            OmegaConf.register_new_resolver(name, func)
        except ValueError:
            # Resolver may already be registered if multiple WRA scripts are imported in-process.
            pass


def create_wireless_channel(
    *,
    n_links: int,
    seed: Optional[int],
    channel_version: Optional[Any],
    deployment_range: Optional[float] = None,
    channel_params: Optional[Mapping[str, Any]] = None,
    default_version: str = "v2",
    **overrides: Any,
) -> WirelessChannel:
    """Create a wireless channel with unified version/kwargs handling."""
    channel_cls = get_channel_class(channel_version, default=default_version)

    kwargs = sanitize_channel_params(channel_params)
    kwargs.update({k: v for k, v in overrides.items() if k in CHANNEL_PARAM_KEYS and v is not None})

    if deployment_range is not None:
        kwargs["deployment_range"] = deployment_range
    kwargs.setdefault("deployment_range", 1000.0)

    return channel_cls(n_links=n_links, seed=seed, **kwargs)
