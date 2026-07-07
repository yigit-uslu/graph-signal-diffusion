"""
Primal-dual power allocation GNN training.

Workflow:
1. Load or generate WRA channel dataset (content-addressed caching)
2. Set up system parameters, model, and dual optimizer
3. Train the GNN policy with primal-dual updates
4. Collect samples for downstream diffusion model training
"""

import logging
import math
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch_geometric.loader import DataLoader

from graph_signal_diffusion.datasets.wra.channel_factory import (
    build_wra_channel_cache_metadata,
    canonicalize_channel_cache_metadata,
    find_channel_cache_metadata_mismatches,
    normalize_channel_version,
    register_wra_hydra_resolvers,
    sanitize_channel_params,
)
from graph_signal_diffusion.datasets.wra.primal_dual_dataset import (
    WRAPrimalDualDataset,
    WRAConstraintProfilePrimalDualDataset,
)
from graph_signal_diffusion.models.power_allocation_gnn import PowerAllocationGNN
from graph_signal_diffusion.models.power_allocation_mlp import PowerAllocationMLP
from graph_signal_diffusion.trainers.dual_optimizer import DualOptimizer
from graph_signal_diffusion.trainers.primal_dual_trainer import (
    WRAPrimalDualTrainer,
    WRAConditionalPrimalDualTrainer,
)
from graph_signal_diffusion.utils.rate_calculator import (
    compute_ergodic_rates,
    compute_system_parameters,
)

register_wra_hydra_resolvers()

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_device(cfg):
    """Return the torch device string, falling back to CUDA / CPU auto-detect."""
    if cfg.device is not None and cfg.device != "":
        return cfg.device
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def _setup_file_logging(output_dir):
    """Attach a file handler to the root logger so every module logs to disk."""
    log_file = Path(output_dir) / "train_primal_dual_power_allocation.log"
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    logging.getLogger().addHandler(handler)
    log.info("Logging to %s", log_file)


# -- Channel config extraction ------------------------------------------------

def _build_channel_kwargs(cfg):
    """Extract channel version and constructor kwargs from Hydra config.

    Returns ``(channel_version, channel_kwargs)`` where *channel_kwargs* can
    be splatted into ``WRAPrimalDualDataset.from_seed_range(**channel_kwargs)``.
    """
    channel_config = cfg.get('channel', {})
    channel_version = normalize_channel_version(
        channel_config.get('version'), default='v2',
    )

    overrides = sanitize_channel_params(channel_config)
    overrides.pop('deployment_range', None)           # dataset.deployment_range is authoritative
    if channel_version != 'v3':
        overrides.pop('max_recursion_depth', None)    # V1/V2 constructors don't accept it

    kwargs = {'deployment_range': cfg.dataset.deployment_range}
    kwargs.update(overrides)
    return channel_version, kwargs


# -- Dataset loading / generation with caching ---------------------------------

def _build_expected_cache_metadata(cfg, channel_version, channel_kwargs):
    """Build the canonicalized metadata we expect the cache file to contain.

    *channel_kwargs* may include ``deployment_range``; the factory's
    canonicalize_wra_channel_params strips it internally so the hash is
    unaffected.
    """
    channel_config_name = cfg.get('channel_config_name', f'default_{cfg.seed}')
    raw = build_wra_channel_cache_metadata(
        seed=int(cfg.seed),
        num_networks=int(cfg.dataset.num_networks),
        n_links=int(cfg.dataset.n_links),
        deployment_range=float(cfg.dataset.deployment_range),
        channel_version=channel_version,
        channel_params=channel_kwargs,
        channel_config_name=channel_config_name,
    )
    # Defensive re-canonicalization — idempotent, but guards against any
    # future divergence between build_wra_channel_cache_metadata output
    # and the comparison-time canonical form.
    return canonicalize_channel_cache_metadata(raw, default_channel_version='v2')


def _try_load_cached_channels(channels_file, expected_metadata):
    """Try to load and validate a cached channel file.

    Returns the channel list on success, or ``None`` on any failure.
    """
    if not channels_file.exists():
        log.info("No cached channels at %s", channels_file)
        return None

    log.info("Found cached channels at %s — validating …", channels_file)
    cached = torch.load(channels_file, map_location='cpu', weights_only=False)

    if not isinstance(cached, dict) or 'channels' not in cached:
        log.warning("Cache file format invalid; will regenerate.")
        return None

    cached_meta = canonicalize_channel_cache_metadata(
        cached.get('metadata'), default_channel_version='v2',
    )
    mismatches = find_channel_cache_metadata_mismatches(expected_metadata, cached_meta)
    if mismatches:
        log.warning("Metadata mismatch — will regenerate channels:")
        for key, (exp, act) in mismatches.items():
            log.warning("  %s: expected=%r  actual=%r", key, exp, act)
        return None

    channels = cached['channels']
    log.info("✓ Loaded %d pre-generated channels from cache", len(channels))
    return channels


def _load_or_generate_dataset(cfg, channel_version, channel_kwargs):
    """Return a ``WRAPrimalDualDataset``, using the content-addressed cache when valid."""
    expected = _build_expected_cache_metadata(cfg, channel_version, channel_kwargs)
    cache_key = expected['channel_cache_key']
    t_chunk = cfg.dataset.get('t_chunk', None)

    cache_dir = Path('data/wra_channel_cache') / cfg.get('name', '')
    cache_dir.mkdir(parents=True, exist_ok=True)
    channels_file = cache_dir / f'{cache_key}.pt'

    # Try cache first
    channels = _try_load_cached_channels(channels_file, expected)
    if channels is not None:
        return WRAPrimalDualDataset(
            channels=channels,
            num_timesteps=cfg.dataset.num_timesteps,
            t_chunk=t_chunk,
        )

    # Generate fresh channels
    log.info("Generating %d channels (seed_start=%d, n_links=%d) …",
             cfg.dataset.num_networks, cfg.seed, cfg.dataset.n_links)

    dataset = WRAPrimalDualDataset.from_seed_range(
        num_networks=cfg.dataset.num_networks,
        n_links=cfg.dataset.n_links,
        seed_start=cfg.seed,
        num_timesteps=cfg.dataset.num_timesteps,
        t_chunk=t_chunk,
        channel_version=channel_version,
        **channel_kwargs,
    )

    # Persist to cache for future runs
    torch.save({
        'metadata': dict(expected),
        'channels': dataset.channels,
    }, channels_file)
    log.info("✓ Cached %d channels → %s", len(dataset.channels), channels_file)

    return dataset


# -- Model creation ------------------------------------------------------------

def _create_model(cfg, P_max, input_dim=2):
    """Instantiate the power-allocation model from config."""
    if cfg.model.type == 'mlp':
        model = PowerAllocationMLP(
            input_dim=input_dim,
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.num_layers,
            P_max=P_max,
        )
    else:
        model = PowerAllocationGNN(
            input_dim=input_dim,
            hidden_dim=cfg.model.hidden_dim,
            num_layers=cfg.model.num_layers,
            K=cfg.model.K,
            P_max=P_max,
            norm_type=cfg.model.norm_type,
            num_groups=cfg.model.num_groups,
            dropout=cfg.model.dropout,
            activation=cfg.model.activation,
        )

    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model: %s  (%s parameters)", cfg.model.type.upper(), f'{n_params:,}')
    return model


# -- Full-power baseline -------------------------------------------------------

def _evaluate_full_power_baseline(dataloader, P_max, noise_var, device):
    """Compute ergodic-rate statistics under full-power allocation."""
    all_rates = []
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            for idx in range(batch.num_graphs):
                H_inst = batch.H_instantaneous[idx]
                associations = batch.associations[idx]
                power = torch.full((associations.shape[0],), P_max, device=device)
                rates = compute_ergodic_rates(
                    power_allocation=power,
                    H_instantaneous=H_inst,
                    associations=associations,
                    noise_var=noise_var,
                )
                all_rates.append(rates)

    all_rates = torch.cat(all_rates)
    return {
        'min_rate': all_rates.min().item(),
        'mean_rate': all_rates.mean().item(),
        'rate_5th_percentile': torch.quantile(all_rates, 0.05).item(),
        'max_rate': all_rates.max().item(),
    }


def _log_baseline_and_feasibility(
    baseline,
    *,
    scalar_r_min=None,
    profile_table=None,
    profile_source="scalar",
):
    """Log full-power baseline and basic feasibility diagnostics."""
    log.info("Full-power baseline:  min=%.4f  p5=%.4f  mean=%.4f  max=%.4f",
             baseline['min_rate'], baseline['rate_5th_percentile'],
             baseline['mean_rate'], baseline['max_rate'])

    if scalar_r_min is not None:
        r_min = float(scalar_r_min)
        log.info("Target r_min:         %.4f", r_min)
        if baseline['rate_5th_percentile'] >= r_min:
            log.warning("⚠  5th-percentile already exceeds r_min — problem may be too easy.")
        elif baseline['mean_rate'] < r_min:
            log.warning("⚠  Mean rate is below r_min — problem may be infeasible.")
        else:
            log.info("✓ Constraint appears reasonable for optimization.")
        return

    if profile_table is None:
        log.warning("Constraint profile metadata missing; skipping feasibility checks.")
        return

    profile_values = np.asarray(profile_table, dtype=np.float64).reshape(-1)
    profile_min = float(np.min(profile_values))
    profile_max = float(np.max(profile_values))
    log.info(
        "Constraint profiles (%s): min=%.4f  max=%.4f  mean=%.4f",
        profile_source,
        profile_min,
        profile_max,
        float(np.mean(profile_values)),
    )
    if baseline['rate_5th_percentile'] >= profile_max:
        log.warning("⚠  5th-percentile exceeds max profile threshold — profiles may be too easy.")
    elif baseline['mean_rate'] < profile_min:
        log.warning("⚠  Mean rate is below min profile threshold — profiles may be too hard.")
    else:
        log.info("✓ Baseline overlaps profile range; optimization should be informative.")


def _to_plain_container(value):
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _coerce_optional_float_literal(value, field_name):
    """Normalize optional scalar config values, including common infinity literals."""
    if value is None:
        return None

    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"", "none", "null", "~"}:
            return None
        if token in {"inf", "+inf", ".inf", "+.inf"}:
            return float("inf")
        if token in {"-inf", "-.inf"}:
            return float("-inf")
        try:
            return float(token)
        except ValueError as exc:
            raise ValueError(
                f"{field_name} must be numeric, null, or an infinity literal; got {value!r}."
            ) from exc

    if isinstance(value, (int, float, np.number)):
        return float(value)

    raise TypeError(
        f"{field_name} must be a scalar numeric value or null; got {type(value).__name__}."
    )


def _resolve_sample_collection_interval(cfg) -> int | float | None:
    """
    Resolve sample-collection interval from config.

    Accepted forms:
    - null/none: auto mode in trainer
    - positive integer: periodic collection every K epochs
    - inf/.inf: disable periodic collection (final collection still runs)
    """
    raw = cfg.training.get("sample_collection_interval", None)
    value = _coerce_optional_float_literal(raw, "training.sample_collection_interval")
    if value is None:
        return None
    if math.isinf(value):
        if value > 0:
            return float("inf")
        raise ValueError(
            "training.sample_collection_interval must be positive when using infinity."
        )
    if not math.isfinite(value):
        raise ValueError("training.sample_collection_interval must be finite, +inf, or null.")
    if value <= 0:
        raise ValueError("training.sample_collection_interval must be > 0, +inf, or null.")

    as_int = int(round(value))
    if not math.isclose(value, as_int, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            "training.sample_collection_interval must be an integer number of epochs, +inf, or null."
        )
    return as_int


def _coerce_nonnegative_int_list(value, field_name: str) -> list[int]:
    """Normalize optional list-valued config fields into sorted unique int lists."""
    plain = _to_plain_container(value)
    if plain is None:
        return []
    if isinstance(plain, str):
        token = plain.strip().lower()
        if token in {"", "none", "null", "~"}:
            return []
        raise TypeError(f"{field_name} must be a list of integers; got string {value!r}.")
    if not isinstance(plain, list):
        raise TypeError(f"{field_name} must be a list of integers.")

    out: list[int] = []
    for idx, item in enumerate(plain):
        if isinstance(item, bool):
            raise TypeError(
                f"{field_name}[{idx}] must be an integer, not boolean {item!r}."
            )
        if isinstance(item, (int, np.integer)):
            parsed = int(item)
        elif isinstance(item, (float, np.floating)) and float(item).is_integer():
            parsed = int(item)
        else:
            raise TypeError(
                f"{field_name}[{idx}] must be an integer; got {item!r} ({type(item).__name__})."
            )
        if parsed < 0:
            raise ValueError(f"{field_name}[{idx}] must be >= 0; got {parsed}.")
        out.append(parsed)

    return sorted(set(out))


def _resolve_trace_logging_config(cfg) -> dict:
    """Resolve optional tracked-network trajectory logging config."""
    trace_cfg = cfg.training.get("trace_logging", {})
    enabled = bool(trace_cfg.get("enabled", False))
    network_ids = _coerce_nonnegative_int_list(
        trace_cfg.get("network_ids", []),
        "training.trace_logging.network_ids",
    )
    receiver_indices = _coerce_nonnegative_int_list(
        trace_cfg.get("receiver_indices", []),
        "training.trace_logging.receiver_indices",
    )
    include_full_vectors = bool(trace_cfg.get("include_full_vectors", True))

    write_interval_raw = trace_cfg.get("write_interval", 50)
    write_interval_val = _coerce_optional_float_literal(
        write_interval_raw,
        "training.trace_logging.write_interval",
    )
    if write_interval_val is None:
        write_interval = 50
    else:
        if math.isinf(write_interval_val) or write_interval_val <= 0:
            raise ValueError("training.trace_logging.write_interval must be a positive integer.")
        write_interval = int(round(write_interval_val))
        if not math.isclose(write_interval_val, write_interval, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                "training.trace_logging.write_interval must be an integer number of epochs."
            )

    output_filename = str(
        trace_cfg.get("output_filename", "tracked_network_trace.jsonl")
    ).strip()
    if not output_filename:
        raise ValueError("training.trace_logging.output_filename must be non-empty.")

    if enabled and not network_ids:
        log.warning(
            "trace_logging.enabled=true but no network_ids were configured; trace logging will be disabled."
        )
        enabled = False

    return {
        "enabled": enabled,
        "network_ids": network_ids,
        "receiver_indices": receiver_indices,
        "include_full_vectors": include_full_vectors,
        "write_interval": write_interval,
        "output_filename": output_filename,
    }


def _canonicalize_constraint_profiles(profiles, n_links):
    if profiles is None:
        return []
    plain_profiles = _to_plain_container(profiles)
    if not isinstance(plain_profiles, list):
        raise ValueError(
            "constraint_profile.explicit.profiles must be a list of scalars or vectors."
        )
    canonical = []
    for idx, profile in enumerate(plain_profiles):
        if isinstance(profile, (int, float)):
            canonical.append(float(profile))
            continue
        arr = np.asarray(profile, dtype=np.float32).reshape(-1)
        if arr.shape[0] != int(n_links):
            raise ValueError(
                f"Profile {idx} has length {arr.shape[0]} but n_links={n_links}."
            )
        canonical.append(arr.tolist())
    return canonical


def _resolve_constraint_profile_spec(cfg):
    """
    Resolve configured constraint profiles into a canonical spec dictionary.

    Returns keys:
    - source: one of {"scalar", "explicit", "sampled"}
    - scalar_r_min: float for scalar mode, else None
    - profiles: list of scalar/vector profile definitions
    - profile_names: optional list of labels aligned with profiles
    - r_min_feature_scale: float feature scale injected into node features
    """
    profile_cfg = cfg.training.get("constraint_profile", {})
    profile_type = str(profile_cfg.get("type", "min_rate")).strip().lower()
    if profile_type != "min_rate":
        raise ValueError(
            f"Unsupported constraint_profile.type={profile_type!r}. Expected 'min_rate'."
        )

    source = str(profile_cfg.get("source", "scalar")).strip().lower()
    feature_scale = float(profile_cfg.get("r_min_feature_scale", 1.0))

    if source == "scalar":
        scalar_cfg = profile_cfg.get("scalar", {})
        scalar_value = scalar_cfg.get("value", cfg.training.get("r_min"))
        if scalar_value is None:
            raise ValueError(
                "constraint_profile.source=scalar requires training.r_min or "
                "constraint_profile.scalar.value."
            )
        return {
            "source": "scalar",
            "scalar_r_min": float(scalar_value),
            "profiles": [float(scalar_value)],
            "profile_names": ["scalar"],
            "r_min_feature_scale": feature_scale,
        }

    if source == "explicit":
        explicit_cfg = profile_cfg.get("explicit", {})
        profiles = _canonicalize_constraint_profiles(
            explicit_cfg.get("profiles", []),
            n_links=cfg.dataset.n_links,
        )
        if len(profiles) == 0:
            raise ValueError(
                "constraint_profile.source=explicit requires non-empty "
                "constraint_profile.explicit.profiles."
            )
        raw_names = _to_plain_container(explicit_cfg.get("names", []))
        names = [str(v) for v in raw_names] if isinstance(raw_names, list) else []
        if names and len(names) != len(profiles):
            raise ValueError(
                "constraint_profile.explicit.names length must match profiles length."
            )
        return {
            "source": "explicit",
            "scalar_r_min": None,
            "profiles": profiles,
            "profile_names": names or None,
            "r_min_feature_scale": feature_scale,
        }

    if source == "sampled":
        sampled_cfg = profile_cfg.get("sampled", {})
        min_v = sampled_cfg.get("min")
        max_v = sampled_cfg.get("max")
        count = sampled_cfg.get("count")
        if min_v is None or max_v is None or count is None:
            raise ValueError(
                "constraint_profile.source=sampled requires sampled.min, sampled.max, and sampled.count."
            )
        count = int(count)
        if count <= 0:
            raise ValueError("constraint_profile.sampled.count must be > 0.")
        seed = sampled_cfg.get("seed")
        rng_seed = int(seed) if seed is not None else int(cfg.seed)
        rng = np.random.default_rng(rng_seed)
        sampled_profiles = rng.uniform(float(min_v), float(max_v), size=count).tolist()
        sampled_names = [f"sampled_{i}" for i in range(count)]
        return {
            "source": "sampled",
            "scalar_r_min": None,
            "profiles": sampled_profiles,
            "profile_names": sampled_names,
            "r_min_feature_scale": feature_scale,
        }

    raise ValueError(
        f"Unsupported constraint_profile.source={source!r}. "
        "Expected one of: scalar, explicit, sampled."
    )


def _build_constraint_conditioned_dataset(
    dataset: WRAPrimalDualDataset,
    profile_spec: dict,
) -> dict:
    """
    Build training dataset + dual threshold input from constraint profile spec.

    Returns a dictionary with:
    - constraint_source
    - constraint_dataset (None for scalar mode)
    - train_dataset
    - model_input_dim
    - dual_r_min_input (scalar or table accepted by DualOptimizer)
    """
    constraint_source = str(profile_spec["source"])
    constraint_dataset = None
    train_dataset = dataset
    model_input_dim = 2
    dual_r_min_input = profile_spec["scalar_r_min"]

    if constraint_source != "scalar":
        constraint_dataset = WRAConstraintProfilePrimalDualDataset(
            base_dataset=dataset,
            constraint_profiles=profile_spec["profiles"],
            profile_names=profile_spec["profile_names"],
            r_min_feature_scale=profile_spec["r_min_feature_scale"],
        )
        train_dataset = constraint_dataset
        model_input_dim = 3
        dual_r_min_input = constraint_dataset.get_r_min_table()

    return {
        "constraint_source": constraint_source,
        "constraint_dataset": constraint_dataset,
        "train_dataset": train_dataset,
        "model_input_dim": model_input_dim,
        "dual_r_min_input": dual_r_min_input,
    }


def _build_pd_trainer(constraint_dataset, trainer_kwargs: dict):
    """Instantiate scalar or conditional trainer based on wrapped dataset presence."""
    if constraint_dataset is None:
        return WRAPrimalDualTrainer(**trainer_kwargs)
    return WRAConditionalPrimalDualTrainer(
        constraint_profile_dataset=constraint_dataset,
        **trainer_kwargs,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../../conf/wra_generation", config_name="pd_training/wra_small")
def main(cfg: DictConfig):
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    _setup_file_logging(output_dir)
    device = _resolve_device(cfg)
    torch.manual_seed(cfg.seed)

    log.info("=" * 60)
    log.info("PRIMAL-DUAL POWER ALLOCATION TRAINING")
    log.info("=" * 60)
    log.info("Output: %s", output_dir)

    # --- 1. Dataset -----------------------------------------------------------
    channel_version, channel_kwargs = _build_channel_kwargs(cfg)
    log.info("Channel: version=%s  kwargs=%s", channel_version, channel_kwargs)

    dataset = _load_or_generate_dataset(cfg, channel_version, channel_kwargs)
    t_chunk = int(getattr(dataset, "t_chunk", cfg.dataset.num_timesteps))
    if t_chunk < int(cfg.dataset.num_timesteps):
        log.info(
            "Using random consecutive timestep chunking for training data: t_chunk=%d / total_T=%d",
            t_chunk,
            int(cfg.dataset.num_timesteps),
        )
    profile_spec = _resolve_constraint_profile_spec(cfg)
    conditioning = _build_constraint_conditioned_dataset(dataset, profile_spec)
    constraint_source = conditioning["constraint_source"]
    constraint_dataset = conditioning["constraint_dataset"]
    train_dataset = conditioning["train_dataset"]
    model_input_dim = conditioning["model_input_dim"]
    dual_r_min_input = conditioning["dual_r_min_input"]

    # Input accepted by DualOptimizer:
    # - scalar float for homogeneous constraints
    # - tensor-like table (num_networks, num_receivers) for profile-conditioned runs
    if constraint_dataset is not None:
        log.info(
            "Constraint profile mode=%s: %d profiles, feature_scale=%.4f, "
            "expanded networks=%d",
            constraint_source,
            constraint_dataset.num_profiles,
            profile_spec["r_min_feature_scale"],
            len(train_dataset),
        )
    else:
        log.info("Constraint profile mode=scalar: r_min=%.4f", float(dual_r_min_input))

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        drop_last=False,
    )

    # --- 2. System parameters -------------------------------------------------
    system_params = compute_system_parameters(
        P_max_dBm=cfg.system.P_max_dBm,
        bandwidth_Hz=cfg.system.bandwidth_Hz,
        noise_psd_dBm_Hz=cfg.system.noise_psd_dBm_Hz,
    )
    log.info(
        "System: P_max=%.3f mW (%.1f dBm)  BW=%.1f MHz  noise=%.1f dBm  SNR_max=%.1f dB",
        system_params['P_max_watts'] * 1000, cfg.system.P_max_dBm,
        cfg.system.bandwidth_Hz / 1e6,
        system_params['noise_power_dBm'], system_params['snr_dB'],
    )

    # --- 3. Model -------------------------------------------------------------
    model = _create_model(cfg, system_params['P_max_watts'], input_dim=model_input_dim)

    # --- 4. Dual optimizer ----------------------------------------------------
    dual_momentum = cfg.training.get('dual_momentum', 0.0)
    num_training_networks = len(train_dataset)
    dual_optimizer = DualOptimizer(
        num_networks=num_training_networks,
        num_receivers=cfg.dataset.n_links,
        r_min=dual_r_min_input,
        alpha_dual=cfg.training.alpha_dual,
        update_frequency=cfg.training.dual_update_frequency,
        momentum=dual_momentum,
        device=device,
    )
    if constraint_source == "scalar":
        log.info(
            "Dual: %d×%d vars  r_min=%.4f  α=%.4f  momentum=%.2f",
            num_training_networks, cfg.dataset.n_links,
            float(dual_r_min_input), cfg.training.alpha_dual, dual_momentum,
        )
    else:
        table = dual_optimizer.r_min_table.detach().cpu()
        log.info(
            "Dual: %d×%d vars  r_min_table_shape=%s  range=[%.4f, %.4f]  α=%.4f  momentum=%.2f",
            num_training_networks,
            cfg.dataset.n_links,
            tuple(table.shape),
            float(table.min().item()),
            float(table.max().item()),
            cfg.training.alpha_dual,
            dual_momentum,
        )

    # --- 5. Baseline feasibility check ----------------------------------------
    # Use the base (un-expanded) dataset so profiles are not evaluated redundantly.
    baseline_prev_t_chunk = getattr(dataset, "t_chunk", None)
    if baseline_prev_t_chunk is not None:
        dataset.t_chunk = int(cfg.dataset.num_timesteps)
        if int(baseline_prev_t_chunk) < int(cfg.dataset.num_timesteps):
            log.info(
                "Full-power baseline uses full timesteps (T=%d); training chunking resumes afterward (t_chunk=%d).",
                int(cfg.dataset.num_timesteps),
                int(baseline_prev_t_chunk),
            )
    baseline_loader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        drop_last=False,
    )
    baseline = _evaluate_full_power_baseline(
        baseline_loader, system_params['P_max_watts'],
        system_params['noise_var'], device,
    )
    if constraint_dataset is None:
        _log_baseline_and_feasibility(
            baseline,
            scalar_r_min=float(dual_r_min_input),
        )
    else:
        _log_baseline_and_feasibility(
            baseline,
            profile_table=constraint_dataset.constraint_profile_table.cpu().numpy(),
            profile_source=constraint_source,
        )
    if baseline_prev_t_chunk is not None:
        dataset.t_chunk = int(baseline_prev_t_chunk)

    # --- 6. Train -------------------------------------------------------------
    dual_update_mode = cfg.training.get('dual_update_mode', 'step')
    trace_logging_cfg = _resolve_trace_logging_config(cfg)
    _warmup = cfg.training.convergence_warmup_epochs
    trainer_kwargs = dict(
        model=model,
        dual_optimizer=dual_optimizer,
        system_params=system_params,
        learning_rate=cfg.training.learning_rate,
        max_epochs=cfg.training.max_epochs,
        checkpoint_dir=output_dir,
        convergence_window=cfg.training.convergence_window,
        convergence_warmup_epochs=None if _warmup is None else int(_warmup),
        convergence_patience=cfg.training.convergence_patience,
        gradient_norm_threshold=float(cfg.training.gradient_norm_threshold),
        dual_variance_threshold=float(cfg.training.dual_variance_threshold),
        dual_stationarity_threshold=float(cfg.training.get('dual_stationarity_threshold', 0.05)),
        violation_fraction_threshold=float(cfg.training.violation_fraction_threshold),
        violation_fraction_on_model_avg_rates_threshold=float(
            cfg.training.violation_fraction_on_model_avg_rates_threshold
        ),
        mean_violation_slack_on_model_avg_rates_threshold=float(
            cfg.training.mean_violation_slack_on_model_avg_rates_threshold
        ),
        num_samples_per_network=cfg.training.num_samples_per_network,
        moving_avg_window=cfg.training.moving_avg_window,
        dual_update_mode=dual_update_mode,
        sample_collection_interval=_resolve_sample_collection_interval(cfg),
        trace_logging_enabled=trace_logging_cfg["enabled"],
        trace_network_ids=trace_logging_cfg["network_ids"],
        trace_receiver_indices=trace_logging_cfg["receiver_indices"],
        trace_include_full_vectors=trace_logging_cfg["include_full_vectors"],
        trace_write_interval=trace_logging_cfg["write_interval"],
        trace_output_filename=trace_logging_cfg["output_filename"],
        channel_version=channel_version,
        device=device,
    )
    trainer = _build_pd_trainer(constraint_dataset, trainer_kwargs)
    log.info("Training: dual_update_mode=%s  lr=%.1e  max_epochs=%d",
             dual_update_mode, cfg.training.learning_rate, cfg.training.max_epochs)
    if trace_logging_cfg["enabled"]:
        log.info(
            "Tracked-network trace logging enabled: networks=%s  receivers=%s  file=%s",
            trace_logging_cfg["network_ids"],
            trace_logging_cfg["receiver_indices"] or "auto-first-two",
            trace_logging_cfg["output_filename"],
        )
    else:
        log.info("Tracked-network trace logging disabled.")

    results = trainer.train(train_loader)

    # --- 7. Summary -----------------------------------------------------------
    log.info("=" * 60)
    log.info("TRAINING COMPLETE  |  converged=%s  epoch=%d  networks=%d",
             results['converged'], results['final_epoch'], len(results['samples']))
    log.info("=" * 60)

    samples_path = Path(output_dir) / "collected_samples.npz"
    log.info("To convert samples for diffusion training:")
    log.info(
        "  python -m graph_signal_diffusion.cli.wra.build_diffusion_dataset "
        "--input %s --output-root data/wra", samples_path,
    )


if __name__ == '__main__':
    main()
