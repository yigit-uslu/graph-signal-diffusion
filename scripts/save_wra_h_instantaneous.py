#!/usr/bin/env python
"""Generate and save compressed H_instantaneous for converted raw WRA datasets.

This script is intended to run after scripts/convert_pd_samples_to_diffusion.py.
It reconstructs each network channel from seed + metadata and supports two output modes:

1) Single-file mode (default):
    <raw_dir>/wra_h_instantaneous.npz
2) Per-network mode (--per-network-files):
    <raw_dir>/wra_h_instantaneous/network_<id>.npz

Stored shape convention:
    H: (T, m, n)
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import yaml

import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graph_signal_diffusion.datasets.wra.channel_factory import (  # noqa: E402
    create_wireless_channel,
    normalize_channel_version,
)


def _get_config_param(config: Mapping[str, Any], key: str, default: Any = None) -> Any:
    if key in config:
        return config[key]
    for section in ("dataset", "system", "training", "channel"):
        section_obj = config.get(section, {})
        if isinstance(section_obj, Mapping) and key in section_obj:
            return section_obj[key]
    return default


def _extract_network_ids(raw_payload: Mapping[str, Any]) -> List[int]:
    if "network_ids" in raw_payload:
        return [int(x) for x in np.asarray(raw_payload["network_ids"], dtype=np.int64).tolist()]

    ids = []
    for key in raw_payload.keys():
        if key.startswith("network_") and key.count("_") == 1:
            try:
                ids.append(int(key.split("_")[1]))
            except (TypeError, ValueError):
                continue
    return sorted(set(ids))


def _load_network_entry(raw_payload: Mapping[str, Any], network_id: int) -> Dict[str, Any]:
    key = f"network_{network_id}"
    if key not in raw_payload:
        raise KeyError(f"Missing {key} in collected_samples payload.")
    item = raw_payload[key]
    item = item.item() if isinstance(item, np.ndarray) else item
    if not isinstance(item, dict):
        raise TypeError(f"{key} must contain a dict payload, found {type(item)}")
    return item


def _to_dtype(dtype_name: str) -> np.dtype:
    name = str(dtype_name).lower().strip()
    if name == "float16":
        return np.float16
    if name == "float64":
        return np.float64
    return np.float32


def _write_npz_entry_np_array(zip_file: zipfile.ZipFile, key: str, arr: np.ndarray) -> None:
    with zip_file.open(f"{key}.npy", "w") as f:
        np.lib.format.write_array(f, np.asarray(arr), allow_pickle=False)


def _resolve_output_paths(raw_dir_path: Path, output_file: Optional[str], output_subdir: Optional[str]) -> Tuple[Path, Path]:
    if output_file:
        output_npz = Path(output_file)
        if not output_npz.is_absolute():
            output_npz = raw_dir_path / output_npz
        output_npz.parent.mkdir(parents=True, exist_ok=True)
        metadata_path = output_npz.with_suffix(".metadata.json")
        return output_npz, metadata_path

    # Backward compatibility: if only subdir is provided, place file in that subdir.
    if output_subdir:
        target_dir = raw_dir_path / output_subdir
        target_dir.mkdir(parents=True, exist_ok=True)
        output_npz = target_dir / "wra_h_instantaneous.npz"
        metadata_path = target_dir / "metadata.json"
        return output_npz, metadata_path

    # Default: raw_dir/wra_h_instantaneous.npz
    output_npz = raw_dir_path / "wra_h_instantaneous.npz"
    metadata_path = raw_dir_path / "wra_h_instantaneous.metadata.json"
    return output_npz, metadata_path


def _resolve_per_network_output_dir(raw_dir_path: Path, output_dir: Optional[str]) -> Tuple[Path, Path]:
    if output_dir:
        out_dir = Path(output_dir)
        if not out_dir.is_absolute():
            out_dir = raw_dir_path / out_dir
    else:
        out_dir = raw_dir_path / "wra_h_instantaneous"
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = out_dir / "metadata.json"
    return out_dir, metadata_path


def generate_h_instantaneous(
    *,
    raw_dir: str,
    num_timesteps: Optional[int] = None,
    dtype: str = "float32",
    per_network_files: bool = False,
    output_dir: Optional[str] = None,
    output_file: Optional[str] = None,
    output_subdir: Optional[str] = None,
    overwrite: bool = False,
) -> Tuple[Path, Path]:
    raw_dir_path = Path(raw_dir)
    if not raw_dir_path.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir_path}")

    samples_path = raw_dir_path / "collected_samples.npz"
    network_info_path = raw_dir_path / "network_info.json"
    config_path = raw_dir_path / "config.yaml"
    for p in (samples_path, network_info_path, config_path):
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    with np.load(samples_path, allow_pickle=True) as npz:
        raw_payload = {k: npz[k] for k in npz.files}

    with open(network_info_path, "r") as f:
        network_info = json.load(f)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    resolved_timesteps = (
        int(num_timesteps)
        if num_timesteps is not None
        else int(_get_config_param(config, "num_timesteps", 200))
    )
    if resolved_timesteps <= 0:
        raise ValueError(f"num_timesteps must be positive, got {resolved_timesteps}")

    if per_network_files and (output_file is not None or output_subdir is not None):
        raise ValueError(
            "--per-network-files cannot be combined with --output-file or --output-subdir. "
            "Use --output-dir instead."
        )

    network_ids = _extract_network_ids(raw_payload)
    if not network_ids:
        raise ValueError(f"No networks found in {samples_path}")

    global_channel_version = normalize_channel_version(network_info.get("channel_version"), default="v2")
    global_channel_params = network_info.get("channel_params", {})
    save_dtype = _to_dtype(dtype)

    saved = 0
    bytes_total = 0
    saved_network_ids: List[int] = []
    network_seeds: List[int] = []

    output_path: Path
    metadata_path: Path
    if per_network_files:
        output_dir_path, metadata_path = _resolve_per_network_output_dir(
            raw_dir_path=raw_dir_path,
            output_dir=output_dir,
        )
        output_path = output_dir_path

        for network_id in network_ids:
            net_key = f"network_{network_id}"
            net_info = network_info.get(net_key, {}) if isinstance(network_info.get(net_key, {}), dict) else {}
            net_payload = _load_network_entry(raw_payload, network_id)

            associations = np.asarray(net_payload.get("associations"), dtype=np.float64)
            if associations.ndim != 2:
                raise ValueError(f"{net_key}.associations malformed with shape {associations.shape}")
            n_links = int(associations.shape[0])

            network_seed = net_info.get("network_seed", net_payload.get("network_seed"))
            if network_seed is None:
                raise ValueError(f"{net_key} missing network_seed in both network_info and collected_samples.")

            channel_version = normalize_channel_version(
                net_info.get("channel_version", global_channel_version),
                default=global_channel_version,
            )
            channel_params = net_info.get("channel_params", global_channel_params)

            channel = create_wireless_channel(
                n_links=n_links,
                seed=int(network_seed),
                channel_version=channel_version,
                channel_params=channel_params,
                default_version="v2",
            )
            sampled = channel.sample_realization(num_timesteps=resolved_timesteps)
            # sampled["H"] is (m, n, T); store as (T, m, n)
            h_tmn = np.transpose(np.asarray(sampled["H"], dtype=save_dtype), (2, 0, 1))

            per_network_file = output_dir_path / f"network_{network_id}.npz"
            if per_network_file.exists() and not overwrite:
                raise FileExistsError(
                    f"Output already exists: {per_network_file}. Use --overwrite to replace it."
                )

            np.savez_compressed(
                per_network_file,
                H=h_tmn,
                network_id=np.array(int(network_id), dtype=np.int64),
                network_seed=np.array(int(network_seed), dtype=np.int64),
                channel_version=np.array(channel_version),
                num_timesteps=np.array(resolved_timesteps, dtype=np.int64),
                shape_convention=np.array("H[T,m,n]"),
            )

            saved += 1
            bytes_total += int(h_tmn.nbytes)
            saved_network_ids.append(int(network_id))
            network_seeds.append(int(network_seed))
    else:
        output_npz_path, metadata_path = _resolve_output_paths(
            raw_dir_path=raw_dir_path,
            output_file=output_file,
            output_subdir=output_subdir,
        )
        if output_npz_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output already exists: {output_npz_path}. Use --overwrite to replace it."
            )
        output_path = output_npz_path

        with zipfile.ZipFile(output_npz_path, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            _write_npz_entry_np_array(zf, "schema_version", np.array(1, dtype=np.int64))
            _write_npz_entry_np_array(zf, "format", np.array("wra_h_instantaneous_v2"))
            _write_npz_entry_np_array(zf, "shape_convention", np.array("H[T,m,n]"))
            _write_npz_entry_np_array(zf, "num_timesteps", np.array(resolved_timesteps, dtype=np.int64))
            _write_npz_entry_np_array(zf, "dtype", np.array(str(save_dtype)))
            _write_npz_entry_np_array(zf, "channel_version", np.array(global_channel_version))

            for network_id in network_ids:
                net_key = f"network_{network_id}"
                net_info = network_info.get(net_key, {}) if isinstance(network_info.get(net_key, {}), dict) else {}
                net_payload = _load_network_entry(raw_payload, network_id)

                associations = np.asarray(net_payload.get("associations"), dtype=np.float64)
                if associations.ndim != 2:
                    raise ValueError(f"{net_key}.associations malformed with shape {associations.shape}")
                n_links = int(associations.shape[0])

                network_seed = net_info.get("network_seed", net_payload.get("network_seed"))
                if network_seed is None:
                    raise ValueError(f"{net_key} missing network_seed in both network_info and collected_samples.")

                channel_version = normalize_channel_version(
                    net_info.get("channel_version", global_channel_version),
                    default=global_channel_version,
                )
                channel_params = net_info.get("channel_params", global_channel_params)

                channel = create_wireless_channel(
                    n_links=n_links,
                    seed=int(network_seed),
                    channel_version=channel_version,
                    channel_params=channel_params,
                    default_version="v2",
                )
                sampled = channel.sample_realization(num_timesteps=resolved_timesteps)
                # sampled["H"] is (m, n, T); store as (T, m, n)
                h_tmn = np.transpose(np.asarray(sampled["H"], dtype=save_dtype), (2, 0, 1))

                _write_npz_entry_np_array(zf, f"network_{network_id}_H", h_tmn)
                _write_npz_entry_np_array(zf, f"network_{network_id}_seed", np.array(int(network_seed), dtype=np.int64))
                _write_npz_entry_np_array(zf, f"network_{network_id}_channel_version", np.array(channel_version))

                saved += 1
                bytes_total += int(h_tmn.nbytes)
                saved_network_ids.append(int(network_id))
                network_seeds.append(int(network_seed))

            _write_npz_entry_np_array(zf, "network_ids", np.asarray(saved_network_ids, dtype=np.int64))
            _write_npz_entry_np_array(zf, "network_seeds", np.asarray(network_seeds, dtype=np.int64))

    metadata = {
        "format": "wra_h_instantaneous_per_network_v1" if per_network_files else "wra_h_instantaneous_v2",
        "shape_convention": "H[T,m,n]",
        "num_networks": int(len(network_ids)),
        "num_timesteps": int(resolved_timesteps),
        "dtype": str(save_dtype),
        "saved_network_files": int(saved),  # Number of networks serialized
        "network_ids": [int(x) for x in saved_network_ids],
        "estimated_uncompressed_bytes": int(bytes_total),
        "estimated_uncompressed_gb": float(bytes_total / 1e9),
        "output_path": str(output_path),
        "storage_mode": "per_network_files" if per_network_files else "single_file",
        "source_raw_dir": str(raw_dir_path),
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    if per_network_files:
        print(f"✓ Saved compressed H_instantaneous per-network files to: {output_path}")
    else:
        print(f"✓ Saved compressed H_instantaneous file: {output_path}")
    print(f"  Metadata: {metadata_path}")
    print(f"  Networks total: {len(network_ids)}")
    print(f"  Saved networks: {saved}")
    print(f"  Timesteps per network: {resolved_timesteps}")
    print(f"  Stored dtype: {save_dtype}")
    print(f"  Estimated uncompressed size: {bytes_total / 1e9:.3f} GB")
    return output_path, metadata_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save compressed H_instantaneous for converted raw WRA datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        required=True,
        help="Converted raw WRA directory (contains collected_samples.npz/network_info.json/config.yaml).",
    )
    parser.add_argument(
        "--num-timesteps",
        type=int,
        default=None,
        help="Number of timesteps per network. Defaults to config's num_timesteps or 200.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Stored dtype for H tensors.",
    )
    parser.add_argument(
        "--per-network-files",
        action="store_true",
        help=(
            "Store one file per network under <raw-dir>/wra_h_instantaneous/network_<id>.npz "
            "instead of a single large NPZ."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Output directory for --per-network-files mode. Relative paths are resolved under raw-dir. "
            "Default: <raw-dir>/wra_h_instantaneous"
        ),
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help=(
            "Output NPZ file path. Relative paths are resolved under raw-dir. "
            "Default: <raw-dir>/wra_h_instantaneous.npz"
        ),
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default=None,
        help=(
            "Deprecated compatibility option. If set and --output-file is omitted, "
            "writes <raw-dir>/<output-subdir>/wra_h_instantaneous.npz."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output file if present.",
    )
    args = parser.parse_args()

    generate_h_instantaneous(
        raw_dir=args.raw_dir,
        num_timesteps=args.num_timesteps,
        dtype=args.dtype,
        per_network_files=args.per_network_files,
        output_dir=args.output_dir,
        output_file=args.output_file,
        output_subdir=args.output_subdir,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
