import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from graph_signal_diffusion.datasets.wra.sample_schema import load_pd_samples_npz


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_train_pd_cli_smoke(tmp_path: Path, mode: str) -> tuple[Path, int]:
    run_dir = tmp_path / f"run_{mode}"
    work_dir = tmp_path / f"work_{mode}"
    work_dir.mkdir(parents=True, exist_ok=True)

    base_overrides = [
        "--config-name=pd_training/wra_small",
        "dataset.num_networks=1",
        "dataset.n_links=2",
        "dataset.num_timesteps=2",
        "training.batch_size=1",
        "training.max_epochs=1",
        "training.num_samples_per_network=1",
        "training.moving_avg_window=1",
        "training.convergence_window=1",
        "training.convergence_warmup_epochs=0",
        "training.convergence_patience=1",
        "training.dual_update_frequency=1",
        "training.alpha_dual=0.01",
        "training.dual_momentum=0.0",
        "model.hidden_dim=8",
        "model.num_layers=1",
        "model.K=1",
        f"hydra.run.dir={run_dir}",
    ]

    if mode == "scalar":
        expected_networks = 1
        mode_overrides = [
            "training.constraint_profile.source=scalar",
            "training.r_min=0.3",
            "training.constraint_profile.scalar.value=0.3",
        ]
    elif mode == "explicit":
        expected_networks = 2
        mode_overrides = [
            "training.constraint_profile.source=explicit",
            "training.constraint_profile.explicit.profiles=[0.2,0.4]",
            "training.constraint_profile.explicit.names=[low,high]",
        ]
    elif mode == "sampled":
        expected_networks = 2
        mode_overrides = [
            "training.constraint_profile.source=sampled",
            "training.constraint_profile.sampled.min=0.2",
            "training.constraint_profile.sampled.max=0.8",
            "training.constraint_profile.sampled.count=2",
            "training.constraint_profile.sampled.seed=7",
        ]
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    cmd = [
        sys.executable,
        "-m",
        "graph_signal_diffusion.cli.wra.train_pd",
        *base_overrides,
        *mode_overrides,
    ]
    env = dict(os.environ)
    py_path = env.get("PYTHONPATH", "")
    src_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src_path if not py_path else f"{src_path}{os.pathsep}{py_path}"

    proc = subprocess.run(
        cmd,
        cwd=work_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=240,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"train_pd CLI failed for mode={mode}\n"
            f"Command: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return run_dir, expected_networks


@pytest.mark.parametrize("mode", ["scalar", "explicit", "sampled"])
def test_train_pd_cli_smoke_modes(tmp_path, mode):
    run_dir, expected_networks = _run_train_pd_cli_smoke(tmp_path, mode)

    npz_path = run_dir / "collected_samples.npz"
    metadata_path = run_dir / "collection_metadata.json"
    assert npz_path.exists(), f"Missing sample artifact for mode={mode}"
    assert metadata_path.exists(), f"Missing collection metadata for mode={mode}"

    with open(metadata_path) as f:
        metadata = json.load(f)
    assert "r_min_summary" in metadata
    r_min_summary = metadata["r_min_summary"]
    assert "r_min_is_scalar" in r_min_summary
    if mode == "scalar":
        assert r_min_summary["r_min_is_scalar"] is True
        assert r_min_summary["r_min"] is not None
        assert r_min_summary["r_min_min"] == pytest.approx(r_min_summary["r_min_max"])
    else:
        assert r_min_summary["r_min_is_scalar"] is False
        assert r_min_summary["r_min"] is None
        assert r_min_summary["r_min_min"] < r_min_summary["r_min_max"]

    with np.load(npz_path, allow_pickle=True) as npz:
        files = set(npz.files)
        network_ids = np.asarray(npz["network_ids"], dtype=np.int64)
        assert network_ids.shape[0] == expected_networks

        if mode == "scalar":
            assert "base_network_ids" not in files
            assert "constraint_profile_ids" not in files
            assert "constraint_profile_names" not in files
        else:
            assert "base_network_ids" in files
            assert "constraint_profile_ids" in files
            assert "constraint_profile_names" in files
            assert "r_min_per_receiver_per_network" in files

            base_ids = np.asarray(npz["base_network_ids"], dtype=np.int64)
            profile_ids = np.asarray(npz["constraint_profile_ids"], dtype=np.int64)
            assert base_ids.shape[0] == expected_networks
            assert profile_ids.shape[0] == expected_networks
            assert np.all(base_ids == 0)  # one base network expanded
            assert set(profile_ids.tolist()) == set(range(expected_networks))

    canonical = load_pd_samples_npz(npz_path)
    assert len(canonical["network_ids"]) == expected_networks
