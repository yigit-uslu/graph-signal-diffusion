import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_subprocess(cmd: list[str], *, cwd: Path, timeout: int = 300) -> None:
    env = dict(os.environ)
    py_path = env.get("PYTHONPATH", "")
    src_path = str(REPO_ROOT / "src")
    env["PYTHONPATH"] = src_path if not py_path else f"{src_path}{os.pathsep}{py_path}"
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"Command failed: {' '.join(cmd)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )


def _run_train_pd_conditional(tmp_path: Path) -> Path:
    run_dir = tmp_path / "pd_run"
    work_dir = tmp_path / "pd_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "graph_signal_diffusion.cli.wra.train_pd",
        "--config-name=pd_training/wra_small",
        "dataset.num_networks=1",
        "dataset.n_links=2",
        "dataset.num_timesteps=2",
        "training.batch_size=1",
        "training.max_epochs=2",
        "training.num_samples_per_network=1",
        "training.moving_avg_window=1",
        "training.convergence_window=1",
        "training.convergence_warmup_epochs=999",
        "training.convergence_patience=1",
        "training.dual_update_frequency=1",
        "training.alpha_dual=0.01",
        "training.dual_momentum=0.0",
        "model.hidden_dim=8",
        "model.num_layers=1",
        "model.K=1",
        "training.constraint_profile.source=explicit",
        "training.constraint_profile.explicit.profiles=[0.2,0.4]",
        "training.constraint_profile.explicit.names=[low,high]",
        f"hydra.run.dir={run_dir}",
    ]
    _run_subprocess(cmd, cwd=work_dir)
    return run_dir


def _run_build_diffusion_dataset(
    *,
    pd_run_dir: Path,
    sample_source: str,
    out_dir: Path,
    work_dir: Path,
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "graph_signal_diffusion.cli.wra.build_diffusion_dataset",
        "--config-name=pd_collection/wra_small",
        f"input_dir={pd_run_dir}",
        f"collection.sample_source={sample_source}",
        "collection.target_samples_per_network=1",
        "collection.primal_history.window_size=10",
        "collection.primal_history.refine_feasible_subset=true",
        "collection.primal_history.refine_objective=min_rate",
        f"output.raw_wra_dir={out_dir}",
        "output.force=true",
        "output.h_instantaneous.enabled=false",
        f"hydra.run.dir={work_dir / f'build_{sample_source}'}",
    ]
    _run_subprocess(cmd, cwd=work_dir)


def test_pd_to_diffusion_dataset_cli_smoke_npz_and_primal_history(tmp_path):
    pd_run_dir = _run_train_pd_conditional(tmp_path)

    for sample_source in ("npz", "primal_history"):
        out_dir = tmp_path / f"raw_{sample_source}"
        work_dir = tmp_path / f"collect_work_{sample_source}"
        work_dir.mkdir(parents=True, exist_ok=True)
        _run_build_diffusion_dataset(
            pd_run_dir=pd_run_dir,
            sample_source=sample_source,
            out_dir=out_dir,
            work_dir=work_dir,
        )

        assert (out_dir / "collected_samples.npz").exists()
        assert (out_dir / "network_info.json").exists()
        with open(out_dir / "network_info.json") as f:
            info = json.load(f)

        assert info["has_per_network_r_min"] is True
        assert info["constraint_mode"] == "per_network_r_min"

        network_keys = sorted(k for k in info.keys() if k.startswith("network_"))
        assert len(network_keys) == 2

        base_ids = [int(info[k]["base_network_id"]) for k in network_keys]
        profile_ids = [int(info[k]["constraint_profile_id"]) for k in network_keys]
        assert all(base_id == 0 for base_id in base_ids)
        assert set(profile_ids) == {0, 1}
        for key in network_keys:
            assert "constraint_profile_name" in info[key]
            assert "r_min_per_receiver" in info[key]

        if sample_source == "primal_history":
            for key in network_keys:
                assert "selection_summary" in info[key]
