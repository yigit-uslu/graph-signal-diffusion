import threading
from pathlib import Path

import numpy as np
import torch

from graph_signal_diffusion.datasets.wra.utils import (
    compute_ergodic_rates,
    compute_ergodic_rates_batched,
)
from graph_signal_diffusion.tasks.wireless_resource_allocation.evaluator import (
    WirelessResourceAllocationTask,
)


def _write_h_timeslots(timeslot_dir: Path, h_tmn: np.ndarray) -> None:
    timeslot_dir.mkdir(parents=True, exist_ok=True)
    for t_idx in range(h_tmn.shape[0]):
        torch.save(torch.from_numpy(h_tmn[t_idx]), timeslot_dir / f"timestep_{t_idx}.pt")


def test_compute_ergodic_rates_batched_matches_single_two_policies():
    torch.manual_seed(0)
    T, m, n = 37, 16, 16
    H = torch.rand(T, m, n, dtype=torch.float32)
    associations = torch.eye(m, n, dtype=torch.float32)
    powers = torch.rand(2, m, dtype=torch.float32)
    noise_var = 1e-10

    batched = compute_ergodic_rates_batched(powers, H, associations, noise_var)
    ref0 = compute_ergodic_rates(powers[0], H, associations, noise_var)
    ref1 = compute_ergodic_rates(powers[1], H, associations, noise_var)
    refs = torch.stack([ref0, ref1], dim=0)

    max_abs_err = (batched - refs).abs().max().item()
    assert max_abs_err <= 1e-6


def test_compute_ergodic_rates_batched_matches_single_many_policies():
    torch.manual_seed(1)
    T, m, n, P = 51, 24, 24, 5
    H = torch.rand(T, m, n, dtype=torch.float32)
    associations = torch.eye(m, n, dtype=torch.float32)
    powers = torch.rand(P, m, dtype=torch.float32)
    noise_var = 1e-10

    batched = compute_ergodic_rates_batched(powers, H, associations, noise_var)
    refs = torch.stack(
        [compute_ergodic_rates(powers[p_idx], H, associations, noise_var) for p_idx in range(P)],
        dim=0,
    )
    max_abs_err = (batched - refs).abs().max().item()
    assert max_abs_err <= 1e-6


def test_h_sidecar_auto_matches_legacy(tmp_path):
    rng = np.random.default_rng(0)
    T, m, n = 20, 6, 6
    timeslot_dir = tmp_path / "H_instantaneous"
    h_tmn = rng.random((T, m, n), dtype=np.float32)
    _write_h_timeslots(timeslot_dir, h_tmn)

    task_legacy = WirelessResourceAllocationTask(
        eval_num_realizations=9,
        ergodic_window_size=3,
        h_io_mode="legacy",
    )
    task_auto = WirelessResourceAllocationTask(
        eval_num_realizations=9,
        ergodic_window_size=3,
        h_io_mode="auto",
    )

    np.random.seed(42)
    h_legacy = task_legacy._load_precomputed_h_samples(
        timeslot_dir=str(timeslot_dir),
        num_available=T,
        dataset_name="wra-legacy",
        network_id=0,
    )
    np.random.seed(42)
    h_auto = task_auto._load_precomputed_h_samples(
        timeslot_dir=str(timeslot_dir),
        num_available=T,
        dataset_name="wra-auto",
        network_id=0,
    )

    sidecar_path = timeslot_dir / "H_instantaneous_tmn.npy"
    assert sidecar_path.exists()
    assert h_legacy.shape == h_auto.shape
    assert np.max(np.abs(h_legacy - h_auto)) <= 1e-6


def test_h_sidecar_auto_falls_back_when_sidecar_is_corrupt(tmp_path):
    rng = np.random.default_rng(1)
    T, m, n = 18, 5, 5
    timeslot_dir = tmp_path / "H_instantaneous"
    h_tmn = rng.random((T, m, n), dtype=np.float32)
    _write_h_timeslots(timeslot_dir, h_tmn)

    sidecar_path = timeslot_dir / "H_instantaneous_tmn.npy"
    sidecar_path.write_bytes(b"not-a-valid-npy")

    task_legacy = WirelessResourceAllocationTask(
        eval_num_realizations=8,
        ergodic_window_size=2,
        h_io_mode="legacy",
    )
    task_auto = WirelessResourceAllocationTask(
        eval_num_realizations=8,
        ergodic_window_size=2,
        h_io_mode="auto",
    )

    np.random.seed(7)
    h_legacy = task_legacy._load_precomputed_h_samples(
        timeslot_dir=str(timeslot_dir),
        num_available=T,
        dataset_name="wra-legacy",
        network_id=0,
    )
    np.random.seed(7)
    h_auto = task_auto._load_precomputed_h_samples(
        timeslot_dir=str(timeslot_dir),
        num_available=T,
        dataset_name="wra-auto",
        network_id=0,
    )

    assert h_legacy.shape == h_auto.shape
    assert np.max(np.abs(h_legacy - h_auto)) <= 1e-6


def test_h_sidecar_lazy_build_atomic_under_thread_race(tmp_path):
    rng = np.random.default_rng(2)
    T, m, n = 24, 4, 4
    timeslot_dir = tmp_path / "H_instantaneous"
    h_tmn = rng.random((T, m, n), dtype=np.float32)
    _write_h_timeslots(timeslot_dir, h_tmn)

    task = WirelessResourceAllocationTask(
        eval_num_realizations=8,
        ergodic_window_size=2,
        h_io_mode="auto",
    )
    timeslot_files, available = task._resolve_timeslot_files(
        timeslot_path=timeslot_dir,
        num_available=T,
    )
    assert available == T

    barrier = threading.Barrier(2)
    errors = []

    def _builder_thread() -> None:
        try:
            barrier.wait(timeout=10)
            task._build_h_sidecar_from_timeslots(
                sidecar_path=timeslot_dir / "H_instantaneous_tmn.npy",
                timeslot_files=timeslot_files,
                dataset_name="wra-race",
                network_id=0,
            )
        except Exception as exc:  # pragma: no cover - failure path assertion below
            errors.append(exc)

    t1 = threading.Thread(target=_builder_thread)
    t2 = threading.Thread(target=_builder_thread)
    t1.start()
    t2.start()
    t1.join(timeout=30)
    t2.join(timeout=30)
    assert not errors

    sidecar_path = timeslot_dir / "H_instantaneous_tmn.npy"
    assert sidecar_path.exists()
    h_sidecar = np.load(sidecar_path, mmap_mode="r")
    assert h_sidecar.shape == (T, m, n)


def test_evaluate_time_shared_matches_single_pass_reference():
    torch.manual_seed(3)
    T0 = 3
    num_slots = 4
    T = T0 * num_slots
    m = n = 7
    noise_var = 1e-10

    task = WirelessResourceAllocationTask(
        eval_num_realizations=T,
        ergodic_window_size=T0,
        num_eval_batches=1,
        eval_h_chunk_size=5,
        h_io_mode="legacy",
    )
    associations = torch.eye(m, n, dtype=torch.float32)
    gen_tx_powers_all = torch.rand(1, m, dtype=torch.float32)  # K=1 => deterministic sampling
    real_tx_powers_all = torch.rand(1, m, dtype=torch.float32)
    H_samples = torch.rand(T, m, n, dtype=torch.float32)  # CPU tensor exercises streaming path

    metrics, record = task._evaluate_time_shared(
        gen_tx_powers_all=gen_tx_powers_all,
        real_tx_powers_all=real_tx_powers_all,
        H_samples=H_samples,
        associations=associations,
        noise_var=noise_var,
        P_max=1.0,
        r_min=0.5,
        network_id=0,
        dataset_name="wra-test",
        eval_batch_idx=0,
    )

    ref_gen = []
    ref_real = []
    for slot_idx in range(num_slots):
        start = slot_idx * T0
        end = start + T0
        H_window = H_samples[start:end]
        ref_gen.append(
            compute_ergodic_rates(gen_tx_powers_all[0], H_window, associations, noise_var)
        )
        ref_real.append(
            compute_ergodic_rates(real_tx_powers_all[0], H_window, associations, noise_var)
        )
    ref_gen = torch.stack(ref_gen, dim=0)
    ref_real = torch.stack(ref_real, dim=0)

    assert np.max(np.abs(record["gen_rates_per_slot"] - ref_gen.numpy())) <= 1e-6
    assert np.max(np.abs(record["real_rates_per_slot"] - ref_real.numpy())) <= 1e-6
    assert abs(metrics["sum_rate_generated"] - float(ref_gen.mean(dim=0).sum().item())) <= 1e-6
    assert abs(metrics["sum_rate_real"] - float(ref_real.mean(dim=0).sum().item())) <= 1e-6
