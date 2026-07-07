import numpy as np

from graph_signal_diffusion.datasets.wra.sample_schema import (
    build_pd_samples_npz_payload,
    load_pd_samples_npz,
    save_pd_samples_npz,
)


def _mock_samples() -> dict:
    return {
        0: {
            "network_seed": 42,
            "associations": np.eye(3, dtype=np.float32),
            "H_instantaneous": np.ones((4, 3, 3), dtype=np.float32),
            "power_samples": [
                {
                    "power": np.array([0.1, 0.2, 0.3], dtype=np.float32),
                    "rates": np.array([1.0, 1.1, 1.2], dtype=np.float32),
                    "checkpoint_epoch": 1,
                }
            ],
        }
    }


def test_build_payload_omits_h_instantaneous_by_default():
    payload = build_pd_samples_npz_payload(_mock_samples(), channel_version="v2")
    assert "H_instantaneous" not in payload


def test_load_pd_samples_handles_missing_h_instantaneous(tmp_path):
    samples_path = tmp_path / "samples_no_h.npz"
    save_pd_samples_npz(samples_path, _mock_samples(), channel_version="v2")

    canonical = load_pd_samples_npz(samples_path)
    net = canonical["networks"][0]

    assert canonical.get("source_has_h_instantaneous") is False
    assert net["associations"].shape == (3, 3)
    assert net["H_instantaneous"].shape == (0, 3, 3)
    assert net["H_instantaneous"].dtype == np.float32


def test_optional_constraint_metadata_roundtrip(tmp_path):
    samples = _mock_samples()
    samples[0]["r_min_per_receiver"] = np.array([0.5, 0.6, 0.7], dtype=np.float32)
    samples[0]["base_network_id"] = 12
    samples[0]["constraint_profile_id"] = 3
    samples[0]["constraint_profile_name"] = "hard"

    samples_path = tmp_path / "samples_with_constraint_metadata.npz"
    save_pd_samples_npz(samples_path, samples, channel_version="v2")
    canonical = load_pd_samples_npz(samples_path)
    net = canonical["networks"][0]

    assert np.allclose(net["r_min_per_receiver"], np.array([0.5, 0.6, 0.7], dtype=np.float32))
    assert net["base_network_id"] == 12
    assert net["constraint_profile_id"] == 3
    assert net["constraint_profile_name"] == "hard"
