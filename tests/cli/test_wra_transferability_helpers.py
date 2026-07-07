"""Unit tests for WRA transferability helper utilities."""

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from graph_signal_diffusion.cli._wra_transferability import (
    build_channel_cache_info_entry,
    build_or_load_transfer_channel_cache,
    enforce_pmax_compatibility,
    extract_graph_build_cfg,
    inject_diffusion_checkpoint_path,
    resolve_target_r_min,
    select_eval_network_ids,
    validate_transfer_baselines,
)
from graph_signal_diffusion.tasks.wireless_resource_allocation.evaluator import (
    WirelessResourceAllocationTask,
)


def test_resolve_target_r_min_inherit_prefers_constraint_profile_scalar_value() -> None:
    source_cfg = OmegaConf.create(
        {
            "training": {
                "constraint_profile": {"scalar": {"value": 0.73}},
                "r_min": 0.21,
            }
        }
    )

    value, key = resolve_target_r_min("inherit", source_cfg)

    assert value == pytest.approx(0.73)
    assert key == "training.constraint_profile.scalar.value"


def test_resolve_target_r_min_inherit_falls_back_to_training_r_min() -> None:
    source_cfg = OmegaConf.create(
        {
            "training": {
                "constraint_profile": {"scalar": {"value": None}},
                "r_min": 0.61,
            }
        }
    )

    value, key = resolve_target_r_min("inherit", source_cfg)

    assert value == pytest.approx(0.61)
    assert key == "training.r_min"


def test_resolve_target_r_min_inherit_falls_back_from_unresolved_interpolation() -> None:
    source_cfg = {
        "training": {
            "constraint_profile": {"scalar": {"value": "${training.r_min}"}},
            "r_min": 0.57,
        }
    }

    value, key = resolve_target_r_min("inherit", source_cfg)

    assert value == pytest.approx(0.57)
    assert key == "training.r_min"


def test_resolve_target_r_min_inherit_raises_when_missing() -> None:
    source_cfg = OmegaConf.create({"training": {"constraint_profile": {"scalar": {}}}})

    with pytest.raises(ValueError, match="Could not resolve inherited target r_min"):
        resolve_target_r_min("inherit", source_cfg)


def test_channel_cache_info_entry_uses_absolute_file_path(tmp_path: Path) -> None:
    transfer_root = tmp_path / "data" / "wra_channel_cache_transfer"
    cache_file = transfer_root / "wra_large_outdoor_mid_density" / "cache.pt"
    entry = build_channel_cache_info_entry(
        cache_file=cache_file,
        cache_metadata={"channel_cache_key": "abc123", "seed": 42},
    )

    resolved_cache_file = Path(entry["channel_cache_file"])
    assert resolved_cache_file.is_absolute()
    assert str(transfer_root.resolve()) in str(resolved_cache_file)


def test_inject_diffusion_checkpoint_path_updates_baseline_config() -> None:
    cfg = OmegaConf.create(
        {"baselines": {"diffusion": {"name": "diffusion", "checkpoint_path": None}}}
    )

    inject_diffusion_checkpoint_path(cfg, "/tmp/source_ckpt.pt")

    assert cfg.baselines.diffusion.checkpoint_path == "/tmp/source_ckpt.pt"


def test_validate_transfer_baselines_rejects_ap_in_expert_free_mode() -> None:
    with pytest.raises(ValueError, match="AveragePowerBaseline"):
        validate_transfer_baselines(["fp", "ap", "diffusion"], reference_policy="none")


def test_enforce_pmax_compatibility_raises_by_default() -> None:
    with pytest.raises(ValueError, match="P_max_dBm mismatch"):
        enforce_pmax_compatibility(
            source_p_max_dBm=10.0,
            target_p_max_dBm=8.0,
            allow_mismatch=False,
        )


def test_enforce_pmax_compatibility_allows_override() -> None:
    enforce_pmax_compatibility(
        source_p_max_dBm=10.0,
        target_p_max_dBm=8.0,
        allow_mismatch=True,
    )


def test_select_eval_network_ids_is_first_n_and_checks_bounds() -> None:
    selected = select_eval_network_ids(cache_num_networks=16, eval_num_networks=5)
    assert selected == [0, 1, 2, 3, 4]

    with pytest.raises(ValueError, match="eval_num_networks must be <= cache_num_networks"):
        select_eval_network_ids(cache_num_networks=8, eval_num_networks=9)


def test_extract_graph_build_cfg_includes_self_loop_weight_scale() -> None:
    source_cfg = OmegaConf.create(
        {
            "dataset": {
                "top_k": 6,
                "min_degree": 2,
                "edge_normalize_method": "log1p",
                "self_loop_weight_scale": 1.75,
                "spectral_normalize_edge_weights": True,
            }
        }
    )

    cfg = extract_graph_build_cfg(source_cfg)

    assert cfg["self_loop_weight_scale"] == pytest.approx(1.75)
    assert cfg["top_k"] == 6
    assert cfg["min_degree"] == 2


def test_transfer_cache_missing_raises_when_auto_generate_disabled(tmp_path: Path) -> None:
    scenario_cfg = OmegaConf.create(
        {
            "seed": 42,
            "dataset": {
                "n_links": 4,
                "deployment_range": 100.0,
            },
            "channel": {"version": "v2"},
        }
    )

    with pytest.raises(FileNotFoundError, match="auto-generation disabled"):
        build_or_load_transfer_channel_cache(
            scenario_cfg=scenario_cfg,
            scenario_name="wra_large_outdoor_mid_density",
            cache_root=tmp_path / "wra_channel_cache_transfer",
            cache_num_networks=8,
            auto_generate=False,
            require_existing_cache=False,
        )


def test_wra_task_reference_policy_none_excludes_reference_metrics() -> None:
    task = WirelessResourceAllocationTask(
        num_channel_realizations=4,
        eval_num_realizations=4,
        ergodic_window_size=2,
        num_eval_batches=1,
        reference_policy="none",
    )
    gen_tx_powers_all = torch.tensor([[0.2], [0.4]], dtype=torch.float32)
    h_samples = torch.ones((4, 1, 1), dtype=torch.float32)
    associations = torch.ones((1, 1), dtype=torch.float32)

    metrics, record = task._evaluate_time_shared(
        gen_tx_powers_all=gen_tx_powers_all,
        real_tx_powers_all=None,
        H_samples=h_samples,
        associations=associations,
        noise_var=1e-3,
        P_max=1.0,
        r_min=0.1,
        network_id=0,
        dataset_name="dummy",
        eval_batch_idx=0,
        include_reference=False,
    )

    assert "sum_rate_generated" in metrics
    assert "sum_rate_real" not in metrics
    assert "rate_violation_percentage_real" not in metrics
    assert "real_rates_final" not in record
