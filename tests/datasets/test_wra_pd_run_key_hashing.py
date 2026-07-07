from graph_signal_diffusion.datasets.wra.channel_factory import build_wra_pd_run_key


def _base_pd_key_kwargs() -> dict:
    return {
        "channel_cache_key": "wrach_v1_dummy",
        "seed": 42,
        "P_max_dBm": 10.0,
        "bandwidth_Hz": 10_000_000.0,
        "noise_psd_dBm_Hz": -174.0,
        "model_type": "gnn",
        "model_hidden_dim": 64,
        "model_num_layers": 3,
        "model_K": 2,
        "model_norm_type": "layer",
        "model_dropout": 0.1,
        "model_activation": "silu",
        "training_r_min": 0.5,
        "training_constraint_profile_type": "min_rate",
        "training_constraint_profile_source": "scalar",
        "training_constraint_profile_scalar_value": 0.5,
        "training_constraint_profile_explicit_profiles": None,
        "training_constraint_profile_sampled_min": None,
        "training_constraint_profile_sampled_max": None,
        "training_constraint_profile_sampled_count": None,
        "training_constraint_profile_sampled_seed": None,
        "training_constraint_profile_r_min_feature_scale": 1.0,
        "training_alpha_dual": 0.2,
        "training_dual_momentum": 0.5,
        "training_learning_rate": 5e-4,
        "training_batch_size": 16,
        "training_max_epochs": 1000,
        "training_convergence_window": 200,
        "training_convergence_warmup_epochs": 1000,
        "training_convergence_patience": 50,
        "training_gradient_norm_threshold": float("inf"),
        "training_dual_variance_threshold": float("inf"),
        "training_dual_stationarity_threshold": 0.3,
        "training_violation_fraction_threshold": 1.0,
        "training_violation_fraction_on_model_avg_rates_threshold": 0.05,
        "training_mean_violation_slack_on_model_avg_rates_threshold": 0.01,
        "training_dual_update_mode": "step",
        "training_dual_update_frequency": 2,
        "training_num_samples_per_network": 200,
        "training_moving_avg_window": 200,
        "training_sample_collection_interval": None,
    }


def test_pd_run_key_explicit_source_ignores_inactive_sampled_fields():
    kwargs = _base_pd_key_kwargs()
    kwargs.update(
        {
            "training_constraint_profile_source": "explicit",
            "training_constraint_profile_scalar_value": None,
            "training_constraint_profile_explicit_profiles": [0.5, [0.3, 0.4, 0.5]],
            "training_constraint_profile_sampled_min": 0.1,
            "training_constraint_profile_sampled_max": 1.0,
            "training_constraint_profile_sampled_count": 7,
            "training_constraint_profile_sampled_seed": 11,
        }
    )
    key_a = build_wra_pd_run_key(**kwargs)

    kwargs["training_constraint_profile_sampled_min"] = 0.25
    kwargs["training_constraint_profile_sampled_max"] = 0.75
    kwargs["training_constraint_profile_sampled_count"] = 3
    kwargs["training_constraint_profile_sampled_seed"] = 999
    key_b = build_wra_pd_run_key(**kwargs)

    assert key_a == key_b


def test_pd_run_key_sampled_source_ignores_inactive_explicit_fields():
    kwargs = _base_pd_key_kwargs()
    kwargs.update(
        {
            "training_constraint_profile_source": "sampled",
            "training_constraint_profile_scalar_value": None,
            "training_constraint_profile_explicit_profiles": [0.1, 0.2, 0.3],
            "training_constraint_profile_sampled_min": 0.2,
            "training_constraint_profile_sampled_max": 0.8,
            "training_constraint_profile_sampled_count": 5,
            "training_constraint_profile_sampled_seed": 22,
        }
    )
    key_a = build_wra_pd_run_key(**kwargs)

    kwargs["training_constraint_profile_explicit_profiles"] = [0.9, 0.95]
    key_b = build_wra_pd_run_key(**kwargs)

    assert key_a == key_b


def test_pd_run_key_explicit_source_changes_when_profiles_change():
    kwargs = _base_pd_key_kwargs()
    kwargs.update(
        {
            "training_constraint_profile_source": "explicit",
            "training_constraint_profile_scalar_value": None,
            "training_constraint_profile_explicit_profiles": [0.4, 0.6],
        }
    )
    key_a = build_wra_pd_run_key(**kwargs)

    kwargs["training_constraint_profile_explicit_profiles"] = [0.4, 0.7]
    key_b = build_wra_pd_run_key(**kwargs)

    assert key_a != key_b
