"""WRA dataset generation pipeline CLI entry points.

Pipeline stages (run in order):

1. ``analyze_channels``      — generate and cache WRA channel networks;
                               compute full-power rate statistics to guide r_min.
2. ``train_pd``              — train PD expert GNN; collect expert trajectory
                               samples for diffusion model training.
3. ``build_diffusion_dataset`` — convert PD expert samples into a processed
                               WRA diffusion dataset.
4. ``verify_pd_samples``     — load and verify collected PD expert samples.
"""
