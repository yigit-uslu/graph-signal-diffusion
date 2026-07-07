#!/bin/bash
# Quick reference for baseline evaluation commands

# ============================================================
# COMPARE MULTIPLE BASELINES
# ============================================================

# Compare GRW and Diffusion on SP100 (default)
python -m graph_signal_diffusion.cli.compare_baselines

# Compare on SP500
python -m graph_signal_diffusion.cli.compare_baselines dataset=sp500

# Compare only GRW (no diffusion)
python -m graph_signal_diffusion.cli.compare_baselines baselines_to_compare=[grw]

# Override GRW parameters
python -m graph_signal_diffusion.cli.compare_baselines \
    baselines.grw.n_samples=50 \
    baselines.grw.shrinkage_strength=20.0

# ============================================================
# EVALUATE SINGLE BASELINE
# ============================================================

# Evaluate GRW only
python -m graph_signal_diffusion.cli.evaluate baseline=grw

# Evaluate with different dataset
python -m graph_signal_diffusion.cli.evaluate baseline=grw dataset=sp500

# Evaluate diffusion from checkpoint
python -m graph_signal_diffusion.cli.evaluate \
    baseline=diffusion \
    checkpoint_path=checkpoints/primal_dual/checkpoint_epoch_4000.pt

# ============================================================
# HYPERPARAMETER EXPERIMENTS
# ============================================================

# Test different shrinkage strengths
for k in 5.0 10.0 20.0 50.0; do
    python -m graph_signal_diffusion.cli.evaluate \
        baseline=grw \
        baselines.grw.shrinkage_strength=$k \
        hydra.run.dir=outputs/grw_sweep/k_${k}
done

# Test different sample counts
for n in 10 20 50 100; do
    python -m graph_signal_diffusion.cli.evaluate \
        baseline=grw \
        baselines.grw.n_samples=$n \
        hydra.run.dir=outputs/grw_sweep/n_${n}
done

# ============================================================
# OUTPUTS
# ============================================================

# Results are saved to:
# - outputs/<date>/<time>/baseline_comparison.csv   # Metrics table
# - outputs/<date>/<time>/<baseline>/                # Per-baseline visualizations
# - outputs/<date>/<time>/comparison_*.png           # Comparison plots
