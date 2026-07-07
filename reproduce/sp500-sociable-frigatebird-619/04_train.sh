#!/bin/bash
# Stage 6 — Train sociable-frigatebird-619 from scratch (the small ~1.88M DS8 model).
#
#   *** OPTIONAL / EXPENSIVE — ~5000 epochs, on the order of days on one GPU. ***
#
# seed=0 is pinned, but AMP + cuDNN are not bitwise-deterministic, so a fresh run
# yields a STATISTICALLY EQUIVALENT model, NOT the identical checkpoint — and
# therefore not the exact test_summary.json numbers. To reproduce the EXACT paper
# numbers, DO NOT retrain: run ./05_evaluate_checkpoint.sh against the shipped
# best_model_epoch_4500.pt instead (see README §4).
#
# The overrides below are the verbatim recipe recovered from the run's
# .hydra/overrides.yaml (seed=0 was the config default; made explicit here).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/00_config.sh"

if [ ! -d "$DATASET_ROOT/raw" ]; then
  echo "Missing cleaned data root '$DATASET_ROOT/raw'. Run ./03_clean.sh first." >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" $PYRUN -m graph_signal_diffusion.cli.train \
    seed=0 \
    task="$TASK" \
    model@task.model="$MODEL" \
    dataset.root="$DATASET_ROOT" \
    dataset.past_window=20 \
    dataset.future_window=5 \
    dataset.batch_size=64 \
    dataset.batch_size_val=200 \
    dataset.num_workers=4 \
    dataset.persistent_workers=true \
    dataset.pin_memory=true \
    dataset.n_split_chunks=10 \
    dataset.shuffle_val=false \
    dataset.standardize_target_in_x_for_revin=true \
    diffusion.revin=true \
    diffusion.ddim_eta=0.2 \
    diffusion.revin_blend_weight=0.7 \
    model.config.base_channels=64 \
    model.config.channel_multipliers=[1,1,1] \
    model.config.pooling_config.gamma=[2,2,2] \
    model.config.gnn_config.num_layers=2 \
    model.config.gnn_config.temporal_mixer.dilations=[1,1] \
    model.config.num_bottleneck_layers=2 \
    +model.config.embedding_config.cond.shared_encoder.temporal.mixer.gated=true \
    +model.config.embedding_config.cond.shared_encoder.temporal.mixer.attention.enabled=true \
    +model.config.embedding_config.cond.shared_encoder.temporal.mixer.attention.num_heads=2 \
    +model.config.embedding_config.cond.shared_encoder.temporal.mixer.attention.dropout=0.1 \
    +model.config.embedding_config.cond.shared_encoder.temporal.mixer.attention.max_timesteps=20 \
    model.config.pooling_config.selector_exploration_noise=1.0 \
    model.config.pooling_config.selector_exploration_noise_min=0.0 \
    model.config.pooling_config.selector_exploration_noise_schedule=linear \
    +trainer.selector_exploration_schedule.enabled=true \
    +trainer.selector_exploration_schedule.warmup_ratio=0.02 \
    +trainer.selector_exploration_schedule.anneal_ratio=0.73 \
    trainer.selector_temperature_schedule.warmup_ratio=0.02 \
    trainer.selector_temperature_schedule.anneal_ratio=0.73 \
    trainer.optimizer.learning_rate=1e-4 \
    trainer.max_epochs=5000 \
    'trainer.eval_schedule.phases=[{period: 50, until_epoch: 500}, {period: 100}]' \
    wandb.enabled=true
