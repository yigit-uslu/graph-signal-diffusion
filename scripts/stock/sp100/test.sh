#!/bin/bash
# Post-hoc test: loop over saved SP100 checkpoints and run evaluation.
# Edit CHECKPOINT_DIR and CONFIG_DIR to point to your run. Run from project root.

CHECKPOINT_DIR="outputs/stock_price_forecasting/sp100/ddim/2026-01-30/16-37-13/trainer_chkpts"
CONFIG_DIR="outputs/stock_price_forecasting/sp100/ddim/2026-01-30/16-37-13/.hydra"

for checkpoint in $CHECKPOINT_DIR/DDIM_epoch_{4850,5000}.pt; do
    echo "Processing checkpoint: $checkpoint"

    CUDA_VISIBLE_DEVICES=0 python -m graph_signal_diffusion.cli.test \
        --config-dir $CONFIG_DIR \
        --checkpoint $checkpoint \
        --diffusion-type ddim --sampling-timesteps 250 --ddim-eta 0.2 \
        --n-samples-per-input 10 \
        --eval-on train-val,val,test \
        --single-batch \
        --use-amp

    echo "Completed: $checkpoint"
    echo "---"
done

echo "All checkpoints processed!"
