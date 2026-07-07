# !/bin/bash



outputs/stock_price_forecasting/sp100/ddim/2026-01-30/16-37-13

CUDA_VISIBLE_DEVICES=1 python -m graph_signal_diffusion.cli.test \
    --config-dir outputs/stock_price_forecasting/sp100/ddim/2026-01-30/16-56-09/.hydra \
    --checkpoint outputs/stock_price_forecasting/sp100/ddim/2026-01-30/16-56-09/trainer_chkpts/DDIM_epoch_5000.pt \
    --diffusion-type ddpm \
    --n-samples-per-input 10 \
    --eval-on train-val \
    --single-batch \
    --use-amp



# CUDA_VISIBLE_DEVICES=1 python -m graph_signal_diffusion.cli.test \
#     --config-dir outputs/stock_price_forecasting/sp100/ddim/2026-01-30/16-37-13/.hydra \
#     --checkpoint outputs/stock_price_forecasting/sp100/ddim/2026-01-30/16-37-13/trainer_chkpts/DDIM_epoch_2350.pt \
#     --diffusion-type ddim --sampling-timesteps 500 --ddim-eta 0.0 \
#     --n-samples-per-input 5 \
#     --eval-on val \
#     --single-batch \
#     --use-amp --compile

