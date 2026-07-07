# DEPRECATED: use scripts/stock/ launchers instead.
# DEPRECATED: use scripts/stock/ launchers instead.
# !/bin/bash


##### Uncomment the following block if you encounter library loading issues (e.g. GLIBCXX, GLIBC, PyTorch) on Linux #####
# # Ensure conda environment is activated
# if [ -z "$CONDA_PREFIX" ]; then
#     echo "Error: Conda environment not activated!"
#     echo "Please run: conda activate graph-signal-diffusion"
#     exit 1
# fi
#
# # Fix library loading by prioritizing conda's libraries
# # This forces system to use conda's newer GLIBCXX, GLIBC, and PyTorch libraries
# # instead of incompatible system versions
# export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib/python3.11/site-packages/torch/lib:${CONDA_PREFIX}/lib/python3.11/site-packages/torch_geometric/lib:${LD_LIBRARY_PATH}"
#
# # Suppress harmless PyG extension warnings (PyG will use CPU fallbacks)
# export PYTHONWARNINGS="ignore::UserWarning"
#
# echo "Using libraries from: ${CONDA_PREFIX}/lib"
# echo ""
##### Uncomment the preceding block if you encounter library loading issues (e.g. GLIBCXX, GLIBC, PyTorch) on Linux #####


# # Uncomment to clean SP500 data before training
# python ./scripts/clean_sp500_data.py \
#     --method drop_incomplete \
#     --min-coverage 0.95 \
#     --correlation-threshold 0.5 \
#     --sector-bonus 0.0



# batch_size=32 
# # Train stock price forecasting v2 model on cleaned SP500 dataset and UGNN config with learned downsampling (gamma=[1,2,2,2], STE selector).
# CUDA_VISIBLE_DEVICES=1 python -m graph_signal_diffusion.cli.train \
#     task=stock_price_forecasting_v2 \
#     dataset@task.dataset=sp500_cleaned \
#     model@task.model=ugnn_sp500_v2_learned_ds2 \
#     dataset.edge_weight_norm=none \
#     dataset.spectral_normalize_edge_weights=True \
#     dataset.past_window=20 \
#     dataset.future_window=5 \
#     dataset.batch_size=${batch_size} \
#     dataset.root=./data/sp500/cleaned_drop_incomplete_min_coverage_0.95_corr_0.5 \
#     trainer.eval_every_n_epochs=50 \
#     trainer.n_samples_per_input=10 \
#     trainer.use_amp=False \
#     trainer.save_checkpoint_every_n_epochs=250 \
#     trainer.max_epochs=10000 \
