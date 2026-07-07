# DEPRECATED: use scripts/wra/diffusion/train_learned_ds2.sh instead.
#!/bin/bash

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

# # Helps reduce CUDA memory fragmentation.
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Train WRA with UGNN learned downsampling (gamma=[1,2,2,2], STE selector).
CUDA_VISIBLE_DEVICES=0 python -m graph_signal_diffusion.cli.train \
    task=wireless_resource_allocation \
    model@task.model=ugnn_wra_learned_ds2
