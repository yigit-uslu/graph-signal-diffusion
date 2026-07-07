#!/bin/bash
# Fix PyTorch Geometric library loading issues
# This script reinstalls PyG packages from conda-forge for better compatibility

set -e

echo "Uninstalling current PyG packages..."
pip uninstall -y pyg-lib torch-scatter torch-cluster torch-sparse torch-spline-conv

echo ""
echo "Installing PyG packages from conda-forge..."
echo "This may take a few minutes..."

# Install from conda-forge which has better binary compatibility
conda install -y -c conda-forge -c pytorch \
    pyg \
    pytorch-scatter \
    pytorch-sparse \
    pytorch-cluster \
    pytorch-spline-conv

echo ""
echo "Installation complete!"
echo "Try running your training script again."
