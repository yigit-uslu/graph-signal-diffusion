#!/usr/bin/env python3
"""
Verify that all required packages are installed and importable.
Run this after installation to ensure environment is properly set up.
"""

import sys
from importlib import import_module
from typing import List, Tuple


# Required packages for the project
REQUIRED_PACKAGES = [
    ('torch', 'PyTorch'),
    ('torch_geometric', 'PyTorch Geometric'),
    ('numpy', 'NumPy'),
    ('pandas', 'Pandas'),
    ('matplotlib', 'Matplotlib'),
    ('seaborn', 'Seaborn'),
    ('networkx', 'NetworkX'),
    ('scipy', 'SciPy'),
    ('sklearn', 'scikit-learn'),
    ('tqdm', 'tqdm'),
    ('hydra', 'Hydra'),
    ('omegaconf', 'OmegaConf'),
    ('yaml', 'PyYAML'),
    ('jaxtyping', 'jaxtyping'),
    ('beartype', 'beartype'),
    ('accelerate', 'Accelerate'),
    ('wandb', 'Weights & Biases'),
    ('torchvision', 'torchvision'),
    ('yfinance', 'yfinance'),
    ('PIL', 'Pillow'),
]

OPTIONAL_PACKAGES = [
    ('torch_scatter', 'torch-scatter'),
    ('pytest', 'pytest'),
]


def check_package(import_name: str, display_name: str) -> Tuple[bool, str]:
    """Check if a package is importable and get its version."""
    try:
        module = import_module(import_name)
        version = getattr(module, '__version__', 'unknown')
        return True, version
    except ImportError as e:
        return False, str(e)


def main():
    """Run verification checks."""
    print("="*70)
    print("DEPENDENCY VERIFICATION")
    print("="*70)
    print()
    
    # Check Python version
    py_version = sys.version_info
    print(f"Python Version: {py_version.major}.{py_version.minor}.{py_version.micro}")
    if py_version.major < 3 or (py_version.major == 3 and py_version.minor < 9):
        print("⚠️  Warning: Python 3.9 or higher is recommended")
    print()
    
    # Check required packages
    print("Required Packages:")
    print("-" * 70)
    missing_packages = []
    
    for import_name, display_name in REQUIRED_PACKAGES:
        success, info = check_package(import_name, display_name)
        if success:
            print(f"✓ {display_name:25s} {info}")
        else:
            print(f"✗ {display_name:25s} NOT FOUND")
            missing_packages.append(display_name)
    
    print()
    
    # Check optional packages
    print("Optional Packages:")
    print("-" * 70)
    
    for import_name, display_name in OPTIONAL_PACKAGES:
        success, info = check_package(import_name, display_name)
        if success:
            print(f"✓ {display_name:25s} {info}")
        else:
            print(f"○ {display_name:25s} not installed (optional)")
    
    print()
    print("="*70)
    
    # Check CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.version.cuda}")
            print(f"  GPU devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  - Device {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("○ CUDA not available (CPU-only mode)")
    except Exception as e:
        print(f"⚠️  Could not check CUDA: {e}")
    
    print("="*70)
    print()
    
    # Summary
    if missing_packages:
        print(f"❌ FAILED: {len(missing_packages)} required package(s) missing:")
        for pkg in missing_packages:
            print(f"   - {pkg}")
        print()
        print("Installation instructions:")
        print("  pip install -r requirements.txt")
        print()
        sys.exit(1)
    else:
        print("✅ SUCCESS: All required packages are installed!")
        print()
        print("You can now run the project:")
        print("  python -m graph_signal_diffusion.cli.train --help")
        print()
        sys.exit(0)


if __name__ == "__main__":
    main()
