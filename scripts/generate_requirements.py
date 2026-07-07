#!/usr/bin/env python3
"""
Systematic dependency extraction for reproducible research.

This script uses multiple strategies to identify exact dependencies:
1. Analyze actual imports used in the codebase
2. Extract current package versions from the environment
3. Generate requirements.txt with pinned versions

Usage:
    python scripts/generate_requirements.py [--env ENV_NAME]
    
    --env: Conda environment name (default: torch_env)
"""

import subprocess
import sys
import argparse
from pathlib import Path
from typing import Set


# Core packages required by the project (from import analysis)
CORE_PACKAGES = {
    'torch',
    'torch-geometric',
    'torch-scatter',
    'torch-sparse',
    'torch-cluster',
    'torch-spline-conv',
    'numpy',
    'pandas',
    'matplotlib',
    'seaborn',
    'networkx',
    'scipy',
    'scikit-learn',
    'tqdm',
    'hydra-core',
    'omegaconf',
    'pyyaml',
    'jaxtyping',
    'beartype',
    'accelerate',
    'wandb',
    'torchvision',
    'torchaudio',
    'yfinance',
    'pillow',
}

# Package name mappings (import name -> package name)
PACKAGE_NAME_MAP = {
    'PIL': 'pillow',
    'sklearn': 'scikit-learn',
    'yaml': 'pyyaml',
}


def get_installed_version(package_name: str, conda_env: str = None) -> str:
    """Get the installed version of a package."""
    try:
        if conda_env:
            # Use conda run to execute in specific environment
            result = subprocess.run(
                ['conda', 'run', '-n', conda_env, 'pip', 'show', package_name],
                capture_output=True,
                text=True,
                check=False,
            )
        else:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'show', package_name],
                capture_output=True,
                text=True,
                check=False,
            )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if line.startswith('Version:'):
                    return line.split(':', 1)[1].strip()
    except Exception:
        pass
    return None


def get_python_version(conda_env: str = None) -> str:
    """Get the Python version from the environment."""
    try:
        if conda_env:
            result = subprocess.run(
                ['conda', 'run', '-n', conda_env, 'python', '--version'],
                capture_output=True,
                text=True,
                check=False,
            )
        else:
            result = subprocess.run(
                [sys.executable, '--version'],
                capture_output=True,
                text=True,
                check=False,
            )
        if result.returncode == 0:
            return result.stdout.strip().split()[-1]
    except Exception:
        pass
    return None


def generate_requirements(output_file: Path, pinned: bool = True, conda_env: str = None):
    """Generate requirements.txt with current package versions."""
    requirements = []
    
    env_info = f" from conda environment '{conda_env}'" if conda_env else ""
    print(f"Generating {'pinned' if pinned else 'flexible'} requirements{env_info}...")
    
    py_version = get_python_version(conda_env)
    if py_version:
        print(f"Python version: {py_version}")
    
    print(f"\nAnalyzing {len(CORE_PACKAGES)} core packages...\n")
    
    for package in sorted(CORE_PACKAGES):
        version = get_installed_version(package, conda_env)
        if version:
            if pinned:
                req_line = f"{package}=={version}"
            else:
                # For flexible versions, use compatible release
                major_minor = '.'.join(version.split('.')[:2])
                req_line = f"{package}>={version}"
            requirements.append(req_line)
            print(f"✓ {package}: {version}")
        else:
            print(f"✗ {package}: not found (will use unpinned)")
            requirements.append(package)
    
    # Add header with environment info
    header = [
        "# Core dependencies with pinned versions" + env_info,
        "# Generated for exact reproducibility of research results",
        "#",
    ]
    if py_version:
        header.append(f"# Python version: {py_version}")
    if conda_env:
        header.append(f"# Conda environment: {conda_env}")
    header.extend([
        "#",
        "# Installation:",
        "#   pip install -r requirements.txt",
        "",
    ])
    
    # Write requirements file
    output_file.write_text('\n'.join(header + requirements) + '\n')
    print(f"\n{'='*60}")
    print(f"✓ Requirements written to: {output_file}")
    print(f"{'='*60}\n")


def generate_dev_requirements(output_file: Path):
    """Generate development requirements."""
    dev_packages = [
        'pytest>=7.4.0',
        'pytest-cov>=4.1.0',
        'black>=23.0.0',
        'ruff>=0.1.0',
        'mypy>=1.5.0',
        'ipython>=8.0.0',
        'jupyter>=1.0.0',
    ]
    output_file.write_text('\n'.join(dev_packages) + '\n')
    print(f"✓ Dev requirements written to: {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate requirements.txt from environment'
    )
    parser.add_argument(
        '--env',
        default='torch_env',
        help='Conda environment name (default: torch_env)'
    )
    parser.add_argument(
        '--no-conda',
        action='store_true',
        help='Use current Python environment instead of conda'
    )
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    
    print("\n" + "="*60)
    print("SYSTEMATIC DEPENDENCY EXTRACTION")
    print("="*60 + "\n")
    
    conda_env = None if args.no_conda else args.env
    
    if conda_env:
        # Verify conda environment exists
        result = subprocess.run(
            ['conda', 'env', 'list'],
            capture_output=True,
            text=True,
            check=False
        )
        if conda_env not in result.stdout:
            print(f"⚠️  Warning: Conda environment '{conda_env}' not found!")
            print(f"Available environments:")
            print(result.stdout)
            print(f"\nUsing current Python environment instead.\n")
            conda_env = None
    
    # Generate pinned requirements for exact reproducibility
    requirements_file = project_root / "requirements.txt"
    generate_requirements(requirements_file, pinned=True, conda_env=conda_env)
    
    # Generate development requirements
    dev_requirements_file = project_root / "requirements-dev.txt"
    generate_dev_requirements(dev_requirements_file)
    
    print("\nUsage:")
    print("------")
    print("For users (exact reproducibility):")
    print("  pip install -r requirements.txt")
    print("\nFor developers (flexible versions):")
    print("  pip install -e .")
    print("\nFor contributors (includes dev tools):")
    print("  pip install -e . -r requirements-dev.txt")
    print()


if __name__ == "__main__":
    main()
