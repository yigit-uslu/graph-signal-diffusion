# Makefile for Graph Signal Generative Diffusion Modeling
# Provides convenient shortcuts for common development tasks

.PHONY: help install install-dev install-exact verify clean test lint format docs

# Default target
help:
	@echo "Available commands:"
	@echo "  make install        - Install package with flexible dependencies (development)"
	@echo "  make install-exact  - Install with pinned versions (reproducibility)"
	@echo "  make install-dev    - Install with development tools"
	@echo "  make verify         - Verify installation"
	@echo "  make test           - Run tests"
	@echo "  make lint           - Run linters (ruff)"
	@echo "  make format         - Format code (black)"
	@echo "  make clean          - Remove build artifacts"
	@echo "  make requirements   - Generate requirements.txt from current environment"

# Installation targets
install:
	pip install -e .

install-exact:
	pip install -r requirements.txt
	pip install -e .

install-dev:
	pip install -e .
	pip install -r requirements-dev.txt

verify:
	python scripts/verify_install.py

# Development targets
test:
	pytest tests/ -v --cov=graph_signal_diffusion --cov-report=term-missing

lint:
	ruff check src/ tests/ scripts/

format:
	black src/ tests/ scripts/
	ruff check --fix src/ tests/ scripts/

# Utility targets
requirements:
	python scripts/generate_requirements.py

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

# Documentation
docs:
	@echo "Documentation files:"
	@echo "  - README.md: Main project documentation"
	@echo "  - INSTALL.md: Installation and dependency management guide"
	@echo "  - docs/: Additional documentation and guides"
