---
name: torch-env-python
description: 
  Use this skill whenever you need to run Python code in this repo (especially unit tests). Always run Python commands inside the Conda environment named "torch_env" using either `conda run -n torch_env ...` or `conda activate torch_env` immediately followed by the Python/test command(s). Never run repo Python scripts/tests with system Python.
license: MIT
---

# Torch Env Python Execution

## Goal
Ensure **all Python script execution and all test runs** happen inside the Conda environment **torch_env**.

## When to use this skill
Use this skill whenever you are about to run any of the following:
- `python ...` (any script, module execution, REPL, one-liners)
- unit tests / test suites (e.g., `pytest`, `unittest`, `python -m pytest`, `python -m unittest`)
- repo scripts such as `./scripts/test.py`, `./scripts/*.py`, `./tools/*.py`, etc.

## Hard requirements (MUST)
1. **MUST run Python in `torch_env`** for any script/test execution.
2. **MUST prefer** the single-command form when possible:
   - `conda run -n torch_env <command>`
3. If you choose activation (interactive/multi-step workflows), you **MUST** run:
   - `conda activate torch_env`
   - then the Python/test command(s) **immediately after**, in the same terminal session.
4. **MUST NOT** run `python`, `pytest`, or any test command outside `torch_env`.

## Preferred command patterns

### Single command (preferred)
Use this when you can express the action as one command:

- Run a script:
  - `conda run -n torch_env python ./scripts/test.py`

- Run pytest:
  - `conda run -n torch_env pytest`
  - or `conda run -n torch_env python -m pytest`

- Run unittest:
  - `conda run -n torch_env python -m unittest`
  - or `conda run -n torch_env python -m unittest discover -s tests`

### Two-step (acceptable for multi-command sessions)
Use this when you’ll run multiple commands and want the environment to stay active:

1) `conda activate torch_env`  
2) `python ./scripts/test.py`  (or `pytest`, etc.)

## Pre-flight check (only if needed)
If there’s any indication `torch_env` might not exist or conda isn’t available, verify first:
- `conda info --envs`

If `torch_env` is missing, **stop** and ask the user how to provision it (do not guess).

## Anti-patterns (DO NOT)
- `python ./scripts/test.py`  (without `conda run` or prior `conda activate torch_env`)
- `pytest` (without being in `torch_env`)
- Using a global/system interpreter just because it is selected in VS Code.

## Notes for VS Code terminals
- `conda run -n torch_env ...` is the most robust because it doesn’t depend on shell initialization state.
- If activation fails due to shell init, fall back to `conda run -n torch_env ...`.
