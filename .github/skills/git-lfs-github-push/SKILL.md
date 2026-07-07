---
name: git-lfs-github-push
description: 
  Ensure Git LFS is properly set up and used before pushing to GitHub to avoid errors with large files. Use this skill whenever preparing to push updates to GitHub for this repo, especially when changes may include large data or binary files. Ensure Git LFS is installed, configured, and tracking the correct file types before pushing.
---

# Git LFS GitHub Push

## Goal
Prevent push failures and oversized file errors by ensuring Git LFS is installed,
configured, and used correctly before any `git push` to GitHub in this repo.

## When to use this skill
Use this skill any time you are about to push updates to GitHub for this project,
particularly if the changes include data files or other large binaries.

## Source of truth
Read `docs/GIT_LFS_SETUP.md` for the full, authoritative setup and troubleshooting
instructions. Only summarize or adapt those steps as needed for the task.

## Pre-push checklist (MUST)
1. Ensure Git LFS is installed on the machine.
2. Initialize Git LFS in the repo: `git lfs install`.
3. Ensure LFS tracking exists for these patterns in `.gitattributes`:
   `*.npy`, `*.npz`, `*.csv`, `*.pdf`.
4. If `.gitattributes` was updated, commit it before the data files.
5. Verify tracking status: `git lfs track` and `git lfs ls-files`.

## Handling large files already committed (CAUTION)
If large files were committed without LFS, do not rewrite history without
explicit user approval. If approved, follow the reset/re-add steps in
`docs/GIT_LFS_SETUP.md` (soft reset, re-add with LFS, recommit, then push).

## Common push issues
- HTTP 408/timeouts: increase buffer with
  `git config http.postBuffer 524288000`.

## Notes
- GitHub enforces a 100 MB file size limit for non-LFS files.
- Keep processed or temporary data out of Git via `.gitignore` whenever possible.
