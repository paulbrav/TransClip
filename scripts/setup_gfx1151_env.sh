#!/usr/bin/env bash
set -euo pipefail

# Rebuild the gfx1151 (AMD ROCm) SERVING virtual environment for transclip.
#
# This env (.venv) is what the systemd unit runs. It is intentionally SEPARATE
# from the dev-tooling env (.venv-dev, created by `make dev-venv`). Never run a
# bare `uv run`/`uv sync` against .venv: torch lives only behind the ROCm index,
# so an unpinned sync prunes the entire ML stack (the failure this guards against).
#
# Idempotent: safe to re-run -- `uv venv --clear` recreates .venv cleanly, so no
# manual `rm -rf .venv` is needed.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
cd "$repo_root"

venv_dir="${VENV_DIR:-.venv}"
python_version="${PYTHON_VERSION:-3.13}"
rocm_index="${ROCM_INDEX:-https://rocm.nightlies.amd.com/v2/gfx1151/}"

# 1. Fresh venv. --clear makes re-runs safe (no "already exists" error).
uv venv --clear --python "$python_version" "$venv_dir"

# 2. Pinned ML stack. --extra-index-url (NOT --index-url) keeps PyPI available
#    for the non-ROCm packages; --pre allows the ROCm prerelease wheels.
uv pip install --python "$venv_dir/bin/python" \
  --extra-index-url "$rocm_index" --pre \
  -r requirements-gfx1151.txt

# 3. Editable project install (pyproject dependencies=[] -> pulls nothing extra).
uv pip install --python "$venv_dir/bin/python" -e .

# 4. flash-attn has no ROCm wheel -> compile locally against the AMD Triton path.
#    Must stay --no-deps --no-build-isolation with the env flag set.
FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE MAX_JOBS="${MAX_JOBS:-4}" \
  uv pip install --python "$venv_dir/bin/python" --no-deps \
  flash-attn==2.8.3 --no-build-isolation

# 5. Smoke test (non-fatal: a slow/offline HF cache should warn, not fail).
if ! "$venv_dir/bin/python" -m transclip.cli models list; then
  echo "WARNING: 'transclip.cli models list' failed (offline HF cache?); env is installed." >&2
fi
