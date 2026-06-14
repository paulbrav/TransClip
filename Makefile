.PHONY: help check test lint lint-fix format format-check typecheck compile ruff-check ruff-fix ruff-format ty dev-venv

UV ?= uv
# Dev tooling (ruff/ty/pytest/mypy/coverage) lives in a DEDICATED venv that never
# contains torch. Pinning UV_PROJECT_ENVIRONMENT here guarantees `uv run`/`uv sync`
# target .venv-dev and can NEVER prune the ROCm serving env (.venv). The systemd
# unit uses an absolute .venv/bin/python3 path and is unaffected.
# `:=` (not `?=`) so a stale UV_PROJECT_ENVIRONMENT exported in the shell cannot
# redirect a sync back onto .venv; `make VAR=...` on the command line still wins.
export UV_PROJECT_ENVIRONMENT := .venv-dev
export UV_NO_SYNC ?= 1
DEV_PY := $(UV_PROJECT_ENVIRONMENT)/bin/python

help:
	@printf '%s\n' \
		'Targets:' \
		'  make check         Run lint, format-check, typecheck, tests, compile, and diff checks' \
		'  make dev-venv      Create/refresh .venv-dev with lint+type+test tooling (no torch)' \
		'  make test          Run the unittest suite' \
		'  make lint          Run Ruff lint checks' \
		'  make lint-fix      Run Ruff lint fixes' \
		'  make format        Format Python files with Ruff' \
		'  make format-check  Check Ruff formatting without writing' \
		'  make typecheck     Run ty type checking' \
		'  make compile       Compile transclip, tests, and scripts' \
		'  make ruff-check    Alias for lint' \
		'  make ruff-fix      Alias for lint-fix' \
		'  make ruff-format   Alias for format' \
		'  make ty            Alias for typecheck'

# Auto-bootstrap the dev venv on first use, so `make test`/`check` work without a
# prior `make dev-venv` (UV_NO_SYNC=1 means `uv run` will not populate it itself).
$(DEV_PY):
	$(UV) sync --extra dev --extra audio

dev-venv:
	$(UV) sync --extra dev --extra audio

check: lint format-check typecheck test compile
	git diff --check

test: | $(DEV_PY)
	$(UV) run -m unittest discover -v

lint: | $(DEV_PY)
	$(UV) run ruff check .

lint-fix: | $(DEV_PY)
	$(UV) run ruff check . --fix

format: | $(DEV_PY)
	$(UV) run ruff format .

format-check: | $(DEV_PY)
	$(UV) run ruff format . --check

typecheck: | $(DEV_PY)
	$(UV) run ty check

compile: | $(DEV_PY)
	$(UV) run -m compileall transclip tests scripts

ruff-check: lint
ruff-fix: lint-fix
ruff-format: format
ty: typecheck
