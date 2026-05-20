.PHONY: help check test lint lint-fix format format-check typecheck compile ruff-check ruff-fix ruff-format ty

UV ?= uv

help:
	@printf '%s\n' \
		'Targets:' \
		'  make check         Run lint, format-check, typecheck, tests, compile, and diff checks' \
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

check: lint format-check typecheck test compile
	git diff --check

test:
	$(UV) run -m unittest discover -v

lint:
	$(UV) run ruff check .

lint-fix:
	$(UV) run ruff check . --fix

format:
	$(UV) run ruff format .

format-check:
	$(UV) run ruff format . --check

typecheck:
	$(UV) run ty check

compile:
	$(UV) run -m compileall transclip tests scripts

ruff-check: lint
ruff-fix: lint-fix
ruff-format: format
ty: typecheck
