# Codex Guidelines

This repository uses **ruff**, **uv**, and **mypy** extensively for code quality and package management.

## Setup

Use `uv` to manage the virtual environment and dependencies:

```bash
uv pip install -e ".[dev]"
```

## Checks

Run the type checker and linter before committing:

```bash
mypy --strict transclip/
ruff check transclip/ tests/
```

## Tests

Execute the unit tests with the built-in `unittest` framework:

```bash
python -m unittest discover -s tests
```

