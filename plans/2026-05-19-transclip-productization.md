# TransClip Python Productization Plan

Status: clean-cutover implementation plan  
Source: decision to replace the broken Rust implementation on `master` with the Python implementation  
Scope: turn the current Python codebase into the canonical TransClip application

## Context

The Python implementation is now the canonical implementation on `master`. It
has stronger runtime architecture and verification than the old public
TransClip implementation: daemon lifecycle, tray controls, hotkey handling,
paste safety, model switching, ROCm laptop optimization, real-usage evals, and
unit coverage.

The code still exposed old internal naming through the package name,
CLI entrypoint, service name, config/cache directories, desktop shortcut names,
README, generated eval prompts, notifications, and tests. Before this becomes
the public TransClip v2 surface, those names need to be changed deliberately.

## Goals

- Make `transclip` the canonical package, CLI, service, config, cache, log, and
  user-facing product name.
- Preserve the current working Python implementation and eval discipline.
- Keep Granite NAR as the default local ASR model without making the product
  identity Granite-specific.
- Keep ROCm/NAR optimizations permanent and automatic, but describe them as
  backend optimizations rather than app identity.
- Make a clean cutover with no old-name CLI aliases or old-name config/path
  fallback. Existing old-name files are not deleted, but TransClip v2 does not
  read or migrate them automatically.
- Replace or archive the old public repo only after the renamed Python app is
  verified.

## Non-Goals

- Do not rewrite the ASR pipeline or model backends during the rename.
- Do not add cloud fallback or transcript telemetry.
- Do not preserve the Rust implementation.
- Do not attempt a broad UI redesign; keep the tray-first local dictation app.
- Do not make Whisper/Faster-Whisper the default path unless a later eval shows
  it beats the current laptop targets.

## Settled Decisions

- `master` is the canonical Python implementation branch.
- `pure-python-daemon` can remain temporarily as an alias branch, but should not
  be the long-term public branch name.
- The primary product name is `TransClip`.
- The primary Python import package should be `transclip`.
- The primary CLI command should be `transclip`.
- Granite model names should appear in model menus and technical docs, not in
  app/service/config names.
- The app remains local-first and offline-capable.

## Unresolved Questions

- Should the old public `paulbrav/TransClip` repository be force-replaced, or
  should it be archived and the private repo renamed/transferred into its place?
- Clean cutover decision: no old CLI alias and no automatic config migration.
- Should `pure-python-daemon` be deleted after `master` is published?

## Critical Files

- `pyproject.toml`
- `README.md`
- `CONTEXT.md`
- `transclip/`
- `scripts/`
- `tests/`
- `eval/`
- `plans/`

## Execution Order

### Slice 1: Establish Product Constants

Purpose: avoid ad hoc string replacement and make product naming testable.

Steps:

1. Add a small product identity module, for example `transclip/product.py`
   before the package rename, with constants for:
   - display name: `TransClip`
   - app id: `transclip`
   - CLI command: `transclip`
   - systemd service: `transclip.service`
   - launchd label: `com.paulbrav.transclip`
   - config/cache/log directory name: `transclip`
2. Replace repeated literal product names in runtime code with these constants.
3. Update tests to assert product constants instead of hard-coded TransClip
   strings.
4. Keep module imports under the old package in this slice to reduce blast
   radius.

Verification:

```bash
uv run ruff check .
git diff --check --no-color
uv run -m compileall transclip scripts tests
uv run -m unittest discover -s tests -v
```

### Slice 2: Rename User-Facing Runtime Surfaces

Purpose: make installed runtime objects look like TransClip while imports still
work.

Steps:

1. Change systemd and launchd generation to install `transclip.service` and
   `com.paulbrav.transclip`.
2. Change config, cache, and log paths to default to `transclip`.
3. Change GNOME shortcut path/name and notifications to TransClip wording.
4. Change tray app id, title, menu text, and offline labels to TransClip.
5. Update doctor output and fix commands to mention `transclip`.
6. Do not read or migrate old-name paths during the clean cutover.
7. Add tests for new path defaults.

Verification:

```bash
uv run ruff check .
git diff --check --no-color
uv run -m compileall transclip scripts tests
uv run -m unittest discover -s tests -v
```

Manual smoke:

```bash
uv run -m transclip.cli doctor --json
uv run -m transclip.cli install --help
```

### Slice 3: Add the New CLI Entrypoint

Purpose: make `transclip` the command users run before renaming the import
package.

Steps:

1. Update `pyproject.toml`:
   - project name: `transclip`
   - primary script: `transclip = "transclip.cli:main"`
2. Do not keep an old CLI compatibility alias.
3. Change `argparse` prog to `transclip`.
4. Update README, doctor messages, and tests to use `transclip`.
5. Keep the old module command working for internal verification until
   the package rename slice.

Verification:

```bash
uv run ruff check .
git diff --check --no-color
uv run -m compileall transclip scripts tests
uv run -m unittest discover -s tests -v
uv run transclip --help
```

### Slice 4: Rename the Python Package

Purpose: make the import package match the product.

Steps:

1. Move the old package directory to `transclip/`.
2. Rewrite imports in package code, scripts, and tests to `transclip`.
3. Update `pyproject.toml` tool configuration:
   - `src`
   - `tool.ty.src.include`
   - coverage source
   - script entrypoint
4. Update subprocess command generation to
   `-m transclip.cli`.
5. Do not add a compatibility package.
6. Update tests and mocks to patch `transclip.*`.

Verification:

```bash
uv run ruff check .
git diff --check --no-color
uv run -m compileall transclip scripts tests
uv run -m unittest discover -s tests -v
uv run -m transclip.cli doctor --json
```

### Slice 5: Clean Cutover Guardrails

Purpose: make old-name compatibility deliberately absent.

Steps:

1. Do not add an old-name console script.
2. Do not add an old-name compatibility package.
3. Do not read old-name config directories or old JSON config automatically.
4. Keep old files untouched on disk.
5. Test and grep for old public/runtime names.

Verification:

```bash
uv run ruff check .
git diff --check --no-color
uv run -m compileall transclip scripts tests
uv run -m unittest discover -s tests -v
```

### Slice 6: Reframe Model Catalog and Menus

Purpose: keep model choices technical but product-neutral.

Steps:

1. Rename tray menu labels from Granite-specific product language to ASR mode
   language, while keeping concrete model names visible:
   - `Fast local ASR - Granite 4.1 NAR`
   - `Keyword-biased ASR - Granite 4.1`
2. Update model catalog descriptions to emphasize speed, keyword preservation,
   and local laptop behavior.
3. Keep ROCm AOTriton defaults inside the NAR backend.
4. Update README to document current eval numbers as model-specific evidence:
   - NAR optimized latency
   - regular Granite keyword behavior
   - current keyword preservation limitation.
5. Keep real eval thresholds and results unchanged unless a new eval is run.

Verification:

```bash
uv run ruff check .
git diff --check --no-color
uv run -m compileall transclip scripts tests
uv run -m unittest discover -s tests -v
```

Optional real eval:

```bash
VIRTUAL_ENV=$PWD/.venv-gfx1151 uv run --active -m transclip.cli eval eval/real-usage/manifest.json --output eval/real-usage/results.json
```

### Slice 7: Rewrite README and Install Flow

Purpose: make the public repo understandable as TransClip v2.

Steps:

1. Rewrite the README around the product, not the implementation codename:
   - local-first push-to-talk dictation
   - tray and global shortcut workflow
   - laptop model defaults
   - Linux Wayland/X11 and macOS expectations
   - privacy/no cloud fallback
   - eval and hardware notes
2. Replace all old-name commands with `transclip`.
3. Note that v2 is a clean cutover and old config is not migrated
   automatically.
4. Document model modes without implying Granite is the whole product.
5. Update generated prompt sheet headings to TransClip.
6. Make install instructions match current `uv` and service-manager behavior.

Verification:

```bash
rg -n "old product names" README.md transclip scripts tests eval plans
uv run ruff check .
uv run -m unittest discover -s tests -v
```

Expected remaining matches should be only historical plan notes.

### Slice 8: Publish Strategy

Purpose: replace the broken Rust public implementation without losing an escape
hatch.

Steps:

1. Confirm the renamed Python implementation passes:
   - unit suite
   - compileall
   - ruff
   - real eval on the ROCm laptop
   - manual tray/install smoke on the target Linux desktop.
2. Tag the old public TransClip state before replacement, for example
   `pre-v2`. Done: `paulbrav/TransClip` has a `pre-v2` tag.
3. Choose one publication path:
   - Force-replace `paulbrav/TransClip` `master` with the renamed Python tree.
   - Or archive old `paulbrav/TransClip`, rename private `transclip` repo
     to `TransClip`, and make it public.
   Done: pushed the verified Python tree to `paulbrav/TransClip` as `master`
   without deleting the old `main` branch.
4. Set default branch to `master`. Done.
5. Delete or hide stale branches that point to the broken Rust implementation.
   Done for the canonical surface by moving the default branch to `master`;
   old branches remain reachable for rollback/history.
6. Keep a private backup branch or private repo until the public v2 release is
   verified after clone/install. Done: the private source repo still has
   `pure-python-daemon` at the same TransClip v2 commit.

Verification:

```bash
git clone https://github.com/paulbrav/TransClip /tmp/transclip-public-smoke
cd /tmp/transclip-public-smoke
uv sync --extra audio --extra models
uv run transclip --help
uv run -m unittest discover -s tests -v
```

Public smoke result: passed after cloning `https://github.com/paulbrav/TransClip`
with default branch `master`.

## Final Verification Gate

Before calling the productization complete:

```bash
uv run ruff check .
git diff --check --no-color
uv run -m compileall transclip scripts tests
uv run -m unittest discover -s tests -v
VIRTUAL_ENV=$PWD/.venv-gfx1151 uv run --active -m transclip.cli eval eval/real-usage/manifest.json --output eval/real-usage/results.json
VIRTUAL_ENV=$PWD/.venv-gfx1151 uv run --active scripts/check_v1_completion.py
```

Expected status at the time this plan was written:

- Latency should pass with optimized NAR.
- Keyword preservation passed after adding explicit local keyword restoration
  for prompted technical terms. The latest real-usage run measured mean keyword
  preservation at 0.952 against the 0.900 gate.
- A blocked keyword gate should not block the rename itself, but it should be
  documented honestly in the README and release notes.

## Rollback Notes

- Keep the pre-productization commit reachable on `pure-python-daemon` or a tag
  until the public replacement is verified.
- Do not delete old config directories automatically, but do not read them in
  TransClip v2.
- If the package rename causes unexpected breakage, the product constants and
  user-facing runtime rename slices can still ship independently while import
  package compatibility remains in place.
