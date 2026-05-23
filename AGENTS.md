# Agent Instructions

This repository is TransClip: a local-only toggle-to-talk dictation app with
local ASR, cleanup, paste injection, daemon lifecycle, tray integration, and an
eval harness. Before changing behavior, read `CONTEXT.md` so you use the
project's domain language consistently. For package boundaries and stable import
paths, read [docs/package-layout.md](docs/package-layout.md).

## Package boundaries

Prefer public package entry points over deep submodule imports:

- `transclip.platform.runtime` — OS/runtime facts
- `transclip.desktop.paste` — clipboard and paste injection
- `transclip.desktop.hotkey` — shortcut install, toggle commands, readiness helpers
- `transclip.desktop.tray` — `run_tray` only
- `transclip.daemon` — service install, status, toggle log
- `transclip.doctor` — readiness checks
- `transclip.paths` — shared path normalization (daemon units, toggle command)

Platform-specific modules (`daemon/linux.py`, `desktop/hotkey/linux_gnome.py`,
`desktop/tray/gtk.py`, etc.) are for tests and adapter work, not general
feature logic. Route cross-cutting calls through the package routers above.

## Development Workflow

Use test-driven development for feature work and bug fixes unless the change is
pure documentation or mechanical formatting.

Work in vertical slices:

1. Write one failing test for one observable behavior.
2. Implement the smallest production change that makes it pass.
3. Run the focused test.
4. Refactor only while tests are green.
5. Repeat for the next behavior.

Do not write a large batch of imagined tests before implementing anything. Tests
written too far ahead tend to encode guessed implementation shape instead of
real product behavior.

## What Good Tests Look Like

Good tests describe what TransClip does from the outside:

- `shell trigger returns command text without submit`
- `record toggle discards under minimum duration`
- `history write failure does not fail transcription`
- `literal escape pastes trigger text without cleanup`

Prefer behavior-oriented tests through public interfaces such as:

- `InferenceEngine.transcribe`, `toggle_recording`, and `health`
- HTTP route behavior through `create_server` and test clients
- CLI-facing functions and rendered command outcomes
- pure domain functions such as routing, cleanup policy, parsing, validation,
  capability detection, and manifest evaluation

Avoid tests that only prove implementation details:

- private helper names or call order inside a module
- exact internal object construction when behavior is unchanged
- patching TransClip modules together until the test only verifies mocks
- asserting that a refactor-preserving rename breaks or passes a test

A useful test should fail when user-visible behavior or a public contract
breaks. It should usually survive a refactor that preserves behavior.

## Mocking Policy

Mock boundaries that are slow, nondeterministic, hardware-dependent, destructive,
or outside this process. Keep TransClip's own domain logic real.

Mock or fake these interfaces:

- ASR and text model backends. Use small fakes such as `FakeASR` and
  `FakeTextBackend`; do not load Hugging Face models in unit tests.
- Microphone recording. Use `FakeRecorder`; do not require real audio devices.
- Desktop and OS facts. Use `FakeRuntime` for platform, environment, executable
  discovery, home/cache paths, and subprocess execution.
- Clipboard and paste injection tools. Stub `wl-copy`, `wl-paste`, `wtype`,
  `ydotool`, `xclip`, `xdotool`, and related runtime failures.
- Native desktop integration. Mock `gsettings`, systemd, launch agents, tray
  libraries, and notification commands.
- Filesystem locations that would touch the user's real config, cache, logs, or
  history. Redirect them to temp directories.
- Time, cooldowns, process errors, network/model cache availability, and
  external command output when those facts are the behavior under test.

Do not mock these by default:

- `route_voice_mode` when testing service transcription behavior.
- Cleanup, shell parsing, shell validation decision logic, settings coercion,
  manifest evaluation, history serialization, and capability selection logic.
- The HTTP dispatch path when the test is about API behavior.
- The CLI/status rendering path when the test is about user-facing output.

When a boundary needs a test double, prefer a small fake with the same narrow
contract over a broad mock with many call assertions. Assert the resulting
behavior and metadata, not the internal choreography.

## Test Commands

Use focused tests while iterating:

```bash
uv run -m unittest tests.test_shell_command -v
uv run -m unittest tests.test_service -v
```

Run the full suite before handing off substantial changes:

```bash
uv run -m unittest discover -v
```

For lint/type-oriented work, use the project tooling in `pyproject.toml` when it
is relevant to the files touched.

## Design Bias

Prefer deep modules: small public interfaces with implementation detail hidden
behind them. Add seams only at real system boundaries or where a fake makes a
behavior test deterministic. If test setup becomes elaborate, first ask whether
the production interface is too shallow or whether the test is aimed at an
implementation detail.

## External Inspiration

This guidance is informed by Matt Pocock's public TDD skill and skills repo:
behavior through public interfaces, one red-green-refactor slice at a time, and
mocking at boundaries rather than internal collaborators.

- https://github.com/mattpocock/skills
- https://raw.githubusercontent.com/mattpocock/skills/main/skills/tdd/SKILL.md
