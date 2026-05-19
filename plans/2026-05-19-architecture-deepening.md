# Architecture Deepening Plan

Status: draft implementation plan  
Source: 2026-05-19 architecture review with explorer passes  
Scope: deepen the major shallow Modules found in the dictation, platform runtime, ASR, cleanup, model, and eval flows

## Context

TransClip is a local-only toggle-to-talk dictation app. The main runtime path is:

```text
shortcut -> toggle record -> Python HTTP transport -> dictation runtime -> clipboard -> paste
```

The architecture review found recurring friction: important domain behavior is spread across many shallow Modules, while several seams expose route strings, subprocess details, platform probes, or raw dictionaries. The goal of this plan is to turn those shallow Modules into deeper Modules with smaller Interfaces, stronger Locality, and better test surfaces.

There is no project-local `LANGUAGE.md`, `CONTEXT.md`, or ADR directory at the time of writing. Use the architecture vocabulary from the review request while executing this plan:

- Module
- Interface
- Implementation
- Depth
- Seam
- Adapter
- Leverage
- Locality

## Goals

- Increase Depth by moving behavior behind smaller Interfaces.
- Improve Locality so recording, paste, platform, ASR, cleanup, model, and eval behavior each has one obvious home.
- Make the Interface the test surface for each deepened Module.
- Keep existing CLI commands, tray behavior, HTTP routes, settings fields, eval output shape, and current tests passing unless a slice explicitly updates the caller contract.
- Preserve local-only/offline model behavior.

## Non-Goals

- Do not change the product behavior of dictation, paste, cleanup, model loading, or eval thresholds as part of architecture-only slices.
- Do not reintroduce runtime glossary or keyword correction.
- Do not rename public CLI commands or settings fields.
- Do not combine unrelated refactors into one reviewable change.
- Do not add compatibility shims unless a current caller or test requires them.

## Settled Decisions

- Use incremental slices. Each slice must leave the repo passing tests.
- Prefer moving behavior into deeper Modules before deleting old helper functions.
- Treat scripts as command Adapters where possible; shared behavior should move into package Modules.
- Keep file-level churn small enough that each slice can be reviewed independently.
- When a new domain term is introduced, add or update `CONTEXT.md` before relying on the term in code or plans.

## Unresolved Questions

- Should `CONTEXT.md` become a required project artifact for future architecture reviews?
- Should route names and response payload names remain stringly typed for backward compatibility, or should typed request/result values be introduced internally while preserving the HTTP JSON shape?
- Should the model Module own all model/backend compatibility, or only catalog/cache compatibility?
- Should eval keyword preservation remain a metric after runtime glossary removal, or should it be renamed as technical-term preservation?

## Critical Files

- `transclip/service.py`
- `transclip/service_routes.py`
- `transclip/client.py`
- `transclip/recording_ops.py`
- `transclip/cli_commands.py`
- `transclip/tray.py`
- `transclip/paste.py`
- `transclip/platform_capabilities.py`
- `transclip/daemon.py`
- `transclip/daemon_lifecycle.py`
- `transclip/doctor.py`
- `transclip/gnome_shortcut.py`
- `transclip/asr.py`
- `transclip/cleanup.py`
- `transclip/models.py`
- `transclip/eval_harness.py`
- `scripts/check_eval_results.py`
- `scripts/prepare_real_eval.py`
- `scripts/run_real_eval_pipeline.py`
- `tests/`

## Execution Order

### Slice 0: Establish Domain Vocabulary

Purpose: make future plans and code easier for agents to navigate.

Steps:

1. Create `CONTEXT.md` if it still does not exist.
2. Define the project domain terms used by the refactor, at minimum:
   - Dictation session
   - Interactive dictation
   - Platform runtime
   - Paste capability
   - Shortcut readiness
   - ASR runtime
   - Cleanup policy
   - Eval gate
3. Keep definitions descriptive and current-state only.
4. Do not record speculative module names as facts until a slice implements them.

Verification:

```bash
uv run ruff check .
uv run -m unittest discover -s tests -v
```

### Slice 1: Deepen Platform Runtime

Purpose: give platform facts one home before higher-level Modules depend on them.

Current friction:

- `platform.system()`, `os.environ`, `Path.home()`, `shutil.which`, and subprocess execution appear across `settings.py`, `platform_capabilities.py`, `daemon_lifecycle.py`, `doctor.py`, and `notify.py`.
- The `platform_capabilities.py` Module has some Depth, but its Seam leaks because callers rebuild environment snapshots and OS branches.

Target shape:

- A deeper platform runtime Module owns platform facts, user paths, executable discovery, and command execution policy.
- OS-specific behavior sits behind Adapters.
- Higher-level Modules ask for facts and command outcomes instead of probing directly.

Implementation steps:

1. Inventory all direct uses of `platform.system`, `os.environ`, `Path.home`, `shutil.which`, and subprocess command execution in `transclip/`.
2. Move low-level platform fact gathering behind one Interface.
3. Update `platform_capabilities.py` to consume that Interface instead of taking ad hoc `which` and `environ` values everywhere.
4. Update `daemon_lifecycle.py`, `doctor.py`, and `notify.py` to use the platform runtime Seam.
5. Keep existing tests by providing a fake Adapter in tests instead of patching global functions where practical.

Tests:

```bash
uv run -m unittest tests.test_doctor tests.test_daemon tests.test_paste -v
uv run -m unittest discover -s tests -v
```

Review notes:

- This slice should not change install paths, daemon commands, notification behavior, or paste backend ordering.
- If the fake Adapter becomes as complex as the Implementation, stop and reduce the Interface.

### Slice 2: Deepen Paste And Clipboard

Purpose: make paste behavior, capability, execution, restore, and diagnostics local.

Current friction:

- `paste.py` owns execution and restore behavior.
- `platform_capabilities.py` owns backend selection and readiness.
- `doctor.py` and `daemon.py` consume readiness separately.
- `_LAST_PASTE_ERRORS` is a leaky Implementation detail.

Target shape:

- A paste and clipboard Module owns the paste capability, execution order, diagnostics, and clipboard restore policy.
- Platform runtime supplies only lower-level facts.
- CLI, doctor, daemon status, and smoke checks consume the same Interface.

Implementation steps:

1. Define the paste result and paste capability vocabulary in one Module.
2. Move command selection and readiness probing next to paste execution.
3. Replace `_LAST_PASTE_ERRORS` with per-attempt diagnostics carried in the result.
4. Update `doctor.py`, `daemon.py`, and tests to consume the deeper Interface.
5. Preserve current backend order:
   - macOS: `osascript`
   - Wayland: `wtype`, then `ydotool`
   - X11/fallback: `xdotool`, then `ydotool`, then `wtype`

Tests:

```bash
uv run -m unittest tests.test_paste tests.test_doctor tests.test_daemon -v
uv run -m unittest discover -s tests -v
```

Review notes:

- Do not cache executable discovery unless the Interface includes an explicit refresh or invalidation story.
- Paste failure must still leave the transcript on the clipboard when copy succeeds.

### Slice 3: Deepen Shortcut Readiness

Purpose: localize GNOME shortcut policy and readiness behavior.

Current friction:

- `gnome_shortcut.py` combines command construction, gsettings parsing, installation, status, and command existence checks.
- Callers still decide when GNOME matters and how readiness affects install, doctor, and tray flows.

Target shape:

- Shortcut readiness is one Module with a small Interface for install, status, validation, and expected binding.
- GNOME-specific gsettings behavior is hidden behind an Adapter.
- Tray and doctor do not duplicate shortcut policy.

Implementation steps:

1. Identify all shortcut callers in `cli_commands.py`, `daemon_lifecycle.py`, `doctor.py`, and `tray.py`.
2. Move readiness decisions into the shortcut Module.
3. Keep command construction local to shortcut handling.
4. Update tray hotkey save/install flow to call the deeper Interface.
5. Keep existing GNOME shortcut names, paths, and binding behavior.

Tests:

```bash
uv run -m unittest tests.test_gnome_shortcut tests.test_doctor tests.test_tray tests.test_cli tests.test_daemon -v
uv run -m unittest discover -s tests -v
```

Review notes:

- Keep `DEFAULT_HOTKEY_LINUX` as the single default binding source.
- Preserve rollback behavior when tray hotkey installation fails.

### Slice 4: Deepen Dictation Session

Purpose: separate recording lifecycle from audio-to-text transcription.

Current friction:

- `InferenceEngine` owns recording state, recorder construction, cooldown, minimum duration discard, temporary WAV handling, transcription, cleanup, debug capture, and history writes.
- Understanding toggle behavior requires reading `start_recording`, `stop_recording`, `toggle_recording`, and `transcribe` together.

Target shape:

- A dictation session Module owns lifecycle, timing, cooldown, discard, and stop-to-WAV behavior.
- The transcription Module owns ASR, cleanup, debug capture, and result construction.
- `InferenceEngine` becomes orchestration over deeper Modules rather than the main Implementation owner.

Implementation steps:

1. Capture the existing result shapes for start, stop, discard, ignored toggle, and stopped toggle in tests.
2. Move recorder state and lifecycle rules behind a dictation session Interface.
3. Make the dictation session call a transcription dependency when a stop should produce text.
4. Keep `record_history` behavior unchanged while moving history write placement only if tests make the ordering clear.
5. Update `service_routes.py` to call through the same public methods it does today, backed by the deeper Module.

Tests:

```bash
uv run -m unittest tests.test_service tests.test_cli tests.test_tray -v
uv run -m unittest discover -s tests -v
```

Review notes:

- Preserve cooldown and minimum duration behavior exactly.
- Avoid tying the dictation session Interface to HTTP route names.

### Slice 5: Deepen Interactive Dictation

Purpose: make hotkey/tray/CLI behavior one local policy instead of caller-side interpretation.

Current friction:

- `recording_ops.py` handles client toggle, paste, logging, and basic errors.
- `cli_commands.py` and `tray.py` still inspect raw payload fields and decide user-visible behavior.
- Toggle logs and transcript history are written through different paths.

Target shape:

- Interactive dictation owns the whole “toggle, maybe paste, log, and produce renderable outcome” policy.
- CLI and tray become Adapters that render an outcome.
- Observation behavior has one clear policy for toggle logs and transcript history.

Implementation steps:

1. Document current interactive outcomes from tests:
   - service unavailable
   - HTTP rejection
   - started
   - ignored cooldown
   - discarded short recording
   - stopped with paste success
   - stopped with paste failure
2. Deepen `recording_ops.py` or replace it with a deeper interactive dictation Module.
3. Move CLI/tray payload interpretation into that Module.
4. Decide whether toggle log and transcript history stay separate or become two Adapters behind one observation policy.
5. Keep JSON output from `toggle-record` compatible unless intentionally changed and tested.

Tests:

```bash
uv run -m unittest tests.test_cli tests.test_tray tests.test_daemon tests.test_history -v
uv run -m unittest discover -s tests -v
```

Review notes:

- Do not move presentation strings into low-level Modules unless they are part of the user-facing outcome Interface.
- Preserve the “Paste failed. The transcript is still on the clipboard.” behavior.

### Slice 6: Deepen ASR Runtime

Purpose: make audio normalization and runtime policy local to ASR instead of repeated per backend.

Current friction:

- `GraniteSpeechTransformersBackend` and `GraniteSpeechNarTransformersBackend` repeat audio loading, channel folding, sample-rate conversion, device choice, cache policy, and timing.
- Adding another ASR Adapter would likely copy the same mechanics.

Target shape:

- ASR owns shared audio preparation and runtime policy.
- Provider-specific Adapters own only model loading and invocation.
- Tests can exercise audio preparation without importing heavy model runtimes.

Implementation steps:

1. Extract shared audio preparation behind an ASR-local Interface.
2. Keep sample rate target at 16000 and mono folding behavior unchanged.
3. Keep ROCm-specific Granite NAR dtype behavior unchanged.
4. Keep `FileTranscriptASRBackend` simple and isolated from model runtime policy.
5. Add focused tests around audio preparation and backend selection.

Tests:

```bash
uv run -m unittest tests.test_asr tests.test_audio_debug -v
uv run -m unittest discover -s tests -v
```

Review notes:

- Avoid importing `torch`, `torchaudio`, or `soundfile` at module import time.
- Preserve offline model loading defaults.

### Slice 7: Deepen Cleanup Policy

Purpose: make faithful cleanup semantics independent of provider Implementation.

Current friction:

- `cleanup.py` mixes faithful cleanup policy, prompt construction, token budgeting, model loading, response parsing, and validation inside provider Implementations.
- Removing a provider would remove policy that should survive.

Target shape:

- Cleanup policy owns prompt intent, token budget rules, output validation, and drift-sensitive invariants.
- llama.cpp and transformers are thinner Adapters.
- Rule cleanup remains the default low-latency Adapter.

Implementation steps:

1. Identify cleanup policy that should apply to every non-rule model Adapter.
2. Move token budgeting and output validation into shared cleanup policy.
3. Keep `faithful_cleanup_messages` behavior available for transformer cleanup.
4. Keep llama.cpp prompt text semantically equivalent.
5. Add tests that exercise policy without loading model runtimes.

Tests:

```bash
uv run -m unittest tests.test_cleanup -v
uv run -m unittest discover -s tests -v
```

Review notes:

- Do not reintroduce glossary substitution.
- Preserve conservative punctuation/capitalization behavior for rule cleanup.

### Slice 8: Deepen Model Capability

Purpose: make model catalog, cache presence, backend compatibility, and runtime compatibility one coherent Module.

Current friction:

- `models.py` is mostly catalog plus cache helpers.
- `asr.py`, `doctor.py`, and `cli_commands.py` re-decide backend compatibility and model runtime expectations.
- Deleting `models.py` would mostly move constants and prefetch branches into callers.

Target shape:

- Model capability owns model identity, backend compatibility, cache paths, and prefetch requirements.
- ASR loading and actual download calls remain Adapters.
- Doctor, model list, model prefetch, and ASR selection use one model capability Interface.

Implementation steps:

1. Map current model IDs, backend markers, cache markers, and expected local paths.
2. Move backend compatibility checks out of `asr.py` where they are model-catalog facts.
3. Update `doctor.py` model checks to consume model capability.
4. Update `cli_commands.py models list/doctor/prefetch` to consume model capability.
5. Keep actual Hugging Face calls isolated and mockable.

Tests:

```bash
uv run -m unittest tests.test_models tests.test_asr tests.test_doctor tests.test_cli -v
uv run -m unittest discover -s tests -v
```

Review notes:

- Preserve `models_local_files_only = true`.
- Do not make dictation download models implicitly.

### Slice 9: Deepen Eval Module

Purpose: localize eval case/result meaning, metric policy, and gate policy.

Current friction:

- `eval_harness.py` owns runtime metrics.
- `scripts/check_eval_results.py` owns pass thresholds.
- `scripts/prepare_real_eval.py` owns manifest construction.
- `scripts/run_real_eval_pipeline.py` knows too much about result JSON and subprocess flow.

Target shape:

- Eval owns manifest shape, warmup rules, metric calculation, result summary, and gate policy.
- Scripts are command Adapters.
- Tests can exercise gate policy without shelling through the full pipeline.

Implementation steps:

1. Move manifest construction concepts from `prepare_real_eval.py` into package eval code.
2. Move pass/fail threshold policy from `check_eval_results.py` into package eval code.
3. Keep `run_real_eval_pipeline.py` focused on orchestration and command-line output.
4. Decide whether `keyword_preservation` should remain named as-is or become technical-term preservation.
5. Keep eval JSON output compatible unless intentionally migrated in one tested slice.

Tests:

```bash
uv run -m unittest tests.test_eval_harness tests.test_check_eval_results tests.test_prepare_real_eval tests.test_run_real_eval_pipeline -v
uv run -m unittest discover -s tests -v
VIRTUAL_ENV=$PWD/.venv-gfx1151 uv run --active scripts/check_v1_completion.py
```

Review notes:

- If renaming keyword preservation, preserve old JSON fields for one compatibility slice or update every consumer in the same slice.
- Keep warmup cases excluded from measured metrics.

## Final Verification

Run after every slice:

```bash
uv run ruff check .
git diff --check --no-color
uv run -m unittest discover -s tests -v
uv run -m compileall transclip scripts tests
```

Run after slices that touch model, ASR, eval, or runtime completion:

```bash
VIRTUAL_ENV=$PWD/.venv-gfx1151 uv run --active scripts/check_v1_completion.py
```

Optional live diagnostics after platform, paste, shortcut, or daemon slices:

```bash
uv run -m transclip.cli doctor
uv run -m transclip.cli status
uv run -m transclip.cli smoke-test
```

Record live diagnostic output under `build/artifacts/` if it is used as evidence for a review.

## Review Strategy

- Prefer one pull request per slice after Slice 0.
- In each slice, show before/after tests that prove the new Interface is the test surface.
- Avoid broad renames until behavior is already behind the new Seam.
- Keep old helper functions temporarily only when they reduce review risk; delete them before considering the slice complete.

## Rollback Notes

- Each slice should be revertible independently.
- If a new deepened Module starts increasing caller knowledge instead of reducing it, stop and re-run the deletion test.
- If live desktop behavior regresses, first revert the platform, paste, shortcut, or interactive dictation slice that touched the failing path.
