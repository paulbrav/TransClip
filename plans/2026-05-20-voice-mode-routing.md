# Voice Mode Routing Plan

Status: decision-complete implementation plan
Branch: `feature/voice-mode-routing`
Source: design decisions through 2026-05-20
Scope: add spoken-prefix routing for normal dictation, model cleanup, and shell-command generation in TransClip

## Context

Current TransClip runtime:

```text
shortcut -> toggle record -> Python HTTP service -> Granite ASR -> keyword restore -> cleanup -> clipboard -> paste
```

The feature adds a post-ASR routing layer at the `InferenceEngine.transcribe()`
choke point. Normal dictation remains unchanged unless a case-insensitive
leading trigger phrase is present or the tray toggle enables model cleanup for
all dictation.

The chosen model path is **PyTorch/Transformers**, not llama.cpp/GGUF. Use the
existing service-hosted model pattern and load the text model lazily in the
Python service.

## Settled Product Behavior

- Model: use `Qwen/Qwen3.5-4B` as the small dense text model for both explicit
  cleanup and shell command generation.
- Runtime: PyTorch/Transformers inside the existing TransClip service process.
- Trigger matching: case-insensitive, prefix-only, deterministic.
- Cleanup triggers:
  - `clean up <text>`
  - `trans cleanup <text>`
- Shell triggers:
  - `shell command <task>`
  - `bash command <task>`
  - `terminal command <task>`
- Literal escape:
  - `literal clean up <text>` pastes `clean up <text>`
  - `literal shell command <text>` pastes `shell command <text>`
  - same rule for all configured triggers
- Empty trigger payload falls back to normal dictation.
- Trigger text is not sent to the model; only the payload is processed.
- If model cleanup is active, use the model cleanup path, not the heuristic rule
  cleanup path.
- Shell mode pastes the command text into the terminal, but TransClip never
  presses Enter or executes it.
- Shell output may be pasted even when operationally risky. The safety boundary
  is paste-without-execute plus user review before Enter.
- Shell output must pass a non-executing syntax check before it is pasted as
  runnable text. If syntax validation fails, paste a commented diagnostic
  instead of the command.

## Key Implementation Changes

### Branch and Settings

- Create or switch to `feature/voice-mode-routing` before implementation.
- Add settings fields:

```toml
voice_mode_routing_enabled = true
voice_model_cleanup_always_on = false
voice_mode_shell_enabled = true
text_model_runtime = "transformers"
text_model = "Qwen/Qwen3.5-4B"
shell_syntax_validation_enabled = true
shellcheck_enabled = true
```

- Keep current default dictation behavior unchanged:
  `voice_model_cleanup_always_on = false`.
- When `voice_model_cleanup_always_on = true`, normal dictation uses model
  cleanup even without a spoken cleanup trigger.
- Preserve strict unknown-field validation and update TOML serialization.

### Mode Routing

- Add a small mode-routing module that returns:

```text
mode: dictation | cleanup | shell
payload: str
trigger: str | None
literal: bool
```

- Normalize only for matching; preserve original payload text for model input.
- Match only at the beginning of the transcript after ASR and keyword restore.
- Case-insensitive matches must handle capitalization and punctuation such as
  `Shell command, list files`.
- Do not add fuzzy aliases in V1. Add aliases only after real ASR evals show a
  consistent misrecognition pattern.

### Text Model Backend

- Introduce a shared Transformers text-generation backend, loaded lazily and
  reused for both cleanup and shell generation.
- Reuse the service process and existing PyTorch/Transformers dependency path.
- Keep Granite 4.1 Speech NAR as ASR only; do not try to use the speech model
  for cleanup or shell generation.
- Default generation should be deterministic for both processors.

### Cleanup Processing

- Keep the existing heuristic cleanup as the normal default when model cleanup
  is not requested.
- Explicit `clean up ...` / `trans cleanup ...` always uses Qwen model cleanup.
- Tray always-on cleanup toggle makes all successful normal dictation pass
  through Qwen model cleanup.
- Cleanup prompt contract:
  - preserve meaning
  - correct punctuation, capitalization, and paragraphing
  - remove filler only when clearly filler
  - preserve technical terms, flags, identifiers, paths, and code-ish text
  - do not add facts

### Shell Command Processing

- Add a shell command processor that sends the payload to Qwen and expects a
  structured response internally.
- Render final pasted text as the shell command only, with no JSON and no
  explanatory prose.
- TransClip must not press Enter, execute the command, or add a trailing newline
  intended to submit the command.
- Use validation to catch malformed output and strip prose/fences.
- Validate shell syntax without execution before pasting runnable command text:
  - run `bash -n -c <command>` when `bash` is available
  - run `shellcheck -s bash -` when `shellcheck` is available and
    `shellcheck_enabled = true`
  - treat Bash parse failure or ShellCheck syntax errors as blocking
  - treat ShellCheck warnings/style notes as metadata, not blockers
- Operationally risky but syntactically valid commands may still be pasted in
  V1 because the workflow is user review before Enter, not automatic execution.
- If no usable command is returned, or syntax validation fails, render a
  commented diagnostic such as `# TransClip could not produce valid Bash: ...`
  so pasted text cannot be accidentally executed as the intended command.
- Record shell metadata in debug capture/history when available, while keeping
  the user-facing `text` field as the pasted command.

### Tray Toggle

- Add a tray menu toggle for always-on model cleanup, backed by
  `voice_model_cleanup_always_on`.
- The menu label should clearly show state, for example:
  - `✓ Model cleanup always on`
  - `Model cleanup always on`
- Persist the setting using the existing settings file path.
- Restart or reload the service consistently after toggling so the service uses
  the new setting for subsequent recordings.
- Update tray tests to cover label state, persistence, and restart/reload
  behavior.

### Documentation and Eval Prompts

- Update `CONTEXT.md` with mode routing, mode trigger, literal escape, model
  cleanup, and shell command generation terms.
- Update `README.md` with trigger examples, literal escapes, the tray cleanup
  toggle, and the rule that shell commands are pasted but never executed.
- Add trigger-recognition eval prompts:
  - `clean up ...`
  - `trans cleanup ...`
  - `shell command ...`
  - `bash command ...`
  - `terminal command ...`
  - `literal shell command ...`
  - ordinary sentences with trigger phrases in the middle

## Test Plan

- `tests.test_mode_routing`
  - case-insensitive cleanup and shell trigger matches
  - prefix-only behavior
  - literal escape behavior
  - empty trigger fallback to dictation
  - original payload preservation
- `tests.test_service`
  - normal dictation remains unchanged by default
  - explicit cleanup trigger uses model cleanup, not heuristic cleanup
  - always-on model cleanup applies to normal dictation
  - shell trigger returns command text as top-level `text`
  - shell mode never asks paste/injection code to execute anything
- `tests.test_cleanup`
  - model cleanup prompt contract and output parsing
  - heuristic cleanup still available for default non-model cleanup
- `tests.test_settings`
  - new settings defaults, TOML serialization, set/get coercion, unknown field
    rejection
- `tests.test_tray`
  - tray toggle label state
  - settings persistence
  - service restart/reload after toggle
- `tests.test_shell_command`
  - structured response parsing
  - prose/fence stripping
  - `bash -n` syntax validation pass/fail behavior
  - optional ShellCheck syntax-error handling when available
  - ShellCheck warnings do not block paste
  - command-only render
  - invalid command renders a commented diagnostic
  - no submit newline

Full verification before PR:

```bash
uv run -m unittest discover -v
```

Optional live eval after unit tests:

```bash
uv run scripts/record_real_eval_session.py ~/transclip-mode-routing-eval --manual-stop
```

Use live ASR clips only to decide whether to add future aliases; unit tests are
the deterministic acceptance gate for V1.

## Assumptions and Non-Goals

- Assume `Qwen/Qwen3.5-4B` is available in the configured local Hugging Face
  cache or can be prefetched through the existing model workflow.
- Do not introduce llama.cpp, Vulkan, ROCm/HIP llama.cpp, Ollama, or an external
  model server in this branch.
- Do not add a second shell-mode hotkey in this branch.
- Do not add fuzzy trigger matching in this branch.
- Do not execute shell commands, simulate Enter, or auto-submit terminal input.
- Do not treat ShellCheck as the only syntax gate. Bash syntax validation is the
  mandatory non-executing parse check; ShellCheck is an additional lint/syntax
  layer when installed.
- Do not change the Granite ASR backend or normal paste pipeline beyond feeding
  it the final routed text.
