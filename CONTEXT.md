# TransClip Context

This file records current project vocabulary used by architecture plans and code
reviews. It describes the existing product behavior and should not be treated as
a list of promised module names.

## Domain Terms

**Dictation session**: The lifecycle that starts when TransClip begins
recording microphone audio and ends when the recording is stopped, discarded, or
transcribed. It includes timing rules such as the toggle cooldown and minimum
recording duration.

**Interactive dictation**: The user-facing toggle workflow used by the native
shortcut, tray, and CLI. It asks the local service to start or stop a dictation
session, then may copy and paste the transcript and report the outcome to the
caller.

**Platform runtime**: The operating-system and desktop-session facts TransClip
relies on, including macOS versus Linux versus Windows behavior, Wayland versus
X11, environment variables, user paths, executable discovery, and subprocess
command execution.

**Paste capability**: The current ability to place transcript text on the
clipboard and inject a paste action into the focused application. Capability is
determined by the platform runtime, available clipboard tools, available input
injection tools, and any desktop permissions those tools require.

**Shortcut readiness**: Whether the native desktop shortcut used to trigger
interactive dictation is installed, points at the expected command, uses the
expected binding, and is available on the current desktop environment. On Linux
this is the GNOME custom shortcut installed by `transclip install`. On macOS
global shortcuts are configured manually in System Settings or Shortcuts.app;
`hotkey_macos` stores the suggested binding while the tray can copy the toggle
command wrapper. On Windows the in-process `keyboard` hook in `transclip tray`
registers `hotkey_windows` (default `ctrl+shift+space`).

**Runtime profile**: Platform-aware defaults for ASR backend, model, device,
service manager, and supported runtime kinds. Linux x86_64 defaults to Granite
NAR with systemd; Linux CPU defaults to Granite AR. macOS Apple Silicon defaults
to MLX Whisper via `mlx-audio` with launchd. Windows defaults to Granite AR
(`ibm-granite/granite-speech-4.1-2b`) with CUDA when available and Task
Scheduler for the background service. Granite NAR is not supported on Windows.

**ASR runtime**: The local speech-to-text execution path for a WAV file. It
includes audio preparation, backend selection, local model loading, transcript
generation, and backend timing details.

**Cleanup policy**: The rules used after ASR to preserve the spoken content
while improving transcript readability. The policy includes conservative
punctuation and capitalization behavior, optional model cleanup, output
validation, and token budgeting for model-backed cleanup.

**Voice mode routing**: The deterministic post-ASR routing step that runs after
keyword restoration and before cleanup or paste output. It matches only
case-insensitive leading trigger phrases and returns dictation, cleanup, or
shell mode with the original payload text preserved for model input.

**Mode trigger**: A configured spoken prefix such as `clean up`, `trans
cleanup`, `shell command`, `bash command`, or `terminal command`. Triggers are
prefix-only; the same words in the middle of an ordinary sentence do not change
dictation mode.

**Literal escape**: The leading `literal` prefix before a configured mode
trigger. It disables mode routing for that utterance and pastes the trigger text
itself, for example `literal shell command list files` becomes `shell command
list files`.

**Model cleanup**: The Qwen-backed cleanup path used for explicit cleanup
triggers and for normal dictation when `voice_model_cleanup_always_on` is
enabled. It is separate from the conservative rule cleanup that remains the
default for normal dictation. Rule cleanup is heuristic punctuation and
capitalization; it does not use a separate cleanup model setting.

**Direct cleanup API**: The `POST /cleanup` route and CLI `cleanup` command clean
already-written text through the dictation cleanup policy only. They do not
parse voice-mode triggers; they apply rule cleanup by default and use the Qwen
`text_model` when `voice_model_cleanup_always_on` is enabled.

**Shell command generation**: The voice mode that turns a spoken task into Bash
text using the shared Qwen text model. It validates syntax without execution and
pastes only command text or commented diagnostics; TransClip does not press
Enter or submit terminal input.

**Eval gate**: The repeatable evaluation checks used to decide whether a build
meets the current dictation quality and latency expectations. It covers manifest
shape, warmup handling, measured cases, metrics, thresholds, and the JSON output
consumed by scripts.
