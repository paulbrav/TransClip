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

**Paste capability**: Text delivery after dictation via clipboard write and optional
shortcut injection into the focused application. On Linux terminals and AI CLIs
(Codex, Cursor CLI), injection uses `Ctrl+Shift+V` so the terminal emits
bracketed paste; bare `Ctrl+V` in Codex triggers image paste and fails for text.
When `focus_aware_paste` is enabled, GUI fields receive `Ctrl+V` instead.
Set `text_delivery_mode = "clipboard_only"` to copy without injecting keys.
Capability is determined by the platform runtime, available clipboard tools,
available input injection tools, and any desktop permissions those tools require.

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

## Code layout

TransClip groups platform integration into packages under `transclip/`. Domain
terms above describe product behavior; the layout below is for navigation and
imports. Full detail: [docs/package-layout.md](docs/package-layout.md).

**Platform runtime** (domain term) maps to `transclip.platform.runtime` and
related helpers in `transclip.platform.capabilities` and
`transclip.platform.profiles`.

**Paste capability** is implemented in `transclip.desktop.paste` (clipboard
read/write and paste injection backends).

**Shortcut readiness** on Linux uses `transclip.desktop.hotkey` (`install_shortcut`,
`get_gnome_shortcut_status`, `shortcut_readiness`). On Windows the tray registers
the global hotkey via `transclip.desktop.hotkey.start_windows_hotkey`. On macOS
there is no programmatic hotkey installer; setup messages and tray copy-to-clipboard
use `transclip.desktop.hotkey` and `transclip.desktop.hotkey.common`.

**Interactive dictation** tray UI is routed through `transclip.desktop.tray.run_tray`
to GTK, Windows, or macOS adapters. Service install and status use
`transclip.daemon`; readiness checks use `transclip.doctor`.

## CLI execution models

TransClip exposes two CLI paths to the dictation server:

**HTTP client commands** talk to a running local service via `InferenceClient`.
Examples: `toggle-record` (shortcut/tray workflow), and health probes from
`doctor`, `status`, and the tray. These require `transclip serve` (or an
installed daemon) to be active.

**In-process engine commands** construct `InferenceEngine` inside the CLI
process and load ASR/text backends directly. Examples: `transcribe`, `cleanup`,
and `eval`. These do not require the HTTP service but pay full model startup
cost per invocation.

Do not unify these paths blindly: toggle must stay lightweight and work when
only the daemon is running; batch transcribe/eval intentionally bypass HTTP.
Caller-side health polling lives in `transclip.service.client_health`, separate
from server-side health builders in `transclip.service.health`.
