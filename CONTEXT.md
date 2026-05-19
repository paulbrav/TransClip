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
relies on, including macOS versus Linux behavior, Wayland versus X11,
environment variables, user paths, executable discovery, and subprocess command
execution.

**Paste capability**: The current ability to place transcript text on the
clipboard and inject a paste action into the focused application. Capability is
determined by the platform runtime, available clipboard tools, available input
injection tools, and any desktop permissions those tools require.

**Shortcut readiness**: Whether the native desktop shortcut used to trigger
interactive dictation is installed, points at the expected command, uses the
expected binding, and is available on the current desktop environment.

**ASR runtime**: The local speech-to-text execution path for a WAV file. It
includes audio preparation, backend selection, local model loading, transcript
generation, and backend timing details.

**Cleanup policy**: The rules used after ASR to preserve the spoken content
while improving transcript readability. The policy includes conservative
punctuation and capitalization behavior, optional model cleanup, output
validation, and token budgeting for model-backed cleanup.

**Eval gate**: The repeatable evaluation checks used to decide whether a build
meets the current dictation quality and latency expectations. It covers manifest
shape, warmup handling, measured cases, metrics, thresholds, and the JSON output
consumed by scripts.

**Runtime profile**: Platform-aware defaults and capability metadata, including
config/cache/log roots, supported inference runtimes (Torch CUDA/ROCm, MLX),
default ASR backend/model, and service manager (`systemd` on Linux, per-user
`launchd` on macOS dev installs).

**Backend catalog**: The mapping from model IDs to backend kind, runtime kind,
platform support, prefetch strategy, and cache layout. ASR factory dispatch and
doctor checks derive from the selected catalog entry.

**MLX ASR path**: macOS Apple Silicon inference through `mlx-audio` STT, with
offline prefetch via `huggingface_hub.snapshot_download` into the Hugging Face
hub cache under `~/Library/Caches/huggingface/hub` unless overridden.
