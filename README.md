# TransClip

Local-only toggle-to-talk dictation for Linux, macOS, and Windows, with local ASR and
faithful cleanup for technical notes. Granite NAR is the default backend on Linux GPU,
Granite autoregressive on Windows, and MLX on macOS Apple Silicon. TransClip is the
product surface.

The default path is now the pure-Python dictation daemon:

```text
shortcut -> transclip toggle-record --paste -> Python service -> clipboard -> paste
```

The runnable app lives in `transclip/`: Python inference service,
settings, audio capture, cleanup, paste injection, daemon install/status/log
commands, debug capture, platform tray UIs, and eval harness.

Platform-specific desktop integration is grouped under `transclip/desktop/`
(paste, hotkey, tray), with service lifecycle in `transclip/daemon/` and
readiness checks in `transclip/doctor/`. See [docs/package-layout.md](docs/package-layout.md)
for the full package map and stable import paths.

## License

TransClip is licensed under the Apache License, Version 2.0. Model weights and
third-party dependencies are governed by their own licenses.

## Quick Start

Create default config files:

```bash
uv run -m transclip.cli init-config
```

This writes `settings.toml` under the platform config directory.

Install the daemon and native shortcut:

```bash
uv run -m transclip.cli install
```

On Linux this writes `~/.config/systemd/user/transclip.service`, enables
and starts it with `systemctl --user`, and installs the GNOME custom shortcut
`TransClip Toggle`. On this HP ZBook, `wev` reports the Copilot key as
`<Super><Shift>XF86TouchpadOff`; press once to start recording and again to
stop, transcribe, copy, and paste.

On macOS Apple Silicon, `install` writes the TransClip service LaunchAgent.
Install the native hotkey helper with `install-macos-hotkey`; it registers
`Option+Space` without relying on Shortcuts.app. Use the menu bar tray for
click-to-record after installing the optional UI extra:

```bash
uv sync --extra audio --extra mlx --extra macos-ui
transclip tray
```

On Windows, `install` registers a Task Scheduler logon task. Global hotkey
`ctrl+shift+space` is registered when `transclip tray` is running (Windows tray
in `transclip.desktop.tray.win32`). Sync optional UI dependencies for the
system tray and in-process hotkey:

```bash
uv sync --extra audio --extra models --extra windows-ui
transclip tray
```

Install a CUDA-enabled PyTorch wheel before prefetching Granite AR models, then
run `transclip models prefetch --model ibm-granite/granite-speech-4.1-2b`.
Granite NAR is not supported on Windows.

Check readiness and logs:

```bash
uv run -m transclip.cli status
uv run -m transclip.cli doctor
uv run -m transclip.cli smoke-test
uv run -m transclip.cli logs
```

### Voice Mode Quick Start

With the service running, press the toggle shortcut once to start recording and
again to stop. Ordinary speech is dictated normally. Start an utterance with one
of these phrases to choose another mode:

```text
clean up <text>              -> Qwen model cleanup
trans cleanup <text>         -> Qwen model cleanup
shell command <task>         -> Bash command generation
bash command <task>          -> Bash command generation
terminal command <task>      -> Bash command generation
literal shell command <text> -> paste "shell command <text>"
literal bash command <text>  -> paste "bash command <text>"
literal clean up <text>      -> paste "clean up <text>"
```

Trigger matching is case-insensitive and only applies at the beginning of the
utterance, so a sentence that mentions "shell command" later is still normal
dictation. Use `literal` when you want to dictate the trigger words themselves
instead of activating cleanup or shell mode.

Run the Python tray:

```bash
transclip tray
```

On Linux this uses PyGObject/Ayatana AppIndicator (GTK tray in
`transclip.desktop.tray.gtk`). When running through `uv`, the command hands
off to system Python if the project virtual environment does not expose `gi`. Install the system bindings if missing:

```bash
sudo apt install -y python3-gi gir1.2-ayatanaappindicator3-0.1
```

On macOS, `transclip tray` uses the native menu bar (`transclip.desktop.tray.macos`)
when `macos-ui` is installed (`uv sync --extra macos-ui`). The tray can copy
the hotkey setup command for Keyboard Shortcuts; global hotkeys are configured
manually in System Settings or Shortcuts.app.

Service controls:

```bash
uv run -m transclip.cli start
uv run -m transclip.cli stop
uv run -m transclip.cli restart
uv run -m transclip.cli uninstall
```

To run the service manually instead of using the service manager:

```bash
uv run -m transclip.cli serve
```

## macOS Apple Silicon Quick Start

Requirements: Apple Silicon, native ARM Python 3.12+, macOS 14+, and Xcode
Command Line Tools for `swiftc`:

For local edit/reinstall workflows, see
[docs/macos-local-development.md](docs/macos-local-development.md).

```bash
xcode-select --install
```

```bash
uv sync --extra audio --extra mlx --extra macos-ui
uv run -m transclip.cli init-config
uv run -m transclip.cli models prefetch --model mlx-community/whisper-large-v3-turbo-asr-fp16
uv run -m transclip.cli install
uv run -m transclip.cli install-macos-hotkey
uv run -m transclip.cli status
uv run -m transclip.cli doctor
transclip tray
```

`install-macos-hotkey` writes:

- `~/bin/transclip-toggle` — robust start/stop wrapper with logging and stale
  lock cleanup. If stop/transcription hangs, the wrapper restarts the service
  after 75 seconds; a later press can clear a stale wrapper after 90 seconds.
- `~/Applications/TransClipHotkey.app` — a tiny native event-tap helper for
  `Option+Space` with a menu-bar status item.
- `~/Library/LaunchAgents/com.paulbrav.transclip-hotkey.plist` — starts the
  helper at login.

After installing or reinstalling `TransClipHotkey.app`, refresh Accessibility
after the final `install-macos-hotkey` run. A later rebuild can invalidate the
grant again. Stop the helper before changing the grant:

```bash
launchctl bootout gui/$(id -u)/com.paulbrav.transclip-hotkey
```

Open **System Settings > Privacy & Security > Accessibility**, delete and re-add
**TransClipHotkey** or toggle it off and back on, then start the helper again:

```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.paulbrav.transclip-hotkey.plist
```

Usage: put the cursor in a text field, press `Option+Space` once to start
recording, speak, then press `Option+Space` again to stop, transcribe, copy, and
paste. Expect several seconds of transcription latency on the stop press.

Menu-bar transition:

| Phase | Label | Color | Meaning |
| --- | --- | --- | --- |
| Idle | `TC` | System label color | Ready and doing nothing. |
| Shortcut received | `TC...` | Yellow | `Option+Space` was detected and the helper is checking service state. |
| Recording | `REC` | Orange | The microphone is recording. |
| Busy | `TC...` | Yellow | A previous action is still running. |
| Transcribing | `TXT...` | Purple | Speech is being converted to text; this is usually the longest wait. |
| Paste requested / pasting | `PST...` | Teal | The transcript was copied and the helper is posting `Command+V`. |
| Finished | `OK` | Green | The transcript was pasted; this resets to `TC` after a short delay. |
| Recovering | `TC...` | Yellow | A stale wrapper is being cleared before trying again. |
| Error | `TC!` | Red | The helper needs attention, such as Accessibility or service recovery. |

Shortcuts.app is only a fallback now. If you use it, bind the command printed by
`install` or copied from the tray menu (`Copy hotkey setup command`).

Supported MLX ASR models on macOS:

- `mlx-community/whisper-large-v3-turbo-asr-fp16` (default)
- `mlx-community/granite-4.0-1b-speech-8bit` (`asr_backend = "granite_mlx"`)

Granite Speech 4.1 NAR (`asr_backend = "granite_nar"`) is also selectable on
Apple Silicon and runs via Torch/MPS; it and the optional Torch/MPS Granite AR
models require `uv sync --extra audio --extra models`.

### Permissions (macOS TCC)

| Action | Permission | Notes |
| --- | --- | --- |
| Recording | Microphone | Grant when macOS prompts for the process that starts recording. |
| Hotkey | Accessibility | Required for `TransClipHotkey.app` to see and consume `Option+Space`. |
| Paste | Accessibility | `TransClipHotkey.app` posts `Command+V` after copying the transcript. |

The native helper path does not require Accessibility entries for Shortcuts.app,
AppleScript `applet`, `osascript`, or `TransClipPaste`. Those names are artifacts
of manual or older setup attempts and can be removed from Accessibility if
present.

To remove the native hotkey helper:

```bash
uv run -m transclip.cli uninstall-macos-hotkey
```

## Windows Quick Start

Requirements: Windows 10+, Python 3.12+, NVIDIA CUDA PyTorch for GPU inference.

```bash
uv sync --extra audio --extra models --extra windows-ui
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
uv run -m transclip.cli init-config
uv run -m transclip.cli models prefetch --model ibm-granite/granite-speech-4.1-2b
uv run -m transclip.cli install
uv run -m transclip.cli status
uv run -m transclip.cli doctor
transclip tray
```

Supported ASR on Windows:

- `ibm-granite/granite-speech-4.1-2b` (default Granite autoregressive)
- `ibm-granite/granite-speech-4.1-2b-plus` (speaker/timestamp features)

Granite Speech 4.1 NAR and ROCm are not supported on Windows. Eval thresholds
for Windows Granite AR are in `eval/windows/manifest.json` (relaxed vs Linux NAR).

### Permissions (Windows)

| Action | Permission | Notes |
| --- | --- | --- |
| Recording | Microphone | Settings > Privacy & security > Microphone |
| Paste | Focused app | SendInput Ctrl+V; elevated apps may block injection (UIPI) |

## Linux CUDA / ROCm Quick Start

For the portable CPU/CUDA path, install the model extras first:

```bash
uv pip install -e '.[models,audio]'
```

On the current Linux `gfx1151` workstation, the V1 latency profile uses AMD's
TheRock ROCm nightly index plus FlashAttention's Triton AMD backend. The
canonical runtime environment is `.venv`; the systemd service and GNOME
shortcut should point at `.venv/bin/python3`. Do not use the local custom wheel
for this app; it fails GPU tensor execution on this host.

Use the helper script:

```bash
scripts/setup_gfx1151_env.sh
```

Or run the setup steps manually:

```bash
uv venv --python 3.13 .venv
uv pip install --python .venv/bin/python \
  --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ \
  --pre torch torchaudio torchvision pytorch-triton-rocm
uv pip install --python .venv/bin/python \
  -e . 'transformers>=4.52.1' 'accelerate>=1.0' 'soundfile>=0.12' 'sounddevice>=0.5'
FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE MAX_JOBS=4 \
  uv pip install --python .venv/bin/python --no-deps \
  flash-attn==2.8.3 --no-build-isolation
uv pip install --python .venv/bin/python einops
uv pip install --python .venv/bin/python flash-linear-attention
```

The default ASR backend is `ibm-granite/granite-speech-4.1-2b-nar`, selected
with `asr_backend = "granite_nar"`, because it is the measured low-latency V1
path on `gfx1151`. The higher-accuracy autoregressive
`ibm-granite/granite-speech-4.1-2b` path remains available with
`asr_backend = "granite"`. The current real-usage NAR run passes the V1 gate:
25 measured clips averaged 286 ms release-to-ready, with mean keyword
preservation at 0.952 and mean WER at 0.192.

### Incremental transcription (long recordings)

On Linux GPU hosts the `granite_nar` backend can transcribe long recordings
incrementally while you are still speaking: once the uncommitted buffer
exceeds ~10 s, audio up to the most recent pause is transcribed in the
background and trimmed from the buffer, so releasing the key only has to
process the short residual tail. Measured on gfx1151
(`eval/real-usage/long-recording-manifest.json`, real-time paced): 38-50 s
recordings finalize in 287-708 ms versus 1.4-2.0 s for the one-shot batch
pass. Because committed segments are transcribed without later audio context,
quality may differ slightly from a single batch pass; a paired same-audio
comparison has not been run yet, so incremental mode is opt-in. Short
utterances are unaffected (still a single batch pass). Committed text never
changes because committed audio is physically removed from the buffer.

```toml
incremental_transcription = true        # opt-in; defaults to false
incremental_commit_threshold_s = 10.0   # uncommitted audio that triggers a background pass
streaming_chunk_ms = 500                # mic chunk size fed to the session
warm_bucket_shapes_s = 16               # pre-warm NAR bucket shapes up to this length after startup (0 disables)
```

After the service reports ready, the remaining NAR tensor-bucket shapes (4 s
through `warm_bucket_shapes_s`, in 2 s steps) are compiled in a background
thread that yields whenever a recording is active, so the first long utterance
after a restart does not pay a multi-second ROCm shape compile.

On macOS MLX, startup warms 1 s buckets from 1-12 s before readiness. MLX skips
post-ready background bucket warming because warming long 16-60 s buckets can
churn the compiled state and make the next short microphone dictation pay a
large first-pass decode cost. For the interactive macOS path, keeping the short
startup buckets hot is more important than pre-compiling long clips.

| Platform | Batch default | Incremental | Notes |
|----------|---------------|-------------|-------|
| Linux GPU (CUDA/ROCm) | Granite NAR | Opt-in | No extra dependencies |
| Linux/Windows CPU | Granite CPU/AR | No | Requires the granite_nar GPU backend |
| Windows CUDA | Granite AR | No | granite_nar is not supported on Windows |
| macOS MLX | MLX Whisper | No | Warms MLX audio buckets for low-latency local dictation |

While recording, `GET /record/partial` and the tray's **Copy partial
transcript** expose the committed text so far. The final text always runs the
normal post-ASR pipeline. Leave `incremental_transcription` unset (or set it
to `false`) for the default single-pass batch behavior.

For fast local plumbing tests
without downloading a model, point `asr_backend` at a transcript file:

```toml
asr_backend = "file:/tmp/transcript.txt"
```

Model loading is offline by default:

```toml
models_local_files_only = true
model_cache_dir = "/path/to/local/huggingface/cache"
```

Populate the cache before running the service; the app should not download
models during dictation. The helper commands are:

```bash
uv run -m transclip.cli models list
uv run -m transclip.cli models doctor
uv run -m transclip.cli models prefetch --model ibm-granite/granite-speech-4.1-2b-nar
uv run -m transclip.cli models prefetch --model Qwen/Qwen3.5-4B
```

Run the helper through the same Python environment that runs the service. On
the current `gfx1151` workstation, model downloads should use:

```bash
.venv/bin/python3 -m transclip.cli models prefetch --model ibm-granite/granite-speech-4.1-2b-nar
.venv/bin/python3 -m transclip.cli models prefetch --model Qwen/Qwen3.5-4B
```

Voice mode routing runs after ASR and keyword restoration. Ordinary dictation
keeps the existing cleanup behavior unless a leading trigger phrase is spoken or
the tray setting enables model cleanup for all dictation. The CLI `cleanup`
command and `POST /cleanup` route follow that same dictation cleanup policy on
already-written text; they do not parse spoken trigger phrases. Shell mode validates
generated Bash with `bash -n -c <command>` when Bash is available and also uses
ShellCheck when installed and enabled. The shell prompt includes the user's
default shell from `$SHELL`, falling back to the login shell, while still asking
for Bash-compatible syntax. Invalid shell output is pasted as commented
diagnostic text. Valid shell commands are pasted for review only; TransClip
never presses Enter, executes the command, or auto-submits terminal input.

The tray menu includes `Model cleanup always on`. Enabling it persists
`voice_model_cleanup_always_on = true` and restarts the service so subsequent
ordinary dictation uses the shared Qwen text model:

```toml
voice_mode_routing_enabled = true
voice_model_cleanup_always_on = false
voice_mode_shell_enabled = true
text_model_runtime = "transformers"
text_model = "Qwen/Qwen3.5-4B"
shell_syntax_validation_enabled = true
shellcheck_enabled = true
```

Then transcribe a WAV:

```bash
uv run -m transclip.cli transcribe sample.wav
```

Install or refresh only the default GNOME shortcut for the Copilot key toggle
workflow:

```bash
uv run -m transclip.cli install-gnome-shortcut
```

This creates or updates the same `TransClip Toggle` shortcut while
preserving unrelated custom shortcuts.

## Linux Desktop

```bash
sudo apt update
sudo apt install -y \
  libayatana-appindicator3-dev \
  gir1.2-ayatanaappindicator3-0.1 \
  python3-gi \
  wl-clipboard \
  wtype \
  xdotool \
  ydotool
```

Linux GNOME sessions use the native custom shortcut installed above. No
`/dev/input` group membership is required for the default toggle workflow.

On GNOME Wayland, clipboard copy/read requires `wl-clipboard` (`wl-copy` and
`wl-paste`). Paste injection uses `wtype` when the compositor supports the
virtual keyboard protocol, then `ydotool` if configured. `xclip`/`xdotool` are
X11-only fallbacks.

## Eval Harness

Create a JSON manifest:

```json
{
  "warmup_cases": [
    {
      "audio_path": "clips/warmup.wav",
      "reference": "PyTorch on ROCm with gfx1151.",
      "keywords": ["PyTorch", "ROCm", "gfx1151"]
    }
  ],
  "cases": [
    {
      "audio_path": "clips/example.wav",
      "reference": "PyTorch on ROCm with gfx1151.",
      "keywords": ["PyTorch", "ROCm", "gfx1151"]
    }
  ]
}
```

Run:

```bash
uv run -m transclip.cli eval eval-manifest.json --output eval-results.json
```

The output includes release-to-ready latency, WER when references exist, keyword
preservation, and the number of warmup cases excluded from measured results.

For the required real-usage V1 eval, put 20 to 30 measured `.wav` clips and
matching reference `.txt` files in one folder. Optional per-clip keyword files
can use the same stem with `.keywords.txt`.

```bash
uv run scripts/record_real_eval_session.py ~/transclip-real-eval --manual-stop
```

To write the prompt list to a Markdown file first:

```bash
uv run scripts/record_real_eval_session.py ~/transclip-real-eval \
  --prompt-sheet eval/real-usage/prompts.md
```

Or add individual custom clips:

```bash
uv run scripts/record_real_eval_clip.py ~/transclip-real-eval case_01 \
  --duration 8 \
  --reference "Use PyTorch on ROCm with gfx1151." \
  --keywords PyTorch ROCm gfx1151
```

Then build and run the eval:

```bash
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
VIRTUAL_ENV=$PWD/.venv uv run --active scripts/run_real_eval_pipeline.py \
  ~/transclip-real-eval
```

## Tests

```bash
uv run -m unittest discover -s tests -v
uv run -m compileall scripts transclip tests
VIRTUAL_ENV=$PWD/.venv uv run --active scripts/check_v1_completion.py
```

Contributors changing imports or adding platform code should read
[docs/package-layout.md](docs/package-layout.md) for package boundaries and
public entry points.

On Wayland, `wtype` is only usable when the compositor supports the virtual
keyboard protocol; GNOME Wayland may reject it. `ydotool` can be used as a
lower-level fallback when its daemon/uinput permissions are configured. On X11
or an XWayland-oriented session, use `xdotool`.

Check host readiness:

```bash
uv run -m transclip.cli doctor
uv run -m transclip.cli doctor --fix
```
