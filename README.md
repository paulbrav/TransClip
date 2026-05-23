# TransClip

Local-only toggle-to-talk dictation for Linux and macOS, with local ASR and
faithful cleanup for technical notes. Granite NAR is the default backend, but
TransClip is the product surface.

The default path is now the pure-Python dictation daemon:

```text
shortcut -> transclip toggle-record --paste -> Python service -> clipboard -> paste
```

The runnable app lives in `transclip/`: Python inference service,
settings, audio capture, cleanup, paste injection, daemon install/status/log
commands, debug capture, Python AppIndicator tray, and eval harness.

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

On macOS Apple Silicon, `install` writes a LaunchAgent plist and prints the
shell command to bind in System Settings or Shortcuts.app. Default suggested
binding is `Option+Space`. Use the menu bar tray for click-to-record after
installing the optional UI extra:

```bash
uv sync --extra audio --extra mlx --extra macos-ui
transclip tray
```

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

On Linux this uses PyGObject/Ayatana AppIndicator. When running through `uv`,
the command hands off to system Python if the project virtual environment does
not expose `gi`. Install the system bindings if missing:

```bash
sudo apt install -y python3-gi gir1.2-ayatanaappindicator3-0.1
```

On macOS, `transclip tray` uses the native menu bar when `macos-ui` is
installed (`uv sync --extra macos-ui`). The tray can copy the hotkey setup
command for Keyboard Shortcuts; global hotkeys are configured manually in
System Settings or Shortcuts.app.

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

Requirements: Apple Silicon, native ARM Python 3.12+, macOS 14+.

```bash
uv sync --extra audio --extra mlx --extra macos-ui
uv run -m transclip.cli init-config
uv run -m transclip.cli models prefetch --model mlx-community/whisper-large-v3-turbo-asr-fp16
uv run -m transclip.cli install
uv run -m transclip.cli status
uv run -m transclip.cli doctor
transclip tray
```

Configure a global shortcut using the command printed by `install` or copied
from the tray menu (`Copy hotkey setup command`). Suggested binding:
`Option+Space`.

Supported MLX ASR models on macOS:

- `mlx-community/whisper-large-v3-turbo-asr-fp16` (default)
- `mlx-community/granite-4.0-1b-speech-8bit` (`asr_backend = "granite_mlx"`)

Granite Speech 4.1 NAR is not supported on macOS. Optional Torch/MPS Granite AR
models require `uv sync --extra audio --extra models`.

### Permissions (macOS TCC)

| Action | Permission | Notes |
| --- | --- | --- |
| Recording | Microphone | Grant to Terminal, IDE, LaunchAgent Python, or Shortcuts |
| Paste | Accessibility / Automation | Required for `osascript` paste injection |

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
preservation at 0.952 and mean WER at 0.192. For fast local plumbing tests
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

On Wayland, `wtype` is only usable when the compositor supports the virtual
keyboard protocol; GNOME Wayland may reject it. `ydotool` can be used as a
lower-level fallback when its daemon/uinput permissions are configured. On X11
or an XWayland-oriented session, use `xdotool`.

Check host readiness:

```bash
uv run -m transclip.cli doctor
uv run -m transclip.cli doctor --fix
```
