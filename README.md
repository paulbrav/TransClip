# TransClip

Local-only toggle-to-talk dictation for Linux and macOS Apple Silicon, with
local ASR and faithful cleanup for technical notes. Linux defaults to Granite
NAR (Torch + FlashAttention 2 on GPU). macOS Apple Silicon defaults to MLX via
`mlx-audio`.

The default path is now the pure-Python dictation daemon:

```text
shortcut -> transclip toggle-record --paste -> Python service -> clipboard -> paste
```

The runnable app lives in `transclip/`: Python inference service,
settings, audio capture, cleanup, paste injection, daemon install/status/log
commands, debug capture, Python AppIndicator tray, and eval harness.

## Quick Start

Create default config files:

```bash
uv run -m transclip.cli init-config
```

This writes `settings.toml` under the platform config directory:

- Linux: `~/.config/transclip/settings.toml`
- macOS: `~/Library/Application Support/transclip/settings.toml`

Install the daemon and native shortcut:

```bash
uv run -m transclip.cli install
```

On Linux this writes `~/.config/systemd/user/transclip.service`, enables
and starts it with `systemctl --user`, and installs the GNOME custom shortcut
`TransClip Toggle`. On this HP ZBook, `wev` reports the Copilot key as
`<Super><Shift>XF86TouchpadOff`; press once to start recording and again to
stop, transcribe, copy, and paste.

Check readiness and logs:

```bash
uv run -m transclip.cli status
uv run -m transclip.cli doctor
uv run -m transclip.cli smoke-test
uv run -m transclip.cli logs
```

Run the Python tray (Linux only):

```bash
transclip tray
```

On macOS, `transclip tray` exits with guidance to use a Keyboard Shortcut or
Shortcuts.app action until a native menu bar UI exists.

On Linux this uses PyGObject/Ayatana AppIndicator. When running through `uv`,
the command hands off to system Python if the project virtual environment does
not expose `gi`. Install the system bindings if missing:

```bash
sudo apt install -y python3-gi gir1.2-ayatanaappindicator3-0.1
```

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
uv sync --extra audio --extra mlx
uv run -m transclip.cli init-config
uv run -m transclip.cli models prefetch --model mlx-community/whisper-large-v3-turbo-asr-fp16
uv run -m transclip.cli install
uv run -m transclip.cli status
uv run -m transclip.cli doctor
```

Configure a global shortcut or Shortcuts.app action to run
`transclip toggle-record --paste`. Grant Microphone permission when recording
starts. Grant Accessibility and/or Automation permission for `osascript` paste
injection. If paste is denied, the transcript remains on the clipboard.

Dev LaunchAgent logs live under `~/Library/Logs/transclip/`. Model cache defaults
to `~/Library/Caches/huggingface/hub`. Selectable MLX models:

- `mlx-community/whisper-large-v3-turbo-asr-fp16` (default)
- `mlx-community/granite-4.0-1b-speech-8bit` (`asr_backend = "granite_mlx"`)

Granite Speech 4.1 NAR is not supported on macOS. Use the MLX backends above.

## Linux CUDA / ROCm Quick Start

For the portable CPU/CUDA path, install the model extras first:

```bash
uv pip install -e '.[models,audio,llama]'
```

On the current Linux `gfx1151` workstation, the V1 latency profile uses AMD's
TheRock ROCm nightly index plus FlashAttention's Triton AMD backend. ROCm
support here is project-validated where FlashAttention ROCm works; it is not an
IBM-documented Granite runtime guarantee. Do not use the local custom wheel for
this app; it fails GPU tensor execution on this host.

```bash
uv venv --python 3.13 .venv-gfx1151
uv pip install --python .venv-gfx1151/bin/python \
  --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ \
  --pre torch torchaudio torchvision pytorch-triton-rocm
uv pip install --python .venv-gfx1151/bin/python \
  -e . 'transformers>=4.52.1' 'accelerate>=1.0' 'soundfile>=0.12' 'sounddevice>=0.5'
FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE MAX_JOBS=4 \
  uv pip install --python .venv-gfx1151/bin/python --no-deps \
  flash-attn==2.8.3 --no-build-isolation
uv pip install --python .venv-gfx1151/bin/python einops
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
models during dictation.

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
VIRTUAL_ENV=$PWD/.venv-gfx1151 uv run --active scripts/run_real_eval_pipeline.py \
  ~/transclip-real-eval
```

## Model / Runtime Matrix

| Platform | Default backend | Default model | Extra |
| --- | --- | --- | --- |
| Linux GPU | `granite_nar` | `ibm-granite/granite-speech-4.1-2b-nar` | `models` |
| macOS ARM | `mlx_audio_whisper` | `mlx-community/whisper-large-v3-turbo-asr-fp16` | `mlx`, `audio` |
| Test/file | `file:/path/to.txt` | n/a | none |

## Permissions (macOS TCC)

| Action | Permission | Identity |
| --- | --- | --- |
| Recording | Microphone | Terminal, LaunchAgent Python, Shortcuts, or packaged `.app` |
| Paste via `osascript` | Accessibility and/or Automation | Same controlling process |

## Tests

```bash
uv run -m unittest discover -s tests -v
uv run -m compileall scripts transclip tests
uv run ruff check .
VIRTUAL_ENV=$PWD/.venv-gfx1151 uv run --active scripts/check_v1_completion.py
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

## ChatGPT UI Bridge

For ChatGPT-only model modes that are visible in the web app but not exposed as
Codex/API model IDs, `scripts/chatgpt_ui_ask.py` can drive the visible ChatGPT
browser UI with Playwright. It uses a dedicated persistent browser profile and
does not extract or replay browser auth tokens.

Install the optional dependency and browser support:

```bash
uv run --extra chatgpt-ui playwright install chrome
```

First run with the browser left open, then log in and select any account prompts
manually:

```bash
uv run --extra chatgpt-ui scripts/chatgpt_ui_ask.py --keep-open "Say ready."
```

After the profile is logged in, ask through the visible UI:

```bash
uv run --extra chatgpt-ui scripts/chatgpt_ui_ask.py --model "Extended Pro" "Analyze this prompt."
```

Use stdin for larger prompts:

```bash
cat prompt.txt | uv run --extra chatgpt-ui scripts/chatgpt_ui_ask.py --model "Extended Pro"
```

The bridge defaults to `~/.cache/transclip-chatgpt-ui`. Use `--profile` to
keep a separate login/profile, `--reuse-current-chat` to continue the current
conversation, and `--keep-open` to inspect or repair the browser state.
