# Granite Speach

Local-only push-to-talk dictation for Linux and macOS, oriented around Granite
ASR and faithful cleanup for technical notes.

The repository has two runnable layers:

- `granite_speach/`: Python inference service, settings, glossary, audio capture,
  cleanup, paste injection, debug capture, and eval harness.
- `desktop/`: Tauri 2 desktop shell with tray actions, global hotkey capture,
  service status, and latest transcript copy/paste controls.

## Quick Start

Create default config files:

```bash
uv run -m granite_speach.cli init-config
```

This writes `settings.toml` and `keywords.txt` under the platform config
directory.

Run the local inference service:

```bash
uv run -m granite_speach.cli serve
```

For the portable CPU/CUDA path, install the model extras first:

```bash
uv pip install -e '.[models,audio,llama]'
```

On the current Linux `gfx1151` workstation, the V1 latency profile uses AMD's
TheRock ROCm nightly index plus FlashAttention's Triton AMD backend. Do not use
the local custom wheel for this app; it fails GPU tensor execution on this host.

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
`asr_backend = "granite"`. For fast local plumbing tests without downloading a
model, point `asr_backend` at a transcript file:

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
uv run -m granite_speach.cli transcribe sample.wav
```

## Desktop

Build the Tauri frontend:

```bash
cd desktop
npm install
npm run build
```

Run the Tauri shell after installing Linux Tauri prerequisites (`libsoup-3.0`,
`javascriptcoregtk-4.1`, WebKitGTK, rsvg) or on macOS with the normal Tauri
toolchain. The shell can start the Python localhost service from the status menu
or secondary status/debug window; set `GRANITE_SPEACH_UV` first when it
should use a specific `uv` binary.

```bash
sudo apt update
sudo apt install -y \
  libwebkit2gtk-4.1-dev \
  build-essential \
  curl \
  wget \
  file \
  libxdo-dev \
  libssl-dev \
  libayatana-appindicator3-dev \
  librsvg2-dev \
  wtype \
  xdotool \
  ydotool
```

```bash
npm run tauri dev
```

On a Linux session with a StatusNotifier/AppIndicator host, the tray
registration path can be smoke-tested without loading local models:

```bash
uv run scripts/linux_tray_smoke.py
```

See `docs/v1-completion-audit.md` for the current implementation and
verification matrix, and `docs/v1-live-validation.md` for the remaining
interactive Linux/macOS checks.

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
uv run -m granite_speach.cli eval eval-manifest.json --output eval-results.json
```

The output includes release-to-ready latency, WER when references exist, keyword
preservation, and the number of warmup cases excluded from measured results.

For the required real-usage V1 eval, put 20 to 30 measured `.wav` clips and
matching reference `.txt` files in one folder. Optional per-clip keyword files
can use the same stem with `.keywords.txt`.

```bash
uv run scripts/record_real_eval_session.py ~/granite-real-eval --manual-stop
```

To write the prompt list to a Markdown file first:

```bash
uv run scripts/record_real_eval_session.py ~/granite-real-eval \
  --prompt-sheet eval/real-usage/prompts.md
```

Or add individual custom clips:

```bash
uv run scripts/record_real_eval_clip.py ~/granite-real-eval case_01 \
  --duration 8 \
  --reference "Use PyTorch on ROCm with gfx1151." \
  --keywords PyTorch ROCm gfx1151
```

Then build and run the eval:

```bash
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
VIRTUAL_ENV=$PWD/.venv-gfx1151 uv run --active scripts/run_real_eval_pipeline.py \
  ~/granite-real-eval
```

To compare glossary on/off behavior for the same manifest:

```bash
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
VIRTUAL_ENV=$PWD/.venv-gfx1151 uv run --active scripts/run_keyword_ablation.py \
  eval/real-usage/manifest.json
```

## Tests

```bash
uv run -m unittest discover -s tests -v
uv run -m compileall scripts granite_speach tests
cd desktop && npm run build
cd desktop && npm run test:recorder
cd desktop && npm run test:tauri-recorder
cd desktop && npm run test:tauri-service-record
cd desktop && npm run test:tauri-paste
VIRTUAL_ENV=$PWD/.venv-gfx1151 uv run --active scripts/check_v1_completion.py
```

`cargo check` for `desktop/src-tauri` requires system Tauri/WebKitGTK
prerequisites on Linux. On Wayland, `wtype` is only usable when the compositor
supports the virtual keyboard protocol; GNOME Wayland may reject it. `ydotool`
can be used as a lower-level fallback when its daemon/uinput permissions are
configured. On X11 or an XWayland-oriented session, use `xdotool`.

Check host readiness:

```bash
uv run -m granite_speach.cli doctor
```
