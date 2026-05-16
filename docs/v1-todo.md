# V1 TODO

## Done

- [x] File-backed settings with Linux/macOS hotkey defaults.
- [x] Editable keyword glossary.
- [x] Custom `--settings` paths use the sibling `keywords.txt` file.
- [x] Granite Speech ASR backend using the documented Transformers
      `AutoProcessor`/`AutoModelForSpeechSeq2Seq` chat-template path.
- [x] Offline model loading by default with `models_local_files_only = true`
      and optional `model_cache_dir`.
- [x] Keyword-biased ASR prompt shape.
- [x] Gemma Transformers cleanup backend as an explicit quality cleanup
      runtime.
- [x] Rule-based faithful cleanup as the default latency cleanup runtime.
- [x] Explicit `llama.cpp` cleanup runtime.
- [x] Local HTTP inference service with `/health`, `/transcribe`,
      `/cleanup/transcribe`, `/cleanup`, `/record/start`, and `/record/stop`.
- [x] Debug capture artifacts, including transcription artifacts and HTTP
      failure `error.log` / `error.json` logs.
- [x] Eval harness with latency, WER, raw ASR WER, cleanup WER delta,
      cleanup semantic-drift failures, keyword preservation metrics, and
      optional paste success/failure metrics separated from model latency.
- [x] Clipboard paste, paste hotkey injection, and safe clipboard restoration.
- [x] Paste failure leaves the transcript on the clipboard and attempts a native
      notification from both the desktop shell and `record-once --paste`.
- [x] Tauri 2 shell with tray menu, global shortcut service-side microphone
      recording, cleanup toggle, latest copy/paste, settings/glossary open
      actions, local service start action, and quit action.
- [x] Tray menu exposes Record, Stop + Paste, and Stop actions directly, so
      recording is discoverable from the status icon without opening the
      secondary status/debug window.
- [x] Tauri shell now follows the status-app product framing: the main window
      is hidden from normal startup/taskbar use, the panel/menu bar icon opens
      the primary menu, and the window is only a secondary status/debug surface.
- [x] Tray/menu behavior cross-checked against the existing TransClip reference:
      status icon primary UI, hold-to-record hotkey, cleanup toggle, recent
      transcript menu, clipboard copy/paste actions, settings/config actions,
      and quit.
- [x] Tauri dev binary compiles and launches on the current Linux session.
- [x] Secondary status/debug window shows concrete health, hotkey, recording,
      transcribe, and paste errors instead of only switching status to `Error`.
- [x] Linux Tauri/WebKit shell grants WebKitGTK `UserMediaPermissionRequest`
      events so Web Audio microphone capture is not denied by default.
- [x] macOS Tauri bundle declares microphone and Apple Events usage strings in
      `Info.plist` for permission prompts.
- [x] Tauri settings/glossary actions create default config files before
      opening them when they are missing.
- [x] Python unit tests, package build, Tauri frontend build, Rust formatting,
      Tauri Rust check, CLI transcribe smoke test, debug capture smoke test,
      and eval smoke test.
- [x] HTTP endpoint tests for `/health`, `/cleanup`, `/transcribe`,
      `/cleanup/transcribe`, and service error handling.
- [x] `doctor` preflight command for config files, clipboard tools, paste
      injection tools, microphone devices, Tauri Linux libraries, local model
      cache directories, and Torch runtime status.
- [x] Configured Granite and Gemma model artifacts downloaded into the local
      Hugging Face cache.
- [x] Configured Granite NAR model artifacts downloaded into the local Hugging
      Face cache.
- [x] Real Granite ASR smoke test against the bundled multilingual sample.
- [x] Real Gemma cleanup smoke test against a short raw transcript.
- [x] Real Granite plus Gemma eval path smoke test against the bundled
      multilingual sample.
- [x] Torch device auto-selection probes CUDA/ROCm usability before selecting
      GPU, avoiding native crashes from incompatible ROCm wheels.
- [x] Identified and smoke-tested a working Linux `gfx1151` Torch runtime via
      AMD's TheRock nightly index. It passes GPU tensor smoke, `doctor`, and
      real Granite ASR inference in a throwaway environment.
- [x] Implemented explicit Granite NAR ASR backend using the model card's
      `AutoModel`/`AutoFeatureExtractor` remote-code path.
- [x] Installed persistent `.venv-gfx1151` runtime with TheRock ROCm Torch,
      FlashAttention Triton AMD, and the app dependencies.
- [x] Linux warm latency smoke test passes the V1 target: Granite NAR plus
      rule cleanup measured `273.7ms` then `243.3ms` end-to-end on a 3s clip
      after model warm-up.
- [x] Eval harness supports explicit `warmup_cases` and excludes them from
      measured case counts and latency summaries.
- [x] Eval harness now passes per-case manifest keywords into ASR and cleanup,
      so keyword-preservation metrics measure the intended glossary path.
- [x] Keyword glossary ablation helper added at
      `scripts/run_keyword_ablation.py`; it runs the same manifest with
      glossary keywords enabled and disabled, then reports preservation and WER
      deltas for the plan's glossary on/off comparison.
- [x] Repeatable synthetic 25-case Piper eval exists under `eval/v1-synthetic`;
      steady Granite NAR plus rule cleanup result after ROCm/MIOpen setup:
      mean latency `289.8ms`, 25/25 under `700ms`, mean WER `0.0856`,
      keyword preservation `0.992`, and zero cleanup semantic-drift failures.
- [x] Real-usage eval manifest builder added at `scripts/prepare_real_eval.py`;
      it validates WAV/reference pairs, enforces the 20-30 measured clip count,
      supports optional warmup clips, and attaches per-case keywords.
- [x] Repeatable Linux tray registration smoke test added at
      `scripts/linux_tray_smoke.py`; it verifies StatusNotifier/AppIndicator
      registration, expected DBusMenu labels, and DBusMenu Record/Stop actions
      through DBus against a fake local health service.
- [x] Secondary status/debug window has manual Record, Stop + Paste, and Stop
      controls so microphone capture and paste can be validated without
      relying on synthetic global-hotkey injection.
- [x] Browser-side WebAudio WAV recorder smoke test added at
      `scripts/desktop_recorder_smoke.py`; it runs the shared desktop recorder
      against Chromium's fake microphone and verifies a valid mono 16-bit WAV.
- [x] Actual Tauri/WebKit recorder smoke test added at
      `scripts/tauri_recorder_smoke.py`; visible-window WebAudio capture
      passes on the current Linux session, while hidden WebAudio timed out, so
      normal hotkey/manual recording now uses service-side microphone capture.
- [x] Actual Tauri paste smoke test added at `scripts/tauri_paste_smoke.py`;
      it exercises clipboard write, Tauri Rust paste injection, and clipboard
      restoration against a focused WebKit text field.
- [x] Tauri status shell can start the localhost Python inference service from
      the tray menu or secondary status/debug window, and avoids starting a
      duplicate service when port `8765` is already listening.
- [x] Added `docs/v1-live-validation.md` with concrete Linux/macOS interaction
      checks and real-usage eval commands.
- [x] Real-usage eval clip recorder added at `scripts/record_real_eval_clip.py`
      to create 16 kHz mono WAV/reference pairs for the manifest builder.
- [x] Guided real-usage eval session recorder added at
      `scripts/record_real_eval_session.py` for the standard 25-case technical
      dictation set, including `--manual-stop` mode that uses the same Python
      microphone recorder as the service.
- [x] Real-usage eval prompt sheet generated at `eval/real-usage/prompts.md`
      and supported by `record_real_eval_session.py --prompt-sheet`.
- [x] Real-usage eval pass/fail gate added at `scripts/check_eval_results.py`;
      it enforces measured case count, latency thresholds, keyword
      preservation, WER thresholds, cleanup semantic-drift failures, and
      recorded paste failures before accepting an eval result.
- [x] V1 readiness gate added at `scripts/check_v1_completion.py`; it combines
      the synthetic eval gate, required real-usage eval result, and `doctor`
      host checks into one pass/blocked report.
- [x] Live program run on the current Linux session: service and Tauri shell are
      running, `/health` reports ready with Granite NAR plus rule cleanup, tray
      menu labels are registered through DBus, `/cleanup` works, and warmed
      `/transcribe` on a short WAV measured `227.866ms` end-to-end.
- [x] Live tray `Start service` action activated through DBusMenu while the
      service was already running; it did not spawn a duplicate service and
      left `/health` ready.
- [x] Live tray `Show status window` action activated through DBusMenu on the
      current Linux session after the WebKit microphone permission fix; the
      shell stayed running and the tray status remained `Ready`.
- [x] Live service-side microphone recording check passed on the current Linux
      session: `/record/start` changed health to `recording`, and discard
      `/record/stop` returned to `ready` after about `602ms` on the detached
      service left running for manual testing.
- [x] Actual Tauri/WebKit frontend service-record smoke added at
      `scripts/tauri_service_record_smoke.py` and passed on the current Linux
      session. It used the visible Tauri status/debug window, entered
      `Recording`, enabled Stop, discarded the short recording, and returned to
      `Ready`.
- [x] Live tray Record and Stop actions activated through DBusMenu on the
      current Linux session: Record changed `/health` to `recording`, and Stop
      returned `/health` to `ready`.
- [x] Diagnostic Tauri global-hotkey smoke added at
      `scripts/tauri_global_hotkey_smoke.py`. On the current GNOME Wayland
      session, synthetic `ydotool key ctrl+space` did not trigger the Tauri
      global-shortcut handler, so physical shortcut validation remains a manual
      target-desktop check.
- [x] Live Tauri paste smoke run on the current GNOME Wayland session now
      passes through `ydotool`: it inserts `Granite Speach paste smoke` into a
      focused Tauri/WebKit field and restores the prior clipboard value. GNOME
      Wayland still rejects `wtype`, so this host uses the local `ydotool`
      binary symlinked in `~/.local/bin`.

## Blocked Verification

- [ ] Run live desktop interaction validation on Linux and macOS. Linux
      StatusNotifier/AppIndicator registration and menu labels have a passing
      DBus smoke test;
      OS-level Linux microphone capture has a passing `arecord` smoke test;
      `doctor` now reports visible microphone devices;
      browser-side WebAudio recording has a passing fake-microphone smoke test;
      visible Tauri/WebKit WebAudio recording has a passing smoke test;
      service-side microphone recording has a passing live HTTP smoke test;
      Tauri/WebKit paste injection has a passing smoke test through `ydotool`;
      tray menu Record/Stop has a passing live DBusMenu action check;
      still verify physical global hotkey press/release, manual Record/Stop from the
      status/debug window in the target physical session, and macOS menu bar/Accessibility behavior
      interactively on the target desktops.
- [ ] Run representative real-usage eval with 20 to 30 local WAV clips.
      Current host has the configured Granite autoregressive, Granite NAR, and
      Gemma model artifacts cached and single-sample real-model smoke tests
      passing, but no local eval WAV clip set was provided.
- [ ] Clear the V1 readiness gate with
      `VIRTUAL_ENV=$PWD/.venv-gfx1151 uv run --active scripts/check_v1_completion.py`.
      Current result is `blocked` because the real-usage eval results are
      missing.

## Host Setup

```bash
sudo apt-get install -y \
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

`wtype` requires compositor support for the Wayland virtual keyboard protocol.
The current GNOME Wayland session rejects that protocol. Automatic paste is
validated here through `ydotool`, with a local binary symlinked to
`~/.local/bin/ydotool` because `sudo` is password-gated. A normal system
install of `ydotool`, or an X11/XWayland path with `xdotool`, is still the
preferred target-machine setup.

Then run a real-model eval manifest with 20 to 30 WAV clips:

```bash
uv run scripts/record_real_eval_session.py ~/granite-real-eval --manual-stop
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

Optionally run the keyword glossary on/off ablation against the generated
manifest:

```bash
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
VIRTUAL_ENV=$PWD/.venv-gfx1151 uv run --active scripts/run_keyword_ablation.py \
  eval/real-usage/manifest.json
```

For the current Linux `gfx1151` host, the working ROCm Torch path found from
the Journey notes is AMD's TheRock nightly index:

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
