# V1 Completion Audit

Objective: fully implement `docs/v1-plan.md` and verify it with concrete tests.

## Deliverable Checklist

| Requirement | Evidence | Status |
| --- | --- | --- |
| Local-only operation, no cloud ASR/cleanup or telemetry | Python service uses local ASR/cleanup backends only; model loading is `local_files_only` by default; no network calls except localhost service access from desktop. | Implemented |
| English-only V1 settings | `granite_speach/settings.py` default `language = "en"`; health endpoint reports language. | Implemented |
| Configurable Linux/macOS hotkeys | `Settings.hotkey_linux`, `Settings.hotkey_macos`; service health exposes `hotkey`; Tauri registers it through global-shortcut. | Implemented |
| Hold-to-record workflow | Tauri global shortcut press/release starts/stops service-side microphone recording through `/record/start` and `/record/stop`, then transcribes and pastes. TransClip reference repo uses the same hold-to-record status-icon app pattern. | Implemented, needs live desktop validation |
| Status states | Tauri uses `Loading`, `Ready`, `Recording`, `Transcribing`, `Cleaning`, `Pasting`, `Error`. | Implemented |
| Status icon/menu primary UI | Tauri config starts the main window hidden and skipped from taskbar; the tray/status icon menu is the primary surface and includes status, cleanup toggle, direct Record/Stop actions, latest transcript actions, recent transcripts, settings/glossary, show status window, and quit. Closing the secondary window hides it instead of quitting. Linux StatusNotifier/AppIndicator registration and DBusMenu labels pass smoke. | Implemented, needs live interaction validation |
| Cleanup toggle | Tauri checkbox and checkable tray menu toggle; service accepts cleanup flag. | Implemented |
| Paste latest transcript | Tauri tray/menu button calls paste path. | Implemented |
| Copy latest transcript | Tauri tray/menu button writes clipboard. | Implemented |
| Open/edit glossary/settings | Tauri tray invokes Rust `open_config_file`, which creates default files first when missing. | Implemented |
| Quit action | Tauri tray invokes `quit`. | Implemented |
| Clipboard insertion first | `granite_speach/paste.py` and Tauri `pasteTranscript` write clipboard, simulate paste, then optionally restore previous clipboard. | Implemented |
| Paste failure behavior | Python and Tauri paste paths leave the transcript on the clipboard on failure; CLI `record-once --paste` and Tauri paste failure paths attempt native notification. | Implemented |
| Two-process architecture | Python local HTTP service plus Tauri shell/client. | Implemented |
| ASR harness | `granite_speach/asr.py`, `cli transcribe`, explicit file test backend, timings. | Implemented |
| Granite ASR default | Default `asr_model = "ibm-granite/granite-speech-4.1-2b-nar"` with explicit `asr_backend = "granite_nar"` for the measured V1 latency profile. The autoregressive `ibm-granite/granite-speech-4.1-2b` model-card `AutoProcessor`/`AutoModelForSpeechSeq2Seq` path remains available with `asr_backend = "granite"`. | Implemented, real-model smoke-tested |
| Keyword glossary | `keywords.txt` defaults, loader, ASR keyword prompt, cleanup preservation instruction; custom `--settings` paths load sibling `keywords.txt`. | Implemented |
| Cleanup harness | Default faithful rule cleanup for latency; explicit Gemma Transformers cleanup for quality benchmarking; optional explicit `llama.cpp`; `cli cleanup`; service cleanup endpoint. | Implemented |
| Gemma cleanup candidate | `cleanup_model = "google/gemma-4-E2B-it"` with explicit `cleanup_runtime = "transformers"`. | Implemented, real-model smoke-tested |
| Local inference service endpoints | `/health`, `/transcribe`, `/cleanup/transcribe`, `/cleanup`, `/record/start`, `/record/stop`; CORS for Tauri; HTTP handler covered by tests; Tauri shell can start the localhost service and avoids duplicate starts when port `8765` is already listening. | Implemented |
| Model warm process | Service constructs ASR and cleanup backends once in `InferenceEngine`. | Implemented |
| Minimal Tauri status shell | `desktop/` Tauri 2 app, hidden secondary status window, tray/menu-bar menu, global shortcut, clipboard plugin, service-side microphone recording, paste command. | Implemented, Rust check passed |
| Microphone capture | Primary desktop capture uses service-side `AudioRecorder` endpoints so hidden status-shell hotkeys do not depend on hidden WebKit `getUserMedia`; secondary WebAudio recorder remains smoke-tested in Chromium and visible Tauri/WebKit; OS-level Linux capture and `doctor` microphone-device detection pass. | Implemented, needs live hotkey/paste validation |
| Recording length bounds | Settings defaults `max_recording_seconds = 60`, `min_recording_ms = 250`; Tauri enforces both. | Implemented |
| Debug capture | Writes successful transcription artifacts (`audio.wav`, `raw_asr.txt`, `cleaned.txt`, `timings.json`, `model_versions.json`) and HTTP failure artifacts (`error.log`, `error.json`) when enabled. | Implemented and smoke-tested |
| Eval harness | `cli eval` computes latency, WER, raw ASR WER, cleanup WER delta, cleanup semantic-drift failures, keyword preservation, and optional paste success/failure metrics separated from model latency. | Implemented and smoke-tested |
| Keyword glossary on/off comparison | `scripts/run_keyword_ablation.py` runs the same manifest with glossary keywords enabled and disabled, then reports keyword-preservation and WER deltas. | Implemented, needs real eval run |
| Linux/macOS paste commands | Python: `wl-copy`/`wl-paste`, `xclip`, `xsel`, `wtype`, `xdotool`, `ydotool`, `pbcopy`/`pbpaste`, `osascript`; Tauri Rust: `wtype`, `xdotool`, `ydotool`, `osascript` with stderr-preserving failure reports; macOS Apple Events usage declaration included for paste automation. Current GNOME Wayland compositor rejects `wtype` virtual keyboard events, but a local `ydotool` binary is wired through `~/.local/bin/ydotool` and the Tauri paste smoke passes. | Implemented and Linux smoke-tested |
| No autostart | No autostart plugin or installer config. | Implemented |
| No default persistence | Debug capture disabled by default; latest transcript only in memory/clipboard. | Implemented |

## Verification Run

- `uv run -m unittest discover -s tests -v`: 77 tests passed.
- `uv run -m compileall scripts granite_speach tests`: passed.
- `uv run -m granite_speach.cli --settings "$tmpdir/settings.toml" init-config`: passed and emitted offline model-loading defaults.
- `uv run -m granite_speach.cli init-config`: created `/home/paulbrav/.config/granite-speach/settings.toml` and `keywords.txt`.
- `VIRTUAL_ENV=$PWD/.venv uv run --active -m granite_speach.cli doctor`: implemented and run; config files, clipboard tools, Tauri Linux libraries, and model cache checks pass. The portable `.venv` reports CPU selection and does not satisfy the default Granite NAR runtime because `flash_attn` is not installed there.
- `VIRTUAL_ENV=$PWD/.venv-gfx1151 uv run --active -m granite_speach.cli doctor`:
  now reports config files, clipboard, paste tools, microphone, Tauri
  libraries, model cache, Torch, and Granite NAR runtime checks passing. The
  current host uses `ydotool` for paste because GNOME Wayland rejects `wtype`
  virtual keyboard events.
- `hf download ibm-granite/granite-speech-4.1-2b --include ...`: completed configured Granite ASR model artifact cache, `4.6G`.
- `hf download ibm-granite/granite-speech-4.1-2b-nar --include ...`: completed configured Granite NAR ASR model artifact cache, about `4.5G` for `model.safetensors` plus model code/config assets.
- `hf download google/gemma-4-E2B-it --include ...`: completed configured Gemma cleanup model artifact cache, `9.6G`.
- `VIRTUAL_ENV=$PWD/.venv uv run --active -m granite_speach.cli transcribe --no-cleanup <granite bundled sample>`: Granite weights loaded and transcribed the bundled multilingual sample in about 38s end-to-end.
- `VIRTUAL_ENV=$PWD/.venv uv run --active -m granite_speach.cli cleanup <short transcript>`: Gemma weights loaded and produced punctuated cleanup output from a short raw transcript.
- `VIRTUAL_ENV=$PWD/.venv uv run --active -m granite_speach.cli eval <one-case granite bundled sample manifest>`: real Granite plus Gemma eval path completed with `keyword_preservation = 1.0`; cold CPU end-to-end latency was about 66s, so this is a path smoke test, not a latency pass.
- ROCm investigation: ROCm sees `gfx1151`; the public ROCm 6.4 PyTorch wheel segfaulted after Granite weights loaded, and the local `/home/paulbrav/build/pytorch-gfx1151/.../torch-2.9.1a0+gitd38164a...whl` failed a basic GPU tensor operation with `no kernel image is available for execution on the device`. Old Journey notes pointed to AMD's TheRock nightly index for `gfx1151`; a throwaway `/tmp/granite-therock-test` env with `torch 2.11.0+rocm7.13.0a20260424`, `hip=7.13.26162`, and `rocm-sdk-libraries-gfx1151` passed GPU tensor smoke, `doctor`, and real Granite ASR inference.
- `VIRTUAL_ENV=/tmp/granite-therock-test uv run --active -m granite_speach.cli doctor`: passed and reported `torch 2.11.0+rocm7.13.0a20260424 hip=7.13.26162; GPU tensor smoke passed`.
- `VIRTUAL_ENV=/tmp/granite-therock-test uv run --active -m granite_speach.cli transcribe --no-cleanup <granite bundled sample>`: completed on the TheRock `gfx1151` ROCm runtime in about 35s end-to-end for the cold CLI path.
- Warm TheRock ROCm ASR measurements: the full 24.94s bundled sample measured about 12.0s after model load; a 3s clipped sample measured about 1.1-1.3s after model load. This verifies a working accelerated runtime, but it still does not meet the V1 700ms short-snippet latency target.
- Granite NAR investigation: the model requires FlashAttention2. Explicit `eager` and `sdpa` attention failed by design with `NLENARDecoder requires flash_attention_2`. PyPI FlashAttention CK build failed on `gfx1151`, while FlashAttention's Triton AMD backend installed with `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE` and `--no-deps` to preserve TheRock Torch.
- Persistent `.venv-gfx1151` created with TheRock ROCm Torch, FlashAttention Triton AMD, `einops`, app dependencies, and local editable install.
- `VIRTUAL_ENV=$PWD/.venv-gfx1151 uv run --active -m granite_speach.cli doctor`: passed with visible microphone devices, `torch 2.11.0+rocm7.13.0a20260424 hip=7.13.26162; GPU tensor smoke passed`, and `Granite NAR flash-attn runtime import passed`.
- Warm V1 latency profile measurement using `InferenceEngine(load_settings())` with Granite NAR plus rule cleanup on a 3s clipped sample: cold run `17.2s` including model load; warm runs `273.7ms` and `243.3ms` end-to-end.
- Eval harness now supports `warmup_cases` so model-load warmup can be excluded from latency metrics. `VIRTUAL_ENV=$PWD/.venv-gfx1151 uv run --active -m granite_speach.cli eval /tmp/granite-warm-eval.json --output /tmp/granite-warm-eval-results.json` passed with one warmup case and one measured case: `mean_release_to_ready_ms = 273.585`, `under_700ms = 1`, `keyword_preservation = 1.0`.
- Eval harness passes per-case manifest keywords into ASR and cleanup, not just the engine-global glossary, so keyword-preservation metrics reflect the case under test.
- Keyword glossary ablation helper added at `scripts/run_keyword_ablation.py`
  and unit-tested. It runs each manifest case once with the case/global
  glossary keywords and once with an empty keyword list, then reports
  preservation and WER deltas for the plan's glossary on/off comparison.
- `scripts/prepare_real_eval.py`: added and unit-tested real-usage manifest builder for 20-30 measured WAV/reference pairs, optional warmup clips, global keyword matching, and per-clip `.keywords.txt` files.
- `scripts/record_real_eval_clip.py`: added and unit-tested `arecord` command builder for creating 16 kHz mono WAV/reference pairs for real-usage eval.
- `scripts/record_real_eval_session.py`: added and unit-tested guided recorder for the standard 25-case technical dictation eval set. It records live microphone WAVs, writes matching references and per-case keyword files for `prepare_real_eval.py`, supports manual Enter-to-stop recording with `--manual-stop`, and can write a prompt sheet with `--prompt-sheet`.
- `scripts/check_eval_results.py`: added and unit-tested pass/fail gate for real-usage eval results. It rejects wrong measured case counts, missing/invalid latencies, slow worst/mean latency, low under-700ms ratio, low keyword preservation, high mean WER, cleanup semantic-drift failures, and recorded paste failures.
- `scripts/run_real_eval_pipeline.py`: added and unit-tested one-command
  real-eval handoff. It builds `eval/real-usage/manifest.json`, runs
  `granite_speach.cli eval`, checks `eval/real-usage/results.json`, and then
  runs the V1 completion gate. With no recorded clips, it exits early with the
  recording command instead of producing fake evidence.
- Eval harness paste metrics fix: manifest cases can include
  `paste_attempted` and `paste_success`; results preserve those fields and the
  summary reports `paste_attempts`, `paste_successes`, and `paste_failures`
  without mixing paste behavior into model latency timings.
- `uv run scripts/generate_synthetic_eval.py`: generated one warmup clip and 25 measured 16 kHz mono Piper clips under `eval/v1-synthetic`.
- `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE VIRTUAL_ENV=$PWD/.venv-gfx1151 uv run --active -m granite_speach.cli eval eval/v1-synthetic/manifest.json --output eval/v1-synthetic/results.json`: steady Granite NAR plus rule cleanup result after ROCm/MIOpen setup was `cases = 25`, `mean_release_to_ready_ms = 289.84368`, `under_700ms = 25`, `under_1500ms = 25`, `mean_wer = 0.08563969363969365`, `mean_raw_asr_wer = 0.18125896325896326`, `mean_cleanup_drift_wer_delta = 0.0`, `cleanup_semantic_drift_failures = 0`, `mean_keyword_preservation = 0.992`, `warmup_cases = 1`.
- `uv run scripts/check_eval_results.py eval/v1-synthetic/results.json`: passed the V1 eval gate for the synthetic benchmark with 25 cases, `mean_release_to_ready_ms = 289.84368`, worst latency `370.199ms`, `under_700_ratio = 1.0`, `mean_keyword_preservation = 0.992`, `mean_wer = 0.08563969363969365`, and `cleanup_semantic_drift_failures = 0`.
- `VIRTUAL_ENV=$PWD/.venv-gfx1151 uv run --active scripts/check_v1_completion.py`:
  added as the V1 readiness gate and run on the current host. It reports
  `status = blocked` only because `eval/real-usage/results.json` is missing.
  The same output records the passing synthetic eval summary and passing model,
  microphone, clipboard, paste, Tauri library, Torch, and Granite NAR runtime
  checks.
- `cd desktop && npm run test:tauri-hotkey`: diagnostic global-shortcut smoke
  added. It launches Tauri against a fake service and sends `Ctrl+Space` with
  `ydotool`, but on the current GNOME Wayland session the synthetic key event
  did not reach the Tauri global-shortcut handler: `/record/start` was not
  called. This is treated as an input-injection/compositor limitation, not a
  passing validation of the physical shortcut.
- `VIRTUAL_ENV=$PWD/.venv-gfx1151 uv run --active -m granite_speach.cli serve` launched the local service with the measured config. `curl /health` reported `asr_backend = "granite-nar-transformers"` and `cleanup_backend = "rule-based"`; `curl /cleanup` returned rule-cleaned text and timings.
- Live program run: service launched with `VIRTUAL_ENV=$PWD/.venv-gfx1151 TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE uv run --active -m granite_speach.cli serve` and Tauri launched with `npm run tauri dev`. Running PIDs were service wrapper `691695`, service child `691707`, npm `692277`, Vite `692387`, and Tauri app `692430`. Logs: `/tmp/granite-speach-service.log` and `/tmp/granite-speach-tauri.log`.
- Live program `/health`: reported `status = ready`, `asr_backend = granite-nar-transformers`, `cleanup_backend = rule-based`, `hotkey = Ctrl+Space`, and `paste_shortcut = Ctrl+V`.
- Live program tray DBus item: latest inspected item
  `:1.7894@/org/ayatana/NotificationItem/tray_icon_tray_app_973789_1`, with
  DBusMenu labels `Status: Ready`, `Cleanup`, `Record`, `Stop + Paste`, `Stop`,
  `Paste latest transcript`, `Copy latest transcript`, `Recent transcripts`,
  `Start service`, `Show status window`, `Open keyword glossary`,
  `Open settings`, and `Quit`.
- Live tray action check: activating `Start service` through DBusMenu while the service was already listening kept the managed service child count at `1` and `/health` still returned `ready`; the tray status returned to `Status: Ready`.
- Live tray action check: activating `Show status window` through DBusMenu on
  the current Linux session after the WebKit user-media permission fix
  completed without crashing the app; the tray menu still reported
  `Status: Ready`.
- Live tray action check: activating `Record` through DBusMenu changed
  `/health` from `ready` to `recording`; activating `Stop` through the rebuilt
  DBusMenu returned `/health` to `ready`.
- Live program service checks: `POST /cleanup` returned `Hello, world from ROCm.` through the rule backend. `POST /transcribe` on `eval/v1-synthetic/audio/case_01.wav` returned `Please check the Tauri tray icon after the service report ready.` First request measured `end_to_end = 11061.535ms` due to first-use model/ROCm setup; second warmed request measured `end_to_end = 227.866ms`.
- Live service recorder check on the detached service left running for manual
  testing: `POST /record/start` returned `status = recording`, discard
  `POST /record/stop` returned `status = ready` with `duration_ms = 602.193`,
  and subsequent `/health` returned `status = ready`.
- `uv run -m granite_speach.cli --settings "$tmpdir/settings.toml" doctor --json`: passed config-file detection after `init-config` and reported host/model prerequisites.
- `uv build`: source distribution and wheel built.
- `cd desktop && npm install`: passed.
- `cd desktop && npm run build`: TypeScript and Vite build passed.
- `cd desktop && npm run test:recorder`: passed; Chromium fake microphone produced a valid WAV through the shared desktop `WavRecorder` with `RIFF`/`WAVE`, one channel, `48000 Hz`, 16-bit PCM, and non-empty data.
- `cd desktop && npm run test:tauri-recorder`: passed on the current Linux
  Tauri/WebKit shell with a visible status window; it produced a valid PCM WAV
  with `RIFF`/`WAVE`, one channel, `44100 Hz`, 16-bit samples, `73728`
  data bytes, and `835ms` duration. The same smoke previously timed out when
  run from the hidden webview, so normal recording was moved to the service
  recorder endpoints instead of hidden WebAudio.
- `cd desktop && npm run test:tauri-service-record`: passed on the current
  Linux Tauri/WebKit shell against the live service. The frontend entered
  `Recording`, enabled the Stop control, discarded the short recording through
  `/record/stop`, and returned to `Ready`.
- `cd desktop && npm run test:tauri-paste`: passes on the current GNOME Wayland
  session after wiring a local `ydotool` binary into `~/.local/bin`. The smoke
  focused a real Tauri/WebKit textarea, inserted `Granite Speach paste smoke`,
  and restored the previous clipboard value.
- Tauri shell now exposes `Start service` from the tray menu and secondary status/debug window. The Rust command finds the repo root, uses `uv run`, preferring `GRANITE_SPEACH_UV` when set and setting `VIRTUAL_ENV` to `.venv-gfx1151` or `.venv` and using `uv run --active` when those environments exist; it stores the managed child and kills it on app quit. It returns `service already running` when `127.0.0.1:8765` is already listening.
- Secondary status/debug window now includes manual Record, Stop + Paste, and Stop controls for live microphone/paste validation without relying on synthetic global-hotkey injection. These controls now use service-side `/record/start` and `/record/stop` so they share the same capture path as the global hotkey. `cd desktop && npm run build` passed after this change.
- Secondary status/debug window now displays concrete frontend errors for health checks, hotkey registration, recording, transcription, and paste. `Ctrl+Space` hotkey registration failure no longer forces the whole UI into `Error`; the service can remain `Ready` and manual Record can still be used.
- Paste failure notification fix: Tauri now invokes a native `show_notification`
  command on paste failure after leaving the transcript on the clipboard, and
  CLI `record-once --paste` uses the existing local `notify` helper when paste
  injection fails.
- Tauri paste injector fix: Linux `simulate_paste` now captures and reports
  injector stdout/stderr and tries another installed injector before returning
  failure. This exposed the current compositor's concrete `wtype` virtual
  keyboard rejection.
- Python paste injector fix: CLI `record-once --paste` now preserves the same
  runtime injector failure details and can try `xdotool` after a failed
  `wtype` attempt when both are installed.
- `ydotool` fallback added to both Python and Tauri paste injection. It is a
  Wayland fallback for compositors that reject `wtype`. On the current host,
  `sudo` is password-gated, so the Ubuntu `ydotool` package was extracted under
  `.local-tools/ydotool-deb` and symlinked to `~/.local/bin/ydotool`; `/dev/uinput`
  already grants this user read/write access.
- `ydotool` command syntax fixed from raw keycode transitions to
  `ydotool key ctrl+v`; the Ubuntu `0.1.8` CLI otherwise typed `2442` into the
  focused field instead of sending Ctrl+V.
- `command -v notify-send`: passed on the current Linux host.
- Linux Tauri/WebKit user-media permission fix: `desktop/src-tauri/src/lib.rs` attaches a WebKitGTK `permission-request` handler to the main webview and allows `UserMediaPermissionRequest`, preventing default `NotAllowedError` denial for visible WebAudio microphone capture.
- `uv run -m unittest tests.test_desktop_config -v`: passed and verifies both
  macOS permission metadata and the Linux WebKit user-media permission handler
  wiring.
- macOS permission metadata fix: `desktop/src-tauri/Info.plist` now declares
  `NSMicrophoneUsageDescription` and `NSAppleEventsUsageDescription`, and
  `desktop/src-tauri/tauri.conf.json` includes it through
  `bundle.macOS.infoPlist`.
- `uv run -m unittest tests.test_desktop_config -v`: passed and verifies the
  macOS permission plist is configured.
- `cd desktop && npm run tauri -- info`: passed and reported the expected
  Tauri/WebKitGTK environment.
- `cargo check --manifest-path desktop/src-tauri/Cargo.toml`: passed after the WebKit user-media permission and notification fixes.
- `cd desktop/src-tauri && cargo fmt --check`: passed.
- `cd desktop/src-tauri && cargo check`: passed.
- `timeout --kill-after=5s 90s npm run tauri dev`: compiled the Tauri dev binary and launched `target/debug/desktop` with the hidden secondary window/status-shell config before the timeout stopped it; no child process remained.
- `uv run scripts/linux_tray_smoke.py --timeout 90`: passed on the current Ubuntu GNOME Wayland session with `ubuntu-appindicators@ubuntu.com` enabled and a StatusNotifier host registered. It launched the Tauri shell against a fake localhost health service, observed a new DBus item `.../org/ayatana/NotificationItem/tray_icon_tray_app_...`, verified DBusMenu labels including `Status: Ready`, `Cleanup`, direct recording actions, latest transcript actions, `Recent transcripts`, `Start service`, `Show status window`, `Open keyword glossary`, `Open settings`, and `Quit`, then clicked `Record` and `Stop` through DBusMenu and observed fake service status transition `recording` -> `ready`. It wrote `/tmp/granite-speach-tauri-tray-smoke.log`, and no npm/Tauri/vite process remained afterward.
- `arecord -l` and `wpctl status`: current Linux host exposes capture devices `HD-Audio Generic ALC245 Analog`, `acp-pdm-mach DMIC capture`, `Headphones Stereo Microphone`, and default `Digital Microphone`. `timeout --kill-after=2s 5s arecord -D default -f S16_LE -r 16000 -c 1 -d 1 /tmp/granite-speach-arecord-smoke.wav` produced a valid `16000 Hz`, mono, `1.000000s` WAV. This verifies OS-level microphone capture, not Tauri/WebAudio permission.
- TransClip reference check: cloned `https://github.com/paulbrav/TransClip` to `/tmp/TransClip` and inspected `transclip/app.py`, `transclip/hotkey.py`, `transclip/clipboard.py`, and `README.md`. Relevant behavior mirrored in this app: status icon primary UI, menu actions, cleanup toggle, recent transcripts, hold-to-record hotkey, clipboard copy/paste, notifications/status, settings/config actions, and quit.
- CLI transcribe smoke test with `asr_backend = "file:..."`: passed.
- HTTP service endpoint test for `/health`, `/cleanup`, `/transcribe`, `/cleanup/transcribe`, `/record/start`, `/record/stop`, short-recording discard, and endpoint error handling: passed.
- Debug capture smoke test: produced `audio.wav`, `raw_asr.txt`, `cleaned.txt`, `timings.json`, `model_versions.json`, plus `error.log` and `error.json` for an HTTP failure.
- Eval smoke test: produced WER, keyword preservation, release-to-ready latency, and threshold counts.
- `doctor` now reports Torch runtime status and whether `auto` will use GPU or CPU.
- Local media search found assorted temporary WAV/MP3/video artifacts, but no curated 20-30 short dictation eval manifest with references.

## Remaining Unverified Items

- See `docs/v1-todo.md` for the actionable TODO list.
- Real Granite ASR and Gemma cleanup quality/latency have been smoke-tested with the cached models, but not benchmarked across a representative eval clip set because no local eval clips were provided in the repo. The low-latency Granite NAR plus rule-cleanup profile meets the V1 target on a 3s smoke clip, but representative 20-30 clip eval is still unverified.
- Live desktop startup, Linux tray registration/menu labels/actions, browser-side fake-microphone WebAudio capture, visible Tauri/WebKit microphone capture, service-side Linux microphone capture, OS-level Linux microphone capture, Linux WebKit user-media permission handling, and Linux Tauri paste injection have been smoke-tested or compile-checked. Full live interaction still needs physical/global-hotkey validation on the target Linux desktop, because synthetic `ydotool` input did not trigger the Tauri global-shortcut handler on this GNOME Wayland session, and macOS validation because menu bar behavior, Accessibility permission, and paste injection vary by platform.

## Commands Needed To Close The Remaining Verification Gap

On Ubuntu/Debian-like Linux, install Tauri prerequisites if needed:

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

`wtype` being installed is not sufficient on every Wayland compositor. The
current GNOME Wayland session rejects the virtual keyboard protocol, so this
host validates automatic paste through `ydotool`. A normal system `ydotool`
install, or an X11/XWayland session with `xdotool`, is still the preferred
target-machine setup.

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

Then compare glossary on/off behavior:

```bash
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
VIRTUAL_ENV=$PWD/.venv-gfx1151 uv run --active scripts/run_keyword_ablation.py \
  eval/real-usage/manifest.json
```

For Linux `gfx1151`, use the TheRock ROCm runtime identified from the Journey
notes instead of the local custom wheel:

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
VIRTUAL_ENV=$PWD/.venv-gfx1151 uv run --active -m granite_speach.cli doctor
```
