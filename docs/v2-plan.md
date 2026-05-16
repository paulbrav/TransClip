# V2 Plan: Bundled Local Dictation App

## Goal

Turn the v1 personal prototype into a more integrated local desktop app that is easier to launch, update, move between machines, and recover when model/runtime dependencies change.

V2 keeps the same core product promise:

- Local-only.
- Push-to-talk dictation.
- Fast insertion at the cursor.
- Faithful cleanup by default.
- Strong support for technical notes and programming-related messaging.

The main change is operational: v2 should feel like one app instead of a tray shell plus a separately managed model environment.

## Promotion Criteria From V1

Do not start v2 packaging until v1 has measured answers for:

- Best ASR model for quality/latency on Linux `gfx1151`.
- Best ASR model for quality/latency on Apple Silicon.
- Best cleanup model/runtime.
- Whether partial/pre-decode materially improves perceived latency.
- Whether clipboard paste is reliable enough on the target desktops.
- Whether keyword biasing meaningfully improves technical terms.

Minimum bar:

- Common short snippets meet the `700 ms` release-to-ready target after warm-up.
- Longer v1 snippets stay under `1.5 s` often enough to feel usable.
- Cleanup semantic drift is rare and understood.
- The app can run for a normal work session without model/runtime crashes.

## Product Scope

V2 remains personal-first, but should be structured like a product:

- One primary app entry point.
- Bundled sidecars/runtimes.
- First-run setup.
- Health checks.
- Model/version management.
- Permission onboarding.
- Better diagnostics.
- Optional autostart.

Still out of scope unless explicitly revisited:

- Cloud fallback.
- Telemetry.
- Public support matrix.
- App-store distribution.
- Multi-user/team features.
- Windows support.

## Locked V2 Decisions

- Target platforms: Linux and macOS only.
- Packaging shape: one app bundle, not one literal executable.
- Models: external app-managed model cache, not embedded in the app package.
- Network: model downloads and updates are explicit user actions only.
- macOS distribution: personal-use build; notarization is not a v2 blocker.
- Autostart: supported, off by default.
- Transcript history: local file history may exist, off by default.
- Overlay: tiny optional recording/status overlay, off by default.
- Insertion: clipboard-based only; no direct typing fallback.
- Long dictation: support up to about 10 minutes.
- Long dictation gesture: same push-to-talk hold gesture.
- Long dictation ASR: chunk/transcribe while held.
- Long dictation UI: do not show partial transcript.
- Long dictation cleanup: run once at the end on the stitched transcript.
- Cleanup failure: no automatic raw fallback and no normal manual raw recovery action.
- Cleanup failure state: keep raw stitched transcript in memory until overwritten.
- Cleanup profile selection: global current mode, with `faithful` as the default.

## Packaging Target

Preferred user experience:

```text
Launch one app
  -> tray/menu appears
  -> bundled local inference manager starts
  -> models warm in background
  -> status becomes Ready
```

The app may still use multiple internal processes, but the user should not manage them manually.

V2 should bundle:

- Tauri desktop app.
- Inference manager sidecar.
- ASR runtime environment or executable.
- Cleanup runtime executable.
- Default config templates.
- Model manifest.
- Diagnostics tools.

Models live in an external app-managed cache. The app bundle ships a pinned model manifest and setup tooling, but the weights are downloaded or imported into the cache through explicit user actions.

For personal use, a large local model cache is acceptable. Avoid making the app bundle itself huge unless a specific runtime requires a bundled asset.

## Architecture

Recommended v2 architecture:

```text
Tauri app
  -> Process supervisor
      -> ASR worker
      -> Cleanup worker
      -> Model manager
  -> Audio capture
  -> Clipboard/paste insertion
  -> Settings and diagnostics UI
```

Key shift from v1:

- V1: Python service is manually managed and easy to change.
- V2: app supervises sidecars and owns lifecycle.

The workers should still be separate processes where practical. One physical binary is not worth losing isolation if a GPU runtime can crash the process.

## One Bundled Binary vs One Bundled App

There are two possible interpretations of "one binary":

1. Literal single executable containing UI, inference, runtimes, and model assets.
2. Single app bundle that contains several internal executables and assets.

Recommended answer: prefer **one app bundle**, not a literal single executable.

Reasoning:

- ASR and LLM runtimes are heavy and platform-specific.
- ROCm, MLX, Metal, `llama.cpp`, and Python have different packaging needs.
- Separate workers let the app restart a failed model runtime without killing the tray UI.
- macOS `.app`/`.dmg` and Linux AppImage/deb packages already support bundled resources.

V2 should feel like one app to the user while preserving internal process boundaries.

## Target Platforms

V2 targets:

- Linux on the current AMD Ryzen AI Max+ Pro 395 / Radeon 8060S / `gfx1151` workstation.
- macOS on Apple Silicon.

V2 does not target Windows. Do not add DirectML, Windows installer, Windows hotkey, or Windows paste support unless the target-machine list changes.

macOS packaging is personal-use first:

- Local `.app` build is sufficient initially.
- Ad hoc signing is acceptable if needed.
- Developer ID signing or notarization should be added only if local macOS permission/quarantine behavior makes it necessary.

## Runtime Strategy

### Cleanup Runtime

Prefer moving cleanup to a non-Python sidecar:

- `llama.cpp` server or embedded library with GGUF.
- MLX on macOS if it is clearly faster or more reliable.
- ONNX Runtime if Gemma 4 ONNX proves simple and fast.

Default target:

- `gemma-4-E2B-it` quantized.
- Upgrade path to `gemma-4-E4B-it`.

The cleanup API should remain stable:

```text
clean(raw_transcript, glossary, mode) -> cleaned_transcript
```

### ASR Runtime

ASR is the harder part. Keep options open until v1 benchmarks are complete.

Default v2 packaging strategy:

- Keep ASR behind a worker API.
- Package the v1 benchmark winner rather than forcing a native runtime too early.
- Allow Linux and macOS to use different internal implementations behind the same API.

Candidate v2 ASR runtime paths:

- Python + Transformers sidecar, packaged with `uv` and a managed local environment.
- ONNX Runtime if a high-quality Granite conversion is available and benchmarked.
- MLX path for Apple Silicon if Granite support becomes practical.
- Separate Linux ROCm and macOS Metal/MPS implementations behind the same API.

The ASR API should hide runtime differences:

```text
transcribe(audio_pcm, sample_rate, glossary, mode) -> raw_transcript + metadata
```

Metadata should include:

- model id
- runtime
- model version/hash
- audio duration
- ASR latency
- confidence-like fields if available
- keyword-bias settings

## Model Management

V2 should introduce a local model manifest.

Example:

```toml
[[models]]
id = "granite-speech-4.1-2b"
role = "asr"
source = "huggingface"
repo = "ibm-granite/granite-speech-4.1-2b"
revision = "pinned_revision_here"
runtime = "transformers-rocm"
required = true

[[models]]
id = "gemma-4-e2b-it-gguf"
role = "cleanup"
source = "huggingface"
repo = "ggml-org/gemma-4-E2B-it-GGUF"
file = "model-q4_k_m.gguf"
revision = "pinned_revision_here"
runtime = "llama.cpp"
required = true
```

Model manager responsibilities:

- Verify files exist.
- Verify checksums.
- Show model status.
- Support redownload.
- Support switching between benchmarked models.
- Keep model cache outside the app bundle.
- Support local import/export of the model cache for offline setup.
- Never download or update models during normal dictation.

Recommended local paths:

- Linux: XDG data/cache dirs.
- macOS: `~/Library/Application Support/<AppName>` and `~/Library/Caches/<AppName>`.

Model downloads are explicit only:

- First-run setup may offer `Download required models`.
- Settings may offer `Repair model` or `Update model`.
- Normal operation must not make network calls.

## First-Run Setup

V2 should add a first-run flow, even for personal use:

1. Explain local-only operation.
2. Check microphone permission.
3. Check Accessibility permission where needed for paste simulation.
4. Check hotkey registration.
5. Check model availability.
6. Warm models.
7. Run a short test recording.
8. Show measured latency.

No marketing screen. This is setup and diagnostics.

## Permissions

Linux:

- Microphone access depends on PipeWire/PulseAudio environment.
- Global shortcuts can be Wayland/compositor-dependent.
- Paste simulation may differ under X11 vs Wayland.
- V2 should detect session type and show a concrete warning when features are degraded.

macOS:

- Microphone permission required.
- Accessibility permission likely required for paste simulation.
- The app should detect missing permissions and open the relevant settings page when possible.

## Insertion Improvements

Keep clipboard paste as default.

Add v2 options:

- Clipboard paste with restore.
- Clipboard paste without restore.
- Copy-only mode.
- "Paste latest transcript" action.

Do not implement direct typing in v2. Clipboard behavior is the only insertion path.

V2 should add paste result tracking:

- last paste attempted
- whether clipboard was restored
- focused app name if available
- error reason if known

Do not log transcript content unless debug capture is enabled.

## UI Additions

V2 can add a compact settings window.

Suggested views:

- Status
  - loaded models
  - runtime health
  - current hotkey
  - last latency
- Models
  - ASR model selector
  - cleanup model selector
  - model health/download status
- Glossary
  - editable keyword list
  - import/export
- Insertion
  - paste mode
  - clipboard restore delay
- Debug
  - enable capture
  - open logs
  - run benchmark

Optional v2 UI additions:

- Local transcript history settings.
- Clear local history action.
- Tiny overlay settings.

Transcript history is off by default. If enabled, it is a bounded local file, stores transcript text only, and has a clear-history action. Audio is never included in history.

The optional overlay is off by default. When enabled, it shows recording state, elapsed time, and transcribing/cleaning/pasting state. It should not show partial transcript text.

## Profiles And Modes

V2 can introduce named modes while keeping faithful cleanup as default.

Initial profiles:

- `raw`: ASR output only.
- `faithful`: punctuation, capitalization, conservative paragraphing.
- `message`: concise chat/message cleanup without changing meaning.
- `notes`: lightly structured technical notes.
- `bullets`: convert into bullet points, explicit non-default mode.

Each profile should have an eval set. Do not ship a profile that is only prompt-vibes.

Profile selection is global:

- The app has one current cleanup profile.
- The current profile is visible in tray/settings.
- Fast profile switching may be added through tray actions or hotkeys.
- Per-recording profile prompts are out of scope.

## Streaming And Partial Decode

If v1 proves pre-decode is valuable, v2 should formalize chunked ASR:

```text
hotkey down
  -> stream audio chunks to ASR worker
  -> produce internal chunk transcripts
hotkey up
  -> stitch chunk transcripts
  -> cleanup final transcript
  -> paste
```

Partial text should not be inserted into external apps and should not be shown in the overlay. Chunking is an internal latency/reliability mechanism.

Risks:

- Chunk boundary errors.
- Repeated words.
- Cleanup model waiting for final context.
- Increased GPU contention if ASR and cleanup overlap poorly.

## Long Dictation

V2 may support long dictation up to about 10 minutes.

Long dictation keeps the same push-to-talk hold gesture:

- Hold the hotkey to record.
- Release the hotkey to finish, clean up, and paste.
- Auto-stop at the configured maximum duration.
- Support a cancel gesture, likely `Esc` while recording.

Implementation:

- Stream audio chunks while recording.
- Run ASR on chunks while the key is held.
- Stitch raw chunk transcripts after release.
- Run cleanup once at the end on the stitched transcript.
- Paste/copy only after final cleanup.

User-visible behavior:

- Optional overlay may show recording state and elapsed time.
- Do not show partial chunk transcripts.
- Do not paste partial results.

Long dictation does not need to meet the short-snippet `700 ms` release-to-paste target.

Risk:

- A final cleanup pass over a 10-minute transcript may be slow or may exceed the cleanup model context window. Benchmark this separately before relying on it.

## Cleanup Failure Behavior

V2 deliberately does not automatically fall back to raw ASR when cleanup fails.

If ASR succeeds but cleanup fails:

- Show an error state.
- Do not paste raw ASR automatically.
- Do not expose a normal `Copy raw transcript` or `Paste raw transcript` recovery action.
- Keep the raw stitched transcript in memory until the next recording or app quit.

This is intentionally strict, but it carries data-loss risk for long dictation. Revisit only if real use shows failures are costly.

## Diagnostics

V2 should provide a local diagnostics bundle that excludes transcript/audio by default.

Include:

- OS and desktop session info.
- GPU/runtime detection.
- model ids and revisions.
- config, with sensitive paths redacted if needed.
- timing summaries.
- recent errors.
- permission state.

Optional debug capture can include audio/transcripts only when explicitly enabled.

Debug/eval clips remain developer-managed files, not a normal app history feature. The app may provide `Run benchmark` and `Open debug folder`, but it should not make the eval corpus part of the everyday UI.

## Updates

For personal use, avoid automatic binary/model updates at first.

Recommended v2 behavior:

- App version pinned.
- Model manifest pinned.
- Manual "check model status" action.
- Manual "update models" action.
- No automatic binary updates.
- No automatic model downloads.

Reasoning:

- A silent model update can alter transcription behavior.
- Reproducibility matters for debugging latency and cleanup drift.

## Testing

V2 needs tests around the app boundaries, not just model quality.

Test areas:

- Config parsing and migration.
- Keyword glossary loading.
- Process supervisor start/stop/restart.
- ASR API contract with fake worker.
- Cleanup API contract with fake worker.
- Clipboard restore behavior with mocks.
- Hotkey lifecycle state machine.
- Debug capture disabled by default.
- Model manifest validation.

Manual acceptance tests:

- Linux X11 if available.
- Linux Wayland on the actual desktop session.
- macOS Apple Silicon.
- Paste into browser, terminal, editor, chat app, and document editor.
- Permission-denied flows.
- Model missing/corrupt flows.
- Long push-to-talk recording up to the configured 10-minute limit.
- Cleanup failure without raw fallback.

## Security And Privacy

V2 must keep the v1 privacy posture:

- No network calls during normal operation after models are installed.
- No telemetry.
- No crash reports containing transcripts.
- No transcript history unless explicitly enabled.
- If transcript history is enabled, it is a bounded local file and can be cleared from settings.
- Debug capture is explicit and visibly indicated.

The app should make network access easy to audit:

- Model downloads only from a manual setup/update action.
- Runtime path does not opportunistically call remote APIs.

## V2 Implementation Milestones

1. Convert v1 service into supervised sidecar.
   - App starts/stops service.
   - Health endpoint.
   - Restart on crash.

2. Add model manifest and local model manager.
   - Verify model files.
   - Pin versions.
   - Surface status in tray/settings.

3. Package cleanup runtime.
   - Start with `llama.cpp`/GGUF for Gemma 4 E2B.
   - Keep API stable.

4. Package ASR runtime.
   - Choose from v1 benchmark winner.
   - Linux and macOS may use different runtime implementations.

5. Add first-run setup and permission checks.
   - Microphone.
   - Hotkey.
   - Paste/accessibility.
   - Model warm-up.

6. Add settings window.
   - Models.
   - Hotkeys.
   - Glossary.
   - Insertion.
   - Debug.

7. Add benchmark runner.
   - Reuse v1 eval clips.
   - Store local timing and quality reports.

8. Add packaging.
   - macOS `.app`/`.dmg`.
   - Linux AppImage or deb first.
   - Keep model cache external.
   - Do not add Windows packaging.

9. Add optional autostart.
   - Disabled by default.
   - User controlled.

10. Harden failure modes.
   - Missing model.
   - Worker crash.
   - Permission denied.
   - Hotkey conflict.
   - Paste failure.
   - Cleanup failure without raw fallback.
   - Long dictation max-duration auto-stop.

## Resolved Defaults

- Models use an external app-managed cache.
- Downloads and updates are explicit only.
- ASR stays behind a worker API; v2 packages the v1 benchmark winner instead of forcing a runtime rewrite.
- Cleanup uses `llama.cpp`/GGUF first unless benchmarks prove another runtime is better.
- Clipboard paste is sufficient for v2; direct typing is not implemented.
- Overlay is status-only, optional, and off by default.
- Debug/eval clips stay developer-managed, with app actions to run benchmarks/open folders.
- Cross-platform means Linux plus macOS for this project.
- Cleanup profile is a global current mode.
- Long dictation uses the same push-to-talk gesture, chunks internally, and cleans once at the end.

## Non-Goals For V2

- Cloud fallback.
- Public app-store release.
- Windows support.
- Team/shared glossary sync.
- Server deployment.
- Mobile app.
- Browser extension.
- Unbounded transcript history.
- Silent model updates.
- Direct typing insertion.
