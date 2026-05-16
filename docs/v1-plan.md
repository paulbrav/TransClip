# V1 Plan: Local Push-To-Talk Transcription

## Goal

Build a personal, local-only, cross-platform dictation tool for this Linux workstation and an Apple Silicon MacBook Pro.

The app primarily lives as a Linux panel app indicator and macOS menu bar status item. The user can click the icon to open a compact menu for status, settings, cleanup toggle, latest transcript actions, and quit.

The primary workflow is icon + hotkey + menu: the user holds a hotkey, speaks, releases the hotkey, and the transcript is inserted at the current cursor position by temporarily using the clipboard and paste shortcut. V1 is optimized for short technical notes and programming-related messaging.

## Terminology

Use platform-native status UI terms:

- Linux: panel, system tray, app indicator, StatusNotifier/AppIndicator.
- macOS: menu bar, status item, NSStatusItem-style behavior.

Do not frame the app as a Dock-first or persistent-window desktop app. Normal operation is status icon + hotkey + click menu.

## Locked Product Decisions

- Scope: personal tool first, not public distribution.
- Privacy: local-only is mandatory. No cloud ASR, no cloud cleanup, no transcript/audio telemetry.
- Language: English only in v1.
- Primary workflow: hold-to-record only.
- Primary home: Linux panel/system tray/app indicator and macOS menu bar/status item.
- Normal operation is icon + hotkey + menu, not a persistent main window.
- Hotkeys:
  - Linux default: `Ctrl+Space`
  - macOS default: `Option+Space`
  - Must be configurable in a settings file.
- Latency targets:
  - Under `700 ms` from hotkey release to inserted text for short dictation under 10 seconds.
  - Under `1.5 s` acceptable for longer snippets.
  - Optional partial/pre-decode while the key is held.
- Snippet length:
  - Main path optimized for under 10 seconds.
  - V1 should tolerate up to about 60 seconds.
  - Several-minute dictation is out of scope for v1.
- Models stay loaded while the app is running.
- No autostart in v1.
- No audio/transcript persistence by default.
- Optional explicit debug capture mode may write local eval artifacts.

## UX

Primary UI:

- A persistent status icon in the Linux panel/system tray/app indicator area or macOS menu bar/status item area.
- Clicking the icon opens a menu.
- The app does not require a persistent main window for normal use.
- A settings/debug window may exist, but it is secondary.

Menu contents:

- Current status: `Loading`, `Ready`, `Recording`, `Transcribing`, `Cleaning`, `Pasting`, `Error`.
- Cleanup on/off toggle.
- Paste latest transcript.
- Copy latest transcript.
- Open keyword glossary.
- Open settings.
- Quit.

On hotkey press:

1. Start microphone capture.
2. Update status to `Recording`.
3. Optionally start chunk/pre-decode.

On hotkey release:

1. Stop capture.
2. Finalize ASR.
3. Run faithful cleanup by default.
4. Temporarily write result to clipboard.
5. Simulate paste into the focused app.
6. Restore previous clipboard content when possible.

If paste fails or no text field is focused:

- Leave the transcript on the clipboard.
- Show a notification.
- Keep transcript available as `latest transcript` in the panel/menu bar menu.

## Insertion Semantics

V1 uses clipboard paste first.

Algorithm:

1. Read current clipboard content.
2. Write transcript to clipboard.
3. Simulate paste hotkey:
   - Linux: `Ctrl+V`
   - macOS: `Command+V`
4. Wait briefly.
5. Restore prior clipboard content if it still appears safe to do so.

Known risk: clipboard preservation is race-prone. If the user copies something during the restore window, the app could overwrite it. For v1, reduce risk with a short restore delay and an explicit setting such as:

```toml
restore_clipboard_after_paste = true
clipboard_restore_delay_ms = 500
```

Direct typing is out of scope for v1 unless clipboard paste proves unusable.

## Architecture

Use a two-process architecture:

```text
Tauri tray/menu-bar shell
  -> status icon + click menu
  -> global hold-to-record hotkey
  -> clipboard/paste injection
  -> localhost Python inference service
      -> ASR backend
      -> cleanup backend
```

Recommended split:

- Desktop shell: thin Tauri 2 / Rust status shell.
- Inference service: Python first.
- Cleanup runtime: prefer `llama.cpp`/GGUF or MLX where practical, exposed through the local inference service.

Reasoning:

- Tauri keeps the status icon, click menu, hotkeys, notifications, clipboard, and packaging separate from model runtime churn.
- Python keeps Granite/ROCm/MLX experimentation fast.
- Sidecar inference can be changed without rewriting the desktop shell.

## ASR Candidates

Primary candidates:

- `ibm-granite/granite-speech-4.1-2b`
  - Best default accuracy candidate.
  - Supports punctuation/capitalization.
  - Supports keyword-biased recognition.
  - Source: https://huggingface.co/ibm-granite/granite-speech-4.1-2b
- `ibm-granite/granite-speech-4.1-2b-nar`
  - Lower-latency candidate.
  - Non-autoregressive ASR by transcript editing.
  - IBM positions it for latency-sensitive applications, with lower accuracy than the autoregressive model.
  - Source: https://huggingface.co/ibm-granite/granite-speech-4.1-2b-nar
- `ibm-granite/granite-speech-4.1-2b-plus`
  - Use only if timestamps, speaker attribution, or incremental decoding prove valuable.
  - Caveat: ASR-only mode does not provide punctuation/capitalization like the base model.
  - Source: https://huggingface.co/ibm-granite/granite-speech-4.1-2b-plus

Older fallback:

- `ibm-granite/granite-4.0-1b-speech`
  - Useful as a compatibility/control option, especially because MLX and ONNX conversions exist.
  - Source: https://huggingface.co/ibm-granite/granite-4.0-1b-speech

## Cleanup Candidates

Default cleanup should be faithful, not creative.

Primary candidates:

- `google/gemma-4-E2B-it`
  - First cleanup candidate.
  - Small, instruction-tuned, Apache 2.0, designed for local execution.
  - Available through Transformers and local runtimes such as `llama.cpp`/GGUF.
  - Source: https://huggingface.co/google/gemma-4-E2B-it
- `google/gemma-4-E4B-it`
  - Second cleanup candidate if E2B is not reliable enough.
  - Higher quality expected, higher latency/resource cost.
- `Qwen/Qwen3.5-2B`
  - Benchmark/control candidate, not default.
  - Source: https://huggingface.co/Qwen/Qwen3.5-2B

Cleanup must prefer semantic fidelity over prettier writing.

Allowed by default:

- Punctuation.
- Capitalization.
- Conservative paragraphing.
- Conservative correction of obvious ASR formatting issues.

Not allowed by default:

- Adding facts.
- Reordering meaning.
- Rewriting tone.
- Replacing technical identifiers with more common words.
- Removing words unless explicitly configured.

Separate future modes may support email rewrite, bullet formatting, or note formatting, but those must be explicit user actions.

## Keyword Glossary

V1 includes a simple keyword glossary from day one.

Implementation:

- Store as a plain editable config file, for example `keywords.txt`.
- Include terms in the Granite ASR keyword-bias prompt.
- Include terms in cleanup instructions as terms to preserve exactly when present.

Initial example terms:

```text
PyTorch
ROCm
gfx1151
Tauri
llama.cpp
Gemma
Granite
Qwen
Transformers
Hugging Face
MLX
Wayland
```

Keyword biasing is likely higher leverage than a larger cleanup model for programming-related dictation.

## Settings Shape

Use a simple file-backed config in v1, for example `settings.toml`.

Candidate fields:

```toml
hotkey_linux = "Ctrl+Space"
hotkey_macos = "Option+Space"
language = "en"

asr_model = "ibm-granite/granite-speech-4.1-2b"
cleanup_model = "google/gemma-4-E2B-it"
cleanup_enabled = true

restore_clipboard_after_paste = true
clipboard_restore_delay_ms = 500

max_recording_seconds = 60
min_recording_ms = 250

debug_capture = false
debug_capture_dir = "debug-captures"
```

## Debug Capture

Default: store nothing persistently.

When `debug_capture = true`, write local-only artifacts:

- `audio.wav`
- `raw_asr.txt`
- `cleaned.txt`
- `timings.json`
- `model_versions.json`
- error logs

This is for benchmarking and quality improvement only.

## Eval Plan

Before investing heavily in status-shell polish, build a latency/quality harness.

Eval data:

- 20 to 30 short clips from real usage.
- Focus on technical notes and programming-related messaging.
- Include names, acronyms, library names, file names, model names, numbers, and punctuation-heavy statements.
- Include a few noisy/fast/casual utterances.

Metrics:

- Capture duration.
- ASR latency.
- Cleanup latency.
- End-to-end release-to-ready latency.
- Word error rate where reference text exists.
- Keyword preservation.
- Semantic-drift failures from cleanup.
- Paste success/failure separately from model latency.

Pass criteria:

- Under `700 ms` release-to-ready for common short snippets after warm-up.
- Under `1.5 s` for longer v1 snippets.
- Cleanup does not alter technical meaning.
- Keyword glossary measurably improves technical term recognition.

## Platform Mapping

Linux target:

- Current machine: AMD Ryzen AI Max+ Pro 395 with Radeon 8060S, `gfx1151`, 125 GiB RAM.
- ROCm has documented `gfx1151` support in recent releases, but the stack should still be treated as a risk area.
- Use StatusNotifier/AppIndicator behavior where available.
- Support common Linux desktop panels as best effort.
- Wayland global shortcuts and paste injection remain compositor-dependent.
- Source: https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.0.2/docs/compatibility/compatibilityryz/native_linux/native_linux_compatibility.html

macOS target:

- Apple Silicon MacBook Pro.
- Prefer MLX or `llama.cpp`/Metal for cleanup.
- ASR runtime needs benchmarking; do not assume PyTorch/MPS, MLX, and Transformers behave equally for Granite.
- Use NSStatusItem-style menu bar behavior.
- The app may hide or minimize any main window.
- Accessibility permission may be required for paste simulation.

Desktop integration:

- Tauri global shortcut and clipboard plugins cover the basic cross-platform shell.
- Electron remains a fallback if Tauri blocks hotkey/tray/paste behavior.
- Sources:
  - https://tauri.app/ko/plugin/global-shortcut/
  - https://tauri.ubitools.com/plugin/clipboard/
  - https://www.electronjs.org/docs/latest/api/global-shortcut/

Known desktop risks:

- Wayland global hotkeys and paste injection vary by compositor.
- macOS paste simulation may require Accessibility permission.
- Clipboard restore can race with user clipboard actions.

## Implementation Milestones

1. Build local ASR harness.
   - Record or load WAV.
   - Run Granite ASR candidates.
   - Emit raw transcript and timings.

2. Add keyword glossary to ASR prompt.
   - Compare glossary on/off.
   - Track keyword preservation.

3. Add cleanup harness.
   - Run Gemma 4 E2B cleanup.
   - Compare raw vs cleaned output.
   - Track latency and semantic drift.

4. Build local inference service.
   - Expose `health`, `transcribe`, and `cleanup/transcribe` operations.
   - Keep models warm.

5. Build minimal Linux panel / macOS menu bar shell.
   - Status icon.
   - Click menu.
   - Cleanup toggle.
   - Latest transcript actions.
   - Settings/glossary open actions.
   - Quit.
   - Optional secondary window only for settings/debug/status.
   - Global hold-to-record hotkey.
   - Microphone capture.

6. Add paste injection.
   - Clipboard write.
   - Paste hotkey.
   - Clipboard restore.
   - Latest transcript fallback.

7. End-to-end benchmark.
   - Linux first.
   - Then macOS.
   - Decide final v1 ASR and cleanup model defaults from measurements.

## Non-Goals For V1

- Public installer/distribution.
- A full Dock-first desktop app.
- A persistent main window as the primary UI.
- Cloud fallback.
- Telemetry.
- Full transcript history UI.
- Several-minute dictation workflow.
- Direct typing insertion.
- Rewrite modes beyond faithful cleanup.
- Multilingual dictation.
