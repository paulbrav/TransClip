# Windows roadmap

Deferred Windows enhancements that are larger than the incremental hardening
already landed. Each entry states why it is not done yet and what adopting it
would take.

## W5 — On-device Windows Speech Recognition as an ASR backend

**Status:** roadmap (not started). **Target:** Windows 11; **primary value:** a
zero-download ASR path on Windows.

### What it is

At Build 2026 Microsoft announced a **Speech Recognition API** (public preview)
as part of the Windows AI APIs / Windows AI Foundry: real-time or batch,
**on-device** speech-to-text from live audio, shipping in-box on capable
Windows 11 devices. See the
[Build 2026 Windows developer recap](https://blogs.windows.com/windowsdeveloper/2026/06/02/build-2026-furthering-windows-as-the-trusted-platform-for-development/)
and the [Windows notifications/AI overview on Microsoft Learn](https://learn.microsoft.com/en-us/windows/apps/develop/).

### Why it is on-strategy for TransClip

TransClip's default Windows backend is Granite AR, which needs a CUDA PyTorch
wheel and a multi-GB model download (`transclip models prefetch …`). An in-box
Windows speech API would give:

- **No model download and no GPU requirement** — works on any capable Win11 box.
- A natural **lightweight/fallback `asr_backend`** alongside `granite`,
  `granite_nar`, `mlx`, and the `file:` test backend.

### Why it is deferred (not "cheap/easy")

- It is a **new ASR backend**, not a one-call hygiene fix: it must implement the
  same engine contract the existing backends use (see `transclip/asr.py` and how
  `asr_backend` is dispatched) and pass the eval harness.
- The API is **public preview** and **Windows 11 only**, so it cannot be the
  default; it would be opt-in via `asr_backend = "windows_speech"`.
- The exact projection surface must be confirmed at implementation time. Two
  candidates, both reachable from Python via [pywinrt](https://github.com/pywinrt/pywinrt)
  `winrt-*` packages (the same mechanism W4 uses for toasts):
  - the **new** Build 2026 Windows AI speech API (preview), or
  - the existing **`Windows.Media.SpeechRecognition`** namespace as an interim
    on-device option.

### What adopting it would take

1. Add a `windows_speech` backend implementing the engine contract; keep it
   behind a capability check that reports unavailable off Windows 11 / when the
   projection is missing (mirror `notifications.windows_toast`'s best-effort
   degradation).
2. Add the required `winrt-*` namespace packages to the `windows-ui` (or a new
   `windows-speech`) extra with a `sys_platform == 'win32'` marker.
3. Wire it into `doctor` (a readiness check) and document it in the README
   Windows section.
4. Run the Windows eval (`eval/windows/manifest.json`) to set/verify accuracy
   and latency gates against Granite AR.
