# V1 Live Validation

Use this checklist on the target Linux desktop and macOS machine before calling
V1 complete. Automated tests cover service, model, eval, and Linux tray
registration, but these workflows depend on desktop permissions and compositor
behavior.

## Linux

Preflight:

```bash
uv run -m granite_speach.cli doctor
VIRTUAL_ENV=$PWD/.venv-gfx1151 uv run --active scripts/check_v1_completion.py
uv run scripts/linux_tray_smoke.py
cd desktop && npm run test:recorder
cd desktop && npm run test:tauri-recorder
cd desktop && npm run test:tauri-service-record
cd desktop && npm run test:tauri-paste
arecord -D default -f S16_LE -r 16000 -c 1 -d 1 /tmp/granite-speach-arecord-smoke.wav
```

Run the service and shell:

```bash
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
VIRTUAL_ENV=$PWD/.venv-gfx1151 uv run --active -m granite_speach.cli serve
```

```bash
cd desktop
npm run tauri dev
```

Alternatively, start only the Tauri shell and use `Start service` from the
status menu or secondary status/debug window. Set `GRANITE_SPEACH_UV` first if the shell should use a specific `uv` binary.

Interactive checks:

- Status icon appears in the panel/app-indicator area.
- Left-clicking the icon opens the menu.
- Menu shows current status, cleanup toggle, latest transcript actions,
  direct Record/Stop actions, service start action, settings/glossary actions,
  status window action, and quit.
- Record from the status menu starts recording and changes the menu status to
  Recording.
- Stop from the status menu transcribes and updates latest transcript.
- Stop + Paste from the status menu inserts into a focused text field.
- Start service from the status menu works when no service is running.
- Start service does not launch a duplicate service when port `8765` is already
  listening.
- Show status window opens the secondary window without making it the primary
  app home.
- Manual Record asks for microphone permission if needed.
- Manual Stop transcribes and updates latest transcript.
- Manual Stop + Paste inserts into a focused text field.
- Holding and releasing the configured global hotkey records, transcribes, and
  pastes.
- Failed paste leaves the transcript available on the clipboard/latest menu.
- If `npm run test:tauri-paste` fails with `Compositor does not support the
  virtual keyboard protocol`, this Linux compositor does not permit `wtype`
  paste injection. The app can still leave text on the clipboard, but automatic
  insertion needs a compositor/tooling path that supports synthetic paste, such
  as working `wtype`, configured `ydotool`, or `xdotool` on X11/XWayland.
- Clipboard restoration does not overwrite a clipboard change made during the
  restore delay.
- Quit exits the tray app and leaves no Tauri/dev process running.

## macOS

Run the same service and Tauri shell on the Mac target, then check:

- Status item appears in the macOS menu bar.
- Clicking the status item opens the menu.
- The app does not require a persistent main window for normal operation.
- Microphone permission prompt and grant path work.
- Accessibility permission prompt/grant path works for paste simulation.
- `Option+Space` hold-to-record records, transcribes, and pastes.
- Copy/paste latest transcript actions work.
- Quit exits cleanly.

## Real-Usage Eval

Record 20 to 30 measured short WAV clips with matching `.txt` references. Use
one additional warmup clip if desired.

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

To compare glossary on/off behavior for the same clips:

```bash
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
VIRTUAL_ENV=$PWD/.venv-gfx1151 uv run --active scripts/run_keyword_ablation.py \
  eval/real-usage/manifest.json
```

Pass criteria:

- Common short snippets are under `700 ms` release-to-ready after warm-up.
- Longer V1 snippets are under `1.5 s`.
- Cleanup does not alter technical meaning.
- Keyword preservation is high enough for technical dictation.
- `scripts/check_v1_completion.py` reports `status = pass`.
