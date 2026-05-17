# Granite Speach Pure-Python Daemon Plan

## Goal

Make Granite Speach a small, reliable dictation daemon:

```text
press shortcut -> start recording
press shortcut -> stop recording -> transcribe -> copy transcript -> paste
```

Tauri should not be in the critical path. It may remain as an optional status UI
or be removed after the daemon path works on Linux and macOS.

## Target Tech Stack

### Shared Core

- Language: Python 3.12+.
- Package/runtime manager: `uv`.
- Configuration: existing TOML settings at `~/.config/granite-speach/settings.toml`
  on Linux and `~/Library/Application Support/granite-speach/settings.toml` on
  macOS.
- Local service: existing Python HTTP service in `granite_speach/service.py`.
- Audio capture: existing `sounddevice`/`numpy` based `AudioRecorder`.
- ASR: existing Granite/PyTorch backend.
- Cleanup: existing rule/model cleanup backends.
- CLI: existing `argparse` command in `granite_speach/cli.py`.
- Logs: JSON Lines under `~/.cache/granite-speach/` on Linux and
  `~/Library/Logs/granite-speach/` on macOS.
- Notifications: existing `notify-send` on Linux and AppleScript notification on
  macOS.

### Linux Desktop Stack

Target environment:

- Ubuntu 24.04+.
- GNOME Wayland.

Linux hotkey:

- GNOME native custom shortcuts through `gsettings`.
- Schema: `org.gnome.settings-daemon.plugins.media-keys`.
- Relocatable schema:
  `org.gnome.settings-daemon.plugins.media-keys.custom-keybinding`.
- Current HP ZBook Copilot key binding from `wev`:
  `<Super><Shift>XF86TouchpadOff`.
- Do not use Tauri global shortcut by default.
- Do not use `/dev/input`/evdev by default.

Linux service manager:

- `systemd --user`.
- Unit file:
  `~/.config/systemd/user/granite-speach.service`.

Linux clipboard:

- Required on GNOME Wayland: `wl-clipboard`.
- Commands:
  - `wl-copy`
  - `wl-paste`
- Do not silently fall back to `xclip` on Wayland. `xclip` is for X11 and can
  write to the wrong clipboard for focused Wayland applications.

Linux paste injection:

- First choice: `wtype`, if the compositor supports virtual keyboard protocol.
- Fallback: `ydotool`, if `ydotoold`/uinput is configured.
- X11 fallback only: `xdotool`.
- Doctor must report the actual chosen paste backend and why other backends are
  unavailable.

Linux optional tray/status:

- Not required for V1 daemon path.
- If kept, prefer a separate optional UI command.
- Tauri can remain temporarily, but the shortcut and recording path must not
  depend on it.

### macOS Desktop Stack

Target environment:

- Modern macOS on Apple Silicon or Intel.

macOS service manager:

- `launchd`.
- LaunchAgent file:
  `~/Library/LaunchAgents/com.paulbrav.granite-speach.plist`.

macOS clipboard:

- `pbcopy`.
- `pbpaste`.

macOS paste injection:

- AppleScript/System Events:

```bash
osascript -e 'tell application "System Events" to keystroke "v" using command down'
```

- Requires Accessibility permission for the invoking process.

macOS hotkey:

- Do not assume pure Python global hotkeys will be reliable without testing.
- Preferred options, in order:
  1. Tiny native/Tauri helper for macOS global hotkey only.
  2. User-configured macOS Keyboard Shortcut or Shortcuts.app action that runs
     `granite-speach toggle-record --paste`.
  3. Python Quartz/pynput listener only if it proves reliable with
     Accessibility/Input Monitoring permissions.

macOS UI:

- Optional.
- If Tauri remains useful, keep it only as a status/settings/hotkey helper.
- The Python service and CLI stay the source of truth.

## Runtime Architecture

```text
systemd user service / launchd agent
        runs
granite-speach serve
        exposes
GET  /health
POST /record/start
POST /record/stop
POST /record/toggle
        called by
granite-speach toggle-record --paste
        launched by
GNOME custom shortcut / macOS shortcut helper
```

## Command Surface

Add or finish these commands:

```bash
granite-speach install
granite-speach uninstall
granite-speach start
granite-speach stop
granite-speach restart
granite-speach status
granite-speach logs
granite-speach doctor
granite-speach doctor --fix
granite-speach smoke-test
granite-speach install-gnome-shortcut
granite-speach toggle-record --paste
```

Command responsibilities:

- `install`: platform-specific complete setup.
- `uninstall`: remove shortcut and service files, but do not delete user
  settings by default.
- `start`/`stop`/`restart`: call `systemctl --user` on Linux and `launchctl` on
  macOS.
- `status`: show service state, `/health`, shortcut binding, clipboard backend,
  paste backend, and last log event.
- `logs`: tail service logs plus shortcut JSONL logs.
- `doctor`: passive diagnostics.
- `doctor --fix`: apply safe fixes, such as creating missing directories or
  reinstalling the shortcut. It must not run `sudo apt install` itself unless
  explicitly confirmed by the user.
- `smoke-test`: run non-destructive checks and optional interactive paste test.

## Linux Systemd User Service

Generate this unit from the installer, using the actual repo path and Python
path detected on the target machine:

```ini
[Unit]
Description=Granite Speach dictation service
After=graphical-session.target

[Service]
Type=simple
WorkingDirectory=/home/paulbrav/github_software/granite_speach
ExecStart=/home/paulbrav/github_software/granite_speach/.venv-gfx1151/bin/python3 -m granite_speach.cli serve
Restart=on-failure
RestartSec=2
Environment=FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
Environment=TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

[Install]
WantedBy=default.target
```

Installer commands:

```bash
systemctl --user daemon-reload
systemctl --user enable --now granite-speach.service
```

Status/log commands:

```bash
systemctl --user status granite-speach.service
journalctl --user -u granite-speach.service -f
```

## Linux GNOME Shortcut

Shortcut name:

```text
Granite Speach Toggle
```

Shortcut binding on current HP ZBook:

```text
<Super><Shift>XF86TouchpadOff
```

Shortcut command should be a logging wrapper:

```bash
/bin/sh -lc 'mkdir -p "$HOME/.cache/granite-speach"; granite-speach toggle-record --paste >> "$HOME/.cache/granite-speach/toggle-record.log" 2>&1'
```

The installer must:

- Preserve unrelated custom shortcuts.
- Reuse the existing Granite shortcut path if present.
- Update name, binding, and command idempotently.
- Print the final installed binding and command.

## Toggle Logging

Every `toggle-record` invocation should append one JSON object per line to:

```text
~/.cache/granite-speach/toggle-record.log
```

Fields:

```json
{
  "timestamp": "2026-05-17T22:06:16+08:00",
  "action": "stopped",
  "service_url": "http://127.0.0.1:8765",
  "duration_ms": 11213.477,
  "text": "Test test here.",
  "raw_asr": "test test here",
  "paste_requested": true,
  "paste": {
    "copied": true,
    "pasted": false,
    "clipboard_backend": "wl-clipboard",
    "paste_backend": "wtype",
    "restored": false,
    "transcript_left_on_clipboard": true,
    "error_detail": "wtype paste command failed: compositor does not support virtual keyboard protocol"
  }
}
```

Error events should also be JSON:

```json
{
  "timestamp": "2026-05-17T22:07:00+08:00",
  "action": "error",
  "error": "Granite service is not running",
  "service_url": "http://127.0.0.1:8765"
}
```

## Paste Behavior

Change `PasteResult` to include explicit backend details:

```python
PasteResult(
    copied: bool,
    pasted: bool,
    restored: bool,
    transcript_left_on_clipboard: bool,
    clipboard_backend: str,
    paste_backend: str | None,
    error_detail: str,
)
```

Linux Wayland behavior:

- If `XDG_SESSION_TYPE=wayland`, require `wl-copy` and `wl-paste`.
- If missing, fail with:

```text
Wayland clipboard requires wl-clipboard. Install: sudo apt install wl-clipboard
```

- Do not use `xclip` unless session is X11.

Debug default:

```toml
restore_clipboard_after_paste = false
```

This makes failures recoverable: if paste injection fails, the user can manually
press `Ctrl+V`.

## Doctor Checks

`granite-speach doctor` should report:

- Config files exist.
- Service manager installed.
- Service active.
- `/health` responds.
- GNOME shortcut installed and bound.
- Current session type.
- Clipboard backend usable.
- Paste backend usable.
- Microphone visible.
- Model cache present.
- ASR runtime import works.
- Last shortcut log event.

On GNOME Wayland, `doctor` must fail if:

- `gsettings` missing.
- shortcut missing.
- binding differs from configured binding.
- `wl-copy`/`wl-paste` missing.
- neither `wtype` nor working `ydotool` exists.

Keep `/dev/input` advice only behind an explicit legacy diagnostic:

```bash
granite-speach doctor --legacy-evdev
```

## Smoke Tests

Non-interactive:

```bash
granite-speach smoke-test
```

Checks:

- Service responds to `/health`.
- `/record/toggle` starts.
- `/record/stop` with `discard=true` stops.
- Clipboard write/read round-trip works.
- Paste injector binary exists.
- Shortcut command exists.

Interactive:

```bash
granite-speach smoke-test --paste
```

Flow:

1. Ask user to focus a text editor.
2. Copy known text.
3. Trigger paste.
4. Ask user to confirm whether text appeared.
5. Log backend and result.

## What Happens To Tauri

Short term:

- Keep `desktop/` in the repo.
- Stop documenting it as the primary path.
- Do not use Tauri for hotkeys.
- Do not require Tauri for recording or paste.

Medium term:

- If a UI is still wanted, make Tauri call the Python service only for:
  - status
  - settings
  - logs
  - manual Record/Stop
- The daemon, shortcut, and paste path must work with Tauri closed.

Removal criteria:

- Linux daemon install works after login.
- macOS daemon install works after login.
- Shortcut toggle works on both platforms.
- `status`, `logs`, and `doctor` are sufficient for support.

If all criteria pass, move `desktop/` to `legacy/desktop` or remove it.

## Implementation Order

1. Make `toggle-record` write JSONL logs.
2. Make paste result include clipboard and paste backend details.
3. Make Wayland require `wl-clipboard` instead of falling back to `xclip`.
4. Add Linux systemd user service manager.
5. Add `start`, `stop`, `restart`, `status`, and `logs`.
6. Add `install` for Linux.
7. Add `smoke-test`.
8. Update README to document daemon-first setup.
9. Validate end-to-end on current GNOME machine.
10. Add macOS LaunchAgent service manager.
11. Decide macOS hotkey helper strategy.
12. Validate end-to-end on macOS.
13. Decide whether to keep, demote, or remove Tauri.

## Acceptance Criteria

Linux:

- `granite-speach install` completes with a clear pass/fail summary.
- After login, `systemctl --user status granite-speach.service` is active.
- `granite-speach status` reports `ready`.
- GNOME Settings shows `Granite Speach Toggle`.
- Press Copilot once: GNOME microphone indicator appears.
- Press Copilot again: transcript appears in the focused text field.
- If paste fails, transcript remains on clipboard.
- `granite-speach logs` shows what happened.

macOS:

- `granite-speach install` installs and starts LaunchAgent.
- `granite-speach status` reports `ready`.
- Hotkey or Shortcuts.app action triggers toggle.
- Transcript pastes into focused app after Accessibility permission.
- Logs show recording, transcription, and paste result.

