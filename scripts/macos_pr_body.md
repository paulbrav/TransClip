## Summary

Makes TransClip install and pass `doctor` on macOS without the Linux ROCm stack.

- **Paths:** cache and toggle logs use `~/Library/Caches/transclip` and `~/Library/Logs/transclip` (not `~/.cache`).
- **LaunchAgent:** `launchctl bootstrap` / `bootout` in the GUI session; `LimitLoadToSessionType: Aqua`; no ROCm env vars in the plist.
- **Service cwd:** `WorkingDirectory` is the config directory so the daemon does not depend on the git checkout path.
- **Doctor:** Granite NAR skips `flash-attn` on Darwin; hotkey readiness uses `hotkey_macos`.
- **Docs:** README macOS Desktop section (install, Accessibility/Microphone, MPS, Library paths).

## Why

The README already advertised Linux + macOS, but install used XDG cache paths, deprecated `launchctl load`, ROCm-only doctor checks, and a repo-relative working directory. That blocked a clean macOS setup on Apple Silicon.

## File notes

| File | Change |
|------|--------|
| `transclip/platform_runtime.py` | `user_cache_dir` uses `~/Library/Caches` on Darwin. |
| `transclip/daemon_lifecycle.py` | macOS LaunchAgent bootstrap/bootout, config `WorkingDirectory`, Linux-only ROCm env, `LimitLoadToSessionType`. |
| `transclip/gnome_shortcut.py` | Toggle wrapper logs to `user_log_dir` (works on macOS and Linux). |
| `transclip/doctor.py` | `flash-attn` required only on ROCm Linux; hotkey binding follows platform. |
| `README.md` | macOS quick start, permissions, paths; tray noted as Linux-only for now. |
| `tests/*` | Platform path, LaunchAgent, doctor, and CLI log path coverage (136 tests). |

## Test plan

- [x] `python3.13 -m unittest discover -s tests -v` — 136 passed locally
- [ ] `uv run -m transclip.cli init-config`
- [ ] `uv run -m transclip.cli install` — plist under `~/Library/LaunchAgents/`, bootstrap succeeds
- [ ] `uv run -m transclip.cli doctor` — no false `flash-attn` failure on macOS
- [ ] Bind **System Settings → Keyboard → Shortcuts** to the printed toggle command (`Option+Space` default)
- [ ] `uv run -m transclip.cli toggle-record --paste` — grant **Accessibility** when prompted
- [ ] `uv run -m transclip.cli status` — LaunchAgent active after install

## Follow-ups (out of scope)

- macOS menu bar tray (Linux AppIndicator only today)
- Automating Services shortcut binding (manual step documented in install output)
