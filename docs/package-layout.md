# TransClip package layout

TransClip organizes platform and desktop integration into domain packages.
Prefer the **public import paths** below; reach into submodules only for
platform-specific tests or when extending a single adapter.

## Directory map

```text
transclip/
  paths.py                 # Neutral path helpers (e.g. service_settings_path)
  platform/                # OS/runtime facts, capabilities, runtime profiles
    runtime.py
    capabilities.py
    profiles.py
  desktop/
    paste/                 # Clipboard + paste injection
      __init__.py          # SystemClipboard, paste_capability, …
      platform.py          # Backend registry / selection
      win32.py
    hotkey/                # Shortcut install + toggle command builders
      __init__.py          # Public router (lazy Linux imports)
      common.py            # Setup messages (macOS, Windows)
      toggle_command.py    # build_toggle_command / build_toggle_invocation
      linux_gnome.py       # gsettings GNOME custom shortcut
      windows.py           # In-process keyboard hook (tray)
    tray/                  # Menu bar / AppIndicator UI
      __init__.py          # run_tray router
      session.py
      menu.py
      menu_update.py       # TrayMenuSnapshot, after_tray_action
      materialize.py       # Shared menu tree walk (TrayMenuSink protocol)
      gtk.py               # Linux GTK/AppIndicator adapter
      win32.py             # Windows pystray adapter
      macos.py             # macOS PyObjC adapter
      sinks/               # Platform TrayMenuSink implementations
  daemon/                  # Service install, lifecycle, status, logs
    __init__.py            # Public re-export hub
    common.py
    lifecycle.py           # Thin OS dispatch
    linux.py               # systemd
    macos.py               # launchd
    windows.py             # Task Scheduler
    status.py              # collect_status, smoke-test, toggle log
  doctor/                  # Readiness checks
    __init__.py            # run_checks
    platform.py
    asr.py
    types.py
  cli.py, service.py, …    # Core dictation pipeline (unchanged flat modules)
```

## Public import paths

Use these from CLI, scripts, tests, and cross-package code:

| Package | Import | Purpose |
|---------|--------|---------|
| Platform | `transclip.platform.runtime` | `get_runtime`, `PlatformRuntime`, user dirs, `open_path` |
| Paste | `transclip.desktop.paste` | `SystemClipboard`, `SystemPasteInjector`, `paste_capability` |
| Hotkey | `transclip.desktop.hotkey` | `install_shortcut`, `hotkey_setup_message`, `build_toggle_command`, `get_gnome_shortcut_status`, `shortcut_readiness`, `start_windows_hotkey` |
| Tray | `transclip.desktop.tray` | `run_tray` |
| Daemon | `transclip.daemon` | `install_daemon`, `service_state`, `collect_status`, `toggle_log_path`, … |
| Doctor | `transclip.doctor` | `run_checks`, `Check` |
| Paths | `transclip.paths` | `service_settings_path` (shared by daemon units and hotkey commands) |

`tests/test_package_imports.py` smoke-imports the main entry points and guards
against circular-import regressions (especially `transclip.desktop.hotkey`).

## Design notes

### Hotkey router

- **Linux:** `install_shortcut` → `linux_gnome` (gsettings), called from `transclip install` and the GTK tray.
- **Windows:** `start_windows_hotkey` when the Windows tray starts; binding from `hotkey_windows` in settings.
- **macOS:** No hotkey implementation module. Users configure System Settings or Shortcuts.app manually. The tray exposes **Copy hotkey setup command**; doctor reports readiness via setup messages, not an installer.

Linux-specific helpers (`get_gnome_shortcut_status`, `shortcut_readiness`) are
re-exported from `transclip.desktop.hotkey` so `daemon/status` and `doctor`
do not import `linux_gnome` directly.

`linux_gnome` is lazy-imported inside the hotkey router so importing
`build_toggle_command` on Windows/macOS does not load gsettings code.

### Tray adapters

Shared menu structure lives in `menu.py` (node tree) and `materialize.py`
(walks the tree through a `TrayMenuSink`). Platform code in `gtk.py`,
`win32.py`, and `macos.py` is thin orchestration; sink classes live under
`tray/sinks/`.

Post-action behavior (refresh history after toggle, update labels) uses
`after_tray_action` in `menu_update.py` across all three tray adapters.

### Daemon split

- `lifecycle.py` — dispatch only (`install_daemon`, `service_state`, …).
- `linux.py` / `macos.py` / `windows.py` — platform install units and service control.
- `status.py` — HTTP-adjacent status, smoke-test, toggle log, log streaming.

### Import migration (historical)

| Old module | New path |
|------------|----------|
| `transclip.platform_runtime` | `transclip.platform.runtime` |
| `transclip.platform_capabilities` | `transclip.platform.capabilities` |
| `transclip.runtime_profile` | `transclip.platform.profiles` |
| `transclip.paste` | `transclip.desktop.paste` |
| `transclip.hotkey_setup` | `transclip.desktop.hotkey.common` |
| `transclip.gnome_shortcut` | `transclip.desktop.hotkey.linux_gnome` |
| `transclip.hotkey_windows` | `transclip.desktop.hotkey.windows` |
| `transclip.tray` | `transclip.desktop.tray` |
| `transclip.daemon` (monolith) | `transclip.daemon` (package) |
| `transclip.daemon_lifecycle` | `transclip.daemon.lifecycle` + platform modules |
| `transclip.doctor` (monolith) | `transclip.doctor` (package) |

There are no compatibility shims for old flat module paths.
