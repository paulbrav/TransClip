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
      controller.py        # TrayController + shared action callbacks
      ports.py             # TrayPorts test seams (health, toggle, history, …)
      session.py
      menu.py
      menu_update.py       # TrayMenuSnapshot, after_tray_action
      materialize.py       # Shared menu tree walk (TrayMenuSink protocol)
      views.py             # RefDrivenMenuView for platform menu refs
      gtk.py               # Linux GTK/AppIndicator adapter
      win32.py             # Windows pystray adapter
      macos.py             # macOS PyObjC adapter
      sinks/               # Platform TrayMenuSink implementations
  daemon/                  # Service install, lifecycle, status, logs
    __init__.py            # Public re-export hub
    protocol.py            # PlatformDaemon protocol
    common.py
    lifecycle.py           # OS registry + dispatch
    linux.py               # systemd
    macos.py               # launchd
    windows.py             # Task Scheduler
    status.py              # collect_status, smoke-test, toggle log
  doctor/                  # Readiness checks
    __init__.py            # run_checks
    platform.py
    asr.py
    types.py
  service/                 # Local HTTP dictation server
    __init__.py            # InferenceEngine, InferenceClient, health helpers, …
    client.py              # InferenceClient
    session.py             # DictationSession
    health.py              # build_health_status, cleanup_labels, settings_health_payload
    client_health.py       # fetch_service_health_result, service_health_is_ready, … (caller-side HTTP)
    serialize.py           # TranscriptOutcome / CleanupResult → typed HTTP responses
    json_response.py       # json_object_response helper for InferenceClient
    types.py               # ServiceHealthResponse, RecordSessionResponse, …
    routes.py              # dispatch_get/post, RouteResponse
    engine.py              # InferenceEngine
    server.py              # create_server, run_server
  cli/                     # Console entry, argparse, command dispatch
    __init__.py            # main(), handle_command re-export
    __main__.py            # python -m transclip.cli
    parser.py              # argparse tree
    dispatch.py            # handle_command router
    formatting.py          # status/history/models stdout helpers
    init_config.py         # init-config
    serve.py               # serve
    doctor_cmd.py          # doctor
    daemon_cmd.py          # install/start/stop/status/logs/smoke-test
    tray_cmd.py            # tray
    history_cmd.py         # history
    models_cmd.py          # models list/doctor/prefetch
    config_cmd.py          # config get/set
    shortcut_cmd.py        # install-gnome-shortcut
    toggle_cmd.py          # toggle-record
    transcribe_cmd.py      # transcribe, cleanup
    eval_cmd.py            # eval
  asr.py, …                # Core dictation pipeline (root domain modules)
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
| Daemon protocol | `transclip.daemon.protocol` | `PlatformDaemon` |
| Doctor | `transclip.doctor` | `run_checks`, `Check` |
| Service | `transclip.service` | `InferenceEngine`, `InferenceClient`, `create_server`, `run_server`, typed HTTP responses in `types` |
| Service client health | `transclip.service.client_health` | `fetch_service_health_result`, `service_health_is_ready`, `service_health_check_detail` |
| CLI | `transclip.cli` | `main`, `handle_command` |
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
(walks the tree through a `TrayMenuSink`). `TrayController` and `TrayPorts`
orchestrate session state, health, and actions; `RefDrivenMenuView` in
`views.py` implements the shared `TrayMenuView` protocol for ref-keyed widgets.
Platform code in `gtk.py`, `win32.py`, and `macos.py` is thin orchestration;
sink classes live under `tray/sinks/`.

Post-action behavior (refresh history after toggle, update labels) uses
`after_tray_action` in `menu_update.py` across all three tray adapters.

### Daemon split

- `protocol.py` — `PlatformDaemon` contract shared by platform modules.
- `lifecycle.py` — registry and dispatch only (`install_daemon`, `service_state`, …).
- `linux.py` / `macos.py` / `windows.py` — platform install units, service control, and `platform_daemon` adapter.
- `status.py` — HTTP-adjacent status, smoke-test, toggle log, log streaming.

### Service package boundaries

- **Server-side health** (`service/health.py`): built by `InferenceEngine.health()` for `/health` responses.
- **Client-side health** (`service/client_health.py`): polls `/health` via `InferenceClient` from doctor, daemon status, and tray ports. Import this module directly; it is not re-exported from `transclip.service`.
- **HTTP response shapes** (`service/types.py`): `ServiceHealthResponse`, `RecordSessionResponse`, etc., used by `InferenceClient` return types.

### CLI execution models

- **HTTP client commands** (`toggle-record`, daemon health probes): use `InferenceClient` against a running service.
- **In-process commands** (`transcribe`, `cleanup`, `eval`): construct `InferenceEngine` in the CLI process. See `CONTEXT.md` for rationale.

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
| `transclip.service` (flat module) | `transclip.service` (package) |
| `transclip.client` | `transclip.service` or `transclip.service.client` |
| `transclip.dictation_session` | `transclip.service` or `transclip.service.session` |
| `transclip.service_health` | `transclip.service.health` (server) or `transclip.service.client_health` (caller) |
| `transclip.service_routes` | `transclip.service.routes` (internal) |
| `transclip.cli` (flat module) | `transclip.cli` (package) |
| `transclip.cli_commands` | `transclip.cli.dispatch` + command submodules |

There are no compatibility shims for old flat module paths.
