from __future__ import annotations

import plistlib
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePath

from transclip.daemon.common import CommandResult, logs_dir, repo_root
from transclip.paths import service_settings_path
from transclip.platform.runtime import PlatformRuntime, get_runtime, user_cache_dir
from transclip.product import CACHE_DIR_NAME, IMPORT_PACKAGE
from transclip.settings import Settings

Runner = Callable[..., subprocess.CompletedProcess[str]]

HOTKEY_APP_NAME = "TransClipHotkey"
HOTKEY_BUNDLE_ID = "com.paulbrav.TransClipHotkey"
HOTKEY_LAUNCHD_LABEL = "com.paulbrav.transclip-hotkey"
HOTKEY_LOG_NAME = "hotkey.log"
TOGGLE_WRAPPER_NAME = "transclip-toggle"
DEFAULT_STOP_TIMEOUT_SECONDS = 75
STALE_LOCK_SECONDS = 90


@dataclass(frozen=True, slots=True)
class MacOSHotkeyInstall:
    app_path: Path
    launch_agent_path: Path
    wrapper_path: Path
    source_path: Path


def macos_hotkey_app_path(runtime: PlatformRuntime | None = None) -> Path:
    return get_runtime(runtime).home_dir() / "Applications" / f"{HOTKEY_APP_NAME}.app"


def macos_hotkey_launch_agent_path(runtime: PlatformRuntime | None = None) -> Path:
    return get_runtime(runtime).home_dir() / "Library" / "LaunchAgents" / f"{HOTKEY_LAUNCHD_LABEL}.plist"


def macos_toggle_wrapper_path(runtime: PlatformRuntime | None = None) -> Path:
    return get_runtime(runtime).home_dir() / "bin" / TOGGLE_WRAPPER_NAME


def macos_hotkey_source_path(runtime: PlatformRuntime | None = None) -> Path:
    return user_cache_dir(CACHE_DIR_NAME, runtime) / "macos-hotkey" / f"{HOTKEY_APP_NAME}.swift"


def macos_hotkey_log_path(runtime: PlatformRuntime | None = None) -> Path:
    return logs_dir(runtime) / HOTKEY_LOG_NAME


def macos_hotkey_state_path(runtime: PlatformRuntime | None = None) -> Path:
    return logs_dir(runtime) / "hotkey-state.tsv"


def macos_hotkey_target(runtime: PlatformRuntime | None = None) -> str:
    return f"{macos_launchd_gui_domain(runtime)}/{HOTKEY_LAUNCHD_LABEL}"


def macos_launchd_gui_domain(runtime: PlatformRuntime | None = None) -> str:
    output = get_runtime(runtime).check_output(["id", "-u"])
    if isinstance(output, bytes):
        output = output.decode()
    return f"gui/{output.strip()}"


def build_macos_toggle_wrapper(
    settings: Settings,
    settings_path: Path | None = None,
    runtime: PlatformRuntime | None = None,
    *,
    stop_timeout_seconds: int = DEFAULT_STOP_TIMEOUT_SECONDS,
    stale_lock_seconds: int = STALE_LOCK_SECONDS,
    restart_command: str | None = None,
) -> str:
    log_path = logs_dir(runtime) / "toggle-record.log"
    state_path = macos_hotkey_state_path(runtime)
    base_url = f"http://{settings.host}:{settings.port}"
    python = shlex.quote(sys.executable)
    cli = f"{python} -m {shlex.quote(IMPORT_PACKAGE + '.cli')}"
    if settings_path:
        cli += f" --settings {shlex.quote(service_settings_path(settings_path))}"
    restart_command_text = restart_command or (
        f'cd {shlex.quote(_macos_path_text(repo_root()))} && {cli} restart >> "$LOG" 2>&1'
    )
    return f"""#!/bin/sh
set -u

LOG={shlex.quote(_macos_path_text(log_path))}
STATE={shlex.quote(_macos_path_text(state_path))}
BASE={shlex.quote(base_url)}
MAX_SECONDS={int(stop_timeout_seconds)}
STALE_LOCK_SECONDS={int(stale_lock_seconds)}
STOP_POLL_SECONDS=0.05
LOCK=/tmp/transclip-toggle.lock

mkdir -p "$(dirname "$LOG")"
mkdir -p "$(dirname "$STATE")"

write_state() {{
  state=$1
  detail=$2
  printf '%s\\t%s\\t%s\\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$state" "$detail" > "$STATE"
  printf '%s\\n' "$(date '+%Y-%m-%dT%H:%M:%S%z') state=${{state}} ${{detail}}" >> "$LOG"
}}

restart_service() {{
  {restart_command_text}
}}

wait_for_ready() {{
  attempts=0
  while [ "$attempts" -lt 30 ]; do
    health=$(curl -sS --max-time 2 "$BASE/health" 2>>"$LOG" || true)
    printf '%s\\n' "$health" >> "$LOG"
    case "$health" in
      *'"status": "ready"'*|*'"status":"ready"'*)
        write_state ready "Ready"
        return 0
        ;;
    esac
    attempts=$((attempts + 1))
    sleep 1
  done
  write_state error "Service restart failed"
  return 1
}}

wait_for_stop_response() {{
  stop_pid=$1
  stop_status_file=$2
  transcribing_written=0
  while [ ! -f "$stop_status_file" ]; do
    if [ "$transcribing_written" = "0" ]; then
      health=$(curl -sS --max-time 2 "$BASE/health" 2>>"$LOG" || true)
      printf '%s\\n' "$health" >> "$LOG"
      case "$health" in
        *'"status": "transcribing"'*|*'"status":"transcribing"'*|*'"status": "ready"'*|*'"status":"ready"'*)
          write_state transcribing "Transcribing"
          transcribing_written=1
          ;;
      esac
    fi
    if [ -f "$stop_status_file" ]; then
      break
    fi
    sleep "$STOP_POLL_SECONDS"
  done
  wait "$stop_pid" 2>/dev/null || true
}}

kill_process_tree() {{
  pid=$1
  for child in $(pgrep -P "$pid" 2>/dev/null || true); do
    kill_process_tree "$child"
  done
  kill "$pid" 2>/dev/null || true
}}

clear_stale_lock_owner() {{
  owner_pid=$(cat "$LOCK/pid" 2>/dev/null || true)
  case "$owner_pid" in
    ''|*[!0-9]*)
      return
      ;;
  esac
  owner_command=$(ps -p "$owner_pid" -o command= 2>/dev/null || true)
  case "$owner_command" in
    *transclip-toggle*)
      write_state recovering "Clearing stale action"
      printf '%s\\n' "$(date '+%Y-%m-%dT%H:%M:%S%z') killing stale TransClip action pid=${{owner_pid}}" >> "$LOG"
      kill_process_tree "$owner_pid"
      ;;
  esac
}}

printf '%s\\n' "$(date '+%Y-%m-%dT%H:%M:%S%z') wrapper invoked" >> "$LOG"
write_state shortcut "Starting recording"

if ! mkdir "$LOCK" 2>/dev/null; then
  now=$(date +%s)
  lock_mtime=$(stat -f %m "$LOCK" 2>/dev/null || printf '0')
  lock_age=$((now - lock_mtime))
  if [ "$lock_age" -lt "$STALE_LOCK_SECONDS" ]; then
    write_state busy "Previous action still running"
    printf '%s\\n' \
      "$(date '+%Y-%m-%dT%H:%M:%S%z') ignored: previous TransClip action still running (${{lock_age}}s)" \
      >> "$LOG"
    exit 0
  fi
  printf '%s\\n' "$(date '+%Y-%m-%dT%H:%M:%S%z') clearing stale TransClip action lock (${{lock_age}}s)" >> "$LOG"
  clear_stale_lock_owner
  rm -rf "$LOCK"
  if ! mkdir "$LOCK" 2>/dev/null; then
    printf '%s\\n' "$(date '+%Y-%m-%dT%H:%M:%S%z') ignored: could not acquire TransClip action lock" >> "$LOG"
    exit 0
  fi
fi
printf '%s\\n' "$$" > "$LOCK/pid"
trap 'rm -rf "$LOCK"' EXIT HUP INT TERM

response=$(curl -sS --max-time 10 -X POST "$BASE/record/start" \
  -H 'content-type: application/json' --data '{{}}' 2>>"$LOG") || {{
  write_state recovering "Restarting service"
  printf '%s\\n' "$(date '+%Y-%m-%dT%H:%M:%S%z') start failed; restarting service" >> "$LOG"
  restart_service
  wait_for_ready || true
  exit 0
}}
printf '%s\\n' "$response" >> "$LOG"
case "$response" in
  *'"already_recording": true'*|*'"already_recording":true'*)
    already_recording=1
    ;;
  *)
    already_recording=0
    ;;
esac
printf '%s\\n' "$(date '+%Y-%m-%dT%H:%M:%S%z') start already_recording=${{already_recording}}" >> "$LOG"

if [ "$already_recording" = "1" ]; then
  write_state stopping "Stopping recording"
  stop_response_file="$LOCK/stop-response.json"
  stop_status_file="$LOCK/stop-status"
  rm -f "$stop_response_file" "$stop_status_file"
  (
    curl -sS --max-time "$MAX_SECONDS" -X POST "$BASE/record/stop" \
      -H 'content-type: application/json' --data '{{}}' > "$stop_response_file" 2>>"$LOG"
    printf '%s\\n' "$?" > "$stop_status_file"
  ) &
  stop_pid=$!
  wait_for_stop_response "$stop_pid" "$stop_status_file"
  stop_status=$(cat "$stop_status_file" 2>/dev/null || printf '1')
  if [ "$stop_status" != "0" ]; then
    write_state recovering "Restarting service"
    printf '%s\\n' "$(date '+%Y-%m-%dT%H:%M:%S%z') stop failed; restarting service" >> "$LOG"
    restart_service
    wait_for_ready || true
    exit 0
  fi
  response=$(cat "$stop_response_file" 2>/dev/null || true)
  printf '%s\\n' "$response" >> "$LOG"
  text=$(printf '%s' "$response" | {python} -c \
    'import json,sys; print(json.load(sys.stdin).get("text", ""), end="")' 2>>"$LOG")
  if [ -n "$text" ]; then
    write_state paste_requested "Paste transcript"
    printf '%s' "$text" | pbcopy
    printf '%s\\n' "$(date '+%Y-%m-%dT%H:%M:%S%z') copied transcript chars=${{#text}}" >> "$LOG"
  else
    write_state finished "No transcript"
    printf '%s\\n' "$(date '+%Y-%m-%dT%H:%M:%S%z') stop returned no text" >> "$LOG"
  fi
else
  write_state listening "Recording"
fi

exit 0
"""


def build_macos_hotkey_source(
    wrapper_path: Path,
    log_path: Path,
    state_path: Path,
) -> str:
    source = """import ApplicationServices
import AppKit
import Carbon
import Foundation

let logPath = "@@LOG_PATH@@"
let wrapperPath = "@@WRAPPER_PATH@@"
let statePath = "@@STATE_PATH@@"
let spaceKeyCode: Int64 = 49

func log(_ message: String) {
    let formatter = ISO8601DateFormatter()
    let line = "\\(formatter.string(from: Date())) \\(message)\\n"
    guard let data = line.data(using: .utf8) else { return }

    if FileManager.default.fileExists(atPath: logPath),
       let handle = try? FileHandle(forWritingTo: URL(fileURLWithPath: logPath)) {
        defer { try? handle.close() }
        _ = try? handle.seekToEnd()
        _ = try? handle.write(contentsOf: data)
    } else {
        try? data.write(to: URL(fileURLWithPath: logPath))
    }
}

func writeStateFile(_ state: String, _ detail: String) {
    let formatter = ISO8601DateFormatter()
    let line = "\\(formatter.string(from: Date()))\\t\\(state)\\t\\(detail)\\n"
    try? line.write(to: URL(fileURLWithPath: statePath), atomically: true, encoding: .utf8)
}

func postCommandV() {
    let source = CGEventSource(stateID: .hidSystemState)
    let commandKeyCode: CGKeyCode = 55
    let vKeyCode: CGKeyCode = 9

    let commandDown = CGEvent(keyboardEventSource: source, virtualKey: commandKeyCode, keyDown: true)
    commandDown?.flags = .maskCommand
    commandDown?.post(tap: .cghidEventTap)
    usleep(20_000)

    let vDown = CGEvent(keyboardEventSource: source, virtualKey: vKeyCode, keyDown: true)
    vDown?.flags = .maskCommand
    vDown?.post(tap: .cghidEventTap)
    usleep(20_000)

    let vUp = CGEvent(keyboardEventSource: source, virtualKey: vKeyCode, keyDown: false)
    vUp?.flags = .maskCommand
    vUp?.post(tap: .cghidEventTap)
    usleep(20_000)

    let commandUp = CGEvent(keyboardEventSource: source, virtualKey: commandKeyCode, keyDown: false)
    commandUp?.post(tap: .cghidEventTap)
}

class HotkeyStatus: NSObject {
    let statusItem: NSStatusItem
    let menu = NSMenu()
    let statusMenuItem: NSMenuItem
    var lastStateLine = ""
    var readyResetTimer: Timer?

    override init() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        statusMenuItem = NSMenuItem(title: "TransClip: Ready", action: nil, keyEquivalent: "")
        super.init()

        statusMenuItem.isEnabled = false
        menu.addItem(statusMenuItem)
        menu.addItem(NSMenuItem.separator())

        let openToggleLog = NSMenuItem(
            title: "Open toggle log",
            action: #selector(openToggleLog(_:)),
            keyEquivalent: ""
        )
        openToggleLog.target = self
        menu.addItem(openToggleLog)

        let openHotkeyLog = NSMenuItem(
            title: "Open hotkey log",
            action: #selector(openHotkeyLog(_:)),
            keyEquivalent: ""
        )
        openHotkeyLog.target = self
        menu.addItem(openHotkeyLog)

        menu.addItem(NSMenuItem.separator())
        let quit = NSMenuItem(title: "Quit TransClip Hotkey", action: #selector(quit(_:)), keyEquivalent: "q")
        quit.target = self
        menu.addItem(quit)

        statusItem.menu = menu
        setStatus("ready", "Ready")
    }

    func setStatus(_ state: String, _ detail: String) {
        if Thread.isMainThread {
            applyStatus(state, detail)
        } else {
            DispatchQueue.main.async {
                self.applyStatus(state, detail)
            }
        }
    }

    private func applyStatus(_ state: String, _ detail: String) {
        readyResetTimer?.invalidate()
        readyResetTimer = nil

        let title: String
        let fallback: String

        switch state {
        case "shortcut":
            title = "TC..."
            fallback = "Shortcut received"
        case "busy":
            title = "TC..."
            fallback = "Already working"
        case "recovering":
            title = "TC..."
            fallback = "Recovering"
        case "listening":
            title = "REC"
            fallback = "Recording"
        case "stopping":
            title = "REC"
            fallback = "Stopping recording"
        case "transcribing":
            title = "TXT..."
            fallback = "Transcribing"
        case "pasting":
            title = "PST..."
            fallback = "Pasting transcript"
        case "paste_requested":
            title = "PST..."
            fallback = "Paste transcript"
        case "finished":
            title = "OK"
            fallback = "Finished"
        case "ready":
            title = "TC"
            fallback = "Ready"
        case "error":
            title = "TC!"
            fallback = "Error"
        default:
            title = "TC"
            fallback = "Ready"
        }

        let message = detail.isEmpty ? fallback : detail
        statusItem.button?.attributedTitle = NSAttributedString(
            string: title,
            attributes: [
                .foregroundColor: color(for: state),
                .font: NSFont.monospacedSystemFont(ofSize: NSFont.systemFontSize, weight: .semibold),
            ]
        )
        statusItem.button?.toolTip = "TransClip: \\(message)"
        statusMenuItem.title = "TransClip: \\(message)"

        if state == "finished" {
            readyResetTimer = Timer.scheduledTimer(withTimeInterval: 2.5, repeats: false) { [weak self] _ in
                self?.setStatus("ready", "Ready")
            }
        }
    }

    private func color(for state: String) -> NSColor {
        switch state {
        case "shortcut", "busy", "recovering":
            return .systemYellow
        case "listening", "stopping":
            return .systemOrange
        case "transcribing":
            return .systemPurple
        case "pasting", "paste_requested":
            return .systemTeal
        case "finished":
            return .systemGreen
        case "ready":
            return .labelColor
        case "error":
            return .systemRed
        default:
            return .labelColor
        }
    }

    @objc func pollState(_ timer: Timer) {
        guard let line = try? String(contentsOfFile: statePath, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines),
              !line.isEmpty,
              line != lastStateLine else {
            return
        }

        lastStateLine = line
        let parts = line.components(separatedBy: "\\t")
        let state = parts.count > 1 ? parts[1] : "ready"
        let detail = parts.count > 2 ? parts[2] : state
        if state == "paste_requested" {
            performPasteRequest(detail)
            return
        }
        setStatus(state, detail)
    }

    func performPasteRequest(_ detail: String) {
        setStatus("pasting", detail.isEmpty ? "Pasting transcript" : detail)
        log("paste requested by wrapper")
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.15) {
            postCommandV()
            log("posted Command+V")
            writeStateFile("finished", "Pasted")
            self.setStatus("finished", "Pasted")
        }
    }

    @objc func openToggleLog(_ sender: Any?) {
        let toggleLogPath = logPath.replacingOccurrences(
            of: "hotkey.log",
            with: "toggle-record.log"
        )
        NSWorkspace.shared.open(URL(fileURLWithPath: toggleLogPath))
    }

    @objc func openHotkeyLog(_ sender: Any?) {
        NSWorkspace.shared.open(URL(fileURLWithPath: logPath))
    }

    @objc func quit(_ sender: Any?) {
        NSApplication.shared.terminate(nil)
    }
}

let app = NSApplication.shared
app.setActivationPolicy(.accessory)
let hotkeyStatus = HotkeyStatus()
Timer.scheduledTimer(
    timeInterval: 0.25,
    target: hotkeyStatus,
    selector: #selector(HotkeyStatus.pollState(_:)),
    userInfo: nil,
    repeats: true
)

func runWrapper() {
    hotkeyStatus.setStatus("shortcut", "Shortcut received")
    let process = Process()
    process.executableURL = URL(fileURLWithPath: "/bin/sh")
    process.arguments = ["-lc", wrapperPath]
    do {
        try process.run()
        log("launched wrapper pid=\\(process.processIdentifier)")
    } catch {
        log("failed to launch wrapper: \\(error)")
    }
}

let promptKey = kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String
let trusted = AXIsProcessTrustedWithOptions([promptKey: true] as CFDictionary)
log("event tap starting axTrusted=\\(trusted)")
if !trusted {
    hotkeyStatus.setStatus("error", "Accessibility required")
}

var activeEventTap: CFMachPort?

func reenableEventTap() {
    guard let tap = activeEventTap else {
        log("event tap re-enable skipped; no active tap")
        return
    }

    CGEvent.tapEnable(tap: tap, enable: true)
    log("event tap re-enabled")
}

let callback: CGEventTapCallBack = { _, type, event, _ in
    if type == .tapDisabledByTimeout || type == .tapDisabledByUserInput {
        log("event tap disabled type=\\(type.rawValue)")
        reenableEventTap()
        return Unmanaged.passUnretained(event)
    }

    guard type == .keyDown else {
        return Unmanaged.passUnretained(event)
    }

    let keyCode = event.getIntegerValueField(.keyboardEventKeycode)
    let flags = event.flags
    let hasOption = flags.contains(.maskAlternate)
    let hasCommand = flags.contains(.maskCommand)
    let hasControl = flags.contains(.maskControl)

    if keyCode == spaceKeyCode && hasOption && !hasCommand && !hasControl {
        log("Option+Space detected")
        runWrapper()
        return nil
    }

    return Unmanaged.passUnretained(event)
}

let mask = CGEventMask(1 << CGEventType.keyDown.rawValue)
guard let eventTap = CGEvent.tapCreate(
    tap: .cgSessionEventTap,
    place: .headInsertEventTap,
    options: .defaultTap,
    eventsOfInterest: mask,
    callback: callback,
    userInfo: nil
) else {
    log("failed to create event tap; Accessibility/Input Monitoring is required")
    hotkeyStatus.setStatus("error", "Accessibility required")
    app.run()
    exit(0)
}

activeEventTap = eventTap
let runLoopSource = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, eventTap, 0)
CFRunLoopAddSource(CFRunLoopGetCurrent(), runLoopSource, .commonModes)
CGEvent.tapEnable(tap: eventTap, enable: true)
log("event tap listening for Option+Space")
if trusted {
    writeStateFile("ready", "Ready")
    hotkeyStatus.setStatus("ready", "Ready")
}
app.run()
"""
    return (
        source.replace("@@LOG_PATH@@", _swift_string(_macos_path_text(log_path)))
        .replace("@@WRAPPER_PATH@@", _swift_string(_macos_path_text(wrapper_path)))
        .replace("@@STATE_PATH@@", _swift_string(_macos_path_text(state_path)))
    )


def build_macos_hotkey_launch_agent(runtime: PlatformRuntime | None = None) -> bytes:
    log_root = logs_dir(runtime)
    payload = {
        "Label": HOTKEY_LAUNCHD_LABEL,
        "ProgramArguments": [str(_macos_hotkey_executable_path(runtime))],
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": str(log_root / "hotkey.out.log"),
        "StandardErrorPath": str(log_root / "hotkey.err.log"),
    }
    return plistlib.dumps(payload, sort_keys=True)


def install_macos_hotkey(
    settings: Settings,
    settings_path: Path | None = None,
    runner: Runner = subprocess.run,
    runtime: PlatformRuntime | None = None,
) -> tuple[MacOSHotkeyInstall, list[CommandResult]]:
    platform_runtime = get_runtime(runtime)
    if platform_runtime.system() != "Darwin":
        raise RuntimeError("macOS hotkey helper is only supported on Darwin")

    paths = MacOSHotkeyInstall(
        app_path=macos_hotkey_app_path(platform_runtime),
        launch_agent_path=macos_hotkey_launch_agent_path(platform_runtime),
        wrapper_path=macos_toggle_wrapper_path(platform_runtime),
        source_path=macos_hotkey_source_path(platform_runtime),
    )
    results: list[CommandResult] = []
    logs_dir(platform_runtime).mkdir(parents=True, exist_ok=True)

    paths.wrapper_path.parent.mkdir(parents=True, exist_ok=True)
    paths.wrapper_path.write_text(
        build_macos_toggle_wrapper(settings, settings_path, platform_runtime),
        encoding="utf-8",
    )
    paths.wrapper_path.chmod(0o755)
    results.append(CommandResult(True, f"wrote {paths.wrapper_path}"))

    swiftc = platform_runtime.which("swiftc")
    if not swiftc:
        results.append(
            CommandResult(False, "swiftc missing; install Xcode Command Line Tools with: xcode-select --install")
        )
        return paths, results

    executable = _macos_hotkey_executable_path(platform_runtime)
    executable.parent.mkdir(parents=True, exist_ok=True)
    paths.source_path.parent.mkdir(parents=True, exist_ok=True)
    paths.source_path.write_text(
        build_macos_hotkey_source(
            paths.wrapper_path,
            macos_hotkey_log_path(platform_runtime),
            macos_hotkey_state_path(platform_runtime),
        ),
        encoding="utf-8",
    )
    _macos_hotkey_info_plist_path(platform_runtime).write_bytes(_build_info_plist())
    results.append(CommandResult(True, f"wrote {paths.source_path}"))
    results.append(_run_command([swiftc, str(paths.source_path), "-o", str(executable)], runner))
    if not results[-1].ok:
        return paths, results
    executable.chmod(0o755)

    codesign = platform_runtime.which("codesign") or "codesign"
    results.append(_run_command([codesign, "--force", "--deep", "--sign", "-", str(paths.app_path)], runner))
    if not results[-1].ok:
        return paths, results

    paths.launch_agent_path.parent.mkdir(parents=True, exist_ok=True)
    paths.launch_agent_path.write_bytes(build_macos_hotkey_launch_agent(platform_runtime))
    results.append(CommandResult(True, f"wrote {paths.launch_agent_path}"))
    target = macos_hotkey_target(platform_runtime)
    domain = macos_launchd_gui_domain(platform_runtime)
    results.append(_run_command(["launchctl", "bootout", target], runner, tolerate_failure=True))
    results.append(_run_command(["launchctl", "bootstrap", domain, str(paths.launch_agent_path)], runner))
    results.append(
        CommandResult(
            True,
            ("Grant Accessibility to TransClipHotkey.app so it can observe Option+Space and post Command+V."),
        )
    )
    return paths, results


def uninstall_macos_hotkey(
    runner: Runner = subprocess.run,
    runtime: PlatformRuntime | None = None,
) -> list[CommandResult]:
    platform_runtime = get_runtime(runtime)
    results = [
        _run_command(["launchctl", "bootout", macos_hotkey_target(platform_runtime)], runner, tolerate_failure=True)
    ]
    path = macos_hotkey_launch_agent_path(platform_runtime)
    if path.exists():
        path.unlink()
        results.append(CommandResult(True, f"removed {path}"))
    app_path = macos_hotkey_app_path(platform_runtime)
    if app_path.exists():
        shutil.rmtree(app_path)
        results.append(CommandResult(True, f"removed {app_path}"))
    wrapper_path = macos_toggle_wrapper_path(platform_runtime)
    if wrapper_path.exists():
        wrapper_path.unlink()
        results.append(CommandResult(True, f"removed {wrapper_path}"))
    source_path = macos_hotkey_source_path(platform_runtime)
    if source_path.exists():
        source_path.unlink()
        results.append(CommandResult(True, f"removed {source_path}"))
    return results


def _run_command(command: list[str], runner: Runner, tolerate_failure: bool = False) -> CommandResult:
    try:
        result = runner(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    except FileNotFoundError as exc:
        return CommandResult(tolerate_failure, f"{command[0]} missing: {exc}")
    output = result.stdout.strip()
    ok = result.returncode == 0 or tolerate_failure
    detail = shlex.join(command)
    if output:
        detail += f": {output}"
    elif result.returncode != 0:
        detail += f": exit {result.returncode}"
    return CommandResult(ok, detail)


def _build_info_plist() -> bytes:
    return plistlib.dumps(
        {
            "CFBundleExecutable": HOTKEY_APP_NAME,
            "CFBundleIdentifier": HOTKEY_BUNDLE_ID,
            "CFBundleName": HOTKEY_APP_NAME,
            "CFBundleDisplayName": HOTKEY_APP_NAME,
            "CFBundlePackageType": "APPL",
            "CFBundleShortVersionString": "1.0",
            "CFBundleVersion": "1",
            "LSUIElement": True,
        },
        sort_keys=True,
    )


def _macos_hotkey_executable_path(runtime: PlatformRuntime | None = None) -> Path:
    return macos_hotkey_app_path(runtime) / "Contents" / "MacOS" / HOTKEY_APP_NAME


def _macos_hotkey_info_plist_path(runtime: PlatformRuntime | None = None) -> Path:
    return macos_hotkey_app_path(runtime) / "Contents" / "Info.plist"


def _swift_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _macos_path_text(path: PurePath) -> str:
    return path.as_posix()
