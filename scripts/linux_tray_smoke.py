from __future__ import annotations

import argparse
import ast
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[1]
DESKTOP = ROOT / "desktop"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Smoke-test Linux StatusNotifier/AppIndicator registration for the Tauri shell.",
    )
    parser.add_argument("--timeout", type=int, default=70)
    parser.add_argument("--log", type=Path, default=Path("/tmp/granite-speach-tauri-tray-smoke.log"))
    parser.add_argument(
        "--skip-menu-check",
        action="store_true",
        help="Only verify tray item registration, not DBusMenu labels",
    )
    parser.add_argument(
        "--skip-record-action-check",
        action="store_true",
        help="Do not click Record/Stop through DBusMenu",
    )
    args = parser.parse_args(argv)

    if sys.platform != "linux":
        print("linux tray smoke only runs on Linux", file=sys.stderr)
        return 2
    if not status_notifier_host_registered():
        print("no StatusNotifier host is registered in this desktop session", file=sys.stderr)
        return 2

    before = registered_items()
    server = ThreadingHTTPServer(("127.0.0.1", 8765), HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    proc: subprocess.Popen[str] | None = None
    try:
        env = os.environ.copy()
        env.setdefault("WEBKIT_DISABLE_COMPOSITING_MODE", "1")
        args.log.parent.mkdir(parents=True, exist_ok=True)
        with args.log.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(
                ["timeout", "--kill-after=5s", f"{args.timeout}s", "npm", "run", "tauri", "dev"],
                cwd=DESKTOP,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
            )
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            new_items = registered_items() - before
            if new_items:
                item = sorted(new_items)[0]
                labels = [] if args.skip_menu_check else wait_for_menu_labels(item, deadline)
                record_actions = (
                    {}
                    if args.skip_menu_check or args.skip_record_action_check
                    else check_record_actions(item, deadline)
                )
                print(
                    json.dumps(
                        {
                            "registered_new_items": sorted(new_items),
                            "menu_labels": labels,
                            "record_actions": record_actions,
                            "log": str(args.log),
                        }
                    )
                )
                return 0
            if proc.poll() is not None:
                print(f"tauri dev exited before tray item appeared; see {args.log}", file=sys.stderr)
                return 1
            time.sleep(1)
        print(f"timed out waiting for tray item; see {args.log}", file=sys.stderr)
        return 1
    finally:
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        server.shutdown()
        server.server_close()


class HealthHandler(BaseHTTPRequestHandler):
    status = "ready"

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:
        payload = {
            "status": type(self).status,
            "hotkey": "Ctrl+Space",
            "max_recording_seconds": 60,
            "min_recording_ms": 250,
            "clipboard_restore_delay_ms": 500,
            "restore_clipboard_after_paste": True,
            "cleanup_enabled": True,
        }
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self._cors()
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        if length:
            self.rfile.read(length)
        if self.path == "/record/start":
            type(self).status = "recording"
            self._json(200, {"status": "recording"})
            return
        if self.path == "/record/stop":
            type(self).status = "ready"
            self._json(
                200,
                {
                    "status": "ready",
                    "text": "Tray smoke transcript.",
                    "raw_asr": "tray smoke transcript",
                    "duration_ms": 400.0,
                    "timings_ms": {"end_to_end": 1.0},
                },
            )
            return
        self._json(404, {"error": f"unknown path: {self.path}"})

    def log_message(self, format: str, *args: object) -> None:
        return

    def _cors(self) -> None:
        self.send_header("access-control-allow-origin", "*")
        self.send_header("access-control-allow-methods", "GET, POST, OPTIONS")
        self.send_header("access-control-allow-headers", "content-type")

    def _json(self, status: int, payload: dict[str, object]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self._cors()
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def status_notifier_host_registered() -> bool:
    try:
        raw = gdbus_property("IsStatusNotifierHostRegistered")
    except subprocess.CalledProcessError:
        return False
    return "<true>" in raw


def registered_items() -> set[str]:
    raw = gdbus_property("RegisteredStatusNotifierItems")
    inside = raw.split("<", 1)[1].rsplit(">", 1)[0]
    return set(ast.literal_eval(inside))


def wait_for_menu_labels(item: str, deadline: float) -> list[str]:
    missing: list[str] = EXPECTED_LABELS[:]
    labels: list[str] = []
    while time.monotonic() < deadline:
        labels = menu_labels(item)
        missing = [label for label in EXPECTED_LABELS if label not in labels]
        has_status = any(label.startswith("Status: ") for label in labels)
        if not missing and has_status:
            return labels
        time.sleep(1)
    raise RuntimeError(f"tray menu labels missing: {missing}; observed: {labels}")


def check_record_actions(item: str, deadline: float) -> dict[str, str]:
    bus, menu_path = menu_address(item)
    record = wait_for_menu_entry(item, "Record", deadline, enabled=True)
    emit_menu_click(bus, menu_path, record["id"])
    wait_for_fake_status("recording", deadline)
    stop = wait_for_menu_entry(item, "Stop", deadline, enabled=True)
    emit_menu_click(bus, menu_path, stop["id"])
    wait_for_fake_status("ready", deadline)
    return {"record": "recording", "stop": "ready"}


def menu_labels(item: str) -> list[str]:
    return [entry["label"] for entry in menu_entries(item)]


def wait_for_menu_entry(
    item: str,
    label: str,
    deadline: float,
    enabled: bool | None = None,
) -> dict[str, object]:
    last_entries: list[dict[str, object]] = []
    while time.monotonic() < deadline:
        last_entries = menu_entries(item)
        for entry in last_entries:
            if entry["label"] == label and (enabled is None or entry["enabled"] is enabled):
                return entry
        time.sleep(0.25)
    raise RuntimeError(f"menu entry not found: label={label!r} enabled={enabled}; observed: {last_entries}")


def menu_entries(item: str) -> list[dict[str, object]]:
    bus, path = item.split("@", 1)
    menu_path = menu_path_for(bus, path)
    layout = subprocess.check_output(
        [
            "busctl",
            "--user",
            "call",
            bus,
            menu_path,
            "com.canonical.dbusmenu",
            "GetLayout",
            "iias",
            "--",
            "0",
            "-1",
            "0",
        ],
        text=True,
    )
    return menu_entries_from_layout(layout)


def menu_entries_from_layout(layout: str) -> list[dict[str, object]]:
    entries = []
    for match in re.finditer(r'\(ia\{sv\}av\) (\d+) (.*?)(?= \(ia\{sv\}av\)|$)', layout):
        body = match.group(2)
        label_match = re.search(r'"label" s "([^"]*)"', body)
        if not label_match:
            continue
        entries.append(
            {
                "id": int(match.group(1)),
                "label": label_match.group(1),
                "enabled": '"enabled" b false' not in body,
            }
        )
    return entries


def menu_address(item: str) -> tuple[str, str]:
    bus, path = item.split("@", 1)
    return bus, menu_path_for(bus, path)


def menu_path_for(bus: str, path: str) -> str:
    raw_menu = gdbus_property_for(bus, path, "org.kde.StatusNotifierItem", "Menu")
    return raw_menu.split("'", 2)[1]


def emit_menu_click(bus: str, menu_path: str, item_id: object) -> None:
    subprocess.check_call(
        [
            "gdbus",
            "call",
            "--session",
            "--dest",
            bus,
            "--object-path",
            menu_path,
            "--method",
            "com.canonical.dbusmenu.Event",
            str(item_id),
            "clicked",
            "<int32 0>",
            "0",
        ],
        stdout=subprocess.DEVNULL,
    )


def wait_for_fake_status(expected: str, deadline: float) -> None:
    while time.monotonic() < deadline:
        if HealthHandler.status == expected:
            return
        time.sleep(0.05)
    raise RuntimeError(f"fake service status did not become {expected}; got {HealthHandler.status}")


def gdbus_property(name: str) -> str:
    return gdbus_property_for(
        "org.kde.StatusNotifierWatcher",
        "/StatusNotifierWatcher",
        "org.kde.StatusNotifierWatcher",
        name,
    )


def gdbus_property_for(dest: str, object_path: str, interface: str, name: str) -> str:
    return subprocess.check_output(
        [
            "gdbus",
            "call",
            "--session",
            "--dest",
            dest,
            "--object-path",
            object_path,
            "--method",
            "org.freedesktop.DBus.Properties.Get",
            interface,
            name,
        ],
        text=True,
    )


EXPECTED_LABELS = [
    "Cleanup",
    "Record",
    "Stop + Paste",
    "Stop",
    "Paste latest transcript",
    "Copy latest transcript",
    "Recent transcripts",
    "Start service",
    "Show status window",
    "Open keyword glossary",
    "Open settings",
    "Quit",
]


if __name__ == "__main__":
    raise SystemExit(main())
