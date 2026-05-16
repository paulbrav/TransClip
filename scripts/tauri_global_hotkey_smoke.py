#!/usr/bin/env python
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import shutil
import socket
import subprocess
import sys
import threading
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DESKTOP = ROOT / "desktop"
SERVICE_HOST = "127.0.0.1"
SERVICE_PORT = 8765


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Smoke-test Tauri global Ctrl+Space hold-to-record wiring on Linux.",
    )
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--log", type=Path, default=Path("/tmp/granite-speach-tauri-hotkey-smoke.log"))
    args = parser.parse_args(argv)

    if sys.platform != "linux":
        print("tauri global hotkey smoke currently runs on Linux", file=sys.stderr)
        return 2
    ydotool = shutil.which("ydotool")
    if not ydotool:
        print("ydotool is required for the global hotkey smoke", file=sys.stderr)
        return 2
    if service_is_running():
        print("port 8765 is already in use; stop the real service before this smoke", file=sys.stderr)
        return 2

    FakeServiceHandler.reset()
    fake_service = ThreadingHTTPServer((SERVICE_HOST, SERVICE_PORT), FakeServiceHandler)
    threading.Thread(target=fake_service.serve_forever, daemon=True).start()
    proc: subprocess.Popen[str] | None = None
    try:
        env = os.environ.copy()
        env.setdefault("WEBKIT_DISABLE_COMPOSITING_MODE", "1")
        args.log.parent.mkdir(parents=True, exist_ok=True)
        with args.log.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(
                ["npm", "run", "tauri", "dev"],
                cwd=DESKTOP,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                start_new_session=True,
            )

        deadline = time.monotonic() + args.timeout
        wait_for(lambda: FakeServiceHandler.health_count > 0, deadline, "frontend did not call /health")
        time.sleep(0.5)
        subprocess.run([ydotool, "key", "ctrl+space"], check=True)
        wait_for(lambda: FakeServiceHandler.start_count > 0, deadline, "hotkey did not call /record/start")
        wait_for(lambda: FakeServiceHandler.stop_count > 0, deadline, "hotkey did not call /record/stop")
        result = {
            "status": "pass",
            "health_count": FakeServiceHandler.health_count,
            "start_count": FakeServiceHandler.start_count,
            "stop_count": FakeServiceHandler.stop_count,
            "log": str(args.log),
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(str(exc), file=sys.stderr)
        print(f"see {args.log}", file=sys.stderr)
        return 1
    finally:
        if proc and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
        fake_service.shutdown()
        fake_service.server_close()


class FakeServiceHandler(BaseHTTPRequestHandler):
    health_count = 0
    start_count = 0
    stop_count = 0
    recording = False

    @classmethod
    def reset(cls) -> None:
        cls.health_count = 0
        cls.start_count = 0
        cls.stop_count = 0
        cls.recording = False

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:
        if self.path != "/health":
            self._json(404, {"error": "not found"})
            return
        type(self).health_count += 1
        self._json(
            200,
            {
                "status": "recording" if type(self).recording else "ready",
                "hotkey": "Ctrl+Space",
                "max_recording_seconds": 60,
                "min_recording_ms": 0,
                "clipboard_restore_delay_ms": 0,
                "restore_clipboard_after_paste": True,
                "cleanup_enabled": True,
            },
        )

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        if length:
            self.rfile.read(length)
        if self.path == "/record/start":
            type(self).start_count += 1
            type(self).recording = True
            self._json(200, {"status": "recording", "already_recording": False})
            return
        if self.path == "/record/stop":
            type(self).stop_count += 1
            type(self).recording = False
            self._json(
                200,
                {
                    "status": "ready",
                    "text": "",
                    "raw_asr": "",
                    "duration_ms": 1.0,
                    "timings_ms": {"end_to_end": 1.0},
                },
            )
            return
        self._json(404, {"error": "not found"})

    def log_message(self, format: str, *args: object) -> None:
        return

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self._cors()
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _cors(self) -> None:
        self.send_header("access-control-allow-origin", "*")
        self.send_header("access-control-allow-methods", "GET, POST, OPTIONS")
        self.send_header("access-control-allow-headers", "content-type")


def wait_for(predicate, deadline: float, message: str) -> None:
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.1)
    raise RuntimeError(message)


def service_is_running() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((SERVICE_HOST, SERVICE_PORT)) == 0


if __name__ == "__main__":
    raise SystemExit(main())
