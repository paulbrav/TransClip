#!/usr/bin/env python
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
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
        description="Smoke-test the Tauri frontend service-side Record/Stop flow.",
    )
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--log", type=Path, default=Path("/tmp/granite-speach-tauri-service-record-smoke.log"))
    args = parser.parse_args(argv)

    if sys.platform != "linux":
        print("tauri service record smoke currently runs on Linux", file=sys.stderr)
        return 2

    callback = CallbackServer(("127.0.0.1", 0), SmokeHandler)
    callback_url = f"http://127.0.0.1:{callback.server_port}/result"
    threading.Thread(target=callback.serve_forever, daemon=True).start()

    fake_service: ThreadingHTTPServer | None = None
    if not service_is_running():
        fake_service = ThreadingHTTPServer((SERVICE_HOST, SERVICE_PORT), FakeServiceHandler)
        threading.Thread(target=fake_service.serve_forever, daemon=True).start()

    proc: subprocess.Popen[str] | None = None
    try:
        env = os.environ.copy()
        env.setdefault("WEBKIT_DISABLE_COMPOSITING_MODE", "1")
        env["VITE_GRANITE_TAURI_SERVICE_RECORD_SMOKE_URL"] = callback_url
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
        while time.monotonic() < deadline:
            if callback.result is not None:
                print(json.dumps(callback.result, indent=2, sort_keys=True))
                return 0 if callback.result.get("status") == "pass" else 1
            if proc.poll() is not None:
                print(f"tauri dev exited before service-record result; see {args.log}", file=sys.stderr)
                return 1
            time.sleep(0.25)
        print(f"timed out waiting for service-record result; see {args.log}", file=sys.stderr)
        return 1
    finally:
        if proc and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
        callback.shutdown()
        callback.server_close()
        if fake_service:
            fake_service.shutdown()
            fake_service.server_close()


class CallbackServer(ThreadingHTTPServer):
    result: dict[str, Any] | None = None


class SmokeHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        result = validate_smoke_payload(payload)
        self.server.result = result  # type: ignore[attr-defined]
        encoded = json.dumps({"ok": True}).encode("utf-8")
        self.send_response(200)
        self._cors()
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:
        return

    def _cors(self) -> None:
        self.send_header("access-control-allow-origin", "*")
        self.send_header("access-control-allow-methods", "POST, OPTIONS")
        self.send_header("access-control-allow-headers", "content-type")


class FakeServiceHandler(BaseHTTPRequestHandler):
    recording = False

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:
        if self.path != "/health":
            self._json(404, {"error": "not found"})
            return
        self._json(
            200,
            {
                "status": "recording" if self.recording else "ready",
                "hotkey": "Ctrl+Space",
                "max_recording_seconds": 60,
                "min_recording_ms": 250,
                "clipboard_restore_delay_ms": 500,
                "restore_clipboard_after_paste": True,
                "cleanup_enabled": True,
            },
        )

    def do_POST(self) -> None:
        if self.path == "/record/start":
            type(self).recording = True
            self._json(200, {"status": "recording", "already_recording": False})
            return
        if self.path == "/record/stop":
            type(self).recording = False
            self._json(200, {"status": "ready", "duration_ms": 100, "discarded": True})
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


def validate_smoke_payload(payload: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "fail",
        "user_agent": payload.get("userAgent", ""),
        "status_while_recording": payload.get("statusWhileRecording", ""),
        "stop_enabled_while_recording": payload.get("stopEnabledWhileRecording", False),
        "final_status": payload.get("finalStatus", ""),
        "error_text": payload.get("errorText", ""),
    }
    if not payload.get("ok"):
        result["error"] = payload.get("error") or payload.get("errorText") or "service record smoke failed"
        return result
    if payload.get("statusWhileRecording") != "Recording":
        result["error"] = "frontend did not enter Recording status"
    elif not payload.get("stopEnabledWhileRecording"):
        result["error"] = "Stop control was not enabled while recording"
    elif payload.get("finalStatus") != "Ready":
        result["error"] = "frontend did not return to Ready status"
    elif payload.get("errorText"):
        result["error"] = payload["errorText"]
    else:
        result["status"] = "pass"
    return result


def service_is_running() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((SERVICE_HOST, SERVICE_PORT)) == 0


if __name__ == "__main__":
    raise SystemExit(main())
