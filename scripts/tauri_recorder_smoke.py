#!/usr/bin/env python
from __future__ import annotations

import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import socket
import struct
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
        description=(
            "Smoke-test WebAudio microphone capture inside the actual Tauri/WebKit shell. "
            "This records about one second from the default microphone."
        ),
    )
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--log", type=Path, default=Path("/tmp/granite-speach-tauri-recorder-smoke.log"))
    args = parser.parse_args(argv)

    if sys.platform != "linux":
        print("tauri recorder smoke currently runs on Linux", file=sys.stderr)
        return 2

    callback = CallbackServer(("127.0.0.1", 0), SmokeHandler)
    callback_url = f"http://127.0.0.1:{callback.server_port}/result"
    callback_thread = threading.Thread(target=callback.serve_forever, daemon=True)
    callback_thread.start()

    fake_health: ThreadingHTTPServer | None = None
    if not service_is_running():
        fake_health = ThreadingHTTPServer((SERVICE_HOST, SERVICE_PORT), HealthHandler)
        threading.Thread(target=fake_health.serve_forever, daemon=True).start()

    proc: subprocess.Popen[str] | None = None
    try:
        env = os.environ.copy()
        env.setdefault("WEBKIT_DISABLE_COMPOSITING_MODE", "1")
        env["VITE_GRANITE_TAURI_RECORDER_SMOKE_URL"] = callback_url
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
                print(f"tauri dev exited before recorder result; see {args.log}", file=sys.stderr)
                return 1
            time.sleep(0.25)
        print(f"timed out waiting for recorder result; see {args.log}", file=sys.stderr)
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
        if fake_health:
            fake_health.shutdown()
            fake_health.server_close()


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


class HealthHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:
        payload = {
            "status": "ready",
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

    def log_message(self, format: str, *args: object) -> None:
        return

    def _cors(self) -> None:
        self.send_header("access-control-allow-origin", "*")
        self.send_header("access-control-allow-methods", "GET, OPTIONS")
        self.send_header("access-control-allow-headers", "content-type")


def validate_smoke_payload(payload: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": "fail",
        "user_agent": payload.get("userAgent", ""),
    }
    if not payload.get("ok"):
        result["error"] = payload.get("error", "recorder smoke failed without an error message")
        return result
    try:
        wav = base64.b64decode(str(payload["wavBase64"]), validate=True)
        info = parse_wav(wav)
    except Exception as error:  # noqa: BLE001
        result["error"] = str(error)
        return result
    result.update(info)
    if info["channels"] != 1:
        result["error"] = f"expected mono WAV, got {info['channels']} channels"
    elif info["bits_per_sample"] != 16:
        result["error"] = f"expected 16-bit WAV, got {info['bits_per_sample']}"
    elif info["sample_rate"] < 8000:
        result["error"] = f"sample rate is unexpectedly low: {info['sample_rate']}"
    elif info["data_bytes"] <= 0:
        result["error"] = "WAV contains no audio data"
    else:
        result["status"] = "pass"
    return result


def parse_wav(wav: bytes) -> dict[str, int]:
    if len(wav) < 44 or wav[:4] != b"RIFF" or wav[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    offset = 12
    fmt: dict[str, int] | None = None
    data_bytes = 0
    while offset + 8 <= len(wav):
        chunk_id = wav[offset : offset + 4]
        chunk_size = struct.unpack_from("<I", wav, offset + 4)[0]
        chunk_start = offset + 8
        chunk_end = chunk_start + chunk_size
        if chunk_end > len(wav):
            raise ValueError("WAV chunk extends past end of file")
        if chunk_id == b"fmt ":
            if chunk_size < 16:
                raise ValueError("WAV fmt chunk is too short")
            audio_format, channels, sample_rate, _byte_rate, _block_align, bits = struct.unpack_from(
                "<HHIIHH",
                wav,
                chunk_start,
            )
            fmt = {
                "audio_format": audio_format,
                "channels": channels,
                "sample_rate": sample_rate,
                "bits_per_sample": bits,
            }
        elif chunk_id == b"data":
            data_bytes = chunk_size
        offset = chunk_end + (chunk_size % 2)
    if fmt is None:
        raise ValueError("missing WAV fmt chunk")
    if fmt["audio_format"] != 1:
        raise ValueError(f"expected PCM WAV, got format {fmt['audio_format']}")
    return {
        **fmt,
        "data_bytes": data_bytes,
        "duration_ms": int((data_bytes / (fmt["sample_rate"] * fmt["channels"] * (fmt["bits_per_sample"] / 8))) * 1000),
    }


def service_is_running() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((SERVICE_HOST, SERVICE_PORT)) == 0


if __name__ == "__main__":
    raise SystemExit(main())
