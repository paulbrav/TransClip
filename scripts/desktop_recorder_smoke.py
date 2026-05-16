#!/usr/bin/env python
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.request import urlopen
from urllib.parse import urlparse


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the desktop WebAudio WAV recorder against Chromium's fake microphone.",
    )
    parser.add_argument("--port", type=int, default=1421)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--chromium", default=find_chromium())
    args = parser.parse_args()

    if not args.chromium:
        print("missing Chromium browser; install chromium or set --chromium", file=sys.stderr)
        return 1

    repo_root = Path(__file__).resolve().parents[1]
    desktop_dir = repo_root / "desktop"
    server = subprocess.Popen(
        ["npm", "run", "dev", "--", "--host", "127.0.0.1", "--port", str(args.port), "--strictPort"],
        cwd=desktop_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        wait_for_http(f"http://127.0.0.1:{args.port}/tests/recorder-smoke.html", args.timeout)
        page_url = f"http://127.0.0.1:{args.port}/tests/recorder-smoke.html"
        result = run_chromium_smoke(args.chromium, page_url, args.timeout)
        print(result)
        return 0 if result.startswith("PASS ") else 1
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=5)


def find_chromium() -> str:
    for candidate in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable"):
        path = shutil.which(candidate)
        if path:
            return path
    return ""


def run_chromium_smoke(chromium: str, page_url: str, timeout: float) -> str:
    debug_port = free_port()
    with tempfile.TemporaryDirectory(prefix="granite-recorder-chrome-") as profile:
        browser = subprocess.Popen(
            [
                chromium,
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                f"--remote-debugging-port={debug_port}",
                f"--user-data-dir={profile}",
                "--use-fake-device-for-media-stream",
                "--use-fake-ui-for-media-stream",
                "--autoplay-policy=no-user-gesture-required",
                page_url,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            ws_url = wait_for_page_ws(debug_port, timeout)
            with CdpConnection(ws_url) as cdp:
                deadline = time.time() + timeout
                last = ""
                while time.time() < deadline:
                    value = cdp.evaluate("document.querySelector('#result')?.textContent || ''")
                    if isinstance(value, str):
                        last = value
                    if last.startswith("PASS ") or last.startswith("FAIL "):
                        return last
                    time.sleep(0.25)
                return f"FAIL timed out waiting for recorder result; last result was {last!r}"
        finally:
            browser.terminate()
            try:
                browser.wait(timeout=5)
            except subprocess.TimeoutExpired:
                browser.kill()
                browser.wait(timeout=5)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_for_page_ws(debug_port: int, timeout: float) -> str:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urlopen(f"http://127.0.0.1:{debug_port}/json/list", timeout=2) as response:
                targets = json.loads(response.read().decode("utf-8"))
            for target in targets:
                if target.get("type") == "page" and target.get("webSocketDebuggerUrl"):
                    return str(target["webSocketDebuggerUrl"])
        except Exception as error:  # noqa: BLE001
            last_error = error
        time.sleep(0.25)
    raise RuntimeError(f"timed out waiting for Chromium DevTools target: {last_error}")


def wait_for_http(url: str, timeout: float) -> None:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except Exception as error:  # noqa: BLE001
            last_error = error
        time.sleep(0.25)
    raise RuntimeError(f"timed out waiting for {url}: {last_error}")


class CdpConnection:
    def __init__(self, ws_url: str):
        parsed = urlparse(ws_url)
        self.host = parsed.hostname or "127.0.0.1"
        self.port = parsed.port or 80
        self.path = parsed.path
        if parsed.query:
            self.path += "?" + parsed.query
        self.sock: socket.socket | None = None
        self.next_id = 1

    def __enter__(self) -> "CdpConnection":
        self.sock = socket.create_connection((self.host, self.port), timeout=5)
        key = base64.b64encode(os.urandom(16)).decode("ascii")
        request = (
            f"GET {self.path} HTTP/1.1\r\n"
            f"Host: {self.host}:{self.port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "\r\n"
        )
        self.sock.sendall(request.encode("ascii"))
        response = self._read_http_response()
        expected = base64.b64encode(
            hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")).digest()
        ).decode("ascii")
        if " 101 " not in response or expected not in response:
            raise RuntimeError(f"websocket handshake failed: {response}")
        return self

    def __exit__(self, _exc_type: object, _exc: object, _tb: object) -> None:
        if self.sock:
            self.sock.close()
            self.sock = None

    def evaluate(self, expression: str) -> object:
        message_id = self.next_id
        self.next_id += 1
        self._send_json(
            {
                "id": message_id,
                "method": "Runtime.evaluate",
                "params": {"expression": expression, "returnByValue": True},
            }
        )
        while True:
            message = self._recv_json()
            if message.get("id") == message_id:
                if "exceptionDetails" in message:
                    return "FAIL " + json.dumps(message["exceptionDetails"])
                return message.get("result", {}).get("result", {}).get("value")

    def _read_http_response(self) -> str:
        assert self.sock
        chunks = []
        while True:
            chunk = self.sock.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
            if b"\r\n\r\n" in b"".join(chunks):
                break
        return b"".join(chunks).decode("iso-8859-1")

    def _send_json(self, payload: object) -> None:
        data = json.dumps(payload).encode("utf-8")
        header = bytearray([0x81])
        length = len(data)
        if length < 126:
            header.append(0x80 | length)
        elif length < 65536:
            header.extend([0x80 | 126, (length >> 8) & 0xFF, length & 0xFF])
        else:
            header.append(0x80 | 127)
            header.extend(length.to_bytes(8, "big"))
        mask = os.urandom(4)
        header.extend(mask)
        masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(data))
        assert self.sock
        self.sock.sendall(bytes(header) + masked)

    def _recv_json(self) -> dict[str, object]:
        assert self.sock
        while True:
            first = self._recv_exact(2)
            opcode = first[0] & 0x0F
            length = first[1] & 0x7F
            if length == 126:
                length = int.from_bytes(self._recv_exact(2), "big")
            elif length == 127:
                length = int.from_bytes(self._recv_exact(8), "big")
            masked = bool(first[1] & 0x80)
            mask = self._recv_exact(4) if masked else b""
            payload = self._recv_exact(length)
            if masked:
                payload = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
            if opcode == 0x8:
                raise RuntimeError("websocket closed")
            if opcode == 0x9:
                continue
            if opcode == 0x1:
                return json.loads(payload.decode("utf-8"))

    def _recv_exact(self, length: int) -> bytes:
        assert self.sock
        chunks = []
        remaining = length
        while remaining:
            chunk = self.sock.recv(remaining)
            if not chunk:
                raise RuntimeError("socket closed")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)


if __name__ == "__main__":
    raise SystemExit(main())
