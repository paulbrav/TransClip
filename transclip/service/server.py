from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

from transclip.settings import Settings, load_settings

from .engine import InferenceEngine
from .routes import dispatch_get, dispatch_post


def create_server(
    settings: Settings | None = None,
    engine: InferenceEngine | None = None,
) -> ThreadingHTTPServer:
    settings = settings or load_settings()
    active_engine = engine or InferenceEngine(settings)

    class Handler(BaseHTTPRequestHandler):
        def do_OPTIONS(self) -> None:
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_GET(self) -> None:
            response = dispatch_get(active_engine, urlparse(self.path).path)
            self._json(response.status, response.payload)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            try:
                body = self._read_json()
                response = dispatch_post(active_engine, path, body)
                self._json(response.status, response.payload)
            except Exception as exc:
                capture_dir = active_engine.debug_capture.write_error(
                    "http_request",
                    exc,
                    {"path": path},
                )
                payload = {"error": str(exc)}
                if capture_dir:
                    payload["debug_capture_dir"] = str(capture_dir)
                self._json(500, payload)

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("content-length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            return json.loads(raw.decode("utf-8"))

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

    return ThreadingHTTPServer((settings.host, settings.port), Handler)


def run_server(settings: Settings | None = None) -> None:
    settings = settings or load_settings()
    server = create_server(settings)
    print(f"transclip service listening on http://{settings.host}:{settings.port}", flush=True)
    server.serve_forever()
