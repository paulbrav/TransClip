from __future__ import annotations

import ipaddress
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from transclip.settings import Settings, load_settings

from .engine import InferenceEngine
from .openai_compat import MAX_UPLOAD_BYTES, handle_transcriptions, openai_error_body
from .routes import dispatch_get, dispatch_post

_LOOPBACK_HOSTNAMES = frozenset({"localhost", "127.0.0.1", "::1"})

# Upper bound on how much of an over-limit upload we will discard to deliver a
# clean 413 (the cap plus one chunk of slack). A client declaring more than this
# is abusive and forfeits clean delivery.
_OVERSIZE_DRAIN_LIMIT = MAX_UPLOAD_BYTES + 65536


class TransclipHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        request_handler: type[BaseHTTPRequestHandler],
        bind_and_activate: bool = True,
    ):
        super().__init__(server_address, request_handler, bind_and_activate)
        self._shutdown_events: list[threading.Event] = []

    def add_shutdown_event(self, event: threading.Event) -> None:
        self._shutdown_events.append(event)

    def shutdown(self) -> None:
        self._signal_shutdown_events()
        super().shutdown()

    def server_close(self) -> None:
        self._signal_shutdown_events()
        super().server_close()

    def _signal_shutdown_events(self) -> None:
        for event in self._shutdown_events:
            event.set()


def _hostname(value: str, *, has_scheme: bool) -> str | None:
    """Extract the host part of an Origin (with scheme) or Host header (no scheme)."""
    if not value:
        return None
    target = value if has_scheme else f"//{value}"
    try:
        return urlsplit(target).hostname
    except ValueError:
        return None


def _is_local_host(hostname: str | None, extra: frozenset[str]) -> bool:
    if hostname is None:
        return False
    name = hostname.strip().lower()
    if name in _LOOPBACK_HOSTNAMES or name in extra:
        return True
    try:
        return ipaddress.ip_address(name).is_loopback
    except ValueError:
        return False


def create_server(
    settings: Settings | None = None,
    engine: InferenceEngine | None = None,
) -> ThreadingHTTPServer:
    settings = settings or load_settings()
    active_engine = engine or InferenceEngine(settings, warm_asr=True)

    # The service has no authentication and exposes the microphone, the local
    # LLM, and arbitrary-file transcription. Without these guards any web page
    # the user visits could reach http://127.0.0.1:<port> from the browser via
    # DNS rebinding or CORS and silently drive recording. Allow only loopback
    # hosts (plus an explicit non-loopback bind) and reject browser-originated
    # cross-origin requests.
    configured = settings.host.strip().lower()
    extra_hosts = frozenset({configured}) if configured else frozenset()
    # When bound to all interfaces the operator has explicitly opted out of
    # loopback-only access, so Host pinning is relaxed; Origin rejection stays.
    enforce_host = configured not in {"0.0.0.0", "::", ""}

    class Handler(BaseHTTPRequestHandler):
        def do_OPTIONS(self) -> None:
            if not self._guard():
                return
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_GET(self) -> None:
            if not self._guard():
                return
            response = dispatch_get(active_engine, urlsplit(self.path).path)
            self._json(response.status, response.payload)

        def do_POST(self) -> None:
            if not self._guard():
                return
            path = urlsplit(self.path).path
            # Self-contained compat branch: runs AFTER _guard() (loopback/origin
            # protection) but BEFORE the generic JSON dispatch and its exception
            # wrapper, so it never flows through _read_json/_json and can never
            # leak the internal {"error": str, "debug_capture_dir": ...} shape.
            if path == "/v1/audio/transcriptions":
                self._handle_openai_transcriptions()
                return
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

        def _handle_openai_transcriptions(self) -> None:
            raw_length = self.headers.get("content-length")
            if raw_length is None:
                self._openai_write(
                    411,
                    "application/json",
                    openai_error_body("Content-Length header is required.", error_type="invalid_request_error"),
                )
                return
            try:
                length = int(raw_length)
            except ValueError:
                self._openai_write(
                    400,
                    "application/json",
                    openai_error_body("Invalid Content-Length header.", error_type="invalid_request_error"),
                )
                return
            if length > MAX_UPLOAD_BYTES:
                # Reject on the declared length rather than buffering the body
                # (memory safety), but DRAIN it first (bounded, in chunks — O(1)
                # memory) so an over-limit client's upload completes and it reads
                # a clean 413 instead of a connection reset. Without the drain,
                # closing the socket mid-upload RSTs on Windows and the OpenAI SDK
                # sees a retryable connection error, re-uploading the oversize
                # body up to max_retries times. A client declaring more than the
                # drain limit forfeits clean delivery — that is the abuse case.
                self._drain_body(min(length, _OVERSIZE_DRAIN_LIMIT))
                self._openai_write(
                    413,
                    "application/json",
                    openai_error_body(
                        f"Uploaded file exceeds the {MAX_UPLOAD_BYTES}-byte limit.",
                        error_type="invalid_request_error",
                    ),
                )
                return
            body = self.rfile.read(length) if length > 0 else b""
            status, content_type, payload = handle_transcriptions(active_engine, self.headers.get("content-type"), body)
            self._openai_write(status, content_type, payload)

        def _drain_body(self, limit: int) -> None:
            # Discard up to `limit` bytes of the request body in fixed chunks so a
            # rejected client's send completes without buffering the payload.
            remaining = limit
            while remaining > 0:
                chunk = self.rfile.read(min(remaining, 65536))
                if not chunk:
                    return
                remaining -= len(chunk)

        def _openai_write(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self._cors()
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _guard(self) -> bool:
            """Reject browser cross-origin and DNS-rebinding requests."""
            origin = self.headers.get("origin")
            if origin is not None and not _is_local_host(
                _hostname(origin, has_scheme=True), extra_hosts
            ):
                self._json(403, {"error": "cross-origin requests are not allowed"})
                return False
            if enforce_host and not _is_local_host(
                _hostname(self.headers.get("host", ""), has_scheme=False), extra_hosts
            ):
                self._json(403, {"error": "invalid host header"})
                return False
            return True

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
            origin = self.headers.get("origin")
            if origin is None or not _is_local_host(
                _hostname(origin, has_scheme=True), extra_hosts
            ):
                return
            self.send_header("access-control-allow-origin", origin)
            self.send_header("vary", "origin")
            self.send_header("access-control-allow-methods", "GET, POST, OPTIONS")
            self.send_header("access-control-allow-headers", "authorization, content-type")

    stop_event = threading.Event()
    server = TransclipHTTPServer((settings.host, settings.port), Handler)
    server.add_shutdown_event(stop_event)
    threading.Thread(
        target=active_engine.warm_bucket_shapes,
        args=(stop_event,),
        name="transclip-bucket-warm",
        daemon=True,
    ).start()
    return server


def run_server(settings: Settings | None = None) -> None:
    settings = settings or load_settings()
    server = create_server(settings)
    print(f"transclip service listening on http://{settings.host}:{settings.port}", flush=True)
    server.serve_forever()
