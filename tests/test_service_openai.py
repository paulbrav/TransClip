import getpass
import http.client
import json
import tempfile
import unittest
from pathlib import Path

from transclip.asr import TranscriptionResult
from transclip.cleanup import FaithfulRuleCleanupBackend
from transclip.service import InferenceEngine
from transclip.service.openai_compat import MAX_UPLOAD_BYTES
from transclip.settings import Settings

from tests.service_helpers import (
    FakeASR,
    build_multipart_body,
    http_multipart,
    serve_test_engine,
    stop_server,
    tiny_wav_bytes,
)

_PATH = "/v1/audio/transcriptions"


class RaisingASR:
    name = "raising"
    model = "raising-model"

    def transcribe(self, wav_path: Path, keywords: list[str] | None = None) -> TranscriptionResult:
        raise RuntimeError("weights on fire at /home/user/secret")


def _serve(engine: InferenceEngine | None = None):
    settings = Settings(host="127.0.0.1", port=0)
    engine = engine or InferenceEngine(
        settings,
        asr_backend=FakeASR(),
        cleanup_backend=FaithfulRuleCleanupBackend(),
    )
    return serve_test_engine(settings, engine)


class OpenAiTranscriptionsRouteTest(unittest.TestCase):
    def test_wav_upload_returns_json_text(self):
        server, thread, host, port = _serve()
        try:
            content_type, body = build_multipart_body(files=[("file", "clip.wav", tiny_wav_bytes())])
            status, headers, raw = http_multipart(host, port, _PATH, content_type=content_type, body=body)
        finally:
            stop_server(server, thread)
        self.assertEqual(status, 200)
        self.assertEqual(headers["content-type"], "application/json")
        # Raw ASR text, NOT the dictation-cleaned "Hello from ROCm."
        self.assertEqual(json.loads(raw.decode("utf-8")), {"text": "hello from ROCm"})

    def test_response_format_text_returns_plain_transcript(self):
        server, thread, host, port = _serve()
        try:
            content_type, body = build_multipart_body(
                fields={"response_format": "text", "model": "whisper-1"},
                files=[("file", "clip.wav", tiny_wav_bytes())],
            )
            status, headers, raw = http_multipart(host, port, _PATH, content_type=content_type, body=body)
        finally:
            stop_server(server, thread)
        self.assertEqual(status, 200)
        self.assertEqual(headers["content-type"], "text/plain; charset=utf-8")
        self.assertEqual(raw, b"hello from ROCm")

    def test_unsupported_response_format_returns_openai_400(self):
        server, thread, host, port = _serve()
        try:
            content_type, body = build_multipart_body(
                fields={"response_format": "srt"},
                files=[("file", "clip.wav", tiny_wav_bytes())],
            )
            status, _headers, raw = http_multipart(host, port, _PATH, content_type=content_type, body=body)
        finally:
            stop_server(server, thread)
        self.assertEqual(status, 400)
        payload = json.loads(raw.decode("utf-8"))
        self.assertEqual(payload["error"]["type"], "invalid_request_error")
        self.assertEqual(payload["error"]["param"], "response_format")
        self.assertIsNone(payload["error"]["code"])

    def test_missing_file_returns_openai_400(self):
        server, thread, host, port = _serve()
        try:
            content_type, body = build_multipart_body(fields={"model": "whisper-1"})
            status, _headers, raw = http_multipart(host, port, _PATH, content_type=content_type, body=body)
        finally:
            stop_server(server, thread)
        self.assertEqual(status, 400)
        payload = json.loads(raw.decode("utf-8"))
        self.assertEqual(payload["error"]["type"], "invalid_request_error")
        self.assertEqual(payload["error"]["param"], "file")

    def test_garbage_audio_returns_openai_400_under_fake_backend(self):
        asr = FakeASR()
        settings = Settings(host="127.0.0.1", port=0)
        engine = InferenceEngine(settings, asr_backend=asr, cleanup_backend=FaithfulRuleCleanupBackend())
        server, thread, host, port = serve_test_engine(settings, engine)
        try:
            content_type, body = build_multipart_body(files=[("file", "clip.wav", b"this is not audio")])
            status, _headers, raw = http_multipart(host, port, _PATH, content_type=content_type, body=body)
        finally:
            stop_server(server, thread)
        self.assertEqual(status, 400)
        payload = json.loads(raw.decode("utf-8"))
        self.assertEqual(payload["error"]["param"], "file")
        # Pre-validation rejected it before the backend was ever invoked.
        self.assertEqual(asr.calls, [])
        # The message is static: libsndfile's exception text embeds the
        # server-local temp path (and with it the OS username) and must never
        # ship on the wire.
        message = payload["error"]["message"]
        self.assertNotIn("transclip-", message)
        self.assertNotIn(tempfile.gettempdir().lower(), message.lower())
        self.assertNotIn(getpass.getuser(), message)

    def test_oversize_upload_gets_a_clean_413_not_a_connection_reset(self):
        # A REAL over-limit upload (body actually sent, matching Content-Length)
        # must receive a clean 413 the client can read — NOT a mid-upload socket
        # reset. The server drains the body before responding; without that drain
        # a Windows RST would surface to the OpenAI SDK as a retryable connection
        # error and re-upload the oversize body. (This is the honest form of the
        # cap test: the header-only variant could never see the reset it dodged.)
        server, thread, host, port = _serve()
        try:
            oversize = b"\x00" * (MAX_UPLOAD_BYTES + 4096)
            content_type, body = build_multipart_body(files=[("file", "big.wav", oversize)])
            conn = http.client.HTTPConnection(host, port, timeout=30)
            try:
                conn.request("POST", _PATH, body=body, headers={"Content-Type": content_type})
                response = conn.getresponse()
                status, raw = response.status, response.read()
            finally:
                conn.close()
        finally:
            stop_server(server, thread)
        self.assertEqual(status, 413)
        payload = json.loads(raw.decode("utf-8"))
        self.assertEqual(payload["error"]["type"], "invalid_request_error")

    def test_missing_content_length_returns_411(self):
        server, thread, host, port = _serve()
        try:
            conn = http.client.HTTPConnection(host, port, timeout=5)
            try:
                # Skip_* keeps http.client from auto-adding a Content-Length header.
                conn.putrequest("POST", _PATH, skip_accept_encoding=True)
                conn.putheader("content-type", "multipart/form-data; boundary=----x")
                conn.endheaders()
                response = conn.getresponse()
                status = response.status
                payload = json.loads(response.read().decode("utf-8"))
            finally:
                conn.close()
        finally:
            stop_server(server, thread)
        self.assertEqual(status, 411)
        self.assertEqual(payload["error"]["type"], "invalid_request_error")

    def test_backend_failure_returns_openai_500_without_internal_leak(self):
        settings = Settings(host="127.0.0.1", port=0)
        engine = InferenceEngine(settings, asr_backend=RaisingASR(), cleanup_backend=FaithfulRuleCleanupBackend())
        server, thread, host, port = serve_test_engine(settings, engine)
        try:
            content_type, body = build_multipart_body(files=[("file", "clip.wav", tiny_wav_bytes())])
            status, _headers, raw = http_multipart(host, port, _PATH, content_type=content_type, body=body)
        finally:
            stop_server(server, thread)
        self.assertEqual(status, 500)
        payload = json.loads(raw.decode("utf-8"))
        # OpenAI-shaped nested error object; the internal {"error": str} /
        # debug_capture_dir shape must NOT reach the wire.
        self.assertIsInstance(payload["error"], dict)
        self.assertEqual(payload["error"]["type"], "api_error")
        self.assertNotIn("debug_capture_dir", payload)
        self.assertNotIn("/home/user/secret", raw.decode("utf-8"))

    def test_foreign_origin_is_rejected_on_new_path(self):
        server, thread, host, port = _serve()
        try:
            content_type, _body = build_multipart_body(files=[("file", "clip.wav", tiny_wav_bytes())])
            # Headers only, no body: the guard rejects before any body read, and
            # an early close with unread body bytes can RST before the client
            # reads the 403 on Windows (same race class as the 413 test).
            conn = http.client.HTTPConnection(host, port, timeout=10)
            try:
                conn.putrequest("POST", _PATH)
                conn.putheader("Content-Type", content_type)
                conn.putheader("Content-Length", "0")
                conn.putheader("Origin", "https://evil.example")
                conn.endheaders()
                status = conn.getresponse().status
            finally:
                conn.close()
        finally:
            stop_server(server, thread)
        self.assertEqual(status, 403)

    def test_preflight_with_authorization_header_succeeds(self):
        server, thread, host, port = _serve()
        try:
            origin = f"http://127.0.0.1:{port}"
            conn = http.client.HTTPConnection(host, port, timeout=5)
            try:
                conn.request(
                    "OPTIONS",
                    _PATH,
                    headers={
                        "origin": origin,
                        "access-control-request-method": "POST",
                        "access-control-request-headers": "authorization, content-type",
                    },
                )
                response = conn.getresponse()
                status = response.status
                headers = {k.lower(): v for k, v in response.getheaders()}
                response.read()
            finally:
                conn.close()
        finally:
            stop_server(server, thread)
        self.assertEqual(status, 204)
        self.assertIn("authorization", headers.get("access-control-allow-headers", "").lower())


if __name__ == "__main__":
    unittest.main()
