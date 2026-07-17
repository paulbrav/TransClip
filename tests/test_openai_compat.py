import json
import unittest

from transclip.asr import TranscriptionResult
from transclip.service.openai_compat import (
    MultipartError,
    handle_transcriptions,
    openai_error_body,
    parse_multipart,
    sanitize_suffix,
)

from tests.service_helpers import build_multipart_body as build_multipart
from tests.service_helpers import tiny_wav_bytes

try:  # pre-validation is best-effort: only installs with the model extras have soundfile
    import soundfile as _soundfile  # noqa: F401

    _HAVE_SOUNDFILE = True
except ImportError:
    _HAVE_SOUNDFILE = False


class FakeRawEngine:
    def __init__(self, text: str = "raw transcript", error: Exception | None = None):
        self.text = text
        self.error = error
        self.calls: list = []

    def transcribe_raw(self, wav_path) -> TranscriptionResult:
        self.calls.append(wav_path)
        if self.error is not None:
            raise self.error
        return TranscriptionResult(self.text, {"asr": 1.0}, "fake", "fake-model")


class ParseMultipartTests(unittest.TestCase):
    def test_well_formed_extracts_file_bytes_filename_and_fields(self):
        content_type, body = build_multipart(
            fields={"model": "whisper-1", "response_format": "json"},
            files=[("file", "clip.wav", b"AUDIOBYTES")],
        )
        parsed = parse_multipart(content_type, body)
        file_part = parsed.get("file")
        self.assertIsNotNone(file_part)
        self.assertEqual(file_part.content, b"AUDIOBYTES")
        self.assertEqual(file_part.filename, "clip.wav")
        self.assertEqual(parsed.field_text("model"), "whisper-1")
        self.assertEqual(parsed.field_text("response_format"), "json")

    def test_missing_file_field_yields_no_file_part(self):
        content_type, body = build_multipart(fields={"model": "whisper-1"})
        parsed = parse_multipart(content_type, body)
        self.assertIsNone(parsed.get("file"))

    def test_file_part_without_filename_is_still_extracted(self):
        content_type, body = build_multipart(files=[("file", None, b"NOFILENAME")])
        parsed = parse_multipart(content_type, body)
        file_part = parsed.get("file")
        self.assertIsNotNone(file_part)
        self.assertIsNone(file_part.filename)
        self.assertEqual(file_part.content, b"NOFILENAME")

    def test_utf8_filename_is_decoded(self):
        content_type, body = build_multipart(files=[("file", "grüße.wav", b"X")])
        parsed = parse_multipart(content_type, body)
        self.assertEqual(parsed.get("file").filename, "grüße.wav")

    def test_rfc2231_extended_filename_is_decoded(self):
        boundary = b"----transcliptestboundary"
        body = b"--" + boundary + b"\r\n"
        body += b"Content-Disposition: form-data; name=\"file\"; filename*=UTF-8''caf%C3%A9.wav\r\n\r\n"
        body += b"DATA\r\n"
        body += b"--" + boundary + b"--\r\n"
        content_type = "multipart/form-data; boundary=" + boundary.decode("ascii")
        parsed = parse_multipart(content_type, body)
        self.assertEqual(parsed.get("file").filename, "café.wav")

    def test_hostile_filename_survives_parse_and_sanitizes_to_wav(self):
        content_type, body = build_multipart(files=[("file", "../../etc/passwd", b"X")])
        parsed = parse_multipart(content_type, body)
        self.assertEqual(parsed.get("file").filename, "../../etc/passwd")
        self.assertEqual(sanitize_suffix(parsed.get("file").filename), ".wav")

    def test_extra_fields_are_ignored_but_parsed(self):
        content_type, body = build_multipart(
            fields={"model": "m", "language": "en", "prompt": "hi", "temperature": "0"},
            files=[("file", "a.wav", b"X")],
        )
        parsed = parse_multipart(content_type, body)
        self.assertEqual(parsed.field_text("language"), "en")
        self.assertEqual(parsed.field_text("temperature"), "0")

    def test_binary_body_integrity_is_preserved(self):
        blob = bytes(range(256)) * 8
        content_type, body = build_multipart(files=[("file", "raw.bin", blob)])
        parsed = parse_multipart(content_type, body)
        self.assertEqual(parsed.get("file").content, blob)

    def test_payloads_with_trailing_line_endings_round_trip_exactly(self):
        # The classic multipart corruption bug is a trailing-CRLF strip; pin
        # payloads ending in \r\n, bare \r, bare \n, and \r\n\r\n byte-for-byte.
        for tail in (b"\r\n", b"\r", b"\n", b"\r\n\r\n"):
            with self.subTest(tail=tail):
                blob = b"RIFFdata" + tail
                content_type, body = build_multipart(files=[("file", "raw.bin", blob)])
                parsed = parse_multipart(content_type, body)
                self.assertEqual(parsed.get("file").content, blob)

    def test_client_set_content_transfer_encoding_does_not_corrupt_binary(self):
        # A stray `Content-Transfer-Encoding: base64` on the file part must not
        # make email base64-DECODE an already-raw binary audio payload. The
        # parser strips per-part CTE so the bytes round-trip verbatim. (RFC 7578
        # deprecates CTE in form-data; real SDK clients never send it, but a
        # corruption path is worth pinning shut.)
        raw = bytes(range(256)) + b"\r\nRIFF\x00\xff"
        boundary = "----ctestboundary"
        body = (
            f"--{boundary}\r\n".encode()
            + b'Content-Disposition: form-data; name="file"; filename="a.wav"\r\n'
            + b"Content-Transfer-Encoding: base64\r\n\r\n"
            + raw
            + f"\r\n--{boundary}--\r\n".encode()
        )
        parsed = parse_multipart(f"multipart/form-data; boundary={boundary}", body)
        self.assertEqual(parsed.get("file").content, raw)

    def test_non_multipart_content_type_raises(self):
        with self.assertRaises(MultipartError):
            parse_multipart("application/json", b"{}")

    def test_boundary_with_preamble_and_epilogue(self):
        boundary = b"----transcliptestboundary"
        body = b"this is ignored preamble\r\n"
        body += b"--" + boundary + b"\r\n"
        body += b'Content-Disposition: form-data; name="file"; filename="a.wav"\r\n\r\n'
        body += b"PAYLOAD\r\n"
        body += b"--" + boundary + b"--\r\n"
        body += b"trailing epilogue is ignored\r\n"
        content_type = "multipart/form-data; boundary=" + boundary.decode("ascii")
        parsed = parse_multipart(content_type, body)
        self.assertEqual(parsed.get("file").content, b"PAYLOAD")


class SanitizeSuffixTests(unittest.TestCase):
    def test_none_filename_defaults_to_wav(self):
        self.assertEqual(sanitize_suffix(None), ".wav")

    def test_empty_filename_defaults_to_wav(self):
        self.assertEqual(sanitize_suffix(""), ".wav")

    def test_plain_wav_suffix_is_kept(self):
        self.assertEqual(sanitize_suffix("clip.wav"), ".wav")

    def test_flac_suffix_is_kept(self):
        self.assertEqual(sanitize_suffix("clip.flac"), ".flac")

    def test_no_suffix_defaults_to_wav(self):
        self.assertEqual(sanitize_suffix("noext"), ".wav")

    def test_suffix_with_hostile_characters_defaults_to_wav(self):
        self.assertEqual(sanitize_suffix("evil.w!v"), ".wav")

    def test_overlong_suffix_defaults_to_wav(self):
        self.assertEqual(sanitize_suffix("clip.superlongextension"), ".wav")


class OpenAiErrorBodyTests(unittest.TestCase):
    def test_error_body_has_exact_openai_shape(self):
        raw = openai_error_body("bad", error_type="invalid_request_error", param="file")
        payload = json.loads(raw.decode("utf-8"))
        self.assertEqual(set(payload), {"error"})
        self.assertEqual(set(payload["error"]), {"message", "type", "param", "code"})
        self.assertEqual(payload["error"]["message"], "bad")
        self.assertEqual(payload["error"]["type"], "invalid_request_error")
        self.assertEqual(payload["error"]["param"], "file")
        self.assertIsNone(payload["error"]["code"])

    def test_error_body_param_defaults_to_null(self):
        payload = json.loads(openai_error_body("boom", error_type="api_error").decode("utf-8"))
        self.assertIsNone(payload["error"]["param"])
        self.assertEqual(payload["error"]["type"], "api_error")


class HandleTranscriptionsTests(unittest.TestCase):
    def test_json_success_returns_text_payload(self):
        engine = FakeRawEngine(text="hello raw")
        content_type, body = build_multipart(files=[("file", "clip.wav", tiny_wav_bytes())])
        status, response_content_type, response_body = handle_transcriptions(engine, content_type, body)
        self.assertEqual(status, 200)
        self.assertEqual(response_content_type, "application/json")
        self.assertEqual(json.loads(response_body.decode("utf-8")), {"text": "hello raw"})
        self.assertEqual(len(engine.calls), 1)

    def test_text_response_format_returns_bare_transcript(self):
        engine = FakeRawEngine(text="bare text out")
        content_type, body = build_multipart(
            fields={"response_format": "text"},
            files=[("file", "clip.wav", tiny_wav_bytes())],
        )
        status, response_content_type, response_body = handle_transcriptions(engine, content_type, body)
        self.assertEqual(status, 200)
        self.assertEqual(response_content_type, "text/plain; charset=utf-8")
        self.assertEqual(response_body, b"bare text out")

    def test_unsupported_response_format_is_rejected(self):
        engine = FakeRawEngine()
        content_type, body = build_multipart(
            fields={"response_format": "verbose_json"},
            files=[("file", "clip.wav", tiny_wav_bytes())],
        )
        status, response_content_type, response_body = handle_transcriptions(engine, content_type, body)
        self.assertEqual(status, 400)
        self.assertEqual(response_content_type, "application/json")
        payload = json.loads(response_body.decode("utf-8"))
        self.assertEqual(payload["error"]["type"], "invalid_request_error")
        self.assertEqual(payload["error"]["param"], "response_format")
        self.assertEqual(engine.calls, [])

    def test_missing_file_is_rejected(self):
        engine = FakeRawEngine()
        content_type, body = build_multipart(fields={"model": "whisper-1"})
        status, _content_type, response_body = handle_transcriptions(engine, content_type, body)
        self.assertEqual(status, 400)
        payload = json.loads(response_body.decode("utf-8"))
        self.assertEqual(payload["error"]["type"], "invalid_request_error")
        self.assertEqual(payload["error"]["param"], "file")
        self.assertEqual(engine.calls, [])

    def test_empty_file_is_rejected(self):
        engine = FakeRawEngine()
        content_type, body = build_multipart(files=[("file", "clip.wav", b"")])
        status, _content_type, response_body = handle_transcriptions(engine, content_type, body)
        self.assertEqual(status, 400)
        payload = json.loads(response_body.decode("utf-8"))
        self.assertEqual(payload["error"]["param"], "file")
        self.assertEqual(engine.calls, [])

    @unittest.skipUnless(_HAVE_SOUNDFILE, "pre-validation (and its 400) requires soundfile")
    def test_garbage_audio_is_rejected_before_backend(self):
        engine = FakeRawEngine()
        content_type, body = build_multipart(files=[("file", "clip.wav", b"not audio at all")])
        status, _content_type, response_body = handle_transcriptions(engine, content_type, body)
        self.assertEqual(status, 400)
        payload = json.loads(response_body.decode("utf-8"))
        self.assertEqual(payload["error"]["type"], "invalid_request_error")
        self.assertEqual(payload["error"]["param"], "file")
        # Pre-validation must run in the route, before the backend is touched.
        self.assertEqual(engine.calls, [])

    def test_backend_failure_returns_api_error_without_internal_shape(self):
        engine = FakeRawEngine(error=RuntimeError("weights on fire at /home/user/secret"))
        content_type, body = build_multipart(files=[("file", "clip.wav", tiny_wav_bytes())])
        status, response_content_type, response_body = handle_transcriptions(engine, content_type, body)
        self.assertEqual(status, 500)
        self.assertEqual(response_content_type, "application/json")
        payload = json.loads(response_body.decode("utf-8"))
        # OpenAI-shaped nested error object, NOT the internal {"error": str} shape.
        self.assertIsInstance(payload["error"], dict)
        self.assertEqual(payload["error"]["type"], "api_error")
        self.assertIsNone(payload["error"]["param"])
        self.assertIsNone(payload["error"]["code"])

    def test_non_multipart_body_is_rejected(self):
        engine = FakeRawEngine()
        status, _content_type, response_body = handle_transcriptions(engine, "application/json", b"{}")
        self.assertEqual(status, 400)
        payload = json.loads(response_body.decode("utf-8"))
        self.assertEqual(payload["error"]["type"], "invalid_request_error")
        self.assertEqual(engine.calls, [])


if __name__ == "__main__":
    unittest.main()
