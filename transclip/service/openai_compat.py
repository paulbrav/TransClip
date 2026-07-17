"""OpenAI-compatible ``POST /v1/audio/transcriptions`` route.

This module is the entire compat surface: a binary-safe multipart parser (stdlib
``email`` — ``cgi`` is gone in 3.13 and ``requires-python >=3.12``), the OpenAI
error/response serializers, and the pure route handler ``handle_transcriptions``.
Everything here is a pure function with an injectable engine, so the whole route
is unit-testable without sockets.

Threading/failure model: ``handle_transcriptions`` catches every exception and
always returns an OpenAI-shaped body — the compat route must NEVER surface the
internal ``{"error": str, "debug_capture_dir": ...}`` shape used by the other
routes. A single module-level lock serializes only the backend call (N concurrent
OpenAI clients must not stampede an unlocked model); it deliberately does not
serialize against ``/transcribe`` or dictation (pre-existing upstream posture).
"""

from __future__ import annotations

import json
import logging
import re
import tempfile
import threading
from dataclasses import dataclass
from email.parser import BytesParser
from email.policy import default as _EMAIL_POLICY
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from transclip.asr import TranscriptionResult

_log = logging.getLogger(__name__)

# OpenAI's own documented upload limit (25 MiB). Pinned to the exact byte count so
# the 413 boundary test is deterministic; this doubles as the memory-safety cap the
# server otherwise lacks. Fixed constant, no knob.
MAX_UPLOAD_BYTES = 26_214_400

# A tempfile suffix derived from an uploaded filename must never carry a hostile
# path/extension into the NamedTemporaryFile call. Keep only a short, benign suffix.
_SAFE_SUFFIX_RE = re.compile(r"[A-Za-z0-9.]{1,8}")

_SUPPORTED_RESPONSE_FORMATS = frozenset({"json", "text"})

# Serializes only the backend call for this route (see module docstring).
_BACKEND_LOCK = threading.Lock()

_CONTENT_TYPE_JSON = "application/json"
_CONTENT_TYPE_TEXT = "text/plain; charset=utf-8"


class RawTranscriber(Protocol):
    def transcribe_raw(self, wav_path: Path) -> TranscriptionResult: ...


class MultipartError(ValueError):
    """The request body is not a decodable multipart/form-data payload."""


@dataclass(frozen=True, slots=True)
class MultipartPart:
    name: str
    filename: str | None
    content: bytes


@dataclass(frozen=True, slots=True)
class ParsedMultipart:
    parts: tuple[MultipartPart, ...]

    def get(self, name: str) -> MultipartPart | None:
        for part in self.parts:
            if part.name == name:
                return part
        return None

    def field_text(self, name: str) -> str | None:
        part = self.get(name)
        if part is None:
            return None
        return part.content.decode("utf-8", errors="replace")


def parse_multipart(content_type: str, body: bytes) -> ParsedMultipart:
    """Parse a multipart/form-data body into its named parts.

    Uses the modern ``email.policy.default`` parser: binary-safe
    (``get_payload(decode=True)`` round-trips the raw bytes) and it decodes both
    RFC 2231 extended filenames and raw-UTF-8 filename parameters (as browsers
    actually send) to proper unicode. Raises ``MultipartError`` if the payload is
    not a decodable multipart body.
    """
    if "multipart/form-data" not in content_type.lower():
        raise MultipartError("Content-Type is not multipart/form-data")
    header_block = b"Content-Type: " + content_type.encode("latin-1", "replace") + b"\r\n\r\n"
    message = BytesParser(policy=_EMAIL_POLICY).parsebytes(header_block + body)
    if not message.is_multipart():
        raise MultipartError("Request body is not a valid multipart/form-data payload")
    parts: list[MultipartPart] = []
    for part in message.iter_parts():
        if part.get("content-disposition") is None:
            continue
        name = part.get_param("name", header="content-disposition")
        if name is None:
            continue
        filename = part.get_filename()
        # A multipart FILE part carries raw bytes; RFC 7578 §4.7 deprecates
        # Content-Transfer-Encoding in form-data. Strip any the client set so
        # get_payload(decode=True) returns the bytes verbatim — otherwise a
        # stray `Content-Transfer-Encoding: base64` header makes email decode
        # (corrupt) an already-raw binary audio payload.
        del part["content-transfer-encoding"]
        payload = part.get_payload(decode=True)
        parts.append(
            MultipartPart(
                name=str(name),
                filename=str(filename) if filename is not None else None,
                content=payload if isinstance(payload, bytes) else b"",
            )
        )
    return ParsedMultipart(parts=tuple(parts))


def sanitize_suffix(filename: str | None) -> str:
    """Return a safe tempfile suffix from an uploaded filename.

    ``Path(name).suffix`` only, accepted when it matches ``[A-Za-z0-9.]{1,8}``;
    anything else (missing, hostile, overlong) falls back to ``.wav``.
    """
    if not filename:
        return ".wav"
    suffix = Path(filename).suffix
    if suffix and _SAFE_SUFFIX_RE.fullmatch(suffix):
        return suffix
    return ".wav"


def openai_error_body(
    message: str,
    *,
    error_type: str,
    param: str | None = None,
    code: str | None = None,
) -> bytes:
    """Serialize the exact OpenAI error envelope for this route."""
    return json.dumps(
        {
            "error": {
                "message": message,
                "type": error_type,
                "param": param,
                "code": code,
            }
        }
    ).encode("utf-8")


def _error(
    status: int,
    message: str,
    *,
    error_type: str = "invalid_request_error",
    param: str | None = None,
) -> tuple[int, str, bytes]:
    return status, _CONTENT_TYPE_JSON, openai_error_body(message, error_type=error_type, param=param)


def handle_transcriptions(
    engine: RawTranscriber,
    content_type: str | None,
    body: bytes,
) -> tuple[int, str, bytes]:
    """Handle a compat transcription request.

    Returns ``(status, content_type, body_bytes)``. Content-Length limits (411/413)
    are enforced by the server before this is called; here we parse, validate, and
    run raw ASR. Every failure path returns an OpenAI-shaped body; the ``except``
    guarantees the internal error shape can never leak.

    The standard OpenAI fields ``model``/``language``/``prompt``/``temperature`` are
    parsed but IGNORED: this service transcribes with its one configured backend, so
    there is no model to select and no decoding-hint plumbing (the local-server norm).
    ``response_format`` is honored for ``json``/``text`` only.
    """
    try:
        if not content_type or "multipart/form-data" not in content_type.lower():
            return _error(400, "Request must be multipart/form-data.")
        try:
            parsed = parse_multipart(content_type, body)
        except MultipartError as exc:
            return _error(400, f"Could not parse multipart request: {exc}")

        response_format = (parsed.field_text("response_format") or "json").strip()
        if response_format not in _SUPPORTED_RESPONSE_FORMATS:
            # Truncate the echoed value: an attacker-supplied field can be up to
            # the whole upload, and reflecting it unbounded is a needless amplifier.
            shown = response_format[:32]
            return _error(
                400,
                f"Unsupported response_format {shown!r}; supported: json, text.",
                param="response_format",
            )

        file_part = parsed.get("file")
        if file_part is None or not file_part.content:
            return _error(400, "No audio file was provided in the 'file' field.", param="file")

        return _transcribe_upload(engine, file_part, response_format)
    except Exception:
        # Opaque api_error on the WIRE (never surface internal exception text —
        # paths, the internal {"error": str} shape), but log server-side so an
        # operator debugging a 500 has the traceback the sibling do_POST path
        # gets via debug_capture. Opaque-to-client, not opaque-to-operator.
        _log.exception("OpenAI-compat transcription route failed")
        return (
            500,
            _CONTENT_TYPE_JSON,
            openai_error_body(
                "The transcription backend failed to process the request.",
                error_type="api_error",
            ),
        )


def _transcribe_upload(
    engine: RawTranscriber,
    file_part: MultipartPart,
    response_format: str,
) -> tuple[int, str, bytes]:
    # Best-effort pre-validation: soundfile ships with every extra that can
    # actually transcribe (models/mlx/openvino) but is not a core dependency,
    # so a fake-backend install (CI, the file: test backend) must not 500 on
    # import. Without soundfile, undecodable audio surfaces as the backend's
    # own failure instead of the friendly 400.
    try:
        import soundfile as sf
    except ImportError:
        sf = None

    suffix = sanitize_suffix(file_part.filename)
    with tempfile.NamedTemporaryFile(prefix="transclip-", suffix=suffix, delete=False) as handle:
        # Bind the path BEFORE the write so a mid-write failure (e.g. disk full)
        # still reaches the finally-unlink instead of leaking the partial temp.
        wav_path = Path(handle.name)
        handle.write(file_part.content)
    try:
        # Pre-validate in the route so undecodable audio is a real, testable 400
        # (reachable even under a fake backend) instead of a 500 from deep in ASR.
        if sf is not None:
            try:
                sf.info(str(wav_path))
            except Exception:
                # Static message, same discipline as the 500 path: libsndfile's
                # exception text embeds the server-local temp path (and with it
                # the OS username) — never echo internal exception text on the wire.
                return _error(
                    400,
                    "Could not decode the uploaded audio; supported formats include WAV, FLAC and OGG.",
                    param="file",
                )
        with _BACKEND_LOCK:
            result = engine.transcribe_raw(wav_path)
        text = result.text
    finally:
        wav_path.unlink(missing_ok=True)

    if response_format == "text":
        return 200, _CONTENT_TYPE_TEXT, text.encode("utf-8")
    return 200, _CONTENT_TYPE_JSON, json.dumps({"text": text}).encode("utf-8")
