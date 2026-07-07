"""Incremental pre-transcription: commit-and-trim audio at pauses during recording.

Reduces release-to-paste latency for long recordings by transcribing committed
audio in the background while the user is still speaking, so finish() only has
to process a small residual tail. Committed text can never change because the
committed audio is physically trimmed from the buffer (text-agreement commits
were measured unsafe; see plans/2026-06-12-streaming-investigation-report.md).
"""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Callable
from typing import Any

from transclip.asr import GRANITE_NAR_BUCKET_SECONDS, TranscriptionResult
from transclip.asr_streaming import PartialTranscript
from transclip.audio import pcm16_to_float32
from transclip.platform.runtime import PlatformRuntime
from transclip.settings import Settings

# ROCm/Triton recompiles novel tensor shapes at 1.9-8.7 s; padding audio with
# trailing silence to fixed bucket multiples keeps pass times stable.
BUCKET_S = GRANITE_NAR_BUCKET_SECONDS
# Never commit into the most recent audio; the model needs trailing context.
COMMIT_MIN_TAIL_S = 2.0
# Only cut at silences so a commit boundary cannot split a word.
SILENCE_MIN_DUR_S = 0.4
SILENCE_RMS_DBFS = -40.0
# A hot/noisy mic can sit above the fixed floor forever; fall back to a
# threshold relative to the quietest frames so commits still happen. The
# speech-gap cap keeps the threshold below the loud frames so uniform audio
# (or pure speech) is never classified as silence.
NOISE_FLOOR_MARGIN_DB = 10.0
SPEECH_GAP_DB = 15.0
FRAME_MS = 20
# Skip pointless passes over tiny or speech-free segments.
MIN_COMMIT_S = 2.0

logger = logging.getLogger(__name__)

TranscribeWaveform = Callable[[Any], TranscriptionResult]


def incremental_transcription_enabled(settings: Settings, runtime: PlatformRuntime | None = None) -> bool:
    if not settings.incremental_transcription:
        return False
    from transclip.platform.profiles import detect_runtime_profile

    return detect_runtime_profile(runtime).incremental_transcription_supported


class IncrementalNarSession:
    """StreamingASRSession over a one-shot waveform transcriber (e.g. Granite NAR).

    feed() is called from the audio capture thread and only appends bytes.
    A single worker thread commits audio at silence boundaries once the
    uncommitted buffer exceeds the threshold. finish() drains the worker and
    transcribes the residual; short utterances never wake the worker, so they
    take exactly one pass, identical to the batch path.
    """

    def __init__(
        self,
        transcribe_waveform: TranscribeWaveform,
        *,
        sample_rate: int = 16000,
        commit_threshold_s: float = 10.0,
        backend_name: str = "incremental",
        model_name: str = "",
    ):
        self._transcribe = transcribe_waveform
        self._sample_rate = sample_rate
        self._threshold_bytes = int(commit_threshold_s * sample_rate) * 2
        self._tail_bytes = int(COMMIT_MIN_TAIL_S * sample_rate) * 2
        self._min_commit_bytes = int(MIN_COMMIT_S * sample_rate) * 2
        self._backend_name = backend_name
        self._model_name = model_name
        self._buffer = bytearray()
        self._committed: list[str] = []
        self._committed_pass_ms = 0.0
        self._commit_count = 0
        self._closing = False
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="transclip-incremental-asr",
            daemon=True,
        )
        self._worker.start()

    @property
    def partial_text(self) -> PartialTranscript:
        with self._lock:
            return PartialTranscript(" ".join(self._committed))

    def feed(self, pcm16_mono: bytes) -> None:
        with self._cond:
            if self._closing:
                return
            self._buffer.extend(pcm16_mono)
            if len(self._buffer) >= self._threshold_bytes:
                self._cond.notify()

    def finish(self) -> TranscriptionResult:
        self._shutdown_worker()
        with self._lock:
            residual = bytes(self._buffer)
            self._buffer.clear()
            committed = list(self._committed)
            committed_ms = self._committed_pass_ms
            commits = self._commit_count
        timings: dict[str, float] = {"asr": 0.0}
        texts = committed
        if residual:
            result = self._transcribe(_pad_to_bucket(pcm16_to_float32(residual), self._sample_rate))
            timings["asr"] = result.timings_ms.get("asr", 0.0)
            if result.text.strip():
                texts = [*committed, result.text.strip()]
        if commits:
            timings["asr_committed_total"] = committed_ms
            timings["commits"] = float(commits)
        return TranscriptionResult(" ".join(texts).strip(), timings, self._backend_name, self._model_name)

    def close(self) -> None:
        self._shutdown_worker()
        with self._lock:
            self._buffer.clear()

    def _shutdown_worker(self) -> None:
        with self._cond:
            self._closing = True
            self._cond.notify_all()
        if self._worker.is_alive():
            self._worker.join()

    def _worker_loop(self) -> None:
        bytes_per_s = self._sample_rate * 2
        wait_above = self._threshold_bytes
        while True:
            with self._cond:
                while not self._closing and len(self._buffer) < wait_above:
                    self._cond.wait()
                if self._closing:
                    return
                snapshot = bytes(self._buffer)
            cut, has_speech = _find_commit_cut(
                snapshot,
                sample_rate=self._sample_rate,
                tail_bytes=self._tail_bytes,
                min_commit_bytes=self._min_commit_bytes,
            )
            if cut is None:
                # No usable pause yet; retry once ~1 s more audio arrives.
                wait_above = len(snapshot) + bytes_per_s
                continue
            wait_above = self._threshold_bytes
            text = ""
            pass_ms = 0.0
            if has_speech:
                try:
                    result = self._transcribe(
                        _pad_to_bucket(pcm16_to_float32(snapshot[:cut]), self._sample_rate)
                    )
                except Exception:
                    # No audio was trimmed, so finish() still transcribes the
                    # full buffer in one batch pass — only the latency benefit
                    # is lost. Log it; a silent worker death is undebuggable.
                    logger.exception(
                        "Incremental commit pass failed; worker stopped, finish() will run a full batch pass"
                    )
                    return
                text = result.text.strip()
                pass_ms = result.timings_ms.get("asr", 0.0)
            with self._cond:
                if text:
                    self._committed.append(text)
                del self._buffer[:cut]
                self._committed_pass_ms += pass_ms
                self._commit_count += 1
                if self._closing:
                    return


def _pad_to_bucket(samples: Any, sample_rate: int) -> Any:
    import numpy as np

    bucket = int(BUCKET_S * sample_rate)
    if len(samples) == 0 or len(samples) % bucket == 0:
        return samples
    target = math.ceil(len(samples) / bucket) * bucket
    padded = np.zeros(target, dtype=np.float32)
    padded[: len(samples)] = samples
    return padded


def _frame_dbfs(samples: Any, frame_samples: int) -> Any | None:
    """Per-frame dBFS for float32 mono samples; None if less than one frame."""
    import numpy as np

    frame_count = len(samples) // frame_samples
    if frame_count == 0:
        return None
    frames = samples[: frame_count * frame_samples].reshape(frame_count, frame_samples)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    return 20.0 * np.log10(rms + 1e-10)


def _silence_threshold(dbfs: Any) -> float:
    """Adaptive silence threshold: fixed floor, relaxed toward the noise floor."""
    import numpy as np

    relative_floor = min(
        float(np.percentile(dbfs, 10)) + NOISE_FLOOR_MARGIN_DB,
        float(np.percentile(dbfs, 90)) - SPEECH_GAP_DB,
    )
    return max(SILENCE_RMS_DBFS, relative_floor)


def _find_commit_cut(
    snapshot: bytes,
    *,
    sample_rate: int,
    tail_bytes: int,
    min_commit_bytes: int,
) -> tuple[int | None, bool]:
    """Find the latest silence midpoint to cut at, within the eligible region.

    Returns (byte offset or None, segment-contains-speech). The offset is
    even (whole PCM16 samples) and leaves at least the tail uncommitted.
    """
    import numpy as np

    eligible_bytes = len(snapshot) - tail_bytes
    if eligible_bytes < min_commit_bytes:
        return None, False
    frame_samples = sample_rate * FRAME_MS // 1000
    samples = np.frombuffer(snapshot[:eligible_bytes], dtype=np.int16).astype(np.float32) / 32768.0
    dbfs = _frame_dbfs(samples, frame_samples)
    if dbfs is None:
        return None, False
    frame_count = len(dbfs)
    silent = dbfs < _silence_threshold(dbfs)
    min_run = max(1, int(SILENCE_MIN_DUR_S * 1000 / FRAME_MS))

    run_end = None
    run_len = 0
    best_mid_frame = None
    for index in range(frame_count):
        if silent[index]:
            run_len += 1
            run_end = index
            if run_len >= min_run:
                best_mid_frame = run_end - run_len // 2
        else:
            run_len = 0
    if best_mid_frame is None:
        return None, False
    cut = (best_mid_frame * frame_samples) * 2
    if cut < min_commit_bytes:
        return None, False
    # Only skip transcription when the whole segment is silence-classified;
    # a wasted pass on noise is cheap, dropping real speech is not.
    has_speech = bool(np.any(~silent[:best_mid_frame]))
    return cut, has_speech
