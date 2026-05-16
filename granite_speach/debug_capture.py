from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import traceback
from typing import Any

from .settings import Settings


class DebugCapture:
    def __init__(self, settings: Settings):
        self.settings = settings

    def write(
        self,
        wav_path: Path,
        raw_asr: str,
        cleaned: str,
        timings: dict[str, float],
        model_versions: dict[str, str],
    ) -> Path | None:
        if not self.settings.debug_capture:
            return None
        root = self._root()
        if wav_path.exists():
            shutil.copy2(wav_path, root / "audio.wav")
        (root / "raw_asr.txt").write_text(raw_asr, encoding="utf-8")
        (root / "cleaned.txt").write_text(cleaned, encoding="utf-8")
        (root / "timings.json").write_text(json.dumps(timings, indent=2), encoding="utf-8")
        (root / "model_versions.json").write_text(
            json.dumps(_jsonable(model_versions), indent=2),
            encoding="utf-8",
        )
        return root

    def write_error(
        self,
        context: str,
        error: BaseException,
        details: dict[str, Any] | None = None,
    ) -> Path | None:
        if not self.settings.debug_capture:
            return None
        root = self._root()
        payload = {
            "context": context,
            "error_type": type(error).__name__,
            "error": str(error),
            "details": details or {},
        }
        (root / "error.log").write_text(
            "\n".join(
                [
                    f"context: {payload['context']}",
                    f"error_type: {payload['error_type']}",
                    f"error: {payload['error']}",
                    "",
                    "".join(traceback.format_exception(type(error), error, error.__traceback__)),
                ]
            ),
            encoding="utf-8",
        )
        (root / "error.json").write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
        return root

    def _root(self) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        root = Path(self.settings.debug_capture_dir) / stamp
        suffix = 1
        while root.exists():
            root = Path(self.settings.debug_capture_dir) / f"{stamp}-{suffix}"
            suffix += 1
        root.mkdir(parents=True, exist_ok=True)
        return root


def _jsonable(value):
    if is_dataclass(value):
        return asdict(value)
    return value
