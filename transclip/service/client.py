from __future__ import annotations

import json
from pathlib import Path
from typing import cast
from urllib import request

from transclip.settings import Settings

from .json_response import json_object_response
from .types import (
    RecordSessionResponse,
    ServiceHealthResponse,
    TranscribeResponse,
)


class InferenceClient:
    def __init__(self, settings: Settings):
        self.base_url = f"http://{settings.host}:{settings.port}"

    def health(self) -> ServiceHealthResponse:
        return json_object_response(self._get("/health"))

    def transcribe(self, wav_path: Path, cleanup: bool | None = None) -> TranscribeResponse:
        payload: dict[str, object] = {"audio_path": str(wav_path)}
        if cleanup is not None:
            payload["cleanup"] = cleanup
        return json_object_response(self._post("/transcribe", payload))

    def cleanup_transcribe(self, wav_path: Path) -> TranscribeResponse:
        return json_object_response(
            self._post("/cleanup/transcribe", {"audio_path": str(wav_path)}),
        )

    def record_toggle(self, cleanup: bool | None = None) -> RecordSessionResponse:
        payload: dict[str, object] = {}
        if cleanup is not None:
            payload["cleanup"] = cleanup
        return json_object_response(self._post("/record/toggle", payload))

    def record_start(self) -> RecordSessionResponse:
        return json_object_response(self._post("/record/start", {}))

    def record_stop(
        self,
        cleanup: bool | None = None,
        discard: bool = False,
    ) -> RecordSessionResponse:
        payload: dict[str, object] = {"discard": discard}
        if cleanup is not None:
            payload["cleanup"] = cleanup
        return json_object_response(self._post("/record/stop", payload))

    def _get(self, path: str) -> dict[str, object]:
        with request.urlopen(f"{self.base_url}{path}", timeout=5) as response:
            return cast(dict[str, object], json.loads(response.read().decode("utf-8")))

    def _post(self, path: str, payload: dict[str, object]) -> dict[str, object]:
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"content-type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=120) as response:
            return cast(dict[str, object], json.loads(response.read().decode("utf-8")))
