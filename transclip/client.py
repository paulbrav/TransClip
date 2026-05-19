from __future__ import annotations

import json
from pathlib import Path
from urllib import request

from .settings import Settings


class InferenceClient:
    def __init__(self, settings: Settings):
        self.base_url = f"http://{settings.host}:{settings.port}"

    def health(self) -> dict:
        return self._get("/health")

    def cleanup(self, text: str) -> dict:
        return self._post("/cleanup", {"text": text})

    def transcribe(self, wav_path: Path, cleanup: bool | None = None) -> dict:
        payload: dict[str, object] = {"audio_path": str(wav_path)}
        if cleanup is not None:
            payload["cleanup"] = cleanup
        return self._post("/transcribe", payload)

    def cleanup_transcribe(self, wav_path: Path) -> dict:
        return self._post("/cleanup/transcribe", {"audio_path": str(wav_path)})

    def record_toggle(self, cleanup: bool | None = None) -> dict:
        payload: dict[str, object] = {}
        if cleanup is not None:
            payload["cleanup"] = cleanup
        return self._post("/record/toggle", payload)

    def record_start(self) -> dict:
        return self._post("/record/start", {})

    def record_stop(
        self,
        cleanup: bool | None = None,
        discard: bool = False,
    ) -> dict:
        payload: dict[str, object] = {"discard": discard}
        if cleanup is not None:
            payload["cleanup"] = cleanup
        return self._post("/record/stop", payload)

    def _get(self, path: str) -> dict:
        with request.urlopen(f"{self.base_url}{path}", timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))

    def _post(self, path: str, payload: dict) -> dict:
        data = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"content-type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
