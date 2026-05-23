import json
import threading
from collections.abc import Callable, Iterable
from pathlib import Path
from urllib import request

from transclip.asr import TranscriptionResult
from transclip.cleanup import FaithfulRuleCleanupBackend
from transclip.service import InferenceEngine, create_server
from transclip.settings import Settings
from transclip.text_generation import TextGenerationResult


class FakeASR:
    name = "fake"
    model = "fake-model"

    def __init__(self, text: str = "hello from ROCm"):
        self.text = text

    def transcribe(self, wav_path: Path, keywords: list[str] | None = None) -> TranscriptionResult:
        self.wav_path = wav_path
        self.keywords = keywords
        return TranscriptionResult(self.text, {"asr": 1.0}, self.name, self.model)


class FakeRecorder:
    def __init__(self, settings):
        self.settings = settings
        self.started = False

    def start(self):
        self.started = True

    def stop_to_wav(self, output_path: Path):
        output_path.write_bytes(b"not really wav")
        return output_path


class FakeTextBackend:
    def __init__(
        self,
        responses: str | list[str] | None = None,
        *,
        name: str = "fake-text",
        model_name: str = "fake-model",
    ):
        if responses is None:
            normalized = ["model output"]
        elif isinstance(responses, str):
            normalized = [responses]
        else:
            normalized = list(responses)
        self.responses = normalized
        self.name = name
        self.model_name = model_name
        self.messages: list[list[dict[str, str]]] = []
        self.max_new_tokens: list[int] = []

    def generate(self, messages: list[dict[str, str]], *, max_new_tokens: int) -> TextGenerationResult:
        self.messages.append(messages)
        self.max_new_tokens.append(max_new_tokens)
        text = self.responses.pop(0) if self.responses else "model output"
        return TextGenerationResult(text, {"text_generation": 1.0}, self.name, self.model_name)


class FakeRuntime:
    def __init__(
        self,
        system: str = "Linux",
        home: Path | None = None,
        env: dict[str, str] | None = None,
        available: Iterable[str] | dict[str, str | None] = (),
        run_func: Callable | None = None,
        check_output_text: str = "",
    ):
        self._system = system
        self._home = home or Path("/home/test")
        self.env = env or {}
        self.available = available
        self.run_func = run_func
        self.check_output_text = check_output_text
        self.run_calls: list[list[str]] = []

    def system(self) -> str:
        return self._system

    def home_dir(self) -> Path:
        return self._home

    def environ(self, name: str, default: str | None = None) -> str | None:
        return self.env.get(name, default)

    def env_snapshot(self, names=None) -> dict[str, str]:
        names = names or tuple(self.env)
        return {name: self.env.get(name, "") for name in names}

    def which(self, program: str) -> str | None:
        if isinstance(self.available, dict):
            return self.available.get(program)
        return f"/usr/bin/{program}" if program in self.available else None

    def run(self, command: list[str], **kwargs):
        self.run_calls.append(command)
        if self.run_func:
            return self.run_func(command, **kwargs)
        return type("Completed", (), {"returncode": 0, "stdout": ""})()

    def check_output(self, command: list[str], **kwargs) -> str:
        return self.check_output_text


def linux_gpu_runtime(home: Path | None = None) -> FakeRuntime:
    return FakeRuntime(system="Linux", home=home or Path("/home/user"))


def patch_linux_gpu_runtime(home: Path | None = None):
    from contextlib import ExitStack, contextmanager
    from unittest.mock import patch

    runtime = linux_gpu_runtime(home)

    def _runtime(runtime_override=None):
        return runtime if runtime_override is None else runtime_override

    get_runtime_targets = (
        "transclip.platform.runtime.get_runtime",
        "transclip.platform.profiles.get_runtime",
        "transclip.models.get_runtime",
        "transclip.settings.get_runtime",
        "transclip.cli_commands.get_runtime",
    )

    @contextmanager
    def _patch():
        with ExitStack() as stack:
            for target in get_runtime_targets:
                stack.enter_context(patch(target, side_effect=_runtime))
            stack.enter_context(patch("transclip.platform.profiles.machine_architecture", return_value="x86_64"))
            yield runtime

    return _patch()


def serve_test_engine(
    settings: Settings | None = None,
    engine: InferenceEngine | None = None,
    transcript: str = "hello from ROCm",
) -> tuple[object, threading.Thread, str, int]:
    settings = settings or Settings(
        host="127.0.0.1",
        port=0,
        min_recording_ms=0,
        toggle_cooldown_ms=0,
    )
    engine = engine or InferenceEngine(
        settings,
        asr_backend=FakeASR(transcript),
        cleanup_backend=FaithfulRuleCleanupBackend(),
    )
    server = create_server(settings, engine)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return server, thread, host, port


def stop_server(server, thread: threading.Thread) -> None:
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def http_json(method: str, url: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = request.Request(
        url,
        data=data,
        headers={"content-type": "application/json"},
        method=method,
    )
    try:
        with request.urlopen(req, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception as exc:
        if hasattr(exc, "read"):
            return json.loads(exc.read().decode("utf-8"))
        raise
