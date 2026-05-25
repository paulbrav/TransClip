import io
import json
import socket
import tempfile
import unittest
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

from transclip.cli import main
from transclip.daemon.lifecycle import toggle_log_path
from transclip.desktop.paste import PasteResult
from transclip.settings import Settings, write_settings

from tests.service_helpers import FakeRuntime


class FakeToggleClient:
    base_url = "http://127.0.0.1:1"

    def __init__(self, settings, *, responses=None, error=None):
        del settings
        self.responses = responses if responses is not None else []
        self.error = error

    def record_toggle(self):
        if self.error is not None:
            raise self.error
        if not self.responses:
            raise AssertionError("record_toggle called without a queued response")
        return dict(self.responses.pop(0))


def fake_paste_transcript(transcript: str, settings, **kwargs) -> PasteResult:
    return PasteResult(
        copied=True,
        pasted=True,
        injected=True,
        restored=False,
        transcript_left_on_clipboard=True,
        clipboard_backend="fake-clipboard",
        paste_backend="fake-paste",
    )


class CliTests(unittest.TestCase):
    def _linux_runtime(self, root: Path) -> FakeRuntime:
        return FakeRuntime(system="Linux", home=root)

    def _toggle_log_path(self, root: Path) -> Path:
        return toggle_log_path(self._linux_runtime(root))

    def _enter_toggle_patches(self, stack: ExitStack, root: Path, **extra) -> None:
        stack.enter_context(patch("transclip.cli.toggle_cmd.notify"))
        stack.enter_context(patch("transclip.daemon.status.toggle_log_path", return_value=self._toggle_log_path(root)))
        stack.enter_context(patch("transclip.recording_ops.paste_transcript", fake_paste_transcript))
        for key, value in extra.items():
            stack.enter_context(patch(key, value))

    def test_toggle_record_starts_without_paste(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings_path = write_test_settings(root, "127.0.0.1", unused_local_port())
            stdout = io.StringIO()
            with ExitStack() as stack:
                self._enter_toggle_patches(stack, root)
                stack.enter_context(
                    patch(
                        "transclip.recording_ops.InferenceClient",
                        lambda settings: FakeToggleClient(
                            settings,
                            responses=[{"action": "started", "status": "recording"}],
                        ),
                    ),
                )
                stack.enter_context(redirect_stdout(stdout))
                code = main(["--settings", str(settings_path), "toggle-record", "--paste"])

            self.assertEqual(code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["action"], "started")
            self.assertEqual(payload["status"], "recording")
            self.assertNotIn("paste", payload)
            log_event = read_last_toggle_log(root)
            self.assertEqual(log_event["action"], "started")

    def test_toggle_record_stops_and_pastes_transcript(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings_path = write_test_settings(root, "127.0.0.1", unused_local_port())
            stdout = io.StringIO()
            responses = [
                {"action": "started", "status": "recording"},
                {"action": "stopped", "status": "ready", "text": "Hello."},
            ]
            with ExitStack() as stack:
                self._enter_toggle_patches(stack, root)
                stack.enter_context(
                    patch(
                        "transclip.recording_ops.InferenceClient",
                        lambda settings: FakeToggleClient(settings, responses=responses),
                    ),
                )
                stack.enter_context(redirect_stdout(io.StringIO()))
                self.assertEqual(main(["--settings", str(settings_path), "toggle-record"]), 0)
                stack.enter_context(redirect_stdout(stdout))
                code = main(["--settings", str(settings_path), "toggle-record", "--paste"])

            self.assertEqual(code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["action"], "stopped")
            self.assertEqual(payload["text"], "Hello.")
            self.assertTrue(payload["paste"]["pasted"])
            self.assertEqual(payload["paste"]["clipboard_backend"], "fake-clipboard")
            self.assertEqual(payload["paste"]["paste_backend"], "fake-paste")
            log_event = read_last_toggle_log(root)
            self.assertEqual(log_event["action"], "stopped")
            self.assertEqual(log_event["text"], "Hello.")

    def test_toggle_record_reports_unavailable_service(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings_path = write_test_settings(root, "127.0.0.1", unused_local_port())
            stderr = io.StringIO()
            with ExitStack() as stack:
                self._enter_toggle_patches(stack, root)
                stack.enter_context(
                    patch(
                        "transclip.recording_ops.InferenceClient",
                        lambda settings: FakeToggleClient(settings, error=URLError("refused")),
                    ),
                )
                stack.enter_context(redirect_stderr(stderr))
                code = main(["--settings", str(settings_path), "toggle-record", "--paste"])

            self.assertEqual(code, 1)
            self.assertIn("TransClip service is not running", stderr.getvalue())
            log_event = read_last_toggle_log(root)
            self.assertEqual(log_event["action"], "error")
            self.assertEqual(log_event["error"], "TransClip service is not running.")

    def test_toggle_record_log_failure_is_nonfatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".cache").write_text("not a directory", encoding="utf-8")
            settings_path = write_test_settings(root, "127.0.0.1", unused_local_port())
            stdout = io.StringIO()
            with ExitStack() as stack:
                self._enter_toggle_patches(stack, root)
                stack.enter_context(
                    patch(
                        "transclip.recording_ops.InferenceClient",
                        lambda settings: FakeToggleClient(
                            settings,
                            responses=[{"action": "started", "status": "recording"}],
                        ),
                    ),
                )
                stack.enter_context(redirect_stdout(stdout))
                code = main(["--settings", str(settings_path), "toggle-record"])

            self.assertEqual(code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["action"], "started")
            self.assertIn("log_error", payload)
            self.assertIn(".cache", payload["log_error"])

    def test_history_json_and_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings_path = write_test_settings(Path(tmp), "127.0.0.1", unused_local_port())
            stdout = io.StringIO()
            events = [{"text": "latest", "timestamp": "now", "source": "/transcribe"}]
            with (
                patch("transclip.cli.history_cmd.read_history", return_value=events),
                redirect_stdout(stdout),
            ):
                code = main(["--settings", str(settings_path), "history", "--json"])

            self.assertEqual(code, 0)
            self.assertIn('"text": "latest"', stdout.getvalue())

            class Clipboard:
                text = ""

                def write(self, text):
                    type(self).text = text

            stdout = io.StringIO()
            with (
                patch("transclip.cli.history_cmd.read_history", return_value=events),
                patch("transclip.cli.formatting.SystemClipboard", Clipboard),
                redirect_stdout(stdout),
            ):
                code = main(["--settings", str(settings_path), "history", "--copy", "1"])

            self.assertEqual(code, 0)
            self.assertEqual(Clipboard.text, "latest")

    def test_config_get_set_uses_settings_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.toml"
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main(["--settings", str(path), "config", "set", "toggle_cooldown_ms", "250"])
            self.assertEqual(code, 0)
            self.assertIn("toggle_cooldown_ms", stdout.getvalue())

            stdout = io.StringIO()
            with redirect_stdout(stdout):
                code = main(["--settings", str(path), "config", "get", "toggle_cooldown_ms"])
            self.assertEqual(code, 0)
            self.assertEqual(stdout.getvalue().strip(), "250")

    def test_install_gnome_shortcut_uses_configured_hotkey(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings_path = Path(tmp) / "settings.toml"
            write_settings(Settings(hotkey_linux="<Control><Alt>space"), settings_path)
            stdout = io.StringIO()
            shortcut = type(
                "Shortcut",
                (),
                {
                    "name": "TransClip Toggle",
                    "path": "/shortcut/",
                    "binding": "<Control><Alt>space",
                    "command": "/bin/sh -lc transclip",
                },
            )()
            with (
                patch("transclip.cli.shortcut_cmd.install_shortcut", return_value=shortcut) as install,
                redirect_stdout(stdout),
            ):
                code = main(["--settings", str(settings_path), "install-gnome-shortcut"])

        self.assertEqual(code, 0)
        self.assertEqual(install.call_args.kwargs["binding"], "<Control><Alt>space")
        self.assertIn("Binding: <Control><Alt>space", stdout.getvalue())

    def test_tray_command_runs_python_tray(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings_path = write_test_settings(Path(tmp), "127.0.0.1", unused_local_port())
            with patch("transclip.desktop.tray.run_tray", return_value=7):
                code = main(["--settings", str(settings_path), "tray"])

            self.assertEqual(code, 7)


def write_test_settings(root: Path, host: str, port: int, **overrides) -> Path:
    settings = Settings(
        host=host,
        port=port,
        min_recording_ms=0,
        toggle_cooldown_ms=0,
        clipboard_restore_delay_ms=0,
        **overrides,
    )
    path = root / "settings.toml"
    write_settings(settings, path)
    return path


def read_last_toggle_log(root: Path) -> dict:
    log_path = toggle_log_path(FakeRuntime(system="Linux", home=root))
    lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return json.loads(lines[-1])


def unused_local_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


if __name__ == "__main__":
    unittest.main()
