import unittest
from pathlib import Path
from unittest.mock import patch

from transclip.desktop.paste import (
    SystemPasteInjector,
    clipboard_capability,
    detect_clipboard_backend,
    paste_capability,
    paste_commands,
    paste_transcript,
)
from transclip.desktop.paste.platform import (
    GUI_PASTE_SHORTCUT,
    TERMINAL_PASTE_SHORTCUT,
    WTYPE_TERMINAL_PASTE_COMMAND,
    YDOTOOL_TERMINAL_SEQUENTIAL_COMMAND,
    build_paste_commands_for_backend,
    paste_specs,
    resolve_paste_shortcut,
)
from transclip.platform.capabilities import session_info
from transclip.settings import Settings

from tests.service_helpers import FakeRuntime


class FakeClipboard:
    backend_name = "fake-clipboard"

    def __init__(self, initial: str):
        self.value = initial

    def read(self) -> str:
        return self.value

    def write(self, text: str) -> None:
        self.value = text


class FakeInjector:
    def __init__(self, pasted: bool = True):
        self.pasted = pasted
        self.backend_name = "fake-paste" if pasted else None

    def paste(self) -> bool:
        return self.pasted


class PasteTests(unittest.TestCase):
    def test_restores_clipboard_when_safe(self):
        settings = Settings(clipboard_restore_delay_ms=0, restore_clipboard_after_paste=True)
        clipboard = FakeClipboard("prior")
        result = paste_transcript("transcript", settings, clipboard, FakeInjector())

        self.assertTrue(result.copied)
        self.assertTrue(result.pasted)
        self.assertTrue(result.injected)
        self.assertTrue(result.restored)
        self.assertEqual(clipboard.value, "prior")
        self.assertEqual(result.clipboard_backend, "fake-clipboard")
        self.assertEqual(result.paste_backend, "fake-paste")

    def test_leaves_transcript_when_paste_fails(self):
        settings = Settings(clipboard_restore_delay_ms=0)
        clipboard = FakeClipboard("prior")
        result = paste_transcript("transcript", settings, clipboard, FakeInjector(False))

        self.assertFalse(result.pasted)
        self.assertFalse(result.injected)
        self.assertFalse(result.restored)
        self.assertEqual(clipboard.value, "transcript")

    def test_missing_system_clipboard_is_structured_failure(self):
        with patch("transclip.desktop.paste.SystemClipboard", side_effect=RuntimeError("missing clipboard")):
            result = paste_transcript("transcript", Settings())

        self.assertFalse(result.copied)
        self.assertFalse(result.pasted)
        self.assertFalse(result.injected)
        self.assertEqual(result.clipboard_backend, "unavailable")
        self.assertIn("missing clipboard", result.error_detail)

    def test_unreadable_prior_clipboard_still_copies_transcript(self):
        class UnreadablePriorClipboard(FakeClipboard):
            def __init__(self):
                super().__init__("prior")
                self.reads = 0

            def read(self):
                self.reads += 1
                if self.reads == 1:
                    raise UnicodeDecodeError("utf-8", b"\x89", 0, 1, "invalid")
                return self.value

        clipboard = UnreadablePriorClipboard()
        result = paste_transcript(
            "transcript",
            Settings(restore_clipboard_after_paste=True, clipboard_restore_delay_ms=0),
            clipboard,
            FakeInjector(),
        )

        self.assertTrue(result.copied)
        self.assertTrue(result.pasted)
        self.assertTrue(result.injected)
        self.assertFalse(result.restored)
        self.assertTrue(result.transcript_left_on_clipboard)
        self.assertEqual(clipboard.value, "transcript")

    def test_injector_error_detail_is_reported(self):
        class ErrorInjector:
            def paste(self):
                return False

            def error_detail(self):
                return "wtype paste command failed: virtual keyboard unsupported"

        settings = Settings(clipboard_restore_delay_ms=0)
        clipboard = FakeClipboard("prior")
        result = paste_transcript("transcript", settings, clipboard, ErrorInjector())

        self.assertFalse(result.pasted)
        self.assertFalse(result.injected)
        self.assertIn("virtual keyboard", result.error_detail)

    def test_injector_exception_is_structured_failure(self):
        class RaisingInjector:
            def paste(self):
                raise RuntimeError("paste exploded")

        settings = Settings(clipboard_restore_delay_ms=0)
        clipboard = FakeClipboard("prior")
        result = paste_transcript("transcript", settings, clipboard, RaisingInjector())

        self.assertTrue(result.copied)
        self.assertFalse(result.pasted)
        self.assertFalse(result.injected)
        self.assertIn("paste exploded", result.error_detail)

    def test_system_injector_reports_selected_xdotool_backend(self):
        def run(command, **_kwargs):
            if command[0] == "wtype":
                return type("Completed", (), {"returncode": 1, "stdout": "virtual keyboard unsupported"})()
            return type("Completed", (), {"returncode": 0, "stdout": ""})()

        runtime = FakeRuntime(
            env={"XDG_SESSION_TYPE": "x11"},
            available={"wtype": "/usr/bin/wtype", "xdotool": "/usr/bin/xdotool"},
            run_func=run,
        )
        injector = SystemPasteInjector(runtime, shortcut=TERMINAL_PASTE_SHORTCUT)
        self.assertTrue(injector.paste())

        self.assertEqual(injector.backend_name, "xdotool")

    def test_wayland_system_injector_reports_selected_ydotool_backend(self):
        def run(command, **_kwargs):
            if command[0] in {"wtype", "xdotool"}:
                return type("Completed", (), {"returncode": 1, "stdout": f"{command[0]} failed"})()
            return type("Completed", (), {"returncode": 0, "stdout": ""})()

        runtime = FakeRuntime(
            env={"XDG_SESSION_TYPE": "wayland"},
            available={
                "wtype": "/usr/bin/wtype",
                "xdotool": "/usr/bin/xdotool",
                "ydotool": "/usr/bin/ydotool",
            },
            run_func=run,
        )
        injector = SystemPasteInjector(runtime, shortcut=TERMINAL_PASTE_SHORTCUT)
        self.assertTrue(injector.paste())

        self.assertEqual(injector.backend_name, "ydotool")

    def test_linux_paste_commands_use_terminal_text_paste_shortcut(self):
        def which(name):
            return f"/usr/bin/{name}"

        wayland_info = session_info(environ={"XDG_SESSION_TYPE": "wayland"}, system="Linux")
        x11_info = session_info(environ={"XDG_SESSION_TYPE": "x11"}, system="Linux")

        wayland_commands = paste_commands(which=which, info=wayland_info)
        x11_commands = paste_commands(which=which, info=x11_info)
        wayland_argv = [command.command for command in wayland_commands]
        x11_argv = [command.command for command in x11_commands]

        self.assertIn(["ydotool", "key", "--delay", "50", "ctrl+shift+v"], wayland_argv)
        self.assertIn(list(YDOTOOL_TERMINAL_SEQUENTIAL_COMMAND), wayland_argv)
        self.assertIn(["xdotool", "key", "ctrl+shift+v"], x11_argv)

    def test_wayland_wtype_command_uses_key_event_for_terminal_text_paste(self):
        command = build_paste_commands_for_backend("wtype", TERMINAL_PASTE_SHORTCUT)[0]
        self.assertEqual(command, list(WTYPE_TERMINAL_PASTE_COMMAND))

    def test_build_paste_command_ydotool_uses_resolved_shortcut(self):
        terminal_commands = build_paste_commands_for_backend("ydotool", TERMINAL_PASTE_SHORTCUT)
        gui_commands = build_paste_commands_for_backend("ydotool", GUI_PASTE_SHORTCUT)
        self.assertEqual(terminal_commands[0], ["ydotool", "key", "--delay", "50", "ctrl+shift+v"])
        self.assertEqual(gui_commands, [["ydotool", "key", "--delay", "50", "ctrl+v"]])
        self.assertEqual(terminal_commands[1], list(YDOTOOL_TERMINAL_SEQUENTIAL_COMMAND))

    def test_resolve_paste_shortcut_uses_gui_for_linux_gui_focus(self):
        info = session_info(environ={"XDG_SESSION_TYPE": "wayland"}, system="Linux")
        self.assertEqual(resolve_paste_shortcut(info, "gui"), GUI_PASTE_SHORTCUT)
        self.assertEqual(resolve_paste_shortcut(info, "terminal"), TERMINAL_PASTE_SHORTCUT)
        self.assertEqual(resolve_paste_shortcut(info, "unknown"), TERMINAL_PASTE_SHORTCUT)

    def test_clipboard_only_skips_injector_but_copies_transcript(self):
        settings = Settings(text_delivery_mode="clipboard_only", clipboard_restore_delay_ms=0)
        clipboard = FakeClipboard("prior")
        injector = FakeInjector(False)
        with patch("transclip.desktop.paste.detect_focused_app") as detect_focus:
            result = paste_transcript("transcript", settings, clipboard, injector)

        detect_focus.assert_not_called()
        self.assertTrue(result.copied)
        self.assertTrue(result.pasted)
        self.assertEqual(result.delivery, "clipboard_only")
        self.assertFalse(result.injected)
        self.assertIsNone(result.paste_backend)
        self.assertFalse(injector.paste())

    def test_paste_result_includes_delivery_metadata(self):
        settings = Settings(clipboard_restore_delay_ms=0, focus_aware_paste=False)
        clipboard = FakeClipboard("prior")
        result = paste_transcript("transcript", settings, clipboard, FakeInjector())

        self.assertEqual(result.delivery, "inject")
        self.assertTrue(result.injected)
        self.assertEqual(result.paste_shortcut, "Ctrl+Shift+V")
        self.assertIsNone(result.focused_app_kind)

    def test_system_injector_reports_runtime_failure(self):
        def run(_command, **_kwargs):
            return type("Completed", (), {"returncode": 1, "stdout": "virtual keyboard unsupported"})()

        runtime = FakeRuntime(
            available={"wtype": "/usr/bin/wtype"},
            run_func=run,
        )
        injector = SystemPasteInjector(runtime, shortcut=TERMINAL_PASTE_SHORTCUT)
        self.assertFalse(injector.paste())
        self.assertIn("virtual keyboard unsupported", injector.error_detail())

    def test_wayland_clipboard_requires_wl_clipboard(self):
        runtime = FakeRuntime(env={"XDG_SESSION_TYPE": "wayland"})
        with self.assertRaisesRegex(RuntimeError, "Wayland clipboard requires wl-clipboard"):
            detect_clipboard_backend(runtime)

    def test_x11_clipboard_can_use_xclip(self):
        runtime = FakeRuntime(
            env={"XDG_SESSION_TYPE": "x11"},
            available={"xclip": "/usr/bin/xclip"},
        )
        backend = detect_clipboard_backend(runtime)

        self.assertEqual(backend.name, "xclip")

    def test_wayland_display_indicator_prefers_wl_clipboard_when_session_unknown(self):
        def which(name):
            return f"/usr/bin/{name}" if name in {"xclip", "wl-copy", "wl-paste"} else None

        info = session_info(
            environ={"WAYLAND_DISPLAY": "wayland-0"},
            system="Linux",
        )
        capability = clipboard_capability(which=which, info=info)

        self.assertTrue(capability.ok)
        self.assertEqual(capability.backend, "wl-clipboard")

    def test_windows_clipboard_and_paste_capabilities_are_available(self):
        runtime = FakeRuntime(system="Windows", home=Path("C:/Users/test"))

        clipboard = clipboard_capability(runtime=runtime)
        paste = paste_commands(runtime=runtime)

        self.assertTrue(clipboard.ok)
        self.assertEqual(clipboard.backend, "win32")
        self.assertEqual(paste[0].backend, "sendinput")

    def test_windows_paste_uses_sendinput_backend(self):
        runtime = FakeRuntime(system="Windows", home=Path("C:/Users/test"))
        with patch("transclip.desktop.paste.win32.send_ctrl_v_paste") as send_paste:
            injector = SystemPasteInjector(runtime, shortcut=GUI_PASTE_SHORTCUT)
            self.assertTrue(injector.paste())

        send_paste.assert_called_once()
        self.assertEqual(injector.backend_name, "sendinput")

    def test_x11_paste_registry_prefers_xdotool_before_ydotool(self):
        def which(name):
            return f"/usr/bin/{name}"

        info = session_info(environ={"XDG_SESSION_TYPE": "x11"}, system="Linux")
        backends = [spec.backend for spec in paste_specs(info)]
        self.assertEqual(backends, ["xdotool", "ydotool", "wtype"])

        commands = paste_commands(which=which, info=info)
        self.assertEqual(commands[0].backend, "xdotool")

    def test_x11_paste_capability_skips_wtype_when_xdotool_available(self):
        def which(name):
            return f"/usr/bin/{name}"

        def run(_command, **_kwargs):
            return type("Completed", (), {"returncode": 0, "stdout": ""})()

        info = session_info(environ={"XDG_SESSION_TYPE": "x11"}, system="Linux")
        capability = paste_capability(which=which, info=info, runner=run)

        self.assertTrue(capability.ok)
        self.assertEqual(capability.backend, "xdotool")

    def test_wayland_paste_registry_prefers_wtype_before_ydotool(self):
        def which(name):
            return f"/usr/bin/{name}"

        info = session_info(environ={"XDG_SESSION_TYPE": "wayland"}, system="Linux")
        backends = [spec.backend for spec in paste_specs(info)]
        self.assertEqual(backends, ["wtype", "ydotool"])

        commands = paste_commands(which=which, info=info)
        self.assertEqual(commands[0].backend, "wtype")



if __name__ == "__main__":
    unittest.main()
