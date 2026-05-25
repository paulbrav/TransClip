import unittest

from transclip.desktop.paste import plan_paste_delivery
from transclip.desktop.paste.focus import (
    FocusedApp,
    classify_wm_class,
    detect_focused_app,
    parse_terminal_wm_class_patterns,
)
from transclip.desktop.paste.platform import GUI_PASTE_SHORTCUT, TERMINAL_PASTE_SHORTCUT
from transclip.settings import Settings

from tests.service_helpers import FakeRuntime


class PasteFocusTests(unittest.TestCase):
    def test_classify_wm_class_matches_terminal_patterns(self):
        patterns = parse_terminal_wm_class_patterns("")
        self.assertEqual(classify_wm_class("Cursor", patterns), "terminal")
        self.assertEqual(classify_wm_class("firefox", patterns), "gui")

    def test_detect_focused_app_falls_back_to_unknown_when_gnome_query_fails(self):
        runtime = FakeRuntime(
            env={
                "XDG_SESSION_TYPE": "wayland",
                "XDG_CURRENT_DESKTOP": "ubuntu:GNOME",
            },
            run_func=lambda _command, **_kwargs: type("Completed", (), {"returncode": 1, "stdout": ""})(),
        )
        focused = detect_focused_app(runtime)
        self.assertEqual(focused, FocusedApp(kind="unknown"))

    def test_plan_paste_delivery_uses_terminal_shortcut_for_linux_terminal_focus(self):
        runtime = FakeRuntime(
            env={
                "XDG_SESSION_TYPE": "wayland",
                "XDG_CURRENT_DESKTOP": "ubuntu:GNOME",
            },
            run_func=lambda _command, **_kwargs: type(
                "Completed",
                (),
                {"returncode": 0, "stdout": "(true, 'Cursor')"},
            )(),
        )
        settings = Settings(focus_aware_paste=True)
        plan = plan_paste_delivery(settings, runtime=runtime)

        self.assertEqual(plan.focused_app_kind, "terminal")
        self.assertEqual(plan.shortcut, TERMINAL_PASTE_SHORTCUT)

    def test_plan_paste_delivery_uses_gui_shortcut_for_linux_gui_focus(self):
        runtime = FakeRuntime(
            env={
                "XDG_SESSION_TYPE": "wayland",
                "XDG_CURRENT_DESKTOP": "ubuntu:GNOME",
            },
            run_func=lambda _command, **_kwargs: type(
                "Completed",
                (),
                {"returncode": 0, "stdout": "(true, 'firefox')"},
            )(),
        )
        settings = Settings(focus_aware_paste=True)
        plan = plan_paste_delivery(settings, runtime=runtime)

        self.assertEqual(plan.focused_app_kind, "gui")
        self.assertEqual(plan.shortcut, GUI_PASTE_SHORTCUT)

    def test_plan_paste_delivery_leaves_windows_unchanged(self):
        runtime = FakeRuntime(system="Windows", home="C:/Users/test")
        settings = Settings(focus_aware_paste=True)
        plan = plan_paste_delivery(settings, runtime=runtime)

        self.assertIsNone(plan.focused_app_kind)
        self.assertEqual(plan.shortcut, GUI_PASTE_SHORTCUT)


if __name__ == "__main__":
    unittest.main()
