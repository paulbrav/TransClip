import unittest
from unittest.mock import MagicMock

from transclip.desktop.tray.controller import TrayController, build_tray_action_callbacks
from transclip.desktop.tray.menu_update import HistoryMenuState, after_tray_action
from transclip.desktop.tray.sinks.gtk import GtkMenuSink
from transclip.desktop.tray.sinks.macos import MacOSMenuSink
from transclip.desktop.tray.sinks.win32 import PystrayMenuSink
from transclip.recording_ops import ToggleOutcome
from transclip.settings import Settings

from tests.service_helpers import FakeRuntime, patch_linux_gpu_runtime


class RecordingView:
    def __init__(self) -> None:
        self.history: list[tuple[str, str]] = []

    def set_label(self, ref: str, text: str) -> None:
        del ref, text

    def set_enabled(self, ref: str, enabled: bool) -> None:
        del ref, enabled

    def set_model_labels(self, rows) -> None:
        del rows

    def rebuild_history(self, entries) -> None:
        self.history = list(entries)

    def set_health_icon(self, icon: str) -> None:
        del icon


class TrayControllerTests(unittest.TestCase):
    def test_copy_history_text_delegates_to_session(self):
        with patch_linux_gpu_runtime():
            from transclip.desktop.tray.session import TraySession

            session = TraySession(Settings(), runtime=FakeRuntime(system="Linux", home="/home/test"))
        view = RecordingView()
        controller = TrayController(session, view, {}, history_state=HistoryMenuState(signature=object()))
        session.copy_text = MagicMock(return_value="copied")  # type: ignore[method-assign]

        controller.copy_history_text("hello")

        session.copy_text.assert_called_once_with("hello")

    def test_refresh_health_calls_session_refresh_and_health_icon(self):
        with patch_linux_gpu_runtime():
            from transclip.desktop.tray.session import TraySession

            session = TraySession(Settings(), runtime=FakeRuntime(system="Linux", home="/home/test"))
        view = RecordingView()
        icon_updates: list[str] = []
        session.refresh_health = MagicMock()  # type: ignore[method-assign]
        controller = TrayController(
            session,
            view,
            {},
            history_state=HistoryMenuState(signature=object()),
            on_health_icon=lambda: icon_updates.append(session.health.icon),
        )

        controller.refresh_health()

        session.refresh_health.assert_called_once()
        self.assertEqual(icon_updates, [session.health.icon])

    def test_update_menu_enables_copy_latest_from_session_latest(self):
        with patch_linux_gpu_runtime():
            from transclip.desktop.tray.session import TraySession

            session = TraySession(Settings(), runtime=FakeRuntime(system="Linux", home="/home/test"))
        view = RecordingView()
        enabled: list[bool] = []
        view.set_enabled = lambda ref, value: enabled.append(value) if ref == "latest_item" else None  # type: ignore[method-assign]
        session.latest = "hello"
        controller = TrayController(
            session,
            view,
            {"latest_item": object()},
            history_state=HistoryMenuState(signature=object()),
        )

        controller.update_menu()

        self.assertEqual(enabled, [True])


class PystrayMenuSinkTests(unittest.TestCase):
    def test_history_submenu_uses_on_copy_history_callback(self):
        menu_refs: dict = {}
        copied: list[str] = []

        def menu_item(label, callback, enabled=True):
            return (label, callback, enabled)

        pystray = MagicMock()
        pystray.MenuItem = menu_item
        pystray.Menu = lambda *items: list(items)

        sink = PystrayMenuSink(
            [],
            menu_refs,
            pystray=pystray,
            after_action=lambda action: action(),
            set_model=lambda *_args: None,
            on_copy_history=copied.append,
        )
        menu_refs["_history_entries"] = [("preview", "full text")]

        submenu = sink._build_history_menu()
        _, copy_handler, _ = submenu[0]
        copy_handler(None, None)

        self.assertEqual(copied, ["full text"])


class BuildTrayActionCallbacksTests(unittest.TestCase):
    def test_builds_shared_service_actions(self):
        with patch_linux_gpu_runtime():
            from transclip.desktop.tray.session import TraySession

            session = TraySession(Settings(), runtime=FakeRuntime(system="Linux", home="/home/test"))
        controller = TrayController(session, RecordingView(), {}, history_state=HistoryMenuState(signature=object()))
        calls: list[str] = []

        callbacks = build_tray_action_callbacks(
            controller,
            session,
            set_hotkey=lambda: calls.append("hotkey"),
            quit=lambda: calls.append("quit"),
        )

        self.assertIn("toggle", callbacks)
        self.assertIn("start_service", callbacks)
        callbacks["set_hotkey"]()
        callbacks["quit"]()
        self.assertEqual(calls, ["hotkey", "quit"])


class GtkMenuSinkTests(unittest.TestCase):
    def test_status_label_disables_item(self):
        menu_refs: dict = {}
        labels: list[tuple[str, bool]] = []

        class FakeItem:
            def __init__(self):
                self.sensitive = True

            def set_sensitive(self, value):
                labels.append(("Service: ready", value))

        def append_label(_menu, text):
            return FakeItem()

        sink = GtkMenuSink(
            object(),
            menu_refs,
            append_separator=lambda _menu: None,
            append_label=append_label,
            append_item=lambda *_args: None,
            after_action=lambda action: action(),
            set_model=lambda *_args: None,
        )

        sink.status_label("status_item", "Service: ready")

        self.assertEqual(labels, [("Service: ready", False)])


class MacOSMenuSinkTests(unittest.TestCase):
    def test_action_uses_action_name_when_ref_empty(self):
        controller = type(
            "Controller",
            (),
            {
                "menu_refs": {},
                "action_callbacks": {},
                "appendItem_action_toMenu_": lambda self, label, action, menu: type(
                    "Item",
                    (),
                    {"setRepresentedObject_": lambda *_args: None, "setEnabled_": lambda *_args: None},
                )(),
                "appendLabel_toMenu_": lambda *_args: None,
            },
        )()
        sink = MacOSMenuSink(controller, object())

        sink.action("", "Start service", "start_service", callback=lambda: None)

        self.assertIn("start_service", controller.action_callbacks)


class AfterTrayActionTests(unittest.TestCase):
    def test_after_tray_action_refreshes_history_when_latest_transcript_set(self):
        history_state = HistoryMenuState(signature=object())
        refreshed: list[bool] = []

        after_tray_action(
            lambda: ToggleOutcome(
                ok=True,
                payload={"action": "stopped", "text": "hello"},
                service_url="http://127.0.0.1:8765",
            ),
            history_state=history_state,
            refresh_history=lambda: refreshed.append(True),
            update_menu=lambda: None,
        )

        self.assertEqual(refreshed, [True])
        self.assertNotEqual(history_state.signature, object())


if __name__ == "__main__":
    unittest.main()
