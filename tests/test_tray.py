import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from transclip.settings import DEFAULT_HOTKEY_LINUX, Settings, load_settings, write_settings
from transclip.tray import _history_file_signature, run_python_tray


class FakeLabel:
    def __init__(self, text: str):
        self.text = text

    def set_text(self, text: str) -> None:
        self.text = text


class FakeMenuItem:
    def __init__(self, label: str = ""):
        self.label = label
        self.child = FakeLabel(label)
        self.sensitive = True
        self.submenu = None
        self.handlers = {}

    def connect(self, signal: str, callback) -> None:
        self.handlers[signal] = callback

    def set_sensitive(self, sensitive: bool) -> None:
        self.sensitive = sensitive

    def set_submenu(self, menu) -> None:
        self.submenu = menu

    def get_child(self):
        return self.child

    def set_label(self, label: str) -> None:
        self.label = label

    def show_all(self) -> None:
        return None


class FakeMenu:
    def __init__(self):
        self.children = []
        self.show_count = 0
        self.handlers = {}

    def append(self, item) -> None:
        self.children.append(item)

    def remove(self, item) -> None:
        self.children.remove(item)

    def get_children(self) -> list:
        return list(self.children)

    def show_all(self) -> None:
        self.show_count += 1

    def connect(self, signal: str, callback) -> None:
        self.handlers[signal] = callback


class FakeSeparatorMenuItem:
    pass


class FakeBox:
    def __init__(self):
        self.children = []

    def add(self, child) -> None:
        self.children.append(child)


class FakeDialog:
    next_response = None
    next_text = ""

    def __init__(self, title: str = ""):
        self.title = title
        self.box = FakeBox()

    def add_button(self, *_args) -> None:
        return None

    def get_content_area(self):
        return self.box

    def set_default_response(self, _response) -> None:
        return None

    def show_all(self) -> None:
        return None

    def run(self):
        for child in self.box.children:
            if hasattr(child, "set_text") and hasattr(child, "get_text"):
                child.set_text(type(self).next_text)
        return type(self).next_response

    def destroy(self) -> None:
        return None


class FakeEntry:
    def __init__(self):
        self.text = ""

    def set_text(self, text: str) -> None:
        self.text = text

    def get_text(self) -> str:
        return self.text

    def set_activates_default(self, _value: bool) -> None:
        return None


class FakeIndicatorInstance:
    def __init__(self):
        self.menus = []
        self.icons = []

    def set_title(self, title: str) -> None:
        self.title = title

    def set_status(self, status) -> None:
        self.status = status

    def set_icon_full(self, icon: str, description: str) -> None:
        self.icons.append((icon, description))

    def set_menu(self, menu) -> None:
        self.menus.append(menu)


class FakeIndicatorFactory:
    current = None

    @classmethod
    def new(cls, *_args):
        cls.current = FakeIndicatorInstance()
        return cls.current


class FakeClient:
    def __init__(self, settings):
        self.settings = settings

    def health(self):
        return {"status": "ready"}


class TrayTests(unittest.TestCase):
    def setUp(self):
        gi = types.ModuleType("gi")
        gi.require_version = lambda *_args: None
        repository = types.ModuleType("gi.repository")
        gtk = types.SimpleNamespace(
            accelerator_parse=lambda value: (1, 0) if value.startswith("<") else (0, 0),
            Dialog=FakeDialog,
            Entry=FakeEntry,
            Label=FakeMenuItem,
            Menu=FakeMenu,
            MenuItem=FakeMenuItem,
            ResponseType=types.SimpleNamespace(CANCEL=0, OK=1),
            SeparatorMenuItem=FakeSeparatorMenuItem,
            main=lambda: None,
            main_quit=lambda: None,
        )
        app_indicator = types.SimpleNamespace(
            Indicator=FakeIndicatorFactory,
            IndicatorCategory=types.SimpleNamespace(APPLICATION_STATUS="application"),
            IndicatorStatus=types.SimpleNamespace(ACTIVE="active"),
        )
        glib = types.SimpleNamespace(timeout_add_seconds=lambda *_args: 1)
        repository.Gtk = gtk
        repository.AyatanaAppIndicator3 = app_indicator
        repository.GLib = glib
        self.modules = {"gi": gi, "gi.repository": repository}

    def test_health_refresh_updates_existing_menu_without_replacing_it(self):
        with (
            patch.dict(sys.modules, self.modules),
            patch("transclip.tray.InferenceClient", FakeClient),
            patch("transclip.tray.read_history", return_value=[]),
        ):
            code = run_python_tray(Settings())

        indicator = FakeIndicatorFactory.current
        self.assertEqual(code, 0)
        self.assertEqual(len(indicator.menus), 1)
        menu = indicator.menus[0]
        self.assertEqual(menu.children[0].child.text, "Service: ready")
        self.assertEqual(menu.children[2].child.text, "Record")
        self.assertFalse(menu.children[3].sensitive)

    def test_recent_transcripts_refresh_on_submenu_map_and_skip_unchanged(self):
        history_events = [
            {"text": "First transcript"},
            {"text": "Second transcript"},
        ]

        with (
            patch.dict(sys.modules, self.modules),
            patch("transclip.tray.InferenceClient", FakeClient),
            patch("transclip.tray.read_history", return_value=history_events) as read_history,
            patch("transclip.tray._history_file_signature", side_effect=[123, 123, 456]),
        ):
            code = run_python_tray(Settings())
            indicator = FakeIndicatorFactory.current
            history_item = menu_item_by_label(indicator, "Recent transcripts")
            history_menu = history_item.submenu
            self.assertEqual(code, 0)
            self.assertIn("map", history_menu.handlers)
            baseline_reads = read_history.call_count

            history_menu.handlers["map"](history_menu)
            self.assertEqual(read_history.call_count, baseline_reads)
            self.assertEqual(len(history_menu.children), 2)

            history_menu.handlers["map"](history_menu)
            self.assertEqual(read_history.call_count, baseline_reads + 1)
            self.assertEqual(len(history_menu.children), 2)

    def test_history_file_signature_uses_mtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.jsonl"
            self.assertIsNone(_history_file_signature(path))
            path.write_text("{}\n", encoding="utf-8")
            first = _history_file_signature(path)
            second = _history_file_signature(path)
            self.assertIsNotNone(first)
            self.assertEqual(first, second)

    def test_set_hotkey_saves_settings_and_installs_shortcut(self):
        FakeDialog.next_response = 1
        FakeDialog.next_text = "<Control><Alt>space"
        with tempfile.TemporaryDirectory() as tmp:
            settings_file = Path(tmp) / "settings.toml"
            settings = Settings()
            write_settings(settings, settings_file)
            with (
                patch.dict(sys.modules, self.modules),
                patch("transclip.tray.InferenceClient", FakeClient),
                patch("transclip.tray.read_history", return_value=[]),
                patch("transclip.tray.install_shortcut") as install_shortcut,
            ):
                code = run_python_tray(settings, explicit_settings_path=settings_file)
                indicator = FakeIndicatorFactory.current
                set_hotkey_item = menu_item_by_label(indicator, "Set hotkey...")
                set_hotkey_item.handlers["activate"](set_hotkey_item)

                self.assertEqual(code, 0)
                self.assertEqual(settings.hotkey_linux, "<Control><Alt>space")
                self.assertEqual(load_settings(settings_file).hotkey_linux, "<Control><Alt>space")
                install_shortcut.assert_called_once()
                self.assertEqual(install_shortcut.call_args.kwargs["binding"], "<Control><Alt>space")
                self.assertIn("Hotkey set", indicator.menus[0].children[0].child.text)

    def test_asr_model_menu_saves_settings_and_restarts_service(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings_file = Path(tmp) / "settings.toml"
            settings = Settings()
            write_settings(settings, settings_file)
            with (
                patch.dict(sys.modules, self.modules),
                patch("transclip.tray.InferenceClient", FakeClient),
                patch("transclip.tray.read_history", return_value=[]),
                patch("transclip.tray.service_action") as service_action,
            ):
                service_action.return_value = types.SimpleNamespace(detail="restarted")
                code = run_python_tray(settings, explicit_settings_path=settings_file)
                indicator = FakeIndicatorFactory.current
                model_item = menu_item_by_label(indicator, "ASR model")
                regular_item = submenu_item_by_label(model_item, "Keyword-biased ASR - Granite 4.1")
                regular_item.handlers["activate"](regular_item)

                self.assertEqual(code, 0)
                self.assertEqual(settings.asr_backend, "granite")
                self.assertEqual(settings.asr_model, "ibm-granite/granite-speech-4.1-2b")
                saved = load_settings(settings_file)
                self.assertEqual(saved.asr_backend, "granite")
                self.assertEqual(saved.asr_model, "ibm-granite/granite-speech-4.1-2b")
                service_action.assert_called_once_with("restart")
                self.assertIn("ASR model set", indicator.menus[0].children[0].child.text)
                self.assertEqual(regular_item.child.text, "✓ Keyword-biased ASR - Granite 4.1")

    def test_model_cleanup_toggle_label_persists_and_restarts_service(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings_file = Path(tmp) / "settings.toml"
            settings = Settings(voice_model_cleanup_always_on=False)
            write_settings(settings, settings_file)
            with (
                patch.dict(sys.modules, self.modules),
                patch("transclip.tray.InferenceClient", FakeClient),
                patch("transclip.tray.read_history", return_value=[]),
                patch("transclip.tray.service_action") as service_action,
            ):
                service_action.return_value = types.SimpleNamespace(detail="restarted")
                code = run_python_tray(settings, explicit_settings_path=settings_file)
                indicator = FakeIndicatorFactory.current
                cleanup_item = menu_item_by_label(indicator, "Model cleanup always on")
                cleanup_item.handlers["activate"](cleanup_item)

                self.assertEqual(code, 0)
                self.assertTrue(settings.voice_model_cleanup_always_on)
                self.assertTrue(load_settings(settings_file).voice_model_cleanup_always_on)
                service_action.assert_called_once_with("restart")
                self.assertEqual(cleanup_item.child.text, "✓ Model cleanup always on")
                self.assertIn("Model cleanup always on", indicator.menus[0].children[0].child.text)

    def test_set_hotkey_preserves_other_current_settings(self):
        FakeDialog.next_response = 1
        FakeDialog.next_text = "<Control><Alt>space"
        with tempfile.TemporaryDirectory() as tmp:
            settings_file = Path(tmp) / "settings.toml"
            settings = Settings(toggle_cooldown_ms=500)
            write_settings(Settings(toggle_cooldown_ms=900), settings_file)
            with (
                patch.dict(sys.modules, self.modules),
                patch("transclip.tray.InferenceClient", FakeClient),
                patch("transclip.tray.read_history", return_value=[]),
                patch("transclip.tray.install_shortcut"),
            ):
                run_python_tray(settings, explicit_settings_path=settings_file)
                set_hotkey_item = menu_item_by_label(FakeIndicatorFactory.current, "Set hotkey...")
                set_hotkey_item.handlers["activate"](set_hotkey_item)

            saved = load_settings(settings_file)

        self.assertEqual(saved.hotkey_linux, "<Control><Alt>space")
        self.assertEqual(saved.toggle_cooldown_ms, 900)

    def test_set_hotkey_rolls_back_settings_when_install_fails(self):
        FakeDialog.next_response = 1
        FakeDialog.next_text = "<Control><Alt>space"
        with tempfile.TemporaryDirectory() as tmp:
            settings_file = Path(tmp) / "settings.toml"
            settings = Settings()
            write_settings(settings, settings_file)
            with (
                patch.dict(sys.modules, self.modules),
                patch("transclip.tray.InferenceClient", FakeClient),
                patch("transclip.tray.read_history", return_value=[]),
                patch("transclip.tray.install_shortcut", side_effect=RuntimeError("gsettings failed")),
            ):
                run_python_tray(settings, explicit_settings_path=settings_file)
                indicator = FakeIndicatorFactory.current
                set_hotkey_item = menu_item_by_label(indicator, "Set hotkey...")
                set_hotkey_item.handlers["activate"](set_hotkey_item)
                detail = indicator.menus[0].children[0].child.text

            saved = load_settings(settings_file)

        self.assertEqual(settings.hotkey_linux, DEFAULT_HOTKEY_LINUX)
        self.assertEqual(saved.hotkey_linux, DEFAULT_HOTKEY_LINUX)
        self.assertIn("Hotkey update failed", detail)

    def test_set_hotkey_rejects_invalid_accelerator(self):
        FakeDialog.next_response = 1
        FakeDialog.next_text = "not a hotkey"
        with tempfile.TemporaryDirectory() as tmp:
            settings_file = Path(tmp) / "settings.toml"
            settings = Settings()
            write_settings(settings, settings_file)
            with (
                patch.dict(sys.modules, self.modules),
                patch("transclip.tray.InferenceClient", FakeClient),
                patch("transclip.tray.read_history", return_value=[]),
                patch("transclip.tray.install_shortcut") as install_shortcut,
            ):
                run_python_tray(settings, explicit_settings_path=settings_file)
                indicator = FakeIndicatorFactory.current
                set_hotkey_item = menu_item_by_label(indicator, "Set hotkey...")
                set_hotkey_item.handlers["activate"](set_hotkey_item)

        install_shortcut.assert_not_called()
        self.assertIn("not a valid", indicator.menus[0].children[0].child.text)


def menu_item_by_label(indicator, label: str):
    for item in indicator.menus[0].children:
        if getattr(item, "label", "") == label:
            return item
    raise AssertionError(f"missing menu item: {label}")


def submenu_item_by_label(item, label: str):
    for child in item.submenu.children:
        if getattr(child, "label", "") == label or getattr(child.child, "text", "") == label:
            return child
    raise AssertionError(f"missing submenu item: {label}")


if __name__ == "__main__":
    unittest.main()
