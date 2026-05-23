import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

from tests.service_helpers import FakeRuntime
from transclip.recording_ops import ToggleOutcome
from transclip.settings import DEFAULT_HOTKEY_LINUX, Settings, load_settings, write_settings
from transclip.tray import run_macos_tray, run_python_tray, run_tray


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


class FakeMenu:
    def __init__(self):
        self.children = []
        self.show_count = 0

    def append(self, item) -> None:
        self.children.append(item)

    def remove(self, item) -> None:
        self.children.remove(item)

    def get_children(self) -> list:
        return list(self.children)

    def show_all(self) -> None:
        self.show_count += 1


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


class FakeNSObject:
    @classmethod
    def alloc(cls):
        return cls.__new__(cls)

    def init(self):
        return self


class FakeNSMenuItem:
    def __init__(self):
        self.title = ""
        self.action = None
        self.target = None
        self.enabled = True
        self.submenu = None
        self.represented = None

    @classmethod
    def alloc(cls):
        return cls()

    @classmethod
    def separatorItem(cls):
        item = cls()
        item.title = "---"
        return item

    def initWithTitle_action_keyEquivalent_(self, title, action, _key):
        self.title = title
        self.action = action
        return self

    def setTarget_(self, target) -> None:
        self.target = target

    def setTitle_(self, title) -> None:
        self.title = title

    def setEnabled_(self, enabled) -> None:
        self.enabled = enabled

    def setSubmenu_(self, menu) -> None:
        self.submenu = menu

    def setRepresentedObject_(self, value) -> None:
        self.represented = value

    def representedObject(self):
        return self.represented

    def activate(self) -> None:
        method_name = self.action.replace(":", "_")
        getattr(self.target, method_name)(self)


class FakeNSMenu:
    def __init__(self):
        self.items = []
        self.delegate = None

    @classmethod
    def alloc(cls):
        return cls()

    def init(self):
        return self

    def setDelegate_(self, delegate) -> None:
        self.delegate = delegate

    def addItem_(self, item) -> None:
        self.items.append(item)

    def itemArray(self):
        return list(self.items)

    def removeItem_(self, item) -> None:
        self.items.remove(item)


class FakeStatusButton:
    def __init__(self):
        self.title = ""
        self.tooltip = ""

    def setTitle_(self, title) -> None:
        self.title = title

    def setToolTip_(self, tooltip) -> None:
        self.tooltip = tooltip


class FakeStatusItem:
    current = None

    def __init__(self):
        self.button_instance = FakeStatusButton()
        self.menu = None
        type(self).current = self

    def button(self):
        return self.button_instance

    def setMenu_(self, menu) -> None:
        self.menu = menu


class FakeStatusBar:
    @classmethod
    def systemStatusBar(cls):
        return cls()

    def statusItemWithLength_(self, _length):
        return FakeStatusItem()


class FakeNSApplication:
    current = None

    def __init__(self):
        self.delegate = None
        self.ran = False
        type(self).current = self

    @classmethod
    def sharedApplication(cls):
        return cls.current or cls()

    def setActivationPolicy_(self, policy) -> None:
        self.policy = policy

    def setDelegate_(self, delegate) -> None:
        self.delegate = delegate

    def run(self) -> None:
        self.ran = True

    def terminate_(self, _sender) -> None:
        self.terminated = True


class FakeNSApp:
    terminated = False

    @classmethod
    def terminate_(cls, _sender) -> None:
        cls.terminated = True


class FakeNSTimer:
    scheduled: ClassVar[list[tuple]] = []

    @classmethod
    def scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(cls, *args):
        cls.scheduled.append(args)
        return object()


def fake_macos_modules():
    appkit = types.ModuleType("AppKit")
    appkit.NSApp = FakeNSApp
    appkit.NSApplication = FakeNSApplication
    appkit.NSApplicationActivationPolicyAccessory = "accessory"
    appkit.NSMenu = FakeNSMenu
    appkit.NSMenuItem = FakeNSMenuItem
    appkit.NSStatusBar = FakeStatusBar
    appkit.NSVariableStatusItemLength = -1
    foundation = types.ModuleType("Foundation")
    foundation.NSObject = FakeNSObject
    foundation.NSTimer = FakeNSTimer
    return {"AppKit": appkit, "Foundation": foundation}


class TrayTests(unittest.TestCase):
    def setUp(self):
        FakeStatusItem.current = None
        FakeNSApplication.current = None
        FakeNSTimer.scheduled = []
        FakeNSApp.terminated = False
        self.runtime_patch = patch("transclip.runtime_profile.get_runtime", return_value=FakeRuntime(system="Linux"))
        self.runtime_patch.start()
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

    def tearDown(self):
        self.runtime_patch.stop()

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

    def test_darwin_tray_runs_native_menubar(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = FakeRuntime(system="Darwin", home=Path(tmp))
            with (
                patch.dict(sys.modules, fake_macos_modules()),
                patch("transclip.tray.get_runtime", return_value=runtime),
                patch("transclip.tray.InferenceClient", FakeClient),
                patch("transclip.tray.read_history", return_value=[]),
                patch("transclip.runtime_profile.get_runtime", side_effect=lambda value=None: value or runtime),
                patch("transclip.runtime_profile.machine_architecture", return_value="arm64"),
            ):
                code = run_tray(Settings())

        self.assertEqual(code, 0)
        self.assertTrue(FakeNSApplication.current.ran)
        menu = FakeStatusItem.current.menu
        self.assertEqual(status_title(menu), "Service: ready")
        self.assertEqual(menu_item_by_title(FakeStatusItem.current, "Record").title, "Record")
        self.assertFalse(menu_item_by_title(FakeStatusItem.current, "Copy latest transcript").enabled)

    def test_macos_tray_reports_missing_pyobjc(self):
        stderr = io.StringIO()
        with (
            patch.dict(sys.modules, {"AppKit": None, "Foundation": None}),
            patch("sys.stderr", stderr),
        ):
            code = run_macos_tray(Settings(), runtime=FakeRuntime(system="Darwin"))

        self.assertEqual(code, 1)
        self.assertIn("PyObjC", stderr.getvalue())
        self.assertIn("macos-ui", stderr.getvalue())

    def test_macos_record_click_updates_menu_to_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = FakeRuntime(system="Darwin", home=Path(tmp))
            outcome = ToggleOutcome(True, {"action": "started"}, "http://127.0.0.1:8765")
            with (
                patch.dict(sys.modules, fake_macos_modules()),
                patch("transclip.tray.InferenceClient", FakeClient),
                patch("transclip.tray.read_history", return_value=[]),
                patch("transclip.tray.toggle_recording", return_value=outcome),
                patch("transclip.runtime_profile.get_runtime", side_effect=lambda value=None: value or runtime),
                patch("transclip.runtime_profile.machine_architecture", return_value="arm64"),
            ):
                run_macos_tray(Settings(), runtime=runtime)
                toggle_item = menu_item_by_title(FakeStatusItem.current, "Record")
                toggle_item.activate()

        self.assertEqual(toggle_item.title, "Stop + paste")
        self.assertEqual(FakeStatusItem.current.button().title, "●")
        self.assertEqual(status_title(FakeStatusItem.current.menu), "Service: recording")

    def test_macos_record_stop_surfaces_paste_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = FakeRuntime(system="Darwin", home=Path(tmp))
            outcome = ToggleOutcome(
                True,
                {"action": "stopped", "text": "hello world"},
                "http://127.0.0.1:8765",
                paste_failed_message="Paste failed. The transcript is still on the clipboard. denied",
            )
            with (
                patch.dict(sys.modules, fake_macos_modules()),
                patch("transclip.tray.InferenceClient", FakeClient),
                patch("transclip.tray.read_history", return_value=[]),
                patch("transclip.tray.toggle_recording", return_value=outcome) as toggle,
                patch("transclip.runtime_profile.get_runtime", side_effect=lambda value=None: value or runtime),
                patch("transclip.runtime_profile.machine_architecture", return_value="arm64"),
            ):
                run_macos_tray(Settings(), runtime=runtime)
                toggle_item = menu_item_by_title(FakeStatusItem.current, "Record")
                toggle_item.activate()

        toggle.assert_called_once()
        self.assertTrue(toggle.call_args.kwargs["paste"])
        self.assertEqual(toggle_item.target.state["latest"], "hello world")
        self.assertIn("Paste failed", status_title(FakeStatusItem.current.menu))
        self.assertIn("denied", status_title(FakeStatusItem.current.menu))

    def test_macos_tray_action_selectors_are_bound_to_controller_methods(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = FakeRuntime(system="Darwin", home=Path(tmp))
            with (
                patch.dict(sys.modules, fake_macos_modules()),
                patch("transclip.tray.InferenceClient", FakeClient),
                patch("transclip.tray.read_history", return_value=[]),
                patch("transclip.runtime_profile.get_runtime", side_effect=lambda value=None: value or runtime),
                patch("transclip.runtime_profile.machine_architecture", return_value="arm64"),
            ):
                code = run_macos_tray(Settings(), runtime=runtime)

        self.assertEqual(code, 0)
        for item in actionable_menu_items(FakeStatusItem.current.menu):
            method_name = item.action.replace(":", "_")
            self.assertTrue(hasattr(item.target, method_name), f"missing selector method for {item.action}")
        timer = FakeNSTimer.scheduled[-1]
        self.assertTrue(hasattr(timer[1], timer[2].replace(":", "_")))

    def test_macos_asr_model_menu_saves_settings_and_restarts_service(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = FakeRuntime(system="Darwin", home=Path(tmp))
            settings_file = Path(tmp) / "settings.toml"
            settings = Settings()
            write_settings(settings, settings_file)
            with (
                patch.dict(sys.modules, fake_macos_modules()),
                patch("transclip.tray.InferenceClient", FakeClient),
                patch("transclip.tray.read_history", return_value=[]),
                patch("transclip.runtime_profile.get_runtime", side_effect=lambda value=None: value or runtime),
                patch("transclip.runtime_profile.machine_architecture", return_value="arm64"),
                patch("transclip.tray.service_action") as service_action,
            ):
                service_action.return_value = types.SimpleNamespace(detail="restarted")
                code = run_macos_tray(settings, explicit_settings_path=settings_file, runtime=runtime)
                model_item = menu_item_by_title(FakeStatusItem.current, "ASR model")
                mlx_item = submenu_item_by_title(model_item, "mlx-community/whisper-large-v3-turbo-asr-fp16")
                mlx_item.activate()

                self.assertEqual(code, 0)
                self.assertEqual(settings.asr_backend, "mlx_audio_whisper")
                self.assertEqual(settings.asr_model, "mlx-community/whisper-large-v3-turbo-asr-fp16")
                saved = load_settings(settings_file)
                self.assertEqual(saved.asr_backend, "mlx_audio_whisper")
                self.assertEqual(saved.asr_model, "mlx-community/whisper-large-v3-turbo-asr-fp16")
                service_action.assert_called_once_with("restart")
                self.assertIn("ASR model set", status_title(FakeStatusItem.current.menu))

    def test_macos_quit_terminates_current_application(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime = FakeRuntime(system="Darwin", home=Path(tmp))
            with (
                patch.dict(sys.modules, fake_macos_modules()),
                patch("transclip.tray.InferenceClient", FakeClient),
                patch("transclip.tray.read_history", return_value=[]),
                patch("transclip.runtime_profile.get_runtime", side_effect=lambda value=None: value or runtime),
                patch("transclip.runtime_profile.machine_architecture", return_value="arm64"),
            ):
                run_macos_tray(Settings(), runtime=runtime)
                quit_item = menu_item_by_title(FakeStatusItem.current, "Quit tray")
                quit_item.activate()

        self.assertTrue(FakeNSApplication.current.terminated)

    def test_macos_copy_latest_uses_history_when_no_recording_in_memory(self):
        copied = []

        class Clipboard:
            def write(self, text):
                copied.append(text)

        with tempfile.TemporaryDirectory() as tmp:
            runtime = FakeRuntime(system="Darwin", home=Path(tmp))
            with (
                patch.dict(sys.modules, fake_macos_modules()),
                patch("transclip.tray.InferenceClient", FakeClient),
                patch("transclip.tray.read_history", return_value=[{"text": "hello history"}]),
                patch("transclip.tray.SystemClipboard", return_value=Clipboard()),
                patch("transclip.runtime_profile.get_runtime", side_effect=lambda value=None: value or runtime),
                patch("transclip.runtime_profile.machine_architecture", return_value="arm64"),
            ):
                run_macos_tray(Settings(), runtime=runtime)
                menu_item_by_title(FakeStatusItem.current, "Copy latest transcript").activate()

        self.assertEqual(copied, ["hello history"])
        self.assertEqual(status_title(FakeStatusItem.current.menu), "Copied latest transcript")

    def test_linux_tray_filters_macos_only_models(self):
        with (
            patch.dict(sys.modules, self.modules),
            patch("transclip.tray.InferenceClient", FakeClient),
            patch("transclip.tray.read_history", return_value=[]),
        ):
            run_python_tray(Settings())
            model_item = menu_item_by_label(FakeIndicatorFactory.current, "ASR model")

        labels = [getattr(child.child, "text", "") for child in model_item.submenu.children]
        self.assertIn("✓ Fast local ASR - Granite 4.1 NAR", labels)
        self.assertNotIn("mlx-community/whisper-large-v3-turbo-asr-fp16", labels)


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


def status_title(menu):
    return menu.items[0].title


def actionable_menu_items(menu):
    items = []
    for item in menu.items:
        if getattr(item, "action", None):
            items.append(item)
        if getattr(item, "submenu", None):
            items.extend(actionable_menu_items(item.submenu))
    return items


def menu_item_by_title(status_item, title: str):
    for item in status_item.menu.items:
        if getattr(item, "title", "") == title:
            return item
    raise AssertionError(f"missing menu item: {title}")


def submenu_item_by_title(item, title: str):
    for child in item.submenu.items:
        child_title = getattr(child, "title", "")
        if child_title == title or child_title.endswith(title):
            return child
    raise AssertionError(f"missing submenu item: {title}")


if __name__ == "__main__":
    unittest.main()
