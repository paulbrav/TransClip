import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tests.service_helpers import FakeRuntime
from transclip.doctor import (
    Check,
    check_asr_runtime,
    check_config_files,
    check_hotkey_readiness,
    check_microphone_devices,
    check_model_cache,
    check_paste_tools,
    checks_as_json,
    checks_as_text,
)
from transclip.gnome_shortcut import GnomeShortcutStatus
from transclip.models import hf_cache_dir
from transclip.settings import Settings


class DoctorTests(unittest.TestCase):
    def test_hf_cache_dir(self):
        self.assertEqual(
            hf_cache_dir("ibm-granite/granite-speech-4.1-2b"),
            "models--ibm-granite--granite-speech-4.1-2b",
        )

    def test_model_cache_checks_transformers_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(
                asr_model="local/asr",
                cleanup_model="local/cleanup",
                cleanup_runtime="transformers",
                model_cache_dir=tmp,
            )
            self.assertFalse(check_model_cache(settings).ok)
            (Path(tmp) / "models--local--asr").mkdir()
            (Path(tmp) / "models--local--cleanup").mkdir()
            self.assertTrue(check_model_cache(settings).ok)

    def test_model_cache_skips_rule_cleanup_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings(asr_model="local/asr", cleanup_runtime="rule", model_cache_dir=tmp)
            (Path(tmp) / "models--local--asr").mkdir()
            self.assertTrue(check_model_cache(settings).ok)

    def test_nar_asr_runtime_checks_flash_attn(self):
        torch = SimpleNamespace(version=SimpleNamespace(hip="6.4"))
        with patch.dict("sys.modules", {"flash_attn": object(), "torch": torch}):
            self.assertTrue(check_asr_runtime(Settings()).ok)

    def test_nar_asr_runtime_skips_flash_attn_on_macos(self):
        torch = SimpleNamespace(version=SimpleNamespace(hip=None))
        with patch.dict("sys.modules", {"torch": torch}):
            check = check_asr_runtime(Settings(), runtime=FakeRuntime(system="Darwin"))

        self.assertTrue(check.ok)
        self.assertIn("macOS", check.detail)

    def test_formatters(self):
        checks = [Check("thing", False, "missing thing")]
        self.assertIn('"name": "thing"', checks_as_json(checks))
        self.assertEqual(checks_as_text(checks), "missing\tthing\tmissing thing")

    def test_config_check_honors_config_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "settings.toml").write_text("", encoding="utf-8")
            self.assertTrue(check_config_files(root).ok)

    def test_wayland_paste_check_rejects_unusable_wtype(self):
        completed = type(
            "Completed",
            (),
            {
                "returncode": 1,
                "stdout": "Compositor does not support the virtual keyboard protocol\n",
            },
        )()

        runtime = FakeRuntime(
            env={"XDG_SESSION_TYPE": "wayland"},
            available={"wtype": "/usr/bin/wtype"},
            run_func=lambda _command, **_kwargs: completed,
        )
        check = check_paste_tools(runtime)

        self.assertFalse(check.ok)
        self.assertIn("virtual keyboard", check.detail)
        self.assertIn("ydotool", check.detail)

    def test_wayland_paste_check_accepts_working_wtype(self):
        completed = type("Completed", (), {"returncode": 0, "stdout": ""})()

        runtime = FakeRuntime(
            env={"XDG_SESSION_TYPE": "wayland"},
            available={"wtype": "/usr/bin/wtype"},
            run_func=lambda _command, **_kwargs: completed,
        )
        check = check_paste_tools(runtime)

        self.assertTrue(check.ok)
        self.assertIn("wtype", check.detail)

    def test_wayland_paste_check_accepts_ydotool_fallback(self):
        completed = type(
            "Completed",
            (),
            {
                "returncode": 1,
                "stdout": "Compositor does not support the virtual keyboard protocol\n",
            },
        )()

        runtime = FakeRuntime(
            env={"XDG_SESSION_TYPE": "wayland"},
            available={"wtype": "/usr/bin/wtype", "ydotool": "/usr/bin/ydotool"},
            run_func=lambda _command, **_kwargs: completed,
        )
        check = check_paste_tools(runtime)

        self.assertTrue(check.ok)
        self.assertIn("ydotool", check.detail)

    def test_gnome_hotkey_check_reports_installed_shortcut(self):
        status = GnomeShortcutStatus(
            installed=True,
            path="/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/transclip-toggle/",
            name="TransClip Toggle",
            binding="<Super><Shift>XF86TouchpadOff",
            command="/usr/bin/python -m transclip.cli toggle-record --paste",
            command_exists=True,
        )

        runtime = FakeRuntime(
            env={"XDG_SESSION_TYPE": "wayland", "XDG_CURRENT_DESKTOP": "GNOME"},
            available={"gsettings"},
        )
        with (
            patch("transclip.gnome_shortcut.get_gnome_shortcut_status", return_value=status),
        ):
            check = check_hotkey_readiness(Settings(), runtime)

        self.assertTrue(check.ok)
        self.assertIn("session=wayland", check.detail)
        self.assertIn("binding=<Super><Shift>XF86TouchpadOff", check.detail)
        self.assertIn("command_exists=True", check.detail)

    def test_gnome_hotkey_check_uses_configured_binding(self):
        status = GnomeShortcutStatus(
            installed=True,
            path="/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/transclip-toggle/",
            name="TransClip Toggle",
            binding="<Control><Alt>space",
            command="/usr/bin/python -m transclip.cli toggle-record --paste",
            command_exists=True,
        )

        runtime = FakeRuntime(
            env={"XDG_SESSION_TYPE": "wayland", "XDG_CURRENT_DESKTOP": "GNOME"},
            available={"gsettings"},
        )
        with (
            patch("transclip.gnome_shortcut.get_gnome_shortcut_status", return_value=status),
        ):
            check = check_hotkey_readiness(Settings(hotkey_linux="<Control><Alt>space"), runtime)

        self.assertTrue(check.ok)
        self.assertIn("binding=<Control><Alt>space", check.detail)

    def test_gnome_hotkey_check_recommends_installer_when_missing(self):
        status = GnomeShortcutStatus(False, None, None, None, None, False)

        runtime = FakeRuntime(
            env={"XDG_SESSION_TYPE": "wayland", "XDG_CURRENT_DESKTOP": "GNOME"},
            available={"gsettings"},
        )
        with (
            patch("transclip.gnome_shortcut.get_gnome_shortcut_status", return_value=status),
        ):
            check = check_hotkey_readiness(Settings(), runtime)

        self.assertFalse(check.ok)
        self.assertIn("transclip install-gnome-shortcut", check.detail)

    def test_microphone_check_uses_arecord_devices(self):
        output = """**** List of CAPTURE Hardware Devices ****
card 1: Generic_1 [HD-Audio Generic], device 0: ALC245 Analog [ALC245 Analog]
  Subdevices: 1/1
"""
        completed = type("Completed", (), {"returncode": 0, "stdout": output})()

        runtime = FakeRuntime(
            available={"arecord": "/usr/bin/arecord"},
            run_func=lambda _command, **_kwargs: completed,
        )
        check = check_microphone_devices(runtime)

        self.assertTrue(check.ok)
        self.assertIn("HD-Audio Generic", check.detail)

    def test_microphone_check_reports_missing_devices(self):
        completed = type("Completed", (), {"returncode": 1, "stdout": "no soundcards found..."})()

        runtime = FakeRuntime(
            available={"arecord": "/usr/bin/arecord"},
            run_func=lambda _command, **_kwargs: completed,
        )
        check = check_microphone_devices(runtime)

        self.assertFalse(check.ok)
        self.assertIn("arecord did not list", check.detail)


if __name__ == "__main__":
    unittest.main()
