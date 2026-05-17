import unittest
from pathlib import Path
import tempfile

from unittest.mock import patch

from granite_speach.doctor import (
    Check,
    check_asr_runtime,
    check_config_files,
    check_evdev_hold_to_talk_readiness,
    check_hotkey_readiness,
    check_microphone_devices,
    check_model_cache,
    check_paste_tools,
    check_tauri_linux_libs,
    checks_as_json,
    checks_as_text,
    hf_cache_dir,
)
from granite_speach.gnome_shortcut import GnomeShortcutStatus
from granite_speach.settings import Settings


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
        with patch.dict("sys.modules", {"flash_attn": object()}):
            self.assertTrue(check_asr_runtime(Settings()).ok)

    def test_formatters(self):
        checks = [Check("thing", False, "missing thing")]
        self.assertIn('"name": "thing"', checks_as_json(checks))
        self.assertEqual(checks_as_text(checks), "missing\tthing\tmissing thing")

    def test_config_check_honors_config_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "settings.toml").write_text("", encoding="utf-8")
            (root / "keywords.txt").write_text("", encoding="utf-8")
            self.assertTrue(check_config_files(root).ok)

    def test_linux_lib_check_includes_apt_guidance(self):
        check = check_tauri_linux_libs()
        if not check.ok:
            self.assertIn("libwebkit2gtk-4.1-dev", check.detail)
            self.assertIn("libayatana-appindicator3-dev", check.detail)

    def test_wayland_paste_check_rejects_unusable_wtype(self):
        completed = type(
            "Completed",
            (),
            {
                "returncode": 1,
                "stdout": "Compositor does not support the virtual keyboard protocol\n",
            },
        )()
        with (
            patch("granite_speach.doctor.platform.system", return_value="Linux"),
            patch("granite_speach.doctor.os_environ", return_value="wayland"),
            patch("granite_speach.doctor.shutil.which", side_effect=lambda name: f"/usr/bin/{name}" if name == "wtype" else None),
            patch("granite_speach.doctor.subprocess.run", return_value=completed),
        ):
            check = check_paste_tools()

        self.assertFalse(check.ok)
        self.assertIn("virtual keyboard", check.detail)
        self.assertIn("ydotool", check.detail)

    def test_wayland_paste_check_accepts_working_wtype(self):
        completed = type("Completed", (), {"returncode": 0, "stdout": ""})()
        with (
            patch("granite_speach.doctor.platform.system", return_value="Linux"),
            patch("granite_speach.doctor.os_environ", return_value="wayland"),
            patch("granite_speach.doctor.shutil.which", side_effect=lambda name: f"/usr/bin/{name}" if name == "wtype" else None),
            patch("granite_speach.doctor.subprocess.run", return_value=completed),
        ):
            check = check_paste_tools()

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

        def which(name):
            return f"/usr/bin/{name}" if name in {"wtype", "ydotool"} else None

        with (
            patch("granite_speach.doctor.platform.system", return_value="Linux"),
            patch("granite_speach.doctor.os_environ", return_value="wayland"),
            patch("granite_speach.doctor.shutil.which", side_effect=which),
            patch("granite_speach.doctor.subprocess.run", return_value=completed),
        ):
            check = check_paste_tools()

        self.assertTrue(check.ok)
        self.assertIn("ydotool", check.detail)

    def test_gnome_hotkey_check_reports_installed_shortcut(self):
        status = GnomeShortcutStatus(
            installed=True,
            path="/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/granite-speach-toggle/",
            name="Granite Speach Toggle",
            binding="<Super><Shift>XF86TouchpadOff",
            command="/usr/bin/python -m granite_speach.cli toggle-record --paste",
            command_exists=True,
        )
        with (
            patch("granite_speach.doctor.platform.system", return_value="Linux"),
            patch("granite_speach.doctor.os_environ", side_effect=lambda name: {"XDG_SESSION_TYPE": "wayland", "XDG_CURRENT_DESKTOP": "GNOME"}.get(name)),
            patch("granite_speach.doctor.shutil.which", return_value="/usr/bin/gsettings"),
            patch("granite_speach.doctor.get_gnome_shortcut_status", return_value=status),
        ):
            check = check_hotkey_readiness()

        self.assertTrue(check.ok)
        self.assertIn("session=wayland", check.detail)
        self.assertIn("binding=<Super><Shift>XF86TouchpadOff", check.detail)
        self.assertIn("command_exists=True", check.detail)

    def test_gnome_hotkey_check_recommends_installer_when_missing(self):
        status = GnomeShortcutStatus(False, None, None, None, None, False)
        with (
            patch("granite_speach.doctor.platform.system", return_value="Linux"),
            patch("granite_speach.doctor.os_environ", side_effect=lambda name: {"XDG_SESSION_TYPE": "wayland", "XDG_CURRENT_DESKTOP": "GNOME"}.get(name)),
            patch("granite_speach.doctor.shutil.which", return_value="/usr/bin/gsettings"),
            patch("granite_speach.doctor.get_gnome_shortcut_status", return_value=status),
        ):
            check = check_hotkey_readiness()

        self.assertFalse(check.ok)
        self.assertIn("granite-speach install-gnome-shortcut", check.detail)

    def test_evdev_hold_to_talk_check_reports_readable_input_events(self):
        with (
            patch("granite_speach.doctor.platform.system", return_value="Linux"),
            patch("granite_speach.doctor.os_environ", return_value="wayland"),
            patch("granite_speach.doctor.global_shortcuts_portal_present", return_value=False),
            patch("granite_speach.doctor.readable_input_events", return_value=([Path("/dev/input/event0")], [Path("/dev/input/event0")])),
        ):
            check = check_evdev_hold_to_talk_readiness()

        self.assertTrue(check.ok)
        self.assertIn("session=wayland", check.detail)
        self.assertIn("readable /dev/input events", check.detail)

    def test_evdev_hold_to_talk_check_recommends_input_group(self):
        with (
            patch("granite_speach.doctor.platform.system", return_value="Linux"),
            patch("granite_speach.doctor.os_environ", return_value="wayland"),
            patch("granite_speach.doctor.global_shortcuts_portal_present", return_value=False),
            patch("granite_speach.doctor.readable_input_events", return_value=([Path("/dev/input/event0")], [])),
        ):
            check = check_evdev_hold_to_talk_readiness()

        self.assertFalse(check.ok)
        self.assertIn("sudo usermod -aG input $USER", check.detail)
        self.assertIn("GlobalShortcuts portal present: False", check.detail)

    def test_microphone_check_uses_arecord_devices(self):
        output = """**** List of CAPTURE Hardware Devices ****
card 1: Generic_1 [HD-Audio Generic], device 0: ALC245 Analog [ALC245 Analog]
  Subdevices: 1/1
"""
        completed = type("Completed", (), {"returncode": 0, "stdout": output})()
        with (
            patch("granite_speach.doctor.platform.system", return_value="Linux"),
            patch("granite_speach.doctor.shutil.which", side_effect=lambda name: f"/usr/bin/{name}" if name == "arecord" else None),
            patch("granite_speach.doctor.subprocess.run", return_value=completed),
        ):
            check = check_microphone_devices()

        self.assertTrue(check.ok)
        self.assertIn("HD-Audio Generic", check.detail)

    def test_microphone_check_reports_missing_devices(self):
        completed = type("Completed", (), {"returncode": 1, "stdout": "no soundcards found..."})()
        with (
            patch("granite_speach.doctor.platform.system", return_value="Linux"),
            patch("granite_speach.doctor.shutil.which", side_effect=lambda name: f"/usr/bin/{name}" if name == "arecord" else None),
            patch("granite_speach.doctor.subprocess.run", return_value=completed),
        ):
            check = check_microphone_devices()

        self.assertFalse(check.ok)
        self.assertIn("arecord did not list", check.detail)


if __name__ == "__main__":
    unittest.main()
