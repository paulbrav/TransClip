import subprocess
import unittest

from granite_speach.gnome_shortcut import (
    GNOME_CUSTOM_KEYBINDINGS_KEY,
    GNOME_MEDIA_KEYS_SCHEMA,
    GRANITE_SHORTCUT_BINDING,
    GRANITE_SHORTCUT_NAME,
    GRANITE_SHORTCUT_PATH,
    build_toggle_command,
    command_exists,
    install_gnome_shortcut,
    shortcut_readiness,
)
from tests.service_helpers import FakeRuntime


class FakeGSettings:
    def __init__(self):
        self.values = {
            (GNOME_MEDIA_KEYS_SCHEMA, GNOME_CUSTOM_KEYBINDINGS_KEY): "['/custom/keep/']",
            (
                "org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:/custom/keep/",
                "name",
            ): "'Other Shortcut'",
        }

    def run(self, command, **_kwargs):
        action = command[1]
        schema = command[2]
        key = command[3]
        if action == "get":
            value = self.values.get((schema, key))
            if value is None:
                raise subprocess.CalledProcessError(1, command)
            return type("Completed", (), {"stdout": value + "\n"})()
        if action == "set":
            self.values[(schema, key)] = command[4]
            return type("Completed", (), {"stdout": ""})()
        raise AssertionError(command)


class GnomeShortcutTests(unittest.TestCase):
    def test_installer_preserves_unrelated_shortcuts_and_is_idempotent(self):
        fake = FakeGSettings()
        command = "/venv/bin/python -m granite_speach.cli toggle-record --paste"
        runtime = FakeRuntime(available={"gsettings"})
        first = install_gnome_shortcut(command, runner=fake.run, runtime=runtime)
        second = install_gnome_shortcut(command, runner=fake.run, runtime=runtime)

        paths = fake.values[(GNOME_MEDIA_KEYS_SCHEMA, GNOME_CUSTOM_KEYBINDINGS_KEY)]
        self.assertIn("/custom/keep/", paths)
        self.assertEqual(paths.count(GRANITE_SHORTCUT_PATH), 1)
        self.assertEqual(first.path, GRANITE_SHORTCUT_PATH)
        self.assertEqual(second.path, GRANITE_SHORTCUT_PATH)
        schema = "org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:" + GRANITE_SHORTCUT_PATH
        self.assertEqual(fake.values[(schema, "name")], GRANITE_SHORTCUT_NAME)
        self.assertEqual(fake.values[(schema, "binding")], GRANITE_SHORTCUT_BINDING)
        self.assertEqual(fake.values[(schema, "command")], command)

    def test_command_exists_checks_absolute_program(self):
        self.assertFalse(command_exists("/definitely/missing/granite-speach toggle-record"))

    def test_command_exists_checks_shell_wrapper_payload(self):
        self.assertFalse(
            command_exists(
                "/bin/sh -lc 'mkdir -p \"$HOME/.cache/granite-speach\"; "
                "/definitely/missing/python -m granite_speach.cli toggle-record --paste'"
            )
        )

    def test_build_toggle_command_is_logging_wrapper(self):
        command = build_toggle_command()

        self.assertTrue(command.startswith("/bin/sh -lc "))
        self.assertIn("toggle-record --paste", command)
        self.assertIn("toggle-record.log", command)
        self.assertTrue(command_exists(command))

    def test_shortcut_readiness_owns_policy(self):
        status = shortcut_readiness(
            expected_binding="<Super><Shift>XF86TouchpadOff",
            runtime=FakeRuntime(
                env={"XDG_SESSION_TYPE": "wayland", "XDG_CURRENT_DESKTOP": "GNOME"},
                available={"gsettings"},
            ),
            runner=FakeGSettings().run,
        )

        self.assertFalse(status.ok)
        self.assertIn("granite-speach install-gnome-shortcut", status.detail)


if __name__ == "__main__":
    unittest.main()
