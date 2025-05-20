import importlib
import sys
import types
import unittest


class DummyKey:
    def __init__(self, name: str) -> None:
        self.name = name

    def __str__(self) -> str:
        return f"Key.{self.name}"

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, DummyKey) and self.name == other.name


# Provide a common key instance used for testing
DummyKey.alt = DummyKey("alt")


class DummyKeyCode:
    def __init__(self, char: str) -> None:
        self.char = char

    @classmethod
    def from_char(cls, char: str) -> "DummyKeyCode":
        return cls(char)

    def __str__(self) -> str:
        return self.char

    def __repr__(self) -> str:
        return f"'{self.char}'"

    def __hash__(self) -> int:
        return hash(self.char)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, DummyKeyCode) and self.char == other.char


class HotkeyManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        # Create stub pynput module
        keyboard_module = types.ModuleType("keyboard")
        keyboard_module.Key = DummyKey
        keyboard_module.KeyCode = DummyKeyCode
        keyboard_module.Listener = object  # not used

        pynput_module = types.ModuleType("pynput")
        pynput_module.keyboard = keyboard_module

        patches = {
            "pynput": pynput_module,
            "pynput.keyboard": keyboard_module,
        }

        # Stub transclip.transcription before importing the package
        app_module = types.ModuleType("transclip.transcription")
        app_module.WhisperModelType = type("WhisperModelType", (), {})
        patches["transclip.transcription"] = app_module

        self.module_patch = unittest.mock.patch.dict(sys.modules, patches)
        self.module_patch.start()
        self.addCleanup(self.module_patch.stop)

        import transclip.hotkey as hotkey
        importlib.reload(hotkey)
        self.hotkey = hotkey

    def test_serialize_deserialize(self):
        manager = self.hotkey.HotkeyManager()
        k1 = DummyKey("alt")
        k2 = DummyKeyCode.from_char("a")
        manager.hotkey = (k1, k2)

        serialized = manager.serialize()
        self.assertEqual(serialized, [str(k1), str(k2)])

        new_manager = self.hotkey.HotkeyManager()
        new_manager.deserialize(serialized)
        self.assertEqual(new_manager.hotkey, manager.hotkey)
        self.assertTrue(new_manager.matches({k1, k2}))

    def test_deserialize_none(self):
        manager = self.hotkey.HotkeyManager()
        manager.deserialize(None)
        self.assertIsNone(manager.hotkey)
        self.assertFalse(manager.matches(set()))


if __name__ == "__main__":
    unittest.main()
