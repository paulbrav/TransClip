import json
from pathlib import Path
import plistlib
import unittest


ROOT = Path(__file__).resolve().parents[1]
TAURI_DIR = ROOT / "desktop" / "src-tauri"


class DesktopConfigTests(unittest.TestCase):
    def test_macos_permission_plist_is_configured(self):
        config = json.loads((TAURI_DIR / "tauri.conf.json").read_text(encoding="utf-8"))
        plist_name = config["bundle"]["macOS"]["infoPlist"]
        plist_path = TAURI_DIR / plist_name

        self.assertTrue(plist_path.exists())
        plist = plistlib.loads(plist_path.read_bytes())
        self.assertIn("NSMicrophoneUsageDescription", plist)
        self.assertIn("NSAppleEventsUsageDescription", plist)
        self.assertIn("microphone", plist["NSMicrophoneUsageDescription"].lower())
        self.assertIn("paste", plist["NSAppleEventsUsageDescription"].lower())

    def test_linux_webkit_microphone_permission_handler_is_wired(self):
        cargo_toml = (TAURI_DIR / "Cargo.toml").read_text(encoding="utf-8")
        lib_rs = (TAURI_DIR / "src" / "lib.rs").read_text(encoding="utf-8")

        self.assertIn('target.\'cfg(target_os = "linux")\'.dependencies', cargo_toml)
        self.assertIn('webkit2gtk = "2"', cargo_toml)
        self.assertIn("fn allow_linux_microphone_requests", lib_rs)
        self.assertIn("connect_permission_request", lib_rs)
        self.assertIn("UserMediaPermissionRequest", lib_rs)
        self.assertIn("request.allow()", lib_rs)
        self.assertIn("allow_linux_microphone_requests(app)", lib_rs)


if __name__ == "__main__":
    unittest.main()
