import unittest
from pathlib import Path

from tests.service_helpers import FakeRuntime
from transclip.platform_runtime import user_cache_dir, user_config_dir, user_log_dir


class PlatformRuntimeTests(unittest.TestCase):
    def test_linux_paths_use_xdg_layout(self):
        runtime = FakeRuntime(system="Linux", home=Path("/home/test"))
        self.assertEqual(user_config_dir("transclip", runtime), Path("/home/test/.config/transclip"))
        self.assertEqual(user_cache_dir("transclip", runtime), Path("/home/test/.cache/transclip"))
        self.assertEqual(user_log_dir("transclip", runtime), Path("/home/test/.cache/transclip"))

    def test_darwin_paths_use_library_layout(self):
        runtime = FakeRuntime(system="Darwin", home=Path("/Users/test"))
        self.assertEqual(
            user_config_dir("transclip", runtime),
            Path("/Users/test/Library/Application Support/transclip"),
        )
        self.assertEqual(user_cache_dir("transclip", runtime), Path("/Users/test/Library/Caches/transclip"))
        self.assertEqual(user_log_dir("transclip", runtime), Path("/Users/test/Library/Logs/transclip"))


if __name__ == "__main__":
    unittest.main()
