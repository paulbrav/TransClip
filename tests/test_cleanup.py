import unittest

from granite_speach.cleanup import (
    FaithfulRuleCleanupBackend,
    GemmaTransformersCleanupBackend,
    build_cleanup_backend,
    conservative_cleanup,
    faithful_cleanup_messages,
)
from granite_speach.settings import Settings


class CleanupTests(unittest.TestCase):
    def test_conservative_cleanup_punctuation_and_capitalization(self):
        self.assertEqual(
            conservative_cleanup("hello world this is PyTorch"),
            "Hello world this is PyTorch.",
        )

    def test_conservative_cleanup_restores_keyword_spellings(self):
        self.assertEqual(
            conservative_cleanup(
                "add pytorch rockham and gfxfx 1151 to mac os",
                ["PyTorch", "ROCm", "gfx1151", "macOS"],
            ),
            "Add PyTorch ROCm and gfx1151 to macOS.",
        )
        self.assertEqual(
            conservative_cleanup("avoid rewriting llama. C", ["llama.cpp"]),
            "Avoid rewriting llama.cpp.",
        )
        self.assertEqual(
            conservative_cleanup(
                "fix the tory status shell and release the global short get",
                ["Tauri", "global shortcut"],
            ),
            "Fix the Tauri status shell and release the global shortcut.",
        )
        self.assertEqual(
            conservative_cleanup(
                "avoid llama. Cpp, torory, radian, and global sh get",
                ["llama.cpp", "Tauri", "Radeon", "global shortcut"],
            ),
            "Avoid llama.cpp, Tauri, Radeon, and global shortcut.",
        )

    def test_cleanup_backend_returns_timing(self):
        result = FaithfulRuleCleanupBackend().cleanup("hello ,world", ["PyTorch"])
        self.assertEqual(result.text, "Hello, world.")
        self.assertIn("cleanup", result.timings_ms)
        self.assertEqual(result.backend, "rule-based")

    def test_cleanup_backend_selection_is_explicit(self):
        backend = build_cleanup_backend(Settings(model_cache_dir="/models"))
        self.assertIsInstance(backend, FaithfulRuleCleanupBackend)
        gemma = build_cleanup_backend(
            Settings(cleanup_runtime="transformers", model_cache_dir="/models")
        )
        self.assertIsInstance(gemma, GemmaTransformersCleanupBackend)
        self.assertTrue(gemma.local_files_only)
        self.assertEqual(gemma.cache_dir, "/models")
        self.assertIsInstance(
            build_cleanup_backend(Settings(cleanup_runtime="test_rule")),
            FaithfulRuleCleanupBackend,
        )
        with self.assertRaises(ValueError):
            build_cleanup_backend(Settings(cleanup_runtime="unknown"))

    def test_faithful_cleanup_messages_preserve_terms(self):
        messages = faithful_cleanup_messages("hello pytorch", ["PyTorch"])
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("Do not add facts", messages[0]["content"])
        self.assertIn("PyTorch", messages[1]["content"])


if __name__ == "__main__":
    unittest.main()
