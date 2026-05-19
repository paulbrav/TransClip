import unittest

from transclip.keyword_restore import restore_keywords


class KeywordRestoreTests(unittest.TestCase):
    def test_restores_explicit_technical_keywords(self):
        text = "Use w type on mac os with rockham and gfx 1151."

        restored = restore_keywords(text, ["wtype", "macOS", "ROCm", "gfx1151"])

        self.assertEqual(restored, "Use wtype on macOS with ROCm and gfx1151.")

    def test_leaves_text_without_keyword_alias_unchanged(self):
        text = "The service is ready."

        self.assertEqual(restore_keywords(text, ["ROCm"]), text)

    def test_restores_keyword_phrases(self):
        text = "Run the unit test after changing the python trend."

        restored = restore_keywords(text, ["Python tray"])

        self.assertEqual(restored, "Run the unit test after changing the Python tray.")


if __name__ == "__main__":
    unittest.main()
