import subprocess
import unittest

class StyleTests(unittest.TestCase):
    def test_mypy_cleanup(self):
        result = subprocess.run([
            "mypy",
            "transclip/cleanup.py",
        ], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)

    def test_ruff_cleanup(self):
        result = subprocess.run([
            "ruff",
            "check",
            "transclip/cleanup.py",
        ], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, msg=result.stdout + result.stderr)

if __name__ == "__main__":
    unittest.main()
