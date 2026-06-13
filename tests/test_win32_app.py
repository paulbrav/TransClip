import sys
import unittest


class DpiAwarenessTests(unittest.TestCase):
    def test_prefers_per_monitor_v2_context(self) -> None:
        from transclip.desktop.win32_app import set_dpi_awareness

        result = set_dpi_awareness(
            set_context=lambda: True,
            set_shcore=lambda: True,
            set_legacy=lambda: True,
        )
        self.assertEqual(result, "per-monitor-v2")

    def test_falls_back_through_shcore_then_legacy_then_unaware(self) -> None:
        from transclip.desktop.win32_app import set_dpi_awareness

        self.assertEqual(
            set_dpi_awareness(set_context=lambda: False, set_shcore=lambda: True, set_legacy=lambda: True),
            "per-monitor",
        )
        self.assertEqual(
            set_dpi_awareness(set_context=lambda: False, set_shcore=lambda: False, set_legacy=lambda: True),
            "system",
        )
        self.assertEqual(
            set_dpi_awareness(set_context=lambda: False, set_shcore=lambda: False, set_legacy=lambda: False),
            "unaware",
        )


@unittest.skipUnless(sys.platform == "win32", "exercises the real DPI awareness API")
class DpiAwarenessLiveTests(unittest.TestCase):
    def test_process_reports_per_monitor_awareness(self) -> None:
        import ctypes
        from ctypes import wintypes

        from transclip.desktop.win32_app import set_dpi_awareness

        set_dpi_awareness()
        user32 = ctypes.windll.user32
        user32.GetThreadDpiAwarenessContext.restype = wintypes.HANDLE
        user32.GetAwarenessFromDpiAwarenessContext.argtypes = [wintypes.HANDLE]
        user32.GetAwarenessFromDpiAwarenessContext.restype = ctypes.c_int
        awareness = user32.GetAwarenessFromDpiAwarenessContext(user32.GetThreadDpiAwarenessContext())
        # DPI_AWARENESS_PER_MONITOR_AWARE == 2 (PMv2 also reports as 2 here).
        self.assertEqual(awareness, 2)


if __name__ == "__main__":
    unittest.main()
