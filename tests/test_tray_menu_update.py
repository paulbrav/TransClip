import unittest
from typing import Any
from unittest.mock import patch

from transclip.settings import Settings
from transclip.tray_menu_update import (
    HistoryMenuState,
    apply_menu_snapshot,
    compute_tray_menu_snapshot,
    history_preview_entries,
    refresh_history_menu,
    should_refresh_history,
)
from transclip.tray_session import TrayHealth, TraySession

from tests.service_helpers import FakeRuntime, patch_linux_gpu_runtime


class RecordingMenuView:
    def __init__(self) -> None:
        self.labels: dict[str, str] = {}
        self.enabled: dict[str, bool] = {}
        self.model_labels: list[tuple[Any, str]] = []
        self.history: list[tuple[str, str]] = []
        self.icon: str = ""

    def set_label(self, ref: str, text: str) -> None:
        self.labels[ref] = text

    def set_enabled(self, ref: str, enabled: bool) -> None:
        self.enabled[ref] = enabled

    def set_model_labels(self, rows: list[tuple[Any, str]]) -> None:
        self.model_labels = list(rows)

    def rebuild_history(self, entries: list[tuple[str, str]]) -> None:
        self.history = list(entries)

    def set_health_icon(self, icon: str) -> None:
        self.icon = icon


class TrayMenuUpdateTests(unittest.TestCase):
    def _session(self, *, recording: bool = False, latest: str = "") -> TraySession:
        with patch_linux_gpu_runtime():
            session = TraySession(Settings(), runtime=FakeRuntime(system="Linux", home="/home/test"))
        session.health = TrayHealth(
            status="recording" if recording else "ready",
            recording=recording,
            detail="",
            icon="recording" if recording else "ready",
        )
        session.latest = latest
        return session

    def test_snapshot_toggle_label_follows_recording_state(self):
        idle = compute_tray_menu_snapshot(self._session(recording=False))
        active = compute_tray_menu_snapshot(self._session(recording=True))

        self.assertEqual(idle.toggle_label, "Record")
        self.assertEqual(active.toggle_label, "Stop + paste")

    def test_snapshot_latest_enabled_reflects_latest_or_history(self):
        with patch_linux_gpu_runtime(), patch("transclip.tray_menu_update.latest_history_text", return_value=""):
            session = self._session()
            session.latest = ""
            empty = compute_tray_menu_snapshot(session)

            session.latest = "hello"
            with_latest = compute_tray_menu_snapshot(session)

        self.assertFalse(empty.latest_enabled)
        self.assertTrue(with_latest.latest_enabled)

    def test_should_refresh_history_skips_unchanged_signature(self):
        state = HistoryMenuState(signature=123)
        self.assertFalse(should_refresh_history(state, 123))
        self.assertTrue(should_refresh_history(state, 456))
        self.assertTrue(should_refresh_history(state, 123, force=True))

    def test_apply_menu_snapshot_updates_standard_refs(self):
        session = self._session(recording=True, latest="text")
        snapshot = compute_tray_menu_snapshot(session)
        view = RecordingMenuView()
        model_rows = [(object(), "label-a"), (object(), "label-b")]

        apply_menu_snapshot(snapshot, view, model_rows=model_rows)

        self.assertEqual(view.labels["status_item"], snapshot.status_label)
        self.assertEqual(view.labels["toggle_item"], "Stop + paste")
        self.assertTrue(view.enabled["latest_item"])
        self.assertEqual(view.model_labels, model_rows)

    def test_refresh_history_menu_skips_when_signature_unchanged(self):
        session = self._session()
        state = HistoryMenuState(signature=999, refreshing=False)
        view = RecordingMenuView()

        refresh_history_menu(session, state, view, signature=999)

        self.assertEqual(view.history, [])

    def test_refresh_history_menu_rebuilds_when_signature_changes(self):
        session = self._session()
        session.history_events = lambda limit=5: [{"text": "First transcript"}]  # type: ignore[method-assign]
        state = HistoryMenuState(signature=object(), refreshing=False)
        view = RecordingMenuView()

        refresh_history_menu(session, state, view, signature=123, force=True)

        self.assertEqual(len(view.history), 1)
        self.assertEqual(view.history[0][1], "First transcript")
        self.assertEqual(state.signature, 123)

    def test_history_preview_entries_returns_preview_and_full_text(self):
        session = self._session()
        session.history_events = lambda limit=5: [  # type: ignore[method-assign]
            {"text": "Line one"},
            {"text": "Line two"},
        ]

        entries = history_preview_entries(session)

        self.assertEqual(len(entries), 2)
        self.assertEqual(entries[0][1], "Line one")
        self.assertIn("Line one", entries[0][0])


if __name__ == "__main__":
    unittest.main()
