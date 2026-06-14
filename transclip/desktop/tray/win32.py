from __future__ import annotations

import gc
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from transclip.desktop.hotkey.windows import is_valid_hotkey, start_windows_hotkey
from transclip.desktop.notifications import recording_toast_message, register_toast_app_id, windows_toast
from transclip.desktop.win32_app import (
    acquire_single_instance,
    release_single_instance,
    set_app_user_model_id,
    set_dpi_awareness,
)
from transclip.platform.runtime import PlatformRuntime
from transclip.product import DISPLAY_NAME
from transclip.settings import Settings, patch_settings, settings_path

from .controller import TrayController, build_tray_action_callbacks
from .materialize import materialize_tray_menu
from .menu import tray_menu_nodes
from .menu_update import HistoryMenuState
from .session import TraySession
from .sinks.win32 import PystrayMenuSink
from .views import RefDrivenMenuView


def health_icon_color(name: str) -> str:
    """Tray icon fill for a health state, with a safe fallback.

    build_image runs inside the pystray event loop, so a KeyError here (e.g. a
    health state added later that this map does not know) would crash the tray
    rather than just show the wrong color.
    """
    return {"recording": "red", "ready": "green", "offline": "orange"}.get(name, "gray")


def _notify_toggle(outcome: Any, settings: Settings) -> None:
    """Toast the result of a recording toggle when notifications are enabled."""
    if not settings.tray_notifications:
        return
    message = recording_toast_message(
        outcome.ok,
        outcome.payload.get("action"),
        paste_failed=bool(outcome.paste_failed_message),
        error=outcome.error_message or "",
    )
    if message:
        windows_toast(DISPLAY_NAME, message)


def _notify_action(message: str, settings: Settings) -> None:
    """Toast a tray action (e.g. a service/model change) when enabled."""
    if settings.tray_notifications:
        windows_toast(DISPLAY_NAME, message)


def _capture_hotkey(
    pause: Callable[[], None],
    resume: Callable[[], None],
    read_hotkey: Callable[[], str | None],
) -> str | None:
    """Pause the live hotkey, capture a chord, then restore the hotkey.

    Pausing stops the keys pressed for capture from also triggering the
    currently registered global hotkey. resume() always runs (even if the read
    fails), and a failed read yields None rather than propagating.
    """
    pause()
    try:
        return read_hotkey()
    except Exception:
        return None
    finally:
        resume()


def run_windows_tray(
    settings: Settings,
    explicit_settings_path: Path | None = None,
    runtime: PlatformRuntime | None = None,
) -> int:
    # Opt into per-monitor-v2 DPI awareness before any window is created so the
    # tray dialog is not bitmap-stretched (blurry) on high-DPI displays.
    set_dpi_awareness()
    # Identify the process to the shell so the taskbar groups it as TransClip
    # and notifications are attributed correctly (rather than to python).
    set_app_user_model_id()
    register_toast_app_id()
    # A second tray would install a second global hotkey hook and double-fire
    # the toggle. Hold the mutex for the lifetime of this run and release it on
    # every exit path so the guard is re-entrant across runs.
    instance = acquire_single_instance()
    if instance is None:
        print("TransClip tray is already running.", file=sys.stderr)
        return 0
    try:
        import pystray
        from PIL import Image, ImageDraw
    except ImportError:
        print(
            "Windows tray UI requires pystray and Pillow. Install with: "
            "uv sync --extra windows-ui or pip install 'transclip[windows-ui]'",
            file=sys.stderr,
        )
        release_single_instance(instance)
        return 1

    session = TraySession(settings, explicit_settings_path, runtime)
    icon_holder: dict[str, Any] = {"icon": None}
    hotkey_holder: dict[str, Callable[[], None] | None] = {"stop": None}
    menu_refs: dict[str, Any] = {}
    history_state = HistoryMenuState(signature=object())

    def build_image(icon_name: str):
        image = Image.new("RGB", (64, 64), health_icon_color(icon_name))
        draw = ImageDraw.Draw(image)
        draw.ellipse((16, 16, 48, 48), fill="white")
        return image

    def restart_hotkey() -> None:
        stop = hotkey_holder["stop"]
        if stop is not None:
            stop()
        hotkey_holder["stop"] = start_windows_hotkey(on_hotkey, session.settings, session.runtime)

    def rebuild_history(entries) -> None:
        menu_refs["_history_entries"] = list(entries)

    def set_health_icon(icon: str) -> None:
        icon_obj = icon_holder["icon"]
        if icon_obj is not None:
            icon_obj.icon = build_image(icon)

    def refresh_menu_display() -> None:
        # pystray items are immutable; the setters above update backing state,
        # so the menu must be repainted for the new labels/enabled to show.
        icon_obj = icon_holder["icon"]
        if icon_obj is not None:
            icon_obj.update_menu()

    menu_view = RefDrivenMenuView(
        menu_refs,
        set_item_label=lambda item, text: setattr(item, "text", text),
        set_item_enabled=lambda item, enabled: setattr(item, "enabled", enabled),
        rebuild_history=rebuild_history,
        set_health_icon=set_health_icon,
        on_updated=refresh_menu_display,
    )
    controller = TrayController(
        session,
        menu_view,
        menu_refs,
        history_state=history_state,
        on_health_icon=lambda: menu_view.set_health_icon(session.health.icon),
    )

    def pause_hotkey() -> None:
        # Stop and clear the handle so resume (restart_hotkey) re-registers
        # cleanly instead of trying to remove the already-removed hotkey.
        stop = hotkey_holder["stop"]
        if stop is not None:
            stop()
            hotkey_holder["stop"] = None

    def capture_hotkey() -> str | None:
        try:
            import keyboard
        except ImportError:
            return None
        return _capture_hotkey(
            pause=pause_hotkey,
            resume=restart_hotkey,
            read_hotkey=lambda: keyboard.read_hotkey(suppress=False),
        )

    def set_hotkey(_icon=None, _item=None) -> None:
        _set_hotkey_dialog(session, restart_hotkey, capture_hotkey)
        controller.update_menu()

    def toggle_and_notify() -> object:
        outcome = controller.toggle_record()
        _notify_toggle(outcome, session.settings)
        return outcome

    action_callbacks = build_tray_action_callbacks(
        controller,
        session,
        set_hotkey=set_hotkey,
        quit=lambda: icon_holder["icon"].stop() if icon_holder["icon"] is not None else None,
    )
    # Notify on both toggle paths (global hotkey and the tray menu item).
    action_callbacks["toggle"] = lambda *_: toggle_and_notify()

    def _action_with_toast(message: str, action: Callable[[], object]) -> Callable[..., object]:
        def run(*_args: object) -> object:
            _notify_action(message, session.settings)
            return action()

        return run

    # These tray actions restart the service (a few seconds of unavailability),
    # which is otherwise invisible - so they get a heads-up toast.
    action_callbacks["start_service"] = _action_with_toast(
        "Starting the dictation service…", lambda: controller.run_tray_action(session.start_service)
    )
    action_callbacks["restart_service"] = _action_with_toast(
        "Restarting the dictation service…", lambda: controller.run_tray_action(session.restart_service)
    )
    action_callbacks["model_cleanup"] = _action_with_toast(
        "Toggling model cleanup — restarting the service…",
        lambda: controller.run_tray_action(session.toggle_model_cleanup),
    )

    def set_model_and_notify(model_id: str, backend: str) -> object:
        _notify_action("Switching ASR model — restarting the service…", session.settings)
        return session.set_asr_model(model_id, backend)

    def build_menu() -> pystray.Menu:
        items: list = []
        materialize_tray_menu(
            tray_menu_nodes(session.runtime.system()),
            session,
            PystrayMenuSink(
                items,
                menu_refs,
                pystray=pystray,
                after_action=controller.run_tray_action,
                set_model=set_model_and_notify,
                on_copy_history=controller.copy_history_text,
            ),
            action_callbacks=action_callbacks,
            initial_status_label=True,
            on_history_open=controller.refresh_history_menu,
            history_state=history_state,
        )
        return pystray.Menu(*items)

    def on_hotkey() -> None:
        # Runs on the keyboard library's low-level hook thread, not the pystray
        # message loop: it mutates menu-item backing state and calls
        # update_menu(). Writes are simple attribute sets, so the worst case is a
        # torn read that self-heals on the next repaint.
        toggle_and_notify()

    icon = pystray.Icon(
        DISPLAY_NAME,
        build_image(session.health.icon),
        DISPLAY_NAME,
        build_menu(),
    )
    icon_holder["icon"] = icon

    def setup(_icon) -> None:
        restart_hotkey()
        _icon.visible = True
        controller.refresh_health()
        controller.refresh_history_menu(force=True)

    try:
        icon.run(setup=setup)
    finally:
        stop = hotkey_holder["stop"]
        if stop is not None:
            stop()
        release_single_instance(instance)
    return 0


def _set_hotkey_dialog(
    session: TraySession,
    restart_hotkey: Callable[[], None],
    capture: Callable[[], str | None] | None = None,
) -> None:
    value, available = _prompt_hotkey(session.settings.hotkey_windows, capture)
    if not available:
        session.set_detail("tkinter is unavailable for hotkey dialog")
        return
    _apply_hotkey_selection(session, value, restart_hotkey)


def _prompt_hotkey(
    initial: str,
    capture: Callable[[], str | None] | None = None,
) -> tuple[str | None, bool]:
    """Prompt for a hotkey on a dedicated thread; return (value, tk_available).

    pystray runs its Win32 message loop on the main thread and invokes the menu
    action synchronously inside that loop, after making its own hidden window
    the foreground window. A tkinter dialog created there is nested inside
    pystray's message pump and can never take the foreground, so it renders but
    every click is dead. Running our own small dialog on a separate thread gives
    it a clean message queue and event loop; topmost + focus_force make it
    interactive. The caller blocks (join) until it closes - i.e. it is modal.

    When ``capture`` is supplied a "Record keys" button captures the pressed
    chord (on a worker thread, since it blocks); the entry is disabled and the
    buttons are inert during capture, so the press cannot close the dialog or
    type into the field, and the result is filled in via after() polling.
    """
    state: dict[str, Any] = {"value": None, "available": True}

    def run() -> None:
        try:
            import tkinter as tk
        except ImportError:
            state["available"] = False
            return
        root = tk.Tk()
        root.title("Set hotkey")
        root.resizable(False, False)
        root.attributes("-topmost", True)
        prompt = "Type a keyboard-library hotkey (e.g. ctrl+shift+space)"
        if capture is not None:
            prompt += ",\nor click Record keys and press the combination."
        tk.Label(root, text=prompt, justify="left").pack(padx=16, pady=(14, 6))
        entry = tk.Entry(root, width=44)
        entry.insert(0, initial)
        entry.select_range(0, "end")
        entry.pack(padx=16, pady=6)

        recording: dict[str, Any] = {"active": False, "result": None}
        record_button: Any = None

        def submit() -> None:
            if recording["active"]:
                return
            state["value"] = entry.get()
            root.destroy()

        def cancel() -> None:
            if recording["active"]:
                return
            state["value"] = None
            root.destroy()

        def poll_capture() -> None:
            if recording["active"]:
                root.after(100, poll_capture)
                return
            entry.config(state="normal")
            record_button.config(text="Record keys")
            chord = recording["result"]
            recording["result"] = None
            if chord:
                entry.delete(0, "end")
                entry.insert(0, chord)
                entry.select_range(0, "end")
            entry.focus_set()

        def begin_capture() -> None:
            if capture is None or recording["active"]:
                return
            recording["active"] = True
            record_button.config(text="Press your hotkey…")
            entry.config(state="disabled")

            def worker() -> None:
                # finally so a failure in capture() can never leave the dialog
                # stuck in the recording state (entry disabled, buttons inert).
                try:
                    recording["result"] = capture()
                finally:
                    recording["active"] = False

            threading.Thread(target=worker, name="transclip-hotkey-capture", daemon=True).start()
            poll_capture()

        buttons = tk.Frame(root)
        buttons.pack(pady=(6, 14))
        if capture is not None:
            record_button = tk.Button(buttons, text="Record keys", width=12, command=begin_capture)
            record_button.pack(side="left", padx=8)
        tk.Button(buttons, text="OK", width=10, command=submit).pack(side="left", padx=8)
        tk.Button(buttons, text="Cancel", width=10, command=cancel).pack(side="left", padx=8)
        root.bind("<Return>", lambda _event: submit())
        root.bind("<Escape>", lambda _event: cancel())
        root.protocol("WM_DELETE_WINDOW", cancel)
        root.lift()
        root.focus_force()
        entry.focus_set()
        root.mainloop()

    def run_and_cleanup() -> None:
        run()
        # The Tk interpreter was created on this thread; finalize it here, after
        # run()'s widgets are out of scope, so the cyclic GC does not free it on
        # the main thread - which aborts with
        # "Tcl_AsyncDelete: async handler deleted by the wrong thread".
        gc.collect()

    thread = threading.Thread(target=run_and_cleanup, name="transclip-hotkey-dialog", daemon=True)
    thread.start()
    thread.join()
    return state["value"], state["available"]


def _apply_hotkey_selection(
    session: TraySession,
    candidate: str | None,
    restart_hotkey: Callable[[], None],
) -> None:
    if not candidate or not candidate.strip():
        session.set_detail("Hotkey was not changed")
        return
    binding = candidate.strip()
    if not is_valid_hotkey(binding):
        # Reject before persisting: an unparseable binding would otherwise be
        # saved and then crash keyboard.add_hotkey on the next registration.
        session.set_detail(f"Invalid hotkey {binding!r}; keeping {session.settings.hotkey_windows!r}")
        return
    path = session.explicit_settings_path or settings_path()
    session.settings = patch_settings(path, hotkey_windows=binding)
    restart_hotkey()
    session.set_detail(f"Hotkey set to {binding}")
    if session.settings.tray_notifications:
        windows_toast(DISPLAY_NAME, f"Hotkey set to {binding}")
