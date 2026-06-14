"""Best-effort Windows toast notifications via the WinRT projection (pywinrt).

All entry points degrade to a no-op (returning False) when the winrt packages
are not installed or the call is made off Windows, so callers can fire a toast
unconditionally without guarding.
"""

from __future__ import annotations

import xml.sax.saxutils as saxutils
from typing import Any

from transclip.product import APP_USER_MODEL_ID, DISPLAY_NAME


def toast_xml(title: str, message: str) -> str:
    """Build a ToastGeneric notification payload with XML-escaped text."""
    return (
        '<toast><visual><binding template="ToastGeneric">'
        f"<text>{saxutils.escape(title)}</text>"
        f"<text>{saxutils.escape(message)}</text>"
        "</binding></visual></toast>"
    )


def recording_toast_message(ok: bool, action: str | None, *, paste_failed: bool, error: str = "") -> str | None:
    """Body text for a recording-toggle toast, or None when no toast is warranted.

    Covers the whole "recording on/off" story: started, stopped+pasted, the
    blocked-paste case (UIPI / elevated target, where the transcript is on the
    clipboard but did not paste), and a failed toggle.
    """
    if not ok:
        return f"Recording failed: {error or 'unknown error'}"
    if action == "started":
        return "Recording… press the hotkey again to stop"
    if action == "stopped":
        if paste_failed:
            return "Transcript copied — paste was blocked; press Ctrl+V to paste"
        return "Transcribed and pasted"
    return None


def _winrt_toast_api() -> tuple[Any, Any, Any]:
    """Import seam for the WinRT toast classes (kept tiny so it is mockable)."""
    from winrt.windows.data.xml.dom import XmlDocument
    from winrt.windows.ui.notifications import ToastNotification, ToastNotificationManager

    return XmlDocument, ToastNotification, ToastNotificationManager


def windows_toast(title: str, message: str, *, app_id: str = APP_USER_MODEL_ID) -> bool:
    """Show a toast attributed to ``app_id``; return whether it was delivered.

    ``create_toast_notifier_with_id`` is the AppUserModelID overload required by
    unpackaged apps (the no-arg variant is for packaged apps only).
    """
    try:
        xml_document, toast_notification, toast_manager = _winrt_toast_api()
    except (ImportError, OSError):
        return False
    try:
        document = xml_document()
        document.load_xml(toast_xml(title, message))
        notifier = toast_manager.create_toast_notifier_with_id(app_id)
        notifier.show(toast_notification(document))
        return True
    except Exception:
        return False


def register_toast_app_id(app_id: str = APP_USER_MODEL_ID, display_name: str = DISPLAY_NAME) -> bool:
    """Register the AUMID under HKCU so unpackaged toasts attribute correctly.

    Windows 11 will display a toast for an unregistered AUMID, but registering a
    DisplayName gives proper attribution and is needed for reliable display and
    Action Center persistence on Windows 10. Idempotent; best-effort.
    """
    try:
        import winreg
    except ImportError:
        return False
    try:
        key_path = rf"Software\Classes\AppUserModelId\{app_id}"
        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, key_path) as key:
            winreg.SetValueEx(key, "DisplayName", 0, winreg.REG_SZ, display_name)
        return True
    except OSError:
        return False
