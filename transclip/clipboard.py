from __future__ import annotations

"""Clipboard helper functions."""

import logging
import subprocess
import sys

import pyperclip
from pynput import keyboard
from pynput.keyboard import Controller

logger = logging.getLogger(__name__)

# Configure pyperclip to use xclip when available
try:  # pragma: no cover - runtime check
    if subprocess.run(["which", "xclip"], capture_output=True).returncode == 0:
        pyperclip.set_clipboard("xclip")
except Exception as exc:  # pragma: no cover - logging
    logger.error("Clipboard setup failed: %s", exc)


def copy_to_clipboard(text: str) -> None:
    """Copy text to the system clipboard."""
    try:
        try:
            process = subprocess.Popen([
                "xclip",
                "-selection",
                "clipboard",
            ], stdin=subprocess.PIPE, text=True)
            process.communicate(input=text)
        except Exception:
            pyperclip.copy(text)
    except Exception as exc:  # pragma: no cover - logging
        logger.error("Failed to copy to clipboard: %s", exc)


def perform_paste() -> None:
    """Simulate a Ctrl+V (or Cmd+V on macOS) keypress."""
    try:
        ctrl = Controller()
        if sys.platform == "darwin":
            with ctrl.pressed(keyboard.Key.cmd):
                ctrl.press("v")
                ctrl.release("v")
        else:
            with ctrl.pressed(keyboard.Key.ctrl):
                ctrl.press("v")
                ctrl.release("v")
    except Exception as exc:  # pragma: no cover - logging
        logger.error("Failed to simulate paste: %s", exc)
