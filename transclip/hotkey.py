"""Hotkey handling module for TransClip.

This module provides functionality for configuring and detecting hotkey
combinations.
"""

from typing import Callable, Optional, Set, Tuple, Union, cast

from pynput import keyboard

KeyType = Union[keyboard.Key, keyboard.KeyCode]

class HotkeyManager:
    """Manages hotkey configuration and detection."""

    def __init__(self) -> None:
        """Initialize the hotkey manager."""
        self.current_keys: Set[KeyType] = set()
        self.hotkey: Optional[Tuple[KeyType, ...]] = None

    def start_capture(
        self,
        callback: Callable[[Optional[Tuple[KeyType, ...]]], None]
    ) -> None:
        """Start capturing a new hotkey combination.

        Args:
            callback: Function to call with the captured hotkey.
        """
        def on_press(key: Optional[KeyType]) -> None:
            if key is None:
                return
            try:
                self.current_keys.add(key)
            except AttributeError:
                pass

        def on_release(key: Optional[KeyType]) -> None:
            if key is None:
                return
            try:
                self.current_keys.remove(key)
                if not self.current_keys:  # All keys released
                    listener.stop()
                    # Convert to list and back to tuple to make sorting work
                    keys_list = list(self.current_keys)
                    keys_list.sort(key=str)  # Sort by string representation
                    self.hotkey = tuple(keys_list)
                    callback(self.hotkey)
            except KeyError:
                pass

        listener = keyboard.Listener(
            on_press=on_press,
            on_release=on_release
        )
        listener.start()

    def matches(self, pressed_keys: Set[KeyType]) -> bool:
        """Check if the pressed keys match the configured hotkey.

        Args:
            pressed_keys: Set of currently pressed keys.

        Returns:
            bool: True if the keys match the hotkey, False otherwise.
        """
        if not self.hotkey:
            return False
        return set(self.hotkey) == pressed_keys

    def serialize(self) -> Optional[list[str]]:
        """Serialize the current hotkey for storage.

        Returns:
            Optional[list[str]]: List of key names, or None if no hotkey is set.
        """
        if not self.hotkey:
            return None
        return [str(k) for k in self.hotkey]

    def deserialize(self, data: Optional[list[str]]) -> None:
        """Deserialize a stored hotkey configuration.

        Args:
            data: List of key names, or None.
        """
        if not data:
            self.hotkey = None
            return

        keys: list[KeyType] = []
        for key_str in data:
            try:
                # Handle special keys
                if key_str.startswith('Key.'):
                    key = cast(KeyType, getattr(keyboard.Key, key_str[4:]))
                else:
                    # Handle regular character keys
                    key = cast(KeyType, keyboard.KeyCode.from_char(key_str))
                keys.append(key)
            except (AttributeError, ValueError):
                continue

        # Sort keys by their string representation
        keys.sort(key=str)
        self.hotkey = tuple(keys) if keys else None
