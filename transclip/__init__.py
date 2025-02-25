"""TransClip - Speech to text transcription with a single keypress.

This module provides functionality to record audio and transcribe it to text
using Faster Whisper when a configured key combination is pressed.
"""

__version__ = "0.1.0"

from .app import WhisperModelType  # This makes WhisperModelType available at package level
