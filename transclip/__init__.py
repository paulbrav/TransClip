"""TransClip - Speech to text transcription with a single keypress.

This module provides functionality to record audio and transcribe it to text
using Faster Whisper when a configured key combination is pressed.
"""

__version__ = "0.1.0"

# Define what symbols are exported when using "from transclip import *"
__all__ = ["WhisperModelType"]

# Re-export the model enum for convenience
from .transcription import WhisperModelType
