"""Main application module for TransClip.

This module contains the core functionality for the TransClip application,
including audio recording, transcription, and clipboard integration via system tray.
The application uses the Whisper model for speech-to-text conversion.
"""

import logging
import subprocess
import sys
import time
from enum import StrEnum
from typing import Any, Dict, List, Optional

import numpy as np
import pyperclip
import sounddevice as sd
from faster_whisper import WhisperModel
from pynput import keyboard
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QApplication,
    QMenu,
    QSystemTrayIcon,
)
from scipy import signal

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [Line %(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


class WhisperModelType(StrEnum):
    """Available Whisper model types with their parameter sizes.

    Inherits from StrEnum to allow direct string usage without .value accessor.
    The value of each enum is the model identifier string used by faster-whisper.
    """
    TINY = "tiny"      # 39M parameters
    BASE = "base"      # 74M parameters
    SMALL = "small"    # 244M parameters
    MEDIUM = "medium"  # 769M parameters
    LARGE = "large"    # 1.5B parameters
    LARGE_V2 = "large-v2"  # 1.5B parameters (improved version)
    LARGE_V3 = "large-v3"  # 1.5B parameters (latest improved version)

    @classmethod
    def get_description(cls, model_type: 'WhisperModelType') -> str:
        """Get a description of the model including its parameter size.

        Args:
            model_type: The WhisperModelType to get description for.

        Returns:
            str: Description of the model including parameter size.
        """
        descriptions: Dict[WhisperModelType, str] = {
            cls.TINY: "Tiny (39M parameters)",
            cls.BASE: "Base (74M parameters)",
            cls.SMALL: "Small (244M parameters)",
            cls.MEDIUM: "Medium (769M parameters)",
            cls.LARGE: "Large (1.5B parameters)",
            cls.LARGE_V2: "Large-v2 (1.5B parameters, improved)",
            cls.LARGE_V3: "Large-v3 (1.5B parameters, latest)"
        }
        return descriptions[model_type]

# Configure pyperclip
try:
    # Check if xclip is available
    if subprocess.run(['which', 'xclip'], capture_output=True).returncode == 0:
        logger.info("Found xclip, setting as clipboard mechanism")
        pyperclip.set_clipboard("xclip")
    else:
        logger.warning("xclip not found, clipboard operations may fail")
except Exception as e:
    logger.error("Error configuring clipboard mechanism: %s", e)

# Constants
SAMPLE_RATE: int = 44100  # Hz - Changed to match default device sample rate
CHANNELS: int = 1
DTYPE = np.float32  # Remove explicit type annotation that causes mypy error

class TranscriptionWorker(QThread):
    """Worker thread for handling audio transcription.

    This class runs the Whisper model in a separate thread to avoid blocking
    the main application during transcription processing.
    """

    finished = pyqtSignal(str)

    def __init__(self, audio_data: np.ndarray, model: WhisperModel):
        """Initialize the transcription worker.

        Args:
            audio_data: The audio data to transcribe as a numpy array.
            model: The initialized Whisper model to use for transcription.
        """
        super().__init__()
        self.audio_data = audio_data
        self.model = model
        logger.info("TranscriptionWorker initialized")

    def run(self) -> None:
        """Run the transcription process.

        Processes the audio data using the Whisper model and emits the
        transcribed text when complete. Emits an empty string on error.
        """
        logger.info("\n=== TranscriptionWorker Run Started ===")
        try:
            # Log audio data properties
            logger.info(f"Audio data shape: {self.audio_data.shape}")
            logger.info(f"Non-zero samples: {np.count_nonzero(self.audio_data)}")
            logger.info(f"Max amplitude: {np.max(np.abs(self.audio_data))}")

            # Ensure audio is the right shape (samples,) instead of (samples, 1)
            if len(self.audio_data.shape) > 1:
                self.audio_data = self.audio_data.flatten()
                logger.info("Flattened audio data")

            logger.info("Starting transcription with Whisper model")
            segments, _info = self.model.transcribe(
                self.audio_data,
                language="en",
                beam_size=5,
                initial_prompt="The following is a transcription of spoken English:",
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )

            # Log each segment for debugging
            all_segments = list(segments)  # Convert generator to list
            logger.info(f"Transcribed {len(all_segments)} segments")
            for i, segment in enumerate(all_segments):
                logger.info(f"Segment {i}: '{segment.text}'")

            text = " ".join([segment.text for segment in all_segments])
            logger.info(f"Final combined text: '{text}'")

            # Emit the result
            logger.info("Emitting transcription result")
            self.finished.emit(text.strip())
            logger.info("Signal emitted successfully")

        except Exception as e:
            logger.error(f"Transcription failed: {e}", exc_info=True)  # Include full traceback
            self.finished.emit("")
            logger.info("Empty signal emitted due to error")

class TransClip(QObject):
    """Main TransClip application class.

    Handles audio recording, transcription, and clipboard integration.
    Uses a system tray icon for user interaction.
    """

    def __init__(self, model_type: WhisperModelType = WhisperModelType.BASE):
        """Initialize the TransClip application.

        Args:
            model_type: The Whisper model type to use for transcription.
                Defaults to WhisperModelType.BASE.
        """
        super().__init__()
        self.app = QApplication(sys.argv)
        self.recording = False
        self.key_pressed = False
        self.key_cooldown = 0.5  # seconds
        self.last_recording_time: float = 0.0  # Timestamp for the last recording cycle
        self.audio_data: List[np.ndarray] = []  # Add type annotation
        self.processing = False  # Flag to track if we're processing audio
        self.stream: Optional[sd.InputStream] = None
        self.transcription_worker: Optional[TranscriptionWorker] = None
        self.tray: Optional[QSystemTrayIcon] = None
        self.listener: Optional[keyboard.Listener] = None

        # Initialize Whisper model
        logger.info("Initializing Whisper model")
        try:
            self.model = WhisperModel(
                model_type,  # Use the provided model type
                device="cuda" if self.has_cuda() else "cpu",
                compute_type="float16" if self.has_cuda() else "float32"
            )
            logger.info(f"Using model: {WhisperModelType.get_description(model_type)}")
        except Exception as e:
            logger.error(f"Failed to initialize Whisper model: {e}")
            sys.exit(1)

        self.init_tray()
        self.init_keyboard_listener()

    def has_cuda(self) -> bool:
        """Check if CUDA is available for GPU acceleration.

        Returns:
            bool: True if CUDA is available, False otherwise.
        """
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def init_tray(self) -> None:
        """Initialize the system tray icon and menu.

        Creates a system tray icon with a context menu containing a quit option.
        """
        self.tray = QSystemTrayIcon()
        self.tray.setIcon(QIcon.fromTheme('audio-input-microphone'))

        menu = QMenu()
        quit_action = menu.addAction('Quit')
        assert quit_action is not None
        quit_action.triggered.connect(self.quit)

        self.tray.setContextMenu(menu)
        self.tray.show()

    def init_keyboard_listener(self) -> None:
        """Initialize the keyboard listener for the recording hotkey.

        Sets up a keyboard listener to detect when the Home key is pressed
        to start and stop recording.
        """
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        self.listener.start()

    def on_press(self, key: Any) -> None:
        """Handle key press events.

        Args:
            key: The key that was pressed.
        """
        try:
            current_time = time.time()
            if (not self.recording and
                not self.processing and
                not self.key_pressed and  # Ensure key wasn't already pressed
                key == keyboard.Key.home and
                (current_time - self.last_recording_time) > self.key_cooldown):

                logger.info("=== Key Press Event ===")
                logger.info(f"Time since last recording: {current_time - self.last_recording_time:.2f}s")
                self.key_pressed = True  # Mark key as pressed
                self.start_recording()
        except AttributeError:
            pass

    def on_release(self, key: Any) -> None:
        """Handle key release events.

        Args:
            key: The key that was released.
        """
        try:
            if self.recording and key == keyboard.Key.home and self.key_pressed:
                logger.info("=== Key Release Event ===")
                self.key_pressed = False  # Reset key press state
                self.last_recording_time = time.time()  # Update timestamp after complete cycle
                self.stop_recording()
        except AttributeError:
            pass

    def start_recording(self) -> None:
        """Start recording audio from the default input device.

        Sets up an audio stream with the correct number of channels for the 
        selected device and begins recording.
        """
        if self.processing:
            logger.warning("Cannot start recording while processing previous audio")
            return

        if time.time() - self.last_recording_time < self.key_cooldown:
            logger.warning("Recording blocked by cooldown")
            return

        logger.info("=== Starting recording ===")
        logger.debug(f"Previous audio_data length: {len(self.audio_data) if self.audio_data else 0}")
        logger.debug(f"Previous recording state: {self.recording}")

        self.recording = True
        self.audio_data = []  # Clear the buffer
        logger.info("Cleared audio buffer")

        # Log available audio devices
        devices = sd.query_devices()
        default_input = sd.query_devices(kind='input')
        logger.info(f"Default input device: {default_input}")
        logger.info(f"Available devices: {devices}")

        def callback(indata: np.ndarray, frames: int,
                    time_info: Dict[str, Any], status: sd.CallbackFlags) -> None:
            """Callback for audio recording.

            Args:
                indata: The input audio data.
                frames: Number of frames.
                time_info: Time information dictionary.
                status: Status flags.
            """
            if status:
                logger.warning(f'Recording error: {status}')
                for flag in dir(status):
                    if not flag.startswith('_'):
                        logger.warning(f'Status flag {flag}: {getattr(status, flag)}')
            if self.recording and np.max(np.abs(indata)) > 0:  # Only append if recording and non-zero audio
                logger.debug(f"Adding audio chunk: shape={indata.shape}, max_amplitude={np.max(np.abs(indata))}")
                self.audio_data.append(indata.copy())

        try:
            # Get the default input device
            default_device = sd.query_devices(kind='input')
            device_id = default_device['index']
            device_channels = int(default_device['max_input_channels'])
            device_sample_rate = int(default_device['default_samplerate'])
            
            logger.info(f"Using default input device (id: {device_id})")
            logger.info(f"  - Channels: {device_channels}")
            logger.info(f"  - Sample rate: {device_sample_rate} Hz")

            # Use the detected number of channels from the device
            self.stream = sd.InputStream(
                device=device_id,
                samplerate=device_sample_rate,
                channels=device_channels,  # Use device-specific channel count
                dtype=DTYPE,
                callback=callback
            )
            self.stream.start()
            logger.info(f"Successfully started recording with device {device_id}")
            if self.tray:
                self.tray.setIcon(QIcon.fromTheme("media-record"))
        except Exception as e:
            logger.exception(f"Failed to start recording: {e}")
            self.recording = False

    def stop_recording(self) -> None:
        """Stop recording and start transcription.

        Stops the audio stream, processes the recorded audio data,
        and starts the transcription process in a separate thread.
        """
        logger.info("\n=== stop_recording called ===")
        logger.info(f"Current audio_data chunks: {len(self.audio_data)}")
        logger.info(f"Recording state: {self.recording}")
        logger.info(f"Processing state: {self.processing}")

        if not self.recording or self.processing:
            logger.warning("stop_recording called but not recording or already processing")
            return

        logger.info("Stopping recording")
        self.recording = False
        self.processing = True  # Set processing flag
        try:
            if self.stream:
                self.stream.stop()
                self.stream.close()
                logger.info("Audio stream stopped and closed")
        except Exception as e:
            logger.exception(f"Error stopping recording: {e}")

        if self.tray:
            self.tray.setIcon(QIcon.fromTheme("audio-input-microphone"))

        # Convert audio data to numpy array
        if not self.audio_data:
            logger.warning("No audio data recorded")
            self.processing = False  # Reset processing flag since we're not continuing
            return

        try:
            audio = np.concatenate(self.audio_data, axis=0)
            logger.info(f"Concatenated audio shape: {audio.shape}")
            self.audio_data = []  # Clear the buffer after concatenation
            logger.info("Audio buffer cleared")

            # Get the actual sample rate used for recording
            if self.stream:
                device_sample_rate = self.stream.samplerate
                logger.info(f"Recording sample rate was: {device_sample_rate}")

                # If we have multiple channels, take the mean to get mono audio
                if audio.shape[1] > 1:
                    logger.info(f"Converting {audio.shape[1]} channels to mono")
                    audio = np.mean(audio, axis=1)

                # Normalize audio to be between -1 and 1
                if np.max(np.abs(audio)) > 0:
                    audio = audio / np.max(np.abs(audio))
                    logger.info(f"Normalized audio - new max amplitude: {np.max(np.abs(audio))}")

                # Log audio statistics
                logger.info(f"Audio statistics: mean={np.mean(audio):.4f}, "
                           f"std={np.std(audio):.4f}, "
                           f"min={np.min(audio):.4f}, "
                           f"max={np.max(audio):.4f}")

                # Resample audio to 16000 Hz for Whisper
                if device_sample_rate != 16000:
                    audio = signal.resample(audio, int(len(audio) * 16000 / device_sample_rate))
                    logger.info(f"Resampled audio from {device_sample_rate}Hz to 16000Hz, new shape: {audio.shape}")

                # Start transcription in a separate thread
                logger.info("\n=== Starting Transcription ===")
                self.transcription_worker = TranscriptionWorker(audio, self.model)
                self.transcription_worker.finished.connect(self.on_transcription_complete)
                logger.debug("Starting worker thread")
                self.transcription_worker.start()
                logger.info("Worker thread started successfully")
            else:
                logger.error("Stream is None, cannot process audio")
                self.processing = False

        except Exception as e:
            logger.error(f"Error processing audio data: {e}")
            self.processing = False  # Reset processing flag on error

    def on_transcription_complete(self, text: str) -> None:
        """Handle completed transcription.

        Args:
            text: The transcribed text to copy to clipboard.
        """
        try:
            logger.info("\n=== Transcription Complete ===")
            logger.debug(f"Thread ID: {QThread.currentThread()}")
            logger.debug(f"Received text: '{text}'")

            self.processing = False  # Reset processing flag

            if text:
                logger.info(f"Copying text to clipboard (length: {len(text)})")
                try:
                    # Try to use xclip directly for Linux
                    try:
                        import subprocess
                        process = subprocess.Popen(['xclip', '-selection', 'clipboard'],
                                                 stdin=subprocess.PIPE,
                                                 text=True)
                        process.communicate(input=text)
                        logger.debug("Text copied to clipboard using xclip")
                    except Exception as e:
                        logger.error(f"Direct xclip call failed: {e}")
                        # Fallback to pyperclip
                        pyperclip.copy(text)
                        logger.debug("Text copied to clipboard using pyperclip")

                    # Verify copy success
                    verification = pyperclip.paste()
                    if verification != text:
                        logger.error("Clipboard verification failed. Text may not be copied correctly.")

                except Exception as e:
                    logger.error(f"Failed to copy to clipboard: {e}")

                if self.tray:
                    self.tray.showMessage(
                        'TransClip',
                        'Transcription copied to clipboard',
                        QSystemTrayIcon.Information,  # Use the enum value
                        2000
                    )
            else:
                logger.warning("Empty transcription result")
                if self.tray:
                    self.tray.showMessage(
                        'TransClip',
                        'Transcription failed',
                        QSystemTrayIcon.Warning,  # Use the enum value
                        2000
                    )
        except Exception as overall_e:
            logger.error(f"Exception in on_transcription_complete: {overall_e}", exc_info=True)

    def quit(self) -> None:
        """Quit the application.

        Stops recording if active and exits the application.
        """
        if self.recording:
            self.stop_recording()
        self.app.quit()

    def run(self) -> int:
        """Run the application.

        Returns:
            int: The application exit code.
        """
        return self.app.exec_()

    def __del__(self) -> None:
        """Destructor for TransClip.

        Performs cleanup when the TransClip object is destroyed.
        """
        logger.info("TransClip object being destroyed")

def main() -> int:
    """Start the application.

    Returns:
        int: Exit code, 0 for success, 1 for error.
    """
    try:
        app = TransClip(WhisperModelType.SMALL)
        return app.run()
    except Exception as e:
        logger.error(f"Application error: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
