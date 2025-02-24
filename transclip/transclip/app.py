"""Main application module for TransClip.

This module contains the core functionality for the TransClip application,
including audio recording, transcription, and system tray integration.
"""

import logging
import sys
from enum import Enum
from typing import List
import time

import numpy as np
import pyperclip
import sounddevice as sd
from faster_whisper import WhisperModel
from pynput import keyboard
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, QObject
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QApplication,
    QMenu,
    QSystemTrayIcon,
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [Line %(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


class WhisperModelType(str, Enum):
    """Available Whisper model types with their parameter sizes.
    
    Inherits from str and Enum to allow direct string usage without .value accessor.
    The value of each enum is the model identifier string used by faster-whisper.
    The comment indicates the number of parameters in the model.
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
        descriptions = {
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
    import subprocess
    # Check if xclip is available
    if subprocess.run(['which', 'xclip'], capture_output=True).returncode == 0:
        logger.info("Found xclip, setting as clipboard mechanism")
        pyperclip.set_clipboard("xclip")
    else:
        logger.warning("xclip not found, clipboard operations may fail")
except Exception as e:
    logger.error("Error configuring clipboard mechanism: %s", e)

# Constants
SAMPLE_RATE = 44100  # Hz - Changed to match default device sample rate
CHANNELS = 1
DTYPE = np.float32

class TranscriptionWorker(QThread):
    """Worker thread for handling audio transcription."""

    finished = pyqtSignal(str)

    def __init__(self, audio_data: np.ndarray, model: WhisperModel):
        """Initialize the transcription worker."""
        super().__init__()
        self.audio_data = audio_data
        self.model = model
        logger.info("TranscriptionWorker initialized")

    def run(self):
        """Run the transcription process."""
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
            segments, info = self.model.transcribe(
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

    def on_transcription_complete(self, text: str):
        """Handle completed transcription."""
        try:
            logger.info("\n=== Transcription Complete ===")
            logger.info(f"Thread ID: {QThread.currentThread()}")
            logger.info(f"Received text: '{text}'")

            self.processing = False  # Reset processing flag
            
            if text:
                logger.info(f"Attempting to copy text (length: {len(text)}) to clipboard")
                try:
                    # Get current clipboard mechanism
                    logger.info(f"Current clipboard mechanism: {pyperclip.determine_clipboard()}")

                    # Attempt copy using subprocess directly
                    try:
                        import subprocess
                        process = subprocess.Popen(['xclip', '-selection', 'clipboard'], 
                                                 stdin=subprocess.PIPE, 
                                                 text=True)
                        process.communicate(input=text)
                        logger.info("Text copied to clipboard using xclip")
                    except Exception as e:
                        logger.error(f"Direct xclip call failed: {e}")
                        # Fallback to pyperclip
                        try:
                            pyperclip.copy(text)
                            logger.info("Text copied to clipboard using pyperclip")
                        except Exception as e:
                            logger.error(f"pyperclip.copy() failed: {e}")

                    # Verify copy
                    try:
                        verification = pyperclip.paste()
                        if verification == text:
                            logger.info("Clipboard verification successful")
                        else:
                            logger.error(f"Clipboard verification failed. Expected '{text}', got '{verification}'")
                    except Exception as e:
                        logger.error(f"Could not verify clipboard content: {e}")

                except Exception as e:
                    logger.error(f"Failed to copy to clipboard: {e}")

                self.tray.showMessage(
                    'TransClip',
                    'Transcription copied to clipboard',
                    QSystemTrayIcon.Information,
                    2000
                )
            else:
                logger.warning("Empty transcription result")
                self.tray.showMessage(
                    'TransClip',
                    'Transcription failed',
                    QSystemTrayIcon.Warning,
                    2000
                )
        except Exception as overall_e:
            logger.error(f"Exception in on_transcription_complete: {overall_e}", exc_info=True)

class TransClip(QObject):
    """Main TransClip application class."""

    test_signal = pyqtSignal()  # Define a test signal

    def __init__(self, model_type: WhisperModelType = WhisperModelType.BASE):
        """Initialize the TransClip application."""
        super().__init__()
        self.app = QApplication(sys.argv)
        self.recording = False
        self.audio_data: List[np.ndarray] = []
        self.last_recording_time = 0  # Timestamp for the last recording cycle
        self.key_cooldown = 2.0  # Increased cooldown period in seconds
        self.processing = False  # Add flag to track if we're processing audio
        self.key_pressed = False  # Track if key is currently pressed

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

        # Ensure signal-slot connections are made in the main thread
        self.transcription_worker = None

        # Add a timer for testing
        self.test_timer = QTimer(self)
        self.test_timer.timeout.connect(self.test_slot)
        self.test_timer.start(1000)  # Emit every 1 second

    def has_cuda(self) -> bool:
        """Check if CUDA is available.

        Returns:
            bool: True if CUDA is available, False otherwise.
        """
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def init_tray(self):
        """Initialize the system tray icon and menu."""
        self.tray = QSystemTrayIcon()
        self.tray.setIcon(QIcon.fromTheme('audio-input-microphone'))

        menu = QMenu()
        quit_action = menu.addAction('Quit')
        quit_action.triggered.connect(self.quit)

        self.tray.setContextMenu(menu)
        self.tray.show()

    def init_keyboard_listener(self):
        """Initialize the keyboard listener for the recording hotkey."""
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release
        )
        self.listener.start()

    def on_press(self, key):
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

    def on_release(self, key):
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

    def start_recording(self):
        """Start recording audio."""
        if self.processing:
            logger.warning("Cannot start recording while processing previous audio")
            return

        if time.time() - self.last_recording_time < self.key_cooldown:
            logger.warning("Recording blocked by cooldown")
            return

        logger.info("=== Starting recording ===")
        logger.info(f"Previous audio_data length: {len(self.audio_data) if self.audio_data else 0}")
        logger.info(f"Previous recording state: {self.recording}")
        
        self.recording = True
        self.audio_data = []  # Clear the buffer
        logger.info("Cleared audio buffer")

        # Log available audio devices
        devices = sd.query_devices()
        default_input = sd.query_devices(kind='input')
        logger.info(f"Default input device: {default_input}")
        logger.info(f"Available devices: {devices}")

        def callback(indata: np.ndarray, frames: int,
                    time_info: dict, status: sd.CallbackFlags):
            """Callback for audio recording."""
            if status:
                logger.warning(f'Recording error: {status}')
                for flag in dir(status):
                    if not flag.startswith('_'):
                        logger.warning(f'Status flag {flag}: {getattr(status, flag)}')
            if self.recording and np.max(np.abs(indata)) > 0:  # Only append if recording and non-zero audio
                logger.debug(f"Adding audio chunk: shape={indata.shape}, max_amplitude={np.max(np.abs(indata))}")
                self.audio_data.append(indata.copy())

        # TODO: make this support more devices
        try:
            # Try to use Bose QC Earbuds II first
            bose_device = sd.query_devices(device=9)
            logger.info(f"Bose device info: {bose_device}")
            logger.info("Attempting to query Bose device capabilities:")
            try:
                supported_formats = sd.query_devices(device=9, kind='input')
                logger.info(f"  Supported formats: {supported_formats}")
            except Exception as e:
                logger.error(f"  Failed to query device formats: {e}")

            device_sample_rate = int(bose_device['default_samplerate'])
            logger.info(f"Using Bose device sample rate: {device_sample_rate}")
            logger.info(f"Attempting to open stream with channels={CHANNELS}")

            self.stream = sd.InputStream(
                device=9,  # Bose QC Earbuds II
                samplerate=device_sample_rate,
                channels=CHANNELS,
                dtype=DTYPE,
                callback=callback
            )
            self.stream.start()
            logger.info("Successfully started recording with Bose QC Earbuds II")
            self.tray.setIcon(QIcon.fromTheme('media-record'))
        except Exception as e:
            logger.error(f"Failed to start recording with Bose QC Earbuds II: {e}")
            try:
                # Fallback to default device
                default_device = sd.query_devices(kind='input')
                device_sample_rate = int(default_device['default_samplerate'])
                logger.info(f"Using default device sample rate: {device_sample_rate}")

                self.stream = sd.InputStream(
                    samplerate=device_sample_rate,
                    channels=CHANNELS,
                    dtype=DTYPE,
                    callback=callback
                )
                self.stream.start()
                logger.info("Successfully started recording with default device")
                self.tray.setIcon(QIcon.fromTheme('media-record'))
            except Exception as e:
                logger.error(f"Failed to start recording with default device: {e}")
                self.recording = False

    def stop_recording(self):
        """Stop recording and start transcription."""
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
            self.stream.stop()
            self.stream.close()
            logger.info("Audio stream stopped and closed")
        except Exception as e:
            logger.error(f"Error stopping recording: {e}")

        self.tray.setIcon(QIcon.fromTheme('audio-input-microphone'))

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
            device_sample_rate = self.stream.samplerate
            logger.info(f"Recording sample rate was: {device_sample_rate}")

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
                from scipy import signal
                audio = signal.resample(audio, int(len(audio) * 16000 / device_sample_rate))
                logger.info(f"Resampled audio from {device_sample_rate}Hz to 16000Hz, new shape: {audio.shape}")

            # Start transcription in a separate thread
            logger.info("\n=== Starting Transcription ===")
            self.transcription_worker = TranscriptionWorker(audio, self.model)
            self.transcription_worker.finished.connect(self.on_transcription_complete)
            logger.info("Starting worker thread")
            self.transcription_worker.start()
            logger.info("Worker thread started successfully")

        except Exception as e:
            logger.error(f"Error processing audio data: {e}")
            self.processing = False  # Reset processing flag on error

    def on_transcription_complete(self, text: str):
        """Handle completed transcription."""
        try:
            logger.info("\n=== Transcription Complete ===")
            logger.info(f"Thread ID: {QThread.currentThread()}")
            logger.info(f"Received text: '{text}'")

            self.processing = False  # Reset processing flag
            
            if text:
                logger.info(f"Attempting to copy text (length: {len(text)}) to clipboard")
                try:
                    # Get current clipboard mechanism
                    logger.info(f"Current clipboard mechanism: {pyperclip.determine_clipboard()}")

                    # Attempt copy using subprocess directly
                    try:
                        import subprocess
                        process = subprocess.Popen(['xclip', '-selection', 'clipboard'], 
                                                 stdin=subprocess.PIPE, 
                                                 text=True)
                        process.communicate(input=text)
                        logger.info("Text copied to clipboard using xclip")
                    except Exception as e:
                        logger.error(f"Direct xclip call failed: {e}")
                        # Fallback to pyperclip
                        try:
                            pyperclip.copy(text)
                            logger.info("Text copied to clipboard using pyperclip")
                        except Exception as e:
                            logger.error(f"pyperclip.copy() failed: {e}")

                    # Verify copy
                    try:
                        verification = pyperclip.paste()
                        if verification == text:
                            logger.info("Clipboard verification successful")
                        else:
                            logger.error(f"Clipboard verification failed. Expected '{text}', got '{verification}'")
                    except Exception as e:
                        logger.error(f"Could not verify clipboard content: {e}")

                except Exception as e:
                    logger.error(f"Failed to copy to clipboard: {e}")

                self.tray.showMessage(
                    'TransClip',
                    'Transcription copied to clipboard',
                    QSystemTrayIcon.Information,
                    2000
                )
            else:
                logger.warning("Empty transcription result")
                self.tray.showMessage(
                    'TransClip',
                    'Transcription failed',
                    QSystemTrayIcon.Warning,
                    2000
                )
        except Exception as overall_e:
            logger.error(f"Exception in on_transcription_complete: {overall_e}", exc_info=True)

    def test_slot(self):
        """Simple slot for testing."""
        logger.info("Test slot called")

    def quit(self):
        """Quit the application."""
        if self.recording:
            self.stop_recording()
        self.app.quit()

    def run(self):
        """Run the application."""
        return self.app.exec_()

    def __del__(self):
        """Destructor for TransClip."""
        logger.info("TransClip object being destroyed")

def main():
    """Main entry point for the application."""
    try:
        app = TransClip(WhisperModelType.LARGE_V3)
        return app.run()
    except Exception as e:
        logger.error(f"Application error: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
