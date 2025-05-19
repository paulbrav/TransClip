"""Main application module for TransClip.

This module contains the core functionality for the TransClip application,
including audio recording, transcription, and clipboard integration via system tray.
The application uses the Whisper model for speech-to-text conversion.
"""

import logging
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from enum import StrEnum
from typing import Any, Dict, List, Optional, Union, cast

import numpy as np
import pyperclip
import sounddevice as sd
from faster_whisper import WhisperModel
from pynput import keyboard
from PyQt5.QtCore import QObject, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QActionGroup,
    QApplication,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSystemTrayIcon,
    QVBoxLayout,
)
from scipy import signal as scipy_signal

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [Line %(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("transclip.log")
    ]
)
logger = logging.getLogger(__name__)

# Global emergency signal handler to prevent crashes
def emergency_signal_handler(signum, frame):
    """Handle signals that might terminate the application unexpectedly."""
    sig_name = signal.Signals(signum).name
    logger.critical(f"Received signal {sig_name} ({signum}), stack trace follows:")
    logger.critical(''.join(traceback.format_stack(frame)))
    logger.critical("NOT exiting, continuing execution")
    # Do not call sys.exit() here - we want to prevent termination

# Install emergency signal handlers
signal.signal(signal.SIGINT, emergency_signal_handler)
signal.signal(signal.SIGTERM, emergency_signal_handler)

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
    PARAKEET_TDT_0_6B_V2 = "nvidia/parakeet-tdt-0.6b-v2"  # 0.6B parameters (NVIDIA Parakeet)

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
            cls.LARGE_V3: "Large-v3 (1.5B parameters, latest)",
            cls.PARAKEET_TDT_0_6B_V2: "Parakeet TDT 0.6B v2 (NVIDIA)"
        }
        return descriptions[model_type]

DEFAULT_RECORDING_KEY = keyboard.Key.home
DEFAULT_MODEL_TYPE = WhisperModelType.BASE

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

    def __init__(self, model_type: WhisperModelType = DEFAULT_MODEL_TYPE) -> None:
        """Initialize the application.

        Args:
            model_type: The Whisper model type to use.
        """
        super().__init__()

        # Initialize instance variables with proper type annotations
        self.recording_key: Union[keyboard.Key, keyboard.KeyCode] = DEFAULT_RECORDING_KEY
        self.listener: Optional[keyboard.Listener] = None
        self._listener_changing: bool = False
        self._listener_restart_complete: bool = True
        self.tray: Optional[QSystemTrayIcon] = None
        self.app: Optional[QApplication] = None

        try:
            # Set up signal handlers to log signals received
            signal.signal(signal.SIGTERM, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)

            # Log process and thread info
            self._log_process_info()

            self.app = QApplication(sys.argv)
            self.recording = False
            self.key_pressed = False
            self.key_cooldown = 0.5  # seconds
            self.last_recording_time: float = 0.0  # Timestamp for the last recording cycle
            self.audio_data: List[np.ndarray] = []  # Add type annotation
            self.processing = False  # Flag to track if we're processing audio
            self.stream: Optional[sd.InputStream] = None
            self.transcription_worker: Optional[TranscriptionWorker] = None
            self.current_model_type = model_type  # Store the current model type

            # Store recent transcriptions
            self.recent_transcriptions: deque[str] = deque(maxlen=5)

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

            # Set up an exit handler for the Qt application
            self.app.aboutToQuit.connect(self._on_app_quit)
        except Exception as e:
            logger.error(f"Error initializing TransClip: {e}", exc_info=True)
            sys.exit(1)

    def _signal_handler(self, signum, frame):
        """Handle signals to log what's happening.

        Args:
            signum: The signal number
            frame: The current stack frame
        """
        signal_names = {
            signal.SIGTERM: "SIGTERM",
            signal.SIGINT: "SIGINT"
        }
        signal_name = signal_names.get(signum, f"Signal {signum}")
        logger.warning(f"Received {signal_name}. Process will terminate.")
        self._log_process_info()

    def _log_process_info(self):
        """Log detailed process and thread information for debugging."""
        try:
            # Log basic process info
            pid = os.getpid()
            logger.info(f"Process ID: {pid}")

            # Log thread info
            main_thread = threading.main_thread()
            current_thread = threading.current_thread()
            all_threads = threading.enumerate()

            logger.info(f"Main thread: {main_thread.name} (ID: {main_thread.ident})")
            logger.info(f"Current thread: {current_thread.name} (ID: {current_thread.ident})")
            logger.info("All threads:")
            for thread in all_threads:
                logger.info(f"  - {thread.name} (ID: {thread.ident}, Daemon: {thread.daemon})")
        except Exception as e:
            logger.error(f"Error logging process info: {e}")

    def _on_app_quit(self) -> None:
        """Clean up resources when the application is about to quit."""
        logger.info("Application is exiting, cleaning up resources")

        # Clean up resources before exiting
        try:
            self.stop_listening()
        except Exception as e:
            logger.error(f"Error stopping listener during exit: {e}")

        try:
            if self.stream:
                self.stream.close()
        except Exception as e:
            logger.error(f"Error closing stream during exit: {e}")

        # Ensure the application exits cleanly
        if self.app is not None:
            # Use cast to tell the type checker this is definitely a QApplication
            app = cast(QApplication, self.app)
            app.quit()

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
        """Initialize the system tray icon and menu with transcriptions and preferences.

        Creates a system tray icon with a context menu containing transcriptions history,
        preferences options, and a quit option.
        """
        self.tray = QSystemTrayIcon()
        self.tray.setIcon(QIcon.fromTheme('audio-input-microphone'))

        # Create main menu
        menu = QMenu()
        logger.debug("Created main menu")

        # Create transcriptions submenu
        self.transcriptions_menu = QMenu('Transcriptions')
        menu.addMenu(self.transcriptions_menu)
        logger.debug("Added transcriptions menu")

        # Add a direct action for key binding
        key_binding_action = menu.addAction("⌨ Configure Key Binding...")
        assert key_binding_action is not None
        key_binding_action.triggered.connect(self.show_key_binding_dialog)
        logger.debug("Added key binding action to main menu")

        # Add direct model selection actions to main menu
        menu.addSeparator()
        model_header_action = menu.addAction("🔊 Select Transcription Model:")
        assert model_header_action is not None
        model_header_action.setEnabled(False)

        # Add model options
        model_group = QActionGroup(self)
        model_group.setExclusive(True)

        # Add each model type as a radio button option directly to main menu
        for model_type in WhisperModelType:
            model_action = menu.addAction(WhisperModelType.get_description(model_type))
            assert model_action is not None
            model_action.setCheckable(True)

            # Check the current model
            if model_type == self.current_model_type:
                model_action.setChecked(True)

            # Add to action group and connect
            model_group.addAction(model_action)

            # Create a separate function for each model type to avoid lambda issues
            def create_model_handler(m_type):
                return lambda checked: self.change_model(m_type)

            model_action.triggered.connect(create_model_handler(model_type))

        logger.debug("Added model options directly to main menu")

        # Add separator before restart and quit
        menu.addSeparator()
        logger.debug("Added separator")

        # Add restart action
        restart_action = menu.addAction('🔄 Restart')
        assert restart_action is not None
        restart_action.triggered.connect(self.restart)
        logger.debug("Added restart action")

        # Add quit action
        quit_action = menu.addAction('Quit')
        assert quit_action is not None
        quit_action.triggered.connect(self.quit)
        logger.debug("Added quit action")

        # Initialize transcriptions list
        self.update_transcriptions_menu()
        logger.debug("Updated transcriptions menu")

        self.tray.setContextMenu(menu)
        logger.debug("Set context menu")
        self.tray.show()
        logger.debug("Showed tray icon")

    def init_keyboard_listener(self) -> None:
        """Initialize the keyboard listener for the recording hotkey.

        Sets up a keyboard listener to detect when the Home key is pressed
        to start and stop recording.
        """
        try:
            logger.info("Initializing keyboard listener")
            self.listener = keyboard.Listener(
                on_press=self.on_press,
                on_release=self.on_release,
                daemon=True  # Set as daemon thread so it doesn't keep app alive
            )
            self.listener.start()
            key_name = getattr(self.recording_key, 'name', str(self.recording_key))
            logger.info(f"Keyboard listener started for key: {key_name}")
        except Exception as e:
            logger.error(f"Error initializing keyboard listener: {e}", exc_info=True)

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
                key == self.recording_key and  # Use the configurable recording key
                (current_time - self.last_recording_time) > self.key_cooldown):

                logger.info("=== Key Press Event ===")
                logger.info(f"Time since last recording: {current_time - self.last_recording_time:.2f}s")
                self.key_pressed = True  # Mark key as pressed
                self.start_recording()
        except AttributeError as e:
            # This might happen with special keys
            logger.debug(f"AttributeError in on_press: {e}")
        except Exception as e:
            # Log other errors but don't crash
            logger.error(f"Error in on_press: {e}", exc_info=True)

    def on_release(self, key: Any) -> None:
        """Handle key release events.

        Args:
            key: The key that was released.
        """
        try:
            if self.recording and key == self.recording_key and self.key_pressed:
                logger.info("=== Key Release Event ===")
                self.key_pressed = False  # Reset key press state
                self.last_recording_time = time.time()  # Update timestamp after complete cycle
                self.stop_recording()
        except AttributeError as e:
            # This might happen with special keys
            logger.debug(f"AttributeError in on_release: {e}")
        except Exception as e:
            # Log other errors but don't crash
            logger.error(f"Error in on_release: {e}", exc_info=True)

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
            # Change icon to indicate processing/transcribing state
            self.tray.setIcon(QIcon.fromTheme("view-refresh"))
            # Alternatively use: self.tray.setIcon(QIcon.fromTheme("system-run"))

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
                    audio = scipy_signal.resample(audio, int(len(audio) * 16000 / device_sample_rate))
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

            # Change icon back to default
            if self.tray:
                self.tray.setIcon(QIcon.fromTheme("audio-input-microphone"))

            if text:
                # Store the transcription in recent list
                self.store_transcription(text)

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
        logger.info("Quitting application")
        try:
            # Stop recording if active
            if self.recording:
                logger.info("Stopping recording before quitting")
                self.stop_recording()

            # Clean up keyboard listener
            if self.listener:
                logger.info("Stopping keyboard listener")
                try:
                    self.listener.stop()
                except Exception as e:
                    logger.error(f"Error stopping listener during quit: {e}")

            logger.info("Quitting application")
            # Use cast to tell the type checker this is definitely a QApplication
            app = cast(QApplication, self.app)
            app.quit()
        except Exception as e:
            logger.error(f"Error during application quit: {e}", exc_info=True)
            # Force quit
            # Use cast to tell the type checker this is definitely a QApplication
            app = cast(QApplication, self.app)
            app.quit()

    def restart(self) -> None:
        """Restart the application.

        Stops recording if active, quits the current instance,
        and starts a new instance of the application.
        """
        if self.recording:
            self.stop_recording()

        logger.info("Restarting application...")

        # Get the path to the current executable
        executable = sys.executable
        args = sys.argv.copy()

        # Prepare to restart by using execv
        logger.info(f"Restarting with: {executable} {' '.join(args)}")

        # Show notification before restart
        if self.tray:
            self.tray.showMessage(
                'TransClip',
                'Restarting application...',
                QSystemTrayIcon.Information,
                1000
            )

        # Small delay to allow notification to show
        time.sleep(0.5)

        # Quit the current instance
        # Use cast to tell the type checker this is definitely a QApplication
        app = cast(QApplication, self.app)
        app.quit()

        # Start a new process
        try:
            os.execv(executable, [executable] + args)
        except Exception as e:
            logger.error(f"Failed to restart: {e}")
            # If restart fails, try to start a new process instead
            try:
                subprocess.Popen([executable] + args)
                sys.exit(0)
            except Exception as e2:
                logger.error(f"Failed to start new process: {e2}")

    def run(self) -> int:
        """Run the application.
        Returns:
            The exit code from the Qt event loop.
        """
        logger.info("Starting application run")

        # Initialize listener change flag
        self._listener_changing = False
        self._listener_restart_complete = True

        # Create a QTimer to periodically check the state of the application
        # This helps keep the main event loop alive and detects problems
        def check_app_health():
            logger.debug("Health check: Application is running")
            self._check_thread_health()

        # Setup periodic health check timer
        health_timer = QTimer()
        health_timer.timeout.connect(check_app_health)
        health_timer.start(5000)  # Check every 5 seconds

        # Install exit handlers
        original_excepthook = sys.excepthook
        def exception_handler(exc_type, exc_value, exc_traceback):
            logger.critical("Unhandled exception:", exc_info=(exc_type, exc_value, exc_traceback))
            return original_excepthook(exc_type, exc_value, exc_traceback)
        sys.excepthook = exception_handler

        logger.info("Starting Qt event loop")
        if self.app is not None:
            # Use cast to tell the type checker this is definitely a QApplication
            app = cast(QApplication, self.app)
            result = app.exec_()
            logger.info(f"Qt event loop completed with result: {result}")
            return result
        else:
            logger.error("Application instance is None, cannot run event loop")
            return 1

    def _check_thread_health(self):
        """Check the health of important threads."""
        try:
            thread_count = threading.active_count()
            if thread_count < 2 and not self._listener_changing:  # At least main thread + keyboard listener
                logger.warning(f"Thread count is unexpectedly low: {thread_count}")

            # Check keyboard listener specifically
            if self.listener is None and not self._listener_changing:
                logger.warning("Keyboard listener is None outside of a key change operation")
                # Try to restart it
                self.init_keyboard_listener()
            elif not self._listener_changing and hasattr(self.listener, 'is_alive') and not self.listener.is_alive():
                logger.warning("Keyboard listener thread is not alive")
                # Try to restart it
                self.init_keyboard_listener()

        except Exception as e:
            logger.error(f"Error in health check: {e}", exc_info=True)

    def __del__(self) -> None:
        """Destructor for TransClip.

        Performs cleanup when the TransClip object is destroyed.
        """
        logger.info("TransClip object being destroyed")
        try:
            # Clean up keyboard listener if it exists
            if hasattr(self, 'listener') and self.listener:
                logger.info("Stopping keyboard listener in destructor")
                try:
                    self.listener.stop()
                except Exception as e:
                    logger.error(f"Error stopping listener in destructor: {e}")
        except Exception as e:
            logger.error(f"Error in TransClip destructor: {e}")

    def store_transcription(self, text: str) -> None:
        """Store a transcription in the recent transcriptions list.

        Args:
            text: The transcribed text to store.
        """
        # Add new transcription to beginning of list
        self.recent_transcriptions.appendleft(text)

        # Update the transcriptions menu
        self.update_transcriptions_menu()

    def update_transcriptions_menu(self) -> None:
        """Update the transcriptions menu with recent transcriptions."""
        # Clear existing menu items
        self.transcriptions_menu.clear()

        if not self.recent_transcriptions:
            # Add a placeholder if no transcriptions
            empty_action = self.transcriptions_menu.addAction("No recent transcriptions")
            assert empty_action is not None
            empty_action.setEnabled(False)
            return

        # Add each recent transcription as a menu item
        for i, text in enumerate(self.recent_transcriptions):
            # Create a shortened version for the menu (first 30 chars)
            display_text = text[:30] + ("..." if len(text) > 30 else "")

            # Create the action
            action = self.transcriptions_menu.addAction(f"{i+1}. {display_text}")
            assert action is not None

            # Connect to a lambda that copies this specific text
            action.triggered.connect(lambda checked, t=text: self.copy_to_clipboard(t))

    def copy_to_clipboard(self, text: str) -> None:
        """Copy the given text to clipboard.

        Args:
            text: The text to copy to clipboard.
        """
        try:
            # Try to use xclip directly for Linux
            try:
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

            # Show notification
            if self.tray:
                self.tray.showMessage(
                    'TransClip',
                    'Transcription copied to clipboard',
                    QSystemTrayIcon.Information,
                    2000
                )
        except Exception as e:
            logger.error(f"Failed to copy to clipboard: {e}")

    def change_model(self, model_type: WhisperModelType) -> None:
        """Change the Whisper model being used.

        Args:
            model_type: The WhisperModelType to change to.
        """
        if self.recording or self.processing:
            # Show warning if recording or processing
            if self.tray:
                self.tray.showMessage(
                    'TransClip',
                    'Cannot change model while recording or processing',
                    QSystemTrayIcon.Warning,
                    2000
                )
            return

        try:
            # Reinitialize the model
            logger.info(f"Changing model to {model_type}")
            self.model = WhisperModel(
                model_type,
                device="cuda" if self.has_cuda() else "cpu",
                compute_type="float16" if self.has_cuda() else "float32"
            )
            self.current_model_type = model_type  # Update the current model type
            logger.info(f"Using model: {WhisperModelType.get_description(model_type)}")

            # Show success message
            if self.tray:
                self.tray.showMessage(
                    'TransClip',
                    f'Changed model to {WhisperModelType.get_description(model_type)}',
                    QSystemTrayIcon.Information,
                    2000
                )
        except Exception as e:
            logger.error(f"Failed to change model: {e}")
            # Show error message
            if self.tray:
                self.tray.showMessage(
                    'TransClip',
                    f'Failed to change model: {str(e)}',
                    QSystemTrayIcon.Critical,
                    2000
                )

    def show_key_binding_dialog(self) -> None:
        """Show dialog to configure key binding.

        Opens a dialog that allows the user to set a new key for starting/stopping recording.
        """
        dialog = QDialog()
        dialog.setWindowTitle("Configure Key Binding")
        dialog.setMinimumWidth(300)  # Set minimum width for better usability

        layout = QVBoxLayout()

        # Instructions label
        label = QLabel("Press a key to set as the recording shortcut:")
        layout.addWidget(label)

        # Current key label
        current_key_name = getattr(self.recording_key, 'name', str(self.recording_key))
        current_key_label = QLabel(f"Current key: {current_key_name}")
        layout.addWidget(current_key_label)

        # Key capture button
        key_button = QPushButton("Press to record new key...")
        layout.addWidget(key_button)

        # Status label
        status_label = QLabel("")
        status_label.setStyleSheet("color: gray;")
        layout.addWidget(status_label)

        # Variable to store the new key
        new_key = [None]  # Use a list to allow modification in the inner function
        temp_listener = [None]  # Store the temporary listener

        # Function to handle key press in the dialog
        def on_key_capture():
            key_button.setText("Press any key...")
            key_button.setEnabled(False)
            status_label.setText("Waiting for key press...")
            status_label.setStyleSheet("color: blue;")

            # Create a temporary keyboard listener
            def on_dialog_key_press(key):
                new_key[0] = key
                key_name = getattr(key, 'name', str(key))
                key_button.setText(f"Key captured: {key_name}")
                key_button.setEnabled(True)
                status_label.setText(f"Captured key: {key_name}")
                status_label.setStyleSheet("color: green;")

                # Stop the listener properly
                if temp_listener[0] is not None:
                    temp_listener[0].stop()
                return False  # Stop listener

            # Start temporary listener
            try:
                # Stop existing listener if any
                if temp_listener[0] is not None:
                    temp_listener[0].stop()

                # Create new listener
                temp_listener[0] = keyboard.Listener(
                    on_press=on_dialog_key_press,
                    daemon=True  # Set as daemon thread
                )
                temp_listener[0].start()
                logger.debug("Started temporary key capture listener")
            except Exception as e:
                logger.error(f"Error creating temporary key listener: {e}")
                key_button.setText("Error capturing key")
                key_button.setEnabled(True)
                status_label.setText(f"Error: {str(e)}")
                status_label.setStyleSheet("color: red;")

        # Connect button to key capture function
        key_button.clicked.connect(on_key_capture)

        # Button row
        button_layout = QHBoxLayout()
        save_button = QPushButton("Save")
        cancel_button = QPushButton("Cancel")
        button_layout.addWidget(save_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        # Set the layout
        dialog.setLayout(layout)

        # Connect buttons
        def on_save():
            # Make sure we stop the temporary listener
            if temp_listener[0] is not None:
                try:
                    temp_listener[0].stop()
                except Exception as e:
                    logger.error(f"Error stopping temporary listener: {e}")

            if new_key[0] is not None:
                # Show saving indicator
                status_label.setText("Saving key binding...")
                status_label.setStyleSheet("color: blue;")
                save_button.setEnabled(False)
                cancel_button.setEnabled(False)

                # Use QTimer to delay dialog closure to allow Qt event processing
                def complete_save():
                    self.change_recording_key(new_key[0])
                    dialog.accept()

                # Use a short delay to allow UI to update
                QTimer.singleShot(200, complete_save)
            else:
                dialog.accept()

        def on_cancel():
            # Make sure we stop the temporary listener
            if temp_listener[0] is not None:
                try:
                    temp_listener[0].stop()
                except Exception as e:
                    logger.error(f"Error stopping temporary listener: {e}")
            dialog.reject()

        save_button.clicked.connect(on_save)
        cancel_button.clicked.connect(on_cancel)

        # Handle dialog close events to ensure listener is stopped
        dialog.finished.connect(lambda: temp_listener[0] and temp_listener[0].stop())

        # Show dialog
        dialog.exec_()

    def _finish_listener_restart(self, old_listener, key_name):
        """Finish the listener restart process.

        Args:
            old_listener: The old keyboard listener to stop.
            key_name: The name of the new key for notification purposes.
        """
        try:
            # Create the new listener
            logger.info("Creating new keyboard listener")
            self.init_keyboard_listener()
            logger.info(f"New listener created and started: {self.listener}")

            # Now we can stop the old listener more safely
            if old_listener:
                logger.info("Stopping the old keyboard listener")
                try:
                    # Try to stop the listener gracefully
                    old_listener.stop()
                    logger.info("Old listener stopped successfully")
                    # Give it a moment to clean up
                    time.sleep(0.1)
                except Exception as stop_e:
                    logger.error(f"Error stopping keyboard listener: {stop_e}", exc_info=True)

            # Log final state
            logger.info("=== KEYBOARD LISTENER RESTART COMPLETE ===")
            logger.info(f"Thread counts after restart: {threading.active_count()}")
            self._log_process_info()

            # Clear the protective flag
            self._listener_changing = False
            self._listener_restart_complete = True

            # Show notification only after everything is complete
            if self.tray:
                self.tray.showMessage(
                    'TransClip',
                    f'Recording key changed to: {key_name}',
                    QSystemTrayIcon.Information,
                    2000
                )
        except Exception as e:
            logger.error(f"Error in _finish_listener_restart: {e}", exc_info=True)
            self._listener_changing = False

            if self.tray:
                self.tray.showMessage(
                    'TransClip',
                    f'Error finalizing keyboard listener restart: {str(e)}',
                    QSystemTrayIcon.Critical,
                    2000
                )

    def change_recording_key(self, key: Union[keyboard.Key, keyboard.KeyCode]) -> None:
        """Change the key used for recording.

        Args:
            key: The new key to use for recording.
        """
        try:
            # Log the state before changing keys
            logger.info("=== STARTING KEY CHANGE OPERATION ===")
            logger.info(f"Thread counts before key change: {threading.active_count()}")
            self._log_process_info()

            # Store the new key
            old_key_name = getattr(self.recording_key, 'name', str(self.recording_key))
            self.recording_key = key
            key_name = getattr(key, 'name', str(key))
            logger.info(f"Changed recording key from {old_key_name} to {key_name}")

            # Create a flag to track listener restart status
            self._listener_restart_complete = False

            # Define a function to restart the listener
            def restart_listener():
                try:
                    logger.info("=== RESTARTING KEYBOARD LISTENER ===")
                    logger.info(f"Thread counts during restart: {threading.active_count()}")

                    # Store reference to the old listener
                    old_listener = self.listener
                    old_listener_alive = old_listener.is_alive() if old_listener else False
                    logger.info(f"Old listener alive: {old_listener_alive}")

                    # Set a temporary protective flag
                    self._listener_changing = True

                    # Create a new listener with the updated key FIRST
                    # This ensures we have a new listener ready before stopping the old one
                    logger.info("Creating new keyboard listener with updated key binding")

                    # Set listener to None first to help with garbage collection
                    self.listener = None

                    # Wait a moment to ensure the old listener is properly dereferenced
                    QTimer.singleShot(50, lambda: self._finish_listener_restart(old_listener, key_name))

                except Exception as e:
                    logger.error(f"Error in restart_listener: {e}", exc_info=True)
                    logger.info("=== ERROR IN KEYBOARD LISTENER RESTART ===")
                    self._log_process_info()
                    self._listener_changing = False

                    if self.tray:
                        self.tray.showMessage(
                            'TransClip',
                            f'Failed to restart keyboard listener: {str(e)}',
                            QSystemTrayIcon.Critical,
                            2000
                        )

            # Set up protective flag before scheduling any async operations
            self._listener_changing = True

            # Use a single-shot timer to handle the listener restart
            # This ensures we return to Qt's event loop before restarting the listener
            logger.info("Scheduling listener restart with QTimer")
            QTimer.singleShot(100, restart_listener)
            logger.info("QTimer setup complete. Control returning to event loop.")

            # Add a timeout to ensure we don't leave the app in an inconsistent state
            def check_restart_completion():
                if not self._listener_restart_complete:
                    logger.warning("Listener restart timed out - forcing cleanup")
                    self._listener_changing = False
                    if self.listener is None:
                        logger.warning("Recreating listener after timeout")
                        self.init_keyboard_listener()

            QTimer.singleShot(5000, check_restart_completion)

        except Exception as e:
            logger.error(f"Error changing recording key: {e}", exc_info=True)
            logger.info("=== ERROR IN KEY CHANGE OPERATION ===")
            self._log_process_info()
            self._listener_changing = False

            if self.tray:
                self.tray.showMessage(
                    'TransClip',
                    f'Failed to change recording key: {str(e)}',
                    QSystemTrayIcon.Critical,
                    2000
                )

def main() -> int:
    """Start the application.

    Returns:
        int: Exit code, 0 for success, 1 for error.
    """
    try:
        # Set up essential exception handling
        def log_unhandled_exception(exc_type, exc_value, exc_traceback):
            logger.critical("Unhandled exception in main thread:",
                           exc_info=(exc_type, exc_value, exc_traceback))
            # Still call the original handler
            sys.__excepthook__(exc_type, exc_value, exc_traceback)

        # Install the global exception hook
        sys.excepthook = log_unhandled_exception

        # Set application name for better integration
        QApplication.setApplicationName("TransClip")
        QApplication.setQuitOnLastWindowClosed(False)

        # Set up the Qt application with additional safeguards
        app = TransClip(DEFAULT_MODEL_TYPE)

        # Safety check to ensure we still have a working app
        if not app.app or not app.app.instance():
            logger.error("QApplication instance is not valid, cannot start TransClip")
            return 1

        # Run the application event loop with additional protection
        logger.info("Starting TransClip application event loop")
        return app.run()

    except Exception as e:
        logger.critical(f"Critical application error: {e}", exc_info=True)

        # Try to clean up any lingering QApplication
        try:
            # Get the current QApplication instance if any
            qapp = QApplication.instance()
            if qapp:
                logger.info("Quitting QApplication instance")
                qapp.quit()
        except Exception as cleanup_error:
            logger.error(f"Error during cleanup: {cleanup_error}")

        return 1

if __name__ == '__main__':
    sys.exit(main())
