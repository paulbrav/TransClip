# TransClip

A desktop application that transcribes speech to text when you hold down a key. It records audio from your microphone, transcribes it using Faster Whisper (an optimized version of OpenAI's Whisper), and automatically copies the transcription to your clipboard.

## Features

- Single key binding for recording (press and hold to record, release to transcribe)
- System tray integration
- Fast transcription using Faster Whisper
- Automatic clipboard copying of transcribed text
- Optional automatic pasting into the active window
- Optional text cleanup with punctuation and language model support
  that can be toggled from the tray menu
- Customizable cleanup prompt for the language model
- Configurable key binding
- Visual feedback during recording (icon changes)
- Error handling and status notifications
- GPU acceleration support (CUDA) for faster transcription
- Logging system for debugging and monitoring
- Selectable Whisper model sizes for different accuracy/speed tradeoffs
- Cross-platform support for Linux and macOS

## Requirements

- Python 3.11+
- PortAudio (for audio recording)
- CUDA-capable GPU (optional, for faster transcription)
- Linux with systemd or macOS with Homebrew
- X11 display server (Linux) or macOS 10.13+

## Installation

### Linux

1. Install system dependencies:

```bash
# For Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y portaudio19-dev python3-dev

# For Fedora
sudo dnf install -y portaudio-devel python3-devel
```

2. Clone the repository:

```bash
git clone https://github.com/yourusername/TransClip.git
cd TransClip
```

3. Run the installation script:

```bash
./install.sh
```

### macOS

1. Install Homebrew if you haven't already:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

2. Clone the repository:

```bash
git clone https://github.com/yourusername/TransClip.git
cd TransClip
```

3. Run the installation script:

```bash
./install.sh
```

The installation script will:
- Install the uv package manager if not already installed
- Create a virtual environment
- Install all Python dependencies
- Download the Whisper base model for transcription
- Offer to download additional Whisper models (you can select multiple)
- Set up a systemd service (Linux) or launchd agent (macOS)
- Configure autostart on login

## Usage

### Starting the Application

The application starts automatically on system login. You can also:

#### On Linux:
- Start manually: `systemctl --user start transclip`
- Stop: `systemctl --user stop transclip`
- Restart: `systemctl --user restart transclip`
- Check status: `systemctl --user status transclip`

#### On macOS:
- Start manually: `launchctl load -w ~/Library/LaunchAgents/com.user.transclip.plist`
- Stop: `launchctl unload -w ~/Library/LaunchAgents/com.user.transclip.plist`
- Restart: `launchctl unload ~/Library/LaunchAgents/com.user.transclip.plist && launchctl load -w ~/Library/LaunchAgents/com.user.transclip.plist`

### Basic Operation

1. Look for the microphone icon in your system tray
2. Press and hold the default hotkey (Home key) to start recording
3. Speak clearly into your microphone
4. Release the key to stop recording and start transcription
5. The transcribed text will automatically be copied to your clipboard
6. If auto paste is enabled, the text will also be inserted into the active window
7. Enable or disable transcript cleanup using the tray menu option
8. Watch for notification popups for status updates

### Changing Models

You can change the Whisper model by right-clicking the system tray icon and selecting the model size you want to use from the menu. Larger models provide better accuracy but are slower and require more resources.

### Configuring Key Binding

You can change the recording key by right-clicking the system tray icon and selecting "⌨ Configure Key Binding...".

The chosen key is stored in `~/.config/transclip/config.json` using the
XDG Base Directory convention so it persists across sessions. You can edit this
file manually if you want to set the key outside of the application.

### Customizing Cleanup Prompt

You can modify the instruction used for the optional LLM cleanup stage by
editing the `cleanup_prompt` value in `~/.config/transclip/config.json`.
Use `{text}` in the string to insert the transcript.

### Accessing Recent Transcriptions

The system tray menu also provides access to recent transcriptions, which you can select to copy again to the clipboard.

### Logs

Log files are located at:
- `~/.local/share/transclip/transclip.log` - General application logs
- `~/.local/share/transclip/transclip.error.log` - Error logs

### Whisper Models

During installation, the base Whisper model is downloaded automatically and you have the option to grab extra models. You can also download models manually later:

```bash
# Activate the virtual environment
source ~/.local/share/transclip/venv/bin/activate

# Download a specific model (tiny, base, small, medium, large, large-v2, large-v3, parakeet-tdt-0.6b-v2)
python -m transclip.download_models --model small
```

Available models:
- `tiny`: ~75MB (fastest, least accurate)
- `base`: ~150MB (good balance for short phrases)
- `small`: ~500MB (good balance of speed and accuracy)
- `medium`: ~1.5GB (slower, more accurate)
- `large`: ~3GB (slowest, most accurate)
- `large-v2`, `large-v3`: ~3GB (improved large models)
- `parakeet-tdt-0.6b-v2`: ~1.2GB (NVIDIA Parakeet model)

## Development

### Setting up Development Environment

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

2. Install development dependencies:
```bash
uv pip install -e ".[dev]"
```

### Code Quality Tools

The project uses several tools to maintain code quality:

- `mypy` for static type checking
- `ruff` for linting and import sorting

Run the checks:
```bash
mypy --strict transclip/
ruff check transclip/
```

### Running Tests

The test suite uses Python's built-in `unittest` framework. To execute all tests,
run:

```bash
python -m unittest discover -s tests
```

### Project Structure

- `transclip/` - Main package directory
  - `__init__.py` - Package initialization and version
  - `app.py` - Core application logic
  - `download_models.py` - Utility for downloading Whisper models
  - `__main__.py` - Entry point
- `install.sh` - Installation script
- `tests/` - Test directory
- `pyproject.toml` - Project configuration and dependencies
- `transclip.service` - Systemd service file

## License

MIT License - See LICENSE file for details

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run code quality checks
5. Submit a pull request

## Troubleshooting

### Common Issues

1. **No microphone icon in system tray**
   - Check if the service is running:
     - Linux: `systemctl --user status transclip`
     - macOS: `launchctl list | grep transclip`
   - Verify environment variables:
     - Linux: `echo $DISPLAY`
     - macOS: Check if you have permission to access the microphone in System Preferences

2. **Recording not working**
   - List audio devices:
     ```bash
     source ~/.local/share/transclip/venv/bin/activate
     python3 -c "import sounddevice as sd; print(sd.query_devices())"
     ```
   - Check microphone permissions:
     - Linux: `pactl list sources short`
     - macOS: Check System Preferences → Security & Privacy → Microphone
   - Ensure audio service is running:
     - Linux: `pulseaudio --check`
     - macOS: Check System Preferences → Sound

3. **Slow transcription**
   - Check CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`
   - Consider using a smaller Whisper model

4. **Clipboard issues**
   - Linux: Ensure xclip is installed: `sudo apt install xclip`
   - macOS: No additional clipboard tools needed
   - Check logs for clipboard errors

### Getting Help

- Check the log files for detailed error messages
- Open an issue on GitHub with the relevant log output
- Include your system information when reporting issues 
