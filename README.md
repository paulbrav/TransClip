# TransClip

A desktop application that transcribes speech to text when you hold down a key. It records audio from your microphone, transcribes it using Faster Whisper (an optimized version of OpenAI's Whisper), and automatically copies the transcription to your clipboard.

## Features

- Single key binding for recording (press and hold to record, release to transcribe)
- System tray integration with GNOME's status area
- Fast transcription using Faster Whisper
- Automatic clipboard copying of transcribed text
- Configurable hotkey combinations
- Visual feedback during recording (icon changes)
- Error handling and status notifications
- GPU acceleration support (CUDA) for faster transcription
- Logging system for debugging and monitoring

## Requirements

- Python 3.8+
- PortAudio (for audio recording)
- CUDA-capable GPU (optional, for faster transcription)
- Linux with systemd (for service installation)
- X11 display server

## Installation

1. Install system dependencies:

```bash
# For Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y portaudio19-dev python3-dev python3-venv

# For Fedora
sudo dnf install -y portaudio-devel python3-devel python3-virtualenv
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
- Create a virtual environment
- Install all Python dependencies
- Set up a systemd user service
- Configure autostart on login

## Usage

### Starting the Application

The application starts automatically on system login. You can also:

- Start manually: `systemctl --user start transclip`
- Stop: `systemctl --user stop transclip`
- Check status: `systemctl --user status transclip`

### Basic Operation

1. Look for the microphone icon in your system tray
2. Press and hold the default hotkey (Ctrl+Left) to start recording
3. Speak clearly into your microphone
4. Release the key to stop recording and start transcription
5. The transcribed text will automatically be copied to your clipboard
6. Watch for notification popups for status updates

### Logs

Log files are located at:
- `~/.local/share/transclip/transclip.log` - General application logs
- `~/.local/share/transclip/transclip.error.log` - Error logs

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

### Project Structure

- `transclip/` - Main package directory
  - `__init__.py` - Package initialization and version
  - `app.py` - Core application logic
  - `hotkey.py` - Hotkey management system
  - `__main__.py` - Entry point

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
   - Check if the service is running: `systemctl --user status transclip`
   - Verify X11 DISPLAY environment variable: `echo $DISPLAY`

2. **Recording not working**
   - List audio devices:
     ```bash
     source ~/.local/share/transclip/venv/bin/activate
     python3 -c "import sounddevice as sd; print(sd.query_devices())"
     ```
   - Check microphone permissions: `pactl list sources short`
   - Ensure PulseAudio is running: `pulseaudio --check`

3. **Slow transcription**
   - Check CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`
   - Consider using a smaller Whisper model

### Getting Help

- Check the log files for detailed error messages
- Open an issue on GitHub with the relevant log output
- Include your system information when reporting issues 