#!/bin/bash

# Exit on any error
set -e

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "Please run this script as a regular user, not as root"
    exit 1
fi

echo "Installing TransClip..."

# Install system dependencies
if command -v apt-get &> /dev/null; then
    # Debian/Ubuntu
    echo "Installing system dependencies for Debian/Ubuntu..."
    sudo apt-get update
    sudo apt-get install -y portaudio19-dev python3-dev
elif command -v dnf &> /dev/null; then
    # Fedora
    echo "Installing system dependencies for Fedora..."
    sudo dnf install -y portaudio-devel python3-devel
else
    echo "Warning: Unsupported package manager. Please install PortAudio development files manually."
fi

# Install uv if not already installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Source the shell configuration to get uv in PATH
    source ~/.bashrc
fi

# Create virtual environment and install dependencies
echo "Creating virtual environment..."
VENV_PATH="${HOME}/.local/share/transclip/venv"
mkdir -p "${HOME}/.local/share/transclip"

# Create venv using uv
uv venv "${VENV_PATH}"

echo "Installing Python dependencies..."
source "${VENV_PATH}/bin/activate"

# Install in development mode from the current directory
cd "$(dirname "$0")"
uv pip install -e .
cd -

# Update systemd service to use the virtual environment's Python
SYSTEMD_DIR="${HOME}/.config/systemd/user"
mkdir -p "${SYSTEMD_DIR}"
mkdir -p "${HOME}/.config/transclip"

# Create systemd service with correct Python path
cat > "${SYSTEMD_DIR}/transclip.service" << EOF
[Unit]
Description=TransClip - Speech to Text Transcription
After=graphical-session.target
PartOf=graphical-session.target

[Service]
Type=simple
ExecStart=${VENV_PATH}/bin/python -m transclip
Restart=on-failure
RestartSec=5
Environment=DISPLAY=:0
Environment=XAUTHORITY=%h/.Xauthority
Environment=PATH=${PATH}:/usr/local/bin
Environment=PYTHONUNBUFFERED=1
StandardOutput=append:${HOME}/.local/share/transclip/transclip.log
StandardError=append:${HOME}/.local/share/transclip/transclip.error.log
WorkingDirectory=${HOME}/.local/share/transclip

[Install]
WantedBy=graphical-session.target
EOF

# Create log files
touch "${HOME}/.local/share/transclip/transclip.log"
touch "${HOME}/.local/share/transclip/transclip.error.log"

# Stop any existing service
systemctl --user stop transclip.service || true

# Install systemd service
echo "Installing systemd service..."
systemctl --user daemon-reload
systemctl --user enable transclip.service
systemctl --user start transclip.service

echo -e "\nTransClip has been installed successfully!"
echo "The application will start automatically on system startup."
echo "You can control the service with:"
echo "  Start:   systemctl --user start transclip"
echo "  Stop:    systemctl --user stop transclip"
echo "  Restart: systemctl --user restart transclip"
echo "  Status:  systemctl --user status transclip"
echo -e "\nLog files are located at:"
echo "  ${HOME}/.local/share/transclip/transclip.log"
echo "  ${HOME}/.local/share/transclip/transclip.error.log" 