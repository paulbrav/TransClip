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
uv pip install -e .

# Download Whisper models
echo "Downloading Whisper models..."
# Download the base model by default
if python -m transclip.download_models --model base; then
    echo "Base model downloaded successfully."
else
    echo "Warning: Failed to download base model. The application may not work correctly."
    echo "You can try downloading models manually later with: python -m transclip.download_models --model base"
fi

# Ask if user wants to download additional models
echo ""
echo "The base model has been downloaded. Would you like to download additional models?"
echo "Larger models provide better accuracy but require more disk space and memory."
echo ""
echo "Available models:"
echo "1) tiny   - ~75MB  (fastest, least accurate)"
echo "2) base   - ~150MB (already downloaded)"
echo "3) small  - ~500MB (good balance of speed and accuracy)"
echo "4) medium - ~1.5GB (slower, more accurate)"
echo "5) large  - ~3GB   (slowest, most accurate)"
echo "6) No additional models"
echo ""
read -p "Enter your choice (1-6): " model_choice

case $model_choice in
    1)
        echo "Downloading tiny model..."
        if python -m transclip.download_models --model tiny; then
            echo "Tiny model downloaded successfully."
        else
            echo "Warning: Failed to download tiny model."
        fi
        ;;
    3)
        echo "Downloading small model..."
        if python -m transclip.download_models --model small; then
            echo "Small model downloaded successfully."
        else
            echo "Warning: Failed to download small model."
        fi
        ;;
    4)
        echo "Downloading medium model..."
        if python -m transclip.download_models --model medium; then
            echo "Medium model downloaded successfully."
        else
            echo "Warning: Failed to download medium model."
        fi
        ;;
    5)
        echo "Downloading large model..."
        if python -m transclip.download_models --model large; then
            echo "Large model downloaded successfully."
        else
            echo "Warning: Failed to download large model."
        fi
        ;;
    6)
        echo "No additional models will be downloaded."
        ;;
    *)
        echo "No additional models will be downloaded."
        ;;
esac

# Update systemd service to use the virtual environment's Python
SYSTEMD_DIR="${HOME}/.config/systemd/user"
mkdir -p "${SYSTEMD_DIR}"
mkdir -p "${HOME}/.config/transclip"

# Copy the systemd service file
cp transclip.service "${SYSTEMD_DIR}/transclip.service"

# Update the service file with the correct paths
sed -i "s|ExecStart=.*|ExecStart=${VENV_PATH}/bin/python -m transclip|g" "${SYSTEMD_DIR}/transclip.service"
sed -i "s|Environment=PATH=.*|Environment=PATH=${PATH}:/usr/local/bin|g" "${SYSTEMD_DIR}/transclip.service"

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