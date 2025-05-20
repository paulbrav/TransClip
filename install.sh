#!/bin/bash

# Exit on any error
set -e

# Parse command line arguments
TEST_SERVICE_CONFIG=false
for arg in "$@"; do
    case $arg in
        --test-service-config)
            TEST_SERVICE_CONFIG=true
            shift
            ;;
        *)
            # Unknown option
            ;;
    esac
done

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "Please run this script as a regular user, not as root"
    exit 1
fi

echo "Installing TransClip..."

# Detect OS
OS_TYPE=$(uname -s)
IS_MAC=false
if [ "$OS_TYPE" = "Darwin" ]; then
    IS_MAC=true
    echo "Detected macOS system"
fi

# Skip system dependencies and Python installation if only testing service config
if [ "$TEST_SERVICE_CONFIG" = false ]; then
    # Install system dependencies
    if [ "$IS_MAC" = true ]; then
        # macOS with Homebrew
        if ! command -v brew &> /dev/null; then
            echo "Homebrew is required for macOS installation."
            echo "Please install Homebrew first: https://brew.sh"
            exit 1
        fi
        
        echo "Installing system dependencies for macOS using Homebrew..."
        brew install portaudio python
        brew install xclip || echo "xclip installation failed, using pyperclip fallback"
    elif command -v apt-get &> /dev/null; then
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
        if [ "$IS_MAC" = true ]; then
            # Check for common macOS shells
            if [ -f "${HOME}/.zshrc" ]; then
                source "${HOME}/.zshrc"
            elif [ -f "${HOME}/.bash_profile" ]; then
                source "${HOME}/.bash_profile"
            fi
        else
            # Linux typically uses .bashrc
            source ~/.bashrc
        fi
        
        # If uv still not in PATH, add it temporarily
        if ! command -v uv &> /dev/null; then
            export PATH="${HOME}/.cargo/bin:$PATH"
        fi
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
    echo "1) tiny          - ~75MB  (fastest, least accurate)"
    echo "2) small         - ~500MB (good balance of speed and accuracy)"
    echo "3) medium        - ~1.5GB (slower, more accurate)"
    echo "4) large         - ~3GB   (slowest, most accurate)"
    echo "5) large-v2      - ~3GB   (improved large model)"
    echo "6) large-v3      - ~3GB   (latest large model)"
    echo "7) parakeet-tdt-0.6b-v2 - ~1.2GB (NVIDIA Parakeet model)"
    echo "0) No additional models"
    echo ""
    read -ra model_choices -p "Enter numbers of models to download separated by spaces (e.g. 1 3 7 or 0 to skip): "

    models_to_download=()
    for choice in "${model_choices[@]}"; do
        case $choice in
            1)
                models_to_download+=("tiny")
                ;;
            2)
                models_to_download+=("small")
                ;;
            3)
                models_to_download+=("medium")
                ;;
            4)
                models_to_download+=("large")
                ;;
            5)
                models_to_download+=("large-v2")
                ;;
            6)
                models_to_download+=("large-v3")
                ;;
            7)
                models_to_download+=("nvidia/parakeet-tdt-0.6b-v2")
                ;;
            0)
                # Skip downloading if user entered 0
                models_to_download=()
                break
                ;;
        esac
    done

    if [[ " ${models_to_download[@]} " == *"nvidia/parakeet-tdt-0.6b-v2"* ]]; then
        echo "Installing nemo_toolkit for Parakeet model..."
        uv pip install "nemo_toolkit[asr]"
    fi

    for model in "${models_to_download[@]}"; do
        echo "Downloading ${model} model..."
        if python -m transclip.download_models --model "${model}"; then
            echo "${model} model downloaded successfully."
        else
            echo "Warning: Failed to download ${model} model."
        fi
    done
fi

# Set VENV_PATH for service configuration
if [ "$TEST_SERVICE_CONFIG" = true ]; then
    VENV_PATH="${HOME}/.local/share/transclip/venv"
fi

# Create log files
mkdir -p "${HOME}/.local/share/transclip"
touch "${HOME}/.local/share/transclip/transclip.log"
touch "${HOME}/.local/share/transclip/transclip.error.log"

if [ "$IS_MAC" = true ]; then
    # macOS service setup with launchd
    echo "Configuring launchd service for macOS..."
    LAUNCH_AGENTS_DIR="${HOME}/Library/LaunchAgents"
    mkdir -p "${LAUNCH_AGENTS_DIR}"
    
    # Get the current directory for the WorkingDirectory
    CURRENT_DIR="$(pwd)"
    
    # Create plist file for launchd
    cat > "${LAUNCH_AGENTS_DIR}/com.user.transclip.plist" << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.user.transclip</string>
    <key>ProgramArguments</key>
    <array>
        <string>${VENV_PATH}/bin/python</string>
        <string>-m</string>
        <string>transclip</string>
    </array>
    <key>WorkingDirectory</key>
    <string>${CURRENT_DIR}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>${PATH}:/usr/local/bin</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>${HOME}/.local/share/transclip/transclip.log</string>
    <key>StandardErrorPath</key>
    <string>${HOME}/.local/share/transclip/transclip.error.log</string>
</dict>
</plist>
EOF

    # Load the service
    echo "Installing and starting the launchd service..."
    launchctl unload "${LAUNCH_AGENTS_DIR}/com.user.transclip.plist" 2>/dev/null || true
    launchctl load -w "${LAUNCH_AGENTS_DIR}/com.user.transclip.plist"
    
    echo -e "\nTransClip has been installed successfully on macOS!"
    echo "The application will start automatically on system startup."
    echo "You can control the service with:"
    echo "  Start:   launchctl load -w ~/Library/LaunchAgents/com.user.transclip.plist"
    echo "  Stop:    launchctl unload -w ~/Library/LaunchAgents/com.user.transclip.plist"
    echo "  Restart: launchctl unload ~/Library/LaunchAgents/com.user.transclip.plist && launchctl load -w ~/Library/LaunchAgents/com.user.transclip.plist"
    echo -e "\nLog files are located at:"
    echo "  ${HOME}/.local/share/transclip/transclip.log"
    echo "  ${HOME}/.local/share/transclip/transclip.error.log"
    
else
    # Linux systemd service setup
    echo "Configuring systemd service for Linux..."
    SYSTEMD_DIR="${HOME}/.config/systemd/user"
    mkdir -p "${SYSTEMD_DIR}"
    mkdir -p "${HOME}/.config/transclip"
    
    # Get the current directory for the WorkingDirectory
    CURRENT_DIR="$(pwd)"
    
    # Copy the systemd service file
    cp transclip.service "${SYSTEMD_DIR}/transclip.service"
    
    # Update the service file with the correct paths
    sed -i "s|ExecStart=.*|ExecStart=${VENV_PATH}/bin/python -m transclip|g" "${SYSTEMD_DIR}/transclip.service"
    sed -i "s|WorkingDirectory=.*|WorkingDirectory=${CURRENT_DIR}|g" "${SYSTEMD_DIR}/transclip.service"
    # If WorkingDirectory line doesn't exist, add it after ExecStart
    if ! grep -q "WorkingDirectory" "${SYSTEMD_DIR}/transclip.service"; then
        sed -i "/ExecStart=.*/a WorkingDirectory=${CURRENT_DIR}" "${SYSTEMD_DIR}/transclip.service"
    fi
    sed -i "s|Environment=PATH=.*|Environment=PATH=${PATH}:/usr/local/bin|g" "${SYSTEMD_DIR}/transclip.service"
    
    if [ "$TEST_SERVICE_CONFIG" = true ]; then
        echo "Service configuration test complete."
        echo "Service file updated at: ${SYSTEMD_DIR}/transclip.service"
        echo "ExecStart: $(grep ExecStart ${SYSTEMD_DIR}/transclip.service)"
        echo "WorkingDirectory: $(grep WorkingDirectory ${SYSTEMD_DIR}/transclip.service)"
        exit 0
    fi
    
    # Stop any existing service
    systemctl --user stop transclip.service || true
    
    # Install systemd service
    echo "Installing systemd service..."
    systemctl --user daemon-reload
    systemctl --user enable transclip.service
    systemctl --user start transclip.service
    
    echo -e "\nTransClip has been installed successfully on Linux!"
    echo "The application will start automatically on system startup."
    echo "You can control the service with:"
    echo "  Start:   systemctl --user start transclip"
    echo "  Stop:    systemctl --user stop transclip"
    echo "  Restart: systemctl --user restart transclip"
    echo "  Status:  systemctl --user status transclip"
    echo -e "\nLog files are located at:"
    echo "  ${HOME}/.local/share/transclip/transclip.log"
    echo "  ${HOME}/.local/share/transclip/transclip.error.log"
fi 
