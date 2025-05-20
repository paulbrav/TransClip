import json
import os
from pathlib import Path
from typing import Any, Dict, cast

CONFIG_HOME = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
CONFIG_DIR = CONFIG_HOME / "transclip"
CONFIG_FILE = CONFIG_DIR / "config.json"

DEFAULT_CONFIG: Dict[str, Any] = {
    "recording_key": "Key.home",
    "auto_paste": False,
    "cleanup_enable": False,
    "cleanup_punct_model": "fastpunct",
    "cleanup_llm_model": "tinyllama.gguf",
    "cleanup_threads": 4,
    "cleanup_gpu_layers": 0,
    "cleanup_prompt": (
        "Rewrite the following transcript into clear, well-punctuated English.\n\n"
        "INPUT:\n{text}\n\nCLEANED:"
    ),
}


def load_config() -> Dict[str, Any]:
    """Load configuration from disk.

    Returns an empty dict if the config file does not exist or is invalid.
    """
    try:
        with CONFIG_FILE.open("r", encoding="utf-8") as fh:
            return cast(Dict[str, Any], json.load(fh))
    except FileNotFoundError:
        return {}
    except Exception:
        return {}


def save_config(data: Dict[str, Any]) -> None:
    """Save configuration to disk."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with CONFIG_FILE.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=4)

