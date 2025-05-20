"""Utility script to download Whisper models in advance.

This script allows downloading Whisper models before running the main application,
which can be useful for slow connections or offline usage.
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

from faster_whisper import download_model

try:
    from nemo.collections.asr.models import EncDecRNNTModel
except Exception:  # pragma: no cover - optional dependency
    EncDecRNNTModel = None

from transclip.transcription import WhisperModelType

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

def get_model_size_mb(model_type: WhisperModelType) -> int:
    """Get the size of a model in megabytes.

    Args:
        model_type: The type of Whisper model.

    Returns:
        int: Size in megabytes.
    """
    sizes = {
        WhisperModelType.TINY: 75,
        WhisperModelType.BASE: 150,
        WhisperModelType.SMALL: 500,
        WhisperModelType.MEDIUM: 1500,
        WhisperModelType.LARGE: 3000,
        WhisperModelType.LARGE_V2: 3000,
        WhisperModelType.LARGE_V3: 3000,
        # WhisperModelType.PARAKEET_TDT_0_6B_V2: 1200,  # Disabled
    }
    return sizes.get(model_type, 3000)

def check_disk_space(required_mb: int, path: str = ".") -> bool:
    """Check if there's enough disk space available.

    Args:
        required_mb: Required space in megabytes.
        path: Path to check space on.

    Returns:
        bool: True if there's enough space, False otherwise.
    """
    total, used, free = shutil.disk_usage(path)
    free_mb = free // (1024 * 1024)
    if free_mb < required_mb:
        logger.error(
            f"Insufficient disk space. Need {required_mb}MB, but only {free_mb}MB available"
        )
        return False
    return True

# def _download_parakeet_model(cache_dir: str, force: bool) -> bool:
#     """Download the NVIDIA Parakeet model using NeMo."""
#     if EncDecRNNTModel is None:  # pragma: no cover - runtime guard
#         logger.error(
#             "nemo_toolkit is required for the Parakeet model. Install with 'pip install nemo_toolkit[asr]'"
#         )
#         return False
#
#     model_file = Path(cache_dir) / "nvidia_parakeet-tdt-0.6b-v2.nemo"
#     if model_file.exists() and not force:
#         logger.info("Model nvidia/parakeet-tdt-0.6b-v2 already exists at %s", model_file)
#         return True
#
#     model_size = get_model_size_mb(WhisperModelType.PARAKEET_TDT_0_6B_V2)
#     if not check_disk_space(model_size * 2, cache_dir):
#         return False
#
#     try:
#         logger.info("Downloading nvidia/parakeet-tdt-0.6b-v2 model with NeMo ...")
#         model = EncDecRNNTModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")
#         model.save_to(str(model_file))
#         logger.info("Successfully downloaded model to %s", model_file)
#         return True
#     except Exception as e:  # pragma: no cover - external call
#         logger.error("Failed to download Parakeet model: %s", e)
#         return False


def download_whisper_model(model_type: WhisperModelType, force: bool = False) -> bool:
    """Download a Whisper or NeMo model.

    Args:
        model_type: The type of model to download.
        force: Whether to force download even if model exists.

    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        # Set cache directory in user's home
        cache_dir = os.path.expanduser("~/.cache/whisper")
        os.makedirs(cache_dir, exist_ok=True)

        # if model_type is WhisperModelType.PARAKEET_TDT_0_6B_V2:
        #     return _download_parakeet_model(cache_dir, force)

        # Check if model is already downloaded
        model_dir = Path(cache_dir) / str(model_type)
        if model_dir.exists() and not force:
            logger.info("Model %s already exists at %s", model_type, model_dir)
            return True

        # Check disk space (need 2x for download + extraction)
        model_size = get_model_size_mb(model_type)
        if not check_disk_space(model_size * 2, cache_dir):
            return False

        # Enable faster downloads
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

        logger.info(f"Downloading {model_type} model (~{model_size}MB)...")
        download_model(
            model_type.value.lower(),
            cache_dir=cache_dir,
            local_files_only=False
        )

        logger.info(f"Successfully downloaded model to {cache_dir}")
        return True

    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        return False

def main() -> int:
    """Download Whisper models for TransClip.

    Returns:
        int: Exit code, 0 for success, 1 for error.
    """
    parser = argparse.ArgumentParser(description="Download Whisper models for TransClip")
    parser.add_argument(
        "--model",
        type=str,
        choices=[m.value for m in WhisperModelType],
        default=WhisperModelType.BASE.value,
        help="Model type to download"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download even if model exists"
    )
    args = parser.parse_args()

    model_type = WhisperModelType(args.model)
    logger.info(f"Selected model: {WhisperModelType.get_description(model_type)}")

    success = download_whisper_model(model_type, args.force)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
