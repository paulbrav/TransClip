#!/usr/bin/env bash
set -euo pipefail

venv_dir="${VENV_DIR:-.venv}"
python_version="${PYTHON_VERSION:-3.13}"

uv venv --python "$python_version" "$venv_dir"
uv pip install --python "$venv_dir/bin/python" \
  --index-url https://rocm.nightlies.amd.com/v2/gfx1151/ \
  --pre torch torchaudio torchvision pytorch-triton-rocm
uv pip install --python "$venv_dir/bin/python" \
  -e . 'transformers>=4.52.1' 'accelerate>=1.0' 'soundfile>=0.12' 'sounddevice>=0.5'
FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE MAX_JOBS="${MAX_JOBS:-4}" \
  uv pip install --python "$venv_dir/bin/python" --no-deps \
  flash-attn==2.8.3 --no-build-isolation
uv pip install --python "$venv_dir/bin/python" einops
uv pip install --python "$venv_dir/bin/python" flash-linear-attention

"$venv_dir/bin/python" -m transclip.cli models list
