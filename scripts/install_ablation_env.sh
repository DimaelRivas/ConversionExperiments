#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "== Python =="
python --version

echo
echo "== Installing Python requirements =="
python -m pip install --upgrade pip
python -m pip install -r "${ROOT_DIR}/requirements.txt"

echo
echo "== Environment checks =="
python - <<'PY'
import importlib
import platform
import sys

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version.split()[0]}")
print(f"Platform: {platform.platform()}")

try:
    import docling
    print(f"Docling import: OK")
    print(f"Docling version: {getattr(docling, '__version__', 'unknown')}")
except Exception as exc:
    print(f"Docling import: FAILED ({exc})")

try:
    import torch
    print(f"PyTorch import: OK")
    print(f"CUDA available through PyTorch: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
except Exception as exc:
    print(f"PyTorch import/check: unavailable ({exc})")
PY

echo
echo "== nvidia-smi =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || true
else
  echo "nvidia-smi not found. This is expected on CPU-only machines."
fi

cat <<'EOF'

Notes:
- Tesseract OCR profiles require the system tesseract binary and language packs.
  Ubuntu example:
    sudo apt-get update
    sudo apt-get install -y tesseract-ocr tesseract-ocr-eng tesseract-ocr-spa
- EasyOCR downloads model weights on first use.
- GPU/VLM profiles are intentionally gated by --allow-gpu in the runner.
EOF
