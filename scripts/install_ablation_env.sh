#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -n "${PYTHON:-}" ]]; then
  PYTHON_BIN="${PYTHON}"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  echo "ERROR: neither python nor python3 was found on PATH." >&2
  exit 1
fi

echo "== Python =="
"${PYTHON_BIN}" --version
"${PYTHON_BIN}" - <<'PY'
import sys

minimum = (3, 10)
if sys.version_info < minimum:
    raise SystemExit(f"Python {minimum[0]}.{minimum[1]}+ is required; found {sys.version.split()[0]}")
print(f"Python executable: {sys.executable}")
PY

echo
echo "== Installing Python requirements =="
"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install -r "${ROOT_DIR}/requirements.txt"

echo
echo "== Python package checks =="
"${PYTHON_BIN}" - <<'PY'
import importlib
import platform
import sys

packages = [
    "docling",
    "pandas",
    "yaml",
    "tqdm",
    "tabulate",
    "openpyxl",
    "psutil",
    "pynvml",
    "easyocr",
    "pytesseract",
    "rapidocr",
    "onnxruntime",
    "transformers",
    "accelerate",
]

print(f"Python version: {sys.version.split()[0]}")
print(f"Platform: {platform.platform()}")

for package in packages:
    try:
        module = importlib.import_module(package)
        version = getattr(module, "__version__", "unknown")
        print(f"{package}: OK ({version})")
    except Exception as exc:
        print(f"{package}: WARNING unavailable ({exc})")

try:
    import docling
    print(f"Docling version: {getattr(docling, '__version__', 'unknown')}")
except Exception as exc:
    print(f"Docling import: WARNING failed ({exc})")

try:
    import torch
    print(f"torch: OK ({getattr(torch, '__version__', 'unknown')})")
    print(f"CUDA available through torch: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
except Exception as exc:
    print(f"torch/CUDA check: WARNING unavailable ({exc})")
PY

echo
echo "== Tesseract =="
if command -v tesseract >/dev/null 2>&1; then
  tesseract --version | head -n 1 || true
  echo "Installed Tesseract languages:"
  tesseract --list-langs 2>/dev/null | sed '1d' | sort | tr '\n' ' '
  echo
else
  echo "WARNING: tesseract binary not found. Tesseract OCR profiles will fail until OS packages are installed."
fi

echo
echo "== nvidia-smi =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || true
else
  echo "WARNING: nvidia-smi not found. This is expected on CPU-only machines."
fi

cat <<'EOF'

System package notes:
- Tesseract OCR profiles require the system tesseract binary and language packs.
  Ubuntu example:
    sudo apt-get update
    sudo apt-get install -y tesseract-ocr tesseract-ocr-eng tesseract-ocr-spa
- This script does not assume sudo is available. Install those packages manually on hosts where OCR is needed.
- EasyOCR and VLM profiles may download model weights on first use.
- On GPU servers, verify the NVIDIA driver with nvidia-smi before running with --allow-gpu.
- If your PyTorch wheel is CPU-only on the GPU server, install the CUDA build recommended for that server first, then rerun this script.
EOF
