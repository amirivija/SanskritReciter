#!/usr/bin/env bash
# One-time local setup: vendor code (if missing) + Python deps + model weights.
# After this, inference needs no network.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
elif ! command -v "$PY" >/dev/null; then
  echo "python3 not found"; exit 1
fi

echo "==> Python: $PY ($($PY --version))"

# Vendor trees
if [[ ! -d vendor/vagdhenu/.git && ! -f vendor/vagdhenu/src/render_core.py ]]; then
  echo "==> Cloning Vāgdhenu"
  git clone --depth 1 https://github.com/prathoshap/vagdhenu.git vendor/vagdhenu
fi
if [[ ! -f vendor/BigVGAN/bigvgan.py ]]; then
  echo "==> Cloning NVIDIA BigVGAN"
  git clone --depth 1 https://github.com/NVIDIA/BigVGAN.git vendor/BigVGAN
fi

echo "==> Installing Python package + deps"
$PY -m pip install -U pip wheel
# Platform torch if missing
$PY -c "import torch" 2>/dev/null || $PY -m pip install torch torchaudio
$PY -m pip install -r requirements.txt
$PY -m pip install -e .

echo "==> Downloading model weights into models/"
$PY scripts/download_models.py

echo ""
echo "✓ Setup complete. Offline check:"
echo "  $PY -m sanskrit_reciter --check"
echo "  $PY -m sanskrit_reciter examples/krishna.txt -o out/krishna.wav"
