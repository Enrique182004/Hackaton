#!/usr/bin/env bash
# Setup for the SAR Whiteboard Assistant on macOS.
# Usage: bash scripts/setup_mac.sh [--pull-model]
set -euo pipefail

PULL_MODEL=0
for arg in "$@"; do
  [[ "$arg" == "--pull-model" ]] && PULL_MODEL=1
done

cd "$(dirname "$0")/.."

# 1. System binaries via Homebrew (skipped when already installed).
command -v brew >/dev/null || { echo "Install Homebrew first: https://brew.sh"; exit 1; }
for formula in portaudio tesseract ffmpeg; do
  brew list --formula "$formula" >/dev/null 2>&1 || brew install "$formula"
done
# Ollama: either the formula or the desktop app counts.
if ! command -v ollama >/dev/null; then
  brew install ollama
fi

# 2. Python venv (3.10-3.12; openai-whisper/torch lag behind newest Python).
if [[ ! -d .venv ]]; then
  if command -v uv >/dev/null; then
    uv venv --python 3.12 .venv
  elif command -v python3.12 >/dev/null; then
    python3.12 -m venv .venv
  else
    echo "Python 3.12 not found. Install with: brew install python@3.12 (or install uv)"; exit 1
  fi
fi

# 3. Python dependencies (pyaudio needs portaudio from step 1).
.venv/bin/python -m pip install --upgrade pip
CFLAGS="-I$(brew --prefix portaudio)/include" LDFLAGS="-L$(brew --prefix portaudio)/lib" \
  .venv/bin/python -m pip install -r requirements.txt

# 4. Optional: pull the local vision-language model (~3 GB).
if [[ "$PULL_MODEL" == "1" ]]; then
  pgrep -x ollama >/dev/null || { (ollama serve >/dev/null 2>&1 &); sleep 3; }
  ollama pull qwen2.5vl:3b
else
  echo "Skipping model pull. Run with --pull-model or: ollama pull qwen2.5vl:3b"
fi

cat <<'MSG'

Setup complete. Next:
  .venv/bin/python scripts/hardware_check.py
  SAR_VLM_BACKEND=mock .venv/bin/python -m biobench_sar.app     # no model needed
  .venv/bin/python -m biobench_sar.app                           # real VLM via Ollama
Open http://127.0.0.1:5000/ fullscreen on the projector, /precharge on your laptop.
MSG
