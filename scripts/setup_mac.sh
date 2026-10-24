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
    uv venv --seed --python 3.12 .venv
  else
    python=""
    for candidate in python3.12 python3.11 python3.10; do
      if command -v "$candidate" >/dev/null; then python="$candidate"; break; fi
    done
    if [[ -z "$python" ]]; then
      echo "Python 3.10-3.12 not found. Install with: brew install python@3.12 (or install uv)"; exit 1
    fi
    "$python" -m venv .venv
  fi
fi

# 3. Python dependencies (pyaudio needs portaudio from step 1).
.venv/bin/python -m pip install --upgrade pip
CFLAGS="-I$(brew --prefix portaudio)/include" LDFLAGS="-L$(brew --prefix portaudio)/lib" \
  .venv/bin/python -m pip install -r requirements.txt

# 4. Room settings: start from the classroom kit (C930e webcam at 1080p, 1280x800 Epson projector).
if [[ ! -f sar.env && -f sar.env.example ]]; then
  cp sar.env.example sar.env
  echo "Created sar.env from sar.env.example (the classroom kit); edit it if your room differs."
fi

# 5. Optional: pull the local vision-language model (~3 GB).
if [[ "$PULL_MODEL" == "1" ]]; then
  pgrep -x ollama >/dev/null || { (ollama serve >/dev/null 2>&1 &); sleep 3; }
  ollama pull qwen2.5vl:3b
else
  echo "Skipping model pull. Run with --pull-model or: ollama pull qwen2.5vl:3b"
fi

cat <<'MSG'

Setup complete. Next:
  .venv/bin/python scripts/hardware_check.py
  SAR_VLM_BACKEND=mock .venv/bin/python -m biobench_sar         # no model needed
  .venv/bin/python -m biobench_sar                               # real VLM via Ollama
Open http://127.0.0.1:5050/ fullscreen on the projector, http://127.0.0.1:5050/dashboard on your laptop.
Room settings are in sar.env; docs/HARDWARE_DAY.md walks through the first run with the kit.
MSG
