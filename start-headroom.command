#!/bin/bash
# Headroom token-saving proxy. Double-click to install (first time) + run.
# Then launch Claude Code through it:  ANTHROPIC_BASE_URL=http://localhost:8787 claude
set -e
B=$'\033[1m'; N=$'\033[0m'

if ! command -v headroom >/dev/null 2>&1; then
  echo "${B}Installing Headroom (one-time)...${N}"
  if command -v uv >/dev/null 2>&1; then
    uv tool install --python 3.13 "headroom-ai[all]"
  else
    echo "uv not found. Install uv first:  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
  fi
fi

echo "${B}Starting Headroom proxy on http://localhost:8787${N}"
echo "In ANOTHER terminal, run Claude through it:"
echo "  ${B}ANTHROPIC_BASE_URL=http://localhost:8787 claude${N}"
echo "(Compression runs locally — nothing is sent anywhere to be compressed.)"
echo
headroom proxy --port 8787
