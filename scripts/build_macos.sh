#!/usr/bin/env bash
# Build the macOS desktop app locally: the same steps as .github/workflows/build.yml.
# Usage: bash scripts/build_macos.sh [--whisper] [--skip-self-test] [--out DIR]
#   --out DIR   put dist/ and the PyInstaller work folder under DIR (default: the repo's dist/ and build/)
# Needs .venv from scripts/setup_mac.sh. Output: <out>/SAR Whiteboard Assistant.app and
# <out>/SAR-Whiteboard-Assistant-<version>-macOS-<arch>.zip
set -euo pipefail

WHISPER=0
SELF_TEST=1
OUT=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --whisper) WHISPER=1 ;;
    --skip-self-test) SELF_TEST=0 ;;
    --out) OUT="$2"; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

cd "$(dirname "$0")/.."
PY=.venv/bin/python
[[ -x "$PY" ]] || { echo "No .venv here. Run: bash scripts/setup_mac.sh" >&2; exit 1; }

if [[ -n "$OUT" ]]; then
  DIST="$OUT/dist"
  WORK="$OUT/build"
else
  DIST=dist
  WORK=build
fi

echo "==> Install PyInstaller"
if "$PY" -m pip --version >/dev/null 2>&1; then
  "$PY" -m pip install --upgrade "pyinstaller>=6.10"
else
  uv pip install --python "$PY" "pyinstaller>=6.10"  # uv-made venvs have no pip
fi

VERSION=$("$PY" -c 'import biobench_sar; print(biobench_sar.__version__)')
ARCH=$(uname -m)
echo "Version $VERSION ($ARCH)"

echo "==> Fetch models"
if [[ "$WHISPER" == "1" ]]; then
  "$PY" packaging/fetch_models.py --whisper
else
  "$PY" packaging/fetch_models.py
fi

echo "==> PyInstaller"
"$PY" -m PyInstaller packaging/sar.spec --noconfirm --distpath "$DIST" --workpath "$WORK"

APP="$DIST/SAR Whiteboard Assistant.app"
if [[ "$SELF_TEST" == "1" ]]; then
  echo "==> Self-test"
  "$APP/Contents/MacOS/SAR Whiteboard Assistant" --self-test
fi

echo "==> Zip"
ZIP="$DIST/SAR-Whiteboard-Assistant-$VERSION-macOS-$ARCH.zip"
rm -f "$ZIP"
ditto -c -k --keepParent "$APP" "$ZIP"
echo "App: $APP"
echo "Zip: $ZIP ($(du -h "$ZIP" | cut -f1))"
