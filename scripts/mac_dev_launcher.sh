#!/usr/bin/env bash
# Builds build/sar-launcher/SAR Launcher.app: a tiny ad-hoc-signed app that asks macOS for the camera and the
# microphone, then runs the assistant from this checkout. Use it when the camera stays dark from Terminal and
# macOS never asks (docs/HARDWARE_DAY.md, step 10). Needs the Xcode command line tools (swiftc).
# Usage: bash scripts/mac_dev_launcher.sh [--open]
set -euo pipefail

OPEN=0
for arg in "$@"; do
  case "$arg" in
    --open) OPEN=1 ;;
    *) echo "Usage: bash scripts/mac_dev_launcher.sh [--open]" >&2; exit 2 ;;
  esac
done

[[ "$(uname -s)" == "Darwin" ]] || { echo "The launcher is for macOS only." >&2; exit 1; }
# /usr/bin/swiftc exists even without the command line tools (it only offers to install them), so ask xcode-select.
if ! xcode-select -p >/dev/null 2>&1 || ! command -v swiftc >/dev/null; then
  echo "swiftc not found. Install the command line tools: xcode-select --install" >&2
  exit 1
fi

cd "$(dirname "$0")/.."
CHECKOUT="$(pwd -P)"
OUT="$CHECKOUT/build/sar-launcher"
APP="$OUT/SAR Launcher.app"
BUNDLE_ID="local.sar.dev-launcher"

if pgrep -f "$APP/Contents/MacOS/SARLauncher" >/dev/null; then
  echo "SAR Launcher is running. Stop the assistant first: pkill -f SARLauncher" >&2
  exit 1
fi

# Build next to the old app and swap at the end, so a failed build leaves the working launcher in place.
NEW="$OUT/.new/SAR Launcher.app"
rm -rf "$OUT/.new"
mkdir -p "$NEW/Contents/MacOS"
swiftc -O -o "$NEW/Contents/MacOS/SARLauncher" scripts/mac_dev_launcher.swift

PLIST="$NEW/Contents/Info.plist"
cat > "$PLIST" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>SAR Launcher</string>
  <key>CFBundleExecutable</key><string>SARLauncher</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>LSUIElement</key><true/>
  <key>NSCameraUsageDescription</key><string>The assistant watches the whiteboard through the webcam.</string>
  <key>NSMicrophoneUsageDescription</key><string>The assistant listens for spoken questions.</string>
</dict>
</plist>
PLIST
plutil -insert CFBundleIdentifier -string "$BUNDLE_ID" "$PLIST"
plutil -insert SARCheckout -string "$CHECKOUT" "$PLIST"
plutil -lint "$PLIST" >/dev/null
codesign --force --sign - "$NEW"
rm -rf "$APP"
mv "$NEW" "$APP"
rm -rf "$OUT/.new"

cat <<MSG
Built $APP
  for the checkout $CHECKOUT
Open it (Finder, or: open "$APP"). macOS asks once for the camera and once for the microphone: allow both.
It runs .venv/bin/python -m biobench_sar here; its output goes to $OUT/launcher.log.
Another checkout, while the launcher is not running: open "$APP" --args /absolute/path/to/that/checkout
Stop it: pkill -f SARLauncher
No question after a rebuild? tccutil reset Camera $BUNDLE_ID; tccutil reset Microphone $BUNDLE_ID; then open it again.
MSG

if [[ "$OPEN" == "1" ]]; then
  open "$APP"
fi
