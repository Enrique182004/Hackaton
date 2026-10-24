"""Check a built macOS app the way a teacher's Mac will see it; CI runs it on the unzipped download.

    python packaging/check_macos_app.py "dist/SAR Whiteboard Assistant.app" --version 0.1.0

Checks Info.plist (camera and microphone usage strings, bundle id, version) and that no bundled binary
needs a newer macOS than LSMinimumSystemVersion promises. Exit code 1, with GitHub ::error:: lines, on
any problem.
"""

from __future__ import annotations

import argparse
import plistlib
import re
import subprocess
import sys
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

BUNDLE_ID = "org.codecore.sarwhiteboard"
# Without these macOS denies the camera and microphone without asking.
USAGE_KEYS = ("NSCameraUsageDescription", "NSMicrophoneUsageDescription")
MACHO_MAGICS = {
    bytes.fromhex(m) for m in ("feedface", "cefaedfe", "feedfacf", "cffaedfe", "cafebabe", "bebafeca", "cafebabf")
}
OTOOL_BATCH = 200

Version = tuple[int, ...]
Otool = Callable[[Sequence[Path]], str]


def parse_version(text: object) -> Version | None:
    if not isinstance(text, str) or not re.fullmatch(r"\d+(\.\d+)*", text.strip()):
        return None
    return tuple(int(part) for part in text.strip().split("."))


def show(version: Version) -> str:
    return ".".join(str(part) for part in version)


def macho_files(root: Path) -> Iterator[Path]:
    """Executables and libraries under `root`, by magic number; symlinks (bundle aliases) are skipped."""
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            with path.open("rb") as handle:
                magic = handle.read(4)
        except OSError:
            continue
        if magic in MACHO_MAGICS:
            yield path


def parse_minimum_os(otool_output: str) -> dict[str, Version]:
    """{binary: the newest minimum macOS among its slices} from `otool -l` output for one or more files.

    Newer binaries state it as `minos` in LC_BUILD_VERSION, older ones as `version` in LC_VERSION_MIN_MACOSX.
    """
    found: dict[str, Version] = {}
    current: str | None = None
    command = ""
    for line in otool_output.splitlines():
        if line and not line[0].isspace() and line.endswith(":") and not line.startswith("Load command"):
            current = re.sub(r" \(architecture [^)]+\)$", "", line[:-1])
            command = ""
            continue
        fields = line.split()
        if current is None or len(fields) != 2:
            continue
        key, value = fields
        if key == "cmd":
            command = value
        elif (key, command) in (("minos", "LC_BUILD_VERSION"), ("version", "LC_VERSION_MIN_MACOSX")):
            version = parse_version(value)
            if version is not None:
                found[current] = max(found.get(current, version), version)
    return found


def run_otool(paths: Sequence[Path]) -> str:
    result = subprocess.run(["otool", "-l", *map(str, paths)], capture_output=True, text=True, check=False)
    return result.stdout


def minimum_os_of(paths: Sequence[Path], otool: Otool | None = None) -> dict[str, Version]:
    otool = otool or run_otool
    found: dict[str, Version] = {}
    for start in range(0, len(paths), OTOOL_BATCH):
        found.update(parse_minimum_os(otool(paths[start : start + OTOOL_BATCH])))
    return found


def check_app(app: Path, version: str, otool: Otool | None = None) -> tuple[list[str], list[str]]:
    """(errors, report lines) for the .app bundle at `app`."""
    errors: list[str] = []
    report: list[str] = []
    plist_path = app / "Contents" / "Info.plist"
    try:
        info = plistlib.loads(plist_path.read_bytes())
    except (OSError, plistlib.InvalidFileException, ValueError) as exc:
        return [f"cannot read {plist_path}: {exc}"], report

    for key in USAGE_KEYS:
        text = str(info.get(key, "")).strip()
        if text:
            report.append(f"{key}: {text}")
        else:
            errors.append(f"{key} is missing or empty in Info.plist: macOS would refuse access without asking")
    if info.get("CFBundleIdentifier") != BUNDLE_ID:
        errors.append(f"bundle id is {info.get('CFBundleIdentifier')!r}, expected {BUNDLE_ID!r}")
    if info.get("CFBundleShortVersionString") != version:
        errors.append(f"app version is {info.get('CFBundleShortVersionString')!r}, expected {version!r}")

    minimum = parse_version(info.get("LSMinimumSystemVersion"))
    if minimum is None:
        errors.append(f"LSMinimumSystemVersion is {info.get('LSMinimumSystemVersion')!r}, not a version")
        return errors, report
    binaries = list(macho_files(app / "Contents"))
    needs = minimum_os_of(binaries, otool)
    if not needs:
        errors.append(f"found no macOS version in {len(binaries)} bundled binaries (is otool installed?)")
        return errors, report
    newest = max(needs.values())
    report.append(f"LSMinimumSystemVersion {show(minimum)}; {len(needs)} binaries need at most macOS {show(newest)}")
    too_new = sorted((v, name) for name, v in needs.items() if v > minimum)
    if too_new:
        examples = ", ".join(f"{Path(name).name} ({show(v)})" for v, name in too_new[-5:])
        errors.append(
            f"LSMinimumSystemVersion is {show(minimum)}, but {len(too_new)} bundled binaries need a newer macOS: "
            f"{examples}. Raise LSMinimumSystemVersion in packaging/sar.spec and the docs."
        )
    return errors, report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("app", type=Path, help="path to SAR Whiteboard Assistant.app")
    parser.add_argument("--version", required=True, help="expected CFBundleShortVersionString")
    args = parser.parse_args(argv)
    errors, report = check_app(args.app, args.version)
    for line in report:
        print(line)
    for error in errors:
        print(f"::error::{error}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
