from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("hardware_check", ROOT / "scripts" / "hardware_check.py")
hc = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(hc)

DISPLAYS = """Graphics/Displays:
    Apple M4:
      Displays:
        Color LCD:
          Resolution: 2560 x 1664 Retina
        EPSON PJ:
          Resolution: 1280 x 800 (WXGA - Widescreen eXtended Graphics Array)
"""

CAMERAS = """Camera:

    MacBook Air Camera:

      Model ID: MacBook Air Camera

    Logitech Webcam C930e:

      Model ID: UVC Camera VendorID_1133 ProductID_2115
"""


def test_parse_resolutions() -> None:
    assert hc.parse_resolutions(DISPLAYS) == [(2560, 1664), (1280, 800)]


def test_projector_found_at_native_resolution() -> None:
    assert hc.projector_display_result([(2560, 1664), (1280, 800)], (1280, 800))[1] == "PASS"


def test_projector_wrong_resolution_warns_with_fix() -> None:
    _name, status, detail = hc.projector_display_result([(2560, 1664), (1920, 1080)], (1280, 800))
    assert status == "WARN" and "SAR_PROJECTOR_WIDTH" in detail


def test_projector_missing_warns() -> None:
    assert hc.projector_display_result([(2560, 1664)], (1280, 800))[1] == "WARN"


def test_parse_camera_names() -> None:
    assert hc.parse_camera_names(CAMERAS) == ["MacBook Air Camera", "Logitech Webcam C930e"]


def test_resolution_status() -> None:
    assert hc.resolution_status((1920, 1080), (1920, 1080)) == "PASS"
    assert hc.resolution_status((1280, 720), (1920, 1080)) == "WARN"
