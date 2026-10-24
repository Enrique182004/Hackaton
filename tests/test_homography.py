from __future__ import annotations

import cv2
import numpy as np
import pytest

from biobench_sar.core.homography import (
    Calibration,
    calibrate_from_frame,
    compute_homography,
    detect_corners,
    generate_checkerboard,
    projector_corner_points,
)

PW, PH = 1280, 720
# Synthetic camera view: projector image seen at an angle in a 960x540 frame.
CAM_SIZE = (960, 540)
SRC = np.float32([[0, 0], [PW, 0], [PW, PH], [0, PH]])
DST = np.float32([[80, 50], [880, 90], [850, 480], [110, 450]])


def _camera_view() -> tuple[np.ndarray, np.ndarray]:
    board = generate_checkerboard(PW, PH)
    m = cv2.getPerspectiveTransform(SRC, DST)
    frame = cv2.warpPerspective(board, m, CAM_SIZE, borderValue=(255, 255, 255))
    return frame, m


def test_checkerboard_shape_and_margin() -> None:
    img = generate_checkerboard(PW, PH)
    assert img.shape == (PH, PW, 3) and img.dtype == np.uint8
    assert (img[0, :] == 255).all() and (img[:, 0] == 255).all()


def test_corner_points_shape() -> None:
    pts = projector_corner_points(PW, PH, 9, 6)
    assert pts.shape == (54, 2) and pts.dtype == np.float32


def test_corner_order_matches_opencv() -> None:
    img = generate_checkerboard(PW, PH)
    detected = detect_corners(img)
    assert detected is not None
    expected = projector_corner_points(PW, PH)
    assert np.abs(detected - expected).max() < 2.0


def test_detect_corners_none_on_blank() -> None:
    assert detect_corners(np.full((300, 300, 3), 255, np.uint8)) is None


def test_compute_homography_failure() -> None:
    with pytest.raises(ValueError):
        compute_homography(np.zeros((2, 2), np.float32), np.zeros((2, 2), np.float32))


def test_full_pipeline_roundtrip() -> None:
    frame, _ = _camera_view()
    cal = calibrate_from_frame(frame, (PW, PH))
    assert cal is not None
    cam_pts = detect_corners(frame)
    assert cam_pts is not None
    mapped = cal.camera_to_projector(cam_pts)
    assert np.abs(mapped - projector_corner_points(PW, PH)).max() < 3.0
    warped = cal.warp_frame(frame)
    assert warped.shape == (PH, PW, 3)


def test_calibrate_returns_none_without_board() -> None:
    assert calibrate_from_frame(np.full((480, 640, 3), 255, np.uint8), (PW, PH)) is None


def test_save_load_roundtrip(tmp_path) -> None:
    cal = Calibration.identity((640, 480), (PW, PH))
    path = tmp_path / "sub" / "cal.json"
    cal.save(path)
    loaded = Calibration.load(path)
    assert loaded is not None
    assert loaded.projector_size == (PW, PH)
    assert np.allclose(loaded.matrix, cal.matrix)


def test_load_missing_or_corrupt(tmp_path) -> None:
    assert Calibration.load(tmp_path / "nope.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert Calibration.load(bad) is None


def test_identity_scales() -> None:
    cal = Calibration.identity((640, 480), (1280, 960))
    out = cal.camera_to_projector(np.array([[320, 240], [640, 480]], np.float32))
    assert np.allclose(out, [[640, 480], [1280, 960]])
    assert cal.warp_frame(np.zeros((480, 640, 3), np.uint8)).shape == (960, 1280, 3)


def test_save_is_atomic_and_leaves_no_tmp(tmp_path) -> None:
    path = tmp_path / "cal.json"
    Calibration.identity((640, 480), (PW, PH)).save(path)
    assert [p.name for p in tmp_path.iterdir()] == ["cal.json"]


def test_failed_save_keeps_previous_calibration(tmp_path, monkeypatch) -> None:
    import os

    path = tmp_path / "cal.json"
    original = Calibration.identity((640, 480), (PW, PH))
    original.save(path)

    def boom(src, dst) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        Calibration.identity((320, 240), (PW, PH)).save(path)
    monkeypatch.undo()

    loaded = Calibration.load(path)
    assert loaded is not None and np.allclose(loaded.matrix, original.matrix)
    assert [p.name for p in tmp_path.iterdir()] == ["cal.json"]


def test_load_rejects_projector_size_mismatch(tmp_path, caplog) -> None:
    path = tmp_path / "cal.json"
    Calibration.identity((640, 480), (PW, PH)).save(path)
    with caplog.at_level("WARNING"):
        assert Calibration.load(path, expected_projector_size=(1920, 1080)) is None
    assert any("projector size" in r.message for r in caplog.records)
    matched = Calibration.load(path, expected_projector_size=(PW, PH))
    assert matched is not None and matched.projector_size == (PW, PH)
    assert Calibration.load(path) is not None
