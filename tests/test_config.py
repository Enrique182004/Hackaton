from __future__ import annotations

from pathlib import Path

import pytest

from biobench_sar.config import DATA_DIR, Settings


def _clear(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("SAR_INK_MIN_AREA", "SAR_INK_BLOCK_SIZE", "SAR_INK_C", "SAR_INK_MAX_REGIONS"):
        monkeypatch.delenv(name, raising=False)


def test_ink_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    s = Settings()
    assert (s.ink_min_area, s.ink_block_size, s.ink_c, s.ink_max_regions) == (150, 31, 15, 10)


def test_ink_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("SAR_INK_MIN_AREA", "300")
    monkeypatch.setenv("SAR_INK_BLOCK_SIZE", "41")
    monkeypatch.setenv("SAR_INK_C", "9")
    monkeypatch.setenv("SAR_INK_MAX_REGIONS", "4")
    s = Settings()
    assert (s.ink_min_area, s.ink_block_size, s.ink_c, s.ink_max_regions) == (300, 41, 9, 4)


@pytest.mark.parametrize(("raw", "expected"), [("30", 31), ("4", 5), ("3", 3), ("2", 3), ("1", 3), ("32", 33)])
def test_block_size_forced_odd_and_at_least_three(monkeypatch: pytest.MonkeyPatch, raw: str, expected: int) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("SAR_INK_BLOCK_SIZE", raw)
    size = Settings().ink_block_size
    assert size == expected and size % 2 == 1 and size >= 3


@pytest.mark.parametrize("raw", ["-5", "0", "abc", "", "1.5", "nan"])
def test_garbage_or_negative_falls_back_to_defaults(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    _clear(monkeypatch)
    for name in ("SAR_INK_MIN_AREA", "SAR_INK_BLOCK_SIZE", "SAR_INK_C", "SAR_INK_MAX_REGIONS"):
        monkeypatch.setenv(name, raw)
    s = Settings()
    assert s.ink_min_area == 150
    assert s.ink_block_size == 31
    assert s.ink_max_regions == 10
    # C == 0 is a legal (if extreme) setting; negatives and garbage are not.
    assert s.ink_c == (0 if raw == "0" else 15)


def test_ink_settings_feed_find_ink_regions(monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np

    from biobench_sar.core.vision_loop import find_ink_regions

    _clear(monkeypatch)
    s = Settings()
    img = np.full((200, 200, 3), 255, np.uint8)
    img[50:150, 50:60] = 0
    regions = find_ink_regions(
        img, min_area=s.ink_min_area, max_regions=s.ink_max_regions, block_size=s.ink_block_size, c=s.ink_c
    )
    assert len(regions) == 1


def test_silence_rms_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SAR_SILENCE_RMS", raising=False)
    assert Settings().silence_rms == 0.01


@pytest.mark.parametrize(("raw", "expected"), [("0.05", 0.05), ("0", 0.0), ("1e-3", 0.001)])
def test_silence_rms_env_override(monkeypatch: pytest.MonkeyPatch, raw: str, expected: float) -> None:
    monkeypatch.setenv("SAR_SILENCE_RMS", raw)
    assert Settings().silence_rms == expected


@pytest.mark.parametrize("raw", ["-0.1", "abc", "", "nan", "inf"])
def test_silence_rms_invalid_or_negative_falls_back(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv("SAR_SILENCE_RMS", raw)
    assert Settings().silence_rms == 0.01


def test_build_services_passes_silence_rms_to_audio(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from biobench_sar import app as app_module
    from biobench_sar.core import vision_loop
    from biobench_sar.llm import tts_engine, vlm_client
    from biobench_sar.rag import vector_store

    class Dummy:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

    for module, name in (
        (vision_loop, "VisionLoop"),
        (vector_store, "VectorStore"),
        (vlm_client, "VLMClient"),
        (tts_engine, "TTSEngine"),
    ):
        monkeypatch.setattr(module, name, Dummy)
    settings = Settings(silence_rms=0.042, calibration_file=tmp_path / "calibration.json")
    svc = app_module.build_services(settings, on_query=lambda q: None, on_state=lambda s: None)
    assert svc.audio.silence_rms == 0.042


def test_selection_file_defaults_to_data_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SAR_SELECTION_FILE", raising=False)
    assert Settings().selection_file == DATA_DIR / "selection.json"


def test_selection_file_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SAR_SELECTION_FILE", str(tmp_path / "picked.json"))
    assert Settings().selection_file == tmp_path / "picked.json"
