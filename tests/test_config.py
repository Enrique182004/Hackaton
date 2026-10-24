from __future__ import annotations

from pathlib import Path

import pytest

from biobench_sar.config import DATA_DIR, PACKAGE_DIR, Settings


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


class Dummy:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.args, self.kwargs = args, kwargs


def _build(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> object:
    from biobench_sar import app as app_module
    from biobench_sar.core import vision_loop
    from biobench_sar.llm import tts_engine, vlm_client
    from biobench_sar.rag import vector_store

    for module, name in (
        (vision_loop, "VisionLoop"),
        (vector_store, "VectorStore"),
        (vlm_client, "VLMClient"),
        (tts_engine, "TTSEngine"),
    ):
        monkeypatch.setattr(module, name, Dummy)
    return app_module.build_services(settings, on_query=lambda q: None, on_state=lambda s: None)


def test_build_services_passes_silence_rms_to_audio(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    svc = _build(monkeypatch, Settings(silence_rms=0.042, calibration_file=tmp_path / "calibration.json"))
    assert svc.audio.silence_rms == 0.042  # type: ignore[attr-defined]


@pytest.mark.parametrize("profile", ["math", "biology"])
def test_build_services_passes_profile_to_vlm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, profile: str) -> None:
    svc = _build(monkeypatch, Settings(profile=profile, calibration_file=tmp_path / "calibration.json"))
    assert svc.vlm.kwargs["profile"] == profile  # type: ignore[attr-defined]


def test_profile_defaults_to_math(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SAR_PROFILE", raising=False)
    assert Settings().profile == "math"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("math", "math"),
        ("biology", "biology"),
        (" Biology ", "biology"),
        ("MATH", "math"),
        ("chem", "math"),
        ("", "math"),
    ],
)
def test_profile_env(monkeypatch: pytest.MonkeyPatch, raw: str, expected: str) -> None:
    monkeypatch.setenv("SAR_PROFILE", raw)
    assert Settings().profile == expected


def test_knowledge_dir_follows_profile() -> None:
    assert Settings(profile="math").knowledge_dir == PACKAGE_DIR / "knowledge" / "math"
    assert Settings(profile="biology").knowledge_dir == PACKAGE_DIR / "knowledge" / "biology"


def test_quiz_bank_file_default_and_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("SAR_QUIZ_BANK_FILE", raising=False)
    assert Settings().quiz_bank_file == DATA_DIR / "quiz_bank.json"
    monkeypatch.setenv("SAR_QUIZ_BANK_FILE", str(tmp_path / "bank.json"))
    assert Settings().quiz_bank_file == tmp_path / "bank.json"


def test_seed_marker_lives_with_the_store(tmp_path: Path) -> None:
    assert Settings(chroma_dir=tmp_path / "db").seed_marker_file == tmp_path / "db" / ".seeded"


def test_selection_file_defaults_to_data_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SAR_SELECTION_FILE", raising=False)
    assert Settings().selection_file == DATA_DIR / "selection.json"


def test_selection_file_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SAR_SELECTION_FILE", str(tmp_path / "picked.json"))
    assert Settings().selection_file == tmp_path / "picked.json"
