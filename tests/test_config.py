from __future__ import annotations

from pathlib import Path

import pytest

from biobench_sar import config
from biobench_sar.config import DATA_DIR, PACKAGE_DIR, Settings


def _clear(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "SAR_INK_MIN_AREA",
        "SAR_INK_BLOCK_SIZE",
        "SAR_INK_C",
        "SAR_INK_MAX_REGIONS",
        "SAR_INK_SMALL_MERGE_PX",
        "SAR_PROFILE",
        "SAR_LANGUAGE",
        "SAR_PROJECTOR_WIDTH",
        "SAR_PROJECTOR_HEIGHT",
    ):
        monkeypatch.delenv(name, raising=False)


def test_ink_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    s = Settings()
    # Math boards are many small symbols: 20 regions, specks within 24 px join their symbol.
    assert (s.ink_min_area, s.ink_block_size, s.ink_c, s.ink_max_regions) == (150, 31, 15, 20)
    assert s.ink_small_merge_px == 24
    assert s.ink_kwargs == {"min_area": 150, "max_regions": 20, "block_size": 31, "c": 15, "small_merge_px": 24}


def test_ink_max_regions_follows_the_profile_unless_set(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    assert Settings(profile="math").ink_max_regions == 20
    assert Settings(profile="biology").ink_max_regions == 10
    monkeypatch.setenv("SAR_PROFILE", "biology")
    assert Settings().ink_max_regions == 10
    monkeypatch.setenv("SAR_INK_MAX_REGIONS", "7")
    assert Settings().ink_max_regions == 7, "an explicit value wins for biology"
    monkeypatch.setenv("SAR_PROFILE", "math")
    assert Settings().ink_max_regions == 7, "and for math"


@pytest.mark.parametrize(("raw", "expected"), [("0", 0), ("40", 40), ("-1", 24), ("abc", 24), ("", 24)])
def test_ink_small_merge_px_env(monkeypatch: pytest.MonkeyPatch, raw: str, expected: int) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("SAR_INK_SMALL_MERGE_PX", raw)
    s = Settings()
    assert s.ink_small_merge_px == expected and s.ink_kwargs["small_merge_px"] == expected


def test_projector_defaults_to_the_epson_wxga_kit(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    s = Settings()
    assert s.projector_size == (1280, 800)
    assert s.projector_width * 10 == s.projector_height * 16, "16:10"


def test_projector_size_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear(monkeypatch)
    monkeypatch.setenv("SAR_PROJECTOR_WIDTH", "1920")
    monkeypatch.setenv("SAR_PROJECTOR_HEIGHT", "1080")
    assert Settings().projector_size == (1920, 1080)
    monkeypatch.setenv("SAR_PROJECTOR_WIDTH", "0")
    monkeypatch.setenv("SAR_PROJECTOR_HEIGHT", "wide")
    assert Settings().projector_size == (1280, 800), "nonsense falls back to the kit"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, "auto"), ("auto", "auto"), ("es", "es"), ("es-MX", "es"), ("English", "en"), ("EN", "en"), ("fr", "auto")],
)
def test_language_setting_env(monkeypatch: pytest.MonkeyPatch, raw: str | None, expected: str) -> None:
    _clear(monkeypatch)
    if raw is not None:
        monkeypatch.setenv("SAR_LANGUAGE", raw)
    assert Settings().language == expected


def test_language_passed_in_is_normalized() -> None:
    assert Settings(language="Español").language == "es"
    assert Settings(language="klingon").language == "auto"


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
    assert s.ink_max_regions == 20
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


@pytest.mark.parametrize("raw", ["abc", "", "0", "-5", "nan", "inf"])
def test_bad_timeout_or_record_seconds_falls_back(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    monkeypatch.setenv("SAR_VLM_TIMEOUT", raw)
    monkeypatch.setenv("SAR_RECORD_SECONDS", raw)
    settings = Settings()
    assert (settings.vlm_timeout, settings.record_seconds) == (90.0, 5.0)


def test_timeout_and_record_seconds_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAR_VLM_TIMEOUT", "30")
    monkeypatch.setenv("SAR_RECORD_SECONDS", "2.5")
    settings = Settings()
    assert (settings.vlm_timeout, settings.record_seconds) == (30.0, 2.5)


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


@pytest.mark.parametrize(("language", "whisper"), [("auto", "auto"), ("es", "es"), ("en", "en")])
def test_build_services_passes_language_to_audio(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, language: str, whisper: str
) -> None:
    svc = _build(monkeypatch, Settings(language=language, calibration_file=tmp_path / "calibration.json"))
    assert svc.audio.language == whisper  # type: ignore[attr-defined]


@pytest.mark.parametrize("profile", ["math", "biology"])
def test_build_services_passes_profile_to_vlm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, profile: str) -> None:
    svc = _build(monkeypatch, Settings(profile=profile, calibration_file=tmp_path / "calibration.json"))
    assert svc.vlm.kwargs["profile"] == profile  # type: ignore[attr-defined]


def test_build_services_builds_the_real_vlm_client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The fakes take **kwargs; this catches a VLMClient signature that the assistant's call does not fit."""
    from biobench_sar import app as app_module
    from biobench_sar.core import vision_loop
    from biobench_sar.llm import tts_engine, vlm_client
    from biobench_sar.rag import vector_store

    for module, name in ((vision_loop, "VisionLoop"), (vector_store, "VectorStore"), (tts_engine, "TTSEngine")):
        monkeypatch.setattr(module, name, Dummy)
    settings = Settings(vlm_backend="mock", calibration_file=tmp_path / "calibration.json")
    svc = app_module.build_services(settings, on_query=lambda q: None, on_state=lambda s: None)
    assert isinstance(svc.vlm, vlm_client.VLMClient) and svc.vlm.profile == "math"
    task = {"kind": "quiz", "stage": "dictate", "number": 1, "total": 3, "problem": "x + 2 = 5", "answer": "x = 3"}
    action = svc.vlm.evaluate(
        "quiz me", None, [], [], settings.projector_size, mode="quiz", task=task, recent_turns=[{"q": "hi", "a": "ok"}]
    )
    assert action.animation_state == "idle" and "x plus two equals five" in action.spoken_text


def test_quiz_length_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    assert Settings().quiz_length == 10
    monkeypatch.setenv("SAR_QUIZ_LENGTH", "5")
    assert Settings().quiz_length == 5
    for bad in ("0", "-3", "ten"):
        monkeypatch.setenv("SAR_QUIZ_LENGTH", bad)
        assert Settings().quiz_length == 10


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


def test_accessibility_file_default_and_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("SAR_ACCESSIBILITY_FILE", raising=False)
    assert Settings().accessibility_file == DATA_DIR / "accessibility.json"
    monkeypatch.setenv("SAR_ACCESSIBILITY_FILE", str(tmp_path / "room.json"))
    assert Settings().accessibility_file == tmp_path / "room.json"


def test_accessibility_file_follows_the_launchers_data_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # The desktop launcher sets config.DATA_DIR after config is imported; the default is read when Settings is built.
    monkeypatch.delenv("SAR_ACCESSIBILITY_FILE", raising=False)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    assert Settings().accessibility_file == tmp_path / "data" / "accessibility.json"


# ---- LAN mode (phones join with a PIN; docs/MOBILE.md) ----------------------------------------------
def _clear_lan(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("SAR_LAN", "SAR_HOST", "SAR_LAN_HOSTNAMES"):
        monkeypatch.delenv(name, raising=False)


def test_lan_mode_is_off_by_default_on_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_lan(monkeypatch)
    s = Settings()
    assert (s.lan, s.host, s.lan_hostnames) == (False, "127.0.0.1", ())


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on", " 1 "])
def test_sar_lan_turns_lan_mode_on_and_listens_on_every_interface(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    _clear_lan(monkeypatch)
    monkeypatch.setenv("SAR_LAN", raw)
    s = Settings()
    assert (s.lan, s.host) == (True, "0.0.0.0")


@pytest.mark.parametrize("raw", ["0", "", "false", "no", "off", "2"])
def test_sar_lan_off_values(monkeypatch: pytest.MonkeyPatch, raw: str) -> None:
    _clear_lan(monkeypatch)
    monkeypatch.setenv("SAR_LAN", raw)
    s = Settings()
    assert (s.lan, s.host) == (False, "127.0.0.1")


def test_an_explicit_sar_host_wins_over_sar_lan(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_lan(monkeypatch)
    monkeypatch.setenv("SAR_LAN", "1")
    monkeypatch.setenv("SAR_HOST", "127.0.0.1")
    s = Settings()
    assert (s.lan, s.host) == (True, "127.0.0.1"), "the gate stays on; main() warns that phones can't reach it"


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("0.0.0.0", True),
        ("", True),
        ("::", True),
        ("192.168.1.20", True),
        ("my-laptop.local", True),
        ("127.0.0.1", False),
        ("127.0.0.2", False),
        ("::1", False),
        ("localhost", False),
        (" 127.0.0.1 ", False),
    ],
)
def test_any_network_bind_turns_the_pin_gate_on_without_sar_lan(
    monkeypatch: pytest.MonkeyPatch, host: str, expected: bool
) -> None:
    _clear_lan(monkeypatch)
    monkeypatch.setenv("SAR_HOST", host)
    assert Settings().lan is expected


def test_a_network_bind_can_never_be_built_ungated() -> None:
    import dataclasses

    assert Settings(lan=False, host="0.0.0.0").lan is True
    assert dataclasses.replace(Settings(lan=False, host="127.0.0.1"), host="0.0.0.0").lan is True
    assert Settings(lan=False, host="127.0.0.1").lan is False
    assert Settings(lan=True, host="127.0.0.1").lan is True


def test_lan_hostnames_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_lan(monkeypatch)
    monkeypatch.setenv("SAR_LAN_HOSTNAMES", "My-Laptop.local, lab-pc ,*.evil, 10.0.0.1, my-laptop.local.")
    assert Settings().lan_hostnames == ("my-laptop.local", "lab-pc")


def test_lan_hostnames_passed_in_are_normalized() -> None:
    assert Settings(lan_hostnames=("My-Laptop.local", "bad name", "lab-pc")).lan_hostnames == (  # type: ignore[arg-type]
        "my-laptop.local",
        "lab-pc",
    )
    assert Settings(lan_hostnames="a.local,b.local").lan_hostnames == ("a.local", "b.local")  # type: ignore[arg-type]
    assert Settings(lan_hostnames=None).lan_hostnames == ()  # type: ignore[arg-type]
