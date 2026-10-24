from __future__ import annotations

import dataclasses
import json
import threading
from pathlib import Path

import pytest

from biobench_sar.accessibility import (
    SPEECH_RATES,
    TEXT_SIZES,
    AccessibilitySettings,
    AccessibilityStore,
    parse_changes,
)
from biobench_sar.config import Settings

KEYS = ["captions_stay", "text_size", "speech_rate", "calm", "high_contrast"]
DEFAULTS = {
    "captions_stay": False,
    "text_size": "normal",
    "speech_rate": "normal",
    "calm": False,
    "high_contrast": False,
}
NOT_AN_OBJECT = "send a JSON object with at least one of: captions_stay, text_size, speech_rate, calm, high_contrast"


# ---- the settings ----------------------------------------------------------------------------------------------------


def test_defaults_are_the_room_without_any_setting() -> None:
    settings = AccessibilitySettings()
    assert settings.to_dict() == DEFAULTS
    assert settings.rate_factor == 1.0


def test_to_dict_has_all_five_keys_in_order() -> None:
    settings = AccessibilitySettings(captions_stay=True, text_size="xlarge", speech_rate="faster", high_contrast=True)
    assert list(settings.to_dict()) == KEYS
    assert settings.to_dict() == {
        "captions_stay": True,
        "text_size": "xlarge",
        "speech_rate": "faster",
        "calm": False,
        "high_contrast": True,
    }


def test_the_allowed_values() -> None:
    assert TEXT_SIZES == ("normal", "large", "xlarge")
    assert SPEECH_RATES == {"slower": 0.75, "normal": 1.0, "faster": 1.25}


@pytest.mark.parametrize(("rate", "factor"), [("slower", 0.75), ("normal", 1.0), ("faster", 1.25)])
def test_rate_factor(rate: str, factor: float) -> None:
    assert AccessibilitySettings(speech_rate=rate).rate_factor == factor


def test_settings_are_frozen() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        AccessibilitySettings().calm = True  # type: ignore[misc]


# ---- parse_changes (the PUT /api/accessibility body) -----------------------------------------------------------------


@pytest.mark.parametrize(
    "body",
    [
        {"captions_stay": True},
        {"captions_stay": False},
        {"text_size": "normal"},
        {"text_size": "large"},
        {"text_size": "xlarge"},
        {"speech_rate": "slower"},
        {"speech_rate": "normal"},
        {"speech_rate": "faster"},
        {"calm": True},
        {"calm": False},
        {"high_contrast": True},
        {"high_contrast": False},
        {"captions_stay": True, "text_size": "large", "speech_rate": "slower", "calm": True, "high_contrast": True},
    ],
)
def test_parse_changes_accepts_each_key(body: dict[str, bool | str]) -> None:
    assert parse_changes(body) == body, "only the keys sent, unchanged"


@pytest.mark.parametrize("body", [None, [], ["calm"], [{"calm": True}], "calm", '{"calm": true}', 1, True, {}])
def test_parse_changes_rejects_anything_but_a_non_empty_object(body: object) -> None:
    with pytest.raises(ValueError) as raised:
        parse_changes(body)
    assert str(raised.value) == NOT_AN_OBJECT


@pytest.mark.parametrize("body", [{"foo": True}, {"calm": True, "foo": 1}, {"Calm": True}, {"text-size": "large"}])
def test_parse_changes_rejects_an_unknown_key(body: dict[str, object]) -> None:
    unknown = next(key for key in body if key not in KEYS)
    with pytest.raises(ValueError) as raised:
        parse_changes(body)
    assert str(raised.value) == f"unknown setting: {unknown}"


@pytest.mark.parametrize("key", ["captions_stay", "calm", "high_contrast"])
@pytest.mark.parametrize("value", [1, 0, "true", "false", None, "on"])
def test_parse_changes_needs_a_real_boolean(key: str, value: object) -> None:
    with pytest.raises(ValueError) as raised:
        parse_changes({key: value})
    assert str(raised.value) == f"{key} must be true or false"


@pytest.mark.parametrize("value", ["Large", "LARGE", "x-large", "huge", "", " large", 1.25, 1, None, True])
def test_parse_changes_rejects_a_text_size_it_does_not_know(value: object) -> None:
    with pytest.raises(ValueError) as raised:
        parse_changes({"text_size": value})
    assert str(raised.value) == "text_size must be one of: normal, large, xlarge"


@pytest.mark.parametrize("value", ["Slower", "FASTER", "slow", "fast", "", 0.75, 1, None, False])
def test_parse_changes_rejects_a_speech_rate_it_does_not_know(value: object) -> None:
    with pytest.raises(ValueError) as raised:
        parse_changes({"speech_rate": value})
    assert str(raised.value) == "speech_rate must be one of: slower, normal, faster"


def test_one_bad_value_rejects_the_whole_body() -> None:
    with pytest.raises(ValueError, match="text_size must be one of"):
        parse_changes({"calm": True, "text_size": "huge"})


# ---- the store (one JSON file for the room) --------------------------------------------------------------------------


@pytest.fixture
def store_path(tmp_path: Path) -> Path:
    return tmp_path / "state" / "accessibility.json"


def _tmp_leftovers(path: Path) -> list[str]:
    return [p.name for p in path.parent.iterdir() if p.name != path.name] if path.parent.exists() else []


def test_store_missing_file_gives_the_defaults(store_path: Path) -> None:
    assert AccessibilityStore(store_path).current == AccessibilitySettings()
    assert not store_path.exists(), "loading never writes"


def test_store_saves_and_reloads(store_path: Path) -> None:
    store = AccessibilityStore(store_path)
    saved = store.update({"text_size": "large", "calm": True})
    assert saved == AccessibilitySettings(text_size="large", calm=True)
    assert store.current == saved
    assert json.loads(store_path.read_text()) == {**DEFAULTS, "text_size": "large", "calm": True}
    assert AccessibilityStore(store_path).current == saved, "survives a restart"
    assert _tmp_leftovers(store_path) == [], "the temp file is renamed into place"


def test_store_partial_update_keeps_the_other_keys(store_path: Path) -> None:
    store = AccessibilityStore(store_path)
    store.update({"captions_stay": True, "speech_rate": "slower"})
    after = store.update({"high_contrast": True})
    assert after == AccessibilitySettings(captions_stay=True, speech_rate="slower", high_contrast=True)
    assert after.rate_factor == 0.75
    assert json.loads(store_path.read_text()) == after.to_dict()
    after = store.update({"captions_stay": False})
    assert after == AccessibilitySettings(speech_rate="slower", high_contrast=True), "switching off saves too"
    assert AccessibilityStore(store_path).current == after


@pytest.mark.parametrize(
    "content",
    [b"{not json", b"", b"[]", b'["large"]', b'"large"', b"null", b"42", b'{"text_size": "large"', b"\xff\xfe\x00"],
)
def test_store_broken_file_gives_the_defaults(store_path: Path, content: bytes) -> None:
    store_path.parent.mkdir(parents=True)
    store_path.write_bytes(content)
    assert AccessibilityStore(store_path).current == AccessibilitySettings()
    assert store_path.read_bytes() == content, "loading never rewrites the file"


def test_store_invalid_values_give_that_keys_default_and_keep_the_rest(store_path: Path) -> None:
    store_path.parent.mkdir(parents=True)
    store_path.write_text(
        json.dumps(
            {
                "captions_stay": 1,
                "text_size": "Large",
                "speech_rate": "faster",
                "calm": True,
                "high_contrast": "yes",
                "theme": "dark",
            }
        )
    )
    assert AccessibilityStore(store_path).current == AccessibilitySettings(speech_rate="faster", calm=True)


def test_store_missing_keys_give_their_defaults(store_path: Path) -> None:
    store_path.parent.mkdir(parents=True)
    store_path.write_text('{"high_contrast": true}')
    assert AccessibilityStore(store_path).current == AccessibilitySettings(high_contrast=True)


def test_store_failed_save_keeps_the_new_value_in_memory(store_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    store_path.mkdir(parents=True)  # a folder where the file should be: reading fails and so does the rename
    store = AccessibilityStore(store_path)
    assert store.current == AccessibilitySettings()
    with caplog.at_level("WARNING", logger="biobench_sar.jsonfile"):
        after = store.update({"text_size": "xlarge"})
    assert after == AccessibilitySettings(text_size="xlarge")
    assert store.current == after, "the new value applies until a restart"
    assert f"could not save accessibility settings to {store_path}" in caplog.text
    assert store_path.is_dir() and not any(store_path.iterdir())
    assert _tmp_leftovers(store_path) == [], "the temp file is cleaned up"


def test_store_concurrent_updates_stay_consistent(store_path: Path) -> None:
    store = AccessibilityStore(store_path)
    changes: list[dict[str, bool | str]] = [
        {"captions_stay": True},
        {"text_size": "xlarge"},
        {"speech_rate": "faster"},
        {"calm": True},
        {"high_contrast": True},
    ]

    def worker(change: dict[str, bool | str]) -> None:
        for _ in range(20):
            store.update(change)

    threads = [threading.Thread(target=worker, args=(change,)) for change in changes]
    [t.start() for t in threads]
    [t.join() for t in threads]
    expected = AccessibilitySettings(
        captions_stay=True, text_size="xlarge", speech_rate="faster", calm=True, high_contrast=True
    )
    assert store.current == expected, "no update lost another key"
    assert json.loads(store_path.read_text()) == expected.to_dict()
    assert AccessibilityStore(store_path).current == expected


def test_tests_never_touch_the_real_accessibility_file(isolated_accessibility_file: Path, settings: Settings) -> None:
    assert Settings().accessibility_file == isolated_accessibility_file
    assert settings.accessibility_file == isolated_accessibility_file
