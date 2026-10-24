# Accessibility settings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A teacher sets captions that stay up, text size on the board, voice speed, calm mode and high contrast for
the whole room from an **Accessibility** card under **More tools**; the projector, the voice and every open
dashboard follow at once, and the settings survive a restart.

**Architecture:** `accessibility.py` holds the validated settings and a JSON store (written through the shared
`jsonfile.save_json`). `Assistant` owns the store and serves `GET`/`PUT /api/accessibility`, pushes an
`accessibility` Socket.IO event on every change and on connect, and hands the voice speed to
`TTSEngine.set_rate` (edge `rate`, `say -r`, pyttsx3 `rate`) and to the speaking-window estimate. The projector
keeps the settings as `<body>` data attributes (rendered by Flask, then kept current by the event and a resync
GET) that `style.css` and `main.js` read; the dashboard card PUTs one key per change.

**Tech Stack:** Python 3.10–3.12, Flask-SocketIO, vanilla JS, pytest (+ node fake DOM for the frontend), ruff.

Spec: [../specs/2026-10-24-accessibility-settings-design.md](../specs/2026-10-24-accessibility-settings-design.md).
Branch: `feat/accessibility-settings` from `release/candidate`.

---

## File map

| File | Change |
| --- | --- |
| `biobench_sar/jsonfile.py` | New: `save_json`, the one atomic JSON write |
| `biobench_sar/assistant.py` | `DocumentSelection._save` uses `save_json`; owns `AccessibilityStore`; `accessibility_body`, `set_accessibility`, `push_speech_rate`; `estimate_speech_seconds(text, rate)` |
| `biobench_sar/rag/quiz_bank.py` | `QuizBank._save` uses `save_json` |
| `biobench_sar/accessibility.py` | New: `AccessibilitySettings`, `parse_changes`, `AccessibilityStore`, `TEXT_SIZES`, `SPEECH_RATES` |
| `biobench_sar/config.py` | `Settings.accessibility_file` (`SAR_ACCESSIBILITY_FILE`) |
| `packaging/launcher.py` | The self-test keeps `accessibility_file` in its temp folder |
| `biobench_sar/llm/tts_engine.py` | `edge_rate`, `TTSEngine.set_rate`, rate for edge, `say` and pyttsx3 |
| `biobench_sar/app.py` | `GET`/`PUT /api/accessibility`, on-connect push, `push_speech_rate` on attach, `access` for the pages |
| `biobench_sar/templates/index.html`, `static/js/main.js`, `static/css/style.css` | Projector: body data attributes, held captions, text scale, calm mode, high contrast |
| `biobench_sar/templates/precharge.html`, `static/js/dashboard/accessibility.js` (new), `dashboard/main.js`, `dashboard/camera.js`, `static/css/dashboard.css`, `static/js/i18n.js` | Dashboard card, EN/ES strings, camera toggle scoped to `#camera-view` |
| `tests/test_jsonfile.py`, `tests/test_accessibility.py` (new); `tests/conftest.py`, `test_config.py`, `test_launcher.py`, `test_tts.py`, `test_assistant.py`, `test_app.py`, `test_lan_wiring.py`, `test_frontend.py`, `tests/js/fake_dom.mjs` | Tests, fakes and the default `GET /api/accessibility` route |
| `docs/BUILD_PLAN.md`, `README.md`, `CONTRIBUTING.md`, `docs/MOBILE.md` | Contract M, routes and event, README section and Files row, modules table, phone guide |

Commands: `.venv/bin/python -m pytest -q tests/<file>` · `.venv/bin/ruff check . && .venv/bin/ruff format --check .`

---

### Task 1: DocumentSelection and QuizBank save through a shared atomic `save_json` helper that the accessibility store will reuse.

**Files:** Create `biobench_sar/jsonfile.py` · Modify `biobench_sar/assistant.py`, `biobench_sar/rag/quiz_bank.py` · Test `tests/test_jsonfile.py` (new)

- [ ] **Step 1: Failing tests.** Create the new file `tests/test_jsonfile.py`. All its imports are included below. The existing failure tests patch `os.replace` on the `os` module, so they keep working once the code moves. Those tests are `test_selection_write_is_atomic` (`tests/test_assistant.py:544`), `test_selection_select_and_select_all_persist` with its leftover check (`tests/test_assistant.py:502`) and `test_bank_write_is_atomic` (`tests/test_quiz_bank.py:320`). None of them changes.
```python
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import pytest

from biobench_sar.assistant import DocumentSelection
from biobench_sar.jsonfile import save_json
from biobench_sar.rag.quiz_bank import QuizBank, QuizProblem


def _names(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir())


def _fail_replace(src: Any, dst: Any) -> None:
    raise OSError("disk full")


# ---- save_json --------------------------------------------------------------------------------------------------


def test_save_json_writes_the_payload_and_creates_folders(tmp_path: Path) -> None:
    path = tmp_path / "state" / "deep" / "thing.json"
    payload = {"all": True, "names": ["a.md", "señal.md"], "n": 3}
    assert save_json(path, payload, "thing") is True
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert _names(path.parent) == ["thing.json"], "no temp file left behind"


def test_save_json_replaces_an_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "thing.json"
    path.write_text('{"old": true, "padding": "a longer file than the new one"}', encoding="utf-8")
    assert save_json(path, [1, 2], "thing") is True
    assert json.loads(path.read_text(encoding="utf-8")) == [1, 2]
    assert _names(tmp_path) == ["thing.json"]


def test_save_json_renames_a_temp_file_from_the_same_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "thing.json"
    calls: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def spy(src: Any, dst: Any) -> None:
        calls.append((Path(src), Path(dst)))
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    assert save_json(path, {"v": 1}, "thing") is True
    [(src, dst)] = calls
    assert dst == path and src.parent == path.parent, "same folder, so the rename is atomic"
    assert src.name.startswith(".thing.json.") and src.name.endswith(".tmp"), "a hidden temp file"


def test_save_json_failure_keeps_the_old_file_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    path = tmp_path / "thing.json"
    assert save_json(path, {"v": 1}, "thing") is True
    before = path.read_text(encoding="utf-8")
    monkeypatch.setattr(os, "replace", _fail_replace)
    with caplog.at_level(logging.WARNING, logger="biobench_sar.jsonfile"):
        assert save_json(path, {"v": 2}, "quiz bank") is False
    assert path.read_text(encoding="utf-8") == before, "a failed write never leaves a half-written file"
    assert _names(tmp_path) == ["thing.json"], "the temp file is cleaned up"
    assert [r.getMessage() for r in caplog.records] == [f"could not save quiz bank to {path}: disk full"]


def test_save_json_failure_when_the_folder_cannot_be_made(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    blocker = tmp_path / "state"
    blocker.write_text("a file where the folder should be", encoding="utf-8")
    path = blocker / "thing.json"
    with caplog.at_level(logging.WARNING, logger="biobench_sar.jsonfile"):
        assert save_json(path, {"v": 1}, "document selection") is False
    assert blocker.read_text(encoding="utf-8") == "a file where the folder should be"
    assert _names(tmp_path) == ["state"]
    assert [r.levelno for r in caplog.records] == [logging.WARNING]
    assert caplog.records[0].getMessage().startswith(f"could not save document selection to {path}: ")


def test_selection_and_quiz_bank_save_through_save_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(os, "replace", _fail_replace)
    with caplog.at_level(logging.WARNING, logger="biobench_sar.jsonfile"):
        DocumentSelection(tmp_path / "selection.json").select(["a.md"])
        QuizBank(tmp_path / "quiz_bank.json").replace_source("a.md", [QuizProblem("1 + 1", "2", "", 1, "a.md")])
    assert [r.getMessage() for r in caplog.records if r.name == "biobench_sar.jsonfile"] == [
        f"could not save document selection to {tmp_path / 'selection.json'}: disk full",
        f"could not save quiz bank to {tmp_path / 'quiz_bank.json'}: disk full",
    ]
    assert _names(tmp_path) == [], "neither caller leaves a temp file"
```
- [ ] **Step 2: Run** `.venv/bin/python -m pytest -q tests/test_jsonfile.py` → FAIL. Collection stops with `ModuleNotFoundError: No module named 'biobench_sar.jsonfile'`.
- [ ] **Step 3: Implement**

Create `biobench_sar/jsonfile.py`:
```python
"""One atomic JSON write for the small state files (document selection, quiz bank, accessibility settings)."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import tempfile
from pathlib import Path

log = logging.getLogger(__name__)


def save_json(path: Path, payload: object, what: str) -> bool:
    """Write `payload` to a temp file in `path`'s folder, then rename it into place, so a crash or a full disk never
    leaves half a file. On OSError logs a warning naming `what`, removes the temp file and returns False: the
    caller's in-memory value still applies, it just won't survive a restart."""
    tmp = ""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp, path)
    except OSError as exc:
        log.warning("could not save %s to %s: %s", what, path, exc)
        if tmp:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
        return False
    return True
```

In `biobench_sar/assistant.py`, edit the imports at lines 5–10 and line 20. After this change `contextlib`, `os` and `tempfile` are unused; `json` is still used by `DocumentSelection._load` at line 222.
```python
import contextlib
import json
import logging
import os
import re
import tempfile
import threading
```
→
```python
import json
import logging
import re
import threading
```
and
```python
from biobench_sar.config import Settings
from biobench_sar.rag.quiz_bank import QuizBank, QuizProblem, pick_progressive
```
→
```python
from biobench_sar.config import Settings
from biobench_sar.jsonfile import save_json
from biobench_sar.rag.quiz_bank import QuizBank, QuizProblem, pick_progressive
```
In the same file, edit `DocumentSelection._save` (lines 265–279):
```python
    def _save(self) -> None:
        payload = {"all": True} if self._sources is None else {"sources": sorted(self._sources)}
        tmp = ""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self._path.parent, prefix=f".{self._path.name}.", suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            os.replace(tmp, self._path)
        except OSError as exc:
            # The in-memory selection still applies; it just won't survive a restart.
            log.warning("could not save document selection to %s: %s", self._path, exc)
            if tmp:
                with contextlib.suppress(OSError):
                    os.unlink(tmp)
```
→
```python
    def _save(self) -> None:
        payload = {"all": True} if self._sources is None else {"sources": sorted(self._sources)}
        # A failed save is logged; the in-memory selection still applies, it just won't survive a restart.
        save_json(self._path, payload, "document selection")
```

In `biobench_sar/rag/quiz_bank.py`, edit the imports at lines 5–12 and the logger block at lines 18–20. After this change `contextlib`, `os` and `tempfile` are unused. `json` is still used by `_load` (line 780) and `log` by line 225.
```python
import contextlib
import csv
import json
import logging
import os
import random
import re
import tempfile
import threading
```
→
```python
import csv
import json
import logging
import random
import re
import threading
```
and
```python
from typing import Any

log = logging.getLogger(__name__)
```
→
```python
from typing import Any

from biobench_sar.jsonfile import save_json

log = logging.getLogger(__name__)
```
In the same file, edit `QuizBank._save` (lines 830–843):
```python
    def _save(self) -> None:
        payload = {"sources": {name: [p.to_json() for p in items] for name, items in self._sources.items()}}
        tmp = ""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self._path.parent, prefix=f".{self._path.name}.", suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            os.replace(tmp, self._path)
        except OSError as exc:
            log.warning("could not save quiz bank to %s: %s", self._path, exc)
            if tmp:
                with contextlib.suppress(OSError):
                    os.unlink(tmp)
```
→
```python
    def _save(self) -> None:
        payload = {"sources": {name: [p.to_json() for p in items] for name, items in self._sources.items()}}
        save_json(self._path, payload, "quiz bank")
```
Leave `Calibration.save` (`biobench_sar/core/homography.py:90`) as it is. It raises on failure on purpose.

- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q tests/test_jsonfile.py` → PASS (6 tests). Then run `.venv/bin/python -m pytest -q tests/test_assistant.py tests/test_quiz_bank.py` → PASS. This checks that `test_selection_write_is_atomic` (`tests/test_assistant.py:544`), `test_selection_select_and_select_all_persist` (`:502`) and `test_bank_write_is_atomic` (`tests/test_quiz_bank.py:320`) still pass unchanged. No test asserts the old logger names `biobench_sar.assistant` or `biobench_sar.rag.quiz_bank`, and no test patches `os` or `tempfile` on those modules.
- [ ] **Step 5: Lint** `.venv/bin/ruff check . && .venv/bin/ruff format --check .` → clean. If `F401` flags `os`, `tempfile` or `contextlib`, one of the removals above was missed.
- [ ] **Step 6: Commit** with the message `refactor(storage): one atomic save_json for the selection and quiz bank files`.
```
git add biobench_sar/jsonfile.py biobench_sar/assistant.py biobench_sar/rag/quiz_bank.py tests/test_jsonfile.py
git commit -m "refactor(storage): one atomic save_json for the selection and quiz bank files" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 2: The room's accessibility settings live in one validated model and one thread-safe store. The store is saved to `Settings.accessibility_file`, and tests never touch the developer's `data/`.

**Files:** Create `biobench_sar/accessibility.py` · Modify `biobench_sar/config.py`, `tests/conftest.py`, `packaging/launcher.py` · Test `tests/test_accessibility.py` (new), `tests/test_config.py`, `tests/test_launcher.py`

Depends on Task 1 (`biobench_sar/jsonfile.py` with `save_json` and `log = logging.getLogger(__name__)`; the failed-save test listens on the logger `biobench_sar.jsonfile`).

- [ ] **Step 1: Failing tests**

  **1a.** Create `tests/test_accessibility.py`:
```python
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
```

  **1b.** In `tests/test_config.py`, line 7 gains a module import:
```python
from biobench_sar.config import DATA_DIR, PACKAGE_DIR, Settings
```
→
```python
from biobench_sar import config
from biobench_sar.config import DATA_DIR, PACKAGE_DIR, Settings
```
  Then add these two tests after `test_selection_file_env_override` (lines 275–277) and before the `# ---- LAN mode` banner (line 280). The default+override shape mirrors `test_quiz_bank_file_default_and_override` (lines 259–263):
```python


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
```

  **1c.** `tests/test_launcher.py` needs one more check. `packaging/launcher.py:733-750` `self_test_settings` moves every data file into the self-test's temp folder. Without the new field there, the self-test would read the teacher's real `accessibility.json` once Task 4 builds the store in `Assistant`. Add `dataclasses` to the imports at lines 3–5:
```python
import builtins
import errno
```
→
```python
import builtins
import dataclasses
import errno
```
  Then append this after `test_self_test_gives_up_after_its_timeout`, which ends at line 466 (`assert any("no result after" in line for line in lines)`):
```python


def test_self_test_keeps_every_data_file_in_its_temp_folder(tmp_path: Path) -> None:
    data_dir = tmp_path / "self-test"
    settings = launcher.self_test_settings(data_dir, 5051)
    paths = {f.name: getattr(settings, f.name) for f in dataclasses.fields(settings)}
    files = {name: path for name, path in paths.items() if isinstance(path, Path)}
    assert "accessibility_file" in files
    outside = sorted(name for name, path in files.items() if not path.is_relative_to(data_dir))
    assert outside == [], "the self-test never reads or writes the teacher's own data folder"
```

- [ ] **Step 2: Run** `.venv/bin/python -m pytest -q tests/test_accessibility.py tests/test_config.py tests/test_launcher.py -k "accessibility or self_test_keeps"` → FAIL:
  - `tests/test_accessibility.py` stops at collection with `ModuleNotFoundError: No module named 'biobench_sar.accessibility'`.
  - The `test_config.py` tests fail with `AttributeError: 'Settings' object has no attribute 'accessibility_file'`.
  - The launcher test fails with `AssertionError` on `"accessibility_file" in files`.

- [ ] **Step 3: Implement**

  **3a.** Create `biobench_sar/accessibility.py`:
```python
"""Accessibility settings for the whole room: captions, text size, voice speed, calm mode and high contrast.

The projector, the voice and every open dashboard follow one set of values, saved as JSON so it survives a restart.
"""

from __future__ import annotations

import contextlib
import json
import threading
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

from biobench_sar.jsonfile import save_json

TEXT_SIZES: tuple[str, ...] = ("normal", "large", "xlarge")
# Multiples of the voice's normal speed; TTSEngine.set_rate takes the number.
SPEECH_RATES: dict[str, float] = {"slower": 0.75, "normal": 1.0, "faster": 1.25}


@dataclass(frozen=True)
class AccessibilitySettings:
    """The defaults are the room as it was before these settings: timed captions, normal text and voice."""

    captions_stay: bool = False
    text_size: str = "normal"
    speech_rate: str = "normal"
    calm: bool = False
    high_contrast: bool = False

    def to_dict(self) -> dict[str, bool | str]:
        """All five keys in field order: the API body, the `accessibility` event and the file."""
        return asdict(self)

    @property
    def rate_factor(self) -> float:
        return SPEECH_RATES[self.speech_rate]


_KEYS = tuple(f.name for f in fields(AccessibilitySettings))
_FLAGS = frozenset({"captions_stay", "calm", "high_contrast"})
_CHOICES: dict[str, tuple[str, ...]] = {"text_size": TEXT_SIZES, "speech_rate": tuple(SPEECH_RATES)}


def _check(key: str, value: object) -> bool | str:
    """`value` when `key` allows it; ValueError naming the key and what it allows otherwise."""
    if key in _FLAGS:
        if isinstance(value, bool):  # a JSON true/false, not 1 or 0
            return value
        raise ValueError(f"{key} must be true or false")
    choices = _CHOICES[key]
    if isinstance(value, str) and value in choices:  # exact, like /api/language: "Large" is not "large"
        return value
    raise ValueError(f"{key} must be one of: {', '.join(choices)}")


def parse_changes(body: object) -> dict[str, bool | str]:
    """The checked changes in a PUT /api/accessibility body: a JSON object with any non-empty subset of the keys.

    Anything else raises ValueError with an English message for the 400 reply, and nothing is applied.
    """
    if not isinstance(body, dict) or not body:
        raise ValueError(f"send a JSON object with at least one of: {', '.join(_KEYS)}")
    changes: dict[str, bool | str] = {}
    for key, value in body.items():
        if key not in _KEYS:
            raise ValueError(f"unknown setting: {key}")
        changes[key] = _check(key, value)
    return changes


class AccessibilityStore:
    """The room's settings: loaded once, saved on every change. Thread-safe.

    File format: the to_dict() keys. A missing or unreadable file gives the defaults; a key that is missing from the
    file or holds a value parse_changes would refuse gives that key's default, and the valid keys are kept.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._settings = self._load()

    def _load(self) -> AccessibilitySettings:
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return AccessibilitySettings()
        if not isinstance(data, dict):
            return AccessibilitySettings()
        kept: dict[str, bool | str] = {}
        for key in _KEYS:
            if key in data:
                with contextlib.suppress(ValueError):
                    kept[key] = _check(key, data[key])
        return AccessibilitySettings(**kept)

    @property
    def current(self) -> AccessibilitySettings:
        with self._lock:
            return self._settings

    def update(self, changes: Mapping[str, bool | str]) -> AccessibilitySettings:
        """Apply changes parse_changes already checked, save them and return the new settings."""
        with self._lock:
            self._settings = replace(self._settings, **changes)
            # save_json logs a failure; the new value still applies, it just won't survive a restart.
            save_json(self._path, self._settings.to_dict(), "accessibility settings")
            return self._settings
```

  **3b.** In `biobench_sar/config.py`, add the field after `quiz_bank_file` (lines 123–125):
```python
    quiz_bank_file: Path = field(
        default_factory=lambda: Path(os.environ.get("SAR_QUIZ_BANK_FILE", DATA_DIR / "quiz_bank.json"))
    )
```
→
```python
    quiz_bank_file: Path = field(
        default_factory=lambda: Path(os.environ.get("SAR_QUIZ_BANK_FILE", DATA_DIR / "quiz_bank.json"))
    )
    # The room's accessibility settings (captions, text size, voice speed, calm mode, high contrast).
    accessibility_file: Path = field(
        default_factory=lambda: Path(os.environ.get("SAR_ACCESSIBILITY_FILE", DATA_DIR / "accessibility.json"))
    )
```

  **3c.** In `tests/conftest.py`, add the autouse fixture after `isolated_quiz_bank` (lines 158–164) and before `no_lan_from_the_shell`:
```python
    monkeypatch.delenv("SAR_PROFILE", raising=False)
    return path


@pytest.fixture(autouse=True)
def no_lan_from_the_shell(monkeypatch: pytest.MonkeyPatch) -> None:
```
→
```python
    monkeypatch.delenv("SAR_PROFILE", raising=False)
    return path


@pytest.fixture(autouse=True)
def isolated_accessibility_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """And for the room's accessibility settings: a test that changes them never changes the developer's room."""
    path = tmp_path / "accessibility.json"
    monkeypatch.setenv("SAR_ACCESSIBILITY_FILE", str(path))
    return path


@pytest.fixture(autouse=True)
def no_lan_from_the_shell(monkeypatch: pytest.MonkeyPatch) -> None:
```
  Then in the `settings` fixture (line 183):
```python
        quiz_bank_file=tmp_path / "quiz_bank.json",
        projector_width=640,
```
→
```python
        quiz_bank_file=tmp_path / "quiz_bank.json",
        accessibility_file=tmp_path / "accessibility.json",
        projector_width=640,
```

  **3d.** In `packaging/launcher.py` `self_test_settings` (lines 748–750):
```python
        selection_file=data_dir / "selection.json",
        quiz_bank_file=data_dir / "quiz_bank.json",
    )
```
→
```python
        selection_file=data_dir / "selection.json",
        quiz_bank_file=data_dir / "quiz_bank.json",
        accessibility_file=data_dir / "accessibility.json",
    )
```

- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q tests/test_accessibility.py tests/test_config.py tests/test_launcher.py -k "accessibility or self_test_keeps"` → PASS (90 tests).
  - Also run `.venv/bin/python -m pytest -q tests/test_config.py tests/test_app.py tests/test_assistant.py tests/test_quiz_bank.py` → PASS. These use the `settings` fixture and the autouse env fixtures, which this task changed.
  - Also run `.venv/bin/python -m pytest -q tests/test_launcher.py -k self_test` → PASS.
  - Then `.venv/bin/ruff check . && .venv/bin/ruff format --check .` → clean.
  - No existing exact assertion changes. `test_documented_settings_exist_in_config` (`tests/test_frontend.py:1874`) only requires README `SAR_*` rows to exist in `config.py`, so the README row can wait for Task 7.
  - `tests/test_glare.py::test_fast_at_1080p` is a timing test. It already fails on this machine without any change, so leave it out of this check.

- [ ] **Step 5: Commit** `feat(access): accessibility settings model and store, saved in SAR_ACCESSIBILITY_FILE`
```bash
git add biobench_sar/accessibility.py biobench_sar/config.py packaging/launcher.py tests/conftest.py tests/test_accessibility.py tests/test_config.py tests/test_launcher.py
git commit -m "feat(access): accessibility settings model and store, saved in SAR_ACCESSIBILITY_FILE" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 3: The voice can speak slower or faster (`TTSEngine.set_rate`), and at 1.0 every backend call stays exactly as it is today

**Files:** Modify `biobench_sar/llm/tts_engine.py` · Test `tests/test_tts.py`

- [ ] **Step 1: Failing tests.** First update the fakes in `tests/test_tts.py`, then append the new tests at the end of the file, after `test_pyttsx3_espeak_switches_back_to_english` (lines 553–556). No new imports are needed.

1a. `FakeComm` (lines 22–26) records the keyword arguments it gets, so a test can check that no `rate` is passed at 1.0:
```python
# old
    voices: ClassVar[list[str]] = []

    def __init__(self, text: str, voice: str) -> None:
        self.text, self.voice = text, voice
        FakeComm.voices.append(voice)
```
```python
# new
    voices: ClassVar[list[str]] = []
    options: ClassVar[list[dict[str, str]]] = []

    def __init__(self, text: str, voice: str, **options: str) -> None:
        self.text, self.voice = text, voice
        FakeComm.voices.append(voice)
        FakeComm.options.append(options)
```

1b. Fixture `fake_edge` (line 44):
```python
# old
    FakeComm.voices = []
    mod = types.ModuleType("edge_tts")
```
```python
# new
    FakeComm.voices = []
    FakeComm.options = []
    mod = types.ModuleType("edge_tts")
```

1c. `_Pyttsx3Engine` (lines 490–502) gets a base `rate` property. It logs a rate change as `("set rate", value)`, so the existing `("set", <voice>)` assertions stay byte-identical:
```python
# old
class _Pyttsx3Engine:
    def __init__(self, voices: list[types.SimpleNamespace], default: object, log: list[tuple[str, str]]) -> None:
        self.props: dict[str, object] = {"voice": default, "voices": voices}
        self.log = log
```
```python
# new
class _Pyttsx3Engine:
    def __init__(
        self,
        voices: list[types.SimpleNamespace],
        default: object,
        log: list[tuple[str, str]],
        base_rate: int | None = 200,
    ) -> None:
        self.props: dict[str, object] = {"voice": default, "voices": voices}
        if base_rate is not None:  # pyttsx3's drivers start at 200 words per minute
            self.props["rate"] = base_rate
        self.log = log
```
```python
# old (setProperty, lines 500–502)
        self.props[name] = value
        self.log.append(("set", str(value)))
```
```python
# new
        self.props[name] = value
        self.log.append(("set", str(value)) if name == "voice" else (f"set {name}", str(value)))
```

1d. `_run_pyttsx3` (lines 517–528) takes optional per-line rates and a base rate. Without `rates` it never calls `set_rate`, so the three existing pyttsx3 tests run unchanged:
```python
# old
    lines: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    FakeComm.fail = True
    log: list[tuple[str, str]] = []
    engine = _Pyttsx3Engine(voices, default, log)
    engine.expected = len(lines)
    mod = types.ModuleType("pyttsx3")
    mod.init = lambda: engine  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyttsx3", mod)
    eng = TTSEngine("pyttsx3", out_dir=tmp_path)
    for text, lang in lines:
        eng.speak(text, lang)
```
```python
# new
    lines: list[tuple[str, str]],
    rates: list[float] | None = None,
    base_rate: int | None = 200,
) -> list[tuple[str, str]]:
    FakeComm.fail = True
    log: list[tuple[str, str]] = []
    engine = _Pyttsx3Engine(voices, default, log, base_rate)
    engine.expected = len(lines)
    mod = types.ModuleType("pyttsx3")
    mod.init = lambda: engine  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pyttsx3", mod)
    eng = TTSEngine("pyttsx3", out_dir=tmp_path)
    for index, (text, lang) in enumerate(lines):
        if rates is not None:
            eng.set_rate(rates[index])
        eng.speak(text, lang)
```

1e. Append these at the end of the file:
```python


# ---- voice speed (accessibility) -------------------------------------------------------------------------------------


def test_edge_rate_is_a_signed_percentage() -> None:
    assert tts_engine.edge_rate(1.0) == "+0%"
    assert tts_engine.edge_rate(0.75) == "-25%"
    assert tts_engine.edge_rate(1.25) == "+25%"
    assert tts_engine.edge_rate(0.5) == "-50%"
    assert tts_engine.edge_rate(2.0) == "+100%"
    assert tts_engine.edge_rate(1.1) == "+10%"  # (1.1 - 1) * 100 is 10.000000000000009


def test_edge_gets_a_rate_only_when_it_is_not_normal(tmp_path: Path) -> None:
    eng = TTSEngine("edge", out_dir=tmp_path)
    eng.speak("hello")
    eng.set_rate(0.75)
    eng.speak("slower")
    eng.set_rate(1.25)
    eng.speak("faster")
    eng.set_rate(1.0)
    eng.speak("normal again")
    assert FakeComm.options == [{}, {"rate": "-25%"}, {"rate": "+25%"}, {}]  # no rate argument at all at 1.0


def test_set_rate_is_clamped_and_non_finite_means_normal(tmp_path: Path) -> None:
    assert (tts_engine.MIN_RATE, tts_engine.MAX_RATE) == (0.5, 2.0)
    eng = TTSEngine("edge", out_dir=tmp_path)
    for factor in (0.1, 9, float("nan"), 0.75, float("inf")):
        eng.set_rate(factor)
        eng.speak("x")
    assert FakeComm.options == [{"rate": "-50%"}, {"rate": "+100%"}, {}, {"rate": "-25%"}, {}]


def test_say_gets_a_words_per_minute_rate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tts_engine.subprocess, "run", _Run())
    procs = _no_say(monkeypatch)
    eng = TTSEngine("say", out_dir=tmp_path)
    eng.speak("hi")
    eng.set_rate(0.75)
    eng.speak("hi")
    eng.speak("hola", "es")
    eng.set_rate(1.25)
    eng.speak("hi")
    eng.set_rate(1.0)
    eng.speak("hi")
    eng.speak("hola", "es")
    assert tts_engine.SAY_BASE_WPM == 175
    assert [p.args for p in procs] == [
        ["say", "hi"],
        ["say", "-r", "131", "hi"],  # round(175 * 0.75)
        ["say", "-v", "Paulina", "-r", "131", "hola"],
        ["say", "-r", "219", "hi"],  # round(175 * 1.25)
        ["say", "hi"],
        ["say", "-v", "Paulina", "hola"],
    ]


def test_edge_fallback_to_say_keeps_the_rate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    FakeComm.fail = True
    procs = _no_say(monkeypatch)
    eng = TTSEngine("edge", out_dir=tmp_path)
    eng.set_rate(0.75)
    eng.speak("hi there")
    assert FakeComm.options == [{"rate": "-25%"}]
    assert procs[0].args == ["say", "-r", "131", "hi there"]


def test_pyttsx3_scales_the_engine_rate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    voices = [_voice("default-en", "Samantha", ["en_US"])]
    lines = [("one", "en"), ("two", "en"), ("three", "en"), ("four", "en"), ("five", "en")]
    log = _run_pyttsx3(tmp_path, monkeypatch, voices, "default-en", lines, rates=[1.0, 0.75, 0.75, 1.25, 1.0])
    assert log == [
        ("say", "one"),  # nothing is set at the normal speed
        ("set rate", "150"),  # 200 * 0.75, set once for both slower lines
        ("say", "two"),
        ("say", "three"),
        ("set rate", "250"),
        ("say", "four"),
        ("set rate", "200"),  # back to the engine's own rate
        ("say", "five"),
    ]


def test_pyttsx3_without_a_rate_property_still_speaks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    voices = [_voice("default-en", "Samantha", ["en_US"])]
    log = _run_pyttsx3(tmp_path, monkeypatch, voices, "default-en", [("slow", "en")], rates=[0.75], base_rate=None)
    assert log == [("say", "slow")]
```

- [ ] **Step 2: Run** `.venv/bin/python -m pytest -q tests/test_tts.py -k rate` → FAIL: 7 failed, 33 deselected, because `TTSEngine` has no `set_rate` and the module has no `edge_rate`. Also run `.venv/bin/python -m pytest -q tests/test_tts.py -k "not rate"` → 33 passed. The updated fakes keep every existing test green against the old code.

- [ ] **Step 3: Implement** in `biobench_sar/llm/tts_engine.py`.

3a. Imports (lines 5–6):
```python
# old
import logging
import queue
```
```python
# new
import logging
import math
import queue
```

3b. Constants after line 24:
```python
# old
SAY_LIST_TIMEOUT_S = 5.0
```
```python
# new
SAY_LIST_TIMEOUT_S = 5.0
# Voice speed (accessibility): 1.0 is each voice's own pace; set_rate clamps to this range.
MIN_RATE = 0.5
MAX_RATE = 2.0
# macOS `say` speaks about 175 words per minute unless told otherwise; its `-r` takes an absolute rate.
SAY_BASE_WPM = 175
```

3c. A new function right before `def parse_say_voices` (line 68):
```python
def edge_rate(factor: float) -> str:
    """edge-tts `rate` for a speed factor: a signed percentage, e.g. 0.75 -> "-25%", 1.25 -> "+25%"."""
    return f"{round((factor - 1) * 100):+d}%"


def parse_say_voices(output: str) -> list[tuple[str, str]]:
```

3d. The queue item type (line 193):
```python
# old
        self._q: queue.Queue[tuple[str, str] | None] | None = None
```
```python
# new
        self._q: queue.Queue[tuple[str, str, float] | None] | None = None
```

3e. The rate field and `set_rate` (lines 199–204):
```python
# old
        self._say_voices_lock = threading.Lock()

    @property
    def default_language(self) -> str:
        """The language of a speak() call that names none: the legacy voice's, else English."""
        return self._voice_lang if self._voice_lang in LANGUAGES else "en"
```
```python
# new
        self._say_voices_lock = threading.Lock()
        # Voice speed factor (set_rate); at 1.0 no backend gets anything extra.
        self._rate = 1.0

    @property
    def default_language(self) -> str:
        """The language of a speak() call that names none: the legacy voice's, else English."""
        return self._voice_lang if self._voice_lang in LANGUAGES else "en"

    def set_rate(self, factor: float) -> None:
        """Voice speed for every later speak(): 0.75 slower, 1.25 faster, clamped to MIN_RATE..MAX_RATE."""
        rate = float(factor)
        self._rate = min(MAX_RATE, max(MIN_RATE, rate)) if math.isfinite(rate) else 1.0
```

3f. `_edge` (lines 278–281). `Communicate` stays inside the `try`, because it validates its arguments when it is built and a failure there must still start the cooldown:
```python
# old
        path = self.out_dir / name
        # Offline venues: fail fast so the say/pyttsx3 fallback speaks instead of hanging the pipeline.
        try:
            asyncio.run(asyncio.wait_for(edge_tts.Communicate(text, voice).save(str(path)), EDGE_TIMEOUT_S))
```
```python
# new
        path = self.out_dir / name
        rate = self._rate
        # Offline venues: fail fast so the say/pyttsx3 fallback speaks instead of hanging the pipeline.
        try:
            # Inside the try: Communicate validates the voice and the rate as it is built.
            if rate == 1.0:
                comm = edge_tts.Communicate(text, voice)
            else:
                comm = edge_tts.Communicate(text, voice, rate=edge_rate(rate))
            asyncio.run(asyncio.wait_for(comm.save(str(path)), EDGE_TIMEOUT_S))
```

3g. `_say` (lines 299–300):
```python
# old
        voice = self.say_voice(lang)
        args = ["say", "-v", voice, text] if voice else ["say", text]
```
```python
# new
        voice = self.say_voice(lang)
        rate = self._rate
        args = ["say", "-v", voice] if voice else ["say"]
        if rate != 1.0:
            args += ["-r", str(round(SAY_BASE_WPM * rate))]
        args.append(text)
```

3h. `_pyttsx3` (line 319). This is the only `put` of a speech item; `stop()` (line 244) still puts `None`:
```python
# old
        q.put((text, lang))
```
```python
# new
        q.put((text, lang, self._rate))
```

3i. `_pyttsx3_worker` (lines 332–339). This is the only `get`:
```python
# old
        default = _property(engine, "voice")
        chosen: dict[str, Any] = {}
        current = default
        while True:
            item = q.get()
            if item is None:
                return
            text, lang = item
```
```python
# new
        default = _property(engine, "voice")
        # The engine's own words per minute, read once: a voice speed scales it and 1.0 puts it back.
        base_rate = _property(engine, "rate")
        chosen: dict[str, Any] = {}
        current = default
        current_rate = 1.0
        while True:
            item = q.get()
            if item is None:
                return
            text, lang, rate = item
```

3j. The rate switch, between the voice switch and the speaking (lines 348–351). It follows the same set-only-on-change pattern as the voice. The value is `round(base × factor)` and not the raw float: pyttsx3's espeak driver passes `rate` to a C `int` (`_espeak.SetParameter`, `c_int`), so a float there raises `ctypes.ArgumentError`. nsss and sapi5 accept an int as well.
```python
# old
            except Exception:
                log.exception("pyttsx3 voice switch failed")
            try:
                engine.say(text)
```
```python
# new
            except Exception:
                log.exception("pyttsx3 voice switch failed")
            if rate != current_rate and isinstance(base_rate, (int, float)):
                try:
                    # Whole words per minute: the espeak driver hands the rate to C as an int.
                    engine.setProperty("rate", base_rate if rate == 1.0 else round(base_rate * rate))
                    current_rate = rate
                except Exception:
                    log.exception("pyttsx3 rate change failed")
            try:
                engine.say(text)
```

- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q tests/test_tts.py` → PASS (40 passed). Then run `.venv/bin/ruff check . && .venv/bin/ruff format --check .` → clean.
  - No existing exact assertion changes. Every existing `["say", ...]` args list (lines 75, 151, 253, 352–355, 367, 379, 389, 403, 418) and every pyttsx3 log (lines 537–543, 550, 556) runs at rate 1.0 and stays byte-identical.
  - No other test file uses `FakeComm`, `edge_tts`, `_Pyttsx3Engine` or the TTS queue.

- [ ] **Step 5: Commit** `feat(tts): speak slower or faster with set_rate on edge, say and pyttsx3`
```bash
git add biobench_sar/llm/tts_engine.py tests/test_tts.py
git commit -m "feat(tts): speak slower or faster with set_rate on edge, say and pyttsx3

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 4: The server owns the room's accessibility settings. A phone or laptop reads and changes them through `GET`/`PUT /api/accessibility`, every open page gets the full settings on connect and on each change, the projector template gets them for its first paint, and the voice speed reaches the TTS and the speaking-window estimate.

**Files:** Modify `biobench_sar/assistant.py` · Modify `biobench_sar/app.py` · Modify `tests/conftest.py` · Modify `tests/js/fake_dom.mjs` · Test `tests/test_assistant.py`, `tests/test_app.py`, `tests/test_lan_wiring.py`

Line numbers are from `release/candidate`, before T1–T3. T1 shortens `DocumentSelection._save`, so `assistant.py` lines after about 265 move up a few. Every edit below is anchored on text that is unique in its file, not on line numbers. Depends on T2: `biobench_sar/accessibility.py`, `Settings.accessibility_file`, and the `isolated_accessibility_file` autouse fixture plus `settings.accessibility_file` in `tests/conftest.py`.

Fakes:
- `tests/conftest.py` `FakeTTS` gains a recording `set_rate`, so the app tests can see the push on attach and on every change.
- The `TTS` fake in `tests/test_assistant.py` (lines 80–90) stays **without** `set_rate`. It shows that the `getattr` + `callable` guard keeps working. A `RatedTTS(TTS)` subclass in the new section records rates.

- [ ] **Step 1: Failing tests**

(a) `tests/conftest.py`: give `FakeTTS` (lines 141–147) a recording `set_rate`.

old:
```python
class FakeTTS:
    def __init__(self) -> None:
        self.spoken: list[tuple[str, str | None]] = []

    def speak(self, text: str, lang: str | None = None) -> str | None:
        self.spoken.append((text, lang))
        return "fake.mp3"
```
new:
```python
class FakeTTS:
    def __init__(self) -> None:
        self.spoken: list[tuple[str, str | None]] = []
        self.rates: list[float] = []  # every set_rate call: the voice speed from the accessibility settings

    def speak(self, text: str, lang: str | None = None) -> str | None:
        self.spoken.append((text, lang))
        return "fake.mp3"

    def set_rate(self, factor: float) -> None:
        self.rates.append(factor)
```

(b) `tests/js/fake_dom.mjs` `DEFAULT_ROUTES` (line 195). No current JS calls this route; T5 and T6 rely on it.

old:
```js
  "GET /api/language": { status: 200, body: { setting: "auto", current: "en" } },
```
new:
```js
  "GET /api/language": { status: 200, body: { setting: "auto", current: "en" } },
  "GET /api/accessibility": {
    status: 200,
    body: { captions_stay: false, text_size: "normal", speech_rate: "normal", calm: false, high_contrast: false },
  },
```

(c) `tests/test_assistant.py`: append after `test_set_language_works_without_an_audio_service` (lines 1101–1102), before `# ---- display text and spoken text ----`. No new imports: `json`, `Any`, `pytest`, `Settings`, `Assistant` and `estimate_speech_seconds` are already imported (lines 4, 12, 15, 18, 19).
```python


# ---- accessibility ----------------------------------------------------------------------------------------------

ACCESS_DEFAULTS = {
    "captions_stay": False,
    "text_size": "normal",
    "speech_rate": "normal",
    "calm": False,
    "high_contrast": False,
}


class RatedTTS(TTS):
    """A voice with TTSEngine.set_rate. The plain TTS fake above has none, like a voice that can't change speed."""

    def __init__(self) -> None:
        super().__init__()
        self.rates: list[float] = []

    def set_rate(self, factor: float) -> None:
        self.rates.append(factor)


def _rated(h: Harness) -> RatedTTS:
    tts = RatedTTS()
    h.tts = tts
    h.assistant.services.tts = tts
    return tts


def test_estimate_speech_seconds_scales_with_the_voice_speed() -> None:
    text = "one two three four five"
    assert estimate_speech_seconds(text) == estimate_speech_seconds(text, 1.0) == pytest.approx(5 / 2.5 + 1.0)
    assert estimate_speech_seconds(text, 0.75) == pytest.approx(5 / (2.5 * 0.75) + 1.0)
    assert estimate_speech_seconds(text, 1.25) == pytest.approx(5 / (2.5 * 1.25) + 1.0)
    for broken in (0.0, -1.0):
        assert estimate_speech_seconds(text, broken) == estimate_speech_seconds(text), broken
    assert estimate_speech_seconds("", 0.75) == 1.0


def test_a_slower_speech_rate_keeps_the_microphone_closed_longer(h: Harness, clock: FakeClock) -> None:
    start = clock.now
    h.assistant.set_accessibility({"speech_rate": "slower"})
    h.assistant.handle_query("q")  # "one two three four" at 0.75x: 4 / 1.875 + 1 = 3.13 s instead of 2.6 s
    window = estimate_speech_seconds("one two three four", 0.75)
    assert window == pytest.approx(4 / 1.875 + 1.0)
    clock.now = start + estimate_speech_seconds("one two three four") + 0.01
    assert h.assistant.should_listen() is False, "a normal-speed answer would be over; the slower one is not"
    clock.now = start + window + 0.01
    assert h.assistant.should_listen() is True


def test_set_accessibility_saves_sets_the_voice_speed_and_emits_every_key(h: Harness) -> None:
    tts = _rated(h)
    expected = {**ACCESS_DEFAULTS, "captions_stay": True, "speech_rate": "slower"}
    assert h.assistant.set_accessibility({"captions_stay": True, "speech_rate": "slower"}) == expected
    assert h.payloads("accessibility") == [expected], "one push, with every key, for every open page"
    assert tts.rates == [0.75]
    assert h.assistant.accessibility_body() == expected
    assert json.loads(Settings().accessibility_file.read_text(encoding="utf-8")) == expected
    later = h.assistant.set_accessibility({"text_size": "large"})
    assert later == {**expected, "text_size": "large"}, "a partial change keeps the other settings"
    assert tts.rates == [0.75, 0.75] and len(h.payloads("accessibility")) == 2


@pytest.mark.parametrize(
    "body",
    [None, ["calm"], {}, {"font": "big"}, {"calm": 1}, {"speech_rate": "Slower"}, {"calm": True, "text_size": "huge"}],
)
def test_set_accessibility_rejects_bad_changes_without_saving_or_pushing(h: Harness, body: Any) -> None:
    tts = _rated(h)
    with pytest.raises(ValueError):
        h.assistant.set_accessibility(body)
    assert h.payloads("accessibility") == [] and tts.rates == []
    assert h.assistant.accessibility_body() == ACCESS_DEFAULTS, "not even the valid key of a bad body"
    assert not Settings().accessibility_file.exists()


def test_set_accessibility_works_with_a_voice_without_set_rate(h: Harness) -> None:
    assert not hasattr(h.tts, "set_rate"), "the plain fake stands for a voice that can't change speed"
    assert h.assistant.set_accessibility({"speech_rate": "faster"})["speech_rate"] == "faster"
    assert h.payloads("accessibility")[-1]["speech_rate"] == "faster"
    h.assistant.handle_query("q")
    assert len(h.tts.langs) == 1, "and it still speaks"


def test_set_accessibility_survives_a_voice_that_cannot_change_speed(h: Harness) -> None:
    class BrokenRate(TTS):
        def set_rate(self, factor: float) -> None:
            raise RuntimeError("voice busy")

    h.assistant.services.tts = BrokenRate()
    assert h.assistant.set_accessibility({"speech_rate": "slower"})["speech_rate"] == "slower"
    assert h.payloads("accessibility")[-1]["speech_rate"] == "slower", "every page still hears it"


def test_push_speech_rate_gives_the_voice_the_saved_speed() -> None:
    Settings().accessibility_file.write_text(json.dumps({"speech_rate": "faster", "calm": True}), encoding="utf-8")
    fresh = Harness()
    assert fresh.assistant.accessibility_body() == {**ACCESS_DEFAULTS, "speech_rate": "faster", "calm": True}
    tts = _rated(fresh)
    fresh.assistant.push_speech_rate()
    assert tts.rates == [1.25]


def test_push_speech_rate_before_the_services_are_attached_is_harmless() -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    assistant = Assistant(Settings(), lambda name, data: events.append((name, data)), lambda fn, *args: fn(*args))
    assistant.push_speech_rate()  # services stays None until create_app attaches them
    assert assistant.set_accessibility({"calm": True})["calm"] is True
    assert [name for name, _ in events] == ["accessibility"]
```

(d) `tests/test_app.py`: append after `test_answers_carry_display_and_spoken_text`, whose last line is line 1162 (`assert (history["display_text"], history["language"]) == ("x + 2 = 5", "en")`), before `# ---- glare ----`. No new imports: `json`, `Any`, `Path`, `pytest`, `Settings`, `app_module`, `EVIL` (line 419) and `collect` (line 36) already exist. `flask.template_rendered` is imported inside the test, like the local import in `test_answers_carry_display_and_spoken_text`. It reads the render context directly, so the stub templates in conftest stay as they are.
```python


# ---- accessibility ----------------------------------------------------------------------------------------------

ACCESS = "/api/accessibility"
ACCESS_DEFAULTS = {
    "captions_stay": False,
    "text_size": "normal",
    "speech_rate": "normal",
    "calm": False,
    "high_contrast": False,
}


def _access_pushes(sio: Any, wait: bool = True) -> list[dict[str, Any]]:
    got = collect(sio, ["accessibility"], timeout=1.0) if wait else sio.get_received()
    return [m["args"][0] for m in got if m["name"] == "accessibility"]


def test_accessibility_starts_at_the_defaults_and_the_voice_gets_its_speed(built: Any, fakes: Any) -> None:
    app, _ = built
    assert app.test_client().get(ACCESS).get_json() == ACCESS_DEFAULTS
    assert fakes.tts.rates == [1.0], "create_app hands the voice its speed when it attaches the services"


def test_accessibility_partial_changes_round_trip_and_are_saved(built: Any, fakes: Any, settings: Settings) -> None:
    app, _ = built
    client = app.test_client()
    r = client.put(ACCESS, json={"speech_rate": "slower"})
    assert r.status_code == 200 and r.get_json() == {**ACCESS_DEFAULTS, "speech_rate": "slower"}
    r = client.put(ACCESS, json={"text_size": "xlarge", "high_contrast": True})
    expected = {**ACCESS_DEFAULTS, "speech_rate": "slower", "text_size": "xlarge", "high_contrast": True}
    assert r.status_code == 200 and r.get_json() == expected, "a partial body keeps the other settings"
    assert client.get(ACCESS).get_json() == expected
    assert json.loads(settings.accessibility_file.read_text(encoding="utf-8")) == expected
    assert fakes.tts.rates == [1.0, 0.75, 0.75], "the voice follows every change"


def test_accessibility_survives_restart(settings: Settings, fakes: app_module.Services, templates_dir: Path) -> None:
    first, _ = app_module.create_app(settings, fakes, template_folder=str(templates_dir))
    first.test_client().put(ACCESS, json={"calm": True, "speech_rate": "faster"})
    fakes.tts.rates.clear()
    second, _ = app_module.create_app(settings, fakes, template_folder=str(templates_dir))
    assert second.test_client().get(ACCESS).get_json() == {**ACCESS_DEFAULTS, "calm": True, "speech_rate": "faster"}
    assert fakes.tts.rates == [1.25], "the restarted voice speaks at the saved speed from the first answer"


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        ("{}", "JSON object"),
        ('["calm"]', "JSON object"),
        ("null", "JSON object"),
        ("not json", "JSON object"),
        ('{"font": "big"}', "unknown setting: font"),
        ('{"captions_stay": 1}', "captions_stay must be true or false"),
        ('{"speech_rate": "Slower"}', "speech_rate must be one of: slower, normal, faster"),
        ('{"calm": true, "text_size": "huge"}', "text_size must be one of: normal, large, xlarge"),
    ],
)
def test_bad_accessibility_body_is_400_and_changes_nothing(
    built: Any, fakes: Any, settings: Settings, payload: str, error: str
) -> None:
    app, socketio = built
    sio = socketio.test_client(app)
    sio.get_received()
    client = app.test_client()
    r = client.put(ACCESS, data=payload, content_type="application/json")
    assert r.status_code == 400 and error in r.get_json()["error"]
    assert not settings.accessibility_file.exists(), "nothing saved"
    assert _access_pushes(sio, wait=False) == [], "nothing pushed"
    assert fakes.tts.rates == [1.0], "only the speed handed over at startup"
    assert client.get(ACCESS).get_json() == ACCESS_DEFAULTS


def test_cross_origin_accessibility_change_blocked(built: Any, settings: Settings) -> None:
    r = built[0].test_client().put(ACCESS, json={"high_contrast": True}, headers=EVIL)
    assert r.status_code == 403 and not settings.accessibility_file.exists()


def test_every_open_page_hears_an_accessibility_change(built: Any) -> None:
    app, socketio = built
    projector, dashboard = socketio.test_client(app), socketio.test_client(app)
    for sio in (projector, dashboard):
        sio.get_received()
    body = app.test_client().put(ACCESS, json={"captions_stay": True}).get_json()
    assert body == {**ACCESS_DEFAULTS, "captions_stay": True}
    for sio in (projector, dashboard):
        assert _access_pushes(sio) == [body], "one push, with every key"


def test_a_page_that_connects_later_gets_the_saved_accessibility(built: Any) -> None:
    app, socketio = built
    app.test_client().put(ACCESS, json={"text_size": "large"})
    got = socketio.test_client(app).get_received()
    assert got[-1]["name"] == "accessibility"
    assert got[-1]["args"][0] == {**ACCESS_DEFAULTS, "text_size": "large"}


def test_pages_render_with_the_accessibility_settings(built: Any) -> None:
    from flask import template_rendered

    app, _ = built
    client = app.test_client()
    client.put(ACCESS, json={"text_size": "xlarge", "calm": True})
    seen: list[tuple[str | None, Any]] = []

    def record(_sender: Any, template: Any, context: dict[str, Any], **_extra: Any) -> None:
        seen.append((template.name, context.get("access")))

    with template_rendered.connected_to(record, app):
        assert client.get("/").status_code == 200
        assert client.get("/dashboard").status_code == 200
    expected = {**ACCESS_DEFAULTS, "text_size": "xlarge", "calm": True}
    assert seen == [("index.html", expected), ("precharge.html", expected)], "the first paint is already right"
```

(e) `tests/test_app.py`: existing tests whose exact assertions the new on-connect emit breaks.

`test_socket_connect_receives_current_status`, line 528:
```python
    assert [m["name"] for m in got] == ["status", "tutor", "language"]
```
→
```python
    assert [m["name"] for m in got] == ["status", "tutor", "language", "accessibility"]
```
Line 531:
```python
    assert got[2]["args"][0] == {"setting": "auto", "current": "en"}
```
→
```python
    assert got[2]["args"][0] == {"setting": "auto", "current": "en"}
    assert got[3]["args"][0] == ACCESS_DEFAULTS, "a reloaded projector gets its text size, captions and contrast"
```
`ACCESS_DEFAULTS` is a module-level constant defined further down the file. Python resolves it when the test runs, so this works.

`test_socket_connect_packet_precedes_status_push`: it must now wait for the fourth push. Otherwise it can stop after `language` and count only three `42`s. Line 555:
```python
            any(p.startswith("40") for p in packets) and any(p.startswith('42["language"') for p in packets)
```
→
```python
            any(p.startswith("40") for p in packets) and any(p.startswith('42["accessibility"') for p in packets)
```
Line 562:
```python
    assert [p[:2] for p in messages] == ["40", "42", "42", "42"], packets
```
→
```python
    assert [p[:2] for p in messages] == ["40", "42", "42", "42", "42"], packets
```
Line 564:
```python
    assert json.loads(messages[2][2:])[0] == "tutor"
```
→
```python
    assert json.loads(messages[2][2:])[0] == "tutor"
    assert json.loads(messages[4][2:])[0] == "accessibility"
```

(f) `tests/test_lan_wiring.py`: append after `test_a_phone_without_the_cookie_cannot_open_a_socket`, which ends at line 163 with `assert len(flask_app.config["SAR_PHONES"]) == 0`, before `# ---- joining ----`. No new imports.
```python


def test_a_phone_needs_the_pin_to_change_accessibility(lan_app: tuple[Flask, SocketIO]) -> None:
    flask_app, _ = lan_app
    client = phone(flask_app)
    assert client.get("/api/accessibility").status_code == 401
    refused = client.put("/api/accessibility", json={"high_contrast": True})
    assert refused.status_code == 401 and refused.get_json() == {"error": "join with the PIN first"}
    assert not flask_app.config["SAR_ASSISTANT"].settings.accessibility_file.exists(), "nothing saved"
    join(client, gate_of(flask_app).pin)
    changed = client.put("/api/accessibility", json={"high_contrast": True})
    assert changed.status_code == 200 and changed.get_json()["high_contrast"] is True, "a joined phone may change it"
```

`test_every_api_response_is_no_store`, line 448. On one line the tuple would be over 120 columns. Ruff format keeps this multi-line form because of the trailing comma.
```python
    for path in ("/api/status", "/api/lan", "/api/config", "/api/tutor", "/api/history", "/api/nope"):
```
→
```python
    for path in (
        "/api/status",
        "/api/lan",
        "/api/config",
        "/api/tutor",
        "/api/history",
        "/api/accessibility",
        "/api/nope",
    ):
```

- [ ] **Step 2: Run** `.venv/bin/python -m pytest -q tests/test_assistant.py tests/test_app.py tests/test_lan_wiring.py -k "accessibility or speech or socket_connect or no_store"` → FAIL. The expected failures:
  - `TypeError: estimate_speech_seconds() takes 1 positional argument but 2 were given`
  - `AttributeError: 'Assistant' object has no attribute 'set_accessibility'` / `'push_speech_rate'`
  - 404 on `/api/accessibility`, including for the joined phone
  - `fakes.tts.rates == []`, not `[1.0]`
  - the connect list has no `"accessibility"`
  - the raw-polling test times out after 5 s

  31 failed and 5 passed: the no-store tests already pass, because the gate and the 404 are both `no-store`. This run takes about 2 minutes because of the `collect` and polling timeouts.

- [ ] **Step 3: Implement**

`biobench_sar/assistant.py`. Import, before line 20. The anchor is this line alone, since T1 may add `from biobench_sar.jsonfile import ...` after it:
```python
from biobench_sar.config import Settings
```
→
```python
from biobench_sar.accessibility import AccessibilityStore, parse_changes
from biobench_sar.config import Settings
```

`estimate_speech_seconds` (lines 181–182). The default stays 2.6 s for four words, as `test_should_listen_false_during_estimated_speaking_window` expects:
```python
def estimate_speech_seconds(text: str) -> float:
    return len(text.split()) / WORDS_PER_SECOND + 1.0
```
→
```python
def estimate_speech_seconds(text: str, rate: float = 1.0) -> float:
    """Seconds the voice needs for `text` at `rate` x its normal speed, plus one; a rate that isn't positive is 1."""
    rate = rate if rate > 0 else 1.0
    return len(text.split()) / (WORDS_PER_SECOND * rate) + 1.0
```

`Assistant.__init__` (line 301):
```python
        self.quiz_bank = QuizBank(settings.quiz_bank_file)
```
→
```python
        self.quiz_bank = QuizBank(settings.quiz_bank_file)
        # Captions, text size, voice speed, calm mode, high contrast: one set for the whole room, kept as JSON.
        self.accessibility = AccessibilityStore(settings.accessibility_file)
```

New section between the end of `__init__` and the language section (lines 310–312). `push_speech_rate` uses the same `getattr` + `callable` + `try/log.exception` guard as `set_language` uses for audio (lines 336–342):
```python
        self._typed = threading.local()

    # ---- language ----------------------------------------------------------
```
→
```python
        self._typed = threading.local()

    # ---- accessibility -----------------------------------------------------
    def accessibility_body(self) -> dict[str, bool | str]:
        """The room's five accessibility settings, every key."""
        return self.accessibility.current.to_dict()

    def set_accessibility(self, body: object) -> dict[str, bool | str]:
        """Apply a partial change ({"speech_rate": "slower"}, ...); ValueError, with nothing saved or pushed, for
        anything else. The voice takes the new speed and every open page gets every key."""
        saved = self.accessibility.update(parse_changes(body)).to_dict()
        self.push_speech_rate()
        self._emit("accessibility", saved)
        return saved

    def push_speech_rate(self) -> None:
        """Hand the saved voice speed to the TTS service; one without set_rate (older fakes) keeps its own."""
        tts = getattr(self.services, "tts", None)
        set_rate = getattr(tts, "set_rate", None)
        if callable(set_rate):
            try:
                set_rate(self.accessibility.current.rate_factor)
            except Exception:
                log.exception("could not set the voice speed")

    # ---- language ----------------------------------------------------------
```

`_run` (line 563):
```python
            self._speaking_until = time.monotonic() + estimate_speech_seconds(action.spoken_text)
```
→
```python
            # A slower voice talks for longer: the wake-word loop stays deaf until it is done.
            rate = self.accessibility.current.rate_factor
            self._speaking_until = time.monotonic() + estimate_speech_seconds(action.spoken_text, rate)
```

`biobench_sar/app.py`. Attach (line 171):
```python
    assistant.services = svc
```
→
```python
    assistant.services = svc
    assistant.push_speech_rate()  # the voice starts at the room's saved speed
```

`page()` (lines 275–281). `index.html` uses `access`; `precharge.html` ignores it. `join.html` goes through `join_page()` and is unchanged.
```python
            lan_enabled=lan_on,
        )
```
→
```python
            lan_enabled=lan_on,
            # The projector's first paint already has the room's text size, captions, calm mode and contrast.
            access=assistant.accessibility_body(),
        )
```

Routes, right after `api_set_language` (lines 536–542). They sit behind the existing LAN PIN gate, with no `only_the_laptop()`:
```python
        return jsonify(assistant.set_language(setting))

    @app.get("/api/history")
```
→
```python
        return jsonify(assistant.set_language(setting))

    # Joined phones may change these too (no only_the_laptop): the teacher walks the room with one.
    @app.get("/api/accessibility")
    def api_accessibility() -> Any:
        return jsonify(assistant.accessibility_body())

    @app.put("/api/accessibility")
    def api_set_accessibility() -> Any:
        body = request.get_json(silent=True)
        try:
            saved = assistant.set_accessibility(body)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify(saved)

    @app.get("/api/history")
```

`on_connect` (line 736):
```python
        socketio.emit("language", assistant.language_body(), to=request.sid)
```
→
```python
        socketio.emit("language", assistant.language_body(), to=request.sid)
        socketio.emit("accessibility", assistant.accessibility_body(), to=request.sid)
```

- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q tests/test_assistant.py tests/test_app.py tests/test_lan_wiring.py -k "accessibility or speech or socket_connect or no_store"` → PASS (36 passed).
  - Also run the full files whose exact assertions or fakes this touches: `.venv/bin/python -m pytest -q tests/test_assistant.py tests/test_app.py tests/test_lan_wiring.py tests/test_startup.py tests/test_answer_ingestion.py tests/test_frontend.py tests/test_config.py` → PASS. These are the on-connect lists, `FakeTTS`, `create_app` users and `DEFAULT_ROUTES` under node.
  - Then `.venv/bin/ruff check . && .venv/bin/ruff format --check .` → clean.
  - I checked all of this on a scratch copy with stand-in T1/T2 modules: 596 passed, and ruff was clean.

- [ ] **Step 5: Commit** `feat(server): accessibility settings API, socket push and voice speed`
```bash
git add biobench_sar/assistant.py biobench_sar/app.py tests/conftest.py tests/js/fake_dom.mjs tests/test_assistant.py tests/test_app.py tests/test_lan_wiring.py
git commit -m "feat(server): accessibility settings API, socket push and voice speed

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 5: The projector page follows the room's accessibility settings: text size, captions that stay up, calm mode and high contrast. Flask paints them on first load, and the `accessibility` event and the resync GET keep them current.

**Files:** Modify `biobench_sar/templates/index.html` · Modify `biobench_sar/static/js/main.js` · Modify `biobench_sar/static/css/style.css` · Test `tests/test_frontend.py`

Depends on T2 (`AccessibilitySettings`) and T4 (`app.py page()` passes `access=`, and `DEFAULT_ROUTES` has `GET /api/accessibility`). These notes on the fake DOM explain how the tests are written:
- `template_nodes` skips `<body>` (test_frontend.py:56). So in the `PROJECTOR` prelude (line 322) the body dataset holds only `projectorWidth` and `projectorHeight`. `captionsStay()` is false and the outline border is 3 px, which keeps every existing projector test unchanged.
- `installDom(..., { bodyData })` (fake_dom.mjs:144-146) is how `PROJECTOR_ACCESS` reproduces Flask's first paint.
- The fake `Audio` (fake_dom.mjs:177) never fires `ended`, so the captions test installs its own.
- `runTimer(ms)` (test_frontend.py:1897) runs a timer even if it was cleared. Because of that, `hideBubbleUnlessSpeaking` checks `captionsStay()` itself.

- [ ] **Step 1: Failing tests** (add one import, then append to `tests/test_frontend.py` after `test_dashboard_steps_fit_one_laptop_screen`, the last test in the file)

Import. Replace line 19:
```python
from biobench_sar.config import Settings
```
with:
```python
from biobench_sar.accessibility import AccessibilitySettings
from biobench_sar.config import Settings
```

Append at the end of the file:
```python


# ---- accessibility: the projector follows the room's settings -------------------------------------------------------

# The server's body for a room that changed nothing, as GET /api/accessibility and the "accessibility" event send it.
ACCESS = json.dumps(AccessibilitySettings().to_dict())
# A projector page whose <body> Flask rendered with every setting changed (index.html's first paint).
PROJECTOR_ACCESS = PROJECTOR_TIMED.replace(
    'projectorHeight: "1080" }',
    'projectorHeight: "1080", textSize: "large", captions: "stay", calm: "on", contrast: "on" }',
)
assert PROJECTOR_ACCESS != PROJECTOR_TIMED
LOOKS = """
const looks = () => {
  const { textSize, captions, calm, contrast } = doc.body.dataset;
  return { textSize, captions, calm, contrast };
};
"""


def style_rules(prefix: str) -> dict[str, str]:
    """style.css rules whose selector list starts with `prefix`, comments left out: {selector: declarations}.

    A selector written twice keeps its first rule; the later one is an override (the reduced-motion block's).
    """
    css = re.sub(r"/\*.*?\*/", "", (STATIC / "css" / "style.css").read_text(encoding="utf-8"), flags=re.S)
    rules: dict[str, str] = {}
    for selector, body in re.findall(rf"(?:(?<=\}})|^)\s*({re.escape(prefix)}[^{{}}]*)\{{([^{{}}]*)\}}", css):
        rules.setdefault(selector.strip(), body)
    return rules


@needs_node
def test_projector_body_follows_the_accessibility_settings(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "index.html",
        PROJECTOR
        + f"const ACCESS = {ACCESS};\n"
        + LOOKS
        + """
installFetch({ "GET /api/accessibility": { status: 200, body: { ...ACCESS, captions_stay: true, text_size: "large", calm: true } } });
sock.fire("connect");
await settle();
// A reconnect asks the server, so a change made while the page was away still shows.
assert.deepEqual(looks(), { textSize: "large", captions: "stay", calm: "on", contrast: "off" });
const sent = sock.emitted.length;
sock.fire("accessibility", { ...ACCESS, text_size: "xlarge", high_contrast: true });
assert.deepEqual(looks(), { textSize: "xlarge", captions: "timed", calm: "off", contrast: "on" });
// Bigger text means bigger boxes: the server hears where the UI is at once.
await settle();
assert.ok(sock.emitted.slice(sent).some(([name]) => name === "ui_rects"));
// A value this page doesn't know leaves that setting as it was; the known ones still apply.
sock.fire("accessibility", { captions_stay: "yes", text_size: "XLARGE", calm: 1, high_contrast: null, speech_rate: "slower" });
assert.deepEqual(looks(), { textSize: "xlarge", captions: "timed", calm: "off", contrast: "on" });
sock.fire("accessibility", { text_size: "huge", calm: true });
assert.deepEqual(looks(), { textSize: "xlarge", captions: "timed", calm: "on", contrast: "on" });
""",
    )


@needs_node
def test_projector_accessibility_ignores_a_stale_get(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "index.html",
        PROJECTOR
        + f"const ACCESS = {ACCESS};\n"
        + LOOKS
        + """
let reply;
installFetch({ "GET /api/accessibility": () => new Promise((resolve) => { reply = resolve; }) });
sock.fire("connect");
await settle();
sock.fire("accessibility", { ...ACCESS, text_size: "xlarge", calm: true });
reply({ status: 200, body: ACCESS });
await settle();
assert.deepEqual(looks(), { textSize: "xlarge", captions: "timed", calm: "on", contrast: "off" });
""",
    )


@needs_node
def test_projector_keeps_the_rendered_settings_until_the_server_says_otherwise(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "index.html",
        PROJECTOR_ACCESS
        + LOOKS
        + """
const hl = $("highlight");
const box = () => ["left", "top", "width", "height"].map((k) => hl.style[k]);
// Captions are held from the first answer, before any GET comes back.
const mark = timers.length;
sock.fire("avatar_update", { animation_state: "idle", spoken_text: "x is 5", display_text: "x = 5" });
assert.deepEqual(timers.slice(mark).map((x) => x.ms), []);
// An older server has no /api/accessibility: the page keeps what Flask rendered.
installFetch({ "GET /api/accessibility": { status: 404, body: { error: "not found" } } });
sock.fire("connect");
await settle();
assert.deepEqual(looks(), { textSize: "large", captions: "stay", calm: "on", contrast: "on" });
// High contrast from the first paint: the outline is measured with its 6 px border.
sock.fire("avatar_update", { animation_state: "pointing", highlight: [400, 300, 200, 100] });
assert.deepEqual(box(), ["380px", "280px", "240px", "140px"]);
""",
    )


@needs_node
def test_projector_captions_stay_up_until_the_next_question(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "index.html",
        PROJECTOR_TIMED
        + f"const ACCESS = {ACCESS};\n"
        + """
const bubble = $("bubble");
const transcript = $("transcript");
const shown = (node) => !node.classList.contains("hidden");
const armed = (mark) => timers.slice(mark).map((x) => x.ms);
// Audio the test can finish: the fake DOM's Audio never ends.
const clips = [];
globalThis.Audio = class {
  constructor(url) { this.src = url; this.paused = true; this.ended = false; this.listeners = {}; clips.push(this); }
  play() { this.paused = false; return Promise.resolve(); }
  pause() { this.paused = true; }
  addEventListener(type, fn) { (this.listeners[type] ||= []).push(fn); }
};
sock.fire("connect");
await settle();
sock.fire("accessibility", { ...ACCESS, captions_stay: true });
const mark = timers.length;
// The student's words stay up while the assistant thinks, instead of 5 s.
sock.fire("transcript", { text: "what do I do next?" });
sock.fire("status", { state: "thinking" });
assert.ok(shown(transcript));
// The answer takes their place, and nothing takes it down: not the answer, not its voice, not a voiceless server.
const answer = { animation_state: "idle", spoken_text: "subtract five", display_text: "Subtract 5." };
sock.fire("avatar_update", answer);
assert.ok(!shown(transcript));
assert.equal(transcript.textContent, "");
sock.fire("status", { state: "speaking" });
sock.fire("speak", { url: "/tts/a.mp3", text: answer.spoken_text, display_text: answer.display_text });
const clip = clips.at(-1);
clip.ended = true;
for (const fn of clip.listeners.ended) fn();
sock.fire("status", { state: "idle" });
sock.fire("speak", { url: null, text: answer.spoken_text, display_text: answer.display_text });
assert.deepEqual(armed(mark), [], "no timer may take a held caption down");
assert.ok(shown(bubble));
assert.equal(bubble.textContent, "Subtract 5.");
// The next answer replaces it; the microphone waking up is not a question yet.
sock.fire("avatar_update", { ...answer, display_text: "Now divide by 3." });
assert.equal(bubble.textContent, "Now divide by 3.");
sock.fire("status", { state: "listening" });
assert.ok(shown(bubble));
// A real new question takes it down before the camera reads the board: the student's words...
sock.fire("transcript", { text: "and then?" });
assert.ok(!shown(bubble));
// ...or the board being read, whichever the projector hears first.
sock.fire("avatar_update", answer);
assert.ok(shown(bubble));
sock.fire("status", { state: "thinking" });
assert.ok(!shown(bubble));
assert.deepEqual(armed(mark), []);
""",
    )


@needs_node
def test_projector_captions_switched_off_go_like_any_other(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "index.html",
        PROJECTOR_TIMED
        + f"const ACCESS = {ACCESS};\n"
        + """
const bubble = $("bubble");
const transcript = $("transcript");
const shown = (node) => !node.classList.contains("hidden");
sock.fire("connect");
await settle();
sock.fire("avatar_update", { animation_state: "idle", spoken_text: "x is 5", display_text: "x = 5" });
// Switched on while a timed caption is up: the hide timer it already had leaves it up too.
sock.fire("accessibility", { ...ACCESS, captions_stay: true });
runTimer(4000);
assert.ok(shown(bubble));
// Another setting changing while it is held arms nothing.
let mark = timers.length;
sock.fire("accessibility", { ...ACCESS, captions_stay: true, text_size: "large" });
assert.deepEqual(timers.slice(mark).map((x) => x.ms), []);
// Switched off while a caption is held: it goes 4 s later, like an answer without a voice.
sock.fire("accessibility", { ...ACCESS, captions_stay: false });
assert.deepEqual(timers.slice(mark).map((x) => x.ms), [4000]);
assert.ok(shown(bubble));
runTimer(4000);
assert.ok(!shown(bubble));
// The student's words held the same way go after their usual 5 s.
sock.fire("accessibility", { ...ACCESS, captions_stay: true });
sock.fire("transcript", { text: "is that right?" });
mark = timers.length;
sock.fire("accessibility", { ...ACCESS, captions_stay: false });
assert.deepEqual(timers.slice(mark).map((x) => x.ms), [5000]);
runTimer(5000);
assert.ok(!shown(transcript));
""",
    )


@needs_node
def test_projector_high_contrast_outline_is_measured_with_its_thicker_border(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "index.html",
        PROJECTOR
        + f"const ACCESS = {ACCESS};\n"
        + """
const hl = $("highlight");
const box = () => ["left", "top", "width", "height"].map((k) => hl.style[k]);
const rects = () => sock.emitted.filter(([name]) => name === "ui_rects").at(-1)[1].rects;
sock.fire("connect");
await settle();
// The outline test's answer: 14 px out from the ink plus the 3 px border...
sock.fire("avatar_update", { animation_state: "pointing", highlight: [400, 300, 200, 100] });
assert.deepEqual(box(), ["383px", "283px", "234px", "134px"]);
// ...and in high contrast the 6 px border style.css draws, placed again at once: the gap to the ink stays 14 px.
sock.fire("accessibility", { ...ACCESS, high_contrast: true });
assert.deepEqual(box(), ["380px", "280px", "240px", "140px"]);
// The server gets the outline as its four 6 px edges (plus the 10 px every UI box gets), after the avatar.
await settle();
assert.deepEqual(rects().slice(1, 5), [[370, 270, 260, 26], [370, 404, 260, 26], [370, 270, 26, 160], [604, 270, 26, 160]]);
// Near a corner it still stays on the screen, and switching back gives the usual box.
sock.fire("avatar_update", { animation_state: "pointing", highlight: [1900, 1070, 100, 50] });
assert.deepEqual(box(), ["1880px", "1050px", "40px", "30px"]);
sock.fire("accessibility", { ...ACCESS, high_contrast: false });
assert.deepEqual(box(), ["1883px", "1053px", "37px", "27px"]);
""",
    )


def test_projector_text_size_scales_what_the_room_reads() -> None:
    scales = {sel: body.strip() for sel, body in style_rules("body").items() if "--text-scale:" in body}
    assert scales == {
        "body": "--text-scale: 1;",
        'body[data-text-size="large"]': "--text-scale: 1.25;",
        'body[data-text-size="xlarge"]': "--text-scale: 1.5;",
    }
    for selector in ("#bubble", "#quiz-card-count", "#quiz-card-problem", "#transcript", "#status"):
        rule = style_rules(selector)[selector]
        assert re.search(r"font-size:\s*calc\(.*\*\s*var\(--text-scale\)\)", rule), selector
    # A bigger text gets a wider bubble, but never more than half the board.
    bubble = style_rules("#bubble")["#bubble"]
    assert "max-width: min(50vw, calc(clamp(200px, 34vw, 680px) * var(--text-scale)))" in bubble


def test_projector_calm_mode_stops_the_motion_and_softens_the_glow() -> None:
    on = 'body[data-calm="on"]'
    calm = style_rules(on)
    # No bob, hop, shake or nudge, and no sprite slide: the avatar only glides to where it points.
    assert "animation: none !important" in calm[f"{on} #avatar"]
    assert "transition-property: left, top" in calm[f"{on} #avatar"]
    for part in ("#avatar #bubble", "#highlight", f"#status .ring, {on} #status .dots i"):
        assert "animation: none !important" in calm[f"{on} {part}"], part
    # Every glow left is a soft one, and the calibration checkerboard is never touched.
    for selector, body in calm.items():
        assert all(int(blur) <= 4 for blur in re.findall(r"\b0 0 (\d+)px", body)), selector
        assert "calibration" not in selector
    # The computer's own reduced-motion setting keeps working without calm mode.
    css = (STATIC / "css" / "style.css").read_text(encoding="utf-8")
    assert "#avatar { animation: none !important" in css_block(css, "@media (prefers-reduced-motion: reduce)")


def test_projector_high_contrast_tells_right_from_wrong_by_more_than_colour() -> None:
    on = 'body[data-contrast="on"]'
    contrast = style_rules(on)
    bubble = contrast[f"{on} #avatar #bubble"]
    assert "border-width: 6px" in bubble and "color: var(--white)" in bubble and "text-shadow: none" in bubble
    assert "border-style: dashed" in contrast[f"{on} #avatar.state-error #bubble"]
    assert "border-style: double" in contrast[f"{on} #avatar.state-celebrating #bubble"]
    # The outline is as thick as main.js measures it, so the gap to the ink stays the same.
    assert "border-width: 6px" in contrast[f"{on} #highlight"]
    js = (STATIC / "js" / "main.js").read_text(encoding="utf-8")
    widths = dict(re.findall(r"const (HIGHLIGHT_BORDER\w*) = (\d+);", js))
    assert widths == {"HIGHLIGHT_BORDER": "3", "HIGHLIGHT_BORDER_CONTRAST": "6"}
    assert "border: 3px" in style_rules("#highlight")["#highlight"]
    # The quiz card's words go to full strength; its border stays thin and glow-free (the camera reads it as ink).
    card = contrast[f"{on} #quiz-card"]
    assert "color: var(--white)" in card and "border" not in card and "shadow" not in card
    assert "color: var(--cyan)" in contrast[f"{on} #quiz-card-count"]
    # The board stays black: high contrast adds no fill anywhere.
    assert not any("background" in body for body in contrast.values())


def test_projector_page_paints_the_rooms_settings_first() -> None:
    size = {"projector_width": 1280, "projector_height": 800}
    changed = AccessibilitySettings(captions_stay=True, text_size="xlarge", calm=True, high_contrast=True)
    changed_looks = {"data-text-size": "xlarge", "data-captions": "stay", "data-calm": "on", "data-contrast": "on"}
    defaults = {"data-text-size": "normal", "data-captions": "timed", "data-calm": "off", "data-contrast": "off"}
    cases = [
        ({"access": changed.to_dict()}, changed_looks),
        ({"access": AccessibilitySettings().to_dict()}, defaults),
        # A render without the settings (an older caller) paints the defaults.
        ({}, defaults),
    ]
    for variables, looks in cases:
        tag = re.search(r"<body\b[^>]*>", render_page("index.html", **size, **variables))
        assert tag, "no <body> tag"
        attrs = dict(re.findall(r'\s(data-[a-z-]+)="([^"]*)"', tag.group(0)))
        assert attrs == {"data-projector-width": "1280", "data-projector-height": "800", **looks}, variables
```

- [ ] **Step 2: Run** `.venv/bin/python -m pytest -q tests/test_frontend.py -k "accessibility or captions or high_contrast or text_size or calm_mode or paints_the_rooms or rendered_settings"` → FAIL. 10 selected, 10 failed:
  - `main.js` has no `accessibility` handler and makes no GET. The body dataset never changes, timed captions still arm 4000/1200/5000 ms timers, and the outline stays at 3 px. The stale-GET test fails with `reply is not a function`.
  - `style.css` has no `--text-scale`, `data-calm` or `data-contrast` rules, so `KeyError` is raised.
  - `index.html` ignores `access`.

- [ ] **Step 3: Implement**

**`biobench_sar/templates/index.html`** (line 9). The `| default` guards are required, not optional: `test_phone_mode_parts_render_only_in_lan_mode` (test_frontend.py:2452) renders `index.html` without `access`, and Jinja's default `Undefined` raises `UndefinedError` on `access.text_size`.

Old:
```html
<body data-projector-width="{{ projector_width }}" data-projector-height="{{ projector_height }}">
```
New:
```html
{# The room's accessibility settings (app.py page()), so the first paint is already right; main.js keeps them
   current. A render without them paints the defaults. #}
{% set access = access | default({}) %}
<body data-projector-width="{{ projector_width }}" data-projector-height="{{ projector_height }}"
      data-text-size="{{ access.text_size | default('normal') }}"
      data-captions="{{ 'stay' if access.captions_stay else 'timed' }}"
      data-calm="{{ 'on' if access.calm else 'off' }}" data-contrast="{{ 'on' if access.high_contrast else 'off' }}">
```

**`biobench_sar/static/js/main.js`**

(a) Line 24. Old:
```js
const HIGHLIGHT_BORDER = 3; // CSS px: style.css #highlight border width
```
New:
```js
const HIGHLIGHT_BORDER = 3; // CSS px: style.css #highlight border width
const HIGHLIGHT_BORDER_CONTRAST = 6; // the same in high contrast (body[data-contrast="on"] #highlight)
```

(b) Line 32. Old:
```js
const UI_JITTER = 12; // projector px: moving less than this (the idle bob) is not a change worth sending
```
New:
```js
const UI_JITTER = 12; // projector px: moving less than this (the idle bob) is not a change worth sending
// The room's accessibility settings (accessibility.py) live on <body> as data-* attributes, which style.css reads:
// data-text-size, data-captions ("stay" | "timed"), data-calm and data-contrast ("on" | "off").
const TEXT_SIZES = ["normal", "large", "xlarge"];
```

(c) Line 63. Old:
```js
let languageVersion = 0; // the same for "language"
```
New:
```js
let languageVersion = 0; // the same for "language"
let accessVersion = 0; // and for "accessibility"
```

(d) Lines 134-141. Old:
```js
// The outline's box in window px: the ink plus the gap, plus the border (drawn inside the box), kept on screen.
function highlightBox([x, y, w, h]) {
  const a = toPixels(x - HIGHLIGHT_GAP, y - HIGHLIGHT_GAP);
  const b = toPixels(x + w + HIGHLIGHT_GAP, y + h + HIGHLIGHT_GAP);
  const left = Math.max(0, a.x - HIGHLIGHT_BORDER);
  const top = Math.max(0, a.y - HIGHLIGHT_BORDER);
  const right = Math.min(window.innerWidth, b.x + HIGHLIGHT_BORDER);
  const bottom = Math.min(window.innerHeight, b.y + HIGHLIGHT_BORDER);
```
New:
```js
function highlightBorder() {
  return body.dataset.contrast === "on" ? HIGHLIGHT_BORDER_CONTRAST : HIGHLIGHT_BORDER;
}

// The outline's box in window px: the ink plus the gap, plus the border (drawn inside the box), kept on screen.
function highlightBox([x, y, w, h]) {
  const border = highlightBorder();
  const a = toPixels(x - HIGHLIGHT_GAP, y - HIGHLIGHT_GAP);
  const b = toPixels(x + w + HIGHLIGHT_GAP, y + h + HIGHLIGHT_GAP);
  const left = Math.max(0, a.x - border);
  const top = Math.max(0, a.y - border);
  const right = Math.min(window.innerWidth, b.x + border);
  const bottom = Math.min(window.innerHeight, b.y + border);
```

(e) Lines 203-204. Old:
```js
function edges(r) {
  const band = HIGHLIGHT_BORDER;
```
New:
```js
function edges(r) {
  const band = highlightBorder();
```

(f) Lines 263-267. Old:
```js
// Single owner of bubbleTimer: every (re)arm clears the previous one so stale timers can't hide a newer bubble.
function armBubbleTimer(ms) {
  clearTimeout(bubbleTimer);
  bubbleTimer = ms ? setTimeout(hideBubbleUnlessSpeaking, ms) : null;
}
```
New:
```js
// Captions held until the next question (accessibility "captions_stay"): no timer takes the bubble down.
function captionsStay() {
  return body.dataset.captions === "stay";
}

// Single owner of bubbleTimer: every (re)arm clears the previous one so stale timers can't hide a newer bubble.
function armBubbleTimer(ms) {
  clearTimeout(bubbleTimer);
  bubbleTimer = ms && !captionsStay() ? setTimeout(hideBubbleUnlessSpeaking, ms) : null;
}
```

(g) Lines 289-293. Old:
```js
function hideBubbleUnlessSpeaking() {
  if (audio && !audio.paused && !audio.ended) return;
  bubble.classList.add("hidden");
  uiChanged();
}
```
New:
```js
// A timer armed before captions were set to stay leaves the caption up too.
function hideBubbleUnlessSpeaking() {
  if (captionsStay() || (audio && !audio.paused && !audio.ended)) return;
  bubble.classList.add("hidden");
  uiChanged();
}

// A held caption goes when a real new question starts (its words, or the board being read), so it never covers
// the board in the camera picture. Timed captions keep their own timer.
function dropHeldCaption() {
  if (!captionsStay() || bubble.classList.contains("hidden")) return;
  bubble.classList.add("hidden");
  uiChanged();
}
```

(h) `onAvatarUpdate`, lines 314-317. Old:
```js
  } else {
    setState(finalState);
  }
  // display_text keeps math in normal notation (3x + 5 = 20); spoken_text spells it out for the voice.
```
New:
```js
  } else {
    setState(finalState);
  }
  // Held captions: the student's words stay up until this answer arrives.
  if (captionsStay()) {
    clearTranscript();
    uiChanged();
  }
  // display_text keeps math in normal notation (3x + 5 = 20); spoken_text spells it out for the voice.
```

(i) `renderStatus`, lines 354-356. Old:
```js
  if (capturing) clearHighlight(true);
  uiChanged();
}
```
New:
```js
  if (capturing) clearHighlight(true);
  if (s === "thinking") dropHeldCaption();
  uiChanged();
}
```

(j) After `refreshLanguage`, lines 382-384. Old:
```js
    .then((d) => { if (d && seen === languageVersion) onLanguage(d); })
    .catch(() => {});
}
```
New:
```js
    .then((d) => { if (d && seen === languageVersion) onLanguage(d); })
    .catch(() => {});
}

// Unknown values are ignored, so a newer server can't put the page in a state style.css doesn't know.
function applyAccessibility(d) {
  const held = captionsStay();
  if (TEXT_SIZES.includes(d.text_size)) body.dataset.textSize = d.text_size;
  if (typeof d.captions_stay === "boolean") body.dataset.captions = d.captions_stay ? "stay" : "timed";
  if (typeof d.calm === "boolean") body.dataset.calm = d.calm ? "on" : "off";
  if (typeof d.high_contrast === "boolean") body.dataset.contrast = d.high_contrast ? "on" : "off";
  // Captions switched off while one is held: it goes like an answer without audio, the student's words after 5 s.
  if (held && !captionsStay()) {
    if (!bubble.classList.contains("hidden")) armBubbleTimer(4000);
    if (!transcriptEl.classList.contains("hidden")) armTranscriptTimer();
  }
  // Bigger text and a thicker outline change the boxes: place them again and tell the server at once.
  placeHighlight();
  fitBubble();
  uiChanged();
}

// Text size, held captions, calm mode and high contrast, set for the whole room from the dashboard.
function onAccessibility(d) {
  accessVersion += 1;
  applyAccessibility(d);
}

// Changes made while the page was away come with the GET on every connect; Flask painted the first ones.
function refreshAccessibility() {
  const seen = accessVersion;
  if (typeof fetch !== "function") return;
  fetch("/api/accessibility")
    .then((r) => (r.ok ? r.json() : null))
    .then((d) => { if (d && seen === accessVersion) onAccessibility(d); })
    .catch(() => {});
}
```

(k) `onTranscript`, lines 403-412. Old:
```js
function onTranscript(d) {
  clearTimeout(transcriptTimer);
  transcriptEl.textContent = d.text || "";
  transcriptEl.classList.toggle("hidden", !d.text);
  transcriptTimer = setTimeout(() => {
    transcriptEl.classList.add("hidden");
    uiChanged();
  }, 5000);
  uiChanged();
}
```
New:
```js
// The student's words go after 5 s, or with captions held when the answer arrives (onAvatarUpdate).
function armTranscriptTimer() {
  clearTimeout(transcriptTimer);
  transcriptTimer = null;
  if (captionsStay()) return;
  transcriptTimer = setTimeout(() => {
    transcriptEl.classList.add("hidden");
    uiChanged();
  }, 5000);
}

function onTranscript(d) {
  transcriptEl.textContent = d.text || "";
  transcriptEl.classList.toggle("hidden", !d.text);
  if (d.text) dropHeldCaption();
  armTranscriptTimer();
  uiChanged();
}
```

(l) `resync`, lines 445-447. Old:
```js
  refreshTutor();
  refreshLanguage();
}
```
New:
```js
  refreshTutor();
  refreshLanguage();
  refreshAccessibility();
}
```

(m) Handlers map, lines 500-501. Old:
```js
    show_calibration: showCalibration, hide_calibration: hideCalibration, tutor: onTutor, language: onLanguage,
  };
```
New:
```js
    show_calibration: showCalibration, hide_calibration: hideCalibration, tutor: onTutor, language: onLanguage,
    accessibility: onAccessibility,
  };
```

**`biobench_sar/static/css/style.css`**

(a) Line 6. Old:
```css
.hidden { display: none !important; }
```
New:
```css
.hidden { display: none !important; }
/* Text size from the room's accessibility settings (main.js sets body data-text-size): it scales what people read
   from across the room, i.e. the bubble, the quiz card, the student's words and the status corner. */
body { --text-scale: 1; }
body[data-text-size="large"] { --text-scale: 1.25; }
body[data-text-size="xlarge"] { --text-scale: 1.5; }
```

(b) `#bubble`, lines 43-44. Old:
```css
  width: max-content; max-width: clamp(200px, 34vw, 680px); padding: 0.3em 0.55em; margin-bottom: 4px;
  font-size: clamp(16px, 1.9vw, 40px); line-height: 1.25; overflow-wrap: anywhere; font-weight: 600; text-align: center;
```
New:
```css
  width: max-content; max-width: min(50vw, calc(clamp(200px, 34vw, 680px) * var(--text-scale)));
  padding: 0.3em 0.55em; margin-bottom: 4px; font-size: calc(clamp(16px, 1.9vw, 40px) * var(--text-scale));
  line-height: 1.25; overflow-wrap: anywhere; font-weight: 600; text-align: center;
```

(c) `#status`, line 75. Old:
```css
  font-size: 16px; color: var(--cyan); z-index: 20; text-shadow: 0 0 6px var(--cyan); }
```
New:
```css
  font-size: calc(16px * var(--text-scale)); color: var(--cyan); z-index: 20; text-shadow: 0 0 6px var(--cyan); }
```

(d) `#transcript`, line 91. Old:
```css
  font-size: 22px; color: var(--white); text-shadow: 0 0 8px var(--cyan); z-index: 20; text-align: center; }
```
New:
```css
  font-size: calc(22px * var(--text-scale)); color: var(--white); text-shadow: 0 0 8px var(--cyan); z-index: 20;
  text-align: center; }
```

(e) Quiz card text, lines 100-102. The first `#quiz-card {` rule (lines 97-99) is not touched. Old:
```css
#quiz-card-count { display: block; font-size: clamp(11px, 0.85vw, 18px); color: rgba(127, 252, 255, 0.75); }
#quiz-card-problem { display: block; font-size: clamp(16px, 1.7vw, 34px); font-weight: 600; line-height: 1.2;
  overflow-wrap: anywhere; }
```
New:
```css
#quiz-card-count { display: block; font-size: calc(clamp(11px, 0.85vw, 18px) * var(--text-scale));
  color: rgba(127, 252, 255, 0.75); }
#quiz-card-problem { display: block; font-size: calc(clamp(16px, 1.7vw, 34px) * var(--text-scale)); font-weight: 600;
  line-height: 1.2; overflow-wrap: anywhere; }
```

(f) End of the file, after the reduced-motion block (lines 131-132). Old:
```css
  #status .ring, #status .dots i { animation: none !important; }
}
```
New:
```css
  #status .ring, #status .dots i { animation: none !important; }
}

/* Calm mode (body data-calm="on", from the room's accessibility settings): the reduced-motion rules above for a
   room whose computer doesn't ask for them, plus no sprite slide (the avatar only glides to where it points) and
   soft glows. The calibration checkerboard is never touched. */
body[data-calm="on"] #avatar { animation: none !important; transition-property: left, top;
  filter: drop-shadow(0 0 4px rgba(127, 252, 255, 0.5)); }
body[data-calm="on"] #avatar.state-error { filter: drop-shadow(0 0 4px rgba(255, 179, 71, 0.6)); }
body[data-calm="on"] #avatar #bubble { animation: none !important; text-shadow: 0 0 3px var(--cyan);
  box-shadow: 0 0 4px var(--cyan); }
body[data-calm="on"] #avatar.state-error #bubble { text-shadow: 0 0 3px var(--amber);
  box-shadow: 0 0 4px var(--amber); }
body[data-calm="on"] #highlight { animation: none !important; transition-duration: 0.01s;
  box-shadow: 0 0 4px var(--hl-glow); }
body[data-calm="on"] #status .ring, body[data-calm="on"] #status .dots i { animation: none !important; }
body[data-calm="on"] #status { text-shadow: 0 0 2px currentColor; }
body[data-calm="on"] #transcript { text-shadow: 0 0 3px var(--cyan); }

/* High contrast (body data-contrast="on"): thick borders, pure white text with no blurred glow, and a wrong or a
   right answer told apart by the border's style as well as its colour. The outline is as thick as main.js measures
   it (HIGHLIGHT_BORDER_CONTRAST). The quiz card keeps its thin, glow-free border, since the camera reads bright
   lines as ink; only its words go to full strength. The background stays black. */
body[data-contrast="on"] #avatar #bubble { border-width: 6px; color: var(--white); text-shadow: none; }
body[data-contrast="on"] #avatar.state-error #bubble { border-style: dashed; text-shadow: none; }
body[data-contrast="on"] #avatar.state-celebrating #bubble { border-style: double; }
body[data-contrast="on"] #highlight { border-width: 6px; }
body[data-contrast="on"] #quiz-card { color: var(--white); }
body[data-contrast="on"] #quiz-card-count { color: var(--cyan); }
body[data-contrast="on"] #transcript { text-shadow: none; }
```

No new i18n strings. `main.js` names no new `prefix.key` strings, so `test_every_key_the_scripts_name_exists` is unaffected.

- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q tests/test_frontend.py -k "accessibility or captions or high_contrast or text_size or calm_mode or paints_the_rooms or rendered_settings"` → PASS (10 passed).
  - Then run all of `.venv/bin/python -m pytest -q tests/test_frontend.py` → PASS (104 passed; takes several minutes under node). These existing tests touch the changed files and still pass unchanged:
    - `test_projector_quiz_card_is_faint_and_hidden_while_capturing` (line 1139): the first `#quiz-card {` rule is untouched.
    - `test_projector_outlines_the_ink_the_answer_is_about` (line 1909): 3 px geometry without `data-contrast`.
    - `test_projector_reports_where_its_ui_is` (line 2003).
    - `test_phone_mode_parts_render_only_in_lan_mode` (line 2452): renders `index.html` without `access`.
    - `test_every_visible_template_text_has_a_dictionary_key` / `test_template_english_is_the_dictionary_english[index.html]`: `{# #}` and `{% %}` are stripped by `template_source`.
  - `test_projector_reports_where_its_ui_is` uses real 600 ms and 700 ms waits against a 1 s heartbeat. On a heavily loaded machine it can fail regardless of this change; if it does, rerun it alone (`-k test_projector_reports_where_its_ui_is`).
  - Then `.venv/bin/ruff check . && .venv/bin/ruff format --check .` → clean.
- [ ] No existing test needs its assertions changed.
- [ ] **Step 5: Commit** `feat(projector): follow the room's text size, held captions, calm mode and high contrast`
  ```bash
  git add biobench_sar/templates/index.html biobench_sar/static/js/main.js biobench_sar/static/css/style.css tests/test_frontend.py
  git commit -m "feat(projector): follow the room's text size, held captions, calm mode and high contrast

  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
  ```

### Task 6: A teacher (on the laptop or a joined phone) can see and change the room's accessibility settings from an **Accessibility** card under **More tools**, and every open dashboard follows the change.

**Files:** Create `biobench_sar/static/js/dashboard/accessibility.js` · Modify `biobench_sar/templates/precharge.html`, `biobench_sar/static/js/dashboard/main.js`, `biobench_sar/static/js/dashboard/camera.js`, `biobench_sar/static/css/dashboard.css`, `biobench_sar/static/js/i18n.js` · Test `tests/test_frontend.py`

Notes before starting:
- The `run_node` harness copies the whole `static/js/dashboard/` folder (`tests/test_frontend.py:122`, `shutil.copytree`). The new module is picked up with no change to the harness.
- In the fake DOM, `document.querySelectorAll` handles only `[data-x]`, and `FakeElement.querySelectorAll` handles only tag names. Every template node is a flat child of `<body>`. So `accessibility.js` reaches every control by `id`, and the camera test nests the two camera buttons itself.
- Task 4 adds the default `GET /api/accessibility` route in `tests/js/fake_dom.mjs`. Every new test below sets that route itself, so none of them depends on the default. Only the existing Spanish sweep (`test_dashboard_in_spanish_leaves_no_english_behind`) uses the default. It passes either way, because the sweep walks hidden nodes too.
- `static/sw.js` `SHELL` is not changed. It already leaves out `language.js`, `phone.js` and `i18n.js`. The worker caches `/static/*` at runtime (stale-while-revalidate once it has seen LAN mode), and `test_service_worker_shell_lists_files_that_exist` only checks the files that are listed.

- [ ] **Step 1: Failing tests** (append at the end of `tests/test_frontend.py`: after `test_dashboard_steps_fit_one_laptop_screen` and after any projector tests Task 5 appended there. No new imports: `json`, `Path`, `DASHBOARD`, `TIMERS`, `run_node` and `needs_node` already exist.)
```python


# ---- dashboard: accessibility card -----------------------------------------------------------------------------------

ROOM_ACCESS = {
    "captions_stay": False,
    "text_size": "normal",
    "speech_rate": "normal",
    "calm": False,
    "high_contrast": False,
}
# The card as a teacher reads it: which boxes are ticked, which size and speed are pressed.
ACCESS_DASHBOARD = (
    DASHBOARD
    + f"const ACCESS = {json.dumps(ROOM_ACCESS)};\n"
    + """
const card = () => ({
  captions_stay: $("access-captions").checked,
  text_size: ["normal", "large", "xlarge"].filter((s) => $(`access-size-${s}`).getAttribute("aria-pressed") === "true"),
  speech_rate: ["slower", "normal", "faster"].filter((r) => $(`access-rate-${r}`).getAttribute("aria-pressed") === "true"),
  calm: $("access-calm").checked,
  high_contrast: $("access-contrast").checked,
});
const shows = (body) => ({ ...body, text_size: [body.text_size], speech_rate: [body.speech_rate] });
const puts = (http) => http.calls.filter((c) => c.method === "PUT" && c.path === "/api/accessibility").map((c) => JSON.parse(c.body));
"""
)


@needs_node
def test_accessibility_card_waits_for_the_server_and_shows_the_rooms_settings(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "precharge.html",
        ACCESS_DASHBOARD
        + """
const sock = installIo();
const replies = [];
installFetch({ "GET /api/accessibility": () => new Promise((resolve) => replies.push(resolve)) });
await import("./dashboard/main.js");
sock.fire("connect");
await settle();
// No guessed defaults: the card shows only what the server says the room has.
assert.equal($("access-card").hidden, true);
const room = { ...ACCESS, captions_stay: true, text_size: "xlarge", speech_rate: "slower", high_contrast: true };
for (const reply of replies) reply({ status: 200, body: room });
await settle();
assert.equal($("access-card").hidden, false);
assert.deepEqual(card(), shows(room));
for (const s of ["normal", "large", "xlarge"]) assert.equal($(`access-size-${s}`).dataset.accessSize, s);
for (const r of ["slower", "normal", "faster"]) assert.equal($(`access-rate-${r}`).dataset.accessRate, r);
""",
    )
    run_node(
        tmp_path / "older",
        "precharge.html",
        ACCESS_DASHBOARD
        + """
// A server from before accessibility settings: the card never shows, and nothing is sent.
const sock = installIo();
const http = installFetch({ "GET /api/accessibility": { status: 404, body: { error: "not found" } } });
await import("./dashboard/main.js");
sock.fire("connect");
await settle();
assert.equal($("access-card").hidden, true);
assert.ok(http.calls.some((c) => c.path === "/api/accessibility"), "the page never asked");
assert.deepEqual(puts(http), []);
""",
    )


@needs_node
def test_accessibility_card_sends_only_the_setting_that_changed(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "precharge.html",
        ACCESS_DASHBOARD
        + """
const sock = installIo();
let room = { ...ACCESS };
let hold = null;
const http = installFetch({
  "GET /api/accessibility": () => ({ status: 200, body: room }),
  "PUT /api/accessibility": async (init) => {
    if (hold) await hold;
    room = { ...room, ...JSON.parse(init.body) };
    return { status: 200, body: room };
  },
});
await import("./dashboard/main.js");
sock.fire("connect");
await settle();
const controls = ["access-captions", "access-calm", "access-contrast", "access-size-normal", "access-size-large",
                  "access-size-xlarge", "access-rate-slower", "access-rate-normal", "access-rate-faster"];
let release;
hold = new Promise((resolve) => { release = resolve; });
const click = $("access-size-large").fire("click");
await settle();
// The whole card waits for the server, so a second tap can't race the first.
assert.deepEqual(controls.filter((id) => !$(id).disabled), []);
release();
await click;
hold = null;
assert.deepEqual(puts(http), [{ text_size: "large" }]);
assert.deepEqual(card(), shows({ ...ACCESS, text_size: "large" }));
assert.deepEqual(controls.filter((id) => $(id).disabled), []);
// A box ticks itself before its change event fires; the PUT carries just that box.
$("access-captions").checked = true;
await $("access-captions").fire("change");
assert.deepEqual(puts(http).at(-1), { captions_stay: true });
await $("access-rate-faster").fire("click");
assert.deepEqual(puts(http).at(-1), { speech_rate: "faster" });
$("access-contrast").checked = true;
await $("access-contrast").fire("change");
assert.deepEqual(puts(http).at(-1), { high_contrast: true });
assert.deepEqual(card(), shows({ ...ACCESS, captions_stay: true, text_size: "large", speech_rate: "faster",
                                 high_contrast: true }));
// Pressing the size that is already on sends nothing.
await $("access-size-large").fire("click");
assert.equal(puts(http).length, 4);
assert.equal($("access-msg").textContent, "");
""",
    )


@needs_node
def test_accessibility_card_follows_a_change_made_elsewhere(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "precharge.html",
        ACCESS_DASHBOARD
        + """
const sock = installIo();
const replies = [];
installFetch({ "GET /api/accessibility": () => new Promise((resolve) => replies.push(resolve)) });
await import("./dashboard/main.js");
sock.fire("connect");
await settle();
// A phone turned on calm mode and a faster voice: every open dashboard follows at once.
const pushed = { ...ACCESS, calm: true, speech_rate: "faster" };
sock.fire("accessibility", pushed);
assert.equal($("access-card").hidden, false);
assert.deepEqual(card(), shows(pushed));
assert.match(logText(), /accessibility captions stay: false · text size: normal · speech rate: faster · calm: true/);
// A GET that left before the push can't bring the old settings back.
for (const reply of replies) reply({ status: 200, body: ACCESS });
await settle();
assert.deepEqual(card(), shows(pushed));
// Half a body (an older or newer server) changes nothing.
sock.fire("accessibility", { text_size: "large" });
assert.deepEqual(card(), shows(pushed));
""",
    )


@needs_node
def test_accessibility_card_failure_is_explained_and_changes_nothing(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "precharge.html",
        ACCESS_DASHBOARD
        + """
const sock = installIo();
const http = installFetch({
  "GET /api/accessibility": { status: 200, body: ACCESS },
  "PUT /api/accessibility": { status: 400, body: { error: "text_size must be one of: normal, large, xlarge" } },
});
await import("./dashboard/main.js");
sock.fire("connect");
await settle();
await $("access-size-large").fire("click");
const note = $("access-msg");
assert.equal(note.textContent,
  "Couldn't change that setting. Check the assistant is still running (restart the assistant if it closed), then try again.");
assert.equal(note.dataset.tone, "warn");
assert.match(logText(), /accessibility text_size must be one of: normal, large, xlarge/);
assert.deepEqual(card(), shows(ACCESS));
// The box the teacher ticked goes back to what the room really has.
$("access-calm").checked = true;
await $("access-calm").fire("change");
assert.deepEqual(puts(http), [{ text_size: "large" }, { calm: true }]);
assert.equal($("access-calm").checked, false);
assert.equal($("access-calm").disabled, false);
assert.equal($("access-size-large").disabled, false);
// The note switches language with the page.
sock.fire("language", { setting: "es", current: "es" });
assert.equal(note.textContent,
  "No se pudo cambiar ese ajuste. Revisa que el asistente siga funcionando (si se cerró, vuelve a iniciar el asistente) " +
  "y vuelve a intentarlo.");
""",
    )


@needs_node
def test_accessibility_card_in_spanish(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "precharge.html",
        ACCESS_DASHBOARD
        + """
const sock = installIo();
installFetch({
  "GET /api/language": { status: 200, body: { setting: "es", current: "es" } },
  "GET /api/accessibility": { status: 200, body: ACCESS },
});
await import("./dashboard/main.js");
sock.fire("connect");
await settle();
const say = (key) => [...doc.body.descendants()].find((n) => n.dataset.i18n === key).textContent;
assert.equal(doc.documentElement.lang, "es");
assert.equal(say("access.title"), "Accesibilidad");
assert.equal(say("access.hint"), "Cambian el proyector y la voz para todo el salón.");
assert.equal(say("access.captions"), "Dejar los subtítulos hasta la siguiente pregunta");
assert.equal(say("access.calm"), "Modo tranquilo (sin saltos ni destellos)");
assert.equal(say("access.contrast"), "Alto contraste");
assert.equal(say("access.size"), "Tamaño del texto en el pizarrón");
assert.equal(say("access.rate"), "Velocidad de la voz");
assert.equal($("access-size").getAttribute("aria-label"), "Tamaño del texto en el pizarrón");
assert.equal($("access-rate").getAttribute("aria-label"), "Velocidad de la voz");
assert.deepEqual(["normal", "large", "xlarge"].map((s) => $(`access-size-${s}`).textContent), ["Normal", "Grande", "Muy grande"]);
assert.deepEqual(["slower", "normal", "faster"].map((r) => $(`access-rate-${r}`).textContent), ["Más lenta", "Normal", "Más rápida"]);
""",
    )


@needs_node
def test_accessibility_card_polls_only_without_a_live_socket(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "precharge.html",
        ACCESS_DASHBOARD
        + TIMERS
        + """
// No socket.io: the 2 s status poll is the only news of a change made on a phone.
let room = { ...ACCESS };
installFetch({ "GET /api/accessibility": () => ({ status: 200, body: room }) });
await import("./dashboard/main.js");
await settle();
assert.deepEqual(card(), shows(ACCESS));
room = { ...ACCESS, text_size: "large" };
runTimer(2000);
await settle();
assert.deepEqual(card(), shows(room));
""",
    )
    run_node(
        tmp_path / "live",
        "precharge.html",
        ACCESS_DASHBOARD
        + TIMERS
        + """
// A live socket brings every change as an event, so the poll leaves the card alone.
const sock = installIo();
const http = installFetch({ "GET /api/accessibility": { status: 200, body: ACCESS } });
await import("./dashboard/main.js");
sock.fire("connect");
await settle();
const asked = () => http.calls.filter((c) => c.path === "/api/accessibility").length;
const seen = asked();
runTimer(2000);
await settle();
assert.equal(asked(), seen);
""",
    )


@needs_node
def test_camera_view_toggle_only_switches_its_own_buttons(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "precharge.html",
        ACCESS_DASHBOARD
        + """
installIo();
installFetch({ "GET /api/accessibility": { status: 200, body: ACCESS } });
// The template nests the camera's two buttons in #camera-view; the fake DOM lists every node flat, so nest them.
const [linedUp, raw] = [...doc.body.descendants()].filter((n) => Object.hasOwn(n.dataset, "view"));
$("camera-view").append(linedUp, raw);
await import("./dashboard/main.js");
await settle();
await raw.fire("click");
assert.deepEqual([linedUp, raw].map((b) => b.getAttribute("aria-pressed")), ["false", "true"]);
// The accessibility groups are .seg too, and keep their own state and their one handler.
assert.equal($("access-size-normal").getAttribute("aria-pressed"), "true");
assert.equal($("access-size-large").listeners.click.length, 1);
""",
    )
```

- [ ] **Step 2: Run** `.venv/bin/python -m pytest -q tests/test_frontend.py -k "accessibility_card or camera_view_toggle"` → FAIL, 7 failed. The template has no `#access-card` or `#camera-view` yet, so node throws errors such as `Cannot read properties of null (reading 'hidden')`, `(reading 'fire')`, `(reading 'checked')` and `(reading 'append')`, and the Spanish test fails with `Cannot read properties of undefined (reading 'textContent')`.

- [ ] **Step 3: Implement**

**3a. `biobench_sar/templates/precharge.html`**

Give the camera's own group an id (line 166). Old:
```html
            <div class="seg" role="group" aria-label="Camera view" data-i18n-aria-label="more.cameraView">
```
New:
```html
            <div class="seg" id="camera-view" role="group" aria-label="Camera view" data-i18n-aria-label="more.cameraView">
```
Put the card at the top of the last `.more-col`, above **Settings** (lines 180–181). Old:
```html
          <div class="more-col">
            <h3 data-i18n="more.settings">Settings</h3>
```
New:
```html
          <div class="more-col">
            {# The room's settings, for every page: hidden until GET /api/accessibility answers (older servers have none). #}
            <section class="access-card" id="access-card" aria-labelledby="access-title" hidden>
              <h3 id="access-title" data-i18n="access.title">Accessibility</h3>
              <p class="hint small" data-i18n="access.hint">These change the projector and the voice for the whole room.</p>
              <label class="check-line"><input type="checkbox" id="access-captions"> <span data-i18n="access.captions">Keep captions up until the next question</span></label>
              <label class="check-line"><input type="checkbox" id="access-calm"> <span data-i18n="access.calm">Calm mode (no bouncing or flashing)</span></label>
              <label class="check-line"><input type="checkbox" id="access-contrast"> <span data-i18n="access.contrast">High contrast</span></label>
              <p class="seg-label" data-i18n="access.size">Text size on the board</p>
              <div class="seg" id="access-size" role="group" aria-label="Text size on the board" data-i18n-aria-label="access.size">
                <button type="button" id="access-size-normal" aria-pressed="true" data-access-size="normal" data-i18n="access.size.normal">Normal</button>
                <button type="button" id="access-size-large" aria-pressed="false" data-access-size="large" data-i18n="access.size.large">Large</button>
                <button type="button" id="access-size-xlarge" aria-pressed="false" data-access-size="xlarge" data-i18n="access.size.xlarge">Extra large</button>
              </div>
              <p class="seg-label" data-i18n="access.rate">Voice speed</p>
              <div class="seg" id="access-rate" role="group" aria-label="Voice speed" data-i18n-aria-label="access.rate">
                <button type="button" id="access-rate-slower" aria-pressed="false" data-access-rate="slower" data-i18n="access.rate.slower">Slower</button>
                <button type="button" id="access-rate-normal" aria-pressed="true" data-access-rate="normal" data-i18n="access.rate.normal">Normal</button>
                <button type="button" id="access-rate-faster" aria-pressed="false" data-access-rate="faster" data-i18n="access.rate.faster">Faster</button>
              </div>
              <p class="note" id="access-msg" role="status"></p>
            </section>
            <h3 data-i18n="more.settings">Settings</h3>
```

**3b. `biobench_sar/static/js/dashboard/accessibility.js`** (new, complete)
```js
import { $, request, setNote } from "./api.js";
import { t } from "../i18n.js";
import { logEvent } from "./log.js";

// "Accessibility" under More tools. The settings belong to the room, not to this browser: the projector, the voice
// and every open dashboard follow them (the "accessibility" event), so a joined phone may change them too.
const SIZES = ["normal", "large", "xlarge"];
const RATES = ["slower", "normal", "faster"];
// The on/off settings and their checkboxes.
const BOXES = { captions_stay: "access-captions", calm: "access-calm", high_contrast: "access-contrast" };

// Bumped on every render, so a GET that was in flight during a newer change or push can't roll the card back.
let version = 0;
// What the room has, as the server last said; a failed change shows this again.
let last = null;

function controls() {
  return [
    ...Object.values(BOXES).map((id) => $(id)),
    ...SIZES.map((s) => $(`access-size-${s}`)),
    ...RATES.map((r) => $(`access-rate-${r}`)),
  ];
}

// GET, PUT and the event all send the five settings; anything less (another server version) is left alone.
function complete(body) {
  return (
    Boolean(body) &&
    Object.keys(BOXES).every((key) => typeof body[key] === "boolean") &&
    SIZES.includes(body.text_size) &&
    RATES.includes(body.speech_rate)
  );
}

function draw(body) {
  $("access-card").hidden = false;
  for (const [key, id] of Object.entries(BOXES)) $(id).checked = body[key];
  for (const s of SIZES) $(`access-size-${s}`).setAttribute("aria-pressed", String(s === body.text_size));
  for (const r of RATES) $(`access-rate-${r}`).setAttribute("aria-pressed", String(r === body.speech_rate));
}

// A /api/accessibility body from GET, PUT or the "accessibility" event.
export function renderAccessibility(body) {
  if (!complete(body)) return;
  version += 1;
  last = body;
  draw(body);
}

export async function refreshAccessibility() {
  const seen = version;
  const body = await request("/api/accessibility");
  if (seen === version) renderAccessibility(body);
}

// Only the setting that changed is sent, so two people changing different settings never undo each other.
async function change(key, value) {
  if (last && last[key] === value) return;
  const all = controls();
  for (const c of all) c.disabled = true;
  try {
    renderAccessibility(await request("/api/accessibility", { method: "PUT", json: { [key]: value } }));
    setNote($("access-msg"), "");
  } catch (err) {
    logEvent("accessibility", err.message);
    setNote($("access-msg"), () => t("access.failed", { hint: t("hint.offline") }), "warn");
    // A checkbox has already ticked itself: show what the room really has.
    if (last) draw(last);
  } finally {
    for (const c of all) c.disabled = false;
  }
}

export function initAccessibility() {
  for (const [key, id] of Object.entries(BOXES)) $(id).addEventListener("change", () => change(key, $(id).checked));
  for (const s of SIZES) $(`access-size-${s}`).addEventListener("click", () => change("text_size", s));
  for (const r of RATES) $(`access-rate-${r}`).addEventListener("click", () => change("speech_rate", r));
  // An older server has no accessibility API: the card simply stays hidden.
  refreshAccessibility().catch(() => {});
}
```

**3c. `biobench_sar/static/js/dashboard/main.js`** (five edits)

Line 6. Old:
```js
import { initLanguage, languageVersion, refreshLanguage, renderLanguage } from "./language.js";
```
New:
```js
import { initLanguage, languageVersion, refreshLanguage, renderLanguage } from "./language.js";
import { initAccessibility, refreshAccessibility, renderAccessibility } from "./accessibility.js";
```
The socket `connect` handler, line 64. Old:
```js
    refreshLanguage().catch(() => {});
    // A restarted server has a new PIN.
```
New:
```js
    refreshLanguage().catch(() => {});
    refreshAccessibility().catch(() => {});
    // A restarted server has a new PIN.
```
After the `language` handler, lines 99–103. Old:
```js
  socket.on("language", (d) => {
    renderLanguage(d);
    logEvent("language", d);
  });
}
```
New:
```js
  socket.on("language", (d) => {
    renderLanguage(d);
    logEvent("language", d);
  });
  // Someone changed the room's accessibility settings: here, on another dashboard or on a phone.
  socket.on("accessibility", (d) => {
    renderAccessibility(d);
    logEvent("accessibility", d);
  });
}
```
Boot, line 140. Old:
```js
initLanguage();
initTabs();
```
New:
```js
initLanguage();
initAccessibility();
initTabs();
```
Status poll, line 163. Old:
```js
  if (!socketLive) refreshTutor().catch(() => {});
});
```
New:
```js
  if (!socketLive) refreshTutor().catch(() => {});
  if (!socketLive) refreshAccessibility().catch(() => {});
});
```

**3d. `biobench_sar/static/js/dashboard/camera.js`** (`initCamera`, lines 104–110). Old:
```js
  for (const button of document.querySelectorAll(".seg button")) {
    button.addEventListener("click", () => {
      warped = button.dataset.view === "1";
      for (const b of document.querySelectorAll(".seg button")) b.setAttribute("aria-pressed", String(b === button));
      $("ink-overlay").replaceChildren();
    });
  }
```
New:
```js
  // Only the camera's own toggle: More tools has other segmented groups (Accessibility).
  const views = [...$("camera-view").querySelectorAll("button")];
  for (const button of views) {
    button.addEventListener("click", () => {
      warped = button.dataset.view === "1";
      for (const b of views) b.setAttribute("aria-pressed", String(b === button));
      $("ink-overlay").replaceChildren();
    });
  }
```

**3e. `biobench_sar/static/css/dashboard.css`**. Add a block after the "more tools" rules, before the phones block (lines 294–296). None of this goes inside the `@media (min-width: 761px) and (min-height: 480px)` block, so `test_dashboard_steps_fit_one_laptop_screen` reads the same rules as before. Old:
```css
.log .ev { color: var(--accent); font-weight: 700; margin-right: var(--s2); }

/* ---- more tools: phones and tablets ---------------------------------------- */
```
New:
```css
.log .ev { color: var(--accent); font-weight: 700; margin-right: var(--s2); }

/* ---- more tools: accessibility (the whole room) ------------------------------------ */
.access-card[hidden] { display: none; }
.access-card .hint.small { margin: 0 0 var(--s1); }
.access-card .check-line input { flex: none; }
.access-card .check-line input:disabled, .access-card .seg button:disabled { cursor: progress; }
.seg-label { margin: var(--s2) 0 var(--s1); font-weight: 700; font-size: 0.9rem; }
/* Three buttons share a narrow column: a long label ("Extra large", "Más rápida") wraps inside its button
   instead of pushing the group past the column's edge. */
.access-card .seg { display: flex; max-width: 100%; }
.access-card .seg button { flex: 1 1 auto; min-width: 0; padding: 4px var(--s2); font-size: 0.88rem; line-height: 1.2; }

/* ---- more tools: phones and tablets ---------------------------------------- */
```

**3f. `biobench_sar/static/js/i18n.js`**

English, lines 268–270. Old:
```js
  "value.tts_backend.none": "Off (answers are only written)",

  // ---- more tools: phones and tablets (LAN mode, on the laptop only)
```
New:
```js
  "value.tts_backend.none": "Off (answers are only written)",

  // ---- more tools: accessibility (the room's settings: the projector, the voice and every open dashboard)
  "access.title": "Accessibility",
  "access.hint": "These change the projector and the voice for the whole room.",
  "access.captions": "Keep captions up until the next question",
  "access.calm": "Calm mode (no bouncing or flashing)",
  "access.contrast": "High contrast",
  "access.size": "Text size on the board",
  "access.size.normal": "Normal",
  "access.size.large": "Large",
  "access.size.xlarge": "Extra large",
  "access.rate": "Voice speed",
  "access.rate.slower": "Slower",
  "access.rate.normal": "Normal",
  "access.rate.faster": "Faster",
  "access.failed": "Couldn't change that setting. {hint}",

  // ---- more tools: phones and tablets (LAN mode, on the laptop only)
```
Spanish, lines 583–585. Old:
```js
  "value.tts_backend.none": "Apagada (las respuestas solo se escriben)",

  "phone.title": "Usar un celular o una tableta",
```
New:
```js
  "value.tts_backend.none": "Apagada (las respuestas solo se escriben)",

  "access.title": "Accesibilidad",
  "access.hint": "Cambian el proyector y la voz para todo el salón.",
  "access.captions": "Dejar los subtítulos hasta la siguiente pregunta",
  "access.calm": "Modo tranquilo (sin saltos ni destellos)",
  "access.contrast": "Alto contraste",
  "access.size": "Tamaño del texto en el pizarrón",
  "access.size.normal": "Normal",
  "access.size.large": "Grande",
  "access.size.xlarge": "Muy grande",
  "access.rate": "Velocidad de la voz",
  "access.rate.slower": "Más lenta",
  "access.rate.normal": "Normal",
  "access.rate.faster": "Más rápida",
  "access.failed": "No se pudo cambiar ese ajuste. {hint}",

  "phone.title": "Usar un celular o una tableta",
```

- [ ] **Step 4: Update an existing exact assertion.** `test_dictionary_has_every_text_in_both_languages` requires every Spanish string to differ from the English one unless its key is in `SAME_IN_SPANISH`. "Normal" is the same in both languages, so add the two keys. This is `tests/test_frontend.py:1183`. Old:
```python
SAME_IN_SPANISH = {"lang.en", "lang.es", "lang.auto", "mode.quiz", "files.meta", "board.latency"}
```
New (the layout `ruff format` produces):
```python
SAME_IN_SPANISH = {
    "lang.en",
    "lang.es",
    "lang.auto",
    "mode.quiz",
    "files.meta",
    "board.latency",
    "access.size.normal",
    "access.rate.normal",
}
```

- [ ] **Step 5: Run** `.venv/bin/python -m pytest -q tests/test_frontend.py -k "accessibility_card or camera_view_toggle"` → PASS (7 passed). Also run these, which must PASS:
  - `.venv/bin/python -m pytest -q tests/test_frontend.py tests/test_mobile.py tests/test_app.py tests/test_lan.py tests/test_lan_wiring.py`. These cover the template, dictionary and script-key checks (`test_every_visible_template_text_has_a_dictionary_key`, `test_template_english_is_the_dictionary_english`, `test_every_key_the_scripts_name_exists`, `test_dictionary_has_every_text_in_both_languages`, `test_spanish_is_mexican_classroom_spanish`), the Spanish sweep (`test_dashboard_in_spanish_leaves_no_english_behind`: none of the new Spanish strings matches `ENGLISH_WORDS`, and "Normal" is not in it), `test_dashboard_steps_fit_one_laptop_screen`, the camera-box test and `test_service_worker_shell_lists_files_that_exist`.
  - `.venv/bin/ruff check . && .venv/bin/ruff format --check .`

  Verified on a copy of this branch with Task 4's `DEFAULT_ROUTES` line added: 494 passed, and ruff is clean.

- [ ] **Step 6: Commit** `feat(dashboard): Accessibility card under More tools changes the room's settings`
```bash
git add biobench_sar/templates/precharge.html biobench_sar/static/js/dashboard/accessibility.js \
  biobench_sar/static/js/dashboard/main.js biobench_sar/static/js/dashboard/camera.js \
  biobench_sar/static/css/dashboard.css biobench_sar/static/js/i18n.js tests/test_frontend.py
git commit -m "feat(dashboard): Accessibility card under More tools changes the room's settings" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 7: The README, build plan, contributing guide and phone guide describe the accessibility settings, and a docs test keeps them in line with the code

**Files:** Modify `docs/BUILD_PLAN.md` · Modify `README.md` · Modify `CONTRIBUTING.md` · Modify `docs/MOBILE.md` · Test `tests/test_frontend.py`

Depends on T2 (`biobench_sar/accessibility.py`, `SAR_ACCESSIBILITY_FILE` in `config.py`) and T6 (`access.title` in `static/js/i18n.js`).

**Pitfall:** the docs must never put a backtick right before a `-` in `README.md` or `docs/BUILD_PLAN.md`. `test_documented_benchmark_flags_exist` (tests/test_frontend.py:1774) reads every "`-x" as a `bench_vlm` flag. So the CSS variable is written `` `var(--text-scale)` ``, and the `say` rate flag only appears inside a fenced code block with no backticks.

- [ ] **Step 1: Failing tests** (append to `tests/test_frontend.py` after `test_documented_settings_exist_in_config`, lines 1874–1883, and before the `# ---- final integration contract: …` banner at line 1886. No new top-level imports are needed: the import goes inside the function, as `test_contract_describes_the_integrated_quiz` already does. `strings`, `DOCS`, `ROOT` and `needs_node` exist already.)
```python
@needs_node
def test_docs_describe_the_accessibility_settings() -> None:
    from biobench_sar.accessibility import SPEECH_RATES, TEXT_SIZES, AccessibilitySettings

    en = strings()["en"]
    keys = list(AccessibilitySettings().to_dict())
    readme = DOCS["README.md"]
    assert f"**{en['tab.more']} → {en['access.title']}**" in readme, "the README should say where the card is"
    assert "| `SAR_ACCESSIBILITY_FILE` | `data/accessibility.json` |" in readme
    assert "`GET /api/accessibility`" in readme and "`PUT /api/accessibility`" in readme
    plan = DOCS["docs/BUILD_PLAN.md"]
    contract = plan.split("### M — `accessibility.py`")[1].split("\n### ")[0]
    assert [key for key in keys if key not in contract] == []
    assert [size for size in TEXT_SIZES if f'"{size}"' not in contract] == []
    assert [name for name, factor in SPEECH_RATES.items() if f'"{name}": {factor}' not in contract] == []
    assert "| GET | `/api/accessibility` |" in plan and "| PUT | `/api/accessibility` |" in plan
    assert f"`accessibility {{{', '.join(keys)}}}`" in plan, "the socket event should list every key"
    assert "`status`, `tutor`, `language` and `accessibility` to the new client" in plan
    assert "`accessibility_file: Path` — env `SAR_ACCESSIBILITY_FILE`" in plan
    modules = DOCS["CONTRIBUTING.md"].split("## Modules")[1]
    assert "`accessibility.py`" in modules and "`jsonfile.py`" in modules
    mobile = (ROOT / "docs" / "MOBILE.md").read_text(encoding="utf-8")
    works = mobile.split("## What works on a phone")[1].split("\n## ")[0]
    assert f"**{en['access.title']}**" in works, "MOBILE.md should say phones can change the card"
```
- [ ] **Step 2: Run** `.venv/bin/python -m pytest -q tests/test_frontend.py -k accessibility_settings` → FAIL. It stops at the first assertion with `AssertionError: the README should say where the card is` (`'**More tools → Accessibility**' in '# Hackaton …'`), because no doc mentions the card yet. I confirmed this in a scratch copy with T2/T6 stand-ins.

- [ ] **Step 3: Implement.** Every "Old" snippet below occurs exactly once in its file. I applied them all in a scratch copy and the full `tests/test_frontend.py` plus `tests/test_mobile.py` passed: 116 passed.

**3a. `docs/BUILD_PLAN.md` — Phase 1 table, new area M (line 50)**

Old:
````markdown
| L | Desktop builds | `packaging/**`, `paths.py`, `scripts/build_*`, `.github/workflows/build.yml` | F (`app.main`) ([DESKTOP_BUILDS.md](DESKTOP_BUILDS.md)) |
````
New:
````markdown
| L | Desktop builds | `packaging/**`, `paths.py`, `scripts/build_*`, `.github/workflows/build.yml` | F (`app.main`) ([DESKTOP_BUILDS.md](DESKTOP_BUILDS.md)) |
| M | Accessibility | `accessibility.py`, `jsonfile.py`, `tests/test_accessibility.py` | F wires it, D (`TTSEngine.set_rate`), E draws it on both pages |
````

**3b. `docs/BUILD_PLAN.md` — Shared contract, `accessibility_file` line (line 107)**

Old:
````markdown
  - `quiz_bank_file: Path` — env `SAR_QUIZ_BANK_FILE`, default `DATA_DIR / "quiz_bank.json"`.
````
New:
````markdown
  - `quiz_bank_file: Path` — env `SAR_QUIZ_BANK_FILE`, default `DATA_DIR / "quiz_bank.json"`.
  - `accessibility_file: Path` — env `SAR_ACCESSIBILITY_FILE`, default `DATA_DIR / "accessibility.json"`: the
    room's accessibility settings (contract M).
````

**3c. `docs/BUILD_PLAN.md` — contract D, `TTSEngine.set_rate` (lines 286–287, end of the D code block)**

Old:
````markdown
        # falls back edge → say → pyttsx3 → none
```
````
New:
````markdown
        # falls back edge → say → pyttsx3 → none
    def set_rate(self, factor: float) -> None  # voice speed (contract M), clamped to 0.5–2.0
        # edge: rate="+25%" / "-25%"; say: "-r", round(175 × factor); pyttsx3: the engine's own rate × factor.
        # At 1.0 nothing extra is passed, so the commands are the same as without it.
```
````

**3d. `docs/BUILD_PLAN.md` — new contract M after K (lines 419–421)**

Old:
````markdown
`HttpOnly`, `SameSite=Lax`, `Max-Age` 12 h), `PhoneSessions`, `SocketIOGate`, `local_ip_addresses`, `join_url`.

### E/F — HTTP + Socket.IO contract
````
New:
````markdown
`HttpOnly`, `SameSite=Lax`, `Max-Age` 12 h), `PhoneSessions`, `SocketIOGate`, `local_ip_addresses`, `join_url`.

### M — `accessibility.py`, `jsonfile.py` (accessibility for the whole room; the routes and the pages are in E/F)
```python
TEXT_SIZES = ("normal", "large", "xlarge")                       # board text 1× / 1.25× / 1.5×
SPEECH_RATES = {"slower": 0.75, "normal": 1.0, "faster": 1.25}  # voice speed factor
@dataclass(frozen=True)
class AccessibilitySettings:
    captions_stay: bool = False; text_size: str = "normal"; speech_rate: str = "normal"; calm: bool = False
    high_contrast: bool = False
    def to_dict(self) -> dict[str, bool | str]  # all five keys, in this order
    rate_factor: float                          # property: SPEECH_RATES[speech_rate]
def parse_changes(body: object) -> dict[str, bool | str]
    # a JSON object holding a non-empty subset of the five keys. Booleans must be real JSON booleans and strings
    # match exactly (case-sensitive, like /api/language). Anything else raises ValueError with an English message
    # naming the key and its allowed values ("text_size must be one of: normal, large, xlarge").
class AccessibilityStore:  # persisted as JSON at Settings.accessibility_file; thread-safe
    def __init__(self, path: Path) -> None  # loads once; a missing or broken file gives the defaults, a bad key its own
    current: AccessibilitySettings         # property
    def update(self, changes: Mapping[str, bool | str]) -> AccessibilitySettings
        # changes already parsed; saves, and a failed save logs a warning and keeps the new value in memory
# jsonfile.py: the one atomic JSON write (mkstemp in the same folder, then os.replace)
def save_json(path: Path, payload: object, what: str) -> bool
    # False, the temp file removed and the warning "could not save {what} to {path}: {exc}" on OSError.
    # DocumentSelection ("document selection") and QuizBank ("quiz bank") use it too. Calibration.save keeps its
    # own write, because it raises on failure on purpose.
```
`Assistant` owns the store (`assistant.accessibility`). `accessibility_body()` is the five keys.
`set_accessibility(body)` parses, saves, calls `push_speech_rate()` and emits `accessibility` to every client; a
`ValueError` becomes the route's 400, with nothing saved or pushed. `push_speech_rate()` hands `rate_factor` to
`TTSEngine.set_rate` (contract D) through a `getattr(tts, "set_rate", None)` + `callable` guard, once when the
services are attached and again on every change. `estimate_speech_seconds(text, rate)` divides the spoken part by
the factor, so slower speech keeps the wake-word loop deaf for longer.

### E/F — HTTP + Socket.IO contract
````

**3e. `docs/BUILD_PLAN.md` — route rows next to `/api/language` (line 429)**

Old:
````markdown
| PUT | `/api/language` | `{"language": "auto" \| "en" \| "es"}` → the `GET` body; 400 otherwise. Also pushed as a `language` event |
````
New:
````markdown
| PUT | `/api/language` | `{"language": "auto" \| "en" \| "es"}` → the `GET` body; 400 otherwise. Also pushed as a `language` event |
| GET | `/api/accessibility` | `{"captions_stay": bool, "text_size": "normal" \| "large" \| "xlarge", "speech_rate": "slower" \| "normal" \| "faster", "calm": bool, "high_contrast": bool}` (contract M) |
| PUT | `/api/accessibility` | any non-empty subset of those keys → the full `GET` body, saved to `SAR_ACCESSIBILITY_FILE`; 400 `{"error": str}` otherwise, with nothing saved or pushed. Behind the PIN gate like every route, and phones may change it. Also pushed as an `accessibility` event |
````

**3f. `docs/BUILD_PLAN.md` — socket event list and the on-connect list (lines 471–474)**

Old:
````markdown
`language {setting, current}`. Both pages show `display_text` in place of the spoken words.
On connect the server sends the current `status`, `tutor` and `language` to the new client. The dashboard re-reads
`/api/config` on every connect, and also takes the subject from `/api/status` `profile`, so a server restarted with
another `SAR_PROFILE` is followed without a reload. The math mode picker stays hidden until the subject is known.
````
New:
````markdown
`language {setting, current}`, `accessibility {captions_stay, text_size, speech_rate, calm, high_contrast}` (after
every change). Both pages show `display_text` in place of the spoken words.
On connect the server sends the current `status`, `tutor`, `language` and `accessibility` to the new client. The
dashboard re-reads `/api/config` on every connect, and also takes the subject from `/api/status` `profile`, so a
server restarted with another `SAR_PROFILE` is followed without a reload. The math mode picker stays hidden until
the subject is known.
````

**3g. `docs/BUILD_PLAN.md` — Projector UI paragraph, then the dashboard card (lines 490–491, end of "Projector UI: …")**

Old:
````markdown
anyway. When `highlight` is set, the page draws a glowing outline around that box, pulsing for about 2 s and
fading after about 8 s or at the next update.
````
New:
````markdown
anyway. When `highlight` is set, the page draws a glowing outline around that box, pulsing for about 2 s and
fading after about 8 s or at the next update.

Accessibility on the projector (contract M): `<body>` carries `data-text-size="normal|large|xlarge"`,
`data-captions="stay|timed"`, `data-calm="on|off"` and `data-contrast="on|off"`. `page()` renders them from
`access` (the `GET /api/accessibility` body), so the first paint is already right. The `accessibility` event and a
`GET /api/accessibility` on every reconnect keep them current, with a version counter like the language's. Every
change re-fits the bubble and sends `ui_rects` again.
- **Captions stay up**: the bubble gets no hide timer. The next answer replaces it, and a real new question
  (`transcript`, or status `thinking`) clears it, so it never covers the board while the camera reads it. The
  heard words stay until the answer arrives instead of 5 s. Switching captions off while one is held hides it
  after 4 s.
- **Text size**: `var(--text-scale)` (1 / 1.25 / 1.5) multiplies the font sizes of the bubble, the quiz card's
  label and problem, the transcript and the status label. The bubble's width grows with it, up to `50vw`.
- **Calm mode**: the `prefers-reduced-motion` rules (no bob, hop, shake, nudge, highlight pulse, status ring or
  dots), the avatar gliding on `left`/`top` only, and softer glows. The calibration checkerboard is untouched.
- **High contrast**: a 6 px bubble border and highlight outline (the JS geometry follows), pure white text with no
  blurred glow, a dashed error border and a double celebrating border, so right and wrong differ by more than
  colour. The quiz card's text goes to full opacity, but its border stays at most 2 px with no glow and its width
  at most `30vw`, because the camera reads thick bright lines as ink.

The dashboard's **Accessibility** card (`static/js/dashboard/accessibility.js`, at the top of the third **More
tools** column) stays hidden until `GET /api/accessibility` answers. A change sends `PUT /api/accessibility` with
only that key, and the card follows the `accessibility` event (the 2 s status poll refreshes it while the socket
is down).
````

**3h. `docs/BUILD_PLAN.md` — Implementation-notes row (line 512)**

Old:
````markdown
| English only | English and Spanish end to end (`lang.py`, `mathspeak_es.py`, es-MX voices, translated dashboard) | The classroom is bilingual |
````
New:
````markdown
| English only | English and Spanish end to end (`lang.py`, `mathspeak_es.py`, es-MX voices, translated dashboard) | The classroom is bilingual |
| Nothing on accessibility (both specs) | An **Accessibility** card for the whole room: captions that stay up, text size on the board, voice speed, calm mode and high contrast (contract M, saved in `SAR_ACCESSIBILITY_FILE`) | A student with a disability uses the same board. It is additive: the defaults are exactly the specs' behaviour |
````

**3i. `docs/BUILD_PLAN.md` — Status row (line 581)**

Old:
````markdown
| L Desktop builds | ✅ Windows installer + macOS arm64 app, smoke-tested in CI (#66) | the launcher's self-test prints `SELF-TEST OK` |
````
New:
````markdown
| L Desktop builds | ✅ Windows installer + macOS arm64 app, smoke-tested in CI (#66) | the launcher's self-test prints `SELF-TEST OK` |
| M Accessibility | ✅ built: captions stay up, text size, voice speed (edge, `say`, pyttsx3), calm mode, high contrast; one atomic JSON write (`jsonfile.py`) shared with the file selection and the quiz bank | `tests/test_accessibility.py`; routes, push and on-connect in `tests/test_app.py`; the voice in `tests/test_tts.py`; both pages under node in `tests/test_frontend.py` |
````

**3j. `README.md` — Accessibility section between "English and Spanish" and "The dashboard" (lines 223–225)**

Old:
````markdown
  `biobench_sar/lang.py`. Math is spoken in Spanish words, and quiz answers in Spanish words are graded.

## The dashboard
````
New:
````markdown
  `biobench_sar/lang.py`. Math is spoken in Spanish words, and quiz answers in Spanish words are graded.

## Accessibility

A teacher can make the assistant easier to follow for a student with a disability from the dashboard's
**More tools → Accessibility** card, on the laptop or a joined phone. The settings apply to the whole room at
once (the projector, the voice and every open dashboard) and are kept in `SAR_ACCESSIBILITY_FILE`.

- **Keep captions up until the next question** (deaf and hard-of-hearing students, slower readers): the speech
  bubble stays until the next answer instead of fading, and clears as soon as a new question starts.
- **Text size on the board**: Normal, Large (1.25×) or Extra large (1.5×) for the bubble, the quiz card, the
  heard words and the status.
- **Voice speed**: Slower (0.75×), Normal or Faster (1.25×), for every voice (Edge, `say` and pyttsx3).
- **Calm mode** (autism, ADHD, motion or light sensitivity): no bouncing, shaking or pulsing, and softer glows.
  Pip still glides to the ink it talks about.
- **High contrast** (low vision, colour-blind students): thicker outlines and pure white text. A wrong answer's
  bubble has a dashed border and a right one's a double border, so they differ by more than colour.

Over HTTP: `GET /api/accessibility` and `PUT /api/accessibility` with only the keys to change, e.g.
`{"speech_rate": "slower"}`. Every change reaches the pages as an `accessibility` Socket.IO event.

## The dashboard
````
(This lands inside the `### Quiz banks` … `## The dashboard` slice that `test_readme_worksheet_rules_match_the_parser` reads. It adds no "says so**:" text and no ```` ```md ```` block, so that test's first matches don't change.)

**3k. `README.md` — the dashboard section names the card (lines 238–239)**

Old:
````markdown
4. **More tools**: the lined-up or raw camera view, detected drawings, file search, settings, the activity
   log and, in phone mode, the **Use a phone or tablet** card.
````
New:
````markdown
4. **More tools**: the lined-up or raw camera view, detected drawings, file search, the **Accessibility** card,
   settings, the activity log and, in phone mode, the **Use a phone or tablet** card.
````

**3l. `README.md` — Files row (line 312)**

Old:
````markdown
| `SAR_QUIZ_BANK_FILE` | `data/quiz_bank.json` | Quiz problems parsed from uploaded CSVs and worksheets |
````
New:
````markdown
| `SAR_QUIZ_BANK_FILE` | `data/quiz_bank.json` | Quiz problems parsed from uploaded CSVs and worksheets |
| `SAR_ACCESSIBILITY_FILE` | `data/accessibility.json` | Accessibility settings for the room (captions, text size, voice speed, calm mode, high contrast) |
````

**3m. `README.md` — Repository map: the new modules go in, as CONTRIBUTING's rule 5 asks (lines 340, 344, 345)**

Old (line 340):
````markdown
| `biobench_sar/assistant.py` | Sense → Think → Act pipeline, status, history, file selection, language, target snapping, UI masking |
````
New:
````markdown
| `biobench_sar/assistant.py` | Sense → Think → Act pipeline, status, history, file selection, language, accessibility, target snapping, UI masking |
````
Old (line 344):
````markdown
| `biobench_sar/lan.py` | Phones on the Wi-Fi: addresses, the join PIN, device cookies, the Socket.IO gate |
````
New:
````markdown
| `biobench_sar/lan.py` | Phones on the Wi-Fi: addresses, the join PIN, device cookies, the Socket.IO gate |
| `biobench_sar/accessibility.py` | The room's accessibility settings (captions, text size, voice speed, calm mode, high contrast): checked, saved and loaded |
````
Old (line 345):
````markdown
| `biobench_sar/config.py`, `envfile.py`, `paths.py`, `schemas.py` | Settings, `sar.env` loading, where files live (checkout or desktop build), shared types |
````
New:
````markdown
| `biobench_sar/config.py`, `envfile.py`, `jsonfile.py`, `paths.py`, `schemas.py` | Settings, `sar.env` loading, the one atomic JSON write, where files live (checkout or desktop build), shared types |
````

**3n. `CONTRIBUTING.md` — cleanup rule 2 names the shared JSON write (line 30)**

Old:
````markdown
   and import it. One home each: shared text like `NO_ANSWER_TEXT`, `config.PROFILES`, one atomic JSON write.
````
New:
````markdown
   and import it. One home each: shared text like `NO_ANSWER_TEXT`, `config.PROFILES`, one atomic JSON write
   (`jsonfile.save_json`).
````

**3o. `CONTRIBUTING.md` — modules table (lines 48 and 50)**

Old (line 48):
````markdown
| Server | `app.py` (routes, Socket.IO, the phone PIN gate), `assistant.py` (Sense → Think → Act pipeline, status, history, file selection, language, target snapping, UI masking), `config.py`, `envfile.py` (`sar.env`), `paths.py` (checkout vs desktop build), `schemas.py` |
````
New:
````markdown
| Server | `app.py` (routes, Socket.IO, the phone PIN gate), `assistant.py` (Sense → Think → Act pipeline, status, history, file selection, language, accessibility, target snapping, UI masking), `config.py`, `envfile.py` (`sar.env`), `jsonfile.py` (`save_json`, the one atomic JSON write), `paths.py` (checkout vs desktop build), `schemas.py` |
````
Old (line 50):
````markdown
| Phones | `lan.py` (PIN, device cookies, host checks, Socket.IO gate), `templates/join.html`, `static/js/dashboard/phone.js`, `static/sw.js`, `static/manifest.webmanifest` |
````
New:
````markdown
| Phones | `lan.py` (PIN, device cookies, host checks, Socket.IO gate), `templates/join.html`, `static/js/dashboard/phone.js`, `static/sw.js`, `static/manifest.webmanifest` |
| Accessibility | `accessibility.py` (the room's settings: captions, text size, voice speed, calm mode, high contrast), `static/js/dashboard/accessibility.js` (the dashboard card); the projector side is in `static/css/style.css` and `static/js/main.js` |
````

**3p. `docs/MOBILE.md` — "What works on a phone" bullet (lines 32–33)**

Old:
````markdown
- **English or Spanish**: the join page follows the phone's language, and the dashboard has the same
  English · Español · Auto switch.
````
New:
````markdown
- **English or Spanish**: the join page follows the phone's language, and the dashboard has the same
  English · Español · Auto switch.
- **Accessibility** (under **More tools**): captions that stay up, text size on the board, voice speed, calm mode
  and high contrast. A change applies to the whole room at once (the projector, the voice and every open
  dashboard), so a teacher can set it from a phone without going back to the laptop.
````

**3q. `docs/MOBILE.md` — Known limits: what a joined device can do (lines 156–157)**

Old:
````markdown
  - **A device with the cookie** has the dashboard's full powers: files, modes and quizzes. It never sees the
    PIN, because `/api/lan` answers only the laptop.
````
New:
````markdown
  - **A device with the cookie** has the dashboard's full powers: files, modes, quizzes and the accessibility
    settings. It never sees the PIN, because `/api/lan` answers only the laptop.
````

- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q tests/test_frontend.py -k accessibility_settings` → PASS. Then run the existing doc checks these edits touch:
  - `.venv/bin/python -m pytest -q tests/test_frontend.py -k "doc or readme or contract or settings_exist or knowledge_paths or cleanup or claude_md or benchmark or files_tab"` → PASS. This covers `test_documented_benchmark_flags_exist` (no backtick before `-`), `test_documented_settings_exist_in_config` (needs T2's `"SAR_ACCESSIBILITY_FILE"` in `config.py`), `test_readme_worksheet_rules_match_the_parser`, `test_docs_call_the_files_tab_by_its_name_in_each_subject`, `test_documented_knowledge_paths_exist`, `test_contract_describes_the_integrated_quiz` and `test_agent_docs_point_at_the_cleanup_pass`.
  - `.venv/bin/python -m pytest -q tests/test_mobile.py -k mobile_doc` → PASS. Every needle in `test_mobile_doc_matches_the_code` (tests/test_mobile.py:485) and `test_mobile_doc_states_the_real_guessing_limits` is untouched.
  - Then `.venv/bin/python -m pytest -q tests/test_frontend.py tests/test_mobile.py` → PASS (about 6 min under node) and `.venv/bin/ruff check . && .venv/bin/ruff format --check .` → clean. The new test is already ruff-formatted.

- [ ] **Step 5: Existing exact-assertion tests.** None break. The edits keep every pinned string: `"### Quiz banks"`, `"## The dashboard"`, `"| `SAR_QUIZ_LENGTH` | `10` |"`, `` "`seed_marker_file: Path`" ``, `"## Cleanup pass (every branch and PR)"`, `"Both profiles share one knowledge base"`, and all of MOBILE.md's needles. The edits also add no `knowledge/…` path and no `bench_vlm` line, so there is no assertion to update.

- [ ] **Step 6: Commit** `docs(access): document the accessibility settings in the build plan, README, contributing guide and phone guide`
```bash
git add docs/BUILD_PLAN.md README.md CONTRIBUTING.md docs/MOBILE.md tests/test_frontend.py
git commit -m "docs(access): document the accessibility settings in the build plan, README, contributing guide and phone guide

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

### Task 8: Verify and open the PR

- [ ] Full `.venv/bin/python -m pytest -q` once, compared with the `release/candidate` baseline (5,293 passed; the
  timing tests `test_glare.py::test_fast_at_1080p` and `test_lang.py::test_lookback_is_linear[clauses]` fail there
  too under machine load).
- [ ] `.venv/bin/ruff check . && .venv/bin/ruff format --check .`
- [ ] Manual run: `SAR_VLM_BACKEND=mock SAR_TTS_BACKEND=none SAR_PORT=5051 .venv/bin/python -m biobench_sar`, open
  `/dashboard` → **More tools → Accessibility**, change each setting and watch `/` (the projector page) follow:
  text size, held caption after a typed question, calm mode (no bob), high contrast (thick borders).
- [ ] Reviewer subagent on `git diff release/candidate...HEAD`.
- [ ] CONTRIBUTING cleanup pass; leftovers in the PR's **Cleanup** section (`Calibration.save` keeps its own atomic
  write because it raises on failure).
- [ ] Push and open the PR into `release/candidate` (not `main`) once the user says so. Expect a small CSS conflict
  with the problem card's `#quiz-card` → `#problem-card` rename, whichever lands second.
