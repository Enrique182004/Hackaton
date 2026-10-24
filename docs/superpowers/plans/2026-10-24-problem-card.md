# Problem Card (PR 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The projector's card shows the problem being worked on in every tutor mode (quiz, practice, lesson
starter, help), in symbols.

**Architecture:** The model's reply carries the problem in symbols (`new_problem` for a dictated one,
`board_problem` for the student's). `VLMClient.evaluate` puts both on the returned `TutorAction` — code's own
problem when code replaced the model's words. The assistant stores the text in `TutorSession`, whose
`to_dict()` already reaches the projector (`tutor` event, `/api/tutor`); `#quiz-card` becomes `#problem-card`.

**Tech Stack:** Python 3.10–3.12, Flask-SocketIO, vanilla JS, pytest (+ node fake DOM for the frontend), ruff.

Spec: [2026-10-24-problem-card-design.md](../specs/2026-10-24-problem-card-design.md) (PR 1 section).
Branch: `feat/problem-card` from `release/candidate`.

---

## File map

| File | Change |
| --- | --- |
| `biobench_sar/tutor.py` | displayed problem: `show_problem`, `clear_problem`, `problem`, `to_dict()["problem"]` |
| `biobench_sar/llm/vlm_client.py` | `HELP_SCHEMA`, `LEARN_SCHEMA`, `new_problem` in practice; prompts en + es; `RawReply.new_problem`; `TutorAction.board_problem/new_problem`; `fallback_problem`; `_finish`, `_salvage`, `_mock` |
| `biobench_sar/assistant.py` | after a non-quiz math turn, `_show_problem(action)` |
| `biobench_sar/templates/index.html`, `static/js/main.js`, `static/css/style.css`, `static/js/i18n.js` | `#problem-card`, labels |
| `tests/test_tutor.py`, `tests/test_vlm_client.py`, `tests/test_assistant.py`, `tests/test_app.py`, `tests/test_frontend.py` | new tests; exact tutor dicts gain `"problem": None`; card ids renamed |

Commands: `.venv/bin/python -m pytest -q tests/<file>` · `.venv/bin/ruff check . && .venv/bin/ruff format --check .`

---

### Task 1: Tutor state holds the displayed problem

**Files:** Modify `biobench_sar/tutor.py` · Test `tests/test_tutor.py`

- [ ] **Step 1: Failing tests** (append to `tests/test_tutor.py`)

```python
# ---- session: the problem on the projector's card -------------------------------------------------------------


def test_show_problem_puts_it_in_the_dict() -> None:
    t = TutorSession()
    t.show_problem("  4x +  7 = 19 ", "dictated")
    assert t.problem == {"text": "4x + 7 = 19", "source": "dictated"}
    assert t.to_dict()["problem"] == {"text": "4x + 7 = 19", "source": "dictated"}


@pytest.mark.parametrize("text", ["", "   ", None])
def test_show_problem_ignores_empty_text(text: str | None) -> None:
    t = TutorSession()
    t.show_problem("2x + 3 = 11", "board")
    t.show_problem(text, "dictated")
    assert t.problem == {"text": "2x + 3 = 11", "source": "board"}


def test_show_problem_caps_long_text_at_a_word() -> None:
    t = TutorSession()
    t.show_problem("word " * 40, "board")
    text = t.problem["text"]
    assert len(text) <= MAX_PROBLEM_CHARS and text.endswith("…") and not text.endswith(" …")


def test_show_problem_rejects_unknown_source() -> None:
    with pytest.raises(ValueError):
        TutorSession().show_problem("x = 1", "guess")


def test_new_topic_and_quiz_clear_the_problem() -> None:
    t = TutorSession()
    t.show_problem("x + 3 = 8", "dictated")
    t.start_learning("fractions")
    assert t.problem is None
    t.show_problem("x + 3 = 8", "dictated")
    t.set_mode("quiz")
    assert t.problem is None


def test_running_quiz_owns_the_card() -> None:
    t = TutorSession()
    t.set_mode("quiz")
    t.start_quiz([], length=3)
    t.show_problem("x + 3 = 8", "board")
    assert t.to_dict()["problem"] is None
```

Import `MAX_PROBLEM_CHARS` from `biobench_sar.tutor` at the top of the file. In the two existing exact
assertions (`test_new_session`, line ~217, and line ~333) add `"problem": None` to the expected dict.

- [ ] **Step 2: Run** `.venv/bin/python -m pytest -q tests/test_tutor.py` → FAIL (`MAX_PROBLEM_CHARS` / `show_problem` missing).

- [ ] **Step 3: Implement** in `biobench_sar/tutor.py`

Module constants (next to `DEFAULT_QUIZ_LENGTH`):

```python
MAX_PROBLEM_CHARS = 80  # the projector card's problem, read from across the room
PROBLEM_SOURCES = ("dictated", "board")  # THEO read it out / copied from the student's writing
```

Helper (next to `_norm`):

```python
def _card_text(text: str | None) -> str | None:
    """`text` on one line, cut at a word to MAX_PROBLEM_CHARS (ending in …); None when nothing is left."""
    flat = " ".join(str(text or "").split())
    if len(flat) <= MAX_PROBLEM_CHARS:
        return flat or None
    cut = flat[: MAX_PROBLEM_CHARS - 1]
    space = cut.rfind(" ")
    return (cut[:space] if space > MAX_PROBLEM_CHARS // 2 else cut).rstrip() + "…"
```

`TutorSession.__init__`: `self._problem: dict[str, str] | None = None  # the projector card's, outside a quiz`.

`set_mode`: inside the lock, `if mode == "quiz": self._problem = None`.
`start_learning`: inside the lock, `self._problem = None` (a new topic starts with an empty card).

New section before `# ---- conversation memory`:

```python
    # ---- the problem on the projector's card ---------------------------------
    @property
    def problem(self) -> dict[str, str] | None:
        with self._lock:
            return dict(self._problem) if self._problem else None

    def show_problem(self, text: str | None, source: str) -> None:
        """Show `text` on the card; empty text keeps what it shows (a model that left the key out)."""
        if source not in PROBLEM_SOURCES:
            raise ValueError(f"source must be one of {', '.join(PROBLEM_SOURCES)}")
        clean = _card_text(text)
        if clean is None:
            return
        with self._lock:
            self._problem = {"text": clean, "source": source}

    def clear_problem(self) -> None:
        with self._lock:
            self._problem = None
```

`to_dict()` return: `"problem": None if self._quiz == "active" else self.problem` added after `"quiz": quiz`.

- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q tests/test_tutor.py` → PASS.

- [ ] **Step 5: Fix the other exact tutor dicts** — add `"problem": None` in `tests/test_assistant.py`
(lines ~682, ~794 block, ~933) and `tests/test_app.py` (lines ~530, ~871, ~881, ~907, ~923). Run
`.venv/bin/python -m pytest -q tests/test_assistant.py tests/test_app.py` → PASS.

- [ ] **Step 6: Commit** `feat(tutor): keep the problem shown on the projector card`

### Task 2: The model's reply carries the problem in symbols

**Files:** Modify `biobench_sar/llm/vlm_client.py` · Test `tests/test_vlm_client.py`

- [ ] **Step 1: Failing tests** (append to `tests/test_vlm_client.py`; import `HELP_SCHEMA`, `LEARN_SCHEMA`,
`PRACTICE_SCHEMA`, `MATH_PROMPTS_BY_LANGUAGE`, `fallback_problem`, `TutorAction`, `VLMClient`)

```python
# ---- the problem for the projector's card -------------------------------------------------------------------


@pytest.mark.parametrize("language", ["en", "es"])
def test_prompts_ask_for_the_problem_in_symbols(language: str) -> None:
    prompts = MATH_PROMPTS_BY_LANGUAGE[language]
    assert '"board_problem": str' in prompts["help"] and '"new_problem"' not in prompts["help"]
    assert '"board_problem": str, "new_problem": str' in prompts["practice"]
    assert '"new_problem": str' in prompts["learn"]
    assert '"new_problem"' not in prompts["quiz"] and '"board_problem"' not in prompts["quiz"]


def test_schemas() -> None:
    assert HELP_SCHEMA.startswith('{"board_problem": str, "spoken_text"')
    assert PRACTICE_SCHEMA.startswith('{"board_problem": str, "new_problem": str, "spoken_text"')
    assert LEARN_SCHEMA.startswith('{"new_problem": str, "spoken_text"')


def _client_replying(monkeypatch: pytest.MonkeyPatch, *replies: str) -> VLMClient:
    client = VLMClient("ollama", "m", "http://x", profile="math")
    queue = list(replies)
    monkeypatch.setattr(client, "_ollama", lambda *a, **k: queue.pop(0))
    return client


def _ask(client: VLMClient, mode: str, task: dict | None = None, query: str = "help me") -> TutorAction:
    action = client.evaluate(query, b"jpg", [], [], (1920, 1080), mode=mode, task=task)
    assert isinstance(action, TutorAction)
    return action


def test_help_reply_keeps_the_board_problem(monkeypatch: pytest.MonkeyPatch) -> None:
    reply = (
        '{"board_problem": "2x + 3 = 11", "spoken_text": "Subtract three from both sides.", '
        '"animation_state": "pointing", "target_x": 700, "target_y": 400}'
    )
    action = _ask(_client_replying(monkeypatch, reply), "help")
    assert (action.board_problem, action.new_problem) == ("2x + 3 = 11", None)


def test_practice_reply_keeps_the_new_problem(monkeypatch: pytest.MonkeyPatch) -> None:
    reply = (
        '{"board_problem": "2x + 3 = 11", "new_problem": "4x + 7 = 19", "spoken_text": "Try this one. Four x plus '
        'seven equals nineteen. Write it down and solve it.", "animation_state": "idle", "target_x": 960, '
        '"target_y": 540}'
    )
    action = _ask(_client_replying(monkeypatch, reply), "practice", {"kind": "practice"}, "another one")
    assert (action.board_problem, action.new_problem) == ("2x + 3 = 11", "4x + 7 = 19")


def test_missing_and_numeric_problem_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    reply = '{"board_problem": 12, "spoken_text": "Write the next step.", "animation_state": "idle", "target_x": 1, "target_y": 1}'
    action = _ask(_client_replying(monkeypatch, reply), "help")
    assert (action.board_problem, action.new_problem) == ("12", None)


def test_practice_read_back_shows_codes_problem(monkeypatch: pytest.MonkeyPatch) -> None:
    # Both replies read the board's problem back: code dictates a variant, and the card must show that one.
    back = (
        '{"board_problem": "2x + 3 = 11", "new_problem": "2x + 3 = 11", "spoken_text": "Two x plus three equals '
        'eleven.", "animation_state": "idle", "target_x": 960, "target_y": 540}'
    )
    action = _ask(_client_replying(monkeypatch, back, back), "practice", {"kind": "practice"}, "another one")
    assert action.new_problem and action.new_problem != "2x + 3 = 11"
    assert action.new_problem in (action.display_text or "")


def test_fallback_problem() -> None:
    assert fallback_problem("practice", {"kind": "practice"})
    assert fallback_problem("learn", {"kind": "learn", "stage": "example", "topic": "fractions"}) == "1/5 + 2/5"
    assert fallback_problem("learn", {"kind": "learn", "stage": "offer", "topic": "fractions"}) is None
    assert fallback_problem("help", None) is None


@pytest.mark.parametrize("language", ["en", "es"])
def test_mock_dictates_a_problem_for_the_card(language: str) -> None:
    client = VLMClient("mock", "m", "", profile="math")
    task = {"kind": "learn", "stage": "example", "topic": "fractions"}
    lesson = client.evaluate("make one up", None, [], [], (1920, 1080), mode="help", task=task, language=language)
    practice = client.evaluate("another", None, [], [], (1920, 1080), mode="practice", task={"kind": "practice"})
    assert lesson.new_problem == "1/5 + 2/5"
    assert practice.new_problem == fallback_problem("practice", {"kind": "practice"})
```

- [ ] **Step 2: Run** `.venv/bin/python -m pytest -q tests/test_vlm_client.py -k "problem or schemas"` → FAIL (imports).

- [ ] **Step 3: Implement** in `biobench_sar/llm/vlm_client.py`

Schemas (replace the `PRACTICE_SCHEMA` line and its comment):

```python
# Same for practice: code can then tell when the "new" problem is the board's one read back.
# The problem in symbols is also what the projector's card shows: new_problem for one THEO dictates, board_problem
# for the student's own.
PRACTICE_SCHEMA = MATH_SCHEMA.replace('{"spoken_text"', '{"board_problem": str, "new_problem": str, "spoken_text"')
HELP_SCHEMA = MATH_SCHEMA.replace('{"spoken_text"', '{"board_problem": str, "spoken_text"')
LEARN_SCHEMA = MATH_SCHEMA.replace('{"spoken_text"', '{"new_problem": str, "spoken_text"')
```

`_SCHEMAS = {"help": HELP_SCHEMA, "quiz": QUIZ_SCHEMA, "practice": PRACTICE_SCHEMA, "learn": LEARN_SCHEMA}`.

Prompt rules and examples:
- `MATH_HELP`: first bullet becomes `- FIRST copy the problem written on the board into board_problem, exactly ("" if the board is empty). Then read the student's latest step.`; each example JSON starts with `"board_problem": "<the board's problem>"` (`2x + 3 = 11`, `1/2 + 1/3`, `x^2 = 49`).
- `MATH_HELP_ES`: same — `- PRIMERO copia el problema escrito en el pizarrón en board_problem, tal cual ("" si el pizarrón está vacío). Luego lee el último paso del estudiante.`; examples likewise.
- `MATH_PRACTICE` gets, after the "Dictate only…" bullet: `- Write the new problem in symbols in new_problem ("4x + 7 = 19"): the same problem you dictate.`; examples add `"new_problem": "4x + 7 = 19"` / `"3/4 + 1/8"` after `board_problem`.
- `MATH_PRACTICE_ES`: `- Escribe el problema nuevo con símbolos en new_problem ("4x + 7 = 19"): el mismo que dictas.`; examples likewise.
- `MATH_LEARN` / `MATH_LEARN_ES` (f-strings: keep `{{ }}`): add the bullet `- new_problem: in stage example the starter problem in symbols ("1/4 + 2/4"); "" in every other stage.` / `- new_problem: en la etapa ejemplo, el problema para empezar con símbolos ("1/4 + 2/4"); "" en las demás etapas.`; examples start with `"new_problem": ""` (offer, own) and `"new_problem": "1/4 + 2/4"` (example).

`TutorAction` (docstring lines + fields):

```python
    board_problem: the problem the model copied off the board ("2x + 3 = 11"), for the projector's card.
    new_problem: the problem this reply dictates, in symbols: the model's, or code's when code replaced its words.
    ...
    board_problem: str | None = None
    new_problem: str | None = None
```

`RawReply`: add `new_problem: str | None = None`. In `_evaluate`, both successful returns build the reply with
the problem keys:

```python
                return RawReply(*_reply_parts(text, parse_vlm_response(text, projector_size)))
            ...
            return RawReply(*_reply_parts(text, action))
```

```python
def _reply_parts(text: str, action: AvatarAction) -> tuple[AvatarAction, str | None, bool, str | None, str | None]:
    """RawReply's fields for a parsed reply: the action and the helper keys read off the same text."""
    return action, board_answer(text), False, _reply_value(text, "board_problem"), _reply_value(text, "new_problem")
```

`_finish`: `return _tutor(speakable(ruled, language), board_problem=reply.board_problem, new_problem=reply.new_problem)`.

In `evaluate`'s retry branch, after `again, board = …`: `again = _tutor(again, board_problem=board)` (a retry
that left the board out keeps the first read).

`fallback_problem` (above `fallback_text`), and `fallback_text` uses it:

```python
def fallback_problem(kind: str, task: Mapping[str, Any] | None) -> str | None:
    """The problem fallback_text dictates, in symbols, for the projector's card; None when it dictates none."""
    if kind == "learn":
        return _starter(_clean((task or {}).get("topic"))) if learn_stage(task) == "example" else None
    return _seed_problem(3, 6) if kind == "practice" else None
```

In `fallback_text`: `problem=fallback_problem(kind, task)` for the learn-example and practice lines.

`_salvage`: the practice variant returns `_tutor(action, spoken_text=…, new_problem=variant)`; the end becomes

```python
    fell_back = not kept
    out = " ".join(kept) if kept else fallback_text(kind, task, has_ink, language)
    if kind == "learn" and _starter_missing(replace(action, spoken_text=_spoken(out, language)), task, language):
        out, fell_back = fallback_text(kind, task, has_ink, language), True
    if kind == "learn" and learn_stage(task) == "offer":
        out = _with_offer_question(out, language)
    if fell_back:
        return _tutor(action, spoken_text=out, new_problem=fallback_problem(kind, task))
    return replace(action, spoken_text=out)
```

`_mock`, learn/practice branch: after building `reply`, `problem = fallback_problem(mode, t)` and
`if problem: reply["new_problem"] = problem`.

- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q tests/test_vlm_client.py tests/test_mathspeak.py tests/test_mathspeak_es.py` → PASS (fix any prompt-snapshot test that pinned the old help/learn schema by updating its expected text).

- [ ] **Step 5: Commit** `feat(vlm): reply carries the problem in symbols for the projector card`

### Task 3: The assistant updates the card after each turn

**Files:** Modify `biobench_sar/assistant.py` · Test `tests/test_assistant.py`

- [ ] **Step 1: Failing tests** (append; import `TutorAction` from `biobench_sar.llm.vlm_client`)

```python
# ---- the problem card ---------------------------------------------------------------------------------------


def _reply(h: Harness, **problems: str | None) -> None:
    h.vlm.action = TutorAction("ok", "idle", 10, 20, **problems)


def test_practice_shows_the_dictated_problem(h: Harness) -> None:
    _reply(h, board_problem="2x + 3 = 11", new_problem="4x + 7 = 19")
    h.vision.frame = np.zeros((4, 4, 3), np.uint8)
    h.assistant.handle_query("give me a similar problem")
    assert _tutor_events(h)[-1]["problem"] == {"text": "4x + 7 = 19", "source": "dictated"}


def test_help_shows_the_students_problem(h: Harness) -> None:
    _reply(h, board_problem="2x + 3 = 11")
    h.assistant.handle_query("help me")
    assert h.assistant.tutor.problem == {"text": "2x + 3 = 11", "source": "board"}
    assert _tutor_events(h)[-1]["problem"] == {"text": "2x + 3 = 11", "source": "board"}


def test_empty_or_failed_reply_keeps_the_card(h: Harness) -> None:
    _reply(h, board_problem="2x + 3 = 11")
    h.assistant.handle_query("help me")
    _reply(h, board_problem="")
    h.assistant.handle_query("and now?")
    h.vlm.action = TutorAction("model down", "error", 10, 20, failed=True, board_problem="9 = 9")
    h.assistant.handle_query("hello?")
    assert h.assistant.tutor.problem == {"text": "2x + 3 = 11", "source": "board"}


def test_quiz_turns_leave_the_card_to_the_quiz(h: Harness) -> None:
    _reply(h, board_problem="7 + 7")
    h.assistant.handle_query("quiz me")
    assert h.assistant.tutor.problem is None and _tutor_events(h)[-1]["problem"] is None


def test_plain_actions_change_nothing(h: Harness) -> None:
    h.assistant.handle_query("hello")  # the stub's AvatarAction has no problem keys
    assert h.assistant.tutor.problem is None
```

- [ ] **Step 2: Run** `.venv/bin/python -m pytest -q tests/test_assistant.py -k "problem or card"` → FAIL.

- [ ] **Step 3: Implement** in `biobench_sar/assistant.py`, in the pipeline right after
`if not failed: self.tutor.add_turn(text, action.spoken_text)`:

```python
            if not failed and self.tutoring and (task is None or task["kind"] != "quiz"):
                self._show_problem(action)
```

and the method (next to `push_tutor`):

```python
    def _show_problem(self, action: AvatarAction) -> None:
        """The projector card follows the turn: a problem THEO dictated, else the student's read off the board."""
        dictated = getattr(action, "new_problem", None)
        if dictated:
            self.tutor.show_problem(dictated, "dictated")
        else:
            self.tutor.show_problem(getattr(action, "board_problem", None), "board")
```

(`show_problem` ignores empty text, and the existing `self.push_tutor()` right after sends the change.)

- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q tests/test_assistant.py tests/test_tutor.py` → PASS.

- [ ] **Step 5: Commit** `feat(assistant): the problem card follows practice, lessons and help`

### Task 4: The projector card shows every mode's problem

**Files:** Modify `biobench_sar/templates/index.html`, `biobench_sar/static/js/main.js`,
`biobench_sar/static/css/style.css`, `biobench_sar/static/js/i18n.js` · Test `tests/test_frontend.py`

- [ ] **Step 1: Rename the ids everywhere** (longest first):
`quiz-card-count` → `problem-card-label`, `quiz-card-problem` → `problem-card-text`, `quiz-card` →
`problem-card`, in the four files above, `tests/test_frontend.py` and any doc that names them
(`grep -rn "quiz-card" biobench_sar tests docs README.md`). In `main.js` rename `quizCard`/`quizCount`/`quizProblem`
to `problemCard`/`problemLabel`/`problemText`. Run `.venv/bin/python -m pytest -q tests/test_frontend.py` → PASS
(pure rename).

- [ ] **Step 2: Failing test** (after `test_projector_quiz_card_follows_tutor_events`)

```python
@needs_node
def test_projector_problem_card_shows_dictated_and_board_problems(tmp_path: Path) -> None:
    run_node(
        tmp_path,
        "index.html",
        PROJECTOR
        + """
const card = $("problem-card");
sock.fire("connect");
await settle();
sock.fire("tutor", { mode: "practice", learn_topic: null, quiz: null, problem: { text: "4x + 7 = 19", source: "dictated" } });
assert.ok(!card.classList.contains("hidden"));
assert.equal($("problem-card-label").textContent, "Try this");
assert.equal($("problem-card-text").textContent, "4x + 7 = 19");
sock.fire("tutor", { mode: "help", learn_topic: null, quiz: null, problem: { text: "<b>2x</b> = 4", source: "board" } });
assert.equal($("problem-card-label").textContent, "Your problem");
assert.equal($("problem-card-text").textContent, "<b>2x</b> = 4", "set as text, never as HTML");
sock.fire("tutor", { mode: "help", learn_topic: "fractions", quiz: null, problem: null });
assert.ok(card.classList.contains("hidden"));
""",
    )
```

Run `.venv/bin/python -m pytest -q tests/test_frontend.py -k problem_card` → FAIL (label empty / card hidden).

- [ ] **Step 3: Implement**

`i18n.js` — English block next to `"proj.quiz"`: `"proj.problem.dictated": "Try this",` and
`"proj.problem.board": "Your problem",`; Spanish block: `"proj.problem.dictated": "Prueba este",` and
`"proj.problem.board": "Tu problema",`.

`main.js` `onTutor`:

```js
function onTutor(d) {
  tutorVersion += 1;
  const quiz = d.quiz && d.quiz.active ? d.quiz : null;
  // Outside a quiz the card shows the problem THEO dictated or read off the board (tutor.py show_problem).
  const shown = !quiz && d.problem && d.problem.text ? d.problem : null;
  problemCard.classList.toggle("hidden", !quiz && !shown);
  if (quiz) {
    bindText(problemLabel, () => t("proj.quiz", { n: quiz.number, total: quiz.total }));
    // No problem text means the assistant makes problems up and reads them out itself.
    problemText.textContent = quiz.problem || "";
  } else if (shown) {
    bindText(problemLabel, () => t(shown.source === "board" ? "proj.problem.board" : "proj.problem.dictated"));
    problemText.textContent = shown.text;
  }
  uiChanged();
}
```

Update the comment above `onLanguage` ("The corner and the quiz card…") to "the problem card".

- [ ] **Step 4: Run** `.venv/bin/python -m pytest -q tests/test_frontend.py` → PASS.

- [ ] **Step 5: Commit** `feat(projector): the problem card shows practice, lesson and help problems`

### Task 5: Verify and open PR 1

- [ ] `.venv/bin/python -m pytest -q` → all pass (compare with the baseline run before Task 1).
- [ ] `.venv/bin/ruff check . && .venv/bin/ruff format --check .` → clean.
- [ ] `SAR_VLM_BACKEND=mock SAR_TTS_BACKEND=none SAR_PORT=5051 .venv/bin/python -m biobench_sar.app`, open
  `http://localhost:5051/`, type "teach me fractions" then "make one up": the card shows `Try this · 1/5 + 2/5`.
- [ ] Reviewer subagent on `git diff release/candidate...HEAD`; fix what it confirms.
- [ ] Cleanup pass from `CONTRIBUTING.md` over the touched files; list leftovers in the PR's **Cleanup** section.
- [ ] Push `feat/problem-card`, open the PR into `release/candidate` (not `main`).
