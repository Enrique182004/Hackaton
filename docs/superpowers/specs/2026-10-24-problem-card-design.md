# Problem card in every mode + board watcher — design

Date: 2026-10-24 · Branches: `feat/problem-card` (PR 1), then `feat/board-watch` (PR 2), both from
`release/candidate`.

## Goal

The projector shows the problem the student is working on, in symbols (`4x + 7 = 19`), in every tutor mode —
not only what THEO says about the next step. Today only a running quiz shows its problem (`#quiz-card`). The
card also follows what the student writes on the board, without the student having to talk to THEO.

## Where the problem text comes from

| Mode | Source | Card heading (en / es) |
| --- | --- | --- |
| Quiz | `TutorSession.current().problem` (worksheet or invented; exact) | `Quiz · problem {n} of {total}` (unchanged) |
| Practice ("similar problem") | new reply key `new_problem` — the problem THEO dictates | `Try this` / `Prueba este` |
| Learn a topic, stage `example` | new reply key `new_problem` — the starter problem | `Try this` / `Prueba este` |
| Help | new reply key `board_problem` in help mode — the student's problem copied from the board | `Your problem` / `Tu problema` |
| Board watcher (PR 2) | short read-only model call returning only `board_problem` | `Your problem` / `Tu problema` |

Practice already returns `board_problem` (the old problem on the board) so code can reject a new problem that
is the old one read back; it keeps doing that. Quiz keeps its schema: the problem is known in code.

## PR 1 — card in every mode, from the replies

### Model replies (`llm/vlm_client.py`)

- `HELP_SCHEMA` = `MATH_SCHEMA` with `board_problem` first; help rules (en + es) say: FIRST copy the problem
  written on the board into `board_problem`, exactly (`""` if the board is empty). Examples updated.
- `PRACTICE_SCHEMA` gains `new_problem` after `board_problem`; rules: write the new problem in symbols in
  `new_problem`, the same one dictated in `spoken_text`. Examples updated.
- `LEARN_SCHEMA` = `MATH_SCHEMA` with `new_problem` first; learn rules: stage `example` fills it with the starter
  problem in symbols, every other stage leaves it `""`. Examples updated.
- `RawReply` gains `new_problem`; `TutorAction` gains `board_problem` and `new_problem` (not in `to_dict()`, so
  the four-key schema sent to clients is unchanged). `evaluate` copies them from the reply that is returned
  (the retry's when it is used, falling back to the first reply's).
- The mock backend fills `new_problem` for practice and learn `example`, so demos without a model show the card.

### Tutor state (`tutor.py`)

`TutorSession` keeps the displayed problem: `problem_text: str | None` and `problem_source:
"dictated" | "board" | None`.

- `show_problem(text, source)` — trims, collapses whitespace, caps at 80 characters (cut at a word, `…`);
  empty text is ignored (the card keeps what it shows).
- `clear_problem()`. The card also clears on `start_learning` (a new topic starts clean), `set_mode("quiz")` and
  `start_quiz` (the dashboard's button skips `set_mode`), and `show_problem` is ignored while a quiz runs (a turn
  that started before the quiz must not leave its problem behind for after it).
- `to_dict()` gains `"problem": {"text": …, "source": …}` or `None`. During an active quiz the quiz block stays
  the source of truth and `problem` is `None`.

### Assistant (`assistant.py`)

After each finished math turn, by the turn's task (a failed call changes nothing; `push_tutor()` follows as
today):

- quiz → nothing (the quiz owns the card);
- practice, or a lesson at stage `example` → `show_problem(new_problem, "dictated")`, or `clear_problem()` when the
  reply has no `new_problem`: an older problem must not stand in for the one just read out;
- help → `show_problem(board_problem, "board")`, only when the frame has ink (an empty board's "problem" is a
  copied example or a placeholder);
- a lesson's offer / own stage → nothing. The offer speaks a worked example solved in code (#88,
  `worked_examples.py`); by decision the card stays empty for it: the bubble already shows it in notation, and the
  card is only for problems the student solves (a solved example under "Try this" would ask them to solve it).

`VLMClient.evaluate` hands over only problems the card may show (`_card_problems`): `board_problem` in help and
practice, `new_problem` only in practice and a lesson's example, and only when the final `spoken_text` reads it
out with its numbers, operations, letters and powers in the same order (`_card_reads`; other words may come in
between, "minus" counts as "negative"); a board value must be an operation on numbers or letters, so
placeholders (`N/A`, `none`), bare numbers and the prompt's own `left=… top=…` box line are dropped.

### Projector (`templates/index.html`, `static/js/main.js`, `static/css/style.css`, `static/js/i18n.js`)

- `#quiz-card` becomes `#problem-card` (`#problem-card-label`, `#problem-card-text`); same place and style.
- `onTutor`: active quiz → label `proj.quiz`, text = quiz problem (as today). Else `d.problem` → label
  `proj.problem.dictated` or `proj.problem.board`, text = `d.problem.text`. Else hidden.
- Text is set with `textContent` only. The card still hides while the camera captures (`CAPTURE_STATES`) and
  is still reported in `ui_rects`.
- New i18n keys in en and es.

### Tests (PR 1)

- `test_vlm_client.py`: `new_problem` / help `board_problem` parsed from clean, missing, empty and numeric
  replies (a reply cut off mid-JSON is rejected as not JSON before the helper keys are read); retry keeps the fields; prompts for help/practice/learn (en + es) contain the new keys.
- `test_tutor.py`: `show_problem` trims/caps/ignores empty; `clear_problem` on new topic and quiz;
  `to_dict()` shape; quiz active → `problem` is `None`.
- `test_assistant.py`: practice/learn-example turns show `dictated`; help shows `board`; failed turn keeps the
  previous card; quiz turn leaves it alone.
- `test_frontend.py`: card shows quiz, dictated and board problems with the right label, hides when the tutor
  sends none, and never renders HTML from the text.

## PR 2 — board watcher + projector-UI masking

### Masking (`assistant.py`)

`app.py` already receives the projector page's `ui_rects` and calls `assistant.set_ui_rects`, which is not on
`release/candidate` yet: it is built on `feat/fin-tutor-core` / `integrate/tutor-core` (ink regions overlapping a
UI rect are dropped, rects age out after 3 s; `docs/BUILD_PLAN.md` "Client → server"). PR 2 starts after that work
lands and reuses it for the watcher's frames; it does not build a second masking path.

### Watcher (`core/board_watch.py`)

`BoardWatcher` — one daemon thread, ~1 Hz, never on the camera thread:

1. Skip unless: math profile, tutor mode is help or practice (not quiz, no pending offer), and
   `assistant.should_listen()` (not busy, not speaking).
2. Take `vision.latest_warped()`, `find_ink_regions`, drop regions under the UI rects (the masking above); fingerprint = ink mask downscaled to 64×36,
   thresholded.
3. Changed from the last read's fingerprint by more than 2 % of cells, then unchanged for 2 s (writing done,
   hand away) → read, if ≥ 8 s since the last read.
4. Read = `VLMClient.read_board(crop)`: a short prompt asking only for `{"board_problem": str}` on the ink crop;
   never raises; mock backend returns `""`.
5. Before applying the result, drop it if a question started meanwhile: `Assistant.query_count` (new; bumped
   when `handle_query` accepts a question) differs from its value when the read began.
   Non-empty → `show_problem(text, "board")` + `push_tutor()`.
6. No ink at all and the card's source is `board` → `clear_problem()` + `push_tutor()`. A `dictated` problem
   waiting to be written stays.

The watcher does not take the assistant's busy lock: a student's question is never refused because of it. It
may wait behind a read inside Ollama (a few seconds at most, capped by the read's own short timeout).

Settings (`config.py`, `sar.env.example`): `SAR_BOARD_WATCH` (default `1`), `SAR_BOARD_WATCH_SETTLE_S` (2),
`SAR_BOARD_WATCH_MIN_GAP_S` (8). Started and stopped with the other services in `app.py`.

### Tests (PR 2)

- The watcher ignores ink under the projected UI (reusing the tutor-core masking).
- `BoardWatcher` with a fake clock, fake frames and a fake VLM: no read while writing; one read after settle;
  respects the min gap; skipped in quiz, while busy, while speaking; result dropped when a turn started; erase
  clears a `board` card but not a `dictated` one.
- `read_board` parses clean / empty / broken replies and never raises.
- Hardware check: the camera loop still reaches ≥ 28 FPS with the watcher running (`scripts/hardware_check.py`).

## Out of scope

- The biology profile: tutor modes are math-only (`Assistant.tutoring`); the card does not show there.
- Showing the current step on the card.
- Changing the TTS voice (separate change).
