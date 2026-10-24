# Build Plan — SAR Whiteboard Assistant

Two specs, one codebase. The subject is chosen by `SAR_PROFILE`:

- `math` (default): [sar_math_tutor_prompt.md](sar_math_tutor_prompt.md), the math tutor (step-by-step help,
  similar problems, quiz, learning a topic, phonetic speech).
- `biology`: [sar_whiteboard_assistant_prompt.md](sar_whiteboard_assistant_prompt.md), the original
  *Acanthamoeba* research assistant. It also defines the module layout both profiles share.

The work is split into **nine areas**. Each area owns its own files and talks to the others **only** through
the contracts below, so anyone can pick up one area without touching the rest.

**How to pick up an area:** find its GitHub issue (label `area:*`), assign yourself, branch
`feat/<area>`, keep to the files listed, and make sure `.venv/bin/python -m pytest -q tests/test_<area>*.py`
passes.

Shared files (change only via PR that updates this doc): `biobench_sar/config.py`, `biobench_sar/schemas.py`.

Run everything from the repo root:

```bash
.venv/bin/python -m pip install -r requirements-dev.txt   # once: pytest + ruff
.venv/bin/python -m biobench_sar                          # SAR_PROFILE=biology for the biology profile
.venv/bin/python -m pytest -q
.venv/bin/ruff check . && .venv/bin/ruff format --check .
```

Windows PowerShell: use `.\.venv\Scripts\python.exe` for `.venv/bin/python`, and set variables first, e.g.
`$env:SAR_PROFILE='biology'; .\.venv\Scripts\python.exe -m biobench_sar` (the setting lasts for that window).

---

## Phase 1 — Build (parallel)

| # | Area | Owns | Depends on (contract only) |
| --- | --- | --- | --- |
| A | Vision & calibration | `core/homography.py`, `core/vision_loop.py`, `tests/test_homography.py`, `tests/test_vision_loop.py` | `schemas.InkRegion` |
| B | RAG knowledge base | `rag/ingestor.py`, `rag/vector_store.py`, `knowledge/math/*`, `knowledge/biology/*` (seed docs), `tests/test_rag.py` | `schemas.RetrievedChunk` |
| C | VLM reasoning | `llm/vlm_client.py`, `tests/test_vlm_client.py` | `schemas.*`, I (`mathspeak.to_speech`, lazy import) |
| D | Voice in / voice out | `core/audio_trigger.py`, `llm/tts_engine.py`, `tests/test_audio.py`, `tests/test_tts.py` | — |
| E | Projector frontend + dashboard | `templates/*.html`, `static/**` | Socket/HTTP contract |
| F | Server & ops | `app.py`, `assistant.py`, `scripts/*`, `tests/test_app.py`, `tests/test_assistant.py` | all contracts (uses fakes) |
| G | Tutor session | `tutor.py`, `tests/test_tutor.py` | H (`QuizProblem`) |
| H | Quiz bank | `rag/quiz_bank.py`, `tests/test_quiz_bank.py` | B (text extraction for `.pdf` / `.docx`) |
| I | Math speech | `llm/mathspeak.py`, `tests/test_mathspeak.py` | — |

## Phase 2 — Integrate
1. All unit tests green: `.venv/bin/python -m pytest -q`; lint clean: `.venv/bin/ruff check . && .venv/bin/ruff format --check .`.
2. `SAR_VLM_BACKEND=mock SAR_TTS_BACKEND=none .venv/bin/python -m biobench_sar` — both pages load, a typed
   query moves the avatar.
3. Knowledge base: on a fresh `SAR_DATA_DIR` the profile's seed files (`biobench_sar/knowledge/<profile>/`) are
   ingested at startup — confirm they are listed on `/dashboard` → step 1 (*Worksheets & notes* in math,
   *Study files* in biology) and `doc_chunks` in `/api/status` is > 0. Upload a new file and confirm
   `doc_chunks` grows and `/api/search` retrieves it. To test seeding by upload instead, start with
   `SAR_SEED_KNOWLEDGE=0` and an empty `SAR_DATA_DIR`. Test each profile with its own `SAR_DATA_DIR` (both
   share one knowledge base otherwise, see Shared below).
4. Switch to a real VLM (`ollama pull qwen2.5vl:3b`, `SAR_VLM_BACKEND=ollama`); benchmark with
   `.venv/bin/python -m scripts.bench_vlm -n 3`. Pass/fail scenarios: `--scenarios --profile math` (math cases)
   or `--scenarios` (`--profile biology` is the default: the flowchart cases); `--only H,QC` runs only those
   scenarios, `--raw` grades the model without `evaluate`'s math post-processing.
5. Math tutor: the math scenario test cases (below) pass with the mock backend for the control flow and with
   `qwen2.5vl:3b` for the wording.

## Phase 3 — Hardware test
1. `.venv/bin/python scripts/hardware_check.py` — camera FPS (PASS at ≥ 28, target ~30), microphone level, TTS
   playback, displays.
2. Projector: open `/` fullscreen on the projector display (`F`), then on your laptop open `/dashboard` →
   *Board* tab → **Line up camera and projector**.
3. Write an equation on the board → check the lined-up view on the *Board* tab (or
   `/api/frame.jpg?warped=1`; without `?warped=1` it is the raw camera frame) lines up with the board.
4. Ask a question by voice (Space on the projector page / **Talk** on the dashboard) → the avatar walks to the
   ink and speaks.
5. Scenario test cases (below) pass end-to-end for both profiles.

---

## Contracts

All paths are inside the `biobench_sar` package. Imports are package-absolute
(`from biobench_sar.core.homography import Calibration`).

### Shared (`schemas.py`, `config.py`)
- `InkRegion` TypedDict `{x, y, w, h, cx, cy}` (projector coords).
- `RetrievedChunk(text, source, score, metadata)`.
- `AvatarAction(spoken_text, animation_state, target_x, target_y)` + `.to_dict()`.
- `parse_vlm_response(text, projector_size) -> AvatarAction` — never raises.
- `error_action(message, projector_size) -> AvatarAction`.
- `Settings` — every knob is an env var `SAR_*` (see `config.py`). Subject profile:
  - `profile: str` — env `SAR_PROFILE`, default `"math"`; allowed `"math" | "biology"`, anything else → `"math"`.
  - `knowledge_dir: Path` (property) — `PACKAGE_DIR / "knowledge" / profile`; the seed docs ingested at startup.
    Biology seed docs live in `knowledge/biology/`; math seed docs in `knowledge/math/` (`quiz_bank.csv` with
    header `problem,answer,topic,level`, `tutoring_guide.md`, `worksheet_linear_equations.md` with `## Level N`
    headings that match the CSV's levels).
  - `seed_marker_file: Path` (property) — `chroma_dir / ".seeded"`: the profiles whose seed docs are already in
    the store. Each profile is seeded once per store; deleting `chroma_dir` seeds it again.
  - `quiz_bank_file: Path` — env `SAR_QUIZ_BANK_FILE`, default `DATA_DIR / "quiz_bank.json"`.
  - `quiz_length: int` — env `SAR_QUIZ_LENGTH`, default `10`: the most problems one quiz asks, and how many the
    model makes up when no ticked file has any.
  - Both profiles share one `SAR_DATA_DIR`: one store, one file selection, one quiz bank. Run each profile with
    its own `SAR_DATA_DIR` (e.g. `data/biology`) to keep one subject's files out of the other's answers.

### A — `core/homography.py`
```python
def generate_checkerboard(width: int, height: int, cols: int = 9, rows: int = 6) -> np.ndarray  # BGR, (rows+1)x(cols+1) squares, white margin
def projector_corner_points(width: int, height: int, cols: int = 9, rows: int = 6) -> np.ndarray  # (cols*rows, 2) float32, same order as cv2.findChessboardCorners
def detect_corners(frame: np.ndarray, cols: int = 9, rows: int = 6) -> np.ndarray | None  # (N,2) float32, sub-pixel refined
def compute_homography(camera_pts: np.ndarray, projector_pts: np.ndarray) -> np.ndarray  # 3x3, cv2.findHomography RANSAC; ValueError if it fails
class Calibration:
    matrix: np.ndarray; projector_size: tuple[int, int]
    def camera_to_projector(self, points: np.ndarray) -> np.ndarray  # (N,2) -> (N,2), cv2.perspectiveTransform
    def warp_frame(self, frame: np.ndarray) -> np.ndarray  # cv2.warpPerspective to projector_size
    def save(self, path: Path) -> None  # JSON
    @classmethod
    def load(cls, path: Path) -> "Calibration | None"
    @classmethod
    def identity(cls, camera_size: tuple[int, int], projector_size: tuple[int, int]) -> "Calibration"  # scales camera frame to projector
def calibrate_from_frame(frame: np.ndarray, projector_size: tuple[int, int], cols: int = 9, rows: int = 6) -> Calibration | None
```

### A — `core/vision_loop.py`
```python
class VisionLoop:
    def __init__(self, camera_index: int = 0, width: int = 1280, height: int = 720,
                 calibration: Calibration | None = None,
                 capture_factory: Callable[[int], CaptureLike] | None = None) -> None
    def start(self) -> None; def stop(self) -> None  # daemon thread, reconnects on read failure
    def latest_frame(self) -> np.ndarray | None      # copy of raw camera frame
    def latest_warped(self) -> np.ndarray | None     # calibration.warp_frame(latest) or raw if uncalibrated
    def set_calibration(self, calibration: Calibration | None) -> None
    fps: float; is_running: bool                      # properties
def encode_jpeg(frame: np.ndarray, quality: int = 80) -> bytes
def find_ink_regions(frame: np.ndarray, min_area: int = 150, max_regions: int = 10) -> list[InkRegion]  # dark ink on light board, largest first
```

### B — `rag/ingestor.py`, `rag/vector_store.py`
```python
SUPPORTED_EXTENSIONS: frozenset[str]  # .pdf .txt .md .csv .docx .pptx .png .jpg .jpeg
class UnsupportedFileType(ValueError)
@dataclass(frozen=True)
class Chunk: text: str; source: str; metadata: dict[str, str | int]
def chunk_paragraphs(text: str, max_chars: int = 1000) -> list[str]
def ingest_file(path: str | Path) -> list[Chunk]

class VectorStore:
    def __init__(self, persist_dir: Path, collection: str = "biobench", embed_model: str = "all-MiniLM-L6-v2",
                 embedder: Callable[[list[str]], list[list[float]]] | None = None) -> None  # embedder injectable for tests
    def add_chunks(self, chunks: list[Chunk]) -> int   # upsert, stable ids; returns number added
    def query(self, text: str, k: int = 4, sources: list[str] | None = None) -> list[RetrievedChunk]  # None = all files, [] = nothing
    def count(self) -> int; def clear(self) -> None
```

### C — `llm/vlm_client.py`
```python
MODES = ("help", "practice", "quiz")
def system_prompt(profile: str, mode: str) -> str  # persona for the profile (math tutor / biology research assistant) + mode rules + strict JSON schema
def build_prompt(query: str, context: list[RetrievedChunk], ink_regions: list[InkRegion], projector_size: tuple[int, int],
                 *, profile: str = "biology", mode: str = "help", task: dict | None = None,
                 recent_turns: list[dict[str, str]] | None = None) -> str
class VLMClient:
    def __init__(self, backend: str, model: str, base_url: str, api_key: str | None = None, timeout: float = 90,
                 *, max_image_side: int = 1024, profile: str = "biology") -> None
    def warmup(self) -> None  # loads the Ollama model ahead of the first question; never raises
    def evaluate(self, query: str, image_jpeg: bytes | None, context: list[RetrievedChunk],
                 ink_regions: list[InkRegion], projector_size: tuple[int, int],
                 *, mode: str = "help", task: dict | None = None,
                 recent_turns: list[dict[str, str]] | None = None) -> AvatarAction  # never raises
# backends: "ollama" (POST /api/chat, images base64, format json), "openai" (OpenAI-compatible /v1/chat/completions), "mock" (points at largest ink region)
```
- `recent_turns`: `[{"q": str, "a": str}]`, oldest first, at most 3, included in the prompt as
  "Conversation so far" so follow-ups ("make one up", "I did it") make sense.
- Profile `math`: the returned `spoken_text` goes through `mathspeak.to_speech` (lazy import; identity if
  the module is missing).
- `task` passed to `evaluate`:

| Task | Dict |
| --- | --- |
| help | `None` |
| practice | `{"kind": "practice"}` |
| quiz | `{"kind": "quiz", "stage": "dictate" \| "check" \| "hint", "number": int, "total": int, "problem": str \| None, "answer": str \| None, "next_problem": str \| None}` — `problem` `None` = no worksheet problems, the model makes up a progressive problem itself; `answer` is set only in stage `check` (`None` in `dictate` and `hint`); `next_problem` only in `check` |
| learn | `{"kind": "learn", "topic": str, "stage": "offer" \| "example" \| "own"}` |

- **Answers never reach the model except when checking.** The expected answer travels only as `task["answer"]`
  in stage `check`, and even there a plain-number answer (`x = 5`, `2/3`, `-4`) is graded in code against the
  model's `board_answer` rather than shown in the prompt. Profile `math` also removes answer keys from the RAG
  context in every task (help and practice too): worksheet `Answer: ...` lines and quiz-bank answers are dropped
  from retrieved chunks before the prompt is built.

- Math semantics in the unchanged JSON schema `{spoken_text, animation_state, target_x, target_y}`
  (plain language for a middle/high-school student, short because it is spoken):

| Task | `spoken_text` | `animation_state` / target |
| --- | --- | --- |
| help | Explains ONLY the next operation, never the final answer | `pointing` at the relevant part of the equation's ink region |
| practice | Reads the board's equation and dictates a new one of the same difficulty (constants/variables swapped) | `idle` |
| quiz `dictate` | Reads the problem out | `idle` |
| quiz `check` | Correct: praise, and when `next_problem` is given also dictates it. Wrong: one hint, never the answer | `celebrating` if the ink answer is correct, `error` if wrong |
| quiz `hint` | The student asked for help on the current problem ("I'm stuck", "hint", "help"): one hint for the next step, never the answer. Nothing is recorded and the quiz stays on the problem | `idle` |
| learn `offer` | One or two plain sentences introducing the idea of the topic, then asks "Do you have an exercise in mind, or should I make one up for you?" | `idle` |
| learn `example` | Dictates one easy starter problem on the topic and invites the student to write it and try the first step | `idle` |
| learn `own` | Asks the student to write their exercise on the board and then ask for help | `idle` |

Learn tasks work with an empty board and in any mode.

### D — `core/audio_trigger.py`, `llm/tts_engine.py`
```python
class AudioTrigger:
    def __init__(self, on_query: Callable[[str], None], whisper_model: str = "base", wake_word: str = "assistant",
                 mode: str = "push", record_seconds: float = 5.0,
                 on_state: Callable[[str], None] | None = None,  # on_state("listening"|"transcribing"|"idle")
                 recorder: Callable[[float], np.ndarray] | None = None,  # injectable for tests
                 transcriber: Callable[[np.ndarray], str] | None = None,
                 wake_chunk_seconds: float = WAKE_CHUNK_SECONDS,
                 should_listen: Callable[[], bool] | None = None,  # False pauses the wake loop (avatar speaking)
                 silence_rms: float = SILENCE_RMS) -> None  # wake chunks quieter than this skip Whisper
    def start(self) -> None; def stop(self) -> None   # "wake" mode starts a background listener
    def trigger(self) -> bool                          # push-to-talk (non-blocking); False if already recording
    def submit_text(self, text: str) -> None           # typed query -> on_query
    def transcribe(self, audio: np.ndarray) -> str     # float32 mono 16 kHz; lazy-loads whisper
    available: bool                                    # False if no mic / pyaudio -> app falls back to text
class TTSEngine:
    def __init__(self, backend: str = "edge", voice: str = "en-US-AriaNeural", out_dir: Path = ...) -> None
    def speak(self, text: str) -> str | None  # edge: writes mp3 to out_dir, returns file name for browser playback;
                                               # say/pyttsx3: plays locally (non-blocking), returns None; falls back say→pyttsx3→none
```

### G — `tutor.py`
```python
MODES = ("help", "practice", "quiz")
def detect_mode(text: str) -> str | None
    # "quiz me" / "start a quiz" -> "quiz"; "stop the quiz" -> "help";
    # "similar problem" / "another one like this" / "more practice" -> "practice";
    # "help" / "next step" / "I'm stuck" / "hint" -> "help"; anything else -> None
def is_quiz_stop(text: str) -> bool  # an explicit "stop / end / quit the quiz": the only words that leave a running quiz
def detect_learn_request(text: str) -> str | None
    # topic: "I want to learn how to solve fractions" -> "solving fractions" (or "fractions"); "teach me equations";
    # "how do I add fractions?" only when it is a general how-to, not about the board
def detect_offer_reply(text: str) -> str | None
    # "make one up" / "give me one" / "you choose" / "yes make one" -> "generate";
    # "I have one" / "I've got my own" / "I'll write mine" -> "own"; else None
class TutorSession:  # thread-safe
    mode: str
    def set_mode(self, mode: str) -> None
    def start_quiz(self, problems: list[QuizProblem]) -> None  # at most Settings.quiz_length problems are asked;
                                                               # none -> the model makes up quiz_length
    def stop_quiz(self) -> None
    def current(self) -> QuizProblem | None
    dictated: bool
    def mark_dictated(self) -> None
    def record(self, correct: bool) -> None          # advances to the next problem on correct
    learn_topic: str | None
    offer_pending: bool
    def start_learning(self, topic: str) -> None     # sets offer_pending
    def resolve_offer(self) -> None                  # clears offer_pending, keeps the topic
    def turns(self) -> list[dict[str, str]]          # last 3 {"q", "a"} pairs, oldest first
    def add_turn(self, q: str, a: str) -> None
    def to_dict(self) -> dict
    # {"mode": str, "learn_topic": str | None,
    #  "quiz": None | {"active": bool, "number": int, "total": int, "score": int, "problem": str | None}}
    # score = problems right on the first try (a fixed answer moves on but does not score)
```

**Per question in the assistant (F), profile `math`:**
1. If `offer_pending` and `detect_offer_reply(text)` → task learn stage `"example"` (generate) or `"own"`, then
   `resolve_offer()`.
2. Elif `detect_learn_request(text)` → `start_learning(topic)` and task learn stage `"offer"`.
3. Else the mode logic: `detect_mode` switches mode (quiz starts a quiz from the ticked files' problems, at
   most `Settings.quiz_length` of them, easiest level first; "stop the quiz" stops it). While a quiz runs, only
   `is_quiz_stop(text)` leaves it: "help" / "I'm stuck" / "hint" stay in the quiz. In quiz mode a problem not
   yet dictated gets stage `"dictate"` + `mark_dictated()`; once dictated, a help request gets stage `"hint"`
   (nothing recorded), anything else stage `"check"` and the result is recorded with `record(correct)`
   (`celebrating` = correct).
4. Always pass `recent_turns=tutor.turns()` and call `add_turn(text, spoken_text)` after the answer.

Profile `biology`: `recent_turns` too, no tutor tasks.

### H — `rag/quiz_bank.py`
```python
@dataclass(frozen=True)
class QuizProblem: problem: str; answer: str | None; topic: str; level: int; source: str
def parse_quiz_file(path: str | Path, source: str) -> list[QuizProblem]
    # CSV: problem (or question) + answer columns, optional topic and level (or difficulty) columns.
    # .md / .txt / .pdf / .docx worksheets: numbered lines like "1. 3x + 5 = 20" or "Problem 2: ...",
    # each optionally followed by an "Answer: ..." line (or an "Answer key" section at the end).
    # Only files that call themselves a worksheet count: the file name, first line or a heading contains
    # worksheet / quiz / exercise / homework / assignment / practice / problem set / drill / test, so a guide's
    # numbered tips never become problems. A numbered line counts only with a digit plus a math operator or a
    # math verb (solve, simplify, find, ...). "## Level N: topic" headings (or a "Level N" line) set the level
    # and topic of the problems below them; without one a problem is level 1.
class QuizBank:  # persisted as JSON at Settings.quiz_bank_file
    def __init__(self, path: Path) -> None
    def replace_source(self, source: str, problems: list[QuizProblem]) -> None
    def remove_source(self, source: str) -> None
    def problems(self, sources: list[str] | None = None) -> list[QuizProblem]  # ordered by level, then file order
```
Quiz problems come only from ticked (selected) files; one quiz asks at most `Settings.quiz_length` of them.

### I — `llm/mathspeak.py`
```python
def to_speech(text: str) -> str      # math notation inside normal sentences -> words a TTS engine reads naturally
def has_math_symbols(text: str) -> bool
```

| Written | Spoken |
| --- | --- |
| `2/3` | two thirds |
| `x^2` | x squared |
| `3x+5=12` | three x plus five equals twelve |
| `-4` | negative four |
| `0.5` | zero point five |
| `25%` | twenty five percent |
| `sqrt(9)`, `√9` | the square root of nine |
| `*`, `×`, `·` | times |
| `÷` | divided by |
| `<=`, `>=`, `<`, `>` | words (less than or equal to, …) |

Prose without math is returned unchanged; `to_speech` is idempotent.

### E/F — HTTP + Socket.IO contract
HTTP (F implements, E calls):
| Method | Path | Body / result |
| --- | --- | --- |
| GET | `/` | projector UI (`templates/index.html`) |
| GET | `/dashboard`, `/precharge` | operator dashboard (`templates/precharge.html`, `static/js/dashboard/`) |
| GET | `/api/config` | active non-secret settings, including `"profile"` |
| GET | `/api/history` | `{"interactions": [...]}` newest first; each item `{id, at, query, spoken_text, animation_state, target_x, target_y, sources, ink_regions, had_frame, latency_s, error, mode, task_kind}` (`ink_regions` = count; `task_kind` = the task's `kind` or `null`). Optional, for the dashboard's notes: `failed: bool` — the answer is an `error_action` because the model call or the pipeline failed (timeout, model unreachable, unreadable reply), not a mistake the model found; `display_text: str` — the answer before `mathspeak`, in normal notation. Without them the dashboard recognises the server's own failure texts and shows `spoken_text` |
| GET | `/api/documents` | `{"documents": [{"source", "type", "chunks", "active": bool}], "total_chunks": int, "active_count": int, "all_active": bool}` — `active` = answers may use this file; `all_active` = no explicit selection (every file, default) |
| PUT | `/api/documents/selection` | `{"all": true}` → every file active, or `{"sources": ["a.md", ...]}` → exactly those active (unknown names ignored; `[]` = none, answers skip RAG); non-list `sources` / other body → 400. Returns the `GET /api/documents` body. Persisted to `SAR_SELECTION_FILE` |
| DELETE | `/api/documents/<source>` | `{"deleted": n}`; 404 if unknown. Also drops it from the selection |
| GET | `/api/search?q=&k=` | `{"results": [{"text", "source", "score"}]}`, k clamped 1–10 |
| GET | `/api/ink` | `{"regions": [InkRegion], "size": [w, h]}`; 503 without a frame |
| POST | `/api/upload` | multipart `files[]` → `{"files": [{"name", "chunks", "replaced"?, "error"?}], "total_chunks": int}`; with an explicit selection, uploaded files become active |
| GET | `/api/status` | `{"fps", "camera_running", "calibrated", "doc_chunks", "active_documents", "total_documents", "vlm_backend", "vlm_model", "audio_available", "tts_backend", "listen_mode", "state", "busy", "last_latency_s", "uptime_s", "profile", "mode"}` |
| GET | `/api/tutor` | `TutorSession.to_dict()` |
| PUT | `/api/tutor/mode` | `{"mode": "help" \| "practice" \| "quiz"}` → `to_dict()`; 400 otherwise. Quiz mode starts a quiz on the next question or via `POST /api/tutor/quiz/start` |
| POST | `/api/tutor/quiz/start` | → `to_dict()`; starts even with 0 problems (the model invents problems); `total` = number of problems in the ticked files capped at `SAR_QUIZ_LENGTH`, or `SAR_QUIZ_LENGTH` (default 10) when they have none |
| POST | `/api/tutor/quiz/stop` | → `to_dict()` |
| POST | `/api/query` | `{"text": str}` → `202 {"accepted": true}` (result arrives over socket) |
| POST | `/api/listen` | push-to-talk trigger → `202` |
| POST | `/api/calibrate` | shows checkerboard, captures, solves → `{"ok": bool, "error"?: str}` |
| GET | `/calibration.png` | checkerboard at projector size |
| GET | `/api/frame.jpg` | latest raw camera frame; `?warped=1` = lined up to projector coordinates (debug preview) |
| GET | `/tts/<name>` | generated speech audio |

Socket.IO events (server → client): `avatar_update {animation_state, target_x, target_y, spoken_text}`,
`speak {url: str | null, text}`, `status {state: "idle"|"listening"|"transcribing"|"thinking"|"speaking", detail?}`,
`transcript {text}`, `show_calibration {url}`, `hide_calibration {}`, `interaction {...history item}`,
`documents_changed {...GET /api/documents body}` (after upload, delete and selection changes),
`tutor {...TutorSession.to_dict()}` (whenever the mode, quiz or learn topic changes).
`avatar_update` and `speak` may also carry `display_text` (see `/api/history`); both pages show it in place of the
spoken words when present.
On connect the server sends the current `status` to the new client. The dashboard re-reads `/api/config` on
every connect, and also takes the subject from `/api/status` `profile`, so a server restarted with another
`SAR_PROFILE` is followed without a reload; the math mode picker stays hidden until the subject is known.
Busy handling: `/api/query` → 409 while a query runs; `/api/listen` → 409 already recording, 503 no mic;
`/api/calibrate` → 503 without a frame, 409 while another calibration runs.
Client → server: `text_query {text}`, `trigger_listen {}`.
Projector UI: hotkey **Space** = push-to-talk, **T** = type a query, **C** = toggle debug overlay, **F** =
fullscreen. It also listens to `tutor` and shows the current quiz problem in normal notation (e.g.
`3x + 5 = 20`) in a small, faint top-left card (thin outline, no glow), hidden when no quiz is running and while
the status is `listening`, `transcribing` or `thinking`, so the camera picture the model reads shows only the
student's ink. For a typed question the frame can be taken before the card has gone, so the card stays faint.

---

## Scenario test cases — Math (default profile)
Seed docs in `biobench_sar/knowledge/math/` (tick `quiz_bank.csv` and `worksheet_linear_equations.md` on the
dashboard). For every case, `spoken_text` contains no `^`, `/`, `*` or `=`. The help and practice cases use
equations that are not in the seed files, so a pass never depends on an answer key the model should not see.
1. **Next step, not the answer:** write `4x + 3 = 19`, say "I'm stuck" → `pointing` at the equation; explains
   only the next operation ("subtract three from both sides"); never says "x equals four".
2. **Phonetic speech:** write `x^2 + 7x + 10 = 0`, ask for help → the answer reads "x squared plus seven x plus
   ten equals zero", and fractions on the board such as `2/3` are read as "two thirds".
3. **Similar problem:** with `3x - 4 = 11` on the board, say "give me a similar problem" → `idle`, dictates a
   new two-step equation with different numbers (e.g. "five x minus two equals thirteen").
4. **Quiz, correct answer:** say "quiz me" → `idle`, dictates problem 1 of N, and the projector shows it
   top-left in normal notation; write the right answer and say "check my answer" → `celebrating`, then the
   next problem is dictated and the card advances.
5. **Quiz, wrong answer:** write a wrong answer → `error`, one hint, the correct answer is never spoken; the
   card stays on the same problem.
6. **Quiz, ask for a hint:** after a problem is dictated, say "I'm stuck" → one hint for the next step (stage
   `hint`), never the answer; the quiz keeps going on the same problem and the score is unchanged.
7. **Quiz length and order:** with both seed files ticked (43 problems), "quiz me" starts a quiz of 10
   (`SAR_QUIZ_LENGTH`): the six level-1 fraction problems, then level-2 one-step equations, never back to an
   easier level.
8. **Quiz without worksheets:** untick every quiz file, say "start a quiz" → the quiz starts with total 10
   (`SAR_QUIZ_LENGTH`) and the model makes up progressive problems.
9. **Stop the quiz:** say "stop the quiz" → back to help mode, the card is hidden.
10. **Learn a topic, empty board:** say "I want to learn how to solve fractions" → `idle`, one or two plain
    sentences on what a fraction is, then "Do you have an exercise in mind, or should I make one up for you?"
11. **Learn → make one up:** reply "make one up" → `idle`, dictates an easy starter (e.g. "one fifth plus two
    fifths") and invites the first step; then "I did it, what now?" continues from that problem.
12. **Learn → own exercise:** after the offer, reply "I have one" → `idle`, asks the student to write it on the
    board and then ask for help.

## Scenario test cases — Environmental Biology (`SAR_PROFILE=biology`)
Seed docs in `biobench_sar/knowledge/biology/`. The assistant must:
1. Verify a drawn sampling flowchart for *Acanthamoeba* in Laguna de Bustillos (collection → filtration → NNA agar + *E. coli* lawn → incubation → microscopy → PCR).
2. Flag a missing sterile step (e.g. no autoclave / flame sterilization) with `animation_state: "error"` and point at it.
3. Explain how runoff from Leptosol vs. Feozem soils in Cuauhtémoc, Chihuahua affects turbidity/nutrients at sampling sites.
4. Celebrate (`celebrating`) when the procedure is complete and correct.

## Status (2026-10-23)
| Area | Status | Verified |
| --- | --- | --- |
| Shared contracts | ✅ done, incl. `profile`, `knowledge_dir`, `seed_marker_file`, `quiz_bank_file`; 🟡 `quiz_length` (`SAR_QUIZ_LENGTH`) | `config.py`, `schemas.py` |
| A Vision | ✅ built + hardened | box-level ink segmentation (#14), stale-frame/atomic calibration/tunable thresholds (#22); threshold sweep in `scripts/tune_ink.py` |
| B RAG | ✅ built + hardened | non-blocking status (#15), content-addressed ids, list/delete documents, per-file selection; math seed docs (worksheet levels match the quiz bank) |
| C VLM | ✅ built + tuned, math profile, modes and conversation memory merged; 🟡 quiz stage `hint`, answer keys kept out of RAG context | ~2 s warm answers, cold first answer 22.5 s → 8 s (#16); missing-step → `error` 5/5 (#18); numeric quiz answers graded in code |
| D Voice | ✅ built + hardened | wake self-trigger guard (#17), push-to-talk wins over wake loop, silence skip, offline TTS fallback (#20) |
| E Frontend | ✅ built + polished; quiz card, mode picker and quiz bar merged | projector reconnect/offline handling (#21); operator dashboard `/dashboard` with four tabs (#23); dashboard follows the server's subject; pages show `display_text` when sent |
| F Server & ops | ✅ built + polished, tutor routes merged; 🟡 `failed` / `display_text` in history and socket events | `Assistant` pipeline, new APIs, warm-up (#23, #24); CI tests + lint (#13, #25); scripts (#19) |
| G Tutor / H Quiz bank / I Math speech | ✅ built and merged on `integration/math`; 🟡 review fixes (stage `hint`, quiz length) | contracts above |
| Phase 2 integrate | ✅ biology profile; 🟡 math profile | biology: full suite green in CI, live loop verified in the browser with `qwen2.5vl:3b`; math: full suite green on `integration/math`, live scenarios pending |
| Phase 3 hardware | 🟡 pending | camera + mic need macOS permissions; projector not yet connected |

## Open work — pick one up
Each item is a GitHub issue labelled [`open-work`](https://github.com/Enrique182004/Hackaton/issues?q=is%3Aopen+label%3Aopen-work); assign yourself before starting. Each item is self-contained. Branch `feat/<item>`, keep to the listed files, add tests.

1. [#2](https://github.com/Enrique182004/Hackaton/issues/2) **Hardware: camera + mic on the demo machine** (any) — grant Camera/Microphone to the terminal, run
   `.venv/bin/python scripts/hardware_check.py`, confirm the camera passes (≥ 28 FPS) and mic RMS > 0. Record results here.
2. [#3](https://github.com/Enrique182004/Hackaton/issues/3) **Hardware: projector calibration** (A) — connect projector as 2nd display, open `/` fullscreen on it,
   open `/dashboard` → *Board* tab → **Line up camera and projector**, verify `/api/frame.jpg?warped=1` matches the board. Tune `SAR_CHECKER_*` if detection fails.
3. [#4](https://github.com/Enrique182004/Hackaton/issues/4) ✅ fixed in #14 — **Ink segmentation per flowchart step** (A, `core/vision_loop.py`) — connected arrows currently merge a whole
   flowchart into one region, so the avatar can't point at a single box. Split boxes from connectors
   (e.g. erode before contouring, or detect rectangles with `approxPolyDP`).
4. [#5](https://github.com/Enrique182004/Hackaton/issues/5) ✅ fixed in #18 — **Prompt tuning** (C, `llm/vlm_client.py`) — when a step is missing the model answers `pointing`, not `error`;
   make the system prompt / few-shot examples push `error` + target the gap between steps.
5. [#6](https://github.com/Enrique182004/Hackaton/issues/6) ✅ fixed in #16 — **Latency** (C/F) — ~26 s per answer on an M4 Air with the 3B model. Try smaller image (downscale to 1024 px),
   `keep_alive` on Ollama, or an API backend (`SAR_VLM_BACKEND=openai`).
6. [#7](https://github.com/Enrique182004/Hackaton/issues/7) **Wake-word mode on hardware** (D) — test `SAR_LISTEN_MODE=wake` with the real mic; tune chunk length.
7. [#8](https://github.com/Enrique182004/Hackaton/issues/8) **Windows demo machine** (F) — run `scripts/setup_windows.ps1` end-to-end on Windows and fix anything that breaks.
8. [#9](https://github.com/Enrique182004/Hackaton/issues/9) **Scenario test cases** (all) — run each math and biology case from the lists above on the real board and record pass/fail.
9. [#10](https://github.com/Enrique182004/Hackaton/issues/10) ✅ fixed in #23 — **Warm-up at startup** (D/F) — preload Whisper and the Ollama model in a background thread so the first
   question isn't slow.
10. [#11](https://github.com/Enrique182004/Hackaton/issues/11) ✅ fixed in #17 — **Wake-word self-trigger** (D) — pause the wake loop while the avatar is speaking, so its own voice can't
    re-trigger it.
11. [#12](https://github.com/Enrique182004/Hackaton/issues/12) ✅ fixed in #15 — **Status poll during big uploads** (B/F) — `/api/status` waits on the store lock while embeddings run;
    cache the chunk count.
