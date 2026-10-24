# Build Plan — SAR Whiteboard Assistant

Two specs, one codebase. The subject is chosen by `SAR_PROFILE`:

- `math` (default): [sar_math_tutor_prompt.md](sar_math_tutor_prompt.md), the math tutor (step-by-step help,
  similar problems, quiz, learning a topic, phonetic speech).
- `biology`: [sar_whiteboard_assistant_prompt.md](sar_whiteboard_assistant_prompt.md), the original
  *Acanthamoeba* research assistant. It also defines the module layout both profiles share.

The work is split into **areas**. Each area owns its own files and talks to the others **only** through
the contracts below, so anyone can pick up one area without touching the rest. Where the code deliberately
differs from a spec, [Implementation notes](#implementation-notes-where-the-code-differs-from-the-specs) says
how and why.

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
| A | Vision & calibration | `core/homography.py`, `core/vision_loop.py`, `core/ink_crop.py`, `core/glare.py`, `tests/test_homography.py`, `tests/test_vision_loop.py`, `tests/test_ink_crop.py`, `tests/test_glare.py` | `schemas.InkRegion` |
| B | RAG knowledge base | `rag/ingestor.py`, `rag/vector_store.py`, `knowledge/math/*`, `knowledge/biology/*` (seed docs), `tests/test_rag.py`, `tests/test_answer_ingestion.py` | `schemas.RetrievedChunk`, H (answer stripping) |
| C | VLM reasoning | `llm/vlm_client.py`, `tests/test_vlm_client.py` | `schemas.*`, H, I, J |
| D | Voice in / voice out | `core/audio_trigger.py`, `llm/tts_engine.py`, `tests/test_audio.py`, `tests/test_tts.py` | J (language codes) |
| E | Projector frontend + dashboard | `templates/*.html`, `static/**` | Socket/HTTP contract |
| F | Server & ops | `app.py`, `assistant.py`, `envfile.py`, `scripts/*`, `tests/test_app.py`, `tests/test_assistant.py` | all contracts (uses fakes) |
| G | Tutor session + grading | `tutor.py`, `grading.py`, `tests/test_tutor.py` | H (`QuizProblem`) |
| H | Quiz bank | `rag/quiz_bank.py`, `tests/test_quiz_bank.py` | B (text extraction for `.pdf` / `.docx`) |
| I | Math speech | `llm/mathspeak.py`, `llm/mathspeak_es.py`, `tests/test_mathspeak*.py` | — |
| J | Language | `lang.py`, `tests/test_lang.py` | — |
| K | Phones (LAN mode) | `lan.py`, `static/sw.js`, `static/manifest.webmanifest`, `static/icons/`, `tests/test_lan.py`, `tests/test_mobile.py` | F wires it ([MOBILE.md](MOBILE.md)) |
| L | Desktop builds | `packaging/**`, `paths.py`, `scripts/build_*`, `.github/workflows/build.yml` | F (`app.main`) ([DESKTOP_BUILDS.md](DESKTOP_BUILDS.md)) |

## Phase 2 — Integrate
1. All unit tests green: `.venv/bin/python -m pytest -q`; lint clean: `.venv/bin/ruff check . && .venv/bin/ruff format --check .`.
2. `SAR_VLM_BACKEND=mock SAR_TTS_BACKEND=none .venv/bin/python -m biobench_sar`: both pages load, and a typed
   query moves the avatar.
3. Knowledge base: on a fresh `SAR_DATA_DIR`, the profile's seed files (`biobench_sar/knowledge/<profile>/`) are
   ingested at startup. Confirm they are listed on `/dashboard` → step 1 (*Worksheets & notes* in math,
   *Study files* in biology) and that `doc_chunks` in `/api/status` is > 0. Upload a new file and confirm
   `doc_chunks` grows and `/api/search` retrieves it. To test seeding by upload instead, start with
   `SAR_SEED_KNOWLEDGE=0` and an empty `SAR_DATA_DIR`. Test each profile with its own `SAR_DATA_DIR` (otherwise
   both share one knowledge base, see Shared below).
4. Switch to a real VLM (`ollama pull qwen2.5vl:3b`, `SAR_VLM_BACKEND=ollama`) and benchmark with
   `.venv/bin/python -m scripts.bench_vlm -n 3`. Pass/fail scenarios: `--scenarios --profile math` (math cases)
   or `--scenarios` (`--profile biology` is the default: the flowchart cases). `--only H,QC` runs only those
   scenarios, and `--raw` grades the model without `evaluate`'s math post-processing.
5. Math tutor: the math scenario test cases (below) pass with the mock backend for the control flow and with
   `qwen2.5vl:3b` for the wording.
6. Desktop builds: `.venv/bin/python packaging/launcher.py --self-test` prints `SELF-TEST OK`.

## Phase 3 — Hardware test
The ordered runbook, with expected results for each step, is [HARDWARE_DAY.md](HARDWARE_DAY.md)
([HARDWARE_DAY.es.md](HARDWARE_DAY.es.md) in Spanish). In short:
1. `cp sar.env.example sar.env` (the classroom kit: C930e 1920×1080, Epson 1280×800), then
   `.venv/bin/python scripts/hardware_check.py`: camera FPS (PASS at ≥ 28, target ~30), microphone level, TTS
   playback, Tesseract, Ollama and the model, displays.
2. Projector: open `/` fullscreen on the projector display (`F`), then on your laptop open `/dashboard` →
   *Board* tab → **Line up camera and projector**. The Board checklist shows no glare warning.
3. Write an equation on the board and check the lined-up view on the *Board* tab (or
   `/api/frame.jpg?warped=1`; without `?warped=1` it is the raw camera frame) lines up with the board.
4. Ask a question by voice (Space on the projector page / **Talk** on the dashboard). The avatar walks to the
   ink, the ink glows and the answer is spoken. Repeat in Spanish.
5. Scenario test cases (below) pass end-to-end for both profiles.

---

## Contracts

All paths are inside the `biobench_sar` package. Imports are package-absolute
(`from biobench_sar.core.homography import Calibration`).

### Shared (`schemas.py`, `config.py`)
- `InkRegion` TypedDict `{x, y, w, h, cx, cy}` (projector coords).
- `RetrievedChunk(text, source, score, metadata)`.
- `AvatarAction(spoken_text, animation_state, target_x, target_y)` + `.to_dict()` (the 4 spec keys, always).
- `parse_vlm_response(text, projector_size) -> AvatarAction`, which never raises: it strips markdown fences, finds
  the `{...}` object, defaults the state to `idle` and clamps the target to the projector.
- `error_action(message, projector_size) -> AvatarAction`.
- `Settings`: every knob is an env var `SAR_*` (the full table with defaults is in the
  [README](../README.md#configuration)). Invalid numbers fall back to the default. In particular:
  - `profile: str` — env `SAR_PROFILE`, default `"math"`. Allowed `"math" | "biology"`, anything else → `"math"`.
  - `knowledge_dir: Path` (property) — `PACKAGE_DIR / "knowledge" / profile`, the seed docs ingested at startup.
    Biology seed docs live in `knowledge/biology/` and math seed docs in `knowledge/math/` (`quiz_bank.csv` with
    header `problem,answer,topic,level`, `tutoring_guide.md`, and `worksheet_linear_equations.md` with `## Level N`
    headings that match the CSV's levels).
  - `seed_marker_file: Path` (property) — `chroma_dir / ".seeded"`, the profiles whose seed docs are already in
    the store. Each profile is seeded once per store, and deleting `chroma_dir` seeds it again.
  - `quiz_bank_file: Path` — env `SAR_QUIZ_BANK_FILE`, default `DATA_DIR / "quiz_bank.json"`.
  - `quiz_length: int` — env `SAR_QUIZ_LENGTH`, default `10`. The most problems one quiz asks, and how many the
    code invents when no ticked file has any.
  - `language: str` — env `SAR_LANGUAGE`, `"auto" | "en" | "es"` (`lang.normalize_language`), default `"auto"`.
  - `projector_width` / `projector_height` — default `1280` / `800` (`config.PROJECTOR_SIZE`, the kit's Epson WXGA).
  - `camera_width` / `camera_height` — default `1280` / `720`. `sar.env.example` sets `1920` / `1080` for the kit.
  - `ink_max_regions` — `SAR_INK_MAX_REGIONS`, else the profile's default (`config.INK_MAX_REGIONS`: math 20,
    biology 10). `ink_kwargs` (property) holds the `find_ink_regions` keyword arguments.
  - `lan: bool` — on for `SAR_LAN=1` **or** any non-loopback `host` (`lan.is_lan_mode`). `SAR_LAN=1` makes
    `host` default to `0.0.0.0`. `lan_hostnames: tuple[str, ...]` — `lan.parse_hostnames(SAR_LAN_HOSTNAMES)`.
  - Both profiles share one `SAR_DATA_DIR`: one store, one file selection, one quiz bank. Run each profile with
    its own `SAR_DATA_DIR` (e.g. `data/biology`) to keep one subject's files out of the other's answers.
- `envfile.load_env_file()` runs in `__main__` before `Settings` exists, so `sar.env` can set `SAR_DATA_DIR`
  too. Real environment variables win, and `SAR_ENV_FILE` names another file.

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
    def load(cls, path: Path, expected_projector_size: tuple[int, int] | None = None) -> "Calibration | None"
    @classmethod
    def identity(cls, camera_size: tuple[int, int], projector_size: tuple[int, int]) -> "Calibration"  # scales camera frame to projector
def calibrate_from_frame(frame: np.ndarray, projector_size: tuple[int, int], cols: int = 9, rows: int = 6) -> Calibration | None
```

### A — `core/vision_loop.py`, `core/ink_crop.py`, `core/glare.py`
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
def find_ink_regions(frame: np.ndarray, min_area: int = 150, max_regions: int = 10, block_size: int = 31,
                     c: int = 15, small_merge_px: int = 24) -> list[InkRegion]  # dark ink on light board, largest first;
                                                                              # specks join the nearby symbol
def crop_ink_view(raw_frame, calibration, regions, *, pad=48, max_side=1600, ...) -> InkCrop | None
    # the inked area rendered from the RAW frame at camera density, so decimals and minus signs survive the
    # model's ~1024 px downscale; InkCrop(jpeg, roi (projector px), size, scale)
def measure_glare(frame, *, clip_level=250, min_blob_frac=0.002, severe_frac=0.01) -> GlareReport
    # near-white flat hotspots (the bulb's reflection); coloured projected light (Pip) is not glare
```

### B — `rag/ingestor.py`, `rag/vector_store.py`
```python
SUPPORTED_EXTENSIONS: frozenset[str]  # .pdf .txt .md .csv .docx .pptx .png .jpg .jpeg
class UnsupportedFileType(ValueError)
@dataclass(frozen=True)
class Chunk: text: str; source: str; metadata: dict[str, str | int]
def chunk_paragraphs(text: str, max_chars: int = 1000) -> list[str]
def ingest_file(path: str | Path, *, strip_answers: bool = False) -> list[Chunk]
    # strip_answers (math profile): answer keys come out of the whole document before chunking
    # (quiz_bank.strip_document_answers; CSVs lose their answer column), so the store never holds one

class VectorStore:
    def __init__(self, persist_dir: Path, collection: str = "biobench", embed_model: str = "all-MiniLM-L6-v2",
                 embedder: Callable[[list[str]], list[list[float]]] | None = None) -> None  # embedder injectable for tests
    def add_chunks(self, chunks: list[Chunk]) -> int   # upsert, stable ids; returns number added
    def replace_source(self, source: str, chunks: list[Chunk]) -> tuple[int, int]  # (added, removed)
    def query(self, text: str, k: int = 4, sources: list[str] | None = None) -> list[RetrievedChunk]  # None = all files, [] = nothing
    def count(self) -> int; def clear(self) -> None
```
**Answer-key stripping** (H owns the patterns, B and C call them) covers, in English and Spanish:
`Answer:` / `Answers:` / `Answer key` sections, `Answer x = 6` (no colon), `Answer - x = 6`, `The answer is 8.`,
`Ans.`, `Respuesta:` / `Respuesta correcta:` / `Respuesta final:`, `Resultado`, `R:`, `Resp.`, `Sol` / `Sol.` /
`Solución`, `Result`, `Q:/A:` and `Q1./A1.` pairs, the full-width colon `：`, answer tables and CSV answer
columns. It runs at ingestion (`strip_document_answers`) and again on every retrieved chunk at prompt time
(`strip_chunk_answers`). A leak can't change a score (grading is in code), but it would put the answer in a
help-mode prompt.

### C — `llm/vlm_client.py`
```python
MODES = ("help", "practice", "quiz")
def system_prompt(profile: str, mode: str, language: str = "en") -> str  # persona + mode rules + strict JSON schema
def build_prompt(query: str, context: list[RetrievedChunk], ink_regions: list[InkRegion], projector_size: tuple[int, int],
                 *, profile: str = "biology", mode: str = "help", task: dict | None = None,
                 recent_turns: list[dict[str, str]] | None = None, language: str = "en", crop: InkCrop | None = None) -> str
class VLMClient:
    def __init__(self, backend: str, model: str, base_url: str, api_key: str | None = None, timeout: float = 90,
                 *, max_image_side: int = 1024, profile: str = "biology") -> None
    def warmup(self) -> None  # loads the Ollama model ahead of the first question; never raises
    def evaluate(self, query: str, image_jpeg: bytes | None, context: list[RetrievedChunk],
                 ink_regions: list[InkRegion], projector_size: tuple[int, int],
                 *, mode: str = "help", task: dict | None = None,
                 recent_turns: list[dict[str, str]] | None = None,
                 language: str = "en", crop: InkCrop | None = None) -> AvatarAction  # never raises
# backends: "ollama" (POST /api/chat, images base64, format json), "openai" (OpenAI-compatible /v1/chat/completions), "mock" (points at largest ink region)
```
- `recent_turns`: `[{"q": str, "a": str}]`, oldest first, at most 3, included in the prompt as
  "Conversation so far" so follow-ups ("make one up", "I did it") make sense.
- `language`: `"en" | "es"`, the reply language. Prompts, few-shot examples, code's own texts and mathspeak
  follow it. A reply in the wrong language is asked for again.
- Profile `math`: the returned `spoken_text` goes through `mathspeak.to_speech` (`mathspeak_es.to_speech_es` in
  Spanish). The reply before that is kept as `display_text` (normal notation, for the screens). A failed call
  (timeout, model unreachable, unreadable reply) comes back with `failed=True`.
- **Helper keys (5-key schema in quiz and practice).** The quiz prompt asks for `{"board_answer": str, ...the 4
  keys}` and the practice prompt for `{"board_problem": str, ...}`. They are internal: code reads them (to
  grade, or to know which problem the board holds) and they never reach `AvatarAction.to_dict()`, so every
  socket event still carries the 4 spec keys (plus `display_text`, `language` and `highlight`, see E/F).
- `task` passed to `evaluate`:

| Task | Dict |
| --- | --- |
| help | `None` |
| practice | `{"kind": "practice"}` |
| quiz | `{"kind": "quiz", "stage": "dictate" \| "check" \| "hint", "number": int, "total": int, "problem": str \| None, "answer": str \| None, "next_problem": str \| None}`. `problem` is always set: a worksheet problem, or one the code invented and solved (`quiz_bank.invent_problem`). `answer` is the expected answer, for code only. (The old rule was "`answer` is set only in stage `check`"; now it travels with every stage, and no prompt shows it.) `next_problem` is set only in `check`. A "repeat the problem" request is stage `dictate` again |
| learn | `{"kind": "learn", "topic": str, "stage": "offer" \| "example" \| "own"}` |

- **Answers never reach the model except when checking — and since grading moved into code, not even then.**
  Quiz answers are graded in code (`grading.grade`, contract G) against the
  model's `board_answer`. The model only copies the bottom line of the student's writing, its own verdict is
  never used to award a point, and the next problem is never in a check prompt (the model copied it as the
  student's answer). Profile `math` also strips answer keys from the RAG context in every task (contract B).
- **Help-mode answer guard.** The board's equation (the model's `board_problem` read, or the quiz problem) is
  solved with sympy. If `spoken_text` or `display_text` states that solution as the final answer ("x = 5",
  "x is 5", "x equals five", "la respuesta es cinco", "x es igual a 5"), the model is asked once more with a
  note. Whatever still states it has the offending sentence(s) removed before it is spoken.
- **Targets are snapped in code**, because a 3B model copies the few-shot pixels or a box corner.
  `pointing` and `error` (and `celebrating` when there is ink) go to the ink region nearest the model's target,
  or to the region the answer is about. `error` and `celebrating` land on the centre, and `pointing` on a point
  just left of the region. `idle` and `walking` go to a safe spot that covers no ink: the side of the board
  farthest from the ink, at mid-height. With no ink at all, drawing states become `idle` (except in learn
  tasks).

- Math semantics in the JSON schema (plain language for a middle/high-school student, short because it is
  spoken):

| Task | `spoken_text` | `animation_state` / target |
| --- | --- | --- |
| help | Explains ONLY the next operation, never the final answer (guarded in code) | `pointing` at the relevant part of the equation's ink region, highlighted |
| practice | Reads the board's equation and dictates a new one of the same difficulty (constants/variables swapped). If the reply echoes the board or misreads it, code dictates `quiz_bank.similar_problem(board)` | `idle` |
| quiz `dictate` | Reads the problem out | `idle` |
| quiz `check` | Correct (code says so): praise, and the next problem is read out. Wrong: one hint, never the answer. Unreadable (`grade()` is `None`): "I couldn't read your answer clearly. Ask me to check it again." (or the Spanish version), not graded | `celebrating` if correct, `error` if wrong, `idle` if not graded |
| quiz `hint` | The student asked for help on the current problem ("I'm stuck", "hint", "help"): one hint for the next step, never the answer. Nothing is recorded and the quiz stays on the problem | `pointing` at the ink to work on |
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
                 silence_rms: float = SILENCE_RMS,  # wake chunks quieter than this skip Whisper
                 language: str = "auto") -> None    # "auto": one Whisper language-ID pass limited to en/es
    def start(self) -> None; def stop(self) -> None   # "wake" mode starts a background listener
    def trigger(self) -> bool                          # push-to-talk (non-blocking); False if already recording
    def submit_text(self, text: str) -> None           # typed query -> on_query
    def transcribe(self, audio: np.ndarray) -> str     # float32 mono 16 kHz; lazy-loads whisper
    def set_language(self, setting: str) -> None       # the dashboard switch; en/es force the decode language
    available: bool                                    # False if no mic / pyaudio -> app falls back to text
    last_language: str                                 # "en" / "es" of the query being delivered
WAKE_ALIASES = {"assistant": ("asistente",)}
class TTSEngine:
    def __init__(self, backend: str = "edge", voice: str = "en-US-AriaNeural", out_dir: Path = ...,
                 voices: dict[str, str] | None = None) -> None
    def speak(self, text: str, lang: str | None = None) -> str | None
        # edge: writes mp3 to out_dir, returns file name for browser playback (en-US-AriaNeural / es-MX-DaliaNeural);
        # say/pyttsx3: plays locally (non-blocking) with the best installed voice for lang (es_MX first), returns None;
        # falls back edge → say → pyttsx3 → none
```

### G — `tutor.py`
```python
MODES = ("help", "practice", "quiz")
def detect_mode(text: str) -> str | None
    # "quiz me" / "start a quiz" -> "quiz"; "stop the quiz" -> "help";
    # "similar problem" / "another one like this" / "more practice" -> "practice";
    # "help" / "next step" / "I'm stuck" / "hint" -> "help"; anything else -> None
def is_quiz_stop(text: str) -> bool  # an explicit "stop / end / quit the quiz": the only words that leave a running quiz
def is_repeat_request(text: str) -> bool  # "repeat the problem", "what was the problem?": dictate it again
def detect_learn_request(text: str, *, board_in_use: bool = False) -> str | None
    # topic: "I want to learn how to solve fractions" -> "solving fractions"; "teach me equations";
    # "how do I add fractions?" only when it is a general how-to, not about the board
def detect_offer_reply(text: str) -> str | None
    # "make one up" / "give me one" / "you choose" / "yes make one" -> "generate";
    # "I have one" / "I've got my own" / "I'll write mine" -> "own"; else None
# *_any(text, language) variants read the detected language first, then the other (lang.py has the Spanish ones)
class TutorSession:  # thread-safe
    mode: str
    def set_mode(self, mode: str) -> None
    def start_quiz(self, problems: list[QuizProblem]) -> None  # at most Settings.quiz_length problems are asked;
                                                               # none -> quiz_length problems invented in code
    def stop_quiz(self) -> None
    def current(self) -> QuizProblem | None
    dictated: bool
    def mark_dictated(self) -> None
    def record(self, correct: bool) -> None          # advances to the next problem on correct
    pending_stop: bool                               # a stop intent during an active quiz, waiting for "yes"
    learn_topic: str | None
    offer_pending: bool
    def start_learning(self, topic: str) -> None     # sets offer_pending
    def resolve_offer(self) -> None                  # clears offer_pending, keeps the topic
    def turns(self) -> list[dict[str, str]]          # last 3 {"q", "a"} pairs, oldest first
    def add_turn(self, q: str, a: str) -> None
    def to_dict(self) -> dict
    # {"mode": str, "learn_topic": str | None, "confirm_stop": bool,
    #  "quiz": None | {"active": bool, "number": int, "total": int, "score": int, "problem": str | None}}
    # score = problems right on the first try (a fixed answer moves on but does not score)
```

### G — `grading.py` (sympy)
```python
def parse_answers(text: str) -> ...      # a normalized set of answers, comparable regardless of order and form
    # "x = 5", "5", "x=2.5" == "5/2", "x = 5 or x = -3", "x = 5, y = 2", "(2, 3)",
    # Spanish words "x es igual a cinco", "x = 5 y x = -3"
def grade(student_text: str, expected_text: str) -> bool | None
    # True / False; None = either side can't be parsed -> the turn is not graded
```
The expected side is the problem's `answer`, or `quiz_bank.solve_problem(problem)` for an invented problem.

**Per question in the assistant (F), profile `math`:**
1. If `pending_stop`: yes / sí / claro / ok / dale / yeah → stop the quiz and speak its score summary. Anything
   else clears `pending_stop` and is handled normally from step 2 (so an answer is graded).
2. If `offer_pending` and `detect_offer_reply(text)` → task learn stage `"example"` (generate) or `"own"`, then
   `resolve_offer()`. Only the very next turn answers the offer.
3. Elif `detect_learn_request(text)` → `start_learning(topic)` and task learn stage `"offer"`.
4. Else the mode logic. `detect_mode` switches mode: quiz starts a quiz from the ticked files' problems
   (`quiz_bank.pick_progressive`: at most `Settings.quiz_length`, spread evenly from easiest to hardest).
   - **While a quiz is active**, a stop intent (voice or typed) does not stop it. It sets `pending_stop`, and the
     assistant asks "Do you want to end the quiz? Say yes to stop, or keep going." / "¿Quieres terminar el quiz?
     Di «sí» para terminar o sigue con tu respuesta." (`idle`). The dashboard's Stop quiz button
     (`POST /api/tutor/quiz/stop`) stops at once.
   - "help" / "I'm stuck" / "hint" stay in the quiz. A problem not yet dictated (or a repeat request) gets stage
     `"dictate"` + `mark_dictated()`. Once it is dictated, a help request gets stage `"hint"` (nothing recorded),
     and anything else gets stage `"check"`.
   - In `"check"` the result is `grade(board_answer, expected)`: `True` → `celebrating` + `record(True)`, `False`
     → `error` + `record(False)`, `None` → not graded. An empty board is not graded either.
5. Always pass `recent_turns=tutor.turns()` and call `add_turn(text, spoken_text)` after the answer (not after a
   failed model call).

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
def pick_progressive(problems: Sequence[QuizProblem], length: int) -> list[QuizProblem]  # even spread, easiest first
def solve_problem(text: str | None) -> str | None        # the answer of a linear problem, or None
def similar_problem(problem: str, rng: random.Random | None = None) -> str | None  # same shape, new numbers
def invent_problem(number: int, total: int, rng: random.Random | None = None) -> QuizProblem
    # with its answer: one-step equations first, x on both sides last
def strip_document_answers(parts: Sequence[str]) -> list[str]           # a document's pages/parts, answers removed (contract B)
def strip_chunk_answers(chunks: Sequence[tuple[str, str]]) -> list[str]  # (source, text) chunks at prompt time
class QuizBank:  # persisted as JSON at Settings.quiz_bank_file
    def __init__(self, path: Path) -> None
    def replace_source(self, source: str, problems: list[QuizProblem]) -> None
    def remove_source(self, source: str) -> None
    def problems(self, sources: list[str] | None = None) -> list[QuizProblem]  # ordered by level, then file order
```
Quiz problems come only from ticked (selected) files. One quiz asks at most `Settings.quiz_length` of them.

### I — `llm/mathspeak.py`, `llm/mathspeak_es.py`
```python
def to_speech(text: str) -> str      # math notation inside normal sentences -> words a TTS engine reads naturally
def to_speech_es(text: str) -> str   # the same in Mexican Spanish: "dos tercios", "x al cuadrado", "es igual a"
def has_math_symbols(text: str) -> bool
```

| Written | Spoken | Hablado |
| --- | --- | --- |
| `2/3` | two thirds | dos tercios |
| `x^2` | x squared | x al cuadrado |
| `3x+5=12` | three x plus five equals twelve | tres x más cinco es igual a doce |
| `-4` | negative four | menos cuatro |
| `0.5` | zero point five | cero punto cinco |
| `25%` | twenty five percent | veinticinco por ciento |
| `sqrt(9)`, `√9` | the square root of nine | la raíz cuadrada de nueve |
| `*`, `×`, `·` | times | por |
| `÷` | divided by | entre |
| `<=`, `>=`, `<`, `>` | words (less than or equal to, …) | words (menor o igual que, …) |

Prose without math is returned unchanged, and `to_speech` is idempotent.

### J — `lang.py`
```python
def normalize_language(code: str | None) -> "auto" | "en" | "es"   # "es-MX" / "Español (México)" -> "es"
def detect_language(text: str, default: str = "en") -> "en" | "es"
def detect_mode_es(text) / is_quiz_stop_es(text) / detect_learn_request_es(text) / detect_offer_reply_es(text)
```

### K — `lan.py` (phones; the wiring and security model are in [MOBILE.md](MOBILE.md))
`is_lan_mode`, `parse_hostnames`, `is_local_request`, `is_allowed_host`, `is_request_allowed`, `JoinPin`
(six-digit PIN, per-device and server-wide lockouts, hashed 12 h device tokens), `cookie_kwargs()` (`sar_lan`,
`HttpOnly`, `SameSite=Lax`, `Max-Age` 12 h), `PhoneSessions`, `SocketIOGate`, `local_ip_addresses`, `join_url`.

### E/F — HTTP + Socket.IO contract
HTTP (F implements, E calls). Every `/api/*` response sends `Cache-Control: no-store`.
| Method | Path | Body / result |
| --- | --- | --- |
| GET | `/` | projector UI (`templates/index.html`) |
| GET | `/dashboard`, `/precharge` | operator dashboard (`templates/precharge.html`, `static/js/dashboard/`) |
| GET | `/api/config` | active non-secret settings, including `"profile"`, `"camera_size"`, `"projector_size"`, `"language"`, `"current_language"` |
| GET | `/api/language` | `{"setting": "auto" \| "en" \| "es", "current": "en" \| "es"}` |
| PUT | `/api/language` | `{"language": "auto" \| "en" \| "es"}` → the `GET` body; 400 otherwise. Also pushed as a `language` event |
| GET | `/api/history` | `{"interactions": [...]}` newest first; each item `{id, at, query, spoken_text, animation_state, target_x, target_y, sources, ink_regions, had_frame, latency_s, error, mode, task_kind, display_text, language, highlight}` (`ink_regions` = count; `task_kind` = the task's `kind` or `null`; `display_text` = the answer in normal notation; `highlight` see below) |
| GET | `/api/documents` | `{"documents": [{"source", "type", "chunks", "active": bool}], "total_chunks": int, "active_count": int, "all_active": bool}`. `active` = answers may use this file; `all_active` = no explicit selection (every file, default) |
| PUT | `/api/documents/selection` | `{"all": true}` → every file active, or `{"sources": ["a.md", ...]}` → exactly those active (unknown names ignored; `[]` = none, answers skip RAG); non-list `sources` / other body → 400. Returns the `GET /api/documents` body. Persisted to `SAR_SELECTION_FILE` |
| DELETE | `/api/documents/<source>` | `{"deleted": n}`; 404 if unknown. Also drops it from the selection |
| GET | `/api/search?q=&k=` | `{"results": [{"text", "source", "score"}]}`, k clamped 1–10 |
| GET | `/api/ink` | `{"regions": [InkRegion], "size": [w, h]}`; 503 without a frame |
| POST | `/api/upload` | multipart `files[]` → `{"files": [{"name", "chunks", "replaced"?, "error"?}], "total_chunks": int}`; with an explicit selection, uploaded files become active |
| GET | `/api/status` | `{"fps", "camera_running", "calibrated", "doc_chunks", "active_documents", "total_documents", "vlm_backend", "vlm_model", "audio_available", "tts_backend", "listen_mode", "state", "busy", "last_latency_s", "uptime_s", "profile", "mode", "language", "current_language", "glare"}`. `glare` = `null` or `{"fraction", "hotspots": [[x, y, w, h]], "severe": bool}` (severe above 1 % of the board) |
| GET | `/api/tutor` | `TutorSession.to_dict()`, including `"confirm_stop": bool` |
| PUT | `/api/tutor/mode` | `{"mode": "help" \| "practice" \| "quiz"}` → `to_dict()`; 400 otherwise. Quiz mode starts a quiz on the next question or via `POST /api/tutor/quiz/start` |
| POST | `/api/tutor/quiz/start` | → `to_dict()`; starts even with 0 problems (the code invents them); `total` = number of problems in the ticked files capped at `SAR_QUIZ_LENGTH`, or `SAR_QUIZ_LENGTH` (default 10) when they have none |
| POST | `/api/tutor/quiz/stop` | → `to_dict()`; stops at once, no confirmation |
| POST | `/api/query` | `{"text": str}` → `202 {"accepted": true}` (result arrives over socket) |
| POST | `/api/listen` | push-to-talk trigger → `202` |
| POST | `/api/calibrate` | shows checkerboard, captures, solves → `{"ok": bool, "error"?: str}` |
| GET | `/calibration.png` | checkerboard at projector size |
| GET | `/api/frame.jpg` | latest raw camera frame; `?warped=1` = lined up to projector coordinates (debug preview) |
| GET | `/tts/<name>` | generated speech audio |
| GET | `/api/lan` | local only (403 from a phone) → `{"enabled": bool, "urls": [str], "pin": "123456" \| null, "phones": int}` |
| POST | `/api/lan/rotate` | local only → a new PIN, every device token revoked, every phone socket disconnected; returns the `GET /api/lan` body |
| POST | `/api/lan/disconnect` | local only → every device token revoked and every phone socket disconnected; the PIN stays |
| GET | `/join` | `templates/join.html` (variables `error: bool`, `language`); `Referrer-Policy: same-origin` |
| POST | `/join` | form field `pin` → right: sets the `sar_lan` cookie and `303` to `/dashboard`; wrong: the form again with `error` (and the lockout message while locked) |
| GET | `/sw.js` | the service worker, in LAN mode only; 404 otherwise |

**LAN mode gate** (on for `SAR_LAN=1` or any non-loopback `SAR_HOST`). A local request (loopback TCP peer AND a
loopback or allowed `Host`) passes. Every other request needs a valid `sar_lan` cookie, else it gets 401 for
`/api/*`, a refused Socket.IO connect, and a redirect to `/join` for pages. `/join` and static files are open.
`/join` uses `Referrer-Policy: same-origin`, not `no-referrer`: with `no-referrer`, browsers send `Origin: null`
on the form POST and `block_cross_origin` would refuse it. The cross-origin guard accepts the same-origin
`POST /join`. Templates receive `lan_enabled: bool`. The startup banner prints the phone URL and PIN.

**`highlight`**: `null` or `[x, y, w, h]` in projector pixels, the ink region the answer is about (help, hint,
error, check, pointing). It is carried by `avatar_update`, `speak` and history items.

Socket.IO events (server → client): `avatar_update {animation_state, target_x, target_y, spoken_text,
display_text, language, highlight}`, `speak {url: str | null, text, display_text, language, highlight}`,
`status {state: "idle"|"listening"|"transcribing"|"thinking"|"speaking", detail?}`, `transcript {text}`,
`show_calibration {url}`, `hide_calibration {}`, `interaction {...history item}`,
`documents_changed {...GET /api/documents body}` (after upload, delete and selection changes),
`tutor {...TutorSession.to_dict()}` (whenever the mode, quiz, learn topic or `confirm_stop` changes),
`language {setting, current}`. Both pages show `display_text` in place of the spoken words.
On connect the server sends the current `status`, `tutor` and `language` to the new client. The dashboard re-reads
`/api/config` on every connect, and also takes the subject from `/api/status` `profile`, so a server restarted with
another `SAR_PROFILE` is followed without a reload. The math mode picker stays hidden until the subject is known.
Busy handling: `/api/query` → 409 while a query runs; `/api/listen` → 409 already recording, 503 no mic;
`/api/calibrate` → 503 without a frame, 409 while another calibration runs.

Client → server: `text_query {text}`, `trigger_listen {}`, `ui_rects {"rects": [[x, y, w, h], ...]}`.
`ui_rects` comes from the projector page: the boxes of the projected UI in projector pixels (avatar, speech
bubble, quiz card, highlight), sent whenever they change and at least every 1 s while visible. The server forwards
them to `Assistant.set_ui_rects(rects)`. Rects older than 3 s are ignored, and an ink region overlapping any UI
rect by more than 50 % of its own area is dropped before the model and before snapping. Then the camera never
reads Pip, the bubble or the quiz card as the student's ink.

Projector UI: hotkey **Space** = push-to-talk, **T** = type a query, **C** = toggle debug overlay, **F** =
fullscreen. The page also listens to `tutor` and shows the current quiz problem in normal notation (e.g.
`3x + 5 = 20`) in a small, faint top-left card (thin outline, no glow). The card is hidden when no quiz is
running and while the status is `listening`, `transcribing` or `thinking`, and it is masked through `ui_rects`
anyway. When `highlight` is set, the page draws a glowing outline around that box, pulsing for about 2 s and
fading after about 8 s or at the next update.

---

## Implementation notes (where the code differs from the specs)

The specs stay the source of truth for what the assistant is for. These are deliberate departures in how it is
built, each with its reason.

| Spec says | The code does | Why |
| --- | --- | --- |
| Audio stack includes `SpeechRecognition` (biology §3) | Raw PyAudio capture (`core/audio_trigger.py`) + local Whisper; `SpeechRecognition` is not a dependency | Whisper already does the recognition; PyAudio gives the exact 16 kHz float buffer it needs and the wake-word chunking, with one dependency fewer |
| The JSON schema is the same 4 keys in every mode (math §6) | Quiz and practice prompts ask for a 5th helper key, `board_answer` / `board_problem`, removed before `AvatarAction` | Grading and the practice check need what the model read on the board; the app and the pages still only see the 4 keys |
| Quiz problems are queried from ChromaDB (math §5B) | Parsed into the quiz bank at upload (`rag/quiz_bank.py`); a quiz takes at most `SAR_QUIZ_LENGTH` from the ticked files, spread evenly from easiest to hardest | Retrieval can't keep an order of difficulty or the answers that grading needs |
| With no worksheet, the AI makes problems up | The code invents and solves them (`quiz_bank.invent_problem`) | A 3B model asked to invent mostly echoed the student, and an invented problem needs a known answer to be graded |
| Quiz mode "evaluates the student's physical ink answer" | The model only copies the answer into `board_answer`; `grading.grade` (sympy) decides; `None` = not graded | The 3B model called nearly every wrong answer correct; code grading is exact, handles several values (`x = 5 or x = -3`, systems) and Spanish words |
| Help never gives the final answer | Prompted, then guarded in code: re-ask once, then cut the sentence | Small models still blurt the answer sometimes |
| Quiz has dictate and evaluate | A third stage, `hint` (`pointing`), for "I'm stuck" during a quiz; a stop asks for confirmation; "repeat the problem" dictates again | Students ask for help mid-quiz, and a stray "stop" shouldn't lose the score |
| RAG context is the retrieved chunks | In math, answer keys are stripped from the whole document at ingestion and from every chunk at prompt time | An answer in a prompt gets read out or copied as the student's |
| `target_x` / `target_y` are the model's | Snapped in code to the ink (pointing/error/celebrating) or a free spot (idle/walking) | The model copies few-shot pixels and box corners |
| "Project a highlight on the relevant part of the equation" (math §5B) | `highlight` box sent with the answer; the projector draws a pulsing outline | — (this is the spec, now built) |
| The camera reads the student's ink | Projected UI boxes (`ui_rects`) are masked from ink detection | Bright projected text and shapes were segmented as ink |
| Camera at 30+ FPS | The camera loop never waits on the model; the hardware check passes at ≥ 28 FPS | A 30 fps webcam that measures 29.x must still pass |
| English only | English and Spanish end to end (`lang.py`, `mathspeak_es.py`, es-MX voices, translated dashboard) | The classroom is bilingual |
| `/precharge` dashboard (biology §5) | `/dashboard`, with `/precharge` kept as an alias | Clearer name for teachers |

---

## Scenario test cases — Math (default profile)
Seed docs in `biobench_sar/knowledge/math/` (tick `quiz_bank.csv` and `worksheet_linear_equations.md` on the
dashboard). For every case, `spoken_text` contains no `^`, `/`, `*` or `=`. The help and practice cases use
equations that are not in the seed files, so a pass never depends on an answer key the model should not see.
1. **Next step, not the answer:** write `4x + 3 = 19`, say "I'm stuck" → `pointing` just left of the equation,
   which glows; explains only the next operation ("subtract three from both sides"); never says "x equals
   four".
2. **Phonetic speech:** write `x^2 + 7x + 10 = 0`, ask for help → the answer reads "x squared plus seven x plus
   ten equals zero", and fractions on the board such as `2/3` are read as "two thirds".
3. **Similar problem:** with `3x - 4 = 11` on the board, say "give me a similar problem" → `idle`, dictates a
   new two-step equation with different numbers (e.g. "five x minus two equals thirteen").
4. **Quiz, correct answer:** say "quiz me" → `idle`, dictates problem 1 of N, and the projector shows it
   top-left in normal notation. Write the right answer and say "check my answer" → `celebrating`, then the
   next problem is dictated and the card advances.
5. **Quiz, wrong answer:** write a wrong answer → `error`, one hint, the correct answer is never spoken, and the
   card stays on the same problem.
6. **Quiz, several values:** on the default quiz's last problem (the quadratic x² − 2x − 15 = 0), writing
   `x = -3 or x = 5` is correct and `x = 4 or x = -4` is wrong, whatever the model says.
7. **Quiz, unreadable answer:** an answer grading can't parse → "I couldn't read your answer clearly. Ask me to
   check it again.", no try counted.
8. **Quiz, ask for a hint:** after a problem is dictated, say "I'm stuck" → `pointing`, one hint for the next
   step (stage `hint`), never the answer; the quiz keeps going on the same problem and the score is unchanged.
9. **Quiz length and order:** with both seed files ticked (43 problems), "quiz me" starts a quiz of 10
   (`SAR_QUIZ_LENGTH`), spread evenly from the level-1 fraction problems to the level-5 systems and quadratics,
   never back to an easier level.
10. **Quiz without worksheets:** untick every quiz file and say "start a quiz" → the quiz starts with total 10
    (`SAR_QUIZ_LENGTH`), with problems the code invented (one-step equations first), graded like any other.
11. **Stop the quiz:** say "stop the quiz" → "Do you want to end the quiz? Say yes to stop, or keep going."; say
    "yes" → the score summary, back to help mode, the card is hidden. Saying an answer instead keeps the quiz
    and grades it. The dashboard's Stop quiz button stops without asking.
12. **Learn a topic, empty board:** say "I want to learn how to solve fractions" → `idle`, one or two plain
    sentences on what a fraction is, then "Do you have an exercise in mind, or should I make one up for you?"
13. **Learn → make one up:** reply "make one up" → `idle`, dictates an easy starter (e.g. "one fifth plus two
    fifths") and invites the first step; then "I did it, what now?" continues from that problem.
14. **Learn → own exercise:** after the offer, reply "I have one" → `idle`, asks the student to write it on the
    board and then ask for help.
15. **Spanish:** with the switch on Auto, "Ayúdame con el siguiente paso", "Dame un problema parecido",
    "Hazme un quiz" and "Quiero aprender a resolver fracciones" behave like cases 1, 3, 4 and 12, in Spanish,
    with a Mexican voice and math in Spanish words.
16. **Projected UI is not ink:** with the speech bubble or quiz card showing over an empty part of the board,
    ask a question → the avatar never points at them, and they are never graded as the student's answer.
17. **Idle stays clear of the writing:** with an equation on the left half of the board, a dictation (`idle`)
    parks Pip on the right half, at mid-height.

## Scenario test cases — Environmental Biology (`SAR_PROFILE=biology`)
Seed docs in `biobench_sar/knowledge/biology/`. The assistant must:
1. Verify a drawn sampling flowchart for *Acanthamoeba* in Laguna de Bustillos (collection → filtration → NNA agar + *E. coli* lawn → incubation → microscopy → PCR).
2. Flag a missing sterile step (e.g. no autoclave / flame sterilization) with `animation_state: "error"` and point at it.
3. Explain how runoff from Leptosol vs. Feozem soils in Cuauhtémoc, Chihuahua affects turbidity/nutrients at sampling sites.
4. Celebrate (`celebrating`) when the procedure is complete and correct.

## Status (2026-10-24)
| Area | Status | Verified |
| --- | --- | --- |
| Shared contracts | ✅ done, incl. `profile`, `knowledge_dir`, `seed_marker_file`, `quiz_bank_file`, `quiz_length`, `language`, kit defaults (projector 1280×800); `lan` / `lan_hostnames` in the final round | `config.py`, `schemas.py`, `tests/test_config.py` |
| A Vision | ✅ built + hardened; high-res ink crop, glare check on the Board tab; UI masking (`ui_rects`) in the final round | box-level ink segmentation (#14), stale-frame/atomic calibration/tunable thresholds (#22), ink crop (#40), glare (#41); threshold sweep in `scripts/tune_ink.py` |
| B RAG | ✅ built + hardened; answer keys stripped at ingestion | non-blocking status (#15), content-addressed ids, list/delete documents, per-file selection; math seed docs (worksheet levels match the quiz bank) |
| C VLM | ✅ built + tuned, math profile, modes, memory, Spanish; grading, help-answer guard, target snapping and `highlight` in the final round | ~2 s warm answers, cold first answer 22.5 s → 8 s (#16); missing-step → `error` 5/5 (#18) |
| D Voice | ✅ built + hardened, English/Spanish in and out | wake self-trigger guard (#17), push-to-talk wins over wake loop, silence skip, offline TTS fallback (#20), es-MX voices |
| E Frontend | ✅ built + polished; Pip redesign (#62), EN/ES dashboard; highlight drawing and `ui_rects` in the final round | projector reconnect/offline handling (#21); operator dashboard `/dashboard` with four tabs (#23) |
| F Server & ops | ✅ built + polished; LAN mode wiring, stop confirmation in the final round | `Assistant` pipeline, APIs, warm-up (#23, #24); CI tests + lint (#13, #25); scripts (#19); classroom kit `sar.env.example` (#50) |
| G Tutor / H Quiz bank / I Math speech / J Language | ✅ built and merged; `grading.py` in the final round | contracts above |
| K Phones | 🟡 groundwork merged (`lan.py`, manifest, icons, `sw.js`); server wiring in the final round | `tests/test_lan.py`, `tests/test_mobile.py` |
| L Desktop builds | ✅ Windows installer + macOS arm64 app, smoke-tested in CI (#66) | the launcher's self-test prints `SELF-TEST OK` |
| Phase 2 integrate | ✅ both profiles | full suite green on `release/candidate`; live loop verified in the browser with `qwen2.5vl:3b` |
| Phase 3 hardware | 🟡 pending: runbook ready ([HARDWARE_DAY.md](HARDWARE_DAY.md)), kit is a C930e + Epson WXGA on a cart | camera + mic need OS permissions; projector not yet connected |

## Open work — pick one up
Each item is a GitHub issue labelled [`open-work`](https://github.com/Enrique182004/Hackaton/issues?q=is%3Aopen+label%3Aopen-work). Assign yourself before starting. Each item is self-contained: branch `feat/<item>`, keep to the listed files, add tests.

1. [#2](https://github.com/Enrique182004/Hackaton/issues/2) **Hardware: camera + mic on the demo machine** (any). Follow [HARDWARE_DAY.md](HARDWARE_DAY.md) parts A–C:
   grant Camera/Microphone, run `.venv/bin/python scripts/hardware_check.py`, and confirm the camera passes (≥ 28 FPS) and mic RMS > 0. Record the results here.
2. [#3](https://github.com/Enrique182004/Hackaton/issues/3) **Hardware: projector calibration** (A). [HARDWARE_DAY.md](HARDWARE_DAY.md) part D: connect the projector as a 2nd display, open `/` fullscreen on it,
   open `/dashboard` → *Board* tab → **Line up camera and projector**, and verify `/api/frame.jpg?warped=1` matches the board. Tune `SAR_CHECKER_*` if detection fails.
3. [#4](https://github.com/Enrique182004/Hackaton/issues/4) ✅ fixed in #14: **Ink segmentation per flowchart step** (A, `core/vision_loop.py`).
4. [#5](https://github.com/Enrique182004/Hackaton/issues/5) ✅ fixed in #18: **Prompt tuning** (C, `llm/vlm_client.py`), `error` + target the gap between steps.
5. [#6](https://github.com/Enrique182004/Hackaton/issues/6) ✅ fixed in #16: **Latency** (C/F).
6. [#7](https://github.com/Enrique182004/Hackaton/issues/7) **Wake-word mode on hardware** (D): test `SAR_LISTEN_MODE=wake` with the real mic, in English and Spanish; tune chunk length.
7. [#8](https://github.com/Enrique182004/Hackaton/issues/8) **Windows demo machine** (F): run `scripts/setup_windows.ps1` end-to-end on Windows and fix anything that breaks.
8. [#9](https://github.com/Enrique182004/Hackaton/issues/9) **Scenario test cases** (all): run each math and biology case from the lists above on the real board and record pass/fail.
9. [#10](https://github.com/Enrique182004/Hackaton/issues/10) ✅ fixed in #23: **Warm-up at startup** (D/F).
10. [#11](https://github.com/Enrique182004/Hackaton/issues/11) ✅ fixed in #17: **Wake-word self-trigger** (D).
11. [#12](https://github.com/Enrique182004/Hackaton/issues/12) ✅ fixed in #15: **Status poll during big uploads** (B/F).
