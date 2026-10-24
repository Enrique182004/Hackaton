# Build Plan — SAR Whiteboard Assistant

The spec ([sar_whiteboard_assistant_prompt.md](sar_whiteboard_assistant_prompt.md)) is split into **six
areas**. Each area owns its own files and talks to the others **only** through the contracts below,
so anyone can pick up one area without touching the rest.

**How to pick up an area:** find its GitHub issue (label `area:*`), assign yourself, branch
`feat/<area>`, keep to the files listed, and make sure `pytest tests/test_<area>*.py` passes.

Shared files (change only via PR that updates this doc): `biobench_sar/config.py`, `biobench_sar/schemas.py`.

Run everything from the repo root: `python -m biobench_sar`, `pytest`.

---

## Phase 1 — Build (parallel)

| # | Area | Owns | Depends on (contract only) |
| --- | --- | --- | --- |
| A | Vision & calibration | `core/homography.py`, `core/vision_loop.py`, `tests/test_homography.py`, `tests/test_vision_loop.py` | `schemas.InkRegion` |
| B | RAG knowledge base | `rag/ingestor.py`, `rag/vector_store.py`, `knowledge/*` (seed docs), `tests/test_rag.py` | `schemas.RetrievedChunk` |
| C | VLM reasoning | `llm/vlm_client.py`, `tests/test_vlm_client.py` | `schemas.*` |
| D | Voice in / voice out | `core/audio_trigger.py`, `llm/tts_engine.py`, `tests/test_audio.py`, `tests/test_tts.py` | — |
| E | Projector frontend | `templates/*.html`, `static/**` | Socket/HTTP contract |
| F | Server & ops | `app.py`, `scripts/*`, `tests/test_app.py` | all contracts (uses fakes) |

## Phase 2 — Integrate
1. All unit tests green: `pytest`.
2. `python -m biobench_sar` with `SAR_VLM_BACKEND=mock` — UI loads, typed query moves the avatar.
3. Upload seed docs on `/precharge`, confirm the RAG count grows and queries retrieve them.
4. Switch to a real VLM (`ollama pull qwen2.5vl:3b`, `SAR_VLM_BACKEND=ollama`).

## Phase 3 — Hardware test
1. `python scripts/hardware_check.py` — camera FPS (must be ≥ 30), microphone level, TTS playback, displays.
2. Projector: open `/` fullscreen on the projector display (F11), run calibration from `/precharge` → *Calibrate*.
3. Draw a box on the board → check `/api/frame.jpg` (warped view) lines up with the board.
4. Ask a question by voice (hotkey/button) → avatar walks to the ink and speaks.
5. Scenario test cases (below) pass end-to-end.

---

## Contracts

All paths are inside the `biobench_sar` package. Imports are package-absolute
(`from biobench_sar.core.homography import Calibration`).

### Shared (`schemas.py`, `config.py`) — done
- `InkRegion` TypedDict `{x, y, w, h, cx, cy}` (projector coords).
- `RetrievedChunk(text, source, score, metadata)`.
- `AvatarAction(spoken_text, animation_state, target_x, target_y)` + `.to_dict()`.
- `parse_vlm_response(text, projector_size) -> AvatarAction` — never raises.
- `error_action(message, projector_size) -> AvatarAction`.
- `Settings` — every knob is an env var `SAR_*` (see `config.py`).

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
    def query(self, text: str, k: int = 4) -> list[RetrievedChunk]
    def count(self) -> int; def clear(self) -> None
```

### C — `llm/vlm_client.py`
```python
SYSTEM_PROMPT: str  # expert research assistant persona + Acanthamoeba scenario + strict JSON schema
def build_prompt(query: str, context: list[RetrievedChunk], ink_regions: list[InkRegion], projector_size: tuple[int, int]) -> str
class VLMClient:
    def __init__(self, backend: str, model: str, base_url: str, api_key: str | None = None, timeout: float = 90) -> None
    def evaluate(self, query: str, image_jpeg: bytes | None, context: list[RetrievedChunk],
                 ink_regions: list[InkRegion], projector_size: tuple[int, int]) -> AvatarAction  # never raises
# backends: "ollama" (POST /api/chat, images base64, format json), "openai" (OpenAI-compatible /v1/chat/completions), "mock" (points at largest ink region)
```

### D — `core/audio_trigger.py`, `llm/tts_engine.py`
```python
class AudioTrigger:
    def __init__(self, on_query: Callable[[str], None], whisper_model: str = "base", wake_word: str = "assistant",
                 mode: str = "push", record_seconds: float = 5.0,
                 on_state: Callable[[str], None] | None = None) -> None  # on_state("listening"|"transcribing"|"idle")
    def start(self) -> None; def stop(self) -> None   # "wake" mode starts a background listener
    def trigger(self) -> None                          # push-to-talk: record, transcribe, on_query (non-blocking)
    def submit_text(self, text: str) -> None           # typed query -> on_query
    def transcribe(self, audio: np.ndarray) -> str     # float32 mono 16 kHz; lazy-loads whisper
    available: bool                                    # False if no mic / pyaudio -> app falls back to text
class TTSEngine:
    def __init__(self, backend: str = "edge", voice: str = "en-US-AriaNeural", out_dir: Path = ...) -> None
    def speak(self, text: str) -> str | None  # edge: writes mp3 to out_dir, returns file name for browser playback;
                                               # say/pyttsx3: plays locally (non-blocking), returns None; falls back say→pyttsx3→none
```

### E/F — HTTP + Socket.IO contract
HTTP (F implements, E calls):
| Method | Path | Body / result |
| --- | --- | --- |
| GET | `/` | projector UI (`templates/index.html`) |
| GET | `/dashboard`, `/precharge` | operator dashboard (`templates/precharge.html`, `static/js/dashboard/`) |
| GET | `/api/config` | active non-secret settings |
| GET | `/api/history` | `{"interactions": [...]}` newest first (query, spoken_text, animation_state, target, sources, latency_s, error) |
| GET | `/api/documents` | `{"documents": [{"source", "type", "chunks"}], "total_chunks"}` |
| DELETE | `/api/documents/<source>` | `{"deleted": n}`; 404 if unknown |
| GET | `/api/search?q=&k=` | `{"results": [{"text", "source", "score"}]}`, k clamped 1–10 |
| GET | `/api/ink` | `{"regions": [InkRegion], "size": [w, h]}`; 503 without a frame |
| POST | `/api/upload` | multipart `files[]` → `{"files": [{"name", "chunks", "error"?}], "total_chunks": int}` |
| GET | `/api/status` | `{"fps", "camera_running", "calibrated", "doc_chunks", "vlm_backend", "vlm_model", "audio_available", "state"}` |
| POST | `/api/query` | `{"text": str}` → `202 {"accepted": true}` (result arrives over socket) |
| POST | `/api/listen` | push-to-talk trigger → `202` |
| POST | `/api/calibrate` | shows checkerboard, captures, solves → `{"ok": bool, "error"?: str}` |
| GET | `/calibration.png` | checkerboard at projector size |
| GET | `/api/frame.jpg?warped=1` | latest camera frame (debug preview) |
| GET | `/tts/<name>` | generated speech audio |

Socket.IO events (server → client): `avatar_update {animation_state, target_x, target_y, spoken_text}`,
`speak {url: str | null, text}`, `status {state: "idle"|"listening"|"transcribing"|"thinking"|"speaking", detail?}`,
`transcript {text}`, `show_calibration {url}`, `hide_calibration {}`, `interaction {...history item}`.
On connect the server sends the current `status` to the new client.
Busy handling: `/api/query` → 409 while a query runs; `/api/listen` → 409 already recording, 503 no mic;
`/api/calibrate` → 503 without a frame, 409 while another calibration runs.
Client → server: `text_query {text}`, `trigger_listen {}`.
Projector UI: hotkey **Space** = push-to-talk, **T** = type a query, **C** = toggle debug overlay.

---

## Scenario test cases (Environmental Biology)
Seed docs in `biobench_sar/knowledge/`. The assistant must:
1. Verify a drawn sampling flowchart for *Acanthamoeba* in Laguna de Bustillos (collection → filtration → NNA agar + *E. coli* lawn → incubation → microscopy → PCR).
2. Flag a missing sterile step (e.g. no autoclave / flame sterilization) with `animation_state: "error"` and point at it.
3. Explain how runoff from Leptosol vs. Feozem soils in Cuauhtémoc, Chihuahua affects turbidity/nutrients at sampling sites.
4. Celebrate (`celebrating`) when the procedure is complete and correct.

## Status (2026-10-24)
| Area | Status | Verified |
| --- | --- | --- |
| Shared contracts | ✅ done | `config.py`, `schemas.py` |
| A Vision | ✅ built + hardened | box-level ink segmentation (#14), stale-frame/atomic calibration/tunable thresholds (#22); threshold sweep in `scripts/tune_ink.py` |
| B RAG | ✅ built + hardened | non-blocking status (#15), content-addressed ids, list/delete documents |
| C VLM | ✅ built + tuned | ~2 s warm answers, cold first answer 22.5 s → 8 s (#16); missing-step → `error` 5/5 (#18) |
| D Voice | ✅ built + hardened | wake self-trigger guard (#17), push-to-talk wins over wake loop, silence skip, offline TTS fallback (#20) |
| E Frontend | ✅ built + polished | projector reconnect/offline handling (#21); operator dashboard `/dashboard` (#23) |
| F Server & ops | ✅ built + polished | `Assistant` pipeline, new APIs, warm-up (#23, #24); CI tests + lint (#13, #25); scripts (#19) |
| Phase 2 integrate | ✅ | full suite 188 passed; live loop verified in the browser with `qwen2.5vl:3b` |
| Phase 3 hardware | 🟡 pending | camera + mic need macOS permissions; projector not yet connected |

## Open work — pick one up
Each item is a GitHub issue labelled [`open-work`](https://github.com/Enrique182004/Hackaton/issues?q=is%3Aopen+label%3Aopen-work); assign yourself before starting. Each item is self-contained. Branch `feat/<item>`, keep to the listed files, add tests.

1. [#2](https://github.com/Enrique182004/Hackaton/issues/2) **Hardware: camera + mic on the demo machine** (any) — grant Camera/Microphone to the terminal, run
   `python scripts/hardware_check.py`, confirm camera ≥ 30 FPS and mic RMS > 0. Record results here.
2. [#3](https://github.com/Enrique182004/Hackaton/issues/3) **Hardware: projector calibration** (A) — connect projector as 2nd display, open `/` fullscreen on it,
   click *Calibrate* on `/precharge`, verify `/api/frame.jpg?warped=1` matches the board. Tune `SAR_CHECKER_*` if detection fails.
3. [#4](https://github.com/Enrique182004/Hackaton/issues/4) ✅ fixed in #14 — **Ink segmentation per flowchart step** (A, `core/vision_loop.py`) — connected arrows currently merge a whole
   flowchart into one region, so the avatar can't point at a single box. Split boxes from connectors
   (e.g. erode before contouring, or detect rectangles with `approxPolyDP`).
4. [#5](https://github.com/Enrique182004/Hackaton/issues/5) ✅ fixed in #18 — **Prompt tuning** (C, `llm/vlm_client.py`) — when a step is missing the model answers `pointing`, not `error`;
   make the system prompt / few-shot examples push `error` + target the gap between steps.
5. [#6](https://github.com/Enrique182004/Hackaton/issues/6) ✅ fixed in #16 — **Latency** (C/F) — ~26 s per answer on an M4 Air with the 3B model. Try smaller image (downscale to 1024 px),
   `keep_alive` on Ollama, or an API backend (`SAR_VLM_BACKEND=openai`).
6. [#7](https://github.com/Enrique182004/Hackaton/issues/7) **Wake-word mode on hardware** (D) — test `SAR_LISTEN_MODE=wake` with the real mic; tune chunk length.
7. [#8](https://github.com/Enrique182004/Hackaton/issues/8) **Windows demo machine** (F) — run `scripts/setup_windows.ps1` end-to-end on Windows and fix anything that breaks.
8. [#9](https://github.com/Enrique182004/Hackaton/issues/9) **Scenario test cases 1–4** (all) — run each case from the list above on the real board and record pass/fail.
9. [#10](https://github.com/Enrique182004/Hackaton/issues/10) ✅ fixed in #23 — **Warm-up at startup** (D/F) — preload Whisper and the Ollama model in a background thread so the first
   question isn't slow.
10. [#11](https://github.com/Enrique182004/Hackaton/issues/11) ✅ fixed in #17 — **Wake-word self-trigger** (D) — pause the wake loop while the avatar is speaking, so its own voice can't
    re-trigger it.
11. [#12](https://github.com/Enrique182004/Hackaton/issues/12) ✅ fixed in #15 — **Status poll during big uploads** (B/F) — `/api/status` waits on the store lock while embeddings run;
    cache the chunk count.
