# CLAUDE.md — SAR Whiteboard Math Assistant (hackathon)

## Project focus (read first)
This repo builds ONE thing: the **SAR Whiteboard Assistant** — webcam + projector + whiteboard, with an
animated 2D AI avatar that sees marker work and reacts (Sense → Think → Act), grounded by a local RAG
knowledge base. Its subject is chosen by `SAR_PROFILE`:

- **`math` (default) — the math tutor.** [docs/sar_math_tutor_prompt.md](docs/sar_math_tutor_prompt.md) is
  the **source of truth for the subject**: tutoring behaviour, the three modes (step-by-step help, similar
  problems, quiz), learning a topic from an empty board, phonetic TTS formatting and the JSON schema. Read it
  before any work.
- **`biology` — kept as the second profile.** [docs/sar_whiteboard_assistant_prompt.md](docs/sar_whiteboard_assistant_prompt.md)
  is the original spec: the *Acanthamoeba* / Laguna de Bustillos scenario, plus the module layout
  (`/biobench_sar`), RAG ingestion formats and frontend rules that both profiles share.

Where this file and a spec conflict on the project itself, the spec wins. How the specs map onto modules,
every cross-module contract, and the deliberate departures from the specs ("Implementation notes"):
[docs/BUILD_PLAN.md](docs/BUILD_PLAN.md). Running it on the real kit:
[docs/HARDWARE_DAY.md](docs/HARDWARE_DAY.md).

Non-negotiables:
- **Math is spoken in words** — never `^ / * =` in `spoken_text` ("two thirds", "x squared", "three x plus
  five equals twelve"); `llm/mathspeak.py` is the safety net, not an excuse to skip it in the prompt.
- **Help mode never gives the final answer** — only the next operation, pointing at that part of the ink
  (which the projector highlights). The prompt says so and code guards it: re-ask once, then cut the sentence.
- **Quiz answers are graded in code** (`biobench_sar/grading.py`, sympy). The model only copies the student's
  answer into `board_answer`; its own verdict never awards a point.
- **Bilingual** — every spoken or shown text has an English and a Mexican Spanish version (`lang.py`,
  `llm/mathspeak_es.py`, `static/js/i18n.js`).
- **The camera reads only the student's ink** — projected UI is masked out (`ui_rects`), and targets are snapped
  to the ink in code.
- **No monoliths** — build one module at a time in the `/biobench_sar` layout; wait for the user to name it.
- **Concurrency first** — the camera loop stays at ~30 FPS (hardware check passes at ≥ 28) whatever the LLM is doing.
- **Setup commands first** — give Windows PowerShell install commands (and the macOS equivalents) for pip and
  system binaries like Tesseract and PortAudio before Python code.
- **Defensive JSON parsing** — `try/except` around VLM output; strip markdown fences before `json.loads`.
- VLM output schema: `{spoken_text, animation_state: idle|walking|pointing|celebrating|error, target_x, target_y}`.
  Quiz and practice prompts add one internal helper key (`board_answer` / `board_problem`) that code removes.
- **Phones only behind the PIN** — any non-loopback bind turns the LAN PIN gate on (`lan.py`,
  [docs/MOBILE.md](docs/MOBILE.md)); never add a route or socket handler that skips it.

## Stack
- Backend: Python 3.10–3.12 (openai-whisper/torch lag the newest Python), Flask, Flask-SocketIO
- Vision: opencv-python, numpy (`cv2.findHomography` / `cv2.warpPerspective`)
- Audio/TTS: pyaudio (raw capture; no SpeechRecognition), openai-whisper (local, en/es), edge-tts / `say` /
  pyttsx3 (en-US and es-MX voices)
- RAG: chromadb (persistent, local), sentence-transformers `all-MiniLM-L6-v2`
- Parsers: PyMuPDF, pandas, python-docx, python-pptx, pytesseract
- VLM: Ollama `qwen2.5vl:3b` by default (`SAR_VLM_BACKEND=openai|mock` for others)
- Grading: sympy (`grading.py`)
- Desktop builds: PyInstaller + Inno Setup ([docs/DESKTOP_BUILDS.md](docs/DESKTOP_BUILDS.md))
- Frontend: vanilla HTML/CSS/JS; the projector page (`templates/index.html`) has a pure `#000000` background
  (black = invisible on the board); the dashboard is a normal light page
- Tests: pytest; lint/format: ruff

## Commands
```bash
bash scripts/setup_mac.sh --pull-model                      # first time on macOS (Windows: scripts/setup_windows.ps1)
.venv/bin/python -m pip install -r requirements-dev.txt     # dev deps (pytest, ruff)
.venv/bin/python -m biobench_sar                            # run (SAR_PROFILE=biology for the biology profile)
SAR_VLM_BACKEND=mock SAR_TTS_BACKEND=none .venv/bin/python -m biobench_sar   # no model needed
.venv/bin/python -m pytest -q
.venv/bin/ruff check . && .venv/bin/ruff format --check .  # `ruff format .` fixes formatting
.venv/bin/python scripts/hardware_check.py
.venv/bin/python -m scripts.bench_vlm -n 3
.venv/bin/python -m scripts.bench_vlm --scenarios --profile math   # pass/fail cases (--profile biology is the default)
.venv/bin/python packaging/launcher.py --self-test                  # what CI runs on every desktop build
cp sar.env.example sar.env                                          # room settings for the classroom kit
```

Windows PowerShell (after `scripts/setup_windows.ps1`, which prints the first and the mock line too; a `$env:`
setting lasts until the window closes):
```powershell
.\.venv\Scripts\python.exe -m biobench_sar
$env:SAR_PROFILE='biology'; .\.venv\Scripts\python.exe -m biobench_sar
$env:SAR_VLM_BACKEND='mock'; $env:SAR_TTS_BACKEND='none'; .\.venv\Scripts\python.exe -m biobench_sar
.\.venv\Scripts\python.exe -m pytest -q
```

Both profiles share `SAR_DATA_DIR` (knowledge base, ticked files, quiz bank); give each its own
(`SAR_DATA_DIR=data/biology`) to keep them apart. Deleting `data/chroma_db` loads the bundled documents again.
Every `SAR_*` setting and its default: the [README's configuration table](README.md#configuration).

## Code rules
- Be terse. No preamble, no summaries unless asked. Output code only.
- Python: type hints on public functions, small modules matching the spec's layout.
- No blocking calls on the camera/WebSocket threads.
- Every behaviour change ships with a test.
- Show only changed sections + minimal context when editing.
- **Cleanup pass on every branch and PR** — before opening a PR, run the checklist in
  [CONTRIBUTING.md](CONTRIBUTING.md#cleanup-pass-every-branch-and-pr) over the files the branch touches; list what
  is left in the PR's **Cleanup** section.

## Token economy (hackathon — maximize the budget)
- **Draft bulk/boilerplate with the local model**, then review: use `draft-with-ollama`
  (CRUD, scaffolding, config, test stubs, first-draft components). Free, zero Claude tokens.
  `qwen2.5-coder:7b` is only for that drafting; the app itself runs `qwen2.5vl:3b`.
- **Model tiers:** Opus only for architecture + hard bugs. Use `/model sonnet` for
  feature work, `/model haiku` for mechanical edits. Cheap subagents for grunt work.
- **Plan before building:** `superpowers:brainstorming` → `superpowers:writing-plans`.
  The biggest token sink is explore-wrong-way-then-redo; a plan prevents it.
- **Prime once, don't re-read:** run `/learn-codebase` once per repo (claude-mem keeps it).
  Keep durable facts (commands, architecture, conventions) in THIS file so they aren't rediscovered.
- **Keep context small:** `/compact` long sessions; start fresh per feature. Use the
  `Explore` agent for codebase sweeps (reads excerpts, not whole files).
- **Verify before 'done':** `superpowers:verification-before-completion` — one pass beats a redo loop.
- **If running Headroom:** launch Claude through the proxy to compress tool output
  (`ANTHROPIC_BASE_URL=http://localhost:8787 claude`).

## Skill routing (auto-apply — don't ask)
| Trigger | Skill |
| --- | --- |
| New feature / component | brainstorming → writing-plans |
| UI / component / styling | frontend-design |
| Charts / dashboards | dataviz |
| Forms + validation | form-builder |
| Auth | auth-flow |
| DB schema / migrations | schema-design / db-migrate |
| API endpoint | api-route |
| Bug / error / test failure | superpowers:systematic-debugging |
| Tests | test-gen |
| Bulk / boilerplate code | draft-with-ollama |

### Gates
- Before done/commit → verification-before-completion.
- Before opening a PR → the cleanup pass (CONTRIBUTING.md).
- Before merge/demo → code-review then security-review.
- Before deploy → deployment-checklist.
