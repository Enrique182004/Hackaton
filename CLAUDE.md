# CLAUDE.md — SAR Whiteboard Math Assistant (hackathon)

## Project focus (read first)
This repo builds ONE thing: the **SAR Whiteboard Assistant** — webcam + projector + whiteboard, with an
animated 2D AI avatar that sees marker work and reacts (Sense → Think → Act), grounded by a local RAG
knowledge base. Its subject is chosen by `SAR_PROFILE`:

- **`math` (default) — the math tutor.** [docs/sar_math_tutor_prompt.md](docs/sar_math_tutor_prompt.md) is
  the **source of truth for the subject**: tutoring behaviour, the three modes (step-by-step help, similar
  problems, quiz), phonetic TTS formatting and the JSON schema. Read it before any work.
- **`biology` — kept as the second profile.** [docs/sar_whiteboard_assistant_prompt.md](docs/sar_whiteboard_assistant_prompt.md)
  is the original spec: the *Acanthamoeba* / Laguna de Bustillos scenario, plus the module layout
  (`/biobench_sar`), RAG ingestion formats and frontend rules that both profiles share.

Where this file and a spec conflict on the project itself, the spec wins. How the specs map onto modules,
and every cross-module contract: [docs/BUILD_PLAN.md](docs/BUILD_PLAN.md).

Non-negotiables:
- **Math is spoken in words** — never `^ / * =` in `spoken_text` ("two thirds", "x squared", "three x plus
  five equals twelve"); `llm/mathspeak.py` is the safety net, not an excuse to skip it in the prompt.
- **Help mode never gives the final answer** — only the next operation, pointing at that part of the ink.
- **No monoliths** — build one module at a time in the `/biobench_sar` layout.
- **Concurrency first** — the camera loop stays at ~30 FPS (hardware check passes at ≥ 28) whatever the LLM is doing.
- **Setup commands first** — give install commands (pip + system binaries like Tesseract, PortAudio) before Python code.
- **Defensive JSON parsing** — `try/except` around VLM output; strip markdown fences before `json.loads`.
- VLM output schema: `{spoken_text, animation_state: idle|walking|pointing|celebrating|error, target_x, target_y}`.

## Stack
- Backend: Python 3.10–3.12 (openai-whisper/torch lag the newest Python), Flask, Flask-SocketIO
- Vision: opencv-python, numpy (`cv2.findHomography` / `cv2.warpPerspective`)
- Audio/TTS: pyaudio, openai-whisper (local), edge-tts / `say` / pyttsx3
- RAG: chromadb (persistent, local), sentence-transformers `all-MiniLM-L6-v2`
- Parsers: PyMuPDF, pandas, python-docx, python-pptx, pytesseract
- VLM: Ollama `qwen2.5vl:3b` by default (`SAR_VLM_BACKEND=openai|mock` for others)
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
```

## Code rules
- Be terse. No preamble, no summaries unless asked. Output code only.
- Python: type hints on public functions, small modules matching the spec's layout.
- No blocking calls on the camera/WebSocket threads.
- Every behaviour change ships with a test.
- Show only changed sections + minimal context when editing.

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
- Before merge/demo → code-review then security-review.
- Before deploy → deployment-checklist.
