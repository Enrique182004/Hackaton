# CLAUDE.md — SAR Whiteboard Assistant (hackathon)

## Project focus (read first)
This repo builds ONE thing: the **SAR Whiteboard Assistant** — webcam + projector + whiteboard,
with an animated 2D AI avatar that sees marker drawings and reacts (Sense → Think → Act),
grounded by a local RAG knowledge base.

**The full spec is [docs/sar_whiteboard_assistant_prompt.md](docs/sar_whiteboard_assistant_prompt.md).
Read it before any work — it is the source of truth** for directory layout (`/biobench_sar`),
stack, homography calibration, RAG ingestion, the VLM JSON schema, the frontend and the
Environmental Biology (*Acanthamoeba* / Laguna de Bustillos) scenario. Where this file and the
spec conflict on the project itself, the spec wins.

Non-negotiables from the spec:
- **No monoliths** — build one module at a time (e.g. `core/vision_loop.py`); wait for the user to name it.
- **Concurrency first** — camera loop stays at 30+ FPS no matter what the LLM is doing.
- **Setup commands first** — give Windows PowerShell install commands (pip + system binaries like Tesseract) before Python code.
- **Defensive JSON parsing** — `try/except` around VLM output; strip markdown fences before `json.loads`.
- VLM output schema: `{spoken_text, animation_state: idle|walking|pointing|celebrating|error, target_x, target_y}`.

## Stack
- Backend: Python 3.10+, Flask, Flask-SocketIO
- Vision: opencv-python, numpy (`cv2.findHomography` / `cv2.warpPerspective`)
- Audio/TTS: SpeechRecognition, pyaudio, openai-whisper (local), edge-tts / pyttsx3
- RAG: chromadb (persistent, local), sentence-transformers `all-MiniLM-L6-v2`
- Parsers: PyMuPDF, pandas, python-docx, python-pptx, pytesseract
- Frontend: vanilla HTML/CSS/JS, pure `#000000` background
- Tests: pytest

## Commands
```bash
python -m venv .venv
pip install -r requirements.txt
python app.py
pytest
```

## Code rules
- Be terse. No preamble, no summaries unless asked. Output code only.
- Python: type hints on public functions, small modules matching the spec's layout.
- No blocking calls on the camera/WebSocket threads.
- Show only changed sections + minimal context when editing.

## Token economy (hackathon — maximize the budget)
- **Draft bulk/boilerplate with the local model**, then review: use `draft-with-ollama`
  (CRUD, scaffolding, config, test stubs, first-draft components). Free, zero Claude tokens.
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
