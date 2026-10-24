# Hackaton — SAR Whiteboard Assistant

A Spatial Augmented Reality whiteboard assistant: a webcam and a projector point at a whiteboard, and an
animated 2D avatar reads your marker drawings, answers by voice, and walks over to point at what it is
talking about. Answers are grounded in a local knowledge base (RAG) of your study material.

Built for the Environmental Biology scenario: verifying *Acanthamoeba* isolation flowcharts and sterile
procedures for Laguna de Bustillos (Leptosol / Feozem runoff, Cuauhtémoc, Chihuahua).

**Spec:** [docs/sar_whiteboard_assistant_prompt.md](docs/sar_whiteboard_assistant_prompt.md) ·
**Plan, contracts and status:** [docs/BUILD_PLAN.md](docs/BUILD_PLAN.md)

## Quick start (macOS)

```bash
bash scripts/setup_mac.sh --pull-model   # brew deps, Python 3.12 venv, requirements, qwen2.5vl:3b (~3.2 GB)
.venv/bin/python -m biobench_sar
```

Windows: run `scripts/setup_windows.ps1` in PowerShell.

Then open:

| Page | URL | Where |
| --- | --- | --- |
| Projector | `http://localhost:5050/` | Fullscreen (`F`) on the projector. Pure black = invisible on the board. |
| Operator dashboard | `http://localhost:5050/dashboard` | Your laptop. |

The bundled scenario documents are loaded into the knowledge base on first run (`SAR_SEED_KNOWLEDGE=0`
to skip). The default port is 5050 because macOS AirPlay uses 5000. Grant **Camera** and **Microphone** access
to the terminal that runs the server (System Settings → Privacy & Security), then restart it.

Check the hardware any time with `python scripts/hardware_check.py`.

## How it works

```
 mic ──► Whisper ─┐                                     ┌─► TTS ──► speaker / projector audio
                  ├─► Assistant ─► RAG (ChromaDB) ─► VLM ┤
 camera ─► homography ─► ink regions ─┘                 └─► Socket.IO ─► avatar walks + points
```

1. **Sense** — push-to-talk (Space / *Listen*) or wake word records a question; the camera frame is warped
   into projector coordinates and marker ink is segmented into regions (one per flowchart box).
2. **Think** — the question retrieves the top chunks from your documents; the question, board image, ink
   regions and context go to the VLM, which must answer with strict JSON
   `{spoken_text, animation_state, target_x, target_y}`.
3. **Act** — the answer is spoken and the avatar glides to the target with the matching state
   (`idle`, `walking`, `pointing`, `celebrating`, `error`).

## The dashboard

Open `http://localhost:5050/dashboard` on your laptop. It is laid out like a lab notebook, in three steps:

1. **Choose your study files**: drag in PDFs, Word or PowerPoint files, CSVs, notes or photos of notes.
   Tick the files the assistant should use. Unticked files are kept but ignored when answering.
2. **Set up the board**: see what the camera sees, check the list (each problem says how to fix it), and
   press **Line up camera and projector** once after moving either one.
3. **Ask a question**: type it, press **Talk**, or click an example. Every answer is written into the
   notebook with a note saying what the assistant did, which files it read and how long it took.

**Behind the scenes** (collapsed) holds the raw camera view, detected drawings, file search, settings and
the activity log.

## Configuration

Everything is an environment variable (see `biobench_sar/config.py`). The ones you are most likely to change:

| Variable | Default | Meaning |
| --- | --- | --- |
| `SAR_PORT` | `5050` | Server port |
| `SAR_SEED_KNOWLEDGE` | `1` | Load `biobench_sar/knowledge/` into an empty knowledge base at startup |
| `SAR_CAMERA_INDEX` | `0` | Which camera |
| `SAR_PROJECTOR_WIDTH` / `_HEIGHT` | `1920` / `1080` | Projector resolution |
| `SAR_VLM_BACKEND` | `ollama` | `ollama`, `openai` (any compatible API) or `mock` (no model) |
| `SAR_VLM_MODEL` | `qwen2.5vl:3b` | Model name |
| `SAR_VLM_BASE_URL` / `SAR_VLM_API_KEY` | `http://localhost:11434` / – | Endpoint and key for the backend |
| `SAR_LISTEN_MODE` | `push` | `push` (Space / button) or `wake` (always listening for the wake word) |
| `SAR_WAKE_WORD` | `assistant` | Wake word |
| `SAR_TTS_BACKEND` | `edge` | `edge` (online), `say` (macOS), `pyttsx3` (offline) or `none` |
| `SAR_INK_MIN_AREA` / `_BLOCK_SIZE` / `_C` | `150` / `31` / `15` | Ink detection thresholds |
| `SAR_RECORD_SECONDS` | `5` | How long push-to-talk records |
| `SAR_SILENCE_RMS` | `0.01` | Wake-word chunks quieter than this are skipped (push-to-talk always records) |

## Development

```bash
uv pip install --python .venv/bin/python -r requirements-dev.txt
.venv/bin/python -m pytest -q        # unit tests, no hardware or network needed
.venv/bin/ruff check . && .venv/bin/ruff format --check .
SAR_VLM_BACKEND=mock SAR_TTS_BACKEND=none .venv/bin/python -m biobench_sar   # UI without a model
```

CI runs lint and tests on every pull request. Every change goes through its own branch and PR — see
[CONTRIBUTING.md](CONTRIBUTING.md). AI agents start at [AGENTS.md](AGENTS.md).

## Repository map

| Path | What |
| --- | --- |
| `biobench_sar/app.py` | Flask + Socket.IO routes |
| `biobench_sar/assistant.py` | Sense → Think → Act pipeline, status, history |
| `biobench_sar/core/` | camera loop, ink detection, homography calibration, mic / wake word |
| `biobench_sar/rag/` | document ingestion and ChromaDB store |
| `biobench_sar/llm/` | VLM client (prompt + JSON schema) and text-to-speech |
| `biobench_sar/templates`, `static/` | projector page and operator dashboard |
| `biobench_sar/knowledge/` | seed documents for the scenario |
| `scripts/` | setup, hardware check, VLM benchmark, ink threshold sweep |
| `tests/` | pytest suite |
