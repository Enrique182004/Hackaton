# Hackaton — SAR Whiteboard Math Assistant

A Spatial Augmented Reality whiteboard tutor: a webcam and a projector point at a whiteboard, and an animated
2D avatar reads your marker work, talks you through it and walks over to point at the part it is talking
about. It tutors middle and high school math (fractions, equations, algebra), never gives away the final
answer, and can quiz you from your own worksheets through a local knowledge base (RAG).

**Subject profiles** (`SAR_PROFILE`):

| Profile | Spec | What the avatar does |
| --- | --- | --- |
| `math` (default) | [docs/sar_math_tutor_prompt.md](docs/sar_math_tutor_prompt.md) | Math tutor: next-step hints, similar problems, quizzes, learning a new topic |
| `biology` | [docs/sar_whiteboard_assistant_prompt.md](docs/sar_whiteboard_assistant_prompt.md) | Research assistant that checks *Acanthamoeba* isolation flowcharts for Laguna de Bustillos |

**Plan, contracts and status:** [docs/BUILD_PLAN.md](docs/BUILD_PLAN.md)

## Quick start (macOS)

```bash
bash scripts/setup_mac.sh --pull-model   # brew deps, Python 3.12 venv, requirements, qwen2.5vl:3b (~3.2 GB)
.venv/bin/python -m biobench_sar                        # math tutor (default)
SAR_PROFILE=biology .venv/bin/python -m biobench_sar    # biology research assistant
```

Windows: run `scripts/setup_windows.ps1` in PowerShell.

Then open:

| Page | URL | Where |
| --- | --- | --- |
| Projector | `http://localhost:5050/` | Fullscreen (`F`) on the projector. Pure black = invisible on the board. |
| Operator dashboard | `http://localhost:5050/dashboard` | Your laptop. |

The profile's bundled documents (`biobench_sar/knowledge/<profile>/`) are loaded into the knowledge base on
first run (`SAR_SEED_KNOWLEDGE=0` to skip). For math that is a quiz bank, a worksheet and a tutoring guide.
The default port is 5050 because macOS AirPlay uses 5000. Grant **Camera** and **Microphone** access to the
terminal that runs the server (System Settings → Privacy & Security), then restart it.

Check the hardware any time with `.venv/bin/python scripts/hardware_check.py`.

## How it works

```
 mic ──► Whisper ─┐                                     ┌─► TTS ──► speaker / projector audio
                  ├─► Assistant ─► RAG (ChromaDB) ─► VLM ┤
 camera ─► homography ─► ink regions ─┘                 └─► Socket.IO ─► avatar walks + points
```

1. **Sense** — push-to-talk (**Space** on the projector page / **Talk** on the dashboard) or the wake word
   records a question; the camera frame is warped into projector coordinates and marker ink is segmented
   into regions (one per equation line or flowchart box).
2. **Think** — the question retrieves the top chunks from your ticked files; the question, board image, ink
   regions, context and the current tutor mode go to the VLM, which must answer with strict JSON
   `{spoken_text, animation_state, target_x, target_y}`.
3. **Act** — the answer is spoken and the avatar glides to the target with the matching state
   (`idle`, `walking`, `pointing`, `celebrating`, `error`).

## The math tutor

Math is always **spoken in words**: the prompt tells the model to write "two thirds", "x squared" and "three
x plus five equals twelve", and `biobench_sar/llm/mathspeak.py` rewrites any `2/3`, `x^2`, `3x+5=12`, `-4`,
`√9`, `25%` that slips through, so the voice never reads out `^ / * =`.

The tutor has three modes. Say one of the phrases (or type it on the dashboard) to switch:

| Mode | Say | What the avatar does |
| --- | --- | --- |
| **Help** (default) | "help", "next step", "I'm stuck", "stop the quiz" | Explains **only the next operation** in simple words, points (`pointing`) at that part of your equation and waits for you to write the step. It never says the final answer. |
| **Practice** | "similar problem", "another one like this", "more practice" | Reads the equation on the board, swaps the numbers or letters to keep the same difficulty and dictates the new equation for you to write (`idle`). |
| **Quiz** | "quiz me", "start a quiz" | Dictates problem 1 of N (also shown top-left on the projector in normal notation, e.g. `3x + 5 = 20`). Write your answer and talk again ("check my answer"): `celebrating` and the next problem if it is right, `error` and one hint if it is wrong — never the answer. |

**Learning a topic** works in any mode, even with an empty board. Say what you want to learn ("I want to
learn how to solve fractions", "teach me equations", "how do I add fractions?"): the avatar introduces the
idea in a sentence or two and asks *"Do you have an exercise in mind, or should I make one up for you?"*

- "make one up", "give me one", "you choose" — it dictates an easy starter problem; write it and try the
  first step, then ask for help as usual.
- "I have one", "I'll write mine" — it asks you to write your exercise on the board and then ask for help.

The assistant remembers the last three questions and answers, so follow-ups like "I did it" or "what now?"
make sense.

The same controls exist over HTTP: `GET /api/tutor`, `PUT /api/tutor/mode {"mode": "help"|"practice"|"quiz"}`,
`POST /api/tutor/quiz/start`, `POST /api/tutor/quiz/stop`. Every change of mode, quiz or topic is pushed to
the pages as a `tutor` Socket.IO event.

### Quiz banks

Quiz problems come **only from ticked files** on the dashboard's *Study files* tab, in order of level, then
file order. Two formats are recognised:

- **CSV** with a `problem` (or `question`) and an `answer` column, plus optional `topic` and `level` (or
  `difficulty`) columns — see [`knowledge/math/quiz_bank.csv`](biobench_sar/knowledge/math/quiz_bank.csv):

  ```csv
  problem,answer,topic,level
  3x + 5 = 20,x = 5,two-step equations,3
  ```

- **Worksheets** (`.md`, `.txt`, `.pdf`, `.docx`) with numbered problems (`1. 3x + 5 = 20` or
  `Problem 2: ...`), each optionally followed by an `Answer: ...` line — see
  [`knowledge/math/worksheet_linear_equations.md`](biobench_sar/knowledge/math/worksheet_linear_equations.md).

The parsed problems are kept in `SAR_QUIZ_BANK_FILE`. With no problems ticked, a quiz still starts: the model
makes up ten progressive problems itself.

## The dashboard

Open `http://localhost:5050/dashboard` on your laptop. It has four tabs:

1. **Study files** — drag in PDFs, Word or PowerPoint files, CSVs, notes or photos of notes (quiz banks and
   worksheets too). Tick the files the assistant should use; unticked files are kept but ignored.
2. **Board** — see what the camera sees, check the list (each problem says how to fix it), and press
   **Line up camera and projector** once after moving either one.
3. **Ask** — type a question, press **Talk**, or click an example. Every answer is written into the notebook
   with a note saying what the assistant did, which files it read and how long it took.
4. **More tools** — the lined-up or raw camera view, detected drawings, file search, settings and the activity
   log.

## Configuration

Everything is an environment variable (see `biobench_sar/config.py`). The ones you are most likely to change:

| Variable | Default | Meaning |
| --- | --- | --- |
| `SAR_PROFILE` | `math` | Subject: `math` (tutor) or `biology` (research assistant); anything else means `math` |
| `SAR_PORT` / `SAR_HOST` | `5050` / `127.0.0.1` | Server port and bind address |
| `SAR_DATA_DIR` | `data/` | Where calibration, the knowledge base, uploads, speech and selections are kept |
| `SAR_SEED_KNOWLEDGE` | `1` | Load `biobench_sar/knowledge/<profile>/` into an empty knowledge base at startup |
| `SAR_QUIZ_BANK_FILE` | `data/quiz_bank.json` | Quiz problems parsed from uploaded CSVs and worksheets |
| `SAR_SELECTION_FILE` | `data/selection.json` | Which files are ticked |
| `SAR_CAMERA_INDEX` | `0` | Which camera |
| `SAR_PROJECTOR_WIDTH` / `_HEIGHT` | `1920` / `1080` | Projector resolution |
| `SAR_VLM_BACKEND` | `ollama` | `ollama`, `openai` (any compatible API) or `mock` (no model) |
| `SAR_VLM_MODEL` | `qwen2.5vl:3b` | Model name |
| `SAR_VLM_BASE_URL` / `SAR_VLM_API_KEY` | `http://localhost:11434` / – | Endpoint and key for the backend |
| `SAR_VLM_TIMEOUT` | `90` | Seconds to wait for the model |
| `SAR_RAG_TOP_K` | `4` | Chunks retrieved per question |
| `SAR_LISTEN_MODE` | `push` | `push` (Space / Talk) or `wake` (always listening for the wake word) |
| `SAR_WAKE_WORD` | `assistant` | Wake word |
| `SAR_WHISPER_MODEL` | `base` | Local Whisper model size |
| `SAR_RECORD_SECONDS` | `5` | How long push-to-talk records |
| `SAR_SILENCE_RMS` | `0.01` | Wake-word chunks quieter than this are skipped (push-to-talk always records) |
| `SAR_TTS_BACKEND` | `edge` | `edge` (online), `say` (macOS), `pyttsx3` (offline) or `none` |
| `SAR_TTS_VOICE` | `en-US-AriaNeural` | Voice for `edge` |
| `SAR_INK_MIN_AREA` / `_BLOCK_SIZE` / `_C` | `150` / `31` / `15` | Ink detection thresholds |

## Development

```bash
.venv/bin/python -m pip install -r requirements-dev.txt   # or: uv pip install --python .venv/bin/python -r requirements-dev.txt
.venv/bin/python -m pytest -q        # unit tests, no hardware or network needed
.venv/bin/ruff check . && .venv/bin/ruff format --check .
SAR_VLM_BACKEND=mock SAR_TTS_BACKEND=none .venv/bin/python -m biobench_sar   # UI without a model
.venv/bin/python -m scripts.bench_vlm -n 3   # VLM latency against Ollama (--scenarios: biology flowchart cases)
```

CI runs lint and tests on every pull request. Every change goes through its own branch and PR — see
[CONTRIBUTING.md](CONTRIBUTING.md). AI agents start at [AGENTS.md](AGENTS.md).

## Repository map

| Path | What |
| --- | --- |
| `biobench_sar/app.py` | Flask + Socket.IO routes |
| `biobench_sar/assistant.py` | Sense → Think → Act pipeline, status, history |
| `biobench_sar/tutor.py` | Tutor modes (help / practice / quiz), learn-a-topic requests, voice-command detection, quiz progress, conversation memory |
| `biobench_sar/core/` | camera loop, ink detection, homography calibration, mic / wake word |
| `biobench_sar/rag/` | document ingestion, ChromaDB store, quiz bank parser |
| `biobench_sar/llm/` | VLM client (prompts per profile and mode + JSON schema), math-to-speech, text-to-speech |
| `biobench_sar/templates`, `static/` | projector page and operator dashboard |
| `biobench_sar/knowledge/math/`, `knowledge/biology/` | seed documents for each profile |
| `scripts/` | setup, hardware check, VLM benchmark, ink threshold sweep |
| `tests/` | pytest suite |
