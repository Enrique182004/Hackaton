# Hackaton — SAR Whiteboard Math Assistant

A Spatial Augmented Reality whiteboard tutor. A webcam and a projector point at a whiteboard, and **Pip**, an
animated 2D avatar projected onto the board, reads your marker work, talks you through it and walks over to
point at the part it is talking about. It tutors middle and high school math (fractions, equations, algebra)
in **English or Spanish**, never gives away the final answer, and can quiz you from your own worksheets
through a local knowledge base (RAG). Everything runs on one laptop: the vision model (Ollama), speech
recognition (Whisper) and the knowledge base (ChromaDB) are all local.

**Subject profiles** (`SAR_PROFILE`):

| Profile | Spec | What the avatar does |
| --- | --- | --- |
| `math` (default) | [docs/sar_math_tutor_prompt.md](docs/sar_math_tutor_prompt.md) | Math tutor: next-step hints, similar problems, quizzes, learning a new topic |
| `biology` | [docs/sar_whiteboard_assistant_prompt.md](docs/sar_whiteboard_assistant_prompt.md) | Research assistant that checks *Acanthamoeba* isolation flowcharts for Laguna de Bustillos |

| You want to… | Read |
| --- | --- |
| Run it in a classroom, step by step | [docs/HARDWARE_DAY.md](docs/HARDWARE_DAY.md) (en español: [docs/HARDWARE_DAY.es.md](docs/HARDWARE_DAY.es.md)) |
| Place the projector, webcam and lights | [docs/HARDWARE_SETUP.md](docs/HARDWARE_SETUP.md) |
| Install it without Python (Windows installer, macOS app) | [docs/DESKTOP_BUILDS.md](docs/DESKTOP_BUILDS.md) |
| Drive it from phones or tablets | [docs/MOBILE.md](docs/MOBILE.md) |
| Work on the code: contracts, status, where the code differs from the specs | [docs/BUILD_PLAN.md](docs/BUILD_PLAN.md) |

## Quick start (macOS)

```bash
bash scripts/setup_mac.sh --pull-model   # brew deps, Python 3.12 venv, requirements, qwen2.5vl:3b (~3.2 GB)
cp sar.env.example sar.env               # room settings, already filled in for the classroom kit
.venv/bin/python -m biobench_sar                        # math tutor (default)
SAR_PROFILE=biology .venv/bin/python -m biobench_sar    # biology research assistant
```

Windows: run `scripts/setup_windows.ps1` in PowerShell, then:

```powershell
Copy-Item sar.env.example sar.env                                               # room settings
.\.venv\Scripts\python.exe -m biobench_sar                                      # math tutor (default)
$env:SAR_PROFILE='biology'; .\.venv\Scripts\python.exe -m biobench_sar          # biology research assistant
$env:SAR_VLM_BACKEND='mock'; .\.venv\Scripts\python.exe -m biobench_sar         # no model needed
```

A `$env:` setting lasts until that PowerShell window closes (`Remove-Item Env:SAR_PROFILE` undoes it).

`sar.env` is optional, but without it the code defaults apply: camera index 0 at 1280×720 and a 1280×800
projector. The classroom kit's webcam needs `SAR_CAMERA_WIDTH=1920`, `SAR_CAMERA_HEIGHT=1080` and usually
`SAR_CAMERA_INDEX=1`, and `sar.env.example` sets those.

Then open:

| Page | URL | Where |
| --- | --- | --- |
| Projector | `http://localhost:5050/` | Fullscreen (`F`) on the projector. Pure black = invisible on the board. |
| Operator dashboard | `http://localhost:5050/dashboard` | Your laptop. |

The profile's bundled documents (`biobench_sar/knowledge/<profile>/`) are loaded into the knowledge base the
first time that profile starts (`SAR_SEED_KNOWLEDGE=0` skips this). For math, that means a quiz bank, a worksheet
and a tutoring guide. Each profile is loaded once per knowledge base (recorded in `data/chroma_db/.seeded`), so
a bundled file you remove stays removed.

**Both profiles share one knowledge base**, file selection and quiz bank in `SAR_DATA_DIR`. After you run
both, math answers also search the biology files, and the other way round, unless you untick them. To keep the
subjects apart, give each its own data folder:

```bash
SAR_PROFILE=biology SAR_DATA_DIR=data/biology .venv/bin/python -m biobench_sar
```

To load the bundled documents again, stop the app and delete `data/chroma_db`. Your uploads also leave the
knowledge base, so upload them again.

The default port is 5050 because macOS AirPlay uses 5000. Grant **Camera** and **Microphone** access to the
terminal that runs the server (System Settings → Privacy & Security), then restart it.

Check the hardware any time with `.venv/bin/python scripts/hardware_check.py`.

## How it works

```
 mic ──► Whisper (en/es) ───────┐                                            ┌─► TTS ──► speaker
                                ├─► Assistant ─► RAG ─► VLM ─► code checks ──┤
 camera ─► homography ─► ink ───┘    (ChromaDB)                              └─► Socket.IO ─► Pip + highlight
```

1. **Sense.** Push-to-talk (**Space** on the projector page or **Talk** on the dashboard) or the wake word
   records a question, and Whisper writes it down in English or Spanish. The camera frame is warped into
   projector coordinates and the marker ink is split into regions, one per equation line or flowchart box.
   Ink regions mostly covered by the projector's own picture (Pip, the speech bubble, the quiz card, the
   highlight) are dropped, so the camera reads only the student's writing. A high-resolution crop of the ink
   goes to the model next to the board picture, so decimal points and minus signs survive.
2. **Think.** The question retrieves the top chunks from your ticked files. In math, answer keys are removed
   from the files when they are uploaded, and again from every chunk. The question, board picture, ink crop,
   ink regions, context, last three turns and the current tutor task go to the VLM. It must answer with strict
   JSON `{spoken_text, animation_state, target_x, target_y}`. In quiz and practice it adds one helper key that
   code reads and removes: `board_answer` (the student's answer, copied from the board) or `board_problem`.
3. **Check (in code).** Quiz answers are graded by `biobench_sar/grading.py` (sympy), never by the model. In
   help mode, a reply that states the solution is sent back to the model once, and then the offending sentence
   is cut. Targets are snapped to the ink. `pointing` and `error` go to the ink region the answer is about
   (`pointing` lands just left of it), and `idle` and `walking` go to a free spot away from the writing. Spoken
   math is rewritten into words.
4. **Act.** The answer is spoken in the student's language. Pip glides to the target in the matching state
   (`idle`, `walking`, `pointing`, `celebrating`, `error`), and a glowing outline pulses around the ink the
   answer is about.

## The avatar

Pip is a holographic study buddy drawn as a CSS sprite (`biobench_sar/static/assets/sprites/avatar.svg`, one
frame per state) on a pure black page, so only Pip shows on the board.

| State | When |
| --- | --- |
| `pointing` | Help, or a hint: the pencil tip lands just left of the ink it is talking about |
| `error` | A mistake or a wrong quiz answer: amber, next to the wrong part |
| `celebrating` | A right quiz answer, or a flowchart that is complete |
| `idle` / `walking` | Talking, dictating or waiting: parked where it covers no writing |

When an answer is about part of your writing (help, a hint, a check or a mistake), the projector draws a glowing
outline around that ink. It pulses for about 2 s and fades after about 8 s, or when the next answer arrives.
The speech bubble shows the answer in normal notation (`3x + 5 = 20`) while the voice says it in words.

Projector page keys: **Space** = push-to-talk, **T** = type a question, **F** = full screen, **C** = debug
overlay. Click the page once so the browser lets it play sound.

## The math tutor

Math is always **spoken in words**. The prompt tells the model to write "two thirds", "x squared" and "three
x plus five equals twelve", and `biobench_sar/llm/mathspeak.py` (`llm/mathspeak_es.py` in Spanish: "dos
tercios", "x al cuadrado", "tres x más cinco es igual a doce") rewrites any `2/3`, `x^2`, `3x+5=12`, `-4`,
`√9` or `25%` that slips through, so the voice never reads out `^ / * =`.

The tutor has three modes. Say one of the phrases (or type it on the dashboard) to switch:

| Mode | Say | What the avatar does |
| --- | --- | --- |
| **Help** (default) | "help", "next step", "I'm stuck" | Explains **only the next operation** in simple words, points (`pointing`) at that part of your work, which glows, and waits for you to write the step. It never says the final answer. If the model's reply states it anyway ("x = 5", "x equals five"), the assistant asks again and cuts that sentence before speaking. |
| **Practice** | "give me a similar problem", "another one like this", "more practice" | Reads the equation on the board, swaps the numbers or letters to keep the same difficulty, and dictates the new equation for you to write (`idle`). If the model just repeats your problem, code makes the new one. |
| **Quiz** | "quiz me", "start a quiz" | Dictates problem 1 of N, also shown top-left on the projector in normal notation (e.g. `3x + 5 = 20`). Write your answer and talk again ("check my answer", "I'm done"). **Code grades what you wrote**: `celebrating` and the next problem if it is right, `error` and one hint if it is wrong, never the answer. |

During a quiz:

- **Answers it can grade:** `x = 5`, `5`, `5/2` or `2.5`, `x = 5 or x = -3`, `x = 5, y = 2`, `(2, 3)`, and Spanish
  words ("x es igual a cinco", "x = 5 y x = -3"). When it can't read your answer it says "I couldn't read your
  answer clearly. Ask me to check it again." That try doesn't count.
- "I'm stuck", "hint" or "help" gives one hint on the same problem without leaving the quiz or costing a try.
  "Repeat the problem" reads it out again.
- **Stopping asks first.** "Stop the quiz" gets *"Do you want to end the quiz? Say yes to stop, or keep
  going."* "Yes" (or "sure", "ok", "yeah", "sí", "claro", "dale") ends it with your score. Anything else carries
  on and is handled as usual, so an answer is graded. The dashboard's **Stop quiz** button stops at once.
- The score counts problems right on the first try.

**Learning a topic** works in any mode, even with an empty board. Say what you want to learn ("I want to
learn how to solve fractions", "teach me equations", "how do I add fractions?"). The avatar introduces the
idea in a sentence or two and asks *"Do you have an exercise in mind, or should I make one up for you?"*

- "make one up", "give me one", "you choose": it dictates an easy starter problem. Write it and try the
  first step, then ask for help as usual.
- "I have one", "I'll write mine": it asks you to write your exercise on the board and then ask for help.

The assistant remembers the last three questions and answers, so follow-ups like "I did it" or "what now?"
make sense.

The same controls exist over HTTP: `GET /api/tutor`, `PUT /api/tutor/mode {"mode": "help"|"practice"|"quiz"}`,
`POST /api/tutor/quiz/start`, `POST /api/tutor/quiz/stop`. Every change of mode, quiz, topic or pending stop
question is pushed to the pages as a `tutor` Socket.IO event, whose `confirm_stop` is `true` while the
assistant waits for the yes.

### Quiz banks

Quiz problems come **only from ticked files** on the dashboard's step 1 tab (*Worksheets & notes*), easiest
level first, then in file order. Two formats are recognised:

- **CSV** with a `problem` (or `question`) column and an `answer` column, plus optional `topic` and `level` (or
  `difficulty`) columns. See [`knowledge/math/quiz_bank.csv`](biobench_sar/knowledge/math/quiz_bank.csv):

  ```csv
  problem,answer,topic,level
  3x + 5 = 20,x = 5,two-step equations,3
  ```

- **Worksheets** (`.md`, `.txt`, `.pdf`, `.docx`) with numbered problems (`1. 3x + 5 = 20` or
  `Problem 2: ...`), each optionally followed by an `Answer: ...` line. See
  [`knowledge/math/worksheet_linear_equations.md`](biobench_sar/knowledge/math/worksheet_linear_equations.md).
  A file only counts as a worksheet when its **file name, first line or a heading says so**: worksheet, quiz,
  exercise, homework, assignment, practice, problem set, drill or test (e.g. `unit3_worksheet.md`, or a first
  line `# Quiz: fractions`). That keeps the numbered tips in a guide out of quizzes. A worksheet called
  `chapter3.md` with no such title gives no problems, so name or title it. A numbered line needs a number plus
  a math sign or word (`=`, `+`, `/`, "solve", "simplify", …).
  `## Level N: topic` headings set the level and topic of the problems below them, so problems from several
  files stay in order of difficulty. Without one, a problem counts as level 1:

  ```md
  # Worksheet: two-step equations

  ## Level 3: two-step equations

  1. 4x + 3 = 19
  Answer: x = 4
  ```

The parsed problems and answers are kept in `SAR_QUIZ_BANK_FILE`. The answers never go into the knowledge
base: in math, answer lines and columns are stripped from a file before it is stored (`Answer:`, `Respuesta:`,
`Solución`, `R:`, `Ans.`, `A1.`, "The answer is 8.", answer-key sections and so on). A quiz asks at most
`SAR_QUIZ_LENGTH` problems (default 10). When the ticked files have more than that, the quiz takes an even spread
from easiest to hardest. With no problems ticked, a quiz still starts: the code makes up and solves
`SAR_QUIZ_LENGTH` problems itself (one-step equations first, x on both sides last), so they are graded exactly
like worksheet problems.

## English and Spanish

- **The switch.** The dashboard's masthead has **English · Español · Auto**. It sets the page language and the
  answer language, and the projector page follows. **Auto** (the default, `SAR_LANGUAGE=auto`) answers in the
  language each question is asked in. English or Español fixes it, and Whisper then listens only for that
  language. Over HTTP: `GET /api/language` and `PUT /api/language {"language": "auto"|"en"|"es"}`.
- **Voice in.** Whisper decides between English and Spanish for each question. The wake word also answers to
  "asistente".
- **Voice out.** The online Edge voices are `en-US-AriaNeural` and `es-MX-DaliaNeural` (Mexican Spanish).
  Offline (`SAR_TTS_BACKEND=say` or `pyttsx3`, or when Edge can't reach the internet), the assistant picks an
  installed Spanish voice, Mexican first. If Spanish comes out with an English voice, install one: on macOS,
  System Settings → Accessibility → Spoken Content → System voice → Manage Voices (e.g. *Paulina*); on Windows,
  Settings → Time & language → Language & region → add *Español (México)* with text-to-speech.
- **Everything else.** Spanish requests ("Ayúdame con el siguiente paso", "Dame un problema parecido", "Hazme
  un quiz", "Quiero aprender a resolver fracciones", "Termina el quiz") are understood by
  `biobench_sar/lang.py`. Math is spoken in Spanish words, and quiz answers in Spanish words are graded.

## The dashboard

Open `http://localhost:5050/dashboard` on your laptop. It has four tabs:

1. **Worksheets & notes** (called *Study files* in the biology profile): drag in PDFs, Word or PowerPoint
   files, CSVs, notes or photos of notes (quiz banks and worksheets too). Tick the files the assistant should
   use. Unticked files are kept but ignored.
2. **Board**: see what the camera sees and go through the checklist (camera, glare, line-up, files, model,
   microphone, voice; each item says how to fix it). Press **Line up camera and projector** once after moving
   either one.
3. **Ask**: pick a mode, start or stop a quiz, type a question, press **Talk**, or click an example. Every
   answer is written into the notebook with a note saying what the assistant did, which files it read and how
   long it took.
4. **More tools**: the lined-up or raw camera view, detected drawings, file search, settings and the activity
   log.

## Phones and tablets

A phone on the same Wi-Fi can be a remote dashboard (type or dictate questions, change modes, run quizzes,
tick files) while the laptop does the work. Start with `SAR_LAN=1`. The terminal prints the phone address and
a six-digit PIN, and phones join with that PIN. Any bind other devices can reach (`SAR_LAN=1`, or a `SAR_HOST`
other than `127.0.0.1`, `localhost` or `::1`) turns the PIN gate on. Setup, limits and the security model:
[docs/MOBILE.md](docs/MOBILE.md).

## Desktop builds

CI builds a per-user Windows installer and a macOS (Apple Silicon) app that bundle Python, every package and the
embedding model, so teachers need no terminal. Ollama and the model are still installed once per computer.
Downloads, first run, where `sar.env` and the data go, and building locally:
[docs/DESKTOP_BUILDS.md](docs/DESKTOP_BUILDS.md).

## Configuration

Everything is an environment variable, read by `Settings` in `biobench_sar/config.py`. The variables can also
go in a `sar.env` file in the repo root: copy `sar.env.example`, or point `SAR_ENV_FILE` at another file.
`python -m biobench_sar` and `scripts/hardware_check.py` load it at startup, and a real environment variable
wins over the file. Desktop builds also look next to the app and in the data folder (see
[DESKTOP_BUILDS.md](docs/DESKTOP_BUILDS.md#settings-sarenv)). Room and hardware advice:
[docs/HARDWARE_SETUP.md](docs/HARDWARE_SETUP.md). Invalid numbers fall back to the default.

| Variable | Default | Meaning |
| --- | --- | --- |
| **Subject and language** | | |
| `SAR_PROFILE` | `math` | Subject: `math` (tutor) or `biology` (research assistant). Anything else means `math` |
| `SAR_LANGUAGE` | `auto` | Answer language: `auto` (the language of each question), `en` or `es` (names and locales such as `es-MX` or `Español` work too). The dashboard switch changes it while running |
| `SAR_QUIZ_LENGTH` | `10` | The most problems one quiz asks, and how many the code makes up when no ticked file has any |
| **Camera and projector** | | |
| `SAR_CAMERA_INDEX` | `0` | Which camera (`scripts/hardware_check.py` lists them). The kit's C930e is usually `1` |
| `SAR_CAMERA_WIDTH` / `_HEIGHT` | `1280` / `720` | Camera resolution to ask for. Use `1920` / `1080` for math (small symbols) |
| `SAR_PROJECTOR_WIDTH` / `_HEIGHT` | `1280` / `800` | Projector resolution in pixels (the kit's Epson WXGA). Targets, the highlight and calibration use it |
| `SAR_CHECKER_COLS` / `_ROWS` | `9` / `6` | Inner corners of the calibration checkerboard |
| **Ink detection** | | |
| `SAR_INK_MIN_AREA` | `150` | Smallest ink blob, in projector pixels², that counts |
| `SAR_INK_BLOCK_SIZE` / `SAR_INK_C` | `31` / `15` | Adaptive-threshold window (odd, ≥ 3) and offset |
| `SAR_INK_MAX_REGIONS` | `20` (math), `10` (biology) | Most ink regions passed on, largest first |
| `SAR_INK_SMALL_MERGE_PX` | `24` | Specks this close to a symbol (decimal points, minus signs, exponents) join its region. `0` turns this off |
| **Model and knowledge base** | | |
| `SAR_VLM_BACKEND` | `ollama` | `ollama`, `openai` (any compatible API) or `mock` (no model) |
| `SAR_VLM_MODEL` | `qwen2.5vl:3b` | Model name |
| `SAR_VLM_BASE_URL` / `SAR_VLM_API_KEY` | `http://localhost:11434` / unset | Endpoint and key for the backend |
| `SAR_VLM_TIMEOUT` | `90` | Seconds to wait for the model |
| `SAR_EMBED_MODEL` | `all-MiniLM-L6-v2` | sentence-transformers model for the knowledge base |
| `SAR_RAG_TOP_K` | `4` | Chunks retrieved per question |
| `SAR_SEED_KNOWLEDGE` | `1` | Load `biobench_sar/knowledge/<profile>/` at startup, once per profile and knowledge base. `0` skips it |
| **Voice** | | |
| `SAR_LISTEN_MODE` | `push` | `push` (Space / Talk) or `wake` (always listening for the wake word) |
| `SAR_WAKE_WORD` | `assistant` | Wake word ("asistente" works too) |
| `SAR_WHISPER_MODEL` | `base` | Local Whisper model size |
| `SAR_RECORD_SECONDS` | `5` | How long push-to-talk records |
| `SAR_SILENCE_RMS` | `0.01` | Wake-word chunks quieter than this skip Whisper. Push-to-talk only skips digital silence |
| `SAR_TTS_BACKEND` | `edge` | `edge` (online), `say` (macOS), `pyttsx3` (offline) or `none`. Falls back edge → say → pyttsx3 |
| `SAR_TTS_VOICE` | `en-US-AriaNeural` | Edge voice for its own language. Spanish uses `es-MX-DaliaNeural` unless this names a Spanish voice |
| **Server and phones** | | |
| `SAR_HOST` | `127.0.0.1` (`0.0.0.0` with `SAR_LAN=1`) | Bind address. Anything other than `127.0.0.1`, `localhost` or `::1` turns the phone PIN gate on |
| `SAR_PORT` | `5050` | Server port |
| `SAR_LAN` | off | `1` = phone mode: listen on the Wi-Fi, behind a six-digit PIN ([docs/MOBILE.md](docs/MOBILE.md)) |
| `SAR_LAN_HOSTNAMES` | empty | Comma list of extra host names phones may use (e.g. `my-laptop.local`). IP addresses and `localhost` always work |
| **Files** | | |
| `SAR_ENV_FILE` | `sar.env` in the repo root | Settings file to load before `Settings` is built |
| `SAR_DATA_DIR` | `data/` in the checkout (desktop builds: the per-user data folder) | Where everything below is kept. Shared by both profiles unless you give each its own |
| `SAR_CALIBRATION_FILE` | `data/calibration.json` | Camera ↔ projector line-up |
| `SAR_CHROMA_DIR` | `data/chroma_db` | The knowledge base. Delete it to load the bundled documents again |
| `SAR_UPLOAD_DIR` | `data/uploads` | Uploads while they are read (deleted afterwards) |
| `SAR_TTS_DIR` | `data/tts` | Generated speech audio |
| `SAR_SELECTION_FILE` | `data/selection.json` | Which files are ticked |
| `SAR_QUIZ_BANK_FILE` | `data/quiz_bank.json` | Quiz problems parsed from uploaded CSVs and worksheets |

## Development

```bash
.venv/bin/python -m pip install -r requirements-dev.txt   # or: uv pip install --python .venv/bin/python -r requirements-dev.txt
.venv/bin/python -m pytest -q        # unit tests, no hardware or network needed
.venv/bin/ruff check . && .venv/bin/ruff format --check .
SAR_VLM_BACKEND=mock SAR_TTS_BACKEND=none .venv/bin/python -m biobench_sar   # UI without a model
.venv/bin/python -m scripts.bench_vlm -n 3   # VLM latency against Ollama
.venv/bin/python -m scripts.bench_vlm --scenarios --profile math      # math pass/fail scenarios
.venv/bin/python -m scripts.bench_vlm --scenarios --profile biology   # biology flowchart cases (the default profile)
.venv/bin/python scripts/tune_ink.py --quick                          # ink-threshold sweep on synthetic boards
.venv/bin/python packaging/launcher.py --self-test                     # what CI runs on every desktop build
```

`--only H,QC` runs only those scenarios, `--runs N` repeats each one, and `--raw` grades the model without the
VLM client's math post-processing.

CI runs lint and tests on every pull request, and desktop builds on pull requests that touch packaging. Every
change goes through its own branch and PR (see [CONTRIBUTING.md](CONTRIBUTING.md)). AI agents start at
[AGENTS.md](AGENTS.md).

## Repository map

| Path | What |
| --- | --- |
| `biobench_sar/app.py` | Flask + Socket.IO routes, the phone PIN gate's wiring, the startup banner |
| `biobench_sar/assistant.py` | Sense → Think → Act pipeline, status, history, file selection, language, target snapping, UI masking |
| `biobench_sar/tutor.py` | Tutor modes (help / practice / quiz), learn-a-topic requests, voice commands, quiz progress, stop confirmation, conversation memory |
| `biobench_sar/grading.py` | Quiz answers graded in code with sympy: `parse_answers`, `grade` |
| `biobench_sar/lang.py` | English/Spanish detection and the Spanish tutor intents |
| `biobench_sar/lan.py` | Phones on the Wi-Fi: addresses, the join PIN, device cookies, the Socket.IO gate |
| `biobench_sar/config.py`, `envfile.py`, `paths.py`, `schemas.py` | Settings, `sar.env` loading, where files live (checkout or desktop build), shared types |
| `biobench_sar/core/` | Camera loop and ink detection, high-resolution ink crop, glare check, homography calibration, mic and wake word |
| `biobench_sar/rag/` | Document ingestion (answer keys stripped for math), ChromaDB store, quiz bank parser and problem generator |
| `biobench_sar/llm/` | VLM client (prompts per profile, mode and language + JSON schema), math-to-speech in English and Spanish, text-to-speech |
| `biobench_sar/templates/`, `static/` | Projector page, operator dashboard, phone join page, translations (`static/js/i18n.js`), PWA manifest and icons |
| `biobench_sar/knowledge/math/`, `knowledge/biology/` | Seed documents for each profile |
| `packaging/` | Desktop builds: launcher, PyInstaller spec, Windows installer, model fetcher |
| `scripts/` | Setup, hardware check, VLM benchmark, ink threshold sweep, local builds, icon generator |
| `docs/` | Specs, build plan and contracts, hardware guides, desktop builds, phones |
| `tests/` | pytest suite |
