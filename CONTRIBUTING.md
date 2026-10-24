# Contributing — SAR Whiteboard Math Assistant

## Start here
1. Clone: `git clone https://github.com/Enrique182004/Hackaton.git && cd Hackaton`
2. Read the specs: [docs/sar_math_tutor_prompt.md](docs/sar_math_tutor_prompt.md) (math, the default profile)
   and [docs/sar_whiteboard_assistant_prompt.md](docs/sar_whiteboard_assistant_prompt.md) (biology profile,
   module layout). The contracts between modules are in [docs/BUILD_PLAN.md](docs/BUILD_PLAN.md).
3. Read [CLAUDE.md](CLAUDE.md) (project rules) and [SETUP.md](SETUP.md) (tooling checklist).
4. Using an AI agent? Open it in the repo root — it picks up `CLAUDE.md` / `AGENTS.md` automatically.

## Workflow
- Once: `bash scripts/setup_mac.sh` (or `scripts/setup_windows.ps1`), then
  `.venv/bin/python -m pip install -r requirements-dev.txt` for pytest and ruff.
- Before pushing: `.venv/bin/python -m pytest -q` and `.venv/bin/ruff check . && .venv/bin/ruff format --check .` (CI runs both).
  In Windows PowerShell: `.\.venv\Scripts\python.exe -m pytest -q` and
  `.\.venv\Scripts\ruff.exe check .; .\.venv\Scripts\ruff.exe format --check .`.
- One module per branch: `git checkout -b feat/vision-loop` (match the spec's file names).
- Keep PRs to one module (e.g. `core/homography.py`) so they're easy to review; every behaviour change needs a test.
- Never commit `.env`, API keys, model weights or `chroma_db/` (already in `.gitignore`).
- Open a PR into `main`; describe what you tested and how.

## Cleanup pass (every branch and PR)
Before you open a PR, clean the files your branch touches, plus anything your change left dead elsewhere.
Fix what is cheap and safe in the same PR; anything bigger goes in the PR description's **Cleanup** list
(`file:line` and what is wrong), so the next PR that touches that file picks it up.

1. **Nothing dead.** No function, constant, parameter, route or JS export that only tests use, or that
   nothing uses: `git grep -nw <name> -- biobench_sar scripts` before you keep it. Wire it in or remove it.
2. **Nothing duplicated.** Before adding a helper, constant or user-facing sentence, grep for an existing one
   and import it. One home each: shared text like `NO_ANSWER_TEXT`, `config.PROFILES`, one atomic JSON write.
3. **No stand-ins for modules that exist.** No `try: from biobench_sar.x import … except ImportError`
   fallbacks, and no "once X lands" comments, for code that is already in the repo.
4. **Settings in one place.** Every `SAR_*` variable is read in `config.py` through the forgiving `_env_*`
   helpers inside a `default_factory`, never at import time and never with a bare `int()`/`float()`. A variable
   is in `sar.env.example` or the README only if `Settings` reads it (the one exception is `SAR_ENV_FILE`, which
   `envfile.py` reads before `Settings` exists). A new setting goes in the README's configuration table with its
   real default.
5. **Docs still true.** Commands, file names and settings that README, CLAUDE.md, AGENTS.md, this file and
   `docs/` mention still exist and agree. New modules go in the table below.
6. **No leftovers.** No debug `print`/`console.log`, commented-out code or TODO without a reason.
7. **Gates green.** pytest, `ruff check .` and `ruff format --check .`.

## Modules
All paths are inside `biobench_sar/`.

| Area | Files |
| --- | --- |
| Server | `app.py` (routes, Socket.IO, the phone PIN gate), `assistant.py` (Sense → Think → Act pipeline, status, history, file selection, language, target snapping, UI masking), `config.py`, `envfile.py` (`sar.env`), `paths.py` (checkout vs desktop build), `ports.py` (ports browsers refuse to open), `schemas.py` |
| Tutor | `tutor.py` (modes, learn-a-topic requests, voice commands, quiz progress, stop confirmation, conversation memory), `grading.py` (quiz answers graded with sympy), `lang.py` (English/Spanish detection and Spanish intents) |
| Phones | `lan.py` (PIN, device cookies, host checks, Socket.IO gate), `templates/join.html`, `static/js/dashboard/phone.js`, `static/sw.js`, `static/manifest.webmanifest` |
| Vision | `core/homography.py`, `core/vision_loop.py`, `core/ink_crop.py`, `core/glare.py` |
| Audio | `core/audio_trigger.py`, `llm/tts_engine.py` |
| RAG | `rag/ingestor.py`, `rag/vector_store.py`, `rag/quiz_bank.py` |
| VLM | `llm/vlm_client.py`, `llm/mathspeak.py`, `llm/mathspeak_es.py` |
| Seed documents | `knowledge/math/`, `knowledge/biology/` |
| Projector page | `templates/index.html`, `static/css/style.css`, `static/js/main.js` |
| Dashboard | `templates/precharge.html`, `static/css/dashboard.css`, `static/js/dashboard/*.js`, `static/js/i18n.js` (English and Spanish text for both pages) |

Outside the package: `packaging/` (desktop builds: `launcher.py`, `sar.spec`, `windows/installer.iss`,
`fetch_models.py`, `check_macos_app.py`), `scripts/` (setup, hardware check, VLM benchmark, ink sweep, local
builds, icons) and `docs/` (specs, build plan, hardware guides in English and Spanish, desktop builds, phones).

Optional: to keep your own agent focused across sessions, add to your personal
`~/.claude/CLAUDE.md`:

```md
## Current focus — SAR Whiteboard Math Assistant
- Repo: <your clone path> ⇄ github.com/Enrique182004/Hackaton
- Specs: docs/sar_math_tutor_prompt.md (math, default) + docs/sar_whiteboard_assistant_prompt.md (biology); rules in the repo's CLAUDE.md.
```
