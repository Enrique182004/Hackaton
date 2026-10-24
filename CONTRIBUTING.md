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
- One module per branch: `git checkout -b feat/vision-loop` (match the spec's file names).
- Keep PRs to one module (e.g. `core/homography.py`) so they're easy to review; every behaviour change needs a test.
- Never commit `.env`, API keys, model weights or `chroma_db/` (already in `.gitignore`).
- Open a PR into `main`; describe what you tested and how.

## Modules
All paths are inside `biobench_sar/`.

| Area | Files |
| --- | --- |
| Server | `app.py` (routes, Socket.IO), `assistant.py` (Sense → Think → Act pipeline, status, history, file selection), `config.py`, `schemas.py` |
| Tutor | `tutor.py` (modes, voice commands, quiz progress) |
| Vision | `core/homography.py`, `core/vision_loop.py` |
| Audio | `core/audio_trigger.py`, `llm/tts_engine.py` |
| RAG | `rag/ingestor.py`, `rag/vector_store.py`, `rag/quiz_bank.py` |
| VLM | `llm/vlm_client.py`, `llm/mathspeak.py` |
| Seed documents | `knowledge/math/`, `knowledge/biology/` |
| Projector page | `templates/index.html`, `static/css/style.css`, `static/js/main.js` |
| Dashboard | `templates/precharge.html`, `static/css/dashboard.css`, `static/js/dashboard/*.js` |

Optional: to keep your own agent focused across sessions, add to your personal
`~/.claude/CLAUDE.md`:

```md
## Current focus — SAR Whiteboard Math Assistant
- Repo: <your clone path> ⇄ github.com/Enrique182004/Hackaton
- Specs: docs/sar_math_tutor_prompt.md (math, default) + docs/sar_whiteboard_assistant_prompt.md (biology); rules in the repo's CLAUDE.md.
```
