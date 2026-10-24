# Contributing — SAR Whiteboard Assistant

## Start here
1. Clone: `git clone https://github.com/Enrique182004/Hackaton.git && cd Hackaton`
2. Read the spec: [docs/sar_whiteboard_assistant_prompt.md](docs/sar_whiteboard_assistant_prompt.md).
3. Read [CLAUDE.md](CLAUDE.md) (project rules) and [SETUP.md](SETUP.md) (tooling checklist).
4. Using an AI agent? Open it in the repo root — it picks up `CLAUDE.md` / `AGENTS.md` automatically.

## Workflow
- One module per branch: `git checkout -b feat/vision-loop` (match the spec's file names).
- Keep PRs to one module (e.g. `core/homography.py`) so they're easy to review.
- Never commit `.env`, API keys, model weights or `chroma_db/` (already in `.gitignore`).
- Open a PR into `main`; describe what you tested and how.

## Modules (from the spec)
| Area | Files |
| --- | --- |
| Server | `app.py` |
| Vision | `core/homography.py`, `core/vision_loop.py` |
| Audio | `core/audio_trigger.py`, `llm/tts_engine.py` |
| RAG | `rag/ingestor.py`, `rag/vector_store.py` |
| VLM | `llm/vlm_client.py` |
| Frontend | `templates/index.html`, `static/css/style.css`, `static/js/main.js` |

Optional: to keep your own agent focused across sessions, add to your personal
`~/.claude/CLAUDE.md`:

```md
## Current focus — SAR Whiteboard Assistant
- Repo: <your clone path> ⇄ github.com/Enrique182004/Hackaton
- Spec: docs/sar_whiteboard_assistant_prompt.md; rules in the repo's CLAUDE.md.
```
