# Hackathon setup — token-lean checklist

Run this once before the event. Order matters.

## Per machine (once)
- [ ] **App setup** — `bash scripts/setup_mac.sh --pull-model` (Windows: `scripts/setup_windows.ps1`), then
      `.venv/bin/python -m pip install -r requirements-dev.txt`. Check with `.venv/bin/python scripts/hardware_check.py`.
- [ ] **Ollama running** — open the Ollama app. The app needs the vision model **`qwen2.5vl:3b`**
      (`ollama pull qwen2.5vl:3b`, or the `--pull-model` flag above). `qwen2.5-coder:7b` is optional and only
      used by the `draft-with-ollama` skill for drafting code.
- [ ] **Headroom installed + running** — double-click `start-headroom.command`.
      Then start Claude through it: `ANTHROPIC_BASE_URL=http://localhost:8787 claude`
      (biggest raw token cut — 20% general, 60–95% on JSON/tool-heavy work).
- [ ] **Model habit:** `/model sonnet` for feature work, `/model haiku` for mechanical
      edits, Opus only for architecture/hard bugs.

## Per project (2 min each)
- [x] `CLAUDE.md`, `AGENTS.md` and `.claudeignore` are already committed in the repo root — nothing to copy.
- [ ] Run `/learn-codebase` once (claude-mem primes it — stops re-reading every session).

## Habits during the hackathon
- [ ] **Plan first:** `/brainstorming` → `/writing-plans` before building anything new.
- [ ] **Offload bulk:** `/draft-with-ollama` for CRUD, scaffolding, test stubs (free).
- [ ] **Compact:** `/compact` long sessions; start fresh per feature.
- [ ] **Verify before 'done':** `/verification-before-completion`.
- [ ] **Watch spend:** `/cost` regularly; split independent work into cheap subagents.

## Order of impact (if short on time)
1. Headroom proxy  2. Model routing (Opus→Sonnet/Haiku)  3. `/learn-codebase` + `.claudeignore`
4. Ollama offload  5. Plan-before-build
