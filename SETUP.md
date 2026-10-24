# Hackathon setup — token-lean checklist

Run this once before the event. Order matters.

## Per machine (once)
- [ ] **Ollama running** — open the Ollama app (model: `qwen2.5-coder:7b`).
- [ ] **Headroom installed + running** — double-click `start-headroom.command`.
      Then start Claude through it: `ANTHROPIC_BASE_URL=http://localhost:8787 claude`
      (biggest raw token cut — 20% general, 60–95% on JSON/tool-heavy work).
- [ ] **Model habit:** `/model sonnet` for feature work, `/model haiku` for mechanical
      edits, Opus only for architecture/hard bugs.

## Per project (2 min each)
- [ ] Copy `CLAUDE.md` into the repo root and fill in the Stack + Commands.
- [ ] Copy `.claudeignore` into the repo root.
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
