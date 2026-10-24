# CLAUDE.md — hackathon project (copy into each repo, fill the Stack)

## Stack
- Frontend: <React/Next.js? fill in>
- Backend: <Node/Express/Next API? fill in>
- DB: <Postgres/Prisma/Supabase? fill in>
- Tests: <Vitest? fill in>

## Commands
```bash
npm run dev
npm run build
npm run test
npm run lint
```

## Code rules
- Be terse. No preamble, no summaries unless asked. Output code only.
- TypeScript always, never `any`. Named exports. Components < 150 lines.
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
