# AGENTS.md — read this first (any AI agent: Claude, Codex, Cursor, Copilot, Gemini…)

This repo builds ONE thing: the **SAR Whiteboard Assistant** (hackathon).

1. Read the spec: [docs/sar_whiteboard_assistant_prompt.md](docs/sar_whiteboard_assistant_prompt.md) — the source of truth.
2. Follow the project rules in [CLAUDE.md](CLAUDE.md) (stack, commands, code rules).
3. Build **one module at a time** in the spec's `/biobench_sar` layout; don't generate the whole app at once.
4. Keep the camera loop non-blocking (30+ FPS); wrap VLM JSON parsing in `try/except`.
5. Before claiming done: run the code/tests and show the output.
