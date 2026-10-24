# Hackaton — SAR Whiteboard Assistant

A Spatial Augmented Reality whiteboard assistant: a webcam and projector pointed at a whiteboard,
with an animated 2D AI avatar that reads marker drawings and responds by voice and movement,
grounded in a local RAG knowledge base.

**Spec:** [docs/sar_whiteboard_assistant_prompt.md](docs/sar_whiteboard_assistant_prompt.md). This is the single source of truth for the build.

## For agents
Start with `CLAUDE.md`, then read the spec. Build one module at a time.

## Files
- `CLAUDE.md`: project focus, stack, rules, skill routing and gates.
- `docs/sar_whiteboard_assistant_prompt.md`: full system spec.
- `SETUP.md`: setup checklist to run before the event.
- `.claudeignore`: keeps dependencies, builds, secrets and media out of Claude's context.
- `start-headroom.command`: installs and starts the Headroom token-compression proxy.
