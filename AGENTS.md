# AGENTS.md — read this first (any AI agent: Claude, Codex, Cursor, Copilot, Gemini…)

This repo builds ONE thing: the **SAR Whiteboard Assistant** (hackathon) — by default a **math tutor**.

1. Read the spec for the subject: [docs/sar_math_tutor_prompt.md](docs/sar_math_tutor_prompt.md) is the source
   of truth (`SAR_PROFILE=math`, default). The biology spec,
   [docs/sar_whiteboard_assistant_prompt.md](docs/sar_whiteboard_assistant_prompt.md), is kept as the second
   profile (`SAR_PROFILE=biology`) and for the shared module layout.
2. Read the contracts in [docs/BUILD_PLAN.md](docs/BUILD_PLAN.md) (including the "Implementation notes" on where
   the code departs from the specs) and the project rules in [CLAUDE.md](CLAUDE.md).
3. Build **one module at a time** in the `/biobench_sar` layout; don't generate the whole app at once.
4. Math in `spoken_text` is always words, never `^ / * =`; help mode never gives the final answer; quiz answers
   are graded in code (`grading.py`), never by the model; every user-facing text exists in English and Spanish.
5. Keep the camera loop non-blocking (~30 FPS); wrap VLM JSON parsing in `try/except`.
6. Before claiming done, run and show the output of:
   ```bash
   .venv/bin/python -m pytest -q
   .venv/bin/ruff check . && .venv/bin/ruff format --check .
   ```
   (dev deps: `.venv/bin/python -m pip install -r requirements-dev.txt`; run the app with
   `.venv/bin/python -m biobench_sar`).
7. Before opening a PR, run the cleanup pass in [CONTRIBUTING.md](CONTRIBUTING.md#cleanup-pass-every-branch-and-pr)
   over the files you touched.
