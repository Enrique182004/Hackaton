"""Benchmark VLMClient.evaluate against a real Ollama backend.

Usage: python scripts/bench_vlm.py [-n 3] [--model qwen2.5vl:3b] [--url http://localhost:11434]
       python scripts/bench_vlm.py --scenarios [--runs 5] [--profile biology|math] [--only H,P]
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from biobench_sar.llm.vlm_client import VLMClient
from biobench_sar.schemas import AvatarAction, InkRegion, RetrievedChunk

SIZE = (1920, 1080)
QUERIES = ["Is my flowchart correct?", "Check my procedure, please.", "Does this look right to you?"]
SCENARIO_QUERY = "Check my flowchart. Is the procedure correct and complete?"
STEPS = ["Lake water sample", "Filter 0.45um", "NNA + E. coli", "Incubate 30C", "Microscopy"]
STERILE_STEPS = ["Autoclave media + tools", "Flame-sterilize loop", *STEPS]

CONTEXT = [
    RetrievedChunk(
        "Acanthamoeba is isolated by plating filtered water on non-nutrient agar (NNA) "
        "overlaid with a heat-killed E. coli lawn and incubating at 30C.",
        "protocol.pdf",
        0.9,
        {},
    ),
    RetrievedChunk(
        "Sterile technique: autoclave media and tools at 121C and flame-sterilize loops.",
        "sterile_technique.pdf",
        0.8,
        {},
    ),
]
MATH_CONTEXT = [
    RetrievedChunk(
        "Two-step equations: undo addition or subtraction first, then multiplication or division. "
        "Ask the student to write each step before moving on.",
        "math_tutoring_guide.md",
        0.8,
        {},
    )
]

EQUATION = "3x + 5 = 20"
SPOKEN_EQUATION = "three x plus five equals twenty"
# The solved value said as a result, not "five" as a constant of the equation ("plus five").
_LEAK_RE = re.compile(
    r"\b(?:x\s*(?:=|equals|is(?:\s+equal\s+to)?)|answer\s+is|solution\s+is)\s*(?:5|five)\b", re.IGNORECASE
)
_SYMBOL_RE = re.compile(r"[\^/*=]")
QUIZ_TASK = {
    "kind": "quiz",
    "stage": "check",
    "number": 1,
    "total": 5,
    "problem": EQUATION,
    "answer": "x = 5",
    "next_problem": "2x - 4 = 10",
}


def leaks_answer(spoken: str) -> bool:
    return bool(_LEAK_RE.search(spoken))


def has_symbols(spoken: str) -> bool:
    return bool(_SYMBOL_RE.search(spoken))


def dictates_new_equation(spoken: str) -> bool:
    """At least one equation in the reply that is not just the board's equation read back."""
    text = " ".join(spoken.lower().split())
    equations = len(re.findall(r"\bequals\b|=", text))
    repeats = text.count(SPOKEN_EQUATION) + len(re.findall(r"3\s*x\s*\+\s*5\s*=\s*20", text))
    return equations > repeats


def reads_next_problem(spoken: str) -> bool:
    return bool(re.search(r"\b(?:two|2)\s*x\s*(?:minus|-)\s*(?:four|4)\b", spoken, re.IGNORECASE))


class Scenario(NamedTuple):
    name: str
    query: str
    board: Callable[[], tuple[bytes, list[InkRegion]]]
    expected: str
    mode: str = "help"
    task: dict[str, Any] | None = None
    context: list[RetrievedChunk] = CONTEXT
    check: Callable[[str], str | None] = lambda _spoken: None  # returns a failure reason or None


def make_flowchart(steps: list[str]) -> tuple[bytes, list[InkRegion]]:
    img = np.full((SIZE[1], SIZE[0], 3), 255, np.uint8)
    n = len(steps)
    bw, bh = min(300, (SIZE[0] - 120) // n - 40), 120
    gap = (SIZE[0] - 2 * 60 - n * bw) // max(n - 1, 1)
    y = SIZE[1] // 2 - bh // 2
    regions: list[InkRegion] = []
    for i, label in enumerate(steps):
        x = 60 + i * (bw + gap)
        cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 0, 0), 4)
        words = label.split(" ", 1) if len(label) > 14 else [label]
        for j, w in enumerate(words):
            cv2.putText(img, w, (x + 15, y + 50 + j * 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        if i < n - 1:
            cv2.arrowedLine(
                img, (x + bw + 5, y + bh // 2), (x + bw + gap - 5, y + bh // 2), (0, 0, 0), 4, tipLength=0.3
            )
        regions.append({"x": x, "y": y, "w": bw, "h": bh, "cx": x + bw // 2, "cy": y + bh // 2})
    return _jpeg(img), regions


def make_board(lines: list[str]) -> tuple[bytes, list[InkRegion]]:
    """Marker-style lines of math, one ink region per line, in projector coordinates."""
    img = np.full((SIZE[1], SIZE[0], 3), 255, np.uint8)
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 3.0, 8
    regions: list[InkRegion] = []
    for i, line in enumerate(lines):
        (w, h), base = cv2.getTextSize(line, font, scale, thick)
        x, y = 300, 300 + i * 260
        cv2.putText(img, line, (x, y), font, scale, (20, 20, 20), thick, cv2.LINE_AA)
        top, height = y - h, h + base
        regions.append({"x": x, "y": top, "w": w, "h": height, "cx": x + w // 2, "cy": top + height // 2})
    return _jpeg(img), regions


def _jpeg(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok
    return buf.tobytes()


def _no_leak(spoken: str) -> str | None:
    return "gave the final answer" if leaks_answer(spoken) else None


def _new_equation(spoken: str) -> str | None:
    return None if dictates_new_equation(spoken) else "no new equation dictated"


def _next_problem(spoken: str) -> str | None:
    return None if reads_next_problem(spoken) else "next problem not read"


BIO_SCENARIOS = {
    "A": Scenario("missing sterile step", SCENARIO_QUERY, lambda: make_flowchart(STEPS), "error"),
    "B": Scenario("complete procedure", SCENARIO_QUERY, lambda: make_flowchart(STERILE_STEPS), "celebrating"),
}
MATH_SCENARIOS = {
    "H": Scenario(
        f"help on {EQUATION}",
        "I'm stuck. What do I do next?",
        lambda: make_board([EQUATION]),
        "pointing",
        context=MATH_CONTEXT,
        check=_no_leak,
    ),
    "P": Scenario(
        f"similar problem to {EQUATION}",
        "Give me a similar problem.",
        lambda: make_board([EQUATION]),
        "idle",
        mode="practice",
        task={"kind": "practice"},
        context=MATH_CONTEXT,
        check=_new_equation,
    ),
    "QC": Scenario(
        "quiz check, correct answer x = 5",
        "Is my answer right?",
        lambda: make_board([EQUATION, "x = 5"]),
        "celebrating",
        mode="quiz",
        task=QUIZ_TASK,
        context=[],
        check=_next_problem,
    ),
    "QW": Scenario(
        "quiz check, wrong answer x = 7",
        "Is my answer right?",
        lambda: make_board([EQUATION, "x = 7"]),
        "error",
        mode="quiz",
        task=QUIZ_TASK,
        context=[],
        check=_no_leak,
    ),
}
SCENARIOS = {"biology": BIO_SCENARIOS, "math": MATH_SCENARIOS}


def judge(scenario: Scenario, action: AvatarAction) -> list[str]:
    problems = [] if action.animation_state == scenario.expected else [f"state {action.animation_state}"]
    reason = scenario.check(action.spoken_text)
    return problems + ([reason] if reason else [])


def run_scenarios(client: VLMClient, runs: int, profile: str, only: list[str] | None = None) -> None:
    for key, scenario in SCENARIOS[profile].items():
        if only and key not in only:
            continue
        jpeg, regions = scenario.board()
        counts: Counter[str] = Counter()
        passed, symbols, times = 0, 0, []
        for i in range(runs):
            t = time.perf_counter()
            a = client.evaluate(
                scenario.query, jpeg, scenario.context, regions, SIZE, mode=scenario.mode, task=scenario.task
            )
            times.append(time.perf_counter() - t)
            counts[a.animation_state] += 1
            fails = judge(scenario, a)
            passed += not fails
            symbols += has_symbols(a.spoken_text)
            verdict = "ok" if not fails else "FAIL: " + ", ".join(fails)
            print(
                f"  [{key}] run {i + 1}: {times[-1]:.1f}s {a.animation_state} ({a.target_x},{a.target_y}) "
                f"{verdict} {a.spoken_text!r}"
            )
        sym = f", symbols in {symbols}/{runs}" if profile == "math" else ""
        print(
            f"case {key} ({scenario.name}): expect {scenario.expected}: {dict(counts)} -> {passed}/{runs} pass"
            f"{sym}, median {statistics.median(times):.1f}s"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=3)
    ap.add_argument("--model", default="qwen2.5vl:3b")
    ap.add_argument("--url", default="http://localhost:11434")
    ap.add_argument("--cold", action="store_true", help="unload the model first (measures load + first answer)")
    ap.add_argument("--profile", choices=sorted(SCENARIOS), default="biology")
    ap.add_argument("--scenarios", action="store_true", help="run the profile's pass/fail scenarios")
    ap.add_argument("--only", help="comma-separated scenario keys, e.g. H,QC")
    ap.add_argument("--runs", type=int, default=5, help="runs per scenario")
    args = ap.parse_args()
    if args.cold:
        import requests

        requests.post(f"{args.url}/api/generate", json={"model": args.model, "keep_alive": 0}, timeout=60)
        time.sleep(2)

    client = VLMClient("ollama", args.model, args.url, timeout=180, profile=args.profile)
    t0 = time.perf_counter()
    client.warmup()
    print(f"warmup: {time.perf_counter() - t0:.1f}s")
    if args.scenarios:
        run_scenarios(client, args.runs, args.profile, args.only.split(",") if args.only else None)
        return
    if args.profile == "math":
        scenario = MATH_SCENARIOS["H"]
        jpeg, regions = scenario.board()
        queries, context = [scenario.query], scenario.context
    else:
        jpeg, regions = make_flowchart(STEPS)
        queries, context = QUERIES, CONTEXT
    times: list[float] = []
    for i in range(args.n):
        t = time.perf_counter()
        a = client.evaluate(queries[i % len(queries)], jpeg, context, regions, SIZE)
        dt = time.perf_counter() - t
        times.append(dt)
        print(f"run {i + 1}: {dt:.1f}s  {a.animation_state} ({a.target_x},{a.target_y}) {a.spoken_text!r}")
    print(f"median: {statistics.median(times):.1f}s  min {min(times):.1f}s  max {max(times):.1f}s")


if __name__ == "__main__":
    main()
