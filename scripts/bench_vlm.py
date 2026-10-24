"""Benchmark VLMClient.evaluate against a real Ollama backend.

Usage: python scripts/bench_vlm.py [-n 3] [--model qwen2.5vl:3b] [--url http://localhost:11434]
"""
from __future__ import annotations

import argparse
import statistics
import time

import cv2
import numpy as np

from biobench_sar.llm.vlm_client import VLMClient
from biobench_sar.schemas import InkRegion, RetrievedChunk

SIZE = (1920, 1080)
QUERIES = ["Is my flowchart correct?", "Check my procedure, please.", "Does this look right to you?"]
STEPS = ["Lake water sample", "Filter 0.45um", "NNA + E. coli", "Incubate 30C", "Microscopy"]

CONTEXT = [
    RetrievedChunk("Acanthamoeba is isolated by plating filtered water on non-nutrient agar (NNA) "
                   "overlaid with a heat-killed E. coli lawn and incubating at 30C.", "protocol.pdf", 0.9, {}),
    RetrievedChunk("Sterile technique: autoclave media and tools at 121C and flame-sterilize loops.",
                   "sterile_technique.pdf", 0.8, {}),
]


def make_flowchart(steps: list[str]) -> tuple[bytes, list[InkRegion]]:
    img = np.full((SIZE[1], SIZE[0], 3), 255, np.uint8)
    n = len(steps)
    bw, bh = 300, 120
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
            cv2.arrowedLine(img, (x + bw + 5, y + bh // 2), (x + bw + gap - 5, y + bh // 2), (0, 0, 0), 4, tipLength=0.3)
        regions.append({"x": x, "y": y, "w": bw, "h": bh, "cx": x + bw // 2, "cy": y + bh // 2})
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok
    return buf.tobytes(), regions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=3)
    ap.add_argument("--model", default="qwen2.5vl:3b")
    ap.add_argument("--url", default="http://localhost:11434")
    ap.add_argument("--cold", action="store_true", help="unload the model first (measures load + first answer)")
    args = ap.parse_args()
    if args.cold:
        import requests

        requests.post(f"{args.url}/api/generate", json={"model": args.model, "keep_alive": 0}, timeout=60)
        time.sleep(2)

    client = VLMClient("ollama", args.model, args.url, timeout=180)
    warm = getattr(client, "warmup", None)
    if warm:
        t0 = time.perf_counter()
        warm()
        print(f"warmup: {time.perf_counter() - t0:.1f}s")
    jpeg, regions = make_flowchart(STEPS)
    times: list[float] = []
    for i in range(args.n):
        t = time.perf_counter()
        a = client.evaluate(QUERIES[i % len(QUERIES)], jpeg, CONTEXT, regions, SIZE)
        dt = time.perf_counter() - t
        times.append(dt)
        print(f"run {i + 1}: {dt:.1f}s  {a.animation_state} ({a.target_x},{a.target_y}) {a.spoken_text!r}")
    print(f"median: {statistics.median(times):.1f}s  min {min(times):.1f}s  max {max(times):.1f}s")


if __name__ == "__main__":
    main()
