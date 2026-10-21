"""The README's numbers, as markdown tables, from the raw results.

    uv run report.py evals evals/*.jsonl                  evals, per scenario and per model
    uv run --env-file .env report.py call <call_id>       latency by stage of one real call

The evals write their results when EVAL_RESULTS names a file (see tests/test_agent.py).
"""

import json
import math
import os
import statistics
import sys
from collections import defaultdict

import psycopg

STAGES = {
    "transcription_delay": "Final transcript after the caller stops",
    "end_of_turn_delay": "End of turn decided",
    "llm_node_ttft": "LLM first token",
    "tts_node_ttfb": "TTS first audio",
}


def wilson(passed: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval of a pass rate. Unlike p ± z·se it stays within 0 and 1 near 100%."""
    p, d = passed / n, 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


def evals(paths: list[str]) -> None:
    models = defaultdict(dict)  # model -> scenario -> totals over every run given
    for path in paths:
        with open(path) as f:
            results = [json.loads(line) for line in f]
        for r in results:
            s = models[r["model"]].setdefault(
                r["scenario"],
                {"label": r["label"], "checked_by": r["checked_by"], "n": 0, "passed": 0, "ttfts": [], "usd": 0.0},
            )
            s["n"] += r["rounds"]
            s["passed"] += r["passed"]
            s["ttfts"] += r["ttfts"]
            s["usd"] += r["usd"] * r["rounds"]
    for model, scenarios in models.items():
        print(f"\n`{model}`\n\n| Scenario | Checked by | Passed |\n|---|---|---|")
        for s in scenarios.values():
            passed = f"{s['passed']}/{s['n']}"
            print(f"| {s['label']} | {s['checked_by']} | {passed if s['passed'] == s['n'] else f'**{passed}**'} |")
    print(
        "\n| Model | Passed | 95% interval | LLM time to first token | Cost per conversation |\n|---|---|---|---|---|"
    )
    for model, scenarios in models.items():
        n = sum(s["n"] for s in scenarios.values())
        passed = sum(s["passed"] for s in scenarios.values())
        low, high = wilson(passed, n)
        ttft = statistics.median(t for s in scenarios.values() for t in s["ttfts"])
        usd = sum(s["usd"] for s in scenarios.values()) / n
        print(f"| `{model}` | {passed}/{n} | {low:.0%} to {high:.0%} | {ttft:.2f} s | US$ {usd:.4f} |")


def call(call_id: str) -> None:
    with psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=2, load_balance_hosts="random") as db:
        (transcript,) = db.execute("select transcript from vetvoice.calls where id = %s", (call_id,)).fetchone()
    values = defaultdict(list)
    stopped = None
    for item in transcript["items"]:
        if item.get("type") != "message":
            continue
        metrics = item.get("metrics") or {}
        for key in STAGES:
            if key in metrics:
                values[key].append(metrics[key])
        # What the caller hears: from the end of their speech to the agent's first audio.
        if item["role"] == "user" and "stopped_speaking_at" in metrics:
            stopped = metrics["stopped_speaking_at"]
        elif item["role"] == "assistant" and stopped and "started_speaking_at" in metrics:
            values["total"].append(metrics["started_speaking_at"] - stopped)
            stopped = None
    print(f"{len(values['total'])} replies\n\n| Stage | p50 | max |\n|---|---|---|")
    for key, label in STAGES.items():
        print(f"| {label} | {statistics.median(values[key]):.2f} s | {max(values[key]):.2f} s |")
    total = values["total"]
    print(f"| **Caller stops → agent speaks** | **{statistics.median(total):.2f} s** | **{max(total):.2f} s** |")


if __name__ == "__main__":
    command, *args = sys.argv[1:] or [""]
    if command == "evals":
        evals(args)
    elif command == "call":
        call(*args)
    else:
        sys.exit(__doc__)
