#!/usr/bin/env python3
"""Merge final per-task clean and randomized RoboTwin success rates."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

TASKS = [
    "adjust_bottle",
    "handover_block",
    "lift_pot",
    "move_can_pot",
    "open_laptop",
    "place_can_basket",
    "place_empty_cup",
    "press_stapler",
    "shake_bottle",
    "stack_blocks_two",
]
RATE_RE = re.compile(r"Success rate:\s*(\d+)\s*/\s*(\d+)\s*=>\s*([0-9.]+)%")
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def read_rates(root: Path) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    missing: list[str] = []
    for task in TASKS:
        path = root / f"{task}.log"
        text = path.read_text(errors="replace") if path.is_file() else ""
        matches = RATE_RE.findall(ANSI_RE.sub("", text))
        if not matches:
            missing.append(task)
            continue
        success, total, rate = matches[-1]
        result[task] = {"success": int(success), "total": int(total), "sr": float(rate)}
    if missing:
        raise SystemExit(f"missing final Success rate for: {', '.join(missing)} under {root}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--randomized", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    clean = read_rates(args.clean)
    randomized = read_rates(args.randomized)
    combined = {
        task: {"clean": clean[task], "randomized": randomized[task]}
        for task in TASKS
    }
    clean_avg = sum(float(clean[t]["sr"]) for t in TASKS) / len(TASKS)
    randomized_avg = sum(float(randomized[t]["sr"]) for t in TASKS) / len(TASKS)
    payload = {
        "tasks": TASKS,
        "clean": clean,
        "randomized": randomized,
        "macro_average_sr": {"clean": round(clean_avg, 4), "randomized": round(randomized_avg, 4)},
        "micro_average_sr": {
            "clean": round(100 * sum(int(clean[t]["success"]) for t in TASKS) / sum(int(clean[t]["total"]) for t in TASKS), 4),
            "randomized": round(100 * sum(int(randomized[t]["success"]) for t in TASKS) / sum(int(randomized[t]["total"]) for t in TASKS), 4),
        },
        "combined": combined,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "combined_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (args.output / "combined_results.tsv").open("w") as handle:
        handle.write("task\tclean_sr\trandomized_sr\tclean_success_total\trandomized_success_total\n")
        for task in TASKS:
            c, r = clean[task], randomized[task]
            handle.write(f"{task}\t{c['sr']:.1f}\t{r['sr']:.1f}\t{c['success']}/{c['total']}\t{r['success']}/{r['total']}\n")
        handle.write(f"macro_average\t{clean_avg:.2f}\t{randomized_avg:.2f}\t-\t-\n")
        handle.write(f"micro_average\t{payload['micro_average_sr']['clean']:.2f}\t{payload['micro_average_sr']['randomized']:.2f}\t-\t-\n")
    print(f"[INFO] merged results: {args.output / 'combined_results.tsv'}")
    print(f"[RESULT] macro SR clean={clean_avg:.2f}% randomized={randomized_avg:.2f}%")


if __name__ == "__main__":
    main()
