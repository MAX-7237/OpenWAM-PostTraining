#!/usr/bin/env python3
"""Plot loss curves for the four official RobotWin10 SFT runs."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "logs" / "sft_robotwin10" / "plots"
TRAIN_RE = re.compile(
    r"\[train\]\s+step=(?P<step>\d+)/(?P<total>\d+)\s+"
    r"opt_step=(?P<opt_step>\d+)\s+epoch=(?P<epoch>\d+)\s+"
    r"loss=(?P<loss>[-+0-9.eE]+)\s+"
    r"video=(?P<video>[-+0-9.eE]+)\s+"
    r"action=(?P<action>[-+0-9.eE]+)\s+"
    r"grad_norm=(?P<grad_norm>[-+0-9.eE]+)\s+"
    r"lr=(?P<lr>[-+0-9.eE]+)\s+"
    r"steps_per_sec=(?P<steps_per_sec>[-+0-9.eE]+)"
)

RUNS = [
    (
        "From Scratch",
        "openwam_official_from_scratch_robotwin10tasks_sft-copy2",
        ROOT
        / "logs/sft_robotwin10/sft-robotwin10-from-scratch-8g-bs16-30k-20260927-retry/train.log",
        "#1f77b4",
    ),
    (
        "Robot Only",
        "openwam_official_robot_only_robotwin10tasks_sft-copy3",
        ROOT
        / "logs/sft_robotwin10/sft-robotwin10-official-robot-only-8g-bs16-30k-20260927-retry/train.log",
        "#d62728",
    ),
    (
        "Ego -> Robot",
        "openwam_official_ego2robot_robotwin10tasks_sft-copy4",
        ROOT
        / "logs/sft_robotwin10/sft-robotwin10-official-ego2robot-8g-bs16-30k-20260927/train.log",
        "#2ca02c",
    ),
    (
        "Ego + Robot",
        "openwam_official_ego_plus_robot_robotwin10tasks_sft-copy1",
        ROOT
        / "logs/sft_robotwin10/sft-robotwin10-official-ego-robot-mutual-8g-bs16-30k-20260927-retry/train.log",
        "#9467bd",
    ),
]


def parse_log(path: Path) -> list[dict[str, float]]:
    points: list[dict[str, float]] = []
    for line in path.read_text(errors="replace").splitlines():
        match = TRAIN_RE.search(line)
        if not match:
            continue
        row = {key: float(value) for key, value in match.groupdict().items()}
        row["step"] = int(row["step"])
        row["opt_step"] = int(row["opt_step"])
        row["epoch"] = int(row["epoch"])
        points.append(row)
    if not points:
        raise RuntimeError(f"No train loss records found in {path}")
    return points


def smooth(values: np.ndarray, window: int = 31) -> np.ndarray:
    if len(values) < 3:
        return values
    window = min(window, len(values))
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return values
    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(values, (window // 2,), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = []
    all_rows = []
    summary = {}
    for label, experiment, path, color in RUNS:
        if not path.is_file():
            raise FileNotFoundError(path)
        rows = parse_log(path)
        datasets.append((label, experiment, color, rows))
        for row in rows:
            all_rows.append({"experiment": experiment, **row})
        losses = np.array([row["loss"] for row in rows])
        tail = losses[-50:]
        summary[experiment] = {
            "label": label,
            "log": str(path),
            "points": len(rows),
            "first_step": rows[0]["step"],
            "last_step": rows[-1]["step"],
            "first_loss": float(losses[0]),
            "final_loss": float(losses[-1]),
            "min_loss": float(losses.min()),
            "mean_last_50_loss": float(tail.mean()),
            "final_video_loss": float(rows[-1]["video"]),
            "final_action_loss": float(rows[-1]["action"]),
        }

    with (OUT_DIR / "loss_records.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "experiment",
                "step",
                "total",
                "opt_step",
                "epoch",
                "loss",
                "video",
                "action",
                "grad_norm",
                "lr",
                "steps_per_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(12, 7), dpi=160)
    for label, _experiment, color, rows in datasets:
        steps = np.array([row["step"] for row in rows])
        losses = np.array([row["loss"] for row in rows])
        ax.plot(steps, losses, color=color, alpha=0.14, linewidth=0.8)
        ax.plot(steps, smooth(losses), color=color, linewidth=2.0, label=label)
    ax.set_title("OpenWAM Official RobotWin10 SFT: Total Loss")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.legend(frameon=True)
    ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "official_sft_total_loss.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=160, sharex=True)
    for ax, (label, _experiment, color, rows) in zip(axes.flat, datasets):
        steps = np.array([row["step"] for row in rows])
        for key, name, line_color in (
            ("loss", "total", color),
            ("video", "video", "#555555"),
            ("action", "action", "#ff7f0e"),
        ):
            values = np.array([row[key] for row in rows])
            ax.plot(steps, values, color=line_color, alpha=0.12, linewidth=0.7)
            ax.plot(steps, smooth(values), color=line_color, linewidth=1.7, label=name)
        ax.set_title(label)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Loss")
        ax.set_xlim(left=0)
        ax.legend(fontsize=8)
    fig.suptitle("OpenWAM Official RobotWin10 SFT: Loss Components", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "official_sft_loss_components.png", bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote plots and metrics to {OUT_DIR}")
    for experiment, values in summary.items():
        print(
            f"{values['label']}: points={values['points']} "
            f"steps={values['first_step']}..{values['last_step']} "
            f"final_loss={values['final_loss']:.6f} "
            f"last50={values['mean_last_50_loss']:.6f}"
        )


if __name__ == "__main__":
    main()
