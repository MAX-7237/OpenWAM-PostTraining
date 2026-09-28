#!/usr/bin/env python3
"""Validate the fixed RoboTwin 10-task SFT data and initialization."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import h5py
import numpy as np


TASKS = (
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
)


def _latest_weights(path: Path) -> Path:
    candidates = list(path.glob("checkpoint_step_*.safetensors"))
    if not candidates:
        raise FileNotFoundError(f"missing checkpoint_step_*.safetensors in {path}")
    return max(candidates, key=lambda p: int(p.stem.rsplit("_", 1)[-1]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--stats", required=True)
    parser.add_argument("--wan-path", required=True)
    parser.add_argument("--init", default=None)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--require-gpus", type=int, default=0)
    args = parser.parse_args()

    root = Path(args.dataset_root).resolve()
    manifest_path = Path(args.manifest).resolve()
    stats_path = Path(args.stats).resolve()
    wan_path = Path(args.wan_path).resolve()

    required_wan = (
        "Wan2.2_VAE.pth",
        "diffusion_pytorch_model-00003-of-00003.safetensors",
    )
    for name in required_wan:
        if not (wan_path / name).is_file():
            raise FileNotFoundError(f"incomplete Wan2.2 assets: {wan_path / name}")

    manifest = json.loads(manifest_path.read_text())
    train = manifest.get("train", [])
    for split in ("val", "test_clean_id", "test_random_ood"):
        if manifest.get(split):
            raise ValueError(f"expected empty {split}, got {len(manifest[split])} records")
    if len(train) != 500:
        raise ValueError(f"expected 500 train records, got {len(train)}")

    counts = Counter(item.get("task") for item in train)
    expected_counts = Counter({task: 50 for task in TASKS})
    if counts != expected_counts:
        raise ValueError(f"manifest task counts differ: expected {expected_counts}, got {counts}")

    manifest_files = set()
    for item in train:
        path = Path(item["path"])
        relative = Path(item["task"]) / "aloha-agilex_clean_50" / "data" / path.name
        manifest_files.add(relative.as_posix())

    disk_files = set()
    for task in TASKS:
        task_root = root / task / "aloha-agilex_clean_50" / "data"
        files = sorted(task_root.glob("episode*.hdf5"))
        if len(files) != 50:
            raise ValueError(f"{task}: expected 50 HDF5 episodes, got {len(files)}")
        disk_files.update(path.relative_to(root).as_posix() for path in files)
    if manifest_files != disk_files:
        missing = sorted(manifest_files - disk_files)[:5]
        extra = sorted(disk_files - manifest_files)[:5]
        raise ValueError(f"manifest/disk mismatch: missing={missing}, extra={extra}")

    if not stats_path.is_file():
        raise FileNotFoundError(f"missing normalization stats: {stats_path}")
    stats = np.load(stats_path, allow_pickle=True).item()
    eef = stats.get("eef")
    if not isinstance(eef, dict):
        raise ValueError("stats file has no eef section")
    for key in ("mean", "std", "min", "max", "q01", "q99"):
        if np.asarray(eef.get(key)).shape != (20,):
            raise ValueError(f"eef.{key} must have shape (20,), got {np.asarray(eef.get(key)).shape}")

    if args.init:
        init = Path(args.init).resolve()
        if not (init / "config.yaml").is_file():
            raise FileNotFoundError(f"missing checkpoint config: {init / 'config.yaml'}")
        weights = _latest_weights(init)
        if weights.stat().st_size < 1_000_000_000:
            raise ValueError(f"checkpoint weights look incomplete: {weights} ({weights.stat().st_size} bytes)")
        tokenizer = init / "tokenizer"
        if not tokenizer.is_dir():
            raise FileNotFoundError(f"checkpoint is not self-contained; missing {tokenizer}")

    if args.sample:
        first = root / TASKS[0] / "aloha-agilex_clean_50" / "data" / "episode0.hdf5"
        with h5py.File(first, "r") as handle:
            required = (
                "endpose/left_endpose",
                "endpose/right_endpose",
                "endpose/left_gripper",
                "endpose/right_gripper",
            )
            for key in required:
                if key not in handle:
                    raise KeyError(f"sample is missing {key}: {first}")

    if args.require_gpus:
        import torch

        count = torch.cuda.device_count()
        if count < args.require_gpus:
            raise RuntimeError(f"need {args.require_gpus} visible GPUs, found {count}")

    print(
        "SFT preflight OK: 10 tasks, 500 clean episodes, eef20->unified80, "
        f"stats={stats_path}, init={args.init or 'Wan2.2 base'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
