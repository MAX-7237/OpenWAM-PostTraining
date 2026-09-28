#!/usr/bin/env python3
"""Compute Figure-10 PKU absolute-EEF quantile normalization statistics."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from openwam.dataloader.pku_native30 import (
    ACTION_REPRESENTATION,
    EEF_DIM,
    FRAME_POLICY,
    GRIPPER_CONVENTION,
    STATS_SCHEMA_VERSION,
    _real_eef_rows,
    _real_proprio_rows,
    _sha256,
    _sim_eef_rows,
    _sim_proprio_rows,
)
from openwam.dataloader.utils.normalization import ROT6D_DIMS_EEF20, pin_rot6d_identity


class MaskedAccumulator:
    """Per-dimension online moments and bounded reservoirs for masked EEF rows."""

    def __init__(self, dim: int = EEF_DIM, reservoir_cap: int = 200_000, seed: int = 42):
        self.dim = dim
        self.count = np.zeros(dim, dtype=np.int64)
        self.mean = np.zeros(dim, dtype=np.float64)
        self.m2 = np.zeros(dim, dtype=np.float64)
        self.min = np.full(dim, np.inf, dtype=np.float64)
        self.max = np.full(dim, -np.inf, dtype=np.float64)
        self.cap = int(reservoir_cap)
        self.reservoir = np.empty((self.cap, dim), dtype=np.float32)
        self.reservoir_size = 0
        self.seen = 0
        self.rng = np.random.RandomState(seed)

    def update(self, values: np.ndarray, valid: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float32)
        valid = np.asarray(valid, dtype=bool)
        if values.shape != valid.shape or values.ndim != 2 or values.shape[1] != self.dim:
            raise ValueError(f"expected values/valid [N,{self.dim}], got {values.shape}/{valid.shape}")
        for dim in range(self.dim):
            x = values[valid[:, dim], dim].astype(np.float64)
            if not len(x):
                continue
            n0, n = int(self.count[dim]), len(x)
            batch_mean = float(x.mean())
            batch_m2 = float(x.var()) * n
            if n0 == 0:
                self.mean[dim] = batch_mean
                self.m2[dim] = batch_m2
            else:
                total = n0 + n
                delta = batch_mean - self.mean[dim]
                self.mean[dim] += delta * n / total
                self.m2[dim] += batch_m2 + delta * delta * n0 * n / total
            self.count[dim] += n
            self.min[dim] = min(self.min[dim], float(x.min()))
            self.max[dim] = max(self.max[dim], float(x.max()))
        sampled_rows = values.copy()
        sampled_rows[~valid] = np.nan
        self._reservoir_add(sampled_rows)

    def _reservoir_add(self, values: np.ndarray) -> None:
        size = int(self.reservoir_size)
        if size < self.cap:
            take = min(self.cap - size, len(values))
            self.reservoir[size : size + take] = values[:take]
            self.reservoir_size += take
            self.seen += take
            values = values[take:]
        if not len(values):
            return
        offsets = self.seen + np.arange(len(values), dtype=np.int64)
        positions = self.rng.randint(0, offsets + 1)
        keep = positions < self.cap
        self.reservoir[positions[keep]] = values[keep]
        self.seen += len(values)

    def finalize(self) -> dict:
        stats = {key: np.zeros(self.dim, dtype=np.float32) for key in ("mean", "std", "min", "max", "q01", "q99")}
        for dim in range(self.dim):
            count = int(self.count[dim])
            if count == 0:
                stats["std"][dim] = 1.0
                stats["min"][dim], stats["max"][dim] = -1.0, 1.0
                stats["q01"][dim], stats["q99"][dim] = -1.0, 1.0
                continue
            stats["mean"][dim] = self.mean[dim]
            stats["std"][dim] = max(np.sqrt(self.m2[dim] / count), 1e-8)
            stats["min"][dim], stats["max"][dim] = self.min[dim], self.max[dim]
            sample = self.reservoir[: self.reservoir_size, dim]
            stats["q01"][dim], stats["q99"][dim] = np.nanquantile(sample, (0.01, 0.99))
        pin_rot6d_identity(stats, ROT6D_DIMS_EEF20)
        return stats


def _records(index_path: Path, max_records_per_source: int | None = None) -> list[dict]:
    payload = json.loads(index_path.read_text())
    records = []
    for source in payload["sources"]:
        shard = index_path.parent / source["index_file"]
        if _sha256(shard) != source["sha256"]:
            raise ValueError(f"pointer checksum mismatch: {shard}")
        count = 0
        with gzip.open(shard, "rt", encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                if record.get("split") != "train":
                    continue
                record["_source_domain"] = source["domain"]
                records.append(record)
                count += 1
                if max_records_per_source is not None and count >= max_records_per_source:
                    break
    return records


def _process_data_file(data_path: str, records: list[dict]) -> tuple[np.ndarray, np.ndarray, int]:
    table = pq.read_table(data_path, memory_map=True)
    wanted = sorted({int(record["episode_index"]) for record in records})
    table = table.filter(pc.is_in(table["episode_index"], value_set=pa.array(wanted)))
    try:
        frame = table.to_pandas()
    except TypeError:
        frame = pd.DataFrame(table.to_pylist())
    by_episode = {
        int(key): value.sort_values("frame_index", kind="stable")
        for key, value in frame.groupby("episode_index")
    }
    file_values = []
    file_valid = []
    for record in records:
        win = by_episode[int(record["episode_index"])]
        expected_length = int(record.get("length", record.get("source_num_frames", len(win))))
        if len(win) != expected_length:
            raise ValueError(
                f"{record['source_episode_id']}: index length={expected_length}, parquet rows={len(win)}"
            )
        for start in range(0, len(win), 33):
            chunk = win.iloc[start : start + 33].reset_index(drop=True)
            if record["_source_domain"] == "sim":
                action, action_valid = _sim_eef_rows(chunk, record)
                state, state_valid = _sim_proprio_rows(chunk, record)
            else:
                action, action_valid = _real_eef_rows(chunk, record)
                state, state_valid = _real_proprio_rows(chunk, record)
            file_values.extend((action, state))
            file_valid.extend((action_valid, state_valid))
    return np.concatenate(file_values, axis=0), np.concatenate(file_valid, axis=0), len(records)


def compute(
    index_path: Path,
    *,
    reservoir_cap: int,
    max_records_per_source: int | None = None,
    workers: int = 8,
    executor: str = "thread",
) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in _records(index_path, max_records_per_source):
        if record["_source_domain"] == "ego":
            continue
        grouped[record["data_path"]].append(record)

    accumulator = MaskedAccumulator(reservoir_cap=reservoir_cap)
    processed_records = 0
    entries = list(grouped.items())
    max_pending = max(1, workers * 2)
    executor_cls = ProcessPoolExecutor if executor == "process" else ThreadPoolExecutor
    with executor_cls(max_workers=max(1, workers)) as pool:
        pending = deque()
        next_entry = 0
        while next_entry < len(entries) and len(pending) < max_pending:
            pending.append(pool.submit(_process_data_file, *entries[next_entry]))
            next_entry += 1
        file_number = 0
        while pending:
            values, valid, record_count = pending.popleft().result()
            accumulator.update(values, valid)
            processed_records += record_count
            file_number += 1
            if next_entry < len(entries):
                pending.append(pool.submit(_process_data_file, *entries[next_entry]))
                next_entry += 1
            if file_number % 100 == 0 or file_number == len(entries):
                print(f"[stats] files={file_number}/{len(entries)} episodes={processed_records}", flush=True)

    stats = accumulator.finalize()
    return {
        "num_timesteps": int(accumulator.count.max()),
        "eef": stats,
        "metadata": {
            "schema_version": STATS_SCHEMA_VERSION,
            "index_path": str(index_path.resolve()),
            "index_sha256": _sha256(index_path),
            "action_representation": ACTION_REPRESENTATION,
            "frame_policy": FRAME_POLICY,
            "gripper_convention": GRIPPER_CONVENTION,
            "population": "train action+state rows pooled; masked dimensions excluded",
            "records": processed_records,
            "per_dim_count": accumulator.count,
            "reservoir_cap": reservoir_cap,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reservoir-cap", type=int, default=200_000)
    parser.add_argument("--max-records-per-source", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--executor", choices=("thread", "process"), default="thread")
    args = parser.parse_args()
    payload = compute(
        args.index.resolve(),
        reservoir_cap=args.reservoir_cap,
        max_records_per_source=args.max_records_per_source,
        workers=args.workers,
        executor=args.executor,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp.npy")
    np.save(tmp, payload, allow_pickle=True)
    tmp.replace(args.output)
    print(f"[stats] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
