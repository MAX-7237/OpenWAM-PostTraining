"""EgoDex LeRobot v3 reader.

EgoDex is a human bimanual source.  Its 22-D command is
``left EEF(10) + left activity + right EEF(10) + right activity``.  The
activity labels are annotation fields and are intentionally excluded from the
control head; the remaining 20 dimensions use the same EEF layout as the
robot readers.
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

from openwam.dataloader.bases import LeRobotV3Reader
from openwam.dataloader.utils.lerobotv3 import apply_info_splits, compute_file_local_offsets


_ACTION_COL = "action"
_STATE_COL = "action"
_NEEDED_COLS = ("task_index", "timestamp", "frame_index", "episode_index", _ACTION_COL)


def _gripper01(values: np.ndarray) -> np.ndarray:
    """Map EgoDex aperture in metres to 0=closed, 1=open.

    The released retargeted EgoDex data uses the calibrated hand aperture
    interval [0.04, 0.10] m.  Values outside this interval are clipped after
    conversion so isolated hand-tracker overshoot cannot create invalid
    control targets.
    """

    values = np.asarray(values, dtype=np.float32)
    return np.clip((values - 0.04) / 0.06, 0.0, 1.0).astype(np.float32)


class EgoDexDataset(LeRobotV3Reader):
    DATASET_NAME = "EgoDex"
    NEEDED_COLS = _NEEDED_COLS
    ACTION_DIM = 20
    PROMPT_FILE_REQUIRED = True
    DEFAULT_NORMALIZE_MODE = None
    STATS_FILENAME = None
    CONFIG_KEYS = LeRobotV3Reader.CONFIG_KEYS + ("invalid_episodes_path",)

    def __init__(self, *args, invalid_episodes_path=None, **kwargs):
        self._invalid_episodes_path = Path(invalid_episodes_path) if invalid_episodes_path else None
        super().__init__(*args, **kwargs)

    def _load_excluded_episode_indices(self) -> set[int]:
        excluded = set(super()._load_excluded_episode_indices())
        if self._invalid_episodes_path is not None and self._invalid_episodes_path.exists():
            payload = json.loads(self._invalid_episodes_path.read_text())
            values = payload.get("episode_indices", payload if isinstance(payload, list) else [])
            excluded.update(int(v) for v in values)
        return excluded

    @staticmethod
    def _physical_data_file_index(logical_index: int) -> int:
        """Translate the stale EgoDex metadata index to the physical shard.

        The released retargeted copy has 500 physical parquet files, while its
        episodes table retained an intermediate numbering with a +4 offset.
        """
        physical = int(logical_index) - 4
        if physical < 0:
            raise ValueError(f"EgoDex invalid data/file_index={logical_index}")
        return physical

    def _video_file_index_map(self) -> dict[int, int]:
        if self._invalid_episodes_path is None or not self._invalid_episodes_path.exists():
            raise FileNotFoundError(
                "EgoDex requires the audited video_file_map; run "
                "scripts/scjd/audit_egodex_videos.py and pass invalid_episodes_path"
            )
        payload = json.loads(self._invalid_episodes_path.read_text())
        mapping = payload.get("video_file_map", {})
        if not mapping:
            raise ValueError("EgoDex audit manifest has no video_file_map")
        return {int(k): int(v) for k, v in mapping.items()}

    def _resolve_cameras(self, info):
        features = info.get("features", {})
        if "observation.images.front_1" not in features:
            raise ValueError(f"{self.DATASET_NAME}({self._dataset_id}): missing front_1 camera")
        return "observation.images.front_1", None, None

    def _build_episode_index(self, info):
        """Load EgoDex episodes without pandas extension-type conversion.

        Some EgoDex shards encode ``tasks`` as an Arrow list extension.  The
        generic helper intentionally uses ``to_pandas`` for ordinary LeRobot
        tables, but that conversion fails with older pandas/pyarrow pairs.
        ``to_pylist`` is lossless for the small episode metadata tables and
        keeps the list-valued task column available for prompt resolution.
        """
        paths = sorted((self._dataset_dir / "meta" / "episodes").rglob("*.parquet"))
        if not paths:
            raise FileNotFoundError(f"No episode parquet under {self._dataset_dir}/meta/episodes")
        rows = []
        for path in paths:
            table = pq.read_table(path)
            keep = [name for name in table.column_names if not name.startswith("stats/")]
            rows.extend(pa.Table.from_arrays([table[name] for name in keep], names=keep).to_pylist())
        eps = pd.DataFrame(rows).sort_values("episode_index").reset_index(drop=True)
        eps["data/file_index"] = eps["data/file_index"].map(self._physical_data_file_index)
        eps["_data_row_offset"] = compute_file_local_offsets(eps, "data/chunk_index", "data/file_index")
        video_file_map = self._video_file_index_map()
        for camera in self._video_cameras():
            chunk_col = f"videos/{camera}/chunk_index"
            file_col = f"videos/{camera}/file_index"
            if chunk_col in eps.columns and file_col in eps.columns:
                missing = sorted(set(int(v) for v in eps[file_col].unique()) - set(video_file_map))
                if missing:
                    raise ValueError(f"EgoDex audit manifest is missing video mappings: {missing[:8]}")
                eps[file_col] = eps[file_col].map(video_file_map)
                from_col = f"videos/{camera}/from_timestamp"
                if from_col not in eps.columns:
                    raise ValueError(f"EgoDex episodes table is missing {from_col}")
                camera_info = info["features"][camera].get("info", {})
                video_fps = float(camera_info.get("video.fps", info["fps"]))
                offsets = np.rint(eps[from_col].to_numpy(dtype=np.float64) * video_fps).astype(np.int64)
                if np.any(offsets < 0):
                    raise ValueError(f"EgoDex {camera} has negative video timestamps")
                eps[self._video_offset_col(camera)] = offsets
        return apply_info_splits(eps, self._split, info.get("splits", {}) or {}, source_name=self.DATASET_NAME)

    @staticmethod
    def _eef20(raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=np.float32)
        if raw.ndim != 2 or raw.shape[1] != 22:
            raise ValueError(f"EgoDex action must have shape [N,22], got {raw.shape}")
        out = np.empty((len(raw), 20), dtype=np.float32)
        out[:, :9] = raw[:, :9]
        out[:, 9] = _gripper01(raw[:, 9])
        out[:, 10:19] = raw[:, 11:20]
        out[:, 19] = _gripper01(raw[:, 20])
        if not np.isfinite(out).all():
            raise ValueError("EgoDex action contains non-finite values")
        return out

    def _action_20d(self, win):
        raw = np.stack(win[_ACTION_COL].values).astype(np.float32)
        return self._eef20(raw)

    def _proprio_20d(self, win):
        # EgoDex does not publish a robot-state EEF stream.  The current human
        # hand command is the only aligned 20-D control state; retaining it
        # keeps the field available while the source provenance remains clear.
        raw = np.stack(win[_STATE_COL].values[:1]).astype(np.float32)
        return self._eef20(raw)

    @classmethod
    def _multibucket_wrapper(cls):
        return None


__all__ = ["EgoDexDataset"]
