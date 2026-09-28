"""OpenWAM reader for audited ``pku_native30_v1`` pointer indexes.

The pointer index selects whole episodes without copying payloads.  This
reader expands them into OpenWAM's frame-level training windows, decodes the
official 33-frame/stride-4 visual clip, and maps bimanual EEF targets into the
shared 80-D action space.  Human ego clips are deliberately video-only, as in
the Figure 10 protocol: their action and proprioception masks are all false.
"""

from __future__ import annotations

import functools
import gzip
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
import yaml

from openwam.dataloader.bases import BaseDataset
from openwam.dataloader.mixture import MixtureDataset
from openwam.dataloader.transforms.multiview import assemble_multiview_layout
from openwam.dataloader.transforms.video import VideoColorJitter, color_jitter_enabled
from openwam.dataloader.utils.eef import quat_wxyz_to_rot6d
from openwam.dataloader.utils.normalization import apply_normalization, materialize_eef_stats
from openwam.dataloader.utils.video_io import decode_video_frames


INDEX_VERSION = "pku_native30_v1"
ACTION_DIM = 80
EEF_DIM = 20
LEFT_DST = slice(0, 10)
RIGHT_DST = slice(34, 44)

# Native aperture conventions, converted below to 0=closed and 1=open.
# AgileX stores the full finger separation in metres (the v3 URDF viewer uses
# half of this value per finger). TienKung's selected 0/1 series has the
# opposite convention: 0=open, 1=closed.
AGILEX_GRIPPER_CLOSED_M = 0.0
AGILEX_GRIPPER_OPEN_M = 0.1
STATS_SCHEMA_VERSION = "pku_native30_eef_stats_v2"
ACTION_REPRESENTATION = "absolute_source_frame_xyz_rot6d_gripper"
FRAME_POLICY = "base_for_base_ee;camera_for_ur_camera_ee"
GRIPPER_CONVENTION = "zero_closed_one_open"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_normalization_stats(
    path: str | Path,
    *,
    index_path: Path,
    normalize_mode: str | None,
) -> tuple[dict | None, dict | None]:
    if not normalize_mode or normalize_mode in ("none", "null"):
        return None, None
    stats_path = Path(path).expanduser().resolve()
    if not stats_path.is_file():
        raise FileNotFoundError(
            f"PKU normalize_mode={normalize_mode!r} requires stats: {stats_path}. "
            "Run pku_native30_stats_computation for the configured robot index."
        )
    payload = np.load(stats_path, allow_pickle=True).item()
    metadata = payload.get("metadata", {})
    expected = {
        "schema_version": STATS_SCHEMA_VERSION,
        "index_sha256": _sha256(index_path),
        "action_representation": ACTION_REPRESENTATION,
        "frame_policy": FRAME_POLICY,
        "gripper_convention": GRIPPER_CONVENTION,
    }
    actual = {key: metadata.get(key) for key in expected}
    if actual != expected:
        raise ValueError(
            f"PKU normalization stats provenance mismatch for {stats_path}: "
            f"expected {expected}, got {actual}. Regenerate stats for {index_path}."
        )
    action_stats = materialize_eef_stats(
        payload.get("eef", {}),
        normalize_mode,
        dim=EEF_DIM,
        strict_minmax=False,
        source_hint=f"{stats_path}:eef",
        force_rot6d_identity=True,
    )
    state_stats = materialize_eef_stats(
        payload.get("eef_state", payload.get("eef", {})),
        normalize_mode,
        dim=EEF_DIM,
        strict_minmax=False,
        source_hint=f"{stats_path}:eef_state",
        force_rot6d_identity=True,
    )
    return action_stats, state_stats


@functools.lru_cache(maxsize=16)
def _episode_table(path: str, episode_index: int) -> pd.DataFrame:
    table = pq.read_table(path, memory_map=True)
    if "episode_index" not in table.column_names:
        raise ValueError(f"{path}: missing episode_index")
    table = table.filter(pc.equal(table["episode_index"], int(episode_index)))
    if table.num_rows == 0:
        raise ValueError(f"{path}: episode_index={episode_index} has no rows")
    frame_index = table["frame_index"].to_numpy() if "frame_index" in table.column_names else None
    if frame_index is not None:
        table = table.take(np.argsort(frame_index, kind="stable"))
    try:
        return table.to_pandas()
    except TypeError:
        return pd.DataFrame(table.to_pylist())


@functools.lru_cache(maxsize=8192)
def _camera_schema(task_dir: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    info = json.loads((Path(task_dir) / "meta" / "info.json").read_text())
    names = info["features"]["action"]["names"]
    if names and isinstance(names[0], list):
        names = names[0]
    prefix = "side_view" if names[0].startswith("side_view.") else "camera_top"
    components = (
        ["ee.position." + axis for axis in "xyz"]
        + [f"ee.rotation_6d.column_{i}.{axis}" for i in range(2) for axis in "xyz"]
        + ["gripper"]
    )
    camera = [f"{prefix}.{side}.{name}" for side in ("left", "right") for name in components]
    base = [f"base.{side}.{name}" for side in ("left", "right") for name in components]
    if not all(name in names for name in camera + base):
        raise ValueError(f"{task_dir}: processed camera/base EEF fields are incomplete")
    return tuple(names.index(name) for name in camera), tuple(names.index(name) for name in base)


@functools.lru_cache(maxsize=64)
def _feature_names(task_dir: str, feature: str) -> tuple[str, ...]:
    info = json.loads((Path(task_dir) / "meta" / "info.json").read_text())
    names = info["features"][feature].get("names", ())
    if names and isinstance(names[0], list):
        names = names[0]
    return tuple(str(name) for name in names)


def _normalize_gripper(values: np.ndarray, record: dict) -> np.ndarray:
    """Map a source gripper signal to aperture fraction (0 closed, 1 open)."""

    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{record['source_episode_id']}: non-finite gripper value")
    source = str(record.get("source_name", "")).lower()
    if source == "agilex":
        values = (values - AGILEX_GRIPPER_CLOSED_M) / (
            AGILEX_GRIPPER_OPEN_M - AGILEX_GRIPPER_CLOSED_M
        )
    elif source == "tienkung":
        values = 1.0 - values
    # UR, RealSource, and the generated InternData action feature already use
    # the canonical 0=closed, 1=open convention.
    return np.clip(values, 0.0, 1.0).astype(np.float32)


def _rotation_from_6d(values: np.ndarray) -> np.ndarray:
    cols = values.reshape(*values.shape[:-1], 2, 3).swapaxes(-1, -2)
    return np.concatenate([cols, np.cross(cols[..., 0], cols[..., 1])[..., None]], axis=-1)


def _absolute_eef_targets(poses: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the 32 future absolute EEF targets used by official RoboCOIN."""

    poses = np.asarray(poses, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if poses.shape != (33, EEF_DIM) or valid.shape != poses.shape:
        raise ValueError(f"expected poses/valid [33,20], got {poses.shape}/{valid.shape}")
    action = np.zeros((32, EEF_DIM), dtype=np.float32)
    dim_mask = np.zeros(EEF_DIM, dtype=bool)
    for offset in (0, 10):
        if not valid[:, offset : offset + 9].all():
            continue
        rotation = _rotation_from_6d(poses[:, offset + 3 : offset + 9])
        if not np.isfinite(poses[:, offset : offset + 3]).all() or not np.isfinite(rotation).all():
            raise ValueError("non-finite EEF pose")
        if not np.allclose(rotation.swapaxes(1, 2) @ rotation, np.eye(3), atol=1e-4):
            raise ValueError("invalid EEF rot6d geometry")
        action[:, offset : offset + 9] = poses[1:, offset : offset + 9].astype(np.float32)
        dim_mask[offset : offset + 9] = True
        if valid[:, offset + 9].all():
            grip = poses[1:, offset + 9]
            if not np.isfinite(grip).all() or np.any((grip < 0.0) | (grip > 1.0)):
                raise ValueError("normalized gripper target is outside [0,1]")
            action[:, offset + 9] = grip.astype(np.float32)
            dim_mask[offset + 9] = True
    return action, dim_mask


def _pad_window(win: pd.DataFrame, count: int = 33) -> pd.DataFrame:
    if len(win) == 0:
        raise ValueError("empty parquet window")
    if len(win) >= count:
        return win.iloc[:count].reset_index(drop=True)
    rows = [win, pd.concat([win.iloc[[-1]]] * (count - len(win)), ignore_index=True)]
    return pd.concat(rows, ignore_index=True)


def _real_eef_rows(win: pd.DataFrame, record: dict) -> tuple[np.ndarray, np.ndarray]:
    raw = np.stack(win["action"].values).astype(np.float64)
    poses = np.zeros((len(win), EEF_DIM), dtype=np.float64)
    valid = np.zeros_like(poses, dtype=bool)
    if record.get("ur_camera_ee"):
        indices = list(record["ee_indices"])
        poses[:, : len(indices)] = raw[:, indices]
        for arm, offset in enumerate((0, 10)):
            if record["ee_arm_available"][arm]:
                valid[:, offset : offset + 9] = np.isfinite(poses[:, offset : offset + 9])
                poses[:, offset + 9] = _normalize_gripper(poses[:, offset + 9], record)
                valid[:, offset + 9] = True
        return np.nan_to_num(poses).astype(np.float32), valid
    if not record.get("camera_ee"):
        raise ValueError(f"{record['source_episode_id']}: no admitted real-robot EEF target")
    camera_indices, base_indices = _camera_schema(record["task_dir"])
    camera = raw[:, camera_indices]
    base = raw[:, base_indices]
    for offset in (0, 10):
        cam = camera[:, offset : offset + 9]
        robot = base[:, offset : offset + 9]
        if not np.isfinite(cam).all() or not np.isfinite(robot).all():
            continue
        r_cam, r_robot = _rotation_from_6d(cam[:, 3:]), _rotation_from_6d(robot[:, 3:])
        extrinsic_r = r_robot @ r_cam.swapaxes(1, 2)
        extrinsic_p = robot[:, :3] - np.einsum("nij,nj->ni", extrinsic_r, cam[:, :3])
        if not np.allclose(extrinsic_r, extrinsic_r[:1], atol=2e-4):
            continue
        if not np.allclose(extrinsic_p, extrinsic_p[:1], atol=2e-4):
            continue
        # Relative targets were frame-invariant, but official RoboCOIN absolute
        # EEF supervision requires the unified robot-base coordinates.
        poses[:, offset : offset + 9] = robot
        valid[:, offset : offset + 9] = True
        poses[:, offset + 9] = _normalize_gripper(base[:, offset + 9], record)
        valid[:, offset + 9] = True
    return poses.astype(np.float32), valid


def _real_eef(win: pd.DataFrame, record: dict) -> tuple[np.ndarray, np.ndarray]:
    poses, valid = _real_eef_rows(win, record)
    return _absolute_eef_targets(poses, valid)


def _sim_arm(win: pd.DataFrame, pose_col: str) -> np.ndarray:
    pose = np.stack(win[pose_col].values).astype(np.float32)
    if pose.ndim != 2 or pose.shape[1] != 7:
        raise ValueError(f"{pose_col}: expected [N,7], got {pose.shape}")
    return np.concatenate([pose[:, :3], quat_wxyz_to_rot6d(pose[:, 3:7])], axis=-1)


def _normalized_action_grippers(win: pd.DataFrame, record: dict) -> tuple[np.ndarray, np.ndarray]:
    raw = np.stack(win["action"].values).astype(np.float64)
    names = _feature_names(record["task_dir"], "action")
    indices = [i for i, name in enumerate(names) if "gripper" in name.lower() and "0_1" in name.lower()]
    if not indices:
        indices = [i for i, name in enumerate(names) if "gripper" in name.lower()]
    if len(indices) not in (1, 2):
        raise ValueError(f"{record['task_dir']}: expected one or two normalized action grippers")
    left = _normalize_gripper(raw[:, indices[0]], record)
    right = _normalize_gripper(raw[:, indices[1]], record) if len(indices) == 2 else np.zeros(len(win), np.float32)
    return left, right


def _sim_eef_rows(win: pd.DataFrame, record: dict) -> tuple[np.ndarray, np.ndarray]:
    poses = np.zeros((len(win), EEF_DIM), dtype=np.float32)
    valid = np.zeros_like(poses, dtype=bool)
    left_grip, right_grip = _normalized_action_grippers(win, record)
    if "actions.ee_to_robot_pose" in win:
        poses[:, :9] = _sim_arm(win, "actions.ee_to_robot_pose")
        valid[:, :9] = True
        poses[:, 9] = left_grip
        valid[:, 9] = True
    else:
        for side, offset in (("left", 0), ("right", 10)):
            column = f"actions.{side}_ee_to_robot_pose"
            if column in win:
                poses[:, offset : offset + 9] = _sim_arm(win, column)
                valid[:, offset : offset + 9] = True
                poses[:, offset + 9] = left_grip if side == "left" else right_grip
                valid[:, offset + 9] = True
    return poses, valid


def _sim_eef(win: pd.DataFrame, record: dict) -> tuple[np.ndarray, np.ndarray]:
    poses, valid = _sim_eef_rows(win, record)
    return _absolute_eef_targets(poses, valid)


def _pose7_to_arm10(
    pose: np.ndarray,
    grip: float,
    *,
    quaternion: str,
    tip_offset_xyz: tuple[float, float, float] | None = None,
) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float32).reshape(7)
    if quaternion == "xyzw":
        quat = pose[[6, 3, 4, 5]]
    elif quaternion == "wxyz":
        quat = pose[3:7]
    else:
        raise ValueError(f"unsupported quaternion convention: {quaternion}")
    rot6d = quat_wxyz_to_rot6d(quat[None])[0]
    position = pose[:3].copy()
    if tip_offset_xyz is not None:
        rotation = _rotation_from_6d(rot6d[None].astype(np.float64))[0]
        position += (rotation @ np.asarray(tip_offset_xyz, dtype=np.float64)).astype(np.float32)
    return np.concatenate([position, rot6d, np.asarray([grip], np.float32)])


def _pose7_to_arm10_rows(
    poses: np.ndarray,
    grips: np.ndarray,
    *,
    quaternion: str,
    tip_offset_xyz: tuple[float, float, float] | None = None,
) -> np.ndarray:
    poses = np.asarray(poses, dtype=np.float32).reshape(-1, 7)
    grips = np.asarray(grips, dtype=np.float32).reshape(-1, 1)
    if len(poses) != len(grips):
        raise ValueError(f"pose/gripper row mismatch: {len(poses)} != {len(grips)}")
    if quaternion == "xyzw":
        quats = poses[:, [6, 3, 4, 5]]
    elif quaternion == "wxyz":
        quats = poses[:, 3:7]
    else:
        raise ValueError(f"unsupported quaternion convention: {quaternion}")
    rot6d = quat_wxyz_to_rot6d(quats)
    positions = poses[:, :3].copy()
    if tip_offset_xyz is not None:
        rotations = _rotation_from_6d(rot6d.astype(np.float64))
        positions += np.einsum(
            "nij,j->ni", rotations, np.asarray(tip_offset_xyz, dtype=np.float64)
        ).astype(np.float32)
    return np.concatenate([positions, rot6d, grips], axis=-1).astype(np.float32)


def _resolve_metadata_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_file():
        return candidate
    text = str(candidate)
    aliases = (
        ("/mnt/world_foundational/datasets/pku-data", "/media/damoxing/datasets/pku-data"),
        ("/media/damoxing/datasets/pku-data", "/mnt/world_foundational/datasets/pku-data"),
    )
    for source, target in aliases:
        if text.startswith(source):
            alternate = Path(target + text[len(source) :])
            if alternate.is_file():
                return alternate
    return candidate


@functools.lru_cache(maxsize=256)
def _ur_camera_extrinsics(task_dir: str) -> dict[str, np.ndarray]:
    """Load the per-arm world-to-camera matrices recorded by UR conversion."""

    info = json.loads((Path(task_dir) / "meta" / "info.json").read_text())
    schema = info.get("metadata.action_ee_schema") or info.get("metadata.action17_schema") or {}
    per_arm = schema.get("per_arm") or {}
    result: dict[str, np.ndarray] = {}
    for side, spec in per_arm.items():
        if not isinstance(spec, dict) or not spec.get("calibration_path"):
            continue
        calibration_path = _resolve_metadata_path(spec["calibration_path"])
        payload = yaml.safe_load(calibration_path.read_text()) or {}
        cameras = payload.get("cameras") or {}
        if not cameras:
            raise ValueError(f"{task_dir}: UR calibration has no cameras: {calibration_path}")
        camera = next(iter(cameras.values()))
        matrix = np.asarray(camera.get("extrinsic_world_to_camera"), dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
            raise ValueError(f"{task_dir}: invalid UR world_to_camera matrix: {calibration_path}")
        result[str(side)] = matrix
    if not result:
        raise ValueError(f"{task_dir}: UR camera EEF has no per-arm calibration")
    return result


def _pose7_to_camera_arm10_rows(
    poses: np.ndarray, grips: np.ndarray, world_to_camera: np.ndarray
) -> np.ndarray:
    """Convert native world/base-frame xyzw poses into UR camera-frame EEF10."""

    poses = np.asarray(poses, dtype=np.float32).reshape(-1, 7)
    grips = np.asarray(grips, dtype=np.float32).reshape(-1, 1)
    rotation = _rotation_from_6d(quat_wxyz_to_rot6d(poses[:, [6, 3, 4, 5]]).reshape(-1, 6))
    extrinsic = np.asarray(world_to_camera, dtype=np.float64)
    rotation = np.einsum("ij,njk->nik", extrinsic[:3, :3], rotation)
    position = np.einsum("ij,nj->ni", extrinsic[:3, :3], poses[:, :3]) + extrinsic[:3, 3]
    rot6d = np.concatenate([rotation[:, :, 0], rotation[:, :, 1]], axis=-1)
    return np.concatenate([position, rot6d, grips], axis=-1).astype(np.float32)


def _real_proprio_rows(win: pd.DataFrame, record: dict) -> tuple[np.ndarray, np.ndarray]:
    state = np.zeros((len(win), EEF_DIM), dtype=np.float32)
    valid = np.zeros_like(state, dtype=bool)
    source = str(record.get("source_name", "")).lower()

    if record.get("ur_camera_ee"):
        extrinsics = _ur_camera_extrinsics(record["task_dir"])
        specs = [("single", 0)] if "puppet.end_effector_single_pose_align.data" in win else [("left", 0), ("right", 10)]
        for side, offset in specs:
            pose_col = f"puppet.end_effector_{side}_pose_align.data"
            grip_col = f"puppet.end_effector_{side}_position_align.data"
            if pose_col not in win or grip_col not in win or side not in extrinsics:
                continue
            raw_grip = np.stack(win[grip_col].values).astype(np.float32).reshape(len(win), -1)[:, 0]
            grip = _normalize_gripper(raw_grip, record)
            state[:, offset : offset + 10] = _pose7_to_camera_arm10_rows(
                np.stack(win[pose_col].values), grip, extrinsics[side]
            )
            valid[:, offset : offset + 10] = True
        return state, valid

    if source == "realsource":
        raw = np.stack(win["observation.state"].values).astype(np.float32)
        names = _feature_names(record["task_dir"], "observation.state")
        for side, offset, prefix in (("left", 0, "Left"), ("right", 10, "Right")):
            pose_names = [f"{prefix}End_{axis}" for axis in ("X", "Y", "Z", "Qw", "Qx", "Qy", "Qz")]
            grip_name = f"{prefix}Gripper.pos"
            if not all(name in names for name in pose_names + [grip_name]):
                continue
            pose = raw[:, [names.index(name) for name in pose_names]]
            grip = _normalize_gripper(raw[:, names.index(grip_name)], record)
            state[:, offset : offset + 10] = _pose7_to_arm10_rows(pose, grip, quaternion="wxyz")
            valid[:, offset : offset + 10] = True
        return state, valid

    for side, offset in (("left", 0), ("right", 10)):
        grip_col = f"puppet.end_effector_{side}_position_align.data"
        if grip_col in win:
            raw_grip = np.stack(win[grip_col].values).astype(np.float32).reshape(len(win), -1)[:, 0]
            state[:, offset + 9] = _normalize_gripper(raw_grip, record)
            valid[:, offset + 9] = True
        pose_col = f"puppet.end_effector_{side}_pose_align.data"
        if pose_col in win and valid[:, offset + 9].all():
            tip_offset = (0.001, 0.0, 0.14) if source == "agilex" else None
            state[:, offset : offset + 10] = _pose7_to_arm10_rows(
                np.stack(win[pose_col].values),
                state[:, offset + 9],
                quaternion="xyzw",
                tip_offset_xyz=tip_offset,
            )
            valid[:, offset : offset + 10] = True
    return state, valid


def _real_proprio(win: pd.DataFrame, record: dict) -> tuple[np.ndarray, np.ndarray]:
    state, valid = _real_proprio_rows(win.iloc[:1], record)
    return state[0], valid[0]


def _sim_state_gripper_rows(
    win: pd.DataFrame,
    record: dict,
    side: str | None,
    normalized_action: np.ndarray,
) -> np.ndarray:
    prefix = f"{side}_" if side else ""
    action_col = f"actions.{prefix}gripper.position"
    state_col = f"states.{prefix}gripper.position"
    raw_action = np.stack(win[action_col].values).astype(np.float64).reshape(-1)
    raw_state = np.stack(win[state_col].values).astype(np.float64).reshape(-1)
    source = str(record.get("source_name", "")).lower()
    candidates = {
        "interndata_sim_franka": (1.0, 0.08),
        "interndata_sim_lift2": (0.088, 1.0),
        "interndata_sim_genie1": (1.0, 5.74),
        "interndata_sim_aloha": (0.1, 1.0),
    }.get(source, (1.0,))

    def alignment_error(raw: np.ndarray, scale: float) -> float:
        scaled = np.clip(raw / scale, 0.0, 1.0)
        errors = []
        for shift in range(-2, 3):
            if shift < 0:
                lhs, rhs = scaled[-shift:], normalized_action[:shift]
            elif shift > 0:
                lhs, rhs = scaled[:-shift], normalized_action[shift:]
            else:
                lhs, rhs = scaled, normalized_action
            if len(lhs):
                errors.append(float(np.max(np.abs(lhs - rhs))))
        return min(errors)

    errors = [
        min(alignment_error(raw_action, scale), alignment_error(raw_state, scale))
        for scale in candidates
    ]
    best = int(np.argmin(errors))
    if errors[best] > 2e-4:
        raise ValueError(
            f"{record['source_episode_id']}: cannot reconcile {state_col} with normalized action gripper"
        )
    return np.clip(raw_state / candidates[best], 0.0, 1.0).astype(np.float32)


def _sim_proprio_rows(win: pd.DataFrame, record: dict) -> tuple[np.ndarray, np.ndarray]:
    state = np.zeros((len(win), EEF_DIM), dtype=np.float32)
    valid = np.zeros_like(state, dtype=bool)
    left_grip, right_grip = _normalized_action_grippers(win, record)
    if "states.ee_to_robot_pose" in win:
        grip = _sim_state_gripper_rows(win, record, None, left_grip)
        state[:, :10] = _pose7_to_arm10_rows(
            np.stack(win["states.ee_to_robot_pose"].values), grip, quaternion="wxyz"
        )
        valid[:, :10] = True
    else:
        for side, offset, normalized_grip in (("left", 0, left_grip), ("right", 10, right_grip)):
            column = f"states.{side}_ee_to_robot_pose"
            if column in win:
                grip = _sim_state_gripper_rows(win, record, side, normalized_grip)
                state[:, offset : offset + 10] = _pose7_to_arm10_rows(
                    np.stack(win[column].values), grip, quaternion="wxyz"
                )
                valid[:, offset : offset + 10] = True
    return state, valid


def _sim_proprio(win: pd.DataFrame, record: dict) -> tuple[np.ndarray, np.ndarray]:
    state, valid = _sim_proprio_rows(win, record)
    return state[0], valid[0]


def _to_unified(action20: np.ndarray, mask20: np.ndarray, n_valid: int) -> tuple[torch.Tensor, torch.Tensor]:
    action = np.zeros((32, ACTION_DIM), dtype=np.float32)
    action[:, LEFT_DST] = action20[:, :10]
    action[:, RIGHT_DST] = action20[:, 10:20]
    dim_mask = np.zeros(ACTION_DIM, dtype=bool)
    dim_mask[LEFT_DST] = mask20[:10]
    dim_mask[RIGHT_DST] = mask20[10:20]
    mask = np.zeros((32, ACTION_DIM), dtype=bool)
    mask[: max(0, min(32, n_valid))] = dim_mask
    return torch.from_numpy(action), torch.from_numpy(mask)


def _proprio_to_unified(state20: np.ndarray, mask20: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    state = np.zeros((1, ACTION_DIM), dtype=np.float32)
    state[:, LEFT_DST] = np.asarray(state20, dtype=np.float32)[None, :10]
    state[:, RIGHT_DST] = np.asarray(state20, dtype=np.float32)[None, 10:20]
    mask = np.zeros((1, ACTION_DIM), dtype=bool)
    mask[:, LEFT_DST] = np.asarray(mask20, dtype=bool)[None, :10]
    mask[:, RIGHT_DST] = np.asarray(mask20, dtype=bool)[None, 10:20]
    return torch.from_numpy(state), torch.from_numpy(mask)


def _robot_video_path(record: dict, info: dict) -> Path:
    root = Path(record["task_dir"])
    if info.get("layout") == "v21":
        return root / (
            f"videos/chunk-{info['episode_chunk']:03d}/{info['__cam_key']}/"
            f"episode_{info['episode_index']:06d}.mp4"
        )
    return root / f"videos/{info['__cam_key']}/chunk-{info['chunk']:03d}/file-{info['file']:03d}.mp4"


def pku_record_payload_paths(record: dict) -> tuple[Path, ...]:
    """Return the files needed to decode one indexed PKU episode."""

    if record["_source_domain"] == "ego":
        return (Path(record["video_path"]),)

    paths = [Path(record["data_path"])]
    supported_cameras = {"cam_high", "cam_left_wrist", "cam_right_wrist"}
    paths.extend(
        _robot_video_path(record, info)
        for alias, info in record.get("video_info", {}).items()
        if alias in supported_cameras
    )
    return tuple(paths)


class PKUNative30Dataset(BaseDataset):
    """Frame-window reader for one audited PKU pointer index."""

    def __init__(
        self,
        index_path: str,
        *,
        num_frames: int = 33,
        video_stride: int = 4,
        window_stride: int = 1,
        height: int = 384,
        width: int = 320,
        multiview: bool = True,
        split: str = "train",
        color_jitter: Any = None,
        seed: int = 42,
        shuffle_index: bool = False,
        normalize_mode: str | None = None,
        normalization_stats_path: str | None = None,
        enable_proprio_supervision: bool = False,
    ):
        if int(num_frames) != 33 or int(video_stride) != 4:
            raise ValueError("pku_native30 Figure 10 loader requires num_frames=33 and video_stride=4")
        if not multiview:
            raise ValueError("pku_native30 Figure 10 loader requires multiview=true")
        self.index_path = Path(index_path).resolve()
        self._normalize_mode = normalize_mode
        # Figure 10's initial protocol keeps the fixed-width proprio tensor in
        # the sample contract but disables all proprio conditioning/supervision.
        self.enable_proprio_supervision = bool(enable_proprio_supervision)
        self.normalization_stats_path = (
            str(Path(normalization_stats_path).expanduser().resolve()) if normalization_stats_path else None
        )
        if normalize_mode not in (None, "none", "null") and self.normalization_stats_path is None:
            raise ValueError("PKUNative30Dataset requires normalization_stats_path when normalization is enabled")
        if self.normalization_stats_path is not None:
            self._action_stats, self._state_stats = _load_normalization_stats(
                self.normalization_stats_path,
                index_path=self.index_path,
                normalize_mode=normalize_mode,
            )
        else:
            self._action_stats, self._state_stats = None, None
        self.height, self.width = int(height), int(width)
        self.window_stride = max(1, int(window_stride))
        self.split = split
        self.owns_shuffled_index = bool(shuffle_index)
        self._base_seed = int(seed)
        self._epoch = -1
        self.video_sample_indices = np.arange(0, 33, 4, dtype=np.int64)
        self._color_jitter = None
        if split == "train" and color_jitter_enabled(color_jitter):
            get = color_jitter.get if hasattr(color_jitter, "get") else lambda key, default: default
            self._color_jitter = VideoColorJitter(
                brightness=float(get("brightness", 0.2)),
                contrast=float(get("contrast", 0.2)),
                saturation=float(get("saturation", 0.2)),
                hue=float(get("hue", 0.0)),
            )

        payload = json.loads(self.index_path.read_text())
        if payload.get("version") != INDEX_VERSION:
            raise ValueError(f"unsupported PKU index version: {payload.get('version')!r}")
        records: list[dict] = []
        for source in payload["sources"]:
            shard = self.index_path.parent / source["index_file"]
            if _sha256(shard) != source["sha256"]:
                raise ValueError(f"pointer checksum mismatch: {shard}")
            with gzip.open(shard, "rt", encoding="utf-8") as stream:
                for line in stream:
                    record = json.loads(line)
                    if record.get("split") == split:
                        record["_source_domain"] = source["domain"]
                        records.append(record)
        if not records:
            raise ValueError(f"{self.index_path}: no records for split={split!r}")
        self.records = records
        lengths = np.asarray(
            [int(record.get("length", record.get("source_num_frames", 0))) for record in records],
            dtype=np.int64,
        )
        if np.any(lengths <= 0):
            raise ValueError("PKU pointer index contains non-positive episode lengths")
        min_len = 1 if split == "train" else 33
        starts = np.where(lengths >= min_len, (lengths - min_len) // self.window_stride + 1, 0)
        self._lengths = lengths
        self._cum_starts = np.concatenate([[0], np.cumsum(starts)]).astype(np.int64)
        self._n_total = int(self._cum_starts[-1])
        self.index_metadata = payload
        self.set_epoch(0)

    @property
    def action_dim(self) -> int:
        return ACTION_DIM

    @property
    def normalization_stats(self):
        return self._action_stats

    def __len__(self) -> int:
        return self._n_total

    def set_epoch(self, epoch: int) -> None:
        """Select a constant-memory permutation for direct single-source training."""

        if not self.owns_shuffled_index:
            self._multiplier, self._offset = 1, 0
            return
        epoch = int(epoch)
        if epoch == self._epoch:
            return
        rng = random.Random(self._base_seed + epoch * 7919)
        multiplier = rng.randrange(1, self._n_total)
        while math.gcd(multiplier, self._n_total) != 1:
            multiplier = (multiplier + 1) % self._n_total or 1
        self._multiplier = multiplier
        self._offset = rng.randrange(self._n_total)
        self._epoch = epoch

    def _decode_video(self, record: dict, offset: int, actual_len: int):
        local = np.minimum(self.video_sample_indices, actual_len - 1)
        frames_by_slot: dict[str, list] = {}
        if record["_source_domain"] == "ego":
            indices = (int(record["video_start_frame"]) + offset + local).tolist()
            frames_by_slot["head"] = decode_video_frames(str(record["video_path"]), indices, 256, 320)
        else:
            for alias, info in record.get("video_info", {}).items():
                first = int(round(float(info.get("from_ts", 0.0)) * float(record["fps"])))
                indices = (first + offset + local).tolist()
                slot = {"cam_high": "head", "cam_left_wrist": "left", "cam_right_wrist": "right"}.get(alias)
                if slot is None:
                    continue
                h, w = (256, 320) if slot == "head" else (128, 160)
                frames_by_slot[slot] = decode_video_frames(str(_robot_video_path(record, info)), indices, h, w)
        if "head" not in frames_by_slot:
            raise ValueError(f"{record['source_episode_id']}: no head-camera frames")
        video, exclusion = [], []
        for i in range(len(self.video_sample_indices)):
            item = {name: frames[i] for name, frames in frames_by_slot.items()}
            frame, missing = assemble_multiview_layout(
                item, ["head", "left", "right"], self.height, self.width, return_missing_mask=True
            )
            video.append(frame)
            exclusion.append(missing)
        if self._color_jitter is not None:
            video = self._color_jitter.apply(
                {"video": video, "video_jitter_exclusion_masks": exclusion}
            )["video"]
        return video

    def __getitem__(self, idx: int) -> dict:
        idx = int(idx)
        if idx < 0:
            idx += self._n_total
        if idx < 0 or idx >= self._n_total:
            raise IndexError(idx)
        idx = (self._multiplier * idx + self._offset) % self._n_total
        ep = int(np.searchsorted(self._cum_starts, idx, side="right") - 1)
        record = self.records[ep]
        offset = int((idx - self._cum_starts[ep]) * self.window_stride)
        actual_len = min(33, int(self._lengths[ep]) - offset)
        video = self._decode_video(record, offset, actual_len)

        if record["_source_domain"] == "ego":
            action = torch.zeros(32, ACTION_DIM, dtype=torch.float32)
            action_mask = torch.zeros(32, ACTION_DIM, dtype=torch.bool)
            proprio = torch.zeros(1, ACTION_DIM, dtype=torch.float32)
            proprio_mask = torch.zeros(1, ACTION_DIM, dtype=torch.bool)
            prompt = str(record.get("language") or "Perform the observed manipulation task.")
        else:
            table = _episode_table(record["data_path"], int(record["episode_index"]))
            win = _pad_window(table.iloc[offset : offset + 33])
            if record["_source_domain"] == "sim":
                action20, dim_mask20 = _sim_eef(win, record)
                state20, state_mask20 = _sim_proprio(win, record)
            else:
                action20, dim_mask20 = _real_eef(win, record)
                state20, state_mask20 = _real_proprio(win, record)
            action20 = apply_normalization(action20, self._action_stats, self._normalize_mode).astype(np.float32)
            state20 = apply_normalization(state20, self._state_stats, self._normalize_mode).astype(np.float32)
            action, action_mask = _to_unified(action20, dim_mask20, actual_len - 1)
            proprio, proprio_mask = _proprio_to_unified(state20, state_mask20)
            if not self.enable_proprio_supervision:
                proprio_mask.zero_()
            prompt = str(record.get("instruction") or "Perform the robot manipulation task.")

        return {
            "video": video,
            "vace_video": None,
            "first_frame_image": [video[0]],
            "action": action,
            "action_mask": action_mask,
            "video_mask": torch.from_numpy(self.video_sample_indices < actual_len),
            "proprio": proprio,
            "proprio_mask": proprio_mask,
            "prompt": prompt,
            "_dataset_name": str(record.get("source_name", "pku")),
            "_source_episode_id": str(record["source_episode_id"]),
        }

    @classmethod
    def from_config(cls, config, split: str = "train") -> "PKUNative30Dataset":
        from openwam.dataloader.utils import get_cfg

        index_path = get_cfg(config, "index_path") or get_cfg(config, "dataset_dir")
        if index_path is None:
            raise ValueError("PKUNative30Dataset requires index_path")
        keys = (
            "num_frames",
            "video_stride",
            "window_stride",
            "height",
            "width",
            "multiview",
            "color_jitter",
            "seed",
            "normalize_mode",
            "normalization_stats_path",
        )
        kwargs = {key: get_cfg(config, key) for key in keys if get_cfg(config, key) is not None}
        shuffle_index = bool(get_cfg(config, "shuffle_index", split == "train"))
        return cls(str(index_path), split=split, shuffle_index=shuffle_index, **kwargs)


class PKUFigure10MixtureDataset(MixtureDataset):
    """Memory-constant proportional mixture for the 64.8M-window co-train set."""

    def __init__(self, datasets: list[PKUNative30Dataset], names: list[str], seed: int = 42):
        if len(datasets) != 2 or len(names) != 2:
            raise ValueError("PKUFigure10MixtureDataset requires exactly ego and robot datasets")
        self._datasets = datasets
        self._names = names
        self._base_seed = int(seed)
        self._lengths = np.asarray([len(dataset) for dataset in datasets], dtype=np.int64)
        if np.any(self._lengths <= 0):
            raise ValueError("PKU Figure 10 mixture contains an empty source")
        self._cum_lengths = np.concatenate([[0], np.cumsum(self._lengths)]).astype(np.int64)
        self._n_total = int(self._cum_lengths[-1])
        self._action_dim = ACTION_DIM
        self.normalization_stats_path = getattr(datasets[1], "normalization_stats_path", None)
        self._epoch = -1
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        epoch = int(epoch)
        if epoch == self._epoch:
            return
        rng = random.Random(self._base_seed + epoch * 7919)
        multiplier = rng.randrange(1, self._n_total)
        while math.gcd(multiplier, self._n_total) != 1:
            multiplier = (multiplier + 1) % self._n_total or 1
        self._multiplier = multiplier
        self._offset = rng.randrange(self._n_total)
        self._epoch = epoch

    def __len__(self) -> int:
        return self._n_total

    def __getitem__(self, idx: int) -> dict:
        shuffled = (self._multiplier * int(idx) + self._offset) % self._n_total
        dataset_index = int(np.searchsorted(self._cum_lengths, shuffled, side="right") - 1)
        local_index = shuffled - int(self._cum_lengths[dataset_index])
        sample = self._datasets[dataset_index][local_index]
        sample["_dataset_index"] = dataset_index
        sample["_dataset_name"] = self._names[dataset_index]
        return sample

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def normalization_stats(self):
        return getattr(self._datasets[1], "normalization_stats", None)

    @property
    def datasets(self):
        return self._datasets

    @property
    def names(self):
        return list(self._names)

    @property
    def weights(self):
        return (self._lengths / self._n_total).tolist()

    def dataset_sample_counts(self):
        return {name: int(length) for name, length in zip(self._names, self._lengths)}

    @classmethod
    def from_config(cls, config, split: str = "train") -> "PKUFigure10MixtureDataset":
        from openwam.dataloader.utils import get_cfg

        ego_path = get_cfg(config, "ego_index_path")
        robot_path = get_cfg(config, "robot_index_path")
        if ego_path is None or robot_path is None:
            raise ValueError("PKU Figure 10 mixture requires ego_index_path and robot_index_path")
        keys = (
            "num_frames",
            "video_stride",
            "window_stride",
            "height",
            "width",
            "multiview",
            "color_jitter",
            "enable_proprio_supervision",
        )
        kwargs = {key: get_cfg(config, key) for key in keys if get_cfg(config, key) is not None}
        robot_kwargs = dict(kwargs)
        for key in ("normalize_mode", "normalization_stats_path"):
            value = get_cfg(config, key)
            if value is not None:
                robot_kwargs[key] = value
        datasets = [
            PKUNative30Dataset(str(ego_path), split=split, **kwargs),
            PKUNative30Dataset(str(robot_path), split=split, **robot_kwargs),
        ]
        return cls(datasets, ["ego350", "robot250"], seed=int(get_cfg(config, "seed", 42)))


__all__ = ["PKUNative30Dataset", "PKUFigure10MixtureDataset", "pku_record_payload_paths"]
