"""Pure time-domain and gripper contracts for SCJD preparation."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

MODEL_HZ = 30
HORIZON = 32 / MODEL_HZ


def aperture(values, closed, opened):
    """Convert physical aperture or closedness; never fit per episode."""
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all() or not np.isfinite([closed, opened]).all() or closed == opened:
        raise ValueError("invalid gripper values or calibration")
    return np.clip((values - closed) / (opened - closed), 0, 1).astype(np.float32)


def canonical_gripper(source, values, *, closed=None, opened=None):
    """Convert source grippers to the shared 0=closed, 1=open convention."""
    x = np.asarray(values, dtype=np.float32)
    source = str(source).lower()
    if source == "tienkung":
        return np.clip(1.0 - x, 0.0, 1.0)
    if source == "agibot":
        # Action values are normalized closedness; state values should pass
        # through ``aperture(..., 123, 34)`` before reaching this helper.
        return np.clip(1.0 - x, 0.0, 1.0)
    if source == "agilex":
        if closed is None or opened is None:
            raise ValueError("Agilex aperture requires explicit hardware endpoints")
        return aperture(x, closed, opened)
    if source == "egodex":
        return aperture(x, 0.04 if closed is None else closed, 0.10 if opened is None else opened)
    raise ValueError(f"unknown gripper source {source!r}")


def time_segments(timestamps):
    """Return row ranges and complete 30-Hz window counts; discard bad gaps."""
    t = np.asarray(timestamps, dtype=np.float64)
    if t.ndim != 1 or len(t) < 2:
        return []
    dt = np.diff(t)
    positive = dt[np.isfinite(dt) & (dt > 0)]
    if not len(positive):
        return []
    median = float(np.median(positive))
    if median > 1 / 15 + 1e-6:
        return []
    limit = min(0.1, 2 * median)
    boundaries = np.flatnonzero(~np.isfinite(dt) | (dt <= 0) | (dt > limit + 1e-6)) + 1
    edges = np.r_[0, boundaries, len(t)]
    result = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b - a < 2 or not np.isfinite(t[a:b]).all():
            continue
        span = t[b - 1] - t[a]
        n = int(np.floor((span - HORIZON) * MODEL_HZ + 1e-6)) + 1
        if n > 0:
            result.append((int(a), int(b), n, float((n - 1) / MODEL_HZ + HORIZON)))
    return result


def resample_eef(t, eef, query, discrete_gripper=False):
    """Resample one validated contiguous EEF20 sequence without extrapolation."""
    t, x, q = np.asarray(t, float), np.asarray(eef, float), np.asarray(query, float)
    if x.shape != (len(t), 20) or not np.isfinite(x).all():
        raise ValueError("EEF must be finite [N,20]")
    if len(t) < 2 or np.any(np.diff(t) <= 0) or not np.isfinite(t).all():
        raise ValueError("timestamps must strictly increase")
    if not np.isfinite(q).all() or q.min() < t[0] or q.max() > t[-1]:
        raise ValueError("extrapolation forbidden")
    dt = np.diff(t)
    if np.median(dt) > 1/15 + 1e-6 or dt.max() > min(.1, 2*np.median(dt)) + 1e-6:
        raise ValueError("low rate or gap")
    left = np.maximum(0, np.searchsorted(t, q, side="right") - 1)
    if 1 / np.median(dt) > 30 + .01:
        # Causal time-domain decimation avoids invented high-rate commands.
        return x[left].astype(np.float32)
    out = np.empty((len(q), 20), np.float32)
    for offset in (0, 10):
        for d in (0, 1, 2):
            out[:, offset+d] = np.interp(q, t, x[:, offset+d])
        columns = x[:, offset+3:offset+9].reshape(-1, 2, 3).transpose(0, 2, 1)
        mats = np.concatenate((columns, np.cross(columns[:,:,0], columns[:,:,1])[:,:,None]), axis=2)
        if not np.allclose(mats.transpose(0,2,1) @ mats, np.eye(3), atol=1e-3):
            raise ValueError("invalid rotation-6D")
        rot = Slerp(t, Rotation.from_matrix(mats))(q).as_matrix()
        out[:, offset+3:offset+9] = rot[:,:,:2].transpose(0,2,1).reshape(-1,6)
        out[:, offset+9] = x[left,offset+9] if discrete_gripper else np.interp(q,t,x[:,offset+9])
    return out


def nested_equal_hours(rows, budgets=(250, 600), seed=42):
    """Stable episode prefixes, equal source budgets, no replacement."""
    import hashlib
    selected = {str(b): [] for b in budgets}
    for source in ("agilex", "tienkung", "agibot"):
        candidates = [r for r in rows if r['source'] == source and r['effective_seconds'] > 0]
        if len({r['episode_id'] for r in candidates}) != len(candidates):
            raise ValueError(f"duplicate episode IDs: {source}")
        candidates.sort(key=lambda r: hashlib.sha256(f"{seed}:{source}:{r['episode_id']}".encode()).digest())
        for budget in budgets:
            target = budget * 3600 / 3
            seconds = 0
            for row in candidates:
                selected[str(budget)].append(row)
                seconds += row['effective_seconds']
                if seconds >= target:
                    break
            if seconds < target:
                raise ValueError(f"{source}: {seconds/3600:.3f}h available, needs {budget/3:.3f}h")
    return selected
