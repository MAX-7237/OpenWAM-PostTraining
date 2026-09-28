import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from openwam.dataloader.pku_native30 import (
    ACTION_REPRESENTATION,
    FRAME_POLICY,
    GRIPPER_CONVENTION,
    STATS_SCHEMA_VERSION,
    PKUFigure10MixtureDataset,
    PKUNative30Dataset,
    _absolute_eef_targets,
    _load_normalization_stats,
    _normalize_gripper,
    _proprio_to_unified,
    _real_proprio,
    _sim_state_gripper_rows,
    _to_unified,
)


def _stats_block():
    return {
        "mean": np.zeros(20, np.float32),
        "std": np.ones(20, np.float32),
        "min": -np.ones(20, np.float32),
        "max": np.ones(20, np.float32),
        "q01": -np.ones(20, np.float32),
        "q99": np.ones(20, np.float32),
    }


def test_absolute_eef_targets_and_unified_slots():
    poses = np.zeros((33, 20), dtype=np.float32)
    identity6 = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    poses[:, 3:9] = identity6
    poses[:, 13:19] = identity6
    poses[:, 0] = np.arange(33) * 0.01
    poses[:, 10] = np.arange(33) * -0.02
    valid = np.zeros_like(poses, dtype=bool)
    valid[:, :9] = True
    valid[:, 10:19] = True

    action20, mask20 = _absolute_eef_targets(poses, valid)
    action80, mask80 = _to_unified(action20, mask20, 32)

    np.testing.assert_allclose(action80[:, 0].numpy(), np.arange(1, 33) * 0.01)
    np.testing.assert_allclose(action80[:, 34].numpy(), np.arange(1, 33) * -0.02)
    assert mask80[:, :9].all()
    assert mask80[:, 34:43].all()
    assert not mask80[:, 9].any()
    assert not mask80[:, 43].any()
    assert int(mask80[0].sum()) == 18


def test_absolute_eef_copies_normalized_future_gripper_targets():
    poses = np.zeros((33, 20), dtype=np.float32)
    identity6 = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    poses[:, 3:9] = identity6
    poses[:, 13:19] = identity6
    poses[:, 9] = np.linspace(0.0, 1.0, 33)
    poses[:, 19] = np.linspace(1.0, 0.0, 33)
    valid = np.ones_like(poses, dtype=bool)

    action20, mask20 = _absolute_eef_targets(poses, valid)

    np.testing.assert_allclose(action20[:, 9], poses[1:, 9])
    np.testing.assert_allclose(action20[:, 19], poses[1:, 19])
    assert mask20.all()


def test_source_gripper_conventions_are_canonicalized():
    agilex = {"source_name": "agilex", "source_episode_id": "a"}
    tienkung = {"source_name": "tienkung", "source_episode_id": "t"}
    ur = {"source_name": "ur", "source_episode_id": "u"}

    np.testing.assert_allclose(_normalize_gripper([0.0, 0.05, 0.1], agilex), [0.0, 0.5, 1.0])
    np.testing.assert_allclose(_normalize_gripper([0.0, 0.25, 1.0], tienkung), [1.0, 0.75, 0.0])
    np.testing.assert_allclose(_normalize_gripper([0.0, 0.25, 1.0], ur), [0.0, 0.25, 1.0])


def test_normalization_stats_are_bound_to_the_pointer_index(tmp_path: Path):
    index = tmp_path / "index.json"
    index.write_text('{"version":"pku_native30_v1"}')
    stats_path = tmp_path / "stats.npy"
    payload = {
        "eef": _stats_block(),
        "metadata": {
            "schema_version": STATS_SCHEMA_VERSION,
            "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
            "action_representation": ACTION_REPRESENTATION,
            "frame_policy": FRAME_POLICY,
            "gripper_convention": GRIPPER_CONVENTION,
        },
    }
    np.save(stats_path, payload, allow_pickle=True)

    action, state = _load_normalization_stats(stats_path, index_path=index, normalize_mode="quantile")
    np.testing.assert_array_equal(action["q01"], -np.ones(20))
    np.testing.assert_array_equal(state["q99"], np.ones(20))

    index.write_text('{"version":"changed"}')
    with np.testing.assert_raises_regex(ValueError, "provenance mismatch"):
        _load_normalization_stats(stats_path, index_path=index, normalize_mode="quantile")


def test_real_native_pose_proprio_includes_both_grippers():
    pose = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    win = pd.DataFrame(
        {
            "puppet.end_effector_left_pose_align.data": [pose],
            "puppet.end_effector_right_pose_align.data": [pose],
            "puppet.end_effector_left_position_align.data": [np.array([0.025], np.float32)],
            "puppet.end_effector_right_position_align.data": [np.array([0.075], np.float32)],
        }
    )
    record = {"source_name": "agilex", "source_episode_id": "a"}

    state20, mask20 = _real_proprio(win, record)
    state80, mask80 = _proprio_to_unified(state20, mask20)

    assert mask20.all()
    assert mask80[:, :10].all() and mask80[:, 34:44].all()
    np.testing.assert_allclose(state80[0, [9, 43]], [0.25, 0.75])


def test_sim_state_gripper_uses_synchronous_state_for_limit_selection():
    # Lift2 action commands lead state by one frame. Limit selection must use
    # the synchronous state signal represented by the unified EEF gripper.
    raw_state = np.array([0.044, 0.022, 0.0044], dtype=np.float32)
    raw_action = np.array([0.022, 0.0044, 0.044], dtype=np.float32)
    win = pd.DataFrame(
        {
            "actions.left_gripper.position": [np.array([x]) for x in raw_action],
            "states.left_gripper.position": [np.array([x]) for x in raw_state],
        }
    )
    normalized = raw_state / 0.088
    result = _sim_state_gripper_rows(
        win,
        {"source_name": "interndata_sim_lift2", "source_episode_id": "lift2:test"},
        "left",
        normalized,
    )

    np.testing.assert_allclose(result, normalized)


def test_sim_state_gripper_can_select_limit_from_command_when_state_lags():
    normalized = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    win = pd.DataFrame(
        {
            "actions.gripper.position": [np.array([x]) for x in normalized],
            "states.gripper.position": [np.array([x]) for x in [1.0, 1.0, 1.0, 0.0]],
        }
    )
    result = _sim_state_gripper_rows(
        win,
        {"source_name": "interndata_sim_franka", "source_episode_id": "franka:test"},
        None,
        normalized,
    )

    np.testing.assert_allclose(result, [1.0, 1.0, 1.0, 0.0])


def test_registry_contains_pku_native30():
    from openwam.dataloader.registry import list_registered_datasets

    assert "pku_native30" in list_registered_datasets()
    assert "pku_figure10_mixture" in list_registered_datasets()


def test_compact_mixture_visits_every_sample_once():
    class Tiny:
        action_dim = 80

        def __init__(self, name, size):
            self.name, self.size = name, size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return {"key": (self.name, idx)}

    mixture = PKUFigure10MixtureDataset([Tiny("ego", 7), Tiny("robot", 5)], ["ego", "robot"], seed=42)
    epoch0 = [mixture[i]["key"] for i in range(len(mixture))]
    expected = {*(('ego', i) for i in range(7)), *(('robot', i) for i in range(5))}
    assert len(set(epoch0)) == 12
    assert set(epoch0) == expected
    mixture.set_epoch(1)
    epoch1 = [mixture[i]["key"] for i in range(len(mixture))]
    assert set(epoch1) == expected
    assert epoch1 != epoch0


def test_native_affine_shuffle_is_bijective_and_epoch_scoped():
    dataset = object.__new__(PKUNative30Dataset)
    dataset._n_total = 17
    dataset.owns_shuffled_index = True
    dataset._base_seed = 42
    dataset._epoch = -1

    dataset.set_epoch(0)
    epoch0 = [(dataset._multiplier * i + dataset._offset) % len(dataset) for i in range(len(dataset))]
    dataset.set_epoch(1)
    epoch1 = [(dataset._multiplier * i + dataset._offset) % len(dataset) for i in range(len(dataset))]

    assert sorted(epoch0) == list(range(17))
    assert sorted(epoch1) == list(range(17))
    assert epoch1 != epoch0
