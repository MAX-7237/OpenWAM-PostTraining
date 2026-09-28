"""Manifest-driven SCJD reader. Only certified time/pose contracts may train."""
import hashlib
import json
from pathlib import Path
from functools import lru_cache

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation
import torch

from openwam.dataloader.bases.dataset import BaseDataset
from openwam.dataloader.transforms.multiview import assemble_multiview_layout
from openwam.dataloader.transforms.video import VideoColorJitter
from openwam.dataloader.utils.scjd_contracts import aperture, resample_eef
from openwam.dataloader.utils.video_io import decode_video_frames


def checked_json(path):
    return json.loads(Path(path).read_text())


@lru_cache(maxsize=2)
def data_table(path):
    return pq.read_table(path)


class SCJDDataset(BaseDataset):
    """Rows identify valid segments, native file offsets and explicit transforms.

    Certification is a separate offline step. Missing calibration, time mapping
    or statistics must fail before the trainer can consume any samples.
    """
    action_dim = 80
    normalization_stats = None

    def __init__(self, manifest_path, stats_path, color_jitter=None):
        self.root = Path(manifest_path).parent
        self.meta = checked_json(manifest_path)
        if self.meta.get('training_ready') is not True:
            raise ValueError('SCJD manifest is not certified for training')
        table_path = self.root / self.meta['segments_file']
        if hashlib.sha256(table_path.read_bytes()).hexdigest() != self.meta['segments_sha256']:
            raise ValueError('SCJD segment manifest checksum mismatch')
        stats_bytes = Path(stats_path).read_bytes()
        if hashlib.sha256(stats_bytes).hexdigest() != self.meta['stats_sha256']:
            raise ValueError('SCJD statistics checksum mismatch')
        self.stats = json.loads(stats_bytes)
        self.rows = pq.read_table(table_path).to_pylist()
        self.ends = np.cumsum([int(r['window_count']) for r in self.rows])
        self.jitter = VideoColorJitter(**color_jitter) if color_jitter else None

    @classmethod
    def from_config(cls, config, split='train'):
        if split != 'train':
            raise ValueError('SCJD training manifests have no implicit validation split')
        return cls(config['manifest_path'], config['stats_path'], config.get('color_jitter'))

    def __len__(self):
        return int(self.ends[-1]) if len(self.ends) else 0

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        ri = int(np.searchsorted(self.ends, index, side='right'))
        row = self.rows[ri]
        local = index - (int(self.ends[ri-1]) if ri else 0)
        contract = self.meta['contracts'][row['contract_id']]
        if contract.get('verified') is not True:
            raise ValueError('unverified SCJD source contract')
        table = data_table(row['data_path']).slice(row['row_offset'], row['row_count'])
        d = table.to_pydict()
        if not np.all(np.asarray(d['episode_index']) == row['episode_index']):
            raise ValueError('SCJD physical row/episode mismatch')
        t = np.asarray(d[contract['timestamp_column']], float).reshape(-1)
        t = t * contract['timestamp_scale'] + row['timestamp_offset']
        q = row['start_time'] + (local + np.arange(33)) / 30
        # Restrict interpolation to the independently certified continuous range.
        a, b = row['valid_row_start'], row['valid_row_end']
        x = np.asarray(d['action'], float)[:, contract['action_indices']]
        for side, off in enumerate((0, 10)):
            cal = contract['action_gripper'][side]
            x[:, off+9] = aperture(x[:, off+9], cal['closed'], cal['open'])
        if contract['action_time_semantics'] not in ('row_aligned_command', 'next_state_at_query'):
            raise ValueError('unsupported action time semantics')
        action_query = q[:-1] if contract['action_time_semantics'] == 'row_aligned_command' else q[1:]
        actions = resample_eef(t[a:b], x[a:b], action_query, contract['discrete_gripper'])
        state_index = int(np.searchsorted(t[a:b], q[0], side='right') - 1) + a
        if state_index < a:
            raise ValueError('no causal proprio sample')
        state = np.zeros((1,20), np.float32)
        for side, off in enumerate((0,10)):
            spec = contract['state'][side]
            pose = np.asarray(d[spec['pose_column']][state_index], float)
            if len(pose) != 7 or not np.isfinite(pose).all():
                raise ValueError('invalid native proprio pose')
            quat = pose[3:][spec['quaternion_xyzw_indices']]
            tf = np.eye(4)
            tf[:3,:3] = Rotation.from_quat(quat).as_matrix()
            tf[:3,3] = pose[:3] * spec['position_scale']
            tf = np.asarray(spec['base_from_native']) @ tf @ np.asarray(spec['native_endpoint_from_target'])
            state[0,off:off+3] = tf[:3,3]
            state[0,off+3:off+9] = tf[:3,:2].T.reshape(-1)
            g = np.asarray(d[spec['gripper_column']][state_index]).item()
            state[0,off+9] = aperture(g,spec['closed'],spec['open'])
        for arr,kind in [(actions,'action'),(state,'state')]:
            stats = self.stats[row['contract_id']][kind]
            for k in [0,1,2,10,11,12]:
                lo,hi = stats['q01'][k],stats['q99'][k]
                if not np.isfinite([lo,hi]).all() or hi <= lo:
                    raise ValueError('invalid position normalization bounds')
                arr[:,k] = np.clip(2*(arr[:,k]-lo)/(hi-lo)-1,-1,1)
        frames = {}
        layout = []
        for camera in json.loads(row['cameras_json']):
            layout.append(camera['name'])
            # Audit supplies the true native frame-to-time map, never info.fps.
            times = np.asarray(d[camera['timestamp_column']],float).reshape(-1)*camera['timestamp_scale']+camera['timestamp_offset']
            if np.any(np.diff(times) <= 0):
                raise ValueError('camera timestamps not strictly increasing')
            indices = np.searchsorted(times,q[::4],side='right')-1
            if indices.min()<0 or q[32]>times[-1]:
                raise ValueError('video extrapolation forbidden')
            frames[camera['name']] = decode_video_frames(camera['path'],(indices+camera['frame_offset']).tolist(),384,320)
        if len(layout)!=3:
            raise ValueError('three verified camera streams required')
        video=[assemble_multiview_layout({k:v[i] for k,v in frames.items()},layout,384,320) for i in range(9)]
        if self.jitter:
            video=self.jitter.apply({'video':video})['video']
        slots=np.r_[0:10,34:44]
        action=np.zeros((32,80),np.float32);action[:,slots]=actions
        proprio=np.zeros((1,80),np.float32);proprio[:,slots]=state
        mask=np.zeros((32,80),bool);mask[:,slots]=True
        return dict(video=video,first_frame_image=[video[0]],vace_video=None,
                    video_mask=torch.ones(9,dtype=torch.bool),action=torch.from_numpy(action),
                    action_mask=torch.from_numpy(mask),proprio=torch.from_numpy(proprio),
                    proprio_mask=torch.from_numpy(mask[:1].copy()),prompt=row['prompt'])
