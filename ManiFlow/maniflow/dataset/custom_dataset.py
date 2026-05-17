from typing import Dict
import torch
import numpy as np
from PIL import Image
import copy
from maniflow.common.pytorch_util import dict_apply
from maniflow.common.replay_buffer import ReplayBuffer
from maniflow.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from maniflow.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from maniflow.dataset.base_dataset import BaseDataset
from termcolor import cprint


# maniflow/dataset/utils_relative.py — add this as a utility module
from scipy.spatial.transform import Rotation


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """q: (..., 4) xyzw → R: (..., 3, 3)"""
    return Rotation.from_quat(q).as_matrix()


def rotmat_to_6d(R: np.ndarray) -> np.ndarray:
    """R: (..., 3, 3) → 6D: (..., 6) = [col0 | col1].

    Standard Zhou-et-al 6D form. Must stay consistent with `sixd_to_rotmat`,
    which Gram-Schmidts (a, b) treating them as the first two columns of R.
    A row-major reshape of R[:, :2] would interleave row entries instead, and
    the decoder's Gram-Schmidt would then operate on garbage 3-vectors.
    """
    return np.concatenate([R[..., 0], R[..., 1]], axis=-1)


def sixd_to_rotmat(d6: np.ndarray) -> np.ndarray:
    """Gram-Schmidt: (..., 6) → (..., 3, 3)."""
    a, b = d6[..., :3], d6[..., 3:]
    c1 = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b_proj = b - (c1 * b).sum(-1, keepdims=True) * c1
    c2 = b_proj / np.linalg.norm(b_proj, axis=-1, keepdims=True)
    c3 = np.cross(c1, c2, axis=-1)
    return np.stack([c1, c2, c3], axis=-1)


def make_hemisphere_continuous(q: np.ndarray) -> np.ndarray:
    """Flip signs so consecutive quaternions live in the same hemisphere.
    q: (T, 4) xyzw"""
    dots = np.einsum('ti,ti->t', q[1:], q[:-1])
    signs = np.cumprod(np.where(dots < 0, -1.0, 1.0))
    signs = np.concatenate([[1.0], signs])
    return q * signs[:, None]


def to_relative_6d(pose_seq: np.ndarray, base_pose: np.ndarray) -> np.ndarray:
    """Express a sequence of (T, 7) poses (3 pos + 4 quat) relative to a base
    (7,) pose, returning (T, 9): 3 relative position + 6D relative rotation.
    """
    T_seq = pose_seq[:, :3]                       # (T, 3)
    q_seq = pose_seq[:, 3:7]                      # (T, 4)
    T_base = base_pose[:3]                        # (3,)
    q_base = base_pose[3:7]                       # (4,)

    R_seq  = quat_to_rotmat(q_seq)                # (T, 3, 3)
    R_base = quat_to_rotmat(q_base)               # (3, 3)

    T_rel = (R_base.T @ (T_seq - T_base).T).T     # (T, 3)
    R_rel = R_base.T @ R_seq                      # (T, 3, 3)
    d6    = rotmat_to_6d(R_rel)                   # (T, 6)
    return np.concatenate([T_rel, d6], axis=-1)   # (T, 9)


def from_relative_6d(rel_seq: np.ndarray, base_pose: np.ndarray) -> np.ndarray:
    """Inverse of to_relative_6d. (T, 9) + (7,) → (T, 7) absolute pose."""
    T_rel = rel_seq[:, :3]
    d6    = rel_seq[:, 3:]
    R_rel = sixd_to_rotmat(d6)
    T_base = base_pose[:3]
    q_base = base_pose[3:7]
    R_base = quat_to_rotmat(q_base)
    T_abs = T_base + (R_base @ T_rel.T).T
    R_abs = R_base @ R_rel
    q_abs = Rotation.from_matrix(R_abs).as_quat()  # xyzw
    return np.concatenate([T_abs, q_abs], axis=-1)

class CustomDataset(BaseDataset):
    def __init__(self,
            zarr_path,
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            task_name=None,
            use_feature_cache=False,
            feature_cache_key='camera_1_r3m',
            ):
        super().__init__()
        self.task_name = task_name
        self.use_feature_cache = use_feature_cache
        self.feature_cache_key = feature_cache_key
        cprint(f'Loading DexArtDataset from {zarr_path}', 'green')

        if use_feature_cache:
            buffer_keys = [feature_cache_key, 'agent_pos', 'action']
            cprint(f'Using pre-computed R3M features: {feature_cache_key}', 'yellow')
        else:
            buffer_keys = ['camera_1', 'agent_pos', 'action']

        self.replay_buffer = ReplayBuffer.copy_from_path(
                zarr_path, keys=buffer_keys)

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

        self.zarr_path = zarr_path
        self.train_episodes_num = np.sum(train_mask)
        self.val_episodes_num = np.sum(val_mask)

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        # Sample N transformed examples to fit the normalizer.
        n_samples = min(len(self), 1000)
        idxs = np.linspace(0, len(self) - 1, n_samples).astype(int)
        actions = []
        agent_poses = []
        for i in idxs:
            sample = self.sampler.sample_sequence(i)
            d = self._sample_to_data(sample)
            actions.append(d['action'])
            agent_poses.append(d['obs']['agent_pos'])
        actions = np.concatenate(actions, axis=0)        # (n_samples * Ha, 26)
        agent_poses = np.concatenate(agent_poses, axis=0)
        
        normalizer = LinearNormalizer()
        normalizer.fit(data={'action': actions}, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = SingleFieldLinearNormalizer.create_identity()
        
        # Also fit agent_pos so its features are normalized (not identity)
        ap_normalizer = SingleFieldLinearNormalizer()
        ap_normalizer.fit(data=agent_poses, last_n_dims=1, mode=mode)
        normalizer['agent_pos'] = ap_normalizer
        return normalizer


    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['agent_pos'][:,].astype(np.float32)
        if self.use_feature_cache:
            image = sample[self.feature_cache_key][:,].astype(np.float32)
        else:
            image = sample['camera_1'][:,].astype(np.float32)
        action = sample['action'].astype(np.float32)

        base_rot = quat_to_rotmat(agent_pos[-1, 3:7])
        base_pos = agent_pos[-1,:3]
        
        observed_pos = agent_pos[:, :3]
        observed_rot = quat_to_rotmat(agent_pos[:, 3:7])
        observed_hand = agent_pos[:, 7:]


        action_pos = action[:,:3]
        action_rot = quat_to_rotmat(action[:,3:7])
        action_hand = action[:, 7:]

        action_pos_in_base = (action_pos - base_pos) @ base_rot   
        action_rot_in_base = rotmat_to_6d(base_rot.T @ action_rot )

        observed_pos_in_base =   (observed_pos - base_pos) @ base_rot
        observed_rot_in_base = rotmat_to_6d(base_rot.T @ observed_rot)


        agent_pos = np.concatenate(
        [observed_pos_in_base, observed_rot_in_base, observed_hand],
        axis=-1).astype(np.float32)
        action = np.concatenate([action_pos_in_base, action_rot_in_base, action_hand],
        axis=-1).astype(np.float32)




        
        data = {
            'obs': {
                'agent_pos': agent_pos,
                'image': image
                },
            'action': action}
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        to_torch_function = lambda x: torch.from_numpy(x) if x.__class__.__name__ == 'ndarray' else x
        torch_data = dict_apply(data, to_torch_function)
        return torch_data
