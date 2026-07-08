from typing import Dict
import copy
import torch
import numpy as np
from termcolor import cprint

from maniflow.common.pytorch_util import dict_apply
from maniflow.common.replay_buffer import ReplayBuffer
from maniflow.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from maniflow.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from maniflow.dataset.base_dataset import BaseDataset
# Reuse the exact pose math the coffee-cup image policy uses.
from maniflow.dataset.custom_dataset import quat_to_rotmat, rotmat_to_6d


class HumanDemoPointcloudDataset(BaseDataset):
    """Human-demo point-cloud + proprioception dataset for ManiFlow (fork of TactilePointcloudDataset,
    kept SEPARATE so the deployed tactile/coffee-cup path is untouched).

    The human zarr (offline_human_demo.py `process`) is OBJECT-ONLY (no robot taxels; force channels 0)
    and — new — is stored **already frame-baked**: the fixed-first-EE (episode-TCP) anchoring AND the
    Vive->ORCA correction `C` are applied OFFLINE, so the cloud+EE live in the robot-EE convention. The
    dataset therefore does NOT apply any corr; it only does quat -> 6D and the OPTIONAL last-observed-EE
    re-anchor (`relative_to_ee`), exactly like the tactile class:

      data/point_cloud  (N, P, 6)   float32   # FFS object pts [x,y,z, 0,0,0]  (robot-EE first-anchored)
      data/agent_pos    (N, 24)     float32   # 3 ee_pos + 4 ee_quat(xyzw) + 17 hand        (observed)
      data/action       (N, 24)     float32   # 3 ee_tgt + 4 ee_quat(xyzw) + 17 hand_target (pose target)
      meta/episode_ends (E,)        int64
      meta.attrs['settings']                  # self-describing: frame, anchor_correction, corr_baked, ...

    Deploy pairing: `relative_to_ee=false` <-> runner `episode_anchored=true`; `relative_to_ee=true`
    <-> runner relative mode. In BOTH, the runner uses `anchor_correction = identity` (C is baked here).
    """

    def __init__(self,
            zarr_path,
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            task_name=None,
            relative_pointcloud=True,
            relative_to_ee=True,
            ):
        super().__init__()
        self.task_name = task_name
        self.relative_pointcloud = relative_pointcloud
        # True: encode obs+action (and, if relative_pointcloud, the cloud) in the LAST-OBSERVED-EE frame.
        # False: keep everything in the stored (robot-EE first-anchored) frame. The Vive->ORCA corr is
        # already baked into the zarr either way.
        self.relative_to_ee = relative_to_ee
        cprint(f'Loading HumanDemoPointcloudDataset from {zarr_path}', 'green')

        buffer_keys = ['point_cloud', 'agent_pos', 'action']
        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=buffer_keys)

        # ── read the self-describing settings so training records how the zarr was baked ──
        self.settings = self._read_settings(zarr_path)
        frame = self.settings.get('frame', 'UNKNOWN')
        corr_baked = self.settings.get('corr_baked', None)
        cprint(f'  dataset frame = {frame}  corr_baked = {corr_baked}  '
               f'anchor_correction = {self.settings.get("anchor_correction", "n/a")}', 'cyan')
        if corr_baked is not True:
            cprint('  WARNING: this zarr is NOT corr-baked (corr_baked != true). At deploy the runner '
                   'anchor_correction MUST equal the training corr (NOT identity), or the frame is off '
                   'by the hand-mount rotation. Bake it via offline_human_demo --anchor-correction.', 'red')

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

    @staticmethod
    def _read_settings(zarr_path):
        """Return meta.attrs['settings'] (self-describing processing settings), or {} for a legacy zarr."""
        try:
            import zarr
            root = zarr.open(str(zarr_path), mode='r')
            return dict(root['meta'].attrs.get('settings', {}) or {})
        except Exception as e:                                   # never block training on a metadata read
            cprint(f'  (could not read zarr settings meta: {e})', 'yellow')
            return {}

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask)
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        # action, agent_pos AND point_cloud are all per-window transformed, so they must be
        # fit from sampled transformed windows (not the raw replay buffer), like CustomDataset.
        n_samples = min(len(self), 1000)
        idxs = np.linspace(0, len(self) - 1, n_samples).astype(int)
        actions, agent_poses, clouds = [], [], []
        for i in idxs:
            d = self._sample_to_data(self.sampler.sample_sequence(i))
            actions.append(d['action'])
            agent_poses.append(d['obs']['agent_pos'])
            clouds.append(d['obs']['point_cloud'])
        actions = np.concatenate(actions, axis=0)            # (n*Ha, 26)  ee_pose 6D + hand(17)
        agent_poses = np.concatenate(agent_poses, axis=0)    # (n*To, 26)
        clouds = np.concatenate(clouds, axis=0)              # (n*To, P, 6)

        normalizer = LinearNormalizer()
        normalizer.fit(data={'action': actions}, last_n_dims=1, mode=mode, **kwargs)

        ap_normalizer = SingleFieldLinearNormalizer()
        ap_normalizer.fit(data=agent_poses, last_n_dims=1, mode=mode)
        normalizer['agent_pos'] = ap_normalizer

        # last_n_dims=1 -> each of the 6 channels [x,y,z,fx,fy,fz] normalized independently
        # over all points and frames.
        pc_normalizer = SingleFieldLinearNormalizer()
        pc_normalizer.fit(data=clouds, last_n_dims=1, mode=mode)
        normalizer['point_cloud'] = pc_normalizer
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['agent_pos'][:,].astype(np.float32)   # (T, 24) ee_pose(7) + hand(17)
        action = sample['action'].astype(np.float32)             # (T, 24) ee_pose_tgt(7) + hand_tgt(17)
        pc = sample['point_cloud'][:,].astype(np.float32)        # (T, P, 6) robot-EE first-anchored

        # obs (agent_pos) AND action are absolute EE POSES in the stored (robot-EE) frame; quat -> 6D,
        # hand passthrough. Both 24 -> 26.
        observed_pos = agent_pos[:, :3]
        observed_rot = quat_to_rotmat(agent_pos[:, 3:7])
        observed_hand = agent_pos[:, 7:]
        action_pos = action[:, :3]
        action_rot = quat_to_rotmat(action[:, 3:7])
        action_hand = action[:, 7:]

        if self.relative_to_ee:
            # Re-express obs + action poses (and, if relative_pointcloud, the cloud) in the
            # LAST-OBSERVED-EE frame (index pad_before). ARMlab / Diffusion-Policy convention.
            base_rot = quat_to_rotmat(agent_pos[self.pad_before, 3:7])   # (3, 3)
            base_pos = agent_pos[self.pad_before, :3]                    # (3,)
            observed_pos_out = (observed_pos - base_pos) @ base_rot
            observed_rot_out = rotmat_to_6d(base_rot.T @ observed_rot)
            action_pos_out = (action_pos - base_pos) @ base_rot
            action_rot_out = rotmat_to_6d(base_rot.T @ action_rot)
            if self.relative_pointcloud:
                # positions translate + rotate into the last-observed-EE frame; force vectors rotate only.
                # Any extra per-point channels (cols 6:, e.g. DINO features) are pose-invariant semantics,
                # so they are carried through UNCHANGED (must not be rotated or dropped).
                pc_pos = (pc[..., :3] - base_pos) @ base_rot
                pc_force = pc[..., 3:6] @ base_rot
                extra = pc[..., 6:]                              # DINO / other features -> passthrough
                pc = np.concatenate([pc_pos, pc_force, extra], axis=-1).astype(np.float32)
        else:
            # Keep the stored (robot-EE first-anchored) poses + cloud unchanged; only quat -> 6D.
            # The predicted action is then an ABSOLUTE first-anchored EE-pose target; the cloud is
            # already in that frame, so relative_pointcloud is ignored here (frame consistency).
            observed_pos_out = observed_pos
            observed_rot_out = rotmat_to_6d(observed_rot)
            action_pos_out = action_pos
            action_rot_out = rotmat_to_6d(action_rot)

        agent_pos_out = np.concatenate(
            [observed_pos_out, observed_rot_out, observed_hand], axis=-1).astype(np.float32)  # (T, 26)
        action_out = np.concatenate(
            [action_pos_out, action_rot_out, action_hand], axis=-1).astype(np.float32)        # (T, 26)

        obs = {
            'point_cloud': pc,
            'agent_pos': agent_pos_out,
        }
        return {'obs': obs, 'action': action_out}

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        to_torch = lambda x: torch.from_numpy(x) if x.__class__.__name__ == 'ndarray' else x
        return dict_apply(data, to_torch)
