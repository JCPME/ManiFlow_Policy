#!/usr/bin/env python3
"""Sanity-check whether the trained policy reproduces recorded trajectories.

For one sample from the training dataset:
  1. Feed the obs window through `policy.predict_action`.
  2. Invert the relative action -> absolute two different ways:
       (A) using the dataset's base (last frame of the 64-frame horizon),
       (B) using the runner's base (last of n_obs_steps observation frames).
  3. Compare both to the recorded absolute actions for that sample.

Interpretation:
  - If (A) matches the recording closely, the model is trained well in the
    dataset's coordinate system. The runner's choice of base then disagrees
    with training -> coordinate-frame bug. Fix the dataset or fix the runner
    so they agree.
  - If neither (A) nor (B) is close, the model is undertrained (or some other
    coordinate/normalization step is broken). Train longer / fix R3M
    pretrained=False before chasing geometry.
  - If only (B) is close, the runner's base is the right one and the dataset
    has a subtle bug elsewhere.

Usage:
    cd ~/Handcontrol/ManiFlow_Policy/ManiFlow
    python ../scripts/sanity_check_policy.py \
        --checkpoint data/outputs/coffee_cup_image-maniflow_image_timm_policy-0901_seed0/checkpoints/latest.ckpt
"""

import argparse
import sys
import pathlib

import numpy as np
import torch
from scipy.spatial.transform import Rotation

MANIFLOW_SRC = pathlib.Path(__file__).resolve().parent.parent / 'ManiFlow'
if str(MANIFLOW_SRC) not in sys.path:
    sys.path.insert(0, str(MANIFLOW_SRC))

from maniflow.workspace.train_maniflow_dex_workspace import TrainManiFlowDexWorkspace
from maniflow.dataset.custom_dataset import (
    quat_to_rotmat, rotmat_to_6d, sixd_to_rotmat,
)


N_OBS_STEPS = 2          # must match training config
ACTION_DIMS = 26         # rel_pos(3) + rot6d(6) + hand(17)
RAW_DIMS    = 24         # pos(3) + quat_xyzw(4) + hand(17)


def relative_action_to_absolute(action_rel: np.ndarray,
                                base_pos: np.ndarray,
                                base_rot: np.ndarray) -> np.ndarray:
    """(T, 26) relative -> (T, 24) absolute (pos+quat_xyzw+hand). Mirrors runner."""
    pos_rel = action_rel[:, :3]
    rot_rel = action_rel[:, 3:9]
    hand    = action_rel[:, 9:]
    pos_abs = pos_rel @ base_rot.T + base_pos[None, :]
    R_rel   = sixd_to_rotmat(rot_rel)                  # (T,3,3)
    R_abs   = base_rot[None, :, :] @ R_rel             # (T,3,3)
    quat    = Rotation.from_matrix(R_abs).as_quat()    # xyzw, (T,4)
    return np.concatenate([pos_abs, quat, hand], axis=-1)


def quat_position_err(pos_a: np.ndarray, pos_b: np.ndarray) -> float:
    """Mean L2 position error in metres."""
    return float(np.linalg.norm(pos_a - pos_b, axis=-1).mean())


def quat_angle_err(quat_a: np.ndarray, quat_b: np.ndarray) -> float:
    """Mean rotation error in degrees between two quaternion (xyzw) sequences."""
    R_a = Rotation.from_quat(quat_a).as_matrix()
    R_b = Rotation.from_quat(quat_b).as_matrix()
    R_diff = np.einsum('tij,tkj->tik', R_a, R_b)        # R_a @ R_b^T
    tr = np.einsum('tii->t', R_diff)
    cos = np.clip((tr - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--sample-idx', type=int, default=None,
                        help='Dataset sample to inspect. Default: middle of dataset.')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--only', choices=['model', 'ema_model', 'both'],
                        default='both', help='Which policy to evaluate.')
    args = parser.parse_args()

    # ---- load workspace + policy --------------------------------------------
    print(f'Loading checkpoint: {args.checkpoint}')
    ws = TrainManiFlowDexWorkspace.create_from_checkpoint(args.checkpoint)
    cfg = ws.cfg

    # Instantiate the dataset ONCE — its ReplayBuffer holds ~7 GB of images and
    # we don't want two copies in RAM. Reuse it across model/ema_model.
    import hydra
    import os, pathlib

    # The dataset config uses a relative zarr_path like `data/coffee_cup_expert.zarr`.
    # Training scripts chdir to ManiFlow/ at __main__; we have to mirror that here
    # because we *import* the workspace class instead of running it as __main__.
    maniflow_root = pathlib.Path(__file__).resolve().parent.parent / 'ManiFlow'
    if maniflow_root.is_dir():
        os.chdir(maniflow_root)
        print(f'cwd → {maniflow_root}')

    print('Instantiating dataset (this loads the zarr into memory)…')
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    print(f'Dataset size: {len(dataset)}')

    candidates = []
    if args.only in ('model', 'both'):
        candidates.append(('model', ws.model))
    if args.only in ('ema_model', 'both'):
        candidates.append(('ema_model', ws.ema_model))

    for tag, model in candidates:
        if model is None:
            print(f'\n[skip] {tag} is None')
            continue
        print(f'\n========== Evaluating: {tag} ==========')
        # Move the OTHER model off-GPU to save VRAM
        for other_tag, other_model in candidates:
            if other_tag != tag and other_model is not None:
                other_model.to('cpu')
        model.eval().to(args.device)
        evaluate_one(model, cfg, args, dataset)


def evaluate_one(policy, cfg, args, dataset):

    # ---- relative-action range across a chunk of samples --------------------
    # The policy predicts in the dataset's relative-to-frame[-1] coordinate
    # system. Knowing the actual data range of those relative values tells us
    # the noise-scale we'd expect from an undertrained flow model.
    n_scan = min(200, len(dataset))
    scan_idxs = np.linspace(0, len(dataset) - 1, n_scan).astype(int)
    rel_actions = []
    for i in scan_idxs:
        d = dataset._sample_to_data(dataset.sampler.sample_sequence(i))
        rel_actions.append(d['action'])
    rel_actions = np.concatenate(rel_actions, axis=0)        # (n_scan*64, 26)
    print(f'\n-- Range of RELATIVE actions across {n_scan} samples '
          f'({rel_actions.shape[0]} frames) --')
    rel_labels = (['rel_x','rel_y','rel_z',
                   'r6d_0','r6d_1','r6d_2','r6d_3','r6d_4','r6d_5'] +
                  [f'hand_{i:02d}' for i in range(17)])
    print(f'  {"dim":<8s}  {"min":>9s}  {"max":>9s}  {"span":>9s}  {"std":>9s}')
    for d, name in enumerate(rel_labels):
        lo, hi = rel_actions[:, d].min(), rel_actions[:, d].max()
        print(f'  {name:<8s}  {lo:9.4f}  {hi:9.4f}  {hi - lo:9.4f}  '
              f'{rel_actions[:, d].std():9.4f}')

    # ---- per-dim stats: catches unit mismatches (deg vs rad, etc.) -----------
    rb = dataset.replay_buffer
    ap_all = np.asarray(rb['agent_pos'])     # (N, 24)
    ac_all = np.asarray(rb['action'])        # (N, 24)
    layout = (['ee_x','ee_y','ee_z','q_x','q_y','q_z','q_w'] +
              [f'hand_{i:02d}' for i in range(17)])
    print('\n-- Per-dim range: agent_pos vs action (same layout, should match) --')
    print(f'  {"dim":<10s}  {"agent_pos[min,max]":>22s}   {"action[min,max]":>22s}   delta_span')
    for i, name in enumerate(layout):
        a_lo, a_hi = ap_all[:, i].min(), ap_all[:, i].max()
        c_lo, c_hi = ac_all[:, i].min(), ac_all[:, i].max()
        span_ratio = (a_hi - a_lo) / max(c_hi - c_lo, 1e-9)
        flag = '  <-- !!' if span_ratio > 10 or span_ratio < 0.1 else ''
        print(f'  {name:<10s}  [{a_lo:9.3f}, {a_hi:9.3f}]   '
              f'[{c_lo:9.3f}, {c_hi:9.3f}]   x{span_ratio:6.2f}{flag}')

    idx = args.sample_idx if args.sample_idx is not None else len(dataset) // 2
    print(f'Inspecting sample {idx}')

    # The dataset returns a 64-frame window with all positions/rotations expressed
    # relative to agent_pos[-1] (the LAST frame of the horizon). We re-derive the
    # absolute (raw) sequence by reading directly from the replay buffer, so we
    # can compare predicted-then-inverted-actions back to the recording in
    # absolute coordinates.
    sample_torch = dataset[idx]
    sample_raw   = dataset.sampler.sample_sequence(idx)   # raw 64-frame slice
    raw_agent_pos = sample_raw['agent_pos'].astype(np.float32)    # (64, 24)
    raw_action    = sample_raw['action'].astype(np.float32)       # (64, 24)
    raw_image     = sample_raw['camera_1'].astype(np.float32)     # (64, 448, 448, 3)

    horizon = raw_agent_pos.shape[0]
    print(f'Horizon: {horizon}, N_OBS_STEPS: {N_OBS_STEPS}, '
          f'n_action_steps: {cfg.n_action_steps}')

    # ---- build obs dict exactly like the policy expects ----------------------
    # The dataset already applied the relative-to-frame[-1] transform; we just
    # take the first n_obs_steps frames of its output.
    obs_dict = {
        'agent_pos': sample_torch['obs']['agent_pos'][:N_OBS_STEPS][None].to(args.device),
        'image':     sample_torch['obs']['image'][:N_OBS_STEPS][None].to(args.device),
    }

    with torch.no_grad():
        out = policy.predict_action(obs_dict)
    # `action_pred` covers the full horizon; `action` is the trimmed chunk used
    # for execution. Compare on action_pred (length = horizon).
    action_pred_rel = out['action_pred'][0].cpu().numpy()   # (horizon, 26)
    print(f'action_pred_rel shape: {action_pred_rel.shape}')

    # ---- inversion A: dataset's base (last frame of horizon) -----------------
    base_pos_A = raw_agent_pos[-1, :3]
    base_rot_A = quat_to_rotmat(raw_agent_pos[-1, 3:7])
    action_abs_A = relative_action_to_absolute(action_pred_rel, base_pos_A, base_rot_A)

    # ---- inversion B: runner's base (last observation frame) -----------------
    base_pos_B = raw_agent_pos[N_OBS_STEPS - 1, :3]
    base_rot_B = quat_to_rotmat(raw_agent_pos[N_OBS_STEPS - 1, 3:7])
    action_abs_B = relative_action_to_absolute(action_pred_rel, base_pos_B, base_rot_B)

    # ---- ground truth: recorded actions, absolute ---------------------------
    gt_abs = raw_action                                              # (horizon, 24)

    # ---- pretty print per-segment errors ------------------------------------
    def report(tag, pred, gt):
        pos_err  = quat_position_err(pred[:, :3], gt[:, :3])
        rot_err  = quat_angle_err(pred[:, 3:7],  gt[:, 3:7])
        hand_err = float(np.linalg.norm(pred[:, 7:] - gt[:, 7:], axis=-1).mean())
        print(f'  {tag:35s}  pos={pos_err*1000:7.2f} mm | '
              f'rot={rot_err:6.2f}° | hand_l2={hand_err:.4f}')

    print('\n-- Errors on FULL horizon (length =', horizon, ') --')
    report('A: invert with base = frame[-1]', action_abs_A, gt_abs)
    report('B: invert with base = obs[-1]  ', action_abs_B, gt_abs)

    # also restrict to the n_action_steps chunk the runner would publish
    start = N_OBS_STEPS - 1
    end   = start + cfg.n_action_steps
    print(f'\n-- Errors on the executed chunk frames [{start}:{end}] --')
    report('A: invert with base = frame[-1]', action_abs_A[start:end], gt_abs[start:end])
    report('B: invert with base = obs[-1]  ', action_abs_B[start:end], gt_abs[start:end])

    # ---- magnitude sanity: how big are the predicted relative steps? --------
    print('\n-- Predicted relative-action magnitudes (frame-by-frame, mean) --')
    print(f'  ||rel_pos||        : {np.linalg.norm(action_pred_rel[:, :3], axis=-1).mean():.4f} m')
    print(f'  ||rot6d - identity6d||: '
          f'{np.linalg.norm(action_pred_rel[:, 3:9] - np.array([1,0,0,0,1,0]), axis=-1).mean():.4f}')
    print(f'  hand range         : [{action_pred_rel[:, 9:].min():.3f}, '
          f'{action_pred_rel[:, 9:].max():.3f}]')

    # ---- compare relative-frame prediction vs dataset's relative ground truth
    # In the dataset's coordinate system, both pred and target are anchored at
    # frame[-1]. If the model is well-trained, these should agree directly.
    gt_rel = sample_torch['action'].numpy()                          # (horizon, 26)
    pos_err_rel  = quat_position_err(action_pred_rel[:, :3], gt_rel[:, :3])
    hand_err_rel = float(np.linalg.norm(action_pred_rel[:, 9:] - gt_rel[:, 9:], axis=-1).mean())
    print('\n-- Errors in the DATASET\'s relative frame (apples-to-apples) --')
    print(f'  rel_pos error : {pos_err_rel*1000:7.2f} mm')
    print(f'  hand_l2 error : {hand_err_rel:.4f}')

    # ---- per-dim summary across the horizon --------------------------------
    dim_labels = (['rel_x', 'rel_y', 'rel_z',
                   'r6d_0', 'r6d_1', 'r6d_2', 'r6d_3', 'r6d_4', 'r6d_5'] +
                  [f'hand_{i:02d}' for i in range(17)])
    print('\n-- Per-dim summary across the 64-frame horizon --')
    print(f'  {"dim":<8s}  {"pred[mean,std]":>22s}   {"target[mean,std]":>22s}   {"mean_err":>9s}')
    for d, name in enumerate(dim_labels):
        p_mu, p_sd = action_pred_rel[:, d].mean(), action_pred_rel[:, d].std()
        g_mu, g_sd = gt_rel[:, d].mean(),          gt_rel[:, d].std()
        print(f'  {name:<8s}  [{p_mu:+7.4f}, {p_sd:6.4f}]   '
              f'[{g_mu:+7.4f}, {g_sd:6.4f}]   {p_mu - g_mu:+9.4f}')

    # ---- detailed pred vs target for a few selected frames ------------------
    sample_frames = [0, 16, 32, 48, 63]
    print(f'\n-- Pred vs target at sample frames (dataset relative coords) --')
    for t in sample_frames:
        print(f'\n  Frame {t}:')
        print(f'    rel_pos    pred  ['
              f'{action_pred_rel[t,0]:+7.4f}, {action_pred_rel[t,1]:+7.4f}, '
              f'{action_pred_rel[t,2]:+7.4f}]')
        print(f'               target['
              f'{gt_rel[t,0]:+7.4f}, {gt_rel[t,1]:+7.4f}, {gt_rel[t,2]:+7.4f}]')
        print(f'    rot6d      pred  [' +
              ', '.join(f'{action_pred_rel[t,d]:+6.3f}' for d in range(3, 9)) + ']')
        print(f'               target[' +
              ', '.join(f'{gt_rel[t,d]:+6.3f}' for d in range(3, 9)) + ']')
        print(f'    hand[0:8]  pred  [' +
              ', '.join(f'{action_pred_rel[t,d]:+6.3f}' for d in range(9, 17)) + ']')
        print(f'               target[' +
              ', '.join(f'{gt_rel[t,d]:+6.3f}' for d in range(9, 17)) + ']')
        print(f'    hand[8:17] pred  [' +
              ', '.join(f'{action_pred_rel[t,d]:+6.3f}' for d in range(17, 26)) + ']')
        print(f'               target[' +
              ', '.join(f'{gt_rel[t,d]:+6.3f}' for d in range(17, 26)) + ']')


if __name__ == '__main__':
    main()
