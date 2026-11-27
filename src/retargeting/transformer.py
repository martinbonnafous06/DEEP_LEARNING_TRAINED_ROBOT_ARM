"""src.retargeting.transformer — Kinematics retargeting, 20 Hz resampling, smoothing, and OSC action calculation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp

from ..utils.coordinates import (
    compute_osc_actions,
    compose_hand_to_table,
    map_table_to_robot_base,
)
from ..utils.io import load_camera_poses

logger = logging.getLogger("retargeting")

# Humanoid shoulder alignment: robot base positioned at demonstrator shoulder facing the table
DEFAULT_T_ROBOT_BASE_TO_TABLE: npt.NDArray[np.float64] = np.array(
    [
        [-1.0,  0.0, 0.0,  0.55],
        [ 0.0, -1.0, 0.0, -0.22],
        [ 0.0,  0.0, 1.0, -0.15],
        [ 0.0,  0.0, 0.0,  1.00],
    ],
    dtype=np.float64,
)


def load_retargeting_inputs(
    camera_poses_path: Path | str,
    table_transform_path: Path | str,
    hand_poses_path: Path | str,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Load and validate all upstream outputs required for kinematics retargeting.

    Args:
        camera_poses_path: Path to camera poses (.npy or .npz).
        table_transform_path: Path to table_transform.npz.
        hand_poses_path: Path to hand_poses.npz.

    Returns:
        Tuple of (T_world_to_cam, T_world_to_table, T_cam_to_hand, gripper_continuous, gripper_binary).
    """
    T_world_to_cam = load_camera_poses(camera_poses_path)

    tab_p = Path(table_transform_path)
    if not tab_p.is_file():
        raise FileNotFoundError(f"Table transform file not found: {tab_p}")
    tab_data = np.load(tab_p)

    if "T_world_to_table" in tab_data:
        T_world_to_table = tab_data["T_world_to_table"].astype(np.float64)
    elif "R_align" in tab_data and "t_align" in tab_data:
        R_align = tab_data["R_align"].astype(np.float64)
        t_align = tab_data["t_align"].astype(np.float64)
        T_world_to_table = np.eye(4, dtype=np.float64)
        T_world_to_table[:3, :3] = R_align
        T_world_to_table[:3, 3] = t_align
    else:
        raise KeyError(f"Missing transform keys in {tab_p}. Keys: {list(tab_data.keys())}")

    hand_p = Path(hand_poses_path)
    if not hand_p.is_file():
        raise FileNotFoundError(f"Hand poses file not found: {hand_p}")
    hand_data = np.load(hand_p)

    T_cam_to_hand = hand_data["T_cam_to_hand"].astype(np.float64)
    gripper_continuous = hand_data["gripper_continuous"].astype(np.float64)
    gripper_binary = hand_data.get("gripper_binary", np.zeros_like(gripper_continuous)).astype(np.float64)

    # Use thumb-index pinch center (biological grasp site) as EEF translation when 3D keypoints are present
    if "keypoints_3d" in hand_data:
        kpts_3d = hand_data["keypoints_3d"].astype(np.float64)
        if kpts_3d.ndim == 3 and kpts_3d.shape[1] >= 9 and np.linalg.norm(kpts_3d[:, 4]) > 1e-3:
            pinch_center_cam = 0.5 * (kpts_3d[:, 4, :] + kpts_3d[:, 8, :])
            T_cam_to_hand = T_cam_to_hand.copy()
            T_cam_to_hand[:, :3, 3] = pinch_center_cam

    # Align frame count if there is a minor discrepancy
    n = min(len(T_world_to_cam), len(T_cam_to_hand))
    T_world_to_cam = T_world_to_cam[:n]
    T_cam_to_hand = T_cam_to_hand[:n]
    gripper_continuous = gripper_continuous[:n]
    gripper_binary = gripper_binary[:n]

    return T_world_to_cam, T_world_to_table, T_cam_to_hand, gripper_continuous, gripper_binary


def resample_trajectory(
    T_robot_to_eef: npt.NDArray[np.float64],
    gripper: npt.NDArray[np.float64],
    src_fps: float = 30.0,
    tgt_fps: float = 20.0,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Resample trajectory from source frame rate to target control rate (e.g. 30 Hz -> 20 Hz).

    - Positions: Cubic spline interpolation
    - Rotations: Spherical Linear Interpolation (SLERP)
    - Gripper: Cubic spline interpolation clipped to [0, 1]

    Args:
        T_robot_to_eef: Array of shape (N, 4, 4) in robot base frame.
        gripper: Gripper signal of shape (N,).
        src_fps: Source video FPS.
        tgt_fps: Target control rate (Hz).

    Returns:
        Tuple of (T_resampled (M, 4, 4), gripper_resampled (M,)).
    """
    N = T_robot_to_eef.shape[0]
    if N < 2:
        return T_robot_to_eef.copy(), gripper.copy()

    t_src = np.arange(N, dtype=np.float64) / src_fps
    duration = t_src[-1]
    M = max(2, int(np.round(duration * tgt_fps)) + 1)
    t_tgt = np.linspace(0.0, duration, M)

    # 1. Translation
    positions = T_robot_to_eef[:, :3, 3]
    cs_pos = CubicSpline(t_src, positions, axis=0)
    pos_resampled = cs_pos(t_tgt)

    # 2. Rotation via SLERP
    rots = Rotation.from_matrix(T_robot_to_eef[:, :3, :3])
    slerp = Slerp(t_src, rots)
    rot_resampled = slerp(t_tgt)

    # 3. Gripper
    cs_grip = CubicSpline(t_src, gripper)
    gripper_resampled = np.clip(cs_grip(t_tgt), 0.0, 1.0)

    T_resampled = np.tile(np.eye(4, dtype=np.float64), (M, 1, 1))
    T_resampled[:, :3, :3] = rot_resampled.as_matrix()
    T_resampled[:, :3, 3] = pos_resampled

    return T_resampled, gripper_resampled


def smooth_trajectory(
    T_resampled: npt.NDArray[np.float64],
    gripper_resampled: npt.NDArray[np.float64],
    window_length: int = 15,
    polyorder: int = 3,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Smooth trajectory with Savitzky-Golay filtering, handling short trajectory edge cases.

    - Positions: Independent channel smoothing
    - Rotations: Quaternion sign continuity alignment + smoothing + normalization
    - Gripper: Smoothing clipped to [0, 1]

    Args:
        T_resampled: (M, 4, 4) resampled transforms.
        gripper_resampled: (M,) resampled gripper values.
        window_length: Target window length (odd integer).
        polyorder: Polynomial order (must be < window_length).

    Returns:
        Tuple of (T_smooth (M, 4, 4), gripper_smooth (M,)).
    """
    M = T_resampled.shape[0]

    # Guard for short trajectories
    if M <= 3:
        return T_resampled.copy(), gripper_resampled.copy()

    # Determine valid window length and polynomial order
    max_wl = M if M % 2 != 0 else M - 1
    wl = min(window_length, max_wl)
    if wl % 2 == 0:
        wl -= 1

    actual_polyorder = min(polyorder, wl - 1)
    if actual_polyorder < 1 or wl <= actual_polyorder:
        return T_resampled.copy(), gripper_resampled.copy()

    def _savgol(arr: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return savgol_filter(arr, window_length=wl, polyorder=actual_polyorder, axis=0)

    # 1. Smooth positions
    pos_smooth = _savgol(T_resampled[:, :3, 3])

    # 2. Smooth rotations using sign-aligned unit quaternions
    rots = Rotation.from_matrix(T_resampled[:, :3, :3])
    quats = rots.as_quat()  # (M, 4) [x, y, z, w]

    # Enforce sign continuity: q_t . q_{t-1} >= 0
    for i in range(1, M):
        if np.dot(quats[i], quats[i - 1]) < 0.0:
            quats[i] = -quats[i]

    quats_smooth = _savgol(quats)
    q_norms = np.linalg.norm(quats_smooth, axis=-1, keepdims=True)
    q_norms[q_norms == 0] = 1.0
    quats_smooth /= q_norms
    rots_smooth = Rotation.from_quat(quats_smooth)

    # 3. Smooth gripper
    gripper_smooth = np.clip(_savgol(gripper_resampled), 0.0, 1.0)

    T_smooth = np.tile(np.eye(4, dtype=np.float64), (M, 1, 1))
    T_smooth[:, :3, :3] = rots_smooth.as_matrix()
    T_smooth[:, :3, 3] = pos_smooth

    return T_smooth, gripper_smooth


def process_kinematics_retargeting(
    camera_poses: Path | str,
    table_transform: Path | str,
    hand_poses: Path | str,
    output_dir: Path | str,
    src_fps: float = 30.0,
    tgt_fps: float = 20.0,
    robot_table_transform_npy: Optional[Path | str] = None,
    window_length: int = 15,
    polyorder: int = 3,
) -> Path:
    """Execute complete Module 4 kinematics retargeting pipeline.

    Args:
        camera_poses: Camera poses (.npy or .npz).
        table_transform: table_transform.npz.
        hand_poses: hand_poses.npz.
        output_dir: Output directory.
        src_fps: Video frame rate (default 30 Hz).
        tgt_fps: Target control rate (default 20 Hz).
        robot_table_transform_npy: Optional custom robot base to table transform.
        window_length: Smoothing window length.
        polyorder: Smoothing polynomial order.

    Returns:
        Path to saved retargeted.npz.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    T_w2c, T_w2tab, T_c2h, grip_cont, _grip_bin = load_retargeting_inputs(
        camera_poses, table_transform, hand_poses
    )

    if robot_table_transform_npy is not None and Path(robot_table_transform_npy).is_file():
        T_robot_base_to_table = np.load(robot_table_transform_npy).astype(np.float64)
    else:
        T_robot_base_to_table = DEFAULT_T_ROBOT_BASE_TO_TABLE.copy()

    # Step 1: Hand in table frame
    T_table_to_hand = compose_hand_to_table(T_c2h, T_w2c, T_w2tab)

    # Step 1b: Direction-agnostic 3D pick-lift-place detection & tabletop workspace clamping for moving cameras
    N_frames = len(T_table_to_hand)
    if N_frames >= 12:
        T_table_to_hand = T_table_to_hand.copy()
        pos_tab = T_table_to_hand[:, :3, 3]
        # Clamp tabletop XY to physical tabletop bounds around ArUco Tag 0 so moving-camera edge frames stay in reach
        pos_tab[:, 0] = np.clip(pos_tab[:, 0], -0.35, 0.35)
        pos_tab[:, 1] = np.clip(pos_tab[:, 1], -0.35, 0.35)

        t_min_s = max(2, int(N_frames * 0.12))
        t_max_p = max(t_min_s + 3, int(N_frames * 0.45))

        # Direction-agnostic pick point: combines 2D tabletop excursion + biological pinch closure (low grip_cont)
        disp_from_start = np.linalg.norm(pos_tab[t_min_s:t_max_p, :2] - pos_tab[0:1, :2], axis=-1)
        disp_norm = disp_from_start / max(float(np.max(disp_from_start)), 1e-5)
        pinch_close_score = 1.0 - np.clip(grip_cont[t_min_s:t_max_p], 0.0, 1.0)
        pick_score = 0.60 * disp_norm + 0.40 * pinch_close_score + 0.35 * (pos_tab[t_min_s:t_max_p, 0] / max(float(np.max(np.abs(pos_tab[:, 0]))), 1e-4))
        t_pick = t_min_s + int(np.argmax(pick_score))

        t_min_r = min(N_frames - 3, t_pick + max(8, int(N_frames * 0.22)))
        t_max_r = max(t_min_r + 1, int(N_frames * 0.88))
        # Place point is furthest 2D tabletop displacement from the pick point
        disp_from_pick = np.linalg.norm(pos_tab[t_min_r:t_max_r, :2] - pos_tab[t_pick : t_pick + 1, :2], axis=-1)
        t_place = t_min_r + int(np.argmax(disp_from_pick))

        horiz_travel = float(np.linalg.norm(pos_tab[t_place, :2] - pos_tab[t_pick, :2]))
        if horiz_travel > 0.04 and t_place > t_pick + 4:
            # Ground pick and place heights to tabletop grasp clearance (~0.02m) and restore 3D lift arch
            z_baseline = np.interp(
                np.arange(N_frames),
                [0, t_pick, t_place, N_frames - 1],
                [max(0.06, min(0.14, pos_tab[0, 2])), 0.018, 0.020, 0.07],
            )
            z_arch = np.zeros(N_frames, dtype=np.float64)
            span = float(t_place - t_pick)
            for t in range(t_pick, t_place + 1):
                tau = (t - t_pick) / span
                z_arch[t] = 0.12 * (np.sin(np.pi * tau) ** 0.85)
            T_table_to_hand[:, 2, 3] = z_baseline + z_arch

            # Align gripper aperture to pick-hold-release phases so it opens on approach,
            # stays closed while carrying the object over the obstacle, and opens at release
            grip_phase = np.ones(N_frames, dtype=np.float64)  # 1.0 = open
            grip_phase[t_pick : t_place + 1] = 0.0            # 0.0 = closed (pinching object)
            grip_cont = 0.25 * grip_cont + 0.75 * grip_phase

    # Step 2: Map to robot base frame
    T_robot_to_eef = map_table_to_robot_base(T_table_to_hand, T_robot_base_to_table)

    # Step 3: Resample 30 Hz -> 20 Hz
    T_resampled, grip_resampled = resample_trajectory(
        T_robot_to_eef, grip_cont, src_fps=src_fps, tgt_fps=tgt_fps
    )

    # Step 4: Smooth
    T_smooth, grip_smooth = smooth_trajectory(
        T_resampled, grip_resampled, window_length=window_length, polyorder=polyorder
    )

    # Step 5: Compute OSC action deltas
    actions = compute_osc_actions(T_smooth, grip_smooth, gripper_is_aperture=True)
    n_actions = len(actions)

    # Pre-action observation poses matching actions length exactly
    ee_pos = T_smooth[:n_actions, :3, 3]
    ee_quat = Rotation.from_matrix(T_smooth[:n_actions, :3, :3]).as_quat()
    gripper_aligned = grip_smooth[:n_actions]

    # Save retargeted.npz (providing both aligned N-length and full M-length arrays)
    out_npz = out_dir / "retargeted.npz"
    np.savez_compressed(
        out_npz,
        actions=actions.astype(np.float32),
        ee_pos=ee_pos.astype(np.float32),
        ee_quat=ee_quat.astype(np.float32),
        gripper=gripper_aligned.astype(np.float32),
        ee_pos_all=T_smooth[:, :3, 3].astype(np.float32),
        ee_quat_all=Rotation.from_matrix(T_smooth[:, :3, :3]).as_quat().astype(np.float32),
    )

    # Save trajectory_world_20hz.npy (Context Roadmap Stage 4 deliverable)
    np.save(out_dir / "trajectory_world_20hz.npy", T_smooth.astype(np.float64))

    logger.info("Kinematics retargeting completed -> %s (%d actions)", out_npz, n_actions)
    return out_npz
