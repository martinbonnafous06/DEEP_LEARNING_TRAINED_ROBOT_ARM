"""kinematics_retargeting.py — Module 4 of the Video-to-LIBERO pipeline.

Compose hand poses from camera/table frames into the LIBERO robot base frame,
resample from source FPS to target FPS, smooth trajectories, and compute
Operational Space Control (OSC) action deltas ready for LIBERO playback.

Typical usage::

    python kinematics_retargeting.py \\
        --camera_poses   outputs/camera_poses.npz \\
        --table_transform outputs/table_transform.npz \\
        --hand_poses     outputs/hand_poses.npz \\
        --output_dir     outputs/

Outputs
-------
retargeted.npz
    actions   : (M-1, 7)  — [dx, dy, dz, drx, dry, drz, gripper_state]
    ee_pos    : (M, 3)    — end-effector position in robot base frame (metres)
    ee_quat   : (M, 4)    — end-effector quaternion [x, y, z, w]
    gripper   : (M,)      — continuous gripper value resampled/smoothed
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp


# ---------------------------------------------------------------------------
# Default robot-base → table offset
# Represents the table surface sitting ~0.75 m in front of and 0.05 m below
# the robot's base origin, with no rotation.
# Override via --robot_table_transform_npy.
# ---------------------------------------------------------------------------
_DEFAULT_T_ROBOT_BASE_TO_TABLE: NDArray[np.float64] = np.array(
    [
        [1.0, 0.0, 0.0, 0.75],
        [0.0, 1.0, 0.0, 0.00],
        [0.0, 0.0, 1.0, -0.05],
        [0.0, 0.0, 0.0, 1.00],
    ],
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_inputs(
    camera_poses_path: str,
    table_transform_path: str,
    hand_poses_path: str,
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Load all upstream module outputs needed for retargeting.

    Args:
        camera_poses_path: Path to ``camera_poses.npz`` containing
            ``T_world_to_cam`` of shape ``(N, 4, 4)``.
        table_transform_path: Path to ``table_transform.npz`` containing
            ``R_align`` ``(3, 3)`` and ``t_align`` ``(3,)`` from Module 2.
        hand_poses_path: Path to ``hand_poses.npz`` containing
            ``T_cam_to_hand`` ``(N, 4, 4)``, ``gripper_continuous`` ``(N,)``,
            and ``gripper_binary`` ``(N,)`` from Module 3.

    Returns:
        Tuple of:
        - T_world_to_cam    (N, 4, 4)
        - R_align           (3, 3)
        - t_align           (3,)
        - T_cam_to_hand     (N, 4, 4)
        - gripper_continuous (N,)
        - gripper_binary     (N,)

    Raises:
        FileNotFoundError: If any of the input files do not exist.
        KeyError: If expected keys are missing from the npz archives.
    """
    for path in (camera_poses_path, table_transform_path, hand_poses_path):
        if not Path(path).is_file():
            raise FileNotFoundError(f"Input file not found: {path}")

    cam_data = np.load(camera_poses_path)
    if isinstance(cam_data, np.lib.npyio.NpzFile):
        if "T_world_to_cam" in cam_data:
            T_world_to_cam = cam_data["T_world_to_cam"].astype(np.float64)
        elif "poses" in cam_data:
            T_world_to_cam = cam_data["poses"].astype(np.float64)
        else:
            raise KeyError(f"Expected 'T_world_to_cam' or 'poses' in {camera_poses_path}, found {list(cam_data.keys())}")
    else:
        T_world_to_cam = cam_data.astype(np.float64)

    tab_data = np.load(table_transform_path)
    hand_data = np.load(hand_poses_path)

    R_align: NDArray[np.float64] = tab_data["R_align"].astype(np.float64)
    t_align: NDArray[np.float64] = tab_data["t_align"].astype(np.float64)
    T_cam_to_hand: NDArray[np.float64] = hand_data["T_cam_to_hand"].astype(np.float64)
    gripper_continuous: NDArray[np.float64] = hand_data["gripper_continuous"].astype(np.float64)
    gripper_binary: NDArray[np.float64] = hand_data.get("gripper_binary", np.zeros_like(gripper_continuous)).astype(np.float64)

    # Frame alignment check
    n_min = min(len(T_world_to_cam), len(T_cam_to_hand))
    T_world_to_cam = T_world_to_cam[:n_min]
    T_cam_to_hand = T_cam_to_hand[:n_min]
    gripper_continuous = gripper_continuous[:n_min]
    gripper_binary = gripper_binary[:n_min]

    return (
        T_world_to_cam,
        R_align,
        t_align,
        T_cam_to_hand,
        gripper_continuous,
        gripper_binary,
    )


# ---------------------------------------------------------------------------
# Transform composition
# ---------------------------------------------------------------------------


def build_T_table_to_world(
    R_align: NDArray[np.float64],
    t_align: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Construct the homogeneous table-to-world transform from Module 2 output.

    Module 2 produces an alignment that maps *world* coordinates into the
    *table* frame, i.e. ``p_table = R_align @ p_world + t_align``.
    This function inverts that relationship to produce ``T_table_to_world``
    so that downstream composition gives end-effector poses in world space.

    .. math::
        T_{table \\to world} = \\begin{bmatrix}
            R_{align}^\\top & -R_{align}^\\top t_{align} \\\\
            0^\\top          & 1
        \\end{bmatrix}

    Args:
        R_align: Rotation matrix ``(3, 3)`` — world-to-table rotation from
            Module 2.
        t_align: Translation vector ``(3,)`` — world-to-table translation from
            Module 2.

    Returns:
        ``T_table_to_world`` as a ``(4, 4)`` homogeneous matrix.
    """
    T = np.eye(4, dtype=np.float64)
    R_inv = R_align.T
    t_inv = -R_inv @ t_align
    T[:3, :3] = R_inv
    T[:3, 3] = t_inv
    return T


def compose_table_to_hand(
    T_world_to_cam: NDArray[np.float64],
    T_cam_to_hand: NDArray[np.float64],
    R_align: NDArray[np.float64],
    t_align: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compose the full chain to get end-effector poses in the table frame.

    Coordinate chain:
        p_hand: point in hand/wrist frame
        p_cam = T_cam_to_hand @ p_hand
        p_world = (T_world_to_cam)^{-1} @ p_cam = T_cam_to_world @ p_cam
        p_table = T_world_to_table @ p_world

    Resulting in:
        T_table_to_hand = T_world_to_table @ (T_world_to_cam)^{-1} @ T_cam_to_hand

    Args:
        T_world_to_cam: World-to-camera transforms ``(N, 4, 4)``.
        T_cam_to_hand: Camera-to-hand transforms ``(N, 4, 4)``.
        R_align: Rotation matrix ``(3, 3)`` from Module 2 (world-to-table).
        t_align: Translation vector ``(3,)`` from Module 2 (world-to-table).

    Returns:
        ``T_table_to_hand`` of shape ``(N, 4, 4)``.
    """
    T_world_to_table = np.eye(4, dtype=np.float64)
    T_world_to_table[:3, :3] = R_align
    T_world_to_table[:3, 3] = t_align

    # Invert world-to-cam to obtain camera-to-world (N, 4, 4)
    N = T_world_to_cam.shape[0]
    R = T_world_to_cam[:, :3, :3]
    t = T_world_to_cam[:, :3, 3]
    R_inv = R.transpose(0, 2, 1)
    t_inv = -np.einsum("nij,nj->ni", R_inv, t)

    T_cam_to_world = np.tile(np.eye(4, dtype=np.float64), (N, 1, 1))
    T_cam_to_world[:, :3, :3] = R_inv
    T_cam_to_world[:, :3, 3] = t_inv

    # Hand pose in world coordinates: (N, 4, 4)
    T_world_to_hand = np.matmul(T_cam_to_world, T_cam_to_hand)

    # Hand pose in table coordinates: (N, 4, 4)
    T_table_to_hand = np.matmul(T_world_to_table[np.newaxis, :, :], T_world_to_hand)

    return T_table_to_hand


def apply_robot_base_transform(
    T_table_to_hand: NDArray[np.float64],
    T_robot_base_to_table: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Map table-frame EEF poses into the robot base frame.

    Computes::

        T_robot_to_eef = T_robot_base_to_table @ T_table_to_hand

    Args:
        T_table_to_hand: EEF poses in table frame ``(N, 4, 4)``.
        T_robot_base_to_table: Constant transform ``(4, 4)`` from robot base
            to table (hardcoded default or loaded from file).

    Returns:
        ``T_robot_to_eef`` of shape ``(N, 4, 4)``.
    """
    return np.matmul(T_robot_base_to_table[np.newaxis], T_table_to_hand)  # (N, 4, 4)


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


def resample_trajectory(
    T_robot_to_eef: NDArray[np.float64],
    gripper: NDArray[np.float64],
    src_fps: float,
    tgt_fps: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Resample the EEF trajectory and gripper signal to a new frame rate.

    Translation channels are resampled with a cubic spline.
    Rotation channels are resampled with SLERP (via ``scipy.spatial.transform``).
    Gripper is resampled with a cubic spline and clipped to ``[0, 1]``.

    Args:
        T_robot_to_eef: EEF poses in robot base frame ``(N, 4, 4)``.
        gripper: Continuous gripper values ``(N,)`` in ``[0, 1]``.
        src_fps: Source frames per second (e.g. 30).
        tgt_fps: Target frames per second (e.g. 20).

    Returns:
        Tuple of:
        - ``T_resampled`` ``(M, 4, 4)`` — resampled EEF transforms.
        - ``gripper_resampled`` ``(M,)`` — resampled gripper.
    """
    N = T_robot_to_eef.shape[0]

    # Time axes
    t_src = np.arange(N, dtype=np.float64) / src_fps
    duration = t_src[-1]
    M = max(2, int(np.round(duration * tgt_fps)) + 1)
    t_tgt = np.linspace(0.0, duration, M)

    # --- Translation (cubic spline) ---
    positions = T_robot_to_eef[:, :3, 3]  # (N, 3)
    cs_pos = CubicSpline(t_src, positions, axis=0)
    pos_resampled: NDArray[np.float64] = cs_pos(t_tgt)  # (M, 3)

    # --- Rotation (SLERP) ---
    rot_matrices = T_robot_to_eef[:, :3, :3]  # (N, 3, 3)
    rots = Rotation.from_matrix(rot_matrices)
    slerp = Slerp(t_src, rots)
    rot_resampled: Rotation = slerp(t_tgt)  # M rotations

    # --- Gripper (cubic spline) ---
    cs_grip = CubicSpline(t_src, gripper)
    gripper_resampled: NDArray[np.float64] = np.clip(cs_grip(t_tgt), 0.0, 1.0)

    # Rebuild (M, 4, 4) homogeneous matrices
    T_resampled = np.tile(np.eye(4, dtype=np.float64), (M, 1, 1))
    T_resampled[:, :3, :3] = rot_resampled.as_matrix()
    T_resampled[:, :3, 3] = pos_resampled

    return T_resampled, gripper_resampled


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------


def smooth_trajectory(
    T_resampled: NDArray[np.float64],
    gripper_resampled: NDArray[np.float64],
    window_length: int = 15,
    polyorder: int = 3,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Smooth the resampled trajectory with Savitzky-Golay filtering.

    Translation channels are smoothed directly.  Rotation channels are
    converted to rotation-vector representation, smoothed, then converted back
    to rotation matrices.  Gripper is also smoothed and clipped to ``[0, 1]``.

    If the trajectory is shorter than ``window_length`` the window is
    automatically reduced to the largest valid odd integer <= ``M``.

    Args:
        T_resampled: Resampled EEF transforms ``(M, 4, 4)``.
        gripper_resampled: Resampled gripper ``(M,)``.
        window_length: Savitzky-Golay window length (must be odd). Default 15.
        polyorder: Savitzky-Golay polynomial order. Default 3.

    Returns:
        Tuple of:
        - ``T_smooth`` ``(M, 4, 4)`` — smoothed EEF transforms.
        - ``gripper_smooth`` ``(M,)`` — smoothed gripper.
    """
    M = T_resampled.shape[0]

    # Guard against short trajectories
    if M <= 3:
        return T_resampled.copy(), gripper_resampled.copy()

    # Adapt window to trajectory length: must be odd and > polyorder
    max_wl = M if M % 2 != 0 else M - 1
    wl = min(window_length, max_wl)
    if wl % 2 == 0:
        wl -= 1

    actual_polyorder = min(polyorder, wl - 1)
    if actual_polyorder < 1 or wl <= actual_polyorder:
        return T_resampled.copy(), gripper_resampled.copy()

    def _savgol(arr: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply Savitzky-Golay along axis-0 for each column independently."""
        return savgol_filter(arr, window_length=wl, polyorder=actual_polyorder, axis=0)

    # --- Translation ---
    pos_smooth: NDArray[np.float64] = _savgol(T_resampled[:, :3, 3])  # (M, 3)

    # --- Rotation via sign-aligned quaternions (avoids rotvec branch-cut jumps) ---
    rots = Rotation.from_matrix(T_resampled[:, :3, :3])
    quats = rots.as_quat()  # (M, 4) [x, y, z, w]

    for i in range(1, M):
        if np.dot(quats[i], quats[i - 1]) < 0.0:
            quats[i] = -quats[i]

    quats_smooth = _savgol(quats)
    q_norms = np.linalg.norm(quats_smooth, axis=-1, keepdims=True)
    q_norms[q_norms == 0] = 1.0
    quats_smooth /= q_norms
    rots_smooth = Rotation.from_quat(quats_smooth)

    # --- Gripper ---
    gripper_smooth: NDArray[np.float64] = np.clip(
        _savgol(gripper_resampled[:, np.newaxis]).squeeze(-1), 0.0, 1.0
    )

    # Reconstruct transforms
    T_smooth = np.tile(np.eye(4, dtype=np.float64), (M, 1, 1))
    T_smooth[:, :3, :3] = rots_smooth.as_matrix()
    T_smooth[:, :3, 3] = pos_smooth

    return T_smooth, gripper_smooth


# ---------------------------------------------------------------------------
# OSC action deltas
# ---------------------------------------------------------------------------


def compute_osc_actions(
    T_smooth: NDArray[np.float64],
    gripper_smooth: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute per-timestep OSC action deltas from the smoothed trajectory.

    Action vector layout (7-DOF)::

        [dx, dy, dz, drx, dry, drz, gripper_state]

    where:
    - ``(dx, dy, dz)`` = ``p[t+1] - p[t]`` (translation delta, metres)
    - ``(drx, dry, drz)`` = ``rotvec(R[t+1] @ R[t].T)`` (rotation delta)
    - ``gripper_state`` = Franka Panda gripper action in [-1, 1] (+1 = closed, -1 = open)

    Args:
        T_smooth: Smoothed EEF transforms ``(M, 4, 4)``.
        gripper_smooth: Smoothed aperture signal ``(M,)`` in [0, 1] (0=closed, 1=open).

    Returns:
        ``actions`` array of shape ``(M-1, 7)``.
    """
    positions = T_smooth[:, :3, 3]   # (M, 3)
    rot_matrices = T_smooth[:, :3, :3]  # (M, 3, 3)

    delta_p = positions[1:] - positions[:-1]  # (M-1, 3)

    # delta_R[t] = R[t+1] @ R[t].T
    R_next = rot_matrices[1:]                          # (M-1, 3, 3)
    R_curr = rot_matrices[:-1]                         # (M-1, 3, 3)
    delta_R = np.matmul(R_next, R_curr.transpose(0, 2, 1))  # (M-1, 3, 3)
    delta_rotvec: NDArray[np.float64] = Rotation.from_matrix(delta_R).as_rotvec()  # (M-1, 3)

    # Map aperture in [0, 1] (0=closed, 1=open) to Robosuite/LIBERO action [-1, 1] (+1=closed, -1=open)
    gripper_target = gripper_smooth[1:]
    gripper_action = 1.0 - 2.0 * np.clip(gripper_target, 0.0, 1.0)

    actions: NDArray[np.float64] = np.concatenate(
        [delta_p, delta_rotvec, gripper_action[:, np.newaxis]], axis=1
    )
    return actions


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def save_outputs(
    output_dir: str,
    actions: NDArray[np.float64],
    T_smooth: NDArray[np.float64],
    gripper_smooth: NDArray[np.float64],
) -> str:
    """Save retargeted trajectory to ``retargeted.npz``.

    Saved arrays:
    - ``actions``   ``(M-1, 7)``
    - ``ee_pos``    ``(M-1, 3)``   — aligned position in robot base frame (metres)
    - ``ee_quat``   ``(M-1, 4)``   — aligned quaternion ``[x, y, z, w]``
    - ``gripper``   ``(M-1,)``
    - ``ee_pos_all`` ``(M, 3)``
    - ``ee_quat_all`` ``(M, 4)``

    Args:
        output_dir: Directory where ``retargeted.npz`` will be written.
        actions: OSC action deltas ``(M-1, 7)``.
        T_smooth: Smoothed EEF transforms ``(M, 4, 4)``.
        gripper_smooth: Smoothed gripper signal ``(M,)``.

    Returns:
        Absolute path to the saved file.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = str(Path(output_dir) / "retargeted.npz")
    n_actions = len(actions)

    ee_pos = T_smooth[:n_actions, :3, 3]
    ee_quat = Rotation.from_matrix(T_smooth[:n_actions, :3, :3]).as_quat()

    np.savez(
        out_path,
        actions=actions.astype(np.float32),
        ee_pos=ee_pos.astype(np.float32),
        ee_quat=ee_quat.astype(np.float32),
        gripper=gripper_smooth[:n_actions].astype(np.float32),
        ee_pos_all=T_smooth[:, :3, 3].astype(np.float32),
        ee_quat_all=Rotation.from_matrix(T_smooth[:, :3, :3]).as_quat().astype(np.float32),
    )

    # Save trajectory_world_20hz.npy (Context Roadmap Stage 4 deliverable)
    np.save(Path(output_dir) / "trajectory_world_20hz.npy", T_smooth.astype(np.float64))

    return out_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_retargeting(
    camera_poses: str,
    table_transform: str,
    hand_poses: str,
    output_dir: str,
    src_fps: float = 30.0,
    tgt_fps: float = 20.0,
    robot_table_transform_npy: str | None = None,
) -> str:
    """Execute the full kinematics retargeting pipeline.

    Args:
        camera_poses: Path to ``camera_poses.npz``.
        table_transform: Path to ``table_transform.npz``.
        hand_poses: Path to ``hand_poses.npz``.
        output_dir: Directory to write ``retargeted.npz``.
        src_fps: Source video frame rate in Hz. Default 30.
        tgt_fps: Target robot control rate in Hz. Default 20.
        robot_table_transform_npy: Optional path to a ``.npy`` file containing
            a ``(4, 4)`` transform from robot base to table.  If ``None``, the
            hardcoded default is used.

    Returns:
        Path to the saved ``retargeted.npz`` file.

    Raises:
        FileNotFoundError: If input files or the optional transform file are
            missing.
        ValueError: If the loaded robot-base transform is not ``(4, 4)``.
    """
    print("[Module 4] Loading inputs ...")
    (
        T_world_to_cam,
        R_align,
        t_align,
        T_cam_to_hand,
        gripper_continuous,
        _gripper_binary,
    ) = load_inputs(camera_poses, table_transform, hand_poses)

    N = T_world_to_cam.shape[0]
    print(f"  Loaded {N} frames from upstream modules.")

    # --- Optional robot-base transform ---
    if robot_table_transform_npy is not None:
        if not Path(robot_table_transform_npy).is_file():
            raise FileNotFoundError(
                f"Robot-table transform file not found: {robot_table_transform_npy}"
            )
        T_robot_base_to_table: NDArray[np.float64] = np.load(
            robot_table_transform_npy
        ).astype(np.float64)
        if T_robot_base_to_table.shape != (4, 4):
            raise ValueError(
                f"Expected (4,4) robot-base transform, got {T_robot_base_to_table.shape}"
            )
        print(f"  Using robot-table transform from: {robot_table_transform_npy}")
    else:
        T_robot_base_to_table = _DEFAULT_T_ROBOT_BASE_TO_TABLE.copy()
        print("  Using default hardcoded robot-table transform.")

    # --- Step 1: Compose transforms ---
    print("[Module 4] Step 1: Composing transforms ...")
    T_table_to_hand = compose_table_to_hand(
        T_world_to_cam, T_cam_to_hand, R_align, t_align
    )

    # --- Step 2: Map to robot base frame ---
    print("[Module 4] Step 2: Mapping to robot base frame ...")
    T_robot_to_eef = apply_robot_base_transform(T_table_to_hand, T_robot_base_to_table)

    # --- Step 3: Resample ---
    print(f"[Module 4] Step 3: Resampling {src_fps} Hz -> {tgt_fps} Hz ...")
    T_resampled, gripper_resampled = resample_trajectory(
        T_robot_to_eef, gripper_continuous, src_fps, tgt_fps
    )
    M = T_resampled.shape[0]
    print(f"  Resampled trajectory: {N} frames -> {M} frames.")

    # --- Step 4: Smooth ---
    print("[Module 4] Step 4: Smoothing trajectory ...")
    T_smooth, gripper_smooth = smooth_trajectory(T_resampled, gripper_resampled)

    # --- Step 5: Compute OSC action deltas ---
    print("[Module 4] Step 5: Computing OSC action deltas ...")
    actions = compute_osc_actions(T_smooth, gripper_smooth)
    print(f"  Actions shape: {actions.shape}")

    # --- Step 6: Save ---
    print("[Module 4] Step 6: Saving outputs ...")
    out_path = save_outputs(output_dir, actions, T_smooth, gripper_smooth)
    print(f"  Saved -> {out_path}")

    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for the CLI.

    Returns:
        Configured ``argparse.ArgumentParser`` instance.
    """
    parser = argparse.ArgumentParser(
        prog="kinematics_retargeting",
        description=(
            "Module 4 -- Kinematics Retargeting: compose hand poses into the "
            "LIBERO robot base frame, resample, smooth, and compute OSC deltas."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--camera_poses",
        type=str,
        required=True,
        help="Path to camera_poses.npz (T_world_to_cam, shape N x 4 x 4).",
    )
    parser.add_argument(
        "--table_transform",
        type=str,
        required=True,
        help="Path to table_transform.npz (R_align 3x3, t_align 3).",
    )
    parser.add_argument(
        "--hand_poses",
        type=str,
        required=True,
        help=(
            "Path to hand_poses.npz "
            "(T_cam_to_hand N x 4 x 4, gripper_continuous N, gripper_binary N)."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory where retargeted.npz will be saved.",
    )
    parser.add_argument(
        "--src_fps",
        type=float,
        default=30.0,
        help="Source frames per second of the input trajectory.",
    )
    parser.add_argument(
        "--tgt_fps",
        type=float,
        default=20.0,
        help="Target frames per second for the resampled trajectory.",
    )
    parser.add_argument(
        "--robot_table_transform_npy",
        type=str,
        default=None,
        help=(
            "Optional path to a .npy file containing a (4,4) "
            "T_robot_base_to_table transform. Uses hardcoded default if omitted."
        ),
    )
    return parser


if __name__ == "__main__":
    _parser = _build_parser()
    _args = _parser.parse_args()

    try:
        _out = run_retargeting(
            camera_poses=_args.camera_poses,
            table_transform=_args.table_transform,
            hand_poses=_args.hand_poses,
            output_dir=_args.output_dir,
            src_fps=_args.src_fps,
            tgt_fps=_args.tgt_fps,
            robot_table_transform_npy=_args.robot_table_transform_npy,
        )
        print(f"\nRetargeting complete. Output: {_out}")
        sys.exit(0)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        sys.exit(1)
