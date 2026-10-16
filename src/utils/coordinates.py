"""src.utils.coordinates — 3D geometric transformations and kinematics math.

This module provides SE(3) transformation composition, matrix inversion,
twist-to-matrix conversions, vector alignment via Rodrigues' formula,
table-plane alignment, and Operational Space Control (OSC) delta calculation.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation


def invert_transform_se3(T: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Invert one or a batch of homogeneous 4x4 SE(3) matrices.

    For a rigid transformation matrix::

        T = [[R, t],
             [0, 1]]

    The exact analytical inverse is::

        T^{-1} = [[R^T, -R^T t],
                  [0,   1     ]]

    Args:
        T: Array of shape (4, 4) or (N, 4, 4) representing rigid SE(3) transforms.

    Returns:
        Array of the same shape representing the inverse transforms.
    """
    if T.ndim == 2:
        R = T[:3, :3]
        t = T[:3, 3]
        R_inv = R.T
        t_inv = -R_inv @ t
        T_inv = np.eye(4, dtype=np.float64)
        T_inv[:3, :3] = R_inv
        T_inv[:3, 3] = t_inv
        return T_inv
    elif T.ndim == 3:
        N = T.shape[0]
        R = T[:, :3, :3]  # (N, 3, 3)
        t = T[:, :3, 3]   # (N, 3)
        R_inv = R.transpose(0, 2, 1)  # (N, 3, 3)
        t_inv = -np.einsum("nij,nj->ni", R_inv, t)  # (N, 3)
        T_inv = np.tile(np.eye(4, dtype=np.float64), (N, 1, 1))
        T_inv[:, :3, :3] = R_inv
        T_inv[:, :3, 3] = t_inv
        return T_inv
    else:
        raise ValueError(f"Expected T of shape (4, 4) or (N, 4, 4), got {T.shape}")


def twist_to_matrix(twists: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Convert an (N, 7) array of [tx, ty, tz, qx, qy, qz, qw] to (N, 4, 4).

    Args:
        twists: Array of shape (N, 7) where each row is [tx, ty, tz, qx, qy, qz, qw].

    Returns:
        Homogeneous transformation matrices of shape (N, 4, 4).
    """
    if twists.ndim == 1 and twists.shape[0] == 7:
        twists = twists[np.newaxis, :]

    n = twists.shape[0]
    mats = np.tile(np.eye(4, dtype=np.float64), (n, 1, 1))

    tx, ty, tz = twists[:, 0], twists[:, 1], twists[:, 2]
    qx, qy, qz, qw = twists[:, 3], twists[:, 4], twists[:, 5], twists[:, 6]

    # Normalize quaternions
    norms = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    norms[norms == 0] = 1.0
    qx, qy, qz, qw = qx / norms, qy / norms, qz / norms, qw / norms

    # Rotation matrix from unit quaternion (Hamilton convention)
    mats[:, 0, 0] = 1 - 2 * (qy**2 + qz**2)
    mats[:, 0, 1] = 2 * (qx * qy - qz * qw)
    mats[:, 0, 2] = 2 * (qx * qz + qy * qw)
    mats[:, 1, 0] = 2 * (qx * qy + qz * qw)
    mats[:, 1, 1] = 1 - 2 * (qx**2 + qz**2)
    mats[:, 1, 2] = 2 * (qy * qz - qx * qw)
    mats[:, 2, 0] = 2 * (qx * qz - qy * qw)
    mats[:, 2, 1] = 2 * (qy * qz + qx * qw)
    mats[:, 2, 2] = 1 - 2 * (qx**2 + qy**2)

    mats[:, 0, 3] = tx
    mats[:, 1, 3] = ty
    mats[:, 2, 3] = tz

    return mats


def rodrigues_align_vectors(
    v_from: npt.NDArray[np.float64],
    v_to: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute 3x3 rotation matrix that rotates unit vector v_from onto v_to.

    Uses Rodrigues' rotation formula.

    Args:
        v_from: Source 3D vector.
        v_to: Target 3D vector.

    Returns:
        3x3 rotation matrix R such that R @ (v_from / ||v_from||) = (v_to / ||v_to||).
    """
    u_from = v_from / np.linalg.norm(v_from)
    u_to = v_to / np.linalg.norm(v_to)

    v = np.cross(u_from, u_to)
    s = np.linalg.norm(v)
    c_dot = float(np.dot(u_from, u_to))

    if s < 1e-9:
        if c_dot > 0:
            return np.eye(3, dtype=np.float64)
        else:
            # 180-degree rotation around an arbitrary orthogonal vector
            ortho = np.array([1.0, 0.0, 0.0])
            if abs(u_from[0]) > 0.9:
                ortho = np.array([0.0, 1.0, 0.0])
            axis = np.cross(u_from, ortho)
            axis /= np.linalg.norm(axis)
            return 2.0 * np.outer(axis, axis) - np.eye(3, dtype=np.float64)

    vx = np.array([
        [0.0,   -v[2],  v[1]],
        [v[2],   0.0,  -v[0]],
        [-v[1],  v[0],  0.0],
    ], dtype=np.float64)

    R = np.eye(3, dtype=np.float64) + vx + (vx @ vx) * ((1.0 - c_dot) / (s**2))
    return R


def compute_table_alignment(
    plane_model: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute rigid transformation mapping world points into the table coordinate frame.

    Given a plane ``ax + by + cz + d = 0`` in the world coordinate frame:
    1. Normalizes the plane normal n = [a, b, c].
    2. Ensures n points upward (+Z component >= 0).
    3. Finds rotation R_align that rotates n to [0, 0, 1]^T.
    4. Translates along Z so that the table surface lies at Z = 0.

    For any point p_world on the table surface:
        p_table = T_world_to_table @ [p_world, 1]^T
        satisfies p_table[2] == 0.

    Args:
        plane_model: 1-D array [a, b, c, d] satisfying ax + by + cz + d = 0.

    Returns:
        Tuple of (T_world_to_table, R_align, t_align) where:
        - T_world_to_table: (4, 4) homogeneous matrix mapping world points to table frame.
        - R_align: (3, 3) rotation block.
        - t_align: (3,) translation block.
    """
    a, b, c, d = plane_model
    normal = np.array([a, b, c], dtype=np.float64)
    normal_len = np.linalg.norm(normal)
    if normal_len < 1e-9:
        raise ValueError(f"Degenerate plane normal with near-zero length: {plane_model}")
    normal /= normal_len
    d = float(d / normal_len)

    # Ensure normal points upward (+Z)
    if normal[2] < 0:
        normal = -normal
        d = -d

    target_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    R_align = rodrigues_align_vectors(normal, target_up)

    # Plane origin in world coordinates: projection of world origin onto plane
    # satisfies normal . p + d = 0  ->  p = -d * normal
    plane_origin_world = -d * normal

    # After rotation, rotated_origin = R_align @ plane_origin_world
    # We want (R_align @ plane_origin_world + t_align)[2] = 0
    rotated_origin = R_align @ plane_origin_world
    t_align = np.array([0.0, 0.0, -rotated_origin[2]], dtype=np.float64)

    T_world_to_table = np.eye(4, dtype=np.float64)
    T_world_to_table[:3, :3] = R_align
    T_world_to_table[:3, 3] = t_align

    return T_world_to_table, R_align, t_align


def compose_hand_to_table(
    T_cam_to_hand: npt.NDArray[np.float64],
    T_world_to_cam: npt.NDArray[np.float64],
    T_world_to_table: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compose transformation chain to compute hand poses in the table coordinate frame.

    Coordinate Transformation Chain:
    - Point in hand frame: p_hand
    - Hand in camera frame: p_cam = T_cam_to_hand @ p_hand
    - Camera in world frame: p_cam = T_world_to_cam @ p_world
      => p_world = (T_world_to_cam)^{-1} @ p_cam = T_cam_to_world @ p_cam
    - World to table frame: p_table = T_world_to_table @ p_world

    Combining the chain gives:
        p_table = T_world_to_table @ (T_world_to_cam)^{-1} @ T_cam_to_hand @ p_hand
        T_table_to_hand = T_world_to_table @ T_cam_to_world @ T_cam_to_hand

    Args:
        T_cam_to_hand: Array of shape (N, 4, 4) from HaMeR.
        T_world_to_cam: Array of shape (N, 4, 4) from DROID-SLAM.
        T_world_to_table: Array of shape (4, 4) from table plane alignment.

    Returns:
        Array of shape (N, 4, 4) representing hand poses in the table coordinate frame.
    """
    N = T_cam_to_hand.shape[0]
    if T_world_to_cam.shape[0] != N:
        raise ValueError(
            f"Frame count mismatch: T_cam_to_hand ({N}) vs T_world_to_cam ({T_world_to_cam.shape[0]})"
        )

    # Invert world-to-cam to get camera-to-world
    T_cam_to_world = invert_transform_se3(T_world_to_cam)  # (N, 4, 4)

    # Hand pose in world coordinates: T_world_to_hand = T_cam_to_world @ T_cam_to_hand
    T_world_to_hand = np.matmul(T_cam_to_world, T_cam_to_hand)  # (N, 4, 4)

    # Hand pose in table coordinates: T_table_to_hand = T_world_to_table @ T_world_to_hand
    T_table_to_hand = np.matmul(T_world_to_table[np.newaxis, :, :], T_world_to_hand)  # (N, 4, 4)

    return T_table_to_hand


def map_table_to_robot_base(
    T_table_to_hand: npt.NDArray[np.float64],
    T_robot_base_to_table: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Map table-frame hand poses into the robot base coordinate frame.

    Computes:
        T_robot_to_eef = T_robot_base_to_table @ T_table_to_hand

    Args:
        T_table_to_hand: Array of shape (N, 4, 4) hand poses in table frame.
        T_robot_base_to_table: (4, 4) transform from robot base to table frame.

    Returns:
        Array of shape (N, 4, 4) end-effector poses in robot base frame.
    """
    return np.matmul(T_robot_base_to_table[np.newaxis, :, :], T_table_to_hand)


def compute_osc_actions(
    T_smooth: npt.NDArray[np.float64],
    gripper_signal: npt.NDArray[np.float64],
    gripper_is_aperture: bool = True,
) -> npt.NDArray[np.float64]:
    """Compute per-timestep Operational Space Control (OSC) delta actions.

    Action vector layout (7-DOF):
        [dx, dy, dz, drx, dry, drz, gripper_action]

    Where:
    - (dx, dy, dz) = p[t+1] - p[t] (translation delta, metres)
    - (drx, dry, drz) = rotvec(R[t+1] @ R[t]^T) (orientation delta, axis-angle)
    - gripper_action = action command for the Franka Panda gripper in [-1, 1],
      where +1.0 = closed and -1.0 = open (Robosuite/LIBERO standard convention).

    Args:
        T_smooth: Array of shape (M, 4, 4) smoothed end-effector poses in robot base frame.
        gripper_signal: Array of shape (M,) gripper values.
        gripper_is_aperture: If True, gripper_signal is aperture in [0, 1] (0=closed, 1=open),
            which is mapped to Robosuite action [-1, 1] via: action = 1.0 - 2.0 * aperture.
            If False, gripper_signal is already in [-1, 1] (+1=closed, -1=open).

    Returns:
        Array of shape (M-1, 7) float64 actions.
    """
    positions = T_smooth[:, :3, 3]       # (M, 3)
    rot_matrices = T_smooth[:, :3, :3]   # (M, 3, 3)

    delta_p = positions[1:] - positions[:-1]  # (M-1, 3)

    # Relative rotation in robot base frame: delta_R = R[t+1] @ R[t]^T
    R_next = rot_matrices[1:]
    R_curr = rot_matrices[:-1]
    delta_R = np.matmul(R_next, R_curr.transpose(0, 2, 1))
    delta_rotvec = Rotation.from_matrix(delta_R).as_rotvec()  # (M-1, 3)

    # Align gripper action to step t (action executed between t and t+1 commanding target state t+1)
    g_target = gripper_signal[1:]
    if gripper_is_aperture:
        # Convert aperture [0, 1] (0=closed, 1=open) -> Robosuite action [-1, 1] (+1=closed, -1=open)
        gripper_action = 1.0 - 2.0 * np.clip(g_target, 0.0, 1.0)
    else:
        gripper_action = np.clip(g_target, -1.0, 1.0)

    actions = np.concatenate(
        [delta_p, delta_rotvec, gripper_action[:, np.newaxis]], axis=1
    )
    return actions
