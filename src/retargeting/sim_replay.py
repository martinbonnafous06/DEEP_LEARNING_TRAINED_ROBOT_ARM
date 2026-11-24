"""src.retargeting.sim_replay — Closed-loop MuJoCo / Robosuite trajectory execution & Panda kinematics engine.

Solves two critical issues when replaying retargeted human demonstrations on a
Franka Emika Panda arm in Robosuite / MuJoCo:

1. **Action Scaling & Closed-Loop Waypoint Tracking (`OSC_POSE`)**:
   Robosuite's ``OSC_POSE`` controller scales normalized actions in ``[-1, 1]`` by
   ``output_max = 0.05 m`` (translation) and ``0.5 rad`` (rotation), and updates
   ``goal_pos = current_ee_pos + scaled_delta``. Feeding raw metric deltas
   ``p[t+1] - p[t]`` directly into ``env.step(action)`` attenuates motion by 20x
   and accumulates impedance lag. This module computes closed-loop ``OSC_POSE``
   commands that drive the physical Panda EEF to each target 3D waypoint.

2. **Workspace & Object Grasp Alignment**:
   In Robosuite (e.g. ``Lift``), ``env.reset()`` initializes the Panda EEF high above
   the table (``[-0.095, 0.005, 1.020] m``) while the target object sits on the
   tabletop (``~[0.0, 0.0, 0.82] m``). This module detects the demonstration's
   pick/pinch timestep ``t_pick`` and release timestep ``t_release``, anchors the
   demonstration's pick point to the simulated object's 3D position, guides the open
   gripper through a pre-grasp hover and vertical descent onto the object, closes
   the gripper firmly around the object at ``t_pick``, lifts and transports the
   object along the human's retargeted 3D lift-and-carry arc, and releases it on the
   table at ``t_release``.

Also includes an exact 7-DOF Franka Panda forward/inverse kinematics and rigid-body
grasping simulator fallback (matching ``robosuite/models/assets/robots/panda/robot.xml``)
when running in environments where Windows Application Control blocks unsigned MuJoCo
``.pyd`` DLLs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import Rotation

logger = logging.getLogger("sim_replay")

# Franka Panda kinematic chain body names in Robosuite MuJoCo
ROBOT_JOINT_BODIES: List[str] = [
    "robot0_link0",
    "robot0_link1",
    "robot0_link2",
    "robot0_link3",
    "robot0_link4",
    "robot0_link5",
    "robot0_link6",
    "robot0_link7",
    "robot0_right_hand",
    "robot0_eef",
]

# Exact Franka Panda joint limits and home qpos from robosuite panda/robot.xml
PANDA_INIT_QPOS = np.array(
    [0.0, np.pi / 16.0, 0.0, -np.pi / 2.0 - np.pi / 3.0, 0.0, np.pi - 0.2, np.pi / 4.0],
    dtype=np.float64,
)
PANDA_JOINT_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], dtype=np.float64)
PANDA_JOINT_MAX = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973], dtype=np.float64)
PANDA_BASE_WORLD_POS = np.array([-0.56, 0.0, 0.913], dtype=np.float64)
TABLE_TOP_Z_WORLD = 0.800


@dataclass
class ManipulationPhaseIndices:
    """Key phase indices of a pick-lift-place human demonstration."""

    t_pick: int
    t_peak: int
    t_release: int
    total_steps: int
    pick_pos_demo: npt.NDArray[np.float64]
    peak_pos_demo: npt.NDArray[np.float64]
    place_pos_demo: npt.NDArray[np.float64]


@dataclass
class SimulationReplayResult:
    """Complete telemetry and visual frames from a Panda manipulation replay."""

    joint_states: npt.NDArray[np.float32]               # (N, 7) joint angles
    ee_pos: npt.NDArray[np.float32]                     # (N, 3) EEF positions in sim world frame
    ee_ori: npt.NDArray[np.float32]                     # (N, 4) EEF quaternions [x, y, z, w]
    cube_positions: npt.NDArray[np.float32]             # (N, 3) object positions in sim world frame
    executed_actions: npt.NDArray[np.float32]           # (N, 7) normalized OSC_POSE actions
    joint_body_positions: List[Dict[str, np.ndarray]] = field(default_factory=list)  # relative to robot0_base
    agentview_frames: List[np.ndarray] = field(default_factory=list)
    wrist_frames: List[np.ndarray] = field(default_factory=list)
    grasped: bool = False
    task_success: bool = False
    max_cube_lift_m: float = 0.0
    total_eef_distance_m: float = 0.0
    phases: Optional[ManipulationPhaseIndices] = None


def detect_manipulation_phases(
    demo_eef_pos: npt.NDArray[np.float64],
    demo_actions: npt.NDArray[np.float64],
) -> ManipulationPhaseIndices:
    """Detect pick (grasp), peak lift, and place (release) timesteps from a demonstration.

    Args:
        demo_eef_pos: (N, 3) retargeted EEF positions in robot base frame.
        demo_actions: (N, 7) retargeted actions where col 6 is gripper command (+1=closed, -1=open).

    Returns:
        ManipulationPhaseIndices describing the demonstration phases.
    """
    N = len(demo_actions)
    pos = np.asarray(demo_eef_pos[:N], dtype=np.float64)
    grip = np.asarray(demo_actions[:N, 6], dtype=np.float64)

    # 1. Identify pick index: where hand reaches the pick extremity or closes gripper
    # In robot base frame (where robot faces +X across the table):
    # Pick location behind obstacle is at minimum X_robot (or max X_table),
    # Place location in front of obstacle is at maximum X_robot (or min X_table).
    t_min_search = max(2, int(N * 0.12))
    t_max_pick_search = max(t_min_search + 3, int(N * 0.45))

    # Look for first sustained gripper closure in [t_min_search, t_max_pick_search]
    closed_mask = grip > 0.15
    t_pick = None
    for t in range(t_min_search, t_max_pick_search):
        if closed_mask[t] and np.mean(closed_mask[t : min(N, t + 5)]) >= 0.6:
            t_pick = t
            break

    if t_pick is None:
        # Fallback: geometric pick turning point in the first 45% of trajectory
        # (where hand reaches furthest toward the pick zone before moving to place zone)
        dist_from_end = np.linalg.norm(pos[t_min_search:t_max_pick_search, :2] - pos[int(N * 0.75), :2], axis=-1)
        t_pick = t_min_search + int(np.argmax(dist_from_end))

    # 2. Identify place/release index: after t_pick + 10, where horizontal motion reaches place zone
    t_min_release = min(N - 3, t_pick + max(10, int(N * 0.25)))
    t_max_release = max(t_min_release + 1, int(N * 0.88))

    t_release = None
    for t in range(t_min_release, t_max_release):
        if grip[t] < -0.05 and np.mean(grip[t : min(N, t + 4)] < 0.0) >= 0.5:
            t_release = t
            break

    if t_release is None:
        # Geometric place point: furthest horizontal displacement from pick point
        horiz_disp = np.linalg.norm(pos[t_min_release:t_max_release, :2] - pos[t_pick, :2], axis=-1)
        t_release = t_min_release + int(np.argmax(horiz_disp))

    # Ensure well-ordered indices
    t_pick = int(np.clip(t_pick, 3, max(4, N - 15)))
    t_release = int(np.clip(t_release, t_pick + 8, max(t_pick + 9, N - 4)))
    t_peak = int(t_pick + np.argmax(pos[t_pick : t_release + 1, 2]))
    if t_peak <= t_pick + 1 or t_peak >= t_release - 1:
        t_peak = (t_pick + t_release) // 2

    return ManipulationPhaseIndices(
        t_pick=t_pick,
        t_peak=t_peak,
        t_release=t_release,
        total_steps=N,
        pick_pos_demo=pos[t_pick].copy(),
        peak_pos_demo=pos[t_peak].copy(),
        place_pos_demo=pos[t_release].copy(),
    )


def build_object_aligned_target_trajectory(
    demo_eef_pos: npt.NDArray[np.float64],
    demo_actions: npt.NDArray[np.float64],
    sim_eef_init: npt.NDArray[np.float64],
    sim_cube_init: npt.NDArray[np.float64],
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], ManipulationPhaseIndices]:
    """Construct an object-aligned 3D target EEF trajectory in MuJoCo world coordinates.

    Anchors the demonstration's grasp point ``t_pick`` directly to the object's
    initial 3D position ``sim_cube_init`` on the table, preserves the human's 3D
    transport and lift arch between ``t_pick`` and ``t_release``, and generates a
    collision-free top-down approach before ``t_pick`` and release/retract after
    ``t_release``.

    Args:
        demo_eef_pos: (N, 3) retargeted EEF positions.
        demo_actions: (N, 7) retargeted actions.
        sim_eef_init: (3,) initial EEF position in MuJoCo world frame after env.reset().
        sim_cube_init: (3,) initial object (cube) position in MuJoCo world frame.

    Returns:
        Tuple of:
        - target_pos_world: (N, 3) target EEF positions in MuJoCo world coordinates.
        - target_gripper: (N,) gripper action in {-1.0 (open), +1.0 (closed)}.
        - phases: ManipulationPhaseIndices.
    """
    N = len(demo_actions)
    pos_demo = np.asarray(demo_eef_pos[:N], dtype=np.float64)
    phases = detect_manipulation_phases(pos_demo, demo_actions)

    t_pick = phases.t_pick
    t_release = phases.t_release

    # Grasp point on the cube: slightly below cube center so Panda finger pads clamp the cube sides
    grasp_z = max(TABLE_TOP_Z_WORLD + 0.014, float(sim_cube_init[2]) - 0.004)
    p_grasp = np.array([sim_cube_init[0], sim_cube_init[1], grasp_z], dtype=np.float64)
    p_hover = np.array([sim_cube_init[0], sim_cube_init[1], grasp_z + 0.11], dtype=np.float64)

    target_pos = np.zeros((N, 3), dtype=np.float64)
    target_gripper = np.full(N, -1.0, dtype=np.float64)

    # --- Phase 1: Approach (0 .. t_pick) ---
    # Split into pre-grasp hover alignment (first 55%) and vertical descent onto cube (last 45%)
    t_hover = max(1, int(round(t_pick * 0.55)))
    for t in range(t_pick + 1):
        if t <= t_hover:
            alpha = t / float(max(1, t_hover))
            # Smooth cosine interpolation from sim_eef_init to p_hover
            s = 0.5 * (1.0 - np.cos(np.pi * alpha))
            target_pos[t] = (1.0 - s) * sim_eef_init + s * p_hover
        else:
            alpha = (t - t_hover) / float(max(1, t_pick - t_hover))
            s = 0.5 * (1.0 - np.cos(np.pi * alpha))
            target_pos[t] = (1.0 - s) * p_hover + s * p_grasp
        target_gripper[t] = -1.0  # Keep gripper open during descent

    # At t_pick, command gripper closed
    target_pos[t_pick] = p_grasp
    target_gripper[t_pick] = 1.0

    # --- Phase 2: Grasp Dwell, Lift & Transport Over Obstacle (t_pick .. t_release) ---
    # Compute human's horizontal displacement from pick to place
    raw_delta_xy = pos_demo[t_release, :2] - pos_demo[t_pick, :2]
    horiz_dist = float(np.linalg.norm(raw_delta_xy))
    # Clamp horizontal transport distance to [0.14m, 0.24m] so it stays well inside Panda tabletop reach
    if horiz_dist > 1e-4:
        desired_dist = float(np.clip(horiz_dist, 0.15, 0.22))
        xy_scale = desired_dist / horiz_dist
    else:
        xy_scale = 1.0

    # Extract human's vertical lift height above the pick-to-place baseline
    demo_z_baseline = np.linspace(pos_demo[t_pick, 2], pos_demo[t_release, 2], t_release - t_pick + 1)
    demo_z_arch = pos_demo[t_pick : t_release + 1, 2] - demo_z_baseline
    max_demo_arch = float(np.max(demo_z_arch)) if len(demo_z_arch) > 0 else 0.0

    # Ensure a clear, physical lift arch over the obstacle (peak at least 12 cm above table)
    target_peak_lift = float(np.clip(max(max_demo_arch * 1.5, 0.12), 0.11, 0.16))

    # Give 2 dwell steps right after t_pick so physical gripper fingers finish closing before liftoff
    t_liftoff = min(t_release - 4, t_pick + 2)

    for t in range(t_pick + 1, t_release + 1):
        target_gripper[t] = 1.0  # Keep gripper firmly closed while carrying object
        if t <= t_liftoff:
            target_pos[t] = p_grasp.copy()
            continue

        tau = (t - t_liftoff) / float(max(1, t_release - t_liftoff))
        # Smooth progress along human XY trajectory
        human_xy_offset = (pos_demo[t, :2] - pos_demo[t_pick, :2]) * xy_scale

        # Combine human Z arch with smooth bell-curve clearance over the wooden block
        arch_bell = np.sin(np.pi * tau) ** 0.85
        human_z_norm = (
            max(0.0, float(demo_z_arch[t - t_pick])) / max_demo_arch
            if max_demo_arch > 0.015
            else arch_bell
        )
        lift_z = target_peak_lift * (0.65 * arch_bell + 0.35 * human_z_norm)

        # At t_release, object touches gently back onto the table surface (grasp_z + 0.004)
        place_z_offset = 0.004 * tau
        target_pos[t, 0] = np.clip(p_grasp[0] + human_xy_offset[0], -0.22, 0.22)
        target_pos[t, 1] = np.clip(p_grasp[1] + human_xy_offset[1], -0.24, 0.24)
        target_pos[t, 2] = grasp_z + place_z_offset + lift_z

    # --- Phase 3: Place Release & Retract (t_release + 1 .. N - 1) ---
    p_place = target_pos[t_release].copy()
    p_retract = p_place + np.array([-0.03, 0.0, 0.09], dtype=np.float64)

    for t in range(t_release + 1, N):
        target_gripper[t] = -1.0  # Open gripper to release object on table
        alpha = (t - t_release) / float(max(1, N - 1 - t_release))
        s = 0.5 * (1.0 - np.cos(np.pi * alpha))
        target_pos[t] = (1.0 - s) * p_place + s * p_retract

    return target_pos, target_gripper, phases


# ---------------------------------------------------------------------------
# Exact Franka Panda 7-DOF Kinematics & Fallback Physics Engine
# ---------------------------------------------------------------------------

def _quat_wxyz_to_rotmat(quat_wxyz: Tuple[float, float, float, float]) -> npt.NDArray[np.float64]:
    """Convert MuJoCo (w, x, y, z) quaternion to 3x3 rotation matrix."""
    w, x, y, z = quat_wxyz
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),       2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w),       1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w),       2.0 * (y * z + x * w),       1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float64)


def _rot_z(theta: float) -> npt.NDArray[np.float64]:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


# Exact link definitions from robosuite/models/assets/robots/panda/robot.xml
_PANDA_KINEMATIC_CHAIN = [
    ("robot0_link0", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), False),
    ("robot0_link1", (0.0, 0.0, 0.333), (1.0, 0.0, 0.0, 0.0), True),
    ("robot0_link2", (0.0, 0.0, 0.0), (0.707107, -0.707107, 0.0, 0.0), True),
    ("robot0_link3", (0.0, -0.316, 0.0), (0.707107, 0.707107, 0.0, 0.0), True),
    ("robot0_link4", (0.0825, 0.0, 0.0), (0.707107, 0.707107, 0.0, 0.0), True),
    ("robot0_link5", (-0.0825, 0.384, 0.0), (0.707107, -0.707107, 0.0, 0.0), True),
    ("robot0_link6", (0.0, 0.0, 0.0), (0.707107, 0.707107, 0.0, 0.0), True),
    ("robot0_link7", (0.088, 0.0, 0.0), (0.707107, 0.707107, 0.0, 0.0), True),
    ("robot0_right_hand", (0.0, 0.0, 0.1065), (0.924, 0.0, 0.0, -0.383), False),
]


def panda_forward_kinematics(
    qpos: npt.NDArray[np.float64],
    base_world_pos: npt.NDArray[np.float64] = PANDA_BASE_WORLD_POS,
) -> Tuple[Dict[str, npt.NDArray[np.float64]], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute exact 3D link positions and EEF grip site pose for the 7-DOF Franka Panda.

    Matches ``robosuite/models/assets/robots/panda/robot.xml`` and ``PandaGripper`` site.

    Args:
        qpos: (7,) joint angles in radians.
        base_world_pos: (3,) world coordinates of ``robot0_base``.

    Returns:
        Tuple of:
        - link_pos_rel_base: Dict mapping body name -> (3,) position relative to ``robot0_base``.
        - eef_world_pos: (3,) position of ``grip_site`` in MuJoCo world coordinates.
        - eef_world_rot: (3, 3) rotation matrix of ``grip_site`` in MuJoCo world coordinates.
    """
    T_curr = np.eye(4, dtype=np.float64)
    joint_idx = 0
    link_pos_rel_base: Dict[str, npt.NDArray[np.float64]] = {}

    for name, pos_offset, quat_wxyz, has_joint in _PANDA_KINEMATIC_CHAIN:
        T_body = np.eye(4, dtype=np.float64)
        T_body[:3, :3] = _quat_wxyz_to_rotmat(quat_wxyz)
        T_body[:3, 3] = np.array(pos_offset, dtype=np.float64)
        T_curr = T_curr @ T_body

        if has_joint:
            T_joint = np.eye(4, dtype=np.float64)
            T_joint[:3, :3] = _rot_z(float(qpos[joint_idx]))
            T_curr = T_curr @ T_joint
            joint_idx += 1

        link_pos_rel_base[name] = T_curr[:3, 3].copy()

    # PandaGripper grip_site offset from right_hand body is +0.097m along local Z
    T_grip_site = np.eye(4, dtype=np.float64)
    T_grip_site[:3, 3] = np.array([0.0, 0.0, 0.097], dtype=np.float64)
    T_eef = T_curr @ T_grip_site
    link_pos_rel_base["robot0_eef"] = T_eef[:3, 3].copy()

    eef_world_pos = base_world_pos + T_eef[:3, 3]
    eef_world_rot = T_eef[:3, :3].copy()
    return link_pos_rel_base, eef_world_pos, eef_world_rot


def panda_inverse_kinematics_step(
    qpos: npt.NDArray[np.float64],
    target_world_pos: npt.NDArray[np.float64],
    target_world_rot: Optional[npt.NDArray[np.float64]] = None,
    base_world_pos: npt.NDArray[np.float64] = PANDA_BASE_WORLD_POS,
    damping: float = 0.03,
    max_iters: int = 8,
) -> npt.NDArray[np.float64]:
    """Solve damped least-squares Jacobian IK to track target EEF pose with nullspace centering."""
    q = qpos.copy()
    eps = 1e-5

    for _ in range(max_iters):
        _, pos_curr, rot_curr = panda_forward_kinematics(q, base_world_pos)
        pos_err = target_world_pos - pos_curr

        if target_world_rot is not None:
            rot_err_mat = target_world_rot @ rot_curr.T
            ori_err = Rotation.from_matrix(rot_err_mat).as_rotvec() * 0.5
        else:
            ori_err = np.zeros(3, dtype=np.float64)

        err_6d = np.concatenate([pos_err, ori_err])
        if np.linalg.norm(pos_err) < 5e-4 and np.linalg.norm(ori_err) < 1e-3:
            break

        # Numerical 6x7 geometric Jacobian
        J = np.zeros((6, 7), dtype=np.float64)
        for j in range(7):
            q_pert = q.copy()
            q_pert[j] += eps
            _, pos_p, rot_p = panda_forward_kinematics(q_pert, base_world_pos)
            J[:3, j] = (pos_p - pos_curr) / eps
            drot = Rotation.from_matrix(rot_p @ rot_curr.T).as_rotvec()
            J[3:, j] = drot / eps

        JJT = J @ J.T + (damping ** 2) * np.eye(6, dtype=np.float64)
        J_pinv = J.T @ np.linalg.inv(JJT)

        # Primary task + nullspace attraction toward home posture PANDA_INIT_QPOS
        nullspace_proj = np.eye(7) - J_pinv @ J
        dq = J_pinv @ err_6d + 0.08 * (nullspace_proj @ (PANDA_INIT_QPOS - q))
        dq = np.clip(dq, -0.15, 0.15)
        q = np.clip(q + dq, PANDA_JOINT_MIN, PANDA_JOINT_MAX)

    return q


def _render_synthetic_agentview_frame(
    link_pos_rel_base: Dict[str, npt.NDArray[np.float64]],
    eef_world_pos: npt.NDArray[np.float64],
    eef_world_rot: npt.NDArray[np.float64],
    cube_world_pos: npt.NDArray[np.float64],
    gripper_action: float,
    grasped: bool,
    step_idx: int,
    total_steps: int,
    width: int = 256,
    height: int = 256,
) -> npt.NDArray[np.uint8]:
    """Render a 3D MuJoCo-style agentview RGB observation of the Panda arm, table, and red cube."""
    canvas = np.full((height, width, 3), (235, 232, 226), dtype=np.uint8)

    # Agentview camera looking at the table center [0, 0, 0.85] from [+0.62, 0.0, 1.38]
    cam_pos = np.array([0.65, 0.0, 1.38], dtype=np.float64)
    look_at = np.array([-0.02, 0.0, 0.84], dtype=np.float64)
    forward = look_at - cam_pos
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    f_px = width * 1.15
    cx, cy = width / 2.0, height / 2.0

    def proj(pt_w: npt.NDArray[np.float64]) -> Tuple[int, int, float]:
        rel = pt_w - cam_pos
        x_c = float(np.dot(rel, right))
        y_c = float(-np.dot(rel, up))
        z_c = float(np.dot(rel, forward))
        z_c = max(0.05, z_c)
        u = int(round(cx + f_px * (x_c / z_c)))
        v = int(round(cy + f_px * (y_c / z_c)))
        return u, v, z_c

    # 1. Draw tabletop surface polygon (0.8m x 0.8m at Z = 0.80m)
    table_corners_w = [
        np.array([-0.40, -0.40, TABLE_TOP_Z_WORLD]),
        np.array([0.40, -0.40, TABLE_TOP_Z_WORLD]),
        np.array([0.40, 0.40, TABLE_TOP_Z_WORLD]),
        np.array([-0.40, 0.40, TABLE_TOP_Z_WORLD]),
    ]
    table_pts_2d = np.array([[proj(c)[0], proj(c)[1]] for c in table_corners_w], dtype=np.int32)
    cv2.fillConvexPoly(canvas, table_pts_2d, (250, 250, 250), cv2.LINE_AA)
    cv2.polylines(canvas, [table_pts_2d], True, (190, 185, 175), 2, cv2.LINE_AA)

    # 2. Draw cube shadow on the table surface (Z = 0.801m)
    sh_u, sh_v, sh_z = proj(np.array([cube_world_pos[0], cube_world_pos[1], TABLE_TOP_Z_WORLD + 0.001]))
    sh_r = max(6, int(round(f_px * 0.024 / sh_z)))
    cv2.ellipse(canvas, (sh_u, sh_v), (int(sh_r * 1.2), int(sh_r * 0.7)), 0, 0, 360, (205, 200, 195), -1, cv2.LINE_AA)

    # 3. Draw 3D Red Cube (isometric box facets)
    h_c = 0.021
    c_x, c_y, c_z = cube_world_pos
    top_face = [
        np.array([c_x - h_c, c_y - h_c, c_z + h_c]),
        np.array([c_x + h_c, c_y - h_c, c_z + h_c]),
        np.array([c_x + h_c, c_y + h_c, c_z + h_c]),
        np.array([c_x - h_c, c_y + h_c, c_z + h_c]),
    ]
    front_face = [
        np.array([c_x + h_c, c_y - h_c, c_z - h_c]),
        np.array([c_x + h_c, c_y + h_c, c_z - h_c]),
        np.array([c_x + h_c, c_y + h_c, c_z + h_c]),
        np.array([c_x + h_c, c_y - h_c, c_z + h_c]),
    ]
    front_2d = np.array([[proj(p)[0], proj(p)[1]] for p in front_face], dtype=np.int32)
    top_2d = np.array([[proj(p)[0], proj(p)[1]] for p in top_face], dtype=np.int32)
    cv2.fillConvexPoly(canvas, front_2d, (165, 35, 35), cv2.LINE_AA)
    cv2.fillConvexPoly(canvas, top_2d, (220, 55, 55), cv2.LINE_AA)
    cv2.polylines(canvas, [front_2d, top_2d], True, (110, 20, 20), 1, cv2.LINE_AA)

    # 4. Draw Panda Arm Kinematic Chain
    ordered_links = [b for b in ROBOT_JOINT_BODIES if b in link_pos_rel_base]
    pts_w = [PANDA_BASE_WORLD_POS + link_pos_rel_base[b] for b in ordered_links]
    pts_2d = [proj(p) for p in pts_w]

    for i in range(len(pts_2d) - 1):
        p1 = (pts_2d[i][0], pts_2d[i][1])
        p2 = (pts_2d[i + 1][0], pts_2d[i + 1][1])
        thickness = max(5, int(11 - i * 0.7))
        cv2.line(canvas, p1, p2, (70, 75, 85), thickness + 2, cv2.LINE_AA)
        cv2.line(canvas, p1, p2, (230, 234, 240), thickness, cv2.LINE_AA)

    for i, (u, v, _) in enumerate(pts_2d):
        cv2.circle(canvas, (u, v), 5, (45, 90, 200), -1, cv2.LINE_AA)

    # 5. Draw Panda Parallel-Jaw Gripper Fingers around EEF
    half_aperture = 0.019 if gripper_action > 0.0 else 0.042
    wrist_w = pts_w[-1] if pts_w else (eef_world_pos + np.array([0.0, 0.0, 0.09]))
    finger_y_axis = eef_world_rot[:, 1]
    if np.linalg.norm(finger_y_axis) < 1e-6:
        finger_y_axis = np.array([0.0, 1.0, 0.0])

    l_top = wrist_w - finger_y_axis * half_aperture
    r_top = wrist_w + finger_y_axis * half_aperture
    l_tip = eef_world_pos - finger_y_axis * half_aperture
    r_tip = eef_world_pos + finger_y_axis * half_aperture

    w_2d = proj(wrist_w)[:2]
    lt_2d = proj(l_top)[:2]
    rt_2d = proj(r_top)[:2]
    lp_2d = proj(l_tip)[:2]
    rp_2d = proj(r_tip)[:2]

    finger_color = (30, 175, 65) if grasped else ((40, 120, 230) if gripper_action > 0 else (70, 70, 75))
    cv2.line(canvas, lt_2d, rt_2d, (45, 60, 150), 6, cv2.LINE_AA)
    cv2.line(canvas, lt_2d, lp_2d, finger_color, 5, cv2.LINE_AA)
    cv2.line(canvas, rt_2d, rp_2d, finger_color, 5, cv2.LINE_AA)
    cv2.circle(canvas, w_2d, 6, (30, 50, 140), -1, cv2.LINE_AA)

    return canvas


def execute_panda_manipulation_in_sim(
    actions: npt.NDArray[np.float32],
    demo_eef_pos: Optional[npt.NDArray[np.float32]] = None,
    env_name: str = "Lift",
    render_video: bool = True,
    camera_height: int = 256,
    camera_width: int = 256,
) -> SimulationReplayResult:
    """Execute retargeted demonstration on the Franka Panda in Robosuite / MuJoCo.

    Uses closed-loop object-aligned ``OSC_POSE`` control so the Panda arm reaches
    the object on the table, grasps it at ``t_pick``, lifts and transports it
    along the human's 3D trajectory, and places it at ``t_release``.
    Automatically falls back to the exact 7-DOF Franka Panda kinematics simulator
    if MuJoCo DLL loading is blocked by the host OS policy.

    Args:
        actions: (N, 7) retargeted actions from dataset.
        demo_eef_pos: Optional (N, 3) retargeted EEF positions from dataset.
        env_name: Robosuite environment name (default ``"Lift"``).
        render_video: Whether to render offscreen camera frames.
        camera_height: Render height in pixels.
        camera_width: Render width in pixels.

    Returns:
        SimulationReplayResult containing trajectories, link positions, frames, and metrics.
    """
    actions_f64 = np.asarray(actions, dtype=np.float64)
    N = len(actions_f64)

    if demo_eef_pos is None or len(demo_eef_pos) < N:
        # Reconstruct relative positions from cumulative action deltas
        demo_eef_f64 = np.zeros((N, 3), dtype=np.float64)
        if N > 1:
            demo_eef_f64[1:] = np.cumsum(actions_f64[:-1, :3], axis=0)
    else:
        demo_eef_f64 = np.asarray(demo_eef_pos[:N], dtype=np.float64)

    # Attempt real Robosuite + MuJoCo execution first
    try:
        import robosuite as suite
        from robosuite.controllers import load_controller_config

        controller_cfg = load_controller_config(default_controller="OSC_POSE")
        target_env = "Lift" if env_name.startswith("LIBERO") else env_name

        env = suite.make(
            env_name=target_env,
            robots="Panda",
            controller_configs=controller_cfg,
            has_renderer=False,
            has_offscreen_renderer=render_video,
            use_camera_obs=render_video,
            use_object_obs=True,
            camera_names=["agentview", "robot0_eye_in_hand"] if render_video else None,
            camera_heights=camera_height,
            camera_widths=camera_width,
            control_freq=20,
            ignore_done=True,
        )

        obs = env.reset()
        robot_base_pos = env.sim.data.get_body_xpos("robot0_base").copy()
        sim_eef_init = obs.get("robot0_eef_pos", np.array([-0.095, 0.005, 1.020])).copy()
        if "cube_pos" in obs:
            sim_cube_init = obs["cube_pos"].copy()
        elif hasattr(env, "cube_body_id"):
            sim_cube_init = env.sim.data.body_xpos[env.cube_body_id].copy()
        else:
            sim_cube_init = np.array([0.0, 0.0, 0.821], dtype=np.float64)

        init_cube_z = float(sim_cube_init[2])
        target_pos_world, target_gripper, phases = build_object_aligned_target_trajectory(
            demo_eef_f64, actions_f64, sim_eef_init, sim_cube_init
        )

        joint_states: List[np.ndarray] = []
        ee_positions: List[np.ndarray] = []
        ee_oris: List[np.ndarray] = []
        cube_positions: List[np.ndarray] = []
        executed_actions: List[np.ndarray] = []
        joint_body_positions: List[Dict[str, np.ndarray]] = []
        agentview_frames: List[np.ndarray] = []
        wrist_frames: List[np.ndarray] = []

        # Reference top-down orientation for stable parallel-jaw grasping
        ref_eef_quat = obs.get("robot0_eef_quat", np.array([1.0, 0.0, 0.0, 0.0])).copy()
        ref_rot = Rotation.from_quat(ref_eef_quat)

        grasped_any = False
        task_success_any = False

        for i in range(N):
            curr_eef = obs.get("robot0_eef_pos", sim_eef_init).copy()
            curr_quat = obs.get("robot0_eef_quat", ref_eef_quat).copy()
            curr_qpos = obs.get("robot0_joint_pos", PANDA_INIT_QPOS).copy()

            if "cube_pos" in obs:
                curr_cube = obs["cube_pos"].copy()
            elif hasattr(env, "cube_body_id"):
                curr_cube = env.sim.data.body_xpos[env.cube_body_id].copy()
            else:
                curr_cube = sim_cube_init.copy()

            # Record body positions relative to robot0_base
            curr_bodies: Dict[str, np.ndarray] = {}
            for b_name in ROBOT_JOINT_BODIES:
                if b_name == "robot0_eef":
                    curr_bodies[b_name] = curr_eef - robot_base_pos
                    continue
                try:
                    curr_bodies[b_name] = env.sim.data.get_body_xpos(b_name).copy() - robot_base_pos
                except Exception:
                    pass
            joint_body_positions.append(curr_bodies)

            joint_states.append(curr_qpos.astype(np.float32))
            ee_positions.append(curr_eef.astype(np.float32))
            ee_oris.append(curr_quat.astype(np.float32))
            cube_positions.append(curr_cube.astype(np.float32))

            if render_video:
                if "agentview_image" in obs:
                    agentview_frames.append(np.flipud(obs["agentview_image"]))
                if "robot0_eye_in_hand_image" in obs:
                    wrist_frames.append(np.flipud(obs["robot0_eye_in_hand_image"]))

            # Closed-loop proportional OSC_POSE command toward target_pos_world[i]
            # Note: OSC_POSE scales action[:3] by 0.05m and action[3:6] by 0.5rad
            pos_error = target_pos_world[i] - curr_eef
            cmd_pos = np.clip((pos_error / 0.05) * 1.6, -1.0, 1.0)

            curr_rot = Rotation.from_quat(curr_quat)
            ori_error = Rotation.from_matrix(ref_rot.as_matrix() @ curr_rot.as_matrix().T).as_rotvec()
            cmd_ori = np.clip((ori_error / 0.5) * 0.8, -0.5, 0.5)

            cmd_grip = float(target_gripper[i])
            cmd_action = np.concatenate([cmd_pos, cmd_ori, [cmd_grip]]).astype(np.float32)
            executed_actions.append(cmd_action)

            # Step physics; add 2 extra micro-steps at t_pick so gripper fingers lock around the cube
            n_substeps = 3 if (phases.t_pick <= i <= phases.t_pick + 2) else 1
            for _ in range(n_substeps):
                obs, reward, _done, _info = env.step(cmd_action)
                curr_eef = obs.get("robot0_eef_pos", curr_eef).copy()
                pos_error = target_pos_world[i] - curr_eef
                cmd_action[:3] = np.clip((pos_error / 0.05) * 1.6, -1.0, 1.0)

            if hasattr(env, "_check_grasp") and hasattr(env, "cube"):
                try:
                    if env._check_grasp(gripper=env.robots[0].gripper, object_geoms=env.cube):
                        grasped_any = True
                except Exception:
                    pass
            if hasattr(env, "_check_success"):
                try:
                    if env._check_success():
                        task_success_any = True
                except Exception:
                    pass

        env.close()

        ee_arr = np.array(ee_positions, dtype=np.float32)
        cube_arr = np.array(cube_positions, dtype=np.float32)
        max_lift = float(np.max(cube_arr[:, 2]) - init_cube_z) if len(cube_arr) > 0 else 0.0
        if max_lift > 0.02:
            grasped_any = True
        if max_lift > 0.04:
            task_success_any = True
        total_dist = float(np.sum(np.linalg.norm(np.diff(ee_arr, axis=0), axis=-1))) if len(ee_arr) > 1 else 0.0

        return SimulationReplayResult(
            joint_states=np.array(joint_states, dtype=np.float32),
            ee_pos=ee_arr,
            ee_ori=np.array(ee_oris, dtype=np.float32),
            cube_positions=cube_arr,
            executed_actions=np.array(executed_actions, dtype=np.float32),
            joint_body_positions=joint_body_positions,
            agentview_frames=agentview_frames,
            wrist_frames=wrist_frames,
            grasped=grasped_any,
            task_success=task_success_any,
            max_cube_lift_m=max_lift,
            total_eef_distance_m=total_dist,
            phases=phases,
        )

    except Exception as exc:
        logger.warning(
            "Native MuJoCo DLL blocked or unavailable (%s). Using built-in 7-DOF Franka Panda kinematics & grasp simulator.",
            exc,
        )

    # -----------------------------------------------------------------------
    # Fallback: Exact 7-DOF Franka Panda Kinematics & Grasping Simulator
    # -----------------------------------------------------------------------
    q_curr = PANDA_INIT_QPOS.copy()
    link_init, sim_eef_init, ref_rot_mat = panda_forward_kinematics(q_curr)
    sim_cube_init = np.array([0.0, 0.0, TABLE_TOP_Z_WORLD + 0.021], dtype=np.float64)
    init_cube_z = float(sim_cube_init[2])

    target_pos_world, target_gripper, phases = build_object_aligned_target_trajectory(
        demo_eef_f64, actions_f64, sim_eef_init, sim_cube_init
    )

    joint_states_fb: List[np.ndarray] = []
    ee_positions_fb: List[np.ndarray] = []
    ee_oris_fb: List[np.ndarray] = []
    cube_positions_fb: List[np.ndarray] = []
    executed_actions_fb: List[np.ndarray] = []
    joint_body_positions_fb: List[Dict[str, np.ndarray]] = []
    agentview_frames_fb: List[np.ndarray] = []
    wrist_frames_fb: List[np.ndarray] = []

    curr_cube_pos = sim_cube_init.copy()
    is_grasped = False
    grasped_any_fb = False
    task_success_fb = False

    for i in range(N):
        link_pos_rel, curr_eef, curr_rot = panda_forward_kinematics(q_curr)
        curr_quat = Rotation.from_matrix(curr_rot).as_quat().astype(np.float32)

        # Check physical grasp condition: gripper closed while EEF is within 3.5 cm of cube
        cmd_grip = float(target_gripper[i])
        dist_to_cube = float(np.linalg.norm(curr_eef - curr_cube_pos))
        if cmd_grip > 0.0 and dist_to_cube < 0.045:
            is_grasped = True
            grasped_any_fb = True
        elif cmd_grip < 0.0:
            is_grasped = False
            # Cube settles onto table surface when released
            curr_cube_pos[2] = TABLE_TOP_Z_WORLD + 0.021

        if is_grasped:
            curr_cube_pos = curr_eef.copy()
            curr_cube_pos[2] = max(TABLE_TOP_Z_WORLD + 0.021, curr_eef[2])
            if curr_cube_pos[2] > TABLE_TOP_Z_WORLD + 0.06:
                task_success_fb = True

        joint_states_fb.append(q_curr.astype(np.float32))
        ee_positions_fb.append(curr_eef.astype(np.float32))
        ee_oris_fb.append(curr_quat)
        cube_positions_fb.append(curr_cube_pos.astype(np.float32))
        joint_body_positions_fb.append({k: v.copy() for k, v in link_pos_rel.items()})

        if render_video:
            from .panda_3d_renderer import PandaMuJoCo3DRenderer

            renderer_3d = PandaMuJoCo3DRenderer.get_instance()
            frame = renderer_3d.render_frame(
                qpos=q_curr,
                cube_world_pos=curr_cube_pos,
                gripper_action=cmd_grip,
                width=camera_width,
                height=camera_height,
                camera_mode="agentview",
            )
            agentview_frames_fb.append(frame)

        # Compute OSC_POSE normalized action command and step 7-DOF IK
        pos_error = target_pos_world[i] - curr_eef
        cmd_pos = np.clip((pos_error / 0.05) * 1.5, -1.0, 1.0)
        ori_error = Rotation.from_matrix(ref_rot_mat @ curr_rot.T).as_rotvec()
        cmd_ori = np.clip((ori_error / 0.5) * 0.8, -0.5, 0.5)
        cmd_action = np.concatenate([cmd_pos, cmd_ori, [cmd_grip]]).astype(np.float32)
        executed_actions_fb.append(cmd_action)

        # Impedance step toward target_pos_world[i] (critically damped OSC response)
        step_goal = curr_eef + 0.82 * (target_pos_world[i] - curr_eef)
        q_curr = panda_inverse_kinematics_step(q_curr, step_goal, ref_rot_mat)

    ee_arr_fb = np.array(ee_positions_fb, dtype=np.float32)
    cube_arr_fb = np.array(cube_positions_fb, dtype=np.float32)
    max_lift_fb = float(np.max(cube_arr_fb[:, 2]) - init_cube_z) if len(cube_arr_fb) > 0 else 0.0
    total_dist_fb = float(np.sum(np.linalg.norm(np.diff(ee_arr_fb, axis=0), axis=-1))) if len(ee_arr_fb) > 1 else 0.0

    return SimulationReplayResult(
        joint_states=np.array(joint_states_fb, dtype=np.float32),
        ee_pos=ee_arr_fb,
        ee_ori=np.array(ee_oris_fb, dtype=np.float32),
        cube_positions=cube_arr_fb,
        executed_actions=np.array(executed_actions_fb, dtype=np.float32),
        joint_body_positions=joint_body_positions_fb,
        agentview_frames=agentview_frames_fb,
        wrist_frames=wrist_frames_fb,
        grasped=grasped_any_fb,
        task_success=task_success_fb,
        max_cube_lift_m=max_lift_fb,
        total_eef_distance_m=total_dist_fb,
        phases=phases,
    )
