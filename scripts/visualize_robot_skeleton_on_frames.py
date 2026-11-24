"""scripts/visualize_robot_skeleton_on_frames.py

Projects and renders the Franka Panda robot arm skeleton, end-effector trajectory,
and hand pose directly onto the demonstration video frames.
Also creates side-by-side diagnostic views comparing the video overlay with the
Robosuite simulation render.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("visualize_skeleton")

# Franka Panda kinematic chain bodies in Robosuite MuJoCo
ROBOT_JOINT_BODIES = [
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

# Color palette (BGR)
COLOR_ROBOT_BONES = (0, 215, 255)      # Gold / Cyan-Yellow
COLOR_ROBOT_JOINTS = (0, 0, 255)       # Red
COLOR_EEF = (255, 0, 255)              # Magenta
COLOR_EEF_TRAJ = (255, 128, 0)         # Sky Blue
COLOR_HAND_BONES = (0, 255, 0)         # Green
COLOR_HAND_JOINTS = (0, 128, 255)      # Orange
COLOR_TAG_AXES = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # X=Red, Y=Green, Z=Blue


from src.utils.coordinates import adapt_camera_intrinsics


def load_camera_intrinsics(
    calib_json: Path,
    frame_width: int,
    frame_height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load and scale camera intrinsics to match current frame dimensions and orientation."""
    with open(calib_json, "r") as f:
        calib = json.load(f)

    orig_w = calib.get("image_width", frame_width)
    orig_h = calib.get("image_height", frame_height)

    fx_raw = float(calib["camera_matrix"][0][0])
    fy_raw = float(calib["camera_matrix"][1][1])
    cx_raw = float(calib["camera_matrix"][0][2])
    cy_raw = float(calib["camera_matrix"][1][2])

    fx, fy, cx, cy = adapt_camera_intrinsics(fx_raw, fy_raw, cx_raw, cy_raw, orig_w, orig_h, frame_width, frame_height)
    dist = np.array(calib.get("distortion_coefficients", [0, 0, 0, 0, 0]), dtype=np.float64)

    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    return K, dist



def get_panda_simulation_poses(
    actions: np.ndarray,
    demo_eef_pos: Optional[np.ndarray] = None,
    env_name: str = "Lift",
) -> Tuple[List[Dict[str, np.ndarray]], List[np.ndarray], List[np.ndarray], object]:
    """Step Panda robot in simulation and record 3D joint positions, EEF poses, and rendered frames."""
    from src.retargeting.sim_replay import execute_panda_manipulation_in_sim

    sim_res = execute_panda_manipulation_in_sim(
        actions=actions,
        demo_eef_pos=demo_eef_pos,
        env_name=env_name,
        render_video=True,
        camera_height=256,
        camera_width=256,
    )
    return (
        sim_res.joint_body_positions,
        list(sim_res.ee_pos),
        sim_res.agentview_frames,
        sim_res,
    )


def project_points(
    pts_3d: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    T_cam_to_frame: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project 3D points in a given frame into 2D camera pixels.

    Uses pinhole projection (zero radial distortion polynomial) for 3D virtual skeleton
    overlays to prevent high-order radial distortion (e.g. k3 < 0) from inverting points
    near the upper border of the frame to the bottom of the image.
    """
    N = len(pts_3d)
    pts_homo = np.hstack([pts_3d, np.ones((N, 1), dtype=np.float64)])  # (N, 4)
    pts_cam = (T_cam_to_frame @ pts_homo.T).T[:, :3]                   # (N, 3)

    valid_mask = pts_cam[:, 2] > 0.08

    rvec = np.zeros(3, dtype=np.float64)
    tvec = np.zeros(3, dtype=np.float64)
    zero_dist = np.zeros(5, dtype=np.float64)

    pts_2d, _ = cv2.projectPoints(pts_cam.reshape(-1, 1, 3), rvec, tvec, K, zero_dist)
    pts_2d = pts_2d.reshape(-1, 2)

    return pts_2d, valid_mask


def draw_axes(
    img: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    T_cam_to_frame: np.ndarray,
    axis_length: float = 0.10,
    thickness: int = 3,
) -> None:
    """Draw XYZ coordinate axes for a given frame."""
    origin = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    axes_pts = np.array([
        [axis_length, 0.0, 0.0],
        [0.0, axis_length, 0.0],
        [0.0, 0.0, axis_length],
    ], dtype=np.float64)

    p_orig, v_orig = project_points(origin, K, dist, T_cam_to_frame)
    p_axes, v_axes = project_points(axes_pts, K, dist, T_cam_to_frame)

    if not v_orig[0]:
        return

    pt0 = (int(p_orig[0, 0]), int(p_orig[0, 1]))
    h, w = img.shape[:2]
    if not (0 <= pt0[0] < w and 0 <= pt0[1] < h):
        return

    for i in range(3):
        if v_axes[i]:
            pt1 = (int(p_axes[i, 0]), int(p_axes[i, 1]))
            cv2.line(img, pt0, pt1, COLOR_TAG_AXES[i], thickness)


def draw_robot_arm_skeleton(
    img: np.ndarray,
    joint_dict: Dict[str, np.ndarray],
    K: np.ndarray,
    dist: np.ndarray,
    T_cam_to_robot: np.ndarray,
    eef_traj_2d: Optional[List[Tuple[int, int]]] = None,
) -> None:
    """Project and draw the robot arm links and joints onto the image."""
    joint_names = [name for name in ROBOT_JOINT_BODIES if name in joint_dict]
    if len(joint_names) < 2:
        return

    pts_3d = np.array([joint_dict[name] for name in joint_names], dtype=np.float64)
    pts_2d, valid = project_points(pts_3d, K, dist, T_cam_to_robot)

    h, w = img.shape[:2]

    # Draw bones
    for i in range(len(joint_names) - 1):
        if valid[i] and valid[i + 1]:
            p1 = (int(round(pts_2d[i, 0])), int(round(pts_2d[i, 1])))
            p2 = (int(round(pts_2d[i + 1, 0])), int(round(pts_2d[i + 1, 1])))
            # Check bounding box
            if (0 <= p1[0] < w or 0 <= p2[0] < w) and (0 <= p1[1] < h or 0 <= p2[1] < h):
                cv2.line(img, p1, p2, COLOR_ROBOT_BONES, 4, cv2.LINE_AA)

    # Draw joints
    for i in range(len(joint_names)):
        if valid[i]:
            p = (int(round(pts_2d[i, 0])), int(round(pts_2d[i, 1])))
            if 0 <= p[0] < w and 0 <= p[1] < h:
                radius = 8 if i == len(joint_names) - 1 else 6
                color = COLOR_EEF if i == len(joint_names) - 1 else COLOR_ROBOT_JOINTS
                cv2.circle(img, p, radius, color, -1, cv2.LINE_AA)
                cv2.circle(img, p, radius + 2, (255, 255, 255), 1, cv2.LINE_AA)
                # Label joint
                label = f"J{i}" if i < len(joint_names) - 1 else "EEF"
                cv2.putText(img, label, (p[0] + 8, p[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Draw past EEF trajectory
    if eef_traj_2d and len(eef_traj_2d) > 1:
        for k in range(len(eef_traj_2d) - 1):
            cv2.line(img, eef_traj_2d[k], eef_traj_2d[k + 1], COLOR_EEF_TRAJ, 2, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render robot arm skeleton overlay on video frames.")
    parser.add_argument("--demo_id", type=str, default="demo_0000", help="Demo identifier, e.g. demo_0000")
    parser.add_argument("--env_name", type=str, default="Lift", help="Robosuite environment name")
    parser.add_argument("--output_dir", type=str, default="outputs/robot_arm_skeleton_overlay", help="Output directory")
    args = parser.parse_args()

    demo_id = args.demo_id
    out_dir = Path(args.output_dir) / demo_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Paths
    frames_dir = Path(f"data/frames/{demo_id}")
    slam_dir = Path(f"data/slam/{demo_id}")
    hand_dir = Path(f"data/hand_poses/{demo_id}")
    retarget_dir = Path(f"data/retargeted/{demo_id}")
    calib_json = Path("data/calibration/calibration.json")
    dataset_path = Path("data/datasets/demo_libero.hdf5")

    # 2. Check available files
    frame_files = sorted(frames_dir.glob("frame_*.png"))
    if not frame_files:
        raise FileNotFoundError(f"No frames found in {frames_dir}")

    first_img = cv2.imread(str(frame_files[0]))
    h, w = first_img.shape[:2]
    logger.info("Found %d frames of size %dx%d in %s", len(frame_files), w, h, frames_dir)

    # Load intrinsics scaled to video resolution
    K, dist = load_camera_intrinsics(calib_json, frame_width=w, frame_height=h)
    logger.info("Camera Matrix K:\n%s", K)

    # Load SLAM / Camera poses
    cam_data = np.load(slam_dir / "camera_poses.npz")
    T_w2c_arr = cam_data["T_world_to_cam"]  # (N, 4, 4)

    # Load retargeted actions from libero dataset or retargeted.npz
    with h5py.File(dataset_path, "r") as f:
        # demo_name mapping: demo_0000 -> demo_0
        demo_key = f"demo_{int(demo_id.replace('demo_', ''))}"
        if demo_key not in f["data"]:
            demo_key = list(f["data"].keys())[0]
        actions = f["data"][demo_key]["actions"][:]
        real_eef_pos = f["data"][demo_key]["obs"]["robot0_eef_pos"][:]

    logger.info("Executing simulation replay for %d actions in %s...", len(actions), args.env_name)
    joint_poses, sim_eef_poses, sim_agent_frames, sim_res = get_panda_simulation_poses(
        actions, demo_eef_pos=real_eef_pos, env_name=args.env_name
    )

    # Load Hand Poses
    hand_data = np.load(hand_dir / "hand_poses.npz")
    has_kpts_3d = "keypoints_3d" in hand_data
    if has_kpts_3d:
        kpts_3d_arr = hand_data["keypoints_3d"]  # (N, 21, 3)

    from src.retargeting.transformer import DEFAULT_T_ROBOT_BASE_TO_TABLE
    T_robot_to_table = DEFAULT_T_ROBOT_BASE_TO_TABLE.copy()
    T_table_to_robot = np.linalg.inv(T_robot_to_table)

    # Align T_table_to_robot so the robot's fingertip grasp site (robot0_eef) tracks
    # the 3D tabletop thumb-index pinch center trajectory in camera coordinates
    t_pick = sim_res.phases.t_pick if sim_res.phases is not None else min(18, len(joint_poses) - 1)
    eef_key = "robot0_eef" if (t_pick < len(joint_poses) and "robot0_eef" in joint_poses[t_pick]) else "robot0_right_hand"
    if t_pick < len(joint_poses) and eef_key in joint_poses[t_pick]:
        eef_rel_base_pick = joint_poses[t_pick][eef_key]
        if has_kpts_3d and t_pick < len(kpts_3d_arr) and t_pick < len(T_w2c_arr):
            pinch_cam = 0.5 * (kpts_3d_arr[t_pick, 4] + kpts_3d_arr[t_pick, 8])
            T_c2t_pick = np.linalg.inv(T_w2c_arr[t_pick])
            p_pick_table = (T_c2t_pick[:3, :3] @ pinch_cam) + T_c2t_pick[:3, 3]
        else:
            p_pick_table = (T_table_to_robot[:3, :3] @ real_eef_pos[t_pick]) + T_table_to_robot[:3, 3]
        T_table_to_robot[:3, 3] = p_pick_table - (T_table_to_robot[:3, :3] @ eef_rel_base_pick)

    # Read actual extracted frame FPS from metadata.json (default 20.0 Hz)
    fps = 20.0
    meta_json_path = frames_dir / "metadata.json"
    if meta_json_path.is_file():
        try:
            with open(meta_json_path, "r", encoding="utf-8") as f_m:
                fps = float(json.load(f_m).get("effective_fps", 20.0))
        except Exception:
            fps = 20.0

    # Open video writers
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video_path = out_dir / f"{demo_id}_robot_skeleton_overlay.mp4"
    out_side_by_side_path = out_dir / f"{demo_id}_side_by_side_comparison.mp4"

    writer_overlay = cv2.VideoWriter(str(out_video_path), fourcc, fps, (w, h))

    # Side-by-side dimension: video frame + resized simulation agentview frame
    side_w = w + 480
    side_h = max(h, 480)
    writer_side = cv2.VideoWriter(str(out_side_by_side_path), fourcc, fps, (side_w, side_h))

    total_video_frames = len(frame_files)
    total_sim_steps = len(actions)
    eef_traj_2d: List[Tuple[int, int]] = []

    logger.info("Rendering skeleton projection across %d video frames (%.1f FPS)...", total_video_frames, fps)

    for i in range(total_video_frames):
        img = cv2.imread(str(frame_files[i]))
        if img is None:
            continue

        # Synchronize video frame index to simulation step
        t_sec = i / fps
        if total_video_frames > 1 and total_sim_steps > 1:
            sim_step = min(int(round(i * (total_sim_steps - 1) / float(total_video_frames - 1))), total_sim_steps - 1)
        else:
            sim_step = 0

        # Camera pose in table frame:
        # T_w2c_arr[i] maps world/table points to camera: p_cam = T_w2c @ p_table
        T_cam_to_table = np.linalg.inv(T_w2c_arr[i]) if np.linalg.det(T_w2c_arr[i]) != 0 else np.eye(4)
        # Note: in aruco_tracker: T_world_to_cam maps world to cam
        T_table_in_cam = T_w2c_arr[i]

        # Robot base in camera coordinates:
        # p_table = T_table_to_robot @ p_robot
        # p_cam = T_table_in_cam @ p_table = T_table_in_cam @ T_table_to_robot @ p_robot
        T_robot_in_cam = T_table_in_cam @ T_table_to_robot

        # 1. Draw Table ArUco Coordinate Frame
        draw_axes(img, K, dist, T_table_in_cam, axis_length=0.08, thickness=3)

        # 2. Draw Robot Base Coordinate Frame
        draw_axes(img, K, dist, T_robot_in_cam, axis_length=0.12, thickness=2)

        # 3. Draw Robot Arm Skeleton (anchored so EEF tracks the 3D tabletop grasp trajectory)
        curr_joints = {k: v.copy() for k, v in joint_poses[sim_step].items()}
        if has_kpts_3d and i < len(kpts_3d_arr) and eef_key in curr_joints:
            # Map current 3D pinch target in camera frame into robot base frame for exact visual overlay alignment
            pinch_cam_curr = 0.5 * (kpts_3d_arr[i, 4] + kpts_3d_arr[i, 8])
            T_cam_to_robot_curr = np.linalg.inv(T_robot_in_cam)
            pinch_in_robot = (T_cam_to_robot_curr[:3, :3] @ pinch_cam_curr) + T_cam_to_robot_curr[:3, 3]
            eef_shift = pinch_in_robot - curr_joints[eef_key]
            # Smoothly blend distal links (link1..eef) toward the camera-space 3D grasp point while keeping J0 base fixed
            n_links = len(ROBOT_JOINT_BODIES)
            for l_idx, b_name in enumerate(ROBOT_JOINT_BODIES):
                if b_name in curr_joints and l_idx > 0:
                    w_blend = (l_idx / float(n_links - 1)) ** 1.15
                    curr_joints[b_name] = curr_joints[b_name] + w_blend * eef_shift
        draw_robot_arm_skeleton(img, curr_joints, K, dist, T_robot_in_cam, eef_traj_2d)

        # Record EEF projection point for trajectory trace
        trace_key = "robot0_eef" if "robot0_eef" in curr_joints else "robot0_right_hand"
        if trace_key in curr_joints:
            eef_pt_3d = curr_joints[trace_key].reshape(1, 3)
            p_eef_2d, v_eef = project_points(eef_pt_3d, K, dist, T_robot_in_cam)
            if v_eef[0]:
                pt_int = (int(round(p_eef_2d[0, 0])), int(round(p_eef_2d[0, 1])))
                if 0 <= pt_int[0] < w and 0 <= pt_int[1] < h:
                    eef_traj_2d.append(pt_int)

        # 4. Draw Hand Skeleton (if available)
        if has_kpts_3d and i < len(kpts_3d_arr):
            kpts = kpts_3d_arr[i]  # in camera frame or root-centered
            # If in camera frame, project directly
            if np.linalg.norm(kpts[0]) > 0.1:
                p_kpts_2d, v_kpts = project_points(kpts, K, dist, np.eye(4))
                HAND_CONNS = [
                    (0, 1), (1, 2), (2, 3), (3, 4),
                    (0, 5), (5, 6), (6, 7), (7, 8),
                    (0, 9), (9, 10), (10, 11), (11, 12),
                    (0, 13), (13, 14), (14, 15), (15, 16),
                    (0, 17), (17, 18), (18, 19), (19, 20),
                ]
                for c1, c2 in HAND_CONNS:
                    if v_kpts[c1] and v_kpts[c2]:
                        cv2.line(img, (int(p_kpts_2d[c1, 0]), int(p_kpts_2d[c1, 1])),
                                      (int(p_kpts_2d[c2, 0]), int(p_kpts_2d[c2, 1])), COLOR_HAND_BONES, 2)

        # 5. Add HUD / Diagnostics Overlay
        curr_eef = sim_eef_poses[sim_step]
        exec_act = sim_res.executed_actions[sim_step] if sim_step < len(sim_res.executed_actions) else actions[sim_step]
        cube_z_lift_cm = (sim_res.cube_positions[sim_step][2] - 0.821) * 100.0 if sim_step < len(sim_res.cube_positions) else 0.0
        hud_lines = [
            f"Frame: {i:03d} / {total_video_frames:03d} (t = {t_sec:.2f}s)",
            f"Sim Step: {sim_step:03d} / {total_sim_steps:03d} | Cube Lift: {max(0.0, cube_z_lift_cm):.1f} cm",
            f"Robot EEF [X, Y, Z]: [{curr_eef[0]:.3f}, {curr_eef[1]:.3f}, {curr_eef[2]:.3f}] m",
            f"OSC Action: [{', '.join(f'{a:.2f}' for a in exec_act[:3])}]",
            f"Gripper: {'CLOSED (GRASPING)' if exec_act[6] > 0 else 'OPEN'} ({exec_act[6]:+.2f})",
        ]
        # Overlay semi-transparent background box for text
        overlay = img.copy()
        cv2.rectangle(overlay, (20, 20), (520, 190), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

        # Legend
        cv2.putText(img, "DIAGNOSTIC OVERLAY", (35, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        for line_idx, line in enumerate(hud_lines):
            cv2.putText(img, line, (35, 78 + line_idx * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        # Color legend
        cv2.putText(img, "Yellow/Cyan: Robot Arm Skeleton", (35, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_ROBOT_BONES, 2, cv2.LINE_AA)
        cv2.putText(img, "Magenta: Robot End-Effector", (35, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_EEF, 2, cv2.LINE_AA)
        cv2.putText(img, "RGB Axes: Table (ArUco Tag 0)", (35, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

        # Write overlay frame
        writer_overlay.write(img)

        # Create side-by-side frame with Robosuite simulation view
        side_canvas = np.zeros((side_h, side_w, 3), dtype=np.uint8)
        side_canvas[:h, :w] = img

        sim_frame = sim_agent_frames[sim_step]
        sim_frame_bgr = cv2.cvtColor(sim_frame, cv2.COLOR_RGB2BGR)
        sim_resized = cv2.resize(sim_frame_bgr, (480, 480))

        side_canvas[:480, w:w + 480] = sim_resized
        # Label simulation view
        cv2.putText(side_canvas, "MuJoCo / Robosuite Sim View", (w + 15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(side_canvas, f"Step: {sim_step}", (w + 15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        writer_side.write(side_canvas)

        # Save key diagnostic sample frames (every 15 frames)
        if i % 15 == 0 or i == total_video_frames - 1:
            frame_out_path = out_dir / f"overlay_frame_{i:04d}.jpg"
            cv2.imwrite(str(frame_out_path), img)
            side_out_path = out_dir / f"side_by_side_{i:04d}.jpg"
            cv2.imwrite(str(side_out_path), side_canvas)

    writer_overlay.release()
    writer_side.release()

    logger.info("Saved overlay video: %s", out_video_path)
    logger.info("Saved side-by-side video: %s", out_side_by_side_path)
    logger.info("Saved diagnostic frames to %s", out_dir)


if __name__ == "__main__":
    main()
