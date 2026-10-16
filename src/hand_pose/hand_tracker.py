"""src.hand_pose.hand_tracker — HaMeR 3D hand tracking, metric scale enforcement, and gripper aperture extraction."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import numpy.typing as npt

from ..utils.io import get_sorted_image_files

logger = logging.getLogger("hand_pose")

# MANO 21-joint topology indices
WRIST_IDX: int = 0
THUMB_TIP_IDX: int = 4
INDEX_TIP_IDX: int = 8
MIDDLE_MCP_IDX: int = 9

# Human anatomical reference: wrist to middle MCP joint distance (~85 mm)
L_REF_METRES: float = 0.085

HAND_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
]


def run_hamer_subprocess(
    hamer_script: Path | str,
    frames_dir: Path | str,
    output_dir: Path | str,
) -> Path:
    """Execute HaMeR demo.py via subprocess to obtain MANO hand predictions.

    Args:
        hamer_script: Path to HaMeR demo.py.
        frames_dir: Directory containing input video frames.
        output_dir: Directory for HaMeR outputs.

    Returns:
        Path to generated .npz archive.
    """
    hamer_p = Path(hamer_script).resolve()
    if not hamer_p.is_file():
        raise FileNotFoundError(f"HaMeR script not found: {hamer_p}")

    out_p = Path(output_dir).resolve()
    out_p.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(hamer_p),
        "--img_folder", str(Path(frames_dir).resolve()),
        "--out_folder", str(out_p),
        "--batch_size", "16",
        "--full_frame",
        "--save_mesh",
    ]

    logger.info("Executing HaMeR: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=str(hamer_p.parent), check=True)
    logger.info("HaMeR completed successfully.")

    # Locate HaMeR output, excluding our own hand_poses.npz
    candidates = [
        out_p / "hamer_output.npz",
        out_p / "results.npz",
        out_p / "output.npz",
        *[p for p in sorted(out_p.glob("*.npz")) if p.name not in {"hand_poses.npz", "table_transform.npz", "camera_poses.npz"}],
    ]
    for c in candidates:
        if c.is_file():
            return c

    raise FileNotFoundError(f"No HaMeR output npz found in {out_p}")


def extract_wrist_transforms(npz_data: dict[str, npt.NDArray]) -> npt.NDArray[np.float64]:
    """Extract wrist SE(3) transforms in camera coordinates (T_cam_to_hand).

    Args:
        npz_data: Dictionary loaded from HaMeR output.

    Returns:
        Array of shape (N, 4, 4) float64.
    """
    # 1. Direct 4x4 matrix
    for key in ("T_cam_to_hand", "wrist_transform", "cam_to_hand"):
        if key in npz_data:
            arr = npz_data[key].astype(np.float64)
            if arr.ndim == 3 and arr.shape[-2:] == (4, 4):
                return arr

    # 2. Reconstruct from rotation and translation
    rot_keys = [k for k in ("global_orient", "smplx_global_orient", "pred_rot") if k in npz_data]
    trans_keys = [k for k in ("pred_cam_t", "cam_t", "translation") if k in npz_data]

    if rot_keys and trans_keys:
        rot_raw = np.squeeze(npz_data[rot_keys[0]].astype(np.float64))
        trans = np.squeeze(npz_data[trans_keys[0]].astype(np.float64))
        n_frames = len(trans)

        transforms = np.tile(np.eye(4, dtype=np.float64), (n_frames, 1, 1))
        if rot_raw.ndim == 2 and rot_raw.shape[1] == 3:
            # Axis-angle
            from scipy.spatial.transform import Rotation
            transforms[:, :3, :3] = Rotation.from_rotvec(rot_raw).as_matrix()
        elif rot_raw.ndim == 3 and rot_raw.shape[1:] == (3, 3):
            transforms[:, :3, :3] = rot_raw
        transforms[:, :3, 3] = trans
        return transforms

    raise KeyError(f"Wrist transform not found in HaMeR output. Keys: {list(npz_data.keys())}")


def extract_keypoints_3d(npz_data: dict[str, npt.NDArray]) -> npt.NDArray[np.float64]:
    """Extract MANO 3D keypoint positions in camera coordinates.

    Args:
        npz_data: Dictionary loaded from HaMeR output.

    Returns:
        Array of shape (N, 21, 3) float64.
    """
    for key in ("keypoints_3d", "pred_keypoints_3d", "joints_3d", "hand_joints", "j3d"):
        if key in npz_data:
            arr = npz_data[key].astype(np.float64)
            if arr.ndim == 4 and arr.shape[1] == 1:
                arr = arr.squeeze(axis=1)
            if arr.ndim == 3 and arr.shape[1] == 21 and arr.shape[2] == 3:
                return arr

    raise KeyError(f"3D keypoints not found in HaMeR output. Keys: {list(npz_data.keys())}")


def enforce_anatomical_scale(
    keypoints_3d: npt.NDArray[np.float64],
    transforms: npt.NDArray[np.float64],
    l_ref: float = L_REF_METRES,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float]:
    """Rescale keypoints and wrist translations to metric meters using human anatomical baseline.

    Args:
        keypoints_3d: Raw keypoints (N, 21, 3).
        transforms: Wrist transforms (N, 4, 4).
        l_ref: Reference wrist-to-middle MCP distance in meters (default 0.085 m).

    Returns:
        Tuple of (scaled_keypoints, scaled_transforms, scale_factor).
    """
    wrist_pos = keypoints_3d[:, WRIST_IDX, :]
    middle_mcp_pos = keypoints_3d[:, MIDDLE_MCP_IDX, :]
    dists = np.linalg.norm(middle_mcp_pos - wrist_pos, axis=-1)
    median_dist = float(np.median(dists))

    if median_dist < 1e-9:
        raise ValueError(f"Median wrist-to-middle-MCP distance is degenerate: {median_dist}")

    scale = float(l_ref / median_dist)
    logger.info("Metric scale anchor: median dist=%.4f m -> scale factor=%.4f", median_dist, scale)

    scaled_keypoints = keypoints_3d * scale
    scaled_transforms = transforms.copy()
    scaled_transforms[:, :3, 3] *= scale

    return scaled_keypoints, scaled_transforms, scale


def compute_gripper_state(
    keypoints_3d: npt.NDArray[np.float64],
    pinch_threshold_m: float = 0.04,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute continuous aperture and binary open/close gripper states.

    Aperture is Euclidean distance between thumb tip and index tip in meters:
        d_pinch = || P_thumb_tip - P_index_tip ||_2

    Args:
        keypoints_3d: Metric scaled keypoints (N, 21, 3).
        pinch_threshold_m: Threshold in meters below which gripper is closed.

    Returns:
        Tuple of:
        - gripper_continuous: (N,) float in [0, 1] where 0 = closed, 1 = open.
        - gripper_binary: (N,) float in { -1.0, 1.0 } matching Robosuite (+1=closed, -1=open).
    """
    thumb = keypoints_3d[:, THUMB_TIP_IDX, :]
    index = keypoints_3d[:, INDEX_TIP_IDX, :]
    d_pinch = np.linalg.norm(thumb - index, axis=-1)

    d_min = float(d_pinch.min())
    d_max = float(d_pinch.max())

    if abs(d_max - d_min) < 1e-9:
        gripper_continuous = np.zeros_like(d_pinch)
    else:
        gripper_continuous = (d_pinch - d_min) / (d_max - d_min)

    # Robosuite / LIBERO convention: +1 = closed, -1 = open
    gripper_binary = np.where(d_pinch < pinch_threshold_m, 1.0, -1.0)

    return gripper_continuous, gripper_binary


def track_hands_mediapipe(
    frames_dir: Path | str,
    output_dir: Path | str,
    pinch_threshold: float = 0.04,
    save_debug: bool = False,
) -> Path:
    """Track 3D hand poses and gripper states using MediaPipe Hands.

    Processes RGB video frames directly, extracts 21 3D landmarks in metric space,
    derives wrist SE(3) orientation from palm geometry, computes anatomical scaling,
    and calculates binary/continuous gripper aperture.

    Args:
        frames_dir: Directory containing extracted video frames.
        output_dir: Destination folder for output files.
        pinch_threshold: Distance (m) between thumb and index tip below which gripper is closed.
        save_debug: Whether to save annotated skeleton images in debug subfolder.

    Returns:
        Path to output hand_poses.npz.
    """
    try:
        import mediapipe as mp
    except ImportError:
        raise ImportError(
            "MediaPipe is not installed. To run automatic 3D hand tracking, install it with:\n\n"
            "    pip install mediapipe\n\n"
            "This provides real-time 3D hand tracking without requiring external model downloads."
        )

    frames_dir = Path(frames_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = get_sorted_image_files(frames_dir)
    if not image_paths:
        raise ValueError(f"No image frames found in: {frames_dir}")

    total_frames = len(image_paths)
    logger.info("Tracking 3D hands with MediaPipe across %d frames...", total_frames)

    debug_dir = out_dir / "hand_debug"
    if save_debug:
        debug_dir.mkdir(parents=True, exist_ok=True)

    keypoints_list: list[Optional[np.ndarray]] = [None] * total_frames
    transforms_list: list[Optional[np.ndarray]] = [None] * total_frames
    detected_count = 0

    # Determine MediaPipe API (Legacy solutions vs. modern 1.x Tasks API)
    use_tasks_api = not hasattr(mp, "solutions")

    if not use_tasks_api:
        # MediaPipe <= 0.10.x Solutions API
        mp_hands = mp.solutions.hands
        mp_draw = mp.solutions.drawing_utils
        hands_detector = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    else:
        # MediaPipe 1.x Tasks API
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        model_path = Path.home() / ".mediapipe" / "hand_landmarker.task"
        if not model_path.is_file():
            logger.info("Downloading official MediaPipe Hand Landmarker model (~7 MB)...")
            import urllib.request
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(model_url, str(model_path))
            logger.info("Downloaded Hand Landmarker to %s", model_path)

        base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        tasks_detector = mp_vision.HandLandmarker.create_from_options(options)

    for i, p in enumerate(image_paths):
        img_bgr = cv2.imread(str(p))
        if img_bgr is None:
            continue
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        world_landmarks = None
        pixel_landmarks = None

        if not use_tasks_api:
            res = hands_detector.process(img_rgb)
            if res.multi_hand_world_landmarks:
                world_landmarks = res.multi_hand_world_landmarks[0].landmark
                if res.multi_hand_landmarks:
                    pixel_landmarks = res.multi_hand_landmarks[0]
        else:
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            res = tasks_detector.detect(mp_img)
            if res.hand_world_landmarks:
                world_landmarks = res.hand_world_landmarks[0]
                if res.hand_landmarks:
                    pixel_landmarks = res.hand_landmarks[0]

        if world_landmarks:
            detected_count += 1
            kpts = np.array([[lm.x, lm.y, lm.z] for lm in world_landmarks], dtype=np.float64)  # (21, 3)

            # Derive 3D wrist orientation matrix from palm vectors
            # v_forward: wrist -> middle MCP
            v_forward = kpts[MIDDLE_MCP_IDX] - kpts[WRIST_IDX]
            norm_f = np.linalg.norm(v_forward)
            v_forward = v_forward / norm_f if norm_f > 1e-9 else np.array([0.0, 0.0, 1.0], dtype=np.float64)

            # v_side: index MCP -> pinky MCP (transverse)
            v_side = kpts[17] - kpts[5]
            norm_s = np.linalg.norm(v_side)
            v_side = v_side / norm_s if norm_s > 1e-9 else np.array([1.0, 0.0, 0.0], dtype=np.float64)

            # v_normal: palm normal via cross product
            v_normal = np.cross(v_forward, v_side)
            norm_n = np.linalg.norm(v_normal)
            v_normal = v_normal / norm_n if norm_n > 1e-9 else np.array([0.0, 1.0, 0.0], dtype=np.float64)

            # Re-orthogonalize v_side
            v_side = np.cross(v_normal, v_forward)
            v_side /= np.linalg.norm(v_side)

            R_wrist = np.column_stack([v_side, v_normal, v_forward])
            t_wrist = kpts[WRIST_IDX]

            T_cam_to_hand = np.eye(4, dtype=np.float64)
            T_cam_to_hand[:3, :3] = R_wrist
            T_cam_to_hand[:3, 3] = t_wrist

            keypoints_list[i] = kpts
            transforms_list[i] = T_cam_to_hand

            if save_debug and (i % 10 == 0 or i == 0):
                annotated = img_bgr.copy()
                if pixel_landmarks is not None:
                    # Draw landmarks on image
                    pts = [(int(lm.x * w), int(lm.y * h)) for lm in pixel_landmarks]
                    for conn in HAND_CONNECTIONS:
                        cv2.line(annotated, pts[conn[0]], pts[conn[1]], (0, 255, 0), 2)
                    for pt in pts:
                        cv2.circle(annotated, pt, 4, (0, 0, 255), -1)
                cv2.imwrite(str(debug_dir / f"hand_debug_{i:06d}.jpg"), annotated)

    if not use_tasks_api:
        hands_detector.close()
    else:
        tasks_detector.close()

    logger.info("MediaPipe detected hand in %d / %d frames (%.1f%%)", detected_count, total_frames, (detected_count / total_frames) * 100.0)

    if detected_count == 0:
        raise RuntimeError("No hand was detected in any frame. Ensure hand is visible and lighting is adequate.")

    # Forward/backward fill missing detections (temporary occlusions during grasp)
    first_idx = next(i for i, k in enumerate(keypoints_list) if k is not None)
    last_kpt = keypoints_list[first_idx]
    last_tf = transforms_list[first_idx]

    filled_kpts: list[np.ndarray] = []
    filled_tfs: list[np.ndarray] = []

    for kpt, tf in zip(keypoints_list, transforms_list):
        if kpt is not None:
            last_kpt = kpt
            last_tf = tf
        filled_kpts.append(last_kpt)
        filled_tfs.append(last_tf)

    all_kpts = np.array(filled_kpts, dtype=np.float64)
    all_tfs = np.array(filled_tfs, dtype=np.float64)

    scaled_kpts, scaled_tfs, _ = enforce_anatomical_scale(all_kpts, all_tfs)
    continuous, binary = compute_gripper_state(scaled_kpts, pinch_threshold_m=pinch_threshold)

    out_npz = out_dir / "hand_poses.npz"
    np.savez_compressed(
        out_npz,
        T_cam_to_hand=scaled_tfs.astype(np.float64),
        gripper_continuous=continuous.astype(np.float64),
        gripper_binary=binary.astype(np.float64),
        keypoints_3d=scaled_kpts.astype(np.float64),
    )

    np.save(out_dir / "hand_poses_cam.npy", scaled_tfs.astype(np.float64))
    np.save(out_dir / "gripper_states.npy", binary.astype(np.float64))

    logger.info("Hand tracking completed successfully -> %s (%d frames)", out_npz, total_frames)
    return out_npz


def process_hand_tracking(
    frames_dir: Path | str,
    output_dir: Path | str,
    hamer_script: Optional[Path | str] = None,
    pinch_threshold: float = 0.04,
    method: str = "auto",
    save_debug: bool = False,
) -> Path:
    """Execute complete Module 3 hand tracking pipeline.

    Supports both HaMeR and MediaPipe Hands tracking with auto-detection.

    Args:
        frames_dir: Directory containing input video frames.
        output_dir: Destination directory.
        hamer_script: Path to HaMeR demo.py.
        pinch_threshold: Gripper closing distance threshold in meters.
        method: 'auto', 'mediapipe', or 'hamer'.
        save_debug: Save visual verification images with hand skeleton.

    Returns:
        Path to output hand_poses.npz.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if method == "mediapipe":
        return track_hands_mediapipe(frames_dir, output_dir, pinch_threshold=pinch_threshold, save_debug=save_debug)

    # Check for HaMeR script or existing HaMeR output
    has_hamer_script = hamer_script is not None and Path(hamer_script).is_file()
    candidates = [
        out_dir / "hamer_output.npz",
        out_dir / "results.npz",
        out_dir / "output.npz",
    ]
    has_existing_hamer = any(c.is_file() for c in candidates)

    if has_hamer_script:
        npz_source = run_hamer_subprocess(hamer_script, frames_dir, out_dir)
    elif has_existing_hamer:
        npz_source = next(c for c in candidates if c.is_file())
    else:
        # Fallback to MediaPipe
        logger.info("No HaMeR model output found in '%s'. Falling back to MediaPipe 3D Hand Tracker...", out_dir)
        try:
            return track_hands_mediapipe(frames_dir, output_dir, pinch_threshold=pinch_threshold, save_debug=save_debug)
        except ImportError as e:
            raise FileNotFoundError(
                f"HaMeR output was not found in '{out_dir}', and MediaPipe is not installed.\n"
                f"To run automatic 3D hand tracking, simply install MediaPipe:\n"
                f"    pip install mediapipe\n"
            ) from e

    data = dict(np.load(npz_source, allow_pickle=True))
    transforms_raw = extract_wrist_transforms(data)
    keypoints_raw = extract_keypoints_3d(data)

    n = min(len(transforms_raw), len(keypoints_raw))
    transforms_raw = transforms_raw[:n]
    keypoints_raw = keypoints_raw[:n]

    keypoints_metric, transforms_metric, _ = enforce_anatomical_scale(keypoints_raw, transforms_raw)
    continuous, binary = compute_gripper_state(keypoints_metric, pinch_threshold_m=pinch_threshold)

    out_npz = out_dir / "hand_poses.npz"
    np.savez_compressed(
        out_npz,
        T_cam_to_hand=transforms_metric.astype(np.float64),
        gripper_continuous=continuous.astype(np.float64),
        gripper_binary=binary.astype(np.float64),
        keypoints_3d=keypoints_metric.astype(np.float64),
    )

    np.save(out_dir / "hand_poses_cam.npy", transforms_metric.astype(np.float64))
    np.save(out_dir / "gripper_states.npy", binary.astype(np.float64))

    logger.info("Hand tracking completed successfully -> %s (%d frames)", out_npz, n)
    return out_npz
