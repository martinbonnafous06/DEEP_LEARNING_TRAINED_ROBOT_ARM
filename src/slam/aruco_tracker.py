"""src.slam.aruco_tracker — High-precision fiducial tracking and table frame calibration.

Uses an ArUco tag (default ID 0) placed on the tabletop surface to:
1. Define the ground-truth table reference frame (Z=0 at tabletop, +Z normal pointing up).
2. Directly track 6DoF camera poses (T_world_to_cam) in metric units (metres).
3. Resolve monocular metric scale ambiguity without requiring depth sensors or LiDAR.
4. Output standard poses.npy, camera_poses.npz, and table_transform.npz compatible
   with Stage 3 (hand tracking), Stage 4 (retargeting), and Stage 5 (LIBERO export).
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import numpy.typing as npt

from ..utils.coordinates import invert_transform_se3
from ..utils.io import get_sorted_image_files, load_calibration_txt, save_camera_poses

logger = logging.getLogger("aruco_tracker")

# Predefined dictionary name mapping
ARUCO_DICTIONARIES: Dict[str, int] = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}


def get_aruco_detector(dictionary_name: str = "DICT_ARUCO_ORIGINAL") -> Tuple[cv2.aruco.Dictionary, cv2.aruco.DetectorParameters]:
    """Retrieve OpenCV ArUco dictionary and detector parameters.

    Args:
        dictionary_name: Name of predefined dictionary (defaults to 'DICT_ARUCO_ORIGINAL').

    Returns:
        Tuple of (dictionary, parameters).
    """
    normalized_name = dictionary_name.upper().strip()
    if normalized_name not in ARUCO_DICTIONARIES:
        raise ValueError(
            f"Unknown ArUco dictionary '{dictionary_name}'. "
            f"Available options: {list(ARUCO_DICTIONARIES.keys())}"
        )

    dict_id = ARUCO_DICTIONARIES[normalized_name]
    dictionary = cv2.aruco.getPredefinedDictionary(dict_id)

    # Detector parameters
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        parameters = cv2.aruco.DetectorParameters_create()
    else:
        parameters = cv2.aruco.DetectorParameters()

    # Optimize for monocular phone cameras: subpixel corner refinement
    if hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX"):
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    return dictionary, parameters


def detect_markers(
    image: np.ndarray,
    dictionary_name: str = "DICT_ARUCO_ORIGINAL",
) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
    """Detect ArUco markers in an image using modern or legacy OpenCV API.

    Args:
        image: BGR or grayscale image.
        dictionary_name: ArUco dictionary name.

    Returns:
        Tuple of (corners, ids).
    """
    dictionary, parameters = get_aruco_detector(dictionary_name)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    # OpenCV 4.7+ provides ArucoDetector class
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)

    return corners, ids


def estimate_marker_pose_pnp(
    corners: np.ndarray,
    marker_size: float,
    camera_matrix: np.ndarray,
    dist_coeffs: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve PnP for a single ArUco marker to get Rvec and Tvec.

    In the marker coordinate system:
    - Origin (0, 0, 0) is at marker centre.
    - Corners are at (+-marker_size/2, +-marker_size/2, 0).
    - Normal vector [0, 0, 1] points perpendicularly out of the marker face.

    Args:
        corners: (1, 4, 2) or (4, 2) corner points in pixel coordinates.
        marker_size: Physical length of marker square edge in metres.
        camera_matrix: (3, 3) intrinsic camera matrix.
        dist_coeffs: Lens distortion coefficients (optional).

    Returns:
        Tuple of (rvec, tvec) representing transform from marker to camera.
    """
    half_size = float(marker_size) / 2.0
    # Object points defined in clockwise order matching OpenCV convention
    obj_points = np.array([
        [-half_size,  half_size, 0.0],
        [ half_size,  half_size, 0.0],
        [ half_size, -half_size, 0.0],
        [-half_size, -half_size, 0.0],
    ], dtype=np.float64)

    img_points = np.ascontiguousarray(corners.reshape(4, 2), dtype=np.float64)
    if dist_coeffs is None:
        dist_coeffs = np.zeros((5, 1), dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        obj_points,
        img_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not success:
        # Fallback to standard iterative solver
        success, rvec, tvec = cv2.solvePnP(
            obj_points,
            img_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

    return rvec, tvec


def rvec_tvec_to_matrix(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """Convert Rodrigues rvec (3,) and tvec (3,) into a 4x4 homogeneous matrix."""
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T


@dataclass
class ArucoTrackingResult:
    """Summary of ArUco tracking across frame sequence."""

    total_frames: int
    detected_frames: int
    detection_ratio: float
    marker_id: int
    marker_size_metres: float
    dictionary_name: str
    output_dir: str


def track_camera_with_aruco(
    frames_dir: Union[Path, str],
    calib_txt: Union[Path, str],
    output_dir: Union[Path, str],
    marker_id: int = 0,
    marker_size: float = 0.05,
    dictionary_name: str = "DICT_ARUCO_ORIGINAL",
    dist_coeffs: Optional[np.ndarray] = None,
    save_annotated_debug_images: bool = False,
) -> Tuple[Path, Path, Path, ArucoTrackingResult]:
    """Track 6DoF camera poses and establish tabletop coordinate frame using Tag 0.

    When the marker with ID 0 is placed flat on the table:
    1. Table surface lies at Z = 0 in the marker/table coordinate frame.
    2. The marker center is the origin (0, 0, 0).
    3. The camera pose in table frame is T_table_to_cam = inv(T_cam_to_marker).
    4. For SLAM convention, T_world_to_cam = inv(T_table_to_cam) = T_cam_to_marker.
    5. Table transformation matrix T_world_to_table is simply identity np.eye(4)
       since the world frame is anchored directly at the table marker!

    Args:
        frames_dir: Directory containing extracted video frames.
        calib_txt: File containing camera intrinsics (fx fy cx cy).
        output_dir: Destination folder for poses.npy, camera_poses.npz, table_transform.npz.
        marker_id: Target ArUco marker ID (default: 0).
        marker_size: Physical edge size of marker in metres (e.g. 0.05 for 50 mm).
        dictionary_name: Predefined OpenCV ArUco dictionary (default: 'DICT_4X4_50').
        dist_coeffs: Optional lens distortion array (1x5 or 1x8).
        save_annotated_debug_images: If True, saves visual verification images with axes.

    Returns:
        Tuple of (poses_npy, camera_poses_npz, table_transform_npz, tracking_result).
    """
    frames_dir = Path(frames_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fx, fy, cx, cy = load_calibration_txt(calib_txt)
    camera_matrix = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    image_paths = get_sorted_image_files(frames_dir)
    if not image_paths:
        raise ValueError(f"No image frames found in: {frames_dir}")

    total_frames = len(image_paths)
    logger.info("Tracking camera using ArUco Tag ID %d across %d frames...", marker_id, total_frames)

    # Store T_cam_to_tag for each frame
    T_cam_to_tag_list: List[Optional[np.ndarray]] = [None] * total_frames
    detected_count = 0

    debug_dir = out_dir / "aruco_debug"
    if save_annotated_debug_images:
        debug_dir.mkdir(parents=True, exist_ok=True)

    for i, img_path in enumerate(image_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        corners, ids = detect_markers(img, dictionary_name)
        if ids is not None and len(ids) > 0:
            flat_ids = ids.flatten()
            if marker_id in flat_ids:
                idx = int(np.where(flat_ids == marker_id)[0][0])
                marker_corners = corners[idx]

                rvec, tvec = estimate_marker_pose_pnp(
                    corners=marker_corners,
                    marker_size=marker_size,
                    camera_matrix=camera_matrix,
                    dist_coeffs=dist_coeffs,
                )
                T_c2m = rvec_tvec_to_matrix(rvec, tvec)
                T_cam_to_tag_list[i] = T_c2m
                detected_count += 1

                if save_annotated_debug_images and (i % 10 == 0 or i == 0):
                    annotated = img.copy()
                    cv2.aruco.drawDetectedMarkers(annotated, [marker_corners], np.array([[marker_id]]))
                    cv2.drawFrameAxes(annotated, camera_matrix, dist_coeffs if dist_coeffs is not None else np.zeros((5, 1)), rvec, tvec, marker_size * 0.75)
                    cv2.imwrite(str(debug_dir / f"tag_debug_{i:06d}.jpg"), annotated)

    detection_ratio = detected_count / total_frames if total_frames > 0 else 0.0
    logger.info(
        "ArUco Tag %d detected in %d / %d frames (%.1f%%)",
        marker_id,
        detected_count,
        total_frames,
        detection_ratio * 100.0,
    )

    if detected_count == 0:
        raise RuntimeError(
            f"ArUco Tag ID {marker_id} was not detected in any frame. "
            f"Check dictionary ('{dictionary_name}') and verify tag visibility."
        )

    # Forward/backward fill missing poses (if marker was briefly occluded by human hand)
    filled_poses: List[np.ndarray] = []
    last_valid: Optional[np.ndarray] = None

    # Find first valid pose
    first_valid_idx = next(i for i, p in enumerate(T_cam_to_tag_list) if p is not None)
    first_valid = T_cam_to_tag_list[first_valid_idx]

    for p in T_cam_to_tag_list:
        if p is not None:
            last_valid = p
            filled_poses.append(p)
        else:
            # Impute using last known pose (or first valid if before detection)
            filled_poses.append(last_valid if last_valid is not None else first_valid)

    # In our pipeline:
    # World frame is defined as the table frame at Tag 0:
    # T_cam_to_tag maps points in camera frame to tag/table frame.
    # Therefore, T_world_to_cam = inv(T_cam_to_tag) maps world/table points to camera frame.
    T_cam_to_tag_arr = np.array(filled_poses, dtype=np.float64)  # (N, 4, 4)
    T_world_to_cam_arr = invert_transform_se3(T_cam_to_tag_arr)   # (N, 4, 4)

    # Save camera poses
    npy_path, npz_path = save_camera_poses(out_dir, T_world_to_cam_arr)

    # Since world coordinate frame is defined by the table tag itself:
    # Table plane is Z=0, normal is [0, 0, 1], distance is 0.
    plane_model = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
    T_world_to_table = np.eye(4, dtype=np.float64)
    R_align = np.eye(3, dtype=np.float64)
    t_align = np.zeros(3, dtype=np.float64)

    table_transform_path = out_dir / "table_transform.npz"
    np.savez_compressed(
        str(table_transform_path),
        T_world_to_table=T_world_to_table,
        R_align=R_align,
        t_align=t_align,
        plane_model=plane_model,
        marker_id=marker_id,
        marker_size_metres=marker_size,
    )

    result = ArucoTrackingResult(
        total_frames=total_frames,
        detected_frames=detected_count,
        detection_ratio=detection_ratio,
        marker_id=marker_id,
        marker_size_metres=marker_size,
        dictionary_name=dictionary_name,
        output_dir=str(out_dir),
    )

    with open(out_dir / "aruco_tracking_manifest.json", "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, indent=2)

    logger.info("ArUco tracking successfully saved:")
    logger.info("  Poses (.npy)       -> %s", npy_path)
    logger.info("  Poses (.npz)       -> %s", npz_path)
    logger.info("  Table transform    -> %s", table_transform_path)

    return npy_path, npz_path, table_transform_path, result
