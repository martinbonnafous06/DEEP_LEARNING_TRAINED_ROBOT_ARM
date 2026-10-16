"""src.calibration.calibrator — Camera intrinsic calibration using OpenCV.

Detects chessboard corners across a set of calibration images, computes the
intrinsic matrix K and distortion coefficients, evaluates RMS reprojection error,
and exports results in DROID-SLAM format (calib.txt), NPZ, and JSON formats.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import numpy.typing as npt

from ..utils.io import export_calibration_txt, get_sorted_image_files

logger = logging.getLogger("calibration")


def load_calibration_images(image_dir: str | Path) -> tuple[list[npt.NDArray[np.uint8]], list[Path]]:
    """Load all valid images from image_dir in natural sorted order.

    Args:
        image_dir: Path to directory containing calibration images.

    Returns:
        Tuple of (images_list, paths_list).

    Raises:
        FileNotFoundError: If image_dir does not exist.
        RuntimeError: If no valid images can be read.
    """
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Calibration image directory not found: {image_dir}")

    paths = get_sorted_image_files(image_dir)
    if not paths:
        raise RuntimeError(
            f"No image files found in '{image_dir}'. Supported: png, jpg, jpeg, bmp, tiff."
        )

    images: list[npt.NDArray[np.uint8]] = []
    valid_paths: list[Path] = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            logger.warning("Could not read image, skipping: %s", p)
            continue
        images.append(img)
        valid_paths.append(p)

    if not images:
        raise RuntimeError(f"Failed to decode any images in {image_dir}.")

    logger.info("Loaded %d calibration image(s) from '%s'.", len(images), image_dir)
    return images, valid_paths


def detect_chessboard_corners(
    images: list[npt.NDArray[np.uint8]],
    board_width: int = 9,
    board_height: int = 6,
    paths: Optional[list[Path]] = None,
) -> tuple[list[npt.NDArray[np.float32]], list[npt.NDArray[np.float32]], tuple[int, int]]:
    """Detect sub-pixel chessboard internal corners.

    Args:
        images: List of BGR images.
        board_width: Internal corners along width (default 9).
        board_height: Internal corners along height (default 6).
        paths: Optional file paths for diagnostic logging.

    Returns:
        Tuple of (object_points, image_points, (width, height)).
    """
    board_size = (board_width, board_height)
    num_corners = board_width * board_height

    # Canonical 3D grid in board coordinates with Z = 0
    objp = np.zeros((num_corners, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:board_width, 0:board_height].T.reshape(-1, 2)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    obj_points: list[npt.NDArray[np.float32]] = []
    img_points: list[npt.NDArray[np.float32]] = []
    image_size: Optional[tuple[int, int]] = None

    for idx, img in enumerate(images):
        label = paths[idx].name if paths and idx < len(paths) else f"image_{idx}"
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(
            gray,
            board_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        if not found:
            logger.warning("Chessboard NOT detected in %s — skipping.", label)
            continue

        corners_subpix = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )
        obj_points.append(objp.copy())
        img_points.append(corners_subpix)

        if image_size is None:
            h, w = gray.shape[:2]
            image_size = (w, h)

        logger.info("Corner detection succeeded: %s", label)

    if not obj_points:
        raise RuntimeError(
            "Chessboard corners could not be detected in ANY image. "
            "Please verify board dimensions (e.g., 9x6) and photo clarity."
        )

    assert image_size is not None
    logger.info("Corner detection complete: %d / %d image(s) usable.", len(obj_points), len(images))
    return obj_points, img_points, image_size


def calibrate_camera(
    obj_points: list[npt.NDArray[np.float32]],
    img_points: list[npt.NDArray[np.float32]],
    image_size: tuple[int, int],
    square_size_m: float = 0.025,
) -> tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64], list, list]:
    """Solve for camera intrinsics and distortion coefficients using OpenCV.

    Args:
        obj_points: 3D corner coordinates (unit = grid square).
        img_points: 2D corner detections in pixels.
        image_size: (width, height) in pixels.
        square_size_m: Physical side length of each square in metres (default 0.025 m).

    Returns:
        Tuple of (rms_error, camera_matrix, dist_coeffs, rvecs, tvecs).
    """
    scaled_obj_points = [pts * square_size_m for pts in obj_points]

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        scaled_obj_points,
        img_points,
        image_size,
        None,
        None,
    )

    logger.info("Calibration solved. RMS reprojection error: %.6f px", rms)
    if rms > 1.0:
        logger.warning(
            "RMS reprojection error is above recommended threshold (%.4f > 1.0 px).", rms
        )

    return rms, camera_matrix, dist_coeffs, list(rvecs), list(tvecs)


def save_calibration_bundle(
    output_dir: Path | str,
    camera_matrix: npt.NDArray[np.float64],
    dist_coeffs: npt.NDArray[np.float64],
    rms: float,
    image_size: tuple[int, int],
    rvecs: Optional[list] = None,
    tvecs: Optional[list] = None,
) -> tuple[Path, Path, Path]:
    """Save calibration results to calib.txt (DROID-SLAM), calibration.npz, and calibration.json.

    Args:
        output_dir: Directory where artefacts are stored.
        camera_matrix: 3x3 intrinsic matrix.
        dist_coeffs: Distortion coefficients vector.
        rms: RMS reprojection error.
        image_size: (width, height) in pixels.
        rvecs: Optional rotation vectors.
        tvecs: Optional translation vectors.

    Returns:
        Tuple of (calib_txt_path, npz_path, json_path).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. DROID-SLAM calib.txt
    calib_txt_path = out_dir / "calib.txt"
    export_calibration_txt(camera_matrix, calib_txt_path)

    # 2. calibration.npz
    npz_path = out_dir / "calibration.npz"
    np.savez_compressed(
        npz_path,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rms=np.float64(rms),
        image_size=np.array(image_size),
        rvecs=np.array(rvecs, dtype=np.float64) if rvecs is not None else np.array([]),
        tvecs=np.array(tvecs, dtype=np.float64) if tvecs is not None else np.array([]),
    )

    # 3. calibration.json
    json_path = out_dir / "calibration.json"
    calib_dict = {
        "rms_reprojection_error": float(rms),
        "image_width": image_size[0],
        "image_height": image_size[1],
        "fx": float(camera_matrix[0, 0]),
        "fy": float(camera_matrix[1, 1]),
        "cx": float(camera_matrix[0, 2]),
        "cy": float(camera_matrix[1, 2]),
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": dist_coeffs.ravel().tolist(),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(calib_dict, f, indent=2)

    logger.info("Saved calibration bundle:")
    logger.info("  calib.txt       -> %s", calib_txt_path)
    logger.info("  calibration.npz -> %s", npz_path)
    logger.info("  calibration.json-> %s", json_path)

    return calib_txt_path, npz_path, json_path
