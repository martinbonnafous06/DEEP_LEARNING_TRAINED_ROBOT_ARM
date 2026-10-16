"""calibration.py – Module 1: Camera Intrinsic Calibration.

Detects chessboard corners in a directory of calibration images, solves
camera intrinsics via OpenCV, and exports results in DROID-SLAM format.

Example:
    python calibration.py --image_dir ./calib_images --visualize
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ImageList = list[npt.NDArray[np.uint8]]
PointsList = list[npt.NDArray[np.float32]]


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def load_images(image_dir: str | Path) -> tuple[ImageList, list[str]]:
    """Load all JPEG/PNG images from *image_dir*.

    Args:
        image_dir: Path to directory containing calibration images.

    Returns:
        A 2-tuple of (images, paths) where *images* is a list of BGR
        uint8 arrays and *paths* is the corresponding list of file paths.

    Raises:
        FileNotFoundError: If *image_dir* does not exist.
        RuntimeError: If no images are found inside the directory.
    """
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
    paths: list[str] = []
    for pattern in patterns:
        paths.extend(sorted(glob.glob(str(image_dir / pattern))))
        paths.extend(sorted(glob.glob(str(image_dir / pattern.upper()))))
    # Deduplicate while preserving order
    seen: set[str] = set()
    paths = [p for p in paths if not (p in seen or seen.add(p))]  # type: ignore[func-returns-value]

    if not paths:
        raise RuntimeError(
            f"No images found in '{image_dir}'. "
            "Supported extensions: jpg, jpeg, png, bmp, tif, tiff."
        )

    images: ImageList = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            logger.warning("Could not read image, skipping: %s", p)
            continue
        images.append(img)

    logger.info("Loaded %d image(s) from '%s'.", len(images), image_dir)
    return images, paths


def detect_corners(
    images: ImageList,
    board_width: int,
    board_height: int,
    paths: Optional[list[str]] = None,
) -> tuple[PointsList, PointsList, tuple[int, int]]:
    """Detect sub-pixel chessboard corners in each image.

    Args:
        images: List of BGR calibration images.
        board_width: Number of *internal* corners along the board width.
        board_height: Number of *internal* corners along the board height.
        paths: Optional list of file paths (used only for log messages).

    Returns:
        A 3-tuple ``(obj_points, img_points, image_size)`` where:

        * *obj_points* – list of 3-D object-space corner arrays (float32).
        * *img_points* – list of 2-D image-space corner arrays (float32).
        * *image_size* – ``(width, height)`` pixel dimensions taken from the
          first successful detection.

    Raises:
        RuntimeError: If corners could not be detected in any image.
    """
    board_size: tuple[int, int] = (board_width, board_height)
    num_corners = board_width * board_height

    # Build the canonical object-space corner grid (Z = 0)
    objp: npt.NDArray[np.float32] = np.zeros(
        (num_corners, 3), dtype=np.float32
    )
    objp[:, :2] = np.mgrid[0:board_width, 0:board_height].T.reshape(-1, 2)

    subpix_criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    obj_points: PointsList = []
    img_points: PointsList = []
    image_size: Optional[tuple[int, int]] = None

    for idx, img in enumerate(images):
        label = paths[idx] if paths else f"image[{idx}]"
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        logger.info("Processing %s ...", label)
        found, corners = cv2.findChessboardCorners(
            gray,
            board_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        if not found:
            logger.warning("Chessboard NOT detected in %s - skipping.", label)
            continue

        # Refine to sub-pixel accuracy
        corners_refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), subpix_criteria
        )
        obj_points.append(objp.copy())
        img_points.append(corners_refined)

        if image_size is None:
            h, w = gray.shape[:2]
            image_size = (w, h)

        logger.info("  Corner detection succeeded: %s.", label)

    if not obj_points:
        raise RuntimeError(
            "Chessboard corners could not be detected in ANY image. "
            "Check --board_width / --board_height and image quality."
        )

    assert image_size is not None
    logger.info(
        "Corner detection complete: %d / %d image(s) usable.",
        len(obj_points),
        len(images),
    )
    return obj_points, img_points, image_size


def calibrate(
    obj_points: PointsList,
    img_points: PointsList,
    image_size: tuple[int, int],
    square_size: float,
) -> tuple[float, npt.NDArray[np.float64], npt.NDArray[np.float64], list, list]:
    """Solve camera intrinsics with cv2.calibrateCamera.

    Object-space points are scaled by *square_size* so that the returned
    camera matrix is in consistent metric units.

    Args:
        obj_points: 3-D object-space corner arrays (unit = 1 internal square).
        img_points: 2-D image-space corner arrays (pixels).
        image_size: ``(width, height)`` of the calibration images.
        square_size: Physical size of one chessboard square in metres.

    Returns:
        A 5-tuple ``(rms, camera_matrix, dist_coeffs, rvecs, tvecs)`` where:

        * *rms* – RMS reprojection error in pixels.
        * *camera_matrix* – 3x3 intrinsic matrix (float64).
        * *dist_coeffs* – Distortion coefficient vector (float64).
        * *rvecs* – List of rotation vectors, one per image.
        * *tvecs* – List of translation vectors, one per image.
    """
    # Scale object points to metric units
    scaled_obj_points = [pts * square_size for pts in obj_points]

    logger.info("Running cv2.calibrateCamera ...")
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        scaled_obj_points,  # type: ignore[arg-type]
        img_points,
        image_size,
        None,
        None,
    )

    logger.info("Calibration finished. RMS reprojection error: %.6f px", rms)
    logger.info(
        "Camera matrix:\n%s",
        np.array2string(camera_matrix, precision=4, suppress_small=True),
    )
    logger.info(
        "Distortion coefficients: %s",
        np.array2string(dist_coeffs.ravel(), precision=6, suppress_small=True),
    )
    return rms, camera_matrix, dist_coeffs, list(rvecs), list(tvecs)


def export_calib_txt(
    camera_matrix: npt.NDArray[np.float64],
    output_path: str | Path,
) -> None:
    """Write ``fx fy cx cy`` to *output_path* in DROID-SLAM format.

    The file contains a single whitespace-separated line with the four
    principal focal and principal-point parameters extracted from the 3x3
    camera matrix::

        fx  fy  cx  cy

    Args:
        camera_matrix: 3x3 OpenCV intrinsic matrix.
        output_path: Destination file path (will be created/overwritten).
    """
    fx: float = float(camera_matrix[0, 0])
    fy: float = float(camera_matrix[1, 1])
    cx: float = float(camera_matrix[0, 2])
    cy: float = float(camera_matrix[1, 2])

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"{fx:.8f} {fy:.8f} {cx:.8f} {cy:.8f}\n")

    logger.info("Exported DROID-SLAM calib.txt -> %s", output_path)
    logger.info("  fx=%.4f  fy=%.4f  cx=%.4f  cy=%.4f", fx, fy, cx, cy)


def save_calibration_npz(
    output_path: str | Path,
    camera_matrix: npt.NDArray[np.float64],
    dist_coeffs: npt.NDArray[np.float64],
    rvecs: list[npt.NDArray[np.float64]],
    tvecs: list[npt.NDArray[np.float64]],
    rms: float,
) -> None:
    """Persist the full calibration bundle to a compressed NumPy archive.

    Saves the following arrays under their respective keys:

    * ``camera_matrix`` - 3x3 float64 intrinsic matrix.
    * ``dist_coeffs``   - Distortion coefficient vector.
    * ``rvecs``         - Stacked rotation vectors (Nx3x1).
    * ``tvecs``         - Stacked translation vectors (Nx3x1).
    * ``rms``           - Scalar RMS reprojection error.

    Args:
        output_path: Destination ``.npz`` file path.
        camera_matrix: 3x3 OpenCV intrinsic matrix (float64).
        dist_coeffs: Distortion coefficient vector (float64).
        rvecs: List of per-image rotation vectors.
        tvecs: List of per-image translation vectors.
        rms: RMS reprojection error in pixels.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        str(output_path),
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rvecs=np.array(rvecs, dtype=np.float64),
        tvecs=np.array(tvecs, dtype=np.float64),
        rms=np.float64(rms),
    )
    logger.info("Saved full calibration bundle -> %s", output_path)


def visualize_undistorted(
    image: npt.NDArray[np.uint8],
    camera_matrix: npt.NDArray[np.float64],
    dist_coeffs: npt.NDArray[np.float64],
) -> None:
    """Display the original and undistorted image side-by-side via cv2.imshow.

    The window blocks until any key is pressed (``cv2.waitKey(0)``).

    Args:
        image: BGR uint8 calibration image to undistort.
        camera_matrix: 3x3 OpenCV intrinsic matrix (float64).
        dist_coeffs: Distortion coefficient vector (float64).
    """
    h, w = image.shape[:2]
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), alpha=1, newImgSize=(w, h)
    )
    undistorted = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # Crop to valid ROI
    x, y, rw, rh = roi
    if rw > 0 and rh > 0:
        undistorted = undistorted[y : y + rh, x : x + rw]

    # Resize original to match undistorted height for side-by-side display
    orig_resized = cv2.resize(image, (undistorted.shape[1], undistorted.shape[0]))
    comparison = np.hstack([orig_resized, undistorted])

    # Draw a dividing line
    mid_x = orig_resized.shape[1]
    cv2.line(comparison, (mid_x, 0), (mid_x, comparison.shape[0]), (0, 255, 0), 2)

    # Label each half
    font_args = dict(
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.8,
        color=(0, 255, 0),
        thickness=2,
        lineType=cv2.LINE_AA,
    )
    cv2.putText(comparison, "Original", (10, 30), **font_args)
    cv2.putText(comparison, "Undistorted", (mid_x + 10, 30), **font_args)

    logger.info("Displaying undistorted comparison. Press any key to close.")
    cv2.imshow("Calibration - Original vs Undistorted", comparison)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list to parse. Defaults to ``sys.argv[1:]``.

    Returns:
        Parsed :class:`argparse.Namespace` with all calibration parameters.
    """
    parser = argparse.ArgumentParser(
        prog="calibration.py",
        description=(
            "Module 1 - Camera Intrinsic Calibration.\n"
            "Detects chessboard corners, solves intrinsics, and exports\n"
            "calib.txt (DROID-SLAM format) + calibration.npz."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--image_dir",
        required=True,
        type=str,
        help="Directory containing 15-25 chessboard calibration images.",
    )
    parser.add_argument(
        "--board_width",
        type=int,
        default=9,
        metavar="W",
        help="Number of internal corners along board width (default: 9).",
    )
    parser.add_argument(
        "--board_height",
        type=int,
        default=6,
        metavar="H",
        help="Number of internal corners along board height (default: 6).",
    )
    parser.add_argument(
        "--square_size",
        type=float,
        default=0.025,
        metavar="M",
        help="Physical size of one chessboard square in metres (default: 0.025).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Directory for calib.txt and calibration.npz. "
            "Defaults to --image_dir."
        ),
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help=(
            "Show an OpenCV window with the original vs undistorted first image."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Run the full calibration pipeline.

    Steps:
        1. Load calibration images from *image_dir*.
        2. Detect chessboard corners with sub-pixel refinement.
        3. Solve camera intrinsics via ``cv2.calibrateCamera``.
        4. Export ``calib.txt`` (DROID-SLAM) and ``calibration.npz``.
        5. Optionally display undistorted first image.

    Args:
        argv: Optional argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Exit code - ``0`` on success, ``1`` on failure.
    """
    args = parse_args(argv)

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir) if args.output_dir else image_dir

    logger.info("=== Camera Calibration - Module 1 ===")
    logger.info("Image directory : %s", image_dir)
    logger.info(
        "Board size      : %d x %d internal corners",
        args.board_width,
        args.board_height,
    )
    logger.info("Square size     : %.4f m", args.square_size)
    logger.info("Output directory: %s", output_dir)

    # ------------------------------------------------------------------
    # Step 1 - Load images
    # ------------------------------------------------------------------
    try:
        images, paths = load_images(image_dir)
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("Image loading failed: %s", exc)
        return 1

    if len(images) < 4:
        logger.warning(
            "Only %d usable image(s) found. Recommend at least 15 for robust calibration.",
            len(images),
        )

    # ------------------------------------------------------------------
    # Step 2 - Detect corners
    # ------------------------------------------------------------------
    try:
        obj_points, img_points, image_size = detect_corners(
            images, args.board_width, args.board_height, paths
        )
    except RuntimeError as exc:
        logger.error("Corner detection failed: %s", exc)
        return 1

    # ------------------------------------------------------------------
    # Step 3 - Calibrate
    # ------------------------------------------------------------------
    try:
        rms, camera_matrix, dist_coeffs, rvecs, tvecs = calibrate(
            obj_points, img_points, image_size, args.square_size
        )
    except cv2.error as exc:
        logger.error("cv2.calibrateCamera failed: %s", exc)
        return 1

    logger.info("RMS reprojection error: %.6f px", rms)
    if rms > 1.0:
        logger.warning(
            "RMS error is high (%.4f > 1.0 px). "
            "Consider recapturing images or verifying board dimensions.",
            rms,
        )

    # ------------------------------------------------------------------
    # Step 4 - Export
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    calib_txt_path = output_dir / "calib.txt"
    export_calib_txt(camera_matrix, calib_txt_path)

    npz_path = output_dir / "calibration.npz"
    save_calibration_npz(npz_path, camera_matrix, dist_coeffs, rvecs, tvecs, rms)

    # Export calibration.json
    import json
    json_path = output_dir / "calibration.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "rms_reprojection_error": float(rms),
                "image_width": image_size[0],
                "image_height": image_size[1],
                "fx": float(camera_matrix[0, 0]),
                "fy": float(camera_matrix[1, 1]),
                "cx": float(camera_matrix[0, 2]),
                "cy": float(camera_matrix[1, 2]),
                "camera_matrix": camera_matrix.tolist(),
                "distortion_coefficients": dist_coeffs.ravel().tolist(),
            },
            f,
            indent=2,
        )

    logger.info("=== Calibration complete ===")
    logger.info("  calib.txt        -> %s", calib_txt_path)
    logger.info("  calibration.npz  -> %s", npz_path)
    logger.info("  calibration.json -> %s", json_path)
    logger.info("  RMS              = %.6f px", rms)

    # ------------------------------------------------------------------
    # Step 5 - Optional visualisation
    # ------------------------------------------------------------------
    if args.visualize:
        visualize_undistorted(images[0], camera_matrix, dist_coeffs)

    return 0


if __name__ == "__main__":
    sys.exit(main())
