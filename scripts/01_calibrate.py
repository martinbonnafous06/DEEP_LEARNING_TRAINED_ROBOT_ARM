"""scripts/01_calibrate.py — CLI entry point for Stage 1: Camera Intrinsic Calibration.

Usage:
    python scripts/01_calibrate.py --image_dir data/calib_images --output_dir data/calibration
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.calibration.calibrator import (
    calibrate_camera,
    detect_chessboard_corners,
    load_calibration_images,
    save_calibration_bundle,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("01_calibrate")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="01_calibrate.py",
        description="Stage 1: Camera Intrinsic Calibration with OpenCV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing checkerboard calibration photos.")
    parser.add_argument("--output_dir", type=str, default="data/calibration", help="Destination folder for calibration files.")
    parser.add_argument("--board_width", type=int, default=9, help="Internal corners along board width.")
    parser.add_argument("--board_height", type=int, default=6, help="Internal corners along board height.")
    parser.add_argument("--square_size", type=float, default=0.025, help="Physical square size in metres (e.g. 0.025 for 25mm).")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)

    logger.info("=== Stage 1: Camera Calibration ===")
    images, paths = load_calibration_images(image_dir)
    obj_points, img_points, image_size = detect_chessboard_corners(images, args.board_width, args.board_height, paths)
    rms, K, dist, rvecs, tvecs = calibrate_camera(obj_points, img_points, image_size, args.square_size)
    save_calibration_bundle(output_dir, K, dist, rms, image_size, rvecs, tvecs)

    logger.info("Stage 1 complete. Reprojection RMS = %.4f px", rms)
    return 0


if __name__ == "__main__":
    sys.exit(main())
