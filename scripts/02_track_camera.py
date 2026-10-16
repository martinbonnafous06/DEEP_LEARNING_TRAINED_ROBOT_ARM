"""scripts/02_track_camera.py — CLI entry point for Stage 2: Visual SLAM & Table Plane Extraction.

Usage:
    python scripts/02_track_camera.py --frames_dir data/frames --calib_txt data/calibration/calib.txt --output_dir data/slam
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.slam.aruco_tracker import track_camera_with_aruco
from src.slam.plane_detector import process_camera_tracking

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("02_track_camera")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="02_track_camera.py",
        description="Stage 2: Camera Tracking & Table Plane Estimation (ArUco Fiducial or Monocular Visual SLAM).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--frames_dir", type=str, required=True, help="Directory containing extracted video frames.")
    parser.add_argument("--calib_txt", type=str, required=True, help="Camera calibration file (fx fy cx cy).")
    parser.add_argument("--output_dir", type=str, default="data/slam", help="Output directory for poses and table transform.")

    # ArUco tracking options
    parser.add_argument("--use_aruco", action="store_true", default=True, help="Use tabletop ArUco marker for exact metric tracking & table frame.")
    parser.add_argument("--no_aruco", dest="use_aruco", action="store_false", help="Disable ArUco tracking and use DROID-SLAM instead.")
    parser.add_argument("--aruco_id", type=int, default=0, help="ArUco marker ID printed on table.")
    parser.add_argument("--marker_size", type=float, default=0.05, help="Physical marker edge size in metres (e.g. 0.05 for 50mm, 0.10 for 100mm).")
    parser.add_argument("--aruco_dict", type=str, default="DICT_ARUCO_ORIGINAL", help="ArUco dictionary (default: DICT_ARUCO_ORIGINAL).")
    parser.add_argument("--save_debug", action="store_true", help="Save annotated debug images showing detected axes.")

    # SLAM options (fallback)
    parser.add_argument("--slam_script", type=str, default=None, help="Path to DROID-SLAM demo.py.")
    parser.add_argument("--distance_threshold", type=float, default=0.01, help="RANSAC plane distance threshold in metres.")
    args = parser.parse_args()

    logger.info("=== Stage 2: Camera Tracking & Table Plane Estimation ===")

    if args.use_aruco:
        logger.info(
            "Tracking via tabletop ArUco Tag (ID=%d, size=%.1f cm, dict=%s)...",
            args.aruco_id,
            args.marker_size * 100.0,
            args.aruco_dict,
        )
        npy_path, npz_path, transform_out, result = track_camera_with_aruco(
            frames_dir=args.frames_dir,
            calib_txt=args.calib_txt,
            output_dir=args.output_dir,
            marker_id=args.aruco_id,
            marker_size=args.marker_size,
            dictionary_name=args.aruco_dict,
            save_annotated_debug_images=args.save_debug,
        )
        logger.info("Stage 2 (ArUco) complete. Detection ratio: %.1f%%", result.detection_ratio * 100.0)
    else:
        logger.info("Tracking via monocular Visual SLAM and RANSAC plane fitting...")
        process_camera_tracking(
            frames_dir=args.frames_dir,
            calib_txt=args.calib_txt,
            output_dir=args.output_dir,
            slam_script=args.slam_script,
            distance_threshold=args.distance_threshold,
        )
        logger.info("Stage 2 (SLAM) complete.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

