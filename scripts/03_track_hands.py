"""scripts/03_track_hands.py — CLI entry point for Stage 3: 3D Hand Pose & Gripper State Estimation.

Usage:
    python scripts/03_track_hands.py --frames_dir data/frames --output_dir data/hand_poses
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hand_pose.hand_tracker import process_hand_tracking

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("03_track_hands")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="03_track_hands.py",
        description="Stage 3: 3D Hand Pose Tracking, Metric Scaling, and Gripper Aperture Extraction (MediaPipe / HaMeR).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--frames_dir", type=str, required=True, help="Directory containing RGB frames.")
    parser.add_argument("--output_dir", type=str, default="data/hand_poses", help="Directory for output hand poses.")
    parser.add_argument("--method", type=str, choices=["auto", "mediapipe", "hamer"], default="auto", help="Tracking backend (default: auto).")
    parser.add_argument("--hamer_script", type=str, default=None, help="Path to HaMeR demo.py entry point (optional).")
    parser.add_argument("--pinch_threshold", type=float, default=0.04, help="Pinch distance (m) below which gripper is closed.")
    parser.add_argument("--save_debug", action="store_true", help="Save annotated debug images showing detected hand skeleton.")
    args = parser.parse_args()

    logger.info("=== Stage 3: 3D Hand Pose Tracking ===")
    process_hand_tracking(
        frames_dir=args.frames_dir,
        output_dir=args.output_dir,
        hamer_script=args.hamer_script,
        pinch_threshold=args.pinch_threshold,
        method=args.method,
        save_debug=args.save_debug,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
