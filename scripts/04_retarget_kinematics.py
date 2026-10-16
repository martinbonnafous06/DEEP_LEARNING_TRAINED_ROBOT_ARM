"""scripts/04_retarget_kinematics.py — CLI entry point for Stage 4: Geometric Fusion & Trajectory Filtering.

Usage:
    python scripts/04_retarget_kinematics.py \
        --camera_poses data/slam/camera_poses.npz \
        --table_transform data/slam/table_transform.npz \
        --hand_poses data/hand_poses/hand_poses.npz \
        --output_dir data/retargeted
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retargeting.transformer import process_kinematics_retargeting

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("04_retarget_kinematics")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="04_retarget_kinematics.py",
        description="Stage 4: Coordinate Alignment, 20Hz Resampling, Smoothing, and OSC Action Calculation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--camera_poses", type=str, required=True, help="Camera poses (.npy or .npz).")
    parser.add_argument("--table_transform", type=str, required=True, help="Table transform (table_transform.npz).")
    parser.add_argument("--hand_poses", type=str, required=True, help="Hand poses (hand_poses.npz).")
    parser.add_argument("--output_dir", type=str, default="data/retargeted", help="Destination folder for retargeted.npz.")
    parser.add_argument("--src_fps", type=float, default=30.0, help="Source video FPS.")
    parser.add_argument("--tgt_fps", type=float, default=20.0, help="Target robot control rate in Hz (default 20 Hz).")
    parser.add_argument("--robot_table_transform_npy", type=str, default=None, help="Optional custom robot-to-table transform (.npy).")
    parser.add_argument("--window_length", type=int, default=15, help="Savitzky-Golay smoothing window length.")
    parser.add_argument("--polyorder", type=int, default=3, help="Savitzky-Golay polynomial order.")
    args = parser.parse_args()

    logger.info("=== Stage 4: Kinematics Retargeting & Trajectory Filtering ===")
    process_kinematics_retargeting(
        camera_poses=args.camera_poses,
        table_transform=args.table_transform,
        hand_poses=args.hand_poses,
        output_dir=args.output_dir,
        src_fps=args.src_fps,
        tgt_fps=args.tgt_fps,
        robot_table_transform_npy=args.robot_table_transform_npy,
        window_length=args.window_length,
        polyorder=args.polyorder,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
