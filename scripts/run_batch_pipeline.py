"""scripts/run_batch_pipeline.py — One-click end-to-end automation for multi-video retargeting.

Processes all demonstration videos/frames in a single command:
  1. (Optional) Stage 0: Extract videos from data/raw_videos to data/frames at 20 Hz
  2. Stage 2: Camera & Table Plane Tracking via ArUco Tag 0 (Original format)
  3. Stage 3: 3D Hand Tracking & Pinch Detection (MediaPipe Hands)
  4. Stage 4: Geometric Kinematics Retargeting to Franka Panda Robot (20 Hz)
  5. Stage 5: Package and Append into a unified LIBERO Benchmark HDF5 Dataset

Usage Examples:
    # 1. Run all remaining stages for all demos in data/frames/:
    python scripts/run_batch_pipeline.py

    # 2. Run completely from raw videos to final HDF5 in one click:
    python scripts/run_batch_pipeline.py --video_dir data/raw_videos

    # 3. Custom tag size or target environment:
    python scripts/run_batch_pipeline.py --marker_size 0.05 --env_name LIBERO_OBJECT
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hand_pose.hand_tracker import process_hand_tracking
from src.retargeting.libero_exporter import export_libero_dataset
from src.retargeting.transformer import process_kinematics_retargeting
from src.slam.aruco_tracker import track_camera_with_aruco
from src.utils.io import natural_sort_key
from src.utils.video_processor import batch_transform_videos_to_frames

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("batch_pipeline")


def get_demo_directories(frames_root: Path) -> list[Path]:
    """Find and naturally sort all demonstration directories inside frames_root."""
    if not frames_root.is_dir():
        return []
    # Match any subdirectory that contains image files
    subdirs = [p for p in frames_root.iterdir() if p.is_dir()]
    valid_dirs = []
    for d in subdirs:
        # Check if contains frame images
        if any(d.glob("*.png")) or any(d.glob("*.jpg")) or any(d.glob("*.jpeg")):
            valid_dirs.append(d)
    return sorted(valid_dirs, key=natural_sort_key)


def run_pipeline_for_demo(
    demo_dir: Path,
    demo_idx: int,
    calib_txt: Path,
    output_hdf5: Path,
    base_data_dir: Path,
    marker_size: float = 0.05,
    aruco_id: int = 0,
    aruco_dict: str = "DICT_ARUCO_ORIGINAL",
    pinch_threshold: float = 0.04,
    tgt_fps: float = 20.0,
    env_name: str = "LIBERO_OBJECT",
    frame_size: tuple[int, int] = (128, 128),
    save_debug: bool = False,
) -> dict:
    """Run Stages 2 through 5 for a single demonstration directory."""
    demo_folder_name = demo_dir.name
    demo_h5_name = f"demo_{demo_idx}"

    logger.info("==================================================================")
    logger.info(">>> Processing [%d]: '%s' -> '%s'", demo_idx, demo_folder_name, demo_h5_name)
    logger.info("==================================================================")

    # Dedicated per-demo intermediate directories
    slam_dir = base_data_dir / "slam" / demo_folder_name
    hand_dir = base_data_dir / "hand_poses" / demo_folder_name
    retarget_dir = base_data_dir / "retargeted" / demo_folder_name

    start_t = time.time()

    # 1. Stage 2: Camera Tracking & Table Plane with ArUco Tag 0
    logger.info("[1/4] Stage 2: Tracking Camera & Table Plane (ArUco Tag %d)...", aruco_id)
    _, _, _, aruco_res = track_camera_with_aruco(
        frames_dir=demo_dir,
        calib_txt=calib_txt,
        output_dir=slam_dir,
        marker_id=aruco_id,
        marker_size=marker_size,
        dictionary_name=aruco_dict,
        save_annotated_debug_images=save_debug,
    )

    # 2. Stage 3: 3D Hand Tracking with MediaPipe Hands
    logger.info("[2/4] Stage 3: Tracking 3D Hand Keypoints & Pinch Gripper...")
    hand_npz = process_hand_tracking(
        frames_dir=demo_dir,
        output_dir=hand_dir,
        pinch_threshold=pinch_threshold,
        method="auto",
        save_debug=save_debug,
    )

    # 3. Stage 4: Geometric Kinematics Retargeting to Franka Panda Base
    logger.info("[3/4] Stage 4: Retargeting Kinematics to Robot Base & Computing OSC Actions...")
    retarget_npz = process_kinematics_retargeting(
        camera_poses=slam_dir / "camera_poses.npz",
        table_transform=slam_dir / "table_transform.npz",
        hand_poses=hand_npz,
        output_dir=retarget_dir,
        tgt_fps=tgt_fps,
    )

    # 4. Stage 5: Package and Append into unified LIBERO HDF5 Dataset
    logger.info("[4/4] Stage 5: Packaging & Appending to '%s' as '%s'...", output_hdf5.name, demo_h5_name)
    export_libero_dataset(
        retargeted_npz=retarget_npz,
        output_hdf5=output_hdf5,
        agentview_dir=demo_dir,
        demo_name=demo_h5_name,
        env_name=env_name,
        frame_height=frame_size[1],
        frame_width=frame_size[0],
    )

    elapsed = time.time() - start_t
    logger.info("Successfully finished '%s' in %.2fs!", demo_h5_name, elapsed)

    return {
        "demo_idx": demo_idx,
        "folder": demo_folder_name,
        "h5_name": demo_h5_name,
        "frames": aruco_res.total_frames,
        "tag_detected": f"{aruco_res.detection_ratio * 100.0:.1f}%",
        "elapsed_sec": f"{elapsed:.1f}s",
        "status": "SUCCESS",
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="run_batch_pipeline.py",
        description="One-Click Batch Automation: Convert multiple demo videos/frames into a unified LIBERO dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--video_dir",
        type=str,
        default=None,
        help="Optional: If specified, first extracts all videos in this folder to frames.",
    )
    parser.add_argument(
        "--frames_root",
        type=str,
        default="data/frames",
        help="Root folder containing demo frame subdirectories (e.g. data/frames/demo_0000).",
    )
    parser.add_argument(
        "--calib_txt",
        type=str,
        default="data/calibration/calib.txt",
        help="Path to camera intrinsics file from Stage 1.",
    )
    parser.add_argument(
        "--output_hdf5",
        type=str,
        default="data/datasets/demo_libero.hdf5",
        help="Path to unified output LIBERO benchmark HDF5 file.",
    )
    parser.add_argument(
        "--marker_size",
        type=float,
        default=0.05,
        help="Physical ArUco marker edge size in metres (e.g. 0.05 for 50 mm).",
    )
    parser.add_argument(
        "--aruco_id",
        type=int,
        default=0,
        help="ArUco marker ID.",
    )
    parser.add_argument(
        "--aruco_dict",
        type=str,
        default="DICT_ARUCO_ORIGINAL",
        help="ArUco dictionary (default: DICT_ARUCO_ORIGINAL).",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="LIBERO_OBJECT",
        help="Target LIBERO/Robosuite environment name.",
    )
    parser.add_argument(
        "--frame_size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=[128, 128],
        help="Target image observation resolution.",
    )
    parser.add_argument(
        "--target_fps",
        type=float,
        default=20.0,
        help="Robot control frequency (Hz).",
    )
    parser.add_argument(
        "--pinch_threshold",
        type=float,
        default=0.04,
        help="Thumb-to-index pinch distance (m) threshold.",
    )
    parser.add_argument(
        "--save_debug",
        action="store_true",
        help="Save visual verification images for tag and hand skeleton.",
    )
    parser.add_argument(
        "--reset_dataset",
        action="store_true",
        help="Delete existing output HDF5 file before starting batch run.",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    base_data_dir = project_root / "data"
    calib_txt = Path(args.calib_txt).resolve()
    output_hdf5 = Path(args.output_hdf5).resolve()
    frames_root = Path(args.frames_root).resolve()

    if not calib_txt.is_file():
        logger.error("Calibration file not found at '%s'. Run Stage 1 (01_calibrate.py) first.", calib_txt)
        return 1

    # Optional Stage 0: Extract raw videos if video_dir is given
    if args.video_dir is not None:
        v_dir = Path(args.video_dir).resolve()
        if v_dir.is_dir():
            logger.info("=== Stage 0: Batch extracting videos from '%s' ===", v_dir)
            batch_transform_videos_to_frames(
                video_sources=v_dir,
                output_root=frames_root,
                naming_scheme="index",
                prefix="demo_",
                target_fps=args.target_fps,
                target_size=tuple(args.frame_size),
                center_crop=True,
            )

    demo_dirs = get_demo_directories(frames_root)
    if not demo_dirs:
        logger.error("No demonstration subfolders with frames found in '%s'.", frames_root)
        return 1

    logger.info("Found %d demonstration(s) to process in '%s':", len(demo_dirs), frames_root)
    for d in demo_dirs:
        logger.info("  - %s", d.name)

    # Optional: Reset HDF5 if user asked to wipe previous runs
    if args.reset_dataset and output_hdf5.is_file():
        output_hdf5.unlink()
        logger.info("Removed existing dataset file: %s", output_hdf5)

    summary_records = []
    total_start_time = time.time()

    for idx, d_dir in enumerate(demo_dirs):
        try:
            record = run_pipeline_for_demo(
                demo_dir=d_dir,
                demo_idx=idx,
                calib_txt=calib_txt,
                output_hdf5=output_hdf5,
                base_data_dir=base_data_dir,
                marker_size=args.marker_size,
                aruco_id=args.aruco_id,
                aruco_dict=args.aruco_dict,
                pinch_threshold=args.pinch_threshold,
                tgt_fps=args.target_fps,
                env_name=args.env_name,
                frame_size=tuple(args.frame_size),
                save_debug=args.save_debug,
            )
            summary_records.append(record)
        except Exception as e:
            logger.error("Failed processing '%s': %s", d_dir.name, e, exc_info=True)
            summary_records.append({
                "demo_idx": idx,
                "folder": d_dir.name,
                "h5_name": f"demo_{idx}",
                "frames": 0,
                "tag_detected": "FAILED",
                "elapsed_sec": "0.0s",
                "status": f"ERROR: {e}",
            })

    total_time = time.time() - total_start_time

    # Print final summary table
    print("\n" + "=" * 76)
    print("                      BATCH PIPELINE SUMMARY")
    print("=" * 76)
    print(f"{'Idx':<4} {'Folder':<14} {'HDF5 Demo':<12} {'Frames':<8} {'ArUco Tag':<11} {'Status':<10} {'Time':<6}")
    print("-" * 76)
    for r in summary_records:
        print(f"{r['demo_idx']:<4} {r['folder']:<14} {r['h5_name']:<12} {r['frames']:<8} {r['tag_detected']:<11} {r['status']:<10} {r['elapsed_sec']:<6}")
    print("=" * 76)
    print(f"Total time: {total_time:.1f}s | Output Dataset: {output_hdf5}")
    print("=" * 76 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
