"""scripts/05_export_libero.py — CLI entry point for Stage 5: Retargeting & LIBERO Dataset Export.

Usage:
    python scripts/05_export_libero.py \
        --retargeted_npz data/retargeted/retargeted.npz \
        --output_hdf5 data/datasets/demo_libero.hdf5 \
        --agentview_dir data/frames/agentview \
        --wrist_dir data/frames/wrist
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retargeting.libero_exporter import export_libero_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("05_export_libero")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="05_export_libero.py",
        description="Stage 5: Packaging Retargeted Trajectories into LIBERO Benchmark HDF5 Datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--retargeted_npz", type=str, required=True, help="Path to retargeted.npz from Stage 4.")
    parser.add_argument("--output_hdf5", type=str, required=True, help="Path to write the output .hdf5 dataset.")
    parser.add_argument("--agentview_dir", type=str, default=None, help="Directory containing agentview RGB images.")
    parser.add_argument("--wrist_dir", type=str, default=None, help="Directory containing wrist camera RGB images.")
    parser.add_argument("--demo_name", type=str, default="demo_0", help="Demo group name in HDF5 file.")
    parser.add_argument("--env_name", type=str, default="LIBERO_OBJECT", help="Target LIBERO/Robosuite environment name.")
    parser.add_argument("--frame_height", type=int, default=84, help="Image height in pixels.")
    parser.add_argument("--frame_width", type=int, default=84, help="Image width in pixels.")
    args = parser.parse_args()

    logger.info("=== Stage 5: LIBERO HDF5 Dataset Export ===")
    export_libero_dataset(
        retargeted_npz=args.retargeted_npz,
        output_hdf5=args.output_hdf5,
        agentview_dir=args.agentview_dir,
        wrist_dir=args.wrist_dir,
        demo_name=args.demo_name,
        env_name=args.env_name,
        frame_height=args.frame_height,
        frame_width=args.frame_width,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
