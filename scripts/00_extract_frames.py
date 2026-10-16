"""scripts/00_extract_frames.py — CLI entry point for Multi-Video Frame Extraction.

Transforms raw video recordings (individual files, directories of videos, or glob patterns)
into structured frame sequences organized per video/demo, with dataset manifest and timestamps.

Usage Examples:
    # 1. Process all videos in a folder into individual demo folders at 20 FPS (LIBERO control rate)
    python scripts/00_extract_frames.py \\
        --video_dir data/raw_videos \\
        --output_dir data/frames \\
        --target_fps 20.0 \\
        --naming_scheme index \\
        --prefix demo_

    # 2. Process specific videos with custom 128x128 resize (matching LIBERO environment)
    python scripts/00_extract_frames.py \\
        --video_files data/raw_videos/take1.mp4 data/raw_videos/take2.mp4 \\
        --output_dir data/frames \\
        --resize 128 128 \\
        --format png

    # 3. High-speed extraction as JPEG with original resolution
    python scripts/00_extract_frames.py \\
        --video_dir data/raw_videos \\
        --output_dir data/frames \\
        --format jpg \\
        --quality 95
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.video_processor import batch_transform_videos_to_frames

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("00_extract_frames")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="00_extract_frames.py",
        description="Stage 0: Standardized Multi-Video to Structured Frame Extraction for LIBERO.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--video_dir",
        type=str,
        help="Directory containing video recordings (scanned recursively for .mp4, .mov, etc.).",
    )
    source_group.add_argument(
        "--video_files",
        type=str,
        nargs="+",
        help="One or more specific video file paths or glob patterns.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/frames",
        help="Root output folder where subdirectories for each video and manifest will be created.",
    )
    parser.add_argument(
        "--naming_scheme",
        type=str,
        choices=["stem", "index"],
        default="index",
        help="'index' assigns sequential folder names (e.g. demo_0000, demo_0001); 'stem' keeps original filename.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="demo_",
        help="Folder prefix used when naming_scheme='index' (e.g. 'demo_' -> demo_0000, demo_0001).",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Starting index number for sequential folder naming.",
    )
    parser.add_argument(
        "--target_fps",
        type=float,
        default=None,
        help="Target sampling rate in Hz (e.g. 20.0 to match LIBERO policy frequency). Defaults to native FPS.",
    )
    parser.add_argument(
        "--resize",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=None,
        help="Optional target resolution (e.g. --resize 128 128 for LIBERO bddl rendering, or 84 84).",
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help="Apply aspect-ratio preserving square center-crop before resizing.",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["png", "jpg", "jpeg"],
        default="png",
        help="Output image format ('png' for lossless, 'jpg' for compact).",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="Encoding quality (1-100 for JPEG, or compression level equivalent for PNG).",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Ceiling on number of frames to extract per video (useful for fast testing).",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="libero_retargeting_frames",
        help="Dataset name written into the global manifest JSON.",
    )
    parser.add_argument(
        "--no_overwrite",
        action="store_true",
        help="Do not overwrite existing frames if already present in destination directories.",
    )

    args = parser.parse_args()

    sources = args.video_dir if args.video_dir else args.video_files
    target_size = tuple(args.resize) if args.resize else None

    logger.info("=== Stage 0: Multi-Video Frame Extraction ===")
    logger.info("Input source: %s", sources)
    logger.info("Output root: %s", args.output_dir)
    logger.info("Target FPS: %s | Target Size: %s", args.target_fps, target_size)

    try:
        manifest = batch_transform_videos_to_frames(
            video_sources=sources,
            output_root=args.output_dir,
            naming_scheme=args.naming_scheme,
            prefix=args.prefix,
            start_index=args.start_index,
            target_fps=args.target_fps,
            target_size=target_size,
            center_crop=args.center_crop,
            frame_format=args.format,
            quality=args.quality,
            max_frames_per_video=args.max_frames,
            overwrite=not args.no_overwrite,
            dataset_name=args.dataset_name,
        )

        logger.info(
            "Success! Extracted %d frames across %d videos into '%s'. Manifest saved.",
            manifest.total_frames,
            manifest.total_videos,
            args.output_dir,
        )
        return 0

    except Exception as exc:
        logger.error("Error during frame extraction: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
