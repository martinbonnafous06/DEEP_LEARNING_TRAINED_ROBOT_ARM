"""src.utils.video_processor — Multi-video frame extraction and dataset structuring.

Provides tools to transform raw video recordings (single or multiple videos/episodes)
into standardized, structured frame sequences compatible with the LIBERO / Robosuite
retargeting pipeline.

Directory Structure Produced
---------------------------
<output_root>/
├── dataset_manifest.json      # Global registry of all extracted videos and stats
├── dataset_manifest.csv       # Tabular summary (name, path, frames, fps, duration)
├── <video_id_0>/
│   ├── metadata.json          # Video properties, timestamps, frame index mapping
│   ├── timestamps.npy         # (N,) float64 timestamps in seconds
│   ├── frame_000000.png       # Zero-padded frame images (natural-sort friendly)
│   ├── frame_000001.png
│   └── ...
└── <video_id_1>/
    ├── metadata.json
    ├── timestamps.npy
    ├── frame_000000.png
    └── ...
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image

logger = logging.getLogger("video_processor")

SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
    ".m4v",
    ".wmv",
    ".flv",
}


@dataclass
class VideoFrameExtractionResult:
    """Metadata for an individual processed video."""

    video_name: str
    source_path: str
    output_dir: str
    source_fps: float
    source_frame_count: int
    source_duration_seconds: float
    source_width: int
    source_height: int
    extracted_frame_count: int
    effective_fps: float
    target_width: int
    target_height: int
    frame_format: str
    timestamps_seconds: List[float] = field(default_factory=list)
    frame_indices: List[int] = field(default_factory=list)
    frame_filenames: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DatasetManifest:
    """Global dataset manifest summarizing multi-video frame extraction."""

    dataset_name: str
    output_root: str
    created_at: str
    total_videos: int
    total_frames: int
    target_fps: Optional[float]
    target_resolution: Optional[Tuple[int, int]]
    center_crop: bool
    frame_format: str
    videos: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def find_video_files(
    sources: Union[str, Path, Sequence[Union[str, Path]]],
    recursive: bool = True,
) -> List[Path]:
    """Discover all supported video files from a path or sequence of paths.

    Args:
        sources: Path to a file/directory or list/tuple of paths.
        recursive: Whether to search subdirectories recursively.

    Returns:
        Sorted list of unique Path objects pointing to valid video files.
    """
    found: List[Path] = []
    if isinstance(sources, (str, Path)):
        sources = [sources]

    for src in sources:
        p = Path(src).expanduser().resolve()
        if p.is_file():
            if p.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
                found.append(p)
            else:
                logger.warning("Skipping unsupported video file extension: %s", p)
        elif p.is_dir():
            pattern = "**/*" if recursive else "*"
            for item in p.glob(pattern):
                if item.is_file() and item.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
                    found.append(item.resolve())
        else:
            # Check for glob pattern in string
            str_path = str(src)
            parent = Path(str_path).parent
            pattern = Path(str_path).name
            if parent.exists() and any(ch in pattern for ch in ["*", "?", "["]):
                for item in parent.glob(pattern):
                    if item.is_file() and item.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
                        found.append(item.resolve())

    # Sort naturally and remove duplicates while preserving order
    seen = set()
    unique_paths: List[Path] = []
    for item in sorted(found, key=lambda x: x.name.lower()):
        if item not in seen:
            seen.add(item)
            unique_paths.append(item)

    return unique_paths


def compute_sample_frame_indices(
    total_frames: int,
    source_fps: float,
    target_fps: Optional[float] = None,
    max_frames: Optional[int] = None,
) -> List[int]:
    """Compute which source frame indices to extract given target FPS and max_frames.

    Args:
        total_frames: Total number of frames in the source video.
        source_fps: Video frame rate (frames per second).
        target_fps: Desired sampling rate. If None or <= 0, preserves all frames.
        max_frames: Optional ceiling on extracted frame count.

    Returns:
        List of integer frame indices [0, 1, ...].
    """
    if total_frames <= 0:
        return []

    if target_fps is None or target_fps <= 0 or target_fps >= source_fps:
        indices = list(range(total_frames))
    else:
        # Time-based uniform subsampling
        duration = total_frames / source_fps
        num_target_frames = max(1, int(round(duration * target_fps)))
        # Sample evenly across time
        times = np.linspace(0, duration, num_target_frames, endpoint=False)
        indices = [min(int(round(t * source_fps)), total_frames - 1) for t in times]
        # Remove consecutive duplicate indices while preserving order
        deduped: List[int] = []
        for idx in indices:
            if not deduped or deduped[-1] != idx:
                deduped.append(idx)
        indices = deduped

    if max_frames is not None and max_frames > 0:
        indices = indices[:max_frames]

    return indices


def process_frame(
    frame_bgr: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None,
    center_crop: bool = False,
) -> np.ndarray:
    """Preprocess frame (optional center square crop and resize).

    Args:
        frame_bgr: Raw BGR image as NumPy array from OpenCV.
        target_size: Tuple (width, height) to resize to.
        center_crop: If True, crops the center square before resizing.

    Returns:
        Processed BGR image array.
    """
    img = frame_bgr
    h, w = img.shape[:2]

    if center_crop and h != w:
        min_dim = min(h, w)
        start_y = (h - min_dim) // 2
        start_x = (w - min_dim) // 2
        img = img[start_y : start_y + min_dim, start_x : start_x + min_dim]

    if target_size is not None:
        target_w, target_h = target_size
        if (img.shape[1], img.shape[0]) != (target_w, target_h):
            img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

    return img


def extract_frames_from_video(
    video_path: Union[str, Path],
    output_dir: Union[str, Path],
    target_fps: Optional[float] = None,
    target_size: Optional[Tuple[int, int]] = None,
    center_crop: bool = False,
    frame_format: str = "png",
    quality: int = 95,
    max_frames: Optional[int] = None,
    overwrite: bool = True,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> VideoFrameExtractionResult:
    """Extract frames from a single video into a dedicated directory with metadata.

    Args:
        video_path: Path to the input video.
        output_dir: Destination folder where frame images and metadata are saved.
        target_fps: Target frame rate (e.g. 20 Hz for LIBERO). If None, keeps native FPS.
        target_size: Optional (width, height) tuple to resize frames to.
        center_crop: Crop center square before resizing.
        frame_format: Image format: 'png' (lossless) or 'jpg'/'jpeg'.
        quality: JPEG compression quality (1-100) or PNG compression level (0-9).
        max_frames: Optional upper bound on extracted frames.
        overwrite: Overwrite existing frames in output_dir.
        progress_callback: Optional callback func(current_frame, total_frames).

    Returns:
        VideoFrameExtractionResult dataclass containing full metadata.
    """
    video_path = Path(video_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")

    try:
        source_fps = float(cap.get(cv2.CAP_PROP_FPS))
        if source_fps <= 0 or np.isnan(source_fps):
            source_fps = 30.0  # sensible default fallback

        source_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = source_total_frames / source_fps if source_total_frames > 0 else 0.0

        sample_indices = set(
            compute_sample_frame_indices(
                total_frames=source_total_frames,
                source_fps=source_fps,
                target_fps=target_fps,
                max_frames=max_frames,
            )
        )

        effective_target_fps = target_fps if (target_fps and target_fps < source_fps) else source_fps
        frame_ext = frame_format.lower().lstrip(".")
        if frame_ext in ("jpeg", "jpg"):
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
            ext = ".jpg"
        else:
            # PNG compression: 0 (fastest, largest) to 9 (slowest, smallest)
            png_level = min(9, max(0, int((100 - quality) / 10))) if quality < 100 else 3
            encode_params = [int(cv2.IMWRITE_PNG_COMPRESSION), png_level]
            ext = ".png"

        extracted_indices: List[int] = []
        timestamps: List[float] = []
        filenames: List[str] = []

        curr_src_idx = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if curr_src_idx in sample_indices:
                t_sec = float(curr_src_idx / source_fps)
                processed = process_frame(
                    frame_bgr=frame,
                    target_size=target_size,
                    center_crop=center_crop,
                )
                filename = f"frame_{saved_count:06d}{ext}"
                frame_out_path = output_dir / filename

                if overwrite or not frame_out_path.exists():
                    cv2.imwrite(str(frame_out_path), processed, encode_params)

                extracted_indices.append(curr_src_idx)
                timestamps.append(t_sec)
                filenames.append(filename)
                saved_count += 1

                if progress_callback is not None:
                    progress_callback(saved_count, len(sample_indices))

                if max_frames is not None and saved_count >= max_frames:
                    break

            curr_src_idx += 1

    finally:
        cap.release()

    final_w = target_size[0] if target_size else source_width
    final_h = target_size[1] if target_size else source_height

    result = VideoFrameExtractionResult(
        video_name=video_path.stem,
        source_path=str(video_path),
        output_dir=str(output_dir),
        source_fps=source_fps,
        source_frame_count=source_total_frames,
        source_duration_seconds=duration,
        source_width=source_width,
        source_height=source_height,
        extracted_frame_count=saved_count,
        effective_fps=effective_target_fps,
        target_width=final_w,
        target_height=final_h,
        frame_format=frame_ext,
        timestamps_seconds=timestamps,
        frame_indices=extracted_indices,
        frame_filenames=filenames,
    )

    # Save video-level metadata JSON and timestamps NPY
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)

    ts_path = output_dir / "timestamps.npy"
    np.save(str(ts_path), np.array(timestamps, dtype=np.float64))

    return result


def batch_transform_videos_to_frames(
    video_sources: Union[str, Path, Sequence[Union[str, Path]]],
    output_root: Union[str, Path],
    naming_scheme: str = "stem",
    prefix: str = "demo_",
    start_index: int = 0,
    target_fps: Optional[float] = None,
    target_size: Optional[Tuple[int, int]] = None,
    center_crop: bool = False,
    frame_format: str = "png",
    quality: int = 95,
    max_frames_per_video: Optional[int] = None,
    overwrite: bool = True,
    dataset_name: str = "libero_video_dataset",
) -> DatasetManifest:
    """Batch transform multiple video files into a structured frame dataset.

    Naming Schemes:
    - 'stem': Uses the original video filename stem (e.g. video_take1 -> <output_root>/video_take1/)
    - 'index': Zero-padded sequential naming (e.g. demo_000, demo_001, ...)
    - 'hierarchical': If source videos are organized in subdirectories, mirrors that structure

    Args:
        video_sources: Video file(s), folder of videos, or glob pattern.
        output_root: Root directory where each video's frames will be stored.
        naming_scheme: 'stem', 'index', or 'hierarchical'.
        prefix: Prefix used when naming_scheme is 'index' (e.g. 'demo_').
        start_index: Starting integer for 'index' naming scheme.
        target_fps: Target frame extraction FPS (e.g. 20 Hz for LIBERO). None for native.
        target_size: Target (width, height) resolution tuple. None for original resolution.
        center_crop: Whether to apply aspect-ratio-preserving square center crop.
        frame_format: Image file format ('png' or 'jpg').
        quality: Image quality parameter (1-100).
        max_frames_per_video: Max frames to extract per video.
        overwrite: Overwrite existing extracted frames.
        dataset_name: Name of the dataset for the manifest file.

    Returns:
        DatasetManifest object summarizing all processed videos.
    """
    video_paths = find_video_files(video_sources)
    if not video_paths:
        raise ValueError(f"No valid video files found matching source: {video_sources}")

    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    results: List[VideoFrameExtractionResult] = []
    total_extracted_frames = 0

    logger.info("Starting batch extraction for %d video(s)...", len(video_paths))

    for idx, v_path in enumerate(video_paths):
        if naming_scheme == "index":
            folder_name = f"{prefix}{start_index + idx:04d}"
        else:  # default to stem
            folder_name = v_path.stem

        video_out_dir = output_root / folder_name
        logger.info(
            "[%d/%d] Processing '%s' -> '%s' (target_fps=%s, size=%s)",
            idx + 1,
            len(video_paths),
            v_path.name,
            video_out_dir.name,
            target_fps,
            target_size,
        )

        res = extract_frames_from_video(
            video_path=v_path,
            output_dir=video_out_dir,
            target_fps=target_fps,
            target_size=target_size,
            center_crop=center_crop,
            frame_format=frame_format,
            quality=quality,
            max_frames=max_frames_per_video,
            overwrite=overwrite,
        )
        results.append(res)
        total_extracted_frames += res.extracted_frame_count

    manifest = DatasetManifest(
        dataset_name=dataset_name,
        output_root=str(output_root),
        created_at=datetime.utcnow().isoformat() + "Z",
        total_videos=len(results),
        total_frames=total_extracted_frames,
        target_fps=target_fps,
        target_resolution=target_size,
        center_crop=center_crop,
        frame_format=frame_format,
        videos=[r.to_dict() for r in results],
    )

    # Save manifest.json at output root
    manifest_json_path = output_root / "dataset_manifest.json"
    with open(manifest_json_path, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2)

    # Save manifest.csv for quick spreadsheet / CLI viewing
    manifest_csv_path = output_root / "dataset_manifest.csv"
    with open(manifest_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index",
            "video_name",
            "folder_name",
            "extracted_frames",
            "effective_fps",
            "resolution",
            "duration_sec",
            "source_path",
        ])
        for i, r in enumerate(results):
            folder_name = Path(r.output_dir).name
            writer.writerow([
                i,
                r.video_name,
                folder_name,
                r.extracted_frame_count,
                f"{r.effective_fps:.2f}",
                f"{r.target_width}x{r.target_height}",
                f"{r.source_duration_seconds:.2f}",
                r.source_path,
            ])

    logger.info(
        "Batch extraction completed: %d video(s), %d total frames extracted to '%s'.",
        len(results),
        total_extracted_frames,
        output_root,
    )
    return manifest
