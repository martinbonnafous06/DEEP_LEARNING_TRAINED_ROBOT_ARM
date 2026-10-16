"""src.utils.io — File I/O, path management, and dataset serialization utilities."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import numpy.typing as npt
from PIL import Image

from .coordinates import twist_to_matrix


def natural_sort_key(item: Path | str) -> list[int | str]:
    """Generate a sort key that orders strings naturally by numeric value.

    Ensures 'frame_2.png' precedes 'frame_10.png' regardless of leading zeroes.

    Args:
        item: Path or string to evaluate.

    Returns:
        List of integer and lowercase string tokens.
    """
    s = item.name if isinstance(item, Path) else str(item)
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]


def get_sorted_image_files(directory: Path | str) -> list[Path]:
    """Retrieve all supported image files in a directory sorted in natural order.

    Args:
        directory: Directory path to scan.

    Returns:
        List of Path objects in natural sorted order.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []

    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
    files = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in valid_exts]
    return sorted(files, key=natural_sort_key)


def load_camera_poses(path: Path | str) -> npt.NDArray[np.float64]:
    """Load camera poses from either .npy or .npz format.

    Supports:
    - .npy containing (N, 4, 4) or (N, 7) array
    - .npz containing 'T_world_to_cam', 'poses', or 'traj_est'

    Args:
        path: Path to pose file.

    Returns:
        Array of shape (N, 4, 4) float64 camera poses (T_world_to_cam).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Camera poses file not found: {path}")

    if path.suffix == ".npz":
        data = np.load(path)
        for key in ("T_world_to_cam", "poses", "traj_est"):
            if key in data:
                raw = data[key].astype(np.float64)
                if raw.ndim == 3 and raw.shape[1:] == (4, 4):
                    return raw
                elif raw.ndim == 2 and raw.shape[1] == 7:
                    return twist_to_matrix(raw)
        raise KeyError(
            f"No recognizable pose array in {path}. Available keys: {list(data.keys())}"
        )
    else:
        raw = np.load(path).astype(np.float64)
        if raw.ndim == 3 and raw.shape[1:] == (4, 4):
            return raw
        elif raw.ndim == 2 and raw.shape[1] == 7:
            return twist_to_matrix(raw)
        raise ValueError(f"Unexpected pose shape {raw.shape} in {path}. Expected (N, 4, 4) or (N, 7).")


def save_camera_poses(output_dir: Path | str, poses: npt.NDArray[np.float64]) -> tuple[Path, Path]:
    """Save camera poses in both .npy and .npz formats for full pipeline compatibility.

    Writes:
    - ``output_dir/poses.npy``: Raw (N, 4, 4) array.
    - ``output_dir/camera_poses.npz``: NPZ archive with keys 'T_world_to_cam' and 'poses'.

    Args:
        output_dir: Directory where files should be written.
        poses: Array of shape (N, 4, 4) float64.

    Returns:
        Tuple of (npy_path, npz_path).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npy_path = out_dir / "poses.npy"
    npz_path = out_dir / "camera_poses.npz"

    np.save(npy_path, poses.astype(np.float64))
    np.savez_compressed(npz_path, T_world_to_cam=poses.astype(np.float64), poses=poses.astype(np.float64))

    return npy_path, npz_path


def load_calibration_txt(calib_txt: Path | str) -> tuple[float, float, float, float]:
    """Parse DROID-SLAM format calibration file (single line: fx fy cx cy).

    Args:
        calib_txt: Path to text file.

    Returns:
        Tuple of (fx, fy, cx, cy) as floats.
    """
    calib_path = Path(calib_txt)
    if not calib_path.is_file():
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")

    for line in calib_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 4:
            return float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])

    raise ValueError(f"No valid calibration line found in {calib_path}")


def export_calibration_txt(
    camera_matrix: npt.NDArray[np.float64],
    output_path: Path | str,
) -> Path:
    """Export camera intrinsic parameters to calib.txt in DROID-SLAM format.

    Args:
        camera_matrix: 3x3 OpenCV camera matrix.
        output_path: Destination file path.

    Returns:
        Path to exported file.
    """
    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        f.write(f"{fx:.8f} {fy:.8f} {cx:.8f} {cy:.8f}\n")

    return out_p


def load_rgb_frames(
    directory: Optional[Path | str],
    n_frames: int,
    height: int = 84,
    width: int = 84,
) -> npt.NDArray[np.uint8]:
    """Load image frames from directory, resize to (height, width), and align to n_frames.

    If directory is None or empty, returns zero frames.
    If image count differs from n_frames, truncates or pads with the last frame.

    Args:
        directory: Directory containing RGB frames.
        n_frames: Target number of frames.
        height: Image height in pixels.
        width: Image width in pixels.

    Returns:
        Array of shape (n_frames, height, width, 3) uint8.
    """
    blank = np.zeros((n_frames, height, width, 3), dtype=np.uint8)
    if directory is None:
        return blank

    dir_p = Path(directory)
    if not dir_p.is_dir():
        return blank

    paths = get_sorted_image_files(dir_p)
    if not paths:
        return blank

    frames: list[npt.NDArray[np.uint8]] = []
    for p in paths[:n_frames]:
        try:
            with Image.open(p) as img:
                resized = img.convert("RGB").resize((width, height), Image.BILINEAR)
                frames.append(np.array(resized, dtype=np.uint8))
        except Exception:
            continue

    if not frames:
        return blank

    # Pad with last frame if needed
    while len(frames) < n_frames:
        frames.append(frames[-1].copy())

    return np.stack(frames[:n_frames], axis=0)
