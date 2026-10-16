"""hand_tracking.py — Module 3: HaMeR-based hand tracking for Video-to-LIBERO pipeline.

This module processes an RGB image sequence directory through the HaMeR hand
reconstruction model (invoked via subprocess), extracts per-frame wrist SE(3)
transforms and MANO 3-D keypoints, computes gripper aperture, enforces metric
scale, and optionally visualises the result with the Rerun SDK.

Typical usage::

    python hand_tracking.py \\
        --frames_dir ./data/frames \\
        --output_dir ./data/hand_output \\
        --hamer_script ./third_party/hamer/demo.py \\
        --pinch_threshold 0.04 \\
        --visualize
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("hand_tracking")

# ---------------------------------------------------------------------------
# MANO keypoint indices (21-joint MANO topology)
# ---------------------------------------------------------------------------
THUMB_TIP_IDX: int = 4
INDEX_TIP_IDX: int = 8
WRIST_IDX: int = 0
MIDDLE_MCP_IDX: int = 9  # Middle finger MCP joint

# Reference wrist-to-middle-MCP distance (metres) used for metric scaling.
L_REF_METRES: float = 0.085

# MANO 21-joint skeleton connectivity for visualisation.
HAND_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index
    (0, 9), (9, 10), (10, 11), (11, 12),   # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
]


# ---------------------------------------------------------------------------
# HaMeR subprocess helpers
# ---------------------------------------------------------------------------

def run_hamer(
    hamer_script: Path,
    frames_dir: Path,
    output_dir: Path,
) -> Path:
    """Invoke HaMeR demo.py via subprocess and return the path of the output .npz.

    HaMeR's demo.py is expected to write a file named ``hamer_output.npz`` (or
    equivalent) inside *output_dir*.  The subprocess is called with the image
    directory as ``--img_folder`` and the output directory as ``--out_folder``.
    Any stdout/stderr from HaMeR is forwarded to the module logger so the
    caller can monitor progress without additional plumbing.

    Args:
        hamer_script: Absolute or relative path to HaMeR's ``demo.py``.
        frames_dir: Directory containing the input RGB frames (``*.png`` /
            ``*.jpg``).
        output_dir: Directory where HaMeR should write its outputs.

    Returns:
        Path to the produced ``hamer_output.npz`` file.

    Raises:
        FileNotFoundError: If *hamer_script* does not exist.
        subprocess.CalledProcessError: If HaMeR exits with a non-zero status.
        FileNotFoundError: If the expected output ``.npz`` is absent after the
            subprocess completes.
    """
    if not hamer_script.exists():
        raise FileNotFoundError(
            f"HaMeR script not found: {hamer_script}. "
            "Ensure the repo is cloned at ./third_party/hamer."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = [
        sys.executable,
        str(hamer_script),
        "--img_folder", str(frames_dir),
        "--out_folder", str(output_dir),
        "--batch_size", "16",
        "--full_frame",
        "--save_mesh",
    ]

    logger.info("Launching HaMeR: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        cwd=str(hamer_script.parent),
        capture_output=False,
        text=True,
        check=True,
    )
    logger.info("HaMeR subprocess finished (return code %d).", result.returncode)

    candidates: list[Path] = [
        output_dir / "hamer_output.npz",
        output_dir / "results.npz",
        output_dir / "output.npz",
        *[
            p for p in sorted(output_dir.glob("*.npz"))
            if p.name not in {"hand_poses.npz", "table_transform.npz", "camera_poses.npz", "retargeted.npz"}
        ],
    ]
    for candidate in candidates:
        if candidate.exists():
            logger.info("Found HaMeR output: %s", candidate)
            return candidate

    raise FileNotFoundError(
        f"No .npz output found in {output_dir} after running HaMeR. "
        "Check HaMeR's --out_folder handling."
    )


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def extract_wrist_transforms(
    npz_data: dict[str, npt.NDArray],
) -> npt.NDArray[np.float64]:
    """Extract per-frame wrist SE(3) transforms from HaMeR output.

    HaMeR stores the wrist-in-camera transform under different key names
    depending on version.  This function tries several plausible keys in order
    of preference and returns the first match.

    Args:
        npz_data: Dictionary loaded from HaMeR's output ``.npz`` file.

    Returns:
        Array of shape ``(N, 4, 4)`` containing the wrist-in-camera homogeneous
        transform for each of the *N* detected frames.

    Raises:
        KeyError: If none of the expected transform keys are present in
            *npz_data*.
    """
    # Best case: full 4x4 matrix already stored.
    for key in ("T_cam_to_hand", "wrist_transform", "cam_to_hand"):
        if key in npz_data:
            arr = npz_data[key].astype(np.float64)
            if arr.ndim == 3 and arr.shape[-2:] == (4, 4):
                logger.info("Loaded wrist transforms from key '%s' (%d frames).", key, len(arr))
                return arr

    # Fallback: reconstruct from rotation + translation.
    rot_keys = [k for k in ("global_orient", "smplx_global_orient", "pred_rot") if k in npz_data]
    trans_keys = [k for k in ("pred_cam_t", "cam_t", "translation") if k in npz_data]

    if rot_keys and trans_keys:
        rot_raw = npz_data[rot_keys[0]].astype(np.float64)
        trans = npz_data[trans_keys[0]].astype(np.float64)
        n_frames = len(trans)
        transforms = np.tile(np.eye(4, dtype=np.float64), (n_frames, 1, 1))
        if rot_raw.ndim == 2 and rot_raw.shape[1] == 3:
            for i, aa in enumerate(rot_raw):
                transforms[i, :3, :3] = _axis_angle_to_rotmat(aa)
        elif rot_raw.ndim == 3 and rot_raw.shape[1:] == (3, 3):
            transforms[:, :3, :3] = rot_raw
        transforms[:, :3, 3] = trans
        logger.info(
            "Reconstructed wrist transforms from '%s' + '%s' (%d frames).",
            rot_keys[0], trans_keys[0], n_frames,
        )
        return transforms

    raise KeyError(
        f"Cannot find wrist transform data in HaMeR output. "
        f"Available keys: {list(npz_data.keys())}."
    )


def _axis_angle_to_rotmat(aa: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Convert a single axis-angle vector to a 3x3 rotation matrix.

    Uses Rodrigues' rotation formula.

    Args:
        aa: Axis-angle vector of shape ``(3,)``.

    Returns:
        Rotation matrix of shape ``(3, 3)``.
    """
    angle = float(np.linalg.norm(aa))
    if angle < 1e-8:
        return np.eye(3, dtype=np.float64)
    axis = aa / angle
    K = np.array([
        [0.0,      -axis[2],  axis[1]],
        [axis[2],   0.0,     -axis[0]],
        [-axis[1],  axis[0],  0.0],
    ], dtype=np.float64)
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def extract_keypoints_3d(
    npz_data: dict[str, npt.NDArray],
) -> npt.NDArray[np.float64]:
    """Extract 3-D MANO keypoints from HaMeR output.

    Tries several plausible key names used across HaMeR versions.

    Args:
        npz_data: Dictionary loaded from HaMeR's output ``.npz`` file.

    Returns:
        Array of shape ``(N, 21, 3)`` -- per-frame 3-D positions of the 21 MANO
        hand joints, in the camera coordinate frame.

    Raises:
        KeyError: If no keypoint array can be found in *npz_data*.
    """
    keypoint_keys: list[str] = [
        "keypoints_3d",
        "pred_keypoints_3d",
        "joints_3d",
        "hand_joints",
        "j3d",
    ]
    for key in keypoint_keys:
        if key in npz_data:
            arr = npz_data[key].astype(np.float64)
            if arr.ndim == 3 and arr.shape[1] == 21 and arr.shape[2] == 3:
                logger.info("Loaded 3-D keypoints from key '%s' (%d frames).", key, len(arr))
                return arr

    raise KeyError(
        f"Cannot find 3-D keypoints in HaMeR output. "
        f"Available keys: {list(npz_data.keys())}. "
        f"Tried: {keypoint_keys}"
    )


# ---------------------------------------------------------------------------
# Metric scale enforcement
# ---------------------------------------------------------------------------

def enforce_metric_scale(
    keypoints_3d: npt.NDArray[np.float64],
    transforms: npt.NDArray[np.float64],
    l_ref: float = L_REF_METRES,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Rescale keypoints and wrist translations to enforce a known metric scale.

    The scale factor is computed so that the *median* wrist-to-middle-MCP
    distance across the trajectory equals *l_ref* (default 0.085 m).  All
    keypoint positions and the translation column of the wrist transforms are
    multiplied by this factor.

    Args:
        keypoints_3d: Raw keypoints of shape ``(N, 21, 3)`` in camera frame.
        transforms: Wrist SE(3) transforms of shape ``(N, 4, 4)``.
        l_ref: Target wrist-to-middle-MCP distance in metres. Defaults to
            ``0.085``.

    Returns:
        A tuple ``(scaled_keypoints, scaled_transforms)`` with the same shapes
        as the inputs but with all spatial quantities rescaled to metric units.

    Raises:
        ValueError: If the median wrist-to-middle-MCP distance is zero,
            making scale estimation impossible.
    """
    wrist_pos = keypoints_3d[:, WRIST_IDX, :]
    middle_mcp_pos = keypoints_3d[:, MIDDLE_MCP_IDX, :]
    dists = np.linalg.norm(middle_mcp_pos - wrist_pos, axis=-1)
    median_dist = float(np.median(dists))

    if median_dist < 1e-9:
        raise ValueError(
            f"Median wrist-to-middle-MCP distance is effectively zero ({median_dist:.6f}). "
            "Cannot compute metric scale factor."
        )

    scale = l_ref / median_dist
    logger.info(
        "Metric scale: median wrist-MCP dist = %.4f -> scale factor = %.4f (target %.4f m).",
        median_dist, scale, l_ref,
    )

    scaled_keypoints = keypoints_3d * scale

    scaled_transforms = transforms.copy()
    scaled_transforms[:, :3, 3] *= scale

    return scaled_keypoints, scaled_transforms


# ---------------------------------------------------------------------------
# Gripper aperture
# ---------------------------------------------------------------------------

def compute_gripper_aperture(
    keypoints_3d: npt.NDArray[np.float64],
    pinch_threshold: float = 0.04,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute normalised gripper aperture and binary open/close signal.

    Gripper aperture is defined as the Euclidean distance between the thumb tip
    (MANO joint 4) and the index tip (MANO joint 8) in 3-D camera space.  The
    raw distance is normalised to ``[0, 1]`` using the observed minimum and
    maximum across the whole trajectory.  A binary signal is derived by
    thresholding the raw (pre-normalisation) distance.

    Args:
        keypoints_3d: Keypoints of shape ``(N, 21, 3)`` -- already scaled to
            metric units.
        pinch_threshold: Raw distance (metres) below which the gripper is
            considered closed.  Defaults to ``0.04`` m.

    Returns:
        A tuple ``(gripper_continuous, gripper_binary)`` where:

        * ``gripper_continuous`` -- shape ``(N,)``, float in ``[0, 1]`` where
          0 = fully closed, 1 = fully open.
        * ``gripper_binary`` -- shape ``(N,)``, float containing ``1.0`` (closed)
          or ``-1.0`` (open), matching LIBERO's gripper action convention.
    """
    thumb_tip = keypoints_3d[:, THUMB_TIP_IDX, :]
    index_tip = keypoints_3d[:, INDEX_TIP_IDX, :]
    d_pinch = np.linalg.norm(thumb_tip - index_tip, axis=-1)

    d_min = float(d_pinch.min())
    d_max = float(d_pinch.max())

    if abs(d_max - d_min) < 1e-9:
        logger.warning(
            "Pinch distance has near-zero range (min=%.4f, max=%.4f). "
            "Continuous gripper will be all zeros.",
            d_min, d_max,
        )
        gripper_continuous = np.zeros_like(d_pinch)
    else:
        gripper_continuous = (d_pinch - d_min) / (d_max - d_min)

    # LIBERO convention: 1 = gripper closed, -1 = gripper open.
    gripper_binary = np.where(d_pinch < pinch_threshold, 1.0, -1.0)

    n_closed = int((gripper_binary > 0).sum())
    logger.info(
        "Gripper: pinch_threshold=%.3f m | closed=%d/%d frames | "
        "d_pinch range [%.4f, %.4f] m.",
        pinch_threshold, n_closed, len(d_pinch), d_min, d_max,
    )
    return gripper_continuous, gripper_binary


# ---------------------------------------------------------------------------
# Rerun visualisation
# ---------------------------------------------------------------------------

def visualize_hand_skeleton(
    keypoints_3d: npt.NDArray[np.float64],
    transforms: npt.NDArray[np.float64],
    gripper_continuous: npt.NDArray[np.float64],
) -> None:
    """Visualise the 3-D hand skeleton over time using the Rerun SDK.

    For each frame the function logs:

    * ``hand/joints`` -- 3-D point cloud of the 21 MANO keypoints, coloured by
      gripper aperture (red = closed, green = open).
    * ``hand/skeleton`` -- line segments connecting adjacent joints according
      to the MANO kinematic tree.
    * ``hand/wrist_transform`` -- the wrist coordinate frame (3 axes).

    Args:
        keypoints_3d: Scaled keypoints ``(N, 21, 3)`` in camera frame.
        transforms: Wrist SE(3) transforms ``(N, 4, 4)``.
        gripper_continuous: Normalised aperture ``(N,)`` in ``[0, 1]``.

    Raises:
        ImportError: If the ``rerun-sdk`` package is not installed.
    """
    try:
        import rerun as rr  # type: ignore[import]
        import rerun.blueprint as rrb  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "Rerun SDK is required for visualisation. "
            "Install it with: pip install rerun-sdk"
        ) from exc

    rr.init("hand_tracking", spawn=True)

    blueprint = rrb.Spatial3DView(origin="/hand")
    rr.send_blueprint(blueprint)

    n_frames = len(keypoints_3d)
    logger.info("Sending %d frames to Rerun viewer...", n_frames)

    for i in range(n_frames):
        rr.set_time_sequence("frame", i)

        kps = keypoints_3d[i]  # (21, 3)
        aperture = float(gripper_continuous[i])

        # Colour: lerp from red (closed, aperture=0) to green (open, aperture=1).
        r = int(255 * (1.0 - aperture))
        g = int(255 * aperture)
        joint_colors = np.array([[r, g, 0]] * 21, dtype=np.uint8)

        rr.log(
            "hand/joints",
            rr.Points3D(
                positions=kps,
                colors=joint_colors,
                radii=0.004,
            ),
        )

        # Skeleton edges.
        edge_starts = np.array([kps[s] for s, _ in HAND_CONNECTIONS], dtype=np.float32)
        edge_ends   = np.array([kps[e] for _, e in HAND_CONNECTIONS], dtype=np.float32)
        segments = np.stack([edge_starts, edge_ends], axis=1)  # (E, 2, 3)
        rr.log(
            "hand/skeleton",
            rr.LineStrips3D(
                strips=segments,
                colors=[[200, 200, 200]] * len(segments),
            ),
        )

        # Wrist coordinate frame axes.
        T = transforms[i]
        origin = T[:3, 3]
        x_axis = T[:3, 0] * 0.02
        y_axis = T[:3, 1] * 0.02
        z_axis = T[:3, 2] * 0.02
        rr.log(
            "hand/wrist_transform",
            rr.Arrows3D(
                origins=np.array([origin, origin, origin], dtype=np.float32),
                vectors=np.array([x_axis, y_axis, z_axis], dtype=np.float32),
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            ),
        )

    logger.info("Rerun visualisation complete.")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    frames_dir: Path,
    output_dir: Path,
    hamer_script: Path,
    pinch_threshold: float,
    visualize: bool,
) -> Path:
    """Execute the full hand-tracking pipeline.

    Steps:

    1. Run HaMeR via subprocess to obtain per-frame MANO predictions.
    2. Load and validate the output ``.npz``.
    3. Extract wrist SE(3) transforms ``(N, 4, 4)``.
    4. Extract 3-D MANO keypoints ``(N, 21, 3)``.
    5. Enforce metric scale (median wrist-to-middle-MCP -> L_ref = 0.085 m).
    6. Compute gripper aperture (continuous + binary).
    7. Save ``hand_poses.npz`` to *output_dir*.
    8. Optionally stream a Rerun visualisation.

    Args:
        frames_dir: Directory containing input RGB frames (``.png`` / ``.jpg``).
        output_dir: Directory where HaMeR and pipeline outputs are written.
        hamer_script: Path to HaMeR's ``demo.py``.
        pinch_threshold: Pinch distance (metres) below which the gripper is
            considered closed.
        visualize: If ``True``, launch a Rerun viewer after processing.

    Returns:
        Path to the saved ``hand_poses.npz`` file.

    Raises:
        FileNotFoundError: If *frames_dir* does not exist or contains no
            recognised image files.
        RuntimeError: If HaMeR returns no frames.
    """
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    import re

    def _nat_key(p: Path) -> list[int | str]:
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", p.name)]

    raw_files = (
        list(frames_dir.glob("*.png"))
        + list(frames_dir.glob("*.jpg"))
        + list(frames_dir.glob("*.jpeg"))
    )
    image_files = sorted(raw_files, key=_nat_key)
    if not image_files:
        raise FileNotFoundError(
            f"No PNG/JPG images found in {frames_dir}. "
            "Ensure the directory contains extracted video frames."
        )
    logger.info("Found %d image frames in %s.", len(image_files), frames_dir)

    # --- Step 1: Run HaMeR ---------------------------------------------------
    npz_path = run_hamer(hamer_script, frames_dir, output_dir)

    # --- Step 2: Load HaMeR output -------------------------------------------
    logger.info("Loading HaMeR output from %s.", npz_path)
    npz_data: dict[str, npt.NDArray] = dict(np.load(npz_path, allow_pickle=True))
    logger.info("HaMeR output keys: %s", list(npz_data.keys()))

    # --- Step 3: Extract wrist transforms ------------------------------------
    transforms_raw = extract_wrist_transforms(npz_data)  # (N, 4, 4)

    # --- Step 4: Extract 3-D keypoints ---------------------------------------
    keypoints_raw = extract_keypoints_3d(npz_data)       # (N, 21, 3)

    n_hamer = len(transforms_raw)
    if n_hamer == 0:
        raise RuntimeError(
            "HaMeR returned zero frames. Check input images and model weights."
        )

    n_kp = len(keypoints_raw)
    if n_kp != n_hamer:
        logger.warning(
            "Frame count mismatch: transforms=%d, keypoints=%d. Truncating to minimum.",
            n_hamer, n_kp,
        )
        n = min(n_hamer, n_kp)
        transforms_raw = transforms_raw[:n]
        keypoints_raw  = keypoints_raw[:n]

    logger.info("Processing %d frames.", len(transforms_raw))

    # --- Step 5: Metric scale enforcement ------------------------------------
    keypoints_scaled, transforms_scaled = enforce_metric_scale(
        keypoints_raw, transforms_raw, l_ref=L_REF_METRES
    )

    # --- Step 6: Gripper aperture --------------------------------------------
    gripper_continuous, gripper_binary = compute_gripper_aperture(
        keypoints_scaled, pinch_threshold=pinch_threshold
    )

    # --- Step 7: Save results ------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    out_npz = output_dir / "hand_poses.npz"
    np.savez(
        str(out_npz),
        T_cam_to_hand=transforms_scaled.astype(np.float64),
        gripper_continuous=gripper_continuous.astype(np.float64),
        gripper_binary=gripper_binary.astype(np.float64),
        keypoints_3d=keypoints_scaled.astype(np.float64),
    )
    # Context Roadmap deliverables
    np.save(output_dir / "hand_poses_cam.npy", transforms_scaled.astype(np.float64))
    np.save(output_dir / "gripper_states.npy", gripper_binary.astype(np.float64))
    logger.info("Saved hand poses to %s (N=%d frames).", out_npz, len(transforms_scaled))

    # --- Step 8: Rerun visualisation -----------------------------------------
    if visualize:
        visualize_hand_skeleton(keypoints_scaled, transforms_scaled, gripper_continuous)

    return out_npz


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for the hand_tracking CLI.

    Returns:
        Configured :class:`argparse.ArgumentParser` instance.
    """
    parser = argparse.ArgumentParser(
        prog="hand_tracking",
        description=(
            "Module 3 -- HaMeR hand tracking for the Video-to-LIBERO pipeline. "
            "Runs HaMeR on an RGB image sequence, extracts wrist transforms and "
            "MANO keypoints, computes gripper aperture, enforces metric scale, "
            "and saves hand_poses.npz."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--frames_dir",
        type=Path,
        required=True,
        help="Directory containing input RGB frames (*.png / *.jpg).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help=(
            "Directory where HaMeR intermediate outputs and "
            "hand_poses.npz are written."
        ),
    )
    parser.add_argument(
        "--hamer_script",
        type=Path,
        default=Path("./third_party/hamer/demo.py"),
        help="Path to HaMeR's demo.py entry point.",
    )
    parser.add_argument(
        "--pinch_threshold",
        type=float,
        default=0.04,
        help=(
            "Thumb-tip to index-tip distance (metres) below which the gripper "
            "is classified as closed (binary = 1). Above threshold -> open (binary = -1)."
        ),
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help=(
            "Launch a Rerun SDK viewer to inspect the 3-D hand skeleton "
            "over time."
        ),
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    """Entry point for the hand_tracking CLI.

    Args:
        argv: Optional list of command-line arguments (defaults to
            ``sys.argv[1:]`` when ``None``).
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    logger.info("=== Hand Tracking Module 3 ===")
    logger.info("  frames_dir      : %s", args.frames_dir)
    logger.info("  output_dir      : %s", args.output_dir)
    logger.info("  hamer_script    : %s", args.hamer_script)
    logger.info("  pinch_threshold : %.4f m", args.pinch_threshold)
    logger.info("  visualize       : %s", args.visualize)

    out_path = run_pipeline(
        frames_dir=args.frames_dir,
        output_dir=args.output_dir,
        hamer_script=args.hamer_script,
        pinch_threshold=args.pinch_threshold,
        visualize=args.visualize,
    )
    logger.info("Pipeline complete. Output: %s", out_path)


if __name__ == "__main__":
    main()
