"""src.slam.plane_detector — Visual SLAM camera tracking and RANSAC table plane estimation."""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

try:
    import open3d as o3d
except ImportError:
    o3d = None

from ..utils.coordinates import compute_table_alignment
from ..utils.io import load_calibration_txt, load_camera_poses, save_camera_poses

logger = logging.getLogger("slam")


def run_droid_slam_subprocess(
    slam_script: Path | str,
    frames_dir: Path | str,
    calib_txt: Path | str,
    output_dir: Path | str,
) -> None:
    """Invoke DROID-SLAM tracking script as an external subprocess.

    Args:
        slam_script: Path to DROID-SLAM demo.py.
        frames_dir: Directory containing extracted video frames.
        calib_txt: Calibration file containing fx fy cx cy.
        output_dir: Destination directory for SLAM outputs.
    """
    slam_path = Path(slam_script)
    if not slam_path.is_file():
        raise FileNotFoundError(f"DROID-SLAM entry point not found: {slam_path}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(slam_path),
        "--imagedir", str(Path(frames_dir)),
        "--calib", str(Path(calib_txt)),
        "--reconstruction_path", str(out_dir),
        "--disable_vis",
    ]

    logger.info("Executing DROID-SLAM: %s", " ".join(cmd))
    result = subprocess.run(cmd, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    logger.info("DROID-SLAM finished successfully.")
    logger.debug("DROID-SLAM output:\n%s", result.stdout)


def build_scene_point_cloud(
    slam_dir: Path | str,
    poses: npt.NDArray[np.float64],
    calib_txt: Path | str,
):
    """Build an Open3D point cloud from SLAM depth maps, bundled PLY, or camera frustums.

    Args:
        slam_dir: Directory with SLAM outputs.
        poses: (N, 4, 4) camera poses (T_world_to_cam).
        calib_txt: Calibration file with fx fy cx cy.

    Returns:
        open3d.geometry.PointCloud instance.
    """
    if o3d is None:
        raise ImportError(
            "Open3D is required for 3D point cloud operations. "
            "Please install it (e.g. conda install -c open3d-admin -c conda-forge open3d or pip install open3d)."
        )
    slam_p = Path(slam_dir)
    fx, fy, cx, cy = load_calibration_txt(calib_txt)

    # 1. Bundled point cloud file
    for candidate in ("pointcloud.ply", "pointcloud.pcd", "reconstruction.ply"):
        ply_p = slam_p / candidate
        if ply_p.is_file():
            logger.info("Loading bundled point cloud from %s", ply_p)
            pcd = o3d.io.read_point_cloud(str(ply_p))
            if len(pcd.points) > 0:
                return pcd

    # 2. Back-project depth maps
    depth_files = sorted(slam_p.glob("depth_*.npy"))
    if depth_files:
        logger.info("Back-projecting %d depth maps ...", len(depth_files))
        all_pts: list[npt.NDArray[np.float64]] = []

        for i, dpath in enumerate(depth_files):
            if i >= len(poses):
                break
            depth = np.load(dpath).astype(np.float32)
            h, w = depth.shape
            ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
            mask = (depth > 0) & (depth < 5.0)

            z = depth[mask].astype(np.float64)
            x = (xs[mask] - cx) * z / fx
            y = (ys[mask] - cy) * z / fy

            pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=1)  # (M, 4)

            # Invert T_world_to_cam to get T_cam_to_world
            T_w2c = poses[i]
            T_c2w = np.linalg.inv(T_w2c)
            pts_world = (T_c2w @ pts_cam.T).T[:, :3]
            all_pts.append(pts_world)

        if all_pts:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(np.concatenate(all_pts, axis=0))
            logger.info("Built point cloud with %d points from depth maps.", len(pcd.points))
            return pcd

    # 3. Fallback: camera optical centres
    logger.warning("No depth maps found. Falling back to camera centres.")
    centres: list[npt.NDArray[np.float64]] = []
    for T in poses:
        R = T[:3, :3]
        t = T[:3, 3]
        centres.append(-R.T @ t)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(centres, dtype=np.float64))
    return pcd


def fit_table_plane_ransac(
    pcd,
    distance_threshold: float = 0.01,
    ransac_n: int = 3,
    num_iterations: int = 1000,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int32]]:
    """Segment the dominant table plane using RANSAC.

    Args:
        pcd: Point cloud containing table points.
        distance_threshold: Maximum distance (m) to plane for inliers.
        ransac_n: Minimum points sampled per iteration.
        num_iterations: Number of RANSAC iterations.

    Returns:
        Tuple of (plane_model, inlier_indices) where plane_model is [a, b, c, d].
    """
    if o3d is None:
        raise ImportError(
            "Open3D is required for RANSAC plane fitting. "
            "Please install it in your conda environment."
        )
    n_pts = len(pcd.points)
    if n_pts < ransac_n:
        raise RuntimeError(f"Point cloud has only {n_pts} points; need at least {ransac_n} for RANSAC.")

    plane_model_list, inlier_indices = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )

    plane_model = np.asarray(plane_model_list, dtype=np.float64)
    inliers = np.asarray(inlier_indices, dtype=np.int32)
    a, b, c, d = plane_model

    logger.info(
        "Fitted table plane: %.4fx + %.4fy + %.4fz + %.4f = 0 (%d inliers / %d points)",
        a, b, c, d, len(inliers), n_pts,
    )
    return plane_model, inliers


def process_camera_tracking(
    frames_dir: Path | str,
    calib_txt: Path | str,
    output_dir: Path | str,
    slam_script: Optional[Path | str] = None,
    distance_threshold: float = 0.01,
) -> tuple[Path, Path, Path]:
    """Execute complete Module 2 tracking and table alignment pipeline.

    Args:
        frames_dir: Video frames directory.
        calib_txt: Intrinsics file.
        output_dir: Root output directory.
        slam_script: Optional path to DROID-SLAM demo.py.
        distance_threshold: RANSAC distance threshold.

    Returns:
        Tuple of (poses_npy, camera_poses_npz, table_transform_npz).
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slam_out = out_dir / "slam"

    if slam_script is not None and Path(slam_script).is_file():
        run_droid_slam_subprocess(slam_script, frames_dir, calib_txt, slam_out)

    poses = load_camera_poses(slam_out)
    npy_path, npz_path = save_camera_poses(out_dir, poses)

    pcd = build_scene_point_cloud(slam_out, poses, calib_txt)
    pcd_out = out_dir / "scene_pointcloud.ply"
    o3d.io.write_point_cloud(str(pcd_out), pcd)

    plane_model, _inliers = fit_table_plane_ransac(pcd, distance_threshold=distance_threshold)
    T_world_to_table, R_align, t_align = compute_table_alignment(plane_model)

    transform_out = out_dir / "table_transform.npz"
    np.savez_compressed(
        transform_out,
        T_world_to_table=T_world_to_table,
        R_align=R_align,
        t_align=t_align,
        plane_model=plane_model,
    )

    logger.info("Camera tracking and table alignment completed.")
    logger.info("  Poses (.npy)       -> %s", npy_path)
    logger.info("  Poses (.npz)       -> %s", npz_path)
    logger.info("  Table transform    -> %s", transform_out)

    return npy_path, npz_path, transform_out
