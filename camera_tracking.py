"""camera_tracking.py — Module 2: Video-to-LIBERO Pipeline.

Runs DROID-SLAM on a directory of extracted video frames, loads the recovered
camera poses, builds an Open3D point cloud from SLAM depth maps or a bundled
point cloud file, fits a ground-plane model with RANSAC, and saves the
table-to-world alignment transform.

Typical usage::

    python camera_tracking.py \\
        --frames_dir  data/frames \\
        --calib_txt   data/calib.txt \\
        --output_dir  data/tracking_out \\
        --visualize
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import numpy.typing as npt
import open3d as o3d

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
_LOG = logging.getLogger("camera_tracking")


# ---------------------------------------------------------------------------
# DROID-SLAM runner
# ---------------------------------------------------------------------------

def run_droid_slam(
    slam_script: Path,
    frames_dir: Path,
    calib_txt: Path,
    output_dir: Path,
) -> None:
    """Invoke DROID-SLAM as a subprocess and block until completion.

    DROID-SLAM is expected to write its results (poses .npy/.npz and
    optionally depth maps) inside *output_dir*.

    Args:
        slam_script: Absolute path to the DROID-SLAM entry-point script
            (e.g. ``third_party/DROID-SLAM/demo.py``).
        frames_dir: Directory that contains the PNG/JPG frame sequence
            (sorted lexicographically = time order).
        calib_txt: Text file with a single line ``fx fy cx cy``.
        output_dir: Destination directory for SLAM outputs.

    Raises:
        FileNotFoundError: If *slam_script* does not exist.
        subprocess.CalledProcessError: If DROID-SLAM exits with a non-zero
            return code.
    """
    if not slam_script.is_file():
        raise FileNotFoundError(f"SLAM script not found: {slam_script}")

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd: list[str] = [
        sys.executable,
        str(slam_script),
        "--imagedir", str(frames_dir),
        "--calib",   str(calib_txt),
        "--reconstruction_path", str(output_dir),
        # DROID-SLAM demo flags that ensure output is written to disk
        "--disable_vis",
    ]

    _LOG.info("Launching DROID-SLAM:\n  %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    _LOG.info("DROID-SLAM stdout:\n%s", result.stdout)
    _LOG.info("DROID-SLAM finished successfully.")


# ---------------------------------------------------------------------------
# Pose loading
# ---------------------------------------------------------------------------

def load_poses(output_dir: Path) -> npt.NDArray[np.float64]:
    """Load camera-to-world (or world-to-camera) poses written by DROID-SLAM.

    DROID-SLAM may write poses as:
      * ``poses.npy``  — shape (N, 4, 4) float64
      * ``poses.npz``  — key ``"poses"`` with the same shape
      * ``traj_est.npy`` — shape (N, 7) quaternion+translation

    The function tries each candidate in order and converts (N, 7) twist
    vectors to (N, 4, 4) homogeneous matrices when needed.

    Args:
        output_dir: Directory that DROID-SLAM wrote its outputs to.

    Returns:
        ``poses`` — NumPy array of shape ``(N, 4, 4)`` with dtype float64,
        representing T_world_to_cam (or T_cam_to_world, depending on the
        SLAM convention — downstream callers should be aware of the sign).

    Raises:
        FileNotFoundError: If no recognised pose file is found in
            *output_dir*.
    """
    candidates: list[tuple[str, str | None]] = [
        ("poses.npy",     None),
        ("poses.npz",     "poses"),
        ("traj_est.npy",  None),
        ("traj_est.npz",  "traj_est"),
    ]

    for filename, npz_key in candidates:
        path = output_dir / filename
        if not path.exists():
            continue

        _LOG.info("Loading poses from %s", path)
        if path.suffix == ".npz":
            data = np.load(path)
            raw: npt.NDArray[np.float64] = data[npz_key or "poses"].astype(np.float64)
        else:
            raw = np.load(path).astype(np.float64)

        if raw.ndim == 3 and raw.shape[1:] == (4, 4):
            return raw

        if raw.ndim == 2 and raw.shape[1] == 7:
            # DROID-SLAM traj format: [tx ty tz qx qy qz qw]
            _LOG.info("Converting (N,7) twist poses to (N,4,4) matrices.")
            return _twist_to_matrices(raw)

        raise ValueError(
            f"Unexpected pose array shape {raw.shape} in {path}. "
            "Expected (N,4,4) or (N,7)."
        )

    raise FileNotFoundError(
        f"No recognised pose file found in {output_dir}. "
        f"Tried: {[c[0] for c in candidates]}"
    )


def _twist_to_matrices(
    twists: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Convert an (N, 7) array of [tx ty tz qx qy qz qw] to (N, 4, 4).

    Args:
        twists: Array of shape ``(N, 7)`` where each row is
            ``[tx, ty, tz, qx, qy, qz, qw]``.

    Returns:
        Homogeneous transformation matrices of shape ``(N, 4, 4)``.
    """
    n = twists.shape[0]
    mats = np.eye(4, dtype=np.float64)[np.newaxis].repeat(n, axis=0)

    tx, ty, tz = twists[:, 0], twists[:, 1], twists[:, 2]
    qx, qy, qz, qw = twists[:, 3], twists[:, 4], twists[:, 5], twists[:, 6]

    # Normalise quaternions
    norms = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    qx, qy, qz, qw = qx / norms, qy / norms, qz / norms, qw / norms

    # Rotation matrix from unit quaternion
    mats[:, 0, 0] = 1 - 2 * (qy**2 + qz**2)
    mats[:, 0, 1] = 2 * (qx * qy - qz * qw)
    mats[:, 0, 2] = 2 * (qx * qz + qy * qw)
    mats[:, 1, 0] = 2 * (qx * qy + qz * qw)
    mats[:, 1, 1] = 1 - 2 * (qx**2 + qz**2)
    mats[:, 1, 2] = 2 * (qy * qz - qx * qw)
    mats[:, 2, 0] = 2 * (qx * qz - qy * qw)
    mats[:, 2, 1] = 2 * (qy * qz + qx * qw)
    mats[:, 2, 2] = 1 - 2 * (qx**2 + qy**2)

    mats[:, 0, 3] = tx
    mats[:, 1, 3] = ty
    mats[:, 2, 3] = tz

    return mats


# ---------------------------------------------------------------------------
# Calibration loading
# ---------------------------------------------------------------------------

def load_calibration(calib_txt: Path) -> tuple[float, float, float, float]:
    """Parse camera intrinsics from a single-line calibration file.

    The file must contain exactly four whitespace-separated floats on the
    first non-comment line: ``fx fy cx cy``.

    Args:
        calib_txt: Path to the calibration text file.

    Returns:
        Tuple ``(fx, fy, cx, cy)`` as Python floats.

    Raises:
        ValueError: If the file cannot be parsed as four floats.
    """
    for line in calib_txt.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(
                f"calib.txt must have at least 4 values (fx fy cx cy), got: {line!r}"
            )
        fx, fy, cx, cy = (float(p) for p in parts[:4])
        _LOG.info("Calibration loaded: fx=%.2f fy=%.2f cx=%.2f cy=%.2f", fx, fy, cx, cy)
        return fx, fy, cx, cy

    raise ValueError(f"No valid intrinsic line found in {calib_txt}")


# ---------------------------------------------------------------------------
# Point cloud construction
# ---------------------------------------------------------------------------

def _depth_to_points(
    depth: npt.NDArray[np.float32],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    T_cam_to_world: npt.NDArray[np.float64],
    max_depth: float = 5.0,
) -> npt.NDArray[np.float64]:
    """Back-project a depth map into 3-D world-frame points.

    Args:
        depth: HxW float32 depth image (metres).
        fx: Focal length in x (pixels).
        fy: Focal length in y (pixels).
        cx: Principal point x (pixels).
        cy: Principal point y (pixels).
        T_cam_to_world: 4x4 homogeneous camera-to-world transform.
        max_depth: Depth values beyond this threshold are discarded.

    Returns:
        Array of shape ``(M, 3)`` with valid 3-D world-frame points.
    """
    h, w = depth.shape
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    mask = (depth > 0) & (depth < max_depth)

    z = depth[mask].astype(np.float64)
    x = (xs[mask] - cx) * z / fx
    y = (ys[mask] - cy) * z / fy

    pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=1)  # (M, 4)
    pts_world = (T_cam_to_world @ pts_cam.T).T[:, :3]        # (M, 3)
    return pts_world


def build_point_cloud(
    output_dir: Path,
    poses: npt.NDArray[np.float64],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> o3d.geometry.PointCloud:
    """Build an Open3D point cloud from SLAM outputs.

    Strategy (tried in order):
      1. ``pointcloud.ply`` / ``pointcloud.pcd`` — directly loaded.
      2. Depth maps ``depth_XXXXXX.npy`` — back-projected using intrinsics
         and the corresponding camera pose.
      3. Camera centres extracted from *poses* — minimal fallback.

    Args:
        output_dir: Directory written by DROID-SLAM.
        poses: ``(N, 4, 4)`` camera pose matrices (world-to-cam convention
            is inverted internally to cam-to-world for back-projection).
        fx: Focal length x.
        fy: Focal length y.
        cx: Principal point x.
        cy: Principal point y.

    Returns:
        An :class:`open3d.geometry.PointCloud` populated with 3-D points.
    """
    # --- 1. Bundled point cloud file ---
    for fname in ("pointcloud.ply", "pointcloud.pcd", "reconstruction.ply"):
        pc_path = output_dir / fname
        if pc_path.exists():
            _LOG.info("Loading bundled point cloud from %s", pc_path)
            pcd = o3d.io.read_point_cloud(str(pc_path))
            if len(pcd.points) > 0:
                return pcd

    # --- 2. Per-frame depth maps ---
    depth_files = sorted(output_dir.glob("depth_*.npy"))
    if depth_files:
        _LOG.info("Back-projecting %d depth maps ...", len(depth_files))
        all_pts: list[npt.NDArray[np.float64]] = []

        for i, dpath in enumerate(depth_files):
            if i >= len(poses):
                break
            depth = np.load(dpath).astype(np.float32)

            # Determine pose convention: attempt world-to-cam -> invert
            T_w2c: npt.NDArray[np.float64] = poses[i]
            T_c2w = np.linalg.inv(T_w2c)

            pts = _depth_to_points(depth, fx, fy, cx, cy, T_c2w)
            all_pts.append(pts)

        if all_pts:
            pts_all = np.concatenate(all_pts, axis=0)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts_all)
            _LOG.info("Built point cloud with %d points from depth maps.", len(pcd.points))
            return pcd

    # --- 3. Fallback: camera centres from poses ---
    _LOG.warning(
        "No depth maps or bundled point cloud found. "
        "Falling back to camera-centre point cloud -- plane fit may be inaccurate."
    )
    centres: list[npt.NDArray[np.float64]] = []
    for T in poses:
        # Camera centre in world frame = -R^T t  (when T is world-to-cam)
        R = T[:3, :3]
        t = T[:3, 3]
        centre = -R.T @ t
        centres.append(centre)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(centres))
    _LOG.info("Fallback: %d camera centres used as point cloud.", len(pcd.points))
    return pcd


# ---------------------------------------------------------------------------
# Plane fitting and alignment
# ---------------------------------------------------------------------------

def fit_table_plane(
    pcd: o3d.geometry.PointCloud,
    distance_threshold: float = 0.01,
    ransac_n: int = 3,
    num_iterations: int = 1000,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Fit a dominant plane (table surface) to the point cloud with RANSAC.

    Args:
        pcd: Input point cloud (should contain the table surface).
        distance_threshold: Max point-to-plane distance for inliers (metres).
        ransac_n: Minimum number of points sampled per RANSAC iteration.
        num_iterations: Number of RANSAC iterations.

    Returns:
        Tuple ``(plane_model, inlier_indices)`` where

        * ``plane_model`` -- 1-D array ``[a, b, c, d]`` satisfying
          ``ax + by + cz + d = 0`` with ``||(a,b,c)|| = 1``.
        * ``inlier_indices`` -- integer array of inlier point indices.

    Raises:
        RuntimeError: If the point cloud has fewer than *ransac_n* points.
    """
    n_pts = len(pcd.points)
    if n_pts < ransac_n:
        raise RuntimeError(
            f"Point cloud has only {n_pts} points; need at least {ransac_n} for RANSAC."
        )

    _LOG.info(
        "Running RANSAC plane segmentation on %d points "
        "(dist_thresh=%.4f, n=%d, iters=%d) ...",
        n_pts, distance_threshold, ransac_n, num_iterations,
    )
    plane_model_list, inlier_indices = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations,
    )
    plane_model: npt.NDArray[np.float64] = np.asarray(plane_model_list, dtype=np.float64)
    inliers: npt.NDArray[np.float64] = np.asarray(inlier_indices)

    a, b, c, d = plane_model
    _LOG.info(
        "Plane found: %.4fx + %.4fy + %.4fz + %.4f = 0  (%d inliers / %d points)",
        a, b, c, d, len(inliers), n_pts,
    )
    return plane_model, inliers


def compute_table_transform(
    plane_model: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute a rigid alignment that maps the table plane to Z = 0.

    Given a plane ``ax + by + cz + d = 0``, the function produces:

    * ``R_align`` -- rotation matrix that rotates the plane normal
      ``n = (a, b, c)`` to ``(0, 0, 1)``.
    * ``t_align`` -- translation that shifts the plane origin to Z = 0.

    After applying ``p' = R_align @ p + t_align`` every table-surface point
    will satisfy ``p'_z ~= 0``.

    Args:
        plane_model: 1-D array ``[a, b, c, d]`` from RANSAC plane fit.

    Returns:
        Tuple ``(R_align, t_align)`` where

        * ``R_align`` -- ``(3, 3)`` float64 rotation matrix.
        * ``t_align`` -- ``(3,)`` float64 translation vector.
    """
    a, b, c, d = plane_model
    normal = np.array([a, b, c], dtype=np.float64)
    normal /= np.linalg.norm(normal)

    # Ensure normal points upward (toward +Z after alignment)
    if normal[2] < 0:
        normal = -normal
        d = -d

    target = np.array([0.0, 0.0, 1.0])

    # Rodrigues' rotation formula: rotate 'normal' onto 'target'
    v = np.cross(normal, target)
    s = np.linalg.norm(v)
    c_dot = float(np.dot(normal, target))

    if s < 1e-9:
        # Already aligned (or anti-aligned)
        R_align = np.eye(3, dtype=np.float64) if c_dot > 0 else -np.eye(3, dtype=np.float64)
    else:
        vx = np.array(
            [[0, -v[2], v[1]],
             [v[2], 0, -v[0]],
             [-v[1], v[0], 0]],
            dtype=np.float64,
        )
        R_align = np.eye(3) + vx + vx @ vx * ((1 - c_dot) / (s**2))

    # A point on the original plane: project origin onto plane along normal
    # plane_point = -d * normal  (satisfies ax+by+cz+d=0)
    plane_origin = -d * normal  # in original coordinates

    # After rotation, this point should have Z = 0
    rotated_origin = R_align @ plane_origin
    t_align = np.array([0.0, 0.0, -rotated_origin[2]])

    _LOG.info("R_align:\n%s", R_align)
    _LOG.info("t_align: %s", t_align)
    return R_align, t_align


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize(
    pcd: o3d.geometry.PointCloud,
    plane_model: npt.NDArray[np.float64],
    inlier_indices: npt.NDArray[np.float64],
    poses: npt.NDArray[np.float64],
) -> None:
    """Launch an interactive Open3D visualisation window.

    Displays:
      * Inlier points (table surface) in red.
      * Outlier points in grey.
      * Camera frustum centres as a green line set.
      * A mesh disk representing the fitted plane (blue, semi-transparent).

    Args:
        pcd: Full point cloud.
        plane_model: ``[a, b, c, d]`` plane coefficients.
        inlier_indices: Indices of table-surface inlier points.
        poses: ``(N, 4, 4)`` camera pose matrices.
    """
    geometries: list[o3d.geometry.Geometry] = []

    # Colour inliers / outliers
    inlier_cloud = pcd.select_by_index(inlier_indices.tolist())
    outlier_cloud = pcd.select_by_index(inlier_indices.tolist(), invert=True)
    inlier_cloud.paint_uniform_color([0.8, 0.1, 0.1])
    outlier_cloud.paint_uniform_color([0.5, 0.5, 0.5])
    geometries += [inlier_cloud, outlier_cloud]

    # Fitted plane disk
    a, b, c, d = plane_model
    normal = np.array([a, b, c], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    plane_centre = -d * normal

    mesh_disk = o3d.geometry.TriangleMesh.create_cylinder(radius=0.4, height=0.003)
    # Rotate disk so its axis aligns with the plane normal
    z_axis = np.array([0.0, 0.0, 1.0])
    v = np.cross(z_axis, normal)
    s = np.linalg.norm(v)
    c_dot = float(np.dot(z_axis, normal))
    if s > 1e-9:
        vx = np.array(
            [[0, -v[2], v[1]],
             [v[2], 0, -v[0]],
             [-v[1], v[0], 0]],
            dtype=np.float64,
        )
        R_disk = np.eye(3) + vx + vx @ vx * ((1 - c_dot) / s**2)
    else:
        R_disk = np.eye(3, dtype=np.float64)

    T_disk = np.eye(4)
    T_disk[:3, :3] = R_disk
    T_disk[:3, 3] = plane_centre
    mesh_disk.transform(T_disk)
    mesh_disk.paint_uniform_color([0.2, 0.4, 0.9])
    geometries.append(mesh_disk)

    # Camera trajectory line set
    if len(poses) >= 2:
        centres = []
        for T in poses:
            R = T[:3, :3]
            t = T[:3, 3]
            centres.append(-R.T @ t)
        pts = np.array(centres)
        lines = [[i, i + 1] for i in range(len(pts) - 1)]
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(pts)
        ls.lines = o3d.utility.Vector2iVector(lines)
        ls.paint_uniform_color([0.1, 0.8, 0.1])
        geometries.append(ls)

    # World frame
    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3))

    _LOG.info("Opening Open3D visualiser -- close the window to continue.")
    o3d.visualization.draw_geometries(
        geometries,
        window_name="Camera Tracking -- Table Plane Fit",
        width=1280,
        height=720,
        point_show_normal=False,
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    frames_dir: Path,
    calib_txt: Path,
    output_dir: Path,
    slam_script: Path,
    visualize_flag: bool,
) -> None:
    """Execute the full camera-tracking and table-alignment pipeline.

    Steps:

    1. Run DROID-SLAM via subprocess.
    2. Load camera poses ``(N, 4, 4)``.
    3. Build Open3D point cloud from depth maps or bundled file.
    4. Fit table plane with RANSAC.
    5. Compute alignment transform and save ``table_transform.npz``.
    6. Optionally visualise results.

    Args:
        frames_dir: Directory of extracted PNG/JPG frames (30 fps).
        calib_txt: Calibration file (single line: ``fx fy cx cy``).
        output_dir: Root output directory for all artefacts.
        slam_script: Path to the DROID-SLAM entry-point script.
        visualize_flag: If ``True``, launch Open3D interactive viewer.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1 -- DROID-SLAM
    slam_output_dir = output_dir / "slam"
    run_droid_slam(slam_script, frames_dir, calib_txt, slam_output_dir)

    # Step 2 -- Load poses
    poses = load_poses(slam_output_dir)
    _LOG.info("Loaded %d camera poses.", len(poses))

    poses_out = output_dir / "poses.npy"
    np.save(poses_out, poses)
    poses_npz = output_dir / "camera_poses.npz"
    np.savez_compressed(poses_npz, T_world_to_cam=poses, poses=poses)
    _LOG.info("Saved poses -> %s and %s", poses_out, poses_npz)

    # Step 3 -- Intrinsics + point cloud
    fx, fy, cx, cy = load_calibration(calib_txt)
    pcd = build_point_cloud(slam_output_dir, poses, fx, fy, cx, cy)

    pcd_out = output_dir / "scene_pointcloud.ply"
    o3d.io.write_point_cloud(str(pcd_out), pcd)
    _LOG.info("Saved point cloud -> %s", pcd_out)

    # Step 4 -- RANSAC plane fit
    plane_model, inlier_indices = fit_table_plane(pcd)

    # Step 5 -- Alignment transform
    R_align, t_align = compute_table_transform(plane_model)
    T_world_to_table = np.eye(4, dtype=np.float64)
    T_world_to_table[:3, :3] = R_align
    T_world_to_table[:3, 3] = t_align

    transform_out = output_dir / "table_transform.npz"
    np.savez(
        transform_out,
        T_world_to_table=T_world_to_table,
        R_align=R_align,
        t_align=t_align,
        plane_model=plane_model,
    )
    _LOG.info("Saved table transform -> %s", transform_out)

    # Step 6 -- Visualise
    if visualize_flag:
        visualize(pcd, plane_model, inlier_indices, poses)

    _LOG.info("Pipeline complete. Outputs in %s", output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for the camera-tracking module.

    Returns:
        Configured :class:`argparse.ArgumentParser` instance.
    """
    parser = argparse.ArgumentParser(
        prog="camera_tracking",
        description=(
            "Module 2 -- Video-to-LIBERO: Run DROID-SLAM, load poses, "
            "fit a table plane, and export the alignment transform."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--frames_dir",
        type=Path,
        required=True,
        help="Directory of extracted PNG/JPG frames (sorted, 30 fps).",
    )
    parser.add_argument(
        "--calib_txt",
        type=Path,
        required=True,
        help="Calibration file with a single line: fx fy cx cy.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Root output directory for poses, point cloud, and transforms.",
    )
    parser.add_argument(
        "--slam_script",
        type=Path,
        default=Path("./third_party/DROID-SLAM/demo.py"),
        help="Path to the DROID-SLAM entry-point Python script.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        default=False,
        help="Launch Open3D interactive visualiser after plane fitting.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point for the camera-tracking module.

    Args:
        argv: Optional list of CLI arguments (defaults to ``sys.argv``).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    run_pipeline(
        frames_dir=args.frames_dir.resolve(),
        calib_txt=args.calib_txt.resolve(),
        output_dir=args.output_dir.resolve(),
        slam_script=args.slam_script.resolve(),
        visualize_flag=args.visualize,
    )


if __name__ == "__main__":
    main()
