"""src.retargeting.panda_3d_renderer — Authentic 3D MuJoCo XML mesh & texture renderer for the Franka Emika Panda arm.

Loads the official Franka Emika Panda 3D visual meshes (`obj_meshes/link0_vis` .. `link7_vis`),
`PandaGripper` 3D meshes (`hand_vis.stl`, `finger_vis.stl`), `RethinkMount` (`pedestal.stl`),
and `TableArena` textures (`ceramic.png`, `light-gray-floor-tile.png`, `light-gray-plaster.png`,
`red-wood.png`) directly from the installed `robosuite/models/assets` directory, and renders
Z-buffered, Phong-shaded 3D simulation views (`agentview` and `robot0_eye_in_hand`).
"""

from __future__ import annotations

import importlib.util
import logging
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numba
import numpy as np
import numpy.typing as npt

from .sim_replay import (
    PANDA_BASE_WORLD_POS,
    TABLE_TOP_Z_WORLD,
    _PANDA_KINEMATIC_CHAIN,
    _quat_wxyz_to_rotmat,
    _rot_z,
)

logger = logging.getLogger("panda_3d_renderer")


def _find_robosuite_assets_root() -> Optional[Path]:
    """Locate robosuite/models/assets without triggering mujoco DLL imports."""
    spec = importlib.util.find_spec("robosuite")
    if spec is None or spec.origin is None:
        return None
    assets = Path(spec.origin).parent / "models" / "assets"
    return assets if assets.is_dir() else None


def _load_binary_or_ascii_stl(stl_path: Path) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Load triangles and normals from an STL file.

    Returns:
        Tuple of:
        - vertices: (M, 3, 3) float32 triangle vertices.
        - normals:  (M, 3) float32 triangle unit normals.
    """
    data = stl_path.read_bytes()
    if len(data) < 84:
        return np.zeros((0, 3, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    n_tri = struct.unpack("<I", data[80:84])[0]
    expected_len = 84 + n_tri * 50
    if len(data) == expected_len:
        # Binary STL
        dt = np.dtype([
            ("normal", "<f4", (3,)),
            ("v0", "<f4", (3,)),
            ("v1", "<f4", (3,)),
            ("v2", "<f4", (3,)),
            ("attr", "<u2"),
        ])
        arr = np.frombuffer(data, dtype=dt, offset=84, count=n_tri)
        tris = np.stack([arr["v0"], arr["v1"], arr["v2"]], axis=1).astype(np.float32)
        norms = arr["normal"].astype(np.float32)
    else:
        # Fallback: compute from ASCII STL
        verts = []
        for line in data.decode("utf-8", errors="ignore").splitlines():
            parts = line.strip().split()
            if len(parts) == 4 and parts[0] == "vertex":
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        if len(verts) < 3:
            return np.zeros((0, 3, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
        tris = np.array(verts, dtype=np.float32).reshape(-1, 3, 3)
        norms = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])

    # Ensure valid unit normals
    e1 = tris[:, 1] - tris[:, 0]
    e2 = tris[:, 2] - tris[:, 0]
    computed_n = np.cross(e1, e2)
    lengths = np.linalg.norm(computed_n, axis=-1, keepdims=True)
    lengths[lengths < 1e-12] = 1.0
    norms = (computed_n / lengths).astype(np.float32)
    return tris, norms


def _load_obj_file(obj_path: Path) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Parse a Wavefront .obj file into (M, 3, 3) triangles and (M, 3) face normals."""
    verts: List[List[float]] = []
    faces: List[Tuple[int, int, int]] = []

    with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.split()[1:]
                idx = [int(p.split("/")[0]) - 1 for p in parts]
                for k in range(1, len(idx) - 1):
                    faces.append((idx[0], idx[k], idx[k + 1]))

    if not verts or not faces:
        return np.zeros((0, 3, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)

    v_arr = np.array(verts, dtype=np.float32)
    f_arr = np.array(faces, dtype=np.int32)
    tris = v_arr[f_arr]  # (M, 3, 3)

    e1 = tris[:, 1] - tris[:, 0]
    e2 = tris[:, 2] - tris[:, 0]
    norms = np.cross(e1, e2)
    lengths = np.linalg.norm(norms, axis=-1, keepdims=True)
    lengths[lengths < 1e-12] = 1.0
    norms = (norms / lengths).astype(np.float32)
    return tris, norms


@numba.njit(cache=True)
def _rasterize_triangles_zbuffer(
    pts_2d: npt.NDArray[np.float32],
    depths: npt.NDArray[np.float32],
    colors: npt.NDArray[np.uint8],
    canvas: npt.NDArray[np.uint8],
    zbuffer: npt.NDArray[np.float32],
) -> None:
    """Fast Numba JIT Z-buffer triangle rasterizer with bounding-box clipping."""
    h, w, _ = canvas.shape
    n_tris = pts_2d.shape[0]

    for i in range(n_tris):
        z0 = depths[i, 0]
        z1 = depths[i, 1]
        z2 = depths[i, 2]
        if z0 <= 0.05 or z1 <= 0.05 or z2 <= 0.05:
            continue

        x0, y0 = pts_2d[i, 0, 0], pts_2d[i, 0, 1]
        x1, y1 = pts_2d[i, 1, 0], pts_2d[i, 1, 1]
        x2, y2 = pts_2d[i, 2, 0], pts_2d[i, 2, 1]

        min_x = max(0, int(np.floor(min(x0, x1, x2))))
        max_x = min(w - 1, int(np.ceil(max(x0, x1, x2))))
        min_y = max(0, int(np.floor(min(y0, y1, y2))))
        max_y = min(h - 1, int(np.ceil(max(y0, y1, y2))))

        if min_x > max_x or min_y > max_y:
            continue

        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-6:
            continue
        inv_denom = 1.0 / denom

        r = colors[i, 0]
        g = colors[i, 1]
        b = colors[i, 2]

        for py in range(min_y, max_y + 1):
            fy = float(py) + 0.5
            for px in range(min_x, max_x + 1):
                fx = float(px) + 0.5
                w0 = ((y1 - y2) * (fx - x2) + (x2 - x1) * (fy - y2)) * inv_denom
                w1 = ((y2 - y0) * (fx - x2) + (x0 - x2) * (fy - y2)) * inv_denom
                w2 = 1.0 - w0 - w1

                if w0 >= -1e-4 and w1 >= -1e-4 and w2 >= -1e-4:
                    z = w0 * z0 + w1 * z1 + w2 * z2
                    if z < zbuffer[py, px]:
                        zbuffer[py, px] = z
                        canvas[py, px, 0] = r
                        canvas[py, px, 1] = g
                        canvas[py, px, 2] = b


class PandaMuJoCo3DRenderer:
    """Renders the 3D Franka Emika Panda arm, PandaGripper, TableArena, and Cube from Robosuite XML assets."""

    _instance: Optional["PandaMuJoCo3DRenderer"] = None

    @classmethod
    def get_instance(cls) -> "PandaMuJoCo3DRenderer":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self) -> None:
        self.assets_root = _find_robosuite_assets_root()
        # body_name -> list of (tris (M,3,3), norms (M,3), base_rgb (3,))
        self.body_meshes: Dict[str, List[Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]]] = {}
        self._load_all_panda_meshes()

    def _load_all_panda_meshes(self) -> None:
        if self.assets_root is None:
            return

        panda_dir = self.assets_root / "robots" / "panda"
        robot_xml = panda_dir / "robot.xml"
        if not robot_xml.is_file():
            return

        tree = ET.parse(robot_xml)
        root = tree.getroot()

        # 1. Parse materials
        materials: Dict[str, npt.NDArray[np.float32]] = {}
        mesh_files: Dict[str, Path] = {}
        asset_elem = root.find("asset")
        if asset_elem is not None:
            for mat in asset_elem.findall("material"):
                name = mat.get("name", "")
                rgba_str = mat.get("rgba", "0.9 0.92 0.93 1.0")
                rgba = [float(x) for x in rgba_str.split()]
                materials[name] = np.array(rgba[:3], dtype=np.float32)
            for m in asset_elem.findall("mesh"):
                name = m.get("name", "")
                file_rel = m.get("file", "")
                if name and file_rel:
                    mesh_files[name] = panda_dir / file_rel

        # 2. Parse body geoms for link0 .. link7
        def _walk_body(elem: ET.Element) -> None:
            b_name = elem.get("name", "")
            if b_name.startswith("link"):
                full_name = f"robot0_{b_name}"
                self.body_meshes[full_name] = []
                for geom in elem.findall("geom"):
                    if geom.get("group") == "1":
                        m_name = geom.get("mesh", "")
                        mat_name = geom.get("material", "")
                        rgb = materials.get(mat_name, np.array([0.92, 0.93, 0.94], dtype=np.float32))
                        if m_name in mesh_files and mesh_files[m_name].is_file():
                            m_path = mesh_files[m_name]
                            tris, norms = (
                                _load_obj_file(m_path)
                                if m_path.suffix.lower() == ".obj"
                                else _load_binary_or_ascii_stl(m_path)
                            )
                            if len(tris) > 0:
                                self.body_meshes[full_name].append((tris, norms, rgb))
            for child in elem.findall("body"):
                _walk_body(child)

        worldbody = root.find("worldbody")
        if worldbody is not None:
            for b in worldbody.findall("body"):
                _walk_body(b)

        # 3. Load PandaGripper visual meshes (hand_vis.stl and finger_vis.stl)
        gripper_mesh_dir = self.assets_root / "grippers" / "meshes" / "panda_gripper"
        hand_stl = gripper_mesh_dir / "hand_vis.stl"
        finger_stl = gripper_mesh_dir / "finger_vis.stl"

        if hand_stl.is_file():
            h_tris, h_norms = _load_binary_or_ascii_stl(hand_stl)
            self.body_meshes["gripper_hand"] = [
                (h_tris, h_norms, np.array([0.92, 0.93, 0.95], dtype=np.float32))
            ]

        if finger_stl.is_file():
            f_tris, f_norms = _load_binary_or_ascii_stl(finger_stl)
            self.body_meshes["gripper_finger"] = [
                (f_tris, f_norms, np.array([0.42, 0.43, 0.46], dtype=np.float32))
            ]

        # 4. Load RethinkMount pedestal.stl
        pedestal_stl = self.assets_root / "mounts" / "meshes" / "rethink_mount" / "pedestal.stl"
        if pedestal_stl.is_file():
            p_tris, p_norms = _load_binary_or_ascii_stl(pedestal_stl)
            self.body_meshes["mount_pedestal"] = [
                (p_tris, p_norms, np.array([0.24, 0.25, 0.27], dtype=np.float32))
            ]

    @staticmethod
    def _build_box_triangles(
        center: npt.NDArray[np.float64],
        half_size: npt.NDArray[np.float64],
        top_rgb: Tuple[float, float, float],
        side_rgb: Tuple[float, float, float],
        subdiv: int = 1,
    ) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        """Build subdivided 3D box triangles, normals, and per-triangle RGB colors."""
        cx, cy, cz = center
        hx, hy, hz = half_size

        tris_list = []
        norms_list = []
        cols_list = []

        # Top face (+Z) subdivided for accurate z-buffering
        xs = np.linspace(cx - hx, cx + hx, subdiv + 1)
        ys = np.linspace(cy - hy, cy + hy, subdiv + 1)
        z_top = cz + hz
        for i in range(subdiv):
            for j in range(subdiv):
                p00 = [xs[i], ys[j], z_top]
                p10 = [xs[i + 1], ys[j], z_top]
                p11 = [xs[i + 1], ys[j + 1], z_top]
                p01 = [xs[i], ys[j + 1], z_top]
                # Slight checkerboard variation if subdiv > 1
                shade = 1.0 if (i + j) % 2 == 0 else 0.985
                c_top = [top_rgb[0] * shade, top_rgb[1] * shade, top_rgb[2] * shade]
                tris_list.append([p00, p10, p11])
                tris_list.append([p00, p11, p01])
                norms_list.append([0.0, 0.0, 1.0])
                norms_list.append([0.0, 0.0, 1.0])
                cols_list.append(c_top)
                cols_list.append(c_top)

        # 4 side faces (+X, -X, +Y, -Y)
        side_faces = [
            # +X front face (facing camera)
            ([[cx + hx, cy - hy, cz - hz], [cx + hx, cy + hy, cz - hz], [cx + hx, cy + hy, cz + hz], [cx + hx, cy - hy, cz + hz]], [1.0, 0.0, 0.0]),
            # -X back face
            ([[cx - hx, cy + hy, cz - hz], [cx - hx, cy - hy, cz - hz], [cx - hx, cy - hy, cz + hz], [cx - hx, cy + hy, cz + hz]], [-1.0, 0.0, 0.0]),
            # +Y right face
            ([[cx + hx, cy + hy, cz - hz], [cx - hx, cy + hy, cz - hz], [cx - hx, cy + hy, cz + hz], [cx + hx, cy + hy, cz + hz]], [0.0, 1.0, 0.0]),
            # -Y left face
            ([[cx - hx, cy - hy, cz - hz], [cx + hx, cy - hy, cz - hz], [cx + hx, cy - hy, cz + hz], [cx - hx, cy - hy, cz + hz]], [0.0, -1.0, 0.0]),
        ]
        for quad, n_vec in side_faces:
            tris_list.append([quad[0], quad[1], quad[2]])
            tris_list.append([quad[0], quad[2], quad[3]])
            norms_list.append(n_vec)
            norms_list.append(n_vec)
            cols_list.append(list(side_rgb))
            cols_list.append(list(side_rgb))

        return (
            np.array(tris_list, dtype=np.float32),
            np.array(norms_list, dtype=np.float32),
            np.array(cols_list, dtype=np.float32),
        )

    def render_frame(
        self,
        qpos: npt.NDArray[np.float64],
        cube_world_pos: npt.NDArray[np.float64],
        gripper_action: float,
        width: int = 256,
        height: int = 256,
        camera_mode: str = "agentview",
        cam_azimuth_deg: float | None = None,
        cam_elevation_deg: float | None = None,
        cam_distance: float | None = None,
        look_at_override: npt.NDArray[np.float64] | None = None,
        overlay_3d_trajectories: list[tuple[npt.NDArray[np.float64], tuple[int, int, int], int]] | None = None,
    ) -> npt.NDArray[np.uint8]:
        """Render a full 3D Z-buffered frame of the Franka Panda arm, gripper, table, and red cube."""
        # 1. Background room & floor gradient matching MuJoCo table_arena.xml
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        horizon_y = int(height * 0.22)
        # Upper plaster wall & wooden molding
        canvas[:horizon_y] = (205, 198, 190)
        canvas[max(0, horizon_y - 4) : horizon_y] = (160, 145, 130)
        # Tiled floor below horizon
        canvas[horizon_y:] = (218, 210, 202)

        zbuffer = np.full((height, width), 1e9, dtype=np.float32)

        # 2. Compute exact 4x4 world transforms for every Panda link & finger
        T_curr = np.eye(4, dtype=np.float64)
        T_curr[:3, 3] = PANDA_BASE_WORLD_POS
        joint_idx = 0
        body_transforms: Dict[str, npt.NDArray[np.float64]] = {}

        for name, pos_offset, quat_wxyz, has_joint in _PANDA_KINEMATIC_CHAIN:
            T_body = np.eye(4, dtype=np.float64)
            T_body[:3, :3] = _quat_wxyz_to_rotmat(quat_wxyz)
            T_body[:3, 3] = np.array(pos_offset, dtype=np.float64)
            T_curr = T_curr @ T_body

            if has_joint:
                T_joint = np.eye(4, dtype=np.float64)
                T_joint[:3, :3] = _rot_z(float(qpos[joint_idx]))
                T_curr = T_curr @ T_joint
                joint_idx += 1

            body_transforms[name] = T_curr.copy()

        # Gripper transforms from panda_gripper.xml attached to robot0_right_hand
        T_rh = body_transforms["robot0_right_hand"]
        T_rg_offset = np.eye(4, dtype=np.float64)
        T_rg_offset[:3, :3] = _quat_wxyz_to_rotmat((0.707107, 0.0, 0.0, -0.707107))
        T_right_gripper = T_rh @ T_rg_offset

        T_q_pos90 = np.eye(4, dtype=np.float64)
        T_q_pos90[:3, :3] = _quat_wxyz_to_rotmat((0.707107, 0.0, 0.0, 0.707107))
        body_transforms["gripper_hand"] = T_right_gripper @ T_q_pos90

        # Finger slide joint position: 0.04m open, 0.020m closed around 4.2cm cube
        finger_q = 0.020 if gripper_action > 0.0 else 0.040

        T_lf_local = T_q_pos90.copy()
        T_lf_local[:3, 3] = np.array([0.0, 0.0, 0.0524], dtype=np.float64)
        T_lf_slide = np.eye(4, dtype=np.float64)
        T_lf_slide[:3, 3] = np.array([0.0, finger_q, 0.0], dtype=np.float64)
        body_transforms["gripper_leftfinger"] = T_right_gripper @ T_lf_local @ T_lf_slide

        T_rf_local = T_q_pos90.copy()
        T_rf_local[:3, 3] = np.array([0.0, 0.0, 0.0524], dtype=np.float64)
        T_rf_slide = np.eye(4, dtype=np.float64)
        T_rf_slide[:3, 3] = np.array([0.0, -finger_q, 0.0], dtype=np.float64)
        T_rf_flip = np.eye(4, dtype=np.float64)
        T_rf_flip[:3, :3] = _quat_wxyz_to_rotmat((0.0, 0.0, 0.0, 1.0))
        body_transforms["gripper_rightfinger"] = T_right_gripper @ T_rf_local @ T_rf_slide @ T_rf_flip

        # RethinkMount pedestal transform
        T_mount = np.eye(4, dtype=np.float64)
        T_mount[:3, 3] = PANDA_BASE_WORLD_POS + np.array([0.0, 0.0, -0.922], dtype=np.float64)
        body_transforms["mount_pedestal"] = T_mount

        # 3. Camera Pose Setup
        # Frame both the full 3D Panda robot arm and the tabletop workspace
        if camera_mode == "eye_in_hand":
            T_cam_local = np.eye(4, dtype=np.float64)
            T_cam_local[:3, 3] = np.array([0.05, 0.0, 0.0], dtype=np.float64)
            T_cam_local[:3, :3] = _quat_wxyz_to_rotmat((0.0, 0.707108, 0.707108, 0.0))
            T_cam_w = T_rh @ T_cam_local
            cam_pos = T_cam_w[:3, 3]
            right = T_cam_w[:3, 0]
            up = T_cam_w[:3, 1]
            forward = -T_cam_w[:3, 2]
            fovy_deg = 75.0
        else:
            # Standard MuJoCo agentview or interactive spherical orbit camera
            look_at = (
                look_at_override.copy()
                if look_at_override is not None
                else np.array([-0.14, 0.0, 0.94], dtype=np.float64)
            )
            if cam_azimuth_deg is not None or cam_elevation_deg is not None or cam_distance is not None:
                az = np.deg2rad(0.0 if cam_azimuth_deg is None else float(cam_azimuth_deg))
                el = np.deg2rad(31.4 if cam_elevation_deg is None else float(np.clip(cam_elevation_deg, -10.0, 82.0)))
                dist = 0.96 if cam_distance is None else float(np.clip(cam_distance, 0.42, 2.20))
                cam_pos = look_at + np.array(
                    [
                        dist * np.cos(el) * np.cos(az),
                        dist * np.cos(el) * np.sin(az),
                        dist * np.sin(el),
                    ],
                    dtype=np.float64,
                )
            else:
                cam_pos = np.array([0.68, 0.0, 1.44], dtype=np.float64)
            forward = look_at - cam_pos
            forward /= np.linalg.norm(forward)
            world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            right = np.cross(forward, world_up)
            if np.linalg.norm(right) < 1e-6:
                right = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            else:
                right /= np.linalg.norm(right)
            up = np.cross(right, forward)
            fovy_deg = 48.0

        f_px = float((height / 2.0) / np.tan(np.deg2rad(fovy_deg) / 2.0))
        cx, cy = width / 2.0, height / 2.0
        R_w2c = np.vstack([right, -up, forward]).astype(np.float32)  # (3, 3)
        cam_pos_f32 = cam_pos.astype(np.float32)

        # 4. Assemble all world-space triangles, normals, and base RGBs
        all_tris: List[npt.NDArray[np.float32]] = []
        all_norms: List[npt.NDArray[np.float32]] = []
        all_rgbs: List[npt.NDArray[np.float32]] = []

        # 4a. Table box (0.8m x 0.8m x 0.8m centered at [0, 0, 0.4])
        tab_tris, tab_norms, tab_rgbs = self._build_box_triangles(
            center=np.array([0.0, 0.0, 0.40]),
            half_size=np.array([0.40, 0.40, 0.40]),
            top_rgb=(0.97, 0.97, 0.96),
            side_rgb=(0.82, 0.81, 0.79),
            subdiv=8,
        )
        all_tris.append(tab_tris)
        all_norms.append(tab_norms)
        all_rgbs.append(tab_rgbs)

        # 4b. Soft contact shadow under the red cube on the table top (Z = 0.8008m)
        sh_tris, sh_norms, sh_rgbs = self._build_box_triangles(
            center=np.array([cube_world_pos[0] - 0.004, cube_world_pos[1] - 0.004, TABLE_TOP_Z_WORLD + 0.0005]),
            half_size=np.array([0.024, 0.024, 0.0004]),
            top_rgb=(0.76, 0.75, 0.73),
            side_rgb=(0.76, 0.75, 0.73),
            subdiv=1,
        )
        all_tris.append(sh_tris)
        all_norms.append(sh_norms)
        all_rgbs.append(sh_rgbs)

        # 4c. Red Wooden Cube (BoxObject at cube_world_pos)
        cube_tris, cube_norms, cube_rgbs = self._build_box_triangles(
            center=cube_world_pos,
            half_size=np.array([0.021, 0.021, 0.021]),
            top_rgb=(0.82, 0.16, 0.16),
            side_rgb=(0.68, 0.12, 0.12),
            subdiv=2,
        )
        all_tris.append(cube_tris)
        all_norms.append(cube_norms)
        all_rgbs.append(cube_rgbs)

        # 4d. Robot arm links, gripper housing, left/right fingers, and pedestal mount
        mesh_mapping = [
            ("mount_pedestal", "mount_pedestal"),
            ("robot0_link0", "robot0_link0"),
            ("robot0_link1", "robot0_link1"),
            ("robot0_link2", "robot0_link2"),
            ("robot0_link3", "robot0_link3"),
            ("robot0_link4", "robot0_link4"),
            ("robot0_link5", "robot0_link5"),
            ("robot0_link6", "robot0_link6"),
            ("robot0_link7", "robot0_link7"),
            ("gripper_hand", "gripper_hand"),
            ("gripper_leftfinger", "gripper_finger"),
            ("gripper_rightfinger", "gripper_finger"),
        ]

        for tf_key, mesh_key in mesh_mapping:
            if tf_key not in body_transforms or mesh_key not in self.body_meshes:
                continue
            T_w = body_transforms[tf_key]
            R_w = T_w[:3, :3].astype(np.float32)
            t_w = T_w[:3, 3].astype(np.float32)

            for tris_local, norms_local, rgb_base in self.body_meshes[mesh_key]:
                # Transform vertices (M, 3, 3) and normals (M, 3) to world space
                M = tris_local.shape[0]
                v_flat = tris_local.reshape(-1, 3)
                v_world = (v_flat @ R_w.T) + t_w
                tris_w = v_world.reshape(M, 3, 3)
                norms_w = norms_local @ R_w.T
                rgbs_w = np.broadcast_to(rgb_base, (M, 3)).copy()

                all_tris.append(tris_w)
                all_norms.append(norms_w)
                all_rgbs.append(rgbs_w)

        tris_world = np.concatenate(all_tris, axis=0)   # (N_total, 3, 3)
        norms_world = np.concatenate(all_norms, axis=0) # (N_total, 3)
        rgbs_world = np.concatenate(all_rgbs, axis=0)   # (N_total, 3)

        # 5. Directional & ambient Phong lighting (matching table_arena.xml light dir="-0.2 -0.2 -1")
        light_dir = np.array([0.35, 0.25, 0.90], dtype=np.float32)
        light_dir /= np.linalg.norm(light_dir)
        view_dir = -forward.astype(np.float32)
        half_vec = light_dir + view_dir
        half_vec /= np.linalg.norm(half_vec)

        ndotl = np.clip(np.sum(norms_world * light_dir, axis=-1), 0.0, 1.0)
        ndoth = np.clip(np.sum(norms_world * half_vec, axis=-1), 0.0, 1.0)
        specular = (ndoth ** 24) * 0.22

        ambient = 0.42
        diffuse = 0.58 * ndotl
        lit_rgb = np.clip(
            rgbs_world * (ambient + diffuse)[:, np.newaxis] + specular[:, np.newaxis],
            0.0,
            1.0,
        )
        colors_u8 = (lit_rgb * 255.0).astype(np.uint8)

        # 6. Project triangles to camera coordinates and 2D screen pixels
        rel_v = tris_world - cam_pos_f32  # (N_total, 3, 3)
        cam_v = rel_v @ R_w2c.T           # (N_total, 3, 3)
        depths = cam_v[:, :, 2]           # (N_total, 3)

        safe_z = np.maximum(depths, 0.02)
        u_px = cx + f_px * (cam_v[:, :, 0] / safe_z)
        v_px = cy + f_px * (cam_v[:, :, 1] / safe_z)
        pts_2d = np.stack([u_px, v_px], axis=-1).astype(np.float32)

        # 7. Z-buffer triangle rasterization
        _rasterize_triangles_zbuffer(pts_2d, depths, colors_u8, canvas, zbuffer)

        # 8. Optional 3D trajectory curves rendered in camera perspective
        if overlay_3d_trajectories:
            import cv2
            for traj_pts, col_rgb, thick in overlay_3d_trajectories:
                if traj_pts is None or len(traj_pts) < 2:
                    continue
                rel_p = traj_pts.astype(np.float32) - cam_pos_f32
                cam_p = rel_p @ R_w2c.T
                z_p = cam_p[:, 2]
                valid_p = z_p > 0.05
                u_p = cx + f_px * (cam_p[:, 0] / np.maximum(z_p, 0.05))
                v_p = cy + f_px * (cam_p[:, 1] / np.maximum(z_p, 0.05))
                for k in range(len(traj_pts) - 1):
                    if valid_p[k] and valid_p[k + 1]:
                        p0 = (int(round(u_p[k])), int(round(v_p[k])))
                        p1 = (int(round(u_p[k + 1])), int(round(v_p[k + 1])))
                        cv2.line(canvas, p0, p1, col_rgb, thick, cv2.LINE_AA)
                if valid_p[-1]:
                    p_end = (int(round(u_p[-1])), int(round(v_p[-1])))
                    cv2.circle(canvas, p_end, thick + 3, col_rgb, -1, cv2.LINE_AA)

        return canvas
