"""tests/test_pipeline.py — Unit and integration tests for the Video-to-LIBERO pipeline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.utils.coordinates import (
    compute_osc_actions,
    compute_table_alignment,
    compose_hand_to_table,
    invert_transform_se3,
    map_table_to_robot_base,
    rodrigues_align_vectors,
    twist_to_matrix,
)
from src.utils.io import (
    load_camera_poses,
    natural_sort_key,
    save_camera_poses,
)


class TestCoordinatesMath(unittest.TestCase):
    """Test geometric math and SE(3) spatial transformations."""

    def test_invert_transform_se3_single(self) -> None:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        T[:3, 3] = np.array([1.5, -2.0, 0.5], dtype=np.float64)

        T_inv = invert_transform_se3(T)
        identity = T @ T_inv
        np.testing.assert_allclose(identity, np.eye(4), atol=1e-12)

    def test_invert_transform_se3_batch(self) -> None:
        N = 10
        T_batch = np.tile(np.eye(4, dtype=np.float64), (N, 1, 1))
        T_batch[:, :3, 3] = np.random.randn(N, 3)

        T_inv_batch = invert_transform_se3(T_batch)
        for i in range(N):
            np.testing.assert_allclose(T_batch[i] @ T_inv_batch[i], np.eye(4), atol=1e-12)

    def test_rodrigues_align_vectors(self) -> None:
        v_from = np.array([0.0, 1.0, 1.0], dtype=np.float64)
        v_to = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        R = rodrigues_align_vectors(v_from, v_to)
        aligned = R @ (v_from / np.linalg.norm(v_from))
        np.testing.assert_allclose(aligned, v_to, atol=1e-12)

    def test_compute_table_alignment(self) -> None:
        # Plane: 0x + 0y + 1z - 0.8 = 0  => z = 0.8
        plane_model = np.array([0.0, 0.0, 1.0, -0.8], dtype=np.float64)
        T_world_to_table, R_align, t_align = compute_table_alignment(plane_model)

        # Arbitrary point on the table surface (z = 0.8)
        p_world = np.array([0.35, -0.15, 0.8, 1.0], dtype=np.float64)
        p_table = T_world_to_table @ p_world

        # Must lie exactly at Z = 0
        self.assertAlmostEqual(p_table[2], 0.0, places=9)

    def test_compute_table_alignment_tilted(self) -> None:
        # Tilted plane: normal [0, 1, 1] / sqrt(2), d = -1.0
        n = np.array([0.0, 1.0, 1.0]) / np.sqrt(2.0)
        plane_model = np.array([n[0], n[1], n[2], -1.0], dtype=np.float64)
        T_world_to_table, _, _ = compute_table_alignment(plane_model)

        # Point on plane: p = -d * n = 1.0 * n
        p_world = np.array([n[0], n[1], n[2], 1.0], dtype=np.float64)
        p_table = T_world_to_table @ p_world
        self.assertAlmostEqual(p_table[2], 0.0, places=7)

    def test_compose_hand_to_table(self) -> None:
        N = 5
        # Camera is translated by [0, 0, 1.0] in world: p_cam = p_world - [0, 0, 1.0]
        # => T_world_to_cam has t = [0, 0, -1.0]
        T_world_to_cam = np.tile(np.eye(4, dtype=np.float64), (N, 1, 1))
        T_world_to_cam[:, 2, 3] = -1.0

        # Hand is at [0.1, 0.2, 0.5] relative to camera
        T_cam_to_hand = np.tile(np.eye(4, dtype=np.float64), (N, 1, 1))
        T_cam_to_hand[:, 0, 3] = 0.1
        T_cam_to_hand[:, 1, 3] = 0.2
        T_cam_to_hand[:, 2, 3] = 0.5

        # Table surface is at world Z = 0.8: T_world_to_table shifts Z by -0.8
        T_world_to_table = np.eye(4, dtype=np.float64)
        T_world_to_table[2, 3] = -0.8

        T_table_to_hand = compose_hand_to_table(T_cam_to_hand, T_world_to_cam, T_world_to_table)

        # In world: camera optical centre is at Z = 1.0.
        # Hand relative to cam is Z = +0.5 => hand in world is Z = 1.0 + 0.5 = 1.5.
        # Relative to table (Z = 0.8): hand in table is Z = 1.5 - 0.8 = 0.7.
        expected_z = 0.7
        np.testing.assert_allclose(T_table_to_hand[:, 2, 3], expected_z, atol=1e-12)

    def test_compute_osc_actions_gripper_mapping(self) -> None:
        M = 4
        T_smooth = np.tile(np.eye(4, dtype=np.float64), (M, 1, 1))
        # Moving along X: 0.0, 0.05, 0.10, 0.15
        T_smooth[:, 0, 3] = np.arange(M) * 0.05

        # Aperture: 0.0 (closed) -> 1.0 (open)
        gripper_aperture = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float64)

        actions = compute_osc_actions(T_smooth, gripper_aperture, gripper_is_aperture=True)
        self.assertEqual(actions.shape, (M - 1, 7))

        # Check translation deltas: dx = 0.05
        np.testing.assert_allclose(actions[:, 0], 0.05, atol=1e-7)

        # Check gripper action mapping:
        # aperture 0.0 -> +1.0 (closed)
        # aperture 1.0 -> -1.0 (open)
        self.assertAlmostEqual(actions[0, 6], 1.0, places=5)
        self.assertAlmostEqual(actions[1, 6], -1.0, places=5)


class TestIOMisc(unittest.TestCase):
    """Test I/O utilities and natural sorting."""

    def test_natural_sort_key(self) -> None:
        filenames = ["frame_10.png", "frame_2.png", "frame_1.png", "FRAME_20.PNG"]
        sorted_files = sorted(filenames, key=natural_sort_key)
        self.assertEqual(sorted_files, ["frame_1.png", "frame_2.png", "frame_10.png", "FRAME_20.PNG"])

    def test_load_and_save_camera_poses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            N = 8
            poses = np.tile(np.eye(4, dtype=np.float64), (N, 1, 1))
            poses[:, :3, 3] = np.random.randn(N, 3)

            npy_p, npz_p = save_camera_poses(tmp_dir, poses)
            self.assertTrue(npy_p.is_file())
            self.assertTrue(npz_p.is_file())

            loaded_npy = load_camera_poses(npy_p)
            loaded_npz = load_camera_poses(npz_p)

            np.testing.assert_allclose(loaded_npy, poses)
            np.testing.assert_allclose(loaded_npz, poses)


class TestEndToEndPipeline(unittest.TestCase):
    """Test full pipeline integration from retargeting to LIBERO HDF5 export."""

    def test_pipeline_retargeting_and_libero_export(self) -> None:
        import h5py
        from src.retargeting.libero_exporter import export_libero_dataset
        from src.retargeting.transformer import process_kinematics_retargeting

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)

            # 1. Synthesize Stage 2 outputs
            N = 30  # 1 second of video at 30 fps
            poses = np.tile(np.eye(4, dtype=np.float64), (N, 1, 1))
            poses[:, 2, 3] = -1.0  # Camera at Z = 1.0 in world

            cam_poses_path = tmp / "camera_poses.npz"
            np.savez_compressed(cam_poses_path, T_world_to_cam=poses, poses=poses)

            table_transform_path = tmp / "table_transform.npz"
            R_align = np.eye(3, dtype=np.float64)
            t_align = np.array([0.0, 0.0, -0.8], dtype=np.float64)
            T_w2t = np.eye(4, dtype=np.float64)
            T_w2t[:3, :3] = R_align
            T_w2t[:3, 3] = t_align
            np.savez_compressed(
                table_transform_path,
                T_world_to_table=T_w2t,
                R_align=R_align,
                t_align=t_align,
                plane_model=np.array([0.0, 0.0, 1.0, -0.8]),
            )

            # 2. Synthesize Stage 3 outputs
            hand_poses_path = tmp / "hand_poses.npz"
            T_cam_to_hand = np.tile(np.eye(4, dtype=np.float64), (N, 1, 1))
            # Smooth hand motion in camera frame
            T_cam_to_hand[:, 0, 3] = np.linspace(-0.1, 0.1, N)
            T_cam_to_hand[:, 1, 3] = np.linspace(0.0, 0.2, N)
            T_cam_to_hand[:, 2, 3] = 0.5
            gripper_cont = np.linspace(1.0, 0.0, N)  # opens then closes
            gripper_bin = np.where(gripper_cont < 0.5, 1.0, -1.0)
            keypoints = np.zeros((N, 21, 3), dtype=np.float64)

            np.savez_compressed(
                hand_poses_path,
                T_cam_to_hand=T_cam_to_hand,
                gripper_continuous=gripper_cont,
                gripper_binary=gripper_bin,
                keypoints_3d=keypoints,
            )

            # 3. Run Stage 4 Retargeting
            retargeted_out = tmp / "retargeted"
            retargeted_npz = process_kinematics_retargeting(
                camera_poses=cam_poses_path,
                table_transform=table_transform_path,
                hand_poses=hand_poses_path,
                output_dir=retargeted_out,
                src_fps=30.0,
                tgt_fps=20.0,
                window_length=5,
                polyorder=2,
            )
            self.assertTrue(retargeted_npz.is_file())

            # Verify retargeted.npz content
            with np.load(retargeted_npz) as data:
                self.assertIn("actions", data)
                self.assertIn("ee_pos", data)
                self.assertIn("ee_quat", data)

                n_samples = len(data["actions"])
                self.assertEqual(len(data["ee_pos"]), n_samples)
                self.assertEqual(len(data["ee_quat"]), n_samples)
                self.assertEqual(data["actions"].shape[1], 7)

            # Verify Stage 4 deliverable trajectory_world_20hz.npy
            traj_20hz_path = retargeted_out / "trajectory_world_20hz.npy"
            self.assertTrue(traj_20hz_path.is_file())

            # 4. Run Stage 5 LIBERO Dataset Export
            out_hdf5 = tmp / "demo_libero.hdf5"
            export_libero_dataset(
                retargeted_npz=retargeted_npz,
                output_hdf5=out_hdf5,
                demo_name="demo_0",
                env_name="LIBERO_OBJECT",
                frame_height=84,
                frame_width=84,
            )
            self.assertTrue(out_hdf5.is_file())

            # 5. Inspect generated HDF5 file
            with h5py.File(out_hdf5, "r") as f:
                self.assertIn("data", f)
                self.assertEqual(f["data"].attrs["env_name"], "LIBERO_OBJECT")
                self.assertIn("demo_0", f["data"])

                demo = f["data/demo_0"]
                self.assertEqual(demo.attrs["num_samples"], n_samples)
                self.assertEqual(demo["actions"].shape, (n_samples, 7))
                self.assertEqual(demo["rewards"].shape, (n_samples,))
                self.assertEqual(demo["dones"].shape, (n_samples,))
                self.assertEqual(demo["dones"][-1], 1.0)

                obs = demo["obs"]
                self.assertEqual(obs["robot0_eef_pos"].shape, (n_samples, 3))
                self.assertEqual(obs["robot0_eef_quat"].shape, (n_samples, 4))
                self.assertEqual(obs["agentview_rgb"].shape, (n_samples, 84, 84, 3))
                self.assertEqual(obs["robot0_eye_in_hand_rgb"].shape, (n_samples, 84, 84, 3))


if __name__ == "__main__":
    unittest.main()

