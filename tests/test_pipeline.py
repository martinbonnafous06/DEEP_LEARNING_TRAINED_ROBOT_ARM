"""tests/test_pipeline.py — Unit and integration tests for the Video-to-LIBERO pipeline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.utils.coordinates import (
    adapt_camera_intrinsics,
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

    def test_adapt_camera_intrinsics_portrait(self) -> None:
        # Landscape 4000x2252 calibration
        orig_w, orig_h = 4000, 2252
        fx, fy = 3080.0, 3178.0
        cx, cy = 1717.0, 1781.0

        # Video is portrait 1080x1920
        new_fx, new_fy, new_cx, new_cy = adapt_camera_intrinsics(
            fx, fy, cx, cy, orig_w, orig_h, 1080, 1920
        )

        # In portrait: focal lengths should be swapped and scaled to maintain square pixels
        # new_fx corresponds to sensor short side (target_w / orig_h = 1080/2252)
        # new_fy corresponds to sensor long side (target_h / orig_w = 1920/4000)
        self.assertAlmostEqual(new_fx, fy * (1080 / 2252), places=1)
        self.assertAlmostEqual(new_fy, fx * (1920 / 4000), places=1)
        # Square pixel ratio should be preserved (~1.0)
        ratio = new_fx / new_fy
        self.assertTrue(0.95 < ratio < 1.08, f"Pixel aspect ratio distorted: {ratio}")
        self.assertEqual(new_cx, 540.0)
        self.assertEqual(new_cy, 960.0)

    def test_adapt_camera_intrinsics_same_orientation(self) -> None:
        orig_w, orig_h = 4000, 2252
        fx, fy = 3000.0, 3000.0
        cx, cy = 2000.0, 1126.0

        # Target is widescreen 1920x1080
        new_fx, new_fy, new_cx, new_cy = adapt_camera_intrinsics(
            fx, fy, cx, cy, orig_w, orig_h, 1920, 1080
        )
        self.assertAlmostEqual(new_fx, fx * (1920 / 4000), places=2)
        self.assertAlmostEqual(new_fy, fy * (1080 / 2252), places=2)



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

    def test_simulation_grasping_and_lifting(self) -> None:
        from src.retargeting.sim_replay import (
            PANDA_INIT_QPOS,
            execute_panda_manipulation_in_sim,
            panda_forward_kinematics,
        )

        # Verify forward kinematics at home configuration matches Robosuite Lift reset pose
        links, eef_pos, _ = panda_forward_kinematics(PANDA_INIT_QPOS)
        self.assertIn("robot0_link0", links)
        self.assertIn("robot0_right_hand", links)
        self.assertTrue(0.95 < eef_pos[2] < 1.10)

        # Synthesize a 60-step pick-lift-place trajectory
        N = 60
        demo_eef = np.zeros((N, 3), dtype=np.float32)
        demo_eef[:, 0] = np.linspace(0.28, 0.50, N)
        demo_eef[:, 1] = np.linspace(-0.10, 0.10, N)
        demo_eef[:, 2] = -0.13 + 0.12 * np.sin(np.linspace(0, np.pi, N))

        actions = np.zeros((N, 7), dtype=np.float32)
        actions[:-1, :3] = np.diff(demo_eef, axis=0)
        actions[:, 6] = -1.0
        actions[15:45, 6] = 1.0  # Pinch/grasp between step 15 and 45

        res = execute_panda_manipulation_in_sim(
            actions=actions,
            demo_eef_pos=demo_eef,
            env_name="Lift",
            render_video=True,
            camera_height=128,
            camera_width=128,
        )
        self.assertTrue(res.grasped, "Panda arm should grasp the object during the pinch phase")
        self.assertTrue(res.task_success, "Panda arm should lift the grasped cube > 4 cm above the table")
        self.assertGreater(res.max_cube_lift_m, 0.08)
        self.assertGreater(res.total_eef_distance_m, 0.35)
        self.assertEqual(len(res.agentview_frames), N)

    def test_multi_dataset_and_lazy_hdf5_policy_and_world_model(self) -> None:
        """Verify schema auto-discovery, lazy HDF5 streaming, and multi-dataset BC + WM training."""
        import importlib
        import h5py
        import torch

        mod_bc = importlib.import_module("scripts.06_train_policy")
        mod_wm = importlib.import_module("scripts.07_eval_world_model")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            custom_h5 = tmp / "custom_robomimic_dataset.hdf5"

            # Synthesize a non-standard Robomimic-style dataset:
            # - demo keys named 'ep_001' .. 'ep_003' with varying lengths (15, 22, 18)
            # - RGB stored under 'obs/agentview_image' at 84x84 resolution
            # - Proprioception stored under 'obs/ee_pos' (3D) + 'obs/ee_ori' (5D) = 8D total
            # - Actions normalized in [-1, 1] with 6D motion + [0, 1] gripper = 7D total
            with h5py.File(custom_h5, "w") as f:
                data_grp = f.create_group("data")
                for idx, ep_len in enumerate([15, 22, 18], start=1):
                    ep = data_grp.create_group(f"ep_{idx:03d}")
                    obs = ep.create_group("obs")
                    obs.create_dataset(
                        "agentview_image",
                        data=np.random.randint(0, 256, size=(ep_len, 84, 84, 3), dtype=np.uint8),
                    )
                    obs.create_dataset(
                        "ee_pos",
                        data=np.random.randn(ep_len, 3).astype(np.float32) * 0.2,
                    )
                    obs.create_dataset(
                        "ee_ori",
                        data=np.random.randn(ep_len, 5).astype(np.float32),
                    )
                    acts = np.random.uniform(-0.8, 0.8, size=(ep_len, 7)).astype(np.float32)
                    acts[:, 6] = np.where(np.arange(ep_len) > ep_len // 2, 1.0, 0.0)
                    ep.create_dataset("actions", data=acts)

            # 1. Test universal schema discovery
            schema = mod_bc.resolve_hdf5_schema(custom_h5)
            self.assertEqual(schema.rgb_key, "agentview_image")
            self.assertEqual(schema.image_shape, (3, 84, 84))
            self.assertEqual(schema.proprio_dim, 8)
            self.assertEqual(schema.action_dim, 7)
            self.assertEqual(schema.total_frames, 55)
            self.assertAlmostEqual(schema.grip_min, 0.0, places=4)
            self.assertAlmostEqual(schema.grip_max, 1.0, places=4)

            # 2. Test both RAM-cached uint8 and Lazy HDF5 Dataset modes
            ds_ram = mod_bc.LiberoHumanoidDataset(custom_h5, schema.demo_keys, schema=schema, lazy_hdf5=False)
            ds_lazy = mod_bc.LiberoHumanoidDataset(custom_h5, schema.demo_keys, schema=schema, lazy_hdf5=True)
            self.assertEqual(len(ds_ram), 55)
            self.assertEqual(len(ds_lazy), 55)
            self.assertEqual(ds_ram[0]["rgb"].dtype, torch.uint8)
            np.testing.assert_array_equal(ds_ram[10]["rgb"].numpy(), ds_lazy[10]["rgb"].numpy())
            if ds_lazy._h5_file is not None:
                ds_lazy._h5_file.close()

            # 3. Test dataset-adaptive BCPolicy forward pass on 84x84 images and 8D proprioception
            policy = mod_bc.BCPolicy(proprio_dim=schema.proprio_dim, action_dim=schema.action_dim)
            policy.set_dataset_statistics(schema)
            sample_rgb = ds_ram[0]["rgb"].unsqueeze(0)
            sample_prop = ds_ram[0]["proprio"].unsqueeze(0)
            pred_act = policy(sample_rgb, sample_prop)
            self.assertEqual(pred_act.shape, (1, 7))
            self.assertTrue(0.0 <= float(pred_act[0, 6]) <= 1.0)

            # 4. Test vectorized multi-trajectory World Model training on variable-length 8D state trajectories
            trajs = mod_wm.load_trajectories(custom_h5)
            self.assertEqual(len(trajs), 3)
            wm = mod_wm.train_world_model(
                trajectories=trajs,
                train_keys=list(trajs.keys()),
                epochs=2,
                device=torch.device("cpu"),
                traj_batch_size=2,
            )
            self.assertEqual(wm.state_dim, 8)

            # 5. Test non-destructive Per-Dataset Model Registry & Continual Fine-Tuning (Resume)
            ckpt_dir = tmp / "checkpoints"
            milestones_a = []
            mod_bc.train_policy(
                dataset_path=custom_h5,
                output_dir=ckpt_dir,
                epochs=2,
                batch_size=16,
                device_name="cpu",
                log_interval=1,
                epoch_callback=lambda info: milestones_a.append(info["epoch"]),
                model_name="dataset_alpha",
                update_active_default=True,
            )
            alpha_ckpt = ckpt_dir / "registry" / "dataset_alpha" / "best_policy.pt"
            self.assertTrue(alpha_ckpt.is_file())
            self.assertEqual(milestones_a, [1, 2])

            # Train a second dataset model profile ("dataset_beta") fine-tuning from "dataset_alpha"
            mod_bc.train_policy(
                dataset_path=custom_h5,
                output_dir=ckpt_dir,
                epochs=2,
                batch_size=16,
                device_name="cpu",
                log_interval=2,
                model_name="dataset_beta",
                resume_from=alpha_ckpt,
                update_active_default=True,
            )
            beta_ckpt = ckpt_dir / "registry" / "dataset_beta" / "best_policy.pt"
            self.assertTrue(alpha_ckpt.is_file(), "dataset_alpha checkpoint must NOT be destroyed when training dataset_beta")
            self.assertTrue(beta_ckpt.is_file(), "dataset_beta checkpoint must be created in its own registry folder")

            # Re-training dataset_alpha must automatically version the previous checkpoint as best_policy_v1.pt
            mod_bc.train_policy(
                dataset_path=custom_h5,
                output_dir=ckpt_dir,
                epochs=1,
                batch_size=16,
                device_name="cpu",
                model_name="dataset_alpha",
                resume_from=alpha_ckpt,
                update_active_default=False,
            )
            self.assertTrue((ckpt_dir / "registry" / "dataset_alpha" / "best_policy_v1.pt").is_file())


if __name__ == "__main__":
    unittest.main()

