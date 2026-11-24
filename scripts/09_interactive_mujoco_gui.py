"""scripts/09_interactive_mujoco_gui.py — Interactive LIBERO / MuJoCo 3D Franka Panda & World Model Tester GUI.

Opens an interactive graphical window allowing you to test and play with:
  1. [Mode 1: World Model Closed-Loop Rollout]:
     Watch the trained Action-Conditioned World Model (`world_model_checkpoint.pt`) predict the
     3D trajectory in closed-loop and drive the 3D Franka Emika Panda arm to grasp, lift, and place
     the red cube, synchronized with the real human demonstration video.
  2. [Mode 2: Visuomotor BC Policy Rollout]:
     Run the trained ResNet-18 + Proprioception Behavioral Cloning Policy (`best_policy.pt`) live.
  3. [Mode 3: Interactive Manual Control + Live World Model Future Horizon Prediction]:
     Drive the 7-DOF Franka Emika Panda arm yourself with keyboard keys (W/A/S/D/Q/E + SPACE to grasp)
     while the World Model predicts your next 15 future 3D states live in the 3D MuJoCo viewport!

Interactive Controls (Mouse & Keyboard):
  - Mouse Left-Drag on 3D View : Orbit 3D Camera 360 deg around the Franka Panda arm & table
  - Mouse Wheel / [+] / [-]    : Zoom 3D Camera In / Out
  - Click On-Screen Buttons    : Switch Modes, Pause/Play, Next/Prev Demo, Reset Cube
  - [1] / [2] / [3]            : Switch between (1) World Model, (2) BC Policy, (3) Manual Control
  - [N] / [P]                  : Next / Previous Demonstration (demo_0 .. demo_17)
  - [SPACE]                    : Pause/Play (in Modes 1 & 2) or Toggle Gripper Grasp (in Mode 3)
  - [W/S], [A/D], [Q/E]        : Move Panda EEF Forward/Back (X), Left/Right (Y), Up/Down (Z) in Mode 3
  - [C]                        : Move Red Cube to a new random tabletop position
  - [R]                        : Reset current episode & camera view
  - [ESC]                      : Close window
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from torchvision import transforms

import importlib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_mod_bc = importlib.import_module("scripts.06_train_policy")
_mod_wm = importlib.import_module("scripts.07_eval_world_model")
BCPolicy = _mod_bc.BCPolicy
ActionConditionedDynamicsModel = _mod_wm.ActionConditionedDynamicsModel
_draw_live_wm_dashboard = _mod_wm._draw_live_wm_dashboard
from src.retargeting.panda_3d_renderer import PandaMuJoCo3DRenderer
from src.retargeting.sim_replay import (
    build_object_aligned_target_trajectory,
    detect_manipulation_phases,
    execute_panda_manipulation_in_sim,
    panda_forward_kinematics,
    panda_inverse_kinematics_step,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("interactive_mujoco")


class InteractiveLiberoMuJoCoApp:
    """Interactive 3D Franka Emika Panda MuJoCo Simulator & World Model / BC Policy Tester."""

    WINDOW_NAME = "LIBERO / MuJoCo 3D Franka Panda - Interactive World Model & Policy Tester"

    def __init__(
        self,
        dataset_path: Path = ROOT / "data/datasets/demo_libero.hdf5",
        wm_ckpt_path: Path = ROOT / "experiments/world_model/world_model_checkpoint.pt",
        bc_ckpt_path: Path = ROOT / "experiments/checkpoints/best_policy.pt",
        start_demo_idx: int = 8,
    ) -> None:
        self.dataset_path = dataset_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.renderer = PandaMuJoCo3DRenderer()

        # Load all 18 demos from HDF5
        self.demos: list[dict] = []
        with h5py.File(self.dataset_path, "r") as f:
            keys = sorted(list(f["data"].keys()), key=lambda k: int(k.replace("demo_", "")))
            for k in keys:
                grp = f["data"][k]
                pos = grp["obs"]["robot0_eef_pos"][:].astype(np.float32)
                quat = grp["obs"]["robot0_eef_quat"][:].astype(np.float32)
                acts = grp["actions"][:].astype(np.float32)
                rgb = grp["obs"]["agentview_rgb"][:] if "agentview_rgb" in grp["obs"] else None
                self.demos.append(
                    {
                        "key": k,
                        "id": f"demo_{int(k.replace('demo_', '')):04d}",
                        "states": np.concatenate([pos, quat], axis=-1),
                        "actions": acts,
                        "rgb": rgb,
                    }
                )

        # Load Trained World Model
        self.wm_model = ActionConditionedDynamicsModel().to(self.device)
        if wm_ckpt_path.exists():
            ckpt = torch.load(wm_ckpt_path, map_location=self.device, weights_only=False)
            self.wm_model.load_state_dict(ckpt["model_state_dict"])
            logger.info("Loaded trained World Model checkpoint: %s", wm_ckpt_path)
        self.wm_model.eval()

        # Load Trained Visuomotor BC Policy
        self.bc_model = BCPolicy(proprio_dim=7, action_dim=7).to(self.device)
        if bc_ckpt_path.exists():
            ckpt_bc = torch.load(bc_ckpt_path, map_location=self.device, weights_only=False)
            self.bc_model.load_state_dict(ckpt_bc["model_state_dict"])
            logger.info("Loaded trained Visuomotor BC Policy checkpoint: %s", bc_ckpt_path)
        self.bc_model.eval()
        self.img_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # Interactive Camera State (360 deg orbit + zoom)
        self.cam_azimuth = 0.0
        self.cam_elevation = 31.4
        self.cam_distance = 0.96
        self.mouse_dragging = False
        self.last_mouse_xy = (0, 0)

        # Mode & Episode State
        # Mode 1: World Model Closed-Loop Rollout | Mode 2: Visuomotor BC Policy | Mode 3: Manual Control + Live WM
        self.mode = 1
        self.paused = False
        self.demo_idx = min(max(0, start_demo_idx), len(self.demos) - 1)
        self.step_idx = 0
        self.show_3d_trails = True

        # Clickable UI buttons: (x0, y0, x1, y1, label, callback_id)
        self.buttons: list[tuple[int, int, int, int, str, str]] = [
            (780, 12, 935, 48, "[1] World Model", "mode_1"),
            (945, 12, 1085, 48, "[2] BC Policy", "mode_2"),
            (1095, 12, 1265, 48, "[3] Manual + WM", "mode_3"),
            (1275, 12, 1425, 48, "[Space] Pause", "toggle_pause"),
            (780, 56, 910, 88, "[P] < Prev Demo", "prev_demo"),
            (920, 56, 1055, 88, "[N] Next Demo >", "next_demo"),
            (1065, 56, 1235, 88, "[C] Move Cube", "move_cube"),
            (1245, 56, 1425, 88, "[R] Reset View", "reset"),
        ]

        self._load_demo_episode(self.demo_idx)

    def _load_demo_episode(self, idx: int, custom_cube_pos: np.ndarray | None = None) -> None:
        self.demo_idx = idx % len(self.demos)
        demo = self.demos[self.demo_idx]
        self.curr_demo_key = demo["key"]
        self.curr_demo_id = demo["id"]
        self.gt_states = demo["states"]
        self.gt_actions = demo["actions"]
        self.T = len(self.gt_actions)
        self.step_idx = 0

        # 1. Run Closed-Loop World Model Rollout from s_0
        with torch.no_grad():
            curr_s = torch.from_numpy(self.gt_states[0:1]).to(self.device)
            h = None
            wm_rollout = [curr_s.cpu().numpy()[0]]
            for t in range(self.T - 1):
                act_t = torch.from_numpy(self.gt_actions[t : t + 1]).to(self.device)
                next_s, h = self.wm_model.forward_step(curr_s, act_t, h)
                wm_rollout.append(next_s.cpu().numpy()[0])
                curr_s = next_s
        self.wm_states = np.array(wm_rollout, dtype=np.float32)
        self.wm_err_cm = np.linalg.norm(self.wm_states[:, :3] - self.gt_states[:, :3], axis=-1) * 100.0

        # 2. Run Visuomotor BC Policy Action Predictions + World Model Rollout
        bc_actions = self.gt_actions.copy()
        if demo["rgb"] is not None:
            with torch.no_grad():
                for t in range(self.T):
                    img_t = self.img_transform(demo["rgb"][t]).unsqueeze(0).to(self.device)
                    prop_t = torch.from_numpy(self.gt_states[t : t + 1]).to(self.device)
                    pred_a = self.bc_model(img_t, prop_t).cpu().numpy()[0]
                    bc_actions[t] = pred_a
        self.bc_actions = bc_actions
        with torch.no_grad():
            curr_s = torch.from_numpy(self.gt_states[0:1]).to(self.device)
            h_bc = None
            bc_rollout = [curr_s.cpu().numpy()[0]]
            for t in range(self.T - 1):
                act_t = torch.from_numpy(self.bc_actions[t : t + 1]).to(self.device)
                next_s, h_bc = self.wm_model.forward_step(curr_s, act_t, h_bc)
                bc_rollout.append(next_s.cpu().numpy()[0])
                curr_s = next_s
        self.bc_states = np.array(bc_rollout, dtype=np.float32)

        # 3. Compute 7-DOF Franka Panda simulation joint trajectory & 3D world waypoints
        self.cube_init_pos = (
            custom_cube_pos.copy()
            if custom_cube_pos is not None
            else np.array([0.0, 0.0, 0.821], dtype=np.float64)
        )
        self._recompute_sim_kinematics()

        # 4. Load real video overlay frames for this demo if available
        frames_dir = ROOT / "data/frames" / self.curr_demo_id
        self.real_video_files = sorted(frames_dir.glob("*.png")) + sorted(frames_dir.glob("*.jpg"))

        # 5. Initialize Manual Sandbox State (Mode 3)
        self._reset_manual_state()

    def _recompute_sim_kinematics(self) -> None:
        active_traj = self.wm_states[:, :3] if self.mode == 1 else self.bc_states[:, :3]
        active_acts = self.gt_actions.copy()
        active_acts[:-1, :3] = np.diff(active_traj, axis=0)

        qpos = np.array([0.0, 0.196, 0.0, -2.43, 0.0, 2.62, 0.785], dtype=np.float64)
        _, ee_init, rot_init = panda_forward_kinematics(qpos)
        target_pos, target_grip, _ = build_object_aligned_target_trajectory(
            demo_eef_pos=active_traj,
            demo_actions=active_acts,
            sim_eef_init=ee_init,
            sim_cube_init=self.cube_init_pos,
        )
        self.sim_target_world = target_pos
        self.sim_grip = target_grip

        # Also compute GT world trajectory for 3D curve comparison
        self.gt_target_world, _, _ = build_object_aligned_target_trajectory(
            demo_eef_pos=self.gt_states[:, :3],
            demo_actions=self.gt_actions,
            sim_eef_init=ee_init,
            sim_cube_init=self.cube_init_pos,
        )

        # Solve 7-DOF IK across the trajectory
        self.qpos_seq = []
        self.ee_world_seq = []
        self.cube_world_seq = []
        cube_pos = self.cube_init_pos.copy()
        grasped = False
        grasp_offset = np.zeros(3, dtype=np.float64)

        for t in range(self.T):
            tgt = target_pos[t]
            g_cmd = float(target_grip[t])
            qpos = panda_inverse_kinematics_step(qpos, tgt, rot_init)
            _, ee_pos, _ = panda_forward_kinematics(qpos)
            dist_xy = float(np.linalg.norm(ee_pos[:2] - cube_pos[:2]))
            dist_z = abs(float(ee_pos[2] - cube_pos[2]))
            if g_cmd > 0.0 and not grasped and dist_xy < 0.045 and dist_z < 0.045:
                grasped = True
                grasp_offset = cube_pos - ee_pos
            elif g_cmd < 0.0 and grasped:
                grasped = False
            if grasped:
                cube_pos = ee_pos + grasp_offset
                cube_pos[2] = max(cube_pos[2], 0.821)
            else:
                if cube_pos[2] > 0.821:
                    cube_pos[2] = max(0.821, cube_pos[2] - 0.025)
            self.qpos_seq.append(qpos.copy())
            self.ee_world_seq.append(ee_pos.copy())
            self.cube_world_seq.append(cube_pos.copy())

    def _reset_manual_state(self) -> None:
        self.man_qpos = np.array([0.0, 0.196, 0.0, -2.43, 0.0, 2.62, 0.785], dtype=np.float64)
        _, self.man_ee_pos, self.man_ee_rot = panda_forward_kinematics(self.man_qpos)
        # Start near hover above cube so user can immediately grasp & move
        self.man_target_pos = self.cube_init_pos + np.array([0.0, 0.0, 0.09], dtype=np.float64)
        self.man_qpos = panda_inverse_kinematics_step(self.man_qpos, self.man_target_pos, self.man_ee_rot, max_iters=15)
        _, self.man_ee_pos, _ = panda_forward_kinematics(self.man_qpos)
        self.man_grip = -1.0
        self.man_cube_pos = self.cube_init_pos.copy()
        self.man_grasped = False
        self.man_grasp_offset = np.zeros(3, dtype=np.float64)
        self.man_last_delta = np.array([0.0, 0.0, -0.008], dtype=np.float32)
        self.man_trail: list[np.ndarray] = [self.man_ee_pos.copy()]

    def _predict_manual_wm_horizon(self, horizon: int = 15) -> np.ndarray:
        """Uses the trained World Model to predict the next 15 future 3D states from current manual state & velocity."""
        with torch.no_grad():
            s_curr = torch.tensor(
                [[self.man_ee_pos[0], self.man_ee_pos[1], self.man_ee_pos[2], 1.0, 0.0, 0.0, 0.0]],
                dtype=torch.float32,
                device=self.device,
            )
            a_curr = torch.tensor(
                [[self.man_last_delta[0], self.man_last_delta[1], self.man_last_delta[2], 0.0, 0.0, 0.0, self.man_grip]],
                dtype=torch.float32,
                device=self.device,
            )
            h = None
            preds = [self.man_ee_pos.copy()]
            for _ in range(horizon):
                s_next, h = self.wm_model.forward_step(s_curr, a_curr, h)
                p_next = s_next[0, :3].cpu().numpy().astype(np.float64)
                p_next[2] = max(0.815, p_next[2])
                preds.append(p_next)
                s_curr = s_next
        return np.array(preds, dtype=np.float64)

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if clicked an on-screen button
            for x0, y0, x1, y1, _, cb_id in self.buttons:
                if x0 <= x <= x1 and y0 <= y <= y1:
                    self._trigger_action(cb_id)
                    return
            if x < 760:
                self.mouse_dragging = True
                self.last_mouse_xy = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.mouse_dragging = False
        elif event == cv2.EVENT_MOUSEMOVE and self.mouse_dragging:
            dx = x - self.last_mouse_xy[0]
            dy = y - self.last_mouse_xy[1]
            self.last_mouse_xy = (x, y)
            self.cam_azimuth = (self.cam_azimuth - dx * 0.45) % 360.0
            self.cam_elevation = float(np.clip(self.cam_elevation + dy * 0.35, 2.0, 80.0))
        elif event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                self.cam_distance = float(np.clip(self.cam_distance - 0.06, 0.45, 2.0))
            else:
                self.cam_distance = float(np.clip(self.cam_distance + 0.06, 0.45, 2.0))

    def _trigger_action(self, cb_id: str) -> None:
        if cb_id == "mode_1":
            self.mode = 1
            self._recompute_sim_kinematics()
        elif cb_id == "mode_2":
            self.mode = 2
            self._recompute_sim_kinematics()
        elif cb_id == "mode_3":
            self.mode = 3
            self._reset_manual_state()
        elif cb_id == "toggle_pause":
            if self.mode == 3:
                self.man_grip = -self.man_grip
            else:
                self.paused = not self.paused
        elif cb_id == "prev_demo":
            self._load_demo_episode(self.demo_idx - 1)
        elif cb_id == "next_demo":
            self._load_demo_episode(self.demo_idx + 1)
        elif cb_id == "move_cube":
            rand_xy = np.random.uniform(-0.11, 0.11, size=2)
            new_cube = np.array([rand_xy[0], rand_xy[1], 0.821], dtype=np.float64)
            self._load_demo_episode(self.demo_idx, custom_cube_pos=new_cube)
        elif cb_id == "reset":
            self.cam_azimuth = 0.0
            self.cam_elevation = 31.4
            self.cam_distance = 0.96
            self._load_demo_episode(self.demo_idx)

    def render_gui_frame(self) -> np.ndarray:
        """Assemble the 1440 x 860 interactive GUI canvas."""
        canvas = np.full((860, 1440, 3), 22, dtype=np.uint8)

        if self.mode in (1, 2):
            t = min(self.step_idx, self.T - 1)
            qpos = self.qpos_seq[t]
            cube_pos = self.cube_world_seq[t]
            grip_cmd = float(self.sim_grip[t])
            ee_pos = self.ee_world_seq[t]
            trajs_3d = []
            if self.show_3d_trails:
                trajs_3d = [
                    (self.gt_target_world, (0, 210, 255), 2),               # Cyan: Human Ground Truth 3D Arch
                    (self.sim_target_world[: t + 1], (255, 80, 40), 3),     # Orange/Red: Active World Model / BC Arch
                ]
        else:
            t = min(self.step_idx, self.T - 1)
            qpos = self.man_qpos
            cube_pos = self.man_cube_pos
            grip_cmd = self.man_grip
            ee_pos = self.man_ee_pos
            wm_future_3d = self._predict_manual_wm_horizon(horizon=15)
            trajs_3d = [
                (np.array(self.man_trail[-40:], dtype=np.float64), (80, 230, 100), 2),
                (wm_future_3d, (255, 70, 220), 3),  # Magenta: Live 15-Step Future World Model Rollout
            ]

        # 1. Left Main 3D MuJoCo Viewport (760 x 760)
        sim_rgb = self.renderer.render_frame(
            qpos=qpos,
            cube_world_pos=cube_pos,
            gripper_action=grip_cmd,
            width=380,
            height=380,
            camera_mode="agentview",
            cam_azimuth_deg=self.cam_azimuth,
            cam_elevation_deg=self.cam_elevation,
            cam_distance=self.cam_distance,
            overlay_3d_trajectories=trajs_3d,
        )
        sim_bgr = cv2.cvtColor(sim_rgb, cv2.COLOR_RGB2BGR)
        sim_large = cv2.resize(sim_bgr, (750, 750), interpolation=cv2.INTER_LINEAR)
        canvas[52:802, 12:762] = sim_large

        # Top header above 3D view
        mode_names = {
            1: "MODE 1: WORLD MODEL CLOSED-LOOP ROLLOUT (3D FRANKA PANDA)",
            2: "MODE 2: VISUOMOTOR BC POLICY (RESNET-18 + MLP) ROLLOUT",
            3: "MODE 3: INTERACTIVE MANUAL CONTROL + LIVE WORLD MODEL HORIZON",
        }
        cv2.rectangle(canvas, (12, 10), (762, 50), (45, 65, 95), -1)
        cv2.putText(canvas, mode_names[self.mode], (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 240, 255), 2, cv2.LINE_AA)

        # Overlay 3D Viewport HUD
        cube_lift_cm = max(0.0, (cube_pos[2] - 0.821) * 100.0)
        cv2.rectangle(canvas, (22, 62), (435, 168), (18, 22, 30), -1)
        cv2.putText(
            canvas,
            f"Demo: {self.curr_demo_id} ({self.curr_demo_key}) | Step: {t:02d}/{self.T - 1:02d}",
            (32, 84),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"Panda EEF World [X,Y,Z]: [{ee_pos[0]:+.3f}, {ee_pos[1]:+.3f}, {ee_pos[2]:+.3f}] m",
            (32, 106),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (180, 245, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"Red Cube Lift: {cube_lift_cm:.1f} cm | Gripper: {'CLOSED (GRASP)' if grip_cmd > 0 else 'OPEN'}",
            (32, 128),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (120, 255, 150) if cube_lift_cm > 1.0 else (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"3D Camera Orbit: Az={self.cam_azimuth:.0f} deg, El={self.cam_elevation:.0f} deg, Dist={self.cam_distance:.2f}m (Drag to Rotate)",
            (32, 152),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.41,
            (200, 200, 210),
            1,
            cv2.LINE_AA,
        )

        # Bottom controls bar under 3D viewport
        cv2.rectangle(canvas, (12, 806), (1428, 848), (32, 36, 46), -1)
        ctrl_text = (
            "CONTROLS: [Mouse Drag] Orbit 3D Camera | [Wheel] Zoom | [1] World Model | [2] BC Policy | [3] Manual Control | "
            "[W/A/S/D/Q/E] Move Arm | [SPACE] Pause / Grasp | [N/P] Next/Prev Demo | [C] Move Cube | [ESC] Quit"
        )
        cv2.putText(canvas, ctrl_text, (22, 832), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (220, 240, 255), 1, cv2.LINE_AA)

        # 2. Draw Clickable Top-Right Buttons
        for x0, y0, x1, y1, label, cb_id in self.buttons:
            is_active = (
                (cb_id == "mode_1" and self.mode == 1)
                or (cb_id == "mode_2" and self.mode == 2)
                or (cb_id == "mode_3" and self.mode == 3)
            )
            bg_col = (30, 130, 210) if is_active else (50, 58, 72)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), bg_col, -1)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (140, 170, 200), 1)
            cv2.putText(canvas, label, (x0 + 10, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

        # 3. Middle-Right Panel 1 (315 x 340): Real Human Demonstration Video Frame
        vid_x0, vid_y0, vid_w, vid_h = 780, 98, 315, 340
        cv2.rectangle(canvas, (vid_x0, vid_y0), (vid_x0 + vid_w, vid_y0 + 28), (45, 52, 65), -1)
        cv2.putText(
            canvas,
            f"REAL HUMAN VIDEO ({self.curr_demo_id})",
            (vid_x0 + 10, vid_y0 + 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (0, 230, 255),
            1,
            cv2.LINE_AA,
        )
        if self.real_video_files:
            v_idx = min(int(round(t * (len(self.real_video_files) - 1) / float(max(1, self.T - 1)))), len(self.real_video_files) - 1)
            v_img = cv2.imread(str(self.real_video_files[v_idx]))
            if v_img is not None:
                v_res = cv2.resize(v_img, (vid_w, vid_h - 28))
                canvas[vid_y0 + 28 : vid_y0 + vid_h, vid_x0 : vid_x0 + vid_w] = v_res

        # 4. Middle-Right Panel 2 (325 x 340): Robot Wrist Camera (eye_in_hand)
        eih_x0, eih_y0, eih_w, eih_h = 1105, 98, 320, 340
        cv2.rectangle(canvas, (eih_x0, eih_y0), (eih_x0 + eih_w, eih_y0 + 28), (45, 52, 65), -1)
        cv2.putText(
            canvas,
            "WRIST CAMERA (eye_in_hand_rgb)",
            (eih_x0 + 10, eih_y0 + 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (0, 230, 255),
            1,
            cv2.LINE_AA,
        )
        eih_rgb = self.renderer.render_frame(
            qpos=qpos,
            cube_world_pos=cube_pos,
            gripper_action=grip_cmd,
            width=192,
            height=192,
            camera_mode="eye_in_hand",
        )
        eih_bgr = cv2.cvtColor(eih_rgb, cv2.COLOR_RGB2BGR)
        canvas[eih_y0 + 28 : eih_y0 + eih_h, eih_x0 : eih_x0 + eih_w] = cv2.resize(eih_bgr, (eih_w, eih_h - 28))

        # 5. Bottom-Right Live World Model Trajectory & Error Dashboard (645 x 354)
        active_pred = self.wm_states if self.mode != 2 else self.bc_states
        active_err = np.linalg.norm(active_pred[:, :3] - self.gt_states[:, :3], axis=-1) * 100.0
        dash = _draw_live_wm_dashboard(
            gt_traj=self.gt_states,
            wm_traj=active_pred,
            err_cm=active_err,
            curr_step=t,
            demo_label=f"{self.curr_demo_id} | {'WM' if self.mode != 2 else 'BC'}",
            width=645,
            height=580,
        )
        canvas[446:802, 780:1425] = cv2.resize(dash, (645, 356))

        return canvas

    def step_manual_key(self, key: int) -> None:
        step_m = 0.018
        moved = False
        delta = np.zeros(3, dtype=np.float64)
        if key in (ord("w"), ord("W")):
            delta[0] = -step_m
            moved = True
        elif key in (ord("s"), ord("S")):
            delta[0] = +step_m
            moved = True
        elif key in (ord("a"), ord("A")):
            delta[1] = -step_m
            moved = True
        elif key in (ord("d"), ord("D")):
            delta[1] = +step_m
            moved = True
        elif key in (ord("q"), ord("Q")):
            delta[2] = +step_m
            moved = True
        elif key in (ord("e"), ord("E")):
            delta[2] = -step_m
            moved = True

        if moved:
            self.man_target_pos = np.clip(
                self.man_target_pos + delta,
                np.array([-0.28, -0.28, 0.816]),
                np.array([0.25, 0.28, 1.18]),
            )
            self.man_last_delta = delta.astype(np.float32)
            self.man_qpos = panda_inverse_kinematics_step(
                self.man_qpos, self.man_target_pos, self.man_ee_rot
            )
            _, self.man_ee_pos, _ = panda_forward_kinematics(self.man_qpos)
            self.man_trail.append(self.man_ee_pos.copy())

        # Update manual cube grasp physics
        dist_xy = float(np.linalg.norm(self.man_ee_pos[:2] - self.man_cube_pos[:2]))
        dist_z = abs(float(self.man_ee_pos[2] - self.man_cube_pos[2]))
        if self.man_grip > 0.0 and not self.man_grasped and dist_xy < 0.048 and dist_z < 0.048:
            self.man_grasped = True
            self.man_grasp_offset = self.man_cube_pos - self.man_ee_pos
        elif self.man_grip < 0.0 and self.man_grasped:
            self.man_grasped = False

        if self.man_grasped:
            self.man_cube_pos = self.man_ee_pos + self.man_grasp_offset
            self.man_cube_pos[2] = max(0.821, self.man_cube_pos[2])
        else:
            if self.man_cube_pos[2] > 0.821:
                self.man_cube_pos[2] = max(0.821, self.man_cube_pos[2] - 0.025)

    def run(self, screenshot_out: Path | None = None, max_frames: int | None = None) -> None:
        if screenshot_out is not None:
            # Render a crystal-clear preview screenshot at peak lift (step 40)
            self.step_idx = min(40, self.T - 1)
            frame = self.render_gui_frame()
            screenshot_out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(screenshot_out), frame)
            logger.info("Saved interactive GUI preview screenshot: %s", screenshot_out)
            self.step_idx = 0
            if max_frames == 1:
                return

        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_NAME, 1440, 860)
        cv2.setMouseCallback(self.WINDOW_NAME, self._on_mouse)

        logger.info("Opened Interactive LIBERO / MuJoCo 3D Franka Panda GUI Window!")
        frame_count = 0
        while True:
            frame = self.render_gui_frame()
            cv2.imshow(self.WINDOW_NAME, frame)

            key = cv2.waitKey(45) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord("1"):
                self._trigger_action("mode_1")
            elif key == ord("2"):
                self._trigger_action("mode_2")
            elif key == ord("3"):
                self._trigger_action("mode_3")
            elif key == ord(" "):
                self._trigger_action("toggle_pause")
            elif key in (ord("n"), ord("N")):
                self._trigger_action("next_demo")
            elif key in (ord("p"), ord("P")):
                self._trigger_action("prev_demo")
            elif key in (ord("c"), ord("C")):
                self._trigger_action("move_cube")
            elif key in (ord("r"), ord("R")):
                self._trigger_action("reset")
            elif key in (ord("+"), ord("=")):
                self.cam_distance = float(np.clip(self.cam_distance - 0.06, 0.45, 2.0))
            elif key in (ord("-"), ord("_")):
                self.cam_distance = float(np.clip(self.cam_distance + 0.06, 0.45, 2.0))
            elif self.mode == 3 and key != 255:
                self.step_manual_key(key)

            if self.mode in (1, 2) and not self.paused:
                self.step_idx = (self.step_idx + 1) % self.T

            frame_count += 1
            if max_frames is not None and frame_count >= max_frames:
                break

            # Detect if user clicked the window [X] close button
            if cv2.getWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyAllWindows()


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive LIBERO / MuJoCo 3D Franka Panda & World Model GUI")
    parser.add_argument("--demo_idx", type=int, default=8, help="Initial demo index (0..17, default 8 = test demo_8)")
    parser.add_argument("--screenshot_out", type=Path, default=None, help="Optional path to save GUI preview screenshot")
    parser.add_argument("--max_frames", type=int, default=None, help="Optional max frames before auto-exit")
    args = parser.parse_args()

    app = InteractiveLiberoMuJoCoApp(start_demo_idx=args.demo_idx)
    app.run(screenshot_out=args.screenshot_out, max_frames=args.max_frames)
    return 0


if __name__ == "__main__":
    sys.exit(main())
