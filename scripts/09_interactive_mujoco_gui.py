"""scripts/09_interactive_mujoco_gui.py — Interactive LIBERO / MuJoCo 3D Franka Panda & World Model Tester GUI.

Opens an interactive graphical window allowing you to test and play with:
  1. [Mode 1: World Model Closed-Loop Rollout]:
     Watch the trained Action-Conditioned World Model (`world_model_checkpoint.pt`) predict the
     3D trajectory in closed-loop and drive the 3D Franka Emika Panda arm to grasp, lift, and place
     the red cube, synchronized with the real human demonstration video.
  2. [Mode 2: Visuomotor BC Policy Rollout]:
     Run the trained Dual-Head Visuomotor Behavioral Cloning Policy (`best_policy.pt`) live on any test video.
  3. [Mode 3: Interactive Manual Control + Live World Model Future Horizon Prediction]:
     Drive the 7-DOF Franka Emika Panda arm yourself with keyboard keys (W/A/S/D/Q/E + SPACE to grasp)
     while the World Model predicts your next 15 future 3D states live in the 3D MuJoCo viewport!

Test Videos Folder & Model Switching Controls:
  - [O] / Click "[O] Open Folder" : Open a specific Test Videos folder (defaults to `data/test_videos/`).
                                    Automatically processes new raw .mp4 videos if not yet cached!
  - [V] / Click "[V] Pick Video"  : Pick a specific single .mp4 test video from `data/test_videos/` (or any folder).
  - [M] / Click "[M] Switch Model": Cycle live between saved trained models in `experiments/checkpoints/registry/`
                                    (e.g., `raw_2`, `demo_libero`).

Interactive Controls (Mouse & Keyboard):
  - Mouse Left-Drag on 3D View : Orbit 3D Camera 360 deg around the Franka Panda arm & table
  - Mouse Wheel / [+] / [-]    : Zoom 3D Camera In / Out
  - Click On-Screen Buttons    : Switch Modes, Open Test Folder/Video, Switch Model, Pause/Play, Next/Prev Video
  - [1] / [2] / [3]            : Switch between (1) World Model, (2) BC Policy, (3) Manual Control
  - [N] / [P]                  : Next / Previous Test Video in the active folder
  - [O] / [V] / [M]            : Open Test Videos Folder / Pick Single Test Video / Switch Trained Model Profile
  - [SPACE]                    : Pause/Play (in Modes 1 & 2) or Toggle Gripper Grasp (in Mode 3)
  - [W/S], [A/D], [Q/E]        : Move Panda EEF Forward/Back (X), Left/Right (Y), Up/Down (Z) in Mode 3
  - [C]                        : Move Red Cube to a new random tabletop position
  - [R]                        : Reset current episode & camera view
  - [ESC]                      : Close window
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_mod_bc = importlib.import_module("scripts.06_train_policy")
_mod_wm = importlib.import_module("scripts.07_eval_world_model")
_mod_batch = importlib.import_module("scripts.run_batch_pipeline")
BCPolicy = _mod_bc.BCPolicy
ActionConditionedDynamicsModel = _mod_wm.ActionConditionedDynamicsModel
_draw_live_wm_dashboard = _mod_wm._draw_live_wm_dashboard

from src.retargeting.panda_3d_renderer import PandaMuJoCo3DRenderer
from src.retargeting.sim_replay import (
    build_object_aligned_target_trajectory,
    panda_forward_kinematics,
    panda_inverse_kinematics_step,
)
from src.utils.video_processor import batch_transform_videos_to_frames

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("interactive_mujoco")

DEFAULT_TEST_VIDEOS_DIR = ROOT / "data/test_videos"


class InteractiveLiberoMuJoCoApp:
    """Interactive 3D Franka Emika Panda MuJoCo Simulator & Test Video Folder Evaluator."""

    WINDOW_NAME = "LIBERO / MuJoCo 3D Franka Panda - Interactive Test Video & Model Simulator"

    def __init__(
        self,
        dataset_path: Path = ROOT / "data/datasets/demo_libero.hdf5",
        wm_ckpt_path: Path = ROOT / "experiments/world_model/world_model_checkpoint.pt",
        bc_ckpt_path: Path = ROOT / "experiments/checkpoints/best_policy.pt",
        test_dir: Path | None = None,
        start_demo_idx: int = 0,
        choose_folder_on_start: bool = False,
    ) -> None:
        DEFAULT_TEST_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.renderer = PandaMuJoCo3DRenderer()

        # Discover available trained model profiles in experiments/checkpoints/registry/
        self.available_models = self._discover_registered_models()
        self.active_model_name = self._infer_model_name(bc_ckpt_path)

        # Match with registered model profile checkpoints if available
        matched_reg = next((m for m in self.available_models if m["name"] == self.active_model_name), None)
        if matched_reg is not None and bc_ckpt_path == ROOT / "experiments/checkpoints/best_policy.pt":
            bc_ckpt_path = Path(matched_reg["bc_ckpt"])
            wm_ckpt_path = Path(matched_reg["wm_ckpt"])

        # Load Trained World Model & Visuomotor BC Policy
        self.wm_model = ActionConditionedDynamicsModel().to(self.device)
        self.bc_model = BCPolicy(proprio_dim=7, action_dim=7).to(self.device)
        self._load_model_checkpoints(bc_ckpt_path, wm_ckpt_path, model_label=self.active_model_name)

        # Interactive Camera State (360 deg orbit + zoom)
        self.cam_azimuth = 0.0
        self.cam_elevation = 31.4
        self.cam_distance = 0.96
        self.mouse_dragging = False
        self.last_mouse_xy = (0, 0)

        # Mode & Episode State
        self.mode = 1
        self.paused = False
        self.step_idx = 0
        self.show_3d_trails = True

        self.dataset_path = dataset_path
        self.dataset_tag = dataset_path.stem
        self.active_test_folder: Path = test_dir if (test_dir is not None and test_dir.exists()) else DEFAULT_TEST_VIDEOS_DIR
        self.demos: list[dict] = []

        # Clickable UI buttons (2 rows on top-right)
        self.buttons: list[tuple[int, int, int, int, str, str]] = [
            (780, 10, 912, 44, "[1] World Model", "mode_1"),
            (918, 10, 1038, 44, "[2] BC Policy", "mode_2"),
            (1044, 10, 1178, 44, "[3] Manual+WM", "mode_3"),
            (1184, 10, 1304, 44, "[O] Open Folder", "open_folder"),
            (1310, 10, 1428, 44, "[V] Pick Video", "pick_video"),
            (780, 50, 895, 84, "[P] < Prev Vid", "prev_demo"),
            (901, 50, 1018, 84, "[N] Next Vid >", "next_demo"),
            (1024, 50, 1176, 84, "[M] Switch Model", "switch_model"),
            (1182, 50, 1304, 84, "[C] Move Cube", "move_cube"),
            (1310, 50, 1428, 84, "[Space] Pause", "toggle_pause"),
        ]

        # Decide initial test source:
        # 1) If user passed --choose_folder, open folder picker immediately
        # 2) If user passed --test_dir (or if data/test_videos has videos and no explicit dataset was forced), load test_dir
        if choose_folder_on_start:
            chosen = self._prompt_folder_dialog(initial_dir=self.active_test_folder)
            if chosen is not None:
                self.load_test_source(chosen, start_idx=0)
            else:
                self.load_test_source(self.active_test_folder, start_idx=start_demo_idx)
        elif test_dir is not None and test_dir.exists():
            self.load_test_source(test_dir, start_idx=start_demo_idx)
        elif self._folder_has_videos(DEFAULT_TEST_VIDEOS_DIR) and dataset_path == ROOT / "data/datasets/demo_libero.hdf5" and start_demo_idx == 0:
            self.load_test_source(DEFAULT_TEST_VIDEOS_DIR, start_idx=0)
        else:
            self._load_hdf5_dataset(self.dataset_path, start_idx=start_demo_idx)

    def _discover_registered_models(self) -> list[dict[str, Path | str]]:
        reg_root = ROOT / "experiments/checkpoints/registry"
        models_list: list[dict[str, Path | str]] = []
        if reg_root.is_dir():
            for d in sorted(reg_root.iterdir()):
                bc_p = d / "best_policy.pt"
                if d.is_dir() and bc_p.is_file():
                    wm_p = d / "world_model_checkpoint.pt"
                    if not wm_p.is_file():
                        wm_p = ROOT / "experiments/world_model/world_model_checkpoint.pt"
                    models_list.append({"name": d.name, "bc_ckpt": bc_p, "wm_ckpt": wm_p})
        if not models_list:
            models_list.append(
                {
                    "name": "default",
                    "bc_ckpt": ROOT / "experiments/checkpoints/best_policy.pt",
                    "wm_ckpt": ROOT / "experiments/world_model/world_model_checkpoint.pt",
                }
            )
        return models_list

    def _infer_model_name(self, bc_ckpt_path: Path) -> str:
        if bc_ckpt_path.parent.parent.name == "registry":
            return bc_ckpt_path.parent.name
        if bc_ckpt_path.is_file():
            try:
                ckpt = torch.load(bc_ckpt_path, map_location="cpu", weights_only=False)
                if ckpt.get("model_name"):
                    return str(ckpt["model_name"])
            except Exception:
                pass
        return "demo_libero"

    def _load_model_checkpoints(self, bc_ckpt_path: Path, wm_ckpt_path: Path, model_label: str = "") -> None:
        if wm_ckpt_path.exists():
            ckpt = torch.load(wm_ckpt_path, map_location=self.device, weights_only=False)
            self.wm_model = ActionConditionedDynamicsModel().to(self.device)
            self.wm_model.load_state_dict(ckpt["model_state_dict"], strict=False)
            logger.info("Loaded trained World Model checkpoint: %s", wm_ckpt_path)
        self.wm_model.eval()

        if bc_ckpt_path.exists():
            ckpt_bc = torch.load(bc_ckpt_path, map_location=self.device, weights_only=False)
            self.bc_model = BCPolicy(
                proprio_dim=int(ckpt_bc.get("proprio_dim", 7)),
                action_dim=int(ckpt_bc.get("action_dim", 7)),
                in_channels=int(ckpt_bc.get("in_channels", 3)),
            ).to(self.device)
            self.bc_model.load_state_dict(ckpt_bc["model_state_dict"], strict=False)
            inferred = ckpt_bc.get("model_name") or self._infer_model_name(bc_ckpt_path)
            self.active_model_name = model_label or str(inferred)
            logger.info("Loaded trained Visuomotor BC Policy checkpoint [%s]: %s", self.active_model_name, bc_ckpt_path)
        self.bc_model.eval()

    @staticmethod
    def _folder_has_videos(folder: Path) -> list[Path]:
        if not folder.is_dir():
            return []
        vids = []
        for ext in ("*.mp4", "*.mov", "*.avi", "*.mkv"):
            vids.extend(folder.glob(ext))
        return sorted(vids)

    def _ensure_test_folder_processed(self, folder: Path) -> tuple[Path, str]:
        """Ensures a folder of raw test videos (.mp4) or extracted frames has an up-to-date HDF5 dataset & frame folder."""
        folder = folder.resolve()
        # Case 1: User picked a single demo frame folder (e.g. data/frames/raw_2/demo_0003)
        if any(folder.glob("frame_*.png")) or any(folder.glob("frame_*.jpg")):
            parent_tag = folder.parent.name if folder.parent.name != "frames" else "demo_libero"
            cand_h5 = ROOT / "data/datasets" / f"{parent_tag}.hdf5"
            if cand_h5.is_file():
                return cand_h5, parent_tag

        # Case 2: User picked a frames container folder (e.g. data/frames/raw_2 or data/frames)
        subdirs = _mod_batch.get_demo_directories(folder)
        if subdirs and not self._folder_has_videos(folder):
            tag = "demo_libero" if folder.name == "frames" else folder.name
            cand_h5 = ROOT / "data/datasets" / f"{tag}.hdf5"
            if cand_h5.is_file():
                return cand_h5, tag

        # Case 3: Folder contains raw video files (.mp4, .mov, .avi) -> e.g. data/test_videos or data/raw_videos/raw_2
        vids = self._folder_has_videos(folder)
        if not vids:
            raise FileNotFoundError(f"No video files (.mp4/.mov/.avi) or extracted frame folders found in '{folder}'.")

        dataset1_dirs = {
            (ROOT / "data/raw_videos").resolve(),
            (ROOT / "data/raw_videos/Nouveau dossier").resolve(),
        }
        if folder in dataset1_dirs:
            safe_tag = "demo_libero"
        else:
            raw_tag = folder.name
            if raw_tag.lower() in ("raw_videos", "nouveau dossier", "nouveau_dossier", "new folder"):
                raw_tag = f"{folder.parent.name}_{folder.name}"
            safe_tag = "".join(c if (c.isalnum() or c in ("_", "-")) else "_" for c in raw_tag).strip("_") or "test_videos"

        frames_root = (ROOT / "data/frames") if safe_tag == "demo_libero" else (ROOT / "data/frames" / safe_tag)
        output_hdf5 = ROOT / "data/datasets" / f"{safe_tag}.hdf5"

        # Check if output_hdf5 already exists and has the same number of demos
        needs_processing = True
        if output_hdf5.is_file() and frames_root.is_dir():
            try:
                with h5py.File(output_hdf5, "r") as f_check:
                    n_h5_demos = len(f_check["data"].keys()) if "data" in f_check else 0
                existing_frame_dirs = _mod_batch.get_demo_directories(frames_root)
                if n_h5_demos == len(vids) and len(existing_frame_dirs) == len(vids):
                    needs_processing = False
            except Exception:
                needs_processing = True

        if needs_processing:
            logger.info("Processing %d test video(s) from '%s' -> '%s'...", len(vids), folder, output_hdf5)
            self._show_processing_overlay(f"Extracting & Retargeting {len(vids)} Test Videos from '{folder.name}'...")

            if frames_root.is_dir() and safe_tag != "demo_libero":
                import shutil
                shutil.rmtree(frames_root)
            frames_root.mkdir(parents=True, exist_ok=True)

            batch_transform_videos_to_frames(
                video_sources=folder,
                output_root=frames_root,
                naming_scheme="index",
                prefix="demo_",
                target_fps=20.0,
                target_size=None,
                center_crop=False,
            )

            demo_dirs = _mod_batch.get_demo_directories(frames_root)
            if output_hdf5.is_file():
                output_hdf5.unlink()

            calib_txt = ROOT / "data/calibration/calib.txt"
            for idx, d_dir in enumerate(demo_dirs):
                self._show_processing_overlay(
                    f"Tracking ArUco + 3D Hand + Panda Retargeting: Video {idx + 1}/{len(demo_dirs)} ({vids[min(idx, len(vids) - 1)].name})..."
                )
                _mod_batch.run_pipeline_for_demo(
                    demo_dir=d_dir,
                    demo_idx=idx,
                    calib_txt=calib_txt,
                    output_hdf5=output_hdf5,
                    base_data_dir=ROOT / "data",
                    marker_size=0.05,
                    aruco_id=0,
                    aruco_dict="DICT_ARUCO_ORIGINAL",
                    pinch_threshold=0.04,
                    tgt_fps=20.0,
                    env_name="LIBERO_OBJECT",
                    frame_size=(128, 128),
                    save_debug=False,
                )

        return output_hdf5, safe_tag

    def _show_processing_overlay(self, message: str) -> None:
        """Displays a live progress banner if the OpenCV simulator window is currently open."""
        try:
            if cv2.getWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1:
                banner = np.full((860, 1440, 3), 24, dtype=np.uint8)
                cv2.rectangle(banner, (220, 360), (1220, 500), (45, 65, 95), -1)
                cv2.rectangle(banner, (220, 360), (1220, 500), (0, 230, 255), 2)
                cv2.putText(
                    banner,
                    "PROCESSING NEW TEST VIDEOS FOR 3D SIMULATOR...",
                    (360, 415),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.80,
                    (0, 240, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    banner,
                    message,
                    (260, 462),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.56,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                cv2.imshow(self.WINDOW_NAME, banner)
                cv2.waitKey(15)
        except Exception:
            pass

    def load_test_source(self, source_path: Path, start_idx: int = 0) -> None:
        """Loads a folder of test videos (.mp4), a single test video (.mp4), a frames folder, or an .hdf5 dataset."""
        source_path = Path(source_path).resolve()
        if source_path.is_file():
            if source_path.suffix.lower() in (".hdf5", ".h5"):
                self.active_test_folder = source_path.parent
                self._load_hdf5_dataset(source_path, start_idx=start_idx)
                return
            elif source_path.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv"):
                parent_dir = source_path.parent
                vids = self._folder_has_videos(parent_dir)
                target_idx = next((i for i, v in enumerate(vids) if v.resolve() == source_path), 0)
                self.active_test_folder = parent_dir
                h5_path, tag = self._ensure_test_folder_processed(parent_dir)
                self._load_hdf5_dataset(h5_path, dataset_tag=tag, start_idx=target_idx)
                return

        if source_path.is_dir():
            # Check if the user clicked a specific demo_XXXX frame subfolder
            if any(source_path.glob("frame_*.png")) or any(source_path.glob("frame_*.jpg")):
                digits = "".join(c for c in source_path.name if c.isdigit())
                target_idx = int(digits) if digits else 0
                h5_path, tag = self._ensure_test_folder_processed(source_path)
                self.active_test_folder = source_path.parent
                self._load_hdf5_dataset(h5_path, dataset_tag=tag, start_idx=target_idx)
                return

            self.active_test_folder = source_path
            h5_path, tag = self._ensure_test_folder_processed(source_path)
            self._load_hdf5_dataset(h5_path, dataset_tag=tag, start_idx=start_idx)

    def _auto_select_unique_model_for_dataset(self, dataset_tag: str) -> None:
        """Automatically loads the unique BC Policy & World Model trained specifically for `dataset_tag`."""
        self.available_models = self._discover_registered_models()
        reg_dir = ROOT / "experiments/checkpoints/registry" / dataset_tag
        if (reg_dir / "best_policy.pt").is_file():
            bc_p = reg_dir / "best_policy.pt"
            wm_p = reg_dir / "world_model_checkpoint.pt"
            if not wm_p.is_file():
                wm_p = ROOT / "experiments/world_model/world_model_checkpoint.pt"
            if self.active_model_name != dataset_tag:
                logger.info("Auto-binding unique trained model profile '%s' for dataset '%s'", dataset_tag, dataset_tag)
                self._load_model_checkpoints(bc_p, wm_p, model_label=dataset_tag)

    def _load_hdf5_dataset(self, hdf5_path: Path, dataset_tag: str | None = None, start_idx: int = 0) -> None:
        self.dataset_path = hdf5_path
        self.dataset_tag = dataset_tag or hdf5_path.stem
        self.available_models = self._discover_registered_models()
        self.demos = []

        frames_base = (
            (ROOT / "data/frames" / self.dataset_tag)
            if (ROOT / "data/frames" / self.dataset_tag).is_dir()
            else (ROOT / "data/frames")
        )

        with h5py.File(self.dataset_path, "r") as f:
            keys = sorted(list(f["data"].keys()), key=lambda k: int("".join(c for c in k if c.isdigit()) or "0"))
            for k in keys:
                grp = f["data"][k]
                pos = grp["obs"]["robot0_eef_pos"][:].astype(np.float32)
                quat = grp["obs"]["robot0_eef_quat"][:].astype(np.float32)
                acts = grp["actions"][:].astype(np.float32)
                rgb = grp["obs"]["agentview_rgb"][:] if "agentview_rgb" in grp["obs"] else None
                demo_num = int("".join(c for c in k if c.isdigit()) or "0")
                demo_id = f"demo_{demo_num:04d}"

                # Discover original video filename and optional source_dataset from metadata.json
                video_label = demo_id
                source_dataset_tag = self.dataset_tag
                meta_p = frames_base / demo_id / "metadata.json"
                if meta_p.is_file():
                    try:
                        with open(meta_p, "r", encoding="utf-8") as fm:
                            m_data = json.load(fm)
                            src_v = m_data.get("source_video", "")
                            if src_v:
                                video_label = Path(src_v).name
                            if m_data.get("source_dataset"):
                                source_dataset_tag = str(m_data["source_dataset"])
                    except Exception:
                        pass

                self.demos.append(
                    {
                        "key": k,
                        "id": demo_id,
                        "video_label": video_label,
                        "source_dataset": source_dataset_tag,
                        "frames_dir": frames_base / demo_id,
                        "states": np.concatenate([pos, quat], axis=-1),
                        "actions": acts,
                        "rgb": rgb,
                    }
                )

        self.demo_idx = min(max(0, start_idx), max(0, len(self.demos) - 1))
        self._load_demo_episode(self.demo_idx)
        logger.info(
            "Loaded %d test videos from '%s' (Frames dir: '%s')",
            len(self.demos),
            self.dataset_path.name,
            frames_base,
        )

    def _load_demo_episode(self, idx: int, custom_cube_pos: np.ndarray | None = None) -> None:
        self.demo_idx = idx % len(self.demos)
        demo = self.demos[self.demo_idx]
        self.curr_demo_key = demo["key"]
        self.curr_demo_id = demo["id"]
        self.curr_video_label = demo.get("video_label", self.curr_demo_id)
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
                rgb_batch = torch.from_numpy(demo["rgb"]).permute(0, 3, 1, 2).float().div_(255.0).to(self.device)
                prop_batch = torch.from_numpy(self.gt_states).float().to(self.device)
                bc_actions = self.bc_model(rgb_batch, prop_batch).cpu().numpy()
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
        if custom_cube_pos is not None:
            self.cube_init_pos = custom_cube_pos.copy()
        else:
            # Anchor initial object tabletop XY position from the video's detected 3D pick point
            from src.retargeting.sim_replay import detect_manipulation_phases

            ep_phases = detect_manipulation_phases(self.gt_states[:, :3], self.gt_actions)
            # Map pick_pos_demo relative to canonical robot-frame tabletop center [0.52, -0.15] into sim table bounds
            obj_x = float(np.clip(ep_phases.pick_pos_demo[0] - 0.52, -0.10, 0.10))
            obj_y = float(np.clip(ep_phases.pick_pos_demo[1] - (-0.15), -0.12, 0.12))
            self.cube_init_pos = np.array([obj_x, obj_y, 0.821], dtype=np.float64)
        self._recompute_sim_kinematics()

        # 4. Load real video frames from the dataset-specific frames folder
        frames_dir = demo.get("frames_dir", ROOT / "data/frames" / self.curr_demo_id)
        if not frames_dir.is_dir():
            frames_dir = ROOT / "data/frames" / self.curr_demo_id
        self.real_video_files = sorted(frames_dir.glob("*.png")) + sorted(frames_dir.glob("*.jpg"))

        # 5. Initialize Manual Sandbox State (Mode 3)
        self._reset_manual_state()

    def _recompute_sim_kinematics(self) -> None:
        active_traj = self.wm_states[:, :3] if self.mode == 1 else self.bc_states[:, :3]

        qpos = np.array([0.0, 0.196, 0.0, -2.43, 0.0, 2.62, 0.785], dtype=np.float64)
        _, ee_init, rot_init = panda_forward_kinematics(qpos)

        # Ground-truth object-aligned 3D target trajectory from human demonstration
        self.gt_target_world, gt_grip, _ = build_object_aligned_target_trajectory(
            demo_eef_pos=self.gt_states[:, :3],
            demo_actions=self.gt_actions,
            sim_eef_init=ee_init,
            sim_cube_init=self.cube_init_pos,
        )

        # Apply exact 3D prediction delta (active_traj - gt_states) so 3D sim matches World Model & BC Policy rollouts 1-to-1
        pred_delta_world = (active_traj - self.gt_states[:, :3]).astype(np.float64)
        self.sim_target_world = self.gt_target_world + pred_delta_world
        self.sim_target_world[:, 2] = np.maximum(self.sim_target_world[:, 2], 0.815)

        if self.mode == 1:
            self.sim_grip = gt_grip
        else:
            # Combine BC Policy gripper prediction with object-aligned grasp phases
            bc_g = np.where(self.bc_actions[:, 6] > 0.0, 1.0, -1.0)
            self.sim_grip = np.where((gt_grip > 0.0) | (bc_g > 0.0), gt_grip, -1.0)
        target_pos = self.sim_target_world
        target_grip = self.sim_grip

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

    def _prompt_folder_dialog(self, initial_dir: Path | None = None) -> Path | None:
        init_d = initial_dir if (initial_dir is not None and initial_dir.exists()) else DEFAULT_TEST_VIDEOS_DIR
        tk_root = tk.Tk()
        tk_root.withdraw()
        tk_root.attributes("-topmost", True)
        chosen = filedialog.askdirectory(
            initialdir=str(init_d),
            title="Select Test Videos Folder (e.g. data/test_videos, data/raw_videos/raw_2, or any video folder)",
        )
        tk_root.destroy()
        return Path(chosen) if chosen else None

    def _prompt_video_dialog(self, initial_dir: Path | None = None) -> Path | None:
        init_d = initial_dir if (initial_dir is not None and initial_dir.exists()) else DEFAULT_TEST_VIDEOS_DIR
        tk_root = tk.Tk()
        tk_root.withdraw()
        tk_root.attributes("-topmost", True)
        chosen = filedialog.askopenfilename(
            initialdir=str(init_d),
            title="Select a Test Video (.mp4 / .mov / .avi) or Dataset (.hdf5)",
            filetypes=[
                ("Test Videos & Datasets", "*.mp4 *.mov *.avi *.mkv *.hdf5 *.h5"),
                ("Video Files", "*.mp4 *.mov *.avi *.mkv"),
                ("HDF5 Datasets", "*.hdf5 *.h5"),
                ("All Files", "*.*"),
            ],
        )
        tk_root.destroy()
        return Path(chosen) if chosen else None

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
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
        elif cb_id == "open_folder":
            chosen_dir = self._prompt_folder_dialog(initial_dir=self.active_test_folder)
            if chosen_dir is not None:
                try:
                    self.load_test_source(chosen_dir, start_idx=0)
                except Exception as e:
                    logger.error("Could not load test folder '%s': %s", chosen_dir, e)
        elif cb_id == "pick_video":
            chosen_file = self._prompt_video_dialog(initial_dir=self.active_test_folder)
            if chosen_file is not None:
                try:
                    self.load_test_source(chosen_file, start_idx=0)
                except Exception as e:
                    logger.error("Could not load test video '%s': %s", chosen_file, e)
        elif cb_id == "switch_model":
            self.available_models = self._discover_registered_models()
            if len(self.available_models) > 0:
                prev_name = self.active_model_name
                curr_idx = next(
                    (i for i, m in enumerate(self.available_models) if m["name"] == self.active_model_name),
                    0,
                )
                next_m = self.available_models[(curr_idx + 1) % len(self.available_models)]
                self._load_model_checkpoints(
                    Path(next_m["bc_ckpt"]),
                    Path(next_m["wm_ckpt"]),
                    model_label=str(next_m["name"]),
                )
                self._load_demo_episode(self.demo_idx)
                logger.info(
                    "Switched active model in simulator: '%s' -> '%s' (evaluating on video '%s')",
                    prev_name,
                    self.active_model_name,
                    self.curr_video_label,
                )
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
                    (self.gt_target_world, (0, 210, 255), 2),
                    (self.sim_target_world[: t + 1], (255, 80, 40), 3),
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
                (wm_future_3d, (255, 70, 220), 3),
            ]

        # 1. Left Main 3D MuJoCo Viewport (750 x 750)
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
            1: f"MODE 1: WORLD MODEL ROLLOUT | MODEL: {self.active_model_name.upper()}",
            2: f"MODE 2: DUAL-HEAD BC POLICY | MODEL: {self.active_model_name.upper()}",
            3: f"MODE 3: MANUAL CONTROL + LIVE WM | MODEL: {self.active_model_name.upper()}",
        }
        cv2.rectangle(canvas, (12, 10), (762, 50), (45, 65, 95), -1)
        cv2.putText(canvas, mode_names[self.mode], (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 240, 255), 2, cv2.LINE_AA)

        # Overlay 3D Viewport HUD
        cube_lift_cm = max(0.0, (cube_pos[2] - 0.821) * 100.0)
        cv2.rectangle(canvas, (22, 62), (485, 190), (18, 22, 30), -1)
        cv2.putText(
            canvas,
            f"Test Folder: '{self.dataset_tag}' | Video [{self.demo_idx + 1}/{len(self.demos)}]: {self.curr_video_label}",
            (32, 84),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 235, 255),
            1,
            cv2.LINE_AA,
        )
        curr_model_idx = next(
            (i for i, m in enumerate(self.available_models) if m["name"] == self.active_model_name),
            0,
        )
        cv2.putText(
            canvas,
            f"Active Model [{curr_model_idx + 1}/{max(1, len(self.available_models))}]: '{self.active_model_name}' ([M] Switch) | Step: {t:02d}/{self.T - 1:02d}",
            (32, 106),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            (255, 230, 140),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"Panda EEF World [X,Y,Z]: [{ee_pos[0]:+.3f}, {ee_pos[1]:+.3f}, {ee_pos[2]:+.3f}] m",
            (32, 128),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (180, 245, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"Red Cube Lift: {cube_lift_cm:.1f} cm | Gripper: {'CLOSED (GRASP)' if grip_cmd > 0 else 'OPEN'}",
            (32, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (120, 255, 150) if cube_lift_cm > 1.0 else (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"Camera Orbit: Az={self.cam_azimuth:.0f} deg, El={self.cam_elevation:.0f} deg | [O] Open Folder | [V] Pick Video | [M] Model",
            (32, 174),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            (200, 200, 210),
            1,
            cv2.LINE_AA,
        )

        # Bottom controls bar under 3D viewport
        cv2.rectangle(canvas, (12, 806), (1428, 848), (32, 36, 46), -1)
        ctrl_text = (
            "CONTROLS: [O] Open Test Folder (data/test_videos) | [V] Pick Video | [M] Switch Model | [N/P] Next/Prev Video | "
            "[1/2/3] Mode | [Mouse Drag] Orbit 3D | [SPACE] Pause/Grasp | [C] Move Cube | [ESC] Quit"
        )
        cv2.putText(canvas, ctrl_text, (20, 832), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 240, 255), 1, cv2.LINE_AA)

        # 2. Draw Clickable Top-Right Buttons
        for x0, y0, x1, y1, label, cb_id in self.buttons:
            is_active = (
                (cb_id == "mode_1" and self.mode == 1)
                or (cb_id == "mode_2" and self.mode == 2)
                or (cb_id == "mode_3" and self.mode == 3)
            )
            if is_active:
                bg_col = (30, 130, 210)
            elif cb_id in ("open_folder", "pick_video"):
                bg_col = (35, 120, 65)
            elif cb_id == "switch_model":
                bg_col = (120, 65, 135)
            else:
                bg_col = (50, 58, 72)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), bg_col, -1)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (140, 170, 200), 1)
            cv2.putText(canvas, label, (x0 + 7, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

        # 3. Middle-Right Panel 1 (315 x 340): Real Human Demonstration Video Frame
        vid_x0, vid_y0, vid_w, vid_h = 780, 98, 315, 340
        cv2.rectangle(canvas, (vid_x0, vid_y0), (vid_x0 + vid_w, vid_y0 + 28), (45, 52, 65), -1)
        cv2.putText(
            canvas,
            f"TEST VIDEO: {self.curr_video_label[:22]}",
            (vid_x0 + 8, vid_y0 + 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
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

        # 4. Middle-Right Panel 2 (320 x 340): Robot Wrist Camera (eye_in_hand)
        eih_x0, eih_y0, eih_w, eih_h = 1105, 98, 320, 340
        cv2.rectangle(canvas, (eih_x0, eih_y0), (eih_x0 + eih_w, eih_y0 + 28), (45, 52, 65), -1)
        cv2.putText(
            canvas,
            "WRIST CAMERA (eye_in_hand_rgb)",
            (eih_x0 + 10, eih_y0 + 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
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
            demo_label=f"{self.curr_video_label[:16]} | {'WM' if self.mode != 2 else 'BC'} ({self.active_model_name})",
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
            self.step_idx = min(35, self.T - 1)
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
            elif key in (ord("o"), ord("O")):
                self._trigger_action("open_folder")
            elif key in (ord("v"), ord("V")):
                self._trigger_action("pick_video")
            elif key in (ord("m"), ord("M")):
                self._trigger_action("switch_model")
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

            if cv2.getWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyAllWindows()


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive LIBERO / MuJoCo 3D Franka Panda & Test Folder Simulator")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/datasets/demo_libero.hdf5", help="HDF5 dataset path")
    parser.add_argument("--test_dir", type=Path, default=None, help="Dedicated Test Videos folder (default: data/test_videos)")
    parser.add_argument("--choose_folder", action="store_true", help="Prompt folder selection dialog on simulator startup")
    parser.add_argument("--bc_ckpt", type=Path, default=ROOT / "experiments/checkpoints/best_policy.pt", help="BC Policy checkpoint .pt")
    parser.add_argument("--wm_ckpt", type=Path, default=ROOT / "experiments/world_model/world_model_checkpoint.pt", help="World Model checkpoint .pt")
    parser.add_argument("--demo_idx", type=int, default=0, help="Initial video index in the loaded test folder")
    parser.add_argument("--screenshot_out", type=Path, default=None, help="Optional path to save GUI preview screenshot")
    parser.add_argument("--max_frames", type=int, default=None, help="Optional max frames before auto-exit")
    args = parser.parse_args()

    app = InteractiveLiberoMuJoCoApp(
        dataset_path=args.dataset,
        wm_ckpt_path=args.wm_ckpt,
        bc_ckpt_path=args.bc_ckpt,
        test_dir=args.test_dir,
        start_demo_idx=args.demo_idx,
        choose_folder_on_start=args.choose_folder,
    )
    app.run(screenshot_out=args.screenshot_out, max_frames=args.max_frames)
    return 0


if __name__ == "__main__":
    sys.exit(main())
