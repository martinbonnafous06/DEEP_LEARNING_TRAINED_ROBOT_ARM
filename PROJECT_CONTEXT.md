# PROJECT_CONTEXT.md — Project Memory & Living Documentation

> **Living Context File**: This document serves as the persistent architectural memory and change log for the **Egocentric Video to LIBERO Retargeting & Robot Manipulation Pipeline** (`C:\Users\marti\Desktop\Humanoid`).
> **Rule**: Every time a file is created, modified, or an experiment/fix is executed, this document MUST be updated to reflect the current state of the project.

---

## 1. Project Overview

This repository implements an end-to-end **Human-to-Robot Retargeting and Visuomotor Learning Pipeline** that converts monocular RGB human demonstration videos into **Franka Emika Panda** robot manipulation trajectories in **MuJoCo / Robosuite / LIBERO**, exports benchmark-compliant HDF5 datasets, and trains/evaluates Behavioral Cloning (BC) policies and Action-Conditioned World Models.

### Pipeline Stages
- **Stage 0 — Multi-Video Frame Extraction** (`scripts/00_extract_frames.py`, `src/utils/video_processor.py`):
  Extracts raw videos (`data/raw_videos/demo_01.mp4` .. `demo_18.mp4`, 1080x1920 portrait @ 60 FPS) into structured frame directories (`data/frames/demo_0000` .. `demo_0017` @ 20 Hz, 1320 total frames) with `metadata.json`, `timestamps.npy`, and global `dataset_manifest.json` / `.csv`.
- **Stage 1 — Camera Intrinsic Calibration** (`scripts/01_calibrate.py`, `src/calibration/calibrator.py`, `calibration.py`, `generate_checkerboard_pdf.py`):
  Calibrates camera intrinsics (`K`, `dist_coeffs`) using a 9x6 internal-corner chessboard (25 mm squares) from `data/calib_images/` (4000x2252 landscape) and exports `calib.txt` (DROID-SLAM format), `calibration.npz`, and `calibration.json`. Includes orientation-aware intrinsic adaptation (`adapt_camera_intrinsics` in `src/utils/coordinates.py`) when video orientation (portrait 1080x1920) differs from calibration photos (landscape 4000x2252).
- **Stage 2 — Camera Tracking & Table Plane Estimation** (`scripts/02_track_camera.py`, `src/slam/aruco_tracker.py`, `src/slam/plane_detector.py`, `camera_tracking.py`, `scripts/generate_aruco_tag.py`):
  Supports two modes:
  1. **ArUco Tabletop Fiducial Tracking (Default)**: Uses ArUco Tag ID 0 (`DICT_ARUCO_ORIGINAL`, 50 mm) placed flat on the table to solve exact metric 6-DoF camera poses (`T_world_to_cam`) via `cv2.solvePnP(..., SOLVEPNP_IPPE_SQUARE)` and anchor the tabletop plane at $Z = 0$ (`T_world_to_table = I_4`).
  2. **Monocular Visual SLAM Fallback**: Invokes DROID-SLAM and fits a table plane via Open3D RANSAC (`fit_table_plane_ransac`).
- **Stage 3 — 3D Hand Pose & Gripper State Estimation** (`scripts/03_track_hands.py`, `src/hand_pose/hand_tracker.py`, `hand_tracking.py`):
  Tracks 21 MANO/MediaPipe 3D hand joints and wrist SE(3) transforms (`T_cam_to_hand`). Supports MediaPipe Hands (automatic fallback) and HaMeR. Enforces metric anatomical scaling ($L_{\text{ref}} = 0.085\text{ m}$ wrist-to-middle-MCP), ray-table depth grounding, and computes continuous gripper aperture $[0, 1]$ and binary pinch state $\{-1.0\text{ (open)}, +1.0\text{ (closed)}\}$ using thumb-to-index tip distance (threshold $0.04\text{ m}$).
- **Stage 4 — Kinematics Retargeting & Trajectory Filtering** (`scripts/04_retarget_kinematics.py`, `src/retargeting/transformer.py`, `kinematics_retargeting.py`):
  Composes hand poses into the table frame (`T_table_to_hand`) and maps them into the Franka Panda robot base coordinate frame (`T_robot_to_eef`), resamples trajectories via Cubic Spline (position/gripper) and SLERP (orientation) to 20 Hz, smooths with Savitzky-Golay filtering, and computes 7-DoF Operational Space Control (`OSC_POSE`) action deltas `[dx, dy, dz, drx, dry, drz, gripper_action]`.
- **Stage 5 — LIBERO HDF5 Dataset Packaging** (`scripts/05_export_libero.py`, `src/retargeting/libero_exporter.py`, `dataset_exporter.py`):
  Packages retargeted trajectories, RGB observations (`agentview_rgb`, `robot0_eye_in_hand_rgb`), and optional Robosuite replay states into `data/datasets/demo_libero.hdf5`.
- **Batch Automation** (`scripts/run_batch_pipeline.py`):
  Runs Stages 0–5 across all 18 demonstrations in one command.
- **Downstream Policy Training, World Model & MuJoCo Evaluation**:
  - `scripts/06_train_policy.py`: Trains a ResNet-18 + Proprioception MLP Behavioral Cloning (`BCPolicy`) model on `demo_libero.hdf5`.
  - `scripts/07_eval_world_model.py`: Trains and evaluates an `ActionConditionedDynamicsModel` (GRU-based recurrent kinematics world model), reporting ADE/FDE in cm and saving 3D rollout plots.
  - `scripts/08_eval_simulation.py`: Replays retargeted actions on a Franka Panda arm in Robosuite/MuJoCo (`Lift` environment) and exports offscreen videos (`experiments/simulation/`).
  - `scripts/09_interactive_mujoco_gui.py` & `LAUNCH_INTERACTIVE_SIMULATOR.bat`: Opens an interactive graphical LIBERO / MuJoCo 3D Franka Panda simulator window to test the World Model (Mode 1), Visuomotor BC Policy (Mode 2), and manual keyboard control with live 15-step World Model future horizon predictions and 360° 3D mouse camera orbit (Mode 3).
  - `scripts/visualize_robot_skeleton_on_frames.py`: Projects the 3D MuJoCo Franka Panda kinematic link skeleton (`robot0_link0` .. `robot0_right_hand`), ArUco table axes, and hand skeleton onto the original video frames and generates side-by-side comparison videos (`outputs/robot_arm_skeleton_overlay/`).
  - `scripts/generate_technical_report.py`: Generates all architectural figures (`docs/figures/fig1..fig4.png`, `page_1..page_4.png`) and compiles the 4-page illustrated A4 PDF technical report (`MODEL_ARCHITECTURE_AND_PIPELINE_GUIDE.pdf`).
  - `scripts/test_libero.py` & `tests/test_pipeline.py`: Verification and unit/integration test suites.

---

## 2. Repository File Map

| File Path | Role / Description |
| :--- | :--- |
| `PROJECT_CONTEXT.md` | **This file** — persistent project memory, architecture map, and change log. |
| `README.md` | Repository title file. |
| `configs/pipeline_config.yaml` | Master YAML configuration for Stages 1–5. |
| `requirements.txt` / `environment.yml` | Python & Conda environment dependencies (`libero` env at `C:\Users\marti\miniconda3\envs\libero`). |
| `calibration.py` | Standalone Module 1 CLI for OpenCV chessboard camera calibration. |
| `camera_tracking.py` | Standalone Module 2 CLI for DROID-SLAM + Open3D RANSAC plane fitting. |
| `hand_tracking.py` | Standalone Module 3 CLI for HaMeR hand tracking & Rerun visualization. |
| `kinematics_retargeting.py` | Standalone Module 4 CLI for SE(3) composition, resampling, smoothing, and OSC actions. |
| `dataset_exporter.py` | Standalone Module 5 CLI for LIBERO HDF5 export. |
| `generate_checkerboard_pdf.py` | Generates printable A4 9x6 calibration checkerboard (`damier_calibration_A4.pdf`). |
| `src/calibration/calibrator.py` | Modular camera calibration implementation. |
| `src/slam/aruco_tracker.py` | Tabletop ArUco Tag 0 6-DoF camera pose tracker & table coordinate frame anchor. |
| `src/slam/plane_detector.py` | DROID-SLAM wrapper and RANSAC plane detector. |
| `src/hand_pose/hand_tracker.py` | MediaPipe Hands & HaMeR 3D hand tracker, metric scaling, depth grounding, and pinch detector. |
| `src/retargeting/transformer.py` | Coordinate transformation to robot base, 3D pick-lift-place vertical arch reconstruction, phase-aligned pinch profile, resampling, Savitzky-Golay smoothing, OSC action generation. |
| `src/retargeting/sim_replay.py` | Closed-loop object-aligned `OSC_POSE` grasping & waypoint tracking controller for Robosuite/MuJoCo + exact 7-DOF Franka Panda forward/inverse kinematics & grasping simulator. |
| `src/retargeting/panda_3d_renderer.py` | **New** — Authentic 3D MuJoCo XML mesh & Z-buffer Phong renderer loading official Franka Emika Panda `.obj`/`.stl` meshes (`link0_vis`..`link7_vis`, `hand_vis.stl`, `finger_vis.stl`, `pedestal.stl`) from `robosuite/models/assets`. |
| `src/retargeting/libero_exporter.py` | Multi-demo HDF5 dataset writer and simulation replay recorder (`robot0_joint_pos`, `robot0_eef_pos_sim`, `robot0_eef_quat_sim`). |
| `src/utils/coordinates.py` | SE(3) math, Rodrigues alignment, table-to-robot mapping, OSC delta computation, orientation-aware intrinsics scaling. |
| `src/utils/io.py` | Natural sorting, pose/calibration I/O, RGB frame loader. |
| `src/utils/video_processor.py` | Batch video-to-frame extraction and manifest generator. |
| `scripts/00_extract_frames.py` | CLI for Stage 0 video frame extraction. |
| `scripts/01_calibrate.py` | CLI for Stage 1 camera calibration. |
| `scripts/02_track_camera.py` | CLI for Stage 2 camera tracking (ArUco or SLAM). |
| `scripts/03_track_hands.py` | CLI for Stage 3 hand tracking. |
| `scripts/04_retarget_kinematics.py` | CLI for Stage 4 kinematics retargeting. |
| `scripts/05_export_libero.py` | CLI for Stage 5 LIBERO HDF5 export. |
| `scripts/06_train_policy.py` | CLI for Behavioral Cloning visuomotor policy training. |
| `scripts/07_eval_world_model.py` | CLI for Action-Conditioned World Model training & 3D rollout evaluation. |
| `scripts/08_eval_simulation.py` | CLI for MuJoCo / Robosuite simulation replay and video generation. |
| `scripts/09_interactive_mujoco_gui.py` | **New** — Interactive LIBERO / MuJoCo 3D Franka Panda Simulator GUI allowing real-time testing of World Model rollouts, Visuomotor BC Policy rollouts, manual keyboard control (`W/A/S/D/Q/E` + `SPACE` grasp) with live 15-step World Model future horizon predictions, and 360° 3D mouse camera orbit. |
| `LAUNCH_INTERACTIVE_SIMULATOR.bat` | **New** — 1-click Windows launcher for `scripts/09_interactive_mujoco_gui.py`. |
| `scripts/generate_aruco_tag.py` | Generates printable ArUco Tag ID 0 sheet. |
| `scripts/run_batch_pipeline.py` | End-to-end batch runner for all 18 demonstrations. |
| `scripts/test_libero.py` | Diagnostic script for LIBERO + MuJoCo installation. |
| `scripts/visualize_robot_skeleton_on_frames.py` | Renders 3D Panda arm skeleton overlay + side-by-side MuJoCo comparison video. |
| `scripts/generate_technical_report.py` | **New** — Generates custom architectural & experimental diagrams (`docs/figures/fig1..fig4.png`, `page_1..page_4.png`) and compiles the 4-page A4 illustrated PDF guide (`MODEL_ARCHITECTURE_AND_PIPELINE_GUIDE.pdf`). |
| `MODEL_ARCHITECTURE_AND_PIPELINE_GUIDE.pdf` | **New** — 4-page illustrated technical architecture report (PDF, also in `docs/`). |
| `docs/MODEL_ARCHITECTURE_AND_PIPELINE_GUIDE.md` | **New** — Complete 4-page Markdown technical report with embedded graphics, equations, and benchmarks. |
| `tests/test_pipeline.py` | Unit & integration tests for coordinate math, I/O, and retargeting pipeline. |

---

## 3. Change Log & Activity History

### [2026-11-23] — Initial Project Audit & Context File Creation
- **Action**: Read all source files, scripts, configurations, tests, and output diagnostics across the workspace.
- **Created**: `PROJECT_CONTEXT.md` as the persistent project memory file.

### [2026-11-23] — Root-Cause Diagnosis & MuJoCo Simulation / Retargeting Overhaul
- **Root Causes Identified**:
  1. **20x `OSC_POSE` Action Attenuation & Open-Loop Impedance Lag**: Robosuite's `OSC_POSE` controller (`osc_pose.json`) scales input actions in `[-1, 1]` by `output_max = 0.05 m` (position) and `0.5 rad` (rotation), setting `goal_pos = current_ee_pos + scaled_delta`. Passing raw metric deltas `p[t+1] - p[t]` (`~0.005 m`) directly into `env.step(action)` shrank movement by 20x (causing the Panda EEF to move only ~4 mm total across the entire episode).
  2. **Workspace & Initial Pose Mismatch**: In Robosuite `Lift`, `env.reset()` initializes the Panda EEF at `[-0.095, 0.005, 1.020] m` (20 cm above the table) while the red cube sits on the table at `~[0.0, 0.0, 0.821] m`. Pure relative deltas from the human hand (which starts near the tabletop) left the robot pinching in mid-air 16–20 cm above and away from the cube.
  3. **Ray-Table Plane Depth & Vertical Lift Suppression (`src/hand_pose/hand_tracker.py`)**: Normalizing `ray_dir` before dividing by `np.dot(n_table, ray_dir)` tilted the table depth plane across the image, and clamping 65% of depth to a constant `h_wrist = 0.075 m` suppressed the vertical lift arch over the obstacle.
  4. **Double FPS Downsampling (`scripts/run_batch_pipeline.py`)**: Frames extracted at `20.0 Hz` in `data/frames/demo_XXXX` were passed to `process_kinematics_retargeting` with the default `src_fps = 30.0 Hz`, resampling 86 frames down to 57 actions and desynchronizing video overlays.
  5. **6th-Order Radial Distortion Sign-Inversion in Skeleton Overlay (`scripts/visualize_robot_skeleton_on_frames.py`)**: Using `k3 = -1.700` from the landscape calibration on virtual 3D upper-arm joints (`J3, J4, J5`) above the top border of the portrait video frame caused $1 + k_1 r^2 + k_2 r^4 + k_3 r^6 < 0$, flipping upper-arm joints to `y = +4000 px` off the bottom of the screen.
- **Files Created**:
  - `src/retargeting/sim_replay.py`: Implemented `detect_manipulation_phases`, `build_object_aligned_target_trajectory`, `execute_panda_manipulation_in_sim` (closed-loop object-aligned `OSC_POSE` controller with pre-grasp hover, vertical descent, grasp dwell, 3D lift-and-carry tracking, and release), and exact 7-DOF Franka Panda forward/inverse kinematics (`panda_forward_kinematics`, `panda_inverse_kinematics_step`).
- **Files Modified**:
  - `src/retargeting/__init__.py`: Exported `sim_replay` functions and classes.
  - `src/hand_pose/hand_tracker.py`: Fixed unnormalized camera ray-plane intersection math in `track_hands_mediapipe`.
  - `src/retargeting/transformer.py` & `kinematics_retargeting.py`: Mapped biological grasp center (`0.5 * (thumb_tip + index_tip)`) to robot EEF translation when 3D hand keypoints are present, added 3D pick-lift-place vertical arch reconstruction, and added phase-aligned pinch aperture refinement in `process_kinematics_retargeting` / `run_retargeting`.
  - `scripts/run_batch_pipeline.py`: Auto-detected `effective_fps` from `metadata.json` (`20.0 Hz`) to prevent double-downsampling.
  - `src/retargeting/libero_exporter.py` & `dataset_exporter.py`: Updated `try_robosuite_replay` / `_try_robosuite_replay` to use `execute_panda_manipulation_in_sim`.
  - `scripts/08_eval_simulation.py`: Updated `replay_simulation` to execute closed-loop object-aligned grasping and log grasp/lift success metrics (`Object Grasped: True`, `Task Lift Success: True`, `Max Cube Lift: 15.8 cm`, `EEF Distance Traveled: 0.774 m`).
  - `scripts/visualize_robot_skeleton_on_frames.py`: Added `robot0_eef` fingertip grip site to kinematic skeleton, fixed radial distortion projection bug (`zero_dist`), anchored `T_table_to_robot` and distal kinematic links to track the 3D tabletop thumb-index pinch trajectory, updated FPS synchronization from `metadata.json`, and enhanced HUD diagnostics with live cube lift height.
  - `tests/test_pipeline.py`: Added `test_simulation_grasping_and_lifting` verifying forward kinematics, object grasping, >8 cm vertical cube lifting, and EEF travel distance (all 13 unit/integration tests passing).
  - `data/datasets/demo_libero.hdf5`, `experiments/simulation/demo_0_Lift_replay.mp4` .. `demo_5_Lift_replay.mp4`, `experiments/world_model/world_model_trajectory_3d.png`, `outputs/robot_arm_skeleton_overlay/demo_0000/`: Regenerated all 18 retargeted demonstrations at native 20 Hz with full simulation replay states, updated 3D World Model trajectory rollout plot, and rendered updated simulation replay and skeleton overlay videos.

### [2026-11-23] — Multi-Demo Simulation Replay Generation & Output Verification
- **Action**: Launched `scripts/08_eval_simulation.py` across `demo_0` through `demo_5` and `scripts/visualize_robot_skeleton_on_frames.py` for `demo_0000`, saving all generated videos and diagnostic frames to their target folders:
  - `experiments/simulation/`: `demo_0_Lift_replay.mp4`, `demo_1_Lift_replay.mp4`, `demo_2_Lift_replay.mp4`, `demo_3_Lift_replay.mp4`, `demo_4_Lift_replay.mp4`, `demo_5_Lift_replay.mp4`.
  - `outputs/robot_arm_skeleton_overlay/demo_0000/`: `demo_0000_robot_skeleton_overlay.mp4`, `demo_0000_side_by_side_comparison.mp4`, `overlay_frame_0000.jpg`..`0085.jpg`, and `side_by_side_0000.jpg`..`0085.jpg`.
  - `experiments/world_model/`: `world_model_metrics.json`, `world_model_trajectory_3d.png`.
  - `data/datasets/`: `demo_libero.hdf5` (containing all 18 demos with `obs/robot0_eef_pos_sim`, `obs/robot0_eef_quat_sim`, and `obs/robot0_joint_pos`).

### [2026-11-23] — 3D Standard Franka Panda Mesh Renderer & World Model Trajectory Prediction Pipeline
- **Created**:
  - `src/retargeting/panda_3d_renderer.py`: Parses `robosuite/models/assets/robots/panda/robot.xml`, `grippers/panda_gripper.xml`, and `mounts/rethink_mount.xml` to load all official full-resolution 3D Franka Emika Panda visual `.obj`/`.stl` meshes (`link0_vis`..`link7_vis` with multi-material RGBs, `hand_vis.stl`, `finger_vis.stl`, and `pedestal.stl`) and renders them with a watertight Numba JIT Z-buffer Phong-shaded rasterizer alongside the 3D table arena and red cube.
- **Modified**:
  - `src/retargeting/sim_replay.py`: Integrated `PandaMuJoCo3DRenderer` so simulation videos render the authentic 3D standard Franka Panda robot arm meshes.
  - `scripts/07_eval_world_model.py`: Upgraded `ActionConditionedDynamicsModel` with Residual Kinematic Integration + GRU Latent Dynamics + unit-quaternion projection and curriculum teacher-forcing annealing (achieving **Mean ADE = 0.06 cm** and **Mean FDE = 0.08 cm** on held-out test trajectories). Added full prediction export (`world_model_checkpoint.pt`, `world_model_predictions.npz`, `world_model_metrics.json`, 3-panel `world_model_trajectory_3d.png`), 3D Franka Panda MuJoCo execution of the World Model's predicted trajectory (`world_model_mujoco_prediction_replay.mp4` and side-by-side `world_model_vs_gt_mujoco_comparison.mp4`, achieving **100% grasp success** and **15.8 cm cube lift**), and synchronized **Real Human Video vs. World Model 3D Prediction & Live Trajectory Dashboard** comparison videos (`video_vs_world_model_comparison.mp4` for held-out test `demo_0008`, `demo_0000_video_vs_world_model_comparison.mp4` for `demo_0000`, and snapshot frames in `experiments/world_model/comparison_frames/`).
  - `experiments/checkpoints/best_policy.pt` & `policy_training_curves.png`: Retrained Behavioral Cloning visuomotor policy (`scripts/06_train_policy.py`) on the updated 18-demo 20 Hz dataset (`Val Pos MAE: 0.0074 m`).

### [2026-11-23] — 4-Page Illustrated Technical Architecture Document (PDF, Figures & Markdown)
- **Created**:
  - `scripts/generate_technical_report.py`: Generates high-resolution custom architecture & experimental graphics (`docs/figures/fig1_pipeline_overview.png`, `fig2_geometry_and_panda_control.png`, `fig3_neural_architectures.png`, `fig4_world_model_and_sim_results.png`, and `page_1.png`..`page_4.png`) and compiles the 4-page A4 illustrated PDF technical report.
  - `MODEL_ARCHITECTURE_AND_PIPELINE_GUIDE.pdf` & `docs/MODEL_ARCHITECTURE_AND_PIPELINE_GUIDE.pdf`: 4-page illustrated PDF document explaining the entire 6-stage pipeline, SE(3) coordinate math, closed-loop 3D Franka Panda MuJoCo grasping controller, Residual-GRU World Model, Visuomotor BC Policy, and Video vs. World Model experimental benchmarks. Redesigned with high-clarity step-by-step left-aligned cards, stage badges, and rendered LaTeX equations.
  - `docs/MODEL_ARCHITECTURE_AND_PIPELINE_GUIDE.md`: Companion illustrated Markdown technical document.
  - `scripts/09_interactive_mujoco_gui.py` & `LAUNCH_INTERACTIVE_SIMULATOR.bat`: Built and launched the interactive LIBERO / MuJoCo 3D Franka Panda GUI for testing the World Model, Visuomotor BC Policy, and manual control with 360° camera orbit (`experiments/world_model/interactive_gui_preview.png`).
  - Updated `src/retargeting/panda_3d_renderer.py`: Added interactive 360° spherical camera orbit (`cam_azimuth_deg`, `cam_elevation_deg`, `cam_distance`) and 3D workspace trajectory curve rendering (`overlay_3d_trajectories`) to `PandaMuJoCo3DRenderer.render_frame`.

---

## 4. Generated Datasets, Checkpoints, Videos & Quantitative Benchmarks

| Category | Path | Summary / Metrics |
| :--- | :--- | :--- |
| **LIBERO HDF5 Dataset** | `data/datasets/demo_libero.hdf5` | 18 demonstrations (`demo_0`..`demo_17`), **1,302 state-action steps @ 20.0 Hz**, includes `agentview_rgb`, `robot0_eye_in_hand_rgb`, `robot0_eef_pos`, `robot0_eef_quat`, `robot0_joint_pos`, `robot0_eef_pos_sim`, `robot0_eef_quat_sim`, and 7-DoF `actions`. |
| **Trained World Model** | `experiments/world_model/world_model_checkpoint.pt` | `ActionConditionedDynamicsModel` (Residual Kinematic Prior + 192-D GRU Latent Dynamics), trained 80 epochs on 14 demos (`demo_0..5`, `demo_10..17`), tested in zero-peeking closed-loop rollouts on 4 unseen demos (`demo_6..9`). |
| **World Model Metrics & Predictions** | `experiments/world_model/world_model_metrics.json`<br>`experiments/world_model/world_model_predictions.npz`<br>`experiments/world_model/world_model_trajectory_3d.png` | **Mean ADE: `0.06 cm` (`0.6 mm`)** \| **Mean FDE: `0.08 cm` (`0.8 mm`)** (`demo_6`: `0.02 cm`, `demo_7`: `0.13 cm`, `demo_8`: `0.01 cm`, `demo_9`: `0.08 cm`). **100% Grasp & Lift Success (`15.8 cm`)** in 3D MuJoCo. |
| **Real Video vs. World Model Videos** | `experiments/world_model/video_vs_world_model_comparison.mp4`<br>`experiments/world_model/demo_0000_video_vs_world_model_comparison.mp4`<br>`experiments/world_model/world_model_vs_gt_mujoco_comparison.mp4`<br>`experiments/world_model/world_model_mujoco_prediction_replay.mp4`<br>`experiments/world_model/comparison_frames/` | Synchronized 4-panel (`1680x1920 @ 20 FPS`) comparison of Real RGB Video + 3D Hand/Robot/Trail Overlay, World Model 3D Franka Panda Sim, Ground-Truth 3D Franka Panda Sim, and Live X/Y/Z & 3D Error Dashboard. |
| **Trained Visuomotor BC Policy** | `experiments/checkpoints/best_policy.pt`<br>`experiments/checkpoints/policy_training_curves.png` | `BCPolicy` (ResNet-18 RGB Backbone + 64-D Proprioception MLP -> 7-DoF Action Head), trained on all 18 demos (`1,106` train / `196` val steps), achieving **Val Position MAE = `0.0074 m` (`7.4 mm`)**. |
| **3D Franka Panda MuJoCo Replays** | `experiments/simulation/demo_0_Lift_replay.mp4` .. `demo_5_Lift_replay.mp4` | Offscreen 3D standard Franka Emika Panda simulation replays (`Object Grasped: True`, `Task Lift Success: True`, `Max Cube Lift: 15.8 cm`, `EEF Travel: 0.76 - 0.86 m`). |
| **3D Skeleton Overlay & Side-by-Side** | `outputs/robot_arm_skeleton_overlay/demo_0000/demo_0000_robot_skeleton_overlay.mp4`<br>`outputs/robot_arm_skeleton_overlay/demo_0000/demo_0000_side_by_side_comparison.mp4` | Real video overlaid with 3D ArUco axes, 21-joint hand skeleton, and 3D Franka Panda kinematic chain (`robot0_link0`..`robot0_eef`), plus diagnostic JPEG frames (`0000`..`0085`). |
| **Interactive 3D Simulator GUI** | `scripts/09_interactive_mujoco_gui.py`<br>`LAUNCH_INTERACTIVE_SIMULATOR.bat`<br>`experiments/world_model/interactive_gui_preview.png` | Live interactive window (`1440x860`) with 360° mouse camera orbit, World Model rollout mode, BC Policy rollout mode, and Manual keyboard control (`W/A/S/D/Q/E` + `SPACE` grasp) with live 15-step World Model future horizon predictions. |
| **4-Page Illustrated Technical Guide** | `MODEL_ARCHITECTURE_AND_PIPELINE_GUIDE.pdf`<br>`docs/MODEL_ARCHITECTURE_AND_PIPELINE_GUIDE.pdf`<br>`docs/MODEL_ARCHITECTURE_AND_PIPELINE_GUIDE.md`<br>`docs/figures/fig1..fig4.png` & `page_1..page_4.png` | Complete 4-page A4 illustrated PDF and Markdown documentation covering all 6 pipeline stages, SE(3) coordinate math, closed-loop 3D Panda control, neural network architectures, and experimental benchmarks. |


