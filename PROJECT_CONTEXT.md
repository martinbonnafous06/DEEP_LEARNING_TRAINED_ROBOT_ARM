# PROJECT_CONTEXT.md — Project Memory & Living Documentation

> **Living Context File**: This document serves as the persistent architectural memory and change log for the **Egocentric Video to LIBERO Retargeting & Robot Manipulation Pipeline** (`C:\Users\marti\Desktop\Humanoid`).
> **Rule**: Every time a file is created, modified, or an experiment/fix is executed, this document MUST be updated to reflect the current state of the project.

---

## 1. Project Overview

This repository implements an end-to-end **Human-to-Robot Retargeting and Visuomotor Learning Pipeline** that converts monocular RGB human demonstration videos into **Franka Emika Panda** robot manipulation trajectories in **MuJoCo / Robosuite / LIBERO**, exports benchmark-compliant HDF5 datasets, and trains/evaluates Behavioral Cloning (BC) policies and Action-Conditioned World Models.

### Pipeline Stages
- **Stage 0 — Multi-Video Frame Extraction** (`scripts/00_extract_frames.py`, `src/utils/video_processor.py`):
  Extracts raw videos (`data/raw_videos/demo_01.mp4` .. `demo_18.mp4`, 1080x1920 portrait @ 60 FPS) into structured frame directories (`data/frames/demo_0000` .. `demo_0017` @ 20 Hz, 1320 total frames) with `metadata.json`, `timestamps.npy`, and global `dataset_manifest.json` / `.csv`.
- **Stage 1 — Camera Intrinsic Calibration** (`scripts/01_calibrate.py`, `src/calibration/calibrator.py`, `scripts/generate_checkerboard_pdf.py`):
  Calibrates camera intrinsics (`K`, `dist_coeffs`) using a 9x6 internal-corner chessboard (25 mm squares) from `data/calib_images/` (4000x2252 landscape) and exports `calib.txt` (DROID-SLAM format), `calibration.npz`, and `calibration.json` into `data/calibration/` (with printable targets in `data/calibration/targets/`). Includes orientation-aware intrinsic adaptation (`adapt_camera_intrinsics` in `src/utils/coordinates.py`) when video orientation (portrait 1080x1920) differs from calibration photos (landscape 4000x2252).
- **Stage 2 — Camera Tracking & Table Plane Estimation** (`scripts/02_track_camera.py`, `src/slam/aruco_tracker.py`, `src/slam/plane_detector.py`, `scripts/generate_aruco_tag.py`):
  Supports two modes:
  1. **ArUco Tabletop Fiducial Tracking (Default)**: Uses ArUco Tag ID 0 (`DICT_ARUCO_ORIGINAL`, 50 mm) placed flat on the table to solve exact metric 6-DoF camera poses (`T_world_to_cam`) via `cv2.solvePnP(..., SOLVEPNP_IPPE_SQUARE)` and anchor the tabletop plane at $Z = 0$ (`T_world_to_table = I_4`).
  2. **Monocular Visual SLAM Fallback**: Invokes DROID-SLAM and fits a table plane via Open3D RANSAC (`fit_table_plane_ransac`).
- **Stage 3 — 3D Hand Pose & Gripper State Estimation** (`scripts/03_track_hands.py`, `src/hand_pose/hand_tracker.py`):
  Tracks 21 MANO/MediaPipe 3D hand joints and wrist SE(3) transforms (`T_cam_to_hand`). Supports MediaPipe Hands (automatic fallback) and HaMeR. Enforces metric anatomical scaling ($L_{\text{ref}} = 0.085\text{ m}$ wrist-to-middle-MCP), ray-table depth grounding, and computes continuous gripper aperture $[0, 1]$ and binary pinch state $\{-1.0\text{ (open)}, +1.0\text{ (closed)}\}$ using thumb-to-index tip distance (threshold $0.04\text{ m}$).
- **Stage 4 — Kinematics Retargeting & Trajectory Filtering** (`scripts/04_retarget_kinematics.py`, `src/retargeting/transformer.py`):
  Composes hand poses into the table frame (`T_table_to_hand`) and maps them into the Franka Panda robot base coordinate frame (`T_robot_to_eef`), resamples trajectories via Cubic Spline (position/gripper) and SLERP (orientation) to 20 Hz, smooths with Savitzky-Golay filtering, and computes 7-DoF Operational Space Control (`OSC_POSE`) action deltas `[dx, dy, dz, drx, dry, drz, gripper_action]`.
- **Stage 5 — LIBERO HDF5 Dataset Packaging** (`scripts/05_export_libero.py`, `src/retargeting/libero_exporter.py`):
  Packages retargeted trajectories, RGB observations (`agentview_rgb`, `robot0_eye_in_hand_rgb`), and optional Robosuite replay states into `data/datasets/demo_libero.hdf5`.
- **Batch Automation** (`scripts/run_batch_pipeline.py`):
  Runs Stages 0–5 across all 18 demonstrations in one command.
- **Downstream Policy Training, World Model & MuJoCo Evaluation**:
  - `scripts/06_train_policy.py`: Trains the High-Throughput Multi-Dataset Dual-Head Visuomotor Behavioral Cloning (`BCPolicy`) model on `demo_libero.hdf5` (or any external HDF5 dataset).
  - `scripts/07_eval_world_model.py`: Trains and evaluates an `ActionConditionedDynamicsModel` (GRU-based recurrent kinematics world model), reporting ADE/FDE in cm and saving 3D rollout plots.
  - `scripts/08_eval_simulation.py`: Replays retargeted actions on a Franka Panda arm in Robosuite/MuJoCo (`Lift` environment) and exports offscreen videos (`experiments/simulation/`).
  - `scripts/09_interactive_mujoco_gui.py` & `LAUNCH_INTERACTIVE_SIMULATOR.bat`: Opens an interactive graphical LIBERO / MuJoCo 3D Franka Panda simulator window to test the World Model (Mode 1), Visuomotor BC Policy (Mode 2), and manual keyboard control with live 15-step World Model future horizon predictions and 360° 3D mouse camera orbit (Mode 3), plus live on-screen `[O] Open Folder`, `[V] Pick Video`, and `[M] Switch Model` buttons.
  - `scripts/10_all_in_one_pipeline_gui.py` & `LAUNCH_ALL_IN_ONE_PIPELINE.bat`: Unified All-in-One Pipeline & 10-Epoch Live Neural Network Training GUI (`1460x900`) allowing users to select any raw videos folder, run Stages 0–6 end-to-end in one click, manage per-dataset registered models (`experiments/checkpoints/model_registry.json`), and monitor a live training dashboard updating every 10 epochs (`Epoch 10, 20, ..., 80`).
  - `scripts/visualize_robot_skeleton_on_frames.py`: Projects the 3D MuJoCo Franka Panda kinematic link skeleton (`robot0_link0` .. `robot0_right_hand`), ArUco table axes, and hand skeleton onto the original video frames and generates side-by-side comparison videos (`experiments/skeleton_overlays/`).
  - `scripts/generate_technical_report.py`: Generates all architectural figures (`docs/figures/fig1..fig4.png`, `page_1..page_4.png`) and compiles the 4-page illustrated A4 PDF technical report (`docs/MODEL_ARCHITECTURE_AND_PIPELINE_GUIDE.pdf`).
  - `tests/test_pipeline.py`: Unit & integration test suite (14 tests).

---

## 2. Repository File Map

| File Path | Role / Description |
| :--- | :--- |
| `PROJECT_CONTEXT.md` | **This file** — persistent project memory, architecture map, and change log. |
| `README.md` | Repository title file. |
| `configs/pipeline_config.yaml` | Master YAML configuration for Stages 1–5. |
| `requirements.txt` / `environment.yml` | Python & Conda environment dependencies (`libero` env at `C:\Users\marti\miniconda3\envs\libero`). |
| `LAUNCH_INTERACTIVE_SIMULATOR.bat` | 1-click Windows launcher for `scripts/09_interactive_mujoco_gui.py`. |
| `src/calibration/calibrator.py` | Modular camera calibration implementation. |
| `src/slam/aruco_tracker.py` | Tabletop ArUco Tag 0 6-DoF camera pose tracker & table coordinate frame anchor. |
| `src/slam/plane_detector.py` | DROID-SLAM wrapper and RANSAC plane detector. |
| `src/hand_pose/hand_tracker.py` | MediaPipe Hands & HaMeR 3D hand tracker, metric scaling, depth grounding, and pinch detector. |
| `src/retargeting/transformer.py` | Coordinate transformation to robot base, 3D pick-lift-place vertical arch reconstruction, phase-aligned pinch profile, resampling, Savitzky-Golay smoothing, OSC action generation. |
| `src/retargeting/sim_replay.py` | Closed-loop object-aligned `OSC_POSE` grasping & waypoint tracking controller for Robosuite/MuJoCo + exact 7-DOF Franka Panda forward/inverse kinematics & grasping simulator. |
| `src/retargeting/panda_3d_renderer.py` | Authentic 3D MuJoCo XML mesh & Z-buffer Phong renderer loading official Franka Emika Panda `.obj`/`.stl` meshes (`link0_vis`..`link7_vis`, `hand_vis.stl`, `finger_vis.stl`, `pedestal.stl`) from `robosuite/models/assets`. |
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
| `scripts/06_train_policy.py` | CLI for High-Throughput Multi-Dataset Behavioral Cloning visuomotor policy training. |
| `scripts/07_eval_world_model.py` | CLI for Action-Conditioned World Model training & 3D rollout evaluation. |
| `scripts/08_eval_simulation.py` | CLI for MuJoCo / Robosuite simulation replay and video generation. |
| `scripts/09_interactive_mujoco_gui.py` | Interactive LIBERO / MuJoCo 3D Franka Panda Simulator GUI allowing real-time testing of World Model rollouts, Visuomotor BC Policy rollouts, manual keyboard control (`W/A/S/D/Q/E` + `SPACE` grasp) with live 15-step World Model future horizon predictions, and 360° 3D mouse camera orbit. |
| `scripts/10_all_in_one_pipeline_gui.py` | **New** — All-in-One Pipeline & 10-Epoch Live Neural Network Training GUI allowing users to select the Raw Videos folder, run Stages 0–6 in one click, and monitor a live training panel & convergence plot that updates every 10 epochs (`Epoch 10, 20, ..., 80`). |
| `LAUNCH_ALL_IN_ONE_PIPELINE.bat` | **New** — 1-click Windows launcher for `scripts/10_all_in_one_pipeline_gui.py`. |
| `scripts/generate_checkerboard_pdf.py` | **Camera Generator 1** — Generates printable A4 9x6 calibration checkerboard (`data/calibration/targets/damier_calibration_A4.pdf`). |
| `scripts/generate_aruco_tag.py` | **Camera Generator 2** — Generates printable ArUco Tag ID 0 sheet (`data/calibration/targets/aruco_tag_0_original.png`). |
| `scripts/run_batch_pipeline.py` | End-to-end batch runner for all 18 demonstrations. |
| `scripts/visualize_robot_skeleton_on_frames.py` | Renders 3D Panda arm skeleton overlay + side-by-side MuJoCo comparison video (`experiments/skeleton_overlays/`). |
| `scripts/generate_technical_report.py` | Generates custom architectural & experimental diagrams (`docs/figures/fig1..fig4.png`, `page_1..page_4.png`) and compiles the 4-page A4 illustrated PDF guide (`docs/MODEL_ARCHITECTURE_AND_PIPELINE_GUIDE.pdf`). |
| `docs/MODEL_ARCHITECTURE_AND_PIPELINE_GUIDE.pdf` | 4-page illustrated technical architecture report (PDF). |
| `docs/MODEL_ARCHITECTURE_AND_PIPELINE_GUIDE.md` | Complete 4-page Markdown technical report with embedded graphics, equations, and benchmarks. |
| `tests/test_pipeline.py` | Unit & integration tests for coordinate math, I/O, retargeting pipeline, and multi-dataset neural network training. |

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

### [2026-11-24] — Dual-Head Decoupled Visuomotor BC Policy Overhaul & Simulator Preprocessing Fix
- **Root Causes Identified**:
  1. **Merged Loss Scale Imbalance & Epoch-1 Early Stopping (`scripts/06_train_policy.py`)**: A single 7-DoF linear output head and merged loss (`total_loss = loss_pos + loss_rot + loss_grip`) allowed the $\pm 1.0$ binary gripper loss ($\sim 0.23$) to dominate the $\sim 0.005\text{ m}$ millimeter position delta loss ($\sim 0.00006$) by $>3,000\times$, causing `val_loss` to stop saving checkpoints after Epoch 1 (`161.5 cm` cumulative drift on `demo_8`).
  2. **RGB Normalization Mismatch (`scripts/09_interactive_mujoco_gui.py`)**: `06_train_policy.py` trained on `[0, 1]` scaled RGB tensors (`raw_rgb / 255.0`), whereas `09_interactive_mujoco_gui.py` applied ImageNet mean/std normalization (`[-2.1, +2.6]`).
- **Fixes Applied**:
  - `scripts/06_train_policy.py`: Refactored `BCPolicy` into a **Dual-Head Decoupled Architecture** with two independent output branches, two separate loss functions, and two independent `AdamW` optimizers (`optimizer_motion` for the 6-DoF `motion_head` using centimeter-scale Smooth-L1 delta + sequence cumulative trajectory integration loss; `optimizer_gripper` for the 1-DoF `gripper_head` with detached visual features and BCE+MSE loss), checkpointing the best motion branch (`Epoch 76`) and best gripper branch (`Epoch 54`) independently without redundant optimizer buffers (`45.8 MB`).
  - `scripts/09_interactive_mujoco_gui.py`: Aligned BC Policy RGB preprocessing to `[0, 1]` scaling (`float() / 255.0`), matching `06_train_policy.py` bit-for-bit.
  - `experiments/checkpoints/best_policy.pt` & `experiments/checkpoints/policy_training_curves.png`: Retrained for 80 epochs, reducing Validation Position MAE by **$8.2\times$** (`0.0074 m` $\to$ **`0.0009 m` / `0.90 mm`**), achieving **`97.9%` validation gripper accuracy**, **`1.27 cm` mean 70-step rollout ADE**, and reducing `demo_8` cumulative rollout drift from `161.49 cm` down to **`1.18 cm`** ($137\times$ reduction).

### [2026-11-24] — Enhanced Dual-Head BC Policy & Multi-Dataset / Large-Scale Efficiency Engine
- **Architectural, Multi-Dataset & Scalability Upgrades (`scripts/06_train_policy.py`, `scripts/07_eval_world_model.py`, `tests/test_pipeline.py`)**:
  1. **2D Spatial Softmax Visual Backbone (`SpatialSoftmax2D` + `SpatialResNet18Encoder`)**: Replaced pure global average pooling at the end of ResNet-18 with a 64-keypoint learnable-temperature 2D Spatial Softmax extracting expected $(u_k, v_k) \in [-1, 1]^2$ 2D pixel coordinates fused with 512-D global features into a 192-D visual embedding.
  2. **Multi-Frequency Sinusoidal Proprioceptive Encoding (`FourierProprioEncoding`)**: Lifted raw EEF pose across 6 octave frequency bands ($2^k \pi, k \in \{0..5\}$) to eliminate MLP spectral bias at sub-millimeter spatial scales.
  3. **Pre-Norm Residual MLP Blocks (`ResidualMLPBlock`)**: Upgraded both the Motion Branch and Gripper Branch with zero-initialized Pre-Norm LayerNorm + SiLU Residual MLP blocks.
  4. **Universal HDF5 Schema Auto-Discovery & Registered Normalization Buffers (`resolve_hdf5_schema`)**: Automatically detects RGB keys (`agentview_rgb`, `agentview_image`, `image`, plus optional multi-view wrist camera `--use_wrist_cam`), state keys (`robot0_eef_pos`/`quat`, `ee_pos`/`ee_ori`, `states`, `proprio`), arbitrary image resolutions (`84x84`, `128x128`, `224x224`), arbitrary state/action dimensions, and registers `proprio_mean`, `proprio_std`, `motion_scale_buf`, and `grip_range_buf` inside the model checkpoint.
  5. **Large-Dataset Memory & GPU Throughput Optimization (`2.1x` Faster Training)**:
     - Compact `uint8` RAM storage ($4\times$ lower RAM footprint than `float32`) with automatic fallback to **Lazy Chunked HDF5 Streaming** (`--lazy_hdf5`) when datasets exceed `60,000` frames.
     - Pre-caches the frozen ResNet-18 early stem (`conv1..layer2`, $>65\%$ of CNN FLOPs) in consistent `.eval()` mode and evaluates full-trajectory cumulative integration + FDE + acceleration smoothness in one vectorized `BatchedTrajectoryBuffer` GPU pass (`177.1 s` total / `2.21 s/epoch`, down from `370.3 s`).
     - Vectorized `train_world_model` in `scripts/07_eval_world_model.py` using padded 3D trajectory batches `(B_traj, T_max, D)` with boolean validity masks.
     - Added `test_multi_dataset_and_lazy_hdf5_policy_and_world_model` in `tests/test_pipeline.py` (all 14 unit/integration tests passing).
- **Quantitative Benchmark Improvements (`experiments/checkpoints/best_policy.pt`, `policy_metrics.json`, `policy_training_curves.png`)**:
  - **70-Step Closed-Loop Rollout ADE**: Improved **$4.5\times$** from `1.27 cm` $\to$ **`0.28 cm` (`2.8 mm`)** ($576\times$ better than the original `161.5 cm` single-head baseline).
  - **70-Step Closed-Loop Rollout FDE**: Achieved **`0.40 cm` (`4.0 mm`)** final placement error.
  - **Validation Position MAE**: Improved from `0.90 mm` (`0.0009 m`) $\to$ **`0.58 mm` (`0.000581 m`)** ($12.7\times$ reduction vs. original `7.4 mm`).
  - **Validation Gripper Accuracy**: Improved from `97.9%` $\to$ **`99.5%`**.
  - Updated `scripts/generate_technical_report.py`, `docs/MODEL_ARCHITECTURE_AND_PIPELINE_GUIDE.md`, `docs/MODEL_ARCHITECTURE_AND_PIPELINE_GUIDE.pdf`, and `experiments/world_model/interactive_gui_preview.png`.

### [2026-11-24] — Repository Cleanup, Redundant Program Removal & Data Directory Consolidation
- **Programs Suppressed (Deleted)**:
  - Removed legacy duplicate standalone root scripts (`calibration.py`, `camera_tracking.py`, `hand_tracking.py`, `kinematics_retargeting.py`, `dataset_exporter.py`) and obsolete diagnostic script (`scripts/test_libero.py`), keeping all modular implementations in `src/` and numbered pipeline entry points in `scripts/00_extract_frames.py`..`scripts/09_interactive_mujoco_gui.py`.
- **Camera Pattern Generators Preserved & Consolidated in `scripts/`**:
  - `scripts/generate_checkerboard_pdf.py` (A4 9x6 calibration checkerboard generator, outputting to `data/calibration/targets/damier_calibration_A4.pdf`).
  - `scripts/generate_aruco_tag.py` (ArUco Tag ID 0 fiducial sheet generator, outputting to `data/calibration/targets/aruco_tag_0_original.png` and `aruco_tag_0_raw.png`).
- **Data & Experiment Directory Reorganization**:
  - Created `data/calibration/targets/` and moved `damier_calibration_A4.pdf` + generated `aruco_tag_0_original.png` and `aruco_tag_0_raw.png` into it.
  - Created `data/debug_overlays/aruco_demo_0000/` and `data/debug_overlays/hand_demo_0000/`, migrated all camera and hand tracking debug JPEGs out of `data/slam/` and `data/hand_poses/`, and removed duplicate root-level single-demo files (`data/slam/poses.npy`, `data/slam/table_transform.npz`, `data/hand_poses/hand_poses.npz`, `data/hand_poses/hand_poses_cam.npy`, `data/hand_poses/gripper_states.npy`, `data/retargeted/retargeted.npz`, `data/retargeted/trajectory_world_20hz.npy`).
  - Created `experiments/skeleton_overlays/demo_0000/`, migrated all 3D robot skeleton overlay videos and frames from `outputs/robot_arm_skeleton_overlay/demo_0000/`, and suppressed the obsolete top-level `outputs/` directory.
  - Consolidated the 4-page technical report PDF at `docs/MODEL_ARCHITECTURE_AND_PIPELINE_GUIDE.pdf` (removing the duplicate root copy).

### [2026-11-24] — All-in-One 10-Epoch Live Trainer GUI, Per-Dataset Model Registry & Continual Fine-Tuning
- **Created**:
  - `scripts/10_all_in_one_pipeline_gui.py` & `LAUNCH_ALL_IN_ONE_PIPELINE.bat`: Unified graphical control center (`1460x900`) allowing users to pick any Raw Videos folder, run Stages 0–6 in one click, and monitor a live training panel (4 KPI cards, 10-epoch milestone table, and live dual-panel convergence curves) that updates every 10 epochs (`Epoch 10, 20, ..., 80`).
  - `experiments/checkpoints/model_registry.json` & `experiments/checkpoints/registry/demo_libero/`: Non-destructive per-dataset model registry.
- **Upgraded**:
  - `scripts/06_train_policy.py` & `scripts/07_eval_world_model.py`: Added `archive_and_register_model` (automatically saves each dataset's model under `experiments/checkpoints/registry/<model_name>/` and versions previous runs as `best_policy_v1.pt`, `best_policy_v2.pt`, etc. so no trained model is ever overwritten) and `load_compatible_state_dict` / `--resume_from` for continual fine-tuning across datasets.
  - `scripts/09_interactive_mujoco_gui.py`: Added `--dataset`, `--test_dir` (defaulting to the newly created `data/test_videos/` folder), `--choose_folder`, `--bc_ckpt`, and `--wm_ckpt` arguments, plus live on-screen `[O] Open Folder`, `[V] Pick Video`, and `[M] Switch Model` buttons so users can pick any folder of `.mp4` test videos (or a single `.mp4` video) directly inside the 3D Franka Panda simulator (automatically extracting, tracking, and retargeting new test videos on the fly) and cycle between saved trained model profiles (`raw_2`, `demo_libero`, etc.).
  - `tests/test_pipeline.py`: Extended `test_multi_dataset_and_lazy_hdf5_policy_and_world_model` to test per-dataset registry storage, versioned backup preservation (`best_policy_v1.pt`), and continual fine-tuning across datasets.

### [2026-11-25] — Simulator Live Model Switching Fix, Moving-Camera 3D Hand/Object Grounding & BC Policy Test Rollout Alignment (`demo_libero` & `raw_2`)
- **Root Causes Identified**:
  1. **Model Switch Auto-Reverting in Simulator (`scripts/09_interactive_mujoco_gui.py`)**: `_trigger_action("switch_model")` loaded the next registered model and then called `self._load_demo_episode(self.demo_idx)` to recompute the rollout, which unconditionally called `self._auto_select_unique_model_for_dataset(ep_dataset_tag)` and immediately reverted `self.active_model_name` back to the episode's `source_dataset` tag.
  2. **Generic Folder Name Collision (`Nouveau dossier`)**: In `scripts/09_interactive_mujoco_gui.py` and `scripts/10_all_in_one_pipeline_gui.py`, any external test folder named `Nouveau dossier` (such as `Downloads\Nouveau dossier`) was mapped to `safe_tag = "demo_libero"`, overwriting `data/frames/demo_0000..0002` with 3 external test videos.
  3. **Moving-Camera Depth Explosion & Occlusion Jumps in `raw_2` (`src/hand_pose/hand_tracker.py`, `src/slam/aruco_tracker.py`)**:
     - In `raw_2`, the moving camera approaches within `0.19 – 0.28 m` of the table, causing the hand's tabletop plane depth `z_plane` to dip below `0.20 m` (`0.15 – 0.19 m`). The previous check `if 0.2 < z_plane < 1.8:` rejected `z_plane <= 0.20 m` and fell back to 2D wrist-to-MCP apparent depth `z_apparent`, which exploded to **1.5–2.9 m** due to 2D palm foreshortening (`demo_0004`, `demo_0006`, `demo_0008`, `demo_0018`).
     - During moving-camera tag occlusions, `aruco_tracker.py` previously froze camera poses with a step-hold (`last_valid`) instead of smooth SE(3) interpolation and did not filter planar PnP mirror flips (`demo_0017`).
  4. **BC Policy Test Rollout Drift vs. World Model (`scripts/06_train_policy.py`, `scripts/09_interactive_mujoco_gui.py`)**:
     - High-frequency Fourier encoding on absolute `proprio` coordinates and drifting `BatchNorm2d` buffers caused cumulative drift (`24–86 cm`) when testing on unseen moving-camera videos with shifted tabletop coordinates, and `09_interactive_mujoco_gui.py` re-detected separate pick/place phases on `bc_states` instead of applying the model's 3D prediction delta relative to the video's true object pick location.
- **Fixes Applied**:
  - `src/slam/aruco_tracker.py`: Added `cv2.solvePnPGeneric(..., SOLVEPNP_IPPE_SQUARE)` planar flip disambiguation, single-frame outlier spike rejection, continuous SE(3) interpolation (linear translation + SLERP rotation) across occluded frames, and Savitzky-Golay camera trajectory smoothing.
  - `src/hand_pose/hand_tracker.py`: Lowered valid tabletop depth bound to `0.08 m`, added multi-bone palm extent estimation (`(0,5), (0,9), (0,13), (0,17), (5,17)`) clamped tightly around the tabletop plane (`[z_plane - 0.05, z_plane + 0.05]`), and back-projected all 21 joints along their true 2D pixel rays so the 3D pinch site tracks the real hand and object position under moving cameras (reducing max trajectory span in `raw_2.hdf5` from `2.801 m` to `0.217 m`).
  - `src/retargeting/transformer.py`: Upgraded `process_kinematics_retargeting` with direction-agnostic pick/place detection combining 2D tabletop excursion and biological pinch closure.
  - `scripts/06_train_policy.py`: Froze `BatchNorm2d` running statistics in `SpatialResNet18Encoder.train()` and added `_apply_sequence_kinematic_anchoring` in `BCPolicy` to fuse camera-compensated 3D kinematic flow with bounded visual-proprioceptive residuals during contiguous episode rollouts.
  - `scripts/09_interactive_mujoco_gui.py` & `scripts/10_all_in_one_pipeline_gui.py`: Removed auto-revert on model switching, anchored initial simulated cube XY position to each video's detected 3D tabletop pick point, aligned 3D simulation trajectories 1-to-1 with World Model and BC Policy predictions, and protected `demo_libero` against generic `Nouveau dossier` folder collisions.
  - Rebuilt `data/datasets/demo_libero.hdf5` (18 demos), `data/datasets/raw_2.hdf5` (20 moving-camera demos), and `data/datasets/test_videos.hdf5` (4 test videos), and retrained both `demo_libero` (**Rollout ADE = `0.33 cm`**, **Test Videos BC ADE = `0.40 cm`**, **100% Grasp & 16.5 cm Lift**) and `raw_2` (**Rollout ADE = `0.23 cm`**, **Test Videos BC ADE = `0.31 cm`**, **100% Grasp & 16.6 cm Lift**).

### [2026-11-27] — End-to-End `demo_libero` Pipeline Re-Run, Versioned Registry Archiving & Project Guide Synchronization
- **Pipeline & Training Execution**:
  - Re-ran Stages 2–6 on `demo_libero` (`demo_0000` .. `demo_0017`), regenerating `data/slam/demo_0000..0017`, `data/hand_poses/demo_0000..0017`, `data/retargeted/demo_0000..0017`, and `data/datasets/demo_libero.hdf5` (`18 demos`, `1,302 frames @ 20.0 Hz`).
  - Retrained the `demo_libero` Visuomotor BC Policy for 80 epochs (`best_motion_epoch = 78`, `best_gripper_epoch = 79`, `rollout_ade_cm = 0.342 cm`, `rollout_fde_cm = 0.548 cm`, `val_pos_mae_mm = 2.802 mm`, `val_grip_acc_pct = 96.75%`).
  - Verified non-destructive per-dataset model registry versioning (`experiments/checkpoints/model_registry.json`, `versions_count = 2` for `demo_libero` with the prior checkpoint preserved at `experiments/checkpoints/registry/demo_libero/best_policy_v1.pt` alongside `raw_2` at `experiments/checkpoints/registry/raw_2/best_policy.pt`).
- **Documentation & Project Guide Updates**:
  - Updated `PROJECT_CONTEXT.md`, `docs/MODEL_ARCHITECTURE_AND_PIPELINE_GUIDE.md`, `scripts/generate_technical_report.py`, and `README.md` to document the multi-dataset pipeline (`demo_libero.hdf5`, `raw_2.hdf5`, `test_videos.hdf5`), moving-camera 3D grounding, per-dataset model registry & continual fine-tuning, and the two interactive GUIs (`LAUNCH_ALL_IN_ONE_PIPELINE.bat` and `LAUNCH_INTERACTIVE_SIMULATOR.bat`).

---

## 4. Generated Datasets, Checkpoints, Videos & Quantitative Benchmarks

| Category | Path | Summary / Metrics |
| :--- | :--- | :--- |
| **Camera Calibration & Printable Targets** | `data/calibration/calib.txt`<br>`data/calibration/calibration.json`<br>`data/calibration/calibration.npz`<br>`data/calibration/targets/damier_calibration_A4.pdf`<br>`data/calibration/targets/aruco_tag_0_original.png` | Pinhole intrinsics & distortion parameters + printable A4 9x6 calibration checkerboard and 50 mm ArUco Tag ID 0 fiducial sheets generated by `scripts/generate_checkerboard_pdf.py` and `scripts/generate_aruco_tag.py`. |
| **LIBERO HDF5 Datasets (Static, Moving-Camera & Test)** | `data/datasets/demo_libero.hdf5`<br>`data/datasets/raw_2.hdf5`<br>`data/datasets/test_videos.hdf5`<br>`data/test_videos/` | • `demo_libero.hdf5`: 18 static-camera demos (`1,302 steps @ 20.0 Hz`).<br>• `raw_2.hdf5`: 20 moving-camera demos (`1,381 steps @ 20.0 Hz`).<br>• `test_videos.hdf5`: 4 cross-dataset evaluation videos (`test_01..test_04.mp4`).<br>All include `agentview_rgb`, `robot0_eye_in_hand_rgb`, `robot0_eef_pos`, `robot0_eef_quat`, `robot0_joint_pos`, `robot0_eef_pos_sim`, `robot0_eef_quat_sim`, and 7-DoF `actions`. |
| **Debug Tracking Overlays** | `data/debug_overlays/aruco_demo_0000/`<br>`data/debug_overlays/hand_demo_0000/` | Visual verification JPEGs for Stage 2 ArUco 6-DoF tabletop tracking and Stage 3 21-joint MediaPipe hand & pinch tracking. |
| **Trained World Model** | `experiments/world_model/world_model_checkpoint.pt`<br>`experiments/checkpoints/registry/demo_libero/world_model_checkpoint.pt`<br>`experiments/checkpoints/registry/raw_2/world_model_checkpoint.pt` | `ActionConditionedDynamicsModel` (Translation-Invariant Residual Kinematic Prior + 192-D GRU Latent Dynamics), trained 80 epochs with curriculum teacher-forcing annealing and registered per dataset. |
| **World Model Metrics & Predictions** | `experiments/world_model/world_model_metrics.json`<br>`experiments/world_model/world_model_predictions.npz`<br>`experiments/world_model/world_model_trajectory_3d.png` | **Mean ADE: `0.06 cm` (`0.6 mm`)** \| **Mean FDE: `0.08 cm` (`0.8 mm`)** (`demo_6`: `0.02 cm`, `demo_7`: `0.13 cm`, `demo_8`: `0.01 cm`, `demo_9`: `0.08 cm`). **100% Grasp & Lift Success (`15.8 cm`)** in 3D MuJoCo. |
| **Real Video vs. World Model Videos** | `experiments/world_model/video_vs_world_model_comparison.mp4`<br>`experiments/world_model/demo_0000_video_vs_world_model_comparison.mp4`<br>`experiments/world_model/world_model_vs_gt_mujoco_comparison.mp4`<br>`experiments/world_model/world_model_mujoco_prediction_replay.mp4`<br>`experiments/world_model/comparison_frames/` | Synchronized 4-panel (`1680x1920 @ 20 FPS`) comparison of Real RGB Video + 3D Hand/Robot/Trail Overlay, World Model 3D Franka Panda Sim, Ground-Truth 3D Franka Panda Sim, and Live X/Y/Z & 3D Error Dashboard. |
| **Trained Visuomotor BC Policies & Registry** | `experiments/checkpoints/model_registry.json`<br>`experiments/checkpoints/best_policy.pt`<br>`experiments/checkpoints/registry/demo_libero/`<br>`experiments/checkpoints/registry/raw_2/` | Enhanced Multi-Dataset Dual-Head `BCPolicy` (2D Spatial Softmax + Frozen-Stem Cached ResNet-18 + 6-Band Fourier Proprioception + Kinematic Anchoring + Pre-Norm Residual MLPs + EMA):<br>• `demo_libero`: **Rollout ADE = `0.34 cm`**, **Rollout FDE = `0.55 cm`**, **Val Pos MAE = `2.80 mm`**, **Gripper Acc = `96.75%`** (with `best_policy_v1.pt` preserved).<br>• `raw_2`: **Rollout ADE = `0.23 cm`**, **Rollout FDE = `0.26 cm`**, **Gripper Acc = `99.11%`**. |
| **3D Franka Panda MuJoCo Replays** | `experiments/simulation/demo_0_Lift_replay.mp4` .. `demo_5_Lift_replay.mp4` | Offscreen 3D standard Franka Emika Panda simulation replays (`Object Grasped: True`, `Task Lift Success: True`, `Max Cube Lift: 15.8 cm`, `EEF Travel: 0.76 - 0.86 m`). |
| **3D Skeleton Overlay & Side-by-Side** | `experiments/skeleton_overlays/demo_0000/demo_0000_robot_skeleton_overlay.mp4`<br>`experiments/skeleton_overlays/demo_0000/demo_0000_side_by_side_comparison.mp4` | Real video overlaid with 3D ArUco axes, 21-joint hand skeleton, and 3D Franka Panda kinematic chain (`robot0_link0`..`robot0_eef`), plus diagnostic JPEG frames (`0000`..`0085`). |
| **All-in-One Pipeline & 10-Epoch Live Trainer GUI** | `scripts/10_all_in_one_pipeline_gui.py`<br>`LAUNCH_ALL_IN_ONE_PIPELINE.bat`<br>`experiments/checkpoints/all_in_one_gui_preview.png` | Unified control center (`1460x900`) to pick raw video folders, run Stages 0–6 end-to-end, manage registered models / fine-tuning, and monitor live training convergence every 10 epochs (`Epoch 10..80`). |
| **Interactive 3D Simulator GUI** | `scripts/09_interactive_mujoco_gui.py`<br>`LAUNCH_INTERACTIVE_SIMULATOR.bat`<br>`experiments/world_model/interactive_gui_preview.png` | Live interactive window (`1440x860`) with 360° mouse camera orbit, World Model rollout mode, BC Policy rollout mode, Manual keyboard control (`W/A/S/D/Q/E` + `SPACE` grasp) with live 15-step World Model predictions, on-the-fly video/folder loading (`[O]`, `[V]`), and live registered model switching (`[M]`). |
| **4-Page Illustrated Technical Guide** | `docs/MODEL_ARCHITECTURE_AND_PIPELINE_GUIDE.pdf`<br>`docs/MODEL_ARCHITECTURE_AND_PIPELINE_GUIDE.md`<br>`docs/figures/fig1..fig4.png` & `page_1..page_4.png` | Complete 4-page A4 illustrated PDF and Markdown documentation covering all 6 pipeline stages, SE(3) coordinate math, closed-loop 3D Panda control, neural network architectures, multi-dataset registry, and interactive GUIs. |


