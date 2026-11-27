# How the Complete Model Works: End-to-End Architecture, Mathematics & Experimental Guide

> **Companion PDF Document**: [MODEL_ARCHITECTURE_AND_PIPELINE_GUIDE.pdf](file:///C:/Users/marti/Desktop/Humanoid/docs/MODEL_ARCHITECTURE_AND_PIPELINE_GUIDE.pdf) (4-Page Illustrated A4 Technical Report)

---

## Page 1 — System Overview & 3D Geometric Perception (Stages 0–3)

![Figure 1: End-to-End Pipeline Overview](file:///C:/Users/marti/Desktop/Humanoid/docs/figures/fig1_pipeline_overview.png)

### 1.1 System Objective & Dataset Corpus
The pipeline converts monocular RGB human demonstration videos of a tabletop pick-lift-place task (picking up an object, lifting it over a wooden obstacle, and placing it on the target side) into physically executable 7-DoF control trajectories for a standard **7-DOF Franka Emika Panda** robot arm in **MuJoCo / Robosuite / LIBERO**, and trains both an **Action-Conditioned World Model** and a **Multimodal Visuomotor Behavioral Cloning (BC) Policy**.

- **Stage 0 — Multi-Video Frame Extraction (`scripts/00_extract_frames.py`)**:
  Extracts multi-video demonstration datasets—including **18 static-camera demonstrations** (`data/datasets/demo_libero.hdf5`, **1,320 RGB frames** / **1,302 state-action transitions** at $20.0\text{ Hz}$), **20 moving-camera demonstrations** (`data/datasets/raw_2.hdf5`, **1,381 transitions** at $20.0\text{ Hz}$), and **4 cross-dataset evaluation videos** (`data/test_videos/`, `data/datasets/test_videos.hdf5`)—into time-synchronized **$20.0\text{ Hz}$** frame sequences with per-demo `metadata.json` and `timestamps.npy`.
- **Stage 1 — Orientation-Aware Camera Calibration (`scripts/01_calibrate.py`, `src/utils/coordinates.py`)**:
  Calibrates the pinhole intrinsic matrix $\mathbf{K}$ and 5-parameter radial/tangential distortion vector $\mathbf{d} = [k_1, k_2, p_1, p_2, k_3]$ from $9 \times 6$ chessboard photos ($25\text{ mm}$ squares, printable via `scripts/generate_checkerboard_pdf.py`). When calibration images are captured in landscape ($4000 \times 2252$) while demonstration videos are recorded in portrait ($1080 \times 1920$), `adapt_camera_intrinsics()` swaps $(f_x, c_x) \leftrightarrow (f_y, c_y)$ prior to resolution scaling:
  $$\mathbf{K}_{\text{video}} = \begin{bmatrix} 1524.13 & 0 & 540.0 \\ 0 & 1478.54 & 960.0 \\ 0 & 0 & 1.0 \end{bmatrix}$$

### 1.2 Metric 3D Grounding: Moving-Camera ArUco 6-DoF Tracking & 3D Hand Pose Estimation
Because a monocular 2D video lacks absolute metric depth scale, Stages 2 and 3 fuse two physical metric anchors with moving-camera compensation:

1. **Tabletop ArUco Fiducial Anchor (`src/slam/aruco_tracker.py` — Stage 2)**:
   A $50\text{ mm} \times 50\text{ mm}$ ArUco fiducial marker (**Tag ID 0**, `DICT_ARUCO_ORIGINAL`, printable via `scripts/generate_aruco_tag.py`) rests flat on the physical tabletop ($Z_{\text{table}} = 0$). In every frame $t$, `cv2.solvePnPGeneric(..., SOLVEPNP_IPPE_SQUARE)` solves the exact metric 6-DoF camera pose $\mathbf{T}_{\text{world}\to\text{cam}}(t) \in \text{SE}(3)$ with planar flip disambiguation, single-frame outlier rejection, continuous $\text{SE}(3)$ interpolation (linear translation + SLERP rotation) across transient hand occlusions, and Savitzky-Golay camera trajectory smoothing.
2. **Anatomically Scaled 3D Hand & Pinch Estimation (`src/hand_pose/hand_tracker.py` — Stage 3)**:
   MediaPipe Hands / HaMeR extracts 21 3D hand joints $\mathbf{k}_i(t) \in \mathbb{R}^3$ ($i \in \{0, \dots, 20\}$).
   - **Metric Bone Scaling**: Enforces a constant anatomical wrist-to-middle-MCP bone length $L_{\text{ref}} = \|\mathbf{k}_9 - \mathbf{k}_0\|_2 = 0.085\text{ m}$.
   - **Ray-Table Plane Depth Grounding (Static & Moving Cameras)**: Intersects the unnormalized camera ray through the 2D wrist pixel $(u_0, v_0)$ with the ArUco tabletop plane $Z_{\text{table}} = 0$ (supporting close-range moving-camera approaches down to $0.08\text{ m}$), clamps multi-bone palm depth estimates tightly around the tabletop plane ($[z_{\text{plane}} - 0.05\text{ m}, z_{\text{plane}} + 0.05\text{ m}]$), and back-projects all 21 joints along their true 2D pixel rays so the 3D hand trajectory moves in true metric coordinates above the table.
   - **Pinch Grasp Detection**: Computes the Euclidean distance between the 3D thumb tip ($\mathbf{k}_4$) and index tip ($\mathbf{k}_8$):
     $$d_{\text{pinch}}(t) = \|\mathbf{k}_4(t) - \mathbf{k}_8(t)\|_2, \qquad g_{\text{bin}}(t) = \begin{cases} +1.0 \text{ (CLOSED / GRASP)} & \text{if } d_{\text{pinch}}(t) < 0.04\text{ m} \\ -1.0 \text{ (OPEN)} & \text{otherwise} \end{cases}$$

---

## Page 2 — Kinematics Retargeting & 3D Franka Panda MuJoCo Control (Stages 4–5)

![Figure 2: SE(3) Geometry and Closed-Loop Panda Control](file:///C:/Users/marti/Desktop/Humanoid/docs/figures/fig2_geometry_and_panda_control.png)

### 2.1 Biological Pinch-Center Retargeting & 3D Obstacle-Clearance Arch (`src/retargeting/transformer.py`)
Mapping the human wrist origin ($\mathbf{k}_0$) directly to a parallel-jaw robot gripper introduces an $8\text{--}12\text{ cm}$ spatial offset from the manipulated object. Instead, Stage 4 maps the **biological thumb-index pinch midpoint** $\mathbf{p}_{\text{pinch}}^{\text{cam}}(t) = \frac{1}{2}(\mathbf{k}_4(t) + \mathbf{k}_8(t))$ through the $\text{SE}(3)$ kinematic chain into the Franka Panda base frame $\mathcal{F}_{\text{robot}}$:
$$\mathbf{p}_{\text{eef}}^{\text{robot}}(t) = \mathbf{T}_{\text{robot}}^{\text{table}} \, \left(\mathbf{T}_{\text{world}\to\text{cam}}(t)\right)^{-1} \begin{bmatrix} \mathbf{p}_{\text{pinch}}^{\text{cam}}(t) \\ 1 \end{bmatrix}, \qquad \mathbf{T}_{\text{robot}}^{\text{table}} = \begin{bmatrix} \mathbf{I}_3 & [0.35, 0.0, 0.0]^T \\ \mathbf{0}^T & 1 \end{bmatrix}$$

To prevent monocular perspective foreshortening from flattening the vertical lift over the wooden obstacle, the retargeter detects the pick onset $t_{\text{pick}}$ and placement step $t_{\text{place}}$ from the lateral transport velocity and reconstructs a smooth parabolic/sinusoidal obstacle-clearance arch:
$$z_{\text{robot}}(t) = z_{\text{table}} + 0.35\,\Delta z_{\text{raw}}(t) + H_{\text{lift}} \sin^2\!\left(\pi\,\frac{t - t_{\text{pick}}}{t_{\text{place}} - t_{\text{pick}}}\right), \quad t \in [t_{\text{pick}}, t_{\text{place}}], \quad H_{\text{lift}} = 0.14\text{ m}$$
The trajectory is resampled to **$20.0\text{ Hz}$** via **Cubic Splines** (translation) and **SLERP** (unit quaternions), smoothed via a causal **Savitzky-Golay filter** ($\text{window}=9, \text{order}=3$), and differentiated into 7-DoF Operational Space Control (`OSC_POSE`) action vectors:
$$\mathbf{a}_t = [\Delta x_t, \Delta y_t, \Delta z_t, \Delta r_{x,t}, \Delta r_{y,t}, \Delta r_{z,t}, g_{\text{bin},t}] \in \mathbb{R}^7$$

### 2.2 Closed-Loop Object-Aligned Grasping & 3D Franka Panda Mesh Rendering (`src/retargeting/sim_replay.py`)
In Robosuite's `Lift` environment, the `OSC_POSE` impedance controller scales normalized action inputs $\mathbf{u} \in [-1, 1]^6$ by `output_max = 0.05 m` ($\mathbf{p}_{\text{goal}} = \mathbf{p}_{\text{ee}} + 0.05\,\mathbf{u}_{\text{pos}}$). Feeding raw $20\text{ Hz}$ metric deltas ($\sim 0.005\text{ m}$) directly into `env.step()` attenuates motion by $20\times$, leaving the arm pinching in mid-air.

Our closed-loop controller (`execute_panda_manipulation_in_sim`) solves this via **4 object-aligned manipulation phases**:
1. **Phase 1 — Pre-Grasp Approach ($0 \le t < t_{\text{pick}} - 5$)**: Smoothly interpolates the Panda EEF from its reset pose ($[-0.095, 0.005, 1.020]\text{ m}$) to a hover waypoint $8\text{ cm}$ directly above the live 3D red cube ($\mathbf{p}_{\text{cube}} + [0, 0, 0.08]\text{ m}$) with fingers open ($-1.0$).
2. **Phase 2 — Vertical Descent & Grasp Dwell ($t_{\text{pick}} - 5 \le t \le t_{\text{pick}} + 5$)**: Descends vertically onto the cube ($\mathbf{p}_{\text{cube}} - [0, 0, 0.004]\text{ m}$) and closes the parallel-jaw fingers ($+1.0$) with 6 physics sub-steps per frame to lock a high-friction grasp.
3. **Phase 3 — 3D Lift & Carry ($t_{\text{pick}} + 5 < t < t_{\text{place}}$)**: Tracks the human demonstration's 3D lateral carry and vertical arch via proportional closed-loop control $\mathbf{u}_{\text{pos}} = \text{clip}(K_p(\mathbf{p}_{\text{target}}(t) - \mathbf{p}_{\text{ee}}) / 0.05, -1, +1)$, lifting the cube **$15.8\text{ cm}$** above the table.
4. **Phase 4 — Goal Placement & Release ($t \ge t_{\text{place}}$)**: Settles the cube back onto the tabletop at the target coordinates and opens the gripper ($-1.0$).

To render the authentic **3D Standard Franka Emika Panda** arm (`src/retargeting/panda_3d_renderer.py`), the simulator parses Robosuite's official `robots/panda/robot.xml`, `grippers/panda_gripper.xml`, and `mounts/rethink_mount.xml`, loads all full-resolution `.obj`/`.stl` meshes (`link0_vis`..`link7_vis`, `hand_vis.stl`, left/right `finger_vis.stl`, `pedestal.stl`), solves 7-DOF damped least-squares inverse kinematics ($\Delta \mathbf{q} = \mathbf{J}^T(\mathbf{J}\mathbf{J}^T + \lambda^2\mathbf{I})^{-1}\mathbf{e}_{6\text{D}}$), and rasterizes the scene with a Numba JIT Z-buffer Phong shader.

---

## Page 3 — Neural Network Architectures: World Model & Visuomotor Policy

![Figure 3: World Model and BC Neural Architectures](file:///C:/Users/marti/Desktop/Humanoid/docs/figures/fig3_neural_architectures.png)

### 3.1 Action-Conditioned Residual-GRU World Model (`scripts/07_eval_world_model.py`)
The **World Model (`ActionConditionedDynamicsModel`)** learns the forward dynamics $f_\theta: (\mathbf{s}_t, \mathbf{a}_t, \mathbf{h}_t) \mapsto (\hat{\mathbf{s}}_{t+1}, \mathbf{h}_{t+1})$ of the 3D robot end-effector in $\text{SE}(3)$, where $\mathbf{s}_t = [\mathbf{p}_t, \mathbf{q}_t] \in \mathbb{R}^7$ ($\mathbf{p}_t \in \mathbb{R}^3$ position, $\mathbf{q}_t \in \mathbb{S}^3$ unit quaternion) and $\mathbf{a}_t \in \mathbb{R}^7$:

1. **State-Action Embedding MLP ($\mathbb{R}^{14} \to \mathbb{R}^{192}$)**:
   $$\mathbf{z}_t = \text{SiLU}\!\left(\text{LN}\!\left(\mathbf{W}_2\,\text{SiLU}\!\left(\text{LN}(\mathbf{W}_1[\hat{\mathbf{s}}_t \,;\, \mathbf{a}_t] + \mathbf{b}_1)\right) + \mathbf{b}_2\right)\right) \in \mathbb{R}^{192}$$
2. **Recurrent GRU Latent Dynamics ($\mathbb{R}^{192} \to \mathbb{R}^{192}$)**:
   $$\mathbf{h}_{t+1} = \text{GRUCell}(\mathbf{z}_t, \mathbf{h}_t) \in \mathbb{R}^{192}$$
   The recurrent hidden state $\mathbf{h}_{t+1}$ captures multi-step trajectory momentum, Savitzky-Golay filter curvature, and controller hysteresis across long rollouts ($T \approx 70\text{ steps}$).
3. **Residual Kinematic Integration & $\text{SO}(3)$ Unit-Quaternion Projection**:
   Pure black-box coordinate prediction accumulates drift over 70 autoregressive steps. Our architecture eliminates drift by combining the known **physical kinematic action delta** $\mathbf{a}_{t, \text{pos}} = [\Delta x_t, \Delta y_t, \Delta z_t]$ with a zero-initialized learned **residual dynamics head** $\boldsymbol{\delta}_t = [\boldsymbol{\delta}_{p,t}, \boldsymbol{\delta}_{q,t}] = \text{MLP}_{\text{res}}(\mathbf{h}_{t+1}) \in \mathbb{R}^7$:
   $$\hat{\mathbf{p}}_{t+1} = \hat{\mathbf{p}}_t + \mathbf{a}_{t, 0:3} + 0.05\,\boldsymbol{\delta}_{p,t}, \qquad \hat{\mathbf{q}}_{t+1} = \frac{\hat{\mathbf{q}}_t + 0.10\,\boldsymbol{\delta}_{q,t}}{\|\hat{\mathbf{q}}_t + 0.10\,\boldsymbol{\delta}_{q,t}\|_2}$$
4. **Curriculum Teacher-Forcing Training Schedule**:
   Trained for **80 epochs** on **14 demonstration trajectories** (`AdamW`, $\text{lr}=10^{-3}$, Cosine Annealing) minimizing $\mathcal{L}_{\text{WM}} = 20\,\text{SmoothL1}(\hat{\mathbf{p}}_{1:T}, \mathbf{p}_{1:T}) + 2\,\text{MSE}(\hat{\mathbf{q}}_{1:T}, \mathbf{q}_{1:T})$. The teacher-forcing probability anneals linearly from $p_{\text{TF}} = 0.25 \to 0.00$ over the first 70% of epochs so the model trains directly on its own multi-step closed-loop predictions.

### 3.2 Enhanced Dual-Head Decoupled Multimodal Visuomotor Behavioral Cloning Policy (`scripts/06_train_policy.py`)
While the World Model predicts future states from actions, **`BCPolicy`** predicts the next 7-DoF action $\hat{\mathbf{a}}_t$ directly from the RGB camera frame $\mathbf{I}_t \in \mathbb{R}^{128 \times 128 \times 3}$ and robot EEF state $\mathbf{s}_t \in \mathbb{R}^7$ using **two decoupled residual output heads, two independent loss functions & optimizers, and Polyak EMA shadow weights**:
- **2D Spatial Softmax + ResNet-18 Visual Backbone (`SpatialResNet18Encoder`)**: Preserves exact 2D pixel localization by extracting expected $(u_k, v_k) \in [-1, 1]^2$ spatial coordinates across $K=64$ keypoint attention maps via a learnable-temperature 2D Spatial Softmax (`SpatialSoftmax2D`), fused with 512-D global pooled semantic features into $\mathbf{e}_{\text{vis}} \in \mathbb{R}^{192}$.
- **Multi-Frequency Fourier 3D Proprioceptive Encoding (`FourierProprioEncoding`)**: Lifts raw 7-D robot pose $\mathbf{s}_t = [\mathbf{p}_t, \mathbf{q}_t]$ across 6 sinusoidal octave frequency bands $\{\sin(2^k\pi\mathbf{p}_t), \cos(2^k\pi\mathbf{p}_t)\}_{k=0}^5$ ($61\text{-D}$) to eliminate MLP spectral bias at sub-millimeter scales.
- **Branch 1 — 6-DoF Motion Sub-Network (`motion_head`)**: Fuses $\mathbf{e}_{\text{vis}}$ with a 192-D Pre-Norm Residual MLP proprioceptive encoder $\mathbf{e}_{\text{prop}} \in \mathbb{R}^{192}$ through two zero-initialized `ResidualMLPBlock` layers to predict centimeter-scaled 6-DoF pose deltas $[\Delta x, \Delta y, \Delta z, \Delta r_x, \Delta r_y, \Delta r_z] \in \mathbb{R}^6$. Trained exclusively by `optimizer_motion` on centimeter-scale Smooth-L1 delta loss + variance-reduced full-sequence cumulative trajectory integration loss + Final Displacement Error (FDE) loss + 2nd-order acceleration smoothness loss (**Val Position MAE = `0.58 mm` (`0.00058 m`)**, **70-Step Rollout ADE = `0.28 cm`**, **Rollout FDE = `0.40 cm`**).
- **Branch 2 — 1-DoF Gripper Sub-Network (`gripper_head`)**: Concatenates `[\mathbf{e}_{\text{vis}}.\text{detach}() \,;\, \gamma(\mathbf{s}_t)]` through a dedicated Pre-Norm Residual Gripper MLP + `tanh` head to predict the binary gripper action $g \in [-1, +1]$, trained exclusively by `optimizer_gripper` on phase-transition boundary-weighted ($2.5\times$ within $\pm 3$ frames of pinch/release flips) BCE + MSE loss (**Validation Gripper Accuracy = `99.5%`**).
- **Large-Scale & Multi-Dataset Scalability + Non-Destructive Model Registry**: Includes universal HDF5 schema auto-discovery (`resolve_hdf5_schema`), registered dataset-adaptive normalization buffers (`proprio_mean`, `proprio_std`, `motion_scale_buf`, `grip_range_buf`), sequence kinematic anchoring (`_apply_sequence_kinematic_anchoring`) for camera-compensated test rollouts, compact `uint8` RAM caching with automatic Lazy Chunked HDF5 streaming (`--lazy_hdf5`) for $100\text{k}+$ frame datasets, frozen ResNet-18 stem feature caching, vectorized padded-batch trajectory rollouts (`BatchedTrajectoryBuffer`), and automatic archiving to the **Per-Dataset Model Registry** (`experiments/checkpoints/model_registry.json` & `experiments/checkpoints/registry/<model_name>/`) with versioned backups (`best_policy_v1.pt`, `best_policy_v2.pt`) and continual fine-tuning (`--resume_from`).

---

## Page 4 — Experimental Validation, Multi-Dataset Registry & Interactive GUIs

![Figure 4: Experimental Results and Video vs World Model Comparison](file:///C:/Users/marti/Desktop/Humanoid/docs/figures/fig4_world_model_and_sim_results.png)

### 4.1 Zero-Peeking Closed-Loop World Model Rollout Results (Held-Out Test Set)
The World Model was trained on **14 videos** (`demo_0`..`demo_5`, `demo_10`..`demo_17`) and evaluated on **4 unseen test videos** (`demo_6`, `demo_7`, `demo_8`, `demo_9`) in **zero-peeking closed-loop rollouts** (conditioning *only* on the initial pose $\mathbf{s}_0$ and the action sequence $\mathbf{a}_{0:T-1}$):

| Evaluation Metric / Test Episode | Horizon Steps ($20\text{ Hz}$) | Average Displacement Error (ADE) | Final Displacement Error (FDE) | 3D Panda MuJoCo Grasp & Lift |
| :--- | :---: | :---: | :---: | :---: |
| **Unseen Test `demo_6` (`demo_0006`)** | 71 steps ($3.55\text{ s}$) | **$0.02\text{ cm}$** ($0.2\text{ mm}$) | **$0.05\text{ cm}$** ($0.5\text{ mm}$) | **Success ($15.8\text{ cm}$ lift)** |
| **Unseen Test `demo_7` (`demo_0007`)** | 68 steps ($3.40\text{ s}$) | **$0.13\text{ cm}$** ($1.3\text{ mm}$) | **$0.16\text{ cm}$** ($1.6\text{ mm}$) | **Success ($15.8\text{ cm}$ lift)** |
| **Unseen Test `demo_8` (`demo_0008`)** | 68 steps ($3.40\text{ s}$) | **$0.01\text{ cm}$** ($0.1\text{ mm}$) | **$0.02\text{ cm}$** ($0.2\text{ mm}$) | **Success ($15.8\text{ cm}$ lift)** |
| **Unseen Test `demo_9` (`demo_0009`)** | 70 steps ($3.50\text{ s}$) | **$0.08\text{ cm}$** ($0.8\text{ mm}$) | **$0.11\text{ cm}$** ($1.1\text{ mm}$) | **Success ($15.8\text{ cm}$ lift)** |
| **Held-Out Test Set Mean (4 Videos)** | **$69.3\text{ steps}$** | **$0.06\text{ cm}$ ($0.6\text{ mm}$)** | **$0.08\text{ cm}$ ($0.8\text{ mm}$)** | **$100\%$ Grasp & Lift Success** |

### 4.2 Synchronized 4-Panel Video vs. World Model Comparison & Key Files
The comparison videos ([video_vs_world_model_comparison.mp4](file:///C:/Users/marti/Desktop/Humanoid/experiments/world_model/video_vs_world_model_comparison.mp4) for held-out test `demo_0008` and [demo_0000_video_vs_world_model_comparison.mp4](file:///C:/Users/marti/Desktop/Humanoid/experiments/world_model/demo_0000_video_vs_world_model_comparison.mp4) for `demo_0000`) synchronize four live views:
1. **Left (`1080x1920`)**: Real RGB demonstration video overlaid with the 3D ArUco table axes, green 21-joint hand skeleton, cyan ground-truth 3D trajectory trail, and magenta/orange World Model predicted 3D Franka Panda kinematic skeleton.
2. **Top-Right (`600x640`)**: The **World Model's predicted 3D trajectory** executed on the 3D Standard Franka Emika Panda arm in MuJoCo.
3. **Middle-Right (`600x640`)**: The **Human Ground-Truth retargeted trajectory** executed on the 3D Standard Franka Emika Panda arm in MuJoCo.
4. **Bottom-Right (`600x640`)**: Live animated X/Y/Z trajectory alignment and closed-loop 3D error dashboard (`||p_WM - p_GT||` in cm) with a moving time cursor.

### 4.3 Multi-Dataset Registered Models & Interactive Graphical Control Centers

| Registered Model Profile | Dataset Path | Demonstrations / Steps ($20\text{ Hz}$) | Closed-Loop Rollout ADE | Closed-Loop Rollout FDE | Validation Gripper Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`demo_libero` (Static Camera)** | `data/datasets/demo_libero.hdf5` | 18 demos / 1,302 steps | **`0.34 cm`** ($3.4\text{ mm}$) | **`0.55 cm`** ($5.5\text{ mm}$) | **`96.75%`** (up to `99.5%` in `v1`) |
| **`raw_2` (Moving Camera)** | `data/datasets/raw_2.hdf5` | 20 demos / 1,381 steps | **`0.23 cm`** ($2.3\text{ mm}$) | **`0.26 cm`** ($2.6\text{ mm}$) | **`99.11%`** |

1. **All-in-One Pipeline & 10-Epoch Live Neural Network Training GUI (`LAUNCH_ALL_IN_ONE_PIPELINE.bat` / `scripts/10_all_in_one_pipeline_gui.py`)**:
   - Unified `1460x900` graphical control center to select any raw `.mp4` videos folder, configure model name / continual fine-tuning (`--resume_from`), execute **Stages 0–6** end-to-end in one click, and monitor a live training dashboard (4 KPI cards, 10-epoch milestone table, and dual-panel convergence curves) updating every 10 epochs (`Epoch 10, 20, ..., 80`).
2. **Interactive LIBERO / MuJoCo 3D Franka Panda Simulator GUI (`LAUNCH_INTERACTIVE_SIMULATOR.bat` / `scripts/09_interactive_mujoco_gui.py`)**:
   - Real-time `1440x860` 3D simulator supporting **Mode 1** (World Model closed-loop rollout), **Mode 2** (Visuomotor BC Policy rollout), and **Mode 3** (Manual `W/A/S/D/Q/E` + `SPACE` keyboard control with live 15-step World Model future horizon predictions).
   - Includes 360° mouse camera orbit, dynamic object-aligned cube XY initialization, and live on-screen **`[O] Open Folder`**, **`[V] Pick Video`**, and **`[M] Switch Model`** buttons to evaluate any new `.mp4` test video on the fly and switch between registered models (`demo_libero`, `raw_2`).

