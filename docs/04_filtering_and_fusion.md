# Module 4: Geometric Fusion & Trajectory Filtering Guide

This guide details **Module 4: Geometric Fusion & Trajectory Filtering** within the **Egocentric Video to LIBERO Retargeting Pipeline**. This stage fuses camera poses, table plane geometry, and hand poses into the unified coordinate frame of the Franka Emika Panda robot base. It then resamples trajectories from 30 Hz to 20 Hz, applies Savitzky-Golay filtering, and generates 7-DOF Operational Space Control (OSC) action deltas for robotic policy training.

The implementation is located in [`kinematics_retargeting.py`](file:///C:/Users/marti/Desktop/Humanoid/kinematics_retargeting.py) and configured via [`configs/pipeline_config.yaml`](file:///C:/Users/marti/Desktop/Humanoid/configs/pipeline_config.yaml).

---

## 1. Mathematical Coordinate Transform Chain

To transfer human hand motions captured from an egocentric perspective onto a simulated Franka Panda manipulator, all measurements must be mapped through a consistent kinematic chain:

```
  [Camera Optic Frame]  ----(T_cam_to_hand)---->  [Human Hand / EEF]
          ^                                              |
          | (T_world_to_cam)                             |
          v                                              v
  [SLAM World Frame]    ----(T_world_to_table)-->  [Table Frame (Z=0)]
                                                         ^
                                                         | (T_robot_base_to_table)
                                                         v
                                              [Robot Base Frame]
```

### 1. Hand in SLAM World Frame
From Module 2, camera tracking provides $\mathbf{T}_{\text{world}\to\text{cam}, t} \in \mathbb{SE}(3)$. The camera-to-world transform is:

$$\mathbf{T}_{\text{cam}\to\text{world}, t} = \mathbf{T}_{\text{world}\to\text{cam}, t}^{-1} = \begin{bmatrix} \mathbf{R}_{\text{w2c}, t}^T & -\mathbf{R}_{\text{w2c}, t}^T \mathbf{t}_{\text{w2c}, t} \\ \mathbf{0}^T & 1 \end{bmatrix}$$

From Module 3, HaMeR provides the hand pose relative to the camera $\mathbf{T}_{\text{cam}\to\text{hand}, t}$. The hand in SLAM world frame is:

$$\mathbf{T}_{\text{world}\to\text{hand}, t} = \mathbf{T}_{\text{cam}\to\text{world}, t} \cdot \mathbf{T}_{\text{cam}\to\text{hand}, t}$$

### 2. Hand in Table Coordinate Frame
Module 2 determined the rigid alignment mapping the workspace table to the horizontal plane $Z = 0.0$:

$$\mathbf{T}_{\text{world}\to\text{table}} = \begin{bmatrix} \mathbf{R}_{\text{align}} & \mathbf{t}_{\text{align}} \\ \mathbf{0}^T & 1 \end{bmatrix}$$

Applying this transformation maps the hand into table coordinates:

$$\mathbf{T}_{\text{table}\to\text{hand}, t} = \mathbf{T}_{\text{world}\to\text{table}} \cdot \mathbf{T}_{\text{world}\to\text{hand}, t}$$

### 3. Hand in Franka Panda Robot Base Frame
The table sits at a known relative offset with respect to the Franka Panda arm base:

$$\mathbf{T}_{\text{robot\_base}\to\text{table}} = \begin{bmatrix} 1 & 0 & 0 & 0.75 \\ 0 & 1 & 0 & 0.00 \\ 0 & 0 & 1 & -0.05 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

* Translation: Table surface is $0.75\text{ m}$ forward ($+X$) and $0.05\text{ m}$ below ($-Z$) the robot mounting base.
* Rotation: $\mathbf{I}_{3\times3}$ (table surface parallel to robot base mounting plate).

The final end-effector target pose in robot base frame is:

$$\mathbf{T}_{\text{robot}\to\text{eef}, t} = \mathbf{T}_{\text{robot\_base}\to\text{table}} \cdot \mathbf{T}_{\text{table}\to\text{hand}, t}$$

---

## 2. Temporal Resampling: 30 Hz to 20 Hz

### Frequency Mismatch Rationale
* **Egocentric Video Capture**: Typically recorded at **30.0 FPS** (standard smartphone, webcam, or wearable glasses).
* **Robosuite & LIBERO Simulation**: Environments and Operational Space Controllers execute at **20 Hz** ($50\text{ ms}$ control step).

To avoid temporal phase shifts or cumulative drift, the trajectory of duration $T$ with $N$ source timesteps is resampled onto $M = \text{round}(T \cdot f_{\text{tgt}}) + 1$ uniformly spaced timesteps.

### Separate Channel Interpolation
1. **Position Interpolation**:
   Cubic spline interpolation (`scipy.interpolate.CubicSpline`) independently for Cartesian axes $(X, Y, Z)$:
   $$\mathbf{p}(t) = \text{CubicSpline}(\mathbf{t}_{\text{src}}, \mathbf{P}_{\text{src}})(t_{\text{tgt}})$$

2. **Orientation Interpolation (SLERP)**:
   Linear interpolation of rotation matrices or Euler angles destroys orthogonality and introduces gimbal lock. We convert rotation matrices to unit quaternions $\mathbf{q} \in \mathbb{S}^3$ and apply Spherical Linear Interpolation:
   $$\text{SLERP}(\mathbf{q}_0, \mathbf{q}_1; \tau) = \frac{\sin((1 - \tau)\Omega)}{\sin \Omega}\mathbf{q}_0 + \frac{\sin(\tau \Omega)}{\sin \Omega}\mathbf{q}_1$$
   where $\cos \Omega = \mathbf{q}_0 \cdot \mathbf{q}_1$. Handled robustly via `scipy.spatial.transform.Slerp`.

3. **Gripper Interpolation**:
   Interpolated via cubic spline and clipped strictly to the valid interval $[0.0, 1.0]$.

---

## 3. Trajectory Smoothing: Savitzky-Golay Filtering

Raw human motion capture trajectories exhibit high-frequency jitter caused by visual tracking noise, monocular depth ambiguity, and micro-tremors. Applying simple Gaussian smoothing flattens sharp peak transitions (such as contact moments). Instead, we apply **Savitzky-Golay filtering** (`scipy.signal.savgol_filter`).

### Filter Configuration
* **Window Length ($W$)**: 15 samples ($750\text{ ms}$ temporal smoothing window at 20 Hz). Must be an odd integer.
* **Polynomial Order ($p$)**: 3 (cubic polynomial preserves velocity and acceleration extrema).

### Preserving Rotation Lie Algebra $\mathfrak{so}(3)$
Filtering quaternions directly as 4D vectors violates the unit-norm constraint $\|\mathbf{q}\| = 1$. Instead:
1. Map resampled rotation matrices to rotation vectors (the Lie algebra $\mathfrak{so}(3)$):
   $$\mathbf{r}_t = \text{rotvec}(\mathbf{R}_t) = \theta \mathbf{u} \in \mathbb{R}^3$$
2. Apply Savitzky-Golay filtering independently along the 3 components of $\mathbf{r}_t$:
   $$\mathbf{r}_{t, \text{smooth}} = \text{SavGol}(\mathbf{r}_t, W=15, p=3)$$
3. Map smoothed rotation vectors back to $\mathbb{SO}(3)$ rotation matrices:
   $$\mathbf{R}_{t, \text{smooth}} = \exp([\mathbf{r}_{t, \text{smooth}}]_{\times})$$

This guarantees strictly valid orthonormal rotation matrices with continuous, smooth derivatives.

---

## 4. Operational Space Control (OSC) Action Deltas

LIBERO policies operate under **Operational Space Pose Impedance Control** (`OSC_POSE`). The action vector at timestep $t$ is a 7-dimensional command:

$$\mathbf{a}_t = [\Delta x_t, \Delta y_t, \Delta z_t, \Delta r_{x, t}, \Delta r_{y, t}, \Delta r_{z, t}, g_{t+1}] \in \mathbb{R}^7$$

### Mathematical Formulation
1. **Translational Deltas**:
   $$\Delta \mathbf{p}_t = \mathbf{p}_{t+1} - \mathbf{p}_t = \begin{bmatrix} x_{t+1} - x_t \\ y_{t+1} - y_t \\ z_{t+1} - z_t \end{bmatrix}$$
2. **Rotational Deltas**:
   The relative rotation matrix from step $t$ to step $t+1$ is:
   $$\Delta \mathbf{R}_t = \mathbf{R}_{t+1} \cdot \mathbf{R}_t^T$$
   Converted to an axis-angle rotation vector:
   $$\Delta \mathbf{r}_t = \text{rotvec}(\Delta \mathbf{R}_t) \in \mathbb{R}^3$$
3. **Gripper Command**:
   $$g_{t+1} \in \{-1.0, +1.0\} \text{ or } [0.0, 1.0]$$

For a trajectory of length $M$, this yields exactly $M-1$ action steps.

---

## 5. Running the Retargeting Pipeline

Execute [`kinematics_retargeting.py`](file:///C:/Users/marti/Desktop/Humanoid/kinematics_retargeting.py):

```bash
python kinematics_retargeting.py \
    --camera_poses outputs/camera_tracking/camera_poses.npz \
    --table_transform outputs/camera_tracking/table_transform.npz \
    --hand_poses outputs/hand_tracking/hand_poses.npz \
    --output_dir outputs/retargeting \
    --src_fps 30.0 \
    --tgt_fps 20.0
```

### CLI Arguments Reference

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--camera_poses` | `str` | *(Required)* | Path to `camera_poses.npz` (contains `T_world_to_cam`). |
| `--table_transform` | `str` | *(Required)* | Path to `table_transform.npz` (contains `R_align`, `t_align`). |
| `--hand_poses` | `str` | *(Required)* | Path to `hand_poses.npz` (contains `T_cam_to_hand`, gripper arrays). |
| `--output_dir` | `str` | `"outputs"` | Destination directory for `retargeted.npz`. |
| `--src_fps` | `float` | `30.0` | Sampling frequency of the camera and hand tracking inputs. |
| `--tgt_fps` | `float` | `20.0` | Target control frequency for the robot trajectory. |
| `--robot_table_transform_npy`| `str` | `None` | Optional custom $4 \times 4$ numpy matrix for robot-to-table transform. |

---

## 6. Output Specification: `retargeted.npz`

The file `outputs/retargeting/retargeted.npz` contains:

| Key | Shape | Dtype | Description |
| :--- | :--- | :--- | :--- |
| `actions` | $(M-1, 7)$ | `float32` | 7-DoF OSC action deltas $[\Delta x, \Delta y, \Delta z, \Delta r_x, \Delta r_y, \Delta r_z, g]$. |
| `ee_pos` | $(M, 3)$ | `float32` | Absolute end-effector position in robot base frame (meters). |
| `ee_quat` | $(M, 4)$ | `float32` | Absolute end-effector orientation as $[x, y, z, w]$ unit quaternion. |
| `gripper` | $(M,)$ | `float32` | Continuous gripper trajectory signal in range $[0.0, 1.0]$. |

Inspect the generated file in Python:
```python
import numpy as np

data = np.load("outputs/retargeting/retargeted.npz")
print("Actions shape:", data["actions"].shape)
print("EE Pos shape :", data["ee_pos"].shape)
print("Sample action:", data["actions"][0])
```
