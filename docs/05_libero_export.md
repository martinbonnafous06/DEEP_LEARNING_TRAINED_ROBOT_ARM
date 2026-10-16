# Module 5: LIBERO Dataset Export Guide

This guide details **Module 5: LIBERO Dataset Export** within the **Egocentric Video to LIBERO Retargeting Pipeline**. This final stage packages retargeted actions, Cartesian end-effector poses, camera visual observations (agentview and eye-in-hand), and optional simulated MuJoCo joint states into an HDF5 archive that conforms strictly to the [LIBERO Benchmark](https://libero-project.github.io/) and [Robomimic](https://robomimic.github.io/) dataset specifications.

The module is implemented in [`dataset_exporter.py`](file:///C:/Users/marti/Desktop/Humanoid/dataset_exporter.py) and configured via [`configs/pipeline_config.yaml`](file:///C:/Users/marti/Desktop/Humanoid/configs/pipeline_config.yaml).

---

## 1. LIBERO HDF5 Schema Layout

LIBERO and Robomimic policies (such as Diffusion Policy, ACT, and OpenVLA) expect imitation datasets formatted in HDF5 with a hierarchical tree structure:

```text
outputs/datasets/libero_dataset.hdf5
├── [Root Attributes]
│   ├── total: 1 (int)
│   ├── env_name: "LIBERO_OBJECT" (str)
│   └── env_args: JSON string containing Robosuite simulation configuration
└── data/
    └── demo_0/
        ├── actions                      (N, 7)       float32   [dx, dy, dz, drx, dry, drz, grip]
        ├── rewards                      (N,)         float32   [0.0, ...]
        ├── dones                        (N,)         float32   [0.0, ..., 1.0]
        └── obs/
            ├── agentview_rgb            (N, 84, 84, 3) uint8   Third-person visual observation
            ├── robot0_eye_in_hand_rgb   (N, 84, 84, 3) uint8   Wrist camera visual observation
            ├── robot0_eef_pos           (N, 3)       float32   End-effector XYZ in robot base frame
            ├── robot0_eef_quat          (N, 4)       float32   End-effector quaternion [x, y, z, w]
            └── robot0_joint_pos         (N, 7)       float32   Franka arm 7-DOF joint angles (sim)
```

---

## 2. Visual Observation Processing: 84x84 Resolution

Standard policy architectures in LIBERO operate on image observations resized to **$84 \times 84$ pixels** in RGB format:

1. **Agentview RGB (`obs/agentview_rgb`)**:
   * Sourced from the extracted video frames or external fixed cameras.
   * Resized using bilinear interpolation: $(H_{\text{orig}}, W_{\text{orig}}, 3) \to (84, 84, 3)$ with `uint8` values $[0, 255]$.
2. **Wrist / Eye-in-Hand RGB (`obs/robot0_eye_in_hand_rgb`)**:
   * Sourced from wrist-mounted camera frames (if available).
   * If wrist frames are omitted or unavailable, [`dataset_exporter.py`](file:///C:/Users/marti/Desktop/Humanoid/dataset_exporter.py) automatically generates a zero-filled `uint8` placeholder array $(N, 84, 84, 3)$, allowing downstream visual encoders to operate without schema errors.
3. **Frame Temporal Synchronization**:
   * If the number of extracted image frames differs from the resampled trajectory length $N$, the loader truncates excess frames or repeats the final frame to ensure strict temporal alignment.

---

## 3. Simulation Replay & Joint State Collection

While kinematic retargeting provides end-effector poses ($\mathbf{p}_{\text{eef}}, \mathbf{q}_{\text{eef}}$), policy training often benefits from true robot joint kinematics ($\mathbf{q}_{\text{joint}}$) and physical contact states.

### Robosuite Replay Loop
When [Robosuite](https://robosuite.ai/) and [MuJoCo](https://mujoco.org/) are installed:
1. An environment instance is instantiated with the Franka Panda arm and Operational Space Controller:
   ```python
   env = suite.make(
       env_name="LIBERO_OBJECT",
       robots="Panda",
       has_renderer=False,
       has_offscreen_renderer=False,
       use_camera_obs=False,
       control_freq=20,
   )
   ```
2. The retargeted OSC action deltas are executed sequentially via `env.step(action)`.
3. True physics feedback (`robot0_joint_pos`, `robot0_eef_pos`, `robot0_eef_quat`) is recorded at every timestep.

### Graceful Fallback Mode
If Robosuite is not installed or MuJoCo execution is disabled, [`dataset_exporter.py`](file:///C:/Users/marti/Desktop/Humanoid/dataset_exporter.py) skips the simulation replay without error. The dataset is exported directly using the kinematically retargeted end-effector poses from Module 4.

---

## 4. Running the Dataset Exporter

Execute [`dataset_exporter.py`](file:///C:/Users/marti/Desktop/Humanoid/dataset_exporter.py):

```bash
python dataset_exporter.py \
    --retargeted_npz outputs/retargeting/retargeted.npz \
    --output_hdf5 outputs/datasets/libero_dataset.hdf5 \
    --agentview_dir data/frames/agentview \
    --wrist_dir data/frames/wrist \
    --demo_name demo_0 \
    --env_name LIBERO_OBJECT \
    --frame_height 84 \
    --frame_width 84
```

### CLI Parameters Reference

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--retargeted_npz` | `Path` | *(Required)* | Path to `retargeted.npz` produced by Module 4. |
| `--output_hdf5` | `Path` | *(Required)* | Destination file path for the HDF5 archive. |
| `--agentview_dir` | `Path` | `None` | Directory containing third-person RGB frames (optional). |
| `--wrist_dir` | `Path` | `None` | Directory containing wrist camera RGB frames (optional). |
| `--demo_name` | `str` | `"demo_0"` | Demonstration group name inside `/data/`. |
| `--env_name` | `str` | `"LIBERO_OBJECT"` | Target LIBERO benchmark task name. |
| `--frame_height` | `int` | `84` | Visual observation height in pixels. |
| `--frame_width` | `int` | `84` | Visual observation width in pixels. |

---

## 5. Dataset Introspection & Verification

[`dataset_exporter.py`](file:///C:/Users/marti/Desktop/Humanoid/dataset_exporter.py) automatically invokes `h5py.File.visititems()` upon writing to print a complete structural diagnostic to the terminal:

```text
================================================================================
HDF5 Structure Summary: outputs/datasets/libero_dataset.hdf5
================================================================================
  [ATTR] total = 1
  [ATTR] env_name = 'LIBERO_OBJECT'
  [ATTR] env_args = '{"env_name": "LIBERO_OBJECT", "type": 1, ...}'
  [GROUP] data
  [GROUP] data/demo_0
  [DATASET] data/demo_0/actions                      shape=(199, 7), dtype=float32
  [DATASET] data/demo_0/dones                        shape=(199,), dtype=float32
  [DATASET] data/demo_0/rewards                      shape=(199,), dtype=float32
  [GROUP] data/demo_0/obs
  [DATASET] data/demo_0/obs/agentview_rgb            shape=(199, 84, 84, 3), dtype=uint8
  [DATASET] data/demo_0/obs/robot0_eye_in_hand_rgb   shape=(199, 84, 84, 3), dtype=uint8
  [DATASET] data/demo_0/obs/robot0_eef_pos           shape=(199, 3), dtype=float32
  [DATASET] data/demo_0/obs/robot0_eef_quat          shape=(199, 4), dtype=float32
  [DATASET] data/demo_0/obs/robot0_joint_pos         shape=(199, 7), dtype=float32
================================================================================
```

### Python Verification Script
You can verify the dataset programmatically using this test snippet:

```python
import h5py
import numpy as np

with h5py.File("outputs/datasets/libero_dataset.hdf5", "r") as f:
    demo = f["data/demo_0"]
    actions = demo["actions"][:]
    eef_pos = demo["obs/robot0_eef_pos"][:]
    img = demo["obs/agentview_rgb"][:]
    
    print("Verification Results:")
    print(f"  Actions count: {len(actions)}, shape: {actions.shape}, dtype: {actions.dtype}")
    print(f"  Action translation range: [{actions[:, :3].min():.4f}, {actions[:, :3].max():.4f}] m")
    print(f"  EEF Z range: [{eef_pos[:, 2].min():.4f}, {eef_pos[:, 2].max():.4f}] m")
    print(f"  Agentview images shape: {img.shape}, dtype: {img.dtype}")
    
    # Assert sanity
    assert len(actions) == len(eef_pos)
    assert not np.isnan(actions).any(), "NaN values found in actions!"
    print("Dataset verification passed successfully!")
```

---

## 6. End-to-End Retargeting Workflow Summary

To run all 5 modules from start to finish:

```bash
# 1. Camera Calibration
python calibration.py --image_dir data/calibration --output_dir outputs/calibration

# 2. Camera Tracking & Table Plane
python camera_tracking.py \
    --frames_dir data/frames \
    --calib_txt outputs/calibration/calib.txt \
    --output_dir outputs/camera_tracking

# 3. 3D Hand Tracking & Gripper Estimation
python hand_tracking.py \
    --frames_dir data/frames \
    --output_dir outputs/hand_tracking

# 4. Kinematics Retargeting & Trajectory Filtering
python kinematics_retargeting.py \
    --camera_poses outputs/camera_tracking/camera_poses.npz \
    --table_transform outputs/camera_tracking/table_transform.npz \
    --hand_poses outputs/hand_tracking/hand_poses.npz \
    --output_dir outputs/retargeting

# 5. Export to LIBERO HDF5
python dataset_exporter.py \
    --retargeted_npz outputs/retargeting/retargeted.npz \
    --output_hdf5 outputs/datasets/libero_dataset.hdf5 \
    --agentview_dir data/frames/agentview
```
