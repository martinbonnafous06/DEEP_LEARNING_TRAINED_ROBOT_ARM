# Module 1: Camera Intrinsic Calibration Guide

This document provides a comprehensive operational guide for camera intrinsic calibration within the **Egocentric Video to LIBERO Retargeting Pipeline**. Camera intrinsic parameters ($f_x, f_y, c_x, c_y$ and lens distortion coefficients) are strictly necessary for recovering accurate 3D metric trajectories, unprojecting visual SLAM depth maps, and anchoring hand poses onto the workspace table.

The calibration workflow is implemented in [`calibration.py`](file:///C:/Users/marti/Desktop/Humanoid/calibration.py) and configured via [`configs/pipeline_config.yaml`](file:///C:/Users/marti/Desktop/Humanoid/configs/pipeline_config.yaml).

---

## 1. Checkerboard Preparation & Printing

Accurate intrinsic estimation requires a rigid, precisely measured planar target. A standard OpenCV planar chessboard pattern with **9 inner corners horizontally by 6 inner corners vertically** ($10 \times 7$ total black and white squares) is utilized.

### Specifications
* **Target Paper Size**: Standard ISO A4 ($297 \times 210\text{ mm}$).
* **Grid Layout**: 10 squares wide $\times$ 7 squares high.
* **Internal Corners**: $9 \times 6 = 54$ detectable corner vertices.
* **Square Edge Length ($s$)**: $25.0\text{ mm}$ ($0.025\text{ m}$).

### Generating and Printing the Pattern
You can generate the print-ready PDF using the bundled utility script [`generate_checkerboard_pdf.py`](file:///C:/Users/marti/Desktop/Humanoid/generate_checkerboard_pdf.py):

```bash
python generate_checkerboard_pdf.py --output damier_calibration_A4.pdf
```

> [!IMPORTANT]
> When printing the generated PDF ([`damier_calibration_A4.pdf`](file:///C:/Users/marti/Desktop/Humanoid/damier_calibration_A4.pdf)):
> 1. Set the print scaling to **100% / Actual Size** (disable "Fit to Page" or "Shrink to Printable Area").
> 2. Measure the printed squares with a physical vernier caliper or millimeter ruler across 5 adjacent squares (should measure exactly $125.0\text{ mm}$).
> 3. Mount the paper firmly onto a flat, rigid substrate (such as a 5 mm foam core board, clipboard, or aluminum sheet) using spray adhesive. **Any warping or curling will introduce systematic focal length distortion.**

---

## 2. Image Capture Protocol

To compute an overdetermined, unbiased non-linear optimization for the pinhole projection matrix and Brown-Conrady radial/tangential distortion model, collect between **15 and 25 diverse calibration photographs**.

```
    +-----------------------------------------------------------+
    | [Capture Zone]                                            |
    |                                                           |
    |   View 1: Top-Left Tilt         View 2: Center Frontal    |
    |          \                         |                      |
    |           [9x6]                  [9x6]                    |
    |                                                           |
    |   View 3: Bottom-Right Yaw      View 4: Extreme Close-up  |
    |                 /                      \                  |
    |              [9x6]                     [9x6]              |
    +-----------------------------------------------------------+
```

### Essential Guidelines
1. **Field of View Coverage**:
   * Distribute chessboard poses across all 4 quadrants and corners of the image sensor. Distortion parameters ($k_1, k_2, p_1, p_2$) are primarily constrained by samples placed near the image perimeter.
2. **Pose Diversity**:
   * **Distance**: Vary the distance between $0.3\text{ m}$ (filling 80% of the field of view) and $1.5\text{ m}$ (filling ~20% of the field of view).
   * **Angular Variations**: Tilt the board with pitch, roll, and yaw angles between $15^\circ$ and $45^\circ$ relative to the camera optical axis. Purely frontal images cause degenerate depth-focal length coupling.
3. **Locking Camera Parameters (Crucial)**:
   * **Disable Autofocus**: Set focus to manual or fixed mode. Any autofocus actuation changes the physical lens distance and alters the effective focal length ($f_x, f_y$).
   * **Disable Optical Image Stabilization (OIS)**: Sensor-shift or lens OIS alters the principal point coordinates ($c_x, c_y$) frame-by-frame.
   * **Constant White Balance & Exposure**: Ensure sharp, motion-blur-free images with uniform illumination to prevent subpixel corner jitter.

Store captured images inside `data/calibration/` (supported extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`).

---

## 3. Mathematical Foundations

The calibration routine estimates the camera intrinsic matrix $\mathbf{K}$ and distortion vector $\mathbf{D}$:

$$\mathbf{K} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}, \quad \mathbf{D} = [k_1, k_2, p_1, p_2, k_3]$$

Where:
* $f_x, f_y$: Focal lengths in pixel units ($f_x = F \cdot m_x, f_y = F \cdot m_y$).
* $c_x, c_y$: Principal point coordinates (optical center offset).
* $k_1, k_2, k_3$: Radial distortion parameters modeling barrel/pincushion effects:
  $$x_{\text{dist}} = x(1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + 2 p_1 x y + p_2 (r^2 + 2 x^2)$$
  $$y_{\text{dist}} = y(1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + p_1 (r^2 + 2 y^2) + 2 p_2 x y$$
  where $r^2 = x^2 + y^2$.

The solver minimizes the overall Root Mean Square (RMS) reprojection error over $M$ valid images and $N = 54$ points:

$$\text{RMS} = \sqrt{\frac{1}{M \cdot N} \sum_{i=1}^{M} \sum_{j=1}^{N} \|\mathbf{p}_{ij} - \hat{\mathbf{p}}(\mathbf{K}, \mathbf{D}, \mathbf{R}_i, \mathbf{t}_i, \mathbf{P}_j)\|^2}$$

---

## 4. Running the Calibration

Execute [`calibration.py`](file:///C:/Users/marti/Desktop/Humanoid/calibration.py) specifying the input image directory and target board parameters:

```bash
python calibration.py \
    --image_dir data/calibration \
    --board_width 9 \
    --board_height 6 \
    --square_size 0.025 \
    --output_dir outputs/calibration \
    --visualize
```

### Command-Line Arguments Reference

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--image_dir` | `str` | *(Required)* | Directory containing the calibration photographs. |
| `--board_width` | `int` | `9` | Number of internal corners horizontally. |
| `--board_height`| `int` | `6` | Number of internal corners vertically. |
| `--square_size` | `float` | `0.025` | Square side length in meters ($25\text{ mm} = 0.025$). |
| `--output_dir` | `str` | `--image_dir` | Directory where output files will be written. |
| `--visualize` | `flag` | `False` | Opens side-by-side OpenCV window comparing original vs undistorted image. |

---

## 5. Evaluating Calibration Quality

Upon completion, inspect the logged **RMS reprojection error**:

```text
[INFO] 20:45:10 – RMS reprojection error: 0.342180 px
[INFO] 20:45:10 – === Calibration complete ===
[INFO] 20:45:10 –   calib.txt       -> outputs/calibration/calib.txt
[INFO] 20:45:10 –   calibration.npz -> outputs/calibration/calibration.npz
[INFO] 20:45:10 –   RMS             = 0.342180 px
```

### Acceptance Thresholds
* **$\text{RMS} < 0.5\text{ px}$**: **Excellent**. High-precision metrics suitable for fine object manipulation.
* **$0.5\text{ px} \le \text{RMS} \le 1.0\text{ px}$**: **Acceptable**. Sufficient for visual SLAM and coarse robotic tracking.
* **$\text{RMS} > 1.0\text{ px}$**: **Unacceptable / Warning**. High likelihood of inaccurate depth back-projection and scale drift.

### Remediation Steps if RMS > 1.0 px:
1. Verify `--square_size` and `--board_width` / `--board_height` matching the printed physical board.
2. Filter out blurry or out-of-focus images where corner sub-pixel refinement settled on erroneous gradients.
3. Ensure no autofocus occurred between shots.
4. Check that the board remained completely flat without flexure during photography.

---

## 6. Output Files Specification

The calibration module writes three output representations:

### 1. `calib.txt` (DROID-SLAM Ingestion Format)
Single ASCII line containing the four pinhole projection parameters:
```text
fx fy cx cy
```
Example:
```text
912.45321876 911.89043210 640.12456321 360.54210987
```

### 2. `calibration.npz` (Compressed NumPy Archive)
Stores the full mathematical solution for downstream Python pipelines:
* `camera_matrix`: $(3, 3)$ `float64` array containing $\mathbf{K}$.
* `dist_coeffs`: $(1, 5)$ or $(5,)$ `float64` array containing $[k_1, k_2, p_1, p_2, k_3]$.
* `rvecs`: $(M, 3, 1)$ `float64` array containing per-view Rodrigues rotation vectors.
* `tvecs`: $(M, 3, 1)$ `float64` array containing per-view 3D translation vectors.
* `rms`: `float64` scalar value of the final reprojection error.

### 3. `calibration.json` (Human-Readable Metadata)
Structured JSON representation containing camera metadata, distortion vectors, and image resolution:
```json
{
  "camera_matrix": {
    "fx": 912.4532,
    "fy": 911.8904,
    "cx": 640.1246,
    "cy": 360.5421
  },
  "distortion_coefficients": [-0.0421, 0.0892, 0.0012, -0.0004, -0.0215],
  "rms_reprojection_error_px": 0.34218,
  "image_dimensions": {
    "width": 1280,
    "height": 720
  },
  "board_configuration": {
    "inner_corners": [9, 6],
    "square_size_m": 0.025
  }
}
```

You can generate `calibration.json` directly from `calibration.npz` using this snippet:
```python
import json
import numpy as np

data = np.load("outputs/calibration/calibration.npz")
K = data["camera_matrix"]
dist = data["dist_coeffs"].flatten().tolist()
meta = {
    "camera_matrix": {"fx": float(K[0,0]), "fy": float(K[1,1]), "cx": float(K[0,2]), "cy": float(K[1,2])},
    "distortion_coefficients": dist,
    "rms_reprojection_error_px": float(data["rms"])
}
with open("outputs/calibration/calibration.json", "w") as f:
    json.dump(meta, f, indent=2)
```
