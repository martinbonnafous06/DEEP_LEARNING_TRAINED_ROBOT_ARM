"""dataset_exporter.py — Module 5: Video-to-LIBERO Pipeline.

Converts a retargeted trajectory (``retargeted.npz``) produced by Module 4
into a LIBERO-compatible HDF5 dataset.

Workflow
--------
1. Load ``retargeted.npz`` -> actions, ee_pos, ee_quat, gripper arrays.
2. (Optional) Replay the trajectory in a Robosuite/LIBERO environment to
   collect ground-truth MuJoCo observations (joint states, ee_pos, ee_ori).
   Guarded with ``try/except`` so the step is skipped gracefully when
   Robosuite is not installed.
3. Write an HDF5 file following LIBERO conventions::

       /data  (attrs: total, env_name, env_args)
       /data/demo_X/actions                      (N, 7)   float32
       /data/demo_X/obs/agentview_rgb             (N,H,W,3) uint8
       /data/demo_X/obs/robot0_eye_in_hand_rgb    (N,H,W,3) uint8
       /data/demo_X/obs/robot0_eef_pos            (N, 3)   float32
       /data/demo_X/obs/robot0_eef_quat           (N, 4)   float32
       /data/demo_X/rewards                       (N,)     float32
       /data/demo_X/dones                         (N,)     float32

4. Print an HDF5 structure summary via ``h5py.File.visititems()``.

Usage
-----
    python dataset_exporter.py \\
        --retargeted_npz retargeted.npz \\
        --output_hdf5    libero_dataset.hdf5 \\
        --agentview_dir  frames/agentview \\
        --wrist_dir      frames/wrist \\
        --demo_name      demo_0 \\
        --env_name       LIBERO_OBJECT \\
        --frame_height   84 \\
        --frame_width    84
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sorted_image_paths(directory: Path) -> list[Path]:
    """Return image files in *directory* sorted by natural numeric order.

    Args:
        directory: Path to a folder containing frame images.

    Returns:
        A list of ``Path`` objects sorted in natural numeric order.
    """
    import re
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
    files = [p for p in directory.iterdir() if p.suffix.lower() in extensions]

    def _key(p: Path) -> list[int | str]:
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", p.name)]

    return sorted(files, key=_key)


def _load_frames(
    directory: Optional[Path],
    n_frames: int,
    height: int,
    width: int,
) -> np.ndarray:
    """Load RGB frames from *directory*, resizing to (height, width).

    If *directory* is ``None`` or does not exist, returns a zero array of
    shape ``(n_frames, height, width, 3)`` with dtype ``uint8``.

    When the number of images found differs from *n_frames* the array is
    either truncated (too many) or the last frame is repeated to pad (too few).

    Args:
        directory: Path to the folder containing frame images, or ``None``.
        n_frames: Expected number of frames (= trajectory length N).
        height: Target frame height in pixels.
        width: Target frame width in pixels.

    Returns:
        ``numpy.ndarray`` of shape ``(n_frames, height, width, 3)``,
        dtype ``uint8``.
    """
    blank = np.zeros((n_frames, height, width, 3), dtype=np.uint8)

    if directory is None or not directory.is_dir():
        return blank

    paths = _sorted_image_paths(directory)
    if not paths:
        print(
            f"[dataset_exporter] WARNING: No images found in {directory}; "
            "using zero frames."
        )
        return blank

    frames: list[np.ndarray] = []
    for p in paths[:n_frames]:
        img = Image.open(p).convert("RGB").resize((width, height), Image.BILINEAR)
        frames.append(np.array(img, dtype=np.uint8))

    if len(frames) < n_frames:
        print(
            f"[dataset_exporter] WARNING: Only {len(frames)} images found in "
            f"{directory} (expected {n_frames}); padding with last frame."
        )
        pad_frame = frames[-1]
        while len(frames) < n_frames:
            frames.append(pad_frame)

    return np.stack(frames, axis=0)  # (N, H, W, 3)


# ---------------------------------------------------------------------------
# Robosuite replay (optional)
# ---------------------------------------------------------------------------

def _try_robosuite_replay(
    actions: np.ndarray,
    env_name: str,
) -> Optional[dict[str, np.ndarray]]:
    """Replay *actions* in a Robosuite environment and collect observations.

    Attempts to import ``robosuite`` at runtime.  If the import fails the
    function prints a warning and returns ``None`` so the rest of the pipeline
    continues without simulation data.

    Args:
        actions: Array of shape ``(N, 7)`` -- the end-effector actions to
            replay in the environment.
        env_name: Robosuite environment name (e.g. ``"Lift"``).

    Returns:
        A dict with keys ``"joint_states"``, ``"ee_pos"``, ``"ee_ori"`` each
        mapping to a ``numpy.ndarray`` of shape ``(N, ...)``, or ``None`` if
        Robosuite is unavailable or replay fails.
    """
    import importlib.util
    if importlib.util.find_spec("robosuite") is None:
        print(
            "[dataset_exporter] INFO: robosuite not installed -- "
            "skipping environment replay."
        )
        return None

    try:
        import robosuite as suite  # type: ignore
        env = suite.make(
            env_name=env_name,
            robots="Panda",
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            control_freq=20,
        )
        obs = env.reset()

        n = len(actions)
        joint_states: list[np.ndarray] = []
        ee_positions: list[np.ndarray] = []
        ee_oris: list[np.ndarray] = []

        for i, action in enumerate(actions):
            # Record observation before stepping action i
            joint_states.append(
                obs.get("robot0_joint_pos", np.zeros(7, dtype=np.float32))
            )
            ee_positions.append(
                obs.get("robot0_eef_pos", np.zeros(3, dtype=np.float32))
            )
            ee_oris.append(
                obs.get("robot0_eef_quat", np.zeros(4, dtype=np.float32))
            )
            obs, _reward, done, _info = env.step(action)
            if done and i < n - 1:
                print(f"[dataset_exporter] Environment terminated at step {i} / {n}.")
                for _ in range(i + 1, n):
                    joint_states.append(joint_states[-1])
                    ee_positions.append(ee_positions[-1])
                    ee_oris.append(ee_oris[-1])
                break

        env.close()
        return {
            "joint_states": np.array(joint_states, dtype=np.float32),
            "ee_pos":       np.array(ee_positions, dtype=np.float32),
            "ee_ori":       np.array(ee_oris, dtype=np.float32),
        }

    except Exception as exc:  # pylint: disable=broad-except
        print(
            f"[dataset_exporter] WARNING: Robosuite replay failed ({exc}); "
            "skipping."
        )
        return None


# ---------------------------------------------------------------------------
# HDF5 writer
# ---------------------------------------------------------------------------

def write_libero_hdf5(
    output_path: Path,
    demo_name: str,
    actions: np.ndarray,
    ee_pos: np.ndarray,
    ee_quat: np.ndarray,
    agentview_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
    env_name: str,
    env_args: Optional[dict] = None,
    sim_obs: Optional[dict[str, np.ndarray]] = None,
) -> None:
    """Write a LIBERO-compatible HDF5 dataset file.

    Creates (or overwrites) *output_path* with the following structure::

        /data  (attrs: total=int, env_name=str, env_args=JSON str)
        /data/<demo_name>/
            actions                      float32 (N, 7)
            rewards                      float32 (N,)
            dones                        float32 (N,)
            obs/
                agentview_rgb            uint8   (N, H, W, 3)
                robot0_eye_in_hand_rgb   uint8   (N, H, W, 3)
                robot0_eef_pos           float32 (N, 3)
                robot0_eef_quat          float32 (N, 4)
                [robot0_joint_pos]       float32 (N, J)  when sim_obs given
                [robot0_eef_pos_sim]     float32 (N, 3)  when sim_obs given
                [robot0_eef_quat_sim]    float32 (N, 4)  when sim_obs given

    Args:
        output_path: Destination ``.hdf5`` file path.
        demo_name: Dataset key for this trajectory (e.g. ``"demo_0"``).
        actions: End-effector delta actions shape ``(N, 7)`` float32.
        ee_pos: Retargeted ee positions ``(N, 3)`` float32.
        ee_quat: Retargeted ee quaternions ``(N, 4)`` float32.
        agentview_rgb: Agentview frames ``(N, H, W, 3)`` uint8.
        wrist_rgb: Wrist-camera frames ``(N, H, W, 3)`` uint8.
        env_name: LIBERO environment name stored in dataset attrs.
        env_args: Optional dict serialised as JSON into dataset attrs.
        sim_obs: Optional dict from :func:`_try_robosuite_replay` containing
            ``"joint_states"``, ``"ee_pos"``, ``"ee_ori"`` arrays.

    Raises:
        ValueError: If *actions*, *ee_pos*, or *ee_quat* have inconsistent
            leading dimensions.
    """
    if env_args is None:
        env_args = {}

    n = len(actions)
    if len(ee_pos) != n or len(ee_quat) != n:
        raise ValueError(
            f"Inconsistent trajectory lengths: actions={n}, "
            f"ee_pos={len(ee_pos)}, ee_quat={len(ee_quat)}"
        )

    rewards = np.zeros(n, dtype=np.float32)
    dones = np.zeros(n, dtype=np.float32)
    dones[-1] = 1.0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        # ---- root /data group attrs ----
        data_grp = f.create_group("data")
        data_grp.attrs["total"] = 1
        data_grp.attrs["env_name"] = env_name
        data_grp.attrs["env_args"] = json.dumps(env_args)

        # ---- per-demo group ----
        demo_grp = data_grp.create_group(demo_name)
        demo_grp.attrs["num_samples"] = n

        demo_grp.create_dataset(
            "actions",
            data=actions.astype(np.float32),
            compression="gzip",
        )
        demo_grp.create_dataset(
            "rewards",
            data=rewards,
            compression="gzip",
        )
        demo_grp.create_dataset(
            "dones",
            data=dones,
            compression="gzip",
        )

        # ---- observations ----
        obs_grp = demo_grp.create_group("obs")

        obs_grp.create_dataset(
            "agentview_rgb",
            data=agentview_rgb,
            dtype=np.uint8,
            compression="gzip",
        )
        obs_grp.create_dataset(
            "robot0_eye_in_hand_rgb",
            data=wrist_rgb,
            dtype=np.uint8,
            compression="gzip",
        )
        obs_grp.create_dataset(
            "robot0_eef_pos",
            data=ee_pos.astype(np.float32),
            compression="gzip",
        )
        obs_grp.create_dataset(
            "robot0_eef_quat",
            data=ee_quat.astype(np.float32),
            compression="gzip",
        )

        # ---- optional simulation observations ----
        if sim_obs is not None:
            if "joint_states" in sim_obs:
                obs_grp.create_dataset(
                    "robot0_joint_pos",
                    data=sim_obs["joint_states"].astype(np.float32),
                    compression="gzip",
                )
            if "ee_pos" in sim_obs:
                obs_grp.create_dataset(
                    "robot0_eef_pos_sim",
                    data=sim_obs["ee_pos"].astype(np.float32),
                    compression="gzip",
                )
            if "ee_ori" in sim_obs:
                obs_grp.create_dataset(
                    "robot0_eef_quat_sim",
                    data=sim_obs["ee_ori"].astype(np.float32),
                    compression="gzip",
                )

    print(f"[dataset_exporter] HDF5 written -> {output_path}")


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_hdf5_summary(hdf5_path: Path) -> None:
    """Print a human-readable summary of an HDF5 file structure.

    Uses ``h5py.File.visititems()`` to enumerate every group and dataset,
    printing their shapes and dtypes.

    Args:
        hdf5_path: Path to an existing HDF5 file.
    """
    print("\n" + "=" * 60)
    print(f"  HDF5 Structure: {hdf5_path.name}")
    print("=" * 60)

    with h5py.File(hdf5_path, "r") as f:
        print("\n[/data] attrs:")
        for k, v in f["data"].attrs.items():
            print(f"    {k}: {v}")

        print("\nDatasets:")

        def _visitor(name: str, obj: h5py.HLObject) -> None:
            if isinstance(obj, h5py.Dataset):
                print(f"    /{name}  shape={obj.shape}  dtype={obj.dtype}")

        f.visititems(_visitor)

    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Core pipeline function
# ---------------------------------------------------------------------------

def export_dataset(
    retargeted_npz: Path,
    output_hdf5: Path,
    agentview_dir: Optional[Path],
    wrist_dir: Optional[Path],
    demo_name: str,
    env_name: str,
    frame_height: int,
    frame_width: int,
) -> None:
    """Run the full export pipeline from retargeted NPZ to LIBERO HDF5.

    Steps:

    1. Load ``retargeted.npz`` and assemble the ``(N, 7)`` action array.
    2. Attempt Robosuite replay to enrich observations (skipped gracefully
       when Robosuite is absent).
    3. Load agentview and wrist RGB frames (zero arrays if dirs not given).
    4. Write the LIBERO-compatible HDF5 file via :func:`write_libero_hdf5`.
    5. Print an HDF5 structure summary via :func:`print_hdf5_summary`.

    Args:
        retargeted_npz: Path to ``retargeted.npz`` (output of Module 4).
        output_hdf5: Destination path for the output HDF5 file.
        agentview_dir: Optional directory of agentview PNG/JPG frames.
        wrist_dir: Optional directory of wrist-camera PNG/JPG frames.
        demo_name: HDF5 group name for this trajectory (e.g. ``"demo_0"``).
        env_name: LIBERO / Robosuite environment name.
        frame_height: Pixel height to resize RGB frames to.
        frame_width: Pixel width to resize RGB frames to.

    Raises:
        FileNotFoundError: If *retargeted_npz* does not exist.
        KeyError: If required keys are missing from *retargeted_npz*.
    """
    if not retargeted_npz.is_file():
        raise FileNotFoundError(f"retargeted.npz not found: {retargeted_npz}")

    # ------------------------------------------------------------------
    # 1. Load retargeted trajectory
    # ------------------------------------------------------------------
    print(f"[dataset_exporter] Loading {retargeted_npz} ...")
    with np.load(retargeted_npz, allow_pickle=True) as data:
        required_keys = {"actions", "ee_pos", "ee_quat"}
        missing = required_keys - set(data.files)
        if missing:
            raise KeyError(
                f"retargeted.npz is missing required keys: {missing}. "
                f"Available keys: {list(data.files)}"
            )

        actions: np.ndarray = data["actions"].astype(np.float32)   # (N, 7)
        ee_pos:  np.ndarray = data["ee_pos"].astype(np.float32)    # (N, 3)
        ee_quat: np.ndarray = data["ee_quat"].astype(np.float32)   # (N, 4)

        n = len(actions)

        # Handle N+1 poses (pre-action observations)
        if len(ee_pos) == n + 1:
            ee_pos = ee_pos[:n]
        if len(ee_quat) == n + 1:
            ee_quat = ee_quat[:n]

        # Append gripper column if the action array is narrower than 7
        if "gripper" in data.files and actions.ndim == 2 and actions.shape[1] < 7:
            gripper: np.ndarray = data["gripper"].astype(np.float32)[:n].reshape(-1, 1)
            actions = np.concatenate([actions, gripper], axis=1)

    # Zero-pad to exactly 7 columns if still narrow
    if actions.ndim == 2 and actions.shape[1] != 7:
        print(
            f"[dataset_exporter] WARNING: action width is {actions.shape[1]}, "
            "expected 7. Zero-padding to 7."
        )
        pad = np.zeros((n, 7 - actions.shape[1]), dtype=np.float32)
        actions = np.concatenate([actions, pad], axis=1)

    print(
        f"[dataset_exporter] Trajectory length N={n}, "
        f"action shape={actions.shape}"
    )

    # ------------------------------------------------------------------
    # 2. Optional Robosuite replay
    # ------------------------------------------------------------------
    sim_obs: Optional[dict[str, np.ndarray]] = _try_robosuite_replay(
        actions, env_name
    )

    # ------------------------------------------------------------------
    # 3. Load RGB frames
    # ------------------------------------------------------------------
    print("[dataset_exporter] Loading agentview frames ...")
    agentview_rgb = _load_frames(agentview_dir, n, frame_height, frame_width)

    print("[dataset_exporter] Loading wrist-camera frames ...")
    wrist_rgb = _load_frames(wrist_dir, n, frame_height, frame_width)

    # ------------------------------------------------------------------
    # 4. Write HDF5
    # ------------------------------------------------------------------
    env_args: dict = {
        "env_name": env_name,
        "robots": "Panda",
        "episode_length": n,
    }

    write_libero_hdf5(
        output_path=output_hdf5,
        demo_name=demo_name,
        actions=actions,
        ee_pos=ee_pos,
        ee_quat=ee_quat,
        agentview_rgb=agentview_rgb,
        wrist_rgb=wrist_rgb,
        env_name=env_name,
        env_args=env_args,
        sim_obs=sim_obs,
    )

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    print_hdf5_summary(output_hdf5)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser.

    Returns:
        A configured :class:`argparse.ArgumentParser` instance.
    """
    parser = argparse.ArgumentParser(
        prog="dataset_exporter",
        description=(
            "Module 5 - Video-to-LIBERO Pipeline: "
            "Export a retargeted trajectory to a LIBERO-compatible HDF5 dataset."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--retargeted_npz",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to retargeted.npz (output of Module 4).",
    )
    parser.add_argument(
        "--output_hdf5",
        type=Path,
        required=True,
        metavar="PATH",
        help="Destination path for the output HDF5 file.",
    )
    parser.add_argument(
        "--agentview_dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory of agentview RGB frame images. "
             "If omitted, zero arrays are used.",
    )
    parser.add_argument(
        "--wrist_dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory of wrist-camera RGB frame images. "
             "If omitted, zero arrays are used.",
    )
    parser.add_argument(
        "--demo_name",
        type=str,
        default="demo_0",
        metavar="NAME",
        help="HDF5 group name for this trajectory.",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="LIBERO_OBJECT",
        metavar="ENV",
        help="LIBERO / Robosuite environment name.",
    )
    parser.add_argument(
        "--frame_height",
        type=int,
        default=84,
        metavar="H",
        help="Height (px) to resize RGB frames to.",
    )
    parser.add_argument(
        "--frame_width",
        type=int,
        default=84,
        metavar="W",
        help="Width (px) to resize RGB frames to.",
    )
    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _parser = _build_parser()
    _args = _parser.parse_args()

    try:
        export_dataset(
            retargeted_npz=_args.retargeted_npz,
            output_hdf5=_args.output_hdf5,
            agentview_dir=_args.agentview_dir,
            wrist_dir=_args.wrist_dir,
            demo_name=_args.demo_name,
            env_name=_args.env_name,
            frame_height=_args.frame_height,
            frame_width=_args.frame_width,
        )
    except (FileNotFoundError, KeyError, ValueError) as _exc:
        print(f"[dataset_exporter] ERROR: {_exc}", file=sys.stderr)
        sys.exit(1)
