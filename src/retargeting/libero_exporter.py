"""src.retargeting.libero_exporter — Export retargeted trajectories into benchmark-compliant LIBERO HDF5 datasets."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import numpy.typing as npt

from ..utils.io import load_rgb_frames

logger = logging.getLogger("libero_exporter")


def try_robosuite_replay(
    actions: npt.NDArray[np.float32],
    env_name: str = "Lift",
) -> Optional[dict[str, npt.NDArray[np.float32]]]:
    """Replay trajectory actions in a simulated Robosuite environment to capture MuJoCo states.

    Records pre-action observations aligned with each action index.

    Args:
        actions: Array of shape (N, 7) float32 delta actions.
        env_name: Robosuite environment name (e.g. 'Lift', 'PickPlaceCan').

    Returns:
        Dict with keys 'joint_states', 'ee_pos', 'ee_ori', or None if Robosuite is unavailable.
    """
    import importlib.util
    if importlib.util.find_spec("robosuite") is None:
        logger.info("robosuite is not installed. Skipping simulation replay.")
        return None

    try:
        import robosuite as suite
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
        joint_states: list[npt.NDArray[np.float32]] = []
        ee_positions: list[npt.NDArray[np.float32]] = []
        ee_oris: list[npt.NDArray[np.float32]] = []

        for i, action in enumerate(actions):
            # Record observation before stepping action i
            joint_states.append(obs.get("robot0_joint_pos", np.zeros(7, dtype=np.float32)))
            ee_positions.append(obs.get("robot0_eef_pos", np.zeros(3, dtype=np.float32)))
            ee_oris.append(obs.get("robot0_eef_quat", np.zeros(4, dtype=np.float32)))

            obs, _reward, done, _info = env.step(action)
            if done and i < n - 1:
                logger.info("Environment indicated episode done at step %d / %d.", i, n)
                for _ in range(i + 1, n):
                    joint_states.append(joint_states[-1])
                    ee_positions.append(ee_positions[-1])
                    ee_oris.append(ee_oris[-1])
                break

        env.close()
        return {
            "joint_states": np.array(joint_states, dtype=np.float32),
            "ee_pos": np.array(ee_positions, dtype=np.float32),
            "ee_ori": np.array(ee_oris, dtype=np.float32),
        }
    except Exception as exc:
        logger.warning("Robosuite replay failed (%s); continuing without simulation observations.", exc)
        return None


def write_hdf5_dataset(
    output_path: Path | str,
    demo_name: str,
    actions: npt.NDArray[np.float32],
    ee_pos: npt.NDArray[np.float32],
    ee_quat: npt.NDArray[np.float32],
    agentview_rgb: npt.NDArray[np.uint8],
    wrist_rgb: npt.NDArray[np.uint8],
    env_name: str = "LIBERO_OBJECT",
    env_args: Optional[dict] = None,
    sim_obs: Optional[dict[str, npt.NDArray[np.float32]]] = None,
) -> Path:
    """Serialize trajectory data into a standard LIBERO HDF5 demonstration file.

    Args:
        output_path: Destination .hdf5 path.
        demo_name: Demo group identifier (e.g. 'demo_0').
        actions: (N, 7) float32 actions.
        ee_pos: (N, 3) float32 end-effector positions.
        ee_quat: (N, 4) float32 end-effector quaternions [x, y, z, w].
        agentview_rgb: (N, H, W, 3) uint8 image frames.
        wrist_rgb: (N, H, W, 3) uint8 image frames.
        env_name: Name of target environment.
        env_args: Optional dict serialised into dataset attributes.
        sim_obs: Optional dictionary containing MuJoCo replay states.

    Returns:
        Path to output file.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    n = len(actions)
    if len(ee_pos) != n or len(ee_quat) != n:
        raise ValueError(
            f"Trajectory length mismatch: actions={n}, ee_pos={len(ee_pos)}, ee_quat={len(ee_quat)}"
        )

    rewards = np.zeros(n, dtype=np.float32)
    dones = np.zeros(n, dtype=np.float32)
    dones[-1] = 1.0

    if env_args is None:
        env_args = {"env_name": env_name, "robots": "Panda", "episode_length": n}

    mode = "a" if out_p.is_file() else "w"
    with h5py.File(out_p, mode) as f:
        if "data" not in f:
            data_grp = f.create_group("data")
            data_grp.attrs["total"] = 0
            data_grp.attrs["env_name"] = env_name
            data_grp.attrs["env_args"] = json.dumps(env_args)
        else:
            data_grp = f["data"]

        if demo_name in data_grp:
            del data_grp[demo_name]
        else:
            data_grp.attrs["total"] = int(data_grp.attrs.get("total", 0)) + 1

        demo_grp = data_grp.create_group(demo_name)
        demo_grp.attrs["num_samples"] = n

        demo_grp.create_dataset("actions", data=actions, compression="gzip")
        demo_grp.create_dataset("rewards", data=rewards, compression="gzip")
        demo_grp.create_dataset("dones", data=dones, compression="gzip")

        obs_grp = demo_grp.create_group("obs")
        obs_grp.create_dataset("agentview_rgb", data=agentview_rgb, dtype=np.uint8, compression="gzip")
        obs_grp.create_dataset("robot0_eye_in_hand_rgb", data=wrist_rgb, dtype=np.uint8, compression="gzip")
        obs_grp.create_dataset("robot0_eef_pos", data=ee_pos, compression="gzip")
        obs_grp.create_dataset("robot0_eef_quat", data=ee_quat, compression="gzip")

        if sim_obs is not None:
            if "joint_states" in sim_obs:
                obs_grp.create_dataset("robot0_joint_pos", data=sim_obs["joint_states"], compression="gzip")
            if "ee_pos" in sim_obs:
                obs_grp.create_dataset("robot0_eef_pos_sim", data=sim_obs["ee_pos"], compression="gzip")
            if "ee_ori" in sim_obs:
                obs_grp.create_dataset("robot0_eef_quat_sim", data=sim_obs["ee_ori"], compression="gzip")

    logger.info("LIBERO HDF5 file successfully written -> %s (N=%d samples)", out_p, n)
    return out_p


def print_hdf5_structure(hdf5_path: Path | str) -> None:
    """Print human-readable hierarchy and shapes of all datasets in an HDF5 file."""
    p = Path(hdf5_path)
    print("\n" + "=" * 60)
    print(f"  HDF5 Dataset Inspection: {p.name}")
    print("=" * 60)

    with h5py.File(p, "r") as f:
        if "data" in f:
            print("\n[/data] Attributes:")
            for k, v in f["data"].attrs.items():
                print(f"    {k}: {v}")

        print("\nDatasets:")
        def _visitor(name: str, obj: h5py.HLObject) -> None:
            if isinstance(obj, h5py.Dataset):
                print(f"    /{name:<40} shape={str(obj.shape):<16} dtype={obj.dtype}")

        f.visititems(_visitor)
    print("=" * 60 + "\n")


def export_libero_dataset(
    retargeted_npz: Path | str,
    output_hdf5: Path | str,
    agentview_dir: Optional[Path | str] = None,
    wrist_dir: Optional[Path | str] = None,
    demo_name: str = "demo_0",
    env_name: str = "LIBERO_OBJECT",
    frame_height: int = 84,
    frame_width: int = 84,
) -> Path:
    """Execute complete Module 5 LIBERO HDF5 dataset export.

    Args:
        retargeted_npz: Path to retargeted.npz.
        output_hdf5: Destination path for .hdf5 file.
        agentview_dir: Directory containing external agentview camera frames.
        wrist_dir: Directory containing eye-in-hand wrist camera frames.
        demo_name: Group name for this trajectory.
        env_name: Target LIBERO / Robosuite environment name.
        frame_height: Target frame height (default 84 px).
        frame_width: Target frame width (default 84 px).

    Returns:
        Path to output HDF5 file.
    """
    retargeted_path = Path(retargeted_npz)
    if not retargeted_path.is_file():
        raise FileNotFoundError(f"Retargeted file not found: {retargeted_path}")

    with np.load(retargeted_path) as data:
        actions = data["actions"].astype(np.float32)
        ee_pos = data["ee_pos"].astype(np.float32)
        ee_quat = data["ee_quat"].astype(np.float32)

        n_actions = len(actions)

        # Seamlessly handle length N+1 from legacy full-state trajectories
        if len(ee_pos) == n_actions + 1:
            ee_pos = ee_pos[:n_actions]
        if len(ee_quat) == n_actions + 1:
            ee_quat = ee_quat[:n_actions]

        # Ensure 7-DOF action width
        if actions.ndim == 2 and actions.shape[1] < 7:
            if "gripper" in data:
                grip = data["gripper"].astype(np.float32)[:n_actions].reshape(-1, 1)
                actions = np.concatenate([actions, grip], axis=1)
            if actions.shape[1] < 7:
                pad = np.zeros((n_actions, 7 - actions.shape[1]), dtype=np.float32)
                actions = np.concatenate([actions, pad], axis=1)

    # Optional Robosuite replay
    sim_obs = try_robosuite_replay(actions, env_name=env_name)

    # Load RGB frames
    agentview_rgb = load_rgb_frames(agentview_dir, n_actions, height=frame_height, width=frame_width)
    wrist_rgb = load_rgb_frames(wrist_dir, n_actions, height=frame_height, width=frame_width)

    out_hdf5_path = write_hdf5_dataset(
        output_path=output_hdf5,
        demo_name=demo_name,
        actions=actions,
        ee_pos=ee_pos,
        ee_quat=ee_quat,
        agentview_rgb=agentview_rgb,
        wrist_rgb=wrist_rgb,
        env_name=env_name,
        sim_obs=sim_obs,
    )

    print_hdf5_structure(out_hdf5_path)
    return out_hdf5_path
