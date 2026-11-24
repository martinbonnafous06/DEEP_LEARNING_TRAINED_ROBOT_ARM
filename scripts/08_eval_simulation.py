"""scripts/08_eval_simulation.py — Replay Retargeted Actions in MuJoCo / Robosuite Simulation.

Addresses Criterion 2:
"High-quality policy execution in simulation and/or World Model (WM) trajectory predictions."

Loads human demonstration actions from data/datasets/demo_libero.hdf5,
steps the Panda robot controller in Robosuite / MuJoCo at 20 Hz,
records off-screen camera observations, and exports an execution video / GIF.

Usage:
    python scripts/08_eval_simulation.py
    python scripts/08_eval_simulation.py --demo_name demo_0 --env_name Lift
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import h5py
import imageio
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval_simulation")


def replay_simulation(
    dataset_path: Path,
    demo_name: str = "demo_0",
    env_name: str = "Lift",
    output_dir: Path = Path("experiments/simulation"),
    render_video: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(dataset_path, "r") as f:
        if demo_name not in f["data"]:
            available = list(f["data"].keys())
            raise KeyError(f"Demo '{demo_name}' not found in {dataset_path}. Available: {available}")
        demo_grp = f["data"][demo_name]
        actions = demo_grp["actions"][:]
        real_eef_pos = demo_grp["obs"]["robot0_eef_pos"][:]

    logger.info("Loaded demo '%s' with %d steps for simulation replay in '%s'...", demo_name, len(actions), env_name)

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.retargeting.sim_replay import execute_panda_manipulation_in_sim

    logger.info("Executing %d closed-loop object-aligned control steps on Franka Panda...", len(actions))
    sim_result = execute_panda_manipulation_in_sim(
        actions=actions,
        demo_eef_pos=real_eef_pos,
        env_name=env_name,
        render_video=render_video,
        camera_height=256,
        camera_width=256,
    )

    recorded_frames = sim_result.agentview_frames
    logger.info(
        "Simulation replay completed (%d steps) | Object Grasped: %s | Task Lift Success: %s | Max Cube Lift: %.1f cm",
        len(sim_result.ee_pos),
        sim_result.grasped,
        sim_result.task_success,
        sim_result.max_cube_lift_m * 100.0,
    )

    # Save video / GIF
    if recorded_frames:
        video_path = output_dir / f"{demo_name}_{env_name}_replay.mp4"
        gif_path = output_dir / f"{demo_name}_{env_name}_replay.gif"
        try:
            imageio.mimsave(str(video_path), recorded_frames, fps=20)
            logger.info("Saved offscreen simulation replay video: %s", video_path)
        except Exception as e:
            logger.warning("Could not save MP4 (%s), saving as GIF...", e)
            imageio.mimsave(str(gif_path), recorded_frames[::2], fps=10)
            logger.info("Saved offscreen simulation replay GIF: %s", gif_path)

    logger.info("Panda EEF Trajectory Distance Traveled in Sim: %.3f meters", sim_result.total_eef_distance_m)


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay retargeted demonstration in Robosuite/MuJoCo")
    parser.add_argument("--dataset", type=Path, default=Path("data/datasets/demo_libero.hdf5"))
    parser.add_argument("--demo_name", type=str, default="demo_0")
    parser.add_argument("--env_name", type=str, default="Lift")
    parser.add_argument("--output_dir", type=Path, default=Path("experiments/simulation"))
    args = parser.parse_args()

    if not args.dataset.exists():
        logger.error("Dataset not found: %s", args.dataset)
        return 1

    replay_simulation(
        dataset_path=args.dataset,
        demo_name=args.demo_name,
        env_name=args.env_name,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
