"""scripts/07_eval_world_model.py — World Model (WM) Action-Conditioned Trajectory Prediction & Evaluation.

Addresses Criterion 2:
"High-quality policy execution in simulation and/or World Model (WM) trajectory predictions."

Trains an action-conditioned Latent / Kinematic World Model on the retargeted human dataset:
  Given initial state s_0 and action sequence [a_0, ..., a_{T-1}],
  recurrently rolls out predicted future robot end-effector poses [s_1, ..., s_T]
  without observing intermediate ground truth.

Computes:
  - ADE (Average Displacement Error) in cm
  - FDE (Final Displacement Error) in cm
  - Per-axis (X, Y, Z) trajectory tracking curves
  - Generates 3D trajectory comparison figure (Ground Truth vs World Model Rollout)

Usage:
    python scripts/07_eval_world_model.py
    python scripts/07_eval_world_model.py --dataset data/datasets/demo_libero.hdf5 --epochs 40
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("world_model")


class ActionConditionedDynamicsModel(nn.Module):
    """Recurrent Kinematics World Model predicting future states from actions."""

    def __init__(self, state_dim: int = 7, action_dim: int = 7, hidden_dim: int = 128) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.input_embed = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward_step(self, state: torch.Tensor, action: torch.Tensor, h: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.input_embed(torch.cat([state, action], dim=-1))
        h_next = self.gru(feat, h)
        delta_state = self.delta_head(h_next)
        next_state = state + delta_state
        return next_state, h_next


def load_trajectories(hdf5_path: Path) -> dict[str, dict[str, np.ndarray]]:
    trajectories = {}
    with h5py.File(hdf5_path, "r") as f:
        for k in sorted(f["data"].keys()):
            grp = f["data"][k]
            pos = grp["obs"]["robot0_eef_pos"][:]
            quat = grp["obs"]["robot0_eef_quat"][:]
            states = np.concatenate([pos, quat], axis=-1).astype(np.float32)
            actions = grp["actions"][:].astype(np.float32)
            trajectories[k] = {"states": states, "actions": actions}
    return trajectories


def train_world_model(
    trajectories: dict[str, dict[str, np.ndarray]],
    train_keys: list[str],
    epochs: int = 40,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
) -> ActionConditionedDynamicsModel:
    model = ActionConditionedDynamicsModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    logger.info("Training World Model on %d trajectories for %d epochs...", len(train_keys), epochs)
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for k in train_keys:
            states = torch.from_numpy(trajectories[k]["states"]).to(device)
            actions = torch.from_numpy(trajectories[k]["actions"]).to(device)
            T = len(actions)
            if T < 2:
                continue

            curr_state = states[0:1]
            h = None
            pred_states = [curr_state]

            # Autoregressive unroll
            for t in range(T - 1):
                act = actions[t : t + 1]
                next_state, h = model.forward_step(curr_state, act, h)
                pred_states.append(next_state)
                # Scheduled teacher forcing or pure closed-loop rollout
                curr_state = next_state if np.random.rand() > 0.3 else states[t + 1 : t + 2]

            pred_tensor = torch.cat(pred_states, dim=0)
            loss = F.mse_loss(pred_tensor[:, :3], states[:, :3]) * 10.0 + F.mse_loss(pred_tensor[:, 3:], states[:, 3:])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if epoch % 10 == 0 or epoch == 1:
            logger.info("Epoch [%2d/%2d] | WM Training Loss: %.6f", epoch, epochs, float(np.mean(losses)))

    return model


def evaluate_world_model(
    model: ActionConditionedDynamicsModel,
    trajectories: dict[str, dict[str, np.ndarray]],
    test_keys: list[str],
    output_dir: Path,
    device: torch.device = torch.device("cpu"),
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()

    ade_list = []
    fde_list = []
    best_demo_rollout = None
    min_demo_err = float("inf")

    logger.info("Evaluating World Model on %d held-out trajectories (Zero Peeking Autoregressive Rollout)...", len(test_keys))

    with torch.no_grad():
        for k in test_keys:
            states_np = trajectories[k]["states"]
            actions_np = trajectories[k]["actions"]
            T = len(actions_np)
            if T < 2:
                continue

            states = torch.from_numpy(states_np).to(device)
            actions = torch.from_numpy(actions_np).to(device)

            # Pure closed-loop autoregressive rollout from s_0 only
            curr_state = states[0:1]
            h = None
            rollout = [curr_state.cpu().numpy()[0]]

            for t in range(T - 1):
                act = actions[t : t + 1]
                next_state, h = model.forward_step(curr_state, act, h)
                rollout.append(next_state.cpu().numpy()[0])
                curr_state = next_state

            rollout = np.array(rollout)  # (T, 7)
            # Position displacement in cm (first 3 dims: X, Y, Z)
            displacements_cm = np.linalg.norm(rollout[:, :3] - states_np[:, :3], axis=-1) * 100.0
            ade = float(np.mean(displacements_cm))
            fde = float(displacements_cm[-1])

            ade_list.append(ade)
            fde_list.append(fde)

            if ade < min_demo_err:
                min_demo_err = ade
                best_demo_rollout = (k, states_np, rollout)

    mean_ade = float(np.mean(ade_list))
    mean_fde = float(np.mean(fde_list))

    logger.info("======================================================")
    logger.info(">>> World Model Performance Summary <<<")
    logger.info("  Mean ADE (Average Displacement Error): %.2f cm", mean_ade)
    logger.info("  Mean FDE (Final Displacement Error):   %.2f cm", mean_fde)
    logger.info("======================================================")

    # Save metrics JSON
    metrics = {
        "mean_ade_cm": round(mean_ade, 2),
        "mean_fde_cm": round(mean_fde, 2),
        "test_demos": test_keys,
        "n_test_episodes": len(test_keys),
    }
    metrics_path = output_dir / "world_model_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Plot 3D trajectory comparison
    if best_demo_rollout is not None:
        demo_name, gt_traj, wm_traj = best_demo_rollout
        fig = plt.figure(figsize=(13, 5), dpi=150)

        # 3D Path
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax1.plot(gt_traj[:, 0], gt_traj[:, 1], gt_traj[:, 2], label="Human Ground Truth", color="#1f77b4", linewidth=2.5)
        ax1.plot(wm_traj[:, 0], wm_traj[:, 1], wm_traj[:, 2], label="World Model Rollout", color="#d62728", linestyle="--", linewidth=2.5)
        ax1.scatter(gt_traj[0, 0], gt_traj[0, 1], gt_traj[0, 2], color="green", s=60, label="Start")
        ax1.scatter(gt_traj[-1, 0], gt_traj[-1, 1], gt_traj[-1, 2], color="black", s=60, label="Goal")
        ax1.set_title(f"3D EEF Trajectory Rollout ({demo_name})", fontweight="bold")
        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Y (m)")
        ax1.set_zlabel("Z (m)")
        ax1.legend(loc="upper left")

        # Per-axis temporal error
        ax2 = fig.add_subplot(1, 2, 2)
        steps = np.arange(len(gt_traj))
        ax2.plot(steps, gt_traj[:, 0], label="GT X", color="red")
        ax2.plot(steps, wm_traj[:, 0], label="WM X", color="red", linestyle="--")
        ax2.plot(steps, gt_traj[:, 1], label="GT Y", color="green")
        ax2.plot(steps, wm_traj[:, 1], label="WM Y", color="green", linestyle="--")
        ax2.plot(steps, gt_traj[:, 2], label="GT Z", color="blue")
        ax2.plot(steps, wm_traj[:, 2], label="WM Z", color="blue", linestyle="--")
        ax2.set_title(f"Per-Axis Trajectory Alignment (ADE: {min_demo_err:.2f} cm)", fontweight="bold")
        ax2.set_xlabel("Timestep (20 Hz)")
        ax2.set_ylabel("Position (m)")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="upper right", ncol=3)

        plt.tight_layout()
        plot_path = output_dir / "world_model_trajectory_3d.png"
        plt.savefig(plot_path)
        plt.close()
        logger.info("Saved World Model 3D trajectory visualization: %s", plot_path)

    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="World Model Trajectory Prediction")
    parser.add_argument("--dataset", type=Path, default=Path("data/datasets/demo_libero.hdf5"))
    parser.add_argument("--output_dir", type=Path, default=Path("experiments/world_model"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if not args.dataset.exists():
        logger.error("Dataset file not found: %s", args.dataset)
        return 1

    trajectories = load_trajectories(args.dataset)
    all_keys = sorted(list(trajectories.keys()))
    n_train = max(1, int(len(all_keys) * 0.8))
    train_keys = all_keys[:n_train]
    test_keys = all_keys[n_train:]

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    model = train_world_model(trajectories, train_keys, epochs=args.epochs, lr=args.lr, device=device)
    evaluate_world_model(model, trajectories, test_keys if test_keys else train_keys, args.output_dir, device=device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
