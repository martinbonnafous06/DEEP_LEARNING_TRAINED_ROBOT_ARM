"""scripts/06_train_policy.py — Train a Behavioral Cloning (BC) Policy on Retargeted Human Demos.

Trains an end-to-end visual-motor policy on data/datasets/demo_libero.hdf5 using:
- Vision Encoder: ResNet-18 visual feature extractor for agentview RGB (128x128).
- Proprioception Encoder: MLP for Franka EEF position & quaternion (7D).
- Policy Head: Temporal MLP / Residual policy predicting 7D delta actions (dx, dy, dz, drx, dry, drz, gripper).
- Multi-demo train/val split (80/20) with MSE action loss and learning rate scheduling.
- Saves best checkpoint to experiments/checkpoints/best_policy.pt and plots training curves.

Usage:
    python scripts/06_train_policy.py
    python scripts/06_train_policy.py --epochs 60 --batch_size 16 --lr 3e-4
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_policy")


class LiberoHumanoidDataset(Dataset):
    """PyTorch Dataset loading demonstrations from LIBERO-format HDF5 (in-memory cached)."""

    def __init__(
        self,
        hdf5_path: Path | str,
        demo_keys: list[str],
    ) -> None:
        rgbs_list = []
        proprios_list = []
        actions_list = []

        with h5py.File(hdf5_path, "r") as f:
            for k in demo_keys:
                grp = f["data"][k]
                raw_rgb = grp["obs"]["agentview_rgb"][:]  # (N, H, W, 3)
                pos = grp["obs"]["robot0_eef_pos"][:]    # (N, 3)
                quat = grp["obs"]["robot0_eef_quat"][:]  # (N, 4)
                acts = grp["actions"][:]                 # (N, 7)

                rgb_tensor = torch.from_numpy(raw_rgb).permute(0, 3, 1, 2).float() / 255.0
                proprio_tensor = torch.from_numpy(np.concatenate([pos, quat], axis=-1)).float()
                act_tensor = torch.from_numpy(acts).float()

                rgbs_list.append(rgb_tensor)
                proprios_list.append(proprio_tensor)
                actions_list.append(act_tensor)

        self.rgbs = torch.cat(rgbs_list, dim=0)
        self.proprios = torch.cat(proprios_list, dim=0)
        self.actions = torch.cat(actions_list, dim=0)

    def __len__(self) -> int:
        return len(self.actions)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "rgb": self.rgbs[idx],
            "proprio": self.proprios[idx],
            "action": self.actions[idx],
        }


class BCPolicy(nn.Module):
    """Visuomotor Behavioral Cloning policy for robot manipulation."""

    def __init__(
        self,
        proprio_dim: int = 7,
        action_dim: int = 7,
        visual_feature_dim: int = 128,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        # Visual backbone: ResNet-18 adapted for 128x128 images
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet.fc = nn.Linear(resnet.fc.in_features, visual_feature_dim)
        self.visual_encoder = resnet

        # Proprioception encoder
        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
        )

        # Policy MLP head
        fused_dim = visual_feature_dim + 64
        self.head = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, rgb: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        v_feat = self.visual_encoder(rgb)
        p_feat = self.proprio_encoder(proprio)
        fused = torch.cat([v_feat, p_feat], dim=-1)
        pred_action = self.head(fused)
        return pred_action


def train_policy(
    dataset_path: Path,
    output_dir: Path,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 3e-4,
    device_name: str = "cuda",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device_name if torch.cuda.is_available() and device_name == "cuda" else "cpu")
    logger.info("Using device: %s (%s)", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU")

    # 1. Inspect dataset demos & train/val split (80/20)
    with h5py.File(dataset_path, "r") as f:
        all_demos = sorted(list(f["data"].keys()))

    n_demos = len(all_demos)
    n_train = max(1, int(n_demos * 0.8))
    train_demos = all_demos[:n_train]
    val_demos = all_demos[n_train:]
    logger.info("Found %d demonstrations: %d train demos, %d val demos", n_demos, len(train_demos), len(val_demos))

    train_dataset = LiberoHumanoidDataset(dataset_path, train_demos)
    val_dataset = LiberoHumanoidDataset(dataset_path, val_demos if val_demos else train_demos)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    logger.info("Loaded %d train transitions, %d val transitions", len(train_dataset), len(val_dataset))

    # 2. Instantiate policy, optimizer, scheduler
    policy = BCPolicy().to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    history = {"train_loss": [], "val_loss": [], "val_pos_loss": [], "val_rot_loss": []}
    best_val_loss = float("inf")
    best_model_path = output_dir / "best_policy.pt"

    start_time = time.time()
    logger.info("Starting Behavioral Cloning training for %d epochs...", epochs)

    for epoch in range(1, epochs + 1):
        policy.train()
        train_losses = []

        for batch in train_loader:
            rgb = batch["rgb"].to(device)
            proprio = batch["proprio"].to(device)
            target_act = batch["action"].to(device)

            pred_act = policy(rgb, proprio)

            # Separate delta pose and gripper losses for fine balance
            loss_pos = F.smooth_l1_loss(pred_act[:, :3], target_act[:, :3]) * 5.0
            loss_rot = F.smooth_l1_loss(pred_act[:, 3:6], target_act[:, 3:6]) * 2.0
            loss_grip = F.mse_loss(pred_act[:, 6], target_act[:, 6])
            total_loss = loss_pos + loss_rot + loss_grip

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
            optimizer.step()

            train_losses.append(total_loss.item())

        scheduler.step()
        train_loss = float(np.mean(train_losses))

        # Validation
        policy.eval()
        val_losses, val_pos_errs, val_rot_errs = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                rgb = batch["rgb"].to(device)
                proprio = batch["proprio"].to(device)
                target_act = batch["action"].to(device)

                pred_act = policy(rgb, proprio)
                loss_pos = F.smooth_l1_loss(pred_act[:, :3], target_act[:, :3]) * 5.0
                loss_rot = F.smooth_l1_loss(pred_act[:, 3:6], target_act[:, 3:6]) * 2.0
                loss_grip = F.mse_loss(pred_act[:, 6], target_act[:, 6])
                total_loss = loss_pos + loss_rot + loss_grip

                val_losses.append(total_loss.item())
                val_pos_errs.append(F.l1_loss(pred_act[:, :3], target_act[:, :3]).item())
                val_rot_errs.append(F.l1_loss(pred_act[:, 3:6], target_act[:, 3:6]).item())

        val_loss = float(np.mean(val_losses))
        val_pos = float(np.mean(val_pos_errs))
        val_rot = float(np.mean(val_rot_errs))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_pos_loss"].append(val_pos)
        history["val_rot_loss"].append(val_rot)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": policy.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "train_demos": train_demos,
                    "val_demos": val_demos,
                },
                best_model_path,
            )

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            logger.info(
                "Epoch [%2d/%2d] | Train Loss: %.5f | Val Loss: %.5f | Val Pos MAE: %.4f m | Best: %.5f",
                epoch,
                epochs,
                train_loss,
                val_loss,
                val_pos,
                best_val_loss,
            )

    elapsed = time.time() - start_time
    logger.info("Training complete in %.1f seconds. Best model saved to: %s", elapsed, best_model_path)

    # 3. Plot & save training curves
    plt.figure(figsize=(10, 4.5), dpi=150)
    plt.plot(range(1, epochs + 1), history["train_loss"], label="Train Loss (Total)", color="#1f77b4", linewidth=2)
    plt.plot(range(1, epochs + 1), history["val_loss"], label="Val Loss (Total)", color="#ff7f0e", linestyle="--", linewidth=2)
    plt.axhline(best_val_loss, color="green", linestyle=":", label=f"Best Val: {best_val_loss:.4f}")
    plt.title("Visuomotor BC Policy Training on Human Retargeted Dataset", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Smooth L1 / MSE Loss", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()

    plot_path = output_dir / "policy_training_curves.png"
    plt.savefig(plot_path)
    plt.close()
    logger.info("Saved training curves plot to: %s", plot_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train BC Policy on LIBERO demonstrations")
    parser.add_argument("--dataset", type=Path, default=Path("data/datasets/demo_libero.hdf5"))
    parser.add_argument("--output_dir", type=Path, default=Path("experiments/checkpoints"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if not args.dataset.exists():
        logger.error("Dataset file not found: %s", args.dataset)
        return 1

    train_policy(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device_name=args.device,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
